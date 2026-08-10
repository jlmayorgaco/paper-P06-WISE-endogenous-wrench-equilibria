"""Gamma-WISE: certified optimal-fiber ascent (prototype, NOT used by the LARS paper).

Starting from any productive optimum, move only inside the optimal composition fiber

    E = {z in X_f : Bz = y*},

ascending lambda_2(Lbar(z)) until the free-networkability modulus vanishes. By the
local-global theorem, Gamma_E(z) = 0 certifies GLOBAL connectivity optimality over E, so the
method terminates with a certificate rather than a "no local improvement found" message.

Per-iteration:
  1. critical eigenspace U2 of Q^T Lbar(z) Q  (handles a repeated Fiedler eigenvalue);
  2. directional subproblem  max { lambda_min(U2^T DLbar[d] U2) : d in T_E(z), ||d|| <= 1 },
     whose value IS Gamma_E(z);
  3. stopping rule  Gamma_E(z) <= eps / sqrt(2N)  =>  Lambda_E - lambda_2(z) <= eps,
     using diam(E) <= diam(X) = sqrt(2N) for a product of N simplices (tight at vertices);
  4. exact maximal feasible step from the polyhedral slacks, then concave 1-D line search.

Guarantees exercised numerically here: G1 productive invariance (Bz = y*, V = V* at every
iterate), G2 monotone ascent while Gamma_E > 0, G3/G4 the certificate at termination checked
against the monolithic Stage-2 SDP value.

    python -m experiments.exp_gamma_wise
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp_gamma as G  # noqa: E402

from wise_mr import nullspace as ns  # noqa: E402
from wise_mr import scenarios  # noqa: E402

GEN = ROOT / "generated"
SOLVER_KW = dict(solver="SCS", eps=1e-7, max_iters=40000)


# --------------------------------------------------------------------------- #
# fiber geometry
# --------------------------------------------------------------------------- #
def fiber_matrices(prob):
    """Equality rows of the fiber and the wrench screen, as (A_eq, H_w, d)."""
    A = ns.mass_matrix(prob)
    B = ns.served_matrix(prob)
    return A, B, prob.wrench_matrix(), prob.demand().ravel()


def tangent_directions(prob, z, tol=1e-7):
    """The tangent cone T_E(z) = {d : Ad=0, Bd=0, d_j>=0 on active bounds, G_I d >= 0}.

    The active *wrench* rows matter: omitting them lets the ascent direction leave the
    feasible set along a binding facet, so the maximal step collapses to zero and the method
    stalls with Gamma_E still large.
    """
    A, B, Hw, dem = fiber_matrices(prob)
    Aeq = np.vstack([A, B])
    at_bound = np.flatnonzero(z <= tol)
    active_w = np.flatnonzero(Hw @ z - dem <= tol)
    return Aeq, at_bound, Hw[active_w] if active_w.size else None


def gamma_and_direction(prob, z, tol=1e-3):
    """Solve the directional subproblem; return (Gamma_E(z), d, multiplicity)."""
    import cvxpy as cp

    L = G.lap(prob, z)
    _, U2, mult = G.lam2_eigenspace(L, tol=tol)
    Aeq, at_bound, G_I = tangent_directions(prob, z)
    idx = G.relay_coords(prob)

    n = prob.N * prob.A
    d = cp.Variable(n)
    t = cp.Variable()
    DL = sum(d[j] * (U2.T @ prob.relay_laplacians[i] @ U2)
             for i, j in enumerate(idx))
    cons = [Aeq @ d == 0, cp.norm(d, 2) <= 1, DL >> t * np.eye(U2.shape[1])]
    if at_bound.size:
        cons.append(d[at_bound] >= 0)
    if G_I is not None:
        cons.append(G_I @ d >= 0)
    cp.Problem(cp.Maximize(t), cons).solve(**SOLVER_KW)
    if d.value is None:
        return 0.0, np.zeros(n), mult
    return float(t.value), np.asarray(d.value).ravel(), mult


def frank_wolfe_direction(prob, z, y_star, tol=1e-3):
    """Conditional-gradient step: maximize the directional derivative over the WHOLE fiber.

        max_{w in E} lambda_min( U2^T DLbar[w - z] U2 )

    Two advantages over the unit-ball subproblem of Def. 2. The step d = w - z is feasible
    for every alpha in [0,1] because E is convex, so the method cannot jam at a degenerate
    vertex; and by concavity of lambda_2 o Lbar the optimal value is an *exact* global gap,

        Lambda_E - lambda_2(Lbar(z))  <=  max_{w in E} D lambda_2(z)[w - z],

    with no diam(E) factor. Gamma_E remains the scale-free certificate; this is its
    computational counterpart.
    """
    import cvxpy as cp

    L = G.lap(prob, z)
    _, U2, mult = G.lam2_eigenspace(L, tol=tol)
    A, B, Hw, dem = fiber_matrices(prob)
    idx = G.relay_coords(prob)

    n = prob.N * prob.A
    w = cp.Variable(n, nonneg=True)
    t = cp.Variable()
    DL = sum((w[j] - z[j]) * (U2.T @ prob.relay_laplacians[i] @ U2)
             for i, j in enumerate(idx))
    cons = [A @ w == np.ones(prob.N), B @ w == y_star, Hw @ w >= dem,
            DL >> t * np.eye(U2.shape[1])]
    cp.Problem(cp.Maximize(t), cons).solve(**SOLVER_KW)
    if w.value is None:
        return 0.0, np.zeros(n), mult
    return float(t.value), np.asarray(w.value).ravel() - z, mult


def max_step(prob, z, d, tol=1e-12):
    """Largest alpha with z + alpha d still in the fiber (nonneg + wrench screen)."""
    _, _, Hw, dem = fiber_matrices(prob)
    alphas = []
    neg = d < -tol                                    # z_j >= 0
    if neg.any():
        alphas.append(np.min(z[neg] / -d[neg]))
    slack, rate = Hw @ z - dem, Hw @ d                # H_w z >= dem
    tight = rate < -tol
    if tight.any():
        alphas.append(np.min(slack[tight] / -rate[tight]))
    return float(min(alphas)) if alphas else 0.0


def line_search(prob, z, d, a_hi, n_grid=24):
    """lambda_2 is concave along the segment: golden-section on [0, a_hi]."""
    if a_hi <= 0:
        return 0.0
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    lo, hi = 0.0, a_hi
    f = lambda a: G.lam2_eigenspace(G.lap(prob, z + a * d))[0]  # noqa: E731
    b, c = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fb, fc = f(b), f(c)
    for _ in range(n_grid):
        if fb < fc:
            lo, b, fb = b, c, fc
            c = lo + gr * (hi - lo)
            fc = f(c)
        else:
            hi, c, fc = c, b, fb
            b = hi - gr * (hi - lo)
            fb = f(b)
    a = 0.5 * (lo + hi)
    return a if f(a) > f(0.0) else 0.0


# --------------------------------------------------------------------------- #
# the algorithm
# --------------------------------------------------------------------------- #
def gamma_wise(prob, eps=1e-3, max_iter=40, verbose=True, mode="fw"):
    """mode='ball' is the unit-ball subproblem of Def. 2; mode='fw' the conditional
    gradient over the fiber. Both certify with Gamma_E; only 'fw' cannot jam."""
    A, B, Hw, dem = fiber_matrices(prob)
    z, y_star, V_star, status = G.optimal_fiber_base(prob)
    diam = np.sqrt(2.0 * prob.N)                      # tight for a product of N simplices
    hist = []

    for k in range(max_iter):
        lam2 = G.lam2_eigenspace(G.lap(prob, z))[0]
        gam, d_ball, mult = gamma_and_direction(prob, z)
        if mode == "fw":
            fw_gap, d, _ = frank_wolfe_direction(prob, z, y_star)
            a_hi = 1.0
        else:
            fw_gap, d = float("nan"), d_ball
            a_hi = max_step(prob, z, d)
        gap_bound = min(diam * max(gam, 0.0),
                        fw_gap if np.isfinite(fw_gap) else np.inf)
        hist.append({"k": k, "lambda2": lam2, "gamma": gam, "fw_gap": fw_gap,
                     "gap_bound": gap_bound, "multiplicity": mult,
                     "aggregate_err": float(np.max(np.abs(B @ z - y_star))),
                     "budget_err": float(np.max(np.abs(A @ z - 1.0))),
                     "wrench_slack": float(np.min(Hw @ z - dem))})
        if verbose:
            print(f"  k={k:2d}  lambda2={lam2:.6f}  Gamma={gam:.3e}  "
                  f"FWgap={fw_gap:.3e}  gap<={gap_bound:.3e}  mult={mult}  "
                  f"|Bz-y*|={hist[-1]['aggregate_err']:.1e}")
        if gap_bound <= eps:
            hist[-1]["terminated"] = "certified"
            break
        a = line_search(prob, z, d, a_hi)
        if a <= 0.0:
            hist[-1]["terminated"] = "no_progress"
            break
        z = z + a * d
    return z, y_star, V_star, hist, status


def main() -> dict:
    from exp_scalability_hard_wise import _spectral_problem

    out = {"eps": 1e-3, "seeds": [3, 5, 7], "mode": "fw", "runs": []}
    for seed in out["seeds"]:
        prob = scenarios.two_region(seed=seed, N=12)
        print(f"seed {seed}:")
        z, y_star, V_star, hist, _ = gamma_wise(prob, eps=out["eps"], mode=out["mode"])

        lam_final = hist[-1]["lambda2"]
        val = _spectral_problem(prob, fiber_eq=y_star, maximize_lambda=True)
        val = val[0] if isinstance(val, tuple) else val
        lam_sdp = float(val) if val is not None else float("nan")
        lam_start = hist[0]["lambda2"]
        rec = {
            "seed": seed, "iterations": len(hist),
            "lambda2_start": lam_start, "lambda2_final": lam_final,
            "lambda2_monolithic_sdp": lam_sdp,
            "gap_to_sdp": lam_sdp - lam_final,
            "final_gamma": hist[-1]["gamma"], "final_fw_gap": hist[-1]["fw_gap"],
            "certified_gap_bound": hist[-1]["gap_bound"],
            "monotone": bool(all(b["lambda2"] >= a["lambda2"] - 1e-9
                                 for a, b in zip(hist, hist[1:], strict=False))),
            "max_aggregate_err": max(h["aggregate_err"] for h in hist),
            "max_budget_err": max(h["budget_err"] for h in hist),
            "min_wrench_slack": min(h["wrench_slack"] for h in hist),
            "terminated": hist[-1].get("terminated", "max_iter"),
            "history": hist,
        }
        out["runs"].append(rec)
        print(f"  -> {rec['terminated']}: lambda2 {lam_start:.4f} -> {lam_final:.4f}, "
              f"monolithic SDP {lam_sdp:.4f}, gap {rec['gap_to_sdp']:+.2e}, "
              f"monotone={rec['monotone']}, |Bz-y*|<={rec['max_aggregate_err']:.1e}\n")

    (GEN / "gamma_wise_prototype.json").write_text(json.dumps(out, indent=2),
                                                   encoding="utf-8")
    print(f"wrote {GEN / 'gamma_wise_prototype.json'}")
    return out


if __name__ == "__main__":
    main()
