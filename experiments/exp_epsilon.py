"""E-epsilon (paper Sec. V + Thm. selection): lexicographic WISE vs. weighted-sum.

The scalarised objective V_eps(z) = V(z) + eps * lambda_2(Lbar(z)) trades productive
value against connectivity, so a finite eps moves the served aggregate away from y*. The
lexicographic WISE selector (Stage-2 SDP) preserves y* exactly. Both the scalarisation
and WISE are solved as **global** CVXPY SDPs over the lifted candidate-site variables
(no local lambda_2 optimisation): the LMI is Q^T Lbar(z) Q >= t I.

Per seed we classify the regime by the exact spectral optima
    Lambda_E = max_{z in E} lambda_2   (over the optimal fiber),
    Lambda_X = max_{z in X_f} lambda_2 (over all wrench-feasible z),
and use only **free** seeds (Lambda_E >= sigma_req) for the zero-productive-loss claim.
We overlay the manuscript's Tikhonov bounds
    0 <= V* - V(z_eps) <= eps * osc(lambda_2),
    ||B z_eps - y*|| <= sqrt(2 eps osc(lambda_2) / alpha).

Writes generated/epsilon_sweep.csv (per seed x eps) and paper/figures/fig_epsilon.pdf.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import nullspace as ns, scenarios, wise_sdp  # noqa: E402
from wise_mr.endogenous_graph import complement_basis  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

EPS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


# --- one canonical vectorization convention: z_mat (N,A) <-> z_vec (N*A,), order C ---
def as_vec(z_mat, prob):
    z = np.asarray(z_mat, float)
    assert z.shape == (prob.N, prob.A), f"expected {(prob.N, prob.A)}, got {z.shape}"
    return z.reshape(-1)          # C order: robot i outer, action a inner


def as_mat(z_vec, prob):
    z = np.asarray(z_vec, float).reshape(-1)
    assert z.size == prob.N * prob.A, f"expected {prob.N * prob.A}, got {z.size}"
    return z.reshape(prob.N, prob.A)


def _lap_expr(prob, z, cp):
    """cvxpy affine Lbar(z) = L0 + sum_i z[i,relay] * relay_L[i]."""
    L = cp.Constant(np.asarray(prob.base_laplacian, float))
    ridx = prob.relay_index
    for i in range(prob.N):
        L = L + z[i * prob.A + ridx] * np.asarray(prob.relay_laplacians[i], float)
    return L


def _data(prob):
    A = ns.mass_matrix(prob); B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix(); d = prob.demand().ravel()
    v = prob._value()
    n = prob.N * prob.A
    assert B.shape[1] == n and Hw.shape[1] == n, "matrix/vector convention mismatch"
    return A, B, Hw, d, v, n


def _y_star(prob):
    """Stage 1: productive optimum aggregate y* and value V*."""
    import cvxpy as cp
    A, B, Hw, d, v, n = _data(prob)
    z = cp.Variable(n, nonneg=True); y = B @ z
    cp.Problem(cp.Maximize(cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))),
               [A @ z == np.ones(prob.N), Hw @ z >= d]).solve()
    z0 = np.maximum(np.asarray(z.value, float), 0.0)
    return np.asarray(B @ z0, float), float(prob.productive_value(as_mat(z0, prob)))


def _max_lambda2(prob, y_star=None):
    """max lambda_2 over X_f (y_star=None) or over the optimal fiber E (y_star given),
    as an exact SDP. Returns (Lambda, z*)."""
    A, B, Hw, d, v, n = _data(prob)
    A_eq, b_eq = A, np.ones(prob.N)
    if y_star is not None:
        A_eq = np.vstack([A, B]); b_eq = np.concatenate([np.ones(prob.N), y_star])
    G = np.vstack([-np.eye(n), -Hw]); h = np.concatenate([np.zeros(n), -d])
    ridx = prob.relay_index
    terms = [(i * prob.A + ridx, prob.relay_laplacians[i]) for i in range(prob.N)]
    r = wise_sdp.solve_wise_sdp(n, prob.base_laplacian, terms, complement_basis(prob.N),
                                prob.sigma, A_eq=A_eq, b_eq=b_eq, G_ineq=G, h_ineq=h)
    return float(r.lambda_star), np.maximum(r.z, 0.0)


def _weighted_sum(prob, eps):
    """Global SDP:  max phi(Bz) + eps * t  s.t. z in X_f,  Q^T Lbar(z) Q >= t I."""
    import cvxpy as cp
    A, B, Hw, d, v, n = _data(prob)
    Q = complement_basis(prob.N)
    z = cp.Variable(n, nonneg=True); t = cp.Variable()
    y = B @ z
    V = cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))
    cons = [A @ z == np.ones(prob.N), Hw @ z >= d,
            Q.T @ _lap_expr(prob, z, cp) @ Q >> t * np.eye(prob.N - 1)]
    prob_cp = cp.Problem(cp.Maximize(V + eps * t), cons)
    prob_cp.solve()
    if z.value is None:
        return None, str(prob_cp.status)
    return np.maximum(np.asarray(z.value, float), 0.0), str(prob_cp.status)


def _regime(Lam_E, Lam_X, sigma):
    if Lam_E >= sigma - 1e-7:
        return "free"
    if Lam_X >= sigma - 1e-7:
        return "costly"
    return "impossible"


def run(seeds=8):
    rows = []
    for s in range(seeds):
        prob = scenarios.two_region(seed=s, N=12, nu=0.4, tau_d=5.0, bridge_gain=3.0)
        sigma = float(prob.sigma)
        y_star, V_star = _y_star(prob)
        Lam_E, _ = _max_lambda2(prob, y_star=y_star)
        Lam_X, _ = _max_lambda2(prob, y_star=None)
        regime = _regime(Lam_E, Lam_X, sigma)

        lam_at_eps = []
        seed_rows = []
        for eps in EPS:
            z_e, status = _weighted_sum(prob, eps)
            if z_e is None:
                seed_rows.append(dict(seed=s, regime=regime, eps=eps, prod_loss=np.nan,
                                      drift=np.nan, lambda2=np.nan, solver_status=status))
                continue
            y_e = as_vec(as_mat(z_e, prob), prob)          # canonical round-trip
            y_e = ns.served_matrix(prob) @ y_e
            zmat = as_mat(z_e, prob)
            prod_loss = float(V_star - prob.productive_value(zmat))
            drift = float(np.linalg.norm(y_e - y_star))
            lam = float(prob.lambda2(zmat))
            lam_at_eps.append(lam)
            seed_rows.append(dict(seed=s, regime=regime, eps=eps, prod_loss=prod_loss,
                                  drift=drift, lambda2=lam, solver_status=status))
        # osc(lambda2) proxy over the explored set: max over sweep - min over sweep
        osc = float(max(lam_at_eps) - min(lam_at_eps)) if len(lam_at_eps) >= 2 else 0.0
        for r in seed_rows:
            r.update(Lambda_E=Lam_E, Lambda_X=Lam_X, sigma_req=sigma, osc_lambda2=osc,
                     V_star=V_star,
                     bound_loss=r["eps"] * osc,
                     bound_drift=float(np.sqrt(2 * r["eps"] * osc / prob.alpha)))
        rows.extend(seed_rows)
        print(f"seed {s}: regime={regime:>10}  Lam_E={Lam_E:.3f} Lam_X={Lam_X:.3f} "
              f"sigma={sigma:.3f}  osc={osc:.3f}")

    fields = ["seed", "regime", "eps", "prod_loss", "drift", "lambda2",
              "Lambda_E", "Lambda_X", "sigma_req", "osc_lambda2", "V_star",
              "bound_loss", "bound_drift", "solver_status"]
    with open(GEN / "epsilon_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    _figure(rows)
    _summary(rows)
    return rows


def _summary(rows):
    free = [r for r in rows if r["regime"] == "free" and np.isfinite(r["prod_loss"])]
    if not free:
        print("WARNING: no free-regime seeds; zero-loss claim unsupported")
        return
    by_eps = {}
    for r in free:
        by_eps.setdefault(r["eps"], []).append(r)
    print("\n free-regime weighted-sum vs. lexicographic WISE (WISE = 0 loss, 0 drift):")
    for eps in EPS:
        rr = by_eps.get(eps, [])
        if rr:
            pl = np.mean([r["prod_loss"] for r in rr])
            dr = np.mean([r["drift"] for r in rr])
            print(f"  eps={eps:>6}: mean prod_loss={pl:.4f}, mean drift={dr:.4f} "
                  f"(n={len(rr)})")


def _figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    free = [r for r in rows if r["regime"] == "free" and np.isfinite(r["prod_loss"])]
    src = free if free else [r for r in rows if np.isfinite(r["prod_loss"])]
    by_eps = {}
    for r in src:
        by_eps.setdefault(r["eps"], []).append(r)
    eps = sorted(by_eps)
    loss = [float(np.mean([r["prod_loss"] for r in by_eps[e]])) for e in eps]
    drift = [float(np.mean([r["drift"] for r in by_eps[e]])) for e in eps]
    b_loss = [float(np.mean([r["bound_loss"] for r in by_eps[e]])) for e in eps]
    b_drift = [float(np.mean([r["bound_drift"] for r in by_eps[e]])) for e in eps]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    a.plot(eps, loss, "o-", color="#c0392b", lw=1.6, ms=3.5, label="observed")
    a.plot(eps, b_loss, "--", color="#c0392b", lw=1.0, alpha=0.7,
           label=r"Tikhonov $\varepsilon\,\mathrm{osc}(\lambda_2)$")
    a.axhline(0, color="#2e8b57", lw=1.4)
    a.text(eps[0], 0, "WISE: 0", color="#2e8b57", fontsize=7, va="bottom")
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=1e-3)
    a.set_xlabel(r"weight $\varepsilon$", fontsize=8)
    a.set_ylabel(r"productive loss $V^\star-V$", fontsize=8)
    a.set_title("(a) productive loss vs. bound", fontsize=8.5)
    a.legend(fontsize=6.5, frameon=False, loc="upper left")

    b.plot(eps, drift, "s-", color="#1f5fbf", lw=1.6, ms=3.5, label="observed")
    b.plot(eps, b_drift, "--", color="#1f5fbf", lw=1.0, alpha=0.7,
           label=r"Tikhonov $\sqrt{2\varepsilon\,\mathrm{osc}/\alpha}$")
    b.axhline(0, color="#2e8b57", lw=1.4)
    b.text(eps[0], 0, "WISE: 0", color="#2e8b57", fontsize=7, va="bottom")
    b.set_xscale("log")
    b.set_xlabel(r"weight $\varepsilon$", fontsize=8)
    b.set_ylabel(r"aggregate drift $\|Bz-y^\star\|$", fontsize=8)
    b.set_title("(b) aggregate drift vs. bound", fontsize=8.5)
    b.legend(fontsize=6.5, frameon=False, loc="upper left")

    for ax in (a, b):
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_epsilon.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    run()
