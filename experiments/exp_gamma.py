"""E-gamma (paper Thm. optimal-fiber networkability): the free-networkability modulus.

At a point ``zbar`` of the optimal fiber ``E = {z in X_f : B z = y*}`` define

    Gamma_E(zbar) = max { D lambda_2(zbar)[d] : d in T_E(zbar), ||d||_2 <= 1 },

the *free-networkability modulus*.  Because ``E`` is a polytope and
``f = lambda_2 . Lbar`` is concave (affine Lbar, pointwise-min eigenvalue), the
directional test is a **global** certificate over the whole fiber:

    Gamma_E(zbar) > 0  <=>  exists z' in E with lambda_2(z') > lambda_2(zbar),
    Gamma_E(zbar) = 0  <=>  zbar in argmax_{z in E} lambda_2(z).

This script certifies both directions numerically:

  (a) at an interior fiber point ``zbar`` we compute Gamma_E three ways
      -- the projection formula ||Pi_{N_p} g_lambda|| (simple lambda_2),
         the eigenspace SDP max lambda_min(U_2^T DLbar[d] U_2) (general),
         and a one-sided finite-difference along the argmax direction --
      and check they agree;
  (b) we solve the stage-2 selection SDP for Lambda_E = max_E lambda_2 and check
      the equivalence  Gamma_E(zbar) > 0  <=>  Lambda_E > lambda_2(zbar), plus
      Gamma_E(z_WISE) ~ 0 at the selector (global optimality certificate);
  (c) we report the *network-visible* neutral dimension
      d_net = rank(DLbar restricted to N_p), separating raw productive
      degeneracy dim E from the part the communication layer can actually see.

Writes generated/gamma_certificate.json and generated/gamma_sweep.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import nullspace as ns, scenarios  # noqa: E402
from wise_mr.endogenous_graph import complement_basis  # noqa: E402

GEN = ROOT / "generated"
GEN.mkdir(exist_ok=True)

EIG_TOL = 1e-7      # eigenvalue clustering tolerance for the lambda_2 eigenspace
NULL_TOL = 1e-8     # singular-value threshold for null spaces / ranks
SEEDS = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


# --------------------------------------------------------------------------- #
# problem plumbing
# --------------------------------------------------------------------------- #

def mat(prob, z: np.ndarray) -> np.ndarray:
    """Reshape a flat decision vector to the (N, A) layout the WiseProblem API wants."""
    return np.asarray(z, dtype=float).reshape(prob.N, prob.A)


def lam2(prob, z: np.ndarray) -> float:
    return float(prob.lambda2(mat(prob, z)))


def lap(prob, z: np.ndarray) -> np.ndarray:
    return np.asarray(prob.laplacian(mat(prob, z)), dtype=float)


def relay_coords(prob) -> np.ndarray:
    """Indices j of z (flattened (N, A)) carrying relay occupancy z_{i,r}."""
    return np.array([i * prob.A + prob.relay_index for i in range(prob.N)])


def dLbar(prob, d: np.ndarray) -> np.ndarray:
    """Directional derivative DLbar[d] = sum_{i,r} d_{ir} L_{ir} (Lbar is affine)."""
    idx = relay_coords(prob)
    out = np.zeros_like(prob.base_laplacian, dtype=float)
    for i, j in enumerate(idx):
        out = out + float(d[j]) * prob.relay_laplacians[i]
    return out


def lam2_eigenspace(L: np.ndarray, tol: float = EIG_TOL):
    """(lambda_2, U_2) with U_2 orthonormal basis of the lambda_2-eigenspace in 1^perp.

    U_2 lives in R^{N x m}: eigenvectors of Lbar itself (all orthogonal to 1 for a
    connected-or-not Laplacian since lambda_1 = 0 has eigenvector 1). This is exactly
    what the repeated-Fiedler directional derivative formula needs.
    """
    w, V = np.linalg.eigh(L)
    lam2 = float(w[1])
    mask = np.abs(w - lam2) <= tol * max(1.0, abs(lam2))
    mask[0] = False                      # never include the consensus mode
    U2 = V[:, mask]
    return lam2, U2, int(mask.sum())


def optimal_fiber_base(prob):
    """zbar in relint(E) via the two-stage QP used by exp_fiber (same construction)."""
    import cvxpy as cp

    n = prob.N * prob.A
    A = ns.mass_matrix(prob)
    B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix()
    dvec = prob.demand().ravel()
    v = prob._value()
    one = np.ones(prob.N)

    z1 = cp.Variable(n, nonneg=True)
    y1 = B @ z1
    V1 = cp.sum(cp.multiply(v, y1) - 0.5 * prob.alpha * cp.square(y1))
    p1 = cp.Problem(cp.Maximize(V1), [A @ z1 == one, Hw @ z1 >= dvec])
    p1.solve()
    y_star = np.asarray(B @ z1.value).ravel()
    V_star = float(V1.value)

    # stage 2: an interior point of the fiber (max-entropy-like: closest to uniform)
    z2 = cp.Variable(n, nonneg=True)
    z_unif = np.full(n, 1.0 / prob.A)
    p2 = cp.Problem(cp.Minimize(cp.sum_squares(z2 - z_unif)),
                    [A @ z2 == one, B @ z2 == y_star, Hw @ z2 >= dvec])
    p2.solve()
    return np.asarray(z2.value).ravel(), y_star, V_star, f"{p1.status}/{p2.status}"


def tangent_cone_data(prob, z, tol: float = 1e-3):
    """Equality rows (A;B) and active inequality rows G_I (as g^T z >= h) at z."""
    A = ns.mass_matrix(prob)
    B = ns.served_matrix(prob)
    G_I = ns.active_inequalities(prob, z, tol)
    return A, B, G_I


# --------------------------------------------------------------------------- #
# Gamma_E
# --------------------------------------------------------------------------- #

def gamma_projection(prob, z, Np: np.ndarray) -> float:
    """||Pi_{N_p} g_lambda||_2 -- valid when lambda_2 is simple and T_E = N_p."""
    L = lap(prob, z)
    lam2v, U2, mult = lam2_eigenspace(L)
    if mult != 1 or Np.shape[0] == 0:
        return float("nan")
    v = U2[:, 0]
    idx = relay_coords(prob)
    g = np.zeros(prob.N * prob.A)
    for i, j in enumerate(idx):
        g[j] = float(v @ prob.relay_laplacians[i] @ v)
    # Np rows are an orthonormal basis of the neutral space
    return float(np.linalg.norm(Np @ g))


def gamma_sdp(prob, z, *, use_cone: bool = True, tol: float = 1e-3):
    """Gamma_E(z) by the eigenspace SDP, valid for simple *and* repeated lambda_2.

        max_{d,t} t   s.t.  A d = 0, B d = 0, G_I d >= 0 (active rows),
                            ||d||_2 <= 1,
                            U_2^T DLbar[d] U_2  >=  t I.
    """
    import cvxpy as cp

    n = prob.N * prob.A
    L = lap(prob, z)
    lam2v, U2, mult = lam2_eigenspace(L)
    A, B, G_I = tangent_cone_data(prob, z, tol)

    d = cp.Variable(n)
    t = cp.Variable()
    idx = relay_coords(prob)
    Md = sum(d[int(j)] * prob.relay_laplacians[i] for i, j in enumerate(idx))
    proj = U2.T @ Md @ U2

    cons = [A @ d == 0, B @ d == 0, cp.norm(d, 2) <= 1,
            proj - t * np.eye(U2.shape[1]) >> 0]
    if use_cone and G_I.size:
        cons.append(G_I @ d >= 0)
    p = cp.Problem(cp.Maximize(t), cons)
    p.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
    if d.value is None:
        return float("nan"), None, mult, p.status
    return float(t.value), np.asarray(d.value).ravel(), mult, p.status


def gamma_finite_difference(prob, z, d: np.ndarray, steps=(1e-4, 1e-5, 1e-6)):
    """One-sided directional difference (lambda_2 need not be differentiable)."""
    lam0 = lam2(prob, z)
    return {f"{s:g}": float((lam2(prob, z + s * d) - lam0) / s) for s in steps}


def net_visible_dimension(prob, Np: np.ndarray) -> int:
    """d_net = rank of the linear map d -> DLbar[d] restricted to N_p.

    Bounded above by dim E; measures how much of the productive degeneracy the
    communication layer can see *at all* (Gamma_E measures how much it can use
    at the current point).
    """
    if Np.shape[0] == 0:
        return 0
    cols = [dLbar(prob, b).ravel() for b in Np]
    Mstack = np.array(cols)
    return int(np.linalg.matrix_rank(Mstack, tol=NULL_TOL))


def lambda_E_sdp(prob, y_star: np.ndarray):
    """Stage-2 selection SDP: Lambda_E = max_{z in E} lambda_2, and the selector."""
    import cvxpy as cp

    n = prob.N * prob.A
    A = ns.mass_matrix(prob)
    B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix()
    dvec = prob.demand().ravel()
    Q = complement_basis(prob.N)

    z = cp.Variable(n, nonneg=True)
    t = cp.Variable()
    idx = relay_coords(prob)
    Lz = cp.Constant(np.asarray(prob.base_laplacian, dtype=float))
    for i, j in enumerate(idx):
        Lz = Lz + z[int(j)] * np.asarray(prob.relay_laplacians[i], dtype=float)
    cons = [A @ z == np.ones(prob.N), B @ z == y_star, Hw @ z >= dvec,
            Q.T @ Lz @ Q - t * np.eye(prob.N - 1) >> 0]
    p = cp.Problem(cp.Maximize(t), cons)
    p.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
    return float(t.value), np.asarray(z.value).ravel(), p.status


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def run_seed(seed: int) -> dict:
    prob = scenarios.two_region(seed=seed)
    zbar, y_star, V_star, status = optimal_fiber_base(prob)

    info = ns.fiber_dimension(prob, zbar)
    Np = info["Np_basis"]

    lam_bar = lam2(prob, zbar)
    g_proj = gamma_projection(prob, zbar, Np)
    g_sdp, d_star, mult, sdp_status = gamma_sdp(prob, zbar)
    fd = gamma_finite_difference(prob, zbar, d_star) if d_star is not None else {}
    d_net = net_visible_dimension(prob, Np)

    Lam_E, z_wise, lamE_status = lambda_E_sdp(prob, y_star)
    g_at_wise, _, mult_w, _ = gamma_sdp(prob, z_wise)

    return {
        "seed": seed,
        "N": prob.N,
        "dim_E": info["dim_E"],
        "d_net": d_net,
        "n_active_ineq": info["n_active_ineq"],
        "lambda2_zbar": lam_bar,
        "multiplicity_zbar": mult,
        "Gamma_projection": g_proj,
        "Gamma_sdp": g_sdp,
        "Gamma_fd": fd,
        "Lambda_E": Lam_E,
        "lambda2_zwise": lam2(prob, z_wise),
        "Gamma_at_wise": g_at_wise,
        "multiplicity_zwise": mult_w,
        "reserve_Lambda_minus_lambda_bar": Lam_E - lam_bar,
        # theorem checks
        "iff_forward_ok": bool((g_sdp > 1e-6) == (Lam_E - lam_bar > 1e-6)),
        # concavity certificate: Lambda_E - lambda_2(zbar) <= Gamma_E(zbar) * ||z_WISE - zbar||
        "step_norm": float(np.linalg.norm(z_wise - zbar)),
        "concavity_lhs": Lam_E - lam_bar,
        "concavity_rhs": g_sdp * float(np.linalg.norm(z_wise - zbar)),
        "concavity_bound_ok": bool(
            Lam_E - lam_bar <= g_sdp * float(np.linalg.norm(z_wise - zbar)) + 1e-7),
        # four-orders separation is the operative certificate, not a hard threshold
        "gamma_separation_ratio": (float(abs(g_at_wise) / g_sdp) if g_sdp > 0
                                   else float("nan")),
        "zero_at_argmax_ok": bool(abs(g_at_wise) <= 1e-3 * g_sdp),
        "proj_vs_sdp_absdiff": (float("nan") if np.isnan(g_proj)
                                else abs(g_proj - g_sdp)),
        "V_star": V_star,
        "status": status,
        "sdp_status": sdp_status,
        "lamE_status": lamE_status,
    }


def main() -> None:
    rows = []
    for s in SEEDS:
        try:
            r = run_seed(s)
        except Exception as exc:                       # noqa: BLE001
            print(f"seed {s}: FAILED {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        print(f"seed {s:3d}  dimE={r['dim_E']:3d}  d_net={r['d_net']:3d}  "
              f"mult={r['multiplicity_zbar']}  lam2={r['lambda2_zbar']:.4f}  "
              f"Gamma={r['Gamma_sdp']:.6f}  Lambda_E={r['Lambda_E']:.4f}  "
              f"Gamma(WISE)={r['Gamma_at_wise']:.2e}  "
              f"iff={r['iff_forward_ok']}  argmax={r['zero_at_argmax_ok']}  "
              f"concav={r['concavity_bound_ok']}")

    if not rows:
        raise SystemExit("no seeds solved")

    n = len(rows)
    summary = {
        "n_seeds": n,
        "eig_tol": EIG_TOL,
        "null_tol": NULL_TOL,
        "iff_forward_ok": int(sum(r["iff_forward_ok"] for r in rows)),
        "zero_at_argmax_ok": int(sum(r["zero_at_argmax_ok"] for r in rows)),
        "max_proj_vs_sdp_absdiff": float(np.nanmax(
            [r["proj_vs_sdp_absdiff"] for r in rows])),
        "max_Gamma_at_wise": float(max(abs(r["Gamma_at_wise"]) for r in rows)),
        "concavity_bound_ok": int(sum(r["concavity_bound_ok"] for r in rows)),
        "max_gamma_separation_ratio": float(max(r["gamma_separation_ratio"] for r in rows)),
        "dim_E_range": [min(r["dim_E"] for r in rows), max(r["dim_E"] for r in rows)],
        "d_net_range": [min(r["d_net"] for r in rows), max(r["d_net"] for r in rows)],
        "rows": rows,
    }
    (GEN / "gamma_certificate.json").write_text(json.dumps(summary, indent=1))

    fields = [k for k in rows[0] if k != "Gamma_fd"]
    with (GEN / "gamma_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    print(f"\n{n} seeds | local-global iff {summary['iff_forward_ok']}/{n} | "
          f"Gamma(argmax)=0 in {summary['zero_at_argmax_ok']}/{n} "
          f"(max |Gamma| = {summary['max_Gamma_at_wise']:.2e}) | "
          f"proj-vs-SDP max diff {summary['max_proj_vs_sdp_absdiff']:.2e}")
    print(f"dim E {summary['dim_E_range']}  vs  d_net {summary['d_net_range']}")
    print(f"concavity bound Lambda_E - lambda_2 <= Gamma*||step||: "
          f"{summary['concavity_bound_ok']}/{n} | "
          f"max Gamma(WISE)/Gamma(zbar) = {summary['max_gamma_separation_ratio']:.1e}")


if __name__ == "__main__":
    main()
