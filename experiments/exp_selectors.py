"""E-selectors: secondary selectors on the SAME optimal fiber (WISE is the max-margin one).

All selectors return a point of E = {z in X_f : Bz = y*} (same productive value V*). They
differ only in the secondary criterion:
  * random-fiber   -- maximize a random linear objective on E (an arbitrary fiber vertex);
  * min-role-change-- minimize ||z - z_uniform||^2 on E (least deviation from a reference);
  * WISE           -- maximize lambda_2(Lbar(z)) on E (the lexicographic tie-break).
We report the mean connectivity margin lambda_2 - sigma_req of each over 20 seeds; WISE
should dominate, showing the tie-break matters among equally productive optima.
Writes generated/selectors.csv.
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
GEN.mkdir(exist_ok=True)


def _fiber_data(prob):
    import cvxpy as cp
    A = ns.mass_matrix(prob); B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix(); d = prob.demand().ravel(); v = prob._value()
    n = prob.N * prob.A
    z = cp.Variable(n, nonneg=True)
    cp.Problem(cp.Maximize(cp.sum(cp.multiply(v, B @ z) - 0.5 * prob.alpha * cp.square(B @ z))),
               [A @ z == np.ones(prob.N), Hw @ z >= d]).solve()
    y_star = np.asarray(B @ np.maximum(z.value, 0.0), float)
    return A, B, Hw, d, n, y_star


def _fiber_cons(prob, z, A, B, Hw, d, y_star, cp):
    return [z >= 0, A @ z == np.ones(prob.N), B @ z == y_star, Hw @ z >= d]


def _margin(prob, zval):
    x = np.maximum(zval, 0.0).reshape(prob.N, prob.A)
    return float(prob.lambda2(x) - prob.sigma)


def run(seeds=20):
    import cvxpy as cp
    rows = {m: [] for m in ("random_fiber", "min_role_change", "wise")}
    for s in range(seeds):
        prob = scenarios.two_region(seed=s, N=12, nu=0.4, tau_d=5.0, bridge_gain=3.0)
        A, B, Hw, d, n, y_star = _fiber_data(prob)
        rng = np.random.default_rng(s)

        z = cp.Variable(n)
        cp.Problem(cp.Maximize(rng.standard_normal(n) @ z),
                   _fiber_cons(prob, z, A, B, Hw, d, y_star, cp)).solve()
        if z.value is not None:
            rows["random_fiber"].append(_margin(prob, z.value))

        z = cp.Variable(n)
        cp.Problem(cp.Minimize(cp.sum_squares(z - 1.0 / prob.A)),
                   _fiber_cons(prob, z, A, B, Hw, d, y_star, cp)).solve()
        if z.value is not None:
            rows["min_role_change"].append(_margin(prob, z.value))

        # WISE: max lambda_2 over the fiber (the selection SDP)
        A_eq = np.vstack([A, B]); b_eq = np.concatenate([np.ones(prob.N), y_star])
        G = np.vstack([-np.eye(n), -Hw]); h = np.concatenate([np.zeros(n), -d])
        ridx = prob.relay_index
        terms = [(i * prob.A + ridx, prob.relay_laplacians[i]) for i in range(prob.N)]
        r = wise_sdp.solve_wise_sdp(n, prob.base_laplacian, terms, complement_basis(prob.N),
                                    prob.sigma, A_eq=A_eq, b_eq=b_eq, G_ineq=G, h_ineq=h)
        rows["wise"].append(float(r.lambda_star - prob.sigma))
        print(f"seed {s}: rand={rows['random_fiber'][-1]:+.2f} "
              f"minrole={rows['min_role_change'][-1]:+.2f} wise={rows['wise'][-1]:+.2f}")

    out = []
    for m, vals in rows.items():
        out.append(dict(selector=m, mean_margin=float(np.mean(vals)),
                        min_margin=float(np.min(vals))))
        print(f"{m:>16}: mean margin {out[-1]['mean_margin']:+.2f} "
              f"(min {out[-1]['min_margin']:+.2f})")
    with open(GEN / "selectors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    return out


if __name__ == "__main__":
    run()
