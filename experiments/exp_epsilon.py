"""E-epsilon (paper Sec. V): lexicographic WISE vs. weighted-sum scalarization.

The scalarized objective V_eps(z) = V(z) + eps * lambda_2(Lbar(z)) trades productive
value against connectivity, so a finite eps moves the served aggregate away from y*.
The lexicographic WISE selector (Stage-2 SDP) preserves y* exactly. We sweep eps and
show the productive loss V* - V(z_eps) and the aggregate drift ||B z_eps - y*|| both
grow with eps, while WISE has zero of each at maximal connectivity.

Writes generated/epsilon_sweep.csv and paper/figures/fig_epsilon.pdf.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import baselines, nullspace as ns, scenarios, wise_sdp  # noqa: E402
from wise_mr.endogenous_graph import complement_basis  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

EPS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def _lap_expr(prob, z, cp):
    """cvxpy affine expression Lbar(z) = L0 + sum_i z[i,relay] relay_L[i]."""
    L = cp.Constant(np.asarray(prob.base_laplacian, float))
    ridx = prob.relay_index
    for i in range(prob.N):
        L = L + z[i * prob.A + ridx] * np.asarray(prob.relay_laplacians[i], float)
    return L


def _weighted_sum(prob, eps):
    """max V(z) + eps * lambda_2(Lbar(z))  s.t. z in X_f (aggregate free)."""
    import cvxpy as cp

    n = prob.N * prob.A
    A = ns.mass_matrix(prob); B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix(); d = prob.demand().ravel()
    v = prob._value()
    Q = complement_basis(prob.N)
    z = cp.Variable(n, nonneg=True); t = cp.Variable()
    y = B @ z
    V = cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))
    cons = [A @ z == np.ones(prob.N), Hw @ z >= d,
            Q.T @ _lap_expr(prob, z, cp) @ Q >> t * np.eye(prob.N - 1)]
    cp.Problem(cp.Maximize(V + eps * t), cons).solve()
    if z.value is None:
        return None
    return np.maximum(np.asarray(z.value, float), 0.0)


def _wise(prob):
    """Lexicographic WISE: y* then max lambda_2 on the fiber (Stage-2 SDP)."""
    import cvxpy as cp
    n = prob.N * prob.A
    A = ns.mass_matrix(prob); B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix(); d = prob.demand().ravel()
    v = prob._value()
    z = cp.Variable(n, nonneg=True); y = B @ z
    cp.Problem(cp.Maximize(cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))),
               [A @ z == np.ones(prob.N), Hw @ z >= d]).solve()
    y_star = B @ z.value
    A_eq = np.vstack([A, B]); b_eq = np.concatenate([np.ones(prob.N), y_star])
    G = np.vstack([-np.eye(n), -Hw]); h = np.concatenate([np.zeros(n), -d])
    ridx = prob.relay_index
    terms = [(i * prob.A + ridx, prob.relay_laplacians[i]) for i in range(prob.N)]
    res = wise_sdp.solve_wise_sdp(n, prob.base_laplacian, terms, complement_basis(prob.N),
                                  prob.sigma, A_eq=A_eq, b_eq=b_eq, G_ineq=G, h_ineq=h)
    return y_star, np.maximum(res.z.reshape(prob.N, prob.A), 0.0)


def run(seeds=8):
    rows = []
    for eps in EPS:
        ploss, adrift, lam = [], [], []
        for s in range(seeds):
            prob = scenarios.two_region(seed=s, N=12, nu=0.4, tau_d=5.0, bridge_gain=3.0)
            y_star, z_w = _wise(prob)
            v_star = prob.productive_value(z_w)
            z_e = _weighted_sum(prob, eps)
            if z_e is None:
                continue
            y_e = ns.served_matrix(prob) @ z_e
            z_e = z_e.reshape(prob.N, prob.A)
            ploss.append(v_star - prob.productive_value(z_e))
            adrift.append(float(np.linalg.norm(y_e - y_star)))
            lam.append(prob.lambda2(z_e))
        rows.append(dict(eps=eps, prod_loss=float(np.mean(ploss)),
                         agg_drift=float(np.mean(adrift)), lambda2=float(np.mean(lam))))
    with open(GEN / "epsilon_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    _figure(rows)
    for r in rows:
        print(f"eps={r['eps']:.0e}: prod_loss={r['prod_loss']:.3f}, "
              f"agg_drift={r['agg_drift']:.3f}, lam2={r['lambda2']:.3f}")
    return rows


def _figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    eps = [r["eps"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    axr = ax.twinx()
    l1, = ax.plot(eps, [r["prod_loss"] for r in rows], "o-", color="#c0392b",
                  lw=1.6, ms=3.5, label=r"productive loss $V^\star-V$")
    l2, = axr.plot(eps, [r["agg_drift"] for r in rows], "s--", color="#1f5fbf",
                   lw=1.6, ms=3.5, label=r"aggregate drift $\|Bz-y^\star\|$")
    ax.axhline(0, color="#2e8b57", lw=1.4)
    ax.text(eps[0], 0, "WISE (lexicographic): 0", color="#2e8b57", fontsize=7,
            va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel(r"scalarization weight $\varepsilon$")
    ax.set_ylabel(r"productive loss", color="#c0392b")
    axr.set_ylabel(r"aggregate drift", color="#1f5fbf")
    ax.set_title("weighted sum moves the optimum; WISE does not", fontsize=8.5)
    ax.legend(handles=[l1, l2], loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_epsilon.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    run()
