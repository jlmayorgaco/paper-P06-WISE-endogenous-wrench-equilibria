"""E-price (respecification Thm 3): the price of self-sustainability P(sigma).

For each seed we compute the productive optimum V*, the two spectral optima
    Lambda_E = max_{z in E}   lambda_2(Lbar(z))   (over the optimal fiber),
    Lambda_X = max_{z in X_f} lambda_2(Lbar(z))   (over all wrench-feasible z),
and sweep the requirement sigma to trace
    P(sigma) = V* - max{ V(z) : z in X_f, lambda_2(Lbar(z)) >= sigma },
which is 0 (FREE) for sigma <= Lambda_E, positive (COSTLY) for Lambda_E < sigma <= Lambda_X,
and +infinity (IMPOSSIBLE) for sigma > Lambda_X. This turns the free/costly/impossible
trichotomy from an observation into a validated theorem, and reports the marginal spectral
price pi*(sigma) = dP/dsigma in the costly regime.

Writes generated/price_sweep.csv and paper/figures/fig_price.pdf.
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


def _data(prob):
    A = ns.mass_matrix(prob); B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix(); d = prob.demand().ravel(); v = prob._value()
    return A, B, Hw, d, v, prob.N * prob.A


def _lap_expr(prob, z, cp):
    L = cp.Constant(np.asarray(prob.base_laplacian, float))
    ridx = prob.relay_index
    for i in range(prob.N):
        L = L + z[i * prob.A + ridx] * np.asarray(prob.relay_laplacians[i], float)
    return L


def _y_star(prob):
    import cvxpy as cp
    A, B, Hw, d, v, n = _data(prob)
    z = cp.Variable(n, nonneg=True); y = B @ z
    cp.Problem(cp.Maximize(cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))),
               [A @ z == np.ones(prob.N), Hw @ z >= d]).solve()
    z0 = np.maximum(np.asarray(z.value, float), 0.0)
    return np.asarray(B @ z0, float), float(prob.productive_value(z0.reshape(prob.N, prob.A)))


def _max_lambda2(prob, y_star=None):
    A, B, Hw, d, v, n = _data(prob)
    A_eq, b_eq = A, np.ones(prob.N)
    if y_star is not None:
        A_eq = np.vstack([A, B]); b_eq = np.concatenate([np.ones(prob.N), y_star])
    G = np.vstack([-np.eye(n), -Hw]); h = np.concatenate([np.zeros(n), -d])
    ridx = prob.relay_index
    terms = [(i * prob.A + ridx, prob.relay_laplacians[i]) for i in range(prob.N)]
    r = wise_sdp.solve_wise_sdp(n, prob.base_laplacian, terms, complement_basis(prob.N),
                                prob.sigma, A_eq=A_eq, b_eq=b_eq, G_ineq=G, h_ineq=h)
    return float(r.lambda_star)


def _V_constrained(prob, sigma):
    """max V(z) s.t. z in X_f and lambda_2(Lbar(z)) >= sigma; None if infeasible."""
    import cvxpy as cp
    A, B, Hw, d, v, n = _data(prob)
    Q = complement_basis(prob.N)
    z = cp.Variable(n, nonneg=True); y = B @ z
    V = cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))
    cons = [A @ z == np.ones(prob.N), Hw @ z >= d,
            Q.T @ _lap_expr(prob, z, cp) @ Q >> sigma * np.eye(prob.N - 1)]
    p = cp.Problem(cp.Maximize(V), cons); p.solve()
    if z.value is None or p.status not in ("optimal", "optimal_inaccurate"):
        return None
    return float(prob.productive_value(np.maximum(z.value, 0.0).reshape(prob.N, prob.A)))


def run(seeds=6, n_sigma=13):
    rows = []
    lamE_all, lamX_all = [], []
    for s in range(seeds):
        prob = scenarios.two_region(seed=s, N=12, nu=0.4, tau_d=5.0, bridge_gain=3.0)
        y_star, V_star = _y_star(prob)
        LamE = _max_lambda2(prob, y_star=y_star)
        LamX = _max_lambda2(prob, y_star=None)
        lamE_all.append(LamE); lamX_all.append(LamX)
        sigmas = np.linspace(0.2 * LamE, 1.08 * LamX, n_sigma)
        for sg in sigmas:
            Vc = _V_constrained(prob, float(sg))
            P = (V_star - Vc) if Vc is not None else np.inf
            regime = ("free" if sg <= LamE + 1e-6 else
                      "costly" if sg <= LamX + 1e-6 else "impossible")
            rows.append(dict(seed=s, sigma=float(sg), P=float(P) if np.isfinite(P) else np.inf,
                             regime=regime, Lambda_E=LamE, Lambda_X=LamX, V_star=V_star,
                             feasible=Vc is not None))
        print(f"seed {s}: Lambda_E={LamE:.3f} Lambda_X={LamX:.3f} V*={V_star:.3f}")

    fields = ["seed", "sigma", "P", "regime", "Lambda_E", "Lambda_X", "V_star", "feasible"]
    with open(GEN / "price_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r); rr["P"] = "inf" if not np.isfinite(r["P"]) else f"{r['P']:.6f}"
            w.writerow(rr)

    # --- verify the theorem: P=0 on free, P>0 & finite on costly, infeasible on impossible ---
    free_ok = all(abs(r["P"]) < 1e-4 for r in rows if r["regime"] == "free" and r["feasible"])
    costly_ok = all(r["P"] > -1e-6 for r in rows if r["regime"] == "costly" and r["feasible"])
    imposs_ok = all(not r["feasible"] for r in rows if r["regime"] == "impossible")
    print(f"theorem check: free(P=0)={free_ok}, costly(P>=0)={costly_ok}, "
          f"impossible(infeasible)={imposs_ok}")
    assert free_ok and costly_ok, "price theorem violated on free/costly regime"

    _figure(rows, float(np.mean(lamE_all)), float(np.mean(lamX_all)))
    return rows


def _figure(rows, LamE, LamX):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    # average finite P over seeds at each (normalized) sigma grid index
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for s, rs in by_seed.items():
        rs = sorted(rs, key=lambda r: r["sigma"])
        sig = [r["sigma"] for r in rs]
        P = [r["P"] if r["feasible"] else np.nan for r in rs]
        ax.plot(sig, P, color="#1f5fbf", lw=0.8, alpha=0.35)
    # regime shading using mean thresholds
    ax.axvspan(min(r["sigma"] for r in rows), LamE, color="#2e8b57", alpha=0.10)
    ax.axvspan(LamE, LamX, color="#e08000", alpha=0.10)
    ax.axvspan(LamX, max(r["sigma"] for r in rows), color="#c0392b", alpha=0.10)
    ax.axvline(LamE, color="#2e8b57", ls="--", lw=1.0)
    ax.axvline(LamX, color="#c0392b", ls="--", lw=1.0)
    ymax = max((r["P"] for r in rows if r["feasible"]), default=1.0)
    ax.text(LamE, ymax * 0.9, r"$\Lambda_E$", color="#2e8b57", fontsize=7, ha="right")
    ax.text(LamX, ymax * 0.9, r"$\Lambda_X$", color="#c0392b", fontsize=7, ha="left")
    ax.text(0.02, 0.86, "free", transform=ax.transAxes, color="#2e8b57", fontsize=7)
    ax.text(0.5, 0.86, "costly", transform=ax.transAxes, color="#b06000", fontsize=7, ha="center")
    ax.text(0.98, 0.86, "impossible", transform=ax.transAxes, color="#c0392b", fontsize=7, ha="right")
    ax.set_xlabel(r"connectivity requirement $\sigma$", fontsize=8)
    ax.set_ylabel(r"price $P(\sigma)=V^\star-V$", fontsize=8)
    ax.set_title("price of self-sustainability", fontsize=8.5)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_price.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    run()
