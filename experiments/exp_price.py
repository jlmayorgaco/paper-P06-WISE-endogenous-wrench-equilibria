"""E-price (Thm 3, central experiment): the price of self-sustainability P(sigma).

For each seed we compute the productive optimum V* and the two spectral capacities
    Lambda_E = max_{z in E}   lambda_2(Lbar(z))   (over the optimal fiber),
    Lambda_X = max_{z in X_f} lambda_2(Lbar(z))   (over all wrench-feasible z),
and trace the price
    P(sigma) = V* - max{ V(z) : z in X_f, lambda_2(Lbar(z)) >= sigma },
which is 0 (FREE) for sigma <= Lambda_E, positive (COSTLY) for Lambda_E < sigma <= Lambda_X,
and +infinity (IMPOSSIBLE) for sigma > Lambda_X (Theorem 3).

To aggregate across heterogeneous seeds (each with its own Lambda_E, Lambda_X) we sweep a
normalized requirement u in [0,2]: u in [0,1] maps to sigma = u*Lambda_E (free), u in [1,2]
maps to sigma = Lambda_E + (u-1)(Lambda_X - Lambda_E) (costly), so Lambda_E<->1, Lambda_X<->2
for every seed. The figure then shows the median normalized price P/V* with a 25-75% band,
and the distribution of the costly-regime width Lambda_X - Lambda_E.

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


def run(seeds=40, n_u=16):
    u_grid = np.linspace(0.06, 1.94, n_u)      # normalized requirement (E<->1, X<->2)
    rows, curves = [], []
    lamE_all, lamX_all, Vstar_all = [], [], []
    for s in range(seeds):
        prob = scenarios.two_region(seed=s, N=12, nu=0.4, tau_d=5.0, bridge_gain=3.0)
        y_star, V_star = _y_star(prob)
        LamE = _max_lambda2(prob, y_star=y_star)
        LamX = _max_lambda2(prob, y_star=None)
        if not (LamX > LamE + 1e-3 and V_star > 1e-6):
            continue                            # need a nonempty costly band to normalize
        lamE_all.append(LamE); lamX_all.append(LamX); Vstar_all.append(V_star)
        pcurve = []
        for u in u_grid:
            sg = u * LamE if u <= 1.0 else LamE + (u - 1.0) * (LamX - LamE)
            Vc = _V_constrained(prob, float(sg))
            P = (V_star - Vc) if Vc is not None else np.inf
            regime = "free" if u <= 1.0 + 1e-9 else "costly"
            pcurve.append(P / V_star if np.isfinite(P) else np.nan)
            rows.append(dict(seed=s, u=float(u), sigma=float(sg),
                             P=float(P) if np.isfinite(P) else np.inf, P_norm=pcurve[-1],
                             regime=regime, Lambda_E=LamE, Lambda_X=LamX, V_star=V_star,
                             feasible=Vc is not None))
        curves.append(pcurve)
        print(f"seed {s}: Lambda_E={LamE:.3f} Lambda_X={LamX:.3f} V*={V_star:.3f} "
              f"width={LamX - LamE:.3f}")

    with open(GEN / "price_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "u", "sigma", "P", "P_norm", "regime",
                                          "Lambda_E", "Lambda_X", "V_star", "feasible"])
        w.writeheader()
        for r in rows:
            rr = dict(r); rr["P"] = "inf" if not np.isfinite(r["P"]) else f"{r['P']:.6f}"
            w.writerow(rr)

    # --- verify Theorem 3: free (P=0), costly (P>=0), and Lambda_E < Lambda_X ---
    free_ok = all(abs(r["P"]) < 1e-4 for r in rows if r["regime"] == "free" and r["feasible"])
    costly_ok = all(r["P"] > -1e-6 for r in rows if r["regime"] == "costly" and r["feasible"])
    print(f"theorem check ({len(curves)} seeds): free(P=0)={free_ok}, costly(P>=0)={costly_ok}")
    assert free_ok and costly_ok, "price theorem violated"

    _figure(u_grid, np.array(curves), np.array(lamE_all), np.array(lamX_all))
    return rows


def _figure(u_grid, curves, lamE, lamX):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    med = np.nanmedian(curves, axis=0)
    lo = np.nanpercentile(curves, 25, axis=0)
    hi = np.nanpercentile(curves, 75, axis=0)
    n_seeds = curves.shape[0]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.35),
                                   gridspec_kw=dict(width_ratios=[1.9, 1.0], wspace=0.42))

    # --- (a) normalized price curve: median + 25-75% band ---
    axL.axvspan(u_grid[0], 1.0, color="#2e8b57", alpha=0.10)
    axL.axvspan(1.0, 2.0, color="#e08000", alpha=0.10)
    axL.axvline(1.0, color="#2e8b57", ls="--", lw=1.0)
    axL.axvline(2.0, color="#c0392b", ls="--", lw=1.0)
    axL.fill_between(u_grid, lo, hi, color="#1f5fbf", alpha=0.25, lw=0)
    axL.plot(u_grid, med, color="#1f5fbf", lw=1.8)
    ytop = float(np.nanmax(hi)) if np.isfinite(np.nanmax(hi)) else 1.0
    axL.set_ylim(-0.02 * max(ytop, 0.1), 1.08 * max(ytop, 0.1))
    axL.text(0.5, 0.90, "free", transform=axL.get_xaxis_transform(), color="#2e8b57",
             fontsize=7.5, ha="center")
    axL.text(1.5, 0.90, "costly", transform=axL.get_xaxis_transform(), color="#b06000",
             fontsize=7.5, ha="center")
    axL.text(2.02, 0.90, "imposs.", transform=axL.get_xaxis_transform(), color="#c0392b",
             fontsize=7.5, ha="left")
    axL.set_xlabel(r"normalized connectivity requirement $\sigma$", fontsize=8)
    axL.set_ylabel(r"price $P(\sigma)/V^\star$", fontsize=8)
    axL.set_title(r"(a) price of self-sustainability ($%d$ seeds)" % n_seeds, fontsize=8.5)
    axL.tick_params(labelsize=7)
    axL.set_xticks([0, 1, 2]); axL.set_xticklabels(["0", r"$\Lambda_E$", r"$\Lambda_X$"])

    # --- (b) spectral capacities: Lambda_E, Lambda_X and the costly width ---
    parts = axR.violinplot([lamE, lamX], positions=[0, 1], widths=0.7, showmedians=True)
    for pc, col in zip(parts["bodies"], ["#2e8b57", "#c0392b"]):
        pc.set_facecolor(col); pc.set_alpha(0.35)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[key].set_color("#444444"); parts[key].set_linewidth(1.0)
    axR.set_xticks([0, 1]); axR.set_xticklabels([r"$\Lambda_E$", r"$\Lambda_X$"], fontsize=8)
    axR.set_ylabel(r"$\lambda_2$ capacity", fontsize=8)
    axR.set_title("(b) fiber vs. feasible", fontsize=8.5)
    axR.tick_params(labelsize=7)
    axR.text(0.5, 0.03, r"costly width $\Lambda_X\!-\!\Lambda_E$: %.2f" % np.median(lamX - lamE),
             transform=axR.transAxes, fontsize=6.8, ha="center", color="#b06000")

    fig.savefig(FIG / "fig_price.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    run()
