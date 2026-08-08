"""E-regime: the free / costly / impossible phase diagram (paper Thm. price).

NOT the same experiment as exp_phase.py. That script compares the relaxed SDP verdict
against the *decentralized heuristic flow's* certified success, which conflates the
relaxation-to-integer gap with the heuristic's own failures. This one answers the
structural question directly:

    WHEN DOES ZERO-COST CONNECTIVITY EXIST AT ALL?

Per cell we solve two SDPs over the same wrench-feasible polytope:

    Lambda_E = max { lambda_2(Lbar(z)) : z in X_f, B z = y* }   (on the optimal fiber)
    Lambda_X = max { lambda_2(Lbar(z)) : z in X_f }             (ignoring productivity)

and classify by Thm. (price):

    free        sigma_req <= Lambda_E          connectivity costs nothing
    costly      Lambda_E < sigma_req <= Lambda_X   connectivity costs productive value
    impossible  sigma_req > Lambda_X           no assignment achieves it

Axes are two independent mechanisms:
    nu     -- fraction of long-range robots (scarcity of the type that can bridge)
    tau_d  -- wrench demand tightness (how much of the team the loads consume)

We also record the free spectral reserve R_free = Lambda_E - lambda_2 at the productive
optimum, and the costly width Lambda_X - Lambda_E.

Writes generated/regime_grid.csv, generated/regime_summary.json,
paper/figures/fig_regime.pdf.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from wise_mr import nullspace as ns, scenarios  # noqa: E402
from wise_mr.endogenous_graph import complement_basis  # noqa: E402
from exp_gamma import lam2, optimal_fiber_base, relay_coords  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

NUS = [0.17, 0.25, 0.33, 0.42, 0.50, 0.67]
TAUS = [1.0, 1.6, 2.2, 2.8, 3.4, 4.0]
SEEDS = [0, 1, 2, 3]
TOL = 1e-6


def _max_lambda2(prob, *, y_star=None):
    """max lambda_2(Lbar(z)) over X_f, optionally restricted to the optimal fiber."""
    import cvxpy as cp

    n = prob.N * prob.A
    A = ns.mass_matrix(prob)
    B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix()
    dvec = prob.demand().ravel()
    Q = complement_basis(prob.N)

    z = cp.Variable(n, nonneg=True)
    t = cp.Variable()
    Lz = cp.Constant(np.asarray(prob.base_laplacian, dtype=float))
    for i, j in enumerate(relay_coords(prob)):
        Lz = Lz + z[int(j)] * np.asarray(prob.relay_laplacians[i], dtype=float)
    cons = [A @ z == np.ones(prob.N), Hw @ z >= dvec,
            Q.T @ Lz @ Q - t * np.eye(prob.N - 1) >> 0]
    if y_star is not None:
        cons.append(B @ z == y_star)
    p = cp.Problem(cp.Maximize(t), cons)
    p.solve(solver=cp.CLARABEL)
    if t.value is None:
        return float("nan"), p.status
    return float(t.value), p.status


def classify(sigma, lam_E, lam_X):
    if not np.isfinite(lam_E) or not np.isfinite(lam_X):
        return "infeasible_instance"
    if sigma <= lam_E + TOL:
        return "free"
    if sigma <= lam_X + TOL:
        return "costly"
    return "impossible"


def main() -> None:
    rows = []
    for it, tau in enumerate(TAUS):
        for iv, nu in enumerate(NUS):
            for sd in SEEDS:
                try:
                    prob = scenarios.two_region(seed=sd, N=12, nu=nu, tau_d=tau)
                    sigma = float(prob.sigma)
                    zbar, y_star, V_star, _ = optimal_fiber_base(prob)
                    lam_E, st_E = _max_lambda2(prob, y_star=y_star)
                    lam_X, st_X = _max_lambda2(prob)
                    lam_prod = lam2(prob, zbar)
                    reg = classify(sigma, lam_E, lam_X)
                except Exception as exc:                      # noqa: BLE001
                    rows.append(dict(nu=nu, tau_d=tau, seed=sd, regime="wrench_infeasible",
                                     sigma_req=float("nan"), lambda_E=float("nan"),
                                     lambda_X=float("nan"), lambda_prod=float("nan"),
                                     R_free=float("nan"), costly_width=float("nan"),
                                     note=f"{type(exc).__name__}"))
                    continue
                rows.append(dict(
                    nu=nu, tau_d=tau, seed=sd, regime=reg, sigma_req=sigma,
                    lambda_E=lam_E, lambda_X=lam_X, lambda_prod=lam_prod,
                    R_free=lam_E - lam_prod, costly_width=lam_X - lam_E,
                    note=f"{st_E}/{st_X}"))
            done = [r for r in rows if r["nu"] == nu and r["tau_d"] == tau]
            tally = {k: sum(1 for r in done if r["regime"] == k)
                     for k in ("free", "costly", "impossible", "wrench_infeasible")}
            print(f"nu={nu:.2f} tau_d={tau:.1f}: {tally}")

    with (GEN / "regime_grid.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r["regime"] in ("free", "costly", "impossible")]
    counts = {k: sum(1 for r in rows if r["regime"] == k)
              for k in ("free", "costly", "impossible", "wrench_infeasible")}
    summary = {
        "grid": {"nu": NUS, "tau_d": TAUS, "seeds": SEEDS},
        "n_cells": len(rows),
        "counts": counts,
        "R_free_median": float(np.nanmedian([r["R_free"] for r in ok])) if ok else None,
        "costly_width_median": (float(np.nanmedian([r["costly_width"] for r in ok]))
                                if ok else None),
        "classification": "free: sigma<=Lambda_E; costly: Lambda_E<sigma<=Lambda_X; "
                          "impossible: sigma>Lambda_X",
        "tol": TOL,
    }
    (GEN / "regime_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{counts}")
    print(f"median R_free = {summary['R_free_median']}, "
          f"median costly width = {summary['costly_width_median']}")
    _figure(rows)


def _figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError:
        print("matplotlib unavailable; skipping figure")
        return
    code = {"free": 0, "costly": 1, "impossible": 2, "wrench_infeasible": 3}
    M = np.full((len(TAUS), len(NUS)), np.nan)
    for it, tau in enumerate(TAUS):
        for iv, nu in enumerate(NUS):
            cell = [r["regime"] for r in rows if r["nu"] == nu and r["tau_d"] == tau]
            if cell:                                     # majority regime of the seeds
                M[it, iv] = code[max(set(cell), key=cell.count)]
    cmap = ListedColormap(["#1e7a46", "#e08b1e", "#c0392b", "#8a8a8a"])
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N))
    ax.set_xticks(range(len(NUS))); ax.set_xticklabels([f"{v:.2f}" for v in NUS], fontsize=6)
    ax.set_yticks(range(len(TAUS))); ax.set_yticklabels([f"{v:.1f}" for v in TAUS], fontsize=6)
    ax.set_xlabel(r"long-range fraction $\nu$", fontsize=8)
    ax.set_ylabel(r"wrench demand $\tau_d$", fontsize=8)
    ax.set_title("free / costly / impossible", fontsize=8)
    for lbl, col in (("free", "#1e7a46"), ("costly", "#e08b1e"),
                     ("impossible", "#c0392b"), ("infeasible", "#8a8a8a")):
        ax.plot([], [], "s", color=col, label=lbl, ms=5)
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_regime.pdf", bbox_inches="tight")
    print(f"wrote {FIG / 'fig_regime.pdf'}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# Post-processing: the actual phase diagram, in (nu, sigma) rather than (nu, tau_d)
# --------------------------------------------------------------------------- #

def phase_map(csv_path=None, n_sigma: int = 40):
    """Classify (nu, sigma) cells from the ALREADY-COMPUTED Lambda_E, Lambda_X.

    The (nu, tau_d) sweep came out uniformly 'free' because sigma_req is fixed at
    sigma_dyn = 0.25 while Lambda_E in [0.56, 3.30]: the grid never crosses a
    boundary, and tau_d turns out not to move Lambda_E at all (the wrench facets
    never bind in this range). The parameter that actually separates the regimes is
    the requirement sigma itself, and the boundaries are exactly Lambda_E and
    Lambda_X -- both already stored. No further SDP is needed.
    """
    import csv as _csv
    p = Path(csv_path) if csv_path else GEN / "regime_grid.csv"
    rows = [r for r in _csv.DictReader(p.open())
            if r["regime"] in ("free", "costly", "impossible")]
    nus = sorted({float(r["nu"]) for r in rows})
    lamX_max = max(float(r["lambda_X"]) for r in rows)
    sigmas = np.linspace(0.05, 1.05 * lamX_max, n_sigma)

    M = np.zeros((n_sigma, len(nus)))
    out = []
    for j, nu in enumerate(nus):
        cell = [r for r in rows if float(r["nu"]) == nu]
        for i, sg in enumerate(sigmas):
            tally = [classify(sg, float(r["lambda_E"]), float(r["lambda_X"]))
                     for r in cell]
            reg = max(set(tally), key=tally.count)
            M[i, j] = {"free": 0, "costly": 1, "impossible": 2}[reg]
            out.append(dict(nu=nu, sigma=float(sg), regime=reg,
                            frac_free=tally.count("free") / len(tally),
                            frac_costly=tally.count("costly") / len(tally),
                            frac_impossible=tally.count("impossible") / len(tally)))
    with (GEN / "phase_map.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    counts = {k: sum(1 for r in out if r["regime"] == k)
              for k in ("free", "costly", "impossible")}
    print(f"phase map over (nu, sigma): {counts}")
    for nu in nus:
        c = [r for r in rows if float(r["nu"]) == nu]
        print(f"  nu={nu:.2f}  Lambda_E={np.mean([float(r['lambda_E']) for r in c]):.3f}"
              f"  Lambda_X={np.mean([float(r['lambda_X']) for r in c]):.3f}"
              f"  costly width={np.mean([float(r['costly_width']) for r in c]):.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError:
        return out
    cmap = ListedColormap(["#1e7a46", "#e08b1e", "#c0392b"])
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N),
              extent=[-0.5, len(nus) - 0.5, sigmas[0], sigmas[-1]])
    lamE = [np.mean([float(r["lambda_E"]) for r in rows if float(r["nu"]) == nu])
            for nu in nus]
    lamX = [np.mean([float(r["lambda_X"]) for r in rows if float(r["nu"]) == nu])
            for nu in nus]
    ax.plot(range(len(nus)), lamE, "k-", lw=1.2, label=r"$\Lambda_{\mathcal{E}}$")
    ax.plot(range(len(nus)), lamX, "k--", lw=1.2, label=r"$\Lambda_{X}$")
    ax.set_xticks(range(len(nus))); ax.set_xticklabels([f"{v:.2f}" for v in nus], fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_xlabel(r"long-range fraction $\nu$", fontsize=8)
    ax.set_ylabel(r"requirement $\sigma$", fontsize=8)
    ax.set_title("free / costly / impossible", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_regime.pdf")
    print(f"wrote {FIG / 'fig_regime.pdf'}")
    return out
