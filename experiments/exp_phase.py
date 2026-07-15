"""E-phase (paper Thm. trichotomy): regime phase diagram with the SDP boundary.

For each (nu, tau_d) cell and seed we
  1. solve the productive optimum y* = argmax phi(Bx) over X_f (QP);
  2. solve the WISE selection SDP for Lambda_E = max_E lambda_2 (Thm. 2);
  3. classify the cell:  wrench-infeasible / WISE-infeasible / WISE-feasible;
  4. run the decentralized flow and record certified success (empirical);
and compare the SDP verdict (Lambda_E >= sigma_req) to the empirical outcome
(TP/TN/FP/FN).  Writes generated/phase_grid.csv, generated/phase_confusion.csv and
paper/figures/fig_phase_sdp.pdf (rate heat-map + the SDP boundary Lambda_E=sigma_req).
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


def _optimal_aggregate(prob, A_eq, b_eq, Hw, d):
    """Stage 1: y* = argmax_x phi(Bx) s.t. Ax=b, Hw x >= d, x>=0  (concave QP)."""
    import cvxpy as cp

    n = prob.N * prob.A
    B = ns.served_matrix(prob)
    v = prob._value()
    x = cp.Variable(n, nonneg=True)
    y = B @ x
    obj = cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))
    cons = [A_eq @ x == b_eq, Hw @ x >= d]
    p = cp.Problem(cp.Maximize(obj), cons)
    p.solve()
    if p.status not in ("optimal", "optimal_inaccurate") or x.value is None:
        return None, None
    return (B @ x.value), str(p.status)


def _sdp_inputs(prob, y_star):
    """Assemble stage-2 SDP data over the fiber E = {x in X_f : Bx = y*}."""
    n = prob.N * prob.A
    A = ns.mass_matrix(prob)                      # (N, n)
    B = ns.served_matrix(prob)                    # (M, n)
    A_eq = np.vstack([A, B])
    b_eq = np.concatenate([np.ones(prob.N), np.asarray(y_star)])
    Hw = prob.wrench_matrix()                     # (M P, n)
    d = prob.demand().ravel()
    # inequalities: -x <= 0  and  -Hw x <= -d
    G = np.vstack([-np.eye(n), -Hw])
    h = np.concatenate([np.zeros(n), -d])
    L0 = prob.base_laplacian
    ridx = prob.relay_index
    lap_terms = [(i * prob.A + ridx, prob.relay_laplacians[i]) for i in range(prob.N)]
    Q = complement_basis(prob.N)
    return n, L0, lap_terms, Q, A_eq, b_eq, G, h, Hw, d


def run(grid=6, seeds=4, out_csv=True):
    # nu spans from 0 (no long-range robot -> cannot relay) to mostly-long-range;
    # tau_d spans from easy to wrench-limited, so the grid crosses all three regimes.
    nus = np.linspace(0.0, 0.9, grid)
    taus = np.linspace(1.0, 12.0, grid)
    cells, confusion = [], dict(TP=0, TN=0, FP=0, FN=0)

    for it, td in enumerate(taus):
        for iv, nu in enumerate(nus):
            sdp_pass, emp_pass, wrench_ok, lamE_vals = [], [], [], []
            for s in range(seeds):
                prob = scenarios.two_region(seed=s, N=12, nu=float(nu),
                                            tau_d=float(td), bridge_gain=3.0)
                A = ns.mass_matrix(prob)
                b = np.ones(prob.N)
                Hw = prob.wrench_matrix()
                d = prob.demand().ravel()
                y_star, st = _optimal_aggregate(prob, A, b, Hw, d)
                if y_star is None:                       # wrench-infeasible cell
                    wrench_ok.append(False)
                    sdp_pass.append(False)
                else:
                    wrench_ok.append(True)
                    n, L0, terms, Q, Aeq, beq, G, h, _, _ = _sdp_inputs(prob, y_star)
                    res = wise_sdp.solve_wise_sdp(n, L0, terms, Q, prob.sigma,
                                                  A_eq=Aeq, b_eq=beq, G_ineq=G, h_ineq=h)
                    lamE_vals.append(res.lambda_star)
                    sdp_pass.append(bool(res.wise_exists))
                # empirical decentralized flow
                r = baselines.wise_primal_dual(prob, max_iters=3000)
                feas = bool(np.all(prob.wrench_price(r.x) <= 1e-3) and r.x.min() >= -1e-6)
                emp = feas and (prob.lambda2(r.x) >= prob.sigma)
                emp_pass.append(bool(emp))

            sdp_rate = float(np.mean(sdp_pass))
            emp_rate = float(np.mean(emp_pass))
            lamE = float(np.mean(lamE_vals)) if lamE_vals else float("nan")
            for sp, ep in zip(sdp_pass, emp_pass):
                key = ("TP" if ep else "FP") if sp else ("FN" if ep else "TN")
                confusion[key] += 1
            cells.append(dict(nu=float(nu), tau_d=float(td), iv=iv, it=it,
                              sdp_feasible_rate=sdp_rate, empirical_rate=emp_rate,
                              lambda_E=lamE, wrench_feasible_rate=float(np.mean(wrench_ok)),
                              sigma_req=float(prob.sigma)))

    if out_csv:
        with open(GEN / "phase_grid.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
            w.writeheader(); w.writerows(cells)
        with open(GEN / "phase_confusion.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["outcome", "count"])
            for k, vv in confusion.items():
                w.writerow([k, vv])

    _figure(nus, taus, cells, grid)
    tot = sum(confusion.values())
    acc = (confusion["TP"] + confusion["TN"]) / tot if tot else float("nan")
    print(f"phase: {grid}x{grid} cells, {seeds} seeds; confusion={confusion}; "
          f"agreement={acc:.2%}")
    return cells, confusion


def _figure(nus, taus, cells, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emp = np.full((grid, grid), np.nan)
    sdp = np.full((grid, grid), np.nan)
    for c in cells:
        emp[c["it"], c["iv"]] = c["empirical_rate"]
        sdp[c["it"], c["iv"]] = c["sdp_feasible_rate"]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    im = ax.imshow(emp, origin="lower", aspect="auto", cmap="YlGn", vmin=0, vmax=1,
                   extent=[nus[0], nus[-1], taus[0], taus[-1]])
    # SDP boundary: contour of sdp-feasible rate = 0.5 (Lambda_E = sigma_req frontier)
    NU, TAU = np.meshgrid(nus, taus)
    try:
        cs = ax.contour(NU, TAU, sdp, levels=[0.5], colors="#20409a",
                        linewidths=1.6, linestyles="-")
        ax.clabel(cs, fmt=r"$\Lambda_\mathcal{E}=\sigma_{\rm req}$", fontsize=6)
    except Exception:
        pass
    ax.set_xlabel(r"long-range fraction $\nu$")
    ax.set_ylabel(r"torque demand $\tau_d$")
    ax.set_title("certified rate + SDP boundary", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("empirical certified rate", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_phase_sdp.pdf")
    plt.close(fig)


if __name__ == "__main__":
    run()
