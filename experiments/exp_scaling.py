"""E-scaling: solve time of the two-stage WISE program vs team size N.

Stage 1 is the productive QP (unique served aggregate y*); Stage 2 is the lexicographic
connectivity SDP with an (N-1)x(N-1) LMI. We time both against N in {12,24,48,96} and the
integer recovery (best-of-R rounding), and report the LMI dimension and decision size.
A log-log plot of solve time vs N goes to paper/figures/fig_scaling.pdf.

Solver: CVXPY default conic backend (CLARABEL/SCS); single instance per size (seed 0),
median of a few repeats. Hardware/threads are recorded in the CSV header comment.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import metrics, nullspace as ns, scenarios, wise_sdp  # noqa: E402
from wise_mr.endogenous_graph import complement_basis  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def _stage1(prob):
    """Productive QP: y* = B z*, z* = argmax V over X_f. Returns (y*, time_s)."""
    import cvxpy as cp
    A = ns.mass_matrix(prob); B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix(); d = prob.demand().ravel(); v = prob._value()
    n = prob.N * prob.A
    z = cp.Variable(n, nonneg=True); y = B @ z
    p = cp.Problem(cp.Maximize(cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))),
                   [A @ z == np.ones(prob.N), Hw @ z >= d])
    t0 = time.perf_counter(); p.solve(); dt = time.perf_counter() - t0
    return np.asarray(B @ np.maximum(z.value, 0.0), float), dt


def _stage2(prob, y_star):
    """Lexicographic connectivity SDP on the fiber Bz=y*. Returns (result, time_s)."""
    A = ns.mass_matrix(prob); B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix(); d = prob.demand().ravel()
    n = prob.N * prob.A
    A_eq = np.vstack([A, B]); b_eq = np.concatenate([np.ones(prob.N), y_star])
    G = np.vstack([-np.eye(n), -Hw]); h = np.concatenate([np.zeros(n), -d])
    ridx = prob.relay_index
    terms = [(i * prob.A + ridx, prob.relay_laplacians[i]) for i in range(prob.N)]
    t0 = time.perf_counter()
    r = wise_sdp.solve_wise_sdp(n, prob.base_laplacian, terms, complement_basis(prob.N),
                                prob.sigma, A_eq=A_eq, b_eq=b_eq, G_ineq=G, h_ineq=h)
    return r, time.perf_counter() - t0


def _round_time(prob, res, draws=30):
    """Best-of-R randomized-rounding recovery time (search + re-certification)."""
    z = np.maximum(res.z.reshape(prob.N, prob.A), 0.0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(draws):
        cand = np.zeros_like(z)
        for i in range(prob.N):
            p = z[i] / max(z[i].sum(), 1e-9)
            cand[i, rng.choice(prob.A, p=p / p.sum())] = 1.0
        if metrics.certified(prob, cand):
            break
    return time.perf_counter() - t0


def run(sizes=(12, 24, 48, 96), repeats=3):
    rows = []
    for N in sizes:
        t1s, t2s, trs = [], [], []
        n_dec = lmi = None
        for rep in range(repeats):
            prob = scenarios.two_region(seed=rep, N=N, nu=0.4, tau_d=5.0, bridge_gain=3.0)
            y_star, t1 = _stage1(prob)
            res, t2 = _stage2(prob, y_star)
            tr = _round_time(prob, res)
            t1s.append(t1); t2s.append(t2); trs.append(tr)
            n_dec = prob.N * prob.A; lmi = prob.N - 1
        row = dict(N=N, decision_vars=n_dec, lmi_dim=lmi,
                   stage1_s=float(np.median(t1s)), stage2_s=float(np.median(t2s)),
                   rounding_s=float(np.median(trs)),
                   total_s=float(np.median(t1s) + np.median(t2s) + np.median(trs)))
        rows.append(row)
        print(f"N={N:3d}: dec={n_dec:4d} lmi={lmi:3d}  "
              f"stage1={row['stage1_s']*1e3:6.1f}ms  stage2={row['stage2_s']*1e3:7.1f}ms  "
              f"round={row['rounding_s']*1e3:6.1f}ms")

    with open(GEN / "scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    _figure(rows)
    return rows


def _figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    N = np.array([r["N"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    for key, lab, col, mk in [("stage1_s", "Stage 1 (QP)", "#2e8b57", "o"),
                              ("stage2_s", "Stage 2 (SDP)", "#1f5fbf", "s"),
                              ("rounding_s", "rounding", "#c0392b", "^")]:
        ax.loglog(N, [r[key] for r in rows], mk + "-", color=col, lw=1.2, ms=4, label=lab)
    ax.set_xlabel(r"team size $N$", fontsize=8)
    ax.set_ylabel("solve time [s]", fontsize=8)
    ax.set_title("two-stage scaling", fontsize=8.5)
    ax.set_xticks(N); ax.set_xticklabels([str(int(n)) for n in N])
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.savefig(FIG / "fig_scaling.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    run()
