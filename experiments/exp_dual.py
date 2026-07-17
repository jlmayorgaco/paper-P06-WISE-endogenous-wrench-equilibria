"""E-dual (Thm 3): the spectral multiplier pi* IS the marginal price dP/dsigma.

Theorem 3 claims pi* in dP(sigma). We verify it numerically. For the connectivity-
constrained productive program
    V_c(sigma) = max { V(z) : z in X_f, Q^T Lbar(z) Q >= sigma I },
the PSD dual Z* of the LMI gives the multiplier pi* = tr(Z*) (paper's Y = pi Z,
pi = tr Y), and by convex sensitivity dP/dsigma = -dV_c/dsigma = pi*. We compare pi* to
the central finite difference
    P'(sigma) ~ (P(sigma+h) - P(sigma-h)) / (2h)
over many seeds and several sigma in the costly regime, and report the correlation and
median relative error. Writes generated/dual_check.csv and paper/figures/fig_dual.pdf.
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


def _V_and_dual(prob, sigma):
    """Return (V_c, pi*). pi* = tr(Y*) where Y* is the UNNORMALIZED PSD dual matrix of the
    connectivity LMI (cvxpy returns the raw KKT multiplier). Paper: Y = pi Z, tr Z = 1, so
    pi* = tr(Y*) and Z* = Y*/pi* is the normalized modal direction. All in physical sigma."""
    import cvxpy as cp
    A, B, Hw, d, v, n = _data(prob)
    Q = complement_basis(prob.N)
    z = cp.Variable(n, nonneg=True); y = B @ z
    V = cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))
    lmi = Q.T @ _lap_expr(prob, z, cp) @ Q >> sigma * np.eye(prob.N - 1)
    p = cp.Problem(cp.Maximize(V), [A @ z == np.ones(prob.N), Hw @ z >= d, lmi])
    p.solve()
    if z.value is None or p.status not in ("optimal", "optimal_inaccurate"):
        return None, None
    Vc = float(prob.productive_value(np.maximum(z.value, 0.0).reshape(prob.N, prob.A)))
    Y = lmi.dual_value                                   # raw (unnormalized) dual matrix
    pi = float(np.trace(Y)) if Y is not None else float("nan")
    return Vc, pi


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


def run(seeds=30, n_sigma=3, h=0.05):
    rows = []
    for s in range(seeds):
        prob = scenarios.two_region(seed=s, N=12, nu=0.4, tau_d=5.0, bridge_gain=3.0)
        y_star, V_star = _y_star(prob)
        LamE = _max_lambda2(prob, y_star=y_star)
        LamX = _max_lambda2(prob, y_star=None)
        if LamX - LamE < 4 * h:
            continue                                    # need room for a central difference
        for frac in np.linspace(0.25, 0.75, n_sigma):   # sigma inside the costly regime
            sg = LamE + frac * (LamX - LamE)
            Vc, pi = _V_and_dual(prob, sg)
            Vp, _ = _V_and_dual(prob, sg + h)
            Vm, _ = _V_and_dual(prob, sg - h)
            if None in (Vc, Vp, Vm) or pi is None or not np.isfinite(pi):
                continue
            dP_fd = ((V_star - Vp) - (V_star - Vm)) / (2 * h)   # central difference of P
            rows.append(dict(seed=s, sigma=float(sg), pi_star=pi, dP_fd=float(dP_fd)))
        print(f"seed {s}: LamE={LamE:.2f} LamX={LamX:.2f} "
              f"pts={sum(r['seed'] == s for r in rows)}")

    pis = np.array([r["pi_star"] for r in rows])
    fds = np.array([r["dP_fd"] for r in rows])
    corr = float(np.corrcoef(pis, fds)[0, 1]) if len(rows) > 2 else float("nan")
    rel = np.abs(pis - fds) / (np.abs(fds) + 1e-9)
    med_rel = float(np.median(rel))
    print(f"\n{len(rows)} points: corr(pi*, dP/dsigma)={corr:.4f}, "
          f"median rel.err={med_rel:.3f}")

    with open(GEN / "dual_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "sigma", "pi_star", "dP_fd"])
        w.writeheader(); w.writerows(rows)
    _figure(pis, fds, corr, med_rel)
    return corr, med_rel


def _figure(pis, fds, corr, med_rel):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    lim = max(float(np.max(pis)), float(np.max(fds))) * 1.08
    ax.plot([0, lim], [0, lim], color="#999", ls="--", lw=0.9, zorder=1)
    ax.scatter(fds, pis, s=14, c="#1f5fbf", alpha=0.6, edgecolors="none", zorder=2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel(r"finite difference $\Delta P/\Delta\sigma$", fontsize=8)
    ax.set_ylabel(r"spectral dual $\pi^\star=\mathrm{tr}\,Y^\star$", fontsize=8)
    ax.set_title(r"$\pi^\star=\partial P$ (r$=%.3f$)" % corr, fontsize=8.5)
    ax.tick_params(labelsize=7)
    fig.savefig(FIG / "fig_dual.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    run()
