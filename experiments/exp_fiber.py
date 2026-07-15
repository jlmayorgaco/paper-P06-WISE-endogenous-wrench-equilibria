"""E-fiber (paper Example 1 / Thm. degeneracy): certify a genuine fiber direction.

We build a base point z_bar on the *exact optimal fiber* (two-stage QP: stage 1 fixes
the productive optimum aggregate y*, stage 2 picks an interior point with B z_bar = y*
exactly), pick a productive-neutral direction d in ker A cap ker B cap ker G_I, and
sweep z(alpha) = z_bar + alpha d over the largest feasible interval. We certify that d is
a genuine fiber direction (||Ad||, ||Bd||, ||G_I d|| at machine precision) and that along
the whole range

    max_alpha |V(z(alpha)) - V*|      < 1e-8   (productive neutrality),
    max_alpha ||B z(alpha) - y*||      < 1e-8   (aggregate invariance),

while lambda_2(Lbar(z(alpha))) crosses sigma_req: optimal safe and unsafe compositions
coexist on one payoff-flat fiber. Not a visually flat curve -- a numerical certificate.

Writes generated/fiber_sweep.csv, generated/fiber_certificate.json,
paper/figures/fig_fiber.pdf.
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

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def _optimal_fiber_base(prob):
    """z_bar on the exact optimal fiber via a two-stage QP.

    Stage 1: y* = B argmax_{z in X_f} V(z).
    Stage 2: z_bar = argmin ||z - z_unif||^2  s.t. A z = 1, B z = y*, H_w z >= d, z >= 0,
    an interior point of the optimal fiber, so B z_bar = y* and V(z_bar) = V* hold to
    solver tolerance and the active-inequality set is minimal (rich neutral space).
    Returns (z_bar, y_star, V_star, solver_status).
    """
    import cvxpy as cp

    n = prob.N * prob.A
    A = ns.mass_matrix(prob); one = np.ones(prob.N)
    B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix(); d = prob.demand().ravel()
    v = prob._value()

    z1 = cp.Variable(n, nonneg=True)
    y1 = B @ z1
    V = cp.sum(cp.multiply(v, y1) - 0.5 * prob.alpha * cp.square(y1))
    p1 = cp.Problem(cp.Maximize(V), [A @ z1 == one, Hw @ z1 >= d])
    p1.solve()
    y_star = np.asarray(B @ z1.value, float)
    V_star = float(prob.productive_value(z1.value.reshape(prob.N, prob.A)))

    z2 = cp.Variable(n, nonneg=True)
    z_unif = np.full(n, 1.0 / prob.A)
    p2 = cp.Problem(cp.Minimize(cp.sum_squares(z2 - z_unif)),
                    [A @ z2 == one, B @ z2 == y_star, Hw @ z2 >= d])
    p2.solve()
    z_bar = np.maximum(np.asarray(z2.value, float), 0.0)
    return z_bar, y_star, V_star, f"{p1.status}/{p2.status}"


def _feasible_alpha_range(z0: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """Largest interval [a_lo, a_hi] with z0 + a d >= 0 elementwise."""
    a_lo, a_hi = -np.inf, np.inf
    for zi, di in zip(z0.ravel(), d.ravel()):
        if di > 1e-12:
            a_lo = max(a_lo, -zi / di)
        elif di < -1e-12:
            a_hi = min(a_hi, -zi / di)
    return a_lo, a_hi


def run(seed: int = 3, N: int = 12, nu: float = 0.5, tau_d: float = 3.0):
    prob = scenarios.two_region(seed=seed, N=N, nu=nu, tau_d=tau_d, bridge_gain=3.0)
    A = ns.mass_matrix(prob)
    B = ns.served_matrix(prob)

    z_bar, y_star, V_star, status = _optimal_fiber_base(prob)
    z0 = z_bar.reshape(prob.N, prob.A)

    # neutral space N_p = ker A cap ker B cap ker G_I at the fiber base
    info = ns.fiber_dimension(prob, z0)
    basis = np.asarray(info["Np_basis"], float)          # (dim_E, n)
    if basis.shape[0] == 0:
        raise SystemExit("fiber is a single point (dim E = 0); no sweep to show")

    # pick the neutral direction with the largest connectivity sensitivity
    def dlam(dvec):
        dd = dvec.reshape(z0.shape)
        e = 1e-4
        return abs(prob.lambda2(z0 + e * dd) - prob.lambda2(z0 - e * dd)) / (2 * e)
    d_vec = max((b for b in basis), key=dlam)
    d_vec = d_vec / np.linalg.norm(d_vec)
    d = d_vec.reshape(z0.shape)

    # ---- certify d is a genuine fiber direction ----
    G_I = ns.active_inequalities(prob, z0)
    res_Ad = float(np.linalg.norm(A @ d_vec))
    res_Bd = float(np.linalg.norm(B @ d_vec))
    res_GId = float(np.linalg.norm(G_I @ d_vec)) if G_I.size else 0.0

    a_lo, a_hi = _feasible_alpha_range(z0, d)
    a_lo, a_hi = 0.9 * a_lo, 0.9 * a_hi                  # stay strictly feasible
    alphas = np.linspace(a_lo, a_hi, 61)

    rows = []
    for a in alphas:
        z = z0 + a * d
        rows.append(dict(
            alpha=float(a),
            V_minus_Vstar=float(prob.productive_value(z) - V_star),
            agg_drift=float(np.linalg.norm(prob.served_capacity(z) - y_star)),
            lambda2=float(prob.lambda2(z)),
        ))

    Vd = np.array([r["V_minus_Vstar"] for r in rows])
    drift = np.array([r["agg_drift"] for r in rows])
    lam = np.array([r["lambda2"] for r in rows])
    # requirement placed strictly inside the fiber's lambda2 range: some optimal
    # compositions clear it, others do not (delta is a design margin in the paper).
    sigma = float(0.5 * (lam.min() + lam.max()))

    cert = {
        "seed": seed, "N": N, "nu": nu, "tau_d": tau_d,
        "dim_E": int(info["dim_E"]), "n_active_ineq": int(info["n_active_ineq"]),
        "solver_status": status,
        "res_Ad": res_Ad, "res_Bd": res_Bd, "res_GId": res_GId,
        "max_abs_V_minus_Vstar": float(np.max(np.abs(Vd))),
        "max_agg_drift": float(np.max(drift)),
        "lambda2_min": float(lam.min()), "lambda2_max": float(lam.max()),
        "sigma_req": sigma,
        "V_star": V_star, "y_star": [float(x) for x in np.atleast_1d(y_star)],
        "fiber_crosses_sigma": bool(lam.min() < sigma < lam.max()),
    }

    # ---- acceptance assertions (the certificate, not a visual) ----
    assert res_Ad < 1e-8 and res_Bd < 1e-8 and res_GId < 1e-8, \
        f"direction not on the fiber: Ad={res_Ad:.1e} Bd={res_Bd:.1e} GId={res_GId:.1e}"
    assert cert["max_abs_V_minus_Vstar"] < 1e-8, \
        f"V not neutral: max|V-V*|={cert['max_abs_V_minus_Vstar']:.2e}"
    assert cert["max_agg_drift"] < 1e-8, \
        f"aggregate drifts: max={cert['max_agg_drift']:.2e}"
    assert cert["fiber_crosses_sigma"], "lambda2 does not cross sigma_req on the fiber"

    (GEN / "fiber_certificate.json").write_text(json.dumps(cert, indent=2), encoding="utf-8")
    with open(GEN / "fiber_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "V_minus_Vstar", "agg_drift", "lambda2"])
        w.writeheader(); w.writerows(rows)

    _figure(alphas, Vd, drift, lam, sigma, cert)

    print(f"fiber: dim E={cert['dim_E']}, |Ad|={res_Ad:.1e} |Bd|={res_Bd:.1e} "
          f"|GId|={res_GId:.1e}; max|V-V*|={cert['max_abs_V_minus_Vstar']:.1e}, "
          f"max drift={cert['max_agg_drift']:.1e}; "
          f"lambda2 in [{lam.min():.3f},{lam.max():.3f}], sigma={sigma:.3f}")
    return cert


def _figure(alphas, Vd, drift, lam, sigma, cert):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    an = (alphas - alphas.min()) / (np.ptp(alphas) + 1e-12)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 1.95))

    a.plot(an, Vd, color="#1f5fbf", lw=1.8)
    a.axhline(0, color="#888", lw=0.6, ls=":")
    a.set_title(r"(a) $V(z(\alpha))-V^\star$", fontsize=8.5)
    a.set_ylabel("productive gap", fontsize=8)
    a.set_ylim(-1e-8, 1e-8)
    a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    b.plot(an, drift, color="#c0392b", lw=1.8)
    b.set_title(r"(b) $\|Bz(\alpha)-y^\star\|$", fontsize=8.5)
    b.set_ylabel("aggregate drift", fontsize=8)
    b.set_ylim(0, 1e-8)
    b.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    c.plot(an, lam, color="#2e8b57", lw=1.8)
    c.axhline(sigma, color="#c03030", ls="--", lw=1.0)
    c.text(0.02, sigma, r"$\sigma_{\rm req}$", color="#c03030", va="bottom", fontsize=7)
    c.fill_between(an, sigma, lam, where=(lam >= sigma), color="#2e8b57", alpha=0.12)
    c.set_title(r"(c) $\lambda_2(\bar L(z(\alpha)))$", fontsize=8.5)
    c.set_ylabel(r"$\lambda_2$", fontsize=8)

    for ax in (a, b, c):
        ax.set_xlabel(r"position on fiber $\alpha$", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_fiber.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    run()
