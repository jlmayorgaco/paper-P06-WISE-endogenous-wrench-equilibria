"""E-fiber (paper Example 1): move along the productive neutral space and show that
the productive value V and the served aggregate stay constant while the induced
connectivity lambda_2 varies across the requirement sigma_req.

Writes generated/fiber_sweep.csv and paper/figures/fig_fiber.pdf.
Acceptance: ptp(V) and ptp(aggregate) at machine precision; lambda_2 has a
visible spread.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import baselines, nullspace as ns, scenarios  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def _feasible_alpha_range(x0: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """Largest interval [a_lo, a_hi] with x0 + a d >= 0 (elementwise)."""
    a_lo, a_hi = -np.inf, np.inf
    for xi, di in zip(x0.ravel(), d.ravel()):
        if di > 1e-12:
            a_lo = max(a_lo, -xi / di)
        elif di < -1e-12:
            a_hi = min(a_hi, -xi / di)
    return a_lo, a_hi


def run(seed: int = 3, N: int = 12, nu: float = 0.5, tau_d: float = 3.0):
    prob = scenarios.two_region(seed=seed, N=N, nu=nu, tau_d=tau_d, bridge_gain=3.0)
    res = baselines.wise_primal_dual(prob, max_iters=4000)
    x0 = np.asarray(res.x, dtype=float)

    info = ns.fiber_dimension(prob, x0)
    basis = np.asarray(info["Np_basis"], dtype=float)      # (dim_E, n)
    if basis.shape[0] == 0:
        raise SystemExit("fiber is a single point (dim E = 0); no sweep to show")

    n = x0.size
    # choose the neutral direction with the largest connectivity sensitivity
    def dlam(dvec):
        d = dvec.reshape(x0.shape)
        eps = 1e-4
        return abs(prob.lambda2(x0 + eps * d) - prob.lambda2(x0 - eps * d)) / (2 * eps)
    d = max((b for b in basis), key=dlam).reshape(x0.shape)
    d = d / np.linalg.norm(d)

    a_lo, a_hi = _feasible_alpha_range(x0, d)
    a_lo, a_hi = 0.9 * a_lo, 0.9 * a_hi                    # stay strictly feasible
    alphas = np.linspace(a_lo, a_hi, 41)

    V0 = prob.productive_value(x0)
    y0 = prob.served_capacity(x0)
    rows = []
    for a in alphas:
        x = x0 + a * d
        rows.append(dict(
            alpha=float(a),
            V=float(prob.productive_value(x)),
            aggregate_residual=float(np.linalg.norm(prob.served_capacity(x) - y0)),
            lambda2=float(prob.lambda2(x)),
        ))

    V = np.array([r["V"] for r in rows])
    agg = np.array([r["aggregate_residual"] for r in rows])
    lam = np.array([r["lambda2"] for r in rows])
    # display the requirement in the strict regime (delta is a design margin):
    # some fiber compositions clear sigma_req and others do not.
    sigma = float(0.5 * (lam.min() + lam.max()))

    # ---- acceptance assertions ----
    assert np.ptp(V) <= 1e-6 * (abs(V0) + 1.0), f"V not flat: ptp={np.ptp(V):.2e}"
    assert agg.max() <= 1e-6, f"aggregate drifts: max={agg.max():.2e}"
    assert np.ptp(lam) > 1e-3, f"lambda2 does not vary: ptp={np.ptp(lam):.2e}"

    # ---- CSV ----
    with open(GEN / "fiber_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "V", "aggregate_residual", "lambda2"])
        w.writeheader()
        w.writerows(rows)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    an = (alphas - alphas.min()) / (np.ptp(alphas) + 1e-12)
    axr = ax.twinx()
    l1, = ax.plot(an, V, color="#1f5fbf", lw=1.8, label=r"$V$ (productive value)")
    l2, = axr.plot(an, lam, color="#2e8b57", lw=1.8, label=r"$\lambda_2$")
    axr.axhline(sigma, color="#c03030", ls="--", lw=1.0)
    axr.text(0.02, sigma, r"$\sigma_{\rm req}$", color="#c03030", va="bottom", fontsize=8)
    wise = lam >= sigma
    if wise.any():
        axr.fill_between(an, sigma, lam, where=wise, color="#2e8b57", alpha=0.12)
    ax.set_xlabel(r"position on the fiber $\alpha$")
    ax.set_ylabel(r"$V$", color="#1f5fbf")
    axr.set_ylabel(r"$\lambda_2$", color="#2e8b57")
    ax.set_title("one payoff, varying connectivity", fontsize=9)
    ax.legend(handles=[l1, l2], loc="center left", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_fiber.pdf", metadata={"CreationDate": None})  # reproducible bytes
    plt.close(fig)

    print(f"fiber sweep: dim E={info['dim_E']}, V flat (ptp={np.ptp(V):.1e}), "
          f"lambda2 in [{lam.min():.3f},{lam.max():.3f}], sigma_req={sigma:.3f}")
    return rows


if __name__ == "__main__":
    run()
