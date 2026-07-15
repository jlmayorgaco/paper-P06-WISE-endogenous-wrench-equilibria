"""Regenerate fig5_threshold.pdf (previously an orphan): closed-loop self-defeat.
Drives a relay robot to the bridge (self-sustaining, B) vs. leaving it withdrawn
(self-defeating, A); the induced lambda_2(t) and the coupled 2x2 estimation error
e(t) are integrated from the dynamics module. Embeds TrueType (not Type 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wise_mr import dynamics as dyn  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_ACC, C_BAD, C_GOOD = "#2e4057", "#d1495b", "#66a182"

mF, c, th = 1.0, 1.0, 0.5
SIGMA = th**2 / (c * mF)                          # sigma_dyn
N, dt, T = 6, 0.02, 350
POS0 = np.array([[2, 5], [2.4, 5.6], [1.8, 4.4], [8, 5], [8.4, 4.6], [2.0, 5.0]], float)
RNG = np.array([2, 2, 2, 2, 2, 6.0])             # robot 5 is long-range
BRIDGE = np.array([5.0, 5.0])


def run(relay_active: bool):
    pos = POS0.copy()
    lam = []
    for _ in range(T):
        if relay_active:
            d = BRIDGE - pos[5]
            pos[5] = pos[5] + dt * 2.0 * d / max(np.linalg.norm(d), 1e-6) * min(np.linalg.norm(d), 1.0)
        mask = np.zeros(N, bool); mask[5] = relay_active
        lam.append(dyn.live_lambda2(pos, RNG, mask))
    return np.array(lam)


def main() -> None:
    lamB, lamA = run(True), run(False)
    eB = dyn.rollout_error(lamB, mF, c, th, dt=dt)
    eA = dyn.rollout_error(lamA, mF, c, th, dt=dt)
    tt, te = np.arange(T) * dt, np.arange(T + 1) * dt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax1.plot(tt, lamB, color=C_GOOD, lw=1.7, label="B: relay engaged")
    ax1.plot(tt, lamA, color=C_BAD, lw=1.7, ls=(0, (4, 2)), label="A: relay withdrawn")
    ax1.axhline(SIGMA, color=C_ACC, lw=0.9, ls=":", label=r"$\sigma_{\mathrm{dyn}}$")
    ax1.set_xlabel("time [s]"); ax1.set_ylabel(r"connectivity $\lambda_2(\bar L(q(t)))$")
    ax1.set_title("(a) induced connectivity", fontsize=8.5)
    ax1.legend(loc="center right", framealpha=0.9)
    ax2.semilogy(te, np.linalg.norm(eB, axis=1), color=C_GOOD, lw=1.7, label="B: self-sustaining")
    ax2.semilogy(te, np.linalg.norm(eA, axis=1), color=C_BAD, lw=1.7, ls=(0, (4, 2)),
                 label="A: self-defeating")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel(r"estimation error $\|e(t)\|$")
    ax2.set_title(r"(b) canonical block $c\,m_F\lambda_2\!>\!\vartheta^2$", fontsize=8.5)
    ax2.legend(loc="center right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(FIG / "fig5_threshold.pdf"); plt.close(fig)
    print("wrote fig5_threshold.pdf; lamB max %.2f lamA max %.2f" % (lamB.max(), lamA.max()))


if __name__ == "__main__":
    main()
