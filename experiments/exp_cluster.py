"""E-cluster (paper Sec. V): spatial robot clustering and the network effect.

Robots live in the x-y plane in two clusters separated by a gap wider than the
short-range radius, so the induced graph has two components (lambda_2 = 0) unless a
long-range robot occupies the gap and bridges them. We show (a) the fragmented
configuration, (b) the bridged configuration, and (c) algebraic connectivity as the
relay sweeps across the gap---the network effect that the WISE composition controls.

Writes generated/cluster_sweep.csv and paper/figures/fig_cluster.pdf.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import dynamics as dyn  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SHORT_RANGE = 2.5
LONG_RANGE = 3.6          # bridges only when the relay sits near the gap centre
SIGMA = 0.20


def _scene(seed=1):
    rng = np.random.default_rng(seed)
    left = np.array([2.0, 5.0]) + 0.9 * rng.standard_normal((5, 2))
    right = np.array([8.0, 5.0]) + 0.9 * rng.standard_normal((5, 2))
    pos = np.vstack([left, right])                       # 10 short-range robots
    ranges = np.full(11, SHORT_RANGE)
    ranges[-1] = LONG_RANGE                              # the 11th is long-range (relay)
    relay_mask = np.zeros(11, bool); relay_mask[-1] = True
    return pos, ranges, relay_mask


def _lambda2(pos, relay_xy, ranges, relay_mask):
    P = np.vstack([pos, relay_xy])
    return dyn.live_lambda2(P, ranges, relay_mask,
                            base_range=SHORT_RANGE, bridge_gain=3.0)


def run():
    pos, ranges, relay_mask = _scene()
    xs = np.linspace(2.0, 8.0, 61)
    lam = [float(_lambda2(pos, np.array([x, 5.0]), ranges, relay_mask)) for x in xs]

    with open(GEN / "cluster_sweep.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["relay_x", "lambda2"])
        for x, l in zip(xs, lam):
            w.writerow([f"{x:.4f}", f"{l:.6f}"])

    _figure(pos, ranges, relay_mask, xs, np.array(lam))
    peak = xs[int(np.argmax(lam))]
    print(f"cluster: lambda2 in [{min(lam):.3f},{max(lam):.3f}], peak at relay_x={peak:.2f}, "
          f"sigma={SIGMA}")
    return xs, lam


def _draw(ax, pos, relay_xy, ranges, relay_mask, title, active=True):
    P = np.vstack([pos, relay_xy])
    mask = relay_mask if active else np.zeros_like(relay_mask)   # inactive relay: no bridge
    lam = dyn.live_lambda2(P, ranges, mask, base_range=SHORT_RANGE, bridge_gain=3.0)
    relay_mask = mask
    # edges
    N = P.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            dij = np.linalg.norm(P[i] - P[j])
            if dij <= SHORT_RANGE:
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="#c0c0c0", lw=0.6, zorder=1)
    for i in np.where(relay_mask)[0]:
        for j in range(N):
            if j != i and np.linalg.norm(P[i] - P[j]) <= ranges[i]:
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="#2e8b57", lw=1.2, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=32, c="#3a6ea5", edgecolors="k", linewidths=0.4, zorder=3)
    ax.scatter([relay_xy[0]], [relay_xy[1]], s=70, marker="D", c="#2e8b57",
               edgecolors="k", linewidths=0.4, zorder=3)
    col = "#2e8b57" if lam >= SIGMA else "#c0392b"
    ax.set_title(rf"{title} ($\lambda_2={lam:.2f}$)", fontsize=8.5, color=col)
    ax.set_xlim(0, 10); ax.set_ylim(3, 7); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def _figure(pos, ranges, relay_mask, xs, lam):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3.4, 2.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], hspace=0.6, wspace=0.15)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, :])
    _draw(a, pos, np.array([2.4, 5.0]), ranges, relay_mask, "(a) no relay",
          active=False)
    _draw(b, pos, np.array([5.0, 5.0]), ranges, relay_mask, "(b) relay bridges")
    c.plot(xs, lam, color="#2e8b57", lw=1.8)
    c.axhline(SIGMA, color="#c0392b", ls="--", lw=1.0)
    c.text(xs[1], SIGMA, r"$\sigma_{\rm req}$", color="#c0392b", fontsize=7, va="bottom")
    c.fill_between(xs, SIGMA, lam, where=(lam >= SIGMA), color="#2e8b57", alpha=0.12)
    c.set_xlabel("relay $x$-position", fontsize=8)
    c.set_ylabel(r"$\lambda_2(L_{\rm geo})$", fontsize=8)
    c.set_title("(c) network effect vs. relay location", fontsize=8.5)
    c.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cluster.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    run()
