"""Physical-scene figure: a real two-region instance, regime A vs B.

Draws actual robot positions (long/short range), the rigid load with its demanded
force/torque, the live communication graph, and the contact forces the lifters
realise under the nonholonomic (heading-aligned) force sets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from wise_mr import dynamics as dyn, scenarios  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 200,
                     "savefig.bbox": "tight", "axes.linewidth": 0.7})
C_LONG, C_SHORT, C_WISE, C_BAD, C_ACC = "#2e6f9e", "#c98a3b", "#66a182", "#d1495b", "#2e4057"


def _edges(pos, rng, relay_mask, base_range=2.5):
    E = []
    N = len(pos)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d <= base_range:
                E.append((i, j, "short"))
    for i in np.where(relay_mask)[0]:
        for j in range(N):
            if j != i and np.linalg.norm(pos[i] - pos[j]) <= rng[i]:
                E.append((i, j, "bridge"))
    return E


def _panel(ax, prob, relay_idx, bridge, title, good):
    m = prob.meta
    pos = m["pos"].copy()
    rng, is_long, load = m["r"], m["is_long"], m["load"]
    relay_mask = np.zeros(prob.N, bool)
    if relay_idx is not None:
        relay_mask[relay_idx] = True
        pos[relay_idx] = bridge
    lam = dyn.live_lambda2(pos, rng, relay_mask)

    ax.set_xlim(-0.5, 10.5); ax.set_ylim(1.5, 8.5); ax.set_aspect("equal"); ax.axis("off")
    # comm edges
    for i, j, kind in _edges(pos, rng, relay_mask):
        col = C_WISE if kind == "bridge" else "#b0b0b0"
        lw = 1.6 if kind == "bridge" else 0.7
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color=col, lw=lw, zorder=1)
    # load + demand arrows
    ax.add_patch(mp.Rectangle(load - [0.7, 0.5], 1.4, 1.0, fc=C_ACC, ec="k", lw=0.6, zorder=2))
    ax.text(load[0], load[1], "load", color="w", ha="center", va="center", fontsize=7, zorder=3)
    ax.annotate("", xy=(load[0], load[1] + 1.7), xytext=(load[0], load[1] + 0.5),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
    ax.text(load[0] + 0.15, load[1] + 1.4, r"$w^{\mathrm{dem}}$", fontsize=7.5)
    ax.add_patch(mp.FancyArrowPatch((load[0] + 0.9, load[1] + 0.7), (load[0] + 0.2, load[1] + 1.0),
                 connectionstyle="arc3,rad=0.5", arrowstyle="-|>", color="k", lw=1.0, mutation_scale=8))
    ax.text(load[0] + 1.0, load[1] + 0.9, r"$\tau_d$", fontsize=7.5)
    # robots
    for i in range(prob.N):
        marker = "s" if is_long[i] else "o"
        col = C_LONG if is_long[i] else C_SHORT
        if relay_mask[i]:
            col = C_WISE
        ax.scatter(*pos[i], s=70, marker=marker, c=col, edgecolors="k", linewidths=0.5, zorder=4)
    # lifters: 3 nearest to load get contact-force arrows
    dist = np.linalg.norm(pos - load, axis=1)
    lifters = np.argsort(dist)[:3]
    offs = pos[lifters] - load
    head = np.arctan2(-offs[:, 1], -offs[:, 0])          # face the load
    F = prob.meta["F"][lifters]
    w_cmd = np.array([0.0, 2.0, 0.6])
    f, _, _ = dyn.realize_wrench(offs, head, F, 0.4, w_cmd)
    for k, i in enumerate(lifters):
        if np.linalg.norm(f[k]) > 1e-2:
            ax.annotate("", xy=pos[i] + 0.6 * f[k] / max(np.linalg.norm(f[k]), 1e-6),
                        xytext=pos[i], arrowprops=dict(arrowstyle="-|>", color=C_BAD, lw=1.2))
    col_t = C_WISE if good else C_BAD
    ax.set_title(title + rf"  ($\lambda_2={lam:.2f}$, $\sigma^\star={prob.sigma:.2f}$)",
                 fontsize=8.5, color=col_t)


def main():
    prob = scenarios.two_region(seed=7, N=10, nu=0.4, tau_d=2.5, bridge_gain=3.0)
    long_ids = np.where(prob.meta["is_long"])[0]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    _panel(axA, prob, None, None, "(a) fragmented", good=False)
    _panel(axB, prob, long_ids[0], np.array([5.0, 5.0]), "(b) WISE", good=True)
    # legend
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=C_SHORT, mec="k", label="short-range"),
               plt.Line2D([], [], marker="s", ls="", mfc=C_LONG, mec="k", label="long-range"),
               plt.Line2D([], [], marker="s", ls="", mfc=C_WISE, mec="k", label="relay"),
               plt.Line2D([], [], color=C_BAD, lw=1.2, label="contact force")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig1_scene.pdf")
    print("wrote fig1_scene.pdf")


if __name__ == "__main__":
    main()
