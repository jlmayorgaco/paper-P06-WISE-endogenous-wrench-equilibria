"""The two-column hero figure: assignment -> networkability -> stability -> mission.

Four panels, all numbers read from ``generated/robot_flagship_timeseries.csv`` and
``generated/robot_flagship_summary.json``:

(a) the composition exchange itself (productive-only vs WISE), with the invariants;
(b) robot/load trajectories and the active communication graph;
(c) lambda_2(L_geo(q(t))) against lambda_2(Lbar), sigma_req and sigma_dyn;
(d) the certified information-layer error and the progress synchronization error.

It replaces both standalone figures of the previous draft. Writes
``paper/figures/fig_robot_hero.pdf`` (and a PNG for inspection).

    python -m experiments.robot_closed_loop.make_hero
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.robot_closed_loop import config as C  # noqa: E402
from experiments.robot_closed_loop import make_figure as MF  # noqa: E402
from experiments.robot_closed_loop import run_flagship as RF  # noqa: E402
from experiments.robot_closed_loop import scenario as S  # noqa: E402
from experiments.robot_closed_loop import simulator as sim  # noqa: E402

GEN = ROOT / "generated"
FIGDIR = ROOT / "figures"
PAPERFIG = ROOT / "paper" / "figures"

RED, GREEN, ORANGE = "#c0392b", "#1e7a46", "#d68910"
INK, GRAY, LOAD_FC, LOAD_EC = "#222222", "#c2c7cc", "#eef1f4", "#5b6470"
COLORS = {"PROD": RED, "HARD": ORANGE, "WISE": GREEN}
STYLES = {"PROD": (0, (4, 2)), "HARD": (0, (1.5, 1.5)), "WISE": "-"}


# --------------------------------------------------------------------------- #
# (a) the composition exchange
# --------------------------------------------------------------------------- #
def _exchange_panel(ax, lifters1, lifters2, relay_long, title, lam):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    xA, xB, xR = 0.0, 5.4, 2.7
    LOAD_Y, LIFT_Y, RELAY_Y = 1.30, 0.35, -0.75
    LW, LH = 2.05, 0.42

    for cx, lift in ((xA, lifters1), (xB, lifters2)):
        ax.add_patch(FancyBboxPatch((cx - LW / 2, LOAD_Y - LH / 2), LW, LH,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fc=LOAD_FC, ec=LOAD_EC, lw=0.9, zorder=3))
        ax.add_patch(FancyArrowPatch((cx, LOAD_Y + 0.55), (cx, LOAD_Y + LH / 2 + 0.03),
                                     arrowstyle="-|>", mutation_scale=7, lw=1.1,
                                     color=INK, zorder=3))
        xs = np.linspace(cx - 0.30 * (len(lift) - 1), cx + 0.30 * (len(lift) - 1), len(lift))
        for x, is_long in zip(xs, lift, strict=True):
            ax.plot([x, x], [LIFT_Y + 0.14, LOAD_Y - LH / 2], color=LOAD_EC, lw=0.6,
                    ls=(0, (2, 1.5)), zorder=1)
            ax.scatter(x, LIFT_Y, marker="^" if is_long else "o", s=64 if is_long else 26,
                       c=RED, edgecolors="k", lw=0.35, zorder=5)
    ax.text(xB + 0.92, LOAD_Y + 0.28, r"$w^{\rm dem}_k$", fontsize=5.2, ha="left", color=INK)
    ax.scatter(xR, RELAY_Y, marker="D", s=120, facecolors="none", edgecolors="#9aa0a6",
               lw=0.7, zorder=2)
    if relay_long:
        for cx in (xA, xB):
            ax.plot([cx, xR], [LIFT_Y, RELAY_Y], color=GREEN, lw=1.5, zorder=1)
        ax.scatter(xR, RELAY_Y, marker="^", s=64, c=GREEN, edgecolors="k", lw=0.35, zorder=5)
    else:
        ax.plot([xR - 0.45, xR + 0.45], [RELAY_Y, RELAY_Y], color=RED, lw=1.1,
                ls=(0, (1, 1.2)), zorder=1)
        ax.text(xR, RELAY_Y - 0.62, "gap", fontsize=5.4, color=RED, ha="center")
    ax.text(xA, LIFT_Y - 0.78, "3S" if len(lifters1) == 3 else "1L+1S",
            fontsize=5.6, ha="center", color=INK)
    ax.text(xB, LIFT_Y - 0.78, "1L+1S", fontsize=5.6, ha="center", color=INK)
    ax.set_title(rf"{title}:  $\lambda_2(\bar L)={lam:.3f}$", fontsize=6.6, pad=4)
    ax.set_xlim(-1.75, 6.4)
    ax.set_ylim(-2.00, 2.60)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    ts = MF.load_timeseries()
    summary = json.loads((GEN / "robot_flagship_summary.json").read_text(encoding="utf-8"))
    sigma_req, sigma_dyn = summary["sigma_req"], summary["sigma_dyn"]
    methods = [m for m in ("PROD", "HARD", "WISE") if m in ts]
    lam = {m: summary["summaries"][m]["lambda2_bar"] for m in methods}

    _, chosen, _ = RF.select()
    runs = {m: sim.simulate(m, chosen[m], S.lbar(chosen[m])) for m in ("PROD", "WISE")}

    fig = plt.figure(figsize=(7.16, 1.08))
    gs = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.52,
                          width_ratios=[1.42, 1.0, 1.0])
    ax_a1, ax_a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_c, ax_d = fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])

    # (a) the exchange, shown on the actual teams: composition + invariants + lambda_2
    comp = {"PROD": "1L+1S | 1L+1S, no relay", "WISE": "3S | 1L+1S, L relays"}
    for ax, m, ttl, ann in ((ax_a1, "PROD", "(a) productive-only", False),
                            (ax_a2, "WISE", "WISE", True)):
        MF._scene_panel(ax, summary, ttl, m, runs[m], chosen[m], annotate=ann)
        ax.set_title(rf"{ttl}:  {comp[m]},  $\lambda_2(\bar L)={lam[m]:.3f}$",
                     fontsize=6.4, pad=2.5)

    # (c) connectivity, (d) the two errors
    for m in methods:
        d = ts[m]
        op = d["phase"] == "operational"
        t = d["t"][op]
        ax_c.plot(t, d["lam_geo"][op], color=COLORS[m], ls=STYLES[m], lw=1.15, label=m)
        ax_c.axhline(lam[m], color=COLORS[m], lw=0.5, alpha=0.5)
        ax_d.semilogy(t, np.maximum(d["info_norm_certified"][op], 1e-16),
                      color=COLORS[m], ls=STYLES[m], lw=1.15)
    ax_c.axhline(sigma_req, color="#2c3e50", ls="--", lw=0.8)
    ax_c.axhline(sigma_dyn, color=RED, ls=":", lw=0.8)
    ax_c.text(ax_c.get_xlim()[0], sigma_req, r"$\sigma_{\rm req}$", fontsize=5.6,
              ha="left", va="bottom", color="#2c3e50")
    ax_c.text(ax_c.get_xlim()[0], sigma_dyn, r"$\sigma_{\rm dyn}$", fontsize=5.6,
              ha="left", va="top", color=RED)
    ax_c.set_ylabel(r"$\lambda_2(L_{\rm geo}(q(t)))$", fontsize=6.6, labelpad=1.5)
    ax_c.set_title(r"(b) realized vs. certified $\lambda_2(\bar L)$", fontsize=6.4, pad=2.5)
    ax_d.set_ylabel(r"$\|[a,b]\|_P$", fontsize=6.6, labelpad=1.5)
    ax_d.legend(handles=[plt.Line2D([], [], color=COLORS[m], ls=STYLES[m], lw=1.15,
                                    label=m) for m in methods],
                fontsize=5.4, frameon=False, loc="lower left", handlelength=1.6,
                borderpad=0.15, labelspacing=0.25)
    ax_d.set_title("(c) information layer and sync. error", fontsize=6.4, pad=2.5)
    ax_e = ax_d.twinx()
    for m in methods:
        d = ts[m]
        op = d["phase"] == "operational"
        ax_e.plot(d["t"][op], d["sync_err"][op], color=COLORS[m], ls=STYLES[m], lw=0.9,
                  alpha=0.45)
    ax_e.set_ylabel(r"$|s_1-s_2|$", fontsize=6.6, labelpad=1.5, color="#555555")
    ax_e.tick_params(labelsize=5.6, colors="#555555")
    ax_e.set_ylim(0, None)

    for ax in (ax_c, ax_d):
        ax.axvspan(C.T_DIST, C.T_DIST + C.DUR_DIST, color="#95a5a6", alpha=0.16, lw=0)
        ax.set_xlabel("time [s]", fontsize=6.6, labelpad=1.5)
        ax.tick_params(labelsize=5.6)
        ax.grid(alpha=0.2, lw=0.35)
    ax_e.axvspan(C.T_DIST, C.T_DIST + C.DUR_DIST, color="#95a5a6", alpha=0.0, lw=0)

    fig.subplots_adjust(left=0.005, right=0.945, top=0.885, bottom=0.155)
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(PAPERFIG / "fig_robot_hero.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(FIGDIR / "robot_hero.png", dpi=210, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PAPERFIG / 'fig_robot_hero.pdf'}")


if __name__ == "__main__":
    main()
