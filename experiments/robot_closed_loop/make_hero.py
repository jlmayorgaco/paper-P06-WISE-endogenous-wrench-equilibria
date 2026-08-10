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


# --------------------------------------------------------------------------- #
# (d) the certified attenuation budget
# --------------------------------------------------------------------------- #
def _budget_panel(ax, sigma_req):
    """lambda_2(Lbar_rho) against rho, with the certified budgets of Cor. 3.

    Curves come from ``generated/robot_margin_sweep.csv`` (column ``attenuation`` is the
    surviving fraction 1-rho); the certified budgets from ``attenuation_budget.json``.
    """
    import csv
    from collections import defaultdict

    curves = defaultdict(list)
    with (GEN / "robot_margin_sweep.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            curves[r["method"]].append((1.0 - float(r["attenuation"]),
                                        float(r["lambda2_bar"])))
    budget = json.loads((GEN / "attenuation_budget.json").read_text(encoding="utf-8"))

    grid = 0.01
    for m in ("HARD", "WISE"):
        xy = np.array(sorted(curves[m]))
        ax.plot(xy[:, 0], xy[:, 1], color=COLORS[m], ls=STYLES[m], lw=1.5,
                label={"WISE": "pair-WISE"}.get(m, m))
        rec = budget["methods"][m]
        rc, rho_pass = rec["certified_rho"], rec["observed_last_rho_clearing_sigma_req"]
        rho_fail = round(rho_pass + grid, 4)
        ax.axvline(rc, color=COLORS[m], ls=(0, (1, 1.4)), lw=0.8, alpha=0.9)
        ax.plot([rho_pass], [np.interp(rho_pass, xy[:, 0], xy[:, 1])], marker="o", ms=3.4,
                mfc="white", mec=COLORS[m], mew=1.0, zorder=6)
        ax.plot([rho_fail], [np.interp(rho_fail, xy[:, 0], xy[:, 1])], marker="x", ms=3.6,
                mec=COLORS[m], mew=1.0, zorder=6)
        ax.text(rc - 0.008, 0.405, rf"${100 * rc:.1f}\%$", fontsize=6.0, rotation=90,
                ha="right", va="top", color=COLORS[m])
    ax.axhline(sigma_req, color="#2c3e50", ls="--", lw=0.9)
    ax.text(0.592, sigma_req + 0.006, r"$\sigma_{\rm req}$", fontsize=6.4, ha="right",
            va="bottom", color="#2c3e50")
    ax.set_xlim(0.0, 0.6)
    ax.set_xlabel(r"relay attenuation $\rho$", fontsize=7.0, labelpad=1.5)
    ax.set_ylabel(r"$\lambda_2(L_{\rm pair,\rho})$", fontsize=7.0, labelpad=1.5)
    ax.set_title("(b) HARD vs. pair-WISE relay attenuation", fontsize=7.4, pad=3.0)
    ax.tick_params(labelsize=6.2)
    ax.grid(alpha=0.2, lw=0.35)


def main():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    ts = MF.load_timeseries()
    summary = json.loads((GEN / "robot_flagship_summary.json").read_text(encoding="utf-8"))
    sigma_req = summary["sigma_req"]
    methods = [m for m in ("PROD", "HARD", "WISE") if m in ts]
    lam = {m: summary["summaries"][m]["lambda2_bar"] for m in methods}

    _, chosen, _ = RF.select()
    runs = {m: sim.simulate(m, chosen[m], S.lbar(chosen[m])) for m in ("PROD", "WISE")}

    fig = plt.figure(figsize=(7.16, 2.16))
    gs = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.42,
                          width_ratios=[1.30, 1.34, 0.96])
    ax_a1, ax_a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[:, 1])                       # (b) the certified budget
    ax_d = fig.add_subplot(gs[:, 2])                       # (c) information-layer error

    # (a) the exchange, shown on the actual teams: composition + invariants + lambda_2
    comp = {"PROD": "no relay", "WISE": "L relays"}
    for ax, m, ttl, ann in ((ax_a1, "PROD", "(a) productive-only", False),
                            (ax_a2, "WISE", "pair-WISE", True)):
        MF._scene_panel(ax, summary, ttl, m, runs[m], chosen[m], annotate=ann)
        ax.set_title(rf"{ttl}:  {comp[m]},  $\lambda_2(L_{{\rm pair}})={lam[m]:.3f}$",
                     fontsize=7.0, pad=3.0)

    # (c) the certified information-layer error
    t0 = None
    for m in methods:
        d = ts[m]
        op = d["phase"] == "operational"
        y = np.maximum(d["info_norm_certified"][op], 1e-16)
        ax_d.semilogy(d["t"][op], y, color=COLORS[m], ls=STYLES[m], lw=1.3)
        if t0 is None:
            t0, y0 = d["t"][op], float(y[0])
    # the certified lower-rate bound of Cor. 2: an envelope, not a fitted curve
    alpha_cert = float(summary["summaries"]["WISE"]["alpha_certified"])
    ax_d.semilogy(t0, y0 * np.exp(-alpha_cert * (t0 - t0[0])), color="#7f8c8d",
                  ls=(0, (5, 2)), lw=1.0, zorder=1)
    ax_d.text(t0[-1], y0 * np.exp(-alpha_cert * (t0[-1] - t0[0])),
              rf"certified $e^{{-{alpha_cert:.4f}t}}$", fontsize=5.8, color="#7f8c8d",
              ha="right", va="bottom")
    ax_d.set_ylabel(r"$\|[a,b]\|_P$", fontsize=7.0, labelpad=1.5)
    lbl = {"PROD": "PROD (diverges)", "HARD": "HARD", "WISE": "pair-WISE"}
    ax_d.legend(handles=[plt.Line2D([], [], color=COLORS[m], ls=STYLES[m], lw=1.3,
                                    label=lbl[m])
                         for m in methods],
                fontsize=6.0, frameon=False, loc="lower left", handlelength=1.7,
                borderpad=0.15, labelspacing=0.25)
    ax_d.set_title("(c) information-layer error", fontsize=7.4, pad=3.0)

    for ax in (ax_d,):
        ax.axvspan(C.T_DIST, C.T_DIST + C.DUR_DIST, color="#95a5a6", alpha=0.16, lw=0)
        ax.set_xlabel("time [s]", fontsize=7.0, labelpad=1.5)
        ax.tick_params(labelsize=6.2)
        ax.grid(alpha=0.2, lw=0.35)

    # (b) the operational reading of the reserve
    _budget_panel(ax_f, sigma_req)

    fig.subplots_adjust(left=0.005, right=0.965, top=0.930, bottom=0.095)
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(PAPERFIG / "fig_robot_hero.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(FIGDIR / "robot_hero.png", dpi=210, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PAPERFIG / 'fig_robot_hero.pdf'}")


if __name__ == "__main__":
    main()
