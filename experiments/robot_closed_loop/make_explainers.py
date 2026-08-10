"""Three explanatory diagrams for slides and the extended version (NOT in the 6-page paper).

    S1  pipeline    : Stage 1 -> fiber E -> Gamma_E -> WISE -> recovery -> re-certification,
                      annotated with what is guaranteed at each arrow and what is not.
    S2  geometry    : the productive-optimum face, the fiber, a neutral direction, the
                      lambda_2 level sets and the Gamma_E = 0 point, drawn from real sweep data.
    S3  closed loop : robot/load trajectories, the disturbance window and completion times --
                      the three things Fig. 1 does not show.

All numbers come from the generated manifests; nothing here is drawn by hand.

    python -m experiments.robot_closed_loop.make_explainers
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
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

GEN, FIGDIR = ROOT / "generated", ROOT / "figures"
RED, GREEN, ORANGE = "#c0392b", "#1e7a46", "#d68910"
INK, MUTED = "#222222", "#7f8c8d"
COLORS = {"PROD": RED, "HARD": ORANGE, "WISE": GREEN}
STYLES = {"PROD": (0, (4, 2)), "HARD": (0, (1.5, 1.5)), "WISE": "-"}
LABEL = {"PROD": "PROD", "HARD": "HARD", "WISE": "pair-WISE"}


# --------------------------------------------------------------------------- #
# S1: what the method guarantees, stage by stage
# --------------------------------------------------------------------------- #
def fig_pipeline(plt):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.16, 2.5))
    stages = [
        ("Stage 1\nmax $V$ over $X_{\\rm f}$", "fixes $y^\\star=Bz^\\star$\nuniquely"),
        ("fiber\n$\\mathcal{E}=\\{Bz=y^\\star\\}$",
         "positive-dimensional\n$\\Leftrightarrow\\ \\mu_F=0$"),
        ("$\\Gamma_{\\mathcal{E}}$\ndirectional SDP",
         "$>0$: free gain exists\n$=0$: global optimum"),
        ("Stage 2\nmax $\\lambda_2$ on $\\mathcal{E}$", "relaxed WISE\n$\\Delta V=0$ exactly"),
        ("integer\nrecovery", "rounding + repair\n(loss reported)"),
        ("direct\nre-certification", "budgets, wrench,\n$\\lambda_2\\geq\\sigma_{\\rm req}$"),
    ]
    n = len(stages)
    w, h, gap = 1.52, 0.72, 0.36
    for i, (title, note) in enumerate(stages):
        x = i * (w + gap)
        filled = i in (2, 3)
        ax.add_patch(FancyBboxPatch((x, 0.0), w, h,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc="#eaf3ee" if filled else "#f4f5f7",
                                    ec=GREEN if filled else "#9aa0a6",
                                    lw=1.4 if filled else 0.9, zorder=3))
        ax.text(x + w / 2, h / 2, title, fontsize=7.6, ha="center", va="center",
                color=INK, zorder=4)
        ax.text(x + w / 2, -0.16, note, fontsize=6.6, ha="center", va="top", color=MUTED)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.04, h / 2), (x + w + gap - 0.04, h / 2),
                                         arrowstyle="-|>", mutation_scale=9, lw=1.1,
                                         color=INK, zorder=3))
    ax.text(0.5 * (2 * (w + gap)) + w, h + 0.30,
            "the paper's contribution: a local test that certifies a global property",
            fontsize=7.4, ha="center", color=GREEN)
    ax.plot([2 * (w + gap) - 0.10, 3 * (w + gap) + w + 0.10], [h + 0.18] * 2,
            color=GREEN, lw=1.0)
    ax.set_xlim(-0.25, n * (w + gap) - gap + 0.25)
    ax.set_ylim(-0.85, h + 0.55)
    ax.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# S2: the geometry the theorem is about, drawn from the real attenuation sweep
# --------------------------------------------------------------------------- #
def fig_geometry(plt):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.16, 2.7),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- left: schematic of the face, the fiber and a neutral direction ----
    from matplotlib.patches import Polygon
    face = np.array([[0.08, 0.14], [0.96, 0.22], [0.86, 0.84], [0.16, 0.76]])
    axL.add_patch(Polygon(face, closed=True, fc="#f4f5f7", ec="#9aa0a6", lw=1.0, zorder=1))
    axL.text(0.16, 0.16, r"$X_{\rm f}$ (wrench-feasible)", fontsize=7.0, color=MUTED)
    p0, p1 = np.array([0.24, 0.40]), np.array([0.82, 0.58])
    axL.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=2.2, zorder=3)
    axL.text(0.53, 0.665, r"fiber $\mathcal{E}:\ Bz=y^\star,\ V\equiv V^\star$",
             fontsize=7.4, color=INK, ha="center")
    for t, lab, col, dy in ((0.0, "PROD", RED, -0.075),
                            (0.52, "HARD", ORANGE, -0.075),
                            (1.0, "pair-WISE", GREEN, 0.045)):
        q = p0 + t * (p1 - p0)
        axL.scatter(*q, s=52, c=col, edgecolors="k", lw=0.4, zorder=5)
        axL.text(q[0], q[1] + dy, lab, fontsize=7.2, color=col, ha="center",
                 va="bottom" if dy > 0 else "top")
    axL.annotate("", xy=p1 - 0.06 * (p1 - p0), xytext=p0 + 0.06 * (p1 - p0),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8))
    axL.text(0.53, 0.235, r"neutral direction $d$: $Bd=0$, $\lambda_2$ rises",
             fontsize=7.2, color=GREEN, ha="center")
    axL.text(p1[0] + 0.045, p1[1] + 0.005, r"$\Gamma_{\mathcal{E}}=0$", fontsize=7.6,
             color=GREEN, va="center")
    axL.text(p0[0] - 0.045, p0[1] + 0.005, r"$\Gamma_{\mathcal{E}}>0$", fontsize=7.6,
             color=RED, ha="right", va="center")
    axL.set_xlim(-0.02, 1.06)
    axL.set_ylim(0.10, 0.95)
    axL.axis("off")
    axL.set_title("(a) one scalar decides, everywhere on the fiber", fontsize=8.0, pad=4)

    # ---- right: the real lambda_2 profile along the attenuation segment ----
    curves = defaultdict(list)
    with (GEN / "robot_margin_sweep.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            curves[r["method"]].append((1.0 - float(r["attenuation"]),
                                        float(r["lambda2_bar"])))
    budget = json.loads((GEN / "attenuation_budget.json").read_text(encoding="utf-8"))
    for m in ("HARD", "WISE"):
        xy = np.array(sorted(curves[m]))
        axR.plot(xy[:, 0], xy[:, 1], color=COLORS[m], ls=STYLES[m], lw=1.6, label=LABEL[m])
        rec = budget["methods"][m]
        lam0, rc = rec["lambda2_bar"], rec["certified_rho"]
        axR.plot([0, 1], [lam0, rec["lambda2_L0"]], color=COLORS[m], lw=0.8, alpha=0.35)
        axR.axvline(rc, color=COLORS[m], ls=(0, (1, 1.4)), lw=0.9)
        axR.annotate(rf"$\rho_{{\rm cert}}={100*rc:.1f}\%$", xy=(rc, C.SIGMA_REQ),
                     xytext=(rc + 0.05, C.SIGMA_REQ + 0.055 + 0.03 * (m == "WISE")),
                     fontsize=7.0, color=COLORS[m],
                     arrowprops=dict(arrowstyle="->", color=COLORS[m], lw=0.8))
    axR.axhline(C.SIGMA_REQ, color="#2c3e50", ls="--", lw=1.0)
    axR.text(0.99, C.SIGMA_REQ + 0.008, r"$\sigma_{\rm req}$", fontsize=7.4,
             ha="right", color="#2c3e50")
    axR.text(0.62, 0.075, "thin line: the concavity chord\nthat $\\rho_{\\rm cert}$ is read from",
             fontsize=6.8, color=MUTED)
    axR.set_xlim(0, 1)
    axR.set_xlabel(r"relay attenuation $\rho$", fontsize=8.0)
    axR.set_ylabel(r"$\lambda_2(L_{\rm pair,\rho})$", fontsize=8.0)
    axR.set_title(r"(b) why the budget is certified: $\lambda_2$ is concave in $\rho$",
                  fontsize=8.0, pad=4)
    axR.legend(fontsize=7.2, frameon=False, loc="upper right")
    axR.tick_params(labelsize=7.0)
    axR.grid(alpha=0.2, lw=0.4)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# S3: what Fig. 1 leaves out -- trajectories, disturbance window, completion
# --------------------------------------------------------------------------- #
def fig_closed_loop(plt):
    ts = MF.load_timeseries()
    summary = json.loads((GEN / "robot_flagship_summary.json").read_text(encoding="utf-8"))
    _, chosen, _ = RF.select()
    runs = {m: sim.simulate(m, chosen[m], S.lbar(chosen[m])) for m in ("PROD", "WISE")}

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    ax0, ax1, ax2 = axes

    # (a) the actual load paths, PROD vs pair-WISE (load_traj is [T, M, 3] poses)
    for m, ls in (("PROD", (0, (4, 2))), ("WISE", "-")):
        traj = np.asarray(runs[m].load_traj)
        for k in range(traj.shape[1]):
            ax0.plot(traj[:, k, 0], traj[:, k, 1], color=COLORS[m], ls=ls, lw=1.5,
                     label=LABEL[m] if k == 0 else None, zorder=3)
            ax0.scatter(traj[0, k, 0], traj[0, k, 1], s=22, c="white",
                        edgecolors=COLORS[m], lw=1.0, zorder=4)
            ax0.scatter(traj[-1, k, 0], traj[-1, k, 1], marker="s", s=22, c=COLORS[m],
                        edgecolors="k", lw=0.4, zorder=4)
    relays = np.asarray([S.RELAY_SITES[r] for r in S.RELAY_SITES])
    ax0.scatter(relays[:, 0], relays[:, 1], marker="D", s=34, facecolors="none",
                edgecolors="#9aa0a6", lw=0.8, zorder=2)
    active = [a[1] for a in chosen["WISE"] if a[0] == "relay"]
    for rname in active:
        q = S.RELAY_SITES[rname]
        ax0.scatter(*q, marker="^", s=54, c=GREEN, edgecolors="k", lw=0.4, zorder=5)
        ax0.text(q[0], q[1] - 0.55, "relay", fontsize=6.8, color=GREEN, ha="center")
    ax0.set_aspect("equal")
    ax0.tick_params(labelsize=7.0)
    ax0.grid(alpha=0.2, lw=0.4)
    ax0.legend(fontsize=7.0, frameon=False, loc="lower center", ncol=2,
               borderpad=0.1, handlelength=1.6)
    ax0.set_title("(a) load paths (open start, filled end)", fontsize=8.0, pad=4)

    # (b) progress of both loads, with the disturbance window shaded
    for m in ("PROD", "HARD", "WISE"):
        d = ts[m]
        op = d["phase"] == "operational"
        ax1.plot(d["t"][op], d["sync_err"][op], color=COLORS[m], ls=STYLES[m], lw=1.5,
                 label=LABEL[m])
    ax1.axvspan(C.T_DIST, C.T_DIST + C.DUR_DIST, color="#95a5a6", alpha=0.20, lw=0)
    ax1.text(C.T_DIST + C.DUR_DIST / 2, ax1.get_ylim()[1] * 0.02, "disturbance",
             fontsize=6.8, ha="center", va="bottom", color="#5b6470")
    ax1.set_xlabel("time [s]", fontsize=8.0)
    ax1.set_ylabel(r"$|s_1-s_2|$", fontsize=8.0)
    ax1.set_title("(b) synchronization error and recovery", fontsize=8.0, pad=4)
    ax1.legend(fontsize=7.2, frameon=False, loc="center left")

    # (c) realized connectivity against the certified surrogate
    for m in ("PROD", "HARD", "WISE"):
        d = ts[m]
        op = d["phase"] == "operational"
        ax2.plot(d["t"][op], d["lam_geo"][op], color=COLORS[m], ls=STYLES[m], lw=1.5)
        ax2.axhline(summary["summaries"][m]["lambda2_bar"], color=COLORS[m], lw=0.7,
                    alpha=0.45)
    ax2.axhline(C.SIGMA_REQ, color="#2c3e50", ls="--", lw=1.0)
    ax2.axhline(C.SIGMA_DYN, color=RED, ls=":", lw=1.0)
    ax2.text(ax2.get_xlim()[0], C.SIGMA_REQ, r"$\sigma_{\rm req}$", fontsize=7.2,
             ha="left", va="bottom", color="#2c3e50")
    ax2.text(ax2.get_xlim()[0], C.SIGMA_DYN, r"$\sigma_{\rm dyn}$", fontsize=7.2,
             ha="left", va="top", color=RED)
    ax2.axvspan(C.T_DIST, C.T_DIST + C.DUR_DIST, color="#95a5a6", alpha=0.20, lw=0)
    ax2.set_xlabel("time [s]", fontsize=8.0)
    ax2.set_ylabel(r"$\lambda_2(L_{\rm geo}(q(t)))$", fontsize=8.0)
    ax2.set_title(r"(c) realized vs. certified $\lambda_2$ (thin)", fontsize=8.0, pad=4)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7.0)
        ax.grid(alpha=0.2, lw=0.4)
    fig.tight_layout()
    return fig


def main():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(exist_ok=True)
    for name, builder in (("s1_pipeline", fig_pipeline),
                          ("s2_geometry", fig_geometry),
                          ("s3_closed_loop", fig_closed_loop)):
        fig = builder(plt)
        fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight",
                    metadata={"CreationDate": None})
        fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {FIGDIR / name}.pdf/.png")


if __name__ == "__main__":
    main()
