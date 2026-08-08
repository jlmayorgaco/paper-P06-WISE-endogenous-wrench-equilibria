"""Paper-candidate figure for E-Robot: four panels, every number read from disk.

(a) trajectories and the active graph, PROD vs WISE;
(b) lambda_2(L_geo(t)) with lambda_2(Lbar), sigma_req and sigma_dyn;
(c) the weighted information-layer error (semilog);
(d) the synchronized-progress error |s_1 - s_2|.

Run after ``run_flagship.py``:  ``python -m experiments.robot_closed_loop.make_figure``
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

GEN = ROOT / "generated"
FIGDIR = ROOT / "figures"
PAPERFIG = ROOT / "paper" / "figures"

COLORS = {"PROD": "#c0392b", "HARD": "#d68910", "WISE": "#1e7a46"}
STYLES = {"PROD": (0, (4, 2)), "HARD": (0, (1.5, 1.5)), "WISE": "-"}


def load_timeseries() -> dict:
    out = defaultdict(lambda: defaultdict(list))
    with (GEN / "robot_flagship_timeseries.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m = row["method"]
            for k, v in row.items():
                if k in ("method", "phase"):
                    out[m][k].append(v)
                else:
                    out[m][k].append(float(v))
    return {m: {k: (np.array(v) if k not in ("method", "phase") else np.array(v))
                for k, v in d.items()} for m, d in out.items()}


def _load_polygon(k, q):
    from experiments.robot_closed_loop import scenario as S
    off = S.SLOT_OFFSETS[k]
    lo, hi = off.min(0) - 0.16, off.max(0) + 0.16
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    c, s = np.cos(q[2]), np.sin(q[2])
    return q[:2] + box @ np.array([[c, -s], [s, c]]).T


def _scene_panel(ax, summary, title, method, res, z, annotate=False):
    from matplotlib.patches import Polygon

    from experiments.robot_closed_loop import assignments as A
    from experiments.robot_closed_loop import scenario as S

    n = len(res.t) - 1
    pos = res.robot_traj[n]
    relay = A.relay_mask(z)

    for k in range(S.M_LOADS):
        q = np.atleast_2d(S.LOAD_PATHS[k].pose(np.linspace(0, 1, 120)))
        ax.plot(q[:, 0], q[:, 1], color="#aeb4ba", lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.add_patch(Polygon(_load_polygon(k, res.load_traj[n, k]), closed=True,
                             fc="#eef1f4", ec="#5b6470", lw=1.0, zorder=3))
    L = S.lgeo(pos, relay)
    for i in range(S.N_ROBOTS):
        for j in range(i + 1, S.N_ROBOTS):
            if -L[i, j] > 1e-9:
                bridging = relay[i] or relay[j]
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color=COLORS["WISE"] if bridging else "#c2c7cc",
                        lw=1.5 if bridging else 0.7, zorder=2, alpha=0.95)
    for site in S.RELAY_SITES.values():
        ax.scatter(*site, marker="D", s=18, facecolors="none",
                   edgecolors="#bfc4c9", linewidths=0.6, zorder=3)
    for i in range(S.N_ROBOTS):
        long_i = bool(S.IS_LONG[i])
        ax.scatter(pos[i, 0], pos[i, 1], marker="^" if long_i else "o",
                   s=62 if long_i else 26,
                   c=COLORS["WISE"] if relay[i] else COLORS["PROD"],
                   edgecolors="k", linewidths=0.4, zorder=5)
    lam = summary["summaries"][method]["lambda2_bar"]
    ax.set_title(f"{title}:  $\\lambda_2(\\bar L)={lam:.3f}$", fontsize=7.5, pad=3)
    if annotate:
        ax.text(0.9, -1.95, "load 1", fontsize=5.4, ha="center", color="#5b6470")
        ax.text(7.1, -1.95, "load 2", fontsize=5.4, ha="center", color="#5b6470")
    ax.set_aspect("equal")
    ax.set_xlim(-0.9, 8.9)
    ax.set_ylim(-2.35, 1.85)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color("#c2c7cc")


def main():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    from experiments.robot_closed_loop import config as C

    ts = load_timeseries()
    summary = json.loads((GEN / "robot_flagship_summary.json").read_text(encoding="utf-8"))
    sigma_req = summary["sigma_req"]
    sigma_dyn = summary["sigma_dyn"]
    methods = [m for m in ("PROD", "HARD", "WISE") if m in ts]

    from experiments.robot_closed_loop import run_flagship as RF
    from experiments.robot_closed_loop import scenario as S
    from experiments.robot_closed_loop import simulator as sim

    fig = plt.figure(figsize=(7.15, 4.5))
    gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.26,
                          height_ratios=[0.82, 0.82, 1.25])
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[0:2, 1])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])

    _, chosen, _ = RF.select()
    for ax, m, ttl, ann in ((ax_a1, "PROD", "(a) productive-only", False),
                            (ax_a2, "WISE", "WISE", True)):
        z = chosen[m]
        _scene_panel(ax, summary, ttl, m, sim.simulate(m, z, S.lbar(z)), z, annotate=ann)

    for m in methods:
        d = ts[m]
        op = d["phase"] == "operational"
        t = d["t"][op]
        ax_b.plot(t, d["lam_geo"][op], color=COLORS[m], ls=STYLES[m], lw=1.3, label=m)
        ax_b.axhline(summary["summaries"][m]["lambda2_bar"], color=COLORS[m],
                     lw=0.6, alpha=0.55)
        ax_c.semilogy(t, np.maximum(d["info_norm_certified"][op], 1e-16),
                      color=COLORS[m], ls=STYLES[m], lw=1.3, label=m)
        ax_d.plot(t, d["sync_err"][op], color=COLORS[m], ls=STYLES[m], lw=1.3, label=m)

    ax_b.axhline(sigma_req, color="#2c3e50", ls="--", lw=0.9)
    ax_b.axhline(sigma_dyn, color="#c0392b", ls=":", lw=0.9)
    ax_b.text(ax_b.get_xlim()[1], sigma_req, r"$\sigma_{\rm req}$", fontsize=6,
              ha="right", va="bottom", color="#2c3e50")
    ax_b.text(ax_b.get_xlim()[1], sigma_dyn, r"$\sigma_{\rm dyn}$", fontsize=6,
              ha="right", va="top", color="#c0392b")
    ax_b.set_ylabel(r"$\lambda_2(L_{\rm geo}(q(t)))$", fontsize=7.5)
    ax_b.set_title(r"(b) realized connectivity vs. $\lambda_2(\bar L)$ (thin)", fontsize=8)

    ax_c.set_ylabel(r"$\|[a,b]\|_P$", fontsize=7.5)
    ax_c.set_title("(c) certified information layer", fontsize=8)
    ax_d.set_ylabel(r"$|s_1-s_2|$", fontsize=7.5)
    ax_d.set_title("(d) synchronization error", fontsize=8)

    for ax in (ax_b, ax_c, ax_d):
        ax.axvspan(C.T_DIST, C.T_DIST + C.DUR_DIST, color="#95a5a6", alpha=0.16, lw=0)
        ax.set_xlabel("time [s]", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.22, lw=0.4)
    ax_d.legend(fontsize=6, frameon=False, loc="upper left")

    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "robot_closed_loop.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(FIGDIR / "robot_closed_loop.png", dpi=190, bbox_inches="tight")
    if PAPERFIG.exists():
        fig.savefig(PAPERFIG / "fig_robot_closed_loop.pdf", bbox_inches="tight",
                    metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {FIGDIR / 'robot_closed_loop.pdf'} and .png")


if __name__ == "__main__":
    main()
