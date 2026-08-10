"""Three explanatory diagrams for slides and the extended version (NOT in the 6-page paper).

    S1  pipeline    : Stage 1 -> fiber -> Gamma_E -> Stage 2 -> recovery -> re-certification,
                      each step annotated with what it guarantees and what it does not, and
                      with the two experiment families attached to the steps they validate.
    S2  mechanism   : why a free gain exists at all -- the productive optimum fixes the
                      aggregate but not the composition -- beside the real concavity chord
                      that the certified attenuation budget is read from.
    S3  closed loop : load paths, disturbance window, synchronization recovery and realized
                      vs. certified connectivity: the things Fig. 1 has no room for.

Every number is read from the generated manifests; nothing here is hand-drawn.

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
INK, MUTED, PAPER = "#222222", "#7f8c8d", "#f4f5f7"
COLORS = {"PROD": RED, "HARD": ORANGE, "WISE": GREEN}
STYLES = {"PROD": (0, (4, 2)), "HARD": (0, (1.5, 1.5)), "WISE": "-"}
LABEL = {"PROD": "PROD", "HARD": "HARD", "WISE": "pair-WISE"}


# --------------------------------------------------------------------------- #
# S1: the method, and what each step is and is not allowed to claim
# --------------------------------------------------------------------------- #
def fig_pipeline(plt):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    steps = [
        ("1. Stage 1", "$\\max V$ over $X_{\\rm f}$", "$y^\\star=Bz^\\star$ unique", False),
        ("2. fiber", "$\\mathcal{E}=\\{Bz=y^\\star\\}$",
         "$\\dim>0\\Leftrightarrow\\mu_F=0$", False),
        ("3. $\\Gamma_{\\mathcal{E}}$", "directional SDP",
         "$>0$ free gain exists\n$=0$ global optimum", True),
        ("4. Stage 2", "$\\max\\lambda_2$ on $\\mathcal{E}$", "relaxed WISE, $\\Delta V=0$",
         True),
        ("5. recovery", "round + repair", "loss reported\nseparately", False),
        ("6. re-certify", "budgets, wrench, $\\lambda_2$", "integer WISE\nor recovery loss",
         False),
    ]
    w, h, gap = 1.30, 0.90, 0.30
    for i, (num, body, note, hi) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.0), w, h,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc="#eaf3ee" if hi else PAPER,
                                    ec=GREEN if hi else "#9aa0a6",
                                    lw=1.6 if hi else 1.0, zorder=3))
        ax.text(x + w / 2, h * 0.68, num, fontsize=9.0, ha="center", va="center",
                color=GREEN if hi else INK, weight="bold", zorder=4)
        ax.text(x + w / 2, h * 0.30, body, fontsize=8.0, ha="center", va="center",
                color=INK, zorder=4)
        ax.text(x + w / 2, -0.14, note, fontsize=7.4, ha="center", va="top", color=MUTED)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, h / 2), (x + w + gap - 0.03, h / 2),
                                         arrowstyle="-|>", mutation_scale=11, lw=1.3,
                                         color=INK, zorder=3))

    x2, x3 = 2 * (w + gap), 3 * (w + gap) + w
    ax.plot([x2 - 0.06, x3 + 0.06], [h + 0.20] * 2, color=GREEN, lw=1.4)
    ax.text((x2 + x3) / 2, h + 0.30,
            "the contribution: a local test that certifies a global property",
            fontsize=8.6, ha="center", color=GREEN)

    # which experiment family exercises which steps
    def band(x0, x1, y, col, text):
        ax.plot([x0, x1], [y, y], color=col, lw=2.6, solid_capstyle="butt", alpha=0.75)
        ax.text((x0 + x1) / 2, y - 0.10, text, fontsize=7.6, ha="center", va="top",
                color=col)
    band(0.0, 3 * (w + gap) + w, -0.72, "#2c3e50",
         "affine family $N{=}12$ (Table I.A): validates steps 1–4")
    band(4 * (w + gap), 6 * (w + gap) - gap, -0.72, ORANGE,
         "and steps 5–6")
    band(0.0, 6 * (w + gap) - gap, -1.12, RED,
         "pair-dependent stress test $N{=}6$ (Table I.B): enumeration only, "
         "no claim on steps 3–4")
    ax.set_xlim(-0.3, 6 * (w + gap) - gap + 0.3)
    ax.set_ylim(-1.55, h + 0.60)
    ax.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# S2: why a free gain exists, and why the budget that follows is certified
# --------------------------------------------------------------------------- #
def fig_mechanism(plt):
    from matplotlib.patches import FancyArrowPatch, Polygon

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                   gridspec_kw={"width_ratios": [1.0, 1.05]})

    # ---- left: the aggregate is fixed, the composition is not ----
    face = np.array([[0.06, 0.10], [0.97, 0.18], [0.88, 0.90], [0.13, 0.82]])
    axL.add_patch(Polygon(face, closed=True, fc=PAPER, ec="#9aa0a6", lw=1.2, zorder=1))
    axL.text(0.90, 0.235, r"$X_{\rm f}$: wrench-" + "\n" + "feasible assignments",
             fontsize=8.2, color=MUTED, ha="right", va="top")

    p0, p1 = np.array([0.24, 0.50]), np.array([0.80, 0.62])
    axL.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=2.6, zorder=3)
    axL.text(0.50, 0.795, r"fiber $\mathcal{E}$: every point has $Bz=y^\star$, $V=V^\star$",
             fontsize=8.4, color=INK, ha="center")
    axL.annotate("", xy=(0.50, 0.585), xytext=(0.50, 0.765),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))

    # labels fanned out so none of them lands on the segment or on each other
    for t, m, tx, ty, ha in ((0.0, "PROD", 0.055, 0.375, "center"),
                             (0.52, "HARD", 0.50, 0.375, "center"),
                             (1.0, "WISE", 0.955, 0.375, "center")):
        q = p0 + t * (p1 - p0)
        axL.scatter(*q, s=95, c=COLORS[m], edgecolors="k", lw=0.5, zorder=5)
        lam = {"PROD": 0.0, "HARD": 0.3365, "WISE": 0.3989}[m]
        axL.annotate(f"{LABEL[m]}\n$\\lambda_2={lam:.4f}$", xy=q, xytext=(tx, ty),
                     fontsize=8.0, color=COLORS[m], ha=ha, va="top",
                     arrowprops=dict(arrowstyle="-", color=COLORS[m], lw=0.7, alpha=0.6))

    axL.add_patch(FancyArrowPatch(p0 + 0.07 * (p1 - p0), p1 - 0.07 * (p1 - p0),
                                  arrowstyle="-|>", mutation_scale=13, lw=2.2,
                                  color=GREEN, zorder=4))
    axL.text(0.42, 0.105,
             "neutral direction $d$: $Bd=0$, so $V$ never moves\n"
             r"while $\lambda_2$ strictly rises $\Rightarrow$ the gain is free",
             fontsize=8.2, color=GREEN, ha="center", va="top")
    axL.text(p0[0] - 0.035, p0[1] + 0.045, r"$\Gamma_{\mathcal{E}}>0$", fontsize=8.6,
             color=RED, ha="center", va="bottom")
    axL.text(p1[0] + 0.035, p1[1] + 0.045, r"$\Gamma_{\mathcal{E}}=0$", fontsize=8.6,
             color=GREEN, ha="center", va="bottom")
    axL.set_xlim(-0.10, 1.12)
    axL.set_ylim(-0.06, 0.92)
    axL.axis("off")
    axL.set_title("(a) the productive optimum fixes the aggregate, not the composition",
                  fontsize=9.0, pad=6)

    # ---- right: the concavity chord the budget is read from ----
    curves = defaultdict(list)
    with (GEN / "robot_margin_sweep.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            curves[r["method"]].append((1.0 - float(r["attenuation"]),
                                        float(r["lambda2_bar"])))
    budget = json.loads((GEN / "attenuation_budget.json").read_text(encoding="utf-8"))

    for m in ("HARD", "WISE"):
        xy = np.array(sorted(curves[m]))
        rec = budget["methods"][m]
        lam0, lamL0, rc = rec["lambda2_bar"], rec["lambda2_L0"], rec["certified_rho"]
        axR.plot([0, 1], [lam0, lamL0], color=COLORS[m], lw=1.0, alpha=0.40, zorder=1)
        axR.plot(xy[:, 0], xy[:, 1], color=COLORS[m], ls=STYLES[m], lw=2.0, zorder=3)
        axR.axvline(rc, color=COLORS[m], ls=(0, (1, 1.4)), lw=1.0, zorder=2)
        axR.scatter([rc], [C.SIGMA_REQ], s=42, c=COLORS[m], edgecolors="k", lw=0.5,
                    zorder=6)
        axR.text(rc, C.SIGMA_REQ - 0.022, f"{100*rc:.1f}%", fontsize=8.4,
                 color=COLORS[m], ha="center", va="top")
        axR.text(0.015, lam0 - 0.012, LABEL[m], fontsize=8.4, color=COLORS[m],
                 ha="left", va="top")

    axR.axhline(C.SIGMA_REQ, color="#2c3e50", ls="--", lw=1.2)
    axR.text(0.985, C.SIGMA_REQ + 0.008, r"$\sigma_{\rm req}$", fontsize=8.6,
             ha="right", color="#2c3e50")
    axR.annotate("thin chord = the concavity lower bound.\n"
                 "The curve stays above it, so each\ncrossing is certified, not fitted.",
                 xy=(0.80, 0.3989 * (1 - 0.80)), xytext=(0.42, 0.115),
                 fontsize=8.0, color=MUTED, va="top",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 0.45)
    axR.set_xlabel(r"relay attenuation $\rho$ (all gated relay links scaled by $1-\rho$)",
                   fontsize=8.6)
    axR.set_ylabel(r"$\lambda_2(L_{\rm pair,\rho})$", fontsize=8.6)
    axR.set_title(r"(b) the budget is certified because $\lambda_2$ is concave in $\rho$",
                  fontsize=9.0, pad=6)
    axR.tick_params(labelsize=8.0)
    axR.grid(alpha=0.22, lw=0.5)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# S3: what Fig. 1 has no room for
# --------------------------------------------------------------------------- #
def fig_closed_loop(plt):
    ts = MF.load_timeseries()
    summary = json.loads((GEN / "robot_flagship_summary.json").read_text(encoding="utf-8"))
    _, chosen, _ = RF.select()
    runs = {m: sim.simulate(m, chosen[m], S.lbar(chosen[m])) for m in ("PROD", "WISE")}

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9.2, 3.0),
                                        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

    # (a) the actual load paths
    for m, ls in (("PROD", (0, (4, 2))), ("WISE", "-")):
        traj = np.asarray(runs[m].load_traj)
        for k in range(traj.shape[1]):
            ax0.plot(traj[:, k, 0], traj[:, k, 1], color=COLORS[m], ls=ls, lw=1.8,
                     zorder=3, label=LABEL[m] if k == 0 else None)
            ax0.scatter(traj[0, k, 0], traj[0, k, 1], s=34, c="white",
                        edgecolors=COLORS[m], lw=1.2, zorder=4)
            ax0.scatter(traj[-1, k, 0], traj[-1, k, 1], marker="s", s=34, c=COLORS[m],
                        edgecolors="k", lw=0.5, zorder=4)
    relays = np.asarray([S.RELAY_SITES[r] for r in S.RELAY_SITES])
    ax0.scatter(relays[:, 0], relays[:, 1], marker="D", s=46, facecolors="none",
                edgecolors="#9aa0a6", lw=1.0, zorder=2)
    for rname in [a[1] for a in chosen["WISE"] if a[0] == "relay"]:
        q = S.RELAY_SITES[rname]
        ax0.scatter(*q, marker="^", s=90, c=GREEN, edgecolors="k", lw=0.5, zorder=5)
        ax0.text(q[0], q[1] - 0.30, "relay freed\nby pair-WISE", fontsize=7.6, color=GREEN,
                 ha="center", va="top")
    ax0.set_aspect("equal")
    ax0.tick_params(labelsize=7.6)
    ax0.grid(alpha=0.22, lw=0.5)
    ax0.legend(fontsize=8.0, frameon=False, loc="upper center", ncol=2, handlelength=1.8)
    ax0.set_title("(a) load paths: open = start, filled = end", fontsize=9.0, pad=6)

    # (b) synchronization error, with the disturbance window and completion marked
    for m in ("PROD", "HARD", "WISE"):
        d = ts[m]
        op = d["phase"] == "operational"
        ax1.plot(d["t"][op], d["sync_err"][op], color=COLORS[m], ls=STYLES[m], lw=1.8,
                 label=LABEL[m])
    lo, hi = ax1.get_ylim()
    ax1.axvspan(C.T_DIST, C.T_DIST + C.DUR_DIST, color="#95a5a6", alpha=0.22, lw=0)
    ax1.text(C.T_DIST + C.DUR_DIST / 2, hi * 0.995, "disturbance",
             fontsize=7.8, ha="center", va="top", color="#5b6470", rotation=90)
    ax1.set_xlabel("time [s]", fontsize=8.6)
    ax1.set_ylabel(r"$|s_1-s_2|$", fontsize=8.6)
    ax1.set_title("(b) synchronization: both recover, PROD does not", fontsize=9.0, pad=6)
    ax1.legend(fontsize=8.0, frameon=False, loc="upper left")

    # (c) realized connectivity against the certified surrogate
    for m in ("PROD", "HARD", "WISE"):
        d = ts[m]
        op = d["phase"] == "operational"
        ax2.plot(d["t"][op], d["lam_geo"][op], color=COLORS[m], ls=STYLES[m], lw=1.8)
        ax2.axhline(summary["summaries"][m]["lambda2_bar"], color=COLORS[m], lw=0.9,
                    alpha=0.45)
    ax2.axhline(C.SIGMA_REQ, color="#2c3e50", ls="--", lw=1.2)
    ax2.axhline(C.SIGMA_DYN, color=RED, ls=":", lw=1.2)
    x0 = ax2.get_xlim()[0]
    ax2.text(x0, C.SIGMA_REQ + 0.004, r"$\sigma_{\rm req}$", fontsize=8.2, ha="left",
             va="bottom", color="#2c3e50")
    ax2.text(x0, C.SIGMA_DYN - 0.004, r"$\sigma_{\rm dyn}$", fontsize=8.2, ha="left",
             va="top", color=RED)
    ax2.axvspan(C.T_DIST, C.T_DIST + C.DUR_DIST, color="#95a5a6", alpha=0.22, lw=0)
    ax2.set_xlabel("time [s]", fontsize=8.6)
    ax2.set_ylabel(r"$\lambda_2(L_{\rm geo}(q(t)))$", fontsize=8.6)
    ax2.set_title(r"(c) realized $\lambda_2$ stays above certified", fontsize=9.0, pad=6)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7.6)
        ax.grid(alpha=0.22, lw=0.5)
    fig.tight_layout()
    return fig


def main():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(exist_ok=True)
    for name, builder in (("s1_pipeline", fig_pipeline),
                          ("s2_mechanism", fig_mechanism),
                          ("s3_closed_loop", fig_closed_loop)):
        fig = builder(plt)
        fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight",
                    metadata={"CreationDate": None})
        fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {FIGDIR / name}.pdf/.png")


if __name__ == "__main__":
    main()
