"""Figure 1: the geometry and the pipeline, with no robots in it.

The point of this figure is that a reader should understand the contribution before
meeting the physical scenario: Stage 1 fixes the productive aggregate and leaves a whole
fiber of equally optimal compositions; a neutral direction moves along that fiber without
touching V while raising lambda_2; Gamma_E decides locally whether such a direction exists
and certifies a global maximum when it does not; Stage 2 computes the maximiser.

Writes ``paper/figures/fig_concept.pdf``.

    python -m experiments.robot_closed_loop.make_concept
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

FIGDIR = ROOT / "figures"
PAPERFIG = ROOT / "paper" / "figures"

RED, GREEN, ORANGE = "#c0392b", "#1e7a46", "#d68910"
INK, MUTED, PAPER = "#222222", "#7f8c8d", "#f2f4f6"


def build(plt):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

    fig = plt.figure(figsize=(7.16, 0.96))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.0], wspace=0.06)
    axG, axP = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ------------------------------------------------------------------ geometry
    face = np.array([[0.05, 0.13], [0.99, 0.09], [0.93, 0.90], [0.11, 0.83]])
    poly = Polygon(face, closed=True, fc=PAPER, ec="#9aa0a6", lw=1.0, zorder=1)
    axG.add_patch(poly)
    # the polytope needs no separate label: the fiber caption already names $X_{\rm f}$

    # lambda_2 contours, clipped to the polytope so they read as level sets of it
    for c, a in zip(np.linspace(0.15, 0.95, 5), np.linspace(0.10, 0.30, 5), strict=True):
        ln, = axG.plot([c - 0.05, c + 0.42], [0.86, 0.10], color=GREEN, lw=0.7,
                       alpha=a, zorder=2)
        ln.set_clip_path(poly)
    axG.text(0.90, 0.245, r"$\lambda_2$ increases", fontsize=6.0, color=GREEN,
             rotation=-58, ha="center", va="center")

    p0, p1 = np.array([0.22, 0.46]), np.array([0.74, 0.60])
    axG.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=2.2, zorder=4,
             solid_capstyle="round")
    axG.text(0.47, 0.745, r"fiber $\mathcal{E}=\{z\in X_{\rm f}: Bz=y^\star\}$",
             fontsize=6.4, color=INK, ha="center")

    axG.add_patch(FancyArrowPatch(p0 + 0.10 * (p1 - p0), p1 - 0.10 * (p1 - p0),
                                  arrowstyle="-|>", mutation_scale=10, lw=1.8,
                                  color=GREEN, zorder=5))
    axG.scatter(*p0, s=44, c=RED, edgecolors="k", lw=0.4, zorder=6)
    axG.scatter(*p1, s=44, c=GREEN, edgecolors="k", lw=0.4, zorder=6)
    axG.text(p0[0] - 0.055, p0[1] + 0.015, r"$z^{\rm prod}$, $\Gamma_{\mathcal{E}}>0$",
             fontsize=6.2, color=RED, ha="right", va="center")
    axG.text(p1[0] + 0.05, p1[1] + 0.015, r"$z^\star_{\rm WISE}$, $\Gamma_{\mathcal{E}}=0$",
             fontsize=6.2, color=GREEN, ha="left", va="center")
    axG.text(0.47, 0.285, r"$d^\star$: $Bd^\star=0$ (free), $D\lambda_2[d^\star]>0$",
             fontsize=6.2, color=GREEN, ha="center", va="top")

    axG.set_xlim(-0.15, 1.22)
    axG.set_ylim(0.06, 0.99)
    axG.axis("off")

    # ------------------------------------------------------------------ pipeline
    boxes = [("Stage 1", r"$\max V$ on $X_{\rm f}$", r"gives $y^\star$", False),
             ("fiber $\\mathcal{E}$", r"$Bz=y^\star$", "equally optimal", False),
             (r"$\Gamma_{\mathcal{E}}$", "directional test",
              r"local $\Rightarrow$ global", True),
             ("Stage 2", r"$\max\lambda_2$ on $\mathcal{E}$", r"$\Delta V=0$", True),
             ("recovery", "round, repair,", "re-certify", False)]
    w, h, gap, y = 0.183, 0.30, 0.024, 0.50
    for i, (t1, t2, t3, hi) in enumerate(boxes):
        x = i * (w + gap)
        axP.add_patch(FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.008,rounding_size=0.03",
                                     fc="#e8f2ec" if hi else PAPER,
                                     ec=GREEN if hi else "#9aa0a6",
                                     lw=1.2 if hi else 0.8, zorder=3))
        axP.text(x + w / 2, y + 0.205, t1, fontsize=6.4, ha="center", va="center",
                 color=GREEN if hi else INK, weight="bold", zorder=4)
        axP.text(x + w / 2, y + 0.115, t2, fontsize=5.5, ha="center", va="center",
                 color=INK, zorder=4)
        axP.text(x + w / 2, y + 0.040, t3, fontsize=5.2, ha="center", va="center",
                 color=MUTED, zorder=4)
        if i < len(boxes) - 1:
            axP.add_patch(FancyArrowPatch((x + w + 0.004, y + h / 2),
                                          (x + w + gap - 0.004, y + h / 2),
                                          arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                          color=INK, zorder=3))

    x_g = 2 * (w + gap)
    axP.add_patch(FancyArrowPatch((x_g + w / 2, y - 0.01), (x_g + w / 2, y - 0.115),
                                  arrowstyle="-|>", mutation_scale=7, lw=0.9, color=GREEN))
    axP.text(x_g + w / 2 + 0.012, y - 0.145,
             r"$\Gamma_{\mathcal{E}}>0$: a better-connected optimum exists;   "
             r"$=0$: this point maximizes $\lambda_2$ on $\mathcal{E}$",
             fontsize=6.0, color=GREEN, ha="center", va="top")
    axP.set_xlim(-0.02, 5 * w + 4 * gap + 0.02)
    axP.set_ylim(0.02, 1.02)
    axP.axis("off")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    return fig


def main():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig = build(plt)
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(PAPERFIG / "fig_concept.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(FIGDIR / "concept.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PAPERFIG / 'fig_concept.pdf'}")


if __name__ == "__main__":
    main()
