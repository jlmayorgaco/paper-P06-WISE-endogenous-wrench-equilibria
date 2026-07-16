"""Central evidence figure (paper Fig. 2): the equilibrium fiber, certified and physical.

Row 1 (from exp_fiber): a genuine fiber direction holds V and the served aggregate at
machine precision while lambda_2(Lbar) crosses sigma_req. Row 2 (from exp_spatial): two
integer compositions on that same fiber -- identical Bz=y* and V* -- one fragmenting the
physical graph, one bridging it, straight from the solver->rounding->pose->L_geo pipeline.

Run exp_fiber.py and exp_spatial.py first (they write the CSV/JSON this reads and
regenerate the assignments). Writes paper/figures/fig_central.pdf.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp_spatial as sp  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"


def main():
    with open(GEN / "fiber_sweep.csv") as f:
        rows = list(csv.DictReader(f))
    fib = json.loads((GEN / "fiber_certificate.json").read_text())
    alpha = np.array([float(r["alpha"]) for r in rows])
    an = (alpha - alpha.min()) / (np.ptp(alpha) + 1e-12)
    Vd = np.array([float(r["V_minus_Vstar"]) for r in rows])
    drift = np.array([float(r["agg_drift"]) for r in rows])
    lam = np.array([float(r["lambda2"]) for r in rows])
    sig_f = float(fib["sigma_req"])

    rec, ctx = sp.select()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    sig = rec["sigma_req"]
    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.42)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1]); c = fig.add_subplot(gs[2])

    # (a) fiber certificate: lambda_2 crosses sigma_req; residuals annotated as a band
    a.plot(an, lam, color="#2e8b57", lw=1.8)
    a.axhline(sig_f, color="#c03030", ls="--", lw=1.0)
    a.text(0.02, sig_f, r"$\sigma_{\rm req}$", color="#c03030", va="bottom", fontsize=7)
    a.fill_between(an, sig_f, lam, where=(lam >= sig_f), color="#2e8b57", alpha=0.12)
    a.set_xlabel(r"position on fiber $\alpha$", fontsize=8)
    a.set_ylabel(r"$\lambda_2(\bar L(z(\alpha)))$", color="#2e8b57", fontsize=8)
    a.set_title(rf"(a) certified fiber (dim $E={fib['dim_E']}$)", fontsize=8.5)
    a.text(0.03, 0.97, r"$\max|V-V^\star|,\,\max\|Bz-y^\star\|<10^{-8}$",
           transform=a.transAxes, va="top", ha="left", fontsize=6.2,
           bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="#bbb", lw=0.5))
    a.set_box_aspect(7.2 / 9.2)                      # same box size as the spatial panels

    # (b,c) comparative-advantage exchange on the same fiber
    sp._draw(b, ctx["prob"], ctx["role_u"], ctx["pos_u"], ctx["mask_u"],
             "(b) long lifts", rec["lambda2_geo_unsafe"], sig, ctx["pair"])
    sp._draw(c, ctx["prob"], ctx["role_w"], ctx["pos_w"], ctx["mask_w"],
             "(c) WISE: long relays", rec["lambda2_geo_wise"], sig, ctx["pair"])
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="^", ls="", mfc="w", mec="k", label="long-range"),
        Line2D([], [], marker="o", ls="", mfc="w", mec="k", label="short-range"),
        Line2D([], [], marker="s", ls="", mfc="#c0392b", mec="k", label="lift"),
        Line2D([], [], marker="s", ls="", mfc="#2e8b57", mec="k", label="relay"),
        Line2D([], [], marker="s", ls="", mfc="#8a8a8a", mec="k", label="idle"),
        Line2D([], [], marker="D", ls="", mfc="none", mec="#2e8b57", label="relay site"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6.5, frameon=False,
               handletextpad=0.25, columnspacing=1.0, bbox_to_anchor=(0.5, -0.02))
    a.tick_params(labelsize=7)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(FIG / "fig_central.pdf", metadata={"CreationDate": None}, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_central.pdf (dim E={fib['dim_E']}, "
          f"lambda2_unsafe={rec['lambda2_geo_unsafe']:.2f}, "
          f"lambda2_wise={rec['lambda2_geo_wise']:.2f}, sigma={sig})")


if __name__ == "__main__":
    main()
