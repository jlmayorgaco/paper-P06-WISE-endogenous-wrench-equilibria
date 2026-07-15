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

    fig = plt.figure(figsize=(7.0, 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.28)
    a = fig.add_subplot(gs[0, 0]); ar = a.twinx()
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0]); d = fig.add_subplot(gs[1, 1])

    # (a) fiber neutrality: V-V* and ||Bz-y*|| both at machine precision
    l1, = a.plot(an, Vd, color="#1f5fbf", lw=1.6)
    l2, = ar.plot(an, drift, color="#c0392b", lw=1.6, ls="--")
    a.set_ylim(-1e-8, 1e-8); ar.set_ylim(-1e-8, 1e-8)
    a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ar.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    a.set_ylabel(r"$V(z(\alpha))-V^\star$", color="#1f5fbf", fontsize=8)
    ar.set_ylabel(r"$\|Bz(\alpha)-y^\star\|$", color="#c0392b", fontsize=8)
    a.set_xlabel(r"position on fiber $\alpha$", fontsize=8)
    a.set_title(rf"(a) productive neutrality "
                rf"($<10^{{-8}}$, dim $E={fib['dim_E']}$)", fontsize=8.2)

    # (b) fiber connectivity: lambda_2 crosses sigma_req
    b.plot(an, lam, color="#2e8b57", lw=1.8)
    b.axhline(sig_f, color="#c03030", ls="--", lw=1.0)
    b.text(0.02, sig_f, r"$\sigma_{\rm req}$", color="#c03030", va="bottom", fontsize=7)
    b.fill_between(an, sig_f, lam, where=(lam >= sig_f), color="#2e8b57", alpha=0.12)
    b.set_xlabel(r"position on fiber $\alpha$", fontsize=8)
    b.set_ylabel(r"$\lambda_2(\bar L(z(\alpha)))$", color="#2e8b57", fontsize=8)
    b.set_title("(b) connectivity varies on the fiber", fontsize=8.2)

    # (c,d) spatial compositions on the same fiber
    sig = rec["sigma_req"]
    sp._draw(c, ctx["prob"], ctx["role_u"], ctx["pos_u"], ctx["mask_u"],
             "(c) productive-optimal, unsafe", rec["lambda2_geo_unsafe"], sig)
    sp._draw(d, ctx["prob"], ctx["role_w"], ctx["pos_w"], ctx["mask_w"],
             "(d) WISE", rec["lambda2_geo_wise"], sig)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="^", ls="", mfc="w", mec="k", label="long"),
        Line2D([], [], marker="o", ls="", mfc="w", mec="k", label="short"),
        Line2D([], [], marker="s", ls="", mfc="#c0392b", mec="k", label="lift"),
        Line2D([], [], marker="s", ls="", mfc="#2e8b57", mec="k", label="relay"),
        Line2D([], [], marker="s", ls="", mfc="#8a8a8a", mec="k", label="idle"),
    ]
    d.legend(handles=handles, fontsize=6.0, frameon=False, loc="upper right",
             handletextpad=0.2, borderpad=0.15, labelspacing=0.2, ncol=1)

    for ax in (a, b):
        ax.tick_params(labelsize=7)
    ar.tick_params(labelsize=7)
    fig.savefig(FIG / "fig_central.pdf", metadata={"CreationDate": None}, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_central.pdf (dim E={fib['dim_E']}, "
          f"lambda2_unsafe={rec['lambda2_geo_unsafe']:.2f}, "
          f"lambda2_wise={rec['lambda2_geo_wise']:.2f}, sigma={sig})")


if __name__ == "__main__":
    main()
