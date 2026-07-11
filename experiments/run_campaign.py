"""Run the WISE campaign and render every paper figure from real simulations.

    python experiments/run_campaign.py            # all figures + table
    python experiments/run_campaign.py --quick    # coarse grid (fast smoke)

Outputs land in paper/figures/ (fig1..fig4 + ablation_table.tex). Everything is
seeded, so re-running reproduces the numbers exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from wise_mr import baselines, metrics, primal_dual as pd, scenarios  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.linewidth": 0.7,
})

C_NOPHYS, C_PHYS, C_WISE, C_ACCENT = "#d1495b", "#edae49", "#66a182", "#2e4057"
C_LONG, C_SHORT = "#2e6f9e", "#c98a3b"
METHODS = ["scalar_capacity", "wrench_only", "connectivity_only",
           "wise_primal_dual", "centralized_wise"]
LABELS = {"scalar_capacity": "scalar", "wrench_only": "wrench-only",
          "connectivity_only": "conn.-only", "wise_primal_dual": "WISE",
          "centralized_wise": "oracle"}


# ===========================================================================
# Fig 1 -- concept schematic
# ===========================================================================
def fig_concept():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    titles = ["(a) Wrench-feasible, network fragmented",
              "(b) WISE: executes and computes itself"]
    for ax, title, wise in zip(axes, titles, [False, True]):
        ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=8.5)
        for cx in (2.0, 8.0):
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx - 1.5, 1.0), 3.0, 3.0,
                boxstyle="round,pad=0.05,rounding_size=0.25",
                fc="#f2f2f2", ec="#bdbdbd", lw=0.8, zorder=0))
        ax.add_patch(mpatches.Rectangle((7.1, 2.1), 1.8, 0.9, fc=C_ACCENT, ec="k", lw=0.6))
        ax.text(8.0, 2.55, "load", color="w", ha="center", va="center", fontsize=7)
        left = [(1.4, 3.2), (2.6, 3.2), (2.0, 1.7)]
        onload = [(7.3, 3.4), (8.0, 3.6), (8.7, 3.4)]
        for (rx, ry) in left + onload:
            ax.add_patch(mpatches.Circle((rx, ry), 0.28, fc="#8ecae6", ec="k", lw=0.6, zorder=3))
        for a, b in [(left[0], left[1]), (left[0], left[2]), (left[1], left[2]),
                     (onload[0], onload[1]), (onload[1], onload[2])]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#8a8a8a", lw=0.8, zorder=1)
        if wise:
            relay = (5.0, 3.0)
            ax.add_patch(mpatches.Circle(relay, 0.30, fc=C_WISE, ec="k", lw=0.7, zorder=3))
            ax.text(relay[0], relay[1] + 0.55, "relay", ha="center", fontsize=6.5, color=C_ACCENT)
            ax.plot([left[1][0], relay[0]], [left[1][1], relay[1]], color=C_WISE, lw=1.4, zorder=2)
            ax.plot([relay[0], onload[0][0]], [relay[1], onload[0][1]], color=C_WISE, lw=1.4, zorder=2)
            ax.text(5.0, 0.55, r"$\lambda_2(L(x)) \geq \sigma^\star$", ha="center", color=C_WISE, fontsize=8)
        else:
            ax.plot([3.5, 6.5], [3.0, 3.0], color=C_NOPHYS, lw=1.2, ls=(0, (2, 2)), zorder=2)
            ax.text(5.0, 3.35, r"$\times$", ha="center", color=C_NOPHYS, fontsize=13)
            ax.text(5.0, 0.55, r"$\lambda_2(L(x)) < \sigma^\star$", ha="center", color=C_NOPHYS, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig1_concept.pdf"); plt.close(fig)


# ===========================================================================
# Fig 2 -- convergence of the distributed primal-dual (the "self-computation")
# ===========================================================================
def fig_convergence():
    prob = scenarios.two_region(seed=3, N=12, nu=0.42, tau_d=3.0, bridge_gain=3.0)
    res = pd.solve(prob, config=pd.PDConfig(max_iters=6000, record_every=5))
    h = res.history
    t = np.arange(len(h["lambda2"])) * 5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    # (a) constraints
    ax1.plot(t, h["lambda2"], color=C_LONG, lw=1.6, label=r"$\lambda_2(L(x))$")
    ax1.axhline(prob.sigma, color=C_ACCENT, lw=0.9, ls=(0, (4, 2)), label=r"$\sigma^\star$")
    ax1.set_xlabel("iteration"); ax1.set_ylabel(r"connectivity $\lambda_2$", color=C_LONG)
    ax1.tick_params(axis="y", labelcolor=C_LONG)
    axr = ax1.twinx()
    axr.plot(t, h["resid"], color=C_NOPHYS, lw=1.6, label="wrench residual")
    axr.set_ylabel("wrench residual", color=C_NOPHYS); axr.tick_params(axis="y", labelcolor=C_NOPHYS)
    ax1.set_title("(a) both constraints satisfied", fontsize=8.5)
    ax1.legend(loc="center right", framealpha=0.9)
    # (b) prices + relay role split
    ax2.plot(t, h["pi"], color=C_ACCENT, lw=1.6, label=r"info price $\pi$")
    ax2.set_xlabel("iteration"); ax2.set_ylabel(r"price $\pi$", color=C_ACCENT)
    ax2.tick_params(axis="y", labelcolor=C_ACCENT)
    axb = ax2.twinx()
    axb.plot(t, h["relay_long"], color=C_LONG, lw=1.6, label="relay mass: long-range")
    axb.plot(t, h["relay_short"], color=C_SHORT, lw=1.6, ls=(0, (3, 2)), label="relay mass: short-range")
    axb.set_ylabel("relay mass")
    ax2.set_title("(b) prices and emergent relay roles", fontsize=8.5)
    lines = ax2.get_lines() + axb.get_lines()
    ax2.legend(lines, [ln.get_label() for ln in lines], loc="center right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(FIG / "fig2_convergence.pdf"); plt.close(fig)
    return res


# ===========================================================================
# Fig 3 -- certified-rate phase diagram over (nu, tau_d), seed-averaged
# ===========================================================================
def fig_phase(seeds, nus, taus):
    rate = np.zeros((len(taus), len(nus)))
    for a, tau in enumerate(taus):
        for b, nu in enumerate(nus):
            flags = []
            for sd in seeds:
                prob = scenarios.two_region(seed=sd, N=12, nu=nu, tau_d=tau, bridge_gain=3.0)
                res = baselines.wise_primal_dual(prob, max_iters=3000)
                flags.append(metrics.certified(prob, res.x))
            rate[a, b] = np.mean(flags)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    pc = ax.pcolormesh(nus, taus, rate, cmap="YlGn", vmin=0, vmax=1, shading="auto", rasterized=True)
    ax.contour(nus, taus, rate, levels=[0.5], colors=[C_ACCENT], linewidths=1.0, linestyles="dashed")
    cb = fig.colorbar(pc, ax=ax, pad=0.02); cb.set_label("certified service rate")
    ax.set_xlabel(r"long-range fraction $\nu$"); ax.set_ylabel(r"torque demand $\tau_d$")
    ax.set_title("phase diagram (12 seeds/cell)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig3_phase.pdf"); plt.close(fig)
    return rate


# ===========================================================================
# Fig 4 -- ablation bars + emergent-role scatter
# ===========================================================================
def fig_ablation_roles(seeds):
    # ablation at a WISE-feasible operating point
    cert = {m: [] for m in METHODS}
    for sd in seeds:
        prob = scenarios.two_region(seed=sd, N=12, nu=0.5, tau_d=3.0, bridge_gain=3.0)
        for m in METHODS:
            res = baselines.run(m, prob)
            cert[m].append(metrics.certified(prob, res.x))
    means, los, his = [], [], []
    for m in METHODS:
        mean, lo, hi = metrics.paired_bootstrap_ci(cert[m], seed=1)
        means.append(mean * 100); los.append((mean - lo) * 100); his.append((hi - mean) * 100)

    # role emergence: relay share vs range at WISE solution, pooled over seeds
    ranges, relays, longs = [], [], []
    for sd in seeds:
        prob = scenarios.two_region(seed=sd, N=12, nu=0.5, tau_d=3.0, bridge_gain=3.0)
        res = baselines.wise_primal_dual(prob, max_iters=4000)
        ranges.append(prob.meta["r"]); relays.append(prob.relay_shares(res.x))
        longs.append(prob.meta["is_long"])
    ranges = np.concatenate(ranges); relays = np.concatenate(relays); longs = np.concatenate(longs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    cols = [C_NOPHYS, C_PHYS, "#8a8a8a", C_WISE, C_ACCENT]
    ax1.bar(range(len(METHODS)), means, yerr=[los, his], color=cols, capsize=2.5,
            error_kw=dict(lw=0.8))
    ax1.set_xticks(range(len(METHODS)))
    ax1.set_xticklabels([LABELS[m] for m in METHODS], rotation=25, ha="right")
    ax1.set_ylabel("certified service rate (\\%)"); ax1.set_ylim(0, 100)
    ax1.set_title("(a) constraint ablation", fontsize=8.5)

    ax2.scatter(ranges[~longs], relays[~longs], s=16, c=C_SHORT, edgecolors="k",
                linewidths=0.3, label="short-range", alpha=0.8)
    ax2.scatter(ranges[longs], relays[longs], s=16, c=C_LONG, edgecolors="k",
                linewidths=0.3, label="long-range", alpha=0.8)
    ax2.set_xlabel(r"communication range $r_i$"); ax2.set_ylabel("relay share $x_{i,\\mathrm{relay}}$")
    ax2.set_title("(b) emergent relay roles", fontsize=8.5)
    ax2.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout(); fig.savefig(FIG / "fig4_ablation_roles.pdf"); plt.close(fig)
    return cert


# ===========================================================================
# Ablation table (real numbers)
# ===========================================================================
def ablation_table(seeds):
    rows_data = {m: dict(cert=[], resid=[], marg=[]) for m in METHODS}
    for sd in seeds:
        prob = scenarios.two_region(seed=sd, N=12, nu=0.5, tau_d=3.0, bridge_gain=3.0)
        for m in METHODS:
            res = baselines.run(m, prob)
            rows_data[m]["cert"].append(metrics.certified(prob, res.x))
            rows_data[m]["resid"].append(prob.max_residual(res.x))
            rows_data[m]["marg"].append(prob.info_margin(res.x))
    disp = {"scalar_capacity": "scalar\\_capacity", "wrench_only": "wrench\\_only",
            "connectivity_only": "connectivity\\_only",
            "wise_primal_dual": "\\textbf{wise\\_pd}", "centralized_wise": "centralized (oracle)"}
    lines = []
    for m in METHODS:
        c = np.mean(rows_data[m]["cert"]) * 100
        r = np.mean(rows_data[m]["resid"]); g = np.mean(rows_data[m]["marg"])
        lines.append(f"{disp[m]} & {r:.2f} & {g:+.2f} & {c:.1f}\\% \\\\")
    tex = ("% Auto-generated by experiments/run_campaign.py -- do not edit by hand.\n"
           "% Real simulation, 12 robots, nu=0.5, tau_d=3.0, averaged over seeds.\n"
           "\\setlength{\\tabcolsep}{5pt}\n"
           "\\begin{tabular}{lccc}\n\\hline\n"
           "Method & Wr.\\ res.\\ $\\downarrow$ & $\\lambda_2\\!-\\!\\sigma^\\star$ $\\uparrow$ "
           "& Cert.\\ succ.\\ $\\uparrow$\\\\\n\\hline\n"
           + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n")
    (FIG / "ablation_table.tex").write_text(tex, encoding="utf-8")
    print("ablation table:")
    for m in METHODS:
        print(f"  {m:20s} cert={np.mean(rows_data[m]['cert'])*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = list(range(6 if args.quick else 16))
    nus = np.linspace(0.0, 1.0, 6 if args.quick else 11)
    taus = np.linspace(0.5, 7.0, 6 if args.quick else 9)

    print("fig1 concept ...");        fig_concept()
    print("fig2 convergence ...");    fig_convergence()
    print("fig3 phase diagram ...");  fig_phase(list(range(6 if args.quick else 12)), nus, taus)
    print("fig4 ablation+roles ...");  fig_ablation_roles(seeds)
    print("ablation table ...");      ablation_table(seeds)
    print("done ->", FIG)


if __name__ == "__main__":
    main()
