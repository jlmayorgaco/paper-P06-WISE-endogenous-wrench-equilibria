"""E-rounding: structured integer recovery beats independent rounding.

Four recovery methods from the fluid WISE optimum z*:
  1. argmax              -- each robot commits to its argmax action;
  2. independent RR      -- each robot samples its own action independently;
  3. dependent RR        -- occupancy-aware sampling (a contact slot is taken at most once;
                            collisions resample among still-available actions);
  4. dependent + repair  -- after dependent RR, a greedy neutral-repair local search moves
                            marginal lifters/idlers to relay while keeping wrench feasibility,
                            raising lambda_2 toward sigma_req; then exact re-certification.

We report, over 30 seeds (N=12): single-draw certified rate, best-of-30 rate, the median
number of draws to first success, and the productive (rounding) gap. A CDF of draws-to-
success goes to paper/figures/fig_rounding.pdf. Structured recovery should reach 30/30 with
fewer draws than independent rounding.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import baselines, metrics, scenarios  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

N_DRAWS = 30
SEEDS = list(range(30))


def _wrench_ok(prob, x):
    return float(prob.max_residual(x)) <= 0.05


def _dependent_draw(prob, z, rng):
    """Occupancy-aware rounding: each contact slot is claimed by at most one robot."""
    A, H, M = prob.A, prob.H, prob.M
    slot_taken = set()
    x = np.zeros((prob.N, A))
    order = rng.permutation(prob.N)
    for i in order:
        p = np.maximum(z[i], 0.0); p = p / p.sum() if p.sum() > 0 else np.ones(A) / A
        avail = np.ones(A, bool)
        for a in range(M * H):                       # contact slots must be exclusive
            if a in slot_taken:
                avail[a] = False
        pa = p * avail
        a = int(rng.choice(A, p=pa / pa.sum())) if pa.sum() > 0 else int(np.argmax(p))
        x[i, a] = 1.0
        if a < M * H:
            slot_taken.add(a)
    return x


def _repair(prob, x, max_steps=12):
    """Greedy neutral repair: move a marginal lifter/idler to relay to raise lambda_2 while
    keeping wrench feasibility. Trades productive value for connectivity (a rounding loss)."""
    x = x.copy()
    ridx = prob.relay_index
    for _ in range(max_steps):
        if metrics.certified(prob, x):
            return x
        base = prob.lambda2(x)
        best, best_lam = None, base
        for i in range(prob.N):
            if x[i, ridx] == 1.0:
                continue
            cand = x[i].copy(); x[i] = 0.0; x[i, ridx] = 1.0
            if _wrench_ok(prob, x) and prob.lambda2(x) > best_lam + 1e-9:
                best, best_lam = i, prob.lambda2(x)
            x[i] = cand                              # revert trial
        if best is None:
            break
        x[best] = 0.0; x[best, ridx] = 1.0
    return x


def _method(prob, z, method, rng):
    """Return (x_int, draws_to_success or None)."""
    if method == "argmax":
        x = metrics.round_argmax(prob, z)
        return x, (1 if metrics.certified(prob, x) else None)
    for k in range(1, N_DRAWS + 1):
        if method == "independent":
            x = np.zeros_like(z)
            for i in range(prob.N):
                p = np.maximum(z[i], 0.0); x[i, rng.choice(prob.A, p=p / p.sum())] = 1.0
        elif method == "dependent":
            x = _dependent_draw(prob, z, rng)
        elif method == "dependent+repair":
            x = _repair(prob, _dependent_draw(prob, z, rng))
        if metrics.certified(prob, x):
            return x, k
    return x, None


def run(seeds=SEEDS):
    methods = ["argmax", "independent", "dependent", "dependent+repair"]
    stats = {m: dict(single=0, best=0, draws=[], gap=[]) for m in methods}
    for sd in seeds:
        prob = scenarios.two_region(seed=sd, N=12, nu=0.5, tau_d=3.0, bridge_gain=3.0)
        z = np.maximum(baselines.wise_primal_dual(prob, max_iters=4000).x, 0.0)
        v_relax = float(prob.productive_value(z))
        for m in methods:
            rng = np.random.default_rng(sd + 5000)
            # single-draw success = first draw certified
            x1, _ = _method_single(prob, z, m, np.random.default_rng(sd + 7000))
            stats[m]["single"] += int(metrics.certified(prob, x1))
            x, k = _method(prob, z, m, rng)
            ok = metrics.certified(prob, x)
            stats[m]["best"] += int(ok)
            if ok:
                stats[m]["draws"].append(k)
                stats[m]["gap"].append((v_relax - float(prob.productive_value(x)))
                                       / (abs(v_relax) + 1e-9))
    rows = []
    for m in methods:
        s = stats[m]
        rows.append(dict(
            method=m, n=len(seeds),
            single_rate=s["single"] / len(seeds),
            best_of_30=s["best"] / len(seeds),
            median_draws=float(np.median(s["draws"])) if s["draws"] else float("nan"),
            gap_mean=float(np.mean(s["gap"])) if s["gap"] else float("nan")))
        print(f"{m:>18}: single={rows[-1]['single_rate']:.0%} "
              f"best-of-30={s['best']}/{len(seeds)} "
              f"med_draws={rows[-1]['median_draws']:.0f} gap={rows[-1]['gap_mean']:.3f}")

    with open(GEN / "rounding.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    _figure(stats, methods, len(seeds))
    return rows


def _method_single(prob, z, method, rng):
    """One draw only (for the single-draw rate)."""
    if method == "argmax":
        return metrics.round_argmax(prob, z), None
    if method == "independent":
        x = np.zeros_like(z)
        for i in range(prob.N):
            p = np.maximum(z[i], 0.0); x[i, rng.choice(prob.A, p=p / p.sum())] = 1.0
        return x, None
    if method == "dependent":
        return _dependent_draw(prob, z, rng), None
    return _repair(prob, _dependent_draw(prob, z, rng)), None


def _figure(stats, methods, n):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    cols = {"argmax": "#888", "independent": "#c0392b", "dependent": "#e08000",
            "dependent+repair": "#2e8b57"}
    grid = np.arange(1, N_DRAWS + 1)
    for m in methods:
        d = np.array(stats[m]["draws"])
        cdf = np.array([(d <= k).sum() / n for k in grid]) if d.size else np.zeros_like(grid, float)
        ax.step(grid, cdf, where="post", color=cols[m], lw=1.4, label=m)
    ax.set_xlabel("draws $R$", fontsize=8)
    ax.set_ylabel(r"P(success $\leq R$)", fontsize=8)
    ax.set_title("recovery vs. draws", fontsize=8.5)
    ax.set_ylim(0, 1.02); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    fig.savefig(FIG / "fig_rounding.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    run()
