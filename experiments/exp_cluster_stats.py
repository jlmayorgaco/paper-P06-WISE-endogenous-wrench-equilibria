"""E-cluster: correct the unit of analysis for the integer-recovery campaign.

The recovery campaign draws ``30 instances x 30 random processing orders = 900``
recoveries. The 30 draws *within* an instance share the same problem, so they are
**not** independent Bernoulli trials: a Wilson interval computed on 900 pooled
draws understates the true uncertainty. The experimental unit is the *instance*.

This script recomputes the campaign keeping the cluster structure and reports

  * per-instance success rate (30 values);
  * macro mean (mean of instance rates) alongside the pooled micro rate;
  * median / IQR / min / max across instances;
  * a **cluster bootstrap** 95% CI resampling *instances* with replacement;
  * the number of instances succeeding on at least one of their 30 draws;
  * the pooled Wilson interval, reported only as the (anticonservative) contrast.

Writes generated/cluster_stats.json and generated/cluster_per_instance.csv.
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

from wise_mr import baselines, metrics, scenarios  # noqa: E402
from exp_rounding import _dependent_draw, _repair_moves, _wilson  # noqa: E402

GEN = ROOT / "generated"
GEN.mkdir(exist_ok=True)

SEEDS = list(range(30))
R = 30
BOOT = 20000
BOOT_SEED = 20260807          # fixed; recorded in the manifest


def collect(seeds=SEEDS, R=R):
    """Per-instance recovery outcomes, preserving the cluster structure."""
    rows = []
    for sd in seeds:
        prob = scenarios.two_region(seed=sd, N=12, nu=0.5, tau_d=3.0, bridge_gain=3.0)
        z = np.maximum(baselines.wise_primal_dual(prob, max_iters=4000).x, 0.0)
        v_relax = float(prob.productive_value(z))
        pre = post = zero = 0
        losses = []
        for r in range(R):
            rng = np.random.default_rng(1000 * sd + r)      # same stream as campaign()
            x0 = _dependent_draw(prob, z, rng)
            pre += int(metrics.certified(prob, x0))
            xr, _ = _repair_moves(prob, x0.copy())
            ok = metrics.certified(prob, xr)
            post += int(ok)
            if ok:
                loss = (v_relax - float(prob.productive_value(xr))) / (abs(v_relax) + 1e-9)
                losses.append(loss)
                zero += int(loss < 1e-4)
        rows.append({
            "seed": sd, "draws": R,
            "pre_successes": pre, "post_successes": post,
            "pre_rate": pre / R, "post_rate": post / R,
            "zero_cost": zero,
            "median_loss": float(np.median(losses)) if losses else float("nan"),
        })
        print(f"  instance {sd:2d}: pre {pre:2d}/{R}  post {post:2d}/{R}  "
              f"zero-cost {zero:2d}")
    return rows


def cluster_bootstrap(rates: np.ndarray, boot: int = BOOT, seed: int = BOOT_SEED):
    """95% CI for the macro mean, resampling *instances* with replacement."""
    rng = np.random.default_rng(seed)
    n = rates.size
    draws = rng.integers(0, n, size=(boot, n))
    means = rates[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    print(f"cluster campaign: {len(SEEDS)} instances x {R} orders")
    rows = collect()

    post = np.array([r["post_successes"] for r in rows], dtype=float)
    rates = post / R
    n_inst, total = len(rows), len(rows) * R
    micro = post.sum() / total
    macro = float(rates.mean())
    lo_b, hi_b = cluster_bootstrap(rates)
    lo_w, hi_w = _wilson(int(post.sum()), total)
    q1, q3 = np.percentile(rates, [25, 75])

    summary = {
        "n_instances": n_inst,
        "draws_per_instance": R,
        "total_draws": total,
        "unit_of_analysis": "instance (draws are nested within instance)",
        "pooled_successes": int(post.sum()),
        "micro_rate_pooled": micro,
        "macro_rate_mean_of_instances": macro,
        "instance_rate_median": float(np.median(rates)),
        "instance_rate_iqr": [float(q1), float(q3)],
        "instance_rate_min": float(rates.min()),
        "instance_rate_max": float(rates.max()),
        "cluster_bootstrap_95ci": [lo_b, hi_b],
        "bootstrap_resamples": BOOT,
        "bootstrap_seed": BOOT_SEED,
        "instances_with_ge1_success": int((post > 0).sum()),
        "pooled_wilson_95ci_ANTICONSERVATIVE": [lo_w, hi_w],
        "note": ("The pooled Wilson interval treats 900 nested draws as independent and is "
                 "reported only for contrast; the cluster bootstrap over instances is the "
                 "defensible interval."),
    }
    (GEN / "cluster_stats.json").write_text(json.dumps(summary, indent=1))
    with (GEN / "cluster_per_instance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\npooled (micro)  {int(post.sum())}/{total} = {micro:.1%}   "
          f"Wilson [{lo_w:.1%},{hi_w:.1%}]  <- anticonservative")
    print(f"macro (mean of {n_inst} instance rates) = {macro:.1%}   "
          f"cluster bootstrap 95% CI [{lo_b:.1%},{hi_b:.1%}]")
    print(f"instance rates: median {np.median(rates):.1%}  "
          f"IQR [{q1:.1%},{q3:.1%}]  min {rates.min():.1%}  max {rates.max():.1%}")
    print(f"instances with >=1 success: {int((post > 0).sum())}/{n_inst}")


if __name__ == "__main__":
    main()
