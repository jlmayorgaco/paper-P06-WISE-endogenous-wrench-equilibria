"""TASK 12 (Option B): obtain integer assignments from the fluid WISE optimum and
re-certify wrench + information feasibility numerically. No O(1/N) claim; report the
integer success rate under argmax and randomized rounding, and the integrality gap.
Writes generated/integer_recovery.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wise_mr import baselines, metrics, scenarios  # noqa: E402

GEN = Path(__file__).resolve().parents[1] / "generated"
GEN.mkdir(exist_ok=True)


def main() -> None:
    seeds = list(range(16))
    fluid, arg, rand, gaps = [], [], [], []
    for sd in seeds:
        prob = scenarios.two_region(seed=sd, N=12, nu=0.5, tau_d=3.0, bridge_gain=3.0)
        res = baselines.wise_primal_dual(prob, max_iters=4000)
        fluid.append(metrics.certified(prob, res.x))
        xa = metrics.round_argmax(prob, res.x)
        arg.append(metrics.certified(prob, xa))
        _, ok = metrics.round_randomized(prob, res.x, n_draws=30, seed=sd)
        rand.append(ok)
        gaps.append(float(prob.potential(res.x) - prob.potential(xa)))
    cert = {
        "n_runs": len(seeds), "N_robots": 12,
        "note": "N=12 physical demonstration; each robot is integer (one action).",
        "fluid_certified_rate": round(float(np.mean(fluid)), 4),
        "integer_argmax_rate": round(float(np.mean(arg)), 4),
        "integer_randomized_rate": round(float(np.mean(rand)), 4),
        "integrality_gap_welfare_mean": round(float(np.mean(gaps)), 4),
    }
    (GEN / "integer_recovery.json").write_text(json.dumps(cert, indent=2), encoding="utf-8")
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
