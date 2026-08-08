"""PHASE R11 -- paired Monte-Carlo campaign over worlds.

The statistical unit is the **seed/world**, never the time step: each seed draws one
world (load masses, damping, initial pose offsets inside the tubes, disturbance
amplitude and onset) and *every* method is run on that same world, so differences
are paired by construction. Reported: paired medians, paired bootstrap intervals,
success counts and the worst-case seed. No p-values, no time-step pseudoreplication.

Packet loss is deliberately **not** modelled: there is no packet-loss model in the
theory, so a result about it would not be a test of anything stated.

    python -m experiments.robot_closed_loop.run_monte_carlo [n_seeds]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.robot_closed_loop import config as C  # noqa: E402
from experiments.robot_closed_loop import metrics as M  # noqa: E402
from experiments.robot_closed_loop import run_flagship as RF  # noqa: E402
from experiments.robot_closed_loop import scenario as S  # noqa: E402
from experiments.robot_closed_loop import simulator as sim  # noqa: E402

GEN = ROOT / "generated"
N_BOOT = 20000
PAIRED_METRICS = ["max_sync_err", "terminal_sync_err", "pose_rmse1", "pose_rmse2",
                  "settling_time", "peak_wrench_residual_quiet", "alpha_fitted",
                  "info_norm_ratio", "control_effort", "min_lambda2_geo",
                  "min_transfer_margin"]


def paired_bootstrap(diff: np.ndarray, rng, n_boot: int = N_BOOT):
    d = np.asarray([x for x in diff if np.isfinite(x)], float)
    if d.size == 0:
        return float("nan"), (float("nan"), float("nan")), 0
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    meds = np.median(d[idx], axis=1)
    return float(np.median(d)), (float(np.percentile(meds, 2.5)),
                                 float(np.percentile(meds, 97.5))), int(d.size)


def main(n_seeds: int = 30, sigma_req: float = C.SIGMA_REQ):
    bl, chosen, certs = RF.select(sigma_req)
    methods = [m for m in RF.PRIMARY + RF.SUPPLEMENT if m in chosen]
    lbars = {m: S.lbar(chosen[m]) for m in methods}

    rows = []
    for seed in range(1, n_seeds + 1):
        pert = C.SeedPerturbation.draw(seed)
        for m in methods:
            res = sim.simulate(m, chosen[m], lbars[m], seed=seed, pert=pert)
            s = M.summarize(res, certs[m], sigma_req, pert)
            s["drag_multiplier"] = pert.drag_multiplier
            s["t_dist"] = pert.t_dist
            rows.append(s)
        print(f"seed {seed:3d}/{n_seeds} done", flush=True)

    with (GEN / "robot_monte_carlo_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = {m: {r["seed"]: r for r in rows if r["method"] == m} for m in methods}
    rng = np.random.default_rng(4242)
    report = {"n_seeds": n_seeds, "sigma_dyn": C.SIGMA_DYN, "sigma_req": sigma_req,
              "unit": "seed/world (paired); no time-step pseudoreplication",
              "success_counts": {m: int(sum(v["success"] for v in by[m].values()))
                                 for m in methods},
              "paired": {}}
    for ref in ("PROD", "HARD"):
        if ref not in by or "WISE" not in by:
            continue
        block = {}
        for key in PAIRED_METRICS:
            diff = np.array([by["WISE"][s][key] - by[ref][s][key]
                             for s in sorted(by["WISE"])], float)
            med, ci, n = paired_bootstrap(diff, rng)
            worst = max(sorted(by["WISE"]),
                        key=lambda s: (by["WISE"][s][key] - by[ref][s][key])
                        if np.isfinite(by["WISE"][s][key] - by[ref][s][key]) else -np.inf)
            block[key] = {"paired_median_WISE_minus_ref": med, "bootstrap_ci95": ci,
                          "n_finite_pairs": n, "worst_case_seed_for_WISE": int(worst)}
        report["paired"][f"WISE_vs_{ref}"] = block

    summary_rows = []
    for m in methods:
        vals = list(by[m].values())
        row = {"method": m}
        for key in PAIRED_METRICS + ["success"]:
            arr = np.array([v[key] for v in vals], float)
            arr = arr[np.isfinite(arr)]
            row[f"{key}_median"] = float(np.median(arr)) if arr.size else float("nan")
            row[f"{key}_min"] = float(arr.min()) if arr.size else float("nan")
            row[f"{key}_max"] = float(arr.max()) if arr.size else float("nan")
        summary_rows.append(row)
    with (GEN / "robot_monte_carlo_summary.csv").open("w", newline="",
                                                      encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)

    (GEN / "robot_statistical_report.json").write_text(json.dumps(report, indent=2),
                                                       encoding="utf-8")
    print(json.dumps(report["success_counts"], indent=2))
    return report


if __name__ == "__main__":
    main(n_seeds=int(sys.argv[1]) if len(sys.argv) > 1 else 30)
