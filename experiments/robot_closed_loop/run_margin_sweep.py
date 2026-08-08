"""Predeclared relay-attenuation sweep: what the *extra* zero-loss margin buys.

HARD and WISE both clear ``sigma_req`` on the nominal graph, and their closed-loop
transport is indistinguishable there (E2). The difference is their zero-loss spectral
reserve, ``m_lambda = lambda_2(Lbar) - sigma_req``: 0.036 for HARD, 0.099 for WISE. If
that reserve means anything, HARD must lose the certificate first as the relay channel
degrades.

The sweep is declared, not tuned: a single scalar ``a`` in (0,1] multiplies *every*
gated relay conductance, in ``Lbar`` and in ``L_geo`` alike, on a fixed grid. For each
method we report the crossing of ``sigma_req`` (certificate lost) and of ``sigma_dyn``
(information layer no longer exponentially stable), and re-run the full closed loop on a
coarser grid. No attenuation value is chosen after looking at the outcome.

    python -m experiments.robot_closed_loop.run_margin_sweep
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
CERT_GRID = np.round(np.arange(1.00, 0.399, -0.01), 3)      # certificate crossings
SIM_GRID = (1.00, 0.90, 0.80, 0.70, 0.60, 0.50)             # closed-loop re-runs
METHODS = ("PROD", "HARD", "WISE")


def _crossing(att: np.ndarray, lam: np.ndarray, level: float) -> float:
    """Largest attenuation at which ``lam`` is still >= ``level`` (nan if never)."""
    ok = np.where(lam >= level)[0]
    if ok.size == 0:
        return float("nan")
    last = int(ok[-1])
    return float(att[last])


def main() -> dict:
    _, chosen, certs = RF.select()
    methods = [m for m in METHODS if m in chosen]

    rows, lam_curves = [], {m: [] for m in methods}
    for a in CERT_GRID:
        prev = S.set_relay_attenuation(float(a))
        try:
            for m in methods:
                lam = S.lambda2(S.lbar(chosen[m]))
                lam_curves[m].append(lam)
                rows.append({"attenuation": float(a), "method": m, "lambda2_bar": lam,
                             "margin_vs_sigma_req": lam - C.SIGMA_REQ,
                             "margin_vs_sigma_dyn": lam - C.SIGMA_DYN,
                             "alpha": C.alpha_rate(lam)})
        finally:
            S.set_relay_attenuation(prev)

    crossings = {}
    for m in methods:
        lam = np.asarray(lam_curves[m])
        crossings[m] = {
            "lambda2_bar_nominal": float(lam[0]),
            "nominal_margin": float(lam[0] - C.SIGMA_REQ),
            "last_attenuation_clearing_sigma_req": _crossing(CERT_GRID, lam, C.SIGMA_REQ),
            "last_attenuation_clearing_sigma_dyn": _crossing(CERT_GRID, lam, C.SIGMA_DYN),
        }

    sim_rows = []
    for a in SIM_GRID:
        prev = S.set_relay_attenuation(float(a))
        try:
            for m in methods:
                z = chosen[m]
                lbar = S.lbar(z)
                res = sim.simulate(m, z, lbar, seed=0, pert=C.SeedPerturbation())
                s = M.summarize(res, certs[m], C.SIGMA_REQ, C.SeedPerturbation())
                sim_rows.append({
                    "attenuation": float(a), "method": m,
                    "lambda2_bar": S.lambda2(lbar),
                    "certified": bool(S.lambda2(lbar) >= C.SIGMA_REQ),
                    "min_lambda2_geo": s["min_lambda2_geo"],
                    "min_transfer_margin": s["min_transfer_margin"],
                    "alpha_fitted": s["alpha_fitted"],
                    "max_sync_err": s["max_sync_err"],
                    "terminal_sync_err": s["terminal_sync_err"],
                    "mission_completed": s["mission_completed"],
                    "success": s["success"],
                })
                print(f"a={a:.2f} {m:<5} lam2_bar={sim_rows[-1]['lambda2_bar']:.4f} "
                      f"alpha_fit={s['alpha_fitted']:+.4f} maxSync={s['max_sync_err']:.4f} "
                      f"ok={s['success']}", flush=True)
        finally:
            S.set_relay_attenuation(prev)

    with (GEN / "robot_margin_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (GEN / "robot_margin_sweep_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sim_rows[0]))
        w.writeheader()
        w.writerows(sim_rows)

    report = {
        "design": ("single declared scalar multiplying every gated relay conductance, in "
                   "Lbar and L_geo alike; grid fixed before any outcome was inspected"),
        "sigma_dyn": C.SIGMA_DYN, "sigma_req": C.SIGMA_REQ,
        "certificate_grid": [float(x) for x in CERT_GRID],
        "simulation_grid": list(SIM_GRID),
        "crossings": crossings,
    }
    (GEN / "robot_margin_sweep.json").write_text(json.dumps(report, indent=2),
                                                 encoding="utf-8")
    print(json.dumps(crossings, indent=2))
    return report


if __name__ == "__main__":
    main()
