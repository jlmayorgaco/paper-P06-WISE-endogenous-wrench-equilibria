"""P1: find the wrench-demand level at which a facet actually binds, and what it costs.

The regime grid reported that wrench demand is inert: Lambda_E was constant across
tau_d in [1,4] because no wrench facet ever became active. That leaves the
"wrench-feasible" half of the formulation empirically unexercised. Here we push
tau_d until the first facet binds and measure the consequence on the fiber:

    dim E,  Gamma_E,  Lambda_E   before and after.

The expected direction (tighter wrench => smaller fiber => less free connectivity)
is NOT assumed; we report what the sweep gives, including the infeasible cutoff.

Writes generated/wrench_bind.csv and generated/wrench_bind.json.
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

from wise_mr import nullspace as ns, scenarios  # noqa: E402
from exp_gamma import (gamma_sdp, lam2, optimal_fiber_base,  # noqa: E402
                       net_visible_dimension)
from exp_regime import _max_lambda2  # noqa: E402

GEN = ROOT / "generated"
GEN.mkdir(exist_ok=True)

TAUS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
SEEDS = [3, 5, 7]
ACTIVE_TOL = 1e-3


def n_active_wrench(prob, z, tol=ACTIVE_TOL) -> int:
    """How many wrench rows H_w z >= d are active (tight) at z."""
    Hw = prob.wrench_matrix()
    d = prob.demand().ravel()
    s = Hw @ np.asarray(z).ravel() - d
    return int(np.sum(np.abs(s) <= tol))


def main() -> None:
    rows = []
    for sd in SEEDS:
        for tau in TAUS:
            rec = {"seed": sd, "tau_d": tau}
            try:
                prob = scenarios.two_region(seed=sd, N=12, nu=0.5, tau_d=tau)
                zbar, y_star, V_star, _ = optimal_fiber_base(prob)
                if zbar is None or not np.all(np.isfinite(zbar)):
                    raise ValueError("no fiber point")
                info = ns.fiber_dimension(prob, zbar)
                g, _, mult, _ = gamma_sdp(prob, zbar)
                lamE, _ = _max_lambda2(prob, y_star=y_star)
                rec.update(
                    status="ok",
                    n_active_wrench=n_active_wrench(prob, zbar),
                    dim_E=info["dim_E"],
                    d_net=net_visible_dimension(prob, info["Np_basis"]),
                    Gamma_E=g,
                    Lambda_E=lamE,
                    lambda2_zbar=lam2(prob, zbar),
                    V_star=V_star,
                )
            except Exception as exc:                      # noqa: BLE001
                rec.update(status=f"infeasible:{type(exc).__name__}",
                           n_active_wrench=-1, dim_E=-1, d_net=-1,
                           Gamma_E=float("nan"), Lambda_E=float("nan"),
                           lambda2_zbar=float("nan"), V_star=float("nan"))
            rows.append(rec)
            print(f"seed {sd} tau_d={tau:5.1f}  {rec['status']:22s} "
                  f"active={rec['n_active_wrench']:3d} dimE={rec['dim_E']:3d} "
                  f"d_net={rec['d_net']:3d} Gamma={rec['Gamma_E']:.4f} "
                  f"Lam_E={rec['Lambda_E']:.4f}")

    with (GEN / "wrench_bind.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    first_bind = {}
    for sd in SEEDS:
        seq = [r for r in ok if r["seed"] == sd]
        binding = [r for r in seq if r["n_active_wrench"] > 0]
        if binding:
            b = min(binding, key=lambda r: r["tau_d"])
            before = [r for r in seq if r["tau_d"] < b["tau_d"]]
            prev = max(before, key=lambda r: r["tau_d"]) if before else None
            first_bind[sd] = {
                "tau_d": b["tau_d"],
                "n_active": b["n_active_wrench"],
                "dim_E_before": prev["dim_E"] if prev else None,
                "dim_E_after": b["dim_E"],
                "Gamma_before": prev["Gamma_E"] if prev else None,
                "Gamma_after": b["Gamma_E"],
                "Lambda_E_before": prev["Lambda_E"] if prev else None,
                "Lambda_E_after": b["Lambda_E"],
            }
    infeasible = {sd: min((r["tau_d"] for r in rows
                           if r["seed"] == sd and r["status"] != "ok"), default=None)
                  for sd in SEEDS}
    out = {"taus": TAUS, "seeds": SEEDS, "active_tol": ACTIVE_TOL,
           "first_binding": first_bind, "first_infeasible_tau": infeasible,
           "note": "direction not assumed; reported as measured"}
    (GEN / "wrench_bind.json").write_text(json.dumps(out, indent=1))
    print("\nfirst binding:", json.dumps(first_bind, indent=1))
    print("first infeasible tau:", infeasible)


if __name__ == "__main__":
    main()
