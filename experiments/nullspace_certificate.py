"""TASK 4: certify the composition-fiber dimension via the minimal-face rank
formula dim E = n - rank([A; B; G_I]), and the productive-nullspace residuals.
Writes generated/nullspace_certificate.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wise_mr import baselines, nullspace as ns, scenarios  # noqa: E402

GEN = Path(__file__).resolve().parents[1] / "generated"
GEN.mkdir(exist_ok=True)


def main() -> None:
    prob = scenarios.two_region(seed=3, N=12, nu=0.5, tau_d=3.0, bridge_gain=3.0)
    res = baselines.wise_primal_dual(prob, max_iters=4000)
    info = ns.fiber_dimension(prob, res.x)
    max_res = 0.0
    for r in info["residuals"]:
        max_res = max(max_res, r["Ad"], r["Bd"], r["GId"])
    cert = {
        "instance": {"seed": 3, "N": int(prob.N), "nu": 0.5, "tau_d": 3.0},
        "n_decision": info["n"],
        "rank_ABGI": info["rank"],
        "n_active_inequalities": info["n_active_ineq"],
        "dim_E_fiber": info["dim_E"],
        "Np_basis_dim": int(info["Np_basis"].shape[0]),
        "max_neutral_residual": float(max_res),
        "formula": "dim E = n - rank([A; B; G_I]) at a relative-interior point",
        "tolerance": 1e-8,
        "positive_dimensional": bool(info["dim_E"] > 0),
    }
    (GEN / "nullspace_certificate.json").write_text(json.dumps(cert, indent=2), encoding="utf-8")
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
