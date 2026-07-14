"""Generate the geometric-bridge certificate (TASK 6): report eps_L, verify Weyl
and the Lipschitz bound, and compose the design margin delta = eps_L + eps_est +
delta_num. Writes generated/geometric_bridge_certificate.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wise_mr import baselines, geometric_bridge as gb, scenarios  # noqa: E402

GEN = Path(__file__).resolve().parents[1] / "generated"
GEN.mkdir(exist_ok=True)

# margin components (documented; eps_est is characterized in the estimator task)
EPS_EST = 0.02     # assumed distributed eigenvalue-estimation error (pending TASK 11)
DELTA_NUM = 1e-3   # numerical/design margin
RHO = 0.30         # candidate-site tracking-error radius
BRIDGE = np.array([5.0, 5.0])


def main() -> None:
    prob = scenarios.two_region(seed=3, N=12, nu=0.42, tau_d=3.0, bridge_gain=3.0)
    res = baselines.wise_primal_dual(prob, max_iters=4000)
    relay_mask = prob.relay_shares(res.x) > 0.5
    ranges = prob.meta["r"]

    # intended candidate positions: relays routed to the bridge site
    intended = prob.meta["pos"].copy()
    intended[relay_mask] = BRIDGE

    C_L = gb.laplacian_lipschitz(prob.N, scale=1.5, bridge_gain=3.0)
    max_eps, weyl_ok = gb.sample_eps_L(intended, ranges, relay_mask, RHO,
                                       n_samples=300, seed=7)
    lip_ok = max_eps <= C_L * RHO + 1e-9
    delta = float(max_eps + EPS_EST + DELTA_NUM)

    cert = {
        "instance": {"seed": 3, "N": int(prob.N), "nu": 0.42, "tau_d": 3.0},
        "n_relays": int(relay_mask.sum()),
        "tracking_error_rho": RHO,
        "eps_L_max_observed": round(float(max_eps), 6),
        "lipschitz_C_L": round(float(C_L), 4),
        "lipschitz_bound_C_L_rho": round(float(C_L * RHO), 4),
        "lipschitz_bound_holds": bool(lip_ok),
        "weyl_holds_all_samples": bool(weyl_ok),
        "margin_components": {"eps_L": round(float(max_eps), 6),
                              "eps_est": EPS_EST, "delta_num": DELTA_NUM},
        "delta_total": round(delta, 6),
        "margin_exceeds_eps_L": bool(delta >= max_eps),
    }
    (GEN / "geometric_bridge_certificate.json").write_text(
        json.dumps(cert, indent=2), encoding="utf-8")
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
