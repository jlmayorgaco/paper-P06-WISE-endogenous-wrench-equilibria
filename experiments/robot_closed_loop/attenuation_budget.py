"""Certified attenuation budget: turning the spectral reserve into an operational number.

For a uniform relay attenuation ``rho`` the surrogate Laplacian is

    Lbar_rho(z) = L0 + (1-rho) sum_{i,r} z_ir L_ir = (1-rho) Lbar(z) + rho L0,

an affine segment between ``Lbar(z)`` and ``L0``. Concavity of ``lambda_2`` on that segment
gives the lower bound

    lambda2(Lbar_rho) >= (1-rho) lambda2(Lbar) + rho lambda2(L0),

so the requirement ``lambda2 >= sigma_req`` is *certified* for every

    rho <= rho_cert(z) = [lambda2(Lbar(z)) - sigma_req] / [lambda2(Lbar(z)) - lambda2(L0)]

whenever the denominator is positive. With a base graph that is disconnected without a
relay (``lambda2(L0) = 0``) this is simply ``rho_cert = 1 - sigma_req / lambda2(Lbar)``.

Nonuniform attenuation ``rho_ir <= rho`` is covered by the same bound: since every
``L_ir >= 0``,

    L0 + sum (1-rho_ir) z_ir L_ir  >=  L0 + (1-rho) sum z_ir L_ir = Lbar_rho,

and eigenvalue monotonicity on ``1^perp`` preserves the inequality.

The consequence is the operational reading of WISE: among all productive optima it
maximizes ``lambda2(Lbar)``, hence maximizes the certified degradation budget.

    python -m experiments.robot_closed_loop.attenuation_budget
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.robot_closed_loop import config as C  # noqa: E402
from experiments.robot_closed_loop import run_flagship as RF  # noqa: E402
from experiments.robot_closed_loop import scenario as S  # noqa: E402

GEN = ROOT / "generated"
METHODS = ("PROD", "HARD", "WISE")


def base_laplacian(assignment: tuple) -> np.ndarray:
    """``L0``: the always-on part of ``Lbar`` with every gated relay link removed."""
    prev = S.set_relay_attenuation(0.0)          # (1-rho) = 0  =>  relay links vanish
    try:
        return S.lbar(assignment)
    finally:
        S.set_relay_attenuation(prev)


def certified_budget(assignment: tuple, sigma_req: float) -> dict:
    lam_bar = S.lambda2(S.lbar(assignment))
    lam_0 = S.lambda2(base_laplacian(assignment))
    denom = lam_bar - lam_0
    rho = (lam_bar - sigma_req) / denom if denom > 1e-12 else float("nan")
    return {"lambda2_bar": lam_bar, "lambda2_L0": lam_0,
            "certified_rho": float(rho) if np.isfinite(rho) else None,
            "clears_nominally": bool(lam_bar >= sigma_req)}


def verify_bound(assignment: tuple, n: int = 61) -> dict:
    """Check lambda2(Lbar_rho) >= (1-rho) lambda2(Lbar) + rho lambda2(L0) numerically."""
    lam_bar = S.lambda2(S.lbar(assignment))
    lam_0 = S.lambda2(base_laplacian(assignment))
    worst, rho_worst = np.inf, None
    for rho in np.linspace(0.0, 1.0, n):
        prev = S.set_relay_attenuation(1.0 - rho)
        try:
            lam = S.lambda2(S.lbar(assignment))
        finally:
            S.set_relay_attenuation(prev)
        slack = lam - ((1.0 - rho) * lam_bar + rho * lam_0)
        if slack < worst:
            worst, rho_worst = float(slack), float(rho)
    return {"min_concavity_slack": worst, "at_rho": rho_worst,
            "bound_holds": bool(worst >= -1e-9)}


def main() -> dict:
    _, chosen, _ = RF.select()
    sweep_path = GEN / "robot_margin_sweep.json"
    observed = {}
    if sweep_path.exists():
        cross = json.loads(sweep_path.read_text(encoding="utf-8"))["crossings"]
        for m, v in cross.items():
            a = v.get("last_attenuation_clearing_sigma_req")
            observed[m] = None if a is None or not np.isfinite(a) else round(1.0 - a, 4)

    out = {"sigma_req": C.SIGMA_REQ, "sigma_dyn": C.SIGMA_DYN, "methods": {}}
    for m in METHODS:
        if m not in chosen:
            continue
        rec = certified_budget(chosen[m], C.SIGMA_REQ)
        rec.update(verify_bound(chosen[m]))
        rec["observed_last_rho_clearing_sigma_req"] = observed.get(m)
        if rec["certified_rho"] is not None and observed.get(m) is not None:
            rec["certified_is_conservative"] = bool(
                rec["certified_rho"] <= observed[m] + 1e-9)
        out["methods"][m] = rec
        cr = rec["certified_rho"]
        print(f"{m:<5} lam2(Lbar)={rec['lambda2_bar']:.4f} lam2(L0)={rec['lambda2_L0']:.4f} "
              f"rho_cert={'n/a' if cr is None else f'{cr:6.4f}'} "
              f"observed<={rec['observed_last_rho_clearing_sigma_req']} "
              f"concavity slack>={rec['min_concavity_slack']:.2e}")

    out["claim"] = ("WISE maximizes lambda2(Lbar) over the productive-optimum fiber, hence "
                    "maximizes the certified zero-productive-loss attenuation budget "
                    "rho_cert = 1 - sigma_req/lambda2(Lbar) when lambda2(L0) = 0")
    (GEN / "attenuation_budget.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {GEN / 'attenuation_budget.json'}")
    return out


if __name__ == "__main__":
    main()
