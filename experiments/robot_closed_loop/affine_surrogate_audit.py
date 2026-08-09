"""Why the N=6 flagship is *not* an instance of the affine model (Assumption 1(ii)).

Assumption 1(ii) and the Stage-2 SDP need a decision-independent family ``{L0, L_ir}`` so
that ``Lbar(z) = L0 + sum_ir z_ir L_ir`` is affine. Building that family from tubes forces
every edge weight to be an infimum over *both* endpoints' admissible positions taken across
*all* their actions -- otherwise the matrix still depends on what the other endpoint chose.

This script constructs exactly that strictly affine surrogate for the flagship and reports
what it certifies. The answer is: nothing. The relay range gate ``R_L = 5.0`` sits just below
the worst-case tube separation ``5.20`` reached when a partner robot is assigned to the far
load, so two of the five relay links are gated off and the graph disconnects,
``lambda2 = 0`` for *every* assignment.

The flagship's induced graph is therefore genuinely action-pair dependent (bilinear in ``z``),
and the paper reports it as an exact *integer* illustration obtained by enumeration -- not as
an instance of Theorem 2 or of the Stage-2 SDP, which are validated on the affine ``N=12``
instances instead. Lemma 2 (geometric transfer), Cor. 2 (moving graph) and Cor. 3 (attenuation
budget) are unaffected: each holds at a *fixed* assignment and never differentiates in ``z``.

    python -m experiments.robot_closed_loop.affine_surrogate_audit
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


def _sup_over_partner(i: int, ai: tuple, j: int) -> float:
    """sup distance from robot ``i`` in action ``ai`` to robot ``j`` over *all* j's actions."""
    return max(S.tube_max_distance(i, ai, j, aj) for aj in S.ACTIONS)


def _sup_over_both(i: int, j: int) -> float:
    return max(S.tube_max_distance(i, ai, j, aj) for ai in S.ACTIONS for aj in S.ACTIONS)


def L0_affine() -> np.ndarray:
    """Decision-independent base Laplacian: infimum weight over both robots' whole ranges."""
    W = np.zeros((S.N_ROBOTS, S.N_ROBOTS))
    for i in range(S.N_ROBOTS):
        for j in range(i + 1, S.N_ROBOTS):
            W[i, j] = W[j, i] = float(S._base_weight(_sup_over_both(i, j)))
    return S._laplacian_from_adj(W)


def L_ir_affine(i: int, r: str) -> np.ndarray:
    """Decision-independent gated relay Laplacian of robot ``i`` at relay site ``r``."""
    W = np.zeros((S.N_ROBOTS, S.N_ROBOTS))
    for j in range(S.N_ROBOTS):
        if j == i:
            continue
        W[i, j] = W[j, i] = float(S._relay_weight(_sup_over_partner(i, ("relay", r), j), i))
    return S._laplacian_from_adj(W)


def lbar_affine(assignment: tuple) -> np.ndarray:
    L = L0_affine()
    for i, a in enumerate(assignment):
        if a[0] == "relay":
            L = L + L_ir_affine(i, a[1])
    return L


def fiber_spectrum_affine() -> dict:
    """lambda2 over the whole integer productive fiber under the strictly affine surrogate.

    The point of this sweep: the surrogate is so coarse that only the centre relay site
    survives worst-case gating, so *every* assignment clearing any viable requirement attains
    the same lambda2. HARD and WISE then coincide and the reserve comparison disappears --
    the flagship's HARD/WISE gap exists precisely because its graph is action-pair dependent.
    """
    from experiments.robot_closed_loop import assignments as A

    feasible = A.enumerate_feasible()
    values = {a: A.productive_value(A.served_aggregate(a)) for a in feasible}
    v_max = max(values.values())
    fiber = [a for a in feasible if abs(values[a] - v_max) <= 1e-12]

    L0 = L0_affine()
    cache: dict = {}

    def lam(a):
        L = L0.copy()
        for i, act in enumerate(a):
            if act[0] == "relay":
                key = (i, act[1])
                if key not in cache:
                    cache[key] = L_ir_affine(i, act[1])
                L = L + cache[key]
        return S.lambda2(L)

    lams = {a: lam(a) for a in fiber}
    prod = min(fiber, key=lambda a: A._tie_break(a, None))
    wise = min(fiber, key=lambda a: (-round(lams[a], 12),) + A._tie_break(a, prod))
    levels = sorted({round(v, 6) for v in lams.values()})

    rows = []
    for sreq in (0.10, 0.15, 0.18, 0.19):
        pool = [a for a in fiber if lams[a] >= sreq - 1e-12]
        hard = min(pool, key=lambda a: A._tie_break(a, prod)) if pool else None
        rows.append({"sigma_req": sreq, "n_clearing": len(pool),
                     "lambda2_hard": lams[hard] if hard else None,
                     "lambda2_wise": lams[wise], "hard_equals_wise": bool(hard == wise)})
    return {"n_fiber": len(fiber), "distinct_lambda2_levels": levels,
            "lambda2_wise": lams[wise], "lambda2_prod": lams[prod],
            "requirement_sweep": rows,
            "note": ("Under the affine surrogate HARD coincides with WISE at every viable "
                     "requirement, so the certified-budget comparison cannot be posed on "
                     "this scenario at all.")}


def main() -> dict:
    _, chosen, _ = RF.select()
    out = {"sigma_req": C.SIGMA_REQ, "lambda2_L0_affine": S.lambda2(L0_affine()),
           "methods": {}, "gated_links": []}

    for m, a in chosen.items():
        cur, aff = S.lambda2(S.lbar(a)), S.lambda2(lbar_affine(a))
        out["methods"][m] = {"lambda2_action_pair": cur, "lambda2_strictly_affine": aff,
                             "affine_clears_sigma_req": bool(aff >= C.SIGMA_REQ)}
        print(f"{m:<14} lambda2  action-pair={cur:7.4f}   strictly-affine={aff:7.4f}   "
              f"clears={aff >= C.SIGMA_REQ}")

    # the two links that the range gate kills, and by how little
    wise = chosen["WISE"]
    for i, a in enumerate(wise):
        if a[0] != "relay":
            continue
        for j in range(S.N_ROBOTS):
            if j == i:
                continue
            d_sup = _sup_over_partner(i, a, j)
            rec = {"relay": i, "partner": j, "d_chosen": S.tube_max_distance(i, a, j, wise[j]),
                   "d_sup_over_partner_actions": d_sup, "range_gate": float(S.R_RANGE[i]),
                   "gated_off": bool(d_sup > S.R_RANGE[i])}
            out["gated_links"].append(rec)
            if rec["gated_off"]:
                print(f"  gated off: relay {i} -- robot {j}: worst case {d_sup:.2f} > "
                      f"R={S.R_RANGE[i]:.1f} (chosen pose {rec['d_chosen']:.2f})")

    out["fiber_spectrum"] = fiber_spectrum_affine()
    fs = out["fiber_spectrum"]
    print(f"\naffine fiber spectrum: {len(fs['distinct_lambda2_levels'])} distinct levels "
          f"{fs['distinct_lambda2_levels']} over {fs['n_fiber']} fiber points")
    for r in fs["requirement_sweep"]:
        print(f"  sigma_req={r['sigma_req']:.2f}: {r['n_clearing']:4d} clearing, "
              f"HARD==WISE: {r['hard_equals_wise']}")

    out["conclusion"] = (
        "The strictly affine (action-independent) tube surrogate disconnects the flagship for "
        "every assignment, so the N=6 instance is reported as an exact integer illustration "
        "under an action-pair-dependent graph, outside Assumption 1(ii), Theorem 2 and the "
        "Stage-2 SDP; those are validated on the affine N=12 instances. Lemma 2, Cor. 2 and "
        "Cor. 3 hold at a fixed assignment and are unaffected.")
    (GEN / "affine_surrogate_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {GEN / 'affine_surrogate_audit.json'}")
    return out


if __name__ == "__main__":
    main()
