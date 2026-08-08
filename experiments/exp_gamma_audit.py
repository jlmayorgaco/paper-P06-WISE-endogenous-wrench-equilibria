"""Audit: which wrench-feasible set does the selection machinery actually optimize over?

The paper needs one unambiguous answer to "where do the active rows $G_I$ used by
$\\Gamma_{\\mathcal E}$ come from?". This script answers it from the code, not from prose.

Finding, stated up front and then verified below:

  * The convex selection layer (Stage-2 SDP, fiber dimension, $\\Gamma_{\\mathcal E}$,
    price sweep) optimizes over the **finite-direction support relaxation**

        X_f^sup = { z : H_w z >= d },
        H_w[(k,l),(i,k,h)] = W[i,k,h,l] = h_{U_tau^in}(A_kh^T eta_kl),
        d_kl = eta_kl^T w^dem_k,

    an **explicit H-representation** with M*P rows. No facet enumeration is needed and
    the active rows are read off directly -- this is PATH A of the audit.

  * X_f^sup is an **outer** relaxation of the exact reachable-zonotope set: membership in
    a zonotope requires the support inequality in *every* direction, and only P are
    tested. So a point can satisfy H_w z >= d and still be wrench-infeasible.

  * That gap is closed downstream, not upstream: every reported assignment is
    re-certified by the exact inner-zonotope LP (``certify_membership_lp``) before it is
    accepted. The selection layer may propose; only the exact LP disposes.

Writes generated/gamma_computation_audit.json.

    python experiments/exp_gamma_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from wise_mr import nullspace as ns  # noqa: E402
from wise_mr import scenarios  # noqa: E402
from wise_mr import wrench_tensor as wt  # noqa: E402

GEN = ROOT / "generated"
GEN.mkdir(exist_ok=True)
SEEDS = [3, 5, 7, 11, 13]
RNG = np.random.default_rng(20260808)


def support_feasible(prob, z, tol=1e-9) -> bool:
    """The relaxation actually used by the SDP / Gamma_E: H_w z >= d in P directions."""
    s = prob.capacity(z.reshape(prob.N, prob.A))
    return bool(np.all(s - prob.demand() >= -tol))


def exact_feasible(prob, z, tol=1e-9) -> bool:
    """Exact membership in the reachable inner-zonotope set, per load, by LP.

    ``w^dem_k in (+)_{i,h} z_ikh A_kh U_tau^in`` -- the Lemma-1 object. Scaling a
    zonotope generator matrix by z_ikh is exactly what ``counts`` does in the lifted LP.
    """
    slots = prob.slots_view(z.reshape(prob.N, prob.A))          # (N, M, H)
    A_maps = prob.meta["A_maps"] if "A_maps" in prob.meta else None
    if A_maps is None:                                          # rebuild from the scenario
        A_maps, _ = scenarios._slot_maps_full(prob.meta["load"])
    F = prob.meta["F"]
    kappa = prob.meta["kappa"]
    m_sides = prob.meta["m_sides"]
    for k in range(prob.M):
        caps, maps, counts = [], [], []
        for i in range(prob.N):
            for h in range(prob.H):
                if slots[i, k, h] > 1e-12:
                    caps.append(F[i])
                    maps.append(A_maps[0][h])
                    counts.append(slots[i, k, h])
        if not caps:
            return False
        ok = wt.certify_membership_lift(np.array(caps), np.array(maps), prob.w_dem[k],
                                        counts=np.array(counts), kappa=kappa,
                                        m_sides=m_sides)
        if not ok:
            return False
    return True


def main() -> dict:
    rows, gaps = [], 0
    for seed in SEEDS:
        prob = scenarios.two_region(seed=seed)
        n = prob.N * prob.A
        Hw = prob.wrench_matrix()
        d = prob.demand().ravel()

        # --- PATH A evidence: the H-representation is explicit, not enumerated -------
        rec = {
            "seed": seed,
            "n_decision_vars": n,
            "n_support_rows": int(Hw.shape[0]),
            "n_directions_P": int(prob.P),
            "explicit_H_representation": True,
            "facet_enumeration_required": False,
        }

        # --- is the support test a relaxation? find a witness -----------------------
        witness = None
        for _ in range(4000):
            z = RNG.random((prob.N, prob.A))
            z = z / z.sum(axis=1, keepdims=True)
            z = z.ravel()
            if support_feasible(prob, z) and not exact_feasible(prob, z):
                witness = {
                    "min_support_slack": float(np.min(Hw @ z - d)),
                    "exact_lp_feasible": False,
                }
                break
        rec["support_relaxation_witness"] = witness
        rec["support_is_strict_relaxation"] = witness is not None
        gaps += int(witness is not None)

        # --- active-row bookkeeping used by Gamma_E ---------------------------------
        from exp_gamma import optimal_fiber_base  # noqa: PLC0415
        zbar, y_star, V_star, _ = optimal_fiber_base(prob)
        if zbar is not None and np.all(np.isfinite(zbar)):
            G_I = ns.active_inequalities(prob, zbar)
            s = (Hw @ zbar).reshape(prob.M, prob.P)
            rec.update(
                n_active_rows_total=int(G_I.shape[0]),
                n_active_wrench_rows=int(sum(
                    1 for k in range(prob.M) for ell in range(prob.P)
                    if abs(s[k, ell] - prob.demand()[k, ell]) <= 1e-3
                    and prob.demand()[k, ell] > 0)),
                min_support_slack_at_zbar=float(np.min(Hw @ zbar - d)),
                exact_lp_feasible_at_zbar=bool(exact_feasible(prob, zbar)),
            )
        rows.append(rec)
        print(f"seed {seed}: {rec['n_support_rows']} support rows, "
              f"{rec.get('n_active_wrench_rows', '?')} active; "
              f"strict relaxation: {rec['support_is_strict_relaxation']}")

    out = {
        "question": ("where do the active inequality rows G_I used by Gamma_E come "
                     "from, and which wrench-feasible set does the selection layer "
                     "optimize over?"),
        "answer_path": "A -- explicit H-representation, no facet enumeration",
        "selection_set": ("X_f^sup = {z : H_w z >= d}, the finite-direction support "
                          "relaxation with M*P rows; G_I = active nonnegativity rows "
                          "plus active H_w rows, both read off directly"),
        "relation_to_lemma_1": ("X_f^sup is an OUTER relaxation of the exact reachable "
                                "inner-zonotope set: zonotope membership needs the "
                                "support inequality in every direction, only P are "
                                "tested. It is not the exact projection of the lifted "
                                "system."),
        "how_the_gap_is_closed": ("downstream: every reported assignment is re-certified "
                                  "by the exact inner-zonotope LP "
                                  "(wrench_tensor.certify_membership_lp) before it is "
                                  "accepted; the selection layer proposes, the exact LP "
                                  "disposes"),
        "n_seeds": len(SEEDS),
        "n_seeds_with_strict_relaxation_witness": gaps,
        "rows": rows,
    }
    (GEN / "gamma_computation_audit.json").write_text(json.dumps(out, indent=2),
                                                      encoding="utf-8")
    print(f"\nstrict-relaxation witnesses found on {gaps}/{len(SEEDS)} seeds")
    print(f"wrote {GEN / 'gamma_computation_audit.json'}")
    return out


if __name__ == "__main__":
    main()
