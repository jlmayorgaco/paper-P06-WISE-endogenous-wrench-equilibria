"""Which set does the convex selection layer optimize over, and is Gamma_E computed on
an exactly characterized cone? See docs/GAMMA_COMPUTATION_AUDIT.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from wise_mr import nullspace as ns
from wise_mr import scenarios
from wise_mr import wrench_tensor as wt

SEEDS = [3, 5, 7]


def _prob(seed):
    return scenarios.two_region(seed=seed)


@pytest.mark.parametrize("seed", SEEDS)
def test_support_system_is_explicit_not_enumerated(seed):
    """H_w has exactly M*P rows: an explicit H-representation, no facet enumeration."""
    prob = _prob(seed)
    Hw = prob.wrench_matrix()
    assert Hw.shape == (prob.M * prob.P, prob.N * prob.A)
    assert prob.M * prob.P == 10


@pytest.mark.parametrize("seed", SEEDS)
def test_exact_feasible_implies_support_feasible(seed):
    """X_f^exact subset X_f^sup: the support test is a necessary condition, so every
    exactly feasible point satisfies it. (The converse is NOT claimed.)"""
    prob = _prob(seed)
    rng = np.random.default_rng(1000 + seed)
    A_maps, _ = scenarios._slot_maps_full(prob.meta["load"])
    F, kappa, m = prob.meta["F"], prob.meta["kappa"], prob.meta["m_sides"]
    checked = 0
    for _ in range(200):
        z = rng.random((prob.N, prob.A))
        z = z / z.sum(axis=1, keepdims=True)
        slots = prob.slots_view(z)
        caps, maps, counts = [], [], []
        for i in range(prob.N):
            for h in range(prob.H):
                if slots[i, 0, h] > 1e-12:
                    caps.append(F[i])
                    maps.append(A_maps[0][h])
                    counts.append(slots[i, 0, h])
        if not caps:
            continue
        exact = wt.certify_membership_lift(np.array(caps), np.array(maps), prob.w_dem[0],
                                           counts=np.array(counts), kappa=kappa, m_sides=m)
        if not exact:
            continue
        checked += 1
        s = prob.capacity(z)
        assert np.all(s - prob.demand() >= -1e-6), "exactly feasible point violates support"
    assert checked > 0, "no exactly feasible sample drawn; test vacuous"


@pytest.mark.parametrize("seed", SEEDS)
def test_active_set_is_empty_at_the_reported_fiber_point(seed):
    """The reported Gamma_E instances live in the G_I = empty regime, where the tangent
    cone is the subspace ker A cap ker B and is representation-independent."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from exp_gamma import optimal_fiber_base

    prob = _prob(seed)
    zbar, _, _, _ = optimal_fiber_base(prob)
    if zbar is None or not np.all(np.isfinite(zbar)):
        pytest.skip("solver did not return a fiber point")
    G_I = ns.active_inequalities(prob, zbar)
    assert G_I.shape[0] == 0, (
        "a wrench/nonnegativity row is active: Gamma_E on the support relaxation is then "
        "only an upper bound (see docs/GAMMA_LIFTED_EQUIVALENCE_PROOF.md)")
    info = ns.fiber_dimension(prob, zbar)
    assert info["n_active_ineq"] == 0
    # with no active row the neutral space is exactly ker A cap ker B
    A, B = ns.mass_matrix(prob), ns.served_matrix(prob)
    stack = np.vstack([A, B])
    assert info["dim_E"] == prob.N * prob.A - np.linalg.matrix_rank(stack, tol=1e-8)


def test_facet_normal_and_lifted_lp_agree_on_the_flagship():
    """The two exact membership oracles of Lemma 1 must agree wherever both apply."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import exp_flagship as FLAG

    cases = [(FLAG.SLOTS1, FLAG.W1_DEM, [(0, FLAG.F_L), (3, FLAG.F_S)]),
             (FLAG.SLOTS1, FLAG.W1_DEM, [(1, FLAG.F_S), (2, FLAG.F_S), (3, FLAG.F_S)]),
             (FLAG.SLOTS2, FLAG.W2_DEM, [(0, FLAG.F_L), (1, FLAG.F_S)]),
             (FLAG.SLOTS2, FLAG.W2_DEM, [(1, FLAG.F_L), (0, FLAG.F_S)])]
    for slots, w_dem, contacts in cases:
        caps = np.array([f for _, f in contacts])
        maps = np.array([slots[h] for h, _ in contacts])
        a = wt.certify_membership_lp(caps, maps, w_dem, kappa=1.0, m_sides=12)
        b = wt.certify_membership_lift(caps, maps, w_dem, kappa=1.0, m_sides=12)
        assert a == b, (contacts, a, b)
