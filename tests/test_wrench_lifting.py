"""Exact zonotopic wrench lifting (paper Lemma 1).

The G-representation lift ``certify_membership_lift`` and the H-representation
facet LP ``certify_membership_lp`` are two exact tests of the SAME inner-zonotope
membership, so they must agree; and the inner-zonotope generators must reproduce
the inner polygon's support.
"""

import numpy as np
import pytest

from wise_mr import wrench_tensor as wt


def test_generators_reproduce_polygon_support():
    """Support of the generator zonotope equals the inner-polygon support."""
    rng = np.random.default_rng(1)
    for m_sides in (3, 4, 6, 8, 12):
        r = 1.7
        G = wt.inner_zonotope_generators(r, m_sides)          # (2, m)
        V = wt.inner_polygon(r, m_sides)                      # (2m, 2)
        for _ in range(50):
            d = rng.standard_normal(2)
            h_zono = wt.support_zonotope(G.T, d)              # sum |<g_j,d>|
            h_poly = wt.support_polygon(V, d)                 # max_k <v_k,d>
            assert h_zono == pytest.approx(h_poly, abs=1e-9)


def test_lift_agrees_with_facet_lp_100_instances():
    """Lifted (generator) LP and facet-normal LP agree on 100 random instances."""
    rng = np.random.default_rng(7)
    n_agree = 0
    n_total = 100
    for _ in range(n_total):
        N = int(rng.integers(1, 5))                           # 1..4 robots
        m_sides = int(rng.choice([3, 4, 6, 8]))
        F = rng.uniform(0.5, 2.0, size=N)
        A = rng.standard_normal((N, 3, 2))
        # demand: a random point scaled to straddle the feasible boundary
        w = rng.standard_normal(3) * rng.uniform(0.1, 3.0)
        kappa = float(rng.uniform(0.6, 1.0))
        a = wt.certify_membership_lift(F, A, w, kappa=kappa, m_sides=m_sides)
        b = wt.certify_membership_lp(F, A, w, kappa=kappa, m_sides=m_sides)
        n_agree += int(a == b)
    assert n_agree == n_total, f"lift vs facet LP disagreed on {n_total - n_agree}/100"


def test_lift_respects_counts_scaling():
    """Doubling the count relaxes feasibility (monotone in z)."""
    rng = np.random.default_rng(3)
    N = 3
    F = rng.uniform(0.8, 1.5, size=N)
    A = rng.standard_normal((N, 3, 2))
    w = rng.standard_normal(3) * 1.5
    feas_small = wt.certify_membership_lift(F, A, w, counts=np.full(N, 0.4))
    feas_big = wt.certify_membership_lift(F, A, w, counts=np.full(N, 4.0))
    # a larger budget can only add feasible points
    assert feas_big or not feas_small
