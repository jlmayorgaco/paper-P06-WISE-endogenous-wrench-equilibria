"""Tensor aggregation, demand projection, and feasibility."""

import numpy as np
import pytest

from wise_mr import wrench_tensor as wt


def test_support_box_axis_aligned():
    lo, hi = np.array([-1.0, -2.0]), np.array([1.0, 3.0])
    # +x picks upper.x, -y picks lower.y (=> +2 contribution)
    assert wt.support_box(lo, hi, [1.0, 0.0]) == pytest.approx(1.0)
    assert wt.support_box(lo, hi, [0.0, -1.0]) == pytest.approx(2.0)
    assert wt.support_box(lo, hi, [1.0, 1.0]) == pytest.approx(1.0 + 3.0)


def test_directional_capacity_shape_and_value():
    N, M, H, P = 2, 1, 2, 3
    W = np.ones((N, M, H, P))
    x = np.zeros((N, M, H))
    x[0, 0, 0] = 1.0
    x[1, 0, 1] = 0.5
    s = wt.directional_capacity(W, x)
    assert s.shape == (M, P)
    # each direction sums the two active shares: 1.0 + 0.5
    assert np.allclose(s, 1.5)


def test_demand_projection_matches_dot():
    M, P = 2, 4
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((M, P, 3))
    w_dem = rng.standard_normal((M, 3))
    d = wt.demand_projection(directions, w_dem)
    assert d.shape == (M, P)
    for k in range(M):
        for ell in range(P):
            assert d[k, ell] == pytest.approx(directions[k, ell] @ w_dem[k])


def test_feasibility_and_residual():
    s = np.array([[1.0, 2.0]])
    d = np.array([[0.5, 3.0]])          # second direction under-supplied
    resid = wt.wrench_residual(s, d)
    assert np.allclose(resid, [[0.0, 1.0]])
    assert wt.is_wrench_feasible(s, d).tolist() == [False]
    assert wt.is_wrench_feasible(s, s).tolist() == [True]


@pytest.mark.skip(reason="tensor assembly + LP membership implemented on Day 1")
def test_build_tensor_matches_lp_membership():
    ...
