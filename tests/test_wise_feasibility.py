"""Membership in X_WISE and the projection onto the simplex."""

import numpy as np
import pytest

from wise_mr import primal_dual as pd


def test_project_simplex_rows_sum_to_one():
    v = np.array([[0.2, 0.5, 0.9], [3.0, -1.0, 0.0]])
    p = pd.project_simplex(v)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= -1e-12)


def test_project_simplex_already_feasible_is_fixed_point():
    v = np.array([[0.25, 0.25, 0.5]])
    assert np.allclose(pd.project_simplex(v), v)


@pytest.mark.skip(reason="requires WiseProblem factory + potential (Day 1/2)")
def test_wise_membership_wrench_and_information():
    ...
