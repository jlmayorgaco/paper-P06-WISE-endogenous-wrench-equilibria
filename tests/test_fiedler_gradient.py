"""Endogenous Laplacian: affine weights, Fiedler value, sigma*, gradient."""

import numpy as np
import pytest

from wise_mr import endogenous_graph as eg


def test_sigma_star_formula():
    assert eg.sigma_star(theta=0.5, c=1.0, m_F=1.0) == pytest.approx(0.25)
    # scales like theta^2 and 1/(c m_F)
    assert eg.sigma_star(1.0, 2.0, 4.0) == pytest.approx(1.0 / 8.0)


def test_sigma_star_rejects_nonpositive_gains():
    with pytest.raises(ValueError):
        eg.sigma_star(0.5, 0.0, 1.0)
    with pytest.raises(ValueError):
        eg.sigma_star(0.5, 1.0, -1.0)


def test_edge_weights_affine():
    C = np.zeros((2, 2, 3))
    C[0, 0, 0] = 1.0
    C[1, 1, 2] = 2.0
    x = np.array([[1.0, 0.0], [0.0, 0.5]])
    a = eg.edge_weights(C, x)
    assert np.allclose(a, [1.0, 0.0, 1.0])


def test_induced_laplacian_and_fiedler_path_graph():
    # two edges of a 3-node path; unit weights => Fiedler value = 1.0
    V = 3
    L01 = np.zeros((V, V)); L01[[0, 1], [0, 1]] = 1; L01[0, 1] = L01[1, 0] = -1
    L12 = np.zeros((V, V)); L12[[1, 2], [1, 2]] = 1; L12[1, 2] = L12[2, 1] = -1
    edge_laplacians = np.stack([L01, L12])
    L = eg.induced_laplacian(edge_laplacians, np.array([1.0, 1.0]))
    assert eg.fiedler_value(L) == pytest.approx(1.0)
    assert eg.is_information_sustaining(L, sigma=0.5)
    assert not eg.is_information_sustaining(L, sigma=1.5)


@pytest.mark.skip(reason="Fiedler gradient implemented on Day 1; finite-diff check")
def test_fiedler_gradient_matches_finite_difference():
    ...
