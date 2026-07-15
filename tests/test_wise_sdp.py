"""WISE existence/selection SDP (paper Thm. 2) vs direct lambda_2 computation."""

import numpy as np
import pytest

pytest.importorskip("cvxpy")

from wise_mr import wise_sdp
from wise_mr.endogenous_graph import complement_basis, fiedler_value


def _edge_laplacian(i: int, j: int, V: int, w: float = 1.0) -> np.ndarray:
    e = np.zeros(V)
    e[i], e[j] = 1.0, -1.0
    return w * np.outer(e, e)


def _instance():
    """Two clusters {0,1},{2,3}; a relay variable adds the bridge edge (1,2)."""
    V = 4
    L0 = _edge_laplacian(0, 1, V) + _edge_laplacian(2, 3, V)   # disconnected: lambda2=0
    M0 = _edge_laplacian(1, 2, V)                              # relay bridge
    Q = complement_basis(V)
    # single decision z in [0,1]
    G = np.array([[1.0], [-1.0]])
    h = np.array([1.0, 0.0])
    return V, L0, [(0, M0)], Q, G, h


def test_sdp_selects_max_connectivity():
    V, L0, terms, Q, G, h = _instance()
    res = wise_sdp.solve_wise_sdp(1, L0, terms, Q, sigma_req=0.0,
                                  G_ineq=G, h_ineq=h)
    assert res.status in ("optimal", "optimal_inaccurate")
    # relay driven to its upper bound -> connected path P4
    assert res.z[0] == pytest.approx(1.0, abs=1e-3)
    # optimal value matches the independently recomputed lambda_2
    assert res.lambda_star == pytest.approx(res.lambda2_check, abs=1e-4)
    # path P4 Laplacian has lambda_2 = 2 - sqrt(2)
    assert res.lambda_star == pytest.approx(2.0 - np.sqrt(2.0), abs=2e-3)


def test_sdp_value_is_the_fiber_maximum():
    """Lambda_E upper-bounds lambda_2 at every feasible z (optimality)."""
    V, L0, terms, Q, G, h = _instance()
    res = wise_sdp.solve_wise_sdp(1, L0, terms, Q, sigma_req=0.0,
                                  G_ineq=G, h_ineq=h)
    rng = np.random.default_rng(0)
    for _ in range(25):
        z = rng.uniform(0.0, 1.0)
        L = L0 + z * terms[0][1]
        assert fiedler_value(L) <= res.lambda_star + 1e-4


def test_sdp_existence_flips_at_requirement():
    V, L0, terms, Q, G, h = _instance()
    lam = 2.0 - np.sqrt(2.0)
    below = wise_sdp.solve_wise_sdp(1, L0, terms, Q, sigma_req=0.5 * lam,
                                    G_ineq=G, h_ineq=h)
    above = wise_sdp.solve_wise_sdp(1, L0, terms, Q, sigma_req=1.5 * lam,
                                    G_ineq=G, h_ineq=h)
    assert below.wise_exists is True      # Lambda_E >= sigma_req
    assert above.wise_exists is False     # no composition clears it
