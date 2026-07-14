"""TASK 6 gates: Weyl bridge from L_bar(x) to L_geo(q), Lipschitz bound, and a
failure test showing insufficient margin can violate physical connectivity.
"""

import numpy as np
import pytest

from wise_mr import endogenous_graph as eg, geometric_bridge as gb


def _random_laplacian(rng, n):
    A = rng.uniform(0, 1, size=(n, n))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return np.diag(A.sum(1)) - A


# 1. Weyl bound across random states ---------------------------------------
def test_weyl_bound_random_states():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(3, 9)
        L1 = _random_laplacian(rng, n)
        L2 = _random_laplacian(rng, n)
        lam2_1 = eg.fiedler_value(L1)
        lam2_2 = eg.fiedler_value(L2)
        eps = gb.spectral_norm(L2 - L1)
        # Weyl: |lambda_2(L2) - lambda_2(L1)| <= ||L2 - L1||_2
        assert lam2_2 >= gb.weyl_lower_bound(lam2_1, eps) - 1e-9
        assert abs(lam2_2 - lam2_1) <= eps + 1e-9


# 2. Candidate-site tracking perturbation obeys the Lipschitz bound ---------
def test_tracking_perturbation_within_lipschitz():
    rng = np.random.default_rng(1)
    N = 8
    intended = rng.uniform(0, 10, size=(N, 2))
    ranges = np.full(N, 6.0)
    relay_mask = np.zeros(N, bool)
    relay_mask[0] = True
    rho = 0.2
    C_L = gb.laplacian_lipschitz(N, scale=1.5, bridge_gain=3.0)
    max_eps, weyl_ok = gb.sample_eps_L(intended, ranges, relay_mask, rho,
                                       n_samples=150, seed=3)
    assert weyl_ok
    assert max_eps <= C_L * rho + 1e-9          # empirical <= conservative bound
    assert max_eps > 0.0                        # perturbation is non-trivial


# 3. Insufficient margin can violate physical connectivity -----------------
def test_insufficient_margin_can_fail_physical_connectivity():
    # Intended graph barely clears sigma; a large tracking error drops the
    # geometric lambda_2 below sigma, so a margin < eps_L is NOT safe.
    rng = np.random.default_rng(4)
    N = 6
    # two clusters bridged only by relay 0 sitting at the midpoint
    intended = np.array([[5.0, 5.0],       # relay at bridge
                         [2.0, 5.0], [2.3, 5.4], [1.7, 4.6],
                         [8.0, 5.0], [8.2, 4.7]])
    ranges = np.array([6.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    relay_mask = np.zeros(N, bool); relay_mask[0] = True
    from wise_mr import dynamics as dyn
    lam2_bar = eg.fiedler_value(dyn.geometric_laplacian(intended, ranges, relay_mask))
    sigma = lam2_bar - 0.02                     # sigma just below the intended margin
    # push the relay far off its site (large tracking error)
    failed = False
    for _ in range(50):
        actual = intended.copy()
        actual[0] = actual[0] + rng.uniform(-3, 3, size=2)   # big relay displacement
        eps_L, _, lam2_geo, weyl_ok = gb.measure_mismatch(
            intended, actual, ranges, relay_mask)
        assert weyl_ok                          # Weyl always holds
        if lam2_geo < sigma and eps_L > (lam2_bar - sigma):
            failed = True                       # margin (lam2_bar - sigma) < eps_L -> unsafe
            break
    assert failed, "expected some large tracking error to break physical connectivity"
