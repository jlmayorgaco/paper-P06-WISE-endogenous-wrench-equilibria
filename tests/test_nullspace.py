"""TASK 4 gates: minimal-face dimension formula and the comparative-advantage
four-cycle (productively neutral mass-preserving exchange).
"""

import numpy as np

from wise_mr import nullspace as ns, scenarios


def test_fiber_dimension_formula_and_residuals():
    prob = scenarios.two_region(seed=3, N=12, nu=0.5, tau_d=3.0)
    x = np.full((prob.N, prob.A), 1.0 / prob.A)
    info = ns.fiber_dimension(prob, x)
    # dim E = n - rank([A;B;G_I])
    assert info["dim_E"] == info["n"] - info["rank"]
    assert info["dim_E"] >= 0
    # every certified neutral direction respects A, B, and active inequalities
    for r in info["residuals"]:
        assert r["Ad"] < 1e-8 and r["Bd"] < 1e-8 and r["GId"] < 1e-8


def test_comparative_advantage_four_cycle_neutral():
    """A mass-preserving four-cycle at (tau,a),(tau,b),(tau',a),(tau',b) is
    productively neutral (B d = 0) iff B_{tau,a}-B_{tau,b}=B_{tau',a}-B_{tau',b}.
    Here B_{i,slot} = c_i, so equal-capacity robots give a neutral exchange.
    """
    prob = scenarios.two_region(seed=1, N=12, nu=0.5, tau_d=3.0)
    B = ns.served_matrix(prob)
    cap = prob._cap()
    A_actions = prob.A
    # pick two robots and two slots of the same load
    i, j, a, b = 0, 1, 0, 1
    d = np.zeros(prob.N * A_actions)
    d[i * A_actions + a] += 1; d[i * A_actions + b] -= 1
    d[j * A_actions + a] -= 1; d[j * A_actions + b] += 1
    Bd = B @ d
    # neutral iff cap_i - cap_i == cap_j - cap_j  (same robot rows) -> always 0 here,
    # since both slots feed the same load column with the same robot weight
    assert np.allclose(Bd, 0.0, atol=1e-12)
    # a cross-robot swap on the SAME slot changes B unless caps are equal
    d2 = np.zeros(prob.N * A_actions)
    d2[i * A_actions + a] += 1; d2[j * A_actions + a] -= 1
    Bd2 = B @ d2
    expected = np.zeros(prob.M); expected[0] = cap[i] - cap[j]
    assert np.allclose(Bd2, expected, atol=1e-12)
