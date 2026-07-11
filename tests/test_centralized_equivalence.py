"""Day-2 gate: centralized oracle and WISE primal-dual agree on aggregates.

Under a strictly concave potential over a convex X_WISE, the distributed
primal-dual fixed point and the centralized potential maximizer must coincide on
the aggregates (s(x), lambda_2(L(x))) and on the certification prices (mu*, pi*).
"""

import pytest


@pytest.mark.skip(reason="centralized oracle + primal-dual implemented on Day 2")
def test_centralized_matches_primal_dual_aggregates():
    ...


@pytest.mark.skip(reason="KKT prices from both solvers agree — Day 2")
def test_certification_prices_agree():
    ...
