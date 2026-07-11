"""Solvers under comparison for the constraint-ablation study.

Method               Wrench  Endog. graph  Expected outcome
-------------------  ------  ------------  -----------------------------------
scalar_capacity      no      no            physical false positives
wrench_only          yes     no            valid coalition that breaks the net
connectivity_only    no      yes           connected net, incapable load
wise_primal_dual     yes     yes           certified execution  (headline)
centralized_wise     yes     yes           team-optimal upper bound (multi-start)

All solvers return a :class:`~wise_mr.primal_dual.SolveResult`-compatible record
via :func:`run`, so the drivers treat them uniformly.
"""

from __future__ import annotations

import numpy as np

from . import primal_dual as pd
from .equilibrium import WiseProblem, uniform_start


def wise_primal_dual(problem: WiseProblem, **kw) -> pd.SolveResult:
    cfg = pd.PDConfig(enforce_wrench=True, enforce_info=True, **kw)
    return pd.solve(problem, config=cfg)


def wrench_only(problem: WiseProblem, **kw) -> pd.SolveResult:
    cfg = pd.PDConfig(enforce_wrench=True, enforce_info=False, **kw)
    res = pd.solve(problem, config=cfg)
    res.method = "wrench_only"
    return res


def connectivity_only(problem: WiseProblem, **kw) -> pd.SolveResult:
    cfg = pd.PDConfig(enforce_wrench=False, enforce_info=True, **kw)
    res = pd.solve(problem, config=cfg)
    res.method = "connectivity_only"
    return res


def scalar_capacity(problem: WiseProblem, **kw) -> pd.SolveResult:
    """Greedy cardinality/scalar-capacity heuristic: ignore direction and graph.

    Assign the cheapest robots to their nearest slot until the *scalar* capacity
    ``sum F_i`` on the load meets the scalar demand ``||w_dem||``; everyone else
    idles. No robot relays, so connectivity is typically broken.
    """
    N, H = problem.N, problem.H
    F = problem.meta["F"]
    scalar_demand = float(np.linalg.norm(problem.w_dem[0]))
    x = np.zeros((N, problem.A))
    x[:, problem.idle_index] = 1.0                       # default: idle
    slot_cost = problem.g[:, : H]                        # cost to each slot
    best_slot = np.argmin(slot_cost, axis=1)
    order = np.argsort(slot_cost[np.arange(N), best_slot])  # cheapest first
    got = 0.0
    for i in order:
        if got >= scalar_demand:
            break
        x[i, problem.idle_index] = 0.0
        x[i, best_slot[i]] = 1.0
        got += F[i]
    return pd.SolveResult(x=x, pi=0.0, method="scalar_capacity", iters=0,
                          converged=True, history={"mu_final": problem.wrench_price(x)})


def centralized_wise_oracle(problem: WiseProblem, n_starts: int = 4, **kw) -> pd.SolveResult:
    """Team-optimal reference: best certified welfare over several warm starts."""
    rng = np.random.default_rng(problem.meta.get("seed", 0) + 777)
    best, best_val = None, -np.inf
    for s in range(n_starts):
        x0 = uniform_start(problem) if s == 0 else _random_start(problem, rng)
        cfg = pd.PDConfig(enforce_wrench=True, enforce_info=True, max_iters=8000)
        res = pd.solve(problem, x0=x0, config=cfg)
        val = problem.potential(res.x) - 1e3 * max(0.0, -problem.info_margin(res.x))
        if val > best_val:
            best, best_val = res, val
    best.method = "centralized_wise"
    return best


def _random_start(problem: WiseProblem, rng) -> np.ndarray:
    x = rng.random((problem.N, problem.A))
    return x / x.sum(axis=1, keepdims=True)


REGISTRY = {
    "scalar_capacity": scalar_capacity,
    "wrench_only": wrench_only,
    "connectivity_only": connectivity_only,
    "wise_primal_dual": wise_primal_dual,
    "centralized_wise": centralized_wise_oracle,
}


def run(name: str, problem: WiseProblem, **kw) -> pd.SolveResult:
    try:
        return REGISTRY[name](problem, **kw)
    except KeyError as exc:
        raise KeyError(f"unknown solver {name!r}; known: {sorted(REGISTRY)}") from exc
