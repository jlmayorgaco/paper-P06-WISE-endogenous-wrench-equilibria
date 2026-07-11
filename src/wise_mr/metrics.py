"""Evaluation metrics for the WISE campaign.

Primary metric --- **certified service rate**: a load counts as served only if it
is both wrench-feasible and information self-sustaining
(``max_residual <= wrench_tol`` and ``lambda_2 - sigma* >= -info_tol``).
"""

from __future__ import annotations

import numpy as np

from .equilibrium import WiseProblem


def certified(problem: WiseProblem, x, wrench_tol: float = 0.05, info_tol: float = 1e-2) -> bool:
    """Whether the (single-load) instance is certified WISE under tolerances."""
    return problem.is_wise(x, wrench_tol=wrench_tol, info_tol=info_tol)


def residual_wrench(problem: WiseProblem, x) -> float:
    return problem.max_residual(x)


def connectivity_margin(problem: WiseProblem, x) -> float:
    return problem.info_margin(x)


def welfare(problem: WiseProblem, x) -> float:
    """Team welfare = tensor-aggregative potential ``Phi(x)``."""
    return problem.potential(x)


def relay_composition(problem: WiseProblem, x) -> dict:
    """Relay mass split by robot type (long- vs short-range) and its cost ratio."""
    r = problem.relay_shares(x)
    is_long = problem.meta["is_long"]
    return {
        "relay_long": float(r[is_long].sum()),
        "relay_short": float(r[~is_long].sum()),
        "relay_total": float(r.sum()),
    }


def certified_service_rate(records) -> float:
    """Fraction of certified instances across a list of ``(problem, x)`` pairs."""
    flags = [certified(p, x) for p, x in records]
    return float(np.mean(flags)) if flags else float("nan")


def paired_bootstrap_ci(values, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean of ``values``."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(v.mean()), float(lo), float(hi))
