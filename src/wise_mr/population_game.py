"""Heterogeneous recruitment population game (Documento IV, sections 2.2-3).

Types ``tau`` (mass ``N_tau``, capacity ``c_tau``, health ``eta_tau``, range
``r_tau``) recruit to tasks ``k`` with decreasing marginal value.  The state is the
occupancy matrix ``x = (x_{tau k}) >= 0`` where ``x_{tau k}`` is the fraction of the
total capacity ``C = sum_tau N_tau c_tau`` contributed by type ``tau`` at task ``k``.
Aggregate served capacity is ``y_k = sum_tau x_{tau k}`` and each type conserves its
mass ``sum_k x_{tau k} = xbar_tau = N_tau c_tau / C``.

The payoff of task ``k`` is common to every type and depends only on the aggregate:

    p_k(x) = v_k - alpha C y_k ,
    Phi(x) = sum_k ( v_k y_k - (alpha C / 2) y_k^2 ) ,

an exact potential game, ``mu``-strongly concave in ``y`` with ``mu = alpha C``
(Documento IV eq. 1).  Two structural facts drive the whole paper:

* **Prop 3.1 (water-filling).** The served aggregate ``y*`` is unique.
* **Prop 3.2 (composition degeneracy).** Every ``x`` with ``sum_tau x_{tau k}=y*_k``
  is a Nash equilibrium; the equilibrium set is a polytope of dimension
  ``(T-1)(M_act-1)`` -- the game fixes *how much* each task gets, never *who* brings it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Population:
    """Type-level description of a heterogeneous fleet and its tasks."""

    mass: np.ndarray      # (T,) N_tau
    capacity: np.ndarray  # (T,) c_tau
    value: np.ndarray     # (M,) v_k  (task marginal values)
    alpha: float          # congestion slope
    health: np.ndarray | None = None   # (T,) eta_tau (optional; for sigma* heterogeneity)
    range_: np.ndarray | None = None    # (T,) r_tau (optional; for the induced graph)

    @property
    def T(self) -> int:
        return int(self.mass.shape[0])

    @property
    def M(self) -> int:
        return int(self.value.shape[0])

    @property
    def C(self) -> float:
        """Total capacity ``C = sum_tau N_tau c_tau``."""
        return float(np.dot(self.mass, self.capacity))

    @property
    def xbar(self) -> np.ndarray:
        """Per-type mass share ``xbar_tau = N_tau c_tau / C`` (sums to 1)."""
        return self.mass * self.capacity / self.C

    @property
    def mu(self) -> float:
        """Strong-concavity modulus ``mu = alpha C`` (Documento IV eq. 1)."""
        return float(self.alpha) * self.C


def water_filling(pop: Population) -> tuple[np.ndarray, float]:
    """Unique served aggregate ``y*`` (Prop 3.1): ``y*_k = [v_k - lam]_+ / (alpha C)``.

    ``lam`` is the water level chosen so ``sum_k y*_k = 1`` (total normalized capacity).
    Returns ``(y_star, lam)``.
    """
    v = np.asarray(pop.value, dtype=float)
    aC = pop.mu  # alpha * C
    # y_k(lam) = max(v_k - lam, 0)/aC is nonincreasing in lam; bisect sum_k y_k = 1.
    lo, hi = float(v.min() - aC), float(v.max())
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        s = np.sum(np.maximum(v - lam, 0.0)) / aC
        if s > 1.0:
            lo = lam
        else:
            hi = lam
    lam = 0.5 * (lo + hi)
    y_star = np.maximum(v - lam, 0.0) / aC
    return y_star, lam


def potential(pop: Population, y: np.ndarray) -> float:
    """Exact potential ``Phi = sum_k (v_k y_k - (alpha C / 2) y_k^2)`` (eq. 1)."""
    y = np.asarray(y, dtype=float)
    return float(np.sum(pop.value * y - 0.5 * pop.mu * y**2))


def payoff(pop: Population, y: np.ndarray) -> np.ndarray:
    """Task payoffs ``p_k(x) = v_k - alpha C y_k`` (common to all types)."""
    return pop.value - pop.mu * np.asarray(y, dtype=float)


def aggregate(x: np.ndarray) -> np.ndarray:
    """Served capacity per task ``y_k = sum_tau x_{tau k}``; ``(T,M) -> (M,)``."""
    return np.asarray(x, dtype=float).sum(axis=0)


def active_tasks(y_star: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Boolean mask of active tasks (``y*_k > 0``)."""
    return np.asarray(y_star) > tol


def degeneracy_dimension(T: int, m_active: int) -> int:
    """Dimension of the composition-equilibrium polytope, ``(T-1)(M_act-1)`` (Prop 3.2)."""
    return max(T - 1, 0) * max(m_active - 1, 0)


def is_equilibrium(pop: Population, x: np.ndarray, tol: float = 1e-6) -> bool:
    """Nash test (Prop 3.2): ``x`` is an equilibrium iff its aggregate equals ``y*``.

    Also checks feasibility: nonnegativity and per-type mass conservation.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < -tol):
        return False
    if not np.allclose(x.sum(axis=1), pop.xbar, atol=1e-6):
        return False
    y_star, _ = water_filling(pop)
    return bool(np.allclose(aggregate(x), y_star, atol=tol))
