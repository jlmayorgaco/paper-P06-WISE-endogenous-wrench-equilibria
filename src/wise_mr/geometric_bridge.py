"""Bridge the affine candidate-site Laplacian ``L_bar(x)`` and the geometric
graph ``L_geo(q)`` via Weyl's eigenvalue-perturbation inequality (paper Lemma,
TASK 6).

The convex theory is stated over ``L_bar(x)``. Robots track their intended
candidate sites with bounded error, giving ``||L_geo(q) - L_bar(x)||_2 <=
eps_L``. Weyl's inequality then yields

    lambda_2(L_geo(q)) >= lambda_2(L_bar(x)) - eps_L,

so requiring ``lambda_2(L_bar(x)) >= sigma_rob + delta`` with
``delta = eps_L + eps_est + delta_num`` transfers the guarantee to the physical
graph: ``lambda_2(L_geo(q)) >= sigma_rob``.

Here ``L_bar`` is realized as the geometric graph at the *intended* candidate
positions and ``L_geo`` at the *actual* (tracked) positions; ``eps_L`` is the
spectral-norm gap induced by the tracking error.
"""

from __future__ import annotations

import numpy as np

from . import dynamics as dyn
from . import endogenous_graph as eg


def spectral_norm(M: np.ndarray) -> float:
    """Spectral (2-)norm ``||M||_2`` = largest singular value."""
    return float(np.linalg.norm(np.asarray(M, dtype=float), 2))


def weyl_lower_bound(lam2_bar: float, eps_L: float) -> float:
    """Weyl lower bound on ``lambda_2(L_geo)``: ``lambda_2(L_bar) - eps_L``."""
    return float(lam2_bar) - float(eps_L)


def laplacian_lipschitz(N: int, scale: float = 1.5, bridge_gain: float = 3.0) -> float:
    """Conservative Lipschitz constant ``C_L`` with ``||Delta L||_2 <= C_L * rho``.

    Edge weight ``a=exp(-d/scale)`` (or bridge ``bridge_gain*exp(-d/3)``) has
    ``|da/dd| <= max(1/scale, bridge_gain/3)``. A tracking error ``rho`` per robot
    changes each incident distance by ``<= 2 rho`` and perturbs at most ``N-1``
    incident weights; ``||Delta L||_2 <= 2 max_i sum_j |Delta a_ij|`` gives
    ``C_L = 4 (N-1) max(1/scale, bridge_gain/3)`` (loose but sound).
    """
    edge_lip = max(1.0 / float(scale), float(bridge_gain) / 3.0)
    return 4.0 * (int(N) - 1) * edge_lip


def measure_mismatch(intended_pos, actual_pos, ranges, relay_mask, **kw):
    """Return ``(eps_L, lam2_bar, lam2_geo, weyl_holds)`` for one position pair."""
    L_bar = dyn.geometric_laplacian(intended_pos, ranges, relay_mask, **kw)
    L_geo = dyn.geometric_laplacian(actual_pos, ranges, relay_mask, **kw)
    eps_L = spectral_norm(L_geo - L_bar)
    lam2_bar = eg.fiedler_value(L_bar)
    lam2_geo = eg.fiedler_value(L_geo)
    weyl_holds = lam2_geo >= weyl_lower_bound(lam2_bar, eps_L) - 1e-9
    return eps_L, lam2_bar, lam2_geo, bool(weyl_holds)


def _disk_perturb(rng, n, rho):
    """Uniform perturbations inside the disk of radius ``rho`` (``(n,2)``)."""
    ang = rng.uniform(0, 2 * np.pi, size=n)
    rad = rho * np.sqrt(rng.uniform(0, 1, size=n))
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)


def sample_eps_L(intended_pos, ranges, relay_mask, rho, n_samples=200, seed=0, **kw):
    """Max observed ``eps_L`` and worst-case Weyl check over random tracking errors."""
    rng = np.random.default_rng(seed)
    intended_pos = np.asarray(intended_pos, dtype=float)
    max_eps, all_weyl = 0.0, True
    for _ in range(n_samples):
        actual = intended_pos + _disk_perturb(rng, intended_pos.shape[0], rho)
        eps_L, _, _, ok = measure_mismatch(intended_pos, actual, ranges, relay_mask, **kw)
        max_eps = max(max_eps, eps_L)
        all_weyl = all_weyl and ok
    return max_eps, all_weyl
