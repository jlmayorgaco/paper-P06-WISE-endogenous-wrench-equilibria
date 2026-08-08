"""The reduced information layer of Prop. "stability", driven by the *real* graph.

    a_dot = -m_y a + theta_1 b + Q^T w(t)
    b_dot =  theta_2 a - c Q^T L_geo(q(t)) Q b,        a, b in R^{N-1}

Two uses, kept strictly apart:

* **certified replay** (``w = 0``, common initial condition): the object the
  time-varying corollary talks about. Its weighted norm
  ``||[a,b]||_P = sqrt(theta_2 ||a||^2 + theta_1 ||b||^2)`` must decay no slower than
  ``exp(-alpha(sigma_req) t)`` whenever ``Q^T L_geo Q >= sigma_req I`` throughout.
* **mission layer** (``w != 0``): the same system excited by the measured progress
  disagreement ``w_i = k_w (s_{k(i)} - s_bar)`` and mapped downstream to per-robot
  progress commands ``u_i = u_0 - k_a [Q a]_i``. This mapping is an *illustrative*
  downstream realization, not a nonlinear closed-loop theorem.

Gains, initial condition and integrator (fixed-step RK4) are identical for every
assignment method.
"""

from __future__ import annotations

import numpy as np

from . import communication as comm
from . import config as C
from . import scenario as S

GAINS = C.InfoGains()


def initial_state(gains: C.InfoGains = GAINS) -> tuple[np.ndarray, np.ndarray]:
    """Common initial condition, aligned with the region-split (bridge) mode.

    ``chi`` is +1 on the robots homed in the left region and -1 on the right; the
    excited mode is therefore exactly the one a bridge has to damp. It is built from
    the *homes*, not from any method's assignment, so it is method-independent.
    """
    chi = np.where(S.HOMES[:, 0] < 0.5 * (S.HOMES[:, 0].min() + S.HOMES[:, 0].max()),
                   1.0, -1.0)
    chi = chi - chi.mean()
    chi = chi / np.linalg.norm(chi)
    d = comm.Q_BASIS.T @ chi
    return C.A0_SCALE * d, C.B0_SCALE * d


def weighted_norm(a: np.ndarray, b: np.ndarray, gains: C.InfoGains = GAINS) -> float:
    """``||[a,b]||_P`` with ``P = diag(theta_2 I, theta_1 I)``; ``V_c = ||.||_P^2 / 2``."""
    return float(np.sqrt(gains.theta_2 * a @ a + gains.theta_1 * b @ b))


def _deriv(a, b, Lred, forcing, gains: C.InfoGains):
    da = -gains.m_y * a + gains.theta_1 * b + forcing
    db = gains.theta_2 * a - gains.c * (Lred @ b)
    return da, db


def rk4_step(a, b, Lred, forcing, dt, gains: C.InfoGains = GAINS):
    """One fixed-step RK4 update; ``Lred`` and ``forcing`` are held over the step."""
    k1a, k1b = _deriv(a, b, Lred, forcing, gains)
    k2a, k2b = _deriv(a + 0.5 * dt * k1a, b + 0.5 * dt * k1b, Lred, forcing, gains)
    k3a, k3b = _deriv(a + 0.5 * dt * k2a, b + 0.5 * dt * k2b, Lred, forcing, gains)
    k4a, k4b = _deriv(a + dt * k3a, b + dt * k3b, Lred, forcing, gains)
    return (a + dt / 6 * (k1a + 2 * k2a + 2 * k3a + k4a),
            b + dt / 6 * (k1b + 2 * k2b + 2 * k3b + k4b))


def progress_commands(a: np.ndarray) -> np.ndarray:
    """Per-robot progress command ``u_i = u_0 - k_a [Q a]_i`` with declared saturation."""
    return np.clip(C.U_0 - C.K_A * (comm.Q_BASIS @ a), C.U_MIN, C.U_MAX)


def disagreement_forcing(assignment: tuple, s: np.ndarray) -> np.ndarray:
    """``Q^T w`` with ``w_i = k_w (s_{k(i)} - mean_k s_k)`` for lifters, 0 otherwise."""
    w = np.zeros(S.N_ROBOTS)
    s_bar = float(np.mean(s))
    for i, act in enumerate(assignment):
        if act[0] == "lift":
            w[i] = C.K_W * (s[act[1]] - s_bar)
    return comm.Q_BASIS.T @ w


def certified_replay(Lred_traj: np.ndarray, dt: float,
                     gains: C.InfoGains = GAINS) -> np.ndarray:
    """Unforced ``||[a,b]||_P`` along a recorded ``Q^T L_geo(q(t)) Q`` trajectory."""
    a, b = initial_state(gains)
    zero = np.zeros_like(a)
    out = [weighted_norm(a, b, gains)]
    for Lred in Lred_traj:
        a, b = rk4_step(a, b, Lred, zero, dt, gains)
        out.append(weighted_norm(a, b, gains))
    return np.asarray(out)
