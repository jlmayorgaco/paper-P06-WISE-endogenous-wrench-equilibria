"""Trajectory PD, progress governor and the reference generator -- method-independent.

Every gain here is shared by PROD, HARD and WISE (asserted by ``tests/``): the only
thing that differs between methods is *who holds which contact slot and which relay
site*, which is exactly the claim under test.

The progress governor is the one place where the physics can talk back: a load whose
team cannot deliver the demanded wrench, or that is falling behind its own reference
pose, slows down. That is what turns an asymmetric physical disturbance into a
*coordination* problem -- the other load has to be told to wait, and telling it
requires the network.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import load_dynamics as dyn
from . import scenario as S

GOVERNOR_WTOL = 0.30        # wrench residual at which a load stops advancing [N]
_DS = 1e-4                  # finite-difference step for path derivatives


def path_pose(k: int, s: float) -> np.ndarray:
    return np.atleast_2d(S.LOAD_PATHS[k].pose(s))[0]


def path_derivative(k: int, s: float) -> np.ndarray:
    """``dq/ds`` by central differences on the analytic path."""
    hi = np.atleast_2d(S.LOAD_PATHS[k].pose(min(1.0, s + _DS)))[0]
    lo = np.atleast_2d(S.LOAD_PATHS[k].pose(max(0.0, s - _DS)))[0]
    return (hi - lo) / max(1e-12, min(1.0, s + _DS) - max(0.0, s - _DS))


def reference(k: int, s: float, s_dot: float):
    """Desired pose, desired *body* twist and its time derivative."""
    q_d = path_pose(k, s)
    dq = path_derivative(k, s)
    nu_world = dq * s_dot
    R = dyn.rot(q_d[2])
    nu_d = np.array([*(R.T @ nu_world[:2]), nu_world[2]])
    return q_d, nu_d


def pose_error(q_d: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Pose error in the *body* frame of the load: ``[R^T (p_d - p); wrap(th_d - th)]``."""
    e_w = q_d[:2] - q[:2]
    R = dyn.rot(q[2])
    return np.array([*(R.T @ e_w), dyn.wrap(q_d[2] - q[2])])


def demanded_wrench(load: dyn.RigidLoad, q, nu, q_d, nu_d, nu_d_dot,
                    drag_scale: float = 1.0) -> np.ndarray:
    """``w^dem = M nu_d_dot + C(nu_d) nu_d - R(nu_d) + Kp e_q + Kd e_nu`` (body frame),
    with ``R`` the load's resistance wrench (so ``-R`` is the feed-forward drag)."""
    e_q = pose_error(q_d, q)
    e_nu = nu_d - nu
    Kp = np.array([C.KP_POS, C.KP_POS, C.KP_ROT])
    Kd = np.array([C.KD_POS, C.KD_POS, C.KD_ROT])
    return (load.M @ nu_d_dot + load.coriolis(nu_d) - load.resistance(nu_d, drag_scale)
            + Kp * e_q + Kd * e_nu)


def governor(pose_err_norm: float, wrench_residual: float) -> float:
    """Speed factor ``h in [0,1]``: a load that cannot track or cannot deliver slows."""
    worst = max(pose_err_norm / C.GOVERNOR_ETOL, wrench_residual / GOVERNOR_WTOL)
    return float(np.clip(1.0 - worst, 0.0, 1.0))


def load_command(assignment: tuple, u: np.ndarray, load: int) -> float:
    """Load-``k`` reference progress speed: the mean command of its lifting robots."""
    idx = [i for i, a in enumerate(assignment) if a[0] == "lift" and a[1] == load]
    return float(np.mean(u[idx])) if idx else 0.0
