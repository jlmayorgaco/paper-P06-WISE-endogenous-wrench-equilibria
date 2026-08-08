"""Planar rigid-load dynamics (body-frame twist), shared by every method.

    M_k nu_dot = -C(nu) nu - D_k nu + w_contact + d_k(t),      q_dot = T(theta) nu

with ``q = (x, y, theta)`` the world pose, ``nu = (v_x^b, v_y^b, omega)`` the body
twist and ``T(theta) = blkdiag(R(theta), 1)``. Contact wrenches and disturbances
are expressed in the *body* frame, exactly like the frozen contact maps ``A_kh``
and the certified demands ``w^dem_k``, so the wrench certificate transfers
verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def twist_to_world(q: np.ndarray, nu: np.ndarray) -> np.ndarray:
    v = rot(q[2]) @ nu[:2]
    return np.array([v[0], v[1], nu[2]])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class RigidLoad:
    """Planar rigid load with *eccentric* linear drag.

    The drag force ``(-D_x v_x, -D_y v_y)`` acts at the body point ``(0, ecc)``, so it
    also produces a torque ``ecc * D_x v_x``. Choosing
    ``ecc = -w^dem_{k,tau} / w^dem_{k,x}`` makes the whole resistance wrench
    *collinear with the certified demand* ``w^dem_k``: the team always has to push
    along the direction its assignment was certified for, and the certified demand is
    exactly the resistance at the fastest mission speed. That is what makes the
    frozen wrench certificate meaningful in closed loop instead of decorative.
    """
    mass: float
    inertia: float
    damp_lin: float
    damp_rot: float
    ecc: float = 0.0

    @property
    def M(self) -> np.ndarray:
        return np.diag([self.mass, self.mass, self.inertia])

    def resistance(self, nu: np.ndarray, drag_scale: float = 1.0) -> np.ndarray:
        """Body-frame resistance wrench (negative of what the team must supply)."""
        dx = drag_scale * self.damp_lin
        fx, fy = -dx * nu[0], -dx * nu[1]
        tau = -self.damp_rot * nu[2] - self.ecc * fx
        return np.array([fx, fy, tau])

    def coriolis(self, nu: np.ndarray) -> np.ndarray:
        """Body-frame Coriolis term C(nu) nu for a planar rigid body."""
        m, w = self.mass, nu[2]
        return np.array([-m * w * nu[1], m * w * nu[0], 0.0])

    def accel(self, nu: np.ndarray, w_contact: np.ndarray, disturbance: np.ndarray,
              drag_scale: float = 1.0) -> np.ndarray:
        rhs = (-self.coriolis(nu) + self.resistance(nu, drag_scale)
               + w_contact + disturbance)
        return np.array([rhs[0] / self.mass, rhs[1] / self.mass, rhs[2] / self.inertia])

    def rk4(self, q: np.ndarray, nu: np.ndarray, w_contact: np.ndarray,
            disturbance: np.ndarray, dt: float, drag_scale: float = 1.0):
        """One fixed-step RK4 update with the contact wrench held over the step."""
        def f(state):
            qq, vv = state[:3], state[3:]
            return np.concatenate([twist_to_world(qq, vv),
                                   self.accel(vv, w_contact, disturbance, drag_scale)])

        y = np.concatenate([q, nu])
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y[:3], y[3:]
