"""Closed-loop planar dynamics for WISE: rigid load + second-order unicycles.

This module grounds the strategic WISE assignment in real dynamics:

* a planar **rigid load** (pose in SE(2), mass ``m_L``, inertia ``J_L``);
* **second-order unicycles** (state ``(x, y, phi, v, omega)``, nonholonomic);
* a **heading-aligned elliptical** admissible contact set ``U_i`` --- the
  differential-drive robot exerts force freely along its heading (cap ``F_i``)
  but only ``kappa * F_i`` laterally, so wrench realizability genuinely depends
  on robot headings (the nonholonomic realizability limit, modelled honestly and
  declared, not proved away);
* the **induced communication graph** from real positions/ranges, and
* the **2x2 estimation-error system** whose stability threshold is
  ``c m_F lambda_2 > vartheta^2`` --- exactly ``sigma* = vartheta^2/(c m_F)``.

Nothing here is quasi-static: forces move mass, mass moves the graph, the graph
gates the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from . import endogenous_graph as eg


# --------------------------------------------------------------------------- #
# 2x2 estimation-error subsystem  ->  exact self-defeat threshold
# --------------------------------------------------------------------------- #


def error_jacobian(lam2: float, m_F: float, c: float, theta: float) -> np.ndarray:
    """Jacobian of the coupled (strategic error, consensus error) system::

        d/dt [e_x; e_eta] = [[-m_F,  theta],
                             [ theta, -c*lam2]] [e_x; e_eta].

    Hurwitz iff ``c m_F lam2 > theta^2`` (trace < 0 always; det > 0 is the test).
    """
    return np.array([[-m_F, theta], [theta, -c * lam2]])


def is_error_stable(lam2, m_F, c, theta) -> bool:
    """Exact stability test ``c m_F lam2 > theta^2``."""
    return c * m_F * lam2 > theta**2


def rollout_error(lam2_traj, m_F, c, theta, e0=(1.0, 0.0), dt=0.02) -> np.ndarray:
    """Integrate the 2x2 error system along a (possibly time-varying) ``lambda_2``."""
    e = np.array(e0, dtype=float)
    out = [e.copy()]
    for lam in np.asarray(lam2_traj, dtype=float):
        e = e + dt * error_jacobian(lam, m_F, c, theta) @ e
        out.append(e.copy())
    return np.array(out)


# --------------------------------------------------------------------------- #
# Rigid load + unicycles
# --------------------------------------------------------------------------- #


@dataclass
class Load:
    pose: np.ndarray            # (x, y, theta)
    twist: np.ndarray           # (vx, vy, omega)
    m: float = 3.0
    J: float = 1.5

    def step(self, wrench: np.ndarray, dt: float, damping: float = 0.6):
        acc = np.array([wrench[0] / self.m, wrench[1] / self.m, wrench[2] / self.J])
        self.twist = (1 - damping * dt) * self.twist + dt * acc
        self.pose = self.pose + dt * self.twist


def rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


@dataclass
class Unicycles:
    """Second-order unicycles: state columns (x, y, phi, v, omega)."""
    state: np.ndarray           # (N, 5)
    a_max: float = 2.0
    alpha_max: float = 4.0
    v_max: float = 2.5

    @property
    def pos(self) -> np.ndarray:
        return self.state[:, :2]

    def drive_to(self, targets: np.ndarray, active: np.ndarray, dt: float):
        """Proportional unicycle controller steering active robots to targets."""
        for i in np.where(active)[0]:
            x, y, phi, v, om = self.state[i]
            d = targets[i] - np.array([x, y])
            desired = np.arctan2(d[1], d[0])
            eang = np.arctan2(np.sin(desired - phi), np.cos(desired - phi))
            alpha = np.clip(4.0 * eang - 1.0 * om, -self.alpha_max, self.alpha_max)
            a = np.clip(1.5 * (np.linalg.norm(d) - 0.0) * np.cos(eang) - 1.0 * v,
                        -self.a_max, self.a_max)
            v = np.clip(v + dt * a, -self.v_max, self.v_max)
            om = om + dt * alpha
            self.state[i] = [x + dt * v * np.cos(phi), y + dt * v * np.sin(phi),
                             phi + dt * om, v, om]


# --------------------------------------------------------------------------- #
# Nonholonomic wrench realization
# --------------------------------------------------------------------------- #


def realize_wrench(offsets, headings, F, kappa, w_cmd, ridge=1e-3):
    """Best realizable load wrench under heading-aligned elliptical force caps.

    Solves ``min_f ||G f - w_cmd||^2 + ridge||f||^2`` s.t. each robot's force lies
    in the ellipse aligned with its heading (longitudinal cap ``F_i``, lateral
    ``kappa F_i``). Returns ``(f, realized_wrench, residual_norm)``. The residual
    is the wrench the nonholonomic team physically cannot deliver.
    """
    n = offsets.shape[0]
    if n == 0:
        return np.zeros((0, 2)), np.zeros(3), float(np.linalg.norm(w_cmd))
    # grasp map G: f_i (world) -> wrench [fx, fy, r x f]
    G = np.zeros((3, 2 * n))
    for i, r in enumerate(offsets):
        G[0, 2 * i] = 1.0
        G[1, 2 * i + 1] = 1.0
        G[2, 2 * i] = -r[1]
        G[2, 2 * i + 1] = r[0]

    def obj(f):
        return np.sum((G @ f - w_cmd) ** 2) + ridge * np.sum(f**2)

    def jac(f):
        return 2 * G.T @ (G @ f - w_cmd) + 2 * ridge * f

    cons = []
    for i in range(n):
        R = rot(headings[i])

        def c_i(f, i=i, R=R):
            fl = R.T @ f[2 * i:2 * i + 2]          # into heading frame
            return 1.0 - (fl[0] / F[i]) ** 2 - (fl[1] / (kappa * F[i])) ** 2
        cons.append({"type": "ineq", "fun": c_i})

    res = minimize(obj, np.zeros(2 * n), jac=jac, constraints=cons,
                   method="SLSQP", options={"maxiter": 100, "ftol": 1e-6})
    f = res.x.reshape(n, 2)
    realized = G @ res.x
    return f, realized, float(np.linalg.norm(realized - w_cmd))


# --------------------------------------------------------------------------- #
# Communication graph from live positions
# --------------------------------------------------------------------------- #


def live_lambda2(pos, ranges, relay_mask, base_range=2.5, scale=1.5,
                 bridge_gain=1.6, delta=0.03) -> float:
    """Algebraic connectivity of the graph induced by current positions.

    Short links (<= base_range) always on; a relaying robot additionally links to
    everyone within its full range. A weak background ``delta`` keeps lambda_2
    simple.
    """
    N = pos.shape[0]
    adj = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d <= base_range:
                adj[i, j] = adj[j, i] = np.exp(-d / scale)
    for i in np.where(relay_mask)[0]:
        for j in range(N):
            if j == i:
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            if d <= ranges[i]:
                w = bridge_gain * np.exp(-d / 3.0)
                adj[i, j] = max(adj[i, j], w)
                adj[j, i] = adj[i, j]
    L = np.diag(adj.sum(1)) - adj + delta * (np.eye(N) - np.ones((N, N)) / N)
    return eg.fiedler_value(L)
