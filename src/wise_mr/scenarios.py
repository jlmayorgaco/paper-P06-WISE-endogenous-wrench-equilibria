"""Reproducible two-region multi-robot scenario generator.

A load sits in the right cluster and demands a planar wrench; short-range links
connect robots only within their own cluster, so the inter-cluster gap is broken
unless a *long-range* robot commits to the ``relay`` action, which activates its
full range and bridges the clusters. This makes wrench feasibility and network
connectivity compete for the same robots --- the WISE tension --- with the
long-range fraction ``nu`` and the torque demand as the primary knobs.
"""

from __future__ import annotations

import numpy as np

from . import endogenous_graph as eg
from . import wrench_tensor as wt
from .equilibrium import WiseProblem

# domain / cluster geometry (fixed so instances are comparable across seeds)
LEFT_CENTER = np.array([2.0, 5.0])
RIGHT_CENTER = np.array([8.0, 5.0])
GAP_CENTER = np.array([5.0, 5.0])
CLUSTER_SPREAD = 1.2
BASE_RANGE = 2.5           # short-link radius (does not cross the gap)
SLOT_RADIUS = 0.6


def _slot_maps(n_slots: int) -> tuple[np.ndarray, np.ndarray]:
    """Contact-to-wrench maps and slot offsets for one planar load."""
    ang = np.linspace(0, 2 * np.pi, n_slots, endpoint=False)
    offs = SLOT_RADIUS * np.stack([np.cos(ang), np.sin(ang)], axis=1)   # (H, 2)
    A = np.zeros((n_slots, 3, 2))
    for h, (rx, ry) in enumerate(offs):
        A[h] = np.array([[1.0, 0.0], [0.0, 1.0], [-ry, rx]])            # wrench of a force
    return A, offs


def _directions() -> np.ndarray:
    """P dual wrench directions: 8 in-plane forces + 2 pure torques (one load)."""
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    force = np.stack([np.cos(ang), np.sin(ang), np.zeros(8)], axis=1)
    torque = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    return np.vstack([force, torque])[None, :, :]                       # (1, P, 3)


def two_region(
    seed: int,
    N: int = 12,
    nu: float = 0.5,
    tau_d: float = 3.0,
    lift: float = 4.0,
    theta: float = 0.5,
    c: float = 1.0,
    m_F: float = 1.0,
    bridge_gain: float = 1.6,
) -> WiseProblem:
    """Generate a reproducible two-region WISE instance."""
    rng = np.random.default_rng(seed)
    n_long = int(round(nu * N))

    # positions: split roughly evenly between the two clusters
    pos = np.zeros((N, 2))
    for i in range(N):
        center = LEFT_CENTER if i % 2 == 0 else RIGHT_CENTER
        pos[i] = center + CLUSTER_SPREAD * rng.standard_normal(2)

    # heterogeneity: capacity and communication range
    F = rng.uniform(0.8, 1.6, size=N)                       # force capacity
    is_long = np.zeros(N, dtype=bool)
    is_long[rng.choice(N, size=n_long, replace=False)] = True
    r = np.where(is_long, rng.uniform(5.5, 7.0, size=N), rng.uniform(1.5, 2.5, size=N))

    # load + contact geometry (M = 1)
    load = RIGHT_CENTER.copy()
    A_maps, offs = _slot_maps_full(load)                     # (M,H,3,2), slot world pos
    H = A_maps.shape[1]
    directions = _directions()
    w_dem = np.array([[0.0, lift, tau_d]])                   # (M, 3): lift + torque

    W = wt.build_wrench_tensor(F, A_maps, directions)        # (N, M, H, P)

    # action costs g[i, a]; A = M*H + 2  (slots, relay, idle)
    A_actions = H + 2
    g = np.zeros((N, A_actions))
    slot_world = load + offs                                 # (H, 2)
    for i in range(N):
        for h in range(H):
            g[i, h] = 0.35 * np.linalg.norm(pos[i] - slot_world[h])   # move-to-slot cost
        g[i, H] = 0.20 * np.linalg.norm(pos[i] - GAP_CENTER)          # relay cost
        g[i, H + 1] = 0.15                                            # idle outside option

    # base Laplacian: short links within reach (no gap crossing) plus a weak
    # background delta*(I - 11^T/N) that keeps lambda_2 simple and > 0, so the
    # Fiedler gradient stays well defined even when the clusters are otherwise
    # disconnected (regularises the eigenvalue-multiplicity nonsmoothness).
    delta = 0.03
    base_L = _proximity_laplacian(pos, np.minimum.outer(r, r), BASE_RANGE, scale=1.5)
    base_L = base_L + delta * (np.eye(N) - np.ones((N, N)) / N)

    # relay Laplacians B_i: robot i using its full range r_i when relaying
    relay_L = np.zeros((N, N, N))
    for i in range(N):
        adj = np.zeros((N, N))
        for j in range(N):
            if j == i:
                continue
            dij = np.linalg.norm(pos[i] - pos[j])
            if dij <= r[i]:
                adj[i, j] = adj[j, i] = bridge_gain * np.exp(-dij / 3.0)
        relay_L[i] = np.diag(adj.sum(1)) - adj

    sigma = eg.sigma_star(theta, c, m_F)

    prob = WiseProblem(
        N=N, M=1, H=H, P=directions.shape[1], W=W, directions=directions, w_dem=w_dem,
        g=g, base_laplacian=base_L, relay_laplacians=relay_L, sigma=sigma,
    )
    prob.meta = dict(seed=seed, nu=nu, tau_d=tau_d, lift=lift, pos=pos, F=F, r=r,
                     is_long=is_long, load=load, slot_world=slot_world)
    return prob


def _slot_maps_full(load: np.ndarray, n_slots: int = 4):
    A, offs = _slot_maps(n_slots)
    return A[None, :, :, :], offs                            # (1,H,3,2), (H,2)


def _proximity_laplacian(pos, reach, radius, scale=1.5) -> np.ndarray:
    """Weighted Laplacian of links shorter than ``min(radius, reach_ij)``."""
    N = pos.shape[0]
    adj = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d <= min(radius, reach[i, j]):
                adj[i, j] = adj[j, i] = np.exp(-d / scale)
    return np.diag(adj.sum(1)) - adj
