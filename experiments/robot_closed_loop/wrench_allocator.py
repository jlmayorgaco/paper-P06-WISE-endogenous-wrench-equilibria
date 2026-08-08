"""Contact-force allocation inside the paper's inner-zonotope actuation model.

Each lifting robot of type ``tau`` may exert a contact force ``f_i = G_tau u_i`` with
``||u_i||_inf <= 1``, where ``G_tau`` are the generators of the regular ``2m``-gon
inscribed in ``disk(kappa F_tau)`` -- *exactly* the set the flagship certificate
uses (``wrench_tensor.inner_zonotope_generators``). So the closed loop can never
apply a force the assignment-level certificate did not already allow.

At each control update we solve, per load,

    min_u  || A_G u - w^dem ||^2 * RHO^2 + || G u ||^2      s.t.  -1 <= u <= 1,

a bounded linear least-squares problem (``scipy.optimize.lsq_linear``, deterministic,
warm-startable). ``RHO`` is large, so the first term enforces the equality
``sum_h A_kh f_h = w^dem_k`` whenever it is attainable and otherwise returns the
minimum-residual point; the second term is the ``min sum_i ||f_i||^2`` objective.
The residual ``r_w = ||A_G u - w^dem||_2`` is recorded at every step -- it is the
wrench the team physically cannot deliver.

The *same* allocator, with the same RHO, ridge and bounds, is used for every
assignment method, so any difference between methods is the assignment, not the
low-level control.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy.optimize import lsq_linear

from wise_mr import wrench_tensor as wt

from . import scenario as S

RHO = 5.0e3          # equality weight
RIDGE = 1.0e-6       # strict-convexity ridge on u
_LSQ_TOL = 1e-10


@cache
def generators(force_cap: float) -> np.ndarray:
    """Inner-zonotope generators ``G`` (2 x m) for a robot with cap ``force_cap``."""
    return wt.inner_zonotope_generators(S.KAPPA * force_cap, S.M_SIDES)


@cache
def _blocks(load: int, contacts: tuple[tuple[int, float], ...]):
    """Return ``(A_G, G_blk, m)``: wrench map and force map of the stacked coefficients."""
    cols = []
    gblk = []
    for h, cap in contacts:
        G = generators(float(cap))                 # (2, m)
        cols.append(S.SLOT_MAPS[load][h] @ G)      # (3, m)
        gblk.append(G)
    A_G = np.hstack(cols)                          # (3, n*m)
    m = gblk[0].shape[1]
    n = len(contacts)
    Gb = np.zeros((2 * n, n * m))
    for i, G in enumerate(gblk):
        Gb[2 * i:2 * i + 2, i * m:(i + 1) * m] = G
    return A_G, Gb, m


class Allocator:
    """Per-load force allocator with a warm start carried across control steps."""

    def __init__(self, load: int, contacts: tuple[tuple[int, float], ...]):
        self.load = load
        self.contacts = tuple(contacts)
        self.A_G, self.G_blk, self.m = _blocks(load, self.contacts)
        self.n = len(self.contacts)
        self.dim = self.A_G.shape[1]
        self._u = np.zeros(self.dim)
        self._stack = np.vstack([RHO * self.A_G, self.G_blk,
                                 np.sqrt(RIDGE) * np.eye(self.dim)])

    def solve(self, w_dem: np.ndarray):
        """Return ``(wrench, forces, residual, saturation, status)``."""
        rhs = np.concatenate([RHO * np.asarray(w_dem, float),
                              np.zeros(2 * self.n), np.zeros(self.dim)])
        res = lsq_linear(self._stack, rhs, bounds=(-1.0, 1.0), method="bvls",
                         tol=_LSQ_TOL, max_iter=200, lsq_solver="exact")
        u = res.x
        self._u = u
        wrench = self.A_G @ u
        forces = (self.G_blk @ u).reshape(self.n, 2)
        resid = float(np.linalg.norm(wrench - np.asarray(w_dem, float)))
        sat = float(np.max(np.abs(u))) if self.dim else 0.0
        return wrench, forces, resid, sat, int(res.status)


def build_allocators(assignment: tuple) -> dict[int, Allocator]:
    from . import assignments as A
    return {k: Allocator(k, A.contacts_of(assignment, k)) for k in range(S.M_LOADS)}
