"""Directional wrench-capacity tensor.

For robot ``i``, load ``k``, contact slot ``h`` and dual wrench direction
``eta_kl in R^3`` the tensor entry is the support function of the robot's
admissible planar force set ``U_i`` evaluated on the pulled-back direction::

    W[i, k, h, l] = h_{U_i}( A_kh^T eta_kl )

with ``A_kh in R^{3x2}`` the contact-to-wrench map (force-x, force-y, torque).
The directional capacity aggregated over the assignment ``x_ikh in [0, 1]`` is
``s_kl(x) = sum_{i,h} W[i,k,h,l] x_ikh`` (affine in x), and the coalition is
wrench-feasible for load ``k`` iff ``s_kl(x) >= d_kl := eta_kl^T w_k^dem`` for a
complete set of facet normals ``{eta_kl}`` (Prop. tensorized membership).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

# --------------------------------------------------------------------------- #
# Support functions of admissible force sets U_i \subset R^2
# --------------------------------------------------------------------------- #


def support_disk(radius: float, direction: np.ndarray) -> float:
    """Support function of a disk of the given radius: ``h_U(d) = radius * ||d||``."""
    return float(radius) * float(np.linalg.norm(direction))


def support_box(lower: np.ndarray, upper: np.ndarray, direction: np.ndarray) -> float:
    """Support function of an axis-aligned box ``[lower, upper]``.

    ``h_U(d) = sum_j max(d_j lo_j, d_j hi_j)``.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    direction = np.asarray(direction, dtype=float)
    return float(np.sum(np.maximum(direction * lower, direction * upper)))


# --------------------------------------------------------------------------- #
# Tensor construction and aggregation
# --------------------------------------------------------------------------- #


def build_wrench_tensor(
    force_caps: np.ndarray, contact_maps: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Assemble ``W[i,k,h,l] = F_i * || A_kh^T eta_kl ||`` for disk force sets.

    Parameters
    ----------
    force_caps   : ndarray ``(N,)`` — force-magnitude limit ``F_i`` of each robot.
    contact_maps : ndarray ``(M, H, 3, 2)`` — contact-to-wrench maps ``A_kh``.
    directions   : ndarray ``(M, P, 3)`` — dual wrench directions ``eta_kl``.

    Returns
    -------
    W : ndarray ``(N, M, H, P)``.
    """
    force_caps = np.asarray(force_caps, dtype=float)
    A = np.asarray(contact_maps, dtype=float)          # (M, H, 3, 2)
    eta = np.asarray(directions, dtype=float)          # (M, P, 3)
    # pulled-back directions g[k,h,l,:] = A_kh^T eta_kl  in R^2
    g = np.einsum("khda,kld->khla", A, eta)            # (M, H, P, 2)
    norms = np.linalg.norm(g, axis=-1)                 # (M, H, P)
    # W[i,k,h,l] = F_i * norms[k,h,l]
    return np.einsum("i,khl->ikhl", force_caps, norms)


def directional_capacity(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Aggregate directional capacity ``s_kl(x) = sum_{i,h} W[i,k,h,l] x_ikh``.

    ``W`` has shape ``(N, M, H, P)`` and ``x`` shape ``(N, M, H)``; returns ``(M, P)``.
    """
    W = np.asarray(W, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.einsum("ikhl,ikh->kl", W, x)


def demand_projection(directions: np.ndarray, w_dem: np.ndarray) -> np.ndarray:
    """Project demands onto directions: ``d_kl = eta_kl^T w_k^dem``. Returns ``(M, P)``."""
    directions = np.asarray(directions, dtype=float)
    w_dem = np.asarray(w_dem, dtype=float)
    return np.einsum("klm,km->kl", directions, w_dem)


def wrench_residual(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Positive part of the per-direction wrench deficit ``[d - s]_+``."""
    return np.maximum(np.asarray(d, dtype=float) - np.asarray(s, dtype=float), 0.0)


def is_wrench_feasible(s: np.ndarray, d: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Per-load feasibility ``all_l s_kl >= d_kl``; returns boolean ``(M,)``."""
    return np.all(wrench_residual(s, d) <= tol, axis=1)


def certify_membership_lp(
    force_caps: np.ndarray,
    contact_maps: np.ndarray,
    w_dem: np.ndarray,
    n_sides: int = 16,
) -> bool:
    """Exact LP check of ``w_dem in W(C) = sum_{i,h} A_ih U_i`` (disks polygonised).

    Solves the feasibility LP: exist contact forces ``f_ih in poly(F_i)`` with
    ``sum_ih A_ih f_ih = w_dem``. Used as a reference oracle for the tensorized
    separation test.
    """
    F = np.asarray(force_caps, dtype=float)             # (N,)
    A = np.asarray(contact_maps, dtype=float)           # (N, 3, 2) one slot per robot
    w = np.asarray(w_dem, dtype=float)                  # (3,)
    N = F.shape[0]
    ang = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (n_sides, 2)
    # variables: f in R^{2N}; equality A f = w (3 rows); ||f_i|| <= F_i via facets
    A_eq = np.zeros((3, 2 * N))
    for i in range(N):
        A_eq[:, 2 * i : 2 * i + 2] = A[i]
    A_ub_blocks, b_ub = [], []
    for i in range(N):
        blk = np.zeros((n_sides, 2 * N))
        blk[:, 2 * i : 2 * i + 2] = dirs
        A_ub_blocks.append(blk)
        b_ub.append(F[i] * np.ones(n_sides))
    res = linprog(
        c=np.zeros(2 * N),
        A_ub=np.vstack(A_ub_blocks),
        b_ub=np.concatenate(b_ub),
        A_eq=A_eq,
        b_eq=w,
        bounds=[(None, None)] * (2 * N),
        method="highs",
    )
    return bool(res.success)
