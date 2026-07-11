"""Endogenous communication graph induced by the assignment.

Each potential edge ``e`` carries an affine weight ``a_e(x) = sum_{i,r} C_ire x_ir``
(``C_ire >= 0``), so the induced Laplacian ``L(x) = sum_e a_e(x) L_e`` is affine in
the decisions. The information constraint is written as the LMI

    Q^T L(x) Q  >=  sigma * I      (Q = orthonormal basis of 1^perp),

equivalently ``lambda_2(L(x)) >= sigma*``. Because ``lambda_2`` is concave in the
weights, the super-level set is convex, and the matrix dual ``Z* >= 0`` collapses
to the rank-one ``Z* = pi* v v^T`` at a simple minimiser. The threshold comes from
Nash seeking: ``sigma* = theta^2 / (c m_F)``.
"""

from __future__ import annotations

import numpy as np


def sigma_star(theta: float, c: float, m_F: float) -> float:
    """Information self-sustainability threshold ``sigma* = theta^2 / (c m_F)``."""
    if c <= 0 or m_F <= 0:
        raise ValueError("consensus gain c and monotonicity m_F must be positive")
    return float(theta) ** 2 / (float(c) * float(m_F))


def complement_basis(n: int) -> np.ndarray:
    """Orthonormal basis ``Q in R^{n x (n-1)}`` of the space orthogonal to ``1``."""
    a = np.ones((n, 1)) / np.sqrt(n)
    # QR of [1/sqrt(n) | I]; columns 1.. give an orthonormal complement
    m = np.hstack([a, np.eye(n)])
    q, _ = np.linalg.qr(m)
    return q[:, 1:n]


def edge_weights(C: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Affine edge weights ``a_e(x) = sum_{i,r} C[i,r,e] x[i,r]``; returns ``(E,)``."""
    C = np.asarray(C, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.einsum("ire,ir->e", C, x)


def induced_laplacian(edge_laplacians: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Assemble ``L(x) = sum_e a_e L_e``; ``(E,V,V) x (E,) -> (V,V)``."""
    edge_laplacians = np.asarray(edge_laplacians, dtype=float)
    a = np.asarray(a, dtype=float)
    return np.einsum("e,eij->ij", a, edge_laplacians)


def fiedler_pair(L: np.ndarray) -> tuple[float, np.ndarray]:
    """Return ``(lambda_2, v)``: the algebraic connectivity and unit Fiedler vector."""
    L = np.asarray(L, dtype=float)
    w, V = np.linalg.eigh(0.5 * (L + L.T))
    return float(w[1]), V[:, 1]


def fiedler_value(L: np.ndarray) -> float:
    """Algebraic connectivity ``lambda_2(L)`` (second-smallest eigenvalue)."""
    return fiedler_pair(L)[0]


def sdp_margin(L: np.ndarray, sigma: float) -> float:
    """Information margin ``m_I = lambda_min(Q^T L Q) - sigma = lambda_2(L) - sigma``."""
    return fiedler_value(L) - float(sigma)


def lambda2_gradient(edge_laplacians: np.ndarray, C: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Gradient ``d lambda_2 / d x_ir`` at Fiedler vector ``v`` (valid where simple).

    ``d lambda_2 / d a_e = v^T L_e v`` and ``d a_e / d x_ir = C[i,r,e]``, so
    ``d lambda_2 / d x_ir = sum_e C[i,r,e] (v^T L_e v)``.
    """
    edge_laplacians = np.asarray(edge_laplacians, dtype=float)
    C = np.asarray(C, dtype=float)
    v = np.asarray(v, dtype=float)
    dl_da = np.einsum("i,eij,j->e", v, edge_laplacians, v)   # (E,)
    return np.einsum("ire,e->ir", C, dl_da)                  # (N, R)


def info_price_matrix(pi: float, v: np.ndarray) -> np.ndarray:
    """Rank-one PSD information price ``Z* = pi * v v^T`` (simple-eigenvalue case)."""
    v = np.asarray(v, dtype=float)
    return float(pi) * np.outer(v, v)


def is_information_sustaining(L: np.ndarray, sigma: float, tol: float = 0.0) -> bool:
    """Information constraint ``lambda_2(L(x)) >= sigma*``."""
    return fiedler_value(L) >= float(sigma) - tol
