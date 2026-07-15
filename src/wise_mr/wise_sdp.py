r"""Exact WISE existence and selection as a semidefinite program (paper Thm. 2).

Given the unique optimal aggregate ``y*`` (stage 1), the connectivity-optimal
productively optimal assignment is the solution of the stage-2 SDP

    Lambda_E = max_{z,t}  t
        s.t.  A_eq z = b_eq            (budget conservation, Bz = y*)
              G z <= h                 (nonnegativity, occupancy, wrench facets)
              Q^T Lbar(z) Q  >=  t I   (algebraic-connectivity LMI)

with the affine Laplacian ``Lbar(z) = L0 + sum_j z_j M_j`` and ``Q`` an orthonormal
basis of ``1^perp``.  A WISE exists iff ``Lambda_E >= sigma_req``; the maximiser
``z*`` is the lexicographic selector -- most connected among productively optimal
assignments -- and preserves ``Bz* = y*`` exactly.  ``lambda_2`` is concave in the
affine weights, so the program is convex.

Requires ``cvxpy`` (optional dependency ``opt``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WiseSDPResult:
    lambda_star: float          # optimal value Lambda_E = max_E lambda_2
    z: np.ndarray               # selector z*_WISE
    wise_exists: bool           # Lambda_E >= sigma_req
    sigma_req: float
    status: str
    lambda2_check: float        # lambda_2(Lbar(z*)) recomputed by eigendecomposition


def _induced(L0: np.ndarray, terms: list[tuple[int, np.ndarray]], z: np.ndarray) -> np.ndarray:
    L = np.array(L0, dtype=float)
    for j, M in terms:
        L = L + float(z[j]) * np.asarray(M, dtype=float)
    return L


def solve_wise_sdp(
    n: int,
    L0: np.ndarray,
    lap_terms: list[tuple[int, np.ndarray]],
    Q: np.ndarray,
    sigma_req: float,
    *,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    G_ineq: np.ndarray | None = None,
    h_ineq: np.ndarray | None = None,
    solver: str | None = None,
) -> WiseSDPResult:
    """Solve the stage-2 WISE selection SDP (paper eq. for Thm. 2).

    Parameters
    ----------
    n : number of decision variables ``z``.
    L0 : base Laplacian ``(V, V)``.
    lap_terms : list of ``(j, M_j)`` so ``Lbar(z) = L0 + sum_j z_j M_j``.
    Q : orthonormal basis of ``1^perp``, shape ``(V, V-1)``.
    sigma_req : robust connectivity requirement.
    A_eq, b_eq, G_ineq, h_ineq : optional linear (in)equality data on ``z``.
    """
    import cvxpy as cp

    V = np.asarray(L0, dtype=float).shape[0]
    m = Q.shape[1]
    z = cp.Variable(n)
    t = cp.Variable()

    Lz = cp.Constant(np.asarray(L0, dtype=float))
    for j, M in lap_terms:
        Lz = Lz + z[j] * np.asarray(M, dtype=float)
    QLQ = Q.T @ Lz @ Q

    cons = [QLQ - t * np.eye(m) >> 0]
    if A_eq is not None:
        cons.append(np.asarray(A_eq, float) @ z == np.asarray(b_eq, float))
    if G_ineq is not None:
        cons.append(np.asarray(G_ineq, float) @ z <= np.asarray(h_ineq, float))

    prob = cp.Problem(cp.Maximize(t), cons)
    prob.solve(solver=solver)

    zval = np.asarray(z.value, dtype=float).reshape(-1) if z.value is not None else np.full(n, np.nan)
    lam = float(t.value) if t.value is not None else float("nan")
    # independent recomputation of lambda_2 at the returned selector
    if np.all(np.isfinite(zval)):
        w = np.linalg.eigvalsh(0.5 * (_induced(L0, lap_terms, zval) + _induced(L0, lap_terms, zval).T))
        lam2 = float(np.sort(w)[1])
    else:
        lam2 = float("nan")
    return WiseSDPResult(
        lambda_star=lam,
        z=zval,
        wise_exists=bool(np.isfinite(lam) and lam >= sigma_req - 1e-7),
        sigma_req=float(sigma_req),
        status=str(prob.status),
        lambda2_check=lam2,
    )
