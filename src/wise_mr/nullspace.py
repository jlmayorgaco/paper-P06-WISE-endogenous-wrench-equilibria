"""Productive-nullspace / composition-fiber dimension (TASK 4).

At a relative-interior point ``xbar`` of the optimal fiber
``E = {x in X_f : B x = y*}``, with the constraint description
``A x = b`` (mass conservation), ``B x = y*`` (served aggregate), and active
inequalities ``G_I x = h_I`` (nonnegativity / active wrench), the *minimal face*
``F`` containing ``E`` has lineality

    lin F = ker(G_I),
    dim E = dim( ker(B) cap ker(A) cap ker(G_I) )
          = n - rank([A; B; G_I])              (removing dependent rows).

The productive neutral space ``N_p(xbar) = ker(A) cap ker(B) cap ker(G_I)`` is the
set of mass-preserving, productively neutral (``B d = 0``) composition changes that
respect the active constraints. ``E`` is locally positive-dimensional iff
``N_p != {0}``. This corrects the earlier ``ker B cap aff(X_f)`` statement (which
mixed a linear space with an affine set and ignored active inequalities).
"""

from __future__ import annotations

import numpy as np

from .equilibrium import WiseProblem


def mass_matrix(problem: WiseProblem) -> np.ndarray:
    """Equality map ``A`` for per-robot mass conservation ``sum_a x_ia = 1``; ``(N, N A)``."""
    N, A = problem.N, problem.A
    M = np.zeros((N, N * A))
    for i in range(N):
        M[i, i * A:(i + 1) * A] = 1.0
    return M


def served_matrix(problem: WiseProblem) -> np.ndarray:
    """Productive aggregate map ``B`` with ``y = B x``; ``(M, N A)``."""
    N, A, H = problem.N, problem.A, problem.H
    cap = problem._cap()
    B = np.zeros((problem.M, N * A))
    for i in range(N):
        for k in range(problem.M):
            for h in range(H):
                B[k, i * A + (k * H + h)] = cap[i]
    return B


def active_inequalities(problem: WiseProblem, x, tol: float = 1e-3) -> np.ndarray:
    """Rows of the active inequality set ``G_I`` at ``x``.

    Nonnegativity ``x_ia >= 0`` active where ``x_ia ~ 0`` (rows are unit vectors),
    plus active wrench rows ``H_w`` where ``s_kl ~ d_kl``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    rows = []
    for j in range(n):
        if x[j] <= tol:
            e = np.zeros(n); e[j] = 1.0
            rows.append(e)
    Hw = problem.wrench_matrix()                     # (M P, N A)
    s = (Hw @ x).reshape(problem.M, problem.P)
    d = problem.demand()
    for k in range(problem.M):
        for ell in range(problem.P):
            if abs(s[k, ell] - d[k, ell]) <= tol and d[k, ell] > 0:
                rows.append(Hw[k * problem.P + ell])
    return np.array(rows) if rows else np.zeros((0, n))


def fiber_dimension(problem: WiseProblem, x, tol: float = 1e-3) -> dict:
    """Local dimension of the composition fiber ``E`` at ``x`` and the neutral space.

    Returns dim, rank, the null-space (N_p) basis, and constraint residuals.
    """
    A = mass_matrix(problem)
    B = served_matrix(problem)
    G_I = active_inequalities(problem, x, tol)
    stack = np.vstack([A, B, G_I]) if G_I.size else np.vstack([A, B])
    n = problem.N * problem.A
    rank = int(np.linalg.matrix_rank(stack, tol=1e-8))
    dim = n - rank
    # basis of N_p = null(stack)
    _, sv, Vt = np.linalg.svd(stack)
    null_mask = np.concatenate([sv, np.zeros(n - sv.size)]) <= 1e-8
    Np = Vt[null_mask] if null_mask.any() else np.zeros((0, n))
    # residuals for each neutral direction
    res = []
    for d in Np:
        res.append({
            "Ad": float(np.linalg.norm(A @ d)),
            "Bd": float(np.linalg.norm(B @ d)),
            "GId": float(np.linalg.norm(G_I @ d)) if G_I.size else 0.0,
        })
    return {"dim_E": int(dim), "rank": rank, "n": n,
            "n_active_ineq": int(G_I.shape[0]), "Np_basis": Np, "residuals": res}
