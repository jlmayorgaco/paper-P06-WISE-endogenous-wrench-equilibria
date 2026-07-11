r"""Support-function wrench certificate and support-set geometry.

Two force models are kept deliberately distinct (see paper Prop.\ 1):

* **Physical** admissible force of a robot with heading ``phi``: the
  *heading-aligned ellipse* with longitudinal semi-axis ``F`` and lateral
  semi-axis ``kappa*F`` (``kappa<=1`` encodes the nonholonomic penalty).
* **Certificate** set: a centrally symmetric **inner regular 2m-gon** inscribed
  in the disk of radius ``kappa*F``. That disk is contained in the ellipse for
  *every* heading (its minimum semi-axis is ``kappa*F``), so the polygon is
  contained in the physical set for every heading. Feasibility computed against
  the inner polygon is therefore **conservative**: a certified wrench is
  realizable by the physical (elliptical) force set, so the certificate cannot
  produce a false positive.

Because the inner set is a zonotope (a regular 2m-gon), the reachable wrench set
``W_k(x) = (+)_{tau,h} x_{tau,k,h} A_{k,h} U_tau^in`` is a zonotope (closed under
linear maps and Minkowski sums), and the finite facet-normal / LP membership test
is *exact for the inner set*.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

# --------------------------------------------------------------------------- #
# Support functions
# --------------------------------------------------------------------------- #


def support_disk(radius: float, direction: np.ndarray) -> float:
    """Support of a disk: ``h(d) = radius * ||d||``."""
    return float(radius) * float(np.linalg.norm(direction))


def support_box(lower: np.ndarray, upper: np.ndarray, direction: np.ndarray) -> float:
    """Support of an axis-aligned box ``[lower, upper]``."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    direction = np.asarray(direction, dtype=float)
    return float(np.sum(np.maximum(direction * lower, direction * upper)))


def _rot(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def support_ellipse(semi_long: float, semi_lat: float, heading: float,
                    direction: np.ndarray) -> float:
    """Support of the heading-aligned ellipse (physical force set).

    ``h(d) = || diag(semi_long, semi_lat) R(heading)^T d ||``.
    """
    d = np.asarray(direction, dtype=float)
    dl = _rot(heading).T @ d
    return float(np.linalg.norm([semi_long * dl[0], semi_lat * dl[1]]))


def support_polygon(vertices: np.ndarray, direction: np.ndarray) -> float:
    """Support of a convex polygon = ``max_j <vertex_j, direction>``."""
    V = np.asarray(vertices, dtype=float)
    return float(np.max(V @ np.asarray(direction, dtype=float)))


def support_zonotope(generators: np.ndarray, direction: np.ndarray) -> float:
    """Support of a centered zonotope ``{sum lambda_i g_i : |lambda_i|<=1}``.

    ``h(d) = sum_i |<g_i, d>|``.
    """
    G = np.asarray(generators, dtype=float)
    return float(np.sum(np.abs(G @ np.asarray(direction, dtype=float))))


# --------------------------------------------------------------------------- #
# Inner certificate set: regular 2m-gon inscribed in disk(radius)
# --------------------------------------------------------------------------- #


def inner_polygon(radius: float, m_sides: int = 8) -> np.ndarray:
    """Vertices of a centrally symmetric regular ``2*m_sides``-gon in disk(radius).

    Vertices lie on the circle of the given radius, so the polygon is contained
    in that disk; a regular ``2m``-gon is a zonotope.
    """
    ang = np.pi * np.arange(2 * m_sides) / m_sides
    return float(radius) * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def polygon_facets(m_sides: int) -> tuple[np.ndarray, float]:
    """Outward facet normals (unit) and apothem of the *unit-circumradius* 2m-gon.

    Returns ``(normals (2m,2), apothem)`` with ``apothem = cos(pi/(2m))``; the
    polygon is ``{f : normals @ f <= apothem}`` scaled by the circumradius.
    """
    ang = np.pi * (np.arange(2 * m_sides) + 0.5) / m_sides
    normals = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return normals, float(np.cos(np.pi / (2 * m_sides)))


# --------------------------------------------------------------------------- #
# Support-capability map and aggregate (certificate uses the inner set)
# --------------------------------------------------------------------------- #


def build_wrench_tensor(force_caps: np.ndarray, contact_maps: np.ndarray,
                        directions: np.ndarray, kappa: float = 1.0,
                        m_sides: int = 8) -> np.ndarray:
    """Assemble ``W[i,k,h,l] = h_{U_i^in}(A_kh^T eta_kl)`` for the inner 2m-gon.

    ``U_i^in`` is the regular ``2*m_sides``-gon inscribed in ``disk(kappa*F_i)``.
    With ``kappa < 1`` the certificate is conservative w.r.t. the heading-aligned
    ellipse ``(F_i, kappa*F_i)``.
    """
    F = np.asarray(force_caps, dtype=float)              # (N,)
    A = np.asarray(contact_maps, dtype=float)            # (M,H,3,2)
    eta = np.asarray(directions, dtype=float)            # (M,P,3)
    g = np.einsum("khda,kld->khla", A, eta)              # (M,H,P,2) pulled-back dirs
    uv = inner_polygon(1.0, m_sides)                     # (2m,2) unit polygon
    # support of the unit inner polygon along each pulled-back direction
    s_unit = np.max(np.einsum("khla,va->khlv", g, uv), axis=-1)   # (M,H,P)
    # W[i,k,h,l] = kappa * F_i * s_unit[k,h,l]
    return float(kappa) * np.einsum("i,khl->ikhl", F, s_unit)


def directional_capacity(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``s_kl(x) = sum_{i,h} W[i,k,h,l] x_ikh``; ``(N,M,H,P)x(N,M,H)->(M,P)``."""
    return np.einsum("ikhl,ikh->kl", np.asarray(W, float), np.asarray(x, float))


def demand_projection(directions: np.ndarray, w_dem: np.ndarray) -> np.ndarray:
    """``d_kl = eta_kl^T w_k^dem``; ``(M,P,3)x(M,3)->(M,P)``."""
    return np.einsum("klm,km->kl", np.asarray(directions, float), np.asarray(w_dem, float))


def wrench_residual(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Positive part of the per-direction deficit ``[d - s]_+``."""
    return np.maximum(np.asarray(d, float) - np.asarray(s, float), 0.0)


def is_wrench_feasible(s: np.ndarray, d: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Per-load necessary support test ``all_l s_kl >= d_kl``; boolean ``(M,)``."""
    return np.all(wrench_residual(s, d) <= tol, axis=1)


# --------------------------------------------------------------------------- #
# Exact membership oracles
# --------------------------------------------------------------------------- #


def certify_membership_lp(force_caps: np.ndarray, contact_maps: np.ndarray,
                          w_dem: np.ndarray, kappa: float = 1.0,
                          m_sides: int = 8) -> bool:
    """Exact membership ``w_dem in (+)_i A_i (kappa F_i * inner-polygon)`` via LP.

    One contact slot per robot (``contact_maps`` shape ``(N,3,2)``). Feasibility
    against the **inner** (certificate) set, hence conservative.
    """
    F = np.asarray(force_caps, dtype=float)              # (N,)
    A = np.asarray(contact_maps, dtype=float)            # (N,3,2)
    w = np.asarray(w_dem, dtype=float)                   # (3,)
    N = F.shape[0]
    normals, apothem = polygon_facets(m_sides)           # (2m,2), scalar
    A_eq = np.zeros((3, 2 * N))
    for i in range(N):
        A_eq[:, 2 * i:2 * i + 2] = A[i]
    ub_rows, ub_b = [], []
    for i in range(N):
        blk = np.zeros((normals.shape[0], 2 * N))
        blk[:, 2 * i:2 * i + 2] = normals
        ub_rows.append(blk)
        ub_b.append(kappa * F[i] * apothem * np.ones(normals.shape[0]))
    res = linprog(c=np.zeros(2 * N), A_ub=np.vstack(ub_rows), b_ub=np.concatenate(ub_b),
                  A_eq=A_eq, b_eq=w, bounds=[(None, None)] * (2 * N), method="highs")
    return bool(res.success)


def realizable_by_ellipse(force_caps: np.ndarray, headings: np.ndarray,
                          contact_maps: np.ndarray, w_dem: np.ndarray,
                          kappa: float = 1.0, tol: float = 1e-6) -> bool:
    """Exact membership in the *physical* ellipse sets (SOCP via QP residual).

    Returns True iff contact forces inside the heading-aligned ellipses realise
    ``w_dem`` (residual below ``tol``). Used to check the certificate never
    produces a false positive.
    """
    from scipy.optimize import minimize

    F = np.asarray(force_caps, dtype=float)
    A = np.asarray(contact_maps, dtype=float)            # (N,3,2)
    w = np.asarray(w_dem, dtype=float)
    N = F.shape[0]
    G = np.zeros((3, 2 * N))
    for i in range(N):
        G[:, 2 * i:2 * i + 2] = A[i]

    def obj(f):
        return float(np.sum((G @ f - w) ** 2))

    cons = []
    for i in range(N):
        R = _rot(headings[i])

        def c_i(f, i=i, R=R):
            fl = R.T @ f[2 * i:2 * i + 2]
            return 1.0 - (fl[0] / F[i]) ** 2 - (fl[1] / (kappa * F[i])) ** 2
        cons.append({"type": "ineq", "fun": c_i})
    res = minimize(obj, np.zeros(2 * N), constraints=cons, method="SLSQP",
                   options={"maxiter": 200, "ftol": 1e-9})
    return bool(res.fun < tol)
