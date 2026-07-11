"""TASK 2 gates: correct support-set geometry and conservative wrench certificate.

Covers: support additivity under Minkowski sums, linear-image support identity,
a triangle counterexample to "polytope implies zonotope", zonotope closure under
linear maps and Minkowski sums, inner-approximation containment, and the key
property that the certificate never certifies a wrench the physical (elliptical)
force set cannot realise.
"""

import numpy as np
import pytest

from wise_mr import wrench_tensor as wt


# --- helpers ---------------------------------------------------------------
def support_pointset(points, d):
    """Support of a finite point set (= support of its convex hull)."""
    return float(np.max(np.asarray(points) @ np.asarray(d)))


def minkowski_sum_points(A, B):
    return np.array([a + b for a in A for b in B])


# 1. Support-function additivity under Minkowski sums ------------------------
def test_support_additive_under_minkowski_sum():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 2))
    B = rng.standard_normal((4, 2))
    S = minkowski_sum_points(A, B)
    for _ in range(20):
        d = rng.standard_normal(2)
        assert support_pointset(S, d) == pytest.approx(
            support_pointset(A, d) + support_pointset(B, d), abs=1e-9)


# 2. Linear-image support identity: h_{MU}(d) = h_U(M^T d) ------------------
def test_linear_image_support_identity():
    rng = np.random.default_rng(1)
    U = rng.standard_normal((6, 2))
    M = rng.standard_normal((3, 2))
    MU = U @ M.T                                  # image points in R^3
    for _ in range(20):
        d = rng.standard_normal(3)
        assert support_pointset(MU, d) == pytest.approx(
            support_pointset(U, M.T @ d), abs=1e-9)


# 3. Triangle counterexample: a polytope need not be a zonotope -------------
def test_triangle_is_not_a_zonotope():
    # A zonotope in R^2 is centrally symmetric; an equilateral triangle is not.
    tri = np.array([[1.0, 0.0],
                    [-0.5, np.sqrt(3) / 2],
                    [-0.5, -np.sqrt(3) / 2]])
    centroid = tri.mean(axis=0)
    # central symmetry about the centroid would require -(v-c)+c to be a vertex
    reflected = 2 * centroid - tri
    # no reflected vertex coincides with an original vertex -> not symmetric
    symmetric = all(any(np.allclose(r, v, atol=1e-9) for v in tri) for r in reflected)
    assert not symmetric, "triangle must NOT be centrally symmetric (not a zonotope)"


# 4. Zonotope closure under linear maps and Minkowski sums ------------------
def test_zonotope_closure():
    rng = np.random.default_rng(2)
    G1 = rng.standard_normal((3, 2))              # generators of Z1
    G2 = rng.standard_normal((2, 2))             # generators of Z2
    M = rng.standard_normal((2, 2))
    # Minkowski sum of two zonotopes is the zonotope with stacked generators
    Gsum = np.vstack([G1, G2])
    # linear image of a zonotope has generators M g_i
    GM = G1 @ M.T
    for _ in range(20):
        d = rng.standard_normal(2)
        # sum closure
        assert wt.support_zonotope(Gsum, d) == pytest.approx(
            wt.support_zonotope(G1, d) + wt.support_zonotope(G2, d), abs=1e-9)
        # linear-image closure: h_{MZ}(d) = h_Z(M^T d)
        assert wt.support_zonotope(GM, d) == pytest.approx(
            wt.support_zonotope(G1, M.T @ d), abs=1e-9)


# 5. Inner-approximation containment: 2m-gon(kappa F) subset of ellipse -----
def test_inner_polygon_inside_ellipse_all_headings():
    F, kappa, m = 1.3, 0.4, 8
    verts = wt.inner_polygon(kappa * F, m)        # certificate set
    rng = np.random.default_rng(3)
    for _ in range(12):
        phi = rng.uniform(0, 2 * np.pi)           # arbitrary heading
        for v in verts:
            # v must lie inside the heading-aligned ellipse (F long, kappa F lat)
            vl = wt._rot(phi).T @ v
            val = (vl[0] / F) ** 2 + (vl[1] / (kappa * F)) ** 2
            assert val <= 1.0 + 1e-9


# 6. No false positive: certified wrench is realizable by the ellipse -------
def test_certificate_never_false_positive():
    rng = np.random.default_rng(4)
    N, kappa, m = 4, 0.4, 8
    F = rng.uniform(2.0, 4.0, size=N)
    # single slot per robot: contact points around a load, maps A_i in R^{3x2}
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    offs = 0.6 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    A = np.zeros((N, 3, 2))
    for i, (rx, ry) in enumerate(offs):
        A[i] = [[1, 0], [0, 1], [-ry, rx]]
    headings = np.arctan2(-offs[:, 1], -offs[:, 0])       # face the load
    n_pos = 0
    for _ in range(60):
        w = rng.standard_normal(3) * np.array([2.0, 2.0, 1.0])
        if wt.certify_membership_lp(F, A, w, kappa=kappa, m_sides=m):
            n_pos += 1
            assert wt.realizable_by_ellipse(F, headings, A, w, kappa=kappa), \
                "certificate certified a wrench the physical ellipse cannot realise"
    assert n_pos > 0, "test vacuous: no wrench was certified"
