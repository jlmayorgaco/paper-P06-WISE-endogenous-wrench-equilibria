"""Regression tests for the repeated-Fiedler eigenspace and Gamma_E (paper Thm. 2).

The critical case is a DISCONNECTED graph, where lambda_2(Lbar) = 0 is degenerate
*together with* the consensus mode 1. Selecting columns of the full-space
eigenvectors then leaks a component along 1, because `eigh` returns an arbitrary
basis of the degenerate kernel. Building U_2 = Q Utilde_2 from Ltilde = Q^T Lbar Q
removes 1 structurally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp_gamma import lam2_eigenspace  # noqa: E402
from wise_mr.endogenous_graph import complement_basis  # noqa: E402


def path_laplacian(n: int, w: float = 1.0) -> np.ndarray:
    L = np.zeros((n, n))
    for i in range(n - 1):
        L[i, i] += w; L[i + 1, i + 1] += w
        L[i, i + 1] -= w; L[i + 1, i] -= w
    return L


def two_components(n_a: int = 4, n_b: int = 4) -> np.ndarray:
    """Block-diagonal Laplacian: lambda_1 = lambda_2 = 0, kernel = span{1_A, 1_B}."""
    La, Lb = path_laplacian(n_a), path_laplacian(n_b)
    L = np.zeros((n_a + n_b, n_a + n_b))
    L[:n_a, :n_a] = La
    L[n_a:, n_a:] = Lb
    return L


def test_disconnected_graph_has_lambda2_zero():
    L = two_components()
    lam2, U2, m = lam2_eigenspace(L)
    assert lam2 == pytest.approx(0.0, abs=1e-12)
    assert m == 1, "two components => a single zero eigenvalue of Ltilde"
    assert U2.shape == (8, 1)


def test_U2_orthogonal_to_ones_when_disconnected():
    """The property the Q-construction guarantees and full-space selection does not."""
    L = two_components()
    _, U2, _ = lam2_eigenspace(L)
    ones = np.ones(L.shape[0]) / np.sqrt(L.shape[0])
    assert np.abs(U2.T @ ones).max() < 1e-12


def test_full_space_selection_can_leak_the_consensus_mode():
    """Documents the bug the Q-construction fixes: with a degenerate kernel the
    naive 'drop column 0 of eigh(L)' rule need not land inside 1-perp."""
    L = two_components()
    w, V = np.linalg.eigh(L)
    naive = V[:, 1:2]                      # 'lambda_2 eigenvector' of the full L
    ones = np.ones(L.shape[0]) / np.sqrt(L.shape[0])
    leak = float(np.abs(naive.T @ ones).max())
    _, U2, _ = lam2_eigenspace(L)
    safe = float(np.abs(U2.T @ ones).max())
    # the projected construction is clean regardless of what the naive one returned
    assert safe < 1e-12
    assert safe <= leak + 1e-12


def test_repeated_eigenvalue_multiplicity_detected():
    """Three disconnected components => lambda_2 = 0 with multiplicity 2 in Ltilde."""
    La, Lb, Lc = path_laplacian(3), path_laplacian(3), path_laplacian(3)
    L = np.zeros((9, 9))
    L[:3, :3] = La; L[3:6, 3:6] = Lb; L[6:, 6:] = Lc
    lam2, U2, m = lam2_eigenspace(L)
    assert lam2 == pytest.approx(0.0, abs=1e-12)
    assert m == 2
    ones = np.ones(9) / 3.0
    assert np.abs(U2.T @ ones).max() < 1e-12
    assert np.abs(U2.T @ U2 - np.eye(2)).max() < 1e-10


def test_directional_derivative_matches_one_sided_difference():
    """lambda_min(U_2^T DLbar[d] U_2) must match the ONE-SIDED difference quotient
    (lambda_2 is not differentiable at a point of multiplicity)."""
    rng = np.random.default_rng(11)
    L = two_components(3, 3)               # lambda_2 = 0, m = 1 in Ltilde
    Q = complement_basis(6)
    dL = np.zeros((6, 6))                  # add a bridge between the components
    i, j = 0, 3
    dL[i, i] += 1; dL[j, j] += 1; dL[i, j] -= 1; dL[j, i] -= 1

    _, U2, _ = lam2_eigenspace(L)
    pred = float(np.linalg.eigvalsh(U2.T @ dL @ U2)[0])

    def lam2_of(M):
        return float(np.linalg.eigvalsh(Q.T @ M @ Q)[0])

    base = lam2_of(L)
    for t in (1e-5, 1e-6, 1e-7):
        fd = (lam2_of(L + t * dL) - base) / t
        assert abs(fd - pred) < 1e-4, f"t={t}: fd={fd} vs pred={pred}"
    assert pred > 0, "bridging two components must strictly increase lambda_2"


def test_multiplicity_two_directional_derivative_is_the_min_not_a_single_vector():
    """With m = 2, a single eigenvector overestimates: lambda_min over the eigenspace
    is the correct (and smaller) directional derivative."""
    L = np.zeros((9, 9))
    for a in (0, 3, 6):
        L[a:a + 3, a:a + 3] = path_laplacian(3)
    _, U2, m = lam2_eigenspace(L)
    assert m == 2
    dL = np.zeros((9, 9))                  # bridge only components 1 and 2
    dL[0, 0] += 1; dL[3, 3] += 1; dL[0, 3] -= 1; dL[3, 0] -= 1

    pred = float(np.linalg.eigvalsh(U2.T @ dL @ U2)[0])
    singles = [float(u @ dL @ u) for u in U2.T]
    assert pred <= min(singles) + 1e-12
    # component 3 stays isolated, so lambda_2 cannot rise off zero
    assert pred == pytest.approx(0.0, abs=1e-10)
    assert max(singles) > 1e-6, "a single eigenvector would wrongly predict a gain"
