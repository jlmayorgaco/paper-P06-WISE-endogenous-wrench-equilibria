"""E-multiplicity: at a repeated Fiedler eigenvalue, the single-vector gradient is wrong
but the eigenspace criterion (Prop. 2) is correct.

We build a C3-symmetric three-cluster graph, where lambda_2 = lambda_3 (a two-fold
degenerate algebraic connectivity). For a candidate relay direction d (strengthen one
inter-cluster bridge), three predictions of the first-order change in lambda_2 are compared:

  * single-vector gradient  g_v = v^T (dL) v   for an ARBITRARY unit Fiedler vector v in the
    degenerate eigenspace (what a naive nabla lambda_2 uses);
  * eigenspace criterion    g_U = lambda_min(U2^T (dL) U2)   (Prop. 2, the correct one);
  * ground truth            (lambda_2(L + eps dL) - lambda_2(L)) / eps.

Because lambda_2 is the MIN over the eigenspace, a single vector that happens to align with
the strengthened bridge predicts a large gain, while the true lambda_2 barely moves (the
orthogonal mode does not improve). The eigenspace criterion predicts the truth. We report,
over many random directions and random in-eigenspace vectors, the fraction of directions on
which the single-vector prediction has the wrong sign relative to ground truth, versus the
eigenspace criterion. Writes generated/multiplicity.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GEN = ROOT / "generated"
GEN.mkdir(exist_ok=True)


def _c3_graph(k=3, w_in=1.0, w_bridge=0.35):
    """Three k-cliques A,B,C with equal A-B,B-C,C-A bridges (C3 symmetry => lambda2=lambda3)."""
    n = 3 * k
    A = np.zeros((n, n))
    for c in range(3):
        idx = range(c * k, c * k + k)
        for i in idx:
            for j in idx:
                if i != j:
                    A[i, j] = w_in
    # one bridge edge between consecutive clusters (symmetric)
    reps = [c * k for c in range(3)]
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        A[reps[a], reps[b]] = A[reps[b], reps[a]] = w_bridge
    return np.diag(A.sum(1)) - A, reps, k


def _lam2(L):
    w = np.linalg.eigvalsh(0.5 * (L + L.T))
    return float(w[1])


def _bridge_dL(reps, n, pair):
    """Direction that strengthens one inter-cluster bridge (a relay activation)."""
    a, b = reps[pair[0]], reps[pair[1]]
    dA = np.zeros((n, n)); dA[a, b] = dA[b, a] = 1.0
    return np.diag(dA.sum(1)) - dA


def run(trials=200, eps=1e-4, seed=0):
    L, reps, k = _c3_graph()
    n = L.shape[0]
    w, V = np.linalg.eigh(0.5 * (L + L.T))
    gap23 = float(w[2] - w[1])
    # degenerate Fiedler eigenspace: columns 1,2 (0-indexed) if lambda2==lambda3
    U2 = V[:, 1:3]
    rng = np.random.default_rng(seed)

    wrong_single = wrong_U = 0
    rel_single, rel_U = [], []
    for _ in range(trials):
        pair = tuple(rng.choice(3, size=2, replace=False))
        dL = _bridge_dL(reps, n, pair)
        truth = (_lam2(L + eps * dL) - _lam2(L)) / eps
        # single arbitrary unit vector in the 2-D eigenspace
        c = rng.standard_normal(2); c /= np.linalg.norm(c)
        v = U2 @ c
        g_v = float(v @ dL @ v)
        # eigenspace criterion = lambda_min(U2^T dL U2)
        g_U = float(np.linalg.eigvalsh(U2.T @ dL @ U2)[0])
        if np.sign(g_v) != np.sign(truth) and abs(truth) > 1e-9:
            wrong_single += 1
        if np.sign(g_U) != np.sign(truth) and abs(truth) > 1e-9:
            wrong_U += 1
        rel_single.append(abs(g_v - truth) / (abs(truth) + 1e-9))
        rel_U.append(abs(g_U - truth) / (abs(truth) + 1e-9))

    res = dict(
        clusters=3, clique_size=k, lambda2=float(w[1]), lambda3=float(w[2]),
        gap_lambda3_lambda2=gap23, degenerate=bool(gap23 < 1e-9),
        trials=trials,
        single_vector_wrong_sign=wrong_single,
        eigenspace_wrong_sign=wrong_U,
        single_vector_median_relerr=float(np.median(rel_single)),
        eigenspace_median_relerr=float(np.median(rel_U)),
    )
    (GEN / "multiplicity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run()
