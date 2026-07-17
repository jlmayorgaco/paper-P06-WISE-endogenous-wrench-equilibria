"""E-mismatch: a-priori geometric transfer by graph monotonicity (Lemma 3).

We build the surrogate as a LOWER-BOUND Laplacian Lbar_lb: every edge weight is the infimum
of the smooth decreasing weight psi over the tracking tube, i.e. psi evaluated at the nominal
distance inflated by 2*rho (the worst case when both endpoints move by rho). For ANY pose q
with each robot inside its tube, w_ij(q) = psi(||q_i-q_j||) >= psi(||qbar_i-qbar_j||+2rho),
so L_geo(q) - Lbar_lb is a nonnegative combination of edge Laplacians:

    L_geo(q) >= Lbar_lb   (Loewner)   ==>   lambda_2(L_geo(q)) >= lambda_2(Lbar_lb).

This is an a-priori guarantee -- no measured mismatch, no post-hoc term. We verify it over
many perturbations: the PSD gap min-eig(Q'(L_geo - Lbar_lb)Q) and the spectral margin
lambda_2(L_geo) - lambda_2(Lbar_lb) must both stay >= 0. Writes generated/mismatch.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wise_mr import scenarios  # noqa: E402

GEN = ROOT / "generated"
GEN.mkdir(exist_ok=True)

L_PSI = 0.5            # Lipschitz constant of the smooth weight psi(r)=exp(-r/2)/2 (|psi'|<=L_psi)


def _psi(r):
    return 0.5 * np.exp(-r / 2.0)          # smooth edge weight; |psi'| <= 0.25 <= L_PSI


def _lgeo(pos, extra=0.0):
    """Smooth-weight Laplacian. `extra` >= 0 inflates every pairwise distance, and since psi
    is decreasing this yields a guaranteed LOWER-BOUND weight (extra=2*rho over the tube)."""
    N = pos.shape[0]
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            W[i, j] = W[j, i] = _psi(np.linalg.norm(pos[i] - pos[j]) + extra)
    return np.diag(W.sum(1)) - W


def _lam2(L):
    w = np.linalg.eigvalsh(0.5 * (L + L.T))
    return float(w[1])


def _Q(N):
    """Orthonormal basis of 1^perp."""
    M = np.eye(N) - np.ones((N, N)) / N
    w, V = np.linalg.eigh(M)
    return V[:, w > 0.5]                                 # the N-1 nonzero eigenvectors


def run(seeds=20, rhos=(0.02, 0.05, 0.1, 0.2, 0.3), draws=100):
    rng = np.random.default_rng(0)
    out = {}
    all_margin, all_psd = [], []
    for rho in rhos:
        margins, psd_gaps, lam_lb_all = [], [], []
        for s in range(seeds):
            prob = scenarios.two_region(seed=s, N=12, nu=0.4, tau_d=5.0, bridge_gain=3.0)
            pos = np.asarray(prob.meta["pos"], float)
            N = pos.shape[0]; Q = _Q(N)
            Lb = _lgeo(pos, extra=2.0 * rho)             # lower-bound Laplacian over the tube
            lam_lb = _lam2(Lb); lam_lb_all.append(lam_lb)
            for _ in range(draws):
                delta = rng.standard_normal(pos.shape)
                delta *= (rho / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9))
                Lq = _lgeo(pos + delta, extra=0.0)       # true geometric graph within the tube
                margins.append(_lam2(Lq) - lam_lb)       # spectral transfer margin (>= 0 a priori)
                D = Q.T @ (Lq - Lb) @ Q
                psd_gaps.append(float(np.linalg.eigvalsh(0.5 * (D + D.T))[0]))  # min eig of gap
        margins = np.array(margins); psd_gaps = np.array(psd_gaps)
        all_margin.append(margins); all_psd.append(psd_gaps)
        out[f"{rho:.2f}"] = dict(
            rho=rho, draws=len(margins),
            spectral_margin_min=float(margins.min()),
            spectral_margin_median=float(np.median(margins)),
            spectral_margin_nonneg_frac=float(np.mean(margins >= -1e-9)),
            psd_gap_min=float(psd_gaps.min()),
            psd_nonneg_frac=float(np.mean(psd_gaps >= -1e-9)),
            lambda2_lb_min=float(np.min(lam_lb_all)),
        )
        print(f"rho={rho:.2f}: lam2(Lgeo)-lam2(Lb) min={margins.min():.4f} "
              f"med={np.median(margins):.3f} nonneg={np.mean(margins>=-1e-9):.0%}; "
              f"PSD-gap min={psd_gaps.min():.2e} nonneg={np.mean(psd_gaps>=-1e-9):.0%}")
    M = np.concatenate(all_margin); P = np.concatenate(all_psd)
    print(f"\nTOTAL {len(M)} perturbations: spectral margin >=0 in {np.mean(M>=-1e-9):.1%}, "
          f"min={M.min():.4f}; Loewner L_geo>=Lbar_lb in {np.mean(P>=-1e-9):.1%}")
    (GEN / "mismatch.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    run()
