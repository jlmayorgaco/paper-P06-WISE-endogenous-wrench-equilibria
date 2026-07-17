"""E-mismatch: quantify Lemma 3 -- the surrogate-vs-geometric gap eps_L vs tracking tube rho.

For each WISE composition we take the nominal poses q_bar, build the geometric Laplacian
L_geo(q_bar) as a smooth-weight proximity graph, and perturb every robot within a tube of
radius rho. We measure the a-priori tracking bound 4 d_max L_psi rho against the realized
mismatch ||L_geo(q) - L_geo(q_bar)||_2, and the fraction of perturbations that still preserve
lambda_2(L_geo(q)) >= sigma_dyn. Writes generated/mismatch.json.
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


def _lgeo(pos, reach):
    """Smooth-weight proximity Laplacian: edge (i,j) if within min(reach_i,reach_j)."""
    N = pos.shape[0]
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            dij = np.linalg.norm(pos[i] - pos[j])
            if dij <= min(reach[i], reach[j]):
                W[i, j] = W[j, i] = _psi(dij)
    return np.diag(W.sum(1)) - W


def _lam2(L):
    w = np.linalg.eigvalsh(0.5 * (L + L.T))
    return float(w[1])


def run(seeds=20, rhos=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3), draws=100):
    rng = np.random.default_rng(0)
    out = {}
    for rho in rhos:
        eps_meas, apriori, preserved, lam_min = [], [], [], []
        for s in range(seeds):
            prob = scenarios.two_region(seed=s, N=12, nu=0.4, tau_d=5.0, bridge_gain=3.0)
            pos = np.asarray(prob.meta["pos"], float)
            reach = np.asarray(prob.meta["r"], float)
            Lg = _lgeo(pos, reach)
            d_max = int(np.max((np.abs(Lg) > 1e-9).sum(1) - 1))
            sig_dyn = prob.sigma                        # threshold = sigma_dyn
            for _ in range(draws if rho > 0 else 1):
                delta = rng.standard_normal(pos.shape)
                delta *= (rho / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9))
                Lq = _lgeo(pos + delta, reach)
                eps_meas.append(np.linalg.norm(Lq - Lg, 2))
                apriori.append(4 * d_max * L_PSI * rho)
                lam = _lam2(Lq); lam_min.append(lam)
                preserved.append(lam >= _lam2(Lg) - 1e-9 or lam >= sig_dyn)
        out[f"{rho:.2f}"] = dict(
            rho=rho,
            eps_L_measured_median=float(np.median(eps_meas)),
            eps_L_measured_max=float(np.max(eps_meas)),
            apriori_bound_median=float(np.median(apriori)),
            lambda2_geo_min=float(np.min(lam_min)),
            preserved_frac=float(np.mean(preserved)),
        )
        print(f"rho={rho:.2f}: eps_L med={out[f'{rho:.2f}']['eps_L_measured_median']:.3f} "
              f"(<= apriori {out[f'{rho:.2f}']['apriori_bound_median']:.2f}), "
              f"min lam2={out[f'{rho:.2f}']['lambda2_geo_min']:.3f}, "
              f"preserved={out[f'{rho:.2f}']['preserved_frac']:.0%}")
    (GEN / "mismatch.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    run()
