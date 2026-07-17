"""Integer recovery from the fluid WISE optimum, re-certified numerically (Rem. integer).

No O(1/N) claim. For each of 30 seeds we compute the exact relaxed productive optimum
V_relax (QP over X_f), round the fluid WISE solution to integers, and report:
  * fluid_certified_rate       -- fraction of fluid solutions that are WISE-certified;
  * argmax_rate                -- certified rate of one-hot argmax rounding;
  * rr_single_rate             -- certified rate of a SINGLE randomized-rounding draw;
  * rr_bestof_rate (n_draws)   -- certified rate of best-of-n_draws randomized rounding;
  * welfare_gap_rel            -- (V_relax - V_integer)/|V_relax| >= 0 (a true gap sign).
Writes generated/integer_recovery.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import baselines, metrics, nullspace as ns, scenarios  # noqa: E402

GEN = ROOT / "generated"
GEN.mkdir(exist_ok=True)

N_DRAWS = 30
SEEDS = list(range(30))


def _relaxed_optimum_value(prob):
    """Exact relaxed productive optimum V_relax = max_{z in X_f} V(z) (QP)."""
    import cvxpy as cp
    n = prob.N * prob.A
    A = ns.mass_matrix(prob); B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix(); d = prob.demand().ravel()
    v = prob._value()
    z = cp.Variable(n, nonneg=True); y = B @ z
    cp.Problem(cp.Maximize(cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))),
               [A @ z == np.ones(prob.N), Hw @ z >= d]).solve()
    return float(prob.productive_value(np.asarray(z.value, float).reshape(prob.N, prob.A)))


def _prop4_margins(prob, x_fluid, x_round):
    """Proposition-4 margins for a rounding. Two radii are compared: the coarse
    r_coarse = min(m_w/kappa_w, m_lambda/kappa_L) and the tighter PER-ROW wrench radius
    r_w* = min_j (H_w z* - d)_j / ||(H_w)_j||_2 (always >= m_w/kappa_w). chi = ||zhat-z*||/r.
    chi < 1 => Proposition 4 certifies the rounding without re-evaluation."""
    zc = np.asarray(x_fluid, float).ravel()
    zh = np.asarray(x_round, float).ravel()
    Hw = prob.wrench_matrix(); d = prob.demand().ravel()
    slack = Hw @ zc - d
    rownorm = np.linalg.norm(Hw, axis=1)
    m_w = float(np.min(slack))                             # wrench slack at z*
    m_lam = float(prob.lambda2(x_fluid) - prob.sigma)      # connectivity slack at z*
    kappa_w = float(np.max(rownorm))                       # ||H_w||_{2->inf}
    R = np.asarray(prob.relay_laplacians, float)
    kappa_L = float(np.sqrt(np.sum([np.linalg.norm(R[i], 2) ** 2 for i in range(prob.N)])))
    # per-row wrench radius (tighter): r_w* = min_j slack_j / ||row_j||  >=  m_w/kappa_w
    pos = rownorm > 1e-12
    r_w_row = float(np.min(slack[pos] / rownorm[pos])) if np.any(pos) else 0.0
    r_coarse = min(m_w / kappa_w, m_lam / kappa_L) if (m_w > 0 and m_lam > 0) else 0.0
    r_tight = min(r_w_row, m_lam / kappa_L) if (r_w_row > 0 and m_lam > 0) else 0.0
    dist = float(np.linalg.norm(zh - zc))
    return dict(m_w=m_w, m_lambda=m_lam, kappa_w=kappa_w, kappa_L=kappa_L,
                rstar=r_coarse, rstar_tight=r_tight, dist=dist,
                chi=dist / r_coarse if r_coarse > 0 else np.inf,
                chi_tight=dist / r_tight if r_tight > 0 else np.inf,
                prop4_certified=bool(r_tight > 0 and dist < r_tight))


def _rr_rates(prob, x_fluid, seed):
    """Single-draw and best-of-N certified rates for randomized rounding."""
    rng = np.random.default_rng(seed + 5000)
    x = np.asarray(x_fluid, float)
    hits, best = 0, False
    x_cert = None
    for _ in range(N_DRAWS):
        cand = np.zeros_like(x)
        for i in range(prob.N):
            p = x[i] / x[i].sum()
            cand[i, rng.choice(prob.A, p=p)] = 1.0
        ok = metrics.certified(prob, cand)
        hits += int(ok); best = best or ok
        if ok and x_cert is None:
            x_cert = cand
    return hits / N_DRAWS, float(best), x_cert


def main() -> None:
    fluid, arg, rr_single, rr_best, gaps = [], [], [], [], []
    m_w_all, m_lam_all, rstar_all, chi_all, p4_cert, direct_cert = [], [], [], [], 0, 0
    chi_tight_all, rstar_tight_all = [], []
    for sd in SEEDS:
        prob = scenarios.two_region(seed=sd, N=12, nu=0.5, tau_d=3.0, bridge_gain=3.0)
        res = baselines.wise_primal_dual(prob, max_iters=4000)
        fluid.append(metrics.certified(prob, res.x))
        xa = metrics.round_argmax(prob, res.x)
        arg.append(metrics.certified(prob, xa))
        s_rate, b_rate, x_cert = _rr_rates(prob, res.x, sd)
        rr_single.append(s_rate); rr_best.append(b_rate)
        v_relax = _relaxed_optimum_value(prob)
        v_int = prob.productive_value(xa)
        gaps.append((v_relax - v_int) / (abs(v_relax) + 1e-9))
        # Proposition 4: margins for a certified rounding (best-of-N draw, else argmax)
        zh = x_cert if x_cert is not None else xa
        mg = _prop4_margins(prob, res.x, zh)
        m_w_all.append(mg["m_w"]); m_lam_all.append(mg["m_lambda"])
        rstar_all.append(mg["rstar"]); chi_all.append(mg["chi"])
        chi_tight_all.append(mg["chi_tight"]); rstar_tight_all.append(mg["rstar_tight"])
        p4_cert += int(mg["prop4_certified"])
        direct_cert += int(metrics.certified(prob, zh))
    cert = {
        "n_seeds": len(SEEDS), "N_robots": 12, "n_draws": N_DRAWS,
        "note": "N=12 physical demonstration; each robot is integer (one action).",
        "fluid_certified_rate": round(float(np.mean(fluid)), 4),
        "argmax_rate": round(float(np.mean(arg)), 4),
        "rr_single_rate": round(float(np.mean(rr_single)), 4),
        "rr_bestof_rate": round(float(np.mean(rr_best)), 4),
        "welfare_gap_rel_mean": round(float(np.mean(gaps)), 4),
        "welfare_gap_rel_max": round(float(np.max(gaps)), 4),
        # Proposition 4 diagnostics (best-of-N certified rounding per seed)
        "prop4_m_w_median": round(float(np.median(m_w_all)), 4),
        "prop4_m_lambda_median": round(float(np.median(m_lam_all)), 4),
        "prop4_rstar_median": round(float(np.median(rstar_all)), 4),
        "prop4_chi_median": round(float(np.median([c for c in chi_all if np.isfinite(c)])
                                     or [float("nan")]), 4),
        "prop4_rstar_tight_median": round(float(np.median(rstar_tight_all)), 4),
        "prop4_chi_tight_median": round(float(np.median([c for c in chi_tight_all
                                                         if np.isfinite(c)]) or [float("nan")]), 4),
        "prop4_certified_seeds": p4_cert,
        "direct_certified_seeds": direct_cert,
        "prop4_active_wrench_seeds": int(np.sum(np.array(m_w_all) <= 1e-6)),
    }
    (GEN / "integer_recovery.json").write_text(json.dumps(cert, indent=2), encoding="utf-8")
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
