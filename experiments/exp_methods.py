"""E-methods (paper Sec. V ablation): paired method comparison with Wilson CIs.

Compares, on the same N=12 instance per seed (paired), seven assignment methods:
  random_feasible   -- random simplex assignment (negative control)
  scalar_capacity   -- single scalar-capacity heuristic
  wrench_only       -- enforce wrench feasibility only
  connectivity_only -- enforce connectivity only
  wise_pd           -- decentralized WISE primal-dual flow (this paper)
  wise_sdp          -- exact WISE selection SDP (Thm. 2)
  centralized       -- centralized multistart heuristic (strong reference, not an oracle)
Metrics per seed: wrench residual, info margin lambda_2 - sigma_req, and certified
success (wrench-feasible AND lambda_2 >= sigma_req). Reports the mean with a Wilson 95%
CI on certified success. Writes generated/methods_comparison.csv and regenerates
paper/figures/ablation_table.tex.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import baselines, nullspace as ns, scenarios, wise_sdp  # noqa: E402
from wise_mr.endogenous_graph import complement_basis  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

METHODS = ["random_feasible", "scalar_capacity", "wrench_only",
           "connectivity_only", "wise_pd", "wise_sdp", "centralized"]


def _random_feasible(prob, rng):
    x = rng.random((prob.N, prob.A))
    return x / x.sum(axis=1, keepdims=True)


def _wise_sdp_assignment(prob):
    """Two-stage: y* via QP, then the selection SDP; returns the assignment x."""
    import cvxpy as cp

    n = prob.N * prob.A
    A = ns.mass_matrix(prob); b = np.ones(prob.N)
    B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix(); d = prob.demand().ravel()
    v = prob._value()
    x = cp.Variable(n, nonneg=True)
    y = B @ x
    p = cp.Problem(cp.Maximize(cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))),
                   [A @ x == b, Hw @ x >= d])
    p.solve()
    if x.value is None:
        return None
    y_star = B @ x.value
    A_eq = np.vstack([A, B]); b_eq = np.concatenate([b, y_star])
    G = np.vstack([-np.eye(n), -Hw]); h = np.concatenate([np.zeros(n), -d])
    ridx = prob.relay_index
    terms = [(i * prob.A + ridx, prob.relay_laplacians[i]) for i in range(prob.N)]
    res = wise_sdp.solve_wise_sdp(n, prob.base_laplacian, terms, complement_basis(prob.N),
                                  prob.sigma, A_eq=A_eq, b_eq=b_eq, G_ineq=G, h_ineq=h)
    if not np.all(np.isfinite(res.z)):
        return None
    return np.maximum(res.z.reshape(prob.N, prob.A), 0.0)   # clip solver noise


def _assignment(method, prob, rng):
    if method == "random_feasible":
        return _random_feasible(prob, rng)
    if method == "wise_sdp":
        return _wise_sdp_assignment(prob)
    if method == "wise_pd":
        return baselines.run("wise_primal_dual", prob, max_iters=3000).x
    if method == "centralized":
        return baselines.centralized_wise(prob).x
    return baselines.run(method, prob).x


def _metrics(prob, x, v_star):
    if x is None:
        return dict(wrench_res=np.nan, info_margin=np.nan, cert=0.0, prod_gap=np.nan)
    x = np.asarray(x, dtype=float)
    # relative wrench deficit: max_l [d - s]_+ / max_l d  (dimensionless, vs demand)
    s = prob.capacity(x)
    d = prob.demand()
    deficit = float(np.max(np.maximum(d - s, 0.0)))
    dref = float(np.max(d)) if float(np.max(d)) > 0 else 1.0
    wres = deficit / dref
    margin = float(prob.lambda2(x) - prob.sigma)
    feas = wres <= 1e-3 and x.min() >= -1e-6
    cert = 1.0 if (feas and margin >= 0.0) else 0.0
    return dict(wrench_res=wres, info_margin=margin, cert=cert,
                prod_gap=float(v_star - prob.productive_value(x)))


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score 95% CI for a binomial proportion (k successes of n)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def run(seeds=30):
    rng0 = np.random.default_rng(12345)
    data = {m: {k: [] for k in ("wrench_res", "info_margin", "cert", "prod_gap")}
            for m in METHODS}
    for s in range(seeds):
        prob = scenarios.two_region(seed=s, N=12, nu=0.4, tau_d=6.5, bridge_gain=3.0)
        # productive optimum value for the gap metric (via the SDP stage-1 QP proxy)
        central_x = baselines.centralized_wise(prob).x
        v_star = prob.productive_value(central_x)
        for m in METHODS:
            met = _metrics(prob, _assignment(m, prob, rng0), v_star)
            for k in data[m]:
                data[m][k].append(met[k])

    rows = []
    for m in METHODS:
        row = {"method": m}
        # continuous metrics: report the mean
        for k in ("wrench_res", "info_margin"):
            vals = [v for v in data[m][k] if np.isfinite(v)]
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        # certified success is binomial: Wilson score 95% CI (not degenerate bootstrap)
        certs = data[m]["cert"]
        p, lo, hi = _wilson_ci(int(sum(certs)), len(certs))
        row["cert_mean"], row["cert_lo"], row["cert_hi"] = p, lo, hi
        rows.append(row)

    with open(GEN / "methods_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    _write_table(rows, seeds)

    for r in rows:
        print(f"{r['method']:>18}: wr={r['wrench_res_mean']:.2f}  "
              f"marg={r['info_margin_mean']:+.2f}  cert={r['cert_mean']:.0%} "
              f"[{r['cert_lo']:.0%},{r['cert_hi']:.0%}]")
    return rows


def _write_table(rows, seeds):
    disp = {"random_feasible": r"random ($\mathcal X$-feas.)",
            "scalar_capacity": r"scalar-capacity",
            "wrench_only": r"wrench-only", "connectivity_only": r"connectivity-only",
            "wise_pd": r"\textbf{WISE-PD}", "wise_sdp": r"\textbf{WISE-SDP}",
            "centralized": r"centralized multistart"}
    lines = [r"% Auto-generated by experiments/exp_methods.py -- do not edit by hand.",
             f"% Real simulation, 12 robots, nu=0.4, tau_d=6.5; {seeds} paired seeds; "
             r"Wilson 95% CI on cert.",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lccc}", r"\hline",
             r"Method & rel.\ wr.\ def.\ $\downarrow$ & "
             r"$\lambda_2\!-\!\sigma_{\mathrm{req}}$ $\uparrow$ "
             r"& Cert.\ succ.\ $\uparrow$\\", r"\hline"]
    for r in rows:
        cert = r"%.0f\%%\,\tiny[%.0f,%.0f]" % (100 * r["cert_mean"], 100 * r["cert_lo"],
                                               100 * r["cert_hi"])
        lines.append(r"%s & %.2f & %+.2f & %s \\" % (
            disp[r["method"]], r["wrench_res_mean"], r["info_margin_mean"], cert))
    lines += [r"\hline", r"\end{tabular}"]
    (FIG / "ablation_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
