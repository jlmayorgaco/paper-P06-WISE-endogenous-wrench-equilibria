"""What the WISE refinement costs relative to the hard connectivity constraint.

The quality comparison (HARD reaches V* but at an arbitrary point of the optimal face,
WISE at the margin-maximal one) is only half the story; a reviewer will ask what the
extra stage costs. This benchmarks, on the same instances and with identical solver
settings and no warm starts:

  * Stage 1  -- the productive QP  max V  s.t.  z in X_f;
  * HARD     -- max V  s.t.  lambda_2(Lbar(z)) >= sigma_req   (one SDP);
  * WISE-2   -- max lambda_2 over the productive fiber        (Stage-2 SDP);
  * Gamma_E  -- closed form ||Pi_{N_p} g_lambda|| where lambda_2 is simple;
  * Gamma_E  -- the general eigenspace SDP.

Only N and the resulting PSD cone change across the grid; T, M, H and |R| are fixed, so
this measures the cost of the spectral refinement, NOT general MRTA scaling.

Writes generated/scalability_hard_wise.csv and generated/scalability_hard_wise.json.

    python experiments/exp_scalability_hard_wise.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from wise_mr import nullspace as ns  # noqa: E402
from wise_mr import scenarios  # noqa: E402
from wise_mr.endogenous_graph import complement_basis  # noqa: E402

GEN = ROOT / "generated"
GEN.mkdir(exist_ok=True)

GRID_N = [12, 24, 48, 96]
SEEDS = [3, 5, 7]
SOLVER_KW = dict(solver="SCS", eps=1e-6, max_iters=20000, warm_start=False)


def _timed(fn):
    t0 = time.perf_counter()
    try:
        val = fn()
        ok = True
    except Exception as exc:                                   # noqa: BLE001
        val, ok = f"{type(exc).__name__}", False
    return (time.perf_counter() - t0) * 1e3, val, ok           # ms


def stage1(prob):
    import cvxpy as cp
    n = prob.N * prob.A
    A, B = ns.mass_matrix(prob), ns.served_matrix(prob)
    Hw, d, v = prob.wrench_matrix(), prob.demand().ravel(), prob._value()
    z = cp.Variable(n, nonneg=True)
    y = B @ z
    obj = cp.Maximize(cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y)))
    cp.Problem(obj, [A @ z == np.ones(prob.N), Hw @ z >= d]).solve(**SOLVER_KW)
    return float(obj.value)


def _spectral_problem(prob, fiber_eq=None, sigma=None, maximize_lambda=False):
    """One SDP: either max V s.t. lambda_2 >= sigma (HARD) or max lambda_2 on the fiber."""
    import cvxpy as cp
    n = prob.N * prob.A
    A, B = ns.mass_matrix(prob), ns.served_matrix(prob)
    Hw, d, v = prob.wrench_matrix(), prob.demand().ravel(), prob._value()
    Q = complement_basis(prob.N)
    z = cp.Variable(n, nonneg=True)
    t = cp.Variable()
    L = prob.base_laplacian + sum(
        z[i * prob.A + prob.relay_index] * prob.relay_laplacians[i]
        for i in range(prob.N))
    cons = [A @ z == np.ones(prob.N), Hw @ z >= d, Q.T @ L @ Q >> t * np.eye(prob.N - 1)]
    if fiber_eq is not None:
        cons.append(B @ z == fiber_eq)
    y = B @ z
    if maximize_lambda:
        obj = cp.Maximize(t)
    else:
        cons.append(t >= sigma)
        obj = cp.Maximize(cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y)))
    cp.Problem(obj, cons).solve(**SOLVER_KW)
    return float(t.value) if maximize_lambda else float(obj.value)


def gamma_closed_form(prob, zbar):
    """||Pi_{N_p} g_lambda||_2 with g_lambda[ir] = v^T L_ir v (simple lambda_2)."""
    info = ns.fiber_dimension(prob, zbar)
    Np = np.asarray(info["Np_basis"], float)
    L = prob.laplacian(zbar.reshape(prob.N, prob.A))
    Q = complement_basis(prob.N)
    w, Vt = np.linalg.eigh(Q.T @ L @ Q)
    v = Q @ Vt[:, 0]
    g = np.zeros(prob.N * prob.A)
    for i in range(prob.N):
        g[i * prob.A + prob.relay_index] = float(v @ prob.relay_laplacians[i] @ v)
    return float(np.linalg.norm(Np @ g)) if Np.size else 0.0


def gamma_sdp(prob, zbar):
    import cvxpy as cp
    info = ns.fiber_dimension(prob, zbar)
    Np = np.asarray(info["Np_basis"], float)
    if Np.size == 0:
        return 0.0
    L = prob.laplacian(zbar.reshape(prob.N, prob.A))
    Q = complement_basis(prob.N)
    w, Vt = np.linalg.eigh(Q.T @ L @ Q)
    mask = np.abs(w - w[0]) <= 1e-7 * max(1.0, abs(w[0]))
    U2 = Q @ Vt[:, mask]
    a = cp.Variable(Np.shape[0])
    d = Np.T @ a
    t = cp.Variable()
    DL = sum(d[i * prob.A + prob.relay_index] * prob.relay_laplacians[i]
             for i in range(prob.N))
    cp.Problem(cp.Maximize(t),
               [cp.norm(d, 2) <= 1,
                U2.T @ DL @ U2 >> t * np.eye(U2.shape[1])]).solve(**SOLVER_KW)
    return float(t.value)


def main() -> dict:
    from exp_gamma import optimal_fiber_base

    rows = []
    for N in GRID_N:
        for seed in SEEDS:
            prob = scenarios.two_region(seed=seed, N=N)
            rec = {"N": N, "seed": seed, "n_vars": prob.N * prob.A,
                   "psd_cone_dim": prob.N - 1, "n_support_rows": prob.M * prob.P}
            ms, val, ok = _timed(lambda p=prob: stage1(p))
            rec["stage1_ms"], rec["stage1_ok"] = ms, ok
            ms, val, ok = _timed(lambda p=prob: _spectral_problem(p, sigma=p.sigma))
            rec["hard_ms"], rec["hard_ok"] = ms, ok
            try:
                zbar, y_star, _, _ = optimal_fiber_base(prob)
            except Exception:                                   # noqa: BLE001
                zbar, y_star = None, None
            if zbar is not None and np.all(np.isfinite(zbar)):
                ms, val, ok = _timed(
                    lambda p=prob, ys=y_star: _spectral_problem(p, fiber_eq=ys,
                                                                maximize_lambda=True))
                rec["wise_stage2_ms"], rec["wise_stage2_ok"] = ms, ok
                ms, val, ok = _timed(lambda p=prob, z=zbar: gamma_closed_form(p, z))
                rec["gamma_closed_ms"], rec["gamma_closed_ok"] = ms, ok
                ms, val, ok = _timed(lambda p=prob, z=zbar: gamma_sdp(p, z))
                rec["gamma_sdp_ms"], rec["gamma_sdp_ok"] = ms, ok
            else:
                for k in ("wise_stage2", "gamma_closed", "gamma_sdp"):
                    rec[f"{k}_ms"], rec[f"{k}_ok"] = float("nan"), False
            rows.append(rec)
            print(f"N={N:3d} seed={seed:2d}  stage1={rec['stage1_ms']:8.1f}  "
                  f"HARD={rec['hard_ms']:8.1f}  WISE2={rec['wise_stage2_ms']:8.1f}  "
                  f"Gamma_cf={rec['gamma_closed_ms']:7.2f}  "
                  f"Gamma_sdp={rec['gamma_sdp_ms']:8.1f}  [ms]", flush=True)

    with (GEN / "scalability_hard_wise.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def _stats(N, key):
        v = np.array([r[f"{key}_ms"] for r in rows if r["N"] == N], float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"median_ms": None, "iqr_ms": None, "n_ok": 0}
        return {"median_ms": float(np.median(v)),
                "iqr_ms": float(np.percentile(v, 75) - np.percentile(v, 25)),
                "n_ok": int(sum(r[f"{key}_ok"] for r in rows if r["N"] == N))}

    summary = {
        "protocol": ("identical solver settings (SCS, eps=1e-6, max_iters=20000), no "
                     "warm starts for any method; T, M, H and |R| fixed, only N and the "
                     "resulting PSD cone vary -- this measures the cost of the spectral "
                     "refinement, not general MRTA scaling"),
        "solver_settings": {k: str(v) for k, v in SOLVER_KW.items()},
        "grid_N": GRID_N, "seeds": SEEDS,
        "by_N": {str(N): {k: _stats(N, k) for k in
                          ("stage1", "hard", "wise_stage2", "gamma_closed", "gamma_sdp")}
                 for N in GRID_N},
        "n_failures": int(sum(1 for r in rows for k in
                              ("stage1", "hard", "wise_stage2", "gamma_sdp")
                              if not r.get(f"{k}_ok", False))),
    }
    (GEN / "scalability_hard_wise.json").write_text(json.dumps(summary, indent=2),
                                                    encoding="utf-8")
    print(json.dumps(summary["by_N"], indent=1))
    return summary


if __name__ == "__main__":
    main()
