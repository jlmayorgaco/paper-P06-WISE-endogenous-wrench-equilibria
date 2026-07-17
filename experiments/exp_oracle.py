"""E-oracle (Reviewer-2 P0): exact integer benchmark for small teams.

For small ``N`` we enumerate *every* pure assignment (each robot picks exactly one of
the ``A = M*H+2`` actions -- a load slot, relay, or idle) and compute, exactly:

  * the integer productive optimum  V_int^prod  = max_{integer, wrench-feasible} V(x);
  * the integer WISE optimum        V_int^wise  = max_{integer, wrench-feasible,
                                                    lambda_2(Lbar) >= sigma_req} V(x);
  * whether an integer WISE exists at all (integer-feasible).

Against the fluid relaxation (Stage-1 productive optimum V* and Stage-2 spectral
capacity Lambda_E of the WISE SDP) this yields the quantities the reviewer asks for:

  relaxed feasible   : Lambda_E >= sigma_req            (a relaxed WISE exists);
  integer feasible   : some pure x has wrench + lambda_2 >= sigma_req;
  false positive     : relaxed feasible but NOT integer feasible;
  integrality gap     : (V* - V_int^wise)  (productive value lost to integrality+connectivity);
  connectivity gap    : (V_int^prod - V_int^wise) (value lost to the connectivity constraint).

The enumeration is exact (no rounding, no solver heuristic). Cost is A^N, so it is run
only for small N. We evaluate lambda_2 lazily: wrench-feasible profiles are sorted by V
and the Fiedler value is computed in that order until the first profile clears
sigma_req -- that profile is, by construction, the integer WISE optimum.

Writes generated/oracle_benchmark.csv and paper/figures/oracle_table.tex.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import nullspace as ns, scenarios, wise_sdp  # noqa: E402
from wise_mr.endogenous_graph import complement_basis  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

WTOL = 1e-6          # wrench feasibility tolerance (exact inner-polygon certificate)
CHUNK = 200_000      # profiles per enumeration chunk


def _relaxed(prob):
    """Fluid relaxation: productive optimum V* and spectral capacity Lambda_E."""
    import cvxpy as cp
    A = ns.mass_matrix(prob); B = ns.served_matrix(prob)
    Hw = prob.wrench_matrix(); d = prob.demand().ravel(); v = prob._value()
    n = prob.N * prob.A
    z = cp.Variable(n, nonneg=True); y = B @ z
    cp.Problem(cp.Maximize(cp.sum(cp.multiply(v, y) - 0.5 * prob.alpha * cp.square(y))),
               [A @ z == np.ones(prob.N), Hw @ z >= d]).solve()
    z0 = np.maximum(np.asarray(z.value, float), 0.0)
    y_star = np.asarray(B @ z0, float)
    V_star = float(prob.productive_value(z0.reshape(prob.N, prob.A)))
    # Stage-2 SDP: Lambda_E = max lambda_2 over the optimal fiber Bz = y*
    A_eq = np.vstack([A, B]); b_eq = np.concatenate([np.ones(prob.N), y_star])
    G = np.vstack([-np.eye(n), -Hw]); h = np.concatenate([np.zeros(n), -d])
    ridx = prob.relay_index
    terms = [(i * prob.A + ridx, prob.relay_laplacians[i]) for i in range(prob.N)]
    r = wise_sdp.solve_wise_sdp(n, prob.base_laplacian, terms, complement_basis(prob.N),
                               prob.sigma, A_eq=A_eq, b_eq=b_eq, G_ineq=G, h_ineq=h)
    return V_star, float(r.lambda_star)


def _oracle(prob, sigma=None):
    """Exact enumeration of all A^N pure assignments; returns the integer optima."""
    N, A, H = prob.N, prob.A, prob.H
    P = prob.P
    ridx = prob.relay_index
    cap = prob._cap()                                   # (N,)
    v = float(prob._value()[0]); alpha = prob.alpha
    d = prob.demand().ravel()                           # (P,)
    # Wpad[i,a,l]: directional wrench of robot i taking action a (0 unless a is a slot)
    Wpad = np.zeros((N, A, P))
    Wpad[:, :H, :] = prob.W[:, 0, :, :]                 # slots 0..H-1 carry wrench
    base = np.asarray(prob.base_laplacian, float)
    relayL = np.asarray(prob.relay_laplacians, float)   # (N,V,V)
    sigma = prob.sigma if sigma is None else float(sigma)

    total = A ** N
    feas_V, feas_relay = [], []                         # wrench-feasible profiles
    best_prod = -np.inf                                 # max V over ALL wrench-feasible
    for start in range(0, total, CHUNK):
        end = min(start + CHUNK, total)
        idx = np.arange(start, end)
        prof = np.stack(np.unravel_index(idx, (A,) * N), axis=1)   # (chunk, N) actions
        lift = prof < H                                 # (chunk,N) is-a-lift
        y = lift @ cap                                  # (chunk,) served capacity
        V = v * y - 0.5 * alpha * y * y
        # directional capacity s[c,l] = sum_i Wpad[i, prof[c,i], l]
        rows = np.broadcast_to(np.arange(N), prof.shape)
        Wsel = Wpad[rows, prof]                         # (chunk, N, P)
        s = Wsel.sum(axis=1)                            # (chunk, P)
        wf = np.all(s >= d - WTOL, axis=1)              # wrench-feasible mask
        if wf.any():
            best_prod = max(best_prod, float(V[wf].max()))
            feas_V.append(V[wf])
            feas_relay.append(prof[wf] == ridx)         # (k,N) relayer mask
    if not feas_V:
        return dict(V_int_prod=np.nan, V_int_wise=np.nan, integer_feasible=False,
                    n_wrench_feasible=0)
    feas_V = np.concatenate(feas_V)
    feas_relay = np.concatenate(feas_relay).astype(float)
    order = np.argsort(-feas_V)                         # V descending
    # lazily compute lambda_2 down the V-sorted list; first clearing sigma is the optimum
    V_int_wise, integer_feasible = np.nan, False
    for j in order:
        L = base + np.einsum("i,ijk->jk", feas_relay[j], relayL)
        w = np.linalg.eigvalsh(0.5 * (L + L.T))
        if float(w[1]) >= sigma:                        # lambda_2 = 2nd smallest
            V_int_wise = float(feas_V[j]); integer_feasible = True
            break
    return dict(V_int_prod=float(best_prod), V_int_wise=V_int_wise,
                integer_feasible=integer_feasible, n_wrench_feasible=int(feas_V.size))


def run(sizes=(6, 7, 8), seeds=8, nu=0.5, tau_d=2.0, lift=3.0):
    ZC_TOL = 1e-6          # exact zero-cost integer WISE: V_Z* == V* (a point on the fiber)
    rows = []
    for N in sizes:
        rel_feas = int_feas = fpos = zero_cost = 0
        gaps, rel_gaps, times, considered = [], [], [], []
        y_target = round(0.75 * N)          # keep enough robots free to relay at small N
        for s in range(seeds):
            prob = scenarios.two_region(seed=s, N=N, nu=nu, tau_d=tau_d, lift=lift,
                                        bridge_gain=3.0, y_target=y_target)
            V_star, LamE = _relaxed(prob)
            if not (LamE >= prob.sigma - 1e-9):
                continue                                 # only score instances with a relaxed WISE
            t0 = time.perf_counter()
            o = _oracle(prob)                            # V_int_wise = integer OPTIMAL self-sustaining
            times.append(time.perf_counter() - t0)
            considered.append(o["n_wrench_feasible"])
            rel_feas += 1
            if o["integer_feasible"]:                    # integer self-sustaining feasible set nonempty
                int_feas += 1
                gz = V_star - o["V_int_wise"]            # productive integrality gap g_Z >= 0
                gaps.append(gz)
                rel_gaps.append(gz / (abs(V_star) + 1e-12))   # relative gap g_Z / |V*|
                if gz <= ZC_TOL:                         # zero-cost integer WISE (B zhat = y*)
                    zero_cost += 1
            else:
                fpos += 1
            print(f"N={N} seed={s}: relaxed=1 integer={int(o['integer_feasible'])} "
                  f"V*={V_star:.3f} V_Z*={o['V_int_wise']:.3f} "
                  f"g_Z={V_star - o['V_int_wise']:.4f} "
                  f"(maps={o['n_wrench_feasible']}, {times[-1]:.2f}s)")
        rows.append(dict(
            N=N, seeds_relaxed_feasible=rel_feas, integer_feasible=int_feas,
            zero_cost_integer_wise=zero_cost, false_positive=fpos,
            gZ_mean=float(np.mean(gaps)) if gaps else float("nan"),
            gZ_max=float(np.max(gaps)) if gaps else float("nan"),
            gZ_rel_max=float(100.0 * np.max(rel_gaps)) if rel_gaps else float("nan"),
            runtime_mean_s=float(np.mean(times)) if times else float("nan"),
            maps_enumerated=int(prob.A ** N),
            wrench_feasible_mean=float(np.mean(considered)) if considered else float("nan"),
        ))
        print(f"== N={N}: relaxed {rel_feas}, integer-feasible {int_feas}, "
              f"zero-cost {zero_cost}, false-positive {fpos}, g_Z~{rows[-1]['gZ_mean']:.4f}, "
              f"{rows[-1]['runtime_mean_s']:.2f}s ==")

    fields = list(rows[0].keys())
    with open(GEN / "oracle_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    _write_table(rows, seeds)
    return rows


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def adversarial(sizes=(6, 8), seeds=50, frac=0.97, nu=0.5, tau_d=2.0, lift=3.0):
    """Boundary-adversarial oracle: sigma_req = frac * Lambda_E (just below the fiber max),
    where integer attainment is hardest, so false positives (relaxed- but not integer-
    feasible) are most likely. Reports the count and a Wilson 95% CI."""
    print("=== adversarial oracle (sigma = %.2f * Lambda_E) ===" % frac)
    tot_rel = tot_fp = 0
    for N in sizes:
        y_target = round(0.75 * N)
        rel = fp = 0
        for s in range(seeds):
            prob = scenarios.two_region(seed=s, N=N, nu=nu, tau_d=tau_d, lift=lift,
                                        bridge_gain=3.0, y_target=y_target)
            _, LamE = _relaxed(prob)
            if LamE <= 1e-6:
                continue
            sig = frac * LamE                            # relaxed-feasible by construction
            rel += 1
            o = _oracle(prob, sigma=sig)
            if not o["integer_feasible"]:
                fp += 1
        lo, hi = _wilson(fp, rel)
        tot_rel += rel; tot_fp += fp
        print(f"N={N}: relaxed-feasible {rel}/{seeds}, false positives {fp}/{rel} "
              f"(Wilson95 [{lo:.1%},{hi:.1%}])")
    lo, hi = _wilson(tot_fp, tot_rel)
    print(f"TOTAL: {tot_fp}/{tot_rel} false positives, Wilson95 [{lo:.1%},{hi:.1%}]")
    return tot_fp, tot_rel, (lo, hi)


def _write_table(rows, seeds):
    def _g(r, k, fmt):
        x = r[k]
        return "--" if (isinstance(x, float) and np.isnan(x)) else (fmt % x)

    def _rt(r):
        t = r["runtime_mean_s"]
        return "--" if np.isnan(t) else (r"$<$0.01\,s" if t < 0.01 else r"%.2f\,s" % t)

    tot_rel = sum(r["seeds_relaxed_feasible"] for r in rows)
    tot_fp = sum(r["false_positive"] for r in rows)
    lines = [r"% Auto-generated by experiments/exp_oracle.py -- do not edit by hand.",
             r"% Exact enumeration of all |A|^N robot-action maps (pruned by budget/occupancy",
             r"% /wrench). Columns denominated in relaxed-feasible seeds.",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{rcccccc}", r"\hline",
             r"$N$ & rel.\ WISE & int.\ feas. & zero-cost & $\max g_{\mathbb Z}$"
             r" & $\max\tfrac{g_{\mathbb Z}}{|V^\star|}$ & time \\", r"\hline"]
    for r in rows:
        k = r["seeds_relaxed_feasible"]
        lines.append(r"%d & %d/%d & %d/%d & %d/%d & %s & %s & %s \\" % (
            r["N"], k, seeds,
            r["integer_feasible"], k,
            r["zero_cost_integer_wise"], k,
            _g(r, "gZ_max", r"%.4f"),
            _g(r, "gZ_rel_max", r"%.3f\%%"), _rt(r)))
    lines += [r"\hline",
              r"\multicolumn{7}{l}{\footnotesize $%d/%d$ false positives observed.}\\"
              % (tot_fp, tot_rel),
              r"\end{tabular}"]
    (FIG / "oracle_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
