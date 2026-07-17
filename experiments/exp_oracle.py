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


def _oracle(prob):
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
    sigma = prob.sigma

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
    rows = []
    for N in sizes:
        rel_feas = int_feas = fpos = 0
        gaps, cgaps, times, considered = [], [], [], []
        used = 0
        y_target = round(0.75 * N)          # keep enough robots free to relay at small N
        for s in range(seeds):
            prob = scenarios.two_region(seed=s, N=N, nu=nu, tau_d=tau_d, lift=lift,
                                        bridge_gain=3.0, y_target=y_target)
            V_star, LamE = _relaxed(prob)
            r_feasible = LamE >= prob.sigma - 1e-9
            if not r_feasible:
                continue                                 # only score instances with a relaxed WISE
            used += 1
            t0 = time.perf_counter()
            o = _oracle(prob)
            times.append(time.perf_counter() - t0)
            considered.append(o["n_wrench_feasible"])
            rel_feas += 1
            if o["integer_feasible"]:
                int_feas += 1
                gaps.append(V_star - o["V_int_wise"])
                cgaps.append(o["V_int_prod"] - o["V_int_wise"])
            else:
                fpos += 1
            print(f"N={N} seed={s}: relaxed=1 integer={int(o['integer_feasible'])} "
                  f"V*={V_star:.3f} V_int^wise={o['V_int_wise']:.3f} "
                  f"gap={V_star - o['V_int_wise']:.3f} "
                  f"(wf profiles={o['n_wrench_feasible']}, {times[-1]:.1f}s)")
        rows.append(dict(
            N=N, seeds_relaxed_feasible=rel_feas, integer_feasible=int_feas,
            false_positive=fpos,
            false_positive_rate=(fpos / rel_feas) if rel_feas else float("nan"),
            integrality_gap_mean=float(np.mean(gaps)) if gaps else float("nan"),
            integrality_gap_max=float(np.max(gaps)) if gaps else float("nan"),
            connectivity_gap_mean=float(np.mean(cgaps)) if cgaps else float("nan"),
            runtime_mean_s=float(np.mean(times)) if times else float("nan"),
            profiles_enumerated=int(6 ** N),
            wrench_feasible_mean=float(np.mean(considered)) if considered else float("nan"),
        ))
        print(f"== N={N}: relaxed-feasible {rel_feas}, integer-feasible {int_feas}, "
              f"false-positive {fpos}, gap~{rows[-1]['integrality_gap_mean']:.3f}, "
              f"{rows[-1]['runtime_mean_s']:.1f}s ==")

    fields = list(rows[0].keys())
    with open(GEN / "oracle_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    _write_table(rows, seeds)
    return rows


def _write_table(rows, seeds):
    def _g(r, k, fmt):
        x = r[k]
        return "--" if (isinstance(x, float) and np.isnan(x)) else (fmt % x)

    lines = [r"% Auto-generated by experiments/exp_oracle.py -- do not edit by hand.",
             r"% Exact enumeration of all $A^N$ pure assignments (no rounding).",
             r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{rccccc}", r"\hline",
             r"$N$ & rel.\ feas. & int.\ feas. & false pos. & int.\ gap $V^\star\!-\!V^{\mathbb Z}$"
             r" & time \\", r"\hline"]
    for r in rows:
        fpr = r["false_positive_rate"]
        fp = "--" if (isinstance(fpr, float) and np.isnan(fpr)) else (r"%.0f\%%" % (100 * fpr))
        lines.append(r"%d & %d/%d & %d/%d & %s & %s & %s \\" % (
            r["N"], r["seeds_relaxed_feasible"], seeds,
            r["integer_feasible"], r["seeds_relaxed_feasible"],
            fp, _g(r, "integrality_gap_mean", r"%.3f"),
            _g(r, "runtime_mean_s", r"%.1f\,s")))
    lines += [r"\hline", r"\end{tabular}"]
    (FIG / "oracle_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
