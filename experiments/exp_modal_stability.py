"""E5 (paper Prop. stability): what connectivity actually buys, on the flagship.

The productive optimum and the WISE selector have the SAME productive aggregate and
the SAME productive value; they differ only in lambda_2. This script propagates the
reduced information-layer interconnection

    d/dt [a_lambda; b_lambda] = A_lambda [a_lambda; b_lambda],
    A_lambda = [[-m_F, theta_1], [theta_2, -c*lambda]]

for the critical mode lambda = lambda_2 of each assignment and shows the qualitative
change: below sigma_dyn the mode does not decay, above it decays at the certified rate
alpha(lambda).

Discipline (so this is a test, not a demonstration):
  * lambda_2 values are READ from generated/flagship.json, never typed in;
  * the gains are the two_region scenario DEFAULTS (theta=0.5, c=1, m_F=1), fixed
    before either curve is computed and identical for both assignments;
  * both runs use the same initial condition;
  * the solution is the exact matrix exponential, not an Euler integration, so no
    integrator error is mistaken for dynamics;
  * the numerically observed decay rate is checked against the analytic alpha(lambda).

Scope: this is the REDUCED information layer. No robot, load or actuator dynamics and
no closed-loop transport claim.

Writes generated/modal_stability.json, generated/modal_stability.csv and
paper/figures/fig_modal.pdf.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import scenarios  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

E0 = np.array([1.0, 1.0]) / np.sqrt(2.0)     # same for both; excites both coordinates
T_END, N_T = 30.0, 601


def A_lambda(lam: float, m_F: float, c: float, th1: float, th2: float) -> np.ndarray:
    return np.array([[-m_F, th1], [th2, -c * lam]])


def alpha(lam: float, m_F: float, c: float, th1: float, th2: float) -> float:
    """Analytic decay rate: alpha = -max Re spec(A_lambda)."""
    disc = (m_F - c * lam) ** 2 + 4.0 * th1 * th2
    return 0.5 * (m_F + c * lam - np.sqrt(disc))


def rollout(lam, m_F, c, th1, th2, t):
    """Exact e(t) = expm(A t) e0 (no integration error)."""
    A = A_lambda(lam, m_F, c, th1, th2)
    return np.array([expm(A * ti) @ E0 for ti in t])


def observed_rate(t, norms):
    """Least-squares slope of log||e|| over the second half (asymptotic regime)."""
    half = len(t) // 2
    good = norms[half:] > 1e-300
    if good.sum() < 5:
        return float("nan")
    sl = np.polyfit(t[half:][good], np.log(norms[half:][good]), 1)[0]
    return float(-sl)


def main() -> None:
    flag = json.loads((GEN / "flagship.json").read_text())
    lam_prod = float(flag["lambda2_bad"])       # productive-only optimum
    lam_wise = float(flag["lambda2_wise"])      # WISE selector

    # gains: scenario defaults, fixed before any curve is computed
    prob = scenarios.two_region(seed=3)
    th = float(prob.meta.get("theta", 0.5))
    c = float(prob.meta.get("c", 1.0))
    m_F = float(prob.meta.get("m_F", 1.0))
    th1 = th2 = th                               # symmetric instance of (theta_1, theta_2)
    sigma_dyn = th1 * th2 / (c * m_F)

    t = np.linspace(0.0, T_END, N_T)
    runs = {}
    for tag, lam in (("productive_only", lam_prod), ("wise", lam_wise)):
        e = rollout(lam, m_F, c, th1, th2, t)
        norms = np.linalg.norm(e, axis=1)
        A = A_lambda(lam, m_F, c, th1, th2)
        runs[tag] = {
            "lambda2": lam,
            "det_A": float(np.linalg.det(A)),
            "trace_A": float(np.trace(A)),
            "eig_max_real": float(np.max(np.linalg.eigvals(A).real)),
            "alpha_analytic": float(alpha(lam, m_F, c, th1, th2)),
            "alpha_observed": observed_rate(t, norms),
            "hurwitz": bool(np.max(np.linalg.eigvals(A).real) < 0),
            "clears_sigma_dyn": bool(lam > sigma_dyn),
            "norm_final": float(norms[-1]),
            "norm_ratio_final": float(norms[-1] / norms[0]),
            "_norms": norms,
        }

    out = {
        "source": "generated/flagship.json",
        "gains": {"theta_1": th1, "theta_2": th2, "c": c, "m_F": m_F,
                  "provenance": "two_region scenario defaults, fixed before comparison"},
        "sigma_dyn": sigma_dyn,
        "initial_condition": E0.tolist(),
        "t_end": T_END,
        "solver": "scipy.linalg.expm (exact matrix exponential)",
        "scope": ("reduced information-layer interconnection only; no robot, load or "
                  "actuator dynamics, no closed-loop transport claim"),
        "runs": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in runs.items()},
    }
    # consistency: analytic alpha must match the observed exponential slope
    out["alpha_max_abs_error"] = float(max(
        abs(v["alpha_analytic"] - v["alpha_observed"]) for v in runs.values()))
    (GEN / "modal_stability.json").write_text(json.dumps(out, indent=1))

    with (GEN / "modal_stability.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "norm_productive_only", "norm_wise"])
        for i, ti in enumerate(t):
            w.writerow([ti, runs["productive_only"]["_norms"][i], runs["wise"]["_norms"][i]])

    for tag, v in runs.items():
        print(f"{tag:16s} lambda2={v['lambda2']:.4f}  det A={v['det_A']:+.4f}  "
              f"alpha={v['alpha_analytic']:+.4f} (obs {v['alpha_observed']:+.4f})  "
              f"Hurwitz={v['hurwitz']}  ||e(T)||/||e(0)||={v['norm_ratio_final']:.3e}")
    print(f"sigma_dyn = {sigma_dyn:.4f}; alpha analytic-vs-observed max err "
          f"{out['alpha_max_abs_error']:.2e}")

    _figure(t, runs, sigma_dyn)


def _figure(t, runs, sigma_dyn):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure")
        return
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    p, w = runs["productive_only"], runs["wise"]
    ax.semilogy(t, p["_norms"], color="#c0392b", lw=1.5, ls="--",
                label=rf"productive-only $\lambda_2\approx0<\sigma_{{\rm dyn}}$")
    ax.semilogy(t, w["_norms"], color="#1e7a46", lw=1.5,
                label=rf"WISE $\lambda_2={w['lambda2']:.2f}>\sigma_{{\rm dyn}}$")
    ax.set_xlabel("time", fontsize=8)
    ax.set_ylabel(r"$\|[a_\lambda,b_\lambda]\|_2$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.2, frameon=False, loc="lower left")
    ax.set_title(rf"same $Bz$, same $V$; $\sigma_{{\rm dyn}}={sigma_dyn:.2f}$", fontsize=8)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_modal.pdf")
    print(f"wrote {FIG / 'fig_modal.pdf'}")


if __name__ == "__main__":
    main()
