"""E-Robot: one deterministic flagship run of PROD, HARD and WISE (+ supplements).

Writes
    generated/robot_experiment_input_manifest.json   (PHASE R0 audit)
    generated/robot_flagship_timeseries.csv
    generated/robot_flagship_summary.json
    generated/robot_experiment_manifest.json

and prints the predeclared hypothesis table H1-H6. Run from the repository root:

    python -m experiments.robot_closed_loop.run_flagship
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.robot_closed_loop import assignments as A  # noqa: E402
from experiments.robot_closed_loop import audit  # noqa: E402
from experiments.robot_closed_loop import config as C  # noqa: E402
from experiments.robot_closed_loop import metrics as M  # noqa: E402
from experiments.robot_closed_loop import scenario as S  # noqa: E402
from experiments.robot_closed_loop import simulator as sim  # noqa: E402

GEN = ROOT / "generated"
PRIMARY = ["PROD", "HARD", "WISE"]
SUPPLEMENT = ["SCALAR", "RANDOM-FIBER"]


def select(sigma_req: float = C.SIGMA_REQ):
    bl = A.build(sigma_req)
    chosen = {"PROD": bl.prod, "WISE": bl.wise}
    if bl.hard is not None:
        chosen["HARD"] = bl.hard
    chosen["SCALAR"] = bl.scalar[min(bl.scalar)]
    chosen["RANDOM-FIBER"] = bl.random_fiber
    certs = {k: A.certify(k, v, bl, sigma_req) for k, v in chosen.items()}
    return bl, chosen, certs


def run_all(seed: int = 0, sigma_req: float = C.SIGMA_REQ):
    bl, chosen, certs = select(sigma_req)
    pert = C.SeedPerturbation.draw(seed) if seed else C.SeedPerturbation()
    runs, summaries = {}, {}
    for name in PRIMARY + SUPPLEMENT:
        if name not in chosen:
            continue
        z = chosen[name]
        res = sim.simulate(name, z, S.lbar(z), seed=seed, pert=pert)
        runs[name] = res
        summaries[name] = M.summarize(res, certs[name], sigma_req, pert)
    return bl, chosen, certs, pert, runs, summaries


# --------------------------------------------------------------------------- #
# predeclared hypotheses
# --------------------------------------------------------------------------- #
def hypotheses(bl, certs, summaries, sigma_req: float) -> dict:
    w = certs["WISE"]
    p = certs["PROD"]
    out = {}
    out["H1_zero_productive_loss"] = {
        "aggregate_error_inf": w.aggregate_error, "tau_B": C.TAU_B,
        "value_error": w.value_error, "tau_V": C.TAU_V,
        "pass": bool(w.aggregate_error <= C.TAU_B and w.value_error <= C.TAU_V),
    }
    out["H2_free_networkability"] = {
        "note": ("Gamma reported here is the *integer-fiber* modulus "
                 "max_{z' in E_Z} lambda2(Lbar(z')) - lambda2(Lbar(z)); the "
                 "continuous Gamma_E of the networkability theorem is not recomputed "
                 "in this experiment."),
        "gamma_at_prod": p.gamma_fiber_integer,
        "gamma_at_wise": w.gamma_fiber_integer, "tau_Gamma": C.TAU_GAMMA,
        "pass": bool(p.gamma_fiber_integer > 0 and w.gamma_fiber_integer <= C.TAU_GAMMA),
    }
    out["H3_geometric_transfer"] = {
        m: {"min_transfer_margin": s["min_transfer_margin"],
            "min_loewner_eig": s["min_loewner_eig"],
            "max_tube_violation": s["max_tube_violation"],
            "pass": bool(s["min_transfer_margin"] >= -C.TAU_EIG)}
        for m, s in summaries.items()}
    out["H3_geometric_transfer"]["pass"] = bool(
        all(v["pass"] for v in out["H3_geometric_transfer"].values() if isinstance(v, dict)))
    out["H4_information_stability"] = {
        m: {"lambda2_bar": s["lambda2_bar"], "sigma_req": sigma_req,
            "sigma_dyn": C.SIGMA_DYN,
            "alpha_certified": s["alpha_certified"], "alpha_fitted": s["alpha_fitted"],
            "clears_sigma_req": bool(s["lambda2_bar"] >= sigma_req),
            "decays_no_slower_than_certified":
                bool(s["alpha_fitted"] >= s["alpha_certified"] - 1e-6)
                if s["lambda2_bar"] >= sigma_req else None}
        for m, s in summaries.items()}
    out["H4_information_stability"]["pass"] = bool(all(
        v["decays_no_slower_than_certified"] for v in out["H4_information_stability"].values()
        if isinstance(v, dict) and v["decays_no_slower_than_certified"] is not None))
    if "WISE" in summaries and "PROD" in summaries:
        sw, sp = summaries["WISE"], summaries["PROD"]
        out["H5_robotic_consequence_vs_PROD"] = {
            "max_sync_err": {"WISE": sw["max_sync_err"], "PROD": sp["max_sync_err"]},
            "terminal_sync_err": {"WISE": sw["terminal_sync_err"],
                                  "PROD": sp["terminal_sync_err"]},
            "pose_rmse1": {"WISE": sw["pose_rmse1"], "PROD": sp["pose_rmse1"]},
            "pose_rmse2": {"WISE": sw["pose_rmse2"], "PROD": sp["pose_rmse2"]},
            "success": {"WISE": sw["success"], "PROD": sp["success"]},
            "pass": bool(sw["max_sync_err"] < sp["max_sync_err"]),
        }
    if "HARD" in summaries:
        sw, sh = summaries["WISE"], summaries["HARD"]
        out["H6_vs_HARD"] = {
            "lambda2_bar": {"WISE": sw["lambda2_bar"], "HARD": sh["lambda2_bar"]},
            "alpha_certified": {"WISE": sw["alpha_certified"],
                                "HARD": sh["alpha_certified"]},
            "alpha_fitted": {"WISE": sw["alpha_fitted"], "HARD": sh["alpha_fitted"]},
            "max_sync_err": {"WISE": sw["max_sync_err"], "HARD": sh["max_sync_err"]},
            "settling_time": {"WISE": sw["settling_time"], "HARD": sh["settling_time"]},
            "wise_faster_information_decay": bool(sw["alpha_fitted"] > sh["alpha_fitted"]),
            "wise_better_sync": bool(sw["max_sync_err"] < sh["max_sync_err"]),
        }
    return out


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def write_timeseries(runs, stride: int = 5):
    cols = ["method", "t", "phase"] + [k for k in next(iter(runs.values())).series
                                       if k != "t"]
    path = GEN / "robot_flagship_timeseries.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name, res in runs.items():
            n = len(res.t)
            for i in range(0, n, stride):
                row = [name, f"{res.t[i]:.4f}",
                       "operational" if res.operational[i] else "deployment"]
                row += [f"{res.series[k][i]:.9g}" for k in cols[3:]]
                w.writerow(row)
    return path


def main(seed: int = 0) -> dict:
    bl, chosen, certs, pert, runs, summaries = run_all(seed=seed)

    man_in = audit.build_manifest(bl, certs)
    audit.write_manifest(man_in)

    hyp = hypotheses(bl, certs, summaries, C.SIGMA_REQ)
    out = {
        "seed": seed,
        "sigma_dyn": C.SIGMA_DYN, "sigma_req": C.SIGMA_REQ,
        "V_star": A.V_STAR, "y_star": A.Y_STAR.tolist(),
        "assignments": {k: A.roles(v) for k, v in chosen.items()},
        "certificates": {k: v.as_dict() for k, v in certs.items()},
        "summaries": summaries,
        "hypotheses": hyp,
        "scope": ("planar auditable simulation of the reduced information layer and "
                  "rigid-load transport; not a nonlinear robot-load stability theorem, "
                  "not hardware, not an actuator-level proof, not distributed "
                  "assignment convergence"),
    }
    (GEN / "robot_flagship_summary.json").write_text(json.dumps(out, indent=2),
                                                     encoding="utf-8")
    run_manifest = {
        "inputs": man_in,
        "outputs": {
            "timeseries": "generated/robot_flagship_timeseries.csv",
            "summary": "generated/robot_flagship_summary.json",
            "figure": "figures/robot_closed_loop.pdf",
            "monte_carlo": ["generated/robot_monte_carlo_runs.csv",
                            "generated/robot_monte_carlo_summary.csv",
                            "generated/robot_statistical_report.json"],
        },
        "hypotheses": hyp,
        "summaries": summaries,
    }
    (GEN / "robot_experiment_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8")
    write_timeseries(runs)
    _report(bl, certs, summaries, hyp, man_in)
    return {"runs": runs, "summaries": summaries, "certs": certs, "hyp": hyp,
            "baselines": bl, "manifest": man_in}


def _report(bl, certs, summaries, hyp, man_in):
    print("=" * 78)
    print(f"E-Robot flagship  |  sigma_dyn = {C.SIGMA_DYN:.4f}   "
          f"sigma_req = {C.SIGMA_REQ:.4f}")
    print(f"integer maps enumerated: {man_in['enumeration']['n_integer_maps']}   "
          f"wrench-feasible: {len(bl.feasible)}   on the fiber E: {len(bl.fiber)}")
    print("-" * 78)
    hdr = ("method", "lam2(Lbar)", "alpha_cert", "alpha_fit", "V/V*", "|Bz-y*|",
           "minlam_geo", "maxSync", "peak r_w", "ok")
    print(f"{hdr[0]:<12} {hdr[1]:>10} {hdr[2]:>10} {hdr[3]:>10} {hdr[4]:>6} "
          f"{hdr[5]:>8} {hdr[6]:>10} {hdr[7]:>8} {hdr[8]:>9} {hdr[9]:>4}")
    for m, s in summaries.items():
        print(f"{m:<12} {s['lambda2_bar']:>10.5f} {s['alpha_certified']:>10.5f} "
              f"{s['alpha_fitted']:>10.5f} {s['V_over_Vstar']:>6.3f} "
              f"{s['aggregate_error_inf']:>8.1e} {s['min_lambda2_geo']:>10.5f} "
              f"{s['max_sync_err']:>8.4f} {s['peak_wrench_residual']:>9.4f} "
              f"{'Y' if s['success'] else 'N':>4}")
    print("-" * 78)
    for k, v in hyp.items():
        status = v.get("pass") if isinstance(v, dict) else None
        print(f"{k}: pass={status}")
    rep = man_in["flagship_lambda2_reproduction"]
    sing = man_in["flagship_lambda2_single_position"]
    print("-" * 78)
    print(f"flagship record reproduced to {rep['max_abs_difference']:.2e}")
    print(f"lambda2(WISE) frozen record         = {rep['lambda2_wise_record']:.6f}")
    print(f"lambda2(WISE) single-position graph = "
          f"{sing['lambda2_wise_single_position']:.6f}   <-- see MISMATCH-1")
    print("=" * 78)


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
