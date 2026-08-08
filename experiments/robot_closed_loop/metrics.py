"""Summary metrics and the predeclared success criterion.

The statistical unit is the seed/world, never the time step: every quantity here
collapses one run to one number, and ``run_monte_carlo.py`` then pairs those numbers
across methods.
"""

from __future__ import annotations

import numpy as np

from . import config as C


def _op(res, key):
    return res.series[key][res.operational]


def fitted_decay_rate(t: np.ndarray, norms: np.ndarray) -> float:
    """Least-squares slope of ``log ||.||`` over the second half of the window."""
    half = len(t) // 2
    good = norms[half:] > 1e-300
    if good.sum() < 5:
        return float("nan")
    return float(-np.polyfit(t[half:][good], np.log(norms[half:][good]), 1)[0])


def settling_time(t: np.ndarray, err: np.ndarray, t0: float, band: float) -> float:
    """First time after ``t0`` from which ``err`` stays inside ``band`` to the end."""
    idx = np.where(t >= t0)[0]
    if idx.size == 0:
        return float("nan")
    inside = err[idx] <= band
    if not inside.any():
        return float("nan")
    bad = np.where(~inside)[0]
    first = 0 if bad.size == 0 else bad[-1] + 1
    if first >= idx.size:
        return float("nan")
    return float(t[idx[first]] - t0)


def summarize(res, cert, sigma_req: float, pert: C.SeedPerturbation) -> dict:
    t = res.t
    op = res.operational
    t_op = t[op]
    lam_geo = _op(res, "lam_geo")
    lam_bar = float(res.series["lam_bar"][0])
    sync = _op(res, "sync_err")
    norm_c = _op(res, "info_norm_certified")
    norm_m = _op(res, "info_norm")
    r1, r2 = _op(res, "wrench_resid1"), _op(res, "wrench_resid2")
    p1, p2 = _op(res, "pose_err1"), _op(res, "pose_err2")

    alpha_cert = C.alpha_rate(min(sigma_req, lam_bar))
    alpha_fit = fitted_decay_rate(t_op, norm_c)
    t_settle = settling_time(t, res.series["sync_err"],
                             pert.t_dist + C.DUR_DIST, C.TAU_S)

    terminal_pose = max(float(res.series["pose_err1"][-1]), float(res.series["pose_err2"][-1]))
    terminal_sync = float(res.series["sync_err"][-1])
    peak_resid = float(max(r1.max(), r2.max()))
    # The declared disturbance is *designed* to exceed the certified capacity, so the
    # success criterion measures wrench delivery outside that window; the in-window
    # peak is reported separately and is (by design) the same for every method.
    # window = the disturbance itself plus the governor's own settling lag, after
    # which the team is expected to deliver the demanded wrench again.
    t_quiet_end = pert.t_dist + C.DUR_DIST + 3.0 * C.GOVERNOR_TAU
    quiet = op & ~((t >= pert.t_dist) & (t < t_quiet_end))
    peak_resid_quiet = float(max(res.series["wrench_resid1"][quiet].max(),
                                 res.series["wrench_resid2"][quiet].max()))
    completion = {}
    for k in (1, 2):
        s = res.series[f"s{k}"]
        done = np.where(s >= 0.999)[0]
        completion[f"t_complete{k}"] = float(t[done[0]]) if done.size else float("nan")

    completed = all(np.isfinite(v) for v in completion.values())
    self_sustaining = (lam_bar < sigma_req) or bool(lam_geo.min() >= C.SIGMA_DYN)
    success = bool(completed and terminal_pose <= C.TAU_Q and terminal_sync <= C.TAU_S
                   and peak_resid_quiet <= C.TAU_W and self_sustaining
                   and lam_geo.min() >= lam_bar - C.TAU_EIG)

    return {
        "method": res.method, "seed": res.seed,
        # assignment / theory
        "V_over_Vstar": cert.V / _v_star(),
        "aggregate_error_inf": cert.aggregate_error,
        "value_error": cert.value_error,
        "gamma_fiber_integer": cert.gamma_fiber_integer,
        "lambda2_bar": lam_bar,
        "lambda2_nominal": cert.lambda2_nominal,
        "margin_vs_sigma_req": cert.margin_vs_sigma_req,
        "wrench_certified": all(cert.wrench_ok.values()),
        # communication
        "min_lambda2_geo": float(lam_geo.min()),
        # conservatism of the surrogate: Delta_geo(t) = lam2(L_geo(q(t))) - lam2(Lbar)
        "min_transfer_margin": float(_op(res, "transfer_margin").min()),
        "p05_transfer_margin": float(np.percentile(_op(res, "transfer_margin"), 5)),
        "median_transfer_margin": float(np.median(_op(res, "transfer_margin"))),
        "n_transfer_violations": int(np.count_nonzero(
            _op(res, "transfer_margin") < -C.TAU_EIG)),
        "n_operational_steps": int(op.sum()),
        "min_loewner_eig": float(_op(res, "loewner_min_eig").min()),
        "max_tube_violation": float(_op(res, "tube_violation").max()),
        "n_steps_outside_tube": int(np.count_nonzero(_op(res, "tube_violation") > 1e-9)),
        # information layer
        "alpha_certified": float(alpha_cert),
        "alpha_at_lambda2_bar": float(C.alpha_rate(lam_bar)),
        "alpha_fitted": float(alpha_fit),
        "info_norm_ratio": float(norm_c[-1] / norm_c[0]),
        "mission_info_norm_max": float(norm_m.max()),
        # transport
        "pose_rmse1": float(np.sqrt(np.mean(p1**2))),
        "pose_rmse2": float(np.sqrt(np.mean(p2**2))),
        "peak_wrench_residual": peak_resid,
        "peak_wrench_residual_quiet": peak_resid_quiet,
        # actuation slack = 1 - max_i |u_i|_inf over the inner-zonotope coefficients;
        # 0 means a robot is on the boundary of U_tau^in, <0 is impossible by construction
        "min_actuation_slack": float(1.0 - max(_op(res, "sat1").max(),
                                               _op(res, "sat2").max())),
        # control steps at which the demanded wrench was not delivered to 1e-6, inside
        # the declared disturbance window (where saturation is by design) and outside it
        "n_infeasible_steps": int(np.count_nonzero(
            (res.series["wrench_resid1"][op] > 1e-6)
            | (res.series["wrench_resid2"][op] > 1e-6)) // C.CONTROL_EVERY),
        "n_infeasible_steps_quiet": int(np.count_nonzero(
            (res.series["wrench_resid1"][quiet] > 1e-6)
            | (res.series["wrench_resid2"][quiet] > 1e-6)) // C.CONTROL_EVERY),
        "mission_completed": completed,
        "wrench_rmse1": float(np.sqrt(np.mean(r1**2))),
        "wrench_rmse2": float(np.sqrt(np.mean(r2**2))),
        "max_sync_err": float(sync.max()),
        "terminal_sync_err": terminal_sync,
        "terminal_pose_err": terminal_pose,
        "settling_time": t_settle,
        "control_effort": float(res.series["control_effort"][-1]),
        **completion,
        "success": success,
    }


def _v_star() -> float:
    from . import assignments as A
    return A.V_STAR
