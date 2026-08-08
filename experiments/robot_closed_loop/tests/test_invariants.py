"""PHASE R13 -- the invariants that make the comparison a test rather than a demo.

Numbering follows the experiment plan:

 1  same initial condition across methods
 2  same controller gains across methods
 3  WISE preserves the productive aggregate and value
 4  wrench feasibility of every reported assignment
 5  integer, directly re-certified assignments
 6  no confusion between Lbar and L_geo
 7  the relay stays inside its admissible tube
 8  the operational-window graph lower bound holds
 9  analytic alpha vs. frozen-graph numerical slope
10  the time-varying common-Lyapunov inequality
11  deterministic reproduction from a seed
12  every plotted number is sourced from the CSV / manifest
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.robot_closed_loop import assignments as A
from experiments.robot_closed_loop import communication as comm
from experiments.robot_closed_loop import config as C
from experiments.robot_closed_loop import information_layer as info
from experiments.robot_closed_loop import scenario as S
from experiments.robot_closed_loop import simulator as sim

PRIMARY = ("PROD", "HARD", "WISE")


# 1 -------------------------------------------------------------------------
def test_same_initial_condition(flagship):
    runs = flagship["runs"]
    a0, b0 = info.initial_state()
    assert np.linalg.norm(a0) > 0 and np.linalg.norm(b0) > 0
    ref = runs["WISE"]
    for m in PRIMARY:
        r = runs[m]
        assert r.load_traj[0] == pytest.approx(ref.load_traj[0], abs=1e-12)
        assert r.series["info_norm"][0] == pytest.approx(ref.series["info_norm"][0],
                                                         abs=1e-12)
        # every method starts from the same homes; the first sample is recorded one
        # deployment step later, so it must sit exactly on that first-order step
        tgt = comm.target_positions(r.assignment, r.load_traj[0])
        expected = S.HOMES + comm.K_DEPLOY * C.DT * (tgt - S.HOMES)
        assert r.robot_traj[0] == pytest.approx(expected, abs=1e-9)


# 2 -------------------------------------------------------------------------
def test_same_controller_gains_and_allocator(flagship):
    """The allocator, PD gains, governor and disturbance are module-level constants,
    so they cannot differ per method; assert the values the run actually used."""
    from experiments.robot_closed_loop import wrench_allocator as wa
    assert (C.KP_POS, C.KD_POS, C.KP_ROT, C.KD_ROT) == (25.0, 5.0, 8.0, 2.0)
    assert wa.RHO == 5.0e3 and wa.RIDGE == 1.0e-6
    pert = flagship["pert"]
    for m in PRIMARY:
        assert flagship["runs"][m].seed == 0
    assert pert.drag_multiplier == C.DRAG_MULTIPLIER and pert.t_dist == C.T_DIST


# 3 -------------------------------------------------------------------------
def test_wise_preserves_aggregate_and_value(flagship):
    for m in PRIMARY:
        cert = flagship["certs"][m]
        assert cert.aggregate_error <= C.TAU_B
        assert cert.value_error <= C.TAU_V
        assert cert.y == pytest.approx(A.Y_STAR, abs=1e-12)


# 4 -------------------------------------------------------------------------
def test_wrench_feasibility_recertified(flagship):
    for m, cert in flagship["certs"].items():
        assert all(cert.wrench_ok.values()), m


# 5 -------------------------------------------------------------------------
def test_assignments_are_integer_and_occupancy_valid(flagship):
    for m, cert in flagship["certs"].items():
        assert cert.integral and cert.occupancy_ok, m
        assert len(cert.assignment) == S.N_ROBOTS
        assert all(a in S.A_INDEX for a in cert.assignment)


# 6 -------------------------------------------------------------------------
def test_lbar_and_lgeo_are_distinct_and_never_swapped(flagship):
    z = flagship["chosen"]["WISE"]
    lbar = S.lbar(z)
    res = flagship["runs"]["WISE"]
    assert not np.allclose(lbar, res.series["lam_geo"][0] * np.eye(S.N_ROBOTS))
    # the recorded lam_bar is constant (it is a property of the assignment only)
    assert res.series["lam_bar"].std() == pytest.approx(0.0, abs=1e-15)
    assert res.series["lam_bar"][0] == pytest.approx(S.lambda2(lbar), abs=1e-12)
    # lam_geo genuinely moves (the robots move), so the two are not the same object
    op = res.operational
    assert res.series["lam_geo"][op].std() > 1e-4


# 7 -------------------------------------------------------------------------
def test_relay_stays_inside_its_tube(flagship):
    for m in PRIMARY:
        res = flagship["runs"][m]
        op = res.operational
        assert res.series["tube_violation"][op].max() == pytest.approx(0.0, abs=1e-9), m


# 8 -------------------------------------------------------------------------
def test_operational_graph_lower_bound(flagship):
    for m in PRIMARY:
        s = flagship["summaries"][m]
        assert s["min_transfer_margin"] >= -C.TAU_EIG, m
        assert s["min_loewner_eig"] >= -1e-9, m


# 9 -------------------------------------------------------------------------
@pytest.mark.parametrize("lam", [0.05, 0.25, 0.30, 0.4, 1.0, 2.5])
def test_alpha_matches_frozen_graph_slope(lam):
    from scipy.linalg import expm
    g = C.InfoGains()
    Amat = np.array([[-g.m_y, g.theta_1], [g.theta_2, -g.c * lam]])
    t = np.linspace(0.0, 60.0, 601)
    e0 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    norms = np.array([np.linalg.norm(expm(Amat * ti) @ e0) for ti in t])
    half = len(t) // 2
    slope = -np.polyfit(t[half:], np.log(norms[half:]), 1)[0]
    assert slope == pytest.approx(C.alpha_rate(lam), abs=1e-8)


def test_alpha_sign_matches_sigma_dyn():
    for sig in (0.10, 0.2499, 0.25, 0.2501, 0.5):
        a = C.alpha_rate(sig)
        if sig > C.SIGMA_DYN:
            assert a > 0
        elif sig < C.SIGMA_DYN:
            assert a < 0
        else:
            assert abs(a) < 1e-12


# 10 ------------------------------------------------------------------------
def test_alpha_is_the_generalized_eigenvalue():
    g = C.InfoGains()
    for sig in (0.05, 0.25, 0.3, 0.7, 2.0):
        M = np.array([[g.theta_2 * g.m_y, -g.theta_1 * g.theta_2],
                      [-g.theta_1 * g.theta_2, g.theta_1 * g.c * sig]])
        P = np.diag([g.theta_2, g.theta_1])
        Ph = np.diag(1.0 / np.sqrt([g.theta_2, g.theta_1]))
        lam_min = np.linalg.eigvalsh(Ph @ M @ Ph).min()
        assert lam_min == pytest.approx(C.alpha_rate(sig), abs=1e-12)
        assert np.linalg.eigvals(np.linalg.solve(P, M)).real.min() == pytest.approx(
            C.alpha_rate(sig), abs=1e-10)


def test_time_varying_lyapunov_inequality(flagship):
    """dV_c/dt <= -2 alpha(sigma_req) V_c at every recorded operational step, for the
    methods whose certificate clears sigma_req."""
    g = C.InfoGains()
    for m in PRIMARY:
        res = flagship["runs"][m]
        if res.series["lam_bar"][0] < C.SIGMA_REQ:
            continue
        a, b = info.initial_state()
        alpha = C.alpha_rate(C.SIGMA_REQ)
        worst = -np.inf
        idx = np.where(res.operational)[0]
        for n in idx[::20]:
            Lred = g.c * res.lred_traj[n]
            Vc = 0.5 * (g.theta_2 * a @ a + g.theta_1 * b @ b)
            da = -g.m_y * a + g.theta_1 * b
            db = g.theta_2 * a - Lred @ b
            dVc = g.theta_2 * a @ da + g.theta_1 * b @ db
            worst = max(worst, dVc + 2 * alpha * Vc)
            for _ in range(20):
                a, b = info.rk4_step(a, b, g.c * res.lred_traj[n], np.zeros_like(a), C.DT)
        assert worst <= 1e-10, (m, worst)


# 11 ------------------------------------------------------------------------
def test_deterministic_reproduction(flagship):
    z = flagship["chosen"]["WISE"]
    r1 = sim.simulate("WISE", z, S.lbar(z), seed=0)
    r2 = sim.simulate("WISE", z, S.lbar(z), seed=0)
    for k in r1.series:
        assert r1.series[k] == pytest.approx(r2.series[k], abs=0.0, rel=0.0)
    ref = flagship["runs"]["WISE"]
    assert r1.series["sync_err"] == pytest.approx(ref.series["sync_err"], abs=0.0)


def test_seed_perturbation_is_reproducible():
    p1, p2 = C.SeedPerturbation.draw(7), C.SeedPerturbation.draw(7)
    assert p1.mass_scale == pytest.approx(p2.mass_scale, abs=0.0)
    assert p1.t_dist == p2.t_dist
    assert C.SeedPerturbation.draw(8).t_dist != p1.t_dist


# 12 ------------------------------------------------------------------------
def test_every_plotted_number_comes_from_disk(flagship, tmp_path):
    """The figure reads only the CSV and the summary JSON; check the CSV agrees with
    the in-memory run it was written from."""
    import json
    from pathlib import Path

    from experiments.robot_closed_loop import make_figure as MF
    from experiments.robot_closed_loop import run_flagship as RF

    RF.write_timeseries(flagship["runs"])
    ts = MF.load_timeseries()
    gen = Path(RF.GEN)
    summary = json.loads((gen / "robot_flagship_summary.json").read_text(encoding="utf-8"))
    for m in PRIMARY:
        rec = flagship["runs"][m].series
        assert ts[m]["lam_geo"][0] == pytest.approx(rec["lam_geo"][0], rel=1e-8)
        assert ts[m]["sync_err"][-1] == pytest.approx(rec["sync_err"][-5], rel=1e-6,
                                                      abs=1e-9)
        assert summary["summaries"][m]["lambda2_bar"] == pytest.approx(
            rec["lam_bar"][0], rel=1e-12)


# extra: the frozen flagship record is reproduced bit-for-bit -----------------
def test_flagship_record_reproduced():
    from experiments.robot_closed_loop import audit
    rep = audit.reproduce_flagship_lambda2()
    assert rep["max_abs_difference"] < 1e-12


def test_prod_is_disconnected_and_wise_is_not(flagship):
    assert flagship["certs"]["PROD"].lambda2_bar == pytest.approx(0.0, abs=1e-12)
    assert flagship["certs"]["WISE"].lambda2_bar > C.SIGMA_REQ
    assert flagship["certs"]["WISE"].lambda2_bar >= flagship["certs"]["HARD"].lambda2_bar


# 13 ------------------------------------------------------------------------
def test_relay_attenuation_is_neutral_by_default():
    """The robustness sweep must not leak into any other result: the attenuation is
    1.0 unless a sweep sets it, and every sweep restores it."""
    assert S.RELAY_ATTENUATION == 1.0
    prev = S.set_relay_attenuation(0.5)
    assert prev == 1.0 and S.RELAY_ATTENUATION == 0.5
    S.set_relay_attenuation(prev)
    assert S.RELAY_ATTENUATION == 1.0


def test_attenuation_scales_only_relay_links(flagship):
    z = flagship["chosen"]["WISE"]
    nominal = S.lambda2(S.lbar(z))
    prev = S.set_relay_attenuation(0.5)
    try:
        attenuated = S.lambda2(S.lbar(z))
    finally:
        S.set_relay_attenuation(prev)
    assert attenuated < nominal
    assert S.lambda2(S.lbar(z)) == pytest.approx(nominal, abs=1e-12)
    # PROD has no relay, so attenuation cannot change it
    zp = flagship["chosen"]["PROD"]
    base = S.lambda2(S.lbar(zp))
    prev = S.set_relay_attenuation(0.5)
    try:
        assert S.lambda2(S.lbar(zp)) == pytest.approx(base, abs=1e-12)
    finally:
        S.set_relay_attenuation(prev)
