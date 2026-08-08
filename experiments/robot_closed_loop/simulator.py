"""The closed loop: one deterministic run of one assignment on one world.

Order of operations inside a control update (identical for every method):

    reference -> demanded wrench -> zonotope allocation -> load dynamics ->
    robot poses -> physical graph L_geo(q) -> information layer -> commands

The world (masses, damping, initial offsets, disturbance amplitude and onset) is a
``SeedPerturbation``; the *same* draw is handed to every method, so a comparison is
paired by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import communication as comm
from . import config as C
from . import controllers as ctrl
from . import information_layer as info
from . import load_dynamics as dyn
from . import scenario as S
from . import wrench_allocator as wa

# Drag eccentricity making each load's resistance wrench collinear with its frozen
# certified demand w^dem_k = (F, 0, tau):  ecc = -tau / F.
ECCENTRICITY = np.array([-S.W_DEM_CERT[k][2] / S.W_DEM_CERT[k][0]
                         for k in range(S.M_LOADS)])


@dataclass
class RunResult:
    method: str
    assignment: tuple
    seed: int
    t: np.ndarray
    series: dict = field(default_factory=dict)
    robot_traj: np.ndarray = None          # (T, N, 2)
    load_traj: np.ndarray = None           # (T, 2, 3)
    operational: np.ndarray = None         # bool mask of the certified window
    lred_traj: np.ndarray = None           # (T, N-1, N-1) reduced physical Laplacian


def _disturbance(t: float, load: int, pert: C.SeedPerturbation) -> float:
    """Drag scale of ``load`` at time ``t``: the predeclared asymmetric disturbance."""
    if load == 1 and pert.t_dist <= t < pert.t_dist + C.DUR_DIST:
        return pert.drag_multiplier
    return 1.0


def simulate(method: str, assignment: tuple, lbar: np.ndarray, seed: int = 0,
             pert: C.SeedPerturbation | None = None) -> RunResult:
    pert = pert or C.SeedPerturbation()
    n_steps = int(round(C.T_END / C.DT))
    loads = [dyn.RigidLoad(mass=C.LOAD_MASS * pert.mass_scale[k],
                           inertia=C.LOAD_INERTIA * pert.mass_scale[k],
                           damp_lin=C.DAMP_LIN * pert.damp_scale[k],
                           damp_rot=C.DAMP_ROT * pert.damp_scale[k],
                           ecc=ECCENTRICITY[k])
             for k in range(S.M_LOADS)]
    alloc = wa.build_allocators(assignment)

    q = np.array([ctrl.path_pose(k, 0.0) + pert.pose_offset[k] for k in range(S.M_LOADS)])
    nu = np.zeros((S.M_LOADS, 3))
    s = np.zeros(S.M_LOADS)
    pos = S.HOMES.copy()
    a_m, b_m = info.initial_state()
    a_c, b_c = info.initial_state()
    u = np.full(S.N_ROBOTS, C.U_0)
    # The governor starts closed and opens through its own lag, so the reference
    # velocity (and hence the demanded wrench) starts at zero instead of stepping.
    h_gov = np.zeros(S.M_LOADS)
    held = [np.zeros(3) for _ in range(S.M_LOADS)]
    resid = np.zeros(S.M_LOADS)
    sat = np.zeros(S.M_LOADS)
    forces_sq = np.zeros(S.M_LOADS)
    nu_d_prev = np.zeros((S.M_LOADS, 3))

    keys = ["t", "s1", "s2", "sync_err", "lam_geo", "lam_bar", "transfer_margin",
            "loewner_min_eig", "info_norm", "info_norm_certified", "pose_err1",
            "pose_err2", "wrench_resid1", "wrench_resid2", "sat1", "sat2",
            "control_effort", "u_spread", "tube_violation", "wdem1", "wdem2"]
    rec = {k: np.zeros(n_steps + 1) for k in keys}
    robot_traj = np.zeros((n_steps + 1, S.N_ROBOTS, 2))
    load_traj = np.zeros((n_steps + 1, S.M_LOADS, 3))
    lred_traj = np.zeros((n_steps + 1, S.N_ROBOTS - 1, S.N_ROBOTS - 1))
    operational = np.zeros(n_steps + 1, bool)
    lam_bar_val = S.lambda2(lbar)
    effort = 0.0
    w_dem_norm = np.zeros(S.M_LOADS)

    for n in range(n_steps + 1):
        t = n * C.DT
        attached = t >= C.T_DEPLOY

        # ---- control update -------------------------------------------------
        # During deployment the loads are still on the ground and untouched: no
        # contact wrench, no motion, no guarantee. Only the robots travel.
        if attached and n % C.CONTROL_EVERY == 0:
            dt_c = C.DT * C.CONTROL_EVERY
            for k in range(S.M_LOADS):
                u_k = ctrl.load_command(assignment, u, k)
                h_raw = ctrl.governor(float(np.linalg.norm(ctrl.pose_error(
                    ctrl.path_pose(k, s[k]), q[k]))), resid[k])
                h_gov[k] += dt_c / C.GOVERNOR_TAU * (h_raw - h_gov[k])
                s_dot = C.PATH_SPEED * u_k * h_gov[k]
                q_d, nu_d = ctrl.reference(k, s[k], s_dot)
                nu_d_dot = (nu_d - nu_d_prev[k]) / (C.DT * C.CONTROL_EVERY)
                nu_d_prev[k] = nu_d
                scale = _disturbance(t, k, pert)
                w_dem = ctrl.demanded_wrench(loads[k], q[k], nu[k], q_d, nu_d,
                                             nu_d_dot, drag_scale=scale)
                w_dem_norm[k] = float(np.linalg.norm(w_dem))
                wr, f, r, sa, _ = alloc[k].solve(w_dem)
                held[k], resid[k], sat[k] = wr, r, sa
                forces_sq[k] = float(np.sum(f**2))
                s[k] = min(1.0, s[k] + s_dot * C.DT * C.CONTROL_EVERY)

        # ---- physics --------------------------------------------------------
        if attached:
            for k in range(S.M_LOADS):
                scale = _disturbance(t, k, pert)
                q[k], nu[k] = loads[k].rk4(q[k], nu[k], held[k], np.zeros(3), C.DT,
                                           drag_scale=scale)
        pos = comm.step_positions(pos, assignment, q, C.DT, attached)

        # ---- graph + information layer --------------------------------------
        # The certified replay is (re)started at the beginning of the certified
        # operational window, from the common initial condition, so that H4 compares
        # methods from the same state and only over the window the guarantee covers.
        if attached and not operational[max(0, n - 1)]:
            a_c, b_c = info.initial_state()
        snap = comm.graph_snapshot(pos, assignment, lbar)
        Lred = comm.reduced(snap["L_geo"])
        forcing = info.disagreement_forcing(assignment, s)
        a_m, b_m = info.rk4_step(a_m, b_m, C.C_CONS * Lred, forcing, C.DT)
        a_c, b_c = info.rk4_step(a_c, b_c, C.C_CONS * Lred, np.zeros_like(a_c), C.DT)
        u = info.progress_commands(a_m)
        effort += float(np.sum(forces_sq)) * C.DT

        # ---- record ---------------------------------------------------------
        rec["t"][n] = t
        rec["s1"][n], rec["s2"][n] = s[0], s[1]
        rec["sync_err"][n] = abs(s[0] - s[1])
        rec["lam_geo"][n] = snap["lambda2_geo"]
        rec["lam_bar"][n] = lam_bar_val
        rec["transfer_margin"][n] = snap["transfer_margin"]
        rec["loewner_min_eig"][n] = snap["loewner_min_eig"]
        rec["info_norm"][n] = info.weighted_norm(a_m, b_m)
        rec["info_norm_certified"][n] = info.weighted_norm(a_c, b_c)
        for k in range(S.M_LOADS):
            rec[f"pose_err{k+1}"][n] = float(np.linalg.norm(
                ctrl.pose_error(ctrl.path_pose(k, s[k]), q[k])[:2]))
            rec[f"wrench_resid{k+1}"][n] = resid[k]
            rec[f"sat{k+1}"][n] = sat[k]
            rec[f"wdem{k+1}"][n] = w_dem_norm[k]
        rec["control_effort"][n] = effort
        rec["u_spread"][n] = float(u.max() - u.min())
        rec["tube_violation"][n] = float(np.max(
            comm.tube_violation(pos, assignment, q))) if attached else 0.0
        robot_traj[n] = pos
        load_traj[n] = q
        lred_traj[n] = Lred
        operational[n] = attached

    return RunResult(method=method, assignment=assignment, seed=seed,
                     t=rec["t"], series=rec, robot_traj=robot_traj,
                     load_traj=load_traj, operational=operational, lred_traj=lred_traj)
