"""PHASE R0 -- audit and freeze the flagship inputs.

Everything is read from code and from ``generated/flagship.json``; nothing is copied
from the manuscript. Writes ``generated/robot_experiment_input_manifest.json``.

The audit deliberately separates three different ``lambda_2`` values that the paper
currently reports under one symbol, and reports the discrepancy rather than papering
over it:

``lambda2_flagship_record``
    the frozen ``generated/flagship.json`` value, recomputed bit-for-bit by re-running
    ``exp_flagship._lambda2``.
``lambda2_single_position``
    the same composition on a *physically consistent* graph, in which the relaying
    robot occupies exactly one position (its relay site) instead of contributing its
    home short links to ``L_0`` *and* its relay links from the gap.
``lambda2_bar`` / ``lambda2_geo``
    the tube-infimum surrogate and the realized physical value for the robot mission.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import assignments as A
from . import config as C
from . import scenario as S

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "generated"


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                          # pragma: no cover
        return "unknown"


def flagship_record() -> dict:
    return json.loads((GEN / "flagship.json").read_text(encoding="utf-8"))


def reproduce_flagship_lambda2() -> dict:
    """Re-run the frozen flagship's own Laplacian and compare with its JSON record."""
    import exp_flagship as FLAG
    lam_bad, _ = FLAG._lambda2([FLAG.NAME.index("S2"), FLAG.NAME.index("S4")])
    lam_wise, _ = FLAG._lambda2([FLAG.NAME.index("L1")])
    rec = flagship_record()
    return {
        "lambda2_prod_recomputed": lam_bad,
        "lambda2_wise_recomputed": lam_wise,
        "lambda2_prod_record": rec["lambda2_bad"],
        "lambda2_wise_record": rec["lambda2_wise"],
        "max_abs_difference": float(max(abs(lam_bad - rec["lambda2_bad"]),
                                        abs(lam_wise - rec["lambda2_wise"]))),
    }


def single_position_lambda2() -> dict:
    """The flagship composition on a graph where each robot has ONE position.

    ``exp_flagship._laplacian`` builds the always-on short links from every robot's
    *home* and then adds the relay links from the gap for the relaying robot, so the
    relaying long robot contributes edges from two places at once. Re-evaluating the
    same composition with the relayer only at its site is what the physical graph
    ``L_geo(q)`` actually gives -- and it is the quantity Assumption (iv) compares
    against.
    """
    import exp_flagship as FLAG
    names = list(FLAG.NAME)
    pos_home = np.array(FLAG.POS, float)

    def lam(relayers, sites):
        pos = pos_home.copy()
        mask = np.zeros(len(names), bool)
        for i, p in zip(relayers, sites, strict=True):
            pos[i] = p
            mask[i] = True
        return S.lambda2(S.lgeo(pos, mask))

    i_l1 = names.index("L1")
    centre = np.array([float(FLAG.xR), 0.0])
    return {
        "lambda2_wise_single_position": lam([i_l1], [centre]),
        "lambda2_prod_single_position": lam([names.index("S2"), names.index("S4")],
                                            [pos_home[names.index("S2")],
                                             pos_home[names.index("S4")]]),
    }


def build_manifest(bl: A.Baselines, certs: dict) -> dict:
    rec = flagship_record()
    man = {
        "git_rev": _git_rev(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "source_of_truth": {
            "flagship_script": "experiments/exp_flagship.py",
            "flagship_record": "generated/flagship.json",
            "wrench_certificate": "src/wise_mr/wrench_tensor.certify_membership_lp",
            "information_gains": "experiments/exp_modal_stability.py (two_region defaults)",
        },
        "robot_types": {
            "names": S.NAMES,
            "is_long": S.IS_LONG.tolist(),
            "comm_range": S.R_RANGE.tolist(),
            "relay_gain": S.GAMMA.tolist(),
            "capacity_c_tau": S.CAP.tolist(),
            "force_cap_F_tau": S.FORCE.tolist(),
            "short_range_R_S": S.R_SHORT,
            "homes": S.HOMES.tolist(),
        },
        "action_set": {
            "n_actions": S.N_ACTIONS,
            "actions": [list(map(str, a)) for a in S.ACTIONS],
            "relay_sites": {k: v.tolist() for k, v in S.RELAY_SITES.items()},
            "contact_slots_per_load": S.N_SLOTS,
        },
        "wrench_model": {
            "A_kh": [m.tolist() for m in S.SLOT_MAPS],
            "slot_offsets_r_kh": [o.tolist() for o in S.SLOT_OFFSETS],
            "w_dem_certified": [w.tolist() for w in S.W_DEM_CERT],
            "kappa": S.KAPPA,
            "m_sides": S.M_SIDES,
            "nominal_demand_fraction": C.NOMINAL_DEMAND_FRACTION,
        },
        "productive_layer": {
            "alpha": A.ALPHA_V, "v": A.V_COEFF.tolist(),
            "y_star": A.Y_STAR.tolist(), "V_star": A.V_STAR,
            "flagship_served_aggregate": rec["served_aggregate_wise"],
        },
        "information_layer": {
            "m_y": C.M_Y, "theta_1": C.THETA_1, "theta_2": C.THETA_2, "c": C.C_CONS,
            "sigma_dyn": C.SIGMA_DYN, "sigma_req": C.SIGMA_REQ,
            "delta_margin": C.DELTA_MARGIN,
            "alpha_sigma_req": C.alpha_rate(C.SIGMA_REQ),
        },
        "mission": {
            "paths": [{"start": p.start.tolist(), "travel": p.travel.tolist(),
                       "bow": p.bow.tolist(), "arclength": p.arclength()}
                      for p in S.LOAD_PATHS],
            "t_deploy": C.T_DEPLOY, "t_end": C.T_END, "dt": C.DT,
            "control_hz": 1.0 / (C.DT * C.CONTROL_EVERY),
            "path_speed": C.PATH_SPEED,
            "disturbance": {"t": C.T_DIST, "duration": C.DUR_DIST,
                            "drag_multiplier": C.DRAG_MULTIPLIER, "load": 2,
                            "kind": "temporary scaling of load 2's resistance wrench"},
            "drag_eccentricity": [-S.W_DEM_CERT[k][2] / S.W_DEM_CERT[k][0]
                                  for k in range(S.M_LOADS)],
            "tubes": {"rho_track": C.RHO_TRACK, "rho_relay": C.RHO_RELAY,
                      "rho_idle": C.RHO_IDLE},
        },
        "tolerances": {"tau_B": C.TAU_B, "tau_V": C.TAU_V, "tau_Gamma": C.TAU_GAMMA,
                       "tau_eig": C.TAU_EIG, "tau_q": C.TAU_Q, "tau_s": C.TAU_S,
                       "tau_w": C.TAU_W},
        "enumeration": {
            "n_integer_maps": S.N_ACTIONS ** S.N_ROBOTS,
            "n_feasible": len(bl.feasible),
            "n_on_fiber": len(bl.fiber),
            "lambda2_star_fiber": bl.lam_star_fiber,
            "lambda2_star_feasible": bl.lam_star_feasible,
            "distinct_lambda2_on_fiber": sorted({round(bl.lam_bar[a], 9)
                                                 for a in bl.fiber}),
        },
        "assignments": {k: v.as_dict() for k, v in certs.items()},
        "flagship_lambda2_reproduction": reproduce_flagship_lambda2(),
        "flagship_lambda2_single_position": single_position_lambda2(),
    }
    return man


def write_manifest(man: dict) -> Path:
    p = GEN / "robot_experiment_input_manifest.json"
    p.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return p
