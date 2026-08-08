"""E-Robot scenario: the frozen flagship, given paths, tubes and an action set.

Everything physical is *imported* from ``experiments/exp_flagship.py`` -- robot names
and homes, ranges ``R_S/R_L``, relay gains ``GAMMA_S/GAMMA_L``, capacities
``C_S/C_L``, force caps ``F_S/F_L``, the two loads' contact-slot offsets and their
certified wrench demands -- so no number is retyped from the manuscript.

Two things are *added* for the robot experiment, and both are declared:

1. **Two candidate relay sites** instead of one. The flagship declares the central
   site only; a finite relay-site set ``R`` is part of the paper's model, and a
   second admissible site is what makes the HARD baseline (a threshold-feasible but
   not margin-maximal selection) exist at all. The central site reproduces the
   frozen flagship ``lambda_2`` bit-for-bit (checked in ``run_flagship.py``).
2. **Transport paths and tubes.** The flagship is a single nominal pose; a mission
   moves the loads, so each action gets an admissible position tube and
   ``Lbar`` is built from tube *infima* -- the lower-bound Laplacian of
   Lemma "bridge". Because the selection here is by exhaustive enumeration over
   integer assignments (``assignments.py``), ``Lbar`` is evaluated per assignment
   with the tubes of its realized actions; no affine-in-``z`` surrogate is needed
   or claimed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import exp_flagship as FLAG  # noqa: E402  (frozen flagship record)

from . import config as C  # noqa: E402

# ---- frozen flagship constants (imported, never retyped) -------------------- #
NAMES = list(FLAG.NAME)                          # ['L1','L2','S1','S2','S3','S4']
HOMES = np.array(FLAG.POS, dtype=float)          # pre-deployment positions
IS_LONG = np.array(FLAG.IS_LONG, dtype=bool)
N_ROBOTS = len(NAMES)
R_RANGE = np.where(IS_LONG, FLAG.R_L, FLAG.R_S)  # comm range per robot
GAMMA = np.where(IS_LONG, FLAG.GAMMA_L, FLAG.GAMMA_S)
CAP = np.where(IS_LONG, FLAG.C_L, FLAG.C_S)      # served-capacity weight c_tau
FORCE = np.where(IS_LONG, FLAG.F_L, FLAG.F_S)    # planar force cap F_tau
R_SHORT = float(FLAG.R_S)

SLOT_MAPS = [np.array(FLAG.SLOTS1), np.array(FLAG.SLOTS2)]     # (H_k, 3, 2) A_kh
W_DEM_CERT = [np.array(FLAG.W1_DEM), np.array(FLAG.W2_DEM)]    # certified demand
N_SLOTS = [SLOT_MAPS[0].shape[0], SLOT_MAPS[1].shape[0]]       # [4, 2]
# contact offsets r_kh recovered from A_kh = [[1,0],[0,1],[-ry,rx]]
SLOT_OFFSETS = [np.array([[A[2, 1], -A[2, 0]] for A in SLOT_MAPS[k]]) for k in range(2)]

M_LOADS = 2
KAPPA = 1.0          # flagship certifies with kappa = 1 (see exp_flagship._wrench_ok)
M_SIDES = 12         # inner 2m-gon used by the flagship certificate

# ---- relay sites ----------------------------------------------------------- #
# The relay corridor is sampled on a declared cross-shaped grid centred on the
# flagship's single site (xR, 0). Using a *grid* rather than a hand-chosen pair is
# what keeps the HARD baseline honest: HARD takes the cheapest grid point that
# clears sigma_req and WISE the one maximising lambda_2, so neither is picked by
# hand. The grid is fixed here before any lambda_2 was computed.
_RELAY_GRID = {"C": (0.0, 0.0), "W": (-0.8, 0.0), "E": (+0.8, 0.0),
               "N": (0.0, +1.2), "S": (0.0, -1.2)}
RELAY_SITES = {k: np.array([float(FLAG.xR) + dx, dy]) for k, (dx, dy) in _RELAY_GRID.items()}
RELAY_KEYS = list(_RELAY_GRID)

# ---- action set ------------------------------------------------------------ #
# ('lift', k, h) | ('relay', key) | ('idle',)
ACTIONS: list[tuple] = ([("lift", k, h) for k in range(M_LOADS) for h in range(N_SLOTS[k])]
                        + [("relay", key) for key in RELAY_KEYS]
                        + [("idle",)])
A_INDEX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

# ---- deployment cost g[i, a] (the repo's cost model, scenarios.py) ---------- #
_MOVE_COST, _RELAY_COST, _IDLE_COST = 0.35, 0.20, 0.15


# --------------------------------------------------------------------------- #
# transport paths
# --------------------------------------------------------------------------- #
def _smoothstep(s: np.ndarray | float) -> np.ndarray | float:
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@dataclass(frozen=True)
class LoadPath:
    """Geometric path of one load: pose ``q_k(s) = (x, y, theta)``, ``s in [0,1]``.

    ``travel`` is the net world displacement (smoothstep profile) and ``bow`` a
    single sine lobe transverse to it, so the two loads follow *distinct* curved
    paths. Orientation is held constant, so the body-frame contact maps ``A_kh``
    -- and hence the frozen wrench certificate -- are untouched.
    """
    start: np.ndarray            # (x, y, theta) at s = 0
    travel: np.ndarray           # net (dx, dy) displacement over the path
    bow: np.ndarray              # transverse sine-lobe amplitude (bx, by)

    def pose(self, s):
        s = np.clip(np.asarray(s, float), 0.0, 1.0)
        f, lobe = _smoothstep(s), np.sin(np.pi * s)
        x = self.start[0] + self.travel[0] * f + self.bow[0] * lobe
        y = self.start[1] + self.travel[1] * f + self.bow[1] * lobe
        return np.stack([x, y, np.full_like(np.asarray(x, float), self.start[2])], axis=-1)

    def arclength(self, n: int = 400) -> float:
        p = np.atleast_2d(self.pose(np.linspace(0.0, 1.0, n)))[:, :2]
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


# The two regions sit either side of the gap. Both loads travel *transversely* to
# the gap axis and bow toward it: the transport is long (1.2 m) while each region
# stays compact, so the inter-region distance never drops to the short range R_S
# and the relay site stays inside the long range R_L throughout. Load 2's body
# frame is rotated by pi and the two loads counter-travel, giving distinct paths.
LOAD_PATHS = [
    LoadPath(start=np.array([0.90, -0.60, 0.5 * np.pi]),
             travel=np.array([0.0, 1.20]), bow=np.array([0.25, 0.0])),
    LoadPath(start=np.array([7.10, 0.60, -0.5 * np.pi]),
             travel=np.array([0.0, -1.20]), bow=np.array([-0.25, 0.0])),
]


def slot_world(k: int, h: int, s) -> np.ndarray:
    """World position of contact slot (k, h) when load k is at progress ``s``."""
    q = LOAD_PATHS[k].pose(s)
    return slot_world_from_pose(k, h, q)


def slot_world_from_pose(k: int, h: int, q: np.ndarray) -> np.ndarray:
    q = np.atleast_2d(np.asarray(q, float))
    th = q[:, 2]
    r = SLOT_OFFSETS[k][h]
    x = q[:, 0] + np.cos(th) * r[0] - np.sin(th) * r[1]
    y = q[:, 1] + np.sin(th) * r[0] + np.cos(th) * r[1]
    out = np.stack([x, y], axis=-1)
    return out[0] if out.shape[0] == 1 else out


# --------------------------------------------------------------------------- #
# admissible tubes (per robot and action)
# --------------------------------------------------------------------------- #
def tube(i: int, a: tuple) -> tuple[np.ndarray, float]:
    """Sampled centre points and inflation radius of the tube of robot ``i`` in ``a``."""
    if a[0] == "lift":
        _, k, h = a
        s = np.linspace(0.0, 1.0, C.N_TUBE_SAMPLES)
        return slot_world(k, h, s), C.RHO_TRACK
    if a[0] == "relay":
        return RELAY_SITES[a[1]][None, :], C.RHO_RELAY
    return HOMES[i][None, :], C.RHO_IDLE


@cache
def tube_max_distance(i: int, ai: tuple, j: int, aj: tuple) -> float:
    """Worst-case (supremum) distance between two tubes -> tube *infimum* weight."""
    Pi, ri = tube(i, ai)
    Pj, rj = tube(j, aj)
    d = np.linalg.norm(Pi[:, None, :] - Pj[None, :, :], axis=-1)
    return float(d.max() + ri + rj)


# --------------------------------------------------------------------------- #
# Laplacians -- identical weight law for Lbar (tube infima) and L_geo (positions)
# --------------------------------------------------------------------------- #
def _base_weight(d: float | np.ndarray) -> np.ndarray:
    """Always-on short link: exp(-d/2) inside R_S, else 0 (flagship weight law)."""
    d = np.asarray(d, float)
    return np.where(d <= R_SHORT, np.exp(-d / 2.0), 0.0)


# Degradation of the relay channel used by the predeclared robustness sweep: a single
# scalar in (0, 1] multiplying every gated relay conductance. It enters *both* Lbar and
# L_geo, so the Loewner comparison of Lemma "bridge" is unaffected by it.
RELAY_ATTENUATION = 1.0


def set_relay_attenuation(value: float) -> float:
    """Set the relay-channel attenuation and return the previous value."""
    global RELAY_ATTENUATION
    prev, RELAY_ATTENUATION = RELAY_ATTENUATION, float(value)
    return prev


def _relay_weight(d: float | np.ndarray, i: int) -> np.ndarray:
    """Gated relay link of robot ``i``: gamma_tau exp(-d/3) inside its range."""
    d = np.asarray(d, float)
    return np.where(d <= R_RANGE[i],
                    RELAY_ATTENUATION * GAMMA[i] * np.exp(-d / 3.0), 0.0)


def _laplacian_from_adj(W: np.ndarray) -> np.ndarray:
    return np.diag(W.sum(1)) - W


def lbar(assignment: tuple) -> np.ndarray:
    """Lower-bound (tube-infimum) Laplacian ``Lbar(z_hat)`` of an integer assignment.

    ``assignment[i]`` is the action tuple of robot ``i``. Every edge weight is the
    infimum over the two robots' admissible tubes, so ``L_geo(q) >= Lbar`` for every
    pose with all robots inside their tubes (Lemma "bridge").
    """
    W = np.zeros((N_ROBOTS, N_ROBOTS))
    for i in range(N_ROBOTS):
        for j in range(i + 1, N_ROBOTS):
            d = tube_max_distance(i, assignment[i], j, assignment[j])
            W[i, j] = W[j, i] = _base_weight(d)
    for i in range(N_ROBOTS):
        if assignment[i][0] != "relay":
            continue
        for j in range(N_ROBOTS):
            if j == i:
                continue
            d = tube_max_distance(i, assignment[i], j, assignment[j])
            w = _relay_weight(d, i)
            W[i, j] += w
            W[j, i] = W[i, j]
    return _laplacian_from_adj(W)


def lgeo(pos: np.ndarray, relay_mask: np.ndarray) -> np.ndarray:
    """Physical Laplacian ``L_geo(q)`` from actual robot positions -- same weight law."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    W = _base_weight(d)
    np.fill_diagonal(W, 0.0)
    for i in np.where(relay_mask)[0]:
        w = _relay_weight(d[i], i)
        w[i] = 0.0
        W[i] = W[i] + w
        W[:, i] = W[i]
    W = 0.5 * (W + W.T)
    return _laplacian_from_adj(W)


def lambda2(L: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(0.5 * (L + L.T))[1])


def nominal_pose_laplacian(assignment: tuple) -> np.ndarray:
    """``L_geo`` at the *flagship nominal pose*: everyone at their home, relayers at
    their site. Used only to reproduce the frozen ``generated/flagship.json``."""
    pos = HOMES.copy()
    relay = np.zeros(N_ROBOTS, bool)
    for i, a in enumerate(assignment):
        if a[0] == "relay":
            pos[i] = RELAY_SITES[a[1]]
            relay[i] = True
    return lgeo(pos, relay)


# --------------------------------------------------------------------------- #
# deployment cost
# --------------------------------------------------------------------------- #
def action_cost(i: int, a: tuple) -> float:
    if a[0] == "lift":
        _, k, h = a
        return _MOVE_COST * float(np.linalg.norm(HOMES[i] - slot_world(k, h, 0.0)))
    if a[0] == "relay":
        return _RELAY_COST * float(np.linalg.norm(HOMES[i] - RELAY_SITES[a[1]]))
    return _IDLE_COST


def assignment_cost(assignment: tuple) -> float:
    return float(sum(action_cost(i, a) for i, a in enumerate(assignment)))
