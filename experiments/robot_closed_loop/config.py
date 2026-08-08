"""E-Robot: every constant, gain and tolerance, declared in one place.

Nothing in this file may be changed after a comparison has been run: the whole point
of the experiment is that PROD, HARD and WISE differ *only* in the assignment, so
every dynamic, gain, tolerance, disturbance and initial condition below is shared.

Provenance of the physical/graph constants is the frozen flagship script
``experiments/exp_flagship.py`` (they are imported, not retyped -- see
``scenario.py``); the information-layer gains are the ``two_region`` scenario
defaults already used by ``experiments/exp_modal_stability.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# information layer (gains fixed before any curve is computed; same as E5)
# --------------------------------------------------------------------------- #
M_Y = 1.0           # damping m_y of the coordination-error state a
THETA_1 = 0.5       # a <- b coupling
THETA_2 = 0.5       # b <- a coupling
C_CONS = 1.0        # consensus gain multiplying Q^T L_geo Q
SIGMA_DYN = THETA_1 * THETA_2 / (C_CONS * M_Y)          # = 0.25, Prop. "stability"
DELTA_MARGIN = 0.05                                      # design margin
SIGMA_REQ = SIGMA_DYN + DELTA_MARGIN                     # = 0.30

# downstream (illustrative) realization of the coordination layer
U_0 = 1.0           # nominal progress command
K_A = 1.0           # command gain on the mode-aligned error  u_i = U_0 - K_A [Q a]_i
K_W = 4.0           # excitation gain of the progress-disagreement measurement
U_MIN, U_MAX = 0.0, 2.0                                  # command saturation (declared)

# --------------------------------------------------------------------------- #
# tubes (admissible position sets used to build the lower-bound Laplacian Lbar)
# --------------------------------------------------------------------------- #
RHO_TRACK = 0.12    # lifter tracking tube around the nominal swept slot path [m]
RHO_RELAY = 0.15    # relay tube radius around its site [m]
RHO_IDLE = 0.15     # idle tube radius around the robot's home [m]
N_TUBE_SAMPLES = 41

# --------------------------------------------------------------------------- #
# load dynamics (identical for both loads up to the per-seed perturbation)
# --------------------------------------------------------------------------- #
LOAD_MASS = 6.0
LOAD_INERTIA = 1.2
DAMP_LIN = 44.0     # so the nominal drag at peak path speed is ~55% of the certified F
DAMP_ROT = 6.0

# fraction of the *certified* wrench demand that the nominal mission actually asks
# for; the certified w^dem of the flagship is the worst case over the mission and
# the closed loop must stay inside it, leaving the rest as control authority.
NOMINAL_DEMAND_FRACTION = 0.55

# pose PD controller (identical for every method). Sized against the certified
# capacity: a position error of GOVERNOR_ETOL asks for roughly the whole budget.
KP_POS, KP_ROT = 25.0, 8.0
KD_POS, KD_ROT = 5.0, 2.0

# --------------------------------------------------------------------------- #
# mission
# --------------------------------------------------------------------------- #
T_DEPLOY = 6.0      # deployment phase: no graph guarantee is claimed
T_END = 70.0        # end of the run (operational phase is [T_DEPLOY, T_END])
DT = 0.01           # integrator step (fixed-step RK4)
CONTROL_EVERY = 5   # wrench allocation / commands run at 1/(DT*CONTROL_EVERY) = 20 Hz

PATH_SPEED = 1.0 / 48.0      # nominal progress rate ds/dt at u = U_0 and h = 1  [1/s]
GOVERNOR_ETOL = 0.06         # pose error at which the progress governor stops the load [m]
GOVERNOR_TAU = 1.5           # first-order lag of the governor [s]: keeps the reference
                             # acceleration bounded so the demand cannot chatter

# asymmetric disturbance on load 2 (predeclared, identical across methods): a
# temporary drag increase, i.e. a scaling of the load's whole resistance wrench.
T_DIST = 22.0
DUR_DIST = 12.0
DRAG_MULTIPLIER = 2.6

# --------------------------------------------------------------------------- #
# tolerances (predeclared, used by the hypotheses H1-H6)
# --------------------------------------------------------------------------- #
TAU_B = 1e-8        # ||B z - y_star||_inf
TAU_V = 1e-8        # |V(z) - V_star|
TAU_GAMMA = 1e-6    # Gamma_E at the WISE point
TAU_EIG = 1e-9      # geometric-transfer slack  lam2_geo >= lam2_bar - TAU_EIG
TAU_Q = 0.10        # mission success: terminal load pose error [m]
TAU_S = 0.05        # mission success: terminal synchronization error (normalized)
TAU_W = 0.35        # mission success: peak wrench residual [N]

# --------------------------------------------------------------------------- #
# information-layer initial condition (identical for every method)
# --------------------------------------------------------------------------- #
A0_SCALE = 0.05     # magnitude of a(0) along the region-split mode
B0_SCALE = 0.05


@dataclass(frozen=True)
class InfoGains:
    m_y: float = M_Y
    theta_1: float = THETA_1
    theta_2: float = THETA_2
    c: float = C_CONS

    @property
    def sigma_dyn(self) -> float:
        return self.theta_1 * self.theta_2 / (self.c * self.m_y)


def alpha_rate(sigma: float, g: InfoGains | None = None) -> float:
    """Certified decay rate alpha(sigma) of Prop. "stability" / the time-varying corollary.

    alpha(sigma) = 0.5 [ m_y + c sigma - sqrt((m_y - c sigma)^2 + 4 theta_1 theta_2) ].
    """
    g = g or InfoGains()
    return 0.5 * (g.m_y + g.c * sigma
                  - np.sqrt((g.m_y - g.c * sigma) ** 2 + 4.0 * g.theta_1 * g.theta_2))


@dataclass
class SeedPerturbation:
    """Per-seed Monte-Carlo variation; the *same* draw is used for every method."""
    mass_scale: np.ndarray = field(default_factory=lambda: np.ones(2))
    damp_scale: np.ndarray = field(default_factory=lambda: np.ones(2))
    pose_offset: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))
    drag_multiplier: float = DRAG_MULTIPLIER
    t_dist: float = T_DIST

    @staticmethod
    def draw(seed: int) -> SeedPerturbation:
        rng = np.random.default_rng(20260808 + int(seed))
        return SeedPerturbation(
            mass_scale=rng.uniform(0.90, 1.10, size=2),
            damp_scale=rng.uniform(0.85, 1.15, size=2),
            pose_offset=np.column_stack([
                rng.uniform(-0.06, 0.06, size=2),        # x, inside the tube
                rng.uniform(-0.06, 0.06, size=2),        # y, inside the tube
                rng.uniform(-0.02, 0.02, size=2),        # theta
            ]),
            drag_multiplier=float(rng.uniform(2.3, 2.9)),
            t_dist=float(rng.uniform(20.0, 24.0)),
        )
