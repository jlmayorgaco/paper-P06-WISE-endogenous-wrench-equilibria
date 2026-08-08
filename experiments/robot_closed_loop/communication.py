"""Robot poses and the *physical* communication graph L_geo(q(t)).

Robot positions come from the assignment and the current load poses:

* a **lifting** robot is rigidly (bilaterally) attached, ``p_i = p_k + R(theta_k) r_kh``
  -- the paper's standing attachment assumption;
* a **relay** robot is regulated to its site by ``p_dot = -k_r (p - p_site)`` and must
  stay inside its declared tube during the certified operational phase;
* an **idle** robot holds its home position.

``L_geo`` uses the *same* weight law as the tube-infimum surrogate ``Lbar``
(``scenario.lgeo`` / ``scenario.lbar``), so the Loewner comparison of
Lemma "bridge" is a statement about distances only: every realized distance is at
most the tube supremum, hence every realized weight is at least the tube infimum.

Nothing here ever substitutes one Laplacian for the other: ``lambda2_bar`` is always
computed from ``Lbar(z_hat)`` and ``lambda2_geo`` always from ``L_geo(q(t))``.
"""

from __future__ import annotations

import numpy as np

from wise_mr.endogenous_graph import complement_basis

from . import scenario as S

Q_BASIS = complement_basis(S.N_ROBOTS)          # (N, N-1) orthonormal basis of 1^perp
K_RELAY = 1.5                                    # relay site regulator gain [1/s]
K_DEPLOY = 1.5                                   # deployment approach gain [1/s]


def target_positions(assignment: tuple, load_poses: np.ndarray) -> np.ndarray:
    """Where every robot *should* be for the given assignment and load poses."""
    p = S.HOMES.copy()
    for i, a in enumerate(assignment):
        if a[0] == "lift":
            p[i] = S.slot_world_from_pose(a[1], a[2], load_poses[a[1]])
        elif a[0] == "relay":
            p[i] = S.RELAY_SITES[a[1]]
    return p


def step_positions(pos: np.ndarray, assignment: tuple, load_poses: np.ndarray,
                   dt: float, attached: bool) -> np.ndarray:
    """Advance robot positions one step.

    ``attached=False`` (deployment) drives every robot to its target with the
    first-order law; ``attached=True`` (operational) makes lifters exactly rigid and
    keeps the relay/idle regulators running.
    """
    tgt = target_positions(assignment, load_poses)
    out = pos.copy()
    for i, a in enumerate(assignment):
        if attached and a[0] == "lift":
            out[i] = tgt[i]
        else:
            k = K_RELAY if a[0] == "relay" else K_DEPLOY
            out[i] = pos[i] + dt * (-k) * (pos[i] - tgt[i])
    return out


def tube_violation(pos: np.ndarray, assignment: tuple, load_poses: np.ndarray) -> np.ndarray:
    """Per-robot distance outside its declared admissible tube (0 when inside)."""
    out = np.zeros(S.N_ROBOTS)
    for i, a in enumerate(assignment):
        pts, rho = S.tube(i, a)
        d = float(np.min(np.linalg.norm(pts - pos[i], axis=1)))
        out[i] = max(0.0, d - rho)
    return out


def graph_snapshot(pos: np.ndarray, assignment: tuple, lbar: np.ndarray) -> dict:
    """lambda_2 of the physical graph and its margins against the surrogate."""
    L = S.lgeo(pos, np.array([a[0] == "relay" for a in assignment], bool))
    lam_geo = S.lambda2(L)
    lam_bar = S.lambda2(lbar)
    Ld = Q_BASIS.T @ (L - lbar) @ Q_BASIS
    return {"L_geo": L, "lambda2_geo": lam_geo, "lambda2_bar": lam_bar,
            "transfer_margin": lam_geo - lam_bar,
            "loewner_min_eig": float(np.linalg.eigvalsh(0.5 * (Ld + Ld.T))[0])}


def reduced(L: np.ndarray) -> np.ndarray:
    """``Q^T L Q`` on ``1^perp`` (the matrix that drives the information layer)."""
    return Q_BASIS.T @ L @ Q_BASIS


def sigma_min_reduced(L: np.ndarray) -> float:
    R = reduced(L)
    return float(np.linalg.eigvalsh(0.5 * (R + R.T))[0])
