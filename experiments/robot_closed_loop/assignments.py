"""Baselines PROD / HARD / WISE / SCALAR / RANDOM-FIBER by *exhaustive* enumeration.

With ``N = 6`` robots and ``|A| = 9`` actions the whole integer assignment space has
``9^6 = 531,441`` maps, so every baseline below is an exact optimum over the integer
feasible set -- no relaxation, no rounding, no recovery heuristic. Every returned
assignment is therefore integer *by construction* and is re-certified directly
against budgets, occupancy, the inner-zonotope wrench LP and ``lambda_2(Lbar)``.

Objectives and tie-breaks are declared here, once, and are never method-specific:

* **PROD**   ``argmax V``; tie-break: minimum deployment cost, then lexicographic.
* **HARD**   ``argmax V`` s.t. ``lambda_2(Lbar) >= sigma_req``; tie-break: fewest role
  changes w.r.t. PROD, then minimum deployment cost, then lexicographic.
* **WISE**   ``argmax lambda_2(Lbar)`` over the productive-optimal fiber ``E``;
  same tie-break chain.
* **SCALAR** ``argmax V + eps * lambda_2(Lbar)`` over the whole feasible set.
* **RANDOM-FIBER** a uniform draw from ``E`` (supplementary).

``V`` is the paper's productive value ``phi(Bz) = sum_k (v_k y_k - alpha y_k^2 / 2)``
with ``alpha = 1`` and ``v_k = 3`` so that the unconstrained optimum is
``y^star = (3, 3)`` -- exactly the frozen flagship aggregate ``served_aggregate_wise``,
which ``run_flagship.py`` checks against ``generated/flagship.json``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from wise_mr import wrench_tensor as wt

from . import scenario as S

ALPHA_V = 1.0
V_COEFF = np.array([3.0, 3.0])          # v_k; y^star = v/alpha = (3, 3)


def productive_value(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    return float(np.sum(V_COEFF * y - 0.5 * ALPHA_V * y**2))


Y_STAR = V_COEFF / ALPHA_V
V_STAR = productive_value(Y_STAR)


# --------------------------------------------------------------------------- #
# direct re-certification
# --------------------------------------------------------------------------- #
@cache
def _wrench_ok(load: int, contacts: tuple[tuple[int, float], ...]) -> bool:
    """Inner-zonotope LP membership for one load; ``contacts`` = ((slot, F), ...)."""
    if not contacts:
        return False
    maps = np.array([S.SLOT_MAPS[load][h] for h, _ in contacts])
    caps = np.array([f for _, f in contacts])
    return bool(wt.certify_membership_lp(caps, maps, S.W_DEM_CERT[load],
                                         kappa=S.KAPPA, m_sides=S.M_SIDES))


def occupancy_ok(assignment: tuple) -> bool:
    used = [a for a in assignment if a[0] != "idle"]
    return len(used) == len(set(used))


def served_aggregate(assignment: tuple) -> np.ndarray:
    y = np.zeros(S.M_LOADS)
    for i, a in enumerate(assignment):
        if a[0] == "lift":
            y[a[1]] += S.CAP[i]
    return y


def contacts_of(assignment: tuple, load: int) -> tuple[tuple[int, float], ...]:
    return tuple(sorted((a[2], float(S.FORCE[i]))
                        for i, a in enumerate(assignment) if a[0] == "lift" and a[1] == load))


def wrench_feasible(assignment: tuple) -> bool:
    return all(_wrench_ok(k, contacts_of(assignment, k)) for k in range(S.M_LOADS))


@dataclass
class Certificate:
    """Direct re-certification record of one integer assignment."""
    name: str
    assignment: tuple
    integral: bool
    occupancy_ok: bool
    wrench_ok: dict
    y: np.ndarray
    V: float
    aggregate_error: float
    value_error: float
    lambda2_bar: float
    lambda2_nominal: float
    margin_vs_sigma_req: float
    deployment_cost: float
    role_changes_vs_prod: int
    gamma_fiber_integer: float = float("nan")
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["assignment"] = [list(map(str, a)) for a in self.assignment]
        d["y"] = self.y.tolist()
        return d


# --------------------------------------------------------------------------- #
# enumeration
# --------------------------------------------------------------------------- #
def enumerate_feasible() -> list[tuple]:
    """All integer assignments passing occupancy *and* wrench feasibility."""
    out = []
    for assignment in itertools.product(S.ACTIONS, repeat=S.N_ROBOTS):
        if not occupancy_ok(assignment):
            continue
        if not wrench_feasible(assignment):
            continue
        out.append(assignment)
    return out


def _tie_break(assignment: tuple, ref: tuple | None) -> tuple:
    """Declared, method-independent tie-break: fewest role changes w.r.t. ``ref``,
    then least deployment cost, then lexicographic action index. Smaller is better."""
    changes = 0 if ref is None else sum(a != b for a, b in zip(assignment, ref, strict=True))
    return (changes, round(S.assignment_cost(assignment), 12),
            tuple(S.A_INDEX[a] for a in assignment))


@dataclass
class Baselines:
    feasible: list[tuple]
    fiber: list[tuple]
    lam_bar: dict
    prod: tuple
    hard: tuple | None
    wise: tuple
    scalar: dict
    random_fiber: tuple
    lam_star_fiber: float
    lam_star_feasible: float


def build(sigma_req: float, eps_grid=(1e-3, 1e-2, 1e-1, 1.0), rf_seed: int = 0) -> Baselines:
    feasible = enumerate_feasible()
    lam_bar = {a: S.lambda2(S.lbar(a)) for a in feasible}
    values = {a: productive_value(served_aggregate(a)) for a in feasible}
    v_max = max(values.values())
    fiber = [a for a in feasible if abs(values[a] - v_max) <= 1e-12]

    prod = min(fiber, key=lambda a: _tie_break(a, None))
    wise = min(fiber, key=lambda a: (-round(lam_bar[a], 12),) + _tie_break(a, prod))
    hard_pool = [a for a in fiber if lam_bar[a] >= sigma_req]
    hard = min(hard_pool, key=lambda a: _tie_break(a, prod)) if hard_pool else None

    scalar = {}
    for eps in eps_grid:
        scalar[eps] = min(feasible,
                          key=lambda a: (-round(values[a] + eps * lam_bar[a], 12),)
                          + _tie_break(a, prod))

    rng = np.random.default_rng(1234 + rf_seed)
    random_fiber = fiber[int(rng.integers(len(fiber)))]

    return Baselines(feasible=feasible, fiber=fiber, lam_bar=lam_bar, prod=prod,
                     hard=hard, wise=wise, scalar=scalar, random_fiber=random_fiber,
                     lam_star_fiber=max(lam_bar[a] for a in fiber),
                     lam_star_feasible=max(lam_bar[a] for a in feasible))


def certify(name: str, assignment: tuple, bl: Baselines, sigma_req: float) -> Certificate:
    y = served_aggregate(assignment)
    V = productive_value(y)
    lam_b = S.lambda2(S.lbar(assignment))
    return Certificate(
        name=name,
        assignment=assignment,
        integral=True,                                  # by construction of the enumeration
        occupancy_ok=occupancy_ok(assignment),
        wrench_ok={f"load{k+1}": bool(_wrench_ok(k, contacts_of(assignment, k)))
                   for k in range(S.M_LOADS)},
        y=y,
        V=V,
        aggregate_error=float(np.max(np.abs(y - Y_STAR))),
        value_error=float(abs(V - V_STAR)),
        lambda2_bar=lam_b,
        lambda2_nominal=S.lambda2(S.nominal_pose_laplacian(assignment)),
        margin_vs_sigma_req=float(lam_b - sigma_req),
        deployment_cost=S.assignment_cost(assignment),
        role_changes_vs_prod=sum(a != b for a, b in zip(assignment, bl.prod, strict=True)),
        gamma_fiber_integer=float(bl.lam_star_fiber - lam_b),
    )


def roles(assignment: tuple) -> dict:
    """Human-readable role map, e.g. {'L1': 'relay@C', 'S1': 'lift(1,h1)'}."""
    out = {}
    for i, a in enumerate(assignment):
        if a[0] == "lift":
            out[S.NAMES[i]] = f"lift(load{a[1]+1},h{a[2]})"
        elif a[0] == "relay":
            out[S.NAMES[i]] = f"relay@{a[1]}"
        else:
            out[S.NAMES[i]] = "idle"
    return out


def lifters(assignment: tuple, load: int) -> list[int]:
    return [i for i, a in enumerate(assignment) if a[0] == "lift" and a[1] == load]


def relay_mask(assignment: tuple) -> np.ndarray:
    return np.array([a[0] == "relay" for a in assignment], dtype=bool)
