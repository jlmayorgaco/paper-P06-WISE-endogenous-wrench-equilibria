"""Projected primal--dual dynamics for the shared-constraint WISE game.

    x_i^+ = proj_simplex[ x_i + alpha ( grad_x Phi(x) + pi * d lambda_2/d x_i ) ]
    pi^+  = proj_>=0    [ pi + beta ( sigma* - lambda_2(L(x)) ) ]

A missing wrench direction raises its price ``mu`` (implicit in ``grad Phi``);
missing connectivity raises the relay price ``pi``; robots migrate toward the
actions of highest marginal contribution. The alternative primal maps (Smith /
BNN / replicator) share the same rest points and are exposed for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .equilibrium import WiseProblem, uniform_start


@dataclass
class PDConfig:
    alpha: float = 0.05        # primal step
    beta: float = 0.10         # dual (connectivity price) step
    max_iters: int = 4000
    tol: float = 1e-7          # movement tolerance for early stop
    enforce_wrench: bool = True
    enforce_info: bool = True
    primal_map: str = "projection"   # {"projection","replicator"}
    record_every: int = 1


@dataclass
class SolveResult:
    x: np.ndarray
    pi: float
    method: str
    iters: int
    converged: bool
    history: dict = field(default_factory=dict)

    @property
    def mu(self):  # convenience
        return self.history.get("mu_final")


def project_simplex_rows(V: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row of ``V`` onto the probability simplex."""
    V = np.atleast_2d(np.asarray(V, dtype=float))
    U = np.sort(V, axis=1)[:, ::-1]
    css = np.cumsum(U, axis=1) - 1.0
    idx = np.arange(1, V.shape[1] + 1)
    rho = np.count_nonzero(U - css / idx > 0, axis=1)
    theta = css[np.arange(V.shape[0]), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0.0)


# backward-compatible alias
project_simplex = project_simplex_rows


def solve(
    problem: WiseProblem,
    x0: np.ndarray | None = None,
    config: PDConfig | None = None,
) -> SolveResult:
    """Integrate the projected primal--dual flow to a WISE fixed point."""
    cfg = config or PDConfig()
    x = uniform_start(problem) if x0 is None else np.array(x0, dtype=float)
    pi = 0.0
    is_long = problem.meta.get("is_long")
    hist = {k: [] for k in
            ("lambda2", "margin", "resid", "welfare", "pi", "relay",
             "relay_long", "relay_short")}
    converged = False
    it = 0
    for it in range(1, cfg.max_iters + 1):
        grad = problem.grad_potential(x) if cfg.enforce_wrench else _cost_only_grad(problem, x)
        if cfg.enforce_info:
            g_relay = problem.lambda2_relay_gradient(x)          # (N,)
            grad[:, problem.relay_index] += pi * g_relay
        if cfg.primal_map == "replicator":
            x_new = _replicator_step(x, grad, cfg.alpha)
        else:
            x_new = project_simplex_rows(x + cfg.alpha * grad)

        lam = problem.lambda2(x_new)
        if cfg.enforce_info:
            pi = max(0.0, pi + cfg.beta * (problem.sigma - lam))

        if it % cfg.record_every == 0:
            resid = float(np.max(problem.wrench_price(x_new) / max(problem.q, 1e-9)))
            hist["lambda2"].append(lam)
            hist["margin"].append(lam - problem.sigma)
            hist["resid"].append(resid)
            hist["welfare"].append(problem.potential(x_new))
            hist["pi"].append(pi)
            r = problem.relay_shares(x_new)
            hist["relay"].append(float(np.sum(r)))
            if is_long is not None:
                hist["relay_long"].append(float(r[is_long].sum()))
                hist["relay_short"].append(float(r[~is_long].sum()))

        move = float(np.max(np.abs(x_new - x)))
        x = x_new
        if move < cfg.tol and it > 50:
            converged = True
            break

    hist = {k: np.asarray(v) for k, v in hist.items()}
    hist["mu_final"] = problem.wrench_price(x)
    return SolveResult(x=x, pi=pi, method="wise_primal_dual", iters=it,
                       converged=converged, history=hist)


def _cost_only_grad(problem: WiseProblem, x: np.ndarray) -> np.ndarray:
    """Gradient with the wrench term dropped (connectivity-only baseline)."""
    grad = np.zeros((problem.N, problem.A))
    grad -= problem.g
    grad -= problem.eps * np.asarray(x)
    return grad


def _replicator_step(x: np.ndarray, grad: np.ndarray, alpha: float) -> np.ndarray:
    """Multiplicative-weights / replicator primal map (ablation)."""
    x = np.asarray(x, dtype=float)
    w = x * np.exp(alpha * grad)
    return w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
