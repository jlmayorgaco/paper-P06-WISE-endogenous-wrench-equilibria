"""WISE equilibrium: problem container, tensor-aggregative potential, feasibility.

The strategic game maximises the tensor-aggregative potential

    Phi(x) = sum_k F_k(s_k(x)) - sum_{i,a} g_ia x_ia - (eps/2)||x||^2

over the product of per-robot simplices, subject to the shared information
constraint ``lambda_2(L(x)) >= sigma*`` (dual price ``pi``). The soft wrench term
``F_k(s_k) = -q/2 ||[d_k - s_k]_+||^2`` induces the directional wrench price
``mu_k = q [d_k - s_k]_+``; the connectivity constraint induces the information
price ``pi`` (matrix dual ``Z* = pi v v^T``). Both are the certification prices.

Decision layout: each robot ``i`` has ``A = M*H + 2`` actions --- one per
(load, slot) pair, then ``relay`` and ``idle``. Only slot actions carry wrench;
only the relay action perturbs the Laplacian.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import endogenous_graph as eg
from . import wrench_tensor as wt


@dataclass
class WiseProblem:
    """One WISE instance."""

    N: int
    M: int
    H: int
    P: int
    W: np.ndarray                 # (N, M, H, P) directional wrench tensor
    directions: np.ndarray        # (M, P, 3) facet normals eta_kl
    w_dem: np.ndarray             # (M, 3) demanded wrench per load
    g: np.ndarray                 # (N, A) action cost
    base_laplacian: np.ndarray    # (V, V) always-on links
    relay_laplacians: np.ndarray  # (N, V, V) B_i added when robot i relays
    sigma: float                  # information threshold sigma*
    q: float = 20.0               # wrench hard-constraint dual step scale
    eps: float = 1e-2             # selection regulariser (refinement, not productive)
    cap: np.ndarray | None = None       # (N,) per-robot served-capacity weight c_i
    task_value: np.ndarray | None = None  # (M,) v_k  (productive value slope)
    alpha: float = 1.0            # congestion; productive Hessian H = alpha I (>0)
    meta: dict = field(default_factory=dict)

    def _cap(self) -> np.ndarray:
        return np.ones(self.N) if self.cap is None else np.asarray(self.cap, float)

    def _value(self) -> np.ndarray:
        # default productive target y* = v/alpha per load (fallback 6 units)
        return (self.alpha * 6.0 * np.ones(self.M) if self.task_value is None
                else np.asarray(self.task_value, float))

    # ---- action bookkeeping ------------------------------------------------
    @property
    def A(self) -> int:
        return self.M * self.H + 2

    @property
    def relay_index(self) -> int:
        return self.M * self.H

    @property
    def idle_index(self) -> int:
        return self.M * self.H + 1

    def slots_view(self, x: np.ndarray) -> np.ndarray:
        """Extract the ``(N, M, H)`` slot shares from the ``(N, A)`` decision."""
        return np.asarray(x)[:, : self.M * self.H].reshape(self.N, self.M, self.H)

    def relay_shares(self, x: np.ndarray) -> np.ndarray:
        """Relay commitment ``(N,)``."""
        return np.asarray(x)[:, self.relay_index]

    # ---- physics / information ---------------------------------------------
    def laplacian(self, x: np.ndarray) -> np.ndarray:
        r = self.relay_shares(x)                        # (N,)
        return self.base_laplacian + np.einsum("i,ijk->jk", r, self.relay_laplacians)

    def capacity(self, x: np.ndarray) -> np.ndarray:
        return wt.directional_capacity(self.W, self.slots_view(x))    # (M, P)

    def demand(self) -> np.ndarray:
        return wt.demand_projection(self.directions, self.w_dem)      # (M, P)

    def wrench_price(self, x: np.ndarray) -> np.ndarray:
        """Directional wrench price ``mu_k = q [d_k - s_k]_+`` of shape ``(M, P)``."""
        return self.q * wt.wrench_residual(self.capacity(x), self.demand())

    def lambda2(self, x: np.ndarray) -> float:
        return eg.fiedler_value(self.laplacian(x))

    def info_margin(self, x: np.ndarray) -> float:
        return self.lambda2(x) - self.sigma

    # ---- productive value B x, and gradients -------------------------------
    def served_capacity(self, x: np.ndarray) -> np.ndarray:
        """Productive aggregate ``y_k = B x = sum_{i,h} c_i x_{ikh}``; shape ``(M,)``.

        Distinct from the directional wrench aggregate ``s = H_w x``: ``B`` is the
        served-capacity map whose nullspace makes the composition degenerate.
        """
        slots = self.slots_view(x)                       # (N, M, H)
        return np.einsum("i,imh->m", self._cap(), slots)

    def productive_value(self, x: np.ndarray) -> float:
        """Genuinely strictly-concave value ``V(x)=phi(B x)``,
        ``phi(y)=sum_k (v_k y_k - 0.5 alpha y_k^2)`` (Hessian ``-alpha I prec 0``).
        Wrench feasibility ``s>=d`` is a *hard constraint* (dual mu), not here.
        """
        y = self.served_capacity(x)
        v = self._value()
        return float(np.sum(v * y - 0.5 * self.alpha * y**2))

    def potential(self, x: np.ndarray) -> float:
        """Solver objective = productive value + selection refinement (cost, reg)."""
        cost = -float(np.sum(self.g * x))
        reg = -0.5 * self.eps * float(np.sum(x**2))
        return self.productive_value(x) + cost + reg

    def grad_potential(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the solver objective (productive + refinement); ``(N, A)``."""
        y = self.served_capacity(x)
        dphi = self._value() - self.alpha * y            # (M,) = grad phi wrt y_k
        cap = self._cap()                                # (N,)
        gslot = np.einsum("i,m->im", cap, dphi)          # (N, M) contribution per (i,k)
        grad = np.zeros((self.N, self.A))
        # broadcast dphi over the H slots of each load
        grad[:, : self.M * self.H] = np.repeat(gslot, self.H, axis=1)
        grad -= self.g
        grad -= self.eps * np.asarray(x)
        return grad

    def wrench_matrix(self) -> np.ndarray:
        """Linear support map ``H_w`` with ``s(x)=H_w vec(x)``; ``(M P, N A)``.
        Distinct symbol from the objective Hessian ``H=alpha I``.
        """
        Hw = np.zeros((self.M * self.P, self.N * self.A))
        for i in range(self.N):
            for k in range(self.M):
                for h in range(self.H):
                    col = i * self.A + (k * self.H + h)
                    Hw[k * self.P:(k + 1) * self.P, col] = self.W[i, k, h, :]
        return Hw

    def wrench_grad(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Gradient contribution of the wrench dual ``mu`` (M,P): ``sum_l mu_kl W``."""
        gw = np.einsum("kl,ikhl->ikh", mu, self.W)       # (N, M, H)
        grad = np.zeros((self.N, self.A))
        grad[:, : self.M * self.H] = gw.reshape(self.N, self.M * self.H)
        return grad

    def lambda2_relay_gradient(self, x: np.ndarray) -> np.ndarray:
        """``d lambda_2 / d x_{i,relay} = v^T B_i v`` at the Fiedler vector; shape ``(N,)``."""
        _, v = eg.fiedler_pair(self.laplacian(x))
        return np.einsum("j,ijk,k->i", v, self.relay_laplacians, v)   # v^T B_i v per robot i

    # ---- feasibility -------------------------------------------------------
    def wrench_feasible(self, x: np.ndarray, tol: float = 0.05) -> np.ndarray:
        return wt.is_wrench_feasible(self.capacity(x), self.demand(), tol)

    def max_residual(self, x: np.ndarray) -> float:
        """Worst-direction wrench deficit ``max_{k,l} [d_kl - s_kl]_+``."""
        return float(np.max(self.wrench_price(x) / self.q))

    def is_wise(self, x: np.ndarray, wrench_tol: float = 0.05, info_tol: float = 1e-2) -> bool:
        return bool(np.all(self.wrench_feasible(x, wrench_tol))) and \
            self.info_margin(x) >= -info_tol


def uniform_start(problem: WiseProblem) -> np.ndarray:
    """Uniform interior start ``x_ia = 1/A``."""
    return np.full((problem.N, problem.A), 1.0 / problem.A)
