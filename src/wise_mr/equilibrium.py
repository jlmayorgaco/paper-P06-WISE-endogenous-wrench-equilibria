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
    q: float = 20.0               # wrench-penalty gain
    eps: float = 1e-2             # strong-concavity regulariser
    meta: dict = field(default_factory=dict)

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

    # ---- potential and gradients -------------------------------------------
    def potential(self, x: np.ndarray) -> float:
        s, d = self.capacity(x), self.demand()
        resid = wt.wrench_residual(s, d)
        wrench = -0.5 * self.q * float(np.sum(resid**2))
        cost = -float(np.sum(self.g * x))
        reg = -0.5 * self.eps * float(np.sum(x**2))
        return wrench + cost + reg

    def grad_potential(self, x: np.ndarray) -> np.ndarray:
        """Gradient of ``Phi`` (excluding the price term) w.r.t. ``x``; shape ``(N, A)``."""
        mu = self.wrench_price(x)                        # (M, P)
        # wrench gradient on slot actions: sum_l mu_kl W[i,k,h,l]
        gw = np.einsum("kl,ikhl->ikh", mu, self.W)       # (N, M, H)
        grad = np.zeros((self.N, self.A))
        grad[:, : self.M * self.H] = gw.reshape(self.N, self.M * self.H)
        grad -= self.g
        grad -= self.eps * np.asarray(x)
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
