"""WISE — Wrench-and-Information Self-Sustaining Equilibria.

Core library for the P07/LARS-2026 paper. The public surface is intentionally
small; each submodule owns one object from the equilibrium definition:

    wrench_tensor    directional capacity tensor W[i,k,h,l] and feasibility
    endogenous_graph affine Laplacian L(x), Fiedler value/gradient, sigma*
    equilibrium      WISE feasible set, tensor-aggregative potential Phi
    primal_dual      projected primal-dual dynamics (x, mu, pi)
    baselines        scalar / wrench-only / connectivity-only / WISE / oracle
    metrics          certified service rate and secondary metrics
"""

__version__ = "0.1.0"

__all__ = [
    "wrench_tensor",
    "endogenous_graph",
    "equilibrium",
    "primal_dual",
    "baselines",
    "metrics",
]
