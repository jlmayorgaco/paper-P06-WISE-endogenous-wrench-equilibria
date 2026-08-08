"""E-Robot: closed-loop robot-level validation of the WISE assignment.

Scope (declared): a planar, auditable NumPy/SciPy simulation of two rigid loads
carried by the frozen flagship team, with the paper's inner-zonotope actuation
model, the physical communication graph built from real robot positions, and the
reduced information layer of Prop. "stability" driven by that time-varying graph.

It is **not**: a full nonlinear robot-load stability theorem, a hardware result, an
actuator-level proof, a distributed assignment-convergence result, or a claim of
robustness to communication phenomena that are not modelled.
"""

from __future__ import annotations

__all__ = ["config", "scenario", "assignments", "load_dynamics", "wrench_allocator",
           "communication", "information_layer", "controllers", "simulator", "metrics"]
