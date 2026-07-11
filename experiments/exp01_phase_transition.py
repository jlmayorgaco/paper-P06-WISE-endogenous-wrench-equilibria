"""E1 — WISE phase diagram.

Sweep the long-range fraction ``nu``, torque demand ``tau_d`` and inter-cluster
separation ``d``; classify each instance into one of

    0  no physical equilibrium          (infeasible wrench)
    1  physical-only equilibrium        (wrench-feasible but info self-defeating)
    2  WISE equilibrium                 (both constraints met)
    3  WISE with margin                 (comfortable slack on both)

Headline figure: the (nu, tau_d) plane coloured by regime.
"""

from __future__ import annotations

from _common import load_config, parse_args, write_summary


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    print(f"[exp01] phase-transition sweep — config={args.config}")
    print(f"[exp01] grid: {cfg.get('sweep', {})}")
    if args.dry_run:
        return
    raise NotImplementedError("phase-transition sweep — Day 3")
    write_summary("exp01_phase_transition", {})  # noqa: F821  (Day 3)


if __name__ == "__main__":
    main()
