"""E3 — Role emergence.

Robots carry two independent heterogeneities: mechanical capacity ``c_i`` and
communication range ``r_i``.  With no hand-assigned roles, measure which robots
become relays vs. torque-slot occupants, the mean capacity/range of relays, and
the opportunity cost.  Prediction: robots with the highest
information-contribution-to-physical-cost ratio specialize as relays.
"""

from __future__ import annotations

from _common import load_config, parse_args, write_summary


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    print(f"[exp03] role emergence — config={args.config}")
    print(f"[exp03] heterogeneity: {cfg.get('heterogeneity', {})}")
    if args.dry_run:
        return
    raise NotImplementedError("role emergence — Day 3")
    write_summary("exp03_role_emergence", {})  # noqa: F821  (Day 3)


if __name__ == "__main__":
    main()
