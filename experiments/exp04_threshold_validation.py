"""E4 — Self-defeat threshold validation.

Fine sweep around the Nash-seeking boundary ``c * lambda_2 * m_F = theta^2``.
Measure allocation variance, limit-cycle amplitude, consensus error,
convergence time, and certification probability, showing a clean transition
between ``lambda_2 > sigma*`` and ``lambda_2 < sigma*``.
"""

from __future__ import annotations

from _common import load_config, parse_args, write_summary


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    print(f"[exp04] threshold validation — config={args.config}")
    print(f"[exp04] boundary sweep: {cfg.get('threshold', {})}")
    if args.dry_run:
        return
    raise NotImplementedError("threshold validation — Day 3")
    write_summary("exp04_threshold_validation", {})  # noqa: F821  (Day 3)


if __name__ == "__main__":
    main()
