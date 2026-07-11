"""E2 — Constraint ablation.

Compare scalar_capacity / wrench_only / connectivity_only / wise_primal_dual /
centralized_wise_oracle on paired seeds.  Primary metric: certified service
rate (a load counts only if rho_k >= 0 AND lambda_2 >= sigma*).  Secondary:
residual wrench, connectivity margin, churn, welfare, runtime, messages.
Paired bootstrap CIs with Holm correction across methods.
"""

from __future__ import annotations

from _common import load_config, parse_args, write_summary


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    print(f"[exp02] constraint ablation — config={args.config}")
    print(f"[exp02] methods: {cfg.get('methods', [])}")
    if args.dry_run:
        return
    raise NotImplementedError("constraint ablation — Day 3")
    write_summary("exp02_constraint_ablation", {})  # noqa: F821  (Day 3)


if __name__ == "__main__":
    main()
