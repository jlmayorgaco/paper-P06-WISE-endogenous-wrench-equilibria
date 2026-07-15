# Experiment Audit — LARS 2026 revision

Each experiment must target exactly one of the three core results and use calibrated language.

| Exp | Purpose | Ties to | Current gap | Target (task) |
|-----|---------|---------|-------------|--------|
| E-fiber | constant V, varying λ₂ over the fiber | fiber geometry | not isolated | sweep α; assert ptp(V)≤1e-8, ptp(λ₂)>vis (T20) |
| E-threshold | measured vs modal exponential rate around σ_dyn | stability | only 2 compositions | sweep λ₂∈[0.5,1.5]σ_dyn; slope(log‖e‖) vs λ_max(J) (T21) |
| E-phase | wrench-infeasible / WISE-infeasible / feasible | existence SDP | no SDP boundary overlaid | solve SDP per cell; overlay Λ⋆=σ_req; TP/TN/FP/FN (T22) |
| E-methods | baselines comparison | selection | oracle==WISE trivial | 7 methods, ≥30 paired seeds, 95% CI (T23) |
| E-roles | relay ∝ marginal spectral benefit / opportunity cost | comparative advantage | short-range γ=0 predetermines it | γ_short<γ_med<γ_long all >0 (T24) |
| E-physical | rigid-load transport closed loop | validation | curves only | snapshots, load pose error, wrench margin, L_geo vs L̄ (T25) |

## Reproducibility
- All numbers from CSV; one `make reproduce` (T31): tests → experiments → CSV → figures → paper → page check.
- Determinism: fixed seeds; record solver + version, tolerances, gains, δ components.

## Current reproducible artifacts (baseline)
- generated/nullspace_certificate.json (dim E=4/72), integer_recovery.json (rand 100%),
  geometric_bridge_certificate.json (Weyl holds). Tests: 24 passed, 5 skipped.
