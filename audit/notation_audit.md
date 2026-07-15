# Notation Audit — LARS 2026 revision

Goal: every symbol has one meaning, consistent units, and one printed form.

## Threshold family (T11)
- `σ_dyn = θ²/(c·m_y)` — dynamic stability boundary of the aggregate/estimator subsystem.
- `δ = ε_L + ε_eig + ε_num + ε_disc` — robust margin (mismatch, eigenvalue estimation, numerics, discretization).
- `σ_req = σ_dyn + δ` — design requirement used by WISE and the SDP.
- Rule: WISE membership and existence use `σ_req` (≥); stability uses `σ_dyn` (>). Never the same symbol for both.

## Decision variable (T5)
- Replace normalized capacity fractions `x_{τa}∈[0,1]` by relaxed robot counts `z_{τa}≥0`.
- Type budget: `Σ_a z_{τa} = n_τ`, total `N = Σ_τ n_τ`.
- Occupancy: `Σ_τ z_{τ,kh} ≤ 1` per slot, `Σ_τ z_{τ,r} ≤ 1` per relay site.
- `z` is a continuous relaxation of a count — not a probability, not a force.

## Aggregate / maps
- `y = Bz`, `y_k = Σ_{τ,h} c_τ z_{τkh}` (served capacity). Units: capacity·count.
- `H_w z ≥ d` wrench feasibility; keep `H_w` (support map) distinct from objective Hessian `H`.
- Laplacian `L̄(z) = L_0 + Σ_{τ,r} z_{τr} γ_{τr} L_r` (T10). Same node set/order as `L_geo(q)`.

## Pending sweep
Run `rg -n "sigma|σ|threshold|lambda_2|λ_2|x_\{|x_{"` and classify each hit as
dynamic-threshold / design-requirement / empirical-margin / existence-condition /
decision-variable. Record exceptions here.
