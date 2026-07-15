# Submission Checklist (Definition of Done) — LARS 2026

## Mathematics
- [ ] Variables have consistent units (relaxed counts).
- [ ] Slots and relay sites have occupancy constraints.
- [ ] Wrench: exact zonotopic lifting, or declared conservative.
- [ ] "Equilibrium" formalized via a VI.
- [ ] Dimension uses the minimal active face.
- [ ] (T−1)(K−1) appears only as a special case.
- [ ] WISE uses σ_req > σ_dyn.
- [ ] Existence certificate is an explicit SDP.
- [ ] Stability restricted to the aggregate/estimator subsystem.
- [ ] Necessity restricted to the isotropic modal case.
- [ ] No exponential-stability claim for a point on a continuous fiber.
- [ ] No incorrect KKT; SDP duals μ, ν, ρ, Z.
- [ ] Endogeneity price defined over the connectivity-feasible set (not identically 0).
- [ ] Informal rounding guarantee removed.
- [ ] Prop. 3 removed (or fully proved).

## Algorithm
- [ ] Centralized SDP certificate vs decentralized realization separated.
- [ ] Fiedler/aggregate estimator described.
- [ ] No global-convergence promise.
- [ ] Metrics computed after integerization.
- [ ] No hidden global information in the "decentralized" flow.

## Experiments
- [ ] Direct fiber figure (V flat, λ₂ varies).
- [ ] Threshold evaluated by a sweep.
- [ ] Phase diagram overlays the SDP boundary.
- [ ] Wrench-infeasibility vs communication-infeasibility distinguished.
- [ ] ≥30 paired seeds.
- [ ] Reasonable baselines compared.
- [ ] Physical simulation shows load + robots.
- [ ] L_geo vs L̄ reported.
- [ ] Data and figures regenerate automatically.

## Writing
- [ ] No "confirms all four".
- [ ] No "matches the oracle" without qualifier.
- [ ] No "exact" except exact math equivalence.
- [ ] No repeated slogans; "executes and computes itself" ≤ once.
- [ ] Each paragraph has one function.
- [ ] Abstract and conclusion not structural copies.
- [ ] Every claim has ledger evidence.
- [ ] Meta paragraph headings removed (self-referential gap / key observation).

## Format
- [ ] ≤ 6 pages incl. references.
- [ ] English throughout.
- [ ] IEEE conference format.
- [ ] Fonts embedded (TrueType/Type1, no Type 3).
- [ ] References verified.
- [ ] No compile errors; no undefined refs; no visible overfull boxes.
- [ ] PDF metadata carries no previous title.
