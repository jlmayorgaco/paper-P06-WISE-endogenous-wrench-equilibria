# Claim audit — every numeric claim traced to its generator

Each manuscript number must point at the script and artifact that produce it.

## Traced claims

| Claim | Value | Generator | Artifact | Verified |
|---|---|---|---|---|
| Fiber dimension (flagship) | `dim E = 58` | `exp_fiber.py` | `fiber_certificate.json` | yes (residuals `<1e-15`) |
| Fiber dimension (single-aggregate seeds) | `dim E = 59` | `exp_gamma.py` | `gamma_certificate.json` | yes |
| Network-visible dimension | `d_net ∈ [10,12]` | `exp_gamma.py` | `gamma_certificate.json` | **new** |
| Local–global equivalence | `12/12` | `exp_gamma.py` | `gamma_certificate.json` | **new** |
| `Γ_E` at the selector | `≤ 3.4e-6` | `exp_gamma.py` | `gamma_certificate.json` | **new** |
| `Γ_E` at interior point | `[0.45, 1.78]` | `exp_gamma.py` | `gamma_certificate.json` | **new** |
| closed form vs SDP | `2e-12` | `exp_gamma.py` | `gamma_certificate.json` | **new** |
| concavity bound | `12/12` | `exp_gamma.py` | `gamma_certificate.json` | **new** |
| flagship connectivity gain | `λ₂ : 0 → 0.41` | `exp_flagship.py` | `flagship.json` | yes |
| direct rounding feasible | `653/900` | `exp_rounding.py` | `rounding.csv` | yes |
| post-repair feasible | `895/900` | `exp_cluster_stats.py` | `cluster_stats.json` | yes — **reproduced exactly** |
| cluster bootstrap CI | `[98.3, 100.0]` | `exp_cluster_stats.py` | `cluster_stats.json` | **new** |
| per-instance median / min | `100%` / `83.3%` | `exp_cluster_stats.py` | `cluster_per_instance.csv` | **new** |
| instances with ≥1 success | `30/30` | `exp_cluster_stats.py` | `cluster_stats.json` | yes |
| zero-cost recoveries | `83/895` (9.3%) | `exp_rounding.py` | `rounding.csv` | yes |
| Prop. 3 certified | **`0/30`** | `integer_recovery.py` | `integer_recovery.json` | **newly surfaced** |
| direct re-certified | `29/30` | `integer_recovery.py` | `integer_recovery.json` | **newly surfaced** |
| `χ` median | `191` | `integer_recovery.py:69` | `integer_recovery.json` | yes — **was undefined in the text** |
| oracle false positives | `0/23` | `exp_oracle.py` | `oracle_benchmark.csv` | yes |
| oracle max gap | `≤ 0.005` (`<0.06%`) | `exp_oracle.py` | `oracle_benchmark.csv` | yes |
| adversarial `N=6` FP | `13/50` | `exp_oracle.py` | `oracle_benchmark.csv` | yes |
| Loewner dominance | `100%` of `10 000` | `exp_mismatch.py` | `mismatch.json` | yes (construction check) |
| dual vs finite difference | `r = 0.996` | `exp_dual.py` | `dual_check.csv` | yes |

## Claims corrected

| Was | Problem | Now |
|---|---|---|
| "double precision, so solver tolerance does not enter the guarantee" | Re-evaluating in floating point removes solver-*status* dependence, not roundoff. Not a proof. | "independently re-evaluated … accepted only when residuals, slacks, wrench feasibility and spectral margin clear prescribed tolerances … not exact-arithmetic certification" |
| "every instance is seeded and reproduces exactly" | Unverifiable without artifacts; and the *environment* was in fact unrecorded (cvxpy absent). | "All experiments use fixed recorded seeds; code, manifests, solver settings and reproduction scripts accompany the paper" |
| Wilson `[98.7, 99.8]` on `895/900` | 900 draws = 30 instances × 30 nested orders. Not independent Bernoulli. | Cluster bootstrap over instances `[98.3, 100.0]`; pooled Wilson retained only as labelled contrast |
| "capping the drop at `(1−η_max)V*`" | Says the wrong thing (a cap on the drop is not the constraint imposed). | "requiring `V(ẑ) ≥ (1−η_max)V*`" |
| `Δw = 0` in the flagship box | `w` is the *demanded* wrench (fixed); the realized wrench was never shown equal. | "`ΔBz = ΔV = 0`, both wrench-feasible, `Δλ₂ > 0`" |
| `χ ≈ 190` | Symbol never defined in the paper. | Defined as `χ = ‖ẑ−z*‖₂/r`, plus the `0/30` vs `29/30` conservatism it implies |
| `E*` | Used in the tie-break, never defined. | `E* := argmax_{z∈E} λ₂(L̄(z)) = {z ∈ E : Γ_E(z) = 0}` |
| `osc(λ₂)` | Undefined in the Tikhonov bound. | `osc_{X_f}(λ₂) = Λ_X − min_{X_f} λ₂` |
| "The dual is a genuine price" | Rhetorical. | "the local sensitivity of optimal productive loss" |
| "9.3% zero-cost" read as a capacity | It is a property of the fast heuristic. | Explicitly attributed to the heuristic, with the oracle bounding the attainable optimum separately |

## Open reproducibility items

1. **Version lockfile.** cvxpy/clarabel were absent; reinstalled to the declared 1.9/0.11.
   Pin before freeze.
2. **Public repository + commit hash.** The paper now promises accompanying artifacts;
   that promise must be discharged or the sentence removed.
3. **References [10]/[11].** `calvo2025heterogeneous` lacks vol/pages;
   `santos2024backbone` is cited as *IEEE Latin America Transactions* 2024 but the
   public record is arXiv:2409.16851 marked "Submitted to". Must verify against
   authoritative metadata or downgrade to preprint. **Not yet done.**
4. **Tolerance table.** The paper now refers to "prescribed tolerances"; the actual
   values (`τ_B = 1e-8`, wrench `0.05`, solver `1e-7`, `EIG_TOL = 1e-7`) must be
   collected into one place.
