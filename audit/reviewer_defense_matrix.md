# Three-Persona Adversarial Review + Defense Matrix — TASK 19

Reviewed state: commit `4340b32` (after TASKs 0,1,2,6). Mandate: **try to reject**.
Honesty note: several blockers are **still open** because the tasks that fix them (3,4,5,8,9,10,12,14)
are not yet implemented. This document does not pretend to close them; the last column says which
task closes each. Counterexamples below are computed, not asserted.

---

## REVIEWER A — Population games / variational equilibria

**Recommendation: reject (major revision).**

1. **Productive degeneracy / rank formula (BLOCKER).** Theorem 1 claims the optimal fiber has
   dimension `(T−1)(K_act−1)`. This is the transportation-polytope dimension and holds only for a
   *type-blind column-sum* aggregate. The paper's aggregate is the **weighted** support map
   `s_{kℓ}=Σ W_{τkhℓ} x`. Counterexample (computed): with `T=2,K=2` and heterogeneous weights the
   fiber has dimension **0**, not `(T−1)(K−1)=1`. The theorem is false as stated.
2. **Objective / potential-game equivalence (BLOCKER).** The "productive value" uses the hinge
   `−½‖[d−s]_+‖²`, which is **flat on the feasible set** `{s≥d}` (value ≡ 0 there), so it is *not*
   strictly concave and cannot yield a unique aggregate. The uniqueness argument is unsupported.
3. **Equilibrium definition.** WISE is defined with `λ₂ ≥ σ*` (non-strict), but the stability result
   (Thm 3) is strict `>`; at equality the Jacobian is singular. The definition and the stability
   claim are inconsistent (needs a margin δ).
4. **Projected dynamics / convergence (BLOCKER).** Prop 3 asserts convergence for "any
   Nash-stationary revision" but the proof sketch (a) treats it as a projected gradient, and (b)
   *assumes* the trajectory stays above threshold — i.e. assumes the property to be proven (forward
   invariance). Not a proof.
5. **Replicator claim.** The text's handling is now correct (replicator is support-trapped), but the
   convergence proposition still overreaches.
6. **Novelty vs. Quijano / Barreiro-Gómez / Martínez-Piazuelo.** Cited, but ref [4] metadata is wrong
   (year/volume/pages), and the novelty paragraph does not do a precise three-way gap statement.

## REVIEWER B — Convex optimization / spectral graph theory

**Recommendation: reject (some items fixed, others blocking).**

1. **Support-function certificate / polytope vs zonotope (RESOLVED, TASK 2).** Prop 1 now correctly
   states polytope⇒polytope, zonotope⇒zonotope, with the triangle counterexample and a conservative
   inner-zonotope certificate. Verified by tests. No action.
2. **λ₂ concavity (RESOLVED).** Correct (pointwise min of linear forms). LMI form is fine.
3. **Eigenvalue multiplicity (MAJOR).** The Fiedler-gradient / `∂λ₂` usage assumes a **simple**
   eigenvalue; at multiplicity the gradient is ill-defined. A background `δ(I−11ᵀ/N)` regulariser is
   used in code but the paper does not state the simplicity assumption for every gradient use.
4. **KKT correctness / two-price corollary (BLOCKER).** Corollary 1 writes `μ = ∇F_k` as the wrench
   *dual multiplier* — conflating a marginal-utility term with a constraint multiplier — and omits
   mass-conservation, nonnegativity, and the full PSD dual. The stationarity equation is incomplete.
5. **Stability threshold "exact/iff" (MAJOR).** Thm 3 says "exact … iff", but the `2×2` `J` is
   *postulated*, not derived from the implemented revision+estimator dynamics; "iff" is only valid for
   that aligned canonical block, not the full system.
6. **Surrogate→geometric mismatch (RESOLVED, TASK 6).** Weyl bridge + margin `δ` now present;
   certificate reports `ε_L` within the Lipschitz bound. No action.
7. **Free/costly/impossible theorem (BLOCKER).** Not present. Current `Δ_endo = V* − max_{E_ss} V ≡ 0`
   (since `E_ss ⊆ argmax V`), so the claimed positive productive loss is identically zero or undefined.

## REVIEWER C — Robotics / experiments / reproducibility

**Recommendation: reject (experimental foundation unsound).**

1. **Finite robot counts / integer recovery (BLOCKER).** Measured: fluid certifies **95.8%**, naive
   integer rounding **4.2%** (integer wrench-feasible only 50%). The headline rate is a fluid number
   that does not survive integer robots. Remark 1's `O(1/N)` guarantee is unquantified and, per this
   measurement, optimistic.
2. **Distributed implementation / estimator (MAJOR).** `λ₂` and the Fiedler vector are computed
   **centrally** (`numpy.linalg.eigh`); "self-computing / distributed" is not supported. `ε_est` in
   the margin is an assumed placeholder.
3. **Physical model (MAJOR).** Second-order unicycle + rigid-load + contact equations are in code but
   not written in the paper; wrench membership is not re-checked at every timestep in the reported
   figures (only in `fig_scene`).
4. **Baseline fairness / stats (MAJOR).** Table I lacks the run count `n`; WISE==oracle is expected
   (same relaxed program) and should be labelled a sanity check, not distributed-optimality proof.
   Bootstrap CIs are computed but the denominator is not stated in captions.
5. **Sample sizes / phase diagram.** The phase map reports empirical success but does not overlay the
   *theoretical* regime boundary (`Λ_E ≥ σ*`); "exactly where predicted" overstates.
6. **Reproducibility (MAJOR).** No single reproduce command; `fig5_threshold.pdf` had no generating
   script (orphan); configs are dead; no run-manifest/seeds/commit metadata; no `results.tex`.
7. **Figure legibility.** Adequate; `fig1_scene` is small. Page 6 is references-heavy with whitespace.

---

## Defense matrix

| # | Criticism | Severity | Location | Answer / evidence | Change still required (task) |
|---|---|---|---|---|---|
| A1 | `(T−1)(K−1)` false for weighted map | **blocker** | Thm 1 | computed counterexample: fiber dim 0 ≠ 1 | **yes — TASK 4** (nullspace rank formula) |
| A2 | hinge objective not strictly concave on `X_f` | **blocker** | §III eq value | hinge ≡0 on `{s≥d}` | **yes — TASK 3** |
| A3 | WISE `≥` vs stability `>` | major | Def 1 / Thm 3 | det J=0 at equality | **yes — TASK 5** (margin δ) |
| A4 | Prop 3 assumes forward invariance | **blocker** | Prop 3 | proof assumes conclusion | **yes — TASK 10** |
| A6 | ref [4] metadata wrong; novelty gap imprecise | major | refs / §I | correct: TAC 69(7) 4427–4442 2024 | **yes — TASK 16** |
| B1 | polytope≠zonotope | resolved | Prop 1 | corrected + tests | no (TASK 2) |
| B3 | eigenvalue multiplicity | major | §II–IV | simplicity assumed, not stated | **yes — TASK 9/10** |
| B4 | incomplete KKT / `μ=∇F` | **blocker** | Cor 1 | multipliers omitted | **yes — TASK 9** |
| B5 | "exact iff" for full system | major | Thm 3 | 2×2 postulated | **yes — TASK 7** |
| B6 | surrogate→geometric | resolved | Lem (bridge) | Weyl + δ + cert | no (TASK 6) |
| B7 | `Δ_endo ≡ 0` | **blocker** | §III eq gap | `E_ss⊆argmax V` | **yes — TASK 5** (trichotomy) |
| C1 | fluid 95.8% vs integer 4.2% | **blocker** | §V, Rem 1 | measured | **yes — TASK 3 + 12** |
| C2 | centralized λ₂ / "distributed" | major | §IV–V | `eigh` centralized | **yes — TASK 11** |
| C3 | physical eqns not in paper | major | §V | in code only | **yes — TASK 14** |
| C4 | Table lacks `n`; WISE==oracle | major | Tab I | caption gap | **yes — TASK 14** |
| C6 | reproducibility gaps | major | repo | orphan fig5, dead configs | **yes — TASK 13** |

## Final-gate status (honest)
- Unresolved blockers: **6** (A1, A2, A4, B4, B7, C1) — each mapped to a pending task.
- Known false/unsupported claims still in the PDF: Thm 1 dimension, hinge strict-concavity, `Δ_endo`,
  Cor 1 KKT, Prop 3 convergence, Rem 1 `O(1/N)`.
- Manually entered numerical results: none in figures/table (all generated); prose `ε_L`/`δ` are
  qualitative with values in the certificate.
- Unverified references: ref [4] (and others pending TASK 16 verification).
- Figures reproducible: yes **except** `fig5_threshold.pdf` (orphan — TASK 13).
- Pages: 6. ✅

**Conclusion:** the paper at `4340b32` is **not yet defensible** — 6 blockers remain, all in the
theory/experiment tasks (3,4,5,7,9,10,11,12,13,14,16) not yet executed. TASKs 0,1,2,6 closed the
notation, support-geometry, and geometric-bridge attacks. The adversarial review confirms the
dependency plan: **TASK 3 is the linchpin blocker** (A2, C1 directly; A1/B7/A4 downstream of it).
