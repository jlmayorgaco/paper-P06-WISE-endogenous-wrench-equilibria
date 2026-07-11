# WISE — Reconstruction Plan (hard reset onto Documento IV)

**Status:** authoritative plan for the LARS 2026 paper. Supersedes the divergent draft in `paper/`.
**Grounding source:** `docs/sources/Heterogeneidad_Grafo_Endogeno_Resultado_Frontera.pdf` ("Documento IV"),
which already contains the rigorous version of every result. The current `paper/` draft drifted into a
tensor/GVE/two-price formulation that the review correctly rejects; we revert to Documento IV.

## 1. One-sentence contribution (keep this, delete everything that isn't it)

> A distributed population equilibrium can realize a spatial composition that **destroys the very
> communication graph its own payoff estimation needs**; heterogeneity of communication range is the
> resource that selects the equilibria that can sustain themselves.

WISE = **W**rench/task-feasible **I**nformation-**S**elf-sustaining **E**quilibrium. For LARS we lead with
the *information* self-sustaining result (Documento IV); the wrench/capacity layer stays as the
feasibility set `y(x)=y*`, not as a separate "tensor" apparatus.

## 2. Result map: review's asks ⇄ Documento IV (all already proven)

| Paper result (new)                         | Source in Documento IV            | Notes |
|--------------------------------------------|-----------------------------------|-------|
| Def. WISE (self-sustaining)                | Def 4.1  `λ₂(Ḡ(x*)) ≥ σ*`         | `σ*=ϑ²/(cμ)`, μ=αC |
| Thm 1 — composition degeneracy             | Prop 3.2  polytope dim `(T−1)(M_act−1)` | replaces trivial potential Thm |
| Water-filling aggregate is unique          | Prop 3.1  `y*_k=[v_k−λ]_+/(αC)`   | context |
| Thm 2 — existence of self-sustaining eq.   | Thm 4.1  `max_{E}λ₂ ≥ σ*`         | coverage Assumption 1 = "iff Λ*≥σ*" |
| Refinement (select within Nash variety)    | Obs 4.1  (max-λ₂ selection)       | "operational refinement" framing |
| Thm 3a — stability with explicit rate      | Thm 4.2  ρ_loop bound             | 3-timescale singular perturbation |
| Thm 3b — unreachability (2-cluster)        | Thm 4.3  cobweb `e⁺=(1−βg)e−βgε`, `βg>2`→2-cycle | **derived** — replaces fake Fig 3(b) |
| PoE index + closed form                    | Def 5.1 + Ex 5.1 (eq 8–9)         | additive-loss framing also fine |
| Range-quantum annuls PoE, `ν*=m_min(0)/ζ`  | Thm 5.2                           | replaces tautological Prop 2 |
| Health dispersion raises σ*                | Prop 5.1                          | optional (space) |
| Emergent division of labor (comp. advantage)| Thm 5.3  `χ_τ`                    | replaces "roles emerge" hand-wave |

## 3. Delete from current `paper/` draft (per review)

- The four-index "tensor" as a central object (rename/remove; it is a support-capability map, not a tensor).
- Trivial "exact tensor-aggregative potential" Theorem 1.
- The two-price Corollary (KKT stationarity is incomplete: missing `∇F_k` and `γψ'(λ₂)∇λ₂`; μ reused for
  deficit-gradient and KKT multiplier; scalar π where an LMI multiplier `Z⪰0` is the honest object).
- PoE-as-`N/(N−m_min)` tautology; Fig 3(b) "limit-cycle amplitude diverges at ρ=1" (unsupported).
- The sentence "Replacing (12a) by Smith/BNN/replicator leaves the rest-point set unchanged" — false in
  general (replicator is support-trapped; Smith/BNN are Nash-stationary). See Documento IV / long ms.
- All of Table I and every reported % until produced by a real run. "Experimental campaign" wording.

## 4. Honest simulation (ONE physical instance, three runs) — replaces the grid classification

Canonical two-cluster relay instance (Documento IV Ex 5.1 + §6 predictions):
- Tasks A,C of value v at distance d; optional relay site B (value 0) at d/2; short-range `r_s<d/2`,
  long-range `r_ℓ≥d/2`. Real positions; distance/`φ`-weighted graph (eq 2); Smith revision (β) driven by
  **consensus-estimated** occupancy on `Ḡ(x(t))` (gain c); optional 2nd-order unicycle motion.
- **Run A (degenerate):** all long-range in one cluster → `λ₂<σ*` → estimator churn / 2-cycle (Thm 4.3).
- **Run B (self-sustaining):** same aggregate `y*`, composition with a bridge type → `λ₂≥σ*`, churn→0.
- **Run C (infeasible):** torque/demand so high all mass must serve → no composition keeps both service
  and connectivity. Yields the physical-vs-informational feasibility frontier.
- Falsifiable numbers to reproduce (Documento IV §6): churn transition across `cλ₂μ=ϑ²`; 2-cycle amplitude
  ∝ βg; welfare saturates at `ν*=m_min(0)/ζ`; relay identity concentrates on long types (hypergeometric
  test vs random) — this is the *measured* division of labor, not an assigned role.

## 5. Figures (exactly three)

1. Physical instance: load, tasks A/B/C, short/long robots, induced graph, a self-defeating vs a
   self-sustaining equilibrium.
2. Nash variety `E` (polytope) with `E_SS = E ∩ {λ₂≥σ*}` — the refinement, visually.
3. Theory vs dynamics: above threshold estimation-error/churn→0; below, persistent; phase boundary
   `cλ₂μ=ϑ²` overlaid on the simulation sweep.

## 6. Code reuse in existing `src/wise_mr/` (continue, don't recreate)

- Keep & align to Documento IV: `endogenous_graph.py` (eq 2, Fiedler gradient — Lemma 2.1),
  `equilibrium.py` (water-filling Prop 3.1, degeneracy polytope), `scenarios.py` (2-cluster instance),
  `metrics.py` (λ₂, churn, PoE).
- Rewrite/repurpose: `wrench_tensor.py` → demote to a feasibility check (`y(x)=y*`) or drop for LARS;
  `primal_dual.py` → the honest object is the **closed-loop** Smith-revision + consensus-estimation
  dynamics (Thm 4.2/4.3), not a generic monotone VI convergence claim.
- Tests to add/keep as gates: Fiedler-gradient vs finite differences; degeneracy dimension; σ* threshold
  reproduces the cobweb `βg>2` transition; existence (max-λ₂ ≥ σ*) on the instance; PoE closed form vs
  numeric; range-quantum ν*.

## 7. Tier ladder (auto-fallback, no fabrication)

- **Tier A:** Def WISE + Thm1 (degeneracy) + Thm2 (existence) + Thm3 (stability + 2-cluster unreachability)
  + PoE range-quantum + emergent-roles, with the 3-run simulation and 3 figures.
- **Tier B:** drop range-quantum/roles if space/time short; keep degeneracy + WISE + existence +
  stability/unreachability + the simulation. Still a complete, honest paper.
- **Tier C:** degeneracy + WISE definition + existence + the 2-cluster unreachability contraejemplo +
  one figure + the sweep. Minimal but self-contained.

## 8. Positioning (adopt, don't hide, the closest prior art)

State plainly: we build on the distributed population-game / GNE machinery of Barreiro-Gómez, Quijano,
Martínez-Piazuelo, and Nash-seeking with estimators (Gadjov–Pavel; Yi–Pavel; Koshal–Nedić–Shanbhag),
and on connectivity control (Zavlanos–Pappas); our extension is the **self-referential** case where the
decision generates the graph that machinery presupposes. (Documento IV §7 already lists these.)

## 9. Blockers (need user input)

- **Affiliation:** current `paper/` cover says *Universidad Militar Nueva Granada*. The TFM affiliation is
  **VIU**. Do NOT use the host institution as affiliation. Default = VIU (MROB) unless a real UMNG
  relationship is confirmed with the tutor. **← decision required.**
- **Author metadata:** name(s), affiliation line, email for IEEEtran front matter. Default = Jorge Luis
  Mayorga Taborda, VIU (MROB), flagged placeholder email until provided.
- **The 1000-line master brief `.md`** is still not in the repo; this plan is grounded in Documento IV +
  the review instead. Drop the brief in if you want its exact proof-obligation wording.
