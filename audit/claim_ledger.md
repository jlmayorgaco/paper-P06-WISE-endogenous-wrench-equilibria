# Claim Ledger — LARS 2026 revision

Baseline: tag `pre-rigorous-revision` (commit with reference/σ/(T−1)(K−1)/wording fixes).
Mandate: every verifiable statement in title, abstract, contributions, captions,
results, and conclusion must map to formal or experimental evidence. Any claim whose
stated scope exceeds its evidence is a **blocker** and must be restricted, replaced,
or removed before the SDP/stability core is rewritten.

Status legend: **retain** (scope matches evidence) · **restrict** (true only under
narrower conditions than stated) · **replace** (needs a different, correct result) ·
**remove** (no defensible version at this scope) · **conditional** (correct given
stated hypotheses).

## Title / scope

| ID | Claim | Type | Assumptions | Evidence | Status | Action (task) |
|----|-------|------|-------------|----------|--------|--------|
| T01 | "Self-Sustaining Population Equilibria over Decision-Induced Robot Networks" | framing | — | whole paper | restrict | too abstract for a robotics venue; retitle around transport + selection (T3) |

## Abstract

| ID | Claim | Type | Assumptions | Evidence | Status | Action (task) |
|----|-------|------|-------------|----------|--------|--------|
| A01 | Distributed seeking assumes graph available independently of the profile computed | context | — | intro cites | retain | — |
| A02 | Productively optimal aggregate is unique | theorem | φ strictly concave on B·X_f | Lem. (unique aggregate) | conditional/retain | keep hypothesis explicit |
| A03 | Optimal composition is degenerate — a positive-dimensional fiber | theorem | nontrivial productive nullspace on the active face | Thm. degeneracy | **blocker/restrict** | positive-dim is not generic for a weighted B; state "under the support/degeneracy condition" or for the transport instance (T8) |
| A04 | WISE keeps λ₂ above σ_dyn (WISE requires σ_req=σ_dyn+δ) | definition | Weyl bridge, margin δ | Def. WISE, Lem. bridge | retain | keep σ_req>σ_dyn split explicit (T11) |
| A05 | Composition degeneracy with a productive-nullspace dimension formula | theorem | active-face tangent | Thm. degeneracy | retain | reground on minimal active face (T8) |
| A06 | free/costly/impossible trichotomy | theorem | Weierstrass on Λ_E ≤ Λ_X | Thm. trichotomy | retain→reframe | reconcile with Δ_endo over connectivity-feasible set (T16) |
| A07 | robust local self-computation certificate c·m_F·λ₂>θ² from coupled dynamics; worst-case tight in canonical aligned block | theorem | 2×2 aligned block postulated | Thm. stability | **blocker/replace** | derive from vector aggregate/estimator subsystem; sufficient in general, necessary only isotropic modal (T13) |
| A08 | safe selector whose productive loss vanishes as ε↓0 | theorem | Tikhonov/Γ | Prop. selection | restrict | make lexicographic the main selector; ε only with explicit bounds (T14) |
| A09 | Closed-loop simulation … supports the theory; relay roles emerge by comparative advantage | experiment | one scenario family | figs 2–4, table | **blocker/restrict** | per-result calibrated language; ensure all types can relay so roles are not predetermined (T24); simulation "is consistent with", not "confirms" (T28) |
| A10 | verified integer recovery | experiment | randomised rounding on N=12 | integer_recovery.json (100%) | restrict | report post-integerization metrics; drop any recovery guarantee (T19) |

## Contributions (current bullets)

| ID | Claim | Type | Assumptions | Evidence | Status | Action (task) |
|----|-------|------|-------------|----------|--------|--------|
| K01 | Composition degeneracy: aggregate unique, fiber positive-dimensional via nullspace rank formula | theorem | active face | Thm. degeneracy | retain | already de-generalized from (T−1)(K−1); reground on active face (T8) |
| K02 | Self-sustaining existence: WISE exists iff Λ_E ≥ σ_req | theorem | max over fiber attained | Thm. existence | replace-form | recast as explicit SDP optimal value ≥ σ_req (T12) |
| K03 | Exact self-defeat threshold, derived | theorem | 2×2 block | Thm. stability | **blocker/replace** | vector subsystem + Schur (T13) |
| K04 | Infinitesimal-reward selection + distributed algorithm, validated | theorem+exp | Tikhonov; supervisory Fiedler | Prop. selection, Prop. conv | **blocker/split** | lexicographic main (T14); delete convergence Prop (T17); "decentralized realization" (T17,T18) |

## Section-level claims

| ID | Claim | Location | Status | Action (task) |
|----|-------|----------|--------|--------|
| S01 | Support-function/zonotope certificate (Prop. membership) | §II | restrict/replace-as-lemma | exact zonotopic lifting; conservative-contact caveat; demote from headline (T6) |
| S02 | x_{τa} population mass, integer via rounding; no O(1/N) claim | §II, Rem. | retain | already no guarantee; move to post-integerization evaluation (T5,T19) |
| S03 | λ₂∘L̄ concave ⇒ convex information set (Lem. convex) | §II | retain | — |
| S04 | Weyl geometric transfer under margin δ (Lem. bridge) | §II | retain | keep same-node/order asserts (T10) |
| S05 | Comparative-advantage neutral exchange raises λ₂ (Prop. cycle) | §III | conditional/retain | requires simple λ₂; ensure all types can relay (T24) |
| S06 | Safe connectivity-selection flow forward-invariant (Prop. conv) | §IV | **blocker/remove** | delete; replace by SDP certificate + decentralized realization (T17) |
| S07 | Replicator avoided (support-trapped) | §IV | retain | trim to one clause (P6) |
| S08 | WISE == centralized oracle (ablation) | §V | restrict | same relaxed program; label numerical sanity check; paired stats (T23) |
| S09 | Phase diagram matches predicted regimes | §V | **blocker/restrict** | overlay SDP boundary Λ⋆=σ_req; TP/TN/FP/FN table (T22) |

## Blocker summary (must clear before/with the theory rewrite)

1. **A07/K03** — full-system "exact iff" stability → vector subsystem + Schur (T13).
2. **S06/K04** — Prop. 3 convergence assumes its own connectivity premise → remove (T17).
3. **A03** — unconditional "positive-dimensional" → restrict to active-face condition (T8).
4. **K02** — existence stated abstractly → explicit computable SDP (T12).
5. **A09/S09** — "supports all four" + phase-diagram match → per-result calibrated
   language and SDP-boundary overlay (T22, T28).
6. **A10/S02** — integer recovery → post-integerization evaluation only (T19).
7. **A08/K04** — ε-selection headline → lexicographic main result (T14).

## Recheck note (already honest in current baseline — do not "re-fix")

- No `O(1/N)` guarantee is asserted (Rem. integer).
- Finite ε is explicitly stated NOT to leave the productive optimum unmoved.
- Fiedler eigenpair is already declared supervisory/centralized.
- Conclusion says "supports", not "confirms".
These are already correct; the revision restructures scope, it does not re-patch them.
