# Theory audit — every formal statement, verdict, and action

Verdicts assigned by reading each statement against its proof and its implementation.
`P0` = must fix before submission. Status as of Checkpoint A.

## Verdict table

| # | Statement | Verdict | Finding | Status |
|---|---|---|---|---|
| L1 | Wrench zonotope membership | correct, mis-titled | Exact for the **inner zonotope**, not for contact physics. "relaxation" is the wrong word: an inner approximation is not a relaxation. | **fixed** — retitled "Exact lifted representation of the inner-zonotope actuation model"; added explicit quasi-static + rigid/bilateral attachment assumption; separated `c_τ` from `(F_τ,G_τ,A_kh)` |
| L2 | Concavity of λ₂ | correct | Standard, correctly applied. Now load-bearing for Thm. `networkability`. | kept |
| L3 | Geometric transfer (Loewner) | algebraically correct | Assumption 1(iv) does most of the work; the 10 000-trial check verifies a dominance imposed by construction. | **fixed** — labelled a construction check in E1; added robust ε_geo variant via Weyl |
| T1 | Composition degeneracy, `dim E` | correct | Dimension is that of the minimal face; correctly stated at a relint point. | kept |
| P1 | Fiber ⇔ monotonicity loss | correct, edge case | `λ_min(P_F^T B^T H B P_F)` undefined when `dim T_F = 0`. | **fixed** — singleton-face hypothesis stated explicitly |
| P2 | Free-connectivity, simple λ₂ | correct | Projection criterion sound. | promoted into Thm. `networkability` |
| P2 | Free-connectivity, **repeated λ₂** | **P0 — dimensionally invalid** | Text defined `U₂` as a basis of the eigenspace of `Qᵀ L̄ Q ∈ R^{(N-1)×(N-1)}`, then wrote `U₂ᵀ DL̄[d] U₂` with `DL̄[d] ∈ R^{N×N}`. Product undefined. **The code was right** (`exp_multiplicity.py:71` uses `V[:,1:3]`, eigenvectors of `L̄` in `R^N`); only the prose was wrong. No re-run needed. | **fixed** — `U₂ ∈ R^{N×m}` redefined as an orthonormal basis of the λ₂-eigenspace of `L̄` itself (⊂ 1⊥) |
| T2 | WISE existence/selection SDP | correct | `E*` used in the tie-break without definition. | **fixed** — `E* := argmax_{z∈E} λ₂(L̄(z))`, characterised as `{Γ_E = 0}` |
| R1 | Stability threshold `σ_dyn` | **P0 — underdefined** | Formula `ϑ₁ϑ₂/(c·m_F)` given with no dynamical system; `c` was not even in the declared gain list. | **fixed** — promoted to Prop. with the explicit 2×2 modal system, Routh–Hurwitz iff, and decay rate `α(λ)` |
| T3 | Price `P(σ)`, trichotomy | correct | Convexity/monotonicity and the dual sensitivity are sound. | kept; "genuine price" → "local sensitivity" |
| P3 | Margin-based integer preservation | correct but near-vacuous | Mixes the smallest row slack with the largest row norm; scale-dependent. Empirically certifies **0/30** recoveries. | **fixed** — reframed as worst-case certificate; conservatism quantified in E2; row-normalised `r_w` noted (changes nothing: no wrench facet active) |
| — | VE wording | **P0 — internal contradiction** | Def. 2 said WISE is "not an equilibrium of the productive game alone" while Thm. 2 proves it *is* a variational equilibrium. | **fixed** — every WISE is a productive VE; connectivity is a lexicographic refinement of `SOL(X_f,F)`, with the two-level condition `0 ∈ ∂[−λ₂] + N_E` |

## New result (Phase 2)

**Theorem (Optimal-fiber networkability).** With
`Γ_E(z̄) = max{ Dλ₂(z̄)[d] : d ∈ T_E(z̄), ‖d‖₂ ≤ 1 }`:

```
Γ_E(z̄) > 0   ⟺   ∃ z' ∈ E : λ₂(L̄(z')) > λ₂(L̄(z̄))
Γ_E(z̄) = 0   ⟺   z̄ ∈ argmax_{z∈E} λ₂(L̄(z))    (i.e. λ₂(L̄(z̄)) = Λ_E)
Λ_E − λ₂(L̄(z̄)) ≤ Γ_E(z̄)·‖z*_WISE − z̄‖₂
```

**Why it upgrades the paper.** Prop. 2 previously answered "does a local improving
direction exist?". The theorem answers "is there *any* free connectivity left on the
entire fiber?" — a local test that certifies global optimality. `Γ_E` becomes a
*quantity*, not a yes/no.

**Proof mechanics.** `f = λ₂∘L̄` is finite and concave (Lemma 2), so `Df(z̄)[·]` exists,
is positively homogeneous, and overestimates increments: `f(z') − f(z̄) ≤ Df(z̄)[z'−z̄]`.
- (⟸) an improving `z'` gives `d₀ = z'−z̄ ∈ T_E(z̄)` with `Df[d₀] > 0`; homogeneity normalises.
- (⟹) `E` **polyhedral** ⟹ `T_E(z̄) = cone(E − z̄)` is already closed, so `z̄ + t₀d ∈ E`
  for some `t₀ > 0`; small `t` improves by definition of the directional derivative.

**Hypothesis declared, not hidden:** polyhedrality of `E` is what makes the tangent
cone a cone of genuinely feasible directions. It holds here (`E = X_f ∩ {Bz = y*}`,
an intersection of a polytope with an affine set). For a general closed convex `E` the
tangent cone is only the *closure* of the feasible-direction cone and the (⟹) direction
needs a continuity argument instead.

**Sign convention fixed in passing.** The old tangent cone was written `G_j d ≤ 0` with
normal cone multiplier `ν ≥ 0`, while the model section defines `X_f = {H_w z ≥ d}`.
Now consistent: `G_j d ≥ 0`, `ν ≤ 0`.

## Degeneracy vs *usable* degeneracy

`d_net = rank(DL̄|_{N_p})` with `dim E ≥ d_net ≥ 1[Γ_E > 0]` and `d_net ≤ N` (only the
`N` relay coordinates enter `L̄`). Measured: `dim E = 59` but `d_net ∈ [10,12]`.
Productive degeneracy is necessary but far from sufficient — the honest version of the
old "58 dimensions of opportunity" framing.

## Control result (Phase 4)

Modal system `[ȧ; ḃ] = A_λ [a; b]`, `A_λ = [[−m_F, ϑ₁], [ϑ₂, −cλ]]`.
`tr A_λ < 0` always; `det A_λ = c·m_F·λ − ϑ₁ϑ₂`, so Hurwitz **iff** `λ > σ_dyn`.
Decay rate `α(λ) = ½(m_F + cλ − √((m_F−cλ)² + 4ϑ₁ϑ₂))`.

**Monotonicity verified analytically** (the plan required this before claiming it):
`α'(λ) = (c/2)[1 − (cλ−m_F)/√((m_F−cλ)²+4ϑ₁ϑ₂)] > 0`, since the radicand strictly
exceeds `(cλ−m_F)²`. Also `α(σ_dyn) = 0`, consistent with `det = 0`. Hence
`λ₂ ≥ σ_req > σ_dyn` gives a *uniform* guaranteed rate `α(σ_req) > 0`, and Lemma 3
transfers it to every admissible pose.

Scope declared: reduced information-layer model. Not robot, load or actuator dynamics.

## Open (later checkpoints)

- Integer networkability capacity `Λ_E^Z` over the integer productive-optimal fiber, and
  the gap `g_net^Z = Λ_E − Λ_E^Z` (Phase 9).
- Fiedler-separation exact solver — **must reproduce the exhaustive oracle before use**
  (Phase 10). If it fails that test it stays out of the paper.
- Repeated-λ₂ instances are not exercised by `exp_gamma` (all 12 seeds gave `m = 1`);
  `exp_multiplicity.py` covers that path separately. A joint test is still owed.
