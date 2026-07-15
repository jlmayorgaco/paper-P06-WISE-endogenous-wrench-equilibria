# Theorem Audit — LARS 2026 revision

Target result set (reduced scope):

1. **Equilibrium fiber geometry** (VI + active-face dimension). T7, T8.
2. **WISE existence & selection as an SDP** (exact certificate). T12, T14.
3. **Aggregate/estimator spectral stability** (sufficient; necessary in isotropic modal case). T13.

Supporting lemmas kept: unique aggregate; λ₂∘L̄ concavity; Weyl bridge; exact
zonotopic wrench lifting (T6).

## Per-theorem plan

| Result | Current form | Problem | Target form | Task |
|--------|--------------|---------|-------------|------|
| Unique aggregate | Lemma | fine under strict concavity | keep, state hypothesis | — |
| Degeneracy | dim E = n − rank[A;B;G_I] | needs relint/active-face wording; (T−1)(K−1) only special case | minimal-active-face tangent formula + 2×2 example | T8, T9 |
| Variational equilibrium | absent (max only) | "equilibrium" not formalized | VI(X_f, −∇V); equivalence to argmax under concavity | T7 |
| WISE existence | Λ_E ≥ σ_req (abstract) | not computable as stated | two-stage: y⋆ then SDP max t s.t. QᵀL̄Q⪰tI; exists iff Λ⋆≥σ_req | T12 |
| Stability | 2×2 J "exact iff" | postulated, full-system claim | vector [[−M,Θ],[Θᵀ,−cL⊥]]; Schur ⇒ c·m_y·λ₂>θ² sufficient; necessary iff M=m_yI,Θ=θI | T13 |
| Selection | ε-reward (Prop.) | headline overreach | lexicographic argmax_{E} λ₂ main; ε only with osc bounds | T14 |
| Prices | (removed corollary) | KKT was incomplete | SDP Lagrangian: μ (wrench), ν (aggregate), ρ (budgets), Z⪰0 (spectral) | T15 |
| Endogeneity price | trichotomy | reconcile | Δ_endo = V⋆ − max_{X_f∩C_δ} V; ≤4 lines, discussion | T16 |
| Convergence | Prop. 3 | assumes conclusion | delete; SDP oracle + decentralized realization (no global claim) | T17 |

## Correctness notes to preserve
- Weyl for ordered eigenvalues: both L̄, L_geo share the constant null vector ⇒ λ₂ ordering aligns.
- Schur target: with J symmetric (M,L⊥ symmetric), Hurwitz ⇔ J≺0 ⇔ cL⊥ ≻ ΘᵀM⁻¹Θ ⇒ c·m_y·λ₂>θ² sufficient.
- Zonotopic lifting is exact for U_τ^in = {G_τ u : ‖u‖_∞≤1}; box-constrained generators ξ with |ξ|≤z·1.
