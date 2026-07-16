# WISE — mathematical respecification (blueprint before the next PDF rewrite)

Purpose: fix the three open bridges the review flags — (a) the decision graph vs. the
physical graph, (b) the service aggregate vs. the delivered wrench, (c) relaxed vs. integer
feasibility — and reorganize the theory into four definitive theorems, **before** editing
the manuscript. Per the standing decision the distributed framing is kept: WISE-PD remains a
heuristic realization (its convergence theorem is listed as the requirement to promote it,
not assumed here).

Target contribution set (three):
- **C1 — Geometry of zero-cost connectivity** (Thm 1 + Thm 2).
- **C2 — Self-sustainability threshold and price** (Thm 3).
- **C3 — Relaxed certificate with finite-team recovery** (SDP + Thm 4).

---

## 0. Notation ledger (one symbol, one object)

| Symbol | Object | Was colliding with |
|---|---|---|
| `K` | number of loads | (was `M`) |
| `H` | contact slots per load | — |
| `T` | number of types | — |
| `N` | number of robots | — |
| `V` | number of **backbone graph nodes** | (was forced `=N`) |
| `C` | mass/budget conservation map, `Cz=n` | (was `A`) |
| `A_{kh}` | contact-to-wrench map | (kept; distinct letterform in text) |
| `B` | served-capacity map, `y=Bz` | — |
| `R_φ = αI` | productive Hessian | (was `H`) |
| `u_{τkh}` | zonotope generator coeffs | (was `ξ`) |
| `ξ, η` | optimizer/estimator errors (Prop. small-gain) | (freed) |
| `s ∈ [0,1]` | fiber coordinate | (was `α`) |
| `h` | neutral direction in `N_p` | (was `d`) |
| `d` | wrench RHS (demand) | — |
| `p_ℓ` | long-range fraction | (was `ν`) |
| `ν` | aggregate dual | — |
| `π` | spectral (connectivity) price (scalar) | (freed) |
| `P_M` | permutation of a robot→site matching `M` | (was `π`) |
| `Y ⪰ 0` | LMI dual; `Z = Y/\tr Y` normalized direction | — |

Units: forces/torques in the wrench block; `V` in utility units; `λ2` in conductance units
(so `σ_req`, `c·m_y` scale with the conductance scale — state the weight function
`w_{ij}(q)` and the consensus gain `c` explicitly).

---

## 1. Site-backbone graph (replaces the ambiguous `V=N`)

**Nodes are sites, not robots.** Let `G_s=(\mathcal V_s,\mathcal E_s)`, `|\mathcal V_s|=V`,
with nodes = {load regions, fixed stations, information endpoints, relay sites}. The
**decision-induced backbone Laplacian** is
```
  L̄(z) = L_base + Σ_{τ,r} γ_{τr} z_{τr} L_r ∈ R^{V×V},   L_base, L_r ⪰ 0,  γ_{τr} ≥ 0,
```
affine in `z` because **only relay occupancies gate predefined edge Laplacians** — not a
general pairwise geometric graph, whose weights `w_{ij}(z) ~ Σ_{a,b} x_{ia}x_{jb}w_{ij}^{ab}`
would be bilinear. Each `L̄(z)` is a symmetric Laplacian (`L̄1=0 ⪯ L̄`).

**Physical comparison via a quotient, not a same-size permutation.** After integer recovery,
a matching `M` assigns robots to occupied sites; let `S(q) ∈ R^{n_robots×V}` be the
site-aggregation (each robot summed into its site, idle robots into a station node). Define
```
  L_quot(q) = S(q)^⊤ L_geo(q) S(q) ∈ R^{V×V}.
```
Transfer (replaces Lemma "bridge"): `‖L_quot(q) − L̄(z)‖_2 ≤ ε_L`. This removes ghost
nodes, ambiguous idle, and the nonexistent robot↔node bijection for fractional `z`.

---

## 2. Service–wrench model (unifies the productive aggregate with delivered wrench)

Introduce a **physical service level** `y_k ∈ [0,1]` per load. The delivered wrench scales
with service:
```
  Σ_{τ,h} A_{kh} G_τ u_{τkh} = y_k · w_k^{dem},   -z_{τkh} 1 ≤ u_{τkh} ≤ z_{τkh} 1,
  y ≤ Bz   (served-capacity cap),     y_k ∈ [0,1].
```
Productive value on the **physical** service vector:
```
  V(y) = Σ_k ( v_k y_k − (α_k/2) y_k² ),   strictly concave (R_φ = diag(α_k) ≻ 0).
```
The optimal `z` induces the optimal `y* = argmax V`. The **fiber** is now
```
  E = { z ∈ X_f : B z = y*,  wrench delivered at y* },
```
i.e. *different compositions delivering exactly the same service vector and wrench*. This is
the robotics-meaningful statement (before, `y=Bz` was a bare capacity aggregate with no
service interpretation).

`X_f = { z ∈ \mathcal X : ∃u, Σ A_{kh}G_τ u_{τkh} = y_k w_k^{dem}, |u|≤z, y=Bz∈[0,1]^K }`;
its projection onto `z` is a polytope `{H_w z ≥ d}` (facets never enumerated — the SDP and
the flow both keep the lifted `(z,u)` and its box duals; **remove any text asserting the
controller knows `H_w`**).

---

## 3. The four theorems

### Theorem 1 — Unique service aggregate and composition fiber
`φ = V∘B` strictly concave in `y` ⇒ `y*` unique. `E` is a nonempty compact convex polytope;
at a relative-interior `z̄` its dimension is
```
  dim E = dim( ker C ∩ ker B ∩ ker G_I ) = n − rank[C;B;G_I],
```
`G_I` the active inequalities (nonnegativity, occupancy, active wrench facets) at `z̄`. The
pseudogradient `F=−∇V` loses strong monotonicity on the composition face exactly on `N_p :=
ker C ∩ ker B ∩ ker G_I` (`μ_F = σ_min²(R_φ^{1/2} B|_{T_F}) = 0 ⟺ dim E > 0`).
*Report an instance with active wrench facets so the fiber is not always the ambient `ker[C;B]`.*

### Theorem 2 — Free spectral-improvement geometry (the new core result)
At `z̄ ∈ relint E`, `λ2` simple with Fiedler `v` (`‖v‖=1, v⊥1`), let
`g_λ = ∇_z λ2(L̄(z̄))`, entries `(g_λ)_{τr} = γ_{τr} v^⊤ L_r v` (0 on non-relay actions).
Then for `h ∈ N_p`, `Dλ2(z̄)[h] = ⟨g_λ,h⟩`, and
```
  max_{h∈N_p, ‖h‖≤1} Dλ2(z̄)[h] = ‖Π_{N_p} g_λ‖,
```
so a **productively neutral connectivity improvement exists ⟺ Π_{N_p} g_λ ≠ 0**, with
optimal direction `h* = Π_{N_p} g_λ / ‖Π_{N_p} g_λ‖`. If `Π_{N_p} g_λ = 0` then
`g_λ ∈ row[C;B;G_I]` — a KKT certificate of no first-order neutral improvement; `λ2` is then
priced entirely by the aggregate/budget/wrench multipliers.

*Multiplicity / boundary generalization.* With `λ2` of multiplicity, use the tangent cone
`T_E(z̄)` and spectral superdifferential `Z_2(z̄)={QZQ^⊤: Z⪰0, \tr Z=1, supp on eigenspace}`:
```
  Δ_λ(z̄) = max_{h∈T_E(z̄), ‖h‖≤1}  min_{Z∈Z_2(z̄)} ⟨Z, Q^⊤ DL̄(z̄)[h] Q⟩,
```
and a first-order productivity-preserving improvement exists ⟺ `Δ_λ(z̄) > 0`.
*Proof:* directional derivative of `λ2` (Fiedler/Danskin); linear functional maximized over
the unit ball of a subspace equals the projected-gradient norm; row-space membership is the
stationarity (KKT) alternative.

### Theorem 3 — Self-sustainability threshold and price
Let `Λ_E = max_{z∈E} λ2(L̄(z))` and `Λ_X = max_{z∈X_f} λ2(L̄(z))` (both attained). Define the
**price of self-sustainability**
```
  P(σ) = V* − max{ V(Bz) : z∈X_f, λ2(L̄(z)) ≥ σ }.
```
Then
```
  P(σ) = 0            for σ ≤ Λ_E        (FREE:      connectivity at no productive cost),
  P(σ) ∈ (0,∞)        for Λ_E < σ ≤ Λ_X  (COSTLY:    connectivity trades productivity),
  P(σ) = +∞           for σ > Λ_X        (IMPOSSIBLE: requirement unattainable).
```
`P` is convex nondecreasing on `(−∞,Λ_X]`; under Slater the spectral multiplier
`π*(σ) ∈ ∂P(σ)` is the marginal productive cost of a unit of required connectivity.
*Proof:* `max V s.t. λ2≥σ` is convex (`V` concave, `λ2` concave); its value `g(σ)` is concave
in the RHS `σ` (perturbation function of a convex program), so `P=V*−g` is convex; the free
regime is `σ≤Λ_E` because `E` already contains a point of connectivity `Λ_E`; KKT gives `π*`.

### Theorem 4 — Robust finite-team recovery (replaces "best-of-30")
Let `z*` be the relaxed WISE selector with margins
```
  m_w = min_j (H_w z* − d)_j ≥ 0,     m_λ = λ2(L̄(z*)) − σ_req ≥ 0,
```
and let `ẑ` be any integer assignment with `‖ẑ − z*‖_1 ≤ Δ`. With
`‖H_w‖_{max} = max_{i,j}|H_{w,ij}|` and `L_{L̄} = max_{τ,r} ‖γ_{τr} L_r‖_2`, linearity and
Weyl's inequality give
```
  min_j (H_w ẑ − d)_j ≥ m_w − ‖H_w‖_{max} Δ,     λ2(L̄(ẑ)) ≥ σ_req + m_λ − L_{L̄} Δ,
  ‖B(ẑ − z*)‖ ≤ ‖B‖_{max} Δ   (aggregate drift).
```
Hence if `m_w > ‖H_w‖_{max} Δ` **and** `m_λ > L_{L̄} Δ`, then `ẑ` is an **integer WISE
assignment**. This is a deterministic design margin (not a claim that any rounding works):
it explains the code margins `WM, IM` and *when* a single rounding is guaranteed safe.

*Ground truth.* For `N ≤ 20`, solve the exact integer program `x_{ia}∈{0,1}` with the LMI
`Q^⊤ L(x) Q ⪰ σ I` (MISDP / spectral branch-and-bound) to report, per instance:
relaxed-feasible, integer-feasible, integrality gap, relaxation false-positive rate,
draws-to-success, runtime. Table column "success" for WISE-SDP → **"relaxed feasible"**;
integer attainment is a separate column.

### (Proposition, kept per framing) — Information-layer small gain
Unchanged in content: on the scalar Fiedler block `ȧ=−m_y a+ϑ1 b, ḃ=ϑ2 a−cλ2 b`, stability
⟺ `c m_y λ2 > ϑ1 ϑ2`; `σ_dyn = ϑ1ϑ2/(c m_y)`. **WISE-PD stays a heuristic**; promoting it to
a contribution requires the connectivity-invariance + convergence theorem below.

---

## 4. What would promote WISE-PD (only if kept as a core claim)
Connectivity-barrier invariance `Dλ2(z)ż ≥ −α(λ2(z)−σ_safe)` ⇒
`λ2(z(0))≥σ_safe ⇒ λ2(z(t))≥σ_safe ∀t`; convergence
`limsup_k dist(z^k, Z*_WISE) ≤ C(τ_eig + h + ε_DAC)` under connected warm start, Slater,
time-scale separation, and step bound; explicit local state/message architecture. Absent
these, WISE-PD is illustrative only (bootstrap disclaimer: WISE *sustains* connectivity, it
does not create global communication from a disconnected start).

---

## 5. Claim ledger (each claim ⇒ theorem + evidence) for the rewrite
| Claim | Theorem | Experiment |
|---|---|---|
| Fiber dimension | T1 | rank ladder + neutral residuals (`fiber_certificate.json`) |
| Free spectral improvement | T2 | `Π_{N_p}g_λ` vs. observed `Δλ2` (new) |
| `Λ_E`, `Λ_X`, three regimes | T3 | free/costly/impossible phase diagram over `(σ, p_ℓ, γ_ℓ/γ_s, τ_d, |R|)` (new) |
| Price `P(σ)` | T3 | productivity–connectivity curve + `π*` slope (new) |
| Integer recovery margin | T4 | predicted margin vs. integer success; exact oracle for `N≤20` (new) |
| Transfer `‖L_quot−L̄‖≤ε_L` | §1 | measured mismatch vs. bound |
| Small-gain threshold | Prop | `λ2` sweep around `σ_dyn` |

## 6. Deferred (not blocking the core rewrite)
Nonholonomic control-allocation error into `m_W` (or switch to omnidirectional bases to make
the elliptical force set exact); scalability `N∈{12,24,48,96}`; robustness (delay, packet
loss, `γ` error, robot failure); paired statistics (Wilcoxon + Holm, ≥50 seeds). These raise
the experimental score to 10 but are P2 relative to the modeling/theorem fixes above.
