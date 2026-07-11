# Claim–Evidence Matrix — TASK 0

Status legend: **PROVED** (correct proof in paper, scope matches) · **PARTIALLY PROVED**
(proved under narrower scope than the wording claims) · **SIMULATED** (only numerical
illustration) · **UNSUPPORTED** (no proof/code for the stated scope) · **FALSE AS WRITTEN**
(a specific stated step is mathematically wrong).

No scientific text was edited in TASK 0. "Required correction" is a plan for later tasks only.

Root cause for most defects: the paper uses a **weighted capability map** aggregate, while the
repo's correct model (`src/wise_mr/population_game.py`, Documento IV) uses a **column-sum**
aggregate. Several claims are true for the column-sum game but false for the weighted map as written.
See `repository_inventory.md` §2.

---

## Title / Abstract / Contributions

| ID | Wording (paper) | Type | Support (eq/proof) | Support (code) | Experiment | Status | Required correction |
|---|---|---|---|---|---|---|---|
| A1 | Title: "…**Decision-Induced** Robot Networks" | interp. | §II-C L(x) affine in x | `endogenous_graph.py`, `scenarios.py` (relay activates candidate links) | figs 1,3 | **PARTIALLY PROVED** | Graph is an affine, candidate-site relaxation, not the continuous position map L(q). Rule-3 word "induced" is defensible only for the relaxation; state it. |
| A2 | Abstract: "the productively optimal aggregate is unique but its composition is degenerate—a positive-dimensional face" | theorem | Thm 1 | `population_game.py::water_filling`, `degeneracy_dimension` (correct model) | — | **PARTIALLY PROVED** | Unique + degenerate hold for the **column-sum** game in `population_game.py`; the paper proves it for the weighted map, where it is not valid (see T1). Re-ground on `population_game.py`. |
| A3 | Abstract/§III: "$\sigma^\star=\vartheta^2/(cm_F)$ … an **exact** local stability threshold … derived from the coupled estimation dynamics" | theorem | Thm 3 (eq. 8–9) | `dynamics.py::error_jacobian,is_error_stable` | fig 5 | **PARTIALLY PROVED** | "exact/iff" holds only for the **postulated 2×2** J, not the full distributed system. Rule 3: downgrade to "sufficient" or scope explicitly to the 2×2 modal reduction; derive J from an explicit revision+estimator or state as assumption. |
| A4 | Abstract: "a self-sustaining equilibrium exists iff $\Lambda^\star\ge\sigma^\star$" | theorem | Thm 2 (eq. 7) | `endogenous_graph.fiedler_value` (max over E not computed in code) | fig 3 | **PARTIALLY PROVED** | The set-level iff is correct **given** E and the LMI; but "exists iff" uses `>=` while stability (A3) needs strict `>`. Add margin δ. `Λ*` is not actually maximized in code. |
| A5 | Abstract/§V: "Closed-loop simulation … **confirms** all four" | simulation | — | `run_campaign.py`, `fig_scene.py`, `dynamics.py` | figs 1–5 | **SIMULATED / OVERSTATED** | Rule 3: replace "confirms" with "illustrates / is consistent with". Sim cannot confirm the theorems; fig 5 validates the same 2×2 it illustrates. |
| A6 | Contribution: "relay roles emerge from radius heterogeneity **alone**" | simulation | Prop 2 (partial) | `run_campaign.fig_ablation_roles` | fig 4b | **SIMULATED** | True for this parameterization only; not a general theorem. Word "alone" unsupported; say "in our scenarios". |

## Section II — model

| ID | Wording | Type | Support | Code | Exp | Status | Required correction |
|---|---|---|---|---|---|---|---|
| P1a | Prop 1: "If every $U_\tau$ is a polytope, $\mathcal W(C_k)$ is a **zonotope**" | theorem | Prop 1 proof | `wrench_tensor.build_wrench_tensor` (disks), `certify_membership_lp` | — | **FALSE AS WRITTEN** | A Minkowski sum of general polytopes is a polytope, not necessarily a zonotope (triangle is a counterexample). Replace with: polytopic ⇒ polytope with finitely many facets; **zonotope only if each $U_\tau$ is a segment/centrally-symmetric zonotope**. |
| P1b | Prop 1: finite-direction test necessary; sufficient when normals are the facet normals | theorem | Prop 1 | same | — | **PROVED** (this half is fine) | Keep; it is the honest support-function statement. |
| M1 | §II-A: "$x_{\tau a}$ is a population mass, not a mixed strategy … Integer assignments recovered by rounding" | assumption | Rem 1 | — | — | **ASSUMPTION** | Legit framing; the rounding claim is R1 below. |
| M2 | §II-C: "$L(x)$ … convex, decision-induced **relaxation** of the position-dependent graph $L(q)$" | interp. | — | `scenarios.two_region`, `dynamics.live_lambda2` | figs 1,3 | **PARTIALLY PROVED** | Honest as a relaxation; keep the caveat. The continuous L(q) is nonconvex and only in sim. |
| L1 | Lemma 1: $x\mapsto\lambda_2(L(x))$ concave ⇒ LMI convex | theorem | Lemma 1 (Fiedler min form) | `endogenous_graph.fiedler_value` | — | **PROVED** | Correct. |

## Section III — equilibria

| ID | Wording | Type | Support | Code | Exp | Status | Required correction |
|---|---|---|---|---|---|---|---|
| T1 | Thm 1: "$\dim\mathcal E=(T-1)(K_{\rm act}-1)$", a transportation polytope, for aggregate $y=Bx$ | theorem | Thm 1 proof | correct in `population_game.py`; **wrong** for the paper's weighted map | — | **FALSE AS WRITTEN** | For the weighted map $s_{k\ell}=\sum W_{\tau kh\ell}x$, fixing $Bx=y^\star$ can also fix composition; fiber dim $\le(T-1)(K-1)$ and can be 0. Correct statement: $\dim_{\rm loc}\mathcal E=\dim(\ker B\cap T_{X_f})$, with $(T-1)(K-1)$ a **special case** of the genuine column-sum game (`population_game.py`). |
| T1b | Thm 1 uses $F_k(s_k)=-\tfrac12\lVert[d_k-s_k]_+\rVert^2$ as "strictly concave" to get unique $y^\star$ | theorem | Thm 1 / eq (5) | `equilibrium.potential` (same flat penalty) | — | **FALSE AS WRITTEN** | On $X_f=\{s\ge d\}$ the deficit is 0, so $F_k\equiv0$ — **not** strictly concave there; cannot yield unique aggregate. Use a genuinely strictly-concave $\Phi_0$ (e.g. `population_game`'s $v_ky_k-\tfrac{\alpha C}{2}y_k^2$). |
| D1 | Def 1 (WISE): $\lambda_2(L(x^\star))\ge\sigma^\star$ | definition | — | `equilibrium.is_wise` (uses `>= -tol`) | — | **INCONSISTENT with T3** | Stability (T3) needs strict `>`; at equality det J=0 (marginal). Define with margin $\lambda_2\ge\sigma^\star+\delta$, $\delta>0$. |
| T2 | Thm 2: $\mathcal E_{\rm ss}\ne\varnothing\iff\Lambda^\star\ge\sigma^\star$; $\mathcal E_{\rm ss}$ convex | theorem | Thm 2 proof | `fiedler_value` (max not computed) | fig 3 | **PROVED (set-level)** but scope caveats | Correct as a statement about E and the LMI; but depends on T1 (E well-defined) and uses `>=` (see D1). Keep, add δ, fix dependency on T1. |
| P2 | Prop 2: $V_\varepsilon=V+\varepsilon\lambda_2$ maximizers → $\arg\max_{\mathcal E}\lambda_2$ as $\varepsilon\downarrow0$ | theorem | Prop 2 (Tikhonov/Γ) | `primal_dual` connectivity price ≈ this selection | fig 4b | **PARTIALLY PROVED** | Argument is standard given $\arg\max V=\mathcal E$; but that requires T1/T1b fixed (flat V on E). Cite a Tikhonov-selection reference; currently uncited. |
| C1 | Cor 1: type-action value $q^\star_{\tau a}=\sum\mu^\star W-g+\pi^\star\partial\lambda_2$; "single wrench price, not a separate multiplier" | theorem | Cor 1 proof | `equilibrium.wrench_price`, `lambda2_relay_gradient` | — | **PARTIALLY PROVED / INCOMPLETE** | KKT omits multipliers for mass conservation $\sum_a x_{\tau a}=m_\tau$, wrench constraints, $Bx=y^\star$, and simplex normal cones. $\mu^\star=\nabla F_k$ is a marginal utility, not the constraint multiplier. Either write full Lagrangian or drop Cor 1. |
| G1 | $\Delta_{\rm endo}=V(x^\star_{\rm exo})-\max_{\mathcal E_{\rm ss}}V\ge0$, "can be positive" | theorem | eq (11) | — | — | **FALSE AS WRITTEN** | $\mathcal E_{\rm ss}\subseteq\mathcal E=\arg\max V$ ⇒ $\max_{\mathcal E_{\rm ss}}V=V^\star$ whenever nonempty ⇒ $\Delta_{\rm endo}\equiv0$ (or undefined if empty). Replace by trichotomy over a margin set $\mathcal C_\delta$: impossible / costly / free, with $\Delta_{\rm endo}=V^\star-\max_{X_f\cap\mathcal C_\delta}V$. |
| R1 | Rem 1: rounding preserves feasibility when margins exceed $\kappa=O(1/N)$ | theorem | Rem 1 (assertion) | — | — | **UNSUPPORTED** | No proof and $O(1/N)$ unjustified. Either prove a rounding lemma or state as conjecture/assumption. |

## Section IV — algorithm

| ID | Wording | Type | Support | Code | Exp | Status | Required correction |
|---|---|---|---|---|---|---|---|
| AL1 | §IV: replicator avoided because not Nash-stationary on the boundary | interp. | text (Sandholm) | `primal_dual._replicator_step` exists | — | **PROVED / defensible** | Correct and well-cited; keep. |
| AL2 | §IV: Fiedler sensitivity $v^\top L_e v$ "obtained by consensus over the current graph" | interp. | — | `endogenous_graph.lambda2_gradient` (centralized) | — | **UNSUPPORTED** | Distributed algebraic-connectivity / Fiedler-vector estimation is a nontrivial algorithm (dedicated literature); code computes it centrally. Cite a distributed-connectivity-estimation method or soften to "assumes an estimate of". |
| PR3 | Prop 3: primal-dual trajectories converge to a WISE, "provided the consensus gain keeps the estimator above the threshold throughout" | theorem | Prop 3 (sketch) | `primal_dual.solve` | fig 2 | **UNSUPPORTED (assumes conclusion)** | The proviso *is* the property to prove (forward invariance of $\{\lambda_2\ge\sigma^\star\}$). Restrict to projection dynamics + prove invariance (barrier/projection), or state as conjecture. Also "strong concavity of V" contradicts the degenerate face (V must be flat on E). |

## Section V — figures / table

| ID | Wording | Type | Support | Code | Exp | Status | Required correction |
|---|---|---|---|---|---|---|---|
| F1 | Fig 1: WISE composition "still realise the demanded wrench (red contact forces)" | simulation | — | `fig_scene.py` + `dynamics.realize_wrench` (elliptical set) | fig 1 | **SIMULATED** | Contact set is an idealized heading-aligned ellipse (declared in limitations). OK; keep caveat. |
| F2 | Fig 2: "$\lambda_2\to\sigma^\star$ and residual $\to0$" | simulation | — | `run_campaign.fig_convergence` (seed 3) | fig 2 | **SIMULATED** | Converging to the boundary is exactly where T3 gives only marginal stability. Note tension; with δ-margin, target $\sigma^\star+\delta$. |
| F3 | Fig 5: "estimation error decays iff $cm_F\lambda_2>\vartheta^2$" | simulation | Thm 3 | **orphan script (not in repo)** | fig 5 | **SIMULATED + NOT REPRODUCIBLE** | Illustrates the same 2×2 as T3 (not independent). **And the generating script is missing** — must be committed. |
| F4 | Fig 3 caption: certification "fails at $\nu\to0$ … and at high $\tau_d$", transition where predicted | simulation | — | `run_campaign.fig_phase` | fig 3 | **SIMULATED** | No theoretical boundary overlaid; caption implies prediction match. Overlay computed $\Lambda^\star=\sigma^\star$ curve or soften wording. |
| TB1 | Table I: WISE 93.8%, oracle 93.8%, others ≤6.2% | simulation | — | `run_campaign.ablation_table` (seeds range(16)) | table | **SIMULATED** | Caption lacks N-runs and CI; WISE==oracle expected since both solve the same relaxed program (state this). Add run count + CI columns. |
| F5 | Fig 4b: relays drawn "exclusively from long-range types" | simulation | Prop 2 | `run_campaign.fig_ablation_roles` | fig 4b | **SIMULATED** | Property of the parameterization; not general. Soften. |

## References

| ID | Wording | Status | Required correction |
|---|---|---|---|
| REF4 | [4] Martínez-Piazuelo et al., TAC, "vol. 68, no. 8, pp. 4529–4544, 2023" | **FALSE AS WRITTEN** | Correct metadata (per UPCommons/author): "J. Martinez-Piazuelo, C. Ocampo-Martinez, N. Quijano, *Distributed Nash Equilibrium Seeking in Strongly Contractive Aggregative Population Games*, IEEE TAC, vol. 69, no. 7, pp. 4427–4442, 2024, doi:10.1109/TAC.2023.3321208." Fix in a dedicated task (Rule 1: verify, do not invent). |
| REF-COV | Only 14 references for a strong novelty claim | **GAP** | Add nearby work: distributed connectivity estimation/control; population-game multi-robot task allocation; mechanism-design formation control (Ocampo-Martinez/Quijano/Barreiro-Gómez). Verify each before citing. |

## Forbidden-word audit (Rule 3)
- "exact" (Thm 3 title, §III-C): allowed only for the 2×2; **flag** (A3).
- "iff" (Thm 2 eq 7, Thm 3 eq 9): proved for the 2×2 (det sign) and set-level; **scope-limited** (A3/A4).
- "unique" (Thm 1 aggregate): supported in `population_game.py`, **not** for the paper's weighted map (T1/T1b).
- "confirms" (abstract, §V, conclusion): **replace** (A5).
- "position-induced" not used verbatim; title uses "decision-induced" (A1) — acceptable as relaxation.
- "guarantees/global convergence/fully distributed/optimal": check in later tasks; PR3 and AL2 currently overreach.
