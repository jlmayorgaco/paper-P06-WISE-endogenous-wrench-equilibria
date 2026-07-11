# TFM ⇄ Conference-paper mapping — TASK 1 (part D)

Purpose: document, internally, how the LARS 2026 paper relates to the author's broader
TFM / master's work on heterogeneous-AGV coalition formation, so the paper neither over-claims
(reproducing the whole thesis) nor hides its lineage.

Sources in `docs/sources/` (author-provided; exact section numbers to be verified against the originals):
- `Heterogeneidad_Grafo_Endogeno_Resultado_Frontera.pdf` — **"Documento IV"**, the rigorous
  endogenous-graph / self-sustaining-equilibrium result that grounds this paper
  (see `RECONSTRUCTION_PLAN.md`).
- `adenda_topologia_endogena.pdf`, `programa_teorico_wrench_EL.pdf`,
  `wrench_market_games_paper.pdf` — related theory (wrench-market games, Euler–Lagrange program).

> Caveat: this mapping is compiled from `RECONSTRUCTION_PLAN.md`, the review correspondence, and the
> source filenames. Items marked **(verify)** must be checked against the TFM text before the paper's
> lineage sentence is finalized.

## A. TFM problem components **reused** in the paper
| TFM component | Reused as (paper) | Notes |
|---|---|---|
| Heterogeneous AGV fleet, coalition/recruitment to tasks | population types $\tau$, recruitment to loads $k$ | (verify) |
| Cooperative transport of a load requiring force/torque | wrench demand $\wdem_k$, feasibility $s(x)\ge d$ | wrench layer kept only as a feasibility set |
| Distributed population dynamics / revision | Smith/projected revision protocol in §IV | (verify TFM protocol) |
| Local (range-limited) communication | undirected weighted graph, $\lambda_2$ constraint | |
| Endogenous coupling decision→network | decision-induced candidate-site Laplacian $\bar L(x)$; geometric $L_{\mathrm{geo}}(q)$ in sim | Documento IV is the direct source |

## B. Robot model **reused**
- Mobile ground robots executing transport; the paper's closed-loop simulation uses **second-order
  unicycles** with a **heading-aligned (nonholonomic) contact-force set**. Whether the TFM uses the
  same second-order unicycle model or a kinematic AGV model is **(verify)**; the paper does not claim
  the TFM's full vehicle/actuation model.

## C. Communication assumption **reused**
- Range-limited, undirected, weighted links; connectivity measured by algebraic connectivity
  $\lambda_2$. The paper adds the **candidate-site affine surrogate** $\bar L(x)$ for the convex
  theory and keeps the **geometric** $L_{\mathrm{geo}}(q)$ only in simulation.

## D. What is **new** in the conference paper (not in the TFM as such)
1. The **self-referential framing**: the equilibrium's own composition decides whether the graph its
   estimator needs exists.
2. **Composition-degeneracy → selection** reading of the endogenous-graph result (optimal composition
   fiber; select the self-sustaining sub-set).
3. The **support-capability map** presentation of the wrench feasibility layer.
4. The **closed-loop rigid-load + unicycle** simulation with the nonholonomic contact set.
(Which of 1–2 are already explicit in Documento IV vs. newly framed here is **(verify)**; several are
attributed to Documento IV in `RECONSTRUCTION_PLAN.md` §2.)

## E. Parts of the TFM **not claimed** in the paper
- Full wrench-market-games / Euler–Lagrange program (`wrench_market_games_paper.pdf`,
  `programa_teorico_wrench_EL.pdf`).
- Cardinality-threshold / sigmoid recruitment laws and spatially-modulated revision, if present in
  the TFM (**verify**) — the paper uses the connectivity-threshold result, not those.
- Any hardware / full-fleet logistics scenario from the TFM.
- The two-price / GVE apparatus that the review rejected (explicitly removed; see audit matrix).

## F. Recommended lineage sentence for the paper (Introduction) — for a later task
> "This work isolates and formalizes the communication-feasibility layer of the author's broader
> heterogeneous-AGV cooperative-transport study, grounded in the endogenous-graph analysis of
> [Documento IV]; it does not reproduce the full transport or market-games program."

(Not inserted in TASK 1: adding a citation to the TFM/Documento IV requires a real bibliographic
entry and belongs to the references task.)
