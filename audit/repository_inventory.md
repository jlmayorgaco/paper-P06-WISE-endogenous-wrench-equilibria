# Repository Inventory — TASK 0 audit

Branch: `hardening/lars-2026-submit-ready`
Frozen at: session 2026-07-11. No commits existed before this audit (fresh repo).

## 1. Directory tree (working files, excluding `.git`, `__pycache__`, LaTeX aux)

```
paper/
  main.tex                     IEEE conference driver (title, abstract, preamble)
  references.bib               14 entries
  sections/01_introduction.tex
  sections/02_model.tex        population model, support-capability map, LMI, Thm 1
  sections/03_wise_equilibrium.tex  Def 1, Thm 2, Thm 3, Prop 2, Cor 1, Delta_endo, Rem 1
  sections/04_algorithm.tex    primal-dual, Prop 3
  sections/05_experiments.tex  Figs 1-5 + Table I
  sections/06_conclusion.tex
  figures/fig1_scene.pdf
  figures/fig2_convergence.pdf
  figures/fig3_phase.pdf
  figures/fig4_ablation_roles.pdf
  figures/fig5_threshold.pdf   <-- ORPHAN: no generating script in repo (see sec. 3)
  figures/ablation_table.tex   generated table body

src/wise_mr/
  __init__.py
  wrench_tensor.py             support functions, capability map, LP membership
  endogenous_graph.py          affine Laplacian, Fiedler value/gradient, sigma*
  equilibrium.py               WiseProblem, potential, feasibility  (USED by paper figs)
  primal_dual.py               projected primal-dual solver
  baselines.py                 5 solvers for the ablation
  metrics.py                   certified rate, bootstrap CI
  scenarios.py                 two_region instance generator
  dynamics.py                  rigid load + unicycle + 2x2 error + wrench realization
  population_game.py           Documento IV population game  <-- NOT IMPORTED ANYWHERE

experiments/
  _common.py                   config/arg plumbing (used by exp01-04 stubs only)
  exp01_phase_transition.py    STUB (raises NotImplementedError)
  exp02_constraint_ablation.py STUB
  exp03_role_emergence.py      STUB
  exp04_threshold_validation.py STUB
  run_campaign.py              generates fig2, fig3, fig4, ablation_table (+ orphan fig1_concept)
  fig_scene.py                 generates fig1_scene

configs/
  phase_transition.yaml        NOT consumed by run_campaign.py (belongs to exp01 stub)
  ablation.yaml                NOT consumed by run_campaign.py
  roles.yaml                   NOT consumed by run_campaign.py

tests/
  conftest.py                  adds src/ to path
  test_wrench_tensor.py        4 pass, 1 skip
  test_fiedler_gradient.py     4 pass, 1 skip
  test_wise_feasibility.py     2 pass, 1 skip  (project_simplex)
  test_centralized_equivalence.py  0 pass, 2 skip

docs/sources/                  author's source manuscripts (pre-existing, not authored this session)
  Heterogeneidad_Grafo_Endogeno_Resultado_Frontera.pdf   "Documento IV" (grounding source)
  adenda_topologia_endogena.pdf
  programa_teorico_wrench_EL.pdf
  wrench_market_games_paper.pdf

RECONSTRUCTION_PLAN.md         authoritative plan: hard-reset paper onto Documento IV
README.md, LICENSE, CITATION.cff, Makefile, pyproject.toml, .gitignore
```

## 2. CRITICAL STRUCTURAL FINDING — paper theory has diverged from the repo's own authoritative model

- `RECONSTRUCTION_PLAN.md` (pre-existing) states the paper must be grounded in **Documento IV**
  (`docs/sources/Heterogeneidad_Grafo_Endogeno_Resultado_Frontera.pdf`) and that the current
  `paper/` draft "drifted into a tensor/GVE/two-price formulation that the review correctly rejects."
- `src/wise_mr/population_game.py` implements the Documento IV game **correctly**:
  - state `x_{tau k}` = capacity fraction; aggregate `y_k = sum_tau x_{tau k}` (**genuine column sum**);
  - potential `Phi = sum_k (v_k y_k - (alpha C/2) y_k^2)`, **strictly concave in y** (`mu = alpha C`);
  - `water_filling` -> unique `y*`; `degeneracy_dimension = (T-1)(M_act-1)` for the **transportation polytope**.
- The paper's Section II-III instead uses a **weighted capability map** `s_{k l}(x)=sum W_{tau,khl} x_{tau kh}`
  and asserts the transportation-polytope dimension for that weighted map. The dimension formula is
  valid for the column-sum game in `population_game.py`, **not** for the weighted map as written.
- `population_game.py` is **not imported** by any paper figure, experiment, or test.

Consequence: the paper claims are not backed by the repository's correct module; they are backed by a
divergent formulation that reproduces the exact defects the reviewer flagged. This is the root cause
behind most FALSE/UNSUPPORTED rows in `claim_evidence_matrix.md`.

## 3. Figure and table provenance

| Artifact | Used in paper (05_experiments.tex) | Generating script | Regenerable from clean checkout? |
|---|---|---|---|
| fig1_scene.pdf | line 16 | `experiments/fig_scene.py` (`seed=7,N=10,nu=0.4,tau_d=2.5`) | Yes |
| fig2_convergence.pdf | line 33 | `experiments/run_campaign.py::fig_convergence` (`seed=3,N=12,nu=0.42,tau_d=3.0`) | Yes |
| fig5_threshold.pdf | line 49 | **NONE — orphan** (created by an ad-hoc inline `python -c` command; not in repo) | **No** |
| fig3_phase.pdf | line 64 | `run_campaign.py::fig_phase` (`seeds=range(12)`, grid hardcoded) | Yes |
| ablation_table.tex | line 83 | `run_campaign.py::ablation_table` (`seeds=range(16),nu=0.5,tau_d=3.0`) | Yes |
| fig4_ablation_roles.pdf | line 95 | `run_campaign.py::fig_ablation_roles` (`seeds=range(16)`) | Yes |
| fig1_concept.pdf | (not used) | `run_campaign.py::fig_concept` | Yes (orphan output) |

## 4. Notes
- Seeds are hardcoded in the scripts, not read from `configs/*.yaml`; the YAML configs are dead
  (belong to the exp01-04 stubs, which raise `NotImplementedError`).
- No `results/raw/` data is persisted; figures are generated directly to `paper/figures/`.
- No repository URL / commit hash is embedded in the paper for reproducibility.
