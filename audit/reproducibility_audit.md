# Reproducibility Audit — TASK 0 (part D)

For each figure/table in the paper: source script, configuration, seeds, raw-data path,
and whether it regenerates from a clean checkout.

| Artifact | Source script | Config source | Seeds | Raw data persisted? | Regenerable from clean checkout? |
|---|---|---|---|---|---|
| Fig 1 (`fig1_scene.pdf`) | `experiments/fig_scene.py` | hardcoded in script | `seed=7`, N=10, ν=0.4, τ_d=2.5 | No | **Yes** — `python experiments/fig_scene.py` |
| Fig 2 (`fig2_convergence.pdf`) | `experiments/run_campaign.py::fig_convergence` | hardcoded | `seed=3`, N=12, ν=0.42, τ_d=3.0 | No | **Yes** — `python experiments/run_campaign.py` |
| Fig 3 (`fig3_phase.pdf`) | `run_campaign.py::fig_phase` | hardcoded grid | `seeds=range(12)`, ν∈[0,1]×11, τ_d∈[0.5,7]×9 | No | **Yes** (slow) |
| Fig 4 (`fig4_ablation_roles.pdf`) | `run_campaign.py::fig_ablation_roles` | hardcoded | `seeds=range(16)`, ν=0.5, τ_d=3.0 | No | **Yes** |
| Fig 5 (`fig5_threshold.pdf`) | **MISSING** — created by an ad-hoc inline `python -c` command | none | positions hardcoded inline; m_F=c=1.0, ϑ=0.5 | No | **NO — script not in repo** |
| Table I (`ablation_table.tex`) | `run_campaign.py::ablation_table` | hardcoded | `seeds=range(16)`, ν=0.5, τ_d=3.0 | No (values written straight to `.tex`) | **Yes** |

## Blocking reproducibility issues
1. **Fig 5 orphan (BLOCKING).** The figure used at `05_experiments.tex:49` has **no generating
   script** committed. It cannot be regenerated. A later task must add a committed script
   (e.g. `experiments/fig_threshold.py`) that reproduces it deterministically.
2. **No persisted raw data / manifests.** `results/raw`, `results/summaries`, `results/manifests`
   contain only `.gitkeep`. Figures are written directly to `paper/figures/`. There is no
   run-manifest (seed list, git hash, parameters, timestamp) for any figure or table.
3. **Configs are dead.** `configs/*.yaml` are not consumed by the figure pipeline
   (`run_campaign.py` hardcodes parameters); they belong to the `exp01-04` stubs that raise
   `NotImplementedError`. Parameters therefore live only inside scripts.
4. **No repository URL / commit hash in the paper.** Nothing links the PDF to the code state.
5. **Determinism unverified.** Regeneration was not re-run in TASK 0 (Rule: no scientific
   changes). A later task must regenerate all figures from a clean checkout and diff.

## Environment (for the record)
- Python 3.13.14; numpy 2.3.5, scipy 1.16.3, matplotlib 3.10.8; pytest 8.4.2.
- MiKTeX 25.12 (pdfTeX 4.23, latexmk 4.88, bibtex 4.2).
- Platform: Windows.
