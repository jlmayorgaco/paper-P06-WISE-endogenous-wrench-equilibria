# Baseline Compile Metrics — TASK 0

Frozen PDF: `artifacts/baseline_before_hardening.pdf`
Full LaTeX log: `audit/baseline_build.log`
Full test log: `audit/baseline_tests.txt`

## Compilation
- Command: `latexmk -C && latexmk -pdf -interaction=nonstopmode main.tex` (run in `paper/`)
- Toolchain: MiKTeX-pdfTeX 4.23 (MiKTeX 25.12), latexmk 4.88, bibtex 4.2
- Exit code: 0

## PDF properties
| Metric | Value |
|---|---|
| Pages | 6 (within LARS 4–6) |
| Size | 485,080 bytes |
| Overfull \hbox | 0 |
| Underfull \hbox | 0 |
| Overfull \vbox | 0 |
| Underfull \vbox | 0 |
| Undefined references | 0 |
| Undefined citations | 0 |
| LaTeX warnings | 0 |

## Fonts (`pdffonts`)
- Text/math fonts (NimbusRomNo9L, CM*, MSBM10, NimbusMonL): **Type 1, all embedded** — OK.
- **FINDING (submission risk):** matplotlib figures embed **Type 3 fonts** (DejaVuSans,
  DejaVuSans-Oblique, DejaVuSerif). Type 3 fonts are frequently rejected by IEEE PDF eXpress.
  They are embedded, so Rule 9 ("all fonts embedded") is met, but a later hardening task should
  force Type 1/TrueType output (`matplotlib.rcParams['pdf.fonttype']=42`, or `usetex`, or
  regenerate figures) to be safe for IEEE submission. **Not changed in TASK 0.**

## Test suite (`python -m pytest -v`)
- Result: **10 passed, 5 skipped** (0 failed), 1.17 s, Python 3.13.14, pytest 8.4.2.
- **FINDING (evidence risk):** tests cover only low-level utilities —
  `support_box`, `directional_capacity`, `demand_projection`, `wrench_residual`,
  `sigma_star` formula, `edge_weights`, `fiedler_value`, `project_simplex`.
  They do **not** test any theorem, the population game, the primal-dual solver,
  the dynamics module, the baselines, or the figure pipeline. The 5 skipped tests are
  exactly the theorem-relevant ones (centralized equivalence x2, Fiedler-gradient
  finite-difference, WISE membership, tensor-vs-LP membership).
- Therefore "10 tests passed" is **not** evidence for any scientific claim in the paper.
