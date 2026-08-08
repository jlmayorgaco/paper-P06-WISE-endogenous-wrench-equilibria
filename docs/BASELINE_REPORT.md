# Baseline freeze — WISE best-paper program

Branch `wise-best-paper`, forked from `revision/lars-2026-submit-ready` @ `ce492ed`.
Date: 2026-08-07.

## Baseline state (before any edit)

| Item | Value |
|---|---|
| PDF build | `latexmk -pdf main.tex` → exit 0 |
| Page count | **6** (LARS limit: 4–6) |
| Test suite | `pytest -q` → 32 passed, 0 failed (4 skipped: cvxpy-gated) |
| Undefined refs/citations | 0 |
| Overfull boxes | 0 |
| Dirty working tree | `generated/mismatch.json` (uncommitted) |

## Environment finding (P0-repro)

**`cvxpy` was not installed in the interpreter that runs the repo.** The 4 skipped
tests are exactly the SDP-gated ones, and every SDP-derived number in the manuscript
(Λ_E, the price sweep, the scaling table) therefore came from an environment that no
longer existed on this machine. Installed to match the versions the paper declares:

| Package | Paper claim | Installed |
|---|---|---|
| CVXPY | 1.9 | 1.9.2 |
| CLARABEL | 0.11 | 0.11.1 |
| numpy | — | 2.3.5 |
| scipy | — | 1.16.3 |

**Reproduction check:** the integer-recovery campaign re-ran to `895/900` post-repair,
bit-identical to the manuscript claim. The seeds do reproduce; the *environment* was
the unrecorded part. Action: pin versions in a lockfile before the final freeze
(Phase 0 item 10 remains open).

## Assets that must not be lost

These are real strengths of the current manuscript and are protected:

1. Optimal fiber `E` + dimension formula (Thm. 1) — certified to `1e-15` residuals.
2. Fiber ⇔ strong-monotonicity-loss equivalence (Prop. 1).
3. Stage-1/Stage-2 lexicographic SDP (Thm. 2).
4. Price of self-sustainability `P(σ)` + dual sensitivity (Thm. 3).
5. **Exhaustive oracle** enumerating every `|A|^N` map for `N ≤ 8` — absolute ground
   truth, `0/23` false positives at the design requirement. Not to be replaced.
6. Structured integer recovery + repair.
7. Flagship `λ₂ : 0 → 0.41` at fixed productive value.

## Pre-existing experiment inventory

Already implemented and generating data, but **not surfaced in the manuscript**:

| Script | Output | Status |
|---|---|---|
| `exp_phase.py` | `phase_grid.csv`, `phase_confusion.csv` | computed, unused in paper |
| `exp_epsilon.py` | `epsilon_sweep.csv` | computed, unused in paper |
| `exp_physical.py` | `physical_run.csv` | closed-loop rigid-load transport |
| `src/wise_mr/dynamics.py` | — | **2×2 aggregate–estimator error system already coded** |

`dynamics.py:35-48` implements `error_jacobian` with threshold `c·m_F·λ₂ > θ²`, i.e.
exactly `σ_dyn = θ²/(c·m_F)`. The manuscript's `ϑ₁ϑ₂/(c·m_F)` is the asymmetric
generalisation; the code is the symmetric instance `ϑ₁ = ϑ₂ = θ`. Consistent.
