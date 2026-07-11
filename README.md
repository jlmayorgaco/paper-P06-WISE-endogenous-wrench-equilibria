# P07 — WISE: Endogenous Wrench Equilibria

**WISE: Wrench- and Information-Self-Sustaining Equilibria for Heterogeneous
Multi-Robot Coalitions**

Target venue: **LARS 2026** (IEEE format, 4–6 pages, English).

---

## The core claim

> A Nash-stable coalition is not operationally meaningful if it cannot generate
> the required wrench, or if its own spatial deployment destroys the
> communication network needed to compute and maintain that equilibrium.

This motivates a new equilibrium concept:

**WISE — Wrench-and-Information Self-Sustaining Equilibrium.** A profile is WISE
when it is simultaneously

1. **strategically stable** (a generalized variational equilibrium),
2. **wrench-feasible** — every coalition produces the demanded force and torque,
3. **information self-sustaining** — the *induced* communication graph keeps
   enough algebraic connectivity to estimate the aggregates and prices that
   define the equilibrium.

> A WISE equilibrium can both **execute itself** and **compute itself**.

## The three technical objects

```
WISE equilibrium =  strategic stability
                  + wrench feasibility
                  + information self-sustainability

built from      =  directional wrench tensor   W[i,k,h,l]
                  + endogenous Laplacian        L(x) = sum_e a_e(x) L_e
                  + shared-constraint vGNE
                  + physical & information prices  (mu*, pi*)
```

- **Directional wrench tensor** `W[i,k,h,l] = h_{U_i}(A_kh^T eta_kl)` — robot ×
  load × contact-slot × wrench-direction. Directional capacity
  `s_kl(x) = sum_{i,h} W[i,k,h,l] x_ikh`; feasibility `s_kl(x) >= d_kl`.
- **Endogenous Laplacian** `L(x) = sum_e a_e(x) L_e` with affine edge weights
  `a_e(x) = sum_{i,r} C_ire x_ir`. The information constraint
  `lambda_2(L(x)) >= sigma*` is a convex super-level set (Fiedler value is
  concave in the Laplacian weights).
- **Nash-seeking threshold** `sigma* = theta^2 / (c * m_F)`.

## Repository layout

```
paper/          IEEE conference manuscript (main.tex + sections/)
src/wise_mr/    core library (tensor, endogenous graph, equilibrium, prices)
experiments/    exp01..exp04 — phase diagram, ablation, roles, threshold
configs/        YAML sweep definitions
tests/          unit + property tests (tensor, Fiedler gradient, feasibility)
results/        raw/, summaries/, figures/, manifests/  (regenerable)
```

## Quickstart

```bash
pip install -e ".[dev]"     # or: make install
make test                   # run the test suite
make exp01                  # phase-transition sweep (once implemented)
make paper                  # build paper/main.pdf
```

> **Status:** Day-1 scaffold. Module signatures and the mathematical contracts
> are in place; the physics/optimization bodies are stubbed
> (`NotImplementedError`) and tracked per the development plan.

## Development plan (5 days)

| Day | Gate |
| --- | --- |
| 1 — math freeze        | model + proofs fit in 2 IEEE pages; tensor & centralized certificate implemented |
| 2 — algorithm & scenes | centralized and WISE primal-dual agree on aggregates in convex cases |
| 3 — campaign           | at least two theoretical predictions show a visible, reproducible effect |
| 4 — writing            | full 7-page draft cut to 6; vector figures; hostile math review |
| 5 — audit & submit     | clean-room regeneration; every abstract number checked; early submission |

## License

MIT — see [`LICENSE`](LICENSE).
