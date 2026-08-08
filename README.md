# P06 — WISE: Endogenous Wrench Equilibria

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
src/wise_mr/    core library (wrench lifting, endogenous graph, equilibrium, SDP)
experiments/    reproduce.py + certificates and experiments (fiber, phase, methods)
generated/      regenerated certificates (*.json) and experiment data (*.csv)
tests/          unit + property tests (lifting, SDP, Fiedler gradient, feasibility)
paper/          IEEEtran manuscript and figures
```

## Reproduce

```bash
pip install -e ".[dev,opt,viz]"   # numpy/scipy + cvxpy (SDP) + matplotlib
make reproduce                    # tests -> certificates -> experiments -> paper -> checks
```

`make reproduce` runs the test suite, regenerates every certificate
(`generated/*.json`) and experiment (`generated/*.csv`, `paper/figures/*.pdf`),
rebuilds `paper/main.pdf`, and asserts the paper is exactly six pages with no
undefined references or overfull boxes. `make reproduce-fast` runs the same
pipeline with smaller sweeps for a quick smoke test.

Individual pieces:

```bash
make test                              # unit + property tests
python experiments/exp_fiber.py        # E-fiber: V flat, lambda2 varying (Example 1)
python experiments/exp_phase.py        # phase diagram with the exact SDP boundary
python experiments/exp_methods.py      # 7-method comparison, 30 seeds, bootstrap CI
make paper                             # build paper/main.pdf
```

## License

MIT — see [`LICENSE`](LICENSE).
