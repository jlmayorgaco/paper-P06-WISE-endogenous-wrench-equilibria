# WISE — Zero-Cost Connectivity within Wrench-Feasible Assignment Fibers

Code, data and manuscript for

> **WISE: Zero-Cost Connectivity within Wrench-Feasible Assignment Fibers for
> Heterogeneous Robot Transport**
> Jorge Luis Mayorga Taborda, Universidad Internacional de Valencia (VIU).
> Latin American Robotics Symposium (LARS) 2026, 6 pages, IEEE format.

Release **`v1.0-wise`** · code and data commit **`14fa38a3`** · MIT licence.

---

## The question

A heterogeneous team must decide two things at once: which robots push the load, and
which relay the information that lets the team agree on the push. An assignment can be
productively optimal — it delivers the demanded wrench — and still commit every robot to
lifting, leaving none to relay. Connectivity collapses, and the estimator that computes
the assignment loses its channel.

## The answer

When productive utility depends only on a low-dimensional **served aggregate** `y = Bz`,
the optimal aggregate `y*` is unique but the optimal *set* is a
**positive-dimensional polytope** — a fiber `E = {z ∈ X_f : Bz = y*}` of
wrench-feasible compositions that all earn `V*`. Connectivity can then be bought inside
that fiber at exactly zero productive loss, or not at all, and one scalar decides which.

**Free-networkability modulus.** For `z̄ ∈ E`, with `T_E(z̄)` the tangent cone,

```
Gamma_E(z̄) = max { D lambda_2(z̄)[d] : d ∈ T_E(z̄), ||d||_2 <= 1 }
```

**Main theorem (local test ⇒ global certificate).**

```
Gamma_E(z̄) > 0  <=>  ∃ z' ∈ E with lambda_2(Lbar(z')) > lambda_2(Lbar(z̄))
Gamma_E(z̄) = 0  <=>  lambda_2(Lbar(z̄)) = Lambda_E
```

A single directional test at one point certifies global connectivity optimality over the
whole fiber. The criterion is stated on `Q^T Lbar Q`, so it covers a **repeated** Fiedler
eigenvalue, not just the simple case.

**WISE** is then the *lexicographic refinement* of the productive variational-equilibrium
set: first `max V`, then `max lambda_2` on the resulting fiber, computed by a Stage-2 SDP.
The capacities `Lambda_E <= Lambda_X` split requirements into **free / costly /
impossible**, and the multiplier prices connectivity in units of utility.

**What clearing the threshold buys.** Above `sigma_dyn = t1·t2/(c·m_y)` the reduced
information layer is exponentially stable at rate `alpha(lambda_2)`, and — new in this
release — a common Lyapunov function gives the **same certified lower rate bound while
the physical graph moves**:

```
V_c = (t2||a||^2 + t1||b||^2)/2 ,  Q^T L_geo(q(t)) Q >= sigma_req I  for all t
  =>  dV_c/dt <= -2 alpha(sigma_req) V_c ,   alpha exactly the frozen-graph rate.
```

## Scope, honestly

* **Centralized selection of an equilibrium set**, not distributed Nash seeking. No
  equilibrium-seeking dynamics are claimed or run.
* Guarantees hold for the **affine relaxation**; they transfer to the physical graph
  `L_geo(q)` only at a **re-certified integer** assignment, via a Loewner lower bound.
* Wrench feasibility is **nominal, assignment-level** certification in a quasi-static
  planar model with rigid (bilateral) attachment and a heading-independent inner
  zonotope. Unilateral contact, friction cones and contact-mode transitions are outside
  the model.
* The closed-loop study is a **planar transport simulation with rigidly attached
  robots** — load dynamics, contact forces, the moving communication graph and the
  reduced information layer. It is *not* a nonlinear robot–load stability theorem, not
  hardware, and it does not simulate individual robot locomotion.
* No packet loss, delay or link failure model — so nothing is claimed about them.

## What is in the paper

| | |
|---|---|
| **E1** | The `N=6` flagship: an exhaustively verified **integer** zero-cost assignment. All `\|A\|^N = 12^6 = 2.99e6` maps enumerated; 3864 wrench-feasible, 1032 on the fiber, `Lambda_E^Z = 0.399`. |
| **E2** | Planar closed-loop transport of two rigid loads. PROD / HARD / WISE share dynamics, controller, allocator, gains, disturbance and initial state. 30 paired worlds: **30/30** success for the selectors clearing `sigma_req`, **0/30** for the productive-only optimum and a random fiber point. |
| **E3** | Structured integer recovery at `N=12` and an exact oracle for `N<=8`. |
| **E4** | Regime map over `(nu, tau_d)`, the price `P(sigma)`, and scalarization controls. |

## Reproduce

```bash
pip install -e ".[dev,opt,viz]"     # numpy/scipy + cvxpy (SDP) + matplotlib
make reproduce                      # tests -> certificates -> experiments -> paper -> gates
make robot                          # E2: PHASE-R0 audit + deterministic closed-loop run
```

`make reproduce` runs the test suite, regenerates every certificate (`generated/*.json`)
and experiment (`generated/*.csv`, `paper/figures/*.pdf`), rebuilds `paper/main.pdf` and
asserts the gates. `make robot` re-runs the closed-loop experiment of E2 from one command.

Individual targets:

```bash
make test          # unit + property tests
make robot-test    # the closed-loop invariant tests
make robot-mc      # 30 paired Monte-Carlo worlds       (SEEDS=30)
make robot-sweep   # predeclared relay-attenuation sweep
make robot-fig     # paper hero figure + supplementary figure
make robot-video   # PROD | HARD | WISE side-by-side animation
make paper         # build paper/main.pdf
python paper/checkbuild.py 6        # build gates: pages, refs, overfull boxes, stray markers
```

## Artifact map

```
paper/                    IEEEtran manuscript (main.tex + sections/) and figures
paper/checkbuild.py       build gates: 6 pages, 0 undefined refs, 0 overfull boxes
src/wise_mr/              core library: wrench lifting, endogenous graph, SDP, dynamics
experiments/              certificates and experiments E1, E3, E4 + reproduce.py
experiments/robot_closed_loop/   E2: the planar closed-loop transport study
docs/ROBOT_EXPERIMENT_DESIGN.md  E2 design, declared parameters, hypotheses, findings
docs/TIME_VARYING_STABILITY_PROOF.md   the moving-graph corollary and its checks
generated/                every number the paper reports, as JSON/CSV manifests
figures/, videos/         inspection renders and the side-by-side animation
tests/                    unit + property tests for the core library
```

### Where each reported number lives

| Paper | Manifest |
|---|---|
| flagship aggregate, wrenches, `lambda_2` | `generated/flagship.json`, `generated/robot_experiment_input_manifest.json` |
| E2 deterministic run | `generated/robot_flagship_summary.json`, `generated/robot_flagship_timeseries.csv` |
| E2 Monte Carlo | `generated/robot_monte_carlo_runs.csv`, `generated/robot_statistical_report.json` |
| E2 attenuation sweep | `generated/robot_margin_sweep.json`, `generated/robot_margin_sweep_runs.csv` |
| certified service rate | `generated/methods_comparison.csv` |
| fiber dimension, `Gamma_E` | `generated/fiber_certificate.json`, `generated/gamma_certificate.json` |
| regimes and price | `generated/regime_grid.csv`, `generated/regime_summary.json`, `generated/price_sweep.csv` |
| integer recovery, oracle | `generated/rounding.csv`, `generated/oracle_benchmark.csv` |

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
