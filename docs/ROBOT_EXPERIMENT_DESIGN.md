# E-Robot — zero-cost information self-sustainability in cooperative dual-load transport

**Design, declared parameters, predeclared hypotheses, and the mismatches the audit found.**

Code: `experiments/robot_closed_loop/`.
Entry points: `run_flagship.py` (deterministic run + PHASE-R0 audit),
`run_monte_carlo.py` (paired campaign), `make_figure.py`, `render.py`.
Companion proof: [`TIME_VARYING_STABILITY_PROOF.md`](TIME_VARYING_STABILITY_PROOF.md).

---

## 1. The question

> Can WISE preserve the exact productive aggregate and wrench feasibility while
> turning a globally uncoordinated robot transport mission into a stable,
> synchronized one?

The chain under test — and *only* this chain:

```
Gamma_E > 0  =>  free composition exchange  =>  lambda2 up
             =>  information layer stable   =>  better robot-level coordination
```

Nothing here claims a nonlinear robot–load stability theorem, hardware validation,
an actuator-level proof, distributed assignment convergence, or robustness to
communication phenomena that are not modelled.

## 2. What is imported and what is declared

Everything physical is **imported** from the frozen `experiments/exp_flagship.py`
(robot names/homes, `R_S=3`, `R_L=5`, `GAMMA_S=0.2`, `GAMMA_L=1.0`, `C_S=1`,
`C_L=2`, `F_S=1`, `F_L=2`, both loads' contact maps `A_kh` and certified demands
`w_dem = (3,0,0.3)` and `(3,0,1.0)`, the edge-weight laws `exp(-d/2)` /
`gamma exp(-d/3)`, `kappa=1`, `m_sides=12`) and the information gains are the
`two_region` defaults already used by E5 (`m_y=1`, `theta_1=theta_2=0.5`, `c=1`,
hence `sigma_dyn = 0.25`). No number is copied from the PDF; the full extraction is
written to `generated/robot_experiment_input_manifest.json`.

Three things are **declared here**, because the flagship is a single static pose and
a mission is not:

| Declared | Value | Why |
|---|---|---|
| Relay-site grid | cross `{(xR,0), (xR±0.8,0), (xR,±1.2)}`, `xR=4` | The paper's model has a finite relay-site set; the flagship declares one point. A *grid* (not a hand-picked pair) is what keeps HARD honest. |
| Transport paths | both loads travel 1.2 m transverse to the gap axis with a 0.25 m bow, arclength 1.385 m, orientation held | Long transport while each region stays compact, so the inter-region distance never falls to `R_S` and the relay stays inside `R_L`. Constant orientation leaves the body-frame `A_kh` — and therefore the frozen certificate — untouched. |
| `sigma_req` | `sigma_dyn + 0.05 = 0.30` | The flagship record carries no `sigma_req` (see MISMATCH-3). |

### Selection is exact, not rounded

With `N = 6` robots and `|A| = 12` actions (6 contact slots + 5 relay sites + idle)
the whole integer assignment space is `12^6 = 2,985,984` maps, so every baseline is
an **exhaustive integer optimum**: 3,864 pass occupancy *and* the inner-zonotope
wrench LP, and 1,032 of those sit on the productive-optimal fiber `E`. There is no
relaxation, no rounding and no recovery heuristic anywhere in this experiment, and
integrality plus direct re-certification are properties of the construction.

Because selection is by enumeration, `Lbar` is evaluated **per assignment** from the
tubes of its realized actions rather than as an affine function of `z`. That is
legitimate for the a-posteriori certificate of Lemma "bridge" (which is stated at an
integer `z_hat`) and it is what exposes MISMATCH-1 below; it is *not* the affine SDP
surrogate, and this experiment makes no claim about that surrogate.

## 3. Baselines

| Method | Problem solved | Tie-break |
|---|---|---|
| **PROD** | `max V` | least deployment cost, then lexicographic |
| **HARD** | `max V` s.t. `lambda2(Lbar) >= sigma_req` | fewest role changes vs PROD, then least cost, then lexicographic |
| **WISE** | `max lambda2(Lbar)` over `E` | same chain |
| **SCALAR** | `max V + eps lambda2(Lbar)`, `eps in {1e-3,1e-2,1e-1,1}` | same chain |
| **RANDOM-FIBER** | uniform draw from `E` | — |

The tie-break chain is declared once, in `assignments.py`, and is identical for every
method. HARD is *not* hand-steered to a weak point: it takes the cheapest grid site
that clears the threshold.

Resulting assignments (all with `Bz = y* = (3,3)`, `V = V* = 9`, both wrenches
certified):

```
PROD  L1 lift(1,h0)  L2 lift(2,h0)  S1 idle        S2 lift(1,h3)  S3 lift(2,h1)  S4 idle
HARD  L1 relay@W     L2 lift(2,h0)  S1 lift(1,h1)  S2 lift(1,h3)  S3 lift(2,h1)  S4 lift(1,h2)
WISE  L1 relay@E     L2 lift(2,h0)  S1 lift(1,h1)  S2 lift(1,h3)  S3 lift(2,h1)  S4 lift(1,h2)
```

HARD and WISE differ in **one coordinate**: which relay site the freed long robot
occupies. Same roles, same wrench, same `V`, same `Bz` — only the spectral margin
differs. That is the cleanest possible isolation of the WISE refinement.

`lambda2(Lbar)` over the relay grid, for the otherwise-identical composition:

| site | position | `lambda2(Lbar)` | `alpha` | deployment cost |
|---|---|---|---|---|
| C | (4.0, 0.0) | 0.38773 | +0.1076 | 5.056 |
| W | (3.2, 0.0) | 0.33646 | +0.0682 | **4.896** (cheapest ⇒ HARD) |
| **E** | (4.8, 0.0) | **0.39888** | **+0.1161** | 5.216 (⇒ WISE) |
| N | (4.0, 1.2) | 0.29219 | +0.0335 | 5.092 |
| S | (4.0, −1.2) | 0.29219 | +0.0335 | 5.092 |

`lambda2(Lbar)` on the fiber takes exactly seven values:
`{0, 0.1436, 0.2149, 0.2922, 0.3365, 0.3877, 0.3989}`.

## 4. Mission and disturbance

Both loads follow separate curved paths and must keep the same normalized progress
`s_1 ≈ s_2`. Phases are separated: **deployment** `[0, 6) s` (robots travel, loads
untouched, *no graph guarantee claimed*) and the **certified operational window**
`[6, 70] s`, in which every robot is inside its declared tube and Lemma "bridge"
applies.

* Loads: `m = 6 kg`, `J = 1.2`, linear drag `44 N s/m`, rotational `6`, planar RK4 at
  `dt = 0.01 s`, control (wrench allocation + commands) at 20 Hz.
* **Resistance is collinear with the certified demand.** The drag acts at the body
  point `(0, ecc)` with `ecc = -tau_cert/F_cert` (`-0.1` and `-1/3`), so the wrench
  the team must supply is always a positive multiple of `w_dem_k`, and `w_dem_k` is
  exactly the resistance at the fastest mission speed. The nominal operating point is
  55 % of it, leaving the rest as control authority.
* **Wrench allocation** (identical for every method): per load,
  `min ||G u||^2 + RHO^2 ||A_G u - w_dem||^2` s.t. `||u||_inf <= 1`, where
  `f_i = G_tau u_i` are exactly the inner-zonotope generators the flagship
  certificate uses. Bounded linear least squares (`scipy lsq_linear`, BVLS),
  deterministic. The residual `r_w` is recorded at every step.
* **Progress governor** (identical for every method): a load that cannot track its
  reference or cannot deliver its demanded wrench slows down,
  `h = clip(1 - max(||e_q||/0.06, r_w/0.30), 0, 1)`, passed through a 1.5 s
  first-order lag so the reference acceleration stays bounded. This is what turns an
  asymmetric *physical* disturbance into a *coordination* problem.
* **Disturbance** (predeclared, identical across methods): load 2's resistance wrench
  is scaled by 2.6 for 12 s starting at `t = 22 s` — enough to exceed its certified
  capacity at nominal speed, so the load must slow and the other must be told to wait.

### Information layer

Exactly the reduced interconnection of Prop. "stability", driven by the **real**
time-varying graph:

```
a_dot = -m_y a + theta_1 b + Q^T w(t)
b_dot =  theta_2 a - c Q^T L_geo(q(t)) Q b        a, b in R^5
```

Two uses, kept apart:

* **certified replay** (`w = 0`, restarted at `t = 6 s` from the common
  region-split initial condition) — the object the time-varying corollary covers;
* **mission layer** (`w_i = 4 (s_{k(i)} - mean_k s_k)` on lifters) mapped downstream
  to `u_i = u_0 - [Q a]_i`, saturated to `[0, 2]`, with load `k`'s reference progress
  speed the mean command of its lifting robots.

The downstream map is an **illustrative realization**, declared before the
comparison and identical across methods. It is not a theorem.

## 5. Predeclared hypotheses and the deterministic result (seed 0)

| | | result |
|---|---|---|
| **H1** | `\|Bz-y*\|_inf <= 1e-8` and `\|V-V*\| <= 1e-8` for WISE | **pass** (both exactly 0) |
| **H2** | integer-fiber `Gamma > 0` at PROD, `<= 1e-6` at WISE | **pass** (0.3989 vs 0) |
| **H3** | `lambda2(L_geo(t)) >= lambda2(Lbar) - 1e-9` throughout the window | **pass** (min margin +0.0367 WISE, +0.0391 HARD; max tube violation 0) |
| **H4** | when `lambda2(Lbar) >= sigma_req`, `\|[a,b]\|_P` decays no slower than `alpha(sigma_req)` | **pass** (fitted 0.156 / 0.108 vs certified 0.0397) |
| **H5** | WISE beats PROD on synchronization and tracking | **pass** |
| **H6** | *does* WISE beat HARD? (not assumed) | **partly — see below** |

```
method       lam2(Lbar) alpha_cert  alpha_fit   V/V*  |Bz-y*| min lam_geo  max|s1-s2|  mission
PROD            0.00000   -0.20711   -0.20711  1.000  0.0e+00     0.00000      0.7343  FAIL
HARD            0.33646    0.03967    0.10822  1.000  0.0e+00     0.37559      0.0539  ok
WISE            0.39888    0.03967    0.15557  1.000  0.0e+00     0.43559      0.0538  ok
SCALAR          0.39888    0.03967    0.15557  1.000  0.0e+00     0.43559      0.0538  ok
RANDOM-FIBER    0.00000   -0.20711   -0.20711  1.000  0.0e+00     0.00000      0.6415  FAIL
```

Under PROD the certified information layer **grows by a factor `5.3e5`** over the
window, load 1 never completes its path (`t_complete1 = n/a` vs 57.7 s for WISE), and
the terminal synchronization error is 0.734 — a mission failure at the *same*
productive aggregate, the same productive value and the same certified wrenches.
RANDOM-FIBER (`lambda2 = 0`, terminal sync 0.641) shows that being on the degenerate
fiber is not by itself enough: it is the *selection* on the fiber that matters.

**H6, honestly.** WISE's larger margin buys a genuinely faster certified information
layer: fitted decay 0.1556 vs 0.1082 (+44 %), terminal `\|[a,b]\|_P` ratio `1.31e-5`
vs `4.07e-4` — a factor of 31. It does **not** measurably improve transport on this
world: max synchronization error 0.0538 vs 0.0539, settling time 2.30 s vs 2.25 s
(HARD marginally *faster*), pose RMSE equal to three digits. Once the threshold is
cleared, the extra margin shows up in the information layer, not in the loads. That
is the honest finding and it should be written as found, not as hoped — with the
caveat that a single world cannot separate two methods this close; the 30-seed
campaign is what would decide it.

**SCALAR, honestly.** On this *integer* instance the scalarized selector returns the
WISE assignment for every `eps` on the grid, with zero aggregate drift. That is not a
contradiction of E4 — the drift there is a property of the *relaxed* problem, where
`z` can move continuously off the fiber — but the supplementary claim should say so.

## 6. Mismatches and failures found by the audit

### MISMATCH-1 (material — affects the paper's E2 numbers and Assumption (iv))

`exp_flagship._laplacian` builds the always-on short links from every robot's **home**
position and then adds the relaying robot's long-range links from the **gap site**. The
relaying long robot therefore contributes edges from two positions at once: it keeps
its `exp(-1/2) = 0.607` links to `S1` and `S2` at `(0,±1)` while simultaneously
bridging from `(4,0)`.

Recomputing the *same* composition on a physically consistent single-position graph:

| | frozen record | single position |
|---|---|---|
| `lambda2` (WISE) | **0.414710** | **0.254393** |
| margin over `sigma_dyn = 0.25` | 0.1647 | 0.0044 |
| `alpha` | 0.128 | ≈ 0.0035 |

The qualitative claim survives (still `> sigma_dyn`, still `>> 0`), but the quoted
margin is ~38× too large and the quoted rate ~37× too fast. More importantly, this is
exactly the failure mode Assumption (iv) forbids: `L_0` must be a
*decision-independent lower bound* on the physical graph, and a `L_0` containing the
home short links of a robot the decision then sends to a relay site is not a lower
bound at that decision — so `L_geo(q) >= Lbar(z_hat)` can fail. The fix inside the
framework is to take `L_0`'s tube infima over each robot's union of admissible
positions **across all its actions** (which removes those edges), or to certify
`Lbar` per assignment as this experiment does. Note the paper already flags the
10,000/10,000 Loewner check as "a construction check of Lemma bridge, not an
independent test of Assumption (iv)" — this is why that caveat matters.

**Applied to the manuscript.** Sec. II now states explicitly that `L_0`'s tube infima
must be taken over each robot's admissible positions *across all its actions*, not over
its idle pose — otherwise a robot the decision sends to a relay site takes its short
links with it and `L_0` stops being a lower bound. With that reading Assumption (iv)
holds and the closed-loop run confirms `L_geo(q(t)) ⪰ L̄(ẑ)` at every operational step.
The manuscript no longer quotes `λ2 = 0.41`; it quotes the tube-certified `0.399`.

### MISMATCH-2 (provenance — no script produces the quoted flagship dimension)

Sec. III and Sec. V call the flagship "the two-load flagship (`N=12`)" with
`dim E = 58` (`n = 72`, `rank[A;B] = 14`). No script in the repository builds a
two-load `N=12` instance:

* `experiments/exp_flagship.py` builds an explicit **`N=6`**, two-load construction
  (`L1,L2,S1..S4`) — it is the only thing that writes `generated/flagship.json`;
* `generated/fiber_certificate.json` is a **single-load (`M=1`)** `N=12` `two_region`
  instance with `dim E = 59`, `n = 72`, `rank[A;B] = 13`;
* `scenarios.two_region` hard-codes `M = 1`.

So `dim E = 58` for "the two-load flagship" was not reproducible from this repository.
**Fixed in the manuscript**: Sec. III now attributes `dim E = 59` (`n = 72`,
`rank[A;B] = 13`) to the `N = 12` instances that actually produce it, and the flagship
is described as what it is — an explicit 2-long + 4-short, two-load construction. Sec. V
no longer calls the flagship `N = 12`.

### MISMATCH-3 (`sigma_req` is not defined for the flagship)

Three different `(sigma_dyn, sigma_req)` pairs appear in the repository:
`modal_stability.json` uses `sigma_dyn = 0.25` and no `sigma_req`; `exp_physical.py`
uses `0.20` / `0.30`; `fiber_certificate.json` reports `sigma_req = 1.087` on a
different instance. The flagship record itself carries neither. This experiment
declares `sigma_dyn = 0.25` (from the E5 gains) and `sigma_req = 0.30`.

### MISMATCH-4 (the flagship is exactly on the actuation boundary)

Both flagship compositions saturate their certified capacity exactly
(`3 = 2+1 = 1+1+1` in `f_x`, with the torque at the vertex of the inner 24-gon). A
closed loop needs headroom, hence the declared 55 % nominal operating point.

Worse, load 2's two-slot geometry **cannot produce a pure torque with zero net
force**: `tau = 0.5 f_x^{gL}` and `|f^{gS}| <= 1`, so any `tau = 0.55` forces
`f_x^{net} >= 0.1`. A constant external torque on load 2 is therefore structurally
unrejectable and drives an unbounded position drift. This is a real modelling limit
of the flagship's contact geometry, and it is why the experiment models the
resistance as collinear with the certified demand rather than as an independent
torque bias.

### MISMATCH-5 (the connectivity-optimal relay site is not the geometric centre)

On the declared grid the east site beats the centre (`0.3989` vs `0.3877`) because
load 1 carries three lifters and load 2 only two, so the graph is not
left–right symmetric. Small, but it is a concrete instance of the paper's own point:
the connectivity-optimal composition is not guessable, which is what
`Gamma_E`/the SDP is for.

### Non-mismatch worth recording

`lambda2(L_geo(t)) >= lambda2(Lbar)` held at **every** recorded operational step for
every method, with a strictly positive minimum margin, and no robot ever left its
tube. The per-assignment tube-infimum construction does what Lemma "bridge" says it
does.

## 7. Monte-Carlo campaign (PHASE R11 — run, 30 paired seeds)

Each seed draws one world (load mass ±10 %, damping ±15 %, initial pose offsets inside
the tubes, disturbance amplitude `U(2.3, 2.9)`, onset `U(20, 24) s`) and **every method
runs on that same world**. The statistical unit is the seed/world; reported quantities
are paired medians, paired bootstrap CIs (20,000 resamples), success counts and the
worst-case seed. No p-values, no time-step pseudoreplication. Packet loss is
deliberately **not** modelled — there is no packet-loss model in the theory.

Mission success (`generated/robot_statistical_report.json`):

| PROD | HARD | WISE | SCALAR | RANDOM-FIBER |
|---|---|---|---|---|
| **0/30** | 30/30 | **30/30** | 30/30 | **0/30** |

Paired medians, WISE minus reference, with bootstrap 95 % CI:

| metric | vs PROD | vs HARD |
|---|---|---|
| peak `|s1−s2|` | **−0.733** [−0.749, −0.713] | **+1.7e−4** [−2.5e−4, +5.5e−4] |
| terminal `|s1−s2|` | −0.785 [−0.799, −0.770] | 0 [0, 0] |
| fitted information rate | +0.365 [+0.3647, +0.3657] | **+0.0478** [+0.0476, +0.0479] |
| `‖[a,b]‖_P` ratio | −5.3e5 | −3.9e−4 |
| min `λ2(L_geo)` | +0.4356 | +0.0600 |
| min transfer margin | +0.0367 | −0.0024 (both > 0) |
| load-2 pose RMSE | −0.0100 | −7.4e−5 |
| settling time | n/a (PROD never settles) | +0.1 s [0, +0.2] |

Two readings matter. Against PROD the effect is total and consistent: 30/30 versus
0/30, with the peak synchronization error a full 0.73 of the path lower. Against HARD
the transport difference is **statistically indistinguishable** (CI straddles zero) while
the information-layer rate is **strictly** higher on every seed. That is the honest
answer to H6: once `sigma_req` is cleared, extra spectral margin buys information-layer
speed, not load-level performance.

## 7b. Relay-attenuation robustness sweep (what the *extra* margin buys)

E2 leaves an obvious question: if HARD already clears `sigma_req`, why maximize the margin?
The answer is graceful degradation, and it is testable without tuning. A single declared
scalar `a in (0,1]` multiplies **every** gated relay conductance, in `L̄` and `L_geo`
alike, on a fixed grid (`run_margin_sweep.py`, `make robot-sweep`). Nominal reserves:
`m_λ(HARD) = 0.336 − 0.30 = 0.036`, `m_λ(WISE) = 0.399 − 0.30 = 0.099`.

| | last `a` clearing `sigma_req` | last `a` clearing `sigma_dyn` |
|---|---|---|
| PROD | never | never |
| HARD | **0.89** (11 % attenuation) | 0.73 (27 %) |
| WISE | **0.73** (27 %) | 0.60 (40 %) |

Both crossings track `sigma/λ2(L̄)` per method (0.30/0.336 = 0.893, 0.30/0.399 = 0.752),
i.e. the degradation is exactly what the reserve predicts. At `a = 0.60` the closed loop
confirms it dynamically: HARD's fitted information rate is **−0.0079** (diverging,
`λ2 = 0.208 < sigma_dyn`) while WISE's is **+0.0289** (`λ2 = 0.253 > sigma_dyn`). The
mission still completes for both in the nominal world across the swept range, so the paper
claims only the certificate crossings.

Attenuation is a module-level scalar restored by every sweep, and two tests
(`test_relay_attenuation_is_neutral_by_default`, `test_attenuation_scales_only_relay_links`)
guarantee it cannot leak into any other reported number.

## 8. Paper integration (applied — the manuscript is now 6 pages)

What changed, and where:

* **Both** standalone figures (`fig_flagship`, `fig_modal`) were replaced by one
  full-width hero figure `fig_robot_hero.pdf` (`make_hero.py`), panels
  (a) transport + active graph, (b) `λ2(L_geo(t))` vs the certified surrogate,
  (c) information layer + synchronization error. There is exactly one main figure.
* Sec. III gained **Corollary "Moving graph, unchanged rate"** (the
  common-Lyapunov result of `TIME_VARYING_STABILITY_PROOF.md`), with a four-line proof
  sketch; the conclusion and abstract now state the closed-loop result instead of
  listing closed-loop transport as future work.
* **E2 was replaced**: the standalone matrix-exponential modal experiment became the
  robot-level closed loop, with the 30-seed paired campaign.
* Compressions to pay for it: the eigenspace/finite-difference block, the Loewner
  construction check, the cluster-bootstrap and worst-case-radius details, the
  scalarization paragraph, the `d_net` per-seed detail, several proof sketches, the
  contributions list (itemize → paragraph), and the abstract.
* Journal names in `references.bib` were abbreviated to IEEE style (also a compliance
  fix, not only a space one).

Claim made:

> A planar closed-loop experiment makes that physical: two assignments with identical
> served aggregate, productive value and certified wrenches differ only in composition,
> and only the WISE one holds the certified communication bound and resynchronizes two
> disturbed loads (30/30 paired worlds against 0/30).

Not claimed anywhere: full nonlinear robot–load stability theorem, hardware validation,
actuator-level proof, distributed assignment convergence, robustness to unmodelled
communication phenomena.

Build gates (`python paper/checkbuild.py 6`): **6 pages, 0 undefined references, 0
overfull boxes > 5 pt, no stray rendered markers.**

### Claim-consistency fixes in the freeze pass

* **Abstract** no longer implies WISE is the *only* success. It compares explicitly against
  the productively identical but disconnected baseline ("30/30 versus 0/30"), and says "the
  same certified **lower rate bound** holds under a moving graph" rather than "rate
  unchanged".
* **Two different baselines were sharing the word *scalarized*.** `scalar_capacity`
  (`src/wise_mr/baselines.py`) is a greedy heuristic that meets `‖w_dem‖` by scalar force
  totals and scores 0 % certified service; E4's `V + eps*lambda2` is a genuine connectivity
  scalarization and succeeds 30/30 in E2. Both are now named and distinguished in the text.
* **E2 now specifies the loop**: load dynamics, the 20 Hz trajectory-PD demand, the
  minimum-effort inner-zonotope allocator, and the `(a,b) -> path-progress` map, plus
  wrench diagnostics — outside the declared disturbance window the demand is met to
  `6e-8` with **zero** infeasible control steps; inside it the team saturates
  (min actuation slack 0, peak residual 0.82 N) and the governor slows the load.
* **"Robot-level" was an overclaim** (no individual robot locomotion dynamics are
  simulated). Renamed everywhere to *planar closed-loop transport simulation with rigidly
  attached robots*.
* **Figure caption**: "best lifter" → "highest-productive-capacity type" (the model keeps
  `c_tau` distinct from wrench capability); "releasing one for three S" → "replacing one L
  by two additional S and sending the released L to a relay"; "two compositions, two
  missions" → "three selectors, panel (a) highlights the PROD–WISE exchange"; the surrogate
  lines are identified as same-coloured thin lines and PROD's clipping is annotated.
* **Flagship size** stated explicitly: `N = 6`, `|A| = 12`, hence `|A|^N = 12^6`.
* **Statistics**: `+0.0478 [0.0476, 0.0479]` (the rounded point estimate previously sat
  above its own CI), with the paired-bootstrap unit named.
* **Wrench binding**: the `Gamma_E` claim was dropped from that sentence. It is evaluated at
  a relative-interior reference point in `exp_wrench_bind.py`, so it was not tautological,
  but the cleaner statement — the facet preserves the fiber dimension and shortens its
  global reach — carries the insight without needing the caveat.
