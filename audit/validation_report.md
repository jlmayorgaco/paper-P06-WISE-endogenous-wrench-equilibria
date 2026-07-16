# Validation report — verifiable gates

A claim is *closed* only when (1) the formulation appears correctly in the PDF, (2) the
equations/algorithms/experiments use exactly that formulation, (3) no other section
contradicts it, (4) the result regenerates from clean code, and (5) the final PDF fits
six pages and stays legible. "Code exists" is **not** closure.

Status legend: ⬜ open · 🟨 code done, PDF not yet verified · ✅ verified against the
compiled PDF and regenerated artifact.

## Gates this round (reviewer priority order)

| # | Gate | Formulation / PDF location | Automated check | Result file | Status |
|---|------|----------------------------|-----------------|-------------|--------|
| B | Fiber neutrality certified | Fig.~2(a,b), §V-E1, Thm. `thm:degeneracy`; `d∈ker A∩ker B∩ker G_I` | `exp_fiber` asserts + JSON cert | `generated/fiber_certificate.json`, in `fig_central.pdf` | ✅ |
| C | Lexicographic vs scalarized | §V-E3 (numbers + Tikhonov bounds); global SDP, regime split | `exp_epsilon` regime split + shape asserts | `generated/epsilon_sweep.csv`, `paper/figures/fig_epsilon.pdf` (repo) | ✅ |
| D | Same-fiber unsafe vs WISE (x,y) | Fig.~2(c,d), §V-E2; same `Bz=y*`, `V*`, different `λ2(L_geo)` from pipeline | `exp_spatial` verifies aggregate equality | `generated/spatial_pair.json`, in `fig_central.pdf` | ✅ |
| E | Baselines + no "oracle" | Table I, §V-E4 (integer recovery), E5 | `exp_methods` + rename; `hard_connectivity` pending | `generated/methods_comparison.csv` | 🟨 |
| F | Clean regen + manifest + 6pp | whole PDF | `reproduce.py` manifest of CSV/PDF hashes | `generated/manifest.json` | 🟨 |

### Verified numbers (from the compiled 6-page PDF, regenerated artifacts)
- **B**: `dim E=59`; `‖Ad‖=3e-16, ‖Bd‖=6e-16, ‖G_Id‖=0`; `max|V−V*|=1.4e-14`,
  `max‖Bz−y*‖=7.1e-15`; λ2 crosses σ=0.42 on the fiber.
- **C**: 8/8 seeds free (Λ_E≥σ); weighted-sum loss 0→0.049, drift 0→0.307 as ε:1e-3→1,
  both under the Tikhonov envelopes; WISE = 0 loss, 0 drift.
- **D**: seed 0, relay robot #10 (long-range); aggregate identical; both wrench-feasible;
  λ2(L_geo) 0.03 (unsafe) → 0.88 (WISE), σ=0.25.
- **E (integer recovery)**: fluid 100%, randomized-rounding 100%, argmax 75%, welfare gap −0.009 (16 seeds).

## Per-gate detail

### B — Fiber neutrality (imprescindible)
Certifies the swept direction is a genuine fiber direction: `‖Ad‖, ‖Bd‖, ‖G_Id‖` at
machine precision, and along the whole admissible α-range `max|V(z(α))−V*| < 1e-8`,
`max‖Bz(α)−y*‖ < 1e-8`, while `λ2(L̄(z(α)))` crosses `σ_req`. Numbers in the JSON cert
and figure caption, not just a visually flat curve.

### C — Lexicographic vs scalarized (imprescindible)
`V_ε = V + ε λ2` solved as a **global** CVXPY SDP (LMI `QᵀL̄(z)Q ⪰ tI`), not local λ2
optimization. Seeds classified free/costly/impossible by `Λ_E` vs `σ_req`; the
zero-productive-loss claim uses **free** seeds only. Physical ε for the Tikhonov bounds
`0 ≤ V*−V(z_ε) ≤ ε·osc(λ2)` and `‖Bz_ε−y*‖ ≤ √(2ε·osc(λ2)/α)`, normalized ε̄ for the
comparative plot. Single vectorization convention with shape asserts.

### D — Same-fiber unsafe vs WISE (high impact)
Two compositions on the **same optimal fiber** (`Bz=y*`, `V=V*`), differing only in
`λ2(L_geo(q))`. Positions/roles/edges come from the real pipeline
`z* → round_argmax → role→pose → geometric_laplacian`, not hand-placed. Panel A:
long-range robots all lift, relay site empty, graph fragmented (`λ2<σ_req`). Panel B:
one short robot lifts, the long-range robot relays, bridge appears (`λ2≥σ_req`).

### E — Baselines
Add `hard_connectivity` (`max V s.t. λ2(L̄(z))≥σ_req`) to quantify what lexicography adds
over a hard constraint (same V* in the free regime, but not necessarily max spectral
margin among productive optima). Rename the multistart heuristic away from "oracle".

### F — Clean regeneration + manifest
`reproduce.py` writes `generated/manifest.json` with SHA-256 of every CSV/JSON/PDF/TeX
artifact it emits, and fails if the paper is not exactly 6 pages, has undefined refs, or
overfull boxes. Central figure is composed by `make_central_fig.py` after `exp_fiber` and
`exp_spatial`.

## Consistency round (reviewer P0s, verified in the 6-page PDF)
Closed against the compiled PDF: (1) dim E reconciled to **59** with the full rank ladder
`n=72, rank[A;B]=13, G_I=∅` (certificate `fiber_certificate.json`), theory text corrected;
(2) false spectral regulariser claim removed — single-Fiedler use gated on a real spectral
gap, PSD supergradient otherwise; (3) robot↔candidate-site bijection `π` and
`L_geo^π = P_π^T L_geo P_π` defined, transfer applies after integer recovery; (4) Fig. 1
says *served-capacity aggregate* `Bz=y*` with wrench realizability noted separately;
(5) lifted-SDP vs projected-KKT reconciled — KKT/flow declared on the projected
representation via a wrench separation oracle; (6) `ε_L, ε_est, δ_num` formalized in
Lemma bridge, exact SDP has `ε_est=δ_num=0`; (7) tie-break made strongly convex (unique);
(8) Thm 3 necessity restricted to the fully isotropic mode-aligned case, dimensions +
`ϑ>0` added; (9) unsafe graph now genuinely disconnected (`λ2≈0`, delta=0); (10) "Schur"
→ "small-gain" in conclusion; (11) scalarization claim softened to "no exact-preservation
guarantee"; (13/14) integer gap sign fixed (`V_relax−V_int=+1.8%`), 30 seeds, single-draw
(66%) vs best-of-30 (100%) reported; (15) "perfect recall" → attainment-gap framing;
(16/30) Table uses `σ_req`, WISE-PD/WISE-SDP; abstract sobered (no unicycle/stability-
boundary claims), promotional phrases removed; Sion reference added.

## Still open (honest ledger — not claimed as closed)
- **E (hard-connectivity baseline)**: `max V s.t. λ2(L̄(z))≥σ_req` not yet added to
  Table I. It would quantify what the lexicographic order adds over a hard constraint
  (same V* in the free regime, but not necessarily maximal spectral margin among
  productive optima). The "oracle" rename is done.
- **Six-page budget**: to fit, the ε-sweep and phase-diagram figures live in the
  repository; their claims appear in the paper as numbers (E3) and a recall statement
  (E5). The central figure carries the two highest-value stories (fiber + spatial).
- **χ-threshold sweep, scalability (N=8..80, runtime), external connectivity-aware
  baseline, uniform time-varying corollary**: deferred; not implemented, not claimed.

## Round 3 (5 residual P0s + comparative exchange + temporal figure) — done, in PDF
- **5 residual P0s**: (1) Thm 3 isotropic case replaced by a scalar mode-aligned Fiedler
  block (dimensionally valid); (2) integer-recovery claims explicit everywhere (single
  66%, best-of-30 100%, argmax 67%); (3) Table I split into existence vs. attainment
  blocks; (4) occupancy dual ρ added to the primal-dual flow (15); (5) eigen-residual
  stopping rule. Theorem 3 → Proposition 2 (Information-layer stability).
- **E2 comparative-advantage exchange** (Fig 2b,c): matched long/short pair of equal
  capacity; unsafe = long lifts + short relays (short range, λ2≈0); WISE = short lifts +
  long relays (λ2=0.83). Verified `aggregate_identical`, `V_identical`,
  `active_count_identical`, both wrench-feasible (`spatial_pair.json`). From the real
  solver→rounding→pose→L_geo^π pipeline.
- **Temporal transport figure** (Fig 3, `fig_transport.pdf`): closed-loop transport under
  the WISE composition — pose error, normalized wrench residual, and
  λ2(L_geo^π)/λ2(L̄)/λ̂2 all above σ_dyn=0.2 (min 0.58); distributed estimator tracks
  within 0.011. 3-figure layout (concept, Fig 2 = certificate+exchange, Fig 3 = transport).

## Round 4 (hard-connectivity baseline + consistency + visual) — done, in PDF
- **Hard-connectivity baseline** `max V s.t. λ2≥σ_req` (`_hard_conn_assignment`, exact SDP)
  added to Table I with a productive-gap column: both it and WISE-SDP reach V* in the free
  regime, but WISE attains margin +0.99 vs +0.15 — the value of the lexicographic order.
  Removed the weak `random` baseline; representative SDP runtime (30–60 ms, N=12) in E5.
- **Robot-level vs type-level**: (15) declared a type-population flow (aggregate of
  per-robot revisions), not a coordinator; `Π_X0` = local per-type budget simplices,
  occupancy via ρ; dropped the "two robots cannot seize a slot" over-claim.
- **Projected-KKT paragraph removed** (kept one sentence: exact = lifted, heuristic =
  projected separation oracle).
- **Unicycle + load dynamics equation** added to §V.
- **Fig 3 restacked** vertically (shared time axis), + σ_req, σ_dyn, and the Weyl bound
  λ2(L̄)−ε_L, with pose/wrench tolerance bands.
- Fig 2 λ2 shown as `<10⁻⁸`; abstract "66% of the 30 seeds"; promotional phrases removed;
  Lemma (unique aggregate) folded into Theorem 1; quotient clause dropped.

## Still open — scoped, not yet done
- **Fig 1 (concept TikZ) regeneration** to the comparative exchange (still shows the old
  "two fates / same delivered wrench" content) — the one remaining visual inconsistency.
- **Notation deduplication** (H, M, V, F, ξ, ν, α, ρ, A, d, Z) — post-freeze.
- **Derive Prop 2 from the implemented dynamics** (journal-level); **Fiber Refinement
  Principle** generalization; **Zenodo/DOI archival** replacing script names.
