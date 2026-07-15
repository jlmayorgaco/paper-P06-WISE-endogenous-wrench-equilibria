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
