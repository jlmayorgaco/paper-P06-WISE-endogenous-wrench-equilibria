# Figure Audit — LARS 2026 revision

IEEE rules: vector PDF, ≥7pt effective type, axes with units, no redundant inner
titles, grayscale-distinguishable, theory vs data distinct styles, visible CIs,
captions state what varies / #seeds / result. Captions ≤3 sentences.

| Fig | Role | Ties to | Status | Action (task) |
|-----|------|---------|--------|--------|
| Fig 1 | concept + fiber (V flat, λ₂ varying, WISE band) | fiber geometry | concept TikZ only | add analytic 2×2 fiber panel (T9, T20, T30) |
| Fig 2 | threshold sweep: measured vs modal rate | stability | 2-case only | rate-vs-λ₂/σ_dyn (T21, T30) |
| Fig 3 | phase diagram w/ SDP boundary + hatching | existence | empirical contour only | overlay Λ⋆=σ_req, wrench/WISE hatch (T22, T30) |
| Fig 4 | method comparison + roles | selection | oracle==WISE | paired CIs; spectral-benefit ranking (T23, T24, T30) |
| Fig 5 | physical closed loop | validation | missing | snapshots + pose error + wrench + L_geo/L̄ (T25, T30) |

Orphans to remove from paper/figures: fig1_scene.pdf, fig1_concept.pdf (unused; concept is TikZ).
