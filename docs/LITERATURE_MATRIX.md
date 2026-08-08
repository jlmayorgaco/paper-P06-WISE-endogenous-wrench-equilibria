# Novelty audit — what is and is not new in WISE

Written to survive the sharpest available objection, which is not about robotics:

> **Reviewer (convex analysis):** "Theorem 2 is first-order optimality of a concave
> function over a convex set. That is textbook."

**This objection is partly correct and must be conceded in the paper, not fought.**
The mechanics of the proof *are* standard: a finite concave `f` on a polyhedron has a
directional derivative that overestimates increments, so a directional test at a point
certifies global optimality. Nothing in that chain is new.

## What is actually being claimed

The claim is **not** a new optimality principle. It is that a specific, non-obvious
object is the right thing to apply it to:

| Layer | Content | Novel? |
|---|---|---|
| First-order optimality of concave `f` over convex `C` | classical | **no** |
| `λ₂` concave in affine Laplacian weights | Fiedler 1973; Ghosh–Boyd 2006 | **no** |
| Directional derivative of a repeated min-eigenvalue over its eigenspace | Lewis 1996; Overton 1992 | **no** |
| Maximising `λ₂` by SDP/LMI | Ghosh–Boyd 2006 | **no** |
| Lexicographic / hierarchical selection over a solution set | classical VI | **no** |
| **That the wrench-feasible productive-optimum set `E` is a positive-dimensional polytope, and that its tangent cone carries the entire zero-loss connectivity question** | — | **yes** |
| **`Γ_E` as a computable scalar deciding whether *any* free connectivity remains, with `Γ_E = 0` certifying that the productive optimum has been fully exploited** | — | **yes** |
| **The separation `dim E` vs `d_net`: productive degeneracy vastly exceeds network-visible degeneracy** | — | **yes** |

The defensible sentence is therefore:

> To the best of our targeted review, existing communication-aware allocation methods
> impose connectivity as a constraint, scalarise it against task utility, or repair a
> backbone after the fact. None characterises the geometry of the *exact* wrench-feasible
> productive-optimum set and uses its neutral directions to decide whether stabilising
> connectivity is attainable at exactly zero productive loss.

**Never write "first" without that hedge**, and never claim novelty for Fiedler cuts,
`λ₂` concavity, LMI connectivity design, or lexicographic optimisation as such.

## Comparison matrix

| Approach | Wrench | Heterog. | Connectivity | Decision-induced graph | Preserves primary optimum exactly | Characterises free directions | Control consequence |
|---|---|---|---|---|---|---|---|
| Population games / distributed NE seeking (Sandholm; Quijano; Barreiro-Gomez; Martínez-Piazuelo; Koshal; Ye–Hu; Gadjov–Pavel) | ✗ | partly | graph **given** | ✗ | n/a | ✗ | convergence, not connectivity |
| Connectivity-constrained MRTA (Williams matroid; Lin) | sometimes | ✓ | hard constraint | ✓ | ✗ (any feasible point) | ✗ | ✗ |
| Connectivity maintenance / control (Zavlanos) | ✗ | partly | ✓ | ✓ | n/a | ✗ | ✓ |
| Spectral network design (Ghosh–Boyd) | ✗ | ✗ | ✓ (objective) | edge weights | n/a | ✗ | ✗ |
| Weighted-sum `V + ε λ₂` | maybe | ✓ | ✓ | ✓ | **✗ — see E6** | ✗ | ✗ |
| Heterogeneous MRTA under physical constraints (Calvo–Capitán) | ✓ | ✓ | relays | ✓ | ✗ | ✗ | ✗ |
| Backbone reconfiguration (Santos et al.) | ✗ | ✓ | ✓ | ✓ | not guaranteed | ✗ | ✗ |
| Lexicographic task allocation (industrial MRTA) | ✗ | ✓ | ✗ | ✗ | ✓ (by construction) | ✗ | ✗ |
| **WISE** | ✓ | ✓ | ✓ | ✓ | **✓** | **✓ (`Γ_E`)** | **✓ (Prop. stability)** |

The two columns that no prior row fills together are the last two. That intersection —
not any single column — is the contribution.

## Why E6 matters to the novelty argument

The weighted-sum row is the one a reviewer will push hardest ("just tune ε"). E6 answers
it with data rather than assertion: aggregate drift is nonzero at *every* tested ε and
grows linearly, while productive loss stays ~0 until ε ≈ 0.1. At a maximiser `V` is flat
to **second** order but the aggregate moves to **first** order, so a small ε buys the
appearance of value preservation while already displacing `y*` — invisibly in `V`.
Lexicographic selection is not a stylistic preference; it is the only form that attains
zero drift without a weight to calibrate.

## Open risk

**References [10] and [11] are not verified against authoritative metadata.**
`calvo2025heterogeneous` lacks volume/pages/DOI; `santos2024backbone` is cited as *IEEE
Latin America Transactions* 2024, but the public record traces to arXiv:2409.16851 marked
"Submitted to". Citing a preprint as a published journal article is the kind of error a
reviewer notices. **This remains open by explicit decision and should be resolved before
submission.**
