# How `Gamma_E` is actually computed, and over which set

Audit of the question a careful reviewer asks: the paper writes the wrench-feasible set
as `X_f = {z : H_w z >= d}` and lets `Gamma_E` use its *active rows* `G_I`, while
Lemma 1 presents wrench feasibility through a *lifted* zonotope system in `(z, xi)`.
Where do those rows come from, and is the set they describe the projection of the lifted
one?

Generated evidence: `generated/gamma_computation_audit.json`
(`python experiments/exp_gamma_audit.py`).

---

## Answer: PATH A — an explicit H-representation, no facet enumeration

The convex selection layer (Stage-2 SDP, fiber dimension, `Gamma_E`, price sweep)
optimizes over

```
X_f^sup = { z : H_w z >= d },
H_w[(k,l),(i,k,h)] = W[i,k,h,l] = h_{U_tau^in}( A_kh^T eta_kl ),      d_kl = eta_kl^T w_dem_k
```

built in `WiseProblem.wrench_matrix()` from the support tensor `W` of
`wrench_tensor.build_wrench_tensor`. This is an **explicit** inequality description with
`M*P` rows — 10 rows on the reported instances (`M=1`, `P=10`: eight in-plane force
directions and two pure torques). Nothing is eliminated, projected or enumerated, so:

* **no facet enumeration is required**, and the paper's claim to that effect is correct
  for this system;
* the active rows are read off directly — `nullspace.active_inequalities` returns the
  active nonnegativity rows (`z_ia ~ 0`) plus the active support rows
  (`s_kl ~ d_kl`), and `Gamma_E` uses exactly those;
* the cost is `O(M*P*N*A)` to assemble and trivial to evaluate.

## But `X_f^sup` is **not** the projection of the lifted system

`W_k(z) = (+)_{tau,h} z_tkh A_kh U_tau^in` is a zonotope, and `w_dem_k in W_k(z)` holds
**iff** the support inequality holds in *every* direction. Testing `P` sampled directions
is necessary, not sufficient, so

```
X_f^exact  ⊆  X_f^sup ,
```

i.e. the convex layer optimizes over an **outer relaxation**. It can propose a `z` that
the exact test would reject. The lifted system of Lemma 1 is the *exact* object; the
support polytope is a tractable superset of its projection, not that projection.

### How the gap is closed

Downstream, not upstream. Every reported assignment is re-certified by the **exact**
inner-zonotope LP (`wrench_tensor.certify_membership_lp`, facet-normal form, or its
G-representation twin `certify_membership_lift`) before it is accepted. The convex layer
proposes; the exact LP disposes. This is the "direct re-certification" the experiments
section already describes — the audit simply names what it is protecting against.

## Why the subtlety does not touch any reported number

Measured at the relative-interior fiber point used by `exp_gamma`/`exp_fiber`, on all
audited seeds (3, 5, 7, 11, 13) and in `generated/fiber_certificate.json`:

| quantity | value |
|---|---|
| support rows in `H_w` | **10** |
| active support rows at `z̄` | **0** |
| active inequality rows total (`G_I`) at `z̄` | **0** |
| exact inner-zonotope LP feasible at `z̄` | **yes**, 5/5 |

So on every instance whose `Gamma_E` the paper reports, **no inequality of `X_f^sup` is
active**: `G_I` is empty, the tangent cone is the subspace
`T_E(z̄) = ker A ∩ ker B`, and `Gamma_E` is computed on an exactly characterized set. The
projected-versus-lifted question is *vacuous* there.

The one place it is not vacuous is the `tau_d` sweep of E4, where the first support row
activates at `tau_d = 8`. That is reported as a support-row activation, and the
conclusion drawn from it — the fiber's dimension is preserved while its reach shortens —
depends only on the active-row count and `Lambda_E`, both computed in the same
representation.

A search for a strict-relaxation witness (a `z` satisfying `H_w z >= d` but rejected by
the exact LP) found none in 4,000 random feasible-simplex draws per seed. That is
**not** evidence of exactness — the gap region is thin and random draws are a weak probe —
and no exactness claim is made anywhere.

## What the paper now says

Lemma 1 no longer asserts that `X_f` is the projection of the lifted system. It states
the lifted representation as the **exact membership test used for certification**, and
separately introduces the support polytope as the tractable **outer relaxation the convex
layer optimizes over**, with re-certification closing the loop. See
[`GAMMA_LIFTED_EQUIVALENCE_PROOF.md`](GAMMA_LIFTED_EQUIVALENCE_PROOF.md) for what PATH B
would have required and why it is not needed here.

## Tests

`tests/test_gamma_representation.py`:

* the support system is explicit — `H_w` has exactly `M*P` rows for every seed;
* `X_f^exact ⊆ X_f^sup` on random draws (every exactly-feasible point satisfies the
  support inequalities);
* `G_I` is empty at the reported relative-interior fiber points, so `T_E = ker A ∩ ker B`;
* with `G_I` empty, the closed-form `Gamma_E = ||Pi_{N_p} g_lambda||` agrees with the
  general eigenspace SDP;
* the exact facet-normal LP and the lifted G-representation LP agree on the flagship
  compositions.
