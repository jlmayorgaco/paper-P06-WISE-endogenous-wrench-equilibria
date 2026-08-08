# PATH B: what a lifted `Gamma_E` would require, and why it is not used

The audit ([`GAMMA_COMPUTATION_AUDIT.md`](GAMMA_COMPUTATION_AUDIT.md)) selects PATH A:
the convex layer optimizes over an explicit support-inequality polytope, so the active
rows `G_I` are directly available and no projection theorem is needed. This note records
the alternative honestly — what PATH B would have demanded, and the gap that stopped it —
so that the choice is visible rather than implicit.

## The lifted fiber

With the Lemma-1 variables,

```
Ẽ = { (z, xi) :  z ∈ X,  Bz = y*,  sum_{tau,h} A_kh G_tau xi_tkh = w_dem_k  ∀k,
                 -z_tkh 1 <= xi_tkh <= z_tkh 1 }
```

and `E = proj_z Ẽ`. One would define

```
Gamma̅_E(z̄) = max { t : (d_z, d_xi) ∈ T_Ẽ(z̄, xi̅),  ||d_z||_2 <= 1,
                       U_2^T DL̄[d_z] U_2 ⪰ t I }
```

for some lift `xi̅` of `z̄`.

## The gap that makes this non-trivial

`proj_z T_Ẽ(z̄, xi̅) ⊆ T_E(z̄)` always. Equality does **not** hold for an arbitrary lift.
The box `-z 1 <= xi <= z 1` couples `d_xi` to `d_z`: if a component `xi_tkh` sits on its
bound `= z_tkh` at the chosen lift, then that bound is active and any admissible `d_z`
must satisfy `d_xi,tkh <= d_z,tkh` there. A direction `d_z` feasible for `E` can then be
blocked in the lift simply because the *particular* `xi̅` was chosen badly. Taking such a
`Gamma̅_E` at face value would report a value **below** the true `Gamma_E` — i.e. it could
certify "no free connectivity remains" when some remains.

So PATH B needs one of:

1. **a relative-interior lift.** If `xi̅ ∈ relint{ xi : (z̄, xi) ∈ Ẽ }` then no box
   constraint is active beyond those forced by `z̄` itself, and the projection is exact.
   This must be *computed and certified*, not assumed — the lift fiber is itself a
   polytope and one needs a strictly-interior point of it (e.g. by maximizing the minimum
   box slack), plus a proof that the remaining active set is exactly the one induced by
   `z̄`.
2. **optimization over the lift.** Maximize over `(d_z, d_xi)` *and* over the admissible
   `xi̅` jointly. This restores exactness by construction but is no longer a single
   convex program in the same variables, since `xi̅` and `d_xi` interact through the box.
3. **a polyhedral projection theorem** giving `proj_z T_Ẽ = T_E` under a constraint
   qualification on the lifted system (e.g. Slater plus a rank condition on the equality
   block). We did not establish one.

None of these is needed for the results reported here, so none is claimed.

## Why PATH A suffices

On every instance whose `Gamma_E` the paper reports, the active set of the support
polytope is **empty** (`G_I = ∅`, verified on seeds 3, 5, 7, 11, 13 and in
`generated/fiber_certificate.json`). The tangent cone is then the subspace
`T_E(z̄) = ker A ∩ ker B`, which is exact and representation-independent: any correct
description of the wrench constraints — projected, lifted, or support-based — yields the
same cone when none of those constraints is active. The lifted formulation would compute
the same number.

**Rule adopted.** If a future instance evaluates `Gamma_E` at a point where a wrench row
*is* active, the value computed on the support relaxation is reported as a **certified
bound**, not as `Gamma_E`: the relaxation has a larger feasible set, hence a possibly
larger tangent cone, so it yields an **upper** bound on the true `Gamma_E`. Positivity
of an upper bound does not certify that free connectivity exists; only `Gamma_E = 0`
computed on the exact cone certifies global optimality. The tests enforce that the
reported instances stay in the `G_I = ∅` regime.
