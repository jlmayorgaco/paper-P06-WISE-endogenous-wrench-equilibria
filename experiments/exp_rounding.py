"""E-rounding: structured integer recovery beats independent rounding.

Four recovery methods from the fluid WISE optimum z*:
  1. argmax              -- each robot commits to its argmax action;
  2. independent RR      -- each robot samples its own action independently;
  3. dependent RR        -- occupancy-aware sampling (a contact slot is taken at most once;
                            collisions resample among still-available actions);
  4. dependent + repair  -- after dependent RR, a greedy neutral-repair local search moves
                            marginal lifters/idlers to relay while keeping wrench feasibility,
                            raising lambda_2 toward sigma_req; then exact re-certification.

We report, over 30 seeds (N=12): single-draw certified rate, best-of-30 rate, the median
number of draws to first success, and the productive (rounding) gap. A CDF of draws-to-
success goes to paper/figures/fig_rounding.pdf. Structured recovery should reach 30/30 with
fewer draws than independent rounding.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import baselines, metrics, scenarios  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

N_DRAWS = 30
SEEDS = list(range(30))


def _wrench_ok(prob, x):
    return float(prob.max_residual(x)) <= 0.05


def _dependent_draw(prob, z, rng):
    """Occupancy-aware rounding: each contact slot is claimed by at most one robot."""
    A, H, M = prob.A, prob.H, prob.M
    slot_taken = set()
    x = np.zeros((prob.N, A))
    order = rng.permutation(prob.N)
    for i in order:
        p = np.maximum(z[i], 0.0); p = p / p.sum() if p.sum() > 0 else np.ones(A) / A
        avail = np.ones(A, bool)
        for a in range(M * H):                       # contact slots must be exclusive
            if a in slot_taken:
                avail[a] = False
        pa = p * avail
        a = int(rng.choice(A, p=pa / pa.sum())) if pa.sum() > 0 else int(np.argmax(p))
        x[i, a] = 1.0
        if a < M * H:
            slot_taken.add(a)
    return x


def _repair(prob, x, max_steps=12):
    """Greedy neutral repair: move a marginal lifter/idler to relay to raise lambda_2 while
    keeping wrench feasibility. Trades productive value for connectivity (a rounding loss)."""
    x = x.copy()
    ridx = prob.relay_index
    for _ in range(max_steps):
        if metrics.certified(prob, x):
            return x
        base = prob.lambda2(x)
        best, best_lam = None, base
        for i in range(prob.N):
            if x[i, ridx] == 1.0:
                continue
            cand = x[i].copy(); x[i] = 0.0; x[i, ridx] = 1.0
            if _wrench_ok(prob, x) and prob.lambda2(x) > best_lam + 1e-9:
                best, best_lam = i, prob.lambda2(x)
            x[i] = cand                              # revert trial
        if best is None:
            break
        x[best] = 0.0; x[best, ridx] = 1.0
    return x


def _method(prob, z, method, rng):
    """Return (x_int, draws_to_success or None)."""
    if method == "argmax":
        x = metrics.round_argmax(prob, z)
        return x, (1 if metrics.certified(prob, x) else None)
    for k in range(1, N_DRAWS + 1):
        if method == "independent":
            x = np.zeros_like(z)
            for i in range(prob.N):
                p = np.maximum(z[i], 0.0); x[i, rng.choice(prob.A, p=p / p.sum())] = 1.0
        elif method == "dependent":
            x = _dependent_draw(prob, z, rng)
        elif method == "dependent+repair":
            x = _repair(prob, _dependent_draw(prob, z, rng))
        if metrics.certified(prob, x):
            return x, k
    return x, None


def run(seeds=SEEDS):
    methods = ["argmax", "independent", "dependent", "dependent+repair"]
    stats = {m: dict(single=0, best=0, draws=[], gap=[]) for m in methods}
    for sd in seeds:
        prob = scenarios.two_region(seed=sd, N=12, nu=0.5, tau_d=3.0, bridge_gain=3.0)
        z = np.maximum(baselines.wise_primal_dual(prob, max_iters=4000).x, 0.0)
        v_relax = float(prob.productive_value(z))
        for m in methods:
            rng = np.random.default_rng(sd + 5000)
            # single-draw success = first draw certified
            x1, _ = _method_single(prob, z, m, np.random.default_rng(sd + 7000))
            stats[m]["single"] += int(metrics.certified(prob, x1))
            x, k = _method(prob, z, m, rng)
            ok = metrics.certified(prob, x)
            stats[m]["best"] += int(ok)
            if ok:
                stats[m]["draws"].append(k)
                stats[m]["gap"].append((v_relax - float(prob.productive_value(x)))
                                       / (abs(v_relax) + 1e-9))
    # loss breakdown: dependent rounding alone vs the connectivity repair (first draw)
    pre_loss, post_loss, zero_cost = [], [], 0
    for sd in seeds:
        prob = scenarios.two_region(seed=sd, N=12, nu=0.5, tau_d=3.0, bridge_gain=3.0)
        z = np.maximum(baselines.wise_primal_dual(prob, max_iters=4000).x, 0.0)
        v_relax = float(prob.productive_value(z))
        x0 = _dependent_draw(prob, z, np.random.default_rng(sd + 9000))
        xr = _repair(prob, x0.copy())
        if metrics.certified(prob, xr):
            post = (v_relax - float(prob.productive_value(xr))) / (abs(v_relax) + 1e-9)
            pre = (v_relax - float(prob.productive_value(x0))) / (abs(v_relax) + 1e-9)
            post_loss.append(post); pre_loss.append(pre)
            zero_cost += int(post < 1e-4)
    print(f"loss breakdown: dependent-only median {np.median(pre_loss):.3f}, "
          f"after repair {np.median(post_loss):.3f}, zero-cost {zero_cost}/{len(post_loss)}")

    rows = []
    for m in methods:
        s = stats[m]
        rows.append(dict(
            method=m, n=len(seeds),
            single_rate=s["single"] / len(seeds),
            best_of_30=s["best"] / len(seeds),
            median_draws=float(np.median(s["draws"])) if s["draws"] else float("nan"),
            gap_mean=float(np.mean(s["gap"])) if s["gap"] else float("nan")))
        print(f"{m:>18}: single={rows[-1]['single_rate']:.0%} "
              f"best-of-30={s['best']}/{len(seeds)} "
              f"med_draws={rows[-1]['median_draws']:.0f} gap={rows[-1]['gap_mean']:.3f}")

    with open(GEN / "rounding.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    _figure(stats, methods, len(seeds))
    return rows


def _method_single(prob, z, method, rng):
    """One draw only (for the single-draw rate)."""
    if method == "argmax":
        return metrics.round_argmax(prob, z), None
    if method == "independent":
        x = np.zeros_like(z)
        for i in range(prob.N):
            p = np.maximum(z[i], 0.0); x[i, rng.choice(prob.A, p=p / p.sum())] = 1.0
        return x, None
    if method == "dependent":
        return _dependent_draw(prob, z, rng), None
    return _repair(prob, _dependent_draw(prob, z, rng)), None


def _figure(stats, methods, n):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    cols = {"argmax": "#888", "independent": "#c0392b", "dependent": "#e08000",
            "dependent+repair": "#2e8b57"}
    grid = np.arange(1, N_DRAWS + 1)
    for m in methods:
        d = np.array(stats[m]["draws"])
        cdf = np.array([(d <= k).sum() / n for k in grid]) if d.size else np.zeros_like(grid, float)
        ax.step(grid, cdf, where="post", color=cols[m], lw=1.4, label=m)
    ax.set_xlabel("draws $R$", fontsize=8)
    ax.set_ylabel(r"P(success $\leq R$)", fontsize=8)
    ax.set_title("recovery vs. draws", fontsize=8.5)
    ax.set_ylim(0, 1.02); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    fig.savefig(FIG / "fig_rounding.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def _repair_moves(prob, x, max_steps=12):
    """Greedy repair returning (x, n_moves)."""
    x = x.copy(); ridx = prob.relay_index; nm = 0
    for _ in range(max_steps):
        if metrics.certified(prob, x):
            return x, nm
        best, best_lam = None, prob.lambda2(x)
        for i in range(prob.N):
            if x[i, ridx] == 1.0:
                continue
            cand = x[i].copy(); x[i] = 0.0; x[i, ridx] = 1.0
            if _wrench_ok(prob, x) and prob.lambda2(x) > best_lam + 1e-9:
                best, best_lam = i, prob.lambda2(x)
            x[i] = cand
        if best is None:
            break
        x[best] = 0.0; x[best, ridx] = 1.0; nm += 1
    return x, nm


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def campaign(seeds=SEEDS, R=30):
    """Robustness of structured recovery over R independent random orders per instance
    (|seeds| x R structured draws). Reports pre/post joint-feasibility rates, worst-instance
    rate, Wilson CI, move counts, relative recovery loss, and zero-cost fraction."""
    pre_ok = post_ok = zero_cost = 0
    total = 0
    per_inst_post = []
    moves, losses = [], []
    for sd in seeds:
        prob = scenarios.two_region(seed=sd, N=12, nu=0.5, tau_d=3.0, bridge_gain=3.0)
        z = np.maximum(baselines.wise_primal_dual(prob, max_iters=4000).x, 0.0)
        v_relax = float(prob.productive_value(z))
        inst_post = 0
        for r in range(R):
            rng = np.random.default_rng(1000 * sd + r)
            x0 = _dependent_draw(prob, z, rng)
            pre = metrics.certified(prob, x0)
            xr, nm = _repair_moves(prob, x0.copy())
            post = metrics.certified(prob, xr)
            pre_ok += int(pre); post_ok += int(post); total += 1
            moves.append(nm); inst_post += int(post)
            if post:
                loss = (v_relax - float(prob.productive_value(xr))) / (abs(v_relax) + 1e-9)
                losses.append(loss); zero_cost += int(loss < 1e-4)
        per_inst_post.append(inst_post / R)
    losses = np.array(losses)
    lo, hi = _wilson(post_ok, total)
    q1, q3 = np.percentile(losses, [25, 75])
    print(f"campaign: {len(seeds)}x{R}={total} structured draws")
    print(f"  p_pre  (jointly feasible before repair) = {pre_ok/total:.1%}")
    print(f"  p_post (jointly feasible after repair)   = {post_ok/total:.1%} "
          f"(Wilson95 [{lo:.1%},{hi:.1%}])")
    print(f"  worst-instance post rate = {min(per_inst_post):.1%}; "
          f"all-instance min>=1 draw = {all(p > 0 for p in per_inst_post)}")
    print(f"  moves: mean={np.mean(moves):.2f} max={max(moves)}")
    print(f"  rel loss: median={np.median(losses):.3f} IQR=[{q1:.3f},{q3:.3f}] max={losses.max():.3f}")
    print(f"  zero-cost fraction = {zero_cost}/{len(losses)} = {zero_cost/len(losses):.1%}")
    return dict(total=total, p_pre=pre_ok/total, p_post=post_ok/total,
                wilson=[lo, hi], worst_instance=min(per_inst_post),
                moves_mean=float(np.mean(moves)), moves_max=int(max(moves)),
                loss_median=float(np.median(losses)), loss_q1=float(q1), loss_q3=float(q3),
                loss_max=float(losses.max()), zero_cost_frac=zero_cost / len(losses))


if __name__ == "__main__":
    run()
    campaign()
