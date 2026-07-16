"""E-physical (paper Sec. V): closed-loop rigid-load transport with a WISE team.

A rigid load is carried from the right cluster toward the gap by lifters whose
contact forces live in the heading-aligned elliptical sets, while one long-range
robot relays in the gap. We integrate the load (second-order rigid body) and the
lifters and record, over time: the load pose error, the wrench residual the
nonholonomic team cannot deliver, and the geometric connectivity lambda_2(L_geo(q))
against its affine surrogate lambda_2(Lbar) and the Weyl lower bound
lambda_2(Lbar) - eps_L (Lemma: geometric transfer). The physical connectivity stays
above the dynamic threshold sigma_dyn throughout.

Writes generated/physical_run.csv and paper/figures/fig_physical.pdf.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wise_mr import dynamics as dyn, geometric_bridge as gb  # noqa: E402
from wise_mr.endogenous_graph import fiedler_value  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def _estimate_lambda2(L, v, iters=6, beta=0.15):
    """Distributed Fiedler estimate: a few deflated, shifted power-iteration steps on
    P = I - beta L - 11^T/N (warm-started at v), returning the updated vector and the
    Rayleigh-quotient estimate lambda_hat = v^T L v. Finite iters -> bounded eps_est."""
    N = L.shape[0]
    P = np.eye(N) - beta * L - np.ones((N, N)) / N
    for _ in range(iters):
        v = P @ v
        v = v - v.mean()
        nv = np.linalg.norm(v)
        v = v / nv if nv > 1e-12 else v
    return v, float(v @ L @ v)


def _scene():
    """Fixed 8-robot two-region scene: 3 left, 4 lifters on the load, 1 relay in the gap."""
    left = np.array([[1.6, 5.6], [1.2, 4.4], [2.4, 5.0]])
    load0 = np.array([8.0, 5.0])
    lift_off = np.array([[-0.9, 0.0], [0.9, 0.0], [0.0, 0.9], [0.0, -0.9]])  # rel. to load
    relay = np.array([[5.0, 5.0]])
    pos0 = np.vstack([left, load0 + lift_off, relay])                        # (8,2)
    N = pos0.shape[0]
    ranges = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 6.5])
    relay_mask = np.zeros(N, bool); relay_mask[-1] = True
    lifters = np.array([3, 4, 5, 6])
    F = np.full(N, 2.4)
    return pos0, load0, lift_off, ranges, relay_mask, lifters, F


def run(steps=200, dt=0.03, kappa=0.5):
    pos0, load0, lift_off, ranges, relay_mask, lifters, F = _scene()
    sigma_dyn = 0.20                                          # physical stability threshold
    fixed = pos0.copy()                                       # left cluster + relay stay put

    def geometry(load_pose):
        """Robot positions when the load is at ``load_pose`` (lifters grip rigidly)."""
        p = fixed.copy()
        p[lifters] = load_pose[:2] + (lift_off @ dyn.rot(load_pose[2]).T)
        return p

    load = dyn.Load(pose=np.array([load0[0], load0[1], 0.0]), twist=np.zeros(3))
    ref_goal = np.array([6.9, 5.0, 0.25])                     # gentle transport toward the gap
    N = pos0.shape[0]
    v_est = np.arange(N) - (N - 1) / 2.0; v_est /= np.linalg.norm(v_est)   # estimator warm start
    rows = []
    for t in range(steps):
        s = min(1.0, t / (0.7 * steps))
        ref = np.array([load0[0], load0[1], 0.0]) * (1 - s) + ref_goal * s
        err = ref - load.pose
        w_cmd = np.array([9.0 * err[0], 9.0 * err[1], 6.0 * err[2]])         # P-control wrench
        act = geometry(load.pose)                             # actual geometry
        offs = act[lifters] - load.pose[:2]
        head = np.arctan2(-offs[:, 1], -offs[:, 0])
        f, realized, resid = dyn.realize_wrench(offs, head, F[lifters], kappa, w_cmd)
        load.step(realized, dt)

        act = geometry(load.pose)                             # after the step
        intended = geometry(ref)                              # planned (candidate-site) geometry
        lam_geo = dyn.live_lambda2(act, ranges, relay_mask)
        lam_unsafe = dyn.live_lambda2(act, ranges, np.zeros(pos0.shape[0], bool))  # relay withdrawn
        eps_L, lam_bar, _, weyl_ok = gb.measure_mismatch(intended, act, ranges, relay_mask)
        L_geo = dyn.geometric_laplacian(act, ranges, relay_mask)   # for the online estimator
        v_est, lam_hat = _estimate_lambda2(L_geo, v_est)
        w_norm = float(np.linalg.norm(w_cmd)) + 1e-9
        rows.append(dict(t=t * dt, pose_err=float(np.linalg.norm(err)),
                         wrench_resid=float(resid), wrench_resid_norm=float(resid / w_norm),
                         lam_geo=float(lam_geo), lam_unsafe=float(lam_unsafe),
                         lam_bar=float(lam_bar),
                         lam_hat=float(lam_hat), weyl=float(lam_bar - eps_L),
                         eps_L=float(eps_L)))
    lam_bar = rows[-1]["lam_bar"]
    pos = geometry(load.pose)

    # snapshots for the scene panel
    snaps = {t: None for t in (0, steps // 2, steps - 1)}
    # (recompute positions at snapshot times by re-simulating pose linearly is fine for a sketch;
    #  we instead store the final geometry + the load path)
    path = np.array([[load0[0], load0[1]]] +
                    [(np.array([load0[0], load0[1], 0.0]) * (1 - min(1, tt / (0.7 * steps))) +
                      ref_goal * min(1, tt / (0.7 * steps)))[:2] for tt in range(steps)])

    sigma_req = sigma_dyn + 0.10                              # sigma_req = sigma_dyn + delta (design margin)

    # information-layer self-defeat: drive the 2x2 optimizer-estimator error system by each
    # connectivity trajectory (Prop. small-gain). WISE keeps lambda2 > sigma_dyn (error
    # decays); withdrawing the relay drops lambda2 < sigma_dyn (error grows) -- the decision
    # defeats its own computation.
    m_F, c = 1.0, 1.0
    theta = float(np.sqrt(sigma_dyn * c * m_F))              # so sigma_dyn = theta^2/(c m_F)
    lam_w = np.array([r["lam_geo"] for r in rows])
    lam_u = np.array([r["lam_unsafe"] for r in rows])
    e_w = np.linalg.norm(dyn.rollout_error(lam_w, m_F, c, theta, dt=dt), axis=1)
    e_u = np.linalg.norm(dyn.rollout_error(lam_u, m_F, c, theta, dt=dt), axis=1)

    _write(rows)
    _figure(rows, pos, load, path, ranges, relay_mask, lifters, sigma_dyn, lam_bar)
    _figure_transport(rows, sigma_dyn, sigma_req, e_w, e_u, dt)
    lam_min = min(r["lam_geo"] for r in rows)
    weyl_min = min(r["weyl"] for r in rows)
    est_err = max(abs(r["lam_hat"] - r["lam_geo"]) for r in rows)
    assert lam_min >= sigma_dyn, f"lambda2(L_geo) dipped below sigma_dyn: {lam_min:.3f}"
    print(f"physical: {steps} steps; min lambda2(L_geo)={lam_min:.3f}, "
          f"min Weyl bound={weyl_min:.3f}, sigma_dyn={sigma_dyn:.3f}; "
          f"max estimator err={est_err:.3f}; final pose err={rows[-1]['pose_err']:.3f}")
    return rows


def _figure_transport(rows, sigma_dyn, sigma_req, e_w, e_u, dt):
    """Paper temporal figure -- the information-layer self-defeat dichotomy during transport.
    (a) geometric connectivity for the WISE composition (relay on) vs. the same team with the
    relay withdrawn; the WISE trajectory stays above sigma_req, the other falls below
    sigma_dyn. (b) the coupled optimizer-estimator error e(t) driven by each connectivity:
    above sigma_dyn it decays (self-sustaining), below it grows (self-defeating). (c) the load
    is actually transported (pose error bounded) with the wrench tracked."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    t = np.array([r["t"] for r in rows])
    te = np.arange(len(e_w)) * dt
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 3.5))

    # (a) connectivity: WISE stays above sigma_req; relay-withdrawn falls below sigma_dyn
    a.axhspan(0, sigma_dyn, color="#c0392b", alpha=0.08)
    a.plot(t, [r["lam_geo"] for r in rows], color="#2e8b57", lw=2.1, label="WISE (relay on)")
    a.plot(t, [r["lam_unsafe"] for r in rows], color="#c0392b", lw=1.9, ls="--",
           label="relay withdrawn")
    a.axhline(sigma_req, color="#2e8b57", ls="--", lw=1.0)
    a.axhline(sigma_dyn, color="#c0392b", ls=":", lw=1.0)
    a.text(t[-1], sigma_req, r"$\sigma_{\rm req}$", color="#2e8b57", fontsize=7, va="bottom", ha="right")
    a.text(t[-1], sigma_dyn, r"$\sigma_{\rm dyn}$", color="#c0392b", fontsize=7, va="bottom", ha="right")
    a.set_ylabel(r"$\lambda_2(L_{\rm geo}^{\pi})$", fontsize=8.5)
    a.set_title("(a) connectivity during transport", fontsize=9)
    a.legend(fontsize=7, loc="center right", frameon=False)
    a.set_ylim(-0.02, None)

    # (b) the payoff: optimizer-estimator error decays (WISE) vs grows (withdrawn)
    b.semilogy(te, e_w, color="#2e8b57", lw=2.1, label=r"WISE: $\lambda_2>\sigma_{\rm dyn}$")
    b.semilogy(te, e_u, color="#c0392b", lw=2.1, ls="--",
               label=r"withdrawn: $\lambda_2<\sigma_{\rm dyn}$")
    b.set_ylabel(r"error $\|e(t)\|$", fontsize=8.5)
    b.set_title("(b) optimizer--estimator self-defeat", fontsize=9)
    b.set_xlabel("time [s]", fontsize=8.5)
    b.legend(fontsize=7, loc="lower left", frameon=False)
    b.text(te[-1], e_w[-1], "self-sustaining", color="#2e8b57", fontsize=7, va="center", ha="right")
    b.text(te[-1], e_u[-1], "self-defeating", color="#c0392b", fontsize=7, va="bottom", ha="right")

    for ax in (a, b):
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    fig.tight_layout(h_pad=0.8)
    fig.savefig(FIG / "fig_transport.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def _write(rows):
    with open(GEN / "physical_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def _figure(rows, pos, load, path, ranges, relay_mask, lifters, sigma_dyn, lam_bar):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3 (IEEE)
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    t = np.array([r["t"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.2))

    # (a) scene: final geometry + load transport path + graph
    a = ax[0, 0]
    a.plot(path[:, 0], path[:, 1], color="#888", ls="--", lw=1.0, zorder=1, label="load path")
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            d = np.linalg.norm(pos[i] - pos[j])
            if d <= 2.5:
                a.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color="#c0c0c0", lw=0.6, zorder=1)
    for i in np.where(relay_mask)[0]:
        for j in range(pos.shape[0]):
            if j != i and np.linalg.norm(pos[i] - pos[j]) <= ranges[i]:
                a.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color="#2e8b57", lw=1.3, zorder=1)
    a.add_patch(mp.Rectangle(load.pose[:2] - [0.5, 0.4], 1.0, 0.8, fc="#555", ec="k", lw=0.5, zorder=2))
    short = np.array([i for i in range(pos.shape[0]) if not relay_mask[i]])
    a.scatter(pos[short, 0], pos[short, 1], s=40, c="#3a6ea5", edgecolors="k", linewidths=0.4, zorder=3)
    a.scatter(pos[relay_mask, 0], pos[relay_mask, 1], s=70, marker="D", c="#2e8b57",
              edgecolors="k", linewidths=0.4, zorder=3)
    a.set_title("(a) rigid-load transport", fontsize=9)
    a.set_aspect("equal"); a.set_xlim(0, 10); a.set_ylim(3, 7); a.set_xticks([]); a.set_yticks([])

    # (b) load pose error
    b = ax[0, 1]
    b.plot(t, [r["pose_err"] for r in rows], color="#1f5fbf", lw=1.5)
    b.set_title("(b) load pose error", fontsize=9); b.set_xlabel("time [s]", fontsize=8)
    b.set_ylabel(r"$\|q_{\rm load}-q_{\rm ref}\|$", fontsize=8); b.grid(alpha=0.25)

    # (c) wrench residual
    c = ax[1, 0]
    c.plot(t, [r["wrench_resid"] for r in rows], color="#c0392b", lw=1.5)
    c.set_title("(c) undeliverable wrench", fontsize=9); c.set_xlabel("time [s]", fontsize=8)
    c.set_ylabel("residual", fontsize=8); c.grid(alpha=0.25)

    # (d) connectivity: L_geo vs Lbar vs Weyl bound
    d = ax[1, 1]
    d.plot(t, [r["lam_geo"] for r in rows], color="#2e8b57", lw=1.6, label=r"$\lambda_2(L_{\rm geo})$")
    d.plot(t, [r["lam_bar"] for r in rows], color="#1f5fbf", ls="-", lw=1.1, label=r"$\lambda_2(\bar L)$")
    d.plot(t, [r["weyl"] for r in rows], color="#8e44ad", ls=":", lw=1.4,
           label=r"$\lambda_2(\bar L)-\varepsilon_L$")
    d.axhline(sigma_dyn, color="#c0392b", ls="--", lw=1.0, label=r"$\sigma_{\rm dyn}$")
    d.set_title("(d) connectivity vs. surrogate", fontsize=9); d.set_xlabel("time [s]", fontsize=8)
    d.set_ylabel(r"$\lambda_2$", fontsize=8); d.grid(alpha=0.25)
    d.legend(fontsize=6, loc="center right", frameon=False)

    fig.tight_layout()
    fig.savefig(FIG / "fig_physical.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    run()
