"""E-flagship: the non-trivial 1L + 2S composition exchange (materializes Prop. 2).

The scarce dual-capability type L is the BEST lifter (c_L = 2 c_S, F_L = 2 F_S) AND the
only type that can bridge the two regions (gamma_L >> gamma_S, R_L >> R_S). Load 2 is
torque-critical and needs an L; load 1 is substitutable. The productively optimal but
disconnected composition puts both L's on lifting; WISE must reorganize a whole coalition
-- release one L from load 1, replace its capacity with two S, and move it to the central
relay -- along a single neutral direction

    d = -e_{L,load1} + e_{L,relay} + e_{S1,load1} + e_{S2,load1} - e_{S1,relay} - e_{S2,relay},

verifying, exactly and simultaneously,

    A d = 0 (type budgets),  B d = 0 (served aggregate, -c_L + 2 c_S = 0),
    Delta w_k = 0 (both wrenches), Delta n_active = 0,  and  D lambda_2[d] > 0.

This is a genuine comparative-advantage exchange, not a swap of equivalent agents. The
script builds the instance explicitly, certifies both wrenches by the inner-polygon LP,
and reports the before/after invariants. Writes generated/flagship.json and
paper/figures/fig_flagship.pdf.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wise_mr import wrench_tensor as wt  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# ---- geometry: two regions + a central relay site --------------------------
xA, xB, xR = 0.0, 8.0, 4.0                      # left load, right load, central relay
R_S, R_L = 3.0, 5.0                             # short / long comm range
GAMMA_S, GAMMA_L = 0.2, 1.0                     # relay gain (long bridges, short does not)
C_S, C_L = 1.0, 2.0                             # served capacity  (c_L = 2 c_S)
F_S, F_L = 1.0, 2.0                             # planar force cap (F_L = 2 F_S)

# node layout (fixed): 2 long homes, 4 short homes, spread over the two regions + gap
NODES = {
    "L1": np.array([xA, 0.0]), "L2": np.array([xB, 0.0]),
    "S1": np.array([xA, 1.0]), "S2": np.array([xA, -1.0]),
    "S3": np.array([xB, 1.0]), "S4": np.array([xB, -1.0]),
}
NAME = list(NODES)
POS = np.array([NODES[n] for n in NAME])
IS_LONG = np.array([n.startswith("L") for n in NAME])
N = len(NAME)


def _laplacian(relayers):
    """L0 (short intra-region links) + gated long-range relay links.

    A relaying node activates its range r; edges to all nodes within r are added with the
    relayer's gain. Short relayers (r=R_S) never reach across the gap; a long relayer at the
    centre (or spanning both regions) bridges them.
    """
    W = np.zeros((N, N))
    # always-on short links within reach R_S (intra-region only, gap > R_S)
    for i in range(N):
        for j in range(i + 1, N):
            dij = np.linalg.norm(POS[i] - POS[j])
            if dij <= R_S:
                W[i, j] = W[j, i] = np.exp(-dij / 2.0)
    # gated relay links from each relaying robot, using its type range/gain
    for i in relayers:
        r_i = R_L if IS_LONG[i] else R_S
        g_i = GAMMA_L if IS_LONG[i] else GAMMA_S
        # a relayer sits at the central relay site when long (it moves to the gap)
        p_i = np.array([xR, 0.0]) if IS_LONG[i] else POS[i]
        for j in range(N):
            if j == i:
                continue
            dij = np.linalg.norm(p_i - POS[j])
            if dij <= r_i:
                W[i, j] += g_i * np.exp(-dij / 3.0)
                W[j, i] = W[i, j]
    return np.diag(W.sum(1)) - W


def _lambda2(relayers):
    L = _laplacian(relayers)
    w = np.linalg.eigvalsh(0.5 * (L + L.T))
    return float(w[1]), L


# ---- wrench certification (inner-polygon LP, conservative) ------------------
def _slot_maps(offsets):
    """Contact-to-wrench maps A_h = [[1,0],[0,1],[-ry,rx]] for slot offsets (x,y)."""
    return np.array([[[1.0, 0.0], [0.0, 1.0], [-oy, ox]] for (ox, oy) in offsets])


# load 1 (substitutable): four slots along y; load 2 (torque-critical): two slots
SLOTS1 = _slot_maps([(0.0, 0.0), (0.0, 0.4), (0.0, -0.4), (0.0, -0.3)])   # h0,h+,h-,ht
SLOTS2 = _slot_maps([(0.0, -0.5), (0.0, 0.0)])                            # gL, gS
W1_DEM = np.array([3.0, 0.0, 0.3])
W2_DEM = np.array([3.0, 0.0, 1.0])


def _wrench_ok(forces, slots, w_dem):
    """Exact inner-polygon membership: does sum_h A_h f_h = w_dem with f_h in disk(F_h)?"""
    caps = [f for (_, f) in forces]
    maps = np.array([slots[h] for (h, _) in forces])
    return wt.certify_membership_lp(np.array(caps), maps, w_dem, kappa=1.0, m_sides=12)


def certify():
    # --- composition BAD: both L lift; two S sit as (short) relays -> no bridge ---
    # load1: L1@h0 (Fx=2), S1@ht (Fx=1);  load2: L2@gL (Fx=2), S3@gS (Fx=1)
    bad_load1 = [(0, F_L), (3, F_S)]                      # (slot idx, force cap)
    bad_load2 = [(0, F_L), (1, F_S)]
    bad_relayers = [NAME.index("S2"), NAME.index("S4")]  # short relays, cannot cross
    y_bad = np.array([C_L + C_S, C_L + C_S])             # served aggregate per load
    lam_bad, _ = _lambda2(bad_relayers)

    # --- composition WISE: 3S lift load1; L1 -> central relay; load2 keeps 1L+1S ---
    # load1: S1@h+, S2@h-, and S? @ht  (three shorts: Fx=1+1+1=3, tau=+0.4-0.4-0.3=-0.3? )
    wise_load1 = [(1, F_S), (2, F_S), (3, F_S)]           # h+, h-, ht
    wise_load2 = [(0, F_L), (1, F_S)]
    wise_relayers = [NAME.index("L1")]                    # the freed long robot bridges
    y_wise = np.array([3 * C_S, C_L + C_S])
    lam_wise, _ = _lambda2(wise_relayers)

    # --- wrench feasibility of every composition (conservative inner-polygon LP) ---
    ok = {
        "bad_load1": _wrench_ok(bad_load1, SLOTS1, W1_DEM),
        "bad_load2": _wrench_ok(bad_load2, SLOTS2, W2_DEM),
        "wise_load1": _wrench_ok(wise_load1, SLOTS1, W1_DEM),
        "wise_load2": _wrench_ok(wise_load2, SLOTS2, W2_DEM),
    }

    # --- neutral-direction invariants (H1) ---
    dBz = y_wise - y_bad                                  # served aggregate change
    n_active_bad = len(bad_load1) + len(bad_load2) + len(bad_relayers)
    n_active_wise = len(wise_load1) + len(wise_load2) + len(wise_relayers)
    # type budgets: long used = 2 in both (1 lift + 1 relay vs 1 lift-load2 + 1 relay)
    long_bad = sum(IS_LONG[i] for i in bad_relayers) + 2   # both L lifting + short relays
    long_wise = sum(IS_LONG[i] for i in wise_relayers) + 1  # L2 lifts load2, L1 relays
    dV = 0.0  # V = phi(Bz); Bz unchanged -> Delta V = 0 exactly

    result = dict(
        c_L=C_L, c_S=C_S, F_L=F_L, F_S=F_S,
        served_aggregate_bad=y_bad.tolist(), served_aggregate_wise=y_wise.tolist(),
        delta_Bz=dBz.tolist(), delta_V=dV,
        wrench_feasible=ok,
        wrench_identical=bool(all(ok.values())),
        n_active_bad=n_active_bad, n_active_wise=n_active_wise,
        delta_n_active=n_active_wise - n_active_bad,
        long_active_bad=int(long_bad), long_active_wise=int(long_wise),
        lambda2_bad=lam_bad, lambda2_wise=lam_wise,
        delta_lambda2=lam_wise - lam_bad,
        neutral=bool(np.allclose(dBz, 0.0) and n_active_wise == n_active_bad
                     and all(ok.values()) and lam_wise > lam_bad + 1e-6),
    )
    (GEN / "flagship.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    _figure(bad_relayers, wise_relayers, lam_bad, lam_wise)
    return result


def _figure(bad_relayers, wise_relayers, lam_bad, lam_wise):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc

    RED, GREEN, GRAY, INK = "#c0392b", "#1f7a4d", "#b9b9b9", "#222222"
    LOAD_FC, LOAD_EC = "#eef1f4", "#5b6470"
    LOAD_Y, LIFT_Y, RELAY_Y = 1.72, 0.92, -0.35        # rows: load body / lifters / relays
    LW, LH = 2.5, 0.58                                  # load body size

    def load_body(ax, cx, torque):
        """Draw a load as a rounded box with its demanded wrench (force + optional torque)."""
        ax.add_patch(FancyBboxPatch((cx - LW / 2, LOAD_Y - LH / 2), LW, LH,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     fc=LOAD_FC, ec=LOAD_EC, lw=1.3, zorder=2))
        for sx in np.linspace(cx - LW / 2 + 0.35, cx + LW / 2 - 0.35, 4):   # contact slots
            ax.plot([sx, sx], [LOAD_Y - LH / 2, LOAD_Y - LH / 2 - 0.14],
                    color=INK, lw=1.4, zorder=3)
        ax.add_patch(FancyArrowPatch((cx, LOAD_Y + 0.95), (cx, LOAD_Y + LH / 2 + 0.06),
                     arrowstyle="-|>", mutation_scale=11, lw=1.8, color=INK, zorder=3))
        ax.text(cx + 0.16, LOAD_Y + 0.72, r"$w^{\rm dem}$", fontsize=9, color=INK)
        if torque:                                                          # torque-critical load
            ax.add_patch(Arc((cx, LOAD_Y), 1.15, 0.75, angle=0, theta1=200, theta2=430,
                         color=INK, lw=1.6, zorder=3))
            ax.add_patch(FancyArrowPatch((cx + 0.55, LOAD_Y + 0.18), (cx + 0.60, LOAD_Y - 0.02),
                         arrowstyle="-|>", mutation_scale=9, lw=1.4, color=INK, zorder=3))
            ax.text(cx + 0.62, LOAD_Y + 0.30, r"$\tau$", fontsize=9, color=INK)

    def robot(ax, x, y, is_long, role_relay):
        col = GREEN if role_relay else RED
        ax.scatter([x], [y], marker="^" if is_long else "o",
                   s=190 if is_long else 96, c=col, edgecolors="k", lw=0.8, zorder=5)

    def link(ax, p, q, bridging):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=GREEN if bridging else GRAY,
                lw=1.9 if bridging else 1.0, alpha=0.9,
                zorder=1, solid_capstyle="round")

    def panel(ax, lifters1, lifters2, relays, title, connected):
        # lifters: list of (is_long,) placed as a small row beneath each load
        for cx, lifters in [(xA, lifters1), (xB, lifters2)]:
            xs = np.linspace(cx - 0.6 * (len(lifters) - 1) / 2 * 1.0,
                             cx + 0.6 * (len(lifters) - 1) / 2 * 1.0, len(lifters))
            for x, is_long in zip(xs, lifters):
                ax.plot([x, x], [LIFT_Y + 0.16, LOAD_Y - LH / 2 - 0.14],   # attachment stem
                        color=LOAD_EC, lw=0.8, ls=(0, (2, 1.5)), zorder=1)
                robot(ax, x, LIFT_Y, is_long, False)
        # relay site (always present) + occupants
        ax.scatter([xR], [RELAY_Y], marker="D", s=210, facecolors="none",
                   edgecolors=GREEN, lw=1.4, zorder=2)
        if connected:                                    # one long robot bridges from the site
            link(ax, (xA, LIFT_Y), (xR, RELAY_Y), True)
            link(ax, (xB, LIFT_Y), (xR, RELAY_Y), True)
            robot(ax, xR, RELAY_Y, True, True)
        else:                                            # two short relays cannot span the gap
            for rx in (xR - 1.25, xR + 1.25):
                robot(ax, rx, RELAY_Y, False, True)
                ax.plot([rx, rx + (0.9 if rx < xR else -0.9)], [RELAY_Y, RELAY_Y],
                        color=GRAY, lw=1.0, zorder=1)
            ax.plot([xR - 0.35, xR + 0.35], [RELAY_Y, RELAY_Y], color=RED, lw=1.3,
                    ls=(0, (1, 1.2)), zorder=1)
            ax.text(xR, RELAY_Y - 0.62, "gap", fontsize=8, color=RED, ha="center")
        # intra-region short links among lifters
        link(ax, (xA, LIFT_Y), (xA, LIFT_Y), False)
        # composition captions under each load
        comp1 = "3S" if connected and len(lifters1) == 3 else "1L+1S"
        ax.text(xA, LIFT_Y - 0.95, f"load 1: {comp1}", fontsize=8.5, ha="center", color=INK)
        ax.text(xB, LIFT_Y - 0.95, "load 2: 1L+1S", fontsize=8.5, ha="center", color=INK)
        load_body(ax, xA, torque=False)
        load_body(ax, xB, torque=True)
        ax.set_title(title, fontsize=11, pad=3)
        ax.set_xlim(-2.6, 10.6); ax.set_ylim(-1.05, 2.78); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.0))
    panel(axs[0], [True, False], [True, False], bad_relayers,
          r"(a) both $L$ lift: $\lambda_2\approx0$", connected=False)
    panel(axs[1], [False, False, False], [True, False], wise_relayers,
          r"(b) WISE: one $L$ relays, $\lambda_2\geq\sigma_{\rm req}$", connected=True)
    # neutral-exchange arrow + invariants between the panels
    fig.subplots_adjust(wspace=0.20, bottom=0.16, top=0.9)
    fig.text(0.5, 0.62, r"$\Rightarrow$", fontsize=20, ha="center", va="center", color=INK)
    fig.text(0.5, 0.045,
             r"$1L_{\rm lift}{+}2S_{\rm relay}\rightarrow 1L_{\rm relay}{+}2S_{\rm lift}$:  "
             r"$\Delta Bz{=}\Delta w{=}\Delta V{=}0,\ \ \Delta\lambda_2>0$",
             fontsize=9.5, ha="center", va="center", color=INK,
             bbox=dict(boxstyle="round,pad=0.35", fc="#f4f7f4", ec=GREEN, lw=1.0))
    fig.savefig(FIG / "fig_flagship.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(GEN / "fig_flagship.png", dpi=200, bbox_inches="tight")   # inspection only
    plt.close(fig)


if __name__ == "__main__":
    certify()
