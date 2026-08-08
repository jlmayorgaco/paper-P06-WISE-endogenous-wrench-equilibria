"""Side-by-side animation of PROD | HARD | WISE on the same world.

Illustrative only -- every quantity it draws is also written to
``generated/robot_flagship_timeseries.csv``, and no claim rests on the video.

    python -m experiments.robot_closed_loop.render [--fps 25]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.robot_closed_loop import assignments as A  # noqa: E402
from experiments.robot_closed_loop import config as C  # noqa: E402
from experiments.robot_closed_loop import run_flagship as RF  # noqa: E402
from experiments.robot_closed_loop import scenario as S  # noqa: E402
from experiments.robot_closed_loop import simulator as sim  # noqa: E402

VID = ROOT / "videos"
GREEN, RED, GRAY = "#1e7a46", "#c0392b", "#c8ccd0"


def _load_polygon(k: int, q: np.ndarray) -> np.ndarray:
    """Rectangle spanning the load's contact slots, in world coordinates."""
    off = S.SLOT_OFFSETS[k]
    lo, hi = off.min(0) - 0.18, off.max(0) + 0.18
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    c, s = np.cos(q[2]), np.sin(q[2])
    R = np.array([[c, -s], [s, c]])
    return q[:2] + box @ R.T


def main(fps: int = 25, stride: int = 8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter
    from matplotlib.patches import Polygon

    _, chosen, certs = RF.select()
    methods = [m for m in RF.PRIMARY if m in chosen]
    runs = {m: sim.simulate(m, chosen[m], S.lbar(chosen[m])) for m in methods}
    n = len(next(iter(runs.values())).t)
    frames = range(0, n, stride)

    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.2))
    axes = np.atleast_1d(axes)
    artists = {}
    for ax, m in zip(axes, methods, strict=True):
        ax.set_xlim(-1.4, 9.4)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{m}   $\\lambda_2(\\bar L)={certs[m].lambda2_bar:.3f}$",
                     fontsize=10)
        for site in S.RELAY_SITES.values():
            ax.scatter(*site, marker="D", s=40, facecolors="none",
                       edgecolors="#7f8c8d", lw=0.8, zorder=2)
        for k in range(S.M_LOADS):
            p = np.atleast_2d(S.LOAD_PATHS[k].pose(np.linspace(0, 1, 120)))
            ax.plot(p[:, 0], p[:, 1], color="#aeb4ba", ls=(0, (3, 2)), lw=0.7, zorder=1)
        relay = A.relay_mask(chosen[m])
        artists[m] = {
            "robots": ax.scatter(np.zeros(S.N_ROBOTS), np.zeros(S.N_ROBOTS),
                                 s=np.where(S.IS_LONG, 90, 45),
                                 c=[GREEN if r else RED for r in relay],
                                 edgecolors="k", linewidths=0.5, zorder=5),
            "links": [ax.plot([], [], color=GRAY, lw=1.0, zorder=1)[0]
                      for _ in range(S.N_ROBOTS * (S.N_ROBOTS - 1) // 2)],
            "loads": [ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc="#eef1f4",
                                           ec="#5b6470", lw=1.2, zorder=3))
                      for _ in range(S.M_LOADS)],
            "text": ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=7.5,
                            va="bottom", family="monospace"),
        }
    fig.tight_layout()

    def draw(idx):
        for m in methods:
            res, art = runs[m], artists[m]
            pos = res.robot_traj[idx]
            art["robots"].set_offsets(pos)
            L = S.lgeo(pos, A.relay_mask(chosen[m]))
            e = 0
            for i in range(S.N_ROBOTS):
                for j in range(i + 1, S.N_ROBOTS):
                    ln = art["links"][e]
                    e += 1
                    if -L[i, j] > 1e-9:
                        bridging = A.relay_mask(chosen[m])[i] or A.relay_mask(chosen[m])[j]
                        ln.set_data([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]])
                        ln.set_color(GREEN if bridging else GRAY)
                        ln.set_linewidth(1.8 if bridging else 0.9)
                    else:
                        ln.set_data([], [])
            for k in range(S.M_LOADS):
                art["loads"][k].set_xy(_load_polygon(k, res.load_traj[idx, k]))
            s = res.series
            art["text"].set_text(
                f"t={s['t'][idx]:5.1f}s  V/V*=1.000\n"
                f"lam2_geo={s['lam_geo'][idx]:.3f}  sig_dyn={C.SIGMA_DYN:.2f}\n"
                f"|s1-s2|={s['sync_err'][idx]:.3f}  "
                f"r_w={max(s['wrench_resid1'][idx], s['wrench_resid2'][idx]):.2f}")
        return []

    def _write(writer, path, dpi):
        with writer.saving(fig, str(path), dpi=dpi):
            for idx in frames:
                draw(idx)
                writer.grab_frame()

    VID.mkdir(exist_ok=True)
    out = VID / "robot_flagship_side_by_side.mp4"
    try:
        _write(FFMpegWriter(fps=fps, bitrate=2400), out, 130)
    except Exception as exc:                                    # no ffmpeg available
        out = VID / "robot_flagship_side_by_side.gif"
        print(f"ffmpeg unavailable ({exc}); writing {out} instead")
        _write(PillowWriter(fps=fps), out, 90)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fps = 25
    if "--fps" in sys.argv:
        fps = int(sys.argv[sys.argv.index("--fps") + 1])
    main(fps=fps)
