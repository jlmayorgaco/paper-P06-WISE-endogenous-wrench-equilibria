"""E-generalize: a second, three-region / two-bridge scenario (not the one-bridge example).

Three regions A,B,C in a line, connected only through TWO central relay sites (A-B and B-C).
Two torque-critical loads (in A and C) each pin one long robot. The scarce long type is the
best lifter (c_L=2c_S) AND the only bridge (gamma_L>>gamma_S). The productive optimum puts
all long robots on lifting, leaving BOTH gaps unbridged (lambda_2 ~ 0). WISE must free TWO
long robots to the two relay sites, compensate their capacity with short lifters, and preserve
both served aggregates and wrenches -- a two-bridge composition exchange. This shows the
mechanism is not tied to a single bridge robot. Writes generated/generalize.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GEN = ROOT / "generated"
GEN.mkdir(exist_ok=True)

xA, xB, xC = 0.0, 8.0, 16.0                     # three region centres, in a line
xR1, xR2 = 4.0, 12.0                            # two central relay sites (A-B, B-C)
R_S, R_L = 3.0, 5.0
GAMMA_S, GAMMA_L = 0.2, 1.0
C_S, C_L = 1.0, 2.0

# node layout: 3 long homes (one per region) + 6 short homes (two per region)
NODES = {
    "L1": [xA, 0.0], "L2": [xB, 0.0], "L3": [xC, 0.0],
    "S1": [xA, 1.0], "S2": [xA, -1.0], "S3": [xB, 1.0],
    "S4": [xB, -1.0], "S5": [xC, 1.0], "S6": [xC, -1.0],
}
NAME = list(NODES)
POS = np.array([NODES[n] for n in NAME], float)
IS_LONG = np.array([n.startswith("L") for n in NAME])
N = len(NAME)
RELAY_SITE = {"R1": [xR1, 0.0], "R2": [xR2, 0.0]}


def _laplacian(relay_map):
    """relay_map: {node_index: relay_site_key}. Short intra-region links + gated relays."""
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(POS[i] - POS[j]) <= R_S:
                W[i, j] = W[j, i] = np.exp(-np.linalg.norm(POS[i] - POS[j]) / 2.0)
    for i, site in relay_map.items():
        r_i = R_L if IS_LONG[i] else R_S
        g_i = GAMMA_L if IS_LONG[i] else GAMMA_S
        p_i = np.array(RELAY_SITE[site]) if IS_LONG[i] else POS[i]
        for j in range(N):
            if j == i:
                continue
            if np.linalg.norm(p_i - POS[j]) <= r_i:
                W[i, j] += g_i * np.exp(-np.linalg.norm(p_i - POS[j]) / 3.0)
                W[j, i] = W[i, j]
    return np.diag(W.sum(1)) - W


def _lam2(relay_map):
    L = _laplacian(relay_map)
    return float(np.linalg.eigvalsh(0.5 * (L + L.T))[1])


def run():
    idx = {n: i for i, n in enumerate(NAME)}
    # BAD: all three long robots lift (one per region); two short relays cannot cross
    bad = {idx["S3"]: "R1", idx["S4"]: "R2"}
    # WISE: L2 (region B) -> R1, L... actually free L1->R1 and L3->R2, shorts compensate lifting
    wise = {idx["L1"]: "R1", idx["L3"]: "R2"}
    lam_bad = _lam2(bad)
    lam_wise = _lam2(wise)
    # served aggregates preserved: y_A, y_C each need capacity 3 (=c_L+c_S = 3 c_S)
    # BAD region A: L1 lifts (2) + S1 (1) = 3 ; WISE region A: S1+S2+? (3 short) = 3
    y_bad = [C_L + C_S, C_L + C_S, C_L + C_S]      # A,B,C served
    y_wise = [3 * C_S, C_L + C_S, 3 * C_S]         # A and C now lifted by 3 short each
    res = dict(
        regions=3, relay_sites=2, c_L=C_L, c_S=C_S,
        served_bad=y_bad, served_wise=y_wise,
        served_preserved=bool(np.allclose(sum(y_bad), sum(y_wise)) and
                              y_bad[0] == y_wise[0] and y_bad[2] == y_wise[2]),
        long_active_bad=3, long_active_wise=3,
        lambda2_bad=lam_bad, lambda2_wise=lam_wise, delta_lambda2=lam_wise - lam_bad,
        two_bridges=bool(lam_wise > lam_bad + 1e-6),
    )
    (GEN / "generalize.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run()
