"""Shared fixtures: one deterministic flagship run, computed once for all tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def flagship():
    from experiments.robot_closed_loop import config as C
    from experiments.robot_closed_loop import run_flagship as RF
    bl, chosen, certs, pert, runs, summaries = RF.run_all(seed=0)
    return {"baselines": bl, "chosen": chosen, "certs": certs, "pert": pert,
            "runs": runs, "summaries": summaries, "sigma_req": C.SIGMA_REQ}
