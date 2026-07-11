"""Shared plumbing for the experiment drivers (config load, arg parsing, IO)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, type=Path, help="YAML sweep config")
    p.add_argument("--out", type=Path, default=None, help="override output dir")
    p.add_argument("--seed", type=int, default=None, help="override base seed")
    p.add_argument("--dry-run", action="store_true", help="print plan and exit")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_summary(name: str, payload: dict) -> Path:
    out = RESULTS / "summaries"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{name}.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return dest
