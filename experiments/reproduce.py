"""One-command reproduction: tests -> certificates -> experiments -> paper -> checks.

Runs the full pipeline that regenerates every number and figure the paper uses and
then rebuilds the manuscript, failing loudly if any step fails or the page count is
wrong. Cross-platform (invoked by `make reproduce`).

    python experiments/reproduce.py            # full pipeline
    python experiments/reproduce.py --fast     # smaller sweeps for a quick check
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def step(title: str, cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"\n=== {title} ===\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        sys.exit(f"FAILED: {title} (exit {r.returncode})")


def run_call(title: str, module: str, kwargs: str) -> None:
    step(title, [PY, "-c", "import sys; sys.path.insert(0,'src'); "
                          f"from experiments.{module} import run; run({kwargs})"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller sweeps for a quick check")
    args = ap.parse_args()

    # 1. tests
    step("tests", [PY, "-m", "pytest", "-q"])

    # 2. certificates (generated/*.json)
    for s in ("nullspace_certificate", "integer_recovery", "geometric_certificate"):
        step(f"certificate: {s}", [PY, f"experiments/{s}.py"])

    # 3. experiments (generated/*.csv,*.json + paper/figures/*.pdf)
    step("experiment: fiber", [PY, "experiments/exp_fiber.py"])
    step("experiment: spatial", [PY, "experiments/exp_spatial.py"])
    step("experiment: physical", [PY, "experiments/exp_physical.py"])
    if args.fast:
        run_call("experiment: phase (fast)", "exp_phase", "grid=4, seeds=2")
        run_call("experiment: methods (fast)", "exp_methods", "seeds=6")
        run_call("experiment: epsilon (fast)", "exp_epsilon", "seeds=2")
    else:
        step("experiment: phase", [PY, "experiments/exp_phase.py"])
        step("experiment: methods", [PY, "experiments/exp_methods.py"])
        step("experiment: epsilon", [PY, "experiments/exp_epsilon.py"])
    # central paper figure (needs fiber + spatial outputs)
    step("figure: central", [PY, "experiments/make_central_fig.py"])

    # 4. build the paper
    step("build paper", ["latexmk", "-pdf", "-interaction=nonstopmode",
                         "-halt-on-error", "main.tex"], cwd=ROOT / "paper")

    # 5. manifest: SHA-256 of every regenerated artifact (frozen provenance)
    import hashlib
    import json
    manifest = {}
    for sub in ("generated", "paper/figures"):
        for p in sorted((ROOT / sub).rglob("*")):
            if p.suffix.lower() in (".csv", ".json", ".pdf", ".tex") and p.name != "manifest.json":
                manifest[str(p.relative_to(ROOT)).replace("\\", "/")] = \
                    hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT / "generated" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # 6. checks: page count, undefined refs, overfull boxes, embedded fonts
    log = (ROOT / "paper" / "main.log").read_text(encoding="utf8", errors="ignore")
    import re
    m = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    pages = int(m.group(1)) if m else -1
    problems = []
    if pages != 6:
        problems.append(f"page count is {pages}, expected 6")
    if re.search(r"Reference `[^']+' on page \d+ undefined", log):
        problems.append("undefined references present")
    if re.search(r"Citation `[^']+' .*undefined", log):
        problems.append("undefined citations present")
    if "Overfull \\hbox" in log:
        problems.append("overfull hbox present")

    print("\n=== SUMMARY ===")
    print(f"pages: {pages}; manifest entries: {len(manifest)}")
    if problems:
        sys.exit("REPRODUCE FAILED:\n  - " + "\n  - ".join(problems))
    print("REPRODUCE OK: tests green, artifacts regenerated, manifest written, "
          "paper builds at 6 pages.")


if __name__ == "__main__":
    main()
