"""Build gates for the manuscript: pages, undefined refs, overfull boxes, stray text.

    python paper/checkbuild.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRAY = [r"\bef\{", r"efsec:", r"\?\?", r"Sections III--III",
         # a backslash eaten by a shell/Python escape leaves the command's tail as prose
         r"\barepsilon", r"(?<![A-Za-z])lpha[^-A-Za-z]", r"\bambda_", r"\bigma_\{",
         r"extbf\{", r"extit\{", r"(?<![A-Za-z])imes(?![A-Za-z])",
         r"(?<![A-Za-z])ho\}", r"(?<![A-Za-z])au_", r"(?<![A-Za-z])ho\)"]
# A backslash eaten by \a \b \f \v \t \r leaves the control byte itself in the source.
# Tabs and CR are included: LaTeX silently accepts both, so only this gate catches them.
CTRL = "\x07\x08\x09\x0b\x0c\r"


def _control_chars() -> list[str]:
    hits = []
    for f in sorted((HERE / "sections").glob("*.tex")) + [HERE / "main.tex"]:
        raw = f.read_bytes().replace(b"\r\n", b"\n")      # tolerate CRLF line endings
        t = raw.decode("utf-8", errors="replace")
        n = sum(t.count(c) for c in CTRL)
        if n:
            hits.append(f"{f.name}:{n}")
    return hits


def main(max_pages: int = 6) -> int:
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"],
                   cwd=HERE, capture_output=True, text=True)
    log = (HERE / "main.log").read_text(encoding="utf-8", errors="ignore")
    pages = re.findall(r"Output written on main\.pdf \((\d+) pages", log)
    n_pages = int(pages[-1]) if pages else -1
    undefined = re.findall(r"Warning: (?:Reference|Citation).*undefined", log)
    over = [float(x) for x in re.findall(r"Overfull \\hbox \((\d+\.\d+)pt", log)]
    bad_over = [x for x in over if x > 5.0]

    txt = ""
    try:
        txt = subprocess.run(["pdftotext", "main.pdf", "-"], cwd=HERE,
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        txt = ""
    stray = [p for p in STRAY if re.search(p, txt)] if txt else ["<pdftotext unavailable>"]

    print(f"pages              : {n_pages}  (limit {max_pages})")
    print(f"undefined refs     : {len(undefined)}")
    print(f"overfull hbox >5pt : {len(bad_over)}  {[f'{x:.1f}' for x in bad_over[:6]]}")
    ctrl = _control_chars()
    print(f"stray markers      : {stray if stray else 'none'}")
    print(f"control chars      : {ctrl if ctrl else 'none'}")
    ok = (n_pages <= max_pages and not undefined and not bad_over and not ctrl
          and (not stray or stray == ["<pdftotext unavailable>"]))
    print("GATES:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
