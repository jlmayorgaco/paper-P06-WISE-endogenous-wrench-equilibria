import pathlib

SP = ("C:/Users/walla/AppData/Local/Temp/claude/"
      "C--Users-walla-Documents-Github-paper-P07-WISE-endogenous-wrench-equilibria/"
      "35df78bb-bc86-4d70-bd22-fe20ba81144e/scratchpad/03p.bak")
pathlib.Path("sections/03_wise_equilibrium.tex").write_text(open(SP).read())

p = pathlib.Path("sections/05_experiments.tex")
s = p.read_text()
old = r"""Sweeping $\sigma$ over $40$ further
seeds traces $\Price$ as predicted (median costly width $0.64$), the dual matching the finite
difference over $84$ interior points ($r=0.996$). Two controls close the loop: the hard connectivity-constrained
optimum also reaches $V^{\star}$ but returns \emph{any} point of that face (margin $+0.15$
versus WISE's $+2.55$), and a scalarised $V+\varepsilon\lamtwo$ moves the aggregate at
\emph{every} tested $\varepsilon\in[10^{-3},1]$ (drift $2\cdot10^{-4}$ to $0.284$) while its
productive loss stays near zero below $\varepsilon\approx0.1$: at a maximiser $V$ is flat to
second order but the aggregate moves to first. WISE attains zero drift and zero loss with no
weight to calibrate; Stage~2 scales with the $(N{-}1)$-LMI ($29$\,ms${\to}98$\,s)."""
new = r"""Sweeping $\sigma$ over $40$ further seeds traces
$\Price$ as predicted (median costly width $0.64$), the dual matching the finite difference
over $84$ points ($r=0.996$). Two controls close the loop: the hard connectivity-constrained
optimum also reaches $V^{\star}$ but returns \emph{any} point of that face ($+0.15$ versus
WISE's $+2.55$), and a scalarised $V+\varepsilon\lamtwo$ moves the aggregate at \emph{every}
tested $\varepsilon\in[10^{-3},1]$ (drift $2\cdot10^{-4}$ to $0.284$) while its productive
loss stays near zero below $\varepsilon\approx0.1$: at a maximiser $V$ is flat to second order
but the aggregate moves to first. WISE attains zero drift and zero loss with no weight to
calibrate."""
assert old in s
s = s.replace(old, new)
p.write_text(s)
print("done")
