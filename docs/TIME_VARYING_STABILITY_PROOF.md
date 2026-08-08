# A time-varying corollary to the information-layer stability proposition

**Status: derived and numerically checked here; NOT yet inserted in the manuscript.**
It should be independently checked before it is.

Proposition "stability" in the paper diagonalises a *fixed* Laplacian
$\widetilde L(z)=Q^{\top}\bar L(z)Q$ and reads off a modal decay rate. The robot
experiment drives the same interconnection with $\widetilde L_{\mathrm{geo}}(q(t))=
Q^{\top}L_{\mathrm{geo}}(q(t))Q$, which changes at every step because the robots
move. Modal diagonalisation no longer applies (the eigenbasis rotates), so the
guarantee has to be re-established with a common Lyapunov function. It is, and with
**no loss of rate**.

## Setting

With $a,b\in\mathbb R^{N-1}$, damping $m_y>0$, consensus gain $c>0$, isotropic
mode-aligned couplings $\vartheta_1,\vartheta_2>0$:

$$\dot a=-m_y\,a+\vartheta_1 b,\qquad
  \dot b=\vartheta_2\,a-c\,\widetilde L(t)\,b ,$$

where $\widetilde L(t)=\widetilde L(t)^{\top}\succeq 0$ is measurable and locally
bounded. Assume the **uniform lower-bound certificate**

$$\widetilde L(t)\succeq\sigma_{\mathrm{req}}I\qquad\text{for all }t\ \text{in the
certified operational window.}\tag{A}$$

## Corollary (uniform exponential stability under a moving graph)

Define

$$\mathcal V_c(a,b)=\tfrac12\big(\vartheta_2\|a\|_2^2+\vartheta_1\|b\|_2^2\big)
 =\tfrac12\begin{bmatrix}a\\b\end{bmatrix}^{\!\top}(P\otimes I)
   \begin{bmatrix}a\\b\end{bmatrix},\qquad P=\operatorname{diag}(\vartheta_2,\vartheta_1)\succ0 .$$

Under (A),

$$\dot{\mathcal V}_c\le-2\,\alpha(\sigma_{\mathrm{req}})\,\mathcal V_c,
\qquad
\alpha(\sigma)=\tfrac12\Big[m_y+c\sigma-\sqrt{(m_y-c\sigma)^2+4\vartheta_1\vartheta_2}\Big],$$

hence $\|[a(t),b(t)]\|_P\le e^{-\alpha(\sigma_{\mathrm{req}})(t-t_0)}\,
\|[a(t_0),b(t_0)]\|_P$ with $\|\cdot\|_P=\sqrt{2\mathcal V_c}$, and
$\alpha(\sigma_{\mathrm{req}})>0$ **iff**
$\sigma_{\mathrm{req}}>\vartheta_1\vartheta_2/(c\,m_y)=\sigma_{\mathrm{dyn}}$.

## Proof

Differentiate along trajectories:

$$\dot{\mathcal V}_c
=\vartheta_2 a^{\top}\dot a+\vartheta_1 b^{\top}\dot b
=-\vartheta_2 m_y\|a\|^2+\vartheta_1\vartheta_2 a^{\top}b
 +\vartheta_1\vartheta_2 b^{\top}a-\vartheta_1 c\,b^{\top}\widetilde L(t)b .$$

The two cross terms are equal scalars, so they sum to $2\vartheta_1\vartheta_2a^{\top}b$.
By (A), $b^{\top}\widetilde L(t)b\ge\sigma_{\mathrm{req}}\|b\|^2$, therefore

$$\dot{\mathcal V}_c\le
-\begin{bmatrix}a\\b\end{bmatrix}^{\!\top}
\underbrace{\begin{bmatrix}\vartheta_2 m_y I&-\vartheta_1\vartheta_2 I\\
-\vartheta_1\vartheta_2 I&\vartheta_1 c\,\sigma_{\mathrm{req}}I\end{bmatrix}}_{=\;M\otimes I}
\begin{bmatrix}a\\b\end{bmatrix},
\qquad
M=\begin{bmatrix}\vartheta_2 m_y&-\vartheta_1\vartheta_2\\
-\vartheta_1\vartheta_2&\vartheta_1 c\,\sigma_{\mathrm{req}}\end{bmatrix}.$$

Since $M\otimes I\succeq\lambda_{\min}(M,P)\,(P\otimes I)$ with $\lambda_{\min}(M,P)$ the
smallest generalized eigenvalue of the pencil $(M,P)$, we get
$\dot{\mathcal V}_c\le-2\lambda_{\min}(M,P)\,\mathcal V_c$. Finally

$$P^{-1/2}MP^{-1/2}
=\begin{bmatrix}m_y&-\sqrt{\vartheta_1\vartheta_2}\\
-\sqrt{\vartheta_1\vartheta_2}&c\,\sigma_{\mathrm{req}}\end{bmatrix},$$

whose eigenvalues are
$\tfrac12[(m_y+c\sigma_{\mathrm{req}})\pm\sqrt{(m_y-c\sigma_{\mathrm{req}})^2
+4\vartheta_1\vartheta_2}]$; the smaller one is exactly
$\alpha(\sigma_{\mathrm{req}})$. Positivity of the pencil is
$\det M>0\iff \vartheta_1\vartheta_2 m_y c\,\sigma_{\mathrm{req}}
>\vartheta_1^2\vartheta_2^2\iff\sigma_{\mathrm{req}}>\vartheta_1\vartheta_2/(cm_y)$
(the trace is positive whenever $\sigma_{\mathrm{req}}>0$). $\blacksquare$

## Three remarks that matter for how it may be cited

1. **No conservatism in the rate.** $\alpha$ here is the *same* function as the modal
   rate of Prop. "stability"; the common Lyapunov function
   $\mathcal V_c$ is precisely the one that symmetrises the non-symmetric modal
   block $A_\lambda=\big[\begin{smallmatrix}-m_y&\vartheta_1\\
   \vartheta_2&-c\lambda\end{smallmatrix}\big]$ (the similarity $P^{1/2}$ makes it
   symmetric). So the time-varying statement costs nothing relative to the frozen one.

2. **It needs a uniform bound, not a pointwise one.** (A) must hold at *every* $t$ of
   the window. This is exactly what Lemma "bridge" supplies when the certificate
   $\lambda_2(\bar L(\hat z))\ge\sigma_{\mathrm{req}}$ holds and every robot stays in
   its declared tube -- and only then. During the deployment phase the tubes are not
   yet occupied, so (A) is not claimed and no rate is asserted; the experiment
   separates the two windows and reports them separately.

3. **It says nothing about the loads.** The corollary is about the reduced
   $(a,b)$ interconnection. The downstream map from $a$ to per-robot progress
   commands used in the experiment is an illustrative realization; no nonlinear
   robot--load closed-loop theorem follows from it.

## Numerical verification

`experiments/robot_closed_loop/tests/test_time_varying_stability.py` checks, on the
recorded $Q^{\top}L_{\mathrm{geo}}(q(t))Q$ trajectory of the flagship run:

* the analytic $\alpha$ equals the smallest generalized eigenvalue of $(M,P)$ to
  machine precision, over a grid of $\sigma$;
* the analytic $\alpha$ equals the frozen-graph numerical decay slope
  (matrix-exponential rollout) to machine precision, matching E5;
* the differential inequality $\dot{\mathcal V}_c\le-2\alpha(\sigma_{\mathrm{req}})
  \mathcal V_c$ holds at every recorded step of the certified window;
* $\alpha(\sigma)>0\iff\sigma>\sigma_{\mathrm{dyn}}$ on a grid straddling the threshold.
