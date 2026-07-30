# CREPE: Algorithm, Implementation, and Experiment Hyperparameters

This report documents **CREPE (Controlling diffusion with REPlica Exchange)**
from *CREPE: Controlling Diffusion with Replica Exchange*, arXiv:2509.23265v2,
dated 2 March 2026 and accepted to ICLR 2026. It covers the mathematical
algorithm, its continuous and discrete implementations, all experiment settings
reported in the paper, additional settings recoverable from the official code,
and the generic PyTorch implementation in this workspace. (P-Abs)

## 1. Scope, Sources, and Evidence Labels

The paper and repositories are not treated as interchangeable authorities. The
paper defines the claimed method and reported experiments. The official code is
used to expose implementation details and defaults, including deviations from
the paper. The workspace implementation is a later, generic interface and is
not the code that produced the paper's experiments.

| Label | Source |
|---|---|
| **P** | Paper v2: [arXiv abstract](https://arxiv.org/abs/2509.23265v2), [HTML](https://arxiv.org/html/2509.23265v2), and its TeX source. Section, appendix, equation, algorithm, table, and figure references refer to v2. |
| **O** | Official experiment repository at commit [`af5b08e`](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/tree/af5b08efc98b39ba5f7916c1ae0563ac18d3ca73), dated 15 March 2026. |
| **G** | Generic PyTorch implementation in this workspace, based on commit [`2c873f1`](https://github.com/Utah-Math-Data-Science/RpCN/tree/2c873f1e905781a2c24ddde15ac7f4cb2a689f35) and updated with the explicit direction API documented here. |
| **D** | Direct derivation from cited values, not an independently reported value. |
| **A** | Audit finding: information is absent, inconsistent, or defective in the cited source. |

Short source keys used below:

| Key | Location |
|---|---|
| **P-Abs** | Paper abstract and arXiv metadata. |
| **P-S1**, **P-S4**, **P-S5** | Paper Sections 1, 4, and 5. |
| **P-S2** | Paper Section 2, Background. |
| **P-S3** | Paper Section 3, Methods. |
| **P Algorithm 1** | Paper Algorithm 1, CREPE pseudocode. |
| **P-A1** | Paper Appendix A, Supplementary Methods and Discussion. |
| **P-B1** through **P-B6** | Paper Appendix B.1 through B.6, experiment details. |
| **P-C/D** | Paper Appendices C and D, additional results. |
| **O-CFG** | [`applications/cfg/crepe.py`](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/blob/af5b08efc98b39ba5f7916c1ae0563ac18d3ca73/applications/cfg/crepe.py). |
| **O-Reward** | [`applications/reward-tilting/crepe-guidance-cfg-reward.py`](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/blob/af5b08efc98b39ba5f7916c1ae0563ac18d3ca73/applications/reward-tilting/crepe-guidance-cfg-reward.py). |
| **O-Temper** | [`applications/tempering/main.py`](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/blob/af5b08efc98b39ba5f7916c1ae0563ac18d3ca73/applications/tempering/main.py). |
| **O-PT** | [`applications/ctmc/lib/pt.py`](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/blob/af5b08efc98b39ba5f7916c1ae0563ac18d3ca73/applications/ctmc/lib/pt.py). |
| **O-RNE** | [`applications/ctmc/lib/smc_weights.py`](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/blob/af5b08efc98b39ba5f7916c1ae0563ac18d3ca73/applications/ctmc/lib/smc_weights.py). |
| **O-CTMC** | [`applications/ctmc/lib/sampler.py`](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/blob/af5b08efc98b39ba5f7916c1ae0563ac18d3ca73/applications/ctmc/lib/sampler.py). |
| **O-Maze** | Official and online stitching notebooks under [`applications/stitching`](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/tree/af5b08efc98b39ba5f7916c1ae0563ac18d3ca73/applications/stitching). Notebook cell references are zero-based JSON cell indices. |
| **G-Core** | Current workspace file [`src/spt_guidance/torch/crepe.py`](src/spt_guidance/torch/crepe.py). |

Keys may be refined by subsection, such as **P-S3.1**, **P-A1.4.2**, or
**P-C2**. **P-D** means Appendix D and **P-D1** means its first subsection.
Inclusive ranges are written with the word "through," not by concatenating
source keys.

## 2. Direction Terminology

The paper and code overload the words `forward` and `backward`. This report does
not use either word without naming its axis and endpoints.

### 2.1 Canonical directions used in this report

| Name used here | Exact direction | Meaning | Source |
|---|---|---|---|
| **SDE-forward/noising** | $t\to t^\prime$ | Data/target-side state to noisier/reference-side state for a Gaussian SDE. In theory, $t=0$ is data and $t=1$ is Gaussian noise. | P-S2, Eq. (5) |
| **SDE-backward/denoising** | $t^\prime\to t$ | Gaussian noise/reference-side state to cleaner/target-side state. | P-S2, Eq. (6) |
| **CTMC-forward/noising** | $t\to t^\prime$ | Data token to noisier or masked reference-side token. | P-S2, Eq. (7) |
| **CTMC-backward/denoising** | $t^\prime\to t$ | Masked/noisy token to cleaner target-side token. | P-S2, Eq. (8) |
| **APT target-to-reference proposal** | Lower-noise replica $\to$ higher-noise replica | The paper calls this the APT **forward proposal** because it increases diffusion time. | P-S2.3; P-S3.1, Eq. (10) |
| **APT reference-to-target proposal** | Higher-noise replica $\to$ lower-noise replica | The paper calls this the APT **backward proposal** because it decreases diffusion time. | P-S2.3; P-S3.1, Eq. (11) |
| **PT iteration direction** | Iteration index $n\to n+1$ | MCMC time, not diffusion time. Target samples are produced sequentially along this axis. | P-S3.2 through P-S3.3 |
| **SMC generation direction** | Reference/noise $\to$ target/data | Particles are propagated sequentially along SDE-backward/denoising time. | P-S1, Fig. 2 |
| **Thermal tempering direction** | High physical temperature $\to$ low physical temperature | Cooling a Boltzmann distribution. This is unrelated to diffusion-time direction. | P-S4, Table 1 |

The user-facing phrase **reference-to-target** always means movement toward the
noiseless target. The paper's phrase **forward proposal** does *not* mean this:
it means APT target-to-reference, because the proposal runs in increasing
diffusion time. This distinction is maintained throughout the report. (P-S2.3,
P-S3.1)

### 2.2 Time and storage conventions

The theory uses $0=t_0<\cdots<t_M=1$, so index 0 is target-side and index $M$ is
reference-side. Most official experiment code stores arrays in the opposite
order, from high noise to low noise. Consequently, increasing array index in the
official code usually means reference-to-target even though increasing the
mathematical level index means target-to-reference. (P-S2.3; O-CFG:L323-L355;
O-PT:L501-L519)

The paper uses $M$ inconsistently: the theory has $M+1$ replicas indexed from 0
through $M$, while Appendix B often says "$M$ levels" for the actual number of
stored levels. Experiment tables below report the literal level count and edge
count separately where code makes them unambiguous. (P-S2.3; P Algorithm 1;
P-B1 through P-B5; A)

## 3. What CREPE Does

CREPE adapts accelerated parallel tempering (APT) to inference-time control when
the desired marginal density is not directly evaluable but pretrained diffusion
transition dynamics are available. It keeps one replica at each selected
diffusion time, updates disjoint neighboring pairs in parallel with stochastic
path proposals, and emits states from the target-side replica sequentially over
MCMC iterations. (P-Abs; P-S1; P-S3)

Compared with SMC, CREPE exchanges the roles of diffusion time and particles:
SMC carries many particles in parallel and advances diffusion time sequentially
in the reference-to-target direction; CREPE carries diffusion-time levels in
parallel and advances sample/MCMC time sequentially. The claimed benefits are
lower particle-batch memory, post-burn-in diversity, anytime termination, and
online refinement. The cost is burn-in and possible accumulated bias from model
and discretization error. (P-S1; P-S3.3; P-S5)

### 3.1 Controlled target families

CREPE defines an annealing path $\pi_t$ whose clean endpoint is the desired
distribution. (P-S3)

| Task | Clean target and annealing path | Source |
|---|---|---|
| Tempering | $\pi_t(x)\propto p_t^j(x)^\beta$, with inverse-temperature exponent $\beta>0$. | P-S3, P-A1.5.1 |
| Reward tilting / posterior sampling | $\pi_t(x)\propto p_t^j(x)\exp(r_t(x))$, with $r_0$ the desired reward or log likelihood. | P-S3, P-A1.5.2 |
| Model composition | $\pi_t(x)\propto\prod_j p_t^j(x)$. | P-S3, P-A1.5.3 |
| Debiased CFG | $\pi_t(x)\propto p_t(x)^{1-w}p_t(x\mid c)^w$; this is weighted composition, including a negative exponent when $w>1$. | P-S3 |
| Combined control | The factors can be combined, as in CFG debiasing plus ImageReward and maze model composition plus reward. | P-S3; P-S4 |

At $t=1$, $\pi_1$ is assumed tractable, normally Gaussian for a continuous
diffusion or fully masked for a mask diffusion. Ratios of $\pi_t$ marginals must
be expressible through pretrained diffusion marginal ratios and endpoint reward
terms. (P-S3.1)

## 4. Mathematical Foundation

### 4.1 Path measures and the RNE

For a grid $t=t_0<\cdots<t_K=t^\prime$, let $F$ denote the transition kernel in the
SDE-forward/noising, target-to-reference direction, and let $B$ denote the
transition kernel in the SDE-backward/denoising, reference-to-target direction.
The path laws factor as

```math
\mathbb F_{t,t'}(x_{0:K})
=\mu(x_0)\prod_{k=1}^K F_{t_k|t_{k-1}}(x_k|x_{k-1}),
```

```math
\mathbb B_{t,t'}(x'_{0:K})
=\mu'(x'_K)\prod_{k=1}^K B_{t_{k-1}|t_k}(x'_{k-1}|x'_k).
```

(P-S2.1, Eq. (1))

The paper defines the Radon-Nikodym estimator (RNE)

```math
R_{t,t'}(x)=
\frac{\prod_{k=1}^K B_{t_{k-1}|t_k}(x_{k-1}|x_k)}
     {\prod_{k=1}^K F_{t_k|t_{k-1}}(x_k|x_{k-1})}
```

in the fine-grid limit. It depends on transition dynamics rather than unknown
endpoint marginal densities. (P-S2.1, Eqs. (2)-(3))

When $F$ and $B$ are exact time reversals of the same pretrained diffusion
process $P$,

```math
\frac{p_{t'}(x_{t'})}{p_t(x_t)}
=R^{\mathbb P}_{t,t'}(x)^{-1}.
```

This identity replaces inaccessible diffusion marginal-density ratios by
transition-density ratios. (P-S2.2, Eq. (4))

### 4.2 Gaussian diffusion

The Gaussian SDE-forward/noising process is

```math
\mathrm dX_t=f_t(X_t)\,\mathrm dt+\sigma_t\,\mathrm dW_t,
\qquad X_0\sim p_0,
```

and increases diffusion time from data to Gaussian noise. The exact
SDE-backward/denoising process is

```math
\mathrm dX_t=g_t(X_t)\,\mathrm dt+\sigma_t\,\mathrm d\bar W_t,
\qquad g_t=f_t-\sigma_t^2\nabla\log p_t,
\qquad X_1\sim p_1,
```

integrated with negative time increments from noise toward data. (P-S2.2, Eqs.
(5)-(6))

For generic drifts $\mu_s$ in the SDE-forward/noising direction and $\nu_s$ in the
SDE-backward/denoising direction, Appendix A.2 expresses $\log R$ as a difference
of two Ito integrals plus drift-energy correction. The SDE-forward/noising Ito
integral evaluates each integrand at the lower-time/left endpoint; the
SDE-backward/denoising Ito integral evaluates it at the higher-time/right
endpoint. The summation orientation matters. (P-A1.2, Eqs. (14)-(17))

### 4.3 Discrete diffusion

For finite state space, the CTMC-forward/noising process evolves data toward a
fully masked reference through

```math
\partial_t p_t=\Lambda_t^T p_t.
```

The CTMC-backward/denoising rate is

```math
\Lambda'_t(x,y)=\Lambda_t(y,x)\frac{p_t(y)}{p_t(x)},\quad y\ne x,
```

with diagonal set to the negative row sum. The learned ratio $\frac{p_t(y)}{p_t(x)}$ is
the concrete score. (P-S2.2, Eqs. (7)-(8))

For generic CTMC-forward/noising rate $M_t$ and CTMC-backward/denoising rate
$N_t$, the path RNE combines the integrated difference of diagonal holding rates
and a sum of log jump-rate ratios. (P-A1.2, Eqs. (18)-(19))

## 5. APT and the CREPE Acceptance Rule

### 5.1 Product target and checkerboard communication

The PT Markov chain is invariant, in the ideal exact construction, with respect
to

```math
\Pi(\mathbf X)=\pi_{t_0}(X^0)\times\cdots\times\pi_{t_M}(X^M).
```

At iteration $n$, only edges whose upper index satisfies $m=n\bmod 2$ are
updated. This alternating checkerboard makes all active pairs disjoint and
parallelizable. It is called non-reversible communication in the PT literature;
that word refers to MCMC communication scheduling, not SDE time reversal.
(P-S2.3)

### 5.2 Two stochastic proposal paths

For adjacent times $t<t^\prime$, CREPE independently simulates:

1. An APT target-to-reference path from current lower-noise state $x_t$ to a
   proposed higher-noise endpoint. The paper calls it the **forward proposal**
   because it follows SDE-forward/noising time.
2. An APT reference-to-target path from current higher-noise state $x^\prime_{t^\prime}$ to
   a proposed lower-noise endpoint. The paper calls it the **backward proposal**
   because it follows SDE-backward/denoising time.

The proposal processes can have arbitrary user-selected drifts $a_t,b_t$ or
rates $A_t,B_t$, need not be time reversals, and need not terminate with the
desired marginal. Metropolis correction compensates for their mismatch if all
path ratios are evaluated consistently. (P-S2.3; P-S3.1, Eqs. (10)-(12))

### 5.3 Acceptance probability

Write $x$ for the APT target-to-reference path and $x^\prime$ for the independently
generated APT reference-to-target path, both represented in increasing
diffusion-time orientation when evaluating an RNE. CREPE accepts both proposed
endpoints jointly with

```math
\alpha=\min\left\{1,
\frac{\pi_{t'}(x_{t'})}{\pi_t(x_t)}
\frac{\pi_t(x'_t)}{\pi_{t'}(x'_{t'})}
\frac{R^\mathbb Q(x)}{R^\mathbb Q(x')}
\right\}.
```

(P-S3.1, Eq. (13))

In log form, if

```math
\Delta_\pi(z)=\log\pi_{t'}(z_{t'})-\log\pi_t(z_t),
\qquad
\ell_Q(z)=\log R^\mathbb Q(z),
```

then

```math
\log A=\Delta_\pi(x)-\Delta_\pi(x')+\ell_Q(x)-\ell_Q(x').
```

The generic workspace implementation uses exactly this expression and tests
$\log U<\min(0,\log A)$. (G-Core)

### 5.4 Task-specific target ratios

For every expression below, path $z$ is always oriented from lower diffusion
time to higher diffusion time, even if it was generated in the
reference-to-target direction.

| Target | Increasing-diffusion-time log ratio $\Delta_\pi(z)$ | Source |
|---|---|---|
| Base diffusion | $-\log R^{\mathbb P}(z)$ | P-S2.2 Eq. (4); G-Core |
| Tempered | $-\beta\log R^{\mathbb P}(z)$ | P-A1.5.1; G-Core |
| Reward tilted | $-\log R^{\mathbb P}(z)+r_{t^\prime}(z_{t^\prime})-r_t(z_t)$ | P-A1.5.2, Eqs. (47)-(49); G-Core |
| Composition | $-\sum_j\log R^{\mathbb P_j}(z)$ | P-A1.5.3, Eqs. (50)-(51) |
| Weighted product / CFG | $-\sum_j a_j\log R^{\mathbb P_j}(z)$; CFG uses weights $(1-w,w)$ for unconditional and conditional models. | P-S3; G-Core |

The reward interpolation recommended by the paper is

```math
r_t(x)=\gamma_t r_0\!\left(\mathbb E[X_0\mid X_t=x]\right),
\qquad \gamma_1=0,\quad\gamma_0=1,
```

where the conditional clean estimate is obtained by Tweedie's formula. (P-A1.5.2,
Eq. (47))

## 6. Algorithm 1, Operationally

Algorithm 1 can be written without ambiguous direction words as follows. (P
Algorithm 1)

1. Choose an annealing path $\pi_t$, level schedule, proposal dynamics, and
   optional local invariant kernels.
2. Initialize one state at every level by running a diffusion model. The paper
   does not specify a unique initialization coupling across levels.
3. At each MCMC iteration, select one checkerboard parity of adjacent edges.
4. For every selected edge, in parallel, simulate one APT target-to-reference
   path and one independent APT reference-to-target path.
5. Compute proposal RNE $R^{\mathbb Q}$ and every pretrained-model RNE $R^{\mathbb P_j}$ on both
   paths.
6. Convert the model RNEs and endpoint rewards to the two target marginal
   ratios, then calculate Eq. (13).
7. Jointly accept both proposed endpoints or retain both current endpoints.
8. Optionally apply same-level local exploration to every replica.
9. At a tractable Gaussian reference level, refresh directly from the Gaussian
   instead of applying a Langevin local move.
10. Record the target-side replica after burn-in and any requested thinning.

Local exploration is essential in ordinary PT with deterministic exchanges but
optional in APT because stochastic communication already introduces randomness.
(P-S3.1, footnote 1)

For high-dimensional Gaussian models, the paper recommends stopping PT at a
small positive diffusion time and then completing the reference-to-target
denoising trajectory without PT. This avoids unstable, low-acceptance edges at
very small noise. (P-S3.2)

## 7. Proposal and Discretization Details

### 7.1 Paper's example proposal choices

The following are examples rather than uniquely required choices. The paper
says the same proposals are used in SMC-RNE comparisons. (P-A1.3)

| Task | Paper statement | Important qualification | Source |
|---|---|---|---|
| Tempering | $b_t=f_t$, $a_t=f_t-\beta\sigma_t^2\nabla\log p_t$. | Under the paper's own definitions, $a_t$ is assigned to the APT target-to-reference/SDE-forward-noising path even though it has a denoising score term. Official molecular code instead puts the score on the APT reference-to-target/SDE-backward-denoising path. | P-A1.3; O-Temper:L163-L204; A |
| Reward, differentiable | $b_t=f_t$, $a_t=f_t-\sigma_t^2(\nabla\log p_t+\nabla r_t)$. | Same naming tension as tempering. | P-A1.3 |
| Reward, expensive/non-differentiable | Use pretrained score only and rely on the MH correction for reward. | Used for ImageReward. | P-A1.3; P-B3 |
| Composition | $b_t=f_t$, $a_t=f_t-\sigma_t^2\sum_j\nabla\log p_t^j$. | Example, not mandatory. | P-A1.3 |
| Gaussian CFG | A weighted conditional/unconditional score, with optional proposal strength $w^\prime$. | Appendix A.3 places $w$ on the unconditional score and $1-w$ on the conditional score, contrary to the target definition and official code. The target definition and code use conditional coefficient $w$. | P-S3; P-A1.3; O-CFG:L249-L307; A |
| CTMC CFG | CTMC-forward/noising proposal $A_t=\Lambda_t$; CTMC-backward/denoising proposal uses conditional concrete-score exponent $w^\prime$ and unconditional exponent $1-w^\prime$. | Target strength and proposal strength may differ. | P-A1.3 |

### 7.2 Gaussian Euler-Maruyama paths

For an APT target-to-reference/SDE-forward-noising substep,

```math
X_{s+\delta t}=X_s+a_s(X_s)\,\delta t+\sigma_s\sqrt{\delta t}\,\epsilon.
```

Interpreting the paper's intended time index and arrow notation, an APT
reference-to-target/SDE-backward-denoising substep is

```math
X_{s-\delta t}=X_s-b_s(X_s)\,\delta t+\sigma_s\sqrt{\delta t}\,\epsilon'.
```

The transition densities used in $R^{\mathbb Q}$ and $R^{\mathbb P}$ must use the exact same
discretization as the simulator. Alternatives such as DDPM kernels or exponential
integrators are allowed, but the paper says its implementation uses
Euler-Maruyama. The literal Eq. (21) writes $b_t$ while stepping at $s$ and mixes
arrow notation; the displayed expression above uses the contextually consistent
$b_s$ interpretation. (P-A1.4.1, Eqs. (20)-(25); A)

### 7.3 Gaussian reference stabilization

Directly dividing SDE-backward/denoising and SDE-forward/noising Gaussian
kernels can be unstable because their variances are misaligned. Appendix A.4.1
introduces a tractable Gaussian reference process $\Gamma$, divides $R^{\mathbb P}$ or
$R^{\mathbb Q}$ by $R^\Gamma$, and restores the known Gaussian endpoint ratio. This turns
the calculation into like-direction kernel ratios. The paper uses this only for
molecular tempering, where it empirically helped. (P-A1.4.1, Eqs. (26)-(34))

### 7.4 CTMC Euler paths

For each token, the APT target-to-reference/CTMC-forward-noising categorical
probability is

```math
p(x_{s+\delta t}=y\mid x_s=x)=\mathbf 1[y=x]+A_s(x,y)\,\delta t+o(\delta t),
```

and the APT reference-to-target/CTMC-backward-denoising probability uses $B_s$
analogously. The paper says to omit $o(\delta t)$, clip category weights to
nonnegative values, and renormalize for simulation. During RNE evaluation it
also replaces NaN/Inf and lower-clips values at `1e-8`, accepting a small loss of
normalization. No Gaussian-style reference process is used for CTMCs. (P-A1.4.2)

### 7.5 Local exploration

| Modality | Paper local move | Used where | Source |
|---|---|---|---|
| Gaussian | ULA using the score of $\pi_t$; step size $\frac12\sigma_t^2\,\delta t$, chosen so ULA noise matches denoising-kernel noise. | Molecular tempering and maze; omitted for image CFG and ImageReward. | P-A1.6.1; P-B1 through P-B4 |
| CTMC | An MH-style rate, e.g. $Q(x,y)=r(x,y)\min(1,s_t(x)_y)$ off diagonal, with uniform or masking proposal $r$. For detailed balance in this formula, define $s_t(x)_y=\frac{p_t(y)}{p_t(x)}$. | Omitted in reported MNIST; paper says CTMC works well without it. | P-S2.2; P-A1.6.2; P-B5; A |

The paper says the usual predictor-corrector step-size rule was unstable and the
denoising-aligned Gaussian ULA step size was selected instead. This is one of
the few explicit empirical hyperparameter-selection statements. (P-A1.6.1)

Appendix A.6.2 literally prints $s_t(x)_y\approx\frac{p_t(x)}{p_t(y)}$, reversing the
concrete-score orientation defined in Section 2.2. With that literal orientation,
the displayed $\min(1,s)$ rate is not the stated detailed-balance construction;
the orientation used in the table above is the one consistent with Section 2.2.
(P-S2.2, Eq. (8); P-A1.6.2, Eq. (52); A)

## 8. Cost, Outputs, and Statistical Status

If SMC uses $N$ particles, $M$ resampling stages, and $K$ model steps between
stages, its nominal cost is $MKN$ neural function evaluations. CREPE updates
half the $M$ levels per iteration but evaluates two path directions, also giving
$MK$ evaluations per iteration and $MKN$ over $N$ iterations. The paper says
local moves reuse scores already calculated for RNEs. This is NFE accounting,
not a measured equality of wall time or memory. (P-A1.7)

CREPE samples are correlated MCMC output, not independent particles. Burn-in,
thinning, autocorrelation, effective sample size, and convergence diagnostics
therefore matter. Numerical discard counts are given only for molecules and
text; ImageNet CFG deliberately retains burn-in and prompted reward tilting
discusses it only qualitatively. The paper reports no general diagnostic,
round-trip statistic, or effective sample size. (P-S3.3; P-S4; P-S5; P-B1;
P-B6; A)

The exact stationary claim requires correct transition densities, a perfect
pretrained diffusion model, and consistent discretization. The paper explicitly
warns that model and discretization errors can accumulate over iterations and
move the chain away from the desired target. (P-S5)

## 9. Official Experiment-Code Implementation

### 9.1 Common Gaussian pattern

The image, molecular, and maze implementations specialize to the VE/EDM SDE
$dX_t=\sqrt{2t}\,dW_t$. They store a high-noise-to-low-noise ladder, refresh the
highest-noise state from a Gaussian, alternate checkerboard edges, generate one
Gaussian Euler candidate in each diffusion-time direction per active edge, and
accept both endpoints jointly. (P-B1 through P-B4; O-Temper:L81-L177;
O-CFG:L227-L319; O-Maze official notebook cells 20-24)

Important code-only behavior:

| Behavior | Official code | Methodological effect | Source |
|---|---|---|---|
| Forced clean-end acceptance | Image CFG forces `mask_last=2`; reward and maze force at least the final active pair. | Bypasses Eq. (13), removes the paper's MH invariance guarantee, and can bias the chain unless the forced proposal independently preserves the product target. It is not disclosed in the paper. | O-CFG:L309-L317, L574-L599; O-Reward:L499-L510; O-Maze official cell 24; A |
| Local-move ordering | Molecular and maze code apply local ULA before communication. | Algorithm 1 lists communication then local exploration. | O-Temper:L139-L153; O-Maze official cells 22-24; P Algorithm 1; A |
| Initialization | Application-specific ladders rather than a single prescribed method. | Algorithm 1 only says "by running diffusion model"; CFG shares one clean ancestor, while reward tilting generates independent clean ancestors by level. | P Algorithm 1; O-CFG:L205-L210; O-Reward:L265-L304 |
| Truncated continuation | Image code completes the target-side path with a custom EDM update. | The helper is not literally the standard Euler-Maruyama continuation described in Appendix B. | O-CFG:L183-L197, L358-L368; O-Reward:L247-L262, L330-L335; A |

### 9.2 Official CTMC pattern and defects

The CTMC code stores states from $t=1$ to $t=\varepsilon$, batches active checkerboard
pairs, proposes an APT reference-to-target/CTMC-backward-denoising endpoint with
a CFG concrete score, proposes an APT target-to-reference/CTMC-forward-noising
endpoint with the base graph rate, and combines unconditional, conditional, and
proposal RNEs according to CFG target strength. (O-PT:L13-L98, L501-L640;
O-RNE:L460-L586)

The released MNIST and text CLI default `method="prob"` is not the paper's
categorical path RNE. Lower-level `get_parallel_tempering` and `get_pt_sampler`
APIs instead default to `score`.
It calculates $\log\!\left(\sum_d p_d\right)$ instead of $\sum_d\log p_d$ over independent token
or pixel coordinates. In addition, the Euler transition-probability helpers
construct incorrect identity contributions for changed states and apply
mask-specific factors that suppress valid probabilities. Both MNIST and text
official commands default to this method. Results from those commands therefore
should not be interpreted as exact implementations of Appendix A.4.2. (O-RNE:L388-L417;
O-CTMC:L126-L171; O MNIST `sample_pt.py`:L84-L94; O text
`generate_text_pt.py`:L167-L205; A)

Other CTMC deviations include raw simulation weights without the paper's
explicit clip-and-renormalize step, evaluation of the denoising proposal score
at the lower-time endpoint, no top-reference refresh, and an optional local move
that updates only accepted pairs rather than all replicas. Reported CTMC runs
set local steps to zero, so the local-move defect is dormant. (O-CTMC:L239-L285;
O-PT:L31-L42, L101-L182, L552-L605; P-A1.4.2; A)

The MNIST and text scripts divide edge acceptance counts by all raw checkerboard
half-sweeps even though each edge is attempted on only one parity, and include
an unused terminal slot in summaries. Their displayed acceptance diagnostics
are therefore roughly halved and diluted; a correct diagnostic divides each
edge by its own attempt count. (O MNIST `sample_pt.py`:L281-L309; O text
`generate_text_pt.py`:L395-L425; A)

## 10. Generic Workspace Implementation

The workspace provides a low-level PyTorch `CREPE` class rather than an
experiment reproduction. It delegates path simulation, target ratios, proposal
RNEs, local moves, and exact reference sampling to callbacks. (G-Core)

### 10.1 Callback contracts

Let `R=len(times)`, `B=batch_size`, $P$ be active checkerboard pairs, and $E$
denote event dimensions. (G-Core)

| Callback | Contract | Direction definition |
|---|---|---|
| `init_path_fn` | `times[R] -> ladder[R,*E]`; vectorized result is `[R,B,*E]`. | Times are strictly target-side/lower-noise to reference-side/higher-noise. |
| `target_to_reference_path_fn` | `(t0,t1,lower[B,*E]) -> (upper_candidate[B,*E], target_ratio[B], proposal_rne[B])`. | APT target-to-reference and SDE-forward/noising: `t0<t1`. |
| `reference_to_target_path_fn` | `(t0,t1,upper[B,*E]) -> (lower_candidate[B,*E], target_ratio[B], proposal_rne[B])`. | APT reference-to-target and SDE-backward/denoising. Returned ratios remain oriented from `t0` to `t1`. |
| `local_exploration_fn` | `(time, replica[B,*E]) -> replica'[B,*E]`. | Same-level MCMC move, so neither diffusion-time direction applies. |
| `refresh_reference_fn` | `(highest_time) -> sample[*E]`; vectorized over $B$. | Exact sample at the noisiest/reference level. |
| `should_save_sample_fn` | `(zero_based_iteration,target_sample[B,*E]) -> bool`. | Controls whole-batch retention. |

`torch.vmap(..., randomness='different')` supplies independent randomness across
active replica pairs and outer chains. Path callbacks receive the whole batch
and remain responsible for independent random draws across their own batch
axis. (G-Core)

### 10.2 Execution and results

Zero-based iteration 0 updates $(0,1),(2,3),\ldots$; iteration 1 updates
$(1,2),(3,4),\ldots$. Communication is followed by optional local exploration,
optional exact reference refresh, and sample retention. Returned tensors are
`samples[S,B,*E]`, `replicas[R,B,*E]`, and edge-by-chain acceptance counts,
attempt counts, and rates `[R-1,B]`. (G-Core)

`SaveSampleFn.burn_in_and_thin` first saves at zero-based iteration `burn_in`
and every `thin` iterations thereafter. Initialization is never returned as a
sample. If the lowest ladder time is positive, callers must complete
reference-to-target denoising to data time themselves. (G-Core)

### 10.3 RNE and target helpers

| Helper | Implements | Limit | Source |
|---|---|---|---|
| `from_transition_log_probabilities` | $\sum_k(\log B_k-\log F_k)$ from `target_to_reference_log_probability` and `reference_to_target_log_probability`; $B$ is SDE-backward/denoising and $F$ is SDE-forward/noising. | Caller computes kernels and path. | G-Core |
| `from_euler_maruyama` | Exact log ratio for one discretized Gaussian Euler transition from `target_to_reference_drift` and `reference_to_target_drift`. | One transition, not an exact continuous-path RNE. | G-Core |
| `from_ito_integrals` | Two-endpoint approximation to paper Eqs. (15)-(17), using the less-noisy endpoint for the target-to-reference Ito term and the more-noisy endpoint for the reference-to-target Ito term. | One adjacent interval with trapezoidal drift-energy term. | G-Core |
| `from_reference_stabilized_transition_log_probabilities` | Paper Eq. (34) using primary and stabilizing-process transition logs in both explicitly named directions. | Does not construct or simulate the stabilizing process. | G-Core |
| `TargetLogRatio` | Base, tempered, reward-tilted, and weighted-product target log ratios. | No dedicated CFG or combined weighted-plus-reward wrapper, though callbacks can sum terms. | G-Core |

The generic implementation has no JAX counterpart, no SDE or CTMC simulator,
no schedule builder, and no streaming iterator or continuation-state input.
Focused workspace tests exercise the explicit direction API, RNE signs,
checkerboard communication, retention, and validation. It is faithful to the
core checkerboard acceptance logic but is not a turnkey reproduction. (G-Core;
`tests/test_torch_crepe.py`; G repository tree; A)

The explicit direction API replaces the original paper-style names without
compatibility aliases:

| Original public name | Current public name |
|---|---|
| `forward_path_fn` | `target_to_reference_path_fn` |
| `backward_path_fn` | `reference_to_target_path_fn` |
| `forward_log_probability` | `target_to_reference_log_probability` |
| `backward_log_probability` | `reference_to_target_log_probability` |
| `reference_forward_log_probability` | `stabilizing_target_to_reference_log_probability` |
| `reference_backward_log_probability` | `stabilizing_reference_to_target_log_probability` |
| `reference_log_probability_less_noisy` | `stabilizing_log_probability_less_noisy` |
| `reference_log_probability_more_noisy` | `stabilizing_log_probability_more_noisy` |
| `forward_drift` | `target_to_reference_drift` |
| `backward_drift` | `reference_to_target_drift` |
| `forward_target` | `target_to_reference_target_log_ratio` |
| `forward_rne` | `target_to_reference_proposal_log_rne` |
| `backward_target` | `reference_to_target_target_log_ratio` |
| `backward_rne` | `reference_to_target_proposal_log_rne` |

(G-Core)

## 11. Experiment Overview

| Experiment | Control target | Modality | Paper location | Official code |
|---|---|---|---|---|
| 1D mixture illustration | Reward tilt | Continuous toy | P-A1.1, Fig. 10 | Not released |
| Molecular Boltzmann sampling | Tempering | Gaussian/coordinates | P-S4, P-B1, Table 1, Figs. 3-4 | `applications/tempering` |
| ImageNet CFG debiasing | Weighted composition | Gaussian/images | P-S4, P-B2, Table 2, Figs. 11-14 | `applications/cfg` |
| Prompted image generation | CFG plus reward tilt | Gaussian/images | P-S4, P-B3, Figs. 1, 5, 15 | `applications/reward-tilting` |
| Maze trajectory stitching | Composition plus reward | Gaussian/trajectories | P-S4, P-B4, Table 3, Figs. 6-7, 16 | `applications/stitching` |
| MNIST CFG debiasing | Weighted composition | Mask CTMC/images | P-S4, P-B5, Figs. 8, 18 | `applications/ctmc/mnist` |
| Sentiment text CFG debiasing | Weighted composition | Mask CTMC/text | P-S4, P-B6, Figs. 9, 17 | `applications/ctmc/text` |

The paper does not describe a general hyperparameter search, validation split,
or schedule optimization procedure. Unless a row below explicitly says
"empirically selected" or "swept," a reported value is a fixed experimental
choice, a derived value, or an implementation default rather than a demonstrated
optimum. Advanced PT schedule and path selection are listed as future work.
(P-S5; A)

## 12. One-Dimensional Mixture Illustration

Appendix A.1 illustrates why aggregating many small SMC runs can recover visible
modes while assigning incorrect relative mode mass. SMC uses aggregate sample
count 150,000 with mini-batches of 4, 10, and 100; CREPE uses 150,000 sequential
samples from one run. The claim is qualitative and visual. (P-A1.1, Fig. 10)

The mixture means, variances, weights, reward, diffusion, SMC resampling rule,
CREPE schedule, burn-in, seeds, and numerical errors are not reported, and no
official toy implementation is present. These hyperparameters cannot be
reconstructed from the cited artifacts. (P-A1.1; O repository tree; A)

## 13. Molecular Boltzmann Tempering

### 13.1 Task and derived inverse temperatures

The pretrained model represents a high-physical-temperature Boltzmann
distribution, while the desired target is colder. Raising the learned density
to $\beta=T_{\mathrm{high}}/T_{\mathrm{low}}$ gives the target in the ideal Boltzmann case. The
physical cooling direction is independent of both diffusion time and PT MCMC
time. (P-S4; O-Temper:L27-L50; D)

| Molecule | Atoms / coordinate dimension | Physical temperature | Derived $\beta$ | Network | Source |
|---|---:|---:|---:|---:|---|
| Alanine dipeptide | 22 / 66 | $800\,\mathrm K\to300\,\mathrm K$ | $\frac83=2.6666667$ | 5 layers, width 256 | P Table 1; P-B1; O-Temper:L39-L75; D |
| Alanine tetrapeptide | 43 / 129 | $800\,\mathrm K\to500\,\mathrm K$ | $1.6$ | 5 layers, width 256 | P Table 1; P-B1; O-Temper:L39-L75; D |
| Alanine hexapeptide | 63 / 189 | $800\,\mathrm K\to600\,\mathrm K$ | $\frac43=1.3333333$ | 5 layers, width 512 | P Table 1; P-B1; O-Temper:L39-L75; D |

The official sampler default is only the tetrapeptide at 500 K. Selecting
another molecule without also replacing trajectory paths and checkpoint still
uses tetrapeptide-oriented defaults. (O-Temper:L27-L39; A)

### 13.2 Data and model training

| Setting | Paper-reported value | Code clarification | Tuning status | Source |
|---|---|---|---|---|
| MD duration | 5 microseconds | Dataset files are external. | Fixed data-generation protocol | P-B1 |
| Solvent | Generalized Born implicit solvent | Not generated by released code. | Fixed | P-B1 |
| Force field | AMBER ff96 | Released A4/A6 diagnostic-energy code instead loads Amber99SB-ILDN with OBC1. | Fixed but inconsistent | P-B1; O `energy/a4.py`:L9-L21; O `energy/a6.py`:L9-L21; A |
| Integrator | Langevin-middle | Data-generation code absent. | Fixed | P-B1 |
| Friction | $\mathrm{ps}^{-1}$ | Data-generation code absent. | Fixed | P-B1 |
| MD step | 2 fs | Released A4/A6 diagnostic context uses 1 fs but does not generate the dataset. | Fixed but inconsistent | P-B1; O `energy/a4.py`:L90-L126; A |
| Architecture | EGNN, VE/EDM, EDM `c_in,c_out,c_skip` | Fully connected directed molecular graph, SiLU, attention, recurrent node updates, sum aggregation, Cartesian output. | Fixed by molecule size | P-B1; O `network/egnn.py`:L624-L867 |
| Optimizer | Adam | No training script released. | Fixed | P-B1 |
| Learning rate | $10^{-4}$ | Adam betas, epsilon, weight decay, and batch size absent. | Fixed | P-B1; A |
| Training duration | Until convergence | No criterion, update count, split, or validation rule reported. | Adaptive but unspecified | P-B1; A |
| EMA | 0.999 | Checkpoint code loads a frozen state dict. | Fixed | P-B1; O-Temper:L58-L79 |
| Coordinate preprocessing | Not fully stated in paper | Geometrically center each frame, flatten, multiply coordinates by 5; runtime standard deviation supplies EDM `data_sigma`. | Data-derived | O-Temper:L54-L75 |
| Diagnostic energy regularization | Not reported | Above `energy_cut=1e8`, apply logarithmic regularization and cap at `energy_max=1e20`. | Code-only safeguard | O `energy/a4.py`:L90-L148; O `energy/a6.py`:L90-L148; O `energy/aldp.py`:L14-L50 |

The code's default checkpoint name `net_800k.pt` is not evidence that the paper
trained exactly 800,000 updates; the checkpoint itself and all trajectories are
absent. (O-Temper:L32-L34; A)

### 13.3 CREPE settings

| Hyperparameter | Paper value | Official-code value | Tuning status | Source |
|---|---:|---:|---|---|
| VE SDE | $dX_t=\sqrt{2t}\,dW_t$ | Same | Fixed | P-B1; O-Temper:L81-L91 |
| Diffusion-time range | $[0.001,10]$ | Same | Fixed | P-B1; O-Temper:L81-L91 |
| EDM schedule points | 201 | Same | Fixed | P-B1; O-Temper:L81-L91 |
| EDM schedule exponent | $\rho=7$ | Same | Fixed inherited schedule | P-B1; O-Temper:L81-L91 |
| PT levels | 51, every fourth schedule point | 51 actual stored replicas | Fixed | P-B1; O-Temper:L81-L91 |
| Communication substeps per edge | $K=4$ | Four Euler substeps | Fixed | P-B1; O-Temper:L154-L177, L233-L305 |
| CREPE iterations | 50,000 | 50,000 parity-half-sweep outputs; 25,000 Python outer calls | Fixed budget | P-B1; O-Temper:L319-L338 |
| Burn-in | First 1,000 iterations | Sampling stores all; evaluation selects indices `1000:50000`. | Fixed | P-B1; O `eval.ipynb`, cell 13 |
| Thinning | None reported | None for candidate samples | Fixed | P-B1; O |
| Local exploration | Enabled | One ULA step before each parity communication half-sweep | Empirically selected step-size rule | P-B1; P-A1.6.1; O-Temper:L117-L153 |
| ULA step | $t\,\delta t$ for VE | Same, because $\frac12\sigma_t^2\delta t=t\,\delta t$ | Selected after usual rule was unstable | P-A1.6.1; O-Temper:L117-L121 |
| Gaussian reference stabilization | Enabled | Always enabled; CLI `--ref` is ignored | Empirically selected because it improved tempering | P-B1; P-A1.4.1; O-Temper:L326-L334; A |
| Reference clean marginal | Gaussian, variance not specified | $\gamma_0=\mathcal N(0,I)$, hence $\gamma_t=\mathcal N(0,(1+t^2)I)$ | Code-only fixed value | O-Temper:L207-L227 |
| APT target-to-reference/SDE-forward-noising score multiplier | Not numerically tabulated | CLI `fwd=0.0`, giving driftless Brownian noising | Fixed code default | O-Temper:L22-L25, L163-L204 |
| APT reference-to-target/SDE-backward-denoising score multiplier | Not numerically tabulated | CLI `bwd=1.0`, full tempered-score denoising | Fixed code default | O-Temper:L22-L25, L163-L204 |
| Highest-noise refresh | Gaussian reference | $\mathcal N(0,t_{\max}^2/\beta)$ in mean-free coordinates before every half-sweep | Derived from tempered reference | P-S3.1; O-Temper:L151-L153 |

The `fwd=0,bwd=1` implementation conflicts with the literal labels in Appendix
A.3, which assign the score term to $a_t$ in the paper's APT
target-to-reference/SDE-forward-noising proposal. The code follows the physically
natural convention: Brownian noising target-to-reference and score-based
denoising reference-to-target. This report does not silently choose one as the
authoritative correction. (P-A1.3; O-Temper:L163-L204; A)

### 13.4 Baselines and budget

All SMC methods use the same 201-point schedule and resample every four
SDE-backward/denoising steps. FKC Anneal Score, FKC Anneal Noise, and SMC-RNE
each use 50 batches of 1,000 particles per target, matching 50,000 nominal
particles/CREPE iterations. The molecular official repository does not include
those samplers, and the paper does not give enough detail to distinguish the
two FKC variants algorithmically or specify resampling method/ESS threshold.
(P-B1; O repository tree; A)

### 13.5 Metrics

| Metric | Paper protocol | Code-level detail | Source |
|---|---|---|---|
| Energy TVD | TVD between energy histograms | Fix 100 bins using 101 edges estimated from 25,000 ground-truth configurations, then compare 5,000 generated and 5,000 ground-truth configurations in each repeat. | P-B1; O `eval.ipynb` |
| Distance TVD | TVD between interatomic-distance histograms | Fix 100 bins using 101 edges estimated from 25,000 ground-truth configurations, then pool all unique atom-pair distances from 5,000 configurations per side in each repeat. | P-B1; O `eval.ipynb` |
| Sample $W_2$ | Kabsch-align to a reference, then compute sample Wasserstein distance | Equal-weight exact discrete OT on flattened Cartesian coordinates; first low-temperature frame is alignment reference. | P-B1; O `eval_util.py`:L4-L102; O `optimal_transport.py`:L93-L122 |
| TICA MMD | Backbone torsions $\to$ two-dimensional TICA with $\mathrm{lag}=8$ $\to$ MMD | TICA fitted on ground truth; code uses raw wrapped angles and a sum of 10 Gaussian kernels. | P-B1; O `eval.ipynb`; O `mmd.py`:L5-L51 |
| Evaluation sample count | 5,000 generated and 5,000 ground truth | Drawn without replacement from candidate windows. | P-B1; O `eval.ipynb` |
| Repetitions | 3 | Three random resamplings from one generated run, not three independent sampler runs. Code reports population standard deviation. | P-B1; O `eval.ipynb`; A |
| MMD bandwidth | Fixed "mid-distance" from ground truth | Code concretely uses the median pairwise squared distance, then 10 scales. | P-B1; O `eval.ipynb`; O `mmd.py`:L5-L51 |

### 13.6 Table 1 results

Lower is better in every column. Values are mean plus/minus the paper's error
bar; released evaluation code suggests population standard deviation over three
subsamples. (P Table 1; O evaluation notebook)

| Molecule | Metric | FKC score | FKC noise | SMC-RNE | CREPE |
|---|---|---:|---:|---:|---:|
| Dipeptide | Energy TVD | $0.345\pm0.010$ | $0.894\pm0.002$ | $0.391\pm0.006$ | $\mathbf{0.224\pm0.005}$ |
| Dipeptide | Distance TVD | $0.023\pm0.001$ | $0.036\pm0.001$ | $0.024\pm0.001$ | $\mathbf{0.019\pm0.000}$ |
| Dipeptide | Sample $W_2$ | $0.293\pm0.001$ | $0.282\pm0.001$ | $0.282\pm0.001$ | $\mathbf{0.264\pm0.001}$ |
| Dipeptide | TICA MMD | $0.116\pm0.003$ | $0.108\pm0.004$ | $0.168\pm0.007$ | $\mathbf{0.096\pm0.014}$ |
| Tetrapeptide | Energy TVD | $\mathbf{0.122\pm0.012}$ | $0.436\pm0.007$ | $0.154\pm0.006$ | $\mathbf{0.122\pm0.004}$ |
| Tetrapeptide | Distance TVD | $0.014\pm0.000$ | $0.015\pm0.000$ | $\mathbf{0.013\pm0.001}$ | $\mathbf{0.013\pm0.001}$ |
| Tetrapeptide | Sample $W_2$ | $0.923\pm0.008$ | $0.892\pm0.001$ | $0.893\pm0.005$ | $\mathbf{0.856\pm0.004}$ |
| Tetrapeptide | TICA MMD | $0.183\pm0.020$ | $0.138\pm0.017$ | $0.155\pm0.009$ | $\mathbf{0.035\pm0.002}$ |
| Hexapeptide | Energy TVD | $0.091\pm0.006$ | $0.206\pm0.005$ | $\mathbf{0.087\pm0.003}$ | $0.398\pm0.001$ |
| Hexapeptide | Distance TVD | $0.018\pm0.000$ | $0.020\pm0.001$ | $0.010\pm0.001$ | $\mathbf{0.009\pm0.001}$ |
| Hexapeptide | Sample $W_2$ | $1.585\pm0.001$ | $1.652\pm0.012$ | $1.618\pm0.001$ | $\mathbf{1.299\pm0.004}$ |
| Hexapeptide | TICA MMD | $0.088\pm0.004$ | $0.068\pm0.010$ | $0.042\pm0.004$ | $\mathbf{0.009\pm0.001}$ |

(P Table 1)

The paper attributes the poor hexapeptide CREPE energy TVD to pretrained-model
error amplified over repeated PT iterations and cautions that a favorable energy
histogram can coexist with mode collapse. (P-S4, Figs. 3-4)

### 13.7 Molecular reproducibility gaps

No source reports training batch size, loss, noise-time sampling law, optimizer
betas, exact update count, validation criterion, MD frame stride, equilibration,
seeds, precision, hardware, runtime, memory, or empirical acceptance rates.
The released dipeptide constructor passes `"cuda"` as an environment argument
and fails; the evaluation notebook is hardcoded for 43 atoms and cannot directly
produce the dipeptide or hexapeptide rows. (P-B1; O-Temper:L48-L50; O
`energy/aldp.py`:L14-L27; O `eval.ipynb`; A)

## 14. ImageNet Classifier-Free Guidance Debiasing

### 14.1 Target and models

The target is

```math
\pi_0(x)\propto p_0(x)^{1-w}p_0(x\mid c)^w,
\qquad w=1.7.
```

This debiases the heuristic CFG sampling dynamics to target their implied
weighted product. ImageNet-64 uses pixel-space EDM2-S and ImageNet-512 uses
latent EDM2-XS. The released code downloads EMA checkpoint
`edm2-img64-s-1073741-0.075.pkl` or
`edm2-img512-xs-2147483-0.135.pkl`. (P-S4; P-B2; O-CFG:L17-L29)

### 14.2 Paper hyperparameters and code realization

| Hyperparameter | Paper value | Official-code value | Tuning status | Source |
|---|---:|---:|---|---|
| Target CFG strength | 1.7 | `guidance_target=1.7` | Fixed task value, no search reported | P-B2; O-CFG:L574-L579 |
| Proposal CFG strength | Not separately reported | `guidance=1.7`, equal to target | Fixed code default | O-CFG:L574-L579 |
| VE SDE | $dX_t=\sqrt{2t}\,dW_t$ | Same | Fixed | P-B2; O-CFG:L231-L280 |
| Diffusion-time range | $[0.002,80]$ | Same | Fixed EDM setting | P-B2; O-CFG:L323-L329 |
| Schedule points | 128 | 128 | Fixed | P-B2; O-CFG:L323-L329 |
| EDM exponent | $\rho=7$ | 7 | Fixed | P-B2; O-CFG:L323-L329 |
| PT region | First 100 diffusion steps | CLI `switch_t_idx=30` retains 98 replicas and 97 edges | Fixed but inconsistent | P-B2; O-CFG:L332-L347, L574-L579; A |
| PT spacing / substeps | One level per step, $K=1$ | One Euler transition per edge | Fixed | P-B2; O-CFG:L243-L307 |
| Continuation | Standard CFG Euler-Maruyama after PT | Custom `edm_sampler` split-like update | Fixed but inconsistent integrator | P-B2; O-CFG:L183-L197, L358-L368; A |
| Local exploration | Off | Off | Fixed | P-B2; O-CFG:L574-L579 |
| Burn-in | Deliberately retained for Table 2 | No burn-in parameter; every output retained | Budget-fairness choice | P Table 2; O-CFG:L343-L368 |
| Exact top refresh | Gaussian | Highest state replaced by `80*epsilon` before each half-sweep | Fixed | P-S3.1; O-CFG:L227-L228 |
| Forced acceptance | Not reported | Last two active pairs per parity forced accepted | Undisclosed heuristic, not a valid MH tuning parameter | O-CFG:L309-L317, L574-L579; A |
| Torch seed | Not reported | 1 at import | Code-only fixed default | O-CFG:L12 |
| Device | Not reported | Current CUDA device | Code-only | O-CFG:L12-L13 |

### 14.3 Initialization and communication

For each class, official code first generates one approximate clean image by a
complete CFG reference-to-target denoising run, broadcasts that image over all
128 levels, and independently applies Gaussian SDE-forward/noising at each
level: $x_t=x_{\mathrm{clean}}+t\epsilon_t$. Thus all initial replicas for one class share
one clean ancestor. (O-CFG:L205-L210, L323-L333)

For an edge with high noise $t_1$, low noise $t_0$, and
$\delta t=t_1-t_0$, code uses

```math
x'_0\sim \mathcal N\!\left(x_1+2t_1\delta t\,s_{w'}(x_1),\;2t_1\delta t\,I\right)
```

for the APT reference-to-target/SDE-backward-denoising proposal, and

```math
x'_1\sim \mathcal N\!\left(x_0,\;2t_0\delta t\,I\right)
```

for the APT target-to-reference/SDE-forward-noising proposal. Conditional,
unconditional, and proposal Gaussian log kernels implement the weighted CFG
version of Eq. (13). (O-CFG:L243-L307)

Each Python outer cycle executes both checkerboard parities and records one
completed image after each half-sweep. Consequently, `num_particles=N` is
implemented as $N/2$ outer cycles for even $N$. The low-noise PT endpoint itself
changes on only one parity; the subsequent denoising completion can still add
randomness to both recorded images. (O-CFG:L343-L368, L376-L415)

### 14.4 Evaluation budget and code mismatch

The paper evaluates $N\in\{8,32,128,512\}$. For each $N$, it uses
$B=\lceil5000/N\rceil$ independent class runs with the same selected classes for FKC
and CREPE, yielding 5,000 to 5,120 images. FKC uses the same 128-step schedule,
is active through step 100, uses systematic resampling, and then continues with
ordinary CFG denoising. (P-B2)

| $N$ | Paper $B$ | Paper aggregate $NB$ |
|---:|---:|---:|
| 8 | 625 | 5,000 |
| 32 | 157 | 5,024 |
| 128 | 40 | 5,120 |
| 512 | 10 | 5,120 |

(P-B2; D)

The official CLI instead sweeps `[8,16,32,64,128,256]`, defaults to 100 classes
per batch and a requested 8,000 total images, often overshoots because whole
batches are saved, omits $N=512$, and contains no FKC implementation or shared
class-selection file. These are reproduction defaults, not Table 2 settings.
(O-CFG:L392-L415, L444-L511, L565-L579; A)

### 14.5 Metrics and Table 2

ImageReward and CLIP use the ImageNet class label as conditioning; higher is
better. FID is lower-is-better. The paper does not identify metric checkpoint
versions, FID reference statistics, preprocessing, or seeds. The official CFG
script parses metric-related arguments but calculates none of the three
metrics. (P-S4; P-B2; O-CFG:L581-L600; A)

| Method | Sequential CREPE samples / parallel FKC particles | ImageReward | CLIP | FID |
|---|---:|---:|---:|---:|
| FKC | 8 | -0.29 | 24.17 | 1.85 |
| FKC | 32 | -0.14 | 23.98 | 1.84 |
| FKC | 128 | -0.03 | 24.04 | 1.89 |
| FKC | 512 | -0.08 | 24.31 | 1.96 |
| CREPE | 8 | -0.30 | 24.10 | 1.92 |
| CREPE | 32 | -0.21 | 24.21 | 1.88 |
| CREPE | 128 | -0.09 | **24.37** | 1.86 |
| CREPE | 512 | **0.09** | 24.28 | **1.79** |

(P Table 2)

The paper interprets FKC's advantage at $N=8$ as CREPE burn-in and CREPE's
improvement at larger $N$, especially FID, as its sequential refinement. Figures
11-14 visualize classes slot 800, tick 78, spiny lobster 123, and toilet tissue
999; visualization thins samples by 8 and levels by 2. (P-S4; P-C1)

## 15. Prompted Image Reward Tilting

### 15.1 Target and fixed settings

This experiment combines CFG debiasing with a prompt-dependent ImageReward
factor. It uses the same EDM2-S ImageNet-64 and EDM2-XS ImageNet-512 checkpoints
as the CFG experiment, but sets target CFG strength $w=1.3$. (P-S4; P-B3;
O-Reward:L112-L126)

| Hyperparameter | Paper value | Official-code value | Tuning status | Source |
|---|---:|---:|---|---|
| Target CFG strength | 1.3 | 1.3 | Fixed, no selection method reported | P-B3; O-Reward:L612-L629 |
| Proposal CFG strength | Not separately stated | 1.3 | Fixed equal to target | O-Reward:L612-L629 |
| Initialization CFG strength | Not stated | 1.3 | Code-only fixed default | O-Reward:L612-L629 |
| ImageReward multiplier | 100 | 100 | Fixed, paper says raw magnitude is small | P-B3; O-Reward:L515-L532 |
| Reward interpolation endpoints | $0\to1$ from reference to target | `1e-10 -> 1`, giving effective scaled reference reward `1e-8` | Fixed but numerically altered | P-B3; O-Reward:L361-L364, L627-L629; A |
| Reward interpolation exponent | $\rho_r=5$ | 5 | Fixed | P-B3; O-Reward:L627-L629 |
| VE SDE | $dX_t=\sqrt{2t}\,dW_t$ | Same | Fixed | P-B3; O-Reward:L377-L400 |
| Diffusion-time range | $[0.002,80]$ | Same | Fixed | P-B3; O-Reward:L275-L304 |
| EDM points / exponent | 64 / 7 | 64 / 7 | Fixed | P-B3; O-Reward:L275-L304 |
| PT region | First 32 steps and $M=32\ \text{levels}$ | 33 retained replicas and 32 edges via `switch_t_idx=31` | One-replica mismatch with the literal paper level count; 32 code edges | P-B3; O-Reward:L311-L335, L612-L629; A |
| Communication substeps | $K=1$ | One Euler transition | Fixed | P-B3; O-Reward:L377-L497 |
| Local exploration | Off because reward is expensive and non-differentiable in implementation | Off | Deliberate computational choice | P-B3; P-A1.6.1; O-Reward:L623-L624 |
| PT iterations | Figure 15 shows first 200 | 200 | Fixed visualization budget | P-C2; O-Reward:L612-L629 |
| Burn-in | Mentioned qualitatively, no count | None discarded; all 200 scored/saved | Unspecified | P-S4; P-C2; O reward notebook; A |
| Torch seed | Not reported | 20 | Code-only | O-Reward:L13 |
| Device | Not reported | `cuda:0` | Code-only | O-Reward:L13-L18 |
| Forced acceptance | Not reported | Deepest active pair forced despite `mask_last=0` call | Undisclosed heuristic | O-Reward:L314-L329, L499-L510; A |

Official reward code initializes every schedule level with its own independent
Gaussian state, completely reference-to-target denoises each one with
`init_guidance=1.3`, then independently applies SDE-forward/noising to its
assigned level and retains the first 33 levels. This differs from ImageNet CFG,
where all levels for one class share one clean ancestor. (O-Reward:L265-L304;
O-CFG:L205-L210)

The reward CLI requires exactly one class index in practice even though
`--class_idxs` is declared optional and accepts multiple integers syntactically:
omitting it fails when code calls `len(None)`, and multiple indices fail at
`class_idxs.item()`. The README's slash-separated class syntax is also invalid.
(O-Reward:L516-L533, L547-L560, L631-L662; O reward README:L37-L40; A)

### 15.2 Intermediate reward actually used

For every current or proposed noisy endpoint, official code computes the
proposal-CFG denoiser output $D_{w^\prime}(x_t,t)$, decodes it to pixels, evaluates
ImageReward v1.0 against the prompt, and multiplies by $100\beta_t$. Thus the
implementation realizes

```math
r_t(x_t)=100\,\beta_t\,
\operatorname{ImageReward}(\operatorname{decode}(D_{w'}(x_t,t)),\text{prompt}).
```

No reward gradient enters the APT reference-to-target/SDE-backward-denoising
proposal; reward affects only Eq. (13). This matches Appendix A.3's fallback for
expensive rewards. (P-A1.3; P-A1.5.2; O-Reward:L402-L497)

Appendix B.3 literally prints $r_t(x_t)=\beta_t\mathbb E[x_0\mid x_t]$, omitting the reward
function around the conditional clean estimate. Appendix A.5.2 and code make the
intended operation clear: evaluate the clean reward on the Tweedie/denoiser
estimate. (P-B3; P-A1.5.2 Eq. (47); O-Reward:L402-L497; A)

### 15.3 Prompts

| ImageNet class | Index | Prompt | Source |
|---|---:|---|---|
| Balloon | 417 | `a blue balloon` in paper; `A blue balloon` in code | P-B3; O-Reward:L547-L561; A |
| Pinwheel | 723 | `a colorful pinwheel` | P-B3; O-Reward:L547-L561 |
| Christmas stocking | 496 | `a green Christmas stocking` | P-B3; O-Reward:L547-L561 |
| Cab | 468 | `a yellow cab with dark background` | P-B3; O-Reward:L547-L561 |
| Shopping cart | 791 | `an empty shopping cart` | P-B3; O-Reward:L547-L561 |

Appendix C also shows an umbrella prompt, `a closed umbrella`, as a failure:
the class-conditional model apparently lacks suitable support and CREPE does not
produce a closed umbrella. Its class index is not reported and it is absent from
the script's prompt map. (P-C2 through P-C3, Fig. 15(f); O-Reward:L547-L561; A)

### 15.4 Samples and metrics

Figure 5 reports mean and standard deviation of prompt-conditioned ImageReward
and CLIP over the five listed prompts for both resolutions, but numerical curve
values are not tabulated. Figure 1 thins by 8; Figure 15 displays all first 200
ImageNet-512 iterations. Neighboring samples can be similar because the
target-side edge is active on only alternating checkerboard parities. (P-S4;
P-C2)

The released evaluation notebook uses ImageReward v1.0 and CLIP
`ViT-L/14@336px`, multiplies CLIP cosine similarity by 100, and scores all 200
saved images. The notebook only directly demonstrates the ImageNet-64 balloon
run rather than reproducing the full Figure 5 aggregation. (O reward notebook;
A)

The paper and code provide no tuned burn-in, reward-model checkpoint hash,
acceptance target, random-seed sweep, runtime, memory, or evidence that CFG 1.3,
reward multiplier 100, interpolation exponent 5, or the PT cutoff was selected
by validation. (P-B3; O; A)

## 16. Maze Trajectory Composition and Online Refinement

### 16.1 Task and data

The target composes independently modeled short trajectories and a joint reward:

```math
\pi_0([x^{(1)},\ldots,x^{(J)}])
\propto \exp(r)\prod_j p_0^j(x^{(j)}).
```

Unlike CompDiffuser, the model is unconditional and endpoint/connection
constraints are imposed only through reward. (P-S4)

| Setting | Paper | Official code | Source |
|---|---|---|---|
| Dataset | OGBench `pointmaze-giant-stitch-v0` | External `all_traj.npy`; stored output says 1,005,000 trajectories | P-B4; O-Maze official cells 4, 10 |
| Short trajectory | 64 two-dimensional points | Flattened model dimension 128 | P-B4; O-Maze official cells 4, 13 |
| Composition count | Not reported | `J=20`, giving 1,280 points and 2,560 coordinates | O-Maze official cell 1; A |
| Normalization | Train in $[-1,1]$, undo for evaluation | Axis minima $(-1.26,-1.18)$, maxima $(53.24,37.29)$ | P-B4; O-Maze official cell 10; O-Maze visualization cells 1-2 |
| Dataset preprocessing | Not specified | Notebook assumes `all_traj.npy` is already normalized | O-Maze official cell 4; A |

### 16.2 Model and training

| Hyperparameter | Value | Status | Source |
|---|---:|---|---|
| Model | Unconditional MLP, input/output 128 | Fixed | P-B4; O `applications/stitching/network/mlp.py`:L6-L66 |
| Hidden layers / width | 5 / 512 | Fixed | P-B4; O-Maze official cell 7 |
| Activation | GELU | Code-only | O `network/mlp.py`:L6-L66 |
| Time embedding width | 512, followed by two 512-to-512 linear layers | Code-only | O `network/mlp.py`:L6-L66 |
| Time transform | `log(t+0.1)/4` | Code-only | O `network/mlp.py`:L22-L66 |
| EDM `sigma_data` | 0.6 | Code-only | O-Maze official cell 7 |
| Raw network-output clamp | `[-10000,10000]` | Code default | O `network/mlp.py`:L22-L66 |
| Optimizer / learning rate | Adam / $10^{-4}$ | Fixed paper setting | P-B4 |
| EMA | 0.999 | Fixed paper setting | P-B4 |
| Training duration | Until convergence | Unspecified criterion | P-B4; A |

The checkpoint `traj_net.pt`, training data, and training code are external.
Batch size, loss, noise distribution, optimizer details, update count, split,
validation, seed, and checkpoint criterion are absent. (P-B4; O-Maze; A)

### 16.3 Clean-space reward

The paper combines squared $L_2$ and $L_1$ distances for the start, goal, and all 19
tail-to-head joins. It accidentally states $\lambda_{L_2}=1$ and then
$\lambda_{L_2}=10$; official code resolves the second quantity as $\lambda_{L_1}=10$.
(P-B4, Eqs. (55)-(59); O-Maze official cell 13; A)

For displacement vector $d$, code actually uses a smoothed distance

```math
\phi(d)=\sum_{k=1}^2\left[d_k^2+10^{-5}
+10\sqrt{d_k^2+10^{-5}}\right].
```

(O-Maze official cell 13)

With $J=20$, the implemented clean reward is

```math
r_N=-100\sum_{j=0}^{18}\phi(X^{j+1,0}-X^{j,63}),
```

```math
r_O=-2000\phi(X^{0,0}-O),
\qquad
r_P=-2000\phi(X^{19,63}-P),
\qquad
r=r_O+r_P+r_N.
```

Code additionally applies `-1e10` when any normalized coordinate has absolute
value above 5, a safeguard absent from the paper. (P-B4; O-Maze official cell
13; A)

| Reward hyperparameter | Paper formula | Effective code value | Tuning status | Source |
|---|---:|---:|---|---|
| $\lambda_{L_2}$ | 1 | 1 | Fixed | P-B4; O-Maze official cell 13 |
| $\lambda_{L_1}$ | Typo, not unambiguously named | 10 | Tunable reward strength | P-B4; O-Maze official cell 13; A |
| $\lambda_N$ | 100 | 100 | Tunable reward strength | P-B4; O-Maze official cell 13 |
| $\lambda_O=\lambda_P$ | $100J$ | 2,000 for `J=20` | Tunable reward strength | P-B4; O-Maze official cell 13; D |
| Smoothing | Not reported | `1e-5` | Code-only numerical choice | O-Maze official cell 13 |
| Out-of-range penalty | Not reported | `-1e10` beyond absolute coordinate 5 | Code-only safeguard | O-Maze official cell 13 |

For online refinement through intermediate point $I$, code evaluates all 1,280
points with soft attention

```math
\alpha_{ji}=\operatorname{softmax}_{j,i}(-10\phi(X^{j,i}-I)),
\qquad
r_I=-2000\sum_{j,i}\alpha_{ji}\phi(X^{j,i}-I).
```

Thus $\lambda_I=100J=2000$ and attention temperature $\tau=10$, matching the
paper after resolving $J$. (P-B4, Eqs. (60)-(61); O-Maze online cell 28; D)

At noisy levels, code evaluates the clean reward on the network's denoised
Tweedie estimate, takes its gradient, and scales it with a reward schedule. The
paper's Eq. (62) recursively writes $r_t$ on the right side; code clarifies that
the clean reward $r$ is intended. (P-B4; O-Maze official cells 17, 22, 24; A)

### 16.4 Reward tuning guidance

This is the paper's clearest experiment-specific tuning rule. If tail-to-head
connections form too slowly, increase reward strength. If communication remains
zero at some PT levels, decrease reward strength. Long runs are unnecessary to
diagnose these two failure modes. The paper says final results are not strongly
sensitive but provides no sweep, per-task selected values, or validation
curves. (P-B4)

### 16.5 Diffusion and CREPE settings

| Hyperparameter | Paper value | Official-code value | Tuning status | Source |
|---|---:|---:|---|---|
| VE SDE | $dX_t=\sqrt{2t}\,dW_t$ | Same | Fixed | P-B4; O-Maze official cells 15, 20 |
| Diffusion-time range | $[0.001,20]$ | Same | Fixed | P-B4; O-Maze official cell 15 |
| EDM points / exponent | 601 / 7 | 601 / 7 | Fixed | P-B4; O-Maze official cell 15 |
| PT replicas / edge substeps | 601 / $K=1$ | Exactly 601 replicas and one Euler edge step | Fixed | P-B4; O-Maze official cells 15, 20, 24 |
| Reward schedule | $\beta_{\mathrm{reference}}=0$, $\beta_{\mathrm{target}}=1$, exponent 10 | `1e-10 -> 1`, exponent 10 | Fixed but endpoint altered | P-B4; O-Maze official cell 15; A |
| Initial composed batch | Not reported | One chain with 20 segments | Code-only | O-Maze official cell 20 |
| Initialization | Run diffusion model | 600-step reward-guided reference-to-target denoising trajectory, all levels retained | Fixed implementation | P Algorithm 1; O-Maze official cell 17 |
| Local exploration | On | One ULA step per replica before each parity half-sweep | Fixed; step rule from P-A1.6.1 | P-B4; O-Maze official cells 22, 24 |
| Local ULA step | Denoising-aligned | $h_i=t_i(t_{i+1}-t_i)$ | Derived | P-A1.6.1; O-Maze official cell 22 |
| Highest-noise refresh | Gaussian | Fresh $\mathcal N(0,20^2I)$ each parity half-sweep | Fixed | P-S3.1; O-Maze official cell 24 |
| Communication | On | Reward-guided APT reference-to-target/SDE-backward-denoising candidate plus driftless APT target-to-reference/SDE-forward-noising candidate | Fixed | P-B4; O-Maze official cell 24 |
| Forced clean-edge accept | Not reported | Last pair in each parity is forced | Undisclosed heuristic | O-Maze official cell 24; A |
| Gradient clamp, initial denoising | Not reported | 100 | Code-only | O-Maze official cell 17 |
| Gradient clamp, local and denoising candidate | Not reported | 1,000 | Code-only | O-Maze official cells 22, 24 |
| Gradient clamp, denoising-kernel density evaluation | Not reported | 100 | Code-only | O-Maze official cell 24 |

### 16.6 Five official tasks

The five start-goal pairs are inherited from CompDiffuser. Code maps grid
$(\mathrm{row},\mathrm{column})$ to world $(x,y)=(4\,\mathrm{column}-4,4\,\mathrm{row}-4)$ and then normalizes. (P-S4;
O-Maze official cells 1, 10-11)

| Task | Grid start $\to$ goal | World start $\to$ goal | Source |
|---:|---|---|---|
| 0 | $(1,1)\to(10,14)$ | $(0,0)\to(52,36)$ | O-Maze official cells 1, 10 |
| 1 | $(1,14)\to(10,1)$ | $(52,0)\to(0,36)$ | O-Maze official cells 1, 10 |
| 2 | $(8,14)\to(1,1)$ | $(52,28)\to(0,0)$ | O-Maze official cells 1, 10 |
| 3 | $(8,3)\to(5,12)$ | $(8,28)\to(44,16)$ | O-Maze official cells 1, 10 |
| 4 | $(5,9)\to(3,8)$ | $(32,16)\to(28,8)$ | O-Maze official cells 1, 10 |

The notebook defaults to task 4. The online notebook instead uses normalized
start $(-0.95,0.95)$, goal $(0.95,-0.95)$, and intermediate point
$(0.2,-0.52)$, corresponding approximately to world coordinates
$(0.1025,36.3283)$, $(51.8775,-0.2183)$, and $(31.44,8.0528)$. (O-Maze official
cell 11; O-Maze online cells 10, 27; D)

### 16.7 Iterations, storage, and online update

One Python outer call performs both checkerboard parities, so the paper counts
two PT iterations per outer call. (O-Maze official cell 24)

| Run phase | Outer calls | PT half-sweep iterations | Source |
|---|---:|---:|---|
| Five-task official run | 75,000 | 150,000 | O-Maze official main loop |
| Online before intermediate reward | 50,000 | 100,000 | O-Maze online cell 25 |
| Online after intermediate reward | 50,000 | 100,000 | O-Maze online cells 28-29 |
| Online total | 100,000 | 200,000 | O-Maze online; P Fig. 16 |

At iteration 100,000, the online notebook redefines the reward to include
$r_I$ and continues with the current replicas and RNG stream; it does not
reinitialize or retrain. The paper reports that the new constraint is visibly
satisfied after about 1,000 more iterations. (P-S4, Fig. 6; P-C5, Fig. 16;
O-Maze online cells 25, 27-29)

There is no explicit maze burn-in cutoff. Table 3 reports the poor range from
$0$ to $50{,}000$
rather than discarding it. The notebooks save every 200 outer calls, retaining
both parity states, and thin the ladder every 50 levels to 13 levels; this is
level thinning, not iteration thinning. (P Table 3; O-Maze official/online main
loops; A)

### 16.8 Success metric and results

A code-level success requires all of the following on the unnormalized scale:
start distance $<0.45$, goal distance $<0.45$, every adjacent tail-head distance
$\le0.45$, and no sampled point strictly inside a wall square. The paper describes
start, joins, and walls but does not explicitly restate the goal check in its
metric paragraph. Code does not test continuous line-segment/wall intersections,
so a segment can cross a wall between sampled points. (P-B4; O-Maze visualization
cells 8, 10; A)

| Method / iteration range | Success rate | Sampling basis | Source |
|---|---:|---|---|
| CompDiffuser | 68.0% | Copied from prior work | P Table 3 |
| CREPE, iterations $0$ to $50{,}000$ | 8.5% | 250 saved samples in range, averaged over five tasks | P Table 3 |
| CREPE, iterations $50{,}000$ to $100{,}000$ | 59.7% | Same | P Table 3 |
| CREPE, iterations $100{,}000$ to $150{,}000$ | **84.6%** | Same | P Table 3 |

Figure 16 adds an intermediate pass-rate curve. The intermediate criterion in
code is minimum point distance $<0.45$, but it is commented out of the ordinary
success boolean. The repository lacks the five-task aggregation, Table 3
calculation, and moving-average window used for Figures 7 and 16. (P-C4 through P-C5;
O-Maze visualization cells 8, 10; A)

Stored notebook outputs indicate about 68 minutes for a 150k-iteration official
run and 49 plus 56 minutes for the two online phases, but no GPU model is stored,
so these are environment-specific observations rather than paper runtime claims.
(O-Maze notebook outputs; A)

## 17. Shared CTMC Implementation Details

The official CTMC library supports a uniform graph and an absorbing-mask graph.
For an absorbing graph constructed with clean vocabulary size $V$, code adds a
mask state, so the actual state count is $V+1$; the CTMC-forward/noising process
sends clean states to mask, and the CTMC-backward/denoising process uses learned
concrete scores to unmask. (O `applications/ctmc/lib/graph.py`:L159-L390)

| Component | Implementation | Source |
|---|---|---|
| Time grid | `steps+1` points from 1 to `eps`; `path_length=N` therefore means $N$ edges and $N+1$ replicas. | O-CTMC:L834-L844 |
| Initial ladder | One complete CFG CTMC-backward/denoising trajectory from the graph limit, with final target-side state replaced by an optional Tweedie-style denoise. | O-CTMC:L632-L715 |
| Checkerboard | Even raw PT iteration updates edges $(0,1),(2,3),\ldots$; odd updates $(1,2),(3,4),\ldots$. | O-PT:L13-L42 |
| Proposal CFG score | $s_q=s_{\mathrm{cond}}^{w^\prime}s_{\mathrm{uncond}}^{1-w^\prime}$, formed in log space. | O `lib/models/utils.py`:L212-L339 |
| Target CFG factor | $(1-w)\log R_{\mathrm{uncond}}+w\log R_{\mathrm{cond}}-\log R_{\mathrm{proposal}}$. | O-RNE:L460-L586 |
| Target endpoint retention | Every second raw PT half-sweep, because the target edge is active on one parity. | O-PT:L778-L790 |
| Numerical reference | None, consistent with paper CTMC appendix. | P-A1.4.2; O |

The paper's `512 steps` for MNIST and `150 burn-in + 1000 sampling steps` for
text refer to retained target-side samples. Official code often expresses twice
those counts as raw checkerboard half-sweeps. This distinction explains some,
but not all, paper/code count differences. (P-B5 through P-B6; O-PT:L778-L790)

## 18. MNIST Mask-Diffusion CFG Debiasing

### 18.1 Paper-reported setup

| Hyperparameter | Paper value | Tuning status | Source |
|---|---:|---|---|
| Target CFG strength | 1.2 | Fixed | P-B5 |
| Diffusion discretization | 200 steps | Fixed; paper separately calls $M=200$ the number of levels, so its exact edge count is ambiguous | P-B5; A |
| SMC particles | 128 | Fixed budget | P-B5 |
| SMC resampling | Partial, 80% fraction | Fixed following Lee et al. | P-B5 |
| SMC ESS trigger | $\mathrm{ESS}\le0.2$ | Fixed; paper does not say whether normalized, but magnitude implies normalized | P-B5; A |
| SMC repetitions | Four batches per class for FID | Fixed budget, 512 samples/class | P-B5; D |
| CREPE levels | Literal paper statement: one per diffusion step, $M=200\ \text{levels}$, $K=1$ | Fixed but off-by-one convention is unresolved | P-B5; A |
| CREPE retained steps | 512 per class | Budget match to SMC | P-B5 |
| Local exploration | Off | Deliberate | P-B5 |

The model follows Campbell et al.'s UNet with `num_scales=3`,
`num_res_blocks=3`, `ch_mult=[1,2,4]`, `class_embed_dim=32`, and base channels
`ch=64`. (P-B5)

### 18.2 Released training defaults

The official repository adds a training script, but its default training graph
and sampling graph are incompatible. (O `applications/ctmc/mnist/train.py`;
O CTMC README; A)

| Training setting | Official default | Source |
|---|---:|---|
| Data | MNIST train split; integer pixels 0-255, flattened to 784 | O `lib/data/mnist.py`:L13-L43 |
| Graph / states | Uniform / 256 | O MNIST `train.py`:L154-L188 |
| Noise | Geometric, `sigma_min=1e-3`, `sigma_max=1` | O MNIST `train.py`:L154-L188 |
| Batch | 256 | O MNIST `train.py`:L154-L188 |
| Optimizer | AdamW | O MNIST `train.py`:L154-L188 |
| Learning rate / weight decay | `4e-4` / `1e-3` | O MNIST `train.py`:L154-L188 |
| Updates | 5,000 | O MNIST `train.py`:L154-L188 |
| EMA | 0.99 | O MNIST `train.py`:L154-L188 |
| Conditioning dropout | 0.2 | O MNIST `train.py`:L154-L188 |
| Model dropout | 0.1 | O `lib/models/unet_tauldr.py`:L193-L235 |
| Scaling trick | Enabled by a `store_true` flag whose default is already true, so the CLI cannot disable it | O MNIST `train.py`:L154-L188; O MNIST sampling parsers; A |
| Validation / explicit seed | None / no seed is set or exposed | O MNIST `train.py`; A |

README sampling commands construct an absorbing graph with 256 clean values plus
mask, requiring 257 output classes. A model trained with the default uniform
256-state command has incompatible output dimensions and should fail strict
checkpoint loading. (O CTMC README:L23-L33, L49-L102; O MNIST
`train.py`:L196-L227; A)

### 18.3 Paper versus executable sampling defaults

| Setting | Paper B.5 | Official MNIST commands | Effect / source |
|---|---:|---:|---|
| Graph/noise | Mask diffusion; schedule unspecified | Absorbing graph, cosine noise, `eps=1e-3` | Code-only detail; O MNIST scripts |
| Target CFG, SMC | 1.2 | 3.0 | Material mismatch; O `sample_smc.py`:L34-L80 |
| Proposal CFG, SMC | Not stated | 1.0 | Code-only; O `sample_smc.py`:L34-L80 |
| Target CFG, CREPE | 1.2 | 1.2 | Match; O `sample_pt.py`:L69-L119 |
| Proposal CFG, CREPE | Not stated | 2.0 | Code-only; O `sample_pt.py`:L69-L119 |
| Diffusion discretization / replicas | Paper says 200 steps and $M=200\ \text{levels}$; exact paper edge count is ambiguous | 100 edges, giving 101 replicas | Clear discretization mismatch without over-resolving paper's off-by-one convention; O `sample_pt.py`:L69-L119; A |
| SMC particles | 128 | 64 | Mismatch | O `sample_smc.py`:L34-L80 |
| SMC batches/class | 4 | 8 | Changed to preserve 512 aggregate samples | O `sample_smc.py`:L157-L186 |
| SMC resampling | Partial 0.8 | Multinomial; parsed fraction 0.2 is irrelevant | Mismatch | O `sample_smc.py`:L34-L80 |
| ESS trigger | 0.2 | 0.5 | Mismatch | O `sample_smc.py`:L34-L80 |
| CREPE raw half-sweeps | Not given | 1,324 | 300 raw burn-in plus 1,024 raw sampling sweeps | O `sample_pt.py`:L69-L119 |
| CREPE retained burn-in | Not stated | 150 samples | Every second raw half-sweep | O-PT:L778-L790; D |
| CREPE retained post-burn | 512 | 512 | Budget match despite different path discretization | O `sample_pt.py`:L69-L119; D |
| Local steps | 0 | 0 | Match | P-B5; O `sample_pt.py`:L69-L119 |
| RNE implementation | Product of categorical token/pixel probabilities | Default broken `prob` mode | Does not implement P-A1.4.2; O-RNE:L388-L417; A |

The official `--keep-burn-in` flag is declared `store_true` with default true,
so it cannot be disabled through the CLI. Burn-in samples are also appended to
`all_pt_samples.npy`, making that file contain 6,620 rather than 5,120
post-burn-in images under defaults. (O MNIST `sample_pt.py`:L84-L94,
L311-L342, L471-L492; A)

### 18.4 Outputs and results

Figure 8 visually compares raw CFG, SMC-debiased, and CREPE-debiased digits. The
paper says both corrections improve plausibility and CREPE is slightly more
diverse. Figure 18 visualizes digits 1, 5, and 9, thinning levels by 4 and
recording every eighth PT iteration. Official code stores every tenth raw
iteration instead, and does not reproduce the exact figure selection. (P-S4,
Fig. 8; P-D1, Fig. 18; O MNIST `sample_pt.py`:L194-L205, L375-L443; A)

Appendix B.5 says samples were collected for FID but reports no FID values, FID
implementation, model checkpoint, data split details, seeds, hardware, or
runtime. The official repository has no MNIST FID evaluator. (P-B5; O; A)

## 19. Sentiment-Controlled Text CTMC

### 19.1 Task and paper setup

The condition prompt is `The sentiment of the text is positive` or
`The sentiment of the text is negative`. SEDD-Medium is fine-tuned for five
epochs on the SST-2 sentiment-control dataset and serves as the conditional
model for both SMC and CREPE. (P-B6)

| Hyperparameter | Paper value | Tuning/sweep status | Source |
|---|---:|---|---|
| CTMC discretization | 100 steps | Fixed | P-B6 |
| Sequence length | 1,024 | Fixed | P-B6 |
| Samples per sentiment/configuration | 1,000 | Fixed evaluation budget | P-B6 |
| SMC particles | 50 | Fixed | P-B6 |
| CREPE burn-in | 150 retained target samples | Fixed | P-B6 |
| CREPE sampling | 1,000 retained target samples | Fixed | P-B6 |
| Target CFG strengths | $\{1.0,1.2,1.4,1.8\}$ | Full sweep | P-S4; P-B6 |
| Proposal CFG strengths | $\{1.0,1.2,1.4,1.8\}$ | Full sweep | P-S4; P-B6 |
| Sweep size | 16 target/proposal pairs per method | Derived Cartesian product | P-B6; D |

### 19.2 Model and fine-tuning information

Official code loads base model `louaaron/sedd-medium`, GPT-2 vocabulary 50,257
plus one mask state, and an external fine-tuned checkpoint plus Hydra config.
The local transformer implementation is a bidirectional DDiT/SEDD with rotary
embeddings, AdaLN conditioning, and FlashAttention. (O text `load_model.py`:L40-L70;
O text `model/transformer.py`:L55-L336)

The paper does not report fine-tuning optimizer, learning rate, batch size,
scheduler, seed, prompt formatting, SST-2 preprocessing, or checkpoint rule, and
the repository contains no fine-tuning script. Default checkpoint number 5 is
consistent with but does not prove one checkpoint per epoch. (P-B6; O text;
A)

### 19.3 SMC executable defaults

| Setting | Official default | Relation to paper | Source |
|---|---:|---|---|
| CTMC-backward/denoising steps | 100 | Matches paper | O text `generate_text.py`:L130-L199 |
| Particles | 50 | Matches paper | O text `generate_text.py`:L130-L199 |
| Independent clouds/class | 20 | Gives 1,000 samples | O text `generate_text.py`:L205-L216; D |
| Graph/noise | Absorbing plus log-linear, `eps=1e-3` | Code-only detail | O text `generate_text.py`:L130-L199 |
| Proposal / target strength | 1.2 / 1.2 per default invocation | One point in paper sweep | O text `generate_text.py`:L130-L199 |
| ESS threshold | 0.3 | Absent from paper | O text `generate_text.py`:L130-L199 |
| Partial resampling fraction | 0.8 | Absent from P-B6 | O text `generate_text.py`:L130-L199 |
| Resampling start | Step 0 | Absent from paper | O text `generate_text.py`:L130-L199 |

### 19.4 CREPE executable defaults and mismatch

| Setting | Paper B.6 | Official default | Effect / source |
|---|---:|---:|---|
| Path edges | 100 | 256, giving 257 replicas | Material mismatch; O text `generate_text_pt.py`:L136-L234 |
| Raw burn-in half-sweeps | Equivalent to 300 when retaining every second sweep | 300 | Produces 150 retained burn-in samples; O-PT:L778-L790 |
| Raw post-burn half-sweeps | Equivalent to 2,000 | 2,000 | Produces 1,000 retained samples; O-PT:L778-L790 |
| Total raw half-sweeps | Not stated in this unit | 2,300 | Code count | O text `generate_text_pt.py`:L136-L234 |
| Proposal / target strength | Swept over four values each | 1.2 / 1.2 per invocation | Sweep driver absent | O text `generate_text_pt.py`:L136-L234 |
| Pair batching | Not stated | 16 edges/model call | Code-only | O text `generate_text_pt.py`:L136-L234 |
| Local steps | Not stated; CTMC appendix says unnecessary | 0 | Fixed | O text `generate_text_pt.py`:L136-L234 |
| Path-buffer size / cadence | Not stated | 50 paths, every 20 raw half-sweeps | Diagnostic default | O text `generate_text_pt.py`:L136-L234 |
| RNE | Paper categorical product | Broken default `prob` mode | Method mismatch | O-RNE:L388-L417; A |

The README's PT commands pass nonexistent `--steps 100`, so they fail rather
than override the 256-edge script default. There is no committed sweep
orchestration for all 16 strength pairs. (O CTMC README:L122-L166; O text
`generate_text_pt.py`:L136-L234; A)

### 19.5 Evaluation

| Metric | Model and implementation | Missing information | Source |
|---|---|---|---|
| Perplexity | GPT2-XL with a user-supplied LoRA adapter fine-tuned on SST-2; averages per-sentence perplexities, max length 1,024, evaluation batch 8. | LoRA training code, rank, optimizer, checkpoint, and exact paper command absent. | P-B6; O text `compute_perplexity_pt.py`:L93-L160, L225-L291; A |
| Sentiment accuracy | Hugging Face `Kai1014/distilbert-finetuned`; label order negative/positive; batch 32, classifier max length 512. | Paper reports no classifier preprocessing or uncertainty. | P-B6; O text `compute_sentiment_accuracy.py`:L18-L176 |

The official sentiment evaluator's README command omits required `--method`.
Moreover, its SMC and CFG loaders do not filter each JSON record's method field,
so a combined default file can evaluate both populations together. (O CTMC
README:L191-L196; O text `compute_sentiment_accuracy.py`:L54-L80, L179-L245; A)

### 19.6 Reported behavior

Figure 9 shows CREPE with substantially lower perplexity across target strengths,
up to about a fivefold improvement at target strength 1.8, while sentiment
accuracy is generally similar to SMC. The exception is target 1.4/proposal 1.0,
where CREPE accuracy drops. (P-S4, Fig. 9)

Figure 17 shows running metrics for $(\mathrm{target},\mathrm{proposal})$ pairs $(1.4,1.0)$,
$(1.4,1.4)$, and $(1.8,1.8)$. Perplexity declines and accuracy remains near 1
except for $(1.4,1.0)$, whose accuracy drops around the paper's undefined
"step 1,200." This cannot be a retained-sample index under the released run,
which has 1,150 total retained samples including burn-in; it may denote a raw
checkerboard half-sweep, but the plotting code is absent. The
paper hypothesizes that one poor accepted state can destabilize a continually
refined CREPE chain, whereas independent mini-batch SMC runs isolate occasional
failures. Exact curve values and plotting code are absent. (P-S4; P-D, Fig. 17;
A)

No source reports seeds, uncertainty bars, experiment hardware, precision,
runtime, memory, CTMC acceptance rates, or effective sample sizes for text.
(P-B6; O; A)

## 20. What Was Actually Tuned?

The word "hyperparameter" does not imply that a value was optimized. This table
separates explicit sweeps or qualitative tuning from fixed settings and code
defaults. (P-B1 through P-B6; O)

| Experiment | Explicitly swept or empirically selected | Fixed but not shown as tuned | Code-only or unclear |
|---|---|---|---|
| 1D mixture | Nothing reported | Aggregate samples 150k; SMC batch sizes 4/10/100 | Almost every generative setting is absent. |
| Molecules | Gaussian reference retained because it improved tempering; ULA step rule changed because the usual rule was unstable | Temperatures, model widths, 201-point schedule, 51 levels, $K=4$, 50k iterations, burn-in 1k, SMC $50\times1000$ | Proposal multipliers `fwd=0,bwd=1`, unit-variance Gaussian reference, energy clamps |
| Image CFG | Sample/particle count $N=\{8,32,128,512\}$ is an inference-scaling sweep, not selection of an optimum | CFG 1.7, 128 points, PT cutoff 100, $K=1$, no local move | Proposal scale 1.7, seed 1, forced accept, released cutoff 97 edges |
| Image reward | No sweep reported | CFG 1.3, reward multiplier 100, reward exponent 5, 64 points, 32 PT edges, 200 visualized iterations, no local move | Proposal/init CFG 1.3, seed 20, reward floor `1e-10`, forced accept |
| Maze | Reward strengths tuned qualitatively: increase if joins form slowly, decrease if an edge's communication remains zero | $5\times512$ MLP, $J$ absent from paper, schedule 601, exponent 7, reward exponent 10, $K=1$, local ULA | `J=20`, reward smoothing, hard penalty, gradient clamps, forced accept, task 4 notebook default |
| MNIST | No sweep reported; settings said to follow prior work | Target 1.2, 200 steps, SMC 128, partial 0.8, ESS 0.2, CREPE 512 retained samples | Released commands materially disagree and default to defective `prob` RNE |
| Text | Full $4\times4$ target/proposal CFG-strength sweep | 100 CTMC steps, length 1024, SMC 50, burn-in 150, 1000 retained samples | Released PT path has 256 edges; no sweep driver; defective `prob` RNE |

`fwd` and `bwd` in the molecular code row are source-code parameter names:
`fwd=0` controls APT target-to-reference/SDE-forward-noising score drift and
`bwd=1` controls APT reference-to-target/SDE-backward-denoising score drift.
(O-Temper:L22-L25, L163-L204)

## 21. Paper Ambiguities and Errata

These issues matter for an independent implementation.

| Issue | Literal source | Best-supported interpretation |
|---|---|---|
| RNE subscript | Appendix A.5.1 writes $R_{t,t}$ in the tempering marginal ratio. | The interval and all surrounding formulas require $R_{t,t^\prime}$. (P-A1.5.1; A) |
| Continuous RNE indices | Appendix A.2 writes denoising drift $\nu_t$ while integrating over $s$, and uses $\sigma_t$ in the first two integrals but $\sigma_s$ in the energy term. | Likely index typos; no correction is stated. (P-A1.2; A) |
| Gaussian Euler denoising notation | Appendix A.4.1 Eq. (21) writes $b_t$ while stepping at $s$ and mixes arrow notation. | The contextually consistent interpretation uses denoising drift $b_s$ on the APT reference-to-target/SDE-backward-denoising state. (P-A1.4.1; A) |
| Gaussian proposal labels | Appendix A.3 assigns noising drift $f_t$ to $b_t$ and score-guided drift to $a_t$, despite Eqs. (10)-(11) defining $a_t$ on the APT target-to-reference/SDE-forward-noising path and $b_t$ on the APT reference-to-target/SDE-backward-denoising path. | Official continuous code puts Brownian drift on target-to-reference and score guidance on reference-to-target. Preserve the discrepancy when reproducing the paper. (P-S3.1; P-A1.3; O-Temper; O-CFG; A) |
| Gaussian CFG exponents | Appendix A.3 weights the unconditional score by $w$ and conditional score by $1-w$. | The target definition, CTMC formula, standard CFG, and code weight conditional by $w$. (P-S3; P-A1.3; O-CFG:L249-L307; A) |
| CTMC local concrete score | Appendix A.6.2 Eq. (52) writes $s_t(x)_y\approx\frac{p_t(x)}{p_t(y)}$, opposite Section 2.2 Eq. (8). | The displayed MH-style $\min(1,s)$ rate is consistent with detailed balance only when $s_t(x)_y=\frac{p_t(y)}{p_t(x)}$. (P-S2.2; P-A1.6.2; A) |
| Image reward formula | Appendix B.3 omits ImageReward around $\mathbb E[x_0\mid x_t]$. | Appendix A.5.2 and code evaluate the clean reward on the Tweedie estimate. (P-A1.5.2; P-B3; O-Reward:L402-L497; A) |
| Maze $L_1$ coefficient | Appendix B.4 assigns $\lambda_{L_2}$ both 1 and 10. | Code uses $L_2$ coefficient 1 and $L_1$ coefficient 10. (P-B4; O-Maze official cell 13; A) |
| Maze noisy reward | Appendix B.4 writes $r_t$ on both sides of Eq. (62). | Code applies clean reward $r$ to denoised estimates and multiplies by $\beta_t$. (P-B4; O-Maze; A) |
| Replica count $M$ | Theory has indices $0,\ldots,M$, while appendices call actual counts $M$. | Report actual replicas and edges separately. (P-S2.3; P-B1 through P-B5; A) |
| CFG evaluation cross-reference | Appendix B.2 refers to Figure 5 while describing the CFG metric sweep associated with Table 2; Figure 5 is the prompted reward-tilting plot. | Use Table 2 for ImageNet-64 CFG metrics and Figure 5 for prompted reward tilting. (P-B2; P-S4; A) |

## 22. Official-Code Deviations That Affect Correctness

These are not merely absent documentation; they can change the invariant
distribution or prevent reproduction.

| Severity | Finding | Evidence |
|---|---|---|
| High | Released MNIST/text CLI default CTMC `prob` RNE takes a logarithm after summing coordinate probabilities instead of summing coordinate log probabilities. | O-RNE:L388-L417; P-A1.4.2; A |
| High | CTMC Euler probability helpers give changed states an incorrect identity contribution and apply masks that suppress valid transitions. | O-CTMC:L126-L171; A |
| High | Gaussian image and maze code forces acceptance on clean-end edges, bypassing Eq. (13) and removing the paper's MH invariance guarantee on those moves. | O-CFG:L309-L317; O-Reward:L499-L510; O-Maze official cell 24; A |
| High | MNIST released training defaults produce a 256-state uniform model, while reproduction commands load it into a 257-state absorbing model. | O CTMC README; O MNIST `train.py`; A |
| Medium | Paper and code disagree on CFG PT depth (100 stated levels versus 98 replicas/97 image edges), MNIST discretization (200 versus 100 edges in code), and text PT edges (100 versus 256). | P-B2, P-B5, P-B6; O-CFG; O MNIST/text parsers; A |
| Medium | CTMC simulation does not explicitly perform the paper's clip-and-renormalize operation before categorical sampling. | P-A1.4.2; O `lib/graph.py`:L106-L124; O `lib/catsample.py`:L11-L27; A |
| Medium | Optional CTMC local exploration does not update every level and does not implement Appendix A.6.2's MH-style rate. | P-A1.6.2; O-PT:L101-L182; O-CTMC:L1820-L1841; A |
| Medium | CTMC application acceptance summaries divide by all raw half-sweeps rather than per-edge attempts and include an unused terminal slot. | O MNIST `sample_pt.py`:L281-L309; O text `generate_text_pt.py`:L395-L425; A |
| Medium | Molecular dipeptide constructor and cross-molecule evaluation notebook are broken without edits. | O-Temper:L48-L50; O `energy/aldp.py`:L14-L27; O `eval.ipynb`; A |
| Low | Several README commands contain invalid flags or syntax, and flags declared `store_true` with default true cannot be disabled. | O application READMEs and parsers; A |

The paper's reported numbers remain the record of the experiments. The audit
does not establish which exact uncommitted/private code version generated every
figure or table. The official commit postdates paper v2 by 13 days, and no
release tag binds it to the submitted artifacts. (P-Abs; O commit metadata; A)

## 23. Global Missing Reproducibility Information

Across experiments, the paper does not report accelerator model/count, CPU,
precision, dependency versions, wall-clock runtime, peak memory, energy use, or
random seeds. Official application folders have different implicit environments
and no repository-wide lockfile. (P; O; A)

The following are also absent for most or all experiments: empirical edgewise
swap rates, round-trip times, autocorrelation, MCMC effective sample size,
convergence diagnostics, a general burn-in rule, adaptive ladder tuning,
acceptance targets, and sensitivity studies. (P-S5; P-B1 through P-B6; A)

Training assets are incomplete: molecular and maze training scripts/checkpoints
are absent; text fine-tuning and LoRA evaluation training are absent; MNIST's
released training defaults are incompatible with its reproduction commands.
(O; A)

Baseline assets are incomplete: molecular FKC/RNE and image FKC code are absent;
MNIST FID and the exact text sweep/plotting pipelines are absent. (O; A)

## 24. Interpretation of the Reported Claims

The experiments support the following paper-level observations, subject to the
implementation qualifications above:

- CREPE generally improves molecular diversity and most distributional metrics
  relative to SMC-RNE, but repeated iterations amplify model error in the
  hexapeptide energy metric. (P-S4, Table 1, Figs. 3-4)
- Image CFG shows an expected burn-in disadvantage at 8 samples and better
  large-$N$ behavior, particularly FID, by 512 sequential samples. (P-S4,
  Table 2)
- Prompted reward tilting can refine detailed attributes but cannot create
  support absent from the class model, as shown by the closed-umbrella failure.
  (P-S4; P-C2 through P-C3)
- Maze CREPE supports genuine online constraint updates and improves from 8.5%
  success in iterations $0$ to $50{,}000$ to 84.6% in iterations
  $100{,}000$ to $150{,}000$. (P Table 3; P Figs. 6-7,
  16)
- CTMC experiments visually improve MNIST and reduce text perplexity, but the
  released MNIST/text CLI-default RNE does not implement the paper's stated categorical
  product, so an exact reproduction requires correcting or identifying the code
  version used for the figures. (P-S4; O-RNE; A)

## 25. Reproduction Checklist

An independent implementation should make the following choices explicit rather
than inheriting ambiguous names:

1. Store each path in increasing diffusion-time orientation when evaluating
   $R$, regardless of generation direction.
2. Name the two APT paths `target_to_reference` and `reference_to_target`; if
   preserving paper/code API names, document that paper `forward` means the
   target-to-reference/SDE-forward-noising path and paper `backward` means the
   reference-to-target/SDE-backward-denoising path.
3. Use the same discretized transition kernel for simulation and RNE evaluation.
4. For a factorized CTMC state, sum coordinate/token log probabilities; do not
   take a logarithm after summing coordinate probabilities.
5. Decide whether to reproduce paper settings or released-code defaults when
   they differ, and record both in run metadata.
6. Disable forced accepts if exact Eq. (13) invariance is required; if retained
   as a numerical heuristic, report affected edges, potential bias, and loss of
   the paper's MH invariance guarantee.
7. Report actual replica count, edge count, communication substeps, raw parity
   half-sweeps, retained target samples, burn-in, and thinning separately.
8. Save edgewise attempts/accepts, round trips, target autocorrelation, and MCMC
   effective sample size, none of which are supplied by the paper.
9. Record model/checkpoint hashes, data hashes, seed, dtype, hardware, software
   environment, and continuation integrator.
10. For truncated PT, state the final PT diffusion time and define the subsequent
    reference-to-target denoising integrator exactly.

## 26. Bottom Line

CREPE's central algorithm is a coherent path-space Metropolis construction: run
replicas at diffusion levels in parallel, propose one path in each explicit
diffusion-time direction across alternating adjacent edges, express unknown
target ratios through pretrained-diffusion RNEs and rewards, and emit target-side
states sequentially after burn-in. (P-S3, Algorithm 1)

The paper gives unusually broad experiment coverage and many ladder settings,
but very little formal tuning methodology. The official Gaussian code exposes
important unreported heuristics, while the released MNIST/text CTMC CLI default
contains material RNE defects. The generic workspace class captures Eq. (13) more cleanly
and makes path orientation explicit, but it delegates all model-specific
simulation and does not reproduce the experiments by itself. (P; O; G; A)
