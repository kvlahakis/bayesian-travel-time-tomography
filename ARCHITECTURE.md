# Architecture

## Dependency chain

```
geometry
   |
forward
   |
prior ----+
   |      |
inversion <
   |
diagnostics
   |
experiments
   |
plots
   |
scripts / notebook
```

- `geometry.py` has no dependencies on other project modules. It defines the
  grid, cell-ordering convention, acquisition geometry, and the sensitivity
  matrix `A`. Every other module depends on it, directly or indirectly.
- `forward.py` depends on `geometry.py` (it maps `s` through `A` to `t`, and
  needs cell centers to build synthetic truth fields).
- `prior.py` depends on `geometry.py` (needs cell centers to build the
  squared-exponential covariance) but not on `forward.py`.
- `inversion.py` depends on `forward.py` (via `A`, `d`) and `prior.py` (via
  `s0`, `Cs`).
- `diagnostics.py` depends on `inversion.py`'s outputs (`s_post`, `C_post`) and
  a ground truth from `forward.py`.
- `experiments.py` wires `geometry`, `forward`, `prior`, `inversion`, and
  `diagnostics` together into the five experiment families, using
  `config.py` for reproducible parameterization.
- `plots.py` consumes `experiments.py` outputs (and raw `geometry`/`inversion`
  quantities for the singular-value / eigenvalue spectra) and produces figures.
- `scripts/run_experiment.py` and `notebooks/report.ipynb` are thin consumers
  at the top of the chain: YAML config in, `experiments.py` + `plots.py` out.
  No mathematical logic lives above `experiments.py`.

## Cell ordering convention

Cell index `k` corresponds to grid position `(i, j)` via `k = i * n + j`
(row-major), with `i` indexing the `y` direction and `j` indexing the `x`
direction. `cell_centers[k]` corresponds exactly to column `k` of `A`, element
`k` of `s`, and row/column `k` of `Cs`. This is fixed once in `geometry.py` and
never deviated from elsewhere.

## Two different meanings of "calibrated"

> Experiment II generates the truth from the same prior the inversion assumes
> (`s_true ~ N(s0, Cs)`, `d | s_true ~ N(A s_true, Sigma_d)`). Under this
> generative model, the posterior is exactly the correct conditional
> distribution, so `Q ~ chi2(n)` is *guaranteed* by the model being
> self-consistent — this experiment validates the implementation, not a deeper
> statistical claim. Experiment III fixes `s_true` as a deterministic field not
> regarded as a draw from the prior. There is *no general guarantee* that
> `Q ~ chi2(n)` here; whether the posterior remains well-calibrated depends on
> how well the fixed truth matches what the prior expects. Never write "the
> posterior is calibrated" without specifying which of these two senses is
> meant.

## Spatial correlation and pooled calibration diagnostics

Per-cell standardized errors `z_j` are spatially correlated within a single
realization: the squared-exponential prior `Cs` induces spatial dependence
between nearby cells, and this dependence is then modified (not simply
inherited) by the likelihood/posterior update that produces `C_post` — the
posterior's correlation structure is not assumed to equal the prior's. A
diagnostic that pools `z` values across cells (e.g. a
goodness-of-fit test against `N(0, 1)`) therefore cannot treat those pooled
values as independent observations: doing so was tried during development
(a Kolmogorov-Smirnov test of `z`, pooled across all cells and realizations,
against `N(0, 1)`) and it failed reliably at the project's real grid scale
even though the underlying model and inference code were correct — the small
p-value reflected the violated independence assumption, not miscalibration.
That pooled-across-cells test has been removed from `test_calibration.py`.

Per-realization statistics such as `Q` do not have this problem: each
realization's `Q` is computed from one independent draw of `s_true` and
`epsilon`, so pooling `Q` *across realizations* is valid, and
`test_calibration.py`'s `Q`-vs-`chi2(n)` check relies on exactly that.
Where per-cell behavior needs checking, `test_calibration.py` instead runs
separate KS tests against `N(0, 1)` for a small number of individual,
deterministically chosen cells (each test pooling only across realizations
for that one cell), Bonferroni-corrected across those tests.

(This project has not attempted to quantify an "effective independent
sample size" for the pooled case — only that treating the pooled count as
the true sample size is wrong.)

## Why Experiment III's per-cell `z` is not `N(0, 1)`, even when well matched

The per-cell KS/z-score diagnostics used in Experiment II rely on `s_true`
itself being a draw from the prior: conditional on the data, `s_true - s_post
~ N(0, C_post)` is exact, so standardizing by `sqrt(diag(C_post))` gives
`z_j ~ N(0, 1)` by construction. Experiment III does not have this property.
There, `s_true` is fixed and only the observation noise `epsilon` is
resampled across realizations, so the *frequentist sampling distribution* of
`z_j` (over repeated noise draws, truth held fixed) is governed by a
different covariance than `C_post`.

Write the posterior mean's fixed-truth error as

$$
s_{\rm true} - s_{\rm post} = (I - KA)(s_{\rm true} - s_0) - K\epsilon,
\qquad
K = C_s A^\top (A C_s A^\top + \Sigma_d)^{-1},
$$

which decomposes into a **deterministic regularization bias**
`(I - KA)(s_true - s0)` (fixed, since `s_true` is fixed — it does not vary
across realizations) and a **noise-driven term** `-K epsilon` (the only
random part, since only `epsilon` varies). The repeated-noise sampling
covariance of `s_post` is therefore

$$
V_{\rm post} = C_{\rm post}\, A^\top \Sigma_d^{-1} A\, C_{\rm post},
$$

which is generally *not* equal to `C_post` itself (`C_post` is the Bayesian
posterior covariance, which also accounts for prior uncertainty about which
`s_true` might be true — a component that is simply absent when `s_true` is
fixed). Standardizing the fixed-truth sampling variability by `C_post`'s
diagonal instead of `V_post`'s gives a per-cell ratio

$$
r_j = \sqrt{\frac{(V_{\rm post})_{jj}}{(C_{\rm post})_{jj}}},
$$

which is the *theoretically expected* `std(z_j)` under repeated noise for a
fixed truth — not 1. Concretely:

- for a well-matched fixed truth (the smooth Gaussian-anomaly truth), `r_j`
  is well below 1 (empirically `r_j` averages ≈0.27 across cells for the
  frozen Experiment III smooth baseline, matching the empirical `std(z_j)`
  per cell to within ≈0.01), producing a *systematically narrow* `z`
  distribution;
- for a poorly matched fixed truth (the sharp checkerboard truth), the
  regularization bias term `(I - KA)(s_true - s0)` is large and the noise
  term is comparatively small, so the fixed-truth `z_j` is dominated by a
  large, nearly-deterministic offset rather than the noise-driven spread
  `V_post` describes — producing a *systematically wide* (and off-center)
  `z` distribution instead.

**Neither behavior is an Experiment-II-style calibration failure** — it is
the expected consequence of removing the prior-variability contribution to
the variance decomposition by fixing `s_true`. Note also that the *pooled*
`std(z)` reported elsewhere for a fixed truth (pooling across cells, not
just realizations) is not directly comparable to `r_j`: by the law of total
variance, `Var(pooled z) = mean_j(Var(z_j)) + Var_j(mean(z_j))`, and the
per-cell theoretical `r_j` only predicts the first (within-cell) term — the
second term (the spread, across cells, of the *bias* `(I-KA)(s_true-s0)`,
which varies spatially for a non-uniform truth like the Gaussian anomaly)
adds further pooled variance on top.

**Coverage and the fixed-truth behavior of `Q` are the appropriate Experiment
III diagnostics** (see the "Two different meanings of calibrated" note
above). The five-cell Bonferroni-corrected KS tests in `test_calibration.py`
are Experiment-II-specific reference checks — under Experiment III they are
expected to fail regardless of how well-matched the fixed truth is, for the
reasons above, and must not be read as a coverage/calibration verdict for
Experiment III.

## Out of scope

This is deliberately a finite-dimensional linear-Gaussian problem with an exact
closed-form posterior. The following are excluded by design, not by oversight:
iterative optimization, MCMC, neural networks or neural-operator surrogates,
curved-ray or nonlinear eikonal solvers, and external tomography frameworks
(e.g. SimPEG).
