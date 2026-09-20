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

## Out of scope

This is deliberately a finite-dimensional linear-Gaussian problem with an exact
closed-form posterior. The following are excluded by design, not by oversight:
iterative optimization, MCMC, neural networks or neural-operator surrogates,
curved-ray or nonlinear eikonal solvers, and external tomography frameworks
(e.g. SimPEG).
