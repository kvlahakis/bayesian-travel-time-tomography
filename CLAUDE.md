# Project reference: Bayesian Linear Travel-Time Tomography

## Goal

This project implements and validates a finite-dimensional Bayesian linear-Gaussian
travel-time tomography model, as a small, well-tested Python research-software
package. The scientific content is fully specified in `experiment.pdf`:
Experiment I (construction/validation), Experiment II (correctly-specified
Bayesian calibration), Experiment III (fixed-truth coverage, under both a
truth well matched to the prior and a truth deliberately mismatched with it),
and Experiment IV (fixed-truth reconstruction and posterior-uncertainty
comparison across three predetermined acquisition geometries).
This document records the *software* structure, module contracts, and
conventions actually used to build it, and is the reference for any further
work on this codebase.

Before writing any new function, consult `ARCHITECTURE.md`. Place new functionality in
the module whose responsibility already matches it. Do not create new modules unless an
existing module has become genuinely overloaded.

## Repo layout

```
tomography/
├── tomography/
│   ├── __init__.py
│   ├── config.py          # hierarchical dataclass schema for every experiment config
│   ├── geometry.py        # domain, grid, sources/receivers, sensitivity matrix A
│   ├── forward.py         # forward model, synthetic truth, noisy data generation
│   ├── prior.py           # s0, Cs (squared-exponential), prior sampling
│   ├── inversion.py       # closed-form Gaussian posterior
│   ├── diagnostics.py     # relative error, z-scores, Mahalanobis Q, coverage
│   ├── experiments.py     # single-run + repeated-run harnesses for Experiments I-III
│   └── plots.py           # all figures
├── scripts/
│   └── run_experiment.py  # thin CLI: YAML -> config -> experiments.py -> results/
├── tests/
│   ├── test_geometry.py
│   ├── test_forward.py
│   ├── test_prior.py
│   ├── test_inversion.py
│   ├── test_diagnostics.py     # deterministic unit tests only
│   ├── test_calibration.py     # stochastic integration tests (fixed seed)
│   └── test_experiments.py     # end-to-end experiment checks, including a
│                                #  same-config/same-seed reproducibility check
├── configs/                # YAML configs (named `configs/`, not `experiments/`,
│   ├── baseline.yaml       #  to avoid clashing with "Experiment I-III" terminology)
│   ├── calibration_correct.yaml
│   └── calibration_fixed_truth.yaml
├── notebooks/
│   └── report.ipynb        # NOT YET CREATED — see "Implementation history"
│                            #  below; this is the only work remaining
├── results/                 # gitignored, EXCEPT results/baselines/ (see
│   └── baselines/           #  .gitignore's explicit tracked-exception rule)
├── figures/                 # tracked — curated final figures for README/site
├── ARCHITECTURE.md
├── README.md
├── pyproject.toml
├── .gitattributes           # Git LFS tracking rule for *.npz baseline files
└── .gitignore
```

If `tests/`, `configs/`, or `results/baselines/` in the actual repository differ
from this listing, treat the repository as authoritative and update this file
to match — this listing should always describe what exists, not what was once
planned. The one deliberate exception is `notebooks/report.ipynb`, explicitly
marked above as not yet created; once it exists, drop that annotation.

## Conventions (do not deviate from these without updating this file)

- **Cell ordering.** Cell index `k` corresponds to `(i, j)` via `k = i * n + j`
  (row-major), with `i` indexing the `y` direction and `j` indexing the `x` direction.
  `cell_centers[k]` corresponds exactly to column `k` of `A`, element `k` of `s`,
  and row/column `k` of `Cs`. Documented once in `geometry.py`; every other module
  assumes it.
- **Source/receiver spacing.** `Ns` sources and `Nr` receivers are placed uniformly on
  the open interval `(0, W)`, excluding the corners:
  `np.linspace(0, W, Ns + 2)[1:-1]` (and likewise for receivers). This avoids the
  degenerate corner-ray geometry that including endpoints would introduce.
- **Cell boundary ownership.** Grid cells use a half-open convention,
  `[x_min, x_max) x [y_min, y_max)`, except the outermost row and column, which
  include the domain's outer boundary. This is documented in `geometry.py`; no test
  exercises this edge case directly (see the geometry test note below for why).

## Module contracts

### `geometry.py`
- `make_grid(W, n)` → uniform `n x n` grid of square cells over `[0,W] x [0,W]`;
  returns cell centers `(n**2, 2)` array (ordered per the convention above) and cell
  boundaries.
- `make_sources_receivers(W, Ns, Nr)` → source and receiver coordinate arrays, per
  the spacing convention above.
- `build_sensitivity_matrix(sources, receivers, grid)` → `A` of shape `(m, n**2)`,
  via an exact ray-cell intersection-length algorithm (Siddon-style / DDA grid
  traversal, not pixel rasterization).
- `check_row_sums(A, ray_lengths, atol=1e-8)` — validation helper: row sums of `A`
  must equal total ray length.

### `forward.py`
- `forward(A, s)` → `t`.
- `add_noise(t, sigma, rng)` → `d`.
- `synthetic_truth_gaussian_anomaly(grid, s_bg, delta_s, x0, y0, r)` → `s_true`
  (background plus localized Gaussian bump; the Experiment I truth, reused as
  Experiment III's well-matched fixed truth).
- `synthetic_truth_checkerboard(grid, s_bg, delta_s, block_size)` → `s_true` with
  sharp/discontinuous structure. Used as Experiment III's mismatched fixed truth,
  to test how the posterior behaves when the fixed truth is poorly represented by
  the smooth prior.

### `prior.py`
- `squared_exponential_cov(cell_centers, tau2, ell, jitter_relative=1e-6)` → `Cs`,
  implemented as `Cs = Cs_raw + (jitter_relative * tau2) * I`. Squared-exponential
  covariance matrices are frequently near-singular at fine grid spacing or large
  `ell`; this jitter is required, not optional.
- `sample_prior(s0, Cs, rng)` → `s`.

### `inversion.py`
Two explicit functions — not one function with an ambiguous signature:
- `posterior_general(A, d, Sigma_d, Cs, s0)` → `(s_post, C_post)`, the general form.
- `posterior_isotropic(A, d, sigma2, Cs, s0)` → `(s_post, C_post)`, the `Sigma_d =
  sigma2 * I` special case. This is the one used throughout the implemented
  experiments.
- `Cs^-1` and `H^-1` (`H = A^T Sigma_d^-1 A + Cs^-1`) are never formed explicitly.
  `Cs^-1` is applied via a Cholesky-based solve against `Cs`; `H` is assembled and
  factored once via Cholesky; `s_post` comes from triangular solves against that
  factorization. `C_post` is formed explicitly only when needed as an output.

### `diagnostics.py`
- `relative_error(s_post, s_true)`.
- `standardized_errors(s_true, s_post, C_post)` → `z` (per-cell).
- `mahalanobis(s_true, s_post, C_post)` → `Q`.
- `marginal_coverage(s_true, s_post, C_post, alpha)` → covered-fraction for a
  single realization.
- `empirical_coverage(truths, posterior_means, posterior_covariances, alpha)` →
  covered-fraction aggregated across repeated realizations. Feeds the
  nominal-vs-empirical coverage plot.
- This module is the foundation of every calibration claim in the project. Its
  deterministic properties are tested directly (`test_diagnostics.py`); its
  statistical behavior is tested separately (`test_calibration.py`).

### `experiments.py`
- `ExperimentResults` (dataclass): `truths`, `posterior_means`, `posterior_stds`,
  `relative_errors`, `z_scores`, `mahalanobis` (each shaped `(n_repeats, ...)`),
  and `coverage` (`alpha -> empirical coverage fraction`).
- `run_once(config)` / `run_repeated(config, n_repeats, seed)`: the generic
  single- and repeated-realization harnesses, drawing `s_true ~ N(s0, Cs)` from
  `config` itself.
- `run_correctly_specified(config, n_repeats)` — Experiment II: delegates to
  `run_repeated`, with a precondition check that `ell_true == ell_infer` and
  `sigma_true == sigma_infer` (see `config.py` below for why these are separate
  fields at all).
- `run_fixed_truth(config, s_true, n_repeats)` — Experiment III: fixed `s_true`,
  repeated noise draws. Called once with the smooth truth and once with the
  sharp/checkerboard truth from `forward.py`.

### `plots.py`
- Three-panel field plot: truth / posterior mean / posterior std.
- `Q` histogram vs. theoretical `chi2_n` density overlay.
- Empirical coverage vs. nominal coverage level.

### `config.py`
Hierarchical schema — not one flat dataclass — so that `sigma_true`/`sigma_infer`
and `ell_true`/`ell_infer` are separate required fields rather than positional
arguments that could be confused with each other:
```python
@dataclass
class GridConfig:
    W: float
    n: int

@dataclass
class AcquisitionConfig:
    Ns: int
    Nr: int

@dataclass
class NoiseConfig:
    sigma_true: float
    sigma_infer: float

@dataclass
class PriorConfig:
    tau2: float
    ell_true: float
    ell_infer: float
    s_bg: float
    jitter_relative: float = 1e-6

@dataclass
class FixedTruthConfig:
    # No default: it materially changes the truth field (and every downstream
    # diagnostic), so a config running the checkerboard variant must state it
    # explicitly. Only used by Experiment III's sharp/checkerboard truth.
    checkerboard_block_size: int

@dataclass
class ExperimentConfig:
    grid: GridConfig
    acquisition: AcquisitionConfig
    noise: NoiseConfig
    prior: PriorConfig
    n_repeats: int
    seed: int
    fixed_truth: FixedTruthConfig | None = None
```
`scripts/run_experiment.py` loads a YAML into `ExperimentConfig`. Every experiment
is fully reproducible from a single config file plus its seed.

`sigma_true`/`sigma_infer` and `ell_true`/`ell_infer` exist as separate fields even
though every experiment in this project sets each pair equal (`_check_correctly_specified`
enforces this for Experiments II and III): the separate-field structure is what
makes that invariant checkable at all, and keeps the naming unambiguous should the
model ever need to generate a truth under one setting and infer under another.

## Out of scope — do not add these

This is deliberately a finite-dimensional linear-Gaussian problem with an exact
closed-form posterior. Do not introduce: iterative optimization, MCMC, neural
networks or neural-operator surrogates, curved-ray or nonlinear eikonal solvers, or
external tomography frameworks (e.g. SimPEG).

The project's scope is Experiments I-IV. A controlled prior-misspecification
experiment (varying the correlation length used to generate the truth against the
one used for inference) and a controlled noise-misspecification experiment (varying
the true versus assumed noise level) are not part of this project. The fixed-truth
comparison in Experiment III already demonstrates the central phenomenon those
experiments would also illustrate — a well-defined posterior failing to describe
physical uncertainty under model mismatch — via a mismatched truth rather than a
mismatched prior or noise model.

Experiment IV compares three *predetermined* acquisition geometries (uniform, and
two boundary-clustered layouts at fixed gamma=0.70 and gamma=0.40) at a fixed ray
count (`Ns=Nr=16`, `m=256`). It is not a systematic acquisition-geometry sweep, not
a continuous or gradient-based sensor-placement optimizer, and not a claim of an
optimal or globally best geometry — those remain out of scope. Do not add:
additional gamma values, asymmetric or randomized layouts, any optimizer over
sensor position (genetic, gradient-based, or random-restart), additional
acquisition densities, or additional grid resolutions for Experiment IV. If a task
seems to call for any of the above, stop and flag it rather than implementing it.

## Testing requirements

- `test_geometry.py`: row sums of `A` equal true ray length to numerical precision;
  a ray passing through grid vertices (but not lying along a shared cell edge) hits
  exactly the expected cells. (A ray lying exactly along a cell boundary is
  deliberately not tested — that case depends on the half-open ownership convention
  and is not worth a dedicated test.)
- `test_forward.py`:
  - noiseless forward model matches a hand-computed toy case (e.g. a 2x2 grid with
    one ray of known geometry);
  - constant-slowness sanity check: for `s = c * ones(n_cells)`, `A @ s` equals
    `c * ray_lengths` exactly (up to floating-point tolerance).
- `test_prior.py`: `Cs` is symmetric (`Cs ≈ Cs.T`) and positive definite
  (`eigvalsh(Cs) > 0`, Cholesky succeeds) across a range of `ell` and grid
  resolutions; large-sample empirical covariance of `sample_prior` draws matches
  `Cs`.
- `test_inversion.py`:
  - `C_post` is symmetric and positive definite;
  - posterior reduces to the prior as `sigma2 -> infinity` (uninformative data);
  - posterior uncertainty shrinks as `sigma2 -> 0` or as ray density increases;
  - `posterior_general` and `posterior_isotropic` agree when `Sigma_d = sigma2 * I`
    is passed to both.
- `test_diagnostics.py` (deterministic only — no repeated sampling, no
  goodness-of-fit checks):
  - `standardized_errors` uses the correct diagonal of `C_post`;
  - `mahalanobis` agrees with a hand-computed small-matrix example;
  - `marginal_coverage` agrees with a hand-computed example;
  - `Q == z^T R^-1 z` in a small known case, where `R` is the correlation form of
    `C_post`.
- `test_calibration.py` (stochastic integration tests — fixed seed, generous
  acceptance tolerances, run at the project's real `n=20` scale, not a toy grid):
  - the global `Q`-vs-`chi2(n_cells)` KS test (valid because each realization
    contributes one independent `Q`);
  - scalar `mean(z)`/`std(z)` checks, retained as descriptive marginal summaries
    without assuming an i.i.d. interpretation of the pooled values;
  - per-cell KS tests against `N(0, 1)` for a small number of deterministic
    representative cells (domain center, near-boundary, max/min posterior
    variance), Bonferroni-corrected across those tests — never a single KS test
    pooling `z` across all cells, which is statistically invalid (see
    `ARCHITECTURE.md`'s "Spatial correlation and pooled calibration diagnostics").
- `test_experiments.py`: end-to-end shape/validity checks on `run_once`/
  `run_repeated`/`run_correctly_specified`/`run_fixed_truth`, plus a same-config,
  same-seed reproducibility check confirming numerically identical
  `ExperimentResults`.

## Implementation history

This is the order in which the project was built and validated, kept here as a
record of the dependency discipline that was followed (each step's tests passed
before the next was built on top of it — several real issues, documented in
`ARCHITECTURE.md`, were caught only because of this ordering):

1. `geometry.py` + `test_geometry.py`.
2. `forward.py` + `test_forward.py`.
3. `prior.py` + `test_prior.py`.
4. `inversion.py` + `test_inversion.py`.
5. Experiment I (construction/validation).
6. `diagnostics.py` + `test_diagnostics.py`.
7. Experiment II (correctly specified Bayesian calibration).
8. Experiment III (fixed truth, both smooth and sharp), including a follow-up
   diagnostic investigation into an observed boundary/interior coverage effect
   (see `ARCHITECTURE.md`).
9. Experiment IV (fixed-truth reconstruction and posterior-uncertainty comparison
   across three predetermined acquisition geometries — uniform, mild boundary
   clustering, strong boundary clustering — at `m=256`, reusing Experiment III's
   smooth truth). **NOT YET IMPLEMENTED as of this writing**: its specification
   and already-validated reference numbers are recorded in `experiment.pdf`. The
   repo layout and testing-requirements sections below describe the codebase as
   it stood through Experiment III and have not yet been updated for Experiment
   IV's new files — do not treat their silence on Experiment IV as evidence it
   doesn't exist or isn't authorized; once implemented, sync those sections in
   the same pass rather than leaving this file stale.

This is the complete planned set of experiments for this project (see "Out of
scope" above, which also governs what Experiment IV itself may and may not grow
into). Beyond implementing Experiment IV and then syncing the repo layout and
testing-requirements sections to match, the only other work remaining is
assembling `notebooks/report.ipynb` — a presentation task drawing on the curated
`figures/` outputs and the frozen `results/baselines/` reference results, not a
new experiment, and it should not introduce any numerical result beyond what
those frozen baselines already contain.

## Documentation requirement: `ARCHITECTURE.md`

Keep this file current. It contains the module dependency chain:

```
             geometry
            /        \
           v          v
       forward       prior
            \        /
             v      v
            inversion
                |
                v
           diagnostics
                |
                v
            experiments
                |
                v
              plots
                |
                v
     scripts / notebook
```

It also states, verbatim or near-verbatim, the single most important conceptual
distinction in the project, which no future edit should blur:

> **Two different meanings of "calibrated."** Experiment II generates the truth from
> the same prior the inversion assumes (`s_true ~ N(s0, Cs)`, `d | s_true ~
> N(A s_true, Sigma_d)`). Under this generative model, the posterior is exactly the
> correct conditional distribution, so `Q ~ chi2(n_cells)` is *guaranteed* by the model
> being self-consistent — this experiment validates the implementation, not a
> deeper statistical claim. Experiment III fixes `s_true` as a deterministic field
> not regarded as a draw from the prior. There is *no general guarantee* that
> `Q ~ chi2(n_cells)` here; whether the posterior remains well-calibrated depends on how
> well the fixed truth matches what the prior expects. Never write "the posterior
> is calibrated" without specifying which of these two senses is meant.

`ARCHITECTURE.md` has since grown further sections built on this same discipline
(why pooling `z` across cells is statistically invalid; why Experiment III's
per-cell `z` isn't `N(0, 1)` even for a well-matched truth; the boundary/interior
coverage investigation). If further diagnostic work happens on this codebase, add
to `ARCHITECTURE.md` in that same style: state plainly what was checked, what it
does and does not establish, and where the investigation stopped.

## Implementation notes

- Every config carries an explicit random seed for reproducibility.
- `scripts/run_experiment.py` is a thin CLI only: parse args -> load YAML into the
  `config.py` schema -> call the appropriate `experiments.py` function -> save to
  `results/` -> optionally call `plots.py`. No mathematical logic lives in
  `scripts/`; if a formula ends up there, move it into the package.
- `results/` is gitignored except `results/baselines/`, which is explicitly
  tracked (see `.gitignore`) and holds the frozen, validated reference results
  named in `README.md` — do not overwrite those files with a different seed or
  configuration without also updating `README.md`'s note about them.
