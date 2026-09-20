# Build instructions: Bayesian Linear Travel-Time Tomography

## Goal

Implement the experiment plan below as a small, well-tested Python research-software
package. The scientific content is fully specified in `experiment.pdf`
(construction/validation, then five experiments: correctly-specified Bayesian
calibration, fixed-truth coverage, prior misspecification, noise misspecification,
and acquisition-geometry sensitivity). This document specifies the *software*
structure, module contracts, conventions, and build order — not the math, which the
PDF covers in full.

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
│   ├── experiments.py     # single-run + repeated-run harnesses for Experiments I-V
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
│   └── test_reproducibility.py
├── configs/                # YAML configs (named `configs/`, not `experiments/`,
│   ├── baseline.yaml       #  to avoid clashing with "Experiment I-V" terminology)
│   ├── calibration_correct.yaml
│   ├── calibration_fixed_truth.yaml
│   ├── misspec_prior.yaml
│   └── misspec_noise.yaml
├── notebooks/
│   └── report.ipynb        # polished walkthrough notebook for the website
├── results/                 # gitignored — raw run outputs (npz/csv/png)
├── figures/                 # tracked — curated final figures for README/site
├── ARCHITECTURE.md
├── README.md
├── pyproject.toml
└── .gitignore
```

## Conventions (fix these before writing code — do not let the agent improvise them)

- **Cell ordering.** Cell index `k` corresponds to `(i, j)` via `k = i * n + j`
  (row-major), with `i` indexing the `y` direction and `j` indexing the `x` direction.
  `cell_centers[k]` must correspond exactly to column `k` of `A`, element `k` of `s`,
  and row/column `k` of `Cs`. Document this once in `geometry.py` and never deviate
  from it — every other module assumes it.
- **Source/receiver spacing.** Place `Ns` sources and `Nr` receivers uniformly on the
  open interval `(0, W)`, excluding the corners:
  `np.linspace(0, W, Ns + 2)[1:-1]` (and likewise for receivers). This avoids the
  degenerate corner-ray geometry that including endpoints would introduce.
- **Cell boundary ownership.** Grid cells use a half-open convention,
  `[x_min, x_max) x [y_min, y_max)`, except the outermost row and column, which
  include the domain's outer boundary. This is a documentation requirement for
  `geometry.py`, not something that needs a dedicated test — see the geometry test
  note below for how to avoid needing to exercise this edge case directly.

## Module contracts

### `geometry.py`
- `make_grid(W, n)` → uniform `n x n` grid of square cells over `[0,W] x [0,W]`;
  returns cell centers `(n**2, 2)` array (ordered per the convention above) and cell
  boundaries.
- `make_sources_receivers(W, Ns, Nr)` → source and receiver coordinate arrays, per
  the spacing convention above.
- `build_sensitivity_matrix(sources, receivers, grid)` → `A` of shape `(m, n**2)`.
  Use an **exact** ray–cell intersection-length algorithm (e.g. a Siddon-style /
  DDA grid-traversal method). Do not use `skimage.draw.line` — it returns pixel
  indices, not exact intersection lengths, and will silently corrupt the forward
  model.
- `check_row_sums(A, ray_lengths, atol=1e-8)` — validation helper per the PDF's
  ray-construction step (row sums of `A` must equal total ray length).

### `forward.py`
- `forward(A, s)` → `t`.
- `add_noise(t, sigma, rng)` → `d`.
- `synthetic_truth_gaussian_anomaly(grid, s_bg, delta_s, x0, y0, r)` → `s_true`
  (background plus localized Gaussian bump, per the PDF's Experiment I truth).
- Also provide a **second** synthetic-truth generator with sharp/discontinuous
  structure (e.g. a checkerboard or a rectangular block anomaly). Use it in
  Experiment III to test how the posterior behaves when the fixed truth is poorly
  represented by the smooth prior — this is an empirical question the experiment
  answers, not a result to assume in advance.

### `prior.py`
- `squared_exponential_cov(cell_centers, tau2, ell, jitter_relative=1e-6)` → `Cs`.
  Implement as `Cs = Cs_raw + (jitter_relative * tau2) * I`, matching the PDF's
  `ε ~ 1e-6 τ²` exactly (not an absolute `1e-6 * I`, which is a different quantity
  and the wrong units). Squared-exponential covariance matrices are frequently
  near-singular at fine grid spacing or large `ell`; omitting this jitter will
  cause intermittent Cholesky failures downstream.
- `sample_prior(s0, Cs, rng)` → `s`.

### `inversion.py`
Two explicit functions — not one function with an ambiguous signature:
- `posterior_general(A, d, Sigma_d, Cs, s0)` → `(s_post, C_post)`, implementing the
  general form from the PDF.
- `posterior_isotropic(A, d, sigma2, Cs, s0)` → `(s_post, C_post)`, implementing the
  `Σd = σ²I` special case. This is the one used throughout the actual experiments.
- **Never form `Cs⁻¹` or `H⁻¹` explicitly**, where `H = A^T Σd^-1 A + Cs^-1`. Apply
  `Cs⁻¹` via a Cholesky-based solve against `Cs`, assemble `H`, factor `H` once via
  Cholesky, and obtain `s_post` via triangular solves against that factorization.
  Only form `C_post` explicitly when it is needed as an output (e.g. for
  `diag(C_post)` or as a diagnostic) — solving `H s = b` does not require it.

### `diagnostics.py`
- `relative_error(s_post, s_true)`.
- `standardized_errors(s_true, s_post, C_post)` → `z` (per-cell).
- `mahalanobis(s_true, s_post, C_post)` → `Q`.
- `marginal_coverage(s_true, s_post, C_post, alpha)` → covered-fraction for a
  **single** realization.
- `empirical_coverage(truths, posterior_means, posterior_covariances, alpha)` →
  covered-fraction aggregated **across repeated realizations**, per-cell and/or
  overall. This is the function that feeds the nominal-vs-empirical coverage plot.
- This module is the foundation of every calibration claim in the write-up. Its
  deterministic properties are tested directly (`test_diagnostics.py`); its
  statistical behavior is tested separately (`test_calibration.py`) — see Testing
  requirements below.

### `experiments.py`
- Define a documented result type (not an arbitrary dict):
  ```python
  @dataclass
  class ExperimentResults:
      truths: np.ndarray              # (n_repeats, n_cells)
      posterior_means: np.ndarray     # (n_repeats, n_cells)
      posterior_stds: np.ndarray      # (n_repeats, n_cells)
      relative_errors: np.ndarray     # (n_repeats,)
      z_scores: np.ndarray            # (n_repeats, n_cells)
      mahalanobis: np.ndarray         # (n_repeats,)
      coverage: dict                  # alpha -> empirical coverage fraction
  ```
  `run_once(config)` returns a single realization's worth of these fields;
  `run_repeated(config, n_repeats, seed)` returns a stacked `ExperimentResults`.
- One function per experiment family, each wiring the modules above under a
  different rule for generating `s_true`. Use `Cs_true`/`Cs_infer` and
  `sigma_true`/`sigma_infer` as variable names throughout — never a bare `Cs` or
  `sigma` in code paths where truth-generation and inference could be confused:
  - `run_correctly_specified(config, n_repeats)` — Experiment II:
    `s_true ~ N(s0, Cs)`, inference uses the same `Cs`.
  - `run_fixed_truth(config, s_true, n_repeats)` — Experiment III: fixed `s_true`,
    repeated noise draws. Run once with the smooth truth and once with the
    sharp/discontinuous truth from `forward.py`.
  - `run_prior_misspecified(config, ell_true, ell_infer, n_repeats)` — Experiment IV.
    Build `Cs_true = squared_exponential_cov(..., ell=ell_true)` and
    `Cs_infer = squared_exponential_cov(..., ell=ell_infer)` as two explicitly
    separate objects. Generate `s_true` and `d` using `Cs_true` only; call
    `posterior_isotropic` using `Cs_infer` only. `ell_infer` must never be used to
    generate the truth.
  - `run_noise_misspecified(config, sigma_true, sigma_infer, n_repeats)` — Experiment
    V. Generate `epsilon ~ N(0, sigma_true**2 * I)`; call `posterior_isotropic`
    with `sigma_infer**2`. `sigma_infer` must never be used to generate data.
  - `run_geometry_sweep(config, geometries)` — geometry sensitivity: for each
    `(Ns, Nr)`, hold the grid, `s0`, `Cs`, and `sigma` fixed — only the acquisition
    geometry changes. For each geometry, record `rank(A)`, the singular values of
    `A` (via `np.linalg.svd`), and, separately, the eigenvalues and condition number
    of the regularized system `H = A^T Σd^-1 A + Cs^-1` via `np.linalg.eigvalsh(H)`
    (H is symmetric positive definite — use the symmetric-matrix routine, not a
    generic condition-number function). Do not conflate "singular values of `A`"
    with "eigenvalues of `H`" in code, plots, or the write-up — they are different
    quantities and only the latter directly governs `C_post`.

### `plots.py`
- Three-panel field plot: truth / posterior mean / posterior std.
- `Q` histogram vs. theoretical χ²_n density overlay.
- Empirical coverage vs. nominal coverage level.
- Singular-value spectrum of `A` **and**, separately, the eigenvalue spectrum of the
  regularized system `H`, both as a function of `(Ns, Nr)`.

### `config.py`
Hierarchical schema — do not use one flat dataclass. This structurally prevents
`sigma_true`/`sigma_infer` or `ell_true`/`ell_infer` from being confused with each
other, since they are separate required fields rather than positional arguments:
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
class ExperimentConfig:
    grid: GridConfig
    acquisition: AcquisitionConfig
    noise: NoiseConfig
    prior: PriorConfig
    n_repeats: int
    seed: int
```
`scripts/run_experiment.py` loads a YAML into `ExperimentConfig`. Every experiment
must be fully reproducible from a single config file plus its seed.

## Out of scope — do not let the agent add these

This is deliberately a finite-dimensional linear-Gaussian problem with an exact
closed-form posterior. Do not introduce: iterative optimization, MCMC, neural
networks or neural-operator surrogates, curved-ray or nonlinear eikonal solvers, or
external tomography frameworks (e.g. SimPEG). If a task seems to call for any of
these, stop and flag it rather than implementing it — it is out of scope for this
project by design, not an oversight to fix.

## Testing requirements

- `test_geometry.py`: row sums of `A` equal true ray length to numerical precision;
  a ray passing through grid **vertices** (but not lying along a shared cell edge)
  hits exactly the expected cells. (Deliberately avoid testing a ray that lies
  exactly along a cell boundary — that case depends on the half-open ownership
  convention and is not worth a dedicated test.)
- `test_forward.py`:
  - noiseless forward model matches a hand-computed toy case (e.g. a 2x2 grid with
    one ray of known geometry);
  - **constant-slowness sanity check**: for `s = c * ones(n_cells)`, `A @ s` must
    equal `c * ray_lengths` exactly (up to floating-point tolerance) — this single
    test validates the physical interpretation of `A` directly and is worth having
    even though `test_geometry.py` already checks row sums.
- `test_prior.py`: `Cs` is symmetric (`Cs ≈ Cs.T`) and positive definite
  (`eigvalsh(Cs) > 0`, Cholesky succeeds) across a range of `ell` and grid
  resolutions; large-sample empirical covariance of `sample_prior` draws matches
  `Cs`.
- `test_inversion.py`:
  - `C_post` is symmetric and positive definite;
  - posterior reduces to the prior as `sigma2 -> infinity` (uninformative data);
  - posterior uncertainty shrinks as `sigma2 -> 0` or as ray density increases
    (informative data);
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
  acceptance tolerances, clearly labeled as statistical rather than exact):
  - under the correctly-specified generative model (Experiment II setup), the
    empirical distribution of `Q` over many realizations matches `chi2(n)`, and `z`
    matches `N(0,1)`. A stochastic test can fail from ordinary Monte Carlo
    variation even when the code is correct — keep this test separate from
    `test_diagnostics.py` for exactly that reason, and use a fixed seed so failures
    are reproducible rather than flaky.
- `test_reproducibility.py`: running the same config with the same seed twice
  produces numerically identical `ExperimentResults`.

## Build order

1. `geometry.py` + `test_geometry.py`. Everything downstream depends on `A` being
   correct — validate it first.
2. `forward.py` + `test_forward.py`.
3. `prior.py` + `test_prior.py`.
4. `inversion.py` + `test_inversion.py`.
5. Experiment I (construction/validation): one run using the modules above, three-
   panel plot. This confirms the whole pipeline works before any calibration claim
   is attempted.
6. `diagnostics.py` + `test_diagnostics.py`, validated in isolation on deterministic
   cases before being trusted on real experiment output.
7. Experiment II (correctly-specified): repeated-run harness in `experiments.py`,
   `test_calibration.py` confirms `Q ~ chi2(n)`.
8. Experiment III (fixed truth, both smooth and sharp): same harness, different
   truth-generation rule. Produces the coverage comparison that is the project's
   central result.
9. Experiment IV (prior misspecification).
10. Experiment V (noise misspecification).
11. Geometry sensitivity sweep: vary `(Ns, Nr)`, plot both the singular-value
    spectrum of `A` and the eigenvalue spectrum of the regularized system `H`.
12. `test_reproducibility.py`.
13. `notebooks/report.ipynb`: assemble the final presentation notebook from the
    curated `figures/` outputs — this is the artifact that goes on the website.

Keep each step's tests passing before moving to the next. If `test_inversion.py`
fails, do not simultaneously debug the experiment harness built on top of it —
failures should always be localizable to a single module.

## Documentation requirement: `ARCHITECTURE.md`

Create this file and keep it current. It must contain the dependency chain:

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

It must also state, verbatim or near-verbatim, the following rule — this is the
single most important conceptual distinction in the project, and the write-up must
never blur it:

> **Two different meanings of "calibrated."** Experiment II generates the truth from
> the same prior the inversion assumes (`s_true ~ N(s0, Cs)`, `d | s_true ~
> N(A s_true, Sigma_d)`). Under this generative model, the posterior is exactly the
> correct conditional distribution, so `Q ~ chi2(n)` is *guaranteed* by the model
> being self-consistent — this experiment validates the implementation, not a
> deeper statistical claim. Experiment III fixes `s_true` as a deterministic field
> not regarded as a draw from the prior. There is *no general guarantee* that
> `Q ~ chi2(n)` here; whether the posterior remains well-calibrated depends on how
> well the fixed truth matches what the prior expects. Never write "the posterior
> is calibrated" without specifying which of these two senses is meant.

## Implementation notes

- Every config must carry an explicit random seed for reproducibility.
- Keep `scripts/run_experiment.py` a thin CLI only: parse args -> load YAML into
  the `config.py` schema -> call the appropriate `experiments.py` function -> save
  to `results/` -> optionally call `plots.py`. No mathematical logic should live
  in `scripts/`; if a formula ends up there, move it into the package.
- `results/` is gitignored (raw sweep outputs); `figures/` is tracked (curated,
  final plots only).
