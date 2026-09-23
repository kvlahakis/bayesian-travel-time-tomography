"""Single-run and repeated-run experiment harnesses for Experiments I-IV.

Experiment I (construction/validation) is implemented here as
`run_construction_validation`. It predates `diagnostics.py` and any
calibration claim: its only purpose is to confirm that the forward model and
Bayesian inversion are wired together correctly, via a single deterministic
run and the three-panel truth/posterior-mean/posterior-std plot.

Experiment II (correctly specified Bayesian calibration) is implemented as
`run_once` / `run_repeated` (the generic single- and repeated-realization
harnesses, since Experiment II needs nothing beyond `config` itself to define
its truth-generation rule) and `run_correctly_specified` (the named entry
point, which additionally checks that `config` really is correctly
specified). See `ARCHITECTURE.md` for what "correctly specified" means and
why `Q ~ chi2(n_cells)` here validates the implementation rather than a
deeper statistical claim.

Experiment III (`run_fixed_truth`) holds a deterministic, externally supplied
`s_true` fixed and repeats only the noise draw. Unlike Experiment II, nothing
about this truth is drawn from `Cs_infer`, so there is no guarantee that
`Q ~ chi2(n_cells)` here (see `ARCHITECTURE.md`); whether the posterior stays
calibrated depends entirely on how well `s_true` matches what `Cs_infer`
expects. `run_repeated` and `run_fixed_truth` differ only in what varies
across realizations (truth and noise, vs. noise alone), so they share a
common per-realization-diagnostics-and-stacking helper,
`_run_repeated_from_draws`.

Experiment IV (`run_acquisition_geometry_comparison`) compares reconstruction
and posterior-uncertainty metrics across a small, fixed set of predetermined
acquisition geometries (see `geometry.make_boundary_clustered_sources_receivers`),
using a *paired* Monte Carlo design: one shared standardized noise draw per
repetition, applied identically to every geometry, so per-repetition
differences isolate the effect of geometry from noise-realization variance.
It reuses Experiment III's smooth fixed truth. This is not a systematic
acquisition-geometry sweep or a sensor-placement optimizer -- the set of
geometries compared is fixed by the caller's config (see `CLAUDE.md`'s "Out
of scope" section for the explicit boundary).

A controlled prior-misspecification experiment and a controlled
noise-misspecification experiment remain deliberately out of this project's
scope (see `CLAUDE.md`), not future work to be added here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig, ExperimentIVConfig
from .diagnostics import (
    empirical_coverage,
    mahalanobis,
    marginal_coverage,
    pearson_correlation,
    posterior_information_metrics,
    relative_error,
    standardized_errors,
)
from .forward import add_noise, forward, synthetic_truth_gaussian_anomaly
from .geometry import (
    Grid,
    build_sensitivity_matrix,
    check_row_sums,
    make_boundary_clustered_sources_receivers,
    make_grid,
    make_sources_receivers,
)
from .inversion import posterior_isotropic
from .prior import sample_prior, squared_exponential_cov

# Nominal levels (1 - alpha) for the marginal credible-interval coverage
# tracked by every repeated-run experiment: 50%, 68%, 80%, 90%, 95%, 99%.
DEFAULT_COVERAGE_ALPHAS: tuple[float, ...] = (0.50, 0.32, 0.20, 0.10, 0.05, 0.01)


@dataclass
class ConstructionValidationResult:
    """Result of a single Experiment I run."""

    grid: Grid
    A: np.ndarray
    s_true: np.ndarray
    d: np.ndarray
    s_post: np.ndarray
    C_post: np.ndarray


def run_construction_validation(
    config: ExperimentConfig,
    x0: float | None = None,
    y0: float | None = None,
    delta_s: float | None = None,
    r: float | None = None,
) -> ConstructionValidationResult:
    """Experiment I: construct and validate the full tomography pipeline.

    Builds the grid and acquisition geometry, validates `A`'s row sums
    against known ray lengths, generates a smooth Gaussian-anomaly truth
    (PDF Section 7, Step 4), simulates noisy data, and computes the
    posterior. This run is not a misspecification experiment: it uses
    `config.noise.sigma_true` / `config.prior.ell_true` to generate the data
    and `config.noise.sigma_infer` / `config.prior.ell_infer` for inference,
    which `configs/baseline.yaml` sets equal to each other by design.

    `x0`, `y0`, `delta_s`, `r` configure the Gaussian anomaly and default to
    the domain center, one prior marginal std, and one sixth of the domain
    width, respectively, if not given.
    """
    grid = make_grid(config.grid.W, config.grid.n)
    sources, receivers = make_sources_receivers(
        config.grid.W, config.acquisition.Ns, config.acquisition.Nr
    )
    A = build_sensitivity_matrix(sources, receivers, grid)

    ray_lengths = np.array(
        [np.linalg.norm(receiver - source) for source in sources for receiver in receivers]
    )
    check_row_sums(A, ray_lengths)

    if x0 is None:
        x0 = config.grid.W / 2.0
    if y0 is None:
        y0 = config.grid.W / 2.0
    if delta_s is None:
        delta_s = np.sqrt(config.prior.tau2)
    if r is None:
        r = config.grid.W / 6.0

    s_true = synthetic_truth_gaussian_anomaly(
        grid, s_bg=config.prior.s_bg, delta_s=delta_s, x0=x0, y0=y0, r=r
    )

    t_true = forward(A, s_true)
    rng = np.random.default_rng(config.seed)
    d = add_noise(t_true, sigma=config.noise.sigma_true, rng=rng)

    s0 = config.prior.s_bg * np.ones(grid.n**2)
    Cs_infer = squared_exponential_cov(
        grid.cell_centers,
        tau2=config.prior.tau2,
        ell=config.prior.ell_infer,
        jitter_relative=config.prior.jitter_relative,
    )
    s_post, C_post = posterior_isotropic(
        A, d, sigma2=config.noise.sigma_infer**2, Cs=Cs_infer, s0=s0
    )

    return ConstructionValidationResult(
        grid=grid, A=A, s_true=s_true, d=d, s_post=s_post, C_post=C_post
    )


@dataclass
class SingleRunResult:
    """A single realization's worth of the `ExperimentResults` fields.

    `coverage` here is `diagnostics.marginal_coverage`'s across-*cells*
    fraction for this one realization -- with only one realization, that is
    the only coverage notion available. `ExperimentResults.coverage` is a
    different quantity: coverage aggregated across many realizations (and
    cells), via `diagnostics.empirical_coverage`.
    """

    truth: np.ndarray
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    relative_error: float
    z_scores: np.ndarray
    mahalanobis: float
    coverage: dict


@dataclass
class ExperimentResults:
    """Stacked diagnostics over `n_repeats` independent realizations."""

    truths: np.ndarray  # (n_repeats, n_cells)
    posterior_means: np.ndarray  # (n_repeats, n_cells)
    posterior_stds: np.ndarray  # (n_repeats, n_cells)
    relative_errors: np.ndarray  # (n_repeats,)
    z_scores: np.ndarray  # (n_repeats, n_cells)
    mahalanobis: np.ndarray  # (n_repeats,)
    coverage: dict  # alpha -> empirical coverage fraction


def _check_correctly_specified(config: ExperimentConfig) -> None:
    """Guard the "correctly specified" invariant that Experiment II's
    calibration guarantee (`Q ~ chi2(n_cells)`) depends on: the *same* Cs and
    sigma must be used to generate the truth and to perform inference. A
    config with `ell_true != ell_infer` or `sigma_true != sigma_infer` is a
    deliberate misspecification, which this project does not implement as a
    controlled experiment (see `CLAUDE.md`'s "Out of scope" section) but which
    this guard rejects here regardless, rather than silently invalidating the
    calibration claim.
    """
    if config.prior.ell_true != config.prior.ell_infer:
        raise ValueError(
            "run_once/run_repeated/run_correctly_specified require "
            f"ell_true == ell_infer (got ell_true={config.prior.ell_true}, "
            f"ell_infer={config.prior.ell_infer}); use "
            "run_prior_misspecified for ell_true != ell_infer."
        )
    if config.noise.sigma_true != config.noise.sigma_infer:
        raise ValueError(
            "run_once/run_repeated/run_correctly_specified require "
            f"sigma_true == sigma_infer (got sigma_true="
            f"{config.noise.sigma_true}, sigma_infer="
            f"{config.noise.sigma_infer}); use run_noise_misspecified for "
            "sigma_true != sigma_infer."
        )


def _build_geometry_and_inference_prior(
    config: ExperimentConfig,
) -> tuple[Grid, np.ndarray, np.ndarray, np.ndarray]:
    """Build the grid, A, Cs_infer, and s0 that a config alone determines --
    shared across every realization of any experiment built from this
    `config` (currently Experiments II and III), since none of these four
    depend on a particular realization's truth or noise draw.
    """
    grid = make_grid(config.grid.W, config.grid.n)
    sources, receivers = make_sources_receivers(
        config.grid.W, config.acquisition.Ns, config.acquisition.Nr
    )
    A = build_sensitivity_matrix(sources, receivers, grid)

    ray_lengths = np.array(
        [np.linalg.norm(receiver - source) for source in sources for receiver in receivers]
    )
    check_row_sums(A, ray_lengths)

    Cs_infer = squared_exponential_cov(
        grid.cell_centers,
        tau2=config.prior.tau2,
        ell=config.prior.ell_infer,
        jitter_relative=config.prior.jitter_relative,
    )
    s0 = config.prior.s_bg * np.ones(grid.n**2)
    return grid, A, Cs_infer, s0


def _run_repeated_from_draws(
    n_repeats: int,
    n_cells: int,
    draw_realization,
    alphas: tuple[float, ...],
) -> ExperimentResults:
    """Run `n_repeats` realizations from `draw_realization`, a zero-argument
    callable returning `(s_true, s_post, C_post)` for one realization
    (closing over whatever is fixed vs. drawn for the calling experiment),
    compute the standard per-realization diagnostics for each, and stack
    them into an `ExperimentResults`.

    Shared by every repeated-run experiment harness that reduces to "compute
    a posterior against some truth and some noisy data, once per
    realization" -- currently `run_repeated` (Experiment II) and
    `run_fixed_truth` (Experiment III), which differ only in what
    `draw_realization` holds fixed.
    """
    truths = np.empty((n_repeats, n_cells))
    posterior_means = np.empty((n_repeats, n_cells))
    posterior_stds = np.empty((n_repeats, n_cells))
    relative_errors = np.empty(n_repeats)
    z_scores = np.empty((n_repeats, n_cells))
    mahalanobis_values = np.empty(n_repeats)

    for k in range(n_repeats):
        s_true, s_post, C_post = draw_realization()
        truths[k] = s_true
        posterior_means[k] = s_post
        posterior_stds[k] = np.sqrt(np.diag(C_post))
        relative_errors[k] = relative_error(s_post, s_true)
        z_scores[k] = standardized_errors(s_true, s_post, C_post)
        mahalanobis_values[k] = mahalanobis(s_true, s_post, C_post)

    coverage = {
        alpha: empirical_coverage(truths, posterior_means, posterior_stds, alpha)["overall"]
        for alpha in alphas
    }

    return ExperimentResults(
        truths=truths,
        posterior_means=posterior_means,
        posterior_stds=posterior_stds,
        relative_errors=relative_errors,
        z_scores=z_scores,
        mahalanobis=mahalanobis_values,
        coverage=coverage,
    )


def _run_correctly_specified_realization(
    A: np.ndarray,
    Cs_infer: np.ndarray,
    s0: np.ndarray,
    sigma_true: float,
    sigma_infer: float,
    rng: np.random.Generator,
    alphas: tuple[float, ...],
) -> SingleRunResult:
    """One realization of Experiment II.

    Draws `s_true ~ N(s0, Cs_infer)` -- under the correctly specified model
    the truth-generating and inference covariances are, by definition, the
    same object, so `Cs_infer` legitimately plays both roles here. Generates
    one noisy data set and computes the posterior and its diagnostics.
    """
    s_true = sample_prior(s0, Cs_infer, rng)
    t_true = forward(A, s_true)
    d = add_noise(t_true, sigma=sigma_true, rng=rng)

    s_post, C_post = posterior_isotropic(
        A, d, sigma2=sigma_infer**2, Cs=Cs_infer, s0=s0
    )

    post_std = np.sqrt(np.diag(C_post))
    z = standardized_errors(s_true, s_post, C_post)
    Q = mahalanobis(s_true, s_post, C_post)
    coverage = {
        alpha: marginal_coverage(s_true, s_post, C_post, alpha) for alpha in alphas
    }

    return SingleRunResult(
        truth=s_true,
        posterior_mean=s_post,
        posterior_std=post_std,
        relative_error=relative_error(s_post, s_true),
        z_scores=z,
        mahalanobis=Q,
        coverage=coverage,
    )


def run_once(
    config: ExperimentConfig,
    alphas: tuple[float, ...] = DEFAULT_COVERAGE_ALPHAS,
) -> SingleRunResult:
    """Experiment II, a single realization.

    Draws `s_true ~ N(s0, Cs)`, generates one noisy data set, and computes
    the posterior and its diagnostics, using `config.seed`. Requires
    `config.prior.ell_true == config.prior.ell_infer` and
    `config.noise.sigma_true == config.noise.sigma_infer` (see
    `_check_correctly_specified`) -- that equality is what "correctly
    specified" means.
    """
    _check_correctly_specified(config)
    _, A, Cs_infer, s0 = _build_geometry_and_inference_prior(config)
    rng = np.random.default_rng(config.seed)
    return _run_correctly_specified_realization(
        A,
        Cs_infer,
        s0,
        config.noise.sigma_true,
        config.noise.sigma_infer,
        rng,
        alphas,
    )


def run_repeated(
    config: ExperimentConfig,
    n_repeats: int,
    seed: int,
    alphas: tuple[float, ...] = DEFAULT_COVERAGE_ALPHAS,
) -> ExperimentResults:
    """Experiment II, repeated over `n_repeats` independent realizations.

    The grid, A, and Cs_infer depend only on `config`, so they are built
    once; each repeat draws its own independent `s_true` and noise
    realization from a single `np.random.default_rng(seed)` advanced
    sequentially across repeats. Identical `(config, n_repeats, seed)`
    therefore always produce numerically identical `ExperimentResults` (see
    `test_reproducibility.py`).
    """
    _check_correctly_specified(config)
    _, A, Cs_infer, s0 = _build_geometry_and_inference_prior(config)
    rng = np.random.default_rng(seed)

    def draw_realization():
        s_true = sample_prior(s0, Cs_infer, rng)
        t_true = forward(A, s_true)
        d = add_noise(t_true, sigma=config.noise.sigma_true, rng=rng)
        s_post, C_post = posterior_isotropic(
            A, d, sigma2=config.noise.sigma_infer**2, Cs=Cs_infer, s0=s0
        )
        return s_true, s_post, C_post

    return _run_repeated_from_draws(n_repeats, A.shape[1], draw_realization, alphas)


def run_correctly_specified(config: ExperimentConfig, n_repeats: int) -> ExperimentResults:
    """Experiment II: Bayesian calibration under the correctly specified model.

    `s_true ~ N(s0, Cs)`; inference uses exactly this same `Cs` and
    `Sigma_d`. Because the generative model and the inference model agree by
    construction, `Q ~ chi2(n_cells)` and `z ~ N(0, 1)` are *guaranteed*
    here -- this experiment validates the implementation, not a deeper
    statistical claim (see `ARCHITECTURE.md`). Uses `config.seed`.
    """
    return run_repeated(config, n_repeats, seed=config.seed)


def run_fixed_truth(
    config: ExperimentConfig,
    s_true: np.ndarray,
    n_repeats: int,
    alphas: tuple[float, ...] = DEFAULT_COVERAGE_ALPHAS,
) -> ExperimentResults:
    """Experiment III: a fixed, deterministic truth with repeated noise draws.

    `s_true` is supplied by the caller as a deterministic field -- e.g.
    `forward.synthetic_truth_gaussian_anomaly` for the smooth truth, or
    `forward.synthetic_truth_checkerboard` for the sharp one (PDF Section 9)
    -- and is *not* regarded as a draw from `Cs_infer`. Only the observation
    noise varies across realizations: `d^(k) = A s_true + eps^(k)`,
    `eps^(k) ~ N(0, sigma_true^2 I)`; inference uses `Cs_infer` (built from
    `config.prior.ell_infer`) and `sigma_infer^2` throughout.

    Unlike `run_repeated`, there is *no general guarantee* that
    `Q ~ chi2(n_cells)` here: whether the posterior stays well calibrated
    depends entirely on how well `s_true` matches what `Cs_infer` expects
    (see `ARCHITECTURE.md`'s "Two different meanings of calibrated" note).
    Run this once with a smooth truth and once with a sharp one to produce
    the coverage comparison that is this project's central result.

    As in `run_construction_validation` (Experiment I), this is not a
    misspecification experiment: it uses `config.noise.sigma_true` to
    generate the data and `config.noise.sigma_infer` for inference, which
    the caller's config should set equal to each other. Uses `config.seed`.
    """
    _, A, Cs_infer, s0 = _build_geometry_and_inference_prior(config)
    rng = np.random.default_rng(config.seed)
    t_true = forward(A, s_true)  # fixed truth -> fixed noiseless travel times

    def draw_realization():
        d = add_noise(t_true, sigma=config.noise.sigma_true, rng=rng)
        s_post, C_post = posterior_isotropic(
            A, d, sigma2=config.noise.sigma_infer**2, Cs=Cs_infer, s0=s0
        )
        return s_true, s_post, C_post

    return _run_repeated_from_draws(n_repeats, A.shape[1], draw_realization, alphas)


@dataclass
class GeometryDiagnostics:
    """Experiment IV: per-geometry results, pooled across every seed and
    repetition (`n_total = len(seeds) * n_repeats_per_seed`).
    """

    name: str
    e_rel: np.ndarray  # (n_total,)
    correlation: np.ndarray  # (n_total,)
    posterior_std: np.ndarray  # (n_cells,) -- deterministic given this geometry
    J_var: float
    J_logdet: float
    rank: int
    r_eff: float


@dataclass
class PairedComparison:
    """Experiment IV: paired-difference results for one geometry relative to
    the uniform baseline (`geometry - uniform`), both pooled across all
    seeds/repetitions and broken out per seed.
    """

    name: str  # e.g. "mild_boundary_minus_uniform"
    delta_e_rel: np.ndarray  # (n_total,)
    delta_correlation: np.ndarray  # (n_total,)
    seed_level_mean_delta_e_rel: np.ndarray  # (n_seeds,)
    seed_level_std_delta_e_rel: np.ndarray  # (n_seeds,)
    seed_level_mean_delta_correlation: np.ndarray  # (n_seeds,)
    seed_level_std_delta_correlation: np.ndarray  # (n_seeds,)
    seed_level_n_improved_e_rel: np.ndarray  # (n_seeds,) int, count(delta_e_rel < 0)
    seed_level_frac_improved_e_rel: np.ndarray  # (n_seeds,)
    seed_level_n_improved_correlation: np.ndarray  # (n_seeds,) int, count(delta_corr > 0)
    seed_level_frac_improved_correlation: np.ndarray  # (n_seeds,)


@dataclass
class ExperimentIVResults:
    """Experiment IV: predetermined acquisition-geometry comparison."""

    seeds: list[int]
    n_repeats_per_seed: int
    geometries: dict[str, GeometryDiagnostics]  # keyed by geometry name
    paired_comparisons: dict[str, PairedComparison]  # keyed by non-uniform geometry name


def run_acquisition_geometry_comparison(config: ExperimentIVConfig) -> ExperimentIVResults:
    """Experiment IV: compare fixed-truth reconstruction and
    posterior-uncertainty metrics across `config.geometries` (predetermined
    acquisition layouts from `geometry.make_boundary_clustered_sources_receivers`),
    holding the grid, prior, noise level, and Experiment III's smooth fixed
    truth constant.

    Paired design: for each of `config.seeds`, a fresh
    `np.random.default_rng(seed)` draws one length-`m` standardized noise
    vector per repetition, and that *same* vector (scaled by `config.sigma`)
    is applied to every geometry -- so per-repetition differences between
    geometries isolate the effect of geometry, not noise-realization
    variance. This is not a general acquisition-geometry sweep or an
    optimizer: `config.geometries` fixes the (small, predetermined) set of
    layouts compared (see `CLAUDE.md`'s "Out of scope" section).

    Exactly one geometry in `config.geometries` must have `gamma == 1.0`
    (the uniform baseline); every other geometry's paired comparison is
    computed against it.
    """
    uniform_candidates = [g.name for g in config.geometries if g.gamma == 1.0]
    if len(uniform_candidates) != 1:
        raise ValueError(
            "run_acquisition_geometry_comparison requires exactly one geometry with "
            f"gamma == 1.0 (the uniform baseline) in config.geometries; found "
            f"{len(uniform_candidates)}: {uniform_candidates}"
        )
    uniform_name = uniform_candidates[0]

    grid = make_grid(config.grid.W, config.grid.n)
    p = grid.n**2
    Cs = squared_exponential_cov(
        grid.cell_centers, tau2=config.tau2, ell=config.ell, jitter_relative=config.jitter_relative
    )
    s0 = config.s_bg * np.ones(p)
    s_true = synthetic_truth_gaussian_anomaly(
        grid, s_bg=config.s_bg, delta_s=np.sqrt(config.tau2),
        x0=config.grid.W / 2, y0=config.grid.W / 2, r=config.grid.W / 6,
    )

    m_expected = config.Ns * config.Nr
    sigma2 = config.sigma**2

    geometry_A: dict[str, np.ndarray] = {}
    geometry_t_true: dict[str, np.ndarray] = {}
    geometry_info: dict[str, dict] = {}
    for geom in config.geometries:
        sources, receivers = make_boundary_clustered_sources_receivers(
            config.grid.W, config.Ns, geom.gamma
        )
        A = build_sensitivity_matrix(sources, receivers, grid)
        assert A.shape[0] == m_expected  # every source connects to every receiver
        geometry_A[geom.name] = A
        geometry_t_true[geom.name] = forward(A, s_true)
        geometry_info[geom.name] = posterior_information_metrics(A, Cs, sigma2)

    n_seeds = len(config.seeds)
    reps = config.n_repeats_per_seed
    n_total = n_seeds * reps

    e_rel = {geom.name: np.empty(n_total) for geom in config.geometries}
    correlation = {geom.name: np.empty(n_total) for geom in config.geometries}
    seed_slices: list[tuple[int, int, int]] = []  # (seed, start, stop)

    idx = 0
    for seed in config.seeds:
        rng = np.random.default_rng(seed)
        start = idx
        for _ in range(reps):
            epsilon = config.sigma * rng.standard_normal(m_expected)
            for geom in config.geometries:
                d = geometry_t_true[geom.name] + epsilon
                s_post, _ = posterior_isotropic(
                    geometry_A[geom.name], d, sigma2=sigma2, Cs=Cs, s0=s0
                )
                e_rel[geom.name][idx] = relative_error(s_post, s_true)
                correlation[geom.name][idx] = pearson_correlation(s_post, s_true)
            idx += 1
        seed_slices.append((seed, start, idx))

    geometries_out = {
        geom.name: GeometryDiagnostics(
            name=geom.name,
            e_rel=e_rel[geom.name],
            correlation=correlation[geom.name],
            posterior_std=np.sqrt(np.diag(geometry_info[geom.name]["C_post"])),
            J_var=geometry_info[geom.name]["J_var"],
            J_logdet=geometry_info[geom.name]["J_logdet"],
            rank=geometry_info[geom.name]["rank"],
            r_eff=geometry_info[geom.name]["r_eff"],
        )
        for geom in config.geometries
    }

    paired_out: dict[str, PairedComparison] = {}
    for geom in config.geometries:
        if geom.name == uniform_name:
            continue
        delta_e = e_rel[geom.name] - e_rel[uniform_name]
        delta_c = correlation[geom.name] - correlation[uniform_name]

        seed_mean_e, seed_std_e, seed_n_e, seed_frac_e = [], [], [], []
        seed_mean_c, seed_std_c, seed_n_c, seed_frac_c = [], [], [], []
        for _, start, stop in seed_slices:
            block_e, block_c = delta_e[start:stop], delta_c[start:stop]
            seed_mean_e.append(block_e.mean())
            seed_std_e.append(block_e.std())
            seed_n_e.append(int(np.sum(block_e < 0)))
            seed_frac_e.append(float(np.mean(block_e < 0)))
            seed_mean_c.append(block_c.mean())
            seed_std_c.append(block_c.std())
            seed_n_c.append(int(np.sum(block_c > 0)))
            seed_frac_c.append(float(np.mean(block_c > 0)))

        paired_out[geom.name] = PairedComparison(
            name=f"{geom.name}_minus_{uniform_name}",
            delta_e_rel=delta_e,
            delta_correlation=delta_c,
            seed_level_mean_delta_e_rel=np.array(seed_mean_e),
            seed_level_std_delta_e_rel=np.array(seed_std_e),
            seed_level_mean_delta_correlation=np.array(seed_mean_c),
            seed_level_std_delta_correlation=np.array(seed_std_c),
            seed_level_n_improved_e_rel=np.array(seed_n_e),
            seed_level_frac_improved_e_rel=np.array(seed_frac_e),
            seed_level_n_improved_correlation=np.array(seed_n_c),
            seed_level_frac_improved_correlation=np.array(seed_frac_c),
        )

    return ExperimentIVResults(
        seeds=list(config.seeds),
        n_repeats_per_seed=reps,
        geometries=geometries_out,
        paired_comparisons=paired_out,
    )
