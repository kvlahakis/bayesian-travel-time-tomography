import numpy as np

from tomography.config import (
    AcquisitionConfig,
    ExperimentConfig,
    GridConfig,
    NoiseConfig,
    PriorConfig,
)
from tomography.experiments import run_construction_validation, run_fixed_truth
from tomography.forward import synthetic_truth_checkerboard, synthetic_truth_gaussian_anomaly
from tomography.geometry import make_grid


def _baseline_config() -> ExperimentConfig:
    return ExperimentConfig(
        grid=GridConfig(W=10.0, n=20),
        acquisition=AcquisitionConfig(Ns=12, Nr=12),
        noise=NoiseConfig(sigma_true=0.02, sigma_infer=0.02),
        prior=PriorConfig(tau2=0.04, ell_true=2.0, ell_infer=2.0, s_bg=1.0),
        n_repeats=1,
        seed=0,
    )


def test_run_construction_validation_shapes_and_validity():
    config = _baseline_config()
    result = run_construction_validation(config)

    n_cells = config.grid.n**2
    m_rays = config.acquisition.Ns * config.acquisition.Nr

    assert result.A.shape == (m_rays, n_cells)
    assert result.s_true.shape == (n_cells,)
    assert result.d.shape == (m_rays,)
    assert result.s_post.shape == (n_cells,)
    assert result.C_post.shape == (n_cells, n_cells)

    # C_post must be a valid covariance matrix
    np.testing.assert_allclose(result.C_post, result.C_post.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(result.C_post) > 0)


def test_run_construction_validation_reconstructs_the_anomaly_reasonably_well():
    # Sanity check only (relative_error itself lives in diagnostics.py, not
    # yet built) -- confirms the pipeline is wired correctly, not a
    # calibration claim.
    config = _baseline_config()
    result = run_construction_validation(config)

    rel_error = np.linalg.norm(result.s_post - result.s_true) / np.linalg.norm(
        result.s_true
    )
    assert rel_error < 0.3


def test_run_construction_validation_is_reproducible():
    config = _baseline_config()
    result_a = run_construction_validation(config)
    result_b = run_construction_validation(config)

    np.testing.assert_array_equal(result_a.d, result_b.d)
    np.testing.assert_array_equal(result_a.s_post, result_b.s_post)


# block_size=4 (not 2): with block_size=2 on a 20x20 grid, *every* cell sits
# on a block edge (its 4-neighbors always cross into a different block), so
# there is no "interior" region to contrast against block boundaries at all.
# block_size=4 gives 100/400 genuine interior cells (25%) alongside boundary
# cells, which is what a boundary-vs-interior coverage comparison needs.
SHARP_TRUTH_BLOCK_SIZE = 4


def _smooth_truth(config: ExperimentConfig) -> np.ndarray:
    # Matches run_construction_validation's (Experiment I's) own default
    # exactly -- delta_s is *derived* from the prior's marginal variance,
    # not a separately chosen literal, so the smooth fixed truth here really
    # is "the Experiment I truth", not a coincidentally similar new field.
    grid = make_grid(config.grid.W, config.grid.n)
    return synthetic_truth_gaussian_anomaly(
        grid,
        s_bg=config.prior.s_bg,
        delta_s=np.sqrt(config.prior.tau2),
        x0=config.grid.W / 2,
        y0=config.grid.W / 2,
        r=config.grid.W / 6,
    )


def _sharp_truth(config: ExperimentConfig) -> np.ndarray:
    grid = make_grid(config.grid.W, config.grid.n)
    return synthetic_truth_checkerboard(
        grid,
        s_bg=config.prior.s_bg,
        delta_s=np.sqrt(config.prior.tau2),
        block_size=SHARP_TRUTH_BLOCK_SIZE,
    )


def test_run_fixed_truth_shapes_and_truth_constant_across_repeats():
    config = _baseline_config()
    s_true = _smooth_truth(config)
    n_repeats = 5

    results = run_fixed_truth(config, s_true, n_repeats=n_repeats)

    n_cells = config.grid.n**2
    assert results.truths.shape == (n_repeats, n_cells)
    assert results.posterior_means.shape == (n_repeats, n_cells)
    assert results.posterior_stds.shape == (n_repeats, n_cells)
    assert results.relative_errors.shape == (n_repeats,)
    assert results.z_scores.shape == (n_repeats, n_cells)
    assert results.mahalanobis.shape == (n_repeats,)

    # s_true is fixed across realizations -- every row of `truths` must
    # equal it exactly, not just approximately.
    for k in range(n_repeats):
        np.testing.assert_array_equal(results.truths[k], s_true)


def test_run_fixed_truth_is_reproducible():
    config = _baseline_config()
    s_true = _smooth_truth(config)

    result_a = run_fixed_truth(config, s_true, n_repeats=5)
    result_b = run_fixed_truth(config, s_true, n_repeats=5)

    np.testing.assert_array_equal(result_a.posterior_means, result_b.posterior_means)
    np.testing.assert_array_equal(result_a.mahalanobis, result_b.mahalanobis)


def test_run_fixed_truth_smooth_vs_sharp_central_calibration_comparison():
    """The project's central result (PDF Section 9): compare a smooth fixed
    truth (the Experiment I Gaussian-anomaly field) against a sharp
    checkerboard truth, using the identical acquisition/prior/noise config
    and the identical posterior-computation code path (`run_fixed_truth`) --
    the only thing that differs is which deterministic field is held fixed.

    This was checked empirically (across several seeds and both n_repeats
    settings) before being turned into an assertion, and the following
    dichotomy holds robustly, so it is asserted directly:

      - the smooth truth's empirical coverage is at or above nominal at
        *every* alpha level tested (i.e. it never under-covers -- if
        anything it is conservative here, since a single localized Gaussian
        bump is a much lower-energy field than a typical draw from the
        squared-exponential prior, so this is not a claim that its coverage
        lands *at* nominal -- see ARCHITECTURE.md: there is no guarantee
        either way for a fixed truth);
      - the sharp/checkerboard truth's empirical coverage is *below*
        nominal at every alpha level tested (systematic under-coverage);
      - the sharp truth's mean Mahalanobis Q is many orders of magnitude
        larger than the smooth truth's (a prior this smooth cannot
        represent hard block edges), while the smooth truth's mean Q stays
        well below n_cells.
    """
    config = _baseline_config()
    n_repeats = 200
    s_smooth = _smooth_truth(config)
    s_sharp = _sharp_truth(config)

    smooth = run_fixed_truth(config, s_smooth, n_repeats=n_repeats)
    sharp = run_fixed_truth(config, s_sharp, n_repeats=n_repeats)

    for alpha in smooth.coverage:
        nominal = 1.0 - alpha
        assert smooth.coverage[alpha] >= nominal, (
            f"smooth truth: alpha={alpha} empirical coverage "
            f"{smooth.coverage[alpha]:.4f} fell below nominal {nominal:.2f}"
        )
        assert sharp.coverage[alpha] < nominal, (
            f"sharp truth: alpha={alpha} empirical coverage "
            f"{sharp.coverage[alpha]:.4f} did not fall below nominal {nominal:.2f}"
        )

    n_cells = config.grid.n**2
    assert np.mean(smooth.mahalanobis) < n_cells
    assert np.mean(sharp.mahalanobis) > 100 * n_cells
