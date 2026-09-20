"""Stochastic integration tests for Experiment II (correctly specified
Bayesian calibration).

These tests are statistical, not exact: under the correctly specified
generative model (`s_true ~ N(s0, Cs)`, `d | s_true ~ N(A s_true, Sigma_d)`,
inference using the same `Cs`/`Sigma_d`), theory guarantees `Q ~ chi2(n_cells)`
and `z ~ N(0, 1)` (see `ARCHITECTURE.md`). A finite sample of realizations can
fail these checks from ordinary Monte Carlo variation even when the code is
completely correct, so:

  - the RNG seed is fixed, so a failure is reproducible rather than flaky;
  - `N_REPEATS` is large enough, and tolerances generous enough, that the
    true failure rate of these tests (given correct code) is negligible;
  - this file is kept separate from `test_diagnostics.py`, which contains
    only exact, deterministic checks.

This module runs at the *real* grid scale (`n=20`, matching
`configs/calibration_correct.yaml`), not a small toy grid: a pooled
Kolmogorov-Smirnov test of `z` against `N(0, 1)` across all `n_cells` cells
was previously found to fail reliably at this scale even though the model is
correctly specified (see `ARCHITECTURE.md`'s calibration-diagnostics note).
The reason is that per-cell `z` values are spatially correlated within a
single realization -- the prior induces spatial dependence, which the
likelihood/posterior update then modifies -- so pooling them across cells
violates the i.i.d. assumption `kstest` relies on: the pooled test's tiny
p-value reflects that violated assumption, not a broken posterior. That
pooled-across-cells KS test has been removed *entirely* (not replaced with a
weaker version, and the underlying model/inference code was not touched to
make it pass). In its place:

  - the global `Q`-vs-`chi2(n_cells)` KS test is kept (each realization
    contributes one *independent* `Q`, so pooling across realizations is
    valid);
  - simple scalar mean(z)/std(z) checks are kept, but only as descriptive
    marginal summaries -- no i.i.d. interpretation of the pooled values is
    assumed;
  - marginal coverage checks are kept (per alpha, aggregated by
    `diagnostics.empirical_coverage`);
  - five *individual* cells each get their own per-cell KS test against
    `N(0, 1)` (each such test pools only across the 1000 independent
    realizations, not across cells, so it doesn't have the pooling problem),
    with a Bonferroni correction across the five.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from tomography.config import (
    AcquisitionConfig,
    ExperimentConfig,
    GridConfig,
    NoiseConfig,
    PriorConfig,
)
from tomography.experiments import run_correctly_specified

# Matches configs/calibration_correct.yaml exactly: the real n=20 grid, not a
# toy-scale grid, so calibration is validated at the scale the project
# actually reports (see the module docstring for why toy-scale results here
# do not necessarily generalize).
GRID_N = 20
N_REPEATS = 1000
SEED = 12345


def _real_scale_config() -> ExperimentConfig:
    return ExperimentConfig(
        grid=GridConfig(W=10.0, n=GRID_N),
        acquisition=AcquisitionConfig(Ns=12, Nr=12),
        noise=NoiseConfig(sigma_true=0.02, sigma_infer=0.02),
        prior=PriorConfig(tau2=0.04, ell_true=2.0, ell_infer=2.0, s_bg=1.0),
        n_repeats=N_REPEATS,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def calibration_results():
    """Run Experiment II once and share the result across every test in this
    module -- n=20/n_repeats=1000 is cheap (a few seconds) but there is no
    reason to redo it once per test function.
    """
    return run_correctly_specified(_real_scale_config(), n_repeats=N_REPEATS)


def _representative_cells(n: int, posterior_std: np.ndarray) -> dict[str, int]:
    """Five deterministic representative cells for per-cell calibration checks.

    Geometric cells are chosen from grid indices alone, independent of any
    realization -- per `geometry.py`'s convention (cell index `k = i * n +
    j`, `i` indexing `y`, `j` indexing `x`):

      - "center": `i = j = n // 2`;
      - "near_left_boundary": `i = n // 2`, `j = 0` (leftmost column, at the
        vertical midpoint so it isn't also a corner);
      - "near_right_boundary": `i = n // 2`, `j = n - 1` (rightmost column,
        same convention).

    The max/min posterior-variance cells are selected from `posterior_std`
    (the diagonal of `C_post`). This is valid only because that diagonal is
    identical across every realization within a single Experiment II run
    (`C_post` depends only on `A`, `Cs`, and `Sigma_d`, never on the data or
    the truth) -- callers must verify that precondition themselves before
    relying on it (see
    `test_posterior_variance_diagonal_is_identical_across_realizations`).
    `np.argmax` / `np.argmin` return the first (lowest-index) occurrence on
    ties, which is the deterministic tie-break used here.
    """
    i_mid = n // 2
    return {
        "center": i_mid * n + i_mid,
        "near_left_boundary": i_mid * n + 0,
        "near_right_boundary": i_mid * n + (n - 1),
        "max_posterior_variance": int(np.argmax(posterior_std)),
        "min_posterior_variance": int(np.argmin(posterior_std)),
    }


def test_mahalanobis_matches_chi2_under_correctly_specified_model(calibration_results):
    n_cells = GRID_N**2
    Q = calibration_results.mahalanobis

    # chi2(n_cells) has mean n_cells and variance 2 * n_cells. With
    # N_REPEATS = 1000 draws, the Monte Carlo standard error of the sample
    # mean is sqrt(2 * n_cells / N_REPEATS) ~= sqrt(2 * 400 / 1000) ~= 0.89,
    # so a 25% relative tolerance on the mean is very generous.
    assert np.mean(Q) == pytest.approx(n_cells, rel=0.25)
    assert np.var(Q) == pytest.approx(2 * n_cells, rel=0.5)

    # Distribution-shape check: two-sided KS test against the theoretical
    # chi2(n_cells) CDF. Each realization contributes one independent Q, so
    # (unlike pooling z across cells) this pooling across realizations is
    # valid. alpha = 0.01 keeps the false-failure rate low for a fixed,
    # pre-selected seed.
    ks_result = kstest(Q, "chi2", args=(n_cells,))
    assert ks_result.pvalue > 0.01, (
        f"Q does not resemble chi2({n_cells}) (KS p-value = {ks_result.pvalue:.4g}); "
        "under the correctly specified model this points to an implementation bug, "
        "not a statistical fluke -- see ARCHITECTURE.md."
    )


def test_standardized_errors_mean_and_std_match_standard_normal(calibration_results):
    z = calibration_results.z_scores.ravel()

    assert abs(np.mean(z)) < 0.1
    assert np.std(z) == pytest.approx(1.0, rel=0.2)


def test_posterior_variance_diagonal_is_identical_across_realizations(calibration_results):
    """The diagonal of `C_post` (per-cell posterior variance, stored as
    `posterior_stds`) must be *exactly* identical across every realization,
    not merely close.

    This checks only `diag(C_post)`, not the full `C_post` matrix -- the full
    matrix is never stacked across realizations (`ExperimentResults` stores
    `posterior_stds`, not `C_post`, precisely to avoid that (n_repeats,
    n_cells, n_cells) storage cost), and the diagonal is all that
    `_representative_cells` needs to select the max/min posterior-variance
    cells. The same argument that justifies checking the diagonal here
    (`C_post` depends only on `A`, `Cs`, and `Sigma_d` -- all fixed within one
    Experiment II run -- never on the data or the truth) applies to the full
    matrix too, but that stronger claim is not verified by this test.

    This is verified directly (not assumed) as a precondition for
    `test_representative_cell_z_scores_pass_bonferroni_corrected_ks`: any
    discrepancy here would indicate a bug (e.g. `C_post` accidentally
    depending on the data or the truth), and must be reported as such rather
    than averaged over.
    """
    posterior_stds = calibration_results.posterior_stds
    first = posterior_stds[0]
    for k in range(1, posterior_stds.shape[0]):
        np.testing.assert_array_equal(
            posterior_stds[k],
            first,
            err_msg=(
                f"posterior_stds differ at realization {k}, but C_post depends only "
                "on A, Cs, and Sigma_d (all fixed within a run) -- this is a bug, "
                "not something to average over."
            ),
        )


def test_representative_cell_z_scores_pass_bonferroni_corrected_ks(calibration_results):
    """Per-cell KS test against N(0, 1) for five deterministic representative
    cells (domain center, near-left-boundary, near-right-boundary, and the
    max/min posterior-variance cells -- see `_representative_cells`).

    Each test pools only across the N_REPEATS independent realizations for
    one fixed cell, never across cells, so (unlike the removed pooled test)
    it does not run into the spatial-correlation problem.

    The five per-cell test statistics need not be independent of each other
    -- spatial errors are correlated within a realization, since the prior
    induces spatial dependence which the likelihood/posterior update then
    modifies (see ARCHITECTURE.md) -- but the Bonferroni
    correction's family-wise error guarantee holds regardless of dependence
    between the tests being corrected for, so it is still valid to apply
    here even though the five cells' z-scores are correlated with each
    other.
    """
    cells = _representative_cells(GRID_N, calibration_results.posterior_stds[0])

    family_wise_alpha = 0.01
    bonferroni_threshold = family_wise_alpha / len(cells)
    assert bonferroni_threshold == pytest.approx(0.002)

    for name, k in cells.items():
        z_k = calibration_results.z_scores[:, k]
        p_value = kstest(z_k, "norm").pvalue
        assert p_value > bonferroni_threshold, (
            f"cell '{name}' (index {k}): KS p-value {p_value:.4g} does not clear the "
            f"Bonferroni-corrected threshold {bonferroni_threshold} (family-wise "
            f"alpha={family_wise_alpha} / {len(cells)} tests)"
        )


def test_empirical_coverage_close_to_nominal_under_correctly_specified_model(
    calibration_results,
):
    # Generous absolute tolerance: this coverage estimate is averaged over
    # N_REPEATS * n_cells ~ 400,000 indicator values, but those indicators
    # are spatially correlated across cells within a realization, so their
    # count is not a valid i.i.d. sample size and no binomial standard error
    # is claimed here. 0.05 is a deliberately generous margin chosen without
    # that justification.
    for alpha, empirical in calibration_results.coverage.items():
        nominal = 1.0 - alpha
        assert abs(empirical - nominal) < 0.05, (
            f"alpha={alpha}: empirical coverage {empirical:.3f} vs nominal "
            f"{nominal:.3f}"
        )


def test_across_cell_calibration_summary(calibration_results):
    """Across-cell calibration summary: median |mean(z_j)| and median
    std(z_j) over all n_cells cells, and the empirical-vs-nominal coverage
    for all six alpha levels (not just their min/max). These are reported
    (not just asserted) as a compact per-cell-resolution complement to the
    pooled Q and scalar mean(z)/std(z) checks above.
    """
    z = calibration_results.z_scores  # (n_repeats, n_cells)
    per_cell_mean_abs = np.median(np.abs(np.mean(z, axis=0)))
    per_cell_std_median = np.median(np.std(z, axis=0))

    coverage = calibration_results.coverage  # alpha -> empirical coverage
    alphas = sorted(coverage)
    empirical = [coverage[a] for a in alphas]

    # Derive the min/max coverage *and which alpha they belong to* directly
    # from the six (alpha, empirical) pairs just printed -- not assumed from
    # any earlier summary.
    min_idx = int(np.argmin(empirical))
    max_idx = int(np.argmax(empirical))

    lines = [
        "\nAcross-cell calibration summary "
        f"(n_cells={z.shape[1]}, n_repeats={z.shape[0]}):",
        f"  median |mean(z_j)| across cells: {per_cell_mean_abs:.4g}",
        f"  median std(z_j) across cells:    {per_cell_std_median:.4g}",
        "  nominal vs. empirical coverage (all 6 alpha levels):",
    ]
    for a, e in zip(alphas, empirical):
        lines.append(f"    alpha={a:<5} nominal={1 - a:.2f}  empirical={e:.4g}")
    lines.append(
        f"  minimum empirical coverage: {empirical[min_idx]:.4g} "
        f"(alpha={alphas[min_idx]}, nominal={1 - alphas[min_idx]:.2f})"
    )
    lines.append(
        f"  maximum empirical coverage: {empirical[max_idx]:.4g} "
        f"(alpha={alphas[max_idx]}, nominal={1 - alphas[max_idx]:.2f})"
    )
    print("\n".join(lines))

    min_coverage, max_coverage = empirical[min_idx], empirical[max_idx]
    assert per_cell_mean_abs < 0.2
    assert 0.75 <= per_cell_std_median <= 1.25
    assert 0.0 < min_coverage <= max_coverage < 1.0
