"""Deterministic unit tests for diagnostics.py.

These are exact, hand-computed cases -- no repeated sampling, no
goodness-of-fit checks. Statistical behavior (Q ~ chi2(n_cells), z ~ N(0, 1) under
the correctly-specified model) is tested separately in test_calibration.py,
since a stochastic test can fail from ordinary Monte Carlo variation even
when the code is correct.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from tomography.diagnostics import (
    empirical_coverage,
    fixed_truth_expected_Q,
    mahalanobis,
    marginal_coverage,
    relative_error,
    standardized_errors,
)

# Shared hand-computed 2-cell example.
#
# C_post = [[4, 1], [1, 9]] is symmetric positive definite (det = 35 > 0).
# e = s_true - s_post = [2, -3], so std = sqrt(diag(C_post)) = [2, 3] and
# z = e / std = [1, -1].
#
# C_post^-1 = (1/35) [[9, -1], [-1, 4]], and by direct computation
# e^T C_post^-1 e = 84 / 35 = 2.4.
_S_TRUE = np.array([5.0, 2.0])
_S_POST = np.array([3.0, 5.0])
_C_POST = np.array([[4.0, 1.0], [1.0, 9.0]])
_Q_EXPECTED = 2.4
_Z_EXPECTED = np.array([1.0, -1.0])


def test_relative_error_matches_hand_computation():
    s_post = np.array([1.0, 2.0])
    s_true = np.array([1.0, 0.0])
    # ||[0, 2]|| / ||[1, 0]|| = 2 / 1 = 2
    assert relative_error(s_post, s_true) == 2.0


def test_standardized_errors_uses_correct_diagonal_of_C_post():
    z = standardized_errors(_S_TRUE, _S_POST, _C_POST)
    np.testing.assert_allclose(z, _Z_EXPECTED)


def test_standardized_errors_ignores_off_diagonal_entries():
    # A covariance matrix with the same diagonal but different off-diagonal
    # entries must give the identical z, since z only depends on diag(C_post).
    C_post_other_offdiag = np.array([[4.0, -2.5], [-2.5, 9.0]])
    z = standardized_errors(_S_TRUE, _S_POST, C_post_other_offdiag)
    np.testing.assert_allclose(z, _Z_EXPECTED)


def test_mahalanobis_matches_hand_computed_example():
    Q = mahalanobis(_S_TRUE, _S_POST, _C_POST)
    np.testing.assert_allclose(Q, _Q_EXPECTED)


def test_mahalanobis_equals_z_transpose_R_inverse_z():
    # R is the correlation form of C_post: R = D^-1 C_post D^-1, D = diag(std).
    std = np.sqrt(np.diag(_C_POST))
    D_inv = np.diag(1.0 / std)
    R = D_inv @ _C_POST @ D_inv

    z = standardized_errors(_S_TRUE, _S_POST, _C_POST)
    Q_via_z_R = z @ np.linalg.solve(R, z)

    Q = mahalanobis(_S_TRUE, _S_POST, _C_POST)
    np.testing.assert_allclose(Q, Q_via_z_R)
    np.testing.assert_allclose(Q, _Q_EXPECTED)


def test_marginal_coverage_matches_hand_computed_example():
    # 4 cells, diagonal C_post so std = [1, 2, 3, 4]; s_post = 0.
    # e = [0.5, 3, 9, 100] -> z = [0.5, 1.5, 3, 25].
    # alpha = 0.05 -> z_crit = norm.ppf(0.975) ~= 1.95996.
    # Covered: |z| <= z_crit for cells 0 (0.5) and 1 (1.5) only -> 2 / 4 = 0.5.
    C_post = np.diag([1.0, 4.0, 9.0, 16.0])
    s_post = np.zeros(4)
    s_true = np.array([0.5, 3.0, 9.0, 100.0])

    coverage = marginal_coverage(s_true, s_post, C_post, alpha=0.05)
    assert coverage == 0.5


def test_marginal_coverage_all_covered_when_residuals_are_zero():
    # s_true == s_post everywhere, so every cell is trivially covered for
    # any alpha in (0, 1).
    C_post = np.diag([1.0, 2.0])
    s_post = np.array([3.0, -1.0])
    s_true = np.array([3.0, -1.0])
    coverage = marginal_coverage(s_true, s_post, C_post, alpha=0.05)
    assert coverage == 1.0


def test_empirical_coverage_aggregates_across_realizations_and_cells():
    # 3 realizations, 2 cells. posterior_means = 0, posterior_stds = 1 for
    # every realization/cell, so a cell is covered iff |truth| <= z_crit.
    alpha = 0.05
    z_crit = norm.ppf(1 - alpha / 2)

    truths = np.array(
        [
            [0.0, 0.0],  # both covered
            [0.0, z_crit + 1.0],  # cell 0 covered, cell 1 not
            [z_crit + 1.0, z_crit + 1.0],  # neither covered
        ]
    )
    posterior_means = np.zeros_like(truths)
    posterior_stds = np.ones_like(truths)

    result = empirical_coverage(truths, posterior_means, posterior_stds, alpha)

    np.testing.assert_allclose(result["per_cell"], [2 / 3, 1 / 3])
    np.testing.assert_allclose(result["overall"], 3 / 6)


def test_fixed_truth_expected_Q_bias_term_vanishes_when_truth_equals_prior_mean():
    # b = (I - KA)(s_true - s0); if s_true == s0, b is exactly the zero
    # vector regardless of A/C_post, so bias must be exactly 0 and the
    # theoretical E[Q] must equal the trace term exactly.
    rng = np.random.default_rng(0)
    n_cells, m = 3, 4
    A = rng.normal(size=(m, n_cells))
    C_post = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
    s0 = np.array([1.0, 1.0, 1.0])

    result = fixed_truth_expected_Q(
        A, C_post, sigma_infer2=0.5, sigma_true2=0.8, s_true=s0, s0=s0
    )

    assert result["bias"] == 0.0
    assert result["theory"] == result["trace"]
    assert result["trace"] > 0  # a genuine noise contribution


def test_fixed_truth_expected_Q_matches_naive_matrix_inverse_computation():
    # Cross-check the Cholesky-solve-based implementation against the same
    # formula computed via explicit matrix inverses -- acceptable here only
    # because n_cells is tiny and this is a one-off cross-check, not
    # production code (which never forms an explicit inverse; see
    # inversion.py).
    rng = np.random.default_rng(1)
    n_cells, m = 3, 5
    A = rng.normal(size=(m, n_cells))
    C_post = np.array([[2.0, 0.4, 0.1], [0.4, 1.8, 0.3], [0.1, 0.3, 1.2]])
    s0 = np.array([1.0, 1.0, 1.0])
    s_true = np.array([1.3, 0.8, 1.1])
    sigma_infer2, sigma_true2 = 0.6, 0.9

    result = fixed_truth_expected_Q(A, C_post, sigma_infer2, sigma_true2, s_true, s0)

    K = C_post @ A.T / sigma_infer2
    b = (np.eye(n_cells) - K @ A) @ (s_true - s0)
    V_post = sigma_true2 * (K @ K.T)
    C_post_inv = np.linalg.inv(C_post)
    trace_expected = np.trace(C_post_inv @ V_post)
    bias_expected = b @ C_post_inv @ b

    np.testing.assert_allclose(result["trace"], trace_expected)
    np.testing.assert_allclose(result["bias"], bias_expected)
    np.testing.assert_allclose(result["theory"], trace_expected + bias_expected)
