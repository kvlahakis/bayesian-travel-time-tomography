import numpy as np
import pytest

from tomography.geometry import make_grid
from tomography.prior import sample_prior, squared_exponential_cov


@pytest.mark.parametrize("n", [3, 5, 8])
@pytest.mark.parametrize("ell", [0.1, 0.5, 2.0, 10.0])
def test_squared_exponential_cov_symmetric_and_positive_definite(n, ell):
    grid = make_grid(W=1.0, n=n)
    Cs = squared_exponential_cov(grid.cell_centers, tau2=1.0, ell=ell)

    assert Cs.shape == (n**2, n**2)
    np.testing.assert_allclose(Cs, Cs.T, atol=1e-12)

    eigvals = np.linalg.eigvalsh(Cs)
    assert np.all(eigvals > 0)

    # must not raise
    np.linalg.cholesky(Cs)


def test_jitter_is_relative_to_tau2_not_absolute():
    grid = make_grid(W=1.0, n=4)
    tau2 = 100.0
    jitter_relative = 1e-3
    Cs = squared_exponential_cov(
        grid.cell_centers, tau2=tau2, ell=1.0, jitter_relative=jitter_relative
    )
    Cs_raw = tau2 * np.exp(
        -np.sum(
            (grid.cell_centers[:, None, :] - grid.cell_centers[None, :, :]) ** 2,
            axis=-1,
        )
        / 2.0
    )
    diag_diff = np.diag(Cs) - np.diag(Cs_raw)
    np.testing.assert_allclose(diag_diff, jitter_relative * tau2)


def test_squared_exponential_cov_diagonal_and_far_field_values():
    grid = make_grid(W=100.0, n=5)
    tau2, ell = 2.0, 0.01  # very short correlation length
    Cs = squared_exponential_cov(grid.cell_centers, tau2=tau2, ell=ell)

    # diagonal ~= tau2 + jitter (self-distance is 0)
    np.testing.assert_allclose(np.diag(Cs), tau2 * (1.0 + 1e-6), rtol=1e-9)

    # far-apart cells with a tiny correlation length are ~uncorrelated
    assert abs(Cs[0, -1]) < 1e-10


def test_sample_prior_large_sample_covariance_matches_Cs():
    # Stochastic check with a fixed seed and a generous tolerance -- some
    # Monte Carlo discrepancy is expected even when the code is correct.
    grid = make_grid(W=4.0, n=4)
    tau2, ell = 1.0, 1.5
    Cs = squared_exponential_cov(grid.cell_centers, tau2=tau2, ell=ell)
    s0 = np.zeros(grid.n**2)

    rng = np.random.default_rng(0)
    n_samples = 50_000
    samples = np.array([sample_prior(s0, Cs, rng) for _ in range(n_samples)])

    empirical_mean = samples.mean(axis=0)
    empirical_cov = np.cov(samples, rowvar=False)

    np.testing.assert_allclose(empirical_mean, s0, atol=0.05)
    np.testing.assert_allclose(empirical_cov, Cs, atol=0.05)
