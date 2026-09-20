import numpy as np
import pytest

from tomography.forward import add_noise, forward, synthetic_truth_gaussian_anomaly
from tomography.geometry import Grid, build_sensitivity_matrix, make_grid, make_sources_receivers
from tomography.inversion import posterior_general, posterior_isotropic
from tomography.prior import squared_exponential_cov


def _build_common(
    n: int = 6, W: float = 6.0, tau2: float = 1.0, ell: float = 1.5, s_bg: float = 1.0
) -> tuple[Grid, np.ndarray, np.ndarray]:
    grid = make_grid(W, n)
    Cs = squared_exponential_cov(grid.cell_centers, tau2=tau2, ell=ell)
    s0 = s_bg * np.ones(grid.n**2)
    return grid, Cs, s0


def _make_data(
    grid: Grid, Ns: int, Nr: int, sigma2: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    sources, receivers = make_sources_receivers(grid.W, Ns, Nr)
    A = build_sensitivity_matrix(sources, receivers, grid)
    s_true = synthetic_truth_gaussian_anomaly(
        grid, s_bg=1.0, delta_s=0.3, x0=grid.W / 2, y0=grid.W / 2, r=grid.W / 6
    )
    t = forward(A, s_true)
    rng = np.random.default_rng(seed)
    d = add_noise(t, sigma=np.sqrt(sigma2), rng=rng)
    return A, d


def test_posterior_isotropic_C_post_symmetric_positive_definite():
    grid, Cs, s0 = _build_common()
    A, d = _make_data(grid, Ns=4, Nr=4, sigma2=0.05, seed=0)

    _, C_post = posterior_isotropic(A, d, sigma2=0.05, Cs=Cs, s0=s0)

    np.testing.assert_allclose(C_post, C_post.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(C_post)
    assert np.all(eigvals > 0)


def test_posterior_reduces_to_prior_as_sigma2_to_infinity():
    grid, Cs, s0 = _build_common()
    A, d = _make_data(grid, Ns=4, Nr=4, sigma2=0.05, seed=0)

    s_post, C_post = posterior_isotropic(A, d, sigma2=1e12, Cs=Cs, s0=s0)

    np.testing.assert_allclose(s_post, s0, atol=1e-4)
    np.testing.assert_allclose(C_post, Cs, atol=1e-4)


def test_posterior_uncertainty_shrinks_as_noise_decreases():
    grid, Cs, s0 = _build_common()
    A, d = _make_data(grid, Ns=4, Nr=4, sigma2=1.0, seed=0)

    _, C_high_noise = posterior_isotropic(A, d, sigma2=10.0, Cs=Cs, s0=s0)
    _, C_low_noise = posterior_isotropic(A, d, sigma2=1e-3, Cs=Cs, s0=s0)

    assert np.mean(np.diag(C_low_noise)) < np.mean(np.diag(C_high_noise))


def test_posterior_uncertainty_shrinks_as_ray_density_increases():
    grid, Cs, s0 = _build_common()
    sigma2 = 0.1

    A_sparse, d_sparse = _make_data(grid, Ns=2, Nr=2, sigma2=sigma2, seed=0)
    A_dense, d_dense = _make_data(grid, Ns=10, Nr=10, sigma2=sigma2, seed=0)

    _, C_sparse = posterior_isotropic(A_sparse, d_sparse, sigma2, Cs, s0)
    _, C_dense = posterior_isotropic(A_dense, d_dense, sigma2, Cs, s0)

    assert np.mean(np.diag(C_dense)) < np.mean(np.diag(C_sparse))


def test_posterior_general_and_isotropic_agree_for_isotropic_noise():
    grid, Cs, s0 = _build_common()
    sigma2 = 0.05
    A, d = _make_data(grid, Ns=4, Nr=4, sigma2=sigma2, seed=0)
    Sigma_d = sigma2 * np.eye(A.shape[0])

    s_post_general, C_post_general = posterior_general(A, d, Sigma_d, Cs, s0)
    s_post_iso, C_post_iso = posterior_isotropic(A, d, sigma2, Cs, s0)

    np.testing.assert_allclose(s_post_general, s_post_iso, atol=1e-8)
    np.testing.assert_allclose(C_post_general, C_post_iso, atol=1e-8)
