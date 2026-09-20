import numpy as np

from tomography.config import (
    AcquisitionConfig,
    ExperimentConfig,
    GridConfig,
    NoiseConfig,
    PriorConfig,
)
from tomography.experiments import run_construction_validation


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
