"""End-to-end smoke test: exercises the full pipeline (geometry -> forward
matrix -> synthetic data -> prior -> inversion -> diagnostics/output) on the
tiny `configs/smoke.yaml` configuration.

This is deliberately not a calibration or scientific-correctness check (that
is `test_calibration.py`'s job, at the project's real `n=20` scale) -- it
only confirms the pipeline runs end-to-end and produces validly shaped,
finite, well-formed output, so it fails fast and clearly if the core stack
is broken (e.g. by an environment change or a refactor).
"""

from __future__ import annotations

import numpy as np

from tomography.config import load_config
from tomography.experiments import (
    run_construction_validation,
    run_correctly_specified,
    run_fixed_truth,
)
from tomography.forward import synthetic_truth_checkerboard, synthetic_truth_gaussian_anomaly
from tomography.geometry import make_grid

SMOKE_CONFIG_PATH = "configs/smoke.yaml"


def test_smoke_config_is_tiny_and_distinct_from_frozen_configs():
    config = load_config(SMOKE_CONFIG_PATH)
    n_cells = config.grid.n**2
    # Tiny by construction -- this is a smoke test, not a calibration run.
    assert n_cells <= 64
    assert config.n_repeats <= 10


def test_smoke_construction_validation_end_to_end():
    config = load_config(SMOKE_CONFIG_PATH)
    result = run_construction_validation(config)

    n_cells = config.grid.n**2
    m_rays = config.acquisition.Ns * config.acquisition.Nr

    assert result.A.shape == (m_rays, n_cells)
    assert result.s_true.shape == (n_cells,)
    assert result.s_post.shape == (n_cells,)
    assert result.C_post.shape == (n_cells, n_cells)
    assert np.all(np.isfinite(result.s_post))
    assert np.all(np.isfinite(result.C_post))
    assert np.all(np.diag(result.C_post) > 0)


def test_smoke_correctly_specified_end_to_end():
    config = load_config(SMOKE_CONFIG_PATH)
    results = run_correctly_specified(config, n_repeats=config.n_repeats)

    n_cells = config.grid.n**2
    assert results.truths.shape == (config.n_repeats, n_cells)
    assert results.mahalanobis.shape == (config.n_repeats,)
    assert np.all(np.isfinite(results.mahalanobis))
    assert np.all(results.mahalanobis >= 0)  # Q is a quadratic form
    assert set(results.coverage.keys())  # non-empty coverage dict


def test_smoke_fixed_truth_smooth_and_sharp_end_to_end():
    config = load_config(SMOKE_CONFIG_PATH)
    grid = make_grid(config.grid.W, config.grid.n)

    smooth_truth = synthetic_truth_gaussian_anomaly(
        grid,
        s_bg=config.prior.s_bg,
        delta_s=np.sqrt(config.prior.tau2),
        x0=config.grid.W / 2,
        y0=config.grid.W / 2,
        r=config.grid.W / 6,
    )
    sharp_truth = synthetic_truth_checkerboard(
        grid,
        s_bg=config.prior.s_bg,
        delta_s=np.sqrt(config.prior.tau2),
        block_size=config.fixed_truth.checkerboard_block_size,
    )

    for s_true in (smooth_truth, sharp_truth):
        results = run_fixed_truth(config, s_true, n_repeats=config.n_repeats)
        assert np.all(np.isfinite(results.mahalanobis))
        assert np.all(results.mahalanobis >= 0)
        # s_true is fixed -> every stored realization must equal it exactly.
        for k in range(config.n_repeats):
            np.testing.assert_array_equal(results.truths[k], s_true)
