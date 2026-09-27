import numpy as np
import pytest

from tomography.forward import (
    add_noise,
    forward,
    synthetic_truth_bimodal,
    synthetic_truth_checkerboard,
    synthetic_truth_gaussian_anomaly,
)
from tomography.geometry import build_sensitivity_matrix, make_grid, make_sources_receivers


def test_forward_matches_hand_computed_toy_case():
    # 2x2 grid of unit cells (n=2, h=1), k = i*n+j so cell order is
    # (i=0,j=0)=0, (i=0,j=1)=1, (i=1,j=0)=2, (i=1,j=1)=3.
    # A single horizontal ray at y=0.5 lies entirely in row i=0, crossing
    # cells j=0 and j=1 each with intersection length 1.
    A = np.array([[1.0, 1.0, 0.0, 0.0]])
    s = np.array([2.0, 3.0, 100.0, -100.0])
    t = forward(A, s)
    np.testing.assert_allclose(t, [5.0])


def test_forward_constant_slowness_matches_ray_lengths():
    W, n = 4.0, 5
    grid = make_grid(W, n)
    sources, receivers = make_sources_receivers(W, Ns=4, Nr=4)
    A = build_sensitivity_matrix(sources, receivers, grid)

    c = 2.5
    s = c * np.ones(grid.n**2)
    t = forward(A, s)

    ray_lengths = np.array(
        [np.linalg.norm(r - so) for so in sources for r in receivers]
    )
    np.testing.assert_allclose(t, c * ray_lengths)


def test_add_noise_zero_sigma_returns_t_unchanged():
    rng = np.random.default_rng(0)
    t = np.array([1.0, 2.0, 3.0])
    d = add_noise(t, sigma=0.0, rng=rng)
    np.testing.assert_allclose(d, t)


def test_add_noise_reproducible_with_seeded_rng():
    t = np.array([1.0, 2.0, 3.0])
    d1 = add_noise(t, sigma=0.5, rng=np.random.default_rng(42))
    d2 = add_noise(t, sigma=0.5, rng=np.random.default_rng(42))
    np.testing.assert_allclose(d1, d2)


def test_add_noise_statistics_match_specified_sigma():
    rng = np.random.default_rng(0)
    t = np.zeros(50_000)
    d = add_noise(t, sigma=2.0, rng=rng)
    assert abs(d.mean()) < 0.05
    assert abs(d.std() - 2.0) < 0.05


def test_add_noise_rejects_negative_sigma():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        add_noise(np.array([1.0]), sigma=-1.0, rng=rng)


def test_synthetic_truth_gaussian_anomaly():
    grid = make_grid(W=10.0, n=10)
    s_true = synthetic_truth_gaussian_anomaly(
        grid, s_bg=1.0, delta_s=0.5, x0=5.0, y0=5.0, r=1.0
    )
    assert s_true.shape == (grid.n**2,)

    # cell nearest the anomaly center attains the maximum value
    center_idx = int(np.argmin(np.sum((grid.cell_centers - [5.0, 5.0]) ** 2, axis=1)))
    assert s_true[center_idx] == s_true.max()

    x, y = grid.cell_centers[center_idx]
    expected_center = 1.0 + 0.5 * np.exp(-((x - 5.0) ** 2 + (y - 5.0) ** 2) / 2.0)
    assert s_true[center_idx] == pytest.approx(expected_center)

    # far corner cell (center (0.5, 0.5)) is essentially background
    corner_idx = 0
    assert s_true[corner_idx] == pytest.approx(1.0, abs=1e-6)


def test_synthetic_truth_checkerboard_alternates_sharply():
    n = 8
    grid = make_grid(W=8.0, n=n)
    s_true = synthetic_truth_checkerboard(grid, s_bg=1.0, delta_s=0.4, block_size=2)

    assert s_true.shape == (n**2,)
    np.testing.assert_allclose(sorted(np.unique(s_true)), [0.8, 1.2])

    # adjacent 2x2 blocks along the same row must take opposite values
    idx_block0 = 0 * n + 0  # (i=0, j=0) -> block (0, 0)
    idx_block1 = 0 * n + 2  # (i=0, j=2) -> block (0, 1)
    assert s_true[idx_block0] != s_true[idx_block1]

    # within a block, values are constant
    idx_same_block = 1 * n + 1  # (i=1, j=1) -> still block (0, 0)
    assert s_true[idx_same_block] == s_true[idx_block0]


def _count_local_maxima(field: np.ndarray, n: int) -> list[tuple[int, int, float]]:
    """8-connected local maxima of a flattened row-major (k = i*n+j) field."""
    grid_field = field.reshape(n, n)
    maxima = []
    for i in range(n):
        for j in range(n):
            v = grid_field[i, j]
            is_max = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and grid_field[ni, nj] > v:
                        is_max = False
                        break
                if not is_max:
                    break
            if is_max:
                maxima.append((i, j, v))
    return maxima


def _frozen_bimodal_truth():
    # The frozen parameterization used as Experiment I's truth / Experiment
    # III's smooth fixed truth / Experiment IV's reconstruction target (see
    # CLAUDE.md's forward.py module contract).
    W, n, tau2 = 10.0, 20, 0.04
    grid = make_grid(W, n)
    s_true = synthetic_truth_bimodal(
        grid, s_bg=1.0, delta_s=np.sqrt(tau2),
        centers=[(0.25 * W, 0.475 * W), (0.75 * W, 0.475 * W)],
        r=W / 6,
    )
    return grid, s_true, n


def test_synthetic_truth_bimodal_has_exactly_two_local_maxima():
    # Regression guard against a merged single peak (centers too close
    # relative to r) or a grid-alignment tie (centers on a cell boundary) --
    # both were real failure modes found and fixed during development.
    _, s_true, n = _frozen_bimodal_truth()
    maxima = _count_local_maxima(s_true, n)
    assert len(maxima) == 2


def test_synthetic_truth_bimodal_has_a_real_dip_not_a_plateau():
    grid, s_true, n = _frozen_bimodal_truth()
    maxima = _count_local_maxima(s_true, n)
    peak_value = max(v for (_, _, v) in maxima)
    s_bg = 1.0

    center_idx = int(
        np.argmin(np.sum((grid.cell_centers - [grid.W / 2, grid.W / 2]) ** 2, axis=1))
    )
    center_value = s_true[center_idx]

    dip_ratio = (center_value - s_bg) / (peak_value - s_bg)
    # A real valley was verified at ~0.65 during development; this only
    # checks it stays comfortably below 1.0 (a merged blob/plateau), not the
    # exact historical ratio, so it isn't brittle to minor numerical changes.
    assert dip_ratio < 0.9


def test_synthetic_truth_bimodal_sums_background_once():
    # Dedicated, well-separated example (not the frozen parameterization
    # above) to isolate the summation formula itself: background must be
    # added once, not once per bump.
    grid = make_grid(W=20.0, n=20)
    # centers placed exactly on cell centers (h=1, centers at 0.5, 1.5, ...)
    # so the peak value can be checked to tight tolerance, not just "close".
    centers = [(4.5, 9.5), (15.5, 9.5)]
    s_bg, delta_s = 1.0, 0.5
    s_true = synthetic_truth_bimodal(grid, s_bg=s_bg, delta_s=delta_s, centers=centers, r=1.0)

    # far from both centers -> background only. A double-counted background
    # (summing two independently-backgrounded fields) would read ~2.0 here.
    corner_idx = 0
    assert s_true[corner_idx] == pytest.approx(s_bg, abs=1e-6)

    # each center attains s_bg + delta_s exactly (not 2*delta_s from
    # cross-talk between the well-separated bumps).
    for x0, y0 in centers:
        idx = int(np.argmin(np.sum((grid.cell_centers - [x0, y0]) ** 2, axis=1)))
        assert s_true[idx] == pytest.approx(s_bg + delta_s, abs=1e-3)
