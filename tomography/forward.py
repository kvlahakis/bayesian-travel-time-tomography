"""Forward model and synthetic data generation.

``t = A @ s`` is the noiseless forward map; ``d = t + eps`` with
``eps ~ N(0, sigma^2 I)`` is the observation model for the independent-noise
case (Sigma_d = sigma^2 I) used throughout the experiments.
"""

from __future__ import annotations

import numpy as np

from .geometry import Grid


def forward(A: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Noiseless forward model: t = A @ s."""
    return A @ s


def add_noise(t: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add iid observation noise: d = t + eps, eps ~ N(0, sigma^2 I)."""
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    return t + rng.normal(scale=sigma, size=t.shape)


def synthetic_truth_gaussian_anomaly(
    grid: Grid, s_bg: float, delta_s: float, x0: float, y0: float, r: float
) -> np.ndarray:
    """Smooth truth: homogeneous background plus a localized Gaussian bump.

    s_true(x, y) = s_bg + delta_s * exp(-((x - x0)^2 + (y - y0)^2) / (2 r^2))

    Used as the Experiment I / II construction truth, and as the "smooth"
    fixed truth in Experiment III.
    """
    x = grid.cell_centers[:, 0]
    y = grid.cell_centers[:, 1]
    return s_bg + delta_s * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * r**2))


def synthetic_truth_checkerboard(
    grid: Grid, s_bg: float, delta_s: float, block_size: int
) -> np.ndarray:
    """Sharp/discontinuous truth: a checkerboard of alternating slowness.

    Cells are grouped into ``block_size x block_size`` blocks (using the same
    (i, j) grid indexing as ``k = i * n + j``); alternating blocks take the
    values ``s_bg + delta_s / 2`` and ``s_bg - delta_s / 2``.

    Used as the "sharp" fixed truth in Experiment III, to test how the
    posterior behaves when the fixed truth is poorly represented by the
    smooth squared-exponential prior.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    n = grid.n
    k = np.arange(n**2)
    i = k // n
    j = k % n
    parity = (i // block_size + j // block_size) % 2
    return np.where(parity == 0, s_bg + delta_s / 2.0, s_bg - delta_s / 2.0)
