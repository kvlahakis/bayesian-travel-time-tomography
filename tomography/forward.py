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

    A single-bump building block. It is no longer used directly as any
    experiment's frozen truth (see `synthetic_truth_bimodal`, which is built
    from two calls to this function), but remains implemented and tested in
    its own right.
    """
    x = grid.cell_centers[:, 0]
    y = grid.cell_centers[:, 1]
    return s_bg + delta_s * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * r**2))


def synthetic_truth_bimodal(
    grid: Grid, s_bg: float, delta_s: float, centers: list[tuple[float, float]], r: float
) -> np.ndarray:
    """Smooth bimodal truth: homogeneous background plus two localized
    Gaussian bumps of identical amplitude `delta_s` and width `r`, summed.

    s_true(x, y) = s_bg + sum_k delta_s * exp(-((x-x_k)^2+(y-y_k)^2)/(2 r^2))

    Built from two calls to `synthetic_truth_gaussian_anomaly` with `s_bg=0`
    (extracting just each bump term), summing the bump terms, and adding
    `s_bg` once here -- not from two independently-backgrounded fields,
    which would double-count the background.

    This is the Experiment I truth, reused as Experiment III's well-matched
    fixed truth and as Experiment IV's fixed reconstruction target. See
    `CLAUDE.md`'s `forward.py` module contract for the frozen
    parameterization (`centers = [(0.25*W, 0.475*W), (0.75*W, 0.475*W)]`,
    `delta_s = sqrt(tau2)`, `r = W/6`) and why the centers' `y = 0.475*W`
    (not the domain's exact center `0.5*W`) is deliberate: at `n=20`,
    `y=0.5*W` falls exactly on a cell boundary, splitting each intended peak
    into a tied pair of grid cells rather than one clean maximum.
    """
    bumps = sum(
        synthetic_truth_gaussian_anomaly(grid, s_bg=0.0, delta_s=delta_s, x0=x0, y0=y0, r=r)
        for x0, y0 in centers
    )
    return s_bg + bumps


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
