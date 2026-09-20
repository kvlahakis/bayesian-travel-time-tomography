"""Figures for the tomography experiments.

Colormap choices follow standard perceptually-uniform, colorblind-safe
sequential ramps (never a rainbow/"jet" colormap): `viridis` for slowness
fields (truth and posterior mean share one color scale, since they are
directly comparable quantities in the same units) and `magma` for posterior
standard deviation (a different quantity, on its own scale).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .geometry import Grid


def _as_image(field: np.ndarray, grid: Grid) -> np.ndarray:
    """Reshape a flat (n**2,) field into an (n, n) image for imshow.

    Cell index k = i * n + j (i indexes y, j indexes x), so a plain
    `reshape(n, n)` already gives an array indexed [i, j] = [y, x], which
    imshow (with `origin="lower"`) renders with y increasing upward and x
    increasing rightward -- matching the physical domain.
    """
    return field.reshape(grid.n, grid.n)


def plot_field_triplet(
    grid: Grid,
    s_true: np.ndarray,
    s_post: np.ndarray,
    s_std: np.ndarray,
    titles: tuple[str, str, str] = ("True field", "Posterior mean", "Posterior std"),
    save_path: str | None = None,
) -> plt.Figure:
    """Three-panel field plot: truth / posterior mean / posterior std.

    Truth and posterior mean share one `viridis` color scale (they are the
    same physical quantity, in the same units, so a shared scale makes them
    directly comparable). Posterior std uses its own `magma` scale, since it
    is a different quantity (uncertainty, not slowness).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    vmin = min(s_true.min(), s_post.min())
    vmax = max(s_true.max(), s_post.max())
    extent = [0.0, grid.W, 0.0, grid.W]

    for ax, field, title in zip(axes[:2], (s_true, s_post), titles[:2]):
        im = ax.imshow(
            _as_image(field, grid),
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85, label="slowness")

    im_std = axes[2].imshow(
        _as_image(s_std, grid),
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    axes[2].set_title(titles[2])
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im_std, ax=axes[2], shrink=0.85, label="posterior std")

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig
