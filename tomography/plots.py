"""Figures for the tomography experiments.

Colormap choices follow standard perceptually-uniform, colorblind-safe
sequential ramps (never a rainbow/"jet" colormap): `viridis` for slowness
fields (truth and posterior mean share one color scale, since they are
directly comparable quantities in the same units) and `magma` for posterior
standard deviation or ray-density fields (a different quantity, on its own
scale).

Every function here only visualizes data already computed elsewhere
(`geometry.py`, `experiments.py`, frozen `.npz` baselines) -- no
mathematical logic (Q, coverage, the E[Q] decomposition, ...) is computed in
this module; it is passed in already computed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

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


def plot_reconstruction_summary(
    grid: Grid,
    s_true: np.ndarray,
    s_post: np.ndarray,
    s_std: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Four-panel reconstruction summary: truth / posterior mean / posterior
    std / absolute reconstruction error `|s_true - s_post|`.

    Truth and posterior mean share one `viridis` scale (same physical
    quantity); posterior std and absolute error are different quantities and
    each get their own `magma` scale.
    """
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), constrained_layout=True)
    extent = [0.0, grid.W, 0.0, grid.W]

    vmin = min(s_true.min(), s_post.min())
    vmax = max(s_true.max(), s_post.max())
    for ax, field, title in zip(
        axes[:2], (s_true, s_post), ("True field", "Posterior mean")
    ):
        im = ax.imshow(
            _as_image(field, grid), origin="lower", extent=extent,
            cmap="viridis", vmin=vmin, vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85, label="slowness")

    im_std = axes[2].imshow(
        _as_image(s_std, grid), origin="lower", extent=extent, cmap="magma"
    )
    axes[2].set_title("Posterior std")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im_std, ax=axes[2], shrink=0.85, label="posterior std")

    error = np.abs(s_true - s_post)
    im_err = axes[3].imshow(
        _as_image(error, grid), origin="lower", extent=extent, cmap="magma"
    )
    axes[3].set_title("|Reconstruction error|")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    fig.colorbar(im_err, ax=axes[3], shrink=0.85, label="|s_true - s_post|")

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_geometry_schematic(
    grid: Grid,
    sources: np.ndarray,
    receivers: np.ndarray,
    max_rays_shown: int = 40,
    rng: np.random.Generator | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Domain/acquisition schematic: grid lines, sources (left, `x=0`),
    receivers (right, `x=W`), and a representative random subset of
    straight source-receiver rays (all m rays are drawn when
    `Ns * Nr <= max_rays_shown`; otherwise a random subset is shown purely
    so the figure stays legible -- the actual sensitivity matrix `A` still
    uses every ray).
    """
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    for edge in grid.edges:
        ax.axhline(edge, color="lightgray", linewidth=0.5, zorder=0)
        ax.axvline(edge, color="lightgray", linewidth=0.5, zorder=0)

    all_pairs = [(s, r) for s in sources for r in receivers]
    if len(all_pairs) > max_rays_shown:
        rng = rng if rng is not None else np.random.default_rng(0)
        idx = rng.choice(len(all_pairs), size=max_rays_shown, replace=False)
        shown_pairs = [all_pairs[i] for i in idx]
    else:
        shown_pairs = all_pairs

    for s, r in shown_pairs:
        ax.plot([s[0], r[0]], [s[1], r[1]], color="steelblue", alpha=0.35, linewidth=0.8, zorder=1)

    ax.scatter(sources[:, 0], sources[:, 1], color="crimson", marker="o", s=40,
               label="sources (x=0)", zorder=2)
    ax.scatter(receivers[:, 0], receivers[:, 1], color="darkorange", marker="^", s=40,
               label="receivers (x=W)", zorder=2)

    ax.set_xlim(-0.05 * grid.W, 1.05 * grid.W)
    ax.set_ylim(-0.05 * grid.W, 1.05 * grid.W)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Acquisition geometry ({len(sources)} sources x {len(receivers)} receivers)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_ray_coverage(grid: Grid, A: np.ndarray, save_path: str | None = None) -> plt.Figure:
    """Per-cell total ray-intersection length, `A.sum(axis=0)`, reshaped onto
    the grid -- shows how the acquisition geometry's rays actually cover the
    domain (cells near the domain's horizontal midline, where most
    source-receiver rays are shortest and most direct, are typically better
    covered than corner cells).
    """
    ray_length_per_cell = A.sum(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(
        _as_image(ray_length_per_cell, grid),
        origin="lower",
        extent=[0.0, grid.W, 0.0, grid.W],
        cmap="magma",
    )
    ax.set_title("Ray coverage: total intersection length per cell")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.85, label="sum of A[:, j] (total ray length)")

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_calibration_summary(
    Q: np.ndarray,
    n_cells: int,
    coverage: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """Experiment II calibration summary: empirical `Q` histogram against
    the theoretical `chi2(n_cells)` density (left), and empirical vs.
    nominal marginal credible-interval coverage (right).

    Uses `chi2(n_cells)` throughout -- never `chi2(n)` -- since the
    Mahalanobis statistic's degrees of freedom equal the number of unknown
    cells, not the grid's side length.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].hist(Q, bins=30, density=True, color="steelblue", alpha=0.7, label="empirical Q")
    q_grid = np.linspace(0, max(Q.max(), chi2.ppf(0.999, n_cells)), 400)
    axes[0].plot(q_grid, chi2.pdf(q_grid, n_cells), color="crimson", linewidth=2,
                 label=f"chi2(n_cells={n_cells}) density")
    axes[0].set_xlabel("Q")
    axes[0].set_ylabel("density")
    axes[0].set_title("Experiment II: Q vs. chi2(n_cells)")
    axes[0].legend(frameon=False)

    alphas = sorted(coverage)
    nominal = [1.0 - a for a in alphas]
    empirical = [coverage[a] for a in alphas]
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="perfect calibration")
    axes[1].plot(nominal, empirical, marker="o", color="steelblue", label="empirical coverage")
    axes[1].set_xlabel("nominal coverage (1 - alpha)")
    axes[1].set_ylabel("empirical coverage")
    axes[1].set_title("Experiment II: coverage vs. nominal")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_aspect("equal")
    axes[1].legend(frameon=False, loc="upper left")

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_fixed_truth_comparison(
    grid: Grid,
    smooth_truth: np.ndarray,
    smooth_post_mean: np.ndarray,
    sharp_truth: np.ndarray,
    sharp_post_mean: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Experiment III: smooth vs. sharp fixed truth, truth and posterior
    mean side by side for each, on one shared `viridis` scale (all four
    panels are the same physical quantity, in the same units).
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    extent = [0.0, grid.W, 0.0, grid.W]

    fields = [
        (smooth_truth, "Smooth truth (Gaussian anomaly)"),
        (smooth_post_mean, "Smooth: posterior mean"),
        (sharp_truth, "Sharp truth (checkerboard)"),
        (sharp_post_mean, "Sharp: posterior mean"),
    ]
    vmin = min(f.min() for f, _ in fields)
    vmax = max(f.max() for f, _ in fields)

    for ax, (field, title) in zip(axes.ravel(), fields):
        im = ax.imshow(
            _as_image(field, grid), origin="lower", extent=extent,
            cmap="viridis", vmin=vmin, vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85, label="slowness")

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_q_decomposition(components: dict, save_path: str | None = None) -> plt.Figure:
    """Fixed-truth `E[Q] = tr(C_post^-1 V_post) + b^T C_post^-1 b`
    decomposition: grouped bars per case (e.g. "smooth", "sharp") showing
    the trace (noise) term, the bias term, the theoretical total, and the
    empirical mean `Q`, on a log scale (the smooth/sharp cases differ by
    several orders of magnitude in this project's frozen baselines).

    `components` maps each case name to a dict with keys "trace", "bias",
    "theory", "empirical" (see `ARCHITECTURE.md`'s fixed-truth `E[Q]`
    section for how these are computed -- this function only plots
    already-computed values).
    """
    cases = list(components.keys())
    quantities = ["trace", "bias", "theory", "empirical"]
    labels = ["trace term\ntr(C_post^-1 V_post)", "bias term\nb^T C_post^-1 b",
              "theoretical E[Q]", "empirical mean Q"]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(len(quantities))
    width = 0.8 / len(cases)
    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(cases)))

    for i, case in enumerate(cases):
        values = [components[case][q] for q in quantities]
        ax.bar(x + i * width, values, width, label=case, color=colors[i])

    ax.set_yscale("log")
    ax.set_xticks(x + width * (len(cases) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value (log scale)")
    ax.set_title("Experiment III: fixed-truth E[Q] decomposition")
    ax.legend(frameon=False)

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_boundary_interior_coverage(
    alphas: list[float],
    boundary_coverage: list[float],
    interior_coverage: list[float],
    save_path: str | None = None,
) -> plt.Figure:
    """Experiment III sharp-truth spatial diagnostic: empirical coverage for
    boundary vs. interior checkerboard cells, at each nominal alpha level
    (see `ARCHITECTURE.md`'s boundary/interior investigation -- this
    function only plots the already-established, already-documented
    numbers; it does not extend that investigation).
    """
    nominal = [1.0 - a for a in alphas]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="perfect calibration")
    ax.plot(nominal, boundary_coverage, marker="o", color="darkorange", label="boundary cells")
    ax.plot(nominal, interior_coverage, marker="s", color="steelblue", label="interior cells")
    ax.set_xlabel("nominal coverage (1 - alpha)")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Experiment III (sharp): boundary vs. interior coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig
