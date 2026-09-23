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
from matplotlib.patches import Rectangle
from scipy.stats import chi2

from .geometry import Grid, build_sensitivity_matrix

# Shared minimal typographic/DPI style for every figure in this module --
# consistency only, not a layout redesign. Figure DPI is set once here
# (`savefig.dpi`), so individual `fig.savefig(save_path)` calls below do not
# repeat it.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 150,
})


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
        fig.savefig(save_path)

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

    # Shared color scale for truth and posterior mean, computed once and
    # passed as the same vmin/vmax to both imshow calls below -- guaranteed
    # to be identical by construction, not merely coincidentally matching.
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
    axes[2].set_title("Posterior standard deviation")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im_std, ax=axes[2], shrink=0.85, label="posterior std")

    # Not smoothed, filtered, or otherwise altered: this panel's unstructured
    # appearance (one noisy realization's error, not a systematic spatial
    # pattern) is scientifically correct and must render exactly as computed.
    error = np.abs(s_true - s_post)
    im_err = axes[3].imshow(
        _as_image(error, grid), origin="lower", extent=extent, cmap="magma"
    )
    axes[3].set_title("Absolute reconstruction error")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    fig.colorbar(im_err, ax=axes[3], shrink=0.85, label="|s_true - s_post|")

    if save_path is not None:
        fig.savefig(save_path)

    return fig


def plot_geometry_schematic(
    grid: Grid,
    sources: np.ndarray,
    receivers: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Domain/acquisition schematic: grid lines, sources (left, `x=0`),
    receivers (right, `x=W`), a small deterministic "context" subsample of
    rays (one per source, offset-paired with a receiver -- `Ns` rays total,
    not all `Ns * Nr`, since drawing every ray renders as an unreadable
    tangle at typical acquisition densities), and one specifically
    highlighted ray with the grid cells it actually intersects shaded, to
    make the ray/cell relationship concrete. The actual sensitivity matrix
    `A` still uses every ray regardless of what this schematic draws.
    """
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    for edge in grid.edges:
        ax.axhline(edge, color="lightgray", linewidth=0.5, zorder=0)
        ax.axvline(edge, color="lightgray", linewidth=0.5, zorder=0)

    Ns, Nr = len(sources), len(receivers)
    # One context ray per source, deterministically offset-paired with a
    # receiver (source i -> receiver (i + 4) mod Nr): the wraparound splits
    # the rays into two mutually-parallel families at different slopes,
    # which cross each other at many distinct points -- a legible fan.
    # Two alternatives were tried and rejected: round-robin (receiver i % Nr)
    # produced all-horizontal, fully overlapping rays whenever Ns == Nr with
    # matching source/receiver spacing (as in this project's configs); a
    # full mirror (receiver Nr-1-i) fixed that but made every ray cross
    # through one common point (the exact domain center) -- an artifact of
    # the pairing rule, not a real property of the acquisition, which could
    # be misread as a geometric fact about the rays.
    offset = 4
    for s_idx in range(Ns):
        s, r = sources[s_idx], receivers[(s_idx + offset) % Nr]
        ax.plot([s[0], r[0]], [s[1], r[1]], color="steelblue", alpha=0.4, linewidth=1.0, zorder=1)

    # Highlight one specific ray (the middle source to the middle receiver)
    # and shade the grid cells it actually intersects, using the real
    # sensitivity-matrix construction for that single ray -- not a visual
    # approximation.
    highlight_source = sources[Ns // 2 : Ns // 2 + 1]
    highlight_receiver = receivers[Nr // 2 : Nr // 2 + 1]
    A_single_ray = build_sensitivity_matrix(highlight_source, highlight_receiver, grid)
    intersected_cells = np.flatnonzero(A_single_ray[0])
    for cell in intersected_cells:
        i, j = divmod(int(cell), grid.n)
        x0, x1 = grid.edges[j], grid.edges[j + 1]
        y0, y1 = grid.edges[i], grid.edges[i + 1]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="gold", alpha=0.25, zorder=0.5))

    s_h, r_h = highlight_source[0], highlight_receiver[0]
    ax.plot([s_h[0], r_h[0]], [s_h[1], r_h[1]], color="crimson", linewidth=2.5, zorder=3,
            label=f"highlighted ray ({len(intersected_cells)} cells)")

    ax.scatter(sources[:, 0], sources[:, 1], color="crimson", marker="o", s=40,
               label="sources (x=0)", zorder=2)
    ax.scatter(receivers[:, 0], receivers[:, 1], color="darkorange", marker="^", s=40,
               label="receivers (x=W)", zorder=2)

    ax.set_xlim(-0.05 * grid.W, 1.05 * grid.W)
    ax.set_ylim(-0.05 * grid.W, 1.05 * grid.W)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Acquisition geometry: {Ns} sources × {Nr} receivers")
    ax.legend(loc="lower right", frameon=True, framealpha=0.85, fontsize=8)

    if save_path is not None:
        fig.savefig(save_path)

    return fig


def plot_ray_coverage(grid: Grid, A: np.ndarray, save_path: str | None = None) -> plt.Figure:
    """Per-cell total ray-intersection length, `A.sum(axis=0)`, reshaped onto
    the grid -- shows how the acquisition geometry's rays actually cover the
    domain (cells near the domain's horizontal midline, where most
    source-receiver rays are shortest and most direct, are typically better
    covered than corner cells).

    The near-zero-looking top/bottom rows are *exactly* zero (sources and
    receivers span an open subinterval of the boundary, per
    `geometry.make_sources_receivers`'s spacing convention, so no ray can
    reach the outermost half-cell-thick strip at either vertical extreme --
    this is expected, not a plotting artifact). The dark-but-nonzero
    left/right edge columns are a different, non-degenerate effect: cells
    there are covered by fewer/longer rays than the domain center, not by
    zero rays (see `README.md`/session notes for the verified per-cell
    values). `imshow`'s pixel centers already align with grid cell centers
    given this reshape and `extent`, so no interpolation is used or needed.
    """
    ray_length_per_cell = A.sum(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(
        _as_image(ray_length_per_cell, grid),
        origin="lower",
        extent=[0.0, grid.W, 0.0, grid.W],
        cmap="magma",
    )
    ax.set_aspect("equal")
    ax.set_title("Ray coverage across the reconstruction grid")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Total ray-intersection length")

    if save_path is not None:
        fig.savefig(save_path)

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
    axes[0].set_title("Global Q calibration")
    axes[0].legend(frameon=False)

    alphas = sorted(coverage)
    nominal = [1.0 - a for a in alphas]
    empirical = [coverage[a] for a in alphas]
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="perfect calibration")
    axes[1].plot(nominal, empirical, marker="o", color="steelblue", label="empirical coverage")
    axes[1].set_xlabel("nominal coverage (1 - alpha)")
    axes[1].set_ylabel("empirical coverage")
    axes[1].set_title("Credible-region coverage")
    # Axis limits are fixed to the full unit square only so the diagonal
    # reference line reads correctly; the empirical-coverage line itself is
    # plotted only over the actually-tested nominal levels, not extended to
    # (0, 0).
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_aspect("equal")
    axes[1].legend(frameon=False, loc="upper left")

    if save_path is not None:
        fig.savefig(save_path)

    return fig


def plot_fixed_truth_comparison(
    grid: Grid,
    smooth_truth: np.ndarray,
    smooth_post_mean: np.ndarray,
    smooth_post_std: np.ndarray,
    sharp_truth: np.ndarray,
    sharp_post_mean: np.ndarray,
    sharp_post_std: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Experiment III: smooth vs. sharp fixed truth -- truth, posterior
    mean, and posterior std, one row per truth (six panels total).

    Truth and posterior mean (both rows) share one `viridis` scale (same
    physical quantity, same units), computed once as `vmin`/`vmax` from all
    four of those panels and passed identically to each -- guaranteed by
    construction, not incidental. The two posterior-std panels likewise
    share one `magma` scale (`std_vmin`/`std_vmax` from both std panels).
    That the two posterior-std panels render identically is expected (see
    `ARCHITECTURE.md`'s fixed-truth decomposition: `C_post` depends only on
    `A`, `Cs`, and `Sigma_d`, never on `s_true`) and confirmed numerically
    for this project's frozen baselines -- see `README.md` for the finding
    stated in prose; this on-image text stays purely descriptive so the
    figure remains reusable with a different caption elsewhere.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    extent = [0.0, grid.W, 0.0, grid.W]

    slowness_fields = [smooth_truth, smooth_post_mean, sharp_truth, sharp_post_mean]
    vmin = min(f.min() for f in slowness_fields)
    vmax = max(f.max() for f in slowness_fields)

    std_fields = [smooth_post_std, sharp_post_std]
    std_vmin = min(f.min() for f in std_fields)
    std_vmax = max(f.max() for f in std_fields)

    column_titles = ["Truth", "Posterior mean", "Posterior standard deviation"]
    row_labels = ["Smooth truth", "Sharp truth"]

    for row, (truth, mean, std, row_label) in enumerate(
        [(smooth_truth, smooth_post_mean, smooth_post_std, row_labels[0]),
         (sharp_truth, sharp_post_mean, sharp_post_std, row_labels[1])]
    ):
        for col, field in enumerate([truth, mean]):
            im = axes[row, col].imshow(
                _as_image(field, grid), origin="lower", extent=extent,
                cmap="viridis", vmin=vmin, vmax=vmax,
            )
            if row == 0:
                axes[row, col].set_title(column_titles[col])
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel(f"{row_label}\ny" if col == 0 else "y")
            fig.colorbar(im, ax=axes[row, col], shrink=0.85, label="slowness")

        im_std = axes[row, 2].imshow(
            _as_image(std, grid), origin="lower", extent=extent,
            cmap="magma", vmin=std_vmin, vmax=std_vmax,
        )
        if row == 0:
            axes[row, 2].set_title(column_titles[2])
        axes[row, 2].set_xlabel("x")
        axes[row, 2].set_ylabel("y")
        fig.colorbar(im_std, ax=axes[row, 2], shrink=0.85, label="posterior std")

    fig.suptitle("Experiment III: fixed-truth reconstructions", fontsize=12)

    if save_path is not None:
        fig.savefig(save_path)

    return fig


def _grouped_log_bars(ax, components: dict, cases: list[str], quantities: list[str],
                       labels: list[str], colors, title: str) -> None:
    """Shared grouped-bar-with-value-labels helper for one panel of
    `plot_q_decomposition`."""
    x = np.arange(len(quantities))
    width = 0.8 / len(cases)
    for i, case in enumerate(cases):
        values = [components[case][q] for q in quantities]
        bars = ax.bar(x + i * width, values, width, label=case, color=colors[i])
        # Numeric value labels above every bar: on a log axis spanning
        # several orders of magnitude, a small term (e.g. the smooth-truth
        # bias, ~0.28) would otherwise be visually indistinguishable from
        # zero.
        ax.bar_label(bars, labels=[f"{v:.3g}" for v in values], fontsize=8, padding=3)

    ax.set_yscale("log")
    ax.set_xticks(x + width * (len(cases) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value (log scale)")
    ax.set_title(title)
    ax.set_ylim(top=ax.get_ylim()[1] * 4)  # headroom for the top value labels
    ax.legend(frameon=False)


def plot_q_decomposition(components: dict, save_path: str | None = None) -> plt.Figure:
    """Fixed-truth `E[Q] = tr(C_post^-1 V_post) + b^T C_post^-1 b`
    decomposition, split into two panels: "Components" (the trace/noise term
    and the bias term) and "Total E[Q]" (the theoretical sum and the
    empirical mean `Q`), grouped by case (e.g. "smooth", "sharp"), each on
    its own log y-axis with numeric value labels on every bar (the
    smooth/sharp cases differ by several orders of magnitude in this
    project's frozen baselines).

    `components` maps each case name to a dict with keys "trace", "bias",
    "theory", "empirical" (see `ARCHITECTURE.md`'s fixed-truth `E[Q]`
    section for how these are computed -- this function only plots
    already-computed values).
    """
    cases = list(components.keys())
    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(cases)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    _grouped_log_bars(
        axes[0], components, cases,
        quantities=["trace", "bias"],
        labels=["trace term\ntr(C_post^-1 V_post)", "bias term\nb^T C_post^-1 b"],
        colors=colors, title="Components",
    )
    _grouped_log_bars(
        axes[1], components, cases,
        quantities=["theory", "empirical"],
        labels=["theoretical E[Q]", "empirical mean Q"],
        colors=colors, title="Total E[Q]",
    )

    fig.suptitle(
        "Experiment III: decomposition of expected Q\n"
        "Left panel's trace + bias terms sum to the right panel's theoretical E[Q]",
        fontsize=11,
    )

    if save_path is not None:
        fig.savefig(save_path)

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
    numbers; it does not extend that investigation). The title is
    deliberately a neutral label, not a finding -- the investigation's
    conclusion (what was checked, and that none of it explains the gap)
    belongs in `README.md`'s caption for this figure, not on the image.
    """
    nominal = [1.0 - a for a in alphas]

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="perfect calibration")
    ax.plot(nominal, boundary_coverage, marker="o", color="darkorange", label="boundary cells")
    ax.plot(nominal, interior_coverage, marker="s", color="steelblue", label="interior cells")
    ax.set_xlabel("nominal coverage (1 - alpha)")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Experiment III: spatial coverage for the sharp truth")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")

    if save_path is not None:
        fig.savefig(save_path)

    return fig
