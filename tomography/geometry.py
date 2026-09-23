"""Domain, grid, acquisition geometry, and the ray sensitivity matrix.

Conventions (fixed here, assumed by every other module):

- **Cell ordering.** Cell index ``k`` corresponds to grid position ``(i, j)``
  via ``k = i * n + j`` (row-major), with ``i`` indexing the ``y`` direction
  and ``j`` indexing the ``x`` direction. ``cell_centers[k]`` corresponds
  exactly to column ``k`` of ``A``, element ``k`` of ``s``, and row/column
  ``k`` of ``Cs``.
- **Source/receiver spacing.** ``Ns`` sources and ``Nr`` receivers are placed
  uniformly on the open interval ``(0, W)`` (excluding the corners), via
  ``np.linspace(0, W, Ns + 2)[1:-1]``. Sources sit on the left boundary
  (``x = 0``), receivers on the right boundary (``x = W``).
- **Cell boundary ownership.** Grid cells use a half-open convention,
  ``[x_min, x_max) x [y_min, y_max)``, except the outermost row and column,
  which include the domain's outer boundary. Concretely, a point is assigned
  to cell index ``clip(floor(coord / h), 0, n - 1)`` along each axis, which
  gives exactly this ownership rule without special-casing the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A uniform n x n grid of square cells over [0, W] x [0, W].

    ``cell_centers[k]`` follows the module's row-major cell-ordering
    convention (``k = i * n + j``). ``edges`` gives the ``n + 1`` grid-line
    coordinates shared by both axes (the domain is square and the grid is
    uniform in both directions, so a single edge array suffices).
    """

    W: float
    n: int
    h: float
    cell_centers: np.ndarray  # (n**2, 2), columns are (x, y)
    edges: np.ndarray  # (n + 1,)


def make_grid(W: float, n: int) -> Grid:
    """Construct a uniform n x n grid of square cells over [0, W] x [0, W].

    Returns a `Grid` bundling the cell centers (ordered per the module
    convention) together with the cell boundaries (`edges`).
    """
    if W <= 0:
        raise ValueError(f"W must be positive, got {W}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    h = W / n
    edges = np.linspace(0.0, W, n + 1)

    i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = (j_idx.ravel() + 0.5) * h
    y = (i_idx.ravel() + 0.5) * h
    cell_centers = np.stack([x, y], axis=1)

    return Grid(W=W, n=n, h=h, cell_centers=cell_centers, edges=edges)


def make_sources_receivers(
    W: float, Ns: int, Nr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Place Ns sources on the left boundary and Nr receivers on the right.

    Both are spaced uniformly on the open interval (0, W), excluding the
    corners, per the module's spacing convention. Returns
    ``(sources, receivers)``, each an ``(N, 2)`` array of ``(x, y)``
    coordinates.
    """
    if Ns <= 0 or Nr <= 0:
        raise ValueError(f"Ns and Nr must be positive, got Ns={Ns}, Nr={Nr}")

    y_sources = np.linspace(0.0, W, Ns + 2)[1:-1]
    y_receivers = np.linspace(0.0, W, Nr + 2)[1:-1]

    sources = np.stack([np.zeros(Ns), y_sources], axis=1)
    receivers = np.stack([np.full(Nr, W), y_receivers], axis=1)
    return sources, receivers


def make_boundary_clustered_sources_receivers(
    W: float, N: int, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Place `N` sources (`x=0`) and `N` receivers (`x=W`) using a symmetric,
    deterministic warp of `make_sources_receivers`'s uniform spacing,
    parameterized by `gamma`:

        u = linspace(0, 1, N + 2)[1:-1]                     (base uniform convention)
        u' = 0.5 + 0.5 * sign(u - 0.5) * (2 * |u - 0.5|) ** gamma
        y = W * u'

    `gamma = 1.0` recovers `make_sources_receivers(W, N, N)`'s uniform
    spacing *exactly* (`u' = u` identically, since `sign(d) * 2|d| = 2d` for
    any real `d`) -- it is not a separate special case, just this same
    formula evaluated at `gamma=1`. `gamma < 1` clusters sensors toward the
    two boundaries, more strongly as `gamma` decreases toward 0; `gamma` is
    otherwise unconstrained here.

    Sources and receivers use the identical resulting normalized layout.
    Used by Experiment IV to construct its three predetermined acquisition
    geometries (uniform: `gamma=1.0`; mild boundary clustering:
    `gamma=0.70`; strong boundary clustering: `gamma=0.40`) -- a fixed,
    non-optimized parameterization for a specific comparison, not a general
    sensor-placement design tool (see `CLAUDE.md`'s "Out of scope" section).
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    u = np.linspace(0.0, 1.0, N + 2)[1:-1]
    d = u - 0.5
    u_prime = 0.5 + 0.5 * np.sign(d) * (2.0 * np.abs(d)) ** gamma
    y = W * u_prime

    sources = np.stack([np.zeros(N), y], axis=1)
    receivers = np.stack([np.full(N, W), y], axis=1)
    return sources, receivers


def _cell_index(x: float, y: float, grid: Grid) -> int:
    """Map a point to a flat cell index per the half-open ownership rule."""
    j = int(np.clip(np.floor(x / grid.h), 0, grid.n - 1))
    i = int(np.clip(np.floor(y / grid.h), 0, grid.n - 1))
    return i * grid.n + j


def _ray_cell_lengths(
    p0: np.ndarray, p1: np.ndarray, grid: Grid, tol: float = 1e-9
) -> tuple[list[int], list[float]]:
    """Exact intersection lengths of segment p0->p1 with every grid cell it
    crosses, via parametric intersection with the grid lines (Siddon-style
    traversal specialized to a uniform grid).

    Returns parallel lists of (cell index, intersection length).
    """
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    total_length = float(np.hypot(dx, dy))
    if total_length == 0.0:
        return [], []

    n = grid.n
    edges = grid.edges

    ts = [0.0, 1.0]
    if dx != 0.0:
        t_vert = (edges[1:-1] - x0) / dx
        ts.extend(t for t in t_vert if 0.0 < t < 1.0)
    if dy != 0.0:
        t_horiz = (edges[1:-1] - y0) / dy
        ts.extend(t for t in t_horiz if 0.0 < t < 1.0)

    ts.sort()
    merged = [ts[0]]
    for t in ts[1:]:
        if t - merged[-1] > tol:
            merged.append(t)

    cell_indices: list[int] = []
    lengths: list[float] = []
    for a, b in zip(merged[:-1], merged[1:]):
        t_mid = 0.5 * (a + b)
        x_mid = x0 + t_mid * dx
        y_mid = y0 + t_mid * dy
        cell_indices.append(_cell_index(x_mid, y_mid, grid))
        lengths.append((b - a) * total_length)

    return cell_indices, lengths


def build_sensitivity_matrix(
    sources: np.ndarray, receivers: np.ndarray, grid: Grid
) -> np.ndarray:
    """Build the sensitivity matrix A of shape (m, n**2).

    Every source is connected to every receiver by a straight line segment
    (full bipartite pairing), giving ``m = len(sources) * len(receivers)``
    rays. Ray index ``i`` corresponds to ``(source index si, receiver index
    ri)`` via ``i = si * len(receivers) + ri``.

    Uses an exact ray-cell intersection-length algorithm (a Siddon-style grid
    traversal), not a pixel-rasterization approximation.
    """
    Ns = len(sources)
    Nr = len(receivers)
    m = Ns * Nr
    A = np.zeros((m, grid.n**2))

    for si in range(Ns):
        for ri in range(Nr):
            row = si * Nr + ri
            cell_indices, lengths = _ray_cell_lengths(
                sources[si], receivers[ri], grid
            )
            for k, length in zip(cell_indices, lengths):
                A[row, k] += length

    return A


def check_row_sums(
    A: np.ndarray, ray_lengths: np.ndarray, atol: float = 1e-8
) -> bool:
    """Validate that each row of A sums to the corresponding total ray length.

    Raises ``AssertionError`` (via `np.testing.assert_allclose`) if any row
    fails to match within `atol`; returns True otherwise.
    """
    row_sums = A.sum(axis=1)
    np.testing.assert_allclose(row_sums, ray_lengths, atol=atol)
    return True
