import numpy as np
import pytest

from tomography.geometry import (
    build_sensitivity_matrix,
    check_row_sums,
    make_grid,
    make_sources_receivers,
)


def test_make_grid_cell_centers_ordering():
    W, n = 4.0, 4
    grid = make_grid(W, n)
    h = W / n
    assert grid.h == pytest.approx(h)
    assert grid.cell_centers.shape == (n**2, 2)

    # k = i * n + j, i indexes y, j indexes x.
    for i in range(n):
        for j in range(n):
            k = i * n + j
            expected = np.array([(j + 0.5) * h, (i + 0.5) * h])
            np.testing.assert_allclose(grid.cell_centers[k], expected)


def test_make_grid_edges():
    grid = make_grid(W=5.0, n=10)
    np.testing.assert_allclose(grid.edges, np.linspace(0.0, 5.0, 11))


def test_make_sources_receivers_spacing_and_boundaries():
    W, Ns, Nr = 10.0, 3, 5
    sources, receivers = make_sources_receivers(W, Ns, Nr)

    assert sources.shape == (Ns, 2)
    assert receivers.shape == (Nr, 2)

    # sources on x = 0, receivers on x = W
    np.testing.assert_allclose(sources[:, 0], 0.0)
    np.testing.assert_allclose(receivers[:, 0], W)

    # spacing excludes the corners (open interval)
    expected_y_sources = np.linspace(0.0, W, Ns + 2)[1:-1]
    expected_y_receivers = np.linspace(0.0, W, Nr + 2)[1:-1]
    np.testing.assert_allclose(sources[:, 1], expected_y_sources)
    np.testing.assert_allclose(receivers[:, 1], expected_y_receivers)
    assert sources[:, 1].min() > 0.0
    assert sources[:, 1].max() < W
    assert receivers[:, 1].min() > 0.0
    assert receivers[:, 1].max() < W


def test_row_sums_equal_ray_lengths():
    W, n = 4.0, 20
    Ns, Nr = 6, 7
    grid = make_grid(W, n)
    sources, receivers = make_sources_receivers(W, Ns, Nr)
    A = build_sensitivity_matrix(sources, receivers, grid)

    ray_lengths = np.array(
        [
            np.linalg.norm(receivers[ri] - sources[si])
            for si in range(Ns)
            for ri in range(Nr)
        ]
    )

    assert A.shape == (Ns * Nr, n**2)
    assert check_row_sums(A, ray_lengths, atol=1e-8)


def test_row_sums_single_axis_aligned_ray():
    # A horizontal ray (dy = 0) is a useful degenerate case for the
    # traversal algorithm: it should cross one full row of cells cleanly.
    W, n = 4.0, 4
    grid = make_grid(W, n)
    source = np.array([[0.0, 1.5]])
    receiver = np.array([[W, 1.5]])
    A = build_sensitivity_matrix(source, receiver, grid)

    ray_length = np.array([W])
    assert check_row_sums(A, ray_length, atol=1e-8)
    # entirely within row i=1 (y in [1, 2))
    nonzero_cells = np.flatnonzero(A[0])
    expected_cells = np.array([1 * n + j for j in range(n)])
    np.testing.assert_array_equal(np.sort(nonzero_cells), expected_cells)


def test_ray_through_grid_vertices_hits_exact_diagonal_cells():
    # A ray along the main diagonal of a 4x4 grid of unit cells passes
    # exactly through the interior grid vertices (1,1), (2,2), (3,3),
    # without ever running along a shared cell edge. Per the half-open
    # [x_min, x_max) x [y_min, y_max) ownership convention, it should
    # therefore traverse exactly the four diagonal cells (0,0), (1,1),
    # (2,2), (3,3) -- never touching the off-diagonal cells it grazes at
    # each vertex.
    W, n = 4.0, 4
    grid = make_grid(W, n)
    source = np.array([[0.0, 0.0]])
    receiver = np.array([[4.0, 4.0]])
    A = build_sensitivity_matrix(source, receiver, grid)

    expected_cells = np.array([i * n + i for i in range(n)])
    nonzero_cells = np.flatnonzero(A[0])
    np.testing.assert_array_equal(np.sort(nonzero_cells), expected_cells)

    # each diagonal cell contributes a length-sqrt(2) segment (unit cells)
    np.testing.assert_allclose(A[0, expected_cells], np.sqrt(2.0))

    ray_length = np.array([np.linalg.norm(receiver[0] - source[0])])
    assert check_row_sums(A, ray_length, atol=1e-8)


def test_check_row_sums_raises_on_mismatch():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(AssertionError):
        check_row_sums(A, ray_lengths=np.array([10.0, 10.0]), atol=1e-8)
