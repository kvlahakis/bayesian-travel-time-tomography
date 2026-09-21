"""Modest computational-scaling benchmark.

Measures, at a few representative grid sizes, how long forward-matrix
construction and the posterior solve take, and how large the resulting
objects are. This documents computational behavior at these sizes -- it is
not a formal asymptotic scaling study, and no complexity claims beyond the
direct measurements should be drawn from it.

Uses only existing project APIs (geometry.py, prior.py, forward.py,
inversion.py) -- no implementation is duplicated here. Acquisition geometry
(W, Ns, Nr) and prior/noise hyperparameters are held fixed at
configs/baseline.yaml's values; only the grid resolution `n` varies, so the
benchmark isolates the cost of increasing the number of unknown cells
`p = n**2`.

Writes a CSV to results/benchmarks/ (gitignored, like the rest of results/
except results/baselines/) -- never touches results/baselines/.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from tomography.forward import add_noise, forward, synthetic_truth_gaussian_anomaly
from tomography.geometry import build_sensitivity_matrix, make_grid, make_sources_receivers
from tomography.inversion import posterior_isotropic
from tomography.prior import squared_exponential_cov

# Fixed across all grid sizes, matching configs/baseline.yaml, so the
# benchmark isolates the effect of the number of cells p = n**2.
W = 10.0
NS = 12
NR = 12
TAU2 = 0.04
ELL = 2.0
S_BG = 1.0
JITTER_RELATIVE = 1.0e-6
SIGMA = 0.02
SEED = 0

GRID_SIZES = [10, 20, 30, 40]


def benchmark_one(n: int) -> dict:
    rng = np.random.default_rng(SEED)

    grid = make_grid(W, n)
    sources, receivers = make_sources_receivers(W, NS, NR)

    t0 = time.perf_counter()
    A = build_sensitivity_matrix(sources, receivers, grid)
    forward_matrix_time_s = time.perf_counter() - t0

    Cs = squared_exponential_cov(
        grid.cell_centers, tau2=TAU2, ell=ELL, jitter_relative=JITTER_RELATIVE
    )
    s0 = S_BG * np.ones(grid.n**2)
    s_true = synthetic_truth_gaussian_anomaly(
        grid, s_bg=S_BG, delta_s=np.sqrt(TAU2), x0=W / 2, y0=W / 2, r=W / 6
    )
    d = add_noise(forward(A, s_true), sigma=SIGMA, rng=rng)

    t0 = time.perf_counter()
    s_post, C_post = posterior_isotropic(A, d, sigma2=SIGMA**2, Cs=Cs, s0=s0)
    posterior_solve_time_s = time.perf_counter() - t0

    n_cells = grid.n**2
    m_rays = A.shape[0]
    n_nonzero = int(np.count_nonzero(A))

    return {
        "n": n,
        "n_cells": n_cells,
        "m_rays": m_rays,
        "A_shape": f"{A.shape[0]}x{A.shape[1]}",
        "A_nonzeros": n_nonzero,
        "A_density": n_nonzero / A.size,
        "A_bytes": A.nbytes,
        "Cs_bytes": Cs.nbytes,
        "C_post_bytes": C_post.nbytes,
        "forward_matrix_time_s": forward_matrix_time_s,
        "posterior_solve_time_s": posterior_solve_time_s,
    }


def main() -> None:
    rows = [benchmark_one(n) for n in GRID_SIZES]

    out_dir = Path("results/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scaling_benchmark.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'n':>4} {'p=n^2':>7} {'m_rays':>7} {'A nnz':>8} {'A MB':>8} "
          f"{'C_post MB':>10} {'fwd-matrix (s)':>15} {'post-solve (s)':>15}")
    for r in rows:
        print(
            f"{r['n']:>4} {r['n_cells']:>7} {r['m_rays']:>7} {r['A_nonzeros']:>8} "
            f"{r['A_bytes'] / 1e6:>8.3f} {r['C_post_bytes'] / 1e6:>10.3f} "
            f"{r['forward_matrix_time_s']:>15.4f} {r['posterior_solve_time_s']:>15.4f}"
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
