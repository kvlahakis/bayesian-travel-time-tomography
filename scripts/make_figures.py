"""Thin CLI: generate the project's curated figure set into figures/.

No mathematical logic lives here -- Figures 1-2 use `geometry.py` output
directly; Figure 3 uses `run_construction_validation` (config-driven,
deterministic); Figures 4-7 read the frozen `results/baselines/` `.npz`
files (never regenerated here) plus `diagnostics.fixed_truth_expected_Q` for
Figure 6's decomposition. All actual figure drawing is in `plots.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tomography.config import load_config
from tomography.diagnostics import empirical_coverage, fixed_truth_expected_Q
from tomography.experiments import _build_geometry_and_inference_prior, run_construction_validation
from tomography.forward import forward
from tomography.geometry import build_sensitivity_matrix, make_grid, make_sources_receivers
from tomography.inversion import posterior_isotropic
from tomography.plots import (
    plot_boundary_interior_coverage,
    plot_calibration_summary,
    plot_fixed_truth_comparison,
    plot_geometry_schematic,
    plot_q_decomposition,
    plot_ray_coverage,
    plot_reconstruction_summary,
)

FIGURES_DIR = Path("figures")
BASELINES_DIR = Path("results/baselines")


def figure_1_and_2_geometry_and_ray_coverage() -> None:
    config = load_config("configs/baseline.yaml")
    grid = make_grid(config.grid.W, config.grid.n)
    sources, receivers = make_sources_receivers(config.grid.W, config.acquisition.Ns, config.acquisition.Nr)
    A = build_sensitivity_matrix(sources, receivers, grid)

    plot_geometry_schematic(
        grid, sources, receivers, save_path=str(FIGURES_DIR / "figure_1_geometry.png")
    )
    plot_ray_coverage(grid, A, save_path=str(FIGURES_DIR / "figure_2_ray_coverage.png"))
    print("Saved figure_1_geometry.png, figure_2_ray_coverage.png")


def figure_3_experiment_I_reconstruction() -> None:
    config = load_config("configs/baseline.yaml")
    result = run_construction_validation(config)
    s_std = np.sqrt(np.diag(result.C_post))
    plot_reconstruction_summary(
        result.grid, result.s_true, result.s_post, s_std,
        save_path=str(FIGURES_DIR / "figure_3_experiment_I_reconstruction.png"),
    )
    print("Saved figure_3_experiment_I_reconstruction.png")


def figure_4_experiment_II_calibration() -> None:
    npz = np.load(BASELINES_DIR / "experiment_II_baseline_n20_N1000_seed12345.npz")
    n_cells = npz["truths"].shape[1]
    coverage = dict(zip(npz["coverage_alphas"], npz["coverage_values"]))
    plot_calibration_summary(
        npz["mahalanobis"], n_cells, coverage,
        save_path=str(FIGURES_DIR / "figure_4_experiment_II_calibration.png"),
    )
    print("Saved figure_4_experiment_II_calibration.png")


def figure_5_experiment_III_fixed_truth() -> None:
    smooth_npz = np.load(BASELINES_DIR / "experiment_III_baseline_smooth_n20_N1000_seed12345.npz")
    sharp_npz = np.load(BASELINES_DIR / "experiment_III_baseline_sharp_n20_N1000_seed12345.npz")
    config = load_config("configs/calibration_fixed_truth.yaml")
    grid = make_grid(config.grid.W, config.grid.n)

    plot_fixed_truth_comparison(
        grid,
        smooth_truth=smooth_npz["truths"][0],
        smooth_post_mean=smooth_npz["posterior_means"][0],
        sharp_truth=sharp_npz["truths"][0],
        sharp_post_mean=sharp_npz["posterior_means"][0],
        save_path=str(FIGURES_DIR / "figure_5_experiment_III_fixed_truth.png"),
    )
    print("Saved figure_5_experiment_III_fixed_truth.png")


def figure_6_fixed_truth_Q_decomposition() -> None:
    config = load_config("configs/calibration_fixed_truth.yaml")
    _, A, Cs_infer, s0 = _build_geometry_and_inference_prior(config)
    dummy_d = forward(A, s0)
    _, C_post = posterior_isotropic(A, dummy_d, sigma2=config.noise.sigma_infer**2, Cs=Cs_infer, s0=s0)

    smooth_npz = np.load(BASELINES_DIR / "experiment_III_baseline_smooth_n20_N1000_seed12345.npz")
    sharp_npz = np.load(BASELINES_DIR / "experiment_III_baseline_sharp_n20_N1000_seed12345.npz")

    components = {}
    for name, npz in [("smooth", smooth_npz), ("sharp", sharp_npz)]:
        s_true = npz["truths"][0]
        decomposition = fixed_truth_expected_Q(
            A, C_post,
            sigma_infer2=config.noise.sigma_infer**2,
            sigma_true2=config.noise.sigma_true**2,
            s_true=s_true, s0=s0,
        )
        components[name] = {
            "trace": decomposition["trace"],
            "bias": decomposition["bias"],
            "theory": decomposition["theory"],
            "empirical": float(np.mean(npz["mahalanobis"])),
        }

    plot_q_decomposition(
        components, save_path=str(FIGURES_DIR / "figure_6_fixed_truth_Q_decomposition.png")
    )
    print("Saved figure_6_fixed_truth_Q_decomposition.png")


def figure_7_boundary_interior_coverage() -> None:
    # Reuses the already-established block_size=4 boundary/interior
    # definition and coverage computation documented in ARCHITECTURE.md --
    # does not extend that investigation.
    config = load_config("configs/calibration_fixed_truth.yaml")
    n = config.grid.n
    n_cells = n**2
    block_size = config.fixed_truth.checkerboard_block_size

    sharp_npz = np.load(BASELINES_DIR / "experiment_III_baseline_sharp_n20_N1000_seed12345.npz")
    truths = sharp_npz["truths"]
    posterior_means = sharp_npz["posterior_means"]
    posterior_stds = sharp_npz["posterior_stds"]
    alphas = sorted(sharp_npz["coverage_alphas"].tolist())

    k = np.arange(n_cells)
    i, j = k // n, k % n
    i_off, j_off = i % block_size, j % block_size
    on_boundary = (i_off == 0) | (i_off == block_size - 1) | (j_off == 0) | (j_off == block_size - 1)

    boundary_coverage, interior_coverage = [], []
    for alpha in alphas:
        per_cell = empirical_coverage(truths, posterior_means, posterior_stds, alpha)["per_cell"]
        boundary_coverage.append(float(per_cell[on_boundary].mean()))
        interior_coverage.append(float(per_cell[~on_boundary].mean()))

    plot_boundary_interior_coverage(
        alphas, boundary_coverage, interior_coverage,
        save_path=str(FIGURES_DIR / "figure_7_boundary_interior_coverage.png"),
    )
    print("Saved figure_7_boundary_interior_coverage.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_1_and_2_geometry_and_ray_coverage()
    figure_3_experiment_I_reconstruction()
    figure_4_experiment_II_calibration()
    figure_5_experiment_III_fixed_truth()
    figure_6_fixed_truth_Q_decomposition()
    figure_7_boundary_interior_coverage()


if __name__ == "__main__":
    main()
