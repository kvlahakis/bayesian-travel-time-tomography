"""Thin CLI: YAML config -> tomography.config -> tomography.experiments -> results/.

No mathematical logic lives here -- only argument parsing, config loading,
dispatch to `experiments.py`, and saving outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tomography.config import ExperimentConfig, load_config
from tomography.experiments import (
    run_construction_validation,
    run_correctly_specified,
    run_fixed_truth,
)
from tomography.forward import synthetic_truth_checkerboard, synthetic_truth_gaussian_anomaly
from tomography.geometry import make_grid
from tomography.plots import plot_field_triplet


def _smooth_fixed_truth(config: ExperimentConfig) -> np.ndarray:
    # Matches run_construction_validation's (Experiment I's) own default
    # exactly -- delta_s is *derived* from the prior's marginal variance,
    # not a separately chosen literal, so the smooth fixed truth here really
    # is "the Experiment I truth", not a coincidentally similar new field.
    grid = make_grid(config.grid.W, config.grid.n)
    return synthetic_truth_gaussian_anomaly(
        grid,
        s_bg=config.prior.s_bg,
        delta_s=np.sqrt(config.prior.tau2),
        x0=config.grid.W / 2,
        y0=config.grid.W / 2,
        r=config.grid.W / 6,
    )


def _sharp_fixed_truth(config: ExperimentConfig) -> np.ndarray:
    if config.fixed_truth is None:
        raise ValueError(
            "--experiment fixed_truth_sharp requires a `fixed_truth: "
            "{checkerboard_block_size: ...}` section in the config -- see "
            "configs/calibration_fixed_truth.yaml."
        )
    grid = make_grid(config.grid.W, config.grid.n)
    return synthetic_truth_checkerboard(
        grid,
        s_bg=config.prior.s_bg,
        delta_s=np.sqrt(config.prior.tau2),
        block_size=config.fixed_truth.checkerboard_block_size,
    )


EXPERIMENTS = {
    "construction_validation": run_construction_validation,
    "correctly_specified": lambda config: run_correctly_specified(
        config, n_repeats=config.n_repeats
    ),
    "fixed_truth_smooth": lambda config: run_fixed_truth(
        config, _smooth_fixed_truth(config), n_repeats=config.n_repeats
    ),
    "fixed_truth_sharp": lambda config: run_fixed_truth(
        config, _sharp_fixed_truth(config), n_repeats=config.n_repeats
    ),
}


def _save_construction_validation(result, out_path: Path) -> None:
    np.savez(
        out_path,
        s_true=result.s_true,
        s_post=result.s_post,
        C_post=result.C_post,
        d=result.d,
        A=result.A,
    )


def _save_experiment_results(result, out_path: Path) -> None:
    """Saver for any repeated-run `ExperimentResults` (Experiments II/III/...):
    `run_correctly_specified` and `run_fixed_truth` both return this same
    stacked-diagnostics shape.
    """
    np.savez(
        out_path,
        truths=result.truths,
        posterior_means=result.posterior_means,
        posterior_stds=result.posterior_stds,
        relative_errors=result.relative_errors,
        z_scores=result.z_scores,
        mahalanobis=result.mahalanobis,
        coverage_alphas=np.array(list(result.coverage.keys())),
        coverage_values=np.array(list(result.coverage.values())),
    )


SAVERS = {
    "construction_validation": _save_construction_validation,
    "correctly_specified": _save_experiment_results,
    "fixed_truth_smooth": _save_experiment_results,
    "fixed_truth_sharp": _save_experiment_results,
}

# Experiments whose result carries a grid/s_true/s_post/C_post suitable for
# the three-panel field plot. The repeated-run experiments stack many
# realizations instead, so they have no single field to plot here -- see
# plots.py for the Q histogram and coverage plots that suit them.
PLOTTABLE = {"construction_validation"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tomography experiment from a YAML config."
    )
    parser.add_argument("config", type=str, help="Path to a YAML config file.")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="construction_validation",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save raw outputs to.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also produce and save the experiment's figure(s).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_fn = EXPERIMENTS[args.experiment]
    result = run_fn(config)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.experiment}.npz"
    SAVERS[args.experiment](result, out_path)
    print(f"Saved results to {out_path}")

    if args.plot:
        if args.experiment not in PLOTTABLE:
            raise ValueError(f"--plot is not supported for --experiment {args.experiment}")
        s_std = np.sqrt(np.diag(result.C_post))
        fig_path = results_dir / f"{args.experiment}.png"
        plot_field_triplet(
            result.grid, result.s_true, result.s_post, s_std, save_path=str(fig_path)
        )
        print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
