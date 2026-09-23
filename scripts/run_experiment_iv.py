"""Thin CLI: YAML config -> tomography.config -> tomography.experiments -> results/.

Runs Experiment IV (predetermined acquisition-geometry comparison) and saves
its per-geometry, per-paired-comparison, and per-seed outputs to a single
``.npz`` file, plus (optionally) the two Experiment IV figures. No
mathematical logic lives here -- only argument parsing, config loading,
dispatch to `experiments.py`, and saving outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tomography.config import load_experiment_iv_config
from tomography.experiments import ExperimentIVResults, run_acquisition_geometry_comparison
from tomography.plots import plot_geometry_reconstruction_comparison, plot_geometry_seed_robustness


def _save_experiment_iv_results(result: ExperimentIVResults, out_path: Path) -> None:
    """Flatten `ExperimentIVResults` (nested by geometry/paired-comparison
    name) into a single flat `.npz` archive, using ``"<name>__<field>"`` keys
    -- `np.savez` has no native support for nested structures.
    """
    arrays: dict[str, np.ndarray] = {
        "seeds": np.array(result.seeds),
        "n_repeats_per_seed": np.array(result.n_repeats_per_seed),
        "geometry_names": np.array(list(result.geometries.keys())),
        "paired_comparison_names": np.array(list(result.paired_comparisons.keys())),
    }

    for name, geom in result.geometries.items():
        arrays[f"{name}__e_rel"] = geom.e_rel
        arrays[f"{name}__correlation"] = geom.correlation
        arrays[f"{name}__posterior_std"] = geom.posterior_std
        arrays[f"{name}__J_var"] = np.array(geom.J_var)
        arrays[f"{name}__J_logdet"] = np.array(geom.J_logdet)
        arrays[f"{name}__rank"] = np.array(geom.rank)
        arrays[f"{name}__r_eff"] = np.array(geom.r_eff)

    for name, paired in result.paired_comparisons.items():
        arrays[f"{name}__delta_e_rel"] = paired.delta_e_rel
        arrays[f"{name}__delta_correlation"] = paired.delta_correlation
        arrays[f"{name}__seed_level_mean_delta_e_rel"] = paired.seed_level_mean_delta_e_rel
        arrays[f"{name}__seed_level_std_delta_e_rel"] = paired.seed_level_std_delta_e_rel
        arrays[f"{name}__seed_level_mean_delta_correlation"] = paired.seed_level_mean_delta_correlation
        arrays[f"{name}__seed_level_std_delta_correlation"] = paired.seed_level_std_delta_correlation
        arrays[f"{name}__seed_level_n_improved_e_rel"] = paired.seed_level_n_improved_e_rel
        arrays[f"{name}__seed_level_frac_improved_e_rel"] = paired.seed_level_frac_improved_e_rel
        arrays[f"{name}__seed_level_n_improved_correlation"] = paired.seed_level_n_improved_correlation
        arrays[f"{name}__seed_level_frac_improved_correlation"] = paired.seed_level_frac_improved_correlation

    np.savez(out_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment IV (acquisition-geometry comparison) from a YAML config."
    )
    parser.add_argument("config", type=str, help="Path to an Experiment IV YAML config file.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save raw outputs to.",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="experiment_iv",
        help="Base filename (without extension) for the saved results/figures.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also produce and save Experiment IV's two figures.",
    )
    args = parser.parse_args()

    config = load_experiment_iv_config(args.config)
    result = run_acquisition_geometry_comparison(config)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.out_name}.npz"
    _save_experiment_iv_results(result, out_path)
    print(f"Saved results to {out_path}")

    if args.plot:
        e_rel_by_geometry = {name: g.e_rel for name, g in result.geometries.items()}
        fig_a_path = results_dir / f"{args.out_name}_geometry_comparison.png"
        plot_geometry_reconstruction_comparison(e_rel_by_geometry, save_path=str(fig_a_path))
        print(f"Saved figure to {fig_a_path}")

        seed_level_deltas = {
            name: paired.seed_level_mean_delta_e_rel
            for name, paired in result.paired_comparisons.items()
        }
        fig_b_path = results_dir / f"{args.out_name}_seed_robustness.png"
        plot_geometry_seed_robustness(result.seeds, seed_level_deltas, save_path=str(fig_b_path))
        print(f"Saved figure to {fig_b_path}")


if __name__ == "__main__":
    main()
