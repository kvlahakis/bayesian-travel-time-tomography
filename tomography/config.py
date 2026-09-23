"""Hierarchical config schema for every experiment.

Deliberately not one flat dataclass: `sigma_true`/`sigma_infer` and
`ell_true`/`ell_infer` are separate required fields on `NoiseConfig` and
`PriorConfig` respectively, so truth-generation and inference parameters
cannot be confused with each other via positional arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class GridConfig:
    W: float
    n: int


@dataclass
class AcquisitionConfig:
    Ns: int
    Nr: int


@dataclass
class NoiseConfig:
    sigma_true: float
    sigma_infer: float


@dataclass
class PriorConfig:
    tau2: float
    ell_true: float
    ell_infer: float
    s_bg: float
    jitter_relative: float = 1e-6


@dataclass
class FixedTruthConfig:
    """Parameters specific to Experiment III's sharp/checkerboard truth.

    `checkerboard_block_size` has no default: it materially changes the
    truth field (and hence every downstream diagnostic), so a config that
    runs the checkerboard variant must state it explicitly rather than
    silently inherit a code-level constant.
    """

    checkerboard_block_size: int


@dataclass
class ExperimentConfig:
    grid: GridConfig
    acquisition: AcquisitionConfig
    noise: NoiseConfig
    prior: PriorConfig
    n_repeats: int
    seed: int
    # Only Experiment III's checkerboard truth needs this; every other
    # experiment (I, II, and III's smooth truth) leaves it unset.
    fixed_truth: FixedTruthConfig | None = None


def load_config(path: str) -> ExperimentConfig:
    """Load a YAML file into an `ExperimentConfig`."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(
        grid=GridConfig(**raw["grid"]),
        acquisition=AcquisitionConfig(**raw["acquisition"]),
        noise=NoiseConfig(**raw["noise"]),
        prior=PriorConfig(**raw["prior"]),
        n_repeats=raw["n_repeats"],
        seed=raw["seed"],
        fixed_truth=FixedTruthConfig(**raw["fixed_truth"]) if "fixed_truth" in raw else None,
    )


@dataclass
class ExperimentIVGeometryConfig:
    """One of Experiment IV's predetermined sensor layouts.

    `gamma` is passed directly to
    `geometry.make_boundary_clustered_sources_receivers`; `gamma=1.0` is the
    uniform baseline. `name` labels results/figures/output keys only -- it
    does not select behavior; `gamma` alone determines the geometry.
    """

    name: str
    gamma: float


@dataclass
class ExperimentIVConfig:
    """Configuration for Experiment IV: comparing predetermined acquisition
    geometries under repeated-noise, fixed-truth reconstruction.

    Deliberately a separate schema from `ExperimentConfig` (Experiments
    I-III), not a reuse/extension of it: Experiment IV's Monte Carlo
    protocol is a multi-seed x multi-geometry *paired* design (one shared
    noise draw per repetition, applied to every geometry), not a single
    `(config, n_repeats, seed)` run, so `seeds` is a list and `geometries`
    fixes the (small, fixed) set of predetermined layouts compared -- see
    `CLAUDE.md`'s "Out of scope" section for why this must not grow into a
    general acquisition-geometry sweep. There is also no truth/inference
    misspecification distinction here (unlike `NoiseConfig`/`PriorConfig`'s
    `_true`/`_infer` split): every geometry in this experiment is always
    correctly specified (the same `tau2`/`ell`/`sigma` generate the data and
    perform inference), so plain, ungated fields are used instead.
    """

    grid: GridConfig
    Ns: int
    Nr: int
    tau2: float
    ell: float
    s_bg: float
    jitter_relative: float
    sigma: float
    n_repeats_per_seed: int
    seeds: list[int]
    geometries: list[ExperimentIVGeometryConfig]


def load_experiment_iv_config(path: str) -> ExperimentIVConfig:
    """Load a YAML file into an `ExperimentIVConfig`."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    return ExperimentIVConfig(
        grid=GridConfig(**raw["grid"]),
        Ns=raw["Ns"],
        Nr=raw["Nr"],
        tau2=raw["tau2"],
        ell=raw["ell"],
        s_bg=raw["s_bg"],
        jitter_relative=raw["jitter_relative"],
        sigma=raw["sigma"],
        n_repeats_per_seed=raw["n_repeats_per_seed"],
        seeds=list(raw["seeds"]),
        geometries=[ExperimentIVGeometryConfig(**g) for g in raw["geometries"]],
    )
