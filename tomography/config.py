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
class ExperimentConfig:
    grid: GridConfig
    acquisition: AcquisitionConfig
    noise: NoiseConfig
    prior: PriorConfig
    n_repeats: int
    seed: int


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
    )
