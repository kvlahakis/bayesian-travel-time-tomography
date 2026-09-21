# Bayesian Linear Travel-Time Tomography

A small, reproducible computational study of a finite-dimensional Bayesian linear
inverse problem motivated by crosshole seismic travel-time tomography. The
slowness field is discretized on a uniform grid, travel times are modeled as
straight-ray line integrals, and the resulting linear-Gaussian posterior is
computed in closed form (no MCMC, no iterative optimization).

The full scientific specification — problem statement, prior/likelihood model,
and the three experiments (construction/validation, correctly-specified
Bayesian calibration, and fixed-truth coverage under a well-matched and a
mismatched truth) — is in `experiment.pdf`.

**Headline result:** under the correctly specified generative model
(Experiment II), the posterior is exactly calibrated — `Q ~ chi2(n_cells)`, confirmed
empirically. Under a fixed physical truth (Experiment III), that guarantee no
longer holds: a smooth truth well matched to the prior's assumed smoothness
remains well calibrated (in fact conservatively over-covers), while a sharp,
prior-mismatched truth produces severe undercoverage — despite an identically
well-defined posterior in both cases. This distinction, between a posterior
being mathematically well defined and a posterior being a trustworthy
description of physical uncertainty, is the project's central finding.

## Model

Domain `Omega = [0, W]^2`, sources on `x=0`, receivers on `x=W`, straight rays
connecting every source to every receiver. Slowness `s = 1/v` is discretized
on an `n x n` grid (`p = n**2` unknown cells), giving the linear forward model
`t = A s`, with `A_ij` the exact length of ray `i` through cell `j`. Synthetic
data are `d = A s_true + epsilon`, `epsilon ~ N(0, Sigma_d)` (`Sigma_d =
sigma^2 I` throughout this project). The prior is `s ~ N(s0, Cs)`, with
constant background mean and a squared-exponential covariance. The exact
Gaussian posterior,

```
C_post = (A^T Sigma_d^-1 A + Cs^-1)^-1
s_post = C_post (A^T Sigma_d^-1 d + Cs^-1 s0)
```

is computed via Cholesky factorizations and triangular solves throughout —
never an explicit matrix inverse (see `inversion.py`).

See `experiment.pdf` for the full mathematical derivation, `ARCHITECTURE.md`
for the module dependency structure and implementation-level diagnostics
(including why Experiment III's per-cell `z`/KS diagnostics don't resemble
Experiment II's even for a well-matched truth, and the follow-up investigation
into a boundary/interior coverage effect observed in Experiment III), and
`CLAUDE.md` for the module contracts and conventions used throughout the
implementation.

## Status

Complete. The project implements and validates Experiment I
(construction/validation), Experiment II (correctly specified Bayesian
calibration), and Experiment III (fixed-truth coverage, both a smooth and a
sharp truth), plus a follow-up diagnostic investigation into an observed
coverage effect in Experiment III (see `ARCHITECTURE.md`). The package
(`geometry.py`, `forward.py`, `prior.py`, `inversion.py`, `diagnostics.py`,
`experiments.py`, `plots.py`, `config.py`) and its test suite are fully
implemented.

Reproduce each experiment with:

```bash
# Experiment I: construction and validation
scripts/run_experiment.py configs/baseline.yaml --plot

# Experiment II: correctly specified Bayesian calibration
scripts/run_experiment.py configs/calibration_correct.yaml --experiment correctly_specified

# Experiment III: fixed-truth coverage (smooth or sharp)
scripts/run_experiment.py configs/calibration_fixed_truth.yaml --experiment fixed_truth_smooth
scripts/run_experiment.py configs/calibration_fixed_truth.yaml --experiment fixed_truth_sharp
```

The `n=20`, `N=1000`, `seed=12345` run under `results/baselines/` (files
`experiment_II_baseline_n20_N1000_seed12345.{npz,yaml}`) is the validated,
frozen reference result for Experiment II. It should not be overwritten by a
run with a different seed or configuration without also updating this note.

Likewise, `experiment_III_baseline_smooth_n20_N1000_seed12345.npz` and
`experiment_III_baseline_sharp_n20_N1000_seed12345.npz` under
`results/baselines/` (sharing config
`experiment_III_baseline_config_n20_N1000_seed12345.yaml`, identical to
`configs/calibration_fixed_truth.yaml`) are the validated, frozen reference
results for Experiment III's smooth-vs-sharp fixed-truth comparison. They
should not be overwritten by a run with a different seed, truth, or
configuration without also updating this note.

## Scope

This project is deliberately limited to a finite-dimensional, closed-form
linear-Gaussian model. It does not include a systematic acquisition-geometry
sweep, a controlled prior-misspecification experiment, or a controlled
noise-misspecification experiment: the fixed-truth comparison in Experiment
III already demonstrates the central phenomenon those experiments would also
illustrate — a well-defined posterior failing to describe physical uncertainty
under model mismatch — via a mismatched truth rather than a mismatched prior
or noise model. See `experiment.pdf`'s "Scope and interpretation" section for
the complete list of excluded effects (curved rays, attenuation, wave-equation
simulation, and others).

## Figures

Curated final figures live under `figures/` (tracked; regenerate with
`scripts/make_figures.py`, which reads `configs/baseline.yaml` and the frozen
`results/baselines/` outputs and never regenerates those baselines):

![Experiment I reconstruction: true field, posterior mean, posterior std, and reconstruction error](figures/figure_3_experiment_I_reconstruction.png)

1. `figure_1_geometry.png` — domain, sources, receivers, a small deterministic
   sample of rays (one per source) plus one highlighted ray with the grid
   cells it actually intersects shaded.
2. `figure_2_ray_coverage.png` — per-cell total ray-intersection length.
3. `figure_3_experiment_I_reconstruction.png` (above) — true field / posterior
   mean / posterior std / reconstruction error. The fourth panel is the
   absolute error for a *single noisy realization*; its unstructured
   appearance reflects that one realization's noise draw, not a systematic
   spatial error pattern.
4. `figure_4_experiment_II_calibration.png` — `Q` vs. `chi2(n_cells)`, and
   coverage vs. nominal.
5. `figure_5_experiment_III_fixed_truth.png` — smooth vs. sharp truth,
   posterior mean, and posterior std (six panels). The two posterior-std
   panels are visually indistinguishable, and this is the *correct* result,
   not a plotting artifact: `C_post` depends only on `A`, `Cs`, and `Sigma_d`
   — never on `s_true` (confirmed numerically: the two frozen baselines'
   `posterior_stds` are bit-identical) — so the posterior's claimed
   uncertainty is the same regardless of whether the fixed truth matches the
   prior.
6. `figure_6_fixed_truth_Q_decomposition.png` — the `E[Q] = tr(C_post^-1
   V_post) + b^T C_post^-1 b` decomposition, smooth vs. sharp, with numeric
   value labels on every bar (the smooth bias term, ~0.28, is otherwise
   invisible on a log axis spanning ~10^0 to ~10^7).
7. `figure_7_boundary_interior_coverage.png` — the sharp-truth
   boundary/interior coverage comparison from `ARCHITECTURE.md`'s
   investigation (shown as-is, not re-derived); the subtitle on the figure
   itself states what was checked and that none of it explains the gap.

## Development

```bash
pip install -e ".[dev]"
pytest                                            # full test suite

# smoke test: exercises the full pipeline on a tiny config, not a
# calibration claim (does not write to results/baselines/)
scripts/run_experiment.py configs/smoke.yaml --experiment construction_validation --results-dir results/smoke

# modest scaling benchmark (forward-matrix construction, posterior solve)
# at a few grid sizes; writes to results/benchmarks/, not results/baselines/
scripts/benchmark_scaling.py

# regenerate figures/ from configs/baseline.yaml and the frozen baselines
scripts/make_figures.py
```
