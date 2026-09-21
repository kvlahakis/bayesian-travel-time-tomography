"""Calibration diagnostics for the Bayesian posterior.

This module is the foundation of every calibration claim in the write-up:
relative reconstruction error, per-cell standardized errors, the global
Mahalanobis statistic, and marginal credible-interval coverage (for a single
realization, and aggregated across repeated realizations).

Note on "calibrated": whether these diagnostics behave as their nominal
theory predicts (``Q ~ chi2(n_cells)``, ``z ~ N(0, 1)``, empirical coverage close to
nominal) depends entirely on *how the truth used to compute them was
generated* -- see `ARCHITECTURE.md` for the distinction between the
correctly-specified generative model (Experiment II) and a fixed physical
truth (Experiment III). This module only computes the diagnostics; it makes
no claim about which regime it is being used in.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm


def relative_error(s_post: np.ndarray, s_true: np.ndarray) -> float:
    """Relative L2 reconstruction error, ||s_post - s_true|| / ||s_true||."""
    return float(np.linalg.norm(s_post - s_true) / np.linalg.norm(s_true))


def standardized_errors(
    s_true: np.ndarray, s_post: np.ndarray, C_post: np.ndarray
) -> np.ndarray:
    """Per-cell standardized error z_j = (s_true_j - s_post_j) / sqrt((C_post)_jj)."""
    post_std = np.sqrt(np.diag(C_post))
    return (s_true - s_post) / post_std


def mahalanobis(s_true: np.ndarray, s_post: np.ndarray, C_post: np.ndarray) -> float:
    """Global Mahalanobis statistic Q = e^T C_post^-1 e, e = s_true - s_post.

    Computed via a Cholesky factorization of `C_post` and a triangular solve,
    never an explicit matrix inverse.
    """
    e = s_true - s_post
    C_post_factor = cho_factor(C_post, lower=True)
    return float(e @ cho_solve(C_post_factor, e))


def marginal_coverage(
    s_true: np.ndarray, s_post: np.ndarray, C_post: np.ndarray, alpha: float
) -> float:
    """Fraction of cells whose marginal credible interval covers the truth.

    For a single realization: the interval for cell j is
    ``s_post_j +/- z_{1-alpha/2} sqrt((C_post)_jj)``, and this returns the
    fraction of cells (out of all n_cells) for which `s_true_j` falls inside
    it.
    """
    post_std = np.sqrt(np.diag(C_post))
    return _covered_fraction(s_true, s_post, post_std, alpha)


def empirical_coverage(
    truths: np.ndarray,
    posterior_means: np.ndarray,
    posterior_stds: np.ndarray,
    alpha: float,
) -> dict:
    """Empirical coverage of marginal credible intervals across repeated realizations.

    `truths`, `posterior_means`, `posterior_stds` are each `(n_repeats,
    n_cells)`. Only the marginal posterior standard deviation per cell is
    needed to build the credible interval (see PDF Section 13), so this
    takes `posterior_stds` rather than a stack of full `(n_cells, n_cells)`
    posterior covariance matrices -- consistent with `ExperimentResults`,
    which likewise stores `posterior_stds` per realization rather than the
    full covariance.

    Returns a dict with keys ``"per_cell"`` (`(n_cells,)`, coverage fraction
    across realizations for each cell) and ``"overall"`` (scalar, coverage
    fraction across all realizations and cells). This is what feeds the
    nominal-vs-empirical coverage plot.
    """
    covered = _covered_fraction_mask(truths, posterior_means, posterior_stds, alpha)
    return {"per_cell": covered.mean(axis=0), "overall": float(covered.mean())}


def _covered_fraction_mask(
    truths: np.ndarray, means: np.ndarray, stds: np.ndarray, alpha: float
) -> np.ndarray:
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    return np.abs(truths - means) <= z_crit * stds


def _covered_fraction(
    s_true: np.ndarray, s_post: np.ndarray, post_std: np.ndarray, alpha: float
) -> float:
    return float(np.mean(_covered_fraction_mask(s_true, s_post, post_std, alpha)))


def fixed_truth_expected_Q(
    A: np.ndarray,
    C_post: np.ndarray,
    sigma_infer2: float,
    sigma_true2: float,
    s_true: np.ndarray,
    s0: np.ndarray,
) -> dict:
    """Theoretical E[Q] for Experiment III's fixed-truth setting (see
    `ARCHITECTURE.md`'s "Experiment III: investigating the boundary/interior
    coverage reversal" section for the derivation):

        E[Q] = tr(C_post^-1 V_post) + b^T C_post^-1 b,

    where `K = C_post A^T / sigma_infer2` (the Kalman-gain identity `K =
    Cs_infer A^T (A Cs_infer A^T + Sigma_d_infer)^-1 = C_post A^T
    Sigma_d_infer^-1`, using the cheaper right-hand form), `b = (I - KA)(s_true
    - s0)` is the deterministic fixed-truth bias, and `V_post = sigma_true2 *
    K K^T` is the repeated-noise sampling covariance of `s_post` (using the
    *true* noise covariance `sigma_true2 * I`, since that governs the actual
    realized noise -- `K` itself is built from the *inference* noise model
    `sigma_infer2 * I`, matching how `C_post` was computed).

    `C_post`, `sigma_infer2` must be the same ones used to compute the
    fixed-truth posterior (e.g. via `inversion.posterior_isotropic`); this
    function performs no inversion of its own beyond Cholesky solves against
    `C_post`.

    Returns a dict with keys ``"trace"`` (the noise-driven term,
    ``tr(C_post^-1 V_post)``, identical for any `s_true` sharing this `A`/
    `C_post`/noise), ``"bias"`` (`b^T C_post^-1 b`, the only term depending on
    `s_true`), and ``"theory"`` (their sum, the theoretical `E[Q]`).
    """
    n_cells = A.shape[1]
    K = (C_post @ A.T) / sigma_infer2
    b = (np.eye(n_cells) - K @ A) @ (s_true - s0)
    V_post = sigma_true2 * (K @ K.T)

    C_post_factor = cho_factor(C_post, lower=True)
    trace_term = float(np.trace(cho_solve(C_post_factor, V_post)))
    bias_term = float(b @ cho_solve(C_post_factor, b))

    return {"trace": trace_term, "bias": bias_term, "theory": trace_term + bias_term}
