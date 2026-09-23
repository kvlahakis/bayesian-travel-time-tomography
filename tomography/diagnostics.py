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


def pearson_correlation(s_post: np.ndarray, s_true: np.ndarray) -> float:
    """Pearson correlation between the posterior mean and the truth, across
    cells. A reconstruction-quality diagnostic distinct from
    `relative_error`: invariant to a uniform rescaling/offset of `s_post`,
    so it measures whether the reconstruction captures the truth's spatial
    *pattern* rather than its exact magnitude.
    """
    return float(np.corrcoef(s_post, s_true)[0, 1])


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


def posterior_information_metrics(A: np.ndarray, Cs: np.ndarray, sigma2: float) -> dict:
    """Posterior-information-content diagnostics for a given acquisition
    matrix `A` and prior/noise (`Cs`, `sigma2`) -- independent of any
    particular truth or noise realization, since none of these quantities
    depend on the data. Used by Experiment IV to compare predetermined
    acquisition geometries (see `ARCHITECTURE.md`).

    Returns a dict with:

    - ``"J_var"``: mean posterior variance, `tr(C_post) / p`;
    - ``"J_logdet"``: `-log(det(P))` where `P = A^T A / sigma2 + Cs^-1`,
      computed from `P`'s own Cholesky factor `L` (`P = L L^T`) as
      `-2 * sum(log(diag(L)))`, never from `det(P)` directly;
    - ``"rank"``: numerical rank of `A` via `numpy.linalg.matrix_rank`'s
      standard documented default tolerance (`sigma_max * max(m, p) *
      eps(float64)`) -- no project-specific rank-tolerance convention
      exists elsewhere in this codebase;
    - ``"r_eff"``: effective rank `(sum_i sigma_i^2)^2 / sum_i sigma_i^4`
      from `A`'s singular values -- an energy-weighted diagnostic quantity,
      explicitly distinct from the numerical rank above (see
      `ARCHITECTURE.md`'s Experiment IV section for why the two can diverge
      substantially);
    - ``"C_post"``: the posterior covariance itself (`P`'s inverse, obtained
      via Cholesky solves, never an explicit matrix inverse), so callers
      needing `sqrt(diag(C_post))` do not need to recompute it.
    """
    p = A.shape[1]
    Cs_factor = cho_factor(Cs, lower=True)
    Cs_inv = cho_solve(Cs_factor, np.eye(p))
    P = (A.T @ A) / sigma2 + Cs_inv

    P_factor = cho_factor(P, lower=True)
    C_post = cho_solve(P_factor, np.eye(p))
    J_var = float(np.trace(C_post) / p)

    L = np.linalg.cholesky(P)
    J_logdet = float(-2.0 * np.sum(np.log(np.diag(L))))

    S = np.linalg.svd(A, compute_uv=False)
    tol = S.max() * max(A.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(S > tol))
    energy = S**2
    r_eff = float((energy.sum()) ** 2 / np.sum(energy**2))

    return {"J_var": J_var, "J_logdet": J_logdet, "rank": rank, "r_eff": r_eff, "C_post": C_post}
