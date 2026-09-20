"""Closed-form Gaussian posterior for the linear-Gaussian inverse problem.

    C_post = (A^T Sigma_d^-1 A + Cs^-1)^-1
    s_post = C_post (A^T Sigma_d^-1 d + Cs^-1 s0)

Neither `Cs^-1` nor `H^-1` (where `H = A^T Sigma_d^-1 A + Cs^-1`) is ever
obtained via a generic matrix inversion (e.g. `np.linalg.inv`). `Cs^-1` is
applied via a Cholesky-based solve against `Cs`; `H` is assembled once and
factored via a single Cholesky decomposition, and `s_post` is obtained from
that factorization via triangular solves. `C_post` is formed explicitly only
because it is itself a required output of these functions (e.g. for
`diag(C_post)` or as a diagnostic) -- solving `H s = b` for `s_post` alone
never requires it.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def posterior_general(
    A: np.ndarray,
    d: np.ndarray,
    Sigma_d: np.ndarray,
    Cs: np.ndarray,
    s0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """General-noise-covariance Gaussian posterior.

    Implements the general form C_post = (A^T Sigma_d^-1 A + Cs^-1)^-1,
    s_post = C_post (A^T Sigma_d^-1 d + Cs^-1 s0), for an arbitrary
    (symmetric positive-definite) observation covariance `Sigma_d`.
    """
    n = A.shape[1]

    Cs_factor = cho_factor(Cs, lower=True)
    Cs_inv = cho_solve(Cs_factor, np.eye(n))

    Sigma_d_factor = cho_factor(Sigma_d, lower=True)
    Sigma_d_inv_A = cho_solve(Sigma_d_factor, A)
    Sigma_d_inv_d = cho_solve(Sigma_d_factor, d)

    H = A.T @ Sigma_d_inv_A + Cs_inv
    b = A.T @ Sigma_d_inv_d + Cs_inv @ s0

    H_factor = cho_factor(H, lower=True)
    s_post = cho_solve(H_factor, b)
    C_post = cho_solve(H_factor, np.eye(n))

    return s_post, C_post


def posterior_isotropic(
    A: np.ndarray,
    d: np.ndarray,
    sigma2: float,
    Cs: np.ndarray,
    s0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian posterior for the independent-noise case Sigma_d = sigma2 * I.

    This is the special case used throughout the actual experiments. It
    avoids ever forming the (m x m) matrix `Sigma_d` explicitly.
    """
    n = A.shape[1]

    Cs_factor = cho_factor(Cs, lower=True)
    Cs_inv = cho_solve(Cs_factor, np.eye(n))

    H = (A.T @ A) / sigma2 + Cs_inv
    b = (A.T @ d) / sigma2 + Cs_inv @ s0

    H_factor = cho_factor(H, lower=True)
    s_post = cho_solve(H_factor, b)
    C_post = cho_solve(H_factor, np.eye(n))

    return s_post, C_post
