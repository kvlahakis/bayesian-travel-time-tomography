"""Gaussian prior on the discretized slowness field.

s ~ N(s0, Cs), with Cs a squared-exponential covariance over cell centers:

    (Cs)_jk = tau^2 * exp(-||x_j - x_k||^2 / (2 ell^2)) + eps * delta_jk

where eps = jitter_relative * tau^2 (not an absolute constant -- see
`squared_exponential_cov`).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def squared_exponential_cov(
    cell_centers: np.ndarray,
    tau2: float,
    ell: float,
    jitter_relative: float = 1e-6,
) -> np.ndarray:
    """Squared-exponential prior covariance Cs over cell centers.

    Cs = Cs_raw + (jitter_relative * tau2) * I, matching eps ~ 1e-6 tau^2
    from the PDF exactly -- a *relative* jitter tied to the marginal
    variance, not an absolute `1e-6 * I` (a different quantity, in the wrong
    units). Squared-exponential covariance matrices are frequently
    numerically near-singular at fine grid spacing or large `ell`; this
    jitter is required, not an optional safeguard, to keep downstream
    Cholesky factorizations from intermittently failing.
    """
    sq_dists = cdist(cell_centers, cell_centers, metric="sqeuclidean")
    Cs_raw = tau2 * np.exp(-sq_dists / (2.0 * ell**2))
    n = cell_centers.shape[0]
    jitter = jitter_relative * tau2
    return Cs_raw + jitter * np.eye(n)


def sample_prior(s0: np.ndarray, Cs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw s ~ N(s0, Cs) via a Cholesky factorization of Cs."""
    L = np.linalg.cholesky(Cs)
    z = rng.standard_normal(size=s0.shape[0])
    return s0 + L @ z
