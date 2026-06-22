"""BOCPD labelling oracle (training-only).

Ported from changepoint-evdecafs/src/phase1/hypersensitive_cpd.py (``run_bocpd`` and its
helpers) — algorithm unchanged. Bayesian Online Change Point Detection (Adams & MacKay 2007)
with a Normal-Gamma conjugate prior, run on the AR(1)-prewhitened series, used as a
deliberately hypersensitive detector to generate Phase-II training labels.

TRAINING ONLY. Per CLAUDE.md's extraction rules and INTAKE.md §8, ``serving/`` must never import
this module — at inference time labels come from ``FourierPNN.predict_proba``, not BOCPD.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger(__name__)


def _log_student_t(
    x: float,
    df: np.ndarray,
    mu: np.ndarray,
    scale_sq: np.ndarray,
) -> np.ndarray:
    """Log pdf of a Student-t distribution (vectorised over parameters)."""
    z = (x - mu) ** 2 / (scale_sq + 1e-12)
    return (
        gammaln(0.5 * (df + 1.0))
        - gammaln(0.5 * df)
        - 0.5 * np.log(np.pi * df * scale_sq + 1e-300)
        - 0.5 * (df + 1.0) * np.log1p(z / (df + 1e-12))
    )


def _update_mu(mu_r: np.ndarray, kappa_r: np.ndarray, x: float) -> np.ndarray:
    """Bayesian update for the Normal mean given a new observation."""
    return (kappa_r * mu_r + x) / (kappa_r + 1.0)


def run_bocpd(
    y: np.ndarray,
    phi: float,
    sigma_v: float,
    threshold: float = 0.5,
    *,
    mu0: float | None = None,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float | None = None,
) -> np.ndarray:
    """Run Bayesian Online Changepoint Detection on a univariate series.

    Uses a Gaussian likelihood with Normal-Gamma conjugate prior. AR(1) autocorrelation is
    incorporated by pre-whitening: ``z_t = y_t - phi * y_{t-1}``. Maintains the run-length
    posterior ``P(r_t | y_{1:t})`` with a constant hazard ``H = 1/n``; flags index ``t`` when
    ``P(CP at t) > threshold``.

    Parameters
    ----------
    y:
        Univariate time series, shape ``(n,)``.
    phi:
        AR(1) autocorrelation coefficient (for pre-whitening).
    sigma_v:
        AR(1) innovation standard deviation (kept for API parity; priors are data-driven).
    threshold:
        Posterior-CP probability threshold. Lower = more sensitive.
    mu0, kappa0, alpha0, beta0:
        Normal-Gamma prior hyperparameters. ``mu0`` defaults to ``mean(z)``; ``beta0`` to
        ``var(z)/2``.

    Returns
    -------
    bocpd_flags : np.ndarray of bool, shape ``(n,)``
        True where ``P(CP) > threshold``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 2:
        return np.zeros(n, dtype=bool)

    # Pre-whiten: remove AR(1) autocorrelation
    z = np.empty(n)
    z[0] = y[0]
    z[1:] = y[1:] - phi * y[:-1]

    if mu0 is None:
        mu0 = float(np.mean(z))
    if beta0 is None:
        beta0 = max(float(np.var(z)) / 2.0, 1e-6)

    H = 1.0 / n  # constant hazard (geometric run-length prior)

    # Run-length distribution and Normal-Gamma sufficient statistics per run length.
    R = np.array([1.0])
    kappa_r = np.array([kappa0])
    mu_r = np.array([mu0])
    alpha_r = np.array([alpha0])
    beta_r = np.array([beta0])

    cp_probs = np.zeros(n)

    for t in range(1, n):
        x = z[t]

        df = 2.0 * alpha_r
        scale_sq = beta_r * (kappa_r + 1.0) / (alpha_r * kappa_r + 1e-12)
        scale_sq = np.maximum(scale_sq, 1e-12)

        log_pred = _log_student_t(x, df, mu_r, scale_sq)
        log_pred_shifted = log_pred - log_pred.max()
        pred = np.exp(log_pred_shifted)

        # Prior predictive for the new run (CP case), on a comparable log scale.
        df0 = 2.0 * alpha0
        scale_sq0 = max(float(beta0 * (kappa0 + 1.0) / (alpha0 * kappa0 + 1e-12)), 1e-12)
        log_pred_prior = float(
            _log_student_t(x, np.array([df0]), np.array([mu0]), np.array([scale_sq0]))[0]
        )
        pred_prior = float(np.exp(log_pred_prior - log_pred.max()))

        R_growth = R * pred * (1.0 - H)
        cp_prob = H * pred_prior
        R_new = np.concatenate([[cp_prob], R_growth])

        Z = R_new.sum()
        if Z > 1e-300:
            R_new = R_new / Z
        else:
            R_new = np.zeros_like(R_new)
            R_new[0] = 1.0

        cp_probs[t] = float(R_new[0])

        # Update Normal-Gamma sufficient stats (beta update uses the *pre-update* mu_r).
        mu_r_new = np.concatenate([[mu0], _update_mu(mu_r, kappa_r, x)])
        kappa_r_new = np.concatenate([[kappa0], kappa_r + 1.0])
        alpha_r_new = np.concatenate([[alpha0], alpha_r + 0.5])
        beta_r_new = np.concatenate(
            [[beta0], beta_r + 0.5 * kappa_r / (kappa_r + 1.0) * (x - mu_r) ** 2]
        )
        R = R_new
        mu_r = mu_r_new
        kappa_r = kappa_r_new
        alpha_r = alpha_r_new
        beta_r = beta_r_new

        # Truncate to keep memory bounded (keep top-probability run lengths).
        if len(R) > 500:
            keep = np.sort(np.argsort(R)[-500:])
            R = R[keep]
            R /= R.sum()
            mu_r = mu_r[keep]
            kappa_r = kappa_r[keep]
            alpha_r = alpha_r[keep]
            beta_r = beta_r[keep]

    bocpd_flags = cp_probs > threshold
    logger.debug("BOCPD — %d flags (threshold=%.2f, n=%d)", int(bocpd_flags.sum()), threshold, n)
    return bocpd_flags
