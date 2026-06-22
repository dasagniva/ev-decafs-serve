"""Changepoint labelling for Phase II training (training-only).

Ported from changepoint-evdecafs ``src/phase2/labelling.py`` (``compute_kappa_mu``,
``label_changepoints`` — Algorithm 3, the feature-based labeller) and
``src/phase2/bocpd_labeller.py`` (``label_with_bocpd``, ``refine_pending_labels`` — the v4.2
primary BOCPD-oracle labeller) — algorithms unchanged.

TRAINING ONLY. Labels exist to fit the FPNN; serving never re-labels (see INTAKE.md §8). The
research repo's ``relabel_with_hypersensitive``/CUSUM/self-supervised variants are not ported:
they fed diagnostic columns that the FPNN was never fit on.
"""

from __future__ import annotations

import numpy as np

from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger(__name__)


def compute_kappa_mu(X: np.ndarray, percentile: float = 75) -> float:
    """Compute the magnitude threshold ``kappa_mu`` = percentile of ``|delta_mu|``.

    Parameters
    ----------
    X:
        Feature matrix, shape ``(m, >=1)``; column 0 is ``delta_mu``.
    percentile:
        Percentile of ``|delta_mu|`` to use as the threshold.
    """
    X = np.asarray(X, dtype=float)
    kappa_mu = float(np.percentile(np.abs(X[:, 0]), percentile))
    logger.debug("kappa_mu (%.0fth pct of |delta_mu|) = %.4f", percentile, kappa_mu)
    return kappa_mu


def label_changepoints(X: np.ndarray, kappa_mu: float, kappa_S: float = 0.5) -> np.ndarray:
    """Assign binary labels (Algorithm 3, feature-based).

    A changepoint is **sustained** (1) iff ``|delta_mu| > kappa_mu AND S > kappa_S``; otherwise
    **recoiled** (0).

    Parameters
    ----------
    X:
        Feature matrix, shape ``(m, >=2)``; columns ``[delta_mu, S, ...]``.
    kappa_mu:
        Magnitude threshold (see :func:`compute_kappa_mu`).
    kappa_S:
        Persistence threshold.

    Returns
    -------
    labels : np.ndarray of int, shape ``(m,)``  (0 = recoiled, 1 = sustained)
    """
    X = np.asarray(X, dtype=float)
    sustained = (np.abs(X[:, 0]) > kappa_mu) & (X[:, 1] > kappa_S)
    labels = sustained.astype(int)
    logger.info(
        "Labelling (feature-based) — %d sustained, %d recoiled (kappa_mu=%.4f, kappa_S=%.4f)",
        int(labels.sum()),
        int((labels == 0).sum()),
        kappa_mu,
        kappa_S,
    )
    return labels


def label_with_bocpd(
    decafs_cps: np.ndarray,
    bocpd_cps: np.ndarray,
    true_cps: np.ndarray,
    tolerance: int,
    has_ground_truth: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Label DeCAFS changepoints by cross-referencing BOCPD output and ground truth.

    For each DeCAFS CP ``tau`` (with ground truth):
      - near a TRUE CP (within ``tolerance``)        -> Sustained (1)
      - near a BOCPD CP but not a true CP            -> Recoiled (0)
      - near neither                                 -> Recoiled (0)

    Without ground truth: near a BOCPD CP -> ``-1`` (pending, caller refines); else Recoiled (0).

    Returns
    -------
    labels : np.ndarray of int  (values in {0, 1, -1})
    reasons : list[str]
    """
    decafs_cps = np.asarray(decafs_cps, dtype=int)
    bocpd_cps = np.asarray(bocpd_cps, dtype=int)
    true_cps = np.asarray(true_cps, dtype=int)

    labels = np.zeros(len(decafs_cps), dtype=int)
    reasons: list[str] = []

    for i, tau in enumerate(decafs_cps):
        near_true = (
            has_ground_truth and len(true_cps) > 0 and np.min(np.abs(true_cps - tau)) <= tolerance
        )
        near_bocpd = len(bocpd_cps) > 0 and np.min(np.abs(bocpd_cps - tau)) <= tolerance

        if has_ground_truth:
            if near_true:
                labels[i] = 1
                reasons.append("sustained: near true CP")
            elif near_bocpd:
                labels[i] = 0
                reasons.append("recoiled: BOCPD-only detection")
            else:
                labels[i] = 0
                reasons.append("recoiled: unconfirmed by BOCPD or ground truth")
        else:
            if near_bocpd:
                labels[i] = -1
                reasons.append("bocpd-confirmed: pending feature check")
            else:
                labels[i] = 0
                reasons.append("recoiled: unconfirmed by BOCPD")

    logger.info(
        "BOCPD labelling — %d sustained, %d recoiled, %d pending (from %d DeCAFS CPs)",
        int(np.sum(labels == 1)),
        int(np.sum(labels == 0)),
        int(np.sum(labels == -1)),
        len(decafs_cps),
    )
    return labels, reasons


def refine_pending_labels(
    labels: np.ndarray,
    features: np.ndarray,
    kappa_mu: float,
    kappa_S: float,
) -> np.ndarray:
    """Resolve ``-1`` (pending) labels via the Algorithm-3 heuristic.

    A pending CP becomes Sustained (1) iff ``|delta_mu| > kappa_mu AND S > kappa_S``, else
    Recoiled (0). Used for datasets without ground truth.
    """
    refined = np.asarray(labels, dtype=int).copy()
    features = np.asarray(features, dtype=float)
    for i in range(len(refined)):
        if refined[i] == -1:
            delta_mu = abs(float(features[i, 0]))
            S = float(features[i, 1])
            refined[i] = 1 if (delta_mu > kappa_mu and S > kappa_S) else 0
    return refined
