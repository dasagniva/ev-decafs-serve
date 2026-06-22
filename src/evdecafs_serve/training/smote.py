"""Minority-class oversampling for Phase II training (training-only).

A small, dependency-free, seeded SMOTE (Chawla et al. 2002): for each synthetic sample, pick a
random minority point, pick one of its ``k`` nearest minority neighbours, and interpolate at a
random fraction along the segment between them. See DECISIONS.md #6 for why this is hand-rolled
rather than pulling in ``imbalanced-learn`` (we no longer chase bit-for-bit research numerics,
and SMOTE never appears in the inference path, so it has zero serving-scalability impact).

TRAINING ONLY — serving never balances classes.
"""

from __future__ import annotations

import numpy as np

from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger(__name__)


def balance_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample every non-majority class up to the majority count via SMOTE.

    Edge cases mirror the research repo's guard: with a single class present, or a minority
    class too small for even one neighbour, the data is returned unchanged. ``k_neighbors`` is
    capped at ``n_minority - 1``.

    Parameters
    ----------
    X_train:
        Feature matrix, shape ``(m, d)``.
    y_train:
        Integer class labels, shape ``(m,)``.
    k_neighbors:
        Number of nearest minority neighbours to interpolate among.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    X_resampled, y_resampled
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)

    classes, counts = np.unique(y_train, return_counts=True)
    before = dict(zip(classes.tolist(), counts.tolist(), strict=True))
    logger.info("Class distribution before SMOTE: %s", before)

    if len(classes) < 2:
        logger.warning("Only one class present (%s); skipping SMOTE.", classes.tolist())
        return X_train.copy(), y_train.copy()

    rng = np.random.default_rng(random_state)
    majority_count = int(counts.max())

    X_parts = [X_train]
    y_parts = [y_train]

    for cls, cnt in zip(classes, counts, strict=True):
        n_needed = majority_count - int(cnt)
        if n_needed <= 0:
            continue

        X_cls = X_train[y_train == cls]
        n_minority = len(X_cls)
        effective_k = min(k_neighbors, n_minority - 1)
        if effective_k < 1:
            logger.warning(
                "Class %s has only %d sample(s); cannot SMOTE — leaving it unbalanced.",
                int(cls),
                n_minority,
            )
            continue
        if effective_k < k_neighbors:
            logger.warning(
                "Reduced k_neighbors from %d to %d for class %s (size=%d).",
                k_neighbors,
                effective_k,
                int(cls),
                n_minority,
            )

        synthetic = _smote_samples(X_cls, n_needed, effective_k, rng)
        X_parts.append(synthetic)
        y_parts.append(np.full(len(synthetic), cls, dtype=int))

    X_res = np.vstack(X_parts)
    y_res = np.concatenate(y_parts)

    classes_after, counts_after = np.unique(y_res, return_counts=True)
    logger.info(
        "Class distribution after SMOTE:  %s",
        dict(zip(classes_after.tolist(), counts_after.tolist(), strict=True)),
    )
    return X_res, y_res


def _smote_samples(
    X_cls: np.ndarray,
    n_samples: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate ``n_samples`` synthetic points within one class via kNN interpolation."""
    n = len(X_cls)

    # Pairwise distances among the (few) minority points; k nearest excluding self.
    diff = X_cls[:, None, :] - X_cls[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    neighbours = np.argsort(dist, axis=1)[:, :k]  # (n, k)

    base_idx = rng.integers(0, n, size=n_samples)
    nbr_choice = rng.integers(0, k, size=n_samples)
    gap = rng.random(size=(n_samples, 1))

    anchors = X_cls[base_idx]
    partners = X_cls[neighbours[base_idx, nbr_choice]]
    return anchors + gap * (partners - anchors)
