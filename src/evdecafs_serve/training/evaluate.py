"""Monte-Carlo evaluation of the EV-DeCAFS model (training-only).

Generates ``B`` synthetic series with known sustained changepoints and outlier (recoiled)
spikes, trains a fresh model on each train split, runs the inference contract on the test split,
and reports the distribution of classification metrics with confidence intervals.

Scope per DECISIONS.md #6: the EV-DeCAFS/FourierPNN model only — no comparison baselines. The
synthetic generator and ``assign_nearest_labels`` are ported from changepoint-evdecafs
``src/evaluation/monte_carlo.py`` (unchanged); the per-dataset tail-diagnostic calibration is
intentionally not ported (DECISIONS.md #5 — this repo measures its own model, it does not
reproduce the research repo's 0.795).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

from evdecafs_serve.config import TrainConfig
from evdecafs_serve.models.inference import detect_and_classify
from evdecafs_serve.training.pipeline import train_model
from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger(__name__)

# Metrics aggregated across replications.
METRIC_NAMES = ("balanced_accuracy", "mcc", "auc_roc")


def generate_synthetic_series(
    n: int = 2000,
    n_changepoints: int = 8,
    n_outliers: int = 15,
    phi: float = 0.5,
    sigma_v: float = 2000.0,
    sigma_eta: float = 100.0,
    jump_magnitude_range: tuple[float, float] = (5000, 30000),
    outlier_magnitude_range: tuple[float, float] = (20000, 40000),
    seed: int | None = None,
) -> dict:
    """Generate one synthetic series with known ground truth (ported, unchanged).

    Returns a dict with ``y``, ``all_event_positions`` (sorted union of CPs + outliers),
    ``all_event_labels`` (1=sustained CP, 0=recoiled outlier), and ``true_changepoints``.
    """
    rng = np.random.default_rng(seed)

    mu = np.zeros(n)
    cp_positions = np.sort(rng.choice(range(100, n - 100), size=n_changepoints, replace=False))
    current_mean = float(rng.uniform(70000, 140000))
    mu[: cp_positions[0]] = current_mean
    for i, cp in enumerate(cp_positions):
        jump = float(rng.uniform(*jump_magnitude_range)) * float(rng.choice([-1, 1]))
        current_mean += jump
        next_cp = int(cp_positions[i + 1]) if i + 1 < len(cp_positions) else n
        mu[cp:next_cp] = current_mean

    eta = rng.normal(0, sigma_eta, n)
    eta[0] = 0.0
    mu_with_drift = mu + np.cumsum(eta)

    epsilon = np.zeros(n)
    v = rng.normal(0, sigma_v, n)
    for t in range(1, n):
        epsilon[t] = phi * epsilon[t - 1] + v[t]
    y = mu_with_drift + epsilon

    candidates = [i for i in range(50, n - 50) if not any(abs(i - cp) < 20 for cp in cp_positions)]
    n_outliers_actual = min(n_outliers, len(candidates), n // 20)
    outlier_positions = np.sort(rng.choice(candidates, size=n_outliers_actual, replace=False))
    for op in outlier_positions:
        spike = float(rng.uniform(*outlier_magnitude_range)) * float(rng.choice([-1, 1]))
        y[op] += spike

    all_event_positions = np.concatenate([cp_positions, outlier_positions])
    all_event_labels = np.concatenate(
        [np.ones(len(cp_positions), dtype=int), np.zeros(len(outlier_positions), dtype=int)]
    )
    sort_idx = np.argsort(all_event_positions)

    return {
        "y": y,
        "true_changepoints": cp_positions,
        "all_event_positions": all_event_positions[sort_idx],
        "all_event_labels": all_event_labels[sort_idx],
    }


def assign_nearest_labels(
    detected_cps: np.ndarray,
    true_cps: np.ndarray,
    true_labels: np.ndarray,
    tolerance: int,
) -> np.ndarray:
    """For each detected CP, inherit the nearest true event's label within ``tolerance``.

    Detections with no true event within ``tolerance`` are scored as spurious -> 0 (recoiled).
    Ported unchanged.
    """
    labels = np.zeros(len(detected_cps), dtype=int)
    if len(true_cps) == 0:
        return labels
    for i, dcp in enumerate(detected_cps):
        distances = np.abs(true_cps - dcp)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tolerance:
            labels[i] = int(true_labels[nearest])
    return labels


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Balanced accuracy, MCC, and AUC-ROC for one replication (FPNN only)."""
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    auc = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, np.asarray(y_proba)[:, 1]))
        except ValueError:
            pass
    return {"balanced_accuracy": bal_acc, "mcc": mcc, "auc_roc": auc}


def _one_replication(cfg: TrainConfig, seed: int) -> dict[str, float] | None:
    """Train + evaluate on one synthetic series; ``None`` if the replicate is unusable."""
    mc = cfg.monte_carlo
    data = generate_synthetic_series(
        n=mc.series_n,
        n_changepoints=mc.n_changepoints,
        n_outliers=mc.n_outliers,
        phi=mc.phi,
        sigma_v=mc.sigma_v,
        sigma_eta=mc.sigma_eta,
        seed=seed,
    )
    y = data["y"]
    split = int(len(y) * mc.train_fraction)
    y_train, y_test = y[:split], y[split:]

    pos, lab = data["all_event_positions"], data["all_event_labels"]
    sustained_train = pos[(pos < split) & (lab == 1)]  # ground-truth sustained CPs (train coords)
    test_mask = pos >= split
    true_cps_test = pos[test_mask] - split
    true_labels_test = lab[test_mask]

    try:
        model = train_model(y_train, cfg, true_cps_train=sustained_train)
    except ValueError:
        return None  # too few training changepoints

    result = detect_and_classify(y_test, model)
    if len(result.changepoints) == 0:
        return None

    tol = int(0.02 * len(y_test))
    y_true = assign_nearest_labels(
        np.asarray(result.changepoints), true_cps_test, true_labels_test, tolerance=tol
    )
    if len(np.unique(y_true)) < 2:
        # Single-class test set makes balanced accuracy / AUC degenerate; skip.
        return None

    return _classification_metrics(
        y_true, np.asarray(result.segment_labels), np.asarray(result.probabilities)
    )


def run_evaluation(cfg: TrainConfig) -> dict:
    """Run ``B`` Monte-Carlo replications and aggregate per-metric distributions.

    Returns
    -------
    dict
        ``{"per_metric": {name: {"values", "mean", "std", "ci_lower", "ci_upper", "median"}},
        "n_successful": int, "B": int}``. CIs are the 2.5/97.5 percentiles across replications.
    """
    mc = cfg.monte_carlo
    logger.info("Monte-Carlo evaluation: B=%d, n=%d", mc.B, mc.series_n)

    rows: list[dict[str, float]] = []
    for i in range(mc.B):
        if (i + 1) % 25 == 0:
            logger.info("  replication %d/%d (%d usable)", i + 1, mc.B, len(rows))
        m = _one_replication(cfg, seed=mc.seed + i)
        if m is not None:
            rows.append(m)

    if not rows:
        raise RuntimeError("All Monte-Carlo replications were unusable; check config.")

    per_metric: dict[str, dict] = {}
    for name in METRIC_NAMES:
        values = np.array([r[name] for r in rows], dtype=float)
        per_metric[name] = {
            "values": values,
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "ci_lower": float(np.nanpercentile(values, 2.5)),
            "ci_upper": float(np.nanpercentile(values, 97.5)),
            "median": float(np.nanmedian(values)),
        }
        r = per_metric[name]
        logger.info(
            "  %s: mean=%.4f  CI[%.4f, %.4f]", name, r["mean"], r["ci_lower"], r["ci_upper"]
        )

    return {"per_metric": per_metric, "n_successful": len(rows), "B": mc.B}
