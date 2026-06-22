"""Unit tests for input-drift monitoring (pure functions, no API)."""

from __future__ import annotations

import numpy as np

from evdecafs_serve.monitoring.drift import run_drift, window_feature_frame
from evdecafs_serve.monitoring.features import (
    WINDOW_FEATURE_COLUMNS,
    series_to_window_features,
)


def test_window_features_shape_and_columns():
    series = np.arange(100, dtype=float)
    rows = series_to_window_features(series, window=20, step=10)
    assert rows.shape[1] == 3
    assert rows.shape[0] == (100 - 20) // 10 + 1
    frame = window_feature_frame(rows)
    assert list(frame.columns) == WINDOW_FEATURE_COLUMNS


def test_window_features_handles_series_shorter_than_window():
    rows = series_to_window_features(np.ones(5), window=50, step=5)
    assert rows.shape == (1, 3)


def _ref_and(current_series, rng_seed_base=0):
    rng = np.random.default_rng(rng_seed_base)
    base = rng.normal(100_000, 2_000, 4_000)
    ref = window_feature_frame(series_to_window_features(base, 30, 5))
    cur = window_feature_frame(series_to_window_features(current_series, 30, 5))
    return ref, cur


def test_drift_quiet_on_in_distribution():
    rng = np.random.default_rng(99)
    ref, cur = _ref_and(rng.normal(100_000, 2_000, 1_000))
    result, _ = run_drift(ref, cur)
    assert result.dataset_drift is False


def test_drift_fires_on_mean_shift():
    rng = np.random.default_rng(7)
    ref, cur = _ref_and(rng.normal(140_000, 2_000, 1_000))  # +40k mean shift
    result, _ = run_drift(ref, cur)
    # A mean shift always moves the "mean" column; dataset-level drift depends on the share
    # threshold, so assert on the column that must change.
    assert any(c.column == "mean" and c.drifted for c in result.columns)


def test_drift_fires_on_variance_shift():
    rng = np.random.default_rng(11)
    ref, cur = _ref_and(rng.normal(100_000, 9_000, 1_000))  # 4.5x variance
    result, _ = run_drift(ref, cur)
    assert result.dataset_drift is True
    assert any(c.column == "std" and c.drifted for c in result.columns)
