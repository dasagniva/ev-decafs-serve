"""Window-feature extraction for drift monitoring (no Evidently dependency).

Kept separate from ``drift.py`` so the training path can build a reference profile without
importing Evidently. The serving input is a univariate series; we summarise sliding windows as
``[mean, std, range]`` and monitor the distribution of those features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_FEATURE_COLUMNS = ["mean", "std", "range"]


def series_to_window_features(series: np.ndarray, window: int, step: int) -> np.ndarray:
    """Slide a window over ``series``, summarising each as ``[mean, std, range]``.

    Returns shape ``(n_windows, 3)``. If the series is shorter than ``window``, a single window
    over the whole series is used.
    """
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n == 0:
        return np.empty((0, len(WINDOW_FEATURE_COLUMNS)))
    w = min(window, n)
    step = max(1, step)

    rows = []
    for start in range(0, n - w + 1, step):
        seg = s[start : start + w]
        rows.append([float(seg.mean()), float(seg.std()), float(seg.max() - seg.min())])
    if not rows:
        seg = s[:w]
        rows.append([float(seg.mean()), float(seg.std()), float(seg.max() - seg.min())])
    return np.asarray(rows, dtype=float)


def window_feature_frame(rows: np.ndarray) -> pd.DataFrame:
    """Wrap window-feature rows as a DataFrame with the canonical column names."""
    return pd.DataFrame(np.asarray(rows, dtype=float), columns=WINDOW_FEATURE_COLUMNS)
