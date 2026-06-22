"""Dataset dispatch: map a :class:`~evdecafs_serve.config.DatasetConfig` to a training series.

Bridges the per-dataset loaders (which return different shapes) to the single
``(y_train, true_cps_train)`` tuple the training pipeline wants. Shared by ``scripts/train.py``
and ``scripts/bootstrap.py`` so the dispatch lives in exactly one place.
"""

from __future__ import annotations

import numpy as np

from evdecafs_serve.config import TrainConfig
from evdecafs_serve.data.loader import (
    load_oilwell_data,
    load_us_ip_growth,
    load_welllog_data,
)


def load_training_series(cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(y_train, true_cps_train)`` for the configured dataset.

    ``true_cps_train`` is the ground-truth changepoint indices in training coordinates, or an
    empty array when the dataset has none.
    """
    ds = cfg.dataset
    if ds.name == "welllog":
        y_train, _, true_cps, _ = load_welllog_data(
            ds.cache_path, train_fraction=ds.train_fraction, random_seed=ds.random_seed
        )
        n_train = len(y_train)
        return y_train, true_cps[true_cps < n_train]
    if ds.name == "oilwell":
        d = load_oilwell_data(
            ds.cache_path, train_fraction=ds.train_fraction, random_seed=ds.random_seed
        )
        return d["y_train"], d["true_cps_train"]
    if ds.name == "us_ip_growth":
        d = load_us_ip_growth(cache_dir=ds.cache_path)
        return d["y_train"], d["true_cps_train"]
    raise ValueError(f"Unknown dataset '{ds.name}'.")
