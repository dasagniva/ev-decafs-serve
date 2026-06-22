"""Shared fixtures for Phase 2 integration tests (fast configs + synthetic data)."""

from __future__ import annotations

import numpy as np
import pytest

from evdecafs_serve.config import (
    DatasetConfig,
    FPNNConfig,
    LabellingConfig,
    MonteCarloConfig,
    Phase1Config,
    Settings,
    SMOTEConfig,
    TrainConfig,
)
from evdecafs_serve.training.evaluate import generate_synthetic_series


@pytest.fixture
def fast_config() -> TrainConfig:
    """A deliberately small config: trains in well under a second, not minutes."""
    return TrainConfig(
        dataset=DatasetConfig(name="welllog", cache_path="unused", train_fraction=0.7),
        phase1=Phase1Config(
            alpha_0_mode="bic",
            bic_multiplier=1.0,
            tune_bic=False,
            n_grid=80,
            window_halfwidth_w=20,
        ),
        labelling=LabellingConfig(window_L=5),
        fpnn=FPNNConfig(J_harmonics=10),
        smote=SMOTEConfig(enabled=True, k_neighbors=5),
        monte_carlo=MonteCarloConfig(
            B=8, series_n=800, n_changepoints=8, n_outliers=8, train_fraction=0.7, seed=1
        ),
    )


@pytest.fixture
def synthetic_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y_train, sustained_true_cps_train, y_test)`` from one synthetic series."""
    d = generate_synthetic_series(n=300, n_changepoints=4, n_outliers=4, seed=1)
    y = d["y"]
    split = int(len(y) * 0.7)
    pos, lab = d["all_event_positions"], d["all_event_labels"]
    sustained_train = pos[(pos < split) & (lab == 1)]
    return y[:split], sustained_train, y[split:]


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch) -> Settings:
    """MLflow settings backed by a throwaway sqlite DB + mlruns dir inside ``tmp_path``."""
    monkeypatch.chdir(tmp_path)  # keep ./mlruns artifacts inside the temp dir
    return Settings(
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        registered_model_name="ev-decafs-test",
        experiment_name="ev-decafs-test",
    )
