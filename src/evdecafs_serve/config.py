"""Configuration for the EV-DeCAFS serving project.

Two complementary layers, per the roadmap's "YAML files + pydantic-settings" stack decision:

- :class:`TrainConfig` (and its nested models) is the *pipeline* config — paths, seeds, and
  every hyperparameter for the two-phase pipeline. It is loaded from a YAML file
  (``configs/<name>.yaml``) via :func:`load_config`. Zero hard-coded values live in the
  pipeline code; they all come from here.
- :class:`Settings` is the *runtime/environment* config (MLflow tracking URI, registered-model
  name, artifact location). It is a ``pydantic-settings`` model so these can be overridden by
  environment variables (prefix ``EVDECAFS_``) — important for containers/CI in later phases.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Phase1Config(BaseModel):
    """Phase I (detection) hyperparameters."""

    alpha_0_mode: str = "bic"  # "bic" -> C*log(n); "fixed" -> alpha_0
    bic_multiplier: float = 2.0
    alpha_0: float = 10.0  # used when alpha_0_mode == "fixed"
    tune_bic: bool = False
    bic_sweep_values: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
    expected_n_changepoints: int = 12  # target for the BIC sweep auto-selector
    window_halfwidth_w: int = 50  # EVI field local window half-width
    gpd_percentile_q0: float = 0.90  # EVI exceedance threshold quantile
    n_grid: int = 1000  # mu-grid resolution for ev_decafs (see DECISIONS.md #3)


class LabellingConfig(BaseModel):
    """Phase I -> Phase II labelling hyperparameters."""

    kappa_mu_percentile: float = 75.0
    kappa_S: float = 0.5
    window_L: int = 5  # feature-extraction window half-width
    bocpd_threshold: float = 0.3  # BOCPD oracle posterior-CP threshold
    bocpd_tolerance_fraction: float = 0.02  # match window as fraction of n


class FPNNConfig(BaseModel):
    """FourierPNN classifier hyperparameters."""

    J_harmonics: int = 10
    scaling_range: tuple[float, float] = (-0.5, 0.5)


class SMOTEConfig(BaseModel):
    """Minority-class oversampling (training-only) hyperparameters."""

    enabled: bool = True
    k_neighbors: int = 5
    random_state: int = 42


class MonteCarloConfig(BaseModel):
    """Monte-Carlo evaluation hyperparameters."""

    B: int = 200  # number of synthetic replications
    series_n: int = 2000  # length of each synthetic series
    n_changepoints: int = 8  # sustained changepoints per series
    n_outliers: int = 15  # outlier (recoiled) spikes per series
    phi: float = 0.5
    sigma_v: float = 2000.0
    sigma_eta: float = 100.0
    train_fraction: float = 0.75
    seed: int = 42


class MonitoringConfig(BaseModel):
    """Input-drift monitoring hyperparameters."""

    window: int = 30  # sliding-window length for window-feature summaries
    step: int = 5  # window stride
    drift_share_threshold: float = 0.5  # dataset drift if >= this share of columns drift
    min_current_windows: int = 10  # below this, /monitoring/drift reports insufficient_data


class DatasetConfig(BaseModel):
    """Which dataset to train/evaluate on and where its cache lives."""

    name: str  # "welllog" | "oilwell" | "us_ip_growth"
    cache_path: str  # CSV cache (welllog/oilwell) or cache dir (us_ip_growth)
    train_fraction: float = 0.75
    random_seed: int = 42


class TrainConfig(BaseModel):
    """Top-level pipeline config loaded from ``configs/<name>.yaml``."""

    dataset: DatasetConfig
    phase1: Phase1Config = Field(default_factory=Phase1Config)
    labelling: LabellingConfig = Field(default_factory=LabellingConfig)
    fpnn: FPNNConfig = Field(default_factory=FPNNConfig)
    smote: SMOTEConfig = Field(default_factory=SMOTEConfig)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


def load_config(path: str | Path) -> TrainConfig:
    """Load and validate a :class:`TrainConfig` from a YAML file.

    Parameters
    ----------
    path:
        Path to a ``configs/<name>.yaml`` file.

    Returns
    -------
    TrainConfig
        The fully-validated config (raises ``pydantic.ValidationError`` on a typo or bad type —
        a deliberate improvement over the research repo's unvalidated dict-of-dicts).
    """
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    return TrainConfig.model_validate(raw)


class Settings(BaseSettings):
    """Runtime/environment settings (env-overridable, prefix ``EVDECAFS_``)."""

    model_config = SettingsConfigDict(env_prefix="EVDECAFS_", extra="ignore")

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    registered_model_name: str = "ev-decafs"
    champion_alias: str = "champion"
    experiment_name: str = "ev-decafs"

    # Serving request bounds (reject out-of-range series with HTTP 422).
    min_series_length: int = 10
    max_series_length: int = 50_000

    # Drift monitoring: max window-feature rows buffered from live traffic, and the minimum
    # buffered before /monitoring/drift reports a verdict (vs "insufficient_data").
    drift_buffer_max_rows: int = 5_000
    drift_min_current_windows: int = 10


def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` (reads env vars at call time)."""
    return Settings()
