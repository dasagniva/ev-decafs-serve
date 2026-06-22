"""The versioned model artifact: everything serving needs to run the inference contract.

Per INTAKE.md §3's corollary, "the model" is **not** just the ``FourierPNN``. Test/serve time
reuses the AR(1)-derived ``phi``/``lambda``/``gamma`` and the tuned ``alpha_0`` from training
unmodified — only the EVI field is recomputed per series. So the registered artifact bundles the
classifier together with all of those frozen parameters as one unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evdecafs_serve.models.fpnn import FourierPNN


@dataclass
class ModelBundle:
    """Frozen, versioned EV-DeCAFS model: detector params + fitted classifier.

    Attributes
    ----------
    fpnn:
        The fitted :class:`FourierPNN` classifier.
    phi, sigma_v_sq, sigma_eta_sq:
        AR(1) parameters estimated on the training series (frozen at train time).
    alpha_0:
        The flat Phase-I penalty actually used at train time (BIC-tuned or fixed).
    n_grid:
        ``ev_decafs`` mu-grid resolution (must match training — see DECISIONS.md #3).
    window_halfwidth_w, gpd_percentile_q0:
        EVI-field parameters (recomputed per series at inference, but with these settings).
    window_L:
        Feature-extraction window half-width.
    feature_names:
        Column labels of the feature matrix the FPNN was fit on.
    dataset:
        Name of the dataset the model was trained on (provenance).
    version:
        Free-form version string (set by the registry on logging).
    """

    fpnn: FourierPNN
    phi: float
    sigma_v_sq: float
    sigma_eta_sq: float
    alpha_0: float
    n_grid: int
    window_halfwidth_w: int
    gpd_percentile_q0: float
    window_L: int
    feature_names: list[str] = field(default_factory=list)
    dataset: str = ""
    version: str = "unversioned"

    # Drift-monitoring reference profile (sliding-window summary features of the training
    # series). Empty when monitoring is disabled / not yet computed.
    reference_windows: list[list[float]] = field(default_factory=list)
    reference_columns: list[str] = field(default_factory=list)
    monitor_window: int = 30
    monitor_step: int = 5
    drift_share_threshold: float = 0.5

    @property
    def lambda_param(self) -> float:
        """Level-process precision ``1 / sigma_eta^2`` (matches training)."""
        return 1.0 / (self.sigma_eta_sq + 1e-12)

    @property
    def gamma(self) -> float:
        """Observation-noise precision ``1 / sigma_v^2`` (matches training)."""
        return 1.0 / (self.sigma_v_sq + 1e-12)
