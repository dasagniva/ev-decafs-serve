"""Pydantic v2 request/response schemas for the serving API.

Strict by design (``extra="forbid"``, finite-only values, validated timestamps) so malformed
input is rejected with a structured 422 long before it reaches the detector.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A float that rejects NaN / +-inf at parse time (-> 422 with a structured error).
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

_LABEL_NAMES = {0: "recoiled", 1: "sustained"}


class DetectRequest(BaseModel):
    """A single univariate time series to analyse."""

    model_config = ConfigDict(extra="forbid")

    series: list[FiniteFloat] = Field(
        ...,
        min_length=1,
        description="Univariate time series values (finite floats). Length bounds enforced "
        "against server config.",
    )
    timestamps: list[FiniteFloat] | None = Field(
        default=None,
        description="Optional timestamps, same length as `series`, strictly increasing.",
    )

    @model_validator(mode="after")
    def _check_timestamps(self) -> DetectRequest:
        if self.timestamps is None:
            return self
        if len(self.timestamps) != len(self.series):
            raise ValueError("`timestamps` must have the same length as `series`.")
        ts = self.timestamps
        if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
            raise ValueError("`timestamps` must be strictly increasing.")
        return self


class ChangePoint(BaseModel):
    """One detected changepoint and its classification."""

    index: int = Field(..., description="Position in the input series.")
    label: int = Field(..., description="0 = recoiled (transient), 1 = sustained (regime shift).")
    label_name: str = Field(..., description="Human-readable label.")
    prob_recoiled: float = Field(..., description="P(recoiled).")
    prob_sustained: float = Field(..., description="P(sustained).")
    uncertainty: float = Field(
        ...,
        description="Classification margin 1 - max(prob). NOT a changepoint-location interval "
        "(no such method exists in the model).",
    )

    @classmethod
    def from_inference(
        cls, index: int, label: int, proba: list[float], uncertainty: float
    ) -> ChangePoint:
        # proba is ordered by FourierPNN.classes_ == [0, 1] -> [P(recoiled), P(sustained)].
        return cls(
            index=index,
            label=label,
            label_name=_LABEL_NAMES.get(label, str(label)),
            prob_recoiled=float(proba[0]),
            prob_sustained=float(proba[1]),
            uncertainty=float(uncertainty),
        )


class DetectResponse(BaseModel):
    """Detection + classification result for one series."""

    changepoints: list[ChangePoint]
    n_changepoints: int
    series_length: int
    model_version: str


class HealthResponse(BaseModel):
    """Liveness + which model version is loaded."""

    status: str
    model_version: str
    model_name: str


class ModelInfoResponse(BaseModel):
    """Registry metadata + frozen detector parameters of the served model."""

    name: str
    version: str
    run_id: str | None
    aliases: list[str]
    dataset: str
    feature_names: list[str]
    n_grid: int
    alpha_0: float
    phi: float
    window_halfwidth_w: int
    gpd_percentile_q0: float
    window_L: int


class ColumnDriftOut(BaseModel):
    """Per-column drift detail."""

    column: str
    p_value: float
    threshold: float
    drifted: bool


class DriftResponse(BaseModel):
    """Input-drift verdict for live traffic vs the training reference profile."""

    status: str = Field(..., description="'ok', 'insufficient_data', or 'no_reference'.")
    dataset_drift: bool | None = None
    share_drifted: float | None = None
    n_drifted_columns: int | None = None
    n_columns: int | None = None
    drift_share_threshold: float | None = None
    n_reference_windows: int = 0
    n_current_windows: int = 0
    columns: list[ColumnDriftOut] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Structured error body (never a stack trace)."""

    error: str
    detail: str | None = None
