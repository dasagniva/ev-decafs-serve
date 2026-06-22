"""Input-drift monitoring via Evidently (API pinned to the installed 0.7.x surface).

The serving input is a univariate series, so "input drift" is framed as distributional drift of
**sliding-window summary features** (mean, std, range) between a reference set (built from
training data) and current traffic. Evidently runs a per-column K-S test; the dataset is flagged
as drifted when the share of drifted columns meets a threshold (Evidently's default 0.5).

Evidently has had breaking releases — this targets 0.7.x:
``from evidently import Report, Dataset, DataDefinition`` + ``from evidently.presets import
DataDriftPreset``, results read from ``run.dict()["metrics"]`` keyed by each metric's
``config.type``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from evdecafs_serve.monitoring.features import (
    WINDOW_FEATURE_COLUMNS,
    series_to_window_features,
    window_feature_frame,
)

__all__ = [
    "WINDOW_FEATURE_COLUMNS",
    "ColumnDrift",
    "DriftResult",
    "run_drift",
    "save_html",
    "series_to_window_features",
    "window_feature_frame",
]


@dataclass
class ColumnDrift:
    column: str
    p_value: float
    threshold: float
    drifted: bool


@dataclass
class DriftResult:
    """Parsed Evidently data-drift outcome."""

    dataset_drift: bool
    share_drifted: float
    n_drifted_columns: int
    n_columns: int
    drift_share_threshold: float
    columns: list[ColumnDrift]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_drift(
    reference: pd.DataFrame, current: pd.DataFrame, drift_share: float = 0.5
) -> tuple[DriftResult, Report]:
    """Run Evidently's data-drift preset comparing ``current`` to ``reference``.

    Returns the parsed :class:`DriftResult` plus the Evidently run (for HTML export).
    """
    columns = list(reference.columns)
    definition = DataDefinition(numerical_columns=columns)
    ref_ds = Dataset.from_pandas(reference, data_definition=definition)
    cur_ds = Dataset.from_pandas(current, data_definition=definition)

    report = Report([DataDriftPreset()])
    run = report.run(current_data=cur_ds, reference_data=ref_ds)
    return _parse_run(run, drift_share), run


def save_html(run: Report, path: str) -> None:
    """Save the Evidently report as a standalone HTML artifact."""
    run.save_html(path)


def _parse_run(run: Report, drift_share: float) -> DriftResult:
    metrics = run.dict().get("metrics", [])
    count = 0
    share = 0.0
    threshold = drift_share
    columns: list[ColumnDrift] = []

    for m in metrics:
        cfg = m.get("config", {})
        mtype = str(cfg.get("type", ""))
        if mtype.endswith("DriftedColumnsCount"):
            value = m.get("value", {})
            count = int(value.get("count", 0))
            share = float(value.get("share", 0.0))
            threshold = float(cfg.get("drift_share", drift_share))
        elif mtype.endswith("ValueDrift"):
            col = str(cfg.get("column", ""))
            p_value = float(m.get("value", 1.0))
            col_threshold = float(cfg.get("threshold", 0.05))
            columns.append(
                ColumnDrift(
                    column=col,
                    p_value=p_value,
                    threshold=col_threshold,
                    drifted=p_value < col_threshold,
                )
            )

    return DriftResult(
        dataset_drift=share >= threshold,
        share_drifted=share,
        n_drifted_columns=count,
        n_columns=len(columns),
        drift_share_threshold=threshold,
        columns=columns,
    )
