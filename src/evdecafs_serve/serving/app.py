"""FastAPI serving layer for the EV-DeCAFS champion model.

Endpoints:
- ``POST /detect`` — detect + classify changepoints in a single series.
- ``GET  /health`` — liveness + loaded model version.
- ``GET  /model``  — registry metadata + frozen detector params of the served model.

The champion model is loaded **once at startup** from ``models:/ev-decafs@champion`` (configurable
via ``EVDECAFS_*`` env vars). If it cannot be loaded, startup fails fast. The request path only
ever calls :func:`detect_and_classify` — no training/eval code (CLAUDE.md extraction rule #3).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from evdecafs_serve.config import Settings, get_settings
from evdecafs_serve.models.bundle import ModelBundle
from evdecafs_serve.models.inference import detect_and_classify
from evdecafs_serve.models.registry import load_champion_bundle
from evdecafs_serve.monitoring.drift import run_drift
from evdecafs_serve.monitoring.features import series_to_window_features, window_feature_frame
from evdecafs_serve.serving.schemas import (
    ChangePoint,
    ColumnDriftOut,
    DetectRequest,
    DetectResponse,
    DriftResponse,
    HealthResponse,
    ModelInfoResponse,
)
from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger("serving")


class _ServingState:
    """Process-wide handle to the loaded model, metadata, and drift-monitoring buffer."""

    settings: Settings
    bundle: ModelBundle | None = None
    metadata: dict[str, Any] = {}
    reference_frame: pd.DataFrame | None = None  # training reference for drift
    monitor_buffer: list[list[float]] = []  # window features of recent /detect inputs


state = _ServingState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the champion model once at startup; fail fast if unavailable."""
    state.settings = get_settings()
    try:
        bundle, metadata = load_champion_bundle(state.settings)
    except Exception as exc:  # noqa: BLE001 — surface any load failure as a fatal startup error
        raise RuntimeError(
            f"Failed to load champion model "
            f"'{state.settings.registered_model_name}@{state.settings.champion_alias}' "
            f"from {state.settings.mlflow_tracking_uri}: {exc}"
        ) from exc
    state.bundle = bundle
    state.metadata = metadata
    state.monitor_buffer = []
    if bundle.reference_windows:
        state.reference_frame = window_feature_frame(np.asarray(bundle.reference_windows))
    else:
        state.reference_frame = None
    logger.info(
        "Loaded champion %s v%s (drift reference: %d windows)",
        metadata["name"],
        metadata["version"],
        0 if state.reference_frame is None else len(state.reference_frame),
    )
    yield
    state.bundle = None


app = FastAPI(
    title="EV-DeCAFS detection API",
    version="0.1.0",
    summary="Two-phase changepoint detection + classification for noisy, autocorrelated series.",
    lifespan=lifespan,
)


def _require_bundle() -> ModelBundle:
    if state.bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return state.bundle


@app.exception_handler(RequestValidationError)
async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Structured 422 that omits the raw input (which may be NaN/inf and unserialisable)."""
    detail = [
        {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": detail})


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: structured JSON, never a stack trace."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An unexpected error occurred."},
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    md = state.metadata
    return HealthResponse(
        status="ok" if state.bundle is not None else "degraded",
        model_version=str(md.get("version", "unknown")),
        model_name=str(md.get("name", "unknown")),
    )


@app.get("/model", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    bundle = _require_bundle()
    md = state.metadata
    return ModelInfoResponse(
        name=str(md["name"]),
        version=str(md["version"]),
        run_id=md.get("run_id"),
        aliases=list(md.get("aliases", [])),
        dataset=bundle.dataset,
        feature_names=bundle.feature_names,
        n_grid=bundle.n_grid,
        alpha_0=bundle.alpha_0,
        phi=bundle.phi,
        window_halfwidth_w=bundle.window_halfwidth_w,
        gpd_percentile_q0=bundle.gpd_percentile_q0,
        window_L=bundle.window_L,
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest) -> DetectResponse:
    bundle = _require_bundle()

    n = len(req.series)
    lo, hi = state.settings.min_series_length, state.settings.max_series_length
    if not (lo <= n <= hi):
        raise HTTPException(
            status_code=422,
            detail=f"series length {n} is outside the allowed range [{lo}, {hi}].",
        )

    series = np.asarray(req.series, dtype=float)

    # Record window features of this input for live drift monitoring (ring buffer).
    if bundle.reference_windows:
        rows = series_to_window_features(series, bundle.monitor_window, bundle.monitor_step)
        state.monitor_buffer.extend(rows.tolist())
        overflow = len(state.monitor_buffer) - state.settings.drift_buffer_max_rows
        if overflow > 0:
            del state.monitor_buffer[:overflow]

    result = detect_and_classify(series, bundle)
    changepoints = [
        ChangePoint.from_inference(idx, label, proba, unc)
        for idx, label, proba, unc in zip(
            result.changepoints,
            result.segment_labels,
            result.probabilities,
            result.uncertainty,
            strict=True,
        )
    ]
    return DetectResponse(
        changepoints=changepoints,
        n_changepoints=len(changepoints),
        series_length=n,
        model_version=result.model_version,
    )


@app.get("/monitoring/drift", response_model=DriftResponse)
async def monitoring_drift() -> DriftResponse:
    """Compare window features of recent /detect traffic to the training reference profile.

    Returns ``status="no_reference"`` if the model carries no reference, ``"insufficient_data"``
    until enough traffic has accumulated, else ``"ok"`` with the drift verdict.
    """
    _require_bundle()
    ref = state.reference_frame
    n_ref = 0 if ref is None else len(ref)
    n_cur = len(state.monitor_buffer)

    if ref is None or n_ref == 0:
        return DriftResponse(status="no_reference", n_reference_windows=0, n_current_windows=n_cur)

    if n_cur < state.settings.drift_min_current_windows:
        return DriftResponse(
            status="insufficient_data",
            n_reference_windows=n_ref,
            n_current_windows=n_cur,
        )

    bundle = _require_bundle()
    current = window_feature_frame(np.asarray(state.monitor_buffer))
    result, _ = run_drift(ref, current, drift_share=bundle.drift_share_threshold)
    return DriftResponse(
        status="ok",
        dataset_drift=result.dataset_drift,
        share_drifted=result.share_drifted,
        n_drifted_columns=result.n_drifted_columns,
        n_columns=result.n_columns,
        drift_share_threshold=result.drift_share_threshold,
        n_reference_windows=n_ref,
        n_current_windows=n_cur,
        columns=[
            ColumnDriftOut(
                column=c.column, p_value=c.p_value, threshold=c.threshold, drifted=c.drifted
            )
            for c in result.columns
        ],
    )
