"""The EV-DeCAFS inference contract.

This is the *only* path that runs at serve time, exactly as fixed in INTAKE.md §8:

    compute_evi_field -> ev_decafs (flat penalty) -> extract_features -> FourierPNN.predict_proba

It deliberately imports nothing from ``training/`` — no BOCPD, no SMOTE, no Monte-Carlo or
evaluation code (CLAUDE.md extraction rule #3). Phase 3's FastAPI layer will wrap this function;
the MLflow pyfunc wrapper (``models/registry.py``) already calls it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evdecafs_serve.features.evt import compute_evi_field
from evdecafs_serve.features.extract import extract_features
from evdecafs_serve.models.bundle import ModelBundle
from evdecafs_serve.models.decafs import ev_decafs


@dataclass
class DetectionResult:
    """Output of :func:`detect_and_classify`.

    ``uncertainty`` is the per-changepoint *classification* margin ``1 - max(proba)`` — NOT a
    changepoint-location confidence interval. No location-CI method exists in the research code
    (INTAKE.md §8); the serving layer must not invent one.
    """

    changepoints: list[int]
    segment_labels: list[int]
    probabilities: list[list[float]]
    uncertainty: list[float]
    model_version: str


def detect_and_classify(series: np.ndarray, model: ModelBundle) -> DetectionResult:
    """Detect changepoints in ``series`` and classify each as sustained/recoiled.

    Parameters
    ----------
    series:
        Univariate time series, shape ``(n,)``.
    model:
        A fitted :class:`ModelBundle` (the registered artifact).

    Returns
    -------
    DetectionResult
    """
    series = np.asarray(series, dtype=float)

    xi_field = compute_evi_field(series, w=model.window_halfwidth_w, q0=model.gpd_percentile_q0)
    alpha_t = np.full(len(series), model.alpha_0)  # flat penalty — confirmed INTAKE.md §1
    res = ev_decafs(
        series,
        alpha_t,
        lambda_param=model.lambda_param,
        gamma=model.gamma,
        phi=model.phi,
        n_grid=model.n_grid,
    )

    cps = res["changepoints"]
    if len(cps) == 0:
        return DetectionResult(
            changepoints=[],
            segment_labels=[],
            probabilities=[],
            uncertainty=[],
            model_version=model.version,
        )

    X, _ = extract_features(series, cps, res["means"], L=model.window_L, xi_field=xi_field)
    proba = model.fpnn.predict_proba(X)  # (m, 2)
    labels = model.fpnn.classes_[np.argmax(proba, axis=1)]

    return DetectionResult(
        changepoints=cps.tolist(),
        segment_labels=labels.astype(int).tolist(),
        probabilities=proba.tolist(),
        uncertainty=(1.0 - proba.max(axis=1)).tolist(),
        model_version=model.version,
    )
