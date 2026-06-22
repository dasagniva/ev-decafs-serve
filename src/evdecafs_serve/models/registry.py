"""MLflow Model Registry I/O for the EV-DeCAFS bundle.

A :class:`ModelBundle` is serialised as an ``mlflow.pyfunc`` model: the whole bundle is pickled
as an artifact, and a thin :class:`EVDeCAFSWrapper` unpickles it in ``load_context`` and runs the
inference contract in ``predict`` (the INTAKE.md §3 spike, promoted to production). Models are
always referenced by registry alias ``models:/<name>@champion`` — never a file path
(roadmap Phase 2).
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from evdecafs_serve.config import Settings, get_settings
from evdecafs_serve.models.bundle import ModelBundle
from evdecafs_serve.models.inference import detect_and_classify
from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger(__name__)

_BUNDLE_ARTIFACT = "bundle"
_BUNDLE_FILENAME = "model_bundle.pkl"


def _to_series(model_input: Any) -> np.ndarray:
    """Coerce a pyfunc ``model_input`` into a single 1-D time series."""
    if isinstance(model_input, pd.DataFrame):
        arr = model_input.to_numpy(dtype=float)
    else:
        arr = np.asarray(model_input, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected a single 1-D time series; got shape {arr.shape}.")
    return arr


class EVDeCAFSWrapper(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper: unpickles a :class:`ModelBundle` and runs the inference contract."""

    def load_context(self, context: Any) -> None:
        path = context.artifacts[_BUNDLE_ARTIFACT]
        with open(path, "rb") as f:
            self.bundle: ModelBundle = pickle.load(f)

    def predict(self, context, model_input, params=None):  # noqa: ANN001, ANN201
        """Run the inference contract on one series; returns a dict of result fields."""
        series = _to_series(model_input)
        result = detect_and_classify(series, self.bundle)
        return {
            "changepoints": result.changepoints,
            "segment_labels": result.segment_labels,
            "probabilities": result.probabilities,
            "uncertainty": result.uncertainty,
            "model_version": result.model_version,
        }


def log_and_register(
    bundle: ModelBundle,
    settings: Settings | None = None,
    *,
    metrics: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
    run_name: str | None = None,
) -> str:
    """Log ``bundle`` as a pyfunc model and register it under the configured name.

    Parameters
    ----------
    bundle:
        The fitted artifact to register.
    settings:
        Runtime settings (tracking URI, model name). Defaults to :func:`get_settings`.
    metrics, params:
        Optional metrics/params to log alongside the model.
    run_name:
        Optional MLflow run name.

    Returns
    -------
    str
        The registered model **version** (as a string).
    """
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        bundle.version = run.info.run_id  # provenance stamp baked into the artifact
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            pkl_path = Path(tmp) / _BUNDLE_FILENAME
            with open(pkl_path, "wb") as f:
                pickle.dump(bundle, f)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=EVDeCAFSWrapper(),
                artifacts={_BUNDLE_ARTIFACT: str(pkl_path)},
                registered_model_name=settings.registered_model_name,
            )

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    versions = client.search_model_versions(f"run_id='{run.info.run_id}'")
    version = max(versions, key=lambda v: int(v.version)).version
    logger.info(
        "Registered %s version %s (run %s)",
        settings.registered_model_name,
        version,
        run.info.run_id,
    )
    return str(version)


def promote_to_champion(version: str, settings: Settings | None = None) -> None:
    """Point the ``@champion`` alias at ``version`` of the registered model."""
    settings = settings or get_settings()
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    client.set_registered_model_alias(
        settings.registered_model_name, settings.champion_alias, version
    )
    logger.info(
        "Alias @%s -> %s version %s",
        settings.champion_alias,
        settings.registered_model_name,
        version,
    )


def load_champion(settings: Settings | None = None) -> mlflow.pyfunc.PyFuncModel:
    """Load the champion model by alias (``models:/<name>@champion``) — never a path."""
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.registered_model_name}@{settings.champion_alias}"
    logger.info("Loading champion model: %s", uri)
    return mlflow.pyfunc.load_model(uri)


def load_champion_bundle(settings: Settings | None = None) -> tuple[ModelBundle, dict[str, Any]]:
    """Load the champion :class:`ModelBundle` plus its registry metadata, by alias.

    Returns ``(bundle, metadata)`` where ``metadata`` has ``name``, ``version``, ``run_id``,
    and ``aliases``. Used by the serving layer at startup so requests call the typed inference
    contract directly (no per-request pyfunc overhead).
    """
    settings = settings or get_settings()
    pyfunc_model = load_champion(settings)
    bundle = pyfunc_model.unwrap_python_model().bundle  # type: ignore[attr-defined]

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    mv = client.get_model_version_by_alias(settings.registered_model_name, settings.champion_alias)
    metadata = {
        "name": mv.name,
        "version": mv.version,
        "run_id": mv.run_id,
        "aliases": list(mv.aliases),
    }
    bundle.version = str(mv.version)
    return bundle, metadata
