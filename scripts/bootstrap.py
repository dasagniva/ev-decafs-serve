"""Ensure a champion model exists, training one if needed (idempotent first-boot bootstrap).

Run by the API container's entrypoint before uvicorn starts. If `models:/<name>@champion`
already exists in the registry, this is a no-op (subsequent boots reuse the persisted model).
Otherwise it trains a quick, offline model from ``EVDECAFS_BOOTSTRAP_CONFIG`` and promotes it.

This is *not* training-in-the-serving-code: it's a separate startup step. The serving app
itself (``serving/app.py``) only ever loads and infers.
"""

from __future__ import annotations

import os
import time

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from evdecafs_serve.config import Settings, get_settings, load_config
from evdecafs_serve.data.datasets import load_training_series
from evdecafs_serve.models.registry import log_and_register, promote_to_champion
from evdecafs_serve.training.pipeline import train_model
from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger("bootstrap")

_DEFAULT_CONFIG = "/app/configs/bootstrap.yaml"


def _wait_for_tracking(settings: Settings, attempts: int = 30, delay: float = 2.0) -> MlflowClient:
    """Return a client once the tracking backend answers (handles container start ordering)."""
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    for i in range(attempts):
        try:
            client.search_registered_models(max_results=1)
            return client
        except MlflowException as exc:
            logger.info("Tracking not ready yet (%d/%d): %s", i + 1, attempts, exc)
            time.sleep(delay)
    raise RuntimeError(
        f"MLflow tracking at {settings.mlflow_tracking_uri} did not become reachable."
    )


def _champion_exists(client: MlflowClient, settings: Settings) -> bool:
    try:
        client.get_model_version_by_alias(settings.registered_model_name, settings.champion_alias)
        return True
    except MlflowException:
        return False


def main() -> None:
    settings = get_settings()
    client = _wait_for_tracking(settings)

    if _champion_exists(client, settings):
        logger.info(
            "Champion %s@%s already present — skipping bootstrap.",
            settings.registered_model_name,
            settings.champion_alias,
        )
        return

    config_path = os.environ.get("EVDECAFS_BOOTSTRAP_CONFIG", _DEFAULT_CONFIG)
    logger.info("No champion found — bootstrapping from %s", config_path)
    cfg = load_config(config_path)

    y_train, true_cps_train = load_training_series(cfg)
    bundle = train_model(y_train, cfg, true_cps_train=true_cps_train)
    version = log_and_register(
        bundle,
        settings,
        params={"dataset": cfg.dataset.name, "bootstrap": True},
        run_name=f"bootstrap-{cfg.dataset.name}",
    )
    promote_to_champion(version, settings)
    logger.info(
        "Bootstrapped and promoted %s v%s to @champion",
        settings.registered_model_name,
        version,
    )


if __name__ == "__main__":
    main()
