"""Train the EV-DeCAFS model from a config file and register it in MLflow.

Usage::

    uv run scripts/train.py --config configs/macro.yaml [--promote]

Runs the full two-phase pipeline (all hyperparameters from the config — zero hard-coded values),
logs the bundled model to the MLflow registry as ``ev-decafs``, and optionally promotes the new
version to ``@champion``.
"""

from __future__ import annotations

import argparse

from evdecafs_serve.config import TrainConfig, get_settings, load_config
from evdecafs_serve.data.datasets import load_training_series
from evdecafs_serve.models.registry import log_and_register, promote_to_champion
from evdecafs_serve.training.pipeline import train_model
from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger("train")


def _params_for_logging(cfg: TrainConfig) -> dict:
    """Flatten the config knobs we want visible in MLflow."""
    return {
        "dataset": cfg.dataset.name,
        "alpha_0_mode": cfg.phase1.alpha_0_mode,
        "tune_bic": cfg.phase1.tune_bic,
        "bic_multiplier": cfg.phase1.bic_multiplier,
        "n_grid": cfg.phase1.n_grid,
        "window_halfwidth_w": cfg.phase1.window_halfwidth_w,
        "gpd_percentile_q0": cfg.phase1.gpd_percentile_q0,
        "window_L": cfg.labelling.window_L,
        "J_harmonics": cfg.fpnn.J_harmonics,
        "smote_enabled": cfg.smote.enabled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register the EV-DeCAFS model.")
    parser.add_argument("--config", required=True, help="Path to configs/<name>.yaml")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote the newly trained version to @champion after registering.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = get_settings()
    logger.info(
        "Training on dataset '%s' (tracking: %s)",
        cfg.dataset.name,
        settings.mlflow_tracking_uri,
    )

    y_train, true_cps_train = load_training_series(cfg)
    bundle = train_model(y_train, cfg, true_cps_train=true_cps_train)

    version = log_and_register(
        bundle,
        settings,
        params=_params_for_logging(cfg),
        metrics={"alpha_0": bundle.alpha_0, "phi": bundle.phi},
        run_name=f"train-{cfg.dataset.name}",
    )
    logger.info("Registered %s version %s", settings.registered_model_name, version)

    if args.promote:
        promote_to_champion(version, settings)
        logger.info("Promoted version %s to @%s", version, settings.champion_alias)

    print(f"Trained and registered {settings.registered_model_name} v{version}")


if __name__ == "__main__":
    main()
