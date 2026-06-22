"""Promote a registered EV-DeCAFS model version to ``@champion``.

Usage::

    uv run scripts/promote_model.py --version 3
    uv run scripts/promote_model.py --latest

The serving layer (Phase 3) loads ``models:/ev-decafs@champion``, so this alias controls which
version is served. Promotion is a deliberate, separate step from training.
"""

from __future__ import annotations

import argparse

from mlflow.tracking import MlflowClient

from evdecafs_serve.config import get_settings
from evdecafs_serve.models.registry import promote_to_champion
from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger("promote")


def _latest_version(model_name: str, tracking_uri: str) -> str:
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for '{model_name}'.")
    return max(versions, key=lambda v: int(v.version)).version


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote an EV-DeCAFS version to @champion.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", help="Explicit model version to promote.")
    group.add_argument("--latest", action="store_true", help="Promote the highest version number.")
    args = parser.parse_args()

    settings = get_settings()
    version = (
        _latest_version(settings.registered_model_name, settings.mlflow_tracking_uri)
        if args.latest
        else args.version
    )

    promote_to_champion(version, settings)
    print(f"Promoted {settings.registered_model_name} v{version} to @{settings.champion_alias}")


if __name__ == "__main__":
    main()
