"""Registry round-trip: train -> log/register -> promote -> load champion -> predict.

Promotes the INTAKE.md §3 spike to a real test: the model loaded by the ``@champion`` alias must
produce the same detection/classification as calling the inference contract on the in-memory
bundle directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evdecafs_serve.models.inference import detect_and_classify
from evdecafs_serve.models.registry import (
    load_champion,
    log_and_register,
    promote_to_champion,
)
from evdecafs_serve.training.pipeline import train_model


def test_champion_roundtrip_matches_direct_inference(
    fast_config, synthetic_split, isolated_settings
):
    y_train, sustained_train, y_test = synthetic_split

    bundle = train_model(y_train, fast_config, true_cps_train=sustained_train)
    expected = detect_and_classify(y_test, bundle)
    assert len(expected.changepoints) > 0  # sanity: the test series yields detections

    version = log_and_register(bundle, isolated_settings, run_name="roundtrip")
    promote_to_champion(version, isolated_settings)

    champion = load_champion(isolated_settings)
    out = champion.predict(pd.DataFrame({"y": y_test}))

    assert out["changepoints"] == expected.changepoints
    assert out["segment_labels"] == expected.segment_labels
    assert np.allclose(out["probabilities"], expected.probabilities)
    assert np.allclose(out["uncertainty"], expected.uncertainty)


def test_load_champion_uses_alias_not_path(fast_config, synthetic_split, isolated_settings):
    """Champion loading must resolve the registry alias, never a filesystem path."""
    y_train, sustained_train, _ = synthetic_split
    bundle = train_model(y_train, fast_config, true_cps_train=sustained_train)
    version = log_and_register(bundle, isolated_settings, run_name="alias")
    promote_to_champion(version, isolated_settings)

    # If the alias weren't set, this raises; a successful load proves alias resolution.
    champion = load_champion(isolated_settings)
    assert champion is not None
