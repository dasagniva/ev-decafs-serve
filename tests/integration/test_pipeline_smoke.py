"""End-to-end smoke test: train + register + Monte-Carlo evaluate on tiny synthetic data.

Runs in seconds (small ``n_grid``/``B``), and asserts a registry entry exists and the evaluation
produces CI-bearing metrics in range — the Phase 2 acceptance check, scaled down for CI.
"""

from __future__ import annotations

from mlflow.tracking import MlflowClient

from evdecafs_serve.models.registry import log_and_register
from evdecafs_serve.training.evaluate import METRIC_NAMES, run_evaluation
from evdecafs_serve.training.pipeline import train_model


def test_train_registers_a_model_version(fast_config, synthetic_split, isolated_settings):
    y_train, sustained_train, _ = synthetic_split
    bundle = train_model(y_train, fast_config, true_cps_train=sustained_train)

    version = log_and_register(
        bundle, isolated_settings, params={"dataset": "test"}, run_name="smoke"
    )

    client = MlflowClient(tracking_uri=isolated_settings.mlflow_tracking_uri)
    mv = client.get_model_version(isolated_settings.registered_model_name, version)
    assert mv is not None
    assert mv.name == isolated_settings.registered_model_name


def test_bundle_carries_frozen_detector_params(fast_config, synthetic_split):
    y_train, sustained_train, _ = synthetic_split
    bundle = train_model(y_train, fast_config, true_cps_train=sustained_train)

    assert bundle.n_grid == fast_config.phase1.n_grid
    assert bundle.window_halfwidth_w == fast_config.phase1.window_halfwidth_w
    assert bundle.window_L == fast_config.labelling.window_L
    assert bundle.alpha_0 > 0
    assert -1.0 < bundle.phi < 1.0
    assert bundle.lambda_param > 0 and bundle.gamma > 0
    assert "xi_local" in bundle.feature_names  # 5-feature variant (xi_field was passed)


def test_monte_carlo_eval_produces_ci_metrics(fast_config):
    results = run_evaluation(fast_config)

    assert results["n_successful"] >= 1
    for name in METRIC_NAMES:
        r = results["per_metric"][name]
        assert set(r) >= {"mean", "std", "median", "ci_lower", "ci_upper", "values"}
        assert r["ci_lower"] <= r["ci_upper"]

    ba = results["per_metric"]["balanced_accuracy"]
    assert 0.0 <= ba["mean"] <= 1.0
