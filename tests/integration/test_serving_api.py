"""Serving-layer integration tests (httpx TestClient).

Spins up a real champion model in an isolated sqlite registry, then exercises the API: happy
path, every validation failure mode, and a response-schema contract check.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evdecafs_serve.config import get_settings
from evdecafs_serve.models.registry import log_and_register, promote_to_champion
from evdecafs_serve.training.pipeline import train_model

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def api_client(tmp_path, monkeypatch, fast_config, synthetic_split):
    """A TestClient backed by a freshly trained+promoted champion in a throwaway registry."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVDECAFS_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("EVDECAFS_REGISTERED_MODEL_NAME", "ev-decafs-test")
    monkeypatch.setenv("EVDECAFS_EXPERIMENT_NAME", "ev-decafs-test")
    monkeypatch.setenv("EVDECAFS_MIN_SERIES_LENGTH", "10")
    monkeypatch.setenv("EVDECAFS_MAX_SERIES_LENGTH", "300")

    settings = get_settings()
    y_train, sustained_train, y_test = synthetic_split
    bundle = train_model(y_train, fast_config, true_cps_train=sustained_train)
    version = log_and_register(bundle, settings, run_name="serve")
    promote_to_champion(version, settings)

    from evdecafs_serve.serving.app import app

    with TestClient(app) as client:
        client.y_test = y_test.tolist()  # type: ignore[attr-defined]
        client.y_train = y_train.tolist()  # type: ignore[attr-defined]
        yield client


def test_health_reports_loaded_model(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "ev-decafs-test"
    assert body["model_version"]  # non-empty


def test_model_info_exposes_frozen_params(api_client):
    resp = api_client.get("/model")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "ev-decafs-test"
    assert body["n_grid"] == 80  # from fast_config
    assert "xi_local" in body["feature_names"]
    assert -1.0 < body["phi"] < 1.0
    assert body["alpha_0"] > 0


def test_detect_happy_path_contract(api_client):
    resp = api_client.post("/detect", json={"series": api_client.y_test})
    assert resp.status_code == 200
    body = resp.json()

    # Contract: exact top-level keys.
    assert set(body) == {"changepoints", "n_changepoints", "series_length", "model_version"}
    assert body["series_length"] == len(api_client.y_test)
    assert body["n_changepoints"] == len(body["changepoints"])

    for cp in body["changepoints"]:
        assert set(cp) == {
            "index",
            "label",
            "label_name",
            "prob_recoiled",
            "prob_sustained",
            "uncertainty",
        }
        assert cp["label"] in (0, 1)
        assert cp["label_name"] in ("recoiled", "sustained")
        assert math.isclose(cp["prob_recoiled"] + cp["prob_sustained"], 1.0, abs_tol=1e-6)
        assert 0.0 <= cp["uncertainty"] <= 0.5 + 1e-9


def test_detect_rejects_nan(api_client):
    # Raw body: httpx won't serialise NaN itself, so the *server* must reject it.
    body = '{"series": [1.0, NaN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}'
    resp = api_client.post("/detect", content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 422


def test_detect_rejects_too_short(api_client):
    resp = api_client.post("/detect", json={"series": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert resp.status_code == 422


def test_detect_rejects_too_long(api_client):
    resp = api_client.post("/detect", json={"series": [100000.0] * 400})
    assert resp.status_code == 422


def test_detect_rejects_empty_series(api_client):
    resp = api_client.post("/detect", json={"series": []})
    assert resp.status_code == 422


def test_detect_rejects_mismatched_timestamps(api_client):
    resp = api_client.post(
        "/detect",
        json={"series": [1.0] * 12, "timestamps": [1.0, 2.0, 3.0]},
    )
    assert resp.status_code == 422


def test_detect_rejects_non_increasing_timestamps(api_client):
    n = 12
    resp = api_client.post(
        "/detect",
        json={"series": [1.0] * n, "timestamps": [0.0] * n},
    )
    assert resp.status_code == 422


def test_detect_rejects_unknown_field(api_client):
    resp = api_client.post("/detect", json={"series": [1.0] * 12, "bogus": 5})
    assert resp.status_code == 422


@pytest.mark.parametrize("example", ["welllog.json", "macro.json"])
def test_example_payloads_return_valid_responses(api_client, example):
    payload = json.loads((_EXAMPLES_DIR / example).read_text())
    resp = api_client.post("/detect", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["series_length"] == len(payload["series"])
    assert body["n_changepoints"] == len(body["changepoints"])


def test_validation_error_is_structured_json(api_client):
    body = '{"series": [1.0, Infinity, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}'
    resp = api_client.post("/detect", content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 422
    assert resp.headers["content-type"].startswith("application/json")
    assert "detail" in resp.json()  # structured, not a stack trace


def test_drift_insufficient_data_before_traffic(api_client):
    resp = api_client.get("/monitoring/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "insufficient_data"
    assert body["n_current_windows"] == 0
    assert body["n_reference_windows"] > 0


def test_drift_quiet_on_in_distribution_traffic(api_client):
    # Feed traffic drawn from the same distribution as the reference (the training series).
    for _ in range(3):
        assert api_client.post("/detect", json={"series": api_client.y_train}).status_code == 200
    body = api_client.get("/monitoring/drift").json()
    assert body["status"] == "ok"
    assert body["dataset_drift"] is False


def test_drift_fires_on_shifted_traffic(api_client):
    # Scale + offset shifts mean, std, and range together (a constant offset alone moves only
    # the mean -> 1/3 of columns -> below the 0.5 dataset-drift threshold).
    shifted = [v * 1.6 + 60_000.0 for v in api_client.y_train]
    for _ in range(3):
        assert api_client.post("/detect", json={"series": shifted}).status_code == 200
    body = api_client.get("/monitoring/drift").json()
    assert body["status"] == "ok"
    assert body["dataset_drift"] is True
    assert body["n_drifted_columns"] >= 2
