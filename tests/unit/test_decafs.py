from __future__ import annotations

import numpy as np

from evdecafs_serve.models.decafs import ev_decafs


def test_ev_decafs_returns_expected_keys_and_shapes(welllog_fixture):
    y, _ = welllog_fixture
    n = len(y)
    alpha_t = np.full(n, 50.0)
    result = ev_decafs(y, alpha_t, lambda_param=1.0, gamma=1.0, phi=0.3, n_grid=80)
    assert set(result.keys()) == {"changepoints", "means", "cost"}
    assert result["means"].shape == (n,)
    assert np.all(np.isfinite(result["means"]))
    assert np.isfinite(result["cost"])
    assert result["changepoints"].dtype.kind == "i"
    assert np.all((result["changepoints"] >= 0) & (result["changepoints"] < n))


def test_ev_decafs_short_series_returns_trivial_result():
    result = ev_decafs(np.array([1.0]), np.array([1.0]), lambda_param=1.0, gamma=1.0, phi=0.0)
    assert result["changepoints"].size == 0
    assert result["cost"] == 0.0


def test_ev_decafs_flat_penalty_detects_known_jump():
    # A single large, sustained level shift with low noise should be detected as one changepoint.
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(0, 1.0, 60), rng.normal(50.0, 1.0, 60)])
    alpha_t = np.full(len(y), 5.0)
    result = ev_decafs(y, alpha_t, lambda_param=1.0, gamma=1.0, phi=0.0, n_grid=100)
    assert len(result["changepoints"]) >= 1
    assert any(abs(int(cp) - 60) <= 5 for cp in result["changepoints"])
