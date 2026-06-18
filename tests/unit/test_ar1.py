from __future__ import annotations

import numpy as np
import pytest

from evdecafs_serve.features.ar1 import compute_bic_penalty, estimate_ar1_params


def test_estimate_ar1_params_on_fixture(welllog_fixture):
    y, _ = welllog_fixture
    params = estimate_ar1_params(y)
    assert -1.0 < params["phi"] < 1.0
    assert params["sigma_v_sq"] > 0
    assert params["sigma_eta_sq"] > 0
    assert np.isfinite(params["phi"])
    assert np.isfinite(params["sigma_v_sq"])
    assert np.isfinite(params["sigma_eta_sq"])


def test_estimate_ar1_params_requires_at_least_three_obs():
    with pytest.raises(ValueError):
        estimate_ar1_params(np.array([1.0, 2.0]))


def test_estimate_ar1_params_sigma_eta_floor():
    # Near-unit-root series should still produce a regularised, strictly positive sigma_eta_sq.
    rng = np.random.default_rng(0)
    n = 200
    eps = np.empty(n)
    eps[0] = 0.0
    for t in range(1, n):
        eps[t] = 0.999 * eps[t - 1] + rng.normal(0, 1.0)
    params = estimate_ar1_params(eps)
    assert params["sigma_eta_sq"] >= 1e-3 * params["sigma_v_sq"] * 0.999


def test_compute_bic_penalty_monotonic_in_C_and_n():
    assert compute_bic_penalty(100, C=2.0) < compute_bic_penalty(100, C=4.0)
    assert compute_bic_penalty(100, C=2.0) < compute_bic_penalty(1000, C=2.0)
    assert compute_bic_penalty(100, C=2.0) == pytest.approx(2.0 * np.log(100))
