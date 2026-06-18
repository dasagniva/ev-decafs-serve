"""Hypothesis property tests required by the roadmap's Phase 1 spec."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from evdecafs_serve.features.extract import extract_features
from evdecafs_serve.models.decafs import ev_decafs

_FINITE_FLOATS = st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False)


@settings(max_examples=25, deadline=None)
@given(
    y=st.lists(_FINITE_FLOATS, min_size=15, max_size=40),
    shift=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
def test_decafs_changepoints_invariant_under_additive_shift(y: list[float], shift: float) -> None:
    """Adding a constant to the whole series must not change which changepoints fire.

    The mu-grid is rebuilt from y's own min/max/std on every call, so shifting y by a
    constant shifts the grid by the same constant — the recursion's relative costs, and
    therefore the detected changepoints, must be identical.
    """
    y_arr = np.array(y, dtype=float)
    alpha_t = np.full(len(y_arr), 5.0)
    base = ev_decafs(y_arr, alpha_t, lambda_param=1.0, gamma=1.0, phi=0.3, n_grid=40)
    shifted = ev_decafs(y_arr + shift, alpha_t, lambda_param=1.0, gamma=1.0, phi=0.3, n_grid=40)
    assert np.array_equal(base["changepoints"], shifted["changepoints"])


@settings(max_examples=30, deadline=None)
@given(
    y=st.lists(_FINITE_FLOATS, min_size=20, max_size=60),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_extract_features_never_returns_nan_or_inf(y: list[float], seed: int) -> None:
    """Feature extraction must never produce NaN/inf for any finite input series."""
    y_arr = np.array(y, dtype=float)
    n = len(y_arr)
    rng = np.random.default_rng(seed)
    means = y_arr + rng.normal(0, 0.01, size=n)
    n_cps = min(5, n)
    cps = np.sort(rng.choice(np.arange(n), size=n_cps, replace=False))

    X, _ = extract_features(y_arr, cps, means, L=3)
    assert np.all(np.isfinite(X))

    xi_field = rng.normal(0, 1.0, size=n)
    X5, _ = extract_features(y_arr, cps, means, L=3, xi_field=xi_field)
    assert np.all(np.isfinite(X5))
