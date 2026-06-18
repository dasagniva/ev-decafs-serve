from __future__ import annotations

import numpy as np

from evdecafs_serve.features.evt import compute_evi_field


def test_compute_evi_field_shape_and_range(welllog_fixture):
    y, _ = welllog_fixture
    xi = compute_evi_field(y, w=20, q0=0.90)
    assert xi.shape == y.shape
    assert np.all(np.isfinite(xi))
    assert np.all(xi >= -1.0) and np.all(xi <= 2.0)  # clip range from the estimator


def test_compute_evi_field_zero_on_constant_series():
    y = np.full(80, 5.0)
    xi = compute_evi_field(y, w=10, q0=0.90)
    assert np.all(xi == 0.0)


def test_compute_evi_field_short_window_does_not_crash():
    y = np.linspace(0, 1, 25)
    xi = compute_evi_field(y, w=5, q0=0.90, min_exceedances=5)
    assert xi.shape == y.shape
    assert np.all(np.isfinite(xi))
