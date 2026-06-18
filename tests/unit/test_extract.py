from __future__ import annotations

import numpy as np

from evdecafs_serve.features.evt import compute_evi_field
from evdecafs_serve.features.extract import extract_features


def test_extract_features_4_columns_without_xi_field(welllog_fixture):
    y, meta = welllog_fixture
    means = np.full_like(y, np.mean(y))
    cps = np.array(meta["true_changepoints"], dtype=int)
    X, names = extract_features(y, cps, means, L=5)
    assert X.shape == (len(cps), 4)
    assert names == ["delta_mu", "S", "phi_local", "V"]
    assert np.all(np.isfinite(X))


def test_extract_features_5_columns_with_xi_field(welllog_fixture):
    y, meta = welllog_fixture
    means = np.full_like(y, np.mean(y))
    cps = np.array(meta["true_changepoints"], dtype=int)
    xi_field = compute_evi_field(y, w=20, q0=0.90)
    X, names = extract_features(y, cps, means, L=5, xi_field=xi_field)
    assert X.shape == (len(cps), 5)
    assert names[-1] == "xi_local"
    assert np.all(np.isfinite(X))


def test_extract_features_empty_changepoints():
    y = np.arange(50, dtype=float)
    means = np.zeros(50)
    X, _ = extract_features(y, np.array([], dtype=int), means, L=5)
    assert X.shape == (0, 4)
