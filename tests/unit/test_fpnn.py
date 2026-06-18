from __future__ import annotations

import numpy as np
import pytest

from evdecafs_serve.models.fpnn import FourierPNN


def _toy_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 5))
    X[:, 0] += rng.choice([0, 6], size=60)
    y = (X[:, 0] > 3).astype(int)
    return X, y


def test_fit_predict_proba_sums_to_one():
    X, y = _toy_dataset()
    fpnn = FourierPNN(J=10, scaling_range=(-0.5, 0.5)).fit(X, y)
    proba = fpnn.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_predict_matches_argmax_of_predict_proba():
    X, y = _toy_dataset()
    fpnn = FourierPNN().fit(X, y)
    proba = fpnn.predict_proba(X)
    pred = fpnn.predict(X)
    assert np.array_equal(pred, fpnn.classes_[np.argmax(proba, axis=1)])


def test_predict_before_fit_raises():
    fpnn = FourierPNN()
    with pytest.raises(RuntimeError):
        fpnn.predict_proba(np.zeros((3, 5)))


def test_get_coefficients_keys_match_classes():
    X, y = _toy_dataset()
    fpnn = FourierPNN().fit(X, y)
    coefs = fpnn.get_coefficients()
    assert set(coefs["cos"].keys()) == set(int(c) for c in fpnn.classes_)
    assert set(coefs["sin"].keys()) == set(int(c) for c in fpnn.classes_)
