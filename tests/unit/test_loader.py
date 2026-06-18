from __future__ import annotations

import numpy as np
import pandas as pd

from evdecafs_serve.data.loader import load_oilwell_data, load_us_ip_growth, load_welllog_data


def test_load_welllog_data_generates_synthetic_when_absent(tmp_path):
    cache_path = tmp_path / "welllog.csv"
    y_train, y_test, true_cps, true_outliers = load_welllog_data(
        cache_path=cache_path, train_fraction=0.75, random_seed=42
    )
    assert cache_path.exists()
    assert len(y_train) == int(4050 * 0.75)
    assert len(y_train) + len(y_test) == 4050
    assert len(true_cps) == 12
    assert len(true_outliers) == 20


def test_load_welllog_data_reads_cache_on_second_call(tmp_path):
    cache_path = tmp_path / "welllog.csv"
    load_welllog_data(cache_path=cache_path, random_seed=42)
    y_train, y_test, _, true_outliers = load_welllog_data(cache_path=cache_path)
    assert true_outliers.size == 0  # no outlier ground truth when reading from a real CSV
    assert len(y_train) + len(y_test) == 4050


def test_load_oilwell_data_generates_synthetic_when_absent(tmp_path):
    path = tmp_path / "oilwell.csv"
    data = load_oilwell_data(path=path, train_fraction=0.75, random_seed=42)
    assert path.exists()
    assert data["name"] == "oilwell"
    assert len(data["y_train"]) + len(data["y_test"]) == 4000
    assert data["split_index"] == len(data["y_train"])


def test_load_us_ip_growth_parses_cached_csv(tmp_path):
    dates = pd.date_range("2018-01-01", periods=80, freq="MS")
    rng = np.random.default_rng(0)
    values = 100 + np.cumsum(rng.normal(0, 0.5, size=80))
    df = pd.DataFrame({"date": dates, "INDPRO": values})
    df.to_csv(tmp_path / "us_indpro.csv", index=False)

    data = load_us_ip_growth(cache_dir=tmp_path, train_end_date="2023-06-01")
    assert "growth_rate" not in data  # not exposed at top level; only y_train/y_test
    assert len(data["y_train"]) + len(data["y_test"]) == 79  # one obs lost to pct_change
    assert data["split_index"] == len(data["y_train"])
    assert np.all(np.isfinite(data["y_train"]))
    assert np.all(np.isfinite(data["y_test"]))
