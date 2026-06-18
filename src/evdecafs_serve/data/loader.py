"""Data loading for evdecafs_serve experiments.

Ported from changepoint-evdecafs/src/data/loader.py — algorithms unchanged. Only the three
roadmap target datasets are ported (``welllog``, ``oilwell``, ``us_ip_growth``); the TCPD and
Brent-crude loaders are not, per DECISIONS.md #1 and INTAKE.md item #6 (TCPD licensing is
restricted/unclear, and Brent crude isn't one of the roadmap's target datasets).

Unlike the research repo, callers must pass explicit paths — there is no cwd-relative
``"data/raw/..."`` default, since this is now an installed package that may run from any
working directory (e.g. inside a container).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger(__name__)

# Synthetic well-log ground-truth changepoint positions (0-indexed)
_WELLLOG_TRUE_CPS = np.array(
    [400, 820, 1210, 1320, 1540, 1790, 2050, 2380, 2690, 2990, 3300, 3590],
    dtype=int,
)
_WELLLOG_N = 4050


def load_welllog_data(
    cache_path: str | Path,
    train_fraction: float = 0.75,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (or synthesise) the well-log nuclear response dataset.

    Attempts to read a local CSV first.  If it does not exist, generates a
    synthetic surrogate with known changepoints and outliers that mirrors
    the statistical properties of Ruanaidh & Fitzgerald (2012).

    Parameters
    ----------
    cache_path:
        Path to a CSV with a single numeric column of observations.
        If the file does not exist, a synthetic dataset is generated and
        saved here for reproducibility.
    train_fraction:
        Fraction of observations used for the training split (chronological).
    random_seed:
        Seed for the synthetic data generator.

    Returns
    -------
    y_train : np.ndarray
    y_test : np.ndarray
    ground_truth_changepoints : np.ndarray
        Indices of true changepoints in the *full* (pre-split) signal.
    ground_truth_outliers : np.ndarray
        Indices of injected outlier spikes in the full signal.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    true_cps = _WELLLOG_TRUE_CPS.copy()

    if cache_path.exists():
        logger.info("Loading well-log data from: %s", cache_path)
        df = pd.read_csv(cache_path)
        col = df.columns[0]
        y = df[col].values.astype(float)
        # When loading real data we don't have ground-truth outliers
        ground_truth_outliers = np.array([], dtype=int)
        logger.info("Loaded %d well-log observations", len(y))
    else:
        logger.info("Well-log CSV not found at %s — generating synthetic surrogate.", cache_path)
        y, ground_truth_outliers = _generate_synthetic_welllog(
            n=_WELLLOG_N,
            changepoints=true_cps,
            random_seed=random_seed,
        )
        pd.DataFrame({"welllog": y}).to_csv(cache_path, index=False)
        logger.info(
            "Synthetic well-log saved to %s (%d obs, %d changepoints, %d outliers)",
            cache_path,
            len(y),
            len(true_cps),
            len(ground_truth_outliers),
        )

    n_train = int(len(y) * train_fraction)
    y_train = y[:n_train]
    y_test = y[n_train:]

    logger.info(
        "Well-log split — n=%d, train=%d (%.0f%%), test=%d (%.0f%%)",
        len(y),
        len(y_train),
        100 * len(y_train) / len(y),
        len(y_test),
        100 * len(y_test) / len(y),
    )
    return y_train, y_test, true_cps, ground_truth_outliers


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_synthetic_welllog(
    n: int,
    changepoints: np.ndarray,
    random_seed: int = 42,
    n_outliers: int = 20,
    outlier_magnitude: float = 30_000.0,
    ar1_phi: float = 0.5,
    ar1_sigma_v: float = 2_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic well-log surrogate.

    Constructs a piecewise-constant signal whose segment means are drawn
    uniformly from [70 000, 140 000], adds AR(1) noise, and injects outlier
    spikes at random positions.

    Parameters
    ----------
    n:
        Total number of observations.
    changepoints:
        Sorted array of changepoint indices (segment boundaries).
    random_seed:
        NumPy random seed for reproducibility.
    n_outliers:
        Number of outlier spikes to inject.
    outlier_magnitude:
        Absolute spike size (sign is chosen randomly).
    ar1_phi:
        AR(1) autocorrelation for the noise process.
    ar1_sigma_v:
        Innovation standard deviation for the AR(1) noise.

    Returns
    -------
    y : np.ndarray, shape (n,)
    outlier_indices : np.ndarray of int
    """
    rng = np.random.default_rng(random_seed)

    # Build piecewise-constant mean
    boundaries = np.concatenate([[0], changepoints, [n]])
    mu = np.empty(n)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        level = rng.uniform(70_000, 140_000)
        mu[start:end] = level

    # AR(1) noise: epsilon_t = phi * epsilon_{t-1} + v_t
    v = rng.normal(0, ar1_sigma_v, size=n)
    epsilon = np.empty(n)
    epsilon[0] = v[0]
    for t in range(1, n):
        epsilon[t] = ar1_phi * epsilon[t - 1] + v[t]

    y = mu + epsilon

    # Inject outlier spikes at random positions (avoid the first and last 10)
    all_positions = np.arange(10, n - 10)
    outlier_indices = rng.choice(all_positions, size=n_outliers, replace=False)
    outlier_indices.sort()
    signs = rng.choice([-1, 1], size=n_outliers)
    y[outlier_indices] += signs * outlier_magnitude

    return y, outlier_indices


def load_oilwell_data(
    path: str | Path,
    train_fraction: float = 0.75,
    random_seed: int = 42,
) -> dict:
    """Load (or generate) oil-well drilling rate dataset.

    If ``path`` exists, loads it directly with no ground-truth changepoints.
    Otherwise generates a synthetic surrogate with 8 changepoints.

    Returns
    -------
    dict with keys: ``y_train``, ``y_test``, ``true_cps_train``,
    ``true_cps_test``, ``split_index``, ``name``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _OIL_N = 4000
    _OIL_TRUE_CPS = np.array([400, 800, 1200, 1600, 2000, 2500, 3000, 3500], dtype=int)

    if path.exists():
        logger.info("Loading oil-well data from: %s", path)
        df = pd.read_csv(path)
        y = df.iloc[:, 0].values.astype(float)
        known_cps = np.array([], dtype=int)
        logger.info("Loaded %d oil-well observations", len(y))
    else:
        logger.info("Oil-well CSV not found at %s — generating synthetic surrogate.", path)
        rng = np.random.default_rng(random_seed)
        boundaries = np.concatenate([[0], _OIL_TRUE_CPS, [_OIL_N]])
        mu = np.empty(_OIL_N)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            mu[start:end] = rng.uniform(50_000, 120_000)
        v = rng.normal(0, 2000.0, size=_OIL_N)
        eps = np.empty(_OIL_N)
        eps[0] = v[0]
        for t in range(1, _OIL_N):
            eps[t] = 0.7 * eps[t - 1] + v[t]
        y = mu + eps
        # Inject 12 outliers
        outlier_idx = rng.choice(np.arange(10, _OIL_N - 10), size=12, replace=False)
        signs = rng.choice([-1, 1], size=12)
        y[outlier_idx] += signs * 20_000.0
        pd.DataFrame({"oilwell": y}).to_csv(path, index=False)
        known_cps = _OIL_TRUE_CPS.copy()
        logger.info("Synthetic oil-well data saved to %s", path)

    split = int(len(y) * train_fraction)
    y_train = y[:split]
    y_test = y[split:]

    train_mask = known_cps < split
    test_mask = known_cps >= split
    true_cps_train = known_cps[train_mask]
    true_cps_test = known_cps[test_mask] - split

    logger.info(
        "Oil-well split — n=%d, train=%d (%.0f%%), test=%d (%.0f%%)",
        len(y),
        len(y_train),
        100 * len(y_train) / len(y),
        len(y_test),
        100 * len(y_test) / len(y),
    )
    return {
        "y_train": y_train,
        "y_test": y_test,
        "true_cps_train": true_cps_train,
        "true_cps_test": true_cps_test,
        "split_index": split,
        "name": "oilwell",
    }


def load_us_ip_growth(
    cache_dir: str | Path,
    series_id: str = "INDPRO",
    start_date: str = "2000-01-01",
    end_date: str = "2026-01-01",
    train_end_date: str = "2023-12-01",
) -> dict:
    """Load US Industrial Production Index and compute monthly growth rate.

    Source: FRED (Federal Reserve Economic Data)
    Series: INDPRO — Industrial Production: Total Index (SA, 2017=100)

    The pipeline operates on MONTH-OVER-MONTH GROWTH RATES (%), not the
    index level. Growth rates are approximately stationary within regimes,
    with regime shifts at recession onsets/recoveries.

    NBER recession dates serve as ground-truth changepoints:
    - Mar 2001 (recession start)
    - Nov 2001 (recession end / recovery start)
    - Dec 2007 (Great Recession start)
    - Jun 2009 (recovery start)
    - Feb 2020 (COVID recession start)
    - Apr 2020 (recovery start)

    Returns dict compatible with the pipeline.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "us_indpro.csv"

    if csv_path.exists():
        # Handle multiple possible column name conventions in cached file
        raw = pd.read_csv(csv_path, nrows=0)
        date_col = next((c for c in raw.columns if c.lower() in ("date", "observation_date")), None)
        if date_col:
            df = pd.read_csv(csv_path, parse_dates=[date_col], index_col=date_col)
            df.index.name = "date"
        else:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index.name = "date"
        logger.info("US IP loaded from cache: %d observations", len(df))
    else:
        # Try fredapi first, fall back to direct CSV download from FRED
        try:
            from fredapi import Fred

            api_key = os.environ.get("FRED_API_KEY", None)
            if api_key:
                fred = Fred(api_key=api_key)
                series = fred.get_series(
                    series_id, observation_start=start_date, observation_end=end_date
                )
                df = pd.DataFrame({"value": series})
                df.index.name = "date"
            else:
                raise ValueError("No FRED API key")
        except (ImportError, ValueError, Exception) as e:
            logger.info("fredapi not available (%s), downloading CSV from FRED...", e)
            import urllib.request as _ur

            url = (
                f"https://fred.stlouisfed.org/graph/fredgraph.csv?"
                f"id={series_id}&cosd={start_date}&coed={end_date}"
            )
            _ur.urlretrieve(url, csv_path)
            raw_dl = pd.read_csv(csv_path, nrows=0)
            date_col = next(
                (c for c in raw_dl.columns if c.lower() in ("date", "observation_date")), None
            )
            if date_col is None:
                raise ValueError(
                    f"Cannot find date column in FRED CSV; columns={list(raw_dl.columns)}"
                ) from e
            df = pd.read_csv(csv_path, parse_dates=[date_col])
            df = df.rename(columns={date_col: "date", series_id: "value"})
            df = df.set_index("date")

        df.to_csv(csv_path)
        logger.info("US IP downloaded and cached: %d observations", len(df))

    if "value" not in df.columns and len(df.columns) == 1:
        df.columns = ["value"]

    # Compute month-over-month growth rate (%)
    df["growth_rate"] = df["value"].pct_change() * 100
    df = df.dropna()

    # NBER recession-based ground truth changepoints
    nber_dates = [
        "2001-03-01",  # recession start
        "2001-11-01",  # recovery
        "2007-12-01",  # Great Recession start
        "2009-06-01",  # recovery
        "2020-02-01",  # COVID start
        "2020-04-01",  # COVID recovery
    ]

    # Convert dates to integer indices in the growth rate series
    nber_indices = []
    for date_str in nber_dates:
        target = pd.Timestamp(date_str)
        if target >= df.index[0] and target <= df.index[-1]:
            idx = df.index.searchsorted(target)
            if idx < len(df):
                nber_indices.append(idx)
    nber_indices = np.array(sorted(set(nber_indices)))

    # Chronological split
    train_end = pd.Timestamp(train_end_date)
    train_mask = df.index <= train_end

    y_train = df.loc[train_mask, "growth_rate"].values
    y_test = df.loc[~train_mask, "growth_rate"].values
    dates_train = df.index[train_mask]
    dates_test = df.index[~train_mask]
    index_train = df.loc[train_mask, "value"].values
    index_test = df.loc[~train_mask, "value"].values

    # Split ground truth CPs
    split_idx = len(y_train)
    cps_train = nber_indices[nber_indices < split_idx]
    cps_test_abs = nber_indices[nber_indices >= split_idx]
    cps_test_rel = cps_test_abs - split_idx

    logger.info(
        "US IP growth rate: train=%d months, test=%d months",
        len(y_train),
        len(y_test),
    )
    logger.info(
        "NBER ground truth: %d CPs total, %d in train, %d in test",
        len(nber_indices),
        len(cps_train),
        len(cps_test_rel),
    )

    return {
        "y_train": y_train,
        "y_test": y_test,
        "true_cps_train": cps_train,
        "true_cps_test": cps_test_rel,
        "dates_train": dates_train,
        "dates_test": dates_test,
        "index_train": index_train,
        "index_test": index_test,
        "split_index": split_idx,
        "name": "us_ip_growth",
        "longname": "US Industrial Production (monthly growth rate)",
        "nber_dates": nber_dates,
        "is_financial": False,
    }
