"""EVT (extreme value index) field computation for EV-DeCAFS Phase II features.

Ported from changepoint-evdecafs/src/phase1/evt_penalty.py — algorithm unchanged.

Only ``compute_evi_field`` is ported. The research repo's ``compute_adaptive_penalty`` and
``compute_exceedance_count_penalty`` are not, per DECISIONS.md #2: every production call site
in the research pipeline uses a flat Phase-I penalty, never the GPD-adaptive one — EVT enters
the system exclusively through the ``xi_local`` feature this module feeds into
``features.extract``.
"""

from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger(__name__)


def compute_evi_field(
    y: np.ndarray,
    w: int = 50,
    q0: float = 0.90,
    min_exceedances: int = 5,
) -> np.ndarray:
    """Compute the local Extreme Value Index (xi_t) at each time point.

    For each time ``t``, a local window ``W_t`` of half-width ``w`` is formed.
    Deviations ``|y_s - mean(W_t)|`` are thresholded at their ``q0``-th
    percentile; the GPD shape parameter ``xi_t`` is estimated via the
    method-of-moments estimator (closed form, ~100x faster than MLE):

        xi_MOM = 0.5 * (1 - 1 / (Var / Mean^2))

    where Mean and Var are the sample mean and variance of the exceedances.

    If a window contains fewer than ``min_exceedances`` points, ``xi_t = 0``.

    Parameters
    ----------
    y:
        Univariate time series, shape ``(n,)``.
    w:
        Half-width of the local window.
    q0:
        Percentile threshold for identifying GPD exceedances (0–1).
    min_exceedances:
        Minimum number of exceedances required to estimate xi.

    Returns
    -------
    xi_field : np.ndarray, shape ``(n,)``
        Local EVI estimates.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    xi_field = np.zeros(n)
    n_successful = 0
    t_start = time.perf_counter()

    # Pad array to avoid per-step boundary checks
    y_padded = np.pad(y, w, mode="reflect")

    for t in tqdm(range(n), desc="EVI field", unit="step", leave=False):
        window = y_padded[t : t + 2 * w + 1]  # length 2w+1
        mu_W = np.mean(window)
        deviations = np.abs(window - mu_W)
        threshold = np.percentile(deviations, q0 * 100.0)

        exceedances = deviations[deviations > threshold] - threshold
        if len(exceedances) < min_exceedances:
            xi_field[t] = 0.0
            continue

        # Method-of-moments GPD estimator (closed form):
        # For GPD(xi, beta): Var/Mean^2 = 1/(1-2*xi)  =>  xi = 0.5*(1 - 1/ratio)
        m = np.mean(exceedances)
        if m <= 0.0:
            xi_field[t] = 0.0
            continue
        v = np.var(exceedances)
        ratio = v / (m * m)
        xi_mom = 0.5 * (1.0 - 1.0 / max(ratio, 1e-10))
        xi_field[t] = float(np.clip(xi_mom, -1.0, 2.0))
        n_successful += 1

    elapsed = time.perf_counter() - t_start
    logger.info(
        "EVI field computed via method-of-moments in %.2fs (n=%d, w=%d); "
        "%d/%d windows estimated, mean xi=%.4f",
        elapsed,
        n,
        w,
        n_successful,
        n,
        float(np.mean(xi_field)),
    )
    return xi_field
