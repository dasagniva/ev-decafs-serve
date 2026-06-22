"""Two-phase training pipeline: time series -> fitted, bundled EV-DeCAFS model.

Orchestration distilled from changepoint-evdecafs ``scripts/run_pipeline.py`` (``run_phase1`` +
``run_phase2_train``), keeping only what produces the registered model:

    estimate_ar1_params -> compute_evi_field -> [BIC sweep ->] ev_decafs (flat) ->
    extract_features -> BOCPD-oracle labelling (feature fallback) -> SMOTE -> FourierPNN.fit

All hyperparameters come from a :class:`~evdecafs_serve.config.TrainConfig`. The research repo's
GPD-adaptive / exceedance-count penalties and the CUSUM/relabelling diagnostics are dropped:
they never fed the FPNN (DECISIONS.md #2, #6).
"""

from __future__ import annotations

import numpy as np

from evdecafs_serve.config import TrainConfig
from evdecafs_serve.features.ar1 import compute_bic_penalty, estimate_ar1_params
from evdecafs_serve.features.evt import compute_evi_field
from evdecafs_serve.features.extract import extract_features
from evdecafs_serve.models.bundle import ModelBundle
from evdecafs_serve.models.decafs import ev_decafs
from evdecafs_serve.models.fpnn import FourierPNN
from evdecafs_serve.monitoring.features import (
    WINDOW_FEATURE_COLUMNS,
    series_to_window_features,
)
from evdecafs_serve.training.bocpd import run_bocpd
from evdecafs_serve.training.labelling import (
    compute_kappa_mu,
    label_changepoints,
    label_with_bocpd,
    refine_pending_labels,
)
from evdecafs_serve.training.smote import balance_training_data
from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger(__name__)


def _select_alpha_0(
    y_train: np.ndarray,
    lambda_param: float,
    gamma: float,
    phi: float,
    cfg: TrainConfig,
) -> float:
    """Resolve the flat Phase-I penalty alpha_0 (fixed, BIC, or BIC sweep)."""
    p1 = cfg.phase1
    n = len(y_train)

    if p1.alpha_0_mode != "bic":
        logger.info("Phase I alpha_0=%.4f (fixed mode)", p1.alpha_0)
        return float(p1.alpha_0)

    if not p1.tune_bic:
        a0 = compute_bic_penalty(n, p1.bic_multiplier)
        logger.info("Phase I alpha_0=%.4f (BIC, C=%.2f)", a0, p1.bic_multiplier)
        return a0

    # BIC sweep: pick the largest C whose detection count still meets the target
    # (conservative — avoids over-detection), falling back to the closest count.
    expected = p1.expected_n_changepoints
    best: tuple[float, float, int] | None = None  # (C, alpha_0, n_detected)
    fallback: tuple[float, float, int] | None = None
    for C in p1.bic_sweep_values:
        a0 = compute_bic_penalty(n, C)
        res = ev_decafs(y_train, np.full(n, a0), lambda_param, gamma, phi, n_grid=p1.n_grid)
        n_det = len(res["changepoints"])
        logger.info("  BIC sweep C=%.2f -> alpha_0=%.2f, n_detected=%d", C, a0, n_det)
        if n_det >= expected and (best is None or C > best[0]):
            best = (C, a0, n_det)
        if fallback is None or abs(n_det - expected) < abs(fallback[2] - expected):
            fallback = (C, a0, n_det)

    chosen = best if best is not None else fallback
    assert chosen is not None  # bic_sweep_values is non-empty
    logger.info(
        "Auto-selected C=%.2f (alpha_0=%.4f, n_detected=%d, target=%d)",
        chosen[0],
        chosen[1],
        chosen[2],
        expected,
    )
    return chosen[1]


def _make_labels(
    y_train: np.ndarray,
    X_train: np.ndarray,
    cps: np.ndarray,
    ar1: dict[str, float],
    true_cps_train: np.ndarray | None,
    cfg: TrainConfig,
) -> np.ndarray:
    """Produce FPNN training labels via the BOCPD oracle, with a feature-based fallback."""
    lab = cfg.labelling
    kappa_mu = compute_kappa_mu(X_train, percentile=lab.kappa_mu_percentile)
    feature_labels = label_changepoints(X_train, kappa_mu, kappa_S=lab.kappa_S)

    bocpd_flags = run_bocpd(
        y_train,
        phi=ar1["phi"],
        sigma_v=float(np.sqrt(ar1["sigma_v_sq"])),
        threshold=lab.bocpd_threshold,
    )
    bocpd_cps = np.where(bocpd_flags)[0]

    has_gt = true_cps_train is not None and len(true_cps_train) > 0
    tol = int(lab.bocpd_tolerance_fraction * len(y_train))
    labels, _ = label_with_bocpd(
        decafs_cps=cps,
        bocpd_cps=bocpd_cps,
        true_cps=np.asarray(true_cps_train, dtype=int) if has_gt else np.array([], dtype=int),
        tolerance=tol,
        has_ground_truth=has_gt,
    )
    if not has_gt:
        labels = refine_pending_labels(labels, X_train, kappa_mu, kappa_S=lab.kappa_S)

    # Fall back to feature-based labels if BOCPD collapsed to a single class.
    if len(np.unique(labels)) < 2:
        logger.warning("BOCPD labelling gave a single class — falling back to feature labels.")
        labels = feature_labels

    return labels


def train_model(
    y_train: np.ndarray,
    cfg: TrainConfig,
    true_cps_train: np.ndarray | None = None,
) -> ModelBundle:
    """Run the full two-phase training pipeline and return a fitted :class:`ModelBundle`.

    Parameters
    ----------
    y_train:
        Training series, shape ``(n,)``.
    cfg:
        Validated pipeline config.
    true_cps_train:
        Optional ground-truth changepoint indices (improve BOCPD oracle labels when available).

    Returns
    -------
    ModelBundle
        The artifact to register in MLflow.

    Raises
    ------
    ValueError
        If Phase I detects fewer than two changepoints (cannot label/balance/fit).
    """
    p1 = cfg.phase1

    # --- Phase I: AR(1) -> EVI -> penalty -> ev_decafs (flat) ---
    ar1 = estimate_ar1_params(y_train)
    lambda_param = 1.0 / (ar1["sigma_eta_sq"] + 1e-12)
    gamma = 1.0 / (ar1["sigma_v_sq"] + 1e-12)

    xi_field = compute_evi_field(y_train, w=p1.window_halfwidth_w, q0=p1.gpd_percentile_q0)
    alpha_0 = _select_alpha_0(y_train, lambda_param, gamma, ar1["phi"], cfg)

    res = ev_decafs(
        y_train,
        np.full(len(y_train), alpha_0),
        lambda_param,
        gamma,
        ar1["phi"],
        n_grid=p1.n_grid,
    )
    cps = res["changepoints"]
    logger.info("Phase I detected %d changepoints", len(cps))
    if len(cps) < 2:
        raise ValueError(
            f"Phase I detected only {len(cps)} changepoint(s); need >= 2 to train Phase II. "
            "Try a smaller BIC multiplier / more sensitive penalty."
        )

    # --- Phase II: features -> labels -> SMOTE -> FPNN ---
    X_train, feature_names = extract_features(
        y_train, cps, res["means"], L=cfg.labelling.window_L, xi_field=xi_field
    )
    labels = _make_labels(y_train, X_train, cps, ar1, true_cps_train, cfg)

    if cfg.smote.enabled:
        X_fit, y_fit = balance_training_data(
            X_train, labels, k_neighbors=cfg.smote.k_neighbors, random_state=cfg.smote.random_state
        )
    else:
        X_fit, y_fit = X_train, labels

    fpnn = FourierPNN(J=cfg.fpnn.J_harmonics, scaling_range=cfg.fpnn.scaling_range)
    fpnn.fit(X_fit, y_fit)
    logger.info("FPNN fitted on %d (balanced) samples, J=%d", len(y_fit), cfg.fpnn.J_harmonics)

    # Drift-monitoring reference profile: window-feature distribution of the training series.
    mon = cfg.monitoring
    reference = series_to_window_features(y_train, mon.window, mon.step)
    logger.info(
        "Drift reference: %d windows (window=%d, step=%d)", len(reference), mon.window, mon.step
    )

    return ModelBundle(
        fpnn=fpnn,
        phi=float(ar1["phi"]),
        sigma_v_sq=float(ar1["sigma_v_sq"]),
        sigma_eta_sq=float(ar1["sigma_eta_sq"]),
        alpha_0=float(alpha_0),
        n_grid=p1.n_grid,
        window_halfwidth_w=p1.window_halfwidth_w,
        gpd_percentile_q0=p1.gpd_percentile_q0,
        window_L=cfg.labelling.window_L,
        feature_names=feature_names,
        dataset=cfg.dataset.name,
        reference_windows=reference.tolist(),
        reference_columns=list(WINDOW_FEATURE_COLUMNS),
        monitor_window=mon.window,
        monitor_step=mon.step,
        drift_share_threshold=mon.drift_share_threshold,
    )
