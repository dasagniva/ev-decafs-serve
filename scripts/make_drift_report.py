"""Generate an Evidently HTML drift report: a series vs the champion's reference profile.

Usage::

    # Report drift for a real input series:
    uv run scripts/make_drift_report.py --series examples/welllog.json --output drift.html

    # Demo: synthesise current traffic, optionally shifted, to see drift fire:
    uv run scripts/make_drift_report.py --demo --shift 40000 --output drift.html

Loads the reference profile from the registered ``@champion`` model (it travels in the bundle),
compares window features of the current series to it, prints the verdict, and writes a
standalone HTML report.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from evdecafs_serve.config import get_settings
from evdecafs_serve.models.registry import load_champion_bundle
from evdecafs_serve.monitoring.drift import run_drift, save_html, window_feature_frame
from evdecafs_serve.monitoring.features import series_to_window_features
from evdecafs_serve.training.evaluate import generate_synthetic_series
from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger("drift-report")


def _load_series(args: argparse.Namespace) -> np.ndarray:
    if args.series:
        payload = json.loads(open(args.series).read())
        series = np.asarray(payload["series"], dtype=float)
    else:
        series = np.asarray(generate_synthetic_series(n=args.demo_n, seed=0)["y"], dtype=float)
    return series + args.shift


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidently drift report vs champion reference.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--series", help="Path to a JSON file with {'series': [...]}.")
    src.add_argument("--demo", action="store_true", help="Synthesise current traffic instead.")
    parser.add_argument("--shift", type=float, default=0.0, help="Add this offset (demo drift).")
    parser.add_argument("--demo-n", type=int, default=1000, help="Length of synthetic demo series.")
    parser.add_argument("--output", default="drift_report.html", help="HTML output path.")
    args = parser.parse_args()

    settings = get_settings()
    bundle, meta = load_champion_bundle(settings)
    if not bundle.reference_windows:
        raise SystemExit("Champion model carries no drift reference profile; retrain to add one.")

    series = _load_series(args)
    reference = window_feature_frame(np.asarray(bundle.reference_windows))
    current = window_feature_frame(
        series_to_window_features(series, bundle.monitor_window, bundle.monitor_step)
    )

    result, run = run_drift(reference, current, drift_share=bundle.drift_share_threshold)
    save_html(run, args.output)

    drifted = [c.column for c in result.columns if c.drifted]
    print(
        f"[{meta['name']} v{meta['version']}] dataset_drift={result.dataset_drift} "
        f"share={result.share_drifted:.2f} drifted_columns={drifted}\n"
        f"HTML report written to {args.output}"
    )


if __name__ == "__main__":
    main()
