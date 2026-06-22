"""Monte-Carlo evaluation of the EV-DeCAFS model; logs metrics + artifacts to MLflow.

Usage::

    uv run scripts/evaluate.py --config configs/macro.yaml

Logs, per metric (balanced accuracy, MCC, AUC-ROC): the mean, std, median, and 2.5/97.5
percentile confidence-interval bounds; the full per-replication values as a CSV artifact; a JSON
summary; and (if matplotlib is available) a histogram per metric. This reports *this repo's own*
model quality with honest uncertainty — not a reproduction of the research repo's number
(DECISIONS.md #5).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from evdecafs_serve.config import get_settings, load_config
from evdecafs_serve.training.evaluate import METRIC_NAMES, run_evaluation
from evdecafs_serve.utils.logging import setup_logger

logger = setup_logger("evaluate")


def _maybe_plot(values: np.ndarray, name: str, out_dir: Path) -> Path | None:
    """Best-effort histogram of one metric's distribution; ``None`` if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib absent — skipping %s histogram (CSV/JSON still logged).", name)
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values[~np.isnan(values)], bins=20, color="#4C72B0", edgecolor="white")
    ax.axvline(float(np.nanmean(values)), color="#C44E52", linestyle="--", label="mean")
    ax.set_title(f"{name} across replications")
    ax.set_xlabel(name)
    ax.set_ylabel("count")
    ax.legend()
    path = out_dir / f"{name}_hist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte-Carlo evaluation of EV-DeCAFS.")
    parser.add_argument("--config", required=True, help="Path to configs/<name>.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    results = run_evaluation(cfg)
    per_metric = results["per_metric"]

    with mlflow.start_run(run_name=f"eval-{cfg.dataset.name}"):
        mlflow.log_params(
            {
                "dataset": cfg.dataset.name,
                "mc_B": cfg.monte_carlo.B,
                "mc_series_n": cfg.monte_carlo.series_n,
                "mc_n_changepoints": cfg.monte_carlo.n_changepoints,
                "mc_seed": cfg.monte_carlo.seed,
            }
        )
        mlflow.log_metric("mc_n_successful", results["n_successful"])

        for name in METRIC_NAMES:
            r = per_metric[name]
            mlflow.log_metric(f"{name}_mean", r["mean"])
            mlflow.log_metric(f"{name}_std", r["std"])
            mlflow.log_metric(f"{name}_median", r["median"])
            mlflow.log_metric(f"{name}_ci_lower", r["ci_lower"])
            mlflow.log_metric(f"{name}_ci_upper", r["ci_upper"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # Per-replication values (CSV) + JSON summary as artifacts.
            values_df = pd.DataFrame({name: per_metric[name]["values"] for name in METRIC_NAMES})
            csv_path = tmp_dir / "mc_per_replication.csv"
            values_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path))

            keys = ("mean", "std", "median", "ci_lower", "ci_upper")
            summary = {name: {k: per_metric[name][k] for k in keys} for name in METRIC_NAMES}
            summary["n_successful"] = results["n_successful"]
            summary["B"] = results["B"]
            json_path = tmp_dir / "mc_summary.json"
            json_path.write_text(json.dumps(summary, indent=2))
            mlflow.log_artifact(str(json_path))

            for name in METRIC_NAMES:
                plot_path = _maybe_plot(per_metric[name]["values"], name, tmp_dir)
                if plot_path is not None:
                    mlflow.log_artifact(str(plot_path))

    ba = per_metric["balanced_accuracy"]
    print(
        f"[{cfg.dataset.name}] balanced accuracy: "
        f"{ba['mean']:.4f}  CI[{ba['ci_lower']:.4f}, {ba['ci_upper']:.4f}]  "
        f"({results['n_successful']}/{results['B']} usable replications)"
    )


if __name__ == "__main__":
    main()
