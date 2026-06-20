"""MLflow experiment logging.

A thin wrapper around MLflow so runs are recorded consistently. MLflow is
imported lazily inside the function — importing this module does not require
MLflow to be installed, which keeps the rest of the package light and lets the
metrics utilities be used on their own.
"""

from __future__ import annotations

from typing import Any


def log_to_mlflow(
    params: dict[str, Any],
    metrics: dict[str, float],
    *,
    experiment_name: str = "sensor-intelligence",
    run_name: str | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log a single run's parameters and metrics to MLflow.

    Parameters
    ----------
    params:
        Hyperparameters or run configuration to record.
    metrics:
        Numeric metrics to record (for example backtest scores).
    experiment_name:
        MLflow experiment to log under; created if missing.
    run_name:
        Optional human-readable run name.
    tracking_uri:
        Optional MLflow tracking URI; defaults to MLflow's local ``./mlruns``.

    Returns
    -------
    str
        The MLflow run id.
    """
    import mlflow

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        return str(run.info.run_id)
