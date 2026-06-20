from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from sensor_intelligence.domain import TimeSeriesWindow
from sensor_intelligence.models import SeasonalNaiveForecaster
from sensor_intelligence.tracking import backtest, forecast_metrics, log_to_mlflow

_T0 = datetime(2024, 1, 1)


def _window(values: list[float]) -> TimeSeriesWindow:
    timestamps = [_T0 + timedelta(minutes=i) for i in range(len(values))]
    return TimeSeriesWindow(sensor_id="s1", timestamps=timestamps, values=values)


def test_metrics_perfect_prediction() -> None:
    actual = [1.0, 2.0, 3.0, 4.0]
    metrics = forecast_metrics(actual, actual, lower=actual, upper=actual)

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["interval_coverage"] == pytest.approx(1.0)


def test_metrics_known_error() -> None:
    metrics = forecast_metrics([10.0, 20.0], [11.0, 18.0])
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt((1 + 4) / 2))


def test_metrics_interval_coverage_partial() -> None:
    metrics = forecast_metrics(
        [1.0, 5.0, 9.0], [1.0, 5.0, 9.0], lower=[0.0, 4.0, 100.0], upper=[2.0, 6.0, 200.0]
    )
    assert metrics["interval_coverage"] == pytest.approx(2 / 3)
    assert metrics["interval_width"] == pytest.approx((2.0 + 2.0 + 100.0) / 3)


def test_metrics_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        forecast_metrics([1.0, 2.0], [1.0])


def test_backtest_scores_held_out_tail() -> None:
    period = 24
    base = list(np.sin(np.linspace(0, 2 * np.pi, period)))
    window = _window(base * 5)

    def fn(train: TimeSeriesWindow, horizon: int) -> object:
        return SeasonalNaiveForecaster(period=period).forecast(train, horizon)

    forecast, metrics = backtest(fn, window, horizon=period)  # type: ignore[arg-type]
    assert len(forecast.mean) == period
    assert metrics["mae"] < 0.2  # seasonal-naive nails a clean seasonal signal
    assert "interval_coverage" in metrics


def test_backtest_requires_window_longer_than_horizon() -> None:
    with pytest.raises(ValueError, match="longer than the horizon"):
        backtest(
            lambda w, h: SeasonalNaiveForecaster(period=2).forecast(w, h),  # type: ignore[arg-type,return-value]
            _window([1.0, 2.0, 3.0]),
            horizon=5,
        )


def test_log_to_mlflow_records_run(tmp_path: object) -> None:
    mlflow = pytest.importorskip("mlflow")
    # Use a sqlite backend: newer MLflow puts the file store in maintenance mode.
    db = (tmp_path / "mlflow.db").as_posix()  # type: ignore[attr-defined]
    uri = f"sqlite:///{db}"

    run_id = log_to_mlflow(
        {"model": "seasonal_naive", "period": 24},
        {"mae": 0.05, "rmse": 0.07},
        experiment_name="test-exp",
        run_name="unit-test",
        tracking_uri=uri,
    )

    assert isinstance(run_id, str) and run_id
    run = mlflow.tracking.MlflowClient(tracking_uri=uri).get_run(run_id)
    assert run.data.metrics["mae"] == pytest.approx(0.05)
    assert run.data.params["model"] == "seasonal_naive"
