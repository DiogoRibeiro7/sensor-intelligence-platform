"""Forecast evaluation metrics and a simple backtest.

These are pure functions over the domain models — no MLflow dependency — so
they can be used in tests, notebooks, or services without side effects. The
:func:`backtest` helper holds out the tail of a series, forecasts it, and
scores point accuracy plus prediction-interval calibration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..domain import Forecast, TimeSeriesWindow

ForecastFn = Callable[[TimeSeriesWindow, int], Forecast]


def forecast_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute point-accuracy and interval-calibration metrics.

    Parameters
    ----------
    actual, predicted:
        Equal-length observed and forecast values.
    lower, upper:
        Optional prediction-interval bounds; when both are given, empirical
        coverage and mean width are added.

    Returns
    -------
    dict[str, float]
        ``mae``, ``rmse``, ``mape`` and, when bounds are supplied,
        ``interval_coverage`` and ``interval_width``.
    """
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if a.size == 0 or a.shape != p.shape:
        raise ValueError("actual and predicted must be non-empty and equal length.")

    error = p - a
    denom = np.where(np.abs(a) < 1e-9, np.nan, a)
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.nanmean(np.abs(error / denom)) * 100.0),
    }

    if lower is not None and upper is not None:
        lo = np.asarray(lower, dtype=np.float64)
        up = np.asarray(upper, dtype=np.float64)
        if lo.shape != a.shape or up.shape != a.shape:
            raise ValueError("lower and upper must match actual length.")
        inside = (a >= lo) & (a <= up)
        metrics["interval_coverage"] = float(np.mean(inside))
        metrics["interval_width"] = float(np.mean(up - lo))

    return metrics


def backtest(
    forecast_fn: ForecastFn, window: TimeSeriesWindow, horizon: int
) -> tuple[Forecast, dict[str, float]]:
    """Hold out the last ``horizon`` points, forecast them, and score.

    Parameters
    ----------
    forecast_fn:
        Callable taking ``(train_window, horizon)`` and returning a forecast.
        This adapts any forecaster (baseline or ML) to a uniform signature.
    window:
        Full series; its tail becomes the test set.
    horizon:
        Number of points to hold out and forecast.

    Returns
    -------
    tuple[Forecast, dict[str, float]]
        The forecast and its metrics against the held-out tail.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if len(window) <= horizon:
        raise ValueError("window must be longer than the horizon.")

    train = TimeSeriesWindow(
        sensor_id=window.sensor_id,
        timestamps=window.timestamps[:-horizon],
        values=window.values[:-horizon],
    )
    test_values = window.values[-horizon:]
    forecast = forecast_fn(train, horizon)
    metrics = forecast_metrics(test_values, forecast.mean, forecast.lower, forecast.upper)
    return forecast, metrics
