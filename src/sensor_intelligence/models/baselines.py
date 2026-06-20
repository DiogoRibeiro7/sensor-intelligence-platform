"""Baseline forecasting models.

Simple, transparent forecasters that establish the bar a learned model must
clear. Each produces a :class:`~sensor_intelligence.domain.Forecast` with
prediction intervals derived from in-sample residuals, so downstream consumers
get uncertainty even from the baselines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

from ..domain import Forecast, TimeSeriesWindow

FloatArray = npt.NDArray[np.float64]


def _infer_step(window: TimeSeriesWindow) -> timedelta:
    """Infer the sampling interval from the last two timestamps."""
    if len(window) < 2:
        return timedelta(minutes=1)
    return window.timestamps[-1] - window.timestamps[-2]


def _future_timestamps(window: TimeSeriesWindow, horizon: int) -> list[datetime]:
    """Build ``horizon`` future timestamps continuing the window's cadence."""
    step = _infer_step(window)
    last = window.timestamps[-1]
    return [last + step * (h + 1) for h in range(horizon)]


def _interval(
    mean: FloatArray, residual_std: float, level: float
) -> tuple[list[float], list[float]]:
    """Return symmetric Gaussian prediction-interval bounds around ``mean``."""
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    half = z * residual_std
    lower = (mean - half).tolist()
    upper = (mean + half).tolist()
    return lower, upper


class BaselineForecaster(ABC):
    """Common interface for baseline forecasters."""

    def __init__(self, interval_level: float = 0.9) -> None:
        if not 0.0 < interval_level < 1.0:
            raise ValueError("interval_level must be in (0, 1).")
        self._interval_level = interval_level

    @abstractmethod
    def forecast(self, window: TimeSeriesWindow, horizon: int) -> Forecast:
        """Return a ``horizon``-step forecast for ``window``."""
        raise NotImplementedError


class SeasonalNaiveForecaster(BaselineForecaster):
    """Repeat the value observed one season ago.

    Parameters
    ----------
    period:
        Season length in samples (for example 1440 for daily seasonality at a
        one-minute cadence).
    interval_level:
        Nominal coverage of the prediction interval.
    """

    def __init__(self, period: int, interval_level: float = 0.9) -> None:
        super().__init__(interval_level)
        if period < 1:
            raise ValueError("period must be at least 1.")
        self._period = period

    def forecast(self, window: TimeSeriesWindow, horizon: int) -> Forecast:
        """Forecast each step from the matching point in the prior season."""
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")
        values = np.asarray(window.values, dtype=float)
        if values.size < self._period:
            raise ValueError("window must contain at least one full season.")

        season = values[-self._period :]
        mean = np.array([season[h % self._period] for h in range(horizon)], dtype=float)

        # Residuals from one-season-ago predictions on the observed history.
        in_sample = values[self._period :]
        prior = values[: -self._period]
        residual_std = float(np.std(in_sample - prior)) if in_sample.size else 0.0

        lower, upper = _interval(mean, residual_std, self._interval_level)
        return Forecast(
            sensor_id=window.sensor_id,
            timestamps=_future_timestamps(window, horizon),
            mean=mean.tolist(),
            lower=lower,
            upper=upper,
            interval_level=self._interval_level,
        )


class RollingMeanForecaster(BaselineForecaster):
    """Forecast a flat line at the mean of the last ``window_size`` samples.

    Parameters
    ----------
    window_size:
        Number of trailing samples averaged for the point forecast.
    interval_level:
        Nominal coverage of the prediction interval.
    """

    def __init__(self, window_size: int = 30, interval_level: float = 0.9) -> None:
        super().__init__(interval_level)
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        self._window_size = window_size

    def forecast(self, window: TimeSeriesWindow, horizon: int) -> Forecast:
        """Forecast a constant equal to the trailing rolling mean."""
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")
        values = np.asarray(window.values, dtype=float)
        if values.size == 0:
            raise ValueError("window must be non-empty.")

        tail = values[-self._window_size :]
        point = float(np.mean(tail))
        residual_std = float(np.std(tail))
        mean = np.full(horizon, point, dtype=float)

        lower, upper = _interval(mean, residual_std, self._interval_level)
        return Forecast(
            sensor_id=window.sensor_id,
            timestamps=_future_timestamps(window, horizon),
            mean=mean.tolist(),
            lower=lower,
            upper=upper,
            interval_level=self._interval_level,
        )
