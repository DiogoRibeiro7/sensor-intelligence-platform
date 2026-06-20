"""Tabular machine-learning forecaster.

A lag-based supervised forecaster: each target value is regressed on a window
of recent lags plus optional cyclical calendar features. Multi-step forecasts
are produced recursively, feeding each prediction back as the newest lag, and
prediction intervals widen with the horizon to reflect compounding
uncertainty.

The regressor is pluggable; the default is scikit-learn's
:class:`~sklearn.ensemble.HistGradientBoostingRegressor`, a strong gradient
boosting baseline in the spirit of XGBoost/LightGBM.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import NormalDist
from typing import Protocol

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import HistGradientBoostingRegressor

from ..domain import Forecast, TimeSeriesWindow

FloatArray = npt.NDArray[np.float64]


class Regressor(Protocol):
    """Minimal scikit-learn-style regressor interface."""

    def fit(self, x: FloatArray, y: FloatArray) -> object:
        """Fit the model to design matrix ``x`` and target ``y``."""
        ...

    def predict(self, x: FloatArray) -> FloatArray:
        """Predict targets for design matrix ``x``."""
        ...


def _calendar_features(timestamp: datetime) -> list[float]:
    """Cyclical hour-of-day and day-of-week encodings for one timestamp."""
    hour = timestamp.hour + timestamp.minute / 60.0
    dow = float(timestamp.weekday())
    return [
        float(np.sin(2.0 * np.pi * hour / 24.0)),
        float(np.cos(2.0 * np.pi * hour / 24.0)),
        float(np.sin(2.0 * np.pi * dow / 7.0)),
        float(np.cos(2.0 * np.pi * dow / 7.0)),
    ]


class TabularForecaster:
    """Recursive lag-feature forecaster with growing prediction intervals.

    Parameters
    ----------
    n_lags:
        Number of trailing values used as predictors.
    add_calendar:
        Whether to append cyclical calendar features for each target time.
    interval_level:
        Nominal coverage of the prediction interval.
    model:
        Optional pre-configured regressor implementing ``fit``/``predict``.
        Defaults to a seeded :class:`HistGradientBoostingRegressor`.
    random_state:
        Seed used for the default model when ``model`` is not supplied.
    """

    def __init__(
        self,
        n_lags: int = 24,
        add_calendar: bool = True,
        interval_level: float = 0.9,
        model: Regressor | None = None,
        random_state: int = 42,
    ) -> None:
        if n_lags < 1:
            raise ValueError("n_lags must be at least 1.")
        if not 0.0 < interval_level < 1.0:
            raise ValueError("interval_level must be in (0, 1).")
        self._n_lags = n_lags
        self._add_calendar = add_calendar
        self._interval_level = interval_level
        self._model: Regressor = model or HistGradientBoostingRegressor(
            random_state=random_state
        )
        self._fitted = False
        self._residual_std = 0.0
        self._recent: list[float] = []
        self._last_time = datetime(1970, 1, 1)
        self._step = timedelta(minutes=1)
        self._sensor_id = ""

    def _row(self, lags: list[float], target_time: datetime) -> list[float]:
        """Build one feature row from ordered lags and the target time.

        ``lags`` is chronological (oldest first); the model sees lag-1 first.
        """
        feats = list(reversed(lags))
        if self._add_calendar:
            feats.extend(_calendar_features(target_time))
        return feats

    def _design_matrix(
        self, values: list[float], timestamps: list[datetime]
    ) -> tuple[FloatArray, FloatArray]:
        """Build the supervised (X, y) matrices from a single series."""
        rows: list[list[float]] = []
        targets: list[float] = []
        for i in range(self._n_lags, len(values)):
            rows.append(self._row(values[i - self._n_lags : i], timestamps[i]))
            targets.append(values[i])
        return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)

    def fit(self, window: TimeSeriesWindow) -> TabularForecaster:
        """Fit the regressor and capture state needed for recursive forecasts."""
        if len(window) <= self._n_lags:
            raise ValueError("window must contain more points than n_lags.")

        x, y = self._design_matrix(window.values, window.timestamps)
        self._model.fit(x, y)

        residuals = y - np.asarray(self._model.predict(x), dtype=float)
        self._residual_std = float(np.std(residuals))
        self._recent = list(window.values[-self._n_lags :])
        self._last_time = window.timestamps[-1]
        self._step = (
            window.timestamps[-1] - window.timestamps[-2]
            if len(window) >= 2
            else timedelta(minutes=1)
        )
        self._sensor_id = window.sensor_id
        self._fitted = True
        return self

    def forecast(self, horizon: int) -> Forecast:
        """Produce a recursive ``horizon``-step forecast with intervals."""
        if not self._fitted:
            raise RuntimeError("forecaster must be fit before forecasting.")
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")

        z = NormalDist().inv_cdf(0.5 + self._interval_level / 2.0)
        lags = list(self._recent)
        timestamps: list[datetime] = []
        mean: list[float] = []
        lower: list[float] = []
        upper: list[float] = []

        for h in range(horizon):
            target_time = self._last_time + self._step * (h + 1)
            row = np.asarray([self._row(lags, target_time)], dtype=float)
            point = float(self._model.predict(row)[0])

            # Uncertainty grows with the square root of the horizon step.
            half = z * self._residual_std * float(np.sqrt(h + 1))
            timestamps.append(target_time)
            mean.append(point)
            lower.append(point - half)
            upper.append(point + half)

            lags = [*lags[1:], point]

        return Forecast(
            sensor_id=self._sensor_id,
            timestamps=timestamps,
            mean=mean,
            lower=lower,
            upper=upper,
            interval_level=self._interval_level,
        )
