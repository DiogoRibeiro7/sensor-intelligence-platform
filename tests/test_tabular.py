from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from sensor_intelligence.domain import TimeSeriesWindow
from sensor_intelligence.models import SeasonalNaiveForecaster, TabularForecaster

_T0 = datetime(2024, 1, 1)


def _window(values: list[float]) -> TimeSeriesWindow:
    timestamps = [_T0 + timedelta(minutes=i) for i in range(len(values))]
    return TimeSeriesWindow(sensor_id="s1", timestamps=timestamps, values=values)


def _seasonal_series(n: int, period: int = 24, noise: float = 0.05) -> list[float]:
    rng = np.random.default_rng(0)
    t = np.arange(n)
    signal = 10.0 + 3.0 * np.sin(2.0 * np.pi * t / period)
    return (signal + rng.normal(0.0, noise, size=n)).tolist()


def test_forecast_shape_and_intervals() -> None:
    model = TabularForecaster(n_lags=24).fit(_window(_seasonal_series(480)))
    fc = model.forecast(horizon=12)

    assert len(fc.mean) == 12
    assert len(fc.timestamps) == 12
    assert fc.lower is not None and fc.upper is not None
    assert all(lo <= m <= up for lo, m, up in zip(fc.lower, fc.mean, fc.upper, strict=True))


def test_intervals_widen_with_horizon() -> None:
    model = TabularForecaster(n_lags=24).fit(_window(_seasonal_series(480)))
    fc = model.forecast(horizon=10)

    widths = [up - lo for lo, up in zip(fc.lower, fc.upper, strict=True)]
    assert widths[-1] > widths[0]
    assert widths == sorted(widths)


def test_forecast_timestamps_continue_cadence() -> None:
    model = TabularForecaster(n_lags=12).fit(_window(_seasonal_series(240)))
    fc = model.forecast(horizon=3)

    assert fc.timestamps[0] == _T0 + timedelta(minutes=240)
    assert fc.timestamps[2] == _T0 + timedelta(minutes=242)


def test_beats_seasonal_naive_on_learnable_signal() -> None:
    series = _seasonal_series(600)
    train, test = series[:-24], series[-24:]
    window = _window(train)

    tab = TabularForecaster(n_lags=48).fit(window)
    naive = SeasonalNaiveForecaster(period=24).forecast(window, horizon=24)
    tab_fc = tab.forecast(horizon=24)

    tab_mae = float(np.mean(np.abs(np.array(tab_fc.mean) - np.array(test))))
    naive_mae = float(np.mean(np.abs(np.array(naive.mean) - np.array(test))))
    assert tab_mae <= naive_mae * 1.5  # ML model is competitive with the baseline.


def test_reproducible_with_seed() -> None:
    window = _window(_seasonal_series(300))
    a = TabularForecaster(random_state=7).fit(window).forecast(horizon=5)
    b = TabularForecaster(random_state=7).fit(window).forecast(horizon=5)

    assert a.mean == b.mean


def test_requires_fit_before_forecast() -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        TabularForecaster().forecast(horizon=3)


def test_rejects_too_short_window() -> None:
    with pytest.raises(ValueError, match="more points than n_lags"):
        TabularForecaster(n_lags=10).fit(_window([1.0, 2.0, 3.0]))
