from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from sensor_intelligence.domain import TimeSeriesWindow
from sensor_intelligence.models import RollingMeanForecaster, SeasonalNaiveForecaster

_T0 = datetime(2024, 1, 1)


def _window(values: list[float]) -> TimeSeriesWindow:
    timestamps = [_T0 + timedelta(minutes=i) for i in range(len(values))]
    return TimeSeriesWindow(sensor_id="s1", timestamps=timestamps, values=values)


def test_seasonal_naive_repeats_prior_season() -> None:
    period = 24
    base = list(np.sin(np.linspace(0, 2 * np.pi, period)))
    window = _window(base * 4)

    fc = SeasonalNaiveForecaster(period=period).forecast(window, horizon=period)

    np.testing.assert_allclose(fc.mean, base, atol=1e-9)
    assert len(fc.timestamps) == period
    assert fc.lower is not None and fc.upper is not None


def test_seasonal_naive_requires_full_season() -> None:
    with pytest.raises(ValueError, match="full season"):
        SeasonalNaiveForecaster(period=50).forecast(_window([1.0, 2.0, 3.0]), horizon=5)


def test_seasonal_forecast_timestamps_continue_cadence() -> None:
    window = _window([float(i) for i in range(48)])
    fc = SeasonalNaiveForecaster(period=24).forecast(window, horizon=3)

    assert fc.timestamps[0] == _T0 + timedelta(minutes=48)
    assert fc.timestamps[1] == _T0 + timedelta(minutes=49)


def test_rolling_mean_is_flat_at_trailing_mean() -> None:
    window = _window([10.0] * 20 + [20.0] * 10)
    fc = RollingMeanForecaster(window_size=10).forecast(window, horizon=4)

    assert fc.mean == pytest.approx([20.0, 20.0, 20.0, 20.0])
    assert all(lo <= m <= up for lo, m, up in zip(fc.lower, fc.mean, fc.upper, strict=True))


def test_interval_widens_with_noise() -> None:
    quiet = RollingMeanForecaster(window_size=10).forecast(_window([5.0] * 30), horizon=1)
    noisy_values = (5.0 + np.random.default_rng(0).normal(0, 3, size=30)).tolist()
    noisy = RollingMeanForecaster(window_size=10).forecast(_window(noisy_values), horizon=1)

    quiet_width = quiet.upper[0] - quiet.lower[0]
    noisy_width = noisy.upper[0] - noisy.lower[0]
    assert noisy_width > quiet_width
