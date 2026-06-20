from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from sensor_intelligence.anomaly import CusumChangePointDetector
from sensor_intelligence.domain import TimeSeriesWindow

_T0 = datetime(2024, 1, 1)


def _window(values: list[float]) -> TimeSeriesWindow:
    timestamps = [_T0 + timedelta(minutes=i) for i in range(len(values))]
    return TimeSeriesWindow(sensor_id="s1", timestamps=timestamps, values=values)


def test_detects_level_shift() -> None:
    rng = np.random.default_rng(0)
    before = rng.normal(10.0, 0.5, size=200)
    after = rng.normal(15.0, 0.5, size=200)
    values = np.concatenate([before, after]).tolist()

    cps = CusumChangePointDetector(slack=0.5, threshold=8.0, reference_size=150).detect(
        _window(values)
    )

    assert len(cps) == 1, "a single regime shift should yield one change point"
    first = cps[0]
    # The shift starts at index 200; CUSUM should fire shortly after.
    assert 200 <= (first.timestamp - _T0).total_seconds() / 60 <= 235
    assert "increase" in first.reason_codes[0]
    assert first.method == "cusum"


def test_quiet_on_stationary_series() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(5.0, 0.5, size=400).tolist()

    cps = CusumChangePointDetector(slack=0.5, threshold=6.0).detect(_window(values))

    assert cps == []


def test_constant_series_returns_no_change_points() -> None:
    cps = CusumChangePointDetector().detect(_window([7.0] * 100))
    assert cps == []


def test_detects_downward_shift_direction() -> None:
    rng = np.random.default_rng(2)
    values = np.concatenate(
        [rng.normal(20.0, 0.5, size=150), rng.normal(12.0, 0.5, size=150)]
    ).tolist()

    cps = CusumChangePointDetector(reference_size=120, threshold=8.0).detect(_window(values))

    assert cps
    assert "decrease" in cps[0].reason_codes[0]
