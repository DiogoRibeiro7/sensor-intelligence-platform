from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from sensor_intelligence.anomaly import EwmaDetector, RobustZScoreDetector
from sensor_intelligence.domain import TimeSeriesWindow

_T0 = datetime(2024, 1, 1)


def _window(values: list[float]) -> TimeSeriesWindow:
    timestamps = [_T0 + timedelta(minutes=i) for i in range(len(values))]
    return TimeSeriesWindow(sensor_id="s1", timestamps=timestamps, values=values)


def _stable_with_spike(spike_index: int, spike_value: float) -> list[float]:
    rng = np.random.default_rng(0)
    values = (50.0 + rng.normal(0.0, 0.5, size=200)).tolist()
    values[spike_index] = spike_value
    return values


def test_ewma_flags_obvious_spike() -> None:
    values = _stable_with_spike(120, 80.0)
    anomalies = EwmaDetector(alpha=0.3, threshold=4.0).detect(_window(values))

    flagged = {a.timestamp for a in anomalies}
    assert (_T0 + timedelta(minutes=120)) in flagged
    assert all(a.method == "ewma" and a.score > 0 for a in anomalies)


def test_ewma_quiet_on_clean_series() -> None:
    rng = np.random.default_rng(1)
    values = (50.0 + rng.normal(0.0, 0.5, size=300)).tolist()
    anomalies = EwmaDetector(alpha=0.3, threshold=5.0).detect(_window(values))

    assert anomalies == []


def test_robust_zscore_flags_spike_and_reports_reason() -> None:
    values = _stable_with_spike(150, 90.0)
    anomalies = RobustZScoreDetector(window_size=50, threshold=3.5).detect(_window(values))

    assert any(a.timestamp == _T0 + timedelta(minutes=150) for a in anomalies)
    assert all(a.reason_codes for a in anomalies)


def test_robust_zscore_handles_short_series() -> None:
    anomalies = RobustZScoreDetector(window_size=50).detect(_window([1.0, 2.0, 3.0]))
    assert anomalies == []


def test_empty_window_returns_no_anomalies() -> None:
    empty = TimeSeriesWindow(sensor_id="s1", timestamps=[], values=[])
    assert EwmaDetector().detect(empty) == []
    assert RobustZScoreDetector().detect(empty) == []
