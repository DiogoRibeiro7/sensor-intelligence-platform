from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from sensor_intelligence.domain import (
    Alert,
    AlertSeverity,
    Anomaly,
    Forecast,
    SensorReading,
    TimeSeriesWindow,
)

_T0 = datetime(2024, 1, 1)


def _times(n: int) -> list[datetime]:
    return [_T0 + timedelta(minutes=i) for i in range(n)]


def test_sensor_reading_quality_bounds() -> None:
    SensorReading(sensor_id="s1", timestamp=_T0, value=1.0, quality=0.5)
    with pytest.raises(ValidationError):
        SensorReading(sensor_id="s1", timestamp=_T0, value=1.0, quality=1.5)


def test_window_rejects_misaligned_lengths() -> None:
    with pytest.raises(ValidationError):
        TimeSeriesWindow(sensor_id="s1", timestamps=_times(3), values=[1.0, 2.0])


def test_window_rejects_unordered_timestamps() -> None:
    ts = _times(3)
    ts[1], ts[2] = ts[2], ts[1]
    with pytest.raises(ValidationError):
        TimeSeriesWindow(sensor_id="s1", timestamps=ts, values=[1.0, 2.0, 3.0])


def test_window_len() -> None:
    window = TimeSeriesWindow(sensor_id="s1", timestamps=_times(4), values=[1.0, 2.0, 3.0, 4.0])
    assert len(window) == 4


def test_forecast_requires_paired_bounds() -> None:
    with pytest.raises(ValidationError):
        Forecast(sensor_id="s1", timestamps=_times(2), mean=[1.0, 2.0], lower=[0.0, 1.0])


def test_forecast_with_intervals() -> None:
    fc = Forecast(
        sensor_id="s1",
        timestamps=_times(2),
        mean=[1.0, 2.0],
        lower=[0.0, 1.0],
        upper=[2.0, 3.0],
    )
    assert fc.interval_level == 0.9


def test_alert_carries_anomalies() -> None:
    anomaly = Anomaly(sensor_id="s1", timestamp=_T0, observed=99.0, score=4.2, method="zscore")
    alert = Alert(
        sensor_id="s1",
        timestamp=_T0,
        severity=AlertSeverity.CRITICAL,
        message="threshold exceeded",
        anomalies=[anomaly],
    )
    assert alert.severity is AlertSeverity.CRITICAL
    assert alert.anomalies[0].method == "zscore"
