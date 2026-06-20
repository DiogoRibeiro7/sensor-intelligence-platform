from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from sensor_intelligence.anomaly import RobustZScoreDetector
from sensor_intelligence.domain import SensorReading
from sensor_intelligence.streaming import (
    CallableAlertSink,
    InMemoryAlertSink,
    StreamingConfig,
    StreamProcessor,
    iter_readings,
)

_T0 = datetime(2024, 1, 1)


def _readings(values: list[float], sensor_id: str = "s1") -> list[SensorReading]:
    return [
        SensorReading(sensor_id=sensor_id, timestamp=_T0 + timedelta(minutes=i), value=v)
        for i, v in enumerate(values)
    ]


def _stable_with_spike(n: int, spike_index: int, spike_value: float) -> list[float]:
    rng = np.random.default_rng(0)
    values = (50.0 + rng.normal(0.0, 0.5, size=n)).tolist()
    values[spike_index] = spike_value
    return values


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="min_window"):
        StreamingConfig(window_size=10, min_window=50)
    with pytest.raises(ValueError, match="batch_size"):
        StreamingConfig(batch_size=0)


def test_bounded_buffer_constant_memory() -> None:
    sink = InMemoryAlertSink()
    proc = StreamProcessor(
        RobustZScoreDetector(), sink, config=StreamingConfig(window_size=100, min_window=30)
    )
    proc.run(_readings([50.0] * 500))

    # Buffer never exceeds the configured window size despite 500 readings.
    buffer = proc._buffer("s1")
    assert len(buffer) == 100


def test_detects_and_emits_spike() -> None:
    sink = InMemoryAlertSink()
    proc = StreamProcessor(
        RobustZScoreDetector(window_size=50, threshold=3.5),
        sink,
        config=StreamingConfig(window_size=300, min_window=60, batch_size=25),
    )
    total = proc.run(_readings(_stable_with_spike(300, 200, 90.0)))

    assert total >= 1
    assert len(sink.alerts) >= 1
    assert all(a.sensor_id == "s1" for a in sink.alerts)


def test_spike_emitted_only_once() -> None:
    sink = InMemoryAlertSink()
    proc = StreamProcessor(
        RobustZScoreDetector(window_size=50, threshold=3.5),
        sink,
        config=StreamingConfig(window_size=300, min_window=60, batch_size=10),
    )
    proc.run(_readings(_stable_with_spike(300, 150, 95.0)))

    spike_ts = _T0 + timedelta(minutes=150)
    emitted_at_spike = [a for a in sink.alerts if a.timestamp == spike_ts]
    assert len(emitted_at_spike) == 1  # de-duplicated across sliding batches


def test_multiple_sensors_isolated() -> None:
    sink = InMemoryAlertSink()
    proc = StreamProcessor(
        RobustZScoreDetector(window_size=50, threshold=3.5),
        sink,
        config=StreamingConfig(window_size=300, min_window=60, batch_size=20),
    )
    clean = _readings([50.0] * 200, sensor_id="quiet")
    spiky = _readings(_stable_with_spike(200, 150, 90.0), sensor_id="noisy")
    # Interleave the two sensors' streams.
    merged = [r for pair in zip(clean, spiky, strict=True) for r in pair]
    proc.run(merged)

    sensors = {a.sensor_id for a in sink.alerts}
    assert "noisy" in sensors
    assert "quiet" not in sensors


def test_callable_sink_receives_alerts() -> None:
    received: list[int] = []
    sink = CallableAlertSink(lambda alerts: received.append(len(alerts)))
    proc = StreamProcessor(
        RobustZScoreDetector(window_size=50, threshold=3.5),
        sink,
        config=StreamingConfig(window_size=300, min_window=60, batch_size=300),
    )
    proc.run(_readings(_stable_with_spike(300, 200, 90.0)))

    assert sum(received) >= 1


def test_iter_readings_adapter() -> None:
    rows = [("s1", _T0, 1.0), ("s1", _T0 + timedelta(minutes=1), 2.0)]
    readings = list(iter_readings(rows))

    assert len(readings) == 2
    assert readings[0].sensor_id == "s1"
    assert readings[1].value == 2.0
