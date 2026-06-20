"""Simulated streaming detection with backpressure-safe batching.

The processor consumes a stream of :class:`~sensor_intelligence.domain.SensorReading`
objects, maintains a bounded sliding window per sensor (constant memory — the
backpressure guarantee), and runs the detector once per fixed-size batch rather
than per reading so inference cost per tick is bounded. Alerts are de-duplicated
by timestamp so a finding is emitted once even as the window slides past it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime

from ..alerting import AlertPolicy
from ..anomaly import AnomalyDetector
from ..domain import Alert, SensorReading, TimeSeriesWindow
from .sinks import AlertSink


@dataclass(frozen=True)
class StreamingConfig:
    """Tuning for the streaming processor.

    Parameters
    ----------
    window_size:
        Maximum readings retained per sensor (the bounded sliding window).
    min_window:
        Minimum readings before detection runs for a sensor.
    batch_size:
        Number of readings ingested before an inference tick fires.
    """

    window_size: int = 500
    min_window: int = 30
    batch_size: int = 50

    def __post_init__(self) -> None:
        """Validate the sizing parameters."""
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if not 1 <= self.min_window <= self.window_size:
            raise ValueError("min_window must be in [1, window_size].")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")


class StreamProcessor:
    """Run an anomaly detector over a simulated reading stream.

    Parameters
    ----------
    detector:
        Detector applied to each sensor's sliding window.
    sink:
        Destination for emitted alerts.
    policy:
        Severity policy converting anomalies to alerts.
    config:
        Streaming configuration.
    """

    def __init__(
        self,
        detector: AnomalyDetector,
        sink: AlertSink,
        policy: AlertPolicy | None = None,
        config: StreamingConfig | None = None,
    ) -> None:
        self._detector = detector
        self._sink = sink
        self._policy = policy or AlertPolicy()
        self._config = config or StreamingConfig()
        self._buffers: dict[str, deque[tuple[datetime, float]]] = {}
        self._last_emitted: dict[str, datetime] = {}

    def _buffer(self, sensor_id: str) -> deque[tuple[datetime, float]]:
        """Return (creating if needed) the bounded buffer for a sensor."""
        if sensor_id not in self._buffers:
            self._buffers[sensor_id] = deque(maxlen=self._config.window_size)
        return self._buffers[sensor_id]

    def _detect_for(self, sensor_id: str) -> list[Alert]:
        """Run detection on a sensor's window and return fresh alerts."""
        buffer = self._buffer(sensor_id)
        if len(buffer) < self._config.min_window:
            return []

        timestamps = [t for t, _ in buffer]
        values = [v for _, v in buffer]
        window = TimeSeriesWindow(sensor_id=sensor_id, timestamps=timestamps, values=values)

        last = self._last_emitted.get(sensor_id)
        anomalies = [
            a for a in self._detector.detect(window) if last is None or a.timestamp > last
        ]
        if not anomalies:
            return []

        self._last_emitted[sensor_id] = max(a.timestamp for a in anomalies)
        return self._policy.from_anomalies(anomalies)

    def process_batch(self, readings: Iterable[SensorReading]) -> list[Alert]:
        """Ingest a batch, run detection for touched sensors, and emit alerts."""
        touched: list[str] = []
        seen: set[str] = set()
        for reading in readings:
            self._buffer(reading.sensor_id).append((reading.timestamp, reading.value))
            if reading.sensor_id not in seen:
                seen.add(reading.sensor_id)
                touched.append(reading.sensor_id)

        alerts: list[Alert] = []
        for sensor_id in touched:
            alerts.extend(self._detect_for(sensor_id))
        if alerts:
            self._sink.emit(alerts)
        return alerts

    def run(self, stream: Iterable[SensorReading]) -> int:
        """Consume a stream in fixed-size batches; return total alerts emitted.

        Batching bounds the work done per inference tick regardless of how fast
        the upstream produces readings.
        """
        total = 0
        batch: list[SensorReading] = []
        for reading in stream:
            batch.append(reading)
            if len(batch) >= self._config.batch_size:
                total += len(self.process_batch(batch))
                batch = []
        if batch:
            total += len(self.process_batch(batch))
        return total


def iter_readings(
    rows: Iterable[tuple[str, datetime, float]],
) -> Iterator[SensorReading]:
    """Adapt ``(sensor_id, timestamp, value)`` tuples into a reading stream."""
    for sensor_id, timestamp, value in rows:
        yield SensorReading(sensor_id=sensor_id, timestamp=timestamp, value=value)
