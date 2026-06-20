"""Alert sinks for the streaming pipeline.

A sink is the egress point for alerts produced by the streaming processor.
Keeping it behind a small interface lets the same processor feed an in-memory
buffer in tests, a logger in development, or a queue/webhook in production.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from ..domain import Alert

logger = logging.getLogger("sensor_intelligence.streaming")


class AlertSink(Protocol):
    """Destination for emitted alerts."""

    def emit(self, alerts: list[Alert]) -> None:
        """Deliver a batch of alerts."""
        ...


class InMemoryAlertSink:
    """Collect alerts in memory; useful for tests and batch runs."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def emit(self, alerts: list[Alert]) -> None:
        """Append alerts to the in-memory list."""
        self.alerts.extend(alerts)


class CallableAlertSink:
    """Adapt an arbitrary callback into an :class:`AlertSink`.

    Parameters
    ----------
    callback:
        Function invoked with each emitted batch of alerts.
    """

    def __init__(self, callback: Callable[[list[Alert]], None]) -> None:
        self._callback = callback

    def emit(self, alerts: list[Alert]) -> None:
        """Forward alerts to the wrapped callback."""
        self._callback(alerts)


class LoggingAlertSink:
    """Log each alert at a configurable level."""

    def __init__(self, level: int = logging.WARNING) -> None:
        self._level = level

    def emit(self, alerts: list[Alert]) -> None:
        """Log one line per alert."""
        for alert in alerts:
            logger.log(self._level, "[%s] %s", alert.severity.value, alert.message)
