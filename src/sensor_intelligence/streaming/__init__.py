"""Simulated streaming detection pipeline."""

from __future__ import annotations

from .engine import StreamingConfig, StreamProcessor, iter_readings
from .sinks import (
    AlertSink,
    CallableAlertSink,
    InMemoryAlertSink,
    LoggingAlertSink,
)

__all__ = [
    "AlertSink",
    "CallableAlertSink",
    "InMemoryAlertSink",
    "LoggingAlertSink",
    "StreamProcessor",
    "StreamingConfig",
    "iter_readings",
]
