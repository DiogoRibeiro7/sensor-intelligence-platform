"""Core domain models shared across the platform.

These models define the vocabulary of the system: the raw sensor reading, a
windowed slice of a series, a forecast with uncertainty, a detected anomaly,
and an operator-facing alert. Keeping them in one place lets the data,
modelling, and API layers exchange validated objects instead of loose dicts.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class AlertSeverity(StrEnum):
    """Operational severity for an alert."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SensorReading(BaseModel):
    """A single timestamped measurement from one sensor channel.

    Parameters
    ----------
    sensor_id:
        Stable identifier for the physical or logical sensor.
    timestamp:
        Acquisition time of the measurement.
    value:
        Measured value in the sensor's native unit.
    unit:
        Optional unit label, useful for reporting and validation.
    quality:
        Data-quality score in ``[0, 1]``; ``1.0`` means fully trusted.
    """

    sensor_id: str = Field(min_length=1)
    timestamp: datetime
    value: float
    unit: str | None = Field(default=None)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)


class TimeSeriesWindow(BaseModel):
    """An ordered slice of readings for a single sensor.

    The window is the unit of work for feature engineering and inference. It
    validates that timestamps and values are aligned and chronologically
    ordered so downstream code can assume a clean contract.
    """

    sensor_id: str = Field(min_length=1)
    timestamps: list[datetime]
    values: list[float]

    @model_validator(mode="after")
    def _check_alignment(self) -> TimeSeriesWindow:
        """Ensure timestamps and values are aligned and time-ordered."""
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have equal length.")
        if any(b < a for a, b in zip(self.timestamps, self.timestamps[1:], strict=False)):
            raise ValueError("timestamps must be non-decreasing.")
        return self

    def __len__(self) -> int:
        """Return the number of readings in the window."""
        return len(self.values)


class Forecast(BaseModel):
    """A multi-step forecast with optional prediction intervals.

    Parameters
    ----------
    sensor_id:
        Sensor the forecast applies to.
    timestamps:
        Future timestamps the forecast covers.
    mean:
        Point forecast for each timestamp.
    lower / upper:
        Optional prediction-interval bounds matching ``mean`` element-wise.
    interval_level:
        Nominal coverage of the interval (for example ``0.9`` for 90%).
    """

    sensor_id: str = Field(min_length=1)
    timestamps: list[datetime]
    mean: list[float]
    lower: list[float] | None = Field(default=None)
    upper: list[float] | None = Field(default=None)
    interval_level: float = Field(default=0.9, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_lengths(self) -> Forecast:
        """Ensure all present series share the forecast horizon length."""
        horizon = len(self.timestamps)
        if len(self.mean) != horizon:
            raise ValueError("mean must match the number of timestamps.")
        for name, series in (("lower", self.lower), ("upper", self.upper)):
            if series is not None and len(series) != horizon:
                raise ValueError(f"{name} must match the number of timestamps.")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("lower and upper bounds must be provided together.")
        return self


class Anomaly(BaseModel):
    """A detected anomaly at a point in a sensor series.

    Parameters
    ----------
    sensor_id:
        Sensor the anomaly was detected on.
    timestamp:
        Time of the anomalous observation.
    observed:
        The value that triggered the detection.
    expected:
        The value the detector expected, when available.
    score:
        Non-negative anomaly score; higher means more anomalous.
    method:
        Identifier of the detector that produced the anomaly.
    reason_codes:
        Human-readable explanations attached to the detection.
    """

    sensor_id: str = Field(min_length=1)
    timestamp: datetime
    observed: float
    expected: float | None = Field(default=None)
    score: float = Field(ge=0.0)
    method: str = Field(min_length=1)
    reason_codes: list[str] = Field(default_factory=list)


class Alert(BaseModel):
    """An operator-facing alert derived from one or more anomalies."""

    sensor_id: str = Field(min_length=1)
    timestamp: datetime
    severity: AlertSeverity
    message: str = Field(min_length=1)
    anomalies: list[Anomaly] = Field(default_factory=list)
