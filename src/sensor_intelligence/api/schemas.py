"""Request and response schemas for the inference API.

These wrap the domain models in transport-friendly payloads. A
:class:`WindowPayload` is the common input — a single sensor's series — which
converts to a validated :class:`~sensor_intelligence.domain.TimeSeriesWindow`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..domain import Alert, Anomaly, Forecast, TimeSeriesWindow
from ..drift import DriftResult


class WindowPayload(BaseModel):
    """A single sensor's series as submitted to the API."""

    sensor_id: str = Field(min_length=1)
    timestamps: list[datetime]
    values: list[float]

    def to_window(self) -> TimeSeriesWindow:
        """Convert to a validated domain window."""
        return TimeSeriesWindow(
            sensor_id=self.sensor_id, timestamps=self.timestamps, values=self.values
        )


class AnomalyRequest(BaseModel):
    """Request to detect anomalies on a window."""

    window: WindowPayload
    method: Literal["ewma", "robust_zscore", "cusum"] = "ewma"
    threshold: float | None = Field(default=None, gt=0.0)


class AnomalyResponse(BaseModel):
    """Detected anomalies and the alerts derived from them."""

    anomalies: list[Anomaly]
    alerts: list[Alert]


class ForecastRequest(BaseModel):
    """Request to forecast future values of a window."""

    window: WindowPayload
    horizon: int = Field(gt=0, le=10_000)
    method: Literal["tabular", "seasonal_naive", "rolling_mean"] = "tabular"
    period: int | None = Field(default=None, gt=0)
    n_lags: int | None = Field(default=None, gt=0)


class ForecastResponse(BaseModel):
    """A forecast for the requested horizon."""

    forecast: Forecast


class DriftRequest(BaseModel):
    """Request to evaluate drift between a reference and current sample."""

    reference: list[float] = Field(min_length=1)
    current: list[float] = Field(min_length=1)
    method: Literal["psi", "error"] = "psi"
    threshold: float | None = Field(default=None, gt=0.0)


class DriftResponse(BaseModel):
    """The computed drift result."""

    result: DriftResult
