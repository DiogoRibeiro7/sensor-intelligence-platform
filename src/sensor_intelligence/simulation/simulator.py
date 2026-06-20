"""Synthetic multivariate sensor data simulator.

The simulator produces reproducible multivariate time series with realistic
structure — trend, daily seasonality, autocorrelated noise, and optional
injected anomalies — so the rest of the platform can be developed and tested
without a live data source. Output is a tidy :class:`pandas.DataFrame` that
mirrors what an ingestion adapter would yield.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SensorSpec:
    """Generative parameters for a single sensor channel.

    Parameters
    ----------
    sensor_id:
        Identifier emitted in the output frame.
    baseline:
        Mean level of the series before trend and seasonality.
    trend_per_day:
        Linear drift added per simulated day.
    daily_amplitude:
        Amplitude of the 24-hour seasonal cycle.
    noise_std:
        Standard deviation of the innovation driving the noise process.
    noise_ar:
        Lag-1 autoregressive coefficient for the noise, in ``[0, 1)``.
    unit:
        Optional unit label carried through to readings.
    """

    sensor_id: str
    baseline: float = 50.0
    trend_per_day: float = 0.0
    daily_amplitude: float = 5.0
    noise_std: float = 1.0
    noise_ar: float = 0.3
    unit: str | None = None


@dataclass(frozen=True)
class AnomalyInjection:
    """Specification of a single injected anomaly.

    Parameters
    ----------
    sensor_id:
        Sensor to perturb.
    start_step:
        Index of the first affected sample.
    duration:
        Number of consecutive samples affected.
    magnitude:
        Additive offset applied across the affected span, in series units.
    kind:
        ``"spike"`` for a transient offset, ``"drift"`` for a ramp that grows
        linearly across the affected span.
    """

    sensor_id: str
    start_step: int
    duration: int = 1
    magnitude: float = 10.0
    kind: str = "spike"


@dataclass
class SimulationConfig:
    """Top-level configuration for a simulation run."""

    sensors: list[SensorSpec]
    start: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    n_steps: int = 1_440
    step_seconds: int = 60
    seed: int = 42
    anomalies: list[AnomalyInjection] = field(default_factory=list)


class SensorSimulator:
    """Generate reproducible multivariate sensor data.

    Parameters
    ----------
    config:
        Simulation configuration describing sensors, horizon, and anomalies.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config

    def _timestamps(self) -> list[datetime]:
        """Build the evenly spaced timestamp index."""
        step = timedelta(seconds=self._config.step_seconds)
        return [self._config.start + i * step for i in range(self._config.n_steps)]

    def _seasonal(self, spec: SensorSpec, seconds: FloatArray) -> FloatArray:
        """Return the deterministic trend-plus-seasonal component."""
        days = seconds / 86_400.0
        trend = spec.trend_per_day * days
        daily = spec.daily_amplitude * np.sin(2.0 * np.pi * days)
        result: FloatArray = spec.baseline + trend + daily
        return result

    def _noise(self, spec: SensorSpec, rng: np.random.Generator) -> FloatArray:
        """Return an AR(1) noise series of length ``n_steps``."""
        n = self._config.n_steps
        innovations = rng.normal(0.0, spec.noise_std, size=n)
        noise = np.empty(n, dtype=float)
        noise[0] = innovations[0]
        for i in range(1, n):
            noise[i] = spec.noise_ar * noise[i - 1] + innovations[i]
        return noise

    def _apply_anomalies(self, sensor_id: str, values: FloatArray) -> FloatArray:
        """Apply all injections targeting ``sensor_id`` in place-safe fashion."""
        out = values.copy()
        n = self._config.n_steps
        for inj in self._config.anomalies:
            if inj.sensor_id != sensor_id:
                continue
            start = max(0, inj.start_step)
            end = min(n, inj.start_step + max(1, inj.duration))
            if start >= end:
                continue
            span = end - start
            if inj.kind == "drift":
                ramp = np.linspace(0.0, inj.magnitude, span)
                out[start:end] += ramp
            else:
                out[start:end] += inj.magnitude
        return out

    def run(self) -> pd.DataFrame:
        """Generate the simulated dataset.

        Returns
        -------
        pandas.DataFrame
            Long-format frame with columns ``sensor_id``, ``timestamp``,
            ``value``, ``unit``, and a boolean ``is_anomaly`` flag marking
            samples touched by an injection.
        """
        timestamps = self._timestamps()
        seconds = (
            np.arange(self._config.n_steps) * self._config.step_seconds
        ).astype(np.float64)
        rng = np.random.default_rng(self._config.seed)

        frames: list[pd.DataFrame] = []
        for spec in self._config.sensors:
            clean = self._seasonal(spec, seconds) + self._noise(spec, rng)
            dirty = self._apply_anomalies(spec.sensor_id, clean)
            is_anomaly = ~np.isclose(dirty, clean)
            frames.append(
                pd.DataFrame(
                    {
                        "sensor_id": spec.sensor_id,
                        "timestamp": timestamps,
                        "value": dirty,
                        "unit": spec.unit,
                        "is_anomaly": is_anomaly,
                    }
                )
            )

        result = pd.concat(frames, ignore_index=True)
        return result.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
