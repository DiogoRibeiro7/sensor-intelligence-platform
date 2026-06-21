"""A reusable reference sensor fleet.

Notebooks, examples, and demos repeatedly need a *plausible* multi-channel
asset to simulate rather than a single hand-rolled sensor. :func:`default_fleet`
returns the generative specs for one such asset — a motor-driven pump skid —
so every consumer simulates the same eight channels with consistent units and
realistic scales. The specs are deliberately diverse (slow drifts, strong daily
cycles, near-stationary noise) so the feature, forecasting, anomaly, and drift
layers each have something interesting to work on.
"""

from __future__ import annotations

from .simulator import SensorSpec

#: Generative parameters for the reference pump-skid asset. Each entry pairs a
#: descriptive ``sensor_id`` with a baseline level, optional daily trend, daily
#: seasonal amplitude, AR(1) noise, and the channel's physical unit.
_FLEET: tuple[SensorSpec, ...] = (
    SensorSpec(
        "temperature", baseline=62.0, trend_per_day=0.15, daily_amplitude=7.0,
        noise_std=0.6, noise_ar=0.40, unit="C",
    ),
    SensorSpec(
        "vibration", baseline=0.42, trend_per_day=0.0, daily_amplitude=0.04,
        noise_std=0.03, noise_ar=0.25, unit="g",
    ),
    SensorSpec(
        "pressure", baseline=101.3, trend_per_day=0.10, daily_amplitude=1.5,
        noise_std=0.40, noise_ar=0.30, unit="kPa",
    ),
    SensorSpec(
        "humidity", baseline=45.0, trend_per_day=-0.20, daily_amplitude=12.0,
        noise_std=1.20, noise_ar=0.50, unit="%RH",
    ),
    SensorSpec(
        "flow_rate", baseline=120.0, trend_per_day=0.0, daily_amplitude=18.0,
        noise_std=2.50, noise_ar=0.35, unit="m3/h",
    ),
    SensorSpec(
        "motor_current", baseline=14.0, trend_per_day=0.05, daily_amplitude=2.2,
        noise_std=0.30, noise_ar=0.30, unit="A",
    ),
    SensorSpec(
        "supply_voltage", baseline=400.0, trend_per_day=0.0, daily_amplitude=1.0,
        noise_std=1.50, noise_ar=0.20, unit="V",
    ),
    SensorSpec(
        "shaft_speed", baseline=1480.0, trend_per_day=0.0, daily_amplitude=6.0,
        noise_std=1.00, noise_ar=0.30, unit="rpm",
    ),
)


def default_fleet() -> list[SensorSpec]:
    """Return the eight reference sensor specs for a motor-driven pump skid.

    The channels are ``temperature`` (C), ``vibration`` (g), ``pressure`` (kPa),
    ``humidity`` (%RH), ``flow_rate`` (m3/h), ``motor_current`` (A),
    ``supply_voltage`` (V), and ``shaft_speed`` (rpm). A fresh list is returned
    on each call so callers may filter or extend it without mutating shared
    state; the :class:`SensorSpec` entries themselves are frozen.

    Returns
    -------
    list[SensorSpec]
        Generative specs ready to drop into a :class:`SimulationConfig`.
    """
    return list(_FLEET)
