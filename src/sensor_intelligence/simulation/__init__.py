"""Synthetic sensor data simulation."""

from __future__ import annotations

from .fleet import default_fleet
from .simulator import (
    AnomalyInjection,
    SensorSimulator,
    SensorSpec,
    SimulationConfig,
)

__all__ = [
    "AnomalyInjection",
    "SensorSimulator",
    "SensorSpec",
    "SimulationConfig",
    "default_fleet",
]
