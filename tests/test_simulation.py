from __future__ import annotations

from datetime import datetime

from sensor_intelligence.simulation import (
    AnomalyInjection,
    SensorSimulator,
    SensorSpec,
    SimulationConfig,
)


def _config(**overrides: object) -> SimulationConfig:
    base = SimulationConfig(
        sensors=[
            SensorSpec(sensor_id="temp", baseline=20.0, daily_amplitude=3.0),
            SensorSpec(sensor_id="vibration", baseline=0.5, noise_std=0.05),
        ],
        n_steps=120,
        start=datetime(2024, 1, 1),
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_output_shape_and_columns() -> None:
    df = SensorSimulator(_config()).run()

    assert set(df.columns) == {"sensor_id", "timestamp", "value", "unit", "is_anomaly"}
    assert len(df) == 120 * 2
    assert set(df["sensor_id"].unique()) == {"temp", "vibration"}


def test_simulation_is_reproducible() -> None:
    a = SensorSimulator(_config(seed=7)).run()
    b = SensorSimulator(_config(seed=7)).run()

    assert a["value"].tolist() == b["value"].tolist()


def test_different_seeds_differ() -> None:
    a = SensorSimulator(_config(seed=1)).run()
    b = SensorSimulator(_config(seed=2)).run()

    assert a["value"].tolist() != b["value"].tolist()


def test_spike_injection_flags_and_shifts_values() -> None:
    cfg = _config(
        anomalies=[
            AnomalyInjection(sensor_id="temp", start_step=50, duration=3, magnitude=25.0)
        ]
    )
    df = SensorSimulator(cfg).run()
    temp = df[df["sensor_id"] == "temp"].reset_index(drop=True)

    flagged = temp.index[temp["is_anomaly"]].tolist()
    assert flagged == [50, 51, 52]
    assert temp.loc[51, "value"] > temp.loc[40, "value"]


def test_drift_injection_ramps() -> None:
    cfg = _config(
        anomalies=[
            AnomalyInjection(
                sensor_id="vibration", start_step=10, duration=20, magnitude=5.0, kind="drift"
            )
        ]
    )
    df = SensorSimulator(cfg).run()
    vib = df[df["sensor_id"] == "vibration"].reset_index(drop=True)

    assert bool(vib.loc[29, "is_anomaly"])
    # Ramp grows across the span: later offset exceeds earlier offset.
    assert vib.loc[29, "value"] > vib.loc[12, "value"]
