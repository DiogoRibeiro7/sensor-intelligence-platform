"""End-to-end walkthrough of the sensor intelligence platform.

Runs the full pipeline on synthetic data and prints a report:

1. Simulate multivariate sensor data with an injected spike and a drift ramp.
2. Engineer features (rolling, lag, calendar, health).
3. Forecast a sensor with the tabular ML model and prediction intervals.
4. Detect point anomalies (robust z-score) and regime shifts (CUSUM).
5. Monitor distribution drift (PSI) between an early and a late window.
6. Turn detections into severity-ranked alerts and render a Markdown report.

Run it with::

    python examples/end_to_end.py
"""

from __future__ import annotations

import sys

import numpy as np

from sensor_intelligence.alerting import AlertPolicy
from sensor_intelligence.anomaly import CusumChangePointDetector, RobustZScoreDetector
from sensor_intelligence.domain import TimeSeriesWindow
from sensor_intelligence.drift import PsiDriftDetector
from sensor_intelligence.features import FeatureBuilder
from sensor_intelligence.models import TabularForecaster
from sensor_intelligence.reporting import build_report
from sensor_intelligence.simulation import (
    AnomalyInjection,
    SensorSimulator,
    SensorSpec,
    SimulationConfig,
)


def main() -> None:
    """Run the pipeline and print a report."""
    # Markdown reports use Unicode glyphs; ensure UTF-8 output on any console.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. Simulate ------------------------------------------------------------
    config = SimulationConfig(
        sensors=[
            SensorSpec(sensor_id="temperature", baseline=60.0, daily_amplitude=8.0, unit="C"),
            SensorSpec(sensor_id="vibration", baseline=0.4, noise_std=0.05, unit="g"),
        ],
        n_steps=720,
        seed=7,
        anomalies=[
            AnomalyInjection(sensor_id="temperature", start_step=420, magnitude=28.0),
            AnomalyInjection(
                sensor_id="vibration", start_step=500, duration=40, magnitude=0.6, kind="drift"
            ),
        ],
    )
    frame = SensorSimulator(config).run()
    print(f"Simulated {len(frame)} rows across {frame['sensor_id'].nunique()} sensors.")

    # 2. Features ------------------------------------------------------------
    features = FeatureBuilder().transform(frame)
    print(f"Engineered {features.shape[1]} feature columns.")

    # Work with one sensor for forecasting and detection.
    temp = frame[frame["sensor_id"] == "temperature"].reset_index(drop=True)
    window = TimeSeriesWindow(
        sensor_id="temperature",
        timestamps=list(temp["timestamp"]),
        values=[float(v) for v in temp["value"]],
    )

    # 3. Forecast ------------------------------------------------------------
    forecast = TabularForecaster(n_lags=48).fit(window).forecast(horizon=24)
    print(f"Forecast horizon: {len(forecast.mean)} steps.")

    # 4. Detect --------------------------------------------------------------
    anomalies = RobustZScoreDetector(window_size=60, threshold=4.0).detect(window)
    anomalies += CusumChangePointDetector(threshold=8.0, reference_size=300).detect(window)
    print(f"Detected {len(anomalies)} anomalies/change points.")

    # 5. Drift ---------------------------------------------------------------
    values = np.asarray(window.values, dtype=np.float64)
    drift = PsiDriftDetector(threshold=0.2).evaluate(values[:300], values[-300:])
    print(f"Drift (PSI): {drift.score:.3f} -> drifted={drift.drifted}")

    # 6. Alerts and report ---------------------------------------------------
    alerts = AlertPolicy().from_anomalies(anomalies)
    report = build_report("End-to-end demo", alerts, forecast=forecast, drift=[drift])
    print("\n" + report)


if __name__ == "__main__":
    main()
