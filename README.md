# sensor-intelligence-platform

Forecasting, anomaly detection, drift monitoring, and predictive maintenance for industrial and IoT sensor data.

This repository is designed for senior data scientist, AI/ML engineer, research engineer, and decision intelligence roles.

## Goal

Build a production-style time-series system that ingests sensor data, generates forecasts with uncertainty, detects anomalies and change points, explains alerts, and exposes results through an API.

## Core capabilities

- Synthetic sensor simulator and adapters for public predictive-maintenance datasets.
- Statistical baselines: EWMA, robust z-score, seasonal baseline, change-point detection.
- ML models: XGBoost/LightGBM-style tabular forecasting adapter, optional neural sequence models.
- Probabilistic forecasts and prediction intervals.
- Anomaly reason codes and alert severity.
- Drift detection for feature and prediction-error distributions.
- FastAPI inference service.
- Batch and simulated streaming execution.
- MLflow-compatible experiment tracking.

## Usage

Run the API and dashboard:

```bash
uvicorn sensor_intelligence.api:app --reload
```

Then open <http://127.0.0.1:8000/> for the monitoring dashboard (forecast band +
alert table), or use the JSON endpoints `/forecast`, `/detect/anomalies`, and
`/drift`. The CLI offers `simulate` and an end-to-end `stream` demo:

```bash
python -m sensor_intelligence.cli stream --steps 720
```

For a full walkthrough of the pipeline (simulate → features → forecast → detect
→ drift → alerts → report), run the worked example:

```bash
python examples/end_to_end.py
```

## Portfolio signal

This project shows your strongest profile: rigorous time-series modelling, statistics, sensor intelligence, anomaly detection, and production ML.
