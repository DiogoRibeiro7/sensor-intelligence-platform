# Sensor Intelligence Platform

[![CI](https://github.com/DiogoRibeiro7/sensor-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/sensor-intelligence-platform/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Typed](https://img.shields.io/badge/typed-mypy%20strict-blue)](https://mypy-lang.org/)
[![Lint](https://img.shields.io/badge/lint-ruff-orange)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A production-style time-series platform for **forecasting, anomaly detection, drift
monitoring, and predictive maintenance** on industrial and IoT sensor data. It ingests
sensor streams, produces probabilistic forecasts, flags anomalies and regime shifts with
human-readable reason codes, monitors distributional drift, and serves everything through
a typed REST API and a live dashboard.

The codebase is small, fully typed (`mypy --strict`), linted (`ruff`), and covered by 82
tests running in CI — it is meant to read like production code, not a notebook dump.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [The pipeline, component by component](#the-pipeline-component-by-component)
- [REST API and dashboard](#rest-api-and-dashboard)
- [Notebooks](#notebooks)
- [Project layout](#project-layout)
- [Development](#development)
- [Design principles](#design-principles)
- [License](#license)

## Why this exists

Most sensor-analytics demos stop at a single model on a single CSV. Real predictive
maintenance needs a *pipeline*: clean domain contracts, transparent statistical baselines
to beat, an ML model with calibrated uncertainty, change-point and drift monitoring so you
know when the model has gone stale, an alerting policy that an operator can act on, and a
serving layer. This repository builds that pipeline end to end on reproducible synthetic
data, so every stage can be inspected, tested, and benchmarked.

## Capabilities

| Area | What it does |
| --- | --- |
| **Simulation** | Reproducible multivariate generator: trend, daily seasonality, AR(1) noise, and injectable spike/drift anomalies with ground-truth labels. |
| **Feature engineering** | Rolling statistics, lag features, cyclical calendar encodings, and sensor-health/missingness features — computed per sensor with no cross-sensor leakage. |
| **Forecasting** | Seasonal-naive and rolling-mean baselines plus a gradient-boosted tabular model with recursive multi-step forecasts and prediction intervals that widen with the horizon. |
| **Anomaly detection** | EWMA residual and rolling median/MAD (robust z-score) point detectors, plus a CUSUM change-point detector that reports one event per regime shift. |
| **Drift monitoring** | Population Stability Index (PSI) for feature/prediction drift and a forecast-error mean-shift monitor. |
| **Alerting** | Severity policy (info/warning/critical) that merges coincident detections and aggregates reason codes for explainability. |
| **Streaming** | Bounded-memory sliding-window processor with backpressure-safe batched inference and pluggable alert sinks. |
| **Serving** | FastAPI service (`/forecast`, `/detect/anomalies`, `/drift`) and a self-contained monitoring dashboard. |
| **Tracking** | Backtest metrics (MAE/RMSE/MAPE, interval coverage/width) and MLflow run logging. |

## Architecture

```text
                     ┌──────────────┐
   simulate ───────▶ │  domain      │  SensorReading · TimeSeriesWindow
                     │  models      │  Forecast · Anomaly · Alert
                     └──────┬───────┘
                            │
   ┌────────────┬───────────┼────────────┬─────────────┬────────────┐
   ▼            ▼           ▼            ▼             ▼            ▼
features     models      anomaly       drift        alerting    tracking
(rolling,   (baselines, (EWMA, MAD,   (PSI, error  (severity,   (backtest,
 lag,        tabular ML)  CUSUM)        drift)       reason       MLflow)
 calendar)                                           codes)
   │            │           │            │             │
   └────────────┴───────────┴────────────┴─────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
            streaming                  api
        (sliding window,         (REST endpoints +
         batched inference,       dashboard +
         alert sinks)             /demo)
```

Every stage exchanges validated [Pydantic](https://docs.pydantic.dev/) domain models
rather than loose dictionaries, so contracts are enforced at the boundaries.

## Installation

Requires Python 3.11 or 3.12 and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/DiogoRibeiro7/sensor-intelligence-platform.git
cd sensor-intelligence-platform
poetry install --with dev
```

## Quickstart

```python
from datetime import datetime, timedelta

import numpy as np
from sensor_intelligence.domain import TimeSeriesWindow
from sensor_intelligence.models import TabularForecaster
from sensor_intelligence.anomaly import RobustZScoreDetector
from sensor_intelligence.alerting import AlertPolicy

# A daily-seasonal signal with a spike.
t = np.arange(600)
values = 60 + 8 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(0).normal(0, 0.5, t.size)
values[420] = 130.0
start = datetime(2024, 1, 1)
window = TimeSeriesWindow(
    sensor_id="temperature",
    timestamps=[start + timedelta(minutes=int(i)) for i in t],
    values=values.tolist(),
)

# Forecast the next 24 steps with prediction intervals.
forecast = TabularForecaster(n_lags=48).fit(window).forecast(horizon=24)

# Detect anomalies and turn them into operator alerts.
anomalies = RobustZScoreDetector(window_size=60, threshold=4.0).detect(window)
alerts = AlertPolicy().from_anomalies(anomalies)
print(alerts[0].message)  # CRITICAL: robust_zscore flagged sensor temperature ...
```

Run the full pipeline (simulate → features → forecast → detect → drift → alerts → report):

```bash
python examples/end_to_end.py
```

Or the simulated streaming demo from the CLI:

```bash
python -m sensor_intelligence.cli stream --steps 720
```

## The pipeline, component by component

**Simulation** — `sensor_intelligence.simulation`. `SensorSimulator` turns a
`SimulationConfig` of `SensorSpec`s into a tidy long-format frame. Anomalies are injected
via `AnomalyInjection` (spike or drift) and flagged with a ground-truth `is_anomaly`
column for evaluation.

**Features** — `sensor_intelligence.features`. `FeatureBuilder` adds rolling
mean/std/min/max, lags, calendar sin/cos encodings, and sensor-health features
(inter-sample gap, missingness rate, observed count). Windowed features are computed
within each sensor group so information never leaks across sensors.

**Forecasting** — `sensor_intelligence.models`. `SeasonalNaiveForecaster` and
`RollingMeanForecaster` are the transparent baselines; `TabularForecaster` regresses on
lag + calendar features with a pluggable scikit-learn regressor (default
`HistGradientBoostingRegressor`), forecasts recursively, and produces prediction intervals
that widen with √horizon.

**Anomaly & change points** — `sensor_intelligence.anomaly`. `EwmaDetector` (online
variance, predict-then-update) and `RobustZScoreDetector` (rolling median/MAD) catch point
anomalies; `CusumChangePointDetector` catches sustained mean shifts and adapts its baseline
after each detection so a regime shift yields one event, not a cascade.

**Drift** — `sensor_intelligence.drift`. `PsiDriftDetector` scores distributional drift
between a reference and current window; `ErrorDriftDetector` flags forecast-error mean
shifts in standard-error units.

**Alerting** — `sensor_intelligence.alerting`. `AlertPolicy` maps scores to severity bands,
merges detections that share a sensor and timestamp, and carries every reason code forward.

**Streaming** — `sensor_intelligence.streaming`. `StreamProcessor` keeps a bounded sliding
window per sensor (constant memory), runs detection once per fixed-size batch, de-duplicates
alerts by timestamp, and emits to a pluggable `AlertSink` (in-memory, callable, or logging).

**Tracking** — `sensor_intelligence.tracking`. `backtest` holds out a series tail and scores
any forecaster; `forecast_metrics` reports MAE/RMSE/MAPE and interval calibration;
`log_to_mlflow` records params and metrics to MLflow.

## REST API and dashboard

```bash
uvicorn sensor_intelligence.api:app --reload
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` and `/dashboard` | GET | Live monitoring dashboard (forecast band + alert table). |
| `/demo` | GET | Self-contained simulated payload powering the dashboard. |
| `/forecast` | POST | Forecast a window (`tabular`, `seasonal_naive`, or `rolling_mean`). |
| `/detect/anomalies` | POST | Detect anomalies (`ewma`, `robust_zscore`, `cusum`) and derive alerts. |
| `/drift` | POST | Evaluate `psi` or `error` drift between two samples. |
| `/health` | GET | Liveness probe. |

Open <http://127.0.0.1:8000/> after starting the server. Domain validation errors are
surfaced as `422` responses, not `500`s.

## Notebooks

The [`notebooks/`](notebooks/) directory contains analytical walkthroughs:

1. **`01_data_simulation_and_eda.ipynb`** — generate sensor data and explore its structure, seasonality, and injected faults.
2. **`02_forecasting_and_backtesting.ipynb`** — baselines vs. the gradient-boosted model, prediction-interval calibration, and rolling backtests.
3. **`03_anomaly_drift_and_alerting.ipynb`** — point anomalies, change points, drift detection, and the alerting policy on a labelled scenario.

## Project layout

```text
src/sensor_intelligence/
├── domain/        # validated Pydantic models (the shared vocabulary)
├── simulation/    # synthetic multivariate sensor generator
├── features/      # rolling, lag, calendar, health features
├── models/        # baseline + tabular ML forecasters
├── anomaly/       # EWMA, robust z-score, CUSUM change points
├── drift/         # PSI and forecast-error drift monitors
├── alerting/      # severity policy and reason-code aggregation
├── streaming/     # bounded streaming processor and alert sinks
├── tracking/      # backtest metrics and MLflow logging
├── reporting/     # Markdown report generator
├── api/           # FastAPI service, schemas, dashboard
└── cli.py         # simulate / stream commands
examples/          # runnable end-to-end walkthrough
notebooks/         # analytical notebooks
tests/             # 82 tests
```

## Development

```bash
poetry run pytest          # tests with coverage
poetry run ruff check .    # lint
poetry run mypy src        # strict type checking
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint, type-check, and
tests on every push and pull request against Python 3.11.

## Design principles

- **Typed contracts at the boundaries.** Components exchange validated domain models, so a
  malformed window or forecast fails fast and loudly.
- **Baselines before ML.** Transparent statistical baselines establish the bar the learned
  model must clear; the tabular forecaster is benchmarked against them.
- **Uncertainty is first-class.** Forecasts carry prediction intervals; detections carry
  scores and reason codes; alerts carry severity.
- **Reproducibility.** Every stochastic component is seeded; the same config yields the same
  output.
- **Small, tested, and honest.** Strict typing, linting, and a real test suite gate every
  change in CI.

## License

Released under the [MIT License](LICENSE). © 2026 Diogo Ribeiro.
