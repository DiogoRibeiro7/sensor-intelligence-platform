# Roadmap

## Phase 1 — Foundation

- Define sensor event, time-series window, forecast, anomaly, and alert models.
- Add simulator for multivariate sensor data.

## Phase 2 — Baselines

- Implement EWMA and robust z-score anomaly detection.
- Add seasonal naive and rolling baseline forecasts.

## Phase 3 — Feature engineering

- Rolling statistics.
- Lag features.
- Calendar features.
- Sensor health and missingness features.

## Phase 4 — Forecasting

- Train tabular forecasting model.
- Add multi-step forecasts.
- Add prediction intervals.

## Phase 5 — Anomaly and drift

- Add change-point detection.
- Add error-drift monitoring.
- Add alert severity and reason codes.

## Phase 6 — API and reporting

- Add FastAPI service.
- Add report generator.
- Add dashboard.

## Phase 7 — Streaming simulation

- Add streaming loop.
- Add backpressure-safe inference batching.
- Add alert sink.
