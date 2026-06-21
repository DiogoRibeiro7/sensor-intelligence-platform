# Roadmap

## Current status

The core platform is in place: simulation, features, baseline and tabular forecasting,
anomaly detection, drift checks, alerting, reporting, streaming, and a local API are
implemented and covered by tests. The next work should focus on consistency hardening,
operational polish, and filling the remaining structural gaps in the package layout.

## Completed phases

### Phase 1 — Foundation

- Define sensor event, time-series window, forecast, anomaly, and alert models.
- Add simulator for multivariate sensor data.

### Phase 2 — Baselines

- Implement EWMA and robust z-score anomaly detection.
- Add seasonal naive and rolling baseline forecasts.

### Phase 3 — Feature engineering

- Rolling statistics.
- Lag features.
- Calendar features.
- Sensor health and missingness features.

### Phase 4 — Forecasting

- Train tabular forecasting model.
- Add multi-step forecasts.
- Add prediction intervals.

### Phase 5 — Anomaly and drift

- Add change-point detection.
- Add error-drift monitoring.
- Add alert severity and reason codes.

### Phase 6 — API and reporting

- Add FastAPI service.
- Add report generator.
- Add dashboard.

### Phase 7 — Streaming simulation

- Add streaming loop.
- Add backpressure-safe inference batching.
- Add alert sink.

## Next milestones

### Phase 8 — Consistency hardening

- Enforce the project rule that every alert carries both severity and at least one reason code.
- Update drift-generated alerts to include explicit reason codes and add tests for that contract.
- Tighten domain validation so empty alert reason-code payloads are rejected where appropriate.
- Review README examples and architecture notes against the current package surface to remove drift.

### Phase 9 — Packaging and developer workflow

- Make the `src/` layout robust for local development and CI so `poetry run pytest` works from a clean environment without manual `PYTHONPATH` fixes.
- Add a lightweight packaging/import smoke test to catch broken editable-install or path issues early.
- Reduce environment fragility around dev extras and document a minimal test/lint install path.

### Phase 10 — Data module completion

- Implement a real `sensor_intelligence.data` layer instead of the current stub package.
- Add reusable loaders, schema validation, timestamp normalization, and sensor-frame adapters.
- Separate synthetic data generation concerns from ingestion/cleaning concerns.

### Phase 11 — Evaluation and monitoring depth

- Add richer forecast backtesting utilities such as rolling-origin evaluation and per-sensor score breakdowns.
- Extend anomaly evaluation with precision/recall-style metrics against simulated ground truth.
- Track detector and forecaster performance by sensor family and scenario in MLflow or generated reports.

### Phase 12 — Serving and operations

- Add model artifact persistence and explicit load/predict flows for the API.
- Version API schemas and model configurations for reproducible local serving.
- Add health, readiness, and model-metadata endpoints suitable for deployment checks.

### Phase 13 — Streaming and fleet-scale monitoring

- Add fleet-level aggregation views across sensors, not only per-sensor alert emission.
- Support configurable alert suppression, deduplication windows, and escalation policies.
- Add drift/anomaly summaries over rolling fleet windows for operator workflows.

### Phase 14 — Production readiness

- Expand CI to verify tests, lint, typing, and an install/import smoke path under supported Python versions.
- Add stronger contract tests for notebooks, examples, and API payload compatibility.
- Document deployment, local operations, and failure modes as part of the repo’s definition of done.
