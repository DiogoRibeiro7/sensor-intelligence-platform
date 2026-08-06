# Changelog

All notable changes to this project are documented here.

This project follows semantic versioning where practical before `1.0.0`: patch releases are
for fixes, minor releases may still refine APIs, and breaking changes are called out
explicitly.

## [Unreleased]

## [0.1.1] - 2026-08-06

### Added

- Repository governance files for contributions, security reporting, issue triage, pull
  requests, pre-commit checks, and dependency updates.

### Changed

- Made `poetry run pytest` work from the source checkout without requiring a manual
  `PYTHONPATH` override.

## [0.1.0] - 2026-08-06

### Added

- Synthetic multivariate sensor simulation with configurable trend, seasonality, noise,
  spike anomalies, drift anomalies, and a reusable reference fleet.
- Leakage-safe feature engineering for rolling statistics, lags, calendar encodings, and
  sensor health signals.
- Forecasting baselines and a tabular scikit-learn forecaster with recursive multi-step
  forecasts and prediction intervals.
- EWMA, robust z-score, and CUSUM anomaly and change-point detectors.
- Drift monitoring with PSI and forecast-error mean-shift detection.
- Reason-coded alert policy with severity aggregation.
- Bounded-memory streaming processor with pluggable alert sinks.
- FastAPI service, dashboard, reports, notebooks, and an end-to-end example.
- Zenodo metadata and citation metadata.

### Changed

- MLflow logging is optional instead of a default runtime dependency.

### Security

- Removed vulnerable transitive dependency paths from the default runtime dependency graph.

[Unreleased]: https://github.com/DiogoRibeiro7/sensor-intelligence-platform/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/DiogoRibeiro7/sensor-intelligence-platform/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/DiogoRibeiro7/sensor-intelligence-platform/releases/tag/v0.1.0
