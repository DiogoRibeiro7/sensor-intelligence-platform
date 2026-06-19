# Architecture

```text
Sensor readings
  -> Validation
  -> Feature engineering
  -> Forecasting model
  -> Residual analysis
  -> Anomaly detection
  -> Drift monitoring
  -> Alerts
  -> API / reports
```

The project is intentionally designed around operational trust: models must produce forecasts, uncertainty estimates, alert reasons, and monitoring artifacts.
