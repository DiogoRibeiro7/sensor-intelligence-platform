from __future__ import annotations

from datetime import datetime, timedelta

from sensor_intelligence.domain import Alert, AlertSeverity, Anomaly, Forecast
from sensor_intelligence.drift import DriftResult
from sensor_intelligence.reporting import build_report, summarize_alerts

_T0 = datetime(2024, 1, 1)


def _alert(severity: AlertSeverity, ts: datetime = _T0) -> Alert:
    return Alert(
        sensor_id="s1",
        timestamp=ts,
        severity=severity,
        message=f"{severity.value} on s1",
        anomalies=[Anomaly(sensor_id="s1", timestamp=ts, observed=9.0, score=4.0, method="ewma")],
        reason_codes=[f"{severity.value} reason"],
    )


def test_summarize_counts_by_severity() -> None:
    alerts = [
        _alert(AlertSeverity.CRITICAL),
        _alert(AlertSeverity.WARNING),
        _alert(AlertSeverity.WARNING),
    ]
    assert summarize_alerts(alerts) == {"critical": 1, "warning": 2}


def test_report_lists_alerts_critical_first() -> None:
    alerts = [
        _alert(AlertSeverity.WARNING, _T0),
        _alert(AlertSeverity.CRITICAL, _T0 + timedelta(hours=1)),
    ]
    report = build_report("Daily report", alerts)

    assert report.startswith("# Daily report")
    assert "Total: 2" in report
    # Critical should be ranked above warning despite later timestamp.
    crit_pos = report.index("CRITICAL")
    warn_pos = report.index("WARNING")
    assert crit_pos < warn_pos
    assert "critical reason" in report


def test_report_includes_forecast_and_drift() -> None:
    forecast = Forecast(
        sensor_id="s1",
        timestamps=[_T0 + timedelta(minutes=i) for i in range(3)],
        mean=[1.0, 2.0, 3.0],
        lower=[0.0, 0.5, 1.0],
        upper=[2.0, 3.5, 5.0],
    )
    drift = [DriftResult(metric="psi", score=0.4, threshold=0.2, drifted=True, detail="PSI 0.4")]

    report = build_report("Report", [], forecast=forecast, drift=drift)
    assert "## Forecast" in report
    assert "Horizon: 3 steps" in report
    assert "## Drift" in report
    assert "DRIFT" in report


def test_empty_report_is_valid_markdown() -> None:
    report = build_report("Empty", [])
    assert report.startswith("# Empty")
    assert "Total: 0" in report
    assert report.endswith("\n")
