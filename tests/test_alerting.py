from __future__ import annotations

from datetime import datetime

import pytest

from sensor_intelligence.alerting import AlertPolicy, SeverityThresholds
from sensor_intelligence.domain import AlertSeverity, Anomaly
from sensor_intelligence.drift import DriftResult

_T0 = datetime(2024, 1, 1)


def _anomaly(score: float, method: str = "zscore", ts: datetime = _T0) -> Anomaly:
    return Anomaly(
        sensor_id="s1",
        timestamp=ts,
        observed=99.0,
        score=score,
        method=method,
        reason_codes=[f"{method} score {score}"],
    )


def test_severity_bands() -> None:
    policy = AlertPolicy(SeverityThresholds(warning=3.0, critical=5.0))

    assert policy.severity_for(2.0) is None
    assert policy.severity_for(3.5) is AlertSeverity.WARNING
    assert policy.severity_for(6.0) is AlertSeverity.CRITICAL


def test_subthreshold_anomalies_dropped() -> None:
    policy = AlertPolicy()
    alerts = policy.from_anomalies([_anomaly(1.0), _anomaly(2.5)])
    assert alerts == []


def test_coincident_anomalies_merge_into_one_alert() -> None:
    policy = AlertPolicy()
    alerts = policy.from_anomalies(
        [_anomaly(3.5, "zscore"), _anomaly(6.0, "cusum")]  # same sensor and timestamp
    )

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.severity is AlertSeverity.CRITICAL  # max severity wins
    assert len(alert.anomalies) == 2
    assert len(alert.reason_codes) == 2
    assert "cusum" in alert.message and "zscore" in alert.message


def test_distinct_timestamps_produce_distinct_alerts() -> None:
    policy = AlertPolicy()
    alerts = policy.from_anomalies(
        [_anomaly(4.0, ts=_T0), _anomaly(4.0, ts=datetime(2024, 1, 2))]
    )
    assert len(alerts) == 2


def test_thresholds_validate_ordering() -> None:
    with pytest.raises(ValueError, match="critical threshold"):
        SeverityThresholds(warning=5.0, critical=3.0)


def test_drift_result_becomes_alert() -> None:
    policy = AlertPolicy()
    drifted = DriftResult(metric="psi", score=0.4, threshold=0.2, drifted=True, detail="PSI 0.4")
    clean = DriftResult(metric="psi", score=0.1, threshold=0.2, drifted=False)

    alert = policy.from_drift(drifted, "s1", _T0)
    assert alert is not None
    assert alert.severity is AlertSeverity.WARNING
    assert "psi" in alert.message
    assert policy.from_drift(clean, "s1", _T0) is None
