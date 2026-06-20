"""Turn raw detections into operator-facing alerts.

The policy maps anomaly scores to a severity, merges detections that land on
the same sensor and timestamp (so a point anomaly and a coincident change
point become one alert rather than two), and carries every contributing reason
code forward for explainability.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

from ..domain import Alert, AlertSeverity, Anomaly
from ..drift import DriftResult


@dataclass(frozen=True)
class SeverityThresholds:
    """Score cut-offs separating the severity bands.

    Scores at or above ``critical`` are critical; at or above ``warning`` but
    below ``critical`` are warnings; anything lower is informational and is not
    emitted as an alert by default.
    """

    warning: float = 3.0
    critical: float = 5.0

    def __post_init__(self) -> None:
        """Validate that the bands are ordered."""
        if self.warning <= 0.0:
            raise ValueError("warning threshold must be positive.")
        if self.critical < self.warning:
            raise ValueError("critical threshold must be >= warning threshold.")


class AlertPolicy:
    """Convert anomalies and drift results into :class:`Alert` objects.

    Parameters
    ----------
    thresholds:
        Severity score bands. Defaults to warning at 3.0, critical at 5.0.
    """

    def __init__(self, thresholds: SeverityThresholds | None = None) -> None:
        self._thresholds = thresholds or SeverityThresholds()

    def severity_for(self, score: float) -> AlertSeverity | None:
        """Map a score to a severity, or ``None`` if below the warning band."""
        if score >= self._thresholds.critical:
            return AlertSeverity.CRITICAL
        if score >= self._thresholds.warning:
            return AlertSeverity.WARNING
        return None

    def from_anomalies(self, anomalies: list[Anomaly]) -> list[Alert]:
        """Group anomalies by sensor and timestamp into merged alerts.

        Anomalies below the warning band are dropped. For each remaining
        ``(sensor_id, timestamp)`` group the alert takes the maximum severity
        and the union of reason codes, preserving detection order.
        """
        groups: OrderedDict[tuple[str, datetime], list[Anomaly]] = OrderedDict()
        for anomaly in anomalies:
            if self.severity_for(anomaly.score) is None:
                continue
            groups.setdefault((anomaly.sensor_id, anomaly.timestamp), []).append(anomaly)

        alerts: list[Alert] = []
        for (sensor_id, timestamp), group in groups.items():
            top_score = max(a.score for a in group)
            severity = self.severity_for(top_score)
            assert severity is not None  # guaranteed: every group member passed the filter.
            methods = sorted({a.method for a in group})
            reason_codes = [code for a in group for code in a.reason_codes]
            alerts.append(
                Alert(
                    sensor_id=sensor_id,
                    timestamp=timestamp,
                    severity=severity,
                    message=(
                        f"{severity.value.upper()}: {', '.join(methods)} "
                        f"flagged sensor {sensor_id} (score {top_score:.2f})"
                    ),
                    anomalies=group,
                    reason_codes=reason_codes,
                )
            )
        return alerts

    def from_drift(
        self, result: DriftResult, sensor_id: str, timestamp: datetime
    ) -> Alert | None:
        """Build an alert from a drift result, or ``None`` if no drift."""
        if not result.drifted:
            return None
        return Alert(
            sensor_id=sensor_id,
            timestamp=timestamp,
            severity=AlertSeverity.WARNING,
            message=(
                f"WARNING: {result.metric} drift on sensor {sensor_id} "
                f"({result.detail or f'score {result.score:.3f}'})"
            ),
            anomalies=[],
        )
