"""Human-readable reporting over detection and forecasting outputs.

Renders a Markdown summary an operator or analyst can read directly: alert
counts by severity, the most severe alerts with their reason codes, an
optional forecast summary, and any drift findings.
"""

from __future__ import annotations

from collections import Counter

from ..domain import Alert, AlertSeverity, Forecast
from ..drift import DriftResult

_SEVERITY_ORDER = {
    AlertSeverity.CRITICAL: 0,
    AlertSeverity.WARNING: 1,
    AlertSeverity.INFO: 2,
}


def summarize_alerts(alerts: list[Alert]) -> dict[str, int]:
    """Return a count of alerts per severity label."""
    counts: Counter[str] = Counter(alert.severity.value for alert in alerts)
    return dict(counts)


def build_report(
    title: str,
    alerts: list[Alert],
    forecast: Forecast | None = None,
    drift: list[DriftResult] | None = None,
    top_n: int = 5,
) -> str:
    """Render a Markdown report.

    Parameters
    ----------
    title:
        Report heading.
    alerts:
        Alerts to summarize and list.
    forecast:
        Optional forecast to summarize.
    drift:
        Optional drift results to report.
    top_n:
        Maximum number of most-severe alerts to list individually.

    Returns
    -------
    str
        The Markdown report.
    """
    lines: list[str] = [f"# {title}", ""]

    counts = summarize_alerts(alerts)
    lines.append("## Alerts")
    lines.append("")
    lines.append(f"- Total: {len(alerts)}")
    for severity in (AlertSeverity.CRITICAL, AlertSeverity.WARNING, AlertSeverity.INFO):
        if counts.get(severity.value):
            lines.append(f"- {severity.value.capitalize()}: {counts[severity.value]}")
    lines.append("")

    if alerts:
        ranked = sorted(
            alerts, key=lambda a: (_SEVERITY_ORDER[a.severity], a.timestamp)
        )[:top_n]
        lines.append(f"### Top {len(ranked)} alerts")
        lines.append("")
        for alert in ranked:
            stamp = alert.timestamp.isoformat()
            lines.append(
                f"- **{alert.severity.value.upper()}** "
                f"`{alert.sensor_id}` @ {stamp} — {alert.message}"
            )
            for code in alert.reason_codes:
                lines.append(f"    - {code}")
        lines.append("")

    if forecast is not None and forecast.mean:
        lines.append("## Forecast")
        lines.append("")
        horizon = len(forecast.mean)
        start = forecast.timestamps[0].isoformat()
        end = forecast.timestamps[-1].isoformat()
        lines.append(f"- Sensor: `{forecast.sensor_id}`")
        lines.append(f"- Horizon: {horizon} steps ({start} → {end})")
        lines.append(
            f"- Mean range: {min(forecast.mean):.3f} … {max(forecast.mean):.3f}"
        )
        if forecast.lower is not None and forecast.upper is not None:
            width = forecast.upper[-1] - forecast.lower[-1]
            lines.append(
                f"- Final interval ({forecast.interval_level:.0%}) width: {width:.3f}"
            )
        lines.append("")

    if drift:
        lines.append("## Drift")
        lines.append("")
        for result in drift:
            flag = "DRIFT" if result.drifted else "stable"
            detail = result.detail or f"score {result.score:.3f}"
            lines.append(f"- **{flag}** [{result.metric}] — {detail}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
