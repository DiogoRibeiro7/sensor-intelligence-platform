from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from fastapi.testclient import TestClient

from sensor_intelligence.api import create_app

client = TestClient(create_app())
_T0 = datetime(2024, 1, 1)


def _payload(values: list[float]) -> dict[str, object]:
    timestamps = [(_T0 + timedelta(minutes=i)).isoformat() for i in range(len(values))]
    return {"sensor_id": "s1", "timestamps": timestamps, "values": values}


def _seasonal(n: int, period: int = 24) -> list[float]:
    rng = np.random.default_rng(0)
    t = np.arange(n)
    return (10.0 + 3.0 * np.sin(2.0 * np.pi * t / period) + rng.normal(0, 0.05, n)).tolist()


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_detect_anomalies_flags_spike() -> None:
    values = _seasonal(300)
    values[200] = 60.0
    resp = client.post(
        "/detect/anomalies",
        json={"window": _payload(values), "method": "robust_zscore"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert any(a["method"] == "robust_zscore" for a in body["anomalies"])
    assert isinstance(body["alerts"], list)


def test_forecast_tabular() -> None:
    resp = client.post(
        "/forecast",
        json={"window": _payload(_seasonal(480)), "horizon": 12, "method": "tabular"},
    )
    assert resp.status_code == 200
    forecast = resp.json()["forecast"]
    assert len(forecast["mean"]) == 12
    assert forecast["lower"] is not None


def test_forecast_seasonal_requires_period() -> None:
    resp = client.post(
        "/forecast",
        json={"window": _payload(_seasonal(100)), "horizon": 5, "method": "seasonal_naive"},
    )
    assert resp.status_code == 422
    assert "period" in resp.json()["detail"]


def test_forecast_invalid_window_returns_422() -> None:
    bad = {"sensor_id": "s1", "timestamps": [_T0.isoformat()], "values": [1.0, 2.0]}
    resp = client.post(
        "/forecast", json={"window": bad, "horizon": 3, "method": "rolling_mean"}
    )
    assert resp.status_code == 422


def test_dashboard_served() -> None:
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Sensor Intelligence Platform" in resp.text


def test_root_serves_dashboard() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<canvas id=\"chart\"" in resp.text


def test_demo_payload_shape() -> None:
    resp = client.get("/demo")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["series"]) == 2
    assert {s["sensor_id"] for s in body["series"]} == {"temperature", "vibration"}
    assert len(body["forecast"]["mean"]) == 48
    # The injected temperature spike should produce at least one alert.
    assert any(a["sensor_id"] == "temperature" for a in body["alerts"])


def test_drift_psi() -> None:
    rng = np.random.default_rng(1)
    resp = client.post(
        "/drift",
        json={
            "reference": rng.normal(0, 1, 2000).tolist(),
            "current": rng.normal(3, 1, 2000).tolist(),
            "method": "psi",
        },
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["metric"] == "psi"
    assert result["drifted"] is True
