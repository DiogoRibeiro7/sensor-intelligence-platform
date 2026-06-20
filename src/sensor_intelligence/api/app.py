"""FastAPI inference service.

Exposes the platform's detection, forecasting, and drift capabilities over
HTTP. Each endpoint converts its payload into domain objects, runs the
relevant component, and returns validated models. Domain validation errors are
surfaced as ``422`` responses rather than ``500``s.
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException

from ..alerting import AlertPolicy
from ..anomaly import (
    AnomalyDetector,
    CusumChangePointDetector,
    EwmaDetector,
    RobustZScoreDetector,
)
from ..drift import ErrorDriftDetector, PsiDriftDetector
from ..models import RollingMeanForecaster, SeasonalNaiveForecaster, TabularForecaster
from .schemas import (
    AnomalyRequest,
    AnomalyResponse,
    DriftRequest,
    DriftResponse,
    ForecastRequest,
    ForecastResponse,
)


def _build_detector(method: str, threshold: float | None) -> AnomalyDetector:
    """Construct the requested anomaly detector with an optional threshold."""
    if method == "ewma":
        return EwmaDetector(threshold=threshold) if threshold else EwmaDetector()
    if method == "robust_zscore":
        return RobustZScoreDetector(threshold=threshold) if threshold else RobustZScoreDetector()
    if method == "cusum":
        return CusumChangePointDetector(threshold=threshold) if threshold else (
            CusumChangePointDetector()
        )
    raise HTTPException(status_code=422, detail=f"unknown method: {method}")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Sensor Intelligence Platform",
        description="Forecasting, anomaly detection, and drift monitoring for sensor data.",
        version="0.1.0",
    )
    policy = AlertPolicy()

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.post("/detect/anomalies", response_model=AnomalyResponse)
    def detect_anomalies(request: AnomalyRequest) -> AnomalyResponse:
        """Detect anomalies on a window and derive alerts."""
        try:
            window = request.window.to_window()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        detector = _build_detector(request.method, request.threshold)
        anomalies = detector.detect(window)
        alerts = policy.from_anomalies(anomalies)
        return AnomalyResponse(anomalies=anomalies, alerts=alerts)

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest) -> ForecastResponse:
        """Forecast future values with the requested method."""
        try:
            window = request.window.to_window()
            if request.method == "tabular":
                model = TabularForecaster(n_lags=request.n_lags or 24).fit(window)
                result = model.forecast(request.horizon)
            elif request.method == "seasonal_naive":
                if request.period is None:
                    raise HTTPException(
                        status_code=422, detail="seasonal_naive requires 'period'."
                    )
                result = SeasonalNaiveForecaster(period=request.period).forecast(
                    window, request.horizon
                )
            else:
                result = RollingMeanForecaster().forecast(window, request.horizon)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ForecastResponse(forecast=result)

    @app.post("/drift", response_model=DriftResponse)
    def drift(request: DriftRequest) -> DriftResponse:
        """Evaluate distribution or error drift between two samples."""
        reference = np.asarray(request.reference, dtype=float)
        current = np.asarray(request.current, dtype=float)
        try:
            if request.method == "psi":
                detector = PsiDriftDetector(threshold=request.threshold or 0.2)
                result = detector.evaluate(reference, current)
            else:
                error_detector = ErrorDriftDetector(threshold=request.threshold or 3.0)
                result = error_detector.evaluate(reference, current)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return DriftResponse(result=result)

    return app


app = create_app()
