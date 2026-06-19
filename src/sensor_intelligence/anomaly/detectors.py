"""Statistical baseline anomaly detectors.

These detectors operate on a :class:`~sensor_intelligence.domain.TimeSeriesWindow`
and emit :class:`~sensor_intelligence.domain.Anomaly` objects. They are
intentionally simple and dependency-light so they can serve as transparent,
fast baselines that ML models must beat.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..domain import Anomaly, TimeSeriesWindow


class AnomalyDetector(ABC):
    """Common interface for point anomaly detectors."""

    @abstractmethod
    def detect(self, window: TimeSeriesWindow) -> list[Anomaly]:
        """Return anomalies found in ``window`` in chronological order."""
        raise NotImplementedError


class EwmaDetector(AnomalyDetector):
    """Exponentially weighted moving-average residual detector.

    The detector tracks an EWMA of the level and an EWMA of squared residuals
    (an online variance estimate). A point is anomalous when its residual
    exceeds ``threshold`` standard deviations of the running estimate.

    Parameters
    ----------
    alpha:
        Smoothing factor in ``(0, 1]``; larger reacts faster to changes.
    threshold:
        Residual cut-off in units of the running standard deviation.
    warmup:
        Number of initial points used only to seed the statistics.
    """

    def __init__(self, alpha: float = 0.3, threshold: float = 3.0, warmup: int = 10) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        if warmup < 1:
            raise ValueError("warmup must be at least 1.")
        self._alpha = alpha
        self._threshold = threshold
        self._warmup = warmup

    def detect(self, window: TimeSeriesWindow) -> list[Anomaly]:
        """Detect anomalies as large EWMA residuals."""
        values = window.values
        anomalies: list[Anomaly] = []
        if not values:
            return anomalies

        level = values[0]
        var = 0.0
        for i in range(1, len(values)):
            x = values[i]
            residual = x - level
            std = float(np.sqrt(var))
            if i >= self._warmup and std > 0.0 and abs(residual) > self._threshold * std:
                anomalies.append(
                    Anomaly(
                        sensor_id=window.sensor_id,
                        timestamp=window.timestamps[i],
                        observed=x,
                        expected=level,
                        score=abs(residual) / std,
                        method="ewma",
                        reason_codes=[
                            f"residual {residual:+.3f} exceeds "
                            f"{self._threshold:g}x running std {std:.3f}"
                        ],
                    )
                )
            # Update running statistics after scoring (predict-then-update).
            level = self._alpha * x + (1.0 - self._alpha) * level
            var = self._alpha * residual**2 + (1.0 - self._alpha) * var
        return anomalies


class RobustZScoreDetector(AnomalyDetector):
    """Rolling median/MAD (robust z-score) detector.

    For each point past the warm-up, the detector computes a modified z-score
    against the median and median absolute deviation (MAD) of the preceding
    ``window_size`` points. MAD-based scoring is resistant to the very outliers
    it is meant to catch, unlike mean/standard-deviation scoring.

    Parameters
    ----------
    window_size:
        Number of trailing points used to estimate median and MAD.
    threshold:
        Cut-off on the absolute modified z-score.
    """

    _MAD_SCALE = 0.6744897501960817  # 0.75 quantile of the standard normal.

    def __init__(self, window_size: int = 50, threshold: float = 3.5) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        self._window_size = window_size
        self._threshold = threshold

    def detect(self, window: TimeSeriesWindow) -> list[Anomaly]:
        """Detect anomalies via the rolling modified z-score."""
        values = np.asarray(window.values, dtype=float)
        anomalies: list[Anomaly] = []

        for i in range(self._window_size, len(values)):
            history = values[i - self._window_size : i]
            median = float(np.median(history))
            mad = float(np.median(np.abs(history - median)))
            if mad <= 0.0:
                continue
            z = self._MAD_SCALE * (values[i] - median) / mad
            if abs(z) > self._threshold:
                anomalies.append(
                    Anomaly(
                        sensor_id=window.sensor_id,
                        timestamp=window.timestamps[i],
                        observed=float(values[i]),
                        expected=median,
                        score=abs(z),
                        method="robust_zscore",
                        reason_codes=[
                            f"modified z-score {z:+.2f} exceeds {self._threshold:g}"
                        ],
                    )
                )
        return anomalies
