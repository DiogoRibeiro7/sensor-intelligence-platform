"""Change-point detection via a two-sided CUSUM scheme.

Where the point detectors in :mod:`.detectors` flag individual outliers, a
change-point detector flags *regime shifts* — a sustained move in the mean of
the series. The tabular CUSUM accumulates standardized deviations and fires
when the running sum crosses a decision threshold, then resets. Detected
shifts are returned as :class:`~sensor_intelligence.domain.Anomaly` objects so
they flow through the same alerting pipeline as point anomalies.
"""

from __future__ import annotations

import numpy as np

from ..domain import Anomaly, TimeSeriesWindow
from .detectors import AnomalyDetector


class CusumChangePointDetector(AnomalyDetector):
    """Detect mean shifts with a standardized two-sided CUSUM.

    Parameters
    ----------
    slack:
        Allowance ``k`` in standard deviations; deviations smaller than this
        do not accumulate, which suppresses noise. Larger values detect only
        bigger shifts.
    threshold:
        Decision interval ``h`` in standard deviations; the CUSUM must exceed
        this to declare a change point.
    reference_size:
        Number of leading samples used to estimate the baseline mean and
        standard deviation. ``None`` uses the whole series.
    """

    def __init__(
        self,
        slack: float = 0.5,
        threshold: float = 5.0,
        reference_size: int | None = None,
        adapt_window: int = 30,
    ) -> None:
        if slack < 0.0:
            raise ValueError("slack must be non-negative.")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        if reference_size is not None and reference_size < 2:
            raise ValueError("reference_size must be at least 2 when provided.")
        if adapt_window < 1:
            raise ValueError("adapt_window must be at least 1.")
        self._slack = slack
        self._threshold = threshold
        self._reference_size = reference_size
        self._adapt_window = adapt_window

    def detect(self, window: TimeSeriesWindow) -> list[Anomaly]:
        """Return change points found in ``window`` in chronological order.

        After each detection the baseline mean is re-estimated from the samples
        immediately following the change, so a single regime shift produces one
        change point rather than a fresh detection on every subsequent sample.
        """
        values = np.asarray(window.values, dtype=float)
        if values.size < 2:
            return []

        ref = values if self._reference_size is None else values[: self._reference_size]
        mean = float(np.mean(ref))
        std = float(np.std(ref))
        if std <= 0.0:
            return []

        anomalies: list[Anomaly] = []
        s_pos = 0.0
        s_neg = 0.0
        cooldown = 0
        for i in range(values.size):
            z = (values[i] - mean) / std
            s_pos = max(0.0, s_pos + z - self._slack)
            s_neg = max(0.0, s_neg - z - self._slack)
            score = max(s_pos, s_neg)

            if cooldown > 0:
                cooldown -= 1
                continue

            if score > self._threshold:
                direction = "increase" if s_pos >= s_neg else "decrease"
                anomalies.append(
                    Anomaly(
                        sensor_id=window.sensor_id,
                        timestamp=window.timestamps[i],
                        observed=float(values[i]),
                        expected=mean,
                        score=score,
                        method="cusum",
                        reason_codes=[
                            f"sustained mean {direction} detected "
                            f"(CUSUM {score:.2f} > {self._threshold:g})"
                        ],
                    )
                )
                # Adapt the baseline to the new regime and pause re-detection
                # until the look-ahead window has been consumed.
                segment = values[i : i + self._adapt_window]
                mean = float(np.mean(segment))
                s_pos = 0.0
                s_neg = 0.0
                cooldown = self._adapt_window
        return anomalies
