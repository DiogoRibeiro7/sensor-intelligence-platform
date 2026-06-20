"""Distribution and prediction-error drift monitoring.

Two complementary monitors:

* :class:`PsiDriftDetector` compares a current sample against a reference
  distribution with the Population Stability Index (PSI), a standard,
  scipy-free drift metric for features or predictions.
* :class:`ErrorDriftDetector` watches the distribution of forecast errors and
  flags when their mean shifts relative to a reference period — the signal
  that a deployed model has gone stale.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

FloatArray = npt.NDArray[np.float64]


class DriftResult(BaseModel):
    """Outcome of a single drift check.

    Parameters
    ----------
    metric:
        Name of the drift metric (for example ``"psi"``).
    score:
        Computed metric value; interpretation depends on ``metric``.
    threshold:
        Decision threshold the score is compared against.
    drifted:
        Whether ``score`` indicates drift.
    detail:
        Optional human-readable explanation.
    """

    metric: str = Field(min_length=1)
    score: float
    threshold: float
    drifted: bool
    detail: str | None = Field(default=None)


def population_stability_index(
    reference: FloatArray, current: FloatArray, bins: int = 10, epsilon: float = 1e-6
) -> float:
    """Compute the Population Stability Index between two samples.

    Bins are quantile edges of the reference sample, so each reference bin
    holds roughly equal mass. PSI sums ``(c - r) * ln(c / r)`` over bins, where
    ``r`` and ``c`` are reference and current proportions.

    Parameters
    ----------
    reference, current:
        One-dimensional samples to compare.
    bins:
        Number of quantile bins.
    epsilon:
        Floor applied to proportions to keep the logarithm finite.

    Returns
    -------
    float
        The PSI value; ``0`` means identical, larger means more drift.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("reference and current must be non-empty.")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_prop = np.clip(ref_counts / ref.size, epsilon, None)
    cur_prop = np.clip(cur_counts / cur.size, epsilon, None)

    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


class PsiDriftDetector:
    """Flag distribution drift using the Population Stability Index.

    Parameters
    ----------
    threshold:
        PSI above this value is reported as drift. The common convention is
        ``0.1`` (minor) and ``0.25`` (major); the default sits between.
    bins:
        Number of quantile bins passed to :func:`population_stability_index`.
    """

    def __init__(self, threshold: float = 0.2, bins: int = 10) -> None:
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        if bins < 2:
            raise ValueError("bins must be at least 2.")
        self._threshold = threshold
        self._bins = bins

    def evaluate(self, reference: FloatArray, current: FloatArray) -> DriftResult:
        """Compare ``current`` against ``reference`` and return a result."""
        psi = population_stability_index(reference, current, bins=self._bins)
        drifted = psi > self._threshold
        return DriftResult(
            metric="psi",
            score=psi,
            threshold=self._threshold,
            drifted=drifted,
            detail=f"PSI {psi:.3f} vs threshold {self._threshold:g}",
        )


class ErrorDriftDetector:
    """Flag drift in forecast-error magnitude via a standardized mean shift.

    The detector compares the mean absolute error of a current window against a
    reference error sample, scaling the difference by the reference standard
    error so the threshold is expressed in standard deviations.

    Parameters
    ----------
    threshold:
        Standardized mean-shift cut-off (in standard errors).
    """

    def __init__(self, threshold: float = 3.0) -> None:
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        self._threshold = threshold

    def evaluate(
        self, reference_errors: FloatArray, current_errors: FloatArray
    ) -> DriftResult:
        """Return a drift result for current errors versus reference errors."""
        ref = np.abs(np.asarray(reference_errors, dtype=float))
        cur = np.abs(np.asarray(current_errors, dtype=float))
        if ref.size < 2 or cur.size == 0:
            raise ValueError("need at least 2 reference errors and 1 current error.")

        ref_std = float(np.std(ref))
        standard_error = ref_std / np.sqrt(cur.size) if ref_std > 0.0 else 0.0
        shift = float(np.mean(cur) - np.mean(ref))
        score = abs(shift) / standard_error if standard_error > 0.0 else 0.0
        drifted = score > self._threshold
        return DriftResult(
            metric="error_mean_shift",
            score=score,
            threshold=self._threshold,
            drifted=drifted,
            detail=f"mean error shift {shift:+.4f} ({score:.2f} standard errors)",
        )
