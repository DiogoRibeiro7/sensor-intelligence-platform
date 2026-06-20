from __future__ import annotations

import numpy as np
import pytest

from sensor_intelligence.drift import (
    ErrorDriftDetector,
    PsiDriftDetector,
    population_stability_index,
)


def test_psi_zero_for_same_distribution() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, size=5000)
    cur = rng.normal(0.0, 1.0, size=5000)

    psi = population_stability_index(ref, cur, bins=10)
    assert psi < 0.05


def test_psi_large_for_shifted_distribution() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 1.0, size=5000)
    cur = rng.normal(3.0, 1.0, size=5000)

    psi = population_stability_index(ref, cur, bins=10)
    assert psi > 0.25


def test_psi_detector_flags_drift() -> None:
    rng = np.random.default_rng(2)
    detector = PsiDriftDetector(threshold=0.2)

    same = detector.evaluate(rng.normal(0, 1, 4000), rng.normal(0, 1, 4000))
    shifted = detector.evaluate(rng.normal(0, 1, 4000), rng.normal(2.5, 1, 4000))

    assert not same.drifted
    assert shifted.drifted
    assert shifted.metric == "psi"


def test_psi_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index(np.array([]), np.array([1.0, 2.0]))


def test_error_drift_flags_growing_error() -> None:
    rng = np.random.default_rng(3)
    reference = rng.normal(0.0, 1.0, size=500)
    degraded = rng.normal(4.0, 1.0, size=100)

    result = ErrorDriftDetector(threshold=3.0).evaluate(reference, degraded)
    assert result.drifted
    assert result.metric == "error_mean_shift"


def test_error_drift_quiet_when_stable() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(0.0, 1.0, size=500)
    current = rng.normal(0.0, 1.0, size=100)

    result = ErrorDriftDetector(threshold=3.0).evaluate(reference, current)
    assert not result.drifted


def test_error_drift_requires_enough_data() -> None:
    with pytest.raises(ValueError, match="at least 2 reference"):
        ErrorDriftDetector().evaluate(np.array([1.0]), np.array([1.0]))
