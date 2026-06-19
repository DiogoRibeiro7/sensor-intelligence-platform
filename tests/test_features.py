from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sensor_intelligence.features import (
    FeatureBuilder,
    FeatureConfig,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
)

_T0 = datetime(2024, 1, 1)


def _frame(n: int, sensors: tuple[str, ...] = ("a", "b")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sid in sensors:
        for i in range(n):
            rows.append(
                {
                    "sensor_id": sid,
                    "timestamp": _T0 + timedelta(minutes=i),
                    "value": float(rng.normal(10.0, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def test_rolling_features_added_and_finite() -> None:
    out = add_rolling_features(_frame(50, ("a",)), "value", (5, 15))

    for col in ("roll_mean_5", "roll_std_5", "roll_min_15", "roll_max_15"):
        assert col in out.columns
        assert out[col].notna().all()


def test_lag_features_shift_values() -> None:
    df = _frame(10, ("a",))
    out = add_lag_features(df, "value", (1, 2))

    assert out["lag_1"].iloc[5] == pytest.approx(df["value"].iloc[4])
    assert out["lag_2"].iloc[5] == pytest.approx(df["value"].iloc[3])
    assert pd.isna(out["lag_1"].iloc[0])


def test_calendar_features_cyclical_bounds() -> None:
    out = add_calendar_features(_frame(48, ("a",)), "timestamp")

    assert out["hour"].between(0, 23).all()
    assert out["hour_sin"].between(-1.0, 1.0).all()
    assert out["is_weekend"].isin([0, 1]).all()


def test_builder_no_cross_sensor_leakage() -> None:
    # First row of each sensor's rolling mean equals its own first value,
    # not a value blended across sensors.
    df = _frame(20)
    out = FeatureBuilder(FeatureConfig(rolling_windows=(3,), lags=(1,))).transform(df)

    for sid in ("a", "b"):
        sub = out[out["sensor_id"] == sid].reset_index(drop=True)
        assert sub["roll_mean_3"].iloc[0] == pytest.approx(sub["value"].iloc[0])
        assert pd.isna(sub["lag_1"].iloc[0])


def test_builder_preserves_row_count_and_health() -> None:
    df = _frame(30)
    out = FeatureBuilder().transform(df)

    assert len(out) == len(df)
    assert {"gap_seconds", "is_missing", "missing_rate", "observed_count"} <= set(out.columns)
    # Even cadence ⇒ constant 60-second gaps after the first row.
    a = out[out["sensor_id"] == "a"]
    assert (a["gap_seconds"].iloc[1:] == 60.0).all()


def test_builder_missing_column_raises() -> None:
    bad = pd.DataFrame({"sensor_id": ["a"], "timestamp": [_T0]})
    with pytest.raises(KeyError, match="value"):
        FeatureBuilder().transform(bad)
