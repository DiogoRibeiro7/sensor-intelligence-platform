"""Feature engineering for sensor time series.

Transforms a tidy long-format frame (one row per sensor/timestamp) into a
feature matrix suitable for tabular forecasting and anomaly models. All
windowed features are computed *within* each sensor group so information never
leaks across sensors, and trailing windows include the current observation so
the features describe the sensor's current state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration controlling which feature families are produced.

    Parameters
    ----------
    rolling_windows:
        Trailing window sizes (in samples) for rolling statistics.
    lags:
        Lag offsets (in samples) for lagged value features.
    add_calendar:
        Whether to add calendar/seasonal encodings.
    add_health:
        Whether to add sensor-health and missingness features.
    value_col, timestamp_col, group_col:
        Column names in the input frame.
    """

    rolling_windows: tuple[int, ...] = (5, 15, 60)
    lags: tuple[int, ...] = (1, 2, 3)
    add_calendar: bool = True
    add_health: bool = True
    value_col: str = "value"
    timestamp_col: str = "timestamp"
    group_col: str = "sensor_id"


def add_rolling_features(
    frame: pd.DataFrame, value_col: str, windows: tuple[int, ...]
) -> pd.DataFrame:
    """Add trailing rolling mean/std/min/max for each window size."""
    out = frame.copy()
    values = out[value_col]
    for window in windows:
        roll = values.rolling(window=window, min_periods=1)
        out[f"roll_mean_{window}"] = roll.mean()
        out[f"roll_std_{window}"] = roll.std().fillna(0.0)
        out[f"roll_min_{window}"] = roll.min()
        out[f"roll_max_{window}"] = roll.max()
    return out


def add_lag_features(frame: pd.DataFrame, value_col: str, lags: tuple[int, ...]) -> pd.DataFrame:
    """Add lagged value columns for each requested lag."""
    out = frame.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[value_col].shift(lag)
    return out


def add_calendar_features(frame: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Add calendar fields and cyclical (sin/cos) encodings of time."""
    out = frame.copy()
    ts = pd.to_datetime(out[timestamp_col])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    out["hour"] = ts.dt.hour
    out["dayofweek"] = dow
    out["is_weekend"] = (dow >= 5).astype(int)
    out["month"] = ts.dt.month
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    return out


def add_health_features(
    frame: pd.DataFrame, value_col: str, timestamp_col: str, gap_window: int = 15
) -> pd.DataFrame:
    """Add sampling-gap, missingness, and rolling data-quality features."""
    out = frame.copy()
    ts = pd.to_datetime(out[timestamp_col])
    gap = ts.diff().dt.total_seconds()
    out["gap_seconds"] = gap.fillna(gap.median() if gap.notna().any() else 0.0)

    is_missing = out[value_col].isna().astype(float)
    out["is_missing"] = is_missing.astype(int)
    out["missing_rate"] = is_missing.rolling(window=gap_window, min_periods=1).mean()
    out["observed_count"] = (1.0 - is_missing).cumsum()
    return out


class FeatureBuilder:
    """Build a feature matrix from a tidy long-format sensor frame.

    Parameters
    ----------
    config:
        Feature configuration. Defaults produce rolling, lag, calendar, and
        health features.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self._config = config or FeatureConfig()

    def _transform_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """Apply per-sensor windowed features to a single sorted group."""
        cfg = self._config
        out = group.sort_values(cfg.timestamp_col).reset_index(drop=True)
        out = add_rolling_features(out, cfg.value_col, cfg.rolling_windows)
        out = add_lag_features(out, cfg.value_col, cfg.lags)
        if cfg.add_health:
            out = add_health_features(out, cfg.value_col, cfg.timestamp_col)
        return out

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``frame`` enriched with the configured features.

        Windowed and lag features are computed within each sensor group;
        calendar features are vectorised across the whole frame.
        """
        cfg = self._config
        required = {cfg.group_col, cfg.timestamp_col, cfg.value_col}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"input frame is missing required columns: {sorted(missing)}")

        groups = [
            self._transform_group(group)
            for _, group in frame.groupby(cfg.group_col, sort=False)
        ]
        result = pd.concat(groups, ignore_index=True)
        if cfg.add_calendar:
            result = add_calendar_features(result, cfg.timestamp_col)
        return result.sort_values([cfg.group_col, cfg.timestamp_col]).reset_index(drop=True)
