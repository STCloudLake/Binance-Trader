"""Label helpers that were never wired into any training path.

Moved verbatim from ``core/ml/features.py``.  ``core.ml.features`` still keeps
every label helper the production trainers actually use
(``create_binary_label``, ``create_regression_label``, ``create_triple_barrier_label``,
``create_volatility_label``) — this module only holds the orphaned ones.

The three "predict the market state" labels below were written for a
volatility/regime prediction head that no trainer ever implemented
(``create_volatility_label``, the one that IS used by
``MLPredictor.train_volatility_model``, stayed in ``core.ml.features``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.ml.features import _estimate_hurst


def create_label(df: pd.DataFrame, forward_periods: int = 4,
                 threshold: float = 0.01) -> pd.Series:
    """Three-class label: 'up' / 'down' / 'hold'."""
    future_close = df["close"].shift(-forward_periods)
    return_pct = (future_close - df["close"]) / df["close"]
    labels = pd.Series("hold", index=df.index)
    labels[return_pct > threshold] = "up"
    labels[return_pct < -threshold] = "down"
    return labels


def triple_barrier_probabilities(
    df: pd.DataFrame,
    forward_periods: int = 24,
    upper_pct: float = 0.02,
    lower_pct: float = 0.02,
) -> pd.DataFrame:
    """Multi-output version: returns P(up_hit), P(down_hit), P(timeout).

    Useful for models that output probability distributions rather
    than binary decisions.
    """
    from core.ml.features import create_triple_barrier_label

    labels = create_triple_barrier_label(
        df, forward_periods, upper_pct, lower_pct, timeout_label=-1.0)
    p_up = (labels == 1.0).astype(float)
    p_down = (labels == 0.0).astype(float)
    p_timeout = (labels == -1.0).astype(float)
    return pd.DataFrame(
        {"p_up": p_up, "p_down": p_down, "p_timeout": p_timeout},
        index=df.index)


def create_regime_label(
    df: pd.DataFrame,
    forward_periods: int = 20,
    hurst_window: int = 100,
) -> pd.Series:
    """Binary label: will the next window be TRENDING (1) or MEAN-REVERTING (0)?

    Uses the Hurst exponent of the *future* window as the ground truth.
    H > 0.55 = trending/persistent.  H <= 0.55 = random/mean-reverting.

    This requires enough data for the future Hurst calculation —
    typically *hurst_window* bars in the future window.

    Args:
        df: OHLCV DataFrame with at least ``"close"`` column.
        forward_periods: Bars to skip before measuring Hurst (avoid overlap).
        hurst_window: Window length for Hurst estimation.

    Returns:
        pd.Series of 1 (trending) / 0 (mean-reverting) / NaN.
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)
    labels = np.full(n, np.nan)

    for i in range(n - forward_periods - hurst_window):
        future_slice = close[i + forward_periods: i + forward_periods + hurst_window]
        h = _estimate_hurst(future_slice)
        labels[i] = 1.0 if h > 0.55 else 0.0

    return pd.Series(labels, index=df.index).astype("Int64")


def create_volatility_magnitude_label(
    df: pd.DataFrame,
    forward_periods: int = 20,
    lookback: int = 200,
) -> pd.Series:
    """Regression label: what QUANTILE will future volatility be in?

    0.0 = lowest vol ever seen.  1.0 = highest vol ever seen.
    This is a regression target — the model predicts a continuous value
    in [0, 1], representing expected relative volatility.

    Args:
        df: OHLCV DataFrame with at least ``"close"`` column.
        forward_periods: Bars ahead to measure volatility.
        lookback: Historical window for quantile calibration (reduced
            default for smaller datasets).

    Returns:
        pd.Series of float in [0, 1] (NaN where insufficient data).
    """
    ret = df["close"].pct_change()
    # Future realised volatility
    future_vol = (
        ret.shift(-1)
        .rolling(forward_periods)
        .std()
        .shift(-forward_periods + 1)
    )
    # Historical volatility quantile
    vol_history = ret.rolling(lookback).std()
    rolling_min = vol_history.rolling(lookback).min()
    rolling_max = vol_history.rolling(lookback).max()
    denom = rolling_max - rolling_min
    labels = (future_vol - rolling_min) / (denom + 1e-12)
    return labels.clip(0.0, 1.0)
