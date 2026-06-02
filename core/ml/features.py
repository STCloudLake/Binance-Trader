"""Feature engineering for ML models.

Provides a 30-dimensional feature set designed for financial time series:
- Price momentum (6): multi-period returns + acceleration
- Volatility (4): rolling std + volatility regime
- Volume (4): volume changes + trend
- Price position (5): distance from EMAs, BB position, BB width trend
- Trend/indicator (6): RSI, MACD, ADX + their changes
- Microstructure (3): intra-bar position, high-low range
- Sequence (3): consecutive direction, return distribution shape
"""

import pandas as pd
import numpy as np


# ── Default feature list (30 features) ──────────────────────────────────

DEFAULT_FEATURES: list[str] = [
    # Price momentum
    "ret_1", "ret_5", "ret_10", "ret_20",
    "acceleration_5", "momentum_ratio",
    # Volatility
    "vol_5", "vol_10", "vol_20", "vol_regime",
    # Volume
    "volume_ratio", "vol_chg_5", "vol_chg_20", "vol_trend",
    # Price position
    "pos_20", "ema20_dist", "ema50_dist",
    "bb_position", "bb_width_ratio",
    # Trend / indicator
    "rsi", "macd_histogram", "bollinger_width", "adx",
    "rsi_momentum", "macd_accel",
    # Microstructure
    "close_position", "high_low_range",
    # Sequence / distribution
    "consecutive_dir", "ret_skew_20", "ret_kurt_20",
]

# Extended features with cross-sectional and temporal signals
EXTENDED_FEATURES: list[str] = DEFAULT_FEATURES + [
    # Temporal (cyclical encoding — captures intraday/weekly patterns)
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    # Volume-quality
    "volume_price_corr_20",
    # Trend quality (how clean is the trend?)
    "adx_trend_strength", "ema_slope_20",
]

# Indicator configs needed to compute the indicator-derived features above
REQUIRED_INDICATORS: dict[str, dict] = {
    "rsi": {"period": 14, "source": "close"},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"period": 20, "stddev": 2},
    "adx": {"period": 14},
}


def build_features(df: pd.DataFrame, feature_list: list[str] | None = None) -> pd.DataFrame:
    """Extract named feature columns from a DataFrame that already has them.

    This is the simple path — call it when indicators are pre-computed on *df*.
    """
    if feature_list is None:
        feature_list = DEFAULT_FEATURES
    available = [f for f in feature_list if f in df.columns]
    if not available:
        return pd.DataFrame(index=df.index)
    result = df[available].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.ffill().fillna(0)
    return result


def compute_features(df: pd.DataFrame,
                     feature_list: list[str] | None = None) -> pd.DataFrame:
    """Compute the full 30-dim feature set from OHLCV data.

    Assumes *df* already has technical indicators attached (RSI, MACD,
    Bollinger Bands, ADX).  Use :func:`add_engineered_features` if you
    need to compute everything from raw OHLCV + indicators in one pass.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least 'open','high','low','close','volume' plus the
        indicator columns listed in REQUIRED_INDICATORS.
    feature_list : list[str] | None
        Subset of DEFAULT_FEATURES to return; None = all.

    Returns
    -------
    pd.DataFrame
        Feature matrix with the same index as *df*.
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    vol = df.get("volume", pd.Series(0, index=df.index)).astype(float)

    # ── Price momentum ──────────────────────────────────────────────
    result["ret_1"] = close.pct_change(1)
    result["ret_5"] = close.pct_change(5)
    result["ret_10"] = close.pct_change(10)
    result["ret_20"] = close.pct_change(20)
    # Acceleration: how much the recent trend is speeding up / slowing down
    result["acceleration_5"] = result["ret_1"] - result["ret_5"].shift(5)
    # Momentum ratio: short-term vs medium-term — >1 = accelerating trend
    result["momentum_ratio"] = (result["ret_5"].abs() + 1e-9) / (
        result["ret_20"].abs() + 1e-9)

    # ── Volatility ──────────────────────────────────────────────────
    ret = result["ret_1"]
    result["vol_5"] = ret.rolling(5).std()
    result["vol_10"] = ret.rolling(10).std()
    result["vol_20"] = ret.rolling(20).std()
    # Volatility regime: expanding (>1) or contracting (<1)
    result["vol_regime"] = (result["vol_5"] + 1e-9) / (result["vol_20"] + 1e-9)

    # ── Volume ──────────────────────────────────────────────────────
    vol_sma_5 = vol.rolling(5).mean()
    vol_sma_20 = vol.rolling(20).mean()
    result["volume_ratio"] = vol / (vol_sma_20 + 1e-9)
    result["vol_chg_5"] = vol.pct_change(5)
    result["vol_chg_20"] = vol.pct_change(20)
    result["vol_trend"] = (vol_sma_5 + 1e-9) / (vol_sma_20 + 1e-9)

    # ── Price position ──────────────────────────────────────────────
    result["pos_20"] = (close - close.rolling(20).min()) / (
        close.rolling(20).max() - close.rolling(20).min() + 1e-9)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    result["ema20_dist"] = (close - ema20) / (ema20 + 1e-9)
    result["ema50_dist"] = (close - ema50) / (ema50 + 1e-9)

    # BB position (normalised 0-1 within bands)
    if "bollinger_upper" in df.columns and "bollinger_lower" in df.columns:
        bb_upper = df["bollinger_upper"]
        bb_lower = df["bollinger_lower"]
        result["bb_position"] = (close - bb_lower) / (
            bb_upper - bb_lower + 1e-9)
        # BB width relative to its own 20-period SMA
        if "bollinger_width" in df.columns:
            bw_sma20 = df["bollinger_width"].rolling(20).mean()
            result["bb_width_ratio"] = (df["bollinger_width"] + 1e-9) / (bw_sma20 + 1e-9)
        else:
            result["bb_width_ratio"] = 1.0
    else:
        result["bb_position"] = 0.5
        result["bb_width_ratio"] = 1.0

    # ── Trend / indicator ───────────────────────────────────────────
    for col in ["rsi", "macd_histogram", "bollinger_width", "adx"]:
        if col in df.columns:
            result[col] = df[col]
    if "rsi" in result.columns:
        result["rsi_momentum"] = result["rsi"] - result["rsi"].shift(5)
    else:
        result["rsi_momentum"] = 0.0
    if "macd_histogram" in result.columns:
        result["macd_accel"] = (result["macd_histogram"] -
                                result["macd_histogram"].shift(3))
    else:
        result["macd_accel"] = 0.0

    # ── Microstructure ──────────────────────────────────────────────
    high = df.get("high", close)
    low = df.get("low", close)
    result["close_position"] = (close - low) / (high - low + 1e-9)
    result["high_low_range"] = (high - low) / (close + 1e-9)

    # ── Sequence / distribution ─────────────────────────────────────
    # Consecutive directional bars (approximate — uses close vs prev close)
    direction = np.sign(close.diff(1))
    result["consecutive_dir"] = (
        direction.groupby((direction != direction.shift(1)).cumsum())
        .cumcount()
        .astype(float)
    )
    # Distribution shape of recent returns
    result["ret_skew_20"] = ret.rolling(20).skew()
    result["ret_kurt_20"] = ret.rolling(20).kurt()

    # ── Temporal features (cyclical encoding for intraday patterns) ──
    if isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour.astype(float)
        days = df.index.dayofweek.astype(float)
        result["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        result["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        result["day_of_week_sin"] = np.sin(2 * np.pi * days / 7)
        result["day_of_week_cos"] = np.cos(2 * np.pi * days / 7)
    else:
        for col in ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"]:
            result[col] = 0.0

    # ── Volume-quality ───────────────────────────────────────────────
    price_dir = np.sign(close.diff(1))
    vol_dir = np.sign(vol.diff(1))
    result["volume_price_corr_20"] = (
        price_dir.rolling(20).corr(vol_dir)
    ).fillna(0)

    # ── Trend quality ───────────────────────────────────────────────
    result["adx_trend_strength"] = df.get("adx", pd.Series(0, index=df.index)) / 100.0
    result["ema_slope_20"] = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-9)

    # ── Cleanup ─────────────────────────────────────────────────────
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.ffill().fillna(0)

    # Subset if a specific list was requested
    if feature_list is not None:
        available = [f for f in feature_list if f in result.columns]
        result = result[available]

    return result


# ── Label helpers ────────────────────────────────────────────────────────


def create_label(df: pd.DataFrame, forward_periods: int = 4,
                 threshold: float = 0.01) -> pd.Series:
    """Three-class label: 'up' / 'down' / 'hold'."""
    future_close = df["close"].shift(-forward_periods)
    return_pct = (future_close - df["close"]) / df["close"]
    labels = pd.Series("hold", index=df.index)
    labels[return_pct > threshold] = "up"
    labels[return_pct < -threshold] = "down"
    return labels


def create_binary_label(df: pd.DataFrame, forward_periods: int = 4,
                         threshold: float = 0.005) -> pd.Series:
    """Binary label with noise filter.

    Returns 1 if price rises >= threshold, 0 if falls >= threshold,
    NaN for insignificant (noise) moves.
    """
    future_close = df["close"].shift(-forward_periods)
    return_pct = (future_close - df["close"]) / df["close"]
    labels = pd.Series(np.nan, index=df.index)
    labels[return_pct >= threshold] = 1.0
    labels[return_pct <= -threshold] = 0.0
    return labels.astype("Int64")


def create_regression_label(df: pd.DataFrame,
                             forward_periods: int = 4) -> pd.Series:
    """Continuous label: forward return percentage."""
    future_close = df["close"].shift(-forward_periods)
    return (future_close - df["close"]) / df["close"]


# ── Triple Barrier Labels ────────────────────────────────────────────────

def create_triple_barrier_label(
    df: pd.DataFrame,
    forward_periods: int = 24,
    upper_pct: float = 0.02,
    lower_pct: float = 0.02,
    timeout_label: float | None = None,
) -> pd.Series:
    """Path-aware label using the Triple Barrier Method.

    Instead of asking "did price go up at time T+N?", we ask "which
    barrier did price hit FIRST within the next N periods?"

    This respects the *path* — a price that rallies 3% then crashes
    5% hits the upper barrier first (label=1), even though the endpoint
    return is negative. Binary labels would incorrectly label this 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'high', 'low', 'close' columns.
    forward_periods : int
        Maximum number of periods to look forward.
    upper_pct : float
        Upper barrier as fraction above entry (e.g. 0.02 = +2%).
    lower_pct : float
        Lower barrier as fraction below entry (e.g. 0.02 = -2%).
    timeout_label : float | None
        Value for samples where no barrier is hit (NaN = filtered out).

    Returns
    -------
    pd.Series with same index as df:
        1  = upper barrier hit first (bullish)
        0  = lower barrier hit first (bearish)
        NaN or timeout_label = neither barrier hit within window
    """
    n = len(df)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)

    labels = np.full(n, np.nan)

    for i in range(n - forward_periods):
        entry = close[i]
        upper = entry * (1.0 + upper_pct)
        lower = entry * (1.0 - lower_pct)

        hit = np.nan
        for j in range(1, forward_periods + 1):
            idx = i + j
            if idx >= n:
                break
            if high[idx] >= upper:
                hit = 1.0  # upper hit first
                break
            if low[idx] <= lower:
                hit = 0.0  # lower hit first
                break
        labels[i] = hit

    result = pd.Series(labels, index=df.index)
    if timeout_label is not None:
        result = result.fillna(timeout_label)
    return result


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
    labels = create_triple_barrier_label(
        df, forward_periods, upper_pct, lower_pct, timeout_label=-1.0)
    p_up = (labels == 1.0).astype(float)
    p_down = (labels == 0.0).astype(float)
    p_timeout = (labels == -1.0).astype(float)
    return pd.DataFrame(
        {"p_up": p_up, "p_down": p_down, "p_timeout": p_timeout},
        index=df.index)
