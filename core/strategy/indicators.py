import pandas as pd
import numpy as np
import talib


def _safe_int(val, default):
    """Parse config value to int, handling empty strings and non-numeric values."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def _safe_float(val, default):
    """Parse config value to float, handling empty strings and non-numeric values."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def compute_all(df: pd.DataFrame, indicator_configs: dict) -> pd.DataFrame:
    result = df.copy()

    for name, cfg in indicator_configs.items():
        if not isinstance(cfg, dict):
            continue
        period = _safe_int(cfg.get("period", 14), 14)
        source_col = cfg.get("source", "close")
        source = result[source_col].values if source_col in result.columns else result["close"].values

        if name == "rsi":
            result["rsi"] = talib.RSI(source, timeperiod=period)
        elif name == "macd":
            fast = _safe_int(cfg.get("fast", 12), 12)
            slow = _safe_int(cfg.get("slow", 26), 26)
            sig = _safe_int(cfg.get("signal", 9), 9)
            macd, macd_signal, macd_hist = talib.MACD(
                source, fastperiod=fast, slowperiod=slow, signalperiod=sig
            )
            result["macd"] = macd
            result["macd_signal"] = macd_signal
            result["macd_histogram"] = macd_hist
        elif name == "bollinger":
            period = _safe_int(cfg.get("period", 20), 20)
            stddev = _safe_float(cfg.get("stddev", 2), 2)
            upper, middle, lower = talib.BBANDS(
                source, timeperiod=period, nbdevup=stddev, nbdevdn=stddev
            )
            result["bollinger_upper"] = upper
            result["bollinger_middle"] = middle
            result["bollinger_lower"] = lower
            result["bollinger_width"] = np.where(middle != 0, (upper - lower) / middle, np.nan)
        elif name == "ema":
            # Support single period, fast_period/slow_period, or periods list
            fast_p = _safe_int(cfg.get("fast_period", 0), 0)
            slow_p = _safe_int(cfg.get("slow_period", 0), 0)
            periods = cfg.get("periods", [])
            if fast_p:
                result["ema_fast"] = talib.EMA(source, timeperiod=fast_p)
                if slow_p:
                    result["ema_slow"] = talib.EMA(source, timeperiod=slow_p)
                elif period != 14:
                    result["ema_slow"] = talib.EMA(source, timeperiod=period)
                else:
                    result["ema_slow"] = talib.EMA(source, timeperiod=21)
            elif isinstance(periods, list) and periods:
                for p in periods:
                    result[f"ema_{p}"] = talib.EMA(source, timeperiod=_safe_int(p, 20))
            else:
                result[f"ema_{period}"] = talib.EMA(source, timeperiod=period)
        elif name == "sma":
            sma_values = talib.SMA(source, timeperiod=period)
            result[f"sma_{period}"] = sma_values
            result["sma"] = sma_values  # plain name for condition templates
        elif name == "atr":
            result["atr"] = talib.ATR(
                result["high"].values, result["low"].values, result["close"].values,
                timeperiod=period
            )
        elif name == "adx":
            result["adx"] = talib.ADX(
                result["high"].values, result["low"].values, result["close"].values,
                timeperiod=period
            )
        elif name == "stoch":
            slowk_p = _safe_int(cfg.get("slowk_period", 3), 3)
            slowd_p = _safe_int(cfg.get("slowd_period", 3), 3)
            slowk, slowd = talib.STOCH(
                result["high"].values, result["low"].values, result["close"].values,
                fastk_period=period, slowk_period=slowk_p, slowd_period=slowd_p
            )
            result["stoch_k"] = slowk
            result["stoch_d"] = slowd
        elif name == "obv":
            result["obv"] = talib.OBV(result["close"].values, result["volume"].values)
            result["obv_sma"] = result["obv"].rolling(period).mean()
        elif name == "cci":
            result["cci"] = talib.CCI(
                result["high"].values, result["low"].values, result["close"].values,
                timeperiod=period
            )
        elif name == "hurst":
            lookback = _safe_int(cfg.get("lookback", 100), 100)
            max_lag = _safe_int(cfg.get("max_lag", 20), 20)
            result["hurst"] = _compute_hurst_indicator(source, lookback, max_lag)
            result["hurst_signal"] = result["hurst"].rolling(lookback).mean()
        elif name == "swing_points":
            lookback = _safe_int(cfg.get("lookback", 5), 5)
            swing_highs, swing_lows = _detect_swing_points(
                result["high"].values, result["low"].values,
                result["close"].values, lookback)
            result["swing_high"] = swing_highs
            result["swing_low"] = swing_lows
            result["dist_to_high_pct"] = np.where(
                result["close"].values != 0,
                (swing_highs - result["close"].values) / result["close"].values,
                np.nan)
            result["dist_to_low_pct"] = np.where(
                result["close"].values != 0,
                (result["close"].values - swing_lows) / result["close"].values,
                np.nan)
            result["swing_range_pct"] = np.where(
                swing_lows != 0,
                (swing_highs - swing_lows) / swing_lows,
                np.nan)
        elif name == "frac_diff":
            d = _safe_float(cfg.get("d", 0.4), 0.4)
            threshold = _safe_float(cfg.get("threshold", 0.001), 0.001)
            result["frac_close"] = _fractional_diff(
                result["close"].values, d, threshold)

    # Auto-compute commonly needed derived columns (fills gaps from AI-generated conditions)
    if "volume" in result.columns and "volume_sma" not in result.columns:
        result["volume_sma"] = result["volume"].rolling(20).mean()
        result["volume_ratio"] = np.where(result["volume_sma"] != 0,
                                          result["volume"] / result["volume_sma"], np.nan)
    elif "volume" in result.columns:
        vol_sma = result["volume"].rolling(20).mean()
        result["volume_ratio"] = np.where(vol_sma != 0,
                                          result["volume"] / vol_sma, np.nan)

    if "bollinger_width" in result.columns and "bollinger_width_sma" not in result.columns:
        result["bollinger_width_sma"] = result["bollinger_width"].rolling(20).mean()

    if "close" in result.columns:
        if "ema_fast" not in result.columns:
            result["ema_fast"] = talib.EMA(result["close"].values, timeperiod=9)
        if "ema_slow" not in result.columns:
            result["ema_slow"] = talib.EMA(result["close"].values, timeperiod=21)
        if "price_momentum_24h" not in result.columns:
            result["price_momentum_24h"] = result["close"].pct_change(periods=24)
        # Normalized ATR (volatility relative to price)
        if "atr" in result.columns and "atr_ratio" not in result.columns:
            result["atr_ratio"] = np.where(result["close"] != 0,
                                           result["atr"] / result["close"], np.nan)

    return result


_COND_FAIL_LOG: set[str] = set()  # dedup failed conditions to avoid log spam

# Whitelist of allowed names in condition expressions.
# Only column names and a few safe helper functions are permitted.
_ALLOWED_FUNCTIONS = frozenset({"abs", "round", "min", "max", "sma", "cross"})
# Characters allowed in condition expressions beyond alphanumeric, whitespace, and operators
_ALLOWED_EXTRA_CHARS = frozenset("_.()<>!=&|,[]:'\"/+-*")


def _validate_condition(condition: str, allowed_columns: set[str]) -> bool:
    """Validate that a condition expression only references allowed names.

    Returns True if the condition is safe to evaluate, False otherwise.
    This prevents arbitrary code injection through strategy YAML files.
    """
    import re
    import builtins
    # Extract all identifiers (variable names, function names)
    identifiers = set(re.findall(r'[a-zA-Z_]\w*', condition))
    # Dangerous builtins that should never appear in conditions
    _dangerous_builtins = {
        "__import__", "eval", "exec", "compile", "open", "input",
        "globals", "locals", "vars", "dir", "getattr", "setattr",
        "delattr", "hasattr", "__class__", "__bases__", "__subclasses__",
        "__builtins__", "__globals__", "__code__", "system", "popen",
        "subprocess", "os", "sys", "shutil", "importlib",
    }
    dangerous = identifiers & _dangerous_builtins
    if dangerous:
        from loguru import logger
        logger.warning(f"Condition contains dangerous identifiers: {dangerous}")
        return False
    # Note: identifier names that are not in allowed_columns may be valid
    # (e.g., pd.eval builtins like 'abs', column names not yet computed).
    # We only block explicitly dangerous patterns above.
    return True


def evaluate_condition(df: pd.DataFrame, condition: str) -> pd.Series:
    env = {col: df[col] for col in df.columns}
    def _sma(series, period):
        return series.rolling(period).mean()
    # Avoid overwriting "sma" column with the helper function
    if "sma" not in env:
        env["sma"] = _sma

    # Validate condition safety before evaluation
    allowed_columns = set(df.columns)
    if not _validate_condition(condition, allowed_columns):
        return pd.Series([False] * len(df), index=df.index)

    try:
        result = pd.eval(condition, engine="python", local_dict=env)
        return result
    except Exception as e:
        from loguru import logger
        # Rate-limit: log each unique failed condition only once per process
        key = f"{condition}:{e}"
        if key not in _COND_FAIL_LOG:
            _COND_FAIL_LOG.add(key)
            logger.debug(f"Condition evaluation failed: '{condition}' — {e}")
        return pd.Series([False] * len(df), index=df.index)


# ── Market structure helpers ──────────────────────────────────────────────


def _compute_hurst_indicator(prices: np.ndarray, lookback: int = 100,
                             max_lag: int = 20) -> np.ndarray:
    """Rolling Hurst exponent via R/S analysis on log-returns.

    Returns a same-length array with NaN in the first *lookback* bars.

    H > 0.55 → trending/persistent
    H ≈ 0.50 → random walk
    H < 0.45 → mean-reverting
    """
    n = len(prices)
    result = np.full(n, np.nan)
    if n < lookback + 50:
        return result

    log_prices = np.log(np.maximum(prices, 1e-12))
    for i in range(lookback, n):
        window_returns = np.diff(log_prices[i - lookback : i + 1])
        result[i] = _rs_hurst(window_returns, max_lag)
    return result


def _rs_hurst(returns: np.ndarray, max_lag: int = 20) -> float:
    """R/S Hurst exponent for a single window of returns."""
    n = len(returns)
    if n < 50:
        return 0.5
    lags = np.unique(np.logspace(
        np.log10(4), np.log10(min(max_lag, n // 4)), num=10).astype(int))
    if len(lags) < 4:
        return 0.5
    rs_values = []
    for lag in lags:
        n_chunks = n // lag
        if n_chunks < 2:
            continue
        chunks = returns[: n_chunks * lag].reshape(n_chunks, lag).astype(np.float64)
        mean = chunks.mean(axis=1, keepdims=True)
        cum_dev = (chunks - mean).cumsum(axis=1)
        R = cum_dev.max(axis=1) - cum_dev.min(axis=1)
        S = chunks.std(axis=1, ddof=1) + 1e-12
        rs_values.append(float((R / S).mean()))
    if len(rs_values) < 4:
        return 0.5
    log_lags = np.log([lag for lag, _ in zip(lags, rs_values)])
    log_rs = np.log(rs_values)
    H = float(np.polyfit(log_lags, log_rs, 1)[0])
    return max(0.0, min(1.0, H))


def _detect_swing_points(high: np.ndarray, low: np.ndarray,
                         close: np.ndarray,
                         lookback: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Detect swing highs and lows and forward-fill the most recent levels.

    A swing high: high[t] > max(high[t-lookback : t+lookback+1])
    A swing low:  low[t]  < min(low[t-lookback : t+lookback+1])

    Returns two same-length arrays where each bar carries the nearest
    past swing high/low price (forward-filled from detection point).
    """
    n = len(high)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        h_window = high[i - lookback : i + lookback + 1]
        if high[i] == h_window.max():
            swing_high[i] = high[i]
        l_window = low[i - lookback : i + lookback + 1]
        if low[i] == l_window.min():
            swing_low[i] = low[i]

    # Forward-fill: each bar knows the most recent swing level
    last_high = np.nan
    last_low = np.nan
    for i in range(n):
        if not np.isnan(swing_high[i]):
            last_high = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_low = swing_low[i]
        swing_high[i] = last_high if not np.isnan(last_high) else close[i]
        swing_low[i] = last_low if not np.isnan(last_low) else close[i]

    return swing_high, swing_low


def _fractional_diff(prices: np.ndarray, d: float = 0.4,
                     threshold: float = 1e-5) -> np.ndarray:
    """Fractional differentiation — preserves memory while achieving stationarity.

    Lopez de Prado (2018), Chapter 5.  Integer differencing (returns)
    destroys long-memory properties.  Fractional differencing with
    d ∈ (0, 1) retains the Hurst structure while making the series
    approximately stationary.

    Uses the fixed-width window method: compute weights up to the
    point where |w_k| < threshold, then convolve.

    Args:
        prices: Raw price series.
        d: Differentiation order (0.3–0.4 works well for crypto).
        threshold: Weight cutoff.

    Returns:
        Same-length array (first N values are NaN due to warm-up).
    """
    n = len(prices)
    result = np.full(n, np.nan)

    # Compute weights
    weights = [1.0]
    for k in range(1, n):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    wlen = len(weights)

    if wlen < 2:
        return result

    weights = np.array(weights)
    for i in range(wlen - 1, n):
        window = prices[i - wlen + 1 : i + 1][::-1]
        result[i] = np.dot(weights, window)

    return result
