import ast
import operator

import pandas as pd
import numpy as np
import talib

#: Indicator-config key that adds the first-class volume/flow columns
#: (``rvol``/``rvol_z``/``vwap``/``mfi``/``ad_line``/``obv_slope``) from the
#: audited P6-B feature implementations (`core.ml.features`, one implementation
#: each: ``volr_20``, ``volz_60``, the rolling VWAP *level*, ``flow_mfi_14``, the
#: A/D line, ``obv_slope_10``).  On demand so the hot path pays nothing for
#: columns a strategy does not read; ``core/ga/genome.py`` enables it for any
#: condition that reads one of them, and ``TEMPLATE_REQUIRED_COLUMNS`` declares
#: the ownership.
VOLUME_FLOW_INDICATOR = "volume_flow"


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

        if name == VOLUME_FLOW_INDICATOR:
            # On demand, never unconditional: the family costs ~96 ms per 8 844
            # bars (measured), which would roughly triple `compute_all` on the
            # GA/backtest hot path for columns most strategies never read.  The
            # GA decoder enables this key automatically for any condition that
            # reads one of the columns (`core/ga/genome.py`), so a template can
            # never reference a column whose producer is off.
            from core.ml.features import volume_flow_indicator_columns
            volume_flow = volume_flow_indicator_columns(result)
            for column in volume_flow.columns:
                result[column] = volume_flow[column]
        elif name == "rsi":
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


class UnsafeConditionError(ValueError):
    """Raised when a condition string is not a permitted expression."""


# ── Safe condition evaluator ─────────────────────────────────────────────
#
# SECURITY: strategy conditions are attacker-influenced input. They can come from
# a YAML file saved by any `trader` account (POST /api/strategy), from the AI
# strategy lifecycle, or from GA chromosomes. The previous implementation ran
# `pd.eval(condition, engine="python")` after a regex *denylist* check; that check
# was bypassable (identifier splitting like `'__cla'+'ss__'`, unrestricted
# attribute/method calls) and allowed arbitrary method invocation such as
# `close.to_csv(...)` — i.e. arbitrary file write from a trading strategy.
#
# The evaluator below is a strict *allowlist*: the expression is parsed with `ast`
# and every node type is checked, attribute access/subscripts/lambdas/
# comprehensions/dunder names are rejected outright, and column references are
# resolved only against the DataFrame's own columns.

#: Expressions may call ONLY these functions.
_SAFE_FUNCTIONS: dict[str, object] = {}


def _safe_abs(value):
    return abs(value)


def _safe_round(value, ndigits=0):
    return round(value, ndigits)


def _safe_min(*args):
    if any(isinstance(a, pd.Series) for a in args):
        return np.minimum.reduce([np.asarray(a) if not isinstance(a, pd.Series) else a for a in args]) \
            if False else _elementwise_min(args)
    return min(args)


def _safe_max(*args):
    if any(isinstance(a, pd.Series) for a in args):
        return _elementwise_max(args)
    return max(args)


def _elementwise_min(args):
    result = args[0]
    for other in args[1:]:
        result = np.minimum(result, other)
    return result


def _elementwise_max(args):
    result = args[0]
    for other in args[1:]:
        result = np.maximum(result, other)
    return result


def _safe_sma(series, period):
    if not isinstance(series, pd.Series):
        raise UnsafeConditionError("sma() expects a column, e.g. sma(rsi, 14)")
    period = int(period)
    if not 1 <= period <= 1000:
        raise UnsafeConditionError("sma() period out of range")
    return series.rolling(period).mean()


def _safe_cross(a, b):
    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        raise UnsafeConditionError("cross() expects two columns")
    return (a > b) & (a.shift(1) <= b.shift(1))


_SAFE_FUNCTIONS.update({
    "abs": _safe_abs,
    "round": _safe_round,
    "min": _safe_min,
    "max": _safe_max,
    "sma": _safe_sma,
    "cross": _safe_cross,
})

#: Node types that may appear in a condition expression.
_ALLOWED_NODE_TYPES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Name, ast.Load, ast.Constant, ast.Call,
)

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _check_condition_ast(tree: ast.AST) -> None:
    """Reject any node/name that is not explicitly allowed."""
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise UnsafeConditionError(
                f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id.endswith("__"):
                raise UnsafeConditionError(f"dunder name not allowed: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsafeConditionError("only direct calls to whitelisted functions are allowed")
            if node.func.id not in _SAFE_FUNCTIONS:
                raise UnsafeConditionError(f"function not allowed: {node.func.id}")
            if node.keywords:
                raise UnsafeConditionError("keyword arguments are not allowed")
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            # String literals have no legitimate use in a numeric condition and
            # they are the building block of identifier-splitting tricks.
            raise UnsafeConditionError("string literals are not allowed in conditions")
        if isinstance(node, ast.BinOp) and type(node.op) not in _BIN_OPS:
            raise UnsafeConditionError(f"operator not allowed: {type(node.op).__name__}")


def _eval_condition_ast(node: ast.AST, env: dict):
    if isinstance(node, ast.Expression):
        return _eval_condition_ast(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise UnsafeConditionError(f"unknown column/identifier: {node.id}")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_condition_ast(node.operand, env)
        if isinstance(node.op, ast.Not):
            return ~operand.astype(bool) if isinstance(operand, pd.Series) else (not operand)
        if isinstance(node.op, ast.USub):
            return -operand
        return +operand
    if isinstance(node, ast.BinOp):
        left = _eval_condition_ast(node.left, env)
        right = _eval_condition_ast(node.right, env)
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.BoolOp):
        values = [_eval_condition_ast(v, env) for v in node.values]
        result = values[0]
        for value in values[1:]:
            if isinstance(node.op, ast.And):
                result = result & value
            else:
                result = result | value
        return result
    if isinstance(node, ast.Compare):
        left = _eval_condition_ast(node.left, env)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_condition_ast(comparator, env)
            comparison = _CMP_OPS[type(op)](left, right)
            result = comparison if result is None else (result & comparison)
            left = right
        return result
    if isinstance(node, ast.Call):
        func = _SAFE_FUNCTIONS[node.func.id]
        args = [_eval_condition_ast(a, env) for a in node.args]
        return func(*args)
    raise UnsafeConditionError(f"unsupported node: {type(node).__name__}")


def evaluate_condition(df: pd.DataFrame, condition: str) -> pd.Series:
    """Evaluate a strategy condition safely, returning a boolean Series.

    Unknown columns, disallowed syntax or evaluation errors yield an all-False
    Series (never an exception, never code execution).
    """
    false_series = pd.Series(False, index=df.index, dtype=bool)
    if not isinstance(condition, str) or not condition.strip():
        return false_series

    env: dict = {}
    for col in df.columns:
        try:
            env[str(col)] = df[col]
        except Exception:
            continue

    try:
        tree = ast.parse(condition, mode="eval")
        _check_condition_ast(tree)
        result = _eval_condition_ast(tree, env)
    except UnsafeConditionError as e:
        key = f"{condition}:{e}"
        if key not in _COND_FAIL_LOG:
            _COND_FAIL_LOG.add(key)
            from loguru import logger
            logger.warning(f"Condition rejected or unevaluable: '{condition}' — {e}")
        return false_series
    except Exception as e:
        key = f"{condition}:{e}"
        if key not in _COND_FAIL_LOG:
            _COND_FAIL_LOG.add(key)
            from loguru import logger
            logger.debug(f"Condition evaluation failed: '{condition}' — {e}")
        return false_series

    if isinstance(result, pd.Series):
        try:
            if result.dtype == bool:
                return result.fillna(False)
            return result.fillna(False).astype(bool)
        except Exception:
            return false_series
    # Scalar result (e.g. "1 > 0") — broadcast to the frame length.
    return pd.Series(bool(result), index=df.index, dtype=bool)


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
    """Detect swing highs/lows and forward-fill the most recent CONFIRMED levels.

    A swing high is ``high[t] > max(high[t-lookback : t+lookback+1])`` — a centred
    window, so it can only be *confirmed* ``lookback`` bars later. The detection
    index is therefore shifted forward by ``lookback`` before forward-filling:
    without that shift, the value stored at bar t depended on bars up to
    ``t + lookback`` (look-ahead bias — changing only bar T+1 altered the swing
    level reported at T, which live trading could never have known).

    Returns two same-length arrays where each bar carries the most recent
    *already confirmed* swing high/low price.
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

    # Shift detections to the bar where they become knowable (t + lookback).
    if lookback > 0:
        swing_high = np.concatenate([np.full(lookback, np.nan), swing_high[:-lookback]])
        swing_low = np.concatenate([np.full(lookback, np.nan), swing_low[:-lookback]])

    # Forward-fill: each bar knows the most recent confirmed swing level
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
