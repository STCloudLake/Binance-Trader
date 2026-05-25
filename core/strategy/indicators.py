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
            result["bollinger_width"] = (upper - lower) / middle
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
            result[f"sma_{period}"] = talib.SMA(source, timeperiod=period)
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
            slowk, slowd = talib.STOCH(
                result["high"].values, result["low"].values, result["close"].values,
                fastk_period=period, slowk_period=3, slowd_period=3
            )
            result["stoch_k"] = slowk
            result["stoch_d"] = slowd
        elif name == "obv":
            result["obv"] = talib.OBV(result["close"].values, result["volume"].values)
        elif name == "cci":
            result["cci"] = talib.CCI(
                result["high"].values, result["low"].values, result["close"].values,
                timeperiod=period
            )

    # Auto-compute commonly needed derived columns (fills gaps from AI-generated conditions)
    if "volume" in result.columns and "volume_sma" not in result.columns:
        result["volume_sma"] = result["volume"].rolling(20).mean()
        result["volume_ratio"] = result["volume"] / result["volume_sma"]
    elif "volume" in result.columns:
        result["volume_ratio"] = result["volume"] / result["volume"].rolling(20).mean()

    if "bollinger_width" in result.columns and "bollinger_width_sma" not in result.columns:
        result["bollinger_width_sma"] = result["bollinger_width"].rolling(20).mean()

    if "close" in result.columns:
        if "ema_fast" not in result.columns:
            result["ema_fast"] = talib.EMA(result["close"].values, timeperiod=9)
        if "ema_slow" not in result.columns:
            result["ema_slow"] = talib.EMA(result["close"].values, timeperiod=21)
        if "price_momentum_24h" not in result.columns:
            result["price_momentum_24h"] = result["close"].pct_change(periods=24)

    return result


def evaluate_condition(df: pd.DataFrame, condition: str) -> pd.Series:
    env = {col: df[col] for col in df.columns}
    def _sma(series, period):
        return series.rolling(period).mean()
    env["sma"] = _sma
    try:
        result = pd.eval(condition, engine="python", local_dict=env)
        return result
    except Exception as e:
        from loguru import logger
        logger.warning(f"Condition evaluation failed: '{condition}' — {e}")
        return pd.Series([False] * len(df), index=df.index)
