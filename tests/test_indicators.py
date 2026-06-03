"""Unit tests for technical indicators — compute_all, evaluate_condition, safe parsers."""

import pytest
import pandas as pd
import numpy as np


def _make_df(n=200):
    """Create synthetic OHLCV DataFrame for indicator testing."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "open": close - np.random.rand(n) * 50,
        "high": close + np.random.rand(n) * 100,
        "low": close - np.random.rand(n) * 100,
        "close": close,
        "volume": np.random.rand(n) * 100 + 50,
    }, index=dates)


def test_compute_all_rsi():
    """RSI should be computed between 0-100."""
    from core.strategy.indicators import compute_all

    df = _make_df()
    result = compute_all(df, {"rsi": {"period": 14, "source": "close"}})

    assert "rsi" in result.columns
    assert not result["rsi"].isna().all()
    # RSI values should be in [0, 100]
    valid = result["rsi"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_compute_all_macd():
    """MACD should produce three columns."""
    from core.strategy.indicators import compute_all

    df = _make_df()
    result = compute_all(df, {
        "macd": {"fast": 12, "slow": 26, "signal": 9}
    })

    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert "macd_histogram" in result.columns


def test_compute_all_bollinger():
    """Bollinger Bands should have upper > middle > lower."""
    from core.strategy.indicators import compute_all

    df = _make_df()
    result = compute_all(df, {
        "bollinger": {"period": 20, "stddev": 2.0}
    })

    assert "bollinger_upper" in result.columns
    assert "bollinger_middle" in result.columns
    assert "bollinger_lower" in result.columns
    assert "bollinger_width" in result.columns
    # upper > middle > lower for all valid rows
    valid = result.dropna()
    assert (valid["bollinger_upper"] >= valid["bollinger_middle"]).all()
    assert (valid["bollinger_middle"] >= valid["bollinger_lower"]).all()


def test_compute_all_adx():
    """ADX should be non-negative."""
    from core.strategy.indicators import compute_all

    df = _make_df()
    result = compute_all(df, {"adx": {"period": 14}})

    assert "adx" in result.columns
    valid = result["adx"].dropna()
    assert (valid >= 0).all()


def test_compute_all_ema():
    """EMA with fast/slow periods."""
    from core.strategy.indicators import compute_all

    df = _make_df()
    result = compute_all(df, {
        "ema": {"fast_period": 9, "slow_period": 21, "source": "close"}
    })

    assert "ema_fast" in result.columns
    assert "ema_slow" in result.columns


def test_compute_all_ema_list():
    """EMA with periods list."""
    from core.strategy.indicators import compute_all

    df = _make_df()
    result = compute_all(df, {
        "ema": {"periods": [9, 21, 50], "source": "close"}
    })

    assert "ema_9" in result.columns
    assert "ema_21" in result.columns
    assert "ema_50" in result.columns


def test_compute_all_auto_derived():
    """Auto-computed columns: volume_sma, ema_fast, ema_slow."""
    from core.strategy.indicators import compute_all

    df = _make_df()
    result = compute_all(df, {"rsi": {"period": 14, "source": "close"}})

    assert "volume_sma" in result.columns
    assert "volume_ratio" in result.columns
    assert "ema_fast" in result.columns
    assert "ema_slow" in result.columns
    assert "price_momentum_24h" in result.columns


def test_compute_all_sma():
    """SMA should be computed."""
    from core.strategy.indicators import compute_all

    df = _make_df()
    result = compute_all(df, {"sma": {"period": 20, "source": "close"}})
    assert "sma_20" in result.columns


def test_compute_all_atr():
    """ATR should be non-negative."""
    from core.strategy.indicators import compute_all

    df = _make_df()
    result = compute_all(df, {"atr": {"period": 14}})
    assert "atr" in result.columns
    valid = result["atr"].dropna()
    assert (valid >= 0).all()


def test_evaluate_condition_valid():
    """Valid condition should return boolean Series."""
    from core.strategy.indicators import compute_all, evaluate_condition

    df = _make_df(300)
    df = compute_all(df, {"rsi": {"period": 14, "source": "close"}})
    result = evaluate_condition(df, "rsi < 30")

    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.dtype == bool


def test_evaluate_condition_invalid():
    """Invalid condition should return all-False Series, not crash."""
    from core.strategy.indicators import compute_all, evaluate_condition

    df = _make_df(300)
    df = compute_all(df, {"rsi": {"period": 14, "source": "close"}})
    # 'nonexistent_column' is not in the DataFrame
    result = evaluate_condition(df, "nonexistent_column > 50")

    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    # Should return all False on error
    assert not result.any()


def test_evaluate_condition_missing_indicator():
    """Condition referencing a missing indicator (e.g. adx not computed) should not crash."""
    from core.strategy.indicators import compute_all, evaluate_condition

    df = _make_df(300)
    # Only compute RSI, don't compute ADX
    df = compute_all(df, {"rsi": {"period": 14, "source": "close"}})
    # 'adx' is NOT in the DataFrame — evaluate_condition must handle this gracefully
    result = evaluate_condition(df, "adx > 20")

    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert not result.any()  # should return all False, not crash


def test_safe_int():
    """_safe_int should handle edge cases."""
    from core.strategy.indicators import _safe_int

    assert _safe_int(14, 14) == 14
    assert _safe_int("14", 14) == 14
    assert _safe_int("", 14) == 14
    assert _safe_int(None, 14) == 14
    assert _safe_int("abc", 14) == 14
    assert _safe_int(0, 14) == 0


def test_safe_float():
    """_safe_float should handle edge cases."""
    from core.strategy.indicators import _safe_float

    assert _safe_float(2.0, 2.0) == 2.0
    assert _safe_float("2.5", 2.0) == 2.5
    assert _safe_float("", 2.0) == 2.0
    assert _safe_float(None, 2.0) == 2.0
    assert _safe_float("abc", 2.0) == 2.0


def test_compute_all_stoch():
    """Stochastic oscillator should produce K and D lines."""
    from core.strategy.indicators import compute_all

    df = _make_df(300)
    result = compute_all(df, {"stoch": {"period": 14}})

    assert "stoch_k" in result.columns
    assert "stoch_d" in result.columns


def test_compute_all_obv():
    """OBV should be computed."""
    from core.strategy.indicators import compute_all

    df = _make_df(200)
    result = compute_all(df, {"obv": {}})
    assert "obv" in result.columns


def test_compute_all_cci():
    """CCI should be computed."""
    from core.strategy.indicators import compute_all

    df = _make_df(200)
    result = compute_all(df, {"cci": {"period": 20}})
    assert "cci" in result.columns


def test_compute_all_invalid_config():
    """Non-dict indicator config should be skipped gracefully."""
    from core.strategy.indicators import compute_all

    df = _make_df(100)
    result = compute_all(df, {"rsi": "invalid_not_a_dict"})
    # Should not crash, just skip the invalid config
    assert len(result.columns) >= len(df.columns)  # at least original + auto-derived
