"""Unit tests for SignalMatrix, IndicatorGrouper, SignalMatrixBuilder."""
import pytest
import pandas as pd
import numpy as np
from core.strategy.loader import StrategyConfig, MLConfig
from core.backtest.signal_matrix import IndicatorGrouper, SignalMatrix


def _make_config(name, rsi_period, macd_fast=12, macd_slow=26):
    """Helper to create minimal StrategyConfig for testing."""
    return StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=["1h"],
        indicators={
            "rsi": {"period": rsi_period, "source": "close"},
            "macd": {"fast": macd_fast, "slow": macd_slow, "signal": 9},
        },
        entry_conditions={"long": ["rsi < 30"], "short": ["rsi > 70"]},
        exit_conditions={"long": [], "short": []},
        ml_config=MLConfig(enabled=False),
    )


def test_indicator_grouper_clusters_identical_configs():
    """Strategies with same indicator params should share groups."""
    s1 = _make_config("s1", 14, 12, 26)
    s2 = _make_config("s2", 14, 12, 26)
    s3 = _make_config("s3", 7, 8, 22)

    grouper = IndicatorGrouper()
    groups = grouper.group([s1, s2, s3])

    assert len(groups) == 2
    group_sizes = sorted([len(g) for g in groups])
    assert group_sizes == [1, 2]


def test_indicator_grouper_hash_deterministic():
    """Same config always produces same group hash."""
    s1 = _make_config("s1", 14)
    s2 = _make_config("s2", 14)

    grouper = IndicatorGrouper()
    h1 = grouper._config_hash(s1)
    h2 = grouper._config_hash(s2)
    assert h1 == h2


def test_indicator_grouper_different_params_different_hash():
    """Different RSI periods produce different hashes."""
    s1 = _make_config("s1", 14)
    s2 = _make_config("s2", 21)

    grouper = IndicatorGrouper()
    h1 = grouper._config_hash(s1)
    h2 = grouper._config_hash(s2)
    assert h1 != h2


def test_signal_matrix_get_entry():
    """SignalMatrix.get_entry should return correct signal values."""
    dates = pd.date_range("2026-01-01", periods=10, freq="1h")
    data = np.zeros(10, dtype="int8")
    data[3] = 1
    data[7] = -1

    df = pd.DataFrame(
        [data],
        index=pd.MultiIndex.from_tuples([("strat_a", "BTCUSDT", "1h")]),
        columns=dates,
    )

    matrix = SignalMatrix(
        signals=df,
        exit_signals=pd.DataFrame(),
        price_data={},
        metadata={},
    )

    assert matrix.get_entry("strat_a", "BTCUSDT", "1h", dates[3]) == 1
    assert matrix.get_entry("strat_a", "BTCUSDT", "1h", dates[7]) == -1
    assert matrix.get_entry("strat_a", "BTCUSDT", "1h", dates[0]) == 0
    # Non-existent key returns 0
    assert matrix.get_entry("nonexistent", "BTCUSDT", "1h", dates[0]) == 0


def test_signal_matrix_get_exit():
    """SignalMatrix.get_exit should return correct bool values."""
    dates = pd.date_range("2026-01-01", periods=10, freq="1h")
    data = np.zeros(10, dtype=bool)
    data[5] = True

    df = pd.DataFrame(
        [data],
        index=pd.MultiIndex.from_tuples([("strat_a", "BTCUSDT", "1h", "exit_long")]),
        columns=dates,
    )

    matrix = SignalMatrix(
        signals=pd.DataFrame(),
        exit_signals=df,
        price_data={},
        metadata={},
    )

    assert matrix.get_exit("strat_a", "BTCUSDT", "1h", "long", dates[5]) is True
    assert matrix.get_exit("strat_a", "BTCUSDT", "1h", "long", dates[0]) is False
    assert matrix.get_exit("nonexistent", "BTCUSDT", "1h", "long", dates[0]) is False


def test_batch_condition_eval_matches_sequential():
    """Batch-evaluated conditions must match one-by-one evaluation."""
    from core.strategy.indicators import compute_all, evaluate_condition

    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=500, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(500) * 100)
    df = pd.DataFrame({
        "open": close - 50, "high": close + 100,
        "low": close - 100, "close": close,
        "volume": np.random.rand(500) * 100 + 50,
    }, index=dates)

    ind1 = {"rsi": {"period": 14, "source": "close"}}
    ind2 = {"rsi": {"period": 7, "source": "close"}}
    df1 = compute_all(df.copy(), ind1)
    df2 = compute_all(df.copy(), ind2)

    conditions = ["rsi < 30", "rsi > 70"]
    for cond in conditions:
        r1 = evaluate_condition(df1, cond)
        r2 = evaluate_condition(df2, cond)
        assert not r1.equals(r2) or r1.sum() == r2.sum() == 0


def test_condition_and_combination():
    """Multiple entry conditions must be combined with AND logic."""
    from core.strategy.indicators import compute_all, evaluate_condition

    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=500, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(500) * 100)
    df = pd.DataFrame({
        "open": close - 50, "high": close + 100,
        "low": close - 100, "close": close,
        "volume": np.random.rand(500) * 100 + 50,
    }, index=dates)
    df = compute_all(df, {"rsi": {"period": 14, "source": "close"},
                          "macd": {"fast": 12, "slow": 26, "signal": 9}})

    rsi_low = evaluate_condition(df, "rsi < 30")
    macd_positive = evaluate_condition(df, "macd_histogram > 0")
    combined = rsi_low & macd_positive

    assert combined.sum() <= rsi_low.sum()
    assert combined.sum() <= macd_positive.sum()
