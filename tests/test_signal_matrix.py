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
