"""Unit tests for EventDrivenExecutor — trade execution from signal matrix."""
import pytest
import pandas as pd
import numpy as np
from core.backtest.signal_matrix import SignalMatrix


def _make_dummy_matrix(with_signals=True):
    """Create a minimal SignalMatrix for testing the executor."""
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(100) * 50)

    entry_data = np.zeros(100, dtype="int8")
    exit_data = np.zeros(100, dtype=bool)
    if with_signals:
        entry_data[10] = 1   # long entry at t=10
        exit_data[20] = True  # exit at t=20

    entry_df = pd.DataFrame(
        [entry_data],
        index=pd.MultiIndex.from_tuples([("test_strat", "BTCUSDT", "1h")]),
        columns=dates,
    )
    exit_df = pd.DataFrame(
        [exit_data],
        index=pd.MultiIndex.from_tuples([("test_strat", "BTCUSDT", "1h", "exit_long")]),
        columns=dates,
    )

    price_df = pd.DataFrame({
        "open": close - 10, "high": close + 50,
        "low": close - 50, "close": close,
        "volume": np.ones(100) * 100,
    }, index=dates)

    return SignalMatrix(
        signals=entry_df,
        exit_signals=exit_df,
        price_data={"BTCUSDT": {"1h": price_df}},
        metadata={"build_time_seconds": 0.01},
    )


def test_executor_enter_long():
    """Executor should open a long position when entry signal is 1."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)

    matrix = _make_dummy_matrix(with_signals=True)
    executor = EventDrivenExecutor(sizer, config.hard_limits)
    result = executor.run(matrix, initial_balance=10000.0)

    assert result.final_balance != 10000.0
    assert len(result.trades) >= 1
    trade = result.trades[0]
    assert trade["side"] == "long"
    assert trade["symbol"] == "BTCUSDT"
    assert trade["strategy"] == "test_strat"


def test_executor_no_signals_no_trades():
    """No signals should produce no trades, unchanged balance."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)

    matrix = _make_dummy_matrix(with_signals=False)
    executor = EventDrivenExecutor(sizer, config.hard_limits)
    result = executor.run(matrix, initial_balance=10000.0)

    assert len(result.trades) == 0
    assert result.final_balance == 10000.0


def test_executor_per_strategy_isolation():
    """Isolation mode: two strategies should not block each other."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.backtest.signal_matrix import SignalMatrix
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    dates = pd.date_range("2026-01-01", periods=50, freq="1h")
    entry_data = np.zeros((2, 50), dtype="int8")
    entry_data[0, 5] = 1   # strat A long
    entry_data[1, 8] = -1  # strat B short

    entry_df = pd.DataFrame(
        entry_data,
        index=pd.MultiIndex.from_tuples([
            ("strat_a", "BTCUSDT", "1h"),
            ("strat_b", "BTCUSDT", "1h"),
        ]),
        columns=dates,
    )
    exit_df = pd.DataFrame(columns=dates)
    exit_df.index = pd.MultiIndex.from_tuples([], names=["strategy", "symbol", "tf", "side"])

    close = 50000 + np.cumsum(np.random.randn(50) * 50)
    price_df = pd.DataFrame({"open": close - 10, "high": close + 50,
                              "low": close - 50, "close": close,
                              "volume": np.ones(50) * 100}, index=dates)

    matrix = SignalMatrix(signals=entry_df, exit_signals=exit_df,
                          price_data={"BTCUSDT": {"1h": price_df}},
                          metadata={"build_time_seconds": 0.01})

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)
    executor = EventDrivenExecutor(sizer, config.hard_limits,
                                     per_strategy_isolation=True)
    result = executor.run(matrix, initial_balance=10000.0)

    strategy_trades = set(t["strategy"] for t in result.trades)
    assert "strat_a" in strategy_trades
    assert "strat_b" in strategy_trades


def test_executor_stop_loss_triggers():
    """Stop-loss should be checked every tick and close the position."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.backtest.signal_matrix import SignalMatrix
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    dates = pd.date_range("2026-01-01", periods=50, freq="1h")
    close = np.linspace(50000, 40000, 50)
    entry_data = np.zeros(50, dtype="int8")
    entry_data[5] = 1

    entry_df = pd.DataFrame(
        [entry_data],
        index=pd.MultiIndex.from_tuples([("test", "BTCUSDT", "1h")]),
        columns=dates,
    )
    exit_df = pd.DataFrame(columns=dates)
    exit_df.index = pd.MultiIndex.from_tuples([], names=["strategy", "symbol", "tf", "side"])

    price_df = pd.DataFrame({"open": close - 10, "high": close + 50,
                              "low": close - 50, "close": close,
                              "volume": np.ones(50) * 100}, index=dates)

    matrix = SignalMatrix(signals=entry_df, exit_signals=exit_df,
                          price_data={"BTCUSDT": {"1h": price_df}},
                          metadata={"build_time_seconds": 0})

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)
    executor = EventDrivenExecutor(sizer, config.hard_limits)
    result = executor.run(matrix, initial_balance=10000.0)

    assert len(result.trades) >= 1
    trade = result.trades[0]
    assert trade["exit_reason"] == "stop_loss"


def test_executor_balance_insufficient():
    """Should not enter when balance is too low."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.backtest.signal_matrix import SignalMatrix
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    dates = pd.date_range("2026-01-01", periods=10, freq="1h")
    entry_data = np.zeros(10, dtype="int8")
    entry_data[1] = 1

    entry_df = pd.DataFrame(
        [entry_data],
        index=pd.MultiIndex.from_tuples([("test", "BTCUSDT", "1h")]),
        columns=dates,
    )
    exit_df = pd.DataFrame(columns=dates)
    exit_df.index = pd.MultiIndex.from_tuples([], names=["strategy", "symbol", "tf", "side"])

    close = np.full(10, 50000.0)
    price_df = pd.DataFrame({"open": close - 10, "high": close + 50,
                              "low": close - 50, "close": close,
                              "volume": np.ones(10) * 100}, index=dates)

    matrix = SignalMatrix(signals=entry_df, exit_signals=exit_df,
                          price_data={"BTCUSDT": {"1h": price_df}},
                          metadata={"build_time_seconds": 0})

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)
    executor = EventDrivenExecutor(sizer, config.hard_limits)
    result = executor.run(matrix, initial_balance=0.0)

    # With $0 balance, position size should be zero
    assert len(result.trades) == 0


def test_executor_max_positions():
    """Should respect max_positions limit."""
    from core.backtest.event_executor import EventDrivenExecutor
    from core.backtest.signal_matrix import SignalMatrix
    from core.risk.position_sizer import PositionSizer
    from app.config import Config

    dates = pd.date_range("2026-01-01", periods=30, freq="1h")
    entry_data = np.zeros((3, 30), dtype="int8")
    entry_data[0, 2] = 1
    entry_data[1, 4] = 1
    entry_data[2, 6] = 1

    entry_df = pd.DataFrame(
        entry_data,
        index=pd.MultiIndex.from_tuples([
            ("s1", "BTCUSDT", "1h"),
            ("s2", "ETHUSDT", "1h"),
            ("s3", "BNBUSDT", "1h"),
        ]),
        columns=dates,
    )
    exit_df = pd.DataFrame(columns=dates)
    exit_df.index = pd.MultiIndex.from_tuples([], names=["strategy", "symbol", "tf", "side"])

    close = np.full(30, 3000.0)
    price_df = pd.DataFrame({"open": close - 10, "high": close + 50,
                              "low": close - 50, "close": close,
                              "volume": np.ones(30) * 200}, index=dates)

    matrix = SignalMatrix(signals=entry_df, exit_signals=exit_df,
                          price_data={
                              "BTCUSDT": {"1h": price_df},
                              "ETHUSDT": {"1h": price_df},
                              "BNBUSDT": {"1h": price_df},
                          },
                          metadata={"build_time_seconds": 0})

    Config._instance = None
    config = Config.load("sim")
    sizer = PositionSizer(config.hard_limits, config.soft_params,
                          config.core_capital_pct, config.satellite_capital_pct)
    executor = EventDrivenExecutor(sizer, config.hard_limits, max_positions=2)
    result = executor.run(matrix, initial_balance=100000.0)

    symbols_entered = set(t["symbol"] for t in result.trades)
    assert len(symbols_entered) <= 2
