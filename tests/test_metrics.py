"""Unit tests for backtest metrics calculation."""

import pytest
import pandas as pd
import numpy as np


def _make_sample_trades(n=50):
    """Generate sample trades with known characteristics."""
    np.random.seed(42)
    trades = []
    price = 50000.0
    base_time = pd.Timestamp("2026-01-15 12:00:00")
    for i in range(n):
        side = "long" if np.random.random() < 0.6 else "short"
        entry_price = price + np.random.randn() * 200
        opened_at = base_time + pd.Timedelta(hours=i * 2)
        closed_at = opened_at + pd.Timedelta(hours=np.random.randint(1, 24))
        if side == "long":
            pnl = np.random.choice([200, -100, 300, -150, 500, -80], p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
            exit_price = entry_price + pnl / 0.01
        else:
            pnl = np.random.choice([150, -80, 250, -120, 400], p=[0.3, 0.2, 0.2, 0.15, 0.15])
            exit_price = entry_price - pnl / 0.01
        pnl_pct = pnl / (entry_price * 0.01) * 100
        trades.append({
            "symbol": "BTCUSDT", "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": 0.01,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "strategy": "test_strategy",
            "amount_usdt": round(entry_price * 0.01, 2),
            "opened_at": str(opened_at),
            "closed_at": str(closed_at),
        })
        price += np.random.randn() * 500
    return trades


def _make_equity_curve(n=200):
    """Generate equity curve with slight upward trend."""
    np.random.seed(42)
    eq = 10000.0
    base_time = pd.Timestamp("2026-01-01")
    curve = []
    for i in range(n):
        eq += np.random.randn() * 50
        if eq < 100:
            eq = 100
        curve.append({
            "time": str(base_time + pd.Timedelta(hours=i)),
            "equity": round(eq, 2),
            "balance": round(eq * 0.8, 2),
            "invested": round(eq * 0.2, 2),
        })
    return curve


def test_calculate_metrics_basic():
    """Basic metrics should return all expected keys."""
    from core.backtest.metrics import calculate_metrics

    trades = _make_sample_trades(50)
    equity = _make_equity_curve(200)

    metrics = calculate_metrics(trades, equity, initial_balance=10000, final_balance=10500)

    assert "total_trades" in metrics
    assert "win_rate_pct" in metrics
    assert "total_return_pct" in metrics
    assert "max_drawdown_pct" in metrics
    assert "sharpe_ratio" in metrics
    assert "profit_factor" in metrics


def test_calculate_metrics_no_trades():
    """Zero trades should produce valid metrics with error indicator."""
    from core.backtest.metrics import calculate_metrics

    equity = _make_equity_curve(100)
    metrics = calculate_metrics([], equity, initial_balance=10000, final_balance=10000)

    assert metrics["total_trades"] == 0
    assert metrics["total_return_pct"] == 0
    assert "error" in metrics


def test_calculate_metrics_win_rate():
    """Win rate should match ratio of positive PnL trades."""
    from core.backtest.metrics import calculate_metrics

    trades = [
        {"symbol": "BTC", "side": "long", "entry_price": 50000, "exit_price": 51000,
         "quantity": 0.01, "pnl": 10.0, "pnl_pct": 2.0, "strategy": "s1", "amount_usdt": 500,
         "opened_at": "2026-01-01 10:00:00", "closed_at": "2026-01-01 14:00:00"},
        {"symbol": "BTC", "side": "long", "entry_price": 50000, "exit_price": 49000,
         "quantity": 0.01, "pnl": -10.0, "pnl_pct": -2.0, "strategy": "s1", "amount_usdt": 500,
         "opened_at": "2026-01-01 15:00:00", "closed_at": "2026-01-01 19:00:00"},
        {"symbol": "BTC", "side": "short", "entry_price": 50000, "exit_price": 49000,
         "quantity": 0.01, "pnl": 10.0, "pnl_pct": 2.0, "strategy": "s1", "amount_usdt": 500,
         "opened_at": "2026-01-02 10:00:00", "closed_at": "2026-01-02 14:00:00"},
        {"symbol": "BTC", "side": "short", "entry_price": 50000, "exit_price": 51000,
         "quantity": 0.01, "pnl": -10.0, "pnl_pct": -2.0, "strategy": "s1", "amount_usdt": 500,
         "opened_at": "2026-01-02 15:00:00", "closed_at": "2026-01-02 19:00:00"},
    ]
    equity = [
        {"time": "2026-01-01 00:00:00", "equity": 10000, "balance": 10000, "invested": 0},
        {"time": "2026-01-03 00:00:00", "equity": 10000, "balance": 10000, "invested": 0},
    ]

    metrics = calculate_metrics(trades, equity, initial_balance=10000, final_balance=10000)
    assert metrics["win_rate_pct"] == 50.0  # 2 wins / 4 trades
    assert metrics["total_trades"] == 4


def test_calculate_metrics_total_return():
    """Total return should reflect PnL vs initial balance."""
    from core.backtest.metrics import calculate_metrics

    trades = [
        {"symbol": "BTC", "side": "long", "entry_price": 50000, "exit_price": 50100,
         "quantity": 0.01, "pnl": 1.0, "pnl_pct": 0.2, "strategy": "s1", "amount_usdt": 500,
         "opened_at": "2026-01-01 10:00:00", "closed_at": "2026-01-01 14:00:00"},
    ]
    equity = [
        {"time": "2026-01-01 00:00:00", "equity": 10000.0, "balance": 10000.0, "invested": 0.0},
        {"time": "2026-01-02 00:00:00", "equity": 10001.0, "balance": 10001.0, "invested": 0.0},
    ]

    metrics = calculate_metrics(trades, equity, initial_balance=10000, final_balance=10001)
    assert metrics["total_return_pct"] > 0


def test_calculate_metrics_profit_factor():
    """Profit factor > 0 when there are winning trades."""
    from core.backtest.metrics import calculate_metrics

    trades = [
        {"symbol": "BTC", "side": "long", "entry_price": 50000, "exit_price": 51000,
         "quantity": 0.01, "pnl": 10.0, "pnl_pct": 2.0, "strategy": "s1", "amount_usdt": 500,
         "opened_at": "2026-01-01 10:00:00", "closed_at": "2026-01-01 14:00:00"},
        {"symbol": "ETH", "side": "long", "entry_price": 3000, "exit_price": 3100,
         "quantity": 0.1, "pnl": 10.0, "pnl_pct": 3.3, "strategy": "s1", "amount_usdt": 300,
         "opened_at": "2026-01-02 10:00:00", "closed_at": "2026-01-02 14:00:00"},
    ]
    equity = [
        {"time": "2026-01-01 00:00:00", "equity": 10000, "balance": 10000, "invested": 0},
        {"time": "2026-01-03 00:00:00", "equity": 10020, "balance": 10020, "invested": 0},
    ]

    metrics = calculate_metrics(trades, equity, initial_balance=10000, final_balance=10020)
    assert metrics["profit_factor"] > 1.0
