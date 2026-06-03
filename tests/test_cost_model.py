"""Unit tests for trading cost model."""
import pytest


class FakeConfig:
    backtest_cost_enabled = True
    backtest_taker_fee_pct = 0.04
    backtest_spread_pct = {"BTCUSDT": 0.01, "ETHUSDT": 0.02, "SOLUSDT": 0.03}


def test_apply_trading_costs_btc():
    """BTC trade: 0.04% fee + 0.01% spread."""
    from core.backtest.cost_model import apply_trading_costs

    cost = apply_trading_costs(entry_price=50000, exit_price=51000, qty=0.01,
                               symbol="BTCUSDT", config=FakeConfig)
    # Entry notional: $500, Exit notional: $510
    # Fee: ($500 + $510) * 0.0004 = $0.404
    # Spread: ($500 + $510) * 0.00005 = $0.0505
    # Total: ~$0.4545
    assert 0.40 < cost < 0.50, f"Expected ~0.45, got {cost}"


def test_apply_trading_costs_disabled():
    """Disabled cost model should return zero."""
    from core.backtest.cost_model import apply_trading_costs

    cfg = FakeConfig()
    cfg.backtest_cost_enabled = False
    cost = apply_trading_costs(50000, 51000, 0.01, "BTCUSDT", cfg)
    assert cost == 0.0


def test_apply_trading_costs_higher_spread():
    """Altcoin with higher spread costs more."""
    from core.backtest.cost_model import apply_trading_costs

    btc_cost = apply_trading_costs(100, 101, 1, "BTCUSDT", FakeConfig)
    sol_cost = apply_trading_costs(100, 101, 1, "SOLUSDT", FakeConfig)
    assert sol_cost > btc_cost  # SOL 0.03% spread > BTC 0.01%


def test_apply_trading_costs_unknown_symbol():
    """Unknown symbol defaults to 0.03% spread."""
    from core.backtest.cost_model import apply_trading_costs

    cost = apply_trading_costs(100, 101, 1, "UNKNOWN", FakeConfig)
    assert cost > 0  # Uses default spread
