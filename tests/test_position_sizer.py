"""Unit tests for PositionSizer — position sizing, stop-loss, take-profit calculation."""

import pytest
from app.config import Config


def _make_sizer():
    """Create a PositionSizer with default config."""
    Config._instance = None
    config = Config.load("sim")
    from core.risk.position_sizer import PositionSizer
    return PositionSizer(
        config.hard_limits, config.soft_params,
        config.core_capital_pct, config.satellite_capital_pct)


def test_calculate_position_size_long():
    """Basic position size calculation for long entry."""
    sizer = _make_sizer()
    qty, risk_amount = sizer.calculate_position_size(
        account_balance=10000.0, current_price=50000.0, position_type="satellite"
    )
    assert qty > 0
    assert risk_amount > 0
    assert risk_amount <= 5000  # max 50% position size


def test_calculate_position_size_core():
    """Core positions should use different capital allocation."""
    sizer = _make_sizer()
    qty_core, risk_core = sizer.calculate_position_size(
        account_balance=10000.0, current_price=50000.0, position_type="core"
    )
    assert qty_core > 0


def test_calculate_position_size_zero_balance():
    """Zero balance should return zero quantity."""
    sizer = _make_sizer()
    qty, risk_amount = sizer.calculate_position_size(
        account_balance=0.0, current_price=50000.0
    )
    assert qty == 0


def test_calculate_position_size_zero_price():
    """Zero price should return zero quantity."""
    sizer = _make_sizer()
    qty, risk_amount = sizer.calculate_position_size(
        account_balance=10000.0, current_price=0.0
    )
    assert qty == 0


def test_calculate_stop_loss_long():
    """Stop-loss for long should be below entry price."""
    sizer = _make_sizer()
    sl = sizer.calculate_stop_loss(entry_price=50000.0, side="long")
    assert sl < 50000.0
    assert sl > 0


def test_calculate_stop_loss_short():
    """Stop-loss for short should be above entry price."""
    sizer = _make_sizer()
    sl = sizer.calculate_stop_loss(entry_price=50000.0, side="short")
    assert sl > 50000.0


def test_calculate_take_profits():
    """Take-profits should return list of (price, pct) tuples."""
    sizer = _make_sizer()
    tps = sizer.calculate_take_profits(entry_price=50000.0, side="long")
    assert len(tps) >= 1
    for tp_price, tp_pct in tps:
        assert tp_price > 50000.0  # long TP above entry
        assert 0.0 < tp_pct <= 1.0


def test_calculate_take_profits_short():
    """Take-profits for short should be below entry."""
    sizer = _make_sizer()
    tps = sizer.calculate_take_profits(entry_price=50000.0, side="short")
    for tp_price, tp_pct in tps:
        assert tp_price < 50000.0
