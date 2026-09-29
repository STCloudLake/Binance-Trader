"""Unit tests for the shared backtest trade bookkeeping primitive.

`core/backtest/trade_book.close_position` replaced two literal clones
(`BacktestEngine._close_position` and `EventDrivenExecutor._close_position`).
These tests pin its arithmetic so the two engines cannot drift apart again.
"""
import pytest

from core.backtest.trade_book import close_position


def _position(side="long", entry=100.0, qty=2.0, **extra):
    pos = {
        "symbol": "BTCUSDT",
        "side": side,
        "quantity": qty,
        "entry_price": entry,
        "amount_usdt": qty * entry,
        "strategy_name": "s1",
        "opened_at": "2026-01-01 00:00:00",
    }
    pos.update(extra)
    return pos


def _matrix():
    return {"s1": {"BTCUSDT": {
        "trades": 0, "pnl": 0.0, "winning": 0, "losing": 0,
        "long_trades": 0, "short_trades": 0,
        "gross_win_pnl": 0.0, "gross_loss_pnl": 0.0,
    }}}


def test_long_close_returns_principal_plus_pnl():
    positions = {"BTCUSDT": _position()}
    trades, matrix = [], _matrix()
    balance = close_position("BTCUSDT", positions["BTCUSDT"], 110.0, "T", "indicator",
                             trades, 1000.0, positions, matrix)
    # 200 principal + 20 profit
    assert balance == pytest.approx(1220.0)
    assert positions == {}
    t = trades[0]
    assert t["pnl"] == pytest.approx(20.0)
    assert t["pnl_pct"] == pytest.approx(10.0)
    assert t["exit_reason"] == "indicator"
    assert t["cost"] == 0.0
    assert t["opened_at"] == "2026-01-01 00:00:00"
    assert t["closed_at"] == "T"


def test_short_close_pnl_is_inverted():
    positions = {"BTCUSDT": _position(side="short")}
    trades, matrix = [], _matrix()
    balance = close_position("BTCUSDT", positions["BTCUSDT"], 90.0, "T", "tp_10pct",
                             trades, 1000.0, positions, matrix)
    assert balance == pytest.approx(1220.0)
    assert trades[0]["pnl"] == pytest.approx(20.0)


def test_costs_are_deducted_from_pnl():
    positions = {"BTCUSDT": _position()}
    trades, matrix = [], _matrix()
    close_position("BTCUSDT", positions["BTCUSDT"], 110.0, "T", "indicator",
                   trades, 1000.0, positions, matrix,
                   cost_fn=lambda ep, xp, q, s: 5.0)
    assert trades[0]["pnl"] == pytest.approx(15.0)
    assert trades[0]["cost"] == pytest.approx(5.0)


def test_per_matrix_accumulates_wins_and_losses():
    matrix = _matrix()
    positions = {"BTCUSDT": _position()}
    close_position("BTCUSDT", positions["BTCUSDT"], 110.0, "T", "indicator",
                   [], 1000.0, positions, matrix)
    cell = matrix["s1"]["BTCUSDT"]
    assert (cell["trades"], cell["winning"], cell["long_trades"]) == (1, 1, 1)
    assert cell["gross_win_pnl"] == pytest.approx(20.0)

    positions = {"BTCUSDT": _position()}
    close_position("BTCUSDT", positions["BTCUSDT"], 95.0, "T", "stop_loss",
                   [], 1000.0, positions, matrix)
    assert (cell["trades"], cell["losing"], cell["short_trades"]) == (2, 1, 0)
    assert cell["gross_loss_pnl"] == pytest.approx(10.0)
    assert cell["gross_win_pnl"] == pytest.approx(20.0)


def test_events_logged_only_when_requested():
    positions = {"BTCUSDT": _position()}
    events = []
    close_position("BTCUSDT", positions["BTCUSDT"], 110.0, "T", "indicator",
                   [], 1000.0, positions, _matrix(), events=events)
    assert len(events) == 1
    assert events[0]["type"] == "exit"
    assert events[0]["reason"] == "indicator"
    assert events[0]["pnl"] == pytest.approx(20.0)

    positions = {"BTCUSDT": _position()}
    close_position("BTCUSDT", positions["BTCUSDT"], 110.0, "T", "indicator",
                   [], 1000.0, positions, _matrix())
    # no events list passed → must not raise


def test_unknown_strategy_is_tolerated():
    """A strategy absent from per_matrix must not break the close (defensive)."""
    positions = {"BTCUSDT": _position(strategy_name="not_in_matrix")}
    balance = close_position("BTCUSDT", positions["BTCUSDT"], 110.0, "T", "indicator",
                             [], 1000.0, positions, _matrix())
    assert balance == pytest.approx(1220.0)


def test_missing_amount_usdt_falls_back_to_notional():
    pos = _position()
    del pos["amount_usdt"]
    positions = {"BTCUSDT": pos}
    balance = close_position("BTCUSDT", pos, 110.0, "T", "indicator",
                             [], 1000.0, positions, _matrix())
    assert balance == pytest.approx(1220.0)
