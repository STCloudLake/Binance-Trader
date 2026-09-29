"""Shared trade bookkeeping for backtest engines.

`BacktestEngine._close_position` and `EventDrivenExecutor._close_position` were
literal clones of each other (the hybrid engine's docstring even admitted it:
"Identical logic to BacktestEngine._close_position"). Cloned PnL bookkeeping is
exactly the kind of drift that makes two engines disagree about the same
strategies — which is what the hybrid/legacy equivalence gate test exists to
prevent. Both engines now call :func:`close_position` here.
"""

from __future__ import annotations


def close_position(pos_key: str,
                   pos: dict,
                   exit_price: float,
                   ts,
                   reason: str,
                   trades: list,
                   balance: float,
                   positions: dict,
                   per_matrix: dict,
                   cost_fn=None,
                   events: list | None = None) -> float:
    """Close ``positions[pos_key]``, record the trade, return the new balance.

    Args:
        pos_key: key of the position in ``positions``.
        pos: the position dict (symbol/side/quantity/entry_price/amount_usdt/…).
        exit_price: fill price for the exit.
        ts: exit timestamp.
        reason: exit reason tag (``stop_loss``/``tp_…``/``indicator``/…).
        trades: list to append the trade record to.
        balance: cash balance before the close.
        positions: open-position dict; the closed key is removed.
        per_matrix: per strategy×symbol accumulators (updated in place).
        cost_fn: optional ``(entry_price, exit_price, qty, symbol) -> cost``.
        events: optional event log; an ``exit`` event is appended when given.

    Returns:
        The updated cash balance (principal + PnL − costs).
    """
    sym = pos["symbol"]
    entry_price = pos["entry_price"]
    qty = pos["quantity"]
    amount = pos.get("amount_usdt", qty * entry_price)
    side = pos["side"]
    strategy_name = pos.get("strategy_name", "")

    if side == "long":
        pnl = (exit_price - entry_price) * qty
    else:
        pnl = (entry_price - exit_price) * qty

    costs = cost_fn(entry_price, exit_price, qty, sym) if cost_fn else 0.0
    pnl -= costs

    trades.append({
        "symbol": sym, "side": side,
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "quantity": round(qty, 6),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / (entry_price * qty) * 100, 2) if entry_price > 0 else 0,
        "strategy": strategy_name,
        "opened_at": str(pos.get("opened_at", ts)),
        "closed_at": str(ts),
        "amount_usdt": round(amount, 2),
        "exit_reason": reason,
        "cost": round(costs, 4),
    })
    balance += amount + pnl

    if events is not None:
        events.append({
            "time": str(ts), "type": "exit", "reason": reason,
            "symbol": sym, "price": exit_price,
            "pnl": round(pnl, 2), "strategy": strategy_name,
        })

    if strategy_name in per_matrix and sym in per_matrix[strategy_name]:
        cell = per_matrix[strategy_name][sym]
        cell["trades"] += 1
        cell["pnl"] += pnl
        if pnl > 0:
            cell["winning"] += 1
            cell["gross_win_pnl"] += pnl
        else:
            cell["losing"] += 1
            cell["gross_loss_pnl"] += abs(pnl)
        if side == "long":
            cell["long_trades"] += 1
        else:
            cell["short_trades"] += 1

    del positions[pos_key]
    return balance
