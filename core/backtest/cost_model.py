"""Trading cost model — fees, spread slippage for realistic backtest PnL.

All costs are deducted at position close (round-trip) to avoid altering
entry prices, which would cascade into stop-loss and take-profit calculations.
"""


def apply_trading_costs(entry_price: float, exit_price: float, qty: float,
                        symbol: str, config) -> float:
    """Calculate total round-trip trading cost for a position.

    Components:
      - Taker fee on entry notional
      - Taker fee on exit notional
      - Half-spread slippage on entry (buy at ask, sell at bid)
      - Half-spread slippage on exit

    Returns the total cost to subtract from trade PnL.
    Returns 0 if cost model is disabled in config.

    Args:
        entry_price: Position entry price per unit.
        exit_price: Position exit price per unit.
        qty: Trade quantity.
        symbol: Trading pair (e.g. "BTCUSDT").
        config: Config instance with backtest_cost_enabled etc.
    """
    if not getattr(config, 'backtest_cost_enabled', True):
        return 0.0

    fee_pct = getattr(config, 'backtest_taker_fee_pct', 0.04) / 100.0
    spread_dict = getattr(config, 'backtest_spread_pct', {})
    spread_pct = spread_dict.get(symbol, 0.03) / 100.0

    entry_notional = qty * entry_price
    exit_notional = qty * exit_price

    entry_fee = entry_notional * fee_pct
    exit_fee = exit_notional * fee_pct
    entry_spread = entry_notional * (spread_pct / 2.0)
    exit_spread = exit_notional * (spread_pct / 2.0)

    return entry_fee + exit_fee + entry_spread + exit_spread
