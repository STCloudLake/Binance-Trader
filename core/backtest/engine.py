"""Backtesting engine — synchronous replay of historical kline data through the trading pipeline."""
import time
from pathlib import Path
import pandas as pd
from core.backtest.data_feeder import DataFeeder
from core.backtest.metrics import calculate_metrics


class BacktestEngine:
    """Synchronous backtesting engine that replays historical kline data."""

    def __init__(self, config, strategy_engine, risk_manager, order_executor):
        self.config = config
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.order_executor = order_executor

    def run(self, strategies: list[str], symbols: list[str],
            date_start: str, date_end: str, mode: str = "full",
            initial_balance: float = 10000.0) -> dict:
        """Run a backtest (simple mode — positions exit on next opposite signal).

        Args:
            strategies: List of strategy YAML names to test
            symbols: List of trading pairs
            date_start: ISO date string e.g. '2025-01-01'
            date_end: ISO date string e.g. '2026-01-01'
            mode: 'quick' (signal only) or 'full' (with position tracking)
            initial_balance: Starting account balance

        Returns:
            dict with keys: trades, equity_curve, events, metrics, ml_accuracy
        """
        t0 = time.time()

        # Load strategy configs
        strategy_configs = []
        for name in strategies:
            try:
                s = self.strategy_engine.loader.load(name)
                strategy_configs.append(s)
            except Exception as e:
                return {"error": f"Strategy '{name}' not found: {e}"}

        # Determine required intervals from strategy timeframes
        intervals = list(set(tf for s in strategy_configs
                            for tf in s.timeframes)) or ["1h"]
        if "1m" not in intervals:
            intervals.append("1m")  # Always include for granularity

        # Load historical data
        cache_dir = str(Path(self.config.data_dir) / "market")
        feeder = DataFeeder(cache_dir, symbols, intervals, date_start, date_end)
        feeder.load()

        if len(feeder) == 0:
            return {"error": "No historical data found for the given symbols and date range"}

        # Initialize backtest state
        balance = initial_balance
        positions: dict[str, dict] = {}
        trades: list[dict] = []
        equity_curve: list[dict] = []
        events: list[dict] = []
        ml_predictions = {"correct": 0, "total": 0}

        from core.strategy.indicators import compute_all, evaluate_condition

        # Main backtest loop
        for slice_data in feeder:
            ts = slice_data["timestamp"]

            # First check exits for existing positions
            for sym in list(positions.keys()):
                pos = positions[sym]
                for strategy in strategy_configs:
                    if strategy.name != pos.get("strategy_name"):
                        continue
                    for interval in strategy.timeframes:
                        df = feeder.get_all_data_for_symbol(sym, interval)
                        if len(df) < 50:
                            continue
                        df = df[df.index <= ts].copy()
                        df = compute_all(df, strategy.indicators)

                        exit_conditions = strategy.exit_conditions.get(pos["side"], [])
                        for cond in exit_conditions:
                            mask = evaluate_condition(df, cond)
                            if hasattr(mask, 'iloc') and mask.iloc[-1]:
                                exit_price = float(df["close"].iloc[-1])
                                entry_price = pos["entry_price"]
                                qty = pos["quantity"]
                                amount = pos["amount_usdt"]
                                if pos["side"] == "long":
                                    pnl = (exit_price - entry_price) * qty
                                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                                else:
                                    pnl = (entry_price - exit_price) * qty
                                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                                trades.append({
                                    "symbol": sym, "side": pos["side"],
                                    "entry_price": round(entry_price, 4),
                                    "exit_price": round(exit_price, 4),
                                    "quantity": round(qty, 6), "pnl": round(pnl, 2),
                                    "pnl_pct": round(pnl_pct, 2),
                                    "strategy": strategy.name,
                                    "opened_at": str(pos.get("opened_at", ts)),
                                    "closed_at": str(ts),
                                    "amount_usdt": round(amount, 2),
                                })
                                balance += amount + pnl
                                events.append({
                                    "time": str(ts), "type": "exit",
                                    "symbol": sym, "price": exit_price,
                                    "pnl": round(pnl, 2),
                                })
                                del positions[sym]
                                break
                        if sym not in positions:
                            break
                    if sym not in positions:
                        break

            # Then check entries
            for strategy in strategy_configs:
                for sym in symbols:
                    if sym in positions:
                        continue
                    for interval in strategy.timeframes:
                        df = feeder.get_all_data_for_symbol(sym, interval)
                        if len(df) < 50:
                            continue
                        df = df[df.index <= ts].copy()
                        df = compute_all(df, strategy.indicators)

                        long_active = False
                        short_active = False
                        for side in ["long", "short"]:
                            for cond in strategy.entry_conditions.get(side, []):
                                mask = evaluate_condition(df, cond)
                                met = bool(hasattr(mask, 'iloc') and mask.iloc[-1])
                                if met and side == "long":
                                    long_active = True
                                elif met and side == "short":
                                    short_active = True

                        if long_active and short_active:
                            continue  # ambiguous
                        side = "long" if long_active else "short" if short_active else None
                        if side is None:
                            continue

                        price = float(df["close"].iloc[-1])
                        amount_usdt = balance * 0.1
                        qty = amount_usdt / price if price > 0 else 0
                        if qty <= 0:
                            continue

                        trade_group = f"bt_{sym}_{int(ts.timestamp())}"
                        positions[sym] = {
                            "symbol": sym, "side": side,
                            "quantity": qty, "entry_price": price,
                            "amount_usdt": amount_usdt,
                            "strategy_name": strategy.name,
                            "opened_at": str(ts), "trade_group": trade_group,
                        }
                        events.append({
                            "time": str(ts), "type": "entry",
                            "symbol": sym, "side": side, "price": price,
                            "qty": round(qty, 6), "amount_usdt": round(amount_usdt, 2),
                            "strategy": strategy.name,
                        })
                        balance -= amount_usdt
                        break

            # Record equity at each timestamp
            invested = sum(p.get("amount_usdt", 0) for p in positions.values())
            equity_curve.append({
                "time": str(ts), "equity": round(balance + invested, 2),
                "balance": round(balance, 2), "invested": round(invested, 2),
            })

        final_balance = balance
        metrics = calculate_metrics(trades, equity_curve, initial_balance, final_balance)
        metrics["ml_accuracy_pct"] = round(
            ml_predictions["correct"] / ml_predictions["total"] * 100
            if ml_predictions["total"] > 0 else 0, 1)
        metrics["runtime_seconds"] = round(time.time() - t0, 1)

        return {
            "trades": trades, "equity_curve": equity_curve, "events": events,
            "metrics": metrics, "final_balance": round(final_balance, 2),
            "initial_balance": initial_balance,
            "strategies": strategies, "symbols": symbols,
            "date_start": date_start, "date_end": date_end, "mode": mode,
        }

    run_with_exit_evaluation = run
    # run_with_exit_evaluation is an alias — the main run() method already
    # evaluates exit conditions at each timestamp, so they are equivalent.
