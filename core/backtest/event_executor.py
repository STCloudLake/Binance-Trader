"""Event-driven executor — consumes precomputed signal matrix, produces trades."""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from core.backtest.signal_matrix import SignalMatrix
from core.risk.position_sizer import PositionSizer


@dataclass
class ExecutorResult:
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    per_matrix: dict = field(default_factory=dict)
    final_balance: float = 0.0
    runtime_seconds: float = 0.0


class EventDrivenExecutor:
    """Consumes a precomputed SignalMatrix and simulates trade execution.

    Reuses the same position sizing, stop-loss, and take-profit logic as the
    legacy engine. The key difference: entry/exit signals are looked up from
    the matrix instead of computed on-the-fly.
    """

    def __init__(self, sizer: PositionSizer, hard_limits,
                 per_strategy_isolation: bool = False,
                 max_positions: int = 15):
        self.sizer = sizer
        self.hard_limits = hard_limits
        self.per_strategy_isolation = per_strategy_isolation
        self.max_positions = max_positions

    def _pkey(self, sym: str, s_name: str = "") -> str:
        return f"{s_name}|{sym}" if self.per_strategy_isolation else sym

    def _close_position(self, pos_key: str, pos: dict, exit_price: float,
                        ts, reason: str, trades: list, balance: float,
                        positions: dict, per_matrix: dict) -> float:
        """Close a position. Identical logic to BacktestEngine._close_position."""
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
        })
        balance += amount + pnl

        if strategy_name in per_matrix and sym in per_matrix[strategy_name]:
            cell = per_matrix[strategy_name][sym]
            cell["trades"] += 1
            cell["pnl"] += pnl
            if pnl > 0:
                cell["winning"] += 1
            else:
                cell["losing"] += 1
            if side == "long":
                cell["long_trades"] += 1
            else:
                cell["short_trades"] += 1

        del positions[pos_key]
        return balance

    def run(self, matrix: SignalMatrix,
            initial_balance: float = 10000.0,
            progress_callback=None) -> ExecutorResult:
        """Execute all trades by consuming the signal matrix chronologically."""
        t0 = time.time()

        if matrix.signals.empty:
            return ExecutorResult(
                trades=[], equity_curve=[],
                per_matrix={}, final_balance=initial_balance,
                runtime_seconds=round(time.time() - t0, 2),
            )

        timestamps = matrix.signals.columns
        balance = initial_balance
        positions: dict[str, dict] = {}
        trades: list[dict] = []
        equity_curve: list[dict] = []
        pos_counter = 0

        # ── Initialize per_matrix ──
        per_matrix: dict[str, dict[str, dict]] = {}
        strategy_names = set(idx[0] for idx in matrix.signals.index)
        symbols = set(idx[1] for idx in matrix.signals.index)
        for s_name in strategy_names:
            per_matrix[s_name] = {}
            for sym in symbols:
                per_matrix[s_name][sym] = {
                    "trades": 0, "pnl": 0.0, "winning": 0, "losing": 0,
                    "long_trades": 0, "short_trades": 0,
                }

        total_steps = len(timestamps)
        for step, ts in enumerate(timestamps):

            if progress_callback and (step % 50 == 0 or step == total_steps - 1):
                progress_callback(step + 1, total_steps, ts)

            # ── Check exits (SL, TP, indicator exits) ──
            for pos_key in list(positions.keys()):
                pos = positions[pos_key]
                sym = pos["symbol"]
                s_name = pos.get("strategy_name", "")
                side = pos["side"]
                tf = pos.get("timeframe", "1h")

                # Get current price
                price_df = matrix.price_data.get(sym, {}).get(tf)
                if price_df is None:
                    continue
                try:
                    idx = price_df.index.get_loc(ts)
                    if isinstance(idx, slice):
                        idx = idx.stop - 1
                    current_price = float(price_df.iloc[idx]["close"])
                except (KeyError, IndexError):
                    continue

                # Stop-loss check
                sl_price = pos.get("stop_loss", 0)
                if sl_price > 0:
                    hit = (side == "long" and current_price <= sl_price) or \
                          (side == "short" and current_price >= sl_price)
                    if hit:
                        balance = self._close_position(
                            pos_key, pos, sl_price, ts, "stop_loss",
                            trades, balance, positions, per_matrix)
                        continue

                # Take-profit check
                tp_levels = pos.get("take_profits", [])
                for tp_price, tp_pct in tp_levels:
                    hit = (side == "long" and current_price >= tp_price) or \
                          (side == "short" and current_price <= tp_price)
                    if hit:
                        balance = self._close_position(
                            pos_key, pos, tp_price, ts, f"tp_{int(tp_pct*100)}pct",
                            trades, balance, positions, per_matrix)
                        break

                if pos_key not in positions:
                    continue

                # Indicator exit check
                exit_hit = matrix.get_exit(s_name, sym, tf, side, ts)
                if exit_hit:
                    balance = self._close_position(
                        pos_key, pos, current_price, ts, "indicator",
                        trades, balance, positions, per_matrix)

            # ── Check entries ──
            for idx_tuple in matrix.signals.index:
                s_name, sym, tf = idx_tuple
                entry_val = matrix.get_entry(s_name, sym, tf, ts)

                if entry_val == 0:
                    continue

                pos_key = self._pkey(sym, s_name)
                if pos_key in positions:
                    continue
                if len(positions) >= self.max_positions:
                    break

                side = "long" if entry_val == 1 else "short"

                # Get current price
                price_df = matrix.price_data.get(sym, {}).get(tf)
                if price_df is None:
                    continue
                try:
                    idx_val = price_df.index.get_loc(ts)
                    if isinstance(idx_val, slice):
                        idx_val = idx_val.stop - 1
                    price = float(price_df.iloc[idx_val]["close"])
                except (KeyError, IndexError):
                    continue

                # Position sizing
                qty, risk_amount = self.sizer.calculate_position_size(
                    account_balance=balance, current_price=price,
                    position_type="satellite")
                if qty <= 0:
                    continue

                amount_usdt = qty * price
                if amount_usdt > balance * 0.95:
                    continue

                pos_counter += 1
                balance -= amount_usdt

                sl = self.sizer.calculate_stop_loss(entry_price=price, side=side)
                tps = self.sizer.calculate_take_profits(entry_price=price, side=side)

                positions[pos_key] = {
                    "symbol": sym, "side": side,
                    "quantity": qty, "entry_price": price,
                    "amount_usdt": amount_usdt,
                    "strategy_name": s_name,
                    "opened_at": str(ts), "trade_group": f"bt_{pos_counter}_{int(ts.timestamp())}",
                    "stop_loss": sl, "take_profits": tps,
                    "timeframe": tf, "reduce_count": 0,
                }

            # ── Equity curve ──
            invested = sum(p.get("amount_usdt", 0) for p in positions.values())
            equity_curve.append({
                "time": str(ts),
                "equity": round(balance + invested, 2),
                "balance": round(balance, 2),
                "invested": round(invested, 2),
            })

        # ── Force-close remaining positions at last timestamp ──
        last_ts = timestamps[-1]
        for pos_key in list(positions.keys()):
            pos = positions[pos_key]
            sym = pos["symbol"]
            tf = pos.get("timeframe", "1h")
            price_df = matrix.price_data.get(sym, {}).get(tf)
            if price_df is not None:
                try:
                    idx = price_df.index.get_loc(last_ts)
                    if isinstance(idx, slice):
                        idx = idx.stop - 1
                    final_price = float(price_df.iloc[idx]["close"])
                except (KeyError, IndexError):
                    final_price = pos["entry_price"]
            else:
                final_price = pos["entry_price"]
            balance = self._close_position(
                pos_key, pos, final_price, last_ts, "end_of_backtest",
                trades, balance, positions, per_matrix)

        # ── Finalize per_matrix ──
        for s_name in per_matrix:
            for sym in per_matrix[s_name]:
                cell = per_matrix[s_name][sym]
                n = cell["trades"]
                cell["win_rate_pct"] = round(cell["winning"] / n * 100, 1) if n > 0 else 0.0
                cell["pnl"] = round(cell["pnl"], 2)

        return ExecutorResult(
            trades=trades,
            equity_curve=equity_curve,
            per_matrix=per_matrix,
            final_balance=round(equity_curve[-1]["equity"] if equity_curve else balance, 2),
            runtime_seconds=round(time.time() - t0, 2),
        )
