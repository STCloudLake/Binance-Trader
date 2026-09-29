"""Event-driven executor — consumes precomputed signal matrix, produces trades."""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from core.backtest.signal_matrix import SignalMatrix
from core.backtest.trade_book import close_position
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
                 max_positions: int = 15,
                 cost_config=None,
                 strategy_risk: dict | None = None):
        self.sizer = sizer
        self.hard_limits = hard_limits
        self.per_strategy_isolation = per_strategy_isolation
        self.max_positions = max_positions
        self.cost_config = cost_config  # optional: (enabled, fee_pct, spread_dict)
        # strategy_name → {max_hold_hours, use_indicator_exits}
        self.strategy_risk = strategy_risk or {}

    def _pkey(self, sym: str, s_name: str = "") -> str:
        return f"{s_name}|{sym}" if self.per_strategy_isolation else sym

    @staticmethod
    def _volatility_expanding(df: pd.DataFrame) -> pd.Series:
        """Vectorized volatility-expansion flag (mirrors the legacy engine).

        True when the 10-bar return std exceeds the 21-bar std — i.e. volatility
        is picking up. Position size is reduced when this is True.
        """
        if df is None or len(df) < 21:
            return pd.Series(False, index=df.index if df is not None else [], dtype=bool)
        ret = df["close"].pct_change()
        recent = ret.rolling(10).std()
        hist = ret.rolling(21).std()
        flag = (recent > hist)
        flag[recent.isna() | hist.isna()] = False
        return flag.astype(bool)

    def _close_position(self, pos_key: str, pos: dict, exit_price: float,
                        ts, reason: str, trades: list, balance: float,
                        positions: dict, per_matrix: dict) -> float:
        """Close a position — delegates to the shared trade book."""
        cost_fn = None
        if self.cost_config:
            enabled, fee_pct, spread_dict = self.cost_config
            if enabled:
                def cost_fn(entry_price, exit_p, qty, sym):  # noqa: F811
                    spread_pct = spread_dict.get(sym, 0.03) / 100.0
                    fee = fee_pct / 100.0
                    entry_notional = qty * entry_price
                    exit_notional = qty * exit_p
                    return ((entry_notional + exit_notional) * fee
                            + (entry_notional + exit_notional) * (spread_pct / 2.0))

        return close_position(
            pos_key, pos, exit_price, ts, reason, trades, balance, positions, per_matrix,
            cost_fn=cost_fn,
        )

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

        # Precompute the volatility-expansion flag per (symbol, timeframe)
        vol_cache: dict[tuple[str, str], pd.Series] = {}
        for _sym, tf_map in (matrix.price_data or {}).items():
            for _tf, _df in (tf_map or {}).items():
                vol_cache[(_sym, _tf)] = self._volatility_expanding(_df)

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
                    "gross_win_pnl": 0.0, "gross_loss_pnl": 0.0,
                }

        total_steps = len(timestamps)
        for step, ts in enumerate(timestamps):

            # Yield GIL every 200 ticks so the asyncio event loop can serve HTTP
            if step % 200 == 0:
                time.sleep(0)

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

                # Trailing stop update
                trailing_pct = pos.get("trailing_stop_pct", 0) / 100.0
                if trailing_pct > 0:
                    best = pos.get("best_price", pos["entry_price"])
                    if side == "long" and current_price > best:
                        pos["best_price"] = current_price
                    elif side == "short" and current_price < best:
                        pos["best_price"] = current_price
                    best_price = pos["best_price"]
                    if side == "long":
                        pos["stop_loss"] = best_price * (1 - trailing_pct)
                    else:
                        pos["stop_loss"] = best_price * (1 + trailing_pct)

                # Indicator exit check — evaluated on EVERY timeframe the strategy
                # configures, matching the legacy engine's exit loop (it iterates
                # strategy.timeframes). Checking only the position's own timeframe
                # made multi-timeframe strategies exit at different ticks.
                if pos.get("use_indicator_exits", True):
                    exit_timeframes = self.strategy_risk.get(s_name, {}).get("timeframes") or [tf]
                    # Iterate in the strategy's DECLARED timeframe order and close at
                    # the first match, exactly like the legacy loop. The exit price is
                    # the triggering timeframe's close (legacy: float(df["close"].iloc[-1])
                    # for that interval), not the position's own timeframe close.
                    for exit_tf in exit_timeframes:
                        if not matrix.get_exit(s_name, sym, exit_tf, side, ts):
                            continue
                        exit_price = current_price
                        tf_df = matrix.price_data.get(sym, {}).get(exit_tf)
                        if tf_df is not None and len(tf_df) > 0:
                            # Last bar of that timeframe at or before ts (mirrors
                            # legacy's `df[df.index <= ts].iloc[-1]`); an exact
                            # get_loc would fail for higher timeframes on most ticks.
                            pos_i = int(tf_df.index.searchsorted(ts, side="right")) - 1
                            if pos_i >= 0:
                                exit_price = float(tf_df.iloc[pos_i]["close"])
                        balance = self._close_position(
                            pos_key, pos, exit_price, ts, "indicator",
                            trades, balance, positions, per_matrix)
                        break

                if pos_key not in positions:
                    continue

                # Max hold time check (strategy.risk_exit.max_hold_hours)
                max_hours = pos.get("max_hold_hours", 0) or 0
                if max_hours > 0:
                    try:
                        held_hours = (pd.Timestamp(ts) - pd.Timestamp(pos["opened_at"])).total_seconds() / 3600
                    except Exception:
                        held_hours = 0.0
                    if held_hours >= max_hours:
                        balance = self._close_position(
                            pos_key, pos, current_price, ts, "max_hold",
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

                # ── Position sizing — identical rules to the legacy engine ──
                vol_series = vol_cache.get((sym, tf))
                vol_expanding = bool(vol_series.get(ts, False)) if vol_series is not None else False
                qty, risk_amount = self.sizer.calculate_position_size(
                    account_balance=balance, current_price=price,
                    position_type="satellite",
                    volatility_expanding=vol_expanding)

                # Kelly-lite: when the strategy declares an explicit stop distance,
                # risk 1% of capital per trade (mirrors BacktestEngine exactly).
                risk_cfg = self.strategy_risk.get(s_name, {})
                sl_pct_override = risk_cfg.get("stop_loss_pct") or 0
                if sl_pct_override:
                    qty_risk = (balance * 0.01) / (price * (sl_pct_override / 100.0))
                    qty = min(qty, qty_risk) if qty_risk > 0 else qty

                # Hard cap on position notional
                max_amount = balance * (self.hard_limits.max_position_size_pct / 100)
                if risk_amount > max_amount:
                    risk_amount = max_amount
                    qty = risk_amount / price if price > 0 else 0

                if qty <= 0:
                    continue

                amount_usdt = qty * price
                if amount_usdt > balance * 0.95:
                    continue

                pos_counter += 1
                balance -= amount_usdt

                # Stop-loss / trailing distance: honour the strategy's risk_exit
                # overrides exactly like the legacy engine does.
                sl_pct_override = risk_cfg.get("stop_loss_pct")
                if sl_pct_override:
                    sl = (price * (1 - sl_pct_override / 100.0) if side == "long"
                          else price * (1 + sl_pct_override / 100.0))
                else:
                    sl = self.sizer.calculate_stop_loss(entry_price=price, side=side)
                tps = self.sizer.calculate_take_profits(entry_price=price, side=side)
                trailing_override = risk_cfg.get("trailing_stop_pct")
                trailing_pct = (float(trailing_override) if trailing_override is not None
                                else self.sizer.trailing_stop_distance_pct())

                positions[pos_key] = {
                    "symbol": sym, "side": side,
                    "quantity": qty, "entry_price": price,
                    "amount_usdt": amount_usdt,
                    "strategy_name": s_name,
                    "opened_at": str(ts), "trade_group": f"bt_{pos_counter}_{int(ts.timestamp())}",
                    "stop_loss": sl, "take_profits": tps,
                    "trailing_stop_pct": trailing_pct,
                    "best_price": price,
                    "timeframe": tf, "reduce_count": 0,
                    # Per-strategy risk-exit overrides (empty for GA strategies,
                    # which never set risk_exit).
                    "max_hold_hours": float(risk_cfg.get("max_hold_hours", 0) or 0),
                    "use_indicator_exits": bool(risk_cfg.get("use_indicator_exits", True)),
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

        # Realise the closing PnL into the final equity point, otherwise the last
        # equity value (and therefore final_balance / total_return) omitted the
        # PnL of positions still open at the end of the window.
        if equity_curve and not positions:
            equity_curve[-1]["balance"] = round(balance, 2)
            equity_curve[-1]["invested"] = 0.0
            equity_curve[-1]["equity"] = round(balance, 2)

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
            final_balance=round(balance, 2),
            runtime_seconds=round(time.time() - t0, 2),
        )
