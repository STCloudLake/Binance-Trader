import asyncio
from loguru import logger

from app.event_bus import EventBus, Event, EventType
from app.config import Config


class PositionGuard:
    """Cross-timeframe risk guard — runs independently of strategy evaluation cycles.

    Two protections:
    1. Trailing stop — moves stop_loss in the favorable direction as price moves,
       locking in profits without waiting for the next strategy kline.
    2. Emergency stop — force-closes any position whose unrealized PnL% drops
       below the configured emergency threshold.
    """

    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self._running = False
        self._executor = None
        self._market_data = None
        self._risk_manager = None
        self._task: asyncio.Task | None = None
        self._check_interval_sec = 15  # check every 15 seconds

    def wire(self, executor, market_data, risk_manager=None):
        self._executor = executor
        self._market_data = market_data
        self._risk_manager = risk_manager

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._guard_loop())
        logger.info("PositionGuard started (trailing + emergency stop)")

    async def _guard_loop(self):
        while self._running:
            try:
                await self._check_all_positions()
            except Exception as e:
                logger.warning(f"PositionGuard check failed: {e}")
            await asyncio.sleep(self._check_interval_sec)

    async def _check_all_positions(self):
        if not self._executor or not self._market_data:
            return
        positions = self._executor.get_open_positions()
        if not positions:
            return

        limits = self.config.hard_limits
        for symbol, pos in list(positions.items()):
            # Re-fetch in case position was closed by another path
            if symbol not in self._executor.get_open_positions():
                continue
            pos = self._executor.get_open_positions()[symbol]

            price = self._market_data.get_current_price(symbol)
            if not price:
                continue

            entry = pos["entry_price"]
            qty = pos["quantity"]
            side = pos["side"]

            # Unrealized PnL %
            if side == "long":
                pnl_pct = (price - entry) / entry * 100
            else:
                pnl_pct = (entry - price) / entry * 100

            # ---- Emergency stop ----
            if getattr(limits, "emergency_stop_enabled", False):
                threshold = getattr(limits, "emergency_stop_threshold_pct", -5.0)
                if pnl_pct <= threshold:
                    await self._emergency_close(symbol, pos, price, pnl_pct)
                    continue

            # ---- Trailing stop ----
            if getattr(limits, "trailing_stop_enabled", False):
                await self._update_trailing_stop(symbol, pos, price, side, pnl_pct)

    async def _emergency_close(self, symbol: str, pos: dict, price: float, pnl_pct: float):
        logger.error(
            f"EMERGENCY STOP: {symbol} {pos['side']} uPnL={pnl_pct:.2f}% "
            f"(threshold={self.config.hard_limits.emergency_stop_threshold_pct}%) — force closing"
        )
        result = await self._executor.close_position(symbol, 100, price)
        if result.get("ok"):
            from db.database import atomic_adjust_balance
            invested_returned = result.get("invested_returned", 0)
            trade_pnl = result.get("pnl", 0)
            new_balance = await atomic_adjust_balance(
                invested_returned + trade_pnl, self.config.db_path
            )
            if self._risk_manager:
                self._risk_manager.update_balance(new_balance)
            await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
                "level": "critical",
                "type": "emergency_stop",
                "message": f"紧急止损 {symbol}: PnL={trade_pnl:.2f} USDT ({pnl_pct:.2f}%)",
                "symbol": symbol,
            }))
            logger.info(f"Emergency stop: {symbol} closed, PnL={trade_pnl:.2f}, Balance={new_balance:.0f}")

    async def _update_trailing_stop(self, symbol: str, pos: dict, price: float,
                                     side: str, _pnl_pct: float):
        """Move stop_loss toward current price, but only in the favorable direction.
        Long: stop moves UP toward price.  Short: stop moves DOWN toward price."""
        entry = pos["entry_price"]
        distance_pct = getattr(self.config.hard_limits, "trailing_stop_distance_pct", 2.0)
        current_sl = pos.get("stop_loss")
        existing_entry_sl = pos.get("entry_stop_loss")

        # Calculate the trailing stop price
        if side == "long":
            new_sl = price * (1 - distance_pct / 100)
            entry_sl = entry * (1 - distance_pct / 100)
            floor_sl = max(entry_sl, entry * 0.99)  # at worst 1% below entry
            if current_sl:
                new_sl = max(new_sl, current_sl, floor_sl)  # only move up, respect floor
            else:
                new_sl = max(new_sl, floor_sl)
        else:  # short
            new_sl = price * (1 + distance_pct / 100)
            entry_sl = entry * (1 + distance_pct / 100)
            ceiling_sl = min(entry_sl, entry * 1.01)  # at worst 1% above entry
            if current_sl:
                new_sl = min(new_sl, current_sl, ceiling_sl)  # only move down, respect ceiling
            else:
                new_sl = min(new_sl, ceiling_sl)

        new_sl = round(new_sl, 2)

        # Only update if the stop actually moved favorably
        if current_sl is None:
            should_update = True
        elif side == "long" and new_sl > current_sl + 0.01:
            should_update = True
        elif side == "short" and new_sl < current_sl - 0.01:
            should_update = True
        else:
            should_update = False

        if should_update:
            pos["stop_loss"] = new_sl
            old_sl_str = f"{current_sl:.2f}" if current_sl else "none"
            logger.debug(f"Trailing stop: {symbol} {side} SL {old_sl_str} → {new_sl:.2f} (price={price:.2f})")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("PositionGuard stopped")
