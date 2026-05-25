import asyncio
from dataclasses import dataclass

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.risk.circuit_breaker import CircuitBreaker
from core.risk.position_sizer import PositionSizer


@dataclass
class RiskResult:
    approved: bool
    reason: str = ""
    adjusted_quantity: float | None = None
    adjusted_stop_loss: float | None = None
    adjusted_leverage: int | None = None


class RiskManager:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.breaker = CircuitBreaker(
            max_daily_drawdown_pct=config.hard_limits.max_daily_drawdown_pct,
            max_weekly_drawdown_pct=config.hard_limits.max_weekly_drawdown_pct,
            max_daily_loss_usdt=config.hard_limits.max_daily_loss_usdt,
            max_consecutive_losses=config.hard_limits.max_consecutive_losses,
        )
        self.sizer = PositionSizer(config.hard_limits, config.soft_params,
                                    config.core_capital_pct, config.satellite_capital_pct)
        self._running = False
        self._open_positions: dict[str, dict] = {}
        self._pending_signals: set[str] = set()  # symbols with approved-but-not-yet-opened positions
        self._account_balance: float = 0.0
        self._last_breaker_alert_time: float = 0.0

    async def start(self):
        self._running = True
        self.event_bus.subscribe(EventType.STRATEGY_SIGNAL, self._on_signal)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._on_position_update)

    async def _on_signal(self, event: Event):
        signal = event.data
        from loguru import logger
        logger.info(f"RiskManager received signal: {signal.get('symbol')} {signal.get('side')} "
                     f"qty={signal.get('quantity',0):.4f} price={signal.get('price',0):.2f}")
        result = await self.check_signal(signal)
        if result.approved:
            logger.info(f"RiskManager APPROVED: {signal.get('symbol')} → ORDER_REQUEST")
            symbol = signal.get("symbol", "")
            # Immediately reserve the symbol to prevent same-symbol race
            # (POSITION_UPDATE arrives async, after order execution)
            self._pending_signals.add(symbol)
            signal["quantity"] = result.adjusted_quantity or signal.get("quantity", 0)
            signal["stop_loss"] = result.adjusted_stop_loss
            signal["leverage"] = result.adjusted_leverage or signal.get("leverage", self.config.soft_params.leverage)
            await self.event_bus.publish(Event(EventType.ORDER_REQUEST, signal))
        else:
            from loguru import logger
            logger.info(f"RiskManager REJECTED: {signal.get('symbol')} — {result.reason}")
            # Only alert on critical rejections; routine ones (e.g. position exists, max trades) are silent
            is_critical = "Circuit breaker" in result.reason or "loss limit" in result.reason
            if is_critical:
                import time as _time
                now = _time.time()
                # Circuit breaker trips are already published via _trip_callback;
                # only throttle alert for non-breaker loss-limit rejections
                if "Circuit breaker" not in result.reason:
                    if now - self._last_breaker_alert_time >= 300:
                        self._last_breaker_alert_time = now
                        await self._log_risk_event("signal_rejected", "warning", result.reason, "RiskManager")

    async def check_signal(self, signal: dict) -> RiskResult:
        # Step 1: Circuit breaker (covers drawdown, daily loss, consecutive losses)
        from loguru import logger
        bal = self._account_balance
        price = signal.get("price", 0)
        soft = self.sizer.soft
        hard = self.sizer.hard
        pos_type = signal.get("position_type", "satellite")
        cap_pct = self.sizer.core_capital_pct if pos_type == "core" else self.sizer.satellite_capital_pct
        logger.info(f"check_signal: bal={bal:.2f} price={price:.4f} type={pos_type} "
                     f"soft.pp={soft.position_size_pct} soft.sl={soft.stop_loss_pct} "
                     f"hard.maxpp={hard.max_position_size_pct} cap_pct={cap_pct}")
        capital_pool = bal * cap_pct
        risk = capital_pool * (soft.position_size_pct / 100)
        qty_test = risk / price if price > 0 else 0
        logger.info(f"check_signal calc: cap_pool={capital_pool:.2f} risk={risk:.4f} qty_test={qty_test:.6f}")
        tripped, reason = self.breaker.check()
        if tripped:
            if self.breaker.is_new_trip():
                drawdown = (
                    (self.breaker.peak_equity - self.breaker.current_equity) / self.breaker.peak_equity * 100
                ) if self.breaker.peak_equity > 0 else 0
                task = asyncio.create_task(self._trip_callback({
                    "reason": reason,
                    "daily_drawdown_pct": drawdown,
                    "daily_pnl": self.breaker.daily_pnl,
                    "consecutive_losses": self.breaker.consecutive_losses,
                    "open_positions": self._open_positions.copy(),
                }))
                task.add_done_callback(
                    lambda t: logger.error(f"trip_callback failed: {t.exception()}") if t.exception() else None
                )
            return RiskResult(approved=False, reason=f"Circuit breaker tripped: {reason}")

        # Step 2: Total exposure check
        total_exposure = sum(p.get("position_value", 0) for p in self._open_positions.values())
        exposure_pct = (total_exposure / self._account_balance * 100) if self._account_balance > 0 else 0
        if exposure_pct >= self.config.hard_limits.max_total_exposure_pct:
            return RiskResult(approved=False, reason=f"Total exposure {exposure_pct:.1f}% exceeds limit")

        # Step 3: Position size check
        symbol = signal.get("symbol", "")
        price = signal.get("price", 0)
        qty, risk_amount = self.sizer.calculate_position_size(
            self._account_balance, price, signal.get("position_type", "satellite")
        )
        if qty <= 0:
            return RiskResult(approved=False, reason="Insufficient balance for position sizing")

        # Step 4: Leverage check
        leverage = signal.get("leverage", self.config.soft_params.leverage)
        if leverage > self.config.hard_limits.max_leverage:
            leverage = self.config.hard_limits.max_leverage

        # Step 5: Stop loss check
        entry_price = signal.get("price", 0)
        side = signal.get("side", "long")
        sl_price = self.sizer.calculate_stop_loss(entry_price, side)

        # Step 6: Same symbol check (includes pending signals to prevent race)
        if symbol in self._open_positions or symbol in self._pending_signals:
            return RiskResult(approved=False, reason=f"Position already open for {symbol}")

        # Step 7: Max open trades check
        if len(self._open_positions) >= self.config.hard_limits.max_open_trades:
            return RiskResult(approved=False, reason=f"Max open trades {self.config.hard_limits.max_open_trades} reached")

        return RiskResult(
            approved=True,
            adjusted_quantity=qty,
            adjusted_stop_loss=sl_price,
            adjusted_leverage=leverage,
        )

    async def _trip_callback(self, data: dict):
        """Publish critical RISK_BREACH when breaker first trips (deduped by is_new_trip)."""
        await self.event_bus.publish(Event(EventType.RISK_BREACH, {
            "event_type": "circuit_breaker_trip",
            "level": "critical",
            "detail": data["reason"],
            "daily_drawdown_pct": data.get("daily_drawdown_pct", 0),
            "daily_pnl": data.get("daily_pnl", 0),
            "consecutive_losses": data.get("consecutive_losses", 0),
            "open_positions": data.get("open_positions", {}),
            "triggered_by": "CircuitBreaker",
        }))

    async def _on_position_update(self, event: Event):
        data = event.data
        symbol = data.get("symbol", "")
        pnl = data.get("pnl", 0)
        # Clear pending flag — position is now open or closed
        self._pending_signals.discard(symbol)
        if data.get("closed"):
            self._open_positions.pop(symbol, None)
        else:
            # Merge with existing data so partial updates don't lose fields
            existing = self._open_positions.get(symbol, {})
            merged = {**existing, **data}
            self._open_positions[symbol] = merged
        # Track PnL from all closes (full and partial) for circuit breaker stats
        if pnl != 0:
            self.breaker.add_trade_result(pnl)

    def update_balance(self, balance: float):
        self._account_balance = balance
        # Total equity = cash + sum of open position values (at entry cost).
        # Using cash alone causes false drawdown trips when positions are opened:
        # opening a position reduces cash but total portfolio value is unchanged.
        total_invested = sum(p.get("amount_usdt", 0) for p in self._open_positions.values())
        self.breaker.set_equity(balance + total_invested)

    async def _log_risk_event(self, event_type: str, level: str, detail: str, triggered_by: str):
        await self.event_bus.publish(Event(EventType.RISK_BREACH, {
            "event_type": event_type, "level": level,
            "detail": detail, "triggered_by": triggered_by,
        }))
        await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
            "level": level, "type": event_type,
            "message": detail,
        }))

    def get_open_positions(self) -> dict:
        return self._open_positions.copy()

    async def stop(self):
        self._running = False
        self.event_bus.unsubscribe(EventType.STRATEGY_SIGNAL, self._on_signal)
        self.event_bus.unsubscribe(EventType.POSITION_UPDATE, self._on_position_update)
