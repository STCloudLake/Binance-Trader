import asyncio
import time
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

    Phase P3 adds a third dimension to the trailing stop: its **distance** can be
    driven by a forecast conditional volatility instead of a fixed percentage.
    Science: conditional volatility is predictable (ARCH/GARCH — Tsay ch. 3) while
    the sign of the next return is not, so the forecast is spent on risk.  A
    turbulent regime widens the trailing distance (fewer premature exits), a calm
    one tightens it (profits are given back less).  The switch is
    ``risk.vol_targeting.enabled`` and ships **false**, so the default path below
    is the unchanged fixed-percentage one.
    """

    #: How long a forecast is reused before the history is re-read (seconds).  The
    #: guard ticks every ``_check_interval_sec``; without this cache it would hit
    #: the exchange for klines on every tick.
    _VOL_CACHE_TTL_SEC = 300.0
    #: Bars requested per refresh — enough for the estimator's window to matter,
    #: cheap enough to keep the REST call small.
    _VOL_HISTORY_LIMIT = 600
    #: Interval used when a position does not record its own timeframe.
    _DEFAULT_INTERVAL = "1h"

    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self._running = False
        self._executor = None
        self._market_data = None
        self._risk_manager = None
        self._task: asyncio.Task | None = None
        self._check_interval_sec = 15  # check every 15 seconds
        # (symbol, interval) → (monotonic_timestamp, vol_pct)
        self._vol_cache: dict[tuple[str, str], tuple[float, float]] = {}
        # Shared trailing-distance helper (same semantics as the backtest engines).
        #
        # NOTE: `risk.vol_targeting` is passed through so the *live* trailing
        # distance can scale with the forecast.  Sizing itself is consumed by
        # `core/risk/manager.py:check_signal`, which is outside this phase's write
        # scope: enabling the block scales the guard's stop width here, while the
        # notional target only takes effect where a caller supplies
        # `forecast_vol_pct` to `PositionSizer.calculate_position_size`.
        from core.risk.position_sizer import PositionSizer
        self._sizer = PositionSizer(
            config.hard_limits, config.soft_params,
            getattr(config, "core_capital_pct", 0.7),
            getattr(config, "satellite_capital_pct", 0.3),
            getattr(config, "risk_vol_targeting", None),
        )

    def wire(self, executor, market_data, risk_manager=None):
        self._executor = executor
        self._market_data = market_data
        self._risk_manager = risk_manager

    # ── volatility plumbing (Phase P3) ──────────────────────────────────

    def vol_targeting_enabled(self) -> bool:
        """True only when ``risk.vol_targeting.enabled`` is set in the config."""
        return self._sizer.vol_targeting_enabled()

    async def forecast_vol_pct(self, symbol: str, interval: str | None = None,
                               timeframe: str | None = None) -> float | None:
        """Forecast conditional volatility in **percent of price per bar**.

        Returns ``None`` whenever the forecast is unavailable — vol targeting off,
        no market-data source, a short/failed history, a zero estimate, or a
        **spliced** series (a calendar gap beyond ``_VOL_MAX_GAP_BARS`` bar
        lengths).  Every caller then falls back to the fixed percentage, which is
        the documented behaviour and what keeps this change inert by default.

        Spliced-series refusal (re-audit finding 1)
        ------------------------------------------
        This guard used to build a ``DatetimeIndex`` frame from the provider's
        history and feed it straight to the estimator, so the *same* cached series
        that ``RiskManager.forecast_vol_pct`` refuses (→ ``None``, fixed sizing)
        produced a forecast here (the re-audit measured ``0.42584 %/bar`` on the
        live spliced cache; the test fixture in
        ``tests/test_reaudit_fixes.py`` produces ``0.41803165815 %/bar`` on a
        synthetic 100-bar hole) — and that number drives the **live trailing-stop
        distance**, i.e. real risk.  The check is now the *same* function the
        manager uses — :func:`core.risk.manager._series_has_gap`, with its own
        ``_VOL_MAX_GAP_BARS`` / bar-length table — so the two consumers of one
        series cannot disagree: a hole beyond 1.5 bar lengths refuses the forecast
        here too, :meth:`_resolve_vol_pct` returns ``None`` and the trailing stop
        falls back to ``hard_limits.trailing_stop_distance_pct`` (the documented
        fixed rule).  A refusal is logged, never silent.
        """
        if not self.vol_targeting_enabled():
            return None
        if not self._market_data:
            return None
        vt = self._sizer.vol_targeting
        tf = interval or timeframe or self._DEFAULT_INTERVAL
        key = (str(symbol), str(tf))
        now = time.monotonic()
        cached = self._vol_cache.get(key)
        if cached is not None and now - cached[0] < self._VOL_CACHE_TTL_SEC:
            return cached[1]
        getter = getattr(self._market_data, "get_historical", None)
        if getter is None:
            return None
        try:
            df = await getter(symbol, tf, limit=self._VOL_HISTORY_LIMIT)
        except Exception as e:  # never let a data hiccup break the risk loop
            logger.warning(f"PositionGuard: vol history unavailable for {symbol} "
                           f"{tf}: {e}")
            return None
        if df is None or len(df) < 2:
            return None
        # The splice guard, shared verbatim with RiskManager (re-audit finding 1):
        # one cached series must not be "too spliced to size from" but good enough
        # to set a live stop.  Refusing falls back to the fixed trailing distance.
        try:
            from core.risk.manager import _VOL_MAX_GAP_BARS, _series_has_gap
            if _series_has_gap(getattr(df, "index", ()), tf):
                logger.warning(
                    f"PositionGuard: refusing a volatility forecast for {symbol} "
                    f"{tf} — the series carries a calendar gap beyond "
                    f"{_VOL_MAX_GAP_BARS} bars; the trailing stop falls back "
                    f"to the fixed distance (run scripts/check_data_integrity.py)")
                return None
        except Exception as e:  # never let the guard itself break the risk loop
            logger.warning(f"PositionGuard: gap check failed for {symbol} {tf}: {e}")
            return None
        try:
            from core.ml.volatility import forecast_vol, to_pct
            vol = to_pct(forecast_vol(
                df, method=getattr(vt, "method", "ewma"),
                window=int(getattr(vt, "window", 500)),
                lam=float(getattr(vt, "lam", 0.94)),
                interval=tf))
        except Exception as e:
            logger.warning(f"PositionGuard: vol forecast failed for {symbol}: {e}")
            return None
        if not vol or vol <= 0.0:
            return None
        self._vol_cache[key] = (now, float(vol))
        return float(vol)

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

            # ---- Stop-loss check (BEFORE emergency stop) ----
            sl_price = pos.get("stop_loss")
            if sl_price and sl_price > 0:
                sl_hit = (side == "long" and price <= sl_price) or \
                         (side == "short" and price >= sl_price)
                if sl_hit:
                    await self._stop_loss_close(symbol, pos, price, sl_price)
                    continue

            # ---- Emergency stop ----
            if getattr(limits, "emergency_stop_enabled", False):
                threshold = getattr(limits, "emergency_stop_threshold_pct", -5.0)
                if pnl_pct <= threshold:
                    await self._emergency_close(symbol, pos, price, pnl_pct)
                    continue

            # ---- Trailing stop ----
            if getattr(limits, "trailing_stop_enabled", False):
                await self._update_trailing_stop(symbol, pos, price, side, pnl_pct)

    async def _stop_loss_close(self, symbol: str, pos: dict, price: float, sl_price: float):
        """Execute stop-loss close and publish alert."""
        logger.warning(
            f"STOP LOSS: {symbol} {pos['side']} @ {price:.2f} "
            f"(SL={sl_price:.2f})"
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
                "level": "warning",
                "type": "stop_loss",
                "message": f"止损触发 {symbol}: PnL={trade_pnl:.2f} USDT (SL={sl_price:.2f})",
                "symbol": symbol,
            }))
            logger.info(f"Stop loss: {symbol} closed, PnL={trade_pnl:.2f}, Balance={new_balance:.0f}")

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

    async def _resolve_vol_pct(self, symbol: str, pos: dict) -> float | None:
        """Forecast vol (%) for a position, honouring its recorded entry width.

        A position opened while vol targeting was on carries ``stop_vol_pct`` —
        the forecast volatility captured **at entry**.  Reusing it keeps the stop
        width of an open trade stable (a stop that re-widens every 15 s as the
        forecast jitters is a stop nobody can reason about), and it is the same
        value the entry-time stop was computed from.  Positions without it (all
        positions while the switch is off, and everything opened before P3) get
        ``None`` and therefore the fixed distance.
        """
        if not self.vol_targeting_enabled():
            return None
        recorded = pos.get("stop_vol_pct")
        if recorded is not None:
            try:
                val = float(recorded)
            except (TypeError, ValueError):
                val = 0.0
            if val > 0.0:
                return val
        return await self.forecast_vol_pct(
            symbol, timeframe=pos.get("timeframe"))

    async def _update_trailing_stop(self, symbol: str, pos: dict, price: float,
                                     side: str, _pnl_pct: float):
        """Move stop_loss toward current price, but only in the favorable direction.
        Long: stop moves UP toward price.  Short: stop moves DOWN toward price."""
        entry = pos["entry_price"]
        # Single source of truth for the trailing distance: a per-position override
        # (from strategy.risk_exit) wins, otherwise the shared PositionSizer helper
        # reads hard_limits.trailing_stop_distance_pct / trailing_stop_enabled. This
        # keeps live trailing identical to what the backtest engines simulate.
        #
        # Phase P3: when `risk.vol_targeting.enabled` and a forecast is available,
        # the same helper returns a forecast-scaled distance instead of the fixed
        # 2 %.  With the switch off (default) `vol_pct` is None and the number is
        # exactly the pre-P3 one.
        override = pos.get("trailing_stop_pct")
        if override is not None:
            distance_pct = float(override)
        else:
            vol_pct = await self._resolve_vol_pct(symbol, pos)
            distance_pct = self._sizer.trailing_stop_distance_pct(
                forecast_vol_pct=vol_pct)
        if distance_pct <= 0:
            return
        current_sl = pos.get("stop_loss")

        # Calculate the trailing stop price
        if side == "long":
            new_sl = price * (1 - distance_pct / 100)
            entry_sl = entry * (1 - distance_pct / 100)
            # Floor: the pre-P3 "at worst 1% below entry", generalised so a wide
            # vol-scaled distance still cannot place the stop between price and
            # entry-level protection... it stays the WIDER of the two, i.e. the
            # floor never tightens a legitimately wide stop.
            floor_pct = max(1.0, distance_pct)
            floor_sl = max(entry_sl, entry * (1 - floor_pct / 100))
            if current_sl:
                new_sl = max(new_sl, current_sl, floor_sl)  # only move up, respect floor
            else:
                new_sl = max(new_sl, floor_sl)
        else:  # short
            new_sl = price * (1 + distance_pct / 100)
            entry_sl = entry * (1 + distance_pct / 100)
            ceiling_pct = max(1.0, distance_pct)
            ceiling_sl = min(entry_sl, entry * (1 + ceiling_pct / 100))
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
            # Persist the move: a trailing stop that lived only in memory was
            # reset to the executor's 2% default by the next restart, i.e. a stop
            # already trailed into profit silently went back to where it started.
            persist = getattr(self._executor, "update_stop_loss", None)
            if persist is not None:
                try:
                    await persist(symbol, new_sl)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(f"Could not persist trailing stop for {symbol}: {e}")
            old_sl_str = f"{current_sl:.2f}" if current_sl else "none"
            logger.debug(f"Trailing stop: {symbol} {side} SL {old_sl_str} → {new_sl:.2f} (price={price:.2f})")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("PositionGuard stopped")
