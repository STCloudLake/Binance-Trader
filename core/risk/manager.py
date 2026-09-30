import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.risk.circuit_breaker import CircuitBreaker
from core.risk.position_sizer import PositionSizer

#: Bars kept per ``(symbol, interval)`` from live ``MARKET_KLINE`` events.  The
#: estimator's own default window is 500 (``risk.vol_targeting.window``); 600
#: leaves headroom without keeping a second history in memory.
_VOL_HISTORY_BARS = 600
#: How long a resolved forecast is reused (seconds).  A signal arrives many times
#: per bar (every tick rebuilds the frame), so the estimate must not be recomputed
#: per signal — the same TTL the guard and the executor use for their P3 caches.
_VOL_CACHE_TTL_SEC = 300.0
#: Interval used when a signal records no timeframe (matches PositionGuard).
_DEFAULT_VOL_INTERVAL = "1h"
#: A series is refused as *spliced* when a calendar gap exceeds this many bar
#: lengths (see ``scripts/check_data_integrity.py``).  Measured reason: the
#: shipped ``BTCUSDT/1h`` cache jumps 2026-07-29 → 2026-09-29 inside one "bar"
#: (+27.63 % log return), and an un-clipped RiskMetrics recursion then reports
#: 5.31 %/bar instead of 0.52 %/bar — a 10.1x overstatement that would shrink
#: every vol-targeted position by that factor.  Refusing the forecast falls back
#: to the fixed fraction, which is the pre-P3 behaviour.
_VOL_MAX_GAP_BARS = 1.5

#: Bar length in hours, for the splice guard only (unknown interval → no guard).
_INTERVAL_HOURS: dict[str, float] = {
    "1m": 1 / 60, "3m": 3 / 60, "5m": 5 / 60, "15m": 15 / 60, "30m": 0.5,
    "1h": 1.0, "2h": 2.0, "4h": 4.0, "6h": 6.0, "8h": 8.0, "12h": 12.0,
    "1d": 24.0, "3d": 72.0, "1w": 168.0, "1M": 730.0,
}


def _frame_close_time(value):
    """``close_time`` (ms epoch or datetime-like) → ``pd.Timestamp`` or ``None``.

    ``None`` means "no usable timestamp", which the kline buffer treats as a
    refusal — never as "index this series with 0, 1, 2…" (audit F1: a RangeIndex
    made the splice guard compare ``1 - 0 = 1`` "seconds" and always pass).
    """
    if value is None:
        return None
    try:
        import pandas as pd
        if isinstance(value, pd.Timestamp):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            ms = float(value)
            # Binance sends milliseconds; a seconds value would land in 1970.
            if abs(ms) < 1e11:
                return None
            return pd.Timestamp(ms, unit="ms", tz="UTC")
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    # Production mixes aware (WS ms epoch) and naive (test/parquet) stamps.
    # Comparing aware to naive raises in pandas; UTC-tagging the naive ones (they
    # are already UTC: `close_time` is the exchange's UTC bar close) makes a mixed
    # buffer representable and keeps the gap arithmetic meaningful.
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def _series_has_gap(index, interval: str, max_bars: float = _VOL_MAX_GAP_BARS) -> bool:
    """True when ``index`` (datetimes) carries a gap beyond ``max_bars`` bars.

    The cheapest possible sanity check on a price series before it is fed to a
    variance estimator: an unknown interval (no bar length known) returns False,
    i.e. the guard is inert rather than wrongly refusing a forecast.
    """
    bar = _INTERVAL_HOURS.get(str(interval or "").strip())
    if bar is None or bar <= 0:
        return False
    try:
        times = sorted(t for t in index)
        if len(times) < 3:
            return False
        limit = float(max_bars) * bar
        return any((times[i] - times[i - 1]).total_seconds() / 3600.0 > limit
                   for i in range(1, len(times)))
    except Exception:
        return False


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
        # ``risk.vol_targeting`` is handed to the sizer so the live sizing path can
        # honour the switch.  Passing it is behaviour-neutral while the switch is
        # off: ``vol_scale`` short-circuits to 1.0 (see PositionSizer.vol_scale),
        # so the arithmetic below is byte-for-byte the pre-P3 calculation.
        self.sizer = PositionSizer(config.hard_limits, config.soft_params,
                                    config.core_capital_pct, config.satellite_capital_pct,
                                    getattr(config, "risk_vol_targeting", None))
        self._running = False
        self._boot_positions: dict[str, dict] = {}  # set once at boot via sync_positions()
        self._pending_signals: dict[str, float] = {}  # symbol → timestamp of approval
        self._pending_timeout_sec = 60  # auto-clear pending after 60s
        self._account_balance: float = 0.0
        self._last_breaker_alert_time: float = 0.0
        self._executor = None  # set by wire_executor()
        self._ALERT_THROTTLE_SEC = 300  # minimum interval between repeated breaker alerts
        # ── Volatility-targeting plumbing (Phase P3 gap fix) ──
        # ``(symbol, interval) → (monotonic_ts, vol_pct)`` forecast cache, and
        # ``(symbol, interval) → deque[close]`` fed by live MARKET_KLINE events.
        self._market_data = None          # optional, set by wire_market_data()
        self._vol_cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._kline_history: dict[tuple[str, str], deque] = {}
        self._last_splice_warning: float = 0.0
        self._last_timestamp_warning: float = 0.0

    def wire_executor(self, executor):
        """Receive executor reference for accurate position valuation."""
        self._executor = executor

    def wire_market_data(self, market_data):
        """Inject a market-data source for volatility forecasts (optional).

        Only used when ``risk.vol_targeting.enabled`` is on; without it the
        forecast still resolves from the live ``MARKET_KLINE`` stream and from
        the executor's published forecast (see :meth:`resolve_forecast_vol_pct`).
        """
        self._market_data = market_data

    def sync_positions(self, positions: dict[str, dict]):
        """One-time boot sync from executor — sets the initial position snapshot.

        After boot, all position queries go directly to the executor (single source of truth).
        This snapshot is only a fallback if executor is somehow unavailable.
        """
        self._boot_positions = dict(positions)

    def _get_positions(self) -> dict[str, dict]:
        """Return current open positions from the executor (single source of truth).

        Falls back to the boot snapshot only if the executor is not wired.
        """
        if self._executor:
            return self._executor.get_open_positions()
        return self._boot_positions

    async def start(self):
        self._running = True
        self.event_bus.subscribe(EventType.STRATEGY_SIGNAL, self._on_signal)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._on_position_update)
        self.event_bus.subscribe(EventType.MARKET_KLINE, self._on_kline)

    # ── volatility-targeted sizing (Phase P3 gap fix) ────────────────────

    async def _on_kline(self, event: Event):
        """Keep a short ``(close_time, close)`` history per ``(symbol, interval)``.

        This is the live source the sizing path can always reach without a new
        component reference: the market-data provider already publishes every
        closed candle, and a deque append is free.  While the switch is off the
        handler returns immediately, so nothing is buffered and nothing changes.

        **A candle without a usable ``close_time`` is refused** (audit F1): the
        buffer is now timestamp-indexed so the splice guard can see a hidden
        calendar hole, and a close that carries no time cannot be guarded.  Both
        production publishers (``provider._handle_ws_message``,
        ``app/main.py`` REST poll) set ``close_time`` from Binance's ``k.T``, so
        the refusal only affects synthetic callers.
        """
        if not self.sizer.vol_targeting_enabled():
            return
        data = event.data or {}
        symbol = data.get("symbol")
        candle = data.get("candle") or {}
        close = candle.get("close")
        if not symbol or close is None:
            return
        try:
            value = float(close)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value) or value <= 0.0:
            return
        close_time = _frame_close_time(candle.get("close_time"))
        if close_time is None:
            if self._should_warn_missing_times():
                from loguru import logger
                logger.warning(
                    "RiskManager: refusing a MARKET_KLINE candle without a usable "
                    "close_time — the volatility splice guard cannot verify a "
                    "buffer with no timestamps; the bar is not buffered")
            return
        key = (str(symbol), str(data.get("interval") or _DEFAULT_VOL_INTERVAL))
        history = self._kline_history.get(key)
        if history is None:
            history = self._kline_history[key] = deque(maxlen=_VOL_HISTORY_BARS)
        history.append((close_time, value))

    def _should_warn_splice(self) -> bool:
        """Cheap bound on a warning that would otherwise fire on every signal."""
        now = time.monotonic()
        if now - self._last_splice_warning < _VOL_CACHE_TTL_SEC:
            return False
        self._last_splice_warning = now
        return True

    def _should_warn_missing_times(self) -> bool:
        """Same TTL bound for the "no close_time" refusal."""
        now = time.monotonic()
        if now - self._last_timestamp_warning < _VOL_CACHE_TTL_SEC:
            return False
        self._last_timestamp_warning = now
        return True

    def _recent_bars(self, symbol: str, interval: str):
        """Buffered ``(close_time, close)`` pairs, or ``None`` when unusable.

        All-or-nothing on purpose: a single un-timestamped bar means the gap
        check cannot be trusted for the series, and the audit's rule is to refuse
        the buffer path rather than silently skip the guard.
        """
        history = self._kline_history.get((str(symbol), str(interval)))
        if not history or len(history) < 3:
            return None
        bars = list(history)
        if any(ts is None for ts, _ in bars):
            if self._should_warn_missing_times():
                from loguru import logger
                logger.warning(
                    f"RiskManager: refusing the buffered volatility series for "
                    f"{symbol} {interval} — {sum(1 for ts, _ in bars if ts is None)} "
                    f"of {len(bars)} bars carry no close_time, so the splice guard "
                    f"cannot be applied (sizing falls back to the fixed fraction)")
            return None
        return bars

    def _buffered_frame(self, symbol: str, interval: str):
        """DatetimeIndexed ``close`` frame from the live kline buffer, or ``None``.

        The index is the whole point (audit F1): with the old
        ``pd.DataFrame({"close": closes})`` the index was a ``RangeIndex``, so
        ``_series_has_gap`` subtracted consecutive integers (``1 - 0`` "seconds")
        and the guard could never fire.  A ``None`` return is a refusal, not a
        licence to fall through to an unguarded frame.
        """
        bars = self._recent_bars(symbol, interval)
        if not bars:
            return None
        try:
            import pandas as pd
            index = pd.DatetimeIndex([ts for ts, _ in bars])
            return pd.DataFrame({"close": [c for _, c in bars]}, index=index)
        except Exception as e:
            from loguru import logger
            logger.warning(f"RiskManager: buffered vol frame failed for {symbol} "
                           f"{interval}: {e}")
            return None

    async def _history_frame(self, symbol: str, interval: str):
        """Recent OHLC frame for ``symbol`` from the injected market-data source."""
        getter = getattr(self._market_data, "get_historical", None) if self._market_data else None
        if getter is None:
            return None
        try:
            return await getter(symbol, interval, limit=_VOL_HISTORY_BARS)
        except Exception as e:  # a data hiccup must never break the risk loop
            from loguru import logger
            logger.warning(f"RiskManager: vol history unavailable for {symbol} "
                           f"{interval}: {e}")
            return None

    async def forecast_vol_pct(self, symbol: str, interval: str | None = None) -> float | None:
        """Forecast conditional volatility in **percent of price per bar**.

        ``None`` whenever the forecast is unavailable — vol targeting off, no
        history, a short series, a zero estimate, or a **spliced** series (a
        calendar gap beyond ``_VOL_MAX_GAP_BARS`` bar lengths, which would let one
        fake ``+27 %`` bar dominate the estimator; see
        ``scripts/check_data_integrity.py``).  Every caller then keeps the fixed
        fraction: refusing to size from corrupt data is deliberately the *safer*
        fallback, not an error.

        The gap guard runs on both sources: the injected market-data frame (a
        ``DatetimeIndex`` from the provider) and the live kline buffer, which is
        timestamp-indexed for exactly this reason (audit F1).  A buffered series
        that cannot be timestamped is **refused** here rather than estimated
        without the guard.

        Cost: one TTL-cached estimate per ``(symbol, interval)`` — at most a few
        hundred closes through the O(window) EWMA, never a full-history scan.
        """
        if not self.sizer.vol_targeting_enabled():
            return None
        tf = str(interval or _DEFAULT_VOL_INTERVAL)
        key = (str(symbol), tf)
        now = time.monotonic()
        cached = self._vol_cache.get(key)
        if cached is not None and now - cached[0] < _VOL_CACHE_TTL_SEC:
            return cached[1]

        vt = self.sizer.vol_targeting
        frame = await self._history_frame(str(symbol), tf)
        from loguru import logger
        if frame is None or len(frame) < 3:
            # Audit F1: the fallback used to be built from bare floats, giving a
            # RangeIndex that made ``_series_has_gap`` inert.  It is now built from
            # ``(close_time, close)`` pairs and **refused** when the buffer cannot
            # supply timestamps — refusing is the safer fallback (fixed fraction),
            # silently skipping the guard is not.
            frame = self._buffered_frame(str(symbol), tf)
            if frame is None:
                return None
        if _series_has_gap(frame.index, tf):
            if self._should_warn_splice():
                logger.warning(
                    f"RiskManager: refusing a volatility forecast for {symbol} {tf} — "
                    f"the series carries a calendar gap beyond {_VOL_MAX_GAP_BARS} "
                    f"bars; sizing falls back to the fixed fraction "
                    f"(run scripts/check_data_integrity.py)")
            return None
        try:
            from core.ml.volatility import forecast_vol, to_pct
            vol = to_pct(forecast_vol(
                frame, method=getattr(vt, "method", "ewma"),
                window=int(getattr(vt, "window", 500)),
                lam=float(getattr(vt, "lam", 0.94)),
                interval=tf))
        except Exception as e:
            logger.warning(f"RiskManager: vol forecast failed for {symbol}: {e}")
            return None
        if not vol or vol <= 0.0 or not math.isfinite(float(vol)):
            return None
        self._vol_cache[key] = (now, float(vol))
        return float(vol)

    async def resolve_forecast_vol_pct(self, signal: dict) -> float | None:
        """Forecast vol (%) for one signal, or ``None`` for the fixed fraction.

        Resolution order — the cheapest source that answers wins, and every
        source is inert while ``risk.vol_targeting.enabled`` is false:

        1. the signal itself (``forecast_vol_pct`` / ``vol_pct``), for an upstream
           caller that already holds the price frame;
        2. the executor's published forecast (``vol_stop_ctx``) — the P3 push
           channel, so the position is **sized** and **stopped** with the same
           number instead of two independently computed ones;
        3. :meth:`forecast_vol_pct` (injected market data → live kline history),
           TTL-cached and gap-guarded.
        """
        if not self.sizer.vol_targeting_enabled():
            return None
        for field in ("forecast_vol_pct", "vol_pct"):
            raw = (signal or {}).get(field)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                return value
        symbol = str((signal or {}).get("symbol") or "")
        interval = (signal or {}).get("timeframe") or (signal or {}).get("interval")
        if self._executor is not None:
            ctx = {}
            try:
                ctx = self._executor.vol_stop_ctx(symbol) or {}
            except Exception:
                ctx = {}
            value = ctx.get("vol_pct")
            if value:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = 0.0
                if math.isfinite(value) and value > 0.0:
                    return value
        return await self.forecast_vol_pct(symbol, interval)

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
            self._pending_signals[symbol] = time.time()
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
                    if now - self._last_breaker_alert_time >= self._ALERT_THROTTLE_SEC:
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
                    "open_positions": self._get_positions(),
                }))
                task.add_done_callback(
                    lambda t: logger.error(f"trip_callback failed: {t.exception()}") if t.exception() else None
                )
            return RiskResult(approved=False, reason=f"Circuit breaker tripped: {reason}")

        # Step 2: Total exposure check (use current price when available)
        total_exposure = 0.0
        positions = self._get_positions()
        for sym, p in positions.items():
            qty = p.get("quantity", 0)
            # Prefer current market price; fall back to entry price
            price = p.get("current_price", p.get("entry_price", 0))
            total_exposure += qty * price
        exposure_pct = (total_exposure / self._account_balance * 100) if self._account_balance > 0 else 0
        if exposure_pct >= self.config.hard_limits.max_total_exposure_pct:
            return RiskResult(approved=False, reason=f"Total exposure {exposure_pct:.1f}% exceeds limit")

        # Step 3: Position size check
        symbol = signal.get("symbol", "")
        price = signal.get("price", 0)
        # Phase P3 gap fix: the sizing path now consumes the volatility forecast.
        # ``None`` (switch off, no history, spliced series) leaves the arithmetic
        # exactly as it was before P3 — see PositionSizer.calculate_position_size.
        forecast_vol_pct = await self.resolve_forecast_vol_pct(signal)
        qty, risk_amount = self.sizer.calculate_position_size(
            self._account_balance, price, signal.get("position_type", "satellite"),
            forecast_vol_pct=forecast_vol_pct
        )
        if forecast_vol_pct is not None:
            scale = self.sizer.vol_scale(forecast_vol_pct)
            target = getattr(self.sizer.vol_targeting, "target_vol_pct", None)
            logger.debug(
                f"vol targeting: {symbol} tf={signal.get('timeframe') or _DEFAULT_VOL_INTERVAL} "
                f"forecast={forecast_vol_pct:.4f}%/bar target={target}%/bar "
                f"scale={scale:.4f}x notional={risk_amount:.4f}")
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

        # Step 6: Same symbol check (includes pending signals with timeout to prevent race)
        # Auto-clear stale pending entries that were never resolved
        now_ts = time.time()
        stale = [s for s, t in self._pending_signals.items() if now_ts - t > self._pending_timeout_sec]
        for s in stale:
            self._pending_signals.pop(s, None)
            logger.warning(f"Pending signal for {s} timed out after {self._pending_timeout_sec}s — auto-cleared")

        if symbol in self._get_positions() or symbol in self._pending_signals:
            return RiskResult(approved=False, reason=f"Position already open for {symbol}")

        # Step 7: Max open trades check
        if len(self._get_positions()) >= self.config.hard_limits.max_open_trades:
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
        self._pending_signals.pop(symbol, None)
        # Track PnL for circuit breaker stats (positions are read directly from executor)
        if pnl != 0:
            self.breaker.add_trade_result(pnl)

    def update_balance(self, balance: float):
        self._account_balance = balance
        # Read positions DIRECTLY from executor (single source of truth).
        positions = self._get_positions()
        total_invested = sum(p.get("amount_usdt", 0) for p in positions.values())
        equity = balance + total_invested
        self.breaker.set_equity(equity)
        # When all positions close and peak was artificially inflated,
        # clamp it down so stale drawdowns don't trigger false alarms.
        if total_invested == 0 and self.breaker.peak_equity > equity * 1.05:
            self.breaker.clamp_peak_to_current()

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
        """Return current open positions from the executor (single source of truth)."""
        return self._get_positions()

    async def stop(self):
        self._running = False
        self.event_bus.unsubscribe(EventType.STRATEGY_SIGNAL, self._on_signal)
        self.event_bus.unsubscribe(EventType.POSITION_UPDATE, self._on_position_update)
        self.event_bus.unsubscribe(EventType.MARKET_KLINE, self._on_kline)
