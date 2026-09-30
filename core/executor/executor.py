import asyncio
import contextlib
import threading
import time
import uuid
from decimal import ROUND_DOWN, Decimal
from typing import Optional
from binance import AsyncClient
from binance.enums import *
from loguru import logger

from app.event_bus import EventBus, Event, EventType
from app.config import Config, load_sim_cost_settings, sim_cost_quote
from db.database import (
    load_sim_balance,
    save_sim_balance,
    atomic_adjust_balance,
    _json_levels,
    parse_levels,
)


class OrderExecutor:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.client: Optional[AsyncClient] = None
        self._running = False
        self._orders: dict[str, dict] = {}
        self._positions: dict[str, dict] = {}
        self._risk_manager = None
        #: One lock per symbol, serialising every cash-moving mutation (open,
        #: close, reduce).  `_positions` is keyed by symbol, so two concurrent
        #: handlers for the same symbol used to read the *same* pre-mutation
        #: snapshot and each book a full close against one basis: cash was created
        #: out of nothing and two rows landed in one trade_group.  Different
        #: symbols still run concurrently (a lock is only ever taken for the
        #: symbol being mutated).  See `_symbol_lock`.
        self._symbol_locks: dict[str, asyncio.Lock] = {}
        #: Monotonic revision of each symbol's in-memory position.  A caller that
        #: read the position *before* waiting on the lock compares it afterwards;
        #: a changed revision means the basis it read is stale, so applying its
        #: reduce again would double-count.  It is refused instead (S3/S12).
        self._position_rev: dict[str, int] = {}
        #: Optional coin universe (``core.market_data.universe.Universe``) used to
        #: resolve a symbol's LOT_SIZE / NOTIONAL rules. Injected, never imported
        #: as a global: without it the executor keeps its 1e-5 fallback.
        self._universe = None
        #: Trade gate — see :meth:`reset_barrier`.  Deliberately built from
        #: ``threading`` primitives, not ``asyncio`` ones: an ``asyncio.Lock`` is
        #: bound to the loop that first waited on it, so a reset served by the web
        #: loop could not serialise an open running on another loop (the test
        #: suite's per-test ``asyncio.run`` loops, and this app's own startup
        #: thread).  A per-loop gate looked serialised in one loop and was no
        #: protection at all across two — the reset erased the row while the open
        #: still owed the cash (Δ −100.160030).
        self._gate_lock = threading.Lock()
        self._gate_idle = threading.Event()
        self._gate_idle.set()
        self._gate_closed = False
        self._gate_active = 0
        #: Diagnostic only: which loops currently have a mutation registered.
        self._active_loops: dict[int, int] = {}
        #: symbol → (monotonic_ts, forecast vol %) published by the predictor
        #: (Phase P3).  Empty by default: with ``risk.vol_targeting.enabled``
        #: false nothing ever writes it and the stop path is the fixed-percentage
        #: one — see :meth:`vol_stop_ctx`.
        self._forecast_vol_cache: dict[str, tuple[float, float]] = {}

    # ---- trade gate ------------------------------------------------------
    def _enter_mutation(self) -> None:
        """Register one in-flight cash movement (its ``finally`` must release)."""
        while True:
            if not self._gate_closed:
                # Atomic enough under the GIL **and** free of ``await``: no other
                # task in this loop can interleave between the check and the
                # increment, and a reset is what flips ``_gate_closed``.
                self._gate_active += 1
                self._gate_idle.clear()
                return
            self._gate_idle.wait()

    def _leave_mutation(self) -> None:
        self._gate_active -= 1
        if self._gate_active <= 0:
            self._gate_active = 0
            self._gate_idle.set()

    @contextlib.asynccontextmanager
    async def _mutation_slot(self):
        """Register one in-flight cash movement, waiting out any pending reset."""
        self._enter_mutation()
        loop_id = id(asyncio.get_running_loop())
        self._active_loops[loop_id] = self._active_loops.get(loop_id, 0) + 1
        try:
            yield
        finally:
            remaining = self._active_loops.get(loop_id, 1) - 1
            if remaining > 0:
                self._active_loops[loop_id] = remaining
            else:
                self._active_loops.pop(loop_id, None)
            self._leave_mutation()

    @contextlib.asynccontextmanager
    async def reset_barrier(self):
        """Exclusive window for an admin reset: no in-flight cash movement survives it.

        ``/api/settings/reset-sim`` used to erase ``trades``/``positions`` and then
        restore the balance to 10000 while an open was still in flight: the open
        committed its row *before* the erase and deducted its cash *after* the
        restore, leaving the ledger identity broken (measured Δ −100.160030) with
        a cash movement no row explains.

        Every cash-moving mutation registers through :meth:`_mutation_slot`, and
        this context manager closes the gate, waits for the in-flight book to
        drain, then holds it closed until the reset is done.  New opens/closes wait
        for the reset instead of running through it.  A close that already read a
        position is refused by the revision check in :meth:`close_position`, which
        the caller bumps while still inside this window.
        """
        self._gate_lock.acquire()
        self._gate_closed = True
        self._gate_lock.release()
        # Wait for the in-flight book to drain.  ``Event.wait`` blocks only this
        # thread's loop while the reset runs elsewhere; it is set again by the
        # last mutation to leave, and by the release below.
        while self._gate_active > 0:
            self._gate_idle.wait(0.05)
            await asyncio.sleep(0)      # let this loop's own tasks settle
        self._active_loops.clear()
        try:
            yield
        finally:
            self._gate_closed = False
            self._gate_idle.set()

    def check_no_mutations(self) -> None:
        """Assert the trade gate is quiet (call while holding :meth:`reset_barrier`).

        A reset is only meaningful if *nothing* is mid-mutation while it erases the
        ledger.  This makes a future caller that forgets to wrap itself in
        :meth:`_mutation_slot` fail loudly here instead of silently corrupting the
        balance.
        """
        if self._gate_active != 0:
            raise RuntimeError(
                f"a cash-moving mutation is in flight during a reset "
                f"(active={self._gate_active}); it must register through "
                f"OrderExecutor._mutation_slot (see reset_barrier)")

    def _symbol_lock(self, symbol: str) -> asyncio.Lock:
        """The per-symbol mutation lock, created on first use.

        No ``await`` between the lookup and the insert, so on the single event
        loop this module runs on two tasks can never end up with two different
        locks for one symbol.
        """
        lock = self._symbol_locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._symbol_locks[symbol] = lock
        return lock

    def _bump_revision(self, symbol: str) -> None:
        """Record that ``symbol``'s in-memory position just changed."""
        self._position_rev[symbol] = self._position_rev.get(symbol, 0) + 1

    def wire_risk_manager(self, rm):
        self._risk_manager = rm

    def wire_universe(self, universe):
        """Inject the shared coin universe (tick/step size, min notional)."""
        self._universe = universe

    def _symbol_info(self, symbol: str, data: dict | None = None):
        """Best-effort ``SymbolInfo`` for ``symbol`` (``None`` when unknown).

        An order may carry its own ``symbol_info``; otherwise the wired universe
        is consulted.  Never raises: a missing universe/rule must not stop a live
        order from using the documented fallback path.
        """
        info = (data or {}).get("symbol_info")
        if info is not None and getattr(info, "symbol", ""):
            return info
        universe = self._universe
        if universe is None:
            return None
        try:
            return universe.get(symbol)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"SymbolInfo lookup failed for {symbol}: {e}")
            return None

    @staticmethod
    def _round_down_to_step(qty: float, step: Optional[float]) -> float:
        """Floor ``qty`` to a multiple of the symbol's LOT_SIZE ``step``.

        Rounding DOWN is what keeps an order legal: ``round(qty, 5)`` could round
        *up* past the tradable size, and a step coarser than 1e-5 produced a
        quantity the exchange rejects.  An unknown/non-positive ``step`` falls
        back to the historical ``round(qty, 5)`` (1e-5 granularity), so a symbol
        whose LOT_SIZE cannot be resolved keeps its previous behaviour exactly.
        """
        if not step or step <= 0:
            return round(qty, 5)
        try:
            d_step = Decimal(str(step))
            if d_step <= 0:
                return round(qty, 5)
            lots = (Decimal(str(qty)) / d_step).to_integral_value(rounding=ROUND_DOWN)
            return float(lots * d_step)
        except Exception:  # pragma: no cover - defensive (non-numeric input)
            return round(qty, 5)

    async def _reject_live_order(self, symbol: str, kind: str, message: str) -> None:
        """Publish a structured rejection instead of sending an illegal order."""
        logger.warning(f"Live order for {symbol} rejected before submission: {message}")
        await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
            "level": "warning", "type": kind, "symbol": symbol, "message": message,
        }))

    async def start(self):
        if self.config.mode == "live" and self.config.binance_api_key:
            self.client = await AsyncClient.create(
                api_key=self.config.binance_api_key,
                api_secret=self.config.binance_api_secret,
                testnet=self.config.binance_testnet,
            )
        self._running = True
        self.event_bus.subscribe(EventType.ORDER_REQUEST, self._on_order_request)
        await self.restore_positions()

    async def restore_positions(self):
        """Restore open positions from DB after restart.

        The authoritative source is the ``positions`` snapshot written when the
        position was opened (and updated by every reduce / trailing-stop move);
        ``trades`` is the fallback for rows that predate that table.

        Two defects are fixed here:

        * ``entry_price`` used to be overwritten with the *fill* price.  The open
          row's ``entry_price`` is the **cash basis** per unit (``qty × entry_price``
          == the cash the open deducted, which is what the ledger identity is
          stated against), so overwriting it with the fill — which excludes the
          buy fee and slippage — made every post-restart close return less cash
          than the open took out.
        * ``r.get("stop_loss")`` was read from ``trades``, which had no such
          column, so it was always ``None`` and every restart silently replaced
          the real stop with a hard-coded 2% default.  Take-profits were not
          stored at all and were lost outright.
        """
        import aiosqlite as aio
        from loguru import logger
        try:
            db = await aio.connect(self.config.db_path)
            db.row_factory = aio.Row
            snapshots = []
            try:
                cursor = await db.execute("SELECT * FROM positions")
                snapshots = [dict(r) for r in await cursor.fetchall()]
            except Exception as e:
                logger.warning(f"No positions snapshot table yet: {e}")
            cursor = await db.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY id ASC")
            rows = [dict(r) for r in await cursor.fetchall()]
            await db.close()

            restored = 0
            # The `trades` open row is the ledger truth: the balance identity
            # subtracts `quantity × entry_price` of exactly those rows, and a close
            # returns that same figure.  A `positions` snapshot is only the *state*
            # of that position (stops, take-profits, trailing basis) and is
            # restored only when it agrees with its row — a snapshot without a row
            # used to be restored blindly, and closing it invented a basis no open
            # row ever deducted (Δ −230 in the audit).  Skips are loud: an
            # operator must see that capital is unaccounted for.
            open_by_group = {r["trade_group"]: r for r in rows if r.get("trade_group")}
            open_by_symbol = {r["symbol"]: r for r in rows}
            orphans: list[str] = []
            mismatches: list[str] = []

            def _ledger_row(snap: dict):
                group = snap.get("trade_group")
                if group and group in open_by_group:
                    return open_by_group[group]
                return open_by_symbol.get(snap["symbol"])

            def _agrees(snap: dict, row: dict) -> bool:
                """Does the snapshot describe the same position as the ledger row?"""
                if (snap.get("side") or "long") != row["side"]:
                    return False
                snap_qty, row_qty = float(snap["quantity"] or 0), float(row["quantity"] or 0)
                if abs(snap_qty - row_qty) > max(1e-9, row_qty * 1e-6):
                    return False
                snap_basis = snap_qty * float(snap["entry_price"] or 0)
                row_basis = row_qty * float(row["entry_price"] or 0)
                return abs(snap_basis - row_basis) <= max(1e-6, abs(row_basis) * 1e-6)

            for snap in snapshots:
                symbol = snap["symbol"]
                row = _ledger_row(snap)
                if row is None:
                    orphans.append(
                        f"{symbol} (qty={snap['quantity']}, basis={snap['entry_price']})")
                    continue
                if not _agrees(snap, row):
                    mismatches.append(
                        f"{symbol} snapshot qty={snap['quantity']}"
                        f"×{snap['entry_price']} vs row id={row['id']}"
                        f" qty={row['quantity']}×{row['entry_price']}")
                    continue
                # The ROW's basis, verbatim: `close_position` returns
                # `quantity × entry_price` from here, and it must be exactly the
                # figure the identity subtracts, not a rounded copy of it.
                qty = float(row["quantity"] or 0)
                entry = float(row["entry_price"] or 0)
                fill_price = float(snap.get("fill_price") or row.get("fill_price") or entry)
                sl = snap.get("stop_loss")
                self._positions[symbol] = {
                    "symbol": symbol,
                    "side": row["side"],
                    "quantity": qty,
                    "entry_price": entry,
                    "current_price": float(snap.get("current_price") or fill_price),
                    "unrealized_pnl": float(snap.get("unrealized_pnl") or 0),
                    "stop_loss": sl,
                    "entry_stop_loss": snap.get("entry_stop_loss", sl),
                    # Phase P3: restored so a position that survived a restart is
                    # still trailed with the width its entry stop was sized from.
                    # Absent (all pre-P3 snapshots) → None → the guard falls back
                    # to the fixed percentage or a fresh forecast.
                    "stop_vol_pct": snap.get("stop_vol_pct"),
                    "take_profits": parse_levels(snap.get("take_profits")),
                    "position_type": snap.get("position_type") or "satellite",
                    # Restored from the recorded basis, not re-derived from a
                    # marked price, so the close returns exactly the cash taken.
                    "position_value": qty * entry,
                    "amount_usdt": qty * entry,
                    "entry_fee": float(snap.get("fee") or 0),
                    "fill_price": fill_price,
                    "fee": float(snap.get("fee") or 0),
                    "slippage": float(snap.get("slippage") or 0),
                    "trade_group": row.get("trade_group") or snap.get("trade_group") or "",
                    "strategy_name": snap.get("strategy_name") or "",
                    "strategy": snap.get("strategy") or "manual",
                    "trader": snap.get("trader") or "manual",
                    "timeframe": snap.get("timeframe") or "1h",
                }
                restored += 1

            if orphans or mismatches:
                logger.error(
                    "restore_positions: {} orphaned positions snapshot(s) refused "
                    "({}); {} snapshot(s) disagreed with their open trades row and "
                    "are ignored, the row's own basis is used instead ({})",
                    len(orphans), "; ".join(orphans) or "-",
                    len(mismatches), "; ".join(mismatches) or "-")
                await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
                    "level": "critical", "type": "ledger_snapshot_mismatch",
                    "message": (f"台账校验：拒绝恢复 {len(orphans)} 个没有 trades 开仓行的"
                                f"持仓快照 {orphans}；{len(mismatches)} 个与开仓行不符"
                                f"（已改用台账行的成本基础）{mismatches}"),
                }))

            for r in rows:
                symbol = r["symbol"]
                if symbol in self._positions:
                    continue
                qty = r["quantity"]
                entry = r["entry_price"]
                # Cost model (v3 migration): the open row records what was really
                # paid (`fill_price`) plus the buy-side `fee`/`slippage`, which the
                # close must charge back.  Older rows have neither, so fall back to
                # the quoted entry price and zero costs.
                fill_price = r["fill_price"] if "fill_price" in r.keys() else None
                fill_price = float(fill_price) if fill_price else entry
                buy_fee = float(r["fee"]) if "fee" in r.keys() and r["fee"] else 0.0
                buy_slippage = (float(r["slippage"])
                                if "slippage" in r.keys() and r["slippage"] else 0.0)
                # Stop-loss and take-profits as persisted at open.  Only a row
                # written before v4 (or with no levels at all) needs the default.
                sl = r["stop_loss"] if "stop_loss" in r.keys() else None
                if sl is None:
                    if entry > 0 and r["side"] == "long":
                        sl = round(entry * 0.98, 2)  # default 2% below entry
                    elif entry > 0:
                        sl = round(entry * 1.02, 2)  # default 2% above entry
                take_profits = parse_levels(
                    r["take_profits"] if "take_profits" in r.keys() else None)
                self._positions[symbol] = {
                    "symbol": symbol,
                    "side": r["side"],
                    "quantity": qty,
                    # The RECORDED cash basis per unit.  `entry` is what the open
                    # deducted per unit, so `qty × entry_price` still equals the
                    # cash the account paid out.
                    "entry_price": entry,
                    "current_price": fill_price,
                    "unrealized_pnl": 0,
                    "stop_loss": sl,
                    "entry_stop_loss": sl,
                    "take_profits": take_profits,
                    "position_type": r.get("position_type", "satellite"),
                    "position_value": qty * entry,
                    "amount_usdt": qty * entry,
                    "entry_fee": buy_fee,
                    "fill_price": fill_price,
                    "fee": buy_fee,
                    "slippage": buy_slippage,
                    "trade_group": r.get("trade_group", ""),
                    "strategy_name": r.get("strategy_name", ""),
                    "strategy": r.get("strategy", "manual"),
                    "trader": r.get("trader", "manual"),
                    # Restored so a close after a restart still attributes the row
                    # to the timeframe the signal was generated on.
                    "timeframe": (r["timeframe"] if "timeframe" in r.keys() else None) or "1h",
                }
                restored += 1
            logger.info(f"Restored {restored} open positions from DB "
                        f"({len(snapshots)} from the positions snapshot)")
        except Exception as e:
            logger.warning(f"Failed to restore positions: {e}")

    #: How long a stop-width forecast is reused (seconds).  Opening a position
    #: must not add a REST round-trip per order: the width is stable over minutes
    #: and only used when ``risk.vol_targeting.enabled`` is on.
    _VOL_STOP_TTL_SEC = 300.0

    def vol_stop_ctx(self, symbol: str, vol_pct: float | None = None) -> dict:
        """Volatility-scaled stop width for a new position (Phase P3).

        Returns ``{}`` — the documented no-op — when ``risk.vol_targeting`` is
        absent/disabled, so with the shipped default ``enabled: false`` this
        method cannot alter a single fill.  Otherwise it returns::

            {"vol_pct": <forecast %/bar>, "stop_pct": <stop distance %>,
             "stop_loss": <price or None>}

        ``stop_pct`` is produced by the **shared** ``PositionSizer`` helper, so the
        live stop a position opens with is computed by the same code that the
        trailing updater and the backtest engines use — one definition of
        "forecast-scaled stop", not three.

        ``vol_pct`` may be supplied by the caller (an upstream predictor already
        has the frame); when omitted the cached forecast for the symbol is used,
        and when that is unavailable the result is ``{}`` — i.e. the caller's
        fixed percentage stands.  That is the documented fallback, not an error.
        """
        vt = getattr(self.config, "risk_vol_targeting", None)
        if vt is None or not getattr(vt, "enabled", False):
            return {}
        vol = vol_pct
        if vol is None:
            vol = self._cached_forecast_vol_pct(symbol)
        try:
            vol = float(vol)
        except (TypeError, ValueError):
            return {}
        if not (vol > 0.0):
            return {}
        from core.risk.position_sizer import PositionSizer
        sizer = PositionSizer(
            self.config.hard_limits, self.config.soft_params,
            getattr(self.config, "core_capital_pct", 0.7),
            getattr(self.config, "satellite_capital_pct", 0.3), vt)
        return {"vol_pct": vol, "stop_pct": sizer.stop_distance_pct(vol)}

    def set_forecast_vol_pct(self, symbol: str, vol_pct: float | None) -> None:
        """Publish the latest forecast vol (%) for ``symbol`` to the stop path.

        The live predictor owns the price frame, so it is the cheapest place to
        compute the forecast; pushing it here keeps :meth:`vol_stop_ctx`
        synchronous at order time.  ``None`` clears the entry (→ fixed fallback).
        """
        if vol_pct is None:
            self._forecast_vol_cache.pop(str(symbol), None)
            return
        try:
            val = float(vol_pct)
        except (TypeError, ValueError):
            return
        if val > 0.0:
            self._forecast_vol_cache[str(symbol)] = (time.monotonic(), val)

    def _cached_forecast_vol_pct(self, symbol: str) -> float | None:
        entry = self._forecast_vol_cache.get(str(symbol))
        if entry is None:
            return None
        ts, val = entry
        if time.monotonic() - ts > self._VOL_STOP_TTL_SEC:
            self._forecast_vol_cache.pop(str(symbol), None)
            return None
        return val

    async def _persist_position(self, db, pos: dict) -> None:
        """Upsert ``pos`` into the ``positions`` snapshot table (same connection).

        This is the row a restart reads, so the basis, stop-loss and
        take-profits survive exactly as the running process knows them.
        """
        await db.execute(
            "INSERT OR REPLACE INTO positions ("
            " symbol, side, quantity, entry_price, current_price, unrealized_pnl,"
            " stop_loss, take_profits, entry_stop_loss, position_type, trader,"
            " strategy_name, strategy, trade_group, timeframe, amount_usdt,"
            " position_value, fill_price, fee, slippage, opened_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
            " COALESCE((SELECT opened_at FROM positions WHERE symbol=?), CURRENT_TIMESTAMP),"
            " CURRENT_TIMESTAMP)",
            (pos.get("symbol"), pos.get("side"), pos.get("quantity"),
             pos.get("entry_price"), pos.get("current_price"),
             pos.get("unrealized_pnl", 0), pos.get("stop_loss"),
             _json_levels(pos.get("take_profits")),
             pos.get("entry_stop_loss"), pos.get("position_type", "satellite"),
             pos.get("trader", "manual"), pos.get("strategy_name", ""),
             pos.get("strategy", "manual"), pos.get("trade_group", ""),
             pos.get("timeframe") or "1h",
             pos.get("amount_usdt", pos.get("position_value")),
             pos.get("position_value"), pos.get("fill_price"),
             pos.get("fee"), pos.get("slippage"), pos.get("symbol")))

    async def update_stop_loss(self, symbol: str, stop_loss: float) -> None:
        """Persist a moved stop-loss for an open position.

        Called by the trailing-stop logic: moving `stop_loss` in memory only used
        to be lost on the next restart (the executor then re-applied its 2%
        default), which silently disabled a trailing stop that had already moved
        into profit.  Best-effort: a DB failure must never break the guard loop.
        """
        import aiosqlite as aio
        from loguru import logger
        if not symbol or stop_loss is None:
            return
        try:
            db = await aio.connect(self.config.db_path)
            try:
                await db.execute(
                    "UPDATE positions SET stop_loss=?, updated_at=CURRENT_TIMESTAMP"
                    " WHERE symbol=?", (float(stop_loss), symbol))
                await db.execute(
                    "UPDATE trades SET stop_loss=?"
                    " WHERE symbol=? AND action='open' AND status='open'",
                    (float(stop_loss), symbol))
                await db.commit()
            finally:
                await db.close()
        except Exception as e:
            logger.warning(f"Could not persist stop_loss for {symbol}: {e}")

    async def _on_order_request(self, event: Event):
        if self.config.mode == "sim":
            await self._execute_sim(event.data)
        elif self.config.mode == "live":
            await self._execute_live(event.data)

    async def _cost_settings(self) -> dict:
        """Effective sim cost model — YAML defaults overridden by the saved tier."""
        return await load_sim_cost_settings(self.config.db_path, self.config)

    async def _execute_sim(self, data: dict):
        """Serialise the open for its symbol, then run the real body.

        One critical section covers the duplicate-open guard, the
        ``await self._cost_settings()`` on the way to the fill, the ``trades``
        insert and the ``positions`` snapshot.  The guard used to sit *before*
        that await, so two concurrent ORDER_REQUESTs for one symbol both passed
        it and both inserted an open row while only the second wrote the
        snapshot: the ledger kept two open rows, one snapshot and one in-memory
        position, and a restart resurrected the symbol as a zombie whose capital
        was stranded for ever (S6).
        """
        async with self._symbol_lock(str(data.get("symbol", ""))):
            # Registered with the trade gate for the whole critical section: an
            # admin reset waits for this open (and this open waits for a reset in
            # progress) so the erase and the cash movement can never interleave.
            async with self._mutation_slot():
                return await self._execute_sim_locked(data)

    async def _execute_sim_locked(self, data: dict):
        order_id = f"sim_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        price = data.get("price", 0)
        qty = data.get("quantity", 0)
        symbol = data.get("symbol", "")
        side = data.get("side", "long")
        trade_group = str(uuid.uuid4())[:8]
        # NOTE: the caller's `amount_usdt` is deliberately ignored from here on.
        # The quantity is the tradable fact; the *cash* the ledger must move is
        # derived below from the real fill (qty × fill_price + buy fee + slippage),
        # so it can never disagree with the open row it is recorded next to.

        # An open must never silently replace a live position for the same symbol.
        # `_positions` is keyed by symbol, so a second open used to overwrite the
        # first while the ledger kept BOTH open rows: the first row's capital stayed
        # deducted for ever (it was never returned by any close) and only the second
        # open's notional came back on exit.  That is a real cash leak —
        # 10000 − open notional + realised ends up above the balance by exactly the
        # orphaned notional.  Refuse the duplicate and book nothing.
        if symbol in self._positions:
            logger.error(
                f"Refusing duplicate open for {symbol}: an open position already exists "
                f"(qty={self._positions[symbol].get('quantity', 0):.8f}). The in-memory "
                f"position is keyed by symbol, so overwriting it would orphan this open "
                f"row's capital and break the balance identity."
            )
            await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
                "level": "warning", "type": "duplicate_open_rejected",
                "message": f"{symbol} 已有持仓，重复开仓已拒绝（避免资金台账漂移）",
                "symbol": symbol,
            }))
            return

        # ---- cost model: the fill is worse than the quote -------------------
        # Buy at quoted × (1 + (spread/2 + slippage)/100), sell at the mirror
        # image (see app/config.sim_cost_quote).  Limit orders fill at their own
        # price and pay the fee only.
        order_type = str(data.get("order_type") or "market").lower()
        quote = sim_cost_quote(symbol, side, order_type, price, qty,
                               await self._cost_settings())
        fill_price = quote["fill_price"] or price
        fee = round(quote["fee_usdt"], 8)
        slippage = round(quote["slippage_usdt"], 8)
        # Cash that actually leaves the account on this open: the worsened fill's
        # notional PLUS the buy-side fee and slippage.  Deducting only the quoted
        # notional used to leave the buy costs uncharged while the close still
        # subtracted them from `pnl`, so the two sides never cancelled.
        amount_usdt = qty * fill_price + fee + slippage
        # Ledger basis per unit, stored in `trades.entry_price`.  The invariant is
        # `10000 − Σ(open qty×entry_price) + Σ(close pnl) == sim_balance`, so the DB
        # `entry_price` is *defined* as the cash basis actually deducted:
        # qty × entry_price == amount_usdt.  That is what makes `invested_returned`
        # below exactly the capital the open took out.
        cost_basis = amount_usdt / qty if qty else fill_price

        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": "market" if order_type != "limit" else "limit",
            "price": price,
            "fill_price": fill_price,
            "fee": fee,
            "slippage": slippage,
            "quantity": qty,
            "filled_qty": qty,
            "status": "filled",
            "binance_order_id": None,
            "stop_loss": data.get("stop_loss"),
            "take_profits": data.get("take_profits", []),
        }
        self._orders[order_id] = order
        # No "overwriting an existing position" branch any more: the guard above
        # runs inside the per-symbol lock, so reaching here means `_positions`
        # holds no position for this symbol.
        #
        # Phase P3: a volatility-scaled stop width overrides the caller's fixed
        # percentage *only* when `risk.vol_targeting.enabled` is on (else
        # `vol_stop_ctx` returns `{}` and nothing below changes).  `stop_vol_pct`
        # is remembered on the position so `PositionGuard` trails with the same
        # width this entry stop was built from instead of re-forecasting per tick.
        stop_sl = data.get("stop_loss")
        vol_ctx = self.vol_stop_ctx(symbol)
        if vol_ctx and float(stop_sl or 0) > 0:
            stop_pct = float(vol_ctx["stop_pct"]) / 100.0
            stop_sl = (cost_basis * (1 - stop_pct) if side == "long"
                       else cost_basis * (1 + stop_pct))
        self._positions[symbol] = {
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            # The ledger basis IS the entry: stops, take-profits and close PnL are
            # measured from what was really paid, and `close_position` returns the
            # full cash basis, so the balance identity
            # 10000 − open notional + realised = balance survives to the cent.
            "entry_price": cost_basis,
            "current_price": fill_price,
            "unrealized_pnl": 0,
            "stop_loss": stop_sl,
            "entry_stop_loss": stop_sl,
            "stop_vol_pct": vol_ctx.get("vol_pct"),
            "take_profits": list(data.get("take_profits") or []),
            "position_type": data.get("position_type", "satellite"),
            "position_value": qty * cost_basis,
            "amount_usdt": amount_usdt,
            "entry_fee": fee,
            "fill_price": fill_price,
            "fee": fee,
            "slippage": slippage,
            "trade_group": trade_group,
            "strategy_name": data.get("strategy_name", ""),
            "strategy": data.get("strategy", "manual"),
            "trader": data.get("trader", "manual"),
            # Kept so the close/reduce rows can carry the same timeframe as the
            # open row instead of a hard-coded "1h" ("1h" only when unknown).
            "timeframe": data.get("timeframe") or "1h",
        }
        # A close/reduce that read this symbol before this open must not be applied
        # on top of the new basis (see `close_position`).
        self._bump_revision(symbol)

        await self.event_bus.publish(Event(EventType.ORDER_UPDATE, {
            "order_id": order_id, "symbol": symbol, "status": "filled", "mode": "sim",
        }))
        await self.event_bus.publish(Event(EventType.POSITION_UPDATE, {
            "symbol": symbol, "side": side, "quantity": qty,
            "entry_price": cost_basis, "current_price": fill_price,
            "position_type": data.get("position_type", "satellite"),
            "position_value": qty * cost_basis,
            "amount_usdt": amount_usdt,
            "stop_loss": data.get("stop_loss"),
            "trade_group": trade_group,
            "closed": False, "pnl": 0,
        }))

        # Persist trade to DB.  `entry_price` is the cash basis per unit (see
        # `cost_basis` above) so that `quantity × entry_price` equals the amount
        # deducted below — the identity the ledger is reconciled against.
        #
        # `stop_loss` / `take_profits` are stored here (and mirrored into the
        # `positions` snapshot below) because they used to live only in memory:
        # every restart threw the real stop away and re-derived a 2% default, and
        # take-profit levels were lost outright.
        import aiosqlite as aio
        db = await aio.connect(self.config.db_path)
        try:
            await db.execute(
                "INSERT INTO trades (symbol, side, entry_price, quantity, strategy, timeframe, position_type, status, trader, strategy_name, action, trade_group, fill_price, fee, slippage, stop_loss, take_profits) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (symbol, side, cost_basis, qty, data.get("strategy", "manual"), data.get("timeframe", "1h"),
                 data.get("position_type", "satellite"), "open",
                 data.get("trader", "manual"), data.get("strategy_name", ""),
                 "open", trade_group, fill_price, fee, slippage,
                 data.get("stop_loss"), _json_levels(data.get("take_profits"))))
            # Same connection, so the open row and its snapshot commit together:
            # a restart can never see a position without its stop-loss.
            await self._persist_position(db, self._positions[symbol])
            await db.commit()
        finally:
            await db.close()

        # Atomically deduct invested amount from sim balance (DB + risk_manager).
        # This is the ONLY place cash leaves the account on an open, and the figure
        # deducted is exactly the open row's notional (qty × entry_price), so
        # `close_position` can return it verbatim without gaining or losing a cent.
        try:
            new_balance = await atomic_adjust_balance(-amount_usdt, self.config.db_path)
            if self._risk_manager:
                self._risk_manager.update_balance(new_balance)
            logger.info(f"Balance deducted: {new_balance + amount_usdt:.0f} -> {new_balance:.0f} (-{amount_usdt:.0f} USDT for {symbol})")
        except Exception as e:
            logger.warning(f"Failed to update balance on position open: {e}")
            # The row exists but no cash moved: the identity is now broken by
            # exactly this notional.  Undo the row so the ledger stays truthful
            # rather than reporting a position the account never paid for.
            try:
                import aiosqlite as aio2
                db = await aio2.connect(self.config.db_path)
                await db.execute(
                    "UPDATE trades SET status='cancelled' WHERE trade_group=? AND action='open'",
                    (trade_group,))
                # The snapshot must go with the ledger row: a position the
                # account never paid for must not be restored on the next start.
                await db.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
                await db.commit()
                await db.close()
                self._positions.pop(symbol, None)
                self._bump_revision(symbol)
                logger.error(f"Rolled back open row for {symbol}: balance deduction failed")
            except Exception as e2:
                logger.critical(
                    f"Could not roll back {symbol} open row after a failed balance "
                    f"deduction ({e2}); ledger identity is broken by "
                    f"{amount_usdt:.6f} USDT and needs manual reconciliation")

    @staticmethod
    def _is_duplicate_order_error(exc: BaseException) -> bool:
        """True when the exchange rejected the request because it already has it."""
        msg = str(exc).lower()
        return "-2010" in msg or "duplicate" in msg

    async def _lookup_order(self, symbol: str, client_order_id: str):
        """Fetch an order we (probably) already placed, by our own client id."""
        try:
            return await self.client.get_order(symbol=symbol,
                                               origClientOrderId=client_order_id)
        except Exception as e:
            logger.warning(f"Could not look up order {client_order_id} for {symbol}: {e}")
            return None

    async def _execute_live(self, data: dict):
        symbol = data.get("symbol", "")
        side = SIDE_BUY if data.get("side") == "long" else SIDE_SELL
        # Round ONCE and use the same value for the exchange request and for local
        # bookkeeping. Previously the exchange received round(qty, 5) while the
        # in-memory position kept the unrounded quantity, so exits could send a
        # size that violates the symbol's LOT_SIZE filter.
        #
        # The rounding now follows the symbol's real LOT_SIZE step (floored, so
        # the order can never round UP past the tradable size).  The rules come
        # from an injected SymbolInfo/universe, never from a hard-coded step.
        info = self._symbol_info(symbol, data)
        step = getattr(info, "step_size", None) if info is not None else None
        signal_price = float(data.get("price", 0) or 0)
        qty = self._round_down_to_step(float(data.get("quantity", 0) or 0), step)

        # LOT_SIZE / NOTIONAL gates — only enforced when the rules are known, so
        # an unresolvable symbol behaves exactly as before.  An order the
        # exchange must reject is never submitted; the reason is alerted instead
        # of surfacing as a generic API error and a lost signal.
        if qty <= 0:
            await self._reject_live_order(
                symbol, "order_below_lot_size",
                f"quantity {data.get('quantity')} rounds to 0 at step_size={step}; "
                f"increase the position size")
            return
        min_notional = getattr(info, "min_notional", None) if info is not None else None
        if min_notional and signal_price > 0 and qty * signal_price < float(min_notional):
            await self._reject_live_order(
                symbol, "order_below_min_notional",
                f"notional {qty * signal_price:.8f} USDT is below the "
                f"{float(min_notional)} USDT minimum for {symbol}")
            return

        # One stable client order id for ALL attempts. If a request times out after
        # the exchange accepted it, the retry is rejected as a duplicate (-2010)
        # instead of opening a second real position.
        client_order_id = f"bt{uuid.uuid4().hex[:20]}"

        order = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                order = await self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=qty,
                    newClientOrderId=client_order_id,
                )
                break
            except Exception as e:
                last_error = e
                if self._is_duplicate_order_error(e):
                    order = await self._lookup_order(symbol, client_order_id)
                    if order is not None:
                        logger.warning(
                            f"Live order for {symbol} had already been accepted "
                            f"(duplicate clientOrderId {client_order_id}); adopting the existing order"
                        )
                        break
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        if order is None:
            await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
                "level": "critical", "type": "order_failed",
                "message": f"Live order failed after 3 attempts: {last_error}",
            }))
            return

        # ── Post-fill bookkeeping — deliberately OUTSIDE the retry loop ──
        # The order is live on the exchange now. A bookkeeping failure must never
        # trigger another market order: previously the whole block was inside the
        # retry `try`, so a KeyError while recording the fill resubmitted the order
        # and produced up to 3 real fills for a single signal, none of them tracked.
        order_id = str(order.get("orderId", ""))
        status = str(order.get("status", "unknown")).lower()
        try:
            self._orders[order_id] = {
                "id": order_id,
                "symbol": symbol,
                "status": status,
                "binance_order_id": order.get("orderId"),
                "client_order_id": client_order_id,
            }
            # Track position in-memory so exit/reduce handlers work
            price = float(order.get("price", data.get("price", 0)) or 0)
            if price == 0:
                price = float(data.get("price", 0) or 0)
            trade_group = str(uuid.uuid4())[:8]
            # Phase P3 stop-width plumbing (no-op with vol targeting off, since
            # `vol_stop_ctx` returns {}): the live position records the same
            # forecast context the sim path does, so PositionGuard trails both
            # book types identically.
            live_vol_ctx = self.vol_stop_ctx(symbol)
            self._positions[symbol] = {
                "symbol": symbol,
                "side": data.get("side", "long"),
                "quantity": qty,
                "entry_price": price,
                "current_price": price,
                "unrealized_pnl": 0,
                "stop_loss": data.get("stop_loss"),
                "stop_vol_pct": live_vol_ctx.get("vol_pct"),
                "position_type": data.get("position_type", "satellite"),
                "position_value": qty * price,
                "amount_usdt": data.get("amount_usdt", qty * price),
                "trade_group": trade_group,
                "strategy_name": data.get("strategy_name", ""),
                "strategy": data.get("strategy", "manual"),
                "client_order_id": client_order_id,
                # Threaded through to the close/reduce rows (see _execute_sim).
                "timeframe": data.get("timeframe") or "1h",
            }
            await self.event_bus.publish(Event(EventType.ORDER_UPDATE, {
                "order_id": order_id, "symbol": symbol,
                "status": status, "mode": "live",
            }))
            await self.event_bus.publish(Event(EventType.POSITION_UPDATE, {
                "symbol": symbol, "side": data.get("side", "long"),
                "quantity": qty, "entry_price": price,
                "current_price": price,
                "position_type": data.get("position_type", "satellite"),
                "position_value": qty * price,
                "closed": False, "pnl": 0,
            }))
        except Exception as e:
            logger.exception(f"Live order {order_id} filled but local bookkeeping failed: {e}")
            await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
                "level": "critical", "type": "order_tracking_failed",
                "message": (f"Order {order_id} ({symbol}) was accepted by the exchange but could not be "
                            f"tracked locally ({e}). It was NOT resubmitted — manual reconciliation "
                            f"required."),
            }))


    def get_open_positions(self) -> dict:
        return self._positions.copy()

    async def close_position(self, symbol: str, reduce_pct: float = 100,
                             current_price: float = 0, reason: str = "manual") -> dict:
        """Close or reduce a position. Returns result dict.

        ``reason`` is stored on the exit row as ``exit_reason`` (``manual``,
        ``stop_loss``, ``take_profit``, ``indicator`` …).  ``closed_at`` is
        always written: it used to stay NULL on every row (1370/1370 in the live
        DB), which made the `/api/db/cleanup` retention predicate
        ``status='closed' AND closed_at < now-365d`` match nothing for ever.

        The whole mutation is atomic per symbol.  Two callers that read the same
        pre-mutation basis and both credit their share used to create money:
        each booked a full-precision close against one basis and both rows landed
        in one ``trade_group``.  Here the revision the caller saw is compared
        once the lock is held, so an overlapping reduce is refused instead of
        applied twice, and a full close that finds the position already gone
        returns a structured ``ok=False`` instead of raising ``KeyError``.
        """
        if symbol not in self._positions:
            return {"ok": False, "error": f"No position for {symbol}"}
        observed = self._position_rev.get(symbol, 0)
        async with self._symbol_lock(symbol):
            # Same gate as `_execute_sim`: a reset cannot erase the ledger in the
            # middle of this close's row-insert + cash movement, and this close
            # cannot run against a ledger that is being reset.  The revision check
            # below is the second line of defence when a reset was already waiting
            # for this symbol's lock.
            async with self._mutation_slot():
                if self._position_rev.get(symbol, 0) != observed:
                    message = (f"{symbol} position changed while this reduce waited for "
                               f"the lock; refusing to apply a stale basis")
                    logger.warning(f"Concurrent close/reduce refused: {message}")
                    await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
                        "level": "warning", "type": "concurrent_close_rejected",
                        "message": f"{symbol} 并发平仓/减仓已拒绝（避免重复计算资金）",
                        "symbol": symbol,
                    }))
                    return {"ok": False, "error": message}
                return await self._close_position_locked(symbol, reduce_pct,
                                                         current_price, reason)

    async def _close_position_locked(self, symbol: str, reduce_pct: float,
                                     current_price: float, reason: str) -> dict:
        import aiosqlite as aio
        # Re-checked under the lock: a concurrent 100% close may have removed the
        # position while this call waited (that used to raise KeyError on `del`).
        if symbol not in self._positions:
            return {"ok": False, "error": f"No position for {symbol}"}
        pos = self._positions[symbol]
        trade_group = pos.get("trade_group", "")
        reduce_pct = min(100, max(1, reduce_pct))
        exit_reason = str(reason or "manual")[:40]

        # Capture original values before any mutation
        original_qty = pos["quantity"]
        original_amount = pos.get("amount_usdt", original_qty * pos["entry_price"])
        entry = pos["entry_price"]
        side = pos["side"]

        # If a reduce would leave less than $10 notional, close fully instead
        remaining_value_after_reduce = original_amount * (1 - reduce_pct / 100)
        if reduce_pct < 100 and remaining_value_after_reduce < 10.0:
            reduce_pct = 100

        close_qty = original_qty * reduce_pct / 100
        remaining_qty = original_qty - close_qty

        # ---- cost model: sell side is marked down by (spread/2 + slippage) ---
        # `entry` is the cash basis the open actually deducted (qty × entry_price
        # of the open row), so it already contains the buy fee and slippage.  The
        # close therefore returns that full basis as `invested_returned` and
        # `pnl` adds only what the exit earned *on top of it*: the sell-side cost.
        # Charging the buy cost here as well (as this used to) took it out of the
        # account twice — once inside the returned basis, once in `pnl` — and put
        # the balance below `10000 − open notional + realised` by that amount on
        # every single close.  `pnl` stays the NET figure: it is exactly what the
        # balance gains beyond the returned basis, so the callers' unchanged
        # `invested_returned + pnl` keeps the identity true to the cent.
        sell_quote = sim_cost_quote(symbol,
                                    # closing a long SELLS into the bid; closing a
                                    # short BUYS at the ask.  `side` above is the
                                    # position's side and would mark the wrong way.
                                    "sell" if side == "long" else "buy",
                                    "market", current_price, close_qty,
                                    await self._cost_settings())
        exit_price = sell_quote["fill_price"] or current_price
        sell_fee = sell_quote["fee_usdt"]
        sell_slippage = sell_quote["slippage_usdt"]
        invested_close = original_amount * reduce_pct / 100.0

        share = (reduce_pct / 100.0) if original_qty else 0.0
        buy_fee = float(pos.get("fee") or 0.0) * share
        buy_slippage = float(pos.get("slippage") or 0.0) * share
        # `cost` is what the EXIT adds on top of the returned basis: the buy fee and
        # slippage are already inside `entry`, so they are reported through
        # `invested_returned` rather than charged a second time here.
        cost_total = sell_fee + sell_slippage

        gross_pnl = ((exit_price - entry) * close_qty if side == "long"
                     else (entry - exit_price) * close_qty)
        # `pnl` is what the balance gains ON TOP of the returned basis, so it is the
        # gross move over that basis less the exit's own cost.  The buy cost lives
        # inside the basis — charging it here as well is precisely the double charge
        # that dragged the balance below the identity on every close.
        pnl = gross_pnl - cost_total
        pnl_pct = ((exit_price - entry) / entry * 100 if side == "long"
                   else (entry - exit_price) / entry * 100) if entry else 0.0

        db = await aio.connect(self.config.db_path)
        # The timeframe the position was opened on (the signal's), not a hard
        # coded "1h" — "1h" only when the open row carried nothing.
        timeframe = pos.get("timeframe") or "1h"
        try:
            if reduce_pct >= 100:
                self._positions.pop(symbol, None)
                # `closed_at` + `exit_reason` on every row of the round trip: the
                # OPEN row is what `/api/history/trades` and the cleanup retention
                # predicate look at, and it used to keep NULL there for ever.
                #
                # UNCONDITIONAL: the old `if trade_group:` guard meant a position
                # whose open row carried `trade_group=''` (legacy rows, a snapshot
                # restored without one) kept its basis counted as an open row while
                # the close row returned it as well — the same cash twice (Δ −200).
                # The group is the precise identity when it exists; a dangling or
                # empty one falls back to the symbol's open row.
                flipped = 0
                if trade_group:
                    cursor = await db.execute(
                        "UPDATE trades SET status='closed',"
                        " closed_at=COALESCE(closed_at, CURRENT_TIMESTAMP),"
                        " exit_reason=COALESCE(exit_reason, ?)"
                        " WHERE trade_group=? AND trade_group!=''"
                        "   AND action='open' AND status='open'",
                        (exit_reason, trade_group))
                    flipped = cursor.rowcount
                if not flipped:
                    await db.execute(
                        "UPDATE trades SET status='closed',"
                        " closed_at=COALESCE(closed_at, CURRENT_TIMESTAMP),"
                        " exit_reason=COALESCE(exit_reason, ?)"
                        " WHERE symbol=? AND action='open' AND status='open'",
                        (exit_reason, symbol))
                await db.execute(
                    "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct,"
                    " strategy, timeframe, position_type, status, trader, strategy_name, action, trade_group, reduce_pct,"
                    " fill_price, fee, slippage, closed_at, exit_reason, stop_loss, take_profits)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,?,?,?)",
                    (symbol, side, entry, exit_price, close_qty,
                     round(pnl, 2), round(pnl_pct, 2), "auto", timeframe,
                     pos.get("position_type", "satellite"), "closed",
                     pos.get("trader", "ai"), pos.get("strategy_name", ""), "close", trade_group, 100,
                     exit_price, round(sell_fee, 8), round(sell_slippage, 8),
                     exit_reason, pos.get("stop_loss"),
                     _json_levels(pos.get("take_profits"))))
                # Flat: the snapshot row goes with the position.  The ledger row
                # above is the record that survives.
                await db.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
            else:
                # A reduce hands part of the cash basis back, so the *per-unit*
                # basis must move with the quantity or the ledger double-counts:
                # `entry_price` still embedded the whole buy cost while the open
                # row's `quantity` was rewritten to the remainder, so
                # `quantity × entry_price` no longer equalled the cash still
                # invested.  Keeping `remaining_qty × entry_price ==
                # remaining_amount` is the same rule the open/close path follows.
                pos["quantity"] = remaining_qty
                pos["amount_usdt"] = (original_amount * remaining_qty / original_qty
                                      if original_qty > 0 else 0.0)
                pos["fee"] = float(pos.get("fee") or 0.0) - buy_fee
                pos["slippage"] = float(pos.get("slippage") or 0.0) - buy_slippage
                pos["entry_price"] = (pos["amount_usdt"] / remaining_qty
                                      if remaining_qty > 0 else pos.get("entry_price", 0.0))
                pos["position_value"] = pos["amount_usdt"]
                # Resize the OPEN row this position belongs to (see the close
                # branch): by group when it has one, by symbol+status otherwise.
                # A group-less open row left un-resized kept the WHOLE original
                # basis counted while the reduce row handed part of it back.
                resized = 0
                if trade_group:
                    cursor = await db.execute(
                        "UPDATE trades SET quantity=?, entry_price=?, fee=?, slippage=?"
                        " WHERE trade_group=? AND trade_group!=''"
                        "   AND action='open' AND status='open'",
                        (round(remaining_qty, 8), pos["entry_price"], pos["fee"],
                         pos["slippage"], trade_group))
                    resized = cursor.rowcount
                if not resized:
                    await db.execute(
                        "UPDATE trades SET quantity=?, entry_price=?, fee=?, slippage=?"
                        " WHERE symbol=? AND action='open' AND status='open'",
                        (round(remaining_qty, 8), pos["entry_price"], pos["fee"],
                         pos["slippage"], symbol))
                await db.execute(
                    "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct,"
                    " strategy, timeframe, position_type, status, trader, strategy_name, action, trade_group, reduce_pct,"
                    " fill_price, fee, slippage, closed_at, exit_reason, stop_loss, take_profits)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,?,?,?)",
                    (symbol, side, entry, exit_price, close_qty,
                     round(pnl, 2), round(pnl_pct, 2), "auto", timeframe,
                     pos.get("position_type", "satellite"), "closed",
                     pos.get("trader", "ai"), pos.get("strategy_name", ""), "reduce", trade_group, round(reduce_pct, 1),
                     exit_price, round(sell_fee, 8), round(sell_slippage, 8),
                     exit_reason, pos.get("stop_loss"),
                     _json_levels(pos.get("take_profits"))))
                # The snapshot carries the NEW basis: restoring from the open row
                # alone would resurrect the pre-reduce quantity.
                await self._persist_position(db, pos)
            await db.commit()
        finally:
            await db.close()
        # Published before the POSITION_UPDATE below, still inside the caller's
        # per-symbol lock: any reduce that read the pre-mutation basis is refused.
        self._bump_revision(symbol)

        closed = reduce_pct >= 100
        # Publish full position data so risk_manager can track accurately
        pos_data = {
            "symbol": symbol, "closed": closed, "pnl": pnl,
        }
        if not closed:
            # On reduce, include full fields so risk_manager doesn't lose data
            pos_data.update({
                "side": pos["side"], "quantity": pos["quantity"],
                "entry_price": pos["entry_price"],
                "current_price": current_price,
                "position_type": pos.get("position_type", "satellite"),
                "position_value": pos.get("amount_usdt", pos["quantity"] * pos["entry_price"]),
                "amount_usdt": pos.get("amount_usdt", pos["quantity"] * pos["entry_price"]),
                "stop_loss": pos.get("stop_loss"),
                "take_profits": parse_levels(pos.get("take_profits")),
                "trade_group": pos.get("trade_group", ""),
            })
        await self.event_bus.publish(Event(EventType.POSITION_UPDATE, pos_data))
        return {"ok": True, "closed": closed, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
                # NOT rounded: the balance must gain back exactly the basis the open
                # row lends to the identity (`quantity × entry_price`).  Rounding
                # this to 2 dp used to lose up to half a cent on every close.
                "invested_returned": invested_close,
                "gross_pnl": round(gross_pnl, 4), "cost": round(cost_total, 4),
                "fee": round(sell_fee, 8), "slippage": round(sell_slippage, 8),
                "fill_price": exit_price}

    def get_orders(self) -> dict:
        return self._orders.copy()

    async def stop(self):
        self._running = False
        self.event_bus.unsubscribe(EventType.ORDER_REQUEST, self._on_order_request)
        if self.client:
            await self.client.close_connection()
