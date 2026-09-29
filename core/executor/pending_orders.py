"""Pending limit-order store + background matcher.

Limit orders are the only order type that is not filled immediately, so they
need three things the market-order path never needed:

1. **Persistence** — ``pending_orders`` (see ``db/database.py``).  An order
   placed before a restart must still be there (and still cancellable) after
   it, so the table is the single source of truth and the matcher restores its
   working set from it on start.
2. **Reserved (frozen) funds** — placing a limit order does *not* move cash.
   The open orders are simply summed and reported as ``frozen`` by
   ``/api/account``, and ``available`` becomes ``balance - frozen``.  Cash is
   only deducted at fill time, by the normal executor path, exactly like a
   market order.
3. **A fill loop** — every ``interval`` seconds each open order is compared
   against the engine's last price; when it crosses, the fill goes through the
   *same* risk pipeline as a manual market order
   (``RiskManager.check_signal`` → ``ORDER_REQUEST``) so no safety gate is
   bypassed by the limit route.

The loop is a plain ``asyncio.Task`` owned by this object.  ``stop()`` cancels
it and awaits it, so an application shutdown never leaves the matcher touching
a database (or an executor) that is being torn down.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import aiosqlite
from loguru import logger

from app.event_bus import Event, EventType

#: How often the matcher compares open orders against the last price.
MATCH_INTERVAL_SEC = 5.0
#: Interval used for the historical-price fallback (a closed 1m candle).
FALLBACK_INTERVAL = "1m"

STATUS_OPEN = "open"
STATUS_FILLED = "filled"
STATUS_CANCELLED = "cancelled"

_COLUMNS = ("id, symbol, side, type, price, quantity, amount_usdt, status, "
            "created_at, filled_at, fill_price, reason, position_type, "
            "stop_loss_pct, trader, strategy_name")


@dataclass
class PendingOrder:
    """One limit order.  ``to_dict`` is the wire shape frozen in the contract."""

    id: int
    symbol: str
    side: str            # long | short
    type: str            # always "limit"
    price: float
    quantity: float
    amount_usdt: float
    status: str
    created_at: Optional[str] = None
    filled_at: Optional[str] = None
    fill_price: Optional[float] = None
    reason: Optional[str] = None
    # Fill-time attributes: needed to build the risk signal when the matcher
    # finally fills the order (possibly after a restart).
    position_type: str = "satellite"
    stop_loss_pct: float = 2.0
    trader: str = "manual"
    strategy_name: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "type": self.type,
            "price": round(float(self.price or 0), 8),
            "quantity": round(float(self.quantity or 0), 6),
            "amount_usdt": round(float(self.amount_usdt or 0), 2),
            "status": self.status,
            "created_at": self.created_at,
            "filled_at": self.filled_at,
            "fill_price": self.fill_price,
            "reason": self.reason,
        }


def _row_to_order(row) -> PendingOrder:
    d = dict(row)
    return PendingOrder(
        id=d["id"], symbol=d["symbol"], side=d["side"], type=d.get("type") or "limit",
        price=float(d.get("price") or 0),
        quantity=float(d.get("quantity") or 0),
        amount_usdt=float(d.get("amount_usdt") or 0),
        status=d.get("status") or STATUS_OPEN,
        created_at=str(d["created_at"]) if d.get("created_at") is not None else None,
        filled_at=str(d["filled_at"]) if d.get("filled_at") is not None else None,
        fill_price=float(d["fill_price"]) if d.get("fill_price") is not None else None,
        reason=d.get("reason"),
        position_type=d.get("position_type") or "satellite",
        stop_loss_pct=float(d.get("stop_loss_pct") or 0.0),
        trader=d.get("trader") or "manual",
        strategy_name=d.get("strategy_name") or "",
    )


class PendingOrderStore:
    """SQLite-backed pending limit-order book (one connection per operation)."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    async def _connect(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        return db

    async def add(self, symbol: str, side: str, price: float, quantity: float,
                  amount_usdt: float, position_type: str = "satellite",
                  stop_loss_pct: float = 2.0, trader: str = "manual",
                  strategy_name: str = "") -> PendingOrder:
        db = await self._connect()
        try:
            cursor = await db.execute(
                "INSERT INTO pending_orders "
                "(symbol, side, type, price, quantity, amount_usdt, position_type, "
                " stop_loss_pct, trader, strategy_name, status) "
                "VALUES (?,?,'limit',?,?,?,?,?,?,?,'open')",
                (symbol, side, float(price), round(float(quantity), 8),
                 round(float(amount_usdt), 2), position_type,
                 float(stop_loss_pct), trader, strategy_name))
            await db.commit()
            order_id = cursor.lastrowid
        finally:
            await db.close()
        order = await self.get(order_id)
        assert order is not None
        return order

    async def get(self, order_id: int) -> Optional[PendingOrder]:
        db = await self._connect()
        try:
            cursor = await db.execute(
                f"SELECT {_COLUMNS} FROM pending_orders WHERE id=?", (order_id,))
            row = await cursor.fetchone()
            return _row_to_order(row) if row else None
        finally:
            await db.close()

    async def list(self, status: str = "all", limit: int = 50) -> list[PendingOrder]:
        where, params = "", []
        if status == "open":
            where = "WHERE status='open'"
        elif status == "history":
            where = "WHERE status!='open'"
        db = await self._connect()
        try:
            cursor = await db.execute(
                f"SELECT {_COLUMNS} FROM pending_orders {where} "
                "ORDER BY id DESC LIMIT ?", (*params, int(limit)))
            return [_row_to_order(r) for r in await cursor.fetchall()]
        finally:
            await db.close()

    async def open_orders(self) -> list[PendingOrder]:
        return await self.list(status="open", limit=1000)

    async def frozen_total(self) -> float:
        """Sum of ``amount_usdt`` over open limit orders (the 'frozen' funds)."""
        db = await self._connect()
        try:
            cursor = await db.execute(
                "SELECT COALESCE(SUM(amount_usdt), 0) AS total "
                "FROM pending_orders WHERE status='open'")
            row = await cursor.fetchone()
            return float(row["total"] or 0.0)
        finally:
            await db.close()

    async def open_count(self) -> int:
        db = await self._connect()
        try:
            cursor = await db.execute(
                "SELECT COUNT(*) AS cnt FROM pending_orders WHERE status='open'")
            row = await cursor.fetchone()
            return int(row["cnt"] or 0)
        finally:
            await db.close()

    async def mark_filled(self, order_id: int, fill_price: float) -> bool:
        db = await self._connect()
        try:
            cursor = await db.execute(
                "UPDATE pending_orders SET status='filled', fill_price=?, "
                "filled_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND status='open'",
                (float(fill_price), order_id))
            await db.commit()
            return cursor.rowcount > 0
        finally:
            await db.close()

    async def mark_cancelled(self, order_id: int, reason: str = "") -> bool:
        db = await self._connect()
        try:
            cursor = await db.execute(
                "UPDATE pending_orders SET status='cancelled', reason=?, "
                "filled_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND status='open'",
                (reason or "", order_id))
            await db.commit()
            return cursor.rowcount > 0
        finally:
            await db.close()


def _has_async_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class LimitOrderMatcher:
    """Background loop that fills crossed limit orders through the risk pipeline."""

    def __init__(self, app, config, event_bus, store: PendingOrderStore,
                 interval: float = MATCH_INTERVAL_SEC):
        self.app = app
        self.config = config
        self.event_bus = event_bus
        self.store = store
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def start(self):
        """Sync-style start (no await before the task exists).

        Deliberately synchronous: ``stop()`` cancels ``self._task``, so if this
        were a coroutine that awaited the restore *before* assigning the task,
        a shutdown during start-up would cancel the start coroutine itself and
        the loop would never run (or be cancelled mid-way).
        """
        if self._running:
            return
        self._running = True
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._loop(), name="limit_order_matcher")

    async def restore(self):
        """Log how many open orders survived the restart (informational)."""
        try:
            restored = await self.store.open_orders()
            logger.info(f"Limit-order matcher: restored {len(restored)} open order(s)")
            return len(restored)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Limit-order matcher: could not restore open orders: {e}")
            return 0
    async def stop(self):
        self._running = False
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except BaseException:  # CancelledError included — shutdown path
                pass
            self._task = None

    async def _loop(self, max_iterations: int | None = None):
        iterations = 0
        try:
            while self._running:
                try:
                    await self.evaluate_once()
                except Exception as e:
                    logger.warning(f"Limit-order matcher iteration failed: {e}")
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
                    break  # stop() was called
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    # price resolution
    # ------------------------------------------------------------------
    async def _price(self, symbol: str) -> Optional[float]:
        """Engine price first, historical last close as the fallback."""
        price_fn = getattr(self.app.state, "get_price", None)
        if price_fn:
            try:
                price = price_fn(symbol)
            except Exception:
                price = None
            if price:
                return float(price)
        try:
            return await self._historical_price(symbol)
        except Exception as e:
            logger.warning(f"Limit-order matcher: no price for {symbol}: {e}")
            return None

    async def _historical_price(self, symbol: str) -> Optional[float]:
        md = getattr(self.app.state, "market_data", None)
        if md is None:
            engine = getattr(self.app.state, "strategy_engine", None)
            md = getattr(engine, "market_data", None)
        if md is None:
            return None
        df = await md.get_historical(symbol, FALLBACK_INTERVAL, limit=2)
        if df is None or len(df) == 0:
            return None
        return float(df["close"].iloc[-1])

    @staticmethod
    def _crossed(order: PendingOrder, last: float) -> bool:
        return last <= order.price if order.side == "long" else last >= order.price

    # ------------------------------------------------------------------
    # matching
    # ------------------------------------------------------------------
    async def evaluate_once(self) -> list[dict]:
        """One matching pass. Returns the outcomes (for tests / logging)."""
        orders = await self.store.open_orders()
        outcomes: list[dict] = []
        for order in orders:
            if not self._running:
                break
            last = await self._price(order.symbol)
            if last is None or last <= 0:
                continue
            if not self._crossed(order, last):
                continue
            outcomes.append(await self._fill(order, last))
        return outcomes

    async def _fill(self, order: PendingOrder, fill_price: float) -> dict:
        """Route a crossed limit order through the shared risk pipeline."""
        side = order.side if order.side in ("long", "short") else "long"
        stop_loss = (fill_price * (1 - order.stop_loss_pct / 100) if side == "long"
                     else fill_price * (1 + order.stop_loss_pct / 100))
        signal = {
            "symbol": order.symbol, "side": side,
            "price": fill_price, "quantity": round(order.quantity, 6),
            "stop_loss": round(stop_loss, 2),
            "position_type": order.position_type,
            "amount_usdt": order.amount_usdt,
            "trader": order.trader,
            "strategy_name": order.strategy_name,
            "strategy": "limit",
            "timeframe": "1h",
            "confidence": 1.0,
            "order_id": order.id,
            "order_type": "limit",
        }

        risk_manager = getattr(self.app.state, "risk_manager", None)
        if risk_manager is None:
            # Fail closed, exactly like the manual market path: without the risk
            # manager none of the safety gates run.
            reason = "风控未就绪，限价单已撤销（不允许绕过风控）"
            await self.store.mark_cancelled(order.id, reason)
            logger.warning(f"Limit order {order.id} cancelled: risk manager unavailable")
            return {"id": order.id, "status": STATUS_CANCELLED, "reason": reason}

        result = await risk_manager.check_signal(signal)
        if not result.approved:
            reason = f"风控拒绝: {result.reason}"
            await self.store.mark_cancelled(order.id, reason)
            logger.info(f"Limit order {order.id} ({order.symbol}) rejected: {result.reason}")
            return {"id": order.id, "status": STATUS_CANCELLED, "reason": reason}

        if result.adjusted_quantity is not None and result.adjusted_quantity < signal["quantity"]:
            signal["quantity"] = result.adjusted_quantity
            signal["amount_usdt"] = result.adjusted_quantity * fill_price
        if result.adjusted_stop_loss is not None:
            signal["stop_loss"] = result.adjusted_stop_loss
        if result.adjusted_leverage is not None:
            signal["leverage"] = result.adjusted_leverage

        await self.store.mark_filled(order.id, fill_price)
        await self.event_bus.publish(Event(EventType.ORDER_REQUEST, signal))
        logger.info(f"Limit order {order.id} filled: {order.symbol} {side} "
                    f"{signal['quantity']} @ {fill_price}")
        return {"id": order.id, "status": STATUS_FILLED, "fill_price": fill_price,
                "quantity": signal["quantity"]}


def start_matcher_task(app, config, event_bus) -> Optional[asyncio.Task]:
    """Create + schedule the matcher, stashing it on ``app.state``.

    Returns the task (or ``None`` when there is no running loop, e.g. a sync
    test context). ``stop_matcher(app)`` cancels it.  Synchronous on purpose —
    see :meth:`LimitOrderMatcher.start`.
    """
    if not _has_async_loop():
        logger.warning("Limit-order matcher not started: no running event loop")
        return None
    store = getattr(app.state, "pending_order_store", None)
    if store is None:
        store = PendingOrderStore(config.db_path)
        app.state.pending_order_store = store
    matcher = getattr(app.state, "limit_order_matcher", None)
    if matcher is None:
        matcher = LimitOrderMatcher(app, config, event_bus, store)
        app.state.limit_order_matcher = matcher
    matcher._running = True
    matcher._stop = asyncio.Event()
    matcher._task = asyncio.create_task(matcher._loop(), name="limit_order_matcher")
    # Fire-and-forget restore log; the loop reads the table on every pass, so the
    # working set is correct even if this never completes.
    asyncio.ensure_future(matcher.restore())
    app.state.limit_order_matcher_task = matcher._task
    return matcher._task


async def stop_matcher(app) -> None:
    """Stop the matcher if it is running (safe to call more than once)."""
    matcher = getattr(app.state, "limit_order_matcher", None)
    if matcher is not None:
        await matcher.stop()
