import asyncio
import time
import uuid
from typing import Optional
from binance import AsyncClient
from binance.enums import *
from loguru import logger

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from db.database import load_sim_balance, save_sim_balance, atomic_adjust_balance


class OrderExecutor:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.client: Optional[AsyncClient] = None
        self._running = False
        self._orders: dict[str, dict] = {}
        self._positions: dict[str, dict] = {}
        self._risk_manager = None

    def wire_risk_manager(self, rm):
        self._risk_manager = rm

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
        """Restore open positions from DB after restart."""
        import aiosqlite as aio
        from loguru import logger
        try:
            db = await aio.connect(self.config.db_path)
            db.row_factory = aio.Row
            cursor = await db.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY id ASC")
            rows = [dict(r) for r in await cursor.fetchall()]
            await db.close()
            for r in rows:
                qty = r["quantity"]
                entry = r["entry_price"]
                symbol = r["symbol"]
                self._positions[symbol] = {
                    "symbol": symbol,
                    "side": r["side"],
                    "quantity": qty,
                    "entry_price": entry,
                    "current_price": entry,
                    "unrealized_pnl": 0,
                    "stop_loss": None,
                    "position_type": r.get("position_type", "satellite"),
                    "position_value": qty * entry,
                    "amount_usdt": qty * entry,
                    "trade_group": r.get("trade_group", ""),
                    "strategy_name": r.get("strategy_name", ""),
                    "strategy": r.get("strategy", "manual"),
                }
            logger.info(f"Restored {len(rows)} open positions from DB")
        except Exception as e:
            logger.warning(f"Failed to restore positions: {e}")

    async def _on_order_request(self, event: Event):
        if self.config.mode == "sim":
            await self._execute_sim(event.data)
        elif self.config.mode == "live":
            await self._execute_live(event.data)

    async def _execute_sim(self, data: dict):
        order_id = f"sim_{int(time.time() * 1000)}"
        price = data.get("price", 0)
        qty = data.get("quantity", 0)
        symbol = data.get("symbol", "")
        side = data.get("side", "long")
        trade_group = str(uuid.uuid4())[:8]
        amount_usdt = data.get("amount_usdt", qty * price)

        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": "market",
            "price": price,
            "quantity": qty,
            "filled_qty": qty,
            "status": "filled",
            "binance_order_id": None,
            "stop_loss": data.get("stop_loss"),
            "take_profits": data.get("take_profits", []),
        }
        self._orders[order_id] = order
        self._positions[symbol] = {
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "entry_price": price,
            "current_price": price,
            "unrealized_pnl": 0,
            "stop_loss": data.get("stop_loss"),
            "position_type": data.get("position_type", "satellite"),
            "position_value": qty * price,
            "amount_usdt": amount_usdt,
            "trade_group": trade_group,
            "strategy_name": data.get("strategy_name", ""),
            "strategy": data.get("strategy", "manual"),
        }

        await self.event_bus.publish(Event(EventType.ORDER_UPDATE, {
            "order_id": order_id, "symbol": symbol, "status": "filled", "mode": "sim",
        }))
        await self.event_bus.publish(Event(EventType.POSITION_UPDATE, {
            "symbol": symbol, "side": side, "quantity": qty,
            "entry_price": price, "current_price": price,
            "position_type": data.get("position_type", "satellite"),
            "position_value": qty * price,
            "amount_usdt": amount_usdt,
            "stop_loss": data.get("stop_loss"),
            "trade_group": trade_group,
            "closed": False, "pnl": 0,
        }))

        # Persist trade to DB
        import aiosqlite as aio
        db = await aio.connect(self.config.db_path)
        await db.execute(
            "INSERT INTO trades (symbol, side, entry_price, quantity, strategy, timeframe, position_type, status, trader, strategy_name, action, trade_group) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (symbol, side, price, qty, data.get("strategy", "manual"), data.get("timeframe", "1h"),
             data.get("position_type", "satellite"), "open",
             data.get("trader", "manual"), data.get("strategy_name", ""),
             "open", trade_group))
        await db.commit()
        await db.close()

        # Atomically deduct invested amount from sim balance (DB + risk_manager)
        try:
            new_balance = await atomic_adjust_balance(-amount_usdt, self.config.db_path)
            if self._risk_manager:
                self._risk_manager.update_balance(new_balance)
            logger.info(f"Balance deducted: {new_balance + amount_usdt:.0f} -> {new_balance:.0f} (-{amount_usdt:.0f} USDT for {symbol})")
        except Exception as e:
            logger.warning(f"Failed to update balance on position open: {e}")

    async def _execute_live(self, data: dict):
        symbol = data.get("symbol", "")
        side = SIDE_BUY if data.get("side") == "long" else SIDE_SELL
        qty = data.get("quantity", 0)

        for attempt in range(3):
            try:
                order = await self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=round(qty, 5),
                )
                order_id = str(order["orderId"])
                self._orders[order_id] = {
                    "id": order_id,
                    "symbol": symbol,
                    "status": order["status"].lower(),
                    "binance_order_id": order["orderId"],
                }
                # Track position in-memory so exit/reduce handlers work
                price = float(order.get("price", data.get("price", 0)))
                if price == 0:
                    price = data.get("price", 0)
                trade_group = str(uuid.uuid4())[:8]
                self._positions[symbol] = {
                    "symbol": symbol,
                    "side": data.get("side", "long"),
                    "quantity": qty,
                    "entry_price": price,
                    "current_price": price,
                    "unrealized_pnl": 0,
                    "stop_loss": data.get("stop_loss"),
                    "position_type": data.get("position_type", "satellite"),
                    "position_value": qty * price,
                    "amount_usdt": data.get("amount_usdt", qty * price),
                    "trade_group": trade_group,
                    "strategy_name": data.get("strategy_name", ""),
                    "strategy": data.get("strategy", "manual"),
                }
                await self.event_bus.publish(Event(EventType.ORDER_UPDATE, {
                    "order_id": order_id, "symbol": symbol,
                    "status": order["status"].lower(), "mode": "live",
                }))
                await self.event_bus.publish(Event(EventType.POSITION_UPDATE, {
                    "symbol": symbol, "side": data.get("side", "long"),
                    "quantity": qty, "entry_price": price,
                    "current_price": price,
                    "position_type": data.get("position_type", "satellite"),
                    "position_value": qty * price,
                    "closed": False, "pnl": 0,
                }))
                return
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
                        "level": "critical", "type": "order_failed",
                        "message": f"Live order failed after 3 attempts: {e}",
                    }))

    def get_open_positions(self) -> dict:
        return self._positions.copy()

    async def close_position(self, symbol: str, reduce_pct: float = 100, current_price: float = 0) -> dict:
        """Close or reduce a position. Returns result dict."""
        import aiosqlite as aio
        if symbol not in self._positions:
            return {"ok": False, "error": f"No position for {symbol}"}
        pos = self._positions[symbol]
        trade_group = pos.get("trade_group", "")
        reduce_pct = min(100, max(1, reduce_pct))

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

        pnl = (current_price - entry) * close_qty if side == "long" else (entry - current_price) * close_qty
        pnl_pct = (current_price - entry) / entry * 100 if side == "long" else (entry - current_price) / entry * 100

        invested_close = original_amount * reduce_pct / 100

        db = await aio.connect(self.config.db_path)
        try:
            if reduce_pct >= 100:
                del self._positions[symbol]
                if trade_group:
                    await db.execute(
                        "UPDATE trades SET status='closed' WHERE trade_group=? AND trade_group!=''",
                        (trade_group,))
                await db.execute(
                    "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct,"
                    " strategy, timeframe, position_type, status, trader, strategy_name, action, trade_group, reduce_pct)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (symbol, side, entry, current_price, close_qty,
                     round(pnl, 2), round(pnl_pct, 2), "auto", "1h",
                     pos.get("position_type", "satellite"), "closed",
                     pos.get("trader", "ai"), pos.get("strategy_name", ""), "close", trade_group, 100))
            else:
                pos["quantity"] = remaining_qty
                pos["amount_usdt"] = original_amount * remaining_qty / original_qty if original_qty > 0 else 0
                if trade_group:
                    await db.execute(
                        "UPDATE trades SET quantity=? WHERE trade_group=? AND action='open' AND status='open'",
                        (round(remaining_qty, 8), trade_group))
                await db.execute(
                    "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct,"
                    " strategy, timeframe, position_type, status, trader, strategy_name, action, trade_group, reduce_pct)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (symbol, side, entry, current_price, close_qty,
                     round(pnl, 2), round(pnl_pct, 2), "auto", "1h",
                     pos.get("position_type", "satellite"), "closed",
                     pos.get("trader", "ai"), pos.get("strategy_name", ""), "reduce", trade_group, round(reduce_pct, 1)))
            await db.commit()
        finally:
            await db.close()

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
                "trade_group": pos.get("trade_group", ""),
            })
        await self.event_bus.publish(Event(EventType.POSITION_UPDATE, pos_data))
        return {"ok": True, "closed": closed, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
                "invested_returned": round(invested_close, 2)}

    def get_orders(self) -> dict:
        return self._orders.copy()

    async def stop(self):
        self._running = False
        self.event_bus.unsubscribe(EventType.ORDER_REQUEST, self._on_order_request)
        if self.client:
            await self.client.close_connection()
