"""Trading endpoints + position/account HTMX partials."""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse

from app.event_bus import Event, EventType
from db.database import atomic_adjust_balance

#: Signal attribution default comes from the interval registry, not a literal.
from core.market_data.provider import DEFAULT_TIMEFRAME

from web.deps import _require_trader
from web.rendering import _render, _T


def register(app: FastAPI, ctx) -> None:
    config = ctx.config
    event_bus = ctx.event_bus
    _balance_lock = ctx._balance_lock

    # ---- Trading endpoints ----
    @app.post("/api/trade")
    async def execute_trade(request: Request, symbol: str = Form(...), side: str = Form(...),
                            amount_usdt: float = Form(100), position_type: str = Form("satellite"),
                            stop_loss_pct: float = Form(2.0), trader: str = Form("manual"),
                            strategy_name: str = Form("")):
        if err := _require_trader(request): return err
        executor = getattr(app.state, "executor", None)
        risk_manager = getattr(app.state, "risk_manager", None)
        if not executor:
            return HTMLResponse(f'<span class="text-red-400">{_T("交易器未就绪")}</span>')

        # Engine cache first, then the configured public data host. Never a bare
        # `AsyncClient.create()` (that targets the unreachable mainnet, which is why
        # manual trades failed for anything outside the watched list).
        price_fn = getattr(app.state, "get_price", None)
        current_price = price_fn(symbol) if price_fn else None
        if current_price is None:
            try:
                from core.market_data.data_client import MarketDataClient
                data_client = getattr(app.state, "fee_price_client", None)
                if data_client is None:
                    data_client = MarketDataClient(getattr(config, "market_data_host", None))
                    app.state.fee_price_client = data_client
                ticker = await data_client.ticker24h(symbol)
                last = ticker.get("lastPrice") if isinstance(ticker, dict) else None
                current_price = float(last) if last else None
            except Exception as e:
                from loguru import logger
                logger.warning(f"Manual-trade price lookup failed for {symbol}: {e}")
        if not current_price or current_price <= 0:
            return HTMLResponse('<span class="text-red-400">无法获取实时价格，请检查网络或 API 配置</span>')

        qty = amount_usdt / current_price
        sl = current_price * (1 - stop_loss_pct / 100) if side == "long" else current_price * (1 + stop_loss_pct / 100)

        # Route through risk manager for all safety checks
        if risk_manager:
            signal = {
                "symbol": symbol, "side": side,
                "price": current_price, "quantity": round(qty, 6),
                "stop_loss": round(sl, 2),
                "position_type": position_type,
                "amount_usdt": amount_usdt,
                "trader": trader,
                "strategy_name": strategy_name,
                "strategy": "manual",
                "timeframe": DEFAULT_TIMEFRAME,
                "confidence": 1.0,
            }
            result = await risk_manager.check_signal(signal)
            if not result.approved:
                return HTMLResponse(f'<span class="text-red-400">风控拒绝: {result.reason}</span>')
            # Use the risk-adjusted quantity (capped at user's request)
            if result.adjusted_quantity is not None and result.adjusted_quantity < signal["quantity"]:
                signal["quantity"] = result.adjusted_quantity
                signal["amount_usdt"] = result.adjusted_quantity * current_price
            if result.adjusted_stop_loss is not None:
                signal["stop_loss"] = result.adjusted_stop_loss
            if result.adjusted_leverage is not None:
                signal["leverage"] = result.adjusted_leverage
            final_qty = signal["quantity"]

            await event_bus.publish(Event(EventType.ORDER_REQUEST, signal))
            resp = HTMLResponse(
                f'<span class="text-green-400">✓ {side.upper()} {final_qty:.6f} {symbol} @ {current_price:.4f} | SL: {signal["stop_loss"]:.4f}</span>'
            )
            resp.headers["HX-Trigger"] = "tradeUpdated"
            return resp
        else:
            # FAIL CLOSED: without a risk manager none of the 7 safety gates run,
            # so a manual trade must be refused rather than executed unchecked.
            # (Previously this branch placed the order directly, bypassing every
            # position-size, exposure, leverage and drawdown limit.)
            logger.error("Manual trade rejected: risk manager unavailable — refusing "
                         "to execute without risk checks")
            return HTMLResponse(
                '<span class="text-red-400">风控未就绪，已拒绝下单（不允许绕过风控）</span>')

    @app.get("/partials/stats", response_class=HTMLResponse)
    async def partial_stats():
        from db.database import load_sim_balance
        executor = getattr(app.state, "executor", None)
        positions = executor.get_open_positions() if executor else {}
        pos_count = len(positions)
        # Reload from DB so auto-trade balance changes are reflected
        balance = await load_sim_balance(config.db_path)
        app.state.balance = balance  # sync web state
        invested = sum(p.get("amount_usdt", p.get("quantity", 0) * p.get("entry_price", 0)) for p in positions.values())
        total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions.values())
        return _render("partials/account_summary.html", {
            "request": None, "balance": balance, "pos_count": pos_count,
            "max_trades": config.hard_limits.max_open_trades,
            "total_pnl": total_pnl, "invested": invested,
        })

    @app.get("/partials/positions", response_class=HTMLResponse)
    async def partial_positions():
        executor = getattr(app.state, "executor", None)
        positions_raw = executor.get_open_positions() if executor else {}
        if not positions_raw:
            return _render("partials/positions_table.html", {"request": None, "positions": []})
        # Fetch live prices for PnL
        price_fn = getattr(app.state, "get_price", None)
        pos_list = []
        for pos in positions_raw.values():
            live_price = price_fn(pos["symbol"]) if price_fn else None
            if live_price is None:
                # Never a bare `AsyncClient.create()` (that targets the unreachable
                # mainnet); use the configured public data host like the rest of the app.
                try:
                    from core.market_data.data_client import MarketDataClient
                    data_client = getattr(app.state, "fee_price_client", None)
                    if data_client is None:
                        data_client = MarketDataClient(getattr(config, "market_data_host", None))
                        app.state.fee_price_client = data_client
                    ticker = await data_client.ticker24h(pos["symbol"])
                    last = ticker.get("lastPrice") if isinstance(ticker, dict) else None
                    live_price = float(last) if last else pos["entry_price"]
                except Exception:
                    live_price = pos["entry_price"]
            pnl = (live_price - pos["entry_price"]) * pos["quantity"] if pos["side"] == "long" else (pos["entry_price"] - live_price) * pos["quantity"]
            pos["unrealized_pnl"] = pnl
            pos["current_price"] = live_price
            pos_list.append({
                "symbol": pos["symbol"], "side": pos.get("side", "long"),
                "quantity": pos["quantity"], "entry_price": pos["entry_price"],
                "live_price": live_price, "pnl": pnl,
                "position_type": pos.get("position_type", "satellite"),
            })
        if rest_client:
            try: await rest_client.close_connection()
            except Exception: pass
        return _render("partials/positions_table.html", {"request": None, "positions": pos_list})

    @app.post("/api/trade/close/{symbol}")
    async def close_position(symbol: str, request: Request, reduce_pct: float = Form(100)):
        if err := _require_trader(request): return err
        executor = getattr(app.state, "executor", None)
        if not executor:
            return HTMLResponse(f'<span class="text-red-400">{_T("交易器未就绪")}</span>')
        positions = executor.get_open_positions()
        if symbol not in positions:
            return HTMLResponse(f'<span class="text-red-400">{_T("无")} {symbol} {_T("持仓")}</span>')

        # Get current price — engine cache first, then the configured public data
        # host. The old fallback used a bare `AsyncClient.create()` (MAINNET, which is
        # unreachable from this host), so a position in a symbol the engine does not
        # stream could never be closed from the UI.
        current_price = None
        price_fn = getattr(app.state, "get_price", None)
        current_price = price_fn(symbol) if price_fn else None
        if current_price is None:
            try:
                from core.market_data.data_client import MarketDataClient
                data_client = getattr(app.state, "fee_price_client", None)
                if data_client is None:
                    data_client = MarketDataClient(getattr(config, "market_data_host", None))
                    app.state.fee_price_client = data_client
                ticker = await data_client.ticker24h(symbol)
                last = ticker.get("lastPrice") if isinstance(ticker, dict) else None
                current_price = float(last) if last else None
            except Exception as e:
                from loguru import logger
                logger.warning(f"Close-position price lookup failed for {symbol}: {e}")
        if not current_price:
            return HTMLResponse('<span class="text-red-400">无法获取实时价格</span>')

        reduce_pct = min(100, max(1, reduce_pct))

        # Delegate to executor for core close logic (DB persistence + position tracking)
        result = await executor.close_position(symbol, reduce_pct, current_price)
        if not result.get("ok"):
            return HTMLResponse(f'<span class="text-red-400">{result.get("error", "关闭失败")}</span>')

        # Atomically update balance (prevents RMW races with auto-trades)
        delta = result.get("invested_returned", 0) + result.get("pnl", 0)
        new_balance = await atomic_adjust_balance(delta, config.db_path)
        app.state.balance = new_balance
        # Sync to risk manager
        rm = getattr(app.state, "risk_manager", None)
        if rm:
            rm.update_balance(new_balance)

        if reduce_pct >= 100:
            msg = f'✓ 已平仓 {symbol} | P&L: {result["pnl"]:.4f} USDT ({result["pnl_pct"]:.4f}%)'
        else:
            msg = f'✓ 减持 {symbol} {reduce_pct:.0f}% | P&L: {result["pnl"]:.4f} USDT ({result["pnl_pct"]:.4f}%)'

        resp = HTMLResponse(f'<span class="text-green-400">{msg}</span>')
        resp.headers["HX-Trigger"] = "tradeUpdated"
        return resp
