"""Dashboard/account HTMX partials + market-data read APIs."""
import html

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from db.database import get_db

from web.rendering import _render, _fmt_time, _T

#: Single source of truth for "what did the last AI market assessment say".
from web.routes.pages import latest_market_assessment, _assessment_text


def register(app: FastAPI, ctx) -> None:
    config = ctx.config

    @app.get("/api/trades")
    async def get_trades(limit: int = 30):
        try:
            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in await cursor.fetchall()]
            await db.close()
            return rows
        except Exception:
            return []

    @app.get("/partials/trades", response_class=HTMLResponse)
    async def partial_trades(page: int = 1, per_page: int = 20):
        try:
            db = await get_db()
            cursor = await db.execute("SELECT COUNT(*) as cnt FROM trades")
            total = (await cursor.fetchone())["cnt"]
            offset = (page - 1) * per_page
            cursor = await db.execute(
                "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ? OFFSET ?",
                (per_page, offset))
            rows = [dict(r) for r in await cursor.fetchall()]
            await db.close()
        except Exception:
            rows = []
            total = 0

        total_pages = max(1, (total + per_page - 1) // per_page)

        # Build trade list for template
        trade_list = []
        for t in rows:
            action = t.get('action', 'open')
            rpct = float(t.get('reduce_pct', 0) or 0)
            trade_list.append({
                "symbol": t["symbol"], "side": t["side"],
                "quantity": t["quantity"], "entry_price": t["entry_price"],
                "exit_price": t.get("exit_price"),
                "pnl": t.get("pnl") or 0,
                "trader": t.get("trader", "manual"),
                "strategy": t.get("strategy", ""),
                "strategy_name": t.get("strategy_name", ""),
                "action": action, "reduce_pct": rpct,
                "status": t.get("status", "open"),
                "opened_at": _fmt_time(t.get("opened_at")),
            })

        return _render("partials/trade_history.html", {
            "request": None, "trades": trade_list,
            "page": page, "total_pages": total_pages, "total": total,
            "per_page": per_page,
        })

    @app.get("/api/market-state")
    async def get_market_state():
        """Latest AI market assessment as the `#market-assessment` card's body.

        Three bugs met here: the query read `ai_suggestions WHERE
        category='market_assessment'`, a category the table's CHECK constraint
        forbids (so no row could ever match); the endpoint therefore answered
        "waiting" forever; and because it answered with bare JSON, every 120s
        poll replaced the nicely server-rendered assessment with
        `{"regime": "waiting", ...}`.  It now reads the dedicated store
        (`system_config.ai_market_assessment`, written by
        `DeepSeekController._market_assessment_loop`), then the live controller's
        in-memory copy, and renders the same fragment the page renders — so a
        good assessment survives its own refresh and "waiting" appears only when
        there genuinely is none.
        """
        text, updated = await latest_market_assessment()
        if not text:
            ctl = getattr(app.state, "ai_controller", None)
            cached = getattr(ctl, "_last_market_assessment", None) if ctl else None
            if cached:
                text = _assessment_text(cached)
        if not text:
            body = f'<div class="text-sm text-slate-500">{_T("等待")} AI {_T("首次市场评估")}</div>'
        else:
            body = (f'<div class="text-sm text-slate-200 leading-relaxed '
                    f'max-h-32 overflow-y-auto">{html.escape(text)}</div>')
            if updated:
                body += f'<div class="text-xs text-slate-600 mt-1">{html.escape(str(updated))}</div>'
        return HTMLResponse(
            f'<h3 class="text-lg mb-2">{_T("最新市场评估")}</h3>{body}')

    @app.get("/api/strategy-monitor")
    async def get_strategy_monitor():
        engine = getattr(app.state, "strategy_engine", None)
        if not engine:
            return {"strategies": [], "active_count": 0, "total_count": 0}
        return engine.get_monitor_state()

    async def _configured_client():
        """Binance client built from config (testnet flag + keys).

        `/api/price` used to call `AsyncClient.create()` with no arguments, which
        targets MAINNET. On a testnet-configured deployment (or a network where
        mainnet is unreachable) price lookups timed out even though the trading
        engine itself was receiving data fine from testnet.

        (`/api/kline` had a second, near-identical copy of this helper plus a whole
        duplicate route; both were deleted — `web/routes/market.py` owns klines.)
        """
        from binance import AsyncClient
        return await AsyncClient.create(
            api_key=config.binance_api_key or None,
            api_secret=config.binance_api_secret or None,
            testnet=getattr(config, "binance_testnet", True),
        )

    @app.get("/api/price/{symbol}")
    async def get_price(symbol: str):
        price_fn = getattr(app.state, "get_price", None)
        price = price_fn(symbol) if price_fn else None
        if price is None:
            client = None
            try:
                client = await _configured_client()
                ticker = await client.get_symbol_ticker(symbol=symbol)
                price = float(ticker["price"])
            except Exception as e:
                return {"error": str(e)}
            finally:
                if client is not None:
                    try:
                        await client.close_connection()
                    except Exception:
                        pass
        return {"symbol": symbol, "price": price}
