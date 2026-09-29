"""Full-page HTML routes: /, /dashboard, /strategies, /ai, /alerts, /settings,
/db-manager, /users, /backtest."""
import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from db.database import get_db

from core.market_data.universe import DEFAULT_WATCHLIST, load_watchlist

from web.rendering import _render, DEFAULT_BALANCE


#: `system_config` key written by `DeepSeekController._market_assessment_loop`.
MARKET_ASSESSMENT_KEY = "ai_market_assessment"


def _assessment_text(raw) -> str:
    """Render a stored market assessment into the card's plain text."""
    if raw is None:
        return ""
    text = str(raw)
    try:
        data = json.loads(text)
    except Exception:
        return text[:500]
    if isinstance(data, dict):
        for key in ("market_regime", "regime", "summary", "reason", "assessment"):
            if data.get(key):
                return str(data[key])[:500]
    return text[:500]


async def latest_market_assessment() -> tuple[str | None, str | None]:
    """``(text, created_at)`` for the newest real market assessment.

    The assessment used to be looked for in ``ai_suggestions`` under a category
    the table's CHECK constraint forbids, so this always returned nothing and the
    ``/ai`` card and ``/api/market-state`` both reported "waiting" even when the
    AI had produced a full assessment.  It now reads the dedicated
    ``system_config`` blob (and keeps a text fallback for older rows).
    """
    try:
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT value, updated_at FROM system_config WHERE key=?",
                (MARKET_ASSESSMENT_KEY,))
            row = await cursor.fetchone()
        finally:
            await db.close()
    except Exception:
        return None, None
    if not row:
        return None, None
    value = row["value"]
    updated = row["updated_at"]
    text = _assessment_text(value)
    return (text or None), (str(updated)[:16] if updated else None)



async def _page_symbols(config) -> list[str]:
    """Symbols a page should offer: the persisted watchlist (never a copy).

    ``core.market_data.universe`` owns the list; ``DEFAULT_WATCHLIST`` is only
    the fallback for a brand-new database.
    """
    try:
        return await load_watchlist(config.db_path, DEFAULT_WATCHLIST)
    except Exception:  # pragma: no cover - load_watchlist already guards
        return list(DEFAULT_WATCHLIST)


def _render_or_pending(template: str, context: dict) -> HTMLResponse:
    """Render a page, or show a friendly notice while its template is missing.

    New pages (行情/数据/代币检测) are added to the navigation as soon as their
    routes exist; a missing template used to surface as an opaque 500 during the
    window before the template landed.
    """
    try:
        return _render(template, context)
    except Exception as e:  # jinja2.TemplateNotFound or a template error
        from loguru import logger
        logger.warning(f"Page template '{template}' unavailable: {e}")
        return HTMLResponse(
            f"""<!doctype html><html><head><meta charset="utf-8">
            <title>页面准备中</title></head>
            <body style="background:#0b1220;color:#e2e8f0;font-family:system-ui;padding:40px">
            <h2 style="margin:0 0 8px">页面正在准备中</h2>
            <p style="color:#94a3b8">模板 <code>{template}</code> 尚未就绪，请稍后刷新。</p>
            <p><a href="/trade" style="color:#38bdf8">← 返回现货交易</a></p>
            </body></html>""",
            status_code=200,
        )


def register(app: FastAPI, ctx) -> None:
    config = ctx.config

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Retired: the dashboard duplicated the spot-trade page.

        Every panel it showed (chart, account, order form, positions, trades) now
        lives on /trade, which is the richer Binance-style page. Kept as a
        permanent redirect so old links/bookmarks keep working.
        """
        return RedirectResponse(url="/trade", status_code=302)

    def _page_user(request: Request):
        return getattr(request.state, "user", None)

    @app.get("/market", response_class=HTMLResponse)
    async def market_page(request: Request):
        """全币种行情 / 币种信息列表（数据来自主网公开行情镜像）。"""
        return _render_or_pending("market.html", {
            "request": request,
            "current_page": "market",
            "mode": config.mode,
            "can_trade": bool(getattr(_page_user(request), "is_trader", False)),
        })

    @app.get("/coin/{symbol}", response_class=HTMLResponse)
    async def coin_page(request: Request, symbol: str):
        """单币种详情：交易规则、24h、盘口、多周期表现、启发式风险评分。"""
        return _render_or_pending("coin.html", {
            "request": request,
            "current_page": "market",
            "mode": config.mode,
            "symbol": symbol.upper(),
            "can_trade": bool(getattr(_page_user(request), "is_trader", False)),
        })

    @app.get("/data", response_class=HTMLResponse)
    async def data_page(request: Request):
        """全市场数据总览：涨跌榜、成交额榜、活跃度、波动率、价差。"""
        return _render_or_pending("data.html", {
            "request": request,
            "current_page": "data",
            "mode": config.mode,
        })

    @app.get("/audit", response_class=HTMLResponse)
    async def audit_page(request: Request):
        """代币检测：基于交易所公开数据的启发式风险筛查（非链上合约审计）。"""
        return _render_or_pending("audit.html", {
            "request": request,
            "current_page": "audit",
            "mode": config.mode,
        })

    @app.get("/trade", response_class=HTMLResponse)
    async def trade_page(request: Request):
        """Binance-style spot trade page (see docs/overhaul/TRADE_PAGE_API.md §五).

        The page is a shell: every panel is filled client-side from the frozen
        /api/market/*, /api/account, /api/order(s) and /api/history/trades
        endpoints, so this handler only supplies the watched symbol list.
        """
        symbols = await _page_symbols(config)
        user = getattr(request.state, "user", None)
        return _render("trade.html", {
            "request": request,
            "current_page": "trade",
            "mode": config.mode,
            "symbols": symbols,
            "can_trade": bool(getattr(user, "is_trader", False)),
        })

    @app.get("/strategies", response_class=HTMLResponse)
    async def strategies_page(request: Request):
        loader = getattr(app.state, "strategy_loader", None)
        strategies = []
        if loader:
            strategies = [s.model_dump() for s in loader.load_all()]

        return _render("strategies.html", {
            "request": request,
            "current_page": "strategies",
            "mode": config.mode,
            # The symbol picker is filled from the persisted watchlist instead of
            # a hard-coded five-pair literal baked into the template.
            "symbols": await _page_symbols(config),
            "strategies": strategies,
            "signal_weights": config.signal_weights.model_dump(),
            "risk_params": config.soft_params.model_dump(),
            "hard_limits": config.hard_limits.model_dump(),
        })

    @app.get("/ai", response_class=HTMLResponse)
    async def ai_page(request: Request):
        pending = []
        lifecycle_events = []
        try:
            db = await get_db()
            cursor = await db.execute("SELECT * FROM ai_suggestions WHERE status='pending' ORDER BY created_at DESC")
            pending = [dict(r) for r in await cursor.fetchall()]
            await db.close()
        except Exception:
            pass
        try:
            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM strategy_lifecycle_events ORDER BY created_at DESC LIMIT 10")
            lifecycle_events = [dict(r) for r in await cursor.fetchall()]
            await db.close()
        except Exception:
            lifecycle_events = []

        # One renderer for both the first paint and the HTMX refresh, so the card
        # cannot lose confidence/body detail when the poller swaps it out.
        from web.routes.ai import render_suggestion_rows

        last_assessment, assessment_at = await latest_market_assessment()
        user = getattr(request.state, "user", None)
        return _render("ai_panel.html", {
            "request": request,
            "current_page": "ai",
            "mode": config.mode,
            "ai_mode": config.ai_mode,
            "api_connected": bool(config.deepseek_api_key),
            "ai_model": config.ai_model,
            "symbols": await _page_symbols(config),
            "pending_suggestions": pending,
            "suggestions_html": render_suggestion_rows(pending),
            "last_assessment": last_assessment,
            "assessment_at": assessment_at,
            "lifecycle_events": lifecycle_events,
            "can_trade": bool(getattr(user, "is_trader", False)),
        })

    @app.get("/alerts", response_class=HTMLResponse)
    async def alerts_page(request: Request):
        mgr = getattr(app.state, "alert_manager", None)
        alerts = await mgr.get_alerts(limit=50) if mgr else []
        rules = mgr.get_rules() if mgr else []
        counts = await mgr.get_counts() if mgr else {"critical": 0, "warning": 0, "info": 0}
        return _render("alerts.html", {
            "request": request,
            "current_page": "alerts",
            "mode": config.mode,
            "alerts": alerts,
            "rules": rules,
            "counts": counts,
            "rules_enabled": sum(1 for r in rules if r.get("enabled")),
        })

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_trader:
            return RedirectResponse(url="/trade", status_code=302)
        rm = getattr(app.state, "risk_manager", None)
        cb_state = {}
        if rm:
            cb = rm.breaker
            cb_state = {
                "is_tripped": cb.is_tripped,
                "trip_reason": cb.trip_reason,
                "daily_pnl": round(cb.daily_pnl, 2),
                "consecutive_losses": cb.consecutive_losses,
            }
        return _render("settings.html", {
            "request": request,
            "current_page": "settings",
            "mode": config.mode,
            "binance_testnet": config.binance_testnet,
            "web_port": config.web_port,
            # SECURITY: never render the live credentials into the DOM.  The
            # template only needs to know *whether* a secret is configured; the
            # save endpoints treat a blank field as "keep the current value"
            # (web/routes/settings.py), so an empty password input is safe.
            "has_binance_key": bool(config.binance_api_key),
            "has_binance_secret": bool(config.binance_api_secret),
            "has_deepseek_key": bool(config.deepseek_api_key),
            "ai_base_url": config.ai_base_url or "https://api.deepseek.com",
            "ai_model": config.ai_model or "deepseek-chat",
            "language": getattr(config, "language", "en"),
            "news_fetch_interval": config.news_fetch_interval,
            "max_articles": config.news_max_articles,
            "anomaly_threshold": config.anomaly_threshold_pct,
            "hard_limits": config.hard_limits,
            "ai_task_intervals": {k: v // 60 for k, v in config.ai_task_intervals.items()},
            "circuit_breaker": cb_state,
            "is_admin": bool(getattr(user, "is_admin", False)),
        })

    @app.get("/db-manager", response_class=HTMLResponse)
    async def db_manager_page(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return RedirectResponse(url="/trade", status_code=302)
        tables = ["trades", "positions", "pending_orders", "alerts", "ai_suggestions", "backtest_records", "strategy_lifecycle_events", "news_sources", "system_config", "users"]
        return _render("db_manager.html", {"request": request, "current_page": "db_manager", "tables": tables})

    @app.get("/users", response_class=HTMLResponse)
    async def users_page(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return RedirectResponse(url="/trade", status_code=302)
        return _render("users.html", {"request": request, "current_page": "users"})

    @app.get("/")
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/trade")
