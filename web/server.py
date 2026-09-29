"""FastAPI application factory for the Binance Trader web UI.

This module used to hold all 97 HTTP/WebSocket handlers as closures inside
``create_app()``.  It has been split by domain (file organization only — no
behaviour change):

    web/rendering.py            Jinja env + _render / _T / _get_lang / _fmt_time
    web/deps.py                 _require_trader / _require_admin / _save_balance
    web/context.py              AppContext (config, event bus, app, locks, caches)
    web/routes/health.py        GET  /health
    web/routes/auth.py          auth + login page
    web/routes/users.py         admin user management + user-list partial
    web/routes/pages.py         full-page HTML routes (/, /dashboard, ...)
    web/routes/dashboard_partials.py  account/trade partials + market APIs
    web/routes/alerts.py        alerts API/rules/partials
    web/routes/strategies.py    strategy CRUD + AI recommendation
    web/routes/trading.py       /api/trade, /api/trade/close, position partials
    web/routes/settings.py      settings writes, circuit breaker, restart
    web/routes/ai.py            AI suggestions, consult, models, heartbeat
    web/routes/backtest.py      backtest page + run/progress/result APIs
    web/routes/lifecycle.py     strategy lifecycle endpoints
    web/routes/ga.py            GA / walk-forward / calibration endpoints
    web/routes/db_manager.py    DB manager APIs
    web/routes/market.py        trade-page APIs (market/account/orders/history)
    web/ws/alerts.py            /ws/alerts WebSocket

``register(app, ctx)`` modules are called in the original route-registration
order so path matching and OpenAPI ordering are unchanged.
"""
import logging
from pathlib import Path

from fastapi import FastAPI

from app.event_bus import EventBus
from app.config import Config

from web.context import AppContext
# Re-exported for backwards compatibility with code that imported these
# helpers straight out of web.server.
from web.rendering import (  # noqa: F401
    CST,
    DEFAULT_BALANCE,
    _jinja_env,
    _templates_dir,
    _fmt_time,
    _get_lang,
    _render,
    _T,
)
from web.routes import (
    health as routes_health,
    auth as routes_auth,
    users as routes_users,
    pages as routes_pages,
    dashboard_partials as routes_dashboard_partials,
    alerts as routes_alerts,
    strategies as routes_strategies,
    trading as routes_trading,
    settings as routes_settings,
    ai as routes_ai,
    backtest as routes_backtest,
    lifecycle as routes_lifecycle,
    ga as routes_ga,
    db_manager as routes_db_manager,
    market as routes_market,
    audit as routes_audit,
)
from web.ws import alerts as ws_alerts

logger = logging.getLogger(__name__)


def create_app(config: Config, event_bus: EventBus, auth_manager=None) -> FastAPI:
    app = FastAPI(title="Binance Trader", docs_url=None, redoc_url=None)
    # Expose the bus on app.state too (the route modules still receive it through
    # AppContext, which is what they close over).
    app.state.event_bus = event_bus

    # Mount static files for favicon and other static assets
    from starlette.staticfiles import StaticFiles
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    if auth_manager:
        from starlette.middleware.base import BaseHTTPMiddleware
        app.add_middleware(auth_manager.create_middleware())
        app.state.auth_manager = auth_manager

    ctx = AppContext(config, event_bus, app)

    # Registration order mirrors the original top-to-bottom layout of this file.
    routes_health.register(app, ctx)
    routes_auth.register(app, ctx)
    routes_users.register(app, ctx)
    routes_pages.register(app, ctx)
    routes_dashboard_partials.register(app, ctx)
    routes_alerts.register(app, ctx)
    ws_alerts.register(app, ctx)
    routes_strategies.register(app, ctx)
    routes_trading.register(app, ctx)
    routes_settings.register(app, ctx)
    routes_ai.register(app, ctx)
    routes_backtest.register(app, ctx)
    routes_lifecycle.register(app, ctx)
    routes_ga.register(app, ctx)
    routes_db_manager.register(app, ctx)
    # Added last: the Binance-style trade page API (market/account/orders/history)
    # + the pending limit-order matcher.  Purely additive to the route table.
    routes_market.register(app, ctx)
    # 代币检测 (heuristic token screen over public mainnet market data).
    routes_audit.register(app, ctx)

    return app
