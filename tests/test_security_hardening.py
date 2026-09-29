"""Regression tests for the reproduced SECURITY audit findings.

Each test below fails on the pre-fix code and passes after it:

  1. stored XSS in ``GET /partials/alerts`` (raw f-string HTML)
  2. client-side XSS in ``alerts.html`` (``prependAlert`` used ``innerHTML``)
  3. live API credentials rendered into ``GET /settings`` HTML
  4. unbounded upstream fan-out: freely-varied ``min_quote_volume`` cache key,
     uncached per-symbol detail, no rate limit on the audit/market reads
  5. hard-coded ``"1h"`` timeframes instead of the interval registry
  6. session hardening: cookie ``secure`` flag, logout revoking the JWT

Everything runs against a **temporary** SQLite database and a stubbed screener:
no network, and ``data/binance_trader.db`` / ``config/*.yaml`` are never touched.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from db.database import init_database

import db.database as database

VIEWER = ("sec_viewer", "V1ewerPass!")
ADMIN = ("sec_admin", "Adm1nPass!")
ROLES = {VIEWER[0]: "viewer", ADMIN[0]: "admin"}
XSS = '<img src=x onerror=alert(document.domain)>XSS'
FAKE_BINANCE_KEY = "AK_FAKE_BINANCE_KEY_0123456789"
FAKE_BINANCE_SECRET = "SK_FAKE_BINANCE_SECRET_9876543210"
FAKE_DEEPSEEK_KEY = "sk-FAKEDEEPSEEKKEY0123456789"


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def sec_app():
    """App on a temp DB with a temp config dir and a stubbed screener."""
    tmpdir = Path(tempfile.mkdtemp(prefix="bt_sechard_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "sec.db")
    config.config_dir = str(tmpdir / "config")
    config.data_dir = str(tmpdir / "data")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    (tmpdir / "data").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")

    async def _setup():
        await init_database(config.db_path)
        database.DB_PATH = config.db_path
        am = AuthManager(config.db_path, "sec-test-secret-at-least-32-bytes!!", 24)
        for username, password in (VIEWER, ADMIN):
            await am.create_user(username, password, ROLES[username], username)
        return am

    auth = _run(_setup())

    from core.market_data import screener as S
    from tests.test_screener import FakeExchange, make_screener
    import web.routes.audit as routes_audit

    exchange = FakeExchange(["AAAUSDT", "BBBUSDT", "CCCUSDT"],
                            depth_qty=(4000.0, 4000.0), spread_bps=0.2)
    exchange.per_symbol = {
        "AAAUSDT": {"quote_volume": 9e8, "count": 1_000_000, "age_days": 3000},
        "BBBUSDT": {"quote_volume": 2e5, "count": 800, "age_days": 12},
    }
    screener = make_screener(exchange, cache_ttl=300.0)
    S.reset_default_screener()
    original_get_screener = routes_audit.get_screener
    routes_audit.get_screener = lambda config=None: screener

    from web.server import create_app

    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    database.DB_PATH = config.db_path
    try:
        yield app, config, exchange, screener
    finally:
        routes_audit.get_screener = original_get_screener
        routes_audit.reset_rate_limits()
        S.reset_default_screener()


def _login(app, creds):
    client = TestClient(app)
    if creds:
        r = client.post("/api/auth/login",
                        json={"username": creds[0], "password": creds[1]})
        assert r.status_code == 200, f"login failed for {creds[0]}: {r.text}"
    return client


def _seed_alert(db_path, message=XSS, level="critical", alert_type="news"):
    async def _insert():
        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO alerts (level, type, message, symbol) VALUES (?,?,?,?)",
                (level, alert_type, message, message))
            await db.commit()

    _run(_insert())


# ── 1. stored XSS: GET /partials/alerts ─────────────────────────────────

def test_partial_alerts_escapes_the_stored_message(sec_app):
    app, config, _exchange, _screener = sec_app
    _seed_alert(config.db_path)
    client = _login(app, VIEWER)
    r = client.get("/partials/alerts", params={"limit": 5})
    assert r.status_code == 200, r.text
    assert XSS not in r.text, "the raw payload must never reach the HTML sink"
    assert "&lt;img" in r.text, "the payload must be entity-escaped"
    assert "<img" not in r.text


def test_alerts_page_escapes_websocket_pushed_fields(sec_app):
    """`prependAlert` must route every pushed field through an esc() helper."""
    app, *_ = sec_app
    html = _login(app, VIEWER).get("/alerts").text
    assert "function esc(" in html, "alerts.html needs the shared esc() helper"
    for field in ("data.message", "data.rule", "data.symbol", "data.timestamp"):
        assert f"esc({field}" in html or f"esc(({field}" in html, \
            f"{field} is still concatenated into innerHTML unescaped"


# ── 2. credentials must not reach the settings page ─────────────────────

def test_settings_page_never_renders_live_credentials(sec_app, monkeypatch):
    app, config, *_ = sec_app
    monkeypatch.setattr(config, "binance_api_key", FAKE_BINANCE_KEY, raising=False)
    monkeypatch.setattr(config, "binance_api_secret", FAKE_BINANCE_SECRET, raising=False)
    monkeypatch.setattr(config, "deepseek_api_key", FAKE_DEEPSEEK_KEY, raising=False)
    r = _login(app, ADMIN).get("/settings")
    assert r.status_code == 200, r.text
    for secret in (FAKE_BINANCE_KEY, FAKE_BINANCE_SECRET, FAKE_DEEPSEEK_KEY):
        assert secret not in r.text, f"secret leaked into /settings HTML: {secret}"
    # The form still exists, as masked inputs with a configured/not hint.
    assert 'type="password" name="api_key"' in r.text
    assert 'type="password" name="api_secret"' in r.text
    assert "已配置" in r.text or "未配置" in r.text


def test_blank_settings_fields_keep_the_stored_secrets(sec_app, monkeypatch):
    """Empty input = "leave unchanged"; the /settings form now always posts blanks."""
    app, config, *_ = sec_app
    monkeypatch.setattr(config, "binance_api_key", FAKE_BINANCE_KEY, raising=False)
    monkeypatch.setattr(config, "binance_api_secret", FAKE_BINANCE_SECRET, raising=False)
    monkeypatch.setattr(config, "deepseek_api_key", FAKE_DEEPSEEK_KEY, raising=False)
    client = _login(app, ADMIN)

    r = client.post("/api/settings/binance",
                    data={"api_key": "", "api_secret": "", "testnet": "1"})
    assert r.status_code == 200, r.text
    assert config.binance_api_key == FAKE_BINANCE_KEY
    assert config.binance_api_secret == FAKE_BINANCE_SECRET

    r = client.post("/api/settings/deepseek",
                    data={"api_key": "", "base_url": "", "model": ""})
    assert r.status_code == 200, r.text
    assert config.deepseek_api_key == FAKE_DEEPSEEK_KEY

    # A non-empty value still replaces the stored one.
    r = client.post("/api/settings/binance",
                    data={"api_key": "AK_NEW", "api_secret": "SK_NEW", "testnet": "1"})
    assert r.status_code == 200, r.text
    assert config.binance_api_key == "AK_NEW"
    assert config.binance_api_secret == "SK_NEW"


# ── 3. unbounded upstream fan-out ───────────────────────────────────────

def test_min_quote_volume_is_bucketed_and_cannot_mint_crawls():
    """Eight near-identical thresholds must produce exactly one crawl."""
    from core.market_data import screener as S
    from tests.test_screener import FakeExchange, make_screener

    exchange = FakeExchange(["AAAUSDT", "BBBUSDT"], depth_qty=(4000.0, 4000.0))
    screener = make_screener(exchange, cache_ttl=300.0)
    for value in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
        payload = _run(screener.screen(limit=3, min_quote_volume=value))
        assert payload["cached"] is (value != 1.0)
    assert screener.stats["screens"] == 1, "varying the threshold re-ran the crawl"

    # Floor-snapping keeps every documented preset exact and never widens the filter.
    assert S.normalize_min_quote_volume(1.0) == 0.0
    assert S.normalize_min_quote_volume(150_000) == 100_000.0
    assert S.normalize_min_quote_volume(999_999) == 100_000.0
    assert S.normalize_min_quote_volume(1_000_000) == 1_000_000.0
    assert S.normalize_min_quote_volume(1e12) == 100_000_000.0
    assert S.normalize_min_quote_volume(-5) == 0.0
    assert S.normalize_min_quote_volume("nonsense") == S.DEFAULT_MIN_QUOTE_VOLUME


def test_audit_screen_endpoint_reuses_one_crawl_across_buckets(sec_app):
    app, _config, exchange, _screener = sec_app
    client = _login(app, VIEWER)
    first = client.get("/api/audit/screen",
                       params={"limit": 7, "min_quote_volume": 100_001})
    assert first.status_code == 200, first.text
    assert first.json()["params"]["min_quote_volume"] == 100_000.0, \
        "the response must echo the bucket actually used"
    calls_after_first = len(exchange.calls)
    for value in (100_002, 100_500, 199_999):
        r = client.get("/api/audit/screen",
                       params={"limit": 7, "min_quote_volume": value})
        assert r.status_code == 200, r.text
        assert r.json()["cached"] is True
    assert len(exchange.calls) == calls_after_first, \
        "bucketed thresholds must reuse the cached screen"


def test_audit_symbol_detail_is_cached(sec_app, monkeypatch):
    app, _config, _exchange, _screener = sec_app
    from tests.test_screener import FakeExchange, make_screener
    import web.routes.audit as routes_audit

    fresh_exchange = FakeExchange(["AAAUSDT"], depth_qty=(4000.0, 4000.0))
    fresh = make_screener(fresh_exchange, cache_ttl=300.0)
    monkeypatch.setattr(routes_audit, "get_screener", lambda config=None: fresh)

    client = _login(app, VIEWER)
    first = client.get("/api/audit/AAAUSDT")
    assert first.status_code == 200, first.text
    calls_after_first = len(fresh_exchange.calls)
    assert calls_after_first > 0

    second = client.get("/api/audit/AAAUSDT")
    assert second.status_code == 200, second.text
    assert len(fresh_exchange.calls) == calls_after_first, \
        "a repeated per-symbol audit must be served from the cache"
    assert first.json() == second.json()


def test_audit_rate_limit_returns_429_after_n_calls(sec_app):
    app, *_ = sec_app
    import web.routes.audit as routes_audit

    routes_audit.reset_rate_limits()
    budget = routes_audit.RATE_LIMITS["audit"]
    assert 0 < budget <= 120, "the audit budget must stay small enough to matter"
    client = _login(app, VIEWER)
    codes = [client.get("/api/audit/screen", params={"limit": 2}).status_code
             for _ in range(budget + 1)]
    assert codes[:budget] == [200] * budget, codes
    assert codes[budget] == 429, f"expected a 429 after {budget} calls, got {codes[budget]}"
    assert "error" in client.get("/api/audit/screen",
                                 params={"limit": 2}).json()
    routes_audit.reset_rate_limits()


def test_market_reads_share_the_rate_limiter(sec_app, monkeypatch):
    app, *_ = sec_app
    import web.routes.audit as routes_audit

    assert "market" in routes_audit.RATE_LIMITS, \
        "the fan-out market endpoints need their own budget"
    routes_audit.reset_rate_limits()
    monkeypatch.setitem(routes_audit.RATE_LIMITS, "market", 0)
    client = _login(app, VIEWER)
    # Budget 0 -> the guard fires before any upstream work, so this is
    # network-free and proves the check is wired into the handler.
    r = client.get("/api/market/ticker", params={"symbol": "BTCUSDT"})
    assert r.status_code == 429, r.text
    assert "error" in r.json()
    routes_audit.reset_rate_limits()


def test_rate_limiter_is_per_session_and_slides():
    from starlette.requests import Request
    import web.routes.audit as routes_audit

    def _req(session, ip="10.0.0.1"):
        headers = [(b"cookie", f"bt_session={session}".encode())] if session else []
        return Request({"type": "http", "headers": headers, "client": (ip, 1234),
                        "path": "/api/audit/screen", "method": "GET"})

    routes_audit.reset_rate_limits()
    try:
        for _ in range(3):
            assert routes_audit.check_rate_limit(_req("s1"), "audit", limit=3) is None
        limited = routes_audit.check_rate_limit(_req("s1"), "audit", limit=3)
        assert limited is not None and limited.status_code == 429
        # A different session (or a cookie-less caller from another IP) is unaffected.
        assert routes_audit.check_rate_limit(_req("s2"), "audit", limit=3) is None
        assert routes_audit.check_rate_limit(_req(None, ip="10.0.0.9"),
                                             "audit", limit=3) is None
        # The window slides: after it expires the caller is served again.
        window = routes_audit.RATE_WINDOW_SEC
        routes_audit._rate_hits[("audit", "session:s1")] = [0.0]
        routes_audit.RATE_WINDOW_SEC = 0.001
        try:
            assert routes_audit.check_rate_limit(_req("s1"), "audit", limit=3) is None
        finally:
            routes_audit.RATE_WINDOW_SEC = window
    finally:
        routes_audit.reset_rate_limits()


# ── 4. hard-coded timeframes ────────────────────────────────────────────

def test_timeframe_defaults_come_from_the_registry():
    from core.market_data.provider import DEFAULT_TIMEFRAME
    from pathlib import Path as _P

    repo = _P(__file__).resolve().parents[1]
    market_src = (repo / "web" / "routes" / "market.py").read_text(encoding="utf-8")
    trading_src = (repo / "web" / "routes" / "trading.py").read_text(encoding="utf-8")

    assert "DEFAULT_TIMEFRAME" in market_src and "DEFAULT_TIMEFRAME" in trading_src
    assert 'timeframe="1h"' not in market_src
    assert 'or "1h"' not in market_src
    assert '"timeframe": "1h"' not in trading_src
    assert DEFAULT_TIMEFRAME, "the registry must expose a default timeframe"


# ── 5. session hardening ────────────────────────────────────────────────

def test_session_cookie_flags(sec_app, monkeypatch):
    app, config, *_ = sec_app
    client = TestClient(app)
    r = client.post("/api/auth/login",
                    json={"username": VIEWER[0], "password": VIEWER[1]})
    assert r.status_code == 200, r.text
    cookie = r.headers.get("set-cookie", "")
    assert "HttpOnly" in cookie, cookie
    assert "SameSite=lax" in cookie, cookie
    # Plain-HTTP localhost default: not Secure (documented), but configurable.
    assert "Secure" not in cookie, cookie

    monkeypatch.setattr(config, "session_cookie_secure", True, raising=False)
    r2 = TestClient(app).post("/api/auth/login",
                               json={"username": VIEWER[0], "password": VIEWER[1]})
    assert r2.status_code == 200, r2.text
    assert "Secure" in r2.headers.get("set-cookie", ""), \
        "session_cookie_secure=True must set the Secure flag"


def test_logout_revokes_the_jwt_and_the_session(sec_app):
    app, *_ = sec_app
    client = TestClient(app)
    r = client.post("/api/auth/login",
                    json={"username": ADMIN[0], "password": ADMIN[1]})
    assert r.status_code == 200, r.text
    token = r.json()["token"]

    bearer = {"Authorization": f"Bearer {token}"}
    assert TestClient(app).get("/api/users", headers=bearer).status_code == 200

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200, logout.text

    after = TestClient(app).get("/api/users", headers=bearer)
    assert after.status_code == 401, \
        "the token issued at login must be dead after logout"
    assert client.get("/api/users").status_code == 401


def test_jwt_minted_without_a_session_still_verifies():
    """Backwards compatibility for scripts that hold a standalone token."""
    from core.auth.auth import AuthManager, User

    am = AuthManager(":memory:", "verify-secret-at-least-32-bytes-long!!", 24)
    user = User(id=1, username="u", role="viewer", display_name="u", enabled=True)
    plain = am.create_jwt(user)
    assert am.verify_jwt(plain)["user_id"] == 1
    assert am.revoke_jwt(plain) is True
    assert am.verify_jwt(plain) is None
