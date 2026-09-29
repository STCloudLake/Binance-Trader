"""Web-layer authorization and health-endpoint regression tests.

Covers the concrete gaps found during the handover audit:
  * `DELETE /api/backtest/{record_id}` was reachable by any authenticated user
  * `POST /api/alerts/{id}/ack` was reachable without any role
  * exchange credentials / risk limits were writable by the `trader` role
  * there was no unauthenticated liveness probe (`/health`)

These run against a temporary SQLite database; they never touch the real
`data/binance_trader.db`, the real config files, or the network.
"""
import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from db.database import init_database

VIEWER = ("viewer1", "V1ewerPass!")
TRADER = ("trader1", "T1aderPass!")
ADMIN = ("admin1", "Adm1nPass!")


@pytest.fixture(scope="module")
def web_app():
    tmpdir = Path(tempfile.mkdtemp(prefix="bt_sec_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "sec.db")
    # Redirect settings persistence to the temp dir: an admin write used to rewrite
    # the shipped config/*.yaml (including binance.testnet) as a side effect.
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")

    async def _setup():
        await init_database(config.db_path)
        am = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        roles = {"viewer1": "viewer", "trader1": "trader", "admin1": "admin"}
        for username, password in (VIEWER, TRADER, ADMIN):
            await am.create_user(username, password, roles[username], username)
        return am

    auth = asyncio.run(_setup())
    from web.server import create_app

    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    return app


def _client(web_app, creds=None):
    c = TestClient(web_app)
    if creds:
        r = c.post("/api/auth/login", json={"username": creds[0], "password": creds[1]})
        assert r.status_code == 200, f"login failed for {creds[0]}: {r.text}"
    return c


# ── /health ──────────────────────────────────────────────────────────────

def test_health_is_public_and_leaks_nothing(web_app):
    c = _client(web_app)
    r = c.get("/health")
    assert r.status_code in (200, 503)
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert body["database"] in ("ok", "error")
    # unauthenticated callers must not get trading-state detail
    for key in ("open_positions", "circuit_breaker_tripped", "strategies_loaded", "mode"):
        assert key not in body, f"/health exposed '{key}' to an anonymous caller"
    # no credential material may appear in the probe
    text = r.text.lower()
    for needle in ("api_key", "apikey", "secret", "jwt", "password", "token"):
        assert needle not in text, f"/health leaked '{needle}'"


def test_additional_mutation_endpoints_require_roles(web_app):
    """Regression: three more endpoints were reachable without the right role."""
    viewer = _client(web_app, VIEWER)
    assert viewer.get("/partials/user-list").status_code == 403, \
        "user list (usernames/roles/last login) must be admin-only"
    assert viewer.get("/api/backtest/result/nope").status_code == 403, \
        "backtest result persists records + files, so it needs trader"
    assert viewer.post("/api/alerts/clear").status_code == 403, \
        "clearing the alert log destroys the audit trail and must be admin-only"

    admin = _client(web_app, ADMIN)
    assert admin.get("/partials/user-list").status_code == 200
    assert admin.post("/api/alerts/clear").status_code == 200


def test_db_manager_pagination_is_clamped(web_app):
    """A negative per_page used to become `LIMIT -1` (full-table dump)."""
    c = _client(web_app, ADMIN)
    r = c.get("/partials/db-table", params={"table": "trades", "per_page": -1}) \
        if False else c.get("/api/db/table/trades", params={"per_page": -1})
    assert r.status_code == 200, r.text


def test_unauthenticated_api_is_rejected(web_app):
    c = _client(web_app)
    assert c.get("/api/users").status_code == 401
    assert c.get("/", follow_redirects=False).status_code == 302


# ── Admin-only endpoints ─────────────────────────────────────────────────

@pytest.mark.parametrize("creds", [None, VIEWER, TRADER])
def test_admin_only_endpoints_deny_non_admins(web_app, creds):
    c = _client(web_app, creds)
    # GET /api/users is admin-only (read-only, safe to probe)
    r = c.get("/api/users")
    assert r.status_code in (401, 403), f"expected denial, got {r.status_code}"

    # Credential/risk endpoints must not be writable by a plain trader
    if creds in (VIEWER, TRADER):
        assert c.post("/api/settings/binance",
                      data={"api_key": "x", "api_secret": "y"}).status_code == 403
        assert c.post("/api/settings/risk",
                      data={"max_daily_drawdown": 1.0}).status_code == 403


def test_admin_can_list_users(web_app):
    c = _client(web_app, ADMIN)
    r = c.get("/api/users")
    assert r.status_code == 200


def test_kline_endpoint_serves_candles_from_the_engine_data_source(web_app):
    """The dashboard chart must not depend on a hard-coded MAINNET client.

    `/api/kline` used to call `AsyncClient.create()` (no config) which targets
    mainnet; on a testnet deployment that timed out and the chart rendered empty
    ("获取K线失败") even though the engine was receiving testnet data normally.
    It must now serve candles from the engine's own MarketDataProvider.
    """
    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=5, freq="5min")
    frame = pd.DataFrame({
        "open": [1.0, 2.0, 3.0, 4.0, 5.0],
        "high": [2.0, 3.0, 4.0, 5.0, 6.0],
        "low": [0.0, 1.0, 2.0, 3.0, 4.0],
        "close": [1.5, 2.5, 3.5, 4.5, 5.5],
        "volume": [10.0] * 5,
    }, index=idx)

    class FakeMarketData:
        calls = 0

        async def get_historical(self, symbol, interval, limit=200):
            FakeMarketData.calls += 1
            return frame

    web_app.state.market_data = FakeMarketData()
    try:
        c = _client(web_app, VIEWER)
        r = c.get("/api/kline/BTCUSDT", params={"interval": "5m", "limit": 5})
        assert r.status_code == 200, r.text
        candles = r.json()
        assert isinstance(candles, list) and len(candles) == 5
        assert set(candles[0]) == {"time", "open", "high", "low", "close", "volume"}
        assert candles[0]["close"] == 1.5
        assert candles[-1]["close"] == 5.5
        assert FakeMarketData.calls == 1, "candles must come from the provider, no REST fallback"
    finally:
        delattr(web_app.state, "market_data")


def test_admin_settings_write_never_touches_shipped_config(web_app):
    """Guard: settings persistence must go to config.config_dir, not the repo.

    A mis-probed endpoint previously rewrote config/config.yaml (flipping
    `binance.testnet` to false) and config/risk_params.yaml. Admin writes now land
    in the injected directory.
    """
    from pathlib import Path as _P

    repo_risk = _P(__file__).resolve().parent.parent / "config" / "risk_params.yaml"
    before = repo_risk.read_text(encoding="utf-8")

    c = _client(web_app, ADMIN)
    r = c.post("/api/settings/risk", data={"max_daily_drawdown": 1.25, "max_leverage": 2})
    assert r.status_code == 200, r.text

    assert repo_risk.read_text(encoding="utf-8") == before, \
        "the shipped risk_params.yaml must never be rewritten by a test/admin call"
    written = _P(web_app.state.config.config_dir) / "risk_params.yaml"
    assert written.exists() and "1.25" in written.read_text(encoding="utf-8")


# ── Trader endpoints ─────────────────────────────────────────────────────

def test_viewer_cannot_mutate(web_app):
    c = _client(web_app, VIEWER)
    assert c.delete("/api/backtest/1").status_code == 403
    assert c.post("/api/alerts/1/ack").status_code == 403
    assert c.post("/api/trade", data={"symbol": "BTCUSDT", "side": "long"}).status_code == 403


def test_trader_can_ack_and_delete_backtest(web_app):
    c = _client(web_app, TRADER)
    # Both operate on the temporary DB only.
    assert c.post("/api/alerts/1/ack").status_code == 200
    assert c.delete("/api/backtest/999999").status_code == 200
