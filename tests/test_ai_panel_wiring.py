"""Wiring tests for the AI panel: suggestion producer, market state, kline owner.

These cover the audit findings that made the AI surfaces permanently empty:

* `ai_suggestions` was never INSERTed (`_publish_suggestion` only published an
  event whose sole subscriber was a no-op class nobody instantiated), so the
  `/ai` card, the approve/reject buttons and the heartbeat's daily count could
  never show anything.
* `GET /api/market-state` read a category the table's CHECK constraint forbids,
  so it answered `{"regime": "waiting"}` forever and overwrote the good
  server-rendered assessment on `/ai` every 120s.
* `GET /api/kline/{symbol}` was registered twice; `web/routes/market.py` deleted
  the other copy at import time, making `dashboard_partials.py`'s version and its
  `_market_data_provider` dead code.
* `POST /api/alerts/{id}/ack` existed with no button anywhere in the UI.
* `ai_panel.html` duplicated `trade.html`'s strategy-monitor poller and carried a
  duplicate `class` attribute on the consult symbol `<select>`.

Everything runs against a throw-away SQLite file; no network, no real database.
"""
import asyncio
import json
import re
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


async def _scalar(db_path: str, sql: str, args=()):
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(sql, args)
        row = await cursor.fetchone()
        return row[0] if row else None


async def _rows(db_path: str, sql: str, args=()):
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(sql, args)
        return [dict(r) for r in await cursor.fetchall()]


@pytest.fixture(scope="module")
def web_app():
    tmpdir = Path(tempfile.mkdtemp(prefix="bt_aiwiring_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "wiring.db")
    config.deepseek_api_key = "test-key-not-used"

    async def _setup():
        await init_database(config.db_path)
        am = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        await am.create_user(VIEWER[0], VIEWER[1], "viewer", VIEWER[0])
        await am.create_user(TRADER[0], TRADER[1], "trader", TRADER[0])
        return am

    auth = asyncio.run(_setup())
    from web.server import create_app

    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    # Real manager over the same temp DB: `/partials/alerts-filtered`, the ack
    # route and `/alerts` all read `app.state.alert_manager`.
    from alerts.manager import AlertManager

    manager = AlertManager(config, app.state.event_bus)
    asyncio.run(manager.start())
    app.state.alert_manager = manager
    return app


def _client(web_app, creds=None):
    c = TestClient(web_app)
    if creds:
        r = c.post("/api/auth/login", json={"username": creds[0], "password": creds[1]})
        assert r.status_code == 200, f"login failed for {creds[0]}: {r.text}"
    return c


@pytest.fixture()
def producer(web_app):
    """A DeepSeekController over the temp DB (no client, so no network).

    Pinned to `suggest` mode: the shipped sim config runs `full_auto`, which
    marks suggestions `applied` because the loops really do apply them.
    """
    from core.ai.deepseek_ctl import DeepSeekController

    previous = web_app.state.config.ai_mode
    web_app.state.config.ai_mode = "suggest"
    ctl = DeepSeekController(web_app.state.config, web_app.state.event_bus)
    yield ctl
    web_app.state.config.ai_mode = previous
    asyncio.run(_wipe_suggestions(web_app.state.config.db_path))


async def _wipe_suggestions(db_path: str):
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM ai_suggestions")
        await db.commit()


def _count(web_app) -> int:
    return asyncio.run(_scalar(web_app.state.config.db_path,
                               "SELECT COUNT(*) FROM ai_suggestions"))


# ── 1. the producer INSERTs ─────────────────────────────────────────────

def test_publish_suggestion_inserts_a_populated_row(web_app, producer):
    assert _count(web_app) == 0, "temp DB starts clean"

    payload = json.dumps({"symbols": ["ADAUSDT", "DOGEUSDT"]})
    row_id = asyncio.run(producer._publish_suggestion("coin_selection", payload, 0.7))

    assert _count(web_app) == 1, "one row must land in ai_suggestions per suggestion"
    rows = asyncio.run(_rows(web_app.state.config.db_path,
                             "SELECT * FROM ai_suggestions WHERE id=?", (row_id,)))
    row = rows[0]
    assert row["category"] == "coin_selection"
    assert row["content"] == payload, "the full AI payload is preserved"
    assert row["confidence"] == pytest.approx(0.7)
    assert row["status"] == "pending"
    assert row["rationale"], "audit provenance must be filled in"
    assert row["created_at"], "created_at must be set by the DB default"


def test_publish_suggestion_keeps_publishing_the_event(web_app, producer):
    """Wiring the table must not silently drop the event contract."""
    from app.event_bus import EventType

    seen = []

    class _Bus:
        async def publish(self, event):
            seen.append(event)

    producer.event_bus = _Bus()
    row_id = asyncio.run(
        producer._publish_suggestion("coin_selection", '{"symbols": ["XRPUSDT"]}', 0.5))

    assert [e.type for e in seen] == [EventType.AI_SUGGESTION]
    assert seen[0].data["content"] == '{"symbols": ["XRPUSDT"]}'
    assert row_id is not None
    assert seen[0].data["id"] == row_id, "event carries the persisted row id"


def test_publish_suggestion_survives_a_broken_database(web_app, producer):
    """A DB failure is logged, never raised into the AI loop."""
    good_path = producer.config.db_path
    producer.config.db_path = "Z:/definitely/not/a/directory/x.db"
    try:
        assert asyncio.run(producer._publish_suggestion("coin_selection", "{}", 0.5)) is None
    finally:
        producer.config.db_path = good_path


def test_full_auto_suggestions_are_recorded_as_already_applied(web_app, producer):
    """full_auto executes the change, so the row must not sit in `pending`."""
    producer.config.ai_mode = "full_auto"
    row_id = asyncio.run(producer._publish_suggestion(
        "risk_adjustment", json.dumps({"leverage": 2}), 0.7))

    assert asyncio.run(_scalar(web_app.state.config.db_path,
                               "SELECT status FROM ai_suggestions WHERE id=?",
                               (row_id,))) == "applied"


# ── 2. dedupe ───────────────────────────────────────────────────────────

def test_repeated_identical_suggestion_is_deduped(web_app, producer):
    payload = json.dumps({"position_size_pct": 7.5})
    first = asyncio.run(producer._publish_suggestion("risk_adjustment", payload, 0.7))
    second = asyncio.run(producer._publish_suggestion("risk_adjustment", payload, 0.7))

    assert first is not None and second is None, "the repeat must not be re-inserted"
    assert _count(web_app) == 1


def test_dedupe_is_per_category_and_content(web_app, producer):
    a = json.dumps({"symbols": ["ADAUSDT"]})
    b = json.dumps({"symbols": ["DOGEUSDT"]})
    asyncio.run(producer._publish_suggestion("coin_selection", a, 0.7))
    asyncio.run(producer._publish_suggestion("coin_selection", b, 0.7))
    asyncio.run(producer._publish_suggestion("risk_adjustment", a, 0.7))

    assert _count(web_app) == 3, "different content or category is a new suggestion"


def test_dedupe_window_expires(web_app, producer):
    payload = json.dumps({"symbols": ["ADAUSDT"]})
    asyncio.run(producer._publish_suggestion("coin_selection", payload, 0.7))
    # Age the existing row past the window; the next identical run must insert.
    asyncio.run(_backdate_suggestions(web_app.state.config.db_path,
                                      producer.suggestion_dedupe_window + 60))
    row_id = asyncio.run(producer._publish_suggestion("coin_selection", payload, 0.7))

    assert row_id is not None, "an old identical suggestion must not block a fresh one"
    assert _count(web_app) == 2


async def _backdate_suggestions(db_path: str, seconds: float):
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE ai_suggestions SET created_at = datetime('now', ?)",
            (f"-{int(seconds)} seconds",))
        await db.commit()


# ── 3. approve / reject are reachable and change state ──────────────────

def _seed_pending(web_app, category: str, content: str, confidence: float = 0.6) -> int:
    import aiosqlite

    async def _insert():
        async with aiosqlite.connect(web_app.state.config.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO ai_suggestions (category, content, confidence, status) "
                "VALUES (?, ?, ?, 'pending')", (category, content, confidence))
            await db.commit()
            return cursor.lastrowid

    return asyncio.run(_insert())


def test_risk_suggestion_can_be_approved_and_state_changes(web_app):
    sid = _seed_pending(web_app, "risk_adjustment",
                        json.dumps({"risk_appetite": "balanced", "position_size_pct": 8.0,
                                    "stop_loss_pct": 2.5, "leverage": 3}))
    c = _client(web_app, TRADER)

    r = c.post(f"/api/ai-suggestions/{sid}/approve")

    assert r.status_code == 200 and "Forbidden" not in r.text
    rows = asyncio.run(_rows(web_app.state.config.db_path,
                             "SELECT * FROM ai_suggestions WHERE id=?", (sid,)))
    assert rows[0]["status"] in ("approved", "applied")
    assert rows[0]["applied_at"], "an approval must be timestamped"
    # The claim on the button ("applied") must be true: the soft params moved.
    assert web_app.state.config.soft_params.leverage == 3


def test_suggestion_can_be_rejected_from_the_ui(web_app):
    sid = _seed_pending(web_app, "coin_selection", json.dumps({"symbols": ["ADAUSDT"]}))
    c = _client(web_app, TRADER)

    r = c.post(f"/api/ai-suggestions/{sid}/reject")

    assert r.status_code == 200 and "Rejected" in r.text
    assert asyncio.run(_scalar(web_app.state.config.db_path,
                               "SELECT status FROM ai_suggestions WHERE id=?", (sid,))) == "rejected"


def test_unparseable_risk_suggestion_is_not_reported_as_applied(web_app):
    """Approving must not claim an action it could not perform."""
    sid = _seed_pending(web_app, "risk_adjustment", "not json at all")
    c = _client(web_app, TRADER)

    r = c.post(f"/api/ai-suggestions/{sid}/approve")

    assert r.status_code == 200
    assert asyncio.run(_scalar(web_app.state.config.db_path,
                               "SELECT status FROM ai_suggestions WHERE id=?", (sid,))) == "approved"


# ── 4. the /ai page and the HTMX partial agree ──────────────────────────

def test_partial_shows_confidence_and_an_untruncated_body(web_app):
    """No silent downgrade: the poller shows what the first paint showed.

    The old pair disagreed — the inline render showed confidence and 200 chars,
    the HTMX partial dropped confidence and cut the body to 100 — so the card
    visibly shrank on the first refresh.
    """
    asyncio.run(_wipe_suggestions(web_app.state.config.db_path))
    sid = _seed_pending(web_app, "strategy_optimization",
                        json.dumps({"strategy": "rsi_macd_trend",
                                    "reason": "x" * 300}), confidence=0.42)
    c = _client(web_app, VIEWER)

    partial = c.get("/partials/ai-suggestions",
                    params={"status": "pending", "limit": 10}).text
    page = c.get("/ai").text
    assert '<div id="ai-pending"' in page, "the poller target must exist"

    for label, body in (("partial", partial), ("first paint", page)):
        assert "42%" in body, f"{label}: confidence must render as a percentage"
        assert "rsi_macd_trend" in body, f"{label}: the AI payload must be summarised"
        assert "x" * 100 in body, f"{label}: the body must not be truncated to 100 chars"
        assert f'id="btns-{sid}"' in body, f"{label}: approve/reject must be present"
        assert f'/api/ai-suggestions/{sid}/approve' in body
        assert f'/api/ai-suggestions/{sid}/reject' in body


# ── 5. market state prefers real data ───────────────────────────────────

def test_market_state_prefers_the_persisted_assessment(web_app):
    ctl = type("Ctl", (), {"_last_market_assessment": {"market_regime": "in-memory"}})()
    web_app.state.ai_controller = ctl
    seed = {"market_regime": "trending_up", "summary": "BTC 突破区间上沿"}
    asyncio.run(_seed_setting(web_app.state.config.db_path,
                              "ai_market_assessment", json.dumps(seed)))
    try:
        r = _client(web_app, VIEWER).get("/api/market-state")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html"), \
            "the card swaps this body in directly; JSON would render as noise"
        body = r.text
        assert "trending_up" in body, "the real assessment must win"
        assert "等待" not in body
        assert "{" not in body, "no raw JSON in the card"
    finally:
        asyncio.run(_seed_setting(web_app.state.config.db_path, "ai_market_assessment", None))
        delattr(web_app.state, "ai_controller")


def test_market_state_falls_back_to_the_in_memory_assessment(web_app):
    """A restart must not blank a good assessment the process still holds."""
    ctl = type("Ctl", (), {"_last_market_assessment": {"market_regime": "ranging"}})()
    web_app.state.ai_controller = ctl
    try:
        body = _client(web_app, VIEWER).get("/api/market-state").text
        assert "ranging" in body
        assert "等待" not in body
    finally:
        delattr(web_app.state, "ai_controller")


def test_market_state_says_waiting_only_when_there_is_none(web_app):
    body = _client(web_app, VIEWER).get("/api/market-state").text
    assert "等待" in body


def test_market_state_poll_does_not_destroy_the_rendered_card(web_app):
    """The regression from the audit: polling replaced a good card with "waiting"."""
    asyncio.run(_seed_setting(web_app.state.config.db_path, "ai_market_assessment",
                              json.dumps({"market_regime": "bullish",
                                          "summary": "顺势做多"})))
    try:
        c = _client(web_app, VIEWER)
        first_paint = c.get("/ai").text
        assert "bullish" in first_paint
        refreshed = c.get("/api/market-state").text
        assert "bullish" in refreshed, "the poll must not blank the assessment"
        assert "等待" not in refreshed
    finally:
        asyncio.run(_seed_setting(web_app.state.config.db_path, "ai_market_assessment", None))


async def _seed_setting(db_path: str, key: str, value):
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        if value is None:
            await db.execute("DELETE FROM system_config WHERE key=?", (key,))
        else:
            await db.execute(
                "INSERT OR REPLACE INTO system_config (key, value, category) VALUES (?, ?, 'ai')",
                (key, value))
        await db.commit()


def test_assessment_loop_stores_what_market_state_reads(web_app, monkeypatch):
    """End-to-end: the producer side feeds the endpoint the UI polls."""
    from core.ai.deepseek_ctl import DeepSeekController

    ctl = DeepSeekController(web_app.state.config, web_app.state.event_bus)
    assessment = {"market_regime": "trending_down", "summary": "风险偏好收缩"}

    async def _fake_assess():
        return assessment

    async def _one_iteration(*a, **k):
        # The loop's trailing `asyncio.sleep(interval)` runs even after this, so
        # the interval is zeroed to keep the test bounded.
        ctl._running = False

    monkeypatch.setattr(ctl, "assess_market", _fake_assess)
    monkeypatch.setattr(ctl, "_heartbeat", _one_iteration)
    monkeypatch.setitem(ctl.config.ai_task_intervals, "market_assessment", 0)
    ctl._running = True
    asyncio.run(ctl._market_assessment_loop())

    try:
        body = _client(web_app, VIEWER).get("/api/market-state").text
        assert "trending_down" in body
        assert ctl._last_market_assessment == assessment, "kept in memory too"
    finally:
        asyncio.run(_seed_setting(web_app.state.config.db_path, "ai_market_assessment", None))


async def _async(value):
    return value


# ── 6. alert acknowledgement ────────────────────────────────────────────

def test_alert_ack_button_posts_and_sets_acknowledged(web_app):
    alert_id = _seed_alert(web_app)
    c = _client(web_app, TRADER)

    listing = c.get("/partials/alerts-filtered", params={"limit": 50})
    assert listing.status_code == 200
    assert f'hx-post="/api/alerts/{alert_id}/ack"' in listing.text
    assert f'hx-target="#ack-{alert_id}"' in listing.text, \
        "the ack response must not replace the whole list"

    r = c.post(f"/api/alerts/{alert_id}/ack")

    assert r.status_code == 200
    assert asyncio.run(_scalar(web_app.state.config.db_path,
                               "SELECT acknowledged FROM alerts WHERE id=?",
                               (alert_id,))) == 1
    # The refreshed list now shows the acknowledged state and no button.
    refreshed = c.get("/partials/alerts-filtered", params={"limit": 50}).text
    assert f'hx-post="/api/alerts/{alert_id}/ack"' not in refreshed
    assert "已确认" in refreshed


def _seed_alert(web_app) -> int:
    import aiosqlite

    async def _insert():
        async with aiosqlite.connect(web_app.state.config.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO alerts (level, type, message, symbol) "
                "VALUES ('critical', 'risk.breach', 'probe alert', 'BTCUSDT')")
            await db.commit()
            return cursor.lastrowid

    return asyncio.run(_insert())


def test_alert_ack_denies_a_viewer(web_app):
    alert_id = _seed_alert(web_app)
    r = _client(web_app, VIEWER).post(f"/api/alerts/{alert_id}/ack")
    assert r.status_code == 403
    assert asyncio.run(_scalar(web_app.state.config.db_path,
                               "SELECT acknowledged FROM alerts WHERE id=?",
                               (alert_id,))) == 0


# ── 7. exactly one /api/kline/{symbol} owner ────────────────────────────

def _kline_routes(app):
    return [r for r in app.router.routes
            if getattr(r, "path", None) == "/api/kline/{symbol}"]


def test_exactly_one_kline_route_exists(web_app):
    routes = _kline_routes(web_app)
    assert len(routes) == 1, \
        f"expected a single owner, found {[getattr(r, 'endpoint', None) for r in routes]}"
    assert routes[0].endpoint.__module__ == "web.routes.market", \
        "market.py must own /api/kline/{symbol}"


def test_kline_route_still_serves_the_frozen_shape(web_app):
    """Deleting the duplicate must not change the response contract."""
    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=3, freq="5min")
    frame = pd.DataFrame({
        "open": [1.0, 2.0, 3.0], "high": [2.0, 3.0, 4.0],
        "low": [0.0, 1.0, 2.0], "close": [1.5, 2.5, 3.5],
        "volume": [10.0] * 3,
    }, index=idx)

    class FakeMarketData:
        async def get_historical(self, symbol, interval, limit=200):
            return frame

    web_app.state.market_data = FakeMarketData()
    try:
        r = _client(web_app, VIEWER).get("/api/kline/BTCUSDT",
                                         params={"interval": "5m", "limit": 3})
        assert r.status_code == 200, r.text
        candles = r.json()
        assert len(candles) == 3
        assert set(candles[0]) == {"time", "open", "high", "low", "close", "volume"}
    finally:
        delattr(web_app.state, "market_data")


def test_dead_market_helpers_are_gone():
    """`_market_data_provider` and its duplicate `_configured_client` were dead."""
    import inspect

    from web.routes import dashboard_partials

    source = inspect.getsource(dashboard_partials)
    assert "_market_data_provider" not in source
    assert source.count("async def _configured_client") == 1, \
        "only /api/price's helper may remain in this module"


# ── 8. ai_panel.html hygiene ────────────────────────────────────────────

def _panel_template() -> str:
    return (Path(__file__).resolve().parents[1]
            / "web" / "templates" / "ai_panel.html").read_text(encoding="utf-8")


def test_ai_panel_has_no_duplicate_class_attribute():
    text = _panel_template()
    for match in re.finditer(r"<select\b[^>]*>", text):
        tag = match.group(0)
        attrs = re.findall(r"\b([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=", tag)
        duplicates = {a for a in attrs if attrs.count(a) > 1}
        assert not duplicates, f"duplicate attribute(s) {duplicates} on {tag!r}"


def test_ai_panel_does_not_duplicate_the_strategy_monitor_poller():
    text = _panel_template()
    assert "id=\"strategy-monitor\"" not in text, \
        "the trade.html monitor was re-implemented here; it must only be linked"
    assert "renderMonitor" not in text
    assert "setInterval(fetchMonitor" not in text
    assert "fetch('/api/strategy-monitor')" not in text
    assert "/trade#strategy" in text, "the page must link to the real monitor"


def test_ai_panel_renders_the_shared_suggestion_partial(web_app):
    body = _client(web_app, VIEWER).get("/ai").text
    assert 'hx-get="/partials/ai-suggestions' in body
    assert 'id="ai-pending"' in body
    # The inline copy of the list is gone: only the partial may render it.
    assert "s.content[:200]" not in body
    assert "暂无待处理建议" in body or "badge badge-blue" in body


# ── 9. news-sources route is gone ───────────────────────────────────────

def test_dead_news_sources_route_is_deleted(web_app):
    paths = {getattr(r, "path", None) for r in web_app.router.routes}
    assert "/api/news-sources" not in paths, \
        "nothing referenced this endpoint; its UI block does not exist"
    assert _client(web_app, VIEWER).get("/api/news-sources").status_code == 404


# ── 10. strategy-lifecycle panel ────────────────────────────────────────

def test_lifecycle_events_endpoint_returns_real_data(web_app):
    """Evidence for keeping the routes: they read a populated table."""
    _seed_lifecycle_event(web_app, "rsi_macd_trend", "retired", "Auto-retired (7-day grace)")

    rows = _client(web_app, VIEWER).get("/api/strategy-lifecycle/events").json()

    assert isinstance(rows, list) and rows
    assert {"strategy_name", "action", "created_at"} <= set(rows[0])
    assert rows[0]["strategy_name"] == "rsi_macd_trend"


def test_ai_page_shows_the_compact_lifecycle_panel(web_app):
    _seed_lifecycle_event(web_app, "ema_volume_breakout", "generated", "Manual probe")
    c = _client(web_app, TRADER)
    body = c.get("/ai").text
    assert 'id="strategy-lifecycle-panel"' in body
    assert 'hx-get="/partials/strategy-lifecycle"' in body
    assert 'hx-post="/api/strategy-lifecycle/optimize"' in body, \
        "the manual optimization trigger must be reachable from /ai"

    partial = c.get("/partials/strategy-lifecycle")
    assert partial.status_code == 200
    assert "strategy-lifecycle-panel" not in partial.text, \
        "the partial must be the inner list only, or the button would be swapped away"
    assert "ema_volume_breakout" in partial.text


def test_lifecycle_optimize_is_hidden_from_a_viewer(web_app):
    body = _client(web_app, VIEWER).get("/ai").text
    assert "strategy-lifecycle-panel" in body
    assert 'hx-post="/api/strategy-lifecycle/optimize"' not in body


def _seed_lifecycle_event(web_app, name: str, action: str, reason: str) -> int:
    import aiosqlite

    async def _insert():
        async with aiosqlite.connect(web_app.state.config.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO strategy_lifecycle_events "
                "(strategy_name, action, trigger_reason) VALUES (?, ?, ?)",
                (name, action, reason))
            await db.commit()
            return cursor.lastrowid

    return asyncio.run(_insert())


def test_lifecycle_optimize_requires_trader(web_app):
    assert _client(web_app, VIEWER).post("/api/strategy-lifecycle/optimize").status_code == 403


# ── 11. the heartbeat's daily count now moves ───────────────────────────

def test_heartbeat_counts_persisted_suggestions(web_app, producer):
    asyncio.run(_wipe_suggestions(web_app.state.config.db_path))
    before = _client(web_app, VIEWER).get("/api/ai-heartbeat").text
    asyncio.run(producer._publish_suggestion(
        "coin_selection", json.dumps({"symbols": ["ADAUSDT"]}), 0.7))

    after = _client(web_app, VIEWER).get("/api/ai-heartbeat").text

    assert "今日建议" in after or "建议" in after
    assert before != after, "the heartbeat must reflect the new suggestion"


def test_suggestion_summary_is_readable_and_lossless_for_odd_payloads():
    from web.routes.ai import _suggestion_summary, _suggestion_view

    assert "ADAUSDT" in _suggestion_summary("coin_selection", json.dumps({"symbols": ["ADAUSDT"]}))
    assert _suggestion_summary("coin_selection", "raw text body") == "raw text body"
    assert _suggestion_summary("coin_selection", "{not json") == "{not json"

    view = _suggestion_view({"id": 1, "category": "coin_selection", "content": "x",
                             "confidence": 0.42})
    assert view["confidence_pct"] == 42
    assert view["content"] == "x", "the raw body stays available next to the summary"
    assert view["summary"], "even an unparseable body yields a displayable string"
    assert len(view["summary"]) <= 401, "the preview is bounded"
