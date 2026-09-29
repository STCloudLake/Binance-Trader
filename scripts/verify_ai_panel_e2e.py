"""Verification harness: real E2E against a temp-DB app (not part of the suite).

Run with:  python scripts/verify_ai_panel_e2e.py
Builds a throw-away SQLite database, drives the AI suggestion producer through
`_coin_selection_loop()` with a stubbed AI response, then asserts every wiring
claim end to end.  No network, no real database.
"""
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:  # the labels below are Chinese; keep the console from mangling them
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from db.database import init_database

tmp = Path(tempfile.mkdtemp(prefix="bt_verify_"))
Config._instance = None
config = Config.load("sim")
config.db_path = str(tmp / "verify.db")
config.deepseek_api_key = "verify-key"


async def _setup():
    await init_database(config.db_path)
    am = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
    await am.create_user("trader1", "T1aderPass!", "trader", "t")
    return am


auth = asyncio.run(_setup())
from web.server import create_app
from alerts.manager import AlertManager

app = create_app(config, EventBus(), auth)
app.state.config = config
app.state.auth_manager = auth
mgr = AlertManager(config, app.state.event_bus)
asyncio.run(mgr.start())
app.state.alert_manager = mgr

# ---------------------------------------------------------------- 1. producer
from core.ai.deepseek_ctl import DeepSeekController

ctl = DeepSeekController(config, app.state.event_bus)
config.ai_mode = "suggest"


async def fake_select():
    return {"symbols": ["ADAUSDT", "DOGEUSDT"], "reason": "momentum + volume"}


async def stop_after_one(*a, **k):
    ctl._running = False


ctl.select_coins = fake_select
ctl._heartbeat = stop_after_one
config.ai_task_intervals["coin_selection"] = 0
ctl._running = True
asyncio.run(ctl._coin_selection_loop())
print("[1] producer: stubbed AI response pushed through _coin_selection_loop()")

import aiosqlite


async def rows(sql, args=(), write=False):
    async with aiosqlite.connect(config.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(sql, args)
        out = [dict(r) for r in await cur.fetchall()]
        if write:
            await db.commit()
        return out


inserted = asyncio.run(rows("SELECT * FROM ai_suggestions ORDER BY id DESC LIMIT 1"))
print("[2] INSERTed row:", json.dumps(inserted[0], ensure_ascii=False, default=str))
assert inserted and inserted[0]["status"] == "pending"
assert inserted[0]["confidence"] == 0.7 and inserted[0]["rationale"]

# dedupe: same payload again
again = asyncio.run(ctl._publish_suggestion("coin_selection", inserted[0]["content"], 0.7))
count = asyncio.run(rows("SELECT COUNT(*) c FROM ai_suggestions"))[0]["c"]
print(f"[3] dedupe: repeat returned {again!r}; table still holds {count} row(s)")

# ------------------------------------------------------------- 2. /ai render
c = TestClient(app)
c.post("/api/auth/login", json={"username": "trader1", "password": "T1aderPass!"})
sid = inserted[0]["id"]
partial = c.get("/partials/ai-suggestions", params={"limit": 10}).text
page = c.get("/ai").text
for label, body in (("/partials/ai-suggestions", partial), ("/ai first paint", page)):
    ok_conf = "70%" in body
    ok_cat = "coin_selection" in body
    ok_body = "ADAUSDT" in body
    ok_btns = f'id="btns-{sid}"' in body and f"/api/ai-suggestions/{sid}/approve" in body
    print(f"[4] {label}: confidence70%={ok_conf} category={ok_cat} content={ok_body} approve/reject={ok_btns}")
    assert ok_conf and ok_cat and ok_body and ok_btns

# --------------------------------------------------------- 3. market state
asyncio.run(rows("INSERT OR REPLACE INTO system_config (key,value,category) VALUES "
                 "('ai_market_assessment', ?, 'ai')",
                 (json.dumps({"market_regime": "trending_up", "summary": "BTC 强势"}),), write=True))
import db.database as _D
print("    [debug] routes will read DB_PATH =", _D.DB_PATH)
print("    [debug] seeded row =", asyncio.run(rows(
    "SELECT key, value FROM system_config WHERE key='ai_market_assessment'")))
from web.routes.pages import latest_market_assessment as _lma
print("    [debug] latest_market_assessment() ->", asyncio.run(_lma()))
ms = c.get("/api/market-state")
print(f"[5] GET /api/market-state -> {ms.status_code} {ms.headers['content-type']}")
print("    body:", ms.text[:160])
assert ms.status_code == 200 and "trending_up" in ms.text and "等待" not in ms.text
empty = c.get("/api/market-state").text  # after clearing below
asyncio.run(rows("DELETE FROM system_config WHERE key='ai_market_assessment'", write=True))
empty = c.get("/api/market-state").text
print("[6] empty store ->", re.sub(r"<[^>]+>", "", empty).strip()[:60], "| 等待 present:", "等待" in empty)
assert "等待" in empty

# ------------------------------------------------------------- 4. alert ack
asyncio.run(rows("INSERT INTO alerts (level,type,message,symbol) "
                 "VALUES ('critical','risk.breach','verify ack probe','BTCUSDT')", write=True))
alert_id = asyncio.run(rows("SELECT id FROM alerts ORDER BY id DESC LIMIT 1"))[0]["id"]
before = c.get("/partials/alerts-filtered", params={"limit": 50}).text
print(f"[7] alert {alert_id} button present:", f'hx-post="/api/alerts/{alert_id}/ack"' in before,
      "| target:", f'hx-target="#ack-{alert_id}"' in before)
assert f'hx-post="/api/alerts/{alert_id}/ack"' in before
r = c.post(f"/api/alerts/{alert_id}/ack")
print(f"[8] POST /api/alerts/{alert_id}/ack -> {r.status_code}; DB acknowledged =",
      asyncio.run(rows("SELECT acknowledged FROM alerts WHERE id=?", (alert_id,)))[0]["acknowledged"])
assert asyncio.run(rows("SELECT acknowledged FROM alerts WHERE id=?", (alert_id,)))[0]["acknowledged"] == 1

# ---------------------------------------------------------------- 5. kline
all_kline = [rt for rt in app.router.routes if getattr(rt, "path", None) == "/api/kline/{symbol}"]
print(f"[9] /api/kline/{{symbol}} route count = {len(all_kline)} "
      f"(owner module={all_kline[0].endpoint.__module__})")
assert len(all_kline) == 1


class FakeMD:
    async def get_historical(self, symbol, interval, limit=200):
        import pandas as pd

        idx = pd.date_range("2026-01-01", periods=4, freq="5min")
        return pd.DataFrame({
            "open": [1.0, 2.0, 3.0, 4.0], "high": [2.0, 3.0, 4.0, 5.0],
            "low": [0.0, 1.0, 2.0, 3.0], "close": [1.5, 2.5, 3.5, 4.5],
            "volume": [10.0] * 4}, index=idx)


app.state.market_data = FakeMD()
kr = c.get("/api/kline/BTCUSDT", params={"interval": "5m", "limit": 4})
print(f"[10] GET /api/kline/BTCUSDT -> {kr.status_code}; keys={sorted(kr.json()[0])}; bars={len(kr.json())}")
assert kr.status_code == 200 and len(kr.json()) == 4

# ------------------------------------------------------------ 6. lifecycle
asyncio.run(rows("INSERT INTO strategy_lifecycle_events (strategy_name,action,trigger_reason) "
                 "VALUES ('verify_strat','retired','verify probe')", write=True))
ev = c.get("/api/strategy-lifecycle/events").json()
lp = c.get("/partials/strategy-lifecycle").text
print(f"[11] lifecycle events = {len(ev)}; panel partial renders 'verify_strat':",
      "verify_strat" in lp, "; optimize button on /ai:", 'hx-post="/api/strategy-lifecycle/optimize"' in c.get("/ai").text)

# ------------------------------------------------------------- 7. cleanliness
panel = (Path("web/templates/ai_panel.html")).read_text(encoding="utf-8")
dup = [re.findall(r"\b([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=", m.group(0)) for m in re.finditer(r"<select\b[^>]*>", panel)]
print("[12] duplicate select attrs:", [x for x in dup if len(x) != len(set(x))] or "none")
print("[13] news-sources route gone:", "/api/news-sources" not in {getattr(rt, 'path', None) for rt in app.router.routes})
print("[14] strategy-monitor poller removed from ai_panel:",
      "renderMonitor" not in panel and "fetch('/api/strategy-monitor')" not in panel)
print("DONE")

# --------------------------------------------------- 8. approve / reject E2E
sid = asyncio.run(rows("SELECT id FROM ai_suggestions ORDER BY id DESC LIMIT 1"))[0]["id"]
r = c.post(f"/api/ai-suggestions/{sid}/approve")
state = asyncio.run(rows("SELECT status, applied_at FROM ai_suggestions WHERE id=?", (sid,)))[0]
print(f"[15] POST /api/ai-suggestions/{sid}/approve -> {r.status_code} {r.text!r}; row -> {state}")

asyncio.run(rows("INSERT INTO ai_suggestions (category,content,confidence,status) "
                 "VALUES ('risk_adjustment','{\"leverage\": 4, \"position_size_pct\": 6}',0.6,'pending')",
                 write=True))
rid = asyncio.run(rows("SELECT id FROM ai_suggestions ORDER BY id DESC LIMIT 1"))[0]["id"]
before_lev = config.soft_params.leverage
r2 = c.post(f"/api/ai-suggestions/{rid}/approve")
print(f"[16] risk approve -> {r2.status_code} {r2.text!r}; leverage {before_lev} -> {config.soft_params.leverage}"
      f"; row status =", asyncio.run(rows("SELECT status FROM ai_suggestions WHERE id=?", (rid,)))[0]["status"])
r3 = c.post(f"/api/ai-suggestions/{rid}/reject")
print(f"[17] reject -> {r3.status_code} {r3.text!r}; row status =",
      asyncio.run(rows("SELECT status FROM ai_suggestions WHERE id=?", (rid,)))[0]["status"])
