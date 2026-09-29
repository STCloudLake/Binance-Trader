"""Scratch repro: reset-sim racing an in-flight sim open (defect 1)."""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiosqlite  # noqa: E402
import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402

import db.database as database  # noqa: E402
import core.executor.executor as executor_mod  # noqa: E402
from app.config import Config  # noqa: E402
from app.event_bus import EventBus  # noqa: E402
from core.executor.executor import OrderExecutor  # noqa: E402
from db.database import init_database, load_sim_balance, save_sim_balance  # noqa: E402
from web.context import AppContext  # noqa: E402


class _FakeRisk:
    def __init__(self):
        self.balance = 10000.0

    def update_balance(self, b):
        self.balance = b


class _Admin:
    is_admin = True
    is_trader = True
    username = "admin"


async def main():
    tmpdir = tempfile.mkdtemp(prefix="bt_reset_race_")
    db_path = os.path.join(tmpdir, "race.db")
    await init_database(db_path)
    await save_sim_balance(10000.0, db_path)
    database.DB_PATH = db_path

    config = Config()
    config.db_path = db_path
    config.config_dir = tmpdir
    config.mode = "sim"

    bus = EventBus()
    executor = OrderExecutor(config, bus)
    executor.wire_risk_manager(_FakeRisk())

    app = FastAPI()
    app.state.executor = executor
    app.state.balance = 10000.0
    app.state.risk_manager = None
    app.state.config = config
    ctx = AppContext(config, bus, app)

    from web.routes import settings as routes_settings
    routes_settings.register(app, ctx)

    # Admin auth stub: the middleware normally sets request.state.user.
    @app.middleware("http")
    async def _auth(request, call_next):
        request.state.user = _Admin()
        return await call_next(request)

    # 0.4s sleep injected at the call site (widening the existing window).
    real_adjust = executor_mod.atomic_adjust_balance

    async def slow_adjust(delta, path=None):
        await asyncio.sleep(0.4)
        return await real_adjust(delta, path)

    executor_mod.atomic_adjust_balance = slow_adjust

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        open_task = asyncio.create_task(executor._execute_sim({
            "symbol": "BTCUSDT", "side": "long", "price": 100.0, "quantity": 1.0,
            "amount_usdt": 100.0, "order_type": "market",
            "strategy": "manual", "trader": "manual"}))
        await asyncio.sleep(0.2)          # open is mid-flight (row written, cash not moved yet)
        r = await client.post("/api/settings/reset-sim")
        await open_task
    executor_mod.atomic_adjust_balance = real_adjust

    db = await aiosqlite.connect(db_path)
    trades = (await (await db.execute("SELECT COUNT(*) FROM trades")).fetchone())[0]
    positions = (await (await db.execute("SELECT COUNT(*) FROM positions")).fetchone())[0]
    open_notional = (await (await db.execute(
        "SELECT COALESCE(SUM(quantity*entry_price),0) FROM trades WHERE status='open'")).fetchone())[0]
    realised = (await (await db.execute(
        "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE status='closed'")).fetchone())[0]
    await db.close()
    balance = await load_sim_balance(db_path)
    identity = 10000.0 - float(open_notional) + float(realised)
    print(f"status={r.status_code} trades rows={trades} positions={positions} "
          f"balance={balance!r} identity={identity!r} delta={balance - identity!r}")


asyncio.run(main())
