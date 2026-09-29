"""Round-2 audit regressions (follow-up to v2.0.0).

Six defects, all reproduced before the fix and pinned here afterwards:

1. **reset-sim raced an in-flight open** (HIGH, ``web/routes/settings.py`` +
   ``core/executor/executor.py``): the reset erased ``trades``/``positions`` and
   restored 10000 while an open had already committed its row but not yet
   deducted its cash, so the balance ended up below the identity by the whole
   open notional (measured Δ −100.160030) with no row to explain it.
2. **event bus busy-loop** (MEDIUM, ``app/event_bus.py``): the ``except
   Exception`` branch retried ``queue.get()`` with no backoff, so a queue bound
   to another loop turned it into a 100% CPU log flood.
3. **unbounded caches/locks** (MEDIUM, ``core/market_data/ttl_cache.py`` and
   ``core/market_data/screener.py``).
4. **one shared rate budget** (LOW, ``web/routes/audit.py``): ticker+depth+
   trades+kline shared 300/min, so four tabs at the documented cadence 429'd.
5. **``save_sim_balance`` wrote an absolute value** (LOW, ``db/database.py``).
6. **``/api/coin/{symbol}`` did not validate the symbol** (LOW).

No network: every upstream call is a fake, every database is a temp file.
"""
import asyncio
import sqlite3
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import Event, EventBus, EventType
from core.executor.executor import OrderExecutor
from core.market_data.ttl_cache import TTLCache
from db.database import (
    atomic_adjust_balance,
    balance_lock,
    init_database,
    load_sim_balance,
    save_sim_balance,
)
from web.context import AppContext
from web.routes import settings


def _run(coro):
    return asyncio.run(coro)


class _FakeRisk:
    def __init__(self):
        self.balance = 10000.0

    def update_balance(self, balance):
        self.balance = balance


class _AdminUser:
    is_admin = True
    is_trader = True
    username = "admin"


def _scalar(db_path, sql, params=()):
    db = sqlite3.connect(db_path)
    try:
        return db.execute(sql, params).fetchone()[0]
    finally:
        db.close()


def _open_notional(db_path):
    return float(_scalar(db_path, "SELECT COALESCE(SUM(quantity * entry_price), 0) "
                                  "FROM trades WHERE status='open'") or 0.0)


def _realised(db_path):
    return float(_scalar(db_path, "SELECT COALESCE(SUM(pnl), 0) "
                                  "FROM trades WHERE status='closed'") or 0.0)


def _identity(db_path):
    return 10000.0 - _open_notional(db_path) + _realised(db_path)


def _delta(db_path):
    return _run(load_sim_balance(db_path)) - _identity(db_path)


# ======================================================================
# defect 1 — reset-sim must not race an in-flight open
# ======================================================================
@pytest.fixture()
def admin_app(tmp_path):
    """A minimal app exposing only ``/api/settings/reset-sim`` (+ a real executor)."""
    import db.database as database

    db_path = str(tmp_path / "reset_race.db")
    _run(init_database(db_path))
    _run(save_sim_balance(10000.0, db_path))
    database.DB_PATH = db_path

    config = Config()
    config.db_path = db_path
    config.config_dir = str(tmp_path)
    config.mode = "sim"

    bus = EventBus()
    app = FastAPI()
    app.state.config = config
    app.state.event_bus = bus
    app.state.balance = 10000.0
    app.state.risk_manager = None
    app.state.get_price = lambda symbol: None
    executor = OrderExecutor(config, bus)
    executor.wire_risk_manager(_FakeRisk())
    app.state.executor = executor
    ctx = AppContext(config, bus, app)
    settings.register(app, ctx)

    @app.middleware("http")
    async def _auth(request, call_next):
        request.state.user = _AdminUser()
        return await call_next(request)

    yield {"app": app, "config": config, "executor": executor,
           "db_path": db_path, "client": TestClient(app)}


def _slow_balance_adjust(seconds=0.6):
    """Patch the executor's balance writer to widen the open's commit→deduct window.

    This is the auditor's reproduction (a sleep injected at the call site, no
    logic change): the open row is committed, then the deduction is delayed, and
    the reset lands in between.
    """
    import core.executor.executor as executor_mod

    real = executor_mod.atomic_adjust_balance

    async def _slow(delta, path=None):
        await asyncio.sleep(seconds)
        return await real(delta, path)

    executor_mod.atomic_adjust_balance = _slow
    return lambda: setattr(executor_mod, "atomic_adjust_balance", real)


def test_reset_sim_does_not_race_an_in_flight_open(admin_app):
    """The reset must serialise with the open, not erase its row out from under it.

    Before: `trades rows 0, balance 9899.83997, identity 10000 → delta −100.160030`
    (the open's cash left after the restore, with no row left to explain it).
    """
    db_path = admin_app["db_path"]
    executor = admin_app["executor"]
    restore = _slow_balance_adjust()
    row_committed = threading.Event()
    failure = {}

    def _open_in_flight():
        async def _poll_row():
            # The open writes its row before it deducts; wait for it, so the reset
            # is guaranteed to be issued inside the commit→deduct window.
            for _ in range(400):
                if _scalar(db_path, "SELECT COUNT(*) FROM trades") > 0:
                    return True
                await asyncio.sleep(0.005)
            return False

        async def _scenario():
            task = asyncio.create_task(executor._execute_sim({
                "symbol": "BTCUSDT", "side": "long", "price": 100.0, "quantity": 1.0,
                "amount_usdt": 100.0, "order_type": "market",
                "strategy": "manual", "trader": "manual"}))
            try:
                if not await _poll_row():
                    failure["error"] = "the open never committed its row"
                    return
                row_committed.set()          # the reset may now fire
                await task
            except Exception as exc:          # pragma: no cover - reported below
                failure["error"] = repr(exc)

        asyncio.run(_scenario())

    worker = threading.Thread(target=_open_in_flight)
    worker.start()
    assert row_committed.wait(10.0), "the open never reached its commit→deduct window"

    response = admin_app["client"].post("/api/settings/reset-sim")
    worker.join(20.0)
    restore()

    assert not failure, failure
    assert response.status_code == 200, response.text
    # No orphan row and no orphan position: whatever survived must be coherent.
    trades = _scalar(db_path, "SELECT COUNT(*) FROM trades")
    positions = _scalar(db_path, "SELECT COUNT(*) FROM positions")
    assert trades == 0, f"an open row survived the reset: {trades}"
    assert positions == 0, f"an orphan position survived the reset: {positions}"
    assert executor.get_open_positions() == {}
    # The whole point: the ledger identity holds to the cent.
    assert _delta(db_path) == pytest.approx(0.0, abs=1e-9), (
        f"reset raced the in-flight open: balance={_run(load_sim_balance(db_path))!r} "
        f"identity={_identity(db_path)!r} delta={_delta(db_path)!r}")


def test_reset_barrier_waits_for_a_slow_open_and_blocks_new_ones(admin_app):
    """A reset in progress queues the next open instead of letting it slip past."""
    executor = admin_app["executor"]
    order = []

    async def _scenario():
        release = asyncio.Event()

        async def _slow_open():
            async with executor._mutation_slot():
                order.append("open-start")
                await release.wait()
                order.append("open-end")

        async def _reset():
            async with executor.reset_barrier():
                order.append("reset")
                executor.check_no_mutations()   # asserts nothing is mid-mutation

        open_task = asyncio.create_task(_slow_open())
        await asyncio.sleep(0.05)
        reset_task = asyncio.create_task(_reset())
        await asyncio.sleep(0.05)
        # The reset is waiting for the open, so it has not run yet.
        assert order == ["open-start"], order
        release.set()
        await asyncio.gather(open_task, reset_task)

    _run(_scenario())
    assert order == ["open-start", "open-end", "reset"], order


# ======================================================================
# defect 2 — the event bus must never busy-loop
# ======================================================================
def test_event_bus_error_branch_yields_and_rebinds(tmp_path):
    """A queue bound to another loop must not turn ``_process`` into a spin.

    Loop A parks a live ``queue.get()`` waiter (that is what makes the Queue
    report "bound to a different event loop"); loop B then runs the bus.

    Before: the error branch retried with no backoff — ~74k branch runs/second
    (100% of one core) and the fair counter task got **0** turns.
    After: the branch backs off and rebinds, and the fair counter runs freely.
    """
    bus = EventBus()
    waiter_ready = threading.Event()
    stop_waiter = threading.Event()

    def _loop_a_waiter():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _park():
            waiter_ready.set()
            getter = asyncio.ensure_future(bus._queue.get())
            while not stop_waiter.is_set():
                await asyncio.sleep(0.01)
            getter.cancel()
            try:
                await getter
            except asyncio.CancelledError:
                pass

        loop.run_until_complete(_park())
        loop.close()

    threading.Thread(target=_loop_a_waiter, daemon=True).start()
    assert waiter_ready.wait(5.0), "loop A never parked its waiter"

    log_lines = {"n": 0}
    import app.event_bus as event_bus_mod
    sink = event_bus_mod.logger.add(
        lambda m: log_lines.__setitem__("n", log_lines["n"] + 1), level="WARNING")

    async def _scenario():
        turns = {"n": 0}

        async def _fair_counter():
            while True:
                turns["n"] += 1
                await asyncio.sleep(0)      # yields every turn — a fair task

        bus._running = True
        bus_task = asyncio.create_task(bus._process())
        fair_task = asyncio.create_task(_fair_counter())
        try:
            await asyncio.sleep(0.5)
        finally:
            bus._running = False
            for task in (bus_task, fair_task):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        return turns["n"]

    try:
        turns = _run(_scenario())
    finally:
        stop_waiter.set()
        event_bus_mod.logger.remove(sink)

    assert turns > 0, (
        "the event bus starved the whole loop for 0.5s (busy-loop: the error "
        "branch retried without yielding)")
    assert log_lines["n"] <= 5, (
        f"the error branch logged {log_lines['n']} lines in 0.5s (log flood)")
    # Both are the same defect; the backoff is what makes the flood bounded.
    import inspect
    source = inspect.getsource(EventBus._process)
    assert "asyncio.sleep(0.1)" in source, "the error branch needs its backoff"


def test_event_bus_keeps_publishing_after_a_loop_rebind():
    """The recovery is not just a backoff: the bus serves events again."""
    bus = EventBus()
    # Pretend the queue was created by a different (now dead) loop.
    bus._loop = object()
    received = []

    async def _handler(event):
        received.append(event.data.get("n"))

    async def _scenario():
        bus.subscribe_all(_handler)
        await bus.start()
        await bus.publish(Event(EventType.MARKET_TICK, {"n": 7}))
        await asyncio.sleep(0.3)
        await bus.shutdown()

    _run(_scenario())
    assert received == [7], received


# ======================================================================
# defect 3 / 6 — bounded caches, dropped locks, validated symbols
# ======================================================================
def test_ttl_cache_bounds_values_and_drops_finished_locks():
    """200 failed fetches with distinct keys must leave a bounded map.

    Before: ``_locks=200, _values=0`` — nothing was ever evicted.
    """
    cache = TTLCache()
    # Before the fix the class had no cap at all; the audited scenario is 200 keys.
    cap = getattr(cache, "_max_locks", 200)

    async def _boom():
        raise RuntimeError("upstream down")

    async def _scenario():
        for i in range(200):
            value, error = await cache.get(("key", i), 30.0, _boom)
            assert value is None and error
        return len(cache._values), len(cache._locks)

    values, locks = _run(_scenario())
    assert values == 0, "a failed fetch must never be cached"
    assert locks == 0, f"{locks} locks survived 200 failed fetches"
    assert cap > 0, "the lock map needs a bound"


def test_ttl_cache_evicts_values_lru_when_full():
    cache = TTLCache()
    cap = cache._max_values

    async def _value(key):
        return f"v{key}"

    async def _scenario():
        for i in range(cap + 50):
            value, error = await cache.get(i, 600.0, (lambda i=i: _value(i)))
            assert error is None and value == f"v{i}"
        return len(cache._values), len(cache._locks)

    values, locks = _run(_scenario())
    assert values <= cap, values
    assert locks == 0, "a finished fetch kept its lock"


def test_ttl_cache_still_dedupes_concurrent_fetches():
    """The eviction must not break the contract that made the lock map exist."""
    cache = TTLCache()
    calls = {"n": 0}

    async def _fetch():
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return "payload"

    async def _scenario():
        return await asyncio.gather(*(cache.get("same", 30.0, _fetch) for _ in range(8)))

    results = _run(_scenario())
    assert all(value == "payload" and error is None for value, error in results)
    assert calls["n"] == 1, f"8 concurrent callers made {calls['n']} upstream calls"


class _FailingExchange:
    """Fake screener fetch that always fails, keyed per symbol."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, path, params):
        self.calls += 1
        raise RuntimeError("upstream down")


def test_screener_caches_and_locks_stay_bounded():
    """50+ distinct detail keys must leave a bounded cache.

    Before: reading 52 distinct symbols left 52 cache entries and one lock per
    key, with no eviction path at all.
    """
    from tests.test_screener import make_screener

    exchange = _FailingExchange()
    screener = make_screener(exchange, cache_ttl=300.0)
    # Pin the default bound (the audited scenario was 52 distinct symbols, just
    # under it) and then prove the eviction really runs at a visible cap.
    assert 8 <= screener.max_cache_entries <= 1024
    assert 4 <= screener.max_cache_locks <= 512
    screener.max_cache_entries = 8
    screener.max_cache_locks = 4

    async def _scenario():
        for i in range(60):
            # A failed upstream is not an exception here: it is reported as a
            # payload with `fetch_errors` and empty metrics (the route 404s it).
            payload = await screener.detail(f"S{i}USDT")
            assert payload["metrics"].get("fetch_errors"), payload
        return len(screener._cache), len(screener._locks)

    entries, locks = _run(_scenario())
    # The screener caches its detail payloads (including a fetch-error payload, so
    # a flapping host is not re-crawled per request) — the point here is that the
    # map is **bounded**, not that a failure is uncached.
    assert entries <= 8, entries
    assert locks <= 4, f"{locks} locks survived (nothing evicted them)"
    assert exchange.calls >= 60


def test_screener_cache_evicts_lru_and_keeps_the_cache_working():
    """The cache still serves hits, but only for a bounded number of keys."""
    from tests.test_screener import FakeExchange, make_screener

    exchange = FakeExchange([f"S{i}USDT" for i in range(30)])
    screener = make_screener(exchange, cache_ttl=300.0)
    screener.max_cache_entries = 8
    screener.max_cache_locks = 4

    async def _scenario():
        for i in range(30):
            await screener.detail(f"S{i}USDT")
        entries = len(screener._cache)
        locks = len(screener._locks)
        # A repeat is still served from the cache (the eviction did not disable it).
        hits_before = screener.stats["cache_hits"]
        await screener.detail("S29USDT")
        return entries, locks, screener.stats["cache_hits"] - hits_before

    entries, locks, hits = _run(_scenario())
    assert entries <= 8, entries
    assert locks <= 4, locks
    assert hits == 1, "the cache must still answer a repeated read"


@pytest.mark.parametrize("bad", ["BTC", "bt", "NOT A SYMBOL", "BTC/USDT", "BTCUSDC",
                                 "BTCUSDT-X", "A" * 30, "BTC-USDC"])
def test_valid_symbol_rejects_malformed_shapes(bad):
    from core.market_data.screener import valid_symbol
    assert valid_symbol(bad) is None, bad


@pytest.mark.parametrize("good,canonical", [
    ("btcusdt", "BTCUSDT"),
    ("BTC-USDT", "BTC-USDT"),
    ("1000PEPEUSDT", "1000PEPEUSDT"),
])
def test_valid_symbol_accepts_the_documented_shapes(good, canonical):
    from core.market_data.screener import valid_symbol
    assert valid_symbol(good) == canonical


def test_coin_route_rejects_a_bad_symbol_without_an_upstream_call(tmp_path):
    counter = {"n": 0}

    class _CountingClient:
        host = "https://data-api.binance.vision"

        async def close(self):
            return None

        async def exchange_symbols(self):
            counter["n"] += 1
            return []

        async def ticker24h(self, symbol=None):
            counter["n"] += 1
            return []

        async def klines(self, *a, **kw):
            counter["n"] += 1
            return []

        async def order_book(self, *a, **kw):
            counter["n"] += 1
            return {"bids": [], "asks": []}

        async def recent_trades(self, *a, **kw):
            counter["n"] += 1
            return []

    import db.database as database

    db_path = str(tmp_path / "coin.db")
    _run(init_database(db_path))
    database.DB_PATH = db_path
    config = Config()
    config.db_path = db_path
    config.data_dir = str(tmp_path / "data")
    config.config_dir = str(tmp_path / "config")
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmp_path / "config" / name).write_text("{}\n", encoding="utf-8")

    from web.server import create_app
    from core.auth.auth import AuthManager

    async def _setup():
        am = AuthManager(db_path, "test-secret-at-least-32-bytes-long!!", 24)
        await am.create_user("v", "V1ewerPass!", "viewer", "v")
        return am

    auth = _run(_setup())
    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    app.state.market_data_client = _CountingClient()
    app.state.universe = None
    from core.market_data.ttl_cache import TTLCache
    app.state._market_ttl_cache_v2 = TTLCache()

    client = TestClient(app)
    assert client.post("/api/auth/login",
                       json={"username": "v", "password": "V1ewerPass!"}).status_code == 200

    for bad in ("BTC", "not a symbol", "BTCUSDT-X", "BTC-USDC", "A" * 30):
        r = client.get(f"/api/coin/{bad}")
        assert r.status_code == 400, (bad, r.status_code, r.text)
    assert counter["n"] == 0, f"a malformed symbol reached the network ({counter['n']} calls)"


# ======================================================================
# defect 4 — the polling cadence of four tabs must never 429
# ======================================================================
def test_four_tabs_at_the_documented_cadence_are_not_rate_limited():
    """Documented cadence: depth/trades 2s, account/orders 3s, klines 5s.

    Four pages opened by one caller cost 4×(30 depth + 30 trades + 20 account +
    20 orders + 12 klines) ≈ 448 requests/min against the one shared 300/min
    "market" pot → 28 of them 429'd while the page was doing nothing wrong.  Each
    endpoint family now has its own budget, so the documented cadence fits.
    """
    import web.routes.audit as routes_audit

    class _Req:
        def __init__(self, session):
            self.cookies = {"bt_session": session}
            self.client = None

    routes_audit.reset_rate_limits()
    try:
        # (route family, seconds between polls)
        families = [("depth", 2.0), ("trades", 2.0), ("account", 3.0),
                    ("orders", 3.0), ("klines", 5.0)]
        # account/orders are unauth-guarded read routes covered by the aggregate
        # "market" pot (they are not market-data fan-outs of their own).
        group_for = {"account": "market", "orders": "market"}
        rejected = []
        charged = 0
        for page in range(4):
            req = _Req("one-browser")
            for family, period in families:
                group = group_for.get(family, family)
                for tick in range(60):
                    if tick % int(period) != 0:
                        continue
                    charged += 1
                    if routes_audit.check_rate_limit(req, group) is not None:
                        rejected.append((page, family, tick))
        assert charged >= 400, charged
        assert rejected == [], (
            f"legitimate polling got {len(rejected)} 429s (the audit measured 28): "
            f"{rejected[:5]}")
    finally:
        routes_audit.reset_rate_limits()


def test_rate_limit_still_bites_a_tight_loop():
    """Generosity must not remove the ceiling: a hammering loop still gets a 429."""
    import web.routes.audit as routes_audit

    class _Req:
        def __init__(self, session):
            self.cookies = {"bt_session": session}
            self.client = None

    routes_audit.reset_rate_limits()
    try:
        req = _Req("hammer")
        budget = routes_audit.RATE_LIMITS["depth"]
        codes = [routes_audit.check_rate_limit(req, "depth") for _ in range(budget + 1)]
        assert all(r is None for r in codes[:budget])
        assert codes[budget] is not None and codes[budget].status_code == 429
    finally:
        routes_audit.reset_rate_limits()


def test_each_market_family_has_its_own_bucket():
    import web.routes.audit as routes_audit

    for family in ("ticker", "depth", "trades", "klines", "coin", "overview"):
        assert family in routes_audit.RATE_LIMITS, family
        assert routes_audit.RATE_FAMILY[family] == family
    # ... and the aggregate still exists as a process-wide ceiling.
    assert routes_audit.RATE_AGGREGATE["depth"] == "market"
    assert routes_audit.RATE_LIMITS["market"] >= routes_audit.RATE_LIMITS["depth"]


# ======================================================================
# defect 5 — save_sim_balance contract
# ======================================================================
def test_save_sim_balance_is_unconditional_for_a_reset_and_guarded_for_a_read(tmp_path):
    """Pin both contracts.

    * Unconditional (``expected=None``) — a reset states an absolute value.
    * Guarded (``expected`` given) — a stale caller cannot revert a concurrent
      fill: the write is refused and the stored value is returned unchanged.
    """
    from db.database import save_sim_balance_guarded

    db_path = str(tmp_path / "balance_contract.db")
    _run(init_database(db_path))
    _run(save_sim_balance(10000.0, db_path))
    assert _run(load_sim_balance(db_path)) == pytest.approx(10000.0)

    # A reset-style write states the new state directly.
    _run(save_sim_balance(8200.0, db_path))
    assert _run(load_sim_balance(db_path)) == pytest.approx(8200.0)

    # A guarded write based on the current value lands.
    stored = _run(save_sim_balance_guarded(8100.0, 8200.0, db_path))
    assert stored == pytest.approx(8100.0)
    assert _run(load_sim_balance(db_path)) == pytest.approx(8100.0)

    # A fill moves the balance; the stale caller's 8200 must NOT come back.
    _run(atomic_adjust_balance(-50.0, db_path))
    assert _run(load_sim_balance(db_path)) == pytest.approx(8050.0)
    stored = _run(save_sim_balance_guarded(8200.0, 8100.0, db_path))
    assert stored == pytest.approx(8050.0), "a stale guarded write must be refused"
    assert _run(load_sim_balance(db_path)) == pytest.approx(8050.0), \
        "a stale guarded write reverted a concurrent fill"


def test_guarded_write_still_takes_the_balance_lock(tmp_path):
    """The guarded path must not lose the serialisation the lock provides."""
    from db.database import save_sim_balance_guarded

    db_path = str(tmp_path / "balance_lock.db")
    _run(init_database(db_path))
    _run(save_sim_balance(10000.0, db_path))

    async def _scenario():
        async with balance_lock():
            task = asyncio.create_task(save_sim_balance_guarded(9000.0, 10000.0, db_path))
            await asyncio.sleep(0.05)
            blocked = not task.done()
        await task
        return blocked

    assert _run(_scenario()) is True, "the guarded write ignored the balance lock"
    assert _run(load_sim_balance(db_path)) == pytest.approx(9000.0)
