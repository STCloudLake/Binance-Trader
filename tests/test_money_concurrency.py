"""Concurrency and admin-route money tests (S2, S3, S5, S6, S7, S12).

Every test drives the real code paths against a temporary database: no network,
and ``data/binance_trader.db`` is never touched.  The invariant under test is the
same one ``tests/test_ledger_invariant.py`` pins:

    identity = 10000 − Σ(open rows: quantity × entry_price) + Σ(close rows: pnl)
    invariant: identity == system_config.sim_balance

The two families here are:

* **concurrency** — two handlers that read the same pre-mutation basis used to
  book two closes against one position (cash created from nothing), or two opens
  for one symbol (capital stranded), or a ``KeyError`` on the second full close.
  The executor now serialises every cash-moving mutation per symbol.
* **admin routes** — ``/api/db/cleanup`` deleted realised PnL without moving the
  balance, ``DELETE /api/db/row`` could delete a ledger row, and
  ``/api/settings/reset-sim`` left pending limit orders alive so the matcher
  filled one against the fresh 10000.
"""
import asyncio
import sqlite3
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import EventBus
from core.executor.executor import OrderExecutor
from core.executor.pending_orders import LimitOrderMatcher, PendingOrderStore
from db.database import (
    atomic_adjust_balance,
    init_database,
    load_sim_balance,
    save_sim_balance,
)
from web.context import AppContext
from web.routes import db_manager, settings


def _run(coro):
    return asyncio.run(coro)


class _FakeRisk:
    def __init__(self):
        self.balance = 10000.0

    def update_balance(self, balance):
        self.balance = balance


def _scalars(db_path, sql, params=()):
    db = sqlite3.connect(db_path)
    try:
        return db.execute(sql, params).fetchall()
    finally:
        db.close()


def _scalar(db_path, sql, params=()):
    return _scalars(db_path, sql, params)[0][0]


def _exec(db_path, sql, params=()):
    db = sqlite3.connect(db_path)
    try:
        db.execute(sql, params)
        db.commit()
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


def _assert_identity(db_path, label=""):
    delta = _delta(db_path)
    assert abs(delta) < 1e-9, (
        f"ledger drift after {label}: balance={_run(load_sim_balance(db_path))!r} "
        f"identity={_identity(db_path)!r} delta={delta!r}")


@pytest.fixture()
def harness(tmp_path):
    db_path = str(tmp_path / "money_concurrency.db")
    _run(init_database(db_path))
    _run(save_sim_balance(10000.0, db_path))

    config = Config()
    config.db_path = db_path
    config.mode = "sim"

    bus = EventBus()
    executor = OrderExecutor(config, bus)
    executor.wire_risk_manager(_FakeRisk())
    return {"db_path": db_path, "executor": executor, "bus": bus, "config": config}


def _open(h, symbol, price, qty, side="long", order_type="market", **extra):
    data = {"symbol": symbol, "side": side, "price": price, "quantity": qty,
            "amount_usdt": price * qty, "order_type": order_type,
            "strategy": "manual", "trader": "manual"}
    data.update(extra)
    return _run(h["executor"]._execute_sim(data))


def _credit(db_path, result):
    _run(atomic_adjust_balance(result["invested_returned"] + result["pnl"], db_path))


# ======================================================================
# S3 — two simultaneous 50% reduces of ONE position
# ======================================================================
def test_two_concurrent_50pct_reduces_book_one_row_and_move_no_extra_cash(harness):
    """Both calls used to read the same basis and both credit it.

    Each computed `close_qty = original_qty × 50%` from the same pre-mutation
    position, so 100% of the basis was handed back twice while only 50% of the
    quantity left the books, and two reduce rows landed in one trade_group.
    """
    _open(harness, "ETHUSDT", price=2000.0, qty=0.5)          # 1000 USDT basis
    _assert_identity(harness["db_path"], "open")

    async def _both():
        return await asyncio.gather(
            harness["executor"].close_position("ETHUSDT", 50, 2000.0),
            harness["executor"].close_position("ETHUSDT", 50, 2000.0),
        )

    first, second = _run(_both())
    ok = [r for r in (first, second) if r.get("ok")]
    refused = [r for r in (first, second) if not r.get("ok")]

    assert len(ok) == 1, f"exactly one reduce may be applied, got {first} / {second}"
    assert len(refused) == 1 and refused[0].get("error"), refused[0]
    assert _scalar(harness["db_path"],
                   "SELECT COUNT(*) FROM trades WHERE action='reduce'") == 1
    assert _scalar(harness["db_path"],
                   "SELECT COUNT(DISTINCT trade_group) FROM trades "
                   "WHERE action='reduce'") == 1

    _credit(harness["db_path"], ok[0])
    _assert_identity(harness["db_path"], "one of two concurrent 50% reduces")

    # The remainder is still a real, closable position.
    rest = _run(harness["executor"].close_position("ETHUSDT", 100, 2000.0))
    assert rest["ok"] and rest["closed"]
    _credit(harness["db_path"], rest)
    assert _open_notional(harness["db_path"]) == 0.0
    _assert_identity(harness["db_path"], "close of the reduced remainder")


# ======================================================================
# S12 — two simultaneous 100% closes
# ======================================================================
def test_two_concurrent_full_closes_do_not_raise_and_book_one_row(harness):
    """The loser used to hit `del self._positions[symbol]` on a missing key."""
    _open(harness, "BTCUSDT", price=100.0, qty=1.0)

    async def _both():
        return await asyncio.gather(
            harness["executor"].close_position("BTCUSDT", 100, 100.0),
            harness["executor"].close_position("BTCUSDT", 100, 100.0),
            return_exceptions=True,
        )

    results = _run(_both())
    for r in results:
        assert not isinstance(r, BaseException), f"close_position raised: {r!r}"
    ok = [r for r in results if r.get("ok")]
    assert len(ok) == 1, results
    assert _scalar(harness["db_path"],
                   "SELECT COUNT(*) FROM trades WHERE action='close'") == 1
    assert _open_notional(harness["db_path"]) == 0.0
    _credit(harness["db_path"], ok[0])
    _assert_identity(harness["db_path"], "one of two concurrent full closes")


# ======================================================================
# S6 — two simultaneous opens of the same symbol
# ======================================================================
def test_two_concurrent_opens_of_one_symbol_book_one_row(harness):
    """The duplicate-open guard sat before an `await`, so both passed it.

    Two open rows existed for one snapshot and one in-memory position: after a
    restart the symbol came back as a zombie and the first row's capital was
    stranded for ever.
    """
    symbol = "SOLUSDT"

    async def _both():
        await asyncio.gather(
            harness["executor"]._execute_sim({
                "symbol": symbol, "side": "long", "price": 120.0, "quantity": 2.0,
                "amount_usdt": 240.0, "order_type": "market"}),
            harness["executor"]._execute_sim({
                "symbol": symbol, "side": "long", "price": 120.0, "quantity": 2.0,
                "amount_usdt": 240.0, "order_type": "market"}),
        )

    _run(_both())

    assert _scalar(harness["db_path"],
                   "SELECT COUNT(*) FROM trades WHERE action='open'") == 1
    assert _scalar(harness["db_path"], "SELECT COUNT(*) FROM positions") == 1
    assert len(harness["executor"].get_open_positions()) == 1
    assert _delta(harness["db_path"]) == pytest.approx(0.0, abs=1e-9)
    _assert_identity(harness["db_path"], "one of two concurrent opens")

    # And after a restart the symbol is a single, closable position.
    restarted = OrderExecutor(harness["config"], EventBus())
    restarted.wire_risk_manager(_FakeRisk())
    _run(restarted.restore_positions())
    assert list(restarted.get_open_positions()) == [symbol]
    result = _run(restarted.close_position(symbol, 100, 120.0))
    assert result["ok"] and result["closed"]
    _credit(harness["db_path"], result)
    assert _open_notional(harness["db_path"]) == 0.0
    _assert_identity(harness["db_path"], "close after concurrent-open restart")


def test_concurrent_opens_of_different_symbols_still_run_in_parallel(harness):
    """The lock is per symbol: serialising one symbol must not serialise the book."""
    symbols = [f"SYM{i}USDT" for i in range(8)]

    async def _all():
        await asyncio.gather(*[
            harness["executor"]._execute_sim({
                "symbol": s, "side": "long", "price": 10.0, "quantity": 1.0,
                "amount_usdt": 10.0, "order_type": "market"}) for s in symbols])

    _run(_all())
    assert _scalar(harness["db_path"],
                   "SELECT COUNT(*) FROM trades WHERE action='open'") == len(symbols)
    assert len(harness["executor"].get_open_positions()) == len(symbols)
    _assert_identity(harness["db_path"], "eight concurrent opens")


def test_balance_movement_survives_a_new_event_loop(harness):
    """A cash movement must never fail because the balance lock is loop-bound.

    `asyncio.Lock` binds permanently to the loop in which it first *waited*, so a
    single module-level lock raised ``Lock ... is bound to a different event
    loop`` in the middle of a deduction (8 parallel opens did exactly that).  An
    open whose deduction fails is rolled back, but a *close* whose credit fails
    loses the returned basis entirely.
    """
    db_path = harness["db_path"]

    async def _three_parallel_opens(round_no):
        # Three opens in one loop → the shared balance lock is WAITED on, which is
        # where the loop binding happens.
        await asyncio.gather(*[
            harness["executor"]._execute_sim({
                "symbol": f"L{round_no}S{j}USDT", "side": "long", "price": 10.0,
                "quantity": 1.0, "amount_usdt": 10.0}) for j in range(3)])

    for round_no in range(3):                    # a fresh event loop each time
        _run(_three_parallel_opens(round_no))
        assert len([s for s in harness["executor"].get_open_positions()
                    if s.startswith(f"L{round_no}")]) == 3, "an open was lost"
        _assert_identity(db_path, f"parallel opens in event loop #{round_no}")


# ======================================================================
# Admin routes: /api/db/cleanup, /api/db/row, /api/settings/reset-sim
# ======================================================================
class _AdminUser:
    is_admin = True
    is_trader = True
    username = "admin"


def _make_app(tmp_path, name="routes.db"):
    """A minimal FastAPI app exposing only the routes under test.

    Auth is injected by middleware (the route bodies read ``request.state.user``),
    so no login round trip is needed and nothing else in ``web/`` is involved.
    """
    db_path = str(tmp_path / name)
    _run(init_database(db_path))
    _run(save_sim_balance(10000.0, db_path))

    config = Config()
    config.db_path = db_path
    config.mode = "sim"
    bus = EventBus()
    app = FastAPI()
    app.state.config = config
    app.state.event_bus = bus
    app.state.balance = 10000.0
    executor = OrderExecutor(config, bus)
    executor.wire_risk_manager(_FakeRisk())
    app.state.executor = executor
    app.state.get_price = lambda symbol: None

    ctx = AppContext(config, bus, app)
    db_manager.register(app, ctx)
    settings.register(app, ctx)

    @app.middleware("http")
    async def _inject_admin(request, call_next):
        request.state.user = _AdminUser()
        return await call_next(request)

    return {"app": app, "db_path": db_path, "executor": executor,
            "config": config, "bus": bus, "client": TestClient(app)}


def test_cleanup_deletes_old_rows_and_keeps_the_ledger_identity(tmp_path):
    """The retention DELETE removed realised PnL but never moved the balance.

    Eight old exit rows carrying PnL moved the ledger off by their whole
    realised sum (Δ −7.17 measured by the audit on the live copy).
    """
    h = _make_app(tmp_path, "cleanup.db")
    db_path = h["db_path"]

    db = sqlite3.connect(db_path)
    try:
        db.execute(
            "INSERT INTO trades (symbol, side, entry_price, quantity, pnl, pnl_pct,"
            " strategy, timeframe, status, action, trade_group, opened_at, closed_at)"
            " VALUES ('LIVEUSDT','long',1.0,200.0,0,0,'auto','1h','open','open',"
            " 'live','2020-01-01 00:00:00',NULL)")
        for i in range(8):
            db.execute(
                "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity,"
                " pnl, pnl_pct, strategy, timeframe, status, action, trade_group,"
                " opened_at, closed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"OLD{i}USDT", "long", 1.0, 1.1, 1.0, 0.9 + i * 0.1, 1.0, "auto",
                 "1h", "closed", "close", f"tg{i}", "2020-01-01 00:00:00",
                 "2020-01-02 00:00:00"))
        # An undatable old exit: never guessed at, never deleted.
        db.execute(
            "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity,"
            " pnl, pnl_pct, strategy, timeframe, status, action, trade_group,"
            " opened_at, closed_at) VALUES ('NODATEUSDT','long',1.0,1.1,1.0,1.0,1.0,"
            " 'auto','1h','closed','close','tgN','2020-01-01 00:00:00',NULL)")
        db.commit()
    finally:
        db.close()
    # Start from a reconciled ledger (open notional + realised of the crafted rows).
    _run(save_sim_balance(_identity(db_path), db_path))
    _assert_identity(db_path, "crafted pre-cleanup state")

    response = h["client"].post("/api/db/cleanup")
    assert response.status_code == 200, response.text

    # Every expired, PnL-carrying exit row is gone …
    assert _scalar(db_path, "SELECT COUNT(*) FROM trades WHERE symbol LIKE 'OLD%'") == 0
    # … the live entry leg and the undatable exit are kept …
    assert _scalar(db_path, "SELECT COUNT(*) FROM trades WHERE symbol='LIVEUSDT'") == 1
    assert _scalar(db_path, "SELECT COUNT(*) FROM trades WHERE symbol='NODATEUSDT'") == 1
    # … and the ledger still reconciles: the deleted PnL was taken off the balance.
    _assert_identity(db_path, "cleanup with realised PnL deleted")

    recon = _scalars(db_path, "SELECT reason, balance_before, balance_after, delta,"
                              " expected_balance FROM ledger_reconciliation")
    assert len(recon) == 1, recon
    reason, before, after, delta, expected = recon[0]
    assert reason == "db_cleanup_retention"
    assert before - after == pytest.approx(sum(0.9 + i * 0.1 for i in range(8)), abs=1e-9)
    assert delta == pytest.approx(0.0, abs=1e-9)
    assert after == pytest.approx(expected, abs=1e-9)


def test_cleanup_without_a_balance_row_refuses_to_delete_pnl(tmp_path):
    """No ledger to adjust → delete nothing (better than an unreconcilable DB)."""
    h = _make_app(tmp_path, "cleanup_refuse.db")
    db_path = h["db_path"]
    _exec(db_path,
          "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl,"
          " pnl_pct, strategy, timeframe, status, action, trade_group, opened_at,"
          " closed_at) VALUES ('OLDXUSDT','long',1,2,1,5.0,1.0,'auto','1h','closed',"
          " 'close','tgX','2020-01-01 00:00:00','2020-01-02 00:00:00')")
    _exec(db_path, "DELETE FROM system_config WHERE key='sim_balance'")

    response = h["client"].post("/api/db/cleanup")
    assert response.status_code == 500
    assert _scalar(db_path, "SELECT COUNT(*) FROM trades WHERE symbol='OLDXUSDT'") == 1


def test_row_delete_refuses_a_row_the_ledger_depends_on(tmp_path):
    """Deleting an open row or a PnL row silently broke the balance identity.

    The audit measured Δ +0.11 by deleting one exit row through
    ``DELETE /api/db/row/trades/{id}``; the endpoint deleted anything.
    """
    h = _make_app(tmp_path, "row_delete.db")
    db_path = h["db_path"]
    client = h["client"]
    _open(h, "ADAUSDT", price=0.25, qty=400.0)
    _assert_identity(db_path, "open")

    open_id = _scalar(db_path, "SELECT id FROM trades WHERE action='open'")
    response = client.delete(f"/api/db/row/trades/{open_id}")
    assert response.status_code == 400, response.text
    assert "ledger" in response.json()["error"]
    assert _scalar(db_path, "SELECT COUNT(*) FROM trades WHERE id=?", (open_id,)) == 1
    _assert_identity(db_path, "refused open-row delete")

    result = _run(h["executor"].close_position("ADAUSDT", 100, 0.25))
    assert result["ok"] and result["closed"]
    _credit(db_path, result)
    _assert_identity(db_path, "closed after refused delete")

    close_id = _scalar(db_path, "SELECT id FROM trades WHERE action='close'")
    response = client.delete(f"/api/db/row/trades/{close_id}")
    assert response.status_code == 400, response.text
    assert _scalar(db_path, "SELECT COUNT(*) FROM trades WHERE id=?", (close_id,)) == 1
    _assert_identity(db_path, "refused PnL-row delete")

    # A closed, PnL-free entry leg is identity-neutral: still deletable.
    entry_id = _scalar(db_path, "SELECT id FROM trades WHERE action='open'")
    assert client.delete(f"/api/db/row/trades/{entry_id}").status_code == 200
    _assert_identity(db_path, "PnL-free entry leg deleted")

    # The balance itself is not a deletable row.
    bal_id = _scalar(db_path, "SELECT id FROM system_config WHERE key='sim_balance'")
    response = client.delete(f"/api/db/row/system_config/{bal_id}")
    assert response.status_code == 400, response.text


def test_reset_sim_cancels_pending_orders(tmp_path):
    """A stale limit order used to survive the reset and fill against the fresh 10000."""
    h = _make_app(tmp_path, "reset.db")
    db_path = h["db_path"]
    store = PendingOrderStore(db_path)
    order = _run(store.add("BTCUSDT", "long", 100.0, 1.0, 100.0))
    assert _run(store.open_count()) == 1
    assert _run(store.frozen_total()) == pytest.approx(100.0)

    response = h["client"].post("/api/settings/reset-sim")
    assert response.status_code == 200, response.text

    assert _run(store.open_count()) == 0, "the pending order survived the reset"
    assert _run(store.frozen_total()) == 0.0, "its frozen exposure is still held"
    assert _run(load_sim_balance(db_path)) == pytest.approx(10000.0)
    _assert_identity(db_path, "reset-sim")

    # The matcher has nothing left to fill: a crossed price opens no position on
    # the freshly reset ledger.
    h["app"].state.get_price = lambda symbol: 50.0
    matcher = LimitOrderMatcher(h["app"], h["config"], h["bus"], store, interval=0.01)
    matcher._running = True
    assert asyncio.run(matcher.evaluate_once()) == []
    assert _scalar(db_path, "SELECT COUNT(*) FROM trades") == 0
    _assert_identity(db_path, "reset-sim then matcher pass")


def test_reset_sim_is_atomic_with_the_pending_book(tmp_path):
    """Clearing trades/positions/pending_orders must be one transaction.

    A partial reset (book cleared, orders left) is exactly what let the matcher
    fill an order whose ledger rows no longer existed.
    """
    h = _make_app(tmp_path, "reset_atomic.db")
    db_path = h["db_path"]
    store = PendingOrderStore(db_path)
    for i in range(3):
        _run(store.add(f"SYM{i}USDT", "long", 10.0, 1.0, 10.0))
    _open(h, "BTCUSDT", price=100.0, qty=1.0)

    response = h["client"].post("/api/settings/reset-sim")
    assert response.status_code == 200, response.text

    assert _scalar(db_path, "SELECT COUNT(*) FROM trades") == 0
    assert _scalar(db_path, "SELECT COUNT(*) FROM positions") == 0
    assert _scalar(db_path, "SELECT COUNT(*) FROM pending_orders WHERE status='open'") == 0
    assert h["executor"].get_open_positions() == {}
    assert _run(load_sim_balance(db_path)) == pytest.approx(10000.0)
    _assert_identity(db_path, "atomic reset-sim")


# ======================================================================
# S10 — the engine's exit reason must reach the exit row
# ======================================================================
def test_engine_exit_records_the_real_reason(harness):
    """`_on_position_exit` dropped the audit's `reason`, so every engine exit was
    recorded as `exit_reason='manual'` and "why did this close?" was unanswerable."""
    from app.main import apply_engine_exit

    _open(harness, "BTCUSDT", price=100.0, qty=1.0)
    data = {"symbol": "BTCUSDT", "price": 105.0,
            "reason": "Exit condition met (long) on 1h",
            "strategy": "ga_champion", "trader": "ai"}
    result = _run(apply_engine_exit(harness["executor"], _FakeRisk(),
                                    harness["db_path"], data, 100, "engine_exit"))
    assert result["ok"] and result["closed"]

    exit_reason = _scalar(harness["db_path"],
                          "SELECT exit_reason FROM trades WHERE action='close'")
    assert exit_reason.startswith("Exit condition met (long)"), exit_reason
    assert exit_reason != "manual"
    _assert_identity(harness["db_path"], "engine exit with a real reason")


def test_engine_exit_without_a_reason_uses_the_documented_default(harness):
    from app.main import apply_engine_exit

    _open(harness, "BTCUSDT", price=100.0, qty=1.0)
    data = {"symbol": "BTCUSDT", "price": 99.0}
    result = _run(apply_engine_exit(harness["executor"], _FakeRisk(),
                                    harness["db_path"], data, 100, "engine_exit"))
    assert result["ok"]
    assert _scalar(harness["db_path"],
                   "SELECT exit_reason FROM trades WHERE action='close'") == "engine_exit"
    _assert_identity(harness["db_path"], "engine exit without a reason")


def test_engine_reduce_uses_the_events_reduce_pct_and_reason(harness):
    from app.main import apply_engine_exit

    _open(harness, "ETHUSDT", price=2000.0, qty=0.5)
    data = {"symbol": "ETHUSDT", "price": 2050.0, "reduce_pct": 25,
            "reason": "Reduce 25%: rsi>70"}
    result = _run(apply_engine_exit(harness["executor"], _FakeRisk(),
                                    harness["db_path"], data,
                                    data.get("reduce_pct", 50), "engine_reduce"))
    assert result["ok"] and result["closed"] is False
    row = dict(zip(("reduce_pct", "exit_reason"),
                   _scalars(harness["db_path"],
                            "SELECT reduce_pct, exit_reason FROM trades"
                            " WHERE action='reduce'")[0]))
    assert row["reduce_pct"] == pytest.approx(25.0)
    assert row["exit_reason"].startswith("Reduce 25%")
    _assert_identity(harness["db_path"], "engine reduce")


# ======================================================================
# S8 — the breaker's tighten_stops must persist, not just mutate memory
# ======================================================================
def test_circuit_breaker_tighten_stops_is_persisted_and_survives_a_restart(harness):
    """The protective stop reverted to the pre-breaker value on the next restart."""
    from app.main import persist_tightened_stop

    _open(harness, "BTCUSDT", price=100.0, qty=1.0, stop_loss=95.0)
    pos = harness["executor"].get_open_positions()["BTCUSDT"]

    new_sl = _run(persist_tightened_stop(harness["executor"], "BTCUSDT", pos, 120.0))
    assert new_sl == pytest.approx(117.6)
    assert pos["stop_loss"] == pytest.approx(117.6)

    # Both rows a restart reads were updated, not just the in-memory dict.
    assert _scalar(harness["db_path"],
                   "SELECT stop_loss FROM positions WHERE symbol='BTCUSDT'") == pytest.approx(117.6)
    assert _scalar(harness["db_path"],
                   "SELECT stop_loss FROM trades WHERE symbol='BTCUSDT'"
                   " AND action='open'") == pytest.approx(117.6)

    restarted = OrderExecutor(harness["config"], EventBus())
    _run(restarted.restore_positions())
    assert restarted.get_open_positions()["BTCUSDT"]["stop_loss"] == pytest.approx(117.6), \
        "the tightened stop was lost on restart"
    _assert_identity(harness["db_path"], "tightened stop persisted")


# ======================================================================
# Startup repair for a wallet that never persisted a balance
# ======================================================================
def test_startup_balance_repair_uses_the_ledger_identity(harness):
    """`10000 − invested` dropped the realised-PnL term, leaving the identity
    broken by Σ(close pnl) for ever on a DB with no `sim_balance` row."""
    from app.main import starting_balance_for_unpersisted_wallet

    db_path = harness["db_path"]
    _open(harness, "ADAUSDT", price=0.25, qty=400.0)
    result = _run(harness["executor"].close_position("ADAUSDT", 100, 0.3))
    assert result["ok"]
    _credit(db_path, result)
    _open(harness, "SOLUSDT", price=120.0, qty=2.0)
    _assert_identity(db_path, "state before the repair")

    # The wallet never persisted a balance: the identity is the honest answer.
    _exec(db_path, "DELETE FROM system_config WHERE key='sim_balance'")
    repaired = _run(starting_balance_for_unpersisted_wallet(db_path, _open_notional(db_path)))
    assert repaired == pytest.approx(10000.0 - _open_notional(db_path) + _realised(db_path))
    assert repaired != pytest.approx(10000.0 - _open_notional(db_path)), \
        "the realised PnL term must not be dropped"

    _run(save_sim_balance(repaired, db_path))
    _assert_identity(db_path, "after the startup repair")
