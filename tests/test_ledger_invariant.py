"""P0 regression: the simulated ledger identity must hold to the cent, on every path.

    identity = 10000 − Σ(open rows: quantity × entry_price) + Σ(close rows: pnl)
    invariant: identity == system_config.sim_balance

These tests exist because the invariant drifted by a real −19.87 USDT in production.
Two defects caused it and both are pinned here:

1. **The buy side was charged twice.**  `_execute_sim` deducted the account's cash
   (now the worsened fill notional *plus* the buy fee and slippage) while storing the
   quoted price in `trades.entry_price`, and `close_position` *additionally* took the
   buy fee/slippage out of `pnl`.  The returned `invested_returned` already contained
   the buy cost, so the round trip lost that cost twice and the balance fell below the
   identity by Σ(buy fee + slippage) over closed trades (≈15.2 USDT on the live DB).
2. **A duplicate open orphaned capital.**  `_positions` is keyed by symbol, so a
   second open for a symbol already held overwrote the first, whose open row was never
   returned by any close — stranding that row's whole notional (≈20 USDT for the two
   ADAUSDT round trips the incident report named).

Every test drives the real code paths against a temporary database with fake prices:
no network, and `data/binance_trader.db` is never touched.
"""
import asyncio
import os
import sqlite3
import tempfile

import pytest

from app.config import Config
from app.event_bus import Event, EventBus, EventType
from core.executor.executor import OrderExecutor
from db.database import (
    LedgerUnavailable,
    atomic_adjust_balance,
    init_database,
    load_sim_balance,
    save_sim_balance,
)

PRICE = 100.0
QTY = 1.0


def _run(coro):
    return asyncio.run(coro)


class _FakeRisk:
    """Minimal stand-in so `update_balance` is exercised without a real manager."""

    def __init__(self):
        self.balance = 10000.0

    def update_balance(self, balance):
        self.balance = balance


@pytest.fixture()
def harness(tmp_path):
    db_path = str(tmp_path / "ledger_invariant.db")
    _run(init_database(db_path))
    _run(save_sim_balance(10000.0, db_path))

    config = Config()
    config.db_path = db_path
    config.mode = "sim"

    bus = EventBus()
    executor = OrderExecutor(config, bus)
    executor.wire_risk_manager(_FakeRisk())

    yield {"db_path": db_path, "executor": executor, "bus": bus, "config": config}


def _scalars(db_path, sql, params=()):
    db = sqlite3.connect(db_path)
    try:
        return db.execute(sql, params).fetchall()
    finally:
        db.close()


def _open_notional(db_path):
    row = _scalars(db_path, "SELECT COALESCE(SUM(quantity * entry_price), 0) "
                           "FROM trades WHERE status='open'")[0][0]
    return float(row)


def _realised(db_path):
    row = _scalars(db_path, "SELECT COALESCE(SUM(pnl), 0) "
                           "FROM trades WHERE status='closed'")[0][0]
    return float(row)


def _assert_identity(db_path, label=""):
    """The one invariant: 10000 − open notional + realised == sim_balance."""
    identity = 10000.0 - _open_notional(db_path) + _realised(db_path)
    balance = _run(load_sim_balance(db_path))
    assert balance == pytest.approx(identity, abs=1e-9), (
        f"ledger drift after {label}: balance={balance!r} identity={identity!r} "
        f"delta={balance - identity!r}")


def _open(h, symbol, price=PRICE, qty=QTY, side="long", order_type="market"):
    """Route an open exactly as ORDER_REQUEST does in sim mode."""
    _run(h["executor"]._execute_sim({
        "symbol": symbol, "side": side, "price": price, "quantity": qty,
        "amount_usdt": price * qty, "order_type": order_type,
        "strategy": "manual", "trader": "manual"}))


def _credit_close(db_path, result):
    """Credit a close exactly as every caller does: invested_returned + pnl.

    Verbatim the arithmetic in web/routes/trading.py and app/main.py.
    """
    _run(atomic_adjust_balance(result["invested_returned"] + result["pnl"], db_path))


def _manual_close(h, symbol, price=PRICE, reduce_pct=100):
    result = _run(h["executor"].close_position(symbol, reduce_pct, price))
    if result.get("ok"):
        _credit_close(h["db_path"], result)
    return result


# ======================================================================
# (a) a manual open + close round trip
# ======================================================================
def test_manual_open_close_round_trip_keeps_the_identity(harness):
    _assert_identity(harness["db_path"], "start")

    _open(harness, "ADAUSDT", price=0.2526, qty=79.176564)
    _assert_identity(harness["db_path"], "manual open")
    # The open row's notional is exactly the cash that left the account.
    balance = _run(load_sim_balance(harness["db_path"]))
    assert balance == pytest.approx(10000.0 - _open_notional(harness["db_path"]), abs=1e-9)

    result = _manual_close(harness, "ADAUSDT", price=0.2526)
    assert result["ok"] and result["closed"]
    _assert_identity(harness["db_path"], "manual close")
    # Round trip at an unchanged mark loses exactly the two sides' costs.
    assert _run(load_sim_balance(harness["db_path"])) < 10000.0


# ======================================================================
# (b) an engine-style auto close (POSITION_EXIT -> _on_position_exit)
# ======================================================================
def test_engine_style_auto_close_keeps_the_identity(harness):
    async def auto_exit(symbol, price):
        """Verbatim app/main.py::_on_position_exit crediting."""
        result = await harness["executor"].close_position(symbol, 100, price)
        if result.get("ok"):
            await atomic_adjust_balance(
                result.get("invested_returned", 0) + result.get("pnl", 0),
                harness["db_path"])
        return result

    _open(harness, "BTCUSDT", price=83820.0, qty=0.0038136194)
    _assert_identity(harness["db_path"], "engine open")

    result = _run(auto_exit("BTCUSDT", 83820.0))
    assert result["ok"] and result["closed"]
    _assert_identity(harness["db_path"], "engine auto close")


# ======================================================================
# (c) a partial reduce (POSITION_REDUCE -> _on_position_reduce)
# ======================================================================
def test_partial_reduce_keeps_the_identity(harness):
    _open(harness, "ETHUSDT", price=2000.0, qty=0.5)      # 1000 USDT notional
    _assert_identity(harness["db_path"], "open before reduce")

    result = _run(harness["executor"].close_position("ETHUSDT", 50, 2000.0))
    assert result["ok"] and result["closed"] is False
    _credit_close(harness["db_path"], result)
    _assert_identity(harness["db_path"], "50% reduce")
    # The reduce handed back half the basis; the remaining half is still on the books.
    assert result["invested_returned"] == pytest.approx(
        _open_notional(harness["db_path"]), abs=1e-9)

    # Closing the remainder must land on the same identity.
    result = _manual_close(harness, "ETHUSDT", price=2000.0)
    assert result["ok"] and result["closed"]
    _assert_identity(harness["db_path"], "reduce then full close")


# ======================================================================
# (d) a market fill with fees and slippage, closed at a profit
# ======================================================================
def test_market_fill_with_costs_keeps_the_identity(harness):
    _open(harness, "SOLUSDT", price=120.0, qty=2.0)
    _assert_identity(harness["db_path"], "market open")

    result = _manual_close(harness, "SOLUSDT", price=132.0)      # +10%
    assert result["ok"] and result["closed"]
    _assert_identity(harness["db_path"], "market close at +10%")
    # The cost model really was applied — the fix must not have disabled it.
    row = _scalars(harness["db_path"],
                   "SELECT fee, slippage, fill_price FROM trades "
                   "WHERE action='open'")[0]
    assert row[0] and row[0] > 0 and row[1] and row[1] > 0
    # Net PnL is the balance's gain beyond the returned basis, and gross beats it by
    # the exit's own cost only (the buy side is inside the basis).
    assert result["gross_pnl"] - result["pnl"] == pytest.approx(
        result["fee"] + result["slippage"], abs=0.011)


# ======================================================================
# (e) two consecutive round trips, then a limit fill
# ======================================================================
def test_two_consecutive_round_trips_keep_the_identity(harness):
    for i in range(2):
        _open(harness, "XRPUSDT", price=1.55, qty=64.0)
        _assert_identity(harness["db_path"], f"round trip {i} open")
        result = _manual_close(harness, "XRPUSDT", price=1.55)
        assert result["ok"] and result["closed"]
        _assert_identity(harness["db_path"], f"round trip {i} close")

    # A limit fill takes the same ORDER_REQUEST path (pending_orders._fill).
    _open(harness, "BNBUSDT", price=764.0, qty=0.13, order_type="limit")
    _assert_identity(harness["db_path"], "limit fill")
    result = _manual_close(harness, "BNBUSDT", price=764.0)
    assert result["ok"] and result["closed"]
    _assert_identity(harness["db_path"], "limit round trip")


# ======================================================================
# The two defects that produced the historical −19.87, pinned individually
# ======================================================================
def test_open_row_notional_equals_the_cash_deducted(harness):
    """The row's `quantity × entry_price` MUST equal what the account paid.

    This is what the identity is stated against; if the row kept only the quoted
    price, the buy-side cost would be invisible to the ledger and charged twice.
    """
    _open(harness, "BTCUSDT", price=PRICE, qty=QTY)
    balance = _run(load_sim_balance(harness["db_path"]))
    notional = _open_notional(harness["db_path"])
    assert notional == pytest.approx(10000.0 - balance, abs=1e-9)
    # …and that is strictly more than the bare quote, because it includes the costs.
    assert notional > PRICE * QTY
    row = _scalars(harness["db_path"],
                   "SELECT fill_price, fee, slippage FROM trades "
                   "WHERE action='open'")[0]
    assert notional == pytest.approx(QTY * row[0] + row[1] + row[2], abs=1e-9)


def test_a_second_open_for_a_held_symbol_is_rejected_and_moves_no_cash(harness):
    """Regression for the capital stranding that produced the −20 USDT of drift.

    Two ADAUSDT round trips ran in the incident window; the second open overwrote the
    first in `_positions`, so the first open row was never returned by any close.  A
    duplicate open must now book nothing at all.
    """
    _open(harness, "ADAUSDT", price=0.2526, qty=79.176564)
    balance_after_first = _run(load_sim_balance(harness["db_path"]))
    rows_after_first = _scalars(harness["db_path"], "SELECT COUNT(*) FROM trades")[0][0]

    _open(harness, "ADAUSDT", price=0.2537, qty=157.666535)
    # Nothing was deducted and no second open row was written.
    assert _run(load_sim_balance(harness["db_path"])) == pytest.approx(
        balance_after_first, abs=1e-9)
    assert _scalars(harness["db_path"], "SELECT COUNT(*) FROM trades")[0][0] == rows_after_first
    _assert_identity(harness["db_path"], "rejected duplicate open")

    # Closing the position that IS held must still close the books exactly.
    result = _manual_close(harness, "ADAUSDT", price=0.2537)
    assert result["ok"] and result["closed"]
    _assert_identity(harness["db_path"], "close after rejected duplicate")


def test_identity_survives_a_restart_and_restore(harness):
    """A restart must not shift the ledger: `restore_positions` re-derives
    `amount_usdt` from the recorded `entry_price`, so the close still returns the
    basis the open deducted."""
    _open(harness, "BTCUSDT", price=100.0, qty=2.0)
    _assert_identity(harness["db_path"], "before restart")

    # A fresh executor over the same DB == the app restarting.
    restarted = OrderExecutor(harness["config"], EventBus())
    restarted.wire_risk_manager(_FakeRisk())
    _run(restarted.restore_positions())
    _assert_identity(harness["db_path"], "after restore")

    result = _run(restarted.close_position("BTCUSDT", 100, 110.0))
    assert result["ok"] and result["closed"]
    _credit_close(harness["db_path"], result)
    _assert_identity(harness["db_path"], "close after restore")


def test_no_drift_accumulates_over_many_closes(harness):
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
    for i, symbol in enumerate(symbols * 2):
        _open(harness, symbol, price=100.0, qty=0.5)
        _manual_close(harness, symbol, price=100.0 + i)
        _assert_identity(harness["db_path"], f"cycle {i} ({symbol})")
    # Every position is flat and the identity is still exact.
    assert _open_notional(harness["db_path"]) == 0.0
    _assert_identity(harness["db_path"], "final")


# ======================================================================
# S4 — a close must flip the open row even when it has NO trade_group
# ======================================================================
def _rows(db_path, sql="SELECT * FROM trades ORDER BY id", params=()):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(sql, params)]
    finally:
        db.close()


def _exec(db_path, sql, params=()):
    db = sqlite3.connect(db_path)
    try:
        db.execute(sql, params)
        db.commit()
    finally:
        db.close()


def test_close_of_a_group_less_open_row_still_flips_it(harness):
    """`if trade_group:` left the open row counted *and* returned its basis.

    A snapshot-restored or legacy position can carry `trade_group=''`; the full
    close then inserted its exit row, credited `invested_returned + pnl` … and
    left the open row `status='open'`, so `10000 − Σ(open qty×entry) + Σ(pnl)` was
    short by the whole basis (Δ −200 measured by the audit).
    """
    _open(harness, "ADAUSDT", price=0.25, qty=400.0)      # ≈100 USDT basis
    _exec(harness["db_path"], "UPDATE trades SET trade_group='' WHERE action='open'")
    harness["executor"]._positions["ADAUSDT"]["trade_group"] = ""

    result = _manual_close(harness, "ADAUSDT", price=0.25)
    assert result["ok"] and result["closed"]

    open_row = [r for r in _rows(harness["db_path"]) if r["action"] == "open"][0]
    assert open_row["status"] == "closed", "the open row must be flipped even without a group"
    assert open_row["closed_at"] is not None
    assert _open_notional(harness["db_path"]) == 0.0
    _assert_identity(harness["db_path"], "group-less full close")


# ======================================================================
# S5 — restore must validate a `positions` snapshot against its ledger row
# ======================================================================
_ORPHAN_SNAPSHOT = (
    "INSERT OR REPLACE INTO positions (symbol, side, quantity, entry_price,"
    " current_price, fill_price, trade_group, position_type) "
    "VALUES ('ORPHUSDT','long',2.0,115.0,115.0,115.0,'','satellite')")


def test_restore_refuses_a_snapshot_that_has_no_open_trades_row(harness):
    """Closing an invented position credited a basis nothing ever deducted.

    `restore_positions` trusted the snapshot alone; a snapshot with no open
    `trades` row was restored, and the next close returned 2×115 = 230 USDT of
    cash the account never paid (the audit's Δ −230).
    """
    _exec(harness["db_path"], _ORPHAN_SNAPSHOT)
    _assert_identity(harness["db_path"], "before restore (orphan ignored by the identity)")

    bus = EventBus()
    restarted = OrderExecutor(harness["config"], bus)
    restarted.wire_risk_manager(_FakeRisk())
    _run(restarted.restore_positions())

    assert "ORPHUSDT" not in restarted.get_open_positions(), \
        "a snapshot without an open ledger row must not be restored"
    # …and the close that used to invent the basis is refused.
    refused = _run(restarted.close_position("ORPHUSDT", 100, 115.0))
    assert refused["ok"] is False
    _assert_identity(harness["db_path"], "after refusing the orphan snapshot")

    # The operator is told, loudly, through the alert bus.
    alerts = []
    while not bus._queue.empty():
        event = bus._queue.get_nowait()
        if event.type == EventType.ALERT_TRIGGER:
            alerts.append(event.data)
    assert any(a.get("type") == "ledger_snapshot_mismatch" for a in alerts), alerts
    assert any("ORPHUSDT" in str(a.get("message", "")) or
               "ORPHUSDT" in str(a.get("symbol", "")) for a in alerts), alerts


def test_restore_of_a_snapshot_that_disagrees_uses_the_ledger_row(harness):
    """The row is the ledger truth; a snapshot that "remembers" more is ignored."""
    _open(harness, "XRPUSDT", price=1.0, qty=100.0)
    # Corrupt the state copy only: it claims twice the quantity at a flat basis.
    _exec(harness["db_path"],
          "UPDATE positions SET quantity=200.0, entry_price=1.0, fill_price=1.0"
          " WHERE symbol='XRPUSDT'")

    restarted = OrderExecutor(harness["config"], EventBus())
    restarted.wire_risk_manager(_FakeRisk())
    _run(restarted.restore_positions())

    pos = restarted.get_open_positions()["XRPUSDT"]
    row = [r for r in _rows(harness["db_path"]) if r["action"] == "open"][0]
    assert pos["quantity"] == pytest.approx(row["quantity"], abs=1e-9)
    assert pos["entry_price"] == pytest.approx(row["entry_price"], abs=1e-9)
    assert pos["amount_usdt"] == pytest.approx(
        _open_notional(harness["db_path"]), abs=1e-9)

    result = _run(restarted.close_position("XRPUSDT", 100, 1.0))
    assert result["ok"] and result["closed"]
    _credit_close(harness["db_path"], result)
    _assert_identity(harness["db_path"], "close after a mismatched snapshot")


# ======================================================================
# S9 — a migrated database must keep the SAME index set as a fresh one
# ======================================================================
#: The v3-era `trades` the old versioned migration assembled (`trader` with no
#: CHECK, `reduce_pct` TEXT): it does not match the canonical DDL, so
#: `init_database` rebuilds it — and the rebuild DROPs every index with the table.
_LEGACY_TRADES_DDL = """
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL,
    pnl REAL DEFAULT 0,
    pnl_pct REAL DEFAULT 0,
    strategy TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    position_type TEXT DEFAULT 'satellite' CHECK(position_type IN ('core', 'satellite')),
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    status TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed', 'cancelled'))
, trader TEXT DEFAULT 'manual', strategy_name TEXT DEFAULT '', action TEXT DEFAULT 'open',
  reduce_pct TEXT DEFAULT 0, trade_group TEXT DEFAULT '', fill_price REAL, fee REAL,
  slippage REAL);
"""


def _index_names(db_path):
    return {r[0] for r in _scalars(
        db_path, "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")}


def test_init_database_is_idempotent_and_keeps_every_index(tmp_path):
    """`_rebuild_trades` recreated 4 of the 12 indexes, so the first boot after a
    migration lost `idx_trades_symbol/status/opened` — `restore_positions` then
    planned `SCAN trades` for ever, and init#1 != init#2."""
    db_path = str(tmp_path / "idx_idem.db")
    _run(init_database(db_path))
    db = sqlite3.connect(db_path)
    try:
        db.execute("DROP TABLE trades")
        db.executescript(_LEGACY_TRADES_DDL)
        db.executemany(
            "INSERT INTO trades (symbol, side, entry_price, quantity, strategy,"
            " timeframe, status, action, trade_group) VALUES (?,?,?,?,?,?,?,?,?)",
            [(f"SYM{i}USDT", "long", 1.0, 1.0, "manual", "1h",
              "open" if i % 2 else "closed", "open", f"tg{i}") for i in range(500)])
        db.commit()
    finally:
        db.close()

    _run(init_database(db_path))          # ← the rebuild runs on this first boot
    first = _index_names(db_path)

    for name in ("idx_trades_symbol", "idx_trades_status", "idx_trades_opened",
                 "idx_trades_action", "idx_trades_trade_group", "idx_trades_closed_at",
                 "idx_positions_symbol", "idx_pending_status", "idx_pending_symbol",
                 "idx_alerts_level", "idx_alerts_created", "idx_ai_status"):
        assert name in first, f"{name} was dropped by the rebuild and never restored"

    _run(init_database(db_path))          # second boot: identical, still complete
    assert _index_names(db_path) == first

    db = sqlite3.connect(db_path)
    try:
        # The exact query `restore_positions` runs on startup.
        plan = " ".join(str(r[3]) for r in db.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM trades WHERE status='open' ORDER BY id ASC"))
    finally:
        db.close()
    assert "SCAN trades" not in plan, plan
    assert "idx_trades_status" in plan, plan


# ======================================================================
# S11 — the balance read/write must never silently report a full wallet
# ======================================================================
def test_load_sim_balance_raises_instead_of_reporting_a_full_wallet(tmp_path):
    """A missing/corrupt DB used to look like 10000 USDT of untouched capital."""
    with pytest.raises(LedgerUnavailable):
        _run(load_sim_balance(str(tmp_path / "does_not_exist.db")))

    # A readable-but-uninitialised database is a different case: still loud.
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not a sqlite database")
    with pytest.raises(LedgerUnavailable):
        _run(load_sim_balance(str(broken)))


def test_save_sim_balance_uses_the_shared_atomic_path(harness):
    """`save_sim_balance` must take the same lock/transaction as
    `atomic_adjust_balance`: it used to autocommit an absolute value with no
    lock, so it could interleave with an engine open/close and revert it."""
    import db.database as database

    async def _save_while_locked():
        async with database.balance_lock():         # an adjust in flight
            task = asyncio.create_task(
                database.save_sim_balance(4321.0, harness["db_path"]))
            await asyncio.sleep(0.05)
            blocked = not task.done()
        await task
        return blocked

    assert _run(_save_while_locked()) is True, \
        "save_sim_balance ignored the balance lock"
    assert _run(load_sim_balance(harness["db_path"])) == pytest.approx(4321.0)
    # A failed save is reported, not swallowed: the caller must know the balance
    # it just displayed was never persisted.
    with pytest.raises(Exception):
        _run(save_sim_balance(1000.0, str(harness["db_path"] + "/nope/x.db")))

