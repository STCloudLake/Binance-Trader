"""Regression tests for the database/persistence defects fixed by schema v4.

Every defect below was verified against the LIVE database
(``data/binance_trader.db``); these tests pin the fix against a temporary one.

    1. ``trades.closed_at`` was NULL on 1381/1381 rows, so the
       ``/api/db/cleanup`` retention predicate could never match.
    2. ``/api/history/trades`` returned both rows of every round trip, because
       ``close_position`` flips the OPEN row to ``status='closed'`` too.
    3. ``restore_positions`` read ``r.get("stop_loss")`` from a table with no such
       column, so every restart replaced the real stop with a 2% default, and
       take-profits (stored nowhere) were lost.
    4. ``restore_positions`` set ``entry_price = fill_price``, while the cash
       basis recorded at open is ``qty × entry_price`` = the cash deducted.
    5. ``action='reduce'`` rewrote ``quantity`` without ``entry_price``, breaking
       ``qty × entry_price == remaining cash basis``.
    6. ``orders`` / ``risk_events`` / ``ml_models`` / ``news_articles`` /
       ``test_tz`` had zero writers and zero readers, and ``positions`` was
       declared but never written.
    7. A fresh install and a migrated DB disagreed about ``trader``'s CHECK
       constraint (the live DB holds real usernames such as ``ft_9ef30e``).
    8. ``trades(trade_group)`` was unindexed (``SCAN trades``) and WAL / FK
       enforcement were off.

No network, and ``data/binance_trader.db`` is never touched.
"""
import asyncio
import hashlib
import sqlite3

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from core.executor.executor import OrderExecutor
from db.database import (
    DEAD_TABLES,
    SCHEMA_VERSION,
    atomic_adjust_balance,
    get_db,
    init_database,
    load_sim_balance,
    save_sim_balance,
)

VIEWER = ("v4_viewer", "V1ewerPass!")
TRADER = ("v4_trader", "T1aderPass!")
ADMIN = ("v4_admin", "Adm1nPass!")
P = 100.0


def _run(coro):
    return asyncio.run(coro)


class _FakeRisk:
    def __init__(self):
        self.balance = 10000.0

    def update_balance(self, balance):
        self.balance = balance


def _connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    return db


def _scalars(db, sql, params=()):
    return db.execute(sql, params).fetchall()


@pytest.fixture()
def harness(tmp_path):
    db_path = str(tmp_path / "v4.db")
    _run(init_database(db_path))
    _run(save_sim_balance(10000.0, db_path))

    config = Config()
    config.db_path = db_path
    config.mode = "sim"

    executor = OrderExecutor(config, EventBus())
    executor.wire_risk_manager(_FakeRisk())
    yield {"db_path": db_path, "executor": executor, "config": config}


def _open(h, symbol="BTCUSDT", price=P, qty=1.0, **extra):
    payload = {"symbol": symbol, "side": "long", "price": price, "quantity": qty,
               "amount_usdt": price * qty, "order_type": "market",
               "strategy": "manual", "trader": "manual"}
    payload.update(extra)
    _run(h["executor"]._execute_sim(payload))


def _close(h, symbol="BTCUSDT", price=P, reduce_pct=100, reason=None):
    kwargs = {"reason": reason} if reason is not None else {}
    result = _run(h["executor"].close_position(symbol, reduce_pct, price, **kwargs))
    if result.get("ok"):
        _run(atomic_adjust_balance(result["invested_returned"] + result["pnl"],
                                   h["db_path"]))
    return result


def _rows(db_path):
    async def _fetch():
        db = await aiosqlite.connect(db_path)
        db.row_factory = aiosqlite.Row
        try:
            cursor = await db.execute("SELECT * FROM trades ORDER BY id")
            return [dict(r) for r in await cursor.fetchall()]
        finally:
            await db.close()

    return _run(_fetch())


# ======================================================================
# (1) closed_at / exit_reason are written on close and on reduce
# ======================================================================
def test_closed_at_and_exit_reason_are_written_on_close(harness):
    _open(harness)
    _close(harness, reason="stop_loss")

    rows = _rows(harness["db_path"])
    assert len(rows) == 2
    for row in rows:
        assert row["closed_at"], f"closed_at still NULL on {row['action']}"
    close_row = [r for r in rows if r["action"] == "close"][0]
    open_row = [r for r in rows if r["action"] == "open"][0]
    assert close_row["exit_reason"] == "stop_loss"
    # The entry leg carries the exit reason too: it is the row the history
    # endpoint and the retention predicate read.
    assert open_row["exit_reason"] == "stop_loss"
    assert open_row["status"] == "closed"


def test_closed_at_and_exit_reason_are_written_on_reduce(harness):
    _open(harness, qty=5.0)                      # 500 USDT, above the MIN_NOTIONAL guard
    result = _close(harness, reduce_pct=50, reason="take_profit")
    assert result["ok"] and result["closed"] is False

    reduce_row = [r for r in _rows(harness["db_path"]) if r["action"] == "reduce"][0]
    assert reduce_row["closed_at"]
    assert reduce_row["exit_reason"] == "take_profit"
    assert float(reduce_row["reduce_pct"]) == 50.0


def test_closed_at_defaults_to_manual_when_no_reason_is_given(harness):
    _open(harness)
    _close(harness)
    close_row = [r for r in _rows(harness["db_path"]) if r["action"] == "close"][0]
    assert close_row["exit_reason"] == "manual"


# ======================================================================
# (1b) backfill correctness on a pre-v4 database
# ======================================================================
_LEGACY_TRADES_DDL = """
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL,
    pnl REAL DEFAULT 0,
    pnl_pct REAL DEFAULT 0,
    strategy TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    position_type TEXT DEFAULT 'satellite',
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    status TEXT DEFAULT 'open',
    trader TEXT DEFAULT 'manual',
    strategy_name TEXT DEFAULT '',
    action TEXT DEFAULT 'open',
    reduce_pct TEXT DEFAULT 0,
    trade_group TEXT DEFAULT '',
    fill_price REAL,
    fee REAL,
    slippage REAL
);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_opened ON trades(opened_at);
"""

_LEGACY_POSITIONS_DDL = """
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL DEFAULT 0,
    stop_loss REAL,
    take_profit_1 REAL,
    take_profit_2 REAL,
    take_profit_3 REAL,
    position_type TEXT DEFAULT 'satellite',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_LEGACY_DEAD_DDL = "\n".join(
    f"CREATE TABLE {t} (id INTEGER PRIMARY KEY AUTOINCREMENT, note TEXT);"
    for t in DEAD_TABLES)


def _legacy_db(path, extra_rows=()):
    """Build a v1/v2-flavoured database with 2 round trips and 1 open position."""
    db = _connect(path)
    db.executescript(_LEGACY_TRADES_DDL)
    db.executescript(_LEGACY_POSITIONS_DDL)
    db.executescript(_LEGACY_DEAD_DDL)
    for row in extra_rows:
        db.execute(row)
    db.commit()
    db.close()


def test_closed_at_backfill_matches_the_exit_row(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    _legacy_db(db_path, extra_rows=(
        # round trip A: open + close, both with closed_at NULL
        "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl,"
        " pnl_pct, strategy, timeframe, status, action, trade_group, opened_at)"
        " VALUES ('ADAUSDT','long',1.0,1.1,100,10.0,10.0,'auto','1h','open','open',"
        " 'aaa','2026-01-01 00:00:00')",
        "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl,"
        " pnl_pct, strategy, timeframe, status, action, trade_group, opened_at)"
        " VALUES ('ADAUSDT','long',1.0,1.1,100,10.0,10.0,'auto','1h','closed','close',"
        " 'aaa','2026-01-02 03:04:05')",
        # round trip B: a row written BEFORE `action` existed (defaults to 'open')
        "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl,"
        " pnl_pct, strategy, timeframe, status, trade_group, opened_at)"
        " VALUES ('SOLUSDT','long',50.0,55.0,2,10.0,10.0,'auto','1h','closed','bbb',"
        " '2026-02-01 00:00:00')",
        # a position that is still open: closed_at must STAY NULL
        "INSERT INTO trades (symbol, side, entry_price, quantity, strategy, timeframe,"
        " status, action, trade_group, opened_at)"
        " VALUES ('XRPUSDT','long',2.0,10,'auto','1h','open','open','ccc',"
        " '2026-03-01 00:00:00')",
    ))

    _run(init_database(db_path))
    rows = {r["symbol"]: r for r in _rows(db_path)}

    # The open leg of round trip A takes the exit leg's timestamp, not its own.
    assert rows["ADAUSDT"]["closed_at"] == "2026-01-02 03:04:05"
    assert rows["SOLUSDT"]["closed_at"] == "2026-02-01 00:00:00"
    assert rows["XRPUSDT"]["closed_at"] is None
    # The pre-`action` exit row is reclassified, so the history endpoint (and the
    # retention delete) can see it as a real exit.
    assert rows["SOLUSDT"]["action"] == "close"
    assert rows["SOLUSDT"]["exit_reason"] == "legacy"
    assert rows["ADAUSDT"]["exit_reason"] == "legacy"
    # The still-open position is left alone.
    assert rows["XRPUSDT"]["action"] == "open"
    assert rows["XRPUSDT"]["exit_reason"] is None
    # No row was lost or duplicated.
    assert len(_rows(db_path)) == 4


def test_backfill_is_replayable_and_does_not_change_a_filled_closed_at(tmp_path):
    db_path = str(tmp_path / "replay.db")
    _legacy_db(db_path, extra_rows=(
        # A row that ALREADY carries its own closed_at.
        "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl,"
        " pnl_pct, strategy, timeframe, status, action, trade_group, opened_at,"
        " closed_at) VALUES ('ADAUSDT','long',1.0,1.1,100,10.0,10.0,'auto','1h',"
        " 'closed','close','aaa','2026-01-02 03:04:05','2020-05-05 05:05:05')",
    ))

    _run(init_database(db_path))
    first = _rows(db_path)[0]
    assert first["closed_at"] == "2020-05-05 05:05:05", \
        "an existing closed_at must never be overwritten by the backfill"
    assert first["exit_reason"] == "legacy"

    _run(init_database(db_path))
    again = _rows(db_path)[0]
    for key in first:
        assert again[key] == first[key], f"replay modified {key}"


# ======================================================================
# (2) /api/history/trades returns real exits only
# ======================================================================
@pytest.fixture(scope="module")
def web_app(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("bt_db_v4")
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "web.db")
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")

    async def _setup():
        await init_database(config.db_path)
        await save_sim_balance(10000.0, config.db_path)
        am = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        for username, password, role in ((VIEWER[0], VIEWER[1], "viewer"),
                                         (TRADER[0], TRADER[1], "trader"),
                                         (ADMIN[0], ADMIN[1], "admin")):
            await am.create_user(username, password, role, username)
        return am

    auth = _run(_setup())
    from web.server import create_app
    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    app.state.balance = 10000.0
    app.state.get_price = lambda symbol: None
    yield app

    matcher = getattr(app.state, "limit_order_matcher", None)
    if matcher is not None:
        _run(matcher.stop())


@pytest.fixture()
def client(web_app):
    c = TestClient(web_app)
    assert c.post("/api/auth/login",
                  json={"username": VIEWER[0], "password": VIEWER[1]}).status_code == 200
    return c


@pytest.fixture()
def admin_client(web_app):
    c = TestClient(web_app)
    assert c.post("/api/auth/login",
                  json={"username": ADMIN[0], "password": ADMIN[1]}).status_code == 200
    return c


def test_cleanup_retention_deletes_only_old_real_exits(web_app, admin_client):
    """The predicate the NULL `closed_at` used to make a no-op, end to end.

    Before v4 this DELETE matched 0 rows for ever: every `closed_at` was NULL, so
    no closed trade was ever cleaned up.
    """
    db_path = web_app.state.config.db_path
    db = _connect(db_path)
    db.execute("DELETE FROM trades")
    db.executemany(
        "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl,"
        " pnl_pct, strategy, timeframe, status, action, trade_group, opened_at,"
        " closed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            # old, real exit → deleted
            ("OLDUSDT", "long", 1, 2, 1, 1.0, 1.0, "auto", "1h", "closed", "close",
             "t1", "2020-01-01 00:00:00", "2020-01-02 00:00:00"),
            # recent, real exit → kept
            ("NEWUSDT", "long", 1, 2, 1, 1.0, 1.0, "auto", "1h", "closed", "close",
             "t2", "2026-01-01 00:00:00", "2030-01-02 00:00:00"),
            # old entry leg of a SHORT round trip that is still open → kept
            ("LIVEUSDT", "long", 1, None, 1, 0.0, 0.0, "auto", "1h", "open", "open",
             "t3", "2020-01-01 00:00:00", None),
            # a 13-month-old exit with NO closed_at → kept (undatable, never guessed)
            ("NODATEUSDT", "long", 1, 2, 1, 1.0, 1.0, "auto", "1h", "closed", "close",
             "t4", "2020-01-01 00:00:00", None),
        ])
    db.commit()
    db.close()

    assert admin_client.post("/api/db/cleanup").status_code == 200

    db = _connect(db_path)
    try:
        left = {r[0] for r in _scalars(db, "SELECT symbol FROM trades")}
    finally:
        db.close()
    assert left == {"NEWUSDT", "LIVEUSDT", "NODATEUSDT"}


def _phantom_count(db_path):
    db = _connect(db_path)
    try:
        return _scalars(db, "SELECT COUNT(*) FROM trades "
                            "WHERE status='closed' AND action='open'")[0][0]
    finally:
        db.close()


def test_history_endpoint_excludes_the_entry_leg(web_app, client):
    """A round trip is ONE history row, not two.

    The open row is flipped to ``status='closed'`` by ``close_position``, so the
    old ``WHERE status='closed'`` returned the entry leg as well (688 of 1376
    closed rows on the live DB, 24 of the newest 50 with ``exit_price NULL``).
    """
    db_path = web_app.state.config.db_path
    db = _connect(db_path)
    db.execute("DELETE FROM trades")
    db.commit()
    db.close()

    executor = OrderExecutor(web_app.state.config, web_app.state.event_bus)
    executor.wire_risk_manager(_FakeRisk())
    _run(save_sim_balance(10000.0, db_path))
    _run(executor._execute_sim({
        "symbol": "BTCUSDT", "side": "long", "price": P, "quantity": 1.0,
        "amount_usdt": P, "order_type": "market"}))
    result = _run(executor.close_position("BTCUSDT", 100, P * 1.05))
    assert result["ok"]

    # Two closed rows in the table …
    assert _phantom_count(db_path) == 1
    # … but only the exit is history.
    hist = client.get("/api/history/trades").json()["trades"]
    assert len([t for t in hist if t["symbol"] == "BTCUSDT"]) == 1
    entry = hist[0]
    assert entry["exit_price"] is not None
    assert entry["net_pnl"] is not None
    # The documented response shape (and every key the trade page reads) is kept.
    assert set(entry) == {"id", "symbol", "side", "entry_price", "exit_price",
                          "quantity", "pnl", "pnl_pct", "strategy", "exit_reason",
                          "opened_at", "closed_at", "fee", "slippage",
                          "fill_price", "net_pnl"}


def test_history_endpoint_still_returns_a_pre_action_exit(web_app, client):
    """Rows written before the `action` column existed are genuine exits.

    They default to ``action='open'``, so filtering on `action` alone would have
    hidden all pre-v4 history; the ``exit_price IS NOT NULL`` arm keeps them.
    """
    db_path = web_app.state.config.db_path
    db = _connect(db_path)
    db.execute("DELETE FROM trades")
    db.execute(
        "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl,"
        " pnl_pct, strategy, status, opened_at, closed_at) "
        "VALUES ('ETHUSDT','long',3000,3100,0.1,10.0,3.33,'manual','closed',"
        " '2026-01-02 00:00:00','2026-01-02 04:00:00')")
    db.commit()
    db.close()

    hist = client.get("/api/history/trades").json()["trades"]
    assert [t["symbol"] for t in hist] == ["ETHUSDT"]
    assert hist[0]["exit_price"] == 3100


# ======================================================================
# (3) stop-loss + take-profits survive a restart
# ======================================================================
def test_stop_loss_and_take_profits_survive_a_restart(harness):
    _open(harness, stop_loss=83_500.0, take_profits=[86_000.0, 90_000.0])
    assert harness["executor"].get_open_positions()["BTCUSDT"]["stop_loss"] == 83_500.0

    restarted = OrderExecutor(harness["config"], EventBus())
    restarted.wire_risk_manager(_FakeRisk())
    _run(restarted.restore_positions())
    pos = restarted.get_open_positions()["BTCUSDT"]

    assert pos["stop_loss"] == 83_500.0, "the stop-loss was replaced by a default"
    assert pos["entry_stop_loss"] == 83_500.0
    assert pos["take_profits"] == [86_000.0, 90_000.0], "take-profits were lost"

    # …and they are in the ledger row too, so a DB-only reader sees them.
    open_row = [r for r in _rows(harness["db_path"])
                if r["action"] == "open"][0]
    assert open_row["stop_loss"] == 83_500.0
    assert open_row["take_profits"] == "[86000.0, 90000.0]"


def test_trailing_stop_move_is_persisted_and_restored(harness):
    _open(harness, stop_loss=96.0)
    _run(harness["executor"].update_stop_loss("BTCUSDT", 101.5))

    db = _connect(harness["db_path"])
    try:
        row = _scalars(db, "SELECT stop_loss FROM positions WHERE symbol='BTCUSDT'")[0]
        trade = _scalars(db, "SELECT stop_loss FROM trades WHERE action='open'")[0]
    finally:
        db.close()
    assert row["stop_loss"] == 101.5
    assert trade["stop_loss"] == 101.5

    restarted = OrderExecutor(harness["config"], EventBus())
    _run(restarted.restore_positions())
    assert restarted.get_open_positions()["BTCUSDT"]["stop_loss"] == 101.5


def test_legacy_open_row_without_a_persisted_stop_gets_the_documented_default(tmp_path):
    """A row from before v4 has no stop to restore: the 2% default is the
    documented fallback, not a silent replacement of a real stop."""
    db_path = str(tmp_path / "legacy_stop.db")
    _run(init_database(db_path))
    _run(save_sim_balance(10000.0, db_path))
    db = _connect(db_path)
    db.execute("INSERT INTO trades (symbol, side, entry_price, quantity, strategy,"
               " timeframe, status, action, trade_group) VALUES"
               " ('BTCUSDT','long',100.0,1.0,'manual','1h','open','open','tg1')")
    db.commit()
    db.close()

    config = Config()
    config.db_path = db_path
    executor = OrderExecutor(config, EventBus())
    _run(executor.restore_positions())
    pos = executor.get_open_positions()["BTCUSDT"]
    assert pos["stop_loss"] == 98.0
    assert pos["entry_price"] == 100.0


# ======================================================================
# (4) the cash basis survives a restart
# ======================================================================
def test_restart_preserves_the_cash_basis_and_the_ledger_identity(harness):
    db_path = harness["db_path"]
    _open(harness, price=100.0, qty=2.0)
    open_row = [r for r in _rows(db_path) if r["action"] == "open"][0]
    basis = open_row["quantity"] * open_row["entry_price"]
    # The open row's notional is the cash the account actually paid.
    assert _run(load_sim_balance(db_path)) == pytest.approx(10000.0 - basis, abs=1e-9)
    assert basis > 200.0, "the basis must include the buy-side costs"

    restarted = OrderExecutor(harness["config"], EventBus())
    restarted.wire_risk_manager(_FakeRisk())
    _run(restarted.restore_positions())
    pos = restarted.get_open_positions()["BTCUSDT"]
    # NOT the fill price: the recorded basis, so `qty × entry_price` is unchanged.
    assert pos["entry_price"] == pytest.approx(open_row["entry_price"])
    assert pos["entry_price"] != pytest.approx(open_row["fill_price"])
    assert pos["quantity"] * pos["entry_price"] == pytest.approx(basis, abs=1e-9)
    assert pos["amount_usdt"] == pytest.approx(basis, abs=1e-9)

    # A close after the restart returns exactly that basis.
    result = _run(restarted.close_position("BTCUSDT", 100, 110.0))
    assert result["ok"] and result["closed"]
    assert result["invested_returned"] == pytest.approx(basis, abs=1e-9)
    # The identity the P0 ledger fix established still holds after a restart.
    realised = sum(r["pnl"] or 0 for r in _rows(db_path) if r["status"] == "closed")
    balance = _run(atomic_adjust_balance(result["invested_returned"] + result["pnl"],
                                         db_path))
    assert balance == pytest.approx(10000.0 + realised, abs=1e-6)


# ======================================================================
# (5) reduce keeps `qty × entry_price` == the remaining cash basis
# ======================================================================
def test_reduce_keeps_the_cash_basis(harness):
    db_path = harness["db_path"]
    _open(harness, price=100.0, qty=5.0)
    open_row = [r for r in _rows(db_path) if r["action"] == "open"][0]
    full_basis = open_row["quantity"] * open_row["entry_price"]

    result = _run(harness["executor"].close_position("BTCUSDT", 50, 100.0))
    assert result["ok"] and result["closed"] is False

    db = _connect(db_path)
    try:
        left = _scalars(db, "SELECT quantity, entry_price FROM trades "
                            "WHERE action='open' AND status='open'")[0]
    finally:
        db.close()
    pos = harness["executor"].get_open_positions()["BTCUSDT"]

    assert pos["quantity"] == pytest.approx(2.5)
    # The rule the P0 fix established for open/close, now for reduce:
    # `qty × entry_price` == the cash basis still invested.
    assert left["quantity"] * left["entry_price"] == pytest.approx(full_basis / 2, abs=1e-9)
    assert pos["quantity"] * pos["entry_price"] == pytest.approx(full_basis / 2, abs=1e-9)
    assert pos["amount_usdt"] == pytest.approx(full_basis / 2, abs=1e-9)
    assert result["invested_returned"] == pytest.approx(full_basis / 2, abs=1e-9)

    # The buy fee charged at open is SPLIT across the rows that now describe the
    # two halves of the position, not duplicated: the open row keeps half (it is
    # the half still invested) and the exit row carries its own sell-side fee.
    rows = _rows(db_path)
    open_row_after = [r for r in rows if r["action"] == "open"][0]
    reduce_row = [r for r in rows if r["action"] == "reduce"][0]
    assert open_row_after["fee"] == pytest.approx(open_row["fee"] / 2, abs=1e-9)
    assert reduce_row["fee"] > 0
    # The buy side is never charged twice: everything that remains of the buy fee
    # plus what was handed back is exactly what was charged once.
    assert open_row_after["fee"] + open_row["fee"] / 2 == pytest.approx(
        open_row["fee"], abs=1e-9)
    # `slippage` is apportioned the same way.
    assert open_row_after["slippage"] == pytest.approx(open_row["slippage"] / 2,
                                                       abs=1e-9)

    # And the ledger closes on the cent.
    _run(atomic_adjust_balance(result["invested_returned"] + result["pnl"], db_path))
    balance = _run(load_sim_balance(db_path))
    db = _connect(db_path)
    try:
        notional = _scalars(db, "SELECT COALESCE(SUM(quantity * entry_price), 0) "
                                "FROM trades WHERE status='open'")[0][0]
        realised = _scalars(db, "SELECT COALESCE(SUM(pnl), 0) FROM trades "
                                "WHERE status='closed'")[0][0]
    finally:
        db.close()
    assert balance == pytest.approx(10000.0 - notional + realised, abs=1e-6)

    # A restart in the middle of a reduced position restores the REMAINDER.
    restarted = OrderExecutor(harness["config"], EventBus())
    _run(restarted.restore_positions())
    restored = restarted.get_open_positions()["BTCUSDT"]
    assert restored["quantity"] == pytest.approx(2.5)
    assert restored["quantity"] * restored["entry_price"] == pytest.approx(
        full_basis / 2, abs=1e-9)


# ======================================================================
# (6) dead tables are gone, `positions` is real
# ======================================================================
def test_dead_tables_are_dropped_by_the_migration(tmp_path):
    db_path = str(tmp_path / "dead.db")
    _legacy_db(db_path, extra_rows=(
        "INSERT INTO trades (symbol, side, entry_price, quantity, strategy, timeframe)"
        " VALUES ('BTCUSDT','long',1.0,1.0,'manual','1h')",
    ))
    db = _connect(db_path)
    assert all(_scalars(db, "SELECT 1 FROM sqlite_master WHERE type='table' "
                            f"AND name='{t}'") for t in DEAD_TABLES)
    db.close()

    _run(init_database(db_path))

    db = _connect(db_path)
    try:
        tables = {r[0] for r in _scalars(
            db, "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        db.close()
    for table in DEAD_TABLES:
        assert table not in tables, f"{table} should have been dropped"
    # `news_sources` is alive (core/news/fetcher.py reads it) and must stay.
    assert "news_sources" in tables
    # `positions` was redefined (not dropped): it is the live position snapshot.
    assert "positions" in tables
    # No trade row was lost by the drop.
    assert len(_rows(db_path)) == 1


def test_fresh_install_does_not_create_the_dead_tables(tmp_path):
    db_path = str(tmp_path / "fresh_tables.db")
    _run(init_database(db_path))
    db = _connect(db_path)
    try:
        tables = {r[0] for r in _scalars(
            db, "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        db.close()
    for table in DEAD_TABLES:
        assert table not in tables
    assert {"trades", "positions", "pending_orders", "news_sources",
            "system_config", "users", "alerts", "ai_suggestions",
            "backtest_records", "strategy_lifecycle_events"} <= tables


def test_positions_snapshot_is_written_on_open_and_cleared_on_close(harness):
    _open(harness, stop_loss=95.0)
    db = _connect(harness["db_path"])
    try:
        rows = [dict(r) for r in _scalars(db, "SELECT * FROM positions")]
    finally:
        db.close()
    assert len(rows) == 1
    snap = rows[0]
    assert snap["symbol"] == "BTCUSDT"
    assert snap["stop_loss"] == 95.0
    # The basis, not the fill.
    assert snap["entry_price"] != pytest.approx(snap["fill_price"])
    assert snap["amount_usdt"] == pytest.approx(snap["quantity"] * snap["entry_price"])

    _close(harness)
    db = _connect(harness["db_path"])
    try:
        assert _scalars(db, "SELECT COUNT(*) FROM positions")[0][0] == 0
    finally:
        db.close()


# ======================================================================
# (7) fresh vs migrated schema parity
# ======================================================================
#: The v3-era `trades` exactly as the old versioned migration assembled it:
#: `trader` added by `ALTER TABLE` with NO check, `reduce_pct` as TEXT.
_V3_TRADES_DDL = """
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


@pytest.fixture()
def migrated_db(tmp_path):
    """A database that looks like the live one before v4, then migrated."""
    db_path = str(tmp_path / "migrated.db")
    _run(init_database(db_path))                 # creates the v4 shape + config rows
    db = _connect(db_path)
    db.execute("DROP TABLE trades")
    db.executescript(_V3_TRADES_DDL)
    db.execute("DROP TABLE positions")
    db.executescript(_LEGACY_POSITIONS_DDL)
    db.executescript(_LEGACY_DEAD_DDL)
    db.execute("INSERT INTO trades (symbol, side, entry_price, exit_price, quantity,"
               " pnl, pnl_pct, strategy, timeframe, status, trader, action,"
               " trade_group, reduce_pct, opened_at, closed_at)"
               " VALUES ('ADAUSDT','long',1.0,1.1,100,10.0,10.0,'auto','1h','closed',"
               " 'ft_9ef30e','close','aaa','50','2026-01-02 03:04:05',NULL)")
    db.execute("INSERT INTO trades (symbol, side, entry_price, quantity, strategy,"
               " timeframe, status, trader, action, trade_group, reduce_pct)"
               " VALUES ('XRPUSDT','long',2.0,10,'auto','1h','open','manual','open',"
               " 'ccc','0')")
    # The legacy version mirror AND the (unset) PRAGMA both claim v3/0.
    db.execute("INSERT OR REPLACE INTO system_config (key, value, category)"
               " VALUES ('schema_version','3','system')")
    db.execute("PRAGMA user_version=0")
    db.commit()
    db.close()
    _run(init_database(db_path))                 # ← the migration under test
    return db_path


def _schema_fingerprint(db_path):
    db = _connect(db_path)
    try:
        out = {}
        for table in ("trades", "positions", "pending_orders"):
            out[table] = [tuple(r) for r in _scalars(db, f"PRAGMA table_info({table})")]
        out["trades_checks"] = db.execute(
            "SELECT sql FROM sqlite_master WHERE name='trades'").fetchone()[0]
        return out
    finally:
        db.close()


def test_fresh_and_migrated_schemas_converge(migrated_db, tmp_path):
    fresh = str(tmp_path / "fresh_parity.db")
    _run(init_database(fresh))

    migrated = _schema_fingerprint(migrated_db)
    reference = _schema_fingerprint(fresh)

    assert migrated["trades"] == reference["trades"], (
        "a migrated trades table must have the same columns, types, NOT NULLs and "
        "defaults as a fresh install")
    assert migrated["positions"] == reference["positions"]
    assert migrated["pending_orders"] == reference["pending_orders"]


def test_migrated_db_accepts_a_real_username_as_trader(migrated_db):
    """The divergence that made a fresh install unable to hold live data.

    v1 added ``trader`` with no CHECK, the fresh ``SCHEMA`` had
    ``CHECK(trader IN ('manual','ai'))``, and the live DB holds
    ``trader='ft_9ef30e'`` — so an INSERT copied from the migrated DB failed with
    "CHECK constraint failed".  Both paths now declare the same thing.
    """
    db = _connect(migrated_db)
    try:
        # The migrated row kept its username (nothing was rewritten) …
        assert _scalars(db, "SELECT trader FROM trades WHERE symbol='ADAUSDT'"
                        )[0][0] == "ft_9ef30e"
        # … and a new row with a username is accepted, as on a fresh install.
        db.execute("INSERT INTO trades (symbol, side, entry_price, quantity,"
                   " strategy, timeframe, status, trader, action)"
                   " VALUES ('ZZZUSDT','long',1,1,'manual','1h','open','v4_trader',"
                   " 'open')")
        db.rollback()
    finally:
        db.close()


def test_fresh_install_accepts_a_username_as_trader(tmp_path):
    db_path = str(tmp_path / "fresh_trader.db")
    _run(init_database(db_path))
    db = _connect(db_path)
    try:
        db.execute("INSERT INTO trades (symbol, side, entry_price, quantity,"
                   " strategy, timeframe, status, trader, action)"
                   " VALUES ('ZZZUSDT','long',1,1,'manual','1h','open','ft_9ef30e',"
                   " 'open')")
        db.commit()
        assert _scalars(db, "SELECT COUNT(*) FROM trades")[0][0] == 1
    finally:
        db.close()


def test_reduce_pct_is_numeric_not_text(migrated_db):
    """`reduce_pct TEXT DEFAULT 0` gave the column TEXT affinity, so a migrated DB
    stored '50' where the DDL said REAL.  The value is kept, the class is fixed."""
    db = _connect(migrated_db)
    try:
        name, decl_type = _scalars(db, "SELECT name, type FROM pragma_table_info"
                                       "('trades') WHERE name='reduce_pct'")[0]
        assert name == "reduce_pct"
        assert decl_type.upper() == "REAL"
        assert _scalars(db, "SELECT typeof(reduce_pct) FROM trades"
                            " WHERE symbol='ADAUSDT'")[0][0] == "real"
        assert _scalars(db, "SELECT reduce_pct FROM trades"
                            " WHERE symbol='ADAUSDT'")[0][0] == 50.0
    finally:
        db.close()


def test_migration_preserves_every_trade_row(tmp_path):
    """No row lost, no row renumbered, on a rebuild-heavy migration."""
    db_path = str(tmp_path / "preserve.db")
    _run(init_database(db_path))
    db = _connect(db_path)
    db.execute("DROP TABLE trades")
    db.executescript(_V3_TRADES_DDL)
    for i in range(25):
        db.execute("INSERT INTO trades (symbol, side, entry_price, quantity, pnl,"
                   " strategy, timeframe, status, action, trade_group)"
                   " VALUES (?, 'long', ?, 1.0, ?, 'auto', '1h', 'open', 'open', ?)",
                   (f"SYM{i}USDT", 100.0 + i, float(i), f"tg{i}"))
    db.commit()
    before = [tuple(r) for r in _scalars(
        db, "SELECT id, symbol, entry_price, quantity FROM trades ORDER BY id")]
    db.close()

    _run(init_database(db_path))

    db = _connect(db_path)
    try:
        after = [tuple(r) for r in _scalars(
            db, "SELECT id, symbol, entry_price, quantity FROM trades ORDER BY id")]
    finally:
        db.close()
    assert after == before


def test_schema_version_is_recorded_in_user_version_and_mirrored(tmp_path):
    db_path = str(tmp_path / "version.db")
    _run(init_database(db_path))
    db = _connect(db_path)
    try:
        assert _scalars(db, "PRAGMA user_version")[0][0] == SCHEMA_VERSION
        mirror = _scalars(db, "SELECT value FROM system_config"
                              " WHERE key='schema_version'")[0][0]
        assert int(mirror) == SCHEMA_VERSION
        # The PRAGMA is authoritative: deleting the system_config mirror (which
        # /api/db/row can do) must not make the DB look like an old install.
        db.execute("DELETE FROM system_config WHERE key='schema_version'")
        db.commit()
    finally:
        db.close()
    _run(init_database(db_path))
    db = _connect(db_path)
    try:
        assert _scalars(db, "PRAGMA user_version")[0][0] == SCHEMA_VERSION
        # The mirror is restored, so legacy readers still see the version.
        assert _scalars(db, "SELECT value FROM system_config WHERE key="
                            "'schema_version'")[0][0] == str(SCHEMA_VERSION)
    finally:
        db.close()


def test_every_migration_step_is_replayable(tmp_path):
    """Running the whole chain twice on the same DB is a no-op."""
    db_path = str(tmp_path / "idem.db")
    _legacy_db(db_path, extra_rows=(
        "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl,"
        " pnl_pct, strategy, timeframe, status, action, trade_group, opened_at)"
        " VALUES ('ADAUSDT','long',1.0,1.1,100,10.0,10.0,'auto','1h','open','open',"
        " 'aaa','2026-01-01 00:00:00')",
        "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl,"
        " pnl_pct, strategy, timeframe, status, action, trade_group, opened_at)"
        " VALUES ('ADAUSDT','long',1.0,1.1,100,10.0,10.0,'auto','1h','closed','close',"
        " 'aaa','2026-01-02 03:04:05')",
    ))
    _run(init_database(db_path))
    first = _rows(db_path)
    first_schema = _schema_fingerprint(db_path)
    _run(init_database(db_path))
    _run(init_database(db_path))
    assert _rows(db_path) == first
    assert _schema_fingerprint(db_path) == first_schema


# ======================================================================
# (8) indexes, WAL and foreign keys
# ======================================================================
def test_required_indexes_exist(tmp_path):
    db_path = str(tmp_path / "indexes.db")
    _run(init_database(db_path))
    db = _connect(db_path)
    try:
        indexes = {r[0] for r in _scalars(
            db, "SELECT name FROM sqlite_master WHERE type='index'")}
    finally:
        db.close()
    for name in ("idx_trades_trade_group", "idx_trades_closed_at",
                 "idx_trades_action"):
        assert name in indexes, f"missing index {name}"


def test_trade_group_lookup_uses_the_index_not_a_full_scan(tmp_path):
    """`close_position` looks a round trip up by trade_group; that used to plan
    `SCAN trades` because the column was unindexed."""
    db_path = str(tmp_path / "plan.db")
    _run(init_database(db_path))
    db = _connect(db_path)
    try:
        db.executemany(
            "INSERT INTO trades (symbol, side, entry_price, quantity, strategy,"
            " timeframe, status, action, trade_group) VALUES (?,?,?,?,?,?,?,?,?)",
            [(f"SYM{i}USDT", "long", 1.0, 1.0, "manual", "1h", "open", "open", f"tg{i}")
             for i in range(400)])
        db.commit()
        db.execute("ANALYZE")
        plan = " ".join(str(r[3]) for r in _scalars(
            db, "EXPLAIN QUERY PLAN SELECT * FROM trades WHERE trade_group='tg7'"))
    finally:
        db.close()
    assert "SCAN trades" not in plan, plan
    assert "idx_trades_trade_group" in plan, plan


def test_closed_at_retention_predicate_can_use_an_index(tmp_path):
    db_path = str(tmp_path / "plan2.db")
    _run(init_database(db_path))
    db = _connect(db_path)
    try:
        db.executemany(
            "INSERT INTO trades (symbol, side, entry_price, quantity, strategy,"
            " timeframe, status, action, trade_group, closed_at) VALUES"
            " (?,?,?,?,?,?,?,?,?,?)",
            [(f"SYM{i}USDT", "long", 1.0, 1.0, "manual", "1h", "closed", "close",
              f"tg{i}", "2020-01-01 00:00:00") for i in range(400)])
        db.commit()
        db.execute("ANALYZE")
        plan = " ".join(str(r[3]) for r in _scalars(
            db, "EXPLAIN QUERY PLAN DELETE FROM trades WHERE action IN"
                " ('close','reduce') AND closed_at IS NOT NULL AND closed_at <"
                " datetime('now','-365 days')"))
    finally:
        db.close()
    assert "idx_trades_closed_at" in plan or "idx_trades_action" in plan, plan


def test_wal_and_foreign_keys_are_enabled(tmp_path):
    db_path = str(tmp_path / "pragmas.db")
    _run(init_database(db_path))
    db = _connect(db_path)
    try:
        assert _scalars(db, "PRAGMA journal_mode")[0][0].lower() == "wal"
    finally:
        db.close()

    async def _fk():
        conn = await get_db()
        try:
            cursor = await conn.execute("PRAGMA foreign_keys")
            row = await cursor.fetchone()
            return row[0]
        finally:
            await conn.close()

    assert _run(_fk()) == 1


def test_wal_and_foreign_keys_survive_a_reopen(tmp_path):
    """WAL is a property of the file (not the connection), so it must persist."""
    db_path = str(tmp_path / "wal_reopen.db")
    _run(init_database(db_path))
    _run(save_sim_balance(1234.0, db_path))
    db = _connect(db_path)
    try:
        assert _scalars(db, "PRAGMA journal_mode")[0][0].lower() == "wal"
    finally:
        db.close()
    assert _run(load_sim_balance(db_path)) == pytest.approx(1234.0)


def test_live_db_style_checksum_is_stable_across_a_migration(tmp_path):
    """The property the real-DB dry run relies on: every value except the two
    columns v4 is meant to fill must be byte-identical after migrating."""
    db_path = str(tmp_path / "checksum.db")
    _run(init_database(db_path))
    db = _connect(db_path)
    db.execute("DROP TABLE trades")
    db.executescript(_V3_TRADES_DDL)
    for i in range(40):
        db.execute(
            "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity,"
            " pnl, pnl_pct, strategy, timeframe, status, trader, action,"
            " trade_group, reduce_pct, opened_at, closed_at, fill_price, fee,"
            " slippage) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"SYM{i}USDT", "long", 10.0 + i, 11.0 + i, 1.0, float(i), 1.0,
             "auto", "1h", "closed" if i % 2 else "open", "manual",
             "close" if i % 2 else "open", f"tg{i}", "0",
             "2026-01-01 00:00:00", None, 10.0 + i, 0.01, 0.02))
    db.commit()

    ignored = ("closed_at", "exit_reason", "action", "reduce_pct")
    cols = [r[1] for r in _scalars(db, "PRAGMA table_info(trades)")]
    kept = [c for c in cols if c not in ignored]
    sql = f"SELECT {','.join(kept)} FROM trades ORDER BY id"

    def digest(cursor):
        h = hashlib.sha256()
        for row in cursor:
            h.update(repr(tuple(row)).encode())
        return h.hexdigest()

    before = digest(_scalars(db, sql))
    db.close()

    _run(init_database(db_path))

    db = _connect(db_path)
    try:
        after = digest(_scalars(db, sql))
        assert _scalars(db, "SELECT COUNT(*) FROM trades")[0][0] == 40
    finally:
        db.close()
    assert after == before, "the migration changed data it was not supposed to touch"
