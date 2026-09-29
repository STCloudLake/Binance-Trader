import asyncio
import json
import re
import weakref
import aiosqlite
from pathlib import Path
from loguru import logger

DB_PATH: str = ""
DEFAULT_BALANCE = 10000.0

#: The sim-balance lock.  ``_balance_lock`` is the lock used when no event loop is
#: running (legacy callers/tests); ``_balance_locks`` maps each running loop to
#: its own lock — see :func:`balance_lock`.
_balance_lock = asyncio.Lock()
_balance_locks: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def balance_lock() -> asyncio.Lock:
    """The sim-balance lock belonging to the **running** event loop.

    ``asyncio.Lock`` binds itself, permanently, to the loop in which it first had
    to *wait* (``asyncio.mixins._LoopBoundMixin``); it never rebinds.  A single
    module-level lock therefore broke the moment this module was used from a
    second loop: ``atomic_adjust_balance`` raised ``Lock ... is bound to a
    different event loop`` in the middle of a cash movement — the audit's eight
    parallel opens hit it, and an open whose deduction fails is rolled back (a
    *close* whose credit fails would be worse: the position is already gone from
    the ledger).

    Keying the lock by loop keeps same-loop contention cheap and correct in every
    long-lived process; real mutual exclusion across loops/processes is provided
    by ``BEGIN IMMEDIATE`` inside the two writers below, which is what makes the
    read-modify-write atomic in the first place.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no loop: the legacy module-level lock
        return _balance_lock
    lock = _balance_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _balance_locks[loop] = lock
    return lock

#: Current schema version.  Kept in **two** places on purpose:
#:
#: * ``PRAGMA user_version`` — the authoritative copy, which the application
#:   never exposes for deletion (``/api/db/manager`` can delete any
#:   ``system_config`` row, and used to be able to delete the version with it).
#: * ``system_config.schema_version`` — a compatibility mirror so older code and
#:   tests that read the key keep working.  The effective version is the MAXIMUM
#:   of the two, so losing either one can never make the DB look older than it is.
SCHEMA_VERSION = 4

#: ``system_config`` key of the legacy version mirror.
LEGACY_VERSION_KEY = "schema_version"


def _utc_now_sql() -> str:
    """The one definition of "now" used by every timestamp in the DB.

    The database stores **UTC** (``CURRENT_TIMESTAMP`` / ``datetime('now')``);
    the API layer formats for display in local time.  Stated once here so a
    reader never has to guess which clock a ``closed_at`` came from.
    """
    return "datetime('now')"


# ----------------------------------------------------------------------
# Single DDL source
# ----------------------------------------------------------------------
# Every table the application still uses is declared here, exactly once: a fresh
# install creates this schema and a migrated database is rebuilt *into* it, so
# the two paths can no longer drift (they used to: v1's ``ALTER TABLE`` added
# ``trader TEXT`` with no CHECK while this file's ``trades`` had
# ``CHECK(trader IN ('manual','ai'))``).
#
# ``trader`` deliberately carries **no** CHECK constraint: the live DB records
# real usernames (``ft_9ef30e``, ``e2e_fee_trader``), which the old constraint
# rejected — a fresh install could not even insert a row copied from a migrated
# database ("CHECK constraint failed").  ``position_type``/``side``/``status``
# keep their checks: those values are a closed set the code owns.
SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL,
    pnl REAL DEFAULT 0,
    pnl_pct REAL DEFAULT 0,
    strategy TEXT NOT NULL DEFAULT 'manual',
    timeframe TEXT NOT NULL DEFAULT '1h',
    position_type TEXT DEFAULT 'satellite' CHECK(position_type IN ('core', 'satellite')),
    trader TEXT DEFAULT 'manual',
    strategy_name TEXT DEFAULT '',
    action TEXT DEFAULT 'open' CHECK(action IN ('open', 'close', 'reduce')),
    trade_group TEXT DEFAULT '',
    reduce_pct REAL DEFAULT 0,
    stop_loss REAL,
    take_profits TEXT,
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    exit_reason TEXT,
    status TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed', 'cancelled')),
    fill_price REAL,
    fee REAL,
    slippage REAL
);

-- Live positions, one row per open symbol.  `trades` remains the append-only
-- ledger (and the thing the balance identity is reconciled against); this table
-- is the authoritative snapshot of position *state* so a restart restores the
-- real basis, stop-loss and take-profits instead of guessing them.
--
-- `entry_price` is the same figure as `trades.entry_price` for the open row: the
-- cash basis per unit (`quantity × entry_price` == the cash the open deducted).
-- `fill_price` holds the raw (cost-worsened) fill.
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL DEFAULT 0,
    stop_loss REAL,
    take_profits TEXT,
    entry_stop_loss REAL,
    position_type TEXT DEFAULT 'satellite' CHECK(position_type IN ('core', 'satellite')),
    trader TEXT DEFAULT 'manual',
    strategy_name TEXT DEFAULT '',
    strategy TEXT DEFAULT 'manual',
    trade_group TEXT DEFAULT '',
    timeframe TEXT DEFAULT '1h',
    amount_usdt REAL,
    position_value REAL,
    fill_price REAL,
    fee REAL,
    slippage REAL,
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pending_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    type TEXT NOT NULL DEFAULT 'limit',
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    amount_usdt REAL NOT NULL,
    position_type TEXT DEFAULT 'satellite',
    stop_loss_pct REAL DEFAULT 2.0,
    trader TEXT DEFAULT 'manual',
    strategy_name TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open'
        CHECK(status IN ('open','filled','cancelled')),
    fill_price REAL,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL CHECK(level IN ('critical', 'warning', 'info')),
    type TEXT NOT NULL,
    message TEXT NOT NULL,
    symbol TEXT,
    acknowledged INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL CHECK(category IN ('coin_selection','strategy_optimization','risk_adjustment','market_assessment','news_analysis')),
    content TEXT NOT NULL,
    rationale TEXT,
    confidence REAL DEFAULT 0.5,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending','approved','rejected','applied')),
    applied_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS news_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('api', 'rss', 'web')),
    endpoint TEXT,
    api_key_encrypted TEXT,
    enabled INTEGER DEFAULT 1,
    rate_limit INTEGER DEFAULT 10,
    priority INTEGER DEFAULT 5
);

CREATE TABLE IF NOT EXISTS system_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'viewer' CHECK(role IN ('admin','trader','viewer')),
    display_name TEXT DEFAULT '',
    enabled INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backtest_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    mode TEXT NOT NULL DEFAULT 'full',
    strategies TEXT NOT NULL,
    symbols TEXT NOT NULL,
    date_start TEXT NOT NULL,
    date_end TEXT NOT NULL,
    initial_balance REAL NOT NULL DEFAULT 10000,
    final_balance REAL,
    metrics TEXT,
    trades_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strategy_lifecycle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    action TEXT NOT NULL,
    trigger_reason TEXT,
    metrics_snapshot TEXT,
    backtest_record_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades(opened_at);
CREATE INDEX IF NOT EXISTS idx_trades_action ON trades(action);
-- `close_position` looks a round trip up by trade_group (and by action+symbol
-- when the group is missing); without these two it planned `SCAN trades`.
CREATE INDEX IF NOT EXISTS idx_trades_trade_group ON trades(trade_group);
-- Serves both the `/api/history/trades` ordering and the `/api/db/cleanup`
-- retention predicate (`closed_at < now-365d`), which could never be planned.
CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_orders(status);
CREATE INDEX IF NOT EXISTS idx_pending_symbol ON pending_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(level);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_ai_status ON ai_suggestions(status);
"""

#: Tables that no code reads and no code writes.  They are dropped by migration
#: v4; keeping them only made the DB manager page offer empty tables and made a
#: reader believe the app has features it does not have.  ``news_sources`` is
#: deliberately NOT here — ``core/news/fetcher.py`` reads it.
DEAD_TABLES = ("orders", "risk_events", "ml_models", "news_articles", "test_tz")

#: Audit trail for manual ledger reconciliations (e.g. the P0 repair of
#: 2026-09-29). Deliberately NOT part of ``SCHEMA``: databases repaired by hand
#: already carry a legacy shape, so it is converged by a migration step instead
#: (see ``_migration_v4_ledger_reconciliation``).
LEDGER_RECONCILIATION_DDL = """
CREATE TABLE IF NOT EXISTS ledger_reconciliation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reconciled_at TEXT DEFAULT CURRENT_TIMESTAMP,
    balance_before REAL,
    balance_after REAL,
    delta REAL,
    open_notional REAL,
    realised_pnl REAL,
    expected_balance REAL,
    reason TEXT,
    detail TEXT,
    backup_path TEXT,
    created_by TEXT DEFAULT 'system'
)"""

#: ``trades`` columns that carry real information.  Anything else is dropped when
#: the table is rebuilt into the single DDL source.  Kept as a list (not a
#: ``SELECT *``) so an unknown *extra* column is never silently thrown away: the
#: rebuild asserts there are none.
TRADE_COLUMNS = (
    "id", "symbol", "side", "entry_price", "exit_price", "quantity", "pnl",
    "pnl_pct", "strategy", "timeframe", "position_type", "trader", "strategy_name",
    "action", "trade_group", "reduce_pct", "stop_loss", "take_profits",
    "opened_at", "closed_at", "exit_reason", "status", "fill_price", "fee",
    "slippage",
)

#: Canonical DEFAULT for a ``NOT NULL`` column, used when a pre-v4 row holds NULL
#: there (the old ``trades`` declared ``strategy``/``timeframe`` NOT NULL without a
#: default, so a row could legitimately exist before the defaults were added).
_NOT_NULL_DEFAULTS = {
    "strategy": "manual",
    "timeframe": "1h",
    "position_type": "satellite",
    "trader": "manual",
    "strategy_name": "",
    "action": "open",
    "trade_group": "",
    "reduce_pct": 0,
    "pnl": 0,
    "pnl_pct": 0,
    "status": "open",
}

#: Columns added to ``trades`` over time by the versioned migrations below.
_TRADE_ADDED_COLUMNS = (
    ("trader", "TEXT DEFAULT 'manual'"),
    ("strategy_name", "TEXT DEFAULT ''"),
    ("action", "TEXT DEFAULT 'open'"),
    ("trade_group", "TEXT DEFAULT ''"),
    ("reduce_pct", "REAL DEFAULT 0"),
    ("fill_price", "REAL"),
    ("fee", "REAL"),
    ("slippage", "REAL"),
    ("stop_loss", "REAL"),
    ("take_profits", "TEXT"),
    ("exit_reason", "TEXT"),
)


async def _table_columns(db, table: str) -> list[str]:
    cursor = await db.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in await cursor.fetchall()]


async def _table_exists(db, table: str) -> bool:
    cursor = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return await cursor.fetchone() is not None


async def _add_missing_columns(db) -> None:
    """Add every ``trades`` column a pre-v4 database may be missing.

    ``ALTER TABLE`` keeps all existing rows untouched, which is what lets the v3
    test pin "no backfill, no rewrite" for the cost columns.
    """
    existing = await _table_columns(db, "trades")
    for col, decl in _TRADE_ADDED_COLUMNS:
        if col in existing:
            continue
        await db.execute(f"ALTER TABLE trades ADD COLUMN {col} {decl}")
        logger.info(f"DB migration: trades.{col} added")
        existing.append(col)


_DDL_TOKEN_RE = re.compile(r"'[^']*'|\"[^\"]*\"|[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|[^\s]")


def _normalise_sql(sql: str) -> str:
    """Format-insensitive fingerprint of one ``CREATE TABLE`` statement.

    SQLite rewrites the statement it stores (it quotes the table name and
    re-indents), and the DDL here is hand-formatted, so comparing the raw text
    would report a difference on a freshly created table and rebuild it on every
    start.  Only the *tokens* are compared here — quoting style, ``IF NOT
    EXISTS``, line breaks and the spaces between tokens are all dropped —
    while keyword order, column order, types and CHECK expressions are kept.
    """
    tokens = _DDL_TOKEN_RE.findall(sql or "")
    # `CREATE TABLE IF NOT EXISTS x` == `CREATE TABLE x`
    if [t.lower() for t in tokens[:5]] == ["create", "table", "if", "not", "exists"]:
        tokens = tokens[:2] + tokens[5:]
    out = []
    for token in tokens:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        out.append(token.lower())
    if out and out[-1] == ";":
        out.pop()  # SQLite does not keep the trailing semicolon of the parsed DDL
    return " ".join(out)


def _normalise_trades_rows(rows, columns: list[str]):
    """Yield one value tuple per row, coerced into the canonical column set.

    Two coercions, both keeping the information the row already carried:

    * ``reduce_pct``: it used to be ``TEXT DEFAULT 0`` (TEXT affinity!), so a
      migrated DB could hold ``'50'`` where the DDL says REAL.  The value is the
      same number; only its storage class is fixed.
    * a NULL in a column the canonical DDL declares ``NOT NULL``: an old
      ``trades`` table could allow NULL ``timeframe``/``strategy`` (defaults were
      added later), and the rebuild would otherwise fail on a row that was
      perfectly valid before.  The canonical DEFAULT is the honest value for
      "the writer recorded nothing".
    """
    idx = {name: i for i, name in enumerate(columns)}
    for row in rows:
        out = []
        for name in TRADE_COLUMNS:
            if name not in idx:
                out.append(None)
                continue
            value = row[idx[name]]
            if name == "reduce_pct" and isinstance(value, str):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = 0
            elif value is None and name in _NOT_NULL_DEFAULTS:
                value = _NOT_NULL_DEFAULTS[name]
            out.append(value)
        missing = [name for name in ("symbol", "side", "entry_price", "quantity")
                   if name in idx and row[idx[name]] is None]
        if missing:
            raise RuntimeError(
                f"refusing to rebuild trades: row id={row[idx['id']]!r} has NULL "
                f"in NOT NULL column(s) {missing}")
        yield tuple(out)


async def _rebuild_trades(db) -> None:
    """Rebuild ``trades`` into exactly the canonical DDL, preserving every row.

    Used for the migration path from a DB whose ``trades`` was assembled from
    ``ALTER TABLE``s (``trader`` with no CHECK, ``reduce_pct`` with TEXT
    affinity).  All rows keep their ``id`` and every value they held; nothing is
    backfilled here and nothing is deleted.
    """
    columns = await _table_columns(db, "trades")
    unknown = [c for c in columns if c not in TRADE_COLUMNS]
    if unknown:
        # Refuse to drop something we do not understand: a silent data loss is
        # worse than an old-but-complete schema.
        raise RuntimeError(f"refusing to rebuild trades: unknown columns {unknown}")

    cur = await db.execute("SELECT * FROM trades")
    rows = await cur.fetchall()
    placeholders = ",".join("?" * len(TRADE_COLUMNS))
    collist = ",".join(TRADE_COLUMNS)
    id_idx = columns.index("id") if "id" in columns else None
    # A pre-v4 `trades` could have been created by ``CREATE TABLE ... AS SELECT``
    # (an old test fixture did), which leaves a plain NULLABLE ``id INT`` column
    # rather than the ``INTEGER PRIMARY KEY`` rowid alias.  A row inserted without
    # naming `id` then sits in the table with ``id IS NULL`` while its rowid is
    # perfectly unique.  Every row must come out of the rebuild with a real,
    # unique id, so NULL ids are numbered from the highest existing id: no
    # existing id ever changes and none can collide.
    next_id = 1
    if id_idx is not None:
        known = [r[id_idx] for r in rows if r[id_idx] is not None]
        next_id = (max(known) + 1) if known else 1
    normalized = []
    for row in _normalise_trades_rows(rows, columns):
        if id_idx is not None and row[0] is None:
            row = (next_id,) + tuple(row[1:])
            next_id += 1
        normalized.append(row)

    await db.execute(
        "CREATE TABLE trades_v4 ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " symbol TEXT NOT NULL,"
        " side TEXT NOT NULL CHECK(side IN ('long','short')),"
        " entry_price REAL NOT NULL,"
        " exit_price REAL,"
        " quantity REAL NOT NULL,"
        " pnl REAL DEFAULT 0,"
        " pnl_pct REAL DEFAULT 0,"
        " strategy TEXT NOT NULL DEFAULT 'manual',"
        " timeframe TEXT NOT NULL DEFAULT '1h',"
        " position_type TEXT DEFAULT 'satellite'"
        "   CHECK(position_type IN ('core','satellite')),"
        " trader TEXT DEFAULT 'manual',"
        " strategy_name TEXT DEFAULT '',"
        " action TEXT DEFAULT 'open' CHECK(action IN ('open','close','reduce')),"
        " trade_group TEXT DEFAULT '',"
        " reduce_pct REAL DEFAULT 0,"
        " stop_loss REAL,"
        " take_profits TEXT,"
        " opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
        " closed_at TIMESTAMP,"
        " exit_reason TEXT,"
        " status TEXT DEFAULT 'open'"
        "   CHECK(status IN ('open','closed','cancelled')),"
        " fill_price REAL,"
        " fee REAL,"
        " slippage REAL)")
    await db.executemany(
        f"INSERT INTO trades_v4 ({collist}) VALUES ({placeholders})",
        normalized)
    await db.execute("DROP TABLE trades")
    await db.execute("ALTER TABLE trades_v4 RENAME TO trades")
    # `DROP TABLE trades` drops every index on it with the table, and the
    # canonical indexes were already created by the `executescript(SCHEMA)` that
    # ran *before* this rebuild — so `IF NOT EXISTS` will not put them back.  All
    # of them are recreated here, or `init_database` leaves the database with an
    # index set a fresh install never has (`idx_trades_symbol/status/opened` gone,
    # `restore_positions` planning `SCAN trades`).
    await _create_all_indexes(db)
    logger.info(f"DB migration: trades rebuilt into the canonical schema "
                f"({len(rows)} rows preserved)")


async def _migration_v4_schema_convergence(db) -> None:
    """v4 step 1 — make every database look exactly like a fresh install."""
    existing = await _table_columns(db, "trades")
    for col, decl in _TRADE_ADDED_COLUMNS:
        if col in existing:
            continue
        await db.execute(f"ALTER TABLE trades ADD COLUMN {col} {decl}")
        logger.info(f"DB migration: trades.{col} added")
        existing.append(col)
    # Rebuild when the assembled table diverges from the single DDL source:
    # a CHECK the parser did not see (the old v1 `ALTER` added `trader` bare,
    # so a migrated DB accepts a username a fresh one rejected), a missing CHECK,
    # or a column whose declared type differs (`reduce_pct` as TEXT).
    normalized = _normalise_sql(await _ddl(db, "trades"))
    fresh = _normalise_sql(_statement_for("trades"))
    if normalized != fresh:
        await _rebuild_trades(db)

    # `positions` was declared but never written; it is redefined as the live
    # position snapshot.  A legacy copy (whatever `ALTER`s assembled) is dropped
    # and recreated from the same DDL source as a fresh install.  This is the one
    # table where a drop discards no information: no code ever inserted a row into
    # it (the audit's point, and the only statement that mentioned it was a
    # `DELETE`), so a legacy copy is empty by construction.
    if await _table_exists(db, "positions"):
        legacy = _normalise_sql(await _ddl(db, "positions"))
        if legacy != _normalise_sql(_statement_for("positions")):
            await db.execute("DROP TABLE positions")
            logger.info("DB migration v4: positions redefined")
    await db.execute(_statement_for("positions"))


async def _migration_v4_indexes(db) -> None:
    # Every index the SCHEMA declares, not a hand-kept subset: this step runs
    # after the schema-convergence rebuild (which drops the indexes with the old
    # table), so it is the net that guarantees a migrated DB ends up with exactly
    # the index set a fresh install has.
    await _create_all_indexes(db)


async def _migration_v4_backfill_closed_at(db) -> None:
    """Fill ``closed_at`` (and ``exit_reason``) on rows written before v4.

    ``closed_at`` was never written on ANY row (1370/1370 NULL in the live DB),
    so the retention predicate ``status='closed' AND closed_at < now-365d`` could
    never match and closed history was never cleaned.  The close/reduce row of a
    round trip carries the real exit time; the open row is backfilled from its
    own group's exit so both rows of a round trip agree.
    """
    await db.execute(
        """UPDATE trades SET closed_at = (
               SELECT MAX(c.opened_at) FROM trades c
               WHERE c.trade_group = trades.trade_group
                 AND c.trade_group != ''
                 AND c.action IN ('close','reduce')
           )
           WHERE action IN ('close','reduce') AND closed_at IS NULL
             AND trade_group != ''
             AND EXISTS (SELECT 1 FROM trades c
                         WHERE c.trade_group = trades.trade_group
                           AND c.action IN ('close','reduce'))""")
    # Rows whose action predates the column (`action` defaults to 'open'): a row
    # that carries an exit price or a closed status and has no close/reduce
    # sibling is itself the exit record.
    await db.execute(
        "UPDATE trades SET closed_at = opened_at "
        "WHERE closed_at IS NULL AND action='open' AND status='closed' "
        "AND exit_price IS NOT NULL")
    await db.execute(
        """UPDATE trades SET action='close'
           WHERE action='open' AND status='closed' AND exit_price IS NOT NULL
             AND trade_group != ''
             AND NOT EXISTS (SELECT 1 FROM trades o
                             WHERE o.trade_group = trades.trade_group
                               AND o.action='open' AND o.exit_price IS NULL)""")
    await db.execute(
        "UPDATE trades SET exit_reason='legacy' "
        "WHERE exit_reason IS NULL AND action IN ('close','reduce')")


async def _migration_v4_drop_dead_tables(db) -> None:
    for table in DEAD_TABLES:
        if await _table_exists(db, table):
            await db.execute(f"DROP TABLE {table}")
            logger.info(f"DB migration v4: dropped unused table {table}")


async def _migration_v4_pragmas(db) -> None:
    """FK enforcement on every connection this module opens (see ``get_db``).

    ``journal_mode=WAL`` is not here: SQLite refuses to switch journal mode
    inside a transaction (and every migration step runs in a SAVEPOINT), so the
    journal is set once, before any DDL, in ``init_database``.
    """
    await db.execute("PRAGMA foreign_keys=ON")


#: Ordered, idempotent migration steps.  Every step is written so that running it
#: on a database that already has the change is a no-op, which is what makes the
#: whole chain replayable (and testable) without a version gate.
async def _migration_v4_ledger_reconciliation(db) -> None:
    """Converge the ``ledger_reconciliation`` audit table.

    It is not in ``SCHEMA`` on purpose: databases that were hand-repaired already
    have it with a DIFFERENT shape (``at``/``realised``), so a
    ``CREATE TABLE IF NOT EXISTS`` + an index on a newer column name raised
    ``no such column`` inside ``executescript`` — outside the per-step SAVEPOINTs,
    which made the whole app fail to start.  Here we rebuild it to the canonical
    shape (preserving the recorded rows) and then create the index.
    """
    cursor = await db.execute("PRAGMA table_info(ledger_reconciliation)")
    cols = {row[1] for row in await cursor.fetchall()}
    if cols and "reconciled_at" not in cols:
        await db.execute("ALTER TABLE ledger_reconciliation RENAME TO ledger_reconciliation_legacy")
        await db.execute(LEDGER_RECONCILIATION_DDL)
        # Legacy columns: id, at, balance_before, balance_after, delta,
        # open_notional, realised, detail.  Map what exists, ignore the rest.
        legacy = {row[1] for row in await (await db.execute(
            "PRAGMA table_info(ledger_reconciliation_legacy)")).fetchall()}
        pairs = [("id", "id"), ("balance_before", "balance_before"),
                 ("balance_after", "balance_after"), ("delta", "delta"),
                 ("open_notional", "open_notional"), ("detail", "detail")]
        if "at" in legacy:
            pairs.append(("reconciled_at", "at"))
        if "realised" in legacy:
            pairs.append(("realised_pnl", "realised"))
        target = ", ".join(dst for dst, _ in pairs)
        source = ", ".join(src for _, src in pairs)
        await db.execute(
            f"INSERT INTO ledger_reconciliation ({target}) "
            f"SELECT {source} FROM ledger_reconciliation_legacy")
        await db.execute("DROP TABLE ledger_reconciliation_legacy")
    elif not cols:
        await db.execute(LEDGER_RECONCILIATION_DDL)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_ledger_recon_at "
        "ON ledger_reconciliation(reconciled_at)")


MIGRATIONS = (
    ("v4:schema-convergence", (_migration_v4_schema_convergence,)),
    ("v4:indexes", (_migration_v4_indexes,)),
    ("v4:closed-at-backfill", (_migration_v4_backfill_closed_at,)),
    ("v4:drop-dead-tables", (_migration_v4_drop_dead_tables,)),
    ("v4:pragmas", (_migration_v4_pragmas,)),
    ("v4:ledger-reconciliation", (_migration_v4_ledger_reconciliation,)),
)


async def _ddl(db, table: str) -> str:
    cursor = await db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
    row = await cursor.fetchone()
    return row[0] if row and row[0] else ""


def _statement_for(table: str) -> str:
    """Extract one ``CREATE TABLE`` statement out of ``SCHEMA``."""
    marker = f"CREATE TABLE IF NOT EXISTS {table} ("
    start = SCHEMA.index(marker)
    end = SCHEMA.index(";", start) + 1
    return SCHEMA[start:end]


_INDEX_RE = re.compile(r"CREATE INDEX IF NOT EXISTS[^;]*")


def _schema_index_statements() -> list[str]:
    """Every ``CREATE INDEX`` statement declared in :data:`SCHEMA`.

    Read back out of the single DDL source instead of being repeated in a second
    list: a hand-kept copy is exactly how ``_rebuild_trades`` came to recreate
    only 4 of the 12 indexes and left the rest of the database a ``SCAN trades``
    for ever (a rebuild DROPs every index on ``trades``).
    """
    return [m.group(0).strip() for m in _INDEX_RE.finditer(SCHEMA)]


async def _create_all_indexes(db) -> None:
    """(Re)create every index from :data:`SCHEMA` — idempotent."""
    for ddl in _schema_index_statements():
        await db.execute(ddl)


async def _read_schema_version(db) -> int:
    version = 0
    try:
        cursor = await db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        version = int(row[0]) if row else 0
    except Exception:  # pragma: no cover - defensive
        version = 0
    try:
        cursor = await db.execute(
            "SELECT value FROM system_config WHERE key=?", (LEGACY_VERSION_KEY,))
        row = await cursor.fetchone()
        if row:
            version = max(version, int(str(row[0])))
    except Exception:  # pragma: no cover - table may not exist yet
        pass
    return version


async def _write_schema_version(db, version: int) -> None:
    await db.execute(f"PRAGMA user_version={int(version)}")
    await db.execute(
        "INSERT OR REPLACE INTO system_config (key, value, category) "
        "VALUES (?, ?, 'system')", (LEGACY_VERSION_KEY, str(int(version))))


async def init_database(db_path: str):
    """Create (or migrate) the schema, then report the version.

    Each migration step runs inside its own SAVEPOINT: a step that fails rolls
    back on its own and leaves the database on the last complete step instead of
    half-applied (the old code committed once per column, so a crash mid-``ALTER``
    left a schema nobody could describe).
    """
    global DB_PATH
    DB_PATH = db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        # WAL before anything else: it cannot be switched on from inside a
        # transaction, and migrations run inside SAVEPOINTs.  WAL survives in the
        # file (it is a database property), so setting it once here is enough.
        try:
            await db.execute("PRAGMA journal_mode=WAL")
        except Exception as e:  # pragma: no cover - e.g. a read-only filesystem
            logger.warning(f"Could not enable WAL journal mode: {e}")
        await db.execute("PRAGMA foreign_keys=ON")

        # Single DDL source: a fresh DB is born at the current version, and
        # `IF NOT EXISTS` makes this a harmless no-op on an existing one.
        await db.executescript(SCHEMA)
        await db.commit()

        db.row_factory = aiosqlite.Row
        version_before = await _read_schema_version(db)

        for name, steps in MIGRATIONS:
            savepoint = f"mig_{name.replace(':', '_').replace('-', '_')}"
            await db.execute(f"SAVEPOINT {savepoint}")
            try:
                for step in steps:
                    await step(db)
            except Exception as e:
                await db.execute(f"ROLLBACK TO {savepoint}")
                await db.execute(f"RELEASE {savepoint}")
                logger.error(f"DB migration step {name} failed and was rolled back: {e}")
                raise
            await db.execute(f"RELEASE {savepoint}")

        await _write_schema_version(db, SCHEMA_VERSION)
        await db.commit()

        version_after = await _read_schema_version(db)
        if version_before != version_after:
            logger.info(f"DB migrated from v{version_before} to v{version_after} "
                        f"({db_path})")


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    # Enforced per connection: the PRAGMA is not persistent in SQLite.
    await db.execute("PRAGMA foreign_keys=ON")
    return db


from contextlib import asynccontextmanager

@asynccontextmanager
async def db_connection():
    """Async context manager that ensures DB connection is always closed."""
    db = await get_db()
    try:
        yield db
    finally:
        await db.close()


def _json_levels(value) -> str:
    """Serialise take-profit levels (``None`` → ``'[]'``), never raising."""
    if value is None:
        return "[]"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(list(value))
    except TypeError:
        return json.dumps([value])


def parse_levels(value) -> list:
    """Deserialise a stored take-profits blob back into a list."""
    if value in (None, "", b""):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]


class LedgerUnavailable(RuntimeError):
    """The persisted sim balance could not be read.

    Raised instead of returning :data:`DEFAULT_BALANCE`: a corrupt, locked or
    missing database used to look like a wallet that never traded (exactly
    ``10000.0``), which is the single most dangerous possible answer — the risk
    manager then sizes positions as if no capital were committed.
    """


async def load_sim_balance(db_path: str = None) -> float:
    """Load sim balance from ``system_config``.

    Raises :class:`LedgerUnavailable` when the database cannot be read.  A
    *readable* database that simply has no ``sim_balance`` row yet is a fresh
    install, and the documented starting balance (see ``DEFAULT_BALANCE``) is the
    honest answer for it.
    """
    path = db_path or DB_PATH
    if not path:
        logger.warning("load_sim_balance: no database path configured — "
                       f"reporting the default {DEFAULT_BALANCE}")
        return DEFAULT_BALANCE
    if not Path(path).exists():
        message = (f"load_sim_balance: database {path} does not exist; refusing "
                   f"to report a full wallet of {DEFAULT_BALANCE}")
        logger.critical(message)
        raise LedgerUnavailable(message)
    db = None
    try:
        db = await aiosqlite.connect(path)
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT value FROM system_config WHERE key='sim_balance'")
        row = await cursor.fetchone()
        if row:
            return float(row["value"])
        logger.warning(
            f"load_sim_balance: {path} has no sim_balance row — a fresh install, "
            f"reporting the default {DEFAULT_BALANCE}")
        return DEFAULT_BALANCE
    except Exception as e:
        message = (f"load_sim_balance: could not read the balance from {path} "
                   f"({e!r}); refusing to report a full wallet of {DEFAULT_BALANCE}")
        logger.critical(message)
        raise LedgerUnavailable(message) from e
    finally:
        if db is not None:
            await db.close()


async def save_sim_balance(balance: float, db_path: str = None):
    """Persist sim balance to ``system_config`` atomically.

    Uses the same ``BEGIN IMMEDIATE`` + shared-lock path as
    :func:`atomic_adjust_balance`: the previous bare ``INSERT OR REPLACE``
    autocommitted on its own, so a concurrent ``atomic_adjust_balance`` (an
    engine open/close) could interleave with it and the *stale* absolute value
    won — silently reverting a real fill's cash movement.
    """
    path = db_path or DB_PATH
    if not path:
        return
    async with balance_lock():
        db = await aiosqlite.connect(path)
        try:
            await db.execute("BEGIN IMMEDIATE")
            await db.execute(
                "INSERT OR REPLACE INTO system_config (key, value, category) "
                "VALUES ('sim_balance', ?, 'trading')",
                (str(balance),))
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.warning(f"Failed to save sim balance: {e}")
            raise
        finally:
            await db.close()


async def atomic_adjust_balance(delta: float, db_path: str = None) -> float:
    """Atomically adjust sim balance by delta and return the new value.

    Uses a single DB connection with BEGIN IMMEDIATE transaction to ensure
    true atomicity — the read and write happen in the same transaction,
    so a crash between them cannot leave the balance inconsistent.
    """
    path = db_path or DB_PATH
    if not path:
        return DEFAULT_BALANCE + delta
    async with balance_lock():
        db = await aiosqlite.connect(path)
        try:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                "SELECT value FROM system_config WHERE key='sim_balance'")
            row = await cursor.fetchone()
            current = float(row["value"]) if row else DEFAULT_BALANCE
            new_balance = current + delta
            await db.execute(
                "INSERT OR REPLACE INTO system_config (key, value, category) "
                "VALUES ('sim_balance', ?, 'trading')",
                (str(new_balance),))
            await db.commit()
            return new_balance
        except Exception as e:
            await db.rollback()
            logger.warning(f"atomic_adjust_balance failed: {e}")
            raise
        finally:
            await db.close()
