"""Read-only ledger audit for ``data/binance_trader.db``.

Prints the ledger identity computed from the **real** database (the file is
copied to a temp dir first, so the live DB is never even opened for writing),
followed by the table/index inventory promised by README §9.4.

The reported drift is ``system_config.sim_balance − (10000 − open notional +
realised pnl)``.  A **fresh** database has no ``sim_balance`` row and no trades;
treating that (absent) value as ``0`` produced a bogus ``delta = -10000`` that
read exactly like real drift.  A database with no ledger yet is therefore
detected explicitly, reported as ``fresh database — no ledger yet`` and exits 0.

Usage::

    python scripts/audit_db.py [path\\to\\binance_trader.db]

Exit codes:

    0   the identity holds — or the database is fresh and has no ledger yet
    1   the identity is broken (real drift), or the database file is missing

Nothing is ever written back to the database.  Run ``--help`` for the same
summary.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "binance_trader.db"

#: Balance the ledger identity is stated against (db.database.DEFAULT_BALANCE).
STARTING_BALANCE = 10000.0

#: Tables that must exist before there can be anything to audit.
LEDGER_TABLES = ("system_config", "trades")

EXIT_OK = 0
EXIT_DRIFT = 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="audit_db.py",
        description="Read-only audit of the simulated-ledger identity "
                    "(README §9.4) plus a table/index inventory.",
        epilog="Exit codes: 0 = identity holds OR the database is fresh "
               "(no ledger yet); 1 = identity broken (real drift) or the "
               "database file is missing.  The live DB is never opened for "
               "writing — it is copied to a temp dir first.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "db", nargs="?", default=str(DEFAULT_DB),
        help="database file to audit (default: data/binance_trader.db)")
    return parser.parse_args(argv)


def _copy_to_temp(src: Path) -> Path:
    """Copy ``src`` (plus its -wal/-shm siblings) into a temp dir."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="bt_audit_"))
    dst = tmp_dir / src.name
    shutil.copy2(src, dst)
    for suffix in ("-wal", "-shm"):
        side = Path(str(src) + suffix)
        if side.exists():
            shutil.copy2(side, str(dst) + suffix)
    return dst


def _scalar(conn: sqlite3.Connection, sql: str, default=0.0) -> float:
    row = conn.execute(sql).fetchone()
    if not row or row[0] is None:
        return default
    return float(row[0])


def _read_sim_balance(conn: sqlite3.Connection) -> tuple[float | None, bool]:
    """``(value, found)`` for ``system_config.sim_balance``.

    ``found`` distinguishes a genuine ``0`` balance from the *absent* row of a
    fresh database — collapsing both to ``0.0`` is what produced the phantom
    ``delta = -10000``.
    """
    row = conn.execute(
        "SELECT value FROM system_config WHERE key='sim_balance'").fetchone()
    if row is None or row[0] is None:
        return None, False
    return float(row[0]), True


def main(argv: list[str]) -> int:
    args = _parse_args(argv[1:])
    src = Path(args.db).resolve()
    if not src.exists():
        print(f"ERROR: database not found: {src}")
        return EXIT_DRIFT

    work = _copy_to_temp(src)
    print(f"source     : {src}")
    print(f"working copy: {work}  (read-only audit; live DB untouched)")
    conn = sqlite3.connect(f"file:{work}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        print(f"tables     : {len(tables)}")

        # ── fresh database: nothing to audit yet ───────────────────────
        missing = [name for name in LEDGER_TABLES if name not in tables]
        if missing:
            print()
            print("RESULT: FRESH DATABASE — NO LEDGER YET "
                  f"(missing table(s): {', '.join(missing)}); "
                  "nothing to compare, no drift.  Exit 0.")
            return EXIT_OK

        # ── identity ───────────────────────────────────────────────────
        # Verbatim the predicate in tests/test_ledger_invariant.py:
        #   identity = 10000 - SUM(open rows: qty*entry_price) + SUM(closed rows: pnl)
        # and the invariant is ``identity == system_config.sim_balance``.
        open_notional = _scalar(
            conn, "SELECT COALESCE(SUM(quantity * entry_price), 0) FROM trades "
                  "WHERE status = 'open'")
        realised = _scalar(
            conn, "SELECT COALESCE(SUM(pnl), 0) FROM trades "
                  "WHERE status = 'closed'")
        balance, has_balance = _read_sim_balance(conn)
        identity = STARTING_BALANCE - open_notional + realised

        print()
        print("ledger identity (README §9.4):")
        print(f"  starting balance       = {STARTING_BALANCE:.6f}")
        print(f"  - open notional        = {open_notional:.6f}")
        print(f"  + realised pnl         = {realised:.6f}")
        print(f"  = identity             = {identity:.6f}")

        if not has_balance:
            # The row is written by the first open; without it there is no
            # recorded cash to compare against, so a delta would be meaningless.
            print("  sim_balance            = (no row)")
            print()
            print("RESULT: FRESH DATABASE — NO LEDGER YET "
                  "(no system_config.sim_balance row); nothing to compare, "
                  "no drift.  Exit 0.")
            return EXIT_OK

        delta = balance - identity
        print(f"  sim_balance            = {balance:.6f}")
        print(f"  delta (balance - ident)= {delta:.6f}")
        holds = abs(delta) < 1e-6

        # ── trades distribution ────────────────────────────────────────
        print()
        print("trades distribution:")
        for action, count in conn.execute(
                "SELECT action, COUNT(*) FROM trades GROUP BY action ORDER BY action"):
            print(f"  action={action!r:<10} {count}")
        null_closed = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE closed_at IS NULL").fetchone()[0]
        null_exit = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE exit_price IS NULL").fetchone()[0]
        print(f"  closed_at IS NULL: {null_closed}   exit_price IS NULL: {null_exit}")

        # ── inventory ──────────────────────────────────────────────────
        print()
        print(f"tables ({len(tables)}):")
        for name in tables:
            rows = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            print(f"  {name:<32} {rows} rows")

        print()
        indexes = conn.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type='index' "
            "ORDER BY tbl_name, name").fetchall()
        print(f"indexes ({len(indexes)}):")
        for name, table in indexes:
            print(f"  {name:<40} on {table}")

        print()
        print("RESULT:", "IDENTITY HOLDS" if holds else "IDENTITY BROKEN")
        return EXIT_OK if holds else EXIT_DRIFT
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
