"""DB Manager page APIs (table view, row delete, backup/restore/optimize/
cleanup, CSV export)."""
import csv
import io
import os
import shutil

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse

from db.database import get_db, LEDGER_RECONCILIATION_DDL

from web.rendering import _render

#: The retention predicate `/api/db/cleanup` deletes on.  Written with a bare
#: ``closed_at <`` comparison on purpose: wrapping it in COALESCE() (or an OR)
#: makes SQLite plan `SCAN trades` and ignore `idx_trades_closed_at` entirely.  A
#: row whose exit time is unknown is kept, which is the safe direction — never
#: delete what cannot be dated.
#:
#: ``action IN ('close','reduce')`` is equally load-bearing: an ``action='open'``
#: row is either a LIVE position (its ``quantity × entry_price`` is what the
#: balance identity subtracts, so deleting it needs the cash credited back) or a
#: historical entry leg whose ``closed_at`` was never written.  Neither is ever
#: eligible here — the 692 entry legs with ``closed_at IS NULL`` in the live DB
#: stay on purpose.  A *closed*, PnL-free entry leg can still be removed one row
#: at a time through ``DELETE /api/db/row``, which is identity-neutral.
RETENTION_PREDICATE = (
    "action IN ('close', 'reduce') "
    "  AND closed_at IS NOT NULL "
    "  AND closed_at < datetime('now', '-365 days')"
)


def register(app: FastAPI, ctx) -> None:
    config = ctx.config
    logger = ctx.logger

    # ---- DB Manager API routes ----
    @app.get("/api/db/table/{table}")
    async def db_table_view(request: Request, table: str, search: str = "", page: int = 1, per_page: int = 50):
        # Clamp pagination: an unclamped or negative value produced `LIMIT -1`,
        # i.e. a full-table dump in a single response.
        page = max(1, min(int(page), 100000))
        per_page = max(1, min(int(per_page), 200))
        try:
            user = getattr(request.state, "user", None)
            if not user or not user.is_admin:
                return HTMLResponse("Forbidden", status_code=403)
            allowed = ["trades", "positions", "pending_orders", "alerts", "ai_suggestions", "backtest_records", "strategy_lifecycle_events", "news_sources", "system_config", "users"]
            if table not in allowed:
                return HTMLResponse("Invalid table")
            db = await get_db()
            try:
                cursor = await db.execute(f"PRAGMA table_info({table})")
                cols = [dict(r) for r in await cursor.fetchall()]
                columns = [c["name"] for c in cols]
                # Count total rows
                count_query = f"SELECT COUNT(*) FROM {table}"
                count_params = []
                if search and table in ("trades", "alerts"):
                    count_query += " WHERE symbol LIKE ? OR message LIKE ?"
                    count_params = [f"%{search}%", f"%{search}%"]
                cursor = await db.execute(count_query, count_params)
                total_rows = (await cursor.fetchone())[0]
                total_pages = max(1, (total_rows + per_page - 1) // per_page)
                page = max(1, min(page, total_pages))
                # Build page range for display
                start_p = max(1, page - 2)
                end_p = min(total_pages, page + 2)
                page_range = list(range(start_p, end_p + 1))
                # Query data
                query = f"SELECT * FROM {table}"
                params = []
                if search and table in ("trades", "alerts"):
                    query += " WHERE symbol LIKE ? OR message LIKE ?"
                    params = [f"%{search}%", f"%{search}%"]
                offset = (page - 1) * per_page
                query += f" ORDER BY id DESC LIMIT {per_page} OFFSET {offset}"
                cursor = await db.execute(query, params)
                rows = [dict(r) for r in await cursor.fetchall()]
            finally:
                await db.close()
            return _render("partials/db_table.html", {
                "request": None, "table": table, "columns": columns, "rows": rows,
                "page": page, "total_pages": total_pages, "total_rows": total_rows,
                "page_range": page_range,
            })
        except Exception as e:
            import traceback
            logger.error(f"db_table_view error: {e}\n{traceback.format_exc()}")
            return HTMLResponse(f"Error: {e}", status_code=500)

    @app.get("/api/db/backup")
    async def db_backup(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        import time as _time
        src = config.db_path
        ts = _time.strftime("%Y%m%d_%H%M%S", _time.localtime())
        dst = src.replace(".db", f"_backup_{ts}.db")
        shutil.copy2(src, dst)
        return FileResponse(dst, filename=f"binance_trader_backup_{ts}.db")

    @app.post("/api/db/restore")
    async def db_restore(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"ok": False, "error": "Forbidden"}, status_code=403)
        form = await request.form()
        file = form.get("file")
        if not file:
            return {"ok": False, "error": "No file uploaded"}
        contents = await file.read()
        if contents[:16] != b"SQLite format 3\x00":
            return {"ok": False, "error": "Not a valid SQLite database"}
        src = config.db_path
        backup_path = src + ".pre_restore"
        shutil.copy2(src, backup_path)
        with open(src, "wb") as f:
            f.write(contents)
        return {"ok": True}

    @app.post("/api/db/optimize")
    async def db_optimize(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"ok": False, "error": "Forbidden"}, status_code=403)
        before = os.path.getsize(config.db_path)
        db = await get_db()
        await db.execute("VACUUM")
        await db.execute("REINDEX")
        await db.close()
        after = os.path.getsize(config.db_path)
        return {"ok": True, "before": before, "after": after}

    @app.post("/api/db/cleanup")
    async def db_cleanup(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"ok": False, "error": "Forbidden"}, status_code=403)
        db = await get_db()
        removed_pnl = 0.0
        removed_rows = 0
        try:
            # One transaction for the delete AND the balance adjustment: the
            # ledger identity is `10000 − Σ(open qty×entry) + Σ(close pnl) ==
            # sim_balance`, so dropping exit rows without moving the balance
            # changes Σ(close pnl) and the ledger stops reconciling (measured
            # Δ −7.17 for 8 old rows).  Every deleted exit row's realised PnL is
            # credited back out of the balance in the same transaction, and the
            # whole operation is written to `ledger_reconciliation`.
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                f"SELECT COALESCE(SUM(pnl), 0) AS pnl, COUNT(*) AS n "
                f"FROM trades WHERE {RETENTION_PREDICATE}")
            row = await cursor.fetchone()
            removed_pnl = float(row["pnl"] or 0.0)
            removed_rows = int(row["n"] or 0)

            await db.execute(f"DELETE FROM trades WHERE {RETENTION_PREDICATE}")
            await db.execute("DELETE FROM alerts WHERE created_at < datetime('now', '-90 days')")
            await db.execute("DELETE FROM ai_suggestions WHERE created_at < datetime('now', '-90 days')")

            if removed_rows:
                cursor = await db.execute(
                    "SELECT value FROM system_config WHERE key='sim_balance'")
                bal_row = await cursor.fetchone()
                if bal_row is None:
                    # No ledger to adjust: refuse to delete the rows rather than
                    # leave the identity unreconcilable.
                    await db.rollback()
                    return JSONResponse({
                        "ok": False,
                        "error": ("system_config.sim_balance is missing, so the realised "
                                  f"PnL of the {removed_rows} expired row(s) cannot be "
                                  "taken off the balance; nothing was deleted"),
                    }, status_code=500)
                balance_before = float(bal_row["value"])
                cursor = await db.execute(
                    "SELECT COALESCE(SUM(quantity * entry_price), 0) FROM trades "
                    "WHERE status='open'")
                open_notional = float((await cursor.fetchone())[0] or 0.0)
                cursor = await db.execute(
                    "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status='closed'")
                realised = float((await cursor.fetchone())[0] or 0.0)
                balance_after = balance_before - removed_pnl
                expected = 10000.0 - open_notional + realised
                await db.execute(
                    "INSERT OR REPLACE INTO system_config (key, value, category) "
                    "VALUES ('sim_balance', ?, 'trading')", (str(balance_after),))
                await db.execute(LEDGER_RECONCILIATION_DDL)
                await db.execute(
                    "INSERT INTO ledger_reconciliation (balance_before, balance_after,"
                    " delta, open_notional, realised_pnl, expected_balance, reason,"
                    " detail, created_by)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (balance_before, balance_after, balance_after - expected,
                     open_notional, realised, expected,
                     "db_cleanup_retention",
                     f"deleted {removed_rows} expired exit row(s) carrying "
                     f"realised PnL {removed_pnl:.8f}; balance reduced by the same "
                     f"amount so the ledger identity holds",
                     getattr(user, "username", None) or "system"))
                logger.info(f"DB cleanup: removed {removed_rows} expired exit row(s), "
                            f"balance {balance_before:.8f} -> {balance_after:.8f} "
                            f"(identity={expected:.8f})")
            await db.commit()
            # VACUUM cannot run inside a transaction: commit first, exactly as
            # this endpoint always did.
            await db.execute("VACUUM")
        finally:
            await db.close()
        return {"ok": True, "removed_rows": removed_rows,
                "balance_adjustment": -removed_pnl}

    @app.get("/api/db/export/{table}")
    async def db_export_csv(table: str, request: Request, search: str = None):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        allowed = ["trades", "positions", "pending_orders", "alerts", "ai_suggestions", "backtest_records", "strategy_lifecycle_events", "news_sources", "system_config", "users"]
        if table not in allowed:
            return HTMLResponse("Invalid table", status_code=400)
        db = await get_db()
        cursor = await db.execute(f"PRAGMA table_info({table})")
        cols = [dict(r) for r in await cursor.fetchall()]
        columns = [c["name"] for c in cols]
        query = f"SELECT * FROM {table}"
        params = []
        if search and table in ("trades", "alerts"):
            query += " WHERE symbol LIKE ? OR message LIKE ?"
            params = [f"%{search}%", f"%{search}%"]
        cursor = await db.execute(query, params)
        rows = [dict(r) for r in await cursor.fetchall()]
        await db.close()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(c, "") for c in columns])
        output.seek(0)
        return StreamingResponse(output, media_type="text/csv",
                                  headers={"Content-Disposition": f"attachment; filename={table}_export.csv"})

    @app.delete("/api/db/row/{table}/{row_id}")
    async def db_delete_row(table: str, row_id: int, request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"ok": False, "error": "Forbidden"}, status_code=403)
        allowed = ["trades", "positions", "pending_orders", "alerts", "ai_suggestions", "backtest_records", "strategy_lifecycle_events", "news_sources", "system_config", "users"]
        if table not in allowed:
            return {"ok": False}
        db = await get_db()
        try:
            # A trades row is not an ordinary record: the balance identity is
            # stated against exactly these rows (`10000 − Σ(open qty×entry) +
            # Σ(close pnl) == sim_balance`).  Deleting an OPEN row (its notional is
            # subtracted) or one carrying realised PnL (deleting it leaves the
            # balance above the identity — measured Δ +0.11) used to be one click
            # away.  Refuse with the reason instead; a closed, PnL-free entry leg
            # is identity-neutral and still deletable.
            if table == "trades":
                cursor = await db.execute(
                    "SELECT status, action, pnl FROM trades WHERE id=?", (row_id,))
                row = await cursor.fetchone()
                if row is not None:
                    status = row["status"]
                    pnl = float(row["pnl"] or 0.0)
                    if status == "open" or pnl != 0.0:
                        return JSONResponse({
                            "ok": False,
                            "error": (f"row {row_id} carries ledger meaning "
                                      f"(status={status}, action={row['action']}, "
                                      f"pnl={pnl}); deleting it would break "
                                      f"sim_balance. Adjust the balance or cancel "
                                      f"the position instead."),
                        }, status_code=400)
            elif table == "system_config":
                cursor = await db.execute(
                    "SELECT key FROM system_config WHERE id=?", (row_id,))
                row = await cursor.fetchone()
                if row is not None and row["key"] in ("sim_balance", "schema_version"):
                    return JSONResponse({
                        "ok": False,
                        "error": (f"system_config.{row['key']} is part of the ledger/"
                                  f"schema contract and cannot be deleted"),
                    }, status_code=400)
            await db.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
            await db.commit()
        finally:
            await db.close()
        return {"ok": True}
