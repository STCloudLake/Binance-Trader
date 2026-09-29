"""Alerts API, alert rules and alert HTMX partials."""
import html as _html

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from db.database import get_db

from web.deps import _require_trader, _require_admin
from web.rendering import _render


def register(app: FastAPI, ctx) -> None:
    logger = ctx.logger

    @app.get("/api/alerts")
    async def get_alerts(limit: int = 50):
        try:
            db = await get_db()
            cursor = await db.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in await cursor.fetchall()]
            await db.close()
            return rows
        except Exception:
            return []

    @app.post("/api/alerts/{alert_id}/ack")
    async def ack_alert(alert_id: int, request: Request):
        if err := _require_trader(request): return err
        try:
            # `AlertManager.acknowledge_alert` is the single owner of the
            # acknowledged=1 write; go through it rather than duplicating the SQL.
            mgr = getattr(app.state, "alert_manager", None)
            if mgr is not None:
                await mgr.acknowledge_alert(alert_id)
            else:
                db = await get_db()
                await db.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
                await db.commit()
                await db.close()
        except Exception as e:
            logger.warning(f"Failed to ack alert {alert_id}: {e}")
        resp = HTMLResponse('<span class="text-green-400 text-xs">✓ Acked</span>')
        resp.headers["HX-Trigger"] = "alertAcked"
        return resp

    @app.get("/partials/alerts", response_class=HTMLResponse)
    async def partial_alerts(limit: int = 5):
        try:
            db = await get_db()
            cursor = await db.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in await cursor.fetchall()]
            await db.close()
        except Exception:
            rows = []
        html = ""
        for a in rows:
            # SECURITY (stored XSS): every column here is attacker-influenced
            # (news/AI text lands in `message`, the engine sets `type`/`symbol`),
            # and this partial is injected with innerHTML by the dashboard.  The
            # f-string used to emit the raw message; `html.escape()` now covers
            # every field that reaches the markup.  Jinja's autoescape covers the
            # *other* alert partials (`partials/alert_list.html`), not this
            # hand-built one.
            level = str(a.get("level") or "")
            message = _html.escape(str(a.get("message") or "")[:80], quote=True)
            level_cls = "badge-red" if level == "critical" else "badge-yellow" if level == "warning" else "badge-blue"
            html += f'''<div class="py-1 border-b border-slate-800 text-sm">
                <span class="badge {level_cls}">{_html.escape(level, quote=True)}</span>
                {message}
            </div>\n'''
        return HTMLResponse(html or '<div class="text-slate-500 text-sm">No alerts</div>')

    # ---- Filtered alerts API ----
    @app.get("/api/alerts/filtered")
    async def get_filtered_alerts(limit: int = 50, level: str = None,
                                   type: str = None, search: str = None):
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr:
            return []
        return await mgr.get_alerts(limit=limit, level=level, alert_type=type, search=search)

    @app.get("/api/alerts/counts")
    async def get_alert_counts(request: Request):
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr:
            counts = {"critical": 0, "warning": 0, "info": 0}
            rules_enabled = 0
        else:
            counts = await mgr.get_counts()
            rules = mgr.get_rules()
            rules_enabled = sum(1 for r in rules if r.get("enabled"))
        return _render("partials/alert_counts.html", {
            "request": request, "counts": counts, "rules_enabled": rules_enabled,
        })

    # ---- Clear all alerts ----
    @app.post("/api/alerts/clear")
    async def clear_all_alerts(request: Request):
        # Deleting the alert log destroys the operational audit trail.
        if err := _require_admin(request): return err
        try:
            db = await get_db()
            await db.execute("DELETE FROM alerts")
            await db.commit()
            await db.close()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---- Alert rules API ----
    @app.get("/api/alert-rules")
    async def get_alert_rules():
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr:
            return {"rules": []}
        return {"rules": mgr.get_rules()}

    @app.post("/api/alert-rules/{index}/toggle")
    async def toggle_alert_rule(index: int, request: Request):
        if err := _require_trader(request): return err
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr or index >= len(mgr.get_rules()):
            return {"ok": False}
        rules = mgr.get_rules()
        new_enabled = not rules[index].get("enabled", True)
        await mgr.update_rule(index, {"enabled": new_enabled})
        return {"ok": True, "enabled": new_enabled}

    @app.post("/api/alert-rules/{index}/remove")
    async def remove_alert_rule(index: int, request: Request):
        if err := _require_trader(request): return err
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr:
            return {"ok": False}
        await mgr.remove_rule(index)
        return {"ok": True}

    # ---- Alert partials (HTMX) ----
    @app.get("/partials/alerts-filtered", response_class=HTMLResponse)
    async def partials_alerts_filtered(request: Request, limit: int = 50,
                                        type: str = None, search: str = None):
        mgr = getattr(app.state, "alert_manager", None)
        alerts = await mgr.get_alerts(limit=limit, alert_type=type, search=search) if mgr else []
        return _render("partials/alert_list.html", {"request": request, "alerts": alerts})

    @app.get("/partials/alert-rules", response_class=HTMLResponse)
    async def partials_alert_rules(request: Request):
        mgr = getattr(app.state, "alert_manager", None)
        rules = mgr.get_rules() if mgr else []
        return _render("partials/alert_rules.html", {"request": request, "rules": rules})
