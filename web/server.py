import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader
import aiosqlite
import asyncio
import yaml
import re
import logging

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.auth.auth import AuthManager, User
from db.database import get_db, save_sim_balance, atomic_adjust_balance
import csv, io, os, shutil

logger = logging.getLogger(__name__)

# China Standard Time (UTC+8)
CST = timezone(timedelta(hours=8))
DEFAULT_BALANCE = 10000.0

# Jinja2 environment — created once and reused
_templates_dir = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(str(_templates_dir)))


def _fmt_time(utc_str):
    """Convert a UTC timestamp string from SQLite to CST (UTC+8) for display."""
    if not utc_str:
        return '-'
    try:
        dt = datetime.strptime(str(utc_str)[:19], '%Y-%m-%d %H:%M:%S')
        dt = dt.replace(tzinfo=timezone.utc).astimezone(CST)
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return str(utc_str)[:16]


_jinja_env.filters["fmt_time"] = _fmt_time


from web.i18n import get_translator


def _T(key: str) -> str:
    """Translate a key for inline HTML usage. Uses current config language."""
    return get_translator(_get_lang())(key)


def _get_lang() -> str:
    try:
        from app.config import Config
        c = Config._instance
        if c and c._loaded:
            return getattr(c, "language", "zh")
    except Exception:
        pass
    return "zh"


def _render(template_name: str, context: dict, lang: str = None) -> HTMLResponse:
    template = _jinja_env.get_template(template_name)
    lang = lang or _get_lang()
    context["_"] = get_translator(lang)
    context["lang"] = lang
    if context.get("request") and hasattr(context["request"], "state"):
        context["current_user"] = getattr(context["request"].state, "user", None)
    return HTMLResponse(template.render(**context))


def create_app(config: Config, event_bus: EventBus, auth_manager=None) -> FastAPI:
    app = FastAPI(title="Binance Trader", docs_url=None, redoc_url=None)
    if auth_manager:
        from starlette.middleware.base import BaseHTTPMiddleware
        app.add_middleware(auth_manager.create_middleware())
        app.state.auth_manager = auth_manager

    _balance_lock = asyncio.Lock()

    def _require_trader(request: Request):
        """Require trader or admin role. Returns None if OK, error response if denied."""
        user = getattr(request.state, "user", None)
        if not user or not user.is_trader:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        return None

    def _require_admin(request: Request):
        """Require admin role. Returns None if OK, error response if denied."""
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        return None

    async def _save_balance():
        await save_sim_balance(getattr(app.state, "balance", DEFAULT_BALANCE), config.db_path)
        # Sync to risk manager so position sizing works
        rm = getattr(app.state, "risk_manager", None)
        if rm:
            rm.update_balance(app.state.balance)

    # ---- Auth routes ----
    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return _render("login.html", {"request": request})

    @app.post("/api/auth/login")
    async def api_login(request: Request):
        import json as _json
        body = await request.body()
        data = _json.loads(body) if body else {}
        username = data.get("username", "")
        password = data.get("password", "")
        if not username or not password:
            return JSONResponse({"error": "Missing credentials"}, status_code=400)

        am = getattr(app.state, "auth_manager", None)
        if not am:
            return JSONResponse({"error": "Auth not configured"}, status_code=500)

        user_data = await am.get_user_by_username(username)
        if not user_data or not am.verify_password(password, user_data["password_hash"]):
            return JSONResponse({"error": "Invalid credentials"}, status_code=401)

        user = User(id=user_data["id"], username=user_data["username"],
                     role=user_data["role"], display_name=user_data.get("display_name", ""),
                     enabled=bool(user_data.get("enabled", 1)))
        session_token = am.create_session(user)
        jwt_token = am.create_jwt(user)
        await am.touch_login(user.id)

        response = JSONResponse({"ok": True, "token": jwt_token, "role": user.role})
        response.set_cookie("bt_session", session_token, httponly=True, samesite="lax",
                           secure=False, max_age=am.session_hours * 3600)
        return response

    @app.post("/api/auth/logout")
    async def api_logout(request: Request):
        session_token = request.cookies.get("bt_session")
        if session_token:
            am = getattr(app.state, "auth_manager", None)
            if am:
                am.destroy_session(session_token)
        response = JSONResponse({"ok": True})
        response.delete_cookie("bt_session")
        return response

    # ---- User management (admin only) ----
    @app.get("/api/users")
    async def list_users(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        am = getattr(app.state, "auth_manager", None)
        users = await am.list_users() if am else []
        return [{"id": u["id"], "username": u["username"], "role": u["role"],
                 "display_name": u.get("display_name",""), "enabled": u.get("enabled",1),
                 "created_at": u.get("created_at",""), "last_login": u.get("last_login","")} for u in users]

    @app.post("/api/users")
    async def create_user(request: Request, username: str = Form(...), password: str = Form(...),
                          role: str = Form("viewer"), display_name: str = Form("")):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        am = getattr(app.state, "auth_manager", None)
        await am.create_user(username, password, role, display_name)
        users = await am.list_users()
        return _render("partials/user_list.html", {"request": request, "users": users})

    @app.post("/api/users/{uid}")
    async def update_user(uid: int, request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        body = await request.json()
        am = getattr(app.state, "auth_manager", None)
        await am.update_user(uid, **body)
        users = await am.list_users()
        return _render("partials/user_list.html", {"request": request, "users": users})

    @app.delete("/api/users/{uid}")
    async def delete_user(uid: int, request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        am = getattr(app.state, "auth_manager", None)
        await am.update_user(uid, enabled=0)
        users = await am.list_users()
        return _render("partials/user_list.html", {"request": request, "users": users})

    @app.post("/api/users/{uid}/toggle")
    async def toggle_user(uid: int, request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        am = getattr(app.state, "auth_manager", None)
        user_data = await am.get_user_by_id(uid, include_disabled=True)
        if user_data:
            new_enabled = 0 if user_data.get("enabled", 1) else 1
            await am.update_user(uid, enabled=new_enabled)
        users = await am.list_users()
        return _render("partials/user_list.html", {"request": request, "users": users})

    @app.post("/api/auth/change-password")
    async def change_password(request: Request):
        user = getattr(request.state, "user", None)
        if not user:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        body = await request.json()
        am = getattr(app.state, "auth_manager", None)
        user_data = await am.get_user_by_id(user.id)
        if not am.verify_password(body.get("current_password", ""), user_data["password_hash"]):
            return JSONResponse({"error": "Current password incorrect"}, status_code=400)
        await am.change_password(user.id, body["new_password"])
        return {"ok": True}

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        ack_alerts = []
        ai_suggestions = []
        try:
            db = await get_db()
            cursor = await db.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT 5")
            ack_alerts = [dict(r) for r in await cursor.fetchall()]
            cursor = await db.execute("SELECT * FROM ai_suggestions WHERE status='pending' ORDER BY created_at DESC LIMIT 5")
            ai_suggestions = [dict(r) for r in await cursor.fetchall()]
            await db.close()
        except Exception:
            pass

        return _render("dashboard.html", {
            "request": request,
            "current_page": "dashboard",
            "mode": config.mode,
            "balance": getattr(app.state, "balance", DEFAULT_BALANCE),
            "positions": [],
            "max_trades": config.hard_limits.max_open_trades,
            "daily_pnl": 0.0,
            "ai_mode": config.ai_mode,
            "alerts": ack_alerts,
            "ai_suggestions": ai_suggestions,
        })

    @app.get("/strategies", response_class=HTMLResponse)
    async def strategies_page(request: Request):
        loader = getattr(app.state, "strategy_loader", None)
        strategies = []
        if loader:
            strategies = [s.model_dump() for s in loader.load_all()]

        return _render("strategies.html", {
            "request": request,
            "current_page": "strategies",
            "mode": config.mode,
            "strategies": strategies,
            "signal_weights": config.signal_weights.model_dump(),
            "risk_params": config.soft_params.model_dump(),
            "hard_limits": config.hard_limits.model_dump(),
        })

    @app.get("/ai", response_class=HTMLResponse)
    async def ai_page(request: Request):
        pending = []
        last_assessment = None
        try:
            db = await get_db()
            cursor = await db.execute("SELECT * FROM ai_suggestions WHERE status='pending' ORDER BY created_at DESC")
            pending = [dict(r) for r in await cursor.fetchall()]
            cursor = await db.execute(
                "SELECT * FROM ai_suggestions WHERE category='market_assessment' ORDER BY created_at DESC LIMIT 1")
            row = await cursor.fetchone()
            if row:
                last_assessment = dict(row).get("content", "")[:500]
            await db.close()
        except Exception:
            pass

        return _render("ai_panel.html", {
            "request": request,
            "current_page": "ai",
            "mode": config.mode,
            "ai_mode": config.ai_mode,
            "api_connected": bool(config.deepseek_api_key),
            "ai_model": config.ai_model,
            "pending_suggestions": pending,
            "last_assessment": last_assessment,
        })

    @app.get("/alerts", response_class=HTMLResponse)
    async def alerts_page(request: Request):
        mgr = getattr(app.state, "alert_manager", None)
        alerts = await mgr.get_alerts(limit=50) if mgr else []
        rules = mgr.get_rules() if mgr else []
        counts = await mgr.get_counts() if mgr else {"critical": 0, "warning": 0, "info": 0}
        return _render("alerts.html", {
            "request": request,
            "current_page": "alerts",
            "mode": config.mode,
            "alerts": alerts,
            "rules": rules,
            "counts": counts,
            "rules_enabled": sum(1 for r in rules if r.get("enabled")),
        })

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_trader:
            return RedirectResponse(url="/dashboard", status_code=302)
        rm = getattr(app.state, "risk_manager", None)
        cb_state = {}
        if rm:
            cb = rm.breaker
            cb_state = {
                "is_tripped": cb.is_tripped,
                "trip_reason": cb.trip_reason,
                "daily_pnl": round(cb.daily_pnl, 2),
                "consecutive_losses": cb.consecutive_losses,
            }
        return _render("settings.html", {
            "request": request,
            "current_page": "settings",
            "mode": config.mode,
            "spot_enabled": config.spot_enabled,
            "futures_enabled": config.futures_enabled,
            "binance_testnet": config.binance_testnet,
            "web_port": config.web_port,
            "binance_api_key": config.binance_api_key or "",
            "binance_api_secret": config.binance_api_secret or "",
            "deepseek_key": config.deepseek_api_key or "",
            "ai_base_url": config.ai_base_url or "https://api.deepseek.com",
            "ai_model": config.ai_model or "deepseek-chat",
            "language": getattr(config, "language", "en"),
            "ai_consult_interval": getattr(config, "ai_consult_interval", 60),
            "news_fetch_interval": config.news_fetch_interval,
            "max_articles": config.news_max_articles,
            "anomaly_threshold": config.anomaly_threshold_pct,
            "hard_limits": config.hard_limits,
            "ai_task_intervals": {k: v // 60 for k, v in config.ai_task_intervals.items()},
            "circuit_breaker": cb_state,
        })

    @app.get("/db-manager", response_class=HTMLResponse)
    async def db_manager_page(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return RedirectResponse(url="/dashboard", status_code=302)
        tables = ["trades", "alerts", "ai_suggestions", "orders", "positions", "system_config", "users"]
        return _render("db_manager.html", {"request": request, "current_page": "db_manager", "tables": tables})

    @app.get("/users", response_class=HTMLResponse)
    async def users_page(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return RedirectResponse(url="/dashboard", status_code=302)
        return _render("users.html", {"request": request, "current_page": "users"})

    # API Routes
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
    async def ack_alert(alert_id: int):
        try:
            db = await get_db()
            await db.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
            await db.commit()
            await db.close()
        except Exception as e:
            logger.warning(f"Failed to ack alert {alert_id}: {e}")
        resp = HTMLResponse('<span class="text-green-400 text-xs">✓ Acked</span>')
        resp.headers["HX-Trigger"] = "alertAcked"
        return resp

    @app.get("/api/ai-suggestions")
    async def get_ai_suggestions(status: str = "pending", limit: int = 10):
        try:
            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM ai_suggestions WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit)
            )
            rows = [dict(r) for r in await cursor.fetchall()]
            await db.close()
            return rows
        except Exception:
            return []

    @app.post("/api/ai-suggestions/{sid}/approve")
    async def approve_suggestion(sid: int, request: Request):
        if err := _require_trader(request): return err
        try:
            db = await get_db()
            cursor = await db.execute("SELECT * FROM ai_suggestions WHERE id=?", (sid,))
            row = await cursor.fetchone()
            if row:
                s = dict(row)
                cat = s.get("category", "")
                content = s.get("content", "")
                # Execute action based on category
                if cat == "risk_adjustment":
                    try:
                        result = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
                        pct = max(result.get("position_size_pct", 5.0), 1.0)
                        sl = max(result.get("stop_loss_pct", 2.0), 0.5)
                        lev = max(result.get("leverage", 2), 1)
                        config.update_soft_params(
                            risk_appetite=result.get("risk_appetite", "balanced"),
                            position_size_pct=pct,
                            stop_loss_pct=sl,
                            leverage=lev)
                    except Exception:
                        pass
                elif cat == "strategy_optimization":
                    # Mark as approved; strategy engine will reload on next tick
                    pass
                elif cat == "coin_selection":
                    # Coin selection recommendations are informational
                    pass
                await db.execute("UPDATE ai_suggestions SET status='approved' WHERE id=?", (sid,))
                await db.commit()
            await db.close()
        except Exception:
            pass
        return HTMLResponse(f'<span class="text-green-400 text-xs">✓ {_T("已批准并执行")}</span>')

    @app.post("/api/ai-suggestions/{sid}/reject")
    async def reject_suggestion(sid: int, request: Request):
        if err := _require_trader(request): return err
        try:
            db = await get_db()
            await db.execute("UPDATE ai_suggestions SET status='rejected' WHERE id=?", (sid,))
            await db.commit()
            await db.close()
        except Exception:
            pass
        return HTMLResponse('<span class="text-red-400 text-xs">✗ Rejected</span>')

    # HTMX Partial HTML routes — return HTML fragments for inline swapping
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
            level_cls = "badge-red" if a["level"] == "critical" else "badge-yellow" if a["level"] == "warning" else "badge-blue"
            html += f'''<div class="py-1 border-b border-slate-800 text-sm">
                <span class="badge {level_cls}">{a["level"]}</span>
                {a["message"][:80]}
            </div>\n'''
        return HTMLResponse(html or '<div class="text-slate-500 text-sm">No alerts</div>')

    # ---- WebSocket for real-time alert push ----
    @app.websocket("/ws/alerts")
    async def ws_alerts(websocket: WebSocket):
        await websocket.accept()
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr:
            await websocket.close()
            return
        client_id = str(id(websocket))
        queue = mgr.register_ws(client_id)
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                    await websocket.send_text(msg)
                except asyncio.TimeoutError:
                    await websocket.send_text('{"ping": true}')
        except WebSocketDisconnect:
            pass
        finally:
            mgr.unregister_ws(client_id)

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
        if err := _require_trader(request): return err
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

    @app.get("/partials/user-list", response_class=HTMLResponse)
    async def partial_user_list(request: Request):
        am = getattr(app.state, "auth_manager", None)
        users = await am.list_users() if am else []
        return _render("partials/user_list.html", {"request": request, "users": users})

    @app.get("/partials/ai-suggestions", response_class=HTMLResponse)
    async def partial_ai_suggestions(status: str = "pending", limit: int = 5):
        try:
            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM ai_suggestions WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit))
            rows = [dict(r) for r in await cursor.fetchall()]
            await db.close()
        except Exception:
            rows = []
        html = ""
        for s in rows:
            approval_btns = ""
            if s["status"] == "pending":
                approval_btns = f'''
                <div class="text-xs text-slate-500 mt-1" id="btns-{s['id']}">
                    <button class="text-green-400 hover:underline" hx-post="/api/ai-suggestions/{s['id']}/approve" hx-target="#btns-{s['id']}" hx-swap="outerHTML">Approve</button>
                    <button class="text-red-400 hover:underline ml-2" hx-post="/api/ai-suggestions/{s['id']}/reject" hx-target="#btns-{s['id']}" hx-swap="outerHTML">Reject</button>
                </div>'''
            html += f'''<div class="py-1 border-b border-slate-800 text-sm">
                <span class="badge badge-blue">{s["category"]}</span>
                {s["content"][:100]}
                {approval_btns}
            </div>\n'''
        return HTMLResponse(html or '<div class="text-slate-500 text-sm">No pending suggestions</div>')

    @app.post("/api/ai-mode")
    async def set_ai_mode(request: Request, mode: str = Form(...)):
        if err := _require_trader(request): return err
        config.ai_mode = mode
        # Persist to config.yaml
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("ai", {})["mode"] = mode
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        except Exception:
            pass
        return HTMLResponse(f'<span class="text-green-400">{_T("AI 模式已更新")}</span>')

    @app.post("/api/signal-weights")
    async def update_signal_weights(request: Request, indicator: float = Form(0.5), ml: float = Form(0.3), news: float = Form(0.2)):
        if err := _require_trader(request): return err
        config.update_signal_weights(indicator=indicator, ml=ml, news=news)
        return HTMLResponse('<span class="text-green-400">Weights updated</span>')

    # ---- Trading endpoints ----
    @app.post("/api/trade")
    async def execute_trade(request: Request, symbol: str = Form(...), side: str = Form(...),
                            amount_usdt: float = Form(100), position_type: str = Form("satellite"),
                            stop_loss_pct: float = Form(2.0), trader: str = Form("manual"),
                            strategy_name: str = Form("")):
        if err := _require_trader(request): return err
        executor = getattr(app.state, "executor", None)
        risk_manager = getattr(app.state, "risk_manager", None)
        if not executor:
            return HTMLResponse(f'<span class="text-red-400">{_T("交易器未就绪")}</span>')

        # Try WebSocket cache first, then REST API
        price_fn = getattr(app.state, "get_price", None)
        current_price = price_fn(symbol) if price_fn else None
        if current_price is None:
            try:
                from binance import AsyncClient
                client = await AsyncClient.create()
                ticker = await client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker["price"])
                await client.close_connection()
            except Exception:
                pass
        if not current_price or current_price <= 0:
            return HTMLResponse('<span class="text-red-400">无法获取实时价格，请检查网络或 API 配置</span>')

        qty = amount_usdt / current_price
        sl = current_price * (1 - stop_loss_pct / 100) if side == "long" else current_price * (1 + stop_loss_pct / 100)

        # Route through risk manager for all safety checks
        if risk_manager:
            signal = {
                "symbol": symbol, "side": side,
                "price": current_price, "quantity": round(qty, 6),
                "stop_loss": round(sl, 2),
                "position_type": position_type,
                "amount_usdt": amount_usdt,
                "trader": trader,
                "strategy_name": strategy_name,
                "strategy": "manual",
                "timeframe": "1h",
                "confidence": 1.0,
            }
            result = await risk_manager.check_signal(signal)
            if not result.approved:
                return HTMLResponse(f'<span class="text-red-400">风控拒绝: {result.reason}</span>')
            # Use the risk-adjusted quantity (capped at user's request)
            if result.adjusted_quantity is not None and result.adjusted_quantity < signal["quantity"]:
                signal["quantity"] = result.adjusted_quantity
                signal["amount_usdt"] = result.adjusted_quantity * current_price
            if result.adjusted_stop_loss is not None:
                signal["stop_loss"] = result.adjusted_stop_loss
            if result.adjusted_leverage is not None:
                signal["leverage"] = result.adjusted_leverage
            final_qty = signal["quantity"]

            await event_bus.publish(Event(EventType.ORDER_REQUEST, signal))
            resp = HTMLResponse(
                f'<span class="text-green-400">✓ {side.upper()} {final_qty:.6f} {symbol} @ {current_price:.4f} | SL: {signal["stop_loss"]:.4f}</span>'
            )
            resp.headers["HX-Trigger"] = "tradeUpdated"
            return resp
        else:
            # Fallback: direct execution (only when risk manager is unavailable)
            async with _balance_lock:
                balance = getattr(app.state, "balance", DEFAULT_BALANCE)
                if amount_usdt > balance:
                    return HTMLResponse(f'<span class="text-red-400">余额不足: 需要 {amount_usdt:.0f} USDT，可用 {balance:.0f} USDT</span>')
                app.state.balance = balance - amount_usdt
                await _save_balance()

            try:
                await executor._execute_sim({
                    "symbol": symbol, "side": side,
                    "price": current_price, "quantity": round(qty, 6),
                    "stop_loss": round(sl, 2),
                    "position_type": position_type,
                    "amount_usdt": amount_usdt,
                    "trader": trader,
                    "strategy_name": strategy_name,
                })
                resp = HTMLResponse(
                    f'<span class="text-green-400">✓ {side.upper()} {qty:.6f} {symbol} @ {current_price:.4f} | SL: {sl:.4f}</span>'
                )
                resp.headers["HX-Trigger"] = "tradeUpdated"
                return resp
            except Exception as e:
                return HTMLResponse(f'<span class="text-red-400">Error: {e}</span>')

    @app.get("/partials/stats", response_class=HTMLResponse)
    async def partial_stats():
        from db.database import load_sim_balance
        executor = getattr(app.state, "executor", None)
        positions = executor.get_open_positions() if executor else {}
        pos_count = len(positions)
        # Reload from DB so auto-trade balance changes are reflected
        balance = await load_sim_balance(config.db_path)
        app.state.balance = balance  # sync web state
        invested = sum(p.get("amount_usdt", p.get("quantity", 0) * p.get("entry_price", 0)) for p in positions.values())
        total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions.values())
        return _render("partials/account_summary.html", {
            "request": None, "balance": balance, "pos_count": pos_count,
            "max_trades": config.hard_limits.max_open_trades,
            "total_pnl": total_pnl, "invested": invested,
        })

    @app.get("/partials/positions", response_class=HTMLResponse)
    async def partial_positions():
        executor = getattr(app.state, "executor", None)
        positions_raw = executor.get_open_positions() if executor else {}
        if not positions_raw:
            return _render("partials/positions_table.html", {"request": None, "positions": []})
        # Fetch live prices for PnL
        price_fn = getattr(app.state, "get_price", None)
        rest_client = None
        pos_list = []
        for pos in positions_raw.values():
            live_price = price_fn(pos["symbol"]) if price_fn else None
            if live_price is None:
                try:
                    if rest_client is None:
                        from binance import AsyncClient
                        rest_client = await AsyncClient.create()
                    ticker = await rest_client.get_symbol_ticker(symbol=pos["symbol"])
                    live_price = float(ticker["price"])
                except Exception:
                    live_price = pos["entry_price"]
            pnl = (live_price - pos["entry_price"]) * pos["quantity"] if pos["side"] == "long" else (pos["entry_price"] - live_price) * pos["quantity"]
            pos["unrealized_pnl"] = pnl
            pos["current_price"] = live_price
            pos_list.append({
                "symbol": pos["symbol"], "side": pos.get("side", "long"),
                "quantity": pos["quantity"], "entry_price": pos["entry_price"],
                "live_price": live_price, "pnl": pnl,
                "position_type": pos.get("position_type", "satellite"),
            })
        if rest_client:
            try: await rest_client.close_connection()
            except Exception: pass
        return _render("partials/positions_table.html", {"request": None, "positions": pos_list})

    @app.post("/api/trade/close/{symbol}")
    async def close_position(symbol: str, request: Request, reduce_pct: float = Form(100)):
        if err := _require_trader(request): return err
        executor = getattr(app.state, "executor", None)
        if not executor:
            return HTMLResponse(f'<span class="text-red-400">{_T("交易器未就绪")}</span>')
        positions = executor.get_open_positions()
        if symbol not in positions:
            return HTMLResponse(f'<span class="text-red-400">{_T("无")} {symbol} {_T("持仓")}</span>')

        # Get current price
        current_price = None
        price_fn = getattr(app.state, "get_price", None)
        current_price = price_fn(symbol) if price_fn else None
        if current_price is None:
            try:
                from binance import AsyncClient
                client = await AsyncClient.create()
                ticker = await client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker["price"])
                await client.close_connection()
            except Exception:
                pass
        if not current_price:
            return HTMLResponse('<span class="text-red-400">无法获取实时价格</span>')

        reduce_pct = min(100, max(1, reduce_pct))

        # Delegate to executor for core close logic (DB persistence + position tracking)
        result = await executor.close_position(symbol, reduce_pct, current_price)
        if not result.get("ok"):
            return HTMLResponse(f'<span class="text-red-400">{result.get("error", "关闭失败")}</span>')

        # Atomically update balance (prevents RMW races with auto-trades)
        delta = result.get("invested_returned", 0) + result.get("pnl", 0)
        new_balance = await atomic_adjust_balance(delta, config.db_path)
        app.state.balance = new_balance
        # Sync to risk manager
        rm = getattr(app.state, "risk_manager", None)
        if rm:
            rm.update_balance(new_balance)

        if reduce_pct >= 100:
            msg = f'✓ 已平仓 {symbol} | P&L: {result["pnl"]:.4f} USDT ({result["pnl_pct"]:.4f}%)'
        else:
            msg = f'✓ 减持 {symbol} {reduce_pct:.0f}% | P&L: {result["pnl"]:.4f} USDT ({result["pnl_pct"]:.4f}%)'

        resp = HTMLResponse(f'<span class="text-green-400">{msg}</span>')
        resp.headers["HX-Trigger"] = "tradeUpdated"
        return resp

    @app.get("/api/price/{symbol}")
    async def get_price(symbol: str):
        price_fn = getattr(app.state, "get_price", None)
        price = price_fn(symbol) if price_fn else None
        if price is None:
            try:
                from binance import AsyncClient
                client = await AsyncClient.create()
                ticker = await client.get_symbol_ticker(symbol=symbol)
                price = float(ticker["price"])
                await client.close_connection()
            except Exception as e:
                return {"error": str(e)}
        return {"symbol": symbol, "price": price}

    @app.get("/api/kline/{symbol}")
    async def get_kline(symbol: str, interval: str = "1h", limit: int = 200):
        """Return OHLCV candle data for chart rendering."""
        try:
            from binance import AsyncClient
            client = await AsyncClient.create()
            klines = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
            await client.close_connection()
            data = []
            for k in klines:
                data.append({
                    "time": k[0] // 1000,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            return data
        except Exception as e:
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": str(e)}, status_code=500)

    # ---- Strategy CRUD ----
    @app.get("/api/strategy/{name}")
    async def get_strategy(name: str):
        loader = getattr(app.state, "strategy_loader", None)
        if not loader:
            return JSONResponse({"error": "No loader"}, status_code=500)
        try:
            s = loader.load(name)
            return s.model_dump()
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    @app.post("/api/strategy")
    async def create_strategy(request: Request):
        if err := _require_trader(request): return err
        loader = getattr(app.state, "strategy_loader", None)
        if not loader:
            return JSONResponse({"error": "No loader"}, status_code=500)
        try:
            from core.strategy.loader import StrategyConfig, MLConfig
            body = await request.json()
            ml = body.get("ml_config")
            if ml:
                body["ml_config"] = MLConfig(**ml)
            config = StrategyConfig(**body)
            loader.save(config)
            return JSONResponse({"ok": True, "name": config.name})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.put("/api/strategy/{name}")
    async def update_strategy(name: str, request: Request):
        if err := _require_trader(request): return err
        loader = getattr(app.state, "strategy_loader", None)
        if not loader:
            return JSONResponse({"error": "No loader"}, status_code=500)
        try:
            from core.strategy.loader import StrategyConfig, MLConfig
            body = await request.json()
            new_name = body.get("name", name)
            ml = body.get("ml_config")
            if ml:
                body["ml_config"] = MLConfig(**ml)
            config = StrategyConfig(**body)
            # Save new config first; only delete old file after save succeeds
            loader.save(config)
            if new_name != name:
                loader.delete(name)
            return JSONResponse({"ok": True, "name": config.name})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/strategy/{name}/toggle")
    async def toggle_strategy(name: str, request: Request):
        if err := _require_trader(request): return err
        loader = getattr(app.state, "strategy_loader", None)
        if not loader:
            return JSONResponse({"error": "No loader"}, status_code=500)
        try:
            s = loader.load(name)
            s.enabled = not s.enabled
            loader.save(s)
            return JSONResponse({"ok": True, "enabled": s.enabled})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.delete("/api/strategy/{name}")
    async def delete_strategy(name: str, request: Request):
        if err := _require_trader(request): return err
        loader = getattr(app.state, "strategy_loader", None)
        if not loader:
            return JSONResponse({"error": "No loader"}, status_code=500)
        try:
            loader.delete(name)
            return JSONResponse({"ok": True})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/strategy/reload")
    async def reload_strategies(request: Request):
        if err := _require_trader(request): return err
        """Reload all strategy YAML files into the engine without restarting."""
        engine = getattr(app.state, "strategy_engine", None)
        if not engine:
            return JSONResponse({"error": "No engine"}, status_code=500)
        try:
            all_s = engine.loader.load_all()
            for s in all_s:
                if not s.timeframes:
                    from loguru import logger
                    logger.warning(f"Strategy '{s.name}' has no timeframes — will never evaluate!")
            engine._strategies = {s.name: s for s in all_s}
            # Re-evaluate to populate signal cache immediately
            await engine.evaluate_all_now()
            count = len(engine._strategies)
            names = list(engine._strategies.keys())
            return JSONResponse({"ok": True, "count": count, "strategies": names})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/strategy-recommend")
    async def recommend_strategy(request: Request):
        if err := _require_trader(request): return err
        api_key = config.deepseek_api_key
        if not api_key:
            return JSONResponse({"error": "No API key"}, status_code=500)
        loader = getattr(app.state, "strategy_loader", None)
        existing = []
        if loader:
            for s in loader.load_all():
                existing.append({
                    "name": s.name, "mode": s.mode, "timeframes": s.timeframes,
                    "indicators": list(s.indicators.keys()),
                    "entry_long": s.entry_conditions.get("long", []),
                    "exit_long": s.exit_conditions.get("long", []),
                })
        existing_str = "\n".join([str(e) for e in existing]) if existing else "No existing strategies"
        prompt = f"""Based on these existing trading strategies:
{existing_str}

Recommend a NEW trading strategy for cryptocurrency. Choose indicators, timeframes, entry/exit/reduce conditions that complement (not duplicate) the existing ones.

CRITICAL RULES — your strategy MUST pass these conflict checks:
1. LONG vs SHORT entry: Use mutually exclusive conditions (e.g. RSI<35 long vs RSI>70 short, or close>middle long vs close<middle short). Never allow both long AND short entry to trigger simultaneously.
2. Entry vs Exit: Entry and exit thresholds must have a gap to prevent whipsaw. Example: enter at RSI<35, exit at RSI>65 (30-point gap). Never use adjacent thresholds.
3. Exit vs Reduce: Exit conditions should be stricter than reduce conditions (e.g. reduce 50% at RSI>55, full exit at RSI>65). Reduce fires first, lock in partial profit, then exit later if trend reverses.
4. Include BOTH long AND short conditions for all sections (entry/exit/reduce). No one-sided strategies.
5. All numeric parameters must be explicit (never use empty strings or null). Default: period=14, stddev=2, fast=12, slow=26, signal=9.

Return ONLY valid JSON in this exact format:
{{
  "name": "Strategy Name",
  "mode": "trend",
  "timeframes": ["1h", "4h"],
  "indicators": {{"rsi": {{"period":14, "source":"close"}}, "macd": {{"fast":12, "slow":26, "signal":9}}}},
  "entry_conditions": {{"long": ["rsi < 35 and macd_histogram > 0"], "short": ["rsi > 70 and macd_histogram < 0"]}},
  "exit_conditions": {{"long": ["rsi > 65"], "short": ["rsi < 35"]}},
  "reduce_conditions": {{"long": [{{"condition": "rsi > 55 and close > bollinger_upper", "reduce_pct": 50}}], "short": [{{"condition": "rsi < 40 and close < bollinger_lower", "reduce_pct": 50}}]}},
  "ml_config": {{"enabled": true, "confidence_threshold": 0.6, "features": ["rsi","macd_histogram","volume_ratio","price_momentum_24h"], "weight": 0.3}},
  "rationale": "Why this strategy complements existing ones AND how it passes all conflict checks"
}}"""
        raw = ""
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, base_url=config.ai_base_url)
            resp = await client.chat.completions.create(
                model=config.ai_model,
                messages=[{"role": "system", "content": "You are a quantitative crypto trading strategist. Design strategies with no logical conflicts: long/short entries must be mutually exclusive, entry/exit thresholds must have gaps, reduce fires before exit, both long and short sides required, all parameters numeric. Always respond in valid JSON only, no markdown."},
                          {"role": "user", "content": prompt}],
                max_tokens=4000, temperature=0.6,
            )
            raw = (resp.choices[0].message.content or "").strip()
            import json as j
            # Extract JSON: find the first { and last }
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                text = raw[start:end + 1]
            else:
                text = raw
            # Strip markdown code fences
            for prefix in ["```json", "```"]:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            for suffix in ["```", "```json"]:
                if text.endswith(suffix):
                    text = text[:-len(suffix)].strip()
            if not text:
                return JSONResponse({"error": "AI returned empty response"}, status_code=500)
            return j.loads(text)
        except Exception as e:
            err = str(e)
            detail = err[:200]
            if "JSONDecodeError" in type(e).__name__ or "Expecting" in err:
                detail = f"JSON解析失败 [{err[:100]}] 原始响应前200字: {raw[:200]}"
            return JSONResponse({"error": detail}, status_code=500)

    @app.get("/api/trades")
    async def get_trades(limit: int = 30):
        try:
            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in await cursor.fetchall()]
            await db.close()
            return rows
        except Exception:
            return []

    @app.get("/partials/trades", response_class=HTMLResponse)
    async def partial_trades(page: int = 1, per_page: int = 20):
        try:
            db = await get_db()
            cursor = await db.execute("SELECT COUNT(*) as cnt FROM trades")
            total = (await cursor.fetchone())["cnt"]
            offset = (page - 1) * per_page
            cursor = await db.execute(
                "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ? OFFSET ?",
                (per_page, offset))
            rows = [dict(r) for r in await cursor.fetchall()]
            await db.close()
        except Exception:
            rows = []
            total = 0

        total_pages = max(1, (total + per_page - 1) // per_page)

        # Build trade list for template
        trade_list = []
        for t in rows:
            action = t.get('action', 'open')
            rpct = float(t.get('reduce_pct', 0) or 0)
            trade_list.append({
                "symbol": t["symbol"], "side": t["side"],
                "quantity": t["quantity"], "entry_price": t["entry_price"],
                "exit_price": t.get("exit_price"),
                "pnl": t.get("pnl") or 0,
                "trader": t.get("trader", "manual"),
                "strategy": t.get("strategy", ""),
                "strategy_name": t.get("strategy_name", ""),
                "action": action, "reduce_pct": rpct,
                "status": t.get("status", "open"),
                "opened_at": _fmt_time(t.get("opened_at")),
            })

        return _render("partials/trade_history.html", {
            "request": None, "trades": trade_list,
            "page": page, "total_pages": total_pages, "total": total,
            "per_page": per_page,
        })

    @app.get("/api/deepseek-models")
    async def get_deepseek_models(refresh: str = "0"):
        # Return cached models unless refresh=1
        import json as j
        cache_path = Path(config.data_dir) / "deepseek_models.json"
        if refresh != "1" and cache_path.exists():
            try:
                with open(cache_path) as f:
                    return j.load(f)
            except Exception:
                pass

        api_key = config.deepseek_api_key
        if not api_key:
            return {"error": "No API key configured", "models": []}
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, base_url=config.ai_base_url)
            models = await client.models.list()
            model_list = [{"id": m.id, "owned_by": m.owned_by} for m in models.data]
            model_list.sort(key=lambda x: x["id"])
            result = {"models": model_list}
            with open(cache_path, "w") as f:
                j.dump(result, f)
            return result
        except Exception as e:
            fallback = {"error": str(e), "models": [{"id": config.ai_model, "owned_by": "current"}]}
            if cache_path.exists():
                try:
                    with open(cache_path) as f:
                        return j.load(f)
                except Exception:
                    pass
            return fallback

    @app.get("/api/market-state")
    async def get_market_state():
        try:
            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM ai_suggestions WHERE category='market_assessment' ORDER BY created_at DESC LIMIT 1")
            row = await cursor.fetchone()
            await db.close()
            if row:
                r = dict(row)
                return {"regime": "assessed", "content": r["content"][:500], "created_at": str(r["created_at"])[:16]}
        except Exception:
            pass
        return {"regime": "waiting", "content": "AI 市场评估尚未生成，后台每小时自动运行"}

    @app.get("/api/strategy-monitor")
    async def get_strategy_monitor():
        engine = getattr(app.state, "strategy_engine", None)
        if not engine:
            return {"strategies": [], "active_count": 0, "total_count": 0}
        return engine.get_monitor_state()

    @app.get("/api/ai-heartbeat", response_class=HTMLResponse)
    async def get_ai_heartbeat():
        """Return AI task status as HTML for the heartbeat panel."""
        import time as time_m
        intervals = config.ai_task_intervals
        tasks = [
            ("market_assessment", "市场评估", intervals["market_assessment"]),
            ("coin_selection", "币种选择", intervals["coin_selection"]),
            ("strategy_optimization", "策略优化", intervals["strategy_optimization"]),
            ("risk_adjustment", "风控调整", intervals["risk_adjustment"]),
        ]
        ctl = getattr(app.state, "ai_controller", None)
        running = ctl._running if ctl else False
        now = time_m.time()
        task_data = []
        try:
            db = await get_db()
            for key, label, interval_sec in tasks:
                cursor = await db.execute("SELECT value FROM system_config WHERE key=?", (f"ai_last_{key}",))
                row = await cursor.fetchone()
                last_ts = float(row["value"]) if row else 0
                cursor = await db.execute("SELECT value FROM system_config WHERE key=?", (f"ai_count_{key}",))
                row2 = await cursor.fetchone()
                count = int(float(row2["value"])) if row2 else 0
                cursor = await db.execute(
                    "SELECT COUNT(*) as cnt FROM ai_suggestions WHERE category=? AND created_at > datetime('now', '-1 day')",
                    (key,))
                row3 = await cursor.fetchone()
                recent = row3["cnt"] if row3 else 0
                mins = int((now - last_ts) / 60) if last_ts > 0 else 0
                hrs = int((now - last_ts) / 3600) if last_ts > 0 else 0
                status = 'ok' if last_ts > 0 and (now - last_ts) < interval_sec * 2 else ('timeout' if last_ts > 0 else 'waiting')
                # Format interval for display
                if interval_sec < 3600:
                    ival_str = f"{interval_sec//60}"
                    ival_unit = "分钟"
                elif interval_sec < 86400:
                    ival_str = f"{interval_sec//3600}"
                    ival_unit = "小时"
                else:
                    ival_str = f"{interval_sec//86400}"
                    ival_unit = "天"
                task_data.append({
                    "label": label, "interval_str": ival_str, "interval_unit": ival_unit,
                    "interval_sec": interval_sec, "last_ts": last_ts, "status": status,
                    "count": count, "recent": recent,
                    "ago_mins": mins, "ago_hrs": hrs,
                })
            await db.close()
        except Exception:
            pass
        return _render("partials/ai_heartbeat.html", {
            "request": None, "running": running, "tasks": task_data,
        })

    @app.post("/api/consult")
    async def consult_ai(request: Request, prompt: str = Form(...)):
        if err := _require_trader(request): return err
        api_key = config.deepseek_api_key
        if not api_key:
            return HTMLResponse('<div class="text-red-400">DeepSeek API key 未配置</div>')
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, base_url=config.ai_base_url)
            resp = await client.chat.completions.create(
                model=config.ai_model,
                messages=[
                    {"role": "system", "content": "You are a professional crypto trading analyst. Reply in Chinese under 300 words. Structure your response with these labeled sections:\n【综合判断】bullish/bearish/neutral with reason\n【建议操作】long/short/wait\n【置信度】0-100%\n【关键支撑】price levels\n【关键阻力】price levels\n【风险提示】key risks"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.4,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # Convert to safe HTML
            safe = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Apply formatting: bold labels, newlines, colored verdict
            safe = re.sub(r'【(.+?)】', r'<b class="text-sky-300">【\1】</b>', safe)
            safe = safe.replace("\n", "<br>")
            # Inject verdict color
            if "bullish" in raw.lower() or "看涨" in raw:
                safe = '<span class="badge badge-green text-sm mb-2">看涨 Bullish</span><br>' + safe
            elif "bearish" in raw.lower() or "看跌" in raw:
                safe = '<span class="badge badge-red text-sm mb-2">看跌 Bearish</span><br>' + safe
            else:
                safe = '<span class="badge badge-yellow text-sm mb-2">观望 Wait</span><br>' + safe
            return HTMLResponse(f'<div class="text-slate-200 text-sm leading-relaxed">{safe}</div>')
        except Exception as e:
            return HTMLResponse(f'<div class="text-red-400">API 错误: {str(e)[:200]}</div>')

    @app.get("/")
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/dashboard")

    @app.get("/api/news-sources")
    async def get_news_sources():
        try:
            db = await get_db()
            cursor = await db.execute("SELECT * FROM news_sources ORDER BY priority DESC")
            rows = [dict(r) for r in await cursor.fetchall()]
            await db.close()
            html = ""
            for ns in rows:
                status = "badge-green" if ns["enabled"] else "badge-red"
                html += f'''<div class="py-2 border-b border-slate-800 flex justify-between items-center">
                    <div><span class="font-bold">{ns["name"]}</span>
                    <span class="text-xs text-slate-500 ml-2">{ns["type"]}</span></div>
                    <span class="badge {status}">{'Enabled' if ns['enabled'] else 'Disabled'}</span>
                </div>\n'''
            if not html:
                html = '<div class="text-slate-500 text-sm">No sources configured. Default: web search.</div>'
            return HTMLResponse(html)
        except Exception:
            return HTMLResponse('<div class="text-slate-500">Error loading sources</div>')

    @app.post("/api/settings/deepseek")
    async def save_deepseek_settings(request: Request, api_key: str = Form(""), base_url: str = Form(""), model: str = Form("")):
        if err := _require_trader(request): return err
        secrets_path = Path(__file__).parent.parent / "config" / "secrets.yaml"
        data = {}
        if secrets_path.exists():
            with open(secrets_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        # Preserve existing binance keys only if they were already in the YAML file
        if "binance" not in data:
            existing_binance = data.get("binance", {})
            if config.binance_api_key and not existing_binance.get("api_key"):
                existing_binance["api_key"] = config.binance_api_key
            if config.binance_api_secret and not existing_binance.get("api_secret"):
                existing_binance["api_secret"] = config.binance_api_secret
            if existing_binance:
                data["binance"] = existing_binance
        data["deepseek"] = {"api_key": api_key}
        with open(secrets_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
        config.deepseek_api_key = api_key
        config.ai_base_url = base_url
        config.ai_model = model
        # Persist model + base_url to config YAML
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("ai", {})["model"] = model
            cfg.setdefault("ai", {})["base_url"] = base_url
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False)
        return HTMLResponse('<span class="text-green-400 text-sm">✓ DeepSeek 设置已保存</span>')

    @app.post("/api/settings/ai-news")
    async def save_ai_news_settings(request: Request,
        language: str = Form("en"), ai_consult_interval: int = Form(60),
        news_fetch_interval: int = Form(30), max_articles: int = Form(10),
        anomaly_threshold: float = Form(3.0),
        task_market_assessment: int = Form(60), task_coin_selection: int = Form(240),
        task_strategy_optimization: int = Form(1440), task_risk_adjustment: int = Form(1440)):
        if err := _require_trader(request): return err
        config.language = language
        config.ai_consult_interval = ai_consult_interval
        config.news_fetch_interval = news_fetch_interval
        config.news_max_articles = max_articles
        config.anomaly_threshold_pct = anomaly_threshold
        config.ai_task_intervals = {
            "market_assessment": task_market_assessment * 60,
            "coin_selection": task_coin_selection * 60,
            "strategy_optimization": task_strategy_optimization * 60,
            "risk_adjustment": task_risk_adjustment * 60,
        }
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("ai", {})["consult_interval_minutes"] = ai_consult_interval
            cfg.setdefault("ai", {})["tasks"] = {
                "market_assessment_minutes": task_market_assessment,
                "coin_selection_minutes": task_coin_selection,
                "strategy_optimization_minutes": task_strategy_optimization,
                "risk_adjustment_minutes": task_risk_adjustment,
            }
            cfg.setdefault("news", {})["fetch_interval_minutes"] = news_fetch_interval
            cfg.setdefault("news", {})["max_articles_per_symbol"] = max_articles
            cfg.setdefault("news", {})["anomaly_threshold_pct"] = anomaly_threshold
            cfg["language"] = language
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        except Exception:
            pass
        return HTMLResponse('<span class="text-green-400 text-sm">✓ AI & 新闻设置已保存</span>')

    @app.post("/api/settings/risk")
    async def save_risk_settings(request: Request, max_daily_drawdown: float = Form(5.0), max_daily_loss: float = Form(500.0),
                                  max_open_trades: int = Form(8), max_position_size_pct: float = Form(10.0),
                                  max_leverage: int = Form(3), max_consecutive_losses: int = Form(5),
                                  circuit_breaker_action: str = Form("block_only"),
                                  trailing_stop_enabled: str = Form("0"),
                                  trailing_stop_distance_pct: float = Form(2.0),
                                  emergency_stop_enabled: str = Form("0"),
                                  emergency_stop_threshold_pct: float = Form(-5.0),
                                  spot_enabled: str = Form("1"), futures_enabled: str = Form("0")):
        if err := _require_trader(request): return err
        config.hard_limits.max_daily_drawdown_pct = max_daily_drawdown
        config.hard_limits.max_daily_loss_usdt = max_daily_loss
        config.hard_limits.max_open_trades = max_open_trades
        config.hard_limits.max_position_size_pct = max_position_size_pct
        config.hard_limits.max_leverage = max_leverage
        config.hard_limits.max_consecutive_losses = max_consecutive_losses
        config.hard_limits.circuit_breaker_action = circuit_breaker_action
        config.hard_limits.trailing_stop_enabled = trailing_stop_enabled == "1"
        config.hard_limits.trailing_stop_distance_pct = trailing_stop_distance_pct
        config.hard_limits.emergency_stop_enabled = emergency_stop_enabled == "1"
        config.hard_limits.emergency_stop_threshold_pct = emergency_stop_threshold_pct
        config.spot_enabled = spot_enabled == "1"
        config.futures_enabled = futures_enabled == "1"
        # Sync to running circuit breaker
        rm = getattr(app.state, "risk_manager", None)
        if rm:
            rm.breaker.max_consecutive_losses = max_consecutive_losses
            rm.breaker.max_daily_drawdown_pct = max_daily_drawdown
            rm.breaker.max_daily_loss_usdt = max_daily_loss
        # Persist to risk_params.yaml (loaded after config.yaml, overrides it)
        risk_path = Path(__file__).parent.parent / "config" / "risk_params.yaml"
        try:
            with open(risk_path, encoding="utf-8") as f:
                rp = yaml.safe_load(f) or {}
            hl = rp.setdefault("hard_limits", {})
            hl["max_daily_drawdown_pct"] = max_daily_drawdown
            hl["max_daily_loss_usdt"] = max_daily_loss
            hl["max_open_trades"] = max_open_trades
            hl["max_position_size_pct"] = max_position_size_pct
            hl["max_leverage"] = max_leverage
            hl["max_consecutive_losses"] = max_consecutive_losses
            hl["circuit_breaker_action"] = circuit_breaker_action
            hl["trailing_stop_enabled"] = trailing_stop_enabled == "1"
            hl["trailing_stop_distance_pct"] = trailing_stop_distance_pct
            hl["emergency_stop_enabled"] = emergency_stop_enabled == "1"
            hl["emergency_stop_threshold_pct"] = emergency_stop_threshold_pct
            with open(risk_path, "w", encoding="utf-8") as f:
                yaml.dump(rp, f, default_flow_style=False, allow_unicode=True)
            # Also persist trading toggles to config.yaml
            cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("trading", {})["spot_enabled"] = spot_enabled == "1"
            cfg.setdefault("trading", {})["futures_enabled"] = futures_enabled == "1"
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        except Exception:
            pass
        return HTMLResponse('<span class="text-green-400 text-sm">✓ Risk settings saved</span>')

    @app.post("/api/settings/binance")
    async def save_binance_settings(request: Request, api_key: str = Form(""), api_secret: str = Form(""), testnet: str = Form("0")):
        if err := _require_trader(request): return err
        secrets_path = Path(__file__).parent.parent / "config" / "secrets.yaml"
        data = {}
        if secrets_path.exists():
            with open(secrets_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        # Preserve existing deepseek key if present
        if "deepseek" not in data:
            data["deepseek"] = {"api_key": config.deepseek_api_key}
        data["binance"] = {"api_key": api_key, "api_secret": api_secret}
        with open(secrets_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
        config.binance_api_key = api_key
        config.binance_api_secret = api_secret
        config.binance_testnet = testnet == "1"
        # Persist testnet to config.yaml
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("binance", {})["testnet"] = (testnet == "1")
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False)
        return HTMLResponse('<span class="text-green-400 text-sm">✓ Binance settings saved</span>')

    @app.post("/api/settings/reset-sim")
    async def reset_sim_trading(request: Request):
        if err := _require_admin(request): return err
        """Clear all sim trading records and reset balance to 10000."""
        try:
            db = await get_db()
            await db.execute("DELETE FROM trades")
            await db.execute("DELETE FROM orders")
            await db.execute("DELETE FROM positions")
            await db.commit()
            await db.close()
        except Exception:
            pass

        executor = getattr(app.state, "executor", None)
        if executor:
            executor._positions.clear()

        app.state.balance = DEFAULT_BALANCE
        await _save_balance()

        resp = HTMLResponse(f'<span class="text-green-400 text-sm">✓ 模拟交易已重置 — 所有记录已清除，余额恢复至 {DEFAULT_BALANCE:.2f} USDT</span>')
        resp.headers["HX-Trigger"] = "tradeUpdated"
        return resp

    @app.post("/api/circuit-breaker/reset")
    async def reset_circuit_breaker(request: Request):
        if err := _require_trader(request): return err
        """Reset the circuit breaker trip state (for manual override)."""
        rm = getattr(app.state, "risk_manager", None)
        if not rm:
            return HTMLResponse('<span class="text-red-400">风控未就绪</span>')
        rm.breaker.reset_trip()
        rm.breaker.reset_daily()
        # Force immediate re-evaluation of all strategies
        engine = getattr(app.state, "strategy_engine", None)
        if engine:
            import asyncio
            asyncio.create_task(engine.evaluate_all_now(publish=True))
        return HTMLResponse('<span class="text-green-400 text-sm">✓ 熔断器已重置 — 交易恢复</span>')

    @app.post("/api/settings/restart")
    async def restart_server(request: Request):
        if err := _require_admin(request): return err
        """Schedule a server restart by spawning a new process and exiting."""
        import subprocess, sys, os, asyncio

        async def _do_restart():
            await asyncio.sleep(0.3)
            cwd = str(Path(__file__).parent.parent)
            subprocess.Popen([sys.executable, "-m", "app.main"], cwd=cwd,
                           creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
            os._exit(0)

        asyncio.ensure_future(_do_restart())
        return HTMLResponse('<span class="text-green-400 text-sm">✓ 服务器正在重启，请等待 5 秒后刷新页面...</span>')

    # ---- DB Manager API routes ----
    @app.get("/api/db/table/{table}")
    async def db_table_view(request: Request, table: str, search: str = "", page: int = 1, per_page: int = 50):
        try:
            user = getattr(request.state, "user", None)
            if not user or not user.is_admin:
                return HTMLResponse("Forbidden", status_code=403)
            allowed = ["trades", "alerts", "ai_suggestions", "orders", "positions", "system_config", "users"]
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
                if search and table in ("trades", "alerts", "orders"):
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
                if search and table in ("trades", "alerts", "orders"):
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

    @app.delete("/api/db/row/{table}/{row_id}")
    async def db_delete_row(table: str, row_id: int, request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"ok": False, "error": "Forbidden"}, status_code=403)
        allowed = ["trades", "alerts", "ai_suggestions", "orders", "positions", "system_config", "users"]
        if table not in allowed:
            return {"ok": False}
        db = await get_db()
        await db.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
        await db.commit()
        await db.close()
        return {"ok": True}

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
        await db.execute("DELETE FROM alerts WHERE created_at < datetime('now', '-90 days')")
        await db.execute("DELETE FROM trades WHERE status='closed' AND closed_at < datetime('now', '-365 days')")
        await db.execute("DELETE FROM ai_suggestions WHERE created_at < datetime('now', '-90 days')")
        await db.commit()
        await db.execute("VACUUM")
        await db.close()
        return {"ok": True}

    @app.get("/api/db/export/{table}")
    async def db_export_csv(table: str, request: Request, search: str = None):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        allowed = ["trades", "alerts", "ai_suggestions", "orders", "positions", "system_config", "users"]
        if table not in allowed:
            return HTMLResponse("Invalid table", status_code=400)
        db = await get_db()
        cursor = await db.execute(f"PRAGMA table_info({table})")
        cols = [dict(r) for r in await cursor.fetchall()]
        columns = [c["name"] for c in cols]
        query = f"SELECT * FROM {table}"
        params = []
        if search and table in ("trades", "alerts", "orders"):
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

    return app
