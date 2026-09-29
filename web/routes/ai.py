"""AI endpoints: suggestions, HTMX partials, consult, model list, heartbeat."""
import json
import re
from html import escape as _html_escape
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from markupsafe import Markup

from db.database import get_db

from web.deps import _require_trader
from web.rendering import _render, _T

#: How much of a suggestion's body one card shows.  The initial server render in
#: ``ai_panel.html`` and ``/partials/ai-suggestions`` both go through
#: :func:`_suggestion_view`, so the HTMX refresh can never quietly downgrade what
#: the first paint showed (it used to drop `confidence` and cut to 100 chars).
SUGGESTION_PREVIEW_CHARS = 400


def _suggestion_summary(category: str, content: str) -> str:
    """Human-readable one-liner for a stored suggestion body.

    Bodies are ``json.dumps`` of the AI payload, which is unreadable in a card.
    Anything unparseable (or an unexpected shape) degrades to the raw text, so
    no information is ever dropped silently.
    """
    text = str(content or "").strip()
    try:
        data = json.loads(text)
    except Exception:
        return text
    if not isinstance(data, dict):
        return text

    fields = {
        "coin_selection": [("symbols", "币种"), ("coins", "币种"), ("reason", "理由")],
        "strategy_optimization": [("strategy", "策略"), ("name", "名称"),
                                  ("changes", "调整"), ("parameters", "参数"),
                                  ("reason", "理由")],
        "risk_adjustment": [("risk_appetite", "风险偏好"), ("position_size_pct", "仓位%"),
                            ("stop_loss_pct", "止损%"), ("leverage", "杠杆"),
                            ("reason", "理由")],
        "market_assessment": [("market_regime", "市场状态"), ("regime", "市场状态"),
                              ("sentiment", "情绪"), ("reason", "理由"),
                              ("summary", "摘要")],
        "news_analysis": [("sentiment", "情绪"), ("impact", "影响"), ("summary", "摘要")],
    }.get(category)
    if fields is None:
        fields = [(k, k) for k in data][:4]

    parts: list[str] = []
    for key, label in fields:
        if key not in data or data[key] in (None, "", [], {}):
            continue
        value = data[key]
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value = ", ".join(f"{k}={v}" for k, v in value.items())
        parts.append(f"{label}: {value}")
        if len(parts) == 4:
            break
    return " · ".join(parts) if parts else text


def _suggestion_view(row: dict) -> dict:
    """The single canonical projection of a suggestion for every renderer."""
    content = str(row.get("content") or "")
    summary = _suggestion_summary(row.get("category") or "", content)
    if len(summary) > SUGGESTION_PREVIEW_CHARS:
        summary = summary[:SUGGESTION_PREVIEW_CHARS] + "…"
    return {
        "id": row.get("id"),
        "category": row.get("category") or "",
        "status": row.get("status") or "pending",
        "confidence": row.get("confidence"),
        "confidence_pct": round(float(row.get("confidence") or 0) * 100),
        "rationale": row.get("rationale") or "",
        "content": content,
        "summary": summary,
        "created_at": row.get("created_at"),
        "applied_at": row.get("applied_at"),
    }


def _execute_approved_suggestion(config, category: str, content: str) -> tuple[bool, str]:
    """Apply an approved suggestion. Returns ``(applied, message_key_or_text)``.

    Approving must mean something.  ``risk_adjustment`` writes the soft risk
    parameters through the same clamped path the AI loop uses; a body that
    cannot be parsed is *not* applied and says so instead of silently claiming
    success.  ``coin_selection`` is informational by contract and is recorded as
    approved only.
    """
    if category == "risk_adjustment":
        try:
            payload = str(content).strip().removeprefix("```json").removesuffix("```").strip()
            result = json.loads(payload)
        except Exception:
            return False, "建议内容无法解析，未应用"
        if not isinstance(result, dict):
            return False, "建议内容无法解析，未应用"
        pct = min(max(float(result.get("position_size_pct", 5.0)), 0.5), 20.0)
        sl = min(max(float(result.get("stop_loss_pct", 2.0)), 0.5), 15.0)
        lev = min(max(int(result.get("leverage", 2)), 1), config.hard_limits.max_leverage)
        try:
            config.update_soft_params(
                risk_appetite=result.get("risk_appetite", "balanced"),
                position_size_pct=pct,
                stop_loss_pct=sl,
                leverage=lev)
        except Exception as e:
            return False, f"应用失败: {str(e)[:120]}"
        return True, "已批准并应用"
    if category == "strategy_optimization":
        # The strategy engine re-reads its parameters on the next tick; the row
        # carries the approval so the change is auditable.
        return True, "已批准（引擎下个周期生效）"
    return True, "已批准"


def render_suggestion_rows(rows: list[dict], status: str = "pending") -> Markup:
    """Render the shared suggestion list (already-escaped HTML, safe to inline).

    Used by ``/partials/ai-suggestions`` *and* by ``/ai`` for its first paint, so
    both show exactly the same fields (confidence included) and the periodic HTMX
    swap cannot regress to a thinner card.  Returned as :class:`Markup` because
    the page inlines it with autoescaping on.
    """
    return Markup(_render("partials/ai_suggestions.html", {
        "request": None,
        "suggestions": [_suggestion_view(r) for r in rows],
        "status": status,
    }).body.decode("utf-8"))


def register(app: FastAPI, ctx) -> None:
    config = ctx.config

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
            return [_suggestion_view(r) for r in rows]
        except Exception:
            return []

    @app.post("/api/ai-suggestions/{sid}/approve")
    async def approve_suggestion(sid: int, request: Request):
        if err := _require_trader(request): return err
        applied = False
        message = "已批准"
        try:
            db = await get_db()
            cursor = await db.execute("SELECT * FROM ai_suggestions WHERE id=?", (sid,))
            row = await cursor.fetchone()
            if row:
                s = dict(row)
                applied, message = _execute_approved_suggestion(
                    config, s.get("category", ""), s.get("content", ""))
                # `applied` records that the action really ran; `approved` is a
                # human sign-off on an informational suggestion.
                new_status = "applied" if applied else "approved"
                await db.execute(
                    "UPDATE ai_suggestions SET status=?, applied_at=CURRENT_TIMESTAMP WHERE id=?",
                    (new_status, sid))
                await db.commit()
            await db.close()
        except Exception as e:
            return HTMLResponse(
                f'<span class="text-red-400 text-xs">✗ {_html_escape(str(e)[:120])}</span>')
        colour = "text-green-400" if applied else "text-yellow-400"
        icon = "✓" if applied else "!"
        return HTMLResponse(f'<span class="{colour} text-xs">{icon} {_html_escape(message)}</span>')

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

    def _render_suggestions(rows: list[dict], status: str) -> HTMLResponse:
        return _render("partials/ai_suggestions.html", {
            "request": None,
            "suggestions": [_suggestion_view(r) for r in rows],
            "status": status,
        })

    def _render_suggestions(rows: list[dict], status: str) -> HTMLResponse:
        return HTMLResponse(render_suggestion_rows(rows, status))

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
        return _render_suggestions(rows, status)

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
