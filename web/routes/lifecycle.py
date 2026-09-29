"""Strategy lifecycle routes (events, generate, optimize, HTMX partial)."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse

from db.database import get_db

from web.deps import _require_trader
from web.rendering import _render


def register(app: FastAPI, ctx) -> None:
    # ---- Strategy Lifecycle routes ----
    @app.get("/api/strategy-lifecycle/events")
    async def lifecycle_events(limit: int = 50):
        db = await get_db()
        cursor = await db.execute(
            "SELECT * FROM strategy_lifecycle_events ORDER BY created_at DESC LIMIT ?",
            (limit,))
        events = [dict(r) for r in await cursor.fetchall()]
        await db.close()
        return events

    @app.post("/api/strategy-lifecycle/generate")
    async def lifecycle_generate(request: Request):
        if err := _require_trader(request): return err
        mgr = getattr(app.state, "lifecycle_manager", None)
        if not mgr:
            return JSONResponse({"error": "Lifecycle manager not initialized"}, status_code=500)
        config = await mgr.generate_strategy()
        if not config:
            return JSONResponse({"ok": False, "error": "AI generation failed"})

        # Save to strategy YAML and reload engine
        from core.strategy.loader import StrategyConfig
        try:
            strategy_config = StrategyConfig(**config)
            loader = getattr(app.state, "strategy_loader", None)
            engine = getattr(app.state, "strategy_engine", None)
            if loader:
                loader.save(strategy_config)
            if engine and loader:
                all_s = loader.load_all()
                engine._strategies = {s.name: s for s in all_s}
                engine._purge_stale_cache()
                await engine.evaluate_all_now()
            # Log lifecycle event
            if mgr:
                await mgr.log_event(strategy_config.name, "generated",
                                     "Manually generated via Web UI")
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Failed to save strategy: {e}"})

        return JSONResponse({"ok": True, "strategy": config, "saved": True})

    @app.post("/api/strategy-lifecycle/optimize")
    async def lifecycle_optimize(request: Request):
        """Manually trigger matrix-based strategy analysis and optimization."""
        if err := _require_trader(request): return err
        mgr = getattr(app.state, "lifecycle_manager", None)
        htmx = request.headers.get("HX-Request") == "true"
        if not mgr:
            if htmx:
                return HTMLResponse('<span class="text-yellow-400">Lifecycle manager not initialized</span>')
            return JSONResponse({"error": "Lifecycle manager not initialized"}, status_code=500)
        try:
            result = await mgr.analyze_and_optimize()
        except Exception as e:
            if htmx:
                return HTMLResponse(f'<span class="text-red-400">- {str(e)[:160]}</span>')
            return JSONResponse({"ok": False, "error": str(e)})
        if htmx:
            # The `/ai` panel swaps this straight into `#lifecycle-result`, where
            # raw JSON would read as noise.
            actions = result if isinstance(result, list) else (
                result.get("actions", result) if isinstance(result, dict) else result)
            count = len(actions) if isinstance(actions, (list, tuple, dict)) else 0
            return HTMLResponse(
                f'<span class="text-green-400">✓ 优化完成，{count} 项调整</span>')
        return JSONResponse({"ok": True, "actions": result})

    @app.get("/partials/strategy-lifecycle")
    async def partial_strategy_lifecycle(request: Request):
        try:
            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM strategy_lifecycle_events ORDER BY created_at DESC LIMIT 50")
            events = [dict(r) for r in await cursor.fetchall()]
            await db.close()
        except Exception:
            events = []
        return _render("partials/strategy_lifecycle.html",
                       {"request": request, "events": events})
