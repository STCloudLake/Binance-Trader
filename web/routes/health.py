"""``GET /health`` — unauthenticated liveness/readiness probe."""
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from db.database import get_db


def register(app: FastAPI, ctx) -> None:
    logger = ctx.logger
    config = ctx.config

    @app.get("/health")
    async def health_check(request: Request):
        """Liveness/readiness probe.

        Unauthenticated callers get only status/database/uptime. Operational detail
        (breaker state, open positions, strategy count) is added for authenticated
        users so a supervisor can probe without credentials while an unauthenticated
        client cannot enumerate the trading state.
        """
        db_ok = True
        try:
            db = await get_db()
            await db.execute("SELECT 1")
            await db.close()
        except Exception as e:
            logger.warning(f"/health db check failed: {e}")
            db_ok = False

        payload = {
            "status": "ok" if db_ok else "degraded",
            "database": "ok" if db_ok else "error",
            "uptime_seconds": round(time.time() - ctx._app_started_at, 1),
        }

        user = getattr(request.state, "user", None)
        if user is not None:
            risk_manager = getattr(app.state, "risk_manager", None)
            executor = getattr(app.state, "executor", None)
            engine = getattr(app.state, "strategy_engine", None)
            payload.update({
                "circuit_breaker_tripped": (bool(risk_manager.breaker.is_tripped)
                                            if risk_manager else None),
                "open_positions": len(executor.get_open_positions()) if executor else None,
                "strategies_loaded": (len(getattr(engine, "_strategies", {}) or {})
                                      if engine else None),
                "mode": getattr(config, "mode", None),
            })
        return JSONResponse(payload, status_code=200 if db_ok else 503)
