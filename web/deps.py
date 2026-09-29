"""Shared request/app helpers for the web route modules.

Extracted verbatim from ``web/server.py`` (file-organization refactor only).

NOTE: The authorization helpers below stay in-handler on purpose — this
refactor must not change behaviour.  A future task should lift them into
FastAPI dependencies raising ``HTTPException(403)``.
"""
from fastapi import Request
from fastapi.responses import JSONResponse

from db.database import save_sim_balance

from web.rendering import DEFAULT_BALANCE


def _require_trader(request: Request):
    """Require trader or admin role. Returns None if OK, error response if denied."""
    # TODO(authz): convert to a FastAPI dependency raising HTTPException(403);
    # kept in-handler here so the {"error": "Forbidden"} / 403 contract is untouched.
    user = getattr(request.state, "user", None)
    if not user or not user.is_trader:
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    return None


def _require_admin(request: Request):
    """Require admin role. Returns None if OK, error response if denied."""
    # TODO(authz): convert to a FastAPI dependency raising HTTPException(403);
    # kept in-handler here so the {"error": "Forbidden"} / 403 contract is untouched.
    user = getattr(request.state, "user", None)
    if not user or not user.is_admin:
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    return None


async def _save_balance(ctx):
    """Persist the in-memory sim balance and sync it to the risk manager.

    This is the **reset** contract: the absolute value on ``app.state.balance``
    *is* the intended state (``/api/settings/reset-sim`` restores 10000), which
    is why it uses the unconditional :func:`save_sim_balance` and why the reset
    route calls it inside ``executor.reset_barrier()`` — nothing is in flight to
    be reverted.  A caller that derived a new balance from a previous read must
    use ``save_sim_balance_guarded`` (compare-and-set) instead, so a fill that
    landed in between is never silently reverted.
    """
    await save_sim_balance(getattr(ctx.app.state, "balance", DEFAULT_BALANCE),
                           ctx.config.db_path)
    # Sync to risk manager so position sizing works
    rm = getattr(ctx.app.state, "risk_manager", None)
    if rm:
        rm.update_balance(ctx.app.state.balance)
