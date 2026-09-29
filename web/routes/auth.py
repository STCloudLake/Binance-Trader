"""Auth routes: login page, login/logout, change-password."""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from core.auth.auth import User

from web.rendering import _render

# Simple in-memory login rate limiter (per IP).
# Kept module-level (shared across app instances) exactly as before.
_login_attempts: dict[str, list[float]] = {}
_MAX_LOGIN_ATTEMPTS = 10    # max attempts
_LOGIN_WINDOW_SEC = 300     # per 5 minutes


def _cookie_secure(config) -> bool:
    """Should the session cookie carry ``Secure``?

    SECURITY: the flag used to be hard-coded ``False``.  It is now configurable
    and defaults to **off**, because this app is normally reached over plain
    HTTP on ``127.0.0.1`` — a ``Secure`` cookie is never sent over http, which
    would silently break login.  Precedence:

      1. ``config.session_cookie_secure`` when the Config object defines it;
      2. the ``BT_SESSION_COOKIE_SECURE`` environment variable (1/true/yes/on
         vs 0/false/no/off) for deployments that set it in the launcher;
      3. default ``False`` (documented residual: a plain-HTTP localhost session
         cookie is not transport-protected; ``httponly`` + ``samesite=lax``
         still block script access and cross-site sends).
    """
    explicit = getattr(config, "session_cookie_secure", None)
    if explicit is not None:
        return bool(explicit)
    import os
    env = os.environ.get("BT_SESSION_COOKIE_SECURE", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return False


def register(app: FastAPI, ctx) -> None:
    logger = ctx.logger
    config = ctx.config

    # ---- Auth routes ----
    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return _render("login.html", {"request": request})

    @app.post("/api/auth/login")
    async def api_login(request: Request):
        # Rate limiting by client IP
        import time as _time
        client_ip = request.client.host if request.client else "unknown"
        now = _time.time()
        attempts = _login_attempts.get(client_ip, [])
        attempts = [t for t in attempts if now - t < _LOGIN_WINDOW_SEC]
        if len(attempts) >= _MAX_LOGIN_ATTEMPTS:
            logger.warning(f"Login rate limit exceeded for IP {client_ip}")
            return JSONResponse({"error": "Too many attempts, try again later"}, status_code=429)
        _login_attempts[client_ip] = attempts

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
            _login_attempts[client_ip].append(now)
            return JSONResponse({"error": "Invalid credentials"}, status_code=401)

        user = User(id=user_data["id"], username=user_data["username"],
                     role=user_data["role"], display_name=user_data.get("display_name", ""),
                     enabled=bool(user_data.get("enabled", 1)))
        session_token = am.create_session(user)
        # Mint the JWT *with* the session id: logout (which only needs the
        # cookie) then revokes the token as well — see AuthManager.destroy_session.
        jwt_token = am.create_jwt(user, session_token=session_token)
        await am.touch_login(user.id)

        response = JSONResponse({"ok": True, "token": jwt_token, "role": user.role})
        response.set_cookie("bt_session", session_token, httponly=True, samesite="lax",
                           secure=_cookie_secure(config), max_age=am.session_hours * 3600)
        return response

    @app.post("/api/auth/logout")
    async def api_logout(request: Request):
        session_token = request.cookies.get("bt_session")
        am = getattr(app.state, "auth_manager", None)
        if am:
            if session_token:
                # Also revokes every JWT that was minted with this session.
                am.destroy_session(session_token)
            # A Bearer-only caller has no cookie: revoke the presented token too.
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                am.revoke_jwt(auth_header[7:])
        response = JSONResponse({"ok": True})
        response.delete_cookie("bt_session")
        return response

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
