import time
import secrets
import bcrypt
import jwt
from typing import Optional
from dataclasses import dataclass


@dataclass
class User:
    id: int
    username: str
    role: str
    display_name: str
    enabled: bool

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_trader(self) -> bool:
        return self.role in ("admin", "trader")


class AuthManager:
    def __init__(self, db_path: str, jwt_secret: str, session_hours: int = 24):
        self.db_path = db_path
        self.jwt_secret = jwt_secret
        self.session_hours = session_hours
        self._sessions: dict[str, dict] = {}
        #: jti -> expiry of revoked JWTs (in-memory, like the sessions store).
        self._revoked_jti: dict[str, float] = {}
        #: session token -> {jti: exp} for the JWTs minted with that session, so
        #: destroying the session also kills every token issued for it.
        self._session_jti: dict[str, dict[str, float]] = {}

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode(), hashed.encode())

    def create_jwt(self, user: User, session_token: Optional[str] = None) -> str:
        """Mint a signed token.

        Every token carries a ``jti`` so it can be revoked server-side (logout),
        and is tied to ``session_token`` when one is supplied: destroying that
        session (``destroy_session``) revokes the token too.  Tokens minted
        without a session stay usable until they expire — that keeps the
        standalone-Bearer flow working for scripts into a server restart.
        """
        now = time.time()
        jti = secrets.token_hex(16)
        payload = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role,
            "exp": int(now) + self.session_hours * 3600,
            "jti": jti,
        }
        if session_token:
            payload["sid"] = session_token
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        if session_token:
            self._session_jti.setdefault(session_token, {})[jti] = float(payload["exp"])
        return token

    def _prune_revoked(self, now: float = None) -> None:
        now = time.time() if now is None else now
        for jti in [j for j, exp in self._revoked_jti.items() if exp <= now]:
            self._revoked_jti.pop(jti, None)

    def verify_jwt(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except jwt.PyJWTError:
            return None
        jti = payload.get("jti")
        if jti:
            self._prune_revoked()
            if jti in self._revoked_jti:
                return None
        return payload

    def revoke_jwt(self, token: str) -> bool:
        """Revoke one token (by ``jti``) until its own expiry. Returns True if
        the token was a valid, unexpired JWT we could recognise."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except jwt.PyJWTError:
            return False
        jti = payload.get("jti")
        if not jti:
            return False
        self._revoked_jti[jti] = float(payload.get("exp") or (time.time() + 3600))
        return True

    def create_session(self, user: User) -> str:
        token = secrets.token_hex(32)
        self._sessions[token] = {
            "user_id": user.id,
            "role": user.role,
            "expires": time.time() + self.session_hours * 3600,
        }
        return token

    def verify_session(self, token: str) -> Optional[dict]:
        session = self._sessions.get(token)
        if not session:
            return None
        if time.time() > session["expires"]:
            self.destroy_session(token)
            return None
        return session

    def destroy_session(self, token: str):
        self._sessions.pop(token, None)
        # Revoke every JWT that was minted together with this session, so a
        # stolen copy of the token dies with the logout even though the logout
        # request itself only carried the session cookie.
        for jti, exp in (self._session_jti.pop(token, None) or {}).items():
            self._revoked_jti[jti] = exp

    async def _connect(self):
        """Open and configure a DB connection."""
        import aiosqlite
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        return db

    async def get_user_by_username(self, username: str) -> Optional[dict]:
        db = await self._connect()
        try:
            cursor = await db.execute("SELECT * FROM users WHERE username=? AND enabled=1", (username,))
            row = await cursor.fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    async def get_user_by_id(self, user_id: int, include_disabled: bool = False) -> Optional[dict]:
        db = await self._connect()
        try:
            if include_disabled:
                cursor = await db.execute("SELECT * FROM users WHERE id=?", (user_id,))
            else:
                cursor = await db.execute("SELECT * FROM users WHERE id=? AND enabled=1", (user_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    async def list_users(self) -> list[dict]:
        db = await self._connect()
        try:
            cursor = await db.execute("SELECT id, username, role, display_name, enabled, created_at, last_login FROM users ORDER BY id")
            rows = [dict(r) for r in await cursor.fetchall()]
            return rows
        finally:
            await db.close()

    async def create_user(self, username: str, password: str, role: str = "viewer", display_name: str = "") -> int:
        db = await self._connect()
        try:
            cursor = await db.execute(
                "INSERT INTO users (username, password_hash, role, display_name) VALUES (?,?,?,?)",
                (username, self.hash_password(password), role, display_name or username))
            await db.commit()
            return cursor.lastrowid
        finally:
            await db.close()

    async def update_user(self, user_id: int, **kwargs):
        allowed = {"username", "role", "display_name", "enabled"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        if "password" in kwargs:
            updates["password_hash"] = self.hash_password(kwargs["password"])
        db = await self._connect()
        try:
            set_clause = ", ".join(f"{k}=?" for k in updates)
            await db.execute(f"UPDATE users SET {set_clause} WHERE id=?", (*updates.values(), user_id))
            await db.commit()
        finally:
            await db.close()

    async def change_password(self, user_id: int, new_password: str):
        db = await self._connect()
        try:
            await db.execute("UPDATE users SET password_hash=? WHERE id=?", (self.hash_password(new_password), user_id))
            await db.commit()
        finally:
            await db.close()

    async def touch_login(self, user_id: int):
        db = await self._connect()
        try:
            await db.execute("UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE id=?", (user_id,))
            await db.commit()
        finally:
            await db.close()

    async def count_users(self) -> int:
        db = await self._connect()
        try:
            cursor = await db.execute("SELECT COUNT(*) FROM users")
            row = await cursor.fetchone()
            return row[0] if row else 0
        finally:
            await db.close()

    def create_middleware(self):
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import RedirectResponse, JSONResponse as _JSONResponse

        auth = self

        class AuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                path = request.url.path
                if path in ("/login", "/api/auth/login", "/api/auth/logout", "/health") or path.startswith("/static") or path.startswith("/ws/"):
                    return await call_next(request)

                user = None
                session_token = request.cookies.get("bt_session")
                if session_token:
                    session = auth.verify_session(session_token)
                    if session:
                        user_data = await auth.get_user_by_id(session["user_id"])
                        if user_data:
                            user = User(id=user_data["id"], username=user_data["username"],
                                        role=user_data["role"], display_name=user_data.get("display_name", ""),
                                        enabled=bool(user_data.get("enabled", 1)))

                if not user:
                    auth_header = request.headers.get("Authorization", "")
                    if auth_header.startswith("Bearer "):
                        payload = auth.verify_jwt(auth_header[7:])
                        if payload:
                            user_data = await auth.get_user_by_id(payload["user_id"])
                            if user_data:
                                user = User(id=user_data["id"], username=user_data["username"],
                                            role=user_data["role"], display_name=user_data.get("display_name", ""),
                                            enabled=bool(user_data.get("enabled", 1)))

                if not user:
                    if path.startswith("/api/") or path.startswith("/partials/"):
                        return _JSONResponse({"error": "Unauthorized"}, status_code=401)
                    return RedirectResponse(url="/login", status_code=302)

                request.state.user = user
                return await call_next(request)

        return AuthMiddleware

    @staticmethod
    def generate_random_password(length: int = 12) -> str:
        return secrets.token_urlsafe(length)
