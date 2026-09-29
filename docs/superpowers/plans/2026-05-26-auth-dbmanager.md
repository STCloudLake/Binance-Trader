# Auth & Database Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-user authentication (admin/trader/viewer roles, session+JWT dual auth) and a comprehensive database management web UI (browse, search, export, backup, restore, optimize, cleanup).

**Architecture:** New `core/auth/` module handles password hashing, JWT, session tokens, and FastAPI middleware. Auth middleware wraps all routes except `/login` and static assets, injecting `request.state.user`. Admin-only routes for `/db-manager` and user management.

**Tech Stack:** bcrypt, PyJWT, FastAPI middleware, aiosqlite

---

## File Map

| File | Role |
|---|---|
| `core/auth/__init__.py` | **New**: Re-exports |
| `core/auth/auth.py` | **New**: AuthManager class — password hashing, JWT, session, middleware factory |
| `db/database.py` | Add `users` table, CRUD helpers |
| `app/main.py` | Create default admin on startup |
| `config/config.yaml` | Add `auth` section (jwt_secret, session_hours) |
| `web/server.py` | Auth routes, middleware wiring, user mgmt routes, db-manager routes |
| `web/templates/login.html` | **New**: Login page |
| `web/templates/db_manager.html` | **New**: DB management page |
| `web/templates/partials/db_table.html` | **New**: Table viewer partial |
| `web/templates/settings.html` | Add user management + change password panels |
| `web/templates/base.html` | Add login state, admin nav entries |

---

### Task 1: Auth Core Module

**Files:**
- Create: `binance_trader/core/auth/__init__.py`
- Create: `binance_trader/core/auth/auth.py`

- [ ] **Step 1: Create `core/auth/__init__.py`**

```python
from core.auth.auth import AuthManager
```

- [ ] **Step 2: Create `core/auth/auth.py`**

```python
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


ROLES = ["admin", "trader", "viewer"]
READONLY_ROLES = ["viewer"]


class AuthManager:
    def __init__(self, db_path: str, jwt_secret: str, session_hours: int = 24):
        self.db_path = db_path
        self.jwt_secret = jwt_secret
        self.session_hours = session_hours
        self._sessions: dict[str, dict] = {}  # token -> {user_id, role, expires}

    # ---- Password ----
    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode(), hashed.encode())

    # ---- JWT ----
    def create_jwt(self, user: User) -> str:
        payload = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role,
            "exp": int(time.time()) + self.session_hours * 3600,
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def verify_jwt(self, token: str) -> Optional[dict]:
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except jwt.PyJWTError:
            return None

    # ---- Session ----
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
            del self._sessions[token]
            return None
        return session

    def destroy_session(self, token: str):
        self._sessions.pop(token, None)

    # ---- Database helpers ----
    async def get_user_by_username(self, username: str) -> Optional[dict]:
        import aiosqlite
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE username=? AND enabled=1", (username,))
        row = await cursor.fetchone()
        await db.close()
        return dict(row) if row else None

    async def get_user_by_id(self, user_id: int) -> Optional[dict]:
        import aiosqlite
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE id=? AND enabled=1", (user_id,))
        row = await cursor.fetchone()
        await db.close()
        return dict(row) if row else None

    async def list_users(self) -> list[dict]:
        import aiosqlite
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT id, username, role, display_name, enabled, created_at, last_login FROM users ORDER BY id")
        rows = [dict(r) for r in await cursor.fetchall()]
        await db.close()
        return rows

    async def create_user(self, username: str, password: str, role: str = "viewer", display_name: str = "") -> int:
        import aiosqlite
        db = await aiosqlite.connect(self.db_path)
        cursor = await db.execute(
            "INSERT INTO users (username, password_hash, role, display_name) VALUES (?,?,?,?)",
            (username, self.hash_password(password), role, display_name or username))
        await db.commit()
        user_id = cursor.lastrowid
        await db.close()
        return user_id

    async def update_user(self, user_id: int, **kwargs):
        import aiosqlite
        allowed = {"username", "role", "display_name", "enabled"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        if "password" in kwargs:
            updates["password_hash"] = self.hash_password(kwargs["password"])
        db = await aiosqlite.connect(self.db_path)
        set_clause = ", ".join(f"{k}=?" for k in updates)
        await db.execute(f"UPDATE users SET {set_clause} WHERE id=?", (*updates.values(), user_id))
        await db.commit()
        await db.close()

    async def change_password(self, user_id: int, new_password: str):
        import aiosqlite
        db = await aiosqlite.connect(self.db_path)
        await db.execute("UPDATE users SET password_hash=? WHERE id=?", (self.hash_password(new_password), user_id))
        await db.commit()
        await db.close()

    async def touch_login(self, user_id: int):
        import aiosqlite
        db = await aiosqlite.connect(self.db_path)
        await db.execute("UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE id=?", (user_id,))
        await db.commit()
        await db.close()

    async def count_users(self) -> int:
        import aiosqlite
        db = await aiosqlite.connect(self.db_path)
        cursor = await db.execute("SELECT COUNT(*) FROM users")
        row = await cursor.fetchone()
        await db.close()
        return row[0] if row else 0

    # ---- Middleware factory ----
    def create_middleware(self):
        """Return a FastAPI-compatible middleware function."""
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import RedirectResponse
        import json

        auth = self  # closure reference

        class AuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Skip login page and static
                path = request.url.path
                if path in ("/login", "/api/auth/login", "/api/auth/logout") or path.startswith("/static"):
                    return await call_next(request)

                user = None
                # 1. Check session cookie
                session_token = request.cookies.get("bt_session")
                if session_token:
                    session = auth.verify_session(session_token)
                    if session:
                        user_data = await auth.get_user_by_id(session["user_id"])
                        if user_data:
                            user = User(id=user_data["id"], username=user_data["username"],
                                        role=user_data["role"], display_name=user_data.get("display_name", ""),
                                        enabled=bool(user_data.get("enabled", 1)))

                # 2. Fallback to JWT header
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
                    # API routes: return 401 JSON. Page routes: redirect to login.
                    if path.startswith("/api/") or path.startswith("/partials/"):
                        return JSONResponse({"error": "Unauthorized"}, status_code=401)
                    return RedirectResponse(url="/login", status_code=302)

                request.state.user = user
                return await call_next(request)

        return AuthMiddleware

    @staticmethod
    def generate_random_password(length: int = 12) -> str:
        return secrets.token_urlsafe(length)
```

- [ ] **Step 3: Verify auth module imports**

Run: `python -c "from core.auth.auth import AuthManager; print('OK')"` from `binance_trader/`

---

### Task 2: Database — Users Table

**Files:**
- Modify: `binance_trader/db/database.py`

- [ ] **Step 1: Add users table to SCHEMA**

Add after the `system_config` table definition:

```sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'viewer' CHECK(role IN ('admin','trader','viewer')),
    display_name TEXT DEFAULT '',
    enabled INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);
```

- [ ] **Step 2: Verify schema loads**

Run: `python -c "from db.database import init_database; import asyncio; asyncio.run(init_database('data/test_auth.db')); print('OK')"` from `binance_trader/`

---

### Task 3: Config + Default Admin

**Files:**
- Modify: `binance_trader/config/config.yaml`
- Modify: `binance_trader/app/main.py`

- [ ] **Step 1: Add auth section to config.yaml**

```yaml
auth:
  jwt_secret: ""  # auto-generated on first start if empty
  session_hours: 24
```

- [ ] **Step 2: Add default admin creation to main.py**

After `await init_database(config.db_path)` (around line 48), add:

```python
    # Initialize auth and create default admin if no users exist
    from core.auth.auth import AuthManager
    jwt_secret = config._get("auth", {}).get("jwt_secret", "") if isinstance(config._get("auth", {}), dict) else ""
    if not jwt_secret:
        import secrets
        jwt_secret = secrets.token_hex(32)
        logger.info(f"Generated JWT secret: {jwt_secret}")
    auth_manager = AuthManager(config.db_path, jwt_secret,
                                config._get("auth", {}).get("session_hours", 24) if isinstance(config._get("auth", {}), dict) else 24)
    if await auth_manager.count_users() == 0:
        admin_pass = AuthManager.generate_random_password()
        await auth_manager.create_user("admin", admin_pass, "admin", "Administrator")
        logger.warning(f"=== DEFAULT ADMIN CREATED: username=admin, password={admin_pass} ===")
```

- [ ] **Step 3: Wire auth_manager to app state**

After `web_app = create_app(config, event_bus)`, add:
```python
    web_app.state.auth_manager = auth_manager
```

And add middleware before `server.serve()`:
```python
    app.add_middleware(auth_manager.create_middleware())
```

Wait — the middleware is added in `create_app()`, not main.py. Let me add it there. Actually, `auth_manager` is created in main.py. I'll pass it to `create_app()` or set it on the app after creation.

Better approach: create `auth_manager` in main.py, pass to `create_app`:

```python
    web_app = create_app(config, event_bus, auth_manager)
```

And in `create_app`, add middleware and routes.

- [ ] **Step 4: Update create_app signature**

Change `def create_app(config: Config, event_bus: EventBus) -> FastAPI:` to:
```python
def create_app(config: Config, event_bus: EventBus, auth_manager=None) -> FastAPI:
```

And inside `create_app`, after `app = FastAPI(...)`, add:
```python
    if auth_manager:
        app.add_middleware(auth_manager.create_middleware())
        app.state.auth_manager = auth_manager
```

---

### Task 4: Auth Routes — Login, Logout, User Management

**Files:**
- Modify: `binance_trader/web/server.py`

- [ ] **Step 1: Add login page and auth API routes**

```python
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
        response.set_cookie("bt_session", session_token, httponly=True, max_age=am.session_hours * 3600)
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
    async def create_user(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        body = await request.json()
        am = getattr(app.state, "auth_manager", None)
        uid = await am.create_user(body["username"], body["password"],
                                    body.get("role", "viewer"), body.get("display_name", ""))
        return {"ok": True, "id": uid}

    @app.post("/api/users/{uid}")
    async def update_user(uid: int, request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        body = await request.json()
        am = getattr(app.state, "auth_manager", None)
        await am.update_user(uid, **body)
        return {"ok": True}

    @app.delete("/api/users/{uid}")
    async def delete_user(uid: int, request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        am = getattr(app.state, "auth_manager", None)
        await am.update_user(uid, enabled=0)  # soft delete
        return {"ok": True}

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
```

- [ ] **Step 2: Add User import at top of server.py**

```python
from core.auth.auth import AuthManager, User
```

---

### Task 5: Login Page Template

**Files:**
- Create: `binance_trader/web/templates/login.html`

- [ ] **Step 1: Create login.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Binance Trader - Login</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: { extend: { colors: { surface: '#0f172a', 'surface-light': '#1e293b', 'border': '#334155', } } }
        }
    </script>
    <style>
        body { background: #0f172a; color: #e2e8f0; font-family: 'Microsoft YaHei', 'Consolas', monospace; }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center">
    <div class="w-full max-w-sm">
        <h1 class="text-2xl font-bold text-sky-400 text-center mb-8">📊 Binance Trader</h1>
        <div class="card">
            <form id="login-form" onsubmit="login(event)" class="space-y-4">
                <div>
                    <label class="text-sm text-slate-400">Username</label>
                    <input type="text" id="username" class="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white" autofocus>
                </div>
                <div>
                    <label class="text-sm text-slate-400">Password</label>
                    <input type="password" id="password" class="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white">
                </div>
                <div id="error" class="text-red-400 text-sm hidden"></div>
                <button type="submit" class="w-full bg-sky-600 hover:bg-sky-500 text-white px-4 py-2 rounded text-sm">Login</button>
            </form>
        </div>
    </div>
    <script>
    async function login(e) {
        e.preventDefault();
        var err = document.getElementById('error');
        err.classList.add('hidden');
        var resp = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                username: document.getElementById('username').value,
                password: document.getElementById('password').value
            })
        });
        var data = await resp.json();
        if (data.ok) {
            window.location.href = '/dashboard';
        } else {
            err.textContent = data.error || 'Login failed';
            err.classList.remove('hidden');
        }
    }
    </script>
</body>
</html>
```

---

### Task 6: Navbar & Settings Updates

**Files:**
- Modify: `binance_trader/web/templates/base.html`
- Modify: `binance_trader/web/templates/settings.html`

- [ ] **Step 1: Update base.html navbar**

After the existing nav links, add before `</nav>`:

```html
        <span class="ml-auto flex items-center gap-3">
            {% if current_user %}
                {% if current_user.is_admin %}
                <a href="/db-manager" class="nav-link text-xs {{ 'active' if current_page == 'db_manager' else '' }}">DB</a>
                {% endif %}
                <span class="text-xs text-slate-500">{{ current_user.display_name or current_user.username }}
                    <span class="badge {{ 'badge-yellow' if current_user.role == 'admin' else 'badge-blue' if current_user.role == 'trader' else 'badge-green' }}">{{ current_user.role }}</span>
                </span>
                <button class="text-xs text-red-400 hover:underline" onclick="logout()">Logout</button>
            {% endif %}
        </span>
    </nav>
    <script>
    async function logout() {
        await fetch('/api/auth/logout', {method:'POST'});
        window.location.href = '/login';
    }
    </script>
```

Add `current_user=None` to every `_render()` context, or better — inject it via the auth middleware. The middleware already sets `request.state.user`, so in server.py, add a helper:

In `_render()`, before rendering:
```python
    if "request" in context and hasattr(context["request"], "state"):
        context["current_user"] = getattr(context["request"].state, "user", None)
```

- [ ] **Step 2: Add user management panel to settings.html**

After the restart service card, add (admin only):

```html
{% if current_user and current_user.is_admin %}
<div class="card mt-6">
    <h3 class="text-lg mb-3">User Management</h3>
    <div id="user-list" hx-get="/partials/user-list" hx-trigger="load">
        <div class="text-slate-500 text-sm">Loading...</div>
    </div>
    <form class="mt-3 grid grid-cols-4 gap-2" hx-post="/api/users" hx-target="#user-list" hx-swap="innerHTML">
        <input type="text" name="username" placeholder="Username" required class="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-white text-xs">
        <input type="password" name="password" placeholder="Password" required class="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-white text-xs">
        <select name="role" class="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-white text-xs">
            <option value="viewer">Viewer</option><option value="trader">Trader</option><option value="admin">Admin</option>
        </select>
        <button type="submit" class="bg-sky-600 hover:bg-sky-500 text-white px-2 py-1 rounded text-xs">Add User</button>
    </form>
</div>
{% endif %}

<div class="card mt-6">
    <h3 class="text-lg mb-3">Change Password</h3>
    <form onsubmit="changePassword(event)" class="space-y-2">
        <input type="password" id="cp-current" placeholder="Current Password" class="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white text-sm">
        <input type="password" id="cp-new" placeholder="New Password" class="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white text-sm">
        <button type="submit" class="bg-sky-600 hover:bg-sky-500 text-white px-4 py-2 rounded text-sm">Update Password</button>
        <span id="cp-msg"></span>
    </form>
</div>
<script>
async function changePassword(e) {
    e.preventDefault();
    var resp = await fetch('/api/auth/change-password', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
            current_password: document.getElementById('cp-current').value,
            new_password: document.getElementById('cp-new').value
        })
    });
    var data = await resp.json();
    document.getElementById('cp-msg').innerHTML = data.ok
        ? '<span class="text-green-400 text-xs">Password updated</span>'
        : '<span class="text-red-400 text-xs">' + (data.error || 'Failed') + '</span>';
}
</script>
```

Add partial for user list: `web/templates/partials/user_list.html`

```html
{% for u in users %}
<div class="flex items-center justify-between py-1 border-b border-slate-800 text-sm">
    <span>{{ u.username }} <span class="badge {{ 'badge-yellow' if u.role=='admin' else 'badge-blue' if u.role=='trader' else 'badge-green' }} text-xs">{{ u.role }}</span></span>
    <span class="text-xs text-slate-500">{{ u.last_login or '-' }}</span>
    <button class="text-red-400 text-xs" hx-delete="/api/users/{{ u.id }}" hx-target="#user-list">Disable</button>
</div>
{% endfor %}
```

Add server.py route for user list partial:
```python
    @app.get("/partials/user-list", response_class=HTMLResponse)
    async def partial_user_list(request: Request):
        am = getattr(app.state, "auth_manager", None)
        users = await am.list_users() if am else []
        return _render("partials/user_list.html", {"request": request, "users": users})
```

---

### Task 7: Database Manager Page

**Files:**
- Create: `binance_trader/web/templates/db_manager.html`
- Create: `binance_trader/web/templates/partials/db_table.html`

- [ ] **Step 1: Create db_manager.html**

```html
{% extends "base.html" %}
{% block content %}
<h2 class="text-xl mb-4">Database Manager</h2>

<!-- Action bar -->
<div class="flex gap-2 mb-4 flex-wrap">
    <button class="px-3 py-1 rounded text-sm bg-sky-700 text-white" onclick="dbAction('backup')">Download Backup</button>
    <form id="restore-form" class="hidden"><input type="file" id="restore-file" accept=".db" onchange="dbAction('restore')"></form>
    <button class="px-3 py-1 rounded text-sm bg-slate-700 text-slate-300" onclick="document.getElementById('restore-file').click()">Restore Backup</button>
    <button class="px-3 py-1 rounded text-sm bg-yellow-700 text-white" onclick="dbAction('optimize')">Optimize (VACUUM)</button>
    <button class="px-3 py-1 rounded text-sm bg-red-700 text-white" onclick="if(confirm('Delete old data?'))dbAction('cleanup')">Cleanup Old Data</button>
    <span id="db-action-msg" class="text-xs"></span>
</div>

<!-- Table browser -->
<div class="flex gap-2 mb-4 items-center">
    <select id="db-table" onchange="loadTable()" class="bg-slate-800 border border-slate-600 rounded px-3 py-1 text-white text-sm">
        {% for t in tables %}
        <option value="{{ t }}">{{ t }}</option>
        {% endfor %}
    </select>
    <input type="text" id="db-search" placeholder="Search symbol..." oninput="loadTable()" class="bg-slate-800 border border-slate-600 rounded px-3 py-1 text-white text-sm w-40">
    <button class="px-3 py-1 rounded text-xs bg-green-700 text-white" onclick="exportCSV()">Export CSV</button>
</div>

<div id="db-table-content" class="card overflow-x-auto max-h-[60vh]">
    Select a table to browse
</div>

<script>
async function loadTable() {
    var table = document.getElementById('db-table').value;
    var search = document.getElementById('db-search').value;
    var resp = await fetch('/api/db/table/' + table + '?search=' + encodeURIComponent(search));
    document.getElementById('db-table-content').innerHTML = await resp.text();
}
async function dbAction(action) {
    var msg = document.getElementById('db-action-msg');
    if (action === 'restore') {
        var file = document.getElementById('restore-file').files[0];
        if (!file) return;
        var form = new FormData(); form.append('file', file);
        var resp = await fetch('/api/db/restore', {method:'POST', body: form});
        var data = await resp.json();
        msg.innerHTML = data.ok ? '<span class="text-green-400">Restored. Restarting...</span>' : '<span class="text-red-400">'+data.error+'</span>';
        if (data.ok) setTimeout(function(){location.reload();}, 3000);
    } else if (action === 'backup') {
        window.location.href = '/api/db/backup';
    } else {
        var resp = await fetch('/api/db/' + action, {method:'POST'});
        var data = await resp.json();
        msg.innerHTML = data.ok ? '<span class="text-green-400">OK</span>' : '<span class="text-red-400">'+data.error+'</span>';
    }
}
function exportCSV() {
    var table = document.getElementById('db-table').value;
    var search = document.getElementById('db-search').value;
    window.location.href = '/api/db/export/' + table + '?search=' + encodeURIComponent(search);
}
</script>
{% endblock %}
```

- [ ] **Step 2: Create partials/db_table.html**

```html
<table class="w-full text-sm">
<thead class="text-slate-400 border-b border-slate-700"><tr>
{% for col in columns %}<th class="text-left p-1">{{ col }}</th>{% endfor %}
<th></th>
</tr></thead><tbody>
{% for row in rows %}
<tr class="border-b border-slate-800 hover:bg-slate-800/50">
    {% for col in columns %}
    <td class="p-1 text-xs max-w-40 truncate">{{ row[col] if row[col] is not none else '-' }}</td>
    {% endfor %}
    <td><button class="text-red-400 text-xs" onclick="deleteRow('{{ table }}', {{ row.id }})">Del</button></td>
</tr>
{% endfor %}
</tbody></table>
{% if not rows %}<div class="p-4 text-slate-500 text-sm">No data</div>{% endif %}
```

---

### Task 8: DB Manager API Routes

**Files:**
- Modify: `binance_trader/web/server.py`

- [ ] **Step 1: Add db-manager page and API routes**

```python
    # ---- Database Manager (admin only) ----
    @app.get("/db-manager", response_class=HTMLResponse)
    async def db_manager_page(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_admin:
            return RedirectResponse(url="/dashboard", status_code=302)
        tables = ["trades", "alerts", "ai_suggestions", "orders", "positions", "system_config", "users"]
        return _render("db_manager.html", {"request": request, "current_page": "db_manager", "tables": tables})

    @app.get("/api/db/table/{table}")
    async def db_table_view(table: str, search: str = None):
        allowed = ["trades", "alerts", "ai_suggestions", "orders", "positions", "system_config", "users"]
        if table not in allowed:
            return HTMLResponse("Invalid table")
        db = await get_db()
        # Get columns
        cursor = await db.execute(f"PRAGMA table_info({table})")
        cols = [dict(r) for r in await cursor.fetchall()]
        columns = [c["name"] for c in cols]
        # Build query
        query = f"SELECT * FROM {table}"
        params = []
        if search and table in ("trades", "alerts", "orders"):
            query += " WHERE symbol LIKE ? OR message LIKE ?"
            params = [f"%{search}%", f"%{search}%"]
        query += " ORDER BY id DESC LIMIT 100"
        cursor = await db.execute(query, params)
        rows = [dict(r) for r in await cursor.fetchall()]
        await db.close()
        return _render("partials/db_table.html", {"request": None, "table": table, "columns": columns, "rows": rows})

    @app.delete("/api/db/row/{table}/{row_id}")
    async def db_delete_row(table: str, row_id: int):
        allowed = ["trades", "alerts", "ai_suggestions", "orders", "positions", "system_config", "users"]
        if table not in allowed:
            return {"ok": False}
        db = await get_db()
        await db.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
        await db.commit()
        await db.close()
        return {"ok": True}

    @app.get("/api/db/backup")
    async def db_backup():
        from fastapi.responses import FileResponse
        import shutil, time as _time
        src = config.db_path
        ts = _time.strftime("%Y%m%d_%H%M%S", _time.localtime())
        dst = src.replace(".db", f"_backup_{ts}.db")
        shutil.copy2(src, dst)
        return FileResponse(dst, filename=f"binance_trader_backup_{ts}.db")

    @app.post("/api/db/restore")
    async def db_restore(request: Request):
        from fastapi import UploadFile, File
        form = await request.form()
        file = form.get("file")
        if not file:
            return {"ok": False, "error": "No file uploaded"}
        import shutil, os
        contents = await file.read()
        # Validate SQLite header
        if contents[:16] != b"SQLite format 3\x00":
            return {"ok": False, "error": "Not a valid SQLite database"}
        # Write backup then replace
        src = config.db_path
        backup_path = src + ".pre_restore"
        shutil.copy2(src, backup_path)
        with open(src, "wb") as f:
            f.write(contents)
        return {"ok": True}

    @app.post("/api/db/optimize")
    async def db_optimize():
        import os
        before = os.path.getsize(config.db_path)
        db = await get_db()
        await db.execute("VACUUM")
        await db.execute("REINDEX")
        await db.close()
        after = os.path.getsize(config.db_path)
        return {"ok": True, "before": before, "after": after}

    @app.post("/api/db/cleanup")
    async def db_cleanup():
        db = await get_db()
        await db.execute("DELETE FROM alerts WHERE created_at < datetime('now', '-90 days')")
        await db.execute("DELETE FROM trades WHERE status='closed' AND closed_at < datetime('now', '-365 days')")
        await db.execute("DELETE FROM ai_suggestions WHERE created_at < datetime('now', '-90 days')")
        await db.commit()
        await db.execute("VACUUM")
        await db.close()
        return {"ok": True}

    @app.get("/api/db/export/{table}")
    async def db_export_csv(table: str, search: str = None):
        from fastapi.responses import StreamingResponse
        import csv, io
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
```

---

### Task 9: Integration & Verification

**Files:** None (verification only)

- [ ] **Step 1: Add missing imports to server.py**

Ensure these are at the top:
```python
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
import csv, io, os, shutil, secrets
```

- [ ] **Step 2: Inject current_user into all templates**

In `_render()`, add:
```python
    if context.get("request") and hasattr(context["request"], "state"):
        context["current_user"] = getattr(context["request"].state, "user", None)
```

- [ ] **Step 3: Start app, verify login flow**

```bash
cd binance_trader
python -m app.main --mode sim --port 8899
```

Check:
1. Console prints: `=== DEFAULT ADMIN CREATED: username=admin, password=<random> ===`
2. Open `http://127.0.0.1:8899` → redirects to `/login`
3. Login with admin / random password → redirects to dashboard
4. Navbar shows username + role badge + DB link
5. `/settings` shows User Management panel (admin only)
6. `/db-manager` shows table browser, backup/restore/optimize buttons
7. Logout → redirects to login
8. Login as trader/viewer → DB Manager hidden, trade restrictions apply
