"""User management (admin only) + the user-list HTMX partial."""
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse

from web.rendering import _render
from web.deps import _require_admin


def register(app: FastAPI, ctx) -> None:
    # ---- User management (admin only) ----
    # TODO(authz): every handler below re-implements the admin check inline;
    # a future task should replace it with a FastAPI dependency.  Behaviour is
    # intentionally unchanged here.
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

    @app.get("/partials/user-list", response_class=HTMLResponse)
    async def partial_user_list(request: Request):
        if err := _require_admin(request): return err
        am = getattr(app.state, "auth_manager", None)
        users = await am.list_users() if am else []
        return _render("partials/user_list.html", {"request": request, "users": users})
