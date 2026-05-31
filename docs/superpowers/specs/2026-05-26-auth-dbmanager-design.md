# Login & Database Management System

Date: 2026-05-26

## Overview

Add multi-user authentication with role-based access control, session/JWT dual auth, and a comprehensive database management web UI for admin users.

## Part 1: Authentication & User System

### Data Model

New table `users`:

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| username | TEXT UNIQUE | |
| password_hash | TEXT | bcrypt |
| role | TEXT | admin / trader / viewer |
| display_name | TEXT | |
| created_at | TIMESTAMP | |
| last_login | TIMESTAMP | |
| enabled | INTEGER | 0/1 |

### Role Permissions

| Feature | admin | trader | viewer |
|---|---|---|---|
| View dashboard/strategies/alerts/settings | ✓ | ✓ | ✓ |
| Manual trade / close position | ✓ | ✓ | ✗ |
| Modify strategies / risk params | ✓ | ✗ | ✗ |
| User management (CRUD) | ✓ | ✗ | ✗ |
| Database manager | ✓ | ✗ | ✗ |
| AI mode switch | ✓ | ✗ | ✗ |
| Circuit breaker reset | ✓ | ✓ | ✗ |

### Auth Flow

1. Unauthenticated users → redirect to `/login`
2. POST `/api/auth/login` with username/password → verify bcrypt hash
3. Success → set HTTP-only session cookie + return JWT token in response body
4. All subsequent requests → middleware checks cookie or `Authorization: Bearer <token>` header
5. JWT payload: `{user_id, username, role, exp}`
6. Default admin account created on first startup (random password printed to console)

### Middleware

- FastAPI middleware intercepts all non-`/login` and non-`/static` routes
- Validates session cookie first, falls back to JWT header
- Injects `request.state.user` with user info
- Template context always includes `current_user`

### UI

- Login page: minimal form, dark theme matching existing design
- Navbar shows current user + role badge
- Settings → User Management panel (admin only): list, add, edit, enable/disable, delete users
- Settings → Change Password panel (all users)

## Part 2: Database Management System

### Access

- Route: `/db-manager` (admin only)
- Navbar entry visible only to admin role

### Features

**Table Browser:**
- Dropdown to select table: trades / alerts / ai_suggestions / orders / positions / system_config / users
- Search/filter: symbol, date range, PnL range
- Paginated display (50 per page)
- Row detail expand on click
- Single row delete (with confirmation)
- Status toggle (for applicable tables)

**Export:**
- Export current filtered view as CSV download

**Backup:**
- Copy the .db file to a backup path, serve as download
- Auto-named with timestamp: `binance_trader_backup_2026-05-26_120000.db`

**Restore:**
- Upload a .db file via file input
- Validate it's a valid SQLite database
- Show "Restore will replace ALL current data. Confirm?" with typed confirmation
- On confirm: close all DB connections, replace file, restart server

**Optimize:**
- Execute `VACUUM` and `REINDEX`
- Show before/after file size

**Cleanup:**
- Delete alerts older than N days (configurable, default 90)
- Delete closed trades older than N days (configurable, default 365)
- Delete old AI suggestions (configurable, default 90)
- Show count of records to be deleted before executing

### Files Changed

| File | Change |
|---|---|
| `db/database.py` | Add `users` table to schema, user CRUD helpers |
| `core/auth/__init__.py` + `core/auth/auth.py` | **New**: Password hashing, JWT, session management, middleware |
| `web/server.py` | Auth routes, middleware wiring, user state injection, db-manager routes |
| `web/templates/login.html` | **New**: Login page |
| `web/templates/db_manager.html` | **New**: Database management page |
| `web/templates/settings.html` | Add user management panel (admin) + change password panel |
| `web/templates/base.html` | Add login state display, admin nav entries |
| `web/templates/partials/` | Partial templates for db-manager table views |
| `config/config.yaml` | Add `auth` section: jwt_secret, session_timeout |
| `app/main.py` | Create default admin user on startup |
