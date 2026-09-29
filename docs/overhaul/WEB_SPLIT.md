# Web layer split — `web/server.py` → `web/routes/*` + `web/ws/*`

> ## ⚠️ Historical snapshot (pre-overhaul)
>
> **This document records the state at the moment of the split and is not kept
> up to date.** Every line count and route count below is as-of-then:
> `web/server.py` was 90 lines and the app exposed **97** routes.
>
> Current facts: `web/server.py` is **115** lines and the app exposes
> **118** routes. The authoritative current route baseline is
> [`docs/overhaul/route-baseline.json`](route-baseline.json) (118 entries) and
> [`README.md` §7.2](../../README.md#72-路由总数).
>
> Read the rest of this file as the split's design record, not as a description
> of today's code.

**Scope:** file organization only. No route, status code, authorization rule,
template, `app.state.*` attribute or side-effect order was changed.

`web/server.py` was a 2,366-line monolith (2,588 lines by the editor's
line count) that declared all 97 HTTP/WebSocket handlers as closures inside
`create_app()`. It is now a 90-line application factory that builds an
`AppContext` and calls one `register(app, ctx)` per domain module.

## Module map

| Module | Lines | Contents |
|---|---:|---|
| `web/server.py` | 90 | `create_app(config, event_bus, auth_manager=None)` — static mount, auth middleware, context build, `register(...)` calls; re-exports the rendering helpers for backwards compatibility |
| `web/context.py` | 39 | `AppContext` — `config`, `event_bus`, `app`, `logger`, `_bt_runs`, `_balance_lock`, `_app_started_at` |
| `web/deps.py` | 33 | `_require_trader`, `_require_admin`, `_save_balance` |
| `web/rendering.py` | 58 | module-level Jinja `Environment`, `fmt_time` filter, `CST`, `DEFAULT_BALANCE`, `_render`, `_T`, `_get_lang`, `_fmt_time` |
| `web/routes/health.py` | 35 | `GET /health` |
| `web/routes/auth.py` | 74 | `GET /login`, `POST /api/auth/{login,logout,change-password`}; login rate limiter |
| `web/routes/users.py` | 66 | `GET/POST /api/users`, `POST /api/users/{uid}`, `DELETE /api/users/{uid}`, `POST /api/users/{uid}/toggle`, `GET /partials/user-list` |
| `web/routes/pages.py` | 144 | `GET /`, `/dashboard`, `/strategies`, `/ai`, `/alerts`, `/settings`, `/db-manager`, `/users` |
| `web/routes/dashboard_partials.py` | 112 | `GET /api/trades`, `/api/market-state`, `/api/strategy-monitor`, `/api/price/{symbol}`, `/api/kline/{symbol}`, `/partials/trades` |
| `web/routes/alerts.py` | 119 | alerts API, alert rules, alert partials |
| `web/routes/strategies.py` | 218 | strategy CRUD, symbol mapping, reload, AI recommendation |
| `web/routes/trading.py` | 190 | `POST /api/trade`, `POST /api/trade/close/{symbol}`, `GET /partials/stats`, `GET /partials/positions` |
| `web/routes/settings.py` | 249 | settings writes, `POST /api/ai-mode`, `POST /api/signal-weights`, circuit-breaker reset, server restart |
| `web/routes/ai.py` | 239 | AI suggestions, `GET /api/deepseek-models`, `GET /api/ai-heartbeat`, `POST /api/consult`, `GET /api/news-sources` |
| `web/routes/backtest.py` | 342 | backtest page + run/progress/result/history APIs + partials |
| `web/routes/lifecycle.py` | 67 | strategy lifecycle events/generate/optimize/partial |
| `web/routes/ga.py` | 379 | GA evolve/status/stop, walk-forward, calibration; `_ga_state`/`_wf_state`/`_calib_state` |
| `web/routes/db_manager.py` | 162 | DB manager APIs (table view, row delete, backup/restore/optimize/cleanup, CSV export) |
| `web/ws/alerts.py` | 38 | `/ws/alerts` WebSocket |
| `web/routes/__init__.py`, `web/ws/__init__.py` | 4, 1 | package markers |

Totals: `web/server.py` **2,366 → 90** lines; the 21 refactor modules total
**2,659** lines (the ~290-line increase is per-module imports and docstrings).

## Invariants deliberately preserved

* `from web.server import create_app` — same name, same signature, same app.
* All **97** routes identical in method + path (baseline:
  `docs/overhaul/route-baseline.json`), verified exact-match. The 15
  `register(...)` calls run in the original top-to-bottom order, so path
  matching and OpenAPI ordering are unchanged.
* `app.state.*` names untouched: `config`, `executor`, `risk_manager`,
  `auth_manager`, `strategy_engine`, `strategy_loader`, `backtest_engine`,
  `lifecycle_manager`, `alert_manager`, `get_price`, `balance` (plus the
  `_ga_process` / `_wf_process` attributes). State is still read live through
  `getattr(app.state, ...)`, so post-construction assignment by `app/main.py`
  and by tests behaves exactly as before.
* Jinja: same module-level `Environment`, same loader dir, same `fmt_time`
  filter, same `_render` / `_T` / `_get_lang` / `_fmt_time`, same template and
  partial names.
* Authorization helpers remain **in-handler** and keep returning
  `{"error": "Forbidden"}` with status 403 — they were *not* converted to
  `HTTPException` dependencies (a `TODO(authz)` marker was added at the
  definitions in `web/deps.py` and above the user-management handlers).
* All original comments (including the security-decision ones) moved with
  their code.

## Known, intentional differences

1. **Registration order between modules is now grouped by domain.** The old
   file registered `/partials/user-list` and `/api/auth/change-password` late
   in the file; they now register with their domain module. Every
   method+path pair is unique, so FastAPI matching cannot change — the route
   parity check confirms identical method+path, and inserting a mount cannot
   shadow anything.
2. **`_ga_state` / `_wf_state` / `_calib_state` are module-level** in
   `web/routes/ga.py` instead of per-`create_app()` closures. They were
   effectively process-global in the single-app deployment; a hypothetical
   second `create_app()` in the same process would now share GA state.
3. `Path(__file__)`-relative config/script paths gained one `parent` because
   the modules sit one directory deeper; all still resolve to the same files
   (`config/config.yaml`, `config/secrets.yaml`, `config/risk_params.yaml`,
   `scripts/ga_worker.py`).

## Verification

```powershell
# 1. route parity (throwaway script in %TEMP%, not in the repo)
python "$env:TEMP\bt_route_dump.py" 'E:\Codes\Binance Trader' "$env:TEMP\bt_routes_after.json"
# actual=97 baseline=97  -> EXACT MATCH

# 2. tests
python -m pytest tests/ -q -p no:cacheprovider
# 153 passed, 2 warnings (baseline count at refactor time)

# 3. bytecode
python -m compileall -q web          # exit 0
```

Additionally, the full route surface was exercised black-box with
`TestClient` across four roles (`anonymous`/`viewer`/`trader`/`admin`) and all
51 GET endpoints returned their expected payloads. See the refactor
handover notes for the probe output.

## Concurrent edits by other workstreams (not part of this split)

The split was carried out while a separate authorisation-hardening pass was
editing the same tree. That pass landed **on top of** the split modules and
changes behaviour (routes/status codes for the endpoints below, nothing else):

* `web/routes/health.py` — `/health` now returns only
  `status`/`database`/`uptime_seconds` to unauthenticated callers and adds
  the operational fields for authenticated ones.
* `web/routes/users.py` — `GET /partials/user-list` now requires admin.
* `web/routes/alerts.py` — added `_require_admin` to the alert-reading
  endpoints.
* `web/rendering.py` — the Jinja environment now sets `autoescape` for
  HTML templates (XSS hardening).

These edits are **not** part of the file-organization refactor and are
recorded here only so the split's "behaviour unchanged" guarantee is not
misread: the split itself preserved every route exactly, and the later
hardening deliberately changed the authorisation of the endpoints listed
above. Re-run the route-parity check after any further edit to `web/`.

