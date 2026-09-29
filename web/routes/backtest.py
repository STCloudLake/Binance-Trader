"""Backtest page, run/progress/result APIs, history and HTMX partials."""
import asyncio
import json
import re
import subprocess
import sys
import tempfile
import time
import uuid as _uuid
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from db.database import get_db

from core.backtest.cost_model import (
    LIVE_SPREAD_TTL,
    default_spread_pct,
    resolve_spreads,
)
from core.market_data.universe import DEFAULT_WATCHLIST

from web.deps import _require_trader
from web.rendering import _render

# ---------------------------------------------------------------------------
# Symbol universe helpers — the backtest page no longer hardcodes five pairs.
# The authoritative list is ``GET /api/market/symbols`` (all Binance USDT
# pairs); the helpers below only build the *fallback* universe that the
# template can render without a round-trip and that keeps the page usable when
# that endpoint (or the network) is down.
# ---------------------------------------------------------------------------

#: Last-resort pairs — the watchlist fallback owned by ``core.market_data.universe``
#: (the same five pairs the engine has always shipped with, declared once there).
_FALLBACK_SYMBOLS = tuple(DEFAULT_WATCHLIST)

#: Caps for ``POST /api/backtest/fetch-data`` — they keep that request
#: synchronous and bounded (one subprocess, ≤5 symbols, ≤2 intervals).
MAX_FETCH_SYMBOLS = 5
MAX_FETCH_INTERVALS = 2
FETCH_TIMEOUT_SECONDS = 300

#: ``GET /api/backtest/spreads`` resolves at most this many pairs per call so a
#: huge selection cannot fan out into hundreds of order-book lookups.
MAX_SPREAD_LOOKUP_SYMBOLS = 30

#: Intervals Binance klines support; anything else is rejected before the
#: downloader subprocess is ever started.  This is a *download-request validator*,
#: deliberately a superset of ``core.market_data.provider.INTERVAL_SPEC``: it also
#: accepts ``1M``, which the interval registry dropped because a monthly bar is not
#: a strategy timeframe — it is only ever fetched to disk here.
ALLOWED_INTERVALS = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
                     "6h", "8h", "12h", "1d", "3d", "1w", "1M")

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{4,24}$")


def _repo_root() -> Path:
    """Repository root (this file lives at ``<root>/web/routes/backtest.py``)."""
    return Path(__file__).resolve().parents[2]


def _data_dir(config) -> Path:
    """Root data dir (``config.data_dir``) — the downloader's ``--data-dir``."""
    return Path(getattr(config, "data_dir", None) or (_repo_root() / "data"))


def _market_dir(config) -> Path:
    """``data/market`` — the parquet cache layout the DataFeeder reads."""
    return _data_dir(config) / "market"


def _cached_intervals(config) -> dict:
    """Map ``symbol -> [interval, ...]`` for pairs that already have parquet data."""
    result: dict = {}
    try:
        root = _market_dir(config)
        if not root.is_dir():
            return result
        for sym_dir in sorted(root.iterdir()):
            if not sym_dir.is_dir():
                continue
            intervals = sorted(p.stem for p in sym_dir.glob("*.parquet"))
            if intervals:
                result[sym_dir.name] = intervals
    except OSError:
        pass
    return result


def _known_symbols(config, loader, cached: dict | None = None) -> list:
    """Fallback universe: locally cached pairs ∪ strategy pairs ∪ built-ins."""
    symbols = set(cached if cached is not None else _cached_intervals(config))
    if loader is not None:
        try:
            for s in loader.load_all():
                symbols.update(s.symbols or [])
        except Exception:
            pass
    return list(_FALLBACK_SYMBOLS) + sorted(symbols.difference(_FALLBACK_SYMBOLS))


def _parquet_rows(path: Path) -> int:
    """Row count of a parquet file (metadata first — never loads the frame)."""
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(str(path)).metadata.num_rows)
    except Exception:
        pass
    try:
        import pandas as pd
        return int(len(pd.read_parquet(path)))
    except Exception:
        return -1


def _locate_parquet(config, symbol: str, interval: str) -> Path | None:
    """Find ``{symbol}/{interval}.parquet`` under the data dir.

    ``data/market/{symbol}/{interval}.parquet`` is the canonical layout; the
    two extra probes make the endpoint tolerant of a downloader that treats
    ``--data-dir`` as the data root rather than the market root.
    """
    mdir = _market_dir(config)
    candidates = [mdir / symbol / f"{interval}.parquet",
                  mdir.parent / symbol / f"{interval}.parquet"]
    for path in candidates:
        if path.is_file():
            return path
    try:
        found = sorted(mdir.parent.glob(f"**/{symbol}/{interval}.parquet"))
    except OSError:
        found = []
    return found[0] if found else None


def _display_path(path: Path) -> str:
    """Repo-relative path when possible (stable across machines)."""
    try:
        return path.resolve().relative_to(_repo_root()).as_posix()
    except Exception:
        return path.as_posix()


def _backtest_context(config, loader) -> dict:
    """Template context shared by the page and the HTMX config partial."""
    strategy_configs = []
    if loader is not None:
        try:
            strategy_configs = loader.load_all()
        except Exception:
            pass
    cached = _cached_intervals(config)
    default_end = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    return {
        "available_strategies": [s.name for s in strategy_configs],
        "available_symbols": _known_symbols(config, loader, cached),
        "cached_map": cached,
        "cached_symbols": sorted(cached),
        "strategy_symbols": {s.name: (s.symbols or []) for s in strategy_configs},
        "strategy_configs": strategy_configs,
        "strategy_timeframes": {s.name: list(s.timeframes or []) for s in strategy_configs},
        "max_fetch_symbols": MAX_FETCH_SYMBOLS,
        "max_fetch_intervals": MAX_FETCH_INTERVALS,
        "allowed_intervals": list(ALLOWED_INTERVALS),
        "default_start": default_start,
        "default_end": default_end,
        # Cost model: the spread is *derived per symbol* now (override → live
        # depth-derived → default).  The page renders the resolution rule and
        # fetches the per-symbol table from /api/backtest/spreads.
        "default_spread_pct": default_spread_pct(config),
        "live_spread_enabled": bool(
            getattr(config, "backtest_live_spread_enabled", False)),
        "live_spread_ttl_seconds": int(getattr(
            config, "backtest_live_spread_ttl", LIVE_SPREAD_TTL) or LIVE_SPREAD_TTL),
        "spread_overrides_cfg": {
            str(k).upper(): float(v)
            for k, v in (getattr(config, "backtest_spread_pct", None) or {}).items()
            if isinstance(v, (int, float))},
    }


def register(app: FastAPI, ctx) -> None:
    config = ctx.config
    _bt_runs = ctx._bt_runs

    # ---- Backtest routes ----
    @app.get("/backtest", response_class=HTMLResponse)
    async def backtest_page(request: Request):
        user = getattr(request.state, "user", None)
        if not user or not user.is_trader:
            return RedirectResponse(url="/dashboard", status_code=302)
        loader = getattr(app.state, "strategy_loader", None)
        context = _backtest_context(config, loader)
        context.update({"request": request, "current_page": "backtest"})
        return _render("backtest.html", context)

    @app.post("/api/backtest/run")
    async def run_backtest(request: Request,
                           strategies: str = Form(...),
                           symbols: str = Form(...),
                           date_start: str = Form("2025-01-01"),
                           date_end: str = Form("2026-01-01"),
                           mode: str = Form("full"),
                           initial_balance: float = Form(10000.0),
                           strategy_symbols: str = Form("{}"),
                           simulate_ai_weights: str = Form("1"),
                           ml_engine: str = Form("lightgbm"),
                           skip_ml_training: str = Form("0"),
                           cost_enabled: str = Form("1"),
                           taker_fee_pct: float = Form(0.04),
                           spread_overrides: str = Form(""),
                           spread_btc: str = Form(""),
                           spread_eth: str = Form(""),
                           spread_bnb: str = Form(""),
                           spread_sol: str = Form(""),
                           spread_xrp: str = Form("")):
        if err := _require_trader(request): return err
        engine = getattr(app.state, "backtest_engine", None)
        if not engine:
            return HTMLResponse('<div class="text-red-400">Backtest engine not initialized</div>')

        strategy_list = [s.strip() for s in strategies.split(",") if s.strip()]
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        if not strategy_list or not symbol_list:
            return HTMLResponse('<div class="text-red-400">Please select strategies and symbols</div>')

        # Parse backtest-only strategy→symbol mapping (does NOT modify live config)
        bt_strategy_symbols = {}
        try:
            bt_strategy_symbols = json.loads(strategy_symbols) if strategy_symbols else {}
        except (json.JSONDecodeError, TypeError):
            pass

        # ── Runtime cost model params ──
        # ``spread_overrides`` is the per-symbol override map the user edited in
        # the cost table (JSON).  The legacy ``spread_*`` form fields are still
        # accepted for backward compatibility and are folded into that same map;
        # the mapping lives in their field names, so there is no second copy of
        # the symbol list here.  Anything *not* overridden is resolved per symbol
        # by the cost model (live order book → configured default).
        spread_map: dict = {}
        if spread_overrides.strip():
            try:
                parsed = json.loads(spread_overrides)
                if isinstance(parsed, dict):
                    spread_map = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        legacy_spreads = {
            "BTCUSDT": spread_btc, "ETHUSDT": spread_eth, "BNBUSDT": spread_bnb,
            "SOLUSDT": spread_sol, "XRPUSDT": spread_xrp,
        }
        for symbol, raw in legacy_spreads.items():
            if str(raw).strip():
                try:
                    spread_map[symbol] = float(raw)
                except (TypeError, ValueError):
                    pass
        bt_config = getattr(app.state, "config", None)
        if bt_config:
            bt_config.backtest_cost_enabled = (cost_enabled == "1")
            bt_config.backtest_taker_fee_pct = taker_fee_pct
            # NOTE: config.backtest_spread_pct (the shipped override table) is
            # deliberately left alone — the user's edits travel with the run via
            # ``spread_overrides`` so a symbol without an override keeps being
            # resolved live instead of being frozen at a stale number.

        # Detect GPU for display
        bt_device = "cpu"
        if ml_engine in ("tft", "patchtst"):
            try:
                import torch
                if torch.cuda.is_available():
                    _t = torch.zeros(1).cuda()
                    bt_device = "cuda"
            except Exception:
                pass

        # Cleanup old completed runs (> 1 hour ago) to prevent memory leak
        _now = time.time()
        for rid in list(_bt_runs.keys()):
            r = _bt_runs[rid]
            if r.get("done") and _now - r.get("started", 0) > 3600:
                _bt_runs.pop(rid, None)

        run_id = _uuid.uuid4().hex[:12]
        _bt_runs[run_id] = {
            "progress": 0, "total": 0, "done": False,
            "date_start": date_start, "date_end": date_end,
            "strategies": strategies,
            "symbols": symbols,
            "ml_engine": ml_engine,
            "device": bt_device,
            "skip_training": (skip_ml_training == "1"),
            "started": time.time(),
        }

        def on_progress(current, total, ts):
            _bt_runs[run_id]["progress"] = current
            _bt_runs[run_id]["total"] = total

        def _run_bt_blocking():
            return engine.run_with_exit_evaluation(
                strategy_list, symbol_list, date_start, date_end,
                initial_balance, mode, progress_callback=on_progress,
                strategy_symbols=bt_strategy_symbols,
                simulate_ai_weights=(simulate_ai_weights == "1"),
                ml_engine=ml_engine,
                skip_ml_training=(skip_ml_training == "1"),
                spread_overrides=spread_map)

        async def _run_bt():
            loop = asyncio.get_event_loop()
            import concurrent.futures
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = await loop.run_in_executor(pool, _run_bt_blocking)
                _bt_runs[run_id]["result"] = result
            except Exception as e:
                from loguru import logger
                logger.error(f"Backtest run {run_id} failed: {e}")
                _bt_runs[run_id]["result"] = {"error": str(e)}
            finally:
                _bt_runs[run_id]["done"] = True

        asyncio.create_task(_run_bt())

        resp = _render("partials/backtest_progress.html", {
            "request": None, "run_id": run_id,
            "date_start": date_start, "date_end": date_end,
            "strategies": strategies,
            "symbols": symbols,
            "ml_engine": ml_engine,
            "device": bt_device,
            "skip_training": (skip_ml_training == "1"),
            "initial_pct": 0,
        })
        resp.set_cookie("bt_active_run", run_id, max_age=3600, httponly=False)
        return resp

    @app.get("/partials/backtest-active")
    async def backtest_active(request: Request):
        """Check for active backtest and return progress bar if running.

        First checks the bt_active_run cookie, then falls back to any
        non-done run (recovers from browser sleep / page refresh).
        """
        run_id = request.cookies.get("bt_active_run", "")
        if not run_id or run_id not in _bt_runs or _bt_runs[run_id].get("done"):
            # Cookie lost or run done — find any active run
            active = [(rid, r) for rid, r in _bt_runs.items() if not r.get("done")]
            if active:
                run_id = active[0][0]
            else:
                return HTMLResponse("")

        run = _bt_runs[run_id]
        if run.get("done"):
            return HTMLResponse("")

        total = run.get("total", 100)
        current = run.get("progress", 0)
        pct = round(current / max(total, 1) * 100, 1)
        return _render("partials/backtest_progress.html", {
            "request": None, "run_id": run_id,
            "date_start": run.get("date_start", ""),
            "date_end": run.get("date_end", ""),
            "strategies": run.get("strategies", ""),
            "symbols": run.get("symbols", ""),
            "ml_engine": run.get("ml_engine", ""),
            "device": run.get("device", ""),
            "skip_training": run.get("skip_training", False),
            "initial_pct": pct,
        })

    @app.get("/api/backtest/progress/{run_id}")
    async def backtest_progress(run_id: str):
        run = _bt_runs.get(run_id)
        if not run:
            return JSONResponse({"error": "Unknown run"}, status_code=404)
        return {
            "progress": run["progress"],
            "total": run["total"],
            "pct": round(run["progress"] / max(run["total"], 1) * 100, 1),
            "done": run["done"],
            "date_start": run.get("date_start", ""),
            "date_end": run.get("date_end", ""),
        }

    @app.get("/api/backtest/result/{run_id}")
    async def backtest_result(run_id: str, request: Request):
        # Despite being a GET this mutates: it inserts a backtest_records row,
        # writes data/backtest/<id>.json and removes the run from the registry.
        if err := _require_trader(request): return err
        run = _bt_runs.get(run_id)
        if not run or not run.get("done"):
            return HTMLResponse('<div class="text-yellow-400">Still running...</div>')
        result = run["result"]
        if result.get("error"):
            return _render("partials/backtest_results.html",
                           {"request": None, "error": result["error"]})

        from core.backtest.report import generate_report
        report = generate_report(result)

        # Persist to DB
        mode = "full"
        strategy_list = result.get("strategies", [])
        symbol_list = result.get("symbols", [])
        date_start = result.get("date_start", "")
        date_end = result.get("date_end", "")
        initial_balance = result.get("initial_balance", 10000)

        record_id = None
        try:
            db = await get_db()
            cursor = await db.execute(
                "INSERT INTO backtest_records (mode, strategies, symbols, date_start,"
                " date_end, initial_balance, final_balance, metrics, trades_count)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (mode, json.dumps(strategy_list), json.dumps(symbol_list),
                 date_start, date_end, initial_balance,
                 result["final_balance"], json.dumps(report["summary"]),
                 len(result["trades"])))
            await db.commit()
            record_id = cursor.lastrowid
            await db.close()

            result_dir = Path(config.data_dir) / "backtest"
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / f"{record_id}.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, default=str)
        except Exception:
            pass

        # Clean up
        _bt_runs.pop(run_id, None)

        resp = _render("partials/backtest_results.html", {
            "request": None,
            "summary": report["summary"],
            "chart_data": report["chart_data"],
            "trades": result["trades"],
            "runtime": result["metrics"].get("runtime_seconds", 0),
            "date_start": date_start,
            "date_end": date_end,
            "record_id": record_id,
            "per_matrix": report.get("per_matrix", {}),
            "config": {"symbols": symbol_list},
        })
        resp.delete_cookie("bt_active_run")
        return resp

    @app.get("/api/backtest/spreads")
    async def backtest_spreads(symbols: str = "", overrides: str = ""):
        """Resolved spread (%) per selected symbol + where each value came from.

        Resolution order (``core/backtest/cost_model.py``): explicit override →
        live depth-derived (``/api/v3/depth``, cached) → documented default.
        Registered *before* ``/api/backtest/{record_id}`` so the int path param
        cannot swallow it.
        """
        symbol_list = list(dict.fromkeys(
            s.strip().upper() for s in (symbols or "").split(",") if s.strip()
        ))[:MAX_SPREAD_LOOKUP_SYMBOLS]
        override_arg: dict = {}
        if (overrides or "").strip():
            try:
                parsed = json.loads(overrides)
                if isinstance(parsed, dict):
                    override_arg = {str(k).upper(): v for k, v in parsed.items()}
            except (json.JSONDecodeError, TypeError):
                pass
        table = await asyncio.to_thread(
            resolve_spreads, symbol_list, config, override_arg)
        return {
            "symbols": [table[s] for s in symbol_list if s in table],
            "default_spread_pct": default_spread_pct(config),
            "live_spread_enabled": bool(
                getattr(config, "backtest_live_spread_enabled", False)),
            "ttl_seconds": int(getattr(
                config, "backtest_live_spread_ttl", LIVE_SPREAD_TTL) or LIVE_SPREAD_TTL),
            "taker_fee_pct": float(
                getattr(config, "backtest_taker_fee_pct", 0.04) or 0.04),
            "max_symbols": MAX_SPREAD_LOOKUP_SYMBOLS,
        }

    @app.get("/api/backtest/{record_id}")
    async def get_backtest(record_id: int, request: Request):
        """Return stored backtest result — reloads full data from JSON file.

        Trader-gated for the same reason as its sibling `/api/backtest/result/{id}`:
        stored results expose strategy internals and trade history, so a read-only
        viewer must not be able to enumerate them.
        """
        if err := _require_trader(request):
            return err
        result_path = Path(config.data_dir) / "backtest" / f"{record_id}.json"
        if not result_path.exists():
            # Fallback: return DB summary only
            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM backtest_records WHERE id=?", (record_id,))
            row = await cursor.fetchone()
            await db.close()
            if not row:
                return JSONResponse({"error": "Not found"}, status_code=404)
            rec = dict(row)
            if rec.get("metrics"):
                rec["metrics_parsed"] = json.loads(rec["metrics"])
            return rec

        try:
            with open(result_path, encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            return JSONResponse({"error": "Failed to load result file"}, status_code=500)

        from core.backtest.report import generate_report
        report = generate_report(result)

        return _render("partials/backtest_results.html", {
            "request": None,
            "summary": report["summary"],
            "chart_data": report["chart_data"],
            "trades": result.get("trades", []),
            "runtime": result.get("metrics", {}).get("runtime_seconds", 0),
            "date_start": result.get("date_start", ""),
            "date_end": result.get("date_end", ""),
            "per_matrix": report.get("per_matrix", {}),
            "config": {"symbols": result.get("symbols", [])},
        })

    @app.delete("/api/backtest/{record_id}")
    async def delete_backtest(record_id: int, request: Request):
        if err := _require_trader(request): return err
        db = await get_db()
        await db.execute("DELETE FROM backtest_records WHERE id=?", (record_id,))
        await db.commit()
        await db.close()
        return {"ok": True}

    @app.get("/partials/backtest-config")
    async def partial_backtest_config():
        loader = getattr(app.state, "strategy_loader", None)
        context = _backtest_context(config, loader)
        context["request"] = None
        return _render("partials/backtest_config.html", context)

    @app.post("/api/backtest/fetch-data")
    async def fetch_backtest_data(request: Request,
                                  symbols: str = Form(...),
                                  intervals: str = Form("1h"),
                                  date_start: str = Form(...),
                                  date_end: str = Form(...)):
        """Download parquet history for the selected pairs.

        Runs ``scripts/download_history.py`` (the frozen CLI:
        ``--symbols A,B --intervals 1h,4h --start YYYY-MM-DD --end YYYY-MM-DD
        [--data-dir PATH]``) as a *synchronous, bounded* subprocess: at most
        :data:`MAX_FETCH_SYMBOLS` symbols × :data:`MAX_FETCH_INTERVALS`
        intervals and a :data:`FETCH_TIMEOUT_SECONDS` timeout.  Selection above
        those caps is rejected up front so the request can never fan out.
        """
        if err := _require_trader(request): return err

        symbol_list = list(dict.fromkeys(
            s.strip().upper() for s in (symbols or "").split(",") if s.strip()))
        interval_list = list(dict.fromkeys(
            i.strip() for i in (intervals or "").split(",") if i.strip())) or ["1h"]

        if not symbol_list:
            return JSONResponse({"ok": False, "error": "请至少选择一个交易对"}, status_code=400)
        if len(symbol_list) > MAX_FETCH_SYMBOLS:
            return JSONResponse(
                {"ok": False, "error": f"一次最多下载 {MAX_FETCH_SYMBOLS} 个交易对（已选 {len(symbol_list)} 个），请分批下载"},
                status_code=400)
        bad_symbols = [s for s in symbol_list if not _SYMBOL_RE.match(s)]
        if bad_symbols:
            return JSONResponse(
                {"ok": False, "error": "交易对格式无效: " + ", ".join(bad_symbols)},
                status_code=400)
        if len(interval_list) > MAX_FETCH_INTERVALS:
            return JSONResponse(
                {"ok": False, "error": f"一次最多下载 {MAX_FETCH_INTERVALS} 个周期（已选 {len(interval_list)} 个）"},
                status_code=400)
        bad_intervals = [i for i in interval_list if i not in ALLOWED_INTERVALS]
        if bad_intervals:
            return JSONResponse(
                {"ok": False, "error": "不支持的周期: " + ", ".join(bad_intervals)},
                status_code=400)
        for label, value in (("起始日期", date_start), ("结束日期", date_end)):
            try:
                datetime.strptime(value.strip(), "%Y-%m-%d")
            except (ValueError, AttributeError):
                return JSONResponse(
                    {"ok": False, "error": f"{label}格式无效（需 YYYY-MM-DD）: {value}"},
                    status_code=400)
        if date_start.strip() > date_end.strip():
            return JSONResponse(
                {"ok": False, "error": "起始日期不能晚于结束日期"}, status_code=400)

        script = _repo_root() / "scripts" / "download_history.py"
        if not script.is_file():
            return JSONResponse(
                {"ok": False, "error": f"下载脚本不存在: {_display_path(script)}"},
                status_code=502)

        cmd = [sys.executable, str(script),
               "--symbols", ",".join(symbol_list),
               "--intervals", ",".join(interval_list),
               "--start", date_start.strip(),
               "--end", date_end.strip(),
               # Frozen CLI contract: --data-dir is the *root* data dir; the
               # script itself writes <data-dir>/market/<SYMBOL>/<interval>.parquet.
               "--data-dir", str(_data_dir(config))]

        def _run_download():
            # stdout/stderr go to a temp file (not a pipe): no reader thread, no
            # locale-dependent pipe decoding, and the output survives a timeout.
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8",
                                        errors="replace") as sink:
                proc = subprocess.run(
                    cmd, cwd=str(_repo_root()), stdout=sink,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    timeout=FETCH_TIMEOUT_SECONDS)
                sink.seek(0)
                return proc.returncode, sink.read()

        try:
            returncode, log = await asyncio.to_thread(_run_download)
        except subprocess.TimeoutExpired:
            return JSONResponse(
                {"ok": False,
                 "error": f"下载超时（超过 {FETCH_TIMEOUT_SECONDS} 秒），请缩短日期区间或减少交易对"},
                status_code=504)
        except Exception as e:  # spawn failure, missing interpreter, ...
            return JSONResponse({"ok": False, "error": f"启动下载进程失败: {e}"},
                                status_code=502)

        results = []
        errors = []
        for sym in symbol_list:
            for iv in interval_list:
                path = _locate_parquet(config, sym, iv)
                rows = _parquet_rows(path) if path is not None else -1
                if path is not None and rows > 0:
                    results.append({
                        "symbol": sym, "interval": iv, "ok": True, "rows": rows,
                        "path": _display_path(path),
                        "downloaded_at": datetime.fromtimestamp(
                            path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    continue
                if path is not None and rows == 0:
                    message = "数据文件为空（该区间可能没有K线）"
                elif returncode != 0:
                    message = f"下载进程退出码 {returncode}，未生成数据文件"
                else:
                    message = "未生成数据文件（交易对不存在或区间无数据）"
                entry = {"symbol": sym, "interval": iv, "ok": False, "rows": 0,
                         "path": _display_path(path) if path is not None else None,
                         "error": message}
                results.append(entry)
                errors.append({"symbol": sym, "interval": iv, "error": message})

        payload = {
            "ok": not errors,
            "results": results,
            "errors": errors,
            "symbols": symbol_list,
            "intervals": interval_list,
            "log_tail": (log or "")[-2000:],
        }
        if errors:
            payload["error"] = ("; ".join(
                f"{e['symbol']}/{e['interval']}: {e['error']}" for e in errors))[:500]
        return JSONResponse(payload)

    @app.get("/partials/backtest-list")
    async def partial_backtest_list():
        db = await get_db()
        cursor = await db.execute(
            "SELECT * FROM backtest_records ORDER BY created_at DESC LIMIT 20")
        records = [dict(r) for r in await cursor.fetchall()]
        await db.close()
        return _render("partials/backtest_list.html",
                       {"request": None, "records": records})
