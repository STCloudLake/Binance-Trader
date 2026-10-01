"""GA evolution and walk-forward endpoints.

The ``_ga_state`` / ``_wf_state`` dicts used to be closures
inside ``create_app()``; they now live at module level exactly as they were
(one shared state per process).

Symbol selection (fix: the GA used to run on a hardcoded five-pair list)
-----------------------------------------------------------------------
Every GA entry point now takes the symbols the user picked and validates them
against the exchange universe (``GET /api/market/symbols``'s source of truth,
shared through ``app.state.universe``):

* accepted as a JSON list (the panel) **or** a comma-separated form field;
* de-duplicated, upper-cased, capped at :data:`MAX_GA_SYMBOLS`;
* unknown / non-``TRADING`` / non-USDT symbols → HTTP 400 with the offenders
  named;
* omitted → the persisted watchlist (:func:`default_ga_symbols`);
* the validated list is written to the job file the worker subprocess reads.
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from web.deps import _require_trader
from web.rendering import _render

#: Hard cap on the symbols one GA / walk-forward / calibration job may use.
#: Keeps a batch bounded (population × generations × symbols) as required.
MAX_GA_SYMBOLS = 20


# ── symbol plumbing ───────────────────────────────────────────────

def _parse_symbols(raw) -> list:
    """Normalize the ``symbols`` parameter to a de-duplicated upper-case list.

    Accepts a comma-separated string (form field), a list of strings (JSON body)
    or a list of comma-separated strings; anything else yields ``[]``.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        items = [piece for entry in raw for piece in str(entry).split(",")]
    else:
        return []
    out: list = []
    for item in items:
        symbol = str(item).strip().upper()
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def _int_param(payload, key, default, minimum=None, maximum=None) -> int:
    """Numeric job parameter — form fields arrive as strings, JSON as numbers."""
    try:
        value = int(float(payload.get(key, default)))
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(value, int(minimum))
    if maximum is not None:
        value = min(value, int(maximum))
    return value


def _float_param(payload, key, default) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _bool_param(payload, key, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _list_param(payload, key) -> list:
    """A JSON-encoded list field (form) or a real list (JSON body)."""
    value = payload.get(key)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _spread_param(payload, default: dict) -> dict:
    """Per-symbol spread override map (dict, or a JSON string from a form).

    Callers pass ``{}``: an empty map means "no explicit override", and the
    per-symbol spread is then resolved by ``core.backtest.cost_model``
    (config table → live order book → its documented default) rather than by a
    second copy of a default table living in this route.
    """
    value = payload.get("spread_pct", None)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return dict(default)
    if isinstance(value, dict):
        return value
    return dict(default)


def _universe_for(app, config):
    """The symbol universe, shared with ``web.routes.market`` when available.

    Tests (and any embedder) can pre-inject a fake under ``app.state.universe``;
    otherwise the same public data host the market pages use is wrapped lazily.
    """
    universe = getattr(app.state, "universe", None)
    if universe is not None:
        return universe
    from core.market_data.data_client import MarketDataClient
    from core.market_data.universe import Universe

    universe = Universe(config, client=MarketDataClient(
        getattr(config, "market_data_host", "https://data-api.binance.vision")))
    try:
        app.state.universe = universe
    except Exception:  # pragma: no cover - defensive
        pass
    return universe


async def default_ga_symbols(config) -> list:
    """Default GA selection: the persisted watchlist (``GET /api/market/watchlist``)."""
    from core.market_data.universe import DEFAULT_WATCHLIST, load_watchlist

    try:
        symbols = await load_watchlist(
            getattr(config, "db_path", ""), DEFAULT_WATCHLIST)
    except Exception:  # pragma: no cover - defensive
        symbols = []
    return _parse_symbols(symbols) or list(DEFAULT_WATCHLIST)


async def resolve_ga_symbols(app, config, payload) -> tuple:
    """Validate the requested GA symbols → ``(symbols, error_message)``.

    ``symbols`` is ``None`` when ``error_message`` is set (caller returns 400).
    A universe that cannot be loaded at all (offline data host) does not fail an
    otherwise valid request — same tolerance as ``POST /api/market/watchlist``.
    """
    symbols = _parse_symbols((payload or {}).get("symbols"))
    if not symbols:
        symbols = await default_ga_symbols(config)
    if not symbols:
        return None, "symbols must contain at least one symbol"
    if len(symbols) > MAX_GA_SYMBOLS:
        return None, (f"too many symbols ({len(symbols)}): a GA run is capped at "
                      f"{MAX_GA_SYMBOLS} symbols")

    universe = _universe_for(app, config)
    try:
        await universe.get_symbols()
    except Exception:
        return symbols, None  # data host unavailable — do not fail the request

    rejected = []
    for symbol in symbols:
        info = universe.get(symbol)
        if info is None:
            rejected.append(f"{symbol} (unknown)")
            continue
        status = str(getattr(info, "status", "") or "").upper()
        if status and status != "TRADING":
            rejected.append(f"{symbol} ({status})")
            continue
        quote = str(getattr(info, "quote_asset", "USDT") or "USDT").upper()
        if quote != "USDT":
            rejected.append(f"{symbol} (quote asset {quote}: only USDT pairs)")
    if rejected:
        return None, "unknown or non-trading symbols: " + ", ".join(rejected)
    return symbols, None


async def _request_payload(request: Request) -> dict:
    """Form fields or JSON body — the panel posts JSON, forms are accepted too."""
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            return {}
        return body if isinstance(body, dict) else {}
    if "form" in content_type:
        try:
            form = await request.form()
        except Exception:
            return {}
        return {key: form[key] for key in form.keys()}
    return {}



# ── GA Evolution ──────────────────────────────────────────────────

_ga_state = {
    "running": False,
    "generation": 0, "total_generations": 0,
    "best_fitness": 0, "best_sharpe": 0, "best_win_rate": 0,
    "best_trades": 0, "champion_name": "", "champion_config": None,
    "population_size": 0, "started": 0, "history": [],
    "error": None,
    # ── Credibility metrics (previously dropped here while the panel template
    # already rendered them — the DSR/WF block was dead UI) ──
    "validation": None, "dsr": None, "provenance": None,
    "published": None, "rejection_reasons": [], "seed": 0,
    # ── Live progress detail (additive; the worker writes these alongside the
    # keys the panel has always read) ──
    "eval_completed": 0, "eval_total": 0, "eval_equivalent": 0.0,
    "avg_fitness": 0, "progress_updated_at": None, "progress": None,
}

# ── Walk-Forward state ────────────────────────────────────────────

_wf_state = {
    "running": False, "current_window": 0, "total_windows": 0,
    "completed": [], "report": None, "error": None, "started": 0,
    "elapsed_seconds": 0, "phase": "idle",
}


def register(app: FastAPI, ctx) -> None:
    config = ctx.config

    @app.post("/api/ga/evolve")
    async def ga_evolve(request: Request):
        if err := _require_trader(request): return err
        body = await _request_payload(request)
        engine = getattr(app.state, "backtest_engine", None)
        loader = getattr(app.state, "strategy_loader", None)
        if not engine or not loader:
            return JSONResponse({"error": "Engine or loader not initialized"}, status_code=500)

        # ── Symbols: user-chosen, validated against the exchange universe ──
        symbols, symbol_error = await resolve_ga_symbols(app, config, body)
        if symbol_error:
            return JSONResponse({"error": symbol_error}, status_code=400)

        date_start = body.get("date_start", "2026-05-01")
        date_end = body.get("date_end", "2026-05-31")
        validation_start = body.get("validation_start") or None
        pop_size = _int_param(body, "population_size", 60, maximum=120)
        generations = _int_param(body, "generations", 20, maximum=50)
        max_workers = _int_param(body, "max_workers", 1, minimum=1, maximum=16)  # clamp 1-16
        seed_strategies = _list_param(body, "seed_strategies")
        resume = _bool_param(body, "resume", False)
        # ── Reproducibility: every job carries a seed ──
        # 0/absent → a fresh seed (recorded in the job file and the provenance
        # block), so a caller can replay the exact same run.
        seed = _int_param(body, "seed", 0, minimum=0)
        if not seed:
            seed = int.from_bytes(os.urandom(4), "big")

        # ── Runtime cost model params (override config.yaml) ──
        cost_enabled = _bool_param(body, "cost_enabled", True)
        taker_fee_pct = _float_param(body, "taker_fee_pct", 0.04)
        # No literal symbol→spread table here: a symbol the caller did not
        # override is resolved by ``core.backtest.cost_model`` (config override →
        # live order book → documented default).  Only user-supplied entries are
        # written into the run's config table.
        spread_pct = _spread_param(body, {})
        # Apply overrides to the active config
        bt_config = getattr(app.state, "config", None)
        if bt_config:
            bt_config.backtest_cost_enabled = cost_enabled
            bt_config.backtest_taker_fee_pct = taker_fee_pct
            if spread_pct:
                # Only clobber the shipped override table when the caller
                # actually supplied per-symbol values for this run.
                bt_config.backtest_spread_pct = spread_pct
        _ga_state["cost_config"] = {
            "enabled": cost_enabled, "fee_pct": taker_fee_pct, "spreads": spread_pct}

        # ── Write job file for subprocess worker ──
        import uuid
        import subprocess
        data_dir = str(Path(loader.strategies_dir).parent) if hasattr(loader, 'strategies_dir') else "data"
        jobs_dir = Path(data_dir) / "data" / "ga_jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job_id = str(uuid.uuid4())[:8]
        job_file = str(jobs_dir / f"ga_{job_id}.json")

        job_data = {
            "population_size": pop_size, "generations": generations,
            "symbols": symbols, "date_start": date_start, "date_end": date_end,
            "validation_start": validation_start,
            "max_workers": max_workers,
            "seed_strategies": seed_strategies,
            "cost_enabled": cost_enabled,
            "taker_fee_pct": taker_fee_pct,
            "spread_pct": spread_pct,
            "resume": resume,
            "seed": seed,
        }
        with open(job_file, "w") as f:
            json.dump(job_data, f)

        # ── Kill previous worker if running ──
        old_proc = getattr(app.state, "_ga_process", None)
        if old_proc and old_proc.poll() is None:
            old_proc.terminate()
            try:
                old_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                old_proc.kill()
        old_wf = getattr(app.state, "_wf_process", None)
        if old_wf and old_wf.poll() is None:
            old_wf.terminate()
            try:
                old_wf.wait(timeout=5)
            except subprocess.TimeoutExpired:
                old_wf.kill()

        # ── Spawn worker subprocess ──
        worker_script = str(Path(__file__).parent.parent.parent / "scripts" / "ga_worker.py")
        ga_start_time = time.time()

        _ga_state.update({
            "running": True, "generation": 0, "total_generations": generations,
            "best_fitness": 0, "best_sharpe": 0, "best_win_rate": 0,
            "best_trades": 0, "champion_name": "", "champion_config": None,
            "population_size": pop_size, "started": ga_start_time,
            "eval_completed": 0, "eval_total": 0, "phase": "init",
            "eval_equivalent": 0.0, "avg_fitness": 0,
            "progress_updated_at": None, "progress": None,
            "history": [], "error": None, "stopped": False, "resumable": False,
            "checkpoint_gen": 0, "job_file": job_file,
            "validation": None, "dsr": None, "provenance": None,
            "published": None, "rejection_reasons": [], "seed": seed,
            "params": {
                "date_start": date_start, "date_end": date_end,
                "validation_start": validation_start,
                "population_size": pop_size, "generations": generations,
                "max_workers": max_workers,
                "symbols": symbols,
                "seed": seed,
                "mode": "ga",
            },
        })

        proc = subprocess.Popen(
            [sys.executable, worker_script, "--job-type", "ga", "--job-file", job_file],
            stdout=subprocess.DEVNULL, stderr=open(job_file + ".log", "w"),
        )
        app.state._ga_process = proc

        # ── Background monitor: poll progress/result files ──
        async def _monitor_ga():
            progress_file = job_file + ".progress"
            result_file = job_file + ".result"
            while _ga_state["running"]:
                await asyncio.sleep(1)
                _ga_state["elapsed_seconds"] = time.time() - ga_start_time

                # Check if process is still alive
                poll_result = proc.poll()
                if poll_result is not None:
                    # Process exited — read result
                    try:
                        if Path(result_file).exists():
                            with open(result_file) as f:
                                result = json.load(f)
                            _ga_state["running"] = False
                            _ga_state["elapsed_seconds"] = time.time() - ga_start_time
                            if "error" in result:
                                _ga_state["error"] = result["error"]
                            else:
                                _ga_state["champion_name"] = result.get("champion_name", "")
                                _ga_state["champion_config"] = result.get("champion_config")
                                _ga_state["best_fitness"] = result.get("fitness", 0)
                                _ga_state["best_sharpe"] = result.get("sharpe", 0)
                                _ga_state["best_win_rate"] = result.get("win_rate", 0)
                                _ga_state["best_trades"] = result.get("trade_count", 0)
                                # Credibility metrics — these were computed by the
                                # worker all along and dropped right here.
                                _ga_state["validation"] = result.get("validation")
                                _ga_state["dsr"] = result.get("dsr")
                                _ga_state["provenance"] = result.get("provenance")
                                _ga_state["published"] = result.get("published")
                                _ga_state["rejection_reasons"] = result.get(
                                    "rejection_reasons", []) or []
                                _ga_state["seed"] = result.get(
                                    "seed", _ga_state.get("seed", 0))
                        else:
                            _ga_state["running"] = False
                            _ga_state["error"] = f"Worker exited with code {poll_result}"
                    except Exception as e:
                        _ga_state["running"] = False
                        _ga_state["error"] = str(e)
                    break

                # Read progress
                try:
                    if Path(progress_file).exists():
                        with open(progress_file) as f:
                            progress = json.load(f)
                        _ga_state["phase"] = progress.get("phase", "evolving")
                        _ga_state["eval_completed"] = progress.get("eval_completed", 0)
                        _ga_state["eval_total"] = progress.get("eval_total", 0)
                        # Sub-generation detail (absent on pre-fix job files, so
                        # every one of these is guarded by a presence check).
                        _ga_state["progress"] = progress
                        if progress.get("eval_equivalent") is not None:
                            _ga_state["eval_equivalent"] = progress["eval_equivalent"]
                        if progress.get("avg_fitness") is not None:
                            _ga_state["avg_fitness"] = progress["avg_fitness"]
                        if progress.get("best_trades") is not None:
                            _ga_state["best_trades"] = progress["best_trades"]
                        if progress.get("updated_at"):
                            _ga_state["progress_updated_at"] = progress["updated_at"]
                        if "generation" in progress:
                            _ga_state["generation"] = progress["generation"]
                            _ga_state["total_generations"] = progress["total_generations"]
                            _ga_state["best_fitness"] = progress.get("best_fitness", 0)
                            _ga_state["best_sharpe"] = progress.get("best_sharpe", 0)
                except Exception:
                    pass

            # Cleanup
            try:
                Path(progress_file).unlink(missing_ok=True)
            except Exception:
                pass
            app.state._ga_process = None

        asyncio.create_task(_monitor_ga())
        return JSONResponse({"ok": True})

    @app.get("/api/ga/status")
    async def ga_status():
        return _ga_state

    @app.post("/api/ga/stop")
    async def ga_stop(request: Request):
        if err := _require_trader(request): return err
        proc = getattr(app.state, "_ga_process", None)
        if proc and proc.poll() is None:
            proc.terminate()
            _ga_state["stopped"] = True
            _ga_state["phase"] = "stopping"
        return JSONResponse({"ok": True})

    # ── Walk-Forward endpoints ───────────────────────────────────

    @app.post("/api/ga/walkforward")
    async def ga_walkforward(request: Request):
        if err := _require_trader(request): return err
        body = await _request_payload(request)
        engine = getattr(app.state, "backtest_engine", None)
        loader = getattr(app.state, "strategy_loader", None)
        if not engine or not loader:
            return JSONResponse({"error": "Engine or loader not initialized"}, status_code=500)

        # ── Symbols: user-chosen, validated against the exchange universe ──
        symbols, symbol_error = await resolve_ga_symbols(app, config, body)
        if symbol_error:
            return JSONResponse({"error": symbol_error}, status_code=400)

        date_start = body.get("date_start", "2025-06-01")
        date_end = body.get("date_end", "2026-06-01")
        train_months = _int_param(body, "train_months", 6, minimum=1)
        val_months = _int_param(body, "val_months", 1, minimum=1)
        step_months = _int_param(body, "step_months", 1, minimum=1)
        pop_size = _int_param(body, "population_size", 80, maximum=120)
        generations = _int_param(body, "generations", 20, maximum=50)
        max_workers = _int_param(body, "max_workers", 1, minimum=1, maximum=16)  # clamp 1-16
        resume = _bool_param(body, "resume", False)

        # Apply cost model overrides
        cost_enabled = _bool_param(body, "cost_enabled", True)
        taker_fee_pct = _float_param(body, "taker_fee_pct", 0.04)
        spread_pct = _spread_param(body, {})
        bt_config = getattr(app.state, "config", None)
        if bt_config:
            bt_config.backtest_cost_enabled = cost_enabled
            bt_config.backtest_taker_fee_pct = taker_fee_pct
            if spread_pct:
                # Only clobber the shipped override table when the caller
                # actually supplied per-symbol values for this run.
                bt_config.backtest_spread_pct = spread_pct

        # ── Write job file for subprocess worker ──
        import uuid
        import subprocess
        data_dir_path = str(Path(loader.strategies_dir).parent) if hasattr(loader, 'strategies_dir') else "data"
        jobs_dir = Path(data_dir_path) / "data" / "ga_jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job_id = str(uuid.uuid4())[:8]
        job_file = str(jobs_dir / f"wf_{job_id}.json")

        job_data = {
            "population_size": pop_size, "generations": generations,
            "symbols": symbols, "date_start": date_start, "date_end": date_end,
            "train_months": train_months, "val_months": val_months,
            "step_months": step_months,
            "max_workers": max_workers,
            "cost_enabled": cost_enabled,
            "taker_fee_pct": taker_fee_pct,
            "spread_pct": spread_pct,
            "resume": resume,
        }
        with open(job_file, "w") as f:
            json.dump(job_data, f)

        # ── Kill previous worker if running ──
        old_proc = getattr(app.state, "_ga_process", None)
        if old_proc and old_proc.poll() is None:
            old_proc.terminate()
            try:
                old_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                old_proc.kill()
        old_wf = getattr(app.state, "_wf_process", None)
        if old_wf and old_wf.poll() is None:
            old_wf.terminate()
            try:
                old_wf.wait(timeout=5)
            except subprocess.TimeoutExpired:
                old_wf.kill()

        # ── Spawn worker subprocess ──
        worker_script = str(Path(__file__).parent.parent.parent / "scripts" / "ga_worker.py")
        t0 = time.time()

        _wf_state.update({
            "running": True, "current_window": 0, "total_windows": 0,
            "completed": [], "report": None, "error": None,
            "started": t0, "elapsed_seconds": 0, "phase": "init",
            "job_file": job_file,
            "params": {
                "date_start": date_start, "date_end": date_end,
                "train_months": train_months, "val_months": val_months,
                "step_months": step_months,
                "population_size": pop_size, "generations": generations,
                "max_workers": max_workers,
                "symbols": symbols,
                "mode": "walkforward",
            },
        })

        proc = subprocess.Popen(
            [sys.executable, worker_script, "--job-type", "walkforward", "--job-file", job_file],
            stdout=subprocess.DEVNULL, stderr=open(job_file + ".log", "w"),
        )
        app.state._wf_process = proc

        # ── Background monitor ──
        async def _monitor_wf():
            progress_file = job_file + ".progress"
            result_file = job_file + ".result"
            while _wf_state["running"]:
                await asyncio.sleep(1)
                _wf_state["elapsed_seconds"] = time.time() - t0

                poll_result = proc.poll()
                if poll_result is not None:
                    try:
                        if Path(result_file).exists():
                            with open(result_file) as f:
                                result = json.load(f)
                            _wf_state["running"] = False
                            _wf_state["elapsed_seconds"] = time.time() - t0
                            if "error" in result:
                                _wf_state["error"] = result["error"]
                                _wf_state["phase"] = "error"
                            elif result.get("type") == "walkforward":
                                _wf_state["report"] = result["report"]
                                _wf_state["phase"] = "complete"
                        else:
                            _wf_state["running"] = False
                            _wf_state["error"] = f"Worker exited with code {poll_result}"
                            _wf_state["phase"] = "error"
                    except Exception as e:
                        _wf_state["running"] = False
                        _wf_state["error"] = str(e)
                    break

                # Read progress
                try:
                    if Path(progress_file).exists():
                        with open(progress_file) as f:
                            progress = json.load(f)
                        _wf_state["phase"] = progress.get("phase", "running")
                        _wf_state["current_window"] = progress.get("current_window", 0)
                        _wf_state["total_windows"] = progress.get("total_windows", 0)
                except Exception:
                    pass

            try:
                Path(progress_file).unlink(missing_ok=True)
            except Exception:
                pass
            app.state._wf_process = None

        asyncio.create_task(_monitor_wf())
        return JSONResponse({"ok": True})

    @app.get("/api/ga/wf_status")
    async def ga_wf_status(request: Request):
        return _wf_state

    @app.get("/partials/ga-panel")
    async def partial_ga_panel(request: Request):
        return _render("partials/ga_panel.html", {
            "request": request, "state": _ga_state,
            # Symbol picker: default = persisted watchlist, cap enforced in the
            # UI exactly like the endpoint enforces it server-side.
            "ga_default_symbols": await default_ga_symbols(config),
            "ga_max_symbols": MAX_GA_SYMBOLS,
        })
