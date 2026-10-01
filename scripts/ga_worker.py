#!/usr/bin/env python3
"""GA/Walk-Forward worker — runs in a subprocess to avoid GIL blocking the main server.

Usage:
    python scripts/ga_worker.py --job-type ga --job-file /path/to/job.json

The job file contains all parameters. Progress is written to job_file.progress,
final results to job_file.result.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


#: Wall-clock start of this worker process (set by :func:`main`, read by the
#: progress reporter so every payload can carry ``started_at``/``elapsed_s``).
_STARTED_AT = time.time()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def update_progress(job_file: str, data: dict, merge: bool = True):
    """Write progress atomically.

    ``merge=True`` (the default) folds ``data`` into whatever is already on disk
    instead of replacing it.  Replacing used to drop the generation/best-so-far
    fields every time an intra-generation tick arrived, so the UI lost the
    generation it had already been told about.
    """
    payload = data
    if merge:
        try:
            with open(job_file + ".progress") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                payload = {**existing, **data}
        except Exception:
            payload = data
    tmp = job_file + ".progress.tmp"
    final = job_file + ".progress"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    Path(tmp).replace(final)


def make_progress_reporter(job_file: str, job_type: str = "ga",
                           clock=time.time):
    """Build the ``on_progress`` callback handed to the evolver.

    The evolver reports two shapes:

    * a **dict** — intra-generation evaluation progress
      (``phase``/``generation``/``eval_completed``/``eval_total``/``elapsed_s``/
      ``eval_equivalent``/``best_*`` from
      ``GAStrategyEvolver._report_progress``);
    * a **tuple** ``(generation, total_generations, gen_info)`` — one generation
      finished, which also gets a per-generation INFO line in the job log.

    Both are merged into the progress file (never replacing it), stamped with
    ``started_at``/``updated_at``/``elapsed_s``, and the result stays plain
    JSON (``json.dump`` would fail on a numpy scalar otherwise).
    """
    state: dict = {
        "phase": "starting",
        "job_type": job_type,
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "elapsed_s": 0.0,
    }
    logged_generation = [0]

    def _publish() -> None:
        state["updated_at"] = _now_iso()
        state["elapsed_s"] = round(clock() - _STARTED_AT, 1)
        update_progress(job_file, dict(state))

    def on_progress(info):
        if isinstance(info, dict):
            state.update({k: v for k, v in info.items() if v is not None})
        else:
            gen, total, gen_info = info
            gen_info = gen_info or {}
            state.update({
                "phase": "gen_complete",
                "generation": int(gen),
                "total_generations": int(total),
                "best_fitness": gen_info.get("best_fitness", 0),
                "avg_fitness": gen_info.get("avg_fitness", 0),
                "best_sharpe": gen_info.get("best_sharpe", 0),
                "best_win_rate": gen_info.get("best_win_rate", 0),
                "best_trades": gen_info.get("best_trades", 0),
                "generation_elapsed_s": round(
                    float(gen_info.get("elapsed", 0) or 0), 1),
            })
            if int(gen) > logged_generation[0]:
                logged_generation[0] = int(gen)
                # One INFO line per generation, so `tail job.log` is informative
                # without the UI (the evolver logs its own per-generation line
                # too; this one carries the job-relative elapsed time).
                logger.info(
                    f"[ga_worker] gen {int(gen)}/{int(total)} complete | "
                    f"best={float(gen_info.get('best_fitness', 0) or 0):.2f} "
                    f"avg={float(gen_info.get('avg_fitness', 0) or 0):.2f} "
                    f"trades={int(gen_info.get('best_trades', 0) or 0)} "
                    f"elapsed={state['elapsed_s']:.0f}s "
                    f"gen_elapsed={state['generation_elapsed_s']:.0f}s")
        _publish()

    on_progress.state = state  # exposed for tests/diagnostics
    return on_progress


def write_result(job_file: str, data: dict):
    """Write final result atomically."""
    tmp = job_file + ".result.tmp"
    final = job_file + ".result"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    Path(tmp).replace(final)


#: Last-resort symbols when a job carries none *and* the watchlist is unreadable.
DEFAULT_FALLBACK_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]


def _watchlist_fallback(config) -> list:
    """The persisted watchlist (same source as ``GET /api/market/watchlist``)."""
    try:
        import asyncio

        from core.market_data.universe import DEFAULT_WATCHLIST, load_watchlist

        db_path = getattr(config, "db_path", "") if config is not None else ""
        symbols = asyncio.run(load_watchlist(db_path, DEFAULT_WATCHLIST))
    except Exception:
        return []
    out = []
    for item in symbols or []:
        symbol = str(item).strip().upper()
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def job_symbols(job: dict, config=None) -> list:
    """Symbols this job must run on — **from the payload**, never a literal list.

    The job file is written by the GA routes after validating the user's
    selection against the exchange universe, so it is authoritative.  A missing
    / empty ``symbols`` key (e.g. an old checkpoint job file) falls back to the
    persisted watchlist and finally to :data:`DEFAULT_FALLBACK_SYMBOLS`.
    """
    raw = job.get("symbols")
    if isinstance(raw, str):
        raw = raw.split(",")
    symbols: list = []
    for item in raw or []:
        symbol = str(item).strip().upper()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    if symbols:
        return symbols
    return _watchlist_fallback(config) or list(DEFAULT_FALLBACK_SYMBOLS)


def _job_seed(job: dict) -> int:
    """Job seed (deterministic when the route supplied one, else job-file seeded)."""
    try:
        seed = int(job.get("seed") or 0)
    except (TypeError, ValueError):
        seed = 0
    if seed:
        return seed
    # Derive from the job file path so a replay of the SAME file is identical.
    import hashlib
    digest = hashlib.sha256(str(job.get("_job_file", "")).encode()).hexdigest()
    return int(digest[:8], 16)


def _seed_everything(seed: int) -> None:
    """Seed Python + NumPy so a job's population and mutation stream repeat."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def run_ga(job: dict, job_file: str):
    """Run standard GA evolution."""
    from app.config import Config
    from core.strategy.engine import StrategyEngine
    from core.strategy.loader import StrategyLoader
    from core.backtest.engine import BacktestEngine
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.risk.manager import RiskManager
    from core.executor.executor import OrderExecutor
    from app.event_bus import EventBus
    from core.backtest.cost_model import clear_live_spread_cache

    config = Config.load("sim")
    config.backtest_cost_enabled = job.get("cost_enabled", True)
    config.backtest_taker_fee_pct = job.get("taker_fee_pct", 0.04)
    # The job's spread map is an OVERRIDE table on top of the config's own
    # overrides; symbols in neither are derived live / by default.
    config.backtest_spread_pct = {
        **(getattr(config, "backtest_spread_pct", None) or {}),
        **(job.get("spread_pct") or {})}
    config.backtest_engine_mode = "legacy"  # subprocess doesn't have full engine stack

    seed = _job_seed(job)
    _seed_everything(seed)
    # Historical fills are never priced from TODAY's order book unless the
    # operator explicitly asks for it (`ga.use_live_spread: true`).
    clear_live_spread_cache()
    live_spread = bool(getattr(config, "ga_use_live_spread", False))
    config.backtest_live_spread_enabled = live_spread

    event_bus = EventBus()
    risk_manager = RiskManager(config, event_bus)
    order_executor = OrderExecutor(config, event_bus)

    loader = StrategyLoader(str(Path(config.data_dir).parent / "strategies"))
    engine = BacktestEngine(config, None, risk_manager, order_executor)

    pop_size = min(job.get("population_size", 60), 120)
    generations = min(job.get("generations", 20), 50)

    ga_cfg = GARunConfig(
        population_size=pop_size,
        generations=generations,
        elite_count=max(4, pop_size // 10),
        immigrant_count=max(4, pop_size // 10),
        max_workers=job.get("max_workers", 1),  # >1 uses multi-process (safe for TA-Lib)
        seed=seed,
    )

    evolver = GAStrategyEvolver(engine, loader, ga_cfg)

    # One reporter fills the progress file for every shape the evolver emits
    # (intra-generation dict / per-generation tuple) and logs one line per
    # generation.  It used to be an inline closure that REPLACED the payload,
    # which lost generation/best fields on every intra-generation tick.
    on_progress = make_progress_reporter(job_file, "ga")

    evolver.set_progress_callback(on_progress)

    # Symbols come from the job payload (validated by the web route); the
    # watchlist is only the fallback for a job file that carries none.
    symbols = job_symbols(job, config)
    date_start = job.get("date_start", "2025-06-01")
    date_end = job.get("date_end", "2026-06-01")
    validation_start = job.get("validation_start") or None
    seed_strategies = job.get("seed_strategies", [])

    result = evolver.evolve(
        symbols, date_start, date_end,
        seed_strategies=seed_strategies,
        validation_start=validation_start,
        # ``resume`` used to be accepted by the route and then dropped here, so
        # "resume" silently started a fresh population.
        resume=bool(job.get("resume", False)),
        seed=seed,
        window_key=f"{date_start}~{validation_start or date_end}",
    )
    result["seed"] = seed
    write_result(job_file, result)


def run_walkforward(job: dict, job_file: str):
    """Run Walk-Forward validation."""
    from app.config import Config
    from core.strategy.engine import StrategyEngine
    from core.strategy.loader import StrategyLoader
    from core.backtest.engine import BacktestEngine
    from core.ga.evolver import GARunConfig
    from core.ga.walkforward import WalkForwardRunner, WFConfig
    from core.risk.manager import RiskManager
    from core.executor.executor import OrderExecutor
    from app.event_bus import EventBus
    from core.backtest.cost_model import clear_live_spread_cache

    config = Config.load("sim")
    config.backtest_cost_enabled = job.get("cost_enabled", True)
    config.backtest_taker_fee_pct = job.get("taker_fee_pct", 0.04)
    # The job's spread map is an OVERRIDE table on top of the config's own
    # overrides; symbols in neither are derived live / by default.
    config.backtest_spread_pct = {
        **(getattr(config, "backtest_spread_pct", None) or {}),
        **(job.get("spread_pct") or {})}
    config.backtest_engine_mode = "legacy"  # subprocess doesn't have full engine stack

    seed = _job_seed(job)
    _seed_everything(seed)
    clear_live_spread_cache()
    config.backtest_live_spread_enabled = bool(
        getattr(config, "ga_use_live_spread", False))

    event_bus = EventBus()
    risk_manager = RiskManager(config, event_bus)
    order_executor = OrderExecutor(config, event_bus)

    loader = StrategyLoader(str(Path(config.data_dir).parent / "strategies"))
    engine = BacktestEngine(config, None, risk_manager, order_executor)

    pop_size = min(job.get("population_size", 60), 120)
    generations = min(job.get("generations", 20), 50)

    ga_cfg = GARunConfig(
        population_size=pop_size,
        generations=generations,
        elite_count=max(4, pop_size // 10),
        immigrant_count=max(4, pop_size // 10),
        max_workers=job.get("max_workers", 1),  # >1 uses multi-process (safe for TA-Lib)
        seed=seed,
    )

    wf_cfg = WFConfig(
        enabled=True,
        train_months=job.get("train_months", 6),
        val_months=job.get("val_months", 1),
        step_months=job.get("step_months", 1),
    )

    # Symbols come from the job payload (validated by the web route); the
    # watchlist is only the fallback for a job file that carries none.
    symbols = job_symbols(job, config)
    date_start = job.get("date_start", "2025-06-01")
    date_end = job.get("date_end", "2026-06-01")

    data_dir = str(Path(loader.strategies_dir).parent)
    runner = WalkForwardRunner(engine, loader, data_dir)

    # Override the runner's run to report window-level progress
    original_compute = runner.compute_windows
    windows = original_compute(date_start, date_end, wf_cfg)
    update_progress(job_file, {
        "phase": "starting",
        "total_windows": len(windows),
        "current_window": 0,
    })

    # ── Background progress reporter: updates current_window from runner state ──
    import threading as _threading
    _progress_stop = False

    def _report_loop():
        while not _progress_stop:
            time.sleep(3)
            try:
                state = runner.get_state()
                if state:
                    data = {
                        "phase": "running",
                        "total_windows": state.get("total_windows", len(windows)),
                        "current_window": state.get("current_window", 0),
                    }
                    # Include GA sub-progress if available
                    ga_phase = state.get("ga_phase", "")
                    ga_gen = state.get("ga_gen", 0)
                    ga_total = state.get("ga_total_gen", 0)
                    if ga_phase:
                        data["ga_phase"] = ga_phase
                    if ga_gen and ga_total:
                        data["ga_gen"] = ga_gen
                        data["ga_total_gen"] = ga_total
                    update_progress(job_file, data)
                else:
                    # State file doesn't exist yet — report at least that we're alive
                    update_progress(job_file, {
                        "phase": "running",
                        "total_windows": len(windows),
                        "current_window": 0,
                    })
            except Exception:
                pass

    _progress_thread = _threading.Thread(target=_report_loop, daemon=True)
    _progress_thread.start()

    try:
        report = runner.run(symbols, date_start, date_end, wf_cfg, ga_cfg,
                            resume=bool(job.get("resume", False)))
    finally:
        _progress_stop = True
        _progress_thread.join(timeout=5)

    write_result(job_file, {
        "type": "walkforward",
        "report": report.to_dict(),
        "seed": seed,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-type", choices=["ga", "walkforward"], required=True)
    parser.add_argument("--job-file", required=True)
    args = parser.parse_args()

    try:
        with open(args.job_file) as f:
            job = json.load(f)
        # Keep the file path on the payload so a replay of the same file derives
        # the same fallback seed.
        job["_job_file"] = args.job_file

        update_progress(args.job_file, {
            "phase": "starting",
            "job_type": args.job_type,
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "elapsed_s": 0.0,
        })

        if args.job_type == "ga":
            run_ga(job, args.job_file)
        else:
            run_walkforward(job, args.job_file)

    except Exception as e:
        write_result(args.job_file, {"error": str(e), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
