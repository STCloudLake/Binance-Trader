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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def update_progress(job_file: str, data: dict):
    """Write progress atomically."""
    tmp = job_file + ".progress.tmp"
    final = job_file + ".progress"
    with open(tmp, "w") as f:
        json.dump(data, f)
    Path(tmp).replace(final)


def write_result(job_file: str, data: dict):
    """Write final result atomically."""
    tmp = job_file + ".result.tmp"
    final = job_file + ".result"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    Path(tmp).replace(final)


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

    config = Config.load("sim")
    config.backtest_cost_enabled = job.get("cost_enabled", True)
    config.backtest_taker_fee_pct = job.get("taker_fee_pct", 0.04)
    config.backtest_spread_pct = job.get("spread_pct", {})
    config.backtest_engine_mode = "legacy"  # subprocess doesn't have full engine stack

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
        max_workers=3,
    )

    evolver = GAStrategyEvolver(engine, loader, ga_cfg)

    def on_progress(info):
        if isinstance(info, dict):
            update_progress(job_file, {
                "phase": info.get("phase", "evolving"),
                "eval_completed": info.get("eval_completed", 0),
                "eval_total": info.get("eval_total", 0),
            })
        else:
            gen, total, gen_info = info
            update_progress(job_file, {
                "phase": "gen_complete",
                "generation": gen,
                "total_generations": total,
                "best_fitness": gen_info.get("best_fitness", 0),
                "best_sharpe": gen_info.get("best_sharpe", 0),
                "best_win_rate": gen_info.get("best_win_rate", 0),
                "best_trades": gen_info.get("best_trades", 0),
            })

    evolver.set_progress_callback(on_progress)

    symbols = job.get("symbols", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])
    date_start = job.get("date_start", "2025-06-01")
    date_end = job.get("date_end", "2026-06-01")
    validation_start = job.get("validation_start") or None
    seed_strategies = job.get("seed_strategies", [])

    result = evolver.evolve(
        symbols, date_start, date_end,
        seed_strategies=seed_strategies,
        validation_start=validation_start,
    )
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

    config = Config.load("sim")
    config.backtest_cost_enabled = job.get("cost_enabled", True)
    config.backtest_taker_fee_pct = job.get("taker_fee_pct", 0.04)
    config.backtest_spread_pct = job.get("spread_pct", {})
    config.backtest_engine_mode = "legacy"  # subprocess doesn't have full engine stack

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
        max_workers=3,
    )

    wf_cfg = WFConfig(
        enabled=True,
        train_months=job.get("train_months", 6),
        val_months=job.get("val_months", 1),
        step_months=job.get("step_months", 1),
    )

    symbols = job.get("symbols", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])
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
                    update_progress(job_file, {
                        "phase": "running",
                        "total_windows": state.get("total_windows", len(windows)),
                        "current_window": state.get("current_window", 0),
                    })
            except Exception:
                pass

    _progress_thread = _threading.Thread(target=_report_loop, daemon=True)
    _progress_thread.start()

    try:
        report = runner.run(symbols, date_start, date_end, wf_cfg, ga_cfg)
    finally:
        _progress_stop = True
        _progress_thread.join(timeout=5)

    write_result(job_file, {
        "type": "walkforward",
        "report": report.to_dict(),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-type", choices=["ga", "walkforward"], required=True)
    parser.add_argument("--job-file", required=True)
    args = parser.parse_args()

    try:
        with open(args.job_file) as f:
            job = json.load(f)

        update_progress(args.job_file, {"phase": "starting", "job_type": args.job_type})

        if args.job_type == "ga":
            run_ga(job, args.job_file)
        else:
            run_walkforward(job, args.job_file)

    except Exception as e:
        write_result(args.job_file, {"error": str(e), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
