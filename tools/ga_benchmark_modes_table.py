"""Print the four ``ga.benchmark_mode`` benchmarks side by side — real numbers.

The publication gate compares a champion against ONE benchmark
(``core/ga/benchmark.py``).  This tool runs the SAME strategy over the SAME
window once per mode and prints what each mode would have measured and how the
gate would have decided, so the operator can see the difference the choice
makes before / after changing ``ga.benchmark_mode``.

It is **read-only**: the champion YAML is loaded with ``StrategyLoader`` and
never written back (the file's ``enabled:`` flag is irrelevant here — the
backtest is a replay), no GA run is started, no database is touched.  The
optional freshly generated candidate is built in memory
(``random_chromosome`` → ``chromosome_to_strategy``) and is never saved.

Usage::

    python tools/ga_benchmark_modes_table.py                 # newest champion
    python tools/ga_benchmark_modes_table.py --strategy ga_champion_1790844776
    python tools/ga_benchmark_modes_table.py --no-candidate --date-start 2026-06-01 \\
        --date-end 2026-07-01

Every number is produced by the real engine + the real scorer
(``stats_from_engine_result`` → ``score_stats``) and the real gate
(``GAStrategyEvolver._publication_decision``), so the table is the production
pipeline restricted to one strategy per run.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ga.benchmark import BENCHMARK_MODES          # noqa: E402


def _load_config(data_dir: str | None):
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    if data_dir:
        config.data_dir = str(data_dir)
    # The GA evaluation contract: legacy engine, no ML, no live order book.
    config.backtest_engine_mode = "legacy"
    config.backtest_ml_enabled = False
    config.backtest_live_spread_enabled = False
    return config


def _engine_stack(config):
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager

    bus = EventBus()
    return BacktestEngine(config, None, RiskManager(config, bus),
                          OrderExecutor(config, bus))


def _newest_champion() -> Path:
    files = sorted((ROOT / "strategies").glob("ga_champion_*.yaml"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit("no strategies/ga_champion_*.yaml to compare")
    return files[-1]


def _champion_setup(name: str | None):
    """``(label, StrategyConfig, symbols, window, validation, n_trials)`` — read-only."""
    import yaml
    from core.strategy.loader import StrategyConfig

    path = (ROOT / "strategies" / f"{name}.yaml") if name else _newest_champion()
    if not path.exists():
        raise SystemExit(f"strategy file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    provenance = raw.get("provenance") or {}
    window = provenance.get("window") or {}
    config = StrategyConfig(**{k: v for k, v in raw.items() if k != "provenance"})
    symbols = list(raw.get("symbols") or provenance.get("symbols") or [])
    setup = {
        "label": f"champion {path.stem}",
        "path": str(path),
        "config": config,
        "symbols": symbols,
        "start": window.get("train_start"),
        "end": window.get("train_end"),
        "validation": provenance.get("validation"),
        "n_trials": int(provenance.get("n_trials") or 1),
        "prior_trials": int(provenance.get("prior_trials") or 0),
    }
    return setup


def _candidate_setup(seed: int, symbols, start, end, timeframe_pool=None):
    from core.ga.genome import chromosome_to_strategy, random_chromosome

    rng_state = random.getstate()
    random.seed(seed)
    chrom = random_chromosome(f"benchmark_candidate_{seed}",
                              timeframe_pool=timeframe_pool or None)
    random.setstate(rng_state)
    return {
        "label": f"fresh candidate (seed {seed})",
        "path": None,
        "config": chromosome_to_strategy(chrom),
        "symbols": list(symbols),
        "start": start, "end": end,
        "validation": None, "n_trials": 1, "prior_trials": 0,
    }


def _finite(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out and abs(out) != float("inf") else default


def _train_result(score: dict, name: str) -> dict:
    """The result dict shape ``evaluate_population_batch`` hands to the gate."""
    from core.ga.fitness import benchmark_result_fields

    return {
        "fitness": score["fitness"],
        "fitness_base": score.get("fitness_base"),
        "fitness_alpha": score.get("fitness_alpha"),
        "sharpe": round(_finite(score["sharpe"]), 4),
        "win_rate": round(_finite(score["win_rate"]), 2),
        "profit_factor": round(_finite(score["profit_factor"]), 4),
        "raw_profit_factor": round(_finite(score["raw_profit_factor"]), 4),
        "max_dd": round(_finite(score["max_dd"]), 2),
        "total_return": round(_finite(score["total_return_pct"]), 2),
        "buy_hold_pct": score["buy_hold_pct"],
        "alpha_vs_buy_hold_pct": round(_finite(score["alpha_vs_buy_hold_pct"]), 4),
        "dsr": round(_finite(score["deflated_sharpe"]), 4),
        "dsr_detail": score["dsr_detail"],
        "observations": int(score["observations"]),
        "trade_count": int(score["trades"]),
        "long_trades": int(score["long_trades"]),
        "short_trades": int(score["short_trades"]),
        "flag": score.get("flag", ""),
        "strategy_name": name,
        **benchmark_result_fields(score),
    }


def _run_one(engine, loader, setup, mode, config, initial_balance=10000.0):
    from core.ga.fitness import score_stats, stats_from_engine_result
    from core.strategy.loader import StrategyConfig

    strategy = StrategyConfig(**{**setup["config"].model_dump(), "name": "bm_probe"})
    started = time.time()
    result = engine.run_with_exit_evaluation(
        strategies=[strategy], symbols=setup["symbols"],
        date_start=setup["start"], date_end=setup["end"],
        initial_balance=initial_balance, mode="full", simulate_ai_weights=False,
        ml_engine="lightgbm", per_strategy_isolation=True, per_genome_ledger=True,
        use_live_spread=False, benchmark_mode=mode)
    if "error" in result:
        return {"mode": mode, "error": result["error"], "elapsed_s": time.time() - started}
    stats = stats_from_engine_result(result, strategy.name, initial_balance)
    score = score_stats(stats, None, n_trials=setup["n_trials"],
                        prior_trials=setup["prior_trials"],
                        alpha_weight=getattr(config, "ga_alpha_weight", None))
    train = _train_result(score, strategy.name)

    # The real gate, on the real scored result (validation from the champion's
    # own provenance for the champion; none for a fresh candidate).
    from core.ga.evolver import GAStrategyEvolver, GARunConfig

    evolver = GAStrategyEvolver(engine, loader, GARunConfig(population_size=1))
    published, reasons = evolver._publication_decision(train, setup["validation"])
    return {
        "mode": mode,
        "benchmark_pct": train["benchmark_pct"],
        "buy_hold_pct": train["buy_hold_pct"],
        "total_return": train["total_return"],
        "alpha_vs_benchmark_pct": train["alpha_vs_benchmark_pct"],
        "alpha_vs_buy_hold_pct": train["alpha_vs_buy_hold_pct"],
        "trade_count": train["trade_count"],
        "published": published,
        "rejection_reasons": reasons,
        "benchmark_sharpe": train["benchmark_sharpe"],
        "benchmark_max_dd": train["benchmark_max_dd"],
        "benchmark_time_in_market_pct": train["benchmark_time_in_market_pct"],
        "strategy_time_in_market_pct": train["strategy_time_in_market_pct"],
        "information_ratio": train["information_ratio"],
        "jensen_alpha_annual_pct": train["jensen_alpha_annual_pct"],
        "benchmark_beta": train["benchmark_beta"],
        "net_edge_per_trade": train["net_edge_per_trade"],
        "net_edge_per_trade_pct": train["net_edge_per_trade_pct"],
        "weighting": (train.get("benchmark") or {}).get("weighting"),
        "elapsed_s": round(time.time() - started, 1),
    }


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_table(setup, rows):
    print()
    print("=" * 118)
    print(f"{setup['label']}  |  {setup['path'] or '(in-memory)'}")
    print(f"symbols={setup['symbols']}  window={setup['start']}~{setup['end']}  "
          f"n_trials={setup['n_trials']}  validation="
          f"{'recorded' if setup['validation'] is not None else 'none'}")
    print("=" * 118)
    header = (f"{'mode':<17}{'benchmark%':>12}{'strat ret%':>12}{'alpha%':>10}"
              f"{'benchSharpe':>12}{'benchMaxDD':>12}{'benchTIM%':>11}"
              f"{'stratTIM%':>11}{'gate':>7}")
    print(header)
    print("-" * 118)
    for row in rows:
        if "error" in row:
            print(f"{row['mode']:<17}  ERROR: {row['error']}")
            continue
        print(f"{row['mode']:<17}{_fmt(row['benchmark_pct']):>12}"
              f"{_fmt(row['total_return']):>12}{_fmt(row['alpha_vs_benchmark_pct']):>10}"
              f"{_fmt(row['benchmark_sharpe']):>12}{_fmt(row['benchmark_max_dd']):>12}"
              f"{_fmt(row['benchmark_time_in_market_pct']):>11}"
              f"{_fmt(row['strategy_time_in_market_pct']):>11}"
              f"{('PASS' if row['published'] else 'FAIL'):>7}")
    print("-" * 118)
    for row in rows:
        if "error" in row:
            continue
        verdict = "published" if row["published"] else "; ".join(row["rejection_reasons"])
        print(f"{row['mode']:<17} trades={row['trade_count']:<5} "
              f"IR={_fmt(row['information_ratio'])} "
              f"jensen_alpha_ann%={_fmt(row['jensen_alpha_annual_pct'])} "
              f"beta={_fmt(row['benchmark_beta'])} "
              f"net_edge/trade={_fmt(row['net_edge_per_trade'])} "
              f"({_fmt(row['net_edge_per_trade_pct'])}%) "
              f"weighting={row['weighting']} [{row['elapsed_s']}s]")
        print(f"{'':<17} gate: {verdict}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default=None,
                        help="champion name/path under strategies/ (default: newest)")
    parser.add_argument("--date-start", default=None)
    parser.add_argument("--date-end", default=None)
    parser.add_argument("--symbols", default=None,
                        help="comma-separated override of the evaluated basket")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-candidate", action="store_true")
    parser.add_argument("--candidate-seed", type=int, default=20261001)
    parser.add_argument("--json", default=None, help="write the rows here as JSON")
    args = parser.parse_args(argv)

    from core.strategy.loader import StrategyLoader

    config = _load_config(args.data_dir)
    engine = _engine_stack(config)
    loader = StrategyLoader(str(ROOT / "strategies"))

    champion = _champion_setup(args.strategy)
    if args.symbols:
        champion["symbols"] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.date_start:
        champion["start"] = args.date_start
    if args.date_end:
        champion["end"] = args.date_end
    if not champion["symbols"] or not champion["start"] or not champion["end"]:
        raise SystemExit("could not determine the basket/window — pass "
                         "--symbols/--date-start/--date-end")

    setups = [champion]
    if not args.no_candidate:
        setups.append(_candidate_setup(args.candidate_seed, champion["symbols"],
                                       champion["start"], champion["end"],
                                       timeframe_pool=None))

    all_rows = {}
    for setup in setups:
        rows = [_run_one(engine, loader, setup, mode, config)
                for mode in BENCHMARK_MODES]
        all_rows[setup["label"]] = rows
        _print_table(setup, rows)

    if args.json:
        Path(args.json).write_text(json.dumps(all_rows, indent=2, default=str),
                                   encoding="utf-8")
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
