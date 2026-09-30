"""Focused, self-timed P1 measurement runs (works on the OLD code too).

    python tools/p1_measure.py --label after --data-dir . --mode population20
    python tools/p1_measure.py --label before --data-dir . --mode generations
    python tools/p1_measure.py --label after  --data-dir . --mode all

Prints JSON so the numbers can be pasted into the report verbatim.
"""
import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EARLY = ["2026-05-01", "2026-06-01"]
WIDE = ["2025-08-01", "2026-05-01"]


def _workdir(data_root: Path) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="p1_measure_"))
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    try:
        (workdir / "data" / "market").symlink_to(
            data_root / "data" / "market", target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copytree(data_root / "data" / "market", workdir / "data" / "market")
    return workdir


def _engine(workdir: Path):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import StrategyLoader

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = str(workdir / "data")
    cfg.backtest_engine_mode = "legacy"
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    cfg.backtest_cost_enabled = True
    bus = EventBus()
    loader = StrategyLoader(str(workdir / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus), OrderExecutor(cfg, bus))
    return cfg, engine, loader


def _population(engine, loader, n, window, seed=1234):
    """Item 1: 20-genome chunk — trades per genome."""
    from core.ga.fitness import evaluate_population_batch
    from core.ga.genome import random_chromosome

    random.seed(seed)
    population = [random_chromosome(f"exp_{i}") for i in range(n)]
    kwargs = dict(population=population, symbols=["BTCUSDT", "ETHUSDT"],
                  date_start=window[0], date_end=window[1],
                  engine=engine, loader=loader, max_workers=1)
    t0 = time.time()
    try:
        evaluate_population_batch(**kwargs, use_live_spread=False,
                                  batch_trials=n, prior_trials=0)
    except TypeError:
        evaluate_population_batch(**kwargs)
    elapsed = round(time.time() - t0, 1)
    trades = []
    for chrom in population:
        r = chrom.get("fitness_result", {}) or {}
        trades.append({"trades": r.get("trade_count"), "fitness": r.get("fitness"),
                       "sharpe": r.get("sharpe"), "max_dd": r.get("max_dd"),
                       "flag": r.get("flag", "")})
    return {"window": window, "genomes": n, "seconds": elapsed,
            "trading": sum(1 for t in trades if (t["trades"] or 0) > 0),
            "zero_trade": sum(1 for t in trades if (t["trades"] or 0) == 0),
            "max_trades": max((t["trades"] or 0) for t in trades),
            "total_trades": sum((t["trades"] or 0) for t in trades),
            "per_genome": trades}


def _generations(engine, loader, window, population=6, generations=3, seed=99):
    """Item 1/3: best-fitness progression across generations."""
    from core.ga.evolver import GAStrategyEvolver, GARunConfig

    cfg = GARunConfig(population_size=population, generations=generations,
                      elite_count=2, immigrant_count=2, max_workers=1, seed=seed)
    t0 = time.time()
    evolver = GAStrategyEvolver(engine, loader, cfg)
    result = evolver.evolve(["BTCUSDT", "ETHUSDT"], window[0], window[1],
                            seed=seed, window_key="measure")
    return {
        "window": window, "population": population, "generations": generations,
        "seconds": round(time.time() - t0, 1),
        "history": [{k: h.get(k) for k in ("generation", "best_fitness",
                                           "best_trades", "best_sharpe")}
                    for h in result.get("history", [])],
        "champion": result.get("champion_name"),
        "fitness": result.get("fitness"),
        "trade_count": result.get("trade_count"),
        "published": result.get("published"),
        "rejection_reasons": result.get("rejection_reasons"),
        "validation": result.get("validation"),
        "dsr": result.get("dsr"),
        "provenance": result.get("provenance"),
    }


def _determinism(engine, loader, window, population=6, generations=2, seed=4242):
    """Item 9: same seed twice → identical generation history."""
    from core.ga.evolver import GAStrategyEvolver, GARunConfig

    runs = []
    for _ in range(2):
        cfg = GARunConfig(population_size=population, generations=generations,
                          elite_count=2, immigrant_count=2, max_workers=1,
                          seed=seed)
        evolver = GAStrategyEvolver(engine, loader, cfg)
        result = evolver.evolve(["BTCUSDT"], window[0], window[1], seed=seed,
                                window_key="determinism")
        runs.append([{k: h.get(k) for k in ("generation", "best_fitness",
                                            "best_trades")}
                     for h in result.get("history", [])])
    return {"identical": runs[0] == runs[1], "run_a": runs[0], "run_b": runs[1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mode", default="all",
                        choices=["all", "population20", "generations",
                                 "determinism", "synthetic"])
    args = parser.parse_args()

    data_root = Path(args.data_dir).resolve()
    workdir = _workdir(data_root)
    result = {"label": args.label, "mode": args.mode}
    if args.mode in ("synthetic", "all"):
        from core.ga.fitness import deflated_sharpe_ratio, score_stats, stats_from_trades

        def equity_for(pnls, initial=10000.0):
            total = float(sum(pnls))
            return [{"time": "2026-01-01", "equity": initial},
                    {"time": "2026-03-01", "equity": initial + total * 0.5},
                    {"time": "2026-06-01", "equity": initial + total}]

        out = {}
        for name, pnls in (("A_5_winning_trades", [100.0] * 5),
                           ("B_200_trades", [30.0] * 120 + [-18.0] * 80)):
            stats = stats_from_trades([{"pnl": p, "side": "long"} for p in pnls],
                                      equity_for(pnls), 10000.0)
            stats["buy_hold_pct"] = 0.0
            scored = score_stats(stats, None, n_trials=100)
            out[name] = {"fitness": scored["fitness"],
                         "raw_pf": round(stats["raw_profit_factor"], 3),
                         "pf_term": round(scored["profit_factor"], 3),
                         "trades": scored["trades"], "flag": scored["flag"]}
        result["fitness_ranking"] = out
        result["dsr"] = {f"T={t}": deflated_sharpe_ratio(1.2, 1200,
                                                         observation_periods=t)
                         for t in (365, 1200)}

    if args.mode == "synthetic":
        print(json.dumps(result, indent=2, default=str))
        return

    _, engine, loader = _engine(workdir)

    if args.mode in ("population20", "all"):
        result["population20"] = _population(engine, loader, 20, EARLY)
    if args.mode in ("generations", "all"):
        result["generations"] = _generations(engine, loader, EARLY)
    if args.mode in ("determinism", "all"):
        result["determinism"] = _determinism(engine, loader, EARLY)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
