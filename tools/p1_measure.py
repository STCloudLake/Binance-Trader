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

EARLY = ["2026-05-01", "2026-05-21"]
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


def _generations(engine, loader, window, population=6, generations=3, seed=99,
                 timeframe="1h"):
    """Item 1/3: best-fitness progression across generations."""
    from core.ga.evolver import GAStrategyEvolver, GARunConfig

    cfg = GARunConfig(population_size=population, generations=generations,
                      elite_count=2, immigrant_count=2, max_workers=1, seed=seed)
    t0 = time.time()
    evolver = GAStrategyEvolver(engine, loader, cfg)

    if timeframe:
        # Pin the timeframe gene so the run is bounded: a 1m genome needs ~44k
        # bars of 1-minute data and dominates the wall clock without adding
        # anything to the measurement.
        original_init = evolver._init_population

        def _init(seed_strategies):
            pop = original_init(seed_strategies)
            for chrom in pop:
                for gene in chrom["categorical"]:
                    if gene.name == "timeframes":
                        gene.value = timeframe
            return pop

        evolver._init_population = _init

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


def _legacy_ranking():
    """The PRE-P1 batch fitness formula (kept here so the report can show the
    measured before/after on the identical scenarios)."""
    def legacy(win_rate, profit_factor, pnl, initial_balance, trades,
               long_trades, short_trades, n_conditions=1, n_indicators=1,
               n_params=1):
        if profit_factor != float("inf") and profit_factor > 0:
            pf = min(profit_factor, 100.0)
        elif profit_factor > 0:
            pf = 100.0
        else:
            pf = 100.0 if pnl > 0 else 0.1
        roc = pnl / max(initial_balance, 1)
        imbalance = abs(long_trades / trades - 0.5) * 2 if trades else 1.0
        fitness = (win_rate * 0.15 + max(pf, 0.1) * 5.0 + roc * 50
                   - imbalance * 10.0)
        if trades < 5:
            fitness -= 20
        elif trades < 15:
            fitness -= 5
        elif trades > 500:
            fitness -= (trades - 500) * 0.02
        if pnl < -50:
            fitness -= abs(pnl) * 0.3
        fitness -= n_conditions * 0.8 + n_indicators * 1.2 + n_params * 0.3
        return round(fitness, 4)

    return {
        # 5 winning trades, +500 USDT, no losing trades → PF = 100 in the old code
        "A_5_winning_trades": legacy(100.0, 100.0, 500.0, 10000.0, 5, 5, 0),
        # 200 trades, 60% win, PF 2.0, +1500 USDT
        "B_200_trades": legacy(60.0, 2.0, 1500.0, 10000.0, 200, 100, 100),
    }


def _emit(result: dict, out: str = "") -> None:
    text = json.dumps(result, indent=2, default=str)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mode", default="all",
                        choices=["all", "population20", "generations",
                                 "determinism", "synthetic", "legacy"])
    parser.add_argument("--genomes", type=int, default=12)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.mode == "legacy":
        print(json.dumps({"label": args.label,
                          "legacy_formula_ranking": _legacy_ranking()}, indent=2))
        return

    data_root = Path(args.data_dir).resolve()
    workdir = _workdir(data_root)
    result = {"label": args.label, "mode": args.mode}
    result["legacy_formula_ranking"] = _legacy_ranking()
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
        _emit(result, args.out)
        return

    _, engine, loader = _engine(workdir)

    if args.mode in ("population20", "all"):
        result["population20"] = _population(engine, loader, args.genomes, EARLY)
        _emit(result, args.out)
    if args.mode in ("generations", "all"):
        result["generations"] = _generations(engine, loader, EARLY)
        _emit(result, args.out)
    if args.mode in ("determinism", "all"):
        result["determinism"] = _determinism(engine, loader, EARLY)
        _emit(result, args.out)

    _emit(result, args.out)


if __name__ == "__main__":
    main()
