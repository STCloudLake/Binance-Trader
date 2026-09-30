"""Synthetic, deterministic BEFORE/AFTER measurement for P1 items 1 and 3.

Runs on a generated market (no cached data, no network, ~1 minute) and prints
JSON. Works on the pre-P1 code as well: the newer keyword arguments are probed.

    python tools/p1_synth_measure.py --label after
    python tools/p1_synth_measure.py --label before      # from a HEAD worktree
"""
import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WINDOW = ("2026-02-01", "2026-03-15")


def build_market(root: Path, symbols=("BTCUSDT", "ETHUSDT"), days=120):
    rng = np.random.default_rng(20261001)
    base = pd.date_range("2026-01-01", periods=days * 96, freq="15min")
    for symbol in symbols:
        close = 20000 + np.cumsum(rng.normal(0, 40, len(base)))
        m15 = pd.DataFrame({"open": close, "high": close + 30, "low": close - 30,
                            "close": close, "volume": rng.random(len(base)) * 100 + 10},
                           index=base)
        market_dir = root / "market" / symbol
        market_dir.mkdir(parents=True, exist_ok=True)
        for tf in ("15m", "1h", "4h"):
            frame = m15 if tf == "15m" else m15.resample(tf).agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}).dropna()
            frame.to_parquet(market_dir / f"{tf}.parquet")
    return root


def build_engine(data_dir: Path, strategies_dir: Path):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import StrategyLoader

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = str(data_dir)
    cfg.backtest_engine_mode = "legacy"
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    cfg.backtest_cost_enabled = True
    bus = EventBus()
    loader = StrategyLoader(str(strategies_dir))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus), OrderExecutor(cfg, bus))
    return cfg, engine, loader


def measure_population(engine, loader, n=20, seed=1234):
    from core.ga.fitness import evaluate_population_batch
    from core.ga.genome import random_chromosome

    random.seed(seed)
    population = [random_chromosome(f"exp_{i}") for i in range(n)]
    kwargs = dict(population=population, symbols=["BTCUSDT", "ETHUSDT"],
                  date_start=WINDOW[0], date_end=WINDOW[1],
                  engine=engine, loader=loader, max_workers=1)
    t0 = time.time()
    try:
        evaluate_population_batch(**kwargs, use_live_spread=False,
                                  batch_trials=n, prior_trials=0)
        variant = "with use_live_spread"
    except TypeError:
        evaluate_population_batch(**kwargs)
        variant = "legacy signature"
    elapsed = round(time.time() - t0, 1)
    per_genome = []
    for chrom in population:
        r = chrom.get("fitness_result", {}) or {}
        per_genome.append({"trades": r.get("trade_count"),
                           "fitness": r.get("fitness"),
                           "sharpe": r.get("sharpe"),
                           "max_dd": r.get("max_dd"),
                           "flag": r.get("flag", "")})
    return {
        "genomes": n, "seconds": elapsed, "batch_kwargs": variant,
        "window": WINDOW,
        "trading": sum(1 for g in per_genome if (g["trades"] or 0) > 0),
        "zero_trade": sum(1 for g in per_genome if (g["trades"] or 0) == 0),
        "max_trades": max((g["trades"] or 0) for g in per_genome),
        "total_trades": sum((g["trades"] or 0) for g in per_genome),
        "max_fitness": max((g["fitness"] if g["fitness"] is not None else -999)
                           for g in per_genome),
        "per_genome": per_genome,
    }


def legacy_formula(win_rate, profit_factor, pnl, initial, trades, longs, shorts,
                   n_conditions=1, n_indicators=1, n_params=1):
    pf = min(profit_factor, 100.0) if profit_factor > 0 else (100.0 if pnl > 0 else 0.1)
    roc = pnl / max(initial, 1)
    imbalance = abs(longs / trades - 0.5) * 2 if trades else 1.0
    fitness = win_rate * 0.15 + max(pf, 0.1) * 5.0 + roc * 50 - imbalance * 10.0
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


def measure_champion(engine, loader, data_dir, population=6, generations=2,
                     seed=20260101):
    """End-to-end GA run → champion YAML text (publication gate + provenance)."""
    import yaml
    from core.ga.evolver import GAStrategyEvolver, GARunConfig

    cfg = GARunConfig(population_size=population, generations=generations,
                      elite_count=2, immigrant_count=2, max_workers=1, seed=seed)
    t0 = time.time()
    evolver = GAStrategyEvolver(engine, loader, cfg)
    evolver._init_population = _one_hour(evolver._init_population)
    result = evolver.evolve(["BTCUSDT", "ETHUSDT"], WINDOW[0], WINDOW[1],
                            seed=seed, window_key="synth")
    name = result.get("champion_name")
    yaml_path = loader.strategies_dir / f"{name}.yaml" if name else None
    return {
        "seed": seed,
        "seconds": round(time.time() - t0, 1),
        "champion": name,
        "fitness": result.get("fitness"),
        "trade_count": result.get("trade_count"),
        "published": result.get("published"),
        "rejection_reasons": result.get("rejection_reasons"),
        "dsr": result.get("dsr"),
        "history": [{k: h.get(k) for k in ("generation", "best_fitness",
                                           "best_trades")}
                    for h in result.get("history", [])],
        "yaml_text": yaml_path.read_text(encoding="utf-8") if yaml_path and yaml_path.exists() else None,
    }


def _one_hour(original_init):
    """Pin the timeframe gene to 1h so a synthetic run stays fast."""
    def _init(seed_strategies):
        pop = original_init(seed_strategies)
        for chrom in pop:
            for gene in chrom["categorical"]:
                if gene.name == "timeframes":
                    gene.value = "1h"
        return pop
    return _init


def _verify_isolated_zero_trade_genomes(engine, loader, seed=1234, n=20,
                                        limit=3):
    """Evaluate the zero-trade genomes ALONE: they must still be inert.

    This is what separates "the genome genuinely does not trade" from the old
    starvation bug (where a genome only traded when nobody else did).
    """
    from core.ga.fitness import evaluate_population_batch
    from core.ga.genome import chromosome_to_strategy, random_chromosome

    random.seed(seed)
    population = [random_chromosome(f"exp_{i}") for i in range(n)]
    evaluate_population_batch(population, ["BTCUSDT", "ETHUSDT"], WINDOW[0],
                              WINDOW[1], engine, loader, max_workers=1,
                              use_live_spread=False, batch_trials=n,
                              prior_trials=0)
    zero = [i for i, c in enumerate(population)
            if (c.get("fitness_result", {}) or {}).get("trade_count") == 0]
    out = []
    for i in zero[:limit]:
        s_cfg = chromosome_to_strategy(population[i])
        s_cfg.name = f"alone_{i}"
        res = engine.run_with_exit_evaluation(
            strategies=[s_cfg], symbols=["BTCUSDT", "ETHUSDT"],
            date_start=WINDOW[0], date_end=WINDOW[1], mode="full",
            simulate_ai_weights=False, per_strategy_isolation=True,
            per_genome_ledger=True, use_live_spread=False)
        out.append({"genome": i, "trades_in_chunk": 0,
                    "trades_alone": len(res.get("trades", []))})
    return {"zero_trade_genomes": zero, "checked": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--genomes", type=int, default=20)
    parser.add_argument("--out", default="")
    parser.add_argument("--mode", default="population",
                        choices=["population", "champion", "verify"])
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="p1_synth_"))
    data_dir = build_market(workdir / "data")
    _, engine, loader = build_engine(data_dir, workdir / "strategies")

    result = {"label": args.label, "mode": args.mode}
    if args.mode == "population":
        result["population"] = measure_population(engine, loader, args.genomes)
    elif args.mode == "verify":
        result["isolation_verify"] = _verify_isolated_zero_trade_genomes(
            engine, loader, n=args.genomes)
    else:
        _, engine2, loader2 = build_engine(data_dir, workdir / "strategies2")
        result["champion_run"] = measure_champion(engine2, loader2, data_dir)
    result["legacy_formula_ranking"] = {
        "A_5_winning_trades": legacy_formula(100.0, 100.0, 500.0, 10000.0, 5, 5, 0),
        "B_200_trades": legacy_formula(60.0, 2.0, 1500.0, 10000.0, 200, 100, 100),
    }

    text = json.dumps(result, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
