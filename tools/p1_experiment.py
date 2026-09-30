"""Reproducible BEFORE/AFTER experiment for ALGO_UPGRADE_PLAN P1 items 1,3,4,5,6.

Usage:
    python tools/p1_experiment.py --label before --data-dir <dir>   # old code worktree
    python tools/p1_experiment.py --label after  --data-dir <dir>   # current tree

Writes JSON to stdout so the numbers can be pasted into the report verbatim.
"""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_engine(config, workdir: Path):
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


def experiment_population(engine, loader, n=12, seed=1234):
    """Item 1: does every genome of a chunk get its own slots and ledger?"""
    from core.ga.fitness import evaluate_population_batch
    from core.ga.genome import random_chromosome

    random.seed(seed)
    population = [random_chromosome(f"exp_{i}") for i in range(n)]
    kwargs = dict(
        population=population, symbols=["BTCUSDT", "ETHUSDT"],
        date_start="2026-05-01", date_end="2026-06-01",
        engine=engine, loader=loader, max_workers=1,
        use_live_spread=False, batch_trials=n, prior_trials=0,
    )
    try:
        evaluate_population_batch(**kwargs)
    except TypeError:
        kwargs.pop("use_live_spread", None)
        kwargs.pop("batch_trials", None)
        kwargs.pop("prior_trials", None)
        evaluate_population_batch(**kwargs)
    out = []
    for i, chrom in enumerate(population):
        r = chrom.get("fitness_result", {}) or {}
        out.append({
            "genome": i,
            "trades": r.get("trade_count"),
            "fitness": r.get("fitness"),
            "sharpe": r.get("sharpe"),
            "max_dd": r.get("max_dd"),
            "flag": r.get("flag", ""),
        })
    return out


def experiment_fitness_ranking():
    """Item 3: '5 winning trades' vs '200 trades / 60% / PF 2.0 / +1500 USDT'."""
    from core.ga.fitness import score_stats

    # A: 5 winning trades, each +100 on 10k.  B: 200 trades, 60% win, PF 2.0, +1500.
    a_trades = [{"pnl": 100.0, "side": "long"}] * 5
    b_wins = [{"pnl": 30.0, "side": "long"}] * 120
    b_losses = [{"pnl": -18.0, "side": "short"}] * 80

    def equity(trades, final):
        return [{"time": "2025-07-01", "equity": 10000.0},
                {"time": "2025-12-01", "equity": 10000.0 + sum(t["pnl"] for t in trades) / 2},
                {"time": "2026-06-01", "equity": 10000.0 + final}]

    from core.ga.fitness import stats_from_trades, score_stats
    stats_a = stats_from_trades(a_trades, equity(a_trades, 500.0), 10000.0)
    stats_b = stats_from_trades(b_wins + b_losses, equity(b_wins + b_losses, 1500.0), 10000.0)
    for s in (stats_a, stats_b):
        s["buy_hold_pct"] = 0.0
    sa = score_stats(stats_a, None, n_trials=100)
    sb = score_stats(stats_b, None, n_trials=100)
    return {
        "A_5_winning_trades": {"fitness": sa["fitness"],
                               "raw_pf": round(stats_a["raw_profit_factor"], 3),
                               "pf_term": round(sa["profit_factor"], 3),
                               "trades": sa["trades"], "flag": sa["flag"]},
        "B_200_trades": {"fitness": sb["fitness"],
                         "raw_pf": round(stats_b["raw_profit_factor"], 3),
                         "pf_term": round(sb["profit_factor"], 3),
                         "trades": sb["trades"], "flag": sb["flag"]},
    }


def experiment_dsr():
    """Item 5: DSR units — annualised Sharpe compared with a per-period hurdle."""
    from core.ga.fitness import deflated_sharpe_ratio
    out = {}
    for t in (1200, 365, 250):
        call = dict(observed_sharpe=1.2, n_trials=1200, observation_periods=t)
        try:
            out[f"T={t}"] = deflated_sharpe_ratio(**call)
        except TypeError:
            out[f"T={t}"] = deflated_sharpe_ratio(
                observed_sharpe=1.2, n_trials=1200, observation_periods=t)
    return out


def experiment_publication_gate(tmpdir: Path):
    """Item 6: a champion whose metrics fail the gate must not be enabled."""
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    loader = StrategyLoader(str(tmpdir / "gate_strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)

    class _Engine:
        config = None

    evolver = GAStrategyEvolver(_Engine(), loader, GARunConfig(population_size=4))
    cases = {
        "zero_trades": {"trade_count": 0, "total_return": 0.0, "profit_factor": 0.1,
                        "dsr": 0.0, "buy_hold_pct": 0.0, "alpha_vs_buy_hold_pct": 0.0},
        "5_winning_trades": {"trade_count": 5, "total_return": 8.0, "profit_factor": 3.0,
                             "dsr": 0.4, "buy_hold_pct": 0.0,
                             "alpha_vs_buy_hold_pct": 8.0},
        "healthy_200_trades": {"trade_count": 200, "total_return": 15.0,
                               "profit_factor": 1.8, "dsr": 0.5,
                               "buy_hold_pct": 3.0, "alpha_vs_buy_hold_pct": 12.0},
    }
    out = {}
    for name, metrics in cases.items():
        published, reasons = evolver._publication_decision(metrics, None)
        out[name] = {"published": published, "reasons": reasons}
    return out


def experiment_generations(engine, loader, workdir, population=12, generations=3,
                           seed=99):
    """Item 1/3: does best fitness actually improve across generations?"""
    from core.ga.evolver import GAStrategyEvolver, GARunConfig

    cfg = GARunConfig(population_size=population, generations=generations,
                      elite_count=2, immigrant_count=2, max_workers=1, seed=seed)
    evolver = GAStrategyEvolver(engine, loader, cfg)
    result = evolver.evolve(["BTCUSDT", "ETHUSDT"], "2026-05-01", "2026-06-01",
                            seed=seed, window_key="experiment")
    return {
        "history": [{k: h[k] for k in ("generation", "best_fitness",
                                       "best_trades", "best_sharpe")}
                    for h in result.get("history", [])],
        "champion": result.get("champion_name"),
        "published": result.get("published"),
        "rejection_reasons": result.get("rejection_reasons"),
        "fitness": result.get("fitness"),
        "trade_count": result.get("trade_count"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-dir", required=True,
                        help="directory that CONTAINS data/market/<SYM>/1h.parquet")
    parser.add_argument("--skip-population", action="store_true")
    parser.add_argument("--generations", action="store_true")
    parser.add_argument("--genomes", type=int, default=6)
    args = parser.parse_args()

    import tempfile as _tf
    workdir = Path(_tf.mkdtemp(prefix="p1_exp_"))
    data_root = Path(args.data_dir).resolve()
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    try:
        (workdir / "data" / "market").symlink_to(data_root / "data" / "market",
                                                 target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        import shutil
        shutil.copytree(data_root / "data" / "market", workdir / "data" / "market")

    cfg, engine, loader = build_engine(None, workdir)
    result = {"label": args.label}
    if not args.skip_population:
        pop = experiment_population(engine, loader, n=args.genomes)
        result["population"] = pop
        result["population_summary"] = {
            "genomes": len(pop),
            "trading": sum(1 for p in pop if (p["trades"] or 0) > 0),
            "zero_trade": sum(1 for p in pop if (p["trades"] or 0) == 0),
            "max_trades": max((p["trades"] or 0) for p in pop),
            "total_trades": sum((p["trades"] or 0) for p in pop),
        }
    result["fitness_ranking"] = experiment_fitness_ranking()
    result["dsr"] = experiment_dsr()
    result["publication_gate"] = experiment_publication_gate(workdir)
    if args.generations:
        result["generations"] = experiment_generations(engine, loader, workdir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
