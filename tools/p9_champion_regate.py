"""P9 deliverable 3 — re-evaluate the shipped champions under `next_open`.

The champions in ``strategies/`` carry the numbers they were selected and gated
on (``provenance.fitness_components`` + ``provenance.validation``).  Those numbers
were produced under the shipped fill convention ``close`` (the engine priced every
fill at the decision bar's own close).  This tool re-evaluates the **same champion
strategies, on the same windows, symbols and costs, with the GA's own scoring
path**, twice — once per convention — and reports the deltas plus whether the
publication-gate verdict changes.

What is *exactly* reproduced, and what is not
---------------------------------------------
Reproduced: the engine run (legacy, ML off, ``use_live_spread=False``,
``benchmark_mode = ga.benchmark_mode`` = ``exposure_matched``), the per-genome
ledger, `stats_from_trades`, and `score_stats` with the champion's own
``provenance.n_trials`` — i.e. the same formula the GA selected on, including the
deflated Sharpe and the benchmark alpha.

Not reproduced, and why it cannot be: the champion YAML is a **decoded strategy**,
not the chromosome, so the genome-complexity penalty and the P6-D executability
model cannot be recomputed.  ``score_stats(stats, chromosome=None, ...)`` omits
both.  The absolute fitness therefore differs a little from the recorded value —
so the recorded value is printed beside the reproduction and the number to read is
the **delta between the two conventions** (the missing terms are identical in both
arms, since the strategy does not change).

The gate verdict is not a reconstruction: it is
``GAStrategyEvolver._publication_decision`` — the production gate.

Usage::

    python tools/p9_champion_regate.py --out data/p9_evidence/champion_regate.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys                                                                  # noqa: E402
sys.path.insert(0, str(ROOT))

#: The champions that carry BOTH a full benchmark provenance block and an
#: out-of-sample validation window — i.e. the ones whose gate outcome is
#: documented.  ``(yaml stem, the GA configuration that produced it)``.
CHAMPIONS = (
    ("ga_champion_1790844776", "GA config A (seed 20261001, 12 generations, n_trials=690+prior)"),
    ("ga_champion_1790855974", "GA config B (seed 20261002, 12 generations, n_trials=930)"),
    ("ga_champion_1790867208", "GA config B resumed (seed 20261002, 32 generations, n_trials=1570)"),
)

INITIAL_BALANCE = 10_000.0


def _engine_stack():
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager

    Config._instance = None
    config = Config.load("sim")
    config.backtest_engine_mode = "legacy"
    config.backtest_ml_enabled = False
    config.backtest_live_spread_enabled = False
    bus = EventBus()
    return config, BacktestEngine(config, None, RiskManager(config, bus),
                                  OrderExecutor(config, bus))


def _load_champion(stem: str):
    raw = yaml.safe_load((ROOT / "strategies" / f"{stem}.yaml").read_text(encoding="utf-8"))
    from core.strategy.loader import StrategyConfig

    return raw, StrategyConfig(**{k: v for k, v in raw.items() if k != "provenance"})


def _ga_score(engine, strategy, symbols, start, end, convention, n_trials,
              benchmark_mode, alpha_weight):
    """One engine run scored by the GA's own formula (mirrors evaluate_chromosome)."""
    from core.ga.fitness import isolated_eval_kwargs, score_stats, stats_from_trades

    result = engine.run_with_exit_evaluation(
        strategies=[strategy], symbols=list(symbols), date_start=start, date_end=end,
        initial_balance=INITIAL_BALANCE, mode="full", simulate_ai_weights=False,
        use_live_spread=False, benchmark_mode=benchmark_mode,
        fill_convention=convention, **isolated_eval_kwargs())
    metrics = result.get("metrics") or {}
    per_eq = (result.get("per_strategy_equity") or {}).get(strategy.name) or {}
    equity = per_eq.get("equity_curve") or result.get("equity_curve") or []
    trades = [t for t in (result.get("trades") or [])
              if t.get("strategy") == strategy.name] or (result.get("trades") or [])

    stats = stats_from_trades(trades, equity, INITIAL_BALANCE)
    stats["buy_hold_pct"] = metrics.get("buy_hold_pct")
    bench = per_eq.get("benchmark") or {}
    stats["benchmark"] = bench
    stats["benchmark_mode"] = bench.get("mode") or "buy_hold"
    stats["benchmark_pct"] = (bench.get("benchmark_pct") if bench
                              else metrics.get("buy_hold_pct"))
    stats["max_dd"] = stats["max_dd_pct"]
    score = score_stats(stats, None, n_trials=int(n_trials), alpha_weight=alpha_weight)
    # The same post-processing `evaluate_chromosome` applies, so the keys read
    # here are the keys the GA selected on (``score_stats`` itself returns
    # ``trades``/``deflated_sharpe``, not ``trade_count``/``dsr``).
    from core.ga.fitness import _finite

    score["sharpe"] = round(_finite(score.get("sharpe")), 4)
    score["win_rate"] = round(_finite(stats["win_rate"]), 2)
    score["profit_factor"] = round(_finite(stats["profit_factor"]), 4)
    score["max_dd"] = round(_finite(score["max_dd"]), 2)
    score["total_return"] = round(_finite(stats["total_return_pct"]), 2)
    score["trade_count"] = int(stats["trades"])
    score["strategy_name"] = strategy.name
    score["dsr"] = round(_finite(score.get("deflated_sharpe")), 4)
    score["raw_profit_factor"] = round(_finite(stats["raw_profit_factor"]), 4)
    score["buy_hold_pct"] = (round(_finite(stats["buy_hold_pct"]), 4)
                             if stats["buy_hold_pct"] is not None else None)
    score["observations"] = int(stats["observations"])
    score["engine_sharpe"] = metrics.get("sharpe_ratio")
    score["engine_total_return_pct"] = metrics.get("total_return_pct")
    score["engine_max_dd_pct"] = metrics.get("max_drawdown_pct")
    score["fill_convention"] = result.get("fill_convention")
    score["fill_convention_accounting"] = metrics.get("fill_convention_accounting")
    score["benchmark_report"] = bench
    return score


def _gate(evolver, score, validation):
    """The production publication gate on a re-evaluated (train, validation) pair."""
    train_result = {
        "trade_count": score.get("trade_count"),
        "total_return": score.get("total_return"),
        "profit_factor": score.get("profit_factor"),
        "dsr": score.get("dsr"),
        "benchmark_mode": score.get("benchmark_mode"),
        "alpha_vs_benchmark_pct": score.get("alpha_vs_benchmark_pct"),
        "alpha_vs_buy_hold_pct": score.get("alpha_vs_buy_hold_pct"),
        "benchmark_pct": score.get("benchmark_pct"),
        "buy_hold_pct": score.get("buy_hold_pct"),
    }
    published, reasons = evolver._publication_decision(train_result, validation)
    return published, list(reasons), train_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    parser.add_argument("--conventions", default="close,next_open")
    args = parser.parse_args()
    conventions = [c.strip() for c in args.conventions.split(",") if c.strip()]

    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    config, engine = _engine_stack()
    loader = StrategyLoader(str(Path(tempfile.mkdtemp(prefix="p9_regate_")) / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    evolver = GAStrategyEvolver(engine, loader, GARunConfig(population_size=2))
    alpha_weight = getattr(config, "ga_alpha_weight", None)
    benchmark_mode = getattr(config, "ga_benchmark_mode", "buy_hold")
    print(f"benchmark_mode={benchmark_mode} alpha_weight={alpha_weight} "
          f"cost: fee={config.backtest_taker_fee_pct}% spreads={config.backtest_spread_pct}")

    report = {"benchmark_mode": benchmark_mode, "alpha_weight": alpha_weight,
              "champions": [], "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t0 = time.time()
    for stem, ga_config in CHAMPIONS:
        raw, strategy = _load_champion(stem)
        prov = raw.get("provenance") or {}
        window = prov.get("window") or {}
        symbols = list(prov.get("symbols") or strategy.symbols or [])
        n_trials = int(prov.get("n_trials") or 1)
        train_start = window.get("train_start") or window.get("key", "~").split("~")[0]
        train_end = window.get("train_end")
        val_start = window.get("validation_start")
        val_end = window.get("validation_end")
        entry = {
            "champion": stem, "ga_config": ga_config,
            "windows": {"train": [train_start, train_end],
                        "validation": [val_start, val_end]},
            "symbols": symbols,
            "timeframes": list(strategy.timeframes or []),
            "n_trials": n_trials,
            "recorded": {
                "fitness_components": prov.get("fitness_components"),
                "validation": prov.get("validation"),
                "published": prov.get("published"),
                "rejection_reasons": prov.get("rejection_reasons"),
            },
            "arms": {},
        }
        print(f"\n== {stem}  window {train_start}~{train_end} "
              f"val {val_start}~{val_end} symbols={symbols} tf={strategy.timeframes}")
        for convention in conventions:
            started = time.time()
            train = _ga_score(engine, strategy, symbols, train_start, train_end,
                              convention, n_trials, benchmark_mode, alpha_weight)
            validation_raw = _ga_score(engine, strategy, symbols, val_start, val_end,
                                       convention, n_trials, benchmark_mode, alpha_weight)
            validation = {
                "sharpe": validation_raw.get("sharpe"),
                "trade_count": validation_raw.get("trade_count"),
                "total_return": validation_raw.get("total_return"),
                "profit_factor": validation_raw.get("profit_factor"),
                "max_dd": validation_raw.get("max_dd"),
                "dsr": validation_raw.get("dsr"),
                "buy_hold_pct": validation_raw.get("buy_hold_pct"),
                "benchmark_mode": validation_raw.get("benchmark_mode"),
                "benchmark_pct": validation_raw.get("benchmark_pct"),
                "alpha_vs_benchmark_pct": validation_raw.get("alpha_vs_benchmark_pct"),
                "start": val_start, "end": val_end,
            }
            published, reasons, gate_input = _gate(evolver, train, validation)
            entry["arms"][convention] = {
                "train": {k: train.get(k) for k in (
                    "fitness", "fitness_base", "fitness_alpha", "sharpe",
                    "deflated_sharpe", "dsr", "trade_count", "total_return",
                    "profit_factor", "max_dd", "buy_hold_pct",
                    "alpha_vs_buy_hold_pct", "alpha_vs_benchmark_pct",
                    "benchmark_pct", "benchmark_mode", "observations",
                    "engine_sharpe", "engine_total_return_pct",
                    "engine_max_dd_pct", "fill_convention",
                    "fill_convention_accounting")},
                "validation": validation,
                "gate": {"published": published, "reasons": reasons},
                "gate_input": gate_input,
                "seconds": round(time.time() - started, 1),
            }
            print(f"   {convention:<10} trades={train.get('trade_count'):<5} "
                  f"fitness={train.get('fitness'):<9} sharpe={train.get('sharpe'):<8} "
                  f"dsr={train.get('dsr'):<9} alpha_bench={train.get('alpha_vs_benchmark_pct')} "
                  f"val_dsr={validation.get('dsr')} published={published} "
                  f"({entry['arms'][convention]['seconds']}s)")
        # ── the deltas ──
        a = entry["arms"].get("close") or {}
        b = entry["arms"].get("next_open") or {}
        if a and b:
            entry["delta"] = {
                key: (None if a["train"].get(key) is None or b["train"].get(key) is None
                      else round(b["train"][key] - a["train"][key], 6))
                for key in ("fitness", "sharpe", "deflated_sharpe", "trade_count",
                            "total_return", "max_dd", "alpha_vs_benchmark_pct",
                            "alpha_vs_buy_hold_pct", "profit_factor")
            }
            entry["delta"]["validation_dsr"] = (
                None if a["validation"].get("dsr") is None
                or b["validation"].get("dsr") is None
                else round(b["validation"]["dsr"] - a["validation"]["dsr"], 6))
            entry["delta"]["validation_sharpe"] = (
                None if a["validation"].get("sharpe") is None
                or b["validation"].get("sharpe") is None
                else round(b["validation"]["sharpe"] - a["validation"]["sharpe"], 6))
            entry["verdict_changes"] = (
                a["gate"]["published"] != b["gate"]["published"]
                or a["gate"]["reasons"] != b["gate"]["reasons"])
        report["champions"].append(entry)

    report["runtime_seconds"] = round(time.time() - t0, 1)
    report["verdict"] = {
        "gate_verdict_changes": sum(1 for c in report["champions"]
                                    if c.get("verdict_changes")),
        "champions": len(report["champions"]),
    }
    print("\n-- gate verdicts --")
    for c in report["champions"]:
        for convention in conventions:
            arm = c["arms"][convention]
            print(f"  {c['champion']} [{convention}] published={arm['gate']['published']} "
                  f"reasons={arm['gate']['reasons']}")
    print(f"verdict: {report['verdict']}  runtime {report['runtime_seconds']}s")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str),
                                  encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
