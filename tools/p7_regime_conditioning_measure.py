"""P7-S1 evidence — does causal regime conditioning change out-of-sample alpha?

The operator's hypothesis (P7): the strategies have **no market-regime
sensitivity**, so they trade in conditions they are unsuited to, and conditioning
them on a causal regime label should improve their out-of-sample result.  This
tool measures that hypothesis on the **real cached parquet**, with the production
engine, the production scorer and the production publication gate, and reports the
answer whether or not it is the hoped-for one.

Design (bounded, reproducible, read-only)
----------------------------------------
* A **fixed cohort** of ``--population`` genomes is drawn from one seed.  Every
  genome is evaluated twice per regime: as drawn (the *unconditioned* arm) and
  with its ``regime_filter`` gene pinned to one of the five gate labels (the
  *conditioned* arm).  The two arms are the SAME genomes, so the comparison is
  paired and no GA selection effect can explain a difference.
* Each candidate is evaluated on the **train** window and on a disjoint
  **out-of-sample** window, both with ``benchmark_mode=exposure_matched`` (the
  shipped gate benchmark), so "alpha" is like-for-like in both arms.
* Reported per candidate: train/OOS total return, exposure-matched benchmark,
  alpha versus it, DSR (with the real trial count), trade count and the
  strategy's time-in-market share.  The summary reports the paired change
  (conditioned minus unconditioned) per regime and overall, including how many
  candidates still clear the gate's trade floor.
* ``--ga`` additionally runs one bounded GA with the switch off and one with it
  on (same seed) and reports both champions' DSR / trades / OOS alpha — the
  end-to-end version of the same question.
* Nothing is written except ``--out`` (JSON) and the GA stub's own temp dir: no
  ``data/binance_trader.db``, no ``strategies/``, no ``data/market`` writes.
  The GA stub mirrors ``tools/ga_real_data_curve.py``'s isolation pattern.

Usage::

    python tools/p7_regime_conditioning_measure.py
    python tools/p7_regime_conditioning_measure.py --population 8 --ga
    python tools/p7_regime_conditioning_measure.py --symbols BTCUSDT ETHUSDT \
        --train-start 2025-11-01 --train-end 2026-02-01 \
        --oos-start 2026-02-01 --oos-end 2026-06-01
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ga.benchmark import EXPOSURE_MATCHED            # noqa: E402
from core.strategy.regime_causal import GATE_REGIME_LABELS  # noqa: E402

DEFAULT_OUT = Path(tempfile.gettempdir()) / "p7_regime_conditioning.json"
NEUTRAL = ""


# ── engine / data plumbing ──────────────────────────────────────────────

def _engine_stack(data_dir: str | None = None):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager

    Config._instance = None
    config = Config.load("sim")
    if data_dir:
        config.data_dir = str(data_dir)
    # The GA evaluation contract: legacy engine, no ML, no live order book.
    config.backtest_engine_mode = "legacy"
    config.backtest_ml_enabled = False
    config.backtest_live_spread_enabled = False
    bus = EventBus()
    engine = BacktestEngine(config, None, RiskManager(config, bus),
                            OrderExecutor(config, bus))
    return config, engine


def _cohort(population: int, seed: int, timeframe: str = "1h"):
    """A fixed cohort of chromosomes (timeframe pinned, volume genes neutral)."""
    from core.ga.genome import random_chromosome

    rng_state = random.getstate()
    random.seed(seed)
    chroms = [random_chromosome(f"p7_base_{i}") for i in range(population)]
    random.setstate(rng_state)
    for chrom in chroms:
        for gene in chrom.get("categorical", []):
            if gene.name == "timeframes":
                gene.value = timeframe
    return chroms


def _with_regime(chrom: dict, regime: str) -> dict:
    """A copy of *chrom* whose regime gene is pinned to *regime* ("" = none)."""
    import copy

    from core.ga.genome import confine_regime_gene

    out = copy.deepcopy(chrom)
    if not regime:
        confine_regime_gene(out, False)
        return out
    confine_regime_gene(out, True)
    found = False
    for gene in out["categorical"]:
        if gene.name == "regime_filter":
            gene.value = regime
            found = True
    if not found:
        from core.ga.genome import CategoricalGene, regime_gene_options
        out["categorical"].append(
            CategoricalGene("regime_filter", regime, regime_gene_options()))
    return out


def _decode(chrom: dict, regime_conditioning: bool):
    from core.ga.genome import chromosome_to_strategy

    return chromosome_to_strategy(chrom, regime_conditioning=regime_conditioning)


def _evaluate_chunk(engine, chroms, symbols, start, end, n_trials, names):
    """Evaluate a whole arm in ONE engine pass → ``{name: row}``.

    The production GA evaluates a generation's genomes as one call over the same
    data (``evaluate_population_batch`` → ``run_with_exit_evaluation`` with
    ``per_genome_ledger``), and each genome gets its own position slots and its
    own cash/equity sub-ledger.  Doing the same here is both faithful and ~an
    order of magnitude cheaper than one engine call per genome (the bar loop and
    the indicator cache are paid once per chunk, not once per candidate).

    ``--serial`` falls back to one call per genome, which is the slower way to
    measure the same thing and is kept so the chunking assumption can be checked
    (the two paths must agree candidate for candidate).
    """
    from core.ga.fitness import (isolated_eval_kwargs, score_stats,
                                 stats_from_engine_result)

    out: dict = {}
    strategies = []
    pairs = []
    for chrom, name in zip(chroms, names):
        try:
            strategy = _decode(chrom, bool(chrom.get("_regime")))
        except Exception as exc:
            out[name] = {"error": f"decode: {exc}"}
            continue
        strategy.name = name
        if strategy.ml_config:
            strategy.ml_config.enabled = False
        strategies.append(strategy)
        pairs.append((strategy, chrom))
    if not strategies:
        return out
    started = time.time()
    result = engine.run_with_exit_evaluation(
        strategies=strategies, symbols=list(symbols), date_start=start,
        date_end=end, initial_balance=10_000.0, mode="full",
        simulate_ai_weights=False, ml_engine="lightgbm", use_live_spread=False,
        benchmark_mode=EXPOSURE_MATCHED, **isolated_eval_kwargs())
    if "error" in result:
        return {name: {"error": str(result["error"])} for name in names}
    per_genome = round((time.time() - started) / max(len(strategies), 1), 2)
    # The engine disambiguates duplicate strategy NAMES inside a chunk
    # (``p7_0_`` → ``p7_0__0``); map its keys back to this tool's names so the
    # per-genome ledger lookup below is the one it looks like it is (and the
    # result dict keeps the names the caller handed in).
    _alias = {}
    _per = result.get("per_strategy_equity") or {}
    for strategy, _chrom in pairs:
        actual = next((key for key in _per
                       if key == strategy.name
                       or key.startswith(f"{strategy.name}_")), strategy.name)
        _alias[actual] = strategy.name
        if actual != strategy.name:
            _per[strategy.name] = _per.pop(actual)
            counts = (result.get("metrics", {}).get("regime_conditioning") or {})
            if actual in counts:
                counts[strategy.name] = counts.pop(actual)
    for strategy, chrom in pairs:
        stats = stats_from_engine_result(
            result, strategy.name, 10_000.0, chromosome=chrom,
            config=getattr(engine, "config", None))
        score = score_stats(stats, chrom, n_trials=n_trials)
        report = stats.get("benchmark") or {}
        out[strategy.name] = {
            "regime_filter": score.get("regime_filter"),
            "trades": int(score["trades"]),
            "total_return_pct": round(float(score["total_return_pct"]), 4),
            "benchmark_pct": (None if score.get("benchmark_pct") is None
                              else round(float(score["benchmark_pct"]), 4)),
            "alpha_vs_benchmark_pct": round(
                float(score["alpha_vs_benchmark_pct"]), 4),
            "sharpe": round(float(score["sharpe"]), 4),
            "dsr": round(float(score["deflated_sharpe"]), 6),
            "max_dd_pct": round(float(score["max_dd_pct"]), 4),
            "time_in_market_pct": report.get("strategy_time_in_market_pct"),
            "observations": int(score["observations"]),
            "insufficient_data": bool(score["insufficient_data"]),
            "regime_conditioning": score.get("regime_conditioning"),
            "seconds_per_genome": per_genome,
        }
    return out


def _evaluate(engine, chrom, symbols, start, end, n_trials, name):
    """One genome on one window in its own engine call (``--serial`` path)."""
    return _evaluate_chunk(engine, [chrom], symbols, start, end, n_trials,
                           [name]).get(name, {"error": "not evaluated"})


# ── the paired table ────────────────────────────────────────────────────

def _median(values):
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 4) if vals else None


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def _delta(row_a, row_b, key):
    """``row_b[key] - row_a[key]`` when both are numbers, else ``None``."""
    if not row_a or not row_b or "error" in row_a or "error" in row_b:
        return None
    a, b = row_a.get(key), row_b.get(key)
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 4)


def run(args) -> dict:
    symbols = [s.strip().upper() for s in args.symbols if s.strip()]
    regimes = [NEUTRAL, *GATE_REGIME_LABELS]
    base_cohort = _cohort(args.population, args.seed, args.timeframe)

    # The trial count the DSR is deflated by: every candidate evaluated by this
    # measurement (both arms, both windows) — the honest N for the table.
    n_trials = args.population * len(regimes) * 2

    artifact = {
        "script": "tools/p7_regime_conditioning_measure.py",
        "data": {"root": str(ROOT / "data" / "market"), "symbols": symbols,
                 "timeframe": args.timeframe},
        "windows": {"train": [args.train_start, args.train_end],
                    "out_of_sample": [args.oos_start, args.oos_end]},
        "cohort": {"population": args.population, "seed": args.seed,
                   "regimes": regimes, "n_trials_for_dsr": n_trials},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidates": [],
        "summary": {},
    }

    config, engine = _engine_stack(args.data_dir)
    print(f"P7-S1 measurement: {args.population} genomes x {len(regimes)} "
          f"conditioning levels x 2 windows = "
          f"{args.population * len(regimes) * 2} evaluations, "
          f"{symbols} {args.timeframe}, train {args.train_start}~{args.train_end}, "
          f"OOS {args.oos_start}~{args.oos_end}", flush=True)
    print(f"ga.regime_conditioning (config) = {config.ga_regime_conditioning}",
          flush=True)

    rows = []
    # One chunk per (arm, window): the production GA's own evaluation shape.
    chunks = {}
    for index, base in enumerate(base_cohort):
        for regime in regimes:
            chrom = _with_regime(base, regime)
            chrom["_regime"] = regime
            name = f"p7_{index}_{regime or 'none'}"
            chunks.setdefault(bool(regime), []).append((index, regime, chrom, name))

    for conditioned, entries in sorted(chunks.items()):
        for window, (start, end) in (("train", (args.train_start, args.train_end)),
                                     ("oos", (args.oos_start, args.oos_end))):
            chroms = [entry[2] for entry in entries]
            names = [entry[3] for entry in entries]
            if args.serial:
                evaluated = {name: _evaluate(
                    engine, chrom, symbols, start, end, n_trials, name)
                    for chrom, name in zip(chroms, names)}
            else:
                evaluated = _evaluate_chunk(engine, chroms, symbols, start, end,
                                            n_trials, names)
            for index, regime, _chrom, name in entries:
                row = next((r for r in rows
                            if r["candidate"] == index and r["regime"] ==
                            (regime or NEUTRAL)), None)
                if row is None:
                    row = {"candidate": index, "regime": regime or NEUTRAL,
                           "name": name, "conditioned": bool(regime)}
                    rows.append(row)
                row[window] = evaluated.get(name, {"error": "not evaluated"})

    for row in sorted(rows, key=lambda r: (r["candidate"], r["regime"])):
        train, oos = row["train"], row["oos"]
        print(
            f"  [{row['candidate']:>2} {row['regime']:<10}] "
            f"train trades={train.get('trades', 'err'):>4} "
            f"dsr={train.get('dsr', 'err')} "
            f"| oos trades={oos.get('trades', 'err'):>4} "
            f"ret={oos.get('total_return_pct', 'err')} "
            f"bench={oos.get('benchmark_pct', 'err')} "
            f"alpha={oos.get('alpha_vs_benchmark_pct', 'err')} "
            f"dsr={oos.get('dsr', 'err')} "
            f"tim={oos.get('time_in_market_pct')}", flush=True)
        artifact["candidates"] = rows
        artifact["summary"] = _summarise(rows, regimes, args)
        _write(Path(args.out), artifact)

    artifact["summary"] = _summarise(rows, regimes, args)
    artifact["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if args.ga:
        artifact["ga"] = _ga_arms(args, symbols)
    _write(Path(args.out), artifact)
    _print_summary(artifact)
    return artifact


def _summarise(rows, regimes, args) -> dict:
    """Paired deltas (conditioned minus unconditioned) per regime and overall."""
    by_candidate = {}
    for row in rows:
        by_candidate.setdefault(row["candidate"], {})[row["regime"]] = row

    per_regime = {}
    for regime in regimes:
        if not regime:
            continue
        pairs = [(by_candidate[i][NEUTRAL], by_candidate[i][regime])
                 for i in sorted(by_candidate)
                 if NEUTRAL in by_candidate[i] and regime in by_candidate[i]]
        per_regime[regime] = _pair_stats(pairs)

    all_pairs = []
    for regime in regimes:
        if not regime:
            continue
        all_pairs.extend(
            (by_candidate[i][NEUTRAL], by_candidate[i][regime])
            for i in sorted(by_candidate)
            if NEUTRAL in by_candidate[i] and regime in by_candidate[i])
    uns = [r for r in rows if not r["conditioned"]]
    cons = [r for r in rows if r["conditioned"]]
    return {
        "unconditioned": _arm_stats(uns),
        "conditioned": _arm_stats(cons),
        "per_regime": per_regime,
        "overall_paired": _pair_stats(all_pairs),
        "trade_floor": {
            "evaluations_unconditioned": len(uns),
            "unconditioned_meeting": sum(
                1 for r in uns if not r["train"].get("insufficient_data", True)),
            "evaluations_conditioned": len(cons),
            "conditioned_meeting": sum(
                1 for r in cons if not r["train"].get("insufficient_data", True)),
            "floor": 30,
        },
        "zero_trade_cells": {
            "oos": sum(1 for r in rows if not r["oos"].get("trades")),
            "train": sum(1 for r in rows if not r["train"].get("trades")),
            "evaluations": len(rows),
        },
        "caveats": [
            "alpha_vs_benchmark is in PERCENTAGE POINTS of the run window's "
            "return, against the exposure-matched basket; it is not annualised",
            "the DSR is deflated with n_trials=" + str(n_trials := args.population
             * len(regimes) * 2) + " (every candidate this measurement evaluated, "
            "both arms and both windows) and is therefore directly comparable "
            "across arms",
            "a conditioned cell with 0 trades has alpha 0.0 by construction (no "
            "exposure to match); those cells are counted, not dropped",
        ],
    }


def _arm_stats(rows) -> dict:
    def col(window, key):
        return [r[window].get(key) for r in rows if "error" not in r[window]]

    return {
        "evaluations": len(rows),
        "train_trades_median": _median(col("train", "trades")),
        "oos_trades_median": _median(col("oos", "trades")),
        "oos_alpha_median": _median(col("oos", "alpha_vs_benchmark_pct")),
        "oos_alpha_mean": _mean(col("oos", "alpha_vs_benchmark_pct")),
        "oos_alpha_positive": sum(
            1 for v in col("oos", "alpha_vs_benchmark_pct") if v is not None and v > 0),
        "oos_dsr_median": _median(col("oos", "dsr")),
        "oos_dsr_positive": sum(
            1 for v in col("oos", "dsr") if v is not None and v > 0),
        "oos_time_in_market_median": _median(col("oos", "time_in_market_pct")),
        "train_alpha_median": _median(col("train", "alpha_vs_benchmark_pct")),
        "train_dsr_median": _median(col("train", "dsr")),
    }


def _pair_stats(pairs) -> dict:
    """Paired change (conditioned − unconditioned) over the same genomes."""
    def deltas(key):
        out = []
        for before, after in pairs:
            value = _delta(before["oos"], after["oos"], key)
            if value is not None:
                out.append(value)
        return out

    train_trades = [_delta(b["train"], a["train"], "trades") for b, a in pairs]
    oos_trades = [_delta(b["oos"], a["oos"], "trades") for b, a in pairs]
    alpha = deltas("alpha_vs_benchmark_pct")
    dsr = deltas("dsr")
    tim = deltas("time_in_market_pct")
    return {
        "pairs": len(pairs),
        "oos_alpha_delta_median": _median(alpha),
        "oos_alpha_delta_mean": _mean(alpha),
        "oos_alpha_improved": sum(1 for v in alpha if v > 0),
        "oos_alpha_worsened": sum(1 for v in alpha if v < 0),
        "oos_dsr_delta_median": _median(dsr),
        "train_trades_delta_median": _median(train_trades),
        "oos_trades_delta_median": _median(oos_trades),
        "oos_time_in_market_delta_median": _median(tim),
    }


# ── optional: one bounded GA per arm ────────────────────────────────────

def _ga_arms(args, symbols) -> dict:
    """Two bounded GA runs (switch off / on) on the SAME seed, in a temp root."""
    import shutil
    import subprocess

    workdir = Path(tempfile.gettempdir()) / "p7_ga_arms"
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    (workdir / "strategies").mkdir(parents=True, exist_ok=True)
    target = workdir / "data" / "market"
    source = ROOT / "data" / "market"
    try:
        target.symlink_to(source, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copytree(source, target)

    out = {}
    for flag, tag in ((False, "conditioning_off"), (True, "conditioning_on")):
        script = _GA_STUB.format(
            root=str(ROOT), workdir=str(workdir), conditioning=repr(flag),
            population=args.ga_population, generations=args.ga_generations,
            seed=args.ga_seed, symbols=repr(symbols), start=repr(args.train_start),
            end=repr(args.train_end), oos_start=repr(args.oos_start),
            oos_end=repr(args.oos_end), timeframe=repr(args.timeframe))
        path = workdir / f"stub_{tag}.py"
        path.write_text(script, encoding="utf-8")
        started = time.time()
        proc = subprocess.run([sys.executable, str(path)],
                              capture_output=True, text=True, cwd=str(ROOT),
                              timeout=max(int(args.ga_max_seconds), 60))
        payload = {}
        marker = "P7_GA_RESULT "
        for line in (proc.stdout or "").splitlines():
            if line.startswith(marker):
                payload = json.loads(line[len(marker):])
        out[tag] = {"payload": payload, "seconds": round(time.time() - started, 1),
                    "returncode": proc.returncode,
                    "stdout_tail": (proc.stdout or "").strip().splitlines()[-6:]}
        print(f"  [ga {tag}] {json.dumps(payload, default=str)[:400]}", flush=True)
    return out


#: The stub keeps the GA out of the repository: temp data root, pinned timeframe,
#: ``keep_checkpoint`` off.  It is a *tool* file, not a source file of the app.
_GA_STUB = '''\
import json, random, sys
from pathlib import Path
sys.path.insert(0, {root!r})
from core.ga.genome import random_chromosome as _orig
import core.ga.evolver as evolver_mod

CONDITIONING = {conditioning}

def _pinned(name="ga_strategy", timeframe_pool=None, regime_conditioning=None):
    chrom = _orig(name, timeframe_pool=timeframe_pool,
                  regime_conditioning=CONDITIONING)
    for gene in chrom["categorical"]:
        if gene.name == "timeframes":
            gene.value = {timeframe}
    return chrom

evolver_mod.random_chromosome = _pinned

from app.config import Config
from app.event_bus import EventBus
from core.backtest.engine import BacktestEngine
from core.executor.executor import OrderExecutor
from core.risk.manager import RiskManager
from core.strategy.loader import StrategyLoader
from core.ga.evolver import GAStrategyEvolver, GARunConfig

Config._instance = None
cfg = Config.load("sim")
cfg.data_dir = str(Path({workdir!r}) / "data")
cfg.backtest_engine_mode = "legacy"
cfg.backtest_ml_enabled = False
cfg.backtest_live_spread_enabled = False
bus = EventBus()
loader = StrategyLoader(str(Path({workdir!r}) / "strategies"))
engine = BacktestEngine(cfg, None, RiskManager(cfg, bus), OrderExecutor(cfg, bus))
run_cfg = GARunConfig(population_size={population}, generations={generations},
                      elite_count=2, immigrant_count=2, max_workers=1,
                      seed={seed}, keep_checkpoint=False,
                      regime_conditioning=CONDITIONING)
ev = GAStrategyEvolver(engine, loader, run_cfg)
result = ev.evolve({symbols}, {start}, {end},
                   validation_start={oos_start}, seed={seed},
                   window_key="p7-ga-arm")

def _fitness(chrom):
    for gene in chrom.get("categorical", []):
        if gene.name == "regime_filter":
            return gene.value
    return None

print("P7_GA_RESULT " + json.dumps({{
    "conditioning": CONDITIONING,
    "champion": result.get("champion_name"),
    "train_fitness": result.get("fitness"),
    "train_trades": result.get("trade_count"),
    "train_dsr": (result.get("dsr") or {{}}).get("dsr"),
    "train_sharpe": result.get("sharpe"),
    "regime_filter_gene": _fitness(ev.population[0]) if ev.population else None,
    "published": result.get("published"),
    "rejection_reasons": result.get("rejection_reasons"),
    "validation": result.get("validation"),
    "provenance_benchmark": (result.get("provenance") or {{}}).get("benchmark"),
    "champion_config_regime_filter": (result.get("champion_config") or {{}}).get("regime_filter"),
}}, default=str))
'''


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _fmt_cell(value) -> str:
    """Right-pad a possibly-``None`` summary cell (an empty window yields None)."""
    return "n/a" if value is None else str(value)


def _print_summary(artifact: dict) -> None:
    s = artifact["summary"]
    print("\n=== P7-S1 regime conditioning — paired summary ===")
    for arm in ("unconditioned", "conditioned"):
        stats = s[arm]
        print(f"{arm:<14} evals={stats['evaluations']:<4} "
              f"train_trades_med={stats['train_trades_median']} "
              f"oos_trades_med={stats['oos_trades_median']} "
              f"oos_alpha_med={stats['oos_alpha_median']} "
              f"oos_alpha_mean={stats['oos_alpha_mean']} "
              f"pos_alpha={stats['oos_alpha_positive']} "
              f"oos_dsr_med={stats['oos_dsr_median']} "
              f"dsr>0={stats['oos_dsr_positive']} "
              f"tim_med={stats['oos_time_in_market_median']}")
    floor = s["trade_floor"]
    print(f"train trade floor (>={floor['floor']} trades): "
          f"unconditioned={floor['unconditioned_meeting']}/"
          f"{floor['evaluations_unconditioned']} "
          f"conditioned={floor['conditioned_meeting']}/"
          f"{floor['evaluations_conditioned']}   "
          f"zero-trade cells (OOS)={s['zero_trade_cells']['oos']}/"
          f"{s['zero_trade_cells']['evaluations']}")
    print("\nper-regime paired change (conditioned - unconditioned), OOS:")
    header = (f"{'regime':<12}{'pairs':>6}{'d_alpha_med':>13}{'d_alpha_mean':>14}"
              f"{'better':>8}{'worse':>7}{'d_dsr_med':>11}"
              f"{'d_tr_trades':>12}{'d_oos_trades':>13}{'d_tim':>9}")
    print(header)
    print("-" * len(header))
    for regime, stats in s["per_regime"].items():
        print(f"{regime:<12}{_fmt_cell(stats['pairs']):>6}"
              f"{_fmt_cell(stats['oos_alpha_delta_median']):>13}"
              f"{_fmt_cell(stats['oos_alpha_delta_mean']):>14}"
              f"{_fmt_cell(stats['oos_alpha_improved']):>8}"
              f"{_fmt_cell(stats['oos_alpha_worsened']):>7}"
              f"{_fmt_cell(stats['oos_dsr_delta_median']):>11}"
              f"{_fmt_cell(stats['train_trades_delta_median']):>12}"
              f"{_fmt_cell(stats['oos_trades_delta_median']):>13}"
              f"{_fmt_cell(stats['oos_time_in_market_delta_median']):>9}")
    overall = s["overall_paired"]
    print(f"{'OVERALL':<12}{_fmt_cell(overall['pairs']):>6}"
          f"{_fmt_cell(overall['oos_alpha_delta_median']):>13}"
          f"{_fmt_cell(overall['oos_alpha_delta_mean']):>14}"
          f"{_fmt_cell(overall['oos_alpha_improved']):>8}"
          f"{_fmt_cell(overall['oos_alpha_worsened']):>7}"
          f"{_fmt_cell(overall['oos_dsr_delta_median']):>11}"
          f"{_fmt_cell(overall['train_trades_delta_median']):>12}"
          f"{_fmt_cell(overall['oos_trades_delta_median']):>13}"
          f"{_fmt_cell(overall['oos_time_in_market_delta_median']):>9}")
    for tag, arm in (artifact.get("ga") or {}).items():
        payload = arm.get("payload") or {}
        print(f"\nGA {tag}: champion={payload.get('champion')} "
              f"train_fitness={payload.get('train_fitness')} "
              f"train_trades={payload.get('train_trades')} "
              f"train_dsr={payload.get('train_dsr')} "
              f"published={payload.get('published')} "
              f"reasons={payload.get('rejection_reasons')} "
              f"champion_regime_filter="
              f"{payload.get('champion_config_regime_filter')}")
        val = payload.get("validation") or {}
        if val:
            print(f"          OOS: trades={val.get('trade_count')} "
                  f"dsr={val.get('dsr')} sharpe={val.get('sharpe')} "
                  f"ret={val.get('total_return')} "
                  f"bench={val.get('benchmark_pct')} "
                  f"alpha={val.get('alpha_vs_benchmark_pct')}")
    print(f"\nartifact: {artifact.get('_out_path', '')}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*",
                        default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--train-start", default="2025-11-01")
    parser.add_argument("--train-end", default="2026-02-01")
    parser.add_argument("--oos-start", default="2026-02-01")
    parser.add_argument("--oos-end", default="2026-06-01")
    parser.add_argument("--population", type=int, default=8,
                        help="genomes in the fixed cohort (x 6 levels x 2 windows)")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--serial", action="store_true",
                        help="one engine call per genome instead of one per arm "
                             "(slower; used to check the chunked path agrees)")
    parser.add_argument("--ga", action="store_true",
                        help="also run one bounded GA per arm (same seed)")
    parser.add_argument("--ga-population", type=int, default=6)
    parser.add_argument("--ga-generations", type=int, default=2)
    parser.add_argument("--ga-seed", type=int, default=20261008)
    parser.add_argument("--ga-max-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    artifact = run(args)
    artifact["_out_path"] = str(args.out)
    _write(Path(args.out), artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
