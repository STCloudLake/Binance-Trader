"""End-to-end check: a genome's ``sma`` conditions are evaluated, not dropped.

The defect
----------
``core.ga.genome.ALWAYS_AVAILABLE_COLUMNS`` declared ``sma`` available, the GA
condition pools *and* the decoder's own sanitiser fallback emit ``close > sma`` /
``close < sma``, but ``compute_all`` wrote the column only for the ``sma``
indicator gene.  With the gene off, ``evaluate_condition`` answered

    Condition rejected or unevaluable: 'close < sma' — unknown column/identifier: sma

with an all-False mask, so the genome was scored as if it declared fewer
conditions than it does (live GA job ``ga_0a442907``: 6 such warnings; its
champion entered on ``close > sma`` and exited on ``close < sma``, both with the
``sma`` gene off).

What this tool measures, on the GA's own evaluation path
-------------------------------------------------------
1. a genome whose entry/exit reference ``sma`` (``sma`` gene OFF) is decoded and
   evaluated with the real ``core.ga.fitness.evaluate_chromosome`` (real
   ``BacktestEngine``, real cached 1h bars, short window — no 1m timeframe);
   the run must log **zero** "Condition rejected" warnings, the warm-up frame
   must carry ``sma == close.rolling(20).mean()``, and ``close > sma`` must fire
   on both sides of the window (not vacuously all-False);
2. a gene expressing a genuinely unknown column must fail **loudly**: the decoder
   raises ``UnevaluableConditionError`` naming the column, ``evaluate_chromosome``
   returns fitness −999 with ``flag = "unevaluable_condition"``, and the
   rejection is counted (``core.ga.fitness.condition_rejection_counts``).

Everything is read from the repo; the run happens in a throwaway work directory
under the system temp dir (data/market is symlinked read-only, the strategy
loader points at a temp ``strategies/``), so the live DB and ``strategies/`` are
never touched.

    python tools/ga_condition_availability_check.py
    python tools/ga_condition_availability_check.py --symbols BTCUSDT \
        --start 2026-09-01 --end 2026-09-20 --timeframe 1h
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WORKDIR = Path(tempfile.gettempdir()) / "ga_condition_availability_check"
#: Standalone genomes the GA would never name this way.
SMA_GENOME = "sma_availability_check"
PHANTOM_GENOME = "phantom_column_check"


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _prepare_workdir(workdir: Path) -> Path:
    """Throwaway data root: real parquet symlinked in, temp strategies dir."""
    workdir = workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    (workdir / "strategies").mkdir(parents=True, exist_ok=True)
    target = workdir / "data" / "market"
    source = PROJECT_ROOT / "data" / "market"
    try:
        target.symlink_to(source, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copytree(source, target)
    return workdir


def _build_engine(workdir: Path):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import StrategyLoader

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = str(workdir / "data")
    cfg.backtest_engine_mode = "legacy"       # the per-genome ledger GA path
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    bus = EventBus()
    loader = StrategyLoader(str(workdir / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus),
                            OrderExecutor(cfg, bus))
    return engine, loader


def _genome(entry_long, name, timeframe, exit_long=("close < sma",)) -> dict:
    """Chromosome with only ``rsi`` on — the shape the defect produced.

    ``macd_histogram > 0`` is sanitised away (macd off) and replaced by the
    decoder's own fallback ``close > sma``; ``close < sma`` is the
    ``EXIT_CONDITION_POOL["long"]`` template.  Either way the ``sma`` gene is off,
    which is exactly the case that used to lose its conditions.
    """
    import core.ga.genome as G
    from core.strategy.loader import MLConfig, StrategyConfig

    config = StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=[timeframe],
        indicators={"rsi": {"period": 14, "source": "close"}},
        entry_conditions={"long": list(entry_long), "short": ["rsi > 70"]},
        exit_conditions={"long": list(exit_long), "short": ["rsi < 30"]},
        ml_config=MLConfig(enabled=False),
    )
    return G.strategy_to_chromosome(config)


def _warmup_frame(market_dir: Path, symbol: str, timeframe: str,
                  start: str, end: str):
    """The raw frame the engine hands to ``compute_all`` (warm-up included)."""
    from core.backtest.data_feeder import DataFeeder

    feeder = DataFeeder(str(market_dir), [symbol], [timeframe], start, end)
    feeder.load()
    return feeder.get_all_data_for_symbol(symbol, timeframe)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2026-09-01")
    parser.add_argument("--end", default="2026-09-20")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    args = parser.parse_args()

    symbols = _split(args.symbols)
    workdir = _prepare_workdir(Path(args.workdir))

    from loguru import logger

    warnings: list[str] = []
    sink_id = logger.add(lambda message: warnings.append(message.record["message"]),
                         level="WARNING", format="{message}")

    import core.ga.genome as G
    from core.ga import fitness as F
    from core.strategy import indicators as IND
    from core.strategy.indicators import compute_all, evaluate_condition

    # `UnevaluableConditionError`, `ALWAYS_DERIVED_SMA_PERIOD`,
    # `condition_failure_log` and `condition_rejection_counts` ship WITH the fix.
    # They are resolved defensively here so this exact tool can also be run
    # against the pre-fix revision (`git checkout` the three files → run → restore)
    # to reproduce the "before" behaviour, instead of dying on a missing name.
    ALWAYS_DERIVED_SMA_PERIOD = getattr(IND, "ALWAYS_DERIVED_SMA_PERIOD", 20)
    UnevaluableConditionError = getattr(G, "UnevaluableConditionError", None)
    condition_failure_log = getattr(IND, "condition_failure_log",
                                    lambda: frozenset())
    rejection_counts = getattr(F, "condition_rejection_counts", lambda: {})

    failures: list[str] = []
    started = time.perf_counter()
    engine, loader = _build_engine(workdir)

    print("=" * 78)
    print("GA condition-availability check — `sma` genome vs unknown-column genome")
    print(f"  window   : {args.start} → {args.end}  ({args.timeframe}, "
          f"{', '.join(symbols)})")
    print(f"  workdir  : {workdir}")
    print("=" * 78)

    # ── 1. the sma genome ────────────────────────────────────────────────
    sma_genome = _genome(["macd_histogram > 0"], SMA_GENOME, args.timeframe)
    decoded = G.chromosome_to_strategy(sma_genome)
    print("\n[1] decoded genome")
    print(f"    indicators            : {sorted(decoded.indicators)}"
          f"   (sma gene OFF: {'sma' not in decoded.indicators})")
    print(f"    entry_conditions.long : {decoded.entry_conditions['long']}")
    print(f"    exit_conditions.long  : {decoded.exit_conditions['long']}")
    if "sma" not in decoded.entry_conditions["long"][0]:
        failures.append("the genome does not reference sma — check the fixture")
    if "sma" in decoded.indicators:
        failures.append("the sma gene is ON — the defect case is not covered")

    # Baseline BEFORE any condition of this genome is evaluated, so both the
    # warning stream and the evaluator's failure log are compared end to end.
    before_warnings = len(warnings)
    before_failures = condition_failure_log()

    frame = _warmup_frame(PROJECT_ROOT / "data" / "market", symbols[0],
                          args.timeframe, args.start, args.end)
    if frame is None or len(frame) == 0:
        print("    no cached bars in this window — abort")
        return 2
    evaluated = compute_all(frame.copy(), decoded.indicators)
    expected_sma = evaluated["close"].rolling(ALWAYS_DERIVED_SMA_PERIOD).mean()
    has_sma = "sma" in evaluated.columns
    matches = bool(has_sma and evaluated["sma"].equals(expected_sma))
    entries = evaluate_condition(evaluated, decoded.entry_conditions["long"][0])
    exits = evaluate_condition(evaluated, decoded.exit_conditions["long"][0])
    print(f"    bars evaluated        : {len(evaluated)}")
    print(f"    'sma' in frame        : {has_sma}")
    print(f"    sma == SMA(close,{ALWAYS_DERIVED_SMA_PERIOD}) : {matches}")
    print(f"    entry 'close > sma' fires {int(entries.sum())}/{len(entries)} bars, "
          f"exit 'close < sma' fires {int(exits.sum())}/{len(exits)} bars")
    if not has_sma:
        failures.append("compute_all did not produce the `sma` column")
    if not matches:
        failures.append("the `sma` column is not SMA(close, 20)")
    if not entries.any() or not exits.any():
        failures.append("the sma conditions are vacuously all-False")

    before_warnings = len(warnings)
    before_failures = condition_failure_log()
    result = F.evaluate_chromosome(sma_genome, symbols, args.start, args.end,
                                   engine, loader)
    rejected = [w for w in warnings[before_warnings:]
                if "Condition rejected" in w]
    new_failures = condition_failure_log() - before_failures
    print("\n[2] GA fitness evaluation of the `sma` genome "
          "(core.ga.fitness.evaluate_chromosome)")
    print(f"    fitness={result.get('fitness')}  trades={result.get('trade_count')}"
          f"  win_rate={result.get('win_rate')}  sharpe={result.get('sharpe')}"
          f"  max_dd={result.get('max_dd')}")
    print(f"    'Condition rejected' warnings : {len(rejected)}")
    print(f"    unevaluable-condition log     : {sorted(new_failures) or 'none'}")
    if "error" in result:
        failures.append(f"the sma genome failed to evaluate: {result['error']}")
    if rejected or new_failures:
        failures.append("an sma condition was still rejected at evaluation time")
    if not result.get("trade_count"):
        failures.append("the sma genome produced no trades — weak evidence")

    # ── 2. the unknown-column genome ─────────────────────────────────────
    phantom_genome = _genome(["phantom_signal > 2"], PHANTOM_GENOME,
                             args.timeframe)
    print("\n[3] genome with a genuinely unknown column ('phantom_signal > 2')")
    refusal: Exception | None = None
    try:
        G.chromosome_to_strategy(phantom_genome)
    except Exception as exc:                      # noqa: BLE001 — reported below
        refusal = exc
    if refusal is None:
        failures.append("chromosome_to_strategy accepted an unknown column")
        print("    decoder               : ACCEPTED (BAD)")
    else:
        print(f"    decoder raises        : {type(refusal).__name__}")
        print(f"    reason                : {refusal}")
        if UnevaluableConditionError is None or not isinstance(
                refusal, UnevaluableConditionError):
            failures.append("the decoder refused the genome for the wrong "
                            f"reason: {type(refusal).__name__}")
        if "phantom_signal" not in str(refusal):
            failures.append("the refusal does not name the offending column")

    counts_before = rejection_counts()
    before_failures = len(warnings)
    phantom_result = F.evaluate_chromosome(phantom_genome, symbols, args.start,
                                           args.end, engine, loader)
    counts_after = rejection_counts()
    logged = [w for w in warnings[before_failures:] if "REJECTED" in w]
    print(f"    fitness={phantom_result.get('fitness')}"
          f"  flag={phantom_result.get('flag')}")
    print(f"    error                 : {phantom_result.get('error')}")
    print(f"    counted rejections    : "
          f"{sum(counts_after.values()) - sum(counts_before.values())} "
          f"(total this process: {sum(counts_after.values())})")
    print(f"    rejection log lines   : {len(logged)}")
    if phantom_result.get("fitness") != -999:
        failures.append("the unknown-column genome was not scored -999")
    if phantom_result.get("flag") != "unevaluable_condition":
        failures.append("the unknown-column genome carries no rejection flag")
    if not logged:
        failures.append("the rejection was not logged")
    if sum(counts_after.values()) - sum(counts_before.values()) != 1:
        failures.append("the rejection was not counted exactly once")

    # ── 3. the batched path, on the real engine ──────────────────────────
    # The GA's shipped path is chunked (`evaluate_population_batch`), so the
    # rejection has to survive a real chunk: the bad genome is scored −999, the
    # good one still trades, and the chunk keeps one strategy per genome (the
    # rejected slot held by a never-trading placeholder, which also pins the
    # engine's per-genome slot divisor).
    print("\n[4] chunked batch path (evaluate_population_batch, real engine)")
    import copy as _copy

    batch_population = [_copy.deepcopy(sma_genome), _copy.deepcopy(phantom_genome)]
    before_failures = len(warnings)
    F.evaluate_population_batch(batch_population, symbols, args.start, args.end,
                                engine, loader, batch_size=2, max_workers=1)
    good = batch_population[0].get("fitness_result", {})
    bad = batch_population[1].get("fitness_result", {})
    print(f"    genome[0] (sma)      : fitness={good.get('fitness')}"
          f"  trades={good.get('trade_count')}  flag={good.get('flag')!r}")
    print(f"    genome[1] (phantom)  : fitness={bad.get('fitness')}"
          f"  flag={bad.get('flag')}")
    if bad.get("flag") != "unevaluable_condition" or bad.get("fitness") != -999:
        failures.append("the chunked path did not isolate the rejected genome")
    if not good.get("trade_count"):
        failures.append("the surviving genome of the chunk did not trade")
    if any("Condition rejected" in w for w in warnings[before_failures:]):
        failures.append("a condition was rejected inside the chunked engine pass")

    elapsed = time.perf_counter() - started
    logger.remove(sink_id)
    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} problem(s)) in {elapsed:.1f}s")
        for problem in failures:
            print(f"  - {problem}")
        return 1
    print(f"RESULT: PASS — sma conditions evaluate, unknown columns fail loudly "
          f"({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
