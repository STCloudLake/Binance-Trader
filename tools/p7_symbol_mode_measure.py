"""P7-S2 evidence — does evolving one population PER SYMBOL beat one pooled basket?

The operator's second hypothesis (P7): a signal that is strong on one symbol is
**averaged away** when the GA scores every candidate on the whole basket (one
pooled equity curve), so specialising the search per symbol should improve the
out-of-sample result.  This tool measures that hypothesis on the **real cached
parquet**, with the production engine, the production scorer and the production
publication gate, and reports the answer whether or not it is the hoped-for one.

Design (bounded, reproducible, read-only)
-----------------------------------------
* Two arms, **same window / symbols / seed(s)**: ``pooled``
  (``symbol_mode: "pooled"``, the default and the pre-S2 search) and
  ``per_symbol`` (``symbol_mode: "per_symbol"``, one independent GA population per
  symbol).  Both go through ``GAStrategyEvolver.evolve`` unchanged — this tool
  stubs nothing.
* The gate benchmark is the shipped ``exposure_matched``, so "alpha" is
  like-for-like in both arms.
* Reported per champion: the **out-of-sample** alpha versus ``exposure_matched``,
  DSR with the run's **honest trial count** (``provenance["n_trials"]``:
  ``len(symbols) × population × generations`` for a per-symbol run — every variant
  really tried), trade count and time-in-market.
* The pooled champion is additionally scored on **each symbol alone**, so the
  comparison is apples-to-apples: "pooled champion on BTCUSDT" versus "per-symbol
  BTCUSDT champion on BTCUSDT".
* Nothing is written outside ``--out`` and a temporary root: the market cache is
  symlinked read-only, and the GA's strategy files, checkpoint and trial ledger all
  live under that temp root.  No ``data/binance_trader.db``, no ``strategies/``.

Usage::

    python tools/p7_symbol_mode_measure.py --population 4 --generations 2 \
        --symbols BTCUSDT ETHUSDT --timeframe 1h \
        --train-start 2025-11-01 --train-end 2026-02-01 \
        --oos-start 2026-02-01 --oos-end 2026-06-01 \
        --seeds 20261011 20261012 --out %TEMP%\\p7_symbol_mode.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ga.benchmark import EXPOSURE_MATCHED          # noqa: E402
from core.ga.evolver import PER_SYMBOL, POOLED          # noqa: E402

DEFAULT_OUT = Path(tempfile.gettempdir()) / "p7_symbol_mode.json"


# ── isolated engine stack (never touches the repository's data/) ────────

def _prepare_root(name: str) -> Path:
    """A temp data root whose ``market`` cache is the repository's, read-only.

    The root is wiped first: each arm must run from a **clean deployment** (no
    leftover checkpoint and, decisively, no leftover trial ledger), otherwise the
    second arm of a seed would be deflated by the first arm's trials and the two
    arms' DSRs would not be comparable.  Each arm's ``provenance.n_trials`` is
    then exactly the evaluations *that arm* performed.
    """
    root = Path(tempfile.gettempdir()) / f"p7_symbol_mode_{name}"
    import shutil
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "strategies").mkdir(parents=True, exist_ok=True)
    target = root / "data" / "market"
    source = ROOT / "data" / "market"
    try:
        target.symlink_to(source, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copytree(source, target)
    return root


def _engine_stack(root: Path):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager

    Config._instance = None
    config = Config.load("sim")
    config.data_dir = str(root / "data")
    # The GA evaluation contract: legacy engine, no ML, no live order book.
    config.backtest_engine_mode = "legacy"
    config.backtest_ml_enabled = False
    config.backtest_live_spread_enabled = False
    bus = EventBus()
    engine = BacktestEngine(config, None, RiskManager(config, bus),
                            OrderExecutor(config, bus))
    return config, engine


def _loader(root: Path):
    from core.strategy.loader import StrategyLoader

    return StrategyLoader(str(root / "strategies"))


# ── one GA arm ─────────────────────────────────────────────────────────

def _run_arm(root: Path, mode: str, symbols, seed: int, population: int,
             generations: int, timeframe: str, train_start: str, train_end: str,
             oos_end: str, max_workers: int, early_stop: int) -> dict:
    """Run one bounded GA in *mode* and return its result (champions included)."""
    from core.ga.evolver import GAStrategyEvolver, GARunConfig

    config, engine = _engine_stack(root)
    loader = _loader(root)
    run_cfg = GARunConfig(
        population_size=population, generations=generations,
        elite_count=max(2, population // 4), immigrant_count=max(2, population // 4),
        max_workers=max_workers, seed=seed, keep_checkpoint=False,
        timeframe_pool=[timeframe], benchmark_mode=EXPOSURE_MATCHED,
        symbol_mode=mode, early_stop_generations=early_stop)
    evolver = GAStrategyEvolver(engine, loader, run_cfg)
    started = time.time()
    # ``evolve(symbols, date_start, date_end, validation_start)`` trains on
    # [date_start, validation_start] and validates on [validation_start, date_end]
    # — so the OOS END is the run's ``date_end``.
    result = evolver.evolve(list(symbols), train_start, oos_end,
                            validation_start=train_end, seed=seed,
                            window_key=f"{train_start}~{train_end}|{mode}")
    result["_seconds"] = round(time.time() - started, 1)
    return result


def _champions(result: dict) -> list[dict]:
    """Normalise the two result shapes into one list of per-champion dicts."""
    if "champions" in result:
        return list(result["champions"])
    return [result]


# ── out-of-sample scoring of a champion, on a chosen basket ─────────────

def _score_champion(champion: dict, symbols, oos_start: str, oos_end: str,
                    root: Path, n_trials: int) -> dict:
    """Score ONE champion on *symbols* over the OOS window (production scorer).

    The champion's chromosome is rebuilt from its published ``StrategyConfig``
    (the same encode path the GA uses to seed a run) plus the
    ``condition_logic`` gene the decoder needs, so the artefact is scored exactly
    as it was published.  ``n_trials`` is the run's honest count.
    """
    from core.ga.fitness import evaluate_chromosome
    from core.ga.genome import strategy_to_chromosome
    from core.strategy.loader import StrategyConfig

    config, engine = _engine_stack(root)
    loader = _loader(root)
    try:
        strategy = StrategyConfig(**champion["champion_config"])
        chromosome = strategy_to_chromosome(strategy)
        chromosome["condition_logic"] = (
            (champion.get("provenance") or {}).get("condition_logic") or "or")
    except Exception as exc:  # pragma: no cover - a malformed artefact
        return {"error": f"decode: {exc}"}
    scored = evaluate_chromosome(
        chromosome, list(symbols), oos_start, oos_end, engine, loader,
        ga_loader=loader, n_trials=n_trials, benchmark_mode=EXPOSURE_MATCHED)
    # ── Alpha versus the exposure-matched benchmark ──
    # ``score_stats`` computes it correctly, but ``evaluate_chromosome`` then
    # overwrites ``alpha_vs_benchmark_pct`` with 0.0: it calls
    # ``benchmark_result_fields(stats, score)`` with the RAW stats dict, and the
    # raw dict never carries the key (``score_stats`` writes to its own copy).
    # The batch paths rebind ``stats = score_stats(...)`` and are unaffected, so
    # only the single-chromosome / OOS-validation path is wrong — a pre-existing
    # defect this tool must not silently inherit.  Alpha is therefore recomputed
    # here as ``total_return_pct - benchmark_pct`` (the same arithmetic
    # ``score_stats`` performs), and the scorer's own (wrong) value is reported
    # next to it so the defect stays visible.
    total_return = scored.get("total_return")
    benchmark = scored.get("benchmark_pct")
    alpha = (None if (total_return is None or benchmark is None)
             else round(float(total_return) - float(benchmark), 4))
    return {
        "symbols": list(symbols),
        "trades": scored.get("trade_count"),
        "total_return_pct": total_return,
        "benchmark_pct": benchmark,
        "alpha_vs_exposure_matched_pct": alpha,
        "scorer_alpha_vs_benchmark_pct": scored.get("alpha_vs_benchmark_pct"),
        "dsr": scored.get("dsr"),
        "sharpe": scored.get("sharpe"),
        "profit_factor": scored.get("profit_factor"),
        "time_in_market_pct": scored.get("strategy_time_in_market_pct"),
        "benchmark_time_in_market_pct": scored.get("benchmark_time_in_market_pct"),
        "max_dd_pct": scored.get("max_dd"),
        "n_trials": n_trials,
    }


# ── the measurement ────────────────────────────────────────────────────

def run(args) -> dict:
    symbols = [s.strip().upper() for s in args.symbols if s.strip()]
    seeds = [int(s) for s in args.seeds]
    artifact = {
        "script": "tools/p7_symbol_mode_measure.py",
        "windows": {"train": [args.train_start, args.train_end],
                    "out_of_sample": [args.oos_start, args.oos_end]},
        "search": {"symbols": symbols, "timeframe": args.timeframe,
                   "population": args.population, "generations": args.generations,
                   "seeds": seeds, "benchmark_mode": EXPOSURE_MATCHED,
                   "early_stop_generations": args.early_stop,
                   "fresh_trial_ledger_per_arm": True},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "known_defect_worked_around": (
            "core.ga.fitness.evaluate_chromosome overwrites "
            "alpha_vs_benchmark_pct with 0.0 (it passes the RAW stats dict to "
            "benchmark_result_fields, while score_stats writes the value to its "
            "own copy); the batch paths rebind stats and are unaffected. "
            "This tool recomputes alpha as total_return_pct - benchmark_pct and "
            "keeps the scorer's value in 'scorer_alpha_vs_benchmark_pct'."),
        "runs": [],
        "champions": [],
        "summary": {},
    }
    evaluations = args.population * args.generations
    print(f"P7-S2 measurement: {symbols} {args.timeframe}, "
          f"train {args.train_start}~{args.train_end}, "
          f"OOS {args.oos_start}~{args.oos_end}", flush=True)
    print(f"per arm per seed: population={args.population} × "
          f"generations={args.generations} = {evaluations} evaluations "
          f"(per_symbol multiplies that by len(symbols)={len(symbols)})",
          flush=True)

    for seed in seeds:
        for mode in (POOLED, PER_SYMBOL):
            root = _prepare_root(f"{mode}_{seed}")
            result = _run_arm(root, mode, symbols, seed, args.population,
                              args.generations, args.timeframe, args.train_start,
                              args.train_end, args.oos_end, args.max_workers,
                              args.early_stop)
            n_trials = int((result.get("provenance") or {}).get("n_trials") or 0)
            published = result.get("published")
            print(f"  [{mode:<10} seed={seed}] seconds={result.get('_seconds')} "
                  f"champions={len(_champions(result))} n_trials={n_trials} "
                  f"(= {evaluations}"
                  + (f" × {len(symbols)}" if mode == PER_SYMBOL else "")
                  + f") published={published}", flush=True)
            artifact["runs"].append({
                "mode": mode, "seed": seed, "seconds": result.get("_seconds"),
                "n_trials": n_trials, "published": published,
                "generations": result.get("generations"),
                "population_size": args.population,
                "expected_evaluations": (evaluations * len(symbols)
                                         if mode == PER_SYMBOL else evaluations),
                "champion_names": [c.get("champion_name")
                                   for c in _champions(result)],
                "search": ((result.get("provenance") or {})
                           .get("trials", {}).get("search")),
            })

            for champion in _champions(result):
                if "error" in champion:
                    artifact["champions"].append({
                        "mode": mode, "seed": seed, "symbol": None,
                        "error": champion["error"]})
                    continue
                symbol = champion.get("champion_symbol")
                rows = []
                if mode == PER_SYMBOL:
                    rows.append((symbol, _score_champion(
                        champion, [symbol], args.oos_start, args.oos_end, root,
                        n_trials)))
                else:
                    rows.append((",".join(symbols), _score_champion(
                        champion, symbols, args.oos_start, args.oos_end, root,
                        n_trials)))
                    # The same-basket comparison: the pooled champion scored on
                    # each symbol ALONE (what a per-symbol champion is judged on).
                    if args.split_pooled:
                        for one in symbols:
                            rows.append((one, _score_champion(
                                champion, [one], args.oos_start, args.oos_end,
                                root, n_trials)))
                for label, scored in rows:
                    row = {"mode": mode, "seed": seed, "symbol": symbol,
                           "evaluation_basket": scored.get("symbols", label),
                           "champion_name": champion.get("champion_name"),
                           "train_fitness": champion.get("fitness"),
                           "train_trades": champion.get("trade_count"),
                           "train_dsr": (champion.get("dsr") or {}).get("dsr"),
                           "published": champion.get("published"),
                           # The evolver's OWN OOS block for the same champion
                           # (a cross-check of the re-evaluation below; it lacks
                           # the strategy's time-in-market, which is why the tool
                           # scores the champion itself).
                           "evolver_validation": champion.get("validation"),
                           "oos": scored}
                    artifact["champions"].append(row)
                    oos = scored
                    print(f"     {label:<16} oos_trades={oos.get('trades')} "
                          f"ret={oos.get('total_return_pct')} "
                          f"bench={oos.get('benchmark_pct')} "
                          f"alpha={oos.get('alpha_vs_exposure_matched_pct')} "
                          f"dsr={oos.get('dsr')} "
                          f"tim={oos.get('time_in_market_pct')}", flush=True)
            artifact["summary"] = _summarise(artifact, symbols, evaluations)
            _write(Path(args.out), artifact)

    artifact["summary"] = _summarise(artifact, symbols, evaluations)
    artifact["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write(Path(args.out), artifact)
    print("")
    _print_summary(artifact, symbols)
    return artifact


def _column(rows, key):
    return [r["oos"].get(key) for r in rows
            if isinstance(r.get("oos"), dict) and r["oos"].get(key) is not None]


def _median(values):
    return round(statistics.median(values), 4) if values else None


def _mean(values):
    return round(statistics.mean(values), 4) if values else None


def _arm_stats(rows) -> dict:
    alpha = _column(rows, "alpha_vs_exposure_matched_pct")
    dsr = _column(rows, "dsr")
    return {
        "champions": len(rows),
        "oos_trades_median": _median(_column(rows, "trades")),
        "oos_alpha_median": _median(alpha),
        "oos_alpha_mean": _mean(alpha),
        "oos_alpha_positive": sum(1 for v in alpha if v > 0),
        "oos_dsr_median": _median(dsr),
        "oos_dsr_positive": sum(1 for v in dsr if v > 0),
        "oos_time_in_market_median": _median(_column(rows, "time_in_market_pct")),
        "oos_return_median": _median(_column(rows, "total_return_pct")),
        "published": sum(1 for r in rows if r.get("published")),
    }


def _summarise(artifact: dict, symbols, evaluations: int) -> dict:
    """Paired (same seed, same symbol) comparison plus the two arm summaries."""
    by_mode: dict[str, list] = {POOLED: [], PER_SYMBOL: []}
    for row in artifact["champions"]:
        if "error" in row or not isinstance(row.get("oos"), dict):
            continue
        if "error" in (row.get("oos") or {}):
            continue
        by_mode.setdefault(row["mode"], []).append(row)

    per_symbol_rows = [r for r in by_mode.get(PER_SYMBOL, [])
                       if r["mode"] == PER_SYMBOL]
    pooled_basket_rows = [r for r in by_mode.get(POOLED, [])
                          if len(r["evaluation_basket"]) == len(symbols)]
    # Same-symbol pairs: (pooled champion scored on symbol X) vs
    # (per-symbol champion for X), matched on seed and symbol.
    pooled_by_key = {(r["seed"], r["evaluation_basket"][0]): r
                     for r in by_mode.get(POOLED, [])
                     if len(r["evaluation_basket"]) == 1}
    pairs = []
    for row in per_symbol_rows:
        key = (row["seed"], row["evaluation_basket"][0])
        if key in pooled_by_key:
            pairs.append((pooled_by_key[key], row))

    def _delta(before, after, key):
        a = before["oos"].get(key)
        b = after["oos"].get(key)
        if a is None or b is None:
            return None
        return round(float(b) - float(a), 4)

    deltas = [_delta(b, a, "alpha_vs_exposure_matched_pct") for b, a in pairs]
    deltas = [d for d in deltas if d is not None]
    dsr_deltas = [d for d in (_delta(b, a, "dsr") for b, a in pairs)
                  if d is not None]
    tim_deltas = [d for d in (_delta(b, a, "time_in_market_pct") for b, a in pairs)
                  if d is not None]
    trade_deltas = [d for d in (_delta(b, a, "trades") for b, a in pairs)
                    if d is not None]
    return {
        "symbols": list(symbols),
        "expected_evaluations_per_arm": {
            POOLED: evaluations,
            PER_SYMBOL: evaluations * len(symbols)},
        "pooled_champion_on_the_basket": _arm_stats(pooled_basket_rows),
        "per_symbol_champions": _arm_stats(per_symbol_rows),
        "same_symbol_pairs": len(pairs),
        "paired_per_symbol_minus_pooled": {
            "oos_alpha_delta_median": _median(deltas),
            "oos_alpha_delta_mean": _mean(deltas),
            "oos_alpha_improved": sum(1 for d in deltas if d > 0),
            "oos_alpha_worsened": sum(1 for d in deltas if d < 0),
            "oos_dsr_delta_median": _median(dsr_deltas),
            "oos_trades_delta_median": _median(trade_deltas),
            "oos_time_in_market_delta_median": _median(tim_deltas),
        },
        "caveats": [
            "alpha is in PERCENTAGE POINTS over the OOS window, against the "
            "exposure-matched basket, and is not annualised",
            "alpha is recomputed as total_return_pct - benchmark_pct because "
            "evaluate_chromosome reports 0.0 for alpha_vs_benchmark_pct (see "
            "known_defect_worked_around)",
            "a per-symbol run's DSR is deflated by len(symbols) x population x "
            "generations (every variant it really tried), so its hurdle is "
            "strictly HIGHER than the pooled run's - the comparison is "
            "conservative against per_symbol",
            "the same-symbol table scores the pooled champion on one symbol at a "
            "time; those split evaluations are diagnostics of an already-selected "
            "champion and reuse the run's n_trials",
            "elite/immigrant counts leave population_size - elite_count children "
            "per generation: keep population_size above elite_count + "
            "immigrant_count or the search is only elites + random immigrants",
            "both arms reseed from the same seed, so a per-symbol arm shares its "
            "initial population with the pooled arm; the FIRST arm can therefore "
            "select the same genome as the pooled run (observed: 0 differing YAML "
            "lines for the BTC pairs of two seeds) and its same-symbol pair is then "
            "degenerate (delta = 0 by construction, no information). Check the "
            "champion artefacts before reading the pair table",
        ],
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str),
                    encoding="utf-8")


def _fmt(value) -> str:
    return "n/a" if value is None else str(value)


def _print_summary(artifact: dict, symbols) -> None:
    summary = artifact["summary"]
    print("=== P7-S2 pooled vs per-symbol — out-of-sample summary ===")
    header = (f"{'arm':<26}{'champs':>7}{'trades':>8}{'alpha_med':>11}"
              f"{'alpha_mean':>12}{'alpha>0':>9}{'dsr_med':>10}{'dsr>0':>7}"
              f"{'tim_med':>9}")
    print(header)
    print("-" * len(header))
    for label, key in (("pooled (basket champion)", "pooled_champion_on_the_basket"),
                       ("per_symbol (per symbol)", "per_symbol_champions")):
        stats = summary[key]
        print(f"{label:<26}{_fmt(stats['champions']):>7}"
              f"{_fmt(stats['oos_trades_median']):>8}"
              f"{_fmt(stats['oos_alpha_median']):>11}"
              f"{_fmt(stats['oos_alpha_mean']):>12}"
              f"{_fmt(stats['oos_alpha_positive']):>9}"
              f"{_fmt(stats['oos_dsr_median']):>10}"
              f"{_fmt(stats['oos_dsr_positive']):>7}"
              f"{_fmt(stats['oos_time_in_market_median']):>9}")
    pair = summary["paired_per_symbol_minus_pooled"]
    print(f"\nsame-symbol pairs (per_symbol - pooled): {summary['same_symbol_pairs']}")
    print(f"  d_alpha median={_fmt(pair['oos_alpha_delta_median'])} "
          f"mean={_fmt(pair['oos_alpha_delta_mean'])} "
          f"improved={_fmt(pair['oos_alpha_improved'])} "
          f"worsened={_fmt(pair['oos_alpha_worsened'])}")
    print(f"  d_dsr median={_fmt(pair['oos_dsr_delta_median'])} "
          f"d_trades median={_fmt(pair['oos_trades_delta_median'])} "
          f"d_time-in-market median={_fmt(pair['oos_time_in_market_delta_median'])}")

    pooled = summary["pooled_champion_on_the_basket"]
    per = summary["per_symbol_champions"]
    pair_alpha = pair["oos_alpha_delta_mean"]
    dsr_positive = (per["oos_dsr_positive"] or 0)
    if pooled["champions"] == 0 or per["champions"] == 0:
        verdict = "no evaluable champions — insufficient evidence"
    elif dsr_positive == 0:
        verdict = ("no risk-adjusted edge in either arm (no OOS DSR > 0): "
                   "per-symbol specialisation does NOT earn its extra trials")
    elif pair_alpha is not None and pair_alpha > 0:
        verdict = ("per-symbol specialisation improves the same-symbol OOS alpha "
                   "on average — but check the DSR column before claiming an edge")
    else:
        verdict = ("per-symbol specialisation does NOT improve the same-symbol "
                   "OOS alpha on average")
    print(f"\nverdict: {verdict}")
    for caveat in summary["caveats"]:
        print(f"  note: {caveat}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--train-start", default="2025-11-01")
    parser.add_argument("--train-end", default="2026-02-01")
    parser.add_argument("--oos-start", default="2026-02-01")
    parser.add_argument("--oos-end", default="2026-06-01")
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seeds", nargs="*", type=int, default=[20261011])
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--early-stop", type=int, default=50,
                        help="generations without improvement before stopping "
                             "(default 50 = effectively off for short runs)")
    parser.add_argument("--split-pooled", action="store_true",
                        help="also score each pooled champion on each symbol alone "
                             "(the same-symbol comparison table)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    return 0 if run(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
