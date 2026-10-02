"""P7-S4 evidence — the **composite** (orchestrator + sub-strategies) out of sample.

P7-S1 measured single conditioned strategies (all negative), P7-S3 shipped a
replayable ``RegimeOrchestrator`` (also negative in its bounded preview).  This
tool evaluates the two **as one strategy** — the defining stage of P7 — with the
controls and the benchmarks the frozen plan requires, and prints the real numbers
even when they are negative (``docs/overhaul/P7_REGIME_PLAN.md`` §3 S4).

What it does, in one bounded deterministic run
----------------------------------------------
1. **Cohort + selection on the TRAIN window only.**  ``--selection-population``
   genomes per symbol are drawn from one seed and evaluated on the training
   window with the production engine; the best genome per symbol (by the
   production fitness) is selected.  Nothing about the out-of-sample window is
   read before that selection is frozen.
2. **The orchestrator rules are fixed on the TRAIN window.**  Each selected
   strategy is mapped to the causal regime label with the best training-window
   PnL for that strategy (ties broken by ``GATE_REGIME_LABELS`` order), written to
   a JSON file, fingerprinted with :func:`core.ai.orchestrator.rules_fingerprint`
   and never touched again.  ``kill_switch.consecutive_losses`` is pinned to 0, so
   the enable/disable timeline depends on the market state only — not on the
   realized trade stream (which would make the controls incomparable).
3. **Fixed train-window weights.**
   ``w_s = clip(mean(amount_usdt over s's TRAIN trades) / initial_balance, 0, 1)``
   normalised to sum to 1 (``core.ai.composite``; the same *deployed-capital*
   rule the gate benchmark already uses).  The weights are printed with the
   artifact and are **not** recomputed out of sample.
4. **Three variants on the OOS window** — always-on (all strategies enabled),
   **random** start/stop (fixed seed, per-strategy enable probability matched to
   the orchestrator's realized enable fraction) and orchestrated.  Each variant's
   composite curve is folded from the same engine trade streams at the same fixed
   weights, so the comparison isolates the orchestrator's **selection**, not an
   exposure cut.
5. **Matched benchmarks** over the composite's own in-market intervals:
   ``exposure_matched`` (``core.ga.benchmark`` reused) and plain ``buy_hold`` of
   the same basket inside the same intervals.
6. **One-shot holdout** (``core.ai.holdout``): the first evaluation of a window
   claims it; a second one is refused unless ``--allow-holdout-reuse`` is passed,
   in which case the reuse is recorded as overfitting in the artifact.
7. **Trials counted**: every sub-strategy candidate (both windows), every arm
   (always-on + orchestrated + each random draw) and every orchestrator rule set
   tried on the training window deflate the composite DSR.
8. **Usability bar**: a composite is never called usable below **100** OOS
   composite trades (``ml.gate_min_trades``), nor with ``DSR <= 0`` — stated in
   the printed output and in the artifact.

Usage::

    python tools/p7_composite_measure.py
    python tools/p7_composite_measure.py --selection-population 2 --symbols BTCUSDT ETHUSDT
    python tools/p7_composite_measure.py --out %TEMP%\\p7_composite.json
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd                                            # noqa: E402

from core.ai import holdout as holdout_mod                     # noqa: E402
from core.ai.composite import (                                # noqa: E402
    DEFAULT_USABILITY_MIN_TRADES,
    CompositeTrials,
    build_composite_spec,
    composite_fund,
    composite_metrics,
    enable_fractions,
    filter_trades_by_timeline,
    flatten_trades,
    matched_benchmarks,
    random_enabled_stamps,
    regime_contribution,
)
from core.ai.orchestrator import (                             # noqa: E402
    RegimeOrchestrator,
    rules_fingerprint,
)
from core.ga.benchmark import EXPOSURE_MATCHED                  # noqa: E402
from core.ga.fitness import (isolated_eval_kwargs, score_stats,   # noqa: E402
                             stats_from_engine_result)
from core.strategy.regime_causal import (GATE_REGIME_LABELS,      # noqa: E402
                                         causal_regime_table)

DEFAULT_OUT = Path(tempfile.gettempdir()) / "p7_composite.json"
DEFAULT_ORCHESTRATOR = Path(tempfile.gettempdir()) / "p7_composite_orchestrator.json"
HOLDOUT_ID = "p7-s4-composite-oos"
INITIAL_BALANCE = 10_000.0


# ── engine / data plumbing (same isolation contract as the S1 measure) ──

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


def _cohort(population: int, seed: int, timeframe: str):
    """A fixed cohort of chromosomes (timeframe pinned, RNG restored after)."""
    from core.ga.genome import random_chromosome

    state = random.getstate()
    random.seed(seed)
    chroms = [random_chromosome(f"p7c_base_{i}") for i in range(population)]
    random.setstate(state)
    for chrom in chroms:
        for gene in chrom.get("categorical", []):
            if gene.name == "timeframes":
                gene.value = timeframe
    return chroms


def _decode(chrom, name: str):
    from core.ga.genome import chromosome_to_strategy

    strategy = chromosome_to_strategy(chrom)
    strategy.name = name
    if strategy.ml_config:
        strategy.ml_config.enabled = False
    return strategy


def _run(engine, strategies, symbols, start, end):
    return engine.run_with_exit_evaluation(
        strategies=strategies, symbols=list(symbols), date_start=start,
        date_end=end, initial_balance=INITIAL_BALANCE, mode="full",
        simulate_ai_weights=False, ml_engine="lightgbm", use_live_spread=False,
        benchmark_mode=EXPOSURE_MATCHED, **isolated_eval_kwargs())


def _ledger(result, name: str) -> dict:
    """``per_strategy_equity`` entry for *name*, tolerating the engine's suffixes."""
    per = result.get("per_strategy_equity") or {}
    if name in per:
        return per[name]
    for key, entry in per.items():
        if str(key).startswith(f"{name}_"):
            return entry
    return {}


def _score(result, name: str, chrom, n_trials: int) -> dict:
    stats = stats_from_engine_result(result, name, INITIAL_BALANCE,
                                     chromosome=chrom)
    return score_stats(stats, chrom, n_trials=n_trials)


def _trades_of(result, name: str) -> list:
    ledger = _ledger(result, name)
    rows = ledger.get("trades")
    if rows is not None:
        return list(rows)
    return [t for t in (result.get("trades") or []) if t.get("strategy") == name]


def _equity_of(result, name: str) -> list:
    return list((_ledger(result, name) or {}).get("equity_curve") or [])


# ── step 1: select one strategy per symbol on the TRAIN window ──────────

def select_per_symbol(engine, args, symbols) -> dict:
    """``{symbol: {"chrom", "train_trades", "train_score", "strategy_name"}}``.

    The selection is a train-window function: the engine is asked for the
    population on the training window, the production scorer ranks each genome for
    that symbol, and only the winner's trade list is kept for the weighting rule.
    """
    population = max(int(args.selection_population), 1)
    cohort = _cohort(population, args.seed, args.timeframe)
    n_trials = population * len(GATE_REGIME_LABELS)
    out: dict = {}
    for symbol in symbols:
        strategies = [_decode(chrom, f"p7c_{symbol}_{i}")
                      for i, chrom in enumerate(cohort)]
        result = _run(engine, strategies, [symbol], args.train_start, args.train_end)
        if "error" in (result or {}):
            raise SystemExit(f"train evaluation failed for {symbol}: {result['error']}")
        ranked = []
        for index, chrom in enumerate(cohort):
            name = f"p7c_{symbol}_{index}"
            score = _score(result, name, chrom, n_trials)
            ranked.append((float(score["fitness"]), -index, name, chrom, score))
        ranked.sort(reverse=True)
        fitness, _neg, name, chrom, score = ranked[0]
        out[str(symbol)] = {
            "chrom": chrom, "strategy_name": name,
            "train_trades": _trades_of(result, name),
            "train_equity": _equity_of(result, name),
            "train_index": [pd.Timestamp(point["time"])
                            for point in (_equity_of(result, name) or [])],
            "train_score": _public_score(score),
            "fitness": round(fitness, 4),
            "population": population,
            "n_trials": n_trials,
        }
        print(f"  [{symbol}] selected {name}: fitness={fitness:.4f} "
              f"train_trades={score['trades']} "
              f"train_alpha={round(float(score['alpha_vs_benchmark_pct']), 4)} "
              f"(population {population} x {len(GATE_REGIME_LABELS)} labels = "
              f"{n_trials} trials)", flush=True)
    return out


def _public_score(score: dict) -> dict:
    return {"fitness": round(float(score["fitness"]), 4),
            "trades": int(score["trades"]),
            "total_return_pct": round(float(score["total_return_pct"]), 4),
            "benchmark_pct": (None if score.get("benchmark_pct") is None
                              else round(float(score["benchmark_pct"]), 4)),
            "alpha_vs_benchmark_pct": round(
                float(score["alpha_vs_benchmark_pct"]), 4),
            "sharpe": round(float(score["sharpe"]), 4),
            "deflated_sharpe": round(float(score["deflated_sharpe"]), 6),
            "max_dd_pct": round(float(score["max_dd_pct"]), 4),
            "observations": int(score["observations"]),
            "time_in_market_pct": (score.get("benchmark") or {}).get(
                "strategy_time_in_market_pct")}


# ── step 2: the orchestrator rules, fixed on the TRAIN window ───────────

def labels_for(regime_tables: dict, symbol: str, stamp):
    table = regime_tables.get(str(symbol))
    if table is None or stamp is None:
        return None
    index = table.index
    cut = int(index.searchsorted(stamp, side="right"))
    if cut <= 0:
        return None
    return str(table["regime"].iloc[cut - 1])


def _candidate_rule_sets(selected, regime_tables):
    """Every orchestrator rule set the TRAIN window is allowed to choose between.

    Per strategy: each single gate label, its best **pair** of labels, every label
    (``["all"]``) and *never* (``[]``) — the search the plan's "选择编排器只用训练
    窗" clause describes.  The chosen candidate's enable/disable timeline is then
    the input to the random control, so the random arm is matched to what the train
    window actually picked.  Returns ``(candidates, evidence)`` where each
    candidate is ``(tag, allowed_mapping, enable_fraction_map)``:
    ``enable_fraction_map`` is the *training-window* fraction of bars each rule
    would allow (a train-only statistic, never a test-window one).
    """
    per_strategy_labels: dict = {}
    evidence: dict = {}
    for symbol, entry in sorted(selected.items()):
        name = entry["strategy_name"]
        trades = entry["train_trades"]
        totals, counts = {}, {}
        for label in GATE_REGIME_LABELS:
            rows = [t for t in trades
                    if labels_for(regime_tables, symbol,
                                  pd.Timestamp(t.get("opened_at"))) == label]
            totals[label] = round(sum(float(t.get("pnl") or 0.0) for t in rows), 4)
            counts[label] = len(rows)
        ranked = sorted(GATE_REGIME_LABELS, key=lambda lab: (-totals[lab], lab))
        per_strategy_labels[name] = (ranked, totals, counts)
        evidence[name] = {"symbol": symbol,
                          "train_pnl_by_label": totals,
                          "train_trades_by_label": counts,
                          "ranked_by_train_pnl": ranked}

    candidates = []
    for candidate_index, label in enumerate(GATE_REGIME_LABELS):
        mapping = {name: [label] for name in per_strategy_labels}
        candidates.append((f"single:{label}", mapping))
    pair = {name: list(ranked[:2]) for name, (ranked, _t, _c) in
            per_strategy_labels.items()}
    candidates.append(("best_pair_per_strategy", pair))
    candidates.append(("all_regimes", {name: list(GATE_REGIME_LABELS)
                                       for name in per_strategy_labels}))
    candidates.append(("never_enable", {name: [] for name in per_strategy_labels}))
    return candidates, evidence


def _rules_from_mapping(mapping, label_counts):
    """The orchestrator rules for one candidate mapping, plus its **enable fraction**.

    ``unknown_label_action: deny`` — an unmeasured (``range_unknown``) bar must not
    be traded on a guess.  ``kill_switch.consecutive_losses: 0`` — pinned off so the
    enable timeline is a function of the market state alone, which is what makes the
    exposure-matched random control comparable (a trade-dependent latch would make
    the controls differ in more than their selection).

    The fraction is the share of the training bars whose causal label is in that
    strategy's allowed set (``0`` for an empty set, ``1`` for all five labels) —
    it is what the random control matches, so it must be computed from the labels
    the strategy may actually trade, never from a "two or more ⇒ everything" rule.
    """
    bars = max(sum(label_counts.values()), 1)
    fractions = {}
    for name, labels in mapping.items():
        allowed = [label for label in labels if label in GATE_REGIME_LABELS]
        fractions[name] = (0.0 if not allowed else
                           min(sum(label_counts.get(label, 0)
                                   for label in allowed) / bars, 1.0))
    return ({
        "enabled": True,
        "regime": {"allowed": {name: list(labels)
                               for name, labels in mapping.items()},
                   "default_action": "allow",
                   "missing_regime_action": "allow",
                   "unknown_label_action": "deny"},
        "kill_switch": {"consecutive_losses": 0, "loss_threshold": 0.0},
        "vol": {"multiple": 0.0},
        "breadth": {},
    }, fractions)


def freeze_orchestrator_rules(selected, regime_tables, train_start, train_end):
    """Pick the orchestrator rule set **on the training window only**.

    The candidate rule sets come from :func:`_candidate_rule_sets`; each one's
    composite is folded over the **training** trades at the (already fixed)
    weights and scored by the production scorer, and the highest total training
    fitness wins.  This is a train-window-only decision made before any
    out-of-sample bar is read; every candidate it considered is part of the DSR
    trial count.
    """
    candidates, evidence = _candidate_rule_sets(selected, regime_tables)
    # The random control's matching fractions are the share of TRAIN bars whose
    # causal label a strategy may trade, counted from the regime table itself (the
    # same shared clock the training timeline below uses), not from the trades —
    # counting trades would match the random arm's *opportunity* to a different
    # quantity than the orchestrator's *enabled bars*.
    clock_symbol = sorted(selected)[0]
    label_counts = _label_counts(regime_tables.get(clock_symbol))
    scored = []
    for tag, mapping in candidates:
        rules, fractions = _rules_from_mapping(mapping, label_counts)
        timeline = _train_timeline(rules, selected, regime_tables)
        trades = {entry["strategy_name"]: entry["train_trades"]
                  for entry in selected.values()}
        kept = filter_trades_by_timeline(trades, timeline)
        fitness = sum(float(_train_fitness(_entry_by_name(selected, name), rows))
                      for name, rows in kept.items())
        scored.append({"tag": tag, "mapping": mapping,
                       "fitness": round(fitness, 4),
                       "train_enable_fractions": {
                           k: round(v, 6) for k, v in sorted(fractions.items())},
                       "train_trades": {k: len(v) for k, v in sorted(kept.items())}})
    scored.sort(key=lambda row: (-row["fitness"], row["tag"]))
    winner = scored[0]
    rules, fractions = _rules_from_mapping(winner["mapping"], label_counts)
    return rules, {"chosen": winner["tag"], "winner": winner,
                   "candidates": scored,
                   "train_window": [str(train_start), str(train_end)],
                   "selection_metric": ("sum over sub-strategies of the "
                                        "train-window fitness of the trades the "
                                        "rule set would keep"),
                   "labels_by_strategy": evidence}, fractions


def _entry_by_name(selected, name):
    for entry in selected.values():
        if entry["strategy_name"] == name:
            return entry
    raise KeyError(name)


def _label_counts(table) -> dict:
    """``{label: bars}`` of a causal regime table (empty table ⇒ empty counts)."""
    if table is None or not hasattr(table, "__getitem__") or "regime" not in getattr(
            table, "columns", []):
        return {}
    try:
        counts = table["regime"].astype(str).value_counts().to_dict()
    except Exception:
        return {}
    return {str(label): int(value) for label, value in counts.items()}


def _train_fitness(entry, rows) -> float:
    """Train-window fitness of one sub-strategy restricted to *rows*."""
    from core.ga.fitness import stats_from_trades
    score = score_stats(stats_from_trades(list(rows), entry["train_equity"],
                                          INITIAL_BALANCE),
                        entry["chrom"], n_trials=int(entry["n_trials"]))
    return float(score["fitness"])


def _train_timeline(rules, selected, regime_tables) -> dict:
    """``{stamp: {name: enabled}}`` over the TRAIN window bars (for the choice).

    The shared clock is the first selected symbol's training bars (with one
    symbol per strategy the label is that symbol's own; with several it is the
    first symbol's, recorded in the artifact).
    """
    from core.ai.orchestrator import TimelineEvent

    symbol = selected and sorted(selected)[0]
    stamps = list((selected.get(symbol) or {}).get("train_index") or [])
    names = sorted(entry["strategy_name"] for entry in selected.values())
    events = [TimelineEvent("decide", at=stamp,
                            label=labels_for(regime_tables, symbol, stamp))
              for stamp in stamps]
    timeline = RegimeOrchestrator.replay(rules, events, names=names)
    return {pd.Timestamp(row["at"]): {item["strategy"]: bool(item["enabled"])
                                      for item in row["rows"]}
            for row in timeline}


# ── step 4: the three variants ──────────────────────────────────────────

def oos_timeline(engine, selected, rules, args, symbols):
    """One engine pass over the OOS window for every selected strategy."""
    strategies = []
    for symbol, entry in sorted(selected.items()):
        strategies.append(_decode(entry["chrom"], entry["strategy_name"]))
    result = _run(engine, strategies, symbols, args.oos_start, args.oos_end)
    if "error" in (result or {}):
        raise SystemExit(f"OOS evaluation failed: {result['error']}")
    trades, equities = {}, {}
    for symbol, entry in sorted(selected.items()):
        name = entry["strategy_name"]
        trades[name] = _trades_of(result, name)
        equities[name] = _equity_of(result, name)
    return trades, equities, result


def stamps_for(symbols, args) -> list:
    """The bar grid of the OOS window (from the cached 1h frames)."""
    grid = None
    for symbol in symbols:
        frame = _market_frame(args, symbol)
        window = frame[(frame.index >= pd.Timestamp(args.oos_start))
                       & (frame.index <= pd.Timestamp(args.oos_end))]
        grid = window.index if grid is None else grid.union(window.index)
    return list(grid) if grid is not None else []


def _market_frame(args, symbol: str):
    root = Path(args.data_dir) if args.data_dir else ROOT / "data"
    path = root / "market" / str(symbol) / f"{args.timeframe}.parquet"
    if not path.exists():
        raise SystemExit(f"missing cached market data: {path}")
    return pd.read_parquet(path)


def _orchestrator_timeline(rules, stamps, names, regime_tables, symbols):
    """``RegimeOrchestrator.replay`` → ``{stamp: {name: enabled}}`` + the machine.

    One shared label per stamp is not assumed: each strategy is queried with the
    causal label of the *market proxy* it trades.  With one symbol per strategy
    the label is that symbol's; with several symbols the first symbol's label is
    used for the shared clock (reported in the artifact).
    """
    from core.ai.orchestrator import TimelineEvent

    events = []
    for stamp in stamps:
        label = labels_for(regime_tables, symbols[0], stamp)
        events.append(TimelineEvent("decide", at=stamp, label=label))
    timeline = RegimeOrchestrator.replay(rules, events, names=names)
    enabled: dict = {}
    reasons: dict = {}
    for row in timeline:
        stamp = pd.Timestamp(row["at"])
        enabled[stamp] = {item["strategy"]: bool(item["enabled"])
                          for item in row["rows"]}
        for item in row["rows"]:
            bucket = reasons.setdefault(item["strategy"], {})
            key = str(item.get("reason"))
            bucket[key] = bucket.get(key, 0) + 1
    return enabled, reasons


def _variant(equities, spec, trades, stamps, args, *,
             trials, frames, name, enabled_by_stamp=None, extra=None):
    """Fold one variant's composite and score it against the matched benchmarks."""
    kept = filter_trades_by_timeline(trades, enabled_by_stamp)
    fund = composite_fund(spec, kept, equities, initial_balance=INITIAL_BALANCE,
                          stamps=stamps)
    flat = flatten_trades(kept)
    benchmarks = matched_benchmarks(flat, spec, frames, window_start=args.oos_start,
                                    window_end=args.oos_end,
                                    initial_balance=INITIAL_BALANCE)
    metrics = composite_metrics(fund, flat, trials=trials,
                                window_start=args.oos_start,
                                window_end=args.oos_end,
                                initial_balance=INITIAL_BALANCE,
                                benchmarks=benchmarks)
    block = {"variant": name, "metrics": metrics,
             "per_strategy_trades": {k: len(v) for k, v in sorted(kept.items())},
             "benchmarks": benchmarks,
             "equity_curve": fund.equity_curve}
    if extra:
        block.update(extra)
    return block, fund, kept


# ── the run ─────────────────────────────────────────────────────────────

def run(args) -> dict:
    symbols = [s.strip().upper() for s in args.symbols if s.strip()]
    if not symbols:
        raise SystemExit("no symbols given")
    started = time.time()
    print(f"P7-S4 composite measurement: {symbols} {args.timeframe}; "
          f"train {args.train_start}~{args.train_end}, "
          f"OOS {args.oos_start}~{args.oos_end}", flush=True)

    config, engine = _engine_stack(args.data_dir)
    print(f"ga.regime_conditioning = {config.ga_regime_conditioning}; "
          f"orchestrator live switch off (this tool never wires the live path)",
          flush=True)

    # ── regime tables (causal, with the attrs S1 requires) ──
    regime_tables = {}
    for symbol in symbols:
        frame = _market_frame(args, symbol)
        table = causal_regime_table(frame)
        regime_tables[symbol] = table
    print("regime tables built: "
          + ", ".join(f"{s}({len(t)})" for s, t in sorted(regime_tables.items())),
          flush=True)

    # ── 1. train-window selection ──
    print("\n[1/5] selecting one strategy per symbol on the TRAIN window", flush=True)
    selected = select_per_symbol(engine, args, symbols)

    # ── 2. fixed weights + frozen orchestrator rules ──
    print("\n[2/5] freezing the weight rule and the orchestrator on TRAIN",
          flush=True)
    train_trades = {entry["strategy_name"]: entry["train_trades"]
                    for entry in selected.values()}
    spec = build_composite_spec(train_trades, initial_balance=INITIAL_BALANCE,
                                symbols=symbols,
                                train_window=(args.train_start, args.train_end))
    rules, rule_evidence, fractions = freeze_orchestrator_rules(
        selected, regime_tables, args.train_start, args.train_end)
    fingerprint = rules_fingerprint(rules)
    Path(args.orchestrator_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.orchestrator_out).write_text(
        json.dumps({"rules": rules, "evidence": rule_evidence,
                    "rules_fingerprint": fingerprint,
                    "fixed_on_window": [args.train_start, args.train_end],
                    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                   indent=2), encoding="utf-8")
    for name, weight in sorted(spec.weights.items()):
        print(f"  weight {name}: {weight:.6f} (deployed share "
              f"{spec.deployed[name]:.6f}, {spec.sources[name]})", flush=True)
    print(f"  orchestrator chosen on TRAIN: {rule_evidence['chosen']} "
          f"(fitness {rule_evidence['winner']['fitness']}); "
          f"candidates={len(rule_evidence['candidates'])}", flush=True)
    print(f"  rules_fingerprint={fingerprint}; "
          f"allowed={json.dumps(rules['regime']['allowed'], sort_keys=True)}",
          flush=True)
    print(f"  train-matched enable fractions: "
          f"{ {k: round(v, 4) for k, v in sorted(fractions.items())} }", flush=True)

    # ── 3. one OOS engine pass ──
    print("\n[3/5] one out-of-sample engine pass for every selected strategy",
          flush=True)
    trades, equities, oos_result = oos_timeline(engine, selected, rules, args,
                                                symbols)
    stamps = stamps_for(symbols, args)
    if not stamps:
        raise SystemExit("no OOS bars in the cached frames")
    names = sorted(trades)
    print(f"  {len(names)} strategies, {len(stamps)} bars, "
          f"always-on trades: "
          f"{ {k: len(v) for k, v in sorted(trades.items())} }", flush=True)

    # ── 4. the variants ──
    print("\n[4/5] variants: always-on / random (matched, fixed seed) / "
          "orchestrated", flush=True)
    frames = {symbol: _market_frame(args, symbol) for symbol in symbols}
    trials = CompositeTrials(
        sub_strategy_candidates=(
            int(args.selection_population) * len(symbols) * len(GATE_REGIME_LABELS)),
        windows=2,
        arm_variants=2 + len(args.random_seeds),
        orchestrator_configs=len(rule_evidence["candidates"]),
        notes=("sub_strategy_candidates = selection_population x symbols x "
               "gate labels, evaluated on BOTH the train and the OOS window; "
               "arm_variants = always-on + orchestrated + one per random seed; "
               "orchestrator_configs = every rule set the TRAIN window chose "
               "between (single labels, best pair, all, never)"))
    print(f"  trials = {json.dumps(trials.as_dict(), sort_keys=True)}", flush=True)

    enabled_by_stamp, reasons = _orchestrator_timeline(rules, stamps, names,
                                                       regime_tables, symbols)
    realized = enable_fractions(
        {name: [enabled_by_stamp[stamp].get(name, False) for stamp in stamps]
         for name in names})
    print(f"  orchestrator realized OOS enable fractions: "
          f"{ {k: round(v, 4) for k, v in sorted(realized.items())} }", flush=True)

    variants = []
    always, _fund_a, kept_always = _variant(
        equities, spec, trades, stamps, args, trials=trials, frames=frames,
        name="always_on", enabled_by_stamp=None,
        extra={"description": "every selected strategy enabled on every bar"})
    variants.append(always)
    print(f"  always_on: trades={always['metrics']['trades']} "
          f"ret={always['metrics']['total_return_pct']} "
          f"maxDD={always['metrics']['max_drawdown_pct']} "
          f"sharpe={always['metrics']['sharpe']} "
          f"tim={always['metrics']['time_in_market_pct']} "
          f"dsr={always['metrics']['deflated_sharpe']}", flush=True)

    for seed in args.random_seeds:
        stamps_by_name = random_enabled_stamps(stamps, names, fractions,
                                               seed=int(seed))
        plan = {stamp: {name: (stamp in stamps_by_name[name]) for name in names}
                for stamp in stamps}
        block, _fund, _kept = _variant(
            equities, spec, trades, stamps, args, trials=trials,
            frames=frames, name=f"random_seed_{int(seed)}",
            enabled_by_stamp=plan,
            extra={"description": ("random start/stop, seeded, per-strategy "
                                   "enable probability matched to the "
                                   "TRAIN-window fraction the chosen rule set "
                                   "allowed"),
                   "seed": int(seed),
                   "matched_fractions": {k: round(v, 6)
                                         for k, v in sorted(fractions.items())},
                   "realized_fractions": enable_fractions(
                       {name: [plan[stamp][name] for stamp in stamps]
                        for name in names})})
        variants.append(block)
        metrics = block["metrics"]
        print(f"  random_seed_{int(seed)}: trades={metrics['trades']} "
              f"ret={metrics['total_return_pct']} "
              f"maxDD={metrics['max_drawdown_pct']} "
              f"sharpe={metrics['sharpe']} tim={metrics['time_in_market_pct']} "
              f"dsr={metrics['deflated_sharpe']}", flush=True)

    orchestrated, _fund_o, kept_o = _variant(
        equities, spec, trades, stamps, args, trials=trials, frames=frames,
        name="orchestrated", enabled_by_stamp=enabled_by_stamp,
        extra={"description": ("RegimeOrchestrator.replay: each strategy enabled "
                               "only in the causal regime(s) the TRAIN window "
                               "chose for it"),
               "rules_fingerprint": fingerprint,
               "rules": rules,
               "train_matched_fractions": {k: round(v, 6)
                                           for k, v in sorted(fractions.items())},
               "realized_fractions": {k: round(v, 6)
                                      for k, v in sorted(realized.items())},
               "block_reasons": reasons})
    orchestrated["regime_contribution"] = regime_contribution(
        flatten_trades(kept_o),
        lambda stamp: labels_for(regime_tables, symbols[0], stamp),
        INITIAL_BALANCE)
    variants.append(orchestrated)
    metrics = orchestrated["metrics"]
    print(f"  orchestrated: trades={metrics['trades']} "
          f"ret={metrics['total_return_pct']} "
          f"maxDD={metrics['max_drawdown_pct']} sharpe={metrics['sharpe']} "
          f"tim={metrics['time_in_market_pct']} "
          f"dsr={metrics['deflated_sharpe']}", flush=True)

    # ── 5. holdout claim (after the numbers exist, before the artifact) ──
    print("\n[5/5] one-shot holdout counter", flush=True)
    before = holdout_mod.holdout_status(HOLDOUT_ID, args.oos_start, args.oos_end,
                                        args.timeframe, path=args.holdout_store)
    reuse_allowed = bool(args.allow_holdout_reuse)
    try:
        claim = holdout_mod.claim_holdout(
            HOLDOUT_ID, args.oos_start, args.oos_end, timeframe=args.timeframe,
            path=args.holdout_store, label="p7_composite_measure",
            rules_fingerprint=fingerprint, allow_reuse=reuse_allowed,
            detail={"variants": [v["variant"] for v in variants],
                    "trials_total": trials.total,
                    "orchestrated_trades": orchestrated["metrics"]["trades"],
                    "orchestrated_dsr": orchestrated["metrics"]["deflated_sharpe"]})
    except holdout_mod.HoldoutRefusal as refusal:
        print(f"  REFUSED: {refusal}", flush=True)
        raise SystemExit(
            "the holdout window was already evaluated; refusing to open it "
            "again silently (re-run with --allow-holdout-reuse to record the "
            "reuse as overfitting)")
    print(f"  claimed {claim['reuse_index']}st/nd evaluation of "
          f"{holdout_mod.holdout_key(HOLDOUT_ID, args.oos_start, args.oos_end, args.timeframe)}"
          f" (reuse={claim['reuse']}, at={claim['at']}, "
          f"revision={claim['revision']})", flush=True)

    verdict = orchestrated["metrics"]["usability"]
    beats = _beats_controls(orchestrated, variants)
    artifact = {
        "script": "tools/p7_composite_measure.py",
        "plan": "docs/overhaul/P7_REGIME_PLAN.md S4",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 1),
        "data": {"root": str(Path(args.data_dir or ROOT / "data")),
                 "symbols": symbols, "timeframe": args.timeframe},
        "windows": {"train": [args.train_start, args.train_end],
                    "out_of_sample": [args.oos_start, args.oos_end]},
        "seeds": {"cohort": int(args.seed),
                  "random": [int(s) for s in args.random_seeds]},
        "selection": {symbol: {"strategy_name": entry["strategy_name"],
                               "fitness": entry["fitness"],
                               "population": entry["population"],
                               "n_trials": entry["n_trials"],
                               "train_score": entry["train_score"]}
                      for symbol, entry in sorted(selected.items())},
        "composite": spec.as_dict(),
        "weight_concentration": _concentration(spec),
        "orchestrator": {"rules": rules, "rules_fingerprint": fingerprint,
                         "rule_evidence": rule_evidence,
                         "train_matched_fractions": {
                             k: round(v, 6) for k, v in sorted(fractions.items())},
                         "oos_realized_fractions": {
                             k: round(v, 6) for k, v in sorted(realized.items())},
                         "rules_file": str(args.orchestrator_out)},
        "trials": trials.as_dict(),
        "holdout": {**holdout_mod.holdout_status(
            HOLDOUT_ID, args.oos_start, args.oos_end, args.timeframe,
            path=args.holdout_store),
            "claim": claim,
            "reuse_allowed": reuse_allowed,
            "before": before},
        "usability": {
            "bar": f">= {DEFAULT_USABILITY_MIN_TRADES} out-of-sample composite "
                   f"trades AND DSR > 0",
            "orchestrated": verdict,
            "always_on": always["metrics"]["usability"],
            "beats_either_control": beats,
        },
        "variants": [{k: v for k, v in block.items()
                      if k != "equity_curve"} for block in variants],
        "equity_curves": {block["variant"]: block["equity_curve"]
                          for block in variants},
        "variants_note": ("every variant folds the SAME engine trade streams at "
                          "the SAME fixed train-window weights; only the "
                          "enable/disable timeline differs"),
    }
    artifact["_out_path"] = str(args.out)
    _write(Path(args.out), artifact)
    _print_summary(artifact)
    return artifact


def _concentration(spec) -> dict:
    """How concentrated the fixed weights are (a one-child composite is a child).

    ``max_weight`` and the Herfindahl index are **reported, never gated**: the
    deployed-capital rule is mechanical, so when one symbol's measured share
    dwarfs the others' the composite is close to a single-strategy evaluation and
    the reader has to be able to see that in the artifact.
    """
    weights = [float(w) for w in spec.weights.values()] or [0.0]
    return {"strategies": len(weights),
            "max_weight": round(max(weights), 6),
            "herfindahl": round(sum(w * w for w in weights), 6),
            "effective_strategies": round(1.0 / sum(w * w for w in weights), 4)
            if sum(w * w for w in weights) > 0 else None}


def _beats_controls(orchestrated, variants) -> dict:
    """Does the orchestrator beat either control on the numbers that matter?"""
    def row(block):
        metrics = block["metrics"]
        return {"return": metrics["total_return_pct"],
                "alpha": metrics["alpha_vs_exposure_matched_pct"],
                "sharpe": metrics["sharpe"], "dsr": metrics["deflated_sharpe"],
                "max_dd": metrics["max_drawdown_pct"]}
    mine = row(orchestrated)
    always = row(variants[0])
    randoms = [row(block) for block in variants[1:-1]]
    best_random = max((r["return"] for r in randoms), default=None)
    return {
        "return_vs_always_on": round(mine["return"] - always["return"], 4),
        "alpha_vs_always_on": (None if mine["alpha"] is None
                               or always["alpha"] is None
                               else round(mine["alpha"] - always["alpha"], 4)),
        "return_vs_best_random": (None if best_random is None
                                  else round(mine["return"] - best_random, 4)),
        "beats_always_on_return": bool(mine["return"] > always["return"]),
        "beats_best_random_return": (None if best_random is None
                                     else bool(mine["return"] > best_random)),
        "dsr_positive": bool(mine["dsr"] > 0),
        "alpha_positive": (None if mine["alpha"] is None
                           else bool(mine["alpha"] > 0)),
        "controls": {"always_on": always, "random": randoms},
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _print_summary(artifact: dict) -> None:
    print("\n=== P7-S4 composite (orchestrator + sub-strategies) - OOS ===")
    header = (f"{'variant':<20}{'trades':>7}{'return%':>11}{'maxDD%':>9}"
              f"{'sharpe':>9}{'tim%':>8}{'bench(em)%':>12}{'alpha(em)':>11}"
              f"{'alpha(bh)':>11}{'DSR':>10}")
    print(header)
    print("-" * len(header))
    for block in artifact["variants"]:
        m = block["metrics"]
        def cell(key, digits=4):
            value = m.get(key)
            return "n/a" if value is None else f"{value:.{digits}f}"
        print(f"{block['variant']:<20}{m['trades']:>7}{cell('total_return_pct'):>11}"
              f"{cell('max_drawdown_pct'):>9}{cell('sharpe'):>9}"
              f"{cell('time_in_market_pct', 2):>8}"
              f"{cell('exposure_matched_pct'):>12}"
              f"{cell('alpha_vs_exposure_matched_pct'):>11}"
              f"{cell('alpha_vs_buy_hold_pct'):>11}"
              f"{cell('deflated_sharpe', 6):>10}")
        if not m.get("dsr_estimated", True):
            print(f"{'':<20}  DSR {m['dsr_note']}")
    trials = artifact["trials"]
    concentration = artifact["weight_concentration"]
    print(f"\nweights: " + ", ".join(
        f"{name}={weight:.4f}" for name, weight
        in sorted(artifact["composite"]["weights"].items()))
        + f"  (max={concentration['max_weight']}, "
          f"effective strategies={concentration['effective_strategies']})")
    print(f"trials deflating the DSR: total={trials['total']} "
          f"({trials['sub_strategy_candidates']} candidates x {trials['windows']} "
          f"windows + {trials['arm_variants']} arms + "
          f"{trials['orchestrator_configs']} orchestrator rule sets)")
    usability = artifact["usability"]
    print(f"usability bar: {usability['bar']}")
    print(f"  always_on    : {usability['always_on']['usable']} "
          f"({usability['always_on']['reason'] or 'passes'})")
    print(f"  orchestrated : {usability['orchestrated']['usable']} "
          f"({usability['orchestrated']['reason'] or 'passes'})")
    beats = usability["beats_either_control"]
    print(f"orchestrated vs controls: d_return vs always_on="
          f"{beats['return_vs_always_on']} "
          f"vs best random={beats['return_vs_best_random']}; "
          f"beats_always_on={beats['beats_always_on_return']} "
          f"beats_random={beats['beats_best_random_return']} "
          f"dsr>0={beats['dsr_positive']} alpha>0={beats['alpha_positive']}")
    hold = artifact["holdout"]
    print(f"holdout: {hold['key']} evaluations={hold['evaluations']} "
          f"reused={hold['reused']} (before this run: {hold['before']['evaluations']})")
    print(f"artifact: {artifact.get('_out_path', '')}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--train-start", default="2025-11-01")
    parser.add_argument("--train-end", default="2026-02-01")
    parser.add_argument("--oos-start", default="2026-02-01")
    parser.add_argument("--oos-end", default="2026-06-01")
    parser.add_argument("--selection-population", type=int, default=2,
                        help="genomes per symbol evaluated on the TRAIN window; "
                             "the best is selected (x 5 gate labels in the trial "
                             "count)")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--random-seeds", nargs="*", type=int, default=[7, 8],
                        help="seeds for the exposure-matched random controls")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--orchestrator-out", default=str(DEFAULT_ORCHESTRATOR),
                        help="where the train-window-frozen rules are written")
    parser.add_argument("--holdout-store", default=None,
                        help=f"holdout registry (default {holdout_mod.DEFAULT_HOLDOUT_PATH})")
    parser.add_argument("--allow-holdout-reuse", action="store_true",
                        help="record a second look at the same window as "
                             "overfitting instead of refusing it")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
