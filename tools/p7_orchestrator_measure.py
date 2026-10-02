"""P7-S3 evidence — always-on vs orchestrated over one out-of-sample window.

WHAT THIS MEASURES (a bounded S4 preview, not S4)
-------------------------------------------------
The plan's S4 asks whether the *composite* (upper-layer orchestrator + the
strategies it permits) is better out-of-sample than the same strategies with no
orchestration at all.  This tool answers a deliberately smaller version of that
question on the **real cached parquet**, with the production engine, the
production cost model and the production scorer, and reports the answer whether
or not it is the hoped-for one:

* **Same strategy set, same costs, same window.**  A fixed cohort of ``--genomes``
  genomes is drawn from one seed and each genome is decoded **five times**, once
  per causal regime label, with the label pinned into the strategy's
  ``regime_filter`` (P7-S1's attribute).  Variant ``(g, L)`` therefore trades only
  on bars whose causal label is ``L``.  Both arms evaluate that identical variant
  set in one engine pass each.
* **always-on** — every variant is allowed on every bar (no orchestrator).
* **orchestrated** — the SAME variants, but a variant's entry is dropped when the
  ``core.ai.orchestrator.RegimeOrchestrator`` says it is disabled at that bar.

NO LOOK-AHEAD IN THE ORCHESTRATION DECISION
-------------------------------------------
* The rule set is fixed before the window is opened and its
  ``rules_fingerprint`` is recorded in the artifact.
* The regime→strategy mapping is the **naive** one (each variant may run in
  exactly its own label) — nothing is selected by looking at returns.
* The kill switch is seeded with the genome's **train-window** trade outcomes
  only, and its per-bar verdicts over the out-of-sample window are produced in
  chronological order (first blocking rule wins), exactly as the live seam would.
* The volatility reference is the causal EWMA series (``core.ml.volatility``:
  entry ``i`` uses returns ``<= i``), also from the train window onwards.

The honest reading of the counterfactual: the orchestrated arm's trades are the
always-on arm's trades **minus** the ones the orchestrator refused.  That is
reported explicitly (``dropped_trades``, ``dropped_pnl``, ``dropped_wins``) so a
"reduced drawdown" cannot be mistaken for "picked better trades" when it is
simply "traded less".

Everything written goes to ``--out`` (JSON): no ``data/binance_trader.db``, no
``strategies/``, no ``data/market`` write.

Usage::

    python tools/p7_orchestrator_measure.py
    python tools/p7_orchestrator_measure.py --genomes 3 --symbols BTCUSDT ETHUSDT \
        --train-start 2025-11-01 --train-end 2026-02-01 \
        --oos-start 2026-02-01 --oos-end 2026-06-01
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

from core.ai.orchestrator import (                       # noqa: E402
    RegimeOrchestrator, TradeOutcome, orchestrator_config_from_raw,
    rules_fingerprint)
from core.strategy.regime_causal import (                # noqa: E402
    GATE_REGIME_LABELS, causal_regime_table)

DEFAULT_OUT = Path(tempfile.gettempdir()) / "p7_orchestrator_measure.json"

#: The rule set this measurement uses.  Chosen by rule-of-thumb BEFORE the
#: out-of-sample window was touched (a "two strikes" stop and a 3x-median
#: volatility gate), never by looking at these results; the fingerprint is in the
#: artifact so a reader can check that the rules were not edited afterwards.
#: S4 is the stage allowed to *select* these numbers — on the train window only.
MEASUREMENT_RULES = {
    "enabled": True,
    "regime": {
        # The naive mapping: a variant may run in exactly its own label.  It is a
        # property of the variant's construction, not a fitted choice.
        "allowed": {},
        "default_action": "deny",
        "missing_regime_action": "deny",
    },
    "kill_switch": {"consecutive_losses": 2, "loss_threshold": 0.0},
    "vol": {"multiple": 3.0, "window": 200, "min_samples": 100,
            "missing_action": "allow"},
    "breadth": {
        # breadth has NO history (forward-only record) and this host's cache is
        # empty for these windows, so the gate is exercised as "missing" — the
        # documented fallback, which ships `allow`.
        "min_up_share": 0.40, "min_coverage": 0.90,
        "max_staleness_ms": 1_800_000,
        "missing_action": "allow", "stale_action": "allow",
    },
}


# ── engine / cohort plumbing (mirrors tools/p7_regime_conditioning_measure.py) ──

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
    engine = BacktestEngine(config, None, RiskManager(config, bus),
                            OrderExecutor(config, bus))
    return config, engine


def _cohort(genomes: int, seed: int, timeframe: str = "1h"):
    """A fixed cohort of chromosomes (timeframe pinned to one interval)."""
    from core.ga.genome import random_chromosome

    state = random.getstate()
    random.seed(seed)
    chroms = [random_chromosome(f"p7s3_{i}") for i in range(genomes)]
    random.setstate(state)
    for chrom in chroms:
        for gene in chrom.get("categorical", []):
            if gene.name == "timeframes":
                gene.value = timeframe
    return chroms


def _with_label(chrom: dict, label: str) -> dict:
    """A deep copy with ``regime_filter`` pinned to *label* (P7-S1's gene)."""
    from core.ga.genome import (CategoricalGene, confine_regime_gene,
                                regime_gene_options)

    out = copy.deepcopy(chrom)
    confine_regime_gene(out, True)
    found = False
    for gene in out["categorical"]:
        if gene.name == "regime_filter":
            gene.value = label
            found = True
    if not found:
        out["categorical"].append(
            CategoricalGene("regime_filter", label, regime_gene_options()))
    return out


def _decode(chrom: dict):
    from core.ga.genome import chromosome_to_strategy

    strategy = chromosome_to_strategy(chrom, regime_conditioning=True)
    if strategy.ml_config:
        strategy.ml_config.enabled = False
    return strategy


def _evaluate_variants(engine, strategies, symbols, start, end):
    """ONE engine pass over every variant → ``({name: trades}, {name: equity})``.

    The production GA shape: ``per_genome_ledger=True`` gives each strategy its own
    cash/equity sub-ledger and its own position slots, so variant PnL never mixes
    with variant PnL.
    """
    from core.ga.fitness import isolated_eval_kwargs

    result = engine.run_with_exit_evaluation(
        strategies=strategies, symbols=list(symbols), date_start=start,
        date_end=end, initial_balance=10_000.0, mode="full",
        simulate_ai_weights=False, ml_engine="lightgbm", use_live_spread=False,
        benchmark_mode="none", **isolated_eval_kwargs())
    if "error" in result:
        raise SystemExit(f"engine error: {result['error']}")
    per = result.get("per_strategy_equity") or {}
    trades = {name: list(entry.get("trades") or []) for name, entry in per.items()}
    equity = {name: list(entry.get("equity_curve") or []) for name, entry in per.items()}
    return trades, equity


# ── the causal inputs the orchestrator consumes ─────────────────────────

def _causal_inputs(symbols, timeframe):
    """``{symbol: labels}``, ``{symbol: vol Series}``, ``{symbol: index}`` (read-only)."""
    import pandas as pd

    labels, vols, index = {}, {}, {}
    for symbol in symbols:
        path = ROOT / "data" / "market" / symbol / f"{timeframe}.parquet"
        frame = pd.read_parquet(path)
        table = causal_regime_table(frame)
        labels[symbol] = table["regime"].astype(str)
        vols[symbol] = _vol_series(frame)
        index[symbol] = frame.index
    return labels, vols, index


def _vol_series(frame):
    """The causal EWMA vol series, **index-aligned with the bars it describes**.

    ``ewma_vol_series`` drops non-finite returns (the first bar's ``pct_change``),
    so the returned Series is one observation shorter than the frame.  Its index
    is therefore the timestamps of the returns it actually consumed — taking
    ``frame.index`` wholesale would misalign every lookup by one bar, which is
    exactly the kind of silent off-by-one a causal seam must not have.
    """
    import numpy as np
    from core.ml.volatility import ewma_vol_series as _ewma

    returns = frame["close"].pct_change()
    finite = np.isfinite(returns.to_numpy(dtype=float))
    return _ewma(returns, index=frame.index[finite])


def _label_at(labels, symbol, ts):
    series = labels[symbol]
    cut = int(series.index.searchsorted(ts, side="right"))
    return None if cut <= 0 else str(series.iloc[cut - 1])


def _vol_at(vols, symbol, ts):
    series = vols[symbol]
    cut = int(series.index.searchsorted(ts, side="right"))
    if cut <= 0:
        return None
    value = float(series.iloc[cut - 1])
    return value if value == value else None


# ── the orchestrated arm ────────────────────────────────────────────────

def _run_orchestrator(variant_names, variant_labels, train_trades_by_variant,
                      labels, vols, index, symbols, oos_start, oos_end):
    """Feed train outcomes, then decide chronologically over the OOS bars.

    Returns ``(enabled_windows, timeline_summary)`` where ``enabled_windows`` maps
    a variant name to the list of timestamps at which it was enabled.
    """
    config = orchestrator_config_from_raw(MEASUREMENT_RULES)
    machine = RegimeOrchestrator(config, names=variant_names)
    # The naive mapping: variant (g, L) may run in exactly L.
    machine.config = type(machine.config)(
        enabled=True,
        regime=type(machine.config.regime)(
            allowed={name: (variant_labels[name],) for name in variant_names},
            default_action="deny", missing_regime_action="deny"),
        kill_switch=machine.config.kill_switch,
        vol=machine.config.vol,
        breadth=machine.config.breadth)

    # 1) the train window's realised outcomes seed the kill switch (this is the
    #    only reason the train window is read at all).
    for name in variant_names:
        for trade in train_trades_by_variant.get(name) or []:
            regime = _label_at(labels, _symbol_of(trade), trade.get("closed_at"))
            machine.record_trade(name, float(trade.get("pnl") or 0.0),
                                 at=trade.get("closed_at"), regime=regime)

    # 2) chronological verdicts over the out-of-sample bars, on the union of the
    #    symbols' bar timestamps (one label per (symbol, bar)).  The verdicts are
    #    produced in bar order — the only order in which a state machine's output
    #    is well defined — and the enabled set per variant is read back from that
    #    same row, so the timeline is reproducible from the artifact.
    stamps = sorted({stamp for symbol in symbols
                     for stamp in index[symbol]
                     if oos_start <= stamp <= oos_end})
    enabled_windows = {name: [] for name in variant_names}
    disabled_at = {name: {} for name in variant_names}
    reasons = {}
    for stamp in stamps:
        for name in variant_names:
            symbol = _symbol_of_name(name)
            verdict = machine.decide(
                name, at=stamp, label=_label_at(labels, symbol, stamp),
                vol=_vol_at(vols, symbol, stamp), vol_key=symbol)
            reasons[verdict.reason] = reasons.get(verdict.reason, 0) + 1
            if verdict.enabled:
                enabled_windows[name].append(stamp)
            else:
                disabled_at[name][stamp] = verdict.blocked_reasons
    return enabled_windows, disabled_at, {
        "bars_decided": len(stamps),
        "per_bar_reasons": dict(sorted(reasons.items())),
        "killed": {name: bool(machine.state(name).killed) for name in variant_names},
        "consecutive_losses": {name: int(machine.state(name).consecutive_losses)
                               for name in variant_names},
        "trades_seen_in_train": {name: int(machine.state(name).trades)
                                 for name in variant_names},
    }


def _symbol_of_name(name: str) -> str:
    """``p7s3_0_trend_up_BTCUSDT_1h`` → ``BTCUSDT`` (the symbol is a name part)."""
    for part in str(name).split("_"):
        if part.endswith("USDT") and len(part) > 4:
            return part
    return ""


def _symbol_of(trade: dict) -> str:
    return str(trade.get("symbol") or "")


# ── metrics for one arm ─────────────────────────────────────────────────

def _arm_metrics(trades_by_variant, index, symbols, start, end, trials, initial=10_000.0):
    """Portfolio-level metrics for one arm (all variants pooled, equal notional)."""
    import numpy as np
    import pandas as pd
    from core.ga.fitness import deflated_sharpe_ratio, max_drawdown_pct

    trades = [t for name in sorted(trades_by_variant)
              for t in (trades_by_variant[name] or [])]
    trades.sort(key=lambda t: str(t.get("closed_at")))
    stamps = sorted({stamp for symbol in symbols for stamp in index[symbol]
                     if start <= stamp <= end})
    # Mark-to-market equity at each bar: realised PnL so far + the cost basis of
    # whatever is still open.  Open positions contribute their entry notional, so
    # the curve never pretends a position is free.
    pnl_by_close: dict = {}
    spans: list[tuple] = []
    for trade in trades:
        try:
            opened = pd.Timestamp(trade.get("opened_at"))
            closed = pd.Timestamp(trade.get("closed_at"))
        except Exception:
            continue
        pnl_by_close[closed] = pnl_by_close.get(closed, 0.0) + float(trade.get("pnl") or 0.0)
        spans.append((opened, closed, float(trade.get("amount_usdt") or 0.0)))
    equity, realised, in_market = [], 0.0, 0.0
    for stamp in stamps:
        realised += pnl_by_close.get(pd.Timestamp(stamp), 0.0)
        open_notional = sum(amount for opened, closed, amount in spans
                            if opened <= stamp < closed)
        if open_notional > 0:
            in_market += 1
        equity.append({"time": str(stamp), "equity": initial + realised})
    series = pd.Series([point["equity"] for point in equity],
                       index=pd.DatetimeIndex([pd.Timestamp(point["time"])
                                               for point in equity]))
    dailies = series.resample("1D").last().dropna().pct_change().dropna()
    sharpe = 0.0
    if len(dailies) >= 2 and float(dailies.std(ddof=1)) > 0:
        periods = max(len(dailies), 1) * (365.0 / max(
            (pd.Timestamp(equity[-1]["time"]) - pd.Timestamp(equity[0]["time"])).days, 1))
        sharpe = float(dailies.mean() / dailies.std(ddof=1)) * (periods ** 0.5)
    dsr = deflated_sharpe_ratio(sharpe, n_trials=int(trials),
                                observation_periods=max(len(dailies), 1),
                                skew=float(dailies.skew()) if len(dailies) > 2 else 0.0,
                                kurtosis=(float(dailies.kurtosis()) + 3.0)
                                if len(dailies) > 3 else 3.0)
    return {
        "trades": len(trades),
        "total_return_pct": round(float(series.iloc[-1] - initial) / initial * 100.0, 4),
        "pnl": round(float(series.iloc[-1] - initial), 4),
        "max_dd_pct": round(float(max_drawdown_pct(equity)), 4),
        "time_in_market_pct": round(in_market / max(len(stamps), 1) * 100.0, 4),
        "daily_observations": int(len(dailies)),
        "sharpe": round(sharpe, 4),
        "dsr": float(dsr["dsr"]),
        "dsr_p_value": dsr["p_value"],
        "dsr_expected_max_random": dsr["expected_max_random"],
        "dsr_n_trials": int(dsr["n_trials"]),
        "bars": len(stamps),
    }


# ── main ────────────────────────────────────────────────────────────────

def run(args) -> dict:
    import pandas as pd

    symbols = [s.strip().upper() for s in args.symbols if s.strip()]
    started = time.time()
    labels, vols, index = _causal_inputs(symbols, args.timeframe)
    cohort = _cohort(args.genomes, args.seed, args.timeframe)

    variants, variant_labels, variant_symbols = [], {}, {}
    for gi, chrom in enumerate(cohort):
        for label in GATE_REGIME_LABELS:
            for symbol in symbols:
                strategy = _decode(_with_label(chrom, label))
                strategy.symbols = [symbol]
                name = f"p7s3_{gi}_{label}_{symbol}_{args.timeframe}"
                strategy.name = name
                variants.append(strategy)
                variant_labels[name] = label
                variant_symbols[name] = symbol

    config, engine = _engine_stack()
    print(f"P7-S3 measurement: {len(variants)} variants "
          f"({args.genomes} genomes x {len(GATE_REGIME_LABELS)} labels x "
          f"{len(symbols)} symbols), timeframe {args.timeframe}", flush=True)
    print(f"  train {args.train_start}~{args.train_end} | "
          f"OOS {args.oos_start}~{args.oos_end} | trials for DSR={args.trials}",
          flush=True)
    print(f"  rules_fingerprint="
          f"{rules_fingerprint(orchestrator_config_from_raw(MEASUREMENT_RULES))}",
          flush=True)

    # The two engine passes are the expensive part (~1-2 min each).  They are
    # cached under --cache so the (cheap, pure) orchestration can be iterated on
    # without paying for them again; the cache key covers every input that changes
    # the trades, and the cache is never read for a different key.
    cache_path = Path(args.cache) if args.cache else None
    cache_key = {"symbols": symbols, "timeframe": args.timeframe,
                 "genomes": args.genomes, "seed": args.seed,
                 "train": [args.train_start, args.train_end],
                 "oos": [args.oos_start, args.oos_end]}
    cached = None
    if cache_path is not None and cache_path.exists():
        try:
            blob = json.loads(cache_path.read_text(encoding="utf-8"))
            if blob.get("key") == cache_key:
                train_trades, oos_trades = blob["train"], blob["oos"]
                cached = True
        except (json.JSONDecodeError, KeyError, TypeError):
            cached = None
    if cached:
        print("  [cache] reusing the two engine passes", flush=True)
    else:
        print("  [train] one engine pass ...", flush=True)
        train_trades, _ = _evaluate_variants(
            engine, variants, symbols, args.train_start, args.train_end)
        print(f"  [train] done in {time.time() - started:.1f}s — "
              f"{sum(len(v) for v in train_trades.values())} trades", flush=True)

        print("  [oos] one engine pass (the ALWAYS-ON arm) ...", flush=True)
        oos_started = time.time()
        oos_trades, _ = _evaluate_variants(
            engine, variants, symbols, args.oos_start, args.oos_end)
        print(f"  [oos] done in {time.time() - oos_started:.1f}s — "
              f"{sum(len(v) for v in oos_trades.values())} trades", flush=True)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(
                {"key": cache_key, "train": train_trades, "oos": oos_trades},
                default=str), encoding="utf-8")

    enabled_windows, disabled_at, orch_summary = _run_orchestrator(
        [s.name for s in variants], variant_labels, train_trades, labels, vols,
        index, symbols, pd.Timestamp(args.oos_start), pd.Timestamp(args.oos_end))

    # The orchestrated arm: the same trades, minus the entries the orchestrator
    # refused at their own entry bar.
    orchestrated: dict = {}
    dropped = []
    for name, trades in oos_trades.items():
        allowed = set(enabled_windows.get(name) or [])
        kept, refused = [], []
        for trade in trades:
            try:
                opened = pd.Timestamp(trade.get("opened_at"))
            except Exception:
                kept.append(trade)
                continue
            (kept if opened in allowed else refused).append(trade)
        orchestrated[name] = kept
        for trade in refused:
            opened = pd.Timestamp(trade.get("opened_at"))
            dropped.append({"variant": name, "label": variant_labels.get(name),
                            "symbol": trade.get("symbol"),
                            "opened_at": str(opened),
                            "reasons": list((disabled_at.get(name) or {}).get(
                                opened) or ["not_in_any_decision_round"]),
                            "pnl": float(trade.get("pnl") or 0.0)})

    full = _arm_metrics(oos_trades, index, symbols, pd.Timestamp(args.oos_start),
                        pd.Timestamp(args.oos_end), args.trials)
    kept = _arm_metrics(orchestrated, index, symbols, pd.Timestamp(args.oos_start),
                        pd.Timestamp(args.oos_end), args.trials)
    dropped_pnl = sum(entry["pnl"] for entry in dropped)
    by_reason: dict = {}
    for entry in dropped:
        key = "+".join(entry["reasons"])
        bucket = by_reason.setdefault(key, {"trades": 0, "pnl": 0.0, "wins": 0,
                                            "losses": 0})
        bucket["trades"] += 1
        bucket["pnl"] = round(bucket["pnl"] + entry["pnl"], 4)
        bucket["wins" if entry["pnl"] > 0 else "losses"] += 1
    artifact = {
        "script": "tools/p7_orchestrator_measure.py",
        "data": {"root": str(ROOT / "data" / "market"), "symbols": symbols,
                 "timeframe": args.timeframe},
        "windows": {"train": [args.train_start, args.train_end],
                    "out_of_sample": [args.oos_start, args.oos_end]},
        "cohort": {"genomes": args.genomes, "seed": args.seed,
                   "labels": list(GATE_REGIME_LABELS),
                   "variants": len(variants),
                   "n_trials_for_dsr": int(args.trials)},
        "rules": MEASUREMENT_RULES,
        "rules_fingerprint": rules_fingerprint(
            orchestrator_config_from_raw(MEASUREMENT_RULES)),
        "orchestrator": orch_summary,
        "always_on": full,
        "orchestrated": kept,
        "counterfactual": {
            "dropped_trades": len(dropped),
            "dropped_pnl": round(dropped_pnl, 4),
            "dropped_wins": sum(1 for entry in dropped if entry["pnl"] > 0),
            "dropped_losses": sum(1 for entry in dropped if entry["pnl"] < 0),
            "dropped_mean_pnl": (round(dropped_pnl / len(dropped), 4)
                                 if dropped else None),
            "dropped_by_reason": dict(sorted(by_reason.items())),
            "sample": dropped[:20],
        },
        "per_variant": {
            name: {"label": variant_labels.get(name),
                   "symbol": variant_symbols.get(name),
                   "always_on_trades": len(oos_trades.get(name) or []),
                   "orchestrated_trades": len(orchestrated.get(name) or []),
                   "train_trades": len(train_trades.get(name) or [])}
            for name in sorted(oos_trades)},
        "honest_notes": [
            "This is a BOUNDED S4 preview, not S4: the composite contract, the "
            "exposure-matched/buy_hold comparison on the composite's own holding "
            "intervals and the one-shot holdout counter are S4's deliverables.",
            "The orchestrated arm is the always-on arm's trades MINUS the refused "
            "ones (same engine, same costs, same window).  It is not a re-run with "
            "different position sizing, so a return difference is an exposure "
            "difference, not better timing.",
            "The regime->strategy mapping is the naive 'each variant may run in its "
            "own label'; nothing was selected by looking at this window.",
            "The kill switch is seeded from the TRAIN window only; the rule set was "
            "fixed before the OOS window was read (rules_fingerprint above).",
            "Breadth has no history on this host, so its gate is exercised in the "
            "missing-data branch (shipped fallback: allow).",
            "DSR is deflated with n_trials=%d (every variant x both arms), and a "
            "0-trade arm yields dsr=0.0 by construction." % int(args.trials),
        ],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 1),
    }
    return artifact


def _print(artifact: dict) -> None:
    full, kept = artifact["always_on"], artifact["orchestrated"]
    print("\n=== P7-S3 always-on vs orchestrated (same variants, same costs) ===")
    header = (f"{'arm':<14}{'trades':>8}{'return%':>10}{'maxDD%':>9}"
              f"{'time%':>8}{'sharpe':>9}{'dsr':>10}")
    print(header)
    print("-" * len(header))
    for tag, arm in (("always-on", full), ("orchestrated", kept)):
        print(f"{tag:<14}{arm['trades']:>8}{arm['total_return_pct']:>10}"
              f"{arm['max_dd_pct']:>9}{arm['time_in_market_pct']:>8}"
              f"{arm['sharpe']:>9}{arm['dsr']:>10}")
    cf = artifact["counterfactual"]
    print(f"\nrefused: {cf['dropped_trades']} trades, pnl {cf['dropped_pnl']} "
          f"({cf['dropped_wins']} wins / {cf['dropped_losses']} losses, "
          f"mean {cf['dropped_mean_pnl']})")
    for reason, bucket in (cf.get("dropped_by_reason") or {}).items():
        print(f"  by rule [{reason}]: {bucket['trades']} trades, "
              f"pnl {bucket['pnl']} ({bucket['wins']}W/{bucket['losses']}L)")
    print(f"rules_fingerprint={artifact['rules_fingerprint']}  "
          f"trials={artifact['cohort']['n_trials_for_dsr']}  "
          f"seconds={artifact['seconds']}")
    print("verdict: " + _verdict(full, kept, cf))


def _verdict(full: dict, kept: dict, cf: dict) -> str:
    """Plain-language reading of the two arms (no marketing)."""
    d_return = round(kept["total_return_pct"] - full["total_return_pct"], 4)
    d_dd = round(kept["max_dd_pct"] - full["max_dd_pct"], 4)
    parts = [f"return {d_return:+} pp", f"maxDD {d_dd:+} pp"]
    if full["dsr"] <= 0 and kept["dsr"] <= 0:
        parts.append("neither arm clears DSR > 0 (not distinguishable from data mining)")
    if cf["dropped_trades"] and cf["dropped_pnl"] < 0:
        parts.append("the refusals removed net-losing exposure")
    elif cf["dropped_trades"]:
        parts.append("the refusals removed net-PROFITABLE exposure")
    return "; ".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--genomes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--train-start", default="2025-11-01")
    parser.add_argument("--train-end", default="2026-02-01")
    parser.add_argument("--oos-start", default="2026-02-01")
    parser.add_argument("--oos-end", default="2026-06-01")
    parser.add_argument("--trials", type=int, default=30,
                        help="n_trials for the DSR (variants x arms; every "
                             "attempted orchestrator/strategy variant counts)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--cache", default=str(DEFAULT_OUT.with_suffix(".cache.json")),
                        help="where to memoise the two engine passes (the "
                             "orchestration itself is re-run every time)")
    args = parser.parse_args(argv)
    artifact = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(artifact, indent=2, default=str),
                              encoding="utf-8")
    _print(artifact)
    print(f"\nartifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
