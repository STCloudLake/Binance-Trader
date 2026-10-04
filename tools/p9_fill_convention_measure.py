"""P9 evidence — how much does the **fill convention** move a backtest?

The finding
-----------
The engine decides and fills on the SAME bar's close: the entry path slices the
indicators to ``ts`` (``core/backtest/engine.py``: ``df_primary =
_get_cached_df(sym, primary_tf, ...)``) and then prices the fill at
``float(df_primary["close"].iloc[-1])`` — the close of the very bar the signal
came from.  That is **zero execution latency**: it assumes you can see a close
and trade at exactly that price.  The conventional treatment — and what the
sibling project does — is: signal at the close of ``ts``, fill at the **open of
``ts+1``**.

What this tool measures
-----------------------
The same candidates, the same window, the same costs, run twice:

``A: close``      the shipped behaviour (``backtest.fill_convention: close``).
``B: next_open``  the same signals; the fill price — entry **and** exit — is the
                  open of the bar one row later on the series that priced it
                  (``backtest.fill_convention: next_open``).

Per arm and per timeframe it reports trade count, total return, Sharpe, max
drawdown, the mean per-trade fill-price difference in **basis points** (entry and
exit separately, matched trade-by-trade on ``(symbol, side, opened_at)``), and an
aggregate verdict.

How the exits are treated (this is the part that decides whether the comparison
means anything)
-----------------------------------------------------------------------------
**The same one-bar shift is applied to every exit, and the exit *trigger* is left
exactly where it was.**  For each exit reason the trigger is still observed on the
bar at ``ts`` (the same close/level comparison the ``close`` arm uses), and only
the *fill price* moves:

* ``indicator``       close of the triggering timeframe's bar  ->  that timeframe's next open;
* ``stop_loss`` / ``tp_*``  the barrier **level**              ->  the position timeframe's next open;
* ``max_hold``        close of the position's timeframe bar    ->  its next open;
* ``reduce``          close of the triggering timeframe bar    ->  its next open;
* ``end_of_backtest``  the window's last close — there is no next bar, see below.

Shifting only the entries would measure half of the round trip and would leave
the exits latency-free, i.e. the comparison itself would be meaningless.  The
cost of this choice is stated plainly rather than hidden: under ``next_open`` a
stop no longer fills at the barrier level, so an SL/TP exit differs by more than
a pure price shift.  The tool therefore also reports the **exit-reason mix** of
both arms, so the reader can see which exits moved and by how much.

End of the window
-----------------
The feeder trims every frame to ``date_end`` (``core.backtest.data_feeder``), so
the last decision bar has no following bar.  That is handled explicitly, never
silently:

* an **entry** whose next bar is outside the window is **refused** (no position
  is opened) and counted in ``unfilled_entries`` — a trade that could not be
  filled must not be booked;
* an **exit** with no next bar is priced at the decision bar's close and counted
  in ``window_end_fallback_fills`` (broken down by reason) — the forced
  liquidation at the window end can only happen at the window's last close.

Both counters come back from the engine in
``metrics["fill_convention_accounting"]`` and are printed per arm.

Bounded and deterministic
-------------------------
Real cached parquet only (``data/market/**``, read-only), the GA's own evaluation
flags (``core.ga.fitness.isolated_eval_kwargs``: per-genome slots + per-genome
cash, legacy engine, ``use_live_spread=False`` so nothing is priced from today's
order book), and a fixed seed for the sampled genomes.  Nothing is written outside
``--out``; ``data/binance_trader.db`` and ``strategies/**`` are never touched.

Usage::

    python tools/p9_fill_convention_measure.py                 # default: 2 symbols, 1h+15m
    python tools/p9_fill_convention_measure.py --timeframes 1h,15m,5m --genomes 5
    python tools/p9_fill_convention_measure.py --hand-check     # the tiny by-hand case
    python tools/p9_fill_convention_measure.py --out %TEMP%\\p9.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd                                                        # noqa: E402

INITIAL_BALANCE = 10000.0

#: Every ``strategies/ga_champion_*.yaml`` is read-only input; these are the ones
#: whose native timeframe list contains 1h or 15m (champions without 1h/15m are
#: skipped rather than silently re-timed).
CHAMPION_PREFERENCE = (
    "ga_champion_1780713878",   # native 1h
    "ga_champion_1790844776",   # native 15m + 4h
    "ga_champion_1780642613",   # native 5m + 1h
    "ga_champion_1780666294",   # native 1m + 5m
)


def sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# -- the tiny by-hand case -----------------------------------------------

#: Three hand-chosen 1h bars.  Decision bar = the FIRST bar (its close is
#: 100.0000).  The bar after it opens at 101.5000 (+150.00 bps) — so the two
#: conventions must print exactly those two numbers, and nothing else.
HAND_FRAME = pd.DataFrame(
    {"open": [98.0, 101.5, 103.0],
     "high": [100.5, 104.0, 105.0],
     "low": [97.5, 101.0, 102.0],
     "close": [100.0, 103.5, 104.0]},
    index=pd.to_datetime(["2026-01-01 00:59:59.999",
                          "2026-01-01 01:59:59.999",
                          "2026-01-01 02:59:59.999"]),
)


def hand_check() -> dict:
    """Compute both conventions by hand and return the numbers.

    ``close``     = ``HAND_FRAME["close"].iloc[0]``          = 100.0000
    ``next_open`` = ``HAND_FRAME["open"].iloc[1]``           = 101.5000
    difference    = (101.5 - 100.0) / 100.0 * 10 000        = +150.00 bps
    """
    from core.backtest.fill_convention import decision_bar_position, next_bar_open

    ts = HAND_FRAME.index[0]
    close_fill = float(HAND_FRAME["close"].iloc[0])
    next_open = next_bar_open(HAND_FRAME, ts)
    pos = decision_bar_position(HAND_FRAME, ts)
    bps = (next_open - close_fill) / close_fill * 10_000.0 if next_open else None
    # The last bar has no following bar: the convention must say so, not guess.
    at_end = next_bar_open(HAND_FRAME, HAND_FRAME.index[-1])
    assert pos == 0 and next_open == 101.5 and at_end is None, \
        "the hand-computed case no longer holds — stop and read fill_convention.py"
    return {
        "decision_bar": str(ts),
        "close_fill": close_fill,
        "next_open_fill": next_open,
        "difference_bps": bps,
        "end_of_window_next_open": at_end,
        "assertion": "decision bar 0; close=100.0000; open[1]=101.5000; "
                     f"+{bps:.2f} bps; last bar -> None",
    }


# -- arms ----------------------------------------------------------------

def champion_arms(timeframes: list[str]) -> list[dict]:
    """Shipped champions (read-only) that can run on the measured timeframes."""
    import yaml
    from core.strategy.loader import StrategyConfig

    arms = []
    for stem in CHAMPION_PREFERENCE:
        path = ROOT / "strategies" / f"{stem}.yaml"
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        native = [str(tf) for tf in (raw.get("timeframes") or [])]
        if not set(native) & set(timeframes):
            continue
        config = StrategyConfig(**{k: v for k, v in raw.items() if k != "provenance"})
        arms.append({
            "name": f"champion:{stem}",
            "kind": "champion",
            "strategy": config,
            "native_timeframes": native,
            "sha256_16": sha256_16(path),
        })
    return arms


def genome_arms(n: int, seed: int, timeframes: list[str]) -> list[dict]:
    """``n`` random genomes from ONE fixed seed (the GA's own generator)."""
    from core.ga.genome import chromosome_to_strategy, random_chromosome

    random.seed(seed)
    arms = []
    for i in range(n):
        name = f"genome_s{seed}_{i}"
        chrom = random_chromosome(name)
        config = chromosome_to_strategy(chrom)
        arms.append({"name": name, "kind": "genome", "strategy": config,
                     "native_timeframes": list(config.timeframes or []),
                     "sha256_16": None})
    return arms


def verify_mechanism(symbols, start, end, timeframe) -> dict:
    """Confirm the shift on REAL cached parquet, trade by trade.

    Runs one shipped champion on a short window under both conventions, takes the
    first trade of each arm (they decide on the same bar) and checks the two fill
    prices against the parquet itself:

    * ``close``      → the decision bar's own ``close``;
    * ``next_open``  → the **next row's** ``open`` on the same series.

    This is the same claim the unit test makes on a synthetic series, made here on
    data nobody chose.
    """
    import yaml
    from core.strategy.loader import StrategyConfig

    raw = yaml.safe_load(
        (ROOT / "strategies" / f"{CHAMPION_PREFERENCE[0]}.yaml").read_text(encoding="utf-8"))
    strategy = StrategyConfig(**{k: v for k, v in raw.items() if k != "provenance"})
    _config, engine = engine_stack()

    runs = {c: run_arm(engine, strategy, symbols, start, end, timeframe, c)
            for c in ("close", "next_open")}
    out = {"window": [start, end], "symbols": list(symbols), "timeframe": timeframe,
           "champion": CHAMPION_PREFERENCE[0]}
    a = runs["close"]["trades"][0]
    b = next(t for t in runs["next_open"]["trades"]
             if (t["symbol"], t["side"], t["opened_at"])
             == (a["symbol"], a["side"], a["opened_at"]))
    frame = pd.read_parquet(ROOT / "data" / "market" / a["symbol"] / f"{timeframe}.parquet")
    frame.index = pd.to_datetime(frame.index)
    ts = pd.Timestamp(a["opened_at"])
    pos = int(frame.index.searchsorted(ts, side="right")) - 1
    out.update({
        "symbol": a["symbol"], "side": a["side"], "decision_bar": str(frame.index[pos]),
        "parquet_close": float(frame.iloc[pos]["close"]),
        "parquet_next_open": float(frame.iloc[pos + 1]["open"]),
        "close_arm_entry": float(a["entry_price"]),
        "next_open_arm_entry": float(b["entry_price"]),
        "difference_bps": round((float(b["entry_price"]) - float(a["entry_price"]))
                                / float(a["entry_price"]) * 10_000.0, 4),
    })
    assert out["close_arm_entry"] == out["parquet_close"], out
    assert out["next_open_arm_entry"] == out["parquet_next_open"], out
    out["assertion"] = ("close arm == parquet close of the decision bar; "
                        "next_open arm == parquet open of the NEXT bar")
    return out


def engine_stack():
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager

    Config._instance = None
    config = Config.load("sim")
    # The GA evaluation contract: legacy engine, ML off, and no historical fill
    # priced from today's order book.
    config.backtest_engine_mode = "legacy"
    config.backtest_ml_enabled = False
    config.backtest_live_spread_enabled = False
    bus = EventBus()
    return config, BacktestEngine(config, None, RiskManager(config, bus),
                                  OrderExecutor(config, bus))


def run_arm(engine, strategy, symbols, start, end, timeframe, convention):
    """One engine run with the strategy pinned to a single timeframe."""
    from core.ga.fitness import isolated_eval_kwargs

    run_strategy = strategy.model_copy(update={
        "name": f"{strategy.name}__{timeframe}",
        "timeframes": [timeframe],
    })
    result = engine.run_with_exit_evaluation(
        strategies=[run_strategy], symbols=list(symbols), date_start=start,
        date_end=end, initial_balance=INITIAL_BALANCE, mode="full",
        simulate_ai_weights=False, use_live_spread=False,
        fill_convention=convention, **isolated_eval_kwargs())
    metrics = result.get("metrics") or {}
    return {
        "trades": result.get("trades") or [],
        "metrics": metrics,
        "accounting": metrics.get("fill_convention_accounting") or {},
        "fill_convention": result.get("fill_convention"),
        "error": result.get("error"),
    }


def _reason_mix(trades) -> dict:
    mix: dict[str, int] = {}
    for t in trades:
        r = str(t.get("exit_reason", "?"))
        mix[r] = mix.get(r, 0) + 1
    return dict(sorted(mix.items(), key=lambda kv: -kv[1]))


def _fill_diffs(close_trades, next_trades) -> dict:
    """Trade-by-trade fill-price difference in bps, matched on the signal bar."""
    def key(t):
        return (t.get("symbol"), t.get("side"), str(t.get("opened_at")))

    a = {key(t): t for t in close_trades}
    b = {key(t): t for t in next_trades}
    matched = sorted(set(a) & set(b))
    entry_bps, exit_bps, round_bps = [], [], []
    adv_entry, adv_exit, adv_total = [], [], []
    for k in matched:
        ta, tb = a[k], b[k]
        ep_a, ep_b = float(ta["entry_price"]), float(tb["entry_price"])
        xp_a, xp_b = float(ta["exit_price"]), float(tb["exit_price"])
        # +1 long / -1 short: the "adverse" families are signed so a POSITIVE
        # number always means "next_open filled worse for this trade".
        direction = 1.0 if str(ta.get("side")) == "long" else -1.0
        if ep_a > 0:
            e = (ep_b - ep_a) / ep_a * 10_000.0
            entry_bps.append(e)
            adv_entry.append(e * direction)
        if xp_a > 0:
            x = (xp_b - xp_a) / xp_a * 10_000.0
            exit_bps.append(x)
            adv_exit.append(-x * direction)
        if ep_a > 0 and xp_a > 0:
            round_bps.append(((ep_b - ep_a) + (xp_b - xp_a)) / ep_a * 10_000.0)
            adv_total.append(adv_entry[-1] + adv_exit[-1])

    def _mean(values):
        return round(statistics.fmean(values), 2) if values else None

    def _mean_abs(values):
        return round(statistics.fmean([abs(v) for v in values]), 2) if values else None

    def _share_positive(values):
        if not values:
            return None
        return round(sum(1 for v in values if v > 0) / len(values) * 100.0, 1)

    return {
        "matched_trades": len(matched),
        "close_only": len(set(a) - set(b)),
        "next_open_only": len(set(b) - set(a)),
        # LEVEL difference, no direction: its mean is ~0 by construction, because
        # `open[k+1] - close[k]` is one bar-boundary gap with no systematic sign.
        "mean_entry_diff_bps": _mean(entry_bps),
        "mean_exit_diff_bps": _mean(exit_bps),
        "mean_round_trip_diff_bps": _mean(round_bps),
        "mean_abs_entry_diff_bps": _mean_abs(entry_bps),
        "mean_abs_exit_diff_bps": _mean_abs(exit_bps),
        "mean_abs_round_trip_diff_bps": _mean_abs(round_bps),
        # ADVERSE difference, signed by the trade's own side: >0 = filled worse.
        # The round-trip mean is the per-trade cost of one bar of latency.
        "mean_adverse_entry_bps": _mean(adv_entry),
        "mean_adverse_exit_bps": _mean(adv_exit),
        "mean_adverse_round_trip_bps": _mean(adv_total),
        "adverse_round_trip_share_pct": _share_positive(adv_total),
    }


def _row(arm, timeframe, convention, run) -> dict:
    m = run["metrics"]
    return {
        "arm": arm["name"], "kind": arm["kind"], "timeframe": timeframe,
        "convention": convention,
        "trades": len(run["trades"]),
        "total_return_pct": m.get("total_return_pct"),
        "sharpe_ratio": m.get("sharpe_ratio"),
        "max_drawdown_pct": m.get("max_drawdown_pct"),
        "win_rate_pct": m.get("win_rate_pct"),
        "profit_factor": m.get("profit_factor"),
        "exit_reason_mix": _reason_mix(run["trades"]),
        "accounting": run["accounting"],
        "error": run["error"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--timeframes", default="1h,15m")
    parser.add_argument("--start", default="2025-05-01")
    parser.add_argument("--end", default="2025-07-01")
    parser.add_argument("--genomes", type=int, default=3,
                        help="random genomes drawn from the fixed --seed")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--no-champions", action="store_true")
    parser.add_argument("--hand-check", action="store_true",
                        help="print the tiny by-hand case and exit")
    parser.add_argument("--verify-mechanism", action="store_true",
                        help="check the shift against the real parquet and exit")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    hand = hand_check()
    print("-- hand-computable mechanism check ----------------------------")
    print(f"  frame: {len(HAND_FRAME)} bars, index "
          f"{HAND_FRAME.index[0]} ... {HAND_FRAME.index[-1]}")
    print(f"  decision bar            : {hand['decision_bar']} (row 0)")
    print(f"  A close  = close[0]     : {hand['close_fill']:.4f}")
    print(f"  B next_open = open[1]   : {hand['next_open_fill']:.4f}")
    print(f"  difference              : {hand['difference_bps']:+.2f} bps")
    print(f"  last bar -> next open    : {hand['end_of_window_next_open']} "
          f"(no following bar; the engine refuses the entry / falls back on an exit)")
    if args.hand_check:
        print(json.dumps(hand, indent=2))
        return 0

    if args.verify_mechanism:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()][:2]
        check = verify_mechanism(symbols, args.start, "2025-06-01",
                                 args.timeframes.split(",")[0].strip())
        print("\n-- real-parquet mechanism check -------------------------------")
        for key, value in check.items():
            print(f"  {key:<22}: {value}")
        return 0

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    arms = ([] if args.no_champions else champion_arms(timeframes))
    arms += genome_arms(args.genomes, args.seed, timeframes)
    if not arms:
        print("no arms to evaluate", file=sys.stderr)
        return 2

    config, engine = engine_stack()
    print("\n-- arms -------------------------------------------------------")
    for arm in arms:
        print(f"  {arm['name']:<28} native={arm['native_timeframes']} "
              f"sha256_16={arm['sha256_16']}")
    print(f"  symbols={symbols} window={args.start}->{args.end} "
          f"cost_model: taker_fee_pct={config.backtest_taker_fee_pct} "
          f"spreads={config.backtest_spread_pct} (live_spread off)")

    t0 = time.time()
    rows: list[dict] = []
    diffs: list[dict] = []
    for timeframe in timeframes:
        for arm in arms:
            runs = {}
            for convention in ("close", "next_open"):
                started = time.time()
                runs[convention] = run_arm(engine, arm["strategy"], symbols,
                                           args.start, args.end, timeframe,
                                           convention)
                runs[convention]["seconds"] = round(time.time() - started, 1)
                rows.append(_row(arm, timeframe, convention, runs[convention]))
                rows[-1]["seconds"] = runs[convention]["seconds"]
            d = _fill_diffs(runs["close"]["trades"], runs["next_open"]["trades"])
            d.update({"arm": arm["name"], "timeframe": timeframe})
            diffs.append(d)
            print(f"  [{timeframe}] {arm['name']:<28} "
                  f"trades {len(runs['close']['trades'])}->{len(runs['next_open']['trades'])} "
                  f"entry {d['mean_entry_diff_bps']} bps "
                  f"exit {d['mean_exit_diff_bps']} bps "
                  f"({runs['close']['seconds']}s+{runs['next_open']['seconds']}s)")

    # -- aggregate: mean across arms, per timeframe --
    aggregate = []
    for timeframe in timeframes:
        for convention in ("close", "next_open"):
            sel = [r for r in rows if r["timeframe"] == timeframe
                   and r["convention"] == convention]
            rets = [r["total_return_pct"] for r in sel if r["total_return_pct"] is not None]
            sharpes = [r["sharpe_ratio"] for r in sel if r["sharpe_ratio"] is not None]
            dds = [r["max_drawdown_pct"] for r in sel if r["max_drawdown_pct"] is not None]
            aggregate.append({
                "timeframe": timeframe, "convention": convention,
                "arms": len(sel),
                "trades_total": sum(r["trades"] for r in sel),
                "mean_total_return_pct": round(statistics.fmean(rets), 4) if rets else None,
                "mean_sharpe": round(statistics.fmean(sharpes), 4) if sharpes else None,
                "mean_max_dd_pct": round(statistics.fmean(dds), 4) if dds else None,
            })

    # -- the verdict --
    verdict = {}
    for timeframe in timeframes:
        d = [x for x in diffs if x["timeframe"] == timeframe]

        def _agg(field):
            vals = [x[field] for x in d if x.get(field) is not None]
            return round(statistics.fmean(vals), 2) if vals else None

        # Per-arm direction of the change, not per-metric average: "did this
        # arm's own number get worse?" is what a reader needs to judge whether
        # the effect is unanimous or an average of opposites.
        by_arm: dict = {}
        for row in rows:
            if row["timeframe"] == timeframe:
                by_arm.setdefault(row["arm"], {})[row["convention"]] = row

        def _count_worse(metric):
            worse = total = 0
            for pair in by_arm.values():
                a, b = pair.get("close"), pair.get("next_open")
                if not a or not b:
                    continue
                if a.get(metric) is None or b.get(metric) is None:
                    continue
                total += 1
                if b[metric] < a[metric]:
                    worse += 1
            return worse, total

        worse_ret, arms_compared = _count_worse("total_return_pct")
        worse_sharpe, _ = _count_worse("sharpe_ratio")
        a = next(x for x in aggregate if x["timeframe"] == timeframe and x["convention"] == "close")
        b = next(x for x in aggregate if x["timeframe"] == timeframe and x["convention"] == "next_open")
        unmatched = sum(x["close_only"] + x["next_open_only"] for x in d)
        verdict[timeframe] = {
            "mean_entry_diff_bps": _agg("mean_entry_diff_bps"),
            "mean_exit_diff_bps": _agg("mean_exit_diff_bps"),
            "mean_abs_round_trip_diff_bps": _agg("mean_abs_round_trip_diff_bps"),
            "mean_adverse_entry_bps": _agg("mean_adverse_entry_bps"),
            "mean_adverse_exit_bps": _agg("mean_adverse_exit_bps"),
            "mean_adverse_round_trip_bps": _agg("mean_adverse_round_trip_bps"),
            "mean_total_return_delta_pct": (
                round(b["mean_total_return_pct"] - a["mean_total_return_pct"], 4)
                if a["mean_total_return_pct"] is not None
                and b["mean_total_return_pct"] is not None else None),
            "mean_sharpe_delta": (
                round(b["mean_sharpe"] - a["mean_sharpe"], 4)
                if a["mean_sharpe"] is not None and b["mean_sharpe"] is not None else None),
            "mean_max_dd_delta_pct": (
                round(b["mean_max_dd_pct"] - a["mean_max_dd_pct"], 4)
                if a["mean_max_dd_pct"] is not None
                and b["mean_max_dd_pct"] is not None else None),
            "arms_worse_return": f"{worse_ret}/{arms_compared}",
            "arms_worse_sharpe": f"{worse_sharpe}/{arms_compared}",
            "unmatched_trades_total": unmatched,
        }

    payload = {
        "tool": "p9_fill_convention_measure",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hand_check": hand,
        "symbols": symbols, "timeframes": timeframes,
        "window": {"start": args.start, "end": args.end},
        "seed": args.seed, "genomes": args.genomes,
        "cost_model": {
            "taker_fee_pct": config.backtest_taker_fee_pct,
            "spread_pct": config.backtest_spread_pct,
            "default_spread_pct": config.backtest_default_spread_pct,
            "live_spread_enabled": config.backtest_live_spread_enabled,
            "engine_mode": config.backtest_engine_mode,
            "isolated_eval_kwargs": True,
        },
        "arms": [{k: v for k, v in arm.items() if k != "strategy"} for arm in arms],
        "rows": rows,
        "fill_diffs": diffs,
        "aggregate": aggregate,
        "verdict": verdict,
        "runtime_seconds": round(time.time() - t0, 1),
        "notes": [
            "exits shift by the same one bar as entries; the exit TRIGGER is unchanged",
            "an entry with no next bar inside the window is refused and counted",
            "an exit with no next bar is priced at the decision bar's close and counted",
            "each strategy is pinned to a SINGLE timeframe so the per-timeframe rows "
            "isolate the bar size (its other timeframes act as filters in the native run)",
        ],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    print("\n-- verdict ----------------------------------------------------")
    for tf, v in verdict.items():
        print(f"  {tf}: level entry {v['mean_entry_diff_bps']} bps | level exit "
              f"{v['mean_exit_diff_bps']} bps | |round trip| "
              f"{v['mean_abs_round_trip_diff_bps']} bps | adverse round trip "
              f"{v['mean_adverse_round_trip_bps']} bps | delta_return "
              f"{v['mean_total_return_delta_pct']} pp | delta_Sharpe "
              f"{v['mean_sharpe_delta']} | delta_dd {v['mean_max_dd_delta_pct']} pp "
              f"| worse return {v['arms_worse_return']} arms | worse Sharpe "
              f"{v['arms_worse_sharpe']} arms | unmatched trades "
              f"{v['unmatched_trades_total']}")
    print(f"  runtime {payload['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
