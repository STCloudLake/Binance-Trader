"""Data-integrity audit for the cached kline parquet tree (Phase P3 defect).

Why this script exists
----------------------
``data/market/BTCUSDT/1h.parquet`` was found to carry **11 calendar gaps > 1.5 h
with a largest of 1 484 h** (2026-07-29 → 2026-09-29).  The backtest feeder
treats the frame as contiguous, so that splice is a single "bar" holding a
``+27.63 %`` log return.  Squaring it makes it dominate any variance estimator:
an *un-clipped* RiskMetrics EWMA (``lambda = 0.94``, effective window
``1/(1-lambda) = 16.7`` bars) reports ~5.5 %/bar instead of ~0.5 %/bar, i.e. a
10× overstatement that would shrink every vol-targeted position by the same
factor.  ``core/ml/volatility.py`` already winsorises returns
(:data:`DEFAULT_OUTLIER_SIGMA`) for exactly this reason, but nothing told the
operator that the *input* was a spliced series.

What it reports (per ``<symbol>/<interval>.parquet``)
----------------------------------------------------
* bar count and first/last timestamp,
* expected-vs-actual span (``expected = span / interval + 1``) and the number of
  missing bars,
* the number of gaps beyond the threshold, and the largest gap (hours + the
  timestamp it precedes),
* the RiskMetrics EWMA vol per bar, **clipped** (the production default) next to
  the un-clipped number — the pair that exposes the overstatement.

Guard behaviour
---------------
``--check-vol`` refuses to print an un-clipped volatility for a file whose
largest gap exceeds the threshold: it reports the number as ``REFUSED`` instead.
That is the documented fallback for an offline machine — a gap that cannot be
refetched (``scripts/download_history.py --merge``) must not be silently priced
into sizing/stop widths.  The same refusal is what ``--strict`` turns into a
non-zero exit code.

Usage::

    python scripts/check_data_integrity.py
    python scripts/check_data_integrity.py --threshold-bars 1.5 --check-vol
    python scripts/check_data_integrity.py --symbols BTCUSDT --intervals 1h --strict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config  # noqa: E402
from core.market_data.universe import MARKET_CACHE_SUBDIR  # noqa: E402

#: Bar length in hours per interval label (the cache uses Binance spot labels).
INTERVAL_HOURS: dict[str, float] = {
    "1m": 1 / 60, "3m": 3 / 60, "5m": 5 / 60, "15m": 15 / 60, "30m": 0.5,
    "1h": 1.0, "2h": 2.0, "4h": 4.0, "6h": 6.0, "8h": 8.0, "12h": 12.0,
    "1d": 24.0, "3d": 72.0, "1w": 168.0, "1M": 730.0,
}

#: A gap is "beyond the threshold" when it is larger than this many bar lengths.
DEFAULT_GAP_THRESHOLD_BARS = 1.5


def interval_hours(interval: str) -> float | None:
    """Bar length in hours for ``interval`` (``None`` when unknown)."""
    return INTERVAL_HOURS.get(str(interval).strip())


def gap_report(df: pd.DataFrame, interval: str,
               threshold_bars: float = DEFAULT_GAP_THRESHOLD_BARS) -> dict:
    """Structure of one cached frame: span, missing bars and gap statistics.

    ``gaps`` lists ``(timestamp_before_gap, gap_hours)`` for every gap larger
    than ``threshold_bars × bar length``, largest first.  A single-row frame has
    no gaps.  ``expected`` is the bar count a contiguous series would have
    (``span / bar + 1``); ``missing = expected − actual`` — the two disagree only
    when the series is spliced, which is the defect this module measures.
    """
    out = {
        "bars": int(len(df)), "first": None, "last": None, "span_hours": 0.0,
        "expected": int(len(df)), "missing": 0, "gaps": [], "gap_count": 0,
        "largest_gap_hours": 0.0, "largest_gap_at": None,
        "threshold_hours": None, "flagged": False, "interval_hours": None,
    }
    bar = interval_hours(interval)
    out["interval_hours"] = bar
    if bar is None or bar <= 0 or len(df) < 2:
        return out
    idx = pd.to_datetime(pd.Index(df.index))
    order = idx.argsort()
    idx = idx[order]
    out["first"] = idx[0].isoformat()
    out["last"] = idx[-1].isoformat()
    span_h = (idx[-1] - idx[0]).total_seconds() / 3600.0
    out["span_hours"] = round(span_h, 3)
    out["expected"] = int(round(span_h / bar)) + 1
    out["missing"] = max(out["expected"] - int(len(df)), 0)
    threshold = float(threshold_bars) * bar
    out["threshold_hours"] = round(threshold, 3)
    diffs = pd.Series(idx).diff().dt.total_seconds().to_numpy() / 3600.0
    for pos in range(1, len(diffs)):
        gap = float(diffs[pos])
        if gap > threshold:
            out["gaps"].append((idx[pos].isoformat(), round(gap, 3)))
    out["gaps"].sort(key=lambda item: item[1], reverse=True)
    out["gap_count"] = len(out["gaps"])
    if out["gaps"]:
        out["largest_gap_at"], out["largest_gap_hours"] = out["gaps"][0]
    out["flagged"] = out["gap_count"] > 0
    return out


def vol_report(df: pd.DataFrame, window: int = 500) -> dict:
    """Clipped vs un-clipped RiskMetrics EWMA vol (%/bar) for one frame.

    Both numbers come from the shipped estimator
    (``core.ml.volatility.ewma_vol`` with ``sigma=6`` and ``sigma=0``); clipping
    is the production default, so the *ratio* is the measured overstatement a
    data splice would otherwise inject into vol-targeted sizing.

    ``close`` is converted to **log returns** first: ``ewma_vol`` takes a return
    series, and feeding it the price level (the first version of this function
    did) reports the volatility *of the price*, not of the returns.
    """
    from core.ml.volatility import (
        DEFAULT_LAMBDA, DEFAULT_OUTLIER_SIGMA, ewma_vol, log_returns, to_pct)

    if "close" not in df.columns or len(df) < 3:
        return {"clipped_pct": None, "unclipped_pct": None, "ratio": None}
    returns = log_returns(df["close"].astype(float).values)
    if returns.size < 3:
        return {"clipped_pct": None, "unclipped_pct": None, "ratio": None}
    clipped = to_pct(ewma_vol(returns, lam=DEFAULT_LAMBDA, window=window,
                              outlier_sigma=DEFAULT_OUTLIER_SIGMA))
    unclipped = to_pct(ewma_vol(returns, lam=DEFAULT_LAMBDA, window=window,
                                outlier_sigma=0.0))
    ratio = (unclipped / clipped) if clipped > 0 else None
    return {"clipped_pct": clipped, "unclipped_pct": unclipped, "ratio": ratio}


def iter_cached_files(data_dir: Path, symbols=None, intervals=None):
    """Yield ``(symbol, interval, path)`` for every cached parquet, sorted."""
    root = Path(data_dir) / MARKET_CACHE_SUBDIR
    if not root.exists():
        return
    for sym_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        symbol = sym_dir.name
        if symbols and symbol.upper() not in symbols:
            continue
        for path in sorted(sym_dir.glob("*.parquet")):
            interval = path.stem
            if intervals and interval not in intervals:
                continue
            yield symbol, interval, path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report bar counts, spans and calendar gaps per cached "
                    "kline parquet, and flag gaps beyond a configurable threshold.")
    parser.add_argument("--data-dir", default=None,
                        help="Root data dir (default: config.data_dir)")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated filter, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--intervals", default=None,
                        help="Comma-separated filter, e.g. 1h,4h")
    parser.add_argument("--threshold-bars", type=float,
                        default=DEFAULT_GAP_THRESHOLD_BARS,
                        help="Flag gaps larger than this many bar lengths "
                             f"(default {DEFAULT_GAP_THRESHOLD_BARS})")
    parser.add_argument("--check-vol", action="store_true",
                        help="Also print clipped vs un-clipped EWMA vol (%%/bar); "
                             "the un-clipped number is REFUSED for a flagged file")
    parser.add_argument("--force-unclipped", action="store_true",
                        help="Research only: print the un-clipped vol even for a "
                             "flagged series (the guard exists because that number "
                             "must never reach sizing / stop widths)")
    parser.add_argument("--window", type=int, default=500,
                        help="Estimator window for --check-vol (default 500)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 when any gap is flagged")
    return parser.parse_args(argv)


def _split(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {v.strip().upper() for v in value.split(",") if v.strip()}


def main(argv=None) -> int:
    args = parse_args(argv)
    config = Config.load("sim")
    data_dir = Path(args.data_dir).resolve() if args.data_dir else Path(config.data_dir)
    symbols = _split(args.symbols)
    intervals = None
    if args.intervals:
        intervals = {v.strip() for v in args.intervals.split(",") if v.strip()}

    files = list(iter_cached_files(data_dir, symbols, intervals))
    print(f"Cache root      : {data_dir / MARKET_CACHE_SUBDIR}")
    print(f"Gap threshold   : > {args.threshold_bars} x bar length")
    print(f"Files inspected : {len(files)}")
    print()

    header = (f"{'symbol':<10} {'tf':<4} {'bars':>7} {'span_h':>9} {'expected':>9} "
              f"{'missing':>8} {'gaps':>5} {'largest_h':>10}  flag")
    print(header)
    print("-" * len(header))

    flagged_total = 0
    vol_rows: list[tuple[str, str, dict, dict]] = []
    for symbol, interval, path in files:
        try:
            df = pd.read_parquet(path)
        except Exception as e:  # a corrupt file is itself a finding
            print(f"{symbol:<10} {interval:<4} {'-':>7}  unreadable: {e}")
            flagged_total += 1
            continue
        rep = gap_report(df, interval, args.threshold_bars)
        flag = "GAP" if rep["flagged"] else "ok"
        if rep["interval_hours"] is None:
            flag = "?tf"
        if rep["flagged"]:
            flagged_total += 1
        print(f"{symbol:<10} {interval:<4} {rep['bars']:>7} "
              f"{rep['span_hours']:>9.1f} {rep['expected']:>9} {rep['missing']:>8} "
              f"{rep['gap_count']:>5} {rep['largest_gap_hours']:>10.1f}  {flag}")
        for at, hours in rep["gaps"][:5]:
            print(f"{'':<10} {'':<4}   gap {hours:>9.1f} h before {at}")
        if args.check_vol:
            vol_rows.append((symbol, interval, rep, vol_report(df, args.window)))
        if rep["largest_gap_at"] and rep["largest_gap_hours"] > 0:
            print(f"{'':<10} {'':<4}   largest gap starts after {rep['largest_gap_at']}")

    if args.check_vol and vol_rows:
        print()
        vh = (f"{'symbol':<10} {'tf':<4} {'clipped %/bar':>14} "
              f"{'unclipped %/bar':>16} {'ratio':>7}  unclipped")
        print(vh)
        print("-" * len(vh))
        for symbol, interval, rep, vol in vol_rows:
            clipped = f"{vol['clipped_pct']:.4f}" if vol["clipped_pct"] is not None else "-"
            if rep["flagged"] and not args.force_unclipped:
                # The guard: a spliced series must never be priced un-clipped.
                unclipped, verdict = "REFUSED", "gap > threshold"
            elif vol["unclipped_pct"] is not None:
                unclipped = f"{vol['unclipped_pct']:.4f}"
                verdict = "gap > threshold (forced)" if rep["flagged"] else "ok"
            else:
                unclipped, verdict = "-", "no close column"
            ratio = (f"{vol['ratio']:.2f}x"
                     if vol["ratio"] is not None
                     and (args.force_unclipped or not rep["flagged"]) else "-")
            print(f"{symbol:<10} {interval:<4} {clipped:>14} {unclipped:>16} "
                  f"{ratio:>7}  {verdict}")

    print()
    if flagged_total:
        print(f"RESULT: {flagged_total}/{len(files)} file(s) carry a gap beyond "
              f"{args.threshold_bars} x bar length.")
        print("Fix: python scripts/download_history.py --symbols <SYM> "
              "--intervals <tf> --start <first> --end <last> --merge")
    else:
        print(f"RESULT: all {len(files)} file(s) contiguous at "
              f"{args.threshold_bars} x bar length.")
    return 1 if (args.strict and flagged_total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
