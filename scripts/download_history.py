"""Download historical klines for ANY symbol / interval / date range.

Writes the parquet layout the backtest :class:`core.backtest.data_feeder.DataFeeder`
reads::

    <data-dir>/market/{SYMBOL}/{interval}.parquet
      index  : close_time (datetime64, UTC — one row per closed candle)
      columns: open, high, low, close, volume   (float64)

Data comes from the public mainnet mirror (``config.market_data_host`` /
``--data-host``), **never** ``api.binance.com`` — that host is unreachable from
this deployment and testnet only carries a handful of pairs.

Usage::

    python scripts/download_history.py --symbols BTCUSDT,SOLUSDT \\
        --intervals 1h,4h --start 2024-01-01 --end 2024-03-01
    python scripts/download_history.py --symbols SOLUSDT --intervals 1h \\
        --start 2024-01-01 --end 2024-01-31 --data-dir %TEMP%\\bt_data

``--end YYYY-MM-DD`` is **inclusive** (the whole end day is downloaded).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config  # noqa: E402
from core.market_data.data_client import KLINES_MAX_LIMIT, MarketDataClient, MarketDataError  # noqa: E402
from core.market_data.universe import MARKET_CACHE_SUBDIR  # noqa: E402

#: Intervals the CLI accepts (Binance spot).
VALID_INTERVALS = [
    "1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
]

#: Pause between pages so a long range does not trip the exchange rate limit.
PAGE_SLEEP_S = 0.12


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Binance klines (any symbol / interval / date range) "
                    "into the data/market/{symbol}/{interval}.parquet cache.")
    parser.add_argument("--symbols", required=True,
                        help="Comma-separated symbols, e.g. BTCUSDT,SOLUSDT")
    parser.add_argument("--intervals", default="1h",
                        help="Comma-separated intervals, e.g. 1m,15m,1h,4h,1d")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (inclusive; default: now)")
    parser.add_argument("--data-dir", default=None,
                        help="Root data dir (default: config.data_dir); parquet goes "
                             "to <data-dir>/market/<SYMBOL>/<interval>.parquet")
    parser.add_argument("--data-host", default=None,
                        help="Market data host (default: config.market_data_host)")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Per-request timeout in seconds (default 15)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Symbols downloaded in parallel (default 4)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with an existing parquet file instead of replacing it")
    parser.add_argument("--list-intervals", action="store_true",
                        help="Print the accepted intervals and exit")
    return parser.parse_args(argv)


def parse_day(value: str, *, end_of_day: bool) -> int:
    """``YYYY-MM-DD`` → epoch milliseconds (UTC)."""
    try:
        day = datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError as e:
        raise SystemExit(f"invalid date '{value}': expected YYYY-MM-DD") from e
    day = day.replace(tzinfo=timezone.utc)
    if end_of_day:
        day = day + timedelta(days=1) - timedelta(milliseconds=1)
    return int(day.timestamp() * 1000)


async def download_interval(client: MarketDataClient, symbol: str, interval: str,
                            start_ms: int, end_ms: int, merge: bool,
                            data_dir: Path) -> dict:
    """Page through klines (max 1000/request) and write one parquet file."""
    rows: list[list] = []
    cursor = start_ms
    pages = 0
    while cursor <= end_ms:
        try:
            batch = await client.klines(symbol, interval, limit=KLINES_MAX_LIMIT,
                                        start_time=cursor, end_time=end_ms)
        except MarketDataError as e:
            if not rows:
                return {"symbol": symbol, "interval": interval, "rows": 0, "error": str(e)}
            print(f"\n      {symbol} {interval}: stopped early at page {pages}: {e}")
            break
        if not batch:
            break
        rows.extend(batch)
        pages += 1
        last_open = int(batch[-1][0])
        if len(batch) < KLINES_MAX_LIMIT and last_open >= end_ms:
            break
        nxt = last_open + 1
        if nxt <= cursor:  # defensive: never loop forever on a stuck page
            break
        cursor = nxt
        await asyncio.sleep(PAGE_SLEEP_S)

    if not rows:
        return {"symbol": symbol, "interval": interval, "rows": 0,
                "error": "no klines returned for this range"}

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df = df[["close_time", "open", "high", "low", "close", "volume"]].copy()
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df.set_index("close_time", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    out_dir = data_dir / MARKET_CACHE_SUBDIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{interval}.parquet"
    if merge and path.exists():
        try:
            old = pd.read_parquet(path)
            old.index = pd.to_datetime(old.index)
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        except Exception as e:
            print(f"      {symbol} {interval}: could not merge ({e}); overwriting")
    df.to_parquet(path)

    return {
        "symbol": symbol, "interval": interval, "rows": len(df),
        "pages": pages, "path": str(path),
        "first": df.index[0].isoformat(), "last": df.index[-1].isoformat(),
    }


async def run(args: argparse.Namespace) -> int:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]
    if not symbols:
        raise SystemExit("--symbols must contain at least one symbol")
    bad = [i for i in intervals if i not in VALID_INTERVALS]
    if bad:
        raise SystemExit(f"invalid interval(s) {bad}; accepted: {', '.join(VALID_INTERVALS)}")

    start_ms = parse_day(args.start, end_of_day=False)
    end_ms = (parse_day(args.end, end_of_day=True) if args.end
              else int(datetime.now(timezone.utc).timestamp() * 1000))
    if end_ms <= start_ms:
        raise SystemExit("--end must be after --start")

    config = Config.load("sim")
    data_dir = Path(args.data_dir).resolve() if args.data_dir else Path(config.data_dir)
    if args.data_dir:
        config.data_dir = str(data_dir)
    host = args.data_host or config.market_data_host
    client = MarketDataClient(host, timeout=args.timeout)

    print(f"Host      : {host}")
    print(f"Symbols   : {', '.join(symbols)}")
    print(f"Intervals : {', '.join(intervals)}")
    print(f"Range     : {args.start} .. {args.end or 'now'} (inclusive)")
    print(f"Output    : {data_dir / MARKET_CACHE_SUBDIR}/<SYMBOL>/<interval>.parquet")
    print()

    ok = failed = 0
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def _one(symbol: str, interval: str) -> dict:
        async with sem:
            print(f"  {symbol} {interval} ...", end=" ", flush=True)
            result = await download_interval(client, symbol, interval, start_ms, end_ms,
                                             args.merge, data_dir)
            if result.get("error"):
                print(f"FAILED: {result['error']}")
            else:
                print(f"{result['rows']} rows in {result['pages']} pages "
                      f"({result['first']} .. {result['last']})")
            return result

    try:
        results = await asyncio.gather(
            *(_one(s, i) for s in symbols for i in intervals), return_exceptions=True)
    finally:
        await client.close()

    for r in results:
        if isinstance(r, BaseException):
            print(f"  unexpected error: {r}")
            failed += 1
        elif r.get("error"):
            failed += 1
        else:
            ok += 1
    print(f"\nDone: {ok} file(s) written, {failed} failed.")
    return 1 if failed else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_intervals:
        print(", ".join(VALID_INTERVALS))
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
