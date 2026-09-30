"""Download historical klines for ANY symbol / interval / date range.

Writes the parquet layout the backtest :class:`core.backtest.data_feeder.DataFeeder`
reads::

    <data-dir>/market/{SYMBOL}/{interval}.parquet
      index  : close_time (datetime64, UTC — one row per closed candle)
      columns: open, high, low, close, volume, quote_volume, trade_count   (float64)

``quote_volume`` (quote-asset / USDT traded notional) and ``trade_count`` are
**P6-B** additions taken straight from the kline payload (fields 7 and 8 —
``quoteAssetVolume`` and ``numberOfTrades``).  They are written whenever the
source supplies them; a **pre-P6-B file lacks both columns**, which every reader
must tolerate: :func:`core.market_data.ohlcv_cache.canonical_columns` keeps a
frame's existing columns, ``pandas.concat`` fills the missing side with ``NaN``,
and ``NaN``/absent is the documented "unknown", never a zero.  See
:func:`backfill_cache` (``--backfill``) for the resumable way to add the two
columns to files that were downloaded before P6-B.

``--merge`` unions with the existing file **one row per bar**: a bar already
present under the other timestamp convention (bar-*open* rather than this
script's close time — see :func:`bar_open_keys`) is updated in place instead of
being appended a second time.  Rows this script downloads keep the
``close_time`` index above; a legacy row it does not re-download keeps its own
stamp, so no stored value or label is rewritten.

Data comes from the public mainnet mirror (``config.market_data_host`` /
``--data-host``), **never** ``api.binance.com`` — that host is unreachable from
this deployment and testnet only carries a handful of pairs.

Usage::

    python scripts/download_history.py --symbols BTCUSDT,SOLUSDT \\
        --intervals 1h,4h --start 2024-01-01 --end 2024-03-01
    python scripts/download_history.py --symbols SOLUSDT --intervals 1h \\
        --start 2024-01-01 --end 2024-01-31 --data-dir %TEMP%\\bt_data
    python scripts/download_history.py --backfill --symbols BTCUSDT \\
        --intervals 1h

``--end YYYY-MM-DD`` is **inclusive** (the whole end day is downloaded).
``--backfill`` ignores ``--start``/``--end`` and rewrites the *existing* cached
rows of every affected file from the source, so the two P6-B columns are filled
for history that was downloaded before them.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
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

#: Bar length in nanoseconds per accepted label, used to derive a bar's **open**
#: key from either timestamp convention a cache can hold.  ``1M`` is absent on
#: purpose: calendar months have no fixed length and are floored to the month
#: start instead (see :func:`bar_open_keys`).
INTERVAL_LENGTH_NS = {
    "1s": 1_000_000_000,
    "1m": 60_000_000_000,
    "3m": 180_000_000_000,
    "5m": 300_000_000_000,
    "15m": 900_000_000_000,
    "30m": 1_800_000_000_000,
    "1h": 3_600_000_000_000,
    "2h": 7_200_000_000_000,
    "4h": 14_400_000_000_000,
    "6h": 21_600_000_000_000,
    "8h": 28_800_000_000_000,
    "12h": 43_200_000_000_000,
    "1d": 86_400_000_000_000,
    "3d": 259_200_000_000_000,
    "1w": 604_800_000_000_000,
}

#: kline payload positions (0-based) → cache column.  The full Binance kline is
#: ``[open_time, open, high, low, close, volume, close_time, quote_asset_volume,
#: number_of_trades, taker_buy_base, taker_buy_quote, ignore]``; P6-B persists
#: the two fields that make cross-symbol volume comparable and let a cost model
#: see how the notional was split into trades.
KLINE_FIELDS: dict[int, str] = {
    1: "open", 2: "high", 3: "low", 4: "close", 5: "volume",
    6: "close_time", 7: "quote_volume", 8: "trade_count",
}

#: Columns persisted, in order.  The first five are the original schema; the
#: last two are the P6-B extension (see the module docstring).
CACHE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume",
                                  "quote_volume", "trade_count")

#: How many bars one ``--backfill`` request asks for (Binance's page cap).
BACKFILL_PAGE_LIMIT = 1000


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Binance klines (any symbol / interval / date range) "
                    "into the data/market/{symbol}/{interval}.parquet cache.")
    parser.add_argument("--symbols", required=True,
                        help="Comma-separated symbols, e.g. BTCUSDT,SOLUSDT")
    parser.add_argument("--intervals", default="1h",
                        help="Comma-separated intervals, e.g. 1m,15m,1h,4h,1d")
    parser.add_argument("--start", required=False, default=None,
                        help="Start date YYYY-MM-DD (inclusive); required unless "
                             "--backfill is given")
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
    parser.add_argument("--backfill", action="store_true",
                        help="Rewrite EXISTING cached files to add the P6-B "
                             "quote_volume/trade_count columns (resumable; "
                             "ignores --start/--end)")
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


def _fold_to_bar(stamps: np.ndarray, base: int, step: int, tol: int) -> np.ndarray:
    """Fold int64 ns ``stamps`` onto one key per bar, anchored at ``base``.

    The two conventions are exactly ``step - 1 ms`` apart, so relative to any
    anchor the stamps of a bar land either on the anchor's own grid or 1 ms past
    it.  A stamp a full bar minus 1 ms past the grid is that bar's close time; a
    stamp 1 ms past the grid is the *open* of the bar whose close is on the grid.
    Both fold onto their bar's key.  Grouping never depends on which convention
    the anchor itself uses — only the absolute value of the key does.
    """
    off = step - tol
    res = (stamps - base) % step
    shift = np.where(res >= step - tol, off,
                     np.where((res >= tol) & (res < 2 * tol), -off, 0))
    return stamps - shift


def bar_open_keys(index, interval: str, *, reference=None) -> pd.DatetimeIndex:
    """One key per **bar** for every stamp, whichever convention wrote it.

    Two conventions coexist in the caches this script merges:

    * the one this script writes — Binance's ``close_time``, i.e.
      ``open + length - 1 ms`` (``1h`` → ``…:59:59.999``), and
    * bar-*open* stamps left by an earlier writer (``1h`` → exact hours).

    Measured on the repaired ``data/market/BTCUSDT/1h.parquet`` at revision
    ``fa028be``: 8 767 open-aligned rows plus 2 910 ``:59:59.999`` rows, i.e.
    **55** bars present under both conventions.  A "timestamps 1 ms apart are
    duplicates" rule is wrong here: the 54 one-millisecond-adjacent pairs in that
    file are all the *close* of hour ``H-1`` (``…(H-1):59:59.999``) next to the
    *open* of hour ``H`` (``…H:00:00``) — two different bars — so folding on
    proximity would delete live rows.  The declared bar length is the only sound
    basis, which is what this function uses.

    ``1M`` has no fixed length and is floored to the month start.  Every other
    accepted label is folded by :func:`_fold_to_bar`; the keys then group the
    rows correctly whatever the anchor convention is.  ``reference``, if given,
    must be a **close-time** stamp (the convention this script writes): it only
    sets the grid's origin, so the returned keys are true bar-open times — a
    constant offset that cannot change grouping.

    The keys are for grouping in :func:`merge_bars` only: the surviving row keeps
    its own stamp, so the documented ``close_time`` index is preserved and no
    stored value changes.
    """
    idx = pd.DatetimeIndex(index)
    if idx.size == 0:
        return idx
    if interval == "1M":
        return pd.DatetimeIndex(idx.to_period("M").to_timestamp())
    step = INTERVAL_LENGTH_NS.get(str(interval))
    if not step or step <= 1_000_000:
        return idx  # unknown label: group on the raw stamp
    tol = 1_000_000  # the two conventions differ by exactly 1 ms
    keys = _fold_to_bar(idx.asi8.astype(np.int64), int(idx[0].value), step, tol)
    if reference is not None:
        ref_ns = int(pd.Timestamp(reference).value)
        ref_key = _fold_to_bar(np.array([ref_ns], dtype=np.int64),
                               int(idx[0].value), step, tol)[0]
        keys = keys - (int(ref_key) - (ref_ns - (step - tol)))
    return pd.DatetimeIndex(keys)


def merge_bars(old: pd.DataFrame, new: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Union two cache frames with **one row per bar**, sorted by the index.

    Replaces the previous exact-timestamp union (``~df.index.duplicated``), which
    only removed a duplicate when both rows carried the *same* stamp: a bar
    stored once as a bar-open stamp and once as this script's close stamp
    survived as two rows, so a second ``--merge`` (or a repair that mixed the
    conventions) grew the file instead of updating it.  Rows are grouped by the
    derived :func:`bar_open_keys`; rows from ``new`` win a conflict — it is the
    fresher download, which preserves the old ``keep="last"`` intent (and makes a
    second merge idempotent) — and every non-conflicting row keeps its own stamp.
    """
    parts = []
    for src, frame in enumerate((old, new)):
        part = frame.copy()
        part["_src"] = src
        parts.append(part)
    combined = pd.concat(parts)
    ref = new.index[0] if len(new) else None
    # Assign the int64 keys (a plain ndarray): assigning an Index would be an
    # align-by-label operation, and the derived keys deliberately repeat.
    combined["_bar"] = bar_open_keys(combined.index, interval, reference=ref).asi8
    combined = combined.sort_values(["_bar", "_src"], kind="stable")
    combined = combined[~combined["_bar"].duplicated(keep="last")]
    return combined.drop(columns=["_src", "_bar"]).sort_index()


def klines_to_frame(rows) -> pd.DataFrame:
    """Binance kline rows → the cached frame (index ``close_time``, P6-B columns).

    The **one** kline→parquet field mapping (P6-B): ``quote_volume`` (field 7) and
    ``trade_count`` (field 8) are kept alongside the original five columns, so the
    download path and the backfill path cannot disagree about what a cached bar
    contains.  ``close_time`` becomes the index in UTC; a duplicate stamp keeps
    the last row; the result is sorted.
    """
    if not rows:
        return pd.DataFrame(columns=list(CACHE_COLUMNS))
    width = max(KLINE_FIELDS) + 1
    padded = [list(row)[:width] + [None] * (width - len(list(row))) for row in rows]
    df = pd.DataFrame(padded)
    df = df.rename(columns={i: name for i, name in KLINE_FIELDS.items()})
    # `close_time` is the INDEX, not a data column, so it is selected separately
    # from CACHE_COLUMNS.
    wanted = ["close_time"] + list(CACHE_COLUMNS)
    df = df[[c for c in wanted if c in df.columns]].copy()
    if "close_time" not in df.columns:
        raise ValueError("kline rows carry no close_time field (position 6)")
    for column in CACHE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms")
    for column in CACHE_COLUMNS:
        df[column] = df[column].astype(float)
    df.set_index("close_time", inplace=True)
    return df[~df.index.duplicated(keep="last")].sort_index()


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

    df = klines_to_frame(rows)

    out_dir = data_dir / MARKET_CACHE_SUBDIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{interval}.parquet"
    if merge and path.exists():
        try:
            old = pd.read_parquet(path)
            old.index = pd.to_datetime(old.index)
            df = merge_bars(old, df, interval)
        except Exception as e:
            print(f"      {symbol} {interval}: could not merge ({e}); overwriting")
    df.to_parquet(path)

    return {
        "symbol": symbol, "interval": interval, "rows": len(df),
        "pages": pages, "path": str(path),
        "first": df.index[0].isoformat(), "last": df.index[-1].isoformat(),
    }


#: Bars fetched per backfill request.  The source's maximum page is 1000, and a
#: bigger page is both fewer round-trips and a smaller window for the live
#: process to append into (which is why the backfill refuses to rewrite a file
#: that moved instead of clobbering it) — the same cap as :data:`BACKFILL_PAGE_LIMIT`.
class CacheMovedError(RuntimeError):
    """The cache file changed under the backfill — refusing to rewrite it."""


def _row_span(frame: pd.DataFrame, columns) -> tuple[int, object, object]:
    """``(rows, first_stamp, last_stamp)`` of ``frame`` over ``columns``.

    Rows are counted over the *frame*, not over one column, so an appended bar
    that carries no ``quote_volume`` is still visible as a change.  Stamps come
    from ``close_time`` when it is a column (the REST-frame shape) and from the
    index otherwise (the cached shape).
    """
    rows = int(len(frame))
    if rows == 0:
        return 0, None, None
    if "close_time" in frame.columns:
        stamps = pd.to_datetime(frame["close_time"], errors="coerce")
        return rows, stamps.iloc[0], stamps.iloc[-1]
    stamps = pd.DatetimeIndex(frame.index)
    return rows, stamps[0], stamps[-1]


def _missing_quote_volume(frame: pd.DataFrame) -> int:
    """Rows whose ``quote_volume`` is absent — a missing column counts every row."""
    if len(frame) == 0:
        return 0
    if "quote_volume" not in frame.columns:
        return int(len(frame))
    return int(pd.isna(frame["quote_volume"]).sum())


def backfill_plan(frame: pd.DataFrame) -> list[dict]:
    """The requests ``--backfill`` must make to fill ``frame``'s missing columns.

    Contiguous runs (gaps wider than four bar lengths split a run) of bars whose
    ``quote_volume`` is absent, oldest first, each as
    ``{"start_time": ms, "end_time": ms, "rows": n}`` where the window is the
    **bar interval itself**: ``start_time`` is the bar's *open* time and
    ``end_time`` its close (``open + length − 1 ms``).  Deriving both from the
    stored stamp is what makes the fetch land on the same bar:

    * a file whose stamps are bar opens (``…06:00:00``, what this checkout's live
      cache holds) uses them directly — a request window of ``[open, close]``
      covers exactly that bar;
    * a file whose stamps are Binance ``close_time`` (``…06:59:59.999``, the
      convention :func:`download_interval` writes) has ``stamp − length + 1 ms``
      as its open, so the same window still covers the bar.

    Pure and network-free, so resumability can be tested without a live host.
    """
    if frame is None or len(frame) == 0:
        return []
    idx = pd.DatetimeIndex(frame.index)
    step_ms = None
    if len(idx) > 1:
        step_ms = int((idx[1] - idx[0]).total_seconds() * 1000)
    missing = (pd.isna(frame["quote_volume"]) if "quote_volume" in frame.columns
               else pd.Series(True, index=frame.index))
    runs: list[dict] = []
    current: list[pd.Timestamp] = []
    previous = None
    for stamp, absent in zip(idx, missing.to_numpy()):
        if not absent:
            if current:
                runs.append(_run_entry(current, step_ms))
                current = []
            previous = stamp
            continue
        if previous is not None and step_ms and current:
            gap = int((stamp - previous).total_seconds() * 1000)
            if gap > 4 * step_ms:
                runs.append(_run_entry(current, step_ms))
                current = []
        current.append(stamp)
        previous = stamp
    if current:
        runs.append(_run_entry(current, step_ms))
    return runs


def _run_entry(stamps: list, step_ms: int | None) -> dict:
    """One backfill request window covering ``stamps`` (stored stamps, in ms).

    ``start_time`` is the **open** of the run's first bar and ``end_time`` the
    close of its last.  The open is resolved from the stored stamp through
    :func:`_bar_starts`, so both timestamp conventions produce the same window:
    a bar-open stamp *is* the open, while Binance's ``close_time`` stamp is the
    open plus one bar (the exchange's ``open_time`` field is the bar's start and
    its ``close_time`` is ``open + length − 1 ms``).  Treating a ``close_time``
    stamp as an open is what made a first backfill stop one bar short of the end
    of the file.

    ``anchor`` is the run's first stored stamp, for callers that want it.
    """
    length = step_ms if step_ms and step_ms > 0 else 3_600_000
    interval = _interval_for_length(length)
    starts = _bar_starts(pd.DatetimeIndex(stamps), interval)
    start = int(starts[0]) // 1_000_000
    end = int(starts[-1]) // 1_000_000 + length - 1
    return {"start_time": start, "end_time": end, "rows": len(stamps),
            "anchor": pd.Timestamp(stamps[0])}


def _interval_for_length(length_ms: int) -> str:
    """The interval label whose bar length is ``length_ms`` (``"1h"`` fallback)."""
    wanted = int(length_ms) * 1_000_000
    for label, ns in INTERVAL_LENGTH_NS.items():
        if ns == wanted:
            return label
    return "1h"


def _bar_starts(index, interval: str) -> np.ndarray:
    """Candidate bar-open ns for every stamp of ``index`` (absolute, not anchored).

    A fixed-length interval puts a well-formed stamp in one of two residues:
    ``0`` (the bar-*open* convention) or ``step - 1 ms`` (Binance's ``close_time``
    — a bar's close is its open plus one bar per the exchange's ``open_time``
    field).  Everything else is off-grid and is passed through as its own
    candidate, so :func:`_align_to_stored_grid` can still match it against the
    stored stamps.

    Unlike :func:`bar_open_keys` — which is anchored on the frame's own first
    stamp so that two *whole frames* group consistently — this returns absolute
    candidates, which is what lets a ``close_time`` fetch be matched against a
    bar-open file.  The two must not be interchanged: with a close-time anchor,
    ``bar_open_keys`` folds the previous bar's close and the next bar's open onto
    the *same* key (documented there), which is exactly the collision that made
    an earlier version of this alignment match nothing.
    """
    stamps = pd.DatetimeIndex(index)
    ns = np.asarray(stamps.asi8, dtype=np.int64)
    if str(interval) == "1M":
        return np.asarray(
            pd.DatetimeIndex(stamps.to_period("M").to_timestamp()).asi8,
            dtype=np.int64)
    step = INTERVAL_LENGTH_NS.get(str(interval))
    if not step or step <= 1_000_000:
        return ns
    residue = ns % step
    # `close_time` → its own bar's open; a bar-open stamp → itself; an off-grid
    # stamp keeps itself (it can still be an exact stored stamp).
    return np.where(residue >= step - 1_000_000, ns - (step - 1_000_000), ns)


def _match_fetched_to_stored(fetched_index, stored_index, interval: str,
                             fetched_starts: np.ndarray) -> dict[int, int]:
    """``{fetched row position: the stored stamp of the same bar}``.

    Resolution is by **bar open**, tried in order of increasing doubt, so a
    well-formed frame matches on its first candidate and a frame written by a
    writer that anchored its grid 1 ms off still matches on the last one:

    1. the absolute bar open (:func:`_bar_starts` — handles both conventions);
    2. that open shifted by ``+1 ms`` (a bar-open stamp anchored 1 ms early);
    3. the open shifted by ``-1 ms`` (the mirror case).

    A fetched row whose bar the file does not hold is simply absent from the
    result: a backfill adds *columns* to the existing history, never bars.
    """
    stored = pd.DatetimeIndex(stored_index)
    stored_by_start: dict[int, int] = {}
    for start, stamp in zip(_bar_starts(stored, interval),
                            np.asarray(stored.asi8, dtype=np.int64)):
        stored_by_start[int(start)] = int(stamp)
    if not stored_by_start:
        return {}
    tolerance = 0 if str(interval) == "1M" else 1_000_000
    matches: dict[int, int] = {}
    for position, start in enumerate(fetched_starts):
        key = int(start)
        if key in stored_by_start:
            matches[position] = stored_by_start[key]
            continue
        if tolerance:
            for delta in (tolerance, -tolerance):
                hit = stored_by_start.get(key + delta)
                if hit is not None:
                    matches[position] = hit
                    break
    return matches


def _align_to_stored_grid(fetched: pd.DataFrame, stored_index,
                          interval: str) -> pd.DataFrame:
    """Re-index ``fetched`` onto the timestamps the file actually stores.

    The fetch always returns Binance ``close_time`` stamps while the file may
    store bar opens (or the reverse), so the same bar has two possible stamps.
    Both sides are resolved to their absolute bar open (:func:`_bar_starts`) and
    matched by :func:`_match_fetched_to_stored`; the surviving rows are re-stamped
    with the stored stamp of their bar, so the result can be assigned onto the
    stored index without a convention assumption.

    Returns an **empty** frame — the caller's signal to leave the file untouched —
    when the two series share no bars.
    """
    if len(fetched) == 0 or len(stored_index) == 0:
        return fetched.iloc[0:0]
    fetched_starts = _bar_starts(fetched.index, interval)
    matches = _match_fetched_to_stored(fetched.index, stored_index, interval,
                                       fetched_starts)
    if not matches:
        return fetched.iloc[0:0]
    positions = sorted(matches)
    aligned = fetched.iloc[positions]
    aligned.index = pd.DatetimeIndex([matches[p] for p in positions])
    return aligned[~aligned.index.duplicated(keep="last")]


async def backfill_cache(client: MarketDataClient, symbol: str, interval: str,
                         data_dir: Path) -> dict:
    """Add the P6-B columns to an **existing** cache file, in place and resumably.

    Why a rewrite is necessary at all: ``quote_volume`` and ``trade_count`` come
    from the kline payload, so they exist for a bar only if the bar was
    downloaded by a P6-B writer.  They cannot be derived from anything already on
    disk (``volume × close`` is the documented *proxy*, not the source number),
    so a pre-P6-B file's history has to be re-read from the source.  What the
    backfill does **not** do is re-download history it does not need: it requests
    exactly the contiguous runs of bars with a missing ``quote_volume``.

    Guarantees (all of them are properties of this function, not of the caller):

    * **never truncates** — the result is ``merge_bars(existing, fetched)``, the
      same one-row-per-bar union ``--merge`` uses, so a bar the source no longer
      serves keeps its stored row and stamp;
    * **never changes a stored price/volume** — only ``quote_volume`` and
      ``trade_count`` are taken from the fetched rows; a fetched row is merged as
      ``existing_row + the two new columns``;
    * **resumable** — a second run finds ``missing == 0`` runs, makes zero
      requests and reports ``done``; an interrupted run keeps every filled block
      it already wrote;
    * **refuses a moved file** — if the file's row count or last stamp changed
      while the requests were in flight (the live service appends to
      ``data/market/**`` as it trades), nothing is written and
      :class:`CacheMovedError` is raised; the caller retries.  Clobbering a
      concurrently-appended live cache is the one irreversible mistake here.
    """
    path = data_dir / MARKET_CACHE_SUBDIR / symbol / f"{interval}.parquet"
    if not path.exists():
        return {"symbol": symbol, "interval": interval, "error": "no cached file"}
    existing = pd.read_parquet(path)
    if len(existing) == 0:
        return {"symbol": symbol, "interval": interval, "rows": 0,
                "missing_before": 0, "filled": 0, "pages": 0, "done": True}
    existing.index = pd.to_datetime(existing.index)
    carries_extended = all(c in existing.columns for c in CACHE_COLUMNS[5:])
    missing_before = _missing_quote_volume(existing)
    if carries_extended and missing_before == 0:
        return {"symbol": symbol, "interval": interval, "rows": int(len(existing)),
                "missing_before": 0, "filled": 0, "pages": 0, "done": True}

    rows_before, first_before, last_before = _row_span(existing, CACHE_COLUMNS)
    plan = backfill_plan(existing)
    fetched: list[list] = []
    pages = 0
    for request in plan:
        cursor = request["start_time"]
        while cursor <= request["end_time"]:
            try:
                batch = await client.klines(symbol, interval,
                                            limit=BACKFILL_PAGE_LIMIT,
                                            start_time=cursor,
                                            end_time=request["end_time"])
            except MarketDataError as e:
                return {"symbol": symbol, "interval": interval,
                        "error": f"source error after {pages} page(s): {e}"}
            if not batch:
                break
            fetched.extend(batch)
            pages += 1
            last_open = int(batch[-1][0])
            if len(batch) < BACKFILL_PAGE_LIMIT:
                break
            cursor = last_open + 1
            await asyncio.sleep(PAGE_SLEEP_S)
    if pages == 0:
        return {"symbol": symbol, "interval": interval, "rows": int(len(existing)),
                "missing_before": missing_before, "filled": 0, "pages": 0,
                "done": False,
                "error": "no run needed a request (nothing filled)"}

    fresh = klines_to_frame(fetched)
    fresh.index = pd.to_datetime(fresh.index)
    # Only the two new columns come from the source; everything already stored is
    # authoritative.  `merge_bars(old, new)` keeps the *new* row on a bar
    # conflict, so the "new" side is the PATCHED frame (stored prices/volumes +
    # fetched quote columns), never the raw fetch — a re-downloaded bar must not
    # rewrite a stored open/high/low/close/volume.
    #
    # The fetched stamps are in the **source's** convention (Binance
    # ``close_time``), the file's may be either (``…:59:59.999`` or bar opens), so
    # the fetched bars are folded onto the stored grid by *position* before the
    # reindex: both are contiguous bar sequences, and the first fetched bar
    # carries its own bar key, which is the same shift for every row.
    extension = fresh[[c for c in CACHE_COLUMNS[5:] if c in fresh.columns]]
    aligned = _align_to_stored_grid(extension, existing.index, interval)
    patched = existing.copy()
    for column in CACHE_COLUMNS[5:]:
        if column not in aligned.columns:
            patched[column] = np.nan
            continue
        values = pd.Series(np.nan, index=existing.index, dtype=float)
        values.loc[aligned.index] = aligned[column].astype(float).to_numpy()
        patched[column] = values
    merged = merge_bars(existing, patched, interval)
    filled = int(missing_before - _missing_quote_volume(merged))

    current = pd.read_parquet(path)
    rows_after, _first_after, last_after = _row_span(current, CACHE_COLUMNS)
    if (rows_after, last_after) != (rows_before, last_before):
        raise CacheMovedError(
            f"{symbol} {interval}: cache changed while backfilling "
            f"({rows_before}→{rows_after} rows, last {last_before}→{last_after}); "
            f"nothing written — re-run the backfill")
    merged.to_parquet(path)
    return {"symbol": symbol, "interval": interval, "rows": int(len(merged)),
            "missing_before": missing_before, "filled": filled, "pages": pages,
            "done": False}


async def run(args: argparse.Namespace) -> int:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]
    if not symbols:
        raise SystemExit("--symbols must contain at least one symbol")
    bad = [i for i in intervals if i not in VALID_INTERVALS]
    if bad:
        raise SystemExit(f"invalid interval(s) {bad}; accepted: {', '.join(VALID_INTERVALS)}")

    if args.backfill:
        start_ms = end_ms = 0  # unused; `--backfill` derives its own windows
    else:
        if not args.start:
            raise SystemExit("--start is required (unless --backfill is given)")
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
    if args.backfill:
        print(f"Mode      : --backfill (existing rows only; --start/--end ignored)")
    else:
        print(f"Range     : {args.start} .. {args.end or 'now'} (inclusive)")
    print(f"Output    : {data_dir / MARKET_CACHE_SUBDIR}/<SYMBOL>/<interval>.parquet")
    print()

    ok = failed = 0
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def _one(symbol: str, interval: str) -> dict:
        async with sem:
            if args.backfill:
                print(f"  {symbol} {interval} (backfill) ...", end=" ", flush=True)
                result = await backfill_cache(client, symbol, interval, data_dir)
                if result.get("error"):
                    print(f"SKIPPED: {result['error']}")
                elif result.get("done"):
                    print(f"already complete ({result['rows']} rows, "
                          f"{result['missing_before']} missing before)")
                else:
                    print(f"{result['rows']} rows, filled {result['filled']} "
                          f"of {result['missing_before']} missing "
                          f"quotes ({result['pages']} pages)")
                return result
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
