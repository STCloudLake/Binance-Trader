"""In-memory OHLCV cache whose parquet store is a durable union.

The store is ``<data_dir>/market/<SYMBOL>/<interval>.parquet`` — a frozen layout
shared with ``core.backtest.data_feeder``, the GA walk-forward, the web routes and
``scripts/check_data_integrity.py``.  Two durability rules live here:

1. every write is the **union** of the on-disk frame and the in-memory frame, so a
   writer that only knows a shorter window can never truncate a longer history
   (:meth:`OHLVCache.save` → :func:`merge_history`);
2. the union is **interval-aware**: one bar stored under the two timestamp
   conventions a cache can hold (bar-*open* stamps vs Binance ``close_time``,
   ``open + bar_length - 1 ms``) is *one* row, not two (:func:`bar_keys`).

Measured defect behind rule 2 (live ``data/market/BTCUSDT/1h.parquet``, revision
``0542e02``): 11 677 rows = 8 767 bar-open stamps + 2 910 ``HH:59:59.999`` close
stamps, i.e. **55 bars stored twice**.  An exact-timestamp union cannot see such a
pair — its two stamps are 3 599.999 s apart — so the running service's periodic
flush put the duplicate rows back after a repair had removed them.

A *proximity* rule ("stamps closer than a fraction of the interval are the same
bar") fixes nothing here and destroys data: the 54 one-millisecond-adjacent pairs
in that same file are the close of hour ``H-1`` beside the open of hour ``H`` —
two *different* bars.  A 1 s tolerance would collapse all 54 of those real hours
while leaving all 55 genuine duplicates in place.  The declared bar length is the
only sound basis, so :func:`bar_keys` folds on it (the same rule
``scripts/download_history.py`` already applies on the download path).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from loguru import logger


#: Bar length in nanoseconds per interval label — a verbatim mirror of
#: ``scripts.download_history.INTERVAL_LENGTH_NS``.  The table is duplicated
#: because ``scripts/*`` sits downstream of ``core`` and cannot be imported here;
#: ``tests/test_cache_durability.py`` pins the two copies equal so they cannot
#: drift apart unnoticed.  ``1M`` is deliberately absent: calendar months have no
#: fixed length, and :func:`bar_keys` floors those to the month start instead.
BAR_LENGTH_NS: dict[str, int] = {
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

#: The two conventions are exactly ``bar_length - 1 ms`` apart, so this 1 ms is
#: the *residue* tolerance of :func:`bar_keys` — not a proximity tolerance.
CONVENTION_GAP_NS = 1_000_000


def step_ns(interval: str) -> int | None:
    """Fixed bar length of ``interval`` in ns (``None``: unknown or calendar)."""
    step = BAR_LENGTH_NS.get(str(interval))
    if step is None or step <= CONVENTION_GAP_NS:
        return None
    return int(step)


def _fold_to_bar(stamps: np.ndarray, base: int, step: int, tol: int) -> np.ndarray:
    """Fold int64-ns ``stamps`` onto one key per bar, gridded from ``base``.

    Mirrors ``scripts.download_history._fold_to_bar``: relative to any anchor a
    bar's two conventions land either on the grid (a bar-*open* stamp) or exactly
    ``step - tol`` past it (that bar's close time), so each folds onto its bar.
    A stamp in neither residue keeps its own value (its own bar).
    """
    off = step - tol
    res = (stamps - base) % step
    shift = np.where(res >= step - tol, off,
                     np.where((res >= tol) & (res < 2 * tol), -off, 0))
    return stamps - shift


def _grid_anchor_ns(ns: np.ndarray, step: int) -> int:
    """Anchor for :func:`_fold_to_bar`: the first stamp in a canonical residue.

    "Canonical" is measured against the epoch grid — residue ``0`` (bar open) or
    ``step - 1 ms`` (bar close).  A file whose first row is an off-grid stamp would
    otherwise fold the whole frame onto that odd residue and stop recognising the
    two conventions (the anchor's residue sets the grid), so the anchor is taken
    from the first *canonical* stamp; a frame with none keeps ``ns[0]`` and simply
    groups on the raw stamps.
    """
    res = ns % step
    canonical = (res == 0) | (res == step - CONVENTION_GAP_NS)
    if canonical.any():
        return int(ns[int(np.argmax(canonical))])
    return int(ns[0])


def bar_keys(index, interval: str) -> pd.DatetimeIndex | None:
    """One key per **bar** for every stamp, whichever convention wrote it.

    ``None`` when the label has no usable grid (unknown label, or a bar length at
    or below the 1 ms convention gap e.g. a hypothetical ``1ms``); the caller then
    falls back to the exact-timestamp union.  ``1M`` floors to the calendar month
    start (no fixed length), matching the download script.

    The keys are for *grouping*: two rows sharing a key are the same bar under the
    two conventions.  They are expressed as bar-**open** times whenever the grid is
    epoch-aligned (all Binance fixed-length labels), so two frames are comparable
    whatever convention each uses.  They are never written back — the surviving row
    is stamped separately by :func:`dominant_convention` / :func:`_canonical_ns`.
    Only a label whose grid is *not* epoch-aligned (e.g. ``1w``, if Binance ever
    aligned it otherwise) leaves them relative to the anchor's own convention; the
    grouping is unaffected either way.
    """
    idx = pd.DatetimeIndex(index)
    if idx.size == 0:
        return idx
    if str(interval) == "1M":
        return pd.DatetimeIndex(idx.to_period("M").to_timestamp())
    step = step_ns(interval)
    if step is None:
        return None
    ns = idx.asi8.astype(np.int64)
    base = _grid_anchor_ns(ns, step)
    keys = _fold_to_bar(ns, base, step, CONVENTION_GAP_NS)
    # An anchor in the close residue grids the keys on the *close* stamps; move
    # them onto the bar-open grid so the keys mean the same thing everywhere.
    if base % step == step - CONVENTION_GAP_NS:
        keys = keys - (step - CONVENTION_GAP_NS)
    return pd.DatetimeIndex(keys)


def _convention_ns(stamp_ns: int, step: int) -> str | None:
    """``"open"`` / ``"close"`` / ``None`` for one stamp on the epoch bar grid."""
    res = stamp_ns % step
    if res == 0:
        return "open"
    if res == step - CONVENTION_GAP_NS:
        return "close"
    return None


def dominant_convention(index, interval: str) -> str | None:
    """Which timestamp convention most rows of ``index`` use.

    ``"open"`` when bar-open stamps outnumber ``close_time`` stamps, ``"close"``
    in the opposite case, ``None`` when the label has no fixed length or no row
    sits in either canonical residue.  A tie resolves to the convention of the
    newest stamp, so the answer is a deterministic function of the frame.
    """
    step = step_ns(interval)
    if step is None:
        return None
    ns = pd.DatetimeIndex(index).asi8.astype(np.int64)
    if ns.size == 0:
        return None
    res = ns % step
    n_open = int((res == 0).sum())
    n_close = int((res == step - CONVENTION_GAP_NS).sum())
    if n_open == 0 and n_close == 0:
        return None
    if n_open == n_close:
        return _convention_ns(int(ns[int(np.argmax(ns))]), step)
    return "open" if n_open > n_close else "close"


def _canonical_ns(stamp_ns: int, step: int, convention: str) -> int | None:
    """``stamp_ns`` re-expressed in ``convention`` for the bar it belongs to.

    ``None`` for an off-grid stamp (its bar cannot be located without guessing).
    """
    own = _convention_ns(stamp_ns, step)
    if own is None:
        return None
    bar_open = stamp_ns if own == "open" else stamp_ns - (step - CONVENTION_GAP_NS)
    if convention == "open":
        return bar_open
    return bar_open + (step - CONVENTION_GAP_NS)


def merge_history(existing: pd.DataFrame | None,
                  incoming: pd.DataFrame | None,
                  interval: str | None = None) -> pd.DataFrame | None:
    """Union two OHLCV frames over their ``close_time`` index — never truncate.

    ``existing`` is normally the frame on disk and ``incoming`` the fresh window
    (a REST page, a live candle, the prefetch result).  The result keeps **every
    bar** either side holds ("a longer existing history is never truncated"), so a
    repair written by ``scripts/download_history.py --merge`` cannot be undone by a
    writer that only knows a shorter window.

    Duplicate *bars* keep the *incoming* row (``keep="last"``), the same
    "newest wins" convention as :meth:`OHLVCache.append_candle`; the index is
    sorted, so the output is a deterministic function of the two inputs.

    ``interval`` — the interval-aware rule (documented, and the reason this
    function changed behaviour)
    -------------------------------------------------------------------------
    With a fixed-length ``interval`` the union groups by **bar**
    (:func:`bar_keys`), not by exact timestamp: the same bar stored once with a
    bar-open stamp and once with Binance's ``close_time`` collapses to one row
    instead of re-expanding the file on the next flush.  A row whose bar had a
    conflict is written back in the **file's dominant convention**
    (:func:`dominant_convention` of ``existing``, falling back to the merged set,
    then to leaving the stamp alone), so the store converges on one convention.
    Rows whose bar did **not** conflict keep their own stamp and label untouched —
    exactly what ``scripts/download_history.py`` promises — and a genuine second
    bar is never collapsed, however close its stamp is (the 54 one-millisecond
    neighbours in the live ``1h`` cache are 55 distinct hours).

    ``interval=None`` (and labels with no usable grid, e.g. an unknown timeframe)
    keeps the historical exact-timestamp union; :meth:`OHLVCache.save` always
    passes its interval, so the write path is always interval-aware.

    Either side may be ``None``/empty, in which case the other is returned
    unchanged (``None`` when both are).
    """
    if existing is None or len(existing) == 0:
        return incoming
    if incoming is None or len(incoming) == 0:
        return existing
    left, right = existing.copy(), incoming.copy()
    left.index = pd.to_datetime(left.index)
    right.index = pd.to_datetime(right.index)

    step = step_ns(interval) if interval is not None else None
    aware = interval is not None and (step is not None or str(interval) == "1M")
    left["_src"], right["_src"] = 0, 1
    combined = pd.concat([left, right])
    keys = bar_keys(combined.index, interval) if aware else None
    if keys is None:
        # Exact-timestamp union: the historical behaviour, kept for callers that
        # do not pass an interval and for labels with no usable bar grid.
        merged = combined.drop(columns=["_src"])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index(inplace=True)
        return merged
    # Assign the int64 keys as a plain ndarray, in the frame's own row order: the
    # derived keys repeat on purpose, and assigning an Index would align by label.
    combined["_bar"] = keys.asi8
    repeated = combined["_bar"].duplicated(keep=False)
    conflicted = set(combined.loc[repeated, "_bar"].tolist())
    # Row order: bar first, and within a bar the incoming side last so that
    # ``duplicated(keep="last")`` implements "the fresher row wins".
    combined = combined.sort_values(["_bar", "_src"], kind="stable")
    combined = combined[~combined["_bar"].duplicated(keep="last")]

    if conflicted and step is not None:
        # "The file's dominant convention" — measured on the on-disk frame, with
        # the merged set as the fallback when the file itself is off-grid.
        convention = (dominant_convention(left.index, interval)
                      or dominant_convention(combined.index, interval))
        if convention is not None:
            bars = combined["_bar"].to_numpy()
            ns = combined.index.asi8.astype(np.int64)
            canon = [(_canonical_ns(int(s), step, convention)
                      if int(b) in conflicted else None)
                     for b, s in zip(bars, ns)]
            if any(c is not None and c != int(s) for c, s in zip(canon, ns)):
                combined.index = pd.DatetimeIndex(
                    [int(s) if c is None else int(c) for c, s in zip(canon, ns)])

    combined = combined.drop(columns=["_bar", "_src"]).sort_index()
    # Invariant guard: canonicalising can never merge two *different* bars (their
    # canonical stamps are distinct grid points), so this drop is a no-op on every
    # input the rule above defines; it just keeps "one row per stamp" true even if
    # a caller hands in an off-grid frame.
    return combined[~combined.index.duplicated(keep="last")]


class OHLVCache:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._cache: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
        self._dirty: set[tuple[str, str]] = set()

    def _path(self, symbol: str, interval: str) -> Path:
        symbol_dir = self.data_dir / "market" / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{interval}.parquet"

    def get(self, symbol: str, interval: str) -> pd.DataFrame | None:
        if symbol in self._cache and interval in self._cache[symbol]:
            return self._cache[symbol][interval]

        path = self._path(symbol, interval)
        if path.exists():
            df = pd.read_parquet(path)
            if not df.empty:
                self._cache[symbol][interval] = df
            return df
        return None

    def update(self, symbol: str, interval: str, df: pd.DataFrame):
        self._cache[symbol][interval] = df

    def append_candle(self, symbol: str, interval: str, candle: dict):
        new_row = pd.DataFrame([candle])
        new_row["close_time"] = pd.to_datetime(new_row["close_time"], unit="ms")
        new_row.set_index("close_time", inplace=True)

        existing = self._cache.get(symbol, {}).get(interval)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_row])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        else:
            combined = new_row

        self._cache[symbol][interval] = combined
        self._dirty.add((symbol, interval))

    def _read_disk(self, path: Path) -> pd.DataFrame | None:
        """Frame currently on disk (``None`` when absent or unreadable).

        A corrupt file must never take the periodic flush down, so a read error
        is logged and reported as "no existing history" — the same behaviour as
        a missing file.
        """
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Could not read existing cache {path}: {e}")
            return None

    def save(self, symbol: str, interval: str):
        """Write one ``symbol/interval`` to disk, **merging with what is there**.

        Durability contract of the parquet cache
        ---------------------------------------
        The file on disk can be *wider* than this instance's in-memory frame: the
        cache loads once (``get``) and afterwards only appends live candles, while
        an operator or ``scripts/download_history.py --merge`` may repair the file
        underneath a running process.  Writing the in-memory frame verbatim
        therefore destroyed that repair — the measured defect this fixes: the
        repaired 11 675-row ``data/market/BTCUSDT/1h.parquet`` was overwritten at
        13:01 with the 8 848-row spliced frame the live app had loaded and never
        re-read.

        Every write is now the union of the on-disk frame and the in-memory frame
        (:func:`merge_history`), so a longer existing history is never truncated by
        a writer that holds a shorter window.  The merged frame replaces the
        in-memory state too, so later writes are unions as well and the process
        converges on the widest known history.

        The union is also **interval-aware**: a bar the on-disk file holds under one
        timestamp convention and the in-memory frame under the other collapses to a
        single row instead of re-expanding the file (the 55 duplicated bars
        measured in the live ``1h`` cache), and a collided survivor is stamped in
        the file's dominant convention.  The interval is passed explicitly — this
        method is the one write path that always knows it, which is why the rule
        lives here rather than in each caller.

        Deliberately a property of the *store* rather than of each caller:
        ``_prefetch_history``, ``get_historical`` and the periodic ``flush_all``
        all reach the file through here, and any future writer inherits it.  The
        callers that still union in memory with the 2-argument
        ``merge_history(existing, incoming)`` (``core.market_data.provider``) keep
        that historical exact-timestamp result, which this write then dedupes.

        Why merge-on-write rather than a separate rolling-window file: the cache
        layout ``<data_dir>/market/<SYMBOL>/<interval>.parquet`` is a frozen
        contract with every reader (``core.backtest.data_feeder``, the GA
        walk-forward, the web routes, ``scripts/check_data_integrity.py``), so a
        second file would push the "union the two" rule into all of them.  Merging
        here keeps one authoritative file and fixes all three writers at once.
        """
        if symbol in self._cache and interval in self._cache[symbol]:
            path = self._path(symbol, interval)
            combined = merge_history(self._read_disk(path),
                                     self._cache[symbol][interval],
                                     interval)
            if combined is not None:
                combined.to_parquet(path)
                self._cache[symbol][interval] = combined
            self._dirty.discard((symbol, interval))

    def flush_all(self):
        """Write all dirty cache entries to disk. Called periodically."""
        for symbol, interval in list(self._dirty):
            try:
                self.save(symbol, interval)
            except Exception as e:
                logger.warning(f"Failed to flush cache {symbol}/{interval}: {e}")
