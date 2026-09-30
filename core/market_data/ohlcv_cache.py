import pandas as pd
from pathlib import Path
from collections import defaultdict
from loguru import logger


def merge_history(existing: pd.DataFrame | None,
                  incoming: pd.DataFrame | None) -> pd.DataFrame | None:
    """Union two OHLCV frames over their ``close_time`` index — never truncate.

    ``existing`` is normally the frame already on disk and ``incoming`` the fresh
    window (a REST page, a live candle, the prefetch result).  The result keeps
    **every** timestamp either side holds ("a longer existing history is never
    truncated"), so a repair written by ``scripts/download_history.py --merge``
    cannot be undone by a writer that only knows a shorter window.

    Duplicate timestamps keep the *incoming* row (``keep="last"``), the same
    "newest wins" convention as :meth:`OHLVCache.append_candle`; the index is
    sorted so the output is a deterministic function of the two inputs.

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
    merged = pd.concat([left, right])
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.sort_index(inplace=True)
    return merged


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
        on ``close_time`` (:func:`merge_history`), so a longer existing history is
        never truncated by a writer that holds a shorter window.  The merged frame
        replaces the in-memory state too, so later writes are unions as well and
        the process converges on the widest known history.

        This is deliberately a property of the *store* rather than of each caller:
        ``_prefetch_history``, ``get_historical`` and the periodic ``flush_all``
        all reach the file through here, and any future writer inherits it.

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
                                     self._cache[symbol][interval])
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
