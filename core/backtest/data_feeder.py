"""Historical OHLCV data feeder for backtesting — reads from Parquet cache."""
import pandas as pd
from pathlib import Path


class DataFeeder:
    """Provides time-aligned historical OHLCV data from Parquet cache files.

    Expects data layout: {cache_dir}/{symbol}/{interval}.parquet
    (the same layout used by MarketDataProvider's OHLCV cache).
    """

    def __init__(self, cache_dir: str, symbols: list[str], intervals: list[str],
                 date_start: str, date_end: str):
        self.cache_dir = Path(cache_dir)
        self.symbols = symbols
        self.intervals = intervals
        self.date_start = pd.Timestamp(date_start)
        self.date_end = pd.Timestamp(date_end)
        self._data: dict[str, dict[str, pd.DataFrame]] = {}
        self._timestamps: list[pd.Timestamp] = []
        self._cursor = 0

    def load(self, _depth: int = 0):
        """Load all OHLCV data from Parquet cache, filter to date range, build unified timeline."""
        if _depth > 1:
            return  # safety guard against infinite recursion
        for symbol in self.symbols:
            self._data[symbol] = {}
            sym_dir = self.cache_dir / symbol
            for interval in self.intervals:
                path = sym_dir / f"{interval}.parquet"
                if path.exists():
                    df = pd.read_parquet(path)
                    # Normalize index: Parquet may store close_time as index name
                    if df.index.name in ("close_time", "timestamp", "time"):
                        df.index.name = "time"
                    df.index = pd.to_datetime(df.index)
                    # Filter to requested date range
                    mask = (df.index >= self.date_start) & (df.index <= self.date_end)
                    self._data[symbol][interval] = df[mask].copy()
                else:
                    self._data[symbol][interval] = pd.DataFrame()

        # Build unified timeline from the finest-granularity interval available
        all_times = set()
        for symbol in self.symbols:
            for interval in self.intervals:
                df = self._data[symbol].get(interval)
                if df is not None and len(df) > 0:
                    all_times.update(df.index)
        self._timestamps = sorted(all_times)
        self._cursor = 0

        # If we found no data, try auto-adjusting the date range to match
        # what's actually in the cache.
        if not self._timestamps and self._data:
            earliest = None
            latest = None
            for sym_data in self._data.values():
                for df in sym_data.values():
                    if len(df) > 0:
                        t0 = pd.Timestamp(df.index.min())
                        t1 = pd.Timestamp(df.index.max())
                        if earliest is None or t0 < earliest:
                            earliest = t0
                        if latest is None or t1 > latest:
                            latest = t1
            if earliest is not None:
                self.date_start = earliest - pd.Timedelta(hours=1)
                self.date_end = latest + pd.Timedelta(hours=1)
                # Reload with corrected range (guarded against recursion)
                return self.load(_depth + 1)

    def __len__(self):
        return len(self._timestamps)

    def __iter__(self):
        self._cursor = 0
        return self

    def __next__(self):
        if self._cursor >= len(self._timestamps):
            raise StopIteration
        ts = self._timestamps[self._cursor]
        self._cursor += 1
        return self.get_slice(ts)

    def get_slice(self, ts: pd.Timestamp) -> dict:
        """Return all available data across symbols/intervals at a given timestamp."""
        result = {"timestamp": ts, "symbols": {}}
        for symbol in self.symbols:
            result["symbols"][symbol] = {}
            for interval in self.intervals:
                df = self._data[symbol].get(interval)
                if df is not None and len(df) > 0:
                    row = df[df.index <= ts]
                    if len(row) > 0:
                        result["symbols"][symbol][interval] = row.iloc[-1]
        return result

    def get_all_data_for_symbol(self, symbol: str, interval: str) -> pd.DataFrame:
        """Get the full filtered DataFrame for a symbol/interval pair."""
        return self._data.get(symbol, {}).get(interval, pd.DataFrame())
