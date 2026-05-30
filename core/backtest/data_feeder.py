"""Historical OHLCV data feeder for backtesting — reads from Parquet cache."""
import pandas as pd
from pathlib import Path


class DataFeeder:
    """Provides time-aligned historical OHLCV data from Parquet cache files."""

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

    def load(self):
        """Load all OHLCV data from Parquet cache, filter to date range, build unified timeline."""
        for symbol in self.symbols:
            self._data[symbol] = {}
            for interval in self.intervals:
                path = self.cache_dir / f"{symbol}_{interval}.parquet"
                if path.exists():
                    df = pd.read_parquet(path)
                    df.index = pd.to_datetime(df.index)
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
