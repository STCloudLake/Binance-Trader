"""Historical OHLCV data feeder for backtesting — reads from Parquet cache."""
import pandas as pd
from pathlib import Path


class DataFeeder:
    """Provides time-aligned historical OHLCV data from Parquet cache files.

    Expects data layout: {cache_dir}/{symbol}/{interval}.parquet
    (the same layout used by MarketDataProvider's OHLCV cache).

    Data is loaded with a *warm-up* prefix before ``date_start`` so indicators
    (RSI, MACD, SMA(100), ADX, Hurst…) are already valid on the first traded bar.
    Only timestamps at/after ``date_start`` are exposed on the trading timeline —
    the warm-up rows exist purely to make the indicators correct. Filtering the
    window at ``date_start`` (the previous behaviour) left the first ~26 bars of
    every backtest with NaN indicators and made the two engines disagree about
    when trading may start.
    """

    #: Bars of history kept before ``date_start`` (covers the longest indicator).
    WARMUP_BARS = 250

    def __init__(self, cache_dir: str, symbols: list[str], intervals: list[str],
                 date_start: str, date_end: str, warmup_bars: int | None = None):
        self.cache_dir = Path(cache_dir)
        self.symbols = symbols
        self.intervals = intervals
        self.date_start = pd.Timestamp(date_start)
        self.date_end = pd.Timestamp(date_end)
        self.warmup_bars = self.WARMUP_BARS if warmup_bars is None else int(warmup_bars)
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
                if not path.exists():
                    self._data[symbol][interval] = pd.DataFrame()
                    continue
                df = pd.read_parquet(path)
                # Normalize index: Parquet may store close_time as index name
                if df.index.name in ("close_time", "timestamp", "time"):
                    df.index.name = "time"
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                # Keep warm-up history before date_start; trim to date_end.
                first_tradable = int(df.index.searchsorted(self.date_start))
                start_pos = max(0, first_tradable - self.warmup_bars)
                end_pos = int(df.index.searchsorted(self.date_end, side="right"))
                self._data[symbol][interval] = df.iloc[start_pos:end_pos].copy()

        # Trading timeline: only real (post-warm-up) timestamps.
        all_times = set()
        for symbol in self.symbols:
            for interval in self.intervals:
                df = self._data[symbol].get(interval)
                if df is not None and len(df) > 0:
                    all_times.update(df.index[df.index >= self.date_start])
        self._timestamps = sorted(all_times)
        self._cursor = 0

        # If the requested range holds no data, keep the timeline empty: exposing the
        # warm-up rows instead would silently run a "backtest" entirely BEFORE
        # date_start (250 bars of the wrong period, reported as if it were the range).
        if not self._timestamps:
            self._timestamps = []

    def __len__(self):
        return len(self._timestamps)

    @property
    def first_timestamp(self) -> pd.Timestamp | None:
        """First timestamp on the trading timeline (None when empty)."""
        return self._timestamps[0] if self._timestamps else None


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
