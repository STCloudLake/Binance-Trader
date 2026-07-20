import pandas as pd
from pathlib import Path
from collections import defaultdict
from loguru import logger


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

    def save(self, symbol: str, interval: str):
        if symbol in self._cache and interval in self._cache[symbol]:
            path = self._path(symbol, interval)
            self._cache[symbol][interval].to_parquet(path)
            self._dirty.discard((symbol, interval))

    def flush_all(self):
        """Write all dirty cache entries to disk. Called periodically."""
        for symbol, interval in list(self._dirty):
            try:
                self.save(symbol, interval)
            except Exception as e:
                logger.warning(f"Failed to flush cache {symbol}/{interval}: {e}")
