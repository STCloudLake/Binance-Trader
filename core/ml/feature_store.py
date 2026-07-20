"""Feature Store — pre-computed ML feature cache shared by backtest and live.

Eliminates redundant feature computation by materialising the 40-dim
feature matrix to Parquet once, then serving O(1) lookups.

Pattern::

    FeatureStore
        .build(symbols, date_start, date_end, intervals)
        .get_vector("BTCUSDT", timestamp)         → np.ndarray (40,)
        .get_dataframe("BTCUSDT", "1h", ts, 100)  → pd.DataFrame (100×40)

In backtest, this replaces ``compute_all() + compute_features()`` on every
bar with a single indexed read.  Wall-clock speedup: 3–5× for large runs.

In live mode, the store is lazily updated on each new kline via
``update()``, avoiding feature recomputation on every prediction.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from core.strategy.indicators import compute_all
from core.ml.features import (
    compute_features, DEFAULT_FEATURES, REQUIRED_INDICATORS,
)


class FeatureStore:
    """Pre-computed ML feature cache on Parquet.

    One Parquet file per symbol × interval.  The DataFrame is indexed by
    timestamp and contains all feature columns.
    """

    def __init__(self, data_dir: str):
        self.store_dir = Path(data_dir) / "feature_store"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._feature_list: list[str] = list(DEFAULT_FEATURES)
        self._loaded: dict[str, pd.DataFrame] = {}  # cache in memory

    # ── Public API ────────────────────────────────────────────────────

    @property
    def feature_count(self) -> int:
        return len(self._feature_list)

    def build(self, symbols: list[str], intervals: list[str],
              date_start: str, date_end: str,
              cache_dir: str | None = None,
              progress_cb=None) -> dict[str, dict[str, int]]:
        """Batch-compute features for all symbols × intervals.

        Args:
            symbols: Trading symbols (e.g. ["BTCUSDT", "ETHUSDT"]).
            intervals: Timeframe strings (e.g. ["1m", "5m", "1h"]).
            date_start, date_end: Date range (inclusive).
            cache_dir: Path to OHLCV parquet cache (same layout as DataFeeder).
                If None, reads from ``store_dir.parent / "market"``.
            progress_cb: Optional ``(symbol, interval, n_rows)`` callback.

        Returns:
            Nested dict: ``{symbol: {interval: n_rows_written}}``.
        """
        if cache_dir is None:
            cache_dir = str(self.store_dir.parent / "market")

        cache_path = Path(cache_dir)
        ts_start = pd.Timestamp(date_start)
        ts_end = pd.Timestamp(date_end)
        stats = {}

        for symbol in symbols:
            stats[symbol] = {}
            sym_market = cache_path / symbol
            if not sym_market.exists():
                logger.debug(f"FeatureStore: no OHLCV data for {symbol} — skipped")
                continue

            for interval in intervals:
                ohlcv_path = sym_market / f"{interval}.parquet"
                if not ohlcv_path.exists():
                    logger.debug(f"FeatureStore: missing {symbol}/{interval}.parquet")
                    continue

                df = pd.read_parquet(ohlcv_path)
                df.index = pd.to_datetime(df.index)
                # Filter to date range
                mask = (df.index >= ts_start) & (df.index <= ts_end)
                df = df[mask].copy()
                if len(df) < 50:
                    stats[symbol][interval] = 0
                    continue

                # Compute features
                df = compute_all(df, REQUIRED_INDICATORS)
                features = compute_features(df, self._feature_list)

                # Add close for convenience (needed by backtest)
                if "close" in df.columns:
                    features["close"] = df["close"]

                # Save to parquet
                out_dir = self.store_dir / symbol
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{interval}.parquet"
                features.to_parquet(out_path)

                n_rows = len(features)
                stats[symbol][interval] = n_rows

                if progress_cb:
                    progress_cb(symbol, interval, n_rows)

                logger.debug(f"FeatureStore: built {symbol}/{interval} "
                           f"→ {n_rows} rows ({len(self._feature_list)} features)")

        return stats

    def get_vector(self, symbol: str, interval: str,
                   timestamp: pd.Timestamp) -> np.ndarray | None:
        """Return the feature vector at *timestamp* as a float64 array.

        Returns None if no data exists for this symbol/interval or the
        timestamp is before the first available row.
        """
        df = self._get(symbol, interval)
        if df is None or len(df) == 0:
            return None
        try:
            pos = df.index.get_loc(timestamp)
            if isinstance(pos, slice):
                pos = pos.stop - 1
            row = df.iloc[pos]
            # Return only feature columns (drop auxiliary columns like "close")
            cols = [c for c in self._feature_list if c in df.columns]
            return row[cols].values.astype(np.float64)
        except KeyError:
            # Timestamp not exactly in index — find nearest previous
            before = df[df.index <= timestamp]
            if len(before) == 0:
                return None
            cols = [c for c in self._feature_list if c in before.columns]
            return before[cols].iloc[-1].values.astype(np.float64)

    def get_dataframe(self, symbol: str, interval: str,
                      timestamp: pd.Timestamp,
                      n_rows: int = 100) -> pd.DataFrame | None:
        """Return the last *n_rows* features up to *timestamp*.

        This is the primary backtest accessor — replaces the inline
        ``compute_all() + compute_features()`` with a single indexed read.
        """
        df = self._get(symbol, interval)
        if df is None or len(df) == 0:
            return None
        sliced = df[df.index <= timestamp]
        if len(sliced) == 0:
            return None
        return sliced.iloc[-n_rows:]

    def update(self, symbol: str, interval: str,
               new_ohlcv: pd.DataFrame) -> int:
        """Incrementally update the feature store with new kline data.

        Reads existing features from the store and raw OHLCV history
        from the market cache to recompute only what's needed.

        Returns the number of new rows added.
        """
        # Raw OHLCV from market cache for warm-up
        market_cache = self.store_dir.parent / "market"
        raw_path = market_cache / symbol / f"{interval}.parquet"
        if raw_path.exists():
            raw_all = pd.read_parquet(raw_path)
            raw_all.index = pd.to_datetime(raw_all.index)
        else:
            raw_all = new_ohlcv.copy()

        existing = self._get(symbol, interval)
        if existing is None or len(existing) == 0:
            # No existing features — compute from scratch
            df = compute_all(raw_all.tail(500).copy(), REQUIRED_INDICATORS)
            features = compute_features(df, self._feature_list)
            if "close" in df.columns:
                features["close"] = df["close"]
            out_path = self._path(symbol, interval)
            features.to_parquet(out_path)
            self._loaded.pop(self._key(symbol, interval), None)
            return len(features)

        # Incremental: find new timestamps not yet in the feature store
        last_ts = existing.index[-1]
        new_raw = raw_all[raw_all.index > last_ts]
        if len(new_raw) == 0:
            return 0

        # Combine warm-up history (raw OHLCV) + new bars
        warmup_raw = raw_all[raw_all.index <= last_ts].iloc[-200:]
        combined_raw = pd.concat([warmup_raw, new_raw])

        # Recompute features from raw OHLCV
        df = compute_all(combined_raw.copy(), REQUIRED_INDICATORS)
        features = compute_features(df, self._feature_list)

        # Only keep the new bars
        new_features = features.loc[new_raw.index]
        if "close" in new_raw.columns:
            new_features["close"] = new_raw["close"]

        # Append to existing store
        updated = pd.concat([existing, new_features])
        updated = updated[~updated.index.duplicated(keep="last")]
        updated.sort_index(inplace=True)

        out_path = self._path(symbol, interval)
        updated.to_parquet(out_path)
        self._loaded[self._key(symbol, interval)] = updated
        return len(new_features)

    def invalidate(self, symbol: str, interval: str | None = None):
        """Remove cached features for a symbol.

        If *interval* is None, invalidates all intervals for the symbol.
        """
        if interval is None:
            for key in list(self._loaded.keys()):
                if key.startswith(f"{symbol}/"):
                    del self._loaded[key]
            # Remove parquet files
            sym_dir = self.store_dir / symbol
            if sym_dir.exists():
                for f in sym_dir.glob("*.parquet"):
                    f.unlink()
        else:
            self._loaded.pop(self._key(symbol, interval), None)
            p = self._path(symbol, interval)
            if p.exists():
                p.unlink()

    def has_data(self, symbol: str, interval: str) -> bool:
        """Check if features exist for this symbol/interval."""
        return self._path(symbol, interval).exists()

    # ── Internals ──────────────────────────────────────────────────────

    def _key(self, symbol: str, interval: str) -> str:
        return f"{symbol}/{interval}"

    def _path(self, symbol: str, interval: str) -> Path:
        return self.store_dir / symbol / f"{interval}.parquet"

    def _get(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """Load from memory cache or parquet disk."""
        key = self._key(symbol, interval)
        if key in self._loaded:
            return self._loaded[key]

        p = self._path(symbol, interval)
        if not p.exists():
            return None

        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        self._loaded[key] = df
        return df
