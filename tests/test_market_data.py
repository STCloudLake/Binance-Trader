import pytest
import tempfile
import os
from pathlib import Path


def test_ohlcv_cache_append_and_get():
    from core.market_data.ohlcv_cache import OHLVCache
    data_dir = tempfile.mkdtemp()
    cache = OHLVCache(data_dir)

    cache.append_candle("BTCUSDT", "1h", {
        "close_time": 1700000000000,
        "open": 50000.0, "high": 51000.0, "low": 49000.0,
        "close": 50500.0, "volume": 100.0,
    })

    df = cache.get("BTCUSDT", "1h")
    assert df is not None
    assert len(df) == 1
    assert float(df.iloc[0]["close"]) == 50500.0

    # Save to disk and verify file exists
    cache.save("BTCUSDT", "1h")
    path = Path(data_dir) / "market" / "BTCUSDT" / "1h.parquet"
    assert path.exists()


def test_ohlcv_cache_persistence():
    from core.market_data.ohlcv_cache import OHLVCache
    data_dir = tempfile.mkdtemp()
    cache1 = OHLVCache(data_dir)
    cache1.append_candle("ETHUSDT", "4h", {
        "close_time": 1700000000000,
        "open": 3000.0, "high": 3100.0, "low": 2900.0,
        "close": 3050.0, "volume": 50.0,
    })
    cache1.save("ETHUSDT", "4h")  # persist to disk

    cache2 = OHLVCache(data_dir)
    df = cache2.get("ETHUSDT", "4h")
    assert df is not None
    assert len(df) == 1


def test_ohlcv_cache_multiple_candles():
    from core.market_data.ohlcv_cache import OHLVCache
    import pandas as pd
    import numpy as np
    data_dir = tempfile.mkdtemp()
    cache = OHLVCache(data_dir)

    cache.append_candle("BTCUSDT", "1h", {
        "close_time": 1700000000000,
        "open": 50000.0, "high": 51000.0, "low": 49000.0,
        "close": 50500.0, "volume": 100.0,
    })
    cache.append_candle("BTCUSDT", "1h", {
        "close_time": 1700003600000,
        "open": 50500.0, "high": 51500.0, "low": 50000.0,
        "close": 51000.0, "volume": 120.0,
    })

    df = cache.get("BTCUSDT", "1h")
    assert df is not None
    assert len(df) == 2
