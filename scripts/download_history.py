"""Download historical kline data from Binance public API.

Fetches ~30 days of 1m data for all symbols, then resamples to 5m/15m/1h/4h.
No API key required — klines are public.
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import aiohttp
import json
import time

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
DATA_DIR = Path("data/market")
LIMIT = 1000  # Binance max per request
DAYS = 365

# OHLCV column names from Binance API
COLUMNS = ["open_time", "open", "high", "low", "close", "volume",
           "close_time", "quote_volume", "trades", "taker_buy_base",
           "taker_buy_quote", "ignore"]

# Resample rules for OHLCV
RESAMPLE_RULES = {
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum",
}


async def download_symbol_1m(session: aiohttp.ClientSession, symbol: str,
                              end_ms: int) -> pd.DataFrame:
    """Download ~30 days of 1m klines for one symbol using backwards pagination.

    Binance returns up to 1000 candles ending at endTime. We paginate backwards
    by setting endTime = oldest_candle_time - 1 after each batch.
    """
    target_start = end_ms - DAYS * 24 * 3600 * 1000
    all_rows = []
    batch_end = end_ms

    print(f"  {symbol}: downloading 1m klines...", end=" ", flush=True)
    t0 = time.time()

    while True:
        try:
            # Only use endTime to get the latest 1000 candles
            params = {"symbol": symbol, "interval": "1m", "limit": LIMIT}
            if batch_end < end_ms:
                params["endTime"] = batch_end
            klines = await fetch_klines_with_params(session, params)
        except Exception as e:
            print(f"\n    Error: {e}")
            break

        if not klines or len(klines) <= 1:
            break

        # Prepend to accumulate from oldest to newest
        all_rows = klines + all_rows

        # The earliest candle in this batch
        earliest_ms = klines[0][0]
        print(f"{len(klines)}.", end="", flush=True)

        # Stop if we've reached the target start date
        if earliest_ms <= target_start:
            break

        # Next batch ends just before the earliest candle in this batch
        batch_end = earliest_ms - 1
        await asyncio.sleep(0.05)  # rate limit

    if not all_rows:
        print(" no data")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df["close_time"] = pd.to_datetime(df["close_time"].astype(np.int64), unit="ms")
    df = df[["close_time", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df.set_index("close_time", inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()

    elapsed = time.time() - t0
    print(f" {len(df)} total rows in {elapsed:.1f}s ({df.index.min()} ~ {df.index.max()})")
    return df


async def fetch_klines_with_params(session: aiohttp.ClientSession,
                                    params: dict) -> list:
    """Fetch klines with given params."""
    url = "https://api.binance.com/api/v3/klines"
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise Exception(f"HTTP {resp.status}: {text[:200]}")
        return await resp.json()


def resample_ohlcv(df_1m: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample 1m OHLCV data to a higher timeframe."""
    rule = interval.replace("m", "min").replace("h", "h").replace("d", "D")
    resampled = df_1m.resample(rule).agg(RESAMPLE_RULES)
    resampled.dropna(inplace=True)
    return resampled


async def main():
    print(f"Downloading {DAYS} days of historical data for {len(SYMBOLS)} symbols...")
    print(f"Target: 1m klines → resample to 5m/15m/1h/4h\n")

    end_ms = int(datetime.now().timestamp() * 1000)

    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        for symbol in SYMBOLS:
            # Download 1m data
            df_1m = await download_symbol_1m(session, symbol, end_ms)
            if df_1m.empty:
                continue

            # Save 1m
            sym_dir = DATA_DIR / symbol
            sym_dir.mkdir(parents=True, exist_ok=True)
            df_1m.to_parquet(sym_dir / "1m.parquet")

            # Resample and save higher timeframes
            for tf in ["5m", "15m", "1h", "4h"]:
                df_tf = resample_ohlcv(df_1m, tf)
                df_tf.to_parquet(sym_dir / f"{tf}.parquet")
                print(f"    → {tf}: {len(df_tf)} rows")

            print()

    print("Done! All data saved to data/market/")


if __name__ == "__main__":
    asyncio.run(main())
