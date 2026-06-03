import asyncio
import time
from typing import Optional
from binance import AsyncClient, BinanceSocketManager
import pandas as pd

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.ohlcv_cache import OHLVCache


class MarketDataProvider:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.cache = OHLVCache(config.data_dir)
        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._watched_symbols: list[str] = []
        self._intervals: list[str] = []
        self._price_cache: dict[str, float] = {}
        self._price_history: dict[str, list[tuple[float, float]]] = {}

    @property
    def watched_symbols(self) -> list[str]:
        return list(self._watched_symbols)

    async def start(self, symbols: list[str], intervals: list[str]):
        self._watched_symbols = symbols
        self._intervals = intervals
        try:
            self.client = await AsyncClient.create(
                api_key=self.config.binance_api_key or None,
                api_secret=self.config.binance_api_secret or None,
                testnet=self.config.binance_testnet,
            )
        except Exception as e:
            from loguru import logger
            logger.warning(f"Binance API unreachable (server will run offline): {e}")
            self.client = None
        self._running = True
        # Pre-fetch historical klines so strategies can evaluate immediately
        try:
            await self._prefetch_history(symbols, intervals)
        except Exception as e:
            from loguru import logger
            logger.warning(f"History prefetch failed: {e}")
        if self.client is not None:
            self._tasks.append(asyncio.create_task(self._run_websocket()))
        self._tasks.append(asyncio.create_task(self._run_price_tracker()))
        self._tasks.append(asyncio.create_task(self._run_cache_flush()))

    async def _prefetch_history(self, symbols: list[str], intervals: list[str]):
        """Fetch historical candles for backtesting via REST.

        Skips intervals that already have sufficient data on disk (e.g. from
        the download_history script) to avoid overwriting larger datasets.
        """
        from loguru import logger

        # Minimum candles we consider "sufficient" per interval
        MIN_CANDLES = {"1m": 10000, "5m": 3000, "15m": 2000, "30m": 1000,
                       "1h": 500, "2h": 300, "4h": 200}
        LIMIT = 1000
        BATCHES = {"1m": 4, "5m": 2, "15m": 1, "30m": 1, "1h": 1,
                   "2h": 1, "4h": 1, "6h": 1, "8h": 1, "12h": 1, "1d": 1}

        for symbol in symbols:
            for interval in intervals:
                # Check existing data
                existing = self.cache.get(symbol, interval)
                min_candles = MIN_CANDLES.get(interval, 200)
                if existing is not None and len(existing) >= min_candles:
                    continue  # already has enough data

                batches = BATCHES.get(interval, 1)
                all_klines = []
                end_time = None

                for batch in range(batches):
                    try:
                        params = {"symbol": symbol, "interval": interval, "limit": LIMIT}
                        if end_time is not None:
                            params["endTime"] = end_time
                        klines = await self.client.get_klines(**params)
                        if not klines or len(klines) <= 1:
                            break
                        all_klines = klines + all_klines
                        end_time = klines[0][0] - 1
                        if batch < batches - 1:
                            await asyncio.sleep(0.2)
                    except Exception:
                        break

                if all_klines:
                    df = pd.DataFrame([{
                        "close_time": k[6],
                        "open": float(k[1]), "high": float(k[2]),
                        "low": float(k[3]), "close": float(k[4]),
                        "volume": float(k[5]),
                    } for k in all_klines])
                    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
                    df.set_index("close_time", inplace=True)
                    df = df[~df.index.duplicated(keep='last')]
                    df.sort_index(inplace=True)
                    self.cache.update(symbol, interval, df)
                    self.cache.save(symbol, interval)

        logger.info(f"Pre-fetched history for {len(symbols)} symbols x {len(intervals)} intervals")

    async def _run_websocket(self):
        self.bsm = BinanceSocketManager(self.client)
        streams = []
        for symbol in self._watched_symbols:
            sym_lower = symbol.lower()
            for interval in self._intervals:
                streams.append(f"{sym_lower}@kline_{interval}")

        msg_count = 0
        while self._running:
            conn_key = self.bsm.multiplex_socket(streams)
            from loguru import logger
            logger.info(f"WebSocket connecting: {len(streams)} streams")
            try:
                async with conn_key as stream:
                    logger.info(f"WebSocket connected, entering receive loop")
                    msg_count = 0
                    while self._running:
                        try:
                            msg = await asyncio.wait_for(stream.recv(), timeout=1.0)
                            msg_count += 1
                            if msg_count <= 3 or msg_count % 100 == 0:
                                logger.info(f"WS msg #{msg_count}: {str(msg)[:150]}")
                            await self._handle_ws_message(msg)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.warning(f"WebSocket recv error: {e}")
                            break
            except Exception as e:
                logger.warning(f"WebSocket connection error: {e}")
            logger.info(f"WebSocket disconnected after {msg_count} msgs, reconnecting in 5s")
            await asyncio.sleep(5)

    async def _handle_ws_message(self, msg: dict):
        if "stream" not in msg:
            return
        data = msg["data"]
        if data.get("e") != "kline":
            return
        kline = data["k"]
        symbol = kline["s"]
        interval = kline["i"]
        # Always cache the latest price from every kline update
        self._price_cache[symbol] = float(kline["c"])
        if not kline["x"]:
            return
        candle = {
            "close_time": kline["T"],
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
        }
        self.cache.append_candle(symbol, interval, candle)
        self._price_cache[symbol] = float(kline["c"])

        # Track kline arrival for diagnostics
        self._last_kline_time = getattr(self, '_last_kline_time', {})
        self._kline_count = getattr(self, '_kline_count', {})
        kk = f"{symbol}_{interval}"
        self._last_kline_time[kk] = time.time()
        self._kline_count[kk] = self._kline_count.get(kk, 0) + 1
        # Log first kline of each stream and then every 50th
        if self._kline_count[kk] in (1, 50, 100, 200):
            from loguru import logger
            logger.info(f"Kline #{self._kline_count[kk]}: {symbol} {interval} close={candle['close']:.4f}")

        await self.event_bus.publish(Event(EventType.MARKET_KLINE, {
            "symbol": symbol,
            "interval": interval,
            "candle": candle,
        }))

    async def _run_cache_flush(self):
        """Periodically flush in-memory candle cache to parquet files."""
        while self._running:
            await asyncio.sleep(300)  # flush every 5 minutes
            try:
                self.cache.flush_all()
            except Exception:
                pass

    async def _run_price_tracker(self):
        while self._running:
            for symbol in self._watched_symbols:
                if symbol in self._price_cache:
                    price = self._price_cache[symbol]
                    if symbol not in self._price_history:
                        self._price_history[symbol] = []
                    self._price_history[symbol].append((time.time(), price))
                    cutoff = time.time() - 3600
                    self._price_history[symbol] = [
                        (t, p) for t, p in self._price_history[symbol] if t > cutoff
                    ]
            await asyncio.sleep(1)

    async def get_historical(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame | None:
        cached = self.cache.get(symbol, interval)
        if cached is not None and len(cached) >= limit:
            return cached.tail(limit)

        if self.client is None:
            return cached

        try:
            klines = await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if not klines:
                return cached
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            df = df[["close_time", "open", "high", "low", "close", "volume"]].copy()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df.set_index("close_time", inplace=True)
            self.cache.update(symbol, interval, df)
            self.cache.save(symbol, interval)
            return df
        except Exception as e:
            from loguru import logger
            logger.warning(f"REST kline fetch failed for {symbol}/{interval}: {e}")
            return cached

    def get_current_price(self, symbol: str) -> float | None:
        return self._price_cache.get(symbol)

    def get_price_change_pct(self, symbol: str, minutes: int = 5) -> float | None:
        history = self._price_history.get(symbol, [])
        if len(history) < 2:
            return None
        cutoff = time.time() - minutes * 60
        old_prices = [p for t, p in history if t <= cutoff]
        if not old_prices:
            return None
        current = history[-1][1]
        old = old_prices[-1]
        return (current - old) / old * 100

    async def get_volume_ratio(self, symbol: str, interval: str = "1h") -> float | None:
        df = await self.get_historical(symbol, interval, limit=100)
        if df is None or len(df) < 20:
            return None
        recent_vol = df["volume"].iloc[-1]
        avg_vol = df["volume"].iloc[-20:-1].mean()
        if avg_vol == 0:
            return None
        return float(recent_vol / avg_vol)

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self.client:
            await self.client.close_connection()
