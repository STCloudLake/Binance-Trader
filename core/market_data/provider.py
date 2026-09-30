import asyncio
import time
from typing import Optional
from binance import AsyncClient, BinanceSocketManager
import pandas as pd

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.data_client import MarketDataClient, MarketDataError
from core.market_data.ohlcv_cache import OHLVCache, merge_history

# ======================================================================
# Interval registry — the single source of truth for kline intervals
# ======================================================================
#: ``interval → spec``.  Every interval handling decision in the system reads
#: this table, so a new interval only has to be added **here**:
#:
#:   minutes      bar length in minutes (ordering / primary-timeframe choice)
#:   min_candles  candles on disk that count as "enough" (prefetch skip test)
#:   batches      upstream 1000-candle pages fetched per symbol/interval
#:   poll_secs    staleness threshold for the REST polling fallback (seconds)
#:   ml_enabled   the ML predictor consumes this interval's MARKET_KLINE events
#:
#: Consumers: :meth:`MarketDataProvider.start` / ``_prefetch_history`` (this
#: module), ``core.ml.predictor`` (the ML gate), ``core.ga.genome`` (timeframe
#: genes), ``core.backtest.signal_matrix`` (timeframe ordering) and
#: ``app.main`` (poll cadence + ML training interval).
INTERVAL_SPEC: dict[str, dict] = {
    "1m":  {"minutes": 1,    "min_candles": 10000, "batches": 4, "poll_secs": 120,
            "ml_enabled": False},
    "3m":  {"minutes": 3,    "min_candles": 200,   "batches": 1, "poll_secs": 360,
            "ml_enabled": False},
    "5m":  {"minutes": 5,    "min_candles": 3000,  "batches": 2, "poll_secs": 300,
            "ml_enabled": False},
    "15m": {"minutes": 15,   "min_candles": 2000,  "batches": 1, "poll_secs": 900,
            "ml_enabled": False},
    "30m": {"minutes": 30,   "min_candles": 1000,  "batches": 1, "poll_secs": 1800,
            "ml_enabled": False},
    "1h":  {"minutes": 60,   "min_candles": 500,   "batches": 1, "poll_secs": 3600,
            "ml_enabled": True},
    "2h":  {"minutes": 120,  "min_candles": 300,   "batches": 1, "poll_secs": 7200,
            "ml_enabled": False},
    "4h":  {"minutes": 240,  "min_candles": 200,   "batches": 1, "poll_secs": 7200,
            "ml_enabled": True},
    "6h":  {"minutes": 360,  "min_candles": 200,   "batches": 1, "poll_secs": 7200,
            "ml_enabled": False},
    "8h":  {"minutes": 480,  "min_candles": 200,   "batches": 1, "poll_secs": 7200,
            "ml_enabled": False},
    "12h": {"minutes": 720,  "min_candles": 200,   "batches": 1, "poll_secs": 7200,
            "ml_enabled": False},
    "1d":  {"minutes": 1440, "min_candles": 200,   "batches": 1, "poll_secs": 7200,
            "ml_enabled": False},
    "3d":  {"minutes": 4320, "min_candles": 200,   "batches": 1, "poll_secs": 7200,
            "ml_enabled": False},
    "1w":  {"minutes": 10080, "min_candles": 200,  "batches": 1, "poll_secs": 7200,
            "ml_enabled": False},
}

#: Intervals the provider streams and pre-fetches when the caller names none,
#: and the timeframe universe the GA evolves over (all of them have data).
DEFAULT_INTERVALS = ["1m", "5m", "15m", "1h", "4h"]

#: Interval the ML models are trained on (the most reliable one for ML).
DEFAULT_ML_INTERVAL = "1h"

#: Interval used whenever a caller/strategy names none.
DEFAULT_TIMEFRAME = "1h"


def _kline_float(kline, key) -> float:
    """``float(kline[key])`` when the payload has it, else ``NaN``.

    The P6-B kline fields are read through here — positionally from a REST kline
    array (``quote_volume`` = 7, ``trade_count`` = 8) or by key from a WebSocket
    kline object (``"q"`` / ``"n"``) — so a short/odd payload produces the
    documented NaN instead of an ``IndexError``/``KeyError`` that would take the
    whole prefetch or the stream handler down.
    """
    try:
        value = kline[key]
    except (IndexError, KeyError, TypeError):
        return float("nan")
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

#: Fallbacks for an interval that is not (or only partly) in INTERVAL_SPEC.
_INTERVAL_DEFAULTS: dict = {"minutes": 60, "min_candles": 200, "batches": 1,
                            "poll_secs": 300, "ml_enabled": False}


def interval_spec(interval: str) -> dict:
    """Registry entry for ``interval``, with documented defaults filled in.

    Unknown/``None`` intervals get :data:`_INTERVAL_DEFAULTS` (the historical
    hard-coded fallbacks), so callers never need their own ``.get(x, default)``.
    """
    spec = dict(_INTERVAL_DEFAULTS)
    spec.update(INTERVAL_SPEC.get(str(interval or ""), {}) or {})
    return spec


def interval_minutes(interval: str, default: int = 60) -> int:
    """Bar length in minutes (``default`` when the interval is blank)."""
    if not interval:
        return default
    return int(interval_spec(interval)["minutes"])


def poll_seconds(interval: str, default: int = 300) -> int:
    """REST-poll staleness threshold for ``interval`` (seconds)."""
    if not interval:
        return default
    return int(interval_spec(interval)["poll_secs"])


def ml_intervals() -> list[str]:
    """Intervals the ML predictor reacts to, in registry order."""
    return [tf for tf, spec in INTERVAL_SPEC.items() if spec.get("ml_enabled")]


class MarketDataProvider:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.cache = OHLVCache(config.data_dir)
        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None
        self._data_client: Optional[MarketDataClient] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._watched_symbols: list[str] = []
        self._intervals: list[str] = []
        self._price_cache: dict[str, float] = {}
        self._price_timestamps: dict[str, float] = {}  # symbol → last update time
        self._price_history: dict[str, list[tuple[float, float]]] = {}

    @property
    def data_client(self) -> MarketDataClient:
        """Public market-data client (``config.market_data_host``).

        Every **kline read** goes here, never through the testnet trading client:
        testnet lists a handful of pairs and its history is synthetic, which is
        what left strategies and charts without data.  (contract §0)
        """
        if self._data_client is None:
            self._data_client = MarketDataClient(
                getattr(self.config, "market_data_host", "https://data-api.binance.vision"))
        return self._data_client

    @property
    def watched_symbols(self) -> list[str]:
        return list(self._watched_symbols)

    async def start(self, symbols: list[str] | None = None, intervals: list[str] | None = None):
        """Start streaming + prefetching for ``symbols``.

        ``symbols=None`` (the production path from ``app/main.py``) means "watch
        the persisted watchlist": the same `system_config.watchlist_symbols`
        list that ``/api/market/watchlist`` writes, capped at
        :data:`WATCHLIST_MAX`, with :data:`DEFAULT_WATCHLIST` as the fallback when
        nothing has been persisted yet.  ``docs/overhaul/MARKET_PAGES_API.md`` §1.
        """
        from core.market_data.universe import (
            DEFAULT_WATCHLIST, WATCHLIST_MAX, load_watchlist)

        if symbols is None:
            try:
                symbols = await load_watchlist(self.config.db_path, DEFAULT_WATCHLIST)
            except Exception as e:
                from loguru import logger
                logger.warning(f"Could not load watchlist ({e}); using defaults")
                symbols = list(DEFAULT_WATCHLIST)
            from loguru import logger
            logger.info(
                f"Watching persisted watchlist ({len(symbols)}/{WATCHLIST_MAX}): "
                f"{', '.join(symbols)}")
        symbols = list(symbols)[:WATCHLIST_MAX]
        intervals = list(intervals or self._intervals or DEFAULT_INTERVALS)
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

        When an interval *is* fetched it is **merged** into the on-disk history
        instead of replacing it: the fetched window is only ``batches × 1000``
        candles, so a file that exists but sits below the ``min_candles`` skip
        threshold (or one repaired while this process was running) must not lose
        the bars it already had.  ``OHLVCache.save`` re-reads the file and unions
        the timestamps again, so the guarantee also covers a repair that lands
        between this read and the write.

        The in-memory union passes ``interval`` too (re-audit finding 3), so the
        frame this process holds is the same bar-aware one the write path stores:
        with the historical 2-argument call the merged frame kept twin-convention
        rows in memory, which meant the next flush of that key rewrote the file
        every time before the write path collapsed them.
        """
        from loguru import logger

        LIMIT = 1000

        for symbol in symbols:
            for interval in intervals:
                # Per-interval specs come from the single registry above.
                spec = interval_spec(interval)
                # Check existing data
                existing = self.cache.get(symbol, interval)
                min_candles = spec["min_candles"]
                if existing is not None and len(existing) >= min_candles:
                    continue  # already has enough data

                batches = spec["batches"]
                all_klines = []
                end_time = None

                for batch in range(batches):
                    try:
                        klines = await self.data_client.klines(
                            symbol, interval, limit=LIMIT, end_time=end_time)
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
                        # P6-B: the kline's quote-asset notional (field 7) and
                        # trade count (field 8) are persisted with the bar, so a
                        # reader of the cache gets the source's USDT volume
                        # instead of the `volume × close` proxy.  A short/odd
                        # payload leaves them NaN (the documented "absent").
                        "quote_volume": _kline_float(k, 7),
                        "trade_count": _kline_float(k, 8),
                    } for k in all_klines])
                    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
                    df.set_index("close_time", inplace=True)
                    df = df[~df.index.duplicated(keep='last')]
                    df.sort_index(inplace=True)
                    # Union with the history already known (disk + memory): the
                    # fetched window is a *widening*, never a replacement.  The
                    # interval is passed so the **in-memory** merge is the same
                    # bar-aware one `OHLVCache.save` writes (re-audit finding 3):
                    # an exact-timestamp union leaves the twin-convention rows in
                    # memory, so every later flush of this key rewrites the file
                    # before the write path collapses them.
                    self.cache.update(symbol, interval,
                                      merge_history(existing, df, interval))
                    self.cache.save(symbol, interval)

        logger.info(f"Pre-fetched history for {len(symbols)} symbols x {len(intervals)} intervals")

    async def _run_websocket(self):
        self.bsm = BinanceSocketManager(self.client)
        # python-binance defaults to `wss://stream.binance.com:9443/`, which is
        # unreachable from this host; point its socket factory at the configured
        # market-data stream host (`wss://data-stream.binance.vision`) instead.
        stream_url = getattr(self.config, "binance_stream_url", None)
        if not stream_url:
            host = getattr(self.config, "market_stream_host", "wss://data-stream.binance.vision")
            stream_url = str(host).rstrip("/") + "/"
        self.bsm._get_stream_url = lambda explicit=None: explicit or stream_url
        streams = []
        for symbol in self._watched_symbols:
            sym_lower = symbol.lower()
            for interval in self._intervals:
                streams.append(f"{sym_lower}@kline_{interval}")

        msg_count = 0
        backoff = 5
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
                # Check for auth failures — don't retry if credentials are wrong
                if "authentication" in str(e).lower() or "api key" in str(e).lower():
                    logger.error("WebSocket auth failed — not reconnecting")
                    self._running = False
                    break
            backoff = min(backoff * 2, 60)  # exponential backoff, max 60s
            logger.info(f"WebSocket disconnected after {msg_count} msgs, reconnecting in {backoff}s")
            await asyncio.sleep(backoff)

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
        self._price_timestamps[symbol] = time.time()
        if not kline["x"]:
            return
        candle = {
            "close_time": kline["T"],
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            # P6-B: `q` = quote-asset volume, `n` = number of trades, both NaN
            # when the stream omits them (the documented "absent" value, not a
            # column that exists for some bars and not others).
            "quote_volume": _kline_float(kline, "q"),
            "trade_count": _kline_float(kline, "n"),
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

        try:
            # Public market data host, not the (testnet) trading client.
            klines = await self.data_client.klines(symbol, interval, limit=limit)
            if not klines:
                return cached
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            # P6-B: the cache schema is `open..volume + quote_volume +
            # trade_count`; the source column is `trades`, the cache column is
            # `trade_count` (same number, the cache's documented name).
            df = df[["close_time", "open", "high", "low", "close", "volume",
                     "quote_volume", "trades"]].rename(
                         columns={"trades": "trade_count"}).copy()
            for col in ["open", "high", "low", "close", "volume",
                        "quote_volume", "trade_count"]:
                df[col] = df[col].astype(float)
            df.set_index("close_time", inplace=True)
            self.cache.update(symbol, interval, df)
            self.cache.save(symbol, interval)
            return df
        except Exception as e:
            from loguru import logger
            logger.warning(f"REST kline fetch failed for {symbol}/{interval}: {e}")
            return cached

    def get_current_price(self, symbol: str, max_age_sec: float = 300.0) -> float | None:
        """Return cached price if fresh (≤ max_age_sec old), else None.

        Default TTL is 5 minutes — WebSocket updates every second, so a stale
        price means the connection is truly down. Manual trades and PositionGuard
        can pass a shorter TTL if fresher data is required.
        """
        ts = self._price_timestamps.get(symbol, 0)
        if time.time() - ts > max_age_sec:
            return None  # Stale price — refuse to use
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
        if self._data_client is not None:
            await self._data_client.close()
            self._data_client = None
