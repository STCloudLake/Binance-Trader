"""Coin universe: exchangeInfo metadata, search/paging/sort, local-data flags.

Source of truth is ``GET {config.market_data_host}/api/v3/exchangeInfo`` — the
public mainnet mirror (``docs/overhaul/MARKET_PAGES_API.md`` §0/§1).  The full
payload is ~3700 symbols / ~3 MB and changes rarely, so it is:

1. cached **in process** for :data:`~core.market_data.ttl_cache.EXCHANGE_INFO_TTL`
   (6h), and
2. persisted to ``{data_dir}/symbols.json`` so a restart does not refetch ~3 MB
   through a 10s round-trip.

Search / paging / sort are pure-Python over the in-memory list; the 24h ticker
numbers come from the caller's cached snapshot (``/api/market/ticker24h``), never
from a second upstream call.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import aiosqlite
from loguru import logger

from core.market_data.data_client import MarketDataClient, MarketDataError
from core.market_data.provider import INTERVAL_SPEC
from core.market_data.ttl_cache import EXCHANGE_INFO_TTL

#: The five symbols the engine shipped with — used as the watchlist fallback and
#: by ``MarketDataProvider.start()`` when the database holds no watchlist yet.
DEFAULT_WATCHLIST = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

#: ``POST /api/market/watchlist`` refuses lists longer than this (contract §1).
WATCHLIST_MAX = 30

#: system_config key holding the persisted watchlist (JSON list).
WATCHLIST_KEY = "watchlist_symbols"

#: Parquet cache layout the backtest DataFeeder reads: ``data/market/{sym}/{itv}.parquet``.
MARKET_CACHE_SUBDIR = "market"

#: Intervals checked for ``has_cached_data`` — the registry's intervals ordered
#: shortest-first, so adding an interval to ``INTERVAL_SPEC`` cannot leave a
#: stale copy behind here.
CACHED_INTERVAL_ORDER = tuple(
    tf for tf, _spec in sorted(INTERVAL_SPEC.items(),
                               key=lambda item: item[1]["minutes"]))

#: How long a single ``/api/market/symbols`` page may spend filling in listing
#: dates before it returns what it has (the field stays ``null`` for the rest and
#: is filled on a later request from the persisted cache).
LISTING_DATE_BUDGET_S = 3.0
LISTING_DATE_CONCURRENCY = 8


@dataclass
class SymbolInfo:
    """Trading rules + metadata for one spot pair."""

    symbol: str
    base_asset: str = ""
    quote_asset: str = ""
    status: str = ""
    base_precision: Optional[int] = None
    quote_precision: Optional[int] = None
    tick_size: Optional[float] = None
    step_size: Optional[float] = None
    min_qty: Optional[float] = None
    min_notional: Optional[float] = None
    is_spot_trading_allowed: bool = True
    listing_date: Optional[str] = None  # YYYY-MM-DD, filled lazily from the 1d kline

    def to_dict(self) -> dict:
        return asdict(self)


def _f(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _filter_value(filters: list[dict], filter_type: str, key: str) -> Optional[float]:
    for f in filters or []:
        if f.get("filterType") == filter_type:
            return _f(f.get(key))
    return None


def parse_symbol(raw: dict) -> SymbolInfo:
    """Turn one ``exchangeInfo.symbols[]`` entry into :class:`SymbolInfo`."""
    filters = raw.get("filters") or []
    # MIN_NOTIONAL is the legacy name; spot now ships NOTIONAL.
    min_notional = _filter_value(filters, "NOTIONAL", "minNotional")
    if min_notional is None:
        min_notional = _filter_value(filters, "MIN_NOTIONAL", "minNotional")
    return SymbolInfo(
        symbol=raw.get("symbol", ""),
        base_asset=raw.get("baseAsset", ""),
        quote_asset=raw.get("quoteAsset", ""),
        status=raw.get("status", ""),
        base_precision=raw.get("baseAssetPrecision"),
        quote_precision=raw.get("quoteAssetPrecision", raw.get("quotePrecision")),
        tick_size=_filter_value(filters, "PRICE_FILTER", "tickSize"),
        step_size=_filter_value(filters, "LOT_SIZE", "stepSize"),
        min_qty=_filter_value(filters, "LOT_SIZE", "minQty"),
        min_notional=min_notional,
        is_spot_trading_allowed=bool(raw.get("isSpotTradingAllowed", True)),
        listing_date=raw.get("listing_date"),
    )


class Universe:
    """In-process + on-disk cache of the tradable coin universe."""

    def __init__(self, config, client: Optional[MarketDataClient] = None):
        self.config = config
        self.client = client or MarketDataClient(
            getattr(config, "market_data_host", "https://data-api.binance.vision"))
        self.data_dir = Path(getattr(config, "data_dir", "data"))
        self._symbols: list[SymbolInfo] = []
        self._by_symbol: dict[str, SymbolInfo] = {}
        self._loaded_at: float = 0.0
        self._lock = asyncio.Lock()
        self._listing_dates: dict[str, str] = {}
        self._listing_dates_loaded = False
        self._cache_interval_cache: dict[str, tuple[float, Optional[str]]] = {}

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    @property
    def symbols_path(self) -> Path:
        return self.data_dir / "symbols.json"

    @property
    def listing_dates_path(self) -> Path:
        return self.data_dir / "listing_dates.json"

    def _read_symbols_file(self) -> tuple[list[SymbolInfo], float]:
        path = self.symbols_path
        if not path.exists():
            return [], 0.0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("symbols") if isinstance(payload, dict) else payload
            saved = float(payload.get("saved_at", 0.0)) if isinstance(payload, dict) else 0.0
            return [SymbolInfo(**row) for row in (rows or [])], saved
        except Exception as e:  # old/corrupt file must not break the app
            logger.warning(f"Could not read {path}: {e}")
            return [], 0.0

    def _write_symbols_file(self, symbols: list[SymbolInfo]) -> None:
        try:
            self.symbols_path.parent.mkdir(parents=True, exist_ok=True)
            self.symbols_path.write_text(json.dumps(
                {"saved_at": time.time(), "host": self.client.host,
                 "symbols": [s.to_dict() for s in symbols]},
                ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not persist {self.symbols_path}: {e}")

    def _adopt(self, symbols: list[SymbolInfo], loaded_at: float) -> None:
        self._symbols = symbols
        self._by_symbol = {s.symbol: s for s in symbols}
        self._loaded_at = loaded_at

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    async def get_symbols(self, force: bool = False) -> list[SymbolInfo]:
        """Return every symbol from exchangeInfo, refreshing after the 6h TTL.

        Never raises: on failure a stale on-disk copy is returned, and only when
        there is nothing at all does :class:`MarketDataError` propagate.
        """
        fresh = (self._symbols and not force
                 and (time.time() - self._loaded_at) < EXCHANGE_INFO_TTL)
        if fresh:
            return self._symbols

        async with self._lock:
            fresh = (self._symbols and not force
                     and (time.time() - self._loaded_at) < EXCHANGE_INFO_TTL)
            if fresh:
                return self._symbols
            disk, saved = self._read_symbols_file()
            if disk and not force and (time.time() - saved) < EXCHANGE_INFO_TTL:
                self._adopt(disk, saved or time.time())
                return self._symbols
            try:
                raw = await self.client.exchange_symbols()
            except MarketDataError as e:
                if disk:  # offline: stale beats nothing
                    logger.warning(f"exchangeInfo unavailable ({e}); using cached symbols.json")
                    self._adopt(disk, time.time())
                    return self._symbols
                raise
            parsed = [parse_symbol(r) for r in raw if r.get("symbol")]
            if not parsed:
                raise MarketDataError("exchangeInfo returned no symbols")
            self._adopt(parsed, time.time())
            self._write_symbols_file(parsed)
            logger.info(f"Universe loaded: {len(parsed)} symbols from {self.client.host}")
            return self._symbols

    def preload_from_disk(self) -> int:
        """Cheap synchronous load of ``symbols.json`` (no network, no refresh)."""
        if self._symbols:
            return len(self._symbols)
        disk, saved = self._read_symbols_file()
        if disk:
            self._adopt(disk, saved or time.time())
        return len(self._symbols)

    def get_symbols_cached_count(self) -> int:
        """How many symbols are loaded right now (0 = universe unavailable)."""
        return len(self._symbols)

    def needs_refresh(self) -> bool:
        return not self._symbols or (time.time() - self._loaded_at) >= EXCHANGE_INFO_TTL

    async def refresh_if_needed(self) -> int:
        """Background warm-up used by ``register()``; swallows network errors."""
        if self.preload_from_disk() and not self.needs_refresh():
            return len(self._symbols)
        try:
            return len(await self.get_symbols())
        except MarketDataError as e:
            logger.warning(f"Universe warm-up skipped (data host unavailable): {e}")
            return len(self._symbols)

    # ------------------------------------------------------------------
    # metadata access
    # ------------------------------------------------------------------
    def get(self, symbol: str) -> Optional[SymbolInfo]:
        return self._by_symbol.get(str(symbol or "").upper())

    def has(self, symbol: str, status: str | None = "TRADING") -> bool:
        info = self.get(symbol)
        if info is None:
            return False
        return status is None or info.status == status

    def filter(self, q: Optional[str] = None, quote: str = "USDT",
               status: str = "TRADING") -> list[SymbolInfo]:
        """Case-insensitive search over both ``symbol`` and ``baseAsset``."""
        needle = (q or "").strip().upper()
        out: list[SymbolInfo] = []
        for info in self._symbols:
            if status and info.status != status:
                continue
            if quote and info.quote_asset != quote.upper():
                continue
            if needle and needle not in info.symbol and needle not in info.base_asset.upper():
                continue
            out.append(info)
        return out

    # ------------------------------------------------------------------
    # listing dates (lazy, best effort)
    # ------------------------------------------------------------------
    def _load_listing_dates(self) -> None:
        if self._listing_dates_loaded:
            return
        self._listing_dates_loaded = True
        path = self.listing_dates_path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._listing_dates = {
                    str(k).upper(): str(v) for k, v in payload.items() if v}
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")

    def listing_date(self, symbol: str) -> Optional[str]:
        self._load_listing_dates()
        return self._listing_dates.get(str(symbol or "").upper())

    def set_listing_date(self, symbol: str, value: Optional[str]) -> None:
        if value:
            self._listing_dates[str(symbol).upper()] = value

    def save_listing_dates(self) -> None:
        self._load_listing_dates()
        try:
            self.listing_dates_path.parent.mkdir(parents=True, exist_ok=True)
            self.listing_dates_path.write_text(
                json.dumps(self._listing_dates, ensure_ascii=False, indent=0),
                encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not persist {self.listing_dates_path}: {e}")

    async def fetch_listing_date(self, symbol: str) -> Optional[str]:
        """First daily candle = listing date.  One cheap upstream call.

        Binance returns the earliest candle at/after ``startTime``, so
        ``startTime=0&limit=1`` on the 1d interval *is* the listing day.
        """
        try:
            rows = await self.client.klines(symbol, "1d", limit=1, start_time=0)
        except MarketDataError as e:
            logger.debug(f"listing date lookup failed for {symbol}: {e}")
            return None
        if not rows:
            return None
        try:
            open_ms = int(rows[0][0])
        except (TypeError, ValueError, IndexError):
            return None
        return time.strftime("%Y-%m-%d", time.gmtime(open_ms / 1000))

    async def batch_fetch_listing_dates(self, symbols: Iterable[str],
                                        budget_s: float = LISTING_DATE_BUDGET_S,
                                        concurrency: int = LISTING_DATE_CONCURRENCY) -> int:
        """Fill in missing listing dates for ``symbols`` within a time budget."""
        self._load_listing_dates()
        missing = [s for s in dict.fromkeys(symbols)
                   if str(s).upper() not in self._listing_dates]
        if not missing:
            return 0
        sem = asyncio.Semaphore(max(1, concurrency))
        deadline = time.monotonic() + max(0.0, budget_s)
        filled = 0

        async def _one(sym: str):
            nonlocal filled
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            async with sem:
                if time.monotonic() >= deadline:
                    return
                value = await asyncio.wait_for(
                    self.fetch_listing_date(sym), timeout=max(1.0, remaining))
                if value:
                    self._listing_dates[sym.upper()] = value
                    filled += 1

        try:
            await asyncio.gather(*(_one(s) for s in missing), return_exceptions=True)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"listing date batch failed: {e}")
        if filled:
            self.save_listing_dates()
        return filled

    # ------------------------------------------------------------------
    # local parquet cache
    # ------------------------------------------------------------------
    def market_cache_dir(self, symbol: str) -> Path:
        return self.data_dir / MARKET_CACHE_SUBDIR / str(symbol).upper()

    def cached_intervals(self, symbol: str) -> list[str]:
        """Intervals that have a non-empty ``{interval}.parquet`` on disk."""
        directory = self.market_cache_dir(symbol)
        if not directory.is_dir():
            return []
        found: list[str] = []
        for path in directory.glob("*.parquet"):
            try:
                if path.stat().st_size > 0:
                    found.append(path.stem)
            except OSError:  # pragma: no cover
                continue
        order = {name: i for i, name in enumerate(CACHED_INTERVAL_ORDER)}
        return sorted(found, key=lambda n: (order.get(n, 99), n))

    def has_cached_data(self, symbol: str) -> bool:
        """True when any ``data/market/{symbol}/{interval}.parquet`` exists.

        ``stat`` only — never parses the file, because this runs once per row of
        a 200-row page.
        """
        directory = self.market_cache_dir(symbol)
        if not directory.is_dir():
            return False
        try:
            for path in directory.glob("*.parquet"):
                try:
                    if path.stat().st_size > 0:
                        return True
                except OSError:  # pragma: no cover
                    continue
        except OSError:  # pragma: no cover
            return False
        return False


# ======================================================================
# watchlist persistence (system_config table)
# ======================================================================
async def load_watchlist(db_path: str,
                         default: Optional[list[str]] = None) -> list[str]:
    """Read the persisted watchlist, falling back to ``DEFAULT_WATCHLIST``.

    A missing table / unreadable DB is not an error here: the app must start even
    on a brand-new or locked database, so the default list is returned instead.
    """
    fallback = list(default if default is not None else DEFAULT_WATCHLIST)
    if not db_path:
        return fallback
    try:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT value FROM system_config WHERE key=?", (WATCHLIST_KEY,))
            row = await cursor.fetchone()
    except Exception as e:
        logger.warning(f"Could not read watchlist from DB ({e}); using defaults")
        return fallback
    if not row or not row[0]:
        return fallback
    try:
        parsed = json.loads(row[0])
    except (TypeError, ValueError):
        logger.warning(f"watchlist_symbols holds invalid JSON; using defaults")
        return fallback
    if not isinstance(parsed, list):
        return fallback
    symbols = [str(s).strip().upper() for s in parsed if str(s).strip()]
    # De-duplicate, preserve order, cap.
    return list(dict.fromkeys(symbols))[:WATCHLIST_MAX] or fallback


async def save_watchlist(db_path: str, symbols: list[str]) -> None:
    """Persist the watchlist as a JSON list under ``watchlist_symbols``."""
    cleaned = list(dict.fromkeys(
        str(s).strip().upper() for s in symbols if str(s).strip()))[:WATCHLIST_MAX]
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT OR REPLACE INTO system_config (key, value, category) "
            "VALUES (?, ?, 'market')",
            (WATCHLIST_KEY, json.dumps(cleaned)))
        await db.commit()


def validate_watchlist(symbols: list[str], universe: Universe,
                       tolerate_unknown: bool = False) -> tuple[list[str], list[str]]:
    """Split ``symbols`` into (accepted, unknown).

    ``unknown`` covers both "not in exchangeInfo" and "not TRADING" — the
    contract requires both to be rejected.  ``tolerate_unknown`` is only used
    when the universe could not be loaded at all (offline data host): a valid
    request then still succeeds instead of 502-ing on a network hiccup.
    """
    known = bool(universe.get_symbols_cached_count())
    accepted, rejected = [], []
    for sym in symbols:
        if not known and tolerate_unknown:
            accepted.append(sym)
            continue
        info = universe.get(sym)
        if info is not None and info.status == "TRADING":
            accepted.append(sym)
        else:
            rejected.append(sym)
    return accepted, rejected
