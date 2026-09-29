"""Market data, account, order and history APIs for the Binance-style trade page.

All endpoints here are additive to the existing route table (the frozen contracts
are ``docs/overhaul/TRADE_PAGE_API.md`` and ``docs/overhaul/MARKET_PAGES_API.md``);
the sibling frontend consumes exactly these shapes.

Three rules drive the implementation:

* **market data comes from the public mainnet mirror.**  Every *read* of public
  market data (klines, ticker, depth, trades, exchangeInfo) goes through
  :func:`_data_client`, bound to ``config.market_data_host``
  (``https://data-api.binance.vision``).  ``api.binance.com`` is unreachable from
  this host, and testnet only lists a handful of pairs — neither can answer
  "show me all 3716 symbols".  The **trading** client
  (:func:`_configured_client`) stays on ``binance.testnet``.
* **order/account calls stay on testnet.**  Only ``/api/order`` and the REST price
  fallback touch the authenticated client.
* **reads are cached.**  A polling page hits ``/api/market/ticker`` every couple
  of seconds per symbol; ``/api/market/ticker24h`` costs ~10s upstream for the
  full market, so it is cached for 60s (contract §1) and the symbol list reuses
  that same snapshot.
"""
import asyncio
import time

import aiosqlite
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse

from app.config import (
    BNB_DISCOUNT_PCT,
    FEE_TIER_NOTE,
    FEE_TIER_TABLE,
    SIM_FEE_TIER_KEY,
    SIM_USE_BNB_DISCOUNT_KEY,
    fee_tier_entry,
    load_sim_cost_settings,
    sim_cost_quote,
    sim_fee_pct,
    sim_spread_pct,
)
from app.event_bus import Event, EventType
from db.database import load_sim_balance

from core.market_data import metrics
from core.market_data.data_client import MarketDataClient, MarketDataError
#: Signal attribution default: sourced from the interval registry so the web
#: layer does not hard-code a timeframe the registry may change.
from core.market_data.provider import DEFAULT_TIMEFRAME
#: Shared symbol-shape validator: both market-data key spaces (this module's coin
#: cache and the audit screener) must reject a malformed symbol *before* it
#: reaches the network or the cache, and they must agree on what "malformed" is.
from core.market_data.screener import valid_symbol
from core.market_data.ttl_cache import (
    COIN_TTL,
    DEPTH_TTL,
    SYMBOLS_TTL,
    TICKER24H_TTL,
    TRADES_TTL,
    TTLCache,
)
from core.market_data.universe import (
    DEFAULT_WATCHLIST,
    WATCHLIST_MAX,
    Universe,
    load_watchlist,
    save_watchlist,
    validate_watchlist,
)

from core.executor.pending_orders import (
    STATUS_OPEN,
    LimitOrderMatcher,
    PendingOrderStore,
    start_matcher_task,
)

from web.deps import _require_trader
#: Shared per-session/IP rate limiter, defined next to the audit routes (the
#: first fan-out consumer).  The market reads below hit the public exchange
#: too, so `RATE_LIMITS["market"]` caps how fast a loop can crawl through them.
from web.routes.audit import check_rate_limit

#: In-memory TTL for /api/market/ticker|depth|trades, per symbol (contract §1).
MARKET_CACHE_TTL = 2.0

#: How many symbols the data-overview leader boards carry.
LEADERBOARD_SIZE = 10

#: Expensive per-symbol overlays (30d volatility, ±1% depth) are sampled from
#: this many top-quote-volume symbols — the contract explicitly allows sampling
#: ("采样前 100 名成交额币种即可") because the full market would be ~500 API calls.
OVERVIEW_VOLATILITY_SAMPLE = 100
OVERVIEW_SPREAD_SAMPLE = 20

#: Concurrency caps for the sampled fan-out.
OVERVIEW_CONCURRENCY = 12

#: Seconds /api/data/overview may spend on the per-symbol overlays.
OVERVIEW_OVERLAY_BUDGET_S = 15.0


def _round(value, digits: int):
    """Round when numeric, pass ``None`` through (contract: missing → null)."""
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def register(app: FastAPI, ctx) -> None:
    config = ctx.config
    event_bus = ctx.event_bus

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------
    def _pending_store() -> PendingOrderStore:
        store = getattr(app.state, "pending_order_store", None)
        if store is None:
            store = PendingOrderStore(config.db_path)
            app.state.pending_order_store = store
        return store

    def _matcher() -> LimitOrderMatcher:
        matcher = getattr(app.state, "limit_order_matcher", None)
        if matcher is None:
            matcher = LimitOrderMatcher(app, config, event_bus, _pending_store())
            app.state.limit_order_matcher = matcher
        return matcher

    def _data_client() -> MarketDataClient:
        """Public market-data client bound to ``config.market_data_host``.

        Cached on ``app.state`` (one connection pool per process); tests can
        pre-inject a fake under the same attribute.
        """
        client = getattr(app.state, "market_data_client", None)
        if client is None:
            client = MarketDataClient(
                getattr(config, "market_data_host", "https://data-api.binance.vision"))
            app.state.market_data_client = client
        return client

    def _universe() -> Universe:
        universe = getattr(app.state, "universe", None)
        if universe is None:
            universe = Universe(config, client=_data_client())
            app.state.universe = universe
        return universe

    def _cache() -> TTLCache:
        cache = getattr(app.state, "_market_ttl_cache_v2", None)
        if cache is None:
            cache = TTLCache()
            app.state._market_ttl_cache_v2 = cache
        return cache

    async def _configured_client():
        """**Trading** client built from config (testnet flag + keys).

        Same pattern as ``web/routes/dashboard_partials.py::_configured_client``:
        never ``AsyncClient.create()`` bare, which targets unreachable mainnet.
        """
        from binance import AsyncClient
        return await AsyncClient.create(
            api_key=config.binance_api_key or None,
            api_secret=config.binance_api_secret or None,
            testnet=getattr(config, "binance_testnet", True),
        )

    async def _close(client):
        if client is not None:
            try:
                await client.close_connection()
            except Exception:
                pass

    def _error(message, status: int = 502):
        return JSONResponse({"error": str(message)}, status_code=status)

    def _bad_request(message):
        return JSONResponse({"error": str(message)}, status_code=400)

    async def _cached(kind: str, cache_key: str, fetcher, ttl: float = MARKET_CACHE_TTL):
        """Per-(kind, key) TTL cache with in-flight de-duplication."""
        return await _cache().get((kind, cache_key), ttl, fetcher)

    async def _ticker_snapshot() -> tuple[dict, str | None]:
        """All-symbol 24h tickers, cached :data:`TICKER24H_TTL` seconds.

        Returns ``(tickers_by_symbol, error)`` where each value is the raw
        upstream payload: the callers pick the fields they need.
        """
        async def _fetch():
            rows = await _data_client().ticker24h()
            if not isinstance(rows, list):
                raise MarketDataError("ticker/24hr did not return a list")
            return {str(r.get("symbol") or ""): r for r in rows if r.get("symbol")}

        return await _cache().get("ticker24h", TICKER24H_TTL, _fetch)

    async def _best_effort_ticker(symbol: str):
        """One 24h ticker via the cached snapshot; ``{}`` on failure (overview only)."""
        tickers, err = await _ticker_snapshot()
        if err:
            return {}
        return tickers.get(symbol, {})

    def _symbol_row(info, ticker: dict, universe: Universe) -> dict:
        """One row of ``/api/market/symbols`` (field order follows the contract)."""
        return {
            "symbol": info.symbol,
            "baseAsset": info.base_asset,
            "quoteAsset": info.quote_asset,
            "status": info.status,
            "last": _round(ticker.get("lastPrice"), 8),
            "change_pct": _round(ticker.get("priceChangePercent"), 4),
            "high": _round(ticker.get("highPrice"), 8),
            "low": _round(ticker.get("lowPrice"), 8),
            "volume": _round(ticker.get("volume"), 8),
            "quote_volume": _round(ticker.get("quoteVolume"), 8),
            "count": int(ticker["count"]) if ticker.get("count") is not None else None,
            "listing_date": info.listing_date or universe.listing_date(info.symbol),
            "tick_size": info.tick_size,
            "step_size": info.step_size,
            "min_notional": info.min_notional,
            "has_cached_data": universe.has_cached_data(info.symbol),
        }

    # ------------------------------------------------------------------
    # one exchange round-trip each (public market data)
    # ------------------------------------------------------------------
    async def _fetch_ticker(symbol: str) -> dict:
        t = await _data_client().ticker24h(symbol)
        if not t:
            raise MarketDataError("empty ticker response")
        return {
            "symbol": t.get("symbol", symbol),
            "last": _round(t.get("lastPrice"), 8),
            "open": _round(t.get("openPrice"), 8),
            "high": _round(t.get("highPrice"), 8),
            "low": _round(t.get("lowPrice"), 8),
            "volume": _round(t.get("volume"), 8),
            "quote_volume": _round(t.get("quoteVolume"), 8),
            "change": _round(t.get("priceChange"), 8),
            "change_pct": _round(t.get("priceChangePercent"), 4),
            "time": (int(t["closeTime"]) // 1000) if t.get("closeTime") else None,
        }

    async def _fetch_depth(symbol: str, limit: int) -> dict:
        book = await _data_client().order_book(symbol, limit)
        bids = [[float(p), float(q)] for p, q in (book.get("bids") or [])][:limit]
        asks = [[float(p), float(q)] for p, q in (book.get("asks") or [])][:limit]
        # The API already sorts bids descending / asks ascending; sort defensively
        # so the contract holds even if an upstream response is malformed.
        bids.sort(key=lambda lv: lv[0], reverse=True)
        asks.sort(key=lambda lv: lv[0])
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
        spread_pct = (spread / best_ask * 100) if (spread is not None and best_ask) else None
        return {
            "symbol": symbol,
            "lastUpdateId": book.get("lastUpdateId"),
            "bids": bids,
            "asks": asks,
            "spread": _round(spread, 8),
            # 6 dp on purpose: a real BTCUSDT spread is ~0.01 USDT ≈ 0.000012%,
            # which rounds to 0.0 (i.e. "no information") at 4 dp.
            "spread_pct": _round(spread_pct, 6),
            "bid_total": _round(sum(p * q for p, q in bids), 2),
            "ask_total": _round(sum(p * q for p, q in asks), 2),
        }

    async def _fetch_trades(symbol: str, limit: int) -> dict:
        raw = await _data_client().recent_trades(symbol, limit)
        trades = []
        for t in (raw or [])[:limit]:
            price = float(t.get("price") or 0)
            qty = float(t.get("qty") or 0)
            trades.append({
                "id": t.get("id"),
                "price": price,
                "qty": qty,
                "quote_qty": _round(t.get("quoteQty", price * qty), 8),
                "time": int(t.get("time") or 0) // 1000,
                "is_buyer_maker": bool(t.get("isBuyerMaker")),
            })
        trades.reverse()  # newest first
        return {"symbol": symbol, "trades": trades}

    async def _fetch_klines(symbol: str, interval: str, limit: int,
                            start_time=None, end_time=None) -> list[dict]:
        """OHLCV candles, ``time`` in epoch **seconds** (frozen kline contract)."""
        rows = await _data_client().klines(symbol, interval, limit,
                                           start_time=start_time, end_time=end_time)
        out = []
        for k in rows or []:
            try:
                out.append({
                    "time": int(k[0]) // 1000,
                    "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]),
                })
            except (TypeError, ValueError, IndexError):
                continue
        return out

    async def _daily_frame(symbol: str, limit: int = 200):
        """Daily closes for the risk/performance overlays (``None`` when unknown)."""
        try:
            rows = await _data_client().klines(symbol, "1d", limit=limit)
        except MarketDataError:
            return None
        if not rows:
            return None
        return metrics.closes_to_daily(
            [float(k[4]) for k in rows], [int(k[0]) for k in rows])

    async def _gather_limited(coro_factory, items, budget_s: float,
                              concurrency: int = OVERVIEW_CONCURRENCY) -> dict:
        """Run ``coro_factory(item)`` for each item under a time budget.

        Failures and timeouts are dropped silently — these overlays are
        best-effort by contract ("数据不足时字段为 null 而非报错").
        """
        sem = asyncio.Semaphore(max(1, concurrency))
        deadline = time.monotonic() + max(0.0, budget_s)
        results: dict = {}

        async def _one(item):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            async with sem:
                if time.monotonic() >= deadline:
                    return
                try:
                    value = await asyncio.wait_for(coro_factory(item),
                                                   timeout=max(1.0, remaining))
                except Exception:
                    return
                if value is not None:
                    results[item] = value

        await asyncio.gather(*(_one(i) for i in items), return_exceptions=True)
        return results

    # ==================================================================
    # 一、market data (read: any logged-in user, viewer included)
    # ==================================================================
    @app.get("/api/market/ticker")
    async def market_ticker(request: Request, symbol: str = "BTCUSDT"):
        if err := check_rate_limit(request, "ticker"):
            return err
        data, err = await _cached("ticker", symbol, lambda: _fetch_ticker(symbol))
        if err:
            return _error(f"ticker unavailable for {symbol}: {err}")
        return data

    @app.get("/api/market/depth")
    async def market_depth(request: Request, symbol: str = "BTCUSDT", limit: int = 20):
        if err := check_rate_limit(request, "depth"):
            return err
        limit = max(1, min(int(limit), 100))
        key = f"{symbol}|{limit}"
        data, err = await _cached("depth", key, lambda: _fetch_depth(symbol, limit),
                                  ttl=DEPTH_TTL)
        if err:
            return _error(f"depth unavailable for {symbol}: {err}")
        return data

    @app.get("/api/market/trades")
    async def market_trades(request: Request, symbol: str = "BTCUSDT", limit: int = 30):
        if err := check_rate_limit(request, "trades"):
            return err
        limit = max(1, min(int(limit), 100))
        key = f"{symbol}|{limit}"
        data, err = await _cached("trades", key, lambda: _fetch_trades(symbol, limit),
                                  ttl=TRADES_TTL)
        if err:
            return _error(f"trades unavailable for {symbol}: {err}")
        return data

    @app.get("/api/market/overview")
    async def market_overview(request: Request):
        """Engine symbols for the current persisted watchlist.

        The list is the same ``system_config.watchlist_symbols`` the engine
        watches (``core.market_data.universe``), not a hard-coded five pairs, so
        the overview follows a watchlist change instead of drifting from it.
        """
        if err := check_rate_limit(request, "overview"):
            return err
        tickers, err = await _ticker_snapshot()
        try:
            overview_symbols = await load_watchlist(config.db_path, DEFAULT_WATCHLIST)
        except Exception:
            overview_symbols = list(DEFAULT_WATCHLIST)
        symbols = []
        for sym in overview_symbols:
            t = {} if err else tickers.get(sym, {})
            symbols.append({
                "symbol": sym,
                "last": _round(t.get("lastPrice"), 8),
                "change_pct": _round(t.get("priceChangePercent"), 4),
                "quote_volume": _round(t.get("quoteVolume"), 8),
            })
        return {"symbols": symbols, "updated_at": int(time.time())}

    # ------------------------------------------------------------------
    # 全币种行情接口 (contract §1)
    # ------------------------------------------------------------------
    @app.get("/api/market/ticker24h")
    async def market_ticker24h(request: Request):
        """Every symbol's 24h stats — cached 60s (upstream costs ~10s)."""
        if err := check_rate_limit(request, "ticker"):
            return err
        tickers, err = await _ticker_snapshot()
        if err:
            return _error(f"ticker24h unavailable: {err}")
        payload = {
            sym: {
                "last": _round(t.get("lastPrice"), 8),
                "change_pct": _round(t.get("priceChangePercent"), 4),
                "high": _round(t.get("highPrice"), 8),
                "low": _round(t.get("lowPrice"), 8),
                "volume": _round(t.get("volume"), 8),
                "quote_volume": _round(t.get("quoteVolume"), 8),
                "count": int(t["count"]) if t.get("count") is not None else None,
            }
            for sym, t in tickers.items()
        }
        return {"updated_at": int(time.time()), "tickers": payload}

    @app.get("/api/market/symbols")
    async def market_symbols(request: Request, q: str = "", quote: str = "USDT", limit: int = 50,
                             offset: int = 0, sort: str = "volume"):
        """Searchable / sortable / pageable view of the whole USDT universe."""
        if err := check_rate_limit(request, "overview"):
            return err
        try:
            limit = max(1, min(int(limit), 200))
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            return _bad_request("limit and offset must be integers")
        sort_key = (sort or "volume").strip().lower()
        if sort_key not in ("volume", "change", "symbol"):
            return _bad_request(f"invalid sort '{sort}': expected volume|change|symbol")

        universe = _universe()
        try:
            await universe.get_symbols()
        except MarketDataError as e:
            return _error(f"symbol universe unavailable: {e}")
        tickers, terr = await _ticker_snapshot()
        if terr:
            return _error(f"symbol universe unavailable: {terr}")

        infos = universe.filter(q=q, quote=quote or "USDT")
        rows = [_symbol_row(i, tickers.get(i.symbol, {}), universe) for i in infos]

        if sort_key == "volume":
            rows.sort(key=lambda r: (r["quote_volume"] is None, -(r["quote_volume"] or 0)))
        elif sort_key == "change":
            rows.sort(key=lambda r: (r["change_pct"] is None, -(r["change_pct"] or 0)))
        else:
            rows.sort(key=lambda r: r["symbol"])

        page = rows[offset:offset + limit]
        # Listing dates cost one upstream call each and are persisted, so only the
        # page that is actually returned is filled in, under a small time budget.
        missing = [r["symbol"] for r in page if not r["listing_date"]]
        if missing:
            await universe.batch_fetch_listing_dates(missing)
            for row in page:
                row["listing_date"] = row["listing_date"] or universe.listing_date(row["symbol"])

        return {"total": len(rows), "offset": offset, "limit": limit, "symbols": page}

    # ------------------------------------------------------------------
    # 自选列表 (contract §1: persisted in system_config, cap 30)
    # ------------------------------------------------------------------
    @app.get("/api/market/watchlist")
    async def get_watchlist():
        symbols = await load_watchlist(config.db_path, DEFAULT_WATCHLIST)
        return {"symbols": symbols, "max": WATCHLIST_MAX}

    @app.post("/api/market/watchlist")
    async def set_watchlist(request: Request, symbols: str = Form("")):
        if err := _require_trader(request):
            return err
        requested = [s.strip().upper() for s in str(symbols or "").split(",") if s.strip()]
        requested = list(dict.fromkeys(requested))
        if not requested:
            return _bad_request("symbols must contain at least one symbol")
        if len(requested) > WATCHLIST_MAX:
            return _bad_request(f"watchlist is capped at {WATCHLIST_MAX} symbols "
                                f"(got {len(requested)})")

        universe = _universe()
        universe_loaded = True
        try:
            await universe.get_symbols()
        except MarketDataError:
            # Offline host: keep the write working rather than failing the page.
            universe_loaded = False
        accepted, rejected = validate_watchlist(
            requested, universe, tolerate_unknown=not universe_loaded)
        if rejected:
            return _bad_request(
                "unknown or non-trading symbols: " + ", ".join(rejected))
        try:
            await save_watchlist(config.db_path, accepted)
        except Exception as e:
            return _error(f"could not persist watchlist: {e}", 500)
        # Hot re-subscribe of the websocket stream is out of scope: the trading
        # engine picks the new list up on the next start.
        return {"ok": True, "symbols": accepted, "restart_required": True}

    # ------------------------------------------------------------------
    # 币种信息 (contract §2)
    # ------------------------------------------------------------------
    async def _coin_detail(symbol: str) -> dict:
        symbol = symbol.upper()
        universe = _universe()
        try:
            await universe.get_symbols()
        except MarketDataError as e:
            raise MarketDataError(f"symbol universe unavailable: {e}")
        info = universe.get(symbol)
        if info is None:
            raise KeyError(symbol)

        tickers, terr = await _ticker_snapshot()
        raw = {} if terr else tickers.get(symbol, {})

        # `listing_date` is one cheap call (limit=1 from the epoch) and cached.
        if not info.listing_date:
            info.listing_date = universe.listing_date(symbol)
        if not info.listing_date:
            info.listing_date = await universe.fetch_listing_date(symbol)
            universe.set_listing_date(symbol, info.listing_date)
            universe.save_listing_dates()

        depth_task = asyncio.create_task(_grab_depth(symbol))
        daily = await _daily_frame(symbol, limit=200)
        depth_raw = await depth_task

        last = _round(raw.get("lastPrice"), 8)
        change_pct = _round(raw.get("priceChangePercent"), 4)
        perf = metrics.performance(daily, change_pct_24h=change_pct)
        vol = metrics.volatility_annualized(daily)
        mdd = metrics.max_drawdown(daily)

        btc_daily = None
        if symbol != "BTCUSDT":
            btc_daily = await _daily_frame("BTCUSDT", limit=200)
        corr = metrics.correlation(daily, btc_daily)

        spread_pct = depth_raw.get("spread_pct")  # plain ratio (spread / best ask)
        quote_volume = _round(raw.get("quoteVolume"), 8)
        count = int(raw["count"]) if raw.get("count") is not None else None
        listing_age = None
        if info.listing_date:
            try:
                listed = time.mktime(time.strptime(info.listing_date, "%Y-%m-%d"))
                listing_age = int(max(0, (time.time() - listed) / 86400.0))
            except ValueError:
                listing_age = None

        liq = metrics.liquidity_score(quote_volume)
        spr = metrics.spread_score(spread_pct)
        volscore = metrics.volume_score(count)
        parts = [s for s in (liq, spr, volscore) if s is not None]
        overall = int(round(sum(parts) / len(parts))) if parts else None

        return {
            "symbol": symbol,
            "base_asset": info.base_asset,
            "quote_asset": info.quote_asset,
            "status": info.status,
            "listing_date": info.listing_date,
            "filters": {
                "tick_size": info.tick_size,
                "step_size": info.step_size,
                "min_notional": info.min_notional,
            },
            "ticker": {
                "last": last,
                "change_pct": change_pct,
                "high": _round(raw.get("highPrice"), 8),
                "low": _round(raw.get("lowPrice"), 8),
                "volume": _round(raw.get("volume"), 8),
                "quote_volume": quote_volume,
                "count": count,
            },
            "depth_summary": depth_raw,
            "performance": perf,
            "risk": {
                "volatility_30d_annualized": vol,
                "max_drawdown_90d": mdd,
                "liquidity_score": liq,
                "spread_score": spr,
                "volume_score": volscore,
                "overall_score": overall,
                "flags": metrics.risk_flags(
                    quote_volume=quote_volume, spread_pct=spread_pct,
                    volatility=vol, drawdown=mdd, trade_count=count,
                    listing_age_days=listing_age),
            },
            "correlation_btc_30d": corr,
            "cached_intervals": universe.cached_intervals(symbol),
        }

    async def _grab_depth(symbol: str) -> dict:
        """±1% depth summary for the coin page (best effort → nulls)."""
        try:
            book = await _data_client().order_book(symbol, 100)
        except MarketDataError:
            return {"spread": None, "spread_pct": None, "bid_depth_1pct": None,
                    "ask_depth_1pct": None, "bid_total": None, "ask_total": None}
        bids = [[float(p), float(q)] for p, q in (book.get("bids") or [])]
        asks = [[float(p), float(q)] for p, q in (book.get("asks") or [])]
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
        # `spread_pct` is a plain ratio (spread / mid), matching the contract's
        # example: 0.01 USDT on SOL ≈ 1.2e-05.
        spread_pct = None
        if spread is not None and best_bid and best_ask:
            mid = (best_bid + best_ask) / 2.0
            spread_pct = spread / mid if mid else None
        return {
            "spread": _round(spread, 8),
            "spread_pct": _round(spread_pct, 8),
            "bid_depth_1pct": _round(sum(p * q for p, q in bids
                                         if best_bid and p >= best_bid * 0.99), 2),
            "ask_depth_1pct": _round(sum(p * q for p, q in asks
                                         if best_ask and p <= best_ask * 1.01), 2),
            "bid_total": _round(sum(p * q for p, q in bids), 2),
            "ask_total": _round(sum(p * q for p, q in asks), 2),
        }

    @app.get("/api/coin/{symbol}")
    async def coin_detail(request: Request, symbol: str):
        # Shape first, and *before* any upstream call or cache entry: the symbol is
        # a path component and a cache key, and it used to reach the network
        # unvalidated here while `/api/audit/{symbol}` validated it — so the coin
        # cache was keyed by whatever the caller sent.
        canonical = valid_symbol(symbol)
        if canonical is None:
            return JSONResponse({"error": f"invalid symbol '{symbol}'"},
                                status_code=400)
        if err := check_rate_limit(request, "coin"):
            return err
        key = canonical

        async def _fetch():
            try:
                return await _coin_detail(key)
            except KeyError:
                raise MarketDataError(f"unknown symbol '{key}'")

        data, err = await _cache().get(("coin", key), COIN_TTL, _fetch)
        if err:
            if "unknown symbol" in err:
                return JSONResponse({"error": err}, status_code=404)
            return _error(f"coin info unavailable for {key}: {err}")
        return data

    # ------------------------------------------------------------------
    # 全市场数据总览 (contract §3)
    # ------------------------------------------------------------------
    async def _data_overview() -> dict:
        universe = _universe()
        try:
            await universe.get_symbols()
        except MarketDataError as e:
            raise MarketDataError(f"symbol universe unavailable: {e}")
        tickers, err = await _ticker_snapshot()
        if err:
            raise MarketDataError(f"ticker24h unavailable: {err}")

        usdt = {s.symbol for s in universe.filter(quote="USDT")}
        pool = [(sym, t) for sym, t in tickers.items() if sym in usdt]

        def _f(t, key):
            try:
                return float(t.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        total_qv = sum(_f(t, "quoteVolume") for _, t in pool)
        up = sum(1 for _, t in pool if _f(t, "priceChangePercent") > 0)
        down = sum(1 for _, t in pool if _f(t, "priceChangePercent") < 0)
        flat = len(pool) - up - down

        def _row(sym, t):
            return {
                "symbol": sym,
                "last": _round(t.get("lastPrice"), 8),
                "change_pct": _round(t.get("priceChangePercent"), 4),
                "quote_volume": _round(_f(t, "quoteVolume"), 2),
                "count": int(t["count"]) if t.get("count") is not None else None,
            }

        top_gainers = sorted(pool, key=lambda kv: -_f(kv[1], "priceChangePercent"))[:LEADERBOARD_SIZE]
        top_losers = sorted(pool, key=lambda kv: _f(kv[1], "priceChangePercent"))[:LEADERBOARD_SIZE]
        top_volume = sorted(pool, key=lambda kv: -_f(kv[1], "quoteVolume"))[:LEADERBOARD_SIZE]
        most_active = sorted(pool, key=lambda kv: -_f(kv[1], "count"))[:LEADERBOARD_SIZE]

        top_by_volume = sorted(pool, key=lambda kv: -_f(kv[1], "quoteVolume"))
        vol_symbols = [s for s, _ in top_by_volume[:OVERVIEW_VOLATILITY_SAMPLE]]

        async def _vol(sym):
            daily = await _daily_frame(sym, limit=60)
            return {"symbol": sym,
                    "volatility_30d_annualized": metrics.volatility_annualized(daily)}

        vol_map = await _gather_limited(_vol, vol_symbols, OVERVIEW_OVERLAY_BUDGET_S)
        volatility_leaders = sorted(
            (v for v in vol_map.values() if v["volatility_30d_annualized"] is not None),
            key=lambda v: -v["volatility_30d_annualized"])[:LEADERBOARD_SIZE]

        async def _spread(sym):
            depth = await _grab_depth(sym)
            if depth.get("spread_pct") is None:
                return None
            return {"symbol": sym, "spread_pct": depth["spread_pct"]}

        spread_map = await _gather_limited(
            _spread, vol_symbols[:OVERVIEW_SPREAD_SAMPLE], OVERVIEW_OVERLAY_BUDGET_S)
        spread_widest = sorted(
            (v for v in spread_map.values() if v is not None),
            key=lambda v: -(v["spread_pct"] or 0))[:LEADERBOARD_SIZE]

        btc_eth = {}
        for sym in ("BTCUSDT", "ETHUSDT"):
            t = tickers.get(sym)
            if t and total_qv > 0:
                btc_eth[sym] = round(_f(t, "quoteVolume") / total_qv, 6)

        return {
            "updated_at": int(time.time()),
            "totals": {
                "symbols": len(pool),
                "quote_volume_usdt": round(total_qv, 2),
                "up": up, "down": down, "flat": flat,
            },
            "top_gainers": [_row(s, t) for s, t in top_gainers],
            "top_losers": [_row(s, t) for s, t in top_losers],
            "top_volume": [_row(s, t) for s, t in top_volume],
            "most_active": [{"symbol": s, "count": _row(s, t)["count"],
                             "quote_volume": _row(s, t)["quote_volume"]}
                            for s, t in most_active],
            "volatility_leaders": volatility_leaders,
            "spread_widest": spread_widest,
            "btc_eth_share": btc_eth,
        }

    @app.get("/api/data/overview")
    async def data_overview(request: Request):
        # One minute of cache on purpose: this fans out over ~120 upstream calls.
        if err := check_rate_limit(request, "overview"):
            return err
        data, err = await _cache().get("data_overview", TICKER24H_TTL, _data_overview)
        if err:
            return _error(f"market overview unavailable: {err}")
        return data

    # ------------------------------------------------------------------
    # /api/kline/{symbol} — frozen shape, but served from the data host.
    #
    # The handler lives in `web/routes/dashboard_partials.py`, which is outside
    # this module's write scope; FastAPI matches routes in registration order and
    # that module registers first, so a second `@app.get` here would be dead code.
    # Instead the stale route (testnet client) is replaced in the router *after*
    # registering the data-host version below.  Shape is unchanged: `time` in
    # epoch SECONDS.
    # ------------------------------------------------------------------
    @app.get("/api/kline/{symbol}")
    async def get_kline(request: Request, symbol: str, interval: str = "1h", limit: int = 200):
        """OHLCV candles for chart rendering, from the market-data host.

        Prefers the engine's own cache (the exact data the strategies trade on,
        and no new connection per chart refresh); falls back to a REST call
        against ``config.market_data_host``.
        """
        if err := check_rate_limit(request, "klines"):
            return err
        limit = max(1, min(int(limit), 1000))

        md = getattr(app.state, "market_data", None)
        if md is None:
            engine = getattr(app.state, "strategy_engine", None)
            md = getattr(engine, "market_data", None)
        if md is not None:
            try:
                df = await md.get_historical(symbol, interval, limit=limit)
                if df is not None and len(df) > 0:
                    return [
                        {
                            "time": int(idx.timestamp()),
                            "open": float(row["open"]), "high": float(row["high"]),
                            "low": float(row["low"]), "close": float(row["close"]),
                            "volume": float(row["volume"]),
                        }
                        for idx, row in df.tail(limit).iterrows()
                    ]
            except Exception as e:
                from loguru import logger
                logger.warning(f"Kline cache path failed for {symbol}/{interval}: {e}")

        try:
            return await _fetch_klines(symbol, interval, limit)
        except MarketDataError as e:
            return JSONResponse({"error": str(e)}, status_code=502)

    # Replace the testnet-based handler registered earlier by
    # `web/routes/dashboard_partials.py` (kept out of that file's scope).
    _mine = app.router.routes[-1]
    for _route in [r for r in app.router.routes
                   if getattr(r, "path", None) == "/api/kline/{symbol}" and r is not _mine]:
        app.router.routes.remove(_route)

    # ==================================================================
    # 二、account + positions (read: any logged-in user)
    # ==================================================================
    def _live_price(symbol: str, fallback: float) -> float:
        price_fn = getattr(app.state, "get_price", None)
        if price_fn:
            try:
                price = price_fn(symbol)
            except Exception:
                price = None
            if price:
                return float(price)
        return float(fallback or 0)

    @app.get("/api/account")
    async def get_account():
        executor = getattr(app.state, "executor", None)
        positions_raw = executor.get_open_positions() if executor else {}

        # The DB is the source of truth. `app.state.balance` is only a cache that the
        # engine never refreshes (it calls atomic_adjust_balance + risk_manager
        # .update_balance), so reading the cache made the UI report a stale, drifting
        # balance — measured +580.46 vs the DB on the live service.
        balance = await load_sim_balance(config.db_path)
        app.state.balance = balance

        positions = []
        positions_value = 0.0
        unrealized = 0.0
        for pos in positions_raw.values():
            entry = float(pos.get("entry_price") or 0)
            qty = float(pos.get("quantity") or 0)
            side = pos.get("side", "long")
            current = _live_price(pos.get("symbol", ""), pos.get("current_price") or entry)
            pnl = (current - entry) * qty if side == "long" else (entry - current) * qty
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0.0
            value = qty * current
            amount = pos.get("amount_usdt", qty * entry)
            positions_value += value
            unrealized += pnl
            positions.append({
                "symbol": pos.get("symbol", ""),
                "side": side,
                "quantity": _round(qty, 6),
                "entry_price": _round(entry, 8),
                "current_price": _round(current, 8),
                "unrealized_pnl": _round(pnl, 4),
                "pnl_pct": _round(pnl_pct, 4),
                "position_value": _round(value, 2),
                "amount_usdt": _round(amount, 2),
                "stop_loss": _round(pos.get("stop_loss"), 8),
                "strategy_name": pos.get("strategy_name", ""),
                "position_type": pos.get("position_type", "satellite"),
            })

        try:
            frozen = await _pending_store().frozen_total()
            pending_count = await _pending_store().open_count()
        except Exception as e:
            from loguru import logger
            logger.warning(f"Could not read pending orders: {e}")
            frozen, pending_count = 0.0, 0

        balance = round(float(balance), 2)
        # Realised costs (contract §五之二).  Purely additive: the five existing
        # keys above keep their exact meaning.
        fees_paid_total, slippage_paid_total, net_pnl_total = await _cost_totals()
        return {
            "balance": balance,
            "available": round(balance - frozen, 2),
            "frozen": round(frozen, 2),
            "equity": round(balance + positions_value, 2),
            "positions_value": round(positions_value, 2),
            "unrealized_pnl": round(unrealized, 2),
            "positions": positions,
            "pending_count": pending_count,
            "mode": getattr(config, "mode", "sim"),
            "fees_paid_total": fees_paid_total,
            "slippage_paid_total": slippage_paid_total,
            "net_pnl_total": net_pnl_total,
        }

    # ==================================================================
    # 三、orders (write: trader or admin)
    # ==================================================================
    def _risk_signal(symbol, side, price, quantity, position_type, stop_loss_pct,
                     amount_usdt, trader, strategy_name, order_type="market",
                     timeframe=DEFAULT_TIMEFRAME):
        stop_loss = (price * (1 - stop_loss_pct / 100) if side == "long"
                     else price * (1 + stop_loss_pct / 100))
        return {
            "symbol": symbol, "side": side,
            "price": price, "quantity": round(quantity, 6),
            "stop_loss": round(stop_loss, 2),
            "position_type": position_type,
            "amount_usdt": amount_usdt,
            "trader": trader,
            "strategy_name": strategy_name,
            "strategy": "manual" if order_type == "market" else "limit",
            # Attribution for the `trades` row: the timeframe the order was
            # placed on, defaulting to the registry's DEFAULT_TIMEFRAME only
            # when none was supplied.
            "timeframe": timeframe or DEFAULT_TIMEFRAME,
            "confidence": 1.0,
            "order_type": order_type,
        }

    @app.post("/api/order")
    async def place_order(request: Request, symbol: str = Form(...), side: str = Form(...),
                          type: str = Form("market"),
                          amount_usdt: float = Form(...),
                          price: float = Form(None),
                          position_type: str = Form("satellite"),
                          stop_loss_pct: float = Form(2.0),
                          timeframe: str = Form(DEFAULT_TIMEFRAME)):
        if err := _require_trader(request):
            return err

        order_type = (type or "market").strip().lower()
        side_key = (side or "").strip().lower()
        symbol = (symbol or "").strip().upper()
        if side_key not in ("long", "short"):
            return _bad_request(f"invalid side '{side}': expected long|short")
        if order_type not in ("market", "limit"):
            return _bad_request(f"invalid type '{type}': expected market|limit")
        if not symbol:
            return _bad_request("symbol is required")
        if amount_usdt is None or float(amount_usdt) <= 0:
            return _bad_request("amount_usdt must be > 0")
        amount_usdt = round(float(amount_usdt), 2)

        user = getattr(request.state, "user", None)
        trader = getattr(user, "username", "manual") if user else "manual"
        executor = getattr(app.state, "executor", None)
        risk_manager = getattr(app.state, "risk_manager", None)
        if not executor:
            return _error("交易器未就绪", 503)

        # ---- limit: validate → freeze → persist ----
        if order_type == "limit":
            if price is None or float(price) <= 0:
                return _bad_request("price must be > 0 for a limit order")
            price = float(price)
            store = _pending_store()
            try:
                # DB-first for the same reason as /api/account (stale cache risk)
                balance = await load_sim_balance(config.db_path)
                app.state.balance = balance
                frozen = await store.frozen_total()
            except Exception as e:
                return _error(f"账户数据不可用: {e}", 503)
            available = float(balance) - float(frozen)
            if amount_usdt > available + 1e-9:
                return _bad_request(
                    f"余额不足: 可用 {available:.2f} USDT, 需要 {amount_usdt:.2f} USDT")
            quantity = round(amount_usdt / price, 8)
            if quantity <= 0:
                return _bad_request("quantity rounds to zero — increase amount_usdt")
            try:
                order = await store.add(
                    symbol=symbol, side=side_key, price=price, quantity=quantity,
                    amount_usdt=amount_usdt, position_type=position_type,
                    stop_loss_pct=float(stop_loss_pct or 0), trader=trader,
                    strategy_name="")
            except Exception as e:
                return _error(f"无法写入待成交订单: {e}", 500)
            return {"ok": True, "order": {
                "id": order.id, "symbol": order.symbol, "side": order.side,
                "type": "limit", "price": round(price, 8),
                "quantity": round(quantity, 6), "amount_usdt": amount_usdt,
                "status": STATUS_OPEN,
            }}

        # ---- market: reuse the existing risk pipeline (no re-implementation) ----
        price_fn = getattr(app.state, "get_price", None)
        current_price = None
        if price_fn:
            try:
                current_price = price_fn(symbol)
            except Exception:
                current_price = None
        if not current_price:
            current_price = await _rest_price(symbol)
        if not current_price or current_price <= 0:
            return _error("无法获取实时价格，请检查网络或 API 配置", 503)

        current_price = float(current_price)
        quantity = round(amount_usdt / current_price, 6)
        if quantity <= 0:
            return _bad_request("amount_usdt is too small for this price")

        if not risk_manager:
            # FAIL CLOSED — without it none of the 7 safety gates run.
            return JSONResponse(
                {"ok": False, "error": "风控未就绪，已拒绝下单（不允许绕过风控）"},
                status_code=503)

        signal = _risk_signal(symbol, side_key, current_price, quantity, position_type,
                              float(stop_loss_pct or 0), amount_usdt, trader, "market",
                              timeframe=timeframe)
        result = await risk_manager.check_signal(signal)
        if not result.approved:
            return {"ok": False, "error": f"风控拒绝: {result.reason}"}

        if result.adjusted_quantity is not None and result.adjusted_quantity < signal["quantity"]:
            signal["quantity"] = result.adjusted_quantity
            signal["amount_usdt"] = result.adjusted_quantity * current_price
        if result.adjusted_stop_loss is not None:
            signal["stop_loss"] = result.adjusted_stop_loss
        if result.adjusted_leverage is not None:
            signal["leverage"] = result.adjusted_leverage

        await event_bus.publish(Event(EventType.ORDER_REQUEST, signal))
        return {"ok": True, "order": {
            "symbol": symbol, "side": side_key, "type": "market",
            "quantity": round(signal["quantity"], 6),
            "price": round(current_price, 8),
            "amount_usdt": round(signal.get("amount_usdt", amount_usdt), 2),
            "status": "filled",
        }}

    async def _rest_price(symbol: str):
        """Last resort: the public market-data host's last price (no keys needed)."""
        try:
            ticker = await _data_client().symbol_price(symbol)
            return float(ticker["price"])
        except Exception as e:
            from loguru import logger
            logger.warning(f"REST price fallback failed for {symbol}: {e}")
            return None

    @app.get("/api/orders")
    async def list_orders(status: str = "all", limit: int = 50):
        status_key = (status or "all").strip().lower()
        if status_key not in ("open", "history", "all"):
            return _bad_request(f"invalid status '{status}': expected open|history|all")
        limit = max(1, min(int(limit), 200))
        try:
            orders = await _pending_store().list(status=status_key, limit=limit)
        except Exception as e:
            return _error(f"无法读取订单: {e}", 500)
        return {"orders": [o.to_dict() for o in orders]}

    @app.post("/api/orders/{order_id}/cancel")
    async def cancel_order(order_id: int, request: Request):
        if err := _require_trader(request):
            return err
        store = _pending_store()
        try:
            order = await store.get(order_id)
            if order is None:
                return JSONResponse({"error": "order not found"}, status_code=404)
            if order.status != STATUS_OPEN:
                return JSONResponse(
                    {"error": f"order is not open (status={order.status})"},
                    status_code=409)
            await store.mark_cancelled(order_id, "cancelled by user")
        except Exception as e:
            return _error(f"撤单失败: {e}", 500)
        refreshed = await store.get(order_id)
        return {"ok": True, "order": refreshed.to_dict()}

    # ==================================================================
    # 五之二、fee tier / cost estimation (contract §五之二)
    #
    # The tier is a MANUAL selection: this host cannot read the real Binance
    # account's 30-day volume or BNB holdings (mainnet account endpoints are
    # unreachable), so the table is the standard Binance spot schedule and the
    # choice is persisted in `system_config`, where the executor picks it up on
    # the next fill.  `sim_cost_quote` is shared with the executor so an estimate
    # can never drift from what is actually charged.
    # ==================================================================
    async def _cost_settings() -> dict:
        return await load_sim_cost_settings(config.db_path, config)

    def _tier_payload(settings: dict) -> dict:
        return {
            "current": settings["fee_tier"],
            "use_bnb_discount": bool(settings["use_bnb_discount"]),
            "bnb_discount_pct": BNB_DISCOUNT_PCT,
            "tiers": [dict(t) for t in FEE_TIER_TABLE],
            "taker_pct": sim_fee_pct(settings, "market"),
            "maker_pct": sim_fee_pct(settings, "limit"),
            "slippage_bps": float(settings.get("slippage_bps") or 0.0),
            "enabled": bool(settings.get("enabled", True)),
            "note": FEE_TIER_NOTE,
        }

    @app.get("/api/fee/tier")
    async def fee_tier():
        return _tier_payload(await _cost_settings())

    @app.post("/api/fee/tier")
    async def set_fee_tier(request: Request, fee_tier: str = Form(""),
                           use_bnb_discount: str = Form("false")):
        """Persist the manual fee-tier selection (trader+); invalid tier → 400."""
        if err := _require_trader(request):
            return err
        wanted = str(fee_tier or "").strip().upper()
        entry = fee_tier_entry(wanted)
        if entry is None:
            return _bad_request(
                f"invalid fee_tier '{fee_tier}': expected one of "
                + ", ".join(t["tier"] for t in FEE_TIER_TABLE))
        use_bnb = str(use_bnb_discount or "").strip().lower() in ("1", "true", "yes", "on")
        try:
            db = await aiosqlite.connect(config.db_path)
            try:
                for key, value in ((SIM_FEE_TIER_KEY, wanted),
                                   (SIM_USE_BNB_DISCOUNT_KEY, "true" if use_bnb else "false")):
                    await db.execute(
                        "INSERT OR REPLACE INTO system_config (key, value, category) "
                        "VALUES (?, ?, 'trading')", (key, value))
                await db.commit()
            finally:
                await db.close()
        except Exception as e:
            return _error(f"无法保存手续费等级: {e}", 500)
        return {"ok": True, **_tier_payload(await _cost_settings())}

    async def _estimate_price(symbol: str) -> float:
        """A reference price for ANY symbol, not just the watched list.

        The engine's cache only covers the watchlist (≤30 pairs), but the trade
        page lets the user pick any of the ~496 USDT pairs — for those the
        estimator used to return 503. Falls back to the public data host.
        """
        price_fn = getattr(app.state, "get_price", None)
        if price_fn:
            try:
                price = price_fn(symbol)
            except Exception:
                price = None
            if price:
                return float(price)
        try:
            data_client = getattr(app.state, "fee_price_client", None)
            if data_client is None:
                from core.market_data.data_client import MarketDataClient
                data_client = MarketDataClient(
                    getattr(config, "market_data_host", None))
                app.state.fee_price_client = data_client
            ticker = await data_client.ticker24h(symbol)
            last = ticker.get("lastPrice") if isinstance(ticker, dict) else None
            return float(last) if last else 0.0
        except Exception:
            return 0.0

    @app.get("/api/fee/estimate")
    async def fee_estimate(symbol: str = "BTCUSDT", side: str = "long",
                           type: str = "market", amount_usdt: float = 0.0,
                           price: float = None, quantity: float = None):
        symbol = str(symbol or "").strip().upper()
        side_key = str(side or "long").strip().lower()
        if side_key not in ("long", "short", "buy", "sell"):
            return _bad_request(f"invalid side '{side}': expected long|short")
        order_type = str(type or "market").strip().lower()
        if order_type not in ("market", "limit"):
            return _bad_request(f"invalid type '{type}': expected market|limit")

        settings = await _cost_settings()
        notes: list[str] = []
        ref_price = float(price or 0) if order_type == "limit" else await _estimate_price(symbol)
        if order_type == "market" and ref_price <= 0:
            # Fall back to the limit-price argument so the page still gets numbers
            # instead of a 502 when the engine has no cached quote for the symbol.
            ref_price = float(price or 0)
            if ref_price > 0:
                notes.append("引擎无该交易对缓存价，已用请求给定的 price 估算。")
        if ref_price <= 0:
            return _error(f"无法获取 {symbol} 的价格，无法估算成本", 503)

        notional = float(amount_usdt or 0)
        qty = float(quantity) if quantity else 0.0
        if notional <= 0 and qty <= 0:
            return _bad_request("amount_usdt (or quantity) must be > 0")
        if qty <= 0:
            qty = notional / ref_price
        elif notional <= 0:
            notional = qty * ref_price
        # The quoted notional is what the page typed; costs are computed on the
        # cost-aware fill inside sim_cost_quote.
        notional = round(notional, 2)

        q = sim_cost_quote(symbol, side_key, order_type, ref_price, qty, settings)
        if order_type == "limit":
            notes.append("限价单按限价成交，不计点差与滑点，只收手续费。")
        else:
            notes.append(f"市价单按每边点差 {sim_spread_pct(settings, symbol):g}% "
                         f"+ 滑点 {float(settings.get('slippage_bps') or 0):g}bp 计入成交价。")
        if settings.get("use_bnb_discount"):
            notes.append(f"已启用 BNB 抵扣，费率按 {BNB_DISCOUNT_PCT}% 折扣。")
        if not settings.get("enabled", True):
            notes.append("成本模型已关闭（sim.cost_model.enabled=false）。")
        notes.append(FEE_TIER_NOTE)

        return {
            "symbol": symbol,
            "side": side_key,
            "type": q["order_type"],
            "notional": notional,
            "quantity": round(qty, 8),
            "price": ref_price,
            "fill_price": round(q["fill_price"], 8),
            "effective_price": round(q["effective_price"], 8),
            "fee_pct": q["fee_pct"],
            "fee_usdt": round(q["fee_usdt"], 6),
            "slippage_bps": q["slippage_bps"],
            "spread_pct": q["spread_pct"],
            "slippage_usdt": round(q["slippage_usdt"], 6),
            "total_cost_usdt": round(q["cost_usdt"], 6),
            "cost_pct": round(q["cost_usdt"] / notional * 100, 6) if notional else 0.0,
            "fee_tier": settings["fee_tier"],
            "use_bnb_discount": bool(settings["use_bnb_discount"]),
            "notes": notes,
        }

    # ==================================================================
    # 成交历史
    # ==================================================================
    @app.get("/api/history/trades")
    async def trade_history(limit: int = 50):
        limit = max(1, min(int(limit), 200))
        try:
            db = await aiosqlite.connect(config.db_path)
            db.row_factory = aiosqlite.Row
            try:
                # SELECT * on purpose: `exit_reason` is not in the base schema (and
                # older databases may predate other columns), so a named-column
                # query would raise OperationalError and 502 the endpoint.
                #
                # Only REAL exits are returned.  `close_position` flips the OPEN
                # row to `status='closed'` as well, so the old `WHERE
                # status='closed'` returned both rows of every round trip — half
                # of the "history" was the entry leg, with `exit_price NULL / pnl
                # 0` (688 of 1376 closed rows on the live DB, 24 of the newest 50).
                # The `exit_price IS NOT NULL` arm is for rows written before the
                # `action` column existed, which are still genuine exits.
                cursor = await db.execute(
                    "SELECT * FROM trades "
                    "WHERE action IN ('close', 'reduce') "
                    "   OR (action = 'open' AND status = 'closed' AND exit_price IS NOT NULL) "
                    "ORDER BY closed_at DESC, id DESC LIMIT ?", (limit,))
                rows = [dict(r) for r in await cursor.fetchall()]
            finally:
                await db.close()
        except Exception as e:
            return _error(f"无法读取成交历史: {e}", 500)

        trades = []
        for r in rows:
            fee = r.get("fee")
            slippage = r.get("slippage")
            pnl = r.get("pnl")
            # `pnl` is already stored net of both sides of the round trip, so for
            # a cost-aware row `net_pnl == pnl`; a row written before migration v3
            # has NULL costs, so its gross figure is the only one available.
            net_pnl = round(float(pnl), 8) if pnl is not None else None
            trades.append({
                "id": r.get("id"),
                "symbol": r.get("symbol"),
                "side": r.get("side"),
                "entry_price": r.get("entry_price"),
                "exit_price": r.get("exit_price"),
                "quantity": r.get("quantity"),
                "pnl": pnl,
                "pnl_pct": r.get("pnl_pct"),
                "strategy": r.get("strategy"),
                "exit_reason": r.get("exit_reason"),
                "opened_at": str(r["opened_at"]) if r.get("opened_at") is not None else None,
                "closed_at": str(r["closed_at"]) if r.get("closed_at") is not None else None,
                # ---- added by contract §五之二 (existing keys untouched) ----
                "fee": fee,
                "slippage": slippage,
                "fill_price": r.get("fill_price"),
                "net_pnl": net_pnl,
            })
        return {"trades": trades}

    async def _cost_totals() -> tuple[float, float, float]:
        """``(fees_paid_total, slippage_paid_total, net_pnl_total)``.

        Costs are summed over **every** row (the buy-side cost sits on the open
        row, the sell-side one on the close row, so summing all rows counts each
        fill exactly once).  Realised PnL is summed over closed rows only, which
        already includes the buy-side cost — hence this is NOT the same as
        ``fees+slippage`` plus the gross result.
        """
        db = await aiosqlite.connect(config.db_path)
        db.row_factory = aiosqlite.Row
        try:
            cursor = await db.execute(
                "SELECT COALESCE(SUM(fee), 0) AS fees, "
                "COALESCE(SUM(slippage), 0) AS slip, "
                "COALESCE(SUM(CASE WHEN status='closed' THEN pnl END), 0) AS net "
                "FROM trades")
            row = await cursor.fetchone()
            return (round(float(row["fees"] or 0.0), 6),
                    round(float(row["slip"] or 0.0), 6),
                    round(float(row["net"] or 0.0), 6))
        except Exception as e:
            from loguru import logger
            logger.warning(f"Could not sum trading costs: {e}")
            return 0.0, 0.0, 0.0
        finally:
            await db.close()

    # ==================================================================
    # matcher wiring — started by app/main.py alongside the other components;
    # exposed on ctx so main.py (and tests) can start/stop it.
    # ==================================================================
    app.state.pending_order_store = _pending_store()
    app.state.limit_order_matcher = _matcher()
    ctx.start_pending_matcher = lambda: start_matcher_task(app, config, event_bus)

    # Warm the universe cache in the background so the first page load does not
    # pay for the ~3 MB exchangeInfo download (6h TTL afterwards).  Failures are
    # swallowed: an unreachable host must not stop the app from starting.
    try:
        _loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - no loop (e.g. sync import)
        _loop = None
    if _loop is not None and not getattr(app.state, "universe_preload_started", False):
        app.state.universe_preload_started = True
        _universe().preload_from_disk()
        app.state.universe_warmup_task = _loop.create_task(_universe().refresh_if_needed())
