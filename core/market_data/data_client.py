"""Minimal async REST client for the public Binance **market data** mirror.

Why this exists (``docs/overhaul/MARKET_PAGES_API.md`` §0):

* ``https://api.binance.com`` — and therefore the bare ``binance.AsyncClient``
  built for mainnet — is **unreachable** from this host (timeout).
* ``https://testnet.binance.vision`` is reachable, but it only lists a handful
  of test pairs, so it cannot answer "show me every coin".
* ``https://data-api.binance.vision`` is reachable and serves the real mainnet
  spot market: 3716 symbols, 496 ``USDT``/``TRADING`` pairs, klines back to 2017.

This client therefore owns every *public market data* call (klines, ticker,
depth, trades).  The trading/order client stays on ``binance.testnet`` and is
built by ``web/routes/market.py::_configured_client``.

Design rules:

* one long-lived ``httpx.AsyncClient`` per instance (connection reuse matters:
  ``/api/market/ticker24h`` for all symbols takes ~10s of pure transfer time);
* a hard timeout on every request (default 15s) so a dead host can never hang a
  request handler;
* failures raise :class:`MarketDataError` with a short message — route handlers
  turn that into ``{"error": "..."}``, never a traceback.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx

#: Binance's documented maximum page size for /api/v3/klines.
KLINES_MAX_LIMIT = 1000

#: Every public market-data request gets this budget.  The contract requires
#: "timeouts ≤15s, never hang".
DEFAULT_TIMEOUT = 15.0


class MarketDataError(RuntimeError):
    """A market-data request failed (network, timeout, or non-2xx payload)."""


class MarketDataClient:
    """Async REST client bound to one market-data host."""

    def __init__(self, host: str, timeout: float = DEFAULT_TIMEOUT):
        self.host = str(host).rstrip("/")
        self.timeout = float(timeout)
        self._client: Optional[httpx.AsyncClient] = None
        self._client_loop = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    @property
    def client(self) -> httpx.AsyncClient:
        """The httpx client, (re)built when the running event loop changed.

        An ``httpx.AsyncClient`` pool is bound to the loop that created its
        connections; using it from another loop raises "Event loop is closed"
        (which is what ``TestClient`` does, one loop per request).  Production
        runs a single uvicorn loop, so this normally never rebuilds.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        stale = self._client is not None and (
            self._client_loop is not loop or getattr(self._client, "is_closed", False))
        if stale:
            # The old pool belongs to a loop that no longer runs; drop it (the
            # closed loop reclaims its sockets).  Never await here — this is a
            # sync property.
            self._client = None
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": "binance-trader/1.0 (market-data)"},
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
            self._client_loop = loop
        return self._client

    async def close(self) -> None:
        client, self._client = self._client, None
        self._client_loop = None
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # pragma: no cover - best effort teardown
                pass

    async def __aenter__(self) -> "MarketDataClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def get_json(self, path: str, params: Optional[dict] = None,
                       timeout: Optional[float] = None) -> Any:
        """GET ``path`` and return the decoded JSON body.

        Raises :class:`MarketDataError` on timeout, transport failure or a
        non-2xx status, so callers never have to catch httpx exceptions.
        """
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            resp = await self.client.get(
                path, params=clean or None,
                timeout=httpx.Timeout(float(timeout)) if timeout else None,
            )
        except httpx.TimeoutException as e:
            raise MarketDataError(
                f"market data host timed out after {self.timeout:.0f}s ({path})") from e
        except httpx.HTTPError as e:
            raise MarketDataError(f"market data host unreachable: {e}") from e
        if resp.status_code != 200:
            body = resp.text[:200].replace("\n", " ")
            raise MarketDataError(f"HTTP {resp.status_code} from {path}: {body}")
        try:
            return resp.json()
        except ValueError as e:
            raise MarketDataError(f"invalid JSON from {path}") from e

    # ------------------------------------------------------------------
    # public market data endpoints (all mirror the spot REST API)
    # ------------------------------------------------------------------
    async def exchange_symbols(self) -> list[dict]:
        """``GET /api/v3/exchangeInfo`` → the raw ``symbols`` list."""
        payload = await self.get_json("/api/v3/exchangeInfo")
        symbols = (payload or {}).get("symbols")
        if not isinstance(symbols, list):
            raise MarketDataError("exchangeInfo response has no 'symbols' list")
        return symbols

    async def ticker24h(self, symbol: Optional[str] = None) -> Any:
        """``GET /api/v3/ticker/24hr``; list for all symbols, dict for one.

        The all-symbols call takes ~10s upstream — callers MUST cache it
        (see ``core/market_data/ttl_cache.py``).
        """
        params = {"symbol": symbol} if symbol else None
        return await self.get_json("/api/v3/ticker/24hr", params)

    async def klines(self, symbol: str, interval: str = "1h", limit: int = 500,
                     start_time: Optional[int] = None,
                     end_time: Optional[int] = None) -> list[list]:
        """``GET /api/v3/klines`` (raw rows, oldest first)."""
        limit = max(1, min(int(limit), KLINES_MAX_LIMIT))
        return await self.get_json("/api/v3/klines", {
            "symbol": symbol, "interval": interval, "limit": limit,
            "startTime": start_time, "endTime": end_time,
        })

    async def order_book(self, symbol: str, limit: int = 20) -> dict:
        return await self.get_json("/api/v3/depth", {"symbol": symbol, "limit": limit})

    async def recent_trades(self, symbol: str, limit: int = 30) -> list[dict]:
        return await self.get_json("/api/v3/trades", {"symbol": symbol, "limit": limit})

    async def symbol_price(self, symbol: str) -> dict:
        return await self.get_json("/api/v3/ticker/price", {"symbol": symbol})
