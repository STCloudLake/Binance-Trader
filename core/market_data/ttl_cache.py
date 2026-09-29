"""Tiny asyncio TTL cache with in-flight de-duplication.

Two upstream facts drive this (``docs/overhaul/MARKET_PAGES_API.md``):

* ``/api/v3/ticker/24hr`` for **all** symbols takes ~10s on the data host.  A
  page that polls it every few seconds would hammer the exchange and pile up
  connections, so it is cached for 60s (:data:`TICKER24H_TTL`).
* Every route in this app is served by a single event loop, so a burst of
  concurrent requests for the same key must collapse into **one** upstream call
  rather than N.  A per-key lock does that.

Errors are *never* cached: a transient outage must not be remembered for the
whole TTL window.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Hashable, Optional

from loguru import logger

#: Contract §1 cache windows (seconds).
TICKER24H_TTL = 60.0
#: Kept for the ``web.routes.market`` import surface (it still imports this name
#: and owns its own cleanup); the symbol-list route now caches under
#: :data:`TICKER24H_TTL`.
SYMBOLS_TTL = 60.0
COIN_TTL = 30.0
DEPTH_TTL = 2.0
TRADES_TTL = 2.0

#: exchangeInfo changes rarely; the contract asks for ~6h.
EXCHANGE_INFO_TTL = 6 * 3600.0


class TTLCache:
    """Per-key TTL cache.

    ``get(key, ttl, fetcher)`` returns ``(value, error)``: exactly one of the
    two is non-``None``.  Concurrent callers for the same key share one fetch.
    """

    def __init__(self) -> None:
        self._values: dict[Hashable, tuple[float, Any]] = {}
        self._locks: dict[Hashable, asyncio.Lock] = {}

    def put(self, key: Hashable, value: Any) -> None:
        self._values[key] = (time.monotonic(), value)

    async def get(self, key: Hashable, ttl: float,
                  fetcher: Callable[[], Awaitable[Any]]) -> tuple[Any, Optional[str]]:
        now = time.monotonic()
        hit = self._values.get(key)
        if hit is not None and (now - hit[0]) < ttl:
            return hit[1], None

        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        async with lock:
            # Another request may have refreshed while we waited for the lock.
            hit = self._values.get(key)
            if hit is not None and (time.monotonic() - hit[0]) < ttl:
                return hit[1], None
            try:
                value = await fetcher()
            except Exception as e:  # noqa: BLE001 - never leak a traceback to the client
                logger.warning(f"Market data fetch failed ({key}): {e}")
                return None, str(e)
            if value is None:
                return None, f"empty response for {key}"
            self._values[key] = (time.monotonic(), value)
            return value, None
