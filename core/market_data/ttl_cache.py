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

Both maps are **bounded**.  The cache is keyed by caller-controlled values (a
symbol path parameter, a watchlist entry), so an unbounded map is a memory leak
an authenticated caller can drive: 200 failed fetches for 200 distinct keys used
to leave ``_values=0, _locks=200`` behind for ever.  Values are evicted
least-recently-used and the per-key locks are dropped as soon as their fetch
finishes.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
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

#: Upper bound on cached values (LRU eviction beyond this) and on the per-key
#: in-flight lock map.  The real caller space is a few hundred symbols × a handful
#: of endpoints, so these are far above any legitimate working set.
MAX_VALUES = 512
MAX_LOCKS = 256


class TTLCache:
    """Per-key TTL cache, bounded in both maps.

    ``get(key, ttl, fetcher)`` returns ``(value, error)``: exactly one of the
    two is non-``None``.  Concurrent callers for the same key share one fetch.
    """

    def __init__(self, max_values: int = MAX_VALUES,
                 max_locks: int = MAX_LOCKS) -> None:
        self._values: "OrderedDict[Hashable, tuple[float, Any]]" = OrderedDict()
        self._locks: dict[Hashable, asyncio.Lock] = {}
        self._max_values = max(1, int(max_values))
        self._max_locks = max(1, int(max_locks))

    def put(self, key: Hashable, value: Any) -> None:
        self._values[key] = (time.monotonic(), value)
        self._values.move_to_end(key)
        self._evict_values()

    def _evict_values(self) -> None:
        """Drop least-recently-used entries until the map is inside its bound."""
        while len(self._values) > self._max_values:
            self._values.popitem(last=False)

    def _prune_locks(self) -> None:
        """Bound the in-flight lock map.

        A lock whose fetch has finished is dropped by :meth:`get`; this is the
        backstop for the pathological case (a caller that dies between creating
        the lock and awaiting it, or a burst wider than the map).  Only *free*
        locks are dropped, so an in-flight fetch keeps its de-duplication.
        """
        if len(self._locks) <= self._max_locks:
            return
        for key in [k for k, lock in self._locks.items() if not lock.locked()]:
            self._locks.pop(key, None)
            if len(self._locks) <= self._max_locks:
                return
        # Still over the bound and everything is busy: drop the oldest free-ish
        # entries anyway rather than let a caller-controlled key space grow.
        while len(self._locks) > self._max_locks:
            self._locks.pop(next(iter(self._locks)), None)

    async def get(self, key: Hashable, ttl: float,
                  fetcher: Callable[[], Awaitable[Any]]) -> tuple[Any, Optional[str]]:
        now = time.monotonic()
        hit = self._values.get(key)
        if hit is not None and (now - hit[0]) < ttl:
            self._values.move_to_end(key)
            return hit[1], None

        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
            self._prune_locks()
        try:
            async with lock:
                # Another request may have refreshed while we waited for the lock.
                hit = self._values.get(key)
                if hit is not None and (time.monotonic() - hit[0]) < ttl:
                    self._values.move_to_end(key)
                    return hit[1], None
                try:
                    value = await fetcher()
                except Exception as e:  # noqa: BLE001 - never leak a traceback to the client
                    logger.warning(f"Market data fetch failed ({key}): {e}")
                    return None, str(e)
                if value is None:
                    return None, f"empty response for {key}"
                self.put(key, value)
                return value, None
        finally:
            # The lock exists for the in-flight fetch only.  A failed fetch caches
            # nothing, so keeping its lock for ever was pure leak (200 failed
            # fetches → `_locks=200, _values=0`).  Dropping it is safe: a caller
            # that is *waiting* on this object still holds a reference, and the
            # de-dup contract is about the in-flight fetch that just ended.
            self._locks.pop(key, None)
