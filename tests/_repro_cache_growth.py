"""Scratch measurement: cache/lock growth before vs after the bounds (defect 3)."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def ttl_cache_growth():
    from core.market_data.ttl_cache import TTLCache

    cache = TTLCache()

    async def _boom():
        raise RuntimeError("upstream down")

    for i in range(200):
        await cache.get(("key", i), 30.0, _boom)
    print(f"RESULT ttl 200 distinct failed keys -> _values={len(cache._values)} "
          f"_locks={len(cache._locks)}")


async def screener_growth():
    from core.market_data.screener import TokenScreener

    class _Boom:
        async def __call__(self, path, params):
            raise RuntimeError("upstream down")

    screener = TokenScreener(fetch=_Boom(), attempts=1, cache_ttl=300.0)
    for i in range(50):
        await screener.detail(f"S{i}USDT")
    print(f"RESULT screener 50 distinct detail keys -> _cache={len(screener._cache)} "
          f"_locks={len(screener._locks)}")


asyncio.run(ttl_cache_growth())
asyncio.run(screener_growth())
