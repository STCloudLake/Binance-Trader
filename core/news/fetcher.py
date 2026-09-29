import httpx
import asyncio
import re
import ipaddress
import socket
from typing import Optional
from datetime import datetime
from urllib.parse import urlparse


# Private/restricted IP ranges that should never be targeted by outbound requests
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("224.0.0.0/4"),     # multicast
    ipaddress.ip_network("240.0.0.0/4"),     # reserved
]


def _is_safe_url(url_str: str) -> bool:
    """Validate that a URL does not target internal/private networks (SSRF protection)."""
    try:
        parsed = urlparse(url_str)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        # Resolve hostname to IP
        addr = ipaddress.ip_address(hostname)
        # Check against blocked networks
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                return False
        return True
    except ValueError:
        # Not an IP address — resolve via DNS
        try:
            resolved = socket.getaddrinfo(hostname, None)
            for _, _, _, _, sockaddr in resolved:
                ip = ipaddress.ip_address(sockaddr[0])
                for net in _BLOCKED_NETWORKS:
                    if ip in net:
                        return False
        except Exception:
            return False
        return True


#: Shared, disk-only coin universe used to turn a pair into its base asset.
#: ``NewsAnalyzer`` builds ``NewsFetcher()`` without arguments, so the lookup
#: resolves ``BTCUSDT → BTC`` from the same ``symbols.json`` the rest of the app
#: uses.  It never performs a network call: an unavailable universe simply means
#: the raw symbol is used.
_SHARED_UNIVERSE = None
_SHARED_UNIVERSE_LOADED = False


def _shared_universe():
    """Lazily preload the app-wide coin universe (disk only)."""
    global _SHARED_UNIVERSE, _SHARED_UNIVERSE_LOADED
    if _SHARED_UNIVERSE_LOADED:
        return _SHARED_UNIVERSE
    _SHARED_UNIVERSE_LOADED = True
    try:
        from app.config import Config
        from core.market_data.universe import Universe
        universe = Universe(Config())
        universe.preload_from_disk()
        _SHARED_UNIVERSE = universe if universe.get_symbols_cached_count() else None
    except Exception:
        _SHARED_UNIVERSE = None
    return _SHARED_UNIVERSE


class NewsFetcher:
    def __init__(self, universe=None):
        self._client: Optional[httpx.AsyncClient] = None
        #: Optional injected coin universe (``core.market_data.universe.Universe``).
        self._universe = universe

    def base_asset(self, symbol: str) -> str:
        """``SymbolInfo.base_asset`` for a pair (``BTCUSDT`` → ``BTC``).

        News APIs are queried by *asset*, not by exchange pair.  An unknown entry
        (or no universe at all) falls back to the raw symbol, so a query is never
        built from an empty or invented name.
        """
        raw = str(symbol or "").strip()
        if not raw:
            return raw
        universe = self._universe if self._universe is not None else _shared_universe()
        if universe is not None:
            try:
                info = universe.get(raw.upper())
            except Exception:
                info = None
            base = getattr(info, "base_asset", None) if info is not None else None
            if base:
                return str(base).upper()
        return raw

    async def start(self):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    async def fetch_from_source(self, source: dict, symbol: str, max_articles: int = 10) -> list[dict]:
        if self._client is None:
            return []

        articles = []
        try:
            if source.get("type") == "api":
                endpoint = source.get("endpoint", "")
                if endpoint:
                    # Query by base asset ("BTC"), not by the exchange pair
                    # ("BTCUSDT") — the news providers do not know the pair.
                    url = endpoint.replace("{symbol}", self.base_asset(symbol)).replace(
                        "{limit}", str(max_articles))
                    if not _is_safe_url(url):
                        from loguru import logger
                        logger.warning(f"Blocked unsafe URL: {url}")
                        return []
                    response = await self._client.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        articles = self._parse_api_response(data, source.get("name", ""))
            elif source.get("type") == "rss":
                from loguru import logger
                logger.warning(f"RSS source type not yet implemented: {source.get('name', 'unknown')}")
        except Exception as e:
            from loguru import logger
            logger.debug(f"News fetch error from {source.get('name', 'unknown')}: {e}")

        return articles[:max_articles]

    def _parse_api_response(self, data, source_name: str) -> list[dict]:
        articles = []
        if isinstance(data, list):
            items = data
        else:
            items = data.get("articles") or data.get("results") or []
        if items is None:
            items = []
        for item in items:
            if isinstance(item, dict):
                articles.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content_summary": item.get("description", item.get("summary", "")),
                    "published_at": item.get("publishedAt", item.get("published_at", datetime.now().isoformat())),
                })
        return articles

    async def close(self):
        if self._client:
            await self._client.aclose()
