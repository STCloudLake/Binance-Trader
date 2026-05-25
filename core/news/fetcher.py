import httpx
import asyncio
from typing import Optional
from datetime import datetime


class NewsFetcher:
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

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
                    url = endpoint.replace("{symbol}", symbol).replace("{limit}", str(max_articles))
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
