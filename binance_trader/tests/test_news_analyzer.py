import pytest
import tempfile
import os
from pathlib import Path


def test_fetcher_start_stop():
    import asyncio
    from core.news.fetcher import NewsFetcher

    async def run():
        f = NewsFetcher()
        await f.start()
        articles = await f.fetch_from_source({"type": "api", "endpoint": ""}, "BTC", 5)
        assert isinstance(articles, list)
        await f.close()

    asyncio.run(run())


def test_source_manager_crud():
    import asyncio
    from db.database import init_database
    from core.news.source_manager import NewsSourceManager

    async def run():
        db_path = tempfile.mktemp(suffix=".db")
        await init_database(db_path)

        mgr = NewsSourceManager()
        sid = await mgr.add_source("TestNewsAPI", "api", "https://test.api/news", 10, 5)
        assert sid > 0

        sources = await mgr.get_all()
        assert len(sources) >= 1

        await mgr.update_source(sid, enabled=0)
        enabled = await mgr.get_all_enabled()
        assert all(s["id"] != sid for s in enabled)

        await mgr.delete_source(sid)

        os.unlink(db_path)

    asyncio.run(run())


def test_impact_classification():
    from core.news.analyzer import NewsAnalyzer
    assert NewsAnalyzer._classify_impact(0.8) == 5
    assert NewsAnalyzer._classify_impact(0.6) == 4
    assert NewsAnalyzer._classify_impact(0.4) == 3
    assert NewsAnalyzer._classify_impact(0.2) == 2
    assert NewsAnalyzer._classify_impact(0.05) == 1
