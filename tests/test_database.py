import pytest
import aiosqlite
import tempfile
import os
from pathlib import Path


@pytest.mark.asyncio
async def test_init_database_creates_tables():
    from db.database import init_database
    db_path = tempfile.mktemp(suffix=".db")
    await init_database(db_path)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in await cursor.fetchall()]
    os.unlink(db_path)
    assert "trades" in tables
    assert "orders" in tables
    assert "positions" in tables
    assert "alerts" in tables
    assert "ai_suggestions" in tables
    assert "news_sources" in tables
    assert "news_articles" in tables
    assert "ml_models" in tables
    assert "system_config" in tables


@pytest.mark.asyncio
async def test_insert_and_query_trade():
    from db.database import init_database
    db_path = tempfile.mktemp(suffix=".db")
    await init_database(db_path)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute(
            "INSERT INTO trades (symbol, side, entry_price, quantity, strategy, timeframe) VALUES (?,?,?,?,?,?)",
            ("BTCUSDT", "long", 50000.0, 0.01, "rsi_macd", "1h")
        )
        await db.commit()
        cursor = await db.execute("SELECT * FROM trades WHERE symbol='BTCUSDT'")
        row = await cursor.fetchone()
    os.unlink(db_path)
    assert row["symbol"] == "BTCUSDT"
    assert row["entry_price"] == 50000.0
    assert row["status"] == "open"
