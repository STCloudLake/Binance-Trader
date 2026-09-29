import os
import tempfile

import aiosqlite
import pytest


@pytest.mark.asyncio
async def test_init_database_creates_tables():
    from db.database import init_database
    db_path = tempfile.mktemp(suffix=".db")
    await init_database(db_path)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in await cursor.fetchall()]
    os.unlink(db_path)
    # Every table the application still uses (schema v4 — see tests/test_db_v4.py).
    for expected in ("trades", "positions", "pending_orders", "alerts",
                     "ai_suggestions", "news_sources", "system_config", "users",
                     "backtest_records", "strategy_lifecycle_events"):
        assert expected in tables, f"missing table {expected}"
    # SCHEMA-CHANGE NOTE (v4): `orders`, `risk_events`, `ml_models`,
    # `news_articles` and the orphan `test_tz` had zero writers and zero readers
    # and are no longer created (migration v4 drops them from an existing DB).
    # `news_sources` stays: core/news/fetcher.py reads it.
    for dropped in ("orders", "risk_events", "ml_models", "news_articles", "test_tz"):
        assert dropped not in tables, f"{dropped} should no longer be created"


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
