import aiosqlite
from typing import Optional
from db.database import get_db


class NewsSourceManager:
    async def get_all_enabled(self) -> list[dict]:
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT * FROM news_sources WHERE enabled = 1 ORDER BY priority DESC"
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            await db.close()

    async def add_source(self, name: str, source_type: str, endpoint: str = None,
                         rate_limit: int = 10, priority: int = 5) -> int:
        db = await get_db()
        try:
            cursor = await db.execute(
                "INSERT INTO news_sources (name, type, endpoint, rate_limit, priority) VALUES (?,?,?,?,?)",
                (name, source_type, endpoint, rate_limit, priority)
            )
            await db.commit()
            return cursor.lastrowid
        finally:
            await db.close()

    async def update_source(self, source_id: int, **kwargs):
        db = await get_db()
        try:
            updates = ", ".join(f"{k} = ?" for k in kwargs)
            values = list(kwargs.values()) + [source_id]
            await db.execute(f"UPDATE news_sources SET {updates} WHERE id = ?", values)
            await db.commit()
        finally:
            await db.close()

    async def delete_source(self, source_id: int):
        db = await get_db()
        try:
            await db.execute("DELETE FROM news_sources WHERE id = ?", (source_id,))
            await db.commit()
        finally:
            await db.close()

    async def get_all(self) -> list[dict]:
        db = await get_db()
        try:
            cursor = await db.execute("SELECT * FROM news_sources ORDER BY priority DESC")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            await db.close()
