import asyncio
import aiosqlite
from pathlib import Path
from loguru import logger

DB_PATH: str = ""
_balance_lock = asyncio.Lock()
DEFAULT_BALANCE = 10000.0


async def init_database(db_path: str):
    global DB_PATH
    DB_PATH = db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()
        # Safe migration for new columns
        for col, default in [
            ("trader", "'manual'"),
            ("strategy_name", "''"),
            ("action", "'open'"),
            ("reduce_pct", "0"),
            ("trade_group", "''"),
        ]:
            try:
                await db.execute(f"ALTER TABLE trades ADD COLUMN {col} TEXT DEFAULT {default}")
                await db.commit()
            except Exception:
                pass



async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL,
    pnl REAL DEFAULT 0,
    pnl_pct REAL DEFAULT 0,
    strategy TEXT NOT NULL DEFAULT 'manual',
    timeframe TEXT NOT NULL DEFAULT '1h',
    position_type TEXT DEFAULT 'satellite' CHECK(position_type IN ('core', 'satellite')),
    trader TEXT DEFAULT 'manual' CHECK(trader IN ('manual', 'ai')),
    strategy_name TEXT DEFAULT '',
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    status TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed', 'cancelled'))
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER REFERENCES trades(id),
    symbol TEXT NOT NULL,
    order_type TEXT NOT NULL CHECK(order_type IN ('market', 'limit', 'stop_loss', 'trailing_stop')),
    side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
    price REAL,
    quantity REAL NOT NULL,
    filled_qty REAL DEFAULT 0,
    binance_order_id TEXT,
    status TEXT DEFAULT 'created' CHECK(status IN ('created','submitted','partially_filled','filled','cancelled','expired','rejected')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL DEFAULT 0,
    stop_loss REAL,
    take_profit_1 REAL,
    take_profit_2 REAL,
    take_profit_3 REAL,
    position_type TEXT DEFAULT 'satellite' CHECK(position_type IN ('core', 'satellite')),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL CHECK(level IN ('critical', 'warning', 'info')),
    type TEXT NOT NULL,
    message TEXT NOT NULL,
    symbol TEXT,
    acknowledged INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL CHECK(category IN ('coin_selection','strategy_optimization','risk_adjustment','market_assessment','news_analysis')),
    content TEXT NOT NULL,
    rationale TEXT,
    confidence REAL DEFAULT 0.5,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending','approved','rejected','applied')),
    applied_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    level TEXT NOT NULL CHECK(level IN ('critical', 'warning', 'info')),
    detail TEXT,
    triggered_by TEXT,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS news_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('api', 'rss', 'web')),
    endpoint TEXT,
    api_key_encrypted TEXT,
    enabled INTEGER DEFAULT 1,
    rate_limit INTEGER DEFAULT 10,
    priority INTEGER DEFAULT 5
);

CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER REFERENCES news_sources(id),
    symbol TEXT,
    title TEXT NOT NULL,
    url TEXT,
    content_summary TEXT,
    sentiment_score REAL,
    impact_level INTEGER DEFAULT 1 CHECK(impact_level BETWEEN 1 AND 5),
    published_at TIMESTAMP,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    model_type TEXT NOT NULL CHECK(model_type IN ('binary', 'multiclass', 'regression')),
    file_path TEXT NOT NULL,
    accuracy REAL,
    f1_score REAL,
    features TEXT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS system_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades(opened_at);
CREATE INDEX IF NOT EXISTS idx_orders_trade ON orders(trade_id);
CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(level);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_articles(symbol);
CREATE INDEX IF NOT EXISTS idx_news_fetched ON news_articles(fetched_at);
CREATE INDEX IF NOT EXISTS idx_ai_status ON ai_suggestions(status);
"""


async def load_sim_balance(db_path: str = None) -> float:
    """Load sim balance from system_config, or return default."""
    path = db_path or DB_PATH
    if not path:
        return DEFAULT_BALANCE
    try:
        db = await aiosqlite.connect(path)
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT value FROM system_config WHERE key='sim_balance'")
        row = await cursor.fetchone()
        await db.close()
        if row:
            return float(row["value"])
    except Exception as e:
        logger.warning(f"Failed to load sim balance: {e}")
    return DEFAULT_BALANCE


async def save_sim_balance(balance: float, db_path: str = None):
    """Persist sim balance to system_config."""
    path = db_path or DB_PATH
    if not path:
        return
    try:
        db = await aiosqlite.connect(path)
        await db.execute(
            "INSERT OR REPLACE INTO system_config (key, value, category) VALUES ('sim_balance', ?, 'trading')",
            (str(balance),))
        await db.commit()
        await db.close()
    except Exception as e:
        logger.warning(f"Failed to save sim balance: {e}")


async def atomic_adjust_balance(delta: float, db_path: str = None) -> float:
    """Atomically adjust sim balance by delta and return the new value.
    Uses asyncio.Lock to prevent concurrent read-modify-write races."""
    async with _balance_lock:
        current = await load_sim_balance(db_path)
        new_balance = current + delta
        await save_sim_balance(new_balance, db_path)
        return new_balance
