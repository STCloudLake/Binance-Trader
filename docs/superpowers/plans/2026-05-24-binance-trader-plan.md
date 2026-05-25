# Binance Trader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-featured automated trading system for Binance with ML prediction, news analysis, DeepSeek AI integration, and Web UI.

**Architecture:** Event-driven modular design with 9 core components communicating via asyncio EventBus. SQLite + Parquet persistence, FastAPI Web UI with HTMX+Alpine.js frontend.

**Tech Stack:** Python 3.11+, python-binance, asyncio, TA-Lib, scikit-learn, XGBoost, FastAPI, SQLite, Parquet, DeepSeek API

---

## Phase 1: Project Foundation

### Task 1.1: Project scaffolding & dependencies

**Files:**
- Create: `binance_trader/requirements.txt`
- Create: `binance_trader/setup.py`

- [ ] **Step 1: Create requirements.txt**

```python
python-binance>=1.0.19
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0
TA-Lib>=0.4.28
scikit-learn>=1.3.0
xgboost>=1.7.0
aiosqlite>=0.19.0
SQLAlchemy>=2.0.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
jinja2>=3.1.0
pydantic>=2.0.0
pyyaml>=6.0
loguru>=0.7.0
openai>=1.0.0
httpx>=0.24.0
pyarrow>=12.0.0
```

- [ ] **Step 2: Create setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="binance_trader",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=open("requirements.txt").read().splitlines(),
)
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

- [ ] **Step 4: Verify TA-Lib installation**

Run: `python -c "import talib; print(talib.__version__)"`
Expected: Version number printed

### Task 1.2: Configuration system

**Files:**
- Create: `binance_trader/app/__init__.py`
- Create: `binance_trader/app/config.py`
- Create: `binance_trader/config/config.yaml`
- Create: `binance_trader/config/risk_params.yaml`
- Create: `binance_trader/config/secrets.yaml.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Create default config YAML**

`binance_trader/config/config.yaml`:
```yaml
mode: sim
web_port: 8899

binance:
  testnet: true

trading:
  spot_enabled: true
  futures_enabled: false

signal_weights:
  indicator: 0.5
  ml: 0.3
  news: 0.2

core_position:
  max_symbols: 5
  capital_pct: 0.7

satellite_position:
  max_symbols: 10
  capital_pct: 0.3

news:
  fetch_interval_minutes: 30
  max_articles_per_symbol: 10
  anomaly_threshold_pct: 3.0
  volume_spike_multiplier: 3.0

ai:
  mode: semi_auto  # suggest / semi_auto / full_auto
  model: deepseek-chat
  base_url: https://api.deepseek.com
```

- [ ] **Step 2: Create risk params YAML**

`binance_trader/config/risk_params.yaml`:
```yaml
hard_limits:
  max_daily_drawdown_pct: 5.0
  max_weekly_drawdown_pct: 10.0
  max_daily_loss_usdt: 500.0
  max_position_size_pct: 10.0
  max_leverage: 3
  min_stop_loss_distance_pct: 0.5
  max_open_trades: 8
  max_total_exposure_pct: 80.0
  max_consecutive_losses: 5

soft_params:
  risk_appetite: balanced  # conservative / balanced / aggressive
  position_size_pct: 5.0
  stop_loss_pct: 2.0
  take_profit_1_pct: 3.0
  take_profit_2_pct: 5.0
  take_profit_3_pct: 10.0
  leverage: 2
```

- [ ] **Step 3: Create secrets example**

`binance_trader/config/secrets.yaml.example`:
```yaml
binance:
  api_key: "your_binance_api_key"
  api_secret: "your_binance_api_secret"

deepseek:
  api_key: "your_deepseek_api_key"
```

- [ ] **Step 4: Implement Config class**

`binance_trader/app/config.py`:
```python
import os
import yaml
from pathlib import Path
from typing import Any
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent


class HardRiskLimits(BaseModel):
    max_daily_drawdown_pct: float = 5.0
    max_weekly_drawdown_pct: float = 10.0
    max_daily_loss_usdt: float = 500.0
    max_position_size_pct: float = 10.0
    max_leverage: int = 3
    min_stop_loss_distance_pct: float = 0.5
    max_open_trades: int = 8
    max_total_exposure_pct: float = 80.0
    max_consecutive_losses: int = 5


class SoftRiskParams(BaseModel):
    risk_appetite: str = "balanced"
    position_size_pct: float = 5.0
    stop_loss_pct: float = 2.0
    take_profit_1_pct: float = 3.0
    take_profit_2_pct: float = 5.0
    take_profit_3_pct: float = 10.0
    leverage: int = 2


class SignalWeights(BaseModel):
    indicator: float = 0.5
    ml: float = 0.3
    news: float = 0.2


class Config:
    _instance = None

    def __new__(cls, mode: str = "sim"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    @classmethod
    def load(cls, mode: str = "sim") -> "Config":
        inst = cls(mode)
        if not inst._loaded:
            inst._load(mode)
        return inst

    def _load(self, mode: str):
        self.mode = mode
        self._load_yaml("config/config.yaml")
        self._load_yaml("config/risk_params.yaml")

        secrets_path = PROJECT_ROOT / "config" / "secrets.yaml"
        if secrets_path.exists():
            self._load_yaml("config/secrets.yaml")

        self.binance_api_key = os.getenv("BINANCE_API_KEY", self._get("binance", {}).get("api_key", ""))
        self.binance_api_secret = os.getenv("BINANCE_API_SECRET", self._get("binance", {}).get("api_secret", ""))
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", self._get("deepseek", {}).get("api_key", ""))

        self.web_port = self._get("web_port", 8899)
        self.binance_testnet = self._get("binance", {}).get("testnet", True)
        self.spot_enabled = self._get("trading", {}).get("spot_enabled", True)
        self.futures_enabled = self._get("trading", {}).get("futures_enabled", False)

        sw = self._get("signal_weights", {})
        self.signal_weights = SignalWeights(**sw)

        self.core_max_symbols = self._get("core_position", {}).get("max_symbols", 5)
        self.core_capital_pct = self._get("core_position", {}).get("capital_pct", 0.7)
        self.satellite_max_symbols = self._get("satellite_position", {}).get("max_symbols", 10)
        self.satellite_capital_pct = self._get("satellite_position", {}).get("capital_pct", 0.3)

        self.news_fetch_interval = self._get("news", {}).get("fetch_interval_minutes", 30)
        self.news_max_articles = self._get("news", {}).get("max_articles_per_symbol", 10)
        self.anomaly_threshold_pct = self._get("news", {}).get("anomaly_threshold_pct", 3.0)
        self.volume_spike_multiplier = self._get("news", {}).get("volume_spike_multiplier", 3.0)

        self.ai_mode = self._get("ai", {}).get("mode", "semi_auto")
        self.ai_model = self._get("ai", {}).get("model", "deepseek-chat")
        self.ai_base_url = self._get("ai", {}).get("base_url", "https://api.deepseek.com")

        rp = self._get("hard_limits", {}) or self._get("risk_params", {}).get("hard_limits", {})
        self.hard_limits = HardRiskLimits(**rp) if rp else HardRiskLimits()

        sp = self._get("soft_params", {}) or self._get("risk_params", {}).get("soft_params", {})
        self.soft_params = SoftRiskParams(**sp) if sp else SoftRiskParams()

        self.db_path = str(PROJECT_ROOT / "data" / "binance_trader.db")
        self.data_dir = str(PROJECT_ROOT / "data")
        self.strategies_dir = str(PROJECT_ROOT / "strategies")

        self._loaded = True

    def _load_yaml(self, relative_path: str):
        path = PROJECT_ROOT / relative_path
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
                self._data = {**getattr(self, "_data", {}), **data}

    def _get(self, *keys, default=None) -> Any:
        d = getattr(self, "_data", {})
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, {})
            else:
                return default
        return d if d != {} else default

    def update_soft_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.soft_params, k):
                setattr(self.soft_params, k, v)

    def update_signal_weights(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.signal_weights, k):
                setattr(self.signal_weights, k, v)
```

- [ ] **Step 5: Write config tests**

`tests/test_config.py`:
```python
import pytest
import os
import tempfile
from pathlib import Path


def test_config_loads_with_defaults():
    from app.config import Config
    Config._instance = None
    config = Config.load("sim")
    assert config.mode == "sim"
    assert config.web_port == 8899
    assert config.hard_limits.max_leverage == 3
    assert config.soft_params.risk_appetite == "balanced"
    assert config.signal_weights.indicator == 0.5


def test_config_env_override():
    from app.config import Config
    Config._instance = None
    os.environ["DEEPSEEK_API_KEY"] = "test_key_123"
    config = Config.load("sim")
    assert config.deepseek_api_key == "test_key_123"
    del os.environ["DEEPSEEK_API_KEY"]


def test_config_singleton():
    from app.config import Config
    Config._instance = None
    c1 = Config.load("sim")
    c2 = Config.load("sim")
    assert c1 is c2
```

- [ ] **Step 6: Run config tests**

Run: `cd binance_trader && python -m pytest tests/test_config.py -v`
Expected: 3 PASS

### Task 1.3: Database layer

**Files:**
- Create: `binance_trader/db/__init__.py`
- Create: `binance_trader/db/database.py`
- Create: `binance_trader/db/models.py`
- Test: `tests/test_database.py`

- [ ] **Step 1: Implement database init**

`binance_trader/db/database.py`:
```python
import aiosqlite
from pathlib import Path

DB_PATH: str = ""


async def init_database(db_path: str):
    global DB_PATH
    DB_PATH = db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()


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
    strategy TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    position_type TEXT DEFAULT 'satellite' CHECK(position_type IN ('core', 'satellite')),
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
```

- [ ] **Step 2: Write database test**

`tests/test_database.py`:
```python
import pytest
import aiosqlite
import tempfile
import os
from pathlib import Path


@pytest.mark.asyncio
async def test_init_database_creates_tables():
    from db.database import init_database, get_db
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
    import aiosqlite
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
```

- [ ] **Step 3: Run database tests**

Run: `cd binance_trader && python -m pytest tests/test_database.py -v`
Expected: 2 PASS

### Task 1.4: Event bus

**Files:**
- Create: `binance_trader/app/event_bus.py`
- Test: `tests/test_event_bus.py`

- [ ] **Step 1: Implement EventBus**

`binance_trader/app/event_bus.py`:
```python
import asyncio
from typing import Any, Callable, Awaitable
from collections import defaultdict
from enum import Enum


class EventType(str, Enum):
    MARKET_TICK = "market.tick"
    MARKET_KLINE = "market.kline"
    STRATEGY_SIGNAL = "strategy.signal"
    ML_PREDICTION = "ml.prediction"
    NEWS_UPDATE = "news.update"
    NEWS_ALERT = "news.alert"
    RISK_CHECK = "risk.check"
    RISK_BREACH = "risk.breach"
    ORDER_REQUEST = "order.request"
    ORDER_UPDATE = "order.update"
    POSITION_UPDATE = "position.update"
    ALERT_TRIGGER = "alert.trigger"
    AI_SUGGESTION = "ai.suggestion"
    AI_MARKET_STATE = "ai.market_state"
    SYSTEM_SHUTDOWN = "system.shutdown"


class Event:
    def __init__(self, event_type: EventType, data: dict[str, Any] = None):
        self.type = event_type
        self.data = data or {}

    def __repr__(self):
        return f"Event({self.type.value}, data={self.data})"


class EventBus:
    def __init__(self):
        self._subscribers: dict[EventType, list[Callable[[Event], Awaitable[None]]]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=10000)
        self._running = False
        self._task: asyncio.Task | None = None

    def subscribe(self, event_type: EventType, callback: Callable[[Event], Awaitable[None]]):
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], Awaitable[None]]):
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    async def publish(self, event: Event):
        await self._queue.put(event)

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._process())

    async def _process(self):
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                subscribers = self._subscribers.get(event.type, [])
                tasks = [cb(event) for cb in subscribers]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.TimeoutError:
                continue
            except Exception:
                pass

    async def shutdown(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
```

- [ ] **Step 2: Write event bus test**

`tests/test_event_bus.py`:
```python
import pytest
import asyncio
from app.event_bus import EventBus, Event, EventType


@pytest.mark.asyncio
async def test_publish_and_subscribe():
    bus = EventBus()
    received = []

    async def handler(event: Event):
        received.append(event.data["value"])

    bus.subscribe(EventType.MARKET_TICK, handler)
    await bus.start()
    await bus.publish(Event(EventType.MARKET_TICK, {"value": 42}))
    await asyncio.sleep(0.2)
    await bus.shutdown()

    assert 42 in received


@pytest.mark.asyncio
async def test_multiple_subscribers():
    bus = EventBus()
    results = []

    async def handler1(event): results.append(("h1", event.data["x"]))
    async def handler2(event): results.append(("h2", event.data["x"]))

    bus.subscribe(EventType.STRATEGY_SIGNAL, handler1)
    bus.subscribe(EventType.STRATEGY_SIGNAL, handler2)
    await bus.start()
    await bus.publish(Event(EventType.STRATEGY_SIGNAL, {"x": 10}))
    await asyncio.sleep(0.2)
    await bus.shutdown()

    assert ("h1", 10) in results
    assert ("h2", 10) in results


@pytest.mark.asyncio
async def test_unsubscribe():
    bus = EventBus()
    received = []

    async def handler(event): received.append(1)

    bus.subscribe(EventType.MARKET_TICK, handler)
    bus.unsubscribe(EventType.MARKET_TICK, handler)
    await bus.start()
    await bus.publish(Event(EventType.MARKET_TICK, {}))
    await asyncio.sleep(0.2)
    await bus.shutdown()

    assert len(received) == 0
```

- [ ] **Step 3: Run event bus tests**

Run: `cd binance_trader && python -m pytest tests/test_event_bus.py -v`
Expected: 3 PASS

---

## Phase 2: Market Data Pipeline

### Task 2.1: Market data provider

**Files:**
- Create: `binance_trader/core/__init__.py`
- Create: `binance_trader/core/market_data/__init__.py`
- Create: `binance_trader/core/market_data/provider.py`
- Create: `binance_trader/core/market_data/ohlcv_cache.py`
- Create: `binance_trader/core/market_data/ws_client.py`
- Test: `tests/test_market_data.py`

- [ ] **Step 1: Implement OHLCV cache**

`binance_trader/core/market_data/ohlcv_cache.py`:
```python
import pandas as pd
from pathlib import Path
from collections import defaultdict


class OHLVCache:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._cache: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)

    def _key(self, symbol: str, interval: str) -> str:
        return f"{symbol}_{interval}"

    def _path(self, symbol: str, interval: str) -> Path:
        symbol_dir = self.data_dir / "market" / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{interval}.parquet"

    def get(self, symbol: str, interval: str) -> pd.DataFrame | None:
        cache_key = self._key(symbol, interval)
        if symbol in self._cache and interval in self._cache[symbol]:
            return self._cache[symbol][interval]

        path = self._path(symbol, interval)
        if path.exists():
            df = pd.read_parquet(path)
            self._cache[symbol][interval] = df
            return df
        return None

    def update(self, symbol: str, interval: str, df: pd.DataFrame):
        self._cache[symbol][interval] = df

    def append_candle(self, symbol: str, interval: str, candle: dict):
        path = self._path(symbol, interval)
        new_row = pd.DataFrame([candle])
        new_row["close_time"] = pd.to_datetime(new_row["close_time"], unit="ms")
        new_row.set_index("close_time", inplace=True)

        existing = self.get(symbol, interval)
        if existing is not None:
            combined = pd.concat([existing, new_row])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        else:
            combined = new_row

        combined.to_parquet(path)
        self._cache[symbol][interval] = combined

    def save(self, symbol: str, interval: str):
        if symbol in self._cache and interval in self._cache[symbol]:
            path = self._path(symbol, interval)
            self._cache[symbol][interval].to_parquet(path)
```

- [ ] **Step 2: Implement MarketDataProvider**

`binance_trader/core/market_data/provider.py`:
```python
import asyncio
import time
from typing import Optional
from binance import AsyncClient, BinanceSocketManager
import pandas as pd

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.ohlcv_cache import OHLVCache


class MarketDataProvider:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.cache = OHLVCache(config.data_dir)
        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._watched_symbols: list[str] = []
        self._intervals: list[str] = []
        self._price_cache: dict[str, float] = {}
        self._price_history: dict[str, list[tuple[float, float]]] = {}  # symbol -> [(timestamp, price), ...]

    async def start(self, symbols: list[str], intervals: list[str]):
        self._watched_symbols = symbols
        self._intervals = intervals
        self.client = await AsyncClient.create(
            api_key=self.config.binance_api_key or None,
            api_secret=self.config.binance_api_secret or None,
            testnet=self.config.binance_testnet,
        )
        self._running = True
        self._tasks.append(asyncio.create_task(self._run_websocket()))
        self._tasks.append(asyncio.create_task(self._run_price_tracker()))

    async def _run_websocket(self):
        self.bsm = BinanceSocketManager(self.client)
        streams = []
        for symbol in self._watched_symbols:
            sym_lower = symbol.lower()
            for interval in self._intervals:
                streams.append(f"{sym_lower}@kline_{interval}")

        conn_key = None
        while self._running:
            try:
                if conn_key is None:
                    conn_key = await self.bsm.multiplex(streams)
                    async with conn_key as stream:
                        while self._running:
                            msg = await asyncio.wait_for(stream.recv(), timeout=1.0)
                            await self._handle_ws_message(msg)
                else:
                    async with conn_key as stream:
                        while self._running:
                            msg = await asyncio.wait_for(stream.recv(), timeout=1.0)
                            await self._handle_ws_message(msg)
            except asyncio.TimeoutError:
                continue
            except Exception:
                conn_key = None
                await asyncio.sleep(5)

    async def _handle_ws_message(self, msg: dict):
        if "stream" not in msg:
            return
        data = msg["data"]
        if data["e"] != "kline":
            return
        kline = data["k"]
        if not kline["x"]:
            return

        symbol = kline["s"]
        interval = kline["i"]
        candle = {
            "close_time": kline["T"],
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
        }
        self.cache.append_candle(symbol, interval, candle)
        self._price_cache[symbol] = float(kline["c"])

        await self.event_bus.publish(Event(EventType.MARKET_KLINE, {
            "symbol": symbol,
            "interval": interval,
            "candle": candle,
        }))

    async def _run_price_tracker(self):
        while self._running:
            for symbol in self._watched_symbols:
                if symbol in self._price_cache:
                    price = self._price_cache[symbol]
                    if symbol not in self._price_history:
                        self._price_history[symbol] = []
                    self._price_history[symbol].append((time.time(), price))
                    cutoff = time.time() - 3600
                    self._price_history[symbol] = [
                        (t, p) for t, p in self._price_history[symbol] if t > cutoff
                    ]
            await asyncio.sleep(1)

    async def get_historical(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        cached = self.cache.get(symbol, interval)
        if cached is not None and len(cached) >= limit:
            return cached.tail(limit)

        klines = await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        df = df[["close_time", "open", "high", "low", "close", "volume"]].copy()
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.set_index("close_time", inplace=True)
        self.cache.update(symbol, interval, df)
        self.cache.save(symbol, interval)
        return df

    def get_current_price(self, symbol: str) -> float | None:
        return self._price_cache.get(symbol)

    def get_price_change_pct(self, symbol: str, minutes: int = 5) -> float | None:
        history = self._price_history.get(symbol, [])
        if len(history) < 2:
            return None
        cutoff = time.time() - minutes * 60
        old_prices = [p for t, p in history if t <= cutoff]
        if not old_prices:
            return None
        current = history[-1][1]
        old = old_prices[-1]
        return (current - old) / old * 100

    async def get_volume_ratio(self, symbol: str, interval: str = "1h") -> float | None:
        df = await self.get_historical(symbol, interval, limit=100)
        if df is None or len(df) < 20:
            return None
        recent_vol = df["volume"].iloc[-1]
        avg_vol = df["volume"].iloc[-20:-1].mean()
        if avg_vol == 0:
            return None
        return recent_vol / avg_vol

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self.client:
            await self.client.close_connection()
```

- [ ] **Step 3: Write market data test**

`tests/test_market_data.py`:
```python
import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd


def test_ohlcv_cache_append_and_get():
    from core.market_data.ohlcv_cache import OHLVCache
    data_dir = tempfile.mkdtemp()
    cache = OHLVCache(data_dir)

    cache.append_candle("BTCUSDT", "1h", {
        "close_time": 1700000000000,
        "open": 50000.0, "high": 51000.0, "low": 49000.0,
        "close": 50500.0, "volume": 100.0,
    })

    df = cache.get("BTCUSDT", "1h")
    assert df is not None
    assert len(df) == 1
    assert float(df.iloc[0]["close"]) == 50500.0

    path = Path(data_dir) / "market" / "BTCUSDT" / "1h.parquet"
    assert path.exists()


def test_ohlcv_cache_persistence():
    from core.market_data.ohlcv_cache import OHLVCache
    data_dir = tempfile.mkdtemp()
    cache1 = OHLVCache(data_dir)
    cache1.append_candle("ETHUSDT", "4h", {
        "close_time": 1700000000000,
        "open": 3000.0, "high": 3100.0, "low": 2900.0,
        "close": 3050.0, "volume": 50.0,
    })

    cache2 = OHLVCache(data_dir)
    df = cache2.get("ETHUSDT", "4h")
    assert df is not None
    assert len(df) == 1
```

- [ ] **Step 4: Run market data tests**

Run: `cd binance_trader && python -m pytest tests/test_market_data.py -v`
Expected: 2 PASS

---

## Phase 3: Strategy Engine & Risk Management

### Task 3.1: Strategy engine

**Files:**
- Create: `binance_trader/core/strategy/__init__.py`
- Create: `binance_trader/core/strategy/engine.py`
- Create: `binance_trader/core/strategy/loader.py`
- Create: `binance_trader/core/strategy/indicators.py`
- Create: `binance_trader/strategies/rsi_macd_trend.yaml`
- Test: `tests/test_strategy_engine.py`

- [ ] **Step 1: Implement strategy loader**

`binance_trader/core/strategy/loader.py`:
```python
import yaml
from pathlib import Path
from typing import Any
from pydantic import BaseModel


class IndicatorConfig(BaseModel):
    period: int = 14
    source: str = "close"
    fast: int | None = None
    slow: int | None = None
    signal: int | None = None
    stddev: int | None = None


class MLConfig(BaseModel):
    enabled: bool = False
    confidence_threshold: float = 0.6
    features: list[str] = []
    weight: float = 0.3


class StrategyConfig(BaseModel):
    name: str
    mode: str = "trend"
    timeframes: list[str] = ["1h"]
    indicators: dict[str, Any] = {}
    entry_conditions: dict[str, list[str]] = {}
    exit_conditions: dict[str, list[str]] = {}
    ml_config: MLConfig | None = None


class StrategyLoader:
    def __init__(self, strategies_dir: str):
        self.strategies_dir = Path(strategies_dir)

    def load(self, name: str) -> StrategyConfig:
        path = self.strategies_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return StrategyConfig(**data)

    def load_all(self) -> list[StrategyConfig]:
        strategies = []
        for path in self.strategies_dir.glob("*.yaml"):
            with open(path) as f:
                data = yaml.safe_load(f)
            strategies.append(StrategyConfig(**data))
        return strategies

    def save(self, config: StrategyConfig):
        path = self.strategies_dir / f"{config.name.lower().replace(' ', '_')}.yaml"
        with open(path, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, allow_unicode=True)

    def list_names(self) -> list[str]:
        return [p.stem for p in self.strategies_dir.glob("*.yaml")]
```

- [ ] **Step 2: Implement indicator functions**

`binance_trader/core/strategy/indicators.py`:
```python
import pandas as pd
import numpy as np
import talib


def compute_all(df: pd.DataFrame, indicator_configs: dict) -> pd.DataFrame:
    result = df.copy()
    for name, cfg in indicator_configs.items():
        period = cfg.get("period", 14)
        source_col = cfg.get("source", "close")
        source = result[source_col].values if source_col in result.columns else result["close"].values

        if name == "rsi":
            result["rsi"] = talib.RSI(source, timeperiod=period)
        elif name == "macd":
            fast = cfg.get("fast", 12)
            slow = cfg.get("slow", 26)
            sig = cfg.get("signal", 9)
            macd, macd_signal, macd_hist = talib.MACD(source, fastperiod=fast, slowperiod=slow, signalperiod=sig)
            result["macd"] = macd
            result["macd_signal"] = macd_signal
            result["macd_histogram"] = macd_hist
        elif name == "bollinger":
            period = cfg.get("period", 20)
            stddev = cfg.get("stddev", 2)
            upper, middle, lower = talib.BBANDS(source, timeperiod=period, nbdevup=stddev, nbdevdn=stddev)
            result["bollinger_upper"] = upper
            result["bollinger_middle"] = middle
            result["bollinger_lower"] = lower
            result["bollinger_width"] = (upper - lower) / middle
        elif name == "ema":
            result[f"ema_{period}"] = talib.EMA(source, timeperiod=period)
        elif name == "sma":
            result[f"sma_{period}"] = talib.SMA(source, timeperiod=period)
        elif name == "atr":
            result["atr"] = talib.ATR(result["high"].values, result["low"].values, result["close"].values, timeperiod=period)
        elif name == "adx":
            result["adx"] = talib.ADX(result["high"].values, result["low"].values, result["close"].values, timeperiod=period)
        elif name == "stoch":
            slowk, slowd = talib.STOCH(result["high"].values, result["low"].values, result["close"].values,
                                       fastk_period=period, slowk_period=3, slowd_period=3)
            result["stoch_k"] = slowk
            result["stoch_d"] = slowd
        elif name == "obv":
            result["obv"] = talib.OBV(result["close"].values, result["volume"].values)
        elif name == "cci":
            result["cci"] = talib.CCI(result["high"].values, result["low"].values, result["close"].values, timeperiod=period)

    result["volume_ratio"] = result["volume"] / result["volume"].rolling(20).mean()
    result["price_momentum_24h"] = result["close"].pct_change(periods=24)
    return result


def evaluate_condition(df: pd.DataFrame, condition: str) -> pd.Series:
    env = {col: df[col] for col in df.columns}
    env["sma"] = lambda series, period: series.rolling(period).mean()
    return pd.eval(condition, engine="python", local_dict=env)
```

- [ ] **Step 3: Implement strategy engine**

`binance_trader/core/strategy/engine.py`:
```python
import asyncio
import pandas as pd
from typing import Optional

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.provider import MarketDataProvider
from core.strategy.loader import StrategyLoader, StrategyConfig
from core.strategy.indicators import compute_all, evaluate_condition


class StrategyEngine:
    def __init__(self, config: Config, event_bus: EventBus, market_data: MarketDataProvider):
        self.config = config
        self.event_bus = event_bus
        self.market_data = market_data
        self.loader = StrategyLoader(config.strategies_dir)
        self._running = False
        self._strategies: dict[str, StrategyConfig] = {}
        self._signal_cache: dict[str, dict[str, float]] = {}
        self._indicators_cache: dict[str, pd.DataFrame] = {}
        self._ml_confidence: dict[str, float] = {}

    async def start(self):
        self._running = True
        self.event_bus.subscribe(EventType.MARKET_KLINE, self._on_kline)
        self.event_bus.subscribe(EventType.ML_PREDICTION, self._on_ml_prediction)
        self._strategies = {s.name: s for s in self.loader.load_all()}

    async def _on_kline(self, event: Event):
        if not self._running:
            return
        symbol = event.data["symbol"]
        interval = event.data["interval"]

        for name, strategy in self._strategies.items():
            if interval not in strategy.timeframes:
                continue
            await self._evaluate(symbol, interval, strategy)

    async def _on_ml_prediction(self, event: Event):
        self._ml_confidence[event.data.get("symbol", "")] = event.data.get("confidence", 0.5)

    async def _evaluate(self, symbol: str, interval: str, strategy: StrategyConfig):
        df = await self.market_data.get_historical(symbol, interval)
        if df is None or len(df) < 50:
            return

        df = compute_all(df, strategy.indicators)

        indicator_signal = 0.0
        for side in ["long", "short"]:
            conditions = strategy.entry_conditions.get(side, [])
            for cond in conditions:
                mask = evaluate_condition(df, cond)
                if pd.api.types.is_bool_dtype(mask) and mask.iloc[-1]:
                    indicator_signal = 1.0 if side == "long" else -1.0

        exit_signal = 0.0
        for side in ["long", "short"]:
            conditions = strategy.exit_conditions.get(side, [])
            for cond in conditions:
                mask = evaluate_condition(df, cond)
                if pd.api.types.is_bool_dtype(mask) and mask.iloc[-1]:
                    exit_signal = -1.0 if side == "long" else 1.0

        ml_conf = self._ml_confidence.get(symbol, 0.5)
        news_sentiment = 0.0

        w = self.config.signal_weights
        final_score = (
            indicator_signal * w.indicator +
            ml_conf * w.ml +
            news_sentiment * w.news
        )

        self._signal_cache[symbol] = {
            "strategy": strategy.name,
            "indicator_signal": indicator_signal,
            "ml_confidence": ml_conf,
            "news_sentiment": news_sentiment,
            "final_score": final_score,
            "exit_signal": exit_signal != 0,
        }

        if abs(final_score) > 0.5:
            await self.event_bus.publish(Event(EventType.STRATEGY_SIGNAL, {
                "symbol": symbol,
                "strategy": strategy.name,
                "side": "long" if final_score > 0 else "short",
                "confidence": abs(final_score),
                "timeframe": interval,
                "indicators": self._signal_cache.get(symbol, {}),
            }))

    def get_signal(self, symbol: str) -> dict | None:
        return self._signal_cache.get(symbol)

    async def add_strategy(self, config: StrategyConfig):
        self.loader.save(config)
        self._strategies[config.name] = config

    async def remove_strategy(self, name: str):
        path = self.loader.strategies_dir / f"{name}.yaml"
        if path.exists():
            path.unlink()
        self._strategies.pop(name, None)

    async def stop(self):
        self._running = False
        self.event_bus.unsubscribe(EventType.MARKET_KLINE, self._on_kline)
        self.event_bus.unsubscribe(EventType.ML_PREDICTION, self._on_ml_prediction)
```

- [ ] **Step 4: Create example strategy YAML**

`binance_trader/strategies/rsi_macd_trend.yaml`:
```yaml
name: "RSI+MACD Trend"
mode: trend
timeframes:
  - "1h"
  - "4h"
indicators:
  rsi:
    period: 14
    source: close
  macd:
    fast: 12
    slow: 26
    signal: 9
  bollinger:
    period: 20
    stddev: 2
entry_conditions:
  long:
    - "rsi < 35 and macd_histogram > 0"
    - "(close > bollinger_middle) and (volume_ratio > 1.5)"
  short:
    - "rsi > 70 and macd_histogram < 0"
exit_conditions:
  long:
    - "rsi > 65 or close < bollinger_lower"
  short:
    - "rsi < 35 or close > bollinger_upper"
ml_config:
  enabled: true
  confidence_threshold: 0.6
  features:
    - rsi
    - macd_histogram
    - bollinger_width
    - volume_ratio
    - price_momentum_24h
  weight: 0.4
```

- [ ] **Step 5: Write strategy engine tests**

`tests/test_strategy_engine.py`:
```python
import pytest
import tempfile
import yaml
from pathlib import Path
import pandas as pd
import numpy as np


def test_strategy_loader_load():
    from core.strategy.loader import StrategyLoader
    d = tempfile.mkdtemp()
    strat_dir = Path(d)
    config = {
        "name": "Test Strategy",
        "mode": "trend",
        "timeframes": ["1h"],
        "indicators": {"rsi": {"period": 14, "source": "close"}},
        "entry_conditions": {"long": ["rsi < 30"]},
        "exit_conditions": {"long": ["rsi > 70"]},
    }
    with open(strat_dir / "test_strategy.yaml", "w") as f:
        yaml.dump(config, f)

    loader = StrategyLoader(str(d))
    loaded = loader.load("test_strategy")
    assert loaded.name == "Test Strategy"
    assert loaded.timeframes == ["1h"]


def test_compute_indicators():
    from core.strategy.indicators import compute_all
    dates = pd.date_range("2024-01-01", periods=200, freq="1h")
    df = pd.DataFrame({
        "open": np.random.randn(200).cumsum() + 50000,
        "high": np.random.randn(200).cumsum() + 50500,
        "low": np.random.randn(200).cumsum() + 49500,
        "close": np.random.randn(200).cumsum() + 50000,
        "volume": np.random.rand(200) * 100 + 50,
    }, index=dates)
    df["high"] = df[["open", "close"]].max(axis=1) + abs(np.random.randn(200))
    df["low"] = df[["open", "close"]].min(axis=1) - abs(np.random.randn(200))

    configs = {"rsi": {"period": 14}, "macd": {"fast": 12, "slow": 26, "signal": 9}}
    result = compute_all(df, configs)

    assert "rsi" in result.columns
    assert "macd" in result.columns
    assert "macd_histogram" in result.columns
    assert not result["rsi"].iloc[-1] is None or np.isnan(result["rsi"].iloc[-1])
```

- [ ] **Step 6: Run strategy engine tests**

Run: `cd binance_trader && python -m pytest tests/test_strategy_engine.py -v`
Expected: 2 PASS

### Task 3.2: Risk manager

**Files:**
- Create: `binance_trader/core/risk/__init__.py`
- Create: `binance_trader/core/risk/manager.py`
- Create: `binance_trader/core/risk/circuit_breaker.py`
- Create: `binance_trader/core/risk/position_sizer.py`
- Test: `tests/test_risk_manager.py`

- [ ] **Step 1: Implement circuit breaker**

`binance_trader/core/risk/circuit_breaker.py`:
```python
import time
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    max_daily_drawdown_pct: float = 5.0
    max_weekly_drawdown_pct: float = 10.0
    max_daily_loss_usdt: float = 500.0
    max_consecutive_losses: int = 5

    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    consecutive_losses: int = 0
    daily_start_equity: float = 0.0
    week_start_equity: float = 0.0
    is_tripped: bool = False
    trip_reason: str = ""
    tripped_at: float = 0.0

    def set_equity(self, equity: float):
        if self.daily_start_equity == 0:
            self.daily_start_equity = equity
        if self.week_start_equity == 0:
            self.week_start_equity = equity
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def add_trade_result(self, pnl: float):
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def check(self) -> tuple[bool, str]:
        if self.is_tripped:
            return True, self.trip_reason

        daily_dd = (self.peak_equity - self.current_equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
        if daily_dd > self.max_daily_drawdown_pct:
            self.is_tripped = True
            self.trip_reason = f"Daily drawdown {daily_dd:.2f}% exceeds limit {self.max_daily_drawdown_pct}%"
            self.tripped_at = time.time()
            return True, self.trip_reason

        if self.daily_pnl < -self.max_daily_loss_usdt:
            self.is_tripped = True
            self.trip_reason = f"Daily loss ${abs(self.daily_pnl):.2f} exceeds limit ${self.max_daily_loss_usdt}"
            self.tripped_at = time.time()
            return True, self.trip_reason

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_tripped = True
            self.trip_reason = f"Consecutive losses {self.consecutive_losses} >= limit {self.max_consecutive_losses}"
            self.tripped_at = time.time()
            return True, self.trip_reason

        return False, ""

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_start_equity = self.current_equity

    def reset_weekly(self):
        self.weekly_pnl = 0.0
        self.week_start_equity = self.current_equity

    def reset_trip(self):
        self.is_tripped = False
        self.trip_reason = ""
        self.tripped_at = 0.0
        self.consecutive_losses = 0
```

- [ ] **Step 2: Implement position sizer**

`binance_trader/core/risk/position_sizer.py`:
```python
from app.config import SoftRiskParams, HardRiskLimits


class PositionSizer:
    def __init__(self, hard_limits: HardRiskLimits, soft_params: SoftRiskParams):
        self.hard = hard_limits
        self.soft = soft_params

    def calculate_position_size(self, account_balance: float, current_price: float,
                                 position_type: str = "satellite") -> tuple[float, float]:
        if position_type == "core":
            capital_pool = account_balance * 0.7
        else:
            capital_pool = account_balance * 0.3

        risk_per_trade = capital_pool * (self.soft.position_size_pct / 100)
        risk_per_trade = min(risk_per_trade, account_balance * (self.hard.max_position_size_pct / 100))

        quantity = risk_per_trade / current_price
        return quantity, risk_per_trade

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        sl_pct = self.soft.stop_loss_pct / 100
        min_sl_pct = self.hard.min_stop_loss_distance_pct / 100
        sl_pct = max(sl_pct, min_sl_pct)
        if side == "long":
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)

    def calculate_take_profits(self, entry_price: float, side: str) -> list[tuple[float, float]]:
        levels = [
            self.soft.take_profit_1_pct / 100,
            self.soft.take_profit_2_pct / 100,
            self.soft.take_profit_3_pct / 100,
        ]
        tps = []
        for pct in levels:
            if side == "long":
                tp_price = entry_price * (1 + pct)
            else:
                tp_price = entry_price * (1 - pct)
            tps.append((tp_price, pct))
        return tps
```

- [ ] **Step 3: Implement risk manager** (8-step check chain)

`binance_trader/core/risk/manager.py`:
```python
import asyncio
from dataclasses import dataclass

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.risk.circuit_breaker import CircuitBreaker
from core.risk.position_sizer import PositionSizer


@dataclass
class RiskResult:
    approved: bool
    reason: str = ""
    adjusted_quantity: float | None = None
    adjusted_stop_loss: float | None = None
    adjusted_leverage: int | None = None


class RiskManager:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.breaker = CircuitBreaker(
            max_daily_drawdown_pct=config.hard_limits.max_daily_drawdown_pct,
            max_weekly_drawdown_pct=config.hard_limits.max_weekly_drawdown_pct,
            max_daily_loss_usdt=config.hard_limits.max_daily_loss_usdt,
            max_consecutive_losses=config.hard_limits.max_consecutive_losses,
        )
        self.sizer = PositionSizer(config.hard_limits, config.soft_params)
        self._running = False
        self._open_positions: dict[str, dict] = {}
        self._account_balance: float = 0.0

    async def start(self):
        self._running = True
        self.event_bus.subscribe(EventType.STRATEGY_SIGNAL, self._on_signal)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._on_position_update)

    async def _on_signal(self, event: Event):
        signal = event.data
        result = await self.check_signal(signal)
        if result.approved:
            signal["quantity"] = result.adjusted_quantity or signal.get("quantity", 0)
            signal["stop_loss"] = result.adjusted_stop_loss
            signal["leverage"] = result.adjusted_leverage or signal.get("leverage", self.config.soft_params.leverage)
            await self.event_bus.publish(Event(EventType.ORDER_REQUEST, signal))
        else:
            await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
                "level": "warning",
                "type": "risk_reject",
                "message": f"Signal rejected: {result.reason}",
                "detail": str(signal),
            }))
            await self._log_risk_event("signal_rejected", "warning", result.reason, "RiskManager")

    async def check_signal(self, signal: dict) -> RiskResult:
        # Step 1: Circuit breaker check
        tripped, reason = self.breaker.check()
        if tripped:
            return RiskResult(approved=False, reason=f"Circuit breaker tripped: {reason}")

        # Step 2: Daily loss check
        if abs(self.breaker.daily_pnl) >= self.config.hard_limits.max_daily_loss_usdt:
            return RiskResult(approved=False, reason="Daily loss limit reached")

        # Step 3: Total exposure check
        total_exposure = sum(p.get("position_value", 0) for p in self._open_positions.values())
        exposure_pct = (total_exposure / self._account_balance * 100) if self._account_balance > 0 else 0
        if exposure_pct >= self.config.hard_limits.max_total_exposure_pct:
            return RiskResult(approved=False, reason=f"Total exposure {exposure_pct:.1f}% exceeds limit")

        # Step 4: Position size check
        symbol = signal.get("symbol", "")
        price = signal.get("price", 0)
        qty, risk_amount = self.sizer.calculate_position_size(self._account_balance, price)
        if qty <= 0:
            return RiskResult(approved=False, reason="Position size is zero")

        # Step 5: Leverage check
        leverage = signal.get("leverage", self.config.soft_params.leverage)
        if leverage > self.config.hard_limits.max_leverage:
            leverage = self.config.hard_limits.max_leverage

        # Step 6: Stop loss check
        entry_price = signal.get("price", 0)
        side = signal.get("side", "long")
        sl_price = self.sizer.calculate_stop_loss(entry_price, side)
        sl_distance = abs(entry_price - sl_price) / entry_price * 100
        if sl_distance < self.config.hard_limits.min_stop_loss_distance_pct:
            sl_price = entry_price * (1 - self.config.hard_limits.min_stop_loss_distance_pct / 100) if side == "long" \
                else entry_price * (1 + self.config.hard_limits.min_stop_loss_distance_pct / 100)

        # Step 7: Same symbol check
        if symbol in self._open_positions:
            return RiskResult(approved=False, reason=f"Position already open for {symbol}")

        # Step 8: Max open trades check
        if len(self._open_positions) >= self.config.hard_limits.max_open_trades:
            return RiskResult(approved=False, reason=f"Max open trades {self.config.hard_limits.max_open_trades} reached")

        return RiskResult(
            approved=True,
            adjusted_quantity=qty,
            adjusted_stop_loss=sl_price,
            adjusted_leverage=leverage,
        )

    async def _on_position_update(self, event: Event):
        data = event.data
        symbol = data.get("symbol", "")
        if data.get("closed"):
            self._open_positions.pop(symbol, None)
            pnl = data.get("pnl", 0)
            self.breaker.add_trade_result(pnl)
        else:
            self._open_positions[symbol] = data

    def update_balance(self, balance: float):
        self._account_balance = balance
        self.breaker.set_equity(balance)

    async def _log_risk_event(self, event_type: str, level: str, detail: str, triggered_by: str):
        # Log to DB asynchronously
        await self.event_bus.publish(Event(EventType.RISK_BREACH, {
            "event_type": event_type, "level": level,
            "detail": detail, "triggered_by": triggered_by,
        }))

    async def stop(self):
        self._running = False
        self.event_bus.unsubscribe(EventType.STRATEGY_SIGNAL, self._on_signal)
        self.event_bus.unsubscribe(EventType.POSITION_UPDATE, self._on_position_update)
```

- [ ] **Step 4: Write risk manager tests**

`tests/test_risk_manager.py`:
```python
import pytest


def test_circuit_breaker_trips_on_drawdown():
    from core.risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(max_daily_drawdown_pct=5.0)
    cb.set_equity(10000)
    cb.set_equity(9400)  # 6% drawdown
    tripped, reason = cb.check()
    assert tripped
    assert "drawdown" in reason.lower()


def test_circuit_breaker_trips_on_daily_loss():
    from core.risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(max_daily_loss_usdt=500)
    cb.add_trade_result(-300)
    cb.add_trade_result(-250)
    tripped, reason = cb.check()
    assert tripped
    assert "loss" in reason.lower()


def test_circuit_breaker_trips_on_consecutive_losses():
    from core.risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(max_consecutive_losses=5)
    for _ in range(5):
        cb.add_trade_result(-10)
    tripped, reason = cb.check()
    assert tripped
    assert "consecutive" in reason.lower()


def test_position_sizer():
    from core.risk.position_sizer import PositionSizer
    from app.config import HardRiskLimits, SoftRiskParams
    hard = HardRiskLimits()
    soft = SoftRiskParams()
    sizer = PositionSizer(hard, soft)
    qty, risk = sizer.calculate_position_size(10000, 50000, "core")
    assert qty > 0
    assert risk > 0


def test_stop_loss_calculation():
    from core.risk.position_sizer import PositionSizer
    from app.config import HardRiskLimits, SoftRiskParams
    hard = HardRiskLimits()
    soft = SoftRiskParams(stop_loss_pct=2.0)
    sizer = PositionSizer(hard, soft)
    sl = sizer.calculate_stop_loss(50000, "long")
    assert sl == 49000.0
```

- [ ] **Step 5: Run risk manager tests**

Run: `cd binance_trader && python -m pytest tests/test_risk_manager.py -v`
Expected: 5 PASS

### Task 3.3: Order executor

**Files:**
- Create: `binance_trader/core/executor/__init__.py`
- Create: `binance_trader/core/executor/executor.py`
- Create: `binance_trader/core/executor/sync.py`
- Test: `tests/test_executor.py`

- [ ] **Step 1: Implement order executor**

`binance_trader/core/executor/executor.py`:
```python
import asyncio
import time
from typing import Optional
from binance import AsyncClient
from binance.enums import *

from app.event_bus import EventBus, Event, EventType
from app.config import Config


class OrderExecutor:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.client: Optional[AsyncClient] = None
        self._running = False
        self._orders: dict[str, dict] = {}
        self._positions: dict[str, dict] = {}

    async def start(self):
        self.client = await AsyncClient.create(
            api_key=self.config.binance_api_key or None,
            api_secret=self.config.binance_api_secret or None,
            testnet=self.config.binance_testnet,
        )
        self._running = True
        self.event_bus.subscribe(EventType.ORDER_REQUEST, self._on_order_request)

    async def _on_order_request(self, event: Event):
        if self.config.mode == "sim":
            await self._execute_sim(event.data)
        elif self.config.mode == "live":
            await self._execute_live(event.data)

    async def _execute_sim(self, data: dict):
        order_id = f"sim_{int(time.time() * 1000)}"
        price = data.get("price", 0)
        qty = data.get("quantity", 0)
        symbol = data.get("symbol", "")
        side = data.get("side", "long")

        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": "market",
            "price": price,
            "quantity": qty,
            "filled_qty": qty,
            "status": "filled",
            "binance_order_id": None,
            "stop_loss": data.get("stop_loss"),
            "take_profits": data.get("take_profits", []),
        }
        self._orders[order_id] = order
        self._positions[symbol] = {
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "entry_price": price,
            "current_price": price,
            "unrealized_pnl": 0,
            "stop_loss": data.get("stop_loss"),
            "position_type": data.get("position_type", "satellite"),
        }

        await self.event_bus.publish(Event(EventType.ORDER_UPDATE, {
            "order_id": order_id, "symbol": symbol, "status": "filled", "mode": "sim",
        }))
        await self.event_bus.publish(Event(EventType.POSITION_UPDATE, {
            "symbol": symbol, "side": side, "quantity": qty,
            "entry_price": price, "current_price": price,
            "position_type": data.get("position_type", "satellite"), "closed": False,
        }))

    async def _execute_live(self, data: dict):
        symbol = data.get("symbol", "")
        side = SIDE_BUY if data.get("side") == "long" else SIDE_SELL
        qty = data.get("quantity", 0)

        for attempt in range(3):
            try:
                order = await self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=round(qty, 5),
                )
                order_id = str(order["orderId"])
                self._orders[order_id] = {
                    "id": order_id,
                    "symbol": symbol,
                    "status": order["status"].lower(),
                    "binance_order_id": order["orderId"],
                }
                await self.event_bus.publish(Event(EventType.ORDER_UPDATE, {
                    "order_id": order_id, "symbol": symbol,
                    "status": order["status"].lower(), "mode": "live",
                }))
                return
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
                        "level": "critical", "type": "order_failed",
                        "message": f"Live order failed after 3 attempts: {e}",
                    }))

    def get_open_positions(self) -> dict:
        return self._positions.copy()

    def get_orders(self) -> dict:
        return self._orders.copy()

    async def stop(self):
        self._running = False
        self.event_bus.unsubscribe(EventType.ORDER_REQUEST, self._on_order_request)
        if self.client:
            await self.client.close_connection()
```

- [ ] **Step 2: Write executor test**

`tests/test_executor.py`:
```python
import pytest
import asyncio
from app.event_bus import EventBus, Event, EventType
from app.config import Config


@pytest.mark.asyncio
async def test_sim_order_execution():
    from core.executor.executor import OrderExecutor
    bus = EventBus()
    await bus.start()

    Config._instance = None
    config = Config.load("sim")
    executor = OrderExecutor(config, bus)
    await executor.start()

    order_events = []
    async def on_order(event): order_events.append(event.data)
    bus.subscribe(EventType.ORDER_UPDATE, on_order)

    await bus.publish(Event(EventType.ORDER_REQUEST, {
        "symbol": "BTCUSDT", "side": "long", "price": 50000,
        "quantity": 0.01, "stop_loss": 49000,
        "position_type": "core",
    }))
    await asyncio.sleep(0.2)

    assert len(order_events) > 0
    assert order_events[0]["status"] == "filled"
    assert order_events[0]["mode"] == "sim"

    positions = executor.get_open_positions()
    assert "BTCUSDT" in positions

    await executor.stop()
    await bus.shutdown()
```

- [ ] **Step 3: Run executor tests**

Run: `cd binance_trader && python -m pytest tests/test_executor.py -v`
Expected: 1 PASS

---

## Phase 4: ML & News

### Task 4.1: ML predictor

**Files:**
- Create: `binance_trader/core/ml/__init__.py`
- Create: `binance_trader/core/ml/predictor.py`
- Create: `binance_trader/core/ml/trainer.py`
- Create: `binance_trader/core/ml/features.py`
- Test: `tests/test_ml_predictor.py`

### Task 4.2: News analyzer

**Files:**
- Create: `binance_trader/core/news/__init__.py`
- Create: `binance_trader/core/news/analyzer.py`
- Create: `binance_trader/core/news/fetcher.py`
- Create: `binance_trader/core/news/source_manager.py`
- Test: `tests/test_news_analyzer.py`

---

## Phase 5: AI & Web

### Task 5.1: DeepSeek controller

**Files:**
- Create: `binance_trader/core/ai/__init__.py`
- Create: `binance_trader/core/ai/deepseek_ctl.py`
- Create: `binance_trader/core/ai/prompts.py`
- Test: `tests/test_deepseek.py`

### Task 5.2: Alert manager

**Files:**
- Create: `binance_trader/alerts/__init__.py`
- Create: `binance_trader/alerts/manager.py`
- Create: `binance_trader/alerts/notifications.py`
- Test: `tests/test_alerts.py`

### Task 5.3: Web UI

**Files:**
- Create: `binance_trader/web/__init__.py`
- Create: `binance_trader/web/server.py`
- Create: `binance_trader/web/routes/dashboard.py`
- Create: `binance_trader/web/routes/strategies.py`
- Create: `binance_trader/web/routes/ai.py`
- Create: `binance_trader/web/routes/alerts.py`
- Create: `binance_trader/web/routes/settings.py`
- Create: `binance_trader/web/templates/base.html`
- Create: `binance_trader/web/templates/dashboard.html`
- Create: `binance_trader/web/templates/strategies.html`
- Create: `binance_trader/web/templates/ai_panel.html`
- Create: `binance_trader/web/templates/alerts.html`
- Create: `binance_trader/web/templates/settings.html`
- Create: `binance_trader/web/ws/manager.py`

### Task 5.4: Main application entry

**Files:**
- Create: `binance_trader/app/main.py`

---

## Phase 6: Integration & Vibe-Trading

### Task 6.1: Vibe-Trading connector

**Files:**
- Create: `binance_trader/core/ai/vibe_connector.py`

### Task 6.2: Integration tests & cleanup

**Files:**
- Create: `tests/test_integration.py`
