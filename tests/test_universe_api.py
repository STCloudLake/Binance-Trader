"""Tests for the all-coins data foundation (``docs/overhaul/MARKET_PAGES_API.md``).

Covers the frozen contract shapes plus the .vision host switch, the coin
universe cache/search/paging, watchlist persistence and the offline behaviour.
**No network**: the public market-data host is replaced by
:class:`FakeDataHostClient` and each test gets a temporary data dir, so no repo
artefact (``data/symbols.json``, ``data/market/**``, ``config/*.yaml``) is read
or written.
"""
import asyncio
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from core.market_data.universe import DEFAULT_WATCHLIST, WATCHLIST_MAX, Universe
from db.database import init_database

VIEWER = ("uni_viewer", "V1ewerPass!")
TRADER = ("uni_trader", "T1aderPass!")


# ======================================================================
# fake public market-data host
# ======================================================================
EXCHANGE_SYMBOLS = [
    {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "status": "TRADING",
     "baseAssetPrecision": 8, "quoteAssetPrecision": 8,
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                 {"filterType": "LOT_SIZE", "stepSize": "0.00001000", "minQty": "0.00001000"},
                 {"filterType": "NOTIONAL", "minNotional": "5.00000000"}]},
    {"symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT", "status": "TRADING",
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                 {"filterType": "LOT_SIZE", "stepSize": "0.00010000"},
                 {"filterType": "NOTIONAL", "minNotional": "5.00000000"}]},
    {"symbol": "SOLUSDT", "baseAsset": "SOL", "quoteAsset": "USDT", "status": "TRADING",
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                 {"filterType": "LOT_SIZE", "stepSize": "0.00100000"},
                 {"filterType": "NOTIONAL", "minNotional": "5.00000000"}]},
    {"symbol": "SOLBTC", "baseAsset": "SOL", "quoteAsset": "BTC", "status": "TRADING",
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.00000100"}]},
    {"symbol": "DEADUSDT", "baseAsset": "DEAD", "quoteAsset": "USDT", "status": "BREAK",
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10000000"}]},
]

TICKERS = {
    "BTCUSDT": {"lastPrice": "84242.50", "highPrice": "85000.00", "lowPrice": "83000.00",
                "volume": "1234.5", "quoteVolume": "103000000.5",
                "priceChangePercent": "0.29", "count": 500000},
    "ETHUSDT": {"lastPrice": "3200.25", "highPrice": "3250.00", "lowPrice": "3150.00",
                "volume": "8888.0", "quoteVolume": "28000000.0",
                "priceChangePercent": "-1.25", "count": 120000},
    "SOLUSDT": {"lastPrice": "120.18", "highPrice": "130.00", "lowPrice": "110.00",
                "volume": "5362.0", "quoteVolume": "640000.0",
                "priceChangePercent": "3.40", "count": 20000},
    "SOLBTC": {"lastPrice": "0.0014", "highPrice": "0.0015", "lowPrice": "0.0013",
               "volume": "100.0", "quoteVolume": "0.14",
               "priceChangePercent": "0.10", "count": 50},
    "DEADUSDT": {"lastPrice": "0.001", "highPrice": "0.002", "lowPrice": "0.0005",
                 "volume": "1.0", "quoteVolume": "0.001",
                 "priceChangePercent": "0.0", "count": 2},
}

CALLS = {"ticker_all": 0, "exchange_info": 0, "klines": 0, "depth": 0, "trades": 0,
         "ticker_one": 0}


class FakeDataHostClient:
    """Same surface as ``core.market_data.data_client.MarketDataClient``."""

    error_mode = False
    host = "https://data-api.binance.vision"

    def __init__(self, host=None, timeout=15.0):
        self.host = host or FakeDataHostClient.host

    async def close(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def _guard(self):
        if self.error_mode:
            from core.market_data.data_client import MarketDataError
            raise MarketDataError("simulated data host outage")

    async def exchange_symbols(self):
        self._guard()
        CALLS["exchange_info"] += 1
        return EXCHANGE_SYMBOLS

    async def ticker24h(self, symbol=None):
        self._guard()
        if symbol:
            CALLS["ticker_one"] += 1
            return dict(TICKERS.get(symbol, {}), symbol=symbol)
        CALLS["ticker_all"] += 1
        return [dict(v, symbol=k) for k, v in TICKERS.items()]

    async def klines(self, symbol, interval="1h", limit=500, start_time=None, end_time=None):
        self._guard()
        CALLS["klines"] += 1
        # 1d from the epoch → the listing date (like the real endpoint).
        if interval == "1d" and start_time == 0:
            base = 1597104000000  # 2020-08-11
        else:
            base = 1774000000000
        step = 86400000 if interval == "1d" else 3600000
        return [[base + i * step, "1.0", "2.0", "0.5", "1.5", "10.0",
                 base + i * step + step - 1, "15.0", 5, "5.0", "7.5", "0"]
                for i in range(min(limit, 60))]

    async def order_book(self, symbol, limit=20):
        self._guard()
        CALLS["depth"] += 1
        return {"lastUpdateId": 1,
                "bids": [[84242.0 - i, 0.5 + i] for i in range(limit)],
                "asks": [[84243.0 + i, 0.4 + i] for i in range(limit)]}

    async def recent_trades(self, symbol, limit=30):
        self._guard()
        CALLS["trades"] += 1
        return [{"id": i, "price": "84243.00", "qty": "0.01", "quoteQty": "842.43",
                 "time": 1774000000000 + i * 1000, "isBuyerMaker": i % 2 == 0}
                for i in range(limit)]

    async def symbol_price(self, symbol):
        self._guard()
        return {"symbol": symbol, "price": TICKERS["BTCUSDT"]["lastPrice"]}


# ======================================================================
# fixtures
# ======================================================================
@pytest.fixture(scope="module")
def web_app():
    tmpdir = Path(tempfile.mkdtemp(prefix="bt_universe_"))
    data_dir = tmpdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "universe.db")
    config.data_dir = str(data_dir)
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")
    # The shipped config declares the .vision hosts; an empty temp config must
    # still resolve the documented defaults.
    assert config.market_data_host == "https://data-api.binance.vision"
    assert config.market_stream_host == "wss://data-stream.binance.vision"

    async def _setup():
        await init_database(config.db_path)
        am = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        for username, password, role in ((VIEWER[0], VIEWER[1], "viewer"),
                                         (TRADER[0], TRADER[1], "trader")):
            await am.create_user(username, password, role, username)
        return am

    auth = asyncio.run(_setup())

    from web.server import create_app
    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    app.state.balance = 10000.0
    app.state.get_price = lambda symbol: None
    yield app


@pytest.fixture(autouse=True)
def _fresh_cache(web_app):
    """Fresh fake host + empty caches + a clean data dir for every test."""
    import aiosqlite

    from core.market_data.ttl_cache import TTLCache

    for key in CALLS:
        CALLS[key] = 0
    FakeDataHostClient.error_mode = False
    web_app.state.market_data_client = FakeDataHostClient()
    web_app.state._market_ttl_cache_v2 = TTLCache()
    web_app.state.universe = None
    data_dir = Path(web_app.state.config.data_dir)
    for path in list(data_dir.glob("symbols.json")) + list(data_dir.glob("listing_dates.json")):
        path.unlink()
    market = data_dir / "market"
    if market.exists():
        for parquet in market.rglob("*.parquet"):
            parquet.unlink()

    async def _clear_watchlist():
        async with aiosqlite.connect(web_app.state.config.db_path) as db:
            await db.execute("DELETE FROM system_config WHERE key='watchlist_symbols'")
            await db.commit()

    asyncio.run(_clear_watchlist())
    yield


@pytest.fixture()
def client(web_app):
    c = TestClient(web_app)
    r = c.post("/api/auth/login", json={"username": VIEWER[0], "password": VIEWER[1]})
    assert r.status_code == 200, r.text
    return c


@pytest.fixture()
def trader_client(web_app):
    c = TestClient(web_app)
    r = c.post("/api/auth/login", json={"username": TRADER[0], "password": TRADER[1]})
    assert r.status_code == 200, r.text
    return c


def _universe(web_app) -> Universe:
    uni = Universe(web_app.state.config, client=FakeDataHostClient())
    web_app.state.universe = uni
    return uni


# ======================================================================
# config
# ======================================================================
def test_config_parses_market_hosts_from_yaml():
    tmpdir = Path(tempfile.mkdtemp(prefix="bt_universe_cfg_"))
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    (tmpdir / "config" / "risk_params.yaml").write_text("{}\n", encoding="utf-8")
    (tmpdir / "config" / "config.yaml").write_text(
        "binance:\n  testnet: true\n"
        "  market_data_host: https://example.invalid\n"
        "  market_stream_host: wss://example.invalid\n", encoding="utf-8")
    import app.config as cfg_mod
    original = cfg_mod.PROJECT_ROOT
    Config._instance = None  # drop the session-wide singleton
    try:
        cfg_mod.PROJECT_ROOT = tmpdir
        Config._instance = None
        config = Config.load("sim")
        assert config.market_data_host == "https://example.invalid"
        assert config.market_stream_host == "wss://example.invalid"
        # The trading client keeps obeying `binance.testnet`.
        assert config.binance_testnet is True
    finally:
        cfg_mod.PROJECT_ROOT = original
        Config._instance = None
        Config.load("sim")

def test_config_defaults_to_the_vision_host():
    config = Config.load("sim")
    assert config.market_data_host == "https://data-api.binance.vision"
    assert config.market_stream_host == "wss://data-stream.binance.vision"


def test_no_hardcoded_api_binance_com_in_scoped_sources():
    """Contract §0: no *code* may hardcode api.binance.com / stream.binance.com.

    Comments and docstrings are allowed to name the unreachable hosts (they
    explain why the .vision mirror is used); executable lines are not.
    """
    import ast

    root = Path(__file__).resolve().parent.parent
    scoped = [root / "core" / "market_data", root / "web" / "routes" / "market.py",
              root / "scripts" / "download_history.py"]
    offenders = []
    for target in scoped:
        files = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in files:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            lines = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    end = getattr(node, "end_lineno", node.lineno)
                    lines.update(range(node.lineno, end + 1))
            for i, line in enumerate(source.splitlines(), start=1):
                if i in lines or line.lstrip().startswith("#"):
                    continue
                if "api.binance.com" in line or "stream.binance.com" in line:
                    offenders.append(f"{path.name}:{i}: {line.strip()[:80]}")
    assert not offenders, offenders


# ======================================================================
# universe: cache, search, paging, sort
# ======================================================================
def test_universe_search_filters_status_and_quote(web_app):
    uni = _universe(web_app)
    symbols = asyncio.run(uni.get_symbols())
    assert {s.symbol for s in symbols} == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "SOLBTC", "DEADUSDT"}
    assert uni.has_cached_data("BTCUSDT") is False

    assert [s.symbol for s in uni.filter(q="SOL")] == ["SOLUSDT"]
    assert [s.symbol for s in uni.filter(q="solusdt")] == ["SOLUSDT"]
    assert [s.symbol for s in uni.filter(quote="BTC")] == ["SOLBTC"]
    assert "DEADUSDT" not in [s.symbol for s in uni.filter()]


def test_universe_persists_symbols_json_across_instances(web_app):
    uni = _universe(web_app)
    asyncio.run(uni.get_symbols())
    path = Path(web_app.state.config.data_dir) / "symbols.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["symbols"]) == 5

    # A brand-new instance must answer from disk without touching the host.
    CALLS["exchange_info"] = 0
    fresh = Universe(web_app.state.config, client=FakeDataHostClient())
    assert asyncio.run(fresh.get_symbols())
    assert CALLS["exchange_info"] == 0, "a restart must not refetch exchangeInfo"


def test_universe_serves_stale_cache_when_host_is_down(web_app):
    uni = _universe(web_app)
    asyncio.run(uni.get_symbols())

    FakeDataHostClient.error_mode = True
    offline = Universe(web_app.state.config, client=FakeDataHostClient())
    symbols = asyncio.run(offline.get_symbols(force=True))
    assert {s.symbol for s in symbols} == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "SOLBTC", "DEADUSDT"}


def test_universe_raises_when_host_is_down_and_nothing_is_cached(web_app):
    from core.market_data.data_client import MarketDataError

    FakeDataHostClient.error_mode = True
    uni = Universe(web_app.state.config, client=FakeDataHostClient())
    with pytest.raises(MarketDataError):
        asyncio.run(uni.get_symbols())


def test_has_cached_data_and_intervals(web_app):
    uni = _universe(web_app)
    assert uni.has_cached_data("SOLUSDT") is False
    symbol_dir = Path(web_app.state.config.data_dir) / "market" / "SOLUSDT"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                  "volume": [1.0]},
                 index=pd.to_datetime(["2026-01-01"])).to_parquet(symbol_dir / "1h.parquet")
    assert uni.has_cached_data("SOLUSDT") is True
    assert uni.has_cached_data("BTCUSDT") is False
    assert uni.cached_intervals("SOLUSDT") == ["1h"]
    assert uni.cached_intervals("BTCUSDT") == []


# ======================================================================
# /api/market/symbols
# ======================================================================
def test_symbols_shape_sort_and_defaults(client):
    r = client.get("/api/market/symbols", params={"limit": 5, "sort": "volume"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"total", "offset", "limit", "symbols"}
    assert body["total"] == 3, "only USDT/TRADING pairs are listed"
    assert body["limit"] == 5 and body["offset"] == 0
    assert [s["symbol"] for s in body["symbols"]] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    row = body["symbols"][0]
    assert set(row) == {"symbol", "baseAsset", "quoteAsset", "status", "last", "change_pct",
                        "high", "low", "volume", "quote_volume", "count", "listing_date",
                        "tick_size", "step_size", "min_notional", "has_cached_data"}
    assert row["last"] == 84242.5
    assert row["change_pct"] == 0.29
    assert row["quote_volume"] == 103000000.5
    assert row["count"] == 500000
    assert row["tick_size"] == 0.01 and row["step_size"] == 0.00001
    assert row["min_notional"] == 5.0
    assert row["has_cached_data"] is False
    assert row["listing_date"] == "2020-08-11", "listing date comes from the first 1d kline"


def test_symbols_search_change_sort_and_paging(client):
    body = client.get("/api/market/symbols", params={"q": "sol"}).json()
    assert [s["symbol"] for s in body["symbols"]] == ["SOLUSDT"]

    body = client.get("/api/market/symbols", params={"sort": "change"}).json()
    assert [s["symbol"] for s in body["symbols"]] == ["SOLUSDT", "BTCUSDT", "ETHUSDT"]

    body = client.get("/api/market/symbols", params={"sort": "symbol", "offset": 1,
                                                     "limit": 1}).json()
    assert body["total"] == 3 and body["offset"] == 1
    assert [s["symbol"] for s in body["symbols"]] == ["ETHUSDT"]


def test_symbols_limit_is_clamped_and_sort_validated(client):
    body = client.get("/api/market/symbols", params={"limit": 999}).json()
    assert body["limit"] == 200
    body = client.get("/api/market/symbols", params={"limit": 0}).json()
    assert body["limit"] == 1
    r = client.get("/api/market/symbols", params={"sort": "nope"})
    assert r.status_code == 400 and "sort" in r.json()["error"]


def test_symbols_reports_cached_parquet(client, web_app):
    symbol_dir = Path(web_app.state.config.data_dir) / "market" / "ETHUSDT"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    (symbol_dir / "1h.parquet").write_bytes(b"stub")
    body = client.get("/api/market/symbols", params={"q": "ETH"}).json()
    assert body["symbols"][0]["has_cached_data"] is True


def test_symbols_offline_returns_structured_error(client):
    FakeDataHostClient.error_mode = True
    r = client.get("/api/market/symbols")
    assert r.status_code == 502, r.text
    assert set(r.json()) == {"error"}
    assert "Traceback" not in r.text


# ======================================================================
# /api/market/ticker24h — cached 60s
# ======================================================================
def test_ticker24h_shape_and_60s_cache(client):
    r = client.get("/api/market/ticker24h")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"updated_at", "tickers"}
    assert set(body["tickers"]["BTCUSDT"]) == {"last", "change_pct", "high", "low",
                                               "volume", "quote_volume", "count"}
    assert body["tickers"]["BTCUSDT"]["last"] == 84242.5
    assert CALLS["ticker_all"] == 1

    # A burst of polls inside the 60s window must not hit the host again.
    for _ in range(5):
        assert client.get("/api/market/ticker24h").status_code == 200
    assert CALLS["ticker_all"] == 1, "ticker24h must be cached for 60s"

    # /api/market/symbols rides the same snapshot (contract §1).
    client.get("/api/market/symbols")
    assert CALLS["ticker_all"] == 1


def test_ticker24h_offline_returns_structured_error(client):
    FakeDataHostClient.error_mode = True
    r = client.get("/api/market/ticker24h")
    assert r.status_code == 502
    assert "error" in r.json()


# ======================================================================
# watchlist persistence
# ======================================================================
def test_watchlist_defaults_to_the_five_engine_symbols(client):
    body = client.get("/api/market/watchlist").json()
    assert body == {"symbols": DEFAULT_WATCHLIST, "max": WATCHLIST_MAX}


def test_watchlist_post_persists_and_requires_restart(trader_client, web_app):
    r = trader_client.post("/api/market/watchlist",
                           data={"symbols": "SOLUSDT, BTCUSDT ,solusdt"})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "symbols": ["SOLUSDT", "BTCUSDT"],
                        "restart_required": True}
    # Persisted in system_config...
    assert trader_client.get("/api/market/watchlist").json()["symbols"] == ["SOLUSDT", "BTCUSDT"]
    # ...and readable straight from the database (what the engine reads at start).
    from core.market_data.universe import load_watchlist
    assert asyncio.run(load_watchlist(web_app.state.config.db_path)) == ["SOLUSDT", "BTCUSDT"]


def test_watchlist_rejects_unknown_unknown_status_and_overlong(trader_client):
    r = trader_client.post("/api/market/watchlist", data={"symbols": "SOLUSDT,NOPEUSDT"})
    assert r.status_code == 400 and "NOPEUSDT" in r.json()["error"]

    # Delisted (status != TRADING) is rejected too.
    r = trader_client.post("/api/market/watchlist", data={"symbols": "DEADUSDT"})
    assert r.status_code == 400 and "DEADUSDT" in r.json()["error"]

    r = trader_client.post("/api/market/watchlist",
                           data={"symbols": ",".join(f"AAA{i}USDT" for i in range(31))})
    assert r.status_code == 400 and "30" in r.json()["error"]

    r = trader_client.post("/api/market/watchlist", data={"symbols": ""})
    assert r.status_code == 400
    # Nothing was persisted by the failed calls.
    assert trader_client.get("/api/market/watchlist").json()["symbols"] == DEFAULT_WATCHLIST


def test_watchlist_post_is_trader_only(web_app):
    c = TestClient(web_app)
    c.post("/api/auth/login", json={"username": VIEWER[0], "password": VIEWER[1]})
    assert c.post("/api/market/watchlist", data={"symbols": "SOLUSDT"}).status_code == 403


def test_watchlist_load_falls_back_when_db_has_no_row(web_app):
    from core.market_data.universe import load_watchlist
    fresh = Path(tempfile.mkdtemp(prefix="bt_universe_db_")) / "empty.db"
    assert asyncio.run(load_watchlist(str(fresh))) == DEFAULT_WATCHLIST


# ======================================================================
# provider watches the persisted list
# ======================================================================
class _StubAsyncClient:
    def __init__(self):
        self.closed = False

    async def close_connection(self):
        self.closed = True


def test_provider_start_watches_the_persisted_watchlist(web_app, monkeypatch, capsys):
    import binance
    from core.market_data.provider import MarketDataProvider
    from core.market_data.universe import save_watchlist

    called = {}

    async def _fake_prefetch(self, symbols, intervals):
        called["prefetch"] = list(symbols)

    monkeypatch.setattr(binance.AsyncClient, "create",
                        classmethod(lambda cls, **kw: _wrap(_StubAsyncClient())))
    monkeypatch.setattr(MarketDataProvider, "_prefetch_history", _fake_prefetch)

    config = web_app.state.config
    asyncio.run(save_watchlist(config.db_path, ["SOLUSDT", "ADAUSDT"]))

    provider = MarketDataProvider(config, EventBus())
    asyncio.run(provider.start(None, ["1h"]))
    try:
        assert provider.watched_symbols == ["SOLUSDT", "ADAUSDT"]
        assert called["prefetch"] == ["SOLUSDT", "ADAUSDT"]
        provider._running = False
        for task in provider._tasks:
            task.cancel()
    finally:
        asyncio.run(provider.stop())


def test_provider_falls_back_to_default_symbols(web_app, monkeypatch):
    import binance
    from core.market_data.provider import MarketDataProvider

    async def _fake_prefetch(self, symbols, intervals):
        return None

    monkeypatch.setattr(binance.AsyncClient, "create",
                        classmethod(lambda cls, **kw: _wrap(_StubAsyncClient())))
    monkeypatch.setattr(MarketDataProvider, "_prefetch_history", _fake_prefetch)

    provider = MarketDataProvider(web_app.state.config, EventBus())
    asyncio.run(provider.start(None, ["1h"]))
    assert provider.watched_symbols == DEFAULT_WATCHLIST


async def _wrap(client):
    return client


# ======================================================================
# /api/coin/{symbol}
# ======================================================================
def test_coin_detail_shape(client):
    r = client.get("/api/coin/SOLUSDT")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"symbol", "base_asset", "quote_asset", "status", "listing_date",
                         "filters", "ticker", "depth_summary", "performance", "risk",
                         "correlation_btc_30d", "cached_intervals"}
    assert body["symbol"] == "SOLUSDT" and body["base_asset"] == "SOL"
    assert body["quote_asset"] == "USDT" and body["status"] == "TRADING"
    assert body["listing_date"] == "2020-08-11"
    assert body["filters"] == {"tick_size": 0.01, "step_size": 0.001, "min_notional": 5.0}
    assert set(body["ticker"]) == {"last", "change_pct", "high", "low", "volume",
                                   "quote_volume", "count"}
    # spread_pct is a RATIO here (contract example: 1.2e-05), not a percentage.
    assert body["depth_summary"]["spread"] == pytest.approx(1.0)
    assert 0 < body["depth_summary"]["spread_pct"] < 0.001
    assert set(body["performance"]) == {"d1", "d7", "d30", "d90"}
    assert body["performance"]["d1"] == 3.4
    assert set(body["risk"]) == {"volatility_30d_annualized", "max_drawdown_90d",
                                 "liquidity_score", "spread_score", "volume_score",
                                 "overall_score", "flags"}
    for key in ("volatility_30d_annualized", "liquidity_score", "spread_score",
                "volume_score", "overall_score"):
        assert body["risk"][key] is not None, key
    assert body["risk"]["overall_score"] == pytest.approx(
        round((body["risk"]["liquidity_score"] + body["risk"]["spread_score"]
               + body["risk"]["volume_score"]) / 3))
    assert isinstance(body["risk"]["flags"], list)
    assert body["cached_intervals"] == []


def test_coin_detail_reports_cached_intervals_and_unknown_symbol(client, web_app):
    symbol_dir = Path(web_app.state.config.data_dir) / "market" / "SOLUSDT"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    for interval in ("1h", "4h"):
        pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                      "volume": [1.0]}, index=pd.to_datetime(["2026-01-01"])
                     ).to_parquet(symbol_dir / f"{interval}.parquet")
    body = client.get("/api/coin/SOLUSDT").json()
    assert body["cached_intervals"] == ["1h", "4h"]

    r = client.get("/api/coin/NOPEUSDT")
    assert r.status_code == 404
    assert "error" in r.json()


def test_coin_detail_is_cached_for_30s(client):
    client.get("/api/coin/SOLUSDT")
    klines_after_first = CALLS["klines"]
    client.get("/api/coin/SOLUSDT")
    assert CALLS["klines"] == klines_after_first, "coin detail must be cached 30s"
    # ...but the cache is per symbol.
    client.get("/api/coin/BTCUSDT")
    assert CALLS["klines"] > klines_after_first


# ======================================================================
# /api/data/overview
# ======================================================================
def test_data_overview_shape_and_numbers(client):
    r = client.get("/api/data/overview")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"updated_at", "totals", "top_gainers", "top_losers", "top_volume",
                         "most_active", "volatility_leaders", "spread_widest", "btc_eth_share"}
    assert body["totals"]["symbols"] == 3
    assert body["totals"]["up"] == 2 and body["totals"]["down"] == 1
    assert body["totals"]["flat"] == 0
    assert body["totals"]["quote_volume_usdt"] == pytest.approx(
        103000000.5 + 28000000.0 + 640000.0, abs=1.0)
    assert [s["symbol"] for s in body["top_gainers"]][:2] == ["SOLUSDT", "BTCUSDT"]
    assert body["top_losers"][0]["symbol"] == "ETHUSDT"
    assert body["top_volume"][0]["symbol"] == "BTCUSDT"
    assert body["most_active"][0]["symbol"] == "BTCUSDT"
    assert set(body["most_active"][0]) == {"symbol", "count", "quote_volume"}
    assert body["btc_eth_share"]["BTCUSDT"] == pytest.approx(
        103000000.5 / (103000000.5 + 28000000.0 + 640000.0), abs=1e-6)
    assert len(body["top_gainers"]) == 3 and len(body["top_losers"]) == 3
    for entry in body["volatility_leaders"]:
        assert set(entry) == {"symbol", "volatility_30d_annualized"}
    for entry in body["spread_widest"]:
        assert set(entry) == {"symbol", "spread_pct"}


def test_data_overview_offline_returns_structured_error(client):
    FakeDataHostClient.error_mode = True
    r = client.get("/api/data/overview")
    assert r.status_code == 502
    assert "error" in r.json()


# ======================================================================
# /api/kline/{symbol} — same shape, data host
# ======================================================================
def test_kline_uses_the_data_host_with_epoch_seconds(client, web_app):
    # No engine cache in this app, so the handler must fall through to the host.
    assert getattr(web_app.state, "market_data", None) is None
    r = client.get("/api/kline/SOLUSDT", params={"interval": "5m", "limit": 3})
    assert r.status_code == 200, r.text
    candles = r.json()
    assert isinstance(candles, list) and len(candles) == 3
    assert set(candles[0]) == {"time", "open", "high", "low", "close", "volume"}
    assert candles[0]["time"] == 1774000000, "time must be epoch SECONDS"
    assert CALLS["klines"] >= 1


def test_kline_offline_is_structured_json(client):
    FakeDataHostClient.error_mode = True
    r = client.get("/api/kline/BTCUSDT", params={"limit": 2})
    assert r.status_code == 502
    assert set(r.json()) == {"error"}


def test_kline_route_is_registered_once(client, web_app):
    paths = [getattr(r, "path", None) for r in web_app.router.routes]
    assert paths.count("/api/kline/{symbol}") == 1, \
        "the testnet handler must be replaced, not shadowed"


# ======================================================================
# authz: reads need a login, the watchlist write needs a trader
# ======================================================================
@pytest.mark.parametrize("path,params", [
    ("/api/market/symbols", {"limit": 2}),
    ("/api/market/ticker24h", {}),
    ("/api/market/watchlist", {}),
    ("/api/coin/BTCUSDT", {}),
    ("/api/data/overview", {}),
    ("/api/kline/BTCUSDT", {"limit": 2}),
])
def test_anonymous_is_rejected(web_app, path, params):
    assert TestClient(web_app).get(path, params=params).status_code == 401


@pytest.mark.parametrize("path,params", [
    ("/api/market/symbols", {"limit": 2}),
    ("/api/market/ticker24h", {}),
    ("/api/market/watchlist", {}),
    ("/api/coin/BTCUSDT", {}),
    ("/api/data/overview", {}),
    ("/api/kline/BTCUSDT", {"limit": 2}),
])
def test_viewer_can_read_every_new_endpoint(client, path, params):
    assert client.get(path, params=params).status_code == 200, path
