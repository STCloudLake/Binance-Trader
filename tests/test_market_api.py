"""Tests for the Binance-style trade-page API (`web/routes/market.py`) and the
pending limit-order matcher (`core/executor/pending_orders.py`).

Everything runs against a temporary SQLite database and a fake exchange client /
fake price function: no network, and `data/binance_trader.db` is never touched.
"""
import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from core.executor.pending_orders import (
    LimitOrderMatcher,
    PendingOrderStore,
    STATUS_CANCELLED,
    STATUS_FILLED,
    STATUS_OPEN,
)
from db.database import init_database

VIEWER = ("mkt_viewer", "V1ewerPass!")
TRADER = ("mkt_trader", "T1aderPass!")

#: Fake 24h ticker per symbol.
FAKE_TICKERS = {
    "BTCUSDT": {"lastPrice": "84242.50", "openPrice": "84000.00", "highPrice": "85000.00",
                "lowPrice": "83000.00", "volume": "1234.567", "quoteVolume": "103000000.5",
                "priceChange": "242.50", "priceChangePercent": "0.29", "closeTime": 1774000000000},
    "ETHUSDT": {"lastPrice": "3200.25", "openPrice": "3190.00", "highPrice": "3250.00",
                "lowPrice": "3150.00", "volume": "8888.0", "quoteVolume": "28000000.0",
                "priceChange": "10.25", "priceChangePercent": "0.32", "closeTime": 1774000000000},
}

#: Fake exchangeInfo symbols — the universe/search/paging endpoints run on these.
FAKE_EXCHANGE_SYMBOLS = [
    {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "status": "TRADING",
     "baseAssetPrecision": 8, "quoteAssetPrecision": 8,
     "filters": [
         {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
         {"filterType": "LOT_SIZE", "stepSize": "0.00001000", "minQty": "0.00001000"},
         {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
     ]},
    {"symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT", "status": "TRADING",
     "baseAssetPrecision": 8, "quoteAssetPrecision": 8,
     "filters": [
         {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
         {"filterType": "LOT_SIZE", "stepSize": "0.00010000", "minQty": "0.00010000"},
         {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
     ]},
    {"symbol": "SOLUSDT", "baseAsset": "SOL", "quoteAsset": "USDT", "status": "TRADING",
     "baseAssetPrecision": 8, "quoteAssetPrecision": 8,
     "filters": [
         {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
         {"filterType": "LOT_SIZE", "stepSize": "0.00100000", "minQty": "0.00100000"},
         {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
     ]},
    # Delisted pair: must never be offered by /api/market/symbols nor accepted by
    # the watchlist validator.
    {"symbol": "OLDUSDT", "baseAsset": "OLD", "quoteAsset": "USDT", "status": "BREAK",
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10000000"}]},
]

EXCHANGE_CALLS = {"create": 0, "ticker": 0, "ticker_all": 0, "depth": 0, "trades": 0,
                  "klines": 0, "exchange_info": 0, "failures": 0}


class FakeBinanceClient:
    """Stand-in for python-binance AsyncClient (testnet-shaped payloads).

    Since the market-data migration (``docs/overhaul/MARKET_PAGES_API.md`` §0)
    this fake is only reached by the **trading** client (``/api/order`` market
    fallback, ``_rest_price``) — public market data no longer goes through
    python-binance, so it is served by :class:`FakeMarketDataClient` below.
    """

    mode = "normal"

    def __init__(self, mode: str = "normal"):
        self.mode = mode
        self.closed = False

    @classmethod
    async def create(cls, **kwargs):
        EXCHANGE_CALLS["create"] += 1
        assert kwargs.get("testnet") is True, "the trading client must use testnet"
        return cls(cls.mode)

    async def get_ticker(self, symbol=None, **kwargs):
        EXCHANGE_CALLS["ticker"] += 1
        if self.mode == "sparse":
            return {"symbol": symbol, "lastPrice": "100.0"}
        return dict(FAKE_TICKERS.get(symbol, FAKE_TICKERS["BTCUSDT"]), symbol=symbol)

    async def get_order_book(self, symbol=None, limit=20, **kwargs):
        EXCHANGE_CALLS["depth"] += 1
        bids = [[84242.0 - i, 0.5 + i] for i in range(limit)]
        asks = [[84243.0 + i, 0.4 + i] for i in range(limit)]
        return {"lastUpdateId": 987654, "bids": bids, "asks": asks}

    async def get_recent_trades(self, symbol=None, limit=30, **kwargs):
        EXCHANGE_CALLS["trades"] += 1
        # Oldest-first, like the real endpoint; `time` is in milliseconds and the
        # endpoint converts to whole seconds, so step by 1000ms to keep order.
        return [{"id": 1000 + i, "price": "84243.00", "qty": "0.01",
                 "quoteQty": "842.43", "time": 1774000000000 + i * 1000,
                 "isBuyerMaker": i % 2 == 0}
                for i in range(limit)]

    async def get_symbol_ticker(self, symbol=None, **kwargs):
        return {"symbol": symbol, "price": FAKE_TICKERS["BTCUSDT"]["lastPrice"]}

    async def close_connection(self):
        self.closed = True


class FakeMarketDataClient:
    """Stand-in for `core.market_data.data_client.MarketDataClient`.

    Answers with the same payloads the real hosts return — the shapes are the
    contract — while counting calls so cache behaviour can be asserted.  Assigned
    to ``app.state.market_data_client`` by the ``patched_data_host`` fixture, so
    tests never touch the network (which the old fake got away with because the
    testnet client was monkeypatched at the class level).
    """

    mode = "normal"
    error_mode = False

    def __init__(self, host="https://data-api.binance.vision", mode="normal"):
        self.host = host
        self.mode = mode
        self.closed = False

    async def close(self):
        self.closed = True

    async def _maybe_fail(self):
        if self.error_mode:
            EXCHANGE_CALLS["failures"] += 1
            from core.market_data.data_client import MarketDataError
            raise MarketDataError("simulated market data outage")

    async def ticker24h(self, symbol=None):
        await self._maybe_fail()
        if symbol:
            EXCHANGE_CALLS["ticker"] += 1
            if self.mode == "sparse":
                return {"symbol": symbol, "lastPrice": "100.0"}
            return dict(FAKE_TICKERS.get(symbol, FAKE_TICKERS["BTCUSDT"]), symbol=symbol)
        EXCHANGE_CALLS["ticker_all"] += 1
        return [dict(v, symbol=k) for k, v in FAKE_TICKERS.items()]

    async def order_book(self, symbol, limit=20):
        await self._maybe_fail()
        EXCHANGE_CALLS["depth"] += 1
        bids = [[84242.0 - i, 0.5 + i] for i in range(limit)]
        asks = [[84243.0 + i, 0.4 + i] for i in range(limit)]
        return {"lastUpdateId": 987654, "bids": bids, "asks": asks}

    async def recent_trades(self, symbol, limit=30):
        await self._maybe_fail()
        EXCHANGE_CALLS["trades"] += 1
        return [{"id": 1000 + i, "price": "84243.00", "qty": "0.01",
                 "quoteQty": "842.43", "time": 1774000000000 + i * 1000,
                 "isBuyerMaker": i % 2 == 0}
                for i in range(limit)]

    async def symbol_price(self, symbol):
        await self._maybe_fail()
        return {"symbol": symbol, "price": FAKE_TICKERS["BTCUSDT"]["lastPrice"]}

    async def klines(self, symbol, interval="1h", limit=500, start_time=None, end_time=None):
        await self._maybe_fail()
        EXCHANGE_CALLS["klines"] += 1
        return [[1774000000000 + i * 3600000, "1.0", "2.0", "0.5", "1.5", "10.0",
                 1774003599999 + i * 3600000, "15.0", 5, "5.0", "7.5", "0"]
                for i in range(min(limit, 5))]

    async def exchange_symbols(self):
        await self._maybe_fail()
        EXCHANGE_CALLS["exchange_info"] += 1
        return FAKE_EXCHANGE_SYMBOLS


class FakeRiskManager:
    """Minimal stand-in: records signals and returns a scripted decision."""

    def __init__(self, approve=True, reason="", adjusted_quantity=None):
        self.approve = approve
        self.reason = reason
        self.adjusted_quantity = adjusted_quantity
        self.signals = []

    async def check_signal(self, signal):
        from core.risk.manager import RiskResult
        self.signals.append(dict(signal))
        if not self.approve:
            return RiskResult(approved=False, reason=self.reason)
        return RiskResult(approved=True, adjusted_quantity=self.adjusted_quantity,
                          adjusted_stop_loss=signal.get("stop_loss"), adjusted_leverage=2)


class RecordingBus:
    """Records published ORDER_REQUEST events (deterministic, no consumer task)."""

    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)

    def signals(self, event_type=None):
        return [e.data for e in self.events if event_type is None or e.type == event_type]


class FakeState:
    pass


class FakeApp:
    def __init__(self):
        self.state = FakeState()


class _StubExecutor:
    """Satisfies the 'executor must exist' guard without executing anything."""

    def get_open_positions(self):
        return {}

    async def close_position(self, symbol, reduce_pct=100, current_price=0):
        return {"ok": False, "error": "stub"}   # pragma: no cover


def _make_app_state(prices=None, risk_manager=None, market_data=None, executor=None):
    app = FakeApp()
    fake_prices = dict(prices or {})
    app.state.get_price = lambda symbol: fake_prices.get(symbol)
    app.state.risk_manager = risk_manager
    if market_data is not None:
        app.state.market_data = market_data
    app.state.executor = executor
    return app, fake_prices


def _run(coro):
    return asyncio.run(coro)


# ======================================================================
# fixtures: temp DB + TestClient app
# ======================================================================
@pytest.fixture(scope="module")
def web_app():
    tmpdir = Path(tempfile.mkdtemp(prefix="bt_market_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "market.db")
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")
    assert "data" not in config.db_path or "bt_market_" in config.db_path

    async def _setup():
        await init_database(config.db_path)
        am = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        for username, password, role in (
            (VIEWER[0], VIEWER[1], "viewer"),
            (TRADER[0], TRADER[1], "trader"),
        ):
            await am.create_user(username, password, role, username)
        # One closed trade so /api/history/trades has a row to project.
        # `action='close'` is required since the endpoint returns only real exits
        # (an `action='open'` row with status='closed' is the entry leg of a round
        # trip, not a history entry).
        import aiosqlite
        db = await aiosqlite.connect(config.db_path)
        await db.execute(
            "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl, "
            "pnl_pct, strategy, status, action, closed_at, opened_at) "
            "VALUES ('BTCUSDT','long',80000,81000,0.001,1.0,1.25,'manual','closed','close',"
            "'2026-01-01 04:00:00','2026-01-01 00:00:00')")
        await db.commit()
        await db.close()
        return am

    auth = _run(_setup())

    from web.server import create_app
    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    app.state.balance = 10000.0
    app.state.get_price = lambda symbol: None
    yield app

    # Make sure no matcher task leaks out of the test session.
    matcher = getattr(app.state, "limit_order_matcher", None)
    if matcher is not None:
        _run(matcher.stop())


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


@pytest.fixture()
def data_host(web_app):
    """Fresh fake market-data host + an empty v2 TTL cache for one test.

    `app.state.market_data_client` is read once per request via the route
    module's `_data_client()`, but the module also caches the Universe; both are
    reset here so a test can change the fake's mode (`sparse`, `error_mode`)
    without the previous test's cached payload leaking in.
    """
    from core.market_data.ttl_cache import TTLCache

    def _inject(mode: str = "normal", error: bool = False) -> FakeMarketDataClient:
        fake = FakeMarketDataClient()
        fake.mode = mode
        fake.error_mode = error
        web_app.state.market_data_client = fake
        web_app.state._market_ttl_cache_v2 = TTLCache()
        web_app.state.universe = None
        return fake

    return _inject


@pytest.fixture()
def patched_exchange(monkeypatch):
    """Fake **trading** client (testnet) + fake market-data host client.

    Public market data moved off python-binance to
    ``config.market_data_host`` (``https://data-api.binance.vision``): the given
    host is unreachable from CI, so every data call is served by
    :class:`FakeMarketDataClient` injected on ``app.state``.
    """
    import binance
    for key in EXCHANGE_CALLS:
        EXCHANGE_CALLS[key] = 0
    monkeypatch.setattr(binance.AsyncClient, "create", FakeBinanceClient.create)
    return FakeBinanceClient


@pytest.fixture(autouse=True)
def _clean_pending_orders(web_app):
    """Each test starts with an empty pending-order book and a default executor."""
    import aiosqlite
    db_path = web_app.state.config.db_path

    async def _clear():
        db = await aiosqlite.connect(db_path)
        await db.execute("DELETE FROM pending_orders")
        await db.execute("DELETE FROM system_config WHERE key='watchlist_symbols'")
        await db.commit()
        await db.close()

    _run(_clear())
    web_app.state._market_ttl_cache = {}
    web_app.state._market_fetch_locks = {}
    # v2 TTL cache + data-host client (market data no longer uses the testnet
    # client, so the fake must be re-injected for every test).
    from core.market_data.ttl_cache import TTLCache
    web_app.state._market_ttl_cache_v2 = TTLCache()
    web_app.state.market_data_client = FakeMarketDataClient()
    web_app.state.universe = None
    # Reset the fake exchange payload mode (a class attribute lives across tests)
    # and make sure a previous test's executor stub does not leak forward: the
    # order routes require *some* executor, though these tests never execute on it.
    FakeBinanceClient.mode = "normal"
    FakeMarketDataClient.mode = "normal"
    FakeMarketDataClient.error_mode = False
    web_app.state.executor = _StubExecutor()
    web_app.state.risk_manager = None
    web_app.state.get_price = lambda symbol: None
    yield


# ======================================================================
# 一、response shapes
# ======================================================================
def test_market_ticker_shape(client, patched_exchange):
    r = client.get("/api/market/ticker", params={"symbol": "BTCUSDT"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"symbol", "last", "open", "high", "low", "volume",
                         "quote_volume", "change", "change_pct", "time"}
    assert body["symbol"] == "BTCUSDT"
    assert body["last"] == 84242.5
    assert body["open"] == 84000.0
    assert body["change"] == 242.5
    assert body["change_pct"] == 0.29
    assert body["volume"] == 1234.567
    # closeTime (ms) → time (s)
    assert body["time"] == 1774000000
    assert isinstance(body["quote_volume"], float)


def test_market_ticker_missing_fields_are_null(client, data_host):
    """A sparse exchange payload must render nulls, not raise."""
    data_host(mode="sparse")
    body = client.get("/api/market/ticker", params={"symbol": "BTCUSDT"}).json()
    assert body["last"] == 100.0
    assert body["volume"] is None and body["change_pct"] is None and body["time"] is None


def test_market_depth_shape_and_ordering(client, patched_exchange):
    r = client.get("/api/market/depth", params={"symbol": "BTCUSDT", "limit": 20})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"symbol", "lastUpdateId", "bids", "asks",
                         "spread", "spread_pct", "bid_total", "ask_total"}
    assert body["lastUpdateId"] == 987654
    assert len(body["bids"]) == 20 and len(body["asks"]) == 20
    bid_prices = [lv[0] for lv in body["bids"]]
    ask_prices = [lv[0] for lv in body["asks"]]
    assert bid_prices == sorted(bid_prices, reverse=True), "bids must be descending"
    assert ask_prices == sorted(ask_prices), "asks must be ascending"
    assert body["spread"] == pytest.approx(1.0)
    # The testnet fake returns whole values, so price and qty are exact here;
    # `spread_pct` is rounded to 6 decimals per the contract.
    assert body["spread_pct"] == pytest.approx(1.0 / 84243.0 * 100, abs=5e-7)
    assert body["bid_total"] > 0 and body["ask_total"] > 0


@pytest.mark.parametrize("bad_limit,expected", [(0, 1), (1000, 100), (-5, 1)])
def test_market_depth_limit_is_clamped(client, patched_exchange, bad_limit, expected):
    body = client.get("/api/market/depth",
                      params={"symbol": "BTCUSDT", "limit": bad_limit}).json()
    assert len(body["bids"]) == expected


def test_market_trades_shape_newest_first(client, patched_exchange):
    r = client.get("/api/market/trades", params={"symbol": "BTCUSDT", "limit": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"symbol", "trades"}
    assert len(body["trades"]) == 5
    assert set(body["trades"][0]) == {"id", "price", "qty", "quote_qty", "time", "is_buyer_maker"}
    times = [t["time"] for t in body["trades"]]
    assert times == sorted(times, reverse=True), "newest trade first"
    # Binance returns trades oldest-first; the endpoint must reverse them.
    ids = [t["id"] for t in body["trades"]]
    assert ids == [1004, 1003, 1002, 1001, 1000], "newest trade first"
    # is_buyer_maker travels with its own trade, not with the slot
    assert [t["is_buyer_maker"] for t in body["trades"]] == [True, False, True, False, True]
    assert body["trades"][0]["price"] == 84243.0
    assert body["trades"][0]["quote_qty"] == 842.43


def test_market_overview_covers_the_five_engine_symbols(client, patched_exchange):
    r = client.get("/api/market/overview")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"symbols", "updated_at"}
    assert [s["symbol"] for s in body["symbols"]] == [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    for entry in body["symbols"]:
        assert set(entry) == {"symbol", "last", "change_pct", "quote_volume"}
    assert body["symbols"][0]["last"] == 84242.5
    assert body["symbols"][1]["last"] == 3200.25
    assert isinstance(body["updated_at"], int)


def test_market_ttl_cache_avoids_a_connection_per_request(client, patched_exchange):
    """Two polls inside the TTL window share one upstream call."""
    first = client.get("/api/market/ticker", params={"symbol": "BTCUSDT"}).json()
    second = client.get("/api/market/ticker", params={"symbol": "BTCUSDT"}).json()
    assert first == second
    assert EXCHANGE_CALLS["ticker"] == 1, "a poll burst must hit the data host once"


def test_market_error_is_structured_json_not_a_traceback(client, data_host):
    data_host(error=True)
    r = client.get("/api/market/ticker", params={"symbol": "BTCUSDT"})
    assert r.status_code == 502, r.text
    body = r.json()
    assert set(body) == {"error"}
    assert "simulated market data outage" in body["error"]
    assert "Traceback" not in r.text


def test_fetch_errors_are_not_cached(client, data_host):
    data_host(error=True)
    before = EXCHANGE_CALLS["failures"]
    assert client.get("/api/market/ticker", params={"symbol": "BTCUSDT"}).status_code == 502
    assert client.get("/api/market/ticker", params={"symbol": "BTCUSDT"}).status_code == 502
    # A failure must not be cached for the TTL window: both polls must have tried
    # the upstream again.
    assert EXCHANGE_CALLS["failures"] - before == 2


def test_account_shape(client, web_app):
    class FakeExecutor:
        def get_open_positions(self):
            return {"BTCUSDT": {
                "symbol": "BTCUSDT", "side": "long", "quantity": 0.00269,
                "entry_price": 84055.6, "current_price": 84055.6, "unrealized_pnl": 0.0,
                "stop_loss": 82374.5, "strategy_name": "ga_champion_x",
                "position_type": "core", "amount_usdt": 226.0,
            }}

    web_app.state.executor = FakeExecutor()
    # Balance is DB-first now: the in-memory cache is deliberately stale here to
    # prove the endpoint no longer reports it (it used to overstate the UI by
    # hundreds of USDT once the engine started trading).
    web_app.state.balance = 8528.96
    web_app.state.get_price = lambda symbol: 84242.0
    import asyncio

    from db.database import load_sim_balance as _load_bal
    expected_balance = asyncio.run(_load_bal(web_app.state.config.db_path))
    try:
        r = client.get("/api/account")
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == {"balance", "available", "frozen", "equity", "positions_value",
                             "unrealized_pnl", "positions", "pending_count", "mode",
                             # added by the fee/slippage contract §五之二
                             "fees_paid_total", "slippage_paid_total", "net_pnl_total"}
        assert body["balance"] == pytest.approx(expected_balance, abs=0.01)
        assert web_app.state.balance == pytest.approx(expected_balance, abs=0.01), \
            "the endpoint must refresh the cached balance from the DB"
        assert body["frozen"] == 0.0
        assert body["available"] == pytest.approx(expected_balance, abs=0.01)
        assert body["mode"] == "sim"
        assert body["pending_count"] == 0
        pos = body["positions"][0]
        assert set(pos) == {"symbol", "side", "quantity", "entry_price", "current_price",
                            "unrealized_pnl", "pnl_pct", "position_value", "amount_usdt",
                            "stop_loss", "strategy_name", "position_type"}
        assert pos["current_price"] == 84242.0
        assert pos["unrealized_pnl"] == pytest.approx((84242.0 - 84055.6) * 0.00269, abs=1e-4)
        assert body["positions_value"] == pytest.approx(0.00269 * 84242.0, abs=0.01)
        assert body["equity"] == pytest.approx(body["balance"] + body["positions_value"], abs=0.02)
    finally:
        web_app.state.executor = None
        web_app.state.get_price = lambda symbol: None


def test_orders_and_history_shapes(client):
    body = client.get("/api/orders").json()
    assert set(body) == {"orders"}
    assert body["orders"] == []

    hist = client.get("/api/history/trades").json()
    assert set(hist) == {"trades"}
    assert hist["trades"], "the fixture inserts one closed trade"
    assert set(hist["trades"][0]) == {"id", "symbol", "side", "entry_price", "exit_price",
                                      "quantity", "pnl", "pnl_pct", "strategy", "exit_reason",
                                      "opened_at", "closed_at",
                                      # added by the fee/slippage contract §五之二
                                      "fee", "slippage", "fill_price", "net_pnl"}
    assert hist["trades"][0]["symbol"] == "BTCUSDT"

    assert client.get("/api/orders", params={"status": "nonsense"}).status_code == 400


# ======================================================================
# 二、auth: viewer reads, viewer cannot write
# ======================================================================
@pytest.mark.parametrize("path,params", [
    ("/api/market/ticker", {"symbol": "BTCUSDT"}),
    ("/api/market/depth", {"symbol": "BTCUSDT"}),
    ("/api/market/trades", {"symbol": "BTCUSDT"}),
    ("/api/market/overview", {}),
    ("/api/account", {}),
    ("/api/orders", {}),
    ("/api/history/trades", {}),
])
def test_viewer_can_read_every_new_endpoint(client, patched_exchange, path, params):
    assert client.get(path, params=params).status_code == 200, path


@pytest.mark.parametrize("path,params", [
    ("/api/market/ticker", {"symbol": "BTCUSDT"}),
    ("/api/market/depth", {"symbol": "BTCUSDT"}),
    ("/api/market/trades", {"symbol": "BTCUSDT"}),
    ("/api/market/overview", {}),
    ("/api/account", {}),
    ("/api/orders", {}),
    ("/api/history/trades", {}),
    ("/api/order", {}),
    ("/api/orders/1/cancel", {}),
])
def test_anonymous_is_rejected(web_app, path, params):
    c = TestClient(web_app)
    r = c.get(path, params=params) if not path.startswith("/api/orders/") else c.post(path)
    assert r.status_code == 401, f"{path} → {r.status_code}"


def test_viewer_cannot_place_or_cancel_orders(client):
    r = client.post("/api/order", data={
        "symbol": "BTCUSDT", "side": "long", "type": "market", "amount_usdt": 100})
    assert r.status_code == 403
    assert r.json() == {"error": "Forbidden"}

    r = client.post("/api/order", data={
        "symbol": "BTCUSDT", "side": "long", "type": "limit",
        "amount_usdt": 100, "price": 80000})
    assert r.status_code == 403

    r = client.post("/api/orders/1/cancel")
    assert r.status_code == 403


# ======================================================================
# 三、market order path reuses the risk pipeline
# ======================================================================
def test_market_order_publishes_order_request_after_risk_approval(web_app, trader_client):
    """The market route must reuse the risk pipeline, not re-implement it."""
    from app.event_bus import EventType

    rm = FakeRiskManager(approve=True, adjusted_quantity=None)
    web_app.state.risk_manager = rm
    web_app.state.get_price = lambda symbol: 84000.0

    # The route module closes over the EventBus created by create_app(); swap its
    # publish for a recorder so the test does not depend on the bus consumer task.
    bus = web_app.state.event_bus
    recorded = []
    original_publish = bus.publish

    async def _record(event):
        recorded.append(event)

    bus.publish = _record
    try:
        r = trader_client.post("/api/order", data={
            "symbol": "BTCUSDT", "side": "long", "type": "market",
            "amount_usdt": 840, "position_type": "satellite", "stop_loss_pct": 2.0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert set(body["order"]) == {"symbol", "side", "type", "quantity",
                                      "price", "amount_usdt", "status"}
        assert body["order"]["status"] == "filled"
        assert body["order"]["type"] == "market"
        assert body["order"]["quantity"] == pytest.approx(0.01, abs=1e-9)
        assert body["order"]["price"] == 84000.0
        # The signal went through check_signal with a computed stop loss.
        assert len(rm.signals) == 1
        assert rm.signals[0]["stop_loss"] == pytest.approx(84000.0 * 0.98, abs=0.01)
        # ... and was then handed to the executor via ORDER_REQUEST.
        assert [e.type for e in recorded] == [EventType.ORDER_REQUEST]
        assert recorded[0].data["quantity"] == pytest.approx(0.01, abs=1e-9)
        assert recorded[0].data["stop_loss"] == pytest.approx(84000.0 * 0.98, abs=0.01)
    finally:
        bus.publish = original_publish
        web_app.state.risk_manager = None
        web_app.state.get_price = lambda symbol: None


def test_market_order_risk_rejection_returns_structured_error(web_app, trader_client):
    rm = FakeRiskManager(approve=False, reason="Max open trades 8 reached")
    web_app.state.risk_manager = rm
    web_app.state.get_price = lambda symbol: 84000.0
    try:
        r = trader_client.post("/api/order", data={
            "symbol": "BTCUSDT", "side": "long", "type": "market", "amount_usdt": 100})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is False
        assert body["error"].startswith("风控拒绝:")
        assert "Max open trades" in body["error"]
    finally:
        web_app.state.risk_manager = None
        web_app.state.get_price = lambda symbol: None


def test_market_order_without_risk_manager_fails_closed(web_app, trader_client):
    web_app.state.risk_manager = None
    web_app.state.get_price = lambda symbol: 84000.0
    try:
        r = trader_client.post("/api/order", data={
            "symbol": "BTCUSDT", "side": "long", "type": "market", "amount_usdt": 100})
        assert r.status_code == 503
        assert r.json()["ok"] is False
    finally:
        web_app.state.get_price = lambda symbol: None


@pytest.mark.parametrize("data,needle", [
    ({"symbol": "BTCUSDT", "side": "sideways", "type": "market", "amount_usdt": 100}, "invalid side"),
    ({"symbol": "BTCUSDT", "side": "long", "type": "stop", "amount_usdt": 100}, "invalid type"),
    ({"symbol": "BTCUSDT", "side": "long", "type": "market", "amount_usdt": 0}, "amount_usdt"),
    ({"symbol": "BTCUSDT", "side": "long", "type": "market", "amount_usdt": -5}, "amount_usdt"),
    ({"symbol": "BTCUSDT", "side": "long", "type": "limit", "amount_usdt": 100}, "price"),
    ({"symbol": "BTCUSDT", "side": "long", "type": "limit", "amount_usdt": 100, "price": 0}, "price"),
])
def test_order_validation_errors_are_400(web_app, trader_client, data, needle):
    web_app.state.risk_manager = FakeRiskManager(approve=True)
    web_app.state.get_price = lambda symbol: 84000.0
    try:
        r = trader_client.post("/api/order", data=data)
        assert r.status_code == 400, r.text
        assert needle in r.json()["error"]
    finally:
        web_app.state.risk_manager = None
        web_app.state.get_price = lambda symbol: None


# ======================================================================
# 四、limit orders: place / freeze / cancel
# ======================================================================
def test_limit_order_placement_freezes_funds_without_deducting_cash(web_app, trader_client):
    web_app.state.balance = 10000.0
    r = trader_client.post("/api/order", data={
        "symbol": "BTCUSDT", "side": "long", "type": "limit",
        "amount_usdt": 200, "price": 80000, "position_type": "satellite",
        "stop_loss_pct": 2.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert set(body["order"]) == {"id", "symbol", "side", "type", "price",
                                 "quantity", "amount_usdt", "status"}
    assert body["order"]["status"] == STATUS_OPEN
    assert body["order"]["type"] == "limit"
    assert body["order"]["quantity"] == pytest.approx(0.0025, abs=1e-9)

    acc = trader_client.get("/api/account").json()
    assert acc["balance"] == 10000.0, "cash is not deducted at placement"
    assert acc["frozen"] == 200.0
    assert acc["available"] == 9800.0
    assert acc["pending_count"] == 1

    listed = trader_client.get("/api/orders", params={"status": "open"}).json()["orders"]
    assert len(listed) == 1
    assert set(listed[0]) == {"id", "symbol", "side", "type", "price", "quantity",
                              "amount_usdt", "status", "created_at", "filled_at",
                              "fill_price", "reason"}
    assert listed[0]["fill_price"] is None
    assert listed[0]["status"] == STATUS_OPEN

    # `all` and `history` agree about what is / is not in the book
    assert len(trader_client.get("/api/orders", params={"status": "all"}).json()["orders"]) == 1
    assert trader_client.get("/api/orders", params={"status": "history"}).json()["orders"] == []


def test_limit_order_rejected_when_available_balance_is_insufficient(web_app, trader_client):
    web_app.state.balance = 10000.0
    first = trader_client.post("/api/order", data={
        "symbol": "BTCUSDT", "side": "long", "type": "limit",
        "amount_usdt": 9900, "price": 80000})
    assert first.status_code == 200, first.text

    second = trader_client.post("/api/order", data={
        "symbol": "ETHUSDT", "side": "long", "type": "limit",
        "amount_usdt": 200, "price": 3000})
    assert second.status_code == 400
    assert "余额不足" in second.json()["error"]


def test_cancel_open_order_and_404_for_unknown(web_app, trader_client):
    placed = trader_client.post("/api/order", data={
        "symbol": "BTCUSDT", "side": "long", "type": "limit",
        "amount_usdt": 150, "price": 79000}).json()["order"]

    r = trader_client.post(f"/api/orders/{placed['id']}/cancel")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["order"]["status"] == STATUS_CANCELLED

    # Cancelling again is a conflict, not a silent success
    assert trader_client.post(f"/api/orders/{placed['id']}/cancel").status_code == 409
    assert trader_client.post("/api/orders/999999/cancel").status_code == 404
    assert trader_client.post("/api/orders/999999/cancel").json() == {"error": "order not found"}

    acc = trader_client.get("/api/account").json()
    assert acc["frozen"] == 0.0 and acc["pending_count"] == 0


# ======================================================================
# matcher unit tests (no HTTP, no network)
# ======================================================================
def _matcher_for(tmp_path, risk_manager, prices, market_data=None, event_bus=None):
    store = PendingOrderStore(str(tmp_path / "matcher.db"))
    app, fake_prices = _make_app_state(prices=prices, risk_manager=risk_manager,
                                      market_data=market_data)
    fake_prices.update(prices)
    matcher = LimitOrderMatcher(app, Config.load("sim"), event_bus or RecordingBus(),
                                store, interval=5.0)
    matcher._running = True
    return matcher, store, app, fake_prices


@pytest.fixture()
def matcher_db(tmp_path):
    async def _init():
        await init_database(str(tmp_path / "matcher.db"))
    _run(_init())
    return tmp_path


@pytest.mark.parametrize("price,order_price,expected", [
    (79999.0, 80000.0, True),    # long: last <= price
    (80001.0, 80000.0, False),
    (80000.0, 80000.0, True),    # exact touch fills
])
def test_long_cross_logic(matcher_db, price, order_price, expected):
    from core.executor.pending_orders import PendingOrder
    order = PendingOrder(id=1, symbol="BTCUSDT", side="long", type="limit",
                         price=order_price, quantity=1, amount_usdt=order_price,
                         status=STATUS_OPEN)
    assert LimitOrderMatcher._crossed(order, price) is expected


@pytest.mark.parametrize("price,order_price,expected", [
    (80001.0, 80000.0, True),    # short: last >= price
    (79999.0, 80000.0, False),
    (80000.0, 80000.0, True),
])
def test_short_cross_logic(matcher_db, price, order_price, expected):
    from core.executor.pending_orders import PendingOrder
    order = PendingOrder(id=1, symbol="BTCUSDT", side="short", type="limit",
                         price=order_price, quantity=1, amount_usdt=order_price,
                         status=STATUS_OPEN)
    assert LimitOrderMatcher._crossed(order, price) is expected


def test_matcher_fills_crossed_limit_order_through_the_risk_pipeline(matcher_db):
    """Place → price crosses → filled, with an ORDER_REQUEST carrying the fill."""
    rm = FakeRiskManager(approve=True, adjusted_quantity=None)
    bus = RecordingBus()
    matcher, store, app, prices = _matcher_for(matcher_db, rm, {"BTCUSDT": 84000.0},
                                               event_bus=bus)

    async def _scenario():
        order = await store.add("BTCUSDT", "long", price=80000.0, quantity=0.0025,
                                amount_usdt=200.0, stop_loss_pct=2.0)
        assert (await matcher.evaluate_once()) == [], "not crossed yet — no fill"
        assert (await store.get(order.id)).status == STATUS_OPEN

        prices["BTCUSDT"] = 79900.0        # crosses below the limit
        outcomes = await matcher.evaluate_once()
        assert len(outcomes) == 1 and outcomes[0]["status"] == STATUS_FILLED

        filled = await store.get(order.id)
        assert filled.status == STATUS_FILLED
        assert filled.fill_price == 79900.0
        assert filled.filled_at is not None, "filled_at must be stamped"
        assert await store.frozen_total() == 0.0, "a filled order no longer freezes funds"

        signals = bus.signals()
        assert len(signals) == 1, "the fill must go out as ORDER_REQUEST"
        sig = signals[0]
        assert sig["symbol"] == "BTCUSDT" and sig["side"] == "long"
        assert sig["price"] == 79900.0
        assert sig["quantity"] == pytest.approx(0.0025)
        assert sig["stop_loss"] == pytest.approx(79900.0 * 0.98, abs=0.01)
        assert rm.signals and rm.signals[0]["order_type"] == "limit"

    _run(_scenario())


def test_matcher_short_order_fills_when_price_rises(matcher_db):
    rm = FakeRiskManager(approve=True)
    bus = RecordingBus()
    matcher, store, app, prices = _matcher_for(matcher_db, rm, {"BTCUSDT": 80000.0},
                                               event_bus=bus)

    async def _scenario():
        order = await store.add("BTCUSDT", "short", price=81000.0, quantity=0.001,
                                amount_usdt=81.0, stop_loss_pct=2.0)
        assert await matcher.evaluate_once() == []
        prices["BTCUSDT"] = 81500.0
        outcomes = await matcher.evaluate_once()
        assert outcomes[0]["status"] == STATUS_FILLED
        sig = bus.signals()[0]
        assert sig["side"] == "short"
        # short stop loss sits ABOVE the fill
        assert sig["stop_loss"] == pytest.approx(81500.0 * 1.02, abs=0.01)
        assert (await store.get(order.id)).status == STATUS_FILLED

    _run(_scenario())


def test_matcher_cancels_order_when_risk_rejects(matcher_db):
    rm = FakeRiskManager(approve=False, reason="Position already open for BTCUSDT")
    bus = RecordingBus()
    matcher, store, app, prices = _matcher_for(matcher_db, rm, {"BTCUSDT": 79900.0},
                                               event_bus=bus)

    async def _scenario():
        order = await store.add("BTCUSDT", "long", price=80000.0, quantity=0.0025,
                                amount_usdt=200.0, stop_loss_pct=2.0)
        outcomes = await matcher.evaluate_once()
        assert outcomes[0]["status"] == STATUS_CANCELLED
        cancelled = await store.get(order.id)
        assert cancelled.status == STATUS_CANCELLED
        assert "风控拒绝" in (cancelled.reason or "")
        assert "Position already open" in cancelled.reason
        assert bus.signals() == [], "a rejected fill must not place an order"
        assert await store.frozen_total() == 0.0, "funds are released on rejection"

    _run(_scenario())


def test_matcher_fails_closed_without_a_risk_manager(matcher_db):
    bus = RecordingBus()
    matcher, store, app, prices = _matcher_for(matcher_db, None, {"BTCUSDT": 79900.0},
                                               event_bus=bus)

    async def _scenario():
        order = await store.add("BTCUSDT", "long", price=80000.0, quantity=0.0025,
                                amount_usdt=200.0)
        outcomes = await matcher.evaluate_once()
        assert outcomes[0]["status"] == STATUS_CANCELLED
        assert "风控未就绪" in (await store.get(order.id)).reason
        assert bus.signals() == []

    _run(_scenario())


def test_matcher_falls_back_to_historical_last_close(matcher_db):
    """No engine price cached → use the last close from market_data."""
    import pandas as pd

    class FakeMarketData:
        calls = 0

        async def get_historical(self, symbol, interval, limit=200):
            FakeMarketData.calls += 1
            return pd.DataFrame({"close": [82000.0, 79500.0]})

    rm = FakeRiskManager(approve=True)
    bus = RecordingBus()
    matcher, store, app, prices = _matcher_for(matcher_db, rm, {}, market_data=FakeMarketData(),
                                               event_bus=bus)

    async def _scenario():
        order = await store.add("BTCUSDT", "long", price=80000.0, quantity=0.0025,
                                amount_usdt=200.0)
        outcomes = await matcher.evaluate_once()
        assert outcomes and outcomes[0]["status"] == STATUS_FILLED
        assert outcomes[0]["fill_price"] == 79500.0
        assert FakeMarketData.calls >= 1
        assert (await store.get(order.id)).status == STATUS_FILLED

    _run(_scenario())


def test_matcher_skips_symbols_with_no_price(matcher_db):
    rm = FakeRiskManager(approve=True)
    bus = RecordingBus()
    matcher, store, app, prices = _matcher_for(matcher_db, rm, {}, event_bus=bus)

    async def _scenario():
        await store.add("BTCUSDT", "long", price=80000.0, quantity=0.0025, amount_usdt=200.0)
        assert await matcher.evaluate_once() == []
        assert (await store.list(status="open"))[0].status == STATUS_OPEN

    _run(_scenario())


# ======================================================================
# restart survival + lifecycle
# ======================================================================
def test_open_orders_survive_a_restart_and_are_still_matched(matcher_db):
    """The pending book is the DB, so a 'restart' (new store + matcher) keeps it."""
    rm = FakeRiskManager(approve=True)
    bus = RecordingBus()

    async def _scenario():
        first_store = PendingOrderStore(str(matcher_db / "matcher.db"))
        kept = await first_store.add("BTCUSDT", "long", price=80000.0, quantity=0.0025,
                                     amount_usdt=200.0, stop_loss_pct=2.0)
        # --- simulate process restart: brand new store/matcher over the same file ---
        second_store = PendingOrderStore(str(matcher_db / "matcher.db"))
        restored = await second_store.open_orders()
        assert [o.id for o in restored] == [kept.id]
        assert restored[0].status == STATUS_OPEN
        assert await second_store.frozen_total() == 200.0

        app, prices = _make_app_state(prices={"BTCUSDT": 79900.0}, risk_manager=rm)
        matcher = LimitOrderMatcher(app, Config.load("sim"), bus, second_store, interval=5.0)
        matcher._running = True
        outcomes = await matcher.evaluate_once()
        assert outcomes[0]["status"] == STATUS_FILLED
        assert (await second_store.get(kept.id)).status == STATUS_FILLED
        assert len(bus.signals()) == 1

    _run(_scenario())


def test_matcher_start_restores_orders_and_stop_is_clean(matcher_db):
    rm = FakeRiskManager(approve=True)
    bus = RecordingBus()

    async def _scenario():
        store = PendingOrderStore(str(matcher_db / "matcher.db"))
        await store.add("BTCUSDT", "long", price=80000.0, quantity=0.0025, amount_usdt=200.0)

        app, prices = _make_app_state(prices={"BTCUSDT": 90000.0}, risk_manager=rm)
        matcher = LimitOrderMatcher(app, Config.load("sim"), bus, store, interval=0.01)
        await matcher.start()
        assert matcher._task is not None and not matcher._task.done()
        await asyncio.sleep(0.05)          # let at least one pass run
        await matcher.stop()
        assert matcher._task is None
        assert matcher._running is False
        # Not crossed (90000 > 80000), so the order is still open and usable.
        assert (await store.list(status="open"))[0].status == STATUS_OPEN
        await matcher.stop()               # idempotent

    _run(_scenario())


def test_matcher_fills_after_start(matcher_db):
    """End-to-end through start(): crossed order is filled by the loop itself."""
    rm = FakeRiskManager(approve=True)
    bus = RecordingBus()

    async def _scenario():
        store = PendingOrderStore(str(matcher_db / "matcher.db"))
        order = await store.add("BTCUSDT", "long", price=80000.0, quantity=0.0025,
                                amount_usdt=200.0)
        app, prices = _make_app_state(prices={"BTCUSDT": 79000.0}, risk_manager=rm)
        matcher = LimitOrderMatcher(app, Config.load("sim"), bus, store, interval=0.01)
        await matcher.start()
        for _ in range(100):
            await asyncio.sleep(0.01)
            if (await store.get(order.id)).status == STATUS_FILLED:
                break
        await matcher.stop()
        assert (await store.get(order.id)).status == STATUS_FILLED
        assert len(bus.signals()) == 1

    _run(_scenario())


def test_start_matcher_task_is_a_noop_without_a_running_loop(web_app):
    """Sync contexts must not explode (used by scripts/CLI callers)."""
    from core.executor.pending_orders import start_matcher_task
    assert start_matcher_task(web_app, web_app.state.config, EventBus()) is None


def test_start_then_immediate_stop_leaves_no_task(web_app):
    """Regression: start used to await before creating the loop task, so a
    shutdown that arrived during start-up cancelled the start coroutine itself
    and silently left no matcher running."""
    from core.executor.pending_orders import start_matcher_task, stop_matcher

    async def _scenario():
        task = start_matcher_task(web_app, web_app.state.config, EventBus())
        try:
            assert task is not None and not task.done()
            await stop_matcher(web_app)          # immediately, no yield in between
            assert task.done()
            matcher = web_app.state.limit_order_matcher
            assert matcher._running is False
            assert matcher._task is None
            await stop_matcher(web_app)          # idempotent
        finally:
            await stop_matcher(web_app)
            web_app.state.__dict__.pop("limit_order_matcher", None)
            web_app.state.__dict__.pop("pending_order_store", None)

    _run(_scenario())


def test_pending_orders_table_exists_with_the_contract_columns(web_app):
    import aiosqlite

    async def _cols():
        db = await aiosqlite.connect(web_app.state.config.db_path)
        try:
            cursor = await db.execute("PRAGMA table_info(pending_orders)")
            return [r[1] for r in await cursor.fetchall()]
        finally:
            await db.close()

    cols = _run(_cols())
    for expected in ("id", "symbol", "side", "type", "price", "quantity", "amount_usdt",
                     "status", "created_at", "filled_at", "fill_price", "reason"):
        assert expected in cols, f"pending_orders is missing {expected}"


def test_schema_version_was_bumped_to_2(web_app):
    import aiosqlite

    async def _version():
        db = await aiosqlite.connect(web_app.state.config.db_path)
        try:
            cursor = await db.execute(
                "SELECT value FROM system_config WHERE key='schema_version'")
            row = await cursor.fetchone()
            return int(row[0])
        finally:
            await db.close()

    assert _run(_version()) >= 2
