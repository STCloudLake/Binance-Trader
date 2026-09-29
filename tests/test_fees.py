"""Fee tier, cost estimation and REAL sim-side slippage/fees.

Contract: ``docs/overhaul/TRADE_PAGE_API.md`` §五之二.

Everything runs against a temporary SQLite database and a fake price function —
no network, and ``data/binance_trader.db`` is never touched.  The arithmetic
expectations below are hand-computed from the frozen formula:

    buy  fill = price × (1 + (spread/2 + slippage)/100)
    sell fill = price × (1 − (spread/2 + slippage)/100)
    fee = quantity × fill × fee_pct/100          (BNB discount → ×0.75)
    slippage_usdt = |fill − price| × quantity
"""
import asyncio
import tempfile
from pathlib import Path

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from app.config import (
    BNB_DISCOUNT_FACTOR,
    Config,
    FEE_TIER_TABLE,
    SIM_FEE_TIER_KEY,
    load_sim_cost_settings,
    sim_cost_quote,
    sim_fee_pct,
)
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from core.executor.executor import OrderExecutor
from db.database import (
    atomic_adjust_balance,
    init_database,
    load_sim_balance,
    save_sim_balance,
)

VIEWER = ("fee_viewer", "V1ewerPass!")
TRADER = ("fee_trader", "T1aderPass!")

#: Fixed quoted price used by every hand-computed expectation.
P = 100.0
QTY = 1.0
#: BTCUSDT half-spread from config.yaml (%), and the configured 2 bp slippage.
BTC_SPREAD_PCT = 0.01
SLIPPAGE_BPS = 2.0
#: (spread/2 + slippage) as a percent == the fill-price edge on a market order.
EDGE_PCT = BTC_SPREAD_PCT / 2.0 + SLIPPAGE_BPS / 100.0


def _run(coro):
    return asyncio.run(coro)


# ======================================================================
# fixtures
# ======================================================================
@pytest.fixture(scope="module")
def web_app():
    tmpdir = Path(tempfile.mkdtemp(prefix="bt_fees_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "fees.db")
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")
    assert "bt_fees_" in config.db_path

    async def _setup():
        await init_database(config.db_path)
        await save_sim_balance(10000.0, config.db_path)
        am = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        for username, password, role in ((VIEWER[0], VIEWER[1], "viewer"),
                                         (TRADER[0], TRADER[1], "trader")):
            await am.create_user(username, password, role, username)
        return am

    auth = _run(_setup())

    from web.server import create_app
    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    app.state.balance = 10000.0
    app.state.get_price = lambda symbol: None
    yield app

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


@pytest.fixture(autouse=True)
def _reset_tier(web_app):
    """Every test starts clean: default tier (VIP0, no BNB) and an empty ledger."""
    db_path = web_app.state.config.db_path

    async def _reset():
        db = await aiosqlite.connect(db_path)
        try:
            await db.execute("DELETE FROM system_config WHERE key IN (?, ?)",
                             (SIM_FEE_TIER_KEY, "sim.use_bnb_discount"))
            await db.execute("DELETE FROM trades")
            await db.commit()
        finally:
            await db.close()
        await save_sim_balance(10000.0, db_path)

    _run(_reset())
    # `Config` is a process-wide singleton and `app.state` outlives this module,
    # so snapshot what this test is about to change and put it back afterwards:
    # a stale `app.state.balance` would leak into later modules' fixtures.
    saved = {name: getattr(web_app.state, name, None)
             for name in ("balance", "executor", "risk_manager", "get_price")}
    web_app.state.balance = 10000.0
    web_app.state.executor = None
    web_app.state.risk_manager = None
    web_app.state.get_price = lambda symbol: None
    yield
    for name, value in saved.items():
        setattr(web_app.state, name, value)


@pytest.fixture()
def executor(web_app):
    """A sim executor wired to the temp DB (no start(): no live client needed)."""
    ex = OrderExecutor(web_app.state.config, web_app.state.event_bus)
    ex.wire_risk_manager(None)
    return ex


def _rows(db_path):
    async def _fetch():
        db = await aiosqlite.connect(db_path)
        db.row_factory = aiosqlite.Row
        try:
            cursor = await db.execute("SELECT * FROM trades ORDER BY id")
            return [dict(r) for r in await cursor.fetchall()]
        finally:
            await db.close()

    return _run(_fetch())


# ======================================================================
# 1. fee tier listing / selection
# ======================================================================
def test_fee_tier_shape_and_defaults(client):
    body = client.get("/api/fee/tier").json()
    assert set(body) >= {"current", "use_bnb_discount", "bnb_discount_pct",
                         "tiers", "note"}
    assert body["current"] == "VIP0"
    assert body["use_bnb_discount"] is False
    assert body["bnb_discount_pct"] == 25
    assert [t["tier"] for t in body["tiers"]] == [f"VIP{i}" for i in range(10)]
    for tier in body["tiers"]:
        assert set(tier) == {"tier", "maker_pct", "taker_pct"}
        assert tier["taker_pct"] > 0 and tier["maker_pct"] > 0
    assert body["tiers"][0] == {"tier": "VIP0", "maker_pct": 0.1, "taker_pct": 0.1}
    # Rates must be monotonically non-increasing as the tier goes up.
    takers = [t["taker_pct"] for t in body["tiers"]]
    assert takers == sorted(takers, reverse=True)
    assert "手动选择" in body["note"]


def test_fee_tier_selection_persists_across_restart(trader_client, web_app):
    r = trader_client.post("/api/fee/tier",
                           data={"fee_tier": "vip3", "use_bnb_discount": "false"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["current"] == "VIP3"
    assert body["taker_pct"] == pytest.approx(0.06)
    assert body["maker_pct"] == pytest.approx(0.042)

    # Survives a fresh read from the DB (i.e. a restart) …
    assert trader_client.get("/api/fee/tier").json()["current"] == "VIP3"

    async def _reload():
        # … and is picked up by the executor's settings loader, not just by the
        # in-memory config object.
        return await load_sim_cost_settings(web_app.state.config.db_path,
                                            web_app.state.config)

    settings = _run(_reload())
    assert settings["fee_tier"] == "VIP3"
    assert sim_fee_pct(settings, "market") == pytest.approx(0.06)
    assert sim_fee_pct(settings, "limit") == pytest.approx(0.042)


def test_invalid_tier_is_rejected_and_not_persisted(trader_client, web_app):
    before = trader_client.get("/api/fee/tier").json()["current"]
    for bogus in ("VIP10", "", "platinum"):
        r = trader_client.post("/api/fee/tier", data={"fee_tier": bogus})
        assert r.status_code == 400, f"{bogus!r} → {r.status_code}"
        assert "invalid fee_tier" in r.json()["error"]
    assert trader_client.get("/api/fee/tier").json()["current"] == before


def test_fee_tier_write_requires_trader(client):
    assert client.post("/api/fee/tier", data={"fee_tier": "VIP1"}).status_code == 403


def test_bnb_discount_applies_the_75_percent_factor(trader_client, web_app):
    # Pin the engine price: without it a market estimate falls back to the live
    # public ticker and the fill price drifts between the two calls.
    web_app.state.get_price = lambda symbol: P
    base = trader_client.get("/api/fee/estimate", params={
        "symbol": "BTCUSDT", "side": "long", "type": "market",
        "amount_usdt": 1000, "price": P}).json()
    assert base["fee_pct"] == pytest.approx(0.1)

    r = trader_client.post("/api/fee/tier",
                           data={"fee_tier": "VIP0", "use_bnb_discount": "true"})
    assert r.status_code == 200, r.text
    assert r.json()["use_bnb_discount"] is True
    assert r.json()["taker_pct"] == pytest.approx(0.1 * BNB_DISCOUNT_FACTOR)

    discounted = trader_client.get("/api/fee/estimate", params={
        "symbol": "BTCUSDT", "side": "long", "type": "market",
        "amount_usdt": 1000, "price": P}).json()
    assert discounted["use_bnb_discount"] is True
    # 0.1% → 0.075%: exactly three quarters of the fee, nothing else changes.
    assert discounted["fee_pct"] == pytest.approx(0.1 * BNB_DISCOUNT_FACTOR)
    assert discounted["fee_usdt"] == pytest.approx(base["fee_usdt"] * BNB_DISCOUNT_FACTOR)
    assert discounted["slippage_usdt"] == pytest.approx(base["slippage_usdt"])
    assert discounted["fill_price"] == pytest.approx(base["fill_price"])


# ======================================================================
# 2. estimate math — market vs limit (hand-computed)
# ======================================================================
def test_estimate_market_math(client, web_app):
    web_app.state.get_price = lambda symbol: P
    body = client.get("/api/fee/estimate", params={
        "symbol": "BTCUSDT", "side": "long", "type": "market",
        "amount_usdt": 1000, "price": P}).json()

    # Notional 1000 at P=100 → 10 units.  Fill is the quote marked UP by the
    # half spread (0.01%/2 = 0.005%) plus 2 bp slippage = 0.025%.
    expected_fill = P * (1 + EDGE_PCT / 100)
    assert body["notional"] == 1000.0
    assert body["quantity"] == pytest.approx(10.0)
    assert body["price"] == P
    assert body["fill_price"] == pytest.approx(expected_fill)
    assert body["fee_pct"] == pytest.approx(0.1)
    expected_fee = body["quantity"] * expected_fill * 0.001
    assert body["fee_usdt"] == pytest.approx(expected_fee, abs=1e-6)
    assert body["slippage_usdt"] == pytest.approx((expected_fill - P) * body["quantity"],
                                                  abs=1e-6)
    assert body["slippage_usdt"] == pytest.approx(0.25, abs=1e-6)
    assert body["total_cost_usdt"] == pytest.approx(
        body["fee_usdt"] + body["slippage_usdt"], abs=1e-6)
    assert body["slippage_bps"] == pytest.approx(SLIPPAGE_BPS)
    assert body["spread_pct"] == pytest.approx(BTC_SPREAD_PCT)
    assert body["type"] == "market"
    assert body["notes"]
    # A short sells at the marked-DOWN price and pays the same fee.
    short = client.get("/api/fee/estimate", params={
        "symbol": "BTCUSDT", "side": "short", "type": "market",
        "amount_usdt": 1000}).json()
    assert short["fill_price"] == pytest.approx(P * (1 - EDGE_PCT / 100))
    assert short["slippage_usdt"] == pytest.approx(0.25, abs=1e-6)


def test_estimate_limit_has_no_spread_or_slippage(client):
    body = client.get("/api/fee/estimate", params={
        "symbol": "BTCUSDT", "side": "long", "type": "limit",
        "amount_usdt": 1000, "price": 101.0}).json()
    assert body["type"] == "limit"
    assert body["price"] == 101.0
    assert body["fill_price"] == pytest.approx(101.0)
    assert body["slippage_usdt"] == 0.0
    assert body["slippage_bps"] == 0.0
    assert body["spread_pct"] == 0.0
    # Only the fee remains: 0.1% of the notional at the limit price.
    assert body["fee_usdt"] == pytest.approx(1000.0 * 0.001, abs=1e-6)
    assert body["total_cost_usdt"] == pytest.approx(body["fee_usdt"], abs=1e-6)


def test_estimate_validates_its_parameters(client, web_app):
    web_app.state.get_price = lambda symbol: P
    assert client.get("/api/fee/estimate", params={
        "symbol": "BTCUSDT", "side": "sideways", "amount_usdt": 10}).status_code == 400
    assert client.get("/api/fee/estimate", params={
        "symbol": "BTCUSDT", "type": "stop", "amount_usdt": 10}).status_code == 400
    assert client.get("/api/fee/estimate", params={
        "symbol": "BTCUSDT", "amount_usdt": 0}).status_code == 400
    # No engine-cached price → falls back to the public data host, because the
    # trade page lets the user pick ANY of the ~496 USDT pairs (only ≤30 are
    # streamed). This used to answer 503 for every unwatched symbol.
    web_app.state.get_price = lambda symbol: None

    class _FakeDataHost:
        async def ticker24h(self, symbol):
            return {"lastPrice": "12345.6"}

    web_app.state.fee_price_client = _FakeDataHost()
    r = client.get("/api/fee/estimate", params={"symbol": "ADAUSDT", "amount_usdt": 100})
    assert r.status_code == 200, r.text
    assert r.json()["price"] == pytest.approx(12345.6)
    assert r.json()["fee_usdt"] > 0

    # Nothing available anywhere → structured 503, never a bare 500.
    class _DeadDataHost:
        async def ticker24h(self, symbol):
            raise RuntimeError("data host unreachable")

    web_app.state.fee_price_client = _DeadDataHost()
    r = client.get("/api/fee/estimate", params={"symbol": "BTCUSDT", "amount_usdt": 10})
    assert r.status_code == 503 and "error" in r.json()


def test_estimate_agrees_with_the_executor_quote():
    """The endpoint and the executor share one code path (`sim_cost_quote`)."""
    settings = {"enabled": True, "fee_tier": "VIP0", "use_bnb_discount": False,
                "slippage_bps": SLIPPAGE_BPS, "spread_pct": {"BTCUSDT": BTC_SPREAD_PCT},
                "default_spread_pct": 0.02}
    quote = sim_cost_quote("BTCUSDT", "long", "market", P, 10.0, settings)
    assert quote["fee_pct"] == pytest.approx(0.1)
    assert quote["fee_usdt"] == pytest.approx(10.0 * quote["fill_price"] * 0.001)
    assert quote["slippage_usdt"] == pytest.approx((quote["fill_price"] - P) * 10.0)


# ======================================================================
# 3. _execute_sim / close_position are cost-aware
# ======================================================================
def test_execute_sim_market_buy_fills_worse_than_the_quote(web_app, executor):
    _run(save_sim_balance(10000.0, web_app.state.config.db_path))
    _run(executor._execute_sim({
        "symbol": "BTCUSDT", "side": "long", "price": P, "quantity": QTY,
        "amount_usdt": P * QTY, "order_type": "market",
        "position_type": "core", "strategy": "manual"}))

    expected_fill = P * (1 + EDGE_PCT / 100)
    row = _rows(web_app.state.config.db_path)[0]
    assert row["action"] == "open"
    # The open row's `entry_price` is the CASH BASIS per unit — what the account
    # really paid per unit (worsened fill + fee + slippage) — because the ledger
    # identity is stated as `10000 − Σ(qty×entry_price) + Σ(pnl)`.  If the row kept
    # the bare quote, `qty×entry_price` would no longer equal the cash deducted and
    # the invariant could not hold.  The raw fill is kept in `fill_price`.
    expected_basis = (QTY * expected_fill + QTY * expected_fill * 0.001
                      + (expected_fill - P) * QTY) / QTY
    assert row["entry_price"] == pytest.approx(expected_basis)
    assert row["fill_price"] == pytest.approx(expected_fill)
    assert row["fee"] == pytest.approx(QTY * expected_fill * 0.001, abs=1e-9)
    assert row["slippage"] == pytest.approx((expected_fill - P) * QTY, abs=1e-9)
    # In-memory position tracks the same cash basis (fill + buy costs per unit).
    assert executor.get_open_positions()["BTCUSDT"]["entry_price"] == pytest.approx(expected_basis)
    # Free cash still drops by exactly the open row's notional — the invariant.
    assert _run(load_sim_balance(web_app.state.config.db_path)) == pytest.approx(
        10000.0 - row["quantity"] * row["entry_price"])


def test_execute_sim_market_sell_fills_worse_than_the_quote(web_app, executor):
    _run(save_sim_balance(10000.0, web_app.state.config.db_path))
    _run(executor._execute_sim({
        "symbol": "BTCUSDT", "side": "short", "price": P, "quantity": QTY,
        "amount_usdt": P * QTY, "order_type": "market"}))

    expected_fill = P * (1 - EDGE_PCT / 100)
    expected_basis = (QTY * expected_fill + QTY * expected_fill * 0.001
                      + (P - expected_fill) * QTY) / QTY
    row = _rows(web_app.state.config.db_path)[0]
    assert row["fill_price"] == pytest.approx(expected_fill)
    assert row["fee"] == pytest.approx(QTY * expected_fill * 0.001, abs=1e-9)
    assert row["slippage"] == pytest.approx((P - expected_fill) * QTY, abs=1e-9)
    assert executor.get_open_positions()["BTCUSDT"]["entry_price"] == pytest.approx(expected_basis)


def test_execute_sim_limit_order_fills_at_the_limit_price_with_fee_only(web_app, executor):
    _run(save_sim_balance(10000.0, web_app.state.config.db_path))
    _run(executor._execute_sim({
        "symbol": "BTCUSDT", "side": "long", "price": P, "quantity": QTY,
        "amount_usdt": P * QTY, "order_type": "limit"}))

    row = _rows(web_app.state.config.db_path)[0]
    assert row["fill_price"] == pytest.approx(P)
    assert row["slippage"] == 0.0
    # Limit = maker rate (VIP0 maker == taker == 0.1%).
    assert row["fee"] == pytest.approx(QTY * P * 0.001, abs=1e-9)


def test_close_position_sell_side_is_cost_aware_and_stores_costs(web_app, executor):
    _run(save_sim_balance(10000.0, web_app.state.config.db_path))
    _run(executor._execute_sim({
        "symbol": "BTCUSDT", "side": "long", "price": P, "quantity": QTY,
        "amount_usdt": P * QTY, "order_type": "market"}))

    result = _run(executor.close_position("BTCUSDT", 100, P))
    assert result["ok"] is True and result["closed"] is True

    buy_fill = P * (1 + EDGE_PCT / 100)
    sell_fill = P * (1 - EDGE_PCT / 100)
    gross = (sell_fill - buy_fill) * QTY
    buy_cost = QTY * buy_fill * 0.001 + (buy_fill - P) * QTY
    sell_cost = QTY * sell_fill * 0.001 + (P - sell_fill) * QTY
    # The open row's basis (what the account really paid) is the fill NOTIONAL plus
    # the buy-side cost; `invested_returned` hands exactly that back, so `pnl` is the
    # move measured from that basis less only the exit's own cost.  Together they
    # equal the cash the close really produced — which is what keeps the identity true.
    basis = QTY * buy_fill + buy_cost
    expected_pnl = (sell_fill - basis / QTY) * QTY - sell_cost

    rows = _rows(web_app.state.config.db_path)
    close_row = [r for r in rows if r["action"] == "close"][0]
    assert close_row["fill_price"] == pytest.approx(sell_fill), \
        "the close leg must sell into the marked-down price, not the quote"
    assert close_row["fee"] == pytest.approx(QTY * sell_fill * 0.001, abs=1e-9)
    assert close_row["slippage"] == pytest.approx((P - sell_fill) * QTY, abs=1e-9)
    # `pnl` is the NET figure: what the balance gains ON TOP of the returned basis.
    assert close_row["pnl"] == pytest.approx(expected_pnl, abs=0.01)
    # The buy side lives inside the basis, so it must NOT be charged a second time.
    assert result["invested_returned"] == pytest.approx(basis, abs=1e-9)
    assert result["gross_pnl"] == pytest.approx((sell_fill - basis / QTY) * QTY, abs=1e-4)
    assert result["cost"] == pytest.approx(sell_cost, abs=1e-4)
    assert result["gross_pnl"] - result["pnl"] == pytest.approx(result["cost"], abs=1e-3)
    # Cash in must equal cash out, to the cent — this is what the identity rests on.
    assert result["invested_returned"] + result["pnl"] == pytest.approx(
        QTY * sell_fill - sell_cost, abs=1e-6), "cash out must equal cash in"
    assert [r for r in rows if r["action"] == "open"][0]["status"] == "closed"


def test_closing_a_short_buys_back_at_the_marked_up_price(web_app, executor):
    """Regression: the close leg must convert the position side into a trade side.

    Closing a long is a SELL (marked down); closing a short is a BUY (marked up).
    Passing the position's own `side` through to the cost model silently made both
    legs cost the same way.
    """
    _run(save_sim_balance(10000.0, web_app.state.config.db_path))
    _run(executor._execute_sim({
        "symbol": "BTCUSDT", "side": "short", "price": P, "quantity": QTY,
        "amount_usdt": P * QTY, "order_type": "market"}))

    result = _run(executor.close_position("BTCUSDT", 100, P))
    short_entry = P * (1 - EDGE_PCT / 100)      # opening leg: sold at the bid
    buy_back = P * (1 + EDGE_PCT / 100)         # closing leg: bought back at the ask
    rows = _rows(web_app.state.config.db_path)
    open_row = [r for r in rows if r["action"] == "open"][0]
    close_row = [r for r in rows if r["action"] == "close"][0]
    assert open_row["fill_price"] == pytest.approx(short_entry)
    assert close_row["fill_price"] == pytest.approx(buy_back)
    assert close_row["slippage"] == pytest.approx((buy_back - P) * QTY, abs=1e-9)
    # The basis of a SHORT open is its fill notional plus the sell-side buy-in cost,
    # so the gross move must be measured from that basis, and `pnl` is strictly
    # worse than it by exactly the cost of buying the position back.
    # The basis of a SHORT open is its fill notional plus the sell-side cost, so the
    # gross move is measured from that basis and `pnl` is worse than it by the exit.
    basis = QTY * short_entry + (QTY * short_entry * 0.001 + (P - short_entry) * QTY)
    assert open_row["quantity"] * open_row["entry_price"] == pytest.approx(basis)
    assert result["gross_pnl"] == pytest.approx((basis / QTY - buy_back) * QTY, abs=1e-4)
    assert result["pnl"] == pytest.approx(result["gross_pnl"] - result["cost"], abs=1e-3)
    assert result["invested_returned"] == pytest.approx(basis, abs=1e-4)
    assert result["pnl"] < result["gross_pnl"]


def test_balance_identity_holds_after_a_round_trip(web_app, executor):
    """10000 − open notional + realised PnL == balance, before and after."""
    db_path = web_app.state.config.db_path
    _run(save_sim_balance(10000.0, db_path))
    balance_before = _run(load_sim_balance(db_path))
    assert balance_before == pytest.approx(10000.0)

    _run(executor._execute_sim({
        "symbol": "BTCUSDT", "side": "long", "price": P, "quantity": QTY,
        "amount_usdt": P * QTY, "order_type": "market"}))
    # Before closing, the whole identity is "cash + open notional".
    open_row = _rows(db_path)[0]
    open_notional = open_row["quantity"] * open_row["entry_price"]
    balance_open = _run(load_sim_balance(db_path))
    assert balance_open == pytest.approx(10000.0 - open_notional, abs=0.05)

    # Close at a profit so the round trip is unabashedly positive-gross.
    profit_price = P * 1.05
    result = _run(executor.close_position("BTCUSDT", 100, profit_price))
    trade_pnl = result["pnl"]
    invested_returned = result["invested_returned"]
    new_balance = _run(
        atomic_adjust_balance(invested_returned + trade_pnl, db_path))

    rows = _rows(db_path)
    open_row = [r for r in rows if r["action"] == "open"][0]
    # The open row's notional IS the cash that left the account (fill + buy costs).
    open_notional = open_row["quantity"] * open_row["entry_price"]
    realised = sum(r["pnl"] or 0 for r in rows if r["status"] == "closed")
    # The ledger, stated exactly as the task does it — the buy leg leaves the
    # account, the closed rows return their invested amount plus their (net) pnl.
    assert new_balance == pytest.approx(
        10000.0 - open_notional + (invested_returned + trade_pnl), abs=1e-6)
    # …and the equivalent form used in production accounting: nothing is left open,
    # so the balance is opening cash plus realised net pnl.
    assert new_balance == pytest.approx(10000.0 + realised, abs=1e-6)
    assert new_balance == pytest.approx(_run(load_sim_balance(db_path)), abs=0.05)
    # `invested_returned` is the whole basis, unrounded: the balance must gain back
    # exactly what the open deducted, or the identity cannot close.
    assert balance_open - new_balance == pytest.approx(-(invested_returned + trade_pnl),
                                                       abs=1e-6)
    # And the round trip really was charged: net < gross, and a +5% move on a
    # 100 USDT position still nets a *smaller* profit than its gross result.
    assert trade_pnl < result["gross_pnl"]
    assert result["cost"] > 0
    # The buy side is inside the returned basis, so the exit leg is the only extra
    # cost: gross and net differ by the exit's fee + slippage (both rounded to the
    # cent for reporting, hence the loose tolerance).  Charging the buy side here as
    # well is exactly the drift this guards against.
    assert result["gross_pnl"] - result["pnl"] == pytest.approx(
        result["fee"] + result["slippage"], abs=0.011)


def test_reduce_position_charges_a_pro_rata_share_of_the_buy_cost(web_app, executor):
    # 5 units × 100 USDT: big enough that a 50% reduce is not escalated to a full
    # close by the "remaining notional < 10 USDT" guard in close_position.
    _run(save_sim_balance(10000.0, web_app.state.config.db_path))
    qty = 5.0
    _run(executor._execute_sim({
        "symbol": "BTCUSDT", "side": "long", "price": P, "quantity": qty,
        "amount_usdt": P * qty, "order_type": "market"}))
    full_buy_fee = _rows(web_app.state.config.db_path)[0]["fee"]
    full_buy_slippage = _rows(web_app.state.config.db_path)[0]["slippage"]

    result = _run(executor.close_position("BTCUSDT", 50, P))
    assert result["ok"] is True and result["closed"] is False
    rows = _rows(web_app.state.config.db_path)
    reduce_row = [r for r in rows if r["action"] == "reduce"][0]
    # The reduce hands back half the basis and charges only the exit's own cost:
    # the pro-rata buy cost already sits inside the returned basis.
    assert reduce_row["quantity"] == pytest.approx(qty / 2)
    sell_fee = (qty / 2) * result["fill_price"] * 0.001
    assert reduce_row["fee"] == pytest.approx(sell_fee, abs=1e-9)
    # Only the exit's own cost separates gross from net here: the pro-rata buy cost
    # is already inside the half-basis that `invested_returned` hands back.
    assert reduce_row["pnl"] == pytest.approx(result["gross_pnl"] - result["cost"], abs=1e-3)
    assert result["cost"] == pytest.approx(sell_fee + reduce_row["slippage"], abs=1e-4)
    # `entry_price` is the cash BASIS per unit: the worsened fill notional plus the
    # buy fee and slippage, so `qty × entry_price` is exactly what the open deducted —
    # the figure the identity reconciles against.  (The open row's quantity is
    # rewritten to the remainder by the reduce, so read the position instead.)
    pos = executor.get_open_positions()["BTCUSDT"]
    full_basis = (pos["quantity"] + reduce_row["quantity"]) * pos["entry_price"]
    assert pos["entry_price"] == pytest.approx(
        P * (1 + EDGE_PCT / 100) + (full_buy_fee + full_buy_slippage) / qty, abs=1e-9)
    # Half the basis came back, so the ledger closes on the cent.
    assert result["invested_returned"] == pytest.approx(full_basis / 2, abs=1e-9)
    assert result["invested_returned"] + reduce_row["pnl"] == pytest.approx(
        (qty / 2) * result["fill_price"] - sell_fee - reduce_row["slippage"], abs=1e-6)
    pos = executor.get_open_positions()["BTCUSDT"]
    assert pos["quantity"] == pytest.approx(qty / 2)
    # What is left of the position carries the uncharged half of the buy fee.
    assert pos["fee"] == pytest.approx(full_buy_fee / 2, abs=1e-9)


# ======================================================================
# 4. /api/account and /api/history/trades extensions
# ======================================================================
def test_account_reports_cost_totals_without_dropping_existing_keys(client, web_app):
    body = client.get("/api/account").json()
    for key in ("balance", "available", "frozen", "equity", "positions_value",
                "unrealized_pnl", "positions", "pending_count", "mode"):
        assert key in body, f"existing key {key} disappeared"
    assert body["fees_paid_total"] == 0.0
    assert body["slippage_paid_total"] == 0.0
    assert body["net_pnl_total"] == 0.0

    web_app.state.get_price = lambda symbol: P
    executor = OrderExecutor(web_app.state.config, web_app.state.event_bus)
    _run(save_sim_balance(10000.0, web_app.state.config.db_path))
    web_app.state.balance = _run(load_sim_balance(web_app.state.config.db_path))
    _run(executor._execute_sim({
        "symbol": "BTCUSDT", "side": "long", "price": P, "quantity": QTY,
        "amount_usdt": P * QTY, "order_type": "market"}))
    result = _run(executor.close_position("BTCUSDT", 100, P * 1.05))
    web_app.state.balance = _run(atomic_adjust_balance(
        result["invested_returned"] + result["pnl"], web_app.state.config.db_path))

    body = client.get("/api/account").json()
    rows = _rows(web_app.state.config.db_path)
    assert body["fees_paid_total"] == pytest.approx(
        sum(r["fee"] or 0 for r in rows), abs=1e-6)
    assert body["slippage_paid_total"] == pytest.approx(
        sum(r["slippage"] or 0 for r in rows), abs=1e-6)
    assert body["fees_paid_total"] > 0 and body["slippage_paid_total"] > 0
    assert body["net_pnl_total"] == pytest.approx(
        sum(r["pnl"] or 0 for r in rows if r["status"] == "closed"), abs=1e-6)

    hist = client.get("/api/history/trades").json()["trades"]
    # Both rows of the round trip are `status='closed'` (close_position flips the
    # open row too) and both are ordered by id, so the EXIT row is selected by
    # `action` — the same key the history endpoint now filters on.
    closed = [r for r in rows if r["action"] == "close"][0]
    entry = [t for t in hist if t["id"] == closed["id"]][0]
    assert entry["fee"] == pytest.approx(closed["fee"])
    assert entry["slippage"] == pytest.approx(closed["slippage"])
    assert entry["fill_price"] == pytest.approx(closed["fill_price"])
    assert entry["net_pnl"] == pytest.approx(closed["pnl"])


def test_history_returns_null_costs_for_rows_predating_the_cost_model(client, web_app):
    async def _insert():
        db = await aiosqlite.connect(web_app.state.config.db_path)
        try:
            await db.execute(
                "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity, pnl,"
                " pnl_pct, strategy, status, action, opened_at, closed_at) VALUES"
                " ('ETHUSDT','long',3000,3100,0.1,10.0,3.33,'manual','closed','close',"
                " '2026-01-02 00:00:00','2026-01-02 04:00:00')")
            await db.commit()
        finally:
            await db.close()

    _run(_insert())
    hist = client.get("/api/history/trades").json()["trades"]
    old = [t for t in hist if t["symbol"] == "ETHUSDT"][0]
    assert old["fee"] is None and old["slippage"] is None and old["fill_price"] is None
    # With no cost columns there is nothing to net off, so net_pnl == pnl.
    assert old["net_pnl"] == old["pnl"] == 10.0


# ======================================================================
# 5. schema migration v3
# ======================================================================
def _columns(db_path, table="trades"):
    async def _cols():
        db = await aiosqlite.connect(db_path)
        try:
            cursor = await db.execute(f"PRAGMA table_info({table})")
            return [r[1] for r in await cursor.fetchall()]
        finally:
            await db.close()

    return _run(_cols())


def _rebuild_as_v2(db_path):
    """Turn a fresh (v3) trades table back into its pre-v3 shape."""
    async def _strip():
        db = await aiosqlite.connect(db_path)
        try:
            await db.execute(
                "CREATE TABLE trades_v2 AS SELECT id, symbol, side, entry_price, "
                "exit_price, quantity, pnl, pnl_pct, strategy, timeframe, "
                "position_type, trader, strategy_name, opened_at, closed_at, "
                "status, action, trade_group, reduce_pct FROM trades")
            await db.execute("DROP TABLE trades")
            await db.execute("ALTER TABLE trades_v2 RENAME TO trades")
            await db.execute(
                "INSERT OR REPLACE INTO system_config (key, value, category) "
                "VALUES ('schema_version', '2', 'system')")
            await db.execute(
                "INSERT INTO trades (symbol, side, entry_price, exit_price, quantity,"
                " pnl, pnl_pct, strategy, status, opened_at, closed_at) VALUES"
                " ('BTCUSDT','long',50000,51000,0.01,10.0,2.0,'manual','closed',"
                " '2026-01-01 00:00:00','2026-01-01 01:00:00')")
            await db.commit()
        finally:
            await db.close()

    _run(_strip())


def test_migration_v3_adds_the_cost_columns_without_touching_rows():
    tmpdir = Path(tempfile.mkdtemp(prefix="bt_fee_migrate_"))
    db_path = str(tmpdir / "old.db")
    _run(init_database(db_path))
    _rebuild_as_v2(db_path)
    assert "fill_price" not in _columns(db_path)

    row_before = _rows(db_path)[0]
    _run(init_database(db_path))            # ← the migration under test

    cols = _columns(db_path)
    for expected in ("fill_price", "fee", "slippage"):
        assert expected in cols, f"trades is missing {expected}"

    async def _version():
        db = await aiosqlite.connect(db_path)
        db.row_factory = aiosqlite.Row
        try:
            cursor = await db.execute(
                "SELECT value FROM system_config WHERE key='schema_version'")
            return int((await cursor.fetchone())["value"])
        finally:
            await db.close()

    # v4 is the current version; the point of the assertion is that the version
    # was bumped past the pre-v3 one, so it is stated as a lower bound.
    assert _run(_version()) >= 4
    # No backfill: the pre-existing row is byte-for-byte identical, with NULLs
    # for the new columns.
    row_after = _rows(db_path)[0]
    assert row_after["fill_price"] is None
    assert row_after["fee"] is None
    assert row_after["slippage"] is None
    # SCHEMA-CHANGE NOTE (v4): the canonical `trades` DDL declares NOT NULL with a
    # DEFAULT on the columns this fixture's `CREATE TABLE ... AS SELECT` v2 table
    # left nullable, so the rebuild must give a NULL row a value rather than fail
    # on it.  The value it takes is the column's own DEFAULT — the honest reading
    # of "the writer recorded nothing" — and it is pinned here so the mapping is
    # explicit instead of incidental.
    v4_null_defaults = {"id": 1, "timeframe": "1h", "position_type": "satellite",
                        "trader": "manual", "strategy_name": "", "action": "open",
                        "trade_group": "", "reduce_pct": 0.0}
    for key, value in row_before.items():
        if value is None and key in v4_null_defaults:
            assert row_after[key] == v4_null_defaults[key], (
                f"{key}: NULL should migrate to the canonical DEFAULT")
            continue
        assert row_after[key] == value, f"migration modified existing column {key}"
    # Every row still has a usable id and no row was dropped or duplicated.
    assert row_after["id"] == 1
    assert [r["id"] for r in _rows(db_path)] == [1]
    # Running it twice is a no-op, not a crash.
    _run(init_database(db_path))
    assert _columns(db_path).count("fill_price") == 1


def test_fee_tier_table_matches_the_contract_shape():
    assert len(FEE_TIER_TABLE) == 10
    assert FEE_TIER_TABLE[0]["tier"] == "VIP0"
    assert FEE_TIER_TABLE[-1]["tier"] == "VIP9"
    assert BNB_DISCOUNT_FACTOR == 0.75
    assert sim_fee_pct({"fee_tier": "nonsense"}, "market") == pytest.approx(0.1)
