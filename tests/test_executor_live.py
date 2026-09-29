"""Regression tests for the LIVE order path: `OrderExecutor._execute_live()`.

`_execute_sim()` was covered by `tests/test_executor.py`, but the live path had
zero coverage even though it is the only code that can move real money. What is
pinned here:

  * long -> SIDE_BUY / short -> SIDE_SELL, always ORDER_TYPE_MARKET (the enums
    are imported from `binance.enums`, exactly like the production module)
  * on success: order record + in-memory position + ORDER_UPDATE/POSITION_UPDATE
  * transient failure -> retry (the 2**attempt backoff is monkeypatched away)
  * 3 failures -> ALERT_TRIGGER critical/order_failed and NO position
  * the order payload carries the rounded quantity

Everything runs against a fake client: no network, no API keys, no real DB.
"""
import asyncio
from contextlib import asynccontextmanager

import pytest
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL

from app.config import Config
from app.event_bus import Event, EventBus, EventType
from core.executor.executor import OrderExecutor

# Captured before any monkeypatching so event polling still really yields.
_REAL_SLEEP = asyncio.sleep


class FakeBinanceClient:
    """Records create_order payloads. Never opens a socket."""

    def __init__(self, failures_before_success=0, always_fail=False,
                 order_price="0", omit_status=False, bad_price=False):
        self.calls: list[dict] = []
        self.failures_before_success = failures_before_success
        self.always_fail = always_fail
        self.order_price = order_price
        self.omit_status = omit_status
        self.bad_price = bad_price
        self.closed = False
        self.duplicate_first = False
        self.lookups = 0

    async def create_order(self, **kwargs):
        self.calls.append(kwargs)
        if self.duplicate_first and len(self.calls) == 1:
            # Binance duplicate-clientOrderId error
            raise RuntimeError("APIError(code=-2010): Duplicate order sent.")
        if self.always_fail or len(self.calls) <= self.failures_before_success:
            raise RuntimeError(f"binance create_order failed (call {len(self.calls)})")
        order = {
            "orderId": 90000 + len(self.calls),
            # bad_price simulates a malformed/unparseable response field, which makes
            # the post-fill bookkeeping raise (see the no-resubmit test).
            "price": "not-a-number" if self.bad_price else self.order_price,
            "symbol": kwargs["symbol"],
        }
        if not self.omit_status:  # the order IS accepted either way
            order["status"] = "FILLED"
        return order

    async def get_order(self, **kwargs):
        """Look an order up by our client order id (used on duplicate detection)."""
        self.lookups += 1
        return {
            "orderId": 90001,
            "price": "0",
            "symbol": kwargs["symbol"],
            "status": "FILLED",
            "clientOrderId": kwargs.get("origClientOrderId"),
        }

    async def close_connection(self):
        self.closed = True


class EventCollector:
    """Subscribes to specific EventTypes and lets tests await delivery."""

    def __init__(self, bus: EventBus, *event_types: EventType):
        self.events = {t: [] for t in event_types}
        for event_type in event_types:
            bus.subscribe(event_type, self._handler(event_type))

    def _handler(self, event_type):
        async def _collect(event: Event):
            self.events[event_type].append(event.data)
        return _collect

    async def wait_for(self, event_type: EventType, n: int = 1, timeout: float = 2.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while len(self.events[event_type]) < n and loop.time() < deadline:
            await _REAL_SLEEP(0.01)
        return self.events[event_type]


def _live_config(tmp_path):
    # Config is a singleton — reset so the mode override cannot leak.
    Config._instance = None
    config = Config.load("sim")
    config.mode = "live"
    config.db_path = str(tmp_path / "live_executor.db")
    # Belt and braces: `start()` must never be able to open a connection.
    config.binance_api_key = ""
    config.binance_api_secret = ""
    return config


def _order_request(symbol="BTCUSDT", side="long", **overrides):
    data = {
        "symbol": symbol,
        "side": side,
        "price": 50000.0,
        "quantity": 0.01,
        "stop_loss": 49000.0,
        "position_type": "core",
        "strategy_name": "live_probe",
        "strategy": "trend",
        "amount_usdt": 500.0,
    }
    data.update(overrides)
    return data


@asynccontextmanager
async def _live_executor(config, client):
    bus = EventBus()
    await bus.start()
    executor = OrderExecutor(config, bus)
    executor.client = client  # injected fake — start() is deliberately not called
    try:
        yield executor, bus
    finally:
        await bus.shutdown()


def _instant_sleep_factory(slept: list):
    """Replacement for asyncio.sleep that records the delay and yields once."""
    async def _instant_sleep(delay, *args, **kwargs):
        slept.append(delay)
        await _REAL_SLEEP(0)
    return _instant_sleep


@pytest.mark.asyncio
async def test_live_long_signal_maps_to_buy_market_order(tmp_path):
    config = _live_config(tmp_path)
    client = FakeBinanceClient()
    async with _live_executor(config, client) as (executor, _bus):
        await executor._execute_live(_order_request(side="long"))

    assert len(client.calls) == 1, "live long must place exactly one order"
    call = client.calls[0]
    assert call["side"] == SIDE_BUY
    assert call["type"] == ORDER_TYPE_MARKET
    assert call["symbol"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_live_short_signal_maps_to_sell_market_order(tmp_path):
    config = _live_config(tmp_path)
    client = FakeBinanceClient()
    async with _live_executor(config, client) as (executor, _bus):
        await executor._execute_live(_order_request(symbol="ETHUSDT", side="short"))

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["side"] == SIDE_SELL
    assert call["type"] == ORDER_TYPE_MARKET
    assert call["symbol"] == "ETHUSDT"


@pytest.mark.asyncio
async def test_live_success_registers_order_position_and_publishes_events(tmp_path):
    config = _live_config(tmp_path)
    client = FakeBinanceClient(order_price="0")  # exchange omits a fill price
    async with _live_executor(config, client) as (executor, bus):
        collector = EventCollector(bus, EventType.ORDER_UPDATE, EventType.POSITION_UPDATE)
        await executor._execute_live(_order_request())

        order_events = await collector.wait_for(EventType.ORDER_UPDATE)
        position_events = await collector.wait_for(EventType.POSITION_UPDATE)
        orders = executor.get_orders()
        positions = executor.get_open_positions()

    # --- order record -------------------------------------------------
    assert list(orders) == ["90001"]
    record = orders["90001"]
    assert record["symbol"] == "BTCUSDT"
    assert record["status"] == "filled"
    assert record["binance_order_id"] == 90001

    # --- in-memory position -------------------------------------------
    assert "BTCUSDT" in positions
    pos = positions["BTCUSDT"]
    assert pos["side"] == "long"
    assert pos["quantity"] == pytest.approx(0.01)
    # order price "0" -> falls back to the signal price
    assert pos["entry_price"] == pytest.approx(50000.0)
    assert pos["current_price"] == pytest.approx(50000.0)
    assert pos["stop_loss"] == pytest.approx(49000.0)
    assert pos["strategy_name"] == "live_probe"
    assert pos["strategy"] == "trend"
    assert pos["position_type"] == "core"
    assert pos["amount_usdt"] == pytest.approx(500.0)
    assert pos["position_value"] == pytest.approx(0.01 * 50000.0)
    assert pos["trade_group"]
    assert pos["unrealized_pnl"] == 0

    # --- events --------------------------------------------------------
    assert order_events, "no ORDER_UPDATE published on a successful live fill"
    assert order_events[0]["order_id"] == "90001"
    assert order_events[0]["status"] == "filled"
    assert order_events[0]["mode"] == "live"
    assert order_events[0]["symbol"] == "BTCUSDT"

    assert position_events, "no POSITION_UPDATE published on a successful live fill"
    assert position_events[0]["symbol"] == "BTCUSDT"
    assert position_events[0]["side"] == "long"
    assert position_events[0]["quantity"] == pytest.approx(0.01)
    assert position_events[0]["entry_price"] == pytest.approx(50000.0)
    assert position_events[0]["closed"] is False
    assert position_events[0]["pnl"] == 0


@pytest.mark.asyncio
async def test_live_prefers_exchange_fill_price_over_signal_price(tmp_path):
    config = _live_config(tmp_path)
    client = FakeBinanceClient(order_price="51234.5")
    async with _live_executor(config, client) as (executor, _bus):
        await executor._execute_live(_order_request(price=50000.0))
        positions = executor.get_open_positions()

    assert positions["BTCUSDT"]["entry_price"] == pytest.approx(51234.5)
    assert positions["BTCUSDT"]["current_price"] == pytest.approx(51234.5)


@pytest.mark.asyncio
async def test_live_retries_transient_failure_then_succeeds(tmp_path, monkeypatch):
    config = _live_config(tmp_path)
    client = FakeBinanceClient(failures_before_success=2)  # fails twice, then fills
    slept: list = []
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep_factory(slept))

    async with _live_executor(config, client) as (executor, bus):
        collector = EventCollector(bus, EventType.ORDER_UPDATE, EventType.ALERT_TRIGGER)
        await executor._execute_live(_order_request())
        await collector.wait_for(EventType.ORDER_UPDATE, n=1)
        positions = executor.get_open_positions()
        orders = executor.get_orders()

    assert len(client.calls) == 3, "must retry up to 3 attempts"
    assert slept == [1, 2], "backoff must be 2**attempt (1s, then 2s)"
    assert collector.events[EventType.ALERT_TRIGGER] == [], \
        "a recovered transient failure must not raise a critical alert"
    assert "BTCUSDT" in positions
    assert positions["BTCUSDT"]["quantity"] == pytest.approx(0.01)
    assert len(orders) == 1


@pytest.mark.asyncio
async def test_live_permanent_failure_alerts_and_registers_no_position(tmp_path, monkeypatch):
    config = _live_config(tmp_path)
    client = FakeBinanceClient(always_fail=True)
    slept: list = []
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep_factory(slept))

    async with _live_executor(config, client) as (executor, bus):
        collector = EventCollector(
            bus, EventType.ALERT_TRIGGER, EventType.ORDER_UPDATE, EventType.POSITION_UPDATE)
        await executor._execute_live(_order_request())
        alerts = await collector.wait_for(EventType.ALERT_TRIGGER)
        orders = executor.get_orders()
        positions = executor.get_open_positions()

    assert len(client.calls) == 3, "expected exactly 3 attempts before giving up"
    assert slept == [1, 2]
    assert len(alerts) == 1
    assert alerts[0]["level"] == "critical"
    assert alerts[0]["type"] == "order_failed"
    assert "3 attempts" in alerts[0]["message"]
    assert orders == {}, "a failed live order must not be recorded as an order"
    assert positions == {}, "a failed live order must not register a position"
    assert collector.events[EventType.POSITION_UPDATE] == []
    assert collector.events[EventType.ORDER_UPDATE] == []


@pytest.mark.asyncio
async def test_live_quantity_is_rounded_in_order_payload(tmp_path):
    config = _live_config(tmp_path)
    client = FakeBinanceClient()
    async with _live_executor(config, client) as (executor, _bus):
        await executor._execute_live(_order_request(quantity=0.0123456789))
        positions = executor.get_open_positions()

    assert client.calls[0]["quantity"] == pytest.approx(0.01235), \
        "exchange payload must carry the exchange-legal rounded quantity"
    # FIXED: the in-memory position now stores the SAME rounded quantity that was
    # sent to the exchange. Previously it kept the unrounded signal quantity, so an
    # exit could send a size violating the symbol's LOT_SIZE filter.
    assert positions["BTCUSDT"]["quantity"] == pytest.approx(0.01235)
    # amount_usdt still comes from the signal when the caller supplies it.
    assert positions["BTCUSDT"]["amount_usdt"] == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_live_every_attempt_reuses_one_client_order_id(tmp_path, monkeypatch):
    """Idempotency: retries must carry the same newClientOrderId."""
    config = _live_config(tmp_path)
    client = FakeBinanceClient(failures_before_success=1)
    slept: list = []
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep_factory(slept))

    async with _live_executor(config, client) as (executor, _bus):
        await executor._execute_live(_order_request())

    assert len(client.calls) == 2, "expected one failure then one success"
    ids = [c.get("newClientOrderId") for c in client.calls]
    assert ids[0] and ids[0] == ids[1], (
        "each attempt must reuse one client order id so the exchange can reject a "
        "duplicate instead of opening a second position"
    )


@pytest.mark.asyncio
async def test_live_post_fill_bookkeeping_error_does_not_resubmit(tmp_path, monkeypatch):
    """FIXED BUG regression: a bookkeeping failure after a fill must not resubmit.

    Previously the retry `try` block also wrapped the post-fill bookkeeping, so a
    fill followed by an exception (e.g. `order["status"]` missing) caused the loop
    to submit *another* market order — one signal produced 3 real fills, none of
    them tracked. The order submission and the bookkeeping are now separate: the
    retry loop only retries the network call, and a bookkeeping failure raises a
    critical `order_tracking_failed` alert instead of re-ordering.
    """
    config = _live_config(tmp_path)
    # The order is accepted by the exchange, but the response carries an
    # unparseable price so the post-fill bookkeeping raises.
    client = FakeBinanceClient(bad_price=True)
    slept: list = []
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep_factory(slept))

    async with _live_executor(config, client) as (executor, bus):
        collector = EventCollector(bus, EventType.ALERT_TRIGGER)
        await executor._execute_live(_order_request())
        alerts = await collector.wait_for(EventType.ALERT_TRIGGER)

    assert len(client.calls) == 1, (
        "a fill followed by a bookkeeping error must NOT be retried — "
        f"{len(client.calls)} market orders were submitted for a single signal"
    )
    assert slept == [], "no retry/backoff should happen after a successful fill"
    assert [a["type"] for a in alerts] == ["order_tracking_failed"], alerts
    assert alerts[0]["level"] == "critical"


@pytest.mark.asyncio
async def test_live_duplicate_client_order_id_is_adopted_not_duplicated(tmp_path, monkeypatch):
    """A timed-out-but-accepted order must be adopted, never re-submitted.

    Every attempt reuses one client order id, so the exchange answers a retry with
    a duplicate error (-2010). The executor then looks the order up and tracks it
    instead of opening a second position.
    """
    config = _live_config(tmp_path)
    client = FakeBinanceClient()
    client.duplicate_first = True  # first call raises a Binance duplicate error
    slept: list = []
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep_factory(slept))

    async with _live_executor(config, client) as (executor, bus):
        collector = EventCollector(bus, EventType.POSITION_UPDATE)
        await executor._execute_live(_order_request())
        await collector.wait_for(EventType.POSITION_UPDATE)

    assert len(executor.get_open_positions()) == 1
    # Both attempts carried the SAME client order id (that is what makes the
    # exchange able to reject the duplicate).
    ids = {c.get("newClientOrderId") for c in client.calls}
    assert len(ids) == 1 and None not in ids
    assert client.lookups == 1


@pytest.mark.asyncio
async def test_order_request_routes_to_live_path_in_live_mode(tmp_path):
    """The ORDER_REQUEST -> _execute_live wiring, not just the private method."""
    config = _live_config(tmp_path)
    client = FakeBinanceClient()
    async with _live_executor(config, client) as (executor, _bus):
        # start() was skipped, so drive the router entry point directly.
        await executor._on_order_request(Event(EventType.ORDER_REQUEST, _order_request()))
        positions = executor.get_open_positions()

    assert len(client.calls) == 1, "config.mode == 'live' must route to _execute_live"
    assert client.calls[0]["side"] == SIDE_BUY
    assert "BTCUSDT" in positions


@pytest.mark.asyncio
async def test_stop_closes_the_client_connection(tmp_path):
    config = _live_config(tmp_path)
    client = FakeBinanceClient()
    async with _live_executor(config, client) as (executor, _bus):
        await executor.stop()

    assert client.closed, "live client connection must be closed on stop()"
