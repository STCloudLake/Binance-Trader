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
    position_events = []

    async def on_order(event): order_events.append(event.data)
    async def on_position(event): position_events.append(event.data)

    bus.subscribe(EventType.ORDER_UPDATE, on_order)
    bus.subscribe(EventType.POSITION_UPDATE, on_position)

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
