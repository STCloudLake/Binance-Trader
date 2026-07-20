import pytest
import asyncio
from app.event_bus import EventBus, Event, EventType
from app.config import Config


@pytest.mark.asyncio
async def test_sim_order_execution():
    import tempfile, os
    from core.executor.executor import OrderExecutor

    Config._instance = None
    config = Config.load("sim")
    # Isolate from production database
    tmp_dir = tempfile.mkdtemp()
    config.db_path = os.path.join(tmp_dir, "test.db")
    from db.database import init_database
    await init_database(config.db_path)

    bus = EventBus()
    await bus.start()

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
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
