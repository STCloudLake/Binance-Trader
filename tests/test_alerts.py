import pytest
import asyncio
import tempfile
import os
from app.event_bus import EventBus, Event, EventType
from app.config import Config


@pytest.mark.asyncio
async def test_alert_manager_saves_alerts():
    from alerts.manager import AlertManager
    from db.database import init_database

    Config._instance = None
    config = Config.load("sim")
    bus = EventBus()
    await bus.start()

    db_path = tempfile.mktemp(suffix=".db")
    config.db_path = db_path
    await init_database(db_path)

    am = AlertManager(config, bus)
    await am.start()

    await bus.publish(Event(EventType.ALERT_TRIGGER, {
        "level": "warning", "type": "test", "message": "Test alert message", "symbol": "BTCUSDT",
    }))
    await asyncio.sleep(0.2)

    alerts = await am.get_recent_alerts(limit=10)
    assert len(alerts) > 0
    assert alerts[0]["message"] == "Test alert message"

    await am.stop()
    await bus.shutdown()
    os.unlink(db_path)


@pytest.mark.asyncio
async def test_alert_manager_rules():
    from alerts.manager import AlertManager
    from db.database import init_database

    Config._instance = None
    config = Config.load("sim")
    bus = EventBus()
    await bus.start()

    db_path = tempfile.mktemp(suffix=".db")
    config.db_path = db_path
    await init_database(db_path)

    am = AlertManager(config, bus)
    await am.start()

    idx = await am.add_rule({"type": "price", "symbol": "BTCUSDT", "condition": "above", "value": 60000})
    assert idx == 1  # First custom rule after default

    rules = am.get_rules()
    assert len(rules) == 2

    await am.remove_rule(idx)
    assert len(am.get_rules()) == 1

    await am.stop()
    await bus.shutdown()
    os.unlink(db_path)
