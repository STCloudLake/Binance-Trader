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

    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test.db")
    config.db_path = db_path
    config.data_dir = tmp_dir  # isolate from project config
    # AlertManager writes rules to data_dir/../config/ — create parent path
    os.makedirs(os.path.join(os.path.dirname(tmp_dir), "config"), exist_ok=True)
    await init_database(db_path)

    am = AlertManager(config, bus)
    await am.start()

    await bus.publish(Event(EventType.ALERT_TRIGGER, {
        "level": "warning", "type": "emergency_stop", "message": "Test alert message", "symbol": "BTCUSDT",
    }))
    await asyncio.sleep(0.2)

    alerts = await am.get_alerts(limit=10)
    assert len(alerts) > 0
    assert alerts[0]["message"] == "Test alert message"

    await am.stop()
    await bus.shutdown()
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_alert_manager_rules():
    from alerts.manager import AlertManager
    from db.database import init_database

    Config._instance = None
    config = Config.load("sim")
    bus = EventBus()
    await bus.start()

    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test.db")
    config.db_path = db_path
    config.data_dir = tmp_dir  # isolate from project config
    # AlertManager writes rules to data_dir/../config/ — create parent path
    os.makedirs(os.path.join(os.path.dirname(tmp_dir), "config"), exist_ok=True)
    await init_database(db_path)

    am = AlertManager(config, bus)
    await am.start()

    idx = await am.add_rule({
        "name": "test_price_rule",
        "event_type": "alert.trigger",
        "condition": {"symbol": "BTCUSDT"},
        "level": "warning",
        "cooldown_seconds": 60,
    })
    assert idx == 5  # First custom rule after 5 defaults

    rules = am.get_rules()
    assert len(rules) == 6

    await am.remove_rule(idx)
    assert len(am.get_rules()) == 5

    await am.stop()
    await bus.shutdown()
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
