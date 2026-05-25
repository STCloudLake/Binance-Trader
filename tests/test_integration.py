"""Integration tests — end-to-end signal flow through all components (sim mode)."""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

from app.event_bus import EventBus, Event, EventType
from app.config import Config


@pytest.mark.asyncio
async def test_full_signal_flow_sim():
    """Test the full high-frequency signal flow in sim mode:
    MarketData → StrategyEngine → RiskManager → OrderExecutor"""
    from db.database import init_database
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager

    Config._instance = None
    config = Config.load("sim")
    db_path = tempfile.mktemp(suffix=".db")
    config.db_path = db_path
    await init_database(db_path)

    bus = EventBus()
    await bus.start()

    risk = RiskManager(config, bus)
    risk.update_balance(10000)
    await risk.start()

    executor = OrderExecutor(config, bus)
    await executor.start()

    order_events = []
    async def capture(event): order_events.append(event.data)
    bus.subscribe(EventType.ORDER_UPDATE, capture)

    # Simulate a strategy signal followed by risk check → order
    await bus.publish(Event(EventType.ORDER_REQUEST, {
        "symbol": "BTCUSDT", "side": "long", "price": 50000,
        "quantity": 0.01, "stop_loss": 49000,
        "position_type": "core",
    }))
    await asyncio.sleep(0.2)

    assert len(order_events) > 0
    assert order_events[0]["status"] == "filled"

    positions = executor.get_open_positions()
    assert "BTCUSDT" in positions

    await executor.stop()
    await risk.stop()
    await bus.shutdown()
    os.unlink(db_path)


@pytest.mark.asyncio
async def test_risk_rejection_on_tripped_breaker():
    """Test that signals are rejected when circuit breaker is tripped."""
    from db.database import init_database
    from core.risk.manager import RiskManager

    Config._instance = None
    config = Config.load("sim")
    db_path = tempfile.mktemp(suffix=".db")
    config.db_path = db_path
    await init_database(db_path)

    bus = EventBus()
    await bus.start()

    risk = RiskManager(config, bus)
    risk.update_balance(10000)
    risk.breaker.set_equity(9400)  # Trigger 6% drawdown
    await risk.start()

    risk_events = []
    async def capture(event): risk_events.append(event.data)
    bus.subscribe(EventType.RISK_BREACH, capture)

    result = await risk.check_signal({
        "symbol": "BTCUSDT", "side": "long", "price": 50000,
    })
    assert not result.approved
    assert "Circuit breaker" in result.reason

    await risk.stop()
    await bus.shutdown()
    os.unlink(db_path)


@pytest.mark.asyncio
async def test_same_symbol_rejection():
    """Test that opening a position on the same symbol is rejected."""
    from db.database import init_database
    from core.risk.manager import RiskManager

    Config._instance = None
    config = Config.load("sim")
    db_path = tempfile.mktemp(suffix=".db")
    config.db_path = db_path
    await init_database(db_path)

    bus = EventBus()
    await bus.start()

    risk = RiskManager(config, bus)
    risk.update_balance(20000)
    risk._open_positions["BTCUSDT"] = {"position_value": 1000}
    await risk.start()

    result = await risk.check_signal({
        "symbol": "BTCUSDT", "side": "long", "price": 50000,
    })
    assert not result.approved
    assert "already open" in result.reason

    await risk.stop()
    await bus.shutdown()
    os.unlink(db_path)


def test_vibe_connector_offline():
    from core.ai.vibe_connector import VibeTradingConnector
    from app.event_bus import EventBus
    Config._instance = None
    config = Config.load("sim")
    bus = EventBus()
    conn = VibeTradingConnector(config, bus)
    assert conn is not None
    assert not conn.is_available()  # Not installed in test environment
