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
