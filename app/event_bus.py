import asyncio
from typing import Any, Callable, Awaitable
from collections import defaultdict
from enum import Enum
from loguru import logger


class EventType(str, Enum):
    MARKET_TICK = "market.tick"
    MARKET_KLINE = "market.kline"
    STRATEGY_SIGNAL = "strategy.signal"
    ML_PREDICTION = "ml.prediction"
    NEWS_UPDATE = "news.update"
    NEWS_ALERT = "news.alert"
    RISK_CHECK = "risk.check"
    RISK_BREACH = "risk.breach"
    ORDER_REQUEST = "order.request"
    ORDER_UPDATE = "order.update"
    POSITION_UPDATE = "position.update"
    POSITION_EXIT = "position.exit"
    POSITION_REDUCE = "position.reduce"
    ALERT_TRIGGER = "alert.trigger"
    AI_SUGGESTION = "ai.suggestion"
    AI_MARKET_STATE = "ai.market_state"
    SYSTEM_SHUTDOWN = "system.shutdown"


class Event:
    def __init__(self, event_type: EventType, data: dict[str, Any] | None = None):
        self.type = event_type
        self.data = data or {}

    def __repr__(self):
        return f"Event({self.type.value}, data={self.data})"


class EventBus:
    def __init__(self):
        self._subscribers: dict[EventType, list[Callable[[Event], Awaitable[None]]]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=10000)
        self._running = False
        self._task: asyncio.Task | None = None

    def subscribe(self, event_type: EventType, callback: Callable[[Event], Awaitable[None]]):
        self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable[[Event], Awaitable[None]]):
        """Subscribe to ALL event types. Used by AlertManager rule engine."""
        for event_type in EventType:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], Awaitable[None]]):
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    async def publish(self, event: Event):
        await self._queue.put(event)

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._process())

    async def _process(self):
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                subscribers = self._subscribers.get(event.type, [])
                tasks = [cb(event) for cb in subscribers]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning(f"EventBus _process error: {e}")

    async def shutdown(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
