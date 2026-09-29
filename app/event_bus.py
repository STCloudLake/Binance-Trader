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
        #: The loop the queue is bound to.  An ``asyncio.Queue`` binds to the loop
        #: of its first *waiter*: once a second loop touches a queue that still has
        #: a waiter parked on the first, every ``get()`` raises ``RuntimeError``.
        self._loop: asyncio.AbstractEventLoop | None = None
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

    def _bind_to_running_loop(self, force: bool = False) -> None:
        """Make sure ``_queue`` is usable from the loop running ``_process``.

        An ``asyncio.Queue`` must not be used from two loops.  Once a second loop
        touches a queue that still has a waiter parked on the first, *every*
        ``get()`` raises ``RuntimeError`` — including for the loop that ran first.
        Callers therefore pass ``force=True`` after a failed ``get()``.

        Rather than fight the stale queue, its pending events are carried over to
        a fresh one: the bus keeps its contract (no event silently dropped) and
        the new loop starts from a clean binding.  A waiter still parked on the
        old queue belongs to a loop that no longer drives this bus, so letting go
        of the old object is safe.
        """
        loop = asyncio.get_running_loop()
        if self._loop is loop and not force:
            return
        old = self._queue
        self._queue = asyncio.Queue(maxsize=10000)
        self._loop = loop
        while True:
            try:
                self._queue.put_nowait(old.get_nowait())
            except asyncio.QueueEmpty:
                break
            except Exception:  # noqa: BLE001 - a stale queue must never stop the bus
                break

    async def start(self):
        self._bind_to_running_loop()
        self._running = True
        self._task = asyncio.create_task(self._process())

    async def _process(self):
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                subscribers = list(self._subscribers.get(event.type, []))
                tasks = [cb(event) for cb in subscribers]
                if tasks:
                    # return_exceptions=True keeps one bad subscriber from killing the
                    # bus, but the exceptions MUST be surfaced — silently dropping them
                    # hid a total failure of the live signal path (see engine.py fix).
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for cb, res in zip(subscribers, results):
                        if isinstance(res, BaseException):
                            name = getattr(cb, "__qualname__", repr(cb))
                            logger.error(
                                f"EventBus subscriber {name} failed on "
                                f"{event.type.value}: {res!r}"
                            )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # The queue can be bound to another loop (a second loop touching
                # one bus), and a bare retry would then raise again immediately:
                # the branch turned into a 100% CPU log flood (200KB in ~15s).
                # Rebind to a clean queue and back off before retrying.
                logger.warning(f"EventBus _process error: {e}")
                try:
                    self._bind_to_running_loop(force=True)
                except Exception:  # noqa: BLE001 - never let recovery spin either
                    pass
                await asyncio.sleep(0.1)

    async def shutdown(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
