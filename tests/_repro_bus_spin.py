"""Scratch measurement: event bus ``_process`` under a cross-loop queue (defect 2).

Loop A parks a live ``queue.get()`` waiter on the bus queue; loop B then runs
either the pre-fix body (verbose retry, no yield) or the fixed one, for a bounded
window.  The pre-fix body never yields, so its window is enforced by a deadline
*inside* the reimplemented body; the fixed body is the real method, stopped by
``_running = False`` from a deadline task.  ``fair_turns`` counts how often a fair
counter task got to run in the same window.
"""
import asyncio
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import event_bus as eb  # noqa: E402

WINDOW = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0


async def _old_process(self, deadline):
    """Exactly the pre-fix body, plus a deadline so the harness can finish."""
    while self._running and time.perf_counter() < deadline:
        try:
            event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            subscribers = list(self._subscribers.get(event.type, []))
            tasks = [cb(event) for cb in subscribers]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            eb.logger.warning(f"EventBus _process error: {e}")


async def measure(bus, label):
    lines = {"n": 0}
    fairness = {"turns": 0}
    eb.logger.remove()
    sink = eb.logger.add(lambda m: lines.__setitem__("n", lines["n"] + 1), level="WARNING")

    async def _fair_counter():
        while True:
            fairness["turns"] += 1
            await asyncio.sleep(0)

    async def _deadline():
        while time.perf_counter() < stop_at:
            await asyncio.sleep(0.01)
        bus._running = False

    started_at = time.perf_counter()
    stop_at = started_at + WINDOW
    cpu0 = time.process_time()
    bus._running = True
    if label == "before":
        bus_task = asyncio.create_task(_old_process(bus, stop_at))
    else:
        bus_task = asyncio.create_task(bus._process())
    fair_task = asyncio.create_task(_fair_counter())
    deadline_task = asyncio.create_task(_deadline())
    try:
        await asyncio.wait_for(asyncio.shield(bus_task), timeout=WINDOW + 5.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    except BaseException:
        pass
    for t in (bus_task, fair_task, deadline_task):
        t.cancel()
    wall = time.perf_counter() - started_at
    cpu = time.process_time() - cpu0
    await asyncio.sleep(0)
    eb.logger.remove(sink)
    eb.logger.add(sys.stderr)
    print(f"RESULT[{label}] wall={wall:.3f}s error_branch_runs={lines['n']} "
          f"runs_per_sec={lines['n']/wall:.0f} cpu={cpu:.3f}s fair_turns={fairness['turns']}")
    sys.stdout.flush()


def _park_waiter_on(bus):
    started = threading.Event()

    def _loop_a_waiter():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_until_complete(bus._queue.get())   # parks a waiter on loop A

    threading.Thread(target=_loop_a_waiter, daemon=True).start()
    started.wait()


async def main():
    for label in ("before", "after"):
        bus = eb.EventBus()
        _park_waiter_on(bus)
        await measure(bus, label)


asyncio.run(main())
os._exit(0)
