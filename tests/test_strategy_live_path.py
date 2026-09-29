"""Regression tests for the LIVE strategy evaluation path.

These tests exist because a refactor left `ml_weight` undefined in
`StrategyEngine._evaluate()`. Every evaluation raised NameError *before*
publishing anything, and the exception was swallowed by the caller's
`except Exception: pass`, so the whole system silently stopped trading while
135 unrelated tests kept passing.

The lesson encoded here: assert on the *observable side effects* of a real
`_evaluate()` call (signal cache + published events), not just on helpers.
"""
import asyncio

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from app.config import Config
from app.event_bus import Event, EventBus, EventType
from core.strategy.engine import StrategyEngine
from core.strategy.loader import MLConfig, StrategyConfig


class FakeMarketData:
    """Minimal stand-in for MarketDataProvider (no network, deterministic df)."""

    def __init__(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        self.df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": rng.random(n) * 10 + 1,
            },
            index=pd.date_range("2026-01-01", periods=n, freq="1h"),
        )
        self.watched_symbols = ["BTCUSDT"]

    async def get_historical(self, symbol, interval, limit=None):
        return self.df.copy()

    def get_current_price(self, symbol):
        return float(self.df["close"].iloc[-1])


def _collect_logs(level="ERROR"):
    """Attach a temporary loguru sink; returns (messages, detach)."""
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level=level)
    return messages, lambda: logger.remove(handler_id)


def _strategy(name="live_probe", long_cond="close > 0", short_cond="close < 0",
              ml_enabled=False, timeframes=("1h",)):
    return StrategyConfig(
        name=name,
        enabled=True,
        mode="trend",
        timeframes=list(timeframes),
        indicators={"rsi": {"period": 14, "source": "close"}},
        entry_conditions={"long": [long_cond], "short": [short_cond]},
        exit_conditions={"long": ["rsi > 100"], "short": ["rsi < 0"]},
        ml_config=MLConfig(enabled=ml_enabled),
    )


def _engine(ml_confidence=None):
    # Config is a singleton — reset so each test starts from file defaults.
    Config._instance = None
    config = Config.load("sim")
    bus = EventBus()
    engine = StrategyEngine(config, bus, FakeMarketData())
    if ml_confidence is not None:
        engine._ml_confidence["BTCUSDT"] = ml_confidence
    return engine, bus


@pytest.mark.asyncio
async def test_evaluate_populates_signal_cache():
    """The core regression: _evaluate must produce a cache entry, not raise."""
    engine, _ = _engine()
    strategy = _strategy()

    await engine._evaluate("BTCUSDT", "1h", strategy)

    assert "live_probe|BTCUSDT" in engine._signal_cache, (
        "_evaluate() produced no signal cache entry — the live signal path is broken"
    )
    entry = engine._signal_cache["live_probe|BTCUSDT"]
    assert entry["indicator_signal"] == 1.0
    assert entry["threshold_met"] is True
    # weights dict must report the effective ML weight, not a stale/undefined name
    assert entry["weights"]["ml"] == pytest.approx(engine.config.signal_weights.ml)


@pytest.mark.asyncio
async def test_evaluate_publishes_entry_signal():
    """A met entry condition must publish STRATEGY_SIGNAL with a usable payload."""
    engine, bus = _engine()
    await bus.start()
    received = []

    async def _collect(event):
        received.append(event)

    bus.subscribe(EventType.STRATEGY_SIGNAL, _collect)
    try:
        await engine._evaluate("BTCUSDT", "1h", _strategy())
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.01)
    finally:
        await bus.shutdown()

    assert received, "no STRATEGY_SIGNAL published — live entries would never fire"
    payload = received[0].data
    assert payload["symbol"] == "BTCUSDT"
    assert payload["side"] == "long"
    assert payload["confidence"] >= 0.5
    assert payload["price"] > 0


@pytest.mark.asyncio
async def test_evaluate_all_now_reports_failures():
    """A failing evaluation must be logged, not silently swallowed."""
    engine, _ = _engine()
    engine._strategies = {"boom": _strategy(name="boom")}
    engine.market_data.watched_symbols = ["BTCUSDT"]

    async def _explode(*args, **kwargs):
        raise RuntimeError("evaluation exploded")

    engine._evaluate = _explode  # type: ignore[assignment]

    messages, detach = _collect_logs()
    try:
        # Must not raise...
        await engine.evaluate_all_now(publish=True)
    finally:
        detach()

    assert any("evaluation exploded" in m or "FAILED" in m for m in messages), (
        "evaluate_all_now swallowed the failure without logging it"
    )


@pytest.mark.asyncio
async def test_ambiguous_long_short_produces_no_signal():
    """Both sides active → indicator forced to 0 (kernel guard)."""
    engine, _ = _engine()
    strategy = _strategy(long_cond="close > 0", short_cond="close > 0")
    await engine._evaluate("BTCUSDT", "1h", strategy)
    entry = engine._signal_cache["live_probe|BTCUSDT"]
    assert entry["indicator_signal"] == 0.0
    assert entry["threshold_met"] is False


@pytest.mark.asyncio
async def test_event_bus_logs_failing_subscriber():
    """EventBus must surface subscriber exceptions instead of dropping them."""
    bus = EventBus()
    await bus.start()

    async def _bad_subscriber(event):
        raise ValueError("subscriber blew up")

    bus.subscribe(EventType.MARKET_TICK, _bad_subscriber)

    messages, detach = _collect_logs()
    try:
        await bus.publish(Event(EventType.MARKET_TICK, {"symbol": "BTCUSDT"}))
        for _ in range(50):
            if messages:
                break
            await asyncio.sleep(0.01)
    finally:
        detach()
        await bus.shutdown()

    assert any("subscriber blew up" in m for m in messages), (
        "EventBus dropped a subscriber exception without logging it"
    )
