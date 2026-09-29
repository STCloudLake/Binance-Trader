"""DECOUPLING PASS 1 — regression tests for the de-hardcoding work.

Everything pinned here used to be welded to "5 symbols / 2 intervals / one
order path":

  * the AI's ``select_coins`` output now reaches the persisted watchlist
  * ``StrategyLifecycleManager`` runs its strategy×symbol matrix over the
    watchlist instead of a private five-symbol copy
  * the signal-matrix timeline is the UNION over every symbol's index (a
    late-listed symbol used to truncate or break the shared index)
  * the backtest regime proxy is configurable and falls back to a symbol that
    is actually in the run
  * live order quantities are floored to the symbol's LOT_SIZE step and the
    NOTIONAL minimum is enforced
  * one interval registry (``INTERVAL_SPEC``) drives the ML gate, the GA
    timeframe genes, the history prefetch and the REST-poll cadence
  * news queries ask for the base asset (``BTC``), not the exchange pair
  * ``trades.timeframe`` carries the signal's timeframe

No network: universes/exchanges are fakes, DBs are temp files, frames are
synthetic.  The HTTP tests run against ``create_app`` with the user injected
through middleware instead of a login round-trip.
"""
from __future__ import annotations

import inspect
import json
import random
import re
from pathlib import Path

import aiosqlite
import numpy as np
import pandas as pd
import pytest

from db.database import init_database

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "web" / "templates"

#: A duplicated watchlist *list* (not a lone ``symbol="BTCUSDT"`` default).
_WATCHLIST_LITERAL = re.compile(r"BTCUSDT['\"]?\s*,\s*['\"]?ETHUSDT")


# ======================================================================
# helpers / fakes
# ======================================================================

class FakeUniverse:
    """``core.market_data.universe.Universe`` stand-in: no network, no disk."""

    def __init__(self, symbols, step_size=0.001, tick_size=0.01,
                 min_qty=0.001, min_notional=10.0, status="TRADING"):
        from core.market_data.universe import SymbolInfo

        self._by_symbol = {}
        for sym in symbols:
            name = str(sym).upper()
            base = name
            for quote in ("USDT", "BTC", "ETH", "BNB"):
                if name.endswith(quote) and len(name) > len(quote):
                    base = name[: -len(quote)]
                    break
            self._by_symbol[name] = SymbolInfo(
                symbol=name, base_asset=base, quote_asset="USDT", status=status,
                step_size=step_size, tick_size=tick_size, min_qty=min_qty,
                min_notional=min_notional)

    def get(self, symbol):
        return self._by_symbol.get(str(symbol or "").upper())

    def get_symbols_cached_count(self):
        return len(self._by_symbol)


class FakeExchange:
    """Records ``create_order`` payloads; never opens a socket."""

    def __init__(self):
        self.calls: list[dict] = []

    async def create_order(self, **kwargs):
        self.calls.append(kwargs)
        return {"orderId": 4242, "price": "0", "status": "FILLED"}

    async def get_order(self, **kwargs):  # pragma: no cover - duplicate path
        return {"orderId": 4242, "price": "0", "status": "FILLED"}

    async def close_connection(self):
        pass


class AlertCollector:
    def __init__(self, bus):
        from app.event_bus import EventType

        self.alerts: list[dict] = []
        bus.subscribe(EventType.ALERT_TRIGGER, self._on)

    async def _on(self, event):
        self.alerts.append(event.data)

    async def wait(self, n: int = 1, timeout: float = 2.0) -> list[dict]:
        import asyncio

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while len(self.alerts) < n and loop.time() < deadline:
            await asyncio.sleep(0.01)
        return self.alerts


class FakeFeeder:
    """``core.backtest.data_feeder.DataFeeder`` stand-in over in-memory frames."""

    def __init__(self, frames: dict, date_start: str):
        self._frames = frames
        self.date_start = pd.Timestamp(date_start)

    def get_all_data_for_symbol(self, symbol: str, interval: str) -> pd.DataFrame:
        return self._frames.get(symbol, {}).get(interval, pd.DataFrame())


def _ohlcv(index, seed):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, len(index)))
    return pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": rng.random(len(index)) * 10 + 1,
    }, index=index)


@pytest.fixture()
def sim_config(tmp_path):
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmp_path / "decoupling.db")
    config.data_dir = str(tmp_path / "data")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return config


def _strategy(name, symbols=None):
    from core.strategy.loader import MLConfig, StrategyConfig

    return StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=["1h"],
        symbols=list(symbols or []),
        indicators={"rsi": {"period": 14, "source": "close"}},
        entry_conditions={"long": ["rsi < 30"], "short": ["rsi > 70"]},
        exit_conditions={"long": ["rsi > 70"], "short": ["rsi < 30"]},
        ml_config=MLConfig(enabled=False),
    )


# ======================================================================
# 1. select_coins() output reaches the watchlist
# ======================================================================

@pytest.mark.asyncio
async def test_ai_coin_selection_is_persisted_as_the_watchlist(sim_config):
    from app.event_bus import EventBus
    from core.ai.deepseek_ctl import DeepSeekController
    from core.market_data.universe import (
        DEFAULT_WATCHLIST, load_watchlist, save_watchlist)

    await init_database(sim_config.db_path)
    await save_watchlist(sim_config.db_path, list(DEFAULT_WATCHLIST))

    ctl = DeepSeekController(sim_config, EventBus())
    ctl.wire_universe(FakeUniverse(["ADAUSDT", "DOGEUSDT"]))

    saved = await ctl.apply_coin_selection(
        {"symbols": ["adausdt", "DOGEUSDT", "NOPEUSDT"]})

    assert saved == ["ADAUSDT", "DOGEUSDT"], \
        "the AI's symbols must be case-normalised and unknown pairs rejected"
    assert await load_watchlist(sim_config.db_path) == ["ADAUSDT", "DOGEUSDT"], \
        "the AI selection must actually reach the persisted watchlist"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    None,
    "a garbage string",
    12345,
    [],
    {"symbols": None},
    {"symbols": 3},
    {"symbols": {"BTCUSDT": 1}},
    {"symbols": [None, {}, 42, "   "]},
    {"symbols": ["NOPEUSDT"]},
    {"coins": []},
    {"result": "no symbols key at all"},
])
async def test_ai_coin_selection_never_clobbers_a_good_watchlist(sim_config, payload):
    from app.event_bus import EventBus
    from core.ai.deepseek_ctl import DeepSeekController
    from core.market_data.universe import load_watchlist, save_watchlist

    await init_database(sim_config.db_path)
    good = ["BTCUSDT", "ETHUSDT"]
    await save_watchlist(sim_config.db_path, good)

    ctl = DeepSeekController(sim_config, EventBus())
    ctl.wire_universe(FakeUniverse(["ADAUSDT", "DOGEUSDT"]))

    assert await ctl.apply_coin_selection(payload) == []
    assert await load_watchlist(sim_config.db_path) == good, \
        f"garbage AI payload {payload!r} must not touch the watchlist"


@pytest.mark.asyncio
async def test_ai_coin_selection_offline_accepts_picks(sim_config):
    """No universe available (offline/empty cache) → accept, never crash."""
    from app.event_bus import EventBus
    from core.ai.deepseek_ctl import DeepSeekController
    from core.market_data.universe import load_watchlist

    await init_database(sim_config.db_path)
    ctl = DeepSeekController(sim_config, EventBus())  # no injected universe

    assert await ctl.apply_coin_selection({"symbols": ["ADAUSDT"]}) == ["ADAUSDT"]
    assert await load_watchlist(sim_config.db_path) == ["ADAUSDT"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ai_mode,expected_write", [
    ("suggest", False), ("semi_auto", False), ("full_auto", True),
])
async def test_coin_selection_loop_applies_only_in_full_auto(
        sim_config, monkeypatch, ai_mode, expected_write):
    from app.event_bus import EventBus
    from core.ai.deepseek_ctl import DeepSeekController

    ctl = DeepSeekController(sim_config, EventBus())
    ctl.config.ai_mode = ai_mode
    ctl.config.ai_task_intervals["coin_selection"] = 0
    applied: list[dict] = []
    published: list[str] = []

    async def fake_select():
        return {"symbols": ["ADAUSDT"]}

    async def fake_apply(result):
        applied.append(result)
        return ["ADAUSDT"]

    async def fake_publish(category, content, confidence):
        published.append(category)

    async def fake_heartbeat(task_name, success):
        ctl._running = False  # one iteration only

    monkeypatch.setattr(ctl, "select_coins", fake_select)
    monkeypatch.setattr(ctl, "apply_coin_selection", fake_apply)
    monkeypatch.setattr(ctl, "_publish_suggestion", fake_publish)
    monkeypatch.setattr(ctl, "_heartbeat", fake_heartbeat)

    ctl._running = True
    await ctl._coin_selection_loop()

    assert published == ["coin_selection"], \
        "the suggestion/alert behaviour must be kept in every AI mode"
    assert bool(applied) is expected_write, \
        f"ai_mode={ai_mode} must {'apply' if expected_write else 'not apply'} the AI watchlist"


# ======================================================================
# 2. lifecycle matrix runs over the persisted watchlist
# ======================================================================

class _StubBacktest:
    def __init__(self, result=None):
        self.result = result or {}
        self.calls: list[dict] = []

    def run(self, strategies, symbols, date_start, date_end, mode, initial_balance):
        self.calls.append({"strategies": list(strategies), "symbols": list(symbols)})
        return dict(self.result)


class _StubDeepSeek:
    async def _call_deepseek(self, system_prompt, user_prompt):
        return None


class _StubStrategyEngine:
    def __init__(self):
        self._strategies: dict = {}


async def _lifecycle(sim_config, tmp_path, backtest_result):
    from core.ai.strategy_lifecycle import StrategyLifecycleManager
    from core.strategy.loader import StrategyLoader

    await init_database(sim_config.db_path)
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    loader = StrategyLoader(str(strategies_dir))
    backtest = _StubBacktest(backtest_result)
    engine = _StubStrategyEngine()
    manager = StrategyLifecycleManager(
        config=sim_config, deepseek_ctl=_StubDeepSeek(),
        backtest_engine=backtest, strategy_loader=loader,
        strategy_engine=engine, alert_manager=None, db_path=sim_config.db_path)
    return manager, loader, backtest, engine


@pytest.mark.asyncio
async def test_lifecycle_matrix_uses_the_persisted_watchlist(sim_config, tmp_path):
    from core.market_data.universe import save_watchlist

    manager, loader, backtest, _ = await _lifecycle(sim_config, tmp_path, {
        "per_matrix": {"unrestricted": {
            "ADAUSDT": {"pnl": -9.0},
            "DOGEUSDT": {"pnl": 5.0},
            "XRPUSDT": {"pnl": 1.0},
        }},
    })
    watchlist = ["ADAUSDT", "DOGEUSDT", "XRPUSDT"]
    await save_watchlist(sim_config.db_path, watchlist)
    loader.save(_strategy("unrestricted"))
    sim_config.ai_mode = "semi_auto"

    await manager.analyze_and_optimize()

    assert backtest.calls[0]["symbols"] == watchlist, \
        ("the strategy×symbol matrix must run over the persisted watchlist, "
         "not a private five-symbol list")
    assert set(loader.load("unrestricted").symbols) == {"DOGEUSDT", "XRPUSDT"}, \
        "the negative-PnL symbol must be written back out of the strategy"


@pytest.mark.asyncio
async def test_lifecycle_retirement_and_deploy_backtest_the_watchlist(sim_config, tmp_path):
    from core.market_data.universe import save_watchlist

    manager, loader, backtest, _ = await _lifecycle(sim_config, tmp_path, {
        "metrics": {"total_trades": 5, "sharpe_ratio": 1.0,
                    "win_rate_pct": 55.0, "max_drawdown_pct": 5.0},
    })
    watchlist = ["ADAUSDT", "DOGEUSDT"]
    await save_watchlist(sim_config.db_path, watchlist)
    loader.save(_strategy("unrestricted"))

    await manager._evaluate_for_retirement("unrestricted")
    assert backtest.calls[-1]["symbols"] == watchlist

    # validate_and_deploy with an unrestricted AI config: same fallback.
    from tests.test_strategy_lifecycle import VALID_AI_CONFIG

    cfg = dict(VALID_AI_CONFIG)
    cfg.pop("symbols", None)
    await manager.validate_and_deploy(cfg)
    assert backtest.calls[-1]["symbols"] == watchlist


# ======================================================================
# 3. signal-matrix timeline = union over all symbols
# ======================================================================

def test_signal_matrix_timeline_is_the_union_of_every_symbol():
    from core.backtest.signal_matrix import SignalMatrixBuilder
    from core.strategy.loader import MLConfig, StrategyConfig

    # The LATE symbol is FIRST: with `symbols[0]` as the timeline its 20 bars
    # used to truncate the whole run (every OLDUSDT bar outside that window was
    # silently dropped).
    old_index = pd.date_range("2026-01-01", periods=60, freq="1h")
    late_index = pd.date_range("2026-01-02", periods=20, freq="1h")
    feeder = FakeFeeder({
        "LATEUSDT": {"1h": _ohlcv(late_index, 1)},
        "OLDUSDT": {"1h": _ohlcv(old_index, 2)},
    }, date_start="2026-01-01")
    strategy = StrategyConfig(
        name="union_probe", enabled=True, mode="trend", timeframes=["1h"],
        indicators={"rsi": {"period": 14, "source": "close"}},
        entry_conditions={"long": ["rsi < 70"], "short": []},
        exit_conditions={"long": [], "short": []},
        ml_config=MLConfig(enabled=False))

    matrix = SignalMatrixBuilder(feeder).build([strategy], ["LATEUSDT", "OLDUSDT"])

    assert matrix.metadata["timestamp_count"] == 60, \
        "the shared timeline must cover every symbol, not just symbols[0]"
    assert len(matrix.signals.columns) == 60
    assert matrix.signals.columns[0] == old_index[0], \
        "the late-listed symbol must not truncate the earlier timestamps"
    assert len(matrix.price_data["LATEUSDT"]["1h"]) == 20
    # Row order stays strategy-major then symbol-major (unchanged contract).
    assert list(dict.fromkeys(idx[1] for idx in matrix.signals.index)) == \
        ["LATEUSDT", "OLDUSDT"]


# ======================================================================
# 4. backtest regime proxy
# ======================================================================

def test_regime_proxy_defaults_to_the_first_symbol_of_the_run():
    from core.backtest.engine import _resolve_regime_proxy

    class Cfg:
        pass

    regime = {"ADAUSDT": "bull", "DOGEUSDT": "bear"}
    proxy = _resolve_regime_proxy(Cfg(), ["ADAUSDT", "DOGEUSDT"], regime)
    assert proxy == "ADAUSDT"
    assert regime.get(proxy, "range") == "bull", \
        "BTCUSDT-less runs must use a real regime, not the silent 'range' fallback"


def test_regime_proxy_is_configurable_and_falls_back_to_an_available_symbol():
    from core.backtest.engine import _resolve_regime_proxy

    class Cfg:
        backtest_regime_symbol = "ETHUSDT"

    regime = {"ADAUSDT": "bull", "ETHUSDT": "bear"}
    assert _resolve_regime_proxy(Cfg(), ["ADAUSDT", "ETHUSDT"], regime) == "ETHUSDT"

    # Configured proxy is not part of this run → first symbol WITH a regime.
    regime = {"ADAUSDT": "bull", "DOGEUSDT": "bear"}
    assert _resolve_regime_proxy(Cfg(), ["ADAUSDT", "DOGEUSDT"], regime) == "ADAUSDT"

    # First symbol has no detected regime → the next symbol that does.
    assert _resolve_regime_proxy(Cfg(), ["SOLUSDT", "ADAUSDT"], regime) == "ADAUSDT"

    # Nothing usable → None (caller keeps "range").
    assert _resolve_regime_proxy(Cfg(), ["SOLUSDT"], regime) is None
    assert _resolve_regime_proxy(Cfg(), [], {}) is None


# ======================================================================
# 5. LOT_SIZE / NOTIONAL enforcement in the live order path
# ======================================================================

@pytest.mark.parametrize("qty,step,expected", [
    (0.0123456789, 0.001, 0.012),        # 0.001-step symbol: floors DOWN
    (1.9999, 0.5, 1.5),
    (0.019, 0.01, 0.01),
    (5.0, 1.0, 5.0),
    (0.0123456789, 0.00001, 0.01234),    # 1e-5 step (real BTCUSDT LOT_SIZE)
    (0.0123456789, None, 0.01235),       # unknown step → legacy 1e-5 fallback
    (0.0123456789, 0.0, 0.01235),
    (0.0123456789, -1, 0.01235),
])
def test_round_down_to_step(qty, step, expected):
    from core.executor.executor import OrderExecutor

    assert OrderExecutor._round_down_to_step(qty, step) == pytest.approx(expected)


@pytest.mark.asyncio
async def test_live_order_uses_the_symbol_step_size(sim_config):
    from app.event_bus import EventBus
    from core.executor.executor import OrderExecutor

    sim_config.mode = "live"
    bus = EventBus()
    await bus.start()
    executor = OrderExecutor(sim_config, bus)
    executor.client = FakeExchange()
    executor.wire_universe(FakeUniverse(["ADAUSDT"], step_size=0.001))

    await executor._execute_live({
        "symbol": "ADAUSDT", "side": "long", "price": 50000.0,
        "quantity": 0.0123456789, "amount_usdt": 600.0,
    })
    await bus.shutdown()

    assert executor.client.calls[0]["quantity"] == pytest.approx(0.012), \
        "the exchange must receive the floored, LOT_SIZE-legal quantity"
    assert executor.get_open_positions()["ADAUSDT"]["quantity"] == pytest.approx(0.012)


@pytest.mark.asyncio
async def test_live_order_for_an_unknown_symbol_keeps_the_legacy_fallback(sim_config):
    from app.event_bus import EventBus
    from core.executor.executor import OrderExecutor

    sim_config.mode = "live"
    bus = EventBus()
    await bus.start()
    executor = OrderExecutor(sim_config, bus)
    executor.client = FakeExchange()
    executor.wire_universe(FakeUniverse(["ADAUSDT"]))  # ZZZUSDT is unknown

    await executor._execute_live({
        "symbol": "ZZZUSDT", "side": "long", "price": 50000.0,
        "quantity": 0.0123456789, "amount_usdt": 600.0,
    })
    await bus.shutdown()

    assert executor.client.calls[0]["quantity"] == pytest.approx(0.01235), \
        "an unresolvable symbol keeps the documented 1e-5 (round(qty, 5)) fallback"


@pytest.mark.asyncio
@pytest.mark.parametrize("quantity,price,kind", [
    (0.001, 5000.0, "order_below_min_notional"),   # 5 USDT < 10 USDT minimum
    (0.0001, 50000.0, "order_below_lot_size"),     # floors to 0 at step 0.001
])
async def test_live_order_below_exchange_limits_is_rejected_not_sent(
        sim_config, quantity, price, kind):
    from app.event_bus import EventBus
    from core.executor.executor import OrderExecutor

    sim_config.mode = "live"
    bus = EventBus()
    await bus.start()
    executor = OrderExecutor(sim_config, bus)
    executor.client = FakeExchange()
    executor.wire_universe(FakeUniverse(
        ["ADAUSDT"], step_size=0.001, min_notional=10.0))
    alerts = AlertCollector(bus)

    await executor._execute_live({
        "symbol": "ADAUSDT", "side": "long", "price": price,
        "quantity": quantity, "amount_usdt": quantity * price,
    })
    await alerts.wait(1)
    await bus.shutdown()

    assert executor.client.calls == [], "an illegal order must never be submitted"
    assert [a["type"] for a in alerts.alerts] == [kind]


# ======================================================================
# 6. one interval registry
# ======================================================================

def test_interval_registry_has_no_second_copy():
    from core.market_data import provider
    from core.backtest import signal_matrix
    from core.ga import genome

    assert genome.TIMEFRAME_OPTIONS == list(provider.DEFAULT_INTERVALS)
    assert set(provider.DEFAULT_INTERVALS) <= set(provider.INTERVAL_SPEC)
    assert provider.DEFAULT_ML_INTERVAL in provider.ml_intervals()
    for tf, spec in provider.INTERVAL_SPEC.items():
        assert set(spec) == {"minutes", "min_candles", "batches", "poll_secs",
                             "ml_enabled"}, tf
        assert signal_matrix._tf_minutes(tf) == spec["minutes"]
    # Unknown intervals keep the documented fallbacks (no KeyError anywhere).
    assert signal_matrix._tf_minutes("nope") == 60
    assert provider.interval_spec("nope")["min_candles"] == 200


def test_new_registry_interval_works_everywhere(monkeypatch):
    """Adding one interval to INTERVAL_SPEC is enough for every consumer."""
    from core.market_data import provider
    from core.backtest import signal_matrix
    from core.strategy.loader import MLConfig, StrategyConfig

    monkeypatch.setitem(provider.INTERVAL_SPEC, "2d", {
        "minutes": 2880, "min_candles": 200, "batches": 1,
        "poll_secs": 5760, "ml_enabled": True,
    })

    assert provider.interval_minutes("2d") == 2880
    assert provider.poll_seconds("2d") == 5760
    assert "2d" in provider.ml_intervals()
    assert signal_matrix._tf_minutes("2d") == 2880

    # ...and it orders correctly as the primary (shortest) timeframe.
    strategy = StrategyConfig(
        name="new_tf", enabled=True, mode="trend", timeframes=["1h", "2d"],
        indicators={"rsi": {"period": 14, "source": "close"}},
        entry_conditions={"long": ["rsi < 30"], "short": []},
        exit_conditions={"long": [], "short": []},
        ml_config=MLConfig(enabled=False))
    assert signal_matrix._primary_timeframe(strategy) == "1h"
    assert provider.interval_spec("2d")["ml_enabled"] is True


def test_poll_cadence_comes_from_the_registry():
    from core.market_data.provider import poll_seconds

    # Values the old inline dict in app.main held (must be unchanged)...
    assert [poll_seconds(tf) for tf in ("1m", "5m", "15m", "1h", "4h")] == \
        [120, 300, 900, 3600, 7200]
    # ...and a non-streamed interval no longer silently gets the 300s default.
    assert poll_seconds("30m") == 1800
    assert poll_seconds("unknown") == 300


@pytest.mark.asyncio
async def test_ml_predictor_gate_is_registry_driven(sim_config):
    from app.event_bus import Event, EventBus, EventType
    from core.market_data.provider import ml_intervals
    from core.ml.predictor import MLPredictor

    class FakeMD:
        watched_symbols = ["BTCUSDT"]

        def __init__(self):
            self.calls: list[tuple] = []

        async def get_historical(self, symbol, interval, limit=500):
            self.calls.append((symbol, interval))
            return None  # short-circuits right after the interval gate

    bus = EventBus()
    market_data = FakeMD()
    predictor = MLPredictor(sim_config, bus, market_data)
    predictor._running = True

    await predictor._on_kline(Event(EventType.MARKET_KLINE, {
        "symbol": "BTCUSDT", "interval": "1m", "candle": {}}))
    assert market_data.calls == [], "a non-ML interval must be ignored"

    await predictor._on_kline(Event(EventType.MARKET_KLINE, {
        "symbol": "BTCUSDT", "interval": "1h", "candle": {}}))
    assert market_data.calls == [("BTCUSDT", "1h")], \
        f"an ML-enabled interval must be processed ({ml_intervals()})"


def test_ga_timeframe_sort_accepts_a_new_interval(monkeypatch):
    from core.ga import genome

    monkeypatch.setattr(genome, "TIMEFRAME_OPTIONS", ["1d", "1m"])
    monkeypatch.setattr(genome.random, "sample", lambda pop, k: list(pop)[:k])
    monkeypatch.setattr(genome.random, "choice", lambda seq: seq[-1])  # k=2
    random.seed(0)

    chrom = genome.random_chromosome("ga_new_tf")
    assert genome.chromosome_to_strategy(chrom).timeframes == ["1m", "1d"], \
        "the timeframe sort must come from the registry (no KeyError on 1d)"


# ======================================================================
# 7/9. news: base asset + core/satellite split
# ======================================================================

class _FakeResponse:
    status_code = 200

    def json(self):
        return {"articles": []}


@pytest.mark.asyncio
async def test_news_query_uses_the_base_asset_not_the_pair():
    from core.news.fetcher import NewsFetcher

    captured: dict = {}

    class FakeHTTP:
        async def get(self, url):
            captured["url"] = url
            return _FakeResponse()

    fetcher = NewsFetcher(universe=FakeUniverse(["BTCUSDT", "ETHUSDT"]))
    fetcher._client = FakeHTTP()
    source = {"type": "api", "name": "probe",
              # IP literal host: _is_safe_url must not need a DNS lookup.
              "endpoint": "http://93.184.216.34/news?symbol={symbol}&limit={limit}"}

    await fetcher.fetch_from_source(source, "BTCUSDT", 5)

    assert captured["url"] == "http://93.184.216.34/news?symbol=BTC&limit=5", \
        "news providers are queried by asset, not by exchange pair"


@pytest.mark.asyncio
async def test_news_query_falls_back_to_the_raw_symbol_when_unknown():
    from core.news.fetcher import NewsFetcher

    captured: dict = {}

    class FakeHTTP:
        async def get(self, url):
            captured["url"] = url
            return _FakeResponse()

    fetcher = NewsFetcher(universe=FakeUniverse(["BTCUSDT"]))
    fetcher._client = FakeHTTP()
    source = {"type": "api", "name": "probe",
              "endpoint": "http://93.184.216.34/news?symbol={symbol}"}

    await fetcher.fetch_from_source(source, "WIFUSDT", 5)

    assert captured["url"] == "http://93.184.216.34/news?symbol=WIFUSDT"
    assert fetcher.base_asset("WIFUSDT") == "WIFUSDT"
    assert fetcher.base_asset("BTCUSDT") == "BTC"


def test_main_splits_news_symbols_by_core_max_symbols():
    """`watchlist[:3]/[3:]` is gone: the split follows `core_position.max_symbols`."""
    import app.main as main_module

    source = inspect.getsource(main_module)
    assert 'core_max = max(0, int(getattr(config, "core_max_symbols", 0) or 0))' in source
    assert "news_analyzer.start(watchlist[:core_max], watchlist[core_max:])" in source
    assert "news_analyzer.start(watchlist[:3], watchlist[3:])" not in source


# ======================================================================
# 10/11. no duplicated literals; trades rows carry the timeframe
# ======================================================================

@pytest.mark.parametrize("relative", [
    "core/ai/deepseek_ctl.py",
    "core/ai/strategy_lifecycle.py",
    "core/executor/pending_orders.py",
    "core/backtest/signal_matrix.py",
    "app/main.py",
    "web/routes/pages.py",
    "web/templates/dashboard.html",
    "web/templates/strategies.html",
    "web/templates/ai_panel.html",
])
def test_no_duplicated_watchlist_literal(relative):
    text = (ROOT / relative).read_text(encoding="utf-8")
    assert not _WATCHLIST_LITERAL.search(text), \
        f"{relative} still hard-codes the duplicated watchlist"
    assert "TRACKED_SYMBOLS" not in text


@pytest.mark.asyncio
async def test_page_symbols_come_from_the_persisted_watchlist(sim_config):
    from core.market_data.universe import save_watchlist
    from web.routes.pages import _page_symbols

    await init_database(sim_config.db_path)
    await save_watchlist(sim_config.db_path, ["ADAUSDT", "DOGEUSDT"])
    assert await _page_symbols(sim_config) == ["ADAUSDT", "DOGEUSDT"]


@pytest.mark.asyncio
async def test_trades_rows_carry_the_signals_timeframe(sim_config):
    from app.event_bus import EventBus
    from core.executor.executor import OrderExecutor

    await init_database(sim_config.db_path)
    bus = EventBus()
    await bus.start()
    executor = OrderExecutor(sim_config, bus)
    await executor.start()
    await executor._execute_sim({
        "symbol": "BTCUSDT", "side": "long", "price": 50000.0,
        "quantity": 0.01, "stop_loss": 49000.0, "strategy": "trend",
        "strategy_name": "probe", "timeframe": "4h", "position_type": "core",
    })

    # A restart must not lose the attribution either.
    restored = OrderExecutor(sim_config, bus)
    await restored.restore_positions()
    assert restored.get_open_positions()["BTCUSDT"]["timeframe"] == "4h"

    await restored.close_position("BTCUSDT", 50, 51000.0)
    await restored.close_position("BTCUSDT", 100, 52000.0)
    await bus.shutdown()

    async with aiosqlite.connect(sim_config.db_path) as db:
        cursor = await db.execute("SELECT action, timeframe FROM trades ORDER BY id")
        rows = await cursor.fetchall()

    assert [r[0] for r in rows] == ["open", "reduce", "close"]
    assert [r[1] for r in rows] == ["4h", "4h", "4h"], \
        "every row must carry the signal's timeframe, not a hard-coded '1h'"


def _market_test_app(sim_config):
    """``create_app`` with a fake trader injected (no login round-trip)."""
    from starlette.middleware.base import BaseHTTPMiddleware

    from app.event_bus import EventBus
    from core.risk.manager import RiskResult
    from web.server import create_app

    class FakeUser:
        username = "tester"
        is_trader = True
        is_admin = False

    class InjectUser(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.user = FakeUser()
            return await call_next(request)

    class FakeRiskManager:
        def __init__(self):
            self.signals: list[dict] = []

        async def check_signal(self, signal):
            self.signals.append(dict(signal))
            return RiskResult(approved=True,
                              adjusted_stop_loss=signal.get("stop_loss"),
                              adjusted_leverage=2)

    class StubExecutor:
        def get_open_positions(self):
            return {}

        async def close_position(self, symbol, reduce_pct=100, current_price=0):
            return {"ok": False, "error": "stub"}  # pragma: no cover

    app = create_app(sim_config, EventBus(), None)
    app.add_middleware(InjectUser)
    app.state.config = sim_config
    app.state.executor = StubExecutor()
    app.state.risk_manager = FakeRiskManager()
    app.state.get_price = lambda symbol: 50000.0
    app.state.balance = 10000.0
    return app, app.state.risk_manager


def test_api_order_threads_the_timeframe_into_the_signal(sim_config):
    from fastapi.testclient import TestClient

    app, risk_manager = _market_test_app(sim_config)
    with TestClient(app) as client:
        ok = client.post("/api/order", data={
            "symbol": "BTCUSDT", "side": "long", "type": "market",
            "amount_usdt": 100.0, "timeframe": "4h",
        })
        assert ok.status_code == 200, ok.text
        assert ok.json().get("ok") is True, ok.text

        default = client.post("/api/order", data={
            "symbol": "BTCUSDT", "side": "long", "type": "market",
            "amount_usdt": 100.0,
        })
        assert default.status_code == 200, default.text

    assert [s["timeframe"] for s in risk_manager.signals] == ["4h", "1h"], \
        "the order's timeframe must reach the signal (1h only when unspecified)"
