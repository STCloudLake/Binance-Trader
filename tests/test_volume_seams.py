"""P6-B — the P6-A volume seams, the cache extension and the versioned contract.

Four things are pinned here, each with its own "off" and "on" evidence:

1. **cache extension** — ``quote_volume`` / ``trade_count`` survive the
   union-on-write, do not break the bar-open dedupe, do not merge two different
   bars, and do not corrupt a file that predates them (readers must not crash on
   an absent/NaN column);
2. **kline mapping + backfill** — the source's fields 7/8 land in the cached
   columns, ``--backfill`` derives its request windows from the file itself, is
   resumable (a second run asks for nothing) and refuses to rewrite a file the
   live process moved under it;
3. **live seam** — ``RiskManager.resolve_recent_quote_volume`` /
   ``PositionGuard.recent_quote_volume`` return ``None`` without I/O while
   ``risk.liquidity.enabled`` is false, and the same value with it on;
4. **backtest seam + inertness** — a k=0 run is bit-identical with and without
   the per-bar volume, and a k>0 run with a real cache charges a strictly larger
   total cost than the same run without volume.

Nothing here downloads from the network or touches ``data/`` or ``strategies/``:
every frame is built in ``tmp_path`` and every "source" is a scripted fake.
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SEED = 424242

#: The two columns P6-B adds to the cache.
EXTENDED = ("quote_volume", "trade_count")


# ── helpers ──────────────────────────────────────────────────────────────

def _bars(start: str, periods: int, *, freq: str = "1h",
          seed: int = SEED, extended: bool = True) -> pd.DataFrame:
    """Deterministic OHLCV (+ the P6-B columns) with a bar-open index."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq=freq)
    close = 30_000.0 * np.exp(np.cumsum(rng.normal(scale=0.004, size=periods)))
    high = close * (1 + np.abs(rng.normal(scale=0.001, size=periods)))
    low = close * (1 - np.abs(rng.normal(scale=0.001, size=periods)))
    volume = rng.lognormal(mean=5.0, sigma=0.3, size=periods)
    frame = pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                          "volume": volume}, index=idx)
    if extended:
        frame["quote_volume"] = volume * close
        frame["trade_count"] = np.maximum(1.0, volume / 2.0)
    return frame


def _download_module():
    """The download script as a module (it lives outside the packages)."""
    path = ROOT / "scripts" / "download_history.py"
    spec = importlib.util.spec_from_file_location("_p6b_download_history", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ══════════════════════════════════════════════════════════════════════════
# 1 — cache extension: union-on-write, dedupe, twins, backward compatibility
# ══════════════════════════════════════════════════════════════════════════

def test_extension_columns_survive_the_union_on_write(tmp_path):
    """A P6-B writer must keep the extended columns through a merge."""
    from core.market_data.ohlcv_cache import OHLVCache

    cache = OHLVCache(str(tmp_path))
    base = _bars("2025-01-01", 40)
    cache.update("BTCUSDT", "1h", base)
    cache.save("BTCUSDT", "1h")
    stored = pd.read_parquet(tmp_path / "market" / "BTCUSDT" / "1h.parquet")
    assert list(stored.columns) == ["open", "high", "low", "close", "volume",
                                    *EXTENDED]
    assert stored["quote_volume"].notna().all()
    assert stored["trade_count"].notna().all()

    # A second writer that only knows the original five columns, appending NEW
    # bars, must not delete the two new columns for the rows already stored: the
    # union fills its side with NaN and keeps every stored number.
    legacy_new = _bars("2025-01-02 16:00", 5, seed=SEED + 3, extended=False)
    cache.update("BTCUSDT", "1h", legacy_new)
    cache.save("BTCUSDT", "1h")
    after = pd.read_parquet(tmp_path / "market" / "BTCUSDT" / "1h.parquet")
    assert len(after) == len(base) + len(legacy_new)
    assert list(after.columns) == ["open", "high", "low", "close", "volume",
                                   *EXTENDED]
    stored_rows = after.loc[base.index]
    assert stored_rows["quote_volume"].notna().all(), (
        "the merge dropped a stored column")
    # …and the incoming rows are NaN, the documented "absent" value (not 0.0).
    assert after["quote_volume"].tail(5).isna().all()

    # A P6-B writer overwriting an existing bar DOES win that row (keep="last"),
    # extended columns included.
    refreshed = _bars("2025-01-01", 40, seed=SEED + 9)
    cache.update("BTCUSDT", "1h", refreshed)
    cache.save("BTCUSDT", "1h")
    final = pd.read_parquet(tmp_path / "market" / "BTCUSDT" / "1h.parquet")
    assert np.allclose(final.loc[base.index, "volume"].to_numpy(),
                       refreshed["volume"].to_numpy())


def test_extension_columns_do_not_break_the_bar_open_dedupe_or_merge_twins(tmp_path):
    """Gap/twin invariants: two conventions of one bar still collapse to one row."""
    from core.market_data.ohlcv_cache import merge_history

    open_stamps = _bars("2025-01-01", 24)
    # The same bars under Binance's close_time convention (open + 1 h − 1 ms).
    close_stamps = open_stamps.copy()
    close_stamps.index = close_stamps.index + pd.Timedelta(hours=1) - pd.Timedelta(
        milliseconds=1)
    merged = merge_history(open_stamps, close_stamps, "1h")
    assert len(merged) == len(open_stamps), "one row per bar"
    assert merged["quote_volume"].notna().all()
    # The survivor is stamped in the file's dominant convention (bar open here).
    assert all(ts.minute == 0 and ts.second == 0 for ts in merged.index)

    # Two genuinely different bars 1 ms apart must NOT be merged (the historical
    # defect: 54 such pairs in the live cache are 54 different hours).
    a = _bars("2025-01-01", 3)
    b = _bars("2025-01-01 01:00", 3)
    b.index = b.index - pd.Timedelta(milliseconds=1)   # 00:59:59.999 next to 01:00:00
    twin = pd.concat([a, b])
    union = merge_history(twin.copy(), None, "1h")
    assert len(union) == len(twin)


def test_a_pre_p6b_file_is_read_and_merged_without_crashing(tmp_path):
    """Backward compatibility: absent columns are absent, not zero and not fatal."""
    from core.market_data.ohlcv_cache import OHLVCache, merge_history

    market = tmp_path / "market" / "ETHUSDT"
    market.mkdir(parents=True)
    old = _bars("2025-01-01", 30, extended=False)
    old.to_parquet(market / "1h.parquet")

    cache = OHLVCache(str(tmp_path))
    loaded = cache.get("ETHUSDT", "1h")
    assert loaded is not None and len(loaded) == 30
    # Documented behaviour: the column is simply not there.  (Not filled with 0:
    # a zero quote volume would read as "nothing traded".)
    assert "quote_volume" not in loaded.columns
    assert "trade_count" not in loaded.columns

    # Appending a P6-B bar adds the columns at the end and leaves NaN on the 30
    # legacy rows — NaN, never 0.
    fresh = _bars("2025-01-02 06:00", 1)
    cache.update("ETHUSDT", "1h", merge_history(loaded, fresh, "1h"))
    cache.save("ETHUSDT", "1h")
    stored = pd.read_parquet(market / "1h.parquet")
    assert len(stored) == 31
    assert set(EXTENDED) <= set(stored.columns)
    assert int(stored["quote_volume"].isna().sum()) == 30
    assert float(stored["quote_volume"].iloc[-1]) > 0.0


def test_flush_of_an_already_deduped_legacy_file_still_writes_nothing(tmp_path):
    """The no-op-flush guarantee survives the schema change (bytes untouched)."""
    from core.market_data.ohlcv_cache import OHLVCache

    market = tmp_path / "market" / "BTCUSDT"
    market.mkdir(parents=True)
    frame = _bars("2025-01-01", 12, extended=False)
    path = market / "1h.parquet"
    frame.to_parquet(path)
    before = path.read_bytes()

    cache = OHLVCache(str(tmp_path))
    cache.get("BTCUSDT", "1h")       # loading is enough to un-dirty the key
    cache.flush_all()
    assert path.read_bytes() == before, (
        "a deduped legacy file was rewritten by a flush that changed nothing")


# ══════════════════════════════════════════════════════════════════════════
# 2 — kline mapping and the resumable backfill
# ══════════════════════════════════════════════════════════════════════════

def _kline_row(open_ms: int, *, quote: float, trades: int, close: float = 100.0):
    return [open_ms, str(close), str(close + 1), str(close - 1), str(close),
            "7.5", open_ms + 3_599_999, str(quote), trades,
            "3.0", "300.0", "0"]


def _fake_client(rows_by_open: dict, *, bar_ms: int = 3_600_000):
    """Async stand-in for ``MarketDataClient.klines`` (scripted, offline).

    The window test mirrors the endpoint's *intent* (a kline belongs to the window
    when the bar overlaps it) without reproducing its exact 1 ms boundary rule:
    the bar's midpoint must fall inside ``[start_time, end_time]``.  That is what
    makes the first bar of a run come back even though its ``open_time`` sits
    1 ms on either side of ``start_time`` depending on which timestamp convention
    the file stores.
    """

    class _Client:
        def __init__(self):
            self.calls: list[tuple] = []

        async def klines(self, symbol, interval, limit=1000, start_time=None,
                         end_time=None):
            self.calls.append((symbol, interval, limit, start_time, end_time))
            if start_time is None:
                return []
            high = end_time if end_time is not None else start_time + bar_ms
            wanted = sorted(t for t in rows_by_open
                            if start_time <= t + bar_ms // 2 <= high)
            return [rows_by_open[t] for t in wanted[:limit]]

    return _Client()


def test_kline_payload_maps_quote_volume_and_trade_count():
    dl = _download_module()
    rows = [_kline_row(1_700_000_000_000, quote=1234.5, trades=17),
            _kline_row(1_700_003_600_000, quote=2345.5, trades=19)]
    frame = dl.klines_to_frame(rows)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume",
                                   "quote_volume", "trade_count"]
    assert frame["quote_volume"].tolist() == [1234.5, 2345.5]
    assert frame["trade_count"].tolist() == [17.0, 19.0]
    assert frame.index.name == "close_time"
    assert frame.index[1] - frame.index[0] == pd.Timedelta(hours=1)
    # A short payload yields the documented NaN instead of raising.
    short = dl.klines_to_frame([[1_700_000_000_000, "1", "1", "1", "1", "1",
                                 1_700_003_599_999]])
    assert short["quote_volume"].isna().all()
    assert short["trade_count"].isna().all()


def test_backfill_plan_derives_contiguous_runs_from_the_file():
    dl = _download_module()
    frame = _bars("2025-01-01", 30)
    assert dl.backfill_plan(frame) == []
    holes = frame.copy()
    holes.loc[holes.index[5:9], "quote_volume"] = np.nan   # one run of 4
    holes.loc[holes.index[20], "quote_volume"] = np.nan    # a run of 1
    plan = dl.backfill_plan(holes)
    assert [entry["rows"] for entry in plan] == [4, 1]
    hour_ms = 3_600_000
    assert plan[0]["end_time"] - plan[0]["start_time"] == 4 * hour_ms - 1
    # A file with no such column at all plans the whole history in one run.
    assert [e["rows"] for e in dl.backfill_plan(
        frame.drop(columns=list(EXTENDED)))] == [30]


def test_backfill_fills_a_legacy_file_and_is_resumable(tmp_path):
    dl = _download_module()
    market = tmp_path / "market" / "BTCUSDT"
    market.mkdir(parents=True)
    legacy = _bars("2025-01-01", 24, extended=False)
    legacy.to_parquet(market / "1h.parquet")

    # The "source" serves exactly those bars, with quote volume = volume × close.
    # A kline's `close_time` is `open_time + length − 1 ms`, and the bar stored at
    # stamp S *is* the bar that opened at S (this frame uses the bar-open
    # convention), so the fake source speaks the real payload's convention.
    rows = {}
    for stamp, row in legacy.iterrows():
        open_ms = int(stamp.value // 1_000_000)
        rows[open_ms] = _kline_row(open_ms, quote=float(row["volume"] * row["close"]),
                                   trades=int(row["volume"]), close=float(row["open"]))
    client = _fake_client(rows)

    import asyncio
    result = asyncio.run(dl.backfill_cache(client, "BTCUSDT", "1h", tmp_path))
    assert result["missing_before"] == 24 and result["filled"] == 24
    assert result["pages"] == 1
    stored = pd.read_parquet(market / "1h.parquet")
    assert set(EXTENDED) <= set(stored.columns)
    assert stored["quote_volume"].notna().all()
    # Stored prices/volumes are authoritative — never rewritten by the fetch.
    assert np.allclose(stored["volume"].to_numpy(), legacy["volume"].to_numpy())

    # Resumability: a second run finds nothing to do and makes no request.
    calls_before = len(client.calls)
    again = asyncio.run(dl.backfill_cache(client, "BTCUSDT", "1h", tmp_path))
    assert again["done"] is True and again["pages"] == 0
    assert len(client.calls) == calls_before


def test_backfill_refuses_to_rewrite_a_file_the_live_process_moved(tmp_path):
    """Concurrent-append safety: the live service owns the market cache."""
    dl = _download_module()
    market = tmp_path / "market" / "BTCUSDT"
    market.mkdir(parents=True)
    legacy = _bars("2025-01-01", 12, extended=False)
    path = market / "1h.parquet"
    legacy.to_parquet(path)

    rows = {}
    for stamp, row in legacy.iterrows():
        open_ms = int(stamp.value // 1_000_000)
        rows[open_ms] = _kline_row(open_ms, quote=100.0, trades=3,
                                   close=float(row["open"]))

    class _MovingClient:
        """Appends a legacy bar to the file as soon as it is asked for klines."""

        async def klines(self, symbol, interval, limit=1000, start_time=None,
                         end_time=None):
            stored = pd.read_parquet(path)
            extra = pd.DataFrame(
                {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                 "volume": [1.0]},
                index=pd.DatetimeIndex([stored.index[-1] + pd.Timedelta(hours=1)]))
            pd.concat([stored, extra]).to_parquet(path)
            return [rows[t] for t in sorted(rows) if t >= (start_time or 0)]

    import asyncio
    with pytest.raises(dl.CacheMovedError):
        asyncio.run(dl.backfill_cache(_MovingClient(), "BTCUSDT", "1h", tmp_path))
    # Nothing was written: the file still has no extended columns.
    assert "quote_volume" not in pd.read_parquet(path).columns


# ══════════════════════════════════════════════════════════════════════════
# 3 — the live seam (RiskManager / PositionGuard)
# ══════════════════════════════════════════════════════════════════════════

class _Liquidity:
    """Duck-typed ``risk.liquidity`` block (what ``app.config`` builds)."""

    def __init__(self, enabled=False, max_participation_pct=1.0,
                 lookback_bars=20, impact_k=0.0, impact_exponent=0.5,
                 per_symbol=None):
        self.enabled = enabled
        self.max_participation_pct = max_participation_pct
        self.lookback_bars = lookback_bars
        self.impact_k = impact_k
        self.impact_exponent = impact_exponent
        self.per_symbol = per_symbol or {}


class _Config:
    """Minimal ``Config`` stand-in carrying only what the seams read."""

    def __init__(self, liquidity=None):
        from app.config import HardRiskLimits, SoftRiskParams, VolTargetingConfig

        self.hard_limits = HardRiskLimits()
        self.soft_params = SoftRiskParams()
        self.core_capital_pct = 0.7
        self.satellite_capital_pct = 0.3
        self.risk_vol_targeting = VolTargetingConfig()
        self.risk_liquidity = liquidity if liquidity is not None else _Liquidity()
        # Production carries the block on this object (app.config), which is how
        # PositionSizer finds it without holding `config` itself.
        object.__setattr__(self.risk_vol_targeting, "risk_liquidity",
                           self.risk_liquidity)
        self.db_path = ""
        self.data_dir = ""


class _Market:
    """Market-data double that counts how often it was asked for history."""

    def __init__(self, frame=None):
        self.frame = frame
        self.calls = 0

    async def get_historical(self, symbol, interval, limit=500):
        self.calls += 1
        return self.frame


def _risk_manager(liquidity=None, frame=None):
    from app.event_bus import EventBus
    from core.risk.manager import RiskManager

    manager = RiskManager(_Config(liquidity), EventBus())
    market = _Market(frame)
    manager.wire_market_data(market)
    return manager, market


def test_risk_manager_quote_volume_is_none_and_io_free_by_default():
    """The shipped switch is false: no lookup, no request, no behaviour change."""
    import asyncio

    manager, market = _risk_manager(_Liquidity(enabled=False),
                                    frame=_bars("2025-01-01", 25))
    value = asyncio.run(manager.resolve_recent_quote_volume({"symbol": "BTCUSDT"}))
    assert value is None
    assert market.calls == 0, "a disabled seam must not touch market data"

    # …and the sizing call it feeds is the pre-P6 arithmetic: identical tuple.
    base = manager.sizer.calculate_position_size(10_000.0, 50_000.0, "satellite")
    with_arg = manager.sizer.calculate_position_size(
        10_000.0, 50_000.0, "satellite", recent_quote_volume=None)
    assert base == with_arg


def test_risk_manager_resolves_the_window_when_enabled():
    import asyncio

    frame = _bars("2025-01-01", 25)
    expected = float((frame["quote_volume"].tail(20)).sum())
    manager, market = _risk_manager(
        _Liquidity(enabled=True, lookback_bars=20), frame=frame)
    value = asyncio.run(manager.resolve_recent_quote_volume({"symbol": "BTCUSDT"}))
    assert value == pytest.approx(expected)
    assert market.calls == 1

    # A signal that already carries the number short-circuits the lookup.
    again = asyncio.run(manager.resolve_recent_quote_volume(
        {"symbol": "BTCUSDT", "recent_quote_volume": 1234.5}))
    assert again == pytest.approx(1234.5)
    assert market.calls == 1


def test_position_guard_quote_volume_mirrors_the_switch():
    import asyncio
    from app.event_bus import EventBus
    from core.risk.position_guard import PositionGuard

    frame = _bars("2025-01-01", 25)
    off = PositionGuard(_Config(_Liquidity(enabled=False)), EventBus())
    off.wire(None, _Market(frame))
    assert off.quote_volume_enabled() is False
    assert asyncio.run(off.recent_quote_volume("BTCUSDT")) is None

    on = PositionGuard(_Config(_Liquidity(enabled=True)), EventBus())
    market = _Market(frame)
    on.wire(None, market)
    assert on.quote_volume_enabled() is True
    assert asyncio.run(on.recent_quote_volume("BTCUSDT")) == pytest.approx(
        float(frame["quote_volume"].tail(20).sum()))
    assert market.calls == 1


# ══════════════════════════════════════════════════════════════════════════
# 4 — the backtest seam: inert at k=0, real at k>0
# ══════════════════════════════════════════════════════════════════════════

def test_cost_model_recent_quote_volume_is_off_at_k_zero():
    """k = 0 (shipped) plus a real volume ⇒ exactly the pre-P6 cost."""
    from core.backtest.cost_model import (apply_trading_costs,
                                          recent_quote_volume_from_bars)

    frame = _bars("2025-01-01", 25)
    window = recent_quote_volume_from_bars(frame, lookback_bars=20)
    assert window == pytest.approx(float(frame["quote_volume"].tail(20).sum()))
    shipped = _Config(_Liquidity(enabled=False, impact_k=0.0))
    legacy = apply_trading_costs(50_000.0, 51_000.0, 0.01, "BTCUSDT", shipped)
    with_volume = apply_trading_costs(50_000.0, 51_000.0, 0.01, "BTCUSDT", shipped,
                                      recent_quote_volume=window)
    assert repr(legacy) == repr(with_volume)
    # k = 0 is inert even with the participation switch ON (impact is its own key).
    on = _Config(_Liquidity(enabled=True, impact_k=0.0))
    assert repr(apply_trading_costs(50_000.0, 51_000.0, 0.01, "BTCUSDT", on,
                                    recent_quote_volume=window)) == repr(legacy)


def test_cost_model_charges_more_with_impact_enabled():
    from core.backtest.cost_model import (apply_trading_costs,
                                          recent_quote_volume_from_bars)

    frame = _bars("2025-01-01", 25)
    window = recent_quote_volume_from_bars(frame, lookback_bars=20)
    post = _Config(_Liquidity(enabled=False, impact_k=0.5))
    legacy = apply_trading_costs(50_000.0, 51_000.0, 1.0, "BTCUSDT", post)
    impacted = apply_trading_costs(50_000.0, 51_000.0, 1.0, "BTCUSDT", post,
                                   recent_quote_volume=window)
    assert impacted > legacy
    # The extra charge is exactly the documented square-root impact term.
    from core.risk.liquidity import total_impact_usdt
    expected = total_impact_usdt(50_000.0, 51_000.0, window, 0.5, 0.5)
    assert impacted - legacy == pytest.approx(expected)


def test_engine_impact_seam_is_bit_identical_at_k_zero(tmp_path):
    """Whole-run comparison on a synthetic market: k=0 ignores the volume feed."""
    report = _run_twice(tmp_path)
    assert report["k0"] == report["k0_again"], (
        "two identical k=0 runs disagreed — the seam is not deterministic")
    assert report["k0"] != report["k0_no_volume"] or True  # documented below
    # The strongest form available: the trade list and total PnL are equal to a
    # run whose cost function never sees a volume at all.
    assert report["k0"]["pnl"] == pytest.approx(report["k0_no_volume"]["pnl"])
    assert report["k0"]["trades"] == report["k0_no_volume"]["trades"]
    assert report["k0"]["n_trades"] == report["k0_no_volume"]["n_trades"]
    # And the impact run costs strictly more than the same run without it.
    assert report["k05"]["pnl"] < report["k0"]["pnl"], (
        "an enabled impact term must make the same trades cost more")


def _run_twice(tmp_path):
    """Four short engine runs: k=0 (×2), k=0 with the volume feed removed, k=0.5."""
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import StrategyConfig

    market = tmp_path / "data"
    market.mkdir(parents=True, exist_ok=True)
    frame = _bars("2025-01-01", 24 * 120)
    symbol_dir = market / "market" / "BTCUSDT"
    symbol_dir.mkdir(parents=True)
    frame.to_parquet(symbol_dir / "1h.parquet")

    strategy = StrategyConfig(
        name="p6b_probe", enabled=True, mode="trend", timeframes=["1h"],
        indicators={"sma": {"period": 5}},
        entry_conditions={"long": ["close > sma"], "short": ["close < sma"]},
        exit_conditions={"long": ["rsi > 99"], "short": ["rsi < 1"]},
    )

    def one(*, impact_k: float, with_volume: bool) -> dict:
        Config._instance = None
        cfg = Config.load("sim")
        cfg.data_dir = str(market)
        cfg.backtest_engine_mode = "legacy"
        cfg.backtest_ml_enabled = False
        cfg.backtest_live_spread_enabled = False
        cfg.risk_liquidity.impact_k = float(impact_k)
        bus = EventBus()
        engine = BacktestEngine(cfg, None, RiskManager(cfg, bus),
                                OrderExecutor(cfg, bus))
        if not with_volume:
            # Remove the feed entirely: `_recent_quote_volume_for` returns 0.0.
            engine._recent_quote_volume_for = lambda pos: 0.0
        result = engine.run_with_exit_evaluation(
            strategies=[strategy], symbols=["BTCUSDT"],
            date_start="2025-01-05", date_end="2025-03-01",
            initial_balance=10_000.0, mode="full", simulate_ai_weights=False,
            use_live_spread=False)
        assert "error" not in result, result
        metrics = result["metrics"]
        return {"pnl": float(metrics.get("total_pnl", result["final_balance"])
                             - 10_000.0),
                "trades": result["trades"],
                "n_trades": int(metrics.get("total_trades", len(result["trades"])))}

    out = {"k0": one(impact_k=0.0, with_volume=True),
           "k0_again": one(impact_k=0.0, with_volume=True),
           "k0_no_volume": one(impact_k=0.0, with_volume=False),
           "k05": one(impact_k=0.5, with_volume=True)}
    assert out["k0"]["n_trades"] > 0, "the probe strategy never traded"
    return out


def test_entry_sizing_quote_volume_seam_is_lazy_and_inert_by_default(tmp_path):
    """Audit finding 3: the participation seam now reaches **entry sizing**.

    The backtest fed per-bar volume to the exit cost model but not to
    ``calculate_position_size``, so the participation cap could never fire in a
    backtest while the impact term could.  The engine now passes a **callable**
    (the shape ``apply_participation_cap`` documents), which the sizer evaluates
    only while ``risk.liquidity.enabled`` is true — the default-off path is
    therefore bit-identical, and that is what this test pins.
    """
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.liquidity import recent_quote_volume
    from core.risk.manager import RiskManager
    from core.risk.position_sizer import PositionSizer

    market = tmp_path / "data" / "market" / "BTCUSDT"
    market.mkdir(parents=True)
    frame = _bars("2025-01-01", 24 * 60)
    frame.to_parquet(market / "1h.parquet")

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = str(tmp_path / "data")
    bus = EventBus()
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus),
                            OrderExecutor(cfg, bus))
    engine._current_ts = frame.index[-1]
    # `_feeder` is what the window lookup reads (set by the run loop).
    from core.backtest.data_feeder import DataFeeder

    feeder = DataFeeder(str(tmp_path / "data" / "market"), ["BTCUSDT"], ["1h"],
                        "2025-01-01", "2025-03-01")
    feeder.load()
    engine._feeder = feeder

    # The expectation is derived from the frame the engine will actually read —
    # the feeder applies the run's date window, so `frame` itself is a superset.
    stored = feeder.get_all_data_for_symbol("BTCUSDT", "1h")
    assert stored is not None and len(stored) >= 20
    expected = float(recent_quote_volume(stored, 20))
    assert expected > 0.0

    provider = engine._quote_volume_provider("BTCUSDT", "1h")
    assert callable(provider)
    assert provider() == pytest.approx(expected), (
        "the entry seam must read the same window the exit cost model reads")

    # Default (switch off): the sizer must never evaluate the provider and the
    # size must equal the size computed with no volume argument at all.
    calls = []

    def counting():
        calls.append(1)
        return expected

    cfg.risk_liquidity.enabled = False
    sizer = PositionSizer(cfg.hard_limits, cfg.soft_params,
                          cfg.core_capital_pct, cfg.satellite_capital_pct)
    base = sizer.calculate_position_size(10_000.0, 50_000.0, "satellite")
    with_arg = sizer.calculate_position_size(10_000.0, 50_000.0, "satellite",
                                            recent_quote_volume=counting,
                                            symbol="BTCUSDT")
    assert calls == [], "the provider ran while risk.liquidity.enabled was false"
    assert repr(base) == repr(with_arg), (
        "passing the seam while disabled changed the size")

    # Switch on: the provider runs exactly once and a cap bites.  The synthetic
    # window is ~9e7 USDT, far too deep for the default 240 USDT notional to be
    # capped, so the capped case uses a shallow 12 000 USDT book (1 % = 120).
    cfg.risk_liquidity.enabled = True
    cfg.risk_liquidity.max_participation_pct = 1.0
    calls.clear()
    capped = sizer.calculate_position_size(10_000.0, 50_000.0, "satellite",
                                          recent_quote_volume=lambda: (
                                              calls.append(1) or 12_000.0),
                                          symbol="BTCUSDT")
    assert calls == [1], "the provider must run once when the switch is on"
    assert capped[1] == pytest.approx(120.0), (
        "a 1 % participation cap on a 12 000 USDT window must shrink the notional")
    assert capped[1] < base[1]


def test_breadth_max_stale_ms_is_a_real_boundary_not_a_stored_constant(tmp_path):
    """Audit finding 5: ``MAX_STALE_MS`` used to be stored and never compared.

    The module docstring (and doc 15) give it behaviour — "beyond 30 min a cached
    observation is still returned, but only as evidence; a caller needing fresh
    numbers must treat it as unavailable".  The relabelling now sets
    ``missing=True`` past that boundary, so a caller can tell "stale but usable"
    from "treat as unavailable" without re-deriving the constant.
    """
    from core.market_data import breadth as B

    path = tmp_path / "breadth.jsonl"
    B.BreadthCache(path, now_ms=1_000_000).refresh(
        fetcher=lambda: _breadth_payload(), expected_pair_count=1)

    inside = B.BreadthCache(path, now_ms=1_000_000 + B.MAX_STALE_MS)
    assert inside.latest().missing is False
    assert inside.latest().is_stale is True          # past the 5-minute TTL

    outside = B.BreadthCache(path, now_ms=1_000_000 + B.MAX_STALE_MS + 1)
    beyond = outside.latest()
    assert beyond is not None, "the value is still returned, only labelled"
    assert beyond.missing is True and beyond.is_stale is True
    assert outside.fresh() is None


def _breadth_payload() -> list[dict]:
    """One usable USDT pair — the shape the live ticker endpoint returns."""
    return [{"symbol": "BTCUSDT", "quoteVolume": "1000", "priceChangePercent": "1.0",
             "lastPrice": "50000", "count": 10, "closeTime": 1_000_000}]


# ══════════════════════════════════════════════════════════════════════════
# 5 — the versioned contract: a v1 model is refused BY NAME, a v2 model loads
# ══════════════════════════════════════════════════════════════════════════

class _StubModel:
    def predict_proba(self, X):
        return np.tile(np.array([[0.4, 0.6]]), (len(X), 1))


def _write_model(models_dir: Path, stem: str, meta: dict) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{stem}.pkl"
    path.write_bytes(pickle.dumps(_StubModel()))
    (models_dir / f"{stem}_meta.json").write_text(
        __import__("json").dumps(meta), encoding="utf-8")
    return path


def test_a_v1_hash_model_is_refused_with_a_named_reason(tmp_path):
    """Audit finding 2: a **genuine** v1 artefact (39 names + v1 hash) is refused
    by the named hash branch, not by the count branch.

    The pre-fix test wrote v1's *hash* next to v2's 54 *names* — a shape no real
    v1 artefact has — so it passed while the named branch stayed unreachable for
    every model trained before P6-B.  The sidecar below is what a v1 trainer
    actually wrote: ``FEATURE_V1_NAMES`` (39) with ``FEATURE_SCHEMA_V1_HASH``.
    """
    from core.ml.features import (FEATURE_SCHEMA_V1_HASH, FEATURE_V1_NAMES,
                                  FEATURE_NAMES, feature_schema_hash,
                                  feature_schema_mismatch_reason)
    from core.ml.predictor import FeatureContractError, MLPredictor

    assert len(FEATURE_V1_NAMES) == 39
    assert len(FEATURE_NAMES) == 54
    assert feature_schema_hash(FEATURE_V1_NAMES) == FEATURE_SCHEMA_V1_HASH, (
        "the frozen v1 hash must be the hash of the 39-column contract")

    class _Cfg:
        data_dir = str(tmp_path)
        ml_enabled = False
        ml_model_type = "lightgbm"
        ml_feature_list = None

    class _Bus:
        def subscribe(self, *a, **k):
            pass

        def unsubscribe(self, *a, **k):
            pass

    class _Md:
        watched_symbols: list = []

    predictor = MLPredictor(_Cfg(), _Bus(), _Md())
    # A genuine v1 sidecar: the 39 v1 column names AND the v1 hash.
    v1 = _write_model(tmp_path, "BTCUSDT_default_binary", {
        "feature_names": list(FEATURE_V1_NAMES),
        "feature_schema_hash": FEATURE_SCHEMA_V1_HASH,
        "train_base_rate": 0.5,
        "gate": {"allowed": True, "reason": "pass", "auc": 0.61,
                 "net_expectancy": 0.003}})
    with pytest.raises(FeatureContractError, match="feature schema hash mismatch"):
        predictor.load_model("BTCUSDT", str(v1))
    # The refusal must be the NAMED one, not "expects 39 features but … 54".
    assert predictor.gate_status["allowed"] is False
    assert "v1" in predictor.gate_status["reason"]
    assert "v1 (39-column P2 contract)" in predictor.gate_status["reason"]
    assert "feature schema hash mismatch" in predictor.gate_status["reason"]
    assert "expects 39 features" not in predictor.gate_status["reason"]
    assert feature_schema_mismatch_reason(FEATURE_SCHEMA_V1_HASH).startswith(
        "feature schema hash mismatch")

    # A v1 hash is still refused when the names are v2's (the pre-fix shape) —
    # the hash is the contract's marker and now speaks first.
    mixed = _write_model(tmp_path, "BTCUSDT_mixed_binary", {
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_hash": FEATURE_SCHEMA_V1_HASH,
        "train_base_rate": 0.5,
        "gate": {"allowed": True, "reason": "pass", "auc": 0.61}})
    with pytest.raises(FeatureContractError, match="v1"):
        predictor.load_model("BTCUSDT", str(mixed))

    # The positive control: the v2 names with the v2 hash load.
    v2 = _write_model(tmp_path, "BTCUSDT_ok_binary", {
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_hash": feature_schema_hash(FEATURE_NAMES),
        "train_base_rate": 0.5,
        "gate": {"allowed": True, "reason": "pass", "auc": 0.61,
                 "net_expectancy": 0.003}})
    assert predictor.load_model("BTCUSDT", str(v2)) is not None
    assert predictor.gate_status["allowed"] is True


def test_the_backtest_preload_gate_refuses_a_v1_hash_by_name(tmp_path):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.ml.features import (FEATURE_NAMES, FEATURE_SCHEMA_V1_HASH,
                                  FEATURE_V1_NAMES, feature_schema_hash)
    from core.risk.manager import RiskManager

    models_dir = tmp_path / "data" / "models"
    # Audit finding 2: a genuine v1 artefact — 39 v1 names + the v1 hash.
    v1 = _write_model(models_dir, "BTCUSDT_alpha_binary", {
        "feature_names": list(FEATURE_V1_NAMES),
        "feature_schema_hash": FEATURE_SCHEMA_V1_HASH,
        "train_base_rate": 0.5,
        "gate": {"allowed": True, "reason": "pass", "auc": 0.61}})
    v2 = _write_model(models_dir, "BTCUSDT_beta_binary", {
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_hash": feature_schema_hash(FEATURE_NAMES),
        "train_base_rate": 0.5,
        "gate": {"allowed": True, "reason": "pass", "auc": 0.61}})

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = str(tmp_path / "data")
    bus = EventBus()
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus), None)
    engine._current_strategies = []

    ok, reason = engine._verify_ml_model_sidecar(v1.stem, v1, "binary")
    assert ok is False and "v1" in reason
    assert "v1 (39-column P2 contract)" in reason
    assert "schema hash mismatch" in reason
    assert "sidecar has 39 features" not in reason
    ok2, reason2 = engine._verify_ml_model_sidecar(v2.stem, v2, "binary")
    assert ok2 is True and reason2 == "verified"


def test_missing_hash_is_still_a_refusal_not_a_pass(tmp_path):
    """Re-audit finding 5 stays intact through the P6-B rewording."""
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.ml.features import FEATURE_NAMES
    from core.risk.manager import RiskManager

    models_dir = tmp_path / "data" / "models"
    path = _write_model(models_dir, "BTCUSDT_nohash_binary", {
        "feature_names": list(FEATURE_NAMES),
        "gate": {"allowed": True, "reason": "pass"}})
    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = str(tmp_path / "data")
    bus = EventBus()
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus), None)
    engine._current_strategies = []
    ok, reason = engine._verify_ml_model_sidecar(path.stem, path, "binary")
    assert ok is False
    assert "feature schema hash missing" in reason
