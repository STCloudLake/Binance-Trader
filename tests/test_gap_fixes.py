"""P3/P4 gap-fix regression suite (the flagged leftovers of phases P3 and P4).

One test class per item, each with its before/after evidence in the docstring:

1. **Volatility-targeted sizing in the LIVE path** — ``RiskManager.check_signal``
   now passes a forecast to ``PositionSizer.calculate_position_size`` and the
   sizer receives ``risk.vol_targeting`` at construction.  Before: the sizer had
   no ``vol_targeting`` object and the call had no ``forecast_vol_pct``, so live
   sizing was fixed-fraction even with the switch on (240.0 USDT notional at
   balance 10 000 on both sides of the change while the switch is off).
2. **Data integrity** — ``BTCUSDT/1h.parquet`` measured 11 calendar gaps > 1.5 h,
   largest 1 484 h (2026-07-29 → 2026-09-29): one "+27.63 %" bar that an
   un-clipped RiskMetrics recursion turns into 5.31 %/bar against 0.52 %/bar
   clipped (10.12x).  ``scripts/check_data_integrity.py`` reports the gaps for
   every cached file and refuses the un-clipped number for a spliced series; the
   live forecast path refuses that series too.
3. **``sim_cost_quote`` spread semantics** — the table entry is the **full**
   quoted spread and the fill model charges ``spread_pct / 2`` per side (the
   ``backtest.cost_model`` convention, and what `core/ml/credibility` and
   `tests/test_fees.py` already assumed).  Behaviour unchanged: BTCUSDT fill
   50 012.5000 / round trip 0.25005 %, ETHUSDT 50 015.0000 / 0.26006 %.
4. **One entry evaluator** — ``core/backtest/engine.py`` calls
   ``strategy.entry_sides`` instead of its old inline AND loop.  Before/after on
   the real BTCUSDT 1h cache: identical entries (40/0/0/40) and signal bars
   (244/300/0/300) for four AND/OR genomes; the reimplemented legacy rule equals
   ``entry_sides`` on every bar.
5. **sklearn feature-name warnings** — the engine fits LightGBM on the named
   39-column contract and predicts on a names-less array, warning on every bar
   (1 602 warnings in ``tests/test_engine_ml_gate.py``, now 0).

Nothing here writes to ``data/`` or ``strategies/``; the real cache is only read.
"""
from __future__ import annotations

import asyncio
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BTC_1H = Path("data/market/BTCUSDT/1h.parquet")
SIGNAL = {"symbol": "BTCUSDT", "side": "long", "price": 50000.0,
          "position_type": "satellite", "timeframe": "1h", "leverage": 2}
BALANCE = 10_000.0


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _isolate_config():
    """Never leave a mutated singleton behind for the next test file."""
    from app.config import Config

    yield
    Config._instance = None


def _config(tmp_path, **attrs):
    from app.config import Config

    Config._instance = None
    cfg = Config.load("sim")
    cfg.db_path = str(tmp_path / "gap_fixes.db")   # never the live DB
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    for key, value in attrs.items():
        setattr(cfg, key, value)
    return cfg


def _risk_manager(cfg, executor=None):
    from app.event_bus import EventBus
    from core.risk.manager import RiskManager

    rm = RiskManager(cfg, EventBus())
    if executor is not None:
        rm.wire_executor(executor)
    rm.update_balance(BALANCE)
    return rm


class _StubExecutor:
    """Minimal executor exposing the two methods RiskManager reads."""

    def __init__(self, vol_pct=None):
        self._vol_pct = vol_pct
        self.positions: dict = {}

    def get_open_positions(self):
        return self.positions

    def vol_stop_ctx(self, symbol, vol_pct=None):
        if self._vol_pct is None:
            return {}
        return {"vol_pct": self._vol_pct, "stop_pct": 1.0}


def _size(rm, signal=None):
    """``(quantity, notional)`` the risk manager approves for one signal."""
    signal = dict(signal or SIGNAL)
    result = asyncio.run(rm.check_signal(signal))
    assert result.approved, result.reason
    return result.adjusted_quantity, result.adjusted_quantity * signal["price"]


def _notional(rm, signal=None):
    return _size(rm, signal)[1]


# ══════════════════════════════════════════════════════════════════════
# 1 — volatility-targeted sizing reaches the live path
# ══════════════════════════════════════════════════════════════════════

def test_switch_off_sizing_is_bit_identical_to_the_pre_p3_calculation(tmp_path):
    """Item 1, switch off: the notional must not move by one ULP.

    ``calculate_position_size`` is compared against a ``PositionSizer`` built
    **without** a ``vol_targeting`` object (the pre-P3 constructor call), for
    three cases: no forecast source, a forecast available on the executor, and a
    forecast pushed with the switch flipped back off.  All three must equal
    0.0048 BTC / 240.0 USDT exactly (``==``, not ``approx``).
    """
    from core.risk.position_sizer import PositionSizer

    cfg = _config(tmp_path)
    assert cfg.risk_vol_targeting.enabled is False
    legacy = PositionSizer(cfg.hard_limits, cfg.soft_params,
                           cfg.core_capital_pct, cfg.satellite_capital_pct)
    qty_ref, risk_ref = legacy.calculate_position_size(
        BALANCE, SIGNAL["price"], "satellite")
    assert (qty_ref, risk_ref) == (0.0048, 240.0)

    # (a) no forecast source at all
    bare = _risk_manager(cfg)
    qty, notional = _size(bare)
    assert qty == qty_ref, (qty, qty_ref)
    assert notional == qty_ref * SIGNAL["price"]
    # (b) a forecast IS available (executor push channel) while the switch is off
    pushed = _risk_manager(cfg, _StubExecutor(vol_pct=9.0))
    assert _size(pushed) == (qty_ref, qty_ref * SIGNAL["price"])
    # (c) the manager's own forecast path short-circuits while the switch is off
    assert asyncio.run(pushed.forecast_vol_pct("BTCUSDT", "1h")) is None
    assert asyncio.run(pushed.resolve_forecast_vol_pct(dict(SIGNAL))) is None
    print(f"\n[item 1 off] qty={qty_ref} notional={risk_ref} USDT "
          f"(executor forecast 9.0 %/bar ignored; identical to the pre-P3 sizer)")


def test_switch_on_scales_the_notional_inversely_with_the_forecast(tmp_path):
    """Item 1, switch on: 2× the target volatility ⇒ exactly ½ the notional."""
    from core.executor.executor import OrderExecutor
    from app.event_bus import EventBus
    from loguru import logger

    cfg = _config(tmp_path)
    cfg.risk_vol_targeting.enabled = True
    target = cfg.risk_vol_targeting.target_vol_pct           # 0.45 %/bar
    executor = OrderExecutor(cfg, EventBus())                # the real P3 push channel
    rm = _risk_manager(cfg, executor)

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="DEBUG")
    try:
        executor.set_forecast_vol_pct("BTCUSDT", target)
        n_target = _notional(rm)
        executor.set_forecast_vol_pct("BTCUSDT", 2.0 * target)
        n_double = _notional(rm)
        executor.set_forecast_vol_pct("BTCUSDT", 0.5 * target)
        n_half = _notional(rm)
    finally:
        logger.remove(sink)

    assert n_double == pytest.approx(0.5 * n_target, rel=1e-12)
    assert n_half == pytest.approx(2.0 * n_target, rel=1e-12)   # clipped by max_scale
    assert n_target == pytest.approx(240.0, rel=1e-12)

    # The scale factor is logged at debug level (operator evidence, not silent).
    assert any("scale=0.5000x" in m for m in messages), messages[-3:]
    assert any("forecast=" in m and "target=" in m for m in messages)

    # ... and the capped case still respects the same bounds as the off switch.
    cfg.risk_vol_targeting.max_position_notional_pct = 3.0    # 300 USDT ceiling
    capped = _notional(rm)
    assert capped == pytest.approx(300.0)
    cfg.hard_limits.max_position_size_pct = 1.0               # 100 USDT ceiling
    assert _notional(rm) == pytest.approx(100.0)
    print(f"\n[item 1 on] target={n_target:.4f} double={n_double:.4f} "
          f"half={n_half:.4f} capped={capped:.4f}")


def test_forecast_is_refused_for_a_spliced_series_and_cached_otherwise(tmp_path):
    """Item 1 + item 2(b) guard: a spliced price history is never priced in.

    A calendar gap beyond 1.5 bar lengths makes ``forecast_vol_pct`` return
    ``None`` — sizing then uses the fixed fraction (the pre-P3 number) instead of
    a volatility inflated by a fake ``+30 %`` bar.  A contiguous series is used,
    and the estimate is cached per TTL (one provider call for many signals).
    """
    cfg = _config(tmp_path)
    cfg.risk_vol_targeting.enabled = True
    rng = np.random.default_rng(20260930)
    idx = pd.date_range("2026-01-01", periods=300, freq="1h")
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, idx.size))

    class _FakeMD:
        def __init__(self, frame):
            self.frame = frame
            self.calls = 0

        async def get_historical(self, symbol, interval, limit=None):
            self.calls += 1
            return self.frame

    contiguous = pd.DataFrame({"close": close}, index=idx)
    rm = _risk_manager(cfg)
    md = _FakeMD(contiguous)
    rm.wire_market_data(md)

    vol = asyncio.run(rm.forecast_vol_pct("BTCUSDT", "1h"))
    assert vol is not None and vol > 0.0
    assert asyncio.run(rm.forecast_vol_pct("BTCUSDT", "1h")) == vol
    assert md.calls == 1, "the forecast must be TTL-cached, not recomputed per signal"
    # ... and the live sizing call consumes it.
    assert asyncio.run(rm.resolve_forecast_vol_pct(dict(SIGNAL))) == vol

    # Same series with a 100 h calendar gap and a violent jump across it.
    spliced_idx = idx[:150].append(
        pd.date_range(idx[149] + pd.Timedelta(hours=100), periods=150, freq="1h"))
    spliced = pd.DataFrame({"close": np.concatenate([close[:150], close[150:] * 1.3])},
                           index=spliced_idx)
    rm2 = _risk_manager(cfg)
    rm2.wire_market_data(_FakeMD(spliced))
    assert asyncio.run(rm2.forecast_vol_pct("BTCUSDT", "1h")) is None
    assert asyncio.run(rm2.resolve_forecast_vol_pct(dict(SIGNAL))) is None
    # Fixed-fraction fallback == the switch-off quantity, not an inflated one.
    from core.risk.position_sizer import PositionSizer
    legacy = PositionSizer(cfg.hard_limits, cfg.soft_params,
                           cfg.core_capital_pct, cfg.satellite_capital_pct)
    qty_ref, _ = legacy.calculate_position_size(BALANCE, SIGNAL["price"], "satellite")
    assert _size(rm2)[0] == qty_ref
    print(f"\n[item 1 guard] contiguous forecast={vol:.4f} %/bar (1 provider call); "
          f"spliced -> None -> qty {qty_ref} (fixed fraction)")


def test_live_kline_stream_is_the_last_resort_forecast_source(tmp_path):
    """The manager buffers ``MARKET_KLINE`` closes, so the live path needs no REST.

    ``start()`` subscribes the handler; while the switch is off nothing is
    buffered (zero cost, nothing changes); with it on, closed candles alone are
    enough to produce a forecast — no market-data provider, no per-signal scan.

    Audit F1 changed the buffer's shape: it keeps ``(close_time, close)`` so the
    splice guard can run on the live series, and a candle with **no** ``close_time``
    is refused (the guard cannot be applied to a timestamp-less series) instead of
    being indexed 0, 1, 2…  Production always sets ``close_time`` (Binance ``k.T``
    in ``provider._handle_ws_message``, ``df.index[-2]`` in the ``app/main.py``
    REST poll), so the candles below carry it and the refusal is exercised
    separately in ``tests/test_final_audit_fixes.py``.
    """
    from app.event_bus import EventBus, Event, EventType

    cfg = _config(tmp_path)
    bus = EventBus()
    from core.risk.manager import RiskManager
    rm = RiskManager(cfg, bus)
    asyncio.run(rm.start())
    assert rm._on_kline in bus._subscribers[EventType.MARKET_KLINE]
    try:
        # Switch off: events are ignored (no history kept).
        for i in range(10):
            asyncio.run(rm._on_kline(Event(EventType.MARKET_KLINE, {
                "symbol": "BTCUSDT", "interval": "1h",
                "candle": {"close_time": 1_767_225_600_000 + i * 3_600_000,
                           "close": 100.0 + i}})))
        assert rm._kline_history == {}
        assert bus._subscribers[EventType.MARKET_KLINE] == [rm._on_kline]

        cfg.risk_vol_targeting.enabled = True
        rng = np.random.default_rng(11)
        price = 100.0
        start_ms = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)
        for i in range(120):
            price *= float(np.exp(rng.normal(0.0, 0.004)))
            asyncio.run(rm._on_kline(Event(EventType.MARKET_KLINE, {
                "symbol": "BTCUSDT", "interval": "1h",
                "candle": {"close_time": start_ms + i * 3_600_000,
                           "close": price}})))
        vol = asyncio.run(rm.forecast_vol_pct("BTCUSDT", "1h"))
        assert vol is not None and 0.0 < vol < 5.0
        assert rm._market_data is None, "no REST source was needed"
    finally:
        asyncio.run(rm.stop())
    assert rm._on_kline not in bus._subscribers[EventType.MARKET_KLINE]
    print(f"\n[item 1 kline stream] forecast from 120 live closes: {vol:.4f} %/bar")


# ══════════════════════════════════════════════════════════════════════
# 2 — data integrity: report, guard, refetch evidence
# ══════════════════════════════════════════════════════════════════════

def test_integrity_checker_flags_a_splice_and_refuses_unclipped_vol(tmp_path, capsys):
    """The checker reports the gap and refuses the un-clipped volatility."""
    from scripts.check_data_integrity import gap_report, main, vol_report

    rng = np.random.default_rng(5)
    idx = pd.date_range("2026-01-01", periods=200, freq="1h")
    # The seam sits 15 bars from the end: the RiskMetrics recursion weights the
    # recent tail (effective memory 1/(1-λ) ≈ 17 bars), which is exactly why the
    # measured BTC case — a gap one bar before the end of the window — inflates the
    # un-clipped forecast by an order of magnitude.
    idx = idx[:170].append(pd.date_range(idx[169] + pd.Timedelta(hours=100),
                                         periods=30, freq="1h"))
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.3, idx.size))
    close[170:] *= 1.3                       # the "+27 %" splice, one fake bar
    frame = pd.DataFrame({"open": close, "high": close * 1.001,
                          "low": close * 0.999, "close": close,
                          "volume": 1.0}, index=idx)

    rep = gap_report(frame, "1h")
    assert rep["flagged"] is True
    assert rep["gap_count"] == 1
    assert rep["largest_gap_hours"] == pytest.approx(100.0, abs=0.1)
    assert rep["missing"] > 90

    vol = vol_report(frame)
    assert vol["unclipped_pct"] > 5.0 * vol["clipped_pct"], vol

    root = tmp_path / "data"
    path = root / "market" / "BTCUSDT" / "1h.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)

    assert main(["--data-dir", str(root), "--check-vol"]) == 0
    out = capsys.readouterr().out
    assert "GAP" in out and "REFUSED" in out
    # --force-unclipped is the research/reporting escape hatch.
    assert main(["--data-dir", str(root), "--check-vol", "--force-unclipped"]) == 0
    forced = capsys.readouterr().out
    assert "REFUSED" not in forced and "x" in forced
    # --strict turns a flagged file into a non-zero exit.
    assert main(["--data-dir", str(root), "--strict"]) == 1
    print(f"\n[item 2] synthetic splice: gap={rep['largest_gap_hours']}h "
          f"clipped={vol['clipped_pct']:.4f} unclipped={vol['unclipped_pct']:.4f} "
          f"ratio={vol['ratio']:.2f}x -> REFUSED")


def test_real_btc_cache_report_matches_its_own_content_whatever_that_is():
    """The report must describe the file that is on disk — no defect required.

    ``data/market/BTCUSDT/1h.parquet`` is a *live* file: the running service
    rewrote it to 8 848 spliced rows at 13:01 after it had been repaired to 11 675
    clean rows at 12:35.  Pinning the measured defect (or a magic multiple such as
    "unclipped > 8.0 x clipped", which measured 7.616x on the current file and
    10.12x on an older slice) makes the suite fail for a reason that has nothing
    to do with the code under test.

    What is asserted instead: (a) every number ``gap_report`` returns is
    recomputed independently from the same frame, so the report is *honest about
    whatever state the cache is in*, and (b) the inflation the guard exists for is
    a property of a splice, demonstrated on a **synthetic** seam injected into a
    contiguous series.  A missing file skips; a repaired or a spliced file both
    pass.
    """
    if not BTC_1H.exists():
        pytest.skip("no cached BTCUSDT 1h parquet in this checkout")
    from core.ml.volatility import log_returns
    from scripts.check_data_integrity import gap_report, vol_report

    frame = pd.read_parquet(BTC_1H)

    # (a) independent recomputation of the gap report from the same frame.
    rep = gap_report(frame, "1h")
    assert rep["bars"] == len(frame)
    assert rep["expected"] == int(round(rep["span_hours"] / 1.0)) + 1
    assert rep["missing"] == max(rep["expected"] - len(frame), 0)
    idx = pd.to_datetime(pd.Index(frame.index))
    idx = idx[idx.argsort()]
    assert (rep["first"], rep["last"]) == (idx[0].isoformat(), idx[-1].isoformat())
    hours = pd.Series(idx).diff().dt.total_seconds().to_numpy() / 3600.0
    # ``hours[pos]`` is the gap *ending* at ``idx[pos]`` (element 0 is NaN).
    expected = sorted(((idx[pos].isoformat(), round(float(hours[pos]), 3))
                       for pos in range(1, len(idx)) if hours[pos] > 1.5),
                      key=lambda item: item[1], reverse=True)
    assert rep["gaps"] == expected
    assert rep["gap_count"] == len(expected)
    assert rep["flagged"] is bool(expected)
    if expected:
        assert rep["largest_gap_hours"] == expected[0][1]
        assert rep["largest_gap_at"] == expected[0][0]

    # The vol report is internally consistent whatever the file holds.
    vol = vol_report(frame)
    assert vol["ratio"] == pytest.approx(vol["unclipped_pct"] / vol["clipped_pct"],
                                         rel=1e-9)
    assert vol["unclipped_pct"] >= vol["clipped_pct"] * (1.0 - 1e-9)
    tail = log_returns(frame["close"].astype(float).values)[-500:]
    if np.abs(tail).max() > 0.1:          # a splice sits inside the estimator window
        assert vol["ratio"] > 2.0, vol    # ... and the clip demonstrably bites
        seam_state = "splice in the measured window"
    else:
        seam_state = "no splice in the measured window"

    # (b) the same estimator on a *synthetic* +27.63 % seam: the clip is what
    # keeps a fake bar out of the variance, and the overstatement is an order of
    # magnitude (the figure the docs quote for the injected splice).
    from core.ml.volatility import ewma_vol, to_pct

    rng = np.random.default_rng(20260930)
    calm = rng.normal(0.0, 0.004, 500)
    injected = np.concatenate([calm[:-1], np.array([0.2763])])
    inj_ratio = (to_pct(ewma_vol(injected, window=500, outlier_sigma=0.0))
                 / to_pct(ewma_vol(injected, window=500)))
    assert inj_ratio > 5.0, inj_ratio
    print(f"\n[item 2 real cache] bars={rep['bars']} gaps={rep['gap_count']} "
          f"largest={rep['largest_gap_hours']}h missing={rep['missing']} "
          f"({seam_state}) | clipped={vol['clipped_pct']:.4f} "
          f"unclipped={vol['unclipped_pct']:.4f} ({vol['ratio']:.2f}x) | "
          f"synthetic +0.2763 seam: {inj_ratio:.2f}x")


# ══════════════════════════════════════════════════════════════════════
# 3 — one meaning for sim.cost_model.spread_pct
# ══════════════════════════════════════════════════════════════════════

def test_sim_spread_pct_is_the_full_quoted_spread(tmp_path):
    """One meaning, pinned: table = full spread, fill impact = spread/2 + slip.

    Also asserts the documentation matches the arithmetic the code charges, so
    the P2.1 mismatch (docstring said "per-side half-spread" while the code
    divided by two again) cannot come back.
    """
    from app.config import (Config, _sim_settings_from_config, sim_cost_quote,
                            sim_spread_pct)

    cfg = _config(tmp_path)
    settings = _sim_settings_from_config(cfg)
    assert sim_spread_pct(settings, "BTCUSDT") == 0.01      # the FULL spread
    assert sim_spread_pct(settings, "ETHUSDT") == 0.02

    for symbol in ("BTCUSDT", "ETHUSDT"):
        quote = sim_cost_quote(symbol, "long", "market", 50_000.0, 0.1, settings)
        half = sim_spread_pct(settings, symbol) / 2.0
        assert quote["edge_pct"] == pytest.approx(
            half + settings["slippage_bps"] / 100.0, rel=1e-12)
        assert quote["fill_price"] == pytest.approx(
            50_000.0 * (1 + quote["edge_pct"] / 100.0), rel=1e-12)

    btc = sim_cost_quote("BTCUSDT", "long", "market", 50_000.0, 0.1, settings)
    eth = sim_cost_quote("ETHUSDT", "long", "market", 50_000.0, 0.1, settings)
    assert btc["edge_pct"] == pytest.approx(0.025)          # 0.005 + 0.02 slippage
    assert eth["edge_pct"] == pytest.approx(0.03)           # 0.01  + 0.02 slippage
    assert btc["fill_price"] == pytest.approx(50_012.5)
    assert eth["fill_price"] == pytest.approx(50_015.0)
    assert btc["cost_usdt"] / (50_000.0 * 0.1) * 100 == pytest.approx(0.12503, rel=1e-4)
    assert eth["cost_usdt"] / (50_000.0 * 0.1) * 100 == pytest.approx(0.13003, rel=1e-4)

    # Documentation regression pin: the words used by the config and the helper.
    yaml_text = (Path(__file__).resolve().parents[1] / "config" / "config.yaml"
                 ).read_text(encoding="utf-8")
    assert "FULL quoted bid-ask spread" in yaml_text
    assert "Per-side half-spread" not in yaml_text
    assert "Full** quoted bid-ask spread" in sim_spread_pct.__doc__
    print(f"\n[item 3] BTCUSDT fill={btc['fill_price']:.4f} "
          f"({btc['edge_pct']}%/side, round-trip "
          f"{2 * btc['cost_usdt'] / 5000.0 * 100:.5f}%) | ETHUSDT "
          f"fill={eth['fill_price']:.4f} ({eth['edge_pct']}%/side, round-trip "
          f"{2 * eth['cost_usdt'] / 5000.0 * 100:.5f}%) — identical before/after "
          f"(docs-only change)")


# ══════════════════════════════════════════════════════════════════════
# 4 — one AND/OR evaluator for live, GA and backtest
# ══════════════════════════════════════════════════════════════════════

def _legacy_inline_conditions(df, s_cfg, side: str):
    """The code the backtest engine used to carry (verbatim, for parity)."""
    from core.strategy.indicators import evaluate_condition

    conditions = (getattr(s_cfg, "entry_conditions", None) or {}).get(side, [])
    if not conditions:
        return False
    if str(getattr(s_cfg, "condition_logic", "or")).lower() != "and":
        return None  # sentinel: caller uses the shared OR kernel
    for cond in conditions:
        try:
            mask = evaluate_condition(df, cond)
        except Exception:
            return False
        if not (hasattr(mask, "iloc") and bool(mask.iloc[-1])):
            return False
    return True


def _legacy_entry_sides(df, s_cfg):
    """The old call-site branch, reconstructed exactly as it was."""
    from core.strategy.evaluation_kernel import evaluate_entry_conditions

    long_and = _legacy_inline_conditions(df, s_cfg, "long")
    short_and = _legacy_inline_conditions(df, s_cfg, "short")
    if long_and is None and short_and is None:
        return evaluate_entry_conditions(df, s_cfg.entry_conditions)
    return bool(long_and), bool(short_and)


def _probe_strategy(logic: str, name: str, case: str = "A"):
    from core.strategy.loader import StrategyConfig

    conditions = {
        "A": {"long": ["rsi < 70", "close > sma"], "short": ["rsi > 30", "close < sma"]},
        "B": {"long": ["rsi < 30", "close > sma"], "short": ["rsi > 70", "close < sma"]},
    }[case]
    return StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=["1h"],
        indicators={"rsi": {"period": 14, "source": "close"},
                    "sma": {"period": 20}},
        entry_conditions=conditions,
        exit_conditions={"long": ["rsi > 90"], "short": ["rsi < 10"]},
        condition_logic=logic,
    )


def _real_frame():
    if not BTC_1H.exists():
        pytest.skip("no cached BTCUSDT 1h parquet in this checkout")
    from core.strategy.indicators import compute_all

    return compute_all(pd.read_parquet(BTC_1H),
                       {"rsi": {"period": 14, "source": "close"},
                        "sma": {"period": 20}})


def test_legacy_inline_and_rule_equals_entry_sides_on_real_data():
    """Per-bar parity on the real cached symbol: old inline rule == entry_sides."""
    frame = _real_frame()
    start = len(frame) - 300
    totals = {}
    for case in ("A", "B"):
        for logic in ("and", "or"):
            cfg = _probe_strategy(logic, f"parity_{case}_{logic}", case)
            shared = legacy = 0
            for i in range(300):
                window = frame.iloc[: start + i + 1]
                assert cfg.entry_sides(window) == _legacy_entry_sides(window, cfg), (
                    f"{case}/{logic} disagrees at {window.index[-1]}")
                if any(cfg.entry_sides(window)):
                    shared += 1
                if any(_legacy_entry_sides(window, cfg)):
                    legacy += 1
            assert shared == legacy
            totals[f"{case}_{logic}"] = shared
    assert totals["A_and"] > 0 and totals["B_or"] > 0, totals
    assert totals["A_and"] < totals["A_or"]
    print(f"\n[item 4 parity] identical on every bar; active-signal bars {totals}")


def test_backtest_entry_path_calls_the_shared_evaluator(monkeypatch, tmp_path):
    """The engine really routes through ``StrategyConfig.entry_sides``.

    A counting wrapper proves the call happens (the inline loop is gone) and every
    entry the engine takes lands on a bar where the shared rule reports that side
    active.
    """
    if not BTC_1H.exists():
        pytest.skip("no cached BTCUSDT 1h parquet in this checkout")
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import StrategyConfig

    calls: list[int] = []
    original = StrategyConfig.entry_sides

    def counting(self, df):
        calls.append(1)
        return original(self, df)

    monkeypatch.setattr(StrategyConfig, "entry_sides", counting)

    cfg = _config(tmp_path)
    cfg.data_dir = str(BTC_1H.resolve().parents[2])          # .../data
    bus = EventBus()
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus), OrderExecutor(cfg, bus))
    result = engine.run_with_exit_evaluation(
        strategies=[_probe_strategy("and", "probe_and")], symbols=["BTCUSDT"],
        date_start="2026-02-01", date_end="2026-03-15", initial_balance=10_000.0,
        mode="full", simulate_ai_weights=False, use_live_spread=False)
    assert "error" not in result
    assert len(calls) > 100, f"entry_sides called only {len(calls)} times"
    entries = [(e["time"], e["side"]) for e in result["events"]
               if e.get("type") == "entry"]
    assert entries, "the probe genome took no entries — the test is vacuous"
    frame = _real_frame()
    for time_str, side in entries:
        pos = frame.index.get_loc(pd.Timestamp(time_str))
        long_active, short_active = _probe_strategy("and", "probe_and").entry_sides(
            frame.iloc[:pos + 1])
        assert (long_active if side == "long" else short_active), (time_str, side)
    print(f"\n[item 4 wiring] entry_sides calls={len(calls)}, entries={len(entries)}, "
          f"0 rule mismatches")


# ══════════════════════════════════════════════════════════════════════
# 5 — consistent sklearn feature names at fit and predict time
# ══════════════════════════════════════════════════════════════════════

def test_ml_matrix_carries_the_model_feature_names():
    """The predict matrix adopts the names the model was fitted with."""
    from core.backtest.engine import BacktestEngine

    frame = pd.DataFrame(np.arange(12, dtype=float).reshape(3, 4),
                         columns=["rsi_14", "sma_20", "atr_14", "close_position"])

    class _Model:
        pass

    model = _Model()
    # No names reported (XGBoost/stub fitted on an array) -> plain array, as before.
    assert isinstance(BacktestEngine._ml_matrix_for_model(frame, model), np.ndarray)
    # Names reported and matching the width -> same values under those names.
    model.feature_names_in_ = np.array(["Column_0", "Column_1", "Column_2", "Column_3"])
    aligned = BacktestEngine._ml_matrix_for_model(frame, model)
    assert isinstance(aligned, pd.DataFrame)
    assert list(aligned.columns) == ["Column_0", "Column_1", "Column_2", "Column_3"]
    assert np.array_equal(aligned.values, frame.values)
    # A width mismatch is not silently mislabelled.
    model.feature_names_in_ = np.array(["a", "b"])
    assert isinstance(BacktestEngine._ml_matrix_for_model(frame, model), np.ndarray)


def test_named_fit_and_predict_emit_no_feature_name_warning():
    """The 1 602-warning root cause, reproduced and fixed.

    LightGBM exposes ``feature_names_in_`` even when fitted on an ndarray, so
    predicting with a names-less array warns on **every** call; predicting with an
    array on a model fitted on a *named* frame is the same mistake.  The two fits
    are numerically identical, which is why switching to named frames is safe.
    """
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(20260930)
    X = rng.normal(size=(240, 39))
    y = (rng.random(240) > 0.5).astype(int)
    names = [f"f{i}" for i in range(39)]
    named = pd.DataFrame(X, columns=names)

    def fit(data):
        return lgb.LGBMClassifier(n_estimators=12, max_depth=3, random_state=7,
                                  verbosity=-1).fit(data, y)

    named_model, array_model = fit(named), fit(X)
    assert np.allclose(named_model.predict_proba(named)[:, 1],
                       array_model.predict_proba(X)[:, 1])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        named_model.predict_proba(named)
    assert not [w for w in caught if "feature names" in str(w.message)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        named_model.predict_proba(X)
    assert [w for w in caught if "feature names" in str(w.message)], (
        "the warning this fix removes did not reproduce")
    print("\n[item 5] named fit/predict: 0 feature-name warnings; "
          "names-less predict on the same model warns (root cause reproduced)")


def test_engine_ml_fit_keeps_the_39_column_contract_without_warnings(tmp_path):
    """Integration: one real ``_train_ml_model`` + ``_predict_ml`` round trip."""
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.ml.features import FEATURE_NAMES
    from core.risk.manager import RiskManager

    cfg = _config(tmp_path)
    bus = EventBus()
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus), OrderExecutor(cfg, bus))

    rng = np.random.default_rng(4242)
    idx = pd.date_range("2026-01-01", periods=600, freq="1h")
    close = 20000.0 + np.cumsum(rng.normal(0.0, 40.0, idx.size))
    # Intrabar high/low must differ per bar: a constant `close_position`
    # ((close-low)/(high-low)) is rejected by the feature contract.
    up = rng.random(idx.size) * 60.0 + 5.0
    down = rng.random(idx.size) * 60.0 + 5.0
    frame = pd.DataFrame({"open": close, "high": close + up, "low": close - down,
                          "close": close, "volume": rng.random(idx.size) * 50 + 1},
                         index=idx)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = engine._train_ml_model(
            frame, {"forward": 4, "threshold": 0.005, "min_candles": 100})
        assert model is not None
        prediction = engine._predict_ml(model, frame)

    assert list(model.feature_names_in_) == list(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == 39
    assert prediction is not None and 0.0 <= prediction["p_up"] <= 1.0
    feature_warnings = [w for w in caught if "feature names" in str(w.message)]
    assert feature_warnings == [], [str(w.message) for w in feature_warnings]
    print(f"\n[item 5 integration] 39 named columns, p_up={prediction['p_up']:.3f}, "
          f"feature-name warnings={len(feature_warnings)}")
