"""Re-audit fixes at revision ``b49883b`` — findings R1–R7, one section each.

Every section asserts the *defect* is gone **and** the shipped default
configuration (vol targeting off, ``risk.liquidity`` off, ML disabled, all P4
switches off) is untouched:

* **R1** — ``PositionGuard.forecast_vol_pct`` skipped the splice guard the
  ``RiskManager`` applies, so one cached series was "too spliced to size from" but
  still set the **live trailing-stop distance** (measured before: guard
  ``0.062009291811329616 %/bar`` on the very series the manager refused with
  ``None`` — the fixture's gap-free twin gives ``0.061993341684305286 %/bar``).
* **R2** — doc 13's impact percentages did not reproduce against the window they
  were pinned to.  Here the *formula* and the ``k = 0`` bit-identity are pinned
  (the absolute USDT deltas move with the live window, so they are a doc
  measurement, not a test constant).
* **R3** — ``flush_all`` only rewrote **dirty** keys, so a cache file that is never
  appended to again kept its twin-convention duplicate bars for ever (live
  ``BTCUSDT/1h``: 11 678 rows / 55 duplicated bars, and doc claim "已合并" was
  false).
* **R4** — ``probabilistic_sharpe`` was ``Φ(mean/se)`` while the docstrings claimed
  it read skew/kurtosis, which made the gate's ``AND`` exactly ``t > 2``.
* **R5** — a sidecar with ``gate.allowed`` and matching feature names but **no**
  ``feature_schema_hash`` was accepted by both preload paths.
* **R6** — the evidence index was not re-pinned.  That guard has since been
  replaced (§11 R2): it used to assert a *fixed* hash list while only checking
  that ``git rev-parse HEAD`` was truthy, so it passed five commits behind; it
  now recomputes the chain from the worktree (see
  ``test_evidence_index_reports_the_current_revision_and_a_current_chain``).
* **R7** — ``core.ml.credibility.__all__`` exported the undefined ``signed_score``,
  so ``from core.ml.credibility import *`` raised.

Nothing here writes ``data/`` (except through ``OHLVCache`` on a ``tmp_path``),
``strategies/`` or the live DB.
"""
from __future__ import annotations

import asyncio
import json
import math
import pickle
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]

#: The plan-freeze commit the evidence index's chain table starts from; the same
#: range the document's own reproduction command uses.
EVIDENCE_BASELINE = "f1f6a4c"


def _evidence_doc() -> Path:
    return REPO / "docs" / "overhaul" / "ALGO_UPGRADE_EVIDENCE.md"


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    from app.config import Config
    yield
    Config._instance = None


def _config(tmp_path):
    from app.config import Config
    Config._instance = None
    cfg = Config.load("sim")
    cfg.db_path = str(tmp_path / "reaudit.db")
    return cfg


# ══════════════════════════════════════════════════════════════════════
# R1 — the PositionGuard must apply the SAME splice guard as RiskManager
# ══════════════════════════════════════════════════════════════════════

def _vol_frames(n=300, hole=100, jump=1.30, level=50_000.0, seed=20260930):
    """``(clean, spliced)`` OHLC frames sharing one price path.

    The spliced index jumps ``hole`` bars after bar 150 and the prices on the
    second leg are multiplied by ``jump`` — the measured shape of the live cache
    (a 1 484 h hole with a fake ``+27 %`` bar).  ``high``/``low`` exist because the
    estimator accepts OHLC frames.
    """
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    spliced_idx = idx[: n // 2].append(
        pd.date_range(idx[n // 2 - 1] + pd.Timedelta(hours=hole),
                      periods=n - n // 2, freq="1h"))
    rng = np.random.default_rng(seed)
    close = level + np.cumsum(rng.normal(0.0, level * 0.0008, n))
    spliced_close = np.concatenate([close[: n // 2], close[n // 2:] * jump])

    def frame(index, prices):
        return pd.DataFrame(
            {"open": prices, "high": prices * 1.0005, "low": prices * 0.9995,
             "close": prices, "volume": 1.0}, index=index)

    return frame(idx, close), frame(spliced_idx, spliced_close)


class _MD:
    """Minimal market-data double: one frame for every ``get_historical`` call."""

    def __init__(self, frame):
        self.frame = frame

    async def get_historical(self, symbol, interval, limit=500):
        return self.frame.tail(limit)


class _Exec:
    """Minimal executor double: tracks ``update_stop_loss`` calls."""

    def __init__(self):
        self.updates: list[tuple[str, float]] = []

    def get_open_positions(self):
        return {}

    async def update_stop_loss(self, symbol, price):
        self.updates.append((symbol, float(price)))


def _guard(cfg, frame):
    from app.event_bus import EventBus
    from core.risk.position_guard import PositionGuard

    guard = PositionGuard(cfg, EventBus())
    execu = _Exec()
    guard.wire(execu, _MD(frame), None)
    return guard, execu


def _position(**kw):
    pos = {"entry_price": 50_000.0, "quantity": 0.1, "side": "long",
           "stop_loss": None, "timeframe": "1h"}
    pos.update(kw)
    return pos


def test_guard_refuses_a_spliced_series_and_the_stop_uses_the_fixed_rule(tmp_path):
    """R1: one spliced series must not be refused by the manager and used here.

    Before the fix the guard built its own ``DatetimeIndex`` frame and fed the
    estimator directly, so the same cache produced a forecast (measured
    ``0.062009291811329616 %/bar``) whose value drove the live trailing distance;
    the ``RiskManager`` refused the identical series with ``None``.
    """
    from core.risk.manager import _series_has_gap

    cfg = _config(tmp_path)
    cfg.risk_vol_targeting.enabled = True
    clean_frame, spliced = _vol_frames()
    assert _series_has_gap(clean_frame.index, "1h") is False
    assert _series_has_gap(spliced.index, "1h") is True, "the fixture must be spliced"

    guard, execu = _guard(cfg, spliced)
    assert asyncio.run(guard.forecast_vol_pct("BTCUSDT", "1h")) is None, (
        "the guard must refuse the series the manager refuses")
    assert asyncio.run(guard._resolve_vol_pct("BTCUSDT", _position())) is None

    # The same refusal through the real trailing-stop path: the distance is the
    # documented fixed one, not a spliced forecast.
    pos = _position()
    asyncio.run(guard._update_trailing_stop("BTCUSDT", pos, 51_000.0, "long", 0.0))
    fixed = guard._sizer.trailing_stop_distance_pct(forecast_vol_pct=None)
    assert fixed == pytest.approx(2.0), "hard_limits.trailing_stop_distance_pct"
    assert pos["stop_loss"] == pytest.approx(round(51_000.0 * (1 - fixed / 100), 2))
    # The unguarded forecast on this exact fixture, for the record: the value the
    # old code fed to the stop.
    from core.ml.volatility import forecast_vol, to_pct
    unguarded = to_pct(forecast_vol(spliced, method="ewma", window=500,
                                    lam=0.94, interval="1h"))
    assert unguarded is not None and unguarded > 0.0

    # A clean series of the same length still produces a forecast...
    guard2, _ = _guard(cfg, clean_frame)
    vol = asyncio.run(guard2.forecast_vol_pct("BTCUSDT", "1h"))
    assert vol is not None and 0.0 < vol < 5.0
    # ... and the manager agrees with the guard, bit for bit, on that series.
    from app.event_bus import EventBus
    from core.risk.manager import RiskManager
    rm = RiskManager(cfg, EventBus())
    rm.wire_market_data(_MD(clean_frame))
    assert asyncio.run(rm.forecast_vol_pct("BTCUSDT", "1h")) == vol


def test_clean_series_forecast_drives_the_trailing_distance(tmp_path):
    """R1: the guard's forecast is used when (and only when) the series is sound."""
    cfg = _config(tmp_path)
    cfg.risk_vol_targeting.enabled = True
    clean_frame, _ = _vol_frames()
    guard, _ = _guard(cfg, clean_frame)
    vol = asyncio.run(guard._resolve_vol_pct("BTCUSDT", _position()))
    assert vol is not None and vol > 0.0

    pos = _position()
    asyncio.run(guard._update_trailing_stop("BTCUSDT", pos, 51_000.0, "long", 0.0))
    forecast_dist = guard._sizer.trailing_stop_distance_pct(forecast_vol_pct=vol)
    fixed_dist = guard._sizer.trailing_stop_distance_pct(forecast_vol_pct=None)
    assert forecast_dist != fixed_dist, (
        "the fixture must make the vol-scaled distance differ from the fixed one")
    assert pos["stop_loss"] == pytest.approx(
        round(51_000.0 * (1 - forecast_dist / 100), 2))


def test_vol_targeting_off_keeps_the_trailing_distance_bit_identical(tmp_path):
    """R1 constraint: the default configuration is untouched by the guard change.

    ``_resolve_vol_pct`` returns ``None`` with the switch off, so the distance is
    computed by exactly the same expression as before P3 — asserted with ``==``,
    not ``approx``, against a plainly constructed pre-P3 guard.
    """
    from app.event_bus import EventBus
    from core.risk.position_guard import PositionGuard
    from core.risk.position_sizer import PositionSizer

    cfg = _config(tmp_path)
    assert cfg.risk_vol_targeting.enabled is False
    clean_frame, spliced = _vol_frames()

    for frame in (clean_frame, spliced):
        guard, _ = _guard(cfg, frame)
        assert asyncio.run(guard.forecast_vol_pct("BTCUSDT", "1h")) is None
        assert asyncio.run(guard._resolve_vol_pct("BTCUSDT", _position())) is None

    guard, _ = _guard(cfg, spliced)
    legacy = PositionSizer(cfg.hard_limits, cfg.soft_params)
    assert (guard._sizer.trailing_stop_distance_pct(forecast_vol_pct=None)
            == legacy.trailing_stop_distance_pct(forecast_vol_pct=None))

    pos, legacy_pos = _position(), _position()
    asyncio.run(guard._update_trailing_stop("BTCUSDT", pos, 51_000.0, "long", 0.0))
    asyncio.run(guard._update_trailing_stop(
        "BTCUSDT", legacy_pos, 51_000.0, "long", 0.0))
    assert pos["stop_loss"] == legacy_pos["stop_loss"]
    # The pre-P3 trailing rule, computed here from first principles: a 2 % trailing
    # distance from the current price (the "at worst 1 % below entry" floor is
    # slack at 51 000 and does not bind).
    assert pos["stop_loss"] == 51_000.0 * (1 - 2.0 / 100)
    assert pos["stop_loss"] == 49_980.0


# ══════════════════════════════════════════════════════════════════════
# R2 — the doc-13 impact arithmetic (the live window is a measurement)
# ══════════════════════════════════════════════════════════════════════

def test_doc13_impact_matches_the_published_formula_and_k0_is_bit_identical():
    """R2: the numbers in doc 13 are the formula's, on a window named in the doc.

    The audited defect was a set of USDT deltas that belonged to *no* stated
    window.  Those absolute deltas are re-measured in the doc against a named
    window (a live value that moves as the service appends bars), so what is
    pinned here is the closed form the doc publishes plus the ``k = 0``
    bit-identity — the part that must never drift.
    """
    from app.config import Config
    from core.backtest.cost_model import apply_trading_costs
    from core.risk.liquidity import participation_pct, total_impact_usdt

    Config._instance = None
    cfg = Config.load("sim")
    volume = 751_885_467.0637        # the window named in doc 13 §3
    price = 83_043.14
    for qty, k, expected in ((0.01, 0.1, 0.0017), (1.0, 0.1, 1.7455),
                             (10.0, 0.1, 55.1963), (1.0, 0.5, 8.7273),
                             (100.0, 0.1, 1_745.4596)):
        entry = qty * price
        part = participation_pct(entry, volume) / 100.0
        closed_form = 2.0 * entry * (k * math.sqrt(part)) / 100.0
        got = total_impact_usdt(entry, entry, volume, k)
        assert got == pytest.approx(closed_form, rel=1e-12)
        assert got == pytest.approx(expected, abs=5e-5), (qty, k, got)

    # k = 0 (and a missing window) is the legacy sum, bit for bit.
    legacy = apply_trading_costs(price, price * 1.02, 1.0, "BTCUSDT", cfg)
    k0_with_volume = apply_trading_costs(price, price * 1.02, 1.0, "BTCUSDT",
                                         cfg, recent_quote_volume=volume)
    assert legacy == k0_with_volume
    assert total_impact_usdt(price, price, volume, 0.0) == 0.0
    assert total_impact_usdt(price, price, None, 0.1) == 0.0


def test_doc13_states_the_window_it_measures(tmp_path):
    """R2: the doc must name the window its percentages belong to.

    The old *table rows* must be gone (the prose is allowed to quote the retired
    figures to explain why they were wrong — that is the audit trail).
    """
    text = (REPO / "docs" / "core-algorithms"
            / "13-volume-liquidity-costs.md").read_text(encoding="utf-8")
    assert "| 1.0 BTC | 7 567.7708 USDT | 7 746.6673 | +178.8965" not in text, (
        "the unreproducible row must be gone")
    assert "| **impact** | **0.8288** |" not in text
    assert "75.6777 USDT" not in text
    assert "751,885,467.06" in text, "the corrected window must be named"
    assert "window 751,885,467.06  ends 2026-09-30 06:00:00" in text
    assert "| 1.0 BTC (83 043.14 USDT) | 0.011045 %" in text
    assert "0.8281" in text


# ══════════════════════════════════════════════════════════════════════
# R3 — a never-dirtied cache file is deduped by the periodic flush
# ══════════════════════════════════════════════════════════════════════

CLOSE_OFFSET_1H = pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)


def _twin_frame(tmp_path, head_rows=345, twins=55):
    """``(root, path, repaired, twin)`` — ``head_rows`` clean bars + ``twins`` twice.

    Writes the *live-shaped* file (both conventions present) to disk and returns
    the repaired one-row-per-bar reference alongside it.
    """
    root = tmp_path / "data"
    path = root / "market" / "BTCUSDT" / "1h.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    head = pd.date_range("2025-06-03 00:00", periods=head_rows, freq="h")
    twin = pd.date_range(head[-1] + pd.Timedelta(hours=1), periods=twins, freq="h")

    def frame(stamps, base):
        step = np.arange(len(stamps), dtype=float)
        return pd.DataFrame({"open": base + step, "high": base + step + 1,
                             "low": base + step - 1, "close": base + step + 0.5,
                             "volume": 1.0 + step}, index=pd.DatetimeIndex(stamps))

    repaired = pd.concat([frame(head, 100.0), frame(twin, 500.0)]).sort_index()
    stale = frame(twin + CLOSE_OFFSET_1H, 900.0)
    live_shaped = pd.concat([repaired, stale]).sort_index()
    live_shaped.to_parquet(path)
    return root, path, repaired, twin


def _twin_counts(frame):
    from core.market_data.ohlcv_cache import bar_keys
    keys = bar_keys(frame.index, "1h")
    dup = keys.duplicated(keep=False)
    return len(frame), int(dup.sum()), int(keys[dup].nunique())


def test_flush_dedupes_a_key_that_was_never_dirty(tmp_path):
    """R3: loading the file is enough to repair it — no append required.

    The measured live defect: ``flush_all`` only rewrote dirty keys, so a file
    that is never appended to again keeps its duplicate bars for ever.
    """
    from core.market_data.ohlcv_cache import OHLVCache

    root, path, repaired, twin = _twin_frame(tmp_path)
    before = pd.read_parquet(path)
    # 345 single-convention bars + 55 bars stored under both conventions
    # (55 bar-open rows + 55 close_time rows) = 455 rows / 110 twin rows / 55 bars.
    assert _twin_counts(before) == (455, 110, 55)
    assert len(repaired) == 400, "one row per bar in the reference frame"

    cache = OHLVCache(str(root))
    cache.get("BTCUSDT", "1h")          # loaded once, nothing appended
    assert cache._dirty == set(), "the defect's precondition: no dirty key"
    cache.flush_all()                   # every 300 s in the service

    after = pd.read_parquet(path)
    # 400 rows: one per bar (the 110 twin rows collapse to the 55 bar-open rows
    # the fresher close_time row canonicalises onto).
    assert _twin_counts(after) == (400, 0, 0), "the flush did not dedupe"
    assert after.index.is_unique
    from core.market_data.ohlcv_cache import bar_keys
    assert set(bar_keys(after.index, "1h").asi8) == set(
        bar_keys(repaired.index, "1h").asi8), "a bar disappeared"
    # The fresher row wins the conflict (the stale frame's values).
    for stamp in twin:
        assert after.loc[stamp, "close"] == pytest.approx(
            900.0 + float(twin.get_loc(stamp)) + 0.5)


def test_flush_does_not_rewrite_an_already_deduped_file(tmp_path):
    """R3 constraint: no needless rewrites — the bytes must be untouched."""
    import hashlib

    from core.market_data.ohlcv_cache import OHLVCache

    root, path, repaired, _ = _twin_frame(tmp_path)
    cache = OHLVCache(str(root))
    cache.get("BTCUSDT", "1h")
    assert cache.dedupe("BTCUSDT", "1h") is True     # first pass collapses them
    before_bytes = path.read_bytes()
    before_hash = hashlib.sha256(before_bytes).hexdigest()

    cache2 = OHLVCache(str(root))
    cache2.get("BTCUSDT", "1h")
    assert cache2.dedupe("BTCUSDT", "1h") is False, "a no-op pass must not write"
    cache2.flush_all()
    assert path.read_bytes() == before_bytes, "an already-deduped file was rewritten"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    assert _twin_counts(pd.read_parquet(path))[2] == 0


def test_dirty_key_still_writes_and_merges(tmp_path):
    """R3 regression guard: ``save`` still writes a dirty key (the old contract)."""
    from core.market_data.ohlcv_cache import OHLVCache

    root, path, repaired, _ = _twin_frame(tmp_path)
    cache = OHLVCache(str(root))
    cache.get("BTCUSDT", "1h")
    new_ts = repaired.index[-1] + pd.Timedelta(hours=1)
    cache.append_candle("BTCUSDT", "1h", {
        "close_time": int(new_ts.value // 10**6),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    assert cache._dirty == {("BTCUSDT", "1h")}
    cache.flush_all()
    assert cache._dirty == set()
    merged = pd.read_parquet(path)
    assert merged.index[-1] == new_ts
    assert _twin_counts(merged)[2] == 0


# ══════════════════════════════════════════════════════════════════════
# R4 — the PSR floor is a real second condition (skew/kurtosis read)
# ══════════════════════════════════════════════════════════════════════

def _fat_left_returns(n=4000, outliers=10, jump=40.0, target_t=2.0, seed=20260930):
    """Net-trade returns with an exactly ``target_t`` t-stat and a fat left tail."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x[:outliers] = -abs(jump)
    x = (x - x.mean()) / x.std(ddof=1)
    sd = 0.005
    return x * sd + target_t * sd / math.sqrt(n)


def test_psr_floor_is_stricter_than_the_t_floor_for_fat_tails():
    """R4: the gate's ``AND`` is no longer equivalent to ``t > 2``.

    The audited state was ``probabilistic_sharpe = Φ(mean/se)`` — no skew, no
    kurtosis — while the docstrings justified the second floor with "PSR reads
    skew and kurtosis".  Measured: a fat-left-tailed sample at ``t = 2.0`` has PSR
    **0.9480** (refused) where the normal approximation reports 0.9772 (allowed).
    """
    from core.ml.credibility import (GATE_MIN_PSR, credibility_gate,
                                     net_trade_stats, probabilistic_sharpe)

    returns = _fat_left_returns()
    stats = net_trade_stats(returns, np.ones(len(returns), bool), +1, cost_pct=0.0)
    assert stats["t_stat"] == pytest.approx(2.0, abs=1e-6)
    plain = probabilistic_sharpe(len(returns), stats["mean"], stats["sd"])
    assert plain == pytest.approx(0.9772, abs=1e-3)
    assert stats["psr"] < GATE_MIN_PSR, (
        "the corrected PSR must refuse where the normal approximation passes")
    assert stats["psr"] == pytest.approx(0.9480, abs=2e-3)

    metrics = {"auc": 0.62, "n": 5000, "n_trades": 300, "cost_pct": 0.26}
    refused = credibility_gate(dict(metrics), 0.003, t_stat=stats["t_stat"],
                               psr=stats["psr"], n_oos=5000)
    assert refused["allowed"] is False and "not significant" in refused["reason"]

    # The same tail shape with a stronger mean still passes: the floor is a
    # *tail* condition, not a blanket refusal.
    strong = _fat_left_returns(target_t=3.0)
    strong_stats = net_trade_stats(strong, np.ones(len(strong), bool), +1,
                                   cost_pct=0.0)
    assert strong_stats["psr"] > GATE_MIN_PSR
    assert credibility_gate(dict(metrics), 0.003, t_stat=strong_stats["t_stat"],
                            psr=strong_stats["psr"], n_oos=5000)["allowed"] is True


def test_psr_normal_case_is_unchanged():
    """R4 constraint: a normal sample gets exactly the old number."""
    from core.ml.credibility import probabilistic_sharpe

    # No returns supplied → the pre-fix expression, unchanged.
    assert probabilistic_sharpe(100, 2 * 0.005 / 10, 0.005) == pytest.approx(
        0.9772, abs=1e-3)
    assert probabilistic_sharpe(100, 0.0, 0.005) == pytest.approx(0.5)
    assert probabilistic_sharpe(0, 0.0, 0.0) == 0.0
    assert probabilistic_sharpe(10, 0.01, 0.0) == 0.0

    # A normal sample supplied as returns lands on the same value (γ₃=0, γ₄=3).
    rng = np.random.default_rng(7)
    n = 20_000
    x = rng.normal(size=n)
    x = (x - x.mean()) / x.std(ddof=1) * 0.005 + 2 * 0.005 / math.sqrt(n)
    with_returns = probabilistic_sharpe(n, float(x.mean()), float(x.std(ddof=1)),
                                        returns=x)
    plain = probabilistic_sharpe(n, float(x.mean()), float(x.std(ddof=1)))
    assert with_returns == pytest.approx(plain, abs=2e-3)


# ══════════════════════════════════════════════════════════════════════
# R5 — a sidecar without `feature_schema_hash` is refused
# ══════════════════════════════════════════════════════════════════════

class _StubModel:
    def predict_proba(self, X):
        return np.tile(np.array([[0.4, 0.6]]), (len(X), 1))


def _write_artefact(models_dir: Path, stem: str, meta: dict | None):
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{stem}.pkl").write_bytes(pickle.dumps(_StubModel()))
    if meta is not None:
        (models_dir / f"{stem}_meta.json").write_text(
            json.dumps(meta), encoding="utf-8")
    return models_dir / f"{stem}.pkl"


def _engine(tmp_path):
    from core.backtest.engine import BacktestEngine
    cfg = _config(tmp_path)
    engine = BacktestEngine(cfg, None, None, None)
    engine._current_strategies = []
    engine.config = cfg
    return engine, cfg


def test_preload_refuses_a_sidecar_without_a_schema_hash(tmp_path):
    """R5: the audited `if stored and ...` accepted it (2 of 7 artefacts loaded)."""
    from core.ml.features import FEATURE_NAMES, feature_schema_hash

    models_dir = tmp_path / "data" / "models"
    good = feature_schema_hash(FEATURE_NAMES)
    no_hash = _write_artefact(models_dir, "BTCUSDT_nohash_binary", meta={
        "feature_names": list(FEATURE_NAMES), "train_base_rate": 0.5,
        "gate": {"allowed": True, "reason": "pass", "auc": 0.61}})
    empty_hash = _write_artefact(models_dir, "BTCUSDT_empty_binary", meta={
        "feature_names": list(FEATURE_NAMES), "feature_schema_hash": "",
        "gate": {"allowed": True, "reason": "pass"}})
    complete = _write_artefact(models_dir, "BTCUSDT_ok_binary", meta={
        "feature_names": list(FEATURE_NAMES), "feature_schema_hash": good,
        "train_base_rate": 0.51,
        "gate": {"allowed": True, "reason": "pass", "auc": 0.61,
                 "net_expectancy": 0.003, "n_oos": 3000}})

    engine, _ = _engine(tmp_path)
    for path in (no_hash, empty_hash):
        ok, reason = engine._verify_ml_model_sidecar(path.stem, path, "binary")
        assert ok is False, "a missing schema hash must not verify"
        assert "feature schema hash missing" in reason
    ok, reason = engine._verify_ml_model_sidecar(complete.stem, complete, "binary")
    assert ok is True and reason == "verified"


def test_predictor_refuses_a_sidecar_without_a_schema_hash(tmp_path):
    """R5 on the live path: same rule, as a ``FeatureContractError``."""
    from core.ml.features import FEATURE_NAMES, feature_schema_hash
    from core.ml.predictor import FeatureContractError, MLPredictor

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
    (tmp_path / "BTCUSDT_default_binary.pkl").write_bytes(
        pickle.dumps(_StubModel()))
    meta_path = tmp_path / "BTCUSDT_default_binary_meta.json"
    meta_path.write_text(json.dumps({
        "feature_names": list(FEATURE_NAMES),
        "gate": {"allowed": True, "reason": "pass", "auc": 0.6,
                 "net_expectancy": 0.001},
        "train_base_rate": 0.5,
    }), encoding="utf-8")
    with pytest.raises(FeatureContractError, match="feature schema"):
        predictor.load_model("BTCUSDT", str(tmp_path / "BTCUSDT_default_binary.pkl"))

    # A complete sidecar passes the contract check and reaches the model load.
    meta_path.write_text(json.dumps({
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_hash": feature_schema_hash(FEATURE_NAMES),
        "gate": {"allowed": True, "reason": "pass", "auc": 0.6,
                 "net_expectancy": 0.001},
        "train_base_rate": 0.5,
    }), encoding="utf-8")
    model = predictor.load_model("BTCUSDT",
                                 str(tmp_path / "BTCUSDT_default_binary.pkl"))
    assert model is not None, "a complete sidecar must load"
    assert predictor._status["allowed"] is True


def test_engine_preload_accepts_a_complete_sidecar(tmp_path):
    """R5: the positive control — nothing was tightened beyond the missing hash."""
    from core.ml.features import FEATURE_NAMES, feature_schema_hash

    models_dir = tmp_path / "data" / "models"
    path = _write_artefact(models_dir, "BTCUSDT_gamma_binary", meta={
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_hash": feature_schema_hash(FEATURE_NAMES),
        "train_base_rate": 0.51,
        "gate": {"allowed": True, "reason": "pass", "auc": 0.61,
                 "net_expectancy": 0.003, "n_oos": 3000}})
    engine, _ = _engine(tmp_path)
    ok, reason = engine._verify_ml_model_sidecar("BTCUSDT_gamma_binary", path,
                                                 "binary")
    assert ok is True and reason == "verified"


# ══════════════════════════════════════════════════════════════════════
# R6/R2 — the evidence index is pinned to the revision under audit
# ══════════════════════════════════════════════════════════════════════

def _git(args, *, cwd: Path = REPO):
    """Run one git command from the worktree; ``None`` when git cannot run.

    ``None`` means *git is not usable here* (not installed, or this checkout was
    copied without ``.git``) -- never "the command failed", which is a defect the
    guard must report rather than swallow.
    """
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_is_installed() -> bool:
    """True when a ``git`` executable is on PATH."""
    return shutil.which("git") is not None


def _git_or_fail(args) -> str:
    value = _git(args)
    if value is None:
        if _git_is_installed():
            pytest.fail(
                f"git is installed but cannot read this worktree "
                f"('git {' '.join(args)}' failed): the evidence guard needs the "
                "commit history, so an unreadable checkout is a failure rather "
                "than a skip")
        pytest.fail("git is not installed, yet the worktree is not readable "
                    "either; the evidence guard cannot recompute the revision")
    return value


def _commit_chain(text: str) -> list[str]:
    """The hash of every row of the document's chain table, in table order."""
    rows = re.findall(r"^\|\s*`([0-9a-f]{7,40})`\s*\|", text, re.MULTILINE)
    if not rows:
        pytest.fail("the evidence index has no commit-chain table")
    return rows


def test_evidence_index_reports_the_current_revision_and_a_current_chain():
    """R2/R1: the evidence index must track the revision under audit.

    The previous guard ran ``git rev-parse HEAD`` and then asserted only that it
    was *truthy* before requiring a fixed list of hashes ending at ``b49883b``.
    That is the mechanism that let the document go stale three times: the guard
    passed at HEAD ``5f50771`` while the index was five commits behind, because
    the assertion never mentioned the revision it was supposed to be checking.

    Both facts are recomputed from the worktree at run time -- the current HEAD
    must appear somewhere in the document (either as the audited revision in the
    header or as the newest row of the chain), and the chain table must have
    exactly ``git rev-list --count f1f6a4c^..HEAD`` rows -- so a stale index
    fails here instead of five commits later.

    Run against a deliberately rolled-back copy of the document, this test
    fails; it is never skipped to make that copy pass.  ``pytest.skip`` happens
    in exactly one situation: ``git`` is **not installed at all**, so no
    revision can be computed anywhere.  A checkout that simply has no ``.git``
    while ``git`` *is* on PATH -- a copy of the tree -- is a **failure**, not a
    skip: silently passing there is how a stale index stayed green.
    """
    if _git(["rev-parse", "--is-inside-work-tree"]) is None \
            and not _git_is_installed():
        pytest.skip("git is not installed: the evidence index's revision "
                    "cannot be recomputed on this machine")

    head = _git_or_fail(["rev-parse", "HEAD"])
    text = _evidence_doc().read_text(encoding="utf-8")

    chain = _commit_chain(text)
    expected = int(_git_or_fail(
        ["rev-list", "--count", f"{EVIDENCE_BASELINE}^..HEAD"]))

    # ── 1. the chain table matches the repository, row for row ───────────
    if len(chain) != expected:
        pytest.fail(
            f"the evidence index's commit chain has {len(chain)} row(s) but "
            f"`git rev-list --count {EVIDENCE_BASELINE}^..HEAD` = {expected}: "
            f"missing={sorted(set(_git_or_fail(['log', '--format=%h',
                                               f'{EVIDENCE_BASELINE}^..HEAD'])
                                   .split()) - set(chain))}, "
            f"table={chain}")

    # ── 2. the audited revision is named in the document ─────────────────
    hashes = set(re.findall(r"`([0-9a-f]{7,40})`", text))
    short = head[:7]
    if short not in hashes:
        pytest.fail(
            f"the evidence index is stale: it never names the current HEAD "
            f"{short} ({head}). State the audited revision explicitly -- either "
            f"the header's 基线提交 or the newest row of the chain table -- and, "
            f"when the document's own edits are doc-only, say so (the audited "
            f"code revision is then the newest commit that touched code, not "
            f"this doc-only commit)")

    # ── 3. the chain starts at the plan freeze, with no gaps ─────────────
    assert chain[0].startswith(EVIDENCE_BASELINE), (
        f"the chain table must start at the plan freeze {EVIDENCE_BASELINE}; "
        f"it starts at {chain[0]}")
    recorded = set(_git_or_fail(
        ["log", "--format=%h", f"{EVIDENCE_BASELINE}^..HEAD"]).split())
    missing = recorded - set(chain)
    if missing:
        pytest.fail(f"the chain table omits {len(missing)} commit(s) of "
                    f"{EVIDENCE_BASELINE}^..HEAD: {sorted(missing)}")

    # ── 4. the stale "uncommitted" claims are gone ───────────────────────
    # These are the concrete strings the stale index carried, including §9's
    # "全部数字均为本轮实测；未提交" for work that had already landed in
    # `828e375`.  A semantic "no landed commit is called uncommitted" scan is
    # not attempted: the document legitimately *quotes* old states while
    # correcting them (§9 R6 records that the header said "最终审计（未提交）",
    # and §11 R1 quotes it again), and a regex cannot tell a quote from a claim.
    assert "全部数字均为本轮实测；未提交" not in text, (
        "the §9 header still calls landed work '未提交'; it landed in 828e375")
    assert "| 最终审计（未提交） |" not in text, (
        "the stale commit-list row must be gone")
    assert "新证据文件：`tests/test_reaudit_fixes.py`。全部数字均为该轮实测；未提交" \
        not in text, "the §9 archive is still described as uncommitted"
    assert "55 个 `twin` 已由并行缓存修复合并" not in text, (
        "the false twin claim must be corrected")
    assert "基线提交**: 本次审计 `fe11ccf`" not in text, (
        "the header must no longer pin the superseded audit revision")
    assert "tests/test_reaudit_fixes.py" in text


# ══════════════════════════════════════════════════════════════════════
# R7 — `from core.ml.credibility import *` must work
# ══════════════════════════════════════════════════════════════════════

def test_star_import_of_credibility_resolves():
    """R7: ``__all__`` exported the undefined ``signed_score``."""
    namespace: dict = {}
    exec("from core.ml.credibility import *", namespace)  # noqa: S102
    import core.ml.credibility as credibility

    missing = [name for name in credibility.__all__
               if not hasattr(credibility, name)]
    assert missing == [], f"__all__ names that do not resolve: {missing}"
    assert "signed_score" not in credibility.__all__, (
        "signed_score lives in core.ml.calibration, not here")
    for name in credibility.__all__:
        assert name in namespace, f"{name} was not exported by the star import"
    assert namespace["credibility_gate"] is credibility.credibility_gate
    # The symbol still exists where it belongs.
    from core.ml.calibration import signed_score
    assert callable(signed_score)
