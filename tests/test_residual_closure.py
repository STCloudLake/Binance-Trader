"""Closure-audit residuals at revision ``09125bd`` — D1 and LOW-6.

**D1 (MEDIUM).** The splice guard false-positived on the live cache.
``data/market/BTCUSDT/1h.parquet`` held 11 624 rows: 11 623 bar-*open* stamps plus
**one** ``close_time`` stamp (``2026-09-30 07:59:59.999``).  Differencing raw
stamps made the adjacent step ``06:00:00 → 07:59:59.999`` = **1.99997 h**, above the
``_VOL_MAX_GAP_BARS = 1.5`` limit, so ``_series_has_gap`` returned ``True`` for a
series that is contiguous on the declared bar grid (folded: max step 1.0 h, 0
gaps).  Both ``RiskManager`` and ``PositionGuard`` therefore refused the forecast
(measured below), and ``scripts/check_data_integrity.py`` reported
``BTCUSDT/1h`` as GAP / missing 1 — 25/29 files flagged instead of 24/29.

The fix folds every stamp onto its bar key (:func:`core.market_data.ohlcv_cache
.bar_keys` — the same rule the cache's write path and
``scripts/download_history.py`` apply) before differencing, in both the guard and
the integrity reporter.  A genuine hole is a 2.0 h step on the folded keys and is
still refused.

**LOW-6.** ``_series_has_gap`` used to swallow the exception a non-datetime index
raises and return ``False`` — "no gap" — so a caller that ever passed a
``RangeIndex`` disarmed the guard (audit F1 measured exactly that).  An unreadable
index is now a **refusal** (``True``): bare integers are 1970 nanosecond epochs to
pandas, so there is no safe silent conversion.

Nothing here writes ``data/`` (only the live parquet is *read*), ``strategies/`` or
the live DB.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

#: Binance ``close_time`` minus bar open for a 1 h bar: ``open + 1 h − 1 ms``.
CLOSE_OFFSET_1H = pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)


def _close_stamp(bar_open):
    """``close_time`` of the 1 h bar that opens at ``bar_open``."""
    return pd.Timestamp(bar_open) + CLOSE_OFFSET_1H


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    from app.config import Config
    yield
    Config._instance = None


def _config(tmp_path):
    from app.config import Config
    Config._instance = None
    cfg = Config.load("sim")
    cfg.db_path = str(tmp_path / "residual_closure.db")
    return cfg


def _walk(n: int, level: float = 50_000.0, seed: int = 20260930) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return level + np.cumsum(rng.normal(0.0, level * 0.0008, n))


def _ohlc(stamps, prices):
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": prices, "high": prices * 1.0005,
                         "low": prices * 0.9995, "close": prices,
                         "volume": 1.0}, index=pd.DatetimeIndex(stamps))


def _live_shaped(n: int = 300):
    """The measured live shape: ``n`` open-convention bars + one close-convention.

    Mirrors ``data/market/BTCUSDT/1h.parquet`` at revision ``09125bd`` (11 623
    open stamps + 1 ``:59:59.999`` tail stamp, 11 623 h span, no hole), scaled down
    so the fixture stays fast.  The tail stamp is the bar **after** the last open
    stamp, so it is a distinct bar in the other convention, not a duplicate.
    """
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    stamps = idx.append(pd.DatetimeIndex([_close_stamp(idx[-1] + pd.Timedelta(hours=1))]))
    return stamps, _ohlc(stamps, _walk(len(stamps)))


def _one_missing_bar(n: int = 300):
    """``n`` bars on the grid with bar 150 deleted, plus the close-convention tail."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1h").delete(150)
    stamps = idx.append(pd.DatetimeIndex([_close_stamp(idx[-1] + pd.Timedelta(hours=1))]))
    return stamps, _ohlc(stamps, _walk(len(stamps)))


def _spliced(n: int = 300, hole: int = 100, jump: float = 1.30):
    """The audited whole-series splice: one ``hole``-bar hole and a fake ``+30 %`` bar."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    stamps = idx[: n // 2].append(
        pd.date_range(idx[n // 2 - 1] + pd.Timedelta(hours=hole),
                      periods=n - n // 2, freq="1h"))
    prices = _walk(n)
    prices[n // 2:] = prices[n // 2:] * jump
    return stamps, _ohlc(stamps, prices)


class _MD:
    """Minimal market-data double: one frame for every ``get_historical`` call."""

    def __init__(self, frame):
        self.frame = frame

    async def get_historical(self, symbol, interval, limit=500):
        return self.frame.tail(limit)


class _Exec:
    """Minimal executor double — the guard only needs ``wire`` to accept it."""

    def get_open_positions(self):
        return {}

    async def update_stop_loss(self, symbol, price):
        return None


def _manager(cfg, frame):
    from app.event_bus import EventBus
    from core.risk.manager import RiskManager

    rm = RiskManager(cfg, EventBus())
    rm.wire_market_data(_MD(frame))
    return rm


def _guard(cfg, frame):
    from app.event_bus import EventBus
    from core.risk.position_guard import PositionGuard

    guard = PositionGuard(cfg, EventBus())
    guard.wire(_Exec(), _MD(frame), None)
    return guard


# ══════════════════════════════════════════════════════════════════════
# D1 (a) — the live-shaped frame is contiguous and produces a forecast
# ══════════════════════════════════════════════════════════════════════

def test_live_shaped_close_convention_tail_is_not_a_gap(tmp_path):
    """D1(a): 11 623 open stamps + one close stamp → no gap, forecast produced.

    Before the fix this exact shape measured ``_series_has_gap → True`` (raw delta
    1.99997 h) and both consumers returned ``None``.
    """
    from core.risk.manager import _series_has_gap
    from scripts.check_data_integrity import gap_report

    stamps, frame = _live_shaped()
    assert str(stamps[-1]).endswith(":59:59.999000"), "the fixture's close stamp"
    assert (stamps[-1] - stamps[-2]).total_seconds() / 3600.0 > 1.5, (
        "the defect's precondition: the raw adjacent step exceeds 1.5 bars")
    assert len(stamps) == len(frame) == 301

    assert _series_has_gap(stamps, "1h") is False, "the folded series is contiguous"

    cfg = _config(tmp_path)
    cfg.risk_vol_targeting.enabled = True
    vol = asyncio.run(_manager(cfg, frame).forecast_vol_pct("BTCUSDT", "1h"))
    assert vol is not None and 0.0 < vol < 5.0, vol
    # The guard sees the same series and reaches the same number, bit for bit.
    assert asyncio.run(_guard(cfg, frame).forecast_vol_pct("BTCUSDT", "1h")) == vol

    # D1(d): the integrity report agrees — ok, no missing bar, no gap.
    rep = gap_report(frame, "1h")
    assert rep["bars"] == 301
    assert rep["gap_count"] == 0 and rep["missing"] == 0 and rep["flagged"] is False
    assert rep["expected"] == 301, (rep["span_hours"], rep["expected"])


def test_live_file_shape_is_read_from_disk_if_present():
    """D1(a) on the real cache: the shipped ``BTCUSDT/1h`` parquet must be ok.

    Skipped when the (gitignored) file is absent.  This is the file the audit
    measured; a *genuine* future hole is allowed to make this test fail — that is
    the point of the reporter — so the assertion is on the folded rule, not on a
    pinned row count.
    """
    from pathlib import Path

    from core.market_data.ohlcv_cache import bar_keys
    from core.risk.manager import _series_has_gap
    from scripts.check_data_integrity import gap_report

    path = Path(__file__).resolve().parents[1] / "data" / "market" / "BTCUSDT" / "1h.parquet"
    if not path.exists():
        pytest.skip("no cached BTCUSDT 1h parquet in this checkout")
    frame = pd.read_parquet(path)
    keys = bar_keys(frame.index, "1h")
    diffs = pd.Series(keys).diff().dropna().dt.total_seconds().to_numpy() / 3600.0
    rep = gap_report(frame, "1h")
    if (diffs > 1.5).any():
        pytest.skip(f"the live cache now carries a genuine {diffs.max():.1f} h gap")
    assert rep["missing"] == 0 and rep["gap_count"] == 0 and rep["flagged"] is False
    assert _series_has_gap(frame.index, "1h") is False


# ══════════════════════════════════════════════════════════════════════
# D1 (b) — a genuinely missing bar is still refused
# ══════════════════════════════════════════════════════════════════════

def test_genuinely_missing_bar_is_still_a_gap_and_refuses_the_forecast(tmp_path):
    """D1(b): one missing bar → guard True, forecast ``None``, reporter flags it."""
    from core.risk.manager import _series_has_gap
    from scripts.check_data_integrity import gap_report

    stamps, frame = _one_missing_bar()
    assert _series_has_gap(stamps, "1h") is True

    cfg = _config(tmp_path)
    cfg.risk_vol_targeting.enabled = True
    assert asyncio.run(_manager(cfg, frame).forecast_vol_pct("BTCUSDT", "1h")) is None
    assert asyncio.run(_guard(cfg, frame).forecast_vol_pct("BTCUSDT", "1h")) is None

    rep = gap_report(frame, "1h")
    assert rep["flagged"] is True and rep["missing"] == 1 and rep["gap_count"] == 1
    assert rep["largest_gap_hours"] == pytest.approx(2.0)


def test_close_convention_stamp_next_to_a_hole_is_still_a_hole():
    """D1(b) edge case: folding must not swallow a real hole beside a close stamp.

    ``…05:00:00`` open stamps, then the *close* stamp of the ``08:00`` bar: the
    folded keys skip ``06:00``/``07:00``, so the step is 3.0 h — refused.
    """
    from core.risk.manager import _series_has_gap

    idx = pd.date_range("2026-01-01", periods=6, freq="1h")
    stamps = idx.append(pd.DatetimeIndex([_close_stamp(pd.Timestamp("2026-01-01 08:00"))]))
    assert _series_has_gap(stamps, "1h") is True


# ══════════════════════════════════════════════════════════════════════
# D1 (c) — a whole-series splice is still refused
# ══════════════════════════════════════════════════════════════════════

def test_whole_series_splice_is_still_refused(tmp_path):
    """D1(c): the 100-bar hole / ``+30 %`` fake bar is refused as before."""
    from core.risk.manager import _series_has_gap
    from scripts.check_data_integrity import gap_report

    stamps, frame = _spliced()
    assert _series_has_gap(stamps, "1h") is True

    cfg = _config(tmp_path)
    cfg.risk_vol_targeting.enabled = True
    assert asyncio.run(_manager(cfg, frame).forecast_vol_pct("BTCUSDT", "1h")) is None
    assert asyncio.run(_guard(cfg, frame).forecast_vol_pct("BTCUSDT", "1h")) is None

    rep = gap_report(frame, "1h")
    assert rep["flagged"] is True
    assert rep["largest_gap_hours"] == pytest.approx(100.0, abs=0.1)
    assert rep["missing"] > 90


def test_a_clean_series_still_produces_a_forecast(tmp_path):
    """D1 control: the fold is a no-op on a purely open-convention series."""
    from core.risk.manager import _series_has_gap

    idx = pd.date_range("2026-01-01", periods=300, freq="1h")
    frame = _ohlc(idx, _walk(300))
    assert _series_has_gap(idx, "1h") is False

    cfg = _config(tmp_path)
    cfg.risk_vol_targeting.enabled = True
    assert asyncio.run(_manager(cfg, frame).forecast_vol_pct("BTCUSDT", "1h")) is not None


# ══════════════════════════════════════════════════════════════════════
# LOW-6 — a non-datetime index is a refusal, not a pass
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("index,label", [
    (pd.RangeIndex(300), "RangeIndex (the audited F1 fallback)"),
    (pd.Index(np.arange(300, dtype=float)), "Float64Index"),
    (pd.Index(["a", "b", "c", "d"]), "object/str index"),
    (pd.Index([pd.Timestamp("2026-01-01"),
               pd.Timestamp("2026-01-02", tz="UTC"),
               pd.Timestamp("2026-01-03"),
               pd.Timestamp("2026-01-04")]), "mixed tz-aware/naive"),
])
def test_non_datetime_index_is_refused_not_cleared(index, label):
    """LOW-6: an unreadable index must fail closed (``True``), never pass."""
    from core.risk.manager import _series_has_gap

    assert _series_has_gap(index, "1h") is True, label


def test_unknown_interval_keeps_the_guard_inert_for_any_index():
    """LOW-6 boundary: no bar length known ⇒ the guard makes no claim (``False``)."""
    from core.risk.manager import _series_has_gap

    for index in (pd.RangeIndex(300), pd.DatetimeIndex([]),
                  pd.date_range("2026-01-01", periods=300, freq="1h")):
        assert _series_has_gap(index, "7h") is False
    # An empty/short *datetime* series has no gap to report either.
    assert _series_has_gap(pd.DatetimeIndex([]), "1h") is False
    assert _series_has_gap(pd.date_range("2026-01-01", periods=2, freq="1h"), "1h") is False


def test_position_guard_refuses_a_frame_with_a_non_datetime_index(tmp_path):
    """LOW-6 end to end: the guard's own frame source cannot clear the check."""
    cfg = _config(tmp_path)
    cfg.risk_vol_targeting.enabled = True
    prices = _walk(300)
    frame = pd.DataFrame({"open": prices, "high": prices * 1.0005,
                          "low": prices * 0.9995, "close": prices,
                          "volume": 1.0})               # the audited RangeIndex
    assert isinstance(frame.index, pd.RangeIndex)
    assert asyncio.run(_guard(cfg, frame).forecast_vol_pct("BTCUSDT", "1h")) is None
