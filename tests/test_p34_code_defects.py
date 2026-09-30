"""P3/P4 audit items 5 and 7 — the two code-level defects, pinned.

Item 7 (MED) — *the volatility MAD clip was not stable over a rolling window*.
The Winsor limit used to be re-derived from whatever window the estimator was
handed, so the same observation could be clipped in one window and not in the
next (the audit measured the limit moving on 8 344 of 8 345 consecutive 500-bar
windows).  The fix: **one anchor per series** (:func:`core.ml.volatility.
series_anchor`), built once from the whole series and applied to it by every
estimator, which then applies ``window`` *after* clipping
(:func:`core.ml.volatility._clipped`).  The tests here assert stability
(``0`` previously emitted values change) instead of counting changes, keep the
genuine-splice guard, and pin the bit-identity of the whole-series calls the
shipped numbers (and ``docs/core-algorithms/10``) are quoted from.

Item 5 (MED) — *the ``barrier_*`` config keys are inert and the forecast is never
published*.  ``risk.vol_targeting.barrier_vol_multiple`` / ``barrier_min_pct`` /
``barrier_max_pct`` are read only by ``PositionSizer.barrier_widths_pct``, which
no production code calls, so a configured barrier width is a no-op.  Wiring it
needs ``core/ml/predictor.py`` (make ``_barrier_params`` consume the resolved
forecast) or ``core/risk/manager.py`` — both outside this change's write scope —
so the defect is closed the permitted other way: the keys are marked **reserved**
and a non-default value now logs a one-time startup WARNING
(``app.config.inert_barrier_key_warnings``).  ``executor.set_forecast_vol_pct``
still has no production caller; the tripwire at the end of this file keeps that
gap visible instead of silent.  Either way, every existing path is bit-identical
while ``risk.vol_targeting.enabled`` is false.

Deterministic: fixed seeds, no sleeps, no network, no writes to ``data/``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.ml.volatility as V
from app.config import (Config, RESERVED_BARRIER_KEYS, VolTargetingConfig,
                        inert_barrier_key_warnings)

SEED = 20250930
ROOT = Path(__file__).resolve().parents[1]
BTC_1H = ROOT / "data/market/BTCUSDT/1h.parquet"

#: The audit's sample length: 8 844 returns → 8 345 rolling 500-bar windows.
AUDIT_N = 8844
AUDIT_WINDOW = 500


# ── helpers ──────────────────────────────────────────────────────────────

def _spliced_series(n: int = AUDIT_N, seed: int = SEED):
    """The audit's series length and splice (+0.276 = the calendar-gap bar).

    The shipped cache no longer carries the splice (the vendor repaired the
    1 484 h gap; ``max|log return| = 0.0494``), so the defect it guards against is
    injected — a test that needs the *shipped file* to be broken stops guarding
    anything the moment it is repaired.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.004, n)
    r[int(n * 0.55)] = 0.276
    return r


def _rolling_windows(r, *, window=AUDIT_WINDOW):
    for end in range(window, r.size + 1):
        yield end - window, r[end - window:end]


def _regime_series(n: int = AUDIT_N, seed: int = SEED):
    """The audited length with volatility regimes **and** the splice.

    ``max |Δ| = 0.048`` is the audit's figure for how far one already-emitted
    clipped value travelled between two windows.  A constant-volatility series
    cannot move a per-window MAD that far (the MAD is robust to the single
    splice: measured 1.5e-3); a real crypto series carries volatility regimes, so
    this one alternates 0.4 %/bar and 2.0 %/bar blocks and carries the splice.
    """
    rng = np.random.default_rng(seed)
    sd = np.where((np.arange(n) // 700) % 2 == 0, 0.004, 0.020)
    r = rng.normal(0.0, sd)
    r[int(n * 0.55)] = 0.276
    return r


def _clipped_value_spread(r, *, mode, window=AUDIT_WINDOW):
    """``(max spread of an emitted clipped value, the bar that spread most)``.

    Every window emits a clipped value for each bar it contains; this is the
    largest ``max − min`` any single bar's emitted value reaches across all the
    windows that contained it — the audit's "previously emitted value changed"
    magnitude, not just the consecutive-step delta.
    """
    anchor = V.series_anchor(r) if mode == "after" else None
    lo = np.full(r.size, np.inf)
    hi = np.full(r.size, -np.inf)
    for start, w in _rolling_windows(r, window=window):
        if mode == "after":
            c = V.clip_outliers(w, sigma=V.DEFAULT_OUTLIER_SIGMA, anchor=anchor)
        elif mode == "plain":
            c = V.clip_outliers(w, sigma=V.DEFAULT_OUTLIER_SIGMA, anchored=False)
        else:
            c = V.clip_outliers(w, sigma=V.DEFAULT_OUTLIER_SIGMA)
        np.minimum(lo[start:start + c.size], c, out=lo[start:start + c.size])
        np.maximum(hi[start:start + c.size], c, out=hi[start:start + c.size])
    spread = hi - lo
    bar = int(np.argmax(spread))
    return float(spread[bar]), bar


def _rolling_limit_changes(r, *, mode, window=AUDIT_WINDOW):
    """(#steps, #steps whose Winsor limit moved, #observations that changed value).

    ``mode="before"`` is the shipped pre-fix default: the limit re-derived from
    every window (``_anchored_mad`` over that window).  ``mode="before_plain"`` is
    the older per-window ``median``/``1.4826·MAD`` rule.  ``mode="after"`` is the
    shipped fix: one :func:`V.series_anchor` per series.
    """
    anchor = V.series_anchor(r) if mode == "after" else None
    seen: dict[int, float] = {}
    steps = limit_changes = obs_changes = 0
    prev_limit = None
    for start, w in _rolling_windows(r, window=window):
        steps += 1
        if mode == "after":
            limit = V.DEFAULT_OUTLIER_SIGMA * anchor.scale
            clipped = V.clip_outliers(w, sigma=V.DEFAULT_OUTLIER_SIGMA, anchor=anchor)
        elif mode == "before_plain":
            med = float(np.median(w))
            limit = V.DEFAULT_OUTLIER_SIGMA * float(np.median(np.abs(w - med))) * 1.4826
            clipped = V.clip_outliers(w, sigma=V.DEFAULT_OUTLIER_SIGMA, anchored=False)
        else:
            _med, scale = V._anchored_mad(w, V.DEFAULT_MAD_HALF_LIFE)
            limit = V.DEFAULT_OUTLIER_SIGMA * scale
            clipped = V.clip_outliers(w, sigma=V.DEFAULT_OUTLIER_SIGMA)
        if prev_limit is not None and limit != prev_limit:
            limit_changes += 1
        prev_limit = limit
        for pos, value in enumerate(clipped):
            index = start + pos
            previous = seen.get(index)
            if previous is not None and float(value) != previous:
                obs_changes += 1
            seen[index] = float(value)
    return steps, limit_changes, obs_changes


# ── 7. the clip decision must be stable per observation ──────────────────

def test_rolling_window_changes_no_previously_emitted_value():
    """The audit's experiment: 8 344 / 8 345 windows moved the limit; now 0.

    A window slides forward one bar at a time over a fixed series.  Before the
    fix the Winsor limit — and therefore the clipped value of observations the
    series had already emitted — was re-derived from every window.  With one
    anchor per series the limit is a property of the data, so the rolled-forward
    windows agree bit for bit and no emitted value moves.
    """
    r = _spliced_series()
    steps, before_changes, before_obs = _rolling_limit_changes(r, mode="before")
    assert steps == AUDIT_N - AUDIT_WINDOW + 1 == 8345
    assert before_changes == steps - 1, (
        f"{before_changes} of {steps - 1} consecutive windows moved the limit "
        "(the audit measured 8 344 of 8 345)")
    assert before_obs > 0

    steps_after, after_changes, after_obs = _rolling_limit_changes(r, mode="after")
    assert steps_after == steps
    assert (after_changes, after_obs) == (0, 0), (
        f"the anchor still moves: {after_changes} limit changes, "
        f"{after_obs} observation changes")

    # The older per-window median/MAD rule moved too (a different defect, same
    # symptom); it stays available only as the explicitly-unstable regression
    # path (`anchored=False`).
    _s, plain_changes, plain_obs = _rolling_limit_changes(r, mode="before_plain")
    assert plain_changes > 0 and plain_obs > 0


def test_no_emitted_clip_value_moves_across_windows():
    """The audit's ``max |Δ| = 0.048``: one bar, many windows, different values.

    On a regime-switching series of the audited length the pre-fix rule moved a
    single emitted clipped value by **0.088** across the windows that contained
    it (the audit measured 0.048 on its own, since-repaired cache); with one
    anchor per series the spread is exactly ``0.0`` — every window clips that bar
    to the same number.  On the shipped series the same defect reads 0.016.
    """
    r = _regime_series()
    before, bar = _clipped_value_spread(r, mode="before")
    assert before > 0.04, before
    assert r[bar] == pytest.approx(0.276)          # the spliced bar spread most
    plain, _ = _clipped_value_spread(r, mode="plain")
    assert plain > 0.04
    after, _ = _clipped_value_spread(r, mode="after")
    assert after == 0.0, f"an emitted clipped value still moves by {after}"

    if BTC_1H.exists():
        shipped = V.log_returns(pd.read_parquet(BTC_1H)["close"].values)
        shipped_before, _ = _clipped_value_spread(shipped, mode="before")
        shipped_after, _ = _clipped_value_spread(shipped, mode="after")
        assert shipped_before > 0.01
        assert shipped_after == 0.0


def test_estimators_clip_the_whole_series_and_not_the_window(monkeypatch):
    """Production wiring: the estimator builds one anchor from the series it got.

    The defect was not the *recipe* — it was that every estimator winsorised
    ``_last(returns, window)``, so the limit came from the window.  This asserts
    the wire: the clip sees the whole series and the series' own anchor, whatever
    ``window`` the caller asked for.
    """
    r = np.random.default_rng(SEED).normal(0.0, 0.004, 2000)
    r[700] = 0.276
    calls: list[tuple[int, object]] = []
    real = V.clip_outliers

    def spy(returns, *, sigma=V.DEFAULT_OUTLIER_SIGMA, anchor=None, **kw):
        arr = np.asarray(returns, dtype=float)
        calls.append((arr.size, anchor))
        return real(returns, sigma=sigma, anchor=anchor, **kw)

    monkeypatch.setattr(V, "clip_outliers", spy)
    V.ewma_vol(r, window=500)
    assert len(calls) == 1
    size, anchor = calls[0]
    assert size == r.size, "the clip was handed the 500-bar window, not the series"
    assert anchor is not None, "the estimator still used the per-window default"
    assert anchor == V.series_anchor(r)

    calls.clear()
    V.ewma_vol_series(r, window=200)
    assert len(calls) == 1 and calls[0][0] == r.size
    assert calls[0][1] == V.series_anchor(r)

    # garch11_params too.  Its own fit re-clips the tail it was handed with the
    # *fit's* looser sigma (8 vs 6) — a pre-existing step, deliberately left
    # bit-identical — so only the first call must be the whole series.
    calls.clear()
    V.garch11_params(r, window=500)
    assert calls, "garch11_params did not clip at all"
    assert calls[0] == (r.size, V.series_anchor(r))


def test_window_size_no_longer_moves_the_clip_or_the_forecast():
    """``window`` caps the estimator's memory; it must not move the Winsor limit."""
    r = _spliced_series(n=2000)
    anchor = V.series_anchor(r)
    tails = {}
    for window in (300, 500, 900):
        clipped = V._clipped(r, window=window, outlier_sigma=V.DEFAULT_OUTLIER_SIGMA)
        assert len(clipped) == window
        tails[window] = clipped[-300:]
    assert np.array_equal(tails[300], tails[500])
    assert np.array_equal(tails[300], tails[900])

    # The forecast itself: the windowed value now equals the whole-series one to
    # the seed weight of the recursion (0.94**400 ≈ 2e-11), where it used to
    # differ by ~3e-8 *because the two computed different limits*.
    whole = V.ewma_vol(r, window=0)
    assert V.ewma_vol(r, window=500) == pytest.approx(whole, rel=1e-9)
    assert V.ewma_vol(r, window=400) == pytest.approx(whole, rel=1e-9)
    # A caller that slices the series itself still needs the anchor — that is the
    # documented residual: the slice has its own anchor, and only passing the
    # *unsliced* series' anchor makes the two agree.  With a tail outlier the two
    # clips (and the two forecasts) genuinely differ.
    assert V.series_anchor(r[-500:]) != anchor
    r_tail = r.copy()
    r_tail[-10] = 0.5
    full = V.clip_outliers(r_tail, anchor=V.series_anchor(r_tail))[-500:]
    sliced = V.clip_outliers(r_tail[-500:])
    assert float(np.abs(full - sliced).max()) > 0.0
    assert V.ewma_vol(r_tail, window=500) != V.ewma_vol(r_tail[-500:], window=0)
    assert V.ewma_vol(r_tail[-500:], window=0,
                      anchor=V.series_anchor(r_tail)) == \
        pytest.approx(V.ewma_vol(r_tail, window=500), rel=1e-9)


def test_whole_series_clip_is_bit_identical_to_the_shipped_recipe():
    """No shipped number moves: the anchor *is* the recipe, computed earlier.

    ``clip_outliers(r)`` (the shipped default) and
    ``clip_outliers(r, anchor=series_anchor(r))`` must be the same array, and the
    measured figure ``docs/core-algorithms/10`` quotes for the injected splice
    (0.7553 %/bar, i.e. 8.96x unclipped) must still come out — otherwise the fix
    would have re-tuned the clip instead of stabilising it.
    """
    r = np.random.default_rng(SEED).normal(0.0, 0.002, 900)
    r[400] = 0.2
    assert np.array_equal(V.clip_outliers(r),
                          V.clip_outliers(r, anchor=V.series_anchor(r)))

    rng = np.random.default_rng(SEED)
    calm = rng.normal(0.0, 0.004, 2000)
    splice = np.concatenate([calm[:1999], np.array([0.276])])
    clipped_pct = V.to_pct(V.ewma_vol(splice, window=0))
    unclipped_pct = V.to_pct(V.ewma_vol(splice, window=0, outlier_sigma=0.0))
    assert clipped_pct == pytest.approx(0.7553, abs=5e-5)
    assert unclipped_pct / clipped_pct == pytest.approx(8.96, abs=0.01)


def test_a_genuine_splice_is_still_clipped():
    """The stable anchor must not have loosened the guard it exists for."""
    r = _spliced_series(n=3000)
    anchor = V.series_anchor(r)
    clipped = V.clip_outliers(r, anchor=anchor)
    assert np.abs(clipped).max() < np.abs(r).max()
    assert np.abs(clipped).max() < 0.05
    unclipped_arr = V.clip_outliers(r, sigma=0.0, anchor=anchor)
    assert np.array_equal(unclipped_arr, r)
    # And through the public estimator, with the splice where the EWMA looks for
    # it (the tail): one spliced bar must not dominate the forecast the sizer
    # reads (the calm series is 0.4 %/bar by construction).
    rng = np.random.default_rng(SEED)
    calm = rng.normal(0.0, 0.004, 2000)
    tail_splice = np.concatenate([calm[:1999], np.array([0.276])])
    assert V.to_pct(V.ewma_vol(tail_splice, window=0)) < 1.0
    assert V.to_pct(V.ewma_vol(tail_splice, window=0, outlier_sigma=0.0)) > \
        5.0 * V.to_pct(V.ewma_vol(tail_splice, window=0))
    # Too-short input keeps the "nothing to anchor on" contract.
    short = np.array([0.01, -0.01])
    assert V.series_anchor(short) == V.AnchorMAD(0.0, 0.0)
    assert np.array_equal(V.clip_outliers(short), short)


@pytest.mark.skipif(not BTC_1H.exists(), reason="no cached BTCUSDT 1h parquet")
def test_shipped_cache_is_clipped_bit_identically_across_windows():
    """The same contract on the live-shaped series (skipped without data/)."""
    r = V.log_returns(pd.read_parquet(BTC_1H)["close"].values)
    anchor = V.series_anchor(r)
    for window in (400, 500, 1000):
        clipped = V._clipped(r, window=window, outlier_sigma=V.DEFAULT_OUTLIER_SIGMA)
        assert np.array_equal(clipped[-400:],
                              V.clip_outliers(r, anchor=anchor)[-400:])
    assert np.abs(V.clip_outliers(r, anchor=anchor)).max() < 0.1


# ── 5. the reserved barrier keys must not be a silent no-op ──────────────

def test_reserved_barrier_warnings_are_empty_for_the_shipped_defaults():
    assert inert_barrier_key_warnings(None) == []
    assert inert_barrier_key_warnings(VolTargetingConfig()) == []
    # The shipped YAML must not itself trip the warning.
    import yaml
    raw = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    shipped = raw["risk"]["vol_targeting"]
    for key, default in RESERVED_BARRIER_KEYS.items():
        assert float(shipped[key]) == float(default), key
    assert inert_barrier_key_warnings(VolTargetingConfig(**shipped)) == []


@pytest.mark.parametrize("key,value", [
    ("barrier_vol_multiple", 2.0),
    ("barrier_min_pct", 0.01),
    ("barrier_max_pct", 0.12),
])
def test_non_default_barrier_key_warns_and_names_the_key(key, value):
    """A configured-but-inert key is reported, not ignored (audit item 5)."""
    msgs = inert_barrier_key_warnings(VolTargetingConfig(**{key: value}))
    assert len(msgs) == 1
    assert f"risk.vol_targeting.{key}={value:g}" in msgs[0]
    assert "RESERVED" in msgs[0] and "NO effect" in msgs[0]
    # It must say where to configure the width instead.
    for effective in ("ml.barrier_atr_period", "ml.barrier_atr_multiple",
                      "ml.barrier_min_pct", "ml.barrier_max_pct"):
        assert effective in msgs[0]
    assert "barrier_widths_pct" in msgs[0]


def test_startup_warning_fires_once_for_a_configured_reserved_key(monkeypatch):
    """End to end through ``Config.load``: one WARNING, one time."""
    from loguru import logger

    captured: list[str] = []
    sink = logger.add(lambda m: captured.append(m), level="WARNING")
    try:
        # A missing YAML keeps every other field at its built-in default.
        monkeypatch.setattr(Config, "_load_yaml", lambda self, path: None)
        Config._instance = None
        cfg = Config.load()
        assert cfg.risk_vol_targeting.barrier_vol_multiple == 1.0
        assert not [m for m in captured if "barrier_vol_multiple" in m]

        captured.clear()
        monkeypatch.setattr(
            Config, "_load_yaml",
            lambda self, path: self._data.update(
                {"risk": {"vol_targeting": {"barrier_vol_multiple": 2.0}}})
            if path.endswith("config.yaml") else None)
        Config._instance = None
        Config.load()
        Config.load()          # the singleton must not warn twice
        warned = [m for m in captured if "barrier_vol_multiple" in m]
        assert len(warned) == 1, warned
        assert "RESERVED" in warned[0] and "ml.barrier_min_pct" in warned[0]
    finally:
        logger.remove(sink)
        Config._instance = None


def test_switch_off_paths_are_bit_identical_whatever_the_barrier_keys_say():
    """With ``enabled: false`` every sizer/hook number is exactly unchanged."""
    from app.config import HardRiskLimits, SoftRiskParams
    from core.ml.labels import barrier_widths
    from core.risk.position_sizer import PositionSizer

    hard, soft = HardRiskLimits(), SoftRiskParams()

    def sizer(**overrides):
        return PositionSizer(hard, soft, 0.7, 0.3,
                             VolTargetingConfig(enabled=False, **overrides))

    a = sizer()
    b = sizer(barrier_vol_multiple=9.0, barrier_min_pct=0.02, barrier_max_pct=0.5,
              stop_vol_multiple=9.0, stop_min_pct=9.0, stop_max_pct=99.0,
              target_vol_pct=99.0, min_scale=0.01, max_scale=9.0)
    for forecast in (None, 0.0, 0.45, 3.0):
        assert a.calculate_position_size(10_000.0, 50_000.0, "satellite",
                                         forecast_vol_pct=forecast) == \
            b.calculate_position_size(10_000.0, 50_000.0, "satellite",
                                      forecast_vol_pct=forecast)
        assert a.stop_distance_pct(forecast) == b.stop_distance_pct(forecast)
        assert a.trailing_stop_distance_pct(forecast_vol_pct=forecast) == \
            b.trailing_stop_distance_pct(forecast_vol_pct=forecast)
        assert a.calculate_stop_loss(100.0, "long", forecast_vol_pct=forecast) == \
            b.calculate_stop_loss(100.0, "long", forecast_vol_pct=forecast)
        assert a.barrier_widths_pct(forecast) is None
        assert b.barrier_widths_pct(forecast) is None

    df = pd.DataFrame({
        "open": np.linspace(100.0, 101.0, 60),
        "high": np.linspace(100.5, 101.5, 60),
        "low": np.linspace(99.5, 100.5, 60),
        "close": np.linspace(100.0, 101.0, 60) + 0.02,
    })
    up_a, dn_a = barrier_widths(df)
    up_b, dn_b = barrier_widths(df)
    assert up_a.equals(up_b) and dn_a.equals(dn_b)


def test_the_reserved_keys_do_change_the_width_once_the_switch_is_on():
    """The mechanism under the reserved keys works — that is *why* they mislead.

    ``barrier_widths_pct`` honours ``barrier_vol_multiple`` (and
    ``core.ml.labels.barrier_widths`` honours the same number as ``vol_multiple``);
    nothing on the live path calls either with the resolved forecast.  This test
    fixes both halves of the statement so the reserved warning cannot be quietly
    dropped.
    """
    from app.config import HardRiskLimits, SoftRiskParams
    from core.ml.labels import barrier_widths
    from core.risk.position_sizer import PositionSizer

    cfg = VolTargetingConfig(enabled=True, barrier_vol_multiple=1.0,
                             barrier_min_pct=0.0001, barrier_max_pct=0.5)
    sizer = PositionSizer(HardRiskLimits(), SoftRiskParams(), 0.7, 0.3, cfg)
    one = sizer.barrier_widths_pct(0.45)[0]
    cfg.barrier_vol_multiple = 3.0
    three = sizer.barrier_widths_pct(0.45)[0]
    assert three == pytest.approx(3.0 * one)

    df = pd.DataFrame({
        "open": np.linspace(100.0, 101.0, 60),
        "high": np.linspace(100.5, 101.5, 60),
        "low": np.linspace(99.5, 100.5, 60),
        "close": np.linspace(100.0, 101.0, 60) + 0.02,
    })
    atr_up, _ = barrier_widths(df, min_pct=0.0001, max_pct=0.5)
    fc_one, _ = barrier_widths(df, min_pct=0.0001, max_pct=0.5,
                               vol_pct=one, vol_multiple=1.0)
    fc_three, _ = barrier_widths(df, min_pct=0.0001, max_pct=0.5,
                                 vol_pct=one, vol_multiple=3.0)
    assert not atr_up.equals(fc_one)
    assert fc_three.iloc[0] == pytest.approx(3.0 * fc_one.iloc[0])


def test_forecast_publication_gap_is_registered_not_silent():
    """``executor.set_forecast_vol_pct`` still has no production caller.

    The push channel exists (``RiskManager.resolve_forecast_vol_pct`` reads
    ``OrderExecutor.vol_stop_ctx``, which reads the published cache), but nothing
    publishes into it.  Wiring it needs ``core/risk/manager.py`` or
    ``core/ml/predictor.py``; both are outside this change's write scope, so the
    gap is asserted here rather than left to be rediscovered.  **When the wiring
    lands, update this test** — it is a report, not a requirement.
    """
    callers: list[str] = []
    for path in list((ROOT / "core").rglob("*.py")) + list((ROOT / "app").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\.set_forecast_vol_pct\s*\(", text):
            callers.append(path.relative_to(ROOT).as_posix())
    assert callers == [], f"set_forecast_vol_pct gained a caller: {callers}"

    executor = (ROOT / "core/executor/executor.py").read_text(encoding="utf-8")
    assert "def set_forecast_vol_pct" in executor
    assert "_cached_forecast_vol_pct" in executor
    manager = (ROOT / "core/risk/manager.py").read_text(encoding="utf-8")
    assert "vol_stop_ctx" in manager, "the reader path disappeared"
