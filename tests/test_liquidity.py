"""P6-A — volume-aware execution realism: participation caps and impact costs.

Claim under test
----------------
The project's volume features are *relative* (volume vs its own rolling mean) and
say nothing about **capacity or cost**.  These tests pin the two P6-A additions
that do, and — just as important — that **nothing changes while
``risk.liquidity.enabled`` is false and ``impact_k`` is 0** (the shipped
defaults):

1. ``core.risk.liquidity`` — :func:`recent_quote_volume`,
   :func:`participation_pct`, :func:`cap_notional` (shrink-only),
   :func:`impact_pct` (square-root law), and the measured per-call cost of a
   per-signal call.
2. ``PositionSizer`` — the participation cap, applied only when the switch is on,
   never growing the notional, with the hard caps still authoritative.
3. ``core.backtest.cost_model`` — the impact term, charged on entry *and* exit,
   with a bit-identical off-path (``k = 0`` / no volume).
4. Config — every ``risk.liquidity`` key has a reader in code (the audit's
   "unread key" defect), and the YAML block matches the model defaults.

Determinism and data policy
---------------------------
No randomness, no network, no writes to ``data/**``: the real-data assertions
only *read* ``data/market/*.parquet`` and are skipped when the checkout does not
ship them.

The cache is mid-migration from v1 (``open/high/low/close/volume``) to v2
(adding ``quote_volume``/``trade_count``), and ``core.risk.liquidity`` prefers
the stored ``quote_volume`` whenever the column carries a usable number, so a
real-data expectation must never pin the column set: it is **derived from the
frame the test just read**, by :func:`_quote_volume_values` — stored
``quote_volume`` when usable, otherwise the documented ``Σ volume × close``
proxy.  Both branches are additionally pinned on synthetic frames the tests
build, and the two cache states are proven end-to-end on temp copies written
under ``tmp_path`` (:func:`test_both_cache_versions_agree_on_a_temp_copy`).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BTC_1H = ROOT / "data/market/BTCUSDT/1h.parquet"
XRP_1M = ROOT / "data/market/XRPUSDT/1m.parquet"

#: Measured bound for one ``cap_notional`` call on a 20-bar window (see
#: ``test_helper_call_cost_is_bounded``).  Deliberately ~100x the measured
#: Python cost so it catches an accidental O(n²) / pandas-per-call regression
#: without failing on a slow CI box.
CALL_COST_BOUND_US = 200.0


def _read(path: Path):
    if not path.exists():
        pytest.skip(f"no cached {path.name} in this checkout")
    return pd.read_parquet(path)


def _frame(volumes, closes):
    """A minimal synthetic frame: the only columns the capacity helpers read."""
    return pd.DataFrame({"volume": volumes, "close": closes})


def _quote_volume_values(frame, lookback_bars: int = 20) -> list[float]:
    """The very values ``recent_quote_volume`` sums for ``frame`` (P6-B rule).

    Derived from the frame in hand, never from a pinned column set: the cache is
    being rewritten from v1 to v2 while this suite runs, and
    ``core/risk/liquidity._series`` takes whichever of the two the frame carries:

    * a **usable** ``quote_volume`` column (at least one finite value in the tail
      window) ⇒ those stored values win — the VWAP-like figure is not equal to
      ``volume × close`` (measured ~0.14 % apart on BTC 1h);
    * otherwise — the column is absent (v1) **or** present and entirely
      non-finite (a v2 file whose backfill has not run) — ⇒ the documented proxy
      ``volume × close`` over the same tail window.

    The tail window is taken **before** non-finite bars are dropped, exactly as
    the helper does (it tails the full column, then filters).  The "usable"
    test, however, is a property of the **column**, not of the window: the
    helper decides ``quote_volume`` vs proxy before it slices, so a column that
    is partly quoted but whose tail window happens to be entirely non-finite
    yields ``[]`` — ``0.0``, the documented "not measured" signal — and *not*
    the proxy.  Only pairwise finite ``volume``/``close`` bars enter the proxy.
    Returned as a list so the mapping / 1-D-sequence input forms can be handed
    the identical numbers.
    """
    if "quote_volume" in list(frame.columns):
        quoted = pd.to_numeric(frame["quote_volume"], errors="coerce")
        if quoted.notna().any():
            window = quoted.tail(lookback_bars) if lookback_bars else quoted
            return [float(v) for v in window.dropna()]
    tail = frame.tail(lookback_bars) if lookback_bars else frame
    volumes = pd.to_numeric(tail["volume"], errors="coerce")
    closes = pd.to_numeric(tail["close"], errors="coerce")
    return [float(v * c) for v, c in zip(volumes, closes)
            if math.isfinite(float(v)) and math.isfinite(float(c))]


def _expected_quote_volume(frame, lookback_bars: int = 20) -> float:
    """The documented expectation for ``frame`` under either cache version."""
    values = _quote_volume_values(frame, lookback_bars)
    return math.fsum(values) if values else 0.0


# ── 1. the helper module ────────────────────────────────────────────────

def test_recent_quote_volume_uses_tail_window_and_real_prices():
    """The window is the LAST ``lookback_bars`` bars — on either cache version.

    Old assertion (P6-B-invalid): ``expected = Σ volume × close`` over the tail,
    which only holds while the file is v1.  Once ``quote_volume`` exists the
    helper prefers the stored column (VWAP ≈ close, not equal), so the pin
    failed *mid-migration*, depending on which file the running backfill had
    reached.  New rule: the expectation is derived from the frame just read —
    ``Σ quote_volume`` when that column is usable, else the documented proxy —
    by :func:`_expected_quote_volume`.
    """
    from core.risk.liquidity import recent_quote_volume

    df = _read(BTC_1H)
    tail = df.tail(20)
    expected = _expected_quote_volume(df, 20)
    got = recent_quote_volume(df, 20)
    assert got == pytest.approx(expected, rel=1e-9)
    # A different window is a different number: the lookback really bounds it.
    assert recent_quote_volume(df, 20) != pytest.approx(
        recent_quote_volume(df, 40), rel=1e-9)
    # It is a quote (USDT) figure, not base-asset volume: per base unit traded it
    # sits at the window's own price level.  Derived from the frame (a 10 % band
    # absorbs the VWAP-vs-close gap and either cache version), not pinned to a
    # price: the old ``got > min(close) × Σ volume`` form was a near-miss tie on
    # the v2 column and is replaced by the ratio it was approximating.
    base = float(tail["volume"].sum())
    assert base > 0.0 and got > 0.0
    per_unit = got / base
    assert float(tail["close"].min()) * 0.9 < per_unit < \
        float(tail["close"].max()) * 1.1


def test_recent_quote_volume_input_forms_agree():
    """Frame, ``frame=``, mapping, provider and Series give the same figure."""
    from core.risk.liquidity import recent_quote_volume

    df = _read(BTC_1H).tail(20)
    by_frame = recent_quote_volume(df, 20)
    # Version-agnostic: the frame's own documented source, not a pinned column.
    assert by_frame == pytest.approx(_expected_quote_volume(df, 20), rel=1e-9)
    assert recent_quote_volume(frame=df, lookback_bars=20) == by_frame
    assert recent_quote_volume(df, lookback_bars=20) == by_frame
    # A mapping with only ``volume`` (+ close) is summed as BASE volume: the
    # documented "a plain volume series is not price-converted" path, so this
    # number does not move with the cache version.
    assert recent_quote_volume({"volume": list(df["volume"]),
                                "close": list(df["close"])}) == pytest.approx(
        float(df["volume"].sum()))  # mapping path: volume already priced
    # A mapping that already carries quote_volume needs no price column.  The
    # values are the frame's own documented source (stored quote_volume when
    # usable, else volume × close), so this agrees with ``by_frame`` on either
    # cache version — the old ``Σ volume × close`` construction did not.
    qv = _quote_volume_values(df, 20)
    assert recent_quote_volume({"quote_volume": qv}, 20) == pytest.approx(by_frame)
    # An injected provider wins over everything else.
    assert recent_quote_volume(df, 20, provider=lambda n: 1234.5) == 1234.5
    assert recent_quote_volume(None, 20, provider=lambda: 7.0) == 7.0
    # A plain 1-D volume series is summed as-is (documented: no price conversion).
    assert recent_quote_volume(list(df["volume"]), 20) == pytest.approx(
        float(df["volume"].sum()))


def test_recent_quote_volume_edges_are_zero_not_nan():
    """No data, an empty window or a bad provider argument → 0.0, never NaN."""
    from core.risk.liquidity import recent_quote_volume

    assert recent_quote_volume(None, 20) == 0.0
    assert recent_quote_volume([], 20) == 0.0
    assert recent_quote_volume(pd.DataFrame(), 20) == 0.0
    assert recent_quote_volume([1.0, 2.0], 0) == 3.0      # 0 = whole series
    assert recent_quote_volume([math.nan, 1.0], 20) == 1.0  # NaN bars dropped
    with pytest.raises(TypeError):
        recent_quote_volume(None, 20, provider=42)


def test_quote_volume_rule_is_pinned_on_synthetic_v1_and_v2_frames():
    """The P6-B source rule, pinned on frames the test builds — never the cache.

    The rule the live cache cannot be asked to prove, because it may be in
    either state: a usable stored ``quote_volume`` wins, otherwise the
    documented proxy ``Σ volume × close`` is used, both over the tail window.
    Building both shapes here keeps the fallback path covered even while every
    file on disk is v2 (and the v2 path covered while they are still v1).
    """
    from core.risk.liquidity import recent_quote_volume

    volumes = [1.0, 2.0, 3.0, 4.0, 5.0]
    closes = [10.0, 20.0, 30.0, 40.0, 50.0]
    v1 = _frame(volumes, closes)                        # no quote_volume column
    v2 = pd.DataFrame({"volume": volumes, "close": closes,
                       "quote_volume": [11.0, 21.0, 33.0, 44.0, 55.0]})
    # Tail window (the last 3 bars) and the whole series (lookback <= 0).
    assert recent_quote_volume(v1, 3) == pytest.approx(3 * 30.0 + 4 * 40.0 + 5 * 50.0)
    assert recent_quote_volume(v1, 0) == pytest.approx(10.0 + 40.0 + 90.0 + 160.0
                                                       + 250.0)
    assert recent_quote_volume(v2, 3) == pytest.approx(33.0 + 44.0 + 55.0)
    assert recent_quote_volume(v2, 0) == pytest.approx(164.0)
    # The stored column really wins: on this frame the proxy is a different
    # number, so a regression to "always price with closes" cannot pass.
    assert recent_quote_volume(v2, 3) != pytest.approx(3 * 30.0 + 4 * 40.0 + 5 * 50.0)
    # Both frames agree when the stored column equals the proxy, which is the
    # only case in which a comparison between them is meaningful.
    same = pd.DataFrame({"volume": volumes, "close": closes,
                         "quote_volume": [v * c for v, c in zip(volumes, closes)]})
    assert recent_quote_volume(same, 20) == pytest.approx(recent_quote_volume(v1, 20))


def test_non_finite_quote_volume_column_falls_back_to_the_proxy():
    """A v2-shaped file whose backfill has not run → the documented proxy.

    ``core/risk/liquidity`` documents both shapes: a ``quote_volume`` column
    that is *present but entirely non-finite* is not a measurement, so the
    proxy is used instead of returning 0.0 ("unknown volume", which would refuse
    orders the proxy can size honestly); a window with only *some* bars quoted
    sums the quoted ones and drops the holes rather than filling them.
    """
    from core.risk.liquidity import recent_quote_volume

    volumes = [1.0, 2.0, 3.0, 4.0]
    closes = [10.0, 20.0, 30.0, 40.0]
    proxy = 10.0 + 40.0 + 90.0 + 160.0                 # Σ volume × close = 300
    all_nan = pd.DataFrame({"volume": volumes, "close": closes,
                            "quote_volume": [float("nan")] * 4})
    assert recent_quote_volume(all_nan, 4) == pytest.approx(proxy)
    # Partially quoted: the two finite bars only — a hole is "fewer bars", not a
    # proxy-priced bar.
    partial = pd.DataFrame({"volume": volumes, "close": closes,
                            "quote_volume": [float("nan"), 5.0,
                                             float("nan"), 7.0]})
    assert recent_quote_volume(partial, 4) == pytest.approx(12.0)
    assert recent_quote_volume(partial, 4) != pytest.approx(proxy)
    # "Entirely non-finite" is decided on the whole COLUMN, before the window:
    # a partly quoted column whose tail window holds no finite bar is 0.0 ("not
    # measured" — the caller refuses rather than filling the hole with a proxy
    # that would look like data).
    late_hole = pd.DataFrame({"volume": volumes, "close": closes,
                              "quote_volume": [8.0, 9.0,
                                               float("nan"), float("nan")]})
    assert recent_quote_volume(late_hole, 2) == 0.0
    assert recent_quote_volume(late_hole, 2) != pytest.approx(3 * 30.0 + 4 * 40.0)
    assert recent_quote_volume(late_hole, 2) == _expected_quote_volume(late_hole, 2)


def test_participation_pct_units_and_undefined():
    """``participation_pct`` is a PERCENT of the window; ``None`` when unmeasurable."""
    from core.risk.liquidity import participation_pct

    assert participation_pct(1_000.0, 100_000.0) == pytest.approx(1.0)
    assert participation_pct(500.0, 750_661_812.73) == pytest.approx(6.66e-5, rel=0.01)
    # A short is the same liquidity event as a long of the same size.
    assert participation_pct(-1_000.0, 100_000.0) == pytest.approx(1.0)
    # Undefined, not zero: unknown / empty window, NaN or a bad notional.
    assert participation_pct(1_000.0, 0.0) is None
    assert participation_pct(1_000.0, None) is None
    assert participation_pct(float("nan"), 100_000.0) is None
    assert participation_pct(None, 100_000.0) is None


def test_cap_notional_is_shrink_only_and_reports_why():
    """Exactly the documented reasons; the cap can never *grow* a notional."""
    from core.risk.liquidity import cap_notional

    # Fits: the value is returned bit-identically (no float perturbation).
    allowed, reason = cap_notional(500.0, 750_661_812.73, 1.0)
    assert allowed == 500.0 and reason.startswith("ok:")
    # 50 000 is 1.28 % of a 3.9 M window → shrunk to exactly 1 %.
    allowed, reason = cap_notional(50_000.0, 3_898_610.87, 1.0)
    assert allowed == pytest.approx(38_986.1087)
    assert reason.startswith("capped:") and "1.2825" in reason
    assert allowed < 50_000.0
    # The cap is an upper bound, never a floor.
    assert cap_notional(10.0, 3_898_610.87, 1.0)[0] == 10.0
    # Unknown window → refused (0.0) with a reason, never silently passed.
    allowed, reason = cap_notional(10_000.0, 0.0, 1.0)
    assert allowed == 0.0 and reason.startswith("no_volume:")
    assert cap_notional(10_000.0, None, 1.0)[0] == 0.0
    # ceiling <= 0 is "no policy" → pass-through.
    assert cap_notional(10_000.0, 100_000.0, 0.0)[0] == 10_000.0
    assert cap_notional(10_000.0, 100_000.0, 0.0)[1].startswith("disabled:")
    # Non-positive notional has nothing to cap.
    assert cap_notional(0.0, 100_000.0, 1.0)[0] == 0.0
    assert cap_notional(-5.0, 100_000.0, 1.0)[1].startswith("ok:")


def test_impact_pct_is_the_documented_square_root_law():
    """``impact_pct = k · participation**e`` in PERCENT of notional, per side."""
    from core.risk.liquidity import impact_pct

    # sqrt law: 4x the participation = 2x the impact.
    assert impact_pct(0.01, 0.5) == pytest.approx(0.05)
    assert impact_pct(0.04, 0.5) == pytest.approx(0.10)
    assert impact_pct(0.04, 0.5) / impact_pct(0.01, 0.5) == pytest.approx(2.0)
    # Off paths: k <= 0 is the pre-P6 cost model; unknown/0 participation is too.
    assert impact_pct(0.5, 0.0) == 0.0
    assert impact_pct(0.5, -1.0) == 0.0
    assert impact_pct(None, 0.5) == 0.0
    assert impact_pct(0.0, 0.5) == 0.0
    assert impact_pct(float("nan"), 0.5) == 0.0
    # floor / cap clamp, and a cap below the floor degrades to the floor.
    assert impact_pct(0.0, 0.5, floor=0.02) == pytest.approx(0.02)
    assert impact_pct(0.9, 0.5, cap=0.1) == pytest.approx(0.1)
    assert impact_pct(0.9, 0.5, floor=0.3, cap=0.1) == pytest.approx(0.3)
    # Vectorisable: a list in, a list out (a per-signal batch call).
    assert impact_pct([0.01, 0.04], 0.5) == [pytest.approx(0.05),
                                              pytest.approx(0.10)]


def test_trade_impact_helpers_are_round_trip():
    """The reporting helpers charge both sides and follow the same off-switch."""
    from core.risk.liquidity import total_impact_usdt, trade_impact_pct

    # 1 000 USDT round trip against a 100 000 USDT window, k = 0.5:
    # participation = 1 % → impact_pct = 0.5·0.01**0.5 = 0.05 % per side.
    per_side = 1_000.0 * 0.0005
    assert total_impact_usdt(1_000.0, 1_000.0, 100_000.0, 0.5) == \
        pytest.approx(2 * per_side)
    assert trade_impact_pct(1_000.0, 1_000.0, 100_000.0, 0.5) == \
        pytest.approx(0.1)  # percent of entry notional
    # Off switch: k <= 0, unknown volume, or no notional.
    assert total_impact_usdt(1_000.0, 1_000.0, 100_000.0, 0.0) == 0.0
    assert total_impact_usdt(1_000.0, 1_000.0, 0.0, 0.5) == 0.0
    assert total_impact_usdt(0.0, 0.0, 100_000.0, 0.5) == 0.0


def test_helper_call_cost_is_bounded():
    """A per-signal call costs microseconds — the measured bound is asserted.

    The helpers sit on the sizing hot path, so the cost of one ``cap_notional``
    call on a 20-bar window (the documented ``lookback_bars``) is measured and
    bounded.  ``CALL_COST_BOUND_US`` is ~100x the measured value so the test
    fails on a complexity regression, not on a slow machine.
    """
    from core.risk.liquidity import cap_notional, impact_pct, participation_pct

    volume = 3_898_610.87
    notional = 50_000.0
    n = 20_000
    start = time.perf_counter()
    for _ in range(n):
        cap_notional(notional, volume, 1.0)
    per_call_us = (time.perf_counter() - start) / n * 1e6
    # The volume itself (a 20-bar frame slice priced with closes) is measured too.
    df = _read(BTC_1H)
    tail = df.tail(20)
    volumes = [float(v) for v in tail["volume"]]
    closes = [float(c) for c in tail["close"]]
    start = time.perf_counter()
    for _ in range(n):
        v = math.fsum(a * b for a, b in zip(volumes, closes))
        p = participation_pct(v, v)
        impact_pct(p / 100.0, 0.5)
    window_us = (time.perf_counter() - start) / n * 1e6

    print(f"\ncap_notional: {per_call_us:.3f} us/call (bound {CALL_COST_BOUND_US})"
          f"\n20-bar frame + participation + impact: {window_us:.3f} us/call")
    assert per_call_us < CALL_COST_BOUND_US
    assert window_us < CALL_COST_BOUND_US


def test_real_data_capacity_table():
    """The cap rule on the shipped cache — derived, branch-complete, the doc's §3.

    Every expectation is **derived from the same frame the test just read**, so a
    cache repair or a new bar cannot fail the test while the code is correct (the
    policy in ``tests/test_measured_threshold_policy.py``).  BTC 1h is deep: even
    50 000 USDT is far under a 1 % ceiling.  XRP 1m is the capacity-limited case
    *when the window is shallow enough* — that premise is a mutable fact of the
    live file, so it is not pinned: the measured window picks the branch and the
    binding branch is asserted from the very numbers in hand
    (``cap_notional`` returns ``notional`` iff ``notional <= window * pct/100``).
    Both branches are pinned on synthetic frames with a **fixed** window in
    :func:`test_cap_rule_holds_for_a_shallow_and_a_deep_window`.  The doc quotes
    one measured snapshot together with the command that printed it.
    """
    from core.risk.liquidity import (cap_notional, impact_pct,
                                     participation_pct, recent_quote_volume)

    btc = _read(BTC_1H)
    btc_vol = recent_quote_volume(btc, 20)
    # Derived from the frame just read under either cache version: the stored
    # P6-B ``quote_volume`` when usable, else the documented Σ volume × close
    # proxy (:func:`_expected_quote_volume`).  This line used to pin the proxy
    # and so failed the moment the migration reached BTC 1h.
    btc_derived = _expected_quote_volume(btc, 20)
    assert btc_vol == pytest.approx(btc_derived, rel=1e-9)
    assert btc_vol > 0.0
    # 500 and 50 000 USDT are both a tiny fraction of a single BTC 1h day.
    small = participation_pct(500.0, btc_vol)
    large = participation_pct(50_000.0, btc_vol)
    assert 0.0 < small < large < 1.0
    assert large / small == pytest.approx(100.0)      # 100x the order, 100x
    assert cap_notional(50_000.0, btc_vol, 1.0)[0] == 50_000.0  # not binding
    # Impact follows the square-root law on the measured participation.
    k = 0.1
    assert impact_pct(large / 100.0, k) == pytest.approx(k * (large / 100.0) ** 0.5)

    xrp = _read(XRP_1M)
    xrp_vol = recent_quote_volume(xrp, 20)
    assert xrp_vol > 0.0
    allowed_500, reason_500 = cap_notional(500.0, xrp_vol, 1.0)
    allowed_50k, reason_50k = cap_notional(50_000.0, xrp_vol, 1.0)
    if participation_pct(500.0, xrp_vol) <= 1.0:
        assert allowed_500 == 500.0 and reason_500.startswith("ok:")
    # Which branch 50 000 USDT falls in is a property of *this* window, not of
    # the code: a live app rewrite can flip it (measured 1.88 M → 6.30 M → 2.80 M
    # USDT across one audit), so the assertion is the rule evaluated on the
    # window just read.  Both branches are exercised deterministically below.
    ceiling = xrp_vol * 1.0 / 100.0
    if 50_000.0 > ceiling:
        # Capped: exactly the ceiling, and participation lands ON the cap.
        assert allowed_50k == pytest.approx(ceiling)
        assert allowed_50k < 50_000.0 and reason_50k.startswith("capped:")
        assert participation_pct(allowed_50k, xrp_vol) == pytest.approx(1.0, rel=1e-9)
    else:
        # Deep enough to absorb it: passed through bit-identically.
        assert allowed_50k == 50_000.0 and reason_50k.startswith("ok:")
        assert participation_pct(allowed_50k, xrp_vol) <= 1.0


def test_both_cache_versions_agree_on_a_temp_copy(tmp_path):
    """Proof on disk: a v1 copy and a v2 copy, each read back and asserted.

    The live cache is rewritten from v1 to v2 *while this suite runs*, so the
    test builds both states itself under ``tmp_path`` (it writes nothing under
    ``data/**``): the frame it just read is written once with the P6-B columns
    dropped (v1), and once carrying a **synthetic** ``quote_volume`` column — the
    proxy × 1.005, a VWAP-like gap that is small but *provable*, so the two
    copies are certainly different figures and "the stored column wins" cannot
    pass by coincidence.  (The real stored column is exercised by the three
    live-data tests above; this test is about the rule, not the measurement.)
    Each copy is read back from disk and compared with the same derived
    expectation, which is the end-to-end proof that one rule covers both versions.
    """
    from core.risk.liquidity import recent_quote_volume

    df = _read(BTC_1H)
    proxy = pd.to_numeric(df["volume"], errors="coerce") * \
        pd.to_numeric(df["close"], errors="coerce")
    v1 = df.drop(columns=[c for c in ("quote_volume", "trade_count")
                          if c in df.columns])
    v2 = v1.copy()
    v2["quote_volume"] = proxy * 1.005
    assert "quote_volume" not in v1.columns and "quote_volume" in v2.columns

    parsed = {}
    for name, frame in (("v1", v1), ("v2", v2)):
        path = tmp_path / f"BTCUSDT_1h_{name}.parquet"
        frame.to_parquet(path)
        parsed[name] = pd.read_parquet(path)
    assert "quote_volume" not in parsed["v1"].columns
    assert "quote_volume" in parsed["v2"].columns
    assert "trade_count" not in parsed["v1"].columns
    assert len(parsed["v1"]) == len(parsed["v2"]) == len(df)

    # Same rule, both files: the derived expectation is what the helper returns.
    for name, frame in parsed.items():
        got = recent_quote_volume(frame, 20)
        assert got == pytest.approx(_expected_quote_volume(frame, 20),
                                    rel=1e-9), name
        # And the rule picked the documented branch for this version.
        if name == "v2":
            stored = float(pd.to_numeric(frame["quote_volume"], errors="coerce")
                           .tail(20).dropna().sum())
            assert got == pytest.approx(stored, rel=1e-9)
        else:
            derived_proxy = float((frame["volume"].tail(20)
                                   * frame["close"].tail(20)).sum())
            assert got == pytest.approx(derived_proxy, rel=1e-9)

    # The two cache states are *provably* different figures (the v2 copy holds
    # 1.005 × the v1 proxy), so the agreement above is not the trivial case of
    # both versions carrying the same number.
    v1_vol = recent_quote_volume(parsed["v1"], 20)
    v2_vol = recent_quote_volume(parsed["v2"], 20)
    assert v1_vol != pytest.approx(v2_vol, rel=1e-6)
    assert v2_vol == pytest.approx(v1_vol * 1.005, rel=1e-9)


def test_cap_rule_holds_for_a_shallow_and_a_deep_window():
    """The 50 000 USDT input, pinned on **two fixed synthetic windows**.

    The branch a live window falls in moves whenever the running app rewrites
    ``data/market/**`` (the XRP 1m 20-bar window measured 1 878 291 → 6 298 710.92
    → 2 801 364.26 USDT inside one audit), so both branches are pinned here on
    frames the test builds: a 1 000 USDT window where the order is 50× the 1 %
    ceiling, and a 10 000 000 USDT window where it is 0.5 % of it.  The
    expectation is the code's own rule — ``cap_notional`` returns the notional
    unchanged iff ``notional <= window * pct/100`` — so this is a unit test of the
    rule, not a measurement of a cache.
    """
    from core.risk.liquidity import cap_notional, participation_pct

    shallow = _frame([1.0] * 20, [50.0] * 20)      # window = 1 000 USDT
    deep = _frame([10.0] * 20, [500_000.0] * 20)   # window = 10 000 000 USDT
    for frame, capped in ((shallow, True), (deep, False)):
        window = float((frame["close"].tail(20) * frame["volume"].tail(20)).sum())
        allowed, reason = cap_notional(50_000.0, window, 1.0)
        if capped:
            assert 50_000.0 > window / 100.0, (window, reason)
            assert allowed == pytest.approx(window / 100.0)
            assert allowed < 50_000.0 and reason.startswith("capped:")
            assert participation_pct(allowed, window) == pytest.approx(1.0, rel=1e-9)
        else:
            assert 50_000.0 <= window / 100.0, (window, reason)
            assert allowed == 50_000.0 and reason.startswith("ok:")


# ── 2. PositionSizer: opt-in, shrink-only, hard caps still authoritative ──

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


def _sizer(liquidity=None, vol_targeting=None):
    from app.config import HardRiskLimits, SoftRiskParams
    from core.risk.position_sizer import PositionSizer

    return PositionSizer(HardRiskLimits(), SoftRiskParams(), 0.7, 0.3,
                         vol_targeting, liquidity)


def test_participation_disabled_is_bit_identical():
    """Off-path: same numbers with and without a volume argument (shipped default)."""
    off = _sizer(_Liquidity(enabled=False))
    reference = _sizer(None)
    args = (10_000.0, 50_000.0, "satellite")
    base = reference.calculate_position_size(*args)
    # The hook is not even called while the switch is off.
    assert off.participation_cap_enabled("BTCUSDT") is False
    assert off.apply_participation_cap(240.0, 1_000_000.0, "BTCUSDT") == \
        (240.0, "disabled: risk.liquidity.enabled is false")
    for volume in (None, 0.0, 123.0, 750_661_812.73, [1.0, 2.0],
                   lambda n: 5.0):
        assert off.calculate_position_size(*args, recent_quote_volume=volume) == base
    assert off.calculate_position_size(*args) == base


def test_participation_cap_shrinks_and_never_grows():
    """On-path: the cap shrinks a big order, passes a small one, never grows."""
    on = _sizer(_Liquidity(enabled=True, max_participation_pct=1.0))
    # 10 000 balance × 30 % satellite × 5 % = 150 USDT of notional.
    _, base_risk = _sizer(None).calculate_position_size(10_000.0, 50_000.0)
    assert base_risk == pytest.approx(150.0)
    # A 1 000 USDT window admits only 10 USDT (1 %) → the order is shrunk.
    qty, risk = on.calculate_position_size(10_000.0, 50_000.0,
                                          recent_quote_volume=1_000.0)
    assert risk == pytest.approx(10.0)
    assert qty == pytest.approx(10.0 / 50_000.0)
    # A window big enough → unchanged to the last bit.
    assert on.calculate_position_size(10_000.0, 50_000.0,
                                      recent_quote_volume=1e9)[1] == base_risk
    # A frame works as the volume source (core.risk.liquidity duck-typing).
    df = pd.DataFrame({"volume": [10.0] * 20, "close": [1.0] * 20})
    assert on.calculate_position_size(10_000.0, 50_000.0,
                                      recent_quote_volume=df)[1] == \
        pytest.approx(2.0)  # 200 USDT window × 1 %
    # A caller cannot talk the cap into GROWING a notional: a huge ceiling is
    # still bounded by the hard caps above it.
    generous = _sizer(_Liquidity(enabled=True, max_participation_pct=100.0))
    assert generous.calculate_position_size(10_000.0, 50_000.0,
                                            recent_quote_volume=1e9)[1] == base_risk
    # And the hook itself refuses to grow even when handed a bigger "allowed".
    assert generous.apply_participation_cap(150.0, 1e9, "BTCUSDT")[0] == 150.0


def test_participation_cap_respects_the_hard_ceiling():
    """The participation cap composes with — never overrides — the hard caps."""
    on = _sizer(_Liquidity(enabled=True, max_participation_pct=50.0))
    # core pool 70 % × 5 % = 350 USDT, but max_position_size_pct 10 % = 1 000.
    _, risk = on.calculate_position_size(10_000.0, 50_000.0, "core",
                                         recent_quote_volume=1e9)
    assert risk == pytest.approx(350.0)
    # A 50 % ceiling on a 100 USDT window is 50 USDT — below the hard caps.
    assert on.calculate_position_size(10_000.0, 50_000.0, "core",
                                      recent_quote_volume=100.0)[1] == \
        pytest.approx(50.0)


def test_participation_hook_accepts_a_provider_and_unmeasured_window():
    """A callable provider is used; an unmeasured window refuses the order."""
    on = _sizer(_Liquidity(enabled=True, max_participation_pct=1.0))
    _, risk = on.calculate_position_size(10_000.0, 50_000.0,
                                         recent_quote_volume=lambda n: 1_000.0)
    assert risk == pytest.approx(10.0)
    # Unknown window → 0.0 (a ceiling with no denominator is not a ceiling).
    assert on.calculate_position_size(10_000.0, 50_000.0,
                                      recent_quote_volume=0.0)[1] == 0.0
    assert on.calculate_position_size(10_000.0, 50_000.0,
                                      recent_quote_volume=[0.0, 0.0])[1] == 0.0


def test_per_symbol_override_changes_only_that_symbol():
    """``per_symbol`` partial overrides merge over the top-level block."""
    block = _Liquidity(enabled=True, max_participation_pct=1.0,
                       per_symbol={"BTCUSDT": {"max_participation_pct": 10.0},
                                   "default": {"lookback_bars": 5}})
    sizer = _sizer(block)
    # Same 1 000 USDT window: BTC gets 10 % (100 USDT), everyone else 1 % (10).
    assert sizer.calculate_position_size(10_000.0, 50_000.0, "satellite",
                                         recent_quote_volume=1_000.0,
                                         symbol="BTCUSDT")[1] == pytest.approx(100.0)
    assert sizer.calculate_position_size(10_000.0, 50_000.0, "satellite",
                                         recent_quote_volume=1_000.0,
                                         symbol="ETHUSDT")[1] == pytest.approx(10.0)
    # The `default` key supplies lookback_bars for the unlisted symbol.
    assert sizer.liquidity_config("ETHUSDT").lookback_bars == 5
    assert sizer.liquidity_config("BTCUSDT").lookback_bars == 20


def test_sizer_finds_the_liquidity_block_through_a_config():
    """Production plumbing: the sizer reads ``risk.liquidity`` off the config."""
    from app.config import Config

    cfg = Config.load()
    sizer = _sizer(None, cfg.risk_vol_targeting)
    assert sizer.liquidity_config("BTCUSDT") is cfg.risk_liquidity
    assert sizer.participation_cap_enabled("BTCUSDT") is cfg.risk_liquidity.enabled


# ── 3. cost model: impact term, off-path bit-identical ──────────────────

class _CostConfig:
    backtest_cost_enabled = True
    backtest_taker_fee_pct = 0.04
    backtest_spread_pct = {"BTCUSDT": 0.01}
    backtest_default_spread_pct = 0.03
    backtest_live_spread_enabled = False

    def __init__(self, impact_k=0.0):
        self.risk_liquidity = _Liquidity(enabled=False, impact_k=impact_k)


def test_impact_k_zero_is_bit_identical():
    """``impact_k = 0`` (and no volume) → byte-for-byte the pre-P6 cost."""
    from core.backtest.cost_model import apply_trading_costs

    cfg = _CostConfig(impact_k=0.0)
    legacy = apply_trading_costs(50_000.0, 51_000.0, 0.01, "BTCUSDT", cfg)
    for volume in (None, 0.0, 1_000.0, 750_661_812.73):
        got = apply_trading_costs(50_000.0, 51_000.0, 0.01, "BTCUSDT", cfg,
                                  recent_quote_volume=volume)
        assert got == legacy and repr(got) == repr(legacy)
    # A config double without the block at all behaves the same way.
    class Bare:
        backtest_cost_enabled = True
        backtest_taker_fee_pct = 0.04
        backtest_spread_pct = {"BTCUSDT": 0.01}

    assert apply_trading_costs(50_000.0, 51_000.0, 0.01, "BTCUSDT", Bare,
                               recent_quote_volume=1e9) == \
        apply_trading_costs(50_000.0, 51_000.0, 0.01, "BTCUSDT", Bare)


def test_impact_term_is_charged_on_entry_and_exit():
    """With ``k > 0`` the difference is exactly the helper's round-trip charge."""
    from core.backtest.cost_model import apply_trading_costs, impact_cost_usdt
    from core.risk.liquidity import total_impact_usdt

    cfg = _CostConfig(impact_k=0.5)
    entry, exit_, qty, volume = 50_000.0, 51_000.0, 0.01, 100_000.0
    legacy = apply_trading_costs(entry, exit_, qty, "BTCUSDT", cfg)
    with_impact = apply_trading_costs(entry, exit_, qty, "BTCUSDT", cfg,
                                      recent_quote_volume=volume)
    expected = total_impact_usdt(entry * qty, exit_ * qty, volume, 0.5)
    assert with_impact - legacy == pytest.approx(expected)
    assert with_impact > legacy
    # Each side carries its OWN participation: 500 USDT of a 100 000 window is
    # 0.5 % (impact_pct = 0.5·0.005**0.5 = 0.03536 %) and the 510 USDT exit side
    # is 0.51 % (0.03570 %), so the exit side costs 1.02x the entry side.
    side = 0.5 * 0.005 ** 0.5 / 100.0
    assert expected == pytest.approx(entry * qty * side
                                     + exit_ * qty * (0.5 * 0.0051 ** 0.5 / 100.0))
    assert expected / (entry * qty) > side  # the bigger exit side costs more
    assert impact_cost_usdt(entry * qty, exit_ * qty, volume, 0.5) == \
        pytest.approx(expected)
    assert impact_cost_usdt(entry * qty, exit_ * qty, volume, 0.0) == 0.0
    # A larger order pays proportionally more (the sqrt law is monotone).
    bigger = apply_trading_costs(entry, exit_, qty * 10, "BTCUSDT", cfg,
                                 recent_quote_volume=volume)
    assert bigger - legacy > 1.0 * (with_impact - legacy)
    # A pinned round-trip impact overrides the volume path.
    pinned = apply_trading_costs(entry, exit_, qty, "BTCUSDT", cfg,
                                 impact_pct_override=4.0)
    assert pinned == pytest.approx(legacy + entry * qty * 0.04)


def test_cost_report_splits_fees_spread_and_impact():
    """The reported cost breakdown adds up, and impact is visible as its own line."""
    from core.backtest.cost_model import total_costs_with_impact

    cfg = _CostConfig(impact_k=0.5)
    out = total_costs_with_impact(50_000.0, 51_000.0, 0.01, "BTCUSDT", cfg,
                                  recent_quote_volume=100_000.0)
    assert out["impact_usdt"] > 0.0
    assert out["fees_usdt"] + out["spread_usdt"] == pytest.approx(out["legacy_usdt"])
    assert out["legacy_usdt"] + out["impact_usdt"] == pytest.approx(out["total_usdt"])
    assert out["impact_k"] == 0.5 and out["recent_quote_volume"] == 100_000.0
    # With the term off, the impact line is exactly zero and total == legacy.
    off = total_costs_with_impact(50_000.0, 51_000.0, 0.01, "BTCUSDT",
                                  _CostConfig(impact_k=0.0),
                                  recent_quote_volume=100_000.0)
    assert off["impact_usdt"] == 0.0
    assert off["total_usdt"] == off["legacy_usdt"]


def test_cost_model_real_run_comparison_is_bit_identical_at_k_zero():
    """Pre/post on the SAME real bars: k = 0 reproduces the legacy cost exactly.

    A tiny deterministic run over the cached BTC 1h window: every trade's cost is
    computed with the shipped defaults (``impact_k`` 0) and with the term
    explicitly disabled, and the two totals must be identical — the required
    "compare pre/post on the same run" check.

    The ``k = 0.5`` half is bounded by a quantity **derived from the window this
    test actually read** (``max(entry, exit) notional / legacy_total`` is exactly
    the legacy cost per unit of notional, so multiplying it by ``impact_pct`` for
    this window gives the share the impact term can add).  It used to assert
    ``share < 0.05`` labelled "BTC 1h is deep", which silently required the live
    20-bar window to stay above ~4.0e7 USDT (it passes at 4.294e7 and fails at
    3.435e7): a live-data premise, not a property of the code.  The invariants
    that survive any window are ``impacted > legacy`` and the impact identity,
    pinned on a synthetic window by
    :func:`test_impact_share_is_bounded_by_the_window_it_was_measured_on`.
    """
    from core.backtest.cost_model import apply_trading_costs
    from core.risk.liquidity import impact_pct, participation_pct, \
        recent_quote_volume

    bars = _read(BTC_1H).tail(200)
    volume = recent_quote_volume(bars, 20)
    shipped = _CostConfig(impact_k=0.0)
    post = _CostConfig(impact_k=0.5)
    if volume <= 0.0:  # pragma: no cover - a repaired/empty cache
        pytest.skip("cached BTC 1h window has no quote volume")
    legacy_total = impacted_total = 0.0
    qty = 0.01
    max_notional = 0.0
    for i in range(1, 100):
        entry = float(bars["close"].iloc[i - 1])
        exit_ = float(bars["close"].iloc[i])
        max_notional = max(max_notional, entry * qty, exit_ * qty)
        legacy_total += apply_trading_costs(entry, exit_, qty, "BTCUSDT", shipped,
                                           recent_quote_volume=volume)
        impacted_total += apply_trading_costs(entry, exit_, qty, "BTCUSDT", post,
                                              recent_quote_volume=volume)
    # Off path: identical to the last bit (no volume arg at all).
    no_volume_total = 0.0
    for i in range(1, 100):
        entry = float(bars["close"].iloc[i - 1])
        exit_ = float(bars["close"].iloc[i])
        no_volume_total += apply_trading_costs(entry, exit_, 0.01, "BTCUSDT",
                                               shipped)
    assert legacy_total == no_volume_total
    # On path: k = 0.5 costs strictly more, and the impact share stays inside the
    # bound this window implies.
    assert impacted_total > legacy_total
    share = (impacted_total - legacy_total) / legacy_total
    # ``side_pct`` is the per-side impact cost already expressed in percent of
    # notional, so ``bound`` multiplies by the notional and divides by the same
    # window's legacy cost -- it is a fraction of that cost, like ``share``.
    side_pct = impact_pct(participation_pct(max_notional, volume) / 100.0, 0.5)
    bound = 2.0 * side_pct * max_notional / legacy_total
    assert share < bound, (share, bound, volume, max_notional)


def test_impact_share_is_bounded_by_the_window_it_was_measured_on():
    """The R4 synthetic branch: the derived bound is exact for *any* window.

    The live test above may only assert what holds whatever ``data/market``
    contains.  This one builds the window instead, so the whole chain is pinned
    deterministically: participation -> ``impact_pct`` -> the round-trip charge
    -> its share of the legacy cost.  A window four times deeper gives a share
    exactly half as large (the square-root law), which is the property the old
    ``share < 0.05  # BTC 1h is deep`` literal was standing in for.
    """
    from core.backtest.cost_model import total_costs_with_impact
    from core.risk.liquidity import impact_pct, participation_pct

    cfg = _CostConfig(impact_k=0.5)
    entry, exit_, qty = 50_000.0, 51_000.0, 0.01
    shallow, deep = 100_000.0, 400_000.0
    shallow_out = total_costs_with_impact(entry, exit_, qty, "BTCUSDT", cfg,
                                          recent_quote_volume=shallow)
    deep_out = total_costs_with_impact(entry, exit_, qty, "BTCUSDT", cfg,
                                       recent_quote_volume=deep)
    shallow_legacy, deep_legacy = (shallow_out["legacy_usdt"],
                                   deep_out["legacy_usdt"])
    # Identical books, different windows: the legacy cost does not move.
    assert shallow_legacy == deep_legacy
    # The bound for this window: ``2 x side_pct`` is the round-trip impact
    # percentage (entry + exit), and scaling it by the largest notional in the
    # sample gives the impact charge in the legacy cost's own units.  Nothing
    # here assumes the window is deep -- a shallow one just raises the bound.
    max_notional = exit_ * qty
    shallow_side = impact_pct(participation_pct(max_notional, shallow) / 100.0, 0.5)
    shallow_bound_pct = 2.0 * shallow_side * max_notional / shallow_legacy
    shallow_share = shallow_out["impact_usdt"] / shallow_legacy
    assert shallow_share < shallow_bound_pct / 100.0
    # The bound is tight (the entry side is 1/51 smaller than the max notional)
    # and its 2 x side_pct core is exactly the charge the cost model added.
    entry_side = impact_pct(participation_pct(entry * qty, shallow) / 100.0, 0.5)
    exit_side = impact_pct(participation_pct(exit_ * qty, shallow) / 100.0, 0.5)
    round_trip = entry_side / 100.0 * entry * qty + exit_side / 100.0 * exit_ * qty
    assert shallow_out["impact_usdt"] == pytest.approx(round_trip)
    assert shallow_share < shallow_bound_pct / 100.0 < shallow_share / 0.8
    assert shallow_share < shallow_bound_pct / 100.0 < shallow_share * 1.05
    # Four times the depth, half the share: the derived bound moves with the
    # window instead of being a constant that quietly requires a deep cache.
    deep_share = deep_out["impact_usdt"] / deep_legacy
    assert deep_share == pytest.approx(shallow_share / 2.0)
    assert deep_share < shallow_share


# ── 4. config: every key has a reader, defaults are off ─────────────────

def test_shipped_defaults_are_off():
    """The switch names and their shipped values — the "do not enable by default"."""
    from app.config import Config, LiquidityConfig

    cfg = Config.load()
    assert cfg.risk_liquidity.enabled is False
    assert cfg.risk_liquidity.impact_k == 0.0
    assert cfg.risk_liquidity.max_participation_pct == pytest.approx(1.0)
    assert cfg.risk_liquidity.lookback_bars == 20
    assert cfg.risk_liquidity.impact_exponent == pytest.approx(0.5)
    assert cfg.risk_liquidity.per_symbol == {}
    # The model defaults are the same objects the helpers would use with no config.
    fresh = LiquidityConfig()
    assert (fresh.enabled, fresh.impact_k) == (False, 0.0)
    assert cfg.liquidity is cfg.risk_liquidity
    # A config double that lacks the attribute is impact-off / cap-off.
    sizer = _sizer(None, cfg.risk_vol_targeting)
    assert sizer.participation_cap_enabled("BTCUSDT") is False


def test_every_liquidity_config_key_has_a_reader():
    """Audit guard: no ``risk.liquidity`` key may exist without a code reader.

    Two halves, both mechanical: every model field must appear in
    ``app.config.LIQUIDITY_KEY_READERS``, and the file that table names must
    really mention the key (so the table cannot rot into a comment).
    """
    from app.config import LIQUIDITY_KEY_READERS, LiquidityConfig

    fields = set(LiquidityConfig.model_fields)
    assert fields == set(LIQUIDITY_KEY_READERS), (
        "risk.liquidity keys and their reader table disagree — add the reader "
        "first, then the key")
    for key, relative in LIQUIDITY_KEY_READERS.items():
        path = ROOT / relative
        assert path.exists(), f"{relative} (reader of {key}) does not exist"
        assert key in path.read_text(encoding="utf-8"), (
            f"{relative} claims to read risk.liquidity.{key} but never names it")


def test_yaml_block_matches_the_model_defaults():
    """The shipped YAML documents the same numbers as the model defaults."""
    import yaml

    from app.config import LiquidityConfig

    raw = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    block = raw["risk"]["liquidity"]
    model = LiquidityConfig()
    for key, value in block.items():
        assert key in LiquidityConfig.model_fields, f"unknown key risk.liquidity.{key}"
        assert value == getattr(model, key), (
            f"risk.liquidity.{key} in YAML is {value!r} but the model default is "
            f"{getattr(model, key)!r}")
    # The switch names the operator needs are all present.
    assert {"enabled", "max_participation_pct", "lookback_bars", "impact_k",
            "impact_exponent", "per_symbol"} <= set(block)
    # And the block is inert as shipped.
    assert block["enabled"] is False and block["impact_k"] == 0.0


def test_nonsense_liquidity_config_is_clamped_and_named():
    """A bad value is clamped *and* reported — never silently ignored."""
    from app.config import LiquidityConfig, liquidity_key_warnings

    assert liquidity_key_warnings(LiquidityConfig()) == []
    bad = LiquidityConfig(max_participation_pct=-1.0, lookback_bars=0,
                          impact_k=-2.0, impact_exponent=0.0,
                          per_symbol={"BTCUSDT": {"nope": 1}, "X": 5})
    messages = liquidity_key_warnings(bad)
    joined = " ".join(messages)
    assert "max_participation_pct" in joined and "DISABLED" in joined
    assert "lookback_bars" in joined
    assert "impact_k" in joined and "impact_exponent" in joined
    assert "per_symbol.BTCUSDT" in joined and "per_symbol.X" in joined
    assert bad.impact_k == 0.0 and bad.impact_exponent == 0.5
    assert bad.lookback_bars == 1 and bad.enabled is False
    assert bad.per_symbol == {"BTCUSDT": {}}  # unknown key dropped, entry kept


def test_liquidity_for_symbol_merges_partial_overrides():
    """A partial override changes one number and inherits the rest."""
    from app.config import LiquidityConfig
    from core.risk.liquidity import liquidity_for_symbol

    block = LiquidityConfig(enabled=True, impact_k=0.4,
                            per_symbol={"BTCUSDT": {"max_participation_pct": 2.0},
                                        "*": {"lookback_bars": 7}})
    btc = liquidity_for_symbol(block, "BTCUSDT")
    other = liquidity_for_symbol(block, "ETHUSDT")
    assert btc.max_participation_pct == 2.0 and btc.lookback_bars == 20
    assert other.max_participation_pct == 1.0 and other.lookback_bars == 7
    # Both keep the top-level values they did not override.
    assert btc.impact_k == other.impact_k == 0.4
    assert btc.enabled is True
    # No overrides / no block: the same object (or None) comes back.
    plain = LiquidityConfig()
    assert liquidity_for_symbol(plain, "BTCUSDT") is plain
    assert liquidity_for_symbol(None, "BTCUSDT") is None
    assert liquidity_for_symbol(plain, "") is plain
