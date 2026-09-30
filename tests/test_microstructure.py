"""Regression tests for phase P4 — microstructure features from depth + trades.

Claims under test:

* every feature is a pure function of one snapshot and the formulas are the
  documented ones (checked against hand-computed books);
* **no look-ahead**: a trade stamped after the decision timestamp is dropped
  *before* any statistic, and injecting a huge future print leaves every feature
  bit-identical (the test that matters most);
* the compute is bounded (``MAX_DEPTH_LEVELS`` / ``MAX_TRADES``) whatever the
  exchange returns;
* the cache is a bounded TTL cache holding computed features only;
* the fetch wrapper works against a stub (offline) and degrades to ``None``
  instead of raising when the transport fails.

**No test here touches the network.**  The "provider is available" branch is
driven by a Binance-shaped snapshot built inside the test, so the suite cannot
depend on connectivity, on the live tape's trade rate, or on the host's clock —
the three things that used to make this file's last test fail in a full run and
pass standalone (see that test's docstring).  The cost of that choice is stated
there too: the real ``MarketDataClient`` is never exercised end to end by this
file.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.market_data.microstructure import (
    FEATURE_KEYS, MAX_CACHE_ENTRIES, MAX_DEPTH_LEVELS, MAX_TRADES,
    MicrostructureCache, aggressor_is_buy, best_bid_ask, book_slope_ratio,
    compute_features, depth_weighted_imbalance, fetch_features, filter_trades,
    microprice, microprice_deviation_bps, mid_price, order_flow_imbalance,
    realized_volatility, spread_bps, trade_arrival_intensity,
    trade_flow_imbalance, trade_size_stats,
)


def _book(bids, asks) -> dict:
    return {"bids": [[str(p), str(q)] for p, q in bids],
            "asks": [[str(p), str(q)] for p, q in asks]}


def _trade(qty, price, ts, buyer_maker=True) -> dict:
    return {"price": str(price), "qty": str(qty), "quoteQty": str(price * qty),
            "time": int(ts), "isBuyerMaker": buyer_maker}


@pytest.fixture(autouse=True)
def _restore_fetch_features_last_error():
    """Restore ``fetch_features.last_error`` around every test in this file.

    That attribute is the **only** module-global mutable state these tests touch
    (``fetch_features`` has no module-level payload cache and no cached provider;
    every ``MicrostructureCache`` here is a local).  It is a diagnostic, but a
    test that read a neighbour's stale message -- e.g. in a skip reason -- would
    be order-dependent, so this file never lets one leak out.
    """
    previous = fetch_features.last_error
    yield
    fetch_features.last_error = previous


# ── order book features ─────────────────────────────────────────────────

def test_best_bid_ask_mid_and_spread_on_a_known_book():
    book = _book([(99.0, 2.0), (98.0, 3.0)], [(101.0, 1.0), (102.0, 4.0)])
    assert best_bid_ask(book) == ((99.0, 2.0), (101.0, 1.0))
    assert mid_price(book) == pytest.approx(100.0)
    # (101 - 99) / 100 * 1e4 = 200 bp
    assert spread_bps(book) == pytest.approx(200.0)
    assert np.isnan(mid_price({"bids": [], "asks": []}))


def test_order_flow_imbalance_is_the_documented_ratio():
    book = _book([(99.0, 6.0)], [(101.0, 2.0)])
    # (6 - 2) / (6 + 2) = 0.5
    assert order_flow_imbalance(book) == pytest.approx(0.5)
    assert order_flow_imbalance(_book([], [(101.0, 2.0)])) == pytest.approx(-1.0)
    # Zero quantity on both sides is *no information*, not a signal.
    assert order_flow_imbalance(_book([(99.0, 0.0)], [(101.0, 0.0)])) == 0.0
    assert np.isnan(order_flow_imbalance({"bids": [], "asks": []}))


def test_depth_weighted_imbalance_puts_more_weight_on_the_touch():
    # A big order far from the touch must not outvote the near book: the
    # unweighted ratio is 1001/1003 = 0.999, the weighted one is far smaller.
    book = _book([(99.0, 1.0), (50.0, 1000.0)], [(101.0, 1.0), (200.0, 1.0)])
    weighted = depth_weighted_imbalance(book)
    unweighted = order_flow_imbalance(book)
    assert unweighted > 0.99
    assert 0.0 < weighted < 0.5
    assert weighted < unweighted - 0.4


def test_microprice_weights_the_touch_by_the_opposite_size():
    book = _book([(99.0, 9.0)], [(101.0, 1.0)])
    # (99 * 1 + 101 * 9) / 10 = 100.8 -> above the mid, buy pressure at the touch
    assert microprice(book) == pytest.approx(100.8)
    assert mid_price(book) == pytest.approx(100.0)
    assert microprice_deviation_bps(book) == pytest.approx(80.0)
    # Zero touch sizes fall back to the mid rather than dividing by zero.
    assert microprice(_book([(99.0, 0.0)], [(101.0, 0.0)])) == pytest.approx(100.0)


def test_book_slope_ratio_measures_relative_depth_per_basis_point():
    # Offer concentrated at the touch, bid spread away -> offer slope is steeper.
    near_offer = _book([(99.0, 1.0), (98.0, 1.0), (97.0, 1.0)],
                       [(101.0, 1.0)])
    assert book_slope_ratio(near_offer) > 1.0
    # Mirror image: the bid is at the touch and the offer is spread away.
    near_bid = _book([(99.0, 1.0)],
                     [(101.0, 1.0), (102.0, 1.0), (103.0, 1.0)])
    assert book_slope_ratio(near_bid) < 1.0
    assert np.isfinite(book_slope_ratio(near_bid))   # never inf/inf


# ── trades ──────────────────────────────────────────────────────────────

def test_aggressor_side_follows_is_buyer_maker():
    assert aggressor_is_buy({"isBuyerMaker": False}) is True   # taker bought
    assert aggressor_is_buy({"isBuyerMaker": True}) is False   # taker sold
    assert aggressor_is_buy({}) is False


def test_trade_flow_imbalance_uses_aggressor_volume():
    trades = [
        _trade(1.0, 100.0, 1000, buyer_maker=False),   # taker buy 1
        _trade(3.0, 100.0, 1001, buyer_maker=True),    # taker sell 3
    ]
    assert trade_flow_imbalance(trades) == pytest.approx((1.0 - 3.0) / 4.0)
    assert trade_flow_imbalance([]) == 0.0


def test_trade_size_stats_report_count_median_and_large_share():
    trades = [_trade(1.0, 100.0, 1000 + i) for i in range(9)]
    trades.append(_trade(91.0, 100.0, 2000))          # one whale
    stats = trade_size_stats(trades)
    assert stats["trade_count"] == 10
    assert stats["trade_median_qty"] == pytest.approx(1.0)
    assert stats["trade_mean_qty"] == pytest.approx(10.0)
    # volume above the 90th percentile is 91 of 100 units
    assert stats["trade_large_share"] == pytest.approx(0.91)
    assert stats["trade_size_skew"] > 0
    assert stats["trade_notional"] == pytest.approx(100.0 * 100.0)


def test_realized_volatility_matches_a_known_geometric_series():
    prices = 100.0 * np.exp(np.cumsum(np.full(40, 0.01)))
    rv = realized_volatility(prices, stride=1)
    assert rv == pytest.approx(np.sqrt(39) * 0.01, rel=1e-9)
    # Every 10th price means each sampled return is 10 x 0.01 = 0.1.
    assert realized_volatility(prices, stride=10) == pytest.approx(
        np.sqrt(3) * 0.1, rel=1e-9)
    assert np.isnan(realized_volatility([100.0], stride=1))
    annual = realized_volatility(prices, stride=1, annualize_bars=8760)
    assert annual == pytest.approx(rv * np.sqrt(8760.0))


def test_trade_arrival_intensity_and_activity_ratio():
    # 10 prints over 0.9 s, then 10 more over 0.45 s -> the tape is speeding up
    older = [_trade(1.0, 100.0, 1000 + i * 100) for i in range(10)]
    newer = [_trade(1.0, 100.0, 2000 + i * 50) for i in range(10)]
    res = trade_arrival_intensity(older + newer)
    assert res["arrival_rate_hz"] > 0
    assert res["span_seconds"] == pytest.approx(1.45)
    assert res["activity_ratio"] > 1.0
    assert trade_arrival_intensity([])["arrival_rate_hz"] == 0.0


# ── no look-ahead ───────────────────────────────────────────────────────

def test_filter_trades_drops_every_print_after_the_decision_time():
    trades = [_trade(1.0, 100.0, 1000), _trade(1.0, 100.0, 2000),
              _trade(1.0, 100.0, 3000)]
    kept, dropped = filter_trades(trades, as_of_ms=2000)
    assert [t["time"] for t in kept] == [1000, 2000]
    assert dropped == 1
    kept, dropped = filter_trades(trades, as_of_ms=None)
    assert len(kept) == 3 and dropped == 0


def test_compute_features_ignores_a_trade_stamped_after_the_decision():
    """The look-ahead test: a future print must change **nothing**."""
    book = _book([(99.0, 5.0)], [(101.0, 5.0)])
    trades = [_trade(1.0, 100.0, 1000), _trade(2.0, 100.0, 1500, buyer_maker=False)]
    as_of = 1500.0
    base = compute_features(book, trades, as_of_ms=as_of, book_time_ms=as_of - 50)
    assert base["dropped_future_trades"] == 0
    future = _trade(1000.0, 100.0, 9999, buyer_maker=False)
    after = compute_features(book, trades + [future], as_of_ms=as_of,
                             book_time_ms=as_of - 50)
    assert after["dropped_future_trades"] == 1
    for key in FEATURE_KEYS:
        if key == "dropped_future_trades":
            continue
        assert after[key] == base[key] or (np.isnan(after[key]) and np.isnan(base[key])), key
    assert after["ofi_trades"] == base["ofi_trades"]
    assert after["trade_count"] == base["trade_count"]
    assert after["book_age_ms"] == pytest.approx(50.0)
    assert after["as_of_ms"] == pytest.approx(as_of)


def test_a_future_print_would_have_flipped_the_flow_imbalance():
    """Proof the previous test is not vacuous: without the drop it flips."""
    trades = [_trade(1.0, 100.0, 1000, buyer_maker=False)]
    future = _trade(1000.0, 100.0, 5000)
    kept, dropped = filter_trades(trades + [future], as_of_ms=1000)
    assert dropped == 1
    assert trade_flow_imbalance(trades) == pytest.approx(1.0)
    assert trade_flow_imbalance(trades + [future]) < -0.9


# ── bounds + cache ──────────────────────────────────────────────────────

def test_bounds_are_enforced_whatever_the_exchange_returns():
    book = _book([(99.0 - i, 1.0) for i in range(500)],
                 [(101.0 + i, 1.0) for i in range(500)])
    trades = [_trade(1.0, 100.0, 1000 + i) for i in range(5 * MAX_TRADES)]
    feats = compute_features(book, trades, as_of_ms=10 ** 9, depth_limit=1000)
    assert feats["n_depth_levels"] <= 2 * MAX_DEPTH_LEVELS
    assert feats["trade_count"] <= MAX_TRADES
    assert set(FEATURE_KEYS) <= set(feats)


def test_feature_contract_is_stable():
    feats = compute_features(_book([(99.0, 1.0)], [(101.0, 1.0)]), [],
                             as_of_ms=1.0)
    for key in FEATURE_KEYS:
        assert key in feats, key
    assert feats["book_age_ms"] is None      # no book_time_ms supplied


def test_cache_is_bounded_and_expires_by_ttl():
    cache = MicrostructureCache(ttl_secs=10.0, max_entries=2)
    cache.put("A", {"mid": 1.0}, now=0.0)
    cache.put("B", {"mid": 2.0}, now=1.0)
    assert cache.get("A", now=5.0) == {"mid": 1.0}
    assert cache.age("A", now=5.0) == pytest.approx(5.0)
    assert cache.get("A", now=11.0) is None          # expired
    cache.put("A", {"mid": 1.0}, now=12.0)
    assert len(cache) == 2                           # B (now=1) + A (now=12)
    cache.put("C", {"mid": 3.0}, now=13.0)           # evicts the oldest stamp
    assert len(cache) == 2
    assert cache.get("B", now=13.5) is None          # B was the oldest
    assert cache.get("C", now=13.5) == {"mid": 3.0}
    cache.clear()
    assert len(cache) == 0
    assert len(MicrostructureCache(max_entries=MAX_CACHE_ENTRIES)) == 0


# ── fetch wrapper (stubbed: no network in the suite) ────────────────────

class _StubClient:
    """Any object with the two documented coroutines is a valid provider."""

    def __init__(self, book, trades, fail: bool = False):
        self._book, self._trades, self._fail = book, trades, fail
        self.calls = 0

    async def order_book(self, symbol, limit=20):
        self.calls += 1
        if self._fail:
            raise RuntimeError("boom")
        return self._book

    async def recent_trades(self, symbol, limit=100):
        if self._fail:
            raise RuntimeError("boom")
        return self._trades


def _snapshot_book(price: float = 100.0, tick: float = 0.01,
                   levels: int = 20) -> dict:
    """A ``/api/v3/depth`` payload: string price/qty pairs, 20 levels a side."""
    return {
        "lastUpdateId": 7,
        "bids": [[f"{price - tick * (i + 1):.2f}", f"{2.0 + i:.6f}"]
                 for i in range(levels)],
        "asks": [[f"{price + tick * (i + 1):.2f}", f"{2.5 + i:.6f}"]
                 for i in range(levels)],
    }


#: The injected tape is stamped in the past, so the live ``as_of_ms=None`` shape
#: keeps every print whatever the wall clock says when the suite runs.
SNAPSHOT_START_MS = 1_700_000_000_000
SNAPSHOT_STEP_MS = 25
SNAPSHOT_TRADES = 100


def _snapshot_trades(count: int = SNAPSHOT_TRADES,
                     start_ms: int = SNAPSHOT_START_MS,
                     step_ms: int = SNAPSHOT_STEP_MS,
                     price: float = 100.0) -> list[dict]:
    """A ``/api/v3/trades`` payload: string price/qty + ``isBuyerMaker`` bools."""
    out = []
    for i in range(count):
        px = price + (0.01 if i % 2 else -0.01)
        out.append({"id": i, "price": f"{px:.2f}", "qty": "0.5",
                    "quoteQty": f"{px * 0.5:.4f}",
                    "time": int(start_ms + i * step_ms),
                    "isBuyerMaker": bool(i % 2), "isBestMatch": True})
    return out


@pytest.mark.asyncio
async def test_fetch_features_uses_the_client_and_the_cache():
    book = _book([(99.0, 5.0)], [(101.0, 5.0)])
    trades = [_trade(1.0, 100.0, 1000)]
    client = _StubClient(book, trades)
    cache = MicrostructureCache(ttl_secs=60.0)
    first = await fetch_features(client, "BTCUSDT", cache=cache)
    assert set(FEATURE_KEYS) <= set(first)
    assert first["symbol"] == "BTCUSDT"
    assert first["mid"] == pytest.approx(100.0)
    calls = client.calls
    second = await fetch_features(client, "BTCUSDT", cache=cache)
    assert client.calls == calls          # served from the cache
    assert second["mid"] == first["mid"]


@pytest.mark.asyncio
async def test_fetch_features_degrades_to_none_on_transport_failure():
    client = _StubClient({}, [], fail=True)
    assert await fetch_features(client, "BTCUSDT") is None
    assert "boom" in (fetch_features.last_error or "")


@pytest.mark.asyncio
async def test_live_snapshot_is_optional_and_correct_when_available():
    """**The claim: the microstructure provider is optional and correct when
    available.**  "Available" is the live-shape payload the real
    ``MarketDataClient`` serves -- ``/api/v3/depth`` + ``/api/v3/trades`` raw JSON
    (string prices/quantities, ``isBuyerMaker`` booleans) -- injected here as a
    synthetic snapshot instead of fetched.

    Why injected: the previous version asserted ``feats["arrival_rate_hz"] > 0``
    on a **live** fetch.  That is a measurement of the host's tape, not a property
    of this module, and ``fetch_features`` stamps ``as_of_ms`` *before* the two
    requests: when the tape is busy enough that 100 prints span less than the
    round-trip, the look-ahead filter drops every print and the rate is exactly
    ``0.0``.  Measured on this host: 3 of 40 standalone live fetches returned
    ``arrival_rate_hz == 0.0`` (dropped 3/90/100 of 100) -- no other test needed
    to run first, so this was a live-timing dependence, not an ordering one.  The
    absent/degenerate branches are asserted in the next test; here the provider
    answers, so every feature must satisfy the documented contract.

    **Stated limitation: nothing here exercises the real provider end to end.**
    The whole module is offline by design (see the module docstring) and this
    rewritten test only feeds the stub client a stored Binance-shaped payload, so
    the wiring from ``MarketDataClient.get_depth``/``get_recent_trades`` into
    ``fetch_features`` -- URL, parsing, authentication-free public endpoints -- is
    not covered by any test in this suite.  That coverage was traded away
    deliberately: the live-fetch version failed on the host's tape timing, not on
    the code, and a flaky gate is worse than a documented gap.  Verify that wiring
    manually (or in a network-enabled check) before trusting a provider change.
    """
    client = _StubClient(_snapshot_book(), _snapshot_trades())
    feats = await fetch_features(client, "BTCUSDT")          # the live-now shape

    assert feats is not None
    assert set(FEATURE_KEYS) <= set(feats)
    assert feats["symbol"] == "BTCUSDT" and feats["source"] == "snapshot"
    # 20 levels a side, touch 99.99/100.01 -> mid 100.00, 2 bp spread.
    assert feats["n_depth_levels"] == 40
    assert feats["mid"] == pytest.approx(100.0)
    assert feats["spread_bps"] == pytest.approx(2.0)
    assert -1.0 <= feats["ofi_depth"] <= 1.0
    assert -1.0 <= feats["ofi_trades"] <= 1.0
    # 100 prints, 25 ms apart: span 2.475 s -> 99/2.475 = 40 Hz.
    assert feats["trade_count"] == 100
    assert feats["arrival_rate_hz"] == pytest.approx(40.0)
    assert feats["dropped_future_trades"] == 0
    assert feats["trade_count"] <= 100

    # The same snapshot at an explicit decision time differs only in the two
    # stamps: nothing in the compute reads the wall clock.
    stamped = await fetch_features(client, "BTCUSDT",
                                   as_of_ms=SNAPSHOT_START_MS
                                   + (SNAPSHOT_TRADES - 1) * SNAPSHOT_STEP_MS)
    assert stamped is not None
    for key in FEATURE_KEYS:
        if key in ("as_of_ms", "book_age_ms"):
            continue
        assert np.isclose(stamped[key], feats[key], equal_nan=True), key


@pytest.mark.asyncio
async def test_an_absent_or_degenerate_snapshot_degrades_to_the_fallback():
    """The **optional** half of the claim, on the three shapes seen live.

    * transport failure -> ``None`` plus ``last_error`` (never an exception);
    * an empty page -> the full key contract with the documented *no information*
      values (``nan`` mid/spread, no depth, no trades, ``0.0`` Hz);
    * a busy tape, where 100 prints span less than the request's round-trip ->
      every print postdates ``as_of_ms`` and is dropped: ``0`` trades,
      ``dropped_future_trades == 100``, ``0.0`` Hz.  This is the exact payload
      that failed the old live assertion ``arrival_rate_hz > 0``;
    * a one-millisecond burst -> prints kept but span ``0`` -> ``0.0`` Hz too.
    """
    assert await fetch_features(_StubClient({}, [], fail=True), "BTCUSDT") is None
    assert fetch_features.last_error

    empty = await fetch_features(
        _StubClient({"lastUpdateId": 7, "bids": [], "asks": []}, []), "BTCUSDT")
    assert empty is not None
    assert set(FEATURE_KEYS) <= set(empty)
    assert np.isnan(empty["mid"]) and np.isnan(empty["spread_bps"])
    assert empty["n_depth_levels"] == 0
    assert empty["trade_count"] == 0 and empty["arrival_rate_hz"] == 0.0

    # Year 2100 stamps: present payload, every print after the decision stamp.
    future = await fetch_features(
        _StubClient(_snapshot_book(),
                    _snapshot_trades(start_ms=4_102_444_800_000)), "BTCUSDT")
    assert future is not None
    assert set(FEATURE_KEYS) <= set(future)
    assert future["dropped_future_trades"] == 100
    assert future["trade_count"] == 0
    assert future["arrival_rate_hz"] == 0.0     # the value the old assert rejected
    assert future["mid"] == pytest.approx(100.0)   # the book half still works

    burst = await fetch_features(
        _StubClient(_snapshot_book(), _snapshot_trades(step_ms=0)), "BTCUSDT")
    assert burst is not None
    assert burst["trade_count"] == 100 and burst["dropped_future_trades"] == 0
    assert burst["arrival_rate_hz"] == 0.0     # zero span is "no rate", not a rate
    assert np.isnan(burst["activity_ratio"])
