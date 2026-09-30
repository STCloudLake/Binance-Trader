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

A live snapshot check is skipped when the network is unavailable, so the suite
never depends on connectivity.
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
    """Skipped when the public market-data host is unreachable (offline CI)."""
    from core.market_data.data_client import MarketDataClient

    client = MarketDataClient("https://data-api.binance.vision", timeout=10.0)
    try:
        feats = await fetch_features(client, "BTCUSDT")
    except Exception as e:  # pragma: no cover - network dependent
        pytest.skip(f"market-data host unavailable: {e}")
    finally:
        await client.close()
    if feats is None:
        pytest.skip(f"market-data host unavailable: {fetch_features.last_error}")
    assert set(FEATURE_KEYS) <= set(feats)
    assert feats["mid"] > 0
    assert -1.0 <= feats["ofi_trades"] <= 1.0
    assert feats["arrival_rate_hz"] > 0        # BTCUSDT prints constantly
    assert feats["trade_count"] <= 100
    # `as_of_ms` is stamped *before* the request, so prints that arrive while it
    # is in flight are correctly counted as unavailable at decision time — the
    # look-ahead guard is expected to drop a few on a live fetch.
    assert feats["dropped_future_trades"] >= 0
    assert feats["spread_bps"] < 100.0         # a sanity band for a major
