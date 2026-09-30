"""P6-C — market volume breadth (``core.market_data.breadth``).

No test here touches the network: every transport path is injected through
``fetch_breadth(fetcher=...)`` / ``fetch_tickers(opener=...)``, and the cache is
redirected to ``tmp_path``.  The live measurements (coverage, autocorrelation,
variance) are recorded in ``docs/core-algorithms/15-volume-bars-breadth.md`` and
reproduced by ``tools/p6_volume_bars_experiment.py breadth``; the series shapes
asserted below are built by the test so they cannot encode a market state.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.market_data.breadth import (
    MAX_STALE_MS, QUOTE_SUFFIX, TICKER24H_TTL_S, BreadthCache,
    BreadthObservation, BreadthUnavailable, TickerRow, aggregate_breadth,
    autocorrelation, availability_report, default_cache_path, differences,
    fetch_breadth, fetch_tickers, herfindahl, is_causal, is_usdt_pair,
    observation_values, parse_ticker, parse_tickers, replay, select_pairs,
    variance,
)


def ticker(symbol: str, quote_volume: float, change_pct: float) -> dict:
    """One endpoint entry in the wire format (everything is a string there)."""
    return {"symbol": symbol, "quoteVolume": str(quote_volume),
            "priceChangePercent": str(change_pct), "lastPrice": "1.0",
            "count": "10"}


def payload() -> list[dict]:
    return [ticker("BTCUSDT", 1000.0, 1.5), ticker("ETHUSDT", 500.0, -0.5),
            ticker("SOLUSDT", 250.0, 0.0), ticker("XRPUSDT", 250.0, 2.0),
            ticker("ETHBTC", 900.0, 0.1), ticker("BTCUPUSDT", 800.0, 9.0)]


# ── parsing ──────────────────────────────────────────────────────────────

def test_is_usdt_pair_keeps_spot_and_drops_leveraged_tokens():
    assert is_usdt_pair("BTCUSDT")
    assert is_usdt_pair("1000SHIBUSDT")
    assert not is_usdt_pair("BTCUPUSDT")
    assert not is_usdt_pair("ETHDOWNUSDT")
    assert not is_usdt_pair("BTCBULLUSDT")
    assert not is_usdt_pair("ETHBTC")
    assert not is_usdt_pair("USDT")
    assert not is_usdt_pair("")


def test_parse_ticker_never_fabricates_a_missing_field():
    assert parse_ticker(ticker("BTCUSDT", 10.0, 1.0)) == TickerRow(
        "BTCUSDT", 10.0, 1.0, 1.0, 10.0)
    for broken in ({"symbol": "BTCUSDT", "quoteVolume": None,
                    "priceChangePercent": "1"},
                   {"symbol": "BTCUSDT", "quoteVolume": "abc",
                    "priceChangePercent": "1"},
                   {"symbol": "", "quoteVolume": "1", "priceChangePercent": "1"},
                   {"symbol": "BTCUSDT", "priceChangePercent": "1"},
                   {"symbol": "BTCUSDT", "quoteVolume": "1"},
                   {"symbol": "BTCUSDT", "quoteVolume": "-1",
                    "priceChangePercent": "1"}):
        assert parse_ticker(broken) is None
    # A genuine zero is data, not a missing field.
    zero = parse_ticker(ticker("BTCUSDT", 0.0, 0.0))
    assert zero is not None and zero.quote_volume == 0.0


def test_parse_tickers_refuses_a_single_symbol_payload():
    with pytest.raises(BreadthUnavailable):
        parse_tickers({"symbol": "BTCUSDT", "quoteVolume": "1"})
    assert parse_tickers(payload()) != []


def test_select_pairs_applies_a_custom_filter_and_a_volume_floor():
    rows = parse_tickers(payload())
    assert [row.symbol for row in select_pairs(rows)] == [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    only_btc = select_pairs(rows, symbol_filter=lambda s: s == "BTCUSDT")
    assert [row.symbol for row in only_btc] == ["BTCUSDT"]
    assert [row.symbol for row in select_pairs(rows, min_quote_volume=300.0)] == [
        "BTCUSDT", "ETHUSDT"]
    # The floor is inclusive, so the default keeps a zero-volume pair.
    rows_with_zero = parse_tickers([ticker("BTCUSDT", 10.0, 1.0),
                                    ticker("ETHUSDT", 0.0, 0.0)])
    assert len(select_pairs(rows_with_zero)) == 2


# ── aggregation ──────────────────────────────────────────────────────────

def test_herfindahl_is_scale_invariant_and_undefined_without_volume():
    assert herfindahl([1.0, 1.0]) == pytest.approx(0.5)
    assert herfindahl([2.0, 2.0]) == pytest.approx(0.5)
    assert herfindahl([1.0]) == pytest.approx(1.0)
    assert herfindahl([]) is None
    assert herfindahl([0.0, 0.0]) is None


def test_aggregate_breadth_matches_a_hand_computed_observation():
    rows = parse_tickers(payload())
    obs = aggregate_breadth(rows, as_of_ms=1234, symbol_count=len(payload()),
                            expected_pair_count=8, top_n=2)
    assert obs is not None
    volumes = [1000.0, 500.0, 250.0, 250.0]          # USDT pairs only
    total = sum(volumes)
    assert obs.total_quote_volume == pytest.approx(total)
    assert obs.symbol_count == len(payload())
    assert obs.pair_count == 4
    assert obs.usable_count == 4
    assert obs.coverage == pytest.approx(4 / 8)
    assert obs.up_share == pytest.approx(2 / 4)       # BTC and XRP are up
    assert obs.down_share == pytest.approx(1 / 4)
    assert obs.flat_share == pytest.approx(1 / 4)
    assert obs.hhi == pytest.approx(sum((v / total) ** 2 for v in volumes))
    assert obs.effective_pairs == pytest.approx(1.0 / obs.hhi)
    assert obs.top_share == pytest.approx(1000.0 / total)
    assert obs.top_symbols[0][0] == "BTCUSDT"
    assert obs.median_quote_volume == pytest.approx(375.0)
    assert obs.as_of_ms == 1234
    assert obs.source == "live"
    assert obs.is_stale is False


def test_coverage_denominator_follows_the_documented_universe():
    rows = parse_tickers(payload())
    observed = aggregate_breadth(rows, as_of_ms=1, symbol_count=6)
    documented = aggregate_breadth(rows, as_of_ms=1, symbol_count=6,
                                   expected_pair_count=496)
    assert observed.coverage == pytest.approx(4 / 4)      # all four USDT pairs
    assert documented.coverage == pytest.approx(4 / 496)
    assert documented.expected_pair_count == 496


def test_zero_volume_pairs_are_a_coverage_miss_not_a_share():
    rows = parse_tickers([ticker("BTCUSDT", 100.0, 1.0),
                          ticker("ETHUSDT", 0.0, 0.0),
                          ticker("SOLUSDT", 0.0, -1.0)])
    obs = aggregate_breadth(rows, as_of_ms=1, symbol_count=3)
    assert obs.usable_count == 1 and obs.pair_count == 3
    assert obs.up_share == pytest.approx(1.0)      # the zero-volume pair is out
    assert obs.coverage == pytest.approx(1 / 3)


def test_degenerate_universes_return_none_rather_than_zero():
    assert aggregate_breadth([], as_of_ms=1) is None
    assert aggregate_breadth([TickerRow("BTCUSDT", 0.0, 1.0)], as_of_ms=1) is None
    assert aggregate_breadth(parse_tickers([ticker("ETHBTC", 10.0, 1.0)]),
                             as_of_ms=1) is None


# ── transport and graceful degradation ───────────────────────────────────

def test_fetch_tickers_retries_then_raises():
    calls = {"n": 0}

    def failing_opener(url: str, timeout: float) -> bytes:
        calls["n"] += 1
        raise OSError("connection reset")

    with pytest.raises(BreadthUnavailable):
        fetch_tickers(opener=failing_opener, attempts=2)
    assert calls["n"] == 2


def test_fetch_tickers_recovers_on_the_retry():
    calls = {"n": 0}

    def flaky_opener(url: str, timeout: float) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return json.dumps(payload()).encode()

    assert fetch_tickers(opener=flaky_opener, attempts=2) == payload()


def test_fetch_tickers_rejects_a_non_list_payload():
    def bad_opener(url: str, timeout: float) -> bytes:
        return b'{"symbol": "BTCUSDT"}'

    with pytest.raises(BreadthUnavailable):
        fetch_tickers(opener=bad_opener, attempts=1)


def test_unreachable_endpoint_yields_none_and_caches_nothing(tmp_path: Path):
    def explode():
        raise OSError("network unreachable")

    assert fetch_breadth(fetcher=explode) is None
    cache = BreadthCache(tmp_path / "breadth.jsonl", now_ms=1_000_000)
    observation, status = cache.refresh(fetcher=explode)
    assert observation is None
    assert status == "unavailable"
    assert cache.load() == []
    assert not (tmp_path / "breadth.jsonl").exists()


def test_fetch_breadth_labels_the_observation_with_its_clock():
    obs = fetch_breadth(fetcher=lambda: payload(), now_ms=555,
                        expected_pair_count=4)
    assert obs is not None
    assert obs.as_of_ms == 555
    assert obs.request_started_ms == 555
    assert obs.fetch_ms is not None and obs.fetch_ms >= 0.0
    assert obs.coverage == pytest.approx(1.0)
    assert obs.top_symbols[0][0] == "BTCUSDT"


# ── cache: TTL, staleness, torn lines ────────────────────────────────────

def test_cache_ttl_and_staleness_labels(tmp_path: Path):
    path = tmp_path / "breadth.jsonl"
    writer = BreadthCache(path, now_ms=1_000_000)
    observation, status = writer.refresh(fetcher=lambda: payload(),
                                         expected_pair_count=4)
    assert status == "live" and observation is not None

    inside = BreadthCache(path, now_ms=1_000_000 + int(TICKER24H_TTL_S * 1000))
    assert inside.fresh() is not None
    assert inside.latest().is_stale is False
    assert inside.refresh(fetcher=None)[1] == "fresh-cache"

    outside = BreadthCache(path, now_ms=1_000_000
                           + int(TICKER24H_TTL_S * 1000) + 1)
    assert outside.fresh() is None
    stale = outside.latest()
    assert stale is not None and stale.is_stale is True
    assert stale.source == "cache"
    assert stale.stale_ms == int(TICKER24H_TTL_S * 1000) + 1

    # Beyond the documented stale bound it is still *labelled*, never fresh.
    old = BreadthCache(path, now_ms=1_000_000 + MAX_STALE_MS + 1)
    assert old.fresh() is None and old.latest().is_stale is True


def test_cache_falls_back_to_the_labelled_stale_value(tmp_path: Path):
    path = tmp_path / "breadth.jsonl"
    BreadthCache(path, now_ms=1_000_000).refresh(fetcher=lambda: payload(),
                                                 expected_pair_count=4)

    def explode():
        raise OSError("down")

    late = BreadthCache(path, now_ms=1_000_000 + MAX_STALE_MS + 1)
    observation, status = late.refresh(fetcher=explode)
    assert status == "stale-cache"
    assert observation is not None and observation.is_stale is True


def test_cache_force_fetches_inside_the_ttl(tmp_path: Path):
    path = tmp_path / "breadth.jsonl"
    cache = BreadthCache(path, now_ms=1_000_000)
    cache.refresh(fetcher=lambda: payload(), expected_pair_count=4)
    assert cache.refresh(fetcher=None)[1] == "fresh-cache"
    observation, status = cache.refresh(fetcher=lambda: payload(),
                                       expected_pair_count=4, force=True)
    assert status == "live" and observation is not None
    assert len(cache.load()) == 2


def test_cache_skips_a_torn_last_line_and_never_writes_none(tmp_path: Path):
    path = tmp_path / "breadth.jsonl"
    cache = BreadthCache(path, now_ms=1_000_000)
    assert cache.append(None) is False
    assert not path.exists()
    good = aggregate_breadth(parse_tickers(payload()), as_of_ms=1_000_000)
    path.write_text(good.to_json() + "\n" + '{"as_of_ms": 2, "total_qu',
                    encoding="utf-8")
    loaded = cache.load()
    assert len(loaded) == 1 and loaded[0].as_of_ms == 1_000_000


def test_observation_json_round_trip_is_lossless(tmp_path: Path):
    obs = aggregate_breadth(parse_tickers(payload()), as_of_ms=99, top_n=3)
    assert BreadthObservation.from_dict(obs.to_dict()) == obs
    path = tmp_path / "b.jsonl"
    cache = BreadthCache(path, now_ms=99)
    cache.append(obs)
    assert cache.load() == [obs]


def test_default_cache_path_is_not_the_live_market_tree():
    path = default_cache_path("data")
    assert "market" not in path.parts
    assert path.name.endswith(".jsonl")
    assert QUOTE_SUFFIX in "BTCUSDT"


# ── series measurement ───────────────────────────────────────────────────

def test_autocorrelation_and_variance_are_exact_on_known_series():
    assert autocorrelation([0.0, 1.0] * 50, 1) == pytest.approx(-1.0)
    assert autocorrelation(list(range(50)), 1) > 0.99
    assert autocorrelation([1.0] * 50, 1) is None       # no variance
    assert autocorrelation([1.0, 2.0], 1) is None       # too short
    assert differences([1.0, 3.0, 6.0]) == [2.0, 3.0]
    assert variance([1.0, 3.0]) == pytest.approx(1.0)
    assert variance([1.0]) is None
    assert observation_values(
        [BreadthObservation(as_of_ms=1, up_share=0.25),
         BreadthObservation(as_of_ms=2, up_share=None)], "up_share") == [0.25]


def test_availability_report_flags_a_rolling_window_series_as_degenerate():
    """A 24 h rolling quantity overlaps itself: |acf1| ≈ 1 is the expected
    result, and the report must say so rather than call it non-degenerate."""
    observations = []
    level = 1000.0
    for index in range(30):
        level += 1.0 if index % 3 else 0.5      # a near-monotone random walk
        observations.append(BreadthObservation(
            as_of_ms=1_000_000 + index * 60_000, total_quote_volume=level,
            up_share=0.5, hhi=0.02, coverage=0.99, pair_count=500,
            usable_count=495))
    report = availability_report(observations)
    assert report["total_quote_volume"]["acf1"] > 0.99
    assert report["non_degenerate_acf_lt_0_99"]["total_quote_volume"] is False
    assert report["variance_gt_zero"]["total_quote_volume"] is True
    # The increments carry the new information the level hides.
    assert abs(report["total_quote_volume"]["acf1_differenced"]) < 0.99
    assert report["coverage"]["min"] == pytest.approx(0.99)
    assert report["causal_labels"]["causal"] is True


def test_is_causal_checks_the_labels_it_can_check():
    ok = [BreadthObservation(as_of_ms=10, request_started_ms=9),
          BreadthObservation(as_of_ms=20, request_started_ms=19)]
    assert is_causal(ok)["causal"] is True
    backwards = [BreadthObservation(as_of_ms=20),
                 BreadthObservation(as_of_ms=10)]
    report = is_causal(backwards)
    assert report["causal"] is False
    assert "backwards" in report["problems"][0]
    impossible = [BreadthObservation(as_of_ms=10, request_started_ms=11)]
    assert is_causal(impossible)["causal"] is False
    assert is_causal([])["causal"] is True


def test_replay_filters_by_label_and_never_backfills():
    observations = [BreadthObservation(as_of_ms=ms) for ms in (10, 20, 30)]
    assert [o.as_of_ms for o in replay(observations, upto_ms=20)] == [10, 20]
    assert replay(observations, upto_ms=5) == []
