"""Tests for the 代币检测 screener (`core/market_data/screener.py`) and its API
(`web/routes/audit.py`).

**No real network.**  Every upstream response comes from :class:`FakeExchange`,
an injectable ``fetch`` callable, so the scoring path, the sampling rules, the
cache and the failure modes are all exercised deterministically.  The web tests
use a temporary SQLite database and a throwaway login, like
``tests/test_market_api.py``.
"""
import asyncio
import math
import sys
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.market_data import screener as S  # noqa: E402


def run(coro):
    return asyncio.run(coro)


# ======================================================================
# stubbed exchange
# ======================================================================
def _oscillating(i: int, amp: float = 0.01) -> float:
    """Deterministic, low-volatility series with no long same-direction run."""
    return 100.0 * (1.0 + amp * math.sin(i * 0.7) + amp * 0.4 * math.cos(i * 1.9))


def _kline(o, h, l, c, v, close_ms):
    return [int(close_ms - 3_599_000), str(o), str(h), str(l), str(c), str(v),
            int(close_ms), str(c * v), 10, "0", "0", "0"]


class FakeExchange:
    """Injectable ``fetch`` producing Binance-shaped payloads per symbol."""

    def __init__(self, symbols, depth_qty=(2000.0, 0.0), spread_bps=0.5,
                 trade_count=1000, default_cfg=None):
        self.symbols = list(symbols)
        self.calls = []
        self.per_symbol = {}
        self.depth_qty = depth_qty
        self.spread_bps = spread_bps
        self.trade_count = trade_count
        self.default_cfg = dict(default_cfg or {})
        self.fail_paths = {}      # path -> exception
        self.raise_on = set()     # symbols whose klines fail

    def cfg(self, symbol):
        merged = dict(self.default_cfg)
        merged.update(self.per_symbol.get(symbol, {}))
        merged.setdefault("quote_volume", 1_000_000.0)
        merged.setdefault("count", self.trade_count)
        merged.setdefault("last", 1.0)
        merged.setdefault("volatility", 0.002)
        merged.setdefault("daily_mode", "flat")
        merged.setdefault("age_days", 400)
        merged.setdefault("wick_ratio", 0.2)
        merged.setdefault("gap", False)
        return merged

    async def __call__(self, path, params):
        self.calls.append((path, dict(params)))
        if path in self.fail_paths:
            raise self.fail_paths[path]
        if path == "/api/v3/ticker/24hr":
            return [{
                "symbol": sym,
                "lastPrice": str(self.cfg(sym)["last"]),
                "priceChangePercent": "1.5",
                "highPrice": "1.1",
                "lowPrice": "0.9",
                "volume": "1000",
                "quoteVolume": str(self.cfg(sym)["quote_volume"]),
                "count": self.cfg(sym)["count"],
            } for sym in self.symbols]
        symbol = params.get("symbol")
        cfg = self.cfg(symbol)
        if symbol in self.raise_on:
            raise RuntimeError(f"upstream refused {symbol}")
        if path == "/api/v3/depth":
            return self._depth(cfg)
        if path == "/api/v3/klines" and params.get("interval") == "1d":
            if params.get("limit") == 1:
                ms = int((time.time() - cfg["age_days"] * 86_400) * 1000)
                return [_kline(1, 1, 1, 1, 1, ms)]
            return self._daily(cfg)
        if path == "/api/v3/klines":
            return self._hourly(cfg)
        raise AssertionError(f"unexpected path {path}")

    def _depth(self, cfg):
        mid = float(cfg["last"])
        half = mid * self.spread_bps / 20_000.0
        best_bid, best_ask = mid - half, mid + half
        bid_qty, ask_qty = self.depth_qty
        bids = [[round(best_bid - i * mid * 1e-5, 8), bid_qty] for i in range(20)]
        asks = [[round(best_ask + i * mid * 1e-5, 8), ask_qty] for i in range(20)]
        return {"lastUpdateId": 1, "bids": bids, "asks": asks}

    def _hourly(self, cfg):
        close_ms = int(time.time() * 1000)
        rows = []
        amp = float(cfg["volatility"])
        n = 720
        for i in range(n):
            if cfg["gap"]:
                close = 100.0 + i * 0.5          # strictly monotone → long run
            elif cfg.get("spikes"):
                close = _oscillating(i, 0.0005)
                if i == n - 2:
                    close = close * 1.25         # one >15% bar
            else:
                close = _oscillating(i, amp)
            high = close * (1.0 + float(cfg["wick_ratio"]) * amp)
            low = close * (1.0 - amp)
            rows.append(_kline(close, high, low, close, 1000.0, close_ms - (n - i) * 3_600_000))
        return rows

    def _daily(self, cfg):
        mode = cfg["daily_mode"]
        rows = []
        close_ms = int(time.time() * 1000)
        for i in range(91):
            if mode == "crash":
                # 60% decline over the window, then flat
                close = 100.0 * (0.4 if i >= 30 else 1.0 - 0.6 * i / 30.0)
            else:
                close = 100.0 + i * 0.1
            rows.append(_kline(close, close, close, close, 10.0, close_ms - (91 - i) * 86_400_000))
        return rows


def make_screener(exchange, **kwargs):
    return S.TokenScreener(fetch=exchange, attempts=1, **kwargs)


# ======================================================================
# pure estimators
# ======================================================================
def test_annualised_volatility_matches_the_definition():
    closes = [100.0]
    for i in range(1, 200):
        closes.append(closes[-1] * (1.0 + 0.01 * (1 if i % 2 else -1)))
    vol = S.annualised_volatility(closes, S.HOURS_PER_YEAR)
    assert vol is not None and vol > 0
    # A ±1% alternating return has stdev ≈ 0.01, so vol ≈ 0.01*sqrt(8760)
    assert vol == pytest.approx(0.01 * math.sqrt(S.HOURS_PER_YEAR), rel=0.15)


def test_annualised_volatility_needs_enough_bars():
    assert S.annualised_volatility([100.0, 101.0], 8760) is None
    assert S.annualised_volatility([], 8760) is None


def test_max_drawdown_is_peak_to_trough_percent():
    assert S.max_drawdown([100, 120, 60, 90]) == pytest.approx(50.0)
    assert S.max_drawdown([100, 110, 120]) == pytest.approx(0.0)
    assert S.max_drawdown([100]) is None


def test_spike_count_counts_only_big_bars():
    closes = [100, 100, 120, 120, 100, 100]      # +20% then -16.7%
    assert S.spike_count(closes) == 2
    assert S.spike_count([100, 101, 102, 103]) == 0


def test_longest_same_direction_run():
    assert S.longest_same_direction_run([1, 2, 3, 4, 3, 2]) == 3
    assert S.longest_same_direction_run([1, 2, 1, 2, 1, 2]) == 1
    assert S.longest_same_direction_run([1]) == 0


def test_kline_normalisation_skips_malformed_rows():
    rows = S._klines([[0, "1", "2", "0.5", "1.5", "3", 123], "junk", [1, 2], [1, "x", 2, 3, 4, 5, 6]])
    assert len(rows) == 1
    assert rows[0][:5] == [1.0, 2.0, 0.5, 1.5, 3.0]
    assert rows[0][5] == 123.0


# ======================================================================
# score mapping
# ======================================================================
@pytest.mark.parametrize("score,level", [
    (100.0, "low"), (60.0, "low"), (59.9, "medium"), (35.0, "medium"),
    (34.9, "high"), (0.0, "high"),
])
def test_risk_level_mapping(score, level):
    assert S.risk_level_for(score) == level


def test_band_tables_bound_every_value():
    """Pins the band direction: the healthiest value must score zero penalty."""
    checks = [
        (S.BANDS_QUOTE_VOLUME, 1e12, 0.0, 0.0, 1.0),
        (S.BANDS_DEPTH, 1e12, 0.0, 0.0, 1.0),
        (S.BANDS_TRADE_SIZE, 400.0, 1.0, 0.0, 1.0),
        (S.BANDS_VOLUME_RATIO, 1.5, 0.0, 0.0, 1.0),
        (S.BANDS_SPREAD, 1e-5, 10.0, 0.0, 1.0),
        (S.BANDS_VOLATILITY, 0.3, 10.0, 0.0, 1.0),
        (S.BANDS_DRAWDOWN, 1.0, 200.0, 0.0, 1.0),
        (S.BANDS_WICK, 0.2, 1.0, 0.0, 1.0),
        (S.BANDS_SPIKE_RATE, 0.0, 5.0, 0.0, 1.0),
        (S.BANDS_GAP_RATE, 0.0, 5.0, 0.0, 1.0),
    ]
    for bands, good, bad, expected_good, expected_bad in checks:
        assert bands[-1] == (None, 0.0), bands
        assert S._band_fraction(good, bands) == expected_good, (bands, good)
        assert S._band_fraction(bad, bands) == expected_bad, (bands, bad)
    assert S._band_fraction(None, S.BANDS_QUOTE_VOLUME) is None
    assert S._band_fraction(30.0, S.BANDS_AGE) == 0.0
    assert S._band_fraction(5.0, S.BANDS_AGE) == 1.0


def test_penalty_constants_and_bands_agree():
    assert set(S.PENALTIES) == {
        "liquidity", "depth", "spread", "volatility", "drawdown", "spike",
        "wick", "trade_size", "gap", "age", "volume_activity"}


def test_score_is_100_minus_the_penalty_terms():
    clean = {k: None for k in S.SCORED_METRICS}
    clean.update({
        "quote_volume": 1e10, "depth_1pct_total": 5e7, "spread_pct": 1e-5,
        "volatility_30d_annualized": 0.4, "max_drawdown_90d": 5.0,
        "spike_count_30d": 0, "avg_wick_ratio": 0.28, "avg_trade_size": 400.0,
        "gap_run_max": 8, "age_days": 3000.0, "volume_ratio": 1.2,
    })
    perfect = S.score_metrics(clean)
    assert perfect["score"] == 100.0 and perfect["risk_level"] == "low"
    assert perfect["total_penalty"] == 0.0
    assert all(t["penalty"] == 0.0 for t in perfect["terms"])
    assert {t["name"] for t in perfect["terms"]} == set(S.PENALTIES)

    # Every one of the 11 terms must be at its maximum for the sum below to hold,
    # so `volume_ratio` has to be overridden too — inheriting the healthy 1.2
    # from `clean` left `volume_activity` at 0 and the total at 108 instead of
    # sum(PENALTIES) == 113.
    awful = dict(clean, quote_volume=10_000.0, depth_1pct_total=1_000.0,
                 spread_pct=1.0, volatility_30d_annualized=4.0, max_drawdown_90d=95.0,
                 spike_count_30d=50, avg_wick_ratio=0.9, avg_trade_size=1.0,
                 gap_run_max=40, age_days=3.0, volume_ratio=0.1)
    worst = S.score_metrics(awful)
    assert 0.0 <= worst["score"] < 20.0 and worst["risk_level"] == "high"
    assert worst["total_penalty"] == pytest.approx(sum(S.PENALTIES.values()))
    for term in worst["terms"]:
        assert term["penalty"] == S.PENALTIES[term["name"]]


def test_missing_metrics_are_not_penalised_but_lower_completeness():
    """An unknown must not be scored as if it were dangerous *or* safe."""
    empty = S.score_metrics({})
    assert empty["score"] == 100.0
    assert all(t["penalty"] == 0.0 or t["penalty"] is None for t in empty["terms"])

    metrics = {}
    S._add_completeness(metrics)
    assert metrics["metrics_available"] == 0
    assert metrics["data_completeness"] == 0.0
    assert set(metrics["missing_metrics"]) == set(S.SCORED_METRICS)
    assert any(f["code"] == "insufficient_data" for f in S.build_flags(metrics))


def test_scoring_is_monotone_in_a_single_dimension():
    """Worsening one input can never raise the score."""
    base = {k: None for k in S.SCORED_METRICS}
    base.update({"quote_volume": 1e7, "depth_1pct_total": 2e6, "spread_pct": 0.001,
                 "volatility_30d_annualized": 1.0, "max_drawdown_90d": 20.0,
                 "spike_count_30d": 0, "avg_wick_ratio": 0.4, "avg_trade_size": 300.0,
                 "gap_run_max": 6, "age_days": 900.0, "volume_ratio": 1.0})
    baseline = S.score_metrics(base)
    assert baseline["score"] == 100.0, baseline["terms"]
    for worse in ({"spread_pct": 0.05}, {"quote_volume": 1e5},
                  {"volatility_30d_annualized": 2.2}, {"max_drawdown_90d": 45.0},
                  {"spike_count_30d": 5}, {"avg_wick_ratio": 0.75},
                  {"gap_run_max": 16}, {"age_days": 10.0},
                  {"depth_1pct_total": 1e4}, {"avg_trade_size": 5.0},
                  {"volume_ratio": 0.3}):
        result = S.score_metrics(dict(base, **worse))
        assert result["score"] < baseline["score"], worse
        assert result["score"] >= 0.0


# ======================================================================
# flags + evidence
# ======================================================================
def _flags(metrics):
    return {f["code"]: f for f in S.build_flags(metrics)}


def test_low_liquidity_flag_carries_the_numbers():
    flags = _flags({"quote_volume": 42_000.0})
    flag = flags["low_liquidity"]
    assert flag["severity"] == "high"
    ev = flag["evidence"]
    assert ev["quote_volume"] == 42_000.0
    assert ev["info_threshold"] == 2_000_000.0
    assert ev["high_threshold"] == 100_000.0
    assert ev["ratio_to_info_threshold"] == pytest.approx(0.021)
    assert "42,000" in flag["message"]
    assert _flags({"quote_volume": 1_000_000.0})["low_liquidity"]["severity"] == "info"
    assert "low_liquidity" not in _flags({"quote_volume": 5e6})


def test_thin_book_flag_evidence_includes_both_sides():
    flags = _flags({"depth_1pct_total": 90_000.0, "bid_depth_1pct": 40_000.0,
                    "ask_depth_1pct": 50_000.0})
    ev = flags["thin_book"]["evidence"]
    assert ev == {"depth_1pct_total": 90_000.0, "info_threshold": 500_000.0,
                  "warn_threshold": 300_000.0, "high_threshold": 100_000.0,
                  "bid_depth_1pct": 40_000.0, "ask_depth_1pct": 50_000.0}
    assert flags["thin_book"]["severity"] == "high"


def test_wide_spread_flag_evidence():
    flags = _flags({"spread_pct": 0.42, "spread": 0.02, "last": 4.76})
    assert flags["wide_spread"]["severity"] == "warn"
    assert flags["wide_spread"]["evidence"]["spread_pct"] == 0.42
    assert flags["wide_spread"]["evidence"]["info_threshold"] == 0.02
    assert flags["wide_spread"]["evidence"]["high_threshold"] == 0.5
    assert "wide_spread" not in _flags({"spread_pct": 0.008})


def test_volatility_and_drawdown_flags_evidence():
    flags = _flags({"volatility_30d_annualized": 2.39, "volatility_bars": 719,
                    "max_drawdown_90d": 62.5, "drawdown_bars": 90})
    assert flags["high_volatility"]["severity"] == "warn"
    assert flags["high_volatility"]["evidence"] == {
        "volatility_30d_annualized": 2.39, "info_threshold": 1.0,
        "warn_threshold": 2.0, "high_threshold": 3.0, "bars_used": 719}
    assert flags["deep_drawdown"]["severity"] == "warn"
    assert flags["deep_drawdown"]["evidence"]["max_drawdown_90d"] == 62.5
    assert flags["deep_drawdown"]["evidence"]["daily_bars_used"] == 90
    assert _flags({"max_drawdown_90d": 25.0})["deep_drawdown"]["severity"] == "info"


def test_spike_flag_evidence():
    flags = _flags({"spike_count_30d": 6, "spikes_per_100_bars": 0.83, "hourly_bars": 720})
    ev = flags["price_spikes"]["evidence"]
    assert ev["spike_count_30d"] == 6 and ev["threshold_pct"] == S.SPIKE_PCT
    assert flags["price_spikes"]["severity"] == "high"
    assert _flags({"spike_count_30d": 1})["price_spikes"]["severity"] == "info"


def test_long_wick_flag_is_distribution_based_not_single_bar():
    """One extreme wick inside 720 bars must NOT flag (BTCUSDT-class data)."""
    calm = _flags({"avg_wick_ratio": 0.28, "wick_hot_bars_pct": 5.4,
                   "max_wick_ratio": 0.85, "hourly_bars": 720})
    assert "long_wick" not in calm, "a single 0.85 wick bar is not a signal"

    hot = _flags({"avg_wick_ratio": 0.32, "wick_hot_bars_pct": 16.5,
                  "wick_over_threshold_bars": 119, "max_wick_ratio": 1.0,
                  "hourly_bars": 720})
    assert hot["long_wick"]["severity"] == "high"
    assert hot["long_wick"]["evidence"]["wick_hot_bars_pct"] == 16.5
    assert hot["long_wick"]["evidence"]["wick_over_threshold_bars"] == 119


def test_trade_size_flag_reports_both_directions():
    tiny = _flags({"avg_trade_size": 3.0, "trade_count": 500_000, "quote_volume": 1.5e6})
    assert "碎片化" in tiny["trade_size_anomaly"]["message"]
    assert tiny["trade_size_anomaly"]["evidence"]["avg_trade_size"] == 3.0

    huge = _flags({"avg_trade_size": 10_655.0, "trade_count": 5700, "quote_volume": 6.1e7})
    assert "集中度代理偏高" in huge["trade_size_anomaly"]["message"]
    assert huge["trade_size_anomaly"]["evidence"]["huge_threshold"] == S.TRADE_SIZE_HUGE

    assert "trade_size_anomaly" not in _flags({"avg_trade_size": 300.0})


def test_gap_run_flag_only_for_long_runs():
    assert "price_gap_run" not in _flags({"gap_run_max": 8})
    assert _flags({"gap_run_max": 10})["price_gap_run"]["severity"] == "info"
    assert _flags({"gap_run_max": 13})["price_gap_run"]["severity"] == "warn"
    high = _flags({"gap_run_max": 17})
    assert high["price_gap_run"]["severity"] == "high"
    assert high["price_gap_run"]["evidence"]["high_threshold"] == S.GAP_RUN_DANGER


def test_new_listing_flag_evidence():
    flags = _flags({"age_days": 12.0, "listing_date": "2026-09-17", "listing_time_ms": 1758})
    ev = flags["new_listing"]["evidence"]
    assert ev["age_days"] == 12.0 and ev["threshold_days"] == 30
    assert ev["listing_date"] == "2026-09-17"
    assert "new_listing" not in _flags({"age_days": 400.0})


def test_volume_activity_drop_flag_evidence():
    flags = _flags({"volume_ratio": 0.22, "quote_volume": 3.2e5, "bar_quote_median": 6e4})
    assert flags["volume_activity_drop"]["severity"] == "warn"
    assert flags["volume_activity_drop"]["evidence"]["volume_ratio"] == 0.22
    assert "volume_activity_drop" not in _flags({"volume_ratio": 2.1})


def test_every_flag_has_code_label_message_and_evidence():
    metrics = {"quote_volume": 1_000.0, "depth_1pct_total": 1_000.0, "spread_pct": 1.0,
               "volatility_30d_annualized": 5.0, "max_drawdown_90d": 99.0,
               "spike_count_30d": 9, "avg_wick_ratio": 0.9, "wick_hot_bars_pct": 30.0,
               "avg_trade_size": 1.0, "gap_run_max": 30, "age_days": 1.0,
               "volume_ratio": 0.1, "fetch_errors": ["depth: timeout"]}
    S._add_completeness(metrics)
    flags = S.build_flags(metrics)
    assert len(flags) >= 11
    for flag in flags:
        assert flag["code"] and flag["label"] and flag["message"]
        assert flag["severity"] in ("info", "warn", "high")
        assert flag["evidence"], f"{flag['code']} carries no evidence"


# ======================================================================
# screen: sampling, limit clamping, cache
# ======================================================================
def _many_symbols(n):
    return [f"SYM{i:03d}USDT" for i in range(n)]


def test_limit_is_clamped_to_1_200():
    assert S._clamp_limit(0) == S.DEFAULT_LIMIT
    assert S._clamp_limit(-5) == S.DEFAULT_LIMIT
    assert S._clamp_limit("junk") == S.DEFAULT_LIMIT
    assert S._clamp_limit(10) == 10
    assert S._clamp_limit(500) == S.MAX_LIMIT


def test_screen_samples_top_n_by_quote_volume():
    symbols = _many_symbols(8)
    exchange = FakeExchange(symbols)
    for i, sym in enumerate(symbols):
        exchange.per_symbol[sym] = {"quote_volume": 1_000_000.0 * (i + 1)}
    screener = make_screener(exchange)

    payload = run(screener.screen(limit=3, min_quote_volume=1.0, sort="volume"))
    assert payload["sampled"] == 3
    assert payload["universe_usdt"] == 8
    # Highest volume first: the top 3 by quote volume, not the first 3 listed.
    assert [r["symbol"] for r in payload["results"]] == [
        "SYM007USDT", "SYM006USDT", "SYM005USDT"]
    # Only the sampled symbols were enriched (depth per sampled symbol + 1 ticker).
    depth_calls = [c for c in exchange.calls if c[0] == "/api/v3/depth"]
    assert {c[1]["symbol"] for c in depth_calls} == {"SYM007USDT", "SYM006USDT", "SYM005USDT"}


def test_screen_applies_min_quote_volume():
    symbols = ["BIGUSDT", "SMALLUSDT"]
    exchange = FakeExchange(symbols)
    exchange.per_symbol["BIGUSDT"] = {"quote_volume": 50_000_000.0}
    exchange.per_symbol["SMALLUSDT"] = {"quote_volume": 10_000.0}
    payload = run(make_screener(exchange).screen(limit=50, min_quote_volume=1_000_000))
    assert [r["symbol"] for r in payload["results"]] == ["BIGUSDT"]
    assert payload["universe_usdt"] == 1


def test_screen_caches_for_the_ttl_and_refetches_after_it():
    clock = {"t": 1000.0}
    exchange = FakeExchange(["AAAUSDT"])
    screener = make_screener(exchange, cache_ttl=300.0, now=lambda: clock["t"])

    first = run(screener.screen(limit=5, min_quote_volume=1.0))
    assert first["cached"] is False
    calls_after_first = len(exchange.calls)

    second = run(screener.screen(limit=5, min_quote_volume=1.0))
    assert second["cached"] is True
    assert len(exchange.calls) == calls_after_first, "a cache hit must not hit the network"
    assert second["results"] == first["results"]

    clock["t"] += 299.0
    assert run(screener.screen(limit=5, min_quote_volume=1.0))["cached"] is True

    clock["t"] += 2.0                        # now past the 300s TTL
    assert run(screener.screen(limit=5, min_quote_volume=1.0))["cached"] is False
    assert len(exchange.calls) > calls_after_first
    assert screener.stats["cache_hits"] == 2


def test_cache_key_covers_every_input_that_changes_the_answer():
    exchange = FakeExchange(["AAAUSDT"])
    screener = make_screener(exchange)
    run(screener.screen(limit=5, min_quote_volume=1.0, sort="score", interval="1h"))
    calls = len(exchange.calls)
    assert run(screener.screen(limit=6, min_quote_volume=1.0, sort="score",
                               interval="1h"))["cached"] is False
    assert run(screener.screen(limit=6, min_quote_volume=1.0, sort="volume",
                               interval="1h"))["cached"] is False
    assert run(screener.screen(limit=6, min_quote_volume=1.0, sort="volume",
                               interval="4h"))["cached"] is False
    assert len(exchange.calls) > calls


def test_screen_sorts_by_score_when_asked():
    symbols = ["CLEANUSDT", "THINUSDT"]
    exchange = FakeExchange(symbols, depth_qty=(5000.0, 0.0), spread_bps=0.2)
    exchange.per_symbol = {
        "CLEANUSDT": {"quote_volume": 5e8, "volatility": 0.001, "age_days": 900},
        "THINUSDT": {"quote_volume": 1.2e5, "volatility": 0.001, "age_days": 900},
    }
    screener = make_screener(exchange)
    by_score = run(screener.screen(limit=10, min_quote_volume=1.0, sort="score"))
    scores = [r["overall_score"] for r in by_score["results"]]
    assert scores == sorted(scores), "sort=score must be ascending (worst first)"
    assert by_score["results"][0]["symbol"] == "THINUSDT"
    assert by_score["results"][0]["overall_score"] < by_score["results"][1]["overall_score"]


# ======================================================================
# screen: real-shaped scenarios per flag
# ======================================================================
def _screen_one(cfg):
    exchange = FakeExchange(["ONEUSDT"], depth_qty=(3000.0, 0.0), spread_bps=0.3)
    exchange.per_symbol["ONEUSDT"] = cfg
    payload = run(make_screener(exchange).screen(limit=1, min_quote_volume=1.0))
    return payload["results"][0]


def test_clean_large_cap_scores_low_with_no_flags():
    row = _screen_one({"quote_volume": 1.2e9, "count": 3_000_000, "last": 84000.0,
                       "volatility": 0.002, "age_days": 3300, "wick_ratio": 0.2})
    assert row["risk_level"] == "low", row["flags"]
    assert row["flags"] == []
    m = row["metrics"]
    assert m["data_completeness"] == 1.0
    assert m["spread_pct"] is not None and m["spread_pct"] < 0.01
    assert m["volatility_30d_annualized"] < 0.6
    assert m["max_drawdown_90d"] is not None
    assert m["listing_date"] and m["age_days"] > 3000
    assert m["avg_trade_size"] == pytest.approx(400.0)
    assert m["depth_1pct_total"] > 1e6


def test_tiny_illiquid_pair_flags_liquidity_depth_and_spread():
    exchange = FakeExchange(["TINYUSDT"], depth_qty=(5.0, 5.0), spread_bps=8.0)
    exchange.per_symbol["TINYUSDT"] = {"quote_volume": 30_000.0, "count": 500,
                                      "last": 0.5, "age_days": 900}
    row = run(make_screener(exchange).screen(
        limit=1, min_quote_volume=1.0))["results"][0]
    codes = {f["code"] for f in row["flag_details"]}
    assert {"low_liquidity", "thin_book"} <= codes, codes
    assert row["metrics"]["quote_volume"] == 30_000.0
    assert row["risk_level"] in ("medium", "high")
    assert row["overall_score"] < 60


def test_single_spike_bar_is_detected():
    row = _screen_one({"spikes": True, "quote_volume": 5e8, "volatility": 0.001,
                       "age_days": 900})
    assert any("单根涨跌" in f for f in row["flags"])
    codes = {f["code"] for f in row["flag_details"]}
    assert "price_spikes" in codes
    assert row["metrics"]["spike_count_30d"] >= 1


def test_monotone_series_produces_a_gap_run():
    row = _screen_one({"gap": True, "quote_volume": 5e8, "age_days": 900})
    codes = {f["code"] for f in row["flag_details"]}
    assert "price_gap_run" in codes
    assert row["metrics"]["gap_run_max"] >= S.GAP_RUN_DANGER


def test_crash_scenario_max_drawdown_is_measured():
    row = _screen_one({"daily_mode": "crash", "quote_volume": 5e8, "age_days": 900})
    assert row["metrics"]["max_drawdown_90d"] == pytest.approx(60.0, abs=1.0)
    codes = {f["code"] for f in row["flag_details"]}
    assert "deep_drawdown" in codes


def test_new_listing_is_flagged_with_its_listing_date():
    exchange = FakeExchange(["NEWUSDT"], depth_qty=(200_000.0, 200_000.0),
                            spread_bps=0.05)
    exchange.per_symbol["NEWUSDT"] = {"age_days": 5, "quote_volume": 5e8,
                                     "count": 1_000_000, "last": 100.0,
                                     "volatility": 0.002}
    row = run(make_screener(exchange).screen(
        limit=1, min_quote_volume=1.0))["results"][0]
    codes = {f["code"] for f in row["flag_details"]}
    assert "new_listing" in codes
    assert row["metrics"]["age_days"] == pytest.approx(5.0, abs=0.5)
    # A brand-new listing loses the age term and is never reported as "low risk".
    age_penalty = [t for t in row["score_breakdown"]["terms"]
                   if t["name"] == "age"][0]["penalty"]
    assert age_penalty == S.PENALTIES["age"]
    assert row["overall_score"] == pytest.approx(100.0 - S.PENALTIES["age"], abs=0.01)
    assert row["risk_level"] == "medium"


def test_new_listing_verdict_is_floored_but_only_below_the_age_cutoff():
    """92 points on a 5-day-old pair must not be presented as "low risk".

    The floor is a verdict rule, not a score adjustment: the penalty breakdown
    still has to explain the 92 (see the test above), and a pair at/over
    ``AGE_NEW_DAYS`` keeps the plain score → level mapping.
    """
    metrics = {k: None for k in S.SCORED_METRICS}
    metrics.update({
        "quote_volume": 1e10, "depth_1pct_total": 5e7, "spread_pct": 1e-5,
        "volatility_30d_annualized": 0.4, "max_drawdown_90d": 5.0,
        "spike_count_30d": 0, "avg_wick_ratio": 0.28, "avg_trade_size": 400.0,
        "gap_run_max": 8, "age_days": 5.0, "volume_ratio": 1.2,
    })
    fresh = S.score_metrics(metrics)
    assert fresh["score"] == 100.0 - S.PENALTIES["age"]
    assert fresh["risk_level"] == "medium"

    metrics["age_days"] = float(S.AGE_NEW_DAYS)      # exactly at the cutoff
    seasoned = S.score_metrics(metrics)
    assert seasoned["score"] == 100.0
    assert seasoned["risk_level"] == "low"


# ======================================================================
# graceful degradation
# ======================================================================
def test_symbol_level_failure_degrades_without_failing_the_screen():
    exchange = FakeExchange(["GOODUSDT", "BADUSDT"])
    exchange.raise_on = {"BADUSDT"}
    payload = run(make_screener(exchange).screen(limit=10, min_quote_volume=1.0))
    assert payload["count"] == 2, "one bad symbol must not drop the others"
    bad = [r for r in payload["results"] if r["symbol"] == "BADUSDT"][0]
    assert bad["metrics"]["fetch_errors"]
    assert bad["metrics"]["data_completeness"] < 1.0
    assert any(f["code"] == "fetch_error" for f in bad["flag_details"])
    assert [f for f in payload["failures"] if f["symbol"] == "BADUSDT"]


def test_ticker_failure_raises_a_structured_upstream_error():
    exchange = FakeExchange(["AAAUSDT"])
    exchange.fail_paths["/api/v3/ticker/24hr"] = RuntimeError("host unreachable")
    with pytest.raises(S.UpstreamError) as exc:
        run(make_screener(exchange).screen(limit=5, min_quote_volume=1.0))
    assert "host unreachable" in str(exc.value)


def test_empty_ticker_payload_is_an_error():
    async def empty_fetch(path, params):
        return []
    with pytest.raises(S.UpstreamError):
        run(make_screener(empty_fetch).screen(limit=5, min_quote_volume=1.0))


def test_unexpected_ticker_shape_is_an_error():
    async def bad_fetch(path, params):
        return {"unexpected": "dict"}
    with pytest.raises(S.UpstreamError):
        run(make_screener(bad_fetch).screen(limit=5, min_quote_volume=1.0))


def test_screen_budget_is_enforced_and_reported():
    """Once the wall-clock budget is gone, symbols degrade instead of hanging."""
    clock = {"t": 0.0}

    def now():
        # Each observation must already be past the 20s budget: the screener
        # checks the budget *before* each request, so a step that only passes it
        # every third call would (correctly) let the first requests through —
        # one request issued at 7s elapsed is inside the budget, not after it.
        clock["t"] += 30.0
        return clock["t"]

    exchange = FakeExchange(["AAAUSDT", "BBBUSDT"])
    screener = make_screener(exchange, budget=20.0, now=now)
    payload = run(screener.screen(limit=5, min_quote_volume=1.0))
    assert payload["count"] == 2
    for row in payload["results"]:
        assert any("budget" in e for e in row["metrics"]["fetch_errors"])
        assert row["overall_score"] is not None
    assert not [c for c in exchange.calls if c[0] == "/api/v3/depth"], \
        "no enrichment request may be issued after the budget is exhausted"


def test_screener_never_raises_on_garbage_kline_payload():
    async def garbage(path, params):
        if path == "/api/v3/ticker/24hr":
            return [{"symbol": "AAAUSDT", "lastPrice": "1", "quoteVolume": "5000000",
                     "count": 100}]
        return "not-a-list"
    payload = run(make_screener(garbage).screen(limit=5, min_quote_volume=1.0))
    row = payload["results"][0]
    assert row["metrics"]["volatility_30d_annualized"] is None
    assert row["metrics"]["spread_pct"] is None
    assert row["overall_score"] == 100.0
    assert any(f["code"] == "insufficient_data" for f in row["flag_details"])


def test_depth_metrics_ignore_inverted_or_empty_books():
    assert S._depth_metrics({"bids": [], "asks": []}) == {}
    assert S._depth_metrics({"bids": [["x", "y"]], "asks": [["1", "1"]]}) == {}
    assert S._depth_metrics(None) == {}


def test_upstream_error_message_is_captured():
    async def failing(path, params):
        raise S.UpstreamError("HTTP 451 from /api/v3/klines: blocked")
    with pytest.raises(S.UpstreamError):
        run(make_screener(failing).screen(limit=1, min_quote_volume=1.0))


def test_detail_returns_flag_details_with_evidence():
    exchange = FakeExchange(["AAAUSDT"], depth_qty=(1.0, 1.0), spread_bps=0.1)
    exchange.per_symbol["AAAUSDT"] = {"quote_volume": 20_000.0, "count": 5000,
                                     "age_days": 10}
    detail = run(make_screener(exchange).detail("AAAUSDT"))
    codes = {f["code"] for f in detail["flag_details"]}
    assert {"low_liquidity", "new_listing"} <= codes
    for flag in detail["flag_details"]:
        assert flag["evidence"]
    assert detail["flags"], "the short message list mirrors the detail flags"
    assert detail["score_breakdown"]["score"] == detail["overall_score"]


def test_default_screener_uses_the_configured_host():
    class Cfg:
        market_data_host = "https://example.invalid"
    S.reset_default_screener()
    try:
        assert S.get_screener(Cfg()).host == "https://example.invalid"
        assert S.get_screener(None).host == S.DEFAULT_HOST   # cached instance reused
        S.reset_default_screener()
        assert S.get_screener(object()).host == S.DEFAULT_HOST
    finally:
        S.reset_default_screener()


# ======================================================================
# web API (temp DB, stubbed fetch, no network)
# ======================================================================
VIEWER = ("audit_viewer", "V1ewerPass!")


@pytest.fixture(scope="module")
def web_app():
    from app.config import Config
    from app.event_bus import EventBus
    from core.auth.auth import AuthManager
    from db.database import init_database

    tmpdir = Path(tempfile.mkdtemp(prefix="bt_audit_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "audit.db")
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")

    async def _setup():
        await init_database(config.db_path)
        am = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        await am.create_user(VIEWER[0], VIEWER[1], "viewer", VIEWER[0])
        return am

    auth = run(_setup())

    from web.server import create_app
    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    yield app
    matcher = getattr(app.state, "limit_order_matcher", None)
    if matcher is not None:
        run(matcher.stop())


@pytest.fixture()
def client(web_app):
    c = TestClient(web_app)
    r = c.post("/api/auth/login", json={"username": VIEWER[0], "password": VIEWER[1]})
    assert r.status_code == 200, r.text
    return c


@pytest.fixture()
def stub(monkeypatch):
    """Point the process-wide screener at a stubbed exchange for each web test."""
    exchange = FakeExchange(["AAAUSDT", "BBBUSDT"], depth_qty=(4000.0, 4000.0),
                            spread_bps=0.2)
    exchange.per_symbol = {
        "AAAUSDT": {"quote_volume": 9e8, "count": 1_000_000, "age_days": 3000},
        "BBBUSDT": {"quote_volume": 2e5, "count": 800, "age_days": 12},
    }
    screener = make_screener(exchange)
    S.reset_default_screener()
    monkeypatch.setattr(S, "get_screener", lambda config=None: screener)
    monkeypatch.setattr("web.routes.audit.get_screener", lambda config=None: screener,
                        raising=False)
    yield exchange
    S.reset_default_screener()


def test_screen_endpoint_shape(client, stub):
    r = client.get("/api/audit/screen", params={"limit": 10, "min_quote_volume": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["disclaimer"] and "非链上合约审计" in body["disclaimer"]
    assert body["heuristic"] is True and body["on_chain_audit"] is False
    assert isinstance(body["updated_at"], int)
    assert body["params"]["limit"] == 10
    assert body["sampled"] == 2 and body["count"] == 2
    assert body["method"]["dimensions"] and body["method"]["not_covered"]
    row = body["results"][0]
    assert set(row) >= {"symbol", "overall_score", "risk_level", "flags", "metrics"}
    assert row["risk_level"] in ("low", "medium", "high")
    assert isinstance(row["flags"], list)
    assert row["metrics"]["quote_volume"] is not None


def test_screen_endpoint_rejects_bad_sort_and_interval(client, stub):
    assert client.get("/api/audit/screen", params={"sort": "nope"}).status_code == 400
    assert client.get("/api/audit/screen", params={"interval": "3m"}).status_code == 400
    body = client.get("/api/audit/screen", params={"sort": "nope"}).json()
    assert "error" in body and "Traceback" not in body["error"]


def test_detail_endpoint_carries_evidence_and_disclaimer(client, stub):
    r = client.get("/api/audit/BBBUSDT")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "BBBUSDT"
    assert body["disclaimer"] and body["on_chain_audit"] is False
    assert body["metrics"]["quote_volume"] == 2e5
    assert body["score_breakdown"]["terms"]
    assert body["flags"] and all(f["evidence"] for f in body["flags"])
    codes = {f["code"] for f in body["flags"]}
    assert "low_liquidity" in codes


def test_detail_endpoint_is_case_insensitive_and_validates(client, stub):
    assert client.get("/api/audit/aaausdt").json()["symbol"] == "AAAUSDT"
    assert client.get("/api/audit/not a symbol").status_code == 400
    assert client.get("/api/audit/screen").status_code == 200      # literal path wins


def test_detail_endpoint_404s_for_an_unknown_symbol(client, stub, monkeypatch):
    empty = FakeExchange(["AAAUSDT"], default_cfg={"quote_volume": 0.0, "count": None})
    screener = make_screener(empty)
    monkeypatch.setattr(S, "get_screener", lambda config=None: screener)
    monkeypatch.setattr("web.routes.audit.get_screener", lambda config=None: screener,
                        raising=False)
    r = client.get("/api/audit/ZZZUSDT")
    assert r.status_code == 404
    assert "error" in r.json()


def test_endpoints_return_structured_errors_when_the_host_is_down(client, monkeypatch):
    async def boom(path, params):
        raise S.UpstreamError("Network is unreachable")

    screener = make_screener(boom)
    monkeypatch.setattr(S, "get_screener", lambda config=None: screener)
    monkeypatch.setattr("web.routes.audit.get_screener", lambda config=None: screener,
                        raising=False)
    r = client.get("/api/audit/screen", params={"limit": 3, "min_quote_volume": 1})
    assert r.status_code == 502
    body = r.json()
    assert set(body) == {"error"}
    assert "Network is unreachable" in body["error"]
    assert "Traceback" not in r.text


def test_audit_endpoints_require_login(web_app):
    c = TestClient(web_app)
    assert c.get("/api/audit/screen").status_code == 401
    assert c.get("/api/audit/BTCUSDT").status_code == 401


def test_audit_page_renders_with_disclaimer_and_table(client):
    r = client.get("/audit")
    assert r.status_code == 200, r.text
    html = r.text
    assert "启发式" in html and "非链上合约审计" in html
    assert 'id="audit-tbody"' in html and 'id="audit-table-wrap"' in html
    assert "/api/audit/screen" in html and "/api/audit/" in html
    assert 'id="btn-refresh"' in html and "重新检测" in html
    assert 'id="drawer"' in html and 'id="audit-error"' in html and 'id="audit-empty"' in html
