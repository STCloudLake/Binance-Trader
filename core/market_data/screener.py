"""Heuristic token screen over Binance **public** market data.

WHAT THIS IS
------------
A screening tool built exclusively from the exchange's public REST feed
(``data-api.binance.vision``): 24h ticker, order-book depth and klines.  It
describes what the *market* looks like — how thick the book is, how volatile
the price is, how old the listing is.  It is the same class of information a
trader can read off a chart and an order book by hand, just computed
consistently for many symbols at once.

WHAT THIS IS NOT
----------------
It is **not** an on-chain contract audit.  Nothing here reads a token contract,
a holder distribution, a mint authority, a transfer tax, a proxy upgrade slot or
a liquidity-pool lock.  This code therefore CANNOT detect honeypots, rug pulls,
freeze/blacklist functions, hidden mints, or any other on-chain trap, and it
never claims to.  ``DISCLAIMER`` is returned by both API endpoints and rendered
prominently on ``/audit`` on purpose: a clean score here means "the exchange
market for this pair looks ordinary", not "this token is safe".

Design notes
------------
* **No pandas / no numpy.**  The estimators are plain Python so the scoring math
  is auditable and the tests need no heavy fixtures.
* **Bounded concurrency.**  A screen of N symbols issues ~3 requests per symbol;
  a semaphore (default 6) keeps the crawler from opening hundreds of sockets,
  and every request carries its own timeout plus a global wall-clock budget.
* **Cache.**  Screening is expensive (~minutes of upstream latency for a wide
  screen), so results are cached in-process for ``CACHE_TTL`` seconds keyed by
  every input that changes the answer.
* **Every flag carries its evidence.**  ``flags`` in the detail payload are
  ``{code, label, severity, message, evidence}`` where ``evidence`` holds the
  raw numbers behind the decision — the UI prints them, so a user can disagree
  with the threshold instead of having to trust it.
* **Missing data is ``None``, never a fabricated number.**  A new listing with
  three candles has no 90d drawdown; the field stays ``null`` and the score
  simply does not apply that term.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict
from statistics import median
from typing import Any, Awaitable, Callable, Iterable, Optional

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

#: Public mainnet market-data mirror (api.binance.com is unreachable here).
DEFAULT_HOST = "https://data-api.binance.vision"

#: In-process screen cache TTL: 5 minutes (contract §4 wants repeat loads cheap).
CACHE_TTL = 300.0

#: Concurrent upstream requests for one screen.
DEFAULT_CONCURRENCY = 6

#: Per-request and whole-screen time budgets (seconds).  The all-symbol 24h
#: ticker is one large response that the host serves in ~8–13s (measured), so it
#: gets its own, longer budget than a single symbol's klines/depth.
REQUEST_TIMEOUT = 12.0
TICKER_TIMEOUT = 25.0
SCREEN_BUDGET = 45.0
#: One retry for transient upstream failures (a single stalled socket must not
#: blank out a whole screen).
REQUEST_ATTEMPTS = 2
RETRY_DELAY = 0.4

#: Sampling bounds for the screen endpoint.
DEFAULT_LIMIT = 50
MAX_LIMIT = 200

#: Bounds on the screen/detail caches.  The detail key carries a caller-supplied
#: symbol, so without an eviction policy 50 distinct `/api/audit/{symbol}` calls
#: left 52 entries for ever and the map was caller-controlled.  The working set is
#: ``ladder × limits × sorts × intervals`` for the screen plus the symbols an
#: operator actually looked at, so these are far above any legitimate use.
MAX_CACHE_ENTRIES = 256
MAX_CACHE_LOCKS = 128

#: Neutral defaults for optional inputs.
DEFAULT_MIN_QUOTE_VOLUME = 100_000.0
#: Allowed liquidity thresholds.  ``min_quote_volume`` is caller-controlled and
#: part of the cache key, so accepting an arbitrary float let a loop mint one
#: cache entry — and pay for one full upstream crawl (ticker + depth + klines
#: per sampled symbol, hundreds of requests) — per value.  Requests are
#: floor-snapped to this ladder, which bounds the number of distinct screens to
#: ``len(ladder) * limits * sorts * intervals`` while keeping every documented
#: preset (0 / 10k / 100k / 1M / 10M / 100M) exact.
MIN_QUOTE_VOLUME_LADDER = (0.0, 10_000.0, 100_000.0, 1_000_000.0,
                           10_000_000.0, 100_000_000.0)
USDT = "USDT"

#: Annualisation constant for hourly log returns (24*365 bars per year).
HOURS_PER_YEAR = 24 * 365

#: Spike / wick / gap thresholds (documented so the UI can show them).
#:
#: These numbers are *calibrated against a measured 25-symbol mainnet sample*
#: (see the calibration table in the hand-off notes), not invented: the observed
#: distributions were max-wick-ratio 0.23–0.35, bars with a wick >0.7 3–16%,
#: longest same-direction hourly run 4–12, 90d drawdown 0.1–34%, 30d annualised
#: volatility 0.4–2.4.  A rule that "looks reasonable" but fires on BTCUSDT is a
#: false positive, so each threshold sits above the ordinary range.
SPIKE_PCT = 15.0          # |1-bar return| above this counts as a spike
SPIKE_COUNT_DANGER = 5    # 5+ spikes over 30 days is flagged
#: A single long-wick candle inside 720 hourly bars is normal, so the wick flag
#: keys off the distribution (average wick ratio, or the share of bars whose
#: wick exceeds ``WICK_RATIO_DANGER``) and never off one bar's maximum — the max
#: is only reported as supporting evidence.
WICK_AVG_WARN = 0.50
WICK_AVG_DANGER = 0.70
WICK_RATIO_DANGER = 0.70
WICK_HOT_BARS_PCT_INFO = 6.0
WICK_HOT_BARS_PCT_WARN = 10.0
WICK_HOT_BARS_PCT_DANGER = 15.0
GAP_RUN_INFO = 10         # 10+ consecutive same-direction hourly bars
GAP_RUN_WARN = 13
GAP_RUN_DANGER = 16       # sustained one-way drift: genuinely unusual
AGE_NEW_DAYS = 30         # <30 days listed → "新上市"
TRADE_SIZE_TINY = 50.0
TRADE_SIZE_HUGE = 10_000.0
MIN_CANDLES = 30          # below this the series is too short to score

#: Score → risk level.  Calibrated against the measured 25-symbol mainnet
#: sample: a clean large-cap (BTCUSDT/ETHUSDT) scores ~45–55, a token with one
#: dangerous dimension (thin book / 2.4x volatility) ~35–40.  A "medium" verdict
#: is the normal state of a crypto pair — it is *not* a warning by itself.
RISK_LOW_MIN = 60
RISK_MEDIUM_MIN = 35

#: Flag penalties (max points each term may remove from 100).  ``None`` tiers
#: mean "no data → no penalty" (we do not punish an unknown).
PENALTIES = {
    "liquidity": 25.0,
    "depth": 15.0,
    "spread": 10.0,
    "volatility": 15.0,
    "drawdown": 10.0,
    "spike": 10.0,
    "wick": 5.0,
    "trade_size": 5.0,
    "gap": 5.0,
    "age": 8.0,
    "volume_activity": 5.0,
}

DISCLAIMER = (
    "启发式筛查，非链上合约审计。本页仅用币安公开行情数据（24h 成交额、盘口深度、"
    "价差、K 线波动与上市时间）做统计推断，不读取链上合约、持仓分布、权限或资金池，"
    "因此无法识别蜜罐、貔貅盘、跑路、隐藏增发或黑名单等链上风险。分数低只代表"
    "“该交易所市场数据看起来异常”，分数高也不代表代币安全。"
)

#: Penalty bands.  Each entry is ``(predicate, fraction)`` and is consulted in
#: order by :func:`_band_fraction`; the first entry whose predicate holds wins.
#: A ``None`` predicate is the catch-all.  Predicates are *explicit* on purpose:
#: an earlier version used a bare ``value < threshold`` test with hand-ordered
#: tables, which is ambiguous — ``spread_pct < 0.5`` is true for *both* the
#: healthiest and the worst measurement — and it applied the maximum penalty to
#: the best values (a $1B book scored as if it were empty).
#:
#: Thresholds are calibrated against a measured 25-symbol mainnet sample; see
#: the module docstring.  ``test_band_tables_bound_every_value`` pins the shape.
def _lt(threshold: float) -> Callable[[float], bool]:
    return lambda value: value < threshold


def _gt(threshold: float) -> Callable[[float], bool]:
    return lambda value: value > threshold


BANDS_QUOTE_VOLUME = (
    (_lt(1e5), 1.0), (_lt(5e5), 0.6), (_lt(2e6), 0.25), (None, 0.0))
BANDS_DEPTH = (
    (_lt(1e5), 1.0), (_lt(3e5), 0.6), (_lt(5e5), 0.25), (None, 0.0))
BANDS_SPREAD = (
    (_gt(0.5), 1.0), (_gt(0.1), 0.6), (_gt(0.02), 0.25), (None, 0.0))
BANDS_VOLATILITY = (
    (_gt(3.0), 1.0), (_gt(2.5), 0.6), (_gt(2.0), 0.25), (None, 0.0))
BANDS_DRAWDOWN = (
    (_gt(80.0), 1.0), (_gt(60.0), 0.6), (_gt(40.0), 0.25), (None, 0.0))
BANDS_WICK = (
    (_gt(0.70), 1.0), (_gt(0.60), 0.6), (_gt(0.50), 0.25), (None, 0.0))
#: A meaningful average trade size sits *inside* a range: both ends are odd.
BANDS_TRADE_SIZE = (
    (_lt(10.0), 1.0), (_lt(50.0), 0.6), (None, 0.0))
BANDS_VOLUME_RATIO = (
    (_lt(0.25), 1.0), (_lt(0.5), 0.6), (_lt(0.8), 0.25), (None, 0.0))
#: spike_count_30d / SPIKE_COUNT_DANGER and gap_run_max / GAP_RUN_DANGER.
BANDS_SPIKE_RATE = (
    (_gt(1.0), 1.0), (_gt(0.8), 0.6), (_gt(0.4), 0.25), (None, 0.0))
BANDS_GAP_RATE = (
    (_gt(1.0), 1.0), (_gt(0.8), 0.6), (_gt(0.6), 0.25), (None, 0.0))
#: Under 30 days listed loses the whole age term (a yes/no rule, not a gradient).
BANDS_AGE = ((_lt(AGE_NEW_DAYS), 1.0), (None, 0.0))

#: Per-dimension aliases (the names this module documents).
LIQUIDITY_BANDS = BANDS_QUOTE_VOLUME
DEPTH_BANDS = BANDS_DEPTH
SPREAD_BANDS = BANDS_SPREAD
VOLATILITY_BANDS = BANDS_VOLATILITY
DRAWDOWN_BANDS = BANDS_DRAWDOWN
WICK_BANDS = BANDS_WICK
TRADE_SIZE_BANDS = BANDS_TRADE_SIZE
VOLUME_ACTIVITY_BANDS = BANDS_VOLUME_RATIO
SPIKE_BANDS = BANDS_SPIKE_RATE
GAP_BANDS = BANDS_GAP_RATE
AGE_BANDS = BANDS_AGE

#: Flag severities.
SEV_HIGH = "high"
SEV_WARN = "warn"
SEV_INFO = "info"


# --------------------------------------------------------------------------
# tiny numeric helpers (all None-tolerant)
# --------------------------------------------------------------------------
def _f(value: Any) -> Optional[float]:
    """Best-effort float; ``None`` when not numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _round(value: Any, digits: int) -> Optional[float]:
    num = _f(value)
    return None if num is None else round(num, digits)


def _median(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    return float(median(clean))


def _band_fraction(value: Optional[float], bands) -> Optional[float]:
    """Fraction of the penalty to apply for ``value`` under ``bands``.

    ``bands`` is a sequence of ``(predicate, fraction)`` consulted in order; the
    first entry whose predicate holds wins, and a ``None`` predicate is the
    catch-all.  A ``None`` value returns ``None`` — an unknown measurement
    carries no penalty (it surfaces through ``data_completeness`` instead).
    """
    if value is None:
        return None
    for predicate, fraction in bands:
        if predicate is None or predicate(value):
            return fraction
    return 0.0


def _klines(raw: Any) -> list[list[float]]:
    """Normalise Binance kline rows to ``[open, high, low, close, volume, close_ms]``.

    Binance mixes strings and ints across endpoints and silently drops malformed
    rows rather than raising: a truncated final candle must not fail a screen.
    """
    out: list[list[float]] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            continue
        o, h, l, c, v = (_f(row[1]), _f(row[2]), _f(row[3]), _f(row[4]), _f(row[5]))
        if None in (o, h, l, c):
            continue
        close_ms = _f(row[6])
        out.append([o, h, l, c, v if v is not None else 0.0,
                    close_ms if close_ms is not None else 0.0])
    return out


def _returns(closes: list[float]) -> list[float]:
    out = []
    for prev, cur in zip(closes, closes[1:]):
        if prev and prev > 0:
            out.append((cur - prev) / prev)
    return out


def _stdev(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def annualised_volatility(closes: Iterable[float], bars_per_year: float) -> Optional[float]:
    """Annualised stdev of log returns (sample stdev, ``bars_per_year`` scaling)."""
    series = [c for c in (_f(c) for c in closes) if c is not None and c > 0]
    if len(series) < 3:
        return None
    logs = [math.log(cur / prev) for prev, cur in zip(series, series[1:]) if prev > 0 and cur > 0]
    sd = _stdev(logs)
    if sd is None:
        return None
    return sd * math.sqrt(bars_per_year)


def max_drawdown(closes: Iterable[float]) -> Optional[float]:
    """Peak-to-trough max drawdown in percent (positive number = loss)."""
    series = [c for c in (_f(c) for c in closes) if c is not None and c > 0]
    if len(series) < 2:
        return None
    peak = series[0]
    worst = 0.0
    for price in series:
        peak = max(peak, price)
        if peak > 0:
            worst = max(worst, (peak - price) / peak)
    return worst * 100.0


def spike_count(closes: Iterable[float], threshold_pct: float = SPIKE_PCT) -> int:
    """Number of bars whose close-to-close return exceeds ``threshold_pct``."""
    limit = threshold_pct / 100.0
    return sum(1 for r in _returns([c for c in (_f(c) for c in closes) if c is not None])
               if abs(r) > limit)


def _wick_metrics(series: list[list[float]]) -> tuple[Optional[float], Optional[int], Optional[float]]:
    """(max_upper_wick_ratio, count over threshold, average wick ratio).

    Ratio = wick length / candle range; candles with a zero range are skipped
    (a flat candle carries no wick information at all).
    """
    ratios: list[float] = []
    for _o, high, low, _c, _v, _t in series:
        rng = high - low
        if rng <= 0:
            continue
        ratios.append((high - max(_o, _c)) / rng)
    if not ratios:
        return None, None, None
    return max(ratios), sum(1 for r in ratios if r > WICK_RATIO_DANGER), sum(ratios) / len(ratios)


def longest_same_direction_run(closes: Iterable[float]) -> int:
    """Longest run of consecutive same-sign bar returns (gaps / one-way moves)."""
    run = 1
    best = 0
    prev_sign = 0
    for r in _returns([c for c in (_f(c) for c in closes) if c is not None]):
        sign = 1 if r > 0 else (-1 if r < 0 else 0)
        if sign != 0 and sign == prev_sign:
            run += 1
        else:
            run = 1
        if sign != 0:
            best = max(best, run)
        prev_sign = sign
    return best


def risk_level_for(score: float) -> str:
    """Map a 0–100 score to ``low|medium|high`` (contract §4)."""
    if score >= RISK_LOW_MIN:
        return "low"
    if score >= RISK_MEDIUM_MIN:
        return "medium"
    return "high"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score_metrics(metrics: dict) -> dict:
    """Penalty-only 0–100 score with a per-term breakdown.

    Starts at 100 and subtracts each term's penalty.  A term whose metric is
    ``None`` (no data) contributes nothing — an unknown must not be silently
    treated as either safe or dangerous; it shows up in ``data_completeness``
    instead.  Returns ``{"score", "risk_level", "terms", "total_penalty"}``.

    One rule sits outside the arithmetic: a listing younger than
    :data:`AGE_NEW_DAYS` is never labelled ``low`` (see the note in the body) —
    the score stays the honest penalty-only number.
    """
    drawdown = metrics.get("max_drawdown_90d")
    spikes = metrics.get("spike_count_30d")
    gap = metrics.get("gap_run_max")
    #: (score term, metric value, penalty bands) — one row per §4 dimension.
    rows = [
        ("liquidity", metrics.get("quote_volume"), BANDS_QUOTE_VOLUME),
        ("depth", metrics.get("depth_1pct_total"), BANDS_DEPTH),
        ("spread", metrics.get("spread_pct"), BANDS_SPREAD),
        ("volatility", metrics.get("volatility_30d_annualized"), BANDS_VOLATILITY),
        ("drawdown", None if drawdown is None else abs(_f(drawdown) or 0.0),
         BANDS_DRAWDOWN),
        ("spike", None if spikes is None else float(spikes) / SPIKE_COUNT_DANGER,
         BANDS_SPIKE_RATE),
        ("wick", metrics.get("avg_wick_ratio"), BANDS_WICK),
        ("trade_size", metrics.get("avg_trade_size"), BANDS_TRADE_SIZE),
        ("gap", None if gap is None else float(gap) / GAP_RUN_DANGER, BANDS_GAP_RATE),
        ("age", metrics.get("age_days"), BANDS_AGE),
        ("volume_activity", metrics.get("volume_ratio"), BANDS_VOLUME_RATIO),
    ]

    penalties: dict[str, Optional[float]] = {}
    for name, value, bands in rows:
        if value is None:
            penalties[name] = None
        else:
            fraction = _band_fraction(_f(value), bands)
            penalties[name] = None if fraction is None else PENALTIES[name] * fraction

    total = sum(p for p in penalties.values() if p is not None)
    score = max(0.0, min(100.0, 100.0 - total))
    level = risk_level_for(score)
    if _is_new(_f(metrics.get("age_days"))):
        # A pair listed for less than AGE_NEW_DAYS has no history to judge, and
        # its age rule is yes/no: with every other dimension healthy it scores 92
        # and the plain mapping would call it "low risk" — the same verdict a
        # multi-year large cap gets.  The age term alone is worth only 8 points,
        # so the verdict is floored at "medium" here instead of pretending the
        # missing history is safe.  The score itself is *not* adjusted: the
        # breakdown must still explain where every point went.
        if level == "low":
            level = "medium"
    terms = [{
        "name": name,
        "penalty": _round(value, 1),
        "max_penalty": PENALTIES[name],
    } for name, value in penalties.items()]
    return {
        "score": round(score, 1),
        "risk_level": level,
        "total_penalty": round(total, 1),
        "terms": terms,
    }


def _is_new(age_days: Optional[float]) -> bool:
    return age_days is not None and age_days < AGE_NEW_DAYS


# --------------------------------------------------------------------------
# flags — each one carries the numbers that produced it
# --------------------------------------------------------------------------
def build_flags(metrics: dict) -> list[dict]:
    """Every triggered flag as ``{code,label,severity,message,evidence}``."""
    flags: list[dict] = []

    def add(code: str, label: str, severity: str, message: str, evidence: dict) -> None:
        flags.append({
            "code": code,
            "label": label,
            "severity": severity,
            "message": message,
            "evidence": evidence,
        })

    qv = _f(metrics.get("quote_volume"))
    if qv is not None and qv < 2e6:
        severity = SEV_HIGH if qv < 1e5 else (SEV_WARN if qv < 5e5 else SEV_INFO)
        add("low_liquidity", "低流动性", severity,
            f"24h 计价成交额仅 {qv:,.0f} USDT，低于 2,000,000 USDT 关注阈值",
            {"quote_volume": round(qv, 2), "info_threshold": 2_000_000.0,
             "warn_threshold": 500_000.0, "high_threshold": 100_000.0,
             "ratio_to_info_threshold": round(qv / 2e6, 4)})

    depth = _f(metrics.get("depth_1pct_total"))
    if depth is not None and depth < 5e5:
        severity = SEV_HIGH if depth < 1e5 else (SEV_WARN if depth < 3e5 else SEV_INFO)
        add("thin_book", "盘口偏薄", severity,
            f"±1% 盘口名义深度 {depth:,.0f} USDT，低于 500,000 USDT 关注阈值",
            {"depth_1pct_total": round(depth, 2), "info_threshold": 500_000.0,
             "warn_threshold": 300_000.0, "high_threshold": 100_000.0,
             "bid_depth_1pct": metrics.get("bid_depth_1pct"),
             "ask_depth_1pct": metrics.get("ask_depth_1pct")})

    spread_pct = _f(metrics.get("spread_pct"))
    if spread_pct is not None and spread_pct > 0.02:
        severity = SEV_HIGH if spread_pct > 0.5 else (SEV_WARN if spread_pct > 0.1 else SEV_INFO)
        add("wide_spread", "价差偏大", severity,
            f"买卖价差 {spread_pct:.4f}%，高于 0.02% 关注阈值",
            {"spread_pct": round(spread_pct, 6), "info_threshold": 0.02,
             "warn_threshold": 0.1, "high_threshold": 0.5,
             "spread": metrics.get("spread"), "last": metrics.get("last")})

    vol = _f(metrics.get("volatility_30d_annualized"))
    if vol is not None and vol > 1.0:
        severity = SEV_HIGH if vol > 3.0 else (SEV_WARN if vol > 2.0 else SEV_INFO)
        add("high_volatility", "高波动", severity,
            f"30d 年化波动率 {vol * 100:.1f}%，高于 100% 关注阈值",
            {"volatility_30d_annualized": round(vol, 4), "info_threshold": 1.0,
             "warn_threshold": 2.0, "high_threshold": 3.0,
             "bars_used": metrics.get("volatility_bars")})

    dd = _f(metrics.get("max_drawdown_90d"))
    if dd is not None and dd > 20.0:
        severity = SEV_HIGH if dd > 80.0 else (SEV_WARN if dd > 40.0 else SEV_INFO)
        add("deep_drawdown", "深度回撤", severity,
            f"90d 最大回撤 {dd:.1f}%，高于 20% 关注阈值",
            {"max_drawdown_90d": round(dd, 2), "info_threshold": 20.0,
             "warn_threshold": 40.0, "high_threshold": 80.0,
             "daily_bars_used": metrics.get("drawdown_bars")})

    spikes = _f(metrics.get("spike_count_30d"))
    if spikes is not None and spikes >= 1:
        severity = SEV_HIGH if spikes >= SPIKE_COUNT_DANGER else SEV_INFO
        add("price_spikes", "单根K线异常涨跌", severity,
            f"30d 内 1h 单根涨跌超 {SPIKE_PCT:.0f}% 共 {int(spikes)} 次",
            {"spike_count_30d": int(spikes), "threshold_pct": SPIKE_PCT,
             "spikes_per_100_bars": metrics.get("spikes_per_100_bars"),
             "hourly_bars_used": metrics.get("hourly_bars")})

    wick = _f(metrics.get("avg_wick_ratio"))
    hot = _f(metrics.get("wick_hot_bars_pct"))
    wick_trigger = (wick is not None and wick > WICK_AVG_WARN) or \
                   (hot is not None and hot >= WICK_HOT_BARS_PCT_INFO)
    if wick_trigger:
        severity = (SEV_HIGH if ((wick is not None and wick > WICK_AVG_DANGER)
                                 or (hot is not None and hot >= WICK_HOT_BARS_PCT_DANGER))
                    else SEV_WARN if (hot is not None and hot >= WICK_HOT_BARS_PCT_WARN)
                    else SEV_INFO)
        add("long_wick", "长影线偏多", severity,
            f"平均影线占振幅 {(wick if wick is not None else 0) * 100:.1f}%"
            f"（阈值 {WICK_AVG_WARN * 100:.0f}%），影线超 {WICK_RATIO_DANGER * 100:.0f}% 的"
            f"K线占 {(hot if hot is not None else 0):.1f}%（阈值 {WICK_HOT_BARS_PCT_INFO:.0f}%）",
            {"avg_wick_ratio": metrics.get("avg_wick_ratio"),
             "avg_threshold": WICK_AVG_WARN,
             "wick_hot_bars_pct": metrics.get("wick_hot_bars_pct"),
             "hot_pct_thresholds": {"info": WICK_HOT_BARS_PCT_INFO,
                                    "warn": WICK_HOT_BARS_PCT_WARN,
                                    "high": WICK_HOT_BARS_PCT_DANGER},
             "wick_over_threshold_bars": metrics.get("wick_over_threshold_bars"),
             "max_wick_ratio": metrics.get("max_wick_ratio"),
             "hourly_bars_used": metrics.get("hourly_bars")})

    avg_trade = _f(metrics.get("avg_trade_size"))
    if avg_trade is not None and (avg_trade > TRADE_SIZE_HUGE or avg_trade < TRADE_SIZE_TINY):
        tiny = avg_trade < TRADE_SIZE_TINY
        add("trade_size_anomaly", "单笔成交额异常", SEV_WARN,
            ("平均单笔成交额仅 %.2f USDT（< %.0f），成交极度碎片化" % (avg_trade, TRADE_SIZE_TINY))
            if tiny else
            ("平均单笔成交额 %.0f USDT（> %.0f），集中度代理偏高" % (avg_trade, TRADE_SIZE_HUGE)),
            {"avg_trade_size": round(avg_trade, 4), "trade_count": metrics.get("trade_count"),
             "quote_volume": metrics.get("quote_volume"),
             "tiny_threshold": TRADE_SIZE_TINY, "huge_threshold": TRADE_SIZE_HUGE})

    run = _f(metrics.get("gap_run_max"))
    if run is not None and run >= GAP_RUN_INFO:
        severity = SEV_HIGH if run >= GAP_RUN_DANGER else (
            SEV_WARN if run >= GAP_RUN_WARN else SEV_INFO)
        add("price_gap_run", "连续同向K线", severity,
            f"最长 {int(run)} 根 1h K 线连续同向（关注阈值 {GAP_RUN_INFO}，"
            f"高危阈值 {GAP_RUN_DANGER}）",
            {"gap_run_max": int(run), "info_threshold": GAP_RUN_INFO,
             "warn_threshold": GAP_RUN_WARN, "high_threshold": GAP_RUN_DANGER,
             "hourly_bars_used": metrics.get("hourly_bars")})

    age = _f(metrics.get("age_days"))
    if _is_new(age):
        add("new_listing", "新上市", SEV_WARN,
            f"上市仅 {age:.0f} 天（< {AGE_NEW_DAYS} 天），历史数据不足以评估",
            {"age_days": round(age, 1), "threshold_days": AGE_NEW_DAYS,
             "listing_date": metrics.get("listing_date"),
             "listing_time_ms": metrics.get("listing_time_ms")})

    ratio = _f(metrics.get("volume_ratio"))
    # NOTE: VOLUME_ACTIVITY_BANDS holds (predicate, penalty) pairs, so indexing it
    # for a numeric threshold compared a float against a function and raised
    # TypeError. The documented threshold is 0.5x (see the message below and the
    # severity split on the next line), so compare against that directly.
    if ratio is not None and ratio < 0.5:
        severity = SEV_WARN if ratio < 0.5 else SEV_INFO
        add("volume_activity_drop", "成交额活跃度低", severity,
            f"近 24h 平均单根成交额 / 前 30d 中位数 = {ratio:.2f}x（低于 0.5x）",
            {"volume_ratio": round(ratio, 4), "threshold": 0.5,
             "quote_volume": metrics.get("quote_volume"),
             "bar_quote_median": metrics.get("bar_quote_median"),
             "hourly_bars_used": metrics.get("hourly_bars")})

    completeness = _f(metrics.get("data_completeness"))
    if completeness is not None and completeness < 0.8:
        add("insufficient_data", "数据不足", SEV_INFO,
            f"可用指标 {completeness * 100:.0f}%（{metrics.get('metrics_available')}"
            f"/{metrics.get('metrics_total')}），评分置信度低",
            {"data_completeness": round(completeness, 4),
             "metrics_available": metrics.get("metrics_available"),
             "metrics_total": metrics.get("metrics_total"),
             "missing": metrics.get("missing_metrics")})

    for reason in metrics.get("fetch_errors") or []:
        add("fetch_error", "数据获取失败", SEV_WARN, f"上游数据缺失: {reason}",
            {"reason": reason})

    return flags


#: Metrics whose availability drives ``data_completeness``.
SCORED_METRICS = (
    "quote_volume", "depth_1pct_total", "spread_pct", "volatility_30d_annualized",
    "max_drawdown_90d", "spike_count_30d", "max_wick_ratio", "avg_trade_size",
    "gap_run_max", "age_days", "volume_ratio",
)


def _add_completeness(metrics: dict) -> None:
    missing = [name for name in SCORED_METRICS if metrics.get(name) is None]
    metrics["metrics_total"] = len(SCORED_METRICS)
    metrics["metrics_available"] = len(SCORED_METRICS) - len(missing)
    metrics["missing_metrics"] = missing
    metrics["data_completeness"] = round(
        (len(SCORED_METRICS) - len(missing)) / len(SCORED_METRICS), 4)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class UpstreamError(RuntimeError):
    """Raised when the public market-data host cannot serve a request."""


class BudgetExceeded(RuntimeError):
    """Raised when the whole-screen wall-clock budget is exhausted."""


async def _aiohttp_get_json(base: str, path: str, params: dict,
                            timeout: float) -> Any:
    """One GET against the market-data host, JSON only (default transport)."""
    import aiohttp  # local import: keeps the module importable without the dep

    url = base.rstrip("/") + path
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    try:
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url, params=params) as response:
                text = await response.text()
                if response.status != 200:
                    body = " ".join(text.split())[:200]
                    raise UpstreamError(f"HTTP {response.status} from {path}: {body}")
                try:
                    import json
                    return json.loads(text)
                except ValueError as exc:
                    raise UpstreamError(f"non-JSON response from {path}: {exc}") from exc
    except UpstreamError:
        raise
    except asyncio.TimeoutError as exc:
        raise UpstreamError(f"timeout after {timeout:.0f}s calling {path}") from exc
    except Exception as exc:  # aiohttp errors, DNS, TLS, ...
        raise UpstreamError(f"{type(exc).__name__}: {exc}") from exc


# --------------------------------------------------------------------------
# the screener
# --------------------------------------------------------------------------
class TokenScreener:
    """Fetches, scores and caches the heuristic screen.

    ``fetch`` is injectable (``async (path, params) -> parsed JSON``) so tests
    exercise the full scoring path against stubbed HTTP — no network, no sleeps.
    """

    def __init__(self, host: Optional[str] = None,
                 fetch: Optional[Callable[[str, dict], Awaitable[Any]]] = None,
                 concurrency: int = DEFAULT_CONCURRENCY,
                 cache_ttl: float = CACHE_TTL,
                 request_timeout: float = REQUEST_TIMEOUT,
                 ticker_timeout: float = TICKER_TIMEOUT,
                 attempts: int = REQUEST_ATTEMPTS,
                 budget: float = SCREEN_BUDGET,
                 now: Callable[[], float] = time.monotonic):
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.request_timeout = float(request_timeout)
        self.ticker_timeout = float(ticker_timeout)
        self.attempts = max(1, int(attempts))
        self.budget = float(budget)
        self.cache_ttl = float(cache_ttl)
        self.concurrency = max(1, int(concurrency))
        self._now = now
        self._fetch_impl = fetch or self._default_fetch
        #: LRU-ordered so a caller-supplied key space cannot grow them for ever.
        self._cache: "OrderedDict[tuple, tuple[float, dict]]" = OrderedDict()
        self._locks: "OrderedDict[tuple, asyncio.Lock]" = OrderedDict()
        self.max_cache_entries = MAX_CACHE_ENTRIES
        self.max_cache_locks = MAX_CACHE_LOCKS
        #: diagnostics for tests / operators (cache hits, upstream calls)
        self.stats = {"upstream_calls": 0, "cache_hits": 0, "cache_misses": 0,
                      "screens": 0, "errors": 0, "retries": 0}

    # ---- transport -------------------------------------------------------
    async def _default_fetch(self, path: str, params: dict) -> Any:
        """One request, retried once on failure; ``X-Request-Timeout`` documents
        the budget by path (the ticker is the only slow endpoint)."""
        timeout = self.ticker_timeout if path.endswith("/ticker/24hr") else self.request_timeout
        last: Optional[Exception] = None
        for attempt in range(self.attempts):
            try:
                return await _aiohttp_get_json(self.host, path, params, timeout)
            except UpstreamError as exc:
                last = exc
                if attempt + 1 < self.attempts:
                    self.stats["retries"] += 1
                    await asyncio.sleep(RETRY_DELAY)
        raise last if last is not None else UpstreamError(f"no attempt made for {path}")

    async def _get(self, path: str, **params) -> Any:
        self.stats["upstream_calls"] += 1
        return await self._fetch_impl(path, params)

    # ---- cache -----------------------------------------------------------
    def cache_key(self, limit: int, min_quote_volume: float, sort: str,
                  interval: str) -> tuple:
        return (int(limit), float(min_quote_volume), str(sort), str(interval))

    def _cache_read(self, key: tuple) -> Optional[dict]:
        """Return a still-fresh cached payload (marking it as such), else None."""
        cached = self._cache.get(key)
        if cached is None or (self._now() - cached[0]) >= self.cache_ttl:
            return None
        self._cache.move_to_end(key)      # LRU: a read is a use
        payload = dict(cached[1])
        payload["cached"] = True
        return payload

    def _cache_write(self, key: tuple, payload: dict) -> None:
        """Store a payload and keep the map bounded (oldest entry out)."""
        self._cache[key] = (self._now(), payload)
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)

    def _cache_lock(self, key: tuple) -> asyncio.Lock:
        """The lock de-duplicating one key's in-flight upstream crawl.

        Cached here only for the duration of that crawl, so it is dropped again in
        the ``finally`` of the callers: 50 distinct detail keys used to leave 50
        locks behind whether or not the fetch succeeded.
        """
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
            self._locks.move_to_end(key)
            self._prune_cache_locks()
        return lock

    def _prune_cache_locks(self) -> None:
        if len(self._locks) <= self.max_cache_locks:
            return
        for stale in [k for k, lock in self._locks.items() if not lock.locked()]:
            self._locks.pop(stale, None)
            if len(self._locks) <= self.max_cache_locks:
                return
        while len(self._locks) > self.max_cache_locks:
            self._locks.popitem(last=False)

    def _drop_cache_lock(self, key: tuple) -> None:
        lock = self._locks.get(key)
        if lock is not None and not lock.locked():
            self._locks.pop(key, None)

    # ---- public surface --------------------------------------------------
    async def screen(self, limit: int = DEFAULT_LIMIT,
                     min_quote_volume: float = DEFAULT_MIN_QUOTE_VOLUME,
                     sort: str = "score", interval: str = "1h") -> dict:
        """Screen the top-``limit`` USDT pairs by 24h quote volume (cached ~5min)."""
        limit = _clamp_limit(limit)
        # Bucketed, not raw: the threshold is both a filter and a cache-key term.
        min_quote_volume = normalize_min_quote_volume(min_quote_volume)
        sort = sort if sort in ("score", "volume") else "score"
        interval = interval or "1h"

        key = self.cache_key(limit, min_quote_volume, sort, interval)
        cached = self._cache_read(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        self.stats["cache_misses"] += 1

        lock = self._cache_lock(key)
        try:
            async with lock:
                # Another request may have filled the cache while we waited.
                cached = self._cache_read(key)
                if cached is not None:
                    self.stats["cache_hits"] += 1
                    return cached

                payload = await self._run_screen(limit, min_quote_volume, sort, interval)
                self._cache_write(key, payload)
                self.stats["screens"] += 1
                out = dict(payload)
                out["cached"] = False
                return out
        finally:
            self._drop_cache_lock(key)

    async def detail(self, symbol: str, interval: str = "1h") -> dict:
        """Full audit detail for one symbol (cached ~5min, like the screen).

        The crawler used to be uncached: ``GET /api/audit/{symbol}`` cost one
        24h-ticker fetch plus four per-symbol requests *every* call, so a loop
        over symbols was an unbounded upstream fan-out.  The same TTL and lock
        discipline as :meth:`screen` now applies, keyed by ``(symbol, interval)``.
        """
        key = ("detail", str(symbol), str(interval or "1h"))
        cached = self._cache_read(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        self.stats["cache_misses"] += 1

        lock = self._cache_lock(key)
        try:
            async with lock:
                # Another request may have filled the cache while we waited.
                cached = self._cache_read(key)
                if cached is not None:
                    self.stats["cache_hits"] += 1
                    return cached

                payload = await self._enrich(symbol, interval)
                self._cache_write(key, payload)
                out = dict(payload)
                out["cached"] = False
                return out
        finally:
            self._drop_cache_lock(key)

    # ---- internals -------------------------------------------------------
    async def _run_screen(self, limit: int, min_quote_volume: float, sort: str,
                          interval: str) -> dict:
        started = self._now()
        try:
            tickers = await self._fetch_universe()
        except Exception as exc:
            self.stats["errors"] += 1
            raise UpstreamError(f"24h ticker unavailable: {exc}") from exc

        candidates = []
        for sym, t in tickers.items():
            if not sym.endswith(USDT):
                continue
            quote_volume = _f(t.get("quoteVolume"))
            if quote_volume is None or quote_volume < min_quote_volume:
                continue
            candidates.append((quote_volume, sym, t))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[:limit]

        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(quote_volume: float, symbol: str, ticker: dict) -> dict:
            async with semaphore:
                metrics = await self._collect(symbol, ticker, interval, started)
                return _finalize(symbol, metrics)

        results = await asyncio.gather(*(one(*item) for item in selected))
        if sort == "volume":
            results.sort(key=lambda r: (-(r["metrics"].get("quote_volume") or 0.0), r["symbol"]))
        else:
            results.sort(key=lambda r: (r["overall_score"], r["symbol"]))

        failures = [{"symbol": r["symbol"], "error": (r["metrics"].get("fetch_errors") or [""])[0]}
                    for r in results if r["metrics"].get("fetch_errors")]
        return {
            "results": results,
            "count": len(results),
            "universe_usdt": len(candidates),
            "sampled": len(selected),
            "failures": failures,
        }

    async def _fetch_universe(self) -> dict:
        raw = await self._get("/api/v3/ticker/24hr")
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            raise UpstreamError("unexpected ticker payload shape")
        out = {}
        for row in raw:
            if isinstance(row, dict) and row.get("symbol"):
                out[str(row["symbol"])] = row
        if not out:
            raise UpstreamError("empty ticker payload")
        return out

    async def _collect(self, symbol: str, ticker: dict, interval: str,
                       started: Optional[float] = None) -> dict:
        """Depth + hourly klines + daily history for one symbol.

        Every request passes through :meth:`_budgeted`, so a slow host degrades
        into per-symbol ``fetch_error`` flags instead of letting one screen run
        for minutes.
        """
        metrics = _base_metrics(symbol, ticker)
        errors: list[str] = []
        started = self._now() if started is None else started

        depth_task = asyncio.ensure_future(
            self._budgeted("/api/v3/depth", started, symbol=symbol, limit=100))
        hourly_task = asyncio.ensure_future(
            self._budgeted("/api/v3/klines", started, symbol=symbol,
                           interval=interval, limit=720))
        daily_task = asyncio.ensure_future(
            self._budgeted("/api/v3/klines", started, symbol=symbol, interval="1d",
                           limit=91))
        listing_task = asyncio.ensure_future(
            self._budgeted("/api/v3/klines", started, symbol=symbol, interval="1d",
                           limit=1, startTime=0))
        raw_depth, raw_hourly, raw_daily, raw_listing = await asyncio.gather(
            depth_task, hourly_task, daily_task, listing_task, return_exceptions=True)

        if isinstance(raw_depth, Exception):
            errors.append(f"depth: {raw_depth}")
        else:
            metrics.update(_depth_metrics(raw_depth))
        if isinstance(raw_hourly, Exception):
            errors.append(f"klines({interval}): {raw_hourly}")
        else:
            metrics.update(_hourly_metrics(_klines(raw_hourly), metrics))
        if isinstance(raw_daily, Exception):
            errors.append(f"klines(1d): {raw_daily}")
        else:
            metrics.update(_daily_metrics(_klines(raw_daily)))
        if isinstance(raw_listing, Exception):
            errors.append(f"klines(1d,listing): {raw_listing}")
        else:
            metrics.update(_listing_metrics(_klines(raw_listing)))

        metrics["fetch_errors"] = errors
        return metrics

    async def _budgeted(self, path: str, started: float, **params) -> Any:
        if (self._now() - started) > self.budget:
            raise BudgetExceeded(
                f"screen budget {self.budget:.0f}s exceeded before {path}")
        return await self._get(path, **params)

    async def _enrich(self, symbol: str, interval: str) -> dict:
        ticker = {}
        errors: list[str] = []
        try:
            ticker = (await self._fetch_universe()).get(symbol, {})
        except Exception as exc:
            errors.append(f"ticker24hr: {exc}")
        metrics = await self._collect(symbol, ticker, interval)
        if errors:
            metrics["fetch_errors"] = list(metrics.get("fetch_errors") or []) + errors
        return _finalize(symbol, metrics)


# --------------------------------------------------------------------------
# metric computation
# --------------------------------------------------------------------------
def _clamp_limit(limit: Any) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = DEFAULT_LIMIT
    if value <= 0:
        value = DEFAULT_LIMIT
    return min(value, MAX_LIMIT)


def _clamp_min_quote_volume(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MIN_QUOTE_VOLUME
    return max(0.0, out)


def normalize_min_quote_volume(value: Any) -> float:
    """Snap a caller-supplied liquidity threshold onto :data:`MIN_QUOTE_VOLUME_LADDER`.

    The cache key contains this value, so an un-normalised float would make the
    cache useless against a loop that varies it by 0.1 each call (each variation
    re-runs the whole crawl).  Floor-snapping keeps the filter conservative —
    the returned bucket is never larger than what was asked for — and bounds the
    key space.
    """
    clamped = _clamp_min_quote_volume(value)
    bucket = MIN_QUOTE_VOLUME_LADDER[0]
    for step in MIN_QUOTE_VOLUME_LADDER:
        if clamped >= step:
            bucket = step
        else:
            break
    return bucket


#: A USDT spot pair: letters/digits, optionally hyphenated, ending in the quote.
SYMBOL_MIN_LEN = 4
SYMBOL_MAX_LEN = 24


def valid_symbol(symbol: Any) -> Optional[str]:
    """Return the canonical ``SYMBOL`` when the shape is a USDT pair, else ``None``.

    The symbol is part of a cache key (``/api/audit/{symbol}``, ``/api/coin/{symbol}``)
    and of every upstream path, so an unvalidated value is a caller-controlled key
    space *and* an SSRF-ish path component.  The shape checked here is the one the
    contract documents — uppercase letters/digits, an optional hyphen (Binance
    tolerates ``BTC-USDT``), 4..24 characters, quoted in USDT, non-empty base.
    Membership of the exchange universe is checked separately (it needs a fetch).
    """
    text = str(symbol or "").strip().upper()
    if not (SYMBOL_MIN_LEN <= len(text) <= SYMBOL_MAX_LEN):
        return None
    if not (text.replace("-", "").isalnum() and text.replace("-", "").isascii()):
        return None
    if "-" in text:
        base, _, quote = text.partition("-")
        if not base or not base.isalnum():
            return None
        if quote != USDT:
            return None
    else:
        base = text[: -len(USDT)]
        if not base:
            return None
        if not text.endswith(USDT):
            return None
    return text


def _base_metrics(symbol: str, ticker: dict) -> dict:
    quote_volume = _f(ticker.get("quoteVolume"))
    count = _f(ticker.get("count"))
    if count is None:
        count = _f(ticker.get("trades"))
    metrics = {
        "symbol": symbol,
        "base_asset": symbol[:-len(USDT)] if symbol.endswith(USDT) else symbol,
        "quote_asset": USDT if symbol.endswith(USDT) else None,
        "last": _round(ticker.get("lastPrice"), 8),
        "change_pct": _round(ticker.get("priceChangePercent"), 4),
        "high_24h": _round(ticker.get("highPrice"), 8),
        "low_24h": _round(ticker.get("lowPrice"), 8),
        "quote_volume": _round(quote_volume, 2),
        "volume": _round(ticker.get("volume"), 8),
        "trade_count": None if count is None else int(count),
        "avg_trade_size": (round(quote_volume / count, 4)
                           if quote_volume is not None and count else None),
        # filled in by the kline/depth passes
        "spread": None, "spread_pct": None,
        "bid_depth_1pct": None, "ask_depth_1pct": None, "depth_1pct_total": None,
        "bid_total": None, "ask_total": None, "depth_levels": None,
        "volatility_30d_annualized": None, "volatility_bars": None,
        "max_drawdown_90d": None, "drawdown_bars": None,
        "spike_count_30d": None, "spikes_per_100_bars": None,
        "max_wick_ratio": None, "avg_wick_ratio": None, "wick_over_threshold_bars": None,
        "wick_hot_bars_pct": None,
        "gap_run_max": None, "hourly_bars": None,
        "bar_quote_median": None, "volume_ratio": None,
        "listing_date": None, "listing_time_ms": None, "age_days": None,
        "fetch_errors": [],
    }
    return metrics


def _depth_metrics(raw: Any) -> dict:
    """Spread and ±1% notional depth from ``/api/v3/depth?limit=100``."""
    if not isinstance(raw, dict):
        return {}
    bids = [(p, q) for p, q in ((_f(lv[0]), _f(lv[1])) for lv in raw.get("bids") or []
                                if isinstance(lv, (list, tuple)) and len(lv) >= 2)
            if p is not None and q is not None]
    asks = [(p, q) for p, q in ((_f(lv[0]), _f(lv[1])) for lv in raw.get("asks") or []
                                if isinstance(lv, (list, tuple)) and len(lv) >= 2)
            if p is not None and q is not None]
    if not bids or not asks:
        return {}
    best_bid = max(p for p, _ in bids)
    best_ask = min(p for p, _ in asks)
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    spread_pct = (spread / mid * 100.0) if mid > 0 else None
    lower = mid * 0.99
    upper = mid * 1.01
    bid_depth = sum(p * q for p, q in bids if p >= lower)
    ask_depth = sum(p * q for p, q in asks if p <= upper)
    return {
        "spread": _round(spread, 10),
        "spread_pct": _round(spread_pct, 6),
        "bid_depth_1pct": _round(bid_depth, 2),
        "ask_depth_1pct": _round(ask_depth, 2),
        "depth_1pct_total": _round(bid_depth + ask_depth, 2),
        "bid_total": _round(sum(p * q for p, q in bids), 2),
        "ask_total": _round(sum(p * q for p, q in asks), 2),
        "depth_levels": [len(bids), len(asks)],
    }


def _hourly_metrics(series: list[list[float]], base: dict) -> dict:
    """Volatility / spikes / wicks / gap runs / volume-ratio from hourly bars."""
    if len(series) < MIN_CANDLES:
        return {}
    closes = [row[3] for row in series]
    out: dict = {"hourly_bars": len(series)}

    vol = annualised_volatility(closes, HOURS_PER_YEAR)
    out["volatility_30d_annualized"] = _round(vol, 4)
    out["volatility_bars"] = len(closes) - 1

    spikes = spike_count(closes)
    out["spike_count_30d"] = spikes
    out["spikes_per_100_bars"] = round(spikes / (len(closes) - 1) * 100.0, 3) if len(closes) > 1 else None

    max_wick, wick_over, avg_wick = _wick_metrics(series)
    out["max_wick_ratio"] = _round(max_wick, 4)
    out["avg_wick_ratio"] = _round(avg_wick, 4)
    out["wick_over_threshold_bars"] = wick_over
    counted = sum(1 for _o, high, low, _c, _v, _t in series if high > low)
    out["wick_hot_bars_pct"] = (round((wick_over or 0) / counted * 100.0, 3)
                                if counted else None)

    out["gap_run_max"] = longest_same_direction_run(closes)

    # volume_ratio: mean hourly quote turnover of the last 24 bars vs the median
    # of the *preceding 30 days*.  Restricting the baseline to that window
    # matters: a median over the full 720+ bars is skewed by a token's early,
    # illiquid weeks and produced meaningless 200x+ readings in calibration.
    bar_quote = [row[3] * row[4] for row in series if row[4] is not None]
    window = bar_quote[-24:]
    hist = bar_quote[-744:-24] or bar_quote[:-24] or bar_quote
    med = _median(hist)
    if med and med > 0 and window:
        out["bar_quote_median"] = _round(med, 2)
        out["volume_ratio"] = _round((sum(window) / len(window)) / med, 4)
    else:
        out["bar_quote_median"] = _round(med, 2)
        out["volume_ratio"] = None
    return out


def _daily_metrics(series: list[list[float]]) -> dict:
    """90d max drawdown from the most recent daily bars (``limit=91``)."""
    if not series:
        return {}
    out: dict = {"daily_bars": len(series)}
    closes = [row[3] for row in series]
    window = closes[-91:]  # 90 periods == 91 closes
    out["max_drawdown_90d"] = _round(max_drawdown(window), 2)
    out["drawdown_bars"] = max(0, len(window) - 1)
    return out


def _listing_metrics(series: list[list[float]]) -> dict:
    """Listing date / age from the **oldest** daily bar.

    ``startTime=0&limit=1`` is the only query that returns the first day a pair
    ever traded: ``startTime=0`` alone always starts at the beginning of the
    pair's history, so a ``limit`` on that query would return bars from years
    ago rather than recent ones (verified against the live host).
    """
    if not series:
        return {}
    first_ms = series[0][5]
    if not first_ms:
        return {}
    return {
        "listing_time_ms": int(first_ms),
        "listing_date": time.strftime("%Y-%m-%d", time.gmtime(first_ms / 1000.0)),
        "age_days": round(max(0.0, (time.time() * 1000.0 - first_ms) / 86_400_000.0), 1),
    }


def _finalize(symbol: str, metrics: dict) -> dict:
    _add_completeness(metrics)
    score = score_metrics(metrics)
    flags = build_flags(metrics)
    return {
        "symbol": symbol,
        "base_asset": metrics.get("base_asset"),
        "quote_asset": metrics.get("quote_asset"),
        "overall_score": score["score"],
        "risk_level": score["risk_level"],
        "flags": [f["message"] for f in flags],
        "flag_count": len(flags),
        "metrics": metrics,
        "score_breakdown": score,
        "flag_details": flags,
    }


# --------------------------------------------------------------------------
# default process-wide instance
# --------------------------------------------------------------------------
_default: Optional[TokenScreener] = None


def get_screener(config: Any = None) -> TokenScreener:
    """Process-wide screener, honouring ``config.market_data_host`` when present.

    Falls back to :data:`DEFAULT_HOST` because ``Config`` does not (yet) define
    ``market_data_host`` — see contract §0.2.
    """
    global _default
    host = getattr(config, "market_data_host", None) or DEFAULT_HOST
    if _default is None:
        _default = TokenScreener(host=host)
    elif host and _default.host != host.rstrip("/"):
        _default.host = host.rstrip("/")
    return _default


def reset_default_screener() -> None:
    """Drop the cached instance (tests, or after a settings change)."""
    global _default
    _default = None
