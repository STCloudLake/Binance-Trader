"""Heuristic market metrics shared by the coin page, the data overview and the
screener.  Everything here is derived from **public exchange data only** — these
are NOT on-chain contract audits (``docs/overhaul/MARKET_PAGES_API.md`` §2/§4).

Definitions (kept here so every caller uses the same numbers):

* **return N-day** — ``close[-1] / close[-1-N] - 1`` on daily candles, ×100.
* **volatility_30d_annualized** — stdev of the last 30 daily log returns × √365.
* **max_drawdown_90d** — largest peak-to-trough close drawdown over the last 90
  daily candles, in percent (positive number = 32.1 means a 32.1% drawdown).
* **correlation_btc_30d** — Pearson correlation of daily log returns.
* **scores** — 0..100, log-scaled against a reference value, so 50 ≈ "average
  for a liquid USDT pair".  They are comparable across coins but arbitrary in
  absolute terms; the UI must present them as heuristic.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

#: Mean daily trading sessions per year — used to annualize daily volatility.
TRADING_DAYS_PER_YEAR = 365.0


def _clean(value) -> Optional[float]:
    """None/NaN/inf → ``None`` so the value survives a JSON round-trip."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def to_daily(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return daily closes from an OHLCV frame of any interval.

    Daily/4h/1h frames are all acceptable inputs; anything finer is resampled.
    The index must be a DatetimeIndex (the parquet layout the DataFeeder reads).
    """
    if df is None or len(df) == 0 or "close" not in df.columns:
        return None
    out = df[["close"]].copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if len(out) < 2:
        return None
    # A coarse guess is enough: 1d data has ~1 bar/day already.
    span_days = (out.index[-1] - out.index[0]).total_seconds() / 86400.0
    if span_days <= 0:
        return None
    per_day = len(out) / span_days
    if per_day > 1.5:
        out = out.resample("1D").last().dropna()
    return out if len(out) >= 2 else None


def closes_to_daily(closes: list[float], timestamps: list[float]) -> Optional[pd.DataFrame]:
    """Build a daily close frame from raw (timestamp_ms, close) pairs."""
    if not closes or len(closes) != len(timestamps) or len(closes) < 2:
        return None
    idx = pd.to_datetime(pd.Series(timestamps), unit="ms")
    df = pd.DataFrame({"close": [float(c) for c in closes]}, index=pd.DatetimeIndex(idx))
    return to_daily(df)


def performance(daily: Optional[pd.DataFrame],
                change_pct_24h: Optional[float] = None) -> dict:
    """1/7/30/90-day performance in percent (``None`` when history is short)."""
    out = {"d1": _clean(change_pct_24h), "d7": None, "d30": None, "d90": None}
    if daily is None or len(daily) < 2:
        return out
    closes = daily["close"].astype(float)
    last = float(closes.iloc[-1])
    if out["d1"] is None:
        out["d1"] = _clean((last / float(closes.iloc[-2]) - 1) * 100) if len(closes) >= 2 else None
    for key, days in (("d7", 7), ("d30", 30), ("d90", 90)):
        if len(closes) > days:
            base = float(closes.iloc[-1 - days])
            if base > 0:
                out[key] = _clean((last / base - 1) * 100)
    return out


def volatility_annualized(daily: Optional[pd.DataFrame], window: int = 30) -> Optional[float]:
    """Annualized stdev of daily log returns over ``window`` candles."""
    if daily is None or len(daily) < 3:
        return None
    closes = daily["close"].astype(float)
    closes = closes[closes > 0]
    if len(closes) < 3:
        return None
    rets = (closes / closes.shift(1)).apply(lambda x: math.log(x) if x > 0 else None).dropna()
    if len(rets) < 2:
        return None
    tail = rets.tail(window)
    if len(tail) < 2:
        return None
    return _clean(float(tail.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(daily: Optional[pd.DataFrame], window: int = 90) -> Optional[float]:
    """Largest peak-to-trough drawdown (percent, positive) over ``window``."""
    if daily is None or len(daily) < 2:
        return None
    closes = daily["close"].astype(float).tail(window)
    if len(closes) < 2:
        return None
    peak = closes.cummax()
    dd = (closes / peak - 1.0) * 100.0
    return _clean(abs(float(dd.min())))


def correlation(a: Optional[pd.DataFrame], b: Optional[pd.DataFrame],
                window: int = 30) -> Optional[float]:
    """Pearson correlation of daily log returns between two daily frames."""
    if a is None or b is None or len(a) < 3 or len(b) < 3:
        return None
    joined = pd.concat(
        [a["close"].astype(float).rename("a"), b["close"].astype(float).rename("b")],
        axis=1, join="inner").dropna()
    if len(joined) < 3:
        return None
    joined = joined[(joined["a"] > 0) & (joined["b"] > 0)]
    if len(joined) < 3:
        return None
    rets = joined.apply(lambda col: col.apply(lambda x: math.log(x)))
    rets = rets.diff().dropna().tail(window)
    if len(rets) < 3:
        return None
    if float(rets["a"].std()) == 0.0 or float(rets["b"].std()) == 0.0:
        return None
    return _clean(float(rets["a"].corr(rets["b"])))


# ----------------------------------------------------------------------
# heuristic 0..100 scores
# ----------------------------------------------------------------------
def _log_score(value: Optional[float], reference: float, above_is_better: bool) -> Optional[int]:
    """Map a positive quantity to 0..100 on a log scale around ``reference``.

    ``value == reference`` → 50; one order of magnitude better → ~83, one order
    worse → ~17.  ``None`` stays ``None`` (no data → no score, never a fake 0).
    """
    v = _clean(value)
    if v is None or v <= 0 or reference <= 0:
        return None
    ratio = v / reference
    if not above_is_better:
        ratio = 1.0 / ratio
    score = 50.0 + 33.0 * math.log10(ratio)
    return int(max(0, min(100, round(score))))


def liquidity_score(quote_volume_24h: Optional[float]) -> Optional[int]:
    """24h quote volume vs 1M USDT (below that a pair is thin for our sizes)."""
    return _log_score(quote_volume_24h, 1_000_000.0, above_is_better=True)


def volume_score(trade_count_24h: Optional[float]) -> Optional[int]:
    """24h trade count vs 20k trades."""
    return _log_score(trade_count_24h, 20_000.0, above_is_better=True)


def spread_score(spread_pct: Optional[float]) -> Optional[int]:
    """Relative bid/ask spread vs 5 bps (0.0005) — lower is better."""
    return _log_score(spread_pct, 0.0005, above_is_better=False)


def risk_flags(*, quote_volume: Optional[float], spread_pct: Optional[float],
               volatility: Optional[float], drawdown: Optional[float],
               trade_count: Optional[float], listing_age_days: Optional[int]) -> list[str]:
    """Human-readable heuristic flags (contract §4 flag vocabulary)."""
    flags: list[str] = []
    qv = _clean(quote_volume)
    if qv is not None and qv < 100_000:
        flags.append("低流动性")
    sp = _clean(spread_pct)
    if sp is not None and sp > 0.001:
        flags.append("价差偏大")
    vol = _clean(volatility)
    if vol is not None and vol > 1.5:
        flags.append("波动率偏高")
    dd = _clean(drawdown)
    if dd is not None and dd > 50:
        flags.append("回撤偏深")
    tc = _clean(trade_count)
    if qv is not None and tc is not None and tc > 0:
        avg_trade = qv / tc
        if avg_trade > 50_000:
            flags.append("单笔成交额异常大")
        elif avg_trade < 50:
            flags.append("单笔成交额异常小")
    if listing_age_days is not None and listing_age_days < 30:
        flags.append("新上市")
    return flags
