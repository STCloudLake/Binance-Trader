"""High-frequency microstructure features from ``/api/v3/depth`` + ``/api/v3/trades``.

Principle (Tsay, *Analysis of Financial Time Series*, ch. 5)
-----------------------------------------------------------
Our features are built from 20-bar OHLCV indicators, which throw away the
intraday information that actually moves a fill: who is crossing the spread, how
much size sits at the touch, how fast trades arrive.  Tsay ch. 5 documents
order-flow / microstructure variables as the classic high-frequency predictors —
the **order-flow imbalance** (buyer- vs seller-initiated volume), the
**microprice** (size-weighted touch, which leads the mid), **trade size**
distribution and **realized volatility** measured from transaction prices rather
than from bar closes.

Formulas
--------
Order-flow imbalance over ``n`` levels (``bᵢ``/``aᵢ`` = bid/ask quantity)::

    OFI_depth = (Σᵢ bᵢ − Σᵢ aᵢ) / (Σᵢ bᵢ + Σᵢ aᵢ)              ∈ [−1, +1]
    OFI_trades = (Σ buy_qty − Σ sell_qty) / (Σ buy_qty + Σ sell_qty)

where ``buy`` is the **aggressor** side: Binance's ``isBuyerMaker = true`` means
the resting order was the buyer, i.e. the *aggressor sold*, so trade flow uses
``aggressor_buy = not isBuyerMaker``.

Distance-weighted depth imbalance (level ``i`` at distance ``dᵢ`` in bp)::

    wᵢ = 1 / (1 + dᵢ / DEPTH_DECAY_BP),   DwOFI = (Σ wᵢbᵢ − Σ wᵢaᵢ) / (Σ wᵢbᵢ + Σ wᵢaᵢ)

so a level ``DEPTH_DECAY_BP`` basis points away carries half the weight of the
touch (``5`` bp is the documented default).  ``1/(1 + dᵢ)`` with ``dᵢ`` in bp
would be a near-binary "top level only" filter (a level 10 bp away would score
0.09) and was rejected for that reason.

Microprice (Stoikov): the touch weighted by the **opposite** side's size::

    micro = (bid_px·ask_qty + ask_px·bid_qty) / (bid_qty + ask_qty)
    micro_dev_bps = (micro − mid) / mid · 10⁴,   mid = (bid_px + ask_px) / 2

Realized volatility from transaction prices, sampling every ``k`` trades::

    RV = sqrt( Σ_j (ln p_{j} − ln p_{j−1})² )        (per sample, not annualised)

Trade-arrival intensity and its change::

    λ = n_trades / span_seconds
    activity_ratio = λ(recent half) / λ(earlier half)

Look-ahead: how it is enforced
------------------------------
Every feature is a pure function of **one** snapshot plus an explicit
``as_of_ms``:

1. :func:`compute_features` **drops every trade whose** ``time > as_of_ms``
   (counted in ``dropped_future_trades``), so a snapshot fetched concurrently
   with the decision cannot leak a later print into the decision;
2. the depth book carries no timestamp, so its age is reported
   (``book_age_ms = as_of_ms − book_time_ms`` when the caller supplies the fetch
   time) and the caller must not reuse a stale book across decisions;
3. **no** feature uses a future bar: realized volatility is computed from the
   trade prints inside the window, and nothing is shifted, averaged or
   diffed against a later sample.
The regression test injects a trade stamped after ``as_of_ms`` that would flip
the flow imbalance and asserts the feature is unchanged.

Bounded / cached / offline-testable
-----------------------------------
* every list is truncated (``MAX_DEPTH_LEVELS``, ``MAX_TRADES``) so the compute
  cost is O(1) per decision;
* :class:`MicrostructureCache` is a bounded TTL cache holding only the computed
  feature dicts (never the raw payloads);
* the fetch wrapper takes **any** object exposing ``order_book`` / ``recent_trades``
  coroutines, so tests stub it and no test touches the network;
* the module needs no change to ``core/market_data/data_client.py`` (its
  ``order_book`` / ``recent_trades`` helpers already accept a limit).

Limitations
-----------
* One snapshot is a *point-in-time* book: it cannot detect spoofing, hidden
  liquidity or the queue position that decides whether a resting order fills.
* ``/api/v3/trades`` is capped at 1000 prints (a few seconds on BTCUSDT 1h
  volumes), so ``λ`` is a burst measure, not a session intensity.
* Depth-weighting and ``k``-trade volatility sampling are documented choices,
  not estimated parameters — no calibration evidence is claimed for them.
* Nothing is wired into the live path: :data:`MICROSTRUCTURE_ENABLED` is
  ``False``, and the engine seam only computes these features when a caller
  explicitly enables them.

Note on ``core/ml/volatility.py`` (phase P3): that module did not exist when this
one was written, so :func:`realized_volatility` here is local and deliberately
minimal.  The two are **not** duplicates — P3's ``realized_vol``/``ewma_vol``
measure *bar-level* volatility over a rolling window, while this one measures the
**tick path** of the trade prints inside one snapshot (``Σ Δln p²`` sampled every
``RV_TRADE_STRIDE`` prints).  A future consumer that wants one annualised number
from both should call ``core.ml.volatility.annualize`` on the bar series and
compare it with this tick measure as a separate feature.
"""

from __future__ import annotations

import math
import time

import numpy as np

# ── switches / bounds (all OFF by default) ──────────────────────────────

#: Master switch.  ``False`` = no live component computes these features.
MICROSTRUCTURE_ENABLED = False
#: TTL of the computed-feature cache (seconds).  A depth book is stale in
#: milliseconds; a few seconds is already generous for a decision cadence of
#: one bar, and the age is reported so a consumer can refuse a stale snapshot.
MICROSTRUCTURE_CACHE_TTL_SECS = 5.0
#: Hard bounds — the compute cost must not depend on the exchange's page size.
MAX_DEPTH_LEVELS = 20
MAX_TRADES = 1000
MAX_CACHE_ENTRIES = 64
#: Volatility sampling stride (every k-th trade) — a documented choice.
RV_TRADE_STRIDE = 10
#: Distance (bp) at which a depth level carries half the weight of the touch.
DEPTH_DECAY_BP = 5.0
#: Floor on the mean distance (bp) in :func:`book_slope_ratio`, so a one-sided
#: touch-only book yields a finite slope instead of a division by zero.
MIN_SLOPE_DIST_BP = 0.5
#: Keys :func:`compute_features` always returns (the feature contract).
FEATURE_KEYS: tuple[str, ...] = (
    "mid", "microprice", "microprice_dev_bps", "spread_bps",
    "ofi_depth", "ofi_depth_weighted", "ofi_trades", "book_slope_ratio",
    "trade_count", "trade_mean_qty", "trade_median_qty", "trade_large_share",
    "trade_notional", "rv_trade", "arrival_rate_hz", "activity_ratio",
    "as_of_ms", "book_age_ms", "dropped_future_trades", "n_depth_levels",
)


# ── order book ──────────────────────────────────────────────────────────

def _levels(rows, limit: int) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for row in list(rows or [])[:max(int(limit), 1)]:
        try:
            out.append((float(row[0]), float(row[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def best_bid_ask(order_book: dict, *, limit: int = MAX_DEPTH_LEVELS
                 ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Top-of-book ``((bid_px, bid_qty), (ask_px, ask_qty))`` or ``(None, None)``."""
    bids = _levels((order_book or {}).get("bids"), limit)
    asks = _levels((order_book or {}).get("asks"), limit)
    return (bids[0] if bids else None), (asks[0] if asks else None)


def mid_price(order_book: dict, *, limit: int = MAX_DEPTH_LEVELS) -> float:
    """``(best_bid + best_ask) / 2``; ``nan`` when either side is missing."""
    bid, ask = best_bid_ask(order_book, limit=limit)
    if not bid or not ask:
        return float("nan")
    return 0.5 * (bid[0] + ask[0])


def order_flow_imbalance(order_book: dict, *, limit: int = MAX_DEPTH_LEVELS) -> float:
    """Depth imbalance over the top ``limit`` levels, in ``[-1, +1]``.

    ``+1`` = bids dominate (buy-side pressure), ``−1`` = asks dominate.
    Returns ``nan`` for an empty book and ``0.0`` when both sides are empty of
    quantity (a zero total is *no information*, not a signal).
    """
    bids = _levels((order_book or {}).get("bids"), limit)
    asks = _levels((order_book or {}).get("asks"), limit)
    bq = float(sum(q for _, q in bids))
    aq = float(sum(q for _, q in asks))
    if not bids and not asks:
        return float("nan")
    if bq + aq <= 0:
        return 0.0
    return (bq - aq) / (bq + aq)


def depth_weighted_imbalance(order_book: dict,
                             *, limit: int = MAX_DEPTH_LEVELS,
                             decay_bp: float = DEPTH_DECAY_BP) -> float:
    """Distance-weighted depth imbalance (near levels count more).

    Weights are ``1/(1 + distance_in_bp / decay_bp)`` relative to the same
    side's best price: a level ``decay_bp`` (5 bp by default) away carries half
    the weight of the touch, so a large order far from the book cannot outvote
    the top of book.
    """
    bid, ask = best_bid_ask(order_book, limit=limit)
    if not bid or not ask:
        return float("nan")
    bids = _levels((order_book or {}).get("bids"), limit)
    asks = _levels((order_book or {}).get("asks"), limit)
    decay = max(float(decay_bp), 1e-9)
    num = den = 0.0
    for px, qty in bids:
        d = (bid[0] - px) / bid[0] * 1e4 if bid[0] else 0.0
        w = 1.0 / (1.0 + max(d, 0.0) / decay)
        num += w * qty
        den += w * qty
    for px, qty in asks:
        d = (px - ask[0]) / ask[0] * 1e4 if ask[0] else 0.0
        w = 1.0 / (1.0 + max(d, 0.0) / decay)
        num -= w * qty
        den += w * qty
    return float(num / den) if den > 0 else 0.0


def book_slope_ratio(order_book: dict, *, limit: int = MAX_DEPTH_LEVELS) -> float:
    """Ask-side vs bid-side quantity per basis point of distance.

    ``> 1`` means the offer is concentrated nearer the touch than the bid (a
    sell-side wall); ``< 1`` the mirror.  The mean distance is floored at
    :data:`MIN_SLOPE_DIST_BP` so a one-level side gives a finite slope instead
    of ``inf/inf``.  ``nan`` when either side is empty.
    """
    bid, ask = best_bid_ask(order_book, limit=limit)
    if not bid or not ask:
        return float("nan")
    bids = _levels((order_book or {}).get("bids"), limit)
    asks = _levels((order_book or {}).get("asks"), limit)
    b_dist = sum(abs(bid[0] - px) / bid[0] * 1e4 for px, _ in bids) / max(len(bids), 1)
    a_dist = sum(abs(px - ask[0]) / ask[0] * 1e4 for px, _ in asks) / max(len(asks), 1)
    b_qty = sum(q for _, q in bids) / max(len(bids), 1)
    a_qty = sum(q for _, q in asks) / max(len(asks), 1)
    if b_qty <= 0 or a_qty <= 0:
        return float("nan")
    b_slope = b_qty / max(b_dist, MIN_SLOPE_DIST_BP)
    a_slope = a_qty / max(a_dist, MIN_SLOPE_DIST_BP)
    return float(a_slope / b_slope)


def microprice(order_book: dict, *, limit: int = MAX_DEPTH_LEVELS) -> float:
    """Size-weighted touch: ``(bid·ask_qty + ask·bid_qty)/(bid_qty + ask_qty)``.

    Falls back to the mid when both touch sizes are zero / missing.
    """
    bid, ask = best_bid_ask(order_book, limit=limit)
    if not bid or not ask:
        return float("nan")
    denom = bid[1] + ask[1]
    if denom <= 0:
        return 0.5 * (bid[0] + ask[0])
    return float((bid[0] * ask[1] + ask[0] * bid[1]) / denom)


def microprice_deviation_bps(order_book: dict, *, limit: int = MAX_DEPTH_LEVELS) -> float:
    """``(microprice − mid)/mid · 10⁴`` — the lead/lag signal at the touch."""
    micro = microprice(order_book, limit=limit)
    mid = mid_price(order_book, limit=limit)
    if not np.isfinite(micro) or not np.isfinite(mid) or mid <= 0:
        return float("nan")
    return float((micro - mid) / mid * 1e4)


def spread_bps(order_book: dict, *, limit: int = MAX_DEPTH_LEVELS) -> float:
    """Relative quoted spread in basis points."""
    bid, ask = best_bid_ask(order_book, limit=limit)
    if not bid or not ask:
        return float("nan")
    mid = 0.5 * (bid[0] + ask[0])
    if mid <= 0:
        return float("nan")
    return float((ask[0] - bid[0]) / mid * 1e4)


# ── trades ──────────────────────────────────────────────────────────────

def filter_trades(trades, *, as_of_ms: float | None = None,
                  limit: int = MAX_TRADES) -> tuple[list[dict], int]:
    """Trades with ``time ≤ as_of_ms``, newest ``limit`` kept.

    Returns ``(kept, dropped_future)``.  This is the single place the
    no-look-ahead rule is applied, and it is applied **before** any statistic.
    """
    rows = list(trades or [])
    dropped = 0
    kept: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        t = row.get("time")
        if as_of_ms is not None and t is not None:
            try:
                if float(t) > float(as_of_ms):
                    dropped += 1
                    continue
            except (TypeError, ValueError):
                continue
        kept.append(row)
    if len(kept) > int(limit):
        kept = kept[-int(limit):]
    return kept, dropped


def _qty(trade: dict) -> float:
    for key in ("qty", "quoteQty", "quantity"):
        if key in trade and trade[key] is not None:
            try:
                return float(trade[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _price(trade: dict) -> float:
    try:
        return float(trade.get("price"))
    except (TypeError, ValueError):
        return float("nan")


def aggressor_is_buy(trade: dict) -> bool:
    """True when the **taker** bought (Binance: ``isBuyerMaker == False``)."""
    maker = trade.get("isBuyerMaker", trade.get("is_buyer_maker"))
    if maker is None:
        return False
    return not bool(maker)


def trade_flow_imbalance(trades) -> float:
    """Aggressor-signed trade-size imbalance in ``[-1, +1]``.

    ``buy`` volume is the taker-buy volume (``not isBuyerMaker``).  ``0.0`` when
    there is no volume at all (no information, not a signal).
    """
    buy = sell = 0.0
    for t in trades or []:
        if not isinstance(t, dict):
            continue
        q = _qty(t)
        if aggressor_is_buy(t):
            buy += q
        else:
            sell += q
    total = buy + sell
    if total <= 0:
        return 0.0
    return float((buy - sell) / total)


def trade_size_stats(trades, *, large_quantile: float = 0.9) -> dict:
    """Trade count, mean/median size, large-trade share and notional.

    ``large_share`` is the fraction of **volume** (not count) printed by trades
    above the ``large_quantile`` of the size distribution — a whale-vs-algo
    proxy that a bar's total volume cannot express.
    """
    sizes = np.asarray([_qty(t) for t in (trades or []) if isinstance(t, dict)],
                       dtype=float)
    notional = float(np.nansum([_price(t) * _qty(t) for t in (trades or [])
                               if isinstance(t, dict)]))
    if len(sizes) == 0:
        return {"trade_count": 0, "trade_mean_qty": 0.0, "trade_median_qty": 0.0,
                "trade_large_share": 0.0, "trade_notional": 0.0,
                "trade_size_skew": 0.0}
    finite = sizes[np.isfinite(sizes)]
    if len(finite) == 0:
        return {"trade_count": 0, "trade_mean_qty": 0.0, "trade_median_qty": 0.0,
                "trade_large_share": 0.0, "trade_notional": 0.0,
                "trade_size_skew": 0.0}
    total = float(finite.sum())
    cut = float(np.quantile(finite, float(large_quantile)))
    large = float(finite[finite >= cut].sum()) if total > 0 else 0.0
    sd = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
    mean = float(finite.mean())
    skew = 0.0
    if len(finite) > 2 and sd > 0:
        skew = float(np.mean(((finite - mean) / sd) ** 3))
    return {
        "trade_count": int(len(finite)), "trade_mean_qty": mean,
        "trade_median_qty": float(np.median(finite)),
        "trade_large_share": float(large / total) if total > 0 else 0.0,
        "trade_notional": notional, "trade_size_skew": skew,
    }


def realized_volatility(prices, *, stride: int = RV_TRADE_STRIDE,
                        annualize_bars: float | None = None) -> float:
    """Realized volatility ``sqrt(Σ Δln p²)`` over every ``stride``-th price.

    ``annualize_bars`` multiplies by ``sqrt(bars_per_year / stride)`` when the
    caller wants an annualised number; the default is the raw per-sample value,
    which is the honest object (it is exactly the quantity a barrier width or a
    position size should scale with).
    """
    p = np.asarray([float(x) for x in (prices if prices is not None else [])],
                   dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    k = max(int(stride), 1)
    if len(p) <= k:
        return float("nan")
    sampled = p[::k]
    r = np.diff(np.log(sampled))
    if len(r) == 0:
        return float("nan")
    value = float(math.sqrt(float(np.sum(r * r))))
    if annualize_bars:
        return float(value * math.sqrt(float(annualize_bars) / k))
    return value


def trade_arrival_intensity(trades, *, as_of_ms: float | None = None) -> dict:
    """Prints per second over the snapshot's span, plus recent/earlier ratio.

    ``activity_ratio`` compares the arrival rate of the newer half of the prints
    with the older half: ``> 1`` = the tape is speeding up.  Both halves are
    inside the same snapshot, so neither looks forward.
    """
    rows, _ = filter_trades(trades, as_of_ms=as_of_ms, limit=MAX_TRADES)
    times = []
    for t in rows:
        try:
            times.append(float(t.get("time")))
        except (TypeError, ValueError):
            continue
    if len(times) < 3:
        return {"arrival_rate_hz": 0.0, "activity_ratio": float("nan"),
                "span_seconds": 0.0}
    times = sorted(times)
    span = (times[-1] - times[0]) / 1000.0
    rate = (len(times) - 1) / span if span > 0 else 0.0
    mid = times[len(times) // 2]
    older = [t for t in times if t <= mid]
    newer = [t for t in times if t > mid]
    span_o = (older[-1] - older[0]) / 1000.0 if len(older) > 1 else 0.0
    span_n = (newer[-1] - newer[0]) / 1000.0 if len(newer) > 1 else 0.0
    rate_o = (len(older) - 1) / span_o if span_o > 0 else 0.0
    rate_n = (len(newer) - 1) / span_n if span_n > 0 else 0.0
    ratio = float(rate_n / rate_o) if rate_o > 0 else float("nan")
    return {"arrival_rate_hz": float(rate), "activity_ratio": ratio,
            "span_seconds": float(span)}


# ── the feature bundle ──────────────────────────────────────────────────

def compute_features(
    order_book: dict,
    trades,
    *,
    as_of_ms: float | None = None,
    book_time_ms: float | None = None,
    depth_limit: int = MAX_DEPTH_LEVELS,
    trades_limit: int = MAX_TRADES,
) -> dict:
    """The bounded microstructure feature bundle for one decision timestamp.

    ``as_of_ms`` is the decision time; **every** trade stamped after it is
    dropped (``dropped_future_trades``) before any statistic is computed.
    ``book_time_ms`` (the moment the book was fetched) makes ``book_age_ms``
    meaningful; passing it is what lets a consumer refuse a stale touch.
    """
    limit = int(min(max(int(depth_limit), 1), MAX_DEPTH_LEVELS))
    tlimit = int(min(max(int(trades_limit), 1), MAX_TRADES))
    kept, dropped = filter_trades(trades, as_of_ms=as_of_ms, limit=tlimit)
    ts = trade_size_stats(kept)
    arr = trade_arrival_intensity(kept)
    prices = [_price(t) for t in kept]
    bids = _levels((order_book or {}).get("bids"), limit)
    asks = _levels((order_book or {}).get("asks"), limit)
    out = {
        "mid": mid_price(order_book, limit=limit),
        "microprice": microprice(order_book, limit=limit),
        "microprice_dev_bps": microprice_deviation_bps(order_book, limit=limit),
        "spread_bps": spread_bps(order_book, limit=limit),
        "ofi_depth": order_flow_imbalance(order_book, limit=limit),
        "ofi_depth_weighted": depth_weighted_imbalance(order_book, limit=limit),
        "ofi_trades": trade_flow_imbalance(kept),
        "book_slope_ratio": book_slope_ratio(order_book, limit=limit),
        "trade_count": ts["trade_count"],
        "trade_mean_qty": ts["trade_mean_qty"],
        "trade_median_qty": ts["trade_median_qty"],
        "trade_large_share": ts["trade_large_share"],
        "trade_notional": ts["trade_notional"],
        "rv_trade": realized_volatility(prices),
        "arrival_rate_hz": arr["arrival_rate_hz"],
        "activity_ratio": arr["activity_ratio"],
        "as_of_ms": None if as_of_ms is None else float(as_of_ms),
        "book_age_ms": (float(as_of_ms) - float(book_time_ms)
                        if as_of_ms is not None and book_time_ms is not None
                        else None),
        "dropped_future_trades": int(dropped),
        "n_depth_levels": int(len(bids) + len(asks)),
    }
    return out


class MicrostructureCache:
    """Bounded TTL cache of **computed** feature dicts (never raw payloads).

    ``now`` is injectable so the TTL logic is testable without sleeping.
    """

    def __init__(self, ttl_secs: float = MICROSTRUCTURE_CACHE_TTL_SECS,
                 max_entries: int = MAX_CACHE_ENTRIES):
        self.ttl_secs = float(ttl_secs)
        self.max_entries = int(max_entries)
        self._store: dict[str, tuple[float, dict]] = {}

    def put(self, symbol: str, features: dict, *, now: float | None = None) -> None:
        now = time.time() if now is None else float(now)
        if len(self._store) >= self.max_entries and symbol not in self._store:
            oldest = min(self._store, key=lambda k: self._store[k][0])
            self._store.pop(oldest, None)
        self._store[str(symbol)] = (now, dict(features))

    def get(self, symbol: str, *, now: float | None = None) -> dict | None:
        now = time.time() if now is None else float(now)
        row = self._store.get(str(symbol))
        if row is None:
            return None
        stamp, features = row
        if now - stamp > self.ttl_secs:
            self._store.pop(str(symbol), None)
            return None
        return dict(features)

    def age(self, symbol: str, *, now: float | None = None) -> float | None:
        now = time.time() if now is None else float(now)
        row = self._store.get(str(symbol))
        return None if row is None else float(now - row[0])

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


async def fetch_features(
    client,
    symbol: str,
    *,
    cache: MicrostructureCache | None = None,
    depth_limit: int = MAX_DEPTH_LEVELS,
    trades_limit: int = 100,
    as_of_ms: float | None = None,
) -> dict | None:
    """Fetch a depth+trades snapshot and return its features (or ``None``).

    ``client`` is **any** object with ``order_book(symbol, limit)`` and
    ``recent_trades(symbol, limit)`` coroutines — in production
    ``core.market_data.data_client.MarketDataClient``, in tests a stub.  A
    transport failure returns ``None`` with ``last_error`` set: a missing
    microstructure snapshot must degrade to "no features", never to a crash of
    the signal path.
    """
    now_ms = float(as_of_ms) if as_of_ms is not None else time.time() * 1000.0
    if cache is not None:
        cached = cache.get(symbol)
        if cached is not None:
            return cached
    try:
        book = await client.order_book(symbol, limit=int(depth_limit))
        trades = await client.recent_trades(symbol, limit=int(trades_limit))
    except Exception as e:  # transport/parse failure — degrade, never raise
        fetch_features.last_error = f"{type(e).__name__}: {e}"
        return None
    fetch_features.last_error = None
    book_time_ms = time.time() * 1000.0
    feats = compute_features(book, trades, as_of_ms=now_ms,
                             book_time_ms=book_time_ms,
                             depth_limit=depth_limit, trades_limit=trades_limit)
    feats["symbol"] = symbol
    feats["source"] = "snapshot"
    if cache is not None:
        cache.put(symbol, feats)
    return feats


fetch_features.last_error = None  # type: ignore[attr-defined]


__all__ = [
    "MICROSTRUCTURE_ENABLED", "MICROSTRUCTURE_CACHE_TTL_SECS",
    "MAX_DEPTH_LEVELS", "MAX_TRADES", "MAX_CACHE_ENTRIES", "RV_TRADE_STRIDE",
    "FEATURE_KEYS", "best_bid_ask", "mid_price", "order_flow_imbalance",
    "depth_weighted_imbalance", "book_slope_ratio", "microprice",
    "microprice_deviation_bps", "spread_bps", "filter_trades",
    "aggressor_is_buy", "trade_flow_imbalance", "trade_size_stats",
    "realized_volatility", "trade_arrival_intensity", "compute_features",
    "MicrostructureCache", "fetch_features",
]
