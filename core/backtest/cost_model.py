"""Trading cost model — fees, spread slippage for realistic backtest PnL.

All costs are deducted at position close (round-trip) to avoid altering
entry prices, which would cascade into stop-loss and take-profit calculations.

Spread resolution (why this module owns it)
-------------------------------------------
``config.backtest_spread_pct`` used to *be* the table of spreads: five hardcoded
pairs plus a silent ``0.03`` constant for everything else, so backtesting any
other pair charged a made-up number.  The spread is now **derived per symbol**,
with this precedence:

1. **override** — an explicit per-symbol entry, either handed in by the caller
   (``overrides=``, e.g. the values the user typed in the backtest/GA form) or
   configured under ``backtest.cost_model.spread_pct``.  A ``default``/``*`` key
   in either map is an explicit override for every symbol.
2. **live** — derived from the public data host's order book
   (``GET {market_data_host}/api/v3/depth?symbol=X&limit=5``) as
   ``(best_ask - best_bid) / mid * 100``, cached for
   ``backtest.cost_model.live_spread.ttl_seconds`` (default 300 s).  A failed
   fetch is *never* cached, so a transient outage cannot be remembered for the
   whole window.
3. **default** — ``backtest.cost_model.default_spread_pct`` (0.03 %), the
   documented last resort for a pair with neither an override nor a live quote.

It lives in ``core/backtest/cost_model.py`` and not in
``core/market_data/metrics.py`` because the precedence rule is a *backtest cost*
rule: :func:`apply_trading_costs` is its only consumer and every key it reads
lives under ``backtest.cost_model``.  ``metrics.py`` is a pure-maths module (no
I/O) that the market routes import; a cached HTTP fetch does not belong there.
The fetch is deliberately **synchronous** (stdlib ``urllib``) because the
backtest engine runs in a worker thread / subprocess with no event loop.

Live derivation is opt-in per config object: :class:`app.config.Config` always
carries ``backtest_live_spread_enabled`` (true unless disabled in YAML), while a
duck-typed test double that simply lacks the attribute is treated as
live-disabled and resolves straight to override/default.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error  # noqa: F401 - kept so callers can name the failure type
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

#: Last-resort spread (%), used when neither an override nor a live quote exists.
DEFAULT_SPREAD_PCT = 0.03

#: Live-derivation defaults (overridable under ``backtest.cost_model.live_spread``).
LIVE_SPREAD_TTL = 300.0
LIVE_SPREAD_TIMEOUT = 3.0

#: Order-book levels requested from ``/api/v3/depth`` (best bid/ask only matter).
DEPTH_LIMIT = 5

#: How many symbols may be resolved concurrently when a run resolves many pairs.
LIVE_SPREAD_CONCURRENCY = 8

SOURCE_OVERRIDE = "override"
SOURCE_LIVE = "live"
SOURCE_DEFAULT = "default"

#: symbol -> (monotonic stamp, spread %).  Only *successful* lookups are stored.
_live_cache: dict[str, tuple[float, float]] = {}
_live_lock = threading.Lock()


# ----------------------------------------------------------------------
# conversion helpers
# ----------------------------------------------------------------------
def _to_float(value):
    """``float(value)`` or ``None`` (never raises, never returns NaN/inf)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _live_enabled(config) -> bool:
    """True when live derivation is switched on for this config object."""
    return bool(getattr(config, "backtest_live_spread_enabled", False))


def default_spread_pct(config=None) -> float:
    """The documented fallback spread (%) — ``backtest_default_spread_pct``."""
    value = _to_float(getattr(config, "backtest_default_spread_pct", None))
    if value is None or value < 0:
        return DEFAULT_SPREAD_PCT
    return value


def _data_host(config) -> str:
    return (getattr(config, "backtest_market_data_host", None)
            or getattr(config, "market_data_host", "") or "")


# ----------------------------------------------------------------------
# live derivation (best bid/ask → spread %)
# ----------------------------------------------------------------------
def spread_pct_from_depth(payload: dict) -> float | None:
    """``(best_ask - best_bid) / mid * 100`` from a Binance depth payload."""
    if not isinstance(payload, dict):
        return None
    bids = payload.get("bids") or []
    asks = payload.get("asks") or []
    if not bids or not asks:
        return None
    try:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
    except (TypeError, ValueError, IndexError):
        return None
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return None
    mid = (best_ask + best_bid) / 2.0
    if mid <= 0:
        return None
    return (best_ask - best_bid) / mid * 100.0


def fetch_live_spread_pct(symbol: str, host: str,
                          timeout: float = LIVE_SPREAD_TIMEOUT) -> float | None:
    """One order-book round-trip; ``None`` on any failure (never raises)."""
    host = str(host or "").rstrip("/")
    symbol = str(symbol or "").strip().upper()
    if not host or not symbol:
        return None
    query = urllib.parse.urlencode({"symbol": symbol, "limit": DEPTH_LIMIT})
    url = f"{host}/api/v3/depth?{query}"
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "binance-trader/1.0 (backtest cost model)"})
        with urllib.request.urlopen(request, timeout=max(0.5, float(timeout))) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return spread_pct_from_depth(payload)


def _cache_ttl(config) -> float:
    value = _to_float(getattr(config, "backtest_live_spread_ttl", None))
    if value is None or value <= 0:
        return LIVE_SPREAD_TTL
    return value


def peek_live_spread(symbol: str, config=None, *, now=None) -> float | None:
    """Cached live spread for ``symbol`` when still fresh (no network I/O)."""
    sym = str(symbol or "").strip().upper()
    stamp = time.monotonic() if now is None else float(now)
    with _live_lock:
        hit = _live_cache.get(sym)
    if hit is None or (stamp - hit[0]) >= _cache_ttl(config):
        return None
    return hit[1]


def live_spread_pct(symbol: str, config=None, *, host=None,
                    now=None) -> float | None:
    """Live spread (%) for one symbol, cached ~5 min; ``None`` when unavailable."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    cached = peek_live_spread(sym, config, now=now)
    if cached is not None:
        return cached
    timeout = _to_float(getattr(config, "backtest_live_spread_timeout", None))
    value = fetch_live_spread_pct(
        sym, host or _data_host(config),
        LIVE_SPREAD_TIMEOUT if timeout is None else timeout)
    if value is None:
        return None  # failures are never cached
    stamp = time.monotonic() if now is None else float(now)
    with _live_lock:
        _live_cache[sym] = (stamp, value)
    return value


def clear_live_spread_cache(symbol: str | None = None) -> None:
    """Drop one symbol (or the whole live cache) — used by tests and callers."""
    with _live_lock:
        if symbol is None:
            _live_cache.clear()
        else:
            _live_cache.pop(str(symbol).strip().upper(), None)


# ----------------------------------------------------------------------
# resolution: override → live → default
# ----------------------------------------------------------------------
def override_map(config=None, overrides=None) -> dict:
    """Merge ``config.backtest_spread_pct`` with the caller's ``overrides``.

    Keys are upper-cased; non-numeric / negative entries are dropped.  Caller
    values win over the config file (a run's explicit choice beats the default
    table), and ``default`` / ``*`` are kept as the explicit fallback key.
    """
    table: dict[str, float] = {}
    for source in (getattr(config, "backtest_spread_pct", None), overrides):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            number = _to_float(value)
            if number is None or number < 0:
                continue
            table[str(key).strip().upper()] = number
    return table


def _override_for(symbol: str, table: dict) -> float | None:
    if symbol and symbol in table:
        return table[symbol]
    if "DEFAULT" in table:
        return table["DEFAULT"]
    if "*" in table:
        return table["*"]
    return None


def resolve_spread_pct(symbol: str, config=None, overrides=None, *,
                       use_live: bool = True,
                       now=None) -> tuple[float, str]:
    """Resolve one symbol to ``(spread_pct, source)``.

    ``source`` is one of ``"override"`` / ``"live"`` / ``"default"`` (the strings
    the UI shows).  Never raises and never returns ``None``.
    """
    sym = str(symbol or "").strip().upper()
    table = override_map(config, overrides)
    explicit = _override_for(sym, table)
    if explicit is not None:
        return explicit, SOURCE_OVERRIDE
    if use_live and _live_enabled(config):
        value = live_spread_pct(sym, config, now=now)
        if value is not None:
            return value, SOURCE_LIVE
    return default_spread_pct(config), SOURCE_DEFAULT


def resolve_spreads(symbols, config=None, overrides=None, *,
                    use_live: bool = True, concurrency: int = LIVE_SPREAD_CONCURRENCY,
                    now=None) -> dict:
    """Resolve many symbols at once → ``{symbol: {"spread_pct", "source", ...}}``.

    Overrides and fresh cache hits are answered without I/O; the remaining
    symbols are looked up concurrently (bounded by ``concurrency``) so a 20-pair
    selection does not serialise 20 order-book round-trips.
    """
    order: list[str] = []
    for raw in symbols or []:
        sym = str(raw or "").strip().upper()
        if sym and sym not in order:
            order.append(sym)

    table = override_map(config, overrides)
    resolved: dict[str, tuple[float, str]] = {}
    pending: list[str] = []
    for sym in order:
        explicit = _override_for(sym, table)
        if explicit is not None:
            resolved[sym] = (explicit, SOURCE_OVERRIDE)
            continue
        if use_live and _live_enabled(config):
            cached = peek_live_spread(sym, config, now=now)
            if cached is not None:
                resolved[sym] = (cached, SOURCE_LIVE)
                continue
            pending.append(sym)
            continue
        resolved[sym] = (default_spread_pct(config), SOURCE_DEFAULT)

    if pending:
        workers = max(1, min(int(concurrency or 1), len(pending)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched = list(pool.map(
                lambda sym: live_spread_pct(sym, config, now=now), pending))
        for sym, value in zip(pending, fetched):
            if value is None:
                resolved[sym] = (default_spread_pct(config), SOURCE_DEFAULT)
            else:
                resolved[sym] = (value, SOURCE_LIVE)

    out: dict[str, dict] = {}
    for sym in order:
        value, source = resolved[sym]
        out[sym] = {"symbol": sym, "spread_pct": value, "source": source}
    return out


def freeze_run_spreads(symbols, config=None, overrides=None, **kwargs) -> dict:
    """``{symbol: spread %}`` for one run.

    The engine pins this map at the start of a run so the trade loop resolves
    costs from an in-memory dict instead of doing I/O per closed trade.
    """
    return {sym: entry["spread_pct"]
            for sym, entry in resolve_spreads(symbols, config, overrides,
                                              **kwargs).items()}


# ----------------------------------------------------------------------
# the cost itself (unchanged formula: fees + half-spread, round trip)
# ----------------------------------------------------------------------
def apply_trading_costs(entry_price: float, exit_price: float, qty: float,
                        symbol: str, config, overrides=None,
                        spread_pct: float | None = None, *,
                        recent_quote_volume: float | None = None,
                        impact_pct_override: float | None = None) -> float:
    """Calculate total round-trip trading cost for a position.

    Components:
      - Taker fee on entry notional
      - Taker fee on exit notional
      - Half-spread slippage on entry (buy at ask, sell at bid)
      - Half-spread slippage on exit
      - **Market impact** on entry and on exit (P6-A, off by default)

    The spread (%) comes from :func:`resolve_spread_pct` — explicit override →
    live depth-derived → documented default — unless the caller pins it via
    ``spread_pct`` or ``overrides``.  The PnL formula itself is unchanged.

    Returns the total cost to subtract from trade PnL.
    Returns 0 if cost model is disabled in config.

    Args:
        entry_price: Position entry price per unit.
        exit_price: Position exit price per unit.
        qty: Trade quantity.
        symbol: Trading pair (e.g. "BTCUSDT").
        config: Config instance with backtest_cost_enabled etc.
        overrides: Optional per-symbol spread map (%) for this run.
        spread_pct: Optional pre-resolved spread (%) — skips resolution.
        recent_quote_volume: Quote (USDT) notional traded over the recent window
            — e.g. ``core.risk.liquidity.recent_quote_volume`` over the signal's
            ``risk.liquidity.lookback_bars``.  **Optional**: ``None`` (the
            default, and what every pre-P6 caller passes by omission) means "the
            window is unknown", and then the impact term contributes exactly
            ``0.0``.  This is the injection point that keeps this module free of
            market-data I/O.
        impact_pct_override: Pre-computed round-trip impact in **percent of
            entry notional**, for a caller that already priced participation
            (e.g. a report recomputing one trade).  Wins over
            ``recent_quote_volume`` when both are given.

    Impact (P6-A)
    -------------
    ``impact_pct = k · participation**e`` per side, in percent of that side's
    notional (``participation`` is the fraction of ``recent_quote_volume``:
    ``0.01`` = 1 %), charged on entry AND on exit, with ``k`` =
    ``risk.liquidity.impact_k`` (default ``0.0``) and ``e`` =
    ``risk.liquidity.impact_exponent`` (default ``0.5``, the square-root law:
    Almgren & Chriss 2000; Grinold & Kahn ch. 16).  See
    :func:`core.risk.liquidity.impact_pct` for the units and
    ``docs/core-algorithms/13-volume-liquidity-costs.md`` for measured examples.

    **The ``k <= 0`` (or no-volume) path short-circuits to the legacy sum**, so
    the shipped default is bit-identical to the pre-P6 cost — the identity is
    pinned by ``tests/test_liquidity.py::test_impact_k_zero_is_bit_identical``.
    """
    if not getattr(config, 'backtest_cost_enabled', True):
        return 0.0

    fee_pct = getattr(config, 'backtest_taker_fee_pct', 0.04) / 100.0
    if spread_pct is None:
        spread_pct, _source = resolve_spread_pct(symbol, config, overrides)
    spread = _to_float(spread_pct)
    if spread is None or spread < 0:
        spread = default_spread_pct(config)
    spread /= 100.0

    entry_notional = qty * entry_price
    exit_notional = qty * exit_price

    entry_fee = entry_notional * fee_pct
    exit_fee = exit_notional * fee_pct
    entry_spread = entry_notional * (spread / 2.0)
    exit_spread = exit_notional * (spread / 2.0)

    legacy = entry_fee + exit_fee + entry_spread + exit_spread

    # ── Market impact (P6-A) — OPT-IN, and a no-op on every shipped path ──
    if impact_pct_override is not None:
        override = _to_float(impact_pct_override)
        if override is None or override <= 0.0:
            return legacy
        return legacy + entry_notional * (override / 100.0)

    k, exponent = _liquidity_impact_params(config)
    if k <= 0.0 or recent_quote_volume is None:
        return legacy

    volume = _to_float(recent_quote_volume)
    if volume is None or volume <= 0.0:
        return legacy
    from core.risk.liquidity import total_impact_usdt

    return legacy + total_impact_usdt(entry_notional, exit_notional, volume,
                                      k, exponent)


def _liquidity_impact_params(config) -> tuple[float, float]:
    """``(impact_k, impact_exponent)`` from ``risk.liquidity``; ``(0.0, 0.5)`` if absent.

    A duck-typed config object without the block is treated as impact-off, which
    is what every pre-P6 config double and every test double is.
    """
    from core.risk.liquidity import (DEFAULT_IMPACT_EXPONENT, DEFAULT_IMPACT_K,
                                     resolve_liquidity_config)

    block = resolve_liquidity_config(config)
    if block is None:
        return DEFAULT_IMPACT_K, DEFAULT_IMPACT_EXPONENT
    k = _to_float(getattr(block, "impact_k", DEFAULT_IMPACT_K))
    exponent = _to_float(getattr(block, "impact_exponent", DEFAULT_IMPACT_EXPONENT))
    return (DEFAULT_IMPACT_K if k is None else k,
            DEFAULT_IMPACT_EXPONENT if exponent is None else exponent)


def impact_cost_usdt(entry_notional: float, exit_notional: float,
                     recent_quote_volume: float, k: float,
                     exponent: float = 0.5) -> float:
    """USDT impact charge for one closed trade (both sides) — thin re-export.

    Lives in :mod:`core.risk.liquidity` (the maths module); this wrapper exists so
    a backtest report can name a cost-model function without importing the risk
    package, and so ``tests/test_liquidity.py`` can compare the cost model's
    number with the helper's directly.  ``0.0`` when the term is off
    (``k <= 0``), the window is unknown, or the notional is non-positive.
    """
    if k is None or _to_float(k) is None or float(k) <= 0.0:
        return 0.0
    volume = _to_float(recent_quote_volume)
    if volume is None or volume <= 0.0:
        return 0.0
    from core.risk.liquidity import total_impact_usdt

    return total_impact_usdt(entry_notional, exit_notional, volume, k, exponent)


def total_costs_with_impact(entry_price: float, exit_price: float, qty: float,
                            symbol: str, config, overrides=None,
                            spread_pct: float | None = None, *,
                            recent_quote_volume: float | None = None) -> dict:
    """Split one trade's costs into the reported components — fees, spread, impact.

    ``{"fees_usdt", "spread_usdt", "impact_usdt", "total_usdt", "impact_pct",
    "recent_quote_volume", "impact_k"}``.  A backtest report should show
    ``impact_usdt`` next to the other two rather than silently folding it in:
    the whole point of P6-A is that a large size's cost is *visible*
    (``docs/core-algorithms/13-volume-liquidity-costs.md`` §4).  Pure arithmetic
    over :func:`apply_trading_costs`, so it cannot drift from what the run paid.
    """
    legacy = apply_trading_costs(entry_price, exit_price, qty, symbol, config,
                                overrides, spread_pct)
    total = apply_trading_costs(entry_price, exit_price, qty, symbol, config,
                                overrides, spread_pct,
                                recent_quote_volume=recent_quote_volume)
    entry_notional = float(qty) * float(entry_price)
    exit_notional = float(qty) * float(exit_price)
    fee_pct = (_to_float(getattr(config, 'backtest_taker_fee_pct', 0.04)) or 0.0) / 100.0
    if spread_pct is None:
        spread_pct, _source = resolve_spread_pct(symbol, config, overrides)
    spread = _to_float(spread_pct)
    if spread is None or spread < 0:
        spread = default_spread_pct(config)
    fees = (entry_notional + exit_notional) * fee_pct
    spread_cost = (entry_notional + exit_notional) * (spread / 200.0)
    impact = max(0.0, total - legacy)
    k, _exponent = _liquidity_impact_params(config)
    return {
        "fees_usdt": fees,
        "spread_usdt": spread_cost,
        "impact_usdt": impact,
        "total_usdt": total,
        "legacy_usdt": legacy,
        "impact_pct": (impact / entry_notional * 100.0) if entry_notional > 0 else 0.0,
        "recent_quote_volume": _to_float(recent_quote_volume),
        "impact_k": k,
    }

