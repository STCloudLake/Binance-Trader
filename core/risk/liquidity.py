"""Volume-aware liquidity: participation caps and market-impact cost (P6-A).

Why this module exists
----------------------
An audit of the volume features found that the strategy stack consumes volume
only as a *coarse relative* feature (volume vs its own rolling mean — a spike
detector).  The highest-value use of volume in this project is not directional at
all: it is **cost and capacity** (Kyle 1985; Almgren & Chriss 2000; Grinold &
Kahn, *Active Portfolio Management*, ch. 16 "The Costs of Trading").  Two facts
follow, and both are pure risk controls — neither claims any predictive edge:

1. **Participation.**  An order is a *fraction* of what the market traded.  A
   notional that is 30 % of the last hour's traded notional cannot be filled at
   the quoted price; a notional that is 0.01 % of it can.  Refusing (or shrinking)
   an order above a participation ceiling is a hard risk control that needs no
   forecast.
2. **Impact.**  The cost model charged ``spread/2 + slippage`` regardless of size,
   so a backtest of a large size was silently optimistic.  A square-root impact
   term ``impact_pct = k · participation**e`` (the standard empirical shape,
   e = 0.5) makes a large order pay for its own size.

Wire map
--------
* :func:`recent_quote_volume` / :func:`cap_notional` — read by
  :meth:`core.risk.position_sizer.PositionSizer.apply_participation_cap`, which
  is called from :meth:`PositionSizer.calculate_position_size` **only** when
  ``risk.liquidity.enabled`` is true (default **false**).
* :func:`impact_pct` / :func:`notional_series` — read by
  :func:`core.backtest.cost_model.apply_trading_costs` when the caller supplies
  ``recent_quote_volume`` **and** ``risk.liquidity.impact_k`` > 0 (default
  ``0.0``, i.e. the pre-P6 arithmetic).

Units (one meaning each, stated once)
-------------------------------------
============================  ===================================================
quantity                      unit
============================  ===================================================
``recent_quote_volume``       **quote currency (USDT)** — Σ ``volume · price`` over
                              the last ``lookback_bars`` bars.  ``volume`` is
                              Binance base-asset volume; ``price`` is the bar's
                              close (or ``quote_volume`` when the frame has one).
``participation``             **fraction** of that volume: 0.01 = 1 %.
``max_participation_pct``     **percent** (the config unit): 1.0 = 1 %.
``impact_pct``                **percent of notional**, per side (0.05 = 5 bp).
``k`` (``impact_k``)          dimensionless scale of the impact law.
``floor`` / ``cap``           percent, same unit as ``impact_pct``.
============================  ===================================================

Why percent and not a fraction for ``participation_pct``: ``lenient=False``
(config) mode *refuses* a trade when the volume window is unknown or empty, and
``None`` is the only value that can express "unknown" — ``0.0`` would read as
"no volume traded", which is the opposite of "we could not measure it".

Cost: pure Python, no I/O, no pandas import at module scope.  A pandas/numpy
frame or Series is duck-typed through ``tolist()`` only when one is actually
passed, so a per-signal call is a list slice plus arithmetic —
``tests/test_liquidity.py::test_helper_call_cost_is_bounded`` measures the cost
(asserted < 200 µs/call for ``cap_notional`` on a 20-bar window).
"""

from __future__ import annotations

import math

__all__ = [
    "LiquidityConfig",
    "DEFAULT_LOOKBACK_BARS",
    "DEFAULT_MAX_PARTICIPATION_PCT",
    "DEFAULT_IMPACT_K",
    "DEFAULT_IMPACT_EXPONENT",
    "resolve_liquidity_config",
    "liquidity_for_symbol",
    "recent_quote_volume",
    "participation_pct",
    "cap_notional",
    "impact_pct",
    "notional_series",
    "trade_impact_pct",
    "total_impact_usdt",
]

#: Bars of quote volume behind one participation decision (config default).
DEFAULT_LOOKBACK_BARS = 20

#: Participation ceiling as a percent of the window (config default, e.g. 1 %).
DEFAULT_MAX_PARTICIPATION_PCT = 1.0

#: Impact-law coefficient.  **0.0 is the shipped default → no impact cost.**
DEFAULT_IMPACT_K = 0.0

#: Square-root law (Almgren-Chriss / Grinold-Kahn square-root shape).
DEFAULT_IMPACT_EXPONENT = 0.5

#: Keys a per-symbol override map may use for "every other symbol".
_DEFAULT_KEYS = ("default", "DEFAULT", "*")


def _finite(value, default: float = 0.0) -> float:
    """``float(value)`` when finite, else ``default`` (never raises, never NaN)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _number_or_none(value):
    """``float(value)`` when finite, else ``None``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class LiquidityConfig:
    """Duck-typed view of the ``risk.liquidity`` block (``app.config`` owns the model).

    This class exists so the helpers work on a *plain* object — a test double, a
    backtest config stand-in — without importing ``app.config`` (which would make
    a pure-maths module depend on the app layer).  Every field mirrors
    ``app.config.LiquidityConfig`` one-for-one, including the defaults, so
    "no config object at all" and "shipped config" are the same thing:
    **disabled, no impact, 1 % ceiling, 20 bars**.
    """

    __slots__ = ("enabled", "max_participation_pct", "lookback_bars",
                 "impact_k", "impact_exponent", "per_symbol")

    def __init__(self, enabled: bool = False,
                 max_participation_pct: float = DEFAULT_MAX_PARTICIPATION_PCT,
                 lookback_bars: int = DEFAULT_LOOKBACK_BARS,
                 impact_k: float = DEFAULT_IMPACT_K,
                 impact_exponent: float = DEFAULT_IMPACT_EXPONENT,
                 per_symbol: dict | None = None):
        self.enabled = bool(enabled)
        self.max_participation_pct = float(max_participation_pct)
        self.lookback_bars = int(lookback_bars)
        self.impact_k = float(impact_k)
        self.impact_exponent = float(impact_exponent)
        self.per_symbol = dict(per_symbol or {})

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"LiquidityConfig(enabled={self.enabled}, "
                f"max_participation_pct={self.max_participation_pct}, "
                f"lookback_bars={self.lookback_bars}, "
                f"impact_k={self.impact_k}, "
                f"impact_exponent={self.impact_exponent})")


def resolve_liquidity_config(value=None):
    """Best-effort ``risk.liquidity`` block from whatever object the caller has.

    Resolution order:

    1. ``value`` itself when it already looks like a liquidity block (it has
       ``max_participation_pct``);
    2. ``value.risk_liquidity`` (``app.config.Config`` carries the block under
       that name);
    3. ``app.config.Config.load().risk_liquidity`` — the lazy import keeps this
       module import-light and cycle-free, and it is why a *caller* never has to
       thread the config through: with no argument at all, the shipped block is
       used;
    4. ``None`` — "not configured", which every helper treats as the shipped
       defaults (disabled / no impact).

    ``None`` is never an error: the P6 default is *off*, so an unknown config
    must behave exactly like ``enabled: false``.
    """
    if value is not None and hasattr(value, "max_participation_pct"):
        return value
    block = getattr(value, "risk_liquidity", None) if value is not None else None
    if block is not None:
        return block
    if value is None:
        try:  # pragma: no cover - exercised through Config.load in production
            from app.config import Config

            return getattr(Config.load(), "risk_liquidity", None)
        except Exception:
            return None
    return None


def liquidity_for_symbol(config, symbol: str):
    """The effective block for ``symbol``: per-symbol override → ``default``/``*`` → block.

    Per-symbol entries are *partial* dicts (``{"BTCUSDT": {"max_participation_pct":
    2.0}}``); a missing key falls back to the top-level value, so an override can
    change one number without restating the rest.
    """
    if config is None:
        return None
    if not symbol:
        return config
    overrides = getattr(config, "per_symbol", None) or {}
    if not isinstance(overrides, dict) or not overrides:
        return config
    entry = overrides.get(str(symbol).strip().upper())
    if entry is None:
        for key in _DEFAULT_KEYS:
            if key in overrides:
                entry = overrides[key]
                break
    if entry is None:
        return config
    if not isinstance(entry, dict):
        return config
    merged = LiquidityConfig(
        enabled=getattr(config, "enabled", False),
        max_participation_pct=getattr(config, "max_participation_pct",
                                      DEFAULT_MAX_PARTICIPATION_PCT),
        lookback_bars=getattr(config, "lookback_bars", DEFAULT_LOOKBACK_BARS),
        impact_k=getattr(config, "impact_k", DEFAULT_IMPACT_K),
        impact_exponent=getattr(config, "impact_exponent", DEFAULT_IMPACT_EXPONENT),
        per_symbol=overrides)
    for key, raw in entry.items():
        if key in LiquidityConfig.__slots__:
            setattr(merged, key, raw)
    return merged


# ----------------------------------------------------------------------
# series extraction (duck-typed: list / numpy array / pandas Series|DataFrame)
# ----------------------------------------------------------------------
def _as_list(value):
    """1-D list of floats from a list / ndarray / Series / column name holder.

    ``None`` for anything that is not a sequence (so callers can tell "absent"
    from "empty").
    """
    if value is None:
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        return None
    out = []
    for item in value:
        out.append(_finite(item, float("nan")))
    return out


def _series(frame):
    """``(values, source)`` for a frame-like object, or ``(None, None)``.

    The frame check comes **first**: a pandas DataFrame is iterable (over its
    column names) and has no ``tolist``, so a naive ``_as_list`` would hand back
    strings or nothing.  Only a 2-D frame-like object is treated as a frame.
    """
    if frame is None:
        return None, None
    if isinstance(frame, dict):
        return None, None
    is_frame = (hasattr(frame, "columns") and hasattr(frame, "__getitem__")
                and hasattr(frame, "shape"))
    if is_frame:
        if "quote_volume" in list(frame.columns):
            quoted = _as_list(frame["quote_volume"])
            # P6-B: a cache file may carry the column but hold only NaN (a bar
            # whose backfill has not run yet).  Falling through to the proxy is
            # the documented behaviour for "the column has no usable number" —
            # returning `"quote_volume"` with an all-NaN list would make every
            # window read 0.0 = "unknown volume", which refuses orders that the
            # proxy can size honestly.
            if quoted is not None and any(math.isfinite(v) for v in quoted):
                return quoted, "quote_volume"
        if "volume" in list(frame.columns):
            if "close" in list(frame.columns):
                return _as_list(frame["volume"]), "volume*close"
            return _as_list(frame["volume"]), "volume"
        return None, None
    values = _as_list(frame)
    if values is None:
        return None, None
    return values, "volume"


def _mapping_values(mapping, keys):
    """First present key of ``keys`` in ``mapping`` → ``(values, key)``."""
    if not isinstance(mapping, dict):
        return None, None
    for key in keys:
        values = _as_list(mapping.get(key))
        if values is not None:
            return values, key
    return None, None


def recent_quote_volume(bars=None, lookback_bars: int = DEFAULT_LOOKBACK_BARS,
                        provider=None, *, frame=None,
                        price_col: str = "close") -> float:
    """Quote-currency (USDT) volume over the last ``lookback_bars`` bars.

    Accepts, in order:

    * ``provider`` — any callable returning the volume.  Called first, so an
      injected source (a live kline cache, a test double) always wins;
    * a **pandas/numpy frame** — uses ``quote_volume`` when the column exists,
      else Σ ``volume × price_col`` (``price_col`` defaults to ``close``; an
      explicit column that is absent raises);
    * a **mapping** with ``quote_volume`` / ``volume`` keys;
    * a **1-D sequence** of volumes — then the returned figure is base-volume ×
      nothing, so pass ``frame=`` (or a frame) when a price conversion is needed.

    ``lookback_bars`` bounds the lookback (``<= 0`` means "all of it"); the
    **most recent** bars are used, i.e. the tail of the series.

    A frame whose ``quote_volume`` column is entirely absent **or entirely
    non-finite** (P6-B: a cache file whose backfill has not run yet) falls back to
    the documented ``Σ volume × close`` proxy.  A frame with only *some* bars
    quoted sums the quoted ones and drops the rest — a hole is reported as "not
    measured" (fewer bars in the sum) rather than being filled with a proxy that
    would look like data.

    Returns ``0.0`` — never NaN, never an exception — when there is no data or
    the window is empty.  ``0.0`` is the documented "unknown volume" signal:
    :func:`participation_pct` maps it to ``None``, :func:`cap_notional` then
    refuses the order (``"no_volume"``), and the "no measured volume ⇒ no impact"
    rule falls out of ``participation is None``.
    """
    if provider is not None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        try:
            value = provider(lookback_bars)
        except TypeError:
            value = provider()
        return max(0.0, _finite(value, 0.0))

    values = None
    source = None
    # The object the values came from — ``bars`` and ``frame=`` are interchangeable,
    # so the price column must be read from whichever one supplied the volumes.
    origin = None
    if frame is not None:
        values, source = _series(frame)
        if values is None:
            values, source = _mapping_values(frame, ("quote_volume", "volume"))
        if values is not None:
            origin = frame
    if values is None and bars is not None:
        values, source = _series(bars)
        if values is None:
            values, source = _mapping_values(bars, ("quote_volume", "volume"))
        if values is not None:
            origin = bars
    if values is None:
        return 0.0

    window = int(lookback_bars) if lookback_bars else 0
    if window > 0:
        values = values[-window:]
    # A non-finite bar (a gap, a bad tick) is dropped rather than poisoning the
    # whole window with NaN: the sum stays a number the caller can branch on.
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return 0.0

    if source == "volume*close":
        closes = _as_list(origin["close"]) if origin is not None else None
        if origin is not None and price_col != "close":
            columns = list(getattr(origin, "columns", []))
            if price_col not in columns:
                raise KeyError(
                    f"price_col '{price_col}' is not a column of the frame "
                    f"({columns})")
            closes = _as_list(origin[price_col])
        if closes is None:
            return 0.0
        # Trim the price series to the SAME tail window as the volumes, so a
        # 20-bar lookback never pairs bar prices with a full history.
        closes = closes[-len(values):]
        if len(closes) != len(values):
            return 0.0
        return max(0.0, _finite(math.fsum(v * c for v, c in zip(values, closes)), 0.0))

    return max(0.0, _finite(math.fsum(values), 0.0))


# ----------------------------------------------------------------------
# participation
# ----------------------------------------------------------------------
def participation_pct(notional: float, recent_quote_volume: float):
    """``notional / recent_quote_volume × 100`` — ``None`` when undefined.

    ``None`` (not ``0.0``) for an unknown/empty volume window (``<= 0``) or a
    non-finite/negative notional: "we cannot measure participation" is a
    different statement from "participation is zero", and the caller is the only
    one that can decide which way to fail.  A *negative* notional is treated as
    its magnitude (``abs``) because a short is the same liquidity event as a long
    of the same size.
    """
    volume = _number_or_none(recent_quote_volume)
    size = _number_or_none(notional)
    if volume is None or volume <= 0.0:
        return None
    if size is None:
        return None
    return abs(size) / volume * 100.0


def cap_notional(notional: float, recent_quote_volume: float,
                 max_participation_pct: float) -> tuple[float, str]:
    """Cap ``notional`` so it is at most ``max_participation_pct`` of the window.

    Returns ``(allowed_notional, reason)``:

    * ``("ok", …)`` — the order fits; the value returned is the **unchanged**
      input (bit-identical to the pre-P6 path, so a pass-through cannot perturb
      a single float);
    * ``("capped", …)`` — the order exceeded the ceiling and was **shrunk** to
      ``max_participation_pct/100 × recent_quote_volume``;
    * ``("no_volume", …)`` — the window is unknown/empty.  The order is refused
      (``0.0``) because a participation ceiling that silently passes when the
      denominator is missing is not a ceiling — for a *refusal* in the backtest
      or a low-liquidity pair, failing closed is the conservative choice;
    * ``("disabled", …)`` — the ceiling is ``<= 0`` ("no participation policy").
      Pass-through, unchanged.

    The reason string is logged at debug level by the caller
    (:meth:`PositionSizer.apply_participation_cap`) and is the free-text the UI /
    audit trail can show; ``tests/test_liquidity.py`` pins the exact prefixes.
    The function **never silently drops an order**: it either returns the input
    unchanged or returns the capped value *with* the reason that says so.

    Pure and allocation-light: one division, one ``min``.  ``notional <= 0`` is
    passed straight through (there is nothing to cap).
    """
    size = _number_or_none(notional)
    if size is None or size <= 0.0:
        return _finite(notional, 0.0), "ok: non-positive notional"
    ceiling = _number_or_none(max_participation_pct)
    if ceiling is None or ceiling <= 0.0:
        return float(size), "disabled: max_participation_pct <= 0"
    volume = _number_or_none(recent_quote_volume)
    if volume is None or volume <= 0.0:
        return 0.0, ("no_volume: recent quote volume is unknown/empty — order "
                     "refused rather than filled against an unmeasured book")

    allowed = volume * ceiling / 100.0
    if size <= allowed:
        return float(size), (f"ok: {size / volume * 100.0:.4f}% of "
                             f"{volume:.2f} USDT window <= {ceiling:g}%")
    return float(allowed), (f"capped: participation {size / volume * 100.0:.4f}% "
                            f"> {ceiling:g}% of {volume:.2f} USDT window — "
                            f"notional shrunk to {allowed:.2f}")


# ----------------------------------------------------------------------
# impact
# ----------------------------------------------------------------------
def impact_pct(participation, k: float, exponent: float = DEFAULT_IMPACT_EXPONENT,
               floor: float = 0.0, cap: float | None = None):
    """``clip(k · participation**exponent, floor, cap)`` in **percent of notional**.

    ``participation`` is a *fraction* (0.01 = 1 %); ``k`` is the dimensionless
    coefficient; the result is the per-side cost in percent, so a notional of
    ``N`` pays ``N · impact_pct/100`` on that side.  The default exponent ``0.5``
    is the square-root law (Almgren & Chriss 2000; Grinold & Kahn ch. 16): impact
    grows with size but sub-linearly, which is why doubling an order costs less
    than twice as much.

    ``k <= 0`` returns ``0.0`` (the pre-P6 cost model — no impact term).  An
    unknown participation (``None``), a non-positive one, or a negative ``k``
    also return ``0.0``: no volume ⇒ no measured impact, and a *negative*
    coefficient is a typo, not a subsidy.  ``floor`` bounds it from below (use it
    to model a minimum cost, e.g. a taker fee already covered elsewhere) and
    ``cap`` bounds it from above (``None`` = uncapped).  Scalars or arrays: an
    ndarray/Series ``participation`` comes back as a list, so the function is
    vectorisable for a whole signal batch.
    """
    if k is None:
        return 0.0
    coefficient = _number_or_none(k)
    if coefficient is None or coefficient <= 0.0:
        return 0.0
    lo = max(0.0, _finite(floor, 0.0))
    hi = None
    if cap is not None:
        hi = _number_or_none(cap)
        if hi is not None and hi < lo:
            hi = lo

    values = participation
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    if isinstance(values, (list, tuple)):
        return [_impact_one(item, coefficient, exponent, lo, hi) for item in values]
    return _impact_one(values, coefficient, exponent, lo, hi)


def _impact_one(participation, k: float, exponent: float,
                floor: float, cap: float | None) -> float:
    part = _number_or_none(participation)
    if part is None or part <= 0.0:
        return float(floor) if cap is None else float(min(floor, cap))
    exp = _number_or_none(exponent)
    if exp is None:
        exp = DEFAULT_IMPACT_EXPONENT
    try:
        value = k * (part ** exp)
    except (OverflowError, ValueError):  # pragma: no cover - defensive
        value = float("inf") if k > 0 else 0.0
    if not math.isfinite(value):
        value = float("inf")
    value = max(value, floor)
    if cap is not None:
        value = min(value, cap)
    return float(value)


# ----------------------------------------------------------------------
# small conveniences used by the docs and by reporting
# ----------------------------------------------------------------------
def notional_series(bars=None, lookback_bars: int = DEFAULT_LOOKBACK_BARS,
                    provider=None, *, frame=None) -> list[float]:
    """Per-bar quote notional over the lookback window (for docs/tables/tests).

    ``[]`` when nothing usable was passed.  A window shorter than
    ``lookback_bars`` is returned as-is rather than padded.
    """
    if provider is not None:
        value = recent_quote_volume(lookback_bars=lookback_bars, provider=provider)
        return [value]
    target = frame if frame is not None else bars
    volumes, source = _series(target)
    if volumes is None:
        volumes, source = _mapping_values(target, ("quote_volume", "volume"))
    if volumes is None:
        return []
    window = int(lookback_bars) if lookback_bars else 0
    if window > 0:
        volumes = volumes[-window:]
    if source == "volume*close":
        closes = _as_list(target["close"])
        if closes is None or len(closes) != len(volumes):
            return []
        closes = closes[-len(volumes):]
        return [v * c for v, c in zip(volumes, closes)]
    return list(volumes)


def trade_impact_pct(entry_notional: float, exit_notional: float,
                     recent_quote_volume: float, k: float,
                     exponent: float = DEFAULT_IMPACT_EXPONENT,
                     floor: float = 0.0, cap: float | None = None) -> float:
    """**Round-trip** impact in percent of entry notional (entry side + exit side).

    The cost model charges both sides, so this is the number a report quotes:
    ``[(entry_notional + exit_notional) / entry_notional] × impact_pct`` with
    ``impact_pct`` evaluated at each side's own participation.  ``0.0`` when the
    coefficient is non-positive or the window is unknown — the documented
    "impact off" case.  ``entry_notional <= 0`` returns ``0.0``.
    """
    if _finite(k, 0.0) <= 0.0:
        return 0.0
    entry = _number_or_none(entry_notional)
    if entry is None or entry <= 0.0:
        return 0.0
    exit_size = _number_or_none(exit_notional)
    if exit_size is None:
        exit_size = entry
    total = 0.0
    for notional in (entry, exit_size):
        part = participation_pct(notional, recent_quote_volume)
        if part is None:
            continue
        side_pct = impact_pct(part / 100.0, k, exponent, floor, cap)
        total += _finite(side_pct, 0.0) * (notional / entry)
    return total


def total_impact_usdt(entry_notional: float, exit_notional: float,
                      recent_quote_volume: float, k: float,
                      exponent: float = DEFAULT_IMPACT_EXPONENT,
                      floor: float = 0.0, cap: float | None = None) -> float:
    """Impact charge in **USDT** for a closed trade (both sides, absolute).

    ``Σ side impact_pct/100 × side notional`` — the number the cost model adds on
    top of fees and half-spread, and the one a backtest report should show
    alongside them.  ``0.0`` when the impact term is off or the volume unknown.
    """
    if _finite(k, 0.0) <= 0.0:
        return 0.0
    entry = _number_or_none(entry_notional)
    if entry is None or entry <= 0.0:
        return 0.0
    exit_size = _number_or_none(exit_notional)
    if exit_size is None:
        exit_size = entry
    total = 0.0
    for notional in (entry, exit_size):
        part = participation_pct(notional, recent_quote_volume)
        if part is None:
            continue
        side_pct = impact_pct(part / 100.0, k, exponent, floor, cap)
        total += _finite(side_pct, 0.0) / 100.0 * notional
    return total
