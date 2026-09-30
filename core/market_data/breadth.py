"""Market-level volume breadth from the public 24h ticker (P6-C, library only).

WHAT THIS IS
------------
A *breadth* series for the USDT spot universe, built from one public endpoint —
``GET {host}/api/v3/ticker/24hr`` on ``data-api.binance.vision`` (the reachable
public mainnet mirror; ``api.binance.com`` is not reachable from this host —
see ``docs/overhaul/MARKET_PAGES_API.md``).  Three numbers per observation:

``total_quote_volume``
    ``Σ quoteVolume`` over the pairs that carry usable data (USDT notional
    traded in the rolling 24 h window).
``up_share``
    share of those pairs whose 24 h ``priceChangePercent`` is positive — the
    advance/decline breadth of the market.
``hhi``
    Herfindahl-Hirschman index of the quote-volume shares
    (``Σ (vᵢ / Σv)²``), i.e. how concentrated the day's turnover is.  ``1/hhi``
    is reported as ``effective_pairs`` — the number of equal-sized pairs that
    would produce the same concentration.

WHAT THIS IS NOT
----------------
* **Not wired into any gate.**  P6-C delivers a library plus evidence; nothing
  imports this module on a production path, and the series is not a feature of
  the 39-column ML contract.
* **Not a historical series.**  The endpoint only ever answers "now", so the
  file cache below is a *forward* record: it can be replayed, not backfilled.
  Every observation carries the ``as_of_ms`` at which it was received, and the
  module can verify that the labelling is causal (labels non-decreasing, no
  observation used before its own ``as_of_ms``).  It cannot reconstruct what the
  breadth was an hour ago, and it never pretends to.
* **Not a substitute for a bar.**  ``quoteVolume`` and ``priceChangePercent``
  are rolling 24 h quantities, so two observations a minute apart share 23 h 59 m
  of their input.  P6-C measures this: the lag-1 autocorrelation of such a
  series is ≈ 1 by construction, which is why the plan's "< 0.99" criterion is
  not attainable at intraday sampling and is reported as **not met** rather than
  massaged away.

FAILURE POLICY (no fabricated numbers)
--------------------------------------
* :func:`fetch_tickers` raises :class:`BreadthUnavailable` on any transport or
  payload failure.
* :func:`fetch_breadth` catches it and returns ``None`` — never a zero, never an
  empty-looking-but-non-empty dict, never the last value relabelled as fresh.
* :func:`aggregate_breadth` returns ``None`` for an empty/degenerate universe
  (no pairs, or every pair reporting zero quote volume) because ``Σv = 0`` makes
  both ``hhi`` and every share undefined.
* :class:`BreadthCache` may serve the **last recorded** observation when the
  endpoint is down, but it is returned with ``is_stale=True``, ``stale_ms`` and
  ``source="cache"`` set, so a consumer can always tell a live number from a
  remembered one.  Nothing is ever served without those labels.

STALENESS / TTL POLICY (documented, all in one place)
-----------------------------------------------------
============================  =========  ==================================================
constant                      value      meaning
============================  =========  ==================================================
``TICKER24H_TTL_S``           300 s      an observation is *fresh* for 5 min.  The exchange
                                         itself updates the rolling 24 h window continuously,
                                         but the breadth aggregate moves on the scale of
                                         minutes, and the all-symbol payload costs ~90 s and
                                         ~1.9 MB (measured on this host), so polling faster
                                         than this is pure cost.
``MAX_STALE_MS``              1 800 000  beyond 30 min a cached observation is still returned,
                                         but only as ``is_stale=True``/``missing=True``
                                         evidence, never as a fresh reading.  A caller that
                                         needs fresh numbers must treat this as
                                         "unavailable".
``FETCH_TIMEOUT_S``           45 s       per-request socket timeout.  The measured full-universe
                                         fetch is ~93 s wall clock, so the timeout alone is not
                                         a complete budget; ``callers`` should also bound their
                                         own loop.
``REQUEST_ATTEMPTS``          2          one retry for a transient stall.
============================  =========  ==================================================

The cache file is JSON Lines — one appended observation per line, so a partial
write can only ever damage the **last** line and every earlier observation stays
readable.  It is bounded to ``MAX_CACHE_LINES`` lines by rewriting the file when
it grows past that; the rewrite is atomic (write a temporary file, then replace).

Dependencies: the standard library only (``json``/``urllib``/``math``/
``statistics``) — no pandas, no numpy, no network client other than the injected
opener.  The network call is a parameter (:func:`fetch_breadth`'s ``fetcher``),
which is what lets the test suite exercise every path without touching the
network.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

#: Public mainnet market-data mirror.  ``api.binance.com`` is unreachable here.
DEFAULT_HOST = "https://data-api.binance.vision"

#: The all-symbol rolling-24h ticker.
TICKER24H_PATH = "/api/v3/ticker/24hr"

#: Quote asset whose pairs form the breadth universe.
QUOTE_SUFFIX = "USDT"

#: Symbols that end in ``USDT`` but are *leveraged tokens* (Binance's
#: ``BULL``/``BEAR``/``UP``/``DOWN`` ETNs), not separate spot assets.  They are
#: excluded from the breadth universe by default because their price change is a
#: geared transform of the underlying and they would double-count it.  A caller
#: with ``exchangeInfo`` in hand can pass its own ``symbol_filter`` instead.
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")

#: Field names in the ticker payload (strings on the wire).
SYMBOL_FIELD = "symbol"
QUOTE_VOLUME_FIELD = "quoteVolume"
CHANGE_PCT_FIELD = "priceChangePercent"
LAST_PRICE_FIELD = "lastPrice"
TRADE_COUNT_FIELD = "count"
CLOSE_TIME_FIELD = "closeTime"

# ── staleness / TTL policy (see the module docstring's table) ────────────
TICKER24H_TTL_S = 300.0
MAX_STALE_MS = 1_800_000
FETCH_TIMEOUT_S = 45.0
REQUEST_ATTEMPTS = 2
RETRY_DELAY_S = 0.5

#: User-Agent: the host answers an anonymous urllib request, but naming the
#: client keeps the traffic identifiable in an outage.
USER_AGENT = "binance-trader/p6c-breadth"

#: Default cache location: ``{data_dir}/breadth/breadth.jsonl``.  Deliberately
#: *not* ``data/market/**`` — the live app rewrites that tree.
CACHE_SUBDIR = "breadth"
CACHE_FILENAME = "breadth.jsonl"
MAX_CACHE_LINES = 20_000

#: Number of top-volume symbols recorded with each observation (evidence for the
#: HHI number, not an input to it).
TOP_N = 10


class BreadthUnavailable(RuntimeError):
    """The breadth endpoint (or its payload) could not be used."""


def _to_float(value: Any) -> Optional[float]:
    """``float(value)`` or ``None`` — never ``0.0`` for a missing field."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ── pure parsing / aggregation helpers ───────────────────────────────────

@dataclass(frozen=True)
class TickerRow:
    """One ticker entry, parsed and validated."""

    symbol: str
    quote_volume: float
    change_pct: float
    last_price: Optional[float] = None
    trade_count: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def is_usdt_pair(symbol: str) -> bool:
    """True for a plain ``…USDT`` spot pair (leveraged tokens excluded)."""
    symbol = str(symbol or "").upper()
    if not symbol.endswith(QUOTE_SUFFIX) or len(symbol) <= len(QUOTE_SUFFIX):
        return False
    return not any(symbol.endswith(suffix) for suffix in LEVERAGED_SUFFIXES)


def parse_ticker(payload: Mapping[str, Any]) -> Optional[TickerRow]:
    """One ticker entry → :class:`TickerRow`, or ``None`` when unusable.

    "Unusable" means: no symbol, or a missing/non-finite ``quoteVolume`` or
    ``priceChangePercent``.  A missing field is never coerced to ``0`` — a
    fabricated zero would silently move ``up_share`` and ``hhi``.
    """
    if not isinstance(payload, Mapping):
        return None
    symbol = str(payload.get(SYMBOL_FIELD) or "").upper()
    quote_volume = _to_float(payload.get(QUOTE_VOLUME_FIELD))
    change_pct = _to_float(payload.get(CHANGE_PCT_FIELD))
    if not symbol or quote_volume is None or change_pct is None:
        return None
    if quote_volume < 0:
        return None
    return TickerRow(symbol=symbol, quote_volume=float(quote_volume),
                     change_pct=float(change_pct),
                     last_price=_to_float(payload.get(LAST_PRICE_FIELD)),
                     trade_count=_to_float(payload.get(TRADE_COUNT_FIELD)))


def parse_tickers(payload: Iterable[Mapping[str, Any]]) -> list[TickerRow]:
    """Every usable row of an all-symbol ticker payload (order preserved)."""
    if payload is None or isinstance(payload, Mapping):
        # A single-symbol payload (``{"symbol": ...}``) is a caller error here:
        # breadth needs the universe, and silently treating one market as the
        # market would be a fabricated aggregate.
        raise BreadthUnavailable(
            "expected a list of ticker entries, got "
            f"{type(payload).__name__}")
    rows: list[TickerRow] = []
    for entry in payload:
        row = parse_ticker(entry)
        if row is not None:
            rows.append(row)
    return rows


def select_pairs(rows: Iterable[TickerRow], *,
                 symbol_filter: Optional[Callable[[str], bool]] = None,
                 min_quote_volume: float = 0.0) -> list[TickerRow]:
    """The breadth universe: USDT pairs with usable data at/above a volume floor.

    ``symbol_filter`` replaces the built-in :func:`is_usdt_pair` (pass e.g. a
    predicate built from ``exchangeInfo``'s ``quoteAsset == "USDT" and
    status == "TRADING"``); the default keeps every plain USDT pair.  The floor
    is inclusive, so the default ``0.0`` keeps a pair that traded nothing —
    it belongs to the *universe* (and therefore to ``pair_count``/``coverage``)
    even though it contributes no volume and is counted as unusable.
    """
    predicate = symbol_filter or is_usdt_pair
    floor = float(min_quote_volume)
    return [row for row in rows
            if predicate(row.symbol) and row.quote_volume >= floor]


def herfindahl(shares: Sequence[float]) -> Optional[float]:
    """HHI of non-negative shares: ``Σ sᵢ²`` (``None`` if nothing is positive)."""
    values = [float(s) for s in shares if math.isfinite(float(s)) and float(s) >= 0]
    total = sum(values)
    if total <= 0:
        return None
    return float(sum((v / total) ** 2 for v in values))


def _quantile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolation quantile of an already sorted sequence."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    pos = q * (n - 1)
    low = int(math.floor(pos))
    high = min(low + 1, n - 1)
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


@dataclass(frozen=True)
class BreadthObservation:
    """One labelled breadth reading — the only thing this module ever emits.

    ``as_of_ms`` is the *local* clock at which the payload finished downloading,
    i.e. an upper bound on the information time; every field is labelled with
    it, which is what makes a replay causal.  ``request_started_ms`` and
    ``fetch_ms`` are recorded as evidence about the endpoint, not as inputs.
    """

    as_of_ms: int
    host: str = DEFAULT_HOST
    window: str = "24h"
    source: str = "live"
    request_started_ms: Optional[int] = None
    fetch_ms: Optional[float] = None
    symbol_count: int = 0
    pair_count: int = 0
    usable_count: int = 0
    coverage: Optional[float] = None
    total_quote_volume: Optional[float] = None
    up_count: int = 0
    down_count: int = 0
    flat_count: int = 0
    up_share: Optional[float] = None
    down_share: Optional[float] = None
    flat_share: Optional[float] = None
    hhi: Optional[float] = None
    effective_pairs: Optional[float] = None
    top_share: Optional[float] = None
    median_quote_volume: Optional[float] = None
    p90_quote_volume: Optional[float] = None
    top_symbols: tuple[tuple[str, float], ...] = ()
    is_stale: bool = False
    stale_ms: Optional[int] = None
    #: True when the observation is past :data:`MAX_STALE_MS` — the documented
    #: "still returned, but treat it as unavailable" state.  Distinct from
    #: *absent* (`latest()` returning ``None``) and from merely stale (past
    #: ``TICKER24H_TTL_S`` but inside the bound).
    missing: bool = False
    expected_pair_count: Optional[int] = None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["top_symbols"] = [list(pair) for pair in self.top_symbols]
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BreadthObservation":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in dict(payload).items() if k in fields}
        data["top_symbols"] = tuple(
            (str(pair[0]), float(pair[1]))
            for pair in (data.get("top_symbols") or []) if len(pair) == 2)
        for key in ("as_of_ms", "symbol_count", "pair_count", "usable_count",
                    "up_count", "down_count", "flat_count"):
            if key in data and data[key] is not None:
                data[key] = int(data[key])
        for key in ("request_started_ms", "stale_ms", "expected_pair_count"):
            if key in data and data[key] is not None:
                data[key] = int(data[key])
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def aggregate_breadth(rows: Iterable[TickerRow], *, as_of_ms: int,
                      host: str = DEFAULT_HOST, window: str = "24h",
                      source: str = "live",
                      request_started_ms: Optional[int] = None,
                      fetch_ms: Optional[float] = None,
                      symbol_count: Optional[int] = None,
                      expected_pair_count: Optional[int] = None,
                      symbol_filter: Optional[Callable[[str], bool]] = None,
                      min_quote_volume: float = 0.0,
                      top_n: int = TOP_N) -> Optional[BreadthObservation]:
    """Turn parsed ticker rows into one labelled observation (or ``None``).

    ``None`` when the universe is empty or every pair reports zero quote volume:
    shares and HHI are undefined at ``Σv = 0``, and returning ``0.0`` there would
    be a fabricated number.

    Counting rules, so ``coverage`` cannot be read two ways:

    ``symbol_count``
        every entry the endpoint returned (all quote assets).
    ``pair_count``
        entries that pass ``symbol_filter`` — the USDT universe *as the endpoint
        reports it*.
    ``usable_count``
        pairs with ``quoteVolume > 0``.  A zero-volume pair is present but
        carries no turnover, so it is counted as a **miss** for coverage and
        excluded from every share.
    ``coverage``
        ``usable_count / expected_pair_count`` when the documented universe size
        is supplied (the plan's "≥ 95 % of the 496 pairs" row), else
        ``usable_count / pair_count``.  It is reported unclamped: a value above
        1 means the endpoint served more pairs than the expectation, which is
        information, not an error to hide.
    """
    rows = list(rows)
    if symbol_count is None:
        symbol_count = len(rows)
    pairs = select_pairs(rows, symbol_filter=symbol_filter,
                         min_quote_volume=min_quote_volume)
    usable = [row for row in pairs if row.quote_volume > 0]
    total = sum(row.quote_volume for row in usable)
    if not usable or total <= 0:
        return None
    denominator = int(expected_pair_count) if expected_pair_count \
        else len(pairs)
    volumes = sorted(row.quote_volume for row in usable)
    shares = [row.quote_volume / total for row in usable]
    hhi = herfindahl(shares)
    up = sum(1 for row in usable if row.change_pct > 0)
    down = sum(1 for row in usable if row.change_pct < 0)
    flat = len(usable) - up - down
    ranked = sorted(usable, key=lambda r: (-r.quote_volume, r.symbol))
    top = tuple((row.symbol, row.quote_volume) for row in ranked[:int(top_n)])
    return BreadthObservation(
        as_of_ms=int(as_of_ms),
        host=str(host), window=str(window), source=str(source),
        request_started_ms=None if request_started_ms is None
        else int(request_started_ms),
        fetch_ms=None if fetch_ms is None else float(fetch_ms),
        symbol_count=int(symbol_count), pair_count=int(len(pairs)),
        usable_count=int(len(usable)),
        coverage=(len(usable) / denominator) if denominator else None,
        total_quote_volume=float(total),
        up_count=up, down_count=down, flat_count=flat,
        up_share=up / len(usable), down_share=down / len(usable),
        flat_share=flat / len(usable),
        hhi=hhi,
        effective_pairs=(1.0 / hhi) if hhi else None,
        top_share=(top[0][1] / total) if top else None,
        median_quote_volume=_quantile(volumes, 0.5),
        p90_quote_volume=_quantile(volumes, 0.9),
        top_symbols=top,
        expected_pair_count=None if expected_pair_count is None
        else int(expected_pair_count),
    )


# ── transport ────────────────────────────────────────────────────────────

def _urllib_open(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_tickers(host: str = DEFAULT_HOST, *, timeout: float = FETCH_TIMEOUT_S,
                  opener: Optional[Callable[[str, float], bytes]] = None,
                  attempts: int = REQUEST_ATTEMPTS) -> list[Mapping[str, Any]]:
    """Fetch and decode the all-symbol 24h ticker; raise on any failure.

    ``opener(url, timeout) -> bytes`` is injectable so tests never touch the
    network.  Every failure mode — socket error, HTTP error, timeout, invalid
    JSON, wrong payload type — becomes :class:`BreadthUnavailable`, because a
    caller must not be able to mistake a failure for an empty market.
    """
    open_fn = opener or _urllib_open
    url = f"{str(host).rstrip('/')}{TICKER24H_PATH}"
    last_error: Optional[str] = None
    for attempt in range(max(1, int(attempts))):
        try:
            raw = open_fn(url, float(timeout))
        except Exception as exc:  # noqa: BLE001 - one failure type for callers
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes)
                                     else raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                last_error = f"undecodable payload: {exc}"
            else:
                if not isinstance(payload, list):
                    last_error = (f"expected a list of ticker entries, got "
                                  f"{type(payload).__name__}")
                else:
                    return payload
        if attempt + 1 < max(1, int(attempts)):
            time.sleep(RETRY_DELAY_S)
    raise BreadthUnavailable(f"{url} unavailable ({last_error})")


def fetch_breadth(host: str = DEFAULT_HOST, *, timeout: float = FETCH_TIMEOUT_S,
                  opener: Optional[Callable[[str, float], bytes]] = None,
                  fetcher: Optional[Callable[[], list[Mapping[str, Any]]]] = None,
                  now_ms: Optional[int] = None,
                  expected_pair_count: Optional[int] = None,
                  symbol_filter: Optional[Callable[[str], bool]] = None,
                  min_quote_volume: float = 0.0,
                  top_n: int = TOP_N) -> Optional[BreadthObservation]:
    """One live breadth observation; ``None`` (never a fabricated value) on failure.

    ``fetcher`` overrides the whole transport (tests pass a stub); ``opener``
    overrides only the socket call.  ``now_ms`` lets a caller pin the label
    clock (tests and replays).
    """
    started = int(time.time() * 1000) if now_ms is None else int(now_ms)
    clock = time.monotonic()
    try:
        payload = (fetcher() if fetcher is not None
                   else fetch_tickers(host, timeout=timeout, opener=opener))
        rows = parse_tickers(payload)
    except BreadthUnavailable:
        return None
    except Exception:  # noqa: BLE001 - a stub/fetcher failure is still a failure
        return None
    received = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return aggregate_breadth(
        rows, as_of_ms=received, host=host, request_started_ms=started,
        fetch_ms=time.monotonic() - clock, symbol_count=len(payload),
        expected_pair_count=expected_pair_count, symbol_filter=symbol_filter,
        min_quote_volume=min_quote_volume, top_n=top_n)


# ── cache, staleness, replay ─────────────────────────────────────────────

def default_cache_path(data_dir: str | os.PathLike[str] = "data") -> Path:
    """``{data_dir}/breadth/breadth.jsonl`` (a new tree, not ``data/market``)."""
    return Path(data_dir) / CACHE_SUBDIR / CACHE_FILENAME


class BreadthCache:
    """Append-only JSONL record of breadth observations, with the TTL policy.

    * :meth:`latest` returns the newest usable observation, labelled with
      ``is_stale``/``stale_ms`` against the caller's clock; past
      ``max_stale_ms`` it is additionally labelled ``missing=True`` (the
      documented "still returned, but treat it as unavailable" state).
    * :meth:`fresh` returns it only while it is inside
      :data:`TICKER24H_TTL_S`; otherwise ``None``.
    * :meth:`refresh` fetches when the cached value is not fresh and falls back
      to the cached (stale-labelled) value when the endpoint is unreachable.
    * Errors are never cached: a failed fetch appends nothing.
    """

    def __init__(self, path: Optional[str | os.PathLike[str]] = None, *,
                 ttl_s: float = TICKER24H_TTL_S, max_stale_ms: int = MAX_STALE_MS,
                 clock: Callable[[], float] = time.time,
                 now_ms: Optional[int] = None) -> None:
        self.path = Path(path) if path is not None else default_cache_path()
        self.ttl_s = float(ttl_s)
        self.max_stale_ms = int(max_stale_ms)
        self._clock = clock
        self._pinned_now_ms = now_ms
        self._observations: Optional[list[BreadthObservation]] = None

    # -- time ----------------------------------------------------------
    def now_ms(self) -> int:
        if self._pinned_now_ms is not None:
            return int(self._pinned_now_ms)
        return int(self._clock() * 1000)

    # -- reading -------------------------------------------------------
    def load(self, *, refresh: bool = False) -> list[BreadthObservation]:
        """Every readable observation, oldest first (a bad line is skipped)."""
        if self._observations is not None and not refresh:
            return list(self._observations)
        out: list[BreadthObservation] = []
        if self.path.exists():
            try:
                text = self.path.read_text(encoding="utf-8")
            except OSError:
                text = ""
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(BreadthObservation.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    # A torn last line (or an old schema) is skipped rather than
                    # crashing the caller: partial data is not a fabricated one.
                    continue
        self._observations = out
        return list(out)

    def latest(self) -> Optional[BreadthObservation]:
        """Newest observation, relabelled with its staleness against ``now``."""
        observations = self.load()
        if not observations:
            return None
        return self._with_staleness(observations[-1])

    def fresh(self) -> Optional[BreadthObservation]:
        """Newest observation only while it is inside the TTL, else ``None``."""
        latest = self.latest()
        if latest is None or latest.is_stale:
            return None
        return latest

    def _with_staleness(self, obs: BreadthObservation) -> BreadthObservation:
        age = max(0, self.now_ms() - int(obs.as_of_ms))
        fresh = age <= self.ttl_s * 1000.0
        # Audit finding 5: `max_stale_ms` used to be stored and never compared,
        # while the module docstring gave it behaviour.  It is now the explicit
        # "treat as unavailable" boundary *inside* a still-returned observation:
        # a value older than it is not merely stale, it is `missing=True`, so a
        # caller that cannot tolerate a 30-minute-old book does not have to
        # re-derive the threshold from the constant (and a torn/absent line stays
        # distinguishable from "the endpoint answered an hour ago").
        return BreadthObservation.from_dict({
            **obs.to_dict(), "source": "cache", "is_stale": not fresh,
            "stale_ms": age, "missing": age > self.max_stale_ms})

    # -- writing -------------------------------------------------------
    def append(self, obs: Optional[BreadthObservation]) -> bool:
        """Append one observation; ``False`` (and no write) for ``None``."""
        if obs is None:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(obs.to_json() + "\n")
        self._observations = None
        self._compact_if_needed()
        return True

    def record(self, obs: Optional[BreadthObservation]) -> Optional[BreadthObservation]:
        """Append then return the observation unchanged (``None`` stays ``None``)."""
        self.append(obs)
        return obs

    def _compact_if_needed(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if len(lines) <= MAX_CACHE_LINES:
            return
        keep = lines[-MAX_CACHE_LINES:]
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
        self._observations = None

    # -- the composite operation --------------------------------------
    def refresh(self, *, host: str = DEFAULT_HOST, timeout: float = FETCH_TIMEOUT_S,
                opener: Optional[Callable[[str, float], bytes]] = None,
                fetcher: Optional[Callable[[], list[Mapping[str, Any]]]] = None,
                expected_pair_count: Optional[int] = None,
                symbol_filter: Optional[Callable[[str], bool]] = None,
                min_quote_volume: float = 0.0,
                top_n: int = TOP_N, force: bool = False
                ) -> tuple[Optional[BreadthObservation], str]:
        """``(observation, status)`` with ``status`` one of the documented cases.

        ``"fresh-cache"``  the cached value is inside the TTL, nothing fetched.
        ``"live"``         a new observation was fetched and appended.
        ``"stale-cache"``  the endpoint failed; the cached value is returned,
                           labelled ``is_stale=True`` (or ``None`` if there is no
                           cached value at all).
        ``"unavailable"``  the endpoint failed and nothing was cached.

        A live observation newer than the cached one always wins; ``None`` is
        never returned as if it were data, and a failed fetch never becomes an
        entry in the cache.

        ``force=True`` fetches even while the cache is fresh.  That is what a
        *recorder* needs (the P6-C experiment samples a series), and it is why
        the default is ``False``: an ordinary caller should honour the TTL
        rather than pay ~90 s per call for a number that has barely moved.
        """
        cached = self.fresh()
        if (cached is not None and not force and fetcher is None
                and opener is None):
            # Only short-circuit when the caller is not forcing a fetch; a
            # caller that injects a fetcher/opener is asking for a live read.
            return cached, "fresh-cache"
        observation = fetch_breadth(
            host, timeout=timeout, opener=opener, fetcher=fetcher,
            now_ms=self._pinned_now_ms, expected_pair_count=expected_pair_count,
            symbol_filter=symbol_filter, min_quote_volume=min_quote_volume,
            top_n=top_n)
        if observation is not None:
            self.append(observation)
            return observation, "live"
        fallback = self.latest()
        if fallback is None:
            return None, "unavailable"
        return fallback, "stale-cache"


def load_series(path: Optional[str | os.PathLike[str]] = None) -> list[BreadthObservation]:
    """Every recorded observation at ``path`` (default: the standard cache)."""
    return BreadthCache(path).load()


# ── series measurement (plan criteria) ───────────────────────────────────

def observation_values(observations: Iterable[BreadthObservation],
                       field_name: str) -> list[float]:
    """Non-``None`` values of one field, in file order (no interpolation)."""
    out: list[float] = []
    for obs in observations:
        value = getattr(obs, field_name, None)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            out.append(value)
    return out


def variance(values: Sequence[float]) -> Optional[float]:
    """Population variance; ``None`` below two values (never ``0.0``)."""
    data = [float(v) for v in values if math.isfinite(float(v))]
    if len(data) < 2:
        return None
    mean = sum(data) / len(data)
    return sum((v - mean) ** 2 for v in data) / len(data)


def autocorrelation(values: Sequence[float], lag: int = 1) -> Optional[float]:
    """Lag-``lag`` autocorrelation (Pearson on overlapping pairs).

    ``None`` when there are fewer than ``lag + 2`` points or the series has no
    variance.  No numpy: breadth series are short and pure Python is exact.
    """
    data = [float(v) for v in values if math.isfinite(float(v))]
    lag = int(lag)
    if lag < 1 or len(data) <= lag + 1:
        return None
    left, right = data[:-lag], data[lag:]
    n = len(left)
    mean_left = sum(left) / n
    mean_right = sum(right) / n
    cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    var_left = sum((a - mean_left) ** 2 for a in left)
    var_right = sum((b - mean_right) ** 2 for b in right)
    if var_left <= 0 or var_right <= 0:
        return None
    return cov / math.sqrt(var_left * var_right)


def differences(values: Sequence[float]) -> list[float]:
    """First differences of a series (what the rolling 24 h window can still add)."""
    data = [float(v) for v in values]
    return [b - a for a, b in zip(data[:-1], data[1:])]


def is_causal(observations: Sequence[BreadthObservation]) -> dict:
    """Verify the *labelling* of a recorded series (all that can be verified).

    Checks that ``as_of_ms`` is non-decreasing and that each observation's
    ``request_started_ms`` (when present) is not after its own ``as_of_ms``.
    A pass means "no observation is labelled with a time before the information
    it contains could exist" — it says nothing about whether a historical value
    could have been reconstructed, which it could not.
    """
    observations = list(observations)
    problems: list[str] = []
    previous = None
    for index, obs in enumerate(observations):
        if previous is not None and int(obs.as_of_ms) < int(previous):
            problems.append(f"row {index}: as_of_ms went backwards "
                            f"({obs.as_of_ms} < {previous})")
        previous = int(obs.as_of_ms)
        if obs.request_started_ms is not None \
                and int(obs.request_started_ms) > int(obs.as_of_ms):
            problems.append(f"row {index}: request started after its own "
                            f"as_of_ms label")
    return {"n": len(observations), "causal": not problems, "problems": problems,
            "verifiable": "labelling only; the endpoint has no history to backfill"}


def replay(observations: Sequence[BreadthObservation], *, upto_ms: int
           ) -> list[BreadthObservation]:
    """The series as it was knowable at ``upto_ms`` (label filter, no backfill)."""
    return [obs for obs in observations if int(obs.as_of_ms) <= int(upto_ms)]


def availability_report(observations: Sequence[BreadthObservation]) -> dict:
    """Coverage / non-degeneracy summary — the plan's breadth acceptance rows.

    Reports the minimum coverage, the variance of each series, and the lag-1
    autocorrelation of the level **and** of its first difference.  The plan's
    criterion is ``acf(level) < 0.99``; because a 24 h rolling window overlaps
    itself at every sampling interval shorter than a day, that criterion is
    expected to fail and is reported as measured rather than restated.
    """
    observations = list(observations)
    coverages = [float(obs.coverage) for obs in observations
                 if obs.coverage is not None]
    report: dict[str, Any] = {"n": len(observations)}
    for name in ("total_quote_volume", "up_share", "hhi"):
        values = observation_values(observations, name)
        report[name] = {
            "n": len(values),
            "variance": variance(values),
            "acf1": autocorrelation(values, 1),
            "acf1_differenced": autocorrelation(differences(values), 1),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    report["coverage"] = {
        "n": len(coverages),
        "min": min(coverages) if coverages else None,
        "mean": (sum(coverages) / len(coverages)) if coverages else None,
    }
    report["non_degenerate_acf_lt_0_99"] = {
        name: (None if report[name]["acf1"] is None
               else bool(abs(report[name]["acf1"]) < 0.99))
        for name in ("total_quote_volume", "up_share", "hhi")}
    report["variance_gt_zero"] = {
        name: (None if report[name]["variance"] is None
               else bool(report[name]["variance"] > 0.0))
        for name in ("total_quote_volume", "up_share", "hhi")}
    report["causal_labels"] = is_causal(observations)
    return report


__all__ = [
    "DEFAULT_HOST", "TICKER24H_PATH", "QUOTE_SUFFIX", "LEVERAGED_SUFFIXES",
    "TICKER24H_TTL_S", "MAX_STALE_MS", "FETCH_TIMEOUT_S", "CACHE_SUBDIR",
    "CACHE_FILENAME", "MAX_CACHE_LINES", "TOP_N",
    "BreadthUnavailable", "TickerRow", "BreadthObservation", "BreadthCache",
    "is_usdt_pair", "parse_ticker", "parse_tickers", "select_pairs",
    "herfindahl", "aggregate_breadth", "fetch_tickers", "fetch_breadth",
    "default_cache_path", "load_series", "observation_values", "variance",
    "autocorrelation", "differences", "is_causal", "replay",
    "availability_report",
]
