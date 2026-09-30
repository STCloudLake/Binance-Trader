"""Volume-clock sampling: dollar bars, volume bars and time bars (P6-C).

WHAT THIS IS
------------
An **offline, off-by-default** alternative to time sampling.  A *dollar bar*
(``dollar_bars``) closes when the cumulative ``close × volume`` traded since the
previous bar reached a fixed notional threshold; a *volume bar*
(``volume_bars``) closes on cumulative base volume; a *time bar*
(``time_bars``) is the ordinary calendar resample of the same 1-minute prints.
The three are produced from the **same** frame, in the cache's column layout
(``open / high / low / close / volume``, ``DatetimeIndex`` named ``close_time``),
so any consumer of ``data/market/<SYM>/<TF>.parquet`` can read them unchanged.

WHY (and what it is not)
------------------------
The hypothesis under test in P6-C is that sampling by traded activity produces
better-behaved return distributions (thinner tails, less serial dependence) than
sampling by the clock.  This module only **builds and measures**; it wires
nothing into a production gate, and the measured verdict on the real cache is
recorded in ``docs/core-algorithms/15-volume-bars-breadth.md``.  A "no
improvement" result is a valid and expected outcome.

CAUSALITY (the one hard requirement)
------------------------------------
Every constructor here is a **prefix function**: bar *i* is a function of the
prints up to and including its own closing print, and of nothing later.  Two
mechanisms enforce it:

* a bar closes at the first print whose weight brings the weight accumulated
  *since the previous bar closed* to the threshold, so a later print can only
  close a newer bar — it can never move an existing boundary;
* a bar whose accumulated weight is still below the threshold at the end of the
  available data is **dropped** (``drop_partial=True``): it is a
  *provisional* bar, not a historical one, so appending prints completes it
  *into a new bar* instead of revising an old one.  ``time_bars`` applies the
  same rule to its final resample bin (``drop_last=True``).  Note the deliberate
  consequence: pandas' default ``origin="start"`` would anchor the bins to the
  first print of whatever prefix it is given, which makes the whole grid move
  when data is appended; ``origin="start_day"`` is pinned instead.

:func:`historical_bars_unchanged` is the acceptance measurement — it compares
the bars built from a prefix against the bars built from the full frame and
returns how many matched rows differ.  P6-C's criterion is **0**.

:class:`VolumeClockBuilder` is the streaming form (``as_of``-safe): it emits a
closed bar the moment the threshold is crossed and keeps the in-progress bar
only in :attr:`VolumeClockBuilder.provisional`.  A closed bar is never revised.

INTEGRATION NOTE
----------------
The sampled frames carry the cache's five columns exactly (no extra columns, so
a parquet round-trip stays schema-compatible).  Anything else a caller needs —
the notional weight of a bar, the number of prints, the provisional bar — is
derived by the caller or exposed by the builder, never smuggled into the frame.

Dependencies: ``numpy`` + ``pandas`` only.  The Jarque-Bera p-value uses the
closed form of the ``chi2(2)`` survival function (``exp(-JB/2)``), so no SciPy
is required and no approximation is involved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

#: The cache's OHLCV layout; every constructor returns exactly these columns.
CACHE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: Name of the returned index (the cache uses the bar's close time).
INDEX_NAME = "close_time"

#: Below this many observations a moment/ACF/JB number is reported as ``None``
#: rather than as a number computed from noise.
MIN_STATS_ROWS = 8

#: x87-safe relative slack used when deciding whether the trailing bar reached
#: its threshold (a bar exactly *at* the threshold is complete).
_WEIGHT_SLACK = 1e-9


# ── input handling ───────────────────────────────────────────────────────

def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, coerce and time-sort a print frame (never mutates the input).

    Raises :class:`ValueError` when a required column is missing or the index is
    not a ``DatetimeIndex``: a silently wrong frame would be measured as if it
    were real, which is worse than refusing.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(frame).__name__}")
    missing = [c for c in CACHE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing column(s) {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("frame index must be a DatetimeIndex (bar time)")
    out = frame.loc[:, list(CACHE_COLUMNS)].astype(float)
    if not out.index.is_monotonic_increasing:
        out = out.sort_index(kind="stable")
    finite = np.isfinite(out.to_numpy()).all(axis=1)
    if not finite.all():
        out = out[finite]
    return out


def _print_weight(frame: pd.DataFrame, notional: bool) -> np.ndarray:
    """Per-print weight: ``close × volume`` (notional) or ``volume`` (base)."""
    volume = frame["volume"].to_numpy(dtype=float)
    if notional:
        return frame["close"].to_numpy(dtype=float) * volume
    return volume


def _empty_bars() -> pd.DataFrame:
    index = pd.DatetimeIndex([], name=INDEX_NAME)
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in CACHE_COLUMNS},
                        index=index)


def _aggregate(frame: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    """Aggregate prints into bars by integer ``groups`` (non-decreasing).

    The bar's timestamp is the **last print's** time in the group, i.e. the bar's
    close time — the same convention the cache uses.
    """
    if len(frame) == 0:
        return _empty_bars()
    if len(np.unique(groups)) == 1:
        # Fast path; keeps the exact same numbers as the groupby path.
        values = {
            "open": float(frame["open"].iloc[0]),
            "high": float(frame["high"].max()),
            "low": float(frame["low"].min()),
            "close": float(frame["close"].iloc[-1]),
            "volume": float(frame["volume"].sum()),
        }
        index = pd.DatetimeIndex([frame.index[-1]], name=INDEX_NAME)
        return pd.DataFrame([values], index=index,
                            columns=list(CACHE_COLUMNS))
    # Group by a positional ndarray: a Series key would be index-aligned, and a
    # frame with duplicated timestamps (the cache's twin bars) would then group
    # the wrong rows together.
    bars = frame.groupby(groups, sort=True).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"))
    closes = pd.Series(frame.index.to_numpy()).groupby(groups, sort=True).last()
    bars.index = pd.DatetimeIndex(closes.to_numpy(), name=INDEX_NAME)
    return bars.loc[:, list(CACHE_COLUMNS)].astype(float)


def _clock_groups(weight: np.ndarray, per_bar: float) -> tuple[np.ndarray, int]:
    """Assign prints to activity-clock bars; return ``(group_id, n_closed)``.

    One bar closes at the **first** print whose weight brings the accumulated
    weight *since the previous bar closed* to ``per_bar``; the overshoot beyond
    the threshold is dropped and the next bar starts at the following print —
    the standard dollar/volume clock, and the exact rule
    :class:`VolumeClockBuilder` applies print by print.  Trailing prints that
    never reached the threshold keep ``group_id = -1`` (a provisional bar, not a
    historical one); the caller decides whether to keep them.

    The scan is over *bars* (``O(n_bars)`` ``searchsorted`` calls), not over
    prints, and every boundary depends only on the prints at or before it, which
    is what makes the sampled series a prefix function of the input.
    """
    n = weight.size
    groups = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return groups, 0
    cumulative = np.cumsum(weight)
    bar_id = 0
    pos = 0
    while pos < n:
        base = cumulative[pos - 1] if pos > 0 else 0.0
        target = base + float(per_bar)
        # `side="left"` with the accumulated slack mirrors the builder's
        # `accumulated >= threshold` test, so the two agree bit for bit.
        end = int(np.searchsorted(cumulative, target, side="left"))
        if end >= n:
            break
        groups[pos:end + 1] = bar_id
        bar_id += 1
        pos = end + 1
    return groups, bar_id


def _clock_bars(frame: pd.DataFrame, per_bar: float, *, notional: bool,
                drop_partial: bool, kind: str) -> pd.DataFrame:
    if per_bar is None or not np.isfinite(per_bar) or per_bar <= 0:
        raise ValueError(f"{kind}: per_bar must be a positive finite number, "
                         f"got {per_bar!r}")
    data = _prepare(frame)
    if len(data) == 0:
        return _empty_bars()
    # The weight is derived from the *prepared* frame, so it is positionally
    # aligned with it even when ``_prepare`` had to sort or drop rows.
    groups, n_closed = _clock_groups(_print_weight(data, notional), float(per_bar))
    if n_closed == 0:
        if drop_partial:
            return _empty_bars()
        groups = np.zeros(len(data), dtype=np.int64)
    if not drop_partial and (groups < 0).any():
        # Keep the provisional tail as an extra (flagged by construction as the
        # only bar that may still change) group.
        groups = np.where(groups < 0, n_closed, groups)
    else:
        keep = groups >= 0
        data = data[keep]
        groups = groups[keep]
        if len(data) == 0:
            return _empty_bars()
    return _aggregate(data, groups)


# ── public constructors ──────────────────────────────────────────────────

def dollar_bars(frame: pd.DataFrame, notional_per_bar: float,
                *, drop_partial: bool = True) -> pd.DataFrame:
    """Bars closing when cumulative ``close × volume`` reaches the threshold.

    ``frame`` is a print frame (the 1-minute cache is the intended input).
    Returns the cache's five columns indexed by each bar's close time.  With
    ``drop_partial=True`` (the default, and the only causal choice for
    evaluation) a trailing bar that has not yet reached
    ``notional_per_bar`` is omitted.
    """
    return _clock_bars(frame, notional_per_bar, notional=True,
                       drop_partial=drop_partial, kind="dollar_bars")


def volume_bars(frame: pd.DataFrame, volume_per_bar: float,
                *, drop_partial: bool = True) -> pd.DataFrame:
    """Bars closing when cumulative base ``volume`` reaches the threshold.

    The volume clock of the literature, on the cache's ``volume`` column.
    """
    return _clock_bars(frame, volume_per_bar, notional=False,
                       drop_partial=drop_partial, kind="volume_bars")


def time_bars(frame: pd.DataFrame, interval: str = "1h",
              *, drop_last: bool = True) -> pd.DataFrame:
    """Calendar resample of the same prints (the control in every comparison).

    ``label="right"``/``closed="right"`` puts a print at exactly 12:00:00 into
    the bin ending at 12:00:00, which is the crypto convention (a bar's close
    time is the last instant it covers).  ``origin="start_day"`` pins the bin
    grid to an absolute edge so appending data cannot move it.

    ``drop_last=True`` omits the final bin: whether more prints belong to it is
    unknown at the time it is built, which makes it provisional rather than
    historical — the exact rule the clock bars use for their trailing group.
    Empty bins (a gap longer than the interval) are dropped: no prints, no bar.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(frame).__name__}")
    data = _prepare(frame)
    if len(data) == 0:
        return _empty_bars()
    resampled = data.resample(interval, label="right", closed="right",
                              origin="start_day")
    bars = resampled.agg(open=("open", "first"), high=("high", "max"),
                         low=("low", "min"), close=("close", "last"),
                         volume=("volume", "sum")).dropna(subset=["close"])
    if drop_last and len(bars):
        bars = bars.iloc[:-1]
    bars = bars.loc[:, list(CACHE_COLUMNS)].astype(float)
    bars.index.name = INDEX_NAME
    return bars


# ── threshold calibration ────────────────────────────────────────────────

def _threshold_from_warmup(frame: pd.DataFrame, target_bars: int,
                           notional: bool, calibrate_on: float) -> float:
    """Threshold = warm-up weight / (target bars × warm-up share).

    The division by ``calibrate_on`` is what makes ``target_bars`` mean *the
    whole sample*: the leading warm-up block is asked to supply
    ``target_bars × calibrate_on`` bars, and the assumption (stated, not
    hidden) is that the rest of the period trades at the warm-up's rate.  The
    realised full-sample count is not forced to the target — how far off it is
    is a measurement of turnover drift, and every caller reports it.
    """
    data = _prepare(frame)
    if len(data) == 0:
        raise ValueError("cannot calibrate a threshold on an empty frame")
    if int(target_bars) < 1:
        raise ValueError(f"target_bars must be >= 1, got {target_bars!r}")
    if not 0.0 < float(calibrate_on) <= 1.0:
        raise ValueError(f"calibrate_on must be in (0, 1], got {calibrate_on!r}")
    warmup = max(1, int(round(len(data) * float(calibrate_on))))
    weight = _print_weight(data.iloc[:warmup], notional)
    total = float(weight.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("the calibration window traded no volume")
    return total / (float(target_bars) * float(calibrate_on))


def notional_threshold(frame: pd.DataFrame, target_bars: int,
                       *, calibrate_on: float = 0.2) -> float:
    """Dollar-bar threshold calibrated on the **first** ``calibrate_on`` share.

    Using the whole sample's average notional would let prints from after the
    evaluation window fix the sampling grid — a look-ahead in the *sampling*
    even when the features are clean.  The threshold here is a function of a
    leading warm-up block only, so it is known before the first bar is built.
    """
    return _threshold_from_warmup(frame, target_bars, True, calibrate_on)


def volume_threshold(frame: pd.DataFrame, target_bars: int,
                     *, calibrate_on: float = 0.2) -> float:
    """Volume-bar threshold; same warm-up rule as :func:`notional_threshold`."""
    return _threshold_from_warmup(frame, target_bars, False, calibrate_on)


def interval_for_bar_count(frame: pd.DataFrame, target_bars: int) -> str:
    """A pandas offset string whose expected bin count is ``target_bars``.

    ``span / target_bars`` rounded to a whole minute (pandas rejects sub-minute
    integer offsets such as ``"43.2min"``).  This is the *first guess* only:
    a cached print frame has gaps, so the realised bin count is always below
    ``target_bars``; use :func:`match_time_interval` when the control series has
    to have the same number of observations as the activity series.
    """
    data = _prepare(frame)
    if len(data) < 2:
        raise ValueError("need at least two prints to size a time interval")
    if int(target_bars) < 1:
        raise ValueError(f"target_bars must be >= 1, got {target_bars!r}")
    span_minutes = (data.index[-1] - data.index[0]).total_seconds() / 60.0
    minutes = max(1, int(round(span_minutes / float(target_bars))))
    return f"{minutes}min"


def match_time_interval(frame: pd.DataFrame, target_bars: int, *,
                        max_iter: int = 12, tolerance: float = 0.01
                        ) -> tuple[str, int, int]:
    """Solve the time interval that yields ≈ ``target_bars`` non-empty bins.

    Fixed-point iteration ``minutes ← minutes · realised / target``, rounded to
    whole minutes.  Because the grid is whole-minute, the count is not
    continuous in the interval, so the iteration can oscillate: the **closest**
    interval seen is returned, not the last one evaluated.  Returns
    ``(offset, realised_bin_count, iterations)``.

    Why this is not look-ahead: the interval is a **sampling specification**, a
    function of the print *timestamps' density only* — it is computed once,
    before any feature, label or gate number exists, and is then held constant
    for the whole series, so it cannot put later prices into an earlier bar.
    The alternative (comparing a 3-hour control bar against a 40-minute
    activity bar) would make every distribution metric a restatement of bar
    length, which is the artefact this function exists to remove.  The realised
    count and the number of iterations are returned so the calibration is
    visible rather than assumed.
    """
    data = _prepare(frame)
    if len(data) < 2:
        raise ValueError("need at least two prints to size a time interval")
    target = max(1, int(target_bars))
    minutes = max(1, int(round(
        (data.index[-1] - data.index[0]).total_seconds() / 60.0 / target)))
    best = (minutes, -1)
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        realised = len(time_bars(data, f"{minutes}min"))
        if realised == 0:
            break
        if best[1] < 0 or abs(realised - target) < abs(best[1] - target):
            best = (minutes, realised)
        if abs(realised - target) <= max(1.0, float(tolerance) * target):
            break
        minutes = max(1, int(round(minutes * realised / target)))
    return f"{best[0]}min", int(best[1]), int(iterations)


# ── as-of / streaming construction ───────────────────────────────────────

def bars_as_of(frame: pd.DataFrame, at, *, kind: str = "dollar",
               notional_per_bar: Optional[float] = None,
               volume_per_bar: Optional[float] = None,
               interval: str = "1h", drop_partial: bool = True) -> pd.DataFrame:
    """Bars reconstructed from prints at or before ``at`` only.

    This is the causal entry point for a backtest: the frame is sliced by time
    and then handed to the same constructor, so the result is by construction a
    function of the past only.  ``at`` is any ``pd.Timestamp``/string the index
    can be compared with.
    """
    data = _prepare(frame)
    cutoff = pd.Timestamp(at)
    window = data[data.index <= cutoff]
    if kind == "dollar":
        return dollar_bars(window, notional_per_bar, drop_partial=drop_partial)
    if kind == "volume":
        return volume_bars(window, volume_per_bar, drop_partial=drop_partial)
    if kind == "time":
        return time_bars(window, interval)
    raise ValueError(f"unknown kind {kind!r} (expected dollar/volume/time)")


@dataclass
class VolumeClockBuilder:
    """Streaming, as-of-safe activity clock (dollar or volume).

    ``push`` accepts one print at a time and returns the closed bar dict
    (``open/high/low/close/volume`` + ``close_time`` + ``bar_id``) the moment
    the accumulated weight reaches the threshold, else ``None``.  The bar that
    is still filling lives in :attr:`provisional` and is **never** emitted as a
    closed bar: feeding more prints completes it into a new bar instead of
    revising a historical one.

    Exactly one of ``notional_per_bar`` / ``volume_per_bar`` must be given.  The
    builder is a faithful incremental form of the batch constructors —
    ``push_frame`` over a frame reproduces ``dollar_bars`` / ``volume_bars``
    boundary for boundary and price for price (``drop_partial`` is irrelevant
    here because only complete bars are ever emitted).  The one column that is
    not guaranteed bit-identical is ``volume``: the builder adds print volumes
    one at a time while the batch path sums each group with pandas' own
    (pairwise) reduction, so the same sum can differ in its last bits.  The
    measured worst case on the real cache is quoted in
    ``docs/core-algorithms/15-volume-bars-breadth.md``; the equivalence test
    asserts exact equality for the boundaries and the four prices and a
    floating-point tolerance for volume.
    """

    notional_per_bar: Optional[float] = None
    volume_per_bar: Optional[float] = None
    bar_id: int = 0
    n_prints: int = 0
    cumulative: float = 0.0
    segment_base: float = 0.0
    _partial: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        given = [self.notional_per_bar is not None, self.volume_per_bar is not None]
        if sum(given) != 1:
            raise ValueError("give exactly one of notional_per_bar / volume_per_bar")
        threshold = (self.notional_per_bar if self.notional_per_bar is not None
                     else self.volume_per_bar)
        if not np.isfinite(threshold) or float(threshold) <= 0:
            raise ValueError(f"threshold must be positive and finite, got {threshold!r}")
        self._threshold = float(threshold)

    @property
    def is_notional(self) -> bool:
        return self.notional_per_bar is not None

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def accumulated(self) -> float:
        """Weight traded since the previous bar closed (the bar still filling)."""
        return float(self.cumulative - self.segment_base)

    @property
    def provisional(self) -> Optional[dict]:
        """The bar still filling (a copy): its high/low/close may still change."""
        return None if self._partial is None else dict(self._partial)

    def push(self, close_time, open_: float, high: float, low: float,
             close: float, volume: float) -> Optional[dict]:
        """Feed one print; return a closed bar dict, or ``None``.

        The crossing test is ``cumulative >= segment_base + threshold`` on a
        running total that is **never reset** — literally the comparison
        :func:`_clock_groups` makes with ``searchsorted`` — so the streaming and
        the batch constructors cannot drift apart on a floating-point boundary.
        """
        print_weight = (float(close) * float(volume) if self.is_notional
                        else float(volume))
        if not np.isfinite(print_weight):
            raise ValueError("print weight must be finite")
        if self._partial is None:
            self._partial = {
                "open": float(open_), "high": float(high), "low": float(low),
                "close": float(close), "volume": float(volume),
                "close_time": pd.Timestamp(close_time), "bar_id": self.bar_id,
            }
        else:
            bar = self._partial
            bar["high"] = max(bar["high"], float(high))
            bar["low"] = min(bar["low"], float(low))
            bar["close"] = float(close)
            bar["volume"] += float(volume)
            bar["close_time"] = pd.Timestamp(close_time)
        self.cumulative += print_weight
        self.n_prints += 1
        if self.cumulative >= self.segment_base + self._threshold:
            closed = dict(self._partial)
            self._partial = None
            self.segment_base = self.cumulative
            self.bar_id += 1
            self.n_prints = 0
            return closed
        return None

    def push_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Feed every print of ``frame`` in index order; return the closed bars."""
        data = _prepare(frame)
        rows = []
        for timestamp, row in data.iterrows():
            closed = self.push(timestamp, row["open"], row["high"], row["low"],
                               row["close"], row["volume"])
            if closed is not None:
                rows.append(closed)
        return bars_from_rows(rows)

    def state(self) -> dict:
        """JSON-serialisable state (a checkpoint keeps the partial bar, too)."""
        return {
            "notional_per_bar": self.notional_per_bar,
            "volume_per_bar": self.volume_per_bar,
            "bar_id": self.bar_id,
            "n_prints": self.n_prints,
            "cumulative": self.cumulative,
            "segment_base": self.segment_base,
            "accumulated": self.accumulated,
            "partial": None if self._partial is None
            else {**self._partial, "close_time": str(self._partial["close_time"])},
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "VolumeClockBuilder":
        builder = cls(notional_per_bar=state.get("notional_per_bar"),
                      volume_per_bar=state.get("volume_per_bar"))
        builder.bar_id = int(state.get("bar_id", 0))
        builder.n_prints = int(state.get("n_prints", 0))
        builder.cumulative = float(state.get("cumulative", 0.0))
        builder.segment_base = float(state.get(
            "segment_base", builder.cumulative - float(state.get("accumulated", 0.0))))
        partial = state.get("partial")
        if partial:
            builder._partial = {
                "open": float(partial["open"]), "high": float(partial["high"]),
                "low": float(partial["low"]), "close": float(partial["close"]),
                "volume": float(partial["volume"]),
                "close_time": pd.Timestamp(partial["close_time"]),
                "bar_id": int(partial.get("bar_id", builder.bar_id)),
            }
        return builder


def bars_from_rows(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build the cache-layout frame from closed-bar dicts (index = close time)."""
    rows = list(rows)
    if not rows:
        return _empty_bars()
    index = pd.DatetimeIndex([pd.Timestamp(r["close_time"]) for r in rows],
                             name=INDEX_NAME)
    frame = pd.DataFrame(
        [{c: float(r[c]) for c in CACHE_COLUMNS} for r in rows], index=index)
    return frame.loc[:, list(CACHE_COLUMNS)].astype(float)


# ── causality measurement ────────────────────────────────────────────────

def historical_bars_unchanged(prefix_bars: pd.DataFrame,
                              full_bars: pd.DataFrame) -> dict:
    """How many bars built from a prefix the full frame revises.

    Every bar in ``prefix_bars`` must also exist in ``full_bars`` (matched on
    close time) with bit-identical open/high/low/close/volume.  Returns
    ``{"n_prefix", "n_compared", "n_changed", "per_column_changed",
    "max_abs_diff", "changed_times"}``; the P6-C acceptance criterion is
    ``n_changed == 0`` and ``n_compared == n_prefix``.

    ``per_column_changed`` exists because the streaming form and the batch form
    disagree on ``volume`` in the last bits (float summation order) while their
    boundaries and four prices are identical: a single ``n_changed`` would hide
    which of the two claims a number supports.
    """
    if len(prefix_bars) == 0:
        return {"n_prefix": 0, "n_compared": 0, "n_changed": 0,
                "per_column_changed": {}, "max_abs_diff": {},
                "changed_times": []}
    common = prefix_bars.index.intersection(full_bars.index)
    left = prefix_bars.loc[common, list(CACHE_COLUMNS)].to_numpy(dtype=float)
    right = full_bars.loc[common, list(CACHE_COLUMNS)].to_numpy(dtype=float)
    differs = left != right
    same_rows = ~np.any(differs, axis=1)
    changed = [str(ts) for ts, ok in zip(common, same_rows) if not ok]
    per_column = {column: int(differs[:, index].sum())
                  for index, column in enumerate(CACHE_COLUMNS)}
    max_abs = {column: (float(np.max(np.abs(left[:, index] - right[:, index])))
                        if len(common) else 0.0)
               for index, column in enumerate(CACHE_COLUMNS)}
    return {
        "n_prefix": int(len(prefix_bars)),
        "n_compared": int(len(common)),
        "n_changed": int(len(changed)),
        "per_column_changed": per_column,
        "max_abs_diff": max_abs,
        "changed_times": changed[:20],
    }


def as_of_consistency(frame: pd.DataFrame, *, kind: str = "dollar",
                      notional_per_bar: Optional[float] = None,
                      volume_per_bar: Optional[float] = None,
                      interval: str = "1h",
                      cuts: Sequence[int] | None = None) -> dict:
    """Feed every prefix of ``frame`` through the constructors and count revisions.

    ``cuts`` are prefix lengths (default: 8 evenly spread values including the
    full length).  The returned dict reports the worst prefix and the total
    number of changed bars, which is what a "no look-ahead" claim has to show.
    """
    data = _prepare(frame)
    n = len(data)
    if n == 0:
        return {"n_cuts": 0, "total_changed": 0, "worst": None}
    if cuts is None:
        cuts = sorted({max(1, int(round(n * f)))
                       for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.97, 1.0)})
    build = lambda sub: bars_as_of(  # noqa: E731 - a local dispatch, not API
        sub, sub.index[-1], kind=kind, notional_per_bar=notional_per_bar,
        volume_per_bar=volume_per_bar, interval=interval)
    full = build(data)
    total_changed = 0
    worst = None
    for cut in cuts:
        report = historical_bars_unchanged(build(data.iloc[:int(cut)]), full)
        total_changed += report["n_changed"]
        if worst is None or report["n_changed"] > worst["n_changed"]:
            worst = {"prefix": int(cut), **report}
    return {"n_cuts": len(cuts), "total_changed": total_changed, "worst": worst}


# ── distribution statistics ──────────────────────────────────────────────

def bar_returns(bars: pd.DataFrame, *, log: bool = True) -> pd.Series:
    """Close-to-close returns of a bar series (``log`` by default).

    The bar clock is not calendar time, so a return is per *bar*, not per
    second; comparisons between a dollar-bar and a time-bar series are therefore
    comparisons of differently-timed observations, which the documented verdict
    has to carry.
    """
    close = bars["close"].astype(float)
    if log:
        out = np.log(close).diff()
    else:
        out = close.pct_change()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out.name = "log_return" if log else "return"
    return out


def spacing_seconds(bars: pd.DataFrame) -> pd.Series:
    """Seconds between consecutive bar close times (indexed like ``bars[1:]``).

    Aligns with :func:`bar_returns`: both drop the first bar, so a spacing and
    the return that spans it share an index label.
    """
    close_time = pd.Series(bars.index, index=bars.index)
    spacing = close_time.diff().dt.total_seconds().iloc[1:]
    spacing.name = "spacing_seconds"
    return spacing


def print_gap_times(frame: pd.DataFrame, *, factor: float = 4.0) -> pd.DatetimeIndex:
    """Timestamps of prints that arrive after a **hole in the print series**.

    A hole is a spacing larger than ``factor ×`` the median print spacing (for a
    1-minute cache: more than 4 minutes).  Measured on ``BTCUSDT/1m``: 23 such
    holes, 166 476 missing minutes = 24 % of the 697 719-minute span, the two
    largest 61.8 and 41.6 days.
    """
    data = _prepare(frame)
    if len(data) < 3:
        return pd.DatetimeIndex([], name=data.index.name)
    spacing = pd.Series(data.index, index=data.index).diff().dt.total_seconds()
    median = float(spacing.median())
    if not np.isfinite(median) or median <= 0:
        return pd.DatetimeIndex([], name=data.index.name)
    return pd.DatetimeIndex(data.index[spacing > float(factor) * median],
                            name=data.index.name)


def gap_spanning_returns(frame: pd.DataFrame, bars: pd.DataFrame, *,
                         factor: float = 4.0) -> pd.Series:
    """Mask (aligned to :func:`bar_returns`) of returns that span a print hole.

    A time bar resampled across a multi-week hole prices a jump that never
    traded, so that return measures the hole rather than the market.  The mask is
    anchored on the **print series**, not on the bar series' own spacing: a
    per-series rule would flag activity droughts in a dollar-bar series (whose
    spacing is irregular by construction) and stay silent about them in a
    time-bar series, which is not a comparison.

    A return is flagged when a print hole falls strictly inside the interval its
    two closing timestamps span.
    """
    returns = bar_returns(bars)
    mask = pd.Series(False, index=returns.index)
    holes = print_gap_times(frame, factor=factor)
    if len(holes) == 0 or len(returns) == 0:
        return mask
    closes = bars.index[1:]                      # closes[k] ends return k
    positions = closes.searchsorted(holes, side="left")
    for position in np.unique(positions):
        if 0 <= int(position) < len(mask):
            mask.iloc[int(position)] = True
    return mask


def returns_excluding_gaps(frame: pd.DataFrame, bars: pd.DataFrame, *,
                           log: bool = True, factor: float = 4.0) -> pd.Series:
    """:func:`bar_returns` with the print-hole-spanning returns removed."""
    returns = bar_returns(bars, log=log)
    mask = gap_spanning_returns(frame, bars, factor=factor)
    return returns[~mask.reindex(returns.index).fillna(False).astype(bool)]


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def skewness(values) -> Optional[float]:
    """Moment (biased) skewness ``m3 / m2**1.5``; ``None`` below the row floor."""
    arr = _finite(values)
    if arr.size < MIN_STATS_ROWS:
        return None
    centered = arr - arr.mean()
    m2 = float((centered ** 2).mean())
    if m2 <= 0:
        return None
    return float((centered ** 3).mean() / m2 ** 1.5)


def excess_kurtosis(values) -> Optional[float]:
    """Moment (biased) excess kurtosis ``m4 / m2**2 - 3`` (0 for a normal)."""
    arr = _finite(values)
    if arr.size < MIN_STATS_ROWS:
        return None
    centered = arr - arr.mean()
    m2 = float((centered ** 2).mean())
    if m2 <= 0:
        return None
    return float((centered ** 4).mean() / m2 ** 2 - 3.0)


def jarque_bera(values) -> dict:
    """Jarque-Bera normality statistic with its **exact** ``chi2(2)`` p-value.

    ``JB = n/6 · (S² + (K−3)²/4)`` with the moment estimators above.  The
    survival function of ``chi2(2)`` is ``exp(-x/2)``, so ``p = exp(-JB/2)`` is
    exact rather than an approximation, and no SciPy is needed.  ``reject_5pct``
    is that p-value's 5 % decision.
    """
    arr = _finite(values)
    out = {"n": int(arr.size), "skew": None, "excess_kurtosis": None,
           "statistic": None, "p_value": None, "reject_5pct": None}
    if arr.size < MIN_STATS_ROWS:
        return out
    s = skewness(arr)
    k = excess_kurtosis(arr)
    if s is None or k is None:
        return out
    statistic = arr.size / 6.0 * (s ** 2 + (k ** 2) / 4.0)
    p_value = float(np.exp(-0.5 * statistic))
    out.update({"skew": float(s), "excess_kurtosis": float(k),
                "statistic": float(statistic), "p_value": p_value,
                "reject_5pct": bool(p_value < 0.05)})
    return out


def autocorrelation(values, lag: int = 1) -> Optional[float]:
    """Sample autocorrelation at ``lag`` (Pearson on overlapping pairs).

    ``None`` when fewer than :data:`MIN_STATS_ROWS` pairs survive or the series
    has no variance — "no evidence" is never reported as ``0.0``.
    """
    arr = _finite(values)
    lag = int(lag)
    if lag < 1 or arr.size <= lag + MIN_STATS_ROWS:
        return None
    left, right = arr[:-lag], arr[lag:]
    if left.std() <= 0 or right.std() <= 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def acf(values, lags: Sequence[int] = (1, 2, 3, 4, 5)) -> dict:
    """``{lag: acf or None}`` for the requested lags."""
    return {int(lag): autocorrelation(values, int(lag)) for lag in lags}


def volatility_clustering(returns, lags: Sequence[int] = (1, 2, 3, 4, 5)) -> dict:
    """ACF of ``|return|`` — the standard volatility-clustering diagnostic.

    Reported with the mean of the absolute-return ACFs so two samplings can be
    compared with one number; a *higher* value means more clustering.
    """
    absolute = np.abs(_finite(returns))
    values = acf(absolute, lags)
    usable = [v for v in values.values() if v is not None]
    return {"lags": values,
            "mean_abs_acf": float(np.mean(usable)) if usable else None,
            "n": int(absolute.size)}


def return_stats(returns, *, lags: Sequence[int] = (1, 2, 3, 4, 5)) -> dict:
    """Skew / excess kurtosis / Jarque-Bera / ACF / volatility clustering.

    One call, one dict — the table row P6-C asks for.
    """
    arr = _finite(returns)
    jb = jarque_bera(arr)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else None,
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "skew": jb["skew"],
        "excess_kurtosis": jb["excess_kurtosis"],
        "jarque_bera": jb["statistic"],
        "jarque_bera_p": jb["p_value"],
        "jb_reject_5pct": jb["reject_5pct"],
        "acf_returns": acf(arr, lags),
        "acf_abs_returns": volatility_clustering(arr, lags)["lags"],
        "mean_abs_acf": volatility_clustering(arr, lags)["mean_abs_acf"],
    }


# ── resampling uncertainty ───────────────────────────────────────────────

def _default_block(n: int) -> int:
    """Circular block length: ``round(sqrt(n))`` bounded to ``[2, n//4]``.

    The square-root rule is the one used for distributional (tail) statistics:
    it preserves enough local dependence for the kurtosis/JB bootstrap to be
    honest, while a block of ``n`` (the degenerate case) would resample the same
    series every time and report a zero-width interval.
    """
    n = int(n)
    if n < 4:
        return 2
    return int(max(2, min(n // 4, int(round(n ** 0.5)))))


def _circular_block_sample(arr: np.ndarray, block: int,
                           rng: np.random.Generator) -> np.ndarray:
    n = arr.size
    if n == 0:
        return arr
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block)
    idx = (starts[:, None] + offsets[None, :]).ravel() % n
    return arr[idx[:n]]


def block_bootstrap_ci(values, stat_fn: Callable[[np.ndarray], float],
                       *, n_boot: int = 400, block: Optional[int] = None,
                       seed: int = 0, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap interval for ``stat_fn`` of one series.

    Fixed ``seed`` ⇒ bit-identical repeats.  The interval is a *sampling*
    interval under the block bootstrap's own dependence model; it is reported
    with its ``n_boot`` and ``block`` so the assumption is visible.
    """
    arr = _finite(values)
    out = {"point": None, "lo": None, "hi": None, "n": int(arr.size),
           "n_boot": int(n_boot), "block": None, "alpha": float(alpha)}
    if arr.size < MIN_STATS_ROWS:
        return out
    point = stat_fn(arr)
    if point is None or not np.isfinite(point):
        return out
    block = _default_block(arr.size) if block is None else int(block)
    block = max(1, min(int(block), arr.size))
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(n_boot)):
        value = stat_fn(_circular_block_sample(arr, block, rng))
        if value is not None and np.isfinite(value):
            draws.append(float(value))
    out["point"] = float(point)
    out["block"] = block
    if len(draws) >= 2:
        lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
        out["lo"], out["hi"] = float(lo), float(hi)
    return out


def bootstrap_metric_difference(a, b, stat_fn: Callable[[np.ndarray], float],
                                *, n_boot: int = 400,
                                block: Optional[int] = None, seed: int = 0,
                                alpha: float = 0.05) -> dict:
    """Block-bootstrap interval for ``stat_fn(a) - stat_fn(b)``.

    The two series are resampled **independently** (they are two samplings of
    the same underlying price process, not paired observations), so the interval
    is the sampling uncertainty of the difference, not a paired test.  A
    difference whose interval excludes 0 is the only shape that supports a
    "the distribution improved" claim; the P6-C verdict is written from that,
    and a straddling interval is reported as *no measurable improvement*.
    """
    left, right = _finite(a), _finite(b)
    out = {"point": None, "lo": None, "hi": None, "n_a": int(left.size),
           "n_b": int(right.size), "n_boot": int(n_boot), "block": None,
           "alpha": float(alpha), "excludes_zero": None}
    if left.size < MIN_STATS_ROWS or right.size < MIN_STATS_ROWS:
        return out
    point_a, point_b = stat_fn(left), stat_fn(right)
    if point_a is None or point_b is None:
        return out
    if not (np.isfinite(point_a) and np.isfinite(point_b)):
        return out
    block = _default_block(max(left.size, right.size)) if block is None \
        else int(block)
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(n_boot)):
        va = stat_fn(_circular_block_sample(left, max(1, min(block, left.size)), rng))
        vb = stat_fn(_circular_block_sample(right, max(1, min(block, right.size)), rng))
        if va is None or vb is None:
            continue
        if np.isfinite(va) and np.isfinite(vb):
            draws.append(float(va) - float(vb))
    out["point"] = float(point_a - point_b)
    out["block"] = int(block)
    if len(draws) >= 2:
        lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
        out["lo"], out["hi"] = float(lo), float(hi)
        out["excludes_zero"] = bool(lo > 0.0 or hi < 0.0)
    return out


__all__ = [
    "CACHE_COLUMNS", "INDEX_NAME", "MIN_STATS_ROWS",
    "dollar_bars", "volume_bars", "time_bars", "bars_as_of",
    "bars_from_rows", "VolumeClockBuilder",
    "notional_threshold", "volume_threshold", "interval_for_bar_count",
    "match_time_interval",
    "bar_returns", "spacing_seconds", "print_gap_times",
    "gap_spanning_returns", "returns_excluding_gaps", "skewness",
    "excess_kurtosis", "jarque_bera",
    "autocorrelation", "acf", "volatility_clustering", "return_stats",
    "block_bootstrap_ci", "bootstrap_metric_difference",
    "historical_bars_unchanged", "as_of_consistency",
]
