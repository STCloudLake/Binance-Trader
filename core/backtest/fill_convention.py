"""The backtest **fill convention** (P9): when is a fill priced?

The engine decides and fills on the same bar's close: the entry signal is
evaluated on the bar at ``ts`` and the fill price is *that same bar's* close
(``core/backtest/engine.py``: ``price = float(df_primary["close"].iloc[-1])``).
That is **zero execution latency** — it assumes you can see a close and trade at
exactly that price.  The conventional treatment is: signal at the close of
``ts``, fill at the **open of the next bar**.

This module is the single, testable definition of the two conventions:

``close``      the shipped default — the fill is the close of the decision bar.
               Bit-identical to the pre-P9 engine.
``next_open``  the fill is the *open of the bar immediately following* the
               decision bar, on the **same series** that priced the fill (the
               strategy's primary/shortest timeframe for an entry; the timeframe
               whose close supplied the exit price for an exit).  "One bar" is
               therefore always one row of the *fill timeframe's own* frame, not
               one row of the feeder's union grid — which matters as soon as
               several timeframes are loaded (a 4h filter never prices a fill).

Nothing here computes indicators, sizes positions or reads the database: it
parses the config value and turns a price frame + a decision timestamp into the
next bar's open.  An unknown convention raises the NAMED
:class:`UnknownFillConventionError` at config load, exactly like
``ga.benchmark_mode`` raises ``UnknownBenchmarkModeError``.

End of the loaded window
------------------------
Every frame the feeder hands out is trimmed to ``date_end`` (see
:mod:`core.backtest.data_feeder`), so the **last decision bar has no following
bar**.  :func:`next_bar_open` returns ``None`` there and the *caller* decides
explicitly: the engine refuses an entry that could not be filled and counts it,
and prices an exit at the last close while counting the fallback.  Neither case
is silent.
"""
from __future__ import annotations

import math

#: Every accepted ``backtest.fill_convention`` value, in contract order.
FILL_CONVENTIONS: tuple[str, ...] = ("close", "next_open")

#: The shipped default.  Changing this **changes every backtest number**.
DEFAULT_FILL_CONVENTION = "close"


class UnknownFillConventionError(ValueError):
    """Raised at config load for a ``backtest.fill_convention`` that is not one
    of :data:`FILL_CONVENTIONS`.  A named type so an operator (and a test) can
    tell "you typed nonsense" apart from "the run failed"."""


class FillConventionUnsupportedError(ValueError):
    """Raised when the SELECTED engine cannot honour the requested convention.

    The hybrid engine has no fill seam (``core/backtest/engine_hybrid.py`` /
    ``event_executor.py`` price every fill from the decision bar's close), so a
    ``next_open`` run routed to it must fail loudly instead of producing a run
    labelled ``next_open`` whose numbers are ``close``.
    """


def parse_fill_convention(raw) -> str:
    """Return the validated convention for a raw config/job value.

    ``None`` (the key absent) → :data:`DEFAULT_FILL_CONVENTION`, so a config
    written before P9 keeps the historical ``close`` behaviour.  Anything that is
    not one of :data:`FILL_CONVENTIONS` raises
    :class:`UnknownFillConventionError` naming both the bad value and the valid
    ones — never a silent fallback, because the two conventions produce
    different numbers.
    """
    if raw is None:
        return DEFAULT_FILL_CONVENTION
    if not isinstance(raw, str):
        raise UnknownFillConventionError(
            f"backtest.fill_convention must be a string, got {type(raw).__name__} "
            f"({raw!r}); valid values: {', '.join(FILL_CONVENTIONS)}")
    value = raw.strip().lower()
    if value not in FILL_CONVENTIONS:
        raise UnknownFillConventionError(
            f"unknown backtest.fill_convention {raw!r}; valid values: "
            f"{', '.join(FILL_CONVENTIONS)} (shipped default "
            f"{DEFAULT_FILL_CONVENTION!r})")
    return value


def decision_bar_position(frame, ts) -> int:
    """Position of the bar a decision at ``ts`` was made on, or ``-1``.

    The engine's own slicing rule: the last row with ``index <= ts``
    (``df[df.index <= ts].iloc[-1]``), expressed as ``searchsorted`` so it is
    O(log n) and identical for an exact hit and a ragged grid.
    """
    if frame is None or len(frame) == 0:
        return -1
    return int(frame.index.searchsorted(ts, side="right")) - 1


def next_bar_open(frame, ts) -> float | None:
    """Open of the bar **immediately after** the decision bar at ``ts``.

    ``frame`` must be the *full* (feeder-trimmed) price frame, not a slice ending
    at ``ts`` — the whole point is the bar that comes after it.

    Returns ``None`` when the decision bar is the last row of ``frame`` (a fill
    that would fall outside the loaded window) or when the open is unusable
    (missing key, non-finite or non-positive).  The caller must handle ``None``
    explicitly; this function never substitutes a price.
    """
    pos = decision_bar_position(frame, ts)
    if pos < 0:
        return None
    nxt = pos + 1
    if nxt >= len(frame):
        return None
    try:
        value = float(frame.iloc[nxt]["open"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value
