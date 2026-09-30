"""Regression: ``--merge`` must not store one bar twice under two conventions.

Why this file exists
--------------------
The cache can hold two timestamp conventions at once: ``scripts/download_history.py``
stamps every row with Binance's ``close_time`` (``open + length - 1 ms``, so a 1h
bar opens at ``12:00`` and is stamped ``12:59:59.999``), while an earlier writer
left bar-**open** stamps (``12:00:00``).  The repaired
``data/market/BTCUSDT/1h.parquet`` at revision ``fa028be`` holds both: 8 767
open-aligned rows plus 2 910 ``:59:59.999`` rows, i.e. **55 bars present under
both conventions**.  The loader used to union on the *exact* index
(``~df.index.duplicated``), which cannot see that pair — every extra ``--merge``
would have appended the same bar again.

The trap this file also pins: in that cache the ``1 ms`` step is **not** the
duplicate marker.  All 54 one-millisecond-adjacent pairs are the close of hour
``H-1`` (``…(H-1):59:59.999``) beside the open of hour ``H`` (``…H:00:00``) — two
*different* bars — so a proximity rule would delete real rows.  The bar length is
the only sound key, and both rows of such a pair must survive.

Everything here is synthetic and in-memory (plus one ``tmp_path`` parquet
round-trip): no network, and the live cache is never read or written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.download_history import bar_open_keys, merge_bars

INTERVAL = "1h"
STEP = pd.Timedelta(hours=1)
CLOSE_OFFSET = STEP - pd.Timedelta(milliseconds=1)


def _frame(stamps, base: float = 100.0) -> pd.DataFrame:
    """One OHLCV row per stamp; values encode the stamp's provenance."""
    idx = pd.DatetimeIndex(stamps)
    n = len(idx)
    return pd.DataFrame(
        {
            "open": base + np.arange(n, dtype=float),
            "high": base + np.arange(n, dtype=float) + 1.0,
            "low": base + np.arange(n, dtype=float) - 1.0,
            "close": base + np.arange(n, dtype=float) + 0.5,
            "volume": 10.0 + np.arange(n, dtype=float),
        },
        index=idx,
    )


def _keys(frame: pd.DataFrame, interval: str = INTERVAL) -> pd.DatetimeIndex:
    """Keys with the convention detected from the frame (the merge's fallback)."""
    return bar_open_keys(frame.index, interval)


def _opens(frame: pd.DataFrame, ref, interval: str = INTERVAL) -> pd.DatetimeIndex:
    """True bar-open keys, anchored on a known close-time stamp ``ref``."""
    return bar_open_keys(frame.index, interval, reference=ref)


def test_mixed_convention_duplicate_bars_are_collapsed():
    """A bar stored as both ``12:00`` and ``12:59:59.999`` becomes one row."""
    hours = pd.date_range("2026-01-01 00:00", periods=24, freq="h")
    old = _frame(hours)                                     # bar-open convention
    mixed = pd.concat([old, _frame(hours[6:12] + CLOSE_OFFSET, base=200.0)])
    before = len(mixed)
    assert before == 30

    merged = merge_bars(mixed.iloc[:0], mixed, INTERVAL)

    assert len(merged) == 24, (before, len(merged))
    assert _keys(merged).is_unique, "a bar survived under both timestamp conventions"
    assert merged.index.is_monotonic_increasing
    # Range/rows otherwise unchanged: the union of bars is exactly the 24 hours.
    assert merged.index.min() == hours[0]
    assert merged.index.max() == hours[-1]
    opens = _opens(merged, hours[6] + CLOSE_OFFSET)
    assert set(opens) == set(hours)
    # The later row wins a conflict, so the collided bars keep this script's stamp.
    assert merged.index[6] == hours[6] + CLOSE_OFFSET
    assert merged["close"].iloc[6] == 200.5


def test_close_of_previous_hour_and_open_of_next_hour_both_survive():
    """The 1 ms trap: two rows 1 ms apart are different bars, not duplicates."""
    prev_close = pd.Timestamp("2026-01-01 00:59:59.999")
    next_open = pd.Timestamp("2026-01-01 01:00:00")
    assert (next_open - prev_close) == pd.Timedelta(milliseconds=1)
    frame = _frame([prev_close, next_open])

    merged = merge_bars(frame.iloc[:0], frame, INTERVAL)

    assert len(merged) == 2, "a proximity-based rule deleted a real bar"
    assert set(merged.index) == {prev_close, next_open}
    assert _keys(merged).is_unique, "the two rows must not share a bar key"
    assert _opens(merged, next_open + CLOSE_OFFSET).tolist() == [
        pd.Timestamp("2026-01-01 00:00"), pd.Timestamp("2026-01-01 01:00")]


def test_fresh_download_updates_a_bar_in_the_other_convention():
    """``new`` wins on a conflict, so ``--merge`` updates instead of adding."""
    hour = pd.Timestamp("2026-01-01 12:00")
    old = _frame([hour], base=100.0)                       # legacy open stamp
    new = _frame([hour + CLOSE_OFFSET], base=500.0)        # this script's stamp

    merged = merge_bars(old, new, INTERVAL)

    assert len(merged) == 1
    assert merged.index[0] == hour + CLOSE_OFFSET, (
        "the freshly downloaded row must keep the documented close_time index")
    assert merged["close"].iloc[0] == 500.5


def test_second_merge_is_idempotent_and_preserves_range_and_rows():
    """Re-running the downloader over an already-merged file adds nothing."""
    old = _frame(pd.date_range("2026-01-01 00:00", periods=24, freq="h"))
    old = pd.concat([old, _frame(pd.date_range("2026-01-02 00:00", periods=3,
                                               freq="h") + CLOSE_OFFSET)])
    new = _frame(pd.date_range("2026-01-02 00:00", periods=6, freq="h")
                 + CLOSE_OFFSET, base=900.0)

    first = merge_bars(old, new, INTERVAL)
    second = merge_bars(first, new, INTERVAL)

    assert len(first) == len(second), "a repeated merge appended duplicates"
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_index_equal(first.index, second.index)
    assert _keys(first).is_unique
    assert first.index.min() == old.index.min()
    assert first.index.max() == new.index.max()
    # Bars that only the legacy frame carries keep their own stamp and values.
    untouched = pd.Timestamp("2026-01-01 00:00")
    assert untouched in first.index
    assert first.loc[untouched, "close"] == old.loc[untouched, "close"]


def test_parquet_round_trip_on_a_temp_directory(tmp_path):
    """The real read/merge/write path, on a temp dir — never the live cache."""
    path = tmp_path / "market" / "BTCUSDT" / "1h.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    hours = pd.date_range("2026-01-01 00:00", periods=48, freq="h")
    stored = pd.concat([_frame(hours),
                        _frame(hours[10:20] + CLOSE_OFFSET, base=300.0)])
    assert len(stored) == 58
    stored.to_parquet(path)

    loaded = pd.read_parquet(path)
    loaded.index = pd.to_datetime(loaded.index)
    fresh_hours = pd.date_range("2026-01-03 00:00", periods=6, freq="h")
    fresh = _frame(fresh_hours + CLOSE_OFFSET, base=700.0)
    merged = merge_bars(loaded, fresh, INTERVAL)
    merged.to_parquet(path)
    reread = pd.read_parquet(path)

    keys = _keys(reread)
    assert len(reread) == 54, len(reread)  # 48 old hours + 6 fresh hours
    assert keys.is_unique
    assert pd.DatetimeIndex(reread.index).min() == hours[0]
    assert pd.DatetimeIndex(reread.index).max() == fresh_hours[-1] + CLOSE_OFFSET
    # A second merge over the reparsed file is a no-op (the --merge regression).
    again = merge_bars(pd.read_parquet(path), fresh, INTERVAL)
    assert len(again) == 54
    assert set(_keys(again)) == set(keys)


@pytest.mark.parametrize("interval,offset,freq", [
    ("1m", pd.Timedelta(milliseconds=59_999), "min"),
    ("15m", pd.Timedelta(milliseconds=899_999), "15min"),
    ("1d", pd.Timedelta(milliseconds=86_399_999), "D"),
    ("1w", pd.Timedelta(milliseconds=604_799_999), "7D"),
])
def test_other_interval_labels_get_the_same_key(interval, offset, freq):
    """Both conventions map to one bar key for the other fixed-length labels."""
    opens = pd.date_range("2026-01-05 00:00", periods=4, freq=freq)
    legacy = _frame(opens)
    fresh = _frame(opens + offset, base=400.0)

    merged = merge_bars(legacy, fresh, interval)

    assert len(merged) == 4
    assert _keys(merged, interval).is_unique
    assert set(bar_open_keys(legacy.index, interval,
                             reference=fresh.index[0])) == set(opens)


def test_monthly_label_floors_to_the_calendar_month():
    """``1M`` has no fixed length: both conventions floor to the month start."""
    opens = pd.DatetimeIndex(["2026-01-01", "2026-02-01"])
    closes = pd.DatetimeIndex(["2026-01-31 23:59:59.999",
                               "2026-02-28 23:59:59.999"])
    legacy = _frame(opens)
    fresh = _frame(closes, base=400.0)

    merged = merge_bars(legacy, fresh, "1M")

    assert len(merged) == 2
    assert bar_open_keys(merged.index, "1M").is_unique
    keys = set(bar_open_keys(pd.concat([legacy, fresh]).index, "1M"))
    assert keys == set(opens)
