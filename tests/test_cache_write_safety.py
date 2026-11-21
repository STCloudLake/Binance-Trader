"""The parquet cache's write path must be able to *refuse*, not just merge.

Why this file exists (the 2026-11-21 incident)
----------------------------------------------
The parquet cache tree lost 27 symbol directories and every surviving file was
cut down to one REST prefetch window (``1m`` 4 035 rows, ``5m`` 2 007, ``15m``
1 002, ``1h``/``4h`` 1 000 — exactly ``INTERVAL_SPEC[interval]["batches"] * 1000``
plus the live bars that closed afterwards).  The write path had two holes that
make that shape reachable from inside the process:

1. :meth:`OHLVCache._read_disk` mapped **every** read error to ``None`` — the
   same answer as "the file is not there".  ``save`` then treated the file as new
   and wrote the short in-memory window over a full history.  A file that a
   ``taskkill /F`` left half-written (``to_parquet`` is not atomic) is *exactly*
   an unreadable file, so the next start's prefetch could replace the whole
   research cache with its ``batches x 1000`` window.
2. Nothing in the write path compared the row count it was about to write with
   the row count already on disk, so no layer below ``merge_history`` could stop
   a shrinking write.

The contract these tests pin:

(i)   an existing-but-unreadable file is **never** overwritten — the bytes are
      left exactly as they are, an error is logged, and the refusal is visible
      on the cache (``refused_writes`` / ``read_failures``);
(ii)  the write path refuses any frame with **fewer rows than the file**, even
      when a buggy merge hands it one (``CacheWriteRefused``);
(iii) a write goes to a temp file in the same directory and is published with
      ``os.replace``, so a kill mid-write cannot corrupt the authoritative file;
(iv)  no cache path deletes a symbol directory that is not in the watchlist
      (there is no pruning path in this tree; this test is the regression guard
      for that finding).

All of it runs on ``tmp_path``; nothing reads or writes the shipped cache.  This
module is therefore deliberately **not** in ``LIVE_CACHE_TEST_FILES`` (it never
touches the live tree, and the policy scanner keys on that tree's path literal).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

SYMBOL = "BTCUSDT"


def _frame(n: int, *, start: str = "2026-01-01", freq: str = "1min",
           seed: int = 20261121, close0: float = 100.0) -> pd.DataFrame:
    """Deterministic OHLCV frame (a random walk, no network)."""
    idx = pd.date_range(start, periods=n, freq=freq)
    close = close0 + np.cumsum(np.random.default_rng(seed).normal(0.0, 0.5, n))
    return pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                         "close": close, "volume": 1.0}, index=idx)


def _path(root, symbol: str = SYMBOL, interval: str = "1m"):
    return root / "market" / symbol / f"{interval}.parquet"


def _write(root, frame: pd.DataFrame, interval: str = "1m", symbol: str = SYMBOL):
    path = _path(root, symbol=symbol, interval=interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


def _rows(path) -> int:
    return len(pd.read_parquet(path))


def _tmp_leftovers(directory) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp"))


# ══════════════════════════════════════════════════════════════════════
# (i) an unreadable existing file is never overwritten
# ══════════════════════════════════════════════════════════════════════

#: What ``taskkill /F`` leaves behind in the middle of a parquet write: a header
#: and a few bytes of a page, no footer.  ``pd.read_parquet`` raises ``ArrowInvalid``.
GARBAGE = b"PAR1\x00\x00\x00\x00half-written page, no footer\x00\x00PAR1"


def test_save_refuses_to_overwrite_an_unreadable_file(tmp_path):
    """The prefetch path: a short window must not replace a corrupt full history.

    This is the incident's shape — a fresh process whose ``_read_disk`` fails,
    holding only the ``batches x 1000`` window it just fetched.  Before the fix
    the file became 120 rows.
    """
    from core.market_data.ohlcv_cache import OHLVCache

    root = tmp_path / "data"
    full = _frame(5_000)
    path = _write(root, full)
    path.write_bytes(GARBAGE)                    # a kill mid-write

    cache = OHLVCache(str(root))
    window = _frame(120, start="2026-09-20")     # the prefetch result
    cache.update(SYMBOL, "1m", window)
    wrote = cache.save(SYMBOL, "1m")

    assert path.read_bytes() == GARBAGE, (
        "an unreadable existing cache file was overwritten with the in-memory "
        f"window: it was {len(full)} rows and is now "
        f"{_rows(path) if path.read_bytes()[:4] == b'PAR1' else 'unreadable'}")
    assert wrote is False
    assert cache.refused_writes, "the refusal was not surfaced on the cache"
    assert cache.read_failures, "the read failure was not surfaced on the cache"
    assert SYMBOL in cache.refused_writes[-1] and "1m" in cache.refused_writes[-1]


def test_flush_refuses_to_overwrite_an_unreadable_file(tmp_path):
    """The periodic flush: a stale short frame must not replace a corrupt file."""
    from core.market_data.ohlcv_cache import OHLVCache

    root = tmp_path / "data"
    full = _frame(4_000)
    path = _write(root, full)

    cache = OHLVCache(str(root))
    cache.update(SYMBOL, "1m", full.tail(1_200))  # the stale window held in memory
    path.write_bytes(GARBAGE)                    # corrupted underneath the process
    cache.append_candle(SYMBOL, "1m", {
        "close_time": int((full.index[-1] + pd.Timedelta(minutes=1)).value // 10**6),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    cache.flush_all()

    assert path.read_bytes() == GARBAGE, (
        "the 5-minute flush overwrote an unreadable file with its stale "
        f"{len(cache.get(SYMBOL, '1m'))}-row window")
    assert cache.refused_writes


def test_a_permission_error_on_read_also_refuses_the_write(tmp_path, monkeypatch):
    """A file another process holds open is "unreadable", not "absent"."""
    from core.market_data import ohlcv_cache

    root = tmp_path / "data"
    full = _frame(300)
    path = _write(root, full)
    before = path.read_bytes()

    def _locked(*args, **kwargs):
        raise PermissionError("the process cannot access the file: it is locked")

    monkeypatch.setattr(pd, "read_parquet", _locked)
    cache = ohlcv_cache.OHLVCache(str(root))
    cache.update(SYMBOL, "1m", _frame(30, start="2026-09-20"))
    wrote = cache.save(SYMBOL, "1m")
    monkeypatch.undo()

    assert path.read_bytes() == before, (
        "a locked/unreadable cache file was overwritten with the in-memory "
        f"window ({len(_frame(30, start='2026-09-20'))} rows) instead of being "
        "left alone")
    assert wrote is False
    assert cache.refused_writes and cache.read_failures


# ══════════════════════════════════════════════════════════════════════
# (ii) the hard shrink invariant
# ══════════════════════════════════════════════════════════════════════

def test_the_write_path_refuses_a_frame_smaller_than_the_file(tmp_path, monkeypatch):
    """Defence in depth: even a buggy merge cannot shrink the file.

    ``merge_history`` is replaced by a stub that returns ten rows, i.e. exactly
    the input ``save`` must never accept.  Before the fix the file became 10 rows.
    """
    from core.market_data import ohlcv_cache

    root = tmp_path / "data"
    full = _frame(400)
    path = _write(root, full)

    cache = ohlcv_cache.OHLVCache(str(root))
    cache.get(SYMBOL, "1m")                      # 400 rows in memory
    short = full.tail(10)
    monkeypatch.setattr(ohlcv_cache, "merge_history",
                        lambda existing, incoming, interval=None: short)

    wrote = cache.save(SYMBOL, "1m")
    assert _rows(path) == 400, (
        "the write path shrank the cache file from 400 rows to "
        f"{_rows(path)} rows with no refusal")
    assert wrote is False
    assert any("shrink" in reason for reason in cache.refused_writes)


def test_guard_no_shrink_raises_the_named_error(tmp_path):
    """The invariant is a named exception, not a silent skip."""
    from core.market_data.ohlcv_cache import CacheWriteRefused, guard_no_shrink

    path = _path(tmp_path / "data")
    with pytest.raises(CacheWriteRefused) as err:
        guard_no_shrink(path, _frame(400), _frame(10), where="save")
    assert "shrink" in str(err.value)
    assert err.value.existing_rows == 400 and err.value.incoming_rows == 10
    # The same frame against an *absent* file is allowed (a new file may be short).
    assert guard_no_shrink(path, None, _frame(10), where="save") == 10


def test_the_shrink_guard_allows_a_twin_convention_collapse(tmp_path):
    """Fewer rows is legitimate when no *bar* disappeared.

    The interval-aware union collapses one bar stored under both timestamp
    conventions, which is :func:`merge_history`'s documented job (455 rows / 401
    bars measured on the live ``1h`` cache).  The guard must not block it, or the
    re-audit dedupe stops working — while a merge that loses a *bar* is still
    refused.
    """
    from core.market_data.ohlcv_cache import CacheWriteRefused, guard_no_shrink

    path = _path(tmp_path / "data")
    bars = _frame(400, freq="1h")
    twins = bars.copy()
    twins.index = twins.index + pd.Timedelta(milliseconds=3_599_999)  # close_time
    doubled = pd.concat([bars, twins])
    assert len(doubled) == 800 and len(bars) == 400

    # 800 rows → 400 rows: no bar lost, allowed.
    assert guard_no_shrink(path, doubled, bars, interval="1h") == 400
    # 400 bars → 399 bars: one bar lost, refused even though rows are unchanged.
    with pytest.raises(CacheWriteRefused):
        guard_no_shrink(path, bars, bars.iloc[1:], interval="1h")


# ══════════════════════════════════════════════════════════════════════
# (iii) atomic write: temp file in the same directory, then os.replace
# ══════════════════════════════════════════════════════════════════════

def test_a_failed_write_leaves_the_authoritative_file_intact(tmp_path, monkeypatch):
    """A kill in the middle of ``to_parquet`` must not destroy the old file.

    The stub writes a few bytes to whatever path it is given and raises — the
    on-disk signature of a force-kill.  Before the fix that path *is* the
    authoritative file, so the history was gone.
    """
    from core.market_data.ohlcv_cache import OHLVCache

    root = tmp_path / "data"
    full = _frame(300)
    path = _write(root, full)
    before = path.read_bytes()

    cache = OHLVCache(str(root))
    cache.get(SYMBOL, "1m")
    cache.append_candle(SYMBOL, "1m", {
        "close_time": int((full.index[-1] + pd.Timedelta(minutes=1)).value // 10**6),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})

    def _killed(self, target, *args, **kwargs):
        with open(os.fspath(target), "wb") as fh:
            fh.write(b"half-written")
        raise RuntimeError("simulated SIGKILL during to_parquet")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _killed)
    with pytest.raises(RuntimeError):
        cache.save(SYMBOL, "1m")
    monkeypatch.undo()

    assert path.read_bytes() == before, (
        "a write that failed mid-way replaced the authoritative file with a "
        f"{path.stat().st_size}-byte fragment (the frame was {len(full)} rows)")
    assert _tmp_leftovers(path.parent) == []
    assert _rows(path) == 300


def test_the_write_goes_through_a_temp_file_and_os_replace(tmp_path, monkeypatch):
    """The mechanism, asserted directly: same-directory temp path then replace."""
    from core.market_data import ohlcv_cache

    root = tmp_path / "data"
    seeded = _frame(50)
    path = _write(root, seeded)
    cache = ohlcv_cache.OHLVCache(str(root))
    cache.get(SYMBOL, "1m")
    cache.append_candle(SYMBOL, "1m", {
        "close_time": int((seeded.index[-1] + pd.Timedelta(minutes=1)).value // 10**6),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})

    calls: list[tuple] = []
    real_replace = os.replace

    def _spy(src, dst, *args, **kwargs):
        calls.append((os.fspath(src), os.fspath(dst)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(ohlcv_cache.os, "replace", _spy)
    assert cache.save(SYMBOL, "1m") is True
    monkeypatch.undo()

    assert calls, ("the write called to_parquet on the authoritative path "
                   "directly; no temp-file + os.replace happened")
    src, dst = calls[0]
    assert dst == os.fspath(path)
    assert os.path.dirname(src) == os.path.dirname(os.fspath(path)), (
        f"the temp file {src} is not in the target directory, so os.replace "
        "would not be atomic")
    assert src != dst and src.endswith(".tmp")
    assert not os.path.exists(src)
    assert _tmp_leftovers(path.parent) == []
    assert _rows(path) == 51


# ══════════════════════════════════════════════════════════════════════
# (iv) no pruning path: an out-of-watchlist symbol directory survives
# ══════════════════════════════════════════════════════════════════════

def test_flush_never_deletes_a_symbol_directory_outside_the_watchlist(tmp_path):
    """There is no cache-pruning path in this tree; this keeps it that way.

    ``ADAUSDT`` is not in ``DEFAULT_WATCHLIST`` and is never touched by the cache
    under test, yet a flush of the watchlist symbols must leave it alone.
    """
    from core.market_data.ohlcv_cache import OHLVCache

    root = tmp_path / "data"
    ada = _write(root, _frame(200, freq="1h"), interval="1h", symbol="ADAUSDT")
    seeded = _frame(100, freq="1h")
    btc = _write(root, seeded, interval="1h")

    cache = OHLVCache(str(root))
    cache.get(SYMBOL, "1h")
    cache.append_candle(SYMBOL, "1h", {
        "close_time": int((seeded.index[-1] + pd.Timedelta(hours=1)).value // 10**6),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    cache.flush_all()

    assert ada.exists(), "flush deleted a symbol directory outside the watchlist"
    assert _rows(ada) == 200
    assert btc.exists() and _rows(btc) == 101
