"""Cache durability: a cache write must never shrink the on-disk history.

Why this file exists (the measured defect)
------------------------------------------
``data/market/BTCUSDT/1h.parquet`` was repaired to **11 675 rows / 0 gaps** at
12:35 and was **8 848 rows with the 1 484 h splice back** at 13:01.  The writer
did it: ``OHLVCache`` loads the parquet once (``get``) and afterwards only
appends live candles, and ``save`` wrote that in-memory frame *verbatim* — so the
running service's 5-minute ``flush_all`` (``MarketDataProvider._run_cache_flush``)
put its stale 8 848-row snapshot back over the repair.  A second path had the same
shape: ``_prefetch_history`` fetched ``batches x 1000`` candles and replaced the
file, so an existing file below the interval's ``min_candles`` skip threshold was
truncated to the fetched window (measured on a temp dir: a 6 000-row ``1m`` file
became 1 000 rows).

The contract these tests pin — every write is a **union** of the on-disk frame and
the in-memory frame over ``close_time`` (``OHLVCache.save`` → ``merge_history``):

(i)   a write never shrinks the file,
(ii)  the wider range (and the values inside it) survives,
(iii) a rewritten file stays gap-free when the source was gap-free,
(iv)  the rewrite is deterministic (same inputs → same output).

All of it runs on ``tmp_path`` with a stubbed market-data client — no network, no
sleep, no read of the shipped cache.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

SYMBOL = "BTCUSDT"
SEED = 20260930


@pytest.fixture(autouse=True)
def _isolate_config():
    """Never leave a mutated ``Config`` singleton behind for the next test file."""
    from app.config import Config

    yield
    Config._instance = None


def _frame(n: int, *, start: str = "2025-06-03", freq: str = "1h",
           seed: int = SEED, close0: float = 100.0) -> pd.DataFrame:
    """Deterministic OHLCV frame (a random walk, no network)."""
    idx = pd.date_range(start, periods=n, freq=freq)
    close = close0 + np.cumsum(np.random.default_rng(seed).normal(0.0, 0.5, n))
    return pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                         "close": close, "volume": 1.0}, index=idx)


def _path(root, symbol: str = SYMBOL, interval: str = "1h"):
    """The parquet the cache reads/writes under a temp data dir."""
    return root / "market" / symbol / f"{interval}.parquet"


def _write(root, frame: pd.DataFrame, interval: str = "1h") -> None:
    path = _path(root, interval=interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _rows(root, interval: str = "1h") -> int:
    return len(pd.read_parquet(_path(root, interval=interval)))


def _gap_count(frame: pd.DataFrame, interval: str = "1h") -> int:
    from scripts.check_data_integrity import gap_report

    return gap_report(frame, interval)["gap_count"]


# ══════════════════════════════════════════════════════════════════════
# (i) + (ii) — the periodic flush of a stale in-memory frame
# ══════════════════════════════════════════════════════════════════════

def test_flush_of_a_stale_frame_never_shrinks_a_repaired_file(tmp_path):
    """The 13:01 defect, reproduced on a temp dir: repair in, spliced frame out.

    ``stale`` is the 8 848-row spliced frame the live service holds; ``repaired``
    is the 11 675-row gap-free file an operator writes underneath it.  The flush
    must keep every repaired bar — before the fix the file ended up 8 848 rows
    with the 1 484-bar hole back.
    """
    from core.market_data.ohlcv_cache import OHLVCache

    root = tmp_path / "data"
    full = _frame(10_332)
    stale = full.drop(full.index[2_000:3_484])          # 8 848 rows, 1 484-bar hole
    assert len(stale) == 8_848
    _write(root, stale)

    cache = OHLVCache(str(root))
    loaded = cache.get(SYMBOL, "1h")                    # service start: load once
    assert len(loaded) == 8_848

    repaired = _frame(11_675)                           # external repair
    _write(root, repaired)
    assert _rows(root) == 11_675

    new_ts = repaired.index[-1] + pd.Timedelta(hours=1)  # a live closed candle
    cache.append_candle(SYMBOL, "1h", {
        "close_time": int(new_ts.value // 10**6),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    cache.flush_all()                                   # every 300 s in the service

    merged = pd.read_parquet(_path(root))
    assert len(merged) == 11_676, (
        f"the stale flush truncated the repair: {len(merged)} < 11 675")
    assert merged.index[0] == repaired.index[0]         # wider range survived
    assert merged.index[-1] == new_ts                   # ... and it still grows
    # Nothing the repair added was lost, and the 1 484 bars the in-memory frame
    # never had kept the repaired values.
    assert repaired.index.difference(merged.index).empty
    hole = full.index[2_000:3_484]
    assert len(hole) == 1_484
    assert merged.loc[hole, "close"].equals(repaired.loc[hole, "close"])
    assert _gap_count(merged) == 0                      # the hole is not re-opened
    # The in-memory state converges on the widest history too.
    assert len(cache.get(SYMBOL, "1h")) == 11_676


def test_merge_keeps_the_newest_value_for_a_shared_timestamp(tmp_path):
    """Dedupe convention: the writer's row wins (``keep="last"``), as before."""
    from core.market_data.ohlcv_cache import OHLVCache

    root = tmp_path / "data"
    disk = _frame(10)
    _write(root, disk)

    cache = OHLVCache(str(root))
    corrected = disk.copy()
    corrected.loc[disk.index[3], "close"] = 999.0
    cache.update(SYMBOL, "1h", corrected)
    cache.save(SYMBOL, "1h")

    merged = pd.read_parquet(_path(root))
    assert len(merged) == len(disk)
    assert merged.loc[disk.index[3], "close"] == 999.0


# ══════════════════════════════════════════════════════════════════════
# (i)+(ii) — the provider prefetch path
# ══════════════════════════════════════════════════════════════════════

def test_prefetch_merges_a_longer_on_disk_file_instead_of_replacing_it(tmp_path):
    """``batches x 1000`` candles are a widening, not a replacement (no network).

    A ``1m`` file with 6 000 rows sits below the interval's 10 000 ``min_candles``
    skip threshold, so the prefetch runs; the fetched 1 000 rows are disjoint from
    it.  Before the fix the file became 1 000 rows.
    """
    from app.config import Config
    from app.event_bus import EventBus
    from core.market_data.provider import MarketDataProvider

    data_dir = tmp_path / "data"
    existing = _frame(6_000, start="2026-01-01", freq="1min")
    _write(data_dir, existing, interval="1m")

    cfg = Config.load("sim")
    cfg.data_dir = str(data_dir)
    provider = MarketDataProvider(cfg, EventBus())

    fetched_start = pd.Timestamp("2026-09-26")
    base_ms = int(fetched_start.value // 10**6)

    class _StubClient:
        """The two REST calls the prefetch makes, without a socket."""

        async def klines(self, symbol, interval, limit=1000, end_time=None):
            return [[base_ms + i * 60_000, 2.0, 2.0, 2.0, 2.0, 2.0,
                     base_ms + i * 60_000 + 59_999] for i in range(limit)]

        async def close(self):
            pass

    provider._data_client = _StubClient()
    asyncio.run(provider._prefetch_history([SYMBOL], ["1m"]))

    merged = pd.read_parquet(_path(data_dir, interval="1m"))
    assert len(merged) == 7_000, (
        f"the prefetch replaced the 6 000-row file: {len(merged)} rows")
    assert merged.index[0] == existing.index[0]
    assert merged.index[-1] == pd.Timestamp(
        base_ms + 999 * 60_000 + 59_999, unit="ms")
    assert merged.loc[existing.index, "close"].equals(existing["close"])
    assert len(provider.cache.get(SYMBOL, "1m")) == 7_000


# ══════════════════════════════════════════════════════════════════════
# (iii) — gap-freeness survives the rewrite
# ══════════════════════════════════════════════════════════════════════

def test_rewrite_keeps_a_gap_free_history_gap_free(tmp_path):
    """A shorter in-memory window (plus a new bar) cannot punch a hole in the file."""
    from core.market_data.ohlcv_cache import OHLVCache

    root = tmp_path / "data"
    disk = _frame(11_675)
    assert _gap_count(disk) == 0
    _write(root, disk)

    cache = OHLVCache(str(root))
    window = disk.tail(500).copy()                      # what a short-lived fetch saw
    window.loc[window.index[-1] + pd.Timedelta(hours=1)] = window.iloc[-1].values
    cache.update(SYMBOL, "1h", window)
    cache.save(SYMBOL, "1h")

    merged = pd.read_parquet(_path(root))
    assert len(merged) == 11_676
    assert merged.index[0] == disk.index[0]             # the hole is not re-opened
    assert _gap_count(merged) == 0
    assert merged.index.is_monotonic_increasing


# ══════════════════════════════════════════════════════════════════════
# (iv) — determinism, and the degenerate inputs
# ══════════════════════════════════════════════════════════════════════

def test_merge_is_deterministic_and_handles_empty_side(tmp_path):
    """Same inputs → same output; a missing/empty side is a pass-through."""
    from core.market_data.ohlcv_cache import merge_history

    wide, narrow = _frame(300), _frame(300).tail(20)
    assert merge_history(wide, narrow).equals(merge_history(wide, narrow))
    assert merge_history(None, narrow).equals(narrow)
    assert merge_history(narrow, None).equals(narrow)
    assert merge_history(None, None) is None
    assert merge_history(_frame(0), None) is None or len(
        merge_history(_frame(0), None)) == 0

    root = tmp_path / "data"
    _write(root, wide)
    from core.market_data.ohlcv_cache import OHLVCache

    hashes = []
    for _ in range(2):
        cache = OHLVCache(str(root))
        cache.get(SYMBOL, "1h")
        cache.update(SYMBOL, "1h", narrow)
        cache.save(SYMBOL, "1h")
        hashes.append(_path(root).read_bytes())
    assert hashes[0] == hashes[1], "the same inputs produced a different file"
    assert _rows(root) == len(wide)                     # union, never the window


# ══════════════════════════════════════════════════════════════════════
# (v) — the *interval-aware* union: one row per bar, whatever convention
#       wrote the stamp
# ══════════════════════════════════════════════════════════════════════
#
# The live ``data/market/BTCUSDT/1h.parquet`` holds 11 677 rows with **54**
# one-millisecond-adjacent pairs and **55** bars stored twice: 8 767 bar-open
# stamps plus 2 910 Binance ``close_time`` stamps (``open + length - 1 ms``).
# The exact-timestamp union above cannot see such a pair — its two stamps are
# 3 599.999 s apart — so the running service's flush re-expands a repaired file
# to 11 677 rows on the next write.
#
# The rule pinned here folds on the **declared bar length** (the same rule
# ``scripts/download_history.py --merge`` applies).  It must not be a proximity
# rule: those 54 one-millisecond neighbours are the close of hour ``H-1`` beside
# the open of hour ``H`` — two *different* bars — so "stamps closer than 1 s are
# the same bar" would delete 54 real hours and keep all 55 duplicates.

CLOSE_OFFSET_1H = pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)


def _frame_at(stamps, base: float = 100.0) -> pd.DataFrame:
    """One OHLCV row per stamp; the row values encode the row's provenance."""
    idx = pd.DatetimeIndex(stamps)
    step = np.arange(len(idx), dtype=float)
    return pd.DataFrame({"open": base + step, "high": base + step + 1.0,
                         "low": base + step - 1.0, "close": base + step + 0.5,
                         "volume": 1.0 + step}, index=idx)


def _one_ms_pairs(frame: pd.DataFrame) -> int:
    """How many adjacent stamp pairs sit exactly 1 ms apart (the live metric)."""
    ns = np.sort(pd.DatetimeIndex(frame.index).asi8)
    return int((np.diff(ns) == 1_000_000).sum())


def _bar_dups(frame: pd.DataFrame, interval: str = "1h") -> int:
    """Rows beyond one per bar, i.e. bars stored under both conventions."""
    from core.market_data.ohlcv_cache import bar_keys

    return int(len(frame) - bar_keys(frame.index, interval).nunique())


def test_live_shaped_re_expansion_is_stopped_by_the_interval_aware_union(tmp_path):
    """The measured live defect and its fix, reproduced deterministically.

    Fixture shape = the live cache's numbers: 11 567 single-convention bars plus a
    tail block of 55 bars stored twice → **11 677 rows / 54 one-millisecond pairs
    / 55 duplicated bars**.  ``repaired`` is the repaired file (one row per bar,
    bar-open stamps); ``stale`` is what a process that never re-read the file
    still holds — those 55 bars' old ``close_time`` stamps.
    """
    from core.market_data.ohlcv_cache import OHLVCache, bar_keys, merge_history

    head = pd.date_range("2025-06-03 00:00", periods=11_567, freq="h")
    twin = pd.date_range(head[-1] + pd.Timedelta(hours=1), periods=55, freq="h")
    repaired = pd.concat([_frame_at(head), _frame_at(twin, base=500.0)])
    stale = _frame_at(twin + CLOSE_OFFSET_1H, base=900.0)
    assert (len(repaired), len(stale)) == (11_622, 55)

    live_shaped = pd.concat([repaired, stale]).sort_index()
    assert len(live_shaped) == 11_677
    assert (_one_ms_pairs(live_shaped), _bar_dups(live_shaped)) == (54, 55)

    # BEFORE — the interval-agnostic union the store performed: the file is
    # re-expanded to the measured 11 677 rows with the 54 twin pairs back.
    before = merge_history(repaired, stale)
    assert (len(before), _one_ms_pairs(before), _bar_dups(before)) == (11_677, 54, 55)

    # AFTER — interval-aware: still one row per bar, no pair, nothing lost.
    after = merge_history(repaired, stale, "1h")
    assert (len(after), _one_ms_pairs(after), _bar_dups(after)) == (11_622, 0, 0)
    assert bar_keys(after.index, "1h").is_unique
    assert set(bar_keys(after.index, "1h").asi8) == set(
        bar_keys(repaired.index, "1h").asi8), "a bar disappeared from the union"
    # The fresher row's *values* win the conflict ...
    assert after.loc[twin[0], "close"] == stale.loc[twin[0] + CLOSE_OFFSET_1H, "close"]
    # ... and re-merging the same stale frame is a no-op (idempotent).
    again = merge_history(after, stale, "1h")
    assert len(again) == len(after) and _bar_dups(again) == 0

    # The same result through the real write path, on a temp dir.
    root = tmp_path / "data"
    _write(root, repaired)
    cache = OHLVCache(str(root))
    cache.get(SYMBOL, "1h")                       # loaded once ...
    cache.update(SYMBOL, "1h", live_shaped)       # ... then the stale frame
    cache.save(SYMBOL, "1h")
    written = pd.read_parquet(_path(root))
    assert len(written) == 11_622, "the flush re-expanded the repaired file"
    assert (_one_ms_pairs(written), _bar_dups(written)) == (0, 0)
    assert written.index.is_unique
    assert set(bar_keys(written.index, "1h").asi8) == set(
        bar_keys(repaired.index, "1h").asi8)


def test_a_genuine_second_bar_one_millisecond_away_is_never_collapsed():
    """close(H-1) and open(H) are 1 ms apart and are *different* bars.

    This is the trap a proximity-based tolerance falls into; the bar-length key
    (like ``scripts.download_history.bar_open_keys``) never shares a key between
    them, so both rows survive — for every supported fixed-length label.
    """
    from core.market_data.ohlcv_cache import bar_keys, merge_history

    cases = [("1h", pd.Timedelta(hours=1), "2026-01-01 00:00"),
             ("15m", pd.Timedelta(minutes=15), "2026-01-01 00:00"),
             ("1m", pd.Timedelta(minutes=1), "2026-01-01 00:00"),
             ("1d", pd.Timedelta(days=1), "2026-01-01 00:00")]
    for interval, length, open0 in cases:
        hour = pd.Timestamp(open0)
        prev_close = hour + length - pd.Timedelta(milliseconds=1)
        next_open = hour + length
        assert (next_open - prev_close) == pd.Timedelta(milliseconds=1)
        merged = merge_history(_frame_at([prev_close]), _frame_at([next_open], base=900.0),
                              interval)
        assert len(merged) == 2, f"{interval}: a proximity rule deleted a real bar"
        assert set(merged.index) == {prev_close, next_open}
        assert bar_keys(merged.index, interval).is_unique
        assert _one_ms_pairs(merged) == 1


def test_bar_keys_match_the_download_script_grouping_exactly():
    """Drift guard: this module's key rule == ``download_history.bar_open_keys``."""
    from core.market_data.ohlcv_cache import BAR_LENGTH_NS, bar_keys
    from scripts.download_history import INTERVAL_LENGTH_NS, bar_open_keys

    assert BAR_LENGTH_NS == INTERVAL_LENGTH_NS, "the two bar-length tables drifted"

    def _partition(keys):
        """Row → group id, independent of the keys' absolute convention."""
        rank = {v: i for i, v in enumerate(dict.fromkeys(map(int, keys.asi8)))}
        return [rank[int(v)] for v in keys.asi8]

    hours = pd.date_range("2026-06-01 00:00", periods=8, freq="h")
    frame = pd.concat([_frame_at(hours),
                       _frame_at(hours[2:5] + CLOSE_OFFSET_1H, base=900.0),
                       _frame_at([hours[7] + CLOSE_OFFSET_1H], base=700.0)])
    assert _partition(bar_keys(frame.index, "1h")) == _partition(
        bar_open_keys(frame.index, "1h"))


def test_collided_survivor_takes_the_files_dominant_convention(tmp_path):
    """The survivor's stamp follows the *file's* convention, not the writer's."""
    from core.market_data.ohlcv_cache import dominant_convention, merge_history

    hours = pd.date_range("2026-06-01 00:00", periods=5, freq="h")
    stale = _frame_at(hours + CLOSE_OFFSET_1H, base=900.0)

    open_file = _frame_at(hours)                       # 5 bar-open rows
    assert dominant_convention(open_file.index, "1h") == "open"
    merged = merge_history(open_file, stale, "1h")
    assert list(merged.index) == list(hours), "the surviving stamp left the file's grid"
    assert merged.loc[hours[0], "close"] == stale.iloc[0]["close"]

    close_file = _frame_at(hours + CLOSE_OFFSET_1H)    # 5 close_time rows
    assert dominant_convention(close_file.index, "1h") == "close"
    fresh_open = _frame_at(hours, base=900.0)
    merged2 = merge_history(close_file, fresh_open, "1h")
    assert list(merged2.index) == list(hours + CLOSE_OFFSET_1H)
    assert merged2.loc[hours[0] + CLOSE_OFFSET_1H, "close"] == fresh_open.iloc[0]["close"]


def test_legacy_two_argument_union_is_unchanged_and_the_write_still_dedupes(tmp_path):
    """``provider``'s 2-argument call keeps its result; ``save`` dedupes it."""
    from core.market_data.ohlcv_cache import OHLVCache, merge_history

    hours = pd.date_range("2026-06-01 00:00", periods=4, freq="h")
    file_frame = _frame_at(hours)
    stale = _frame_at(hours + CLOSE_OFFSET_1H, base=900.0)
    legacy = merge_history(file_frame, stale)          # no interval: as before
    assert len(legacy) == 8

    root = tmp_path / "data"
    _write(root, file_frame)
    cache = OHLVCache(str(root))
    cache.get(SYMBOL, "1h")
    cache.update(SYMBOL, "1h", legacy)                 # ... holding both conventions
    cache.save(SYMBOL, "1h")
    merged = pd.read_parquet(_path(root))
    assert len(merged) == 4 and merged.index.is_unique  # the write is aware
    assert list(merged.index) == list(hours)


def test_interval_aware_write_only_grows_the_range_and_is_byte_identical(tmp_path):
    """A stale flush neither shrinks the range nor produces a different file.

    Determinism is asserted the strong way — two independent caches started from
    the *same* on-disk snapshot with the same in-memory frame write identical
    bytes — and repeated flushes of that frame must settle (a mixed file
    converges onto one convention within a couple of writes, then stops
    changing).
    """
    from core.market_data.ohlcv_cache import OHLVCache, bar_keys

    hours = pd.date_range("2026-06-01 00:00", periods=10, freq="h")
    stored = _frame_at(hours[:6] + CLOSE_OFFSET_1H)     # 6 close_time rows
    incoming = _frame_at(hours)                         # 10 bar-open rows
    root = tmp_path / "data"
    _write(root, stored)
    snapshot = _path(root).read_bytes()

    def _flush() -> bytes:
        cache = OHLVCache(str(root))
        cache.get(SYMBOL, "1h")
        cache.update(SYMBOL, "1h", incoming)
        cache.save(SYMBOL, "1h")
        return _path(root).read_bytes()

    outs = []
    for _ in range(2):                                  # same inputs, same start
        _path(root).write_bytes(snapshot)
        outs.append(_flush())
    assert outs[0] == outs[1], "the same inputs produced a different file"

    frames = [outs[0]]
    for _ in range(3):                                  # the file must settle
        frames.append(_flush())
    assert frames[-1] == frames[-2], "repeated writes never settled"

    merged = pd.read_parquet(_path(root))
    assert len(merged) == 10, "a bar was collapsed or duplicated"
    assert merged.index.is_unique
    assert set(bar_keys(merged.index, "1h").asi8) >= set(
        bar_keys(stored.index, "1h").asi8), "the stored range shrank"
    assert set(bar_keys(merged.index, "1h").asi8) == set(
        bar_keys(incoming.index, "1h").asi8)


def test_integrity_report_counts_bars_stored_twice_without_flagging_them():
    """The auditor names the defect; ``--strict`` still means "calendar gap"."""
    from scripts.check_data_integrity import duplicate_bar_rows, gap_report

    hours = pd.date_range("2026-06-01 00:00", periods=6, freq="h")
    frame = pd.concat([_frame_at(hours),
                       _frame_at(hours[2:5] + CLOSE_OFFSET_1H, base=900.0)])
    assert len(frame) == 9
    assert duplicate_bar_rows(frame, "1h") == 3
    rep = gap_report(frame, "1h")
    assert rep["duplicate_bar_rows"] == 3 and rep["bars"] == 9
    assert rep["flagged"] is False and rep["gap_count"] == 0
    assert duplicate_bar_rows(_frame_at(hours), "1h") == 0
    assert duplicate_bar_rows(frame, "7h") == 0        # no grid → no claim


def test_monthly_label_floors_and_an_unknown_label_falls_back_to_exact_union():
    """No fixed length (``1M``) → calendar floor; no grid at all → exact union."""
    from core.market_data.ohlcv_cache import bar_keys, merge_history

    jan_open = pd.Timestamp("2026-01-01 00:00")
    jan_close = pd.Timestamp("2026-01-31 23:59:59.999")
    merged = merge_history(_frame_at([jan_open]), _frame_at([jan_close], base=900.0), "1M")
    assert len(merged) == 1
    assert bar_keys(merged.index, "1M").is_unique

    a = _frame_at([pd.Timestamp("2026-01-01 00:00")])
    b = _frame_at([pd.Timestamp("2026-01-01 00:59:59.999")], base=900.0)
    assert bar_keys(a.index, "7h") is None
    fallback = merge_history(a, b, "7h")
    assert len(fallback) == 2, "an unknown label must keep the exact-timestamp union"
