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
