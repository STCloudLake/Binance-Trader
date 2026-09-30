"""Item 1: the UI "fetch data" route must download with ``--merge``.

Why this file exists
--------------------
``POST /api/backtest/fetch-data`` runs ``scripts/download_history.py`` as a
subprocess.  It used to invoke the CLI **without** ``--merge``, and the CLI's
default is "replace the file".  So a narrow re-download (e.g. the last month)
over a long cached parquet replaced it with the narrow window: the merge-on-write
rule in ``core.market_data.ohlcv_cache`` protects the *runtime* flush path, not
this subprocess, so the UI path could still truncate a cache an operator (or
``--merge``) had just widened.

What is pinned
--------------
1. the argv the route builds carries ``--merge``, and the frozen CLI parser
   accepts that argv with ``merge=True`` (name and semantics verified against
   ``scripts/download_history.py``: ``--merge`` is a ``store_true`` flag that
   makes :func:`download_interval` union with the existing parquet one row per
   bar instead of overwriting it);
2. the route's response shape is unchanged (success and per-symbol error);
3. a narrow fetch over a longer temp-dir parquet keeps the longer range with
   ``--merge`` and truncates it without — on ``tmp_path``, with a stubbed client,
   no network and no read/write of the shipped ``data/`` tree.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from db.database import init_database
from web.routes import backtest as routes_backtest

TRADER = ("btfm_trader", "T1aderPass!")
SYMBOL = "BTCUSDT"
FORM = {"symbols": SYMBOL, "intervals": "1h",
        "date_start": "2026-01-01", "date_end": "2026-02-01"}
RESULT_KEYS = {"symbol", "interval", "ok", "rows", "path", "downloaded_at"}
FAILED_KEYS = {"symbol", "interval", "ok", "rows", "path", "error"}
PAYLOAD_KEYS = {"ok", "results", "errors", "symbols", "intervals", "log_tail"}


class FakeProc:
    def __init__(self, returncode=0):
        self.returncode = returncode


@pytest.fixture(scope="module")
def web_app():
    """The real app over a temp DB / data dir (auth included)."""
    tmpdir = Path(tempfile.mkdtemp(prefix="bt_fetch_merge_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "fetch_merge.db")
    config.data_dir = str(tmpdir / "data")
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")

    async def _setup():
        await init_database(config.db_path)
        am = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        await am.create_user(TRADER[0], TRADER[1], "trader", TRADER[0])
        return am

    auth = asyncio.run(_setup())
    from web.server import create_app

    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    yield app
    Config._instance = None


@pytest.fixture()
def trader_client(web_app):
    c = TestClient(web_app)
    r = c.post("/api/auth/login", json={"username": TRADER[0], "password": TRADER[1]})
    assert r.status_code == 200, r.text
    return c


@pytest.fixture()
def fake_download(monkeypatch):
    """Stub ``subprocess.run``; ``produce`` decides which parquet appears."""
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append({"cmd": list(cmd), "kwargs": kwargs})
        producer = _fake_run.produce
        if callable(producer):
            producer(cmd, kwargs)
        sink = kwargs.get("stdout")
        if hasattr(sink, "write"):
            sink.write("downloading...\ndone\n")
        return FakeProc()

    _fake_run.produce = None
    _fake_run.calls = calls
    monkeypatch.setattr(routes_backtest.subprocess, "run", _fake_run)
    return _fake_run


def _write_parquet(path: Path, rows: int = 3, stamp: str = "2026-01-01") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(stamp, periods=rows, freq="h", name="close_time")
    pd.DataFrame({"open": [1.0] * rows, "high": [2.0] * rows, "low": [0.5] * rows,
                  "close": [1.5] * rows, "volume": [10.0] * rows},
                 index=idx).to_parquet(path)
    return path


# ======================================================================
# 1 — the argv contract
# ======================================================================
def test_fetch_route_passes_the_merge_flag(trader_client, fake_download, web_app):
    """The route downloads with ``--merge``, and the CLI accepts that argv."""
    from scripts.download_history import parse_args

    market_dir = Path(web_app.state.config.data_dir) / "market"
    fake_download.produce = lambda cmd, kwargs: _write_parquet(
        market_dir / SYMBOL / "1h.parquet")
    r = trader_client.post("/api/backtest/fetch-data", data=FORM)

    assert r.status_code == 200, r.text
    assert len(fake_download.calls) == 1
    cmd = fake_download.calls[0]["cmd"]
    assert "--merge" in cmd, f"the UI fetch would replace (truncate) the cache: {cmd}"
    assert cmd.count("--merge") == 1
    # The flag is the *last* argument, after the frozen positional contract.
    repo_root = Path(__file__).resolve().parents[1]
    assert cmd == [sys.executable, str(repo_root / "scripts" / "download_history.py"),
                   "--symbols", SYMBOL, "--intervals", "1h",
                   "--start", "2026-01-01", "--end", "2026-02-01",
                   "--data-dir", str(market_dir.parent), "--merge"]

    # The frozen CLI parses it: `--merge` is a store_true flag, not a value.
    args = parse_args(cmd[2:])
    assert args.merge is True
    assert (args.symbols, args.intervals) == (SYMBOL, "1h")
    assert (args.start, args.end) == ("2026-01-01", "2026-02-01")
    assert Path(args.data_dir) == market_dir.parent
    assert fake_download.calls[0]["kwargs"]["cwd"] == str(repo_root)
    assert fake_download.calls[0]["kwargs"]["timeout"] == routes_backtest.FETCH_TIMEOUT_SECONDS


# ======================================================================
# 2 — the route's response shape is unchanged
# ======================================================================
def test_response_shape_is_unchanged(trader_client, fake_download, web_app):
    """Success and per-symbol-failure payloads keep their exact keys."""
    market_dir = Path(web_app.state.config.data_dir) / "market"
    fake_download.produce = lambda cmd, kwargs: _write_parquet(
        market_dir / SYMBOL / "1h.parquet", rows=5)
    body = trader_client.post("/api/backtest/fetch-data", data=FORM).json()
    assert set(body) == PAYLOAD_KEYS
    assert body["ok"] is True and body["errors"] == []
    assert set(body["results"][0]) == RESULT_KEYS
    assert body["results"][0]["ok"] is True and body["results"][0]["rows"] == 5
    assert (body["symbols"], body["intervals"]) == ([SYMBOL], ["1h"])
    assert "downloading" in body["log_tail"]

    fake_download.produce = None                        # no file is produced
    failed = trader_client.post("/api/backtest/fetch-data",
                                data=dict(FORM, symbols="XRPUSDT")).json()
    assert set(failed) == PAYLOAD_KEYS | {"error"}
    assert failed["ok"] is False
    assert set(failed["errors"][0]) == FAILED_KEYS - {"ok", "rows", "path"}
    assert failed["errors"][0]["symbol"] == "XRPUSDT"
    assert failed["results"][0]["path"] is None


# ======================================================================
# 3 — the semantics: a narrow fetch keeps a longer cached range
# ======================================================================
def _stub_client(n: int, first_open_ms: int):
    """``klines`` for ``n`` closed 1h bars from ``first_open_ms``, no socket.

    Behaves like the REST endpoint: only bars inside ``[start_time, end_time]``
    are returned, at most ``limit`` per call (Binance's row layout).
    """

    class _Client:
        def __init__(self):
            self.calls = 0

        async def klines(self, symbol, interval, limit=1000, start_time=None,
                         end_time=None):
            self.calls += 1
            rows = []
            for i in range(n):
                open_ms = first_open_ms + i * 3_600_000
                if start_time is not None and open_ms < start_time:
                    continue
                if end_time is not None and open_ms > end_time:
                    continue
                rows.append([open_ms, 2.0, 2.0, 2.0, 2.0, 2.0,
                             open_ms + 3_599_999, 12.0, 3, 2.0, 12.0, "0"])
                if len(rows) >= limit:
                    break
            return rows

        async def close(self):
            pass

    return _Client()


@pytest.mark.parametrize("merge,expected_rows", [(True, 420), (False, 24)])
def test_narrow_fetch_keeps_a_longer_file_only_with_merge(tmp_path, merge,
                                                          expected_rows):
    """``--merge`` unions; without it the CLI replaces (this is the defect).

    The temp dir holds 400 hourly bars; the "download" is the last 4 of them
    plus 20 fresh hours, i.e. a narrow window over a longer file.  With
    ``merge=True`` the file keeps all 400 old bars and grows to 420; with
    ``merge=False`` it collapses to the 24 fetched bars — which is exactly what
    the UI route did before it passed the flag.
    """
    from scripts.download_history import download_interval
    from core.market_data.universe import MARKET_CACHE_SUBDIR

    data_dir = tmp_path / "data"
    opens = pd.date_range("2026-01-01 00:00", periods=400, freq="h")
    existing = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=opens + pd.Timedelta(milliseconds=3_599_999))
    path = data_dir / MARKET_CACHE_SUBDIR / SYMBOL / "1h.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing.to_parquet(path)

    start_ms = int(pd.Timestamp("2026-01-17 12:00").value // 10**6)
    end_ms = int((pd.Timestamp("2026-01-17 12:00")
                  + pd.Timedelta(hours=23)).value // 10**6)   # 24 bars, 4 re-downloaded
    client = _stub_client(24, start_ms)
    result = asyncio.run(download_interval(client, SYMBOL, "1h", start_ms, end_ms,
                                           merge, data_dir))

    assert result.get("error") is None, result
    assert client.calls == 1, "the stub was paged more than once"
    merged = pd.read_parquet(path)
    assert len(merged) == expected_rows, (merge, len(merged))
    assert merged.index.is_unique
    assert merged.index.is_monotonic_increasing
    if merge:
        assert merged.index[0] == existing.index[0], "the old range was truncated"
        assert merged.index[-1] == pd.Timestamp(end_ms + 3_599_999, unit="ms")
        # Every old bar survives; only the 4 re-downloaded ones take the new value.
        assert merged["close"].value_counts().to_dict() == {1.0: 396, 2.0: 24}
    else:
        assert merged.index[0] > existing.index[0], "replace mode is the defect"
