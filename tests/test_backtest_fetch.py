"""Tests for ``POST /api/backtest/fetch-data`` and the free-symbol backtest page.

Covers: trader-only auth, the symbol/interval caps, the exact downloader
subprocess contract (frozen CLI, ``sys.executable``, repo-root cwd, no
``shell=True``), timeout / spawn-failure mapping and per-symbol error mapping.

Everything runs against a temporary SQLite database and a temporary data dir —
``data/market`` and ``scripts/download_history.py`` are never touched (the
subprocess is always mocked).
"""
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from core.strategy.loader import StrategyConfig, StrategyLoader
from db.database import init_database
from web.routes import backtest as routes_backtest

VIEWER = ("btf_viewer", "V1ewerPass!")
TRADER = ("btf_trader", "T1aderPass!")


def _write_parquet(path: Path, rows: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "open": [1.0] * rows, "high": [2.0] * rows, "low": [0.5] * rows,
        "close": [1.5] * rows, "volume": [10.0] * rows,
    }, index=pd.date_range("2026-01-01", periods=rows, freq="1h", name="close_time"))
    df.to_parquet(path)
    return path


class FakeProc:
    def __init__(self, returncode=0):
        self.returncode = returncode


@pytest.fixture(scope="module")
def web_app():
    tmpdir = Path(tempfile.mkdtemp(prefix="bt_fetch_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "fetch.db")
    config.data_dir = str(tmpdir / "data")
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")

    async def _setup():
        await init_database(config.db_path)
        am = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        for username, password, role in ((VIEWER[0], VIEWER[1], "viewer"),
                                        (TRADER[0], TRADER[1], "trader")):
            await am.create_user(username, password, role, username)
        return am

    auth = asyncio.run(_setup())

    from web.server import create_app
    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    # A real loader over a temp strategies dir (routes read it for the fallback
    # symbol universe / matrix rows).
    loader = StrategyLoader(str(tmpdir / "strategies"))
    loader.save(StrategyConfig(
        name="btf_demo", timeframes=["1h"], symbols=["SOLUSDT"],
        entry_conditions={"long": ["rsi < 30"]}, exit_conditions={"long": ["rsi > 70"]},
        reduce_conditions={"long": []}))
    app.state.strategy_loader = loader
    yield app


@pytest.fixture()
def trader_client(web_app):
    c = TestClient(web_app)
    r = c.post("/api/auth/login", json={"username": TRADER[0], "password": TRADER[1]})
    assert r.status_code == 200, r.text
    return c


@pytest.fixture()
def viewer_client(web_app):
    c = TestClient(web_app)
    r = c.post("/api/auth/login", json={"username": VIEWER[0], "password": VIEWER[1]})
    assert r.status_code == 200, r.text
    return c


@pytest.fixture()
def market_dir(web_app):
    d = Path(web_app.state.config.data_dir) / "market"
    return d


@pytest.fixture()
def fake_download(monkeypatch):
    """Mock ``subprocess.run``; ``produce`` decides which parquet files appear."""
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        producer = _fake_run.produce
        if isinstance(producer, Exception):
            raise producer
        if callable(producer):
            producer(cmd, kwargs)
        sink = kwargs.get("stdout")
        if hasattr(sink, "write"):
            sink.write("downloading...\ndone\n")
        return FakeProc(getattr(_fake_run, "returncode", 0))

    _fake_run.produce = None
    _fake_run.calls = calls
    monkeypatch.setattr(routes_backtest.subprocess, "run", _fake_run)
    return _fake_run


FORM = {"symbols": "SOLUSDT", "intervals": "1h",
        "date_start": "2026-01-01", "date_end": "2026-02-01"}


# ======================================================================
# auth
# ======================================================================
def test_fetch_data_requires_trader(web_app, viewer_client):
    anon = TestClient(web_app)
    assert anon.post("/api/backtest/fetch-data", data=FORM).status_code == 401
    denied = viewer_client.post("/api/backtest/fetch-data", data=FORM)
    assert denied.status_code == 403
    assert denied.json() == {"error": "Forbidden"}


# ======================================================================
# caps + validation (rejected before any subprocess is started)
# ======================================================================
def test_rejects_more_than_five_symbols(trader_client, fake_download):
    form = dict(FORM, symbols="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT")
    r = trader_client.post("/api/backtest/fetch-data", data=form)
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    assert "5" in body["error"] and "6" in body["error"]
    assert fake_download.calls == [], "the downloader must not run above the cap"


def test_rejects_more_than_two_intervals(trader_client, fake_download):
    r = trader_client.post("/api/backtest/fetch-data",
                           data=dict(FORM, intervals="1h,4h,15m"))
    assert r.status_code == 400
    assert r.json()["ok"] is False
    assert "2" in r.json()["error"]
    assert fake_download.calls == []


def test_rejects_no_symbols(trader_client, fake_download):
    r = trader_client.post("/api/backtest/fetch-data", data=dict(FORM, symbols=" , "))
    assert r.status_code == 400
    assert "至少选择一个交易对" in r.json()["error"]
    assert fake_download.calls == []


def test_intervals_default_to_1h(trader_client, fake_download, market_dir):
    fake_download.produce = lambda cmd, kwargs: _write_parquet(
        market_dir / "BNBUSDT" / "1h.parquet")
    form = {k: v for k, v in FORM.items() if k != "intervals"}
    r = trader_client.post("/api/backtest/fetch-data", data=dict(form, symbols="BNBUSDT"))
    assert r.status_code == 200
    assert fake_download.calls[0]["cmd"][5] == "1h"
    assert r.json()["results"][0]["interval"] == "1h"


@pytest.mark.parametrize("field,value,needle", [
    ("symbols", "BTC-USDT", "交易对格式无效"),
    ("symbols", "'; drop table", "交易对格式无效"),
    ("intervals", "7h", "不支持的周期"),
    ("intervals", "1h;rm -rf", "不支持的周期"),
    ("date_start", "2026-13-01", "起始日期格式无效"),
    ("date_end", "01/02/2026", "结束日期格式无效"),
])
def test_rejects_malformed_input(trader_client, fake_download, field, value, needle):
    r = trader_client.post("/api/backtest/fetch-data", data=dict(FORM, **{field: value}))
    assert r.status_code == 400
    assert needle in r.json()["error"]
    assert fake_download.calls == []


def test_rejects_inverted_date_range(trader_client, fake_download):
    r = trader_client.post("/api/backtest/fetch-data",
                           data=dict(FORM, date_start="2026-03-01", date_end="2026-01-01"))
    assert r.status_code == 400
    assert "起始日期" in r.json()["error"]
    assert fake_download.calls == []


# ======================================================================
# subprocess contract
# ======================================================================
def test_subprocess_invocation_contract(trader_client, fake_download, market_dir):
    fake_download.produce = lambda cmd, kwargs: _write_parquet(
        market_dir / "SOLUSDT" / "1h.parquet")
    r = trader_client.post("/api/backtest/fetch-data",
                           data={"symbols": "solusdt,ethusdt", "intervals": "1h,4h",
                                 "date_start": "2026-01-01", "date_end": "2026-02-01"})
    assert r.status_code == 200
    assert len(fake_download.calls) == 1
    call = fake_download.calls[0]
    repo_root = Path(__file__).resolve().parents[1]
    # Frozen CLI, exact argv, symbols upper-cased and de-duplicated.
    assert call["cmd"] == [
        sys.executable, str(repo_root / "scripts" / "download_history.py"),
        "--symbols", "SOLUSDT,ETHUSDT",
        "--intervals", "1h,4h",
        "--start", "2026-01-01",
        "--end", "2026-02-01",
        # --data-dir is the ROOT data dir; the CLI appends market/<symbol>/...
        "--data-dir", str(market_dir.parent),
        # --merge: a narrow fetch must union with (never replace) a longer cache.
        "--merge",
    ]
    assert call["kwargs"]["cwd"] == str(repo_root)
    # `shell` is never passed, i.e. it keeps its default of False.
    assert "shell" not in call["kwargs"]
    assert call["kwargs"]["timeout"] == routes_backtest.FETCH_TIMEOUT_SECONDS == 300
    # stdout must be a file object, never a pipe (no reader thread / decoding issues)
    assert hasattr(call["kwargs"]["stdout"], "write")


def test_success_payload_has_rows_and_path(trader_client, fake_download, market_dir):
    fake_download.produce = lambda cmd, kwargs: _write_parquet(
        market_dir / "SOLUSDT" / "1h.parquet", rows=7)
    r = trader_client.post("/api/backtest/fetch-data", data=FORM)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["errors"] == []
    assert len(body["results"]) == 1
    entry = body["results"][0]
    assert entry["symbol"] == "SOLUSDT" and entry["interval"] == "1h"
    assert entry["ok"] is True and entry["rows"] == 7
    assert entry["path"].endswith("SOLUSDT/1h.parquet")
    assert "downloading" in body["log_tail"]


def test_per_symbol_error_mapping(trader_client, fake_download, market_dir):
    # Only the first symbol is produced by the (mocked) downloader.
    fake_download.produce = lambda cmd, kwargs: _write_parquet(
        market_dir / "SOLUSDT" / "1h.parquet")
    r = trader_client.post("/api/backtest/fetch-data",
                           data=dict(FORM, symbols="SOLUSDT,XRPUSDT"))
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    by_symbol = {e["symbol"]: e for e in body["results"]}
    assert by_symbol["SOLUSDT"]["ok"] is True
    assert by_symbol["SOLUSDT"]["rows"] > 0
    assert by_symbol["XRPUSDT"]["ok"] is False
    assert by_symbol["XRPUSDT"]["rows"] == 0
    assert by_symbol["XRPUSDT"]["path"] is None
    assert "XRPUSDT" in body["error"]
    assert [e["symbol"] for e in body["errors"]] == ["XRPUSDT"]


def test_nonzero_exit_is_reported_per_symbol(trader_client, fake_download):
    fake_download.returncode = 1
    r = trader_client.post("/api/backtest/fetch-data",
                           data=dict(FORM, symbols="DOGEUSDT"))
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "退出码 1" in body["results"][0]["error"]


def test_timeout_mapping(trader_client, monkeypatch):
    def _boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, routes_backtest.FETCH_TIMEOUT_SECONDS)

    monkeypatch.setattr(routes_backtest.subprocess, "run", _boom)
    r = trader_client.post("/api/backtest/fetch-data", data=FORM)
    assert r.status_code == 504
    assert r.json()["ok"] is False
    assert "超时" in r.json()["error"] and "300" in r.json()["error"]


def test_spawn_failure_returns_502(trader_client, fake_download):
    fake_download.produce = OSError("cannot execute")
    r = trader_client.post("/api/backtest/fetch-data", data=FORM)
    assert r.status_code == 502
    assert "启动下载进程失败" in r.json()["error"]


def test_missing_download_script_returns_502(trader_client, tmp_path, monkeypatch):
    monkeypatch.setattr(routes_backtest, "_repo_root", lambda: tmp_path)
    r = trader_client.post("/api/backtest/fetch-data", data=FORM)
    assert r.status_code == 502
    assert "下载脚本不存在" in r.json()["error"]


# ======================================================================
# fallback universe helpers
# ======================================================================
def test_cached_intervals_and_known_symbols(web_app, market_dir):
    _write_parquet(market_dir / "ADAUSDT" / "4h.parquet")
    cached = routes_backtest._cached_intervals(web_app.state.config)
    assert cached["ADAUSDT"] == ["4h"]
    known = routes_backtest._known_symbols(web_app.state.config,
                                          web_app.state.strategy_loader)
    # cached pair + strategy pair + the built-in five
    for sym in ("ADAUSDT", "SOLUSDT", "BTCUSDT", "XRPUSDT"):
        assert sym in known
    assert known.index("BTCUSDT") < known.index("ADAUSDT")


# ======================================================================
# page rendering (requirement 3)
# ======================================================================
def test_backtest_page_contains_symbol_selector(trader_client):
    r = trader_client.get("/backtest")
    assert r.status_code == 200
    html = r.text
    assert 'id="bt-symbol-selector"' in html
    assert 'id="bt-symbol-chips"' in html
    assert 'id="bt-symbol-search"' in html
    assert "/api/market/symbols" in html
    assert "下载所选数据" in html
    assert "bt-data-panel" in html


def test_backtest_config_partial_contains_symbol_selector(trader_client):
    r = trader_client.get("/partials/backtest-config")
    assert r.status_code == 200
    assert 'id="bt-symbol-selector"' in r.text


def test_strategies_page_contains_symbol_selector(trader_client):
    r = trader_client.get("/strategies")
    assert r.status_code == 200
    html = r.text
    assert 'id="sg-symbol-selector-host"' in html
    assert 'id="sg-symbol-selector"' in html
    assert 'id="sg-symbol-chips"' in html
    assert "/api/market/symbols" in html
    assert "BTCUSDT" in html  # server-rendered fallback universe


def test_backtest_page_requires_trader(web_app):
    c = TestClient(web_app)
    r = c.get("/backtest", follow_redirects=False)
    assert r.status_code in (302, 303, 307)


def test_strategy_symbols_still_persist_through_save_path(trader_client):
    """The editor posts `symbols`; the YAML schema is unchanged."""
    payload = {"name": "btf_symbols", "mode": "trend", "timeframes": ["1h"],
               "symbols": ["ETHUSDT", "SOLUSDT"], "indicators": {},
               "entry_conditions": {"long": []}, "exit_conditions": {"long": []},
               "reduce_conditions": {"long": []}}
    r = trader_client.post("/api/strategy", json=payload)
    assert r.status_code == 200, r.text
    got = trader_client.get("/api/strategy/btf_symbols").json()
    assert got["symbols"] == ["ETHUSDT", "SOLUSDT"]
