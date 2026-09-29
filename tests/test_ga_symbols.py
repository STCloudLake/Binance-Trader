"""Tests for GA symbol selection — the GA no longer runs on a hardcoded list.

Covers the route contract (JSON body *and* comma-separated form field, universe
validation, the 20-symbol cap, watchlist default) and the worker side (symbols
read from the job payload, watchlist fallback).  Nothing here spawns a real GA
worker: ``subprocess.Popen`` is replaced and the job file is inspected instead.
"""
import asyncio
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from core.market_data.universe import DEFAULT_WATCHLIST, save_watchlist
from db.database import init_database
from web.routes import ga as routes_ga

TRADER = ("ga_symbols_trader", "T1aderPass!")
VIEWER = ("ga_symbols_viewer", "V1ewerPass!")

#: 25 valid USDT pairs — enough to exercise the ≤20 cap.
_UNIVERSE_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "ATOMUSDT", "NEARUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "INJUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT", "ORDIUSDT", "PEPEUSDT",
    "WIFUSDT",
]


class FakeSymbolInfo:
    def __init__(self, symbol, status="TRADING", quote_asset="USDT"):
        self.symbol = symbol
        self.status = status
        self.quote_asset = quote_asset
        self.base_asset = symbol[:-len(quote_asset)] if quote_asset else symbol


class FakeUniverse:
    """Minimal ``Universe`` double (``get`` / ``get_symbols`` + cached count)."""

    def __init__(self, symbols=None, extra=()):
        table = {s: FakeSymbolInfo(s) for s in (symbols or _UNIVERSE_SYMBOLS)}
        for info in extra:
            table[info.symbol] = info
        self._table = table

    def get(self, symbol):
        return self._table.get(str(symbol).upper())

    async def get_symbols(self, force: bool = False):
        return list(self._table.values())

    def get_symbols_cached_count(self):
        return len(self._table)


class FakeLoader:
    def __init__(self, root: Path):
        self.strategies_dir = str(root / "strategies")


class FakeProc:
    """Records the spawn instead of running a real GA worker."""

    calls = []

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.returncode = 0
        FakeProc.calls.append({"argv": argv, "kwargs": kwargs})

    def poll(self):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


@pytest.fixture(scope="module")
def web_app():
    tmpdir = Path(tempfile.mkdtemp(prefix="ga_symbols_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "ga.db")
    config.data_dir = str(tmpdir / "data")
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")

    async def _setup():
        await init_database(config.db_path)
        manager = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        for username, password, role in ((TRADER[0], TRADER[1], "trader"),
                                         (VIEWER[0], VIEWER[1], "viewer")):
            await manager.create_user(username, password, role, username)
        return manager

    auth = asyncio.run(_setup())

    from web.server import create_app
    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    app.state.strategy_loader = FakeLoader(tmpdir)
    app.state.backtest_engine = SimpleNamespace(name="fake-engine")
    app.state.universe = FakeUniverse()
    yield app


@pytest.fixture()
def trader_client(web_app):
    client = TestClient(web_app)
    response = client.post("/api/auth/login",
                           json={"username": TRADER[0], "password": TRADER[1]})
    assert response.status_code == 200, response.text
    FakeProc.calls = []
    return client


@pytest.fixture()
def viewer_client(web_app):
    client = TestClient(web_app)
    response = client.post("/api/auth/login",
                           json={"username": VIEWER[0], "password": VIEWER[1]})
    assert response.status_code == 200, response.text
    return client


@pytest.fixture(autouse=True)
def _fake_popen(monkeypatch):
    monkeypatch.setattr(routes_ga.subprocess, "Popen", FakeProc)
    return FakeProc


@pytest.fixture(autouse=True)
def _reset_states():
    routes_ga._ga_state.update({"running": False, "error": None, "job_file": None,
                                "params": None, "stopped": False})
    routes_ga._wf_state.update({"running": False, "error": None, "job_file": None,
                                "params": None, "report": None, "phase": "idle"})
    yield


def _job(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ======================================================================
# auth
# ======================================================================
def test_evolve_requires_trader(web_app, viewer_client):
    anon = TestClient(web_app)
    denied = anon.post("/api/ga/evolve", json={"symbols": ["ADAUSDT"]})
    assert denied.status_code == 401
    assert viewer_client.post("/api/ga/evolve",
                              json={"symbols": ["ADAUSDT"]}).status_code == 403


# ======================================================================
# the chosen symbols reach the worker's job file
# ======================================================================
def test_evolve_writes_the_chosen_symbols_to_the_job_file(trader_client):
    chosen = ["ADAUSDT", "XRPUSDT", "DOGEUSDT"]
    response = trader_client.post("/api/ga/evolve", json={
        "symbols": chosen, "population_size": 4, "generations": 2})
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True

    job_file = routes_ga._ga_state["job_file"]
    job = _job(job_file)
    assert job["symbols"] == chosen
    assert job["population_size"] == 4 and job["generations"] == 2
    # The worker was spawned with the job file that carries those symbols.
    argv = FakeProc.calls[0]["argv"]
    assert argv[0]  # sys.executable
    assert argv[1].endswith("ga_worker.py")
    assert argv[2:4] == ["--job-type", "ga"]
    assert argv[4:6] == ["--job-file", job_file]
    assert routes_ga._ga_state["params"]["symbols"] == chosen


def test_evolve_accepts_a_comma_separated_form_field(trader_client):
    response = trader_client.post("/api/ga/evolve", data={
        "symbols": " adausdt , XRPUSDT,xrpusdt ,, ",
        "population_size": "4", "generations": "2"})
    assert response.status_code == 200, response.text
    job = _job(routes_ga._ga_state["job_file"])
    assert job["symbols"] == ["ADAUSDT", "XRPUSDT"]


def test_walkforward_uses_the_chosen_symbols(trader_client):
    response = trader_client.post("/api/ga/walkforward", data={
        "symbols": "ADAUSDT,DOGEUSDT,LINKUSDT",
        "population_size": "4", "generations": "2"})
    assert response.status_code == 200, response.text
    wf_job = _job(routes_ga._wf_state["job_file"])
    assert wf_job["symbols"] == ["ADAUSDT", "DOGEUSDT", "LINKUSDT"]
    assert routes_ga._wf_state["params"]["symbols"] == wf_job["symbols"]


# ======================================================================
# validation against the universe
# ======================================================================
def test_unknown_symbol_is_rejected(trader_client):
    response = trader_client.post("/api/ga/evolve", json={
        "symbols": ["ADAUSDT", "FAKECOINUSDT"]})
    assert response.status_code == 400
    error = response.json()["error"]
    assert "FAKECOINUSDT" in error and "unknown" in error
    # Rejected before anything was written or spawned.
    assert routes_ga._ga_state["job_file"] is None
    assert FakeProc.calls == []


def test_non_trading_symbol_is_rejected(trader_client, web_app):
    web_app.state.universe = FakeUniverse(
        extra=[FakeSymbolInfo("HALTEDUSDT", status="BREAK")])
    try:
        response = trader_client.post("/api/ga/evolve", json={
            "symbols": ["HALTEDUSDT"]})
        assert response.status_code == 400
        assert "BREAK" in response.json()["error"]
    finally:
        web_app.state.universe = FakeUniverse()


def test_non_usdt_quote_symbol_is_rejected(trader_client, web_app):
    web_app.state.universe = FakeUniverse(
        extra=[FakeSymbolInfo("ETHBTC", quote_asset="BTC")])
    try:
        response = trader_client.post("/api/ga/evolve", json={
            "symbols": ["ETHBTC"]})
        assert response.status_code == 400
        assert "quote asset" in response.json()["error"]
    finally:
        web_app.state.universe = FakeUniverse()


def test_symbol_cap_is_enforced(trader_client):
    too_many = _UNIVERSE_SYMBOLS[:21]
    response = trader_client.post("/api/ga/evolve", json={"symbols": too_many})
    assert response.status_code == 400
    error = response.json()["error"]
    assert "21" in error and str(routes_ga.MAX_GA_SYMBOLS) in error

    at_cap = _UNIVERSE_SYMBOLS[:routes_ga.MAX_GA_SYMBOLS]
    response = trader_client.post("/api/ga/evolve", json={
        "symbols": at_cap, "population_size": 4, "generations": 2})
    assert response.status_code == 200, response.text
    assert _job(routes_ga._ga_state["job_file"])["symbols"] == at_cap


# ======================================================================
# default = the persisted watchlist (GET /api/market/watchlist)
# ======================================================================
def test_missing_symbols_default_to_the_watchlist(trader_client, web_app):
    saved = trader_client.post("/api/market/watchlist",
                               data={"symbols": "ADAUSDT,DOGEUSDT"})
    assert saved.status_code == 200, saved.text

    response = trader_client.post("/api/ga/evolve", json={
        "population_size": 4, "generations": 2})
    assert response.status_code == 200, response.text
    assert _job(routes_ga._ga_state["job_file"])["symbols"] == ["ADAUSDT", "DOGEUSDT"]

    # unknown symbols are still rejected when the list is explicit
    response = trader_client.post("/api/ga/evolve", json={"symbols": ["NOPEUSDT"]})
    assert response.status_code == 400
    assert web_app.state.universe.get("ADAUSDT") is not None


def test_watchlist_default_helper_reads_the_db(web_app):
    config = web_app.state.config
    asyncio.run(save_watchlist(config.db_path, ["LINKUSDT", "SUIUSDT"]))
    assert asyncio.run(routes_ga.default_ga_symbols(config)) == ["LINKUSDT", "SUIUSDT"]
    asyncio.run(save_watchlist(config.db_path, []))  # restore the shipped default
    assert asyncio.run(routes_ga.default_ga_symbols(config)) == list(DEFAULT_WATCHLIST)


def test_universe_outage_does_not_block_an_explicit_selection(trader_client, web_app):
    class OfflineUniverse(FakeUniverse):
        def get_symbols(self):
            raise RuntimeError("data host unreachable")

    web_app.state.universe = OfflineUniverse()
    try:
        response = trader_client.post("/api/ga/evolve", json={
            "symbols": ["ADAUSDT"], "population_size": 4, "generations": 2})
        assert response.status_code == 200, response.text
        assert _job(routes_ga._ga_state["job_file"])["symbols"] == ["ADAUSDT"]
    finally:
        web_app.state.universe = FakeUniverse()


# ======================================================================
# parsing helper
# ======================================================================
@pytest.mark.parametrize("raw,expected", [
    ("ada, xrp", ["ADA", "XRP"]),
    (["ADAUSDT", "adausdt"], ["ADAUSDT"]),
    (["ADAUSDT,XRPUSDT"], ["ADAUSDT", "XRPUSDT"]),
    (None, []),
    (123, []),
    (" ", []),
])
def test_parse_symbols(raw, expected):
    assert routes_ga._parse_symbols(raw) == expected


def test_panel_partial_exposes_the_picker_defaults(trader_client):
    html = trader_client.get("/partials/ga-panel").text
    assert 'id="ga-symbol-selector"' in html
    assert "GA_MAX_SYMBOLS = %d" % routes_ga.MAX_GA_SYMBOLS in html
    assert "GA_DEFAULT_SYMBOLS" in html
    assert "/api/market/symbols" in html


# ======================================================================
# worker: symbols come from the job payload, watchlist is the fallback
# ======================================================================
def _worker_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ga_worker.py"
    spec = importlib.util.spec_from_file_location("ga_worker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_reads_symbols_from_the_job_payload():
    worker = _worker_module()
    assert worker.job_symbols({"symbols": ["adausdt", "XRPUSDT", "ADAUSDT"]}) == \
        ["ADAUSDT", "XRPUSDT"]
    assert worker.job_symbols({"symbols": "ADAUSDT, XRPUSDT"}) == ["ADAUSDT", "XRPUSDT"]
    # A payload that carries symbols never consults the watchlist.
    assert worker.job_symbols({"symbols": ["ADAUSDT"]},
                              SimpleNamespace(db_path="does-not-exist.db")) == ["ADAUSDT"]


def test_worker_falls_back_to_the_watchlist(tmp_path):
    worker = _worker_module()
    db_path = str(tmp_path / "worker.db")

    async def _prepare():
        await init_database(db_path)
        await save_watchlist(db_path, ["LINKUSDT", "SUIUSDT"])

    asyncio.run(_prepare())
    config = SimpleNamespace(db_path=db_path)
    assert worker.job_symbols({}, config) == ["LINKUSDT", "SUIUSDT"]
    assert worker.job_symbols({"symbols": []}, config) == ["LINKUSDT", "SUIUSDT"]


def test_worker_fallback_without_a_usable_db(monkeypatch):
    worker = _worker_module()
    assert worker.job_symbols({}) == worker.DEFAULT_FALLBACK_SYMBOLS
    # An unreadable watchlist (bad DB) still resolves to the shipped pairs.
    monkeypatch.setattr(worker, "_watchlist_fallback", lambda config: [])
    assert worker.job_symbols({}, SimpleNamespace(db_path="unused.db")) == \
        worker.DEFAULT_FALLBACK_SYMBOLS
