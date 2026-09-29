"""Unit tests for trading cost model."""
import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


class FakeConfig:
    backtest_cost_enabled = True
    backtest_taker_fee_pct = 0.04
    backtest_spread_pct = {"BTCUSDT": 0.01, "ETHUSDT": 0.02, "SOLUSDT": 0.03}


class LiveConfig:
    """Config double that looks like the real one: live derivation enabled."""

    backtest_cost_enabled = True
    backtest_taker_fee_pct = 0.04
    backtest_spread_pct = {"BTCUSDT": 0.01}
    backtest_default_spread_pct = 0.05
    backtest_live_spread_enabled = True
    backtest_live_spread_ttl = 300.0
    backtest_live_spread_timeout = 0.5
    market_data_host = "https://data.example"


@pytest.fixture(autouse=True)
def _clean_live_cache():
    from core.backtest import cost_model as cm

    cm.clear_live_spread_cache()
    yield
    cm.clear_live_spread_cache()


def test_apply_trading_costs_btc():
    """BTC trade: 0.04% fee + 0.01% spread."""
    from core.backtest.cost_model import apply_trading_costs

    cost = apply_trading_costs(entry_price=50000, exit_price=51000, qty=0.01,
                               symbol="BTCUSDT", config=FakeConfig)
    # Entry notional: $500, Exit notional: $510
    # Fee: ($500 + $510) * 0.0004 = $0.404
    # Spread: ($500 + $510) * 0.00005 = $0.0505
    # Total: ~$0.4545
    assert 0.40 < cost < 0.50, f"Expected ~0.45, got {cost}"


def test_apply_trading_costs_disabled():
    """Disabled cost model should return zero."""
    from core.backtest.cost_model import apply_trading_costs

    cfg = FakeConfig()
    cfg.backtest_cost_enabled = False
    cost = apply_trading_costs(50000, 51000, 0.01, "BTCUSDT", cfg)
    assert cost == 0.0


def test_apply_trading_costs_higher_spread():
    """Altcoin with higher spread costs more."""
    from core.backtest.cost_model import apply_trading_costs

    btc_cost = apply_trading_costs(100, 101, 1, "BTCUSDT", FakeConfig)
    sol_cost = apply_trading_costs(100, 101, 1, "SOLUSDT", FakeConfig)
    assert sol_cost > btc_cost  # SOL 0.03% spread > BTC 0.01%


def test_apply_trading_costs_unknown_symbol():
    """Unknown symbol defaults to 0.03% spread."""
    from core.backtest.cost_model import apply_trading_costs

    cost = apply_trading_costs(100, 101, 1, "UNKNOWN", FakeConfig)
    assert cost > 0  # Uses default spread


# ======================================================================
# depth → spread derivation
# ======================================================================
def test_spread_pct_from_depth_is_a_percentage():
    from core.backtest.cost_model import spread_pct_from_depth

    payload = {"bids": [["100.00", "1"], ["99.5", "2"]],
               "asks": [["100.02", "1"], ["100.5", "2"]]}
    # (100.02 - 100.00) / 100.01 * 100
    assert spread_pct_from_depth(payload) == pytest.approx(
        0.02 / 100.01 * 100, rel=1e-9)


@pytest.mark.parametrize("payload", [
    None, {}, {"bids": [], "asks": [["1", "1"]]}, {"bids": [["1", "1"]], "asks": []},
    {"bids": [["abc", "1"]], "asks": [["1", "1"]]},
    {"bids": [["2", "1"]], "asks": [["1", "1"]]},          # crossed book
    {"bids": [["0", "1"]], "asks": [["0", "1"]]},          # zero mid
    {"bids": [[]], "asks": [["1", "1"]]},
])
def test_spread_pct_from_depth_bad_payloads(payload):
    from core.backtest.cost_model import spread_pct_from_depth

    assert spread_pct_from_depth(payload) is None


# ======================================================================
# resolution order: override → live → default
# ======================================================================
def test_resolution_override_beats_live(monkeypatch):
    from core.backtest import cost_model as cm

    calls = []

    def _boom(*args, **kwargs):
        calls.append(args)
        return 9.99

    monkeypatch.setattr(cm, "fetch_live_spread_pct", _boom)
    pct, source = cm.resolve_spread_pct(
        "ADAUSDT", LiveConfig, overrides={"ADAUSDT": 0.007})
    assert (pct, source) == (0.007, cm.SOURCE_OVERRIDE)
    assert calls == []  # an override never touches the network

    # Config map entries are overrides too.
    assert cm.resolve_spread_pct("BTCUSDT", LiveConfig)[1] == cm.SOURCE_OVERRIDE


def test_resolution_live_when_no_override(monkeypatch):
    from core.backtest import cost_model as cm

    calls = []
    monkeypatch.setattr(cm, "fetch_live_spread_pct",
                        lambda symbol, host, timeout: calls.append(symbol) or 0.0123)

    pct, source = cm.resolve_spread_pct("ADAUSDT", LiveConfig)
    assert (pct, source) == (0.0123, cm.SOURCE_LIVE)
    assert calls == ["ADAUSDT"]


def test_resolution_live_is_cached_for_the_ttl(monkeypatch):
    from core.backtest import cost_model as cm

    calls = []
    monkeypatch.setattr(cm, "fetch_live_spread_pct",
                        lambda symbol, host, timeout: calls.append(symbol) or 0.02)

    assert cm.resolve_spread_pct("ADAUSDT", LiveConfig, now=1000.0) == (0.02, "live")
    # within the 300s window → cache hit, no second fetch
    assert cm.resolve_spread_pct("ADAUSDT", LiveConfig, now=1200.0) == (0.02, "live")
    assert len(calls) == 1
    # past the window → refetch
    assert cm.resolve_spread_pct("ADAUSDT", LiveConfig, now=1400.0) == (0.02, "live")
    assert len(calls) == 2
    assert cm.peek_live_spread("ADAUSDT", LiveConfig, now=1500.0) == 0.02


def test_resolution_falls_back_to_default_when_live_fails(monkeypatch):
    from core.backtest import cost_model as cm

    monkeypatch.setattr(cm, "fetch_live_spread_pct", lambda *a, **k: None)
    assert cm.resolve_spread_pct("NOPEUSDT", LiveConfig) == (0.05, "default")
    # A config double without the live flag never even tries (test/offline path).
    monkeypatch.setattr(cm, "fetch_live_spread_pct",
                        lambda *a, **k: pytest.fail("live must be disabled here"))
    assert cm.resolve_spread_pct("NOPEUSDT", FakeConfig) == (0.03, "default")


def test_failed_live_lookup_is_not_cached(monkeypatch):
    from core.backtest import cost_model as cm

    calls = []

    def _flaky(*args, **kwargs):
        calls.append(args)
        return None if len(calls) == 1 else 0.015

    monkeypatch.setattr(cm, "fetch_live_spread_pct", _flaky)
    assert cm.resolve_spread_pct("ADAUSDT", LiveConfig)[1] == "default"
    assert cm.resolve_spread_pct("ADAUSDT", LiveConfig) == (0.015, "live")
    assert len(calls) == 2  # the failure was not remembered


def test_default_key_in_override_map_covers_every_symbol():
    from core.backtest import cost_model as cm

    cfg = LiveConfig()
    cfg.backtest_spread_pct = {"BTCUSDT": 0.01, "default": 0.02}
    assert cm.resolve_spread_pct("BTCUSDT", cfg) == (0.01, "override")
    assert cm.resolve_spread_pct("WHATEVERUSDT", cfg) == (0.02, "override")


def test_per_symbol_fallback_without_live():
    from core.backtest import cost_model as cm

    table = cm.resolve_spreads(["BTCUSDT", "ADAUSDT"], FakeConfig)
    assert table["BTCUSDT"] == {"symbol": "BTCUSDT", "spread_pct": 0.01,
                                "source": "override"}
    assert table["ADAUSDT"] == {"symbol": "ADAUSDT", "spread_pct": 0.03,
                                "source": "default"}


def test_resolve_spreads_mixes_sources(monkeypatch):
    from core.backtest import cost_model as cm

    monkeypatch.setattr(cm, "fetch_live_spread_pct",
                        lambda symbol, host, timeout: {"ADAUSDT": 0.021}.get(symbol))
    table = cm.resolve_spreads(["BTCUSDT", "ADAUSDT", "NOPEUSDT"], LiveConfig)
    assert [(s, table[s]["source"]) for s in table] == [
        ("BTCUSDT", "override"), ("ADAUSDT", "live"), ("NOPEUSDT", "default")]
    assert table["ADAUSDT"]["spread_pct"] == 0.021


def test_freeze_run_spreads_is_a_plain_map(monkeypatch):
    from core.backtest import cost_model as cm

    monkeypatch.setattr(cm, "fetch_live_spread_pct", lambda *a, **k: 0.021)
    frozen = cm.freeze_run_spreads(["BTCUSDT", "ADAUSDT"], LiveConfig)
    assert frozen == {"BTCUSDT": 0.01, "ADAUSDT": 0.021}


# ======================================================================
# the cost formula consumes the resolved spread (unchanged maths)
# ======================================================================
def test_apply_trading_costs_uses_live_spread(monkeypatch):
    from core.backtest import cost_model as cm

    monkeypatch.setattr(cm, "fetch_live_spread_pct", lambda *a, **k: 0.05)
    cost = cm.apply_trading_costs(100, 101, 1, "ADAUSDT", LiveConfig)
    fee = (100 + 101) * 0.0004
    spread = (100 + 101) * (0.05 / 100 / 2.0)
    assert cost == pytest.approx(fee + spread)


def test_apply_trading_costs_honours_run_overrides(monkeypatch):
    from core.backtest import cost_model as cm

    monkeypatch.setattr(cm, "fetch_live_spread_pct",
                        lambda *a, **k: pytest.fail("no I/O once the run is frozen"))
    cost = cm.apply_trading_costs(100, 101, 1, "ADAUSDT", LiveConfig,
                                  overrides={"ADAUSDT": 0.02})
    fee = (100 + 101) * 0.0004
    spread = (100 + 101) * (0.02 / 100 / 2.0)
    assert cost == pytest.approx(fee + spread)


# ======================================================================
# GET /api/backtest/spreads — the table the backtest UI renders
# ======================================================================
@pytest.fixture(scope="module")
def spreads_client():
    from app.config import Config
    from app.event_bus import EventBus
    from core.auth.auth import AuthManager
    from db.database import init_database

    tmpdir = Path(tempfile.mkdtemp(prefix="bt_spreads_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "spreads.db")
    config.data_dir = str(tmpdir / "data")
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")

    async def _setup():
        await init_database(config.db_path)
        manager = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        await manager.create_user("bt_spreads", "T1aderPass!", "trader", "bt_spreads")
        return manager

    auth = asyncio.run(_setup())

    from web.server import create_app
    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    client = TestClient(app)
    login = client.post("/api/auth/login",
                        json={"username": "bt_spreads", "password": "T1aderPass!"})
    assert login.status_code == 200, login.text
    yield client, config


def test_spreads_endpoint_reports_source_per_symbol(spreads_client, monkeypatch):
    from core.backtest import cost_model as cm

    client, _config = spreads_client
    monkeypatch.setattr(cm, "fetch_live_spread_pct",
                        lambda symbol, host, timeout: {"ADAUSDT": 0.021}.get(symbol))
    body = client.get(
        "/api/backtest/spreads",
        params={"symbols": "btcusdt,ADAUSDT,NOPEUSDT", "overrides": '{"NOPEUSDT": 0.09}'},
    ).json()
    assert [(r["symbol"], r["source"]) for r in body["symbols"]] == [
        ("BTCUSDT", "override"), ("ADAUSDT", "live"), ("NOPEUSDT", "override")]
    assert [r["spread_pct"] for r in body["symbols"]] == [0.01, 0.021, 0.09]
    assert body["default_spread_pct"] == 0.03
    assert body["live_spread_enabled"] is True


def test_spreads_endpoint_empty_selection_is_not_an_error(spreads_client):
    client, _config = spreads_client
    body = client.get("/api/backtest/spreads").json()
    assert body["symbols"] == []
