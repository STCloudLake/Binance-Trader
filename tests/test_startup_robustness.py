"""Release-audit robustness regressions: startup must fail *cleanly* and helpfully.

Every gap pinned here was a non-blocking defect found by the release audit:

1. malformed ``config/*.yaml`` produced a raw ``yaml.parser.ParserError``
   traceback (exit 2) naming neither the file nor the line;
2. ``web_port: "abc"`` reached uvicorn and died with ``getaddrinfo failed``
   *after* the 70-90 s warm-up;
3. a missing config file was silently replaced by code defaults;
4. ``SignalMatrixBuilder`` raised ``ValueError: No timestamps found in data
   feeder`` while the legacy/hybrid engines returned a (different)
   ``{"error": ...}`` dict, so the same "no candles" condition produced three
   different messages;
5. ``scripts/audit_db.py`` on a fresh database reported a phantom
   ``delta = -10000`` and exited 1, which reads like real ledger drift;
6. (cheap pre-flight) a taken port is detected before the warm-up.

Everything runs against temp directories: the repo ``config/`` and
``data/binance_trader.db`` are never written.
"""
from __future__ import annotations

import re
import socket
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ======================================================================
# helpers
# ======================================================================

@pytest.fixture(autouse=True)
def _config_singleton_isolated():
    """Never leak a temp-rooted Config singleton into another test module."""
    from app.config import Config

    yield
    Config._instance = None


@contextmanager
def captured_logs(level: str = "WARNING"):
    """Collect loguru messages at ``level`` and above."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level=level)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _write_config(tmp_path: Path, config_yaml=None, risk_yaml="{}\n",
                  secrets_yaml=None) -> Path:
    """Write files under ``<tmp>/config``; ``None`` means "do not create"."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if config_yaml is not None:
        (cfg_dir / "config.yaml").write_text(config_yaml, encoding="utf-8")
    if risk_yaml is not None:
        (cfg_dir / "risk_params.yaml").write_text(risk_yaml, encoding="utf-8")
    if secrets_yaml is not None:
        (cfg_dir / "secrets.yaml").write_text(secrets_yaml, encoding="utf-8")
    return cfg_dir


def _load(tmp_path: Path, monkeypatch, **files):
    """Load ``Config`` with PROJECT_ROOT redirected at ``tmp_path``."""
    import app.config as cfg_mod
    from app.config import Config

    _write_config(tmp_path, **files)
    monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
    Config._instance = None
    return Config.load("sim")


#: A YAML syntax error that PyYAML reports exactly on line 4 (the indented
#: continuation of an already-closed mapping key).
BAD_YAML = "binance:\n  testnet: true\nweb_port: 8899\n  bad: indent\n"


def _strategy(name: str = "probe", timeframes=("1h",)):
    from core.strategy.loader import MLConfig, StrategyConfig

    return StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=list(timeframes),
        indicators={"rsi": {"period": 14, "source": "close"}},
        entry_conditions={"long": ["rsi < 30"], "short": []},
        exit_conditions={"long": [], "short": []},
        ml_config=MLConfig(enabled=False),
    )


# ======================================================================
# 1. malformed YAML → one clean, actionable line (no traceback)
# ======================================================================

def test_malformed_config_yaml_names_the_file_and_line(tmp_path, monkeypatch):
    from app.config import Config, ConfigError

    _write_config(tmp_path, config_yaml=BAD_YAML)
    import app.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
    Config._instance = None

    with pytest.raises(ConfigError) as excinfo:
        Config.load("sim")

    message = str(excinfo.value)
    assert message.startswith("config/config.yaml is not valid YAML (line 4): "), message
    assert "mapping values are not allowed here" in message
    # The typed config error — not the raw parser error the audit reported.
    import yaml
    assert not isinstance(excinfo.value, yaml.YAMLError)


def test_malformed_config_yaml_exits_1_at_the_startup_boundary(tmp_path, monkeypatch):
    """The boundary logs the clean message and raises SystemExit(1) — no traceback."""
    from app.config import Config
    from app.main import load_config_or_exit

    _write_config(tmp_path, config_yaml=BAD_YAML)
    import app.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
    Config._instance = None

    with captured_logs("ERROR") as logs:
        with pytest.raises(SystemExit) as excinfo:
            load_config_or_exit("sim")

    assert excinfo.value.code == 1
    assert any("config/config.yaml is not valid YAML (line 4)" in m for m in logs), logs


def test_malformed_secrets_yaml_names_the_file_and_line(tmp_path, monkeypatch):
    from app.config import Config, ConfigError

    _write_config(tmp_path, config_yaml="binance:\n  testnet: true\n",
                  secrets_yaml="auth:\n\tjwt_secret: x\n")
    import app.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
    Config._instance = None

    with pytest.raises(ConfigError) as excinfo:
        Config.load("sim")

    assert str(excinfo.value).startswith(
        "config/secrets.yaml is not valid YAML (line 2): "), str(excinfo.value)


def test_config_yaml_must_be_a_mapping(tmp_path, monkeypatch):
    """Valid YAML of the wrong shape is a config error, not an AttributeError."""
    from app.config import Config, ConfigError

    _write_config(tmp_path, config_yaml="- a\n- b\n")
    import app.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
    Config._instance = None

    with pytest.raises(ConfigError) as excinfo:
        Config.load("sim")

    assert "config/config.yaml must contain a YAML mapping" in str(excinfo.value)


# ======================================================================
# 2. web_port validation
# ======================================================================

@pytest.mark.parametrize("raw_yaml,expected", [
    ('"abc"', 8899),      # the audit case: non-int → default + warning
    ("70000", 65535),     # above the range → clamp
    ("0", 1),             # below the range → clamp
    ("-5", 1),
    ("null", 8899),       # key present but empty → default
])
def test_unusable_web_port_is_repaired_with_a_warning(tmp_path, monkeypatch,
                                                      raw_yaml, expected):
    with captured_logs() as logs:
        config = _load(tmp_path, monkeypatch,
                       config_yaml=f"binance:\n  testnet: true\nweb_port: {raw_yaml}\n")

    assert config.web_port == expected
    assert any("web_port" in m for m in logs), logs


def test_numeric_string_web_port_is_accepted_silently(tmp_path, monkeypatch):
    with captured_logs() as logs:
        config = _load(tmp_path, monkeypatch, config_yaml='web_port: "9001"\n')

    assert config.web_port == 9001
    assert not any("web_port" in m for m in logs), logs


def test_valid_web_port_is_untouched(tmp_path, monkeypatch):
    with captured_logs() as logs:
        config = _load(tmp_path, monkeypatch, config_yaml="web_port: 8123\n")

    assert config.web_port == 8123
    assert not any("web_port" in m for m in logs), logs


# ======================================================================
# 3. missing config files are announced (defaults are in use)
# ======================================================================

def test_missing_config_yaml_warns_that_defaults_are_in_use(tmp_path, monkeypatch):
    with captured_logs() as logs:
        config = _load(tmp_path, monkeypatch, config_yaml=None, risk_yaml="{}\n")

    assert any(m.startswith("config/config.yaml not found")
               and "default" in m for m in logs), logs
    assert config.web_port == 8899          # the documented default


def test_missing_risk_params_yaml_warns_that_defaults_are_in_use(tmp_path, monkeypatch):
    with captured_logs() as logs:
        _load(tmp_path, monkeypatch, config_yaml="web_port: 8899\n", risk_yaml=None)

    assert any(m.startswith("config/risk_params.yaml not found")
               and "default" in m for m in logs), logs


def test_missing_alert_rules_json_warns(tmp_path, monkeypatch):
    from app.event_bus import EventBus
    from app.main import warn_if_alert_rules_missing
    from alerts.manager import AlertManager

    config = _load(tmp_path, monkeypatch)
    manager = AlertManager(config, EventBus())
    with captured_logs() as logs:
        assert warn_if_alert_rules_missing(manager) is False

    assert any("alert_rules.json not found" in m for m in logs), logs


def test_present_alert_rules_json_is_not_warned(tmp_path, monkeypatch):
    from app.event_bus import EventBus
    from app.main import warn_if_alert_rules_missing
    from alerts.manager import AlertManager

    config = _load(tmp_path, monkeypatch)
    (tmp_path / "config" / "alert_rules.json").write_text("[]", encoding="utf-8")
    manager = AlertManager(config, EventBus())

    with captured_logs() as logs:
        assert warn_if_alert_rules_missing(manager) is True
    assert not any("alert_rules.json" in m for m in logs), logs


# ======================================================================
# 4. one "no market data" message, returned (never raised) on every path
# ======================================================================

class _EmptyFeeder:
    """DataFeeder stand-in with no frames at all."""

    def __init__(self, date_start: str = "2026-01-01"):
        self.date_start = pd.Timestamp(date_start)

    def get_all_data_for_symbol(self, symbol: str, interval: str) -> pd.DataFrame:
        return pd.DataFrame()


def _assert_actionable(message: str):
    # The legacy phrase stays in the message: `tests/test_hybrid_equivalence.py`
    # detects "no data for this period" by testing for it, and a reworded message
    # silently turned that graceful skip into a failure.
    assert "No historical data" in message, message
    assert "scripts/download_history.py" in message, message
    assert "/api/backtest/fetch-data" in message, message


def test_signal_matrix_no_data_returns_the_unified_message_without_raising():
    from core.backtest.signal_matrix import (NO_MARKET_DATA_MESSAGE,
                                             SignalMatrixBuilder)

    with captured_logs() as logs:
        matrix = SignalMatrixBuilder(_EmptyFeeder()).build([_strategy()], ["BTCUSDT"])

    assert matrix.metadata["error"] == NO_MARKET_DATA_MESSAGE
    assert matrix.metadata["timestamp_count"] == 0
    assert matrix.signals.empty and matrix.exit_signals.empty
    _assert_actionable(matrix.metadata["error"])
    assert any(NO_MARKET_DATA_MESSAGE in m for m in logs), logs


def test_signal_matrix_no_data_with_no_symbols_does_not_raise():
    from core.backtest.signal_matrix import (NO_MARKET_DATA_MESSAGE,
                                             SignalMatrixBuilder)

    matrix = SignalMatrixBuilder(_EmptyFeeder()).build([_strategy()], [])
    assert matrix.metadata["error"] == NO_MARKET_DATA_MESSAGE


def test_hybrid_engine_no_data_returns_the_unified_message(tmp_path):
    from core.backtest.engine_hybrid import run_hybrid
    from core.backtest.signal_matrix import NO_MARKET_DATA_MESSAGE

    class _Cfg:
        data_dir = str(tmp_path / "data")

    (tmp_path / "data").mkdir()

    result = run_hybrid([_strategy()], ["BTCUSDT"], "2026-01-01", "2026-02-01",
                        _Cfg(), None)

    assert result == {"error": NO_MARKET_DATA_MESSAGE}
    _assert_actionable(result["error"])


def test_legacy_engine_no_data_returns_the_unified_message(tmp_path):
    from core.backtest.engine import BacktestEngine
    from core.backtest.signal_matrix import NO_MARKET_DATA_MESSAGE

    class _Cfg:
        data_dir = str(tmp_path / "data")
        backtest_engine_mode = "legacy"
        backtest_spread_pct = {}
        backtest_default_spread_pct = 0.03
        backtest_live_spread_enabled = False
        backtest_live_spread_timeout = 3.0
        backtest_live_spread_ttl = 300.0

    (tmp_path / "data").mkdir()
    engine = BacktestEngine(_Cfg(), None, None, None)

    result = engine.run_with_exit_evaluation(
        [_strategy()], ["BTCUSDT"], "2026-01-01", "2026-02-01")

    assert result == {"error": NO_MARKET_DATA_MESSAGE}
    _assert_actionable(result["error"])


def test_backtest_result_card_shows_the_unified_message(tmp_path, monkeypatch):
    """The web run must surface the message in the RESULT CARD, not only in JSON."""
    from fastapi.testclient import TestClient
    from starlette.middleware.base import BaseHTTPMiddleware

    from app.event_bus import EventBus
    from core.backtest.signal_matrix import NO_MARKET_DATA_MESSAGE
    from web.server import create_app

    config = _load(tmp_path, monkeypatch)
    (tmp_path / "data").mkdir(exist_ok=True)

    class _StubEngine:
        def run_with_exit_evaluation(self, *args, **kwargs):
            return {"error": NO_MARKET_DATA_MESSAGE}

    class _User:
        username = "tester"
        is_trader = True
        is_admin = False

    class _InjectUser(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.user = _User()
            return await call_next(request)

    app = create_app(config, EventBus(), None)
    app.add_middleware(_InjectUser)
    app.state.config = config
    app.state.backtest_engine = _StubEngine()

    # `_bt_runs` lives on the per-create_app AppContext, so this test cannot
    # leak run state into another test.
    with TestClient(app) as client:
        response = client.post("/api/backtest/run", data={
            "strategies": "trend", "symbols": "BTCUSDT",
            "date_start": "2026-01-01", "date_end": "2026-02-01"})
        assert response.status_code == 200, response.text
        run_id = (response.cookies.get("bt_active_run")
                  or re.search(r"var runId = '([^']+)'", response.text).group(1))
        assert run_id

        card = ""
        for _ in range(100):
            card = client.get(f"/api/backtest/result/{run_id}").text
            if "Still running" not in card:
                break
            time.sleep(0.05)

    assert NO_MARKET_DATA_MESSAGE in card, card[:500]
    assert "scripts/download_history.py" in card


# ======================================================================
# 5. audit_db on a fresh database: "no ledger yet", exit 0
# ======================================================================

def _audit_main(argv):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_db_under_test", ROOT / "scripts" / "audit_db.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv)


def test_audit_db_fresh_empty_file_is_not_drift(tmp_path, capsys):
    db = tmp_path / "fresh.db"
    db.write_bytes(b"")

    assert _audit_main(["audit_db.py", str(db)]) == 0

    out = capsys.readouterr().out
    assert "FRESH DATABASE — NO LEDGER YET" in out
    assert "delta" not in out            # no phantom -10000
    assert "-10000" not in out


def test_audit_db_schema_without_sim_balance_row_is_not_drift(tmp_path, capsys):
    db = tmp_path / "no_balance.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE system_config (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, quantity REAL,"
                     " entry_price REAL, pnl REAL, status TEXT, action TEXT,"
                     " closed_at TEXT, exit_price REAL)")
        conn.commit()
    finally:
        conn.close()

    assert _audit_main(["audit_db.py", str(db)]) == 0

    out = capsys.readouterr().out
    assert "FRESH DATABASE — NO LEDGER YET" in out
    assert "sim_balance            = (no row)" in out


def test_audit_db_real_drift_still_exits_1(tmp_path, capsys):
    db = tmp_path / "drifted.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE system_config (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, quantity REAL,"
                     " entry_price REAL, pnl REAL, status TEXT, action TEXT,"
                     " closed_at TEXT, exit_price REAL)")
        conn.execute("INSERT INTO system_config VALUES ('sim_balance', '9000')")
        conn.commit()
    finally:
        conn.close()

    assert _audit_main(["audit_db.py", str(db)]) == 1

    out = capsys.readouterr().out
    assert "IDENTITY BROKEN" in out
    assert "delta (balance - ident)= -1000.000000" in out


def test_audit_db_healthy_ledger_exits_0(tmp_path, capsys):
    db = tmp_path / "healthy.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE system_config (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, quantity REAL,"
                     " entry_price REAL, pnl REAL, status TEXT, action TEXT,"
                     " closed_at TEXT, exit_price REAL)")
        conn.execute("INSERT INTO system_config VALUES ('sim_balance', '10000')")
        conn.commit()
    finally:
        conn.close()

    assert _audit_main(["audit_db.py", str(db)]) == 0
    assert "IDENTITY HOLDS" in capsys.readouterr().out


def test_audit_db_help_documents_the_fresh_database_case(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _audit_main(["audit_db.py", "--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "fresh" in out
    assert "Exit codes" in out


# ======================================================================
# 6. pre-flight port probe
# ======================================================================

def test_port_probe_flags_a_listener_and_clears_a_closed_port():
    from app.main import port_in_use

    server = socket.socket()
    busy_port = None
    try:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        busy_port = server.getsockname()[1]
        assert port_in_use(busy_port) is True
    finally:
        server.close()

    # A closed listener must read as free — a TIME_WAIT socket from the previous
    # instance may not abort a valid restart.
    assert port_in_use(busy_port) is False
