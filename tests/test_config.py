import pytest
import os
import tempfile
from pathlib import Path
import sys

# Ensure the package root is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loads_with_defaults():
    """The loader must reflect the YAML files, and the schema must expose sane defaults.

    Deliberately NOT asserting hard-coded shipped values (e.g. "max_leverage == 4"):
    risk limits are user-editable through the settings UI, so a legitimate settings
    change must not break the test suite. We assert loader fidelity (config value ==
    value in the YAML) plus the pydantic schema defaults instead.
    """
    import yaml

    from app.config import Config, HardRiskLimits
    Config._instance = None
    config = Config.load("sim")
    assert config.mode == "sim"
    assert config.web_port == 8899
    assert config.soft_params.risk_appetite == "balanced"

    risk_yaml = Path(__file__).parent.parent / "config" / "risk_params.yaml"
    if risk_yaml.exists():
        raw = yaml.safe_load(risk_yaml.read_text(encoding="utf-8")) or {}
        file_leverage = raw.get("hard_limits", {}).get("max_leverage")
        if file_leverage is not None:
            assert config.hard_limits.max_leverage == file_leverage, \
                "hard_limits must come from risk_params.yaml (it overrides config.yaml)"

    # Schema defaults (what applies when a key is absent from every YAML file)
    assert HardRiskLimits().max_leverage == 3
    assert HardRiskLimits().max_open_trades == 8


def test_config_env_override():
    from app.config import Config
    Config._instance = None
    os.environ["DEEPSEEK_API_KEY"] = "test_key_123"
    config = Config.load("sim")
    assert config.deepseek_api_key == "test_key_123"
    del os.environ["DEEPSEEK_API_KEY"]


def test_backtest_config_defaults():
    from app.config import Config
    Config._instance = None
    config = Config.load("sim")
    assert config.backtest_engine_mode == "auto"
    assert config.backtest_ml_enabled is False


def test_backtest_cost_model_config():
    from app.config import Config
    Config._instance = None
    config = Config.load("sim")
    assert config.backtest_cost_enabled is True
    assert config.backtest_taker_fee_pct == 0.04
    spreads = config.backtest_spread_pct
    assert isinstance(spreads, dict)
    assert "BTCUSDT" in spreads
    assert spreads["BTCUSDT"] == 0.01


def test_config_singleton():
    from app.config import Config
    Config._instance = None
    c1 = Config.load("sim")
    c2 = Config.load("sim")
    assert c1 is c2
