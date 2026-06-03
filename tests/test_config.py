import pytest
import os
import tempfile
from pathlib import Path
import sys

# Ensure the package root is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loads_with_defaults():
    from app.config import Config
    Config._instance = None
    config = Config.load("sim")
    assert config.mode == "sim"
    assert config.web_port == 8899
    assert config.hard_limits.max_leverage == 4
    assert config.soft_params.risk_appetite == "balanced"
    assert config.signal_weights.indicator == 0.5


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


def test_config_singleton():
    from app.config import Config
    Config._instance = None
    c1 = Config.load("sim")
    c2 = Config.load("sim")
    assert c1 is c2
