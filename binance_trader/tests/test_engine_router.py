"""Unit tests for engine mode routing logic."""
import pytest
from core.strategy.loader import StrategyConfig, MLConfig


def _make_configs(n=3, ml_enabled=False):
    """Helper: create N minimal StrategyConfigs."""
    configs = []
    for i in range(n):
        configs.append(StrategyConfig(
            name=f"test_{i}", enabled=True, mode="trend", timeframes=["1h"],
            indicators={"rsi": {"period": 14, "source": "close"}},
            entry_conditions={"long": [], "short": []},
            exit_conditions={"long": [], "short": []},
            ml_config=MLConfig(enabled=ml_enabled),
        ))
    return configs


def test_router_auto_hybrid_with_3_strategies():
    """3+ strategies, no ML -> hybrid."""
    from core.backtest.engine import BacktestEngine
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config

    result = engine._select_engine(_make_configs(3), "auto")
    assert result == "hybrid"


def test_router_auto_legacy_with_1_strategy():
    """1 strategy -> legacy."""
    from core.backtest.engine import BacktestEngine
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config

    result = engine._select_engine(_make_configs(1), "auto")
    assert result == "legacy"


def test_router_force_legacy():
    """Explicit legacy mode always returns legacy."""
    from core.backtest.engine import BacktestEngine
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config

    result = engine._select_engine(_make_configs(5), "legacy")
    assert result == "legacy"


def test_router_force_hybrid():
    """Explicit hybrid mode returns hybrid when no ML."""
    from core.backtest.engine import BacktestEngine
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config

    result = engine._select_engine(_make_configs(1, ml_enabled=False), "hybrid")
    assert result == "hybrid"


def test_router_hybrid_with_ml_raises():
    """Hybrid mode with ML enabled should raise ValueError."""
    from core.backtest.engine import BacktestEngine
    from app.config import Config

    Config._instance = None
    config = Config.load("sim")
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = config

    with pytest.raises(ValueError, match="Hybrid engine does not support ML"):
        engine._select_engine(_make_configs(1, ml_enabled=True), "hybrid")
