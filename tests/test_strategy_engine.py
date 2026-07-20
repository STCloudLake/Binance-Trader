import pytest
import tempfile
import yaml
from pathlib import Path
import pandas as pd
import numpy as np


def test_strategy_loader_load():
    from core.strategy.loader import StrategyLoader
    d = tempfile.mkdtemp()
    strat_dir = Path(d)
    config = {
        "name": "Test Strategy",
        "mode": "trend",
        "timeframes": ["1h"],
        "indicators": {"rsi": {"period": 14, "source": "close"}},
        "entry_conditions": {"long": ["rsi < 30"]},
        "exit_conditions": {"long": ["rsi > 70"]},
    }
    with open(strat_dir / "test_strategy.yaml", "w") as f:
        yaml.dump(config, f)

    loader = StrategyLoader(str(strat_dir))
    loaded = loader.load("test_strategy")
    assert loaded.name == "Test Strategy"
    assert loaded.timeframes == ["1h"]


def test_strategy_loader_list_all():
    from core.strategy.loader import StrategyLoader
    d = tempfile.mkdtemp()
    strat_dir = Path(d)
    for i in range(3):
        config = {
            "name": f"Strategy {i}",
            "mode": "trend",
            "timeframes": ["1h"],
            "indicators": {},
            "entry_conditions": {},
            "exit_conditions": {},
        }
        with open(strat_dir / f"s{i}.yaml", "w") as f:
            yaml.dump(config, f)

    loader = StrategyLoader(str(strat_dir))
    names = loader.list_names()
    assert len(names) == 3


def test_compute_indicators():
    from core.strategy.indicators import compute_all
    dates = pd.date_range("2024-01-01", periods=200, freq="1h")
    np.random.seed(42)
    close = np.random.randn(200).cumsum() + 50000
    df = pd.DataFrame({
        "open": close + np.random.randn(200) * 100,
        "high": close + abs(np.random.randn(200)) * 200,
        "low": close - abs(np.random.randn(200)) * 200,
        "close": close,
        "volume": np.random.rand(200) * 100 + 50,
    }, index=dates)

    configs = {"rsi": {"period": 14, "source": "close"},
               "macd": {"fast": 12, "slow": 26, "signal": 9}}
    result = compute_all(df, configs)

    assert "rsi" in result.columns
    assert "macd" in result.columns
    assert "macd_histogram" in result.columns
    assert "volume_ratio" in result.columns
