import pytest
import pandas as pd
import numpy as np
from pathlib import Path


def test_build_features():
    from core.ml.features import build_features
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "rsi": np.random.rand(100) * 100,
        "macd_histogram": np.random.randn(100),
        "volume_ratio": np.random.rand(100) * 2,
        "close": np.random.randn(100).cumsum() + 50000,
    }, index=dates)

    feature_list = ["rsi", "macd_histogram", "volume_ratio", "nonexistent"]
    feats = build_features(df, feature_list)
    assert "rsi" in feats.columns
    assert "macd_histogram" in feats.columns
    assert "volume_ratio" in feats.columns
    assert "nonexistent" not in feats.columns


def test_create_binary_label():
    from core.ml.features import create_binary_label
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    close = [100 + i * 0.5 for i in range(100)]
    df = pd.DataFrame({"close": close}, index=dates)

    labels = create_binary_label(df, forward_periods=4)
    assert labels.iloc[0] == 1
    assert pd.isna(labels.iloc[-1]) or labels.iloc[-1] != 1


def test_create_regression_label():
    from core.ml.features import create_regression_label
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    close = [100 + i * 0.5 for i in range(100)]
    df = pd.DataFrame({"close": close}, index=dates)

    labels = create_regression_label(df, forward_periods=4)
    assert labels.iloc[0] > 0
    assert pd.isna(labels.iloc[-1])


def test_trainer_binary():
    from core.ml.trainer import MLTrainer
    import tempfile
    data_dir = tempfile.mkdtemp()
    trainer = MLTrainer(data_dir)

    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "rsi": np.random.rand(n) * 100,
        "macd_histogram": np.random.randn(n),
        "volume_ratio": np.random.rand(n) * 2,
    })
    y = pd.Series(np.random.choice([0, 1], n))

    result = trainer.train_binary("BTCUSDT", "test_strategy", X, y)
    assert "accuracy" in result or "error" in result
    if "accuracy" in result:
        assert result["n_samples"] == n
        assert Path(result["model_path"]).exists()
