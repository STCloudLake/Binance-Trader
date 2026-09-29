"""``MLTrainer`` methods that were never called.

Moved from ``core/ml/trainer.py``.  They take the live ``MLTrainer`` instance as
``trainer`` (previously ``self``) and use ``trainer._save_model`` for
persistence.  ``train_binary``, ``load_model``, ``save_training_data`` and the
``_train_lgb``/``_train_xgb`` helpers it uses stayed in ``core.ml.trainer``.

Why these are here:

* ``train_regression`` — only the binary classifier is wired up; the TFT path
  that wanted regression labels builds its own training loop in
  ``core/backtest/engine.py`` and never calls this.  Its two private helpers
  (:func:`_train_lgb_reg`, :func:`_train_xgb_reg`) had no other caller either,
  so they moved with it.
* ``load_training_data`` — the write side (``save_training_data``) is used by
  ``MLPredictor.train_model``; nothing ever read the parquet back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def train_regression(trainer, symbol: str, strategy_name: str,
                     X: pd.DataFrame, y: pd.Series,
                     engine: str = "lightgbm") -> dict:
    """Train a regressor (returns)."""
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.dropna()
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    if len(X) < 50:
        return {"error": "Insufficient training data"}

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if engine in ("lightgbm", "auto"):
        model, used_engine = _train_lgb_reg(X_train, y_train, X_test, y_test)
        if model is None and engine == "auto":
            model, used_engine = _train_xgb_reg(X_train, y_train, X_test, y_test)
    else:
        model, used_engine = _train_xgb_reg(X_train, y_train, X_test, y_test)

    if model is None:
        return {"error": "Failed to train any model"}

    preds = model.predict(X_test)
    mse = float(np.mean((y_test.values - preds) ** 2))
    mae = float(np.mean(np.abs(y_test.values - preds)))

    model_path = trainer._save_model(model, symbol, strategy_name, "regression")
    feature_importance = dict(zip(
        X.columns,
        (model.feature_importances_.tolist()
         if hasattr(model, "feature_importances_") else []),
    ))

    return {
        "mse": mse, "mae": mae,
        "model_path": model_path,
        "feature_importance": feature_importance,
        "n_samples": len(X),
        "engine": used_engine,
    }


def load_training_data(trainer, symbol: str,
                       strategy_name: str) -> Optional[pd.DataFrame]:
    path = trainer.training_dir / f"{symbol}_{strategy_name}_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def _train_lgb_reg(X_train, y_train, X_test, y_test):
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbosity=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model, "lightgbm"
    except ImportError:
        return None, "lightgbm_unavailable"
    except Exception:
        return None, "lightgbm_error"


def _train_xgb_reg(X_train, y_train, X_test, y_test):
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model, "xgboost"
    except Exception:
        return None, "xgboost_error"
