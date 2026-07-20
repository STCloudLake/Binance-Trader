"""Model trainer — LightGBM (primary) with XGBoost fallback."""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score


class MLTrainer:
    """Trains and persists ML models.

    Uses LightGBM by default (faster, often more accurate on tabular data).
    Falls back to XGBoost if LightGBM is unavailable.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir = self.data_dir / "ml_training"
        self.training_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────────

    def train_binary(self, symbol: str, strategy_name: str,
                     X: pd.DataFrame, y: pd.Series,
                     engine: str = "lightgbm") -> dict:
        """Train a binary classifier.

        Parameters
        ----------
        engine : str
            'lightgbm' (default), 'xgboost', or 'auto'.
            'auto' tries LightGBM first, falls back to XGBoost.
        """
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

        if engine == "auto":
            model, used_engine = self._train_lgb(X_train, y_train, X_test, y_test)
            if model is None:
                model, used_engine = self._train_xgb(X_train, y_train, X_test, y_test)
        elif engine == "lightgbm":
            model, used_engine = self._train_lgb(X_train, y_train, X_test, y_test)
        else:
            model, used_engine = self._train_xgb(X_train, y_train, X_test, y_test)

        if model is None:
            return {"error": "Failed to train any model"}

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        model_path = self._save_model(model, symbol, strategy_name, "binary")
        feature_importance = dict(zip(
            X.columns,
            (model.feature_importances_.tolist()
             if hasattr(model, "feature_importances_") else []),
        ))

        return {
            "accuracy": acc,
            "f1_score": f1,
            "model_path": model_path,
            "feature_importance": feature_importance,
            "n_samples": len(X),
            "engine": used_engine,
        }

    def train_regression(self, symbol: str, strategy_name: str,
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
            model, used_engine = self._train_lgb_reg(X_train, y_train, X_test, y_test)
            if model is None and engine == "auto":
                model, used_engine = self._train_xgb_reg(X_train, y_train, X_test, y_test)
        else:
            model, used_engine = self._train_xgb_reg(X_train, y_train, X_test, y_test)

        if model is None:
            return {"error": "Failed to train any model"}

        preds = model.predict(X_test)
        mse = float(np.mean((y_test.values - preds) ** 2))
        mae = float(np.mean(np.abs(y_test.values - preds)))

        model_path = self._save_model(model, symbol, strategy_name, "regression")
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

    # ── LightGBM trainers ────────────────────────────────────────────

    def _train_lgb(self, X_train, y_train, X_test, y_test):
        try:
            import lightgbm as lgb

            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            scale_pos_weight = max(1.0, n_neg / max(n_pos, 1))

            model = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
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

    def _train_lgb_reg(self, X_train, y_train, X_test, y_test):
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

    # ── XGBoost trainers (fallback) ──────────────────────────────────

    def _train_xgb(self, X_train, y_train, X_test, y_test):
        try:
            from xgboost import XGBClassifier

            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            scale_pos_weight = max(1.0, n_neg / max(n_pos, 1))

            model = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss', verbosity=0, random_state=42,
            )
            model.fit(X_train, y_train)
            return model, "xgboost"
        except Exception:
            return None, "xgboost_error"

    def _train_xgb_reg(self, X_train, y_train, X_test, y_test):
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

    # ── Persistence ──────────────────────────────────────────────────

    def _save_model(self, model, symbol: str, strategy_name: str,
                    model_type: str) -> str:
        filename = f"{symbol}_{strategy_name}_{model_type}.pkl"
        path = self.models_dir / filename
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return str(path)

    def load_model(self, file_path: str):
        if not Path(file_path).exists():
            return None
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def save_training_data(self, symbol: str, strategy_name: str,
                           X: pd.DataFrame, y: pd.Series):
        df = X.copy()
        df["label"] = y
        path = self.training_dir / f"{symbol}_{strategy_name}_features.parquet"
        df.to_parquet(path)

    def load_training_data(self, symbol: str,
                           strategy_name: str) -> Optional[pd.DataFrame]:
        path = self.training_dir / f"{symbol}_{strategy_name}_features.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None
