import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier, XGBRegressor


class MLTrainer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir = self.data_dir / "ml_training"
        self.training_dir.mkdir(parents=True, exist_ok=True)

    def train_binary(self, symbol: str, strategy_name: str,
                     X: pd.DataFrame, y: pd.Series) -> dict:
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

        model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        model_path = self._save_model(model, symbol, strategy_name, "binary")
        feature_importance = dict(zip(X.columns, model.feature_importances_.tolist()))

        return {
            "accuracy": acc,
            "f1_score": f1,
            "model_path": model_path,
            "feature_importance": feature_importance,
            "n_samples": len(X),
        }

    def train_regression(self, symbol: str, strategy_name: str,
                         X: pd.DataFrame, y: pd.Series) -> dict:
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

        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = np.mean((y_test.values - preds) ** 2)
        mae = np.mean(np.abs(y_test.values - preds))

        model_path = self._save_model(model, symbol, strategy_name, "regression")
        feature_importance = dict(zip(X.columns, model.feature_importances_.tolist()))

        return {
            "mse": mse,
            "mae": mae,
            "model_path": model_path,
            "feature_importance": feature_importance,
            "n_samples": len(X),
        }

    def _save_model(self, model, symbol: str, strategy_name: str, model_type: str) -> str:
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

    def load_training_data(self, symbol: str, strategy_name: str) -> Optional[pd.DataFrame]:
        path = self.training_dir / f"{symbol}_{strategy_name}_features.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None
