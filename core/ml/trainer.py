"""Model trainer — LightGBM (primary) with XGBoost fallback.

Phase P2 changes
----------------
* the reported metrics come from a **purged + embargoed** split
  (:mod:`core.ml.evaluation`), not a chronological tail that shares its forward
  window with the training rows;
* ``accuracy`` alone is no longer the selection signal — the caller goes through
  :func:`core.ml.credibility.evaluate_model_oos`, which selects the probability
  threshold that maximises **net-of-cost expectancy**;
* every model artefact is persisted with a ``*_meta.json`` sidecar holding the
  feature contract, the training base rate, the calibrator and the gate status,
  so a loader can refuse a model whose inputs have drifted;
* ``f1`` is returned *and* ``f1_score`` is kept as an alias (the old key was
  always ``N/A`` in the logs because the writer and reader disagreed);
* ``eval_set`` early stopping is enabled for LightGBM/XGBoost;
* a train-time scaler is persisted (``core.ml.scalers.TrainTimeScaler``) so TFT
  inference uses the same normalisation as training instead of global stats.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

#: Deterministic seed for every trainer (the plan requires same-seed reproducibility).
SEED = 42


def default_binary_factory(feature_names: list[str] | None = None,
                           n_estimators: int = 150):
    """Return ``fit(X, y, sample_weight=None) -> model`` (LightGBM, XGB fallback).

    Used by :func:`core.ml.credibility.evaluate_model_oos`, which calls the
    factory once per purged fold with that fold's uniqueness weights.  A
    ``sample_weight`` the backend cannot accept is *raised*, never swallowed —
    silently dropping the weights (or the fold) is how a gate ends up "passing"
    on no data at all.
    """
    def _factory(X: pd.DataFrame, y: pd.Series, sample_weight=None):
        if len(X) == 0 or len(np.unique(y)) < 2:
            raise ValueError(
                f"default_binary_factory: need >=2 classes and rows "
                f"(rows={len(X)}, classes={len(np.unique(y))})")
        try:
            import lightgbm as lgb
        except ImportError:
            lgb = None
        if lgb is not None:
            n_pos = int(np.asarray(y).sum())
            n_neg = len(y) - n_pos
            model = lgb.LGBMClassifier(
                n_estimators=int(n_estimators), max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=max(1.0, n_neg / max(n_pos, 1)),
                min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
                verbosity=-1, random_state=SEED, deterministic=True,
                force_col_wise=True,
            )
            model.fit(X, y, sample_weight=sample_weight)
            return model
        from xgboost import XGBClassifier
        n_pos = int(np.asarray(y).sum())
        n_neg = len(y) - n_pos
        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=max(1.0, n_neg / max(n_pos, 1)),
            eval_metric="logloss", verbosity=0, random_state=SEED, n_jobs=1)
        model.fit(X, y, sample_weight=sample_weight, verbose=False)
        return model

    return _factory


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
                     engine: str = "lightgbm",
                     *,
                     calibrate: bool = True,
                     base_rate: float | None = None,
                     threshold: float | None = None,
                     threshold_up: float | None = None,
                     threshold_down: float | None = None,
                     threshold_side: str | None = None,
                     gate: dict | None = None,
                     extra_meta: dict | None = None) -> dict:
        """Train a binary classifier.

        Parameters
        ----------
        engine : str
            'lightgbm' (default), 'xgboost', or 'auto'.
        calibrate : bool
            Fit an isotonic calibrator on a held-out 20 % of the *training* rows
            and persist it with the model (default True).
        base_rate : float | None
            Training-set positive rate; persisted so the signed score is centred
            correctly at inference time.
        threshold : float | None
            Cost-aware decision threshold chosen OOS; persisted with the gate.
        threshold_up, threshold_down, threshold_side : optional
            **Both** side thresholds and the side the outer protocol selected
            (audit P2 #4).  Persisting only ``threshold_up`` while the short side
            won made the live band meaningless; all three are written to
            ``thresholds`` in the sidecar.  ``threshold`` stays for compatibility
            readers and is the selected side's own threshold.
        gate : dict | None
            Status from :func:`core.ml.credibility.credibility_gate`.  When it
            says ``enabled`` is False the artefact is still written (for
            inspection) but the meta records the refusal and its reason.
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
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        # Out-of-sample quality on the held-out tail (NOT the purged OOS gate —
        # that is `credibility.evaluate_model_oos`, which the caller runs).
        proba = self._proba_up(model, X_test)
        oos = {
            "accuracy": float(acc),
            "auc": self._auc(y_test.values, proba),
            "brier": float(brier_score_loss(y_test.values, proba))
            if len(np.unique(y_test.values)) > 1 else float(np.mean((proba - y_test.values) ** 2)),
            "log_loss": self._log_loss(y_test.values, proba),
            "base_rate": float(y_train.mean()) if len(y_train) else 0.5,
        }

        # Calibration fitted on the tail of the TRAIN block (never the test rows).
        calibrator = None
        cal_cut = int(len(X_train) * 0.75)
        if calibrate and cal_cut >= 30 and len(X_train) - cal_cut >= 10:
            from core.ml.calibration import ProbabilityCalibrator
            p_fit = self._proba_up(model, X_train.iloc[cal_cut:])
            calibrator = ProbabilityCalibrator("isotonic").fit(p_fit, y_train.iloc[cal_cut:].values)
            if calibrator.fitted:
                oos["auc_calibrated"] = self._auc(
                    y_test.values, calibrator.transform(proba))

        model_path = self._save_model(model, symbol, strategy_name, "binary")
        meta_path = self._save_meta(
            symbol, strategy_name, "binary", X=X, y_train=y_train, oos=oos,
            calibrator=calibrator, threshold=threshold,
            threshold_up=threshold_up, threshold_down=threshold_down,
            threshold_side=threshold_side,
            base_rate=(base_rate if base_rate is not None else oos["base_rate"]),
            gate=gate, engine=used_engine, extra=extra_meta)

        feature_importance = dict(zip(
            X.columns,
            (model.feature_importances_.tolist()
             if hasattr(model, "feature_importances_") else []),
        ))

        result = {
            "accuracy": float(acc),
            "f1": float(f1),          # honest name
            "f1_score": float(f1),    # kept for existing log/report readers
            "auc": oos["auc"],
            "brier": oos["brier"],
            "log_loss": oos["log_loss"],
            "model_path": model_path,
            "meta_path": meta_path,
            "feature_importance": feature_importance,
            "n_samples": len(X),
            "n_test": int(len(X_test)),
            "engine": used_engine,
            "calibrated": bool(calibrator is not None and calibrator.fitted),
            "base_rate": oos["base_rate"],
        }
        if gate is not None:
            result["enabled"] = bool(gate.get("enabled", False))
            result["gate"] = gate
        return result

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
                random_state=SEED,
                deterministic=True,
                force_col_wise=True,
            )
            fit_kwargs = {}
            if X_test is not None and len(X_test) > 0 and len(np.unique(y_test)) > 1:
                fit_kwargs["eval_set"] = [(X_test, y_test)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(20, verbose=False)]
            model.fit(X_train, y_train, **fit_kwargs)
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
                eval_metric='logloss', verbosity=0, random_state=SEED, n_jobs=1,
            )
            fit_kwargs = {}
            if X_test is not None and len(X_test) > 0 and len(np.unique(y_test)) > 1:
                fit_kwargs["eval_set"] = [(X_test, y_test)]
                fit_kwargs["verbose"] = False
            model.fit(X_train, y_train, **fit_kwargs)
            return model, "xgboost"
        except Exception:
            return None, "xgboost_error"

    # ── metric helpers ───────────────────────────────────────────────

    @staticmethod
    def _proba_up(model, X) -> np.ndarray:
        p = np.asarray(model.predict_proba(X), dtype=float)
        if p.ndim == 1:
            return p
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return p[:, classes.index(1)]
        return p[:, -1]

    @staticmethod
    def _auc(y, p) -> float:
        y = np.asarray(y, dtype=float)
        if len(np.unique(y)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(y, np.asarray(p, dtype=float)))
        except Exception:
            return 0.5

    @staticmethod
    def _log_loss(y, p) -> float:
        y = np.asarray(y, dtype=float)
        p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # ── Persistence ──────────────────────────────────────────────────

    def _save_model(self, model, symbol: str, strategy_name: str,
                    model_type: str) -> str:
        filename = f"{symbol}_{strategy_name}_{model_type}.pkl"
        path = self.models_dir / filename
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return str(path)

    def _meta_path(self, symbol: str, strategy_name: str, model_type: str) -> Path:
        return self.models_dir / f"{symbol}_{strategy_name}_{model_type}_meta.json"

    def _save_meta(self, symbol: str, strategy_name: str, model_type: str, *,
                   X: pd.DataFrame, y_train: pd.Series, oos: dict,
                   calibrator=None, threshold=None, base_rate: float | None = None,
                   threshold_up: float | None = None,
                   threshold_down: float | None = None,
                   threshold_side: str | None = None,
                   gate: dict | None = None, engine: str = "",
                   extra: dict | None = None) -> str:
        """Write the sidecar that makes a model artefact self-describing."""
        payload = {
            "symbol": symbol,
            "strategy_name": strategy_name,
            "model_type": model_type,
            "engine": engine,
            "feature_names": [str(c) for c in X.columns],
            "n_features": int(X.shape[1]),
            "n_samples": int(len(X)),
            "n_train": int(len(y_train)),
            "train_base_rate": float(base_rate if base_rate is not None
                                     else (y_train.mean() if len(y_train) else 0.5)),
            "threshold": None if threshold is None else float(threshold),
            # Both side thresholds + the side that was selected (audit P2 #4).
            "thresholds": {
                "threshold_up": None if threshold_up is None else float(threshold_up),
                "threshold_down": None if threshold_down is None else float(threshold_down),
                "side": threshold_side,
                "selection": "nested_per_fold_oos",
            },
            "oos_holdout": {k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in oos.items()},
            "calibrator": calibrator.to_dict() if calibrator is not None else None,
            "gate": gate,
            "seed": SEED,
        }
        if extra:
            payload.update(extra)
        path = self._meta_path(symbol, strategy_name, model_type)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return str(path)

    def load_meta(self, symbol: str, strategy_name: str,
                  model_type: str = "binary") -> dict | None:
        path = self._meta_path(symbol, strategy_name, model_type)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

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
