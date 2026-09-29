"""``MLPredictor`` capabilities that were never called.

Moved from ``core/ml/predictor.py``.  They take the live ``MLPredictor``
instance as ``predictor`` (previously ``self``), so nothing about them is
rewritten — only the receiver became explicit.  ``MLPredictor`` itself,
``predict()``, ``_predict_lgb``/``_predict_tft``/``_predict_volatility`` and
``train_model``/``train_tft_model``/``train_volatility_model`` all stayed in
``core.ml.predictor`` because the live path uses them.

Why these four are here:

* ``load_tft_model`` / ``load_patchtst_model`` — the loaders exist, but nothing
  ever called them; models are trained by ``train_tft_model`` or the backtest
  engine, which loads its own TFT/PatchTST trainers directly.
* ``ensemble_predict`` — the Phase-4d multi-architecture vote.  ``predict()``
  routes to a single architecture, so the ensemble was never reachable; its
  private helper :func:`_predict_patchtst` came with it.
* ``incremental_retrain`` — online LightGBM update, never scheduled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from core.ml.features import compute_features, create_binary_label


def load_tft_model(predictor, symbol: str, strategy_name: str):
    """Load a pre-trained TFT model into ``predictor._tft_models[symbol]``."""
    tft = predictor._get_tft_trainer()
    model = tft.load(symbol, strategy_name)
    if model:
        predictor._tft_models[symbol] = model


async def ensemble_predict(predictor, symbol: str, X: pd.DataFrame) -> dict:
    """Weighted ensemble across all available model architectures.

    Combines LightGBM, TFT, and PatchTST predictions.  Weights are
    fixed for now (can be evolved by future GA work).

    Returns:
        dict with ``confidence`` (float [0,1]) and ``volatility_expanding`` (bool).
        Falls back to individual models when only one is available.
    """
    predictions: dict[str, float | None] = {}

    # LightGBM direction
    if f"{symbol}_binary" in predictor._models:
        predictions["lgb"] = await predictor._predict_lgb(symbol, X)

    # TFT
    if symbol in predictor._tft_models:
        predictions["tft"] = await predictor._predict_tft(symbol, X)

    # PatchTST (lazy init)
    if hasattr(predictor, '_patch_models') and symbol in predictor._patch_models:
        predictions["patch"] = await _predict_patchtst(predictor, symbol, X)

    # Ensemble: weighted average of valid predictions
    weights = {"lgb": 0.40, "tft": 0.35, "patch": 0.25}
    valid = {k: v for k, v in predictions.items()
             if v is not None and not np.isnan(v)}

    if not valid:
        return {"confidence": 0.5, "volatility_expanding": False}

    total_w = sum(weights.get(k, 0.0) for k in valid)
    if total_w > 0:
        confidence = sum(v * weights[k] / total_w for k, v in valid.items())
    else:
        confidence = 0.5

    vol_expanding = await predictor._predict_volatility(symbol, X)
    return {"confidence": float(confidence),
            "volatility_expanding": vol_expanding}


async def incremental_retrain(predictor, symbol: str, interval: str = "1h") -> dict:
    """Incrementally update the LightGBM direction model.

    Uses LightGBM's ``init_model`` parameter to continue training
    from the existing booster rather than training from scratch.
    Falls back to full retrain if no model exists yet.

    This reduces retrain time from ~30s to ~2s and preserves knowledge
    accumulated over previous training cycles.
    """
    from core.ml.features import REQUIRED_INDICATORS

    df = await predictor.market_data.get_historical(symbol, interval, limit=500)
    if df is None or len(df) < 100:
        return {"error": f"Insufficient data: {len(df) if df is not None else 0} rows"}

    from core.strategy.indicators import compute_all
    df = compute_all(df, REQUIRED_INDICATORS)
    X = compute_features(df, predictor._feature_list)
    y = create_binary_label(df, forward_periods=4, threshold=0.005)

    common_idx = X.index.intersection(y.dropna().index)
    if len(common_idx) < 40:
        return {"error": f"Insufficient labelled: {len(common_idx)}"}
    X = X.loc[common_idx].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.loc[common_idx]

    model_key = f"{symbol}_binary"
    existing_model = predictor._models.get(model_key)

    if existing_model is not None:
        # Incremental: use init_model to continue training
        try:
            latest = X.iloc[-200:]  # only recent data for incremental
            y_latest = y.iloc[-200:]
            existing_model.fit(
                latest, y_latest,
                init_model=existing_model.booster_,
                keep_training_booster=True,
            )
            predictor._models[model_key] = existing_model
            return {"status": "incremental", "n_samples": len(latest)}
        except Exception as e:
            logger.warning(f"Incremental retrain failed ({e}), falling back to full")
            # Fall through to full retrain
            predictor._models.pop(model_key, None)

    # Full retrain (first time or fallback)
    result = predictor.trainer.train_binary(
        symbol, "periodic", X, y, engine="lightgbm")
    if "model_path" in result:
        model = predictor.trainer.load_model(result["model_path"])
        if model:
            predictor._models[model_key] = model
    return {**result, "status": "full"}


def load_patchtst_model(predictor, symbol: str, strategy_name: str):
    """Load a pre-trained PatchTST model for ensemble participation."""
    try:
        from core.ml.patchtst_trainer import PatchTSTTrainer
        if not hasattr(predictor, '_patch_trainer') or predictor._patch_trainer is None:
            predictor._patch_trainer = PatchTSTTrainer(
                data_dir=str(predictor.config.data_dir),
                seq_len=100, patch_len=16, stride=8,
                d_model=128, num_heads=8, num_layers=3, dropout=0.15)
        if not hasattr(predictor, '_patch_models'):
            predictor._patch_models = {}
        model = predictor._patch_trainer.load(symbol, strategy_name)
        if model:
            predictor._patch_models[symbol] = model
    except Exception as e:
        logger.warning(f"PatchTST model not available: {e}")


async def _predict_patchtst(predictor, symbol: str, X: pd.DataFrame) -> float | None:
    """Predict with PatchTST model (used by ensemble)."""
    if not hasattr(predictor, '_patch_trainer') or not hasattr(predictor, '_patch_models'):
        return None
    model = predictor._patch_models.get(symbol)
    trainer = predictor._patch_trainer
    if model is None or trainer is None:
        return None
    try:
        result = trainer.predict(model, X)
        if result is None:
            return None
        direction = result["direction"]
        confidence = result["confidence"]
        if direction > 0:
            return confidence
        elif direction < 0:
            return 1.0 - confidence
        else:
            return 0.5
    except Exception:
        return None
