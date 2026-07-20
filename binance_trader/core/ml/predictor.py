"""ML predictor — LightGBM + TFT with auto-selection.

LightGBM (tree-based): fast, stable baseline for single-step prediction.
TFT (transformer): sequence-aware, outputs direction + uncertainty.
Set model_type='tft' in config or per-strategy ml_config to use TFT.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.provider import MarketDataProvider
from core.ml.trainer import MLTrainer
from core.ml.features import (
    compute_features, create_binary_label, create_volatility_label,
    DEFAULT_FEATURES, REQUIRED_INDICATORS,
)


class MLPredictor:
    """Real-time ML predictor supporting LightGBM and TFT models.

    Subscribes to MARKET_KLINE events on 1h/4h timeframes and publishes
    ML_PREDICTION events with directional confidence scores.
    """

    def __init__(self, config: Config, event_bus: EventBus,
                 market_data: MarketDataProvider):
        self.config = config
        self.event_bus = event_bus
        self.market_data = market_data
        self.trainer = MLTrainer(config.data_dir)
        self._running = False
        self._models: dict[str, object] = {}     # "symbol_binary" → direction model
        self._vol_models: dict[str, object] = {} # "symbol_vol" → volatility model (NEW)
        self._tft_models: dict[str, object] = {} # "symbol" → TFT model
        self._tft_trainer = None  # Lazy init
        self._task: asyncio.Task | None = None
        self._feature_list: list[str] = list(DEFAULT_FEATURES)
        self._model_type: str = getattr(config, 'ml_model_type', 'lightgbm')

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self.event_bus.subscribe(EventType.MARKET_KLINE, self._on_kline)
        self._task = asyncio.create_task(self._retrain_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        self.event_bus.unsubscribe(EventType.MARKET_KLINE, self._on_kline)

    # ── TFT lazy init ────────────────────────────────────────────────

    def _get_tft_trainer(self):
        if self._tft_trainer is None:
            from core.ml.tft_trainer import TFTTrainer
            self._tft_trainer = TFTTrainer(
                data_dir=str(self.config.data_dir),
                seq_len=100, d_model=64, num_heads=4,
                lstm_layers=2, dropout=0.2)
        return self._tft_trainer

    # ── Retrain loop ─────────────────────────────────────────────────

    async def _retrain_loop(self):
        from loguru import logger
        await asyncio.sleep(300)
        while self._running:
            for symbol in self.market_data.watched_symbols:
                try:
                    # Direction model (existing)
                    if self._model_type == 'tft':
                        await self.train_tft_model(symbol, "periodic", "1h")
                    else:
                        await self.train_model(symbol, "periodic", "1h")
                    # Volatility model (NEW)
                    await self.train_volatility_model(symbol, "volatility", "1h")
                except Exception as e:
                    logger.warning(f"ML retrain failed for {symbol}: {e}")
            await asyncio.sleep(86400)

    # ── Event handlers ───────────────────────────────────────────────

    async def _on_kline(self, event: Event):
        if not self._running:
            return
        symbol = event.data["symbol"]
        interval = event.data["interval"]

        if interval not in ["1h", "4h"]:
            return

        df = await self.market_data.get_historical(symbol, interval)
        if df is None or len(df) < 100:
            return

        from core.strategy.indicators import compute_all
        df = compute_all(df, REQUIRED_INDICATORS)
        X = compute_features(df, self._feature_list)

        if self._model_type == 'tft':
            confidence = await self._predict_tft(symbol, X)
        else:
            confidence = await self._predict_lgb(symbol, X)

        # ── Volatility prediction (NEW — independent model) ──
        vol_expanding = await self._predict_volatility(symbol, X)

        await self.event_bus.publish(Event(EventType.ML_PREDICTION, {
            "symbol": symbol,
            "interval": interval,
            "confidence": round(float(confidence), 4),
            "volatility_expanding": vol_expanding,
        }))

    # ── LightGBM prediction ──────────────────────────────────────────

    async def _predict_lgb(self, symbol: str, X: pd.DataFrame) -> float:
        model_key = f"{symbol}_binary"
        if model_key in self._models:
            try:
                latest = X.iloc[-1:].fillna(0)
                proba = self._models[model_key].predict_proba(latest)[0]
                return proba[1] if len(proba) > 1 else 0.5
            except Exception:
                return 0.5
        return 0.5

    # ── TFT prediction ───────────────────────────────────────────────

    async def _predict_tft(self, symbol: str, X: pd.DataFrame) -> float:
        """TFT prediction: returns confidence ∈ [0, 1] from direction + uncertainty."""
        model = self._tft_models.get(symbol)
        if model is None:
            return 0.5

        tft = self._get_tft_trainer()
        result = tft.predict(model, X, feature_cols=self._feature_list)
        if result is None:
            return 0.5

        # TFT outputs direction and confidence
        # Map to a single confidence score for signal fusion
        tft_conf = result["confidence"]
        tft_dir = result["direction"]

        # Convert to directional confidence (like LightGBM's proba[1])
        # TFT confidence × direction → signal-compatible format
        if tft_dir > 0:
            return tft_conf  # bullish confidence
        else:
            return 1.0 - tft_conf  # bearish → low value

    # ── Volatility prediction (NEW) ─────────────────────────────────────

    async def _predict_volatility(self, symbol: str, X: pd.DataFrame) -> bool:
        """Predict whether volatility will expand (True) or contract (False).

        Uses a separately-trained volatility model.  Falls back to
        a naive heuristic (recent volatility trend) if no model is loaded.
        """
        vol_key = f"{symbol}_vol"
        if vol_key in self._vol_models:
            try:
                latest = X.iloc[-1:].fillna(0)
                pred = int(self._vol_models[vol_key].predict(latest)[0])
                return bool(pred)
            except Exception:
                pass
        # Fallback: compare recent vol to longer-term vol
        try:
            ret = X.get("ret_1", pd.Series(dtype=float))
            if len(ret) < 21:
                return False
            recent_vol = float(ret.iloc[-10:].std())
            hist_vol = float(ret.iloc[-21:].std())
            return recent_vol > hist_vol
        except Exception:
            return False

    async def train_volatility_model(self, symbol: str,
                                     strategy_name: str = "volatility",
                                     interval: str = "1h") -> dict:
        """Train a volatility expansion classifier (LightGBM binary)."""
        df = await self.market_data.get_historical(symbol, interval, limit=2000)
        if df is None or len(df) < 100:
            return {"error": f"Insufficient data for vol model: {len(df) if df is not None else 0} rows"}

        from core.strategy.indicators import compute_all
        df = compute_all(df, REQUIRED_INDICATORS)
        X = compute_features(df, self._feature_list)
        y = create_volatility_label(df, forward_periods=20)

        common_idx = X.index.intersection(y.dropna().index)
        if len(common_idx) < 50:
            return {"error": f"Insufficient labelled data: {len(common_idx)} rows"}
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        result = self.trainer.train_binary(
            symbol, f"{strategy_name}_vol", X, y, engine="lightgbm")

        if "model_path" in result:
            model = self.trainer.load_model(result["model_path"])
            if model:
                self._vol_models[f"{symbol}_vol"] = model
        return result

    # ── LightGBM training ────────────────────────────────────────────

    async def train_model(self, symbol: str, strategy_name: str,
                          interval: str = "1h") -> dict:
        df = await self.market_data.get_historical(symbol, interval, limit=2000)
        if df is None or len(df) < 60:
            return {"error": f"Insufficient data: {len(df) if df is not None else 0} rows"}

        from core.strategy.indicators import compute_all
        df = compute_all(df, REQUIRED_INDICATORS)
        X = compute_features(df, self._feature_list)
        y = create_binary_label(df, forward_periods=4, threshold=0.005)

        self.trainer.save_training_data(symbol, strategy_name, X, y)
        result = self.trainer.train_binary(symbol, strategy_name, X, y, engine="lightgbm")

        if "model_path" in result:
            model = self.trainer.load_model(result["model_path"])
            if model:
                self._models[f"{symbol}_binary"] = model
        return result

    # ── TFT training ─────────────────────────────────────────────────

    async def train_tft_model(self, symbol: str, strategy_name: str,
                              interval: str = "1h") -> dict:
        df = await self.market_data.get_historical(symbol, interval, limit=2000)
        if df is None or len(df) < 120:
            return {"error": f"Insufficient data: {len(df) if df is not None else 0} rows"}

        from core.strategy.indicators import compute_all
        from core.ml.features import create_regression_label

        df = compute_all(df, REQUIRED_INDICATORS)
        X = compute_features(df, self._feature_list)

        # TFT uses regression labels (forward return %)
        y = create_regression_label(df, forward_periods=4)
        X["label"] = y  # use pandas index alignment (safer than .values)

        tft = self._get_tft_trainer()
        model, metrics = tft.train(
            X, feature_cols=self._feature_list,
            label_col="label", epochs=40, batch_size=32,
            learning_rate=1e-3, patience=8)

        if model is not None:
            tft.save(model, symbol, strategy_name)
            self._tft_models[symbol] = model

        return metrics

    # ── Common helpers ───────────────────────────────────────────────

    async def predict(self, symbol: str, df: pd.DataFrame,
                      features: list[str] | None = None) -> float:
        flist = features or self._feature_list
        X = compute_features(df, flist)

        if self._model_type == 'tft':
            return await self._predict_tft(symbol, X)
        return await self._predict_lgb(symbol, X)

    def load_model(self, symbol: str, model_path: str):
        model = self.trainer.load_model(model_path)
        if model:
            self._models[f"{symbol}_binary"] = model

    def load_tft_model(self, symbol: str, strategy_name: str):
        tft = self._get_tft_trainer()
        model = tft.load(symbol, strategy_name)
        if model:
            self._tft_models[symbol] = model

    # ── Ensemble prediction (Phase 4d) ─────────────────────────────────

    async def ensemble_predict(self, symbol: str, X: pd.DataFrame) -> dict:
        """Weighted ensemble across all available model architectures.

        Combines LightGBM, TFT, and PatchTST predictions.  Weights are
        fixed for now (can be evolved by future GA work).

        Returns:
            dict with ``confidence`` (float [0,1]) and ``volatility_expanding`` (bool).
            Falls back to individual models when only one is available.
        """
        predictions: dict[str, float | None] = {}

        # LightGBM direction
        if f"{symbol}_binary" in self._models:
            predictions["lgb"] = await self._predict_lgb(symbol, X)

        # TFT
        if symbol in self._tft_models:
            predictions["tft"] = await self._predict_tft(symbol, X)

        # PatchTST (lazy init)
        if hasattr(self, '_patch_models') and symbol in self._patch_models:
            predictions["patch"] = await self._predict_patchtst(symbol, X)

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

        vol_expanding = await self._predict_volatility(symbol, X)
        return {"confidence": float(confidence),
                "volatility_expanding": vol_expanding}

    # ── Online / incremental learning (Phase 4d) ───────────────────────

    async def incremental_retrain(self, symbol: str, interval: str = "1h") -> dict:
        """Incrementally update the LightGBM direction model.

        Uses LightGBM's ``init_model`` parameter to continue training
        from the existing booster rather than training from scratch.
        Falls back to full retrain if no model exists yet.

        This reduces retrain time from ~30s to ~2s and preserves knowledge
        accumulated over previous training cycles.
        """
        df = await self.market_data.get_historical(symbol, interval, limit=500)
        if df is None or len(df) < 100:
            return {"error": f"Insufficient data: {len(df) if df is not None else 0} rows"}

        from core.strategy.indicators import compute_all
        df = compute_all(df, REQUIRED_INDICATORS)
        X = compute_features(df, self._feature_list)
        y = create_binary_label(df, forward_periods=4, threshold=0.005)

        common_idx = X.index.intersection(y.dropna().index)
        if len(common_idx) < 40:
            return {"error": f"Insufficient labelled: {len(common_idx)}"}
        X = X.loc[common_idx].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.loc[common_idx]

        model_key = f"{symbol}_binary"
        existing_model = self._models.get(model_key)

        if existing_model is not None:
            # Incremental: use init_model to continue training
            try:
                import lightgbm as lgb
                latest = X.iloc[-200:]  # only recent data for incremental
                y_latest = y.iloc[-200:]
                existing_model.fit(
                    latest, y_latest,
                    init_model=existing_model.booster_,
                    keep_training_booster=True,
                )
                self._models[model_key] = existing_model
                return {"status": "incremental", "n_samples": len(latest)}
            except Exception as e:
                logger.warning(f"Incremental retrain failed ({e}), falling back to full")
                # Fall through to full retrain
                self._models.pop(model_key, None)

        # Full retrain (first time or fallback)
        result = self.trainer.train_binary(symbol, "periodic", X, y, engine="lightgbm")
        if "model_path" in result:
            model = self.trainer.load_model(result["model_path"])
            if model:
                self._models[model_key] = model
        return {**result, "status": "full"}

    # ── PatchTST support (Phase 4d, lightweight integration) ───────────

    def load_patchtst_model(self, symbol: str, strategy_name: str):
        """Load a pre-trained PatchTST model for ensemble participation."""
        try:
            from core.ml.patchtst_trainer import PatchTSTTrainer
            if not hasattr(self, '_patch_trainer') or self._patch_trainer is None:
                self._patch_trainer = PatchTSTTrainer(
                    data_dir=str(self.config.data_dir),
                    seq_len=100, patch_len=16, stride=8,
                    d_model=128, num_heads=8, num_layers=3, dropout=0.15)
            if not hasattr(self, '_patch_models'):
                self._patch_models = {}
            model = self._patch_trainer.load(symbol, strategy_name)
            if model:
                self._patch_models[symbol] = model
        except Exception as e:
            logger.warning(f"PatchTST model not available: {e}")

    async def _predict_patchtst(self, symbol: str, X: pd.DataFrame) -> float | None:
        """Predict with PatchTST model (used by ensemble)."""
        if not hasattr(self, '_patch_trainer') or not hasattr(self, '_patch_models'):
            return None
        model = self._patch_models.get(symbol)
        trainer = self._patch_trainer
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

    @property
    def feature_count(self) -> int:
        return len(self._feature_list)

    @property
    def model_type(self) -> str:
        return self._model_type
