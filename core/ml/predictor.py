"""ML predictor — real-time prediction with LightGBM + expanded features."""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.provider import MarketDataProvider
from core.ml.trainer import MLTrainer
from core.ml.features import (
    compute_features, create_binary_label,
    DEFAULT_FEATURES, REQUIRED_INDICATORS,
)


class MLPredictor:
    """Real-time ML predictor using LightGBM with 30-dim features.

    Subscribes to MARKET_KLINE events on 1h/4h timeframes, computes the
    full 30-dimension feature set, and publishes ML_PREDICTION events
    with directional confidence scores.
    """

    def __init__(self, config: Config, event_bus: EventBus,
                 market_data: MarketDataProvider):
        self.config = config
        self.event_bus = event_bus
        self.market_data = market_data
        self.trainer = MLTrainer(config.data_dir)
        self._running = False
        self._models: dict[str, object] = {}
        self._task: asyncio.Task | None = None
        self._feature_list: list[str] = list(DEFAULT_FEATURES)

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

    # ── Retrain loop ─────────────────────────────────────────────────

    async def _retrain_loop(self):
        """Daily model retraining for all watched symbols."""
        from loguru import logger
        await asyncio.sleep(300)  # Wait 5 min for initial data
        while self._running:
            for symbol in self.market_data._watched_symbols:
                try:
                    await self.train_model(symbol, "periodic", "1h")
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

        # Compute indicators then full feature set
        from core.strategy.indicators import compute_all
        df = compute_all(df, REQUIRED_INDICATORS)
        X = compute_features(df, self._feature_list)
        latest = X.iloc[-1:].fillna(0)

        model_key = f"{symbol}_binary"
        if model_key in self._models:
            try:
                proba = self._models[model_key].predict_proba(latest)[0]
                confidence = proba[1] if len(proba) > 1 else 0.5
            except Exception:
                confidence = 0.5
        else:
            # Fallback: use recent trend as weak signal
            direction = np.sign(
                df["close"].pct_change().rolling(50).mean().iloc[-1] or 0)
            confidence = 0.5 + direction * 0.1

        await self.event_bus.publish(Event(EventType.ML_PREDICTION, {
            "symbol": symbol,
            "interval": interval,
            "confidence": round(float(confidence), 4),
        }))

    # ── Model training ───────────────────────────────────────────────

    async def train_model(self, symbol: str, strategy_name: str,
                          interval: str = "1h") -> dict:
        """Train a LightGBM binary classifier for *symbol*.

        Returns a dict with accuracy, f1_score, model_path, and
        feature_importance.
        """
        df = await self.market_data.get_historical(symbol, interval, limit=2000)
        if df is None or len(df) < 60:
            return {"error": f"Insufficient data for {symbol}: "
                    f"{len(df) if df is not None else 0} rows"}

        from core.strategy.indicators import compute_all

        df = compute_all(df, REQUIRED_INDICATORS)
        X = compute_features(df, self._feature_list)
        y = create_binary_label(df, forward_periods=4, threshold=0.005)

        self.trainer.save_training_data(symbol, strategy_name, X, y)
        result = self.trainer.train_binary(
            symbol, strategy_name, X, y, engine="lightgbm")

        if "model_path" in result:
            model = self.trainer.load_model(result["model_path"])
            if model:
                self._models[f"{symbol}_binary"] = model

        return result

    # ── Prediction helpers ───────────────────────────────────────────

    async def predict(self, symbol: str, df: pd.DataFrame,
                      features: list[str] | None = None) -> float:
        """Return confidence score in [0,1] for *symbol*."""
        model_key = f"{symbol}_binary"
        if model_key not in self._models:
            return 0.5
        flist = features or self._feature_list
        X = compute_features(df, flist)
        latest = X.iloc[-1:].fillna(0)
        try:
            proba = self._models[model_key].predict_proba(latest)[0]
            return float(proba[1] if len(proba) > 1 else 0.5)
        except Exception:
            return 0.5

    def load_model(self, symbol: str, model_path: str):
        model = self.trainer.load_model(model_path)
        if model:
            self._models[f"{symbol}_binary"] = model

    @property
    def feature_count(self) -> int:
        return len(self._feature_list)
