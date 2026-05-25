import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.provider import MarketDataProvider
from core.ml.trainer import MLTrainer
from core.ml.features import build_features, create_binary_label


class MLPredictor:
    def __init__(self, config: Config, event_bus: EventBus, market_data: MarketDataProvider):
        self.config = config
        self.event_bus = event_bus
        self.market_data = market_data
        self.trainer = MLTrainer(config.data_dir)
        self._running = False
        self._models: dict[str, object] = {}
        self._task: asyncio.Task | None = None

    async def start(self):
        self._running = True
        self.event_bus.subscribe(EventType.MARKET_KLINE, self._on_kline)
        self._task = asyncio.create_task(self._retrain_loop())

    async def _retrain_loop(self):
        """Periodically retrain models (daily)."""
        from loguru import logger
        await asyncio.sleep(300)  # Wait 5 min for initial data
        while self._running:
            for symbol in self.market_data._watched_symbols:
                try:
                    await self.train_model(symbol, "periodic", "1h")
                except Exception as e:
                    logger.warning(f"ML retrain failed for {symbol}: {e}")
            await asyncio.sleep(86400)

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

        # Compute technical indicators before building features
        from core.strategy.indicators import compute_all
        indicator_configs = {
            "rsi": {"period": 14, "source": "close"},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"period": 20, "stddev": 2},
        }
        df = compute_all(df, indicator_configs)

        features = ["rsi", "macd_histogram", "bollinger_width", "volume_ratio", "price_momentum_24h"]
        available = [f for f in features if f in df.columns]

        if len(available) < 3:
            return

        X = build_features(df, available)
        latest = X.iloc[-1:].fillna(0)

        model_key = f"{symbol}_binary"
        if model_key in self._models:
            try:
                proba = self._models[model_key].predict_proba(latest)[0]
                confidence = proba[1] if len(proba) > 1 else 0.5
            except Exception:
                confidence = 0.5
        else:
            direction = np.sign(df["close"].pct_change().rolling(50).mean().iloc[-1] or 0)
            confidence = 0.5 + direction * 0.1

        await self.event_bus.publish(Event(EventType.ML_PREDICTION, {
            "symbol": symbol,
            "interval": interval,
            "confidence": round(float(confidence), 4),
        }))

    async def train_model(self, symbol: str, strategy_name: str, interval: str = "1h") -> dict:
        df = await self.market_data.get_historical(symbol, interval, limit=1000)
        if df is None or len(df) < 60:
            return {"error": f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows"}

        from core.strategy.indicators import compute_all

        indicator_configs = {
            "rsi": {"period": 14, "source": "close"},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"period": 20, "stddev": 2},
        }
        df = compute_all(df, indicator_configs)

        features = ["rsi", "macd_histogram", "bollinger_width", "volume_ratio", "price_momentum_24h"]
        X = build_features(df, features)
        y = create_binary_label(df, forward_periods=4)

        self.trainer.save_training_data(symbol, strategy_name, X, y)
        result = self.trainer.train_binary(symbol, strategy_name, X, y)

        if "model_path" in result:
            model = self.trainer.load_model(result["model_path"])
            if model:
                self._models[f"{symbol}_binary"] = model

        return result

    async def predict(self, symbol: str, df: pd.DataFrame, features: list[str]) -> float:
        model_key = f"{symbol}_binary"
        if model_key not in self._models:
            return 0.5
        X = build_features(df, features)
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

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        self.event_bus.unsubscribe(EventType.MARKET_KLINE, self._on_kline)
