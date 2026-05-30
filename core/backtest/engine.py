"""Backtesting engine — synchronous replay with ML, signal fusion, and risk controls."""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from core.backtest.data_feeder import DataFeeder
from core.backtest.metrics import calculate_metrics
from core.strategy.indicators import compute_all, evaluate_condition


class BacktestEngine:
    """Synchronous backtesting engine with ML prediction and signal fusion."""

    def __init__(self, config, strategy_engine, risk_manager, order_executor):
        self.config = config
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.order_executor = order_executor

    def run(self, strategies: list[str], symbols: list[str],
            date_start: str, date_end: str, mode: str = "full",
            initial_balance: float = 10000.0) -> dict:
        """Alias for run_with_exit_evaluation."""
        return self.run_with_exit_evaluation(
            strategies, symbols, date_start, date_end, initial_balance, mode)

    def run_with_exit_evaluation(self, strategies, symbols, date_start, date_end,
                                  initial_balance=10000.0, mode="full"):
        """Full backtest with ML predictions, signal fusion, and risk controls.

        Walk-forward approach: at each timestamp, we only use data available up to
        that point (no look-ahead bias). ML models are retrained every 50 candles.
        """
        t0 = time.time()

        # Load strategy configs
        strategy_configs = []
        for name in strategies:
            try:
                s = self.strategy_engine.loader.load(name)
                strategy_configs.append(s)
            except Exception as e:
                return {"error": f"Strategy '{name}' not found: {e}"}

        # Determine required intervals
        intervals = list(set(tf for s in strategy_configs
                            for tf in s.timeframes)) or ["1h"]
        if "1h" not in intervals:
            intervals.append("1h")  # ML training uses 1h

        # Load historical data
        cache_dir = str(Path(self.config.data_dir) / "market")
        feeder = DataFeeder(cache_dir, symbols, intervals, date_start, date_end)
        feeder.load()

        if len(feeder) == 0:
            return {"error": "No historical data found for the given symbols and date range"}

        # ---- Backtest State ----
        balance = initial_balance
        positions: dict[str, dict] = {}
        trades: list[dict] = []
        equity_curve: list[dict] = []
        events: list[dict] = []

        # ML state — per-symbol models and predictions
        ml_models: dict[str, object] = {}  # symbol -> trained XGBoost model
        ml_predictions: dict[str, float] = {}  # symbol -> latest confidence (0-1)
        ml_correct = 0
        ml_total = 0
        ml_retrain_counter: dict[str, int] = {}  # symbol -> candles since last retrain
        ml_retrain_interval = 50  # retrain every 50 candles

        # Position sizing
        from core.risk.position_sizer import PositionSizer
        sizer = PositionSizer(
            self.config.hard_limits, self.config.soft_params,
            self.config.core_capital_pct, self.config.satellite_capital_pct)

        # Signal weights
        w = self.config.signal_weights
        w_ind = w.indicator
        w_ml = w.ml

        # Entry threshold
        ENTRY_THRESHOLD = 0.5

        # Max positions
        max_positions = self.config.hard_limits.max_open_trades

        pos_counter = 0  # unique position ID

        # ---- Main Loop ----
        for slice_data in feeder:
            ts = slice_data["timestamp"]

            # --- ML PREDICTION (walk-forward) ---
            for sym in symbols:
                df_1h = feeder.get_all_data_for_symbol(sym, "1h")
                if len(df_1h) < 100:
                    continue

                # Retrain periodically
                key = sym
                counter = ml_retrain_counter.get(key, 0)
                if counter >= ml_retrain_interval or key not in ml_models:
                    model = self._train_ml_model(df_1h[df_1h.index <= ts])
                    if model is not None:
                        ml_models[key] = model
                    ml_retrain_counter[key] = 0
                ml_retrain_counter[key] = counter + 1

                # Predict using latest trained model
                model = ml_models.get(key)
                if model is not None:
                    try:
                        conf = self._predict_ml(model, df_1h[df_1h.index <= ts])
                        if conf is not None:
                            ml_predictions[sym] = conf

                            # Track accuracy: compare prediction to actual future movement
                            future_df = df_1h[df_1h.index > ts]
                            if len(future_df) >= 1:
                                actual_close = float(future_df.iloc[0]["close"])
                                current_close = float(df_1h[df_1h.index <= ts].iloc[-1]["close"])
                                actual_direction = "up" if actual_close > current_close else "down"
                                pred_direction = "up" if conf >= 0.5 else "down"
                                ml_total += 1
                                if actual_direction == pred_direction:
                                    ml_correct += 1
                    except Exception:
                        pass

            # --- CHECK EXITS ---
            for sym in list(positions.keys()):
                pos = positions[sym]
                for strategy in strategy_configs:
                    if strategy.name != pos.get("strategy_name"):
                        continue
                    for interval in strategy.timeframes:
                        df = feeder.get_all_data_for_symbol(sym, interval)
                        if len(df) < 50:
                            continue
                        df = df[df.index <= ts].copy()
                        if len(df) < 20:
                            continue
                        try:
                            df = compute_all(df, strategy.indicators)
                        except Exception:
                            continue
                        if len(df) == 0:
                            continue

                        exit_conds = strategy.exit_conditions.get(pos["side"], [])
                        for cond in exit_conds:
                            mask = evaluate_condition(df, cond)
                            if hasattr(mask, 'iloc') and mask.iloc[-1]:
                                exit_price = float(df["close"].iloc[-1])
                                entry_price = pos["entry_price"]
                                qty = pos["quantity"]
                                amount = pos["amount_usdt"]
                                if pos["side"] == "long":
                                    pnl = (exit_price - entry_price) * qty
                                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                                else:
                                    pnl = (entry_price - exit_price) * qty
                                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                                trades.append({
                                    "symbol": sym, "side": pos["side"],
                                    "entry_price": round(entry_price, 4),
                                    "exit_price": round(exit_price, 4),
                                    "quantity": round(qty, 6),
                                    "pnl": round(pnl, 2),
                                    "pnl_pct": round(pnl_pct, 2),
                                    "strategy": strategy.name,
                                    "opened_at": str(pos.get("opened_at", ts)),
                                    "closed_at": str(ts),
                                    "amount_usdt": round(amount, 2),
                                })
                                balance += amount + pnl
                                events.append({
                                    "time": str(ts), "type": "exit",
                                    "symbol": sym, "price": exit_price,
                                    "pnl": round(pnl, 2),
                                    "strategy": strategy.name,
                                })
                                del positions[sym]
                                break
                        if sym not in positions:
                            break
                    if sym not in positions:
                        break

            # --- CHECK ENTRIES ---
            for strategy in strategy_configs:
                for sym in symbols:
                    if sym in positions:
                        continue
                    if len(positions) >= max_positions:
                        break

                    for interval in strategy.timeframes:
                        df = feeder.get_all_data_for_symbol(sym, interval)
                        if len(df) < 50:
                            continue
                        df = df[df.index <= ts].copy()
                        if len(df) < 20:
                            continue
                        try:
                            df = compute_all(df, strategy.indicators)
                        except Exception:
                            continue
                        if len(df) == 0:
                            continue

                        # Evaluate entry conditions (indicator signal)
                        long_active = False
                        short_active = False
                        for side in ["long", "short"]:
                            for cond in strategy.entry_conditions.get(side, []):
                                mask = evaluate_condition(df, cond)
                                met = bool(hasattr(mask, 'iloc') and mask.iloc[-1])
                                if met and side == "long":
                                    long_active = True
                                elif met and side == "short":
                                    short_active = True

                        if long_active and short_active:
                            continue
                        indicator_signal = 1.0 if long_active else -1.0 if short_active else 0.0
                        if indicator_signal == 0.0:
                            continue

                        # ---- SIGNAL FUSION ----
                        ml_conf = ml_predictions.get(sym, 0.5)
                        ml_directional = (ml_conf - 0.5) * 2  # -1 to +1

                        # Per-strategy ML weight override
                        strategy_ml_weight = w_ml
                        if strategy.ml_config and strategy.ml_config.enabled:
                            strategy_ml_weight = strategy.ml_config.weight

                        total_weight = w_ind + strategy_ml_weight
                        if total_weight > 0:
                            final_score = (indicator_signal * w_ind +
                                           ml_directional * strategy_ml_weight) / total_weight
                        else:
                            final_score = indicator_signal

                        if abs(final_score) < ENTRY_THRESHOLD:
                            continue

                        side = "long" if final_score > 0 else "short"
                        price = float(df["close"].iloc[-1])

                        # ---- POSITION SIZING ----
                        qty, risk_amount = sizer.calculate_position_size(
                            balance, price, "satellite")
                        if qty <= 0:
                            continue

                        # Check max position size
                        max_amount = balance * (self.config.hard_limits.max_position_size_pct / 100)
                        if risk_amount > max_amount:
                            risk_amount = max_amount
                            qty = risk_amount / price

                        amount_usdt = qty * price
                        if amount_usdt > balance * 0.95:
                            continue  # don't use >95% of balance

                        pos_counter += 1
                        trade_group = f"bt_{pos_counter}_{int(ts.timestamp())}"
                        balance -= amount_usdt

                        positions[sym] = {
                            "symbol": sym, "side": side,
                            "quantity": qty, "entry_price": price,
                            "amount_usdt": amount_usdt,
                            "strategy_name": strategy.name,
                            "opened_at": str(ts), "trade_group": trade_group,
                        }
                        events.append({
                            "time": str(ts), "type": "entry",
                            "symbol": sym, "side": side, "price": price,
                            "qty": round(qty, 6), "amount_usdt": round(amount_usdt, 2),
                            "strategy": strategy.name,
                            "signal_score": round(final_score, 3),
                            "ml_confidence": round(ml_conf, 3),
                        })
                        break  # one entry per symbol per timestamp

            # --- EQUITY CURVE ---
            invested = sum(p.get("amount_usdt", 0) for p in positions.values())
            equity_curve.append({
                "time": str(ts), "equity": round(balance + invested, 2),
                "balance": round(balance, 2), "invested": round(invested, 2),
            })

        final_balance = balance
        metrics = calculate_metrics(trades, equity_curve, initial_balance, final_balance)
        metrics["ml_accuracy_pct"] = round(
            ml_correct / ml_total * 100 if ml_total > 0 else 0, 1)
        metrics["runtime_seconds"] = round(time.time() - t0, 1)

        return {
            "trades": trades, "equity_curve": equity_curve, "events": events,
            "metrics": metrics, "final_balance": round(final_balance, 2),
            "initial_balance": initial_balance,
            "strategies": strategies, "symbols": symbols,
            "date_start": date_start, "date_end": date_end, "mode": mode,
        }

    # ---- ML Helpers ----

    # Indicator config used for ML feature computation (same as live system)
    _ML_INDICATORS = {
        "rsi": {"period": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bollinger": {"period": 20, "stddev": 2},
    }
    _ML_FEATURE_NAMES = ["rsi", "macd_histogram", "bollinger_width",
                         "volume_ratio", "price_momentum_24h"]

    def _compute_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicators then extract ML feature columns."""
        from core.strategy.indicators import compute_all
        from core.ml.features import build_features

        df_ind = compute_all(df.copy(), self._ML_INDICATORS)
        feature_df = build_features(df_ind, self._ML_FEATURE_NAMES)
        # Add returns and volatility as basic features
        if len(df_ind) > 5:
            feature_df["returns_1"] = df_ind["close"].pct_change()
            feature_df["returns_5"] = df_ind["close"].pct_change(5)
            feature_df["volatility_20"] = feature_df["returns_1"].rolling(20).std()
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.ffill().fillna(0)
        return feature_df

    def _train_ml_model(self, df: pd.DataFrame):
        """Train an XGBoost binary classifier on indicator features. Returns the model or None."""
        if len(df) < 100:
            return None
        try:
            from core.ml.features import create_binary_label
            import xgboost as xgb

            feature_df = self._compute_ml_features(df)
            labels = create_binary_label(df)

            common_idx = feature_df.index.intersection(labels.dropna().index)
            if len(common_idx) < 50:
                return None
            X = feature_df.loc[common_idx].values.astype(float)
            y = labels.loc[common_idx].values

            model = xgb.XGBClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', verbosity=0, random_state=42)
            model.fit(X, y)
            return model
        except Exception as e:
            logger.debug(f"ML train failed: {e}")
            return None

    def _predict_ml(self, model, df: pd.DataFrame) -> float | None:
        """Predict probability(price up) using the trained model. Returns confidence 0-1 or None."""
        if len(df) < 50:
            return None
        try:
            feature_df = self._compute_ml_features(df)
            if len(feature_df) == 0:
                return None
            X = feature_df.iloc[-1:].values.astype(float)
            proba = model.predict_proba(X)
            if proba.shape[1] >= 2:
                return float(proba[0][1])  # P(up)
            return 0.5
        except Exception as e:
            logger.debug(f"ML predict failed: {e}")
            return None
