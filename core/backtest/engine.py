"""Backtesting engine — synchronous replay with ML, signal fusion, and risk controls."""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from core.backtest.data_feeder import DataFeeder
from core.backtest.metrics import calculate_metrics
from core.backtest.engine_hybrid import run_hybrid
from core.strategy.indicators import compute_all, evaluate_condition

# Timeframe → minutes mapping for sorting and trend-filter logic
_TIMEFRAME_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
    "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080,
}

def _tf_minutes(tf: str) -> int:
    """Convert a timeframe string to minutes for comparison/sorting."""
    return _TIMEFRAME_MINUTES.get(tf, 60)

def _check_higher_tf_trend(df: pd.DataFrame, entry_side: str) -> float:
    """Return a confidence multiplier (0.0–1.0) based on higher-TF trend alignment.

    Instead of a hard block, this penalises counter-trend entries:
    - 1.0 = strongly aligned (boost confidence)
    - 0.6 = weakly counter-trend (reduced but not blocked)
    - 0.0 = extreme counter-trend (should not enter)

    Formula: compare close to EMA(50). The farther the price is against the
    trend direction, the lower the multiplier.
    """
    if len(df) < 50:
        return 1.0  # not enough data — no penalty
    close = df["close"].values
    ema50 = float(pd.Series(close).ewm(span=50, adjust=False).mean().iloc[-1])
    last_close = float(close[-1])
    if ema50 <= 0:
        return 1.0

    # deviation = how far price is from EMA, as a fraction
    deviation = (last_close - ema50) / ema50  # positive = above EMA, negative = below

    if entry_side == "long":
        if deviation >= 0:
            return 1.0                          # price above EMA — aligned
        elif deviation > -0.02:                  # within 2% below EMA
            return 0.6                           # mild penalty
        else:
            return 0.0                           # extreme counter-trend
    else:  # short
        if deviation <= 0:
            return 1.0                          # price below EMA — aligned
        elif deviation < 0.02:                   # within 2% above EMA
            return 0.6                           # mild penalty
        else:
            return 0.0                           # extreme counter-trend


class BacktestEngine:
    """Synchronous backtesting engine with ML prediction and signal fusion."""

    def __init__(self, config, strategy_engine, risk_manager, order_executor):
        self.config = config
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.order_executor = order_executor

    def _select_engine(self, strategies, engine_mode: str) -> str:
        """Determine which engine to use: 'hybrid' or 'legacy'."""
        if engine_mode == "legacy":
            return "legacy"

        # Check if any strategy has ML enabled
        strategy_configs = []
        if isinstance(strategies, list) and strategies and not isinstance(strategies[0], str):
            strategy_configs = strategies
        else:
            for name in strategies:
                try:
                    s = self.strategy_engine.loader.load(name)
                    strategy_configs.append(s)
                except Exception:
                    pass

        has_ml = any(
            s.ml_config and s.ml_config.enabled
            for s in strategy_configs
        )

        if engine_mode == "hybrid":
            if has_ml:
                raise ValueError(
                    "Hybrid engine does not support ML training/prediction. "
                    "Set config backtest.ml_enabled=false or use engine_mode='legacy'.")
            return "hybrid"

        # engine_mode == "auto"
        n = len(strategies) if isinstance(strategies, list) else 1
        if n >= 3 and not has_ml:
            return "hybrid"
        return "legacy"

    def run(self, strategies: list[str], symbols: list[str],
            date_start: str, date_end: str,
            initial_balance: float = 10000.0, mode: str = "full",
            progress_callback=None,
            strategy_symbols: dict[str, list[str]] = None,
            simulate_ai_weights: bool = True) -> dict:
        """Alias for run_with_exit_evaluation (parameter order matches)."""
        return self.run_with_exit_evaluation(
            strategies, symbols, date_start, date_end,
            initial_balance, mode,
            progress_callback=progress_callback,
            strategy_symbols=strategy_symbols,
            simulate_ai_weights=simulate_ai_weights)

    def run_with_exit_evaluation(self, strategies, symbols, date_start, date_end,
                                  initial_balance=10000.0, mode="full",
                                  progress_callback=None,
                                  strategy_symbols: dict[str, list[str]] = None,
                                  simulate_ai_weights: bool = True,
                                  ml_engine: str = "lightgbm",
                                  skip_ml_training: bool = False,
                                  per_strategy_isolation: bool = False):
        """Full backtest with ML predictions, signal fusion, and risk controls.

        Args:
            per_strategy_isolation: If True, each strategy gets independent
                positions (no blocking). Position key = 'strategy|symbol'.
                Used for GA batch evaluation where multiple strategies
                run in a single data pass.
            simulate_ai_weights: If True, adjust signal weights based on detected
                market regime (mimicking what the live AI market assessment does).
            ml_engine: 'lightgbm' (tree), 'tft' (transformer), 'patchtst' (patch-transformer).
            skip_ml_training: If True, load pre-trained models from disk instead of
                training. Useful for repeat backtests over the same period.
        """
        t0 = time.time()

        # ── Engine mode selection ──
        _engine_mode = getattr(self.config, 'backtest_engine_mode', 'auto')
        _ml_enabled = getattr(self.config, 'backtest_ml_enabled', False)

        # Override ML based on config
        if not _ml_enabled and isinstance(strategies, list) and strategies:
            if not isinstance(strategies[0], str):
                for s in strategies:
                    if s.ml_config:
                        s.ml_config.enabled = False
            else:
                for name in strategies:
                    try:
                        s = self.strategy_engine.loader.load(name)
                        if s.ml_config:
                            s.ml_config.enabled = False
                    except Exception:
                        pass

        # Route to hybrid engine if applicable
        try:
            use_hybrid = self._select_engine(strategies, _engine_mode) == "hybrid"
        except ValueError:
            use_hybrid = False

        if use_hybrid:
            try:
                return run_hybrid(
                    strategies, symbols, date_start, date_end,
                    self.config, self.strategy_engine.loader,
                    initial_balance=initial_balance,
                    per_strategy_isolation=per_strategy_isolation,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                logger.warning(f"Hybrid engine failed ({e}), falling back to legacy")
                # Fall through to legacy engine below

        # Load strategy configs — support direct config objects for GA
        strategy_configs = []
        if isinstance(strategies, list) and strategies and not isinstance(strategies[0], str):
            # strategies is already a list of StrategyConfig objects
            strategy_configs = strategies
            strategies = [s.name for s in strategy_configs]
        else:
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

        # ML state — per-strategy×symbol models, each matched to the strategy's timeframe
        ml_models: dict[str, object] = {}  # "strategy_name|symbol" -> trained model
        ml_predictions: dict[str, float] = {}  # "strategy_name|symbol" -> latest confidence
        ml_correct = 0
        ml_total = 0
        # Training cost per retrain: LightGBM ~0.3s, PatchTST ~8s, TFT ~15s
        # Use longer intervals for expensive models to keep backtest time reasonable.
        if ml_engine == "tft":
            ml_retrain_interval = 800
        elif ml_engine == "patchtst":
            ml_retrain_interval = 500  # PatchTST is ~2x faster than TFT
        else:
            ml_retrain_interval = 100
        market_regime: dict[str, str] = {}

        # Per-timeframe ML parameters: forward_periods and threshold
        # Shorter TFs need more lookahead periods to capture a meaningful move
        _ML_TF_PARAMS = {
            "1m":  {"forward": 20, "threshold": 0.003, "min_candles": 300},
            "3m":  {"forward": 15, "threshold": 0.004, "min_candles": 200},
            "5m":  {"forward": 12, "threshold": 0.005, "min_candles": 200},
            "15m": {"forward": 8,  "threshold": 0.005, "min_candles": 150},
            "30m": {"forward": 6,  "threshold": 0.005, "min_candles": 120},
            "1h":  {"forward": 4,  "threshold": 0.005, "min_candles": 100},
            "2h":  {"forward": 4,  "threshold": 0.006, "min_candles": 80},
            "4h":  {"forward": 4,  "threshold": 0.008, "min_candles": 60},
            "6h":  {"forward": 4,  "threshold": 0.010, "min_candles": 50},
            "8h":  {"forward": 4,  "threshold": 0.012, "min_candles": 40},
            "12h": {"forward": 4,  "threshold": 0.015, "min_candles": 30},
            "1d":  {"forward": 4,  "threshold": 0.020, "min_candles": 25},
            "3d":  {"forward": 4,  "threshold": 0.030, "min_candles": 20},
            "1w":  {"forward": 4,  "threshold": 0.050, "min_candles": 15},
        }

        # Build round-robin model keys (staggered retraining — 1 model/step)
        _ml_keys: list[tuple[str, str, str, dict, list[str]]] = []  # (key, sym, tf, params, features)
        _ml_key_idx: dict[str, int] = {}  # key → index in _ml_keys
        for strategy in strategy_configs:
            if not (strategy.ml_config and strategy.ml_config.enabled):
                continue
            primary_tf = min(strategy.timeframes, key=_tf_minutes) if strategy.timeframes else "1h"
            tf_params = _ML_TF_PARAMS.get(primary_tf, _ML_TF_PARAMS["1h"])
            # Respect per-strategy feature selection: empty = all features
            ml_features = (strategy.ml_config.features
                          if (strategy.ml_config and strategy.ml_config.features)
                          else None)
            for sym in symbols:
                key = f"{strategy.name}|{sym}"
                _ml_key_idx[key] = len(_ml_keys)
                _ml_keys.append((key, sym, primary_tf, tf_params, ml_features))
        # Scale interval up if more models than steps in the interval
        _effective_interval = max(ml_retrain_interval, len(_ml_keys))
        _ml_retrain_stagger = max(1, _effective_interval // max(len(_ml_keys), 1))
        if _ml_keys:
            logger.info(f"ML round-robin: {len(_ml_keys)} models, "
                       f"retrain 1 every {_ml_retrain_stagger} steps "
                       f"(= each model every ~{_ml_retrain_stagger * len(_ml_keys)} steps)")

        # TFT state (only when ml_engine == 'tft')
        tft_trainer = None
        if ml_engine == "tft":
            try:
                from core.ml.tft_trainer import TFTTrainer as _TFTTrainer
                tft_trainer = _TFTTrainer(
                    data_dir=str(self.config.data_dir),
                    seq_len=100, d_model=96, num_heads=4,
                    lstm_layers=3, dropout=0.2)
            except Exception as e:
                logger.warning(f"TFT unavailable, falling back to LightGBM: {e}")
                ml_engine = "lightgbm"

        # PatchTST state (only when ml_engine == 'patchtst')
        patchtst_trainer = None
        if ml_engine == "patchtst":
            try:
                from core.ml.patchtst_trainer import PatchTSTTrainer as _PTTrainer
                patchtst_trainer = _PTTrainer(
                    data_dir=str(self.config.data_dir),
                    seq_len=100, patch_len=16, stride=8,
                    d_model=128, num_heads=8, num_layers=3, dropout=0.15)
            except Exception as e:
                logger.warning(f"PatchTST unavailable: {e}")
                ml_engine = "lightgbm"

        # ── Skip training: preload cached models from disk ──
        if skip_ml_training:
            from core.ml.trainer import MLTrainer as _MLTrainer
            _disk_trainer = _MLTrainer(str(self.config.data_dir))
            models_dir = Path(self.config.data_dir) / "models"
            for strategy in strategy_configs:
                if not (strategy.ml_config and strategy.ml_config.enabled):
                    continue
                for sym in symbols:
                    key = f"{strategy.name}|{sym}"
                    if ml_engine == "tft" and tft_trainer is not None:
                        model = tft_trainer.load(sym, strategy.name)
                        if model is not None:
                            ml_models[key] = model
                    elif ml_engine == "patchtst" and patchtst_trainer is not None:
                        model = patchtst_trainer.load(sym, strategy.name)
                        if model is not None:
                            ml_models[key] = model
                    else:
                        pkl_path = models_dir / f"{sym}_{strategy.name}_binary.pkl"
                        if pkl_path.exists():
                            model = _disk_trainer.load_model(str(pkl_path))
                            if model is not None:
                                ml_models[key] = model
            preloaded = len(ml_models)
            if preloaded > 0:
                logger.info(f"Preloaded {preloaded} cached ML models from disk")

        # ── Indicator cache: precompute each unique indicator config once ──
        # Key: (json_hash_of_indicators, symbol, interval) → full DataFrame
        # Eliminates ~1.25M compute_all calls (hottest path in the engine).
        _indicator_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
        import json as _json
        for strategy in strategy_configs:
            config_hash = _json.dumps(strategy.indicators, sort_keys=True, ensure_ascii=True)
            for sym in symbols:
                for tf in strategy.timeframes:
                    cache_key = (config_hash, sym, tf)
                    if cache_key in _indicator_cache:
                        continue
                    df_full = feeder.get_all_data_for_symbol(sym, tf)
                    if len(df_full) >= 20:
                        _indicator_cache[cache_key] = compute_all(df_full.copy(), strategy.indicators)

        def _get_cached_df(sym: str, interval: str, indicators: dict, ts) -> pd.DataFrame | None:
            """Return indicator DataFrame sliced to ≤ ts, from cache if possible.

            Uses iloc for O(log n) lookup instead of boolean indexing (O(n)).
            """
            config_hash = _json.dumps(indicators, sort_keys=True, ensure_ascii=True)
            cached = _indicator_cache.get((config_hash, sym, interval))
            if cached is not None:
                try:
                    pos = cached.index.get_loc(ts)
                    if isinstance(pos, slice):
                        pos = pos.stop - 1
                    return cached.iloc[:pos + 1]
                except KeyError:
                    # ts not exactly in index — fall through to boolean
                    return cached[cached.index <= ts]
            # Fallback: compute on the fly
            df = feeder.get_all_data_for_symbol(sym, interval)
            if len(df) < 20:
                return None
            try:
                pos = df.index.get_loc(ts)
                if isinstance(pos, slice):
                    pos = pos.stop - 1
                return compute_all(df.iloc[:pos + 1], indicators)
            except KeyError:
                return compute_all(df[df.index <= ts], indicators)

        # Position sizing
        from core.risk.position_sizer import PositionSizer
        sizer = PositionSizer(
            self.config.hard_limits, self.config.soft_params,
            self.config.core_capital_pct, self.config.satellite_capital_pct)

        # Signal weights — dynamically adjustable to simulate AI market assessment.
        # In live trading, the DeepSeek AI can change these hourly. The backtest
        # re-evaluates weights periodically based on detected market regime.
        base_weights = self.config.signal_weights
        w_ind = base_weights.indicator
        w_ml = base_weights.ml
        w_news = base_weights.news  # included in divisor, not numerator
        _last_weight_update = 0
        _weight_update_interval = 24  # update weights every 24 candles (~24h for 1h)

        def _update_weights(regime: str, step_num: int) -> tuple[float, float, float]:
            """Adjust indicator/ML weights based on market regime.

            Mimics what the live AI market assessment does:
            - Bull market: increase indicator weight (trend is clear), decrease ML
            - Bear market: increase ML weight (need more confirmation), decrease indicator
            - Range market: balanced weights
            """
            nonlocal w_ind, w_ml, w_news, _last_weight_update
            if not simulate_ai_weights:
                return w_ind, w_ml, w_news
            if step_num - _last_weight_update < _weight_update_interval:
                return w_ind, w_ml, w_news
            _last_weight_update = step_num

            base = base_weights
            if regime == "bull":
                # Trend is clear — trust indicators more
                w_ind = max(0.3, base.indicator + 0.1)
                w_ml = max(0.1, base.ml - 0.05)
                w_news = base.news
            elif regime == "bear":
                # Downtrend — indicators can give false reversal signals, trust ML more
                w_ind = max(0.3, base.indicator - 0.05)
                w_ml = min(0.5, base.ml + 0.1)
                w_news = base.news
            else:  # range
                # Choppy — balanced, slightly favor mean-reversion (indicators)
                w_ind = base.indicator
                w_ml = base.ml
                w_news = base.news
            return w_ind, w_ml, w_news

        # Entry threshold
        ENTRY_THRESHOLD = 0.5

        # Max positions (per-strategy when isolated)
        max_positions = self.config.hard_limits.max_open_trades
        if per_strategy_isolation:
            max_positions = max(1, max_positions // max(len(strategy_configs), 1))

        # Position key helper — includes strategy name when isolated
        def _pkey(sym: str, s_name: str = "") -> str:
            return f"{s_name}|{sym}" if per_strategy_isolation else sym

        # Per-strategy×symbol results matrix (use YAML config names as keys)
        per_matrix: dict[str, dict[str, dict]] = {}
        for s_cfg in strategy_configs:
            s_name = s_cfg.name
            per_matrix[s_name] = {}
            # Determine effective symbols for this strategy
            bt_override2 = (strategy_symbols or {}).get(s_name)
            if bt_override2 is not None:
                eff = bt_override2 if bt_override2 else symbols
            else:
                cfg_s = getattr(s_cfg, 'symbols', None)
                eff = cfg_s if cfg_s else symbols
            for sym in eff:
                if sym not in symbols:
                    continue
                per_matrix[s_name][sym] = {
                    "trades": 0, "pnl": 0.0, "winning": 0, "losing": 0,
                    "long_trades": 0, "short_trades": 0,
                }

        pos_counter = 0  # unique position ID
        total_steps = len(feeder)
        step = 0

        # ---- Main Loop ----
        for slice_data in feeder:
            step += 1
            ts = slice_data["timestamp"]
            # Report progress every 10 steps or at start/end
            if progress_callback and (step % 10 == 0 or step == 1 or step == total_steps):
                progress_callback(step, total_steps, ts)

            # --- ML PREDICTION (walk-forward, per-strategy×symbol) ---
            # Each strategy gets its own ML model matched to its primary timeframe.
            for strategy in strategy_configs:
                primary_tf = min(strategy.timeframes, key=_tf_minutes) if strategy.timeframes else "1h"
                tf_params = _ML_TF_PARAMS.get(primary_tf, _ML_TF_PARAMS["1h"])
                if not (strategy.ml_config and strategy.ml_config.enabled):
                    continue

                for sym in symbols:
                    df_tf = feeder.get_all_data_for_symbol(sym, primary_tf)
                    if len(df_tf) < tf_params["min_candles"]:
                        continue

                    key = f"{strategy.name}|{sym}"
                    retrain_idx = _ml_key_idx.get(key, 0)
                    should_retrain = (
                        not skip_ml_training and
                        key not in ml_models and
                        step % _ml_retrain_stagger == retrain_idx % _ml_retrain_stagger and
                        step > 0
                    )

                    # Regime detection on 1h for this symbol (once)
                    if sym not in market_regime:
                        df_1h = feeder.get_all_data_for_symbol(sym, "1h")
                        try:
                            pos_1h = df_1h.index.get_loc(ts)
                            if isinstance(pos_1h, slice): pos_1h = pos_1h.stop - 1
                            market_regime[sym] = self._detect_market_regime(df_1h.iloc[:pos_1h + 1])
                        except KeyError:
                            market_regime[sym] = self._detect_market_regime(df_1h[df_1h.index <= ts])

                    # Fast slice: get_loc is O(log n) vs boolean indexing O(n)
                    try:
                        pos_tf = df_tf.index.get_loc(ts)
                        if isinstance(pos_tf, slice): pos_tf = pos_tf.stop - 1
                        sliced = df_tf.iloc[:pos_tf + 1]
                    except KeyError:
                        sliced = df_tf[df_tf.index <= ts]

                    # ── PatchTST path ──
                    if ml_engine == "patchtst" and patchtst_trainer is not None:
                        if should_retrain:
                            if len(sliced) >= 150:
                                model = self._train_patchtst_model(
                                    patchtst_trainer, sliced, sym, primary_tf,
                                    feature_list=strategy.ml_config.features if strategy.ml_config else None)
                                if model is not None:
                                    ml_models[key] = model

                        model = ml_models.get(key)
                        if model is not None:
                            try:
                                conf = self._predict_patchtst(
                                    patchtst_trainer, model, sliced)
                                if conf is not None:
                                    ml_predictions[key] = conf
                                    fwd = tf_params["forward"]
                                    th = tf_params["threshold"]
                                    future_df = df_tf[df_tf.index > ts]
                                    if len(future_df) >= fwd:
                                        cur_close = float(sliced.iloc[-1]["close"])
                                        fut_close = float(future_df.iloc[fwd - 1]["close"])
                                        ret = (fut_close - cur_close) / cur_close
                                        if abs(ret) >= th:
                                            ml_total += 1
                                            if (ret >= th and conf >= 0.5) or (ret <= -th and conf < 0.5):
                                                ml_correct += 1
                            except Exception:
                                pass

                    # ── TFT path ──
                    elif ml_engine == "tft" and tft_trainer is not None:
                        if should_retrain:
                            if len(sliced) >= 150:
                                model = self._train_tft_model(
                                    tft_trainer, sliced, sym, primary_tf,
                                    feature_list=strategy.ml_config.features if strategy.ml_config else None)
                                if model is not None:
                                    ml_models[key] = model
                            # model updated (round-robin)

                        model = ml_models.get(key)
                        if model is not None:
                            try:
                                conf = self._predict_tft(
                                    tft_trainer, model, sliced)
                                if conf is not None:
                                    ml_predictions[key] = conf
                                    # Accuracy tracking
                                    fwd = tf_params["forward"]
                                    th = tf_params["threshold"]
                                    future_df = df_tf[df_tf.index > ts]
                                    if len(future_df) >= fwd:
                                        cur_close = float(sliced.iloc[-1]["close"])
                                        fut_close = float(future_df.iloc[fwd - 1]["close"])
                                        ret = (fut_close - cur_close) / cur_close
                                        if abs(ret) >= th:
                                            ml_total += 1
                                            if (ret >= th and conf >= 0.5) or (ret <= -th and conf < 0.5):
                                                ml_correct += 1
                            except Exception:
                                pass

                    # ── LightGBM path ──
                    else:
                        if should_retrain:
                            model = self._train_ml_model(sliced, tf_params)
                            if model is not None:
                                ml_models[key] = model
                            # model updated (round-robin)

                        model = ml_models.get(key)
                        if model is not None:
                            try:
                                conf = self._predict_ml(model, sliced)
                                if conf is not None:
                                    ml_predictions[key] = conf

                                    # Accuracy tracking
                                    fwd = tf_params["forward"]
                                    th = tf_params["threshold"]
                                    future_df = df_tf[df_tf.index > ts]
                                    if len(future_df) >= fwd:
                                        cur_close = float(sliced.iloc[-1]["close"])
                                        fut_close = float(future_df.iloc[fwd - 1]["close"])
                                        ret = (fut_close - cur_close) / cur_close
                                        if abs(ret) >= th:
                                            ml_total += 1
                                            if (ret >= th and conf >= 0.5) or (ret <= -th and conf < 0.5):
                                                ml_correct += 1
                            except Exception:
                                pass

            # --- CHECK REDUCE CONDITIONS (partial profit-taking) ---
            for pos_key in list(positions.keys()):
                pos = positions[pos_key]
                sym = pos["symbol"]
                reduce_count = pos.get("reduce_count", 0)
                if reduce_count >= 4:
                    continue

                for strategy in strategy_configs:
                    if strategy.name != pos.get("strategy_name"):
                        continue
                    reduce_cfg = strategy.reduce_conditions
                    if not reduce_cfg:
                        continue
                    conditions = reduce_cfg.get(pos["side"], [])
                    if not conditions:
                        continue

                    for interval in strategy.timeframes:
                        try:
                            df = _get_cached_df(sym, interval, strategy.indicators, ts)
                        except Exception:
                            continue
                        if df is None or len(df) < 20:
                            continue

                        for rc in conditions:
                            cond_str = rc.get("condition", "") if isinstance(rc, dict) else str(rc)
                            rpct = rc.get("reduce_pct", 50) if isinstance(rc, dict) else 50
                            if not cond_str:
                                continue
                            try:
                                mask = evaluate_condition(df, cond_str)
                                if hasattr(mask, 'iloc') and mask.iloc[-1]:
                                    price = float(df["close"].iloc[-1])
                                    qty = pos["quantity"]
                                    reduce_qty = qty * rpct / 100.0
                                    if reduce_qty <= 0:
                                        continue

                                    # Reduce the position
                                    reduce_amount = reduce_qty * price
                                    if pos["side"] == "long":
                                        reduce_pnl = (price - pos["entry_price"]) * reduce_qty
                                    else:
                                        reduce_pnl = (pos["entry_price"] - price) * reduce_qty

                                    pos["quantity"] -= reduce_qty
                                    pos["amount_usdt"] -= reduce_amount
                                    pos["reduce_count"] = reduce_count + 1
                                    balance += reduce_amount + reduce_pnl

                                    events.append({
                                        "time": str(ts), "type": "reduce",
                                        "symbol": sym, "side": pos["side"],
                                        "price": round(price, 4),
                                        "reduce_pct": rpct,
                                        "reduce_qty": round(reduce_qty, 6),
                                        "pnl": round(reduce_pnl, 2),
                                        "strategy": strategy.name,
                                    })
                                    trades.append({
                                        "symbol": sym, "side": pos["side"],
                                        "entry_price": round(pos["entry_price"], 4),
                                        "exit_price": round(price, 4),
                                        "quantity": round(reduce_qty, 6),
                                        "pnl": round(reduce_pnl, 2),
                                        "pnl_pct": round(reduce_pnl / (pos["entry_price"] * reduce_qty) * 100, 2) if pos["entry_price"] > 0 else 0,
                                        "strategy": strategy.name,
                                        "opened_at": str(pos.get("opened_at", ts)),
                                        "closed_at": str(ts),
                                        "amount_usdt": round(reduce_amount, 2),
                                        "is_reduce": True,
                                    })
                                    # Track per-strategy×symbol
                                    cell = per_matrix[strategy.name][sym]
                                    cell["trades"] += 1
                                    cell["pnl"] += reduce_pnl
                                    if reduce_pnl > 0:
                                        cell["winning"] += 1
                                    else:
                                        cell["losing"] += 1
                                    break  # one reduce per timestamp
                            except Exception:
                                pass
                        break  # one interval is enough

            # --- CHECK EXITS ---
            for pos_key in list(positions.keys()):
                pos = positions[pos_key]
                sym = pos["symbol"]

                # Get current close price from the feeder data (using primary TF)
                price_now = 0.0
                try:
                    df_1m = feeder.get_all_data_for_symbol(sym, "1m")
                    df_slice = df_1m[df_1m.index <= ts]
                    if len(df_slice) > 0:
                        price_now = float(df_slice.iloc[-1]["close"])
                except Exception:
                    pass

                if price_now > 0:
                    # ---- STOP-LOSS CHECK ----
                    sl_price = pos.get("stop_loss", 0)
                    if sl_price > 0:
                        if (pos["side"] == "long" and price_now <= sl_price) or \
                           (pos["side"] == "short" and price_now >= sl_price):
                            balance = self._close_position(
                                pos_key, pos, price_now, ts, "stop_loss",
                                trades, events, balance, positions, per_matrix)
                            continue

                    # ---- TAKE-PROFIT CHECK ----
                    tp_levels = pos.get("take_profits", [])
                    for tp_price, tp_pct in tp_levels:
                        if (pos["side"] == "long" and price_now >= tp_price) or \
                           (pos["side"] == "short" and price_now <= tp_price):
                            balance = self._close_position(
                                pos_key, pos, price_now, ts, f"tp_{int(tp_pct*100)}pct",
                                trades, events, balance, positions, per_matrix)
                            break

                if sym not in positions:
                    continue

                for strategy in strategy_configs:
                    if strategy.name != pos.get("strategy_name"):
                        continue
                    for interval in strategy.timeframes:
                        try:
                            df = _get_cached_df(sym, interval, strategy.indicators, ts)
                        except Exception:
                            continue
                        if df is None or len(df) < 20:
                            continue

                        exit_conds = strategy.exit_conditions.get(pos["side"], [])
                        for cond in exit_conds:
                            mask = evaluate_condition(df, cond)
                            if hasattr(mask, 'iloc') and mask.iloc[-1]:
                                exit_price = float(df["close"].iloc[-1])
                                balance = self._close_position(
                                    pos_key, pos, exit_price, ts, "indicator",
                                    trades, events, balance, positions, per_matrix)
                                break
                        if sym not in positions:
                            break
                    if sym not in positions:
                        break

            # --- UPDATE SIGNAL WEIGHTS (simulate AI market assessment) ---
            # Use BTC regime as the broad market indicator
            dominant_regime = market_regime.get("BTCUSDT", "range")
            w_ind, w_ml, w_news = _update_weights(dominant_regime, step)

            # --- CHECK ENTRIES ---
            for strategy in strategy_configs:
                # Backtest-run mapping override takes priority; else fall back to strategy config
                bt_override = (strategy_symbols or {}).get(strategy.name) if strategy_symbols else None
                if bt_override is not None:
                    effective_symbols = bt_override if bt_override else symbols
                else:
                    cfg_syms = getattr(strategy, 'symbols', None)
                    effective_symbols = cfg_syms if cfg_syms else symbols

                for sym in symbols:
                    if sym not in effective_symbols:
                        continue  # strategy not assigned to this symbol
                    if _pkey(sym, strategy.name) in positions:
                        continue
                    if len(positions) >= max_positions:
                        break

                    # Sort timeframes: shortest first (primary signal), rest act as filters
                    sorted_tfs = sorted(strategy.timeframes, key=_tf_minutes)
                    if not sorted_tfs:
                        continue
                    primary_tf = sorted_tfs[0]
                    higher_tfs = sorted_tfs[1:]

                    # --- Evaluate primary (shortest) timeframe for entry signal ---
                    try:
                        df_primary = _get_cached_df(sym, primary_tf, strategy.indicators, ts)
                    except Exception:
                        continue
                    if df_primary is None or len(df_primary) < 20:
                        continue

                    # Evaluate entry conditions on primary timeframe
                    long_active = False
                    short_active = False
                    for side in ["long", "short"]:
                        for cond in strategy.entry_conditions.get(side, []):
                            mask = evaluate_condition(df_primary, cond)
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

                    entry_side = "long" if indicator_signal > 0 else "short"

                    # ---- SIGNAL FUSION ----
                    ml_key = f"{strategy.name}|{sym}"
                    ml_conf = ml_predictions.get(ml_key, 0.5)
                    ml_directional = (ml_conf - 0.5) * 2  # -1 to +1

                    # Per-strategy ML weight override
                    strategy_ml_weight = w_ml
                    if strategy.ml_config and strategy.ml_config.enabled:
                        strategy_ml_weight = strategy.ml_config.weight

                    total_weight = w_ind + strategy_ml_weight + w_news
                    if total_weight > 0:
                        final_score = (indicator_signal * w_ind +
                                       ml_directional * strategy_ml_weight) / total_weight
                    else:
                        final_score = indicator_signal

                    # --- Higher-timeframe trend alignment ---
                    # Apply a confidence multiplier instead of a hard block.
                    # Mild counter-trend trades get a penalty but can still pass.
                    tf_multiplier = 1.0
                    for htf in higher_tfs:
                        df_htf = feeder.get_all_data_for_symbol(sym, htf)
                        if len(df_htf) < 50:
                            continue
                        df_htf = df_htf[df_htf.index <= ts].copy()
                        mult = _check_higher_tf_trend(df_htf, entry_side)
                        tf_multiplier = min(tf_multiplier, mult)
                    final_score *= tf_multiplier

                    # Regime-aware threshold: counter-trend trades need stronger signals
                    regime = market_regime.get(sym, "range")
                    effective_threshold = ENTRY_THRESHOLD
                    if (entry_side == "long" and regime == "bear") or (entry_side == "short" and regime == "bull"):
                        effective_threshold = 0.65  # harder to counter-trend

                    if abs(final_score) < effective_threshold:
                        continue

                    side = "long" if final_score > 0 else "short"
                    price = float(df_primary["close"].iloc[-1])

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

                    pos_key = _pkey(sym, strategy.name)
                    positions[pos_key] = {
                        "symbol": sym, "side": side,
                        "quantity": qty, "entry_price": price,
                        "amount_usdt": amount_usdt,
                        "strategy_name": strategy.name,
                        "opened_at": str(ts), "trade_group": trade_group,
                        "stop_loss": sizer.calculate_stop_loss(price, side),
                        "take_profits": sizer.calculate_take_profits(price, side),
                        "reduce_count": 0,
                    }
                    events.append({
                        "time": str(ts), "type": "entry",
                        "symbol": sym, "side": side, "price": price,
                        "qty": round(qty, 6), "amount_usdt": round(amount_usdt, 2),
                        "strategy": strategy.name,
                        "signal_score": round(final_score, 3),
                        "ml_confidence": round(ml_conf, 3),
                        "timeframe": primary_tf,
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

        # Compute per-cell metrics
        for s_name in per_matrix:
            for sym in per_matrix[s_name]:
                cell = per_matrix[s_name][sym]
                n = cell["trades"]
                cell["win_rate_pct"] = round(cell["winning"] / n * 100, 1) if n > 0 else 0.0
                cell["pnl"] = round(cell["pnl"], 2)

        return {
            "trades": trades, "equity_curve": equity_curve, "events": events,
            "metrics": metrics, "final_balance": round(final_balance, 2),
            "initial_balance": initial_balance,
            "strategies": strategies, "symbols": symbols,
            "date_start": date_start, "date_end": date_end, "mode": mode,
            "per_matrix": per_matrix,
        }

    def _close_position(self, pos_key, pos, exit_price, ts, reason, trades, events,
                         balance, positions, per_matrix):
        """Close a position and record the trade. Used for SL/TP/indicator exits."""
        sym = pos["symbol"]
        entry_price = pos["entry_price"]
        qty = pos["quantity"]
        amount = pos.get("amount_usdt", qty * entry_price)
        side = pos["side"]
        strategy_name = pos.get("strategy_name", "")

        if side == "long":
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty

        trades.append({
            "symbol": sym, "side": side,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "quantity": round(qty, 6),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / (entry_price * qty) * 100, 2) if entry_price > 0 else 0,
            "strategy": strategy_name,
            "opened_at": str(pos.get("opened_at", ts)),
            "closed_at": str(ts),
            "amount_usdt": round(amount, 2),
            "exit_reason": reason,
        })
        balance += amount + pnl
        events.append({
            "time": str(ts), "type": "exit", "reason": reason,
            "symbol": sym, "price": exit_price,
            "pnl": round(pnl, 2), "strategy": strategy_name,
        })

        # Track per-strategy×symbol
        if strategy_name in per_matrix and sym in per_matrix[strategy_name]:
            cell = per_matrix[strategy_name][sym]
            cell["trades"] += 1
            cell["pnl"] += pnl
            if pnl > 0:
                cell["winning"] += 1
            else:
                cell["losing"] += 1
            if side == "long":
                cell["long_trades"] += 1
            else:
                cell["short_trades"] += 1

        del positions[pos_key]
        return balance

    # ---- ML Helpers ----

    # Indicator config used for ML feature computation (imported from features.py)
    # Kept as instance attribute for consistent access
    _ML_INDICATORS = {
        "rsi": {"period": 14, "source": "close"},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bollinger": {"period": 20, "stddev": 2},
        "adx": {"period": 14},
    }

    def _compute_ml_features(self, df: pd.DataFrame,
                             feature_list: list[str] | None = None) -> pd.DataFrame:
        """Compute the full feature set for ML, optionally filtered.

        Uses the shared compute_features() from core.ml.features.
        feature_list=None → all features; non-empty list → filter.
        """
        from core.strategy.indicators import compute_all
        from core.ml.features import compute_features as _compute_features

        df_ind = compute_all(df.copy(), self._ML_INDICATORS)
        # None = all features; [] or non-empty list = filter
        fl = feature_list if feature_list else None
        return _compute_features(df_ind, fl)

    def _train_ml_model(self, df: pd.DataFrame, tf_params: dict = None,
                         feature_list: list[str] | None = None):
        """Train a LightGBM classifier (XGBoost fallback) with timeframe-appropriate labels.

        Each strategy's primary timeframe gets its own model with matched
        forward_periods and threshold (e.g., 1m→20 periods, 1h→4 periods).
        Tries LightGBM first; falls back to XGBoost if LightGBM is unavailable.
        """
        if tf_params is None:
            tf_params = {"forward": 4, "threshold": 0.005, "min_candles": 100}
        min_candles = tf_params.get("min_candles", 100)
        min_samples = max(40, min_candles // 3)
        if len(df) < min_candles:
            return None
        try:
            from core.ml.features import create_binary_label

            feature_df = self._compute_ml_features(df, feature_list)
            labels = create_binary_label(
                df, forward_periods=tf_params["forward"],
                threshold=tf_params["threshold"])

            common_idx = feature_df.index.intersection(labels.dropna().index)
            if len(common_idx) < min_samples:
                return None
            X = feature_df.loc[common_idx].values.astype(float)
            y = labels.loc[common_idx].values

            n_up = int(y.sum())
            n_down = len(y) - n_up
            scale_pos_weight = max(1.0, n_down / max(n_up, 1))

            # Try LightGBM first (faster, often more accurate)
            try:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(
                    n_estimators=150, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    min_child_samples=20,
                    reg_alpha=0.1, reg_lambda=0.1,
                    verbosity=-1, random_state=42)
                model.fit(X, y)
                return model
            except ImportError:
                pass

            # XGBoost fallback
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss', verbosity=0, random_state=42)
            model.fit(X, y)
            return model
        except Exception as e:
            logger.debug(f"ML train failed: {e}")
            return None

    def _predict_ml(self, model, df: pd.DataFrame) -> float | None:
        """Predict probability(price up >= 0.5%) using the trained model.

        Returns confidence in [0,1] or None if data is insufficient.
        If the model is too uncertain (0.38–0.62), returns 0.5 (neutral).
        The wider neutral band reflects the 30-dim feature set's higher
        dimensionality.
        """
        if len(df) < 50:
            return None
        try:
            feature_df = self._compute_ml_features(df)
            if len(feature_df) == 0:
                return None
            X = feature_df.iloc[-1:].values.astype(float)
            proba = model.predict_proba(X)
            if proba.shape[1] >= 2:
                conf = float(proba[0][1])
                # Shrink toward 0.5 if model is uncertain
                if 0.38 <= conf <= 0.62:
                    return 0.5  # neutral — model doesn't know
                return conf
            return 0.5
        except Exception as e:
            logger.debug(f"ML predict failed: {e}")
            return None

    # ── TFT helpers ──────────────────────────────────────────────────

    def _train_tft_model(self, tft_trainer, df: pd.DataFrame, symbol: str,
                         interval: str, max_train_rows: int = 5000):
        """Train a TFT model on sliced DataFrame (walk-forward safe).

        Caps training data to *max_train_rows* most recent candles so that
        retrain time stays constant regardless of how far the backtest has
        progressed.
        """
        try:
            from core.strategy.indicators import compute_all
            from core.ml.features import compute_features as _cf, create_regression_label, REQUIRED_INDICATORS

            # Cap to recent data to keep training time constant
            if len(df) > max_train_rows:
                df = df.iloc[-max_train_rows:]

            df_ind = compute_all(df.copy(), REQUIRED_INDICATORS)
            X = _cf(df_ind, None)
            y = create_regression_label(df_ind, forward_periods=4)
            X["label"] = y.values

            feature_cols = [c for c in X.columns if c != "label"
                          and X[c].dtype in ('float64', 'float32', 'int64')]

            t0 = time.time()
            model, metrics = tft_trainer.train(
                X, feature_cols=feature_cols, label_col="label",
                epochs=60, batch_size=64, learning_rate=1e-3,
                validation_split=0.2, patience=15)
            elapsed = time.time() - t0

            if model is not None:
                dev = next(model.parameters()).device
                logger.info(
                    f"TFT trained {symbol} {interval} | "
                    f"device={dev} rows={len(X)} "
                    f"acc={metrics.get('val_accuracy', 0):.1%} "
                    f"epochs={metrics.get('epochs_trained', 0)} "
                    f"time={elapsed:.1f}s")
            return model
        except Exception as e:
            logger.debug(f"TFT train failed for {symbol}: {e}")
            return None

    def _predict_tft(self, tft_trainer, model, df: pd.DataFrame) -> float | None:
        """Predict with TFT — returns confidence in [0, 1]."""
        if len(df) < 100:
            return None
        try:
            from core.strategy.indicators import compute_all
            from core.ml.features import compute_features as _cf, REQUIRED_INDICATORS

            df_ind = compute_all(df.copy(), REQUIRED_INDICATORS)
            X = _cf(df_ind, None)
            result = tft_trainer.predict(model, X)
            if result is None:
                return None

            tft_conf = result["confidence"]
            tft_dir = result["direction"]
            if tft_dir > 0:
                return tft_conf
            else:
                return 1.0 - tft_conf
        except Exception as e:
            logger.debug(f"TFT predict failed: {e}")
            return None

    # ── PatchTST helpers ─────────────────────────────────────────────

    def _train_patchtst_model(self, trainer, df: pd.DataFrame, symbol: str,
                              interval: str, max_train_rows: int = 5000,
                              feature_list: list[str] | None = None):
        """Train a PatchTST model with triple-barrier labels."""
        try:
            from core.strategy.indicators import compute_all
            from core.ml.features import (compute_features as _cf,
                  create_triple_barrier_label, REQUIRED_INDICATORS)

            if len(df) > max_train_rows:
                df = df.iloc[-max_train_rows:]

            df_ind = compute_all(df.copy(), REQUIRED_INDICATORS)
            # feature_list=None or [] → use all features; non-empty → filter
            fl = feature_list if feature_list else None
            X = _cf(df_ind, fl)
            # Triple barrier: 2% up/down, 24 periods lookahead
            y = create_triple_barrier_label(
                df_ind, forward_periods=24, upper_pct=0.02, lower_pct=0.02,
                timeout_label=2.0)  # timeout = class 2
            X["label"] = y.values

            t0 = time.time()
            model, metrics = trainer.train(
                X, feature_cols=None, label_col="label",
                epochs=50, batch_size=64, learning_rate=1e-3,
                validation_split=0.2, patience=12)
            elapsed = time.time() - t0

            if model is not None:
                dev = next(model.parameters()).device
                logger.info(
                    f"PatchTST trained {symbol} {interval} | "
                    f"device={dev} rows={len(X)} "
                    f"acc={metrics.get('val_accuracy', 0):.1%} "
                    f"epochs={metrics.get('epochs_trained', 0)} "
                    f"time={elapsed:.1f}s")
            return model
        except Exception as e:
            logger.debug(f"PatchTST train failed for {symbol}: {e}")
            return None

    def _predict_patchtst(self, trainer, model, df: pd.DataFrame) -> float | None:
        """Predict with PatchTST — returns confidence in [0, 1]."""
        if len(df) < 100:
            return None
        try:
            from core.strategy.indicators import compute_all
            from core.ml.features import compute_features as _cf, REQUIRED_INDICATORS

            df_ind = compute_all(df.copy(), REQUIRED_INDICATORS)
            X = _cf(df_ind, None)
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
                return 0.5  # timeout/neutral
        except Exception as e:
            logger.debug(f"PatchTST predict failed: {e}")
            return None

    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect the prevailing market regime from 1h data.

        Returns 'bull', 'bear', or 'range' based on EMA alignment and ADX.
        """
        if len(df) < 100:
            return "range"
        close = df["close"].values
        ema20 = float(pd.Series(close).ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(pd.Series(close).ewm(span=50, adjust=False).mean().iloc[-1])
        last_close = float(close[-1])
        if last_close > ema20 > ema50:
            return "bull"
        elif last_close < ema20 < ema50:
            return "bear"
        else:
            return "range"
