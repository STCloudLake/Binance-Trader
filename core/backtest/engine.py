"""Backtesting engine — synchronous replay with ML, signal fusion, and risk controls."""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from core.backtest.data_feeder import DataFeeder
from core.backtest.metrics import calculate_metrics
from core.backtest.engine_hybrid import run_hybrid
from core.backtest.signal_matrix import NO_MARKET_DATA_MESSAGE
from core.backtest.cost_model import apply_trading_costs, freeze_run_spreads
from core.backtest.trade_book import close_position
from core.strategy.indicators import compute_all, evaluate_condition
from core.strategy.evaluation_kernel import (
    evaluate_entry_conditions,
    evaluate_exit_conditions,
    fuse_signals,
    check_higher_tf_trend,
    detect_market_regime,
)
from core.market_data.provider import INTERVAL_SPEC, interval_minutes


def _tf_minutes(tf: str) -> int:
    """Convert a timeframe string to minutes for comparison/sorting.

    Delegates to the shared :data:`~core.market_data.provider.INTERVAL_SPEC`
    registry (this module used to keep a private copy of the same map).
    """
    return interval_minutes(tf, 60)


#: Config attribute naming the symbol whose regime drives the AI weight profile.
REGIME_PROXY_CONFIG_KEY = "backtest_regime_symbol"


def _resolve_regime_proxy(config, symbols, market_regime) -> str | None:
    """Symbol whose detected regime represents the broad market for a run.

    ``config.backtest_regime_symbol`` overrides; the default is the FIRST symbol
    of this run (not a hard-coded BTCUSDT, which pinned every run without BTC to
    "range").  A configured proxy that is not part of the run — or has no
    detected regime — falls back to the first symbol that does.
    """
    regime = market_regime or {}
    proxy = getattr(config, REGIME_PROXY_CONFIG_KEY, None) or (
        symbols[0] if symbols else None)
    if proxy not in regime:
        proxy = next((s for s in (symbols or []) if s in regime), None)
    return proxy


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
            # Strategies with partial-reduce rules cannot be modelled by the hybrid
            # engine (it has no reduce path), so route them to the legacy engine
            # rather than silently under-reporting their trades.
            if any(getattr(s, "reduce_conditions", None) for s in strategy_configs):
                logger.warning("Strategy set contains reduce_conditions — using the legacy "
                               "engine (hybrid does not support partial reduces)")
                return "legacy"
            return "hybrid"
        return "legacy"

    def run(self, strategies: list[str], symbols: list[str],
            date_start: str, date_end: str,
            initial_balance: float = 10000.0, mode: str = "full",
            progress_callback=None,
            strategy_symbols: dict[str, list[str]] = None,
            simulate_ai_weights: bool = True,
            spread_overrides: dict | None = None) -> dict:
        """Alias for run_with_exit_evaluation (parameter order matches)."""
        return self.run_with_exit_evaluation(
            strategies, symbols, date_start, date_end,
            initial_balance, mode,
            progress_callback=progress_callback,
            strategy_symbols=strategy_symbols,
            simulate_ai_weights=simulate_ai_weights,
            spread_overrides=spread_overrides)

    def run_with_exit_evaluation(self, strategies, symbols, date_start, date_end,
                                  initial_balance=10000.0, mode="full",
                                  progress_callback=None,
                                  strategy_symbols: dict[str, list[str]] = None,
                                  simulate_ai_weights: bool = True,
                                  ml_engine: str = "lightgbm",
                                  skip_ml_training: bool = False,
                                  per_strategy_isolation: bool = False,
                                  spread_overrides: dict | None = None):
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
            spread_overrides: Optional per-symbol spread (%) map for THIS run —
                an explicit override beats the live depth-derived value and the
                documented default (core/backtest/cost_model.py).
        """
        t0 = time.time()

        # ── Cost model: derive the spread for the symbols THIS run uses ──
        # Override → live depth-derived → documented default (cost_model.py).
        # Resolved once here (cached ~5 min) and pinned on the instance so the
        # trade loop never does I/O and every symbol of the run gets its own
        # spread instead of the old "5 hardcoded pairs, else 0.03" guess.
        self._run_spread_pct = freeze_run_spreads(symbols, self.config, spread_overrides)

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
            # engine_hybrid/EventDrivenExecutor read the spread table off the
            # config object; expose the run's derived per-symbol spreads for the
            # duration of the call only (restored in `finally`).
            _saved_spreads = getattr(self.config, "backtest_spread_pct", None)
            try:
                self.config.backtest_spread_pct = dict(self._run_spread_pct)
                return run_hybrid(
                    strategies, symbols, date_start, date_end,
                    self.config, self.strategy_engine.loader,
                    initial_balance=initial_balance,
                    per_strategy_isolation=per_strategy_isolation,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                logger.warning(f"Hybrid engine failed ({e}), falling back to legacy. "
                              "Results may differ from hybrid mode.")
                # Fall through to legacy engine below
            finally:
                if _saved_spreads is None:
                    try:
                        del self.config.backtest_spread_pct
                    except AttributeError:
                        pass
                else:
                    self.config.backtest_spread_pct = _saved_spreads

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
            return {"error": NO_MARKET_DATA_MESSAGE}

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
        # Collect (prediction, actual_return) pairs for post-hoc accuracy evaluation
        ml_eval_pairs: list[tuple[float, float, float]] = []  # (confidence, actual_ret, threshold)
        # Training cost per retrain: LightGBM ~0.3s, PatchTST ~8s, TFT ~15s
        # Use longer intervals for expensive models to keep backtest time reasonable.
        if ml_engine == "tft":
            ml_retrain_interval = 800
        elif ml_engine == "patchtst":
            ml_retrain_interval = 500  # PatchTST is ~2x faster than TFT
        else:
            ml_retrain_interval = 100
        market_regime: dict[str, str] = {}

        # Per-timeframe ML parameters.
        #
        # The *interval set* comes from ``INTERVAL_SPEC`` (the single registry),
        # so adding an interval there teaches the ML path about it automatically
        # (falls back to ``_ML_PARAM_DEFAULT`` until it is tuned).  ``forward``
        # (lookahead bars) and ``threshold`` are ML-only tuning values that the
        # registry deliberately does not carry, and ``min_candles`` here is the
        # *model-training* gate, which is intentionally stricter than the
        # registry's prefetch "enough data on disk" threshold.
        _ML_PARAM_DEFAULT = {"forward": 4, "threshold": 0.005, "min_candles": 100}
        _ML_PARAM_TUNED = {
            "1m":  {"forward": 20, "threshold": 0.003, "min_candles": 300},
            "3m":  {"forward": 15, "threshold": 0.004, "min_candles": 200},
            "5m":  {"forward": 12, "threshold": 0.005, "min_candles": 200},
            "15m": {"forward": 8,  "threshold": 0.005, "min_candles": 150},
            "30m": {"forward": 6,  "threshold": 0.005, "min_candles": 120},
            "2h":  {"forward": 4,  "threshold": 0.006, "min_candles": 80},
            "4h":  {"forward": 4,  "threshold": 0.008, "min_candles": 60},
            "6h":  {"forward": 4,  "threshold": 0.010, "min_candles": 50},
            "8h":  {"forward": 4,  "threshold": 0.012, "min_candles": 40},
            "12h": {"forward": 4,  "threshold": 0.015, "min_candles": 30},
            "1d":  {"forward": 4,  "threshold": 0.020, "min_candles": 25},
            "3d":  {"forward": 4,  "threshold": 0.030, "min_candles": 20},
            "1w":  {"forward": 4,  "threshold": 0.050, "min_candles": 15},
        }
        _ML_TF_PARAMS = {
            tf: {**_ML_PARAM_DEFAULT, **_ML_PARAM_TUNED.get(tf, {})}
            for tf in INTERVAL_SPEC
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
        # Bounded to ~500 MB to prevent OOM during GA/WF with many unique configs.
        _indicator_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
        _MAX_CACHE_BYTES = 500 * 1024 * 1024  # 500 MB
        _cache_total_bytes = 0
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
                        # compute_all internally does df.copy(), so no need for outer copy
                        cached_df = compute_all(df_full, strategy.indicators)
                        entry_bytes = len(cached_df) * len(cached_df.columns) * 8
                        # Evict oldest entries until we fit under the cap
                        while _indicator_cache and _cache_total_bytes + entry_bytes > _MAX_CACHE_BYTES:
                            oldest_key = next(iter(_indicator_cache))
                            oldest_df = _indicator_cache.pop(oldest_key)
                            _cache_total_bytes -= len(oldest_df) * len(oldest_df.columns) * 8
                        _indicator_cache[cache_key] = cached_df
                        _cache_total_bytes += entry_bytes

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
                    "gross_win_pnl": 0.0, "gross_loss_pnl": 0.0,
                }

        pos_counter = 0  # unique position ID
        total_steps = len(feeder)
        step = 0

        # ---- Main Loop ----
        for slice_data in feeder:
            step += 1
            # Yield GIL periodically so asyncio event loop stays responsive
            if step % 200 == 0:
                time.sleep(0)
            ts = slice_data["timestamp"]
            # Report progress every 10 steps or at start/end
            if progress_callback and (step % 10 == 0 or step == 1 or step == total_steps):
                progress_callback(step, total_steps, ts)

            # --- MARKET REGIME (once per symbol) ---
            # Must run for EVERY step regardless of ML: the regime gates the
            # counter-trend entry threshold (0.65 vs 0.5). Previously this lived
            # inside the ML branch below, so with ML disabled the regime was always
            # "range" and the legacy engine disagreed with the hybrid engine.
            for sym in symbols:
                if sym in market_regime:
                    continue
                df_1h = feeder.get_all_data_for_symbol(sym, "1h")
                if df_1h is None or len(df_1h) == 0:
                    market_regime[sym] = "range"
                    continue
                try:
                    pos_1h = df_1h.index.get_loc(ts)
                    if isinstance(pos_1h, slice): pos_1h = pos_1h.stop - 1
                    market_regime[sym] = detect_market_regime(df_1h.iloc[:pos_1h + 1])
                except KeyError:
                    market_regime[sym] = detect_market_regime(df_1h[df_1h.index <= ts])

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

                    # Regime detection happens once per symbol in the main loop
                    # (see below) so it applies with or without ML.
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
                                    # Defer accuracy evaluation to post-loop
                                    fwd = tf_params["forward"]
                                    th = tf_params["threshold"]
                                    future_df = df_tf[df_tf.index > ts]
                                    if len(future_df) >= fwd:
                                        cur_close = float(sliced.iloc[-1]["close"])
                                        fut_close = float(future_df.iloc[fwd - 1]["close"])
                                        ret = (fut_close - cur_close) / cur_close
                                        ml_eval_pairs.append((conf, ret, th))
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
                                    # Defer accuracy evaluation to post-loop
                                    fwd = tf_params["forward"]
                                    th = tf_params["threshold"]
                                    future_df = df_tf[df_tf.index > ts]
                                    if len(future_df) >= fwd:
                                        cur_close = float(sliced.iloc[-1]["close"])
                                        fut_close = float(future_df.iloc[fwd - 1]["close"])
                                        ret = (fut_close - cur_close) / cur_close
                                        ml_eval_pairs.append((conf, ret, th))
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

                                    # Defer accuracy evaluation to post-loop
                                    fwd = tf_params["forward"]
                                    th = tf_params["threshold"]
                                    future_df = df_tf[df_tf.index > ts]
                                    if len(future_df) >= fwd:
                                        cur_close = float(sliced.iloc[-1]["close"])
                                        fut_close = float(future_df.iloc[fwd - 1]["close"])
                                        ret = (fut_close - cur_close) / cur_close
                                        ml_eval_pairs.append((conf, ret, th))
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
                                        cell["gross_win_pnl"] += reduce_pnl
                                    else:
                                        cell["losing"] += 1
                                        cell["gross_loss_pnl"] += abs(reduce_pnl)
                                    break  # one reduce per timestamp
                            except Exception:
                                pass
                        break  # one interval is enough

            # --- CHECK EXITS ---
            for pos_key in list(positions.keys()):
                pos = positions[pos_key]
                sym = pos["symbol"]

                # Get current close price from the feeder data (using the position's
                # own timeframe). Must match the entry price basis and the hybrid
                # engine: previously this read the *1m* bar labelled at the start of
                # the higher-TF bar, mixing two price bases between the engines.
                price_now = 0.0
                pos_tf = pos.get("timeframe", "1h")
                try:
                    df_tf = feeder.get_all_data_for_symbol(sym, pos_tf)
                    df_slice = df_tf[df_tf.index <= ts]
                    if len(df_slice) > 0:
                        price_now = float(df_slice.iloc[-1]["close"])
                except Exception:
                    pass

                if price_now > 0:
                    # ---- STOP-LOSS CHECK ----
                    # Fill at the stop level (not the bar close): with bar-close-only
                    # evaluation the close can be far through the level, and filling
                    # at the close made the legacy engine systematically more
                    # pessimistic than the hybrid engine on stop-outs. Slippage is
                    # modelled by the cost model, not by the gap between close and stop.
                    sl_price = pos.get("stop_loss", 0)
                    if sl_price > 0:
                        if (pos["side"] == "long" and price_now <= sl_price) or \
                           (pos["side"] == "short" and price_now >= sl_price):
                            balance = self._close_position(
                                pos_key, pos, sl_price, ts, "stop_loss",
                                trades, events, balance, positions, per_matrix)
                            continue

                    # ---- TAKE-PROFIT CHECK ----
                    tp_levels = pos.get("take_profits", [])
                    for tp_price, tp_pct in tp_levels:
                        if (pos["side"] == "long" and price_now >= tp_price) or \
                           (pos["side"] == "short" and price_now <= tp_price):
                            balance = self._close_position(
                                pos_key, pos, tp_price, ts, f"tp_{int(tp_pct*100)}pct",
                                trades, events, balance, positions, per_matrix)
                            break

                if pos_key not in positions:
                    continue

                # ---- TRAILING STOP UPDATE ----
                # Update best price seen and check trailing stop
                trailing_pct = pos.get("trailing_stop_pct", 0) / 100.0
                if trailing_pct > 0 and price_now > 0:
                    side = pos["side"]
                    best = pos.get("best_price", pos["entry_price"])
                    if side == "long" and price_now > best:
                        pos["best_price"] = price_now
                    elif side == "short" and price_now < best:
                        pos["best_price"] = price_now
                    # Update trailing stop level
                    best_price = pos["best_price"]
                    if side == "long":
                        pos["stop_loss"] = best_price * (1 - trailing_pct)
                    else:
                        pos["stop_loss"] = best_price * (1 + trailing_pct)

                # ---- MAX HOLD TIME CHECK ----
                max_hours = pos.get("max_hold_hours", 0)
                if max_hours > 0 and price_now > 0:
                    try:
                        opened = pd.Timestamp(pos["opened_at"])
                        held_hours = (ts - opened).total_seconds() / 3600
                        if held_hours >= max_hours:
                            balance = self._close_position(
                                pos_key, pos, price_now, ts, "max_hold",
                                trades, events, balance, positions, per_matrix)
                            continue
                    except Exception:
                        pass

                if pos_key not in positions:
                    continue

                # ---- INDICATOR EXITS (skip if strategy uses risk-only exits) ----
                use_indicator = pos.get("use_indicator_exits", True)
                if use_indicator:
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

                            # ── Shared Kernel: Exit condition evaluation ──
                            if evaluate_exit_conditions(
                                df, strategy.exit_conditions, pos["side"]):
                                exit_price = float(df["close"].iloc[-1])
                                balance = self._close_position(
                                    pos_key, pos, exit_price, ts, "indicator",
                                    trades, events, balance, positions, per_matrix)
                            if pos_key not in positions:
                                break
                        if pos_key not in positions:
                            break

            # --- UPDATE SIGNAL WEIGHTS (simulate AI market assessment) ---
            # The broad-market regime proxy is configurable
            # (``config.backtest_regime_symbol``); by default it is the FIRST
            # symbol of this run, no longer a hard-coded BTCUSDT.  Pinning the
            # weights to BTC silently forced every run whose symbol list did not
            # contain BTC to "range", so e.g. an altcoin-only run never got the
            # bull/bear weight profile its own regime implied.
            proxy = _resolve_regime_proxy(self.config, symbols, market_regime)
            dominant_regime = market_regime.get(proxy, "range") if proxy else "range"
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

                    # ── Shared Kernel: Entry condition evaluation ──
                    long_active, short_active = evaluate_entry_conditions(
                        df_primary, strategy.entry_conditions)

                    if long_active and short_active:
                        continue
                    indicator_signal = 1.0 if long_active else -1.0 if short_active else 0.0
                    if indicator_signal == 0.0:
                        continue

                    entry_side = "long" if indicator_signal > 0 else "short"

                    # ── Shared Kernel: Signal fusion ──
                    ml_key = f"{strategy.name}|{sym}"
                    ml_conf = ml_predictions.get(ml_key, 0.5)
                    ml_enabled = bool(strategy.ml_config and strategy.ml_config.enabled)

                    # ── Volatility expansion heuristic (backtest mode) ──
                    # Compare recent vol (last 10 bars) to longer-term vol (20 bars).
                    # Used by position sizer to scale position down during high-vol.
                    vol_expanding = False
                    if len(df_primary) >= 21:
                        ret_series = df_primary["close"].pct_change()
                        recent_vol = float(ret_series.iloc[-10:].std())
                        hist_vol = float(ret_series.iloc[-21:].std())
                        vol_expanding = recent_vol > hist_vol

                    final_score = fuse_signals(
                        indicator_signal=indicator_signal,
                        ml_confidence=ml_conf,
                        news_sentiment=None,  # backtest mode — no historical news
                        w_indicator=w_ind,
                        w_ml=w_ml,
                        w_news=w_news,
                        ml_enabled=ml_enabled,
                        strategy_ml_weight=strategy.ml_config.weight if ml_enabled else None,
                    )

                    # ── Shared Kernel: Higher-timeframe trend alignment ──
                    tf_multiplier = 1.0
                    for htf in higher_tfs:
                        df_htf = feeder.get_all_data_for_symbol(sym, htf)
                        if len(df_htf) < 50:
                            continue
                        df_htf = df_htf[df_htf.index <= ts].copy()
                        mult = check_higher_tf_trend(df_htf, entry_side)
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

                    # ---- POSITION SIZING (volatility-aware) ----
                    qty, risk_amount = sizer.calculate_position_size(
                        balance, price, "satellite",
                        volatility_expanding=vol_expanding)
                    if qty <= 0:
                        continue

                    # ── Risk-based position sizing (Kelly-lite) ──
                    # Tighter stop → larger position for same risk budget
                    re = strategy.risk_exit
                    if re is not None:
                        risk_capital = balance * 0.01  # risk 1% of capital per trade
                        sl_dist = re.stop_loss_pct / 100.0
                        qty_risk = risk_capital / (price * sl_dist)
                        # Blend: use risk-based if it's more conservative, else keep base
                        qty = min(qty, qty_risk) if qty_risk > 0 else qty

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

                    # Use strategy-specific risk exits if configured, else PositionSizer defaults
                    re = strategy.risk_exit
                    if re is not None:
                        sl_pct = re.stop_loss_pct / 100.0
                        sl = price * (1 - sl_pct) if side == "long" else price * (1 + sl_pct)
                    else:
                        sl = sizer.calculate_stop_loss(price, side)
                    # Trailing distance comes from the shared sizer helper so live,
                    # legacy and hybrid all trail identically (was a hardcoded 1.5%).
                    tp_dist = sizer.trailing_stop_distance_pct(re) / 100.0

                    pos_key = _pkey(sym, strategy.name)
                    positions[pos_key] = {
                        "symbol": sym, "side": side,
                        "quantity": qty, "entry_price": price,
                        "amount_usdt": amount_usdt,
                        "strategy_name": strategy.name,
                        "opened_at": str(ts), "trade_group": trade_group,
                        "timeframe": primary_tf,
                        "stop_loss": sl,
                        "trailing_stop_pct": round(tp_dist * 100, 1),
                        "best_price": price,  # for trailing stop tracking
                        "max_hold_hours": re.max_hold_hours if re else 0,
                        "use_indicator_exits": re.use_indicator_exits if re else True,
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
                    # NOTE: `continue`, not `break`. This loop iterates the symbols of
                    # the current strategy; breaking after the first entry silently
                    # starved every later symbol in the list (BTCUSDT always won over
                    # ETHUSDT), so a strategy could never open more than one position
                    # per timestamp. "One entry per symbol" is already guaranteed by
                    # the `_pkey(...) in positions` check above. The hybrid engine
                    # never had this artefact, which is why the two engines disagreed.
                    continue  # next symbol for this strategy

            # --- EQUITY CURVE ---
            invested = sum(p.get("amount_usdt", 0) for p in positions.values())
            equity_curve.append({
                "time": str(ts), "equity": round(balance + invested, 2),
                "balance": round(balance, 2), "invested": round(invested, 2),
            })

        # ── Force-close positions still open at the end ──
        # Without this, open positions never appear in `trades`, so win rate,
        # profit factor, Sharpe and max consecutive losses silently ignored the
        # final (possibly largest) PnL while the equity curve still counted the
        # capital as invested. The hybrid engine always did this, which is why
        # the two engines reported different trade sets.
        if positions and len(equity_curve) > 0:
            last_ts = pd.Timestamp(equity_curve[-1]["time"])
            for pos_key in list(positions.keys()):
                pos = positions[pos_key]
                sym = pos["symbol"]
                final_price = pos["entry_price"]
                try:
                    df_tf = feeder.get_all_data_for_symbol(sym, pos.get("timeframe", "1h"))
                    df_slice = df_tf[df_tf.index <= last_ts]
                    if len(df_slice) > 0:
                        final_price = float(df_slice.iloc[-1]["close"])
                except Exception:
                    pass
                balance = self._close_position(
                    pos_key, pos, final_price, last_ts, "end_of_backtest",
                    trades, events, balance, positions, per_matrix)
            # Reflect the realised cash in the final equity point.
            equity_curve[-1]["balance"] = round(balance, 2)
            equity_curve[-1]["invested"] = 0.0
            equity_curve[-1]["equity"] = round(balance, 2)

        # Use last equity value (includes open position value), not just cash balance
        final_balance = equity_curve[-1]["equity"] if equity_curve else balance

        # ── Post-hoc ML accuracy evaluation (no look-ahead bias) ──
        # Accuracy is computed from the collected eval pairs after the loop,
        # ensuring it is purely diagnostic and doesn't affect trading decisions.
        for conf, ret, th in ml_eval_pairs:
            if abs(ret) >= th:
                ml_total += 1
                if (ret >= th and conf >= 0.5) or (ret <= -th and conf < 0.5):
                    ml_correct += 1

        metrics = calculate_metrics(trades, equity_curve, initial_balance, final_balance)
        metrics["ml_accuracy_pct"] = round(
            ml_correct / ml_total * 100 if ml_total > 0 else 0, 1)
        metrics["runtime_seconds"] = round(time.time() - t0, 1)

        # ── Monte Carlo robustness assessment ──
        try:
            from core.backtest.monte_carlo import monte_carlo_simulation
            mc_result = monte_carlo_simulation(
                trades, n_simulations=2000, initial_balance=initial_balance)
        except Exception as e:
            logger.warning(f"Monte Carlo simulation failed: {e}")
            mc_result = None

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
            "monte_carlo": mc_result,
        }

    def _close_position(self, pos_key, pos, exit_price, ts, reason, trades, events,
                         balance, positions, per_matrix):
        """Close a position and record the trade. Used for SL/TP/indicator exits.

        Thin wrapper over the shared implementation in
        :mod:`core.backtest.trade_book` (previously a copy-paste clone that the
        hybrid engine duplicated).
        """
        return close_position(
            pos_key, pos, exit_price, ts, reason, trades, balance, positions, per_matrix,
            cost_fn=lambda ep, xp, q, s: apply_trading_costs(
                ep, xp, q, s, self.config,
                overrides=getattr(self, "_run_spread_pct", None)),
            events=events,
        )

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
