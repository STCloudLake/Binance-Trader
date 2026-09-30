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
from core.backtest.cost_model import apply_trading_costs
from core.backtest.trade_book import close_position
from core.strategy.indicators import compute_all, evaluate_condition
from core.strategy.evaluation_kernel import (
    evaluate_exit_conditions,
    fuse_signals,
    check_higher_tf_trend,
    detect_market_regime,
)
from core.market_data.provider import INTERVAL_SPEC, interval_minutes
from core.ml.features import FEATURE_NAMES, REQUIRED_INDICATORS


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


# ── ML prediction contract helpers (Phase P2 item 4, consumed here by P1.5) ──
#
# The engine's ML path used to carry a bare ``confidence`` float.  Phase P2
# added the richer prediction contract (``p_up`` — a real probability — plus the
# model's own ``base_rate``, the signed ``score`` and an ``abstained`` flag) and
# published it from the live path (``core.ml.predictor``: on abstention
# ``confidence`` becomes the base rate so a consumer that only reads
# ``confidence`` fuses a zero vote).
#
# The legacy neutral band (``0.38 ≤ confidence ≤ 0.62`` → 0.5), which only made
# sense while the engines could not tell "P(up) = 0.6 against a 0.5 base rate"
# from "P(up) = 0.6 against a 0.62 base rate", is kept for the legacy contract
# (a bare float, no ``base_rate``/``score``) so those callers stay
# bit-identical.  Callers that DO supply the model's base rate get the signed
# score (``core.ml.calibration.signed_score``): a bullish ``p_up`` above the
# model's own base rate is a positive contribution, and an exact
# ``p_up == base_rate`` is a neutral one.

#: Confidence band treated as "no call" for the legacy bare-float contract.
_ML_NEUTRAL_BAND = (0.38, 0.62)


def _normalise_ml_prediction(raw) -> dict:
    """Normalise an ML model output to the P2 prediction contract.

    Accepts what the engine's ML paths emit:

    * a bare ``float`` — legacy ``confidence`` (no ``p_up``/``base_rate``), the
      neutral band above is applied and ``base_rate``/``score`` stay ``None`` so
      ``fuse_signals`` reproduces its historical ``(conf − 0.5) × 2`` term;
    * ``(direction, confidence)`` — the PatchTST/TFT tri-class contract, where
      ``confidence`` is a magnitude: ``p_up`` is derived by sign;
    * a ``dict`` — ``p_up`` (or ``confidence``/``direction``), ``base_rate``,
      ``score``, ``abstained``.  The signed score is preferred when supplied.

    Returns ``{p_up, base_rate, score, abstained}``; ``base_rate``/``score`` are
    ``None`` when the caller did not supply them.
    """
    out = {"p_up": 0.5, "base_rate": None, "score": None, "abstained": False}

    if isinstance(raw, dict):
        base_rate = raw.get("base_rate")
        base_rate = None if base_rate is None else float(base_rate)
        if not (base_rate and 0.0 < base_rate < 1.0):
            base_rate = None
        if "p_up" in raw and raw["p_up"] is not None:
            p_up = float(raw["p_up"])
        else:
            conf = float(raw.get("confidence", 0.5))
            direction = raw.get("direction")
            if direction is None:
                p_up = conf
            else:
                direction = float(direction)
                p_up = conf if direction > 0 else (1.0 - conf if direction < 0 else 0.5)
        if base_rate is not None and p_up == 0.5:
            # P2 publishes ``confidence = base_rate`` on abstention; the same
            # neutral vote appears as a bare 0.5 in the legacy tri-class path.
            p_up = base_rate
        score = raw.get("score")
        out.update({
            "p_up": float(min(1.0, max(0.0, p_up))),
            "base_rate": base_rate,
            "score": None if score is None else float(score),
            "abstained": bool(raw.get("abstained", False)),
        })
        return out

    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        direction, conf = float(raw[0]), float(raw[1])
        out["p_up"] = conf if direction > 0 else (1.0 - conf if direction < 0 else 0.5)
        return out

    conf = float(raw)
    if _ML_NEUTRAL_BAND[0] <= conf <= _ML_NEUTRAL_BAND[1]:
        conf = 0.5  # neutral — model doesn't know (legacy band)
    out["p_up"] = conf
    return out


def _ml_fusion_inputs(pred: dict) -> dict:
    """``fuse_signals`` keyword arguments for a normalised ML prediction.

    ``ml_score`` is passed **only** when the caller supplied a base rate or a
    signed score; without either, ``ml_base_rate`` stays at the kernel default
    (0.5) and ``ml_score`` stays ``None``, which makes the fused term exactly the
    historical ``(confidence − 0.5) × 2`` — the legacy behaviour bit-identical.
    """
    kwargs: dict = {"ml_confidence": float(pred.get("p_up", 0.5))}
    base_rate = pred.get("base_rate")
    score = pred.get("score")
    if base_rate is not None:
        kwargs["ml_base_rate"] = float(base_rate)
    if score is not None or base_rate is not None:
        # An abstention carries no directional information: score 0.0 (neutral).
        kwargs["ml_score"] = 0.0 if pred.get("abstained") else score
    return kwargs


def _ml_feature_indicators(indicators: dict) -> dict:
    """``REQUIRED_INDICATORS`` overlaid with the strategy's own indicator params.

    The feature contract (``core.ml.features``) computes every input it needs,
    but a strategy may tune e.g. ``rsi.period``; honouring that keeps the
    strategy's own indicator series (which its entry conditions read) identical
    to the ML feature series.
    """
    ranges = {k: dict(v) for k, v in REQUIRED_INDICATORS.items()}
    for name, cfg in (indicators or {}).items():
        if name in ranges and isinstance(cfg, dict):
            ranges[name].update(cfg)
    return ranges


class BacktestEngine:
    """Synchronous backtesting engine with ML prediction and signal fusion."""

    def __init__(self, config, strategy_engine, risk_manager, order_executor):
        self.config = config
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.order_executor = order_executor
        #: Base rate of the most recent ``_train_ml_model`` fit (P2 item 4) —
        #: lets a prediction be centred on the model's own rate without a
        #: persisted sidecar.
        self._ml_last_base_rate: float | None = None

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
                                  spread_overrides: dict | None = None,
                                  use_live_spread: bool = True,
                                  per_genome_ledger: bool | None = None):
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
            use_live_spread: When False the live order book is NOT queried and an
                unmapped symbol resolves straight to
                ``backtest.default_spread_pct``.  GA/WF evaluation passes False:
                a walk-forward over 2025 must not price 2025 fills with today's
                order book (and the run would otherwise depend on the network,
                which breaks reproducibility).  The resolved source of every
                symbol is reported under ``metrics["spread_sources"]``.
            per_genome_ledger: The GA-evaluation semantics.  Each strategy gets
                (a) its OWN position slots — ``max_open_trades // n_strategies``
                each, counted over its own positions — and (b) its OWN
                cash/equity sub-ledger, returned under ``per_strategy_equity``.
                Off by default so the hybrid/legacy parity contract (one shared
                cash balance, one shared position counter) is untouched.
        """
        t0 = time.time()
        if per_genome_ledger is None:
            per_genome_ledger = False

        # ── Per-run state, kept in ONE attribute ──
        # ``self._run_state`` is set at the start of every run and is what the
        # exit path / cost model read.  A single attribute (rather than several)
        # keeps a concurrent chunk from seeing another chunk's half-updated
        # tables: GA's parallel paths run one engine per thread/process, and the
        # previous per-field writes were a data race during a threaded batch.
        self._run_state = {"spread_pct": {}, "spread_sources": {},
                           "run_id": time.monotonic_ns()}
        _run_id = self._run_state["run_id"]

        # ── Cost model: derive the spread for the symbols THIS run uses ──
        # Override → live depth-derived → documented default (cost_model.py).
        # Resolved once here and pinned for the duration of the call so the trade
        # loop never does I/O and every symbol of the run gets its own spread
        # instead of the old "5 hardcoded pairs, else 0.03" guess.
        from core.backtest.cost_model import resolve_spreads as _resolve_spreads
        if use_live_spread:
            _resolved = _resolve_spreads(symbols, self.config, spread_overrides)
        else:
            # No network, no "today's book" for a historical window: an unmapped
            # symbol takes the configured default spread and the source is
            # recorded so the run's provenance shows it was not live-derived.
            _resolved = _resolve_spreads(symbols, self.config, spread_overrides,
                                         use_live=False)
        self._run_state["spread_pct"] = {
            sym: entry["spread_pct"] for sym, entry in _resolved.items()}
        self._run_state["spread_sources"] = {
            sym: entry["source"] for sym, entry in _resolved.items()}

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
                self.config.backtest_spread_pct = dict(
                    self._run_state.get("spread_pct") or {})
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

        #: The run's strategy configs, for `_ml_feature_contract` (the preload
        #: verification needs each strategy's `ml.features` subset — audit F4).
        self._current_strategies = list(strategy_configs)

        # Load historical data
        cache_dir = str(Path(self.config.data_dir) / "market")
        feeder = DataFeeder(cache_dir, symbols, intervals, date_start, date_end)
        feeder.load()

        #: P6-B: the feeder and the bar being replayed, for the per-bar
        #: quote-volume lookup the P6-A impact seam needs
        #: (:meth:`_recent_quote_volume_for`).  Both are set on this instance
        #: because ``close_position`` (core.backtest.trade_book) calls the cost
        #: function with only ``(entry, exit, qty, symbol)`` — the window has to
        #: come from the run, not from the call.
        self._feeder = feeder
        self._current_ts = None
        self._impact_bars_cache = None

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
        ml_predictions: dict[str, float] = {}  # "strategy_name|symbol" -> latest P(up)
        #: Training base rate per model key (P2 contract): persisted model
        #: metadata when available, else the label mean of the fitting window.
        #: ``None`` ⇒ unknown, so the fusion kernel keeps its 0.5 default.
        ml_base_rates: dict[str, float] = {}
        #: Latest signed score + abstention flag per key (P2 contract, item 4).
        ml_scores: dict[str, float] = {}
        ml_abstained: dict[str, bool] = {}
        #: ML feature matrix cache — one full-history matrix per feature
        #: configuration, sliced by timestamp (see ``_ml_features_up_to``).
        #: Without it the per-bar ML path re-ran ``compute_all`` (including the
        #: per-row Hurst loop) over the whole expanding history — O(bars²).
        _ml_feature_cache: dict[tuple, pd.DataFrame] = {}

        #: Indicator frames for the ML contract, keyed by the same feature
        #: configuration.  The strategy's own indicator cache holds only *its*
        #: indicators, while the contract also needs atr/hurst/swing_points/
        #: frac_diff — computed once per configuration here.
        _ml_indicator_cache: dict[tuple, pd.DataFrame] = {}

        def _ml_indicator_frame(sym: str, tf: str, indicators: dict) -> pd.DataFrame | None:
            """Full-history frame with every indicator the ML contract needs."""
            key = self._ml_feature_config(indicators, None)
            frame = _ml_indicator_cache.get(key)
            if frame is None:
                raw = feeder.get_all_data_for_symbol(sym, tf)
                if raw is None or len(raw) < 2:
                    return None
                frame = compute_all(raw, _ml_feature_indicators(indicators or {}))
                if len(_ml_indicator_cache) >= self._ML_FEATURE_CACHE_MAX:
                    _ml_indicator_cache.pop(next(iter(_ml_indicator_cache)), None)
                _ml_indicator_cache[key] = frame
            return frame

        # Collect (prediction, actual_return, label_threshold) pairs for the
        # post-hoc accuracy diagnostic.  The first element is the model's P(up).
        ml_eval_pairs: list[tuple[float, float, float]] = []
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
        # Audit F4: this used to `MLTrainer.load_model(path)` straight off disk,
        # with **no gate verdict and no `*_meta.json` / feature-schema check** — so
        # a pickle that the live path would refuse was scored by the backtest, and
        # the claim "no path can use an un-gated model" was false.  Every artefact
        # (LightGBM pickle, TFT, PatchTST) now goes through
        # :meth:`_verify_ml_model_sidecar`, which is the same verification
        # `MLPredictor.load_model` applies: sidecar present → gate verdict present
        # → `gate.allowed` → feature names equal the contract → schema hash equal.
        # Whatever fails is refused and logged, never loaded.
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
                        tft_path = tft_trainer.models_dir / f"{sym}_{strategy.name}_tft.pt"
                        ok, reason = self._verify_ml_model_sidecar(
                            f"{sym}_{strategy.name}", tft_path, "tft")
                        if ok:
                            model = tft_trainer.load(sym, strategy.name)
                            if model is not None:
                                ml_models[key] = model
                        else:
                            self._refuse_preloaded_ml(sym, strategy.name, tft_path, reason)
                    elif ml_engine == "patchtst" and patchtst_trainer is not None:
                        pt_path = (patchtst_trainer.models_dir
                                   / f"{sym}_{strategy.name}_patchtst.pt")
                        ok, reason = self._verify_ml_model_sidecar(
                            f"{sym}_{strategy.name}", pt_path, "patchtst")
                        if ok:
                            model = patchtst_trainer.load(sym, strategy.name)
                            if model is not None:
                                ml_models[key] = model
                        else:
                            self._refuse_preloaded_ml(sym, strategy.name, pt_path, reason)
                    else:
                        pkl_path = models_dir / f"{sym}_{strategy.name}_binary.pkl"
                        if pkl_path.exists():
                            ok, reason = self._verify_ml_model_sidecar(
                                f"{sym}_{strategy.name}", pkl_path, "binary")
                            if ok:
                                model = _disk_trainer.load_model(str(pkl_path))
                                if model is not None:
                                    ml_models[key] = model
                            else:
                                self._refuse_preloaded_ml(sym, strategy.name, pkl_path, reason)
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

        # ── Per-genome slots (isolated evaluation only) ──────────────────
        # ``max_positions`` above is a per-strategy allowance.  The entry loop
        # must therefore count only THIS strategy's open positions: the old
        # ``len(positions) >= max_positions`` test counted every strategy of the
        # chunk and then ``break``-ed, so the first genome of a 20-genome chunk
        # took the single slot and the other 19 never traded at all (measured:
        # 1 genome × 407 trades, 19 × 0 trades, identical best fitness for
        # generations 1-3).  Non-isolated runs keep the plain length test so the
        # hybrid/legacy parity contract is untouched.
        def _own_positions(s_name: str) -> int:
            if not per_genome_ledger:
                return len(positions)
            return sum(1 for p in positions.values()
                       if p.get("strategy_name") == s_name)

        # ── Per-genome sub-ledger (GA evaluation only) ───────────────────
        # One shared ``balance`` made every genome's PnL depend on its siblings'
        # fills (a genome that never traded still showed the chunk's losses).
        # When isolated, each strategy gets its own cash balance + equity series,
        # which is also what the fitness Sharpe/DSR/max-drawdown components need.
        ledger_balances: dict[str, float] = {}
        ledger_equity: dict[str, list[dict]] = {}
        if per_genome_ledger:
            for _s_cfg in strategy_configs:
                ledger_balances[_s_cfg.name] = float(initial_balance)
                ledger_equity[_s_cfg.name] = []

        def _strategy_balance(s_name: str) -> float:
            if per_genome_ledger:
                return ledger_balances.get(s_name, float(initial_balance))
            return balance

        def _credit_ledger(s_name: str, delta: float) -> None:
            """Add *delta* to genome *s_name*'s sub-ledger (isolated runs only)."""
            if per_genome_ledger:
                ledger_balances[s_name] = ledger_balances.get(
                    s_name, float(initial_balance)) + delta

        #: Ledger handed to every ``_close_position`` call (None when not isolated).
        __ledger = ledger_balances if per_genome_ledger else None

        def _close_and_credit(pos_key, pos, exit_price, ts, reason):
            """Close a position and route the realised cash to the right ledger.

            With per-genome ledgers the position's OWN sub-ledger is both the base
            and the target of the close.  Passing the shared ``balance`` variable
            worked only while one genome closed at a time: the forced end-of-run
            close loop walks every genome's positions, so genome #2 would be
            credited ``balance`` *plus* its own notional — every genome after the
            first gained the previous one's whole account (measured: a genome whose
            trades summed to -41.86 USDT ended at 90 390).
            """
            nonlocal balance
            new_balance = self._close_position(
                pos_key, pos, exit_price, ts, reason, trades, events,
                _strategy_balance(pos.get("strategy_name", "")), positions,
                per_matrix, ledger_balances=__ledger)
            if not per_genome_ledger:
                # Non-isolated runs keep the single shared cash balance.
                balance = new_balance
            return new_balance

        # ── Entry conditions: ONE evaluator for live, GA and backtest ────
        # ``StrategyConfig.entry_sides`` is the shared rule (P1.6): it delegates to
        # the OR kernel for ``condition_logic == "or"`` and requires every
        # condition for ``"and"``.  This engine used to keep its own inline AND
        # loop next to the kernel — two implementations of one rule that could
        # drift (a fix or a clamp applied in one and not the other), and the GA
        # scored a genome through a different path than the live engine trades it.
        # The call site is directly below; parity is pinned by
        # tests/test_gap_fixes.py (identical entry sets on a real cached symbol)
        # and tests/test_condition_logic.py.

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
        #: Hoisted price-slice cache for the exit checks (one entry per
        #: ``(timestamp, symbol, timeframe)`` — see the CHECK EXITS comment).
        _price_slice_cache: dict[tuple, pd.DataFrame] = {}

        # ---- Main Loop ----
        for slice_data in feeder:
            step += 1
            # Yield GIL periodically so asyncio event loop stays responsive
            if step % 200 == 0:
                time.sleep(0)
            ts = slice_data["timestamp"]
            # P6-B: the bar whose close the P6-A impact window ends at.
            self._current_ts = ts
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
            def _record_ml_prediction(raw, key: str, df_tf, sliced, ts) -> None:
                """Publish one raw model output under the P2 prediction contract.

                Normalises ``raw`` (bare float / ``(direction, confidence)`` /
                dict), stores the *model's* P(up), base rate, signed score and
                abstention flag, and queues the pair for the post-loop
                diagnostic.  Called from all three model paths so none of them
                can drift back to "confidence only".
                """
                pred = _normalise_ml_prediction(raw)
                ml_conf = float(pred.get("p_up", 0.5))
                ml_predictions[key] = ml_conf
                ml_base_rates[key] = pred.get("base_rate")
                ml_scores[key] = pred.get("score")
                ml_abstained[key] = bool(pred.get("abstained", False))
                # Defer the accuracy comparison to post-loop (purely diagnostic).
                fwd = tf_params["forward"]
                th = tf_params["threshold"]
                future_df = df_tf[df_tf.index > ts]
                if len(future_df) >= fwd:
                    cur_close = float(sliced.iloc[-1]["close"])
                    fut_close = float(future_df.iloc[fwd - 1]["close"])
                    ret = (fut_close - cur_close) / cur_close
                    ml_eval_pairs.append((ml_conf, ret, th))

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
                    # Phase P2 item 6: ``key not in ml_models`` used to be part of
                    # this test, which meant every model was fitted ONCE (on the
                    # first step of its rotation slot) and then frozen for the
                    # whole backtest while the module comments claimed round-robin
                    # retraining.  The rotation schedule below is the retraining
                    # schedule: each key refits when the step lands on its slot.
                    should_retrain = (
                        not skip_ml_training and
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
                                    feature_list=strategy.ml_config.features if strategy.ml_config else None,
                                    indicators=strategy.indicators)
                                if model is not None:
                                    ml_models[key] = model
                                    # model updated (round-robin)

                        model = ml_models.get(key)
                        if model is not None:
                            try:
                                raw = self._predict_patchtst(
                                    patchtst_trainer, model, sliced,
                                    indicators=strategy.indicators)
                                if raw is not None:
                                    _record_ml_prediction(raw, key, df_tf, sliced, ts)
                            except Exception:
                                pass

                    # ── TFT path ──
                    elif ml_engine == "tft" and tft_trainer is not None:
                        if should_retrain:
                            if len(sliced) >= 150:
                                model = self._train_tft_model(
                                    tft_trainer, sliced, sym, primary_tf,
                                    indicators=strategy.indicators)
                                if model is not None:
                                    ml_models[key] = model
                                    # model updated (round-robin)

                        model = ml_models.get(key)
                        if model is not None:
                            try:
                                raw = self._predict_tft(
                                    tft_trainer, model, sliced,
                                    indicators=strategy.indicators)
                                if raw is not None:
                                    _record_ml_prediction(raw, key, df_tf, sliced, ts)
                            except Exception:
                                pass

                    # ── LightGBM path ──
                    else:
                        ml_full_ind = _ml_indicator_frame(sym, primary_tf, strategy.indicators)
                        if should_retrain:
                            model = self._train_ml_model(
                                sliced, tf_params,
                                feature_list=strategy.ml_config.features if strategy.ml_config else None,
                                indicators=strategy.indicators,
                                full_df=ml_full_ind, ts=ts, cache=_ml_feature_cache)
                            if model is not None:
                                ml_models[key] = model
                                # model updated (round-robin)

                        model = ml_models.get(key)
                        if model is not None:
                            try:
                                raw = self._predict_ml(
                                    model, sliced,
                                    feature_list=strategy.ml_config.features if strategy.ml_config else None,
                                    indicators=strategy.indicators,
                                    full_df=ml_full_ind, ts=ts, cache=_ml_feature_cache)
                                if raw is not None:
                                    if raw.get("base_rate") is None and raw.get("score") is None:
                                        # The engine's own fit has no sidecar: use
                                        # the base rate of the window this model was
                                        # fitted on, or the persisted model metadata
                                        # when it exists.  A model that publishes a
                                        # neutral P(up) is left neutral — attaching a
                                        # base rate to it must not turn "no call"
                                        # into a directional vote.
                                        rate = (getattr(self, "_ml_last_base_rate", None)
                                                or self._meta_base_rate(sym, strategy.name))
                                        if rate is not None and float(raw.get("p_up", 0.5)) != 0.5:
                                            raw["base_rate"] = float(rate)
                                    _record_ml_prediction(raw, key, df_tf, sliced, ts)
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
                                    # Credited to the position's own genome ledger
                                    # (isolated runs) or to the shared balance.
                                    if per_genome_ledger:
                                        _credit_ledger(
                                            strategy.name, reduce_amount + reduce_pnl)
                                    else:
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
            # `_price_slice_cache` is hoisted OUT of this loop (it used to be
            # allocated inside the per-timestamp body, so it died every bar and
            # the comment below claimed a fix it did not deliver).  Its key
            # includes `ts`, so a hit can only ever return the slice for the bar
            # being evaluated — the same object `df_tf[df_tf.index <= ts]`
            # produced, which is why results are unchanged.  Measured on a
            # 2-symbol × 2-timeframe feed (BTCUSDT/ETHUSDT 1h/4h): the slice count
            # drops from `positions × bars` to one per (bar, symbol, timeframe),
            # and the slice itself is now a `searchsorted` positional cut
            # (78 µs against 195 µs for the boolean mask on the 3 875-row frame).
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
                    _slice_key = (ts, sym, pos_tf)
                    df_slice = _price_slice_cache.get(_slice_key)
                    if df_slice is None:
                        df_tf = feeder.get_all_data_for_symbol(sym, pos_tf)
                        # `searchsorted` + `iloc` is the positional form of
                        # `df_tf[df_tf.index <= ts]` and was bit-identical on the
                        # shipped feed (0 mismatches over both timeframes).
                        _cut = int(df_tf.index.searchsorted(ts, side="right"))
                        df_slice = df_tf.iloc[:_cut]
                        _price_slice_cache[_slice_key] = df_slice
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
                            _close_and_credit(pos_key, pos, sl_price, ts, "stop_loss")
                            continue

                    # ---- TAKE-PROFIT CHECK ----
                    tp_levels = pos.get("take_profits", [])
                    for tp_price, tp_pct in tp_levels:
                        if (pos["side"] == "long" and price_now >= tp_price) or \
                           (pos["side"] == "short" and price_now <= tp_price):
                            _close_and_credit(
                                pos_key, pos, tp_price, ts, f"tp_{int(tp_pct*100)}pct")
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
                            _close_and_credit(pos_key, pos, price_now, ts, "max_hold")
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
                                _close_and_credit(
                                    pos_key, pos, exit_price, ts, "indicator")
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
                    if _own_positions(strategy.name) >= max_positions:
                        # This genome has used up its own slots — move on to the
                        # NEXT genome instead of starving the rest of the chunk.
                        # (The old `break` fired on the first genome that filled
                        # the shared counter, leaving every later genome with 0
                        # trades and a -999-class fitness.)
                        break
                    _strat_balance = _strategy_balance(strategy.name)

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
                    # One evaluator for every path (live engine, GA scoring and
                    # this backtest): ``entry_sides`` applies the genome's
                    # ``condition_logic`` gene itself, so there is no second
                    # implementation left to drift from it.
                    long_active, short_active = strategy.entry_sides(df_primary)

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

                    # P2 item 4: feed the *signed* score, centred on the model's
                    # own base rate, whenever the ML path published one.  With no
                    # base rate and no score (the legacy contract) the kernel
                    # falls back to its historical (conf − 0.5) × 2 term, so
                    # non-ML and legacy runs stay bit-identical.
                    fusion_ml = {
                        "ml_confidence": ml_conf,
                        "ml_base_rate": ml_base_rates.get(ml_key),
                        "ml_score": ml_scores.get(ml_key),
                    }
                    if fusion_ml["ml_base_rate"] is None and fusion_ml["ml_score"] is None:
                        fusion_ml = {"ml_confidence": ml_conf}

                    final_score = fuse_signals(
                        indicator_signal=indicator_signal,
                        news_sentiment=None,  # backtest mode — no historical news
                        w_indicator=w_ind,
                        w_ml=w_ml,
                        w_news=w_news,
                        ml_enabled=ml_enabled,
                        strategy_ml_weight=strategy.ml_config.weight if ml_enabled else None,
                        **fusion_ml,
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
                    # Sizing uses THIS genome's ledger balance when isolated: with
                    # one shared balance a genome's position size (and therefore
                    # its fills) depended on its siblings' cash.
                    qty, risk_amount = sizer.calculate_position_size(
                        _strat_balance, price, "satellite",
                        volatility_expanding=vol_expanding)
                    if qty <= 0:
                        continue

                    # ── Risk-based position sizing (Kelly-lite) ──
                    # Tighter stop → larger position for same risk budget
                    re = strategy.risk_exit
                    if re is not None:
                        risk_capital = _strat_balance * 0.01  # risk 1% of capital per trade
                        sl_dist = re.stop_loss_pct / 100.0
                        qty_risk = risk_capital / (price * sl_dist)
                        # Blend: use risk-based if it's more conservative, else keep base
                        qty = min(qty, qty_risk) if qty_risk > 0 else qty

                    # Check max position size
                    max_amount = _strat_balance * (self.config.hard_limits.max_position_size_pct / 100)
                    if risk_amount > max_amount:
                        risk_amount = max_amount
                        qty = risk_amount / price

                    amount_usdt = qty * price
                    if amount_usdt > _strat_balance * 0.95:
                        continue  # don't use >95% of balance

                    pos_counter += 1
                    trade_group = f"bt_{pos_counter}_{int(ts.timestamp())}"
                    if per_genome_ledger:
                        _credit_ledger(strategy.name, -amount_usdt)
                    else:
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
            # Release the previous bar's price slices: the key carries `ts`, so
            # entries from earlier bars can never be hit again.  Without this the
            # hoisted cache would hold one frame per (bar × symbol × timeframe).
            if _price_slice_cache:
                _price_slice_cache.clear()
            invested = sum(p.get("amount_usdt", 0) for p in positions.values())
            if per_genome_ledger:
                # One equity point per genome + an aggregate point.  The aggregate
                # is the sum of the sub-ledgers, so calculate_metrics() (used for
                # the run's headline numbers) still sees a coherent portfolio.
                agg_balance = 0.0
                for _s_name in ledger_balances:
                    _inv = sum(p.get("amount_usdt", 0) for p in positions.values()
                               if p.get("strategy_name") == _s_name)
                    _bal = ledger_balances[_s_name]
                    ledger_equity[_s_name].append({
                        "time": str(ts), "equity": round(_bal + _inv, 2),
                        "balance": round(_bal, 2), "invested": round(_inv, 2),
                    })
                    agg_balance += _bal
                equity_curve.append({
                    "time": str(ts), "equity": round(agg_balance + invested, 2),
                    "balance": round(agg_balance, 2), "invested": round(invested, 2),
                })
            else:
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
                _close_and_credit(pos_key, pos, final_price, last_ts, "end_of_backtest")
            # Reflect the realised cash in the final equity point.
            if per_genome_ledger:
                for _s_name in ledger_balances:
                    if ledger_equity[_s_name]:
                        ledger_equity[_s_name][-1]["balance"] = round(
                            ledger_balances[_s_name], 2)
                        ledger_equity[_s_name][-1]["invested"] = 0.0
                        ledger_equity[_s_name][-1]["equity"] = round(
                            ledger_balances[_s_name], 2)
                equity_curve[-1]["balance"] = round(sum(ledger_balances.values()), 2)
                equity_curve[-1]["invested"] = 0.0
                equity_curve[-1]["equity"] = round(sum(ledger_balances.values()), 2)
            else:
                equity_curve[-1]["balance"] = round(balance, 2)
                equity_curve[-1]["invested"] = 0.0
                equity_curve[-1]["equity"] = round(balance, 2)

        # Use last equity value (includes open position value), not just cash balance
        final_balance = equity_curve[-1]["equity"] if equity_curve else balance

        # ── Post-hoc ML accuracy evaluation (no look-ahead bias, item 1) ──
        # Accuracy is computed from the collected eval pairs after the loop,
        # ensuring it is purely diagnostic and doesn't affect trading decisions.
        #
        # Phase P2 handed the Lead the corrected diagnostic
        # (``core.ml.credibility.ml_accuracy_neutral_abstention``): the inline
        # loop that used to live here scored a 0.38–0.62 "neutral" prediction as
        # a *bearish* call (``conf < 0.5``), which biased the number toward the
        # market's direction — measured band share on the deployed model is
        # 25.7 % / 24.1 %.  A neutral prediction is now an **abstention**:
        # excluded from accuracy and reported separately as ``coverage_pct`` /
        # ``neutral_pct``.
        from core.ml.credibility import ml_accuracy_neutral_abstention

        _ml_diag = ml_accuracy_neutral_abstention(
            [(conf, ret) for conf, ret, _th in ml_eval_pairs],
            [ret for _conf, ret, _th in ml_eval_pairs],
            threshold=0.005)
        ml_total = int(_ml_diag["n_predictions"])

        metrics = calculate_metrics(trades, equity_curve, initial_balance, final_balance)
        metrics["ml_accuracy_pct"] = float(_ml_diag["accuracy_pct"])
        metrics["ml_coverage_pct"] = float(_ml_diag["coverage_pct"])
        metrics["ml_neutral_pct"] = float(_ml_diag["neutral_pct"])
        metrics["ml_scored"] = int(_ml_diag["n_scored"])
        metrics["ml_predictions"] = ml_total
        metrics["ml_abstained"] = int(sum(1 for v in ml_abstained.values() if v))
        metrics["runtime_seconds"] = round(time.time() - t0, 1)

        # ── Buy & hold benchmark of the SAME window and symbols ──
        # GA selection used to score pure market drift as alpha (a synthetic
        # random walk scored Sharpe 13.7).  The equal-weighted buy & hold return
        # of the run's window is the beta baseline the fitness subtracts.
        # Cached per (data_dir, symbols, window): a GA chunk reuses the identical
        # benchmark for every genome, so it is computed once per window.
        buy_hold_pct = None
        try:
            if equity_curve:
                first_ts = pd.Timestamp(equity_curve[0]["time"])
                last_ts_bh = pd.Timestamp(equity_curve[-1]["time"])
                _bh_key = (str(getattr(self.config, "data_dir", "")),
                           tuple(symbols), first_ts, last_ts_bh)
                _bh_cache = self._run_state.setdefault("buy_hold_cache", {})
                if _bh_key in _bh_cache:
                    buy_hold_pct = _bh_cache[_bh_key]
                else:
                    rets = []
                    for sym in symbols:
                        df_bh = feeder.get_all_data_for_symbol(sym, "1h")
                        if df_bh is None or len(df_bh) < 2:
                            continue
                        window = df_bh[(df_bh.index >= first_ts) & (df_bh.index <= last_ts_bh)]
                        if len(window) < 2:
                            window = df_bh[df_bh.index <= last_ts_bh]
                        if len(window) < 2:
                            continue
                        first_close = float(window.iloc[0]["close"])
                        last_close = float(window.iloc[-1]["close"])
                        if first_close > 0:
                            rets.append((last_close - first_close) / first_close)
                    if rets:
                        buy_hold_pct = sum(rets) / len(rets) * 100.0
                    _bh_cache[_bh_key] = buy_hold_pct
        except Exception as e:  # never let the benchmark break a backtest
            logger.debug(f"Buy&hold benchmark unavailable: {e}")
        metrics["buy_hold_pct"] = round(buy_hold_pct, 4) if buy_hold_pct is not None else None
        metrics["spread_sources"] = dict(
            (getattr(self, "_run_state", {}) or {}).get("spread_sources") or {})

        # ── Per-genome ledgers (isolated GA evaluation) ──
        # ``trades`` is append-only and therefore chronological, so slicing it per
        # strategy reconstructs exactly that genome's trade list.  Fitness needs
        # the trades (mean win for the shrunk profit factor) and the equity series
        # (Sharpe / DSR / max drawdown) *per genome* — the shared ones are the
        # sum over all genomes of the chunk and cannot score an individual.
        per_strategy_equity = None
        if per_genome_ledger:
            per_strategy_equity = {
                s_name: {
                    "trades": [t for t in trades if t.get("strategy") == s_name],
                    "equity_curve": ledger_equity.get(s_name, []),
                    "initial_balance": initial_balance,
                    "final_balance": round(ledger_balances.get(s_name, initial_balance), 2),
                    "buy_hold_pct": metrics["buy_hold_pct"],
                }
                for s_name in ledger_balances
            }

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
            "per_strategy_equity": per_strategy_equity,
            "monte_carlo": mc_result,
        }

    def _close_position(self, pos_key, pos, exit_price, ts, reason, trades, events,
                         balance, positions, per_matrix,
                         ledger_balances: dict | None = None):
        """Close a position and record the trade. Used for SL/TP/indicator exits.

        Thin wrapper over the shared implementation in
        :mod:`core.backtest.trade_book` (previously a copy-paste clone that the
        hybrid engine duplicated).

        When *ledger_balances* is given (isolated GA evaluation) the realised
        cash is credited to ``pos["strategy_name"]``'s own sub-ledger, so one
        genome's PnL never touches another's.  The returned value is the
        ledger balance of that genome, or the shared balance otherwise.
        """
        new_balance = close_position(
            pos_key, pos, exit_price, ts, reason, trades, balance, positions, per_matrix,
            cost_fn=lambda ep, xp, q, s: apply_trading_costs(
                ep, xp, q, s, self.config,
                overrides=(getattr(self, "_run_state", {}) or {}).get("spread_pct"),
                # ── P6-A impact seam, fed per bar (P6-B) ──
                # ``recent_quote_volume`` is the position's own timeframe summed
                # over ``risk.liquidity.lookback_bars`` bars ending at this close
                # (the cache's `quote_volume` column when the file has it, the
                # documented `volume × close` proxy otherwise).  It is **inert**:
                # `apply_trading_costs` short-circuits to the pre-P6 arithmetic
                # when `risk.liquidity.impact_k <= 0` (the shipped value) or when
                # the window is unknown, and `_current_ts` is only used for that
                # lookup, so a k=0 run is bit-identical to before.
                recent_quote_volume=self._recent_quote_volume_for(pos)),
            events=events,
        )
        if ledger_balances is not None:
            s_name = pos.get("strategy_name", "")
            ledger_balances[s_name] = new_balance
            return new_balance
        return new_balance

    def _recent_quote_volume_for(self, pos) -> float:
        """Quote (USDT) notional behind one close, on the position's own timeframe.

        The backtest half of the P6-A seam (P6-B): until now the engine could not
        pass ``recent_quote_volume`` to
        :func:`core.backtest.cost_model.apply_trading_costs`, so the impact term
        could never price a backtest trade.  The window is
        ``risk.liquidity.lookback_bars`` bars ending at the **current** bar
        (:attr:`_current_ts`, set once per feeder step), measured with
        :func:`core.backtest.cost_model.recent_quote_volume_from_bars` so the
        backtest and the live sizer read the same number.

        ``0.0`` (the documented "unknown ⇒ no impact") on any failure — a missing
        feed, a position without a timeframe, a data hiccup.  The lookup runs
        **only** when the impact term is on: ``impact_k <= 0`` (the shipped value)
        returns immediately, so a default run does not even slice a frame.
        """
        if self._impact_bars() <= 0:
            return 0.0
        try:
            ts = getattr(self, "_current_ts", None)
            if ts is None:
                return 0.0
            if isinstance(pos, dict):
                symbol = pos.get("symbol")
                interval = pos.get("timeframe") or "1h"
            else:
                symbol, interval = pos, "1h"
            if not symbol:
                return 0.0
            store = getattr(self, "_feeder", None)
            if store is None or not hasattr(store, "get_all_data_for_symbol"):
                return 0.0
            frame = store.get_all_data_for_symbol(symbol, interval)
            if frame is None or len(frame) == 0:
                return 0.0
            cut = int(frame.index.searchsorted(ts, side="right"))
            if cut <= 0:
                return 0.0
            from core.backtest.cost_model import recent_quote_volume_from_bars

            return recent_quote_volume_from_bars(
                frame.iloc[:cut], lookback_bars=self._impact_bars())
        except Exception:  # a cost lookup must never take a run down
            return 0.0

    def _impact_bars(self) -> int:
        """``risk.liquidity.lookback_bars`` when the impact term is enabled, else 0.

        One cached answer per run (the config cannot change mid-run): ``0`` means
        "the impact seam is off", which is the value that makes
        :meth:`_recent_quote_volume_for` return without doing any work.
        """
        cached = getattr(self, "_impact_bars_cache", None)
        if cached is not None:
            return cached
        from core.backtest.cost_model import (_liquidity_impact_params,
                                              liquidity_lookback_bars)

        k, _exponent = _liquidity_impact_params(self.config)
        value = liquidity_lookback_bars(self.config) if k > 0.0 else 0
        self._impact_bars_cache = value
        return value

    # ---- ML Helpers ----
    # ── ML helpers ────────────────────────────────────────────────────
    #
    # The feature contract (``core.ml.features``) is the single source of truth
    # for both the live and the backtest ML path (plan §二 P2.4 / §一 "47 vs 40
    # features").  The engine used to carry its own ``_ML_INDICATORS`` (rsi /
    # macd / bollinger / adx only) and call ``compute_features`` with
    # ``feature_list=None``; with the P2 contract that raises
    # ``FeatureContractError`` because the hurst / swing-point / fractional-diff
    # inputs were never computed.  The indicator config now comes from
    # ``REQUIRED_INDICATORS`` and the column list from ``FEATURE_NAMES`` (39), so
    # the matrix the engine trains on is the matrix the live predictor scores.

    @property
    def _ML_INDICATORS(self) -> dict:
        """Indicator config the ML feature contract needs (kept for compatibility)."""
        return REQUIRED_INDICATORS

    #: Upper bound on cached ML feature matrices per run (FIFO eviction).
    _ML_FEATURE_CACHE_MAX = 16

    @staticmethod
    def _ml_feature_config(indicators: dict, feature_list) -> tuple:
        """Hashable cache key for one ML feature configuration."""
        import json as _json

        cols = list(feature_list) if feature_list else list(FEATURE_NAMES)
        return (_json.dumps(indicators or {}, sort_keys=True, ensure_ascii=True),
                tuple(cols))

    def _ml_features_up_to(self, df: pd.DataFrame, ts, *,
                           feature_list=None, indicators=None,
                           full_df: pd.DataFrame | None = None,
                           cache: dict | None = None) -> pd.DataFrame:
        """Feature matrix rows ``≤ ts`` of *df* — computed once per bar, cached.

        Before this, every bar re-ran ``compute_all`` (which includes the
        per-row Hurst/swing-point loops) over the whole expanding history, i.e.
        O(bars²) work per backtest.  The full-history matrix depends only on the
        price history, so it is computed once per configuration and sliced by
        timestamp; only the last row is new at each step.

        ``full_df`` is the unsliced indicator-bearing frame the caller already
        holds (the strategy has its own indicator cache in the engine); when it
        is given, no second ``compute_all`` is done here.
        """
        key = self._ml_feature_config(indicators, feature_list)
        matrix = None if cache is None else cache.get(key)
        if matrix is None:
            from core.strategy.indicators import compute_all
            from core.ml.features import compute_features as _compute_features

            source = full_df if full_df is not None else df
            if full_df is None:
                source = compute_all(source.copy(),
                                     _ml_feature_indicators(indicators or {}))
            cols = list(feature_list) if feature_list else list(FEATURE_NAMES)
            matrix = _compute_features(source, cols)
            if cache is not None:
                if len(cache) >= self._ML_FEATURE_CACHE_MAX:
                    cache.pop(next(iter(cache)), None)
                cache[key] = matrix
        if ts is None:
            return matrix
        return matrix[matrix.index <= ts]

    def _meta_base_rate(self, symbol: str, strategy_name: str) -> float | None:
        """Training base rate from the persisted model sidecar, when present."""
        try:
            from core.ml.trainer import MLTrainer
            meta = MLTrainer(str(self.config.data_dir)).load_meta(
                symbol, strategy_name, "binary")
        except Exception:
            return None
        if not meta:
            return None
        rate = meta.get("train_base_rate", meta.get("base_rate"))
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            return None
        return rate if 0.0 < rate < 1.0 else None

    # ── audit F4: an artefact may only be used if its sidecar proves the gate ──

    def _ml_feature_contract(self, strategy_name: str) -> list[str]:
        """Feature contract for ``strategy_name`` — ``ml.feature_list`` or canonical."""
        explicit = getattr(self.config, "ml_feature_list", None)
        if explicit:
            return [str(c) for c in explicit]
        for strategy in (self._current_strategies or []):
            if getattr(strategy, "name", None) != strategy_name:
                continue
            ml = getattr(strategy, "ml_config", None)
            features = getattr(ml, "features", None) if ml is not None else None
            if features:
                return [str(c) for c in features]
        return list(FEATURE_NAMES)

    def _verify_ml_model_sidecar(self, stem: str, model_path,
                                 model_type: str = "binary") -> tuple[bool, str]:
        """``(ok, reason)`` for a **preloaded** model artefact (audit F4).

        The verification is deliberately the same five steps
        ``core.ml.predictor.MLPredictor.load_model`` performs on the live path, so
        "web backtest routes the preload through the same verification the
        predictor uses" is a fact about the code and not a claim:

        1. a ``*_meta.json`` sidecar exists next to the pickle (its absence is a
           refusal, because the gate verdict cannot be reconstructed),
        2. the sidecar carries a ``gate`` verdict,
        3. ``gate["allowed"]`` is true — i.e. the model really passed
           :func:`core.ml.credibility.credibility_gate` when it was trained,
        4. the sidecar's ``feature_names`` equal the contract the engine will score
           with (positional scoring on a different contract is silent corruption),
        5. the sidecar's ``feature_schema_hash`` is **present and equal** to
           :func:`core.ml.features.feature_schema_hash` of that contract — a
           missing hash is a refusal, not a pass (re-audit finding 5: the check
           used to be ``if stored and ...``, so a sidecar with ``gate.allowed``
           and matching names but no hash was accepted).

        It deliberately does **not** instantiate a predictor (no market-data
        provider exists at this point in a backtest) and deliberately does not
        re-run the gate: the numbers it would need are not persisted, and the
        persisted verdict is what the live path trusts too.
        """
        from pathlib import Path as _Path
        path = _Path(model_path)
        if not path.exists():
            return False, f"no artefact at {path.name}"
        meta_path = path.with_name(path.stem + "_meta.json")
        if not meta_path.exists():
            return False, ("no metadata sidecar (*_meta.json) — cannot verify the "
                           "OOS gate")
        import json
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception as e:
            return False, f"metadata sidecar unreadable ({e})"
        if not isinstance(meta, dict):
            return False, "metadata sidecar is not an object"
        gate = meta.get("gate") or {}
        if not gate:
            return False, "metadata has no gate verdict"
        if not gate.get("allowed"):
            return False, f"gate refused the model: {gate.get('reason', 'gate failed')}"
        try:
            from core.ml.features import (feature_schema_hash,
                                          feature_schema_mismatch_reason)
            strategy_name = stem.split("_", 1)[1] if "_" in stem else stem
            expected = self._ml_feature_contract(strategy_name)
            names = list(meta.get("feature_names") or [])
            if names != list(expected):
                have = set(expected)
                missing = [c for c in names if c not in have]
                extra = [c for c in expected if c not in set(names)]
                return False, (f"feature contract mismatch: sidecar has {len(names)} "
                               f"features, engine scores {len(expected)} "
                               f"(missing_from_engine={missing[:4]} "
                               f"extra_in_engine={extra[:4]})")
            stored = meta.get("feature_schema_hash")
            current = feature_schema_hash(expected)
            # Re-audit finding 5: `if stored and ...` accepted a sidecar that had
            # `gate.allowed` and matching feature *names* but **no schema hash**
            # (the audit's e2e loaded 2 of 7 artefacts that way).  A hash that is
            # absent cannot be compared, so it is a refusal — the same rule the
            # mismatch below already follows, and the same reason text
            # `MLPredictor.load_model` produces (P6-B shares it so the live and
            # backtest refusals cannot drift).
            if not stored or str(stored) != current:
                return False, feature_schema_mismatch_reason(stored, current)
        except Exception as e:
            return False, f"feature contract could not be verified ({e})"
        return True, "verified"

    @staticmethod
    def _refuse_preloaded_ml(symbol: str, strategy_name: str, path, reason: str) -> None:
        """One first-class log line per refused preload — never a silent skip."""
        logger.warning(
            f"ML preload REFUSED {symbol}/{strategy_name} ({getattr(path, 'name', path)}): "
            f"{reason}")

    @staticmethod
    def _ml_matrix_for_model(feature_df: pd.DataFrame, model):
        """Feature matrix in the *shape the model was fitted with*.

        Root cause of the 1 604 sklearn warnings in the ML path: LightGBM's
        scikit-learn wrapper exposes ``feature_names_in_`` as soon as it is fitted
        — ``Column_0…Column_N`` even when it was fitted on a bare ndarray
        (``lightgbm/sklearn.py``: ``feature_names_in_`` is a property over the
        booster's names).  Scikit-learn's ``validate_data(..., reset=False)`` then
        sees "fitted with names, X has none" on **every** ``predict_proba`` and
        warns — once per bar, i.e. thousands of times per backtest, drowning the
        real warnings.

        The fix is to make the two ends agree instead of silencing anything:

        * the model reports names (LightGBM, or XGBoost fitted on a DataFrame) →
          hand it a DataFrame carrying **those exact names**, so a model fitted on
          the 39-column contract is predicted on the contract's columns;
        * the model reports no names (XGBoost or a stub fitted on an ndarray) →
          hand it an array, exactly as before.

        The values and their order are untouched either way, so predictions are
        numerically identical to the previous (warning-producing) call.
        """
        names = getattr(model, "feature_names_in_", None)
        try:
            names = None if names is None else [str(n) for n in names]
        except TypeError:
            names = None
        if names and len(names) == feature_df.shape[1]:
            out = feature_df.copy()
            out.columns = names
            return out
        return feature_df.values.astype(float)

    def _train_ml_model(self, df: pd.DataFrame, tf_params: dict = None,
                         feature_list: list[str] | None = None,
                         indicators: dict | None = None,
                         full_df: pd.DataFrame | None = None,
                         ts=None, cache: dict | None = None):
        """Train a LightGBM classifier (XGBoost fallback) with timeframe-appropriate labels.

        Each strategy's primary timeframe gets its own model with matched
        forward_periods and threshold (e.g., 1m→20 periods, 1h→4 periods).
        Tries LightGBM first; falls back to XGBoost if LightGBM is unavailable.

        Phase P2 added ``eval_set`` early stopping to the shared trainer; the
        engine's own fit uses the same chronological tail hold-out so a retrain
        stops when the held-out log-loss stops improving instead of always
        running the full ``n_estimators``.
        """
        if tf_params is None:
            tf_params = {"forward": 4, "threshold": 0.005, "min_candles": 100}
        min_candles = tf_params.get("min_candles", 100)
        min_samples = max(40, min_candles // 3)
        if len(df) < min_candles:
            return None
        try:
            from core.ml.features import create_binary_label

            feature_df = self._ml_features_up_to(
                df, ts, feature_list=feature_list, indicators=indicators,
                full_df=full_df, cache=cache)
            labels = create_binary_label(
                df, forward_periods=tf_params["forward"],
                threshold=tf_params["threshold"])

            common_idx = feature_df.index.intersection(labels.dropna().index)
            if len(common_idx) < min_samples:
                return None
            # Fit on a **named** frame: the contract's column names travel with
            # the model (``feature_names_in_``), which is what the predict side
            # below re-attaches.  The values are the same float64 matrix the
            # previous ``.values.astype(float)`` produced, in the same order, so
            # the fitted trees are identical.
            X = feature_df.loc[common_idx].astype(float)
            y = labels.loc[common_idx].values.astype(int)

            n_up = int(y.sum())
            n_down = len(y) - n_up
            scale_pos_weight = max(1.0, n_down / max(n_up, 1))

            # Training base rate of THIS fitting window: the fusion kernel centres
            # the signed score on it (P2 item 4), so a P(up) above the label rate
            # is a bullish contribution and one below it is bearish.
            base_rate = (n_up / len(y)) if len(y) else None
            self._ml_last_base_rate = (
                float(base_rate) if base_rate and 0.0 < base_rate < 1.0 else None)

            # Chronological hold-out for early stopping (never random: the label
            # has a forward window, so a shuffled split would leak).
            cut = max(1, int(len(X) * 0.8))
            X_tr, y_tr = X[:cut], y[:cut]
            X_es, y_es = X[cut:], y[cut:]

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
                fit_kwargs: dict = {}
                if len(X_es) > 0 and len(set(y_es.tolist())) > 1 and len(X_tr) >= 40:
                    fit_kwargs["eval_set"] = [(X_es, y_es)]
                    fit_kwargs["callbacks"] = [lgb.early_stopping(20, verbose=False)]
                model.fit(X_tr if fit_kwargs else X, y_tr if fit_kwargs else y,
                          **fit_kwargs)
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
            fit_kwargs = {}
            if len(X_es) > 0 and len(set(y_es.tolist())) > 1 and len(X_tr) >= 40:
                fit_kwargs["eval_set"] = [(X_es, y_es)]
                fit_kwargs["verbose"] = False
            model.fit(X_tr if fit_kwargs else X, y_tr if fit_kwargs else y,
                      **fit_kwargs)
            return model
        except Exception as e:
            logger.debug(f"ML train failed: {e}")
            return None

    def _predict_ml(self, model, df: pd.DataFrame,
                    feature_list: list[str] | None = None,
                    indicators: dict | None = None,
                    full_df: pd.DataFrame | None = None,
                    ts=None, cache: dict | None = None) -> dict | None:
        """Predict P(price rises ≥ the timeframe threshold) for the latest bar.

        Returns the P2 prediction contract — ``{p_up, base_rate, score,
        abstained}`` — or ``None`` when the data is insufficient.  ``base_rate``
        and ``score`` are left ``None`` here: the caller fills ``base_rate`` in
        from :attr:`_ml_last_base_rate` / the persisted model metadata, and
        ``fuse_signals`` then derives the signed score from ``p_up`` against it.

        The 0.38–0.62 "neutral band" is **not** applied here any more: with a
        real base rate the band is meaningless (a 0.6 P(up) against a 0.62 base
        rate is a bearish call).  ``_normalise_ml_prediction`` keeps the band for
        callers that hand over a bare legacy confidence float.
        """
        if len(df) < 50:
            return None
        try:
            feature_df = self._ml_features_up_to(
                df, ts, feature_list=feature_list, indicators=indicators,
                full_df=full_df, cache=cache)
            if len(feature_df) == 0:
                return None
            # Names must match what the fit saw, or scikit-learn warns on every
            # bar and — worse — a model really fitted with another column set is
            # never told.  See ``_ml_matrix_for_model``.
            X = self._ml_matrix_for_model(feature_df.iloc[-1:], model)
            proba = model.predict_proba(X)
            if proba.shape[1] >= 2:
                return {"p_up": float(proba[0][1]), "base_rate": None,
                        "score": None, "abstained": False}
            return {"p_up": 0.5, "base_rate": None, "score": None,
                    "abstained": False}
        except Exception as e:
            logger.debug(f"ML predict failed: {e}")
            return None

    # ── TFT helpers ──────────────────────────────────────────────────

    def _train_tft_model(self, tft_trainer, df: pd.DataFrame, symbol: str,
                         interval: str, max_train_rows: int = 5000,
                         indicators: dict | None = None):
        """Train a TFT model on sliced DataFrame (walk-forward safe).

        Caps training data to *max_train_rows* most recent candles so that
        retrain time stays constant regardless of how far the backtest has
        progressed.  Features come from the shared contract (``FEATURE_NAMES``).
        """
        try:
            from core.strategy.indicators import compute_all
            from core.ml.features import compute_features as _cf, create_regression_label

            # Cap to recent data to keep training time constant
            if len(df) > max_train_rows:
                df = df.iloc[-max_train_rows:]

            df_ind = compute_all(df.copy(), _ml_feature_indicators(indicators or {}))
            X = _cf(df_ind, list(FEATURE_NAMES))
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

    def _predict_tft(self, tft_trainer, model, df: pd.DataFrame,
                     indicators: dict | None = None) -> dict | None:
        """Predict with TFT — returns the P2 prediction contract (or ``None``).

        ``result["confidence"]`` is a *magnitude* (``sigmoid(|P50|/IQR)``), not a
        probability, so the directional sign is carried separately: ``p_up`` is
        the neutral 0.5 when the model is directionless (a real neutral vote),
        and ``score`` is the direction × magnitude signed score.
        """
        if len(df) < 100:
            return None
        try:
            from core.strategy.indicators import compute_all
            from core.ml.features import compute_features as _cf

            df_ind = compute_all(df.copy(), _ml_feature_indicators(indicators or {}))
            X = _cf(df_ind, list(FEATURE_NAMES))
            result = tft_trainer.predict(model, X)
            if result is None:
                return None

            tft_conf = float(result["confidence"])
            tft_dir = int(result["direction"])
            # None ⇒ the caller's base rate (or the kernel default) centres it.
            p_up = (tft_conf if tft_dir > 0
                    else (1.0 - tft_conf if tft_dir < 0 else 0.5))
            return {"p_up": p_up, "base_rate": None,
                    "score": float(result.get("score", tft_conf)) * (1 if tft_dir > 0 else -1 if tft_dir < 0 else 0),
                    "abstained": tft_dir == 0}
        except Exception as e:
            logger.debug(f"TFT predict failed: {e}")
            return None

    # ── PatchTST helpers ─────────────────────────────────────────────

    def _train_patchtst_model(self, trainer, df: pd.DataFrame, symbol: str,
                              interval: str, max_train_rows: int = 5000,
                              feature_list: list[str] | None = None,
                              indicators: dict | None = None):
        """Train a PatchTST model with triple-barrier labels."""
        try:
            from core.strategy.indicators import compute_all
            from core.ml.features import (compute_features as _cf,
                  create_triple_barrier_label)

            if len(df) > max_train_rows:
                df = df.iloc[-max_train_rows:]

            df_ind = compute_all(df.copy(), _ml_feature_indicators(indicators or {}))
            # None/[] → the full contract; non-empty → the strategy's subset
            fl = list(feature_list) if feature_list else list(FEATURE_NAMES)
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

    def _predict_patchtst(self, trainer, model, df: pd.DataFrame,
                          indicators: dict | None = None) -> dict | None:
        """Predict with PatchTST — returns the P2 prediction contract (or ``None``).

        The trainer publishes true class probabilities (``p_up`` /
        ``p_up_conditional``); ``p_up_conditional`` (P(up) normalised over the
        two directional classes) is the probability the fusion kernel expects,
        with 0.5 for a timeout-dominated (directionless) bar.
        """
        if len(df) < 100:
            return None
        try:
            from core.strategy.indicators import compute_all
            from core.ml.features import compute_features as _cf

            df_ind = compute_all(df.copy(), _ml_feature_indicators(indicators or {}))
            X = _cf(df_ind, list(FEATURE_NAMES))
            result = trainer.predict(model, X)
            if result is None:
                return None

            direction = int(result["direction"])
            confidence = float(result["confidence"])
            p_up = result.get("p_up_conditional", result.get("p_up"))
            if p_up is None:
                p_up = (confidence if direction > 0
                        else (1.0 - confidence if direction < 0 else 0.5))
            score = float(result.get("score", confidence))
            return {"p_up": float(p_up), "base_rate": None,
                    "score": score * (1 if direction > 0 else -1 if direction < 0 else 0),
                    "abstained": direction == 0}
        except Exception as e:
            logger.debug(f"PatchTST predict failed: {e}")
            return None
