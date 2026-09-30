"""ML predictor — LightGBM + TFT with auto-selection.

LightGBM (tree-based): fast, stable baseline for single-step prediction.
TFT (transformer): sequence-aware, outputs direction + uncertainty.
Set model_type='tft' in config or per-strategy ml_config to use TFT.

Phase P2 notes
--------------
* Only models that pass the credibility gate
  (:func:`core.ml.credibility.credibility_gate`: OOS AUC > 0.55 **and** net
  expectancy > 0 after costs) may be loaded and used.  A model without a
  passing ``*_meta.json`` is refused — the measured live model was a negative
  contribution (accuracy 0.41–0.47 vs a 0.54–0.67 majority baseline), so
  ``ml.enabled`` ships **false** until the gate passes.
* ``confidence`` in the published event is ``P(up)`` from a **calibrated**
  model, and the event also carries ``ml_base_rate`` so the fusion kernel can
  centre the signed score correctly (item 4).
* The feature contract (:data:`core.ml.features.FEATURE_NAMES`) is asserted
  against the model metadata at load time (item 5).
"""

import asyncio
import numpy as np
import pandas as pd
from loguru import logger

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.provider import (
    DEFAULT_ML_INTERVAL, MarketDataProvider, ml_intervals)
from core.ml.trainer import MLTrainer
from core.ml.calibration import ProbabilityCalibrator
from core.ml.features import (
    FEATURE_NAMES, MIN_FEATURE_ROWS, REQUIRED_INDICATORS,
    compute_features, create_volatility_label,
    feature_schema_hash, feature_schema_mismatch_reason, FeatureContractError,
)


def _log_gate_refusal(symbol: str, strategy_name: str, meta: dict | None,
                      reason: str) -> None:
    """The gate refusal is a first-class log line, not a silent skip."""
    gate = (meta or {}).get("gate") or {}
    logger.warning(
        f"ML gate REFUSED {symbol}/{strategy_name}: {reason} | "
        f"auc={gate.get('auc')} net_expectancy={gate.get('net_expectancy')} "
        f"n_oos={gate.get('n_oos')}")


def _refusal_status(reason: str, gate: dict | None = None) -> dict:
    """Status dict for a refused model — the UI/logs show this verbatim."""
    gate = gate or {}
    return {
        "allowed": False, "enabled": False, "reason": reason,
        "auc": gate.get("auc"), "net_expectancy": gate.get("net_expectancy"),
        "n_oos": gate.get("n_oos"),
    }


class MLPredictor:
    """Real-time ML predictor supporting LightGBM and TFT models.

    Subscribes to MARKET_KLINE events on 1h/4h timeframes and publishes
    ML_PREDICTION events with directional probabilities.
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
        self._feature_list: list[str] = self._resolve_feature_list(config)
        self._model_type: str = getattr(config, 'ml_model_type', 'lightgbm')
        #: symbol → {"calibrator": ProbabilityCalibrator|None, "base_rate": float,
        #:            "threshold": float|None, "thresholds": dict, "schema_hash": str}
        self._meta: dict[str, dict] = {}
        #: (symbol, interval) → (last_bar_label, feature_matrix).  The feature
        #: pipeline is recomputed only when a new bar closes (audit P2 #7).
        self._feature_cache: dict[tuple[str, str], tuple[object, pd.DataFrame]] = {}
        self._status: dict = {"allowed": False, "reason": "not evaluated"}

    # ── configuration ────────────────────────────────────────────────

    @staticmethod
    def _resolve_feature_list(config: Config) -> list[str]:
        """The feature contract, optionally pinned by ``ml.feature_list``.

        An explicit ``ml.feature_list`` **replaces** the contract (some strategy
        families train on a small subset); it never silently drops columns from
        :data:`FEATURE_NAMES`, which is what made live (40) and backtest (47)
        disagree.
        """
        configured = getattr(config, "ml_feature_list", None)
        if configured:
            unknown = [c for c in configured if c not in FEATURE_NAMES]
            if unknown:
                raise FeatureContractError(
                    f"ml.feature_list contains non-contract column(s): {unknown}")
            return list(configured)
        return list(FEATURE_NAMES)

    @property
    def feature_contract(self) -> list[str]:
        return list(self._feature_list)

    @property
    def gate_status(self) -> dict:
        """Last gate verdict — surfaced in logs/status endpoints."""
        return dict(self._status)

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self):
        if not bool(getattr(self.config, "ml_enabled", False)):
            # Safe default (plan §二 P2.7): until a model passes the gate, ML is
            # off and the predictor does not subscribe at all.
            self._status = {"allowed": False, "enabled": False,
                            "reason": "ml.enabled is false in config"}
            logger.warning("ML predictor not started: ml.enabled=false "
                           "(gate requires OOS AUC>0.55 and net expectancy>0)")
            return
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
        # `ml.retrain_interval_hours` is the documented cadence (audit P2 #8): it
        # used to be loaded and ignored in favour of a hard-coded 86400 s.
        interval_hours = float(getattr(self.config, "ml_retrain_interval_hours", 24.0) or 24.0)
        interval_hours = max(interval_hours, 0.25)
        logger.info(f"ML retrain loop: every {interval_hours:.2f} h")
        await asyncio.sleep(300)
        while self._running:
            for symbol in self.market_data.watched_symbols:
                try:
                    # Direction model (existing)
                    if self._model_type == 'tft':
                        await self.train_tft_model(symbol, "periodic", DEFAULT_ML_INTERVAL)
                    else:
                        await self.train_model(symbol, "periodic", DEFAULT_ML_INTERVAL)
                    # Volatility model (NEW)
                    await self.train_volatility_model(symbol, "volatility", DEFAULT_ML_INTERVAL)
                except Exception as e:
                    logger.warning(f"ML retrain failed for {symbol}: {e}")
            await asyncio.sleep(interval_hours * 3600.0)

    # ── Event handlers ───────────────────────────────────────────────

    async def _on_kline(self, event: Event):
        if not self._running:
            return
        symbol = event.data["symbol"]
        interval = event.data["interval"]

        # The ML-enabled intervals are declared once in INTERVAL_SPEC
        # (core/market_data/provider.py), so a new interval needs no edit here.
        if interval not in ml_intervals():
            return

        df = await self.market_data.get_historical(symbol, interval)
        # One data requirement everywhere (audit P2 #10): the feature code needs
        # MIN_FEATURE_ROWS, and scoring below it used to produce an all-constant
        # matrix (the old guard was 100).
        if df is None or len(df) < MIN_FEATURE_ROWS:
            return

        # The whole feature pipeline runs synchronously inside an async handler
        # and the expensive part is O(n) rolling Hurst (`compute_features`), so
        # it is recomputed only when a new closed bar actually arrives.  Republish
        # the cached verdict for repeated ticks (audit P2 #7 documents the cost).
        cached = self._feature_cache.get((symbol, interval))
        last_bar = df.index[-1]
        if cached is not None and cached[0] == last_bar:
            X = cached[1]
        else:
            from core.strategy.indicators import compute_all
            ind = compute_all(df, REQUIRED_INDICATORS)
            X = compute_features(ind, self._feature_list)
            self._feature_cache[(symbol, interval)] = (last_bar, X)

        pred = await self._predict(symbol, X)

        # ── Volatility prediction (NEW — independent model) ──
        vol_expanding = await self._predict_volatility(symbol, X)

        # Gated abstention (see `_predict_lgb`): a call the gate never validated
        # is published as a neutral P(up) = base rate, so a consumer that only
        # reads `confidence` (core.strategy.engine) fuses a zero ML vote instead
        # of an unvalidated one.  `ml_score` is still published for consumers
        # that honour the signed score directly.
        abstained = bool(pred.get("abstained"))
        base_rate = float(pred["base_rate"])
        await self.event_bus.publish(Event(EventType.ML_PREDICTION, {
            "symbol": symbol,
            "interval": interval,
            # P(up) from the calibrated model — the fusion kernel centers it on
            # `ml_base_rate` (core.ml.calibration.signed_score).
            "confidence": round(base_rate if abstained else float(pred["p_up"]), 4),
            "ml_base_rate": base_rate,
            "ml_score": round(float(pred["score"]), 4),
            "ml_abstained": abstained,
            "ml_model_kind": pred["kind"],
            "volatility_expanding": vol_expanding,
        }))

    # ── prediction dispatch ──────────────────────────────────────────

    async def _predict(self, symbol: str, X: pd.DataFrame) -> dict:
        """Returns ``{p_up, base_rate, score, kind}`` (abstention-aware)."""
        if self._model_type == 'tft':
            base = await self._predict_tft(symbol, X)
        else:
            base = await self._predict_lgb(symbol, X)
        meta = self._meta.get(f"{symbol}_binary", {})
        base_rate = float(meta.get("base_rate", 0.5) or 0.5)
        return {
            "p_up": float(base["p_up"]),
            "base_rate": base_rate,
            "score": float(base.get("score", 0.0)),
            "kind": base.get("kind", self._model_type),
        }

    # ── LightGBM prediction ──────────────────────────────────────────

    async def _predict_lgb(self, symbol: str, X: pd.DataFrame) -> dict:
        """Returns ``{p_up, base_rate, score, kind, abstained}``.

        ``p_up`` is a calibrated ``P(up)``.  ``base_rate`` is the model's own
        training base rate — without it the fusion kernel cannot centre the
        signed score (audit P2 #5).
        """
        model_key = f"{symbol}_binary"
        if model_key not in self._models:
            return {"p_up": 0.5, "base_rate": 0.5, "score": 0.0,
                    "kind": "lightgbm", "abstained": True}
        try:
            latest = X.iloc[-1:].fillna(0)
            proba = self._models[model_key].predict_proba(latest)[0]
            p_up = float(proba[1]) if len(proba) > 1 else 0.5
            meta = self._meta.get(model_key, {})
            cal = meta.get("calibrator")
            if isinstance(cal, ProbabilityCalibrator) and cal.fitted:
                p_up = float(cal.transform([p_up])[0])
            base_rate = float(meta.get("base_rate", 0.5) or 0.5)
            if not (0.0 < base_rate < 1.0):
                base_rate = 0.5
            from core.ml.calibration import signed_score
            score = float(signed_score(p_up, base_rate))

            # ── Gated decision (Phase P2 items 2/3/6, audit P2 #4) ──
            # Both side thresholds and the side they belong to are persisted by
            # the trainer.  The old code kept only `threshold_up` and then
            # rebuilt the band as `|threshold − base_rate|` with a `band > 0`
            # guard: with base_rate 0.4 / threshold 0.4 the band was exactly 0
            # and the model **never abstained** — every bar was published as a
            # directional call.  A zero/negative band is now an abstention.
            thresholds = meta.get("thresholds") or {}
            side = thresholds.get("side") or meta.get("threshold_side")
            if side == "long":
                threshold = thresholds.get("threshold_up")
            elif side == "short":
                threshold = thresholds.get("threshold_down")
            else:
                threshold = meta.get("threshold")
            if threshold is None:
                threshold = thresholds.get("threshold_up")
            if threshold is None:
                threshold = getattr(self.config, "ml_confidence_threshold", None)
            if threshold is None:
                threshold = getattr(self.config, "ml_default_confidence_threshold", None)
            threshold = None if threshold is None else float(threshold)
            if threshold is None:
                return {"p_up": p_up, "base_rate": base_rate, "score": score,
                        "kind": "lightgbm", "abstained": False,
                        "threshold": None}
            # The persisted side decides WHICH decision it is; the threshold is
            # compared to that side's own base rate, never to `1 − base_rate`.
            # A zero-width band (the audit's `base_rate 0.4 / threshold_up 0.4`)
            # means the stored threshold cannot distinguish a call from the base
            # rate at all, so the model must abstain — the pre-fix code guarded
            # `band > 0` on `|threshold_up − base_rate|`, fell through and traded
            # **every** bar (audit P2 #4).
            if side == "short":
                degenerate = threshold >= base_rate
            elif side == "long":
                degenerate = threshold <= base_rate
            else:
                degenerate = not np.isfinite(threshold) or abs(threshold - base_rate) <= 0.0
            if degenerate:
                logger.warning(
                    f"ML {symbol}: decision threshold {threshold} does not "
                    f"separate the {side or 'two-sided'} band from the base rate "
                    f"{base_rate:.4f} — abstaining (zero-width band)")
                return {"p_up": p_up, "base_rate": base_rate, "score": 0.0,
                        "kind": "lightgbm", "abstained": True,
                        "threshold": threshold, "side": side}
            if side == "long":
                outside = p_up < threshold
            elif side == "short":
                outside = p_up > threshold
            else:
                # No side recorded: symmetric two-sided band.
                outside = abs(p_up - base_rate) < abs(threshold - base_rate)
            if outside:
                return {"p_up": p_up, "base_rate": base_rate,
                        "score": 0.0, "kind": "lightgbm",
                        "abstained": True, "threshold": threshold, "side": side}
            return {"p_up": p_up, "base_rate": base_rate,
                    "score": score, "kind": "lightgbm",
                    "abstained": False, "threshold": threshold, "side": side}
        except Exception as e:
            logger.debug(f"LGB predict failed for {symbol}: {e}")
            return {"p_up": 0.5, "base_rate": 0.5, "score": 0.0,
                    "kind": "lightgbm", "abstained": True}

    # ── TFT prediction ───────────────────────────────────────────────

    async def _predict_tft(self, symbol: str, X: pd.DataFrame) -> dict:
        """TFT prediction: returns a {p_up, score} dict.

        ``result["confidence"]`` is **not** a probability (it is
        ``sigmoid(|P50|/IQR)``), so it is published as ``score`` only.  ``p_up``
        is ``0.5`` for TFT unless a calibrated binary model supplies one — the
        fusion kernel treats 0.5-at-base-rate as a neutral vote rather than
        inventing a direction.
        """
        model = self._tft_models.get(symbol)
        if model is None:
            return {"p_up": 0.5, "score": 0.0, "kind": "tft"}

        tft = self._get_tft_trainer()
        result = tft.predict(model, X, feature_cols=self._feature_list)
        if result is None:
            return {"p_up": 0.5, "score": 0.0, "kind": "tft"}

        tft_dir = int(result["direction"])
        tft_score = float(result.get("score", result["confidence"]))
        # Direction × score → signed score in [-1, +1] (renamed from
        # "confidence": it never was a probability).
        signed = tft_score if tft_dir > 0 else (-tft_score if tft_dir < 0 else 0.0)
        meta = self._meta.get(f"{symbol}_binary", {})
        base_rate = float(meta.get("base_rate", 0.5) or 0.5)
        p_up = 0.5 + 0.5 * signed  # score → probability-shaped value
        return {"p_up": p_up, "base_rate": base_rate,
                "score": signed, "kind": "tft"}

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
        if df is None or len(df) < MIN_FEATURE_ROWS:
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
        if df is None or len(df) < MIN_FEATURE_ROWS:
            return {"error": f"Insufficient data: {len(df) if df is not None else 0} rows"}

        from core.strategy.indicators import compute_all
        from core.ml.labels import (
            create_three_class_label, CLASS_UP, CLASS_DOWN, CLASS_FLAT)
        from core.ml.credibility import cost_pct_for, evaluate_model_oos, gate_from_evaluation

        df = compute_all(df, REQUIRED_INDICATORS)
        X = compute_features(df, self._feature_list)

        forward, threshold = self._label_params(interval)
        cost_multiple = float(getattr(self.config, "ml_label_cost_multiple", 0.0) or 0.0)
        # `flat` is kept as a class (item 1a) — the binary target is built on the
        # labelled subset and the *decision* abstains on the flat regime; the
        # coverage of every decision is therefore reportable.
        three = create_three_class_label(
            df, forward_periods=forward, threshold=threshold,
            cost_pct=cost_pct_for(self.config, symbol=symbol),
            cost_multiple=cost_multiple)
        # The volatility-scaled triple-barrier label is built from the configured
        # `ml.barrier_*` parameters and its class distribution is persisted
        # (audit P2 #8: those four keys and `max_hold_bars` were loaded and never
        # read).  Its time barrier is the strategy's real holding period.
        # `_barrier_labels` is the ONE place the persisted distribution is
        # computed, so the sidecar can never disagree with the labels
        # (audit D-19: it used to read `timeout_share = 0.0` because the
        # vol-scaled path never filled the timeout class).
        barrier, barrier_dist, barrier_kwargs = self._barrier_labels(df)
        fwd = (df["close"].shift(-forward) - df["close"]) / df["close"]
        binary = pd.Series(pd.NA, index=df.index, dtype="Float64")
        binary[three == CLASS_UP] = 1.0
        binary[three == CLASS_DOWN] = 0.0

        common_idx = X.index.intersection(binary.dropna().index)
        if len(common_idx) < 100:
            return {"error": f"Insufficient labelled data: {len(common_idx)} rows"}
        X = X.loc[common_idx]
        y = binary.loc[common_idx].astype(float)
        fwd = fwd.loc[common_idx]

        # ── Out-of-sample gate (purged K-fold, embargo = label horizon) ──
        cost_pct = cost_pct_for(self.config, symbol=symbol)
        gate_cfg = self._gate_config()
        calibration = str(getattr(self.config, "ml_calibration", "isotonic") or "isotonic")
        evaluation = evaluate_model_oos(
            X, y, fwd, n_splits=5, label_span=forward,
            # `ml.embargo_bars` (default = label horizon) feeds the splitter.
            embargo=int(getattr(self.config, "ml_embargo_bars", forward) or forward),
            cost_pct=cost_pct,
            calibrate=calibration,
            min_trades=int(gate_cfg["min_trades"]))
        gate = gate_from_evaluation(evaluation, **gate_cfg)
        self._status = gate
        if not gate["allowed"]:
            _log_gate_refusal(symbol, strategy_name, {"gate": gate}, gate["reason"])

        # Both side thresholds + the selected side are persisted (audit P2 #4):
        # persisting only `threshold_up` while the short side won left the live
        # band computed from the wrong threshold.
        oos_thresholds = evaluation.get("thresholds_oos") or evaluation.get("thresholds") or {}
        result = self.trainer.train_binary(
            symbol, strategy_name, X, y, engine="lightgbm",
            base_rate=float(y.mean()),
            threshold=oos_thresholds.get("threshold_up"
                                         if oos_thresholds.get("side") != "short"
                                         else "threshold_down"),
            threshold_up=oos_thresholds.get("threshold_up"),
            threshold_down=oos_thresholds.get("threshold_down"),
            threshold_side=oos_thresholds.get("side"),
            gate=gate,
            extra_meta={
                "feature_schema_hash": feature_schema_hash(self._feature_list),
                "label_kind": "three_class_up_down_subset",
                "label_forward_periods": int(forward),
                "label_threshold": float(threshold),
                "label_cost_multiple": float(cost_multiple),
                "cost_pct": float(cost_pct),
                "flat_share": float((three == CLASS_FLAT).sum()) / max(len(three.dropna()), 1),
                "barrier": {
                    "forward_periods": int(getattr(self.config, "ml_max_hold_bars", 24) or 24),
                    **barrier_kwargs,
                    "distribution": barrier_dist,
                },
                "oos_evaluation": {k: v for k, v in evaluation.items()
                                   if k not in ("p_oos", "p_oos_calibrated",
                                                "p_raw_oos", "y_oos", "fwd_oos",
                                                "index_oos", "thresholds",
                                                "take_oos", "decisions_oos",
                                                "side_oos")},
        })

        if not gate["allowed"]:
            # Refuse to install the model: `enabled` may only be true past the gate.
            result["enabled"] = False
            result["gate"] = gate
            return result

        if "model_path" in result:
            self.load_model(symbol, result["model_path"])
        return result

    def _label_params(self, interval: str) -> tuple[int, float]:
        """Horizon/threshold for an interval, from the ``ml:`` config block.

        Precedence: ``ml.label_params.<interval>`` (explicit per-interval table) →
        ``ml.forward_periods`` / ``ml.label_threshold`` (documented defaults that
        a user can set once) → the built-in 4 / 0.005.
        """
        table = getattr(self.config, "ml_label_params", None) or {}
        entry = table.get(str(interval)) if isinstance(table, dict) else None
        if isinstance(entry, dict):
            return int(entry.get("forward", 4)), float(entry.get("threshold", 0.005))
        forward = getattr(self.config, "ml_forward_periods", None)
        threshold = getattr(self.config, "ml_label_threshold", None)
        return (int(forward) if forward else 4,
                float(threshold) if threshold else 0.005)

    def _gate_config(self) -> dict:
        """The ``ml.gate_*`` / ``ml.min_oos_rows`` keys, as gate kwargs.

        Previously these 12 ``ml:`` keys were loaded and never read (audit P2 #8);
        they now feed the gate, the purged splitter, the label/barrier parameters
        and the retrain cadence.
        """
        cfg = self.config
        # `min_oos_rows` is a floor, never a licence to gate on fewer rows than
        # the feature contract needs: 2 × features out-of-sample rows.
        contract_floor = 2 * len(self._feature_list or FEATURE_NAMES)
        return {
            "auc_min": float(getattr(cfg, "ml_gate_auc_min", 0.55)),
            "min_net_expectancy": float(getattr(cfg, "ml_gate_net_expectancy_min", 0.0)),
            "min_trades": int(getattr(cfg, "ml_gate_min_trades", 100)),
            "min_t_stat": float(getattr(cfg, "ml_gate_min_t_stat", 2.0)),
            # Audit F3/F5: `ml.gate_min_psr` was loaded and read nowhere (the gate
            # hard-coded GATE_MIN_PSR), so the key is wired through here — and the
            # `min_psr` kwarg now reaches `credibility_gate` via
            # `gate_from_evaluation` instead of being silently dropped.
            "min_psr": float(getattr(cfg, "ml_gate_min_psr", 0.95)),
            "min_oos": max(int(getattr(cfg, "ml_min_oos_rows", 100)), contract_floor),
        }

    def _barrier_params(self) -> dict:
        """``ml.barrier_*`` / ``ml.max_hold_bars`` → triple-barrier label kwargs."""
        cfg = self.config
        return {
            "atr_period": int(getattr(cfg, "ml_barrier_atr_period", 14)),
            "atr_multiple": float(getattr(cfg, "ml_barrier_atr_multiple", 1.5)),
            "min_pct": float(getattr(cfg, "ml_barrier_min_pct", 0.004)),
            "max_pct": float(getattr(cfg, "ml_barrier_max_pct", 0.06)),
        }

    def _barrier_labels(self, df) -> tuple:
        """``(labels, distribution, kwargs)`` for the persisted barrier sidecar.

        The volatility-scaled triple barrier is built here and **only** here, so
        the ``barrier.distribution`` written into the model metadata is by
        construction ``class_distribution`` of the labels that were actually
        produced (audit D-19: the two used to be able to disagree, and the
        persisted ``timeout_share`` was always ``0.0``).
        """
        from core.ml.labels import (class_distribution,
                                    create_triple_barrier_label_vol)

        kwargs = self._barrier_params()
        labels = create_triple_barrier_label_vol(
            df,
            forward_periods=int(getattr(self.config, "ml_max_hold_bars", 24) or 24),
            timeout_label=2.0, **kwargs)
        return labels, class_distribution(labels), kwargs

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
        """Legacy scalar API — returns calibrated ``P(up)``."""
        flist = features or self._feature_list
        X = compute_features(df, flist)
        return float((await self._predict(symbol, X))["p_up"])

    async def predict_detail(self, symbol: str, df: pd.DataFrame,
                             features: list[str] | None = None) -> dict:
        """``{p_up, base_rate, score, kind}`` — what the fusion path needs."""
        flist = features or self._feature_list
        X = compute_features(df, flist)
        return await self._predict(symbol, X)

    def load_model(self, symbol: str, model_path: str):
        """Load a model **only** if its metadata proves it passed the gate.

        The schema hash is verified against :attr:`feature_contract` so a model
        trained on a different feature set can never be scored with this one.
        """
        meta = self._meta_for_path(model_path)
        strategy_name = self._strategy_from_filename(model_path)
        if meta is None:
            reason = ("no metadata sidecar (*_meta.json) — cannot verify "
                      "the OOS gate")
            _log_gate_refusal(symbol, strategy_name, None, reason)
            self._status = _refusal_status(reason)
            return None
        gate = meta.get("gate") or {}
        if not gate:
            reason = "metadata has no gate verdict"
            _log_gate_refusal(symbol, strategy_name, meta, reason)
            self._status = _refusal_status(reason)
            return None
        if not gate.get("allowed"):
            reason = gate.get("reason", "gate failed")
            _log_gate_refusal(symbol, strategy_name, meta, reason)
            self._status = _refusal_status(reason, gate)
            return None
        names = list(meta.get("feature_names") or [])
        if names != list(self._feature_list):
            have = set(self._feature_list)
            missing = [c for c in names if c not in have]
            extra = [c for c in self._feature_list if c not in set(names)]
            raise FeatureContractError(
                f"model {model_path} expects {len(names)} features but the "
                f"predictor is configured for {len(self._feature_list)}; "
                f"missing_from_predictor={missing[:4]} extra_in_predictor={extra[:4]}")
        # The stored schema hash is verified too (audit P2 #10; tightened by
        # re-audit finding 5): a sidecar whose hash does not match its own column
        # list was written by a different contract and must not be scored
        # positionally — and a sidecar with **no** hash at all is refused the same
        # way, because an absent hash cannot be compared.  The old `if stored and
        # ...` let a hand-written or truncated sidecar through on names alone.
        #
        # P6-B: the refusal is **named by contract version** — a model whose
        # sidecar carries the v1 hash (`335e63360104`, the 39-column P2 contract)
        # is refused with "v1" in the reason instead of a bare hex pair, and the
        # wording is shared with the engine's preload gate through
        # `feature_schema_mismatch_reason` so the two cannot drift.
        stored_hash = meta.get("feature_schema_hash")
        expected_hash = feature_schema_hash(self._feature_list)
        if not stored_hash or str(stored_hash) != expected_hash:
            reason = feature_schema_mismatch_reason(stored_hash, expected_hash)
            _log_gate_refusal(symbol, strategy_name, meta, reason)
            self._status = _refusal_status(reason, gate)
            raise FeatureContractError(f"model {model_path}: {reason}")

        model = self.trainer.load_model(model_path)
        if model is None:
            return None
        self._models[f"{symbol}_binary"] = model
        cal_payload = meta.get("calibrator")
        self._meta[f"{symbol}_binary"] = {
            "calibrator": ProbabilityCalibrator.from_dict(cal_payload) if cal_payload else None,
            "base_rate": float(meta.get("train_base_rate", 0.5) or 0.5),
            "threshold": meta.get("threshold"),
            "thresholds": dict(meta.get("thresholds") or {}),
            # Verified against `feature_contract` above — kept for the monitor.
            "schema_hash": expected_hash,
            "gate": gate,
        }
        self._status = gate
        return model

    def _meta_for_path(self, model_path: str) -> dict | None:
        from pathlib import Path
        p = Path(model_path)
        meta = p.with_name(p.stem + "_meta.json")
        if not meta.exists():
            return None
        import json
        try:
            with open(meta, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _strategy_from_filename(model_path: str) -> str:
        """``BTCUSDT_default_binary.pkl`` → ``default`` (not the whole path)."""
        from pathlib import Path
        stem = Path(model_path).stem
        parts = stem.split("_")
        # <SYMBOL>_<strategy...>_<model_type>
        if len(parts) >= 3:
            return "_".join(parts[1:-1])
        return stem

    @property
    def feature_count(self) -> int:
        return len(self._feature_list)

    @property
    def model_type(self) -> str:
        return self._model_type
