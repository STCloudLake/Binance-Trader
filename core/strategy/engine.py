import asyncio
import pandas as pd

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.provider import MarketDataProvider, interval_minutes
from core.strategy.loader import StrategyLoader, StrategyConfig
from core.strategy.indicators import compute_all, evaluate_condition
from core.strategy.evaluation_kernel import (
    evaluate_exit_conditions as eval_exit_conds,
    fuse_signals,
    check_higher_tf_trend,
)

# ── P4 integration seams — all OFF (no live behaviour change by default) ──
#
# Phase P4 adds four new capabilities (pairs/cointegration, meta-labelling,
# microstructure features, regime gating).  Every one of them is a *new*
# component, so the live path must be byte-for-byte the pre-P4 path until a
# caller flips a flag **and** registers the component.  These three module
# constants are the whole story:
#
#: Report the detected regime in the signal cache (no gating, no signal change).
P4_REGIME_DIAGNOSTICS_ENABLED = False
#: Let a registered ``MetaLabeler`` filter/size live entries (never a direction).
P4_META_FILTER_ENABLED = False
#: Let a registered pairs provider override a strategy's indicator signal.
P4_PAIRS_SIGNALS_ENABLED = False


class StrategyEngine:
    def __init__(self, config: Config, event_bus: EventBus, market_data: MarketDataProvider):
        self.config = config
        self.event_bus = event_bus
        self.market_data = market_data
        self.loader = StrategyLoader(config.strategies_dir)
        self._executor = None
        self._running = False
        self._strategies: dict[str, StrategyConfig] = {}
        self._signal_cache: dict[str, dict] = {}
        self._ml_confidence: dict[str, float] = {}
        #: symbol → the ML predictor's full verdict
        #: ``{"confidence", "base_rate", "score", "abstained"}``.  `engine.py`
        #: used to keep only `confidence` and call `fuse_signals` without
        #: `ml_base_rate`/`ml_score`, so an abstention published as
        #: ``confidence = base_rate`` scored as a small **bearish** vote
        #: (audit P2 #5).
        self._ml_prediction: dict[str, dict] = {}
        self._news_sentiment: dict[str, float] = {}
        #: P4 seam: ``(symbol, interval) -> PairsSignal | None`` (see
        #: :mod:`core.strategy.pairs`).  ``None`` = the seam is inert.
        self._p4_pairs_provider = None
        #: P4 seam: ``(MetaLabeler, probability_fn)`` where
        #: ``probability_fn(df, symbol) -> float`` is the secondary model's
        #: ``P(primary trade hits its profit barrier)``.
        self._p4_meta_labeler = None
        self._p4_meta_probability = None

    # ── P4 wiring (explicit, opt-in, no effect until a flag is enabled) ──

    def wire_pairs_provider(self, provider) -> None:
        """Register ``provider(symbol, interval) -> PairsSignal | None``.

        The signal is used only when :data:`P4_PAIRS_SIGNALS_ENABLED` is true,
        and only as the ``indicator_signal`` input of the **shared** fusion
        kernel — so a pairs entry travels the identical risk/execution pipeline
        as any other strategy signal.
        """
        self._p4_pairs_provider = provider

    def wire_meta_filter(self, labeler, probability_fn) -> None:
        """Register the meta-labelling filter (see :mod:`core.ml.meta`).

        ``probability_fn(df, symbol)`` must return the secondary model's
        probability for the *current* bar; it is the caller's job to supply a
        causal feature matrix.  The seam multiplies the fused score by the
        meta size multiplier and suppresses the entry when the filter says no —
        it can never change the side.
        """
        self._p4_meta_labeler = labeler
        self._p4_meta_probability = probability_fn

    def _p4_pairs_indicator(self, symbol: str, interval: str) -> float | None:
        """The pairs provider's ``indicator_signal``, or ``None`` when inert."""
        if not P4_PAIRS_SIGNALS_ENABLED or self._p4_pairs_provider is None:
            return None
        try:
            signal = self._p4_pairs_provider(symbol, interval)
        except Exception:
            from loguru import logger
            logger.exception(f"P4 pairs provider failed for {symbol} {interval}")
            return None
        if signal is None or not getattr(signal, "allowed", False):
            return None
        return float(getattr(signal, "indicator_signal", 0.0))

    def _p4_regime(self, df) -> dict | None:
        """Regime label for diagnostics, or ``None`` when the seam is off."""
        if not (P4_REGIME_DIAGNOSTICS_ENABLED or P4_META_FILTER_ENABLED):
            return None
        try:
            from core.strategy.regime import classify_last
            return classify_last(df)
        except Exception:
            from loguru import logger
            logger.exception("P4 regime classification failed")
            return None

    def _p4_meta(self, symbol: str, df, side: str, score: float) -> tuple[float, dict | None]:
        """Apply the meta filter/size to ``score`` (no-op unless enabled)."""
        if not P4_META_FILTER_ENABLED or self._p4_meta_labeler is None \
                or self._p4_meta_probability is None:
            return float(score), None
        try:
            p = float(self._p4_meta_probability(df, symbol))
            decision = self._p4_meta_labeler.decide(p, 1.0 if side == "long" else -1.0)
        except Exception:
            from loguru import logger
            logger.exception(f"P4 meta filter failed for {symbol} — entry kept unfiltered")
            return float(score), None
        info = {"probability": p, "threshold": decision.threshold,
                "take": bool(decision.take), "size_multiplier": decision.size_multiplier,
                "reason": decision.reason}
        if not decision.take:
            return 0.0, info
        return float(score) * float(decision.size_multiplier), info

    def wire_executor(self, executor):
        self._executor = executor

    def _purge_stale_cache(self):
        """Remove signal cache entries for strategies/symbols that no longer apply."""
        stale_keys = []
        for key in self._signal_cache:
            # key format: "strategy_name|symbol"
            if "|" not in key:
                continue
            s_name, sym = key.split("|", 1)
            strategy = self._strategies.get(s_name)
            if not strategy:
                stale_keys.append(key)
                continue
            if strategy.symbols and sym not in strategy.symbols:
                stale_keys.append(key)
        for k in stale_keys:
            self._signal_cache.pop(k, None)
        if stale_keys:
            from loguru import logger
            logger.info(f"Purged {len(stale_keys)} stale signal cache entries")

    async def start(self):
        from loguru import logger
        self._running = True
        self.event_bus.subscribe(EventType.MARKET_KLINE, self._on_kline)
        self.event_bus.subscribe(EventType.ML_PREDICTION, self._on_ml_prediction)
        self.event_bus.subscribe(EventType.NEWS_UPDATE, self._on_news_update)
        all_strategies = self.loader.load_all()
        for s in all_strategies:
            if not s.timeframes:
                logger.warning(f"Strategy '{s.name}' has no timeframes configured — will never evaluate!")
            if s.symbols:
                logger.info(f"Strategy '{s.name}' restricted to symbols: {s.symbols}")
        self._strategies = {s.name: s for s in all_strategies}
        self._purge_stale_cache()

    async def _on_kline(self, event: Event):
        if not self._running:
            return
        symbol = event.data["symbol"]
        interval = event.data["interval"]

        # Diagnostic: track evaluation counts
        self._eval_count = getattr(self, '_eval_count', {})
        ek = f"{symbol}_{interval}"
        self._eval_count[ek] = self._eval_count.get(ek, 0) + 1
        if self._eval_count[ek] in (1, 10, 50, 100):
            from loguru import logger
            logger.info(f"Strategy eval #{self._eval_count[ek]}: {symbol} {interval} ({len(self._strategies)} strategies)")

        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue
            if interval not in strategy.timeframes:
                continue
            # Respect strategy→symbol mapping
            if strategy.symbols and symbol not in strategy.symbols:
                continue
            await self._evaluate(symbol, interval, strategy)

    async def _on_ml_prediction(self, event: Event):
        symbol = event.data.get("symbol", "")
        self._ml_confidence[symbol] = event.data.get("confidence", 0.5)
        payload = {"confidence": event.data.get("confidence", 0.5)}
        if "ml_base_rate" in event.data or "base_rate" in event.data:
            payload["base_rate"] = event.data.get("ml_base_rate",
                                                  event.data.get("base_rate"))
        if "ml_score" in event.data or "score" in event.data:
            payload["score"] = event.data.get("ml_score", event.data.get("score"))
        if "ml_abstained" in event.data or "abstained" in event.data:
            payload["abstained"] = event.data.get("ml_abstained",
                                                  event.data.get("abstained"))
        self._ml_prediction[symbol] = payload

    def _ml_fusion_inputs(self, symbol: str) -> dict:
        """``fuse_signals`` kwargs for *symbol*'s latest ML prediction.

        Audit P2 #5: the live path kept only ``confidence``.  With no base rate
        the kernel defaults to 0.5, so an abstention published as
        ``confidence = base_rate`` (e.g. 0.40) became a **bearish** vote of
        ``(0.40 − 0.5) × 2 = −0.20``.  The verdict now carries the model's own
        base rate and its signed score; an abstention contributes exactly 0.

        Callers that only have a plain confidence (older tests, an external
        event producer) keep the historical behaviour: no ``base_rate``/``score``
        keys → the kernel's 0.5 default reproduces ``(conf − 0.5) × 2``.
        """
        stored = self._ml_prediction.get(symbol)
        if stored is None:
            return {"ml_confidence": float(self._ml_confidence.get(symbol, 0.5))}
        kwargs: dict = {"ml_confidence": float(stored.get("confidence", 0.5))}
        score = stored.get("score")
        base_rate = stored.get("base_rate")
        if base_rate is not None:
            kwargs["ml_base_rate"] = float(base_rate)
        if score is not None or base_rate is not None:
            kwargs["ml_score"] = 0.0 if stored.get("abstained") else score
        return kwargs

    async def _on_news_update(self, event: Event):
        symbol = event.data.get("symbol", "")
        self._news_sentiment[symbol] = event.data.get("sentiment", 0.0)

    async def _evaluate(self, symbol: str, interval: str, strategy: StrategyConfig, publish: bool = True):
        df = await self.market_data.get_historical(symbol, interval)
        if df is None or len(df) < 50:
            return

        df = compute_all(df, strategy.indicators)

        # ── Shared Kernel: Entry condition evaluation ──
        # ``StrategyConfig.entry_sides`` is the ONE entry-structure evaluator:
        # it honours the persisted ``condition_logic`` gene ("or" → the shared
        # OR kernel below, "and" → every condition must hold) exactly as the
        # GA/backtest entry path does, so post-load trading cannot diverge from
        # the structure the genome was scored under.
        long_active, short_active = strategy.entry_sides(df)

        # Build per-condition diagnostic results (kernel only returns active flags)
        entry_results = {"long": [], "short": []}
        for side in ("long", "short"):
            for cond in strategy.entry_conditions.get(side, []):
                mask = evaluate_condition(df, cond)
                met = bool(hasattr(mask, 'iloc') and mask.iloc[-1])
                entry_results[side].append({"condition": cond, "met": met})

        # If both sides are active, the signal is ambiguous — don't trade.
        if long_active and short_active:
            from loguru import logger
            logger.warning(f"Strategy '{strategy.name}' {symbol}: long AND short conditions both met — "
                          f"signal set to 0 (ambiguous). Check for conflicting entry conditions.")
            indicator_signal = 0.0
        elif long_active:
            indicator_signal = 1.0
        elif short_active:
            indicator_signal = -1.0
        else:
            indicator_signal = 0.0

        # ── P4 seam: a registered pairs provider may supply the indicator ──
        # signal (see `core.strategy.pairs`).  Default-off: with
        # `P4_PAIRS_SIGNALS_ENABLED = False` this is a single `None` check and
        # the rest of the evaluation is exactly the pre-P4 path.
        pairs_indicator = self._p4_pairs_indicator(symbol, interval)
        if pairs_indicator is not None:
            indicator_signal = pairs_indicator
            entry_results["long"].append({"condition": "pairs_signal", "met": pairs_indicator > 0})
            entry_results["short"].append({"condition": "pairs_signal", "met": pairs_indicator < 0})

        # ── Shared Kernel: Exit condition evaluation ──
        exit_signal_long = eval_exit_conds(df, strategy.exit_conditions, "long")
        exit_signal_short = eval_exit_conds(df, strategy.exit_conditions, "short")

        # Build per-condition diagnostic results
        exit_results = {"long": [], "short": []}
        for side in ("long", "short"):
            for cond in strategy.exit_conditions.get(side, []):
                mask = evaluate_condition(df, cond)
                met = bool(hasattr(mask, 'iloc') and mask.iloc[-1])
                exit_results[side].append({"condition": cond, "met": met})

        ml_conf = self._ml_confidence.get(symbol, 0.5)  # 0.5 = neutral (no prediction)
        news_sent = self._news_sentiment.get(symbol, 0.0)

        # ── Shared Kernel: Signal fusion ──
        w = self.config.signal_weights
        ml_enabled = bool(strategy.ml_config and strategy.ml_config.enabled)
        strategy_ml_weight = strategy.ml_config.weight if ml_enabled else None

        # Full ML verdict (base rate + signed score + abstention), not just the
        # scalar confidence — see `_ml_fusion_inputs` (audit P2 #5).  The helper
        # returns `ml_confidence` itself, so it is not also passed explicitly.
        ml_inputs = self._ml_fusion_inputs(symbol)
        final_score = fuse_signals(
            indicator_signal=indicator_signal,
            news_sentiment=news_sent,  # live mode — news is available
            w_indicator=w.indicator,
            w_ml=w.ml,
            w_news=w.news,
            ml_enabled=ml_enabled,
            strategy_ml_weight=strategy_ml_weight,
            **ml_inputs,
        )

        # Determine entry side from the signal
        entry_side = "long" if final_score > 0 else "short"

        # ── P4 seams (regime diagnostics / meta filter+size; all default-off) ──
        # The meta filter can only scale the fused score towards zero or
        # suppress the entry — it can never flip the side (Prado's
        # meta-labelling contract, enforced by `MetaDecision.apply`).
        p4_regime = self._p4_regime(df)
        final_score, p4_meta = self._p4_meta(symbol, df, entry_side, final_score)

        # ── Shared Kernel: Higher-timeframe trend alignment ──
        if indicator_signal != 0.0 and len(strategy.timeframes) > 1:
            # Bar lengths come from the interval registry (INTERVAL_SPEC) — no
            # local copy to drift when an interval is added there.
            current_min = interval_minutes(interval)
            higher_tfs = [tf for tf in strategy.timeframes
                          if interval_minutes(tf) > current_min]
            tf_multiplier = 1.0
            for htf in higher_tfs:
                try:
                    df_htf = await self.market_data.get_historical(symbol, htf)
                    if df_htf is not None and len(df_htf) >= 50:
                        mult = check_higher_tf_trend(df_htf, entry_side)
                        tf_multiplier = min(tf_multiplier, mult)
                except Exception:
                    pass
            final_score *= tf_multiplier

        # An exit signal only blocks entry on the SAME side, and only when a position exists
        # for that side (or would be opened). No position open → exit signals are advisory only.
        has_position = self._executor and symbol in self._executor.get_open_positions()
        if has_position:
            pos = self._executor.get_open_positions().get(symbol, {})
            pos_side = pos.get("side", "")
        else:
            pos_side = ""

        # Exit blocks entry for same side when position exists OR would contradict
        exit_blocks_entry = False
        if entry_side == "long" and exit_signal_long:
            exit_blocks_entry = has_position and pos_side == "long"
        elif entry_side == "short" and exit_signal_short:
            exit_blocks_entry = has_position and pos_side == "short"

        # Extract key indicator values for diagnostics
        indicator_snapshots = {}
        for col in ["rsi", "macd_histogram", "bollinger_width", "volume_ratio",
                     "ema_fast", "ema_slow", "close", "adx", "bollinger_upper",
                     "bollinger_middle", "bollinger_lower"]:
            if col in df.columns and len(df) > 0:
                val = df[col].iloc[-1]
                indicator_snapshots[col] = round(float(val), 6) if not pd.isna(val) else None

        # Aggregate exit_signal for monitor display (backwards-compat)
        exit_signal = exit_signal_long or exit_signal_short

        # Effective ML weight actually used by fuse_signals — kept in sync so the
        # dashboard reports the weight that produced this score (regression: the
        # Shared Kernel refactor renamed ml_weight → strategy_ml_weight and this
        # dict kept referencing the old name, raising NameError on every
        # evaluation and silently killing the whole live signal path).
        effective_ml_weight = (
            strategy_ml_weight
            if (ml_enabled and strategy_ml_weight is not None)
            else w.ml
        )

        key = f"{strategy.name}|{symbol}"
        self._signal_cache[key] = {
            "strategy": strategy.name,
            "symbol": symbol,
            "indicator_signal": indicator_signal,
            "ml_confidence": ml_conf,
            # The full ML verdict that produced this score (audit P2 #5) — the
            # dashboard can now show the base rate and the signed contribution
            # instead of a bare probability.
            "ml_base_rate": ml_inputs.get("ml_base_rate"),
            "ml_score": ml_inputs.get("ml_score"),
            "news_sentiment": news_sent,
            "final_score": final_score,
            "exit_signal": exit_signal,
            "exit_signal_long": exit_signal_long,
            "exit_signal_short": exit_signal_short,
            "exit_blocks_entry": exit_blocks_entry,
            "entry_results": entry_results,
            "exit_results": exit_results,
            "indicators": indicator_snapshots,
            "threshold_met": abs(final_score) >= 0.5,
            "weights": {"indicator": w.indicator, "ml": effective_ml_weight, "news": w.news},
        }
        # P4 diagnostics are attached only when a seam actually produced
        # something, so the default cache payload is unchanged.
        if p4_regime is not None or p4_meta is not None or pairs_indicator is not None:
            self._signal_cache[key]["p4"] = {
                "regime": p4_regime, "meta": p4_meta,
                "pairs_indicator": pairs_indicator,
            }

        # Signal publishing — only when driven by real-time klines
        if not publish:
            return

        # Publish entry signal — exit only blocks same-side entry when position exists
        if abs(final_score) >= 0.5 and not exit_blocks_entry:
            price = self.market_data.get_current_price(symbol)
            if not price and df is not None and len(df) > 0:
                price = float(df["close"].iloc[-1])
            if not price:
                from loguru import logger
                logger.warning(f"Signal suppressed: no price for {symbol}")
                return
            from loguru import logger
            logger.info(f"SIGNAL: {strategy.name} {entry_side.upper()} {symbol} @ {price:.4f} score={final_score:.4f}")
            await self.event_bus.publish(Event(EventType.STRATEGY_SIGNAL, {
                "symbol": symbol,
                "strategy": strategy.name,
                "side": entry_side,
                "confidence": abs(final_score),
                "timeframe": interval,
                "price": price,
                "trader": "ai",
                "strategy_name": strategy.name,
                "position_type": "satellite" if strategy.mode == "scalp" else "core",
            }))

        # Check reduce conditions FIRST — partial profit-take before full exit.
        # Only manage positions opened by THIS strategy (not other strategies).
        reduce_fired = False
        reduce_cfg = strategy.reduce_conditions
        if reduce_cfg and self._executor:
            open_pos = self._executor.get_open_positions()
            if symbol in open_pos:
                pos = open_pos[symbol]
                # Only manage positions this strategy opened
                pos_strategy = pos.get("strategy_name", "") or pos.get("strategy", "")
                if pos_strategy != strategy.name:
                    pass  # Skip — position belongs to a different strategy
                else:
                    side = pos.get("side", "long")
                    reduce_key = f"reduce_count_{strategy.name}_{symbol}_{side}"
                    reduce_count = self._signal_cache.get(reduce_key, 0)
                    if reduce_count < 4:
                        conditions = reduce_cfg.get(side, [])
                        for rc in conditions:
                            cond_str = rc.get("condition", "") if isinstance(rc, dict) else str(rc)
                            rpct = rc.get("reduce_pct", 50) if isinstance(rc, dict) else 50
                            if not cond_str:
                                continue
                            try:
                                mask = evaluate_condition(df, cond_str)
                                if hasattr(mask, 'iloc') and mask.iloc[-1]:
                                    price = self.market_data.get_current_price(symbol)
                                    if not price and df is not None and len(df) > 0:
                                        price = float(df["close"].iloc[-1])
                                    if price:
                                        await self.event_bus.publish(Event(EventType.POSITION_REDUCE, {
                                            "symbol": symbol, "strategy": strategy.name,
                                            "price": price, "reduce_pct": rpct, "trader": "ai",
                                            "reason": f"Reduce {rpct}% (#{reduce_count+1}): {cond_str}",
                                        }))
                                        self._signal_cache[reduce_key] = reduce_count + 1
                                        reduce_fired = True
                                        break
                            except Exception:
                                pass

        # Publish exit signal — only for positions opened by THIS strategy
        if has_position and self._executor and not reduce_fired:
            pos = self._executor.get_open_positions().get(symbol, {})
            pos_strategy = pos.get("strategy_name", "") or pos.get("strategy", "")
            if pos_strategy == strategy.name:
                if (pos_side == "long" and exit_signal_long) or (pos_side == "short" and exit_signal_short):
                    open_pos = self._executor.get_open_positions()
                    if symbol in open_pos:
                        price = self.market_data.get_current_price(symbol)
                        if not price and df is not None and len(df) > 0:
                            price = float(df["close"].iloc[-1])
                        if price:
                            # Reset the reduce counter with the SAME key format used
                            # when incrementing it (regression: the reset used to drop
                            # a key without the strategy prefix, so counters never
                            # cleared and reduce conditions stopped firing after 4 hits).
                            reduce_key = f"reduce_count_{strategy.name}_{symbol}_{pos_side}"
                            self._signal_cache.pop(reduce_key, None)
                            await self.event_bus.publish(Event(EventType.POSITION_EXIT, {
                                "symbol": symbol, "strategy": strategy.name,
                                "price": price, "trader": "ai",
                                "reason": f"Exit condition met ({pos_side}) on {interval}",
                            }))

    async def evaluate_all_now(self, publish: bool = False):
        """Evaluate all strategies immediately.
        If publish=True, real trade signals fire (used after breaker reset recovery).
        If publish=False (default), only seed the signal cache."""
        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue
            for interval in strategy.timeframes:
                # Use public interface instead of private attribute
                effective_symbols = strategy.symbols if strategy.symbols else self.market_data.watched_symbols
                for symbol in effective_symbols:
                    try:
                        await self._evaluate(symbol, interval, strategy, publish=publish)
                    except Exception:
                        # Never swallow silently: a failure here means signals stop
                        # flowing entirely, which is indistinguishable from "no signal"
                        # unless it is logged loudly.
                        from loguru import logger
                        logger.exception(
                            f"Strategy evaluation FAILED: {strategy.name} {symbol} {interval} "
                            f"(publish={publish}) — signals will not be produced for this pair"
                        )

    @staticmethod
    def _sanitize(obj):
        """Recursively convert numpy types to Python native types for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: StrategyEngine._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [StrategyEngine._sanitize(v) for v in obj]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    def get_monitor_state(self) -> dict:
        """Return current strategy evaluation status with detailed diagnostics."""
        strategies = []
        for name, s in self._strategies.items():
            prefix = f"{name}|"
            signals = {sig.get("symbol", k.split("|",1)[1] if "|" in k else k): sig
                      for k, sig in self._signal_cache.items() if k.startswith(prefix)}
            strat_signals = {}
            for sym, sig in signals.items():
                entry_diag = sig.get("entry_results", {"long": [], "short": []})
                exit_diag = sig.get("exit_results", {"long": [], "short": []})
                strat_signals[sym] = {
                    "indicator": round(sig["indicator_signal"], 2),
                    "ml_confidence": round(sig["ml_confidence"], 2),
                    "ml_base_rate": (None if sig.get("ml_base_rate") is None
                                     else round(sig["ml_base_rate"], 4)),
                    "ml_score": (None if sig.get("ml_score") is None
                                 else round(sig["ml_score"], 4)),
                    "news_sentiment": round(sig["news_sentiment"], 2),
                    "final_score": round(sig["final_score"], 3),
                    "exit_signal": sig["exit_signal"],
                    "exit_signal_long": sig.get("exit_signal_long", False),
                    "exit_signal_short": sig.get("exit_signal_short", False),
                    "exit_blocks_entry": sig.get("exit_blocks_entry", False),
                    "threshold_met": sig.get("threshold_met", False),
                    "indicators": sig.get("indicators", {}),
                    "weights": sig.get("weights", {}),
                    # P4 diagnostics (absent unless a P4 seam is enabled).
                    "p4": sig.get("p4"),
                    "entry_conditions": {
                        side: [
                            {"condition": c["condition"], "met": c["met"]}
                            for c in conds
                        ]
                        for side, conds in entry_diag.items()
                    },
                    "exit_conditions": {
                        side: [
                            {"condition": c["condition"], "met": c["met"]}
                            for c in conds
                        ]
                        for side, conds in exit_diag.items()
                    },
                }
            strategies.append({
                "name": name,
                "enabled": s.enabled,
                "mode": s.mode,
                "timeframes": s.timeframes,
                "symbols": s.symbols if s.symbols else [],
                "signal_count": len(signals),
                "signals": strat_signals,
            })
        return self._sanitize({
            "strategies": strategies,
            "active_count": sum(1 for s in self._strategies.values() if s.enabled),
            "total_count": len(self._strategies),
        })

    async def stop(self):
        self._running = False
        self.event_bus.unsubscribe(EventType.MARKET_KLINE, self._on_kline)
        self.event_bus.unsubscribe(EventType.ML_PREDICTION, self._on_ml_prediction)
        self.event_bus.unsubscribe(EventType.NEWS_UPDATE, self._on_news_update)
