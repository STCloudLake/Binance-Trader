import asyncio
import pandas as pd

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.provider import MarketDataProvider
from core.strategy.loader import StrategyLoader, StrategyConfig
from core.strategy.indicators import compute_all, evaluate_condition


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
        self._news_sentiment: dict[str, float] = {}

    def wire_executor(self, executor):
        self._executor = executor

    async def start(self):
        self._running = True
        self.event_bus.subscribe(EventType.MARKET_KLINE, self._on_kline)
        self.event_bus.subscribe(EventType.ML_PREDICTION, self._on_ml_prediction)
        self.event_bus.subscribe(EventType.NEWS_UPDATE, self._on_news_update)
        all_strategies = self.loader.load_all()
        for s in all_strategies:
            if not s.timeframes:
                from loguru import logger
                logger.warning(f"Strategy '{s.name}' has no timeframes configured — will never evaluate!")
        self._strategies = {s.name: s for s in all_strategies}

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
            await self._evaluate(symbol, interval, strategy)

    async def _on_ml_prediction(self, event: Event):
        symbol = event.data.get("symbol", "")
        self._ml_confidence[symbol] = event.data.get("confidence", 0.5)

    async def _on_news_update(self, event: Event):
        symbol = event.data.get("symbol", "")
        self._news_sentiment[symbol] = event.data.get("sentiment", 0.0)

    async def _evaluate(self, symbol: str, interval: str, strategy: StrategyConfig, publish: bool = True):
        df = await self.market_data.get_historical(symbol, interval)
        if df is None or len(df) < 50:
            return

        df = compute_all(df, strategy.indicators)

        # Evaluate entry conditions individually for diagnostics.
        # Each side independently: ANY condition met = side active.
        entry_results = {"long": [], "short": []}
        long_active = False
        short_active = False
        for side in ["long", "short"]:
            conditions = strategy.entry_conditions.get(side, [])
            for cond in conditions:
                mask = evaluate_condition(df, cond)
                met = bool(hasattr(mask, 'iloc') and mask.iloc[-1])
                entry_results[side].append({"condition": cond, "met": met})
                if met and side == "long":
                    long_active = True
                elif met and side == "short":
                    short_active = True

        # If both sides are active, the signal is ambiguous — don't trade.
        if long_active and short_active:
            indicator_signal = 0.0
        elif long_active:
            indicator_signal = 1.0
        elif short_active:
            indicator_signal = -1.0
        else:
            indicator_signal = 0.0

        # Evaluate exit conditions per side (not global — long exit shouldn't block short entry)
        exit_results = {"long": [], "short": []}
        exit_signal_long = False
        exit_signal_short = False
        for side in ["long", "short"]:
            conditions = strategy.exit_conditions.get(side, [])
            for cond in conditions:
                mask = evaluate_condition(df, cond)
                met = bool(hasattr(mask, 'iloc') and mask.iloc[-1])
                exit_results[side].append({"condition": cond, "met": met})
                if met:
                    if side == "long":
                        exit_signal_long = True
                    else:
                        exit_signal_short = True

        ml_conf = self._ml_confidence.get(symbol, 0.5)  # 0.5 = neutral (no prediction)
        news_sent = self._news_sentiment.get(symbol, 0.0)

        # Transform ML confidence (0-1, P(price up)) into directional signal (-1 to +1).
        # Neutral (0.5) → 0 contribution. Bullish (1.0) → +1. Bearish (0.0) → -1.
        ml_directional = (ml_conf - 0.5) * 2

        w = self.config.signal_weights
        ml_weight = strategy.ml_config.weight if (strategy.ml_config and strategy.ml_config.enabled) else w.ml
        total_weight = w.indicator + ml_weight + w.news
        if total_weight > 0:
            final_score = (
                (indicator_signal * w.indicator +
                 ml_directional * ml_weight +
                 news_sent * w.news) / total_weight
            )
        else:
            final_score = float(indicator_signal)

        # Determine entry side from the signal
        entry_side = "long" if final_score > 0 else "short"

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

        key = f"{strategy.name}|{symbol}"
        self._signal_cache[key] = {
            "strategy": strategy.name,
            "symbol": symbol,
            "indicator_signal": indicator_signal,
            "ml_confidence": ml_conf,
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
            "weights": {"indicator": w.indicator, "ml": ml_weight, "news": w.news},
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
                    reduce_key = f"reduce_count_{symbol}_{side}"
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
                            reduce_key = f"reduce_count_{symbol}_{pos_side}"
                            self._signal_cache.pop(reduce_key, None)
                            await self.event_bus.publish(Event(EventType.POSITION_EXIT, {
                                "symbol": symbol, "strategy": strategy.name,
                                "price": price, "trader": "ai",
                                "reason": f"Exit condition met ({pos_side}) on {interval}",
                            }))

    def get_signal(self, symbol: str, strategy_name: str = None) -> dict | None:
        """Get latest signal for a symbol. If strategy_name is given, match exactly; otherwise return any."""
        if strategy_name:
            return self._signal_cache.get(f"{strategy_name}|{symbol}")
        # Return first matching signal for the symbol
        for key, sig in self._signal_cache.items():
            if key.endswith(f"|{symbol}"):
                return sig
        return None

    def get_strategies(self) -> list[dict]:
        return [s.model_dump() for s in self._strategies.values()]

    async def evaluate_all_now(self, publish: bool = False):
        """Evaluate all strategies immediately.
        If publish=True, real trade signals fire (used after breaker reset recovery).
        If publish=False (default), only seed the signal cache."""
        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue
            for interval in strategy.timeframes:
                for symbol in self.market_data._watched_symbols:
                    try:
                        await self._evaluate(symbol, interval, strategy, publish=publish)
                    except Exception:
                        pass

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
                    "news_sentiment": round(sig["news_sentiment"], 2),
                    "final_score": round(sig["final_score"], 3),
                    "exit_signal": sig["exit_signal"],
                    "exit_signal_long": sig.get("exit_signal_long", False),
                    "exit_signal_short": sig.get("exit_signal_short", False),
                    "exit_blocks_entry": sig.get("exit_blocks_entry", False),
                    "threshold_met": sig.get("threshold_met", False),
                    "indicators": sig.get("indicators", {}),
                    "weights": sig.get("weights", {}),
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
                "signal_count": len(signals),
                "signals": strat_signals,
            })
        return self._sanitize({
            "strategies": strategies,
            "active_count": sum(1 for s in self._strategies.values() if s.enabled),
            "total_count": len(self._strategies),
        })

    async def add_strategy(self, config: StrategyConfig):
        self.loader.save(config)
        self._strategies[config.name] = config

    async def remove_strategy(self, name: str):
        self.loader.delete(name)
        self._strategies.pop(name, None)

    async def stop(self):
        self._running = False
        self.event_bus.unsubscribe(EventType.MARKET_KLINE, self._on_kline)
        self.event_bus.unsubscribe(EventType.ML_PREDICTION, self._on_ml_prediction)
        self.event_bus.unsubscribe(EventType.NEWS_UPDATE, self._on_news_update)
