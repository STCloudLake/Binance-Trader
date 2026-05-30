import asyncio
import json
import time
from typing import Optional
from openai import AsyncOpenAI
from loguru import logger

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.ai.prompts import (
    COIN_SELECTION_PROMPT, STRATEGY_OPTIMIZATION_PROMPT,
    RISK_ADJUSTMENT_PROMPT, MARKET_ASSESSMENT_PROMPT, NEWS_ANALYSIS_PROMPT,
)


class DeepSeekController:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.client: Optional[AsyncOpenAI] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._market_data: Optional[object] = None
        self._executor: Optional[object] = None
        self._risk_manager: Optional[object] = None
        self._strategy_engine: Optional[object] = None
        # AI decision cache
        self._last_coin_selection: Optional[dict] = None
        self._last_market_assessment: Optional[dict] = None
        self._lifecycle_manager = None
        # Heartbeat tracking
        self._last_run: dict[str, float] = {}
        self._run_count: dict[str, int] = {}
        self._run_errors: dict[str, int] = {}

    def wire(self, market_data, executor, risk_manager, strategy_engine=None):
        self._market_data = market_data
        self._executor = executor
        self._risk_manager = risk_manager
        self._strategy_engine = strategy_engine

    def wire_lifecycle(self, lifecycle_manager):
        self._lifecycle_manager = lifecycle_manager

    def _build_breaker_context(self, breaker_data: dict = None) -> str:
        """Build context string for breaker-related AI decisions."""
        parts = []
        if breaker_data:
            parts.append(f"Trip reason: {breaker_data.get('reason', 'Unknown')}")
            parts.append(f"Daily drawdown: {breaker_data.get('daily_drawdown_pct', 0):.2f}%")
            parts.append(f"Daily PnL: {breaker_data.get('daily_pnl', 0):.2f} USDT")
            parts.append(f"Consecutive losses: {breaker_data.get('consecutive_losses', 0)}")

        if self._risk_manager:
            try:
                bal = self._risk_manager._account_balance
                parts.append(f"Account balance: {bal:.0f} USDT")
                cb = self._risk_manager.breaker
                parts.append(f"Breaker tripped: {cb.is_tripped}")
                if cb.is_tripped:
                    parts.append(f"Trip reason: {cb.trip_reason}")
                    parts.append(f"Peak equity: {cb.peak_equity:.0f} USDT")
                    parts.append(f"Current equity: {cb.current_equity:.0f} USDT")
            except Exception as e:
                logger.warning(f"Breaker context: failed to read balance/breaker state: {e}")

        if self._executor:
            try:
                positions = self._executor.get_open_positions()
                if positions:
                    pos_list = []
                    for s, p in positions.items():
                        upnl = p.get("unrealized_pnl", 0)
                        pos_list.append(f"{s}: {p['side']} qty={p['quantity']:.4f} @ {p['entry_price']:.2f} uPnL={upnl:.2f}")
                    parts.append(f"Open positions ({len(positions)}): " + "; ".join(pos_list))
                else:
                    parts.append("Open positions: none")
            except Exception as e:
                logger.warning(f"Breaker context: failed to read open positions: {e}")

        if self._market_data:
            try:
                prices = []
                for sym in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]:
                    price = self._market_data.get_current_price(sym)
                    if price:
                        prices.append(f"{sym}={price:.2f}")
                parts.append("Current prices: " + ", ".join(prices))
            except Exception as e:
                logger.warning(f"Breaker context: failed to read market prices: {e}")

        return "\n".join(parts)

    async def decide_breaker_action(self, breaker_data: dict) -> str:
        """Ask DeepSeek which breaker action to take. Returns action string. Timeout 15s, fallback close_all."""
        from core.ai.prompts import BREAKER_ACTION_PROMPT
        context = self._build_breaker_context(breaker_data)
        prompt = BREAKER_ACTION_PROMPT.format(context=context)

        try:
            result = await asyncio.wait_for(
                self._call_deepseek(
                    "You are a risk management expert. Always respond in valid JSON.",
                    prompt
                ),
                timeout=15.0
            )
            if result:
                data = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
                action = data.get("action", "close_all")
                logger.info(f"AI breaker decision: {action} — {data.get('rationale', '')}")
                if action in ("block_only", "tighten_stops", "close_all", "close_worst"):
                    return action
        except asyncio.TimeoutError:
            logger.warning("AI breaker decision timed out, fallback to close_all")
        except Exception as e:
            logger.warning(f"AI breaker decision failed: {e}, fallback to close_all")

        return "close_all"

    async def _breaker_recovery_loop(self):
        """Background task: periodically evaluate if breaker can be reset. Exits when breaker is no longer tripped."""
        from core.ai.prompts import BREAKER_RECOVERY_PROMPT
        await asyncio.sleep(120)  # Wait 2 minutes before first evaluation

        while self._running and self._risk_manager:
            try:
                cb = self._risk_manager.breaker
                if not cb.is_tripped:
                    logger.info("Breaker recovery loop: breaker already reset, exiting")
                    return

                context = self._build_breaker_context()
                prompt = BREAKER_RECOVERY_PROMPT.format(context=context)
                result = await self._call_deepseek(
                    "You are a risk management expert. Always respond in valid JSON.",
                    prompt
                )

                if result:
                    data = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
                    if data.get("reset"):
                        cb.reset_trip()
                        cb.reset_daily()
                        logger.info(f"AI recovery: breaker reset — {data.get('reason', '')}")
                        await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {
                            "level": "info",
                            "type": "breaker_recovery",
                            "message": f"AI 已恢复交易: {data.get('reason', '自动恢复')}",
                        }))
                        # Force immediate re-evaluation of all strategies so
                        # medium/long-term strategies don't wait for next kline.
                        if self._strategy_engine:
                            try:
                                await self._strategy_engine.evaluate_all_now(publish=True)
                                logger.info("AI recovery: forced strategy re-evaluation complete")
                            except Exception as e:
                                logger.warning(f"AI recovery: strategy re-evaluation failed: {e}")
                        await self._heartbeat("breaker_recovery", True)
                        return
                    else:
                        logger.info(f"AI recovery: keep breaker tripped — {data.get('reason', '')}")

                await self._heartbeat("breaker_recovery", True)
            except Exception as e:
                logger.warning(f"Breaker recovery evaluation failed: {e}")
                await self._heartbeat("breaker_recovery", False)

            await asyncio.sleep(300)  # Re-check every 5 minutes

    async def start(self):
        api_key = self.config.deepseek_api_key
        if not api_key:
            return
        self.client = AsyncOpenAI(api_key=api_key, base_url=self.config.ai_base_url)
        self._running = True
        self._tasks.append(asyncio.create_task(self._market_assessment_loop()))
        self._tasks.append(asyncio.create_task(self._coin_selection_loop()))
        self._tasks.append(asyncio.create_task(self._strategy_optimization_loop()))
        self._tasks.append(asyncio.create_task(self._risk_adjustment_loop()))
        if self._lifecycle_manager:
            self._tasks.append(asyncio.create_task(self._lifecycle_loop()))

    async def _lifecycle_loop(self):
        """Periodic AI strategy generation and retirement evaluation."""
        await asyncio.sleep(300)  # Wait 5 min after startup before first check
        while self._running:
            try:
                if self._lifecycle_manager:
                    await self._lifecycle_manager.generate_strategy()
                    await self._lifecycle_manager.check_and_retire()
            except Exception as e:
                logger.warning(f"Lifecycle loop error: {e}")
            await asyncio.sleep(3600)  # Check every hour

    async def _heartbeat(self, task_name: str, success: bool):
        """Record that an AI task just ran."""
        import aiosqlite as aio
        now = time.time()
        self._last_run[task_name] = now
        self._run_count[task_name] = self._run_count.get(task_name, 0) + 1
        if not success:
            self._run_errors[task_name] = self._run_errors.get(task_name, 0) + 1
        try:
            db = await aio.connect(self.config.db_path)
            await db.execute(
                "INSERT OR REPLACE INTO system_config (key, value, category) VALUES (?, ?, 'ai_heartbeat')",
                (f"ai_last_{task_name}", str(now)))
            await db.execute(
                "INSERT OR REPLACE INTO system_config (key, value, category) VALUES (?, ?, 'ai_heartbeat')",
                (f"ai_count_{task_name}", str(self._run_count.get(task_name, 0))))
            await db.commit()
            await db.close()
        except Exception:
            pass

    async def _market_assessment_loop(self):
        while self._running:
            try:
                assessment = await self.assess_market()
                if assessment:
                    await self.event_bus.publish(Event(EventType.AI_MARKET_STATE, assessment))
                    if self.config.ai_mode in ("semi_auto", "full_auto"):
                        weights = assessment.get("signal_weights", {})
                        if weights:
                            self.config.update_signal_weights(**weights)
                await self._heartbeat("market_assessment", True)
            except Exception as e:
                logger.warning(f"Market assessment failed: {e}")
                await self._heartbeat("market_assessment", False)
            await asyncio.sleep(self.config.ai_task_intervals.get("market_assessment", 3600))

    async def _coin_selection_loop(self):
        while self._running:
            try:
                result = await self.select_coins()
                if result:
                    await self._publish_suggestion("coin_selection", json.dumps(result), 0.7)
                    if self.config.ai_mode == "full_auto":
                        self._last_coin_selection = result
                await self._heartbeat("coin_selection", True)
            except Exception as e:
                logger.warning(f"Coin selection failed: {e}")
                await self._heartbeat("coin_selection", False)
            await asyncio.sleep(self.config.ai_task_intervals.get("coin_selection", 14400))

    async def _strategy_optimization_loop(self):
        while self._running:
            try:
                result = await self.optimize_strategy()
                if result:
                    await self._publish_suggestion("strategy_optimization", json.dumps(result), 0.6)
                await self._heartbeat("strategy_optimization", True)
            except Exception as e:
                logger.warning(f"Strategy optimization failed: {e}")
                await self._heartbeat("strategy_optimization", False)
            await asyncio.sleep(self.config.ai_task_intervals.get("strategy_optimization", 86400))

    async def _risk_adjustment_loop(self):
        while self._running:
            try:
                result = await self.adjust_risk()
                if result:
                    await self._publish_suggestion("risk_adjustment", json.dumps(result), 0.7)
                    if self.config.ai_mode == "full_auto":
                        pct = max(result.get("position_size_pct", 5.0), 1.0)  # floor 1%
                        sl = max(result.get("stop_loss_pct", 2.0), 0.5)       # floor 0.5%
                        lev = max(result.get("leverage", 2), 1)                # floor 1x
                        self.config.update_soft_params(
                            risk_appetite=result.get("risk_appetite", "balanced"),
                            position_size_pct=pct,
                            stop_loss_pct=sl,
                            leverage=lev,
                        )
                await self._heartbeat("risk_adjustment", True)
            except Exception as e:
                logger.warning(f"Risk adjustment failed: {e}")
                await self._heartbeat("risk_adjustment", False)
            await asyncio.sleep(self.config.ai_task_intervals.get("risk_adjustment", 86400))

    async def _call_deepseek(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.client:
            return None
        try:
            response = await self.client.chat.completions.create(
                model=self.config.ai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2000,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception:
            return None

    async def _publish_suggestion(self, category: str, content: str, confidence: float):
        status = "approved" if self.config.ai_mode == "full_auto" else "pending"
        await self.event_bus.publish(Event(EventType.AI_SUGGESTION, {
            "category": category,
            "content": content,
            "confidence": confidence,
            "status": status,
        }))

    def _build_market_context(self) -> str:
        """Build a context string with current portfolio and market state."""
        parts = []
        if self._market_data:
            try:
                for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                    price = self._market_data.get_current_price(sym)
                    if price:
                        parts.append(f"{sym}: {price:.2f}")
            except Exception:
                pass
        if self._executor:
            try:
                positions = self._executor.get_open_positions()
                if positions:
                    pos_list = [f"{s}: {p['side']} qty={p['quantity']:.4f} @ {p['entry_price']:.2f}" for s, p in positions.items()]
                    parts.append(f"Open positions ({len(positions)}): " + "; ".join(pos_list))
            except Exception:
                pass
        if self._risk_manager:
            try:
                bal = self._risk_manager._account_balance
                parts.append(f"Account balance: {bal:.0f} USDT")
            except Exception:
                pass
        parts.append(f"Current weights: indicator={self.config.signal_weights.indicator}, ml={self.config.signal_weights.ml}, news={self.config.signal_weights.news}")
        parts.append(f"Risk params: appetite={self.config.soft_params.risk_appetite}, pos_size={self.config.soft_params.position_size_pct}%, sl={self.config.soft_params.stop_loss_pct}%, leverage={self.config.soft_params.leverage}")
        return "\n".join(parts)

    async def assess_market(self) -> Optional[dict]:
        context = self._build_market_context()
        prompt = MARKET_ASSESSMENT_PROMPT.format(context=context)
        result = await self._call_deepseek(
            "You are a professional crypto market analyst. Always respond in valid JSON.",
            prompt
        )
        if result:
            try:
                return json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
            except json.JSONDecodeError:
                return {"market_regime": result[:200]}
        return None

    async def select_coins(self) -> Optional[dict]:
        context = self._build_market_context()
        prompt = COIN_SELECTION_PROMPT.format(context=context)
        result = await self._call_deepseek(
            "You are a professional cryptocurrency portfolio analyst. Always respond in valid JSON.",
            prompt
        )
        if result:
            try:
                return json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
            except json.JSONDecodeError:
                return None
        return None

    async def optimize_strategy(self) -> Optional[dict]:
        context = self._build_market_context()
        prompt = STRATEGY_OPTIMIZATION_PROMPT.format(context=context)
        result = await self._call_deepseek(
            "You are a quantitative trading strategist. Always respond in valid JSON.",
            prompt
        )
        if result:
            try:
                return json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
            except json.JSONDecodeError:
                return None
        return None

    async def adjust_risk(self) -> Optional[dict]:
        context = self._build_market_context()
        prompt = RISK_ADJUSTMENT_PROMPT.format(context=context)
        result = await self._call_deepseek(
            "You are a risk management expert. Always respond in valid JSON.",
            prompt
        )
        if result:
            try:
                return json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
            except json.JSONDecodeError:
                return None
        return None

    async def analyze_news(self, title: str, summary: str, symbol: str) -> Optional[dict]:
        prompt = NEWS_ANALYSIS_PROMPT.format(title=title, summary=summary, symbol=symbol)
        result = await self._call_deepseek(
            "You are a financial news analyst. Always respond in valid JSON.",
            prompt
        )
        if result:
            try:
                return json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
            except json.JSONDecodeError:
                return None
        return None

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
