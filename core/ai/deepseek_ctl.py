import asyncio
import json
import time
from typing import Optional
from openai import AsyncOpenAI
from loguru import logger

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.universe import (
    DEFAULT_WATCHLIST, Universe, load_watchlist, save_watchlist, validate_watchlist,
)
from core.ai.prompts import (
    COIN_SELECTION_PROMPT, STRATEGY_OPTIMIZATION_PROMPT,
    RISK_ADJUSTMENT_PROMPT, MARKET_ASSESSMENT_PROMPT,
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
        #: Duplicate-suppression window for persisted suggestions (seconds).
        #: ``_publish_suggestion`` refuses to INSERT the same category+content
        #: again inside this window so a fast loop cannot flood ``ai_suggestions``.
        self.suggestion_dedupe_window = 600.0
        # Symbols the AI reasons about — the live watchlist when the market-data
        # provider is wired, DEFAULT_WATCHLIST otherwise (never a private copy).
        self._watchlist: list[str] = list(DEFAULT_WATCHLIST)
        #: Coin universe used to validate AI picks (injected; lazily disk-loaded).
        self._universe = None
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

    def wire_universe(self, universe):
        """Inject the coin universe used to validate AI coin selections."""
        self._universe = universe

    def _watched_symbols(self) -> list[str]:
        """The symbols the AI should reason about (live watchlist first)."""
        try:
            live = list(getattr(self._market_data, "watched_symbols", None) or [])
        except Exception:
            live = []
        return live or list(self._watchlist or DEFAULT_WATCHLIST)

    def _resolve_universe(self):
        """Universe for validation: injected, else a disk-only lazy load.

        Never performs a network call — the coin-selection loop must not block on
        the data host, so an unavailable universe simply means "cannot validate"
        (see :func:`validate_watchlist`'s ``tolerate_unknown``).
        """
        if self._universe is not None:
            return self._universe
        try:
            universe = Universe(self.config)
            universe.preload_from_disk()
            self._universe = universe
        except Exception as e:
            logger.warning(f"Coin selection: coin universe unavailable ({e})")
            return None
        return self._universe

    @staticmethod
    def _normalize_ai_symbols(result) -> list[str]:
        """Pull a clean, de-duplicated symbol list out of a raw AI payload.

        Tolerates every shape the model actually returns: ``{"symbols": [...]}``,
        ``{"coins": [...]}``, a bare string, dicts with ``symbol``/``pair``/``name``
        and plain garbage (→ ``[]``).
        """
        if not isinstance(result, dict):
            return []
        raw = result.get("symbols", result.get("coins", []))
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        out: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                item = item.get("symbol") or item.get("pair") or item.get("name")
            sym = str(item or "").strip().upper()
            if sym and sym not in out:
                out.append(sym)
        return out

    async def apply_coin_selection(self, result: dict) -> list[str]:
        """Resolve the AI's chosen symbols and persist them as the watchlist.

        Returns the saved list, or ``[]`` when nothing was written.  Garbage or
        unusable input is a no-op by design: an AI hiccup must never overwrite a
        good watchlist (least of all with an empty one).
        """
        candidates = self._normalize_ai_symbols(result)
        if not candidates:
            logger.warning("Coin selection: AI returned no usable symbols — watchlist unchanged")
            return []

        universe = self._resolve_universe()
        if universe is None:
            accepted, rejected = list(candidates), []
        else:
            accepted, rejected = validate_watchlist(
                candidates, universe, tolerate_unknown=True)
        if rejected:
            logger.info("Coin selection: rejected unknown/non-TRADING symbols: "
                        + ", ".join(rejected))
        if not accepted:
            logger.warning("Coin selection: none of the AI's symbols are tradable "
                           "— watchlist unchanged")
            return []

        try:
            current = await load_watchlist(self.config.db_path, DEFAULT_WATCHLIST)
        except Exception as e:  # pragma: no cover - load_watchlist already guards
            logger.warning(f"Coin selection: could not read the current watchlist ({e})")
            current = list(DEFAULT_WATCHLIST)

        if accepted == current:
            logger.info(f"Coin selection: watchlist already {', '.join(accepted)} — no change")
            return accepted

        await save_watchlist(self.config.db_path, accepted)
        self._watchlist = list(accepted)
        logger.info(f"Coin selection: watchlist updated [{', '.join(current)}] → "
                    f"[{', '.join(accepted)}] (AI selection applied)")
        return accepted

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
                for sym in self._watched_symbols():
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
                data = json.loads(self._extract_json(result))
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
                    data = json.loads(self._extract_json(result))
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
        """Periodic AI strategy generation, retirement, and matrix-based optimization."""
        await asyncio.sleep(300)  # Wait 5 min after startup before first check
        while self._running:
            try:
                if self._lifecycle_manager:
                    await self._lifecycle_manager.generate_strategy()
                    await self._lifecycle_manager.check_and_retire()
                    # Matrix-based analysis: runs every 12h internally (rate-limited)
                    await self._lifecycle_manager.analyze_and_optimize()
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
        except Exception as e:
            logger.debug(f"Heartbeat persist failed for {task_name}: {e}")

    async def _market_assessment_loop(self):
        while self._running:
            try:
                assessment = await self.assess_market()
                if assessment:
                    # Keep the last good assessment in memory AND on disk: the web
                    # layer's `/api/market-state` used to read an `ai_suggestions`
                    # row that category can never legally hold (the table's CHECK
                    # constraint excludes 'market_assessment'), so it always fell
                    # back to "waiting" and blanked out the server-rendered card.
                    self._last_market_assessment = assessment
                    await self._persist_json_setting("ai_market_assessment", assessment)
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

    async def _persist_json_setting(self, key: str, value) -> None:
        """Best-effort JSON blob into ``system_config`` (never raises)."""
        import aiosqlite
        try:
            db = await aiosqlite.connect(self.config.db_path)
            try:
                await db.execute(
                    "INSERT OR REPLACE INTO system_config (key, value, category) "
                    "VALUES (?, ?, 'ai')",
                    (key, json.dumps(value, ensure_ascii=False)))
                await db.commit()
            finally:
                await db.close()
        except Exception as e:
            logger.debug(f"Could not persist {key}: {e}")

    async def _coin_selection_loop(self):
        while self._running:
            try:
                result = await self.select_coins()
                if result:
                    # The suggestion/alert behaviour is unchanged...
                    await self._publish_suggestion("coin_selection", json.dumps(result), 0.7)
                    if self.config.ai_mode == "full_auto":
                        self._last_coin_selection = result
                        # ...and in full_auto the AI's picks actually become the
                        # watchlist (they used to be recorded and then ignored,
                        # so the AI could never change which coins are traded).
                        try:
                            await self.apply_coin_selection(result)
                        except Exception as e:
                            logger.warning(
                                f"Coin selection: could not apply to the watchlist: {e}")
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
                        pct = result.get("position_size_pct", 5.0)
                        # Floor 0.5% (AI can reduce risk) and ceiling 20% (prevent extreme values)
                        pct = min(max(pct, 0.5), 20.0)
                        sl = max(result.get("stop_loss_pct", 2.0), 0.5)       # floor 0.5%
                        sl = min(sl, 15.0)                                      # ceiling 15%
                        lev = max(result.get("leverage", 2), 1)                # floor 1x
                        lev = min(lev, self.config.hard_limits.max_leverage)    # ceiling from hard limits
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

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON object/array from AI response, handling markdown code blocks."""
        import re
        text = text.strip()
        # Try markdown code block: ```json ... ``` or ``` ... ```
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if m:
            return m.group(1).strip()
        # Try raw JSON object or array
        m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if m:
            return m.group(1).strip()
        return text

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
        except Exception as e:
            import asyncio as _asyncio
            err_str = str(e).lower()
            if "authentication" in err_str or "api key" in err_str or "401" in err_str:
                logger.error(f"DeepSeek auth failure — check API key: {e}")
            elif "rate" in err_str or "429" in err_str:
                logger.warning(f"DeepSeek rate limited: {e}")
            elif isinstance(e, _asyncio.TimeoutError) or "timeout" in err_str:
                logger.warning(f"DeepSeek request timed out: {e}")
            else:
                logger.warning(f"DeepSeek API call failed: {e}")
            return None

    async def _publish_suggestion(self, category: str, content: str, confidence: float):
        """Persist an AI suggestion, then publish the event.

        The row is the source of truth for the web layer: the ``/ai`` card,
        ``/partials/ai-suggestions``, the heartbeat's "今日建议 N 条" and the
        approve/reject buttons all read ``ai_suggestions``.  Before this, only
        the (subscriber-less) ``AI_SUGGESTION`` event existed, so the table was
        never written and every one of those surfaces stayed empty forever.

        Both halves are best-effort: a DB problem is logged and never allowed to
        escape into the AI loop (an AI hiccup must not kill market assessment).
        Returns the new row id, or ``None`` when nothing was written.
        """
        status = "applied" if self.config.ai_mode == "full_auto" else "pending"
        row_id = await self._insert_suggestion_row(category, content, confidence, status)
        try:
            await self.event_bus.publish(Event(EventType.AI_SUGGESTION, {
                "id": row_id,
                "category": category,
                "content": content,
                "confidence": confidence,
                "status": status,
            }))
        except Exception as e:
            logger.debug(f"Suggestion event publish failed for {category}: {e}")
        return row_id

    async def _insert_suggestion_row(self, category: str, content: str,
                                     confidence: float, status: str) -> Optional[int]:
        """INSERT one ``ai_suggestions`` row, skipping obvious repeats.

        De-duplication is deliberate: several loops can legitimately produce the
        same recommendation again minutes apart (the assessment interval is much
        shorter than a resolution), and one row per repeat would flood the table
        and the pending list.  The comparison is (category, content) inside
        ``suggestion_dedupe_window`` seconds, done in Python so an existing row's
        ``created_at`` is interpreted as UTC regardless of the SQLite build.
        """
        import aiosqlite
        try:
            db = await aiosqlite.connect(self.config.db_path)
            try:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT id, created_at FROM ai_suggestions "
                    "WHERE category = ? AND content = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (category, content))
                dup = await cursor.fetchone()
                if dup is not None and self._is_recent(dup["created_at"]):
                    logger.debug(
                        f"Suggestion dedupe: {category} unchanged within "
                        f"{int(self.suggestion_dedupe_window)}s — not re-inserted")
                    return None
                cursor = await db.execute(
                    "INSERT INTO ai_suggestions (category, content, rationale, confidence, status) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (category, content, self._suggestion_rationale(category, confidence),
                     float(confidence), status))
                await db.commit()
                row_id = cursor.lastrowid
            finally:
                await db.close()
        except Exception as e:
            logger.warning(f"Suggestion persist failed for {category}: {e}")
            return None
        logger.info(f"AI suggestion persisted: {category} (id={row_id}, confidence={confidence})")
        return row_id

    def _is_recent(self, created_at) -> bool:
        """True when a SQLite ``created_at`` is inside the dedupe window (UTC)."""
        import datetime as _dt
        if not created_at:
            return False
        try:
            stamp = str(created_at)[:19]
            parsed = _dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return False
        age = (_dt.datetime.utcnow() - parsed).total_seconds()
        return -60.0 <= age < self.suggestion_dedupe_window

    def _suggestion_rationale(self, category: str, confidence: float) -> str:
        """One-line, human-readable provenance for the suggestion row."""
        labels = {
            "coin_selection": "AI 币种选择",
            "strategy_optimization": "AI 策略优化",
            "risk_adjustment": "AI 风控调整",
            "market_assessment": "AI 市场评估",
            "news_analysis": "AI 新闻分析",
        }
        return (f"{labels.get(category, category)} — 后台定时分析输出 "
                f"(model={self.config.ai_model or 'n/a'}, confidence={confidence:.2f}, "
                f"mode={self.config.ai_mode})")

    def _build_market_context(self) -> str:
        """Build a context string with current portfolio and market state."""
        parts = []
        if self._market_data:
            try:
                for sym in self._watched_symbols()[:3]:
                    price = self._market_data.get_current_price(sym)
                    if price:
                        parts.append(f"{sym}: {price:.2f}")
            except Exception as e:
                logger.debug(f"AI context: price section unavailable ({e})")
        if self._executor:
            try:
                positions = self._executor.get_open_positions()
                if positions:
                    pos_list = [f"{s}: {p['side']} qty={p['quantity']:.4f} @ {p['entry_price']:.2f}" for s, p in positions.items()]
                    parts.append(f"Open positions ({len(positions)}): " + "; ".join(pos_list))
            except Exception as e:
                logger.debug(f"AI context: position section unavailable ({e})")
        if self._risk_manager:
            try:
                bal = self._risk_manager._account_balance
                parts.append(f"Account balance: {bal:.0f} USDT")
            except Exception as e:
                logger.debug(f"AI context: balance section unavailable ({e})")
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
                return json.loads(self._extract_json(result))
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
                return json.loads(self._extract_json(result))
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
                return json.loads(self._extract_json(result))
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
                return json.loads(self._extract_json(result))
            except json.JSONDecodeError:
                return None
        return None

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
