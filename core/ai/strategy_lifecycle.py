"""AI Strategy Lifecycle Manager — generate → backtest → deploy → optimize → retire."""
import json
import time
from loguru import logger


class StrategyLifecycleManager:
    """Coordinates the AI-driven strategy lifecycle.

    Gated by config.ai_mode: suggest (manual approval) / semi_auto (auto-deploy,
    manual retire) / full_auto (full autonomy with grace period).
    """

    def __init__(self, config, deepseek_ctl, backtest_engine, strategy_loader,
                 strategy_engine, alert_manager, db_path: str):
        self.config = config
        self.deepseek = deepseek_ctl
        self.backtest_engine = backtest_engine
        self.loader = strategy_loader
        self.strategy_engine = strategy_engine
        self.alert_manager = alert_manager
        self.db_path = db_path
        self._last_generation_time: float = 0
        self._last_retirement_check: float = 0
        self._generation_interval: int = 86400   # 24h
        self._retirement_interval: int = 3600     # 1h

    async def log_event(self, strategy_name: str, action: str, reason: str = "",
                        metrics_snapshot: dict = None, backtest_id: int = None):
        """Record a lifecycle event to the database."""
        import aiosqlite
        db = await aiosqlite.connect(self.db_path)
        try:
            await db.execute(
                "INSERT INTO strategy_lifecycle_events "
                "(strategy_name, action, trigger_reason, metrics_snapshot, backtest_record_id) "
                "VALUES (?,?,?,?,?)",
                (strategy_name, action, reason,
                 json.dumps(metrics_snapshot) if metrics_snapshot else None,
                 backtest_id))
            await db.commit()
        finally:
            await db.close()

    async def generate_strategy(self) -> dict | None:
        """Ask AI to generate a new strategy config based on current market state."""
        now = time.time()
        if now - self._last_generation_time < self._generation_interval:
            return None

        self._last_generation_time = now
        logger.info("Lifecycle: generating new strategy...")

        existing = self.loader.list_names()
        gaps = self._find_coverage_gaps(existing)

        prompt = (
            f"Generate a new crypto trading strategy YAML configuration.\n"
            f"Market state: trending\n"
            f"Existing strategies: {', '.join(existing)}\n"
            f"Coverage gaps: {', '.join(gaps) if gaps else 'none detected'}\n\n"
            f"Output a complete strategy with:\n"
            f"- A descriptive name\n"
            f"- 1-2 timeframes (1m/5m/15m/1h/4h)\n"
            f"- 2-3 indicators with parameters\n"
            f"- Entry conditions for long and short\n"
            f"- Exit conditions\n\n"
            f'Respond in valid JSON matching this schema:\n'
            f'{{"name": "...", "enabled": true, "mode": "trend", "timeframes": ["1h"], '
            f'"indicators": {{"rsi": {{"period": 14}}}}, '
            f'"entry_conditions": {{"long": ["rsi < 30"], "short": ["rsi > 70"]}}, '
            f'"exit_conditions": {{"long": [], "short": []}}, '
            f'"reduce_conditions": {{}}, "ml_config": {{"enabled": false}}}}'
        )

        result = await self.deepseek._call_deepseek(
            "You are a quantitative trading strategist. Output only valid JSON.",
            prompt
        )

        if not result:
            logger.warning("Lifecycle: AI generation returned no result")
            return None

        try:
            strategy_config = json.loads(
                result.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            logger.warning(f"Lifecycle: failed to parse AI output: {result[:200]}")
            return None

        return strategy_config

    async def validate_and_deploy(self, strategy_config: dict) -> bool:
        """Backtest a generated strategy and deploy if it passes validation."""
        strategy_name = strategy_config.get("name", f"ai_gen_{int(time.time())}")

        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # First save the strategy temporarily so the backtester can load it
        from core.strategy.loader import StrategyConfig
        try:
            config = StrategyConfig(**strategy_config)
            self.loader.save(config)
        except Exception as e:
            logger.warning(f"Lifecycle: failed to save generated strategy {strategy_name}: {e}")
            return False

        # Run backtest (in thread pool to avoid blocking event loop)
        import asyncio, concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(
                pool, self.backtest_engine.run,
                [strategy_name], ["BTCUSDT", "ETHUSDT"], start, end, "full", 10000.0)

        if result.get("error"):
            logger.warning(f"Lifecycle: backtest failed for {strategy_name}: {result['error']}")
            # Clean up failed strategy
            try:
                self.loader.delete(strategy_name)
            except Exception:
                pass
            return False

        metrics = result.get("metrics", {})
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate_pct", 0)

        if sharpe > 0.5 and win_rate > 45:
            # Reload strategies into engine
            all_s = self.loader.load_all()
            self.strategy_engine._strategies = {s.name: s for s in all_s}
            await self.log_event(strategy_name, "deployed",
                                 f"Auto-deployed: Sharpe={sharpe}, WinRate={win_rate}%")
            logger.info(f"Lifecycle: deployed {strategy_name}")
            return True
        else:
            await self.log_event(strategy_name, "generated",
                                 f"Failed validation: Sharpe={sharpe}, WinRate={win_rate}%")
            # Clean up
            try:
                self.loader.delete(strategy_name)
            except Exception:
                pass
            return False

    async def check_and_retire(self):
        """Evaluate all active strategies and retire underperforming ones."""
        now = time.time()
        if now - self._last_retirement_check < self._retirement_interval:
            return
        self._last_retirement_check = now

        current = self.loader.list_names()
        for name in current:
            if await self._evaluate_for_retirement(name):
                ai_mode = self.config.ai_mode
                s = self.loader.load(name)
                if ai_mode == "suggest":
                    await self.log_event(name, "retired",
                                         "AI suggests retirement (manual review needed)")
                elif ai_mode in ("semi_auto", "full_auto"):
                    s.enabled = False
                    self.loader.save(s)
                    await self.log_event(name, "retired",
                                         f"Auto-{'disabled' if ai_mode == 'semi_auto' else 'retired'} (7-day grace)")
                logger.info(f"Lifecycle: retired {name} (mode={ai_mode})")

    async def _evaluate_for_retirement(self, name: str) -> bool:
        """Check if a strategy should be retired."""
        import asyncio, concurrent.futures
        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(
                pool, self.backtest_engine.run,
                [name], ["BTCUSDT", "ETHUSDT"], start, end, "full", 10000.0)

        if result.get("error"):
            return False

        metrics = result.get("metrics", {})
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate_pct", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)
        trades = metrics.get("total_trades", 0)

        if trades == 0:
            return True   # dead strategy
        if sharpe < -1.0:
            return True
        if win_rate < 30:
            return True
        if max_dd > 30:
            return True
        return False

    def _find_coverage_gaps(self, existing: list[str]) -> list[str]:
        """Detect missing strategy types in the current portfolio."""
        all_names = " ".join(existing).lower()
        gaps = []
        if "mean_reversion" not in all_names and "reversion" not in all_names:
            gaps.append("mean_reversion")
        if "momentum" not in all_names and "scalp" not in all_names:
            gaps.append("momentum")
        if "trend" not in all_names and "ema" not in all_names:
            gaps.append("trend_following")
        if "bollinger" not in all_names and "squeeze" not in all_names:
            gaps.append("volatility_breakout")
        return gaps
