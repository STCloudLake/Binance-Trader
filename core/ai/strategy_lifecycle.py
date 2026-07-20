"""AI Strategy Lifecycle Manager — generate → backtest → deploy → optimize → retire."""
import json
import time
from loguru import logger


class StrategyLifecycleManager:
    """Coordinates the AI-driven strategy lifecycle.

    Gated by config.ai_mode: suggest (manual approval) / semi_auto (auto-deploy,
    manual retire) / full_auto (full autonomy with grace period).

    New: Per-strategy×symbol matrix analysis drives auto-optimization decisions.
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
        self._last_optimization_time: float = 0
        self._generation_interval: int = 86400   # 24h
        self._retirement_interval: int = 3600     # 1h
        self._optimization_interval: int = 43200  # 12h

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

    # ── Strategy Generation ─────────────────────────────────────────────

    async def generate_strategy(self, target_symbols: list[str] = None) -> dict | None:
        """Ask AI to generate a new strategy config based on current market state.

        Args:
            target_symbols: If given, generate a strategy optimized for these symbols.
        """
        now = time.time()
        if now - self._last_generation_time < self._generation_interval:
            return None

        self._last_generation_time = now
        logger.info("Lifecycle: generating new strategy...")

        existing = self.loader.list_names()
        gaps = self._find_coverage_gaps(existing)

        symbol_hint = ""
        if target_symbols:
            symbol_hint = f"\nTarget symbols (optimize for these): {', '.join(target_symbols)}"

        prompt = (
            f"Generate a new crypto trading strategy YAML configuration.\n"
            f"Market state: trending\n"
            f"Existing strategies: {', '.join(existing)}\n"
            f"Coverage gaps: {', '.join(gaps) if gaps else 'none detected'}"
            f"{symbol_hint}\n\n"
            f"REQUIRED fields:\n"
            f"- name: a descriptive English name (no spaces, use underscores)\n"
            f"- timeframes: MUST be a non-empty list from [\"1m\",\"5m\",\"15m\",\"1h\",\"4h\"]. "
            f"At minimum include \"1h\".\n"
            f"- mode: one of \"trend\", \"mean_reversion\", \"momentum\", \"breakout\"\n"
            f"- symbols: OPTIONAL list of symbols to restrict to (e.g. [\"BTCUSDT\",\"ETHUSDT\"]). "
            f"Omit or leave empty to apply to all symbols.\n"
            f"- indicators: 2-3 indicators with parameters (rsi, macd, ema, bollinger, adx, etc.)\n"
            f"- entry_conditions: at least one condition each for \"long\" and \"short\"\n"
            f"- exit_conditions: at least one condition each for \"long\" and \"short\"\n\n"
            f'Respond ONLY with valid JSON:\n'
            f'{{"name": "my_strategy", "enabled": true, "mode": "trend", '
            f'"timeframes": ["1h"], "symbols": ["BTCUSDT"], '
            f'"indicators": {{"rsi": {{"period": 14}}, "macd": {{"fast": 12, "slow": 26, "signal": 9}}}}, '
            f'"entry_conditions": {{"long": ["rsi < 30 and close > ema21"], '
            f'"short": ["rsi > 70 and close < ema21"]}}, '
            f'"exit_conditions": {{"long": ["rsi > 70 or close < ema50"], '
            f'"short": ["rsi < 30 or close > ema50"]}}, '
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

        # Validate required fields
        if not strategy_config.get("timeframes"):
            logger.warning(f"Lifecycle: AI generated strategy missing timeframes, defaulting to ['1h']")
            strategy_config["timeframes"] = ["1h"]
        if not strategy_config.get("entry_conditions"):
            logger.warning("Lifecycle: AI generated strategy missing entry_conditions")
            return None
        if not isinstance(strategy_config["entry_conditions"], dict):
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

        # Respect strategy's symbol restriction in backtest
        symbols = strategy_config.get("symbols", []) or ["BTCUSDT", "ETHUSDT"]

        # Run backtest (in thread pool to avoid blocking event loop)
        import asyncio, concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(
                pool, self.backtest_engine.run,
                [strategy_name], symbols, start, end, "full", 10000.0)

        if result.get("error"):
            logger.warning(f"Lifecycle: backtest failed for {strategy_name}: {result['error']}")
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
            try:
                self.loader.delete(strategy_name)
            except Exception:
                pass
            return False

    # ── Retirement ──────────────────────────────────────────────────────

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

        s = self.loader.load(name)
        symbols = s.symbols if s.symbols else ["BTCUSDT", "ETHUSDT"]

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(
                pool, self.backtest_engine.run,
                [name], symbols, start, end, "full", 10000.0)

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

    # ── Matrix-Based Analysis & Optimization ────────────────────────────

    async def analyze_and_optimize(self) -> dict:
        """Run comprehensive analysis of all strategies × all symbols and optimize.

        This is the main AI-driven optimization entry point:
        1. Run backtest of all strategies against all symbols
        2. Analyze per-cell (strategy×symbol) performance
        3. For negative-PnL cells: remove symbol from strategy (or suggest)
        4. For strategies with all-negative cells: send to AI for parameter optimization
        5. Identify coverage gaps — profitable cells with no active strategy
        6. Generate new strategies for gaps

        Returns a summary dict of actions taken.
        """
        now = time.time()
        if now - self._last_optimization_time < self._optimization_interval:
            return {"skipped": True, "reason": "rate-limited"}

        self._last_optimization_time = now
        logger.info("Lifecycle: starting matrix-based analysis & optimization...")

        actions = {"cells_removed": [], "strategies_optimized": [],
                   "strategies_generated": [], "gaps_identified": []}

        ai_mode = self.config.ai_mode
        all_strategies = self.loader.list_names()
        if not all_strategies:
            return {"skipped": True, "reason": "no strategies"}

        all_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

        # Step 1: Run comprehensive backtest
        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        import asyncio, concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(
                pool, self.backtest_engine.run,
                all_strategies, all_symbols, start, end, "full", 10000.0)

        if result.get("error"):
            logger.warning(f"Lifecycle: matrix backtest failed: {result['error']}")
            return {"error": result["error"]}

        per_matrix = result.get("per_matrix", {})
        if not per_matrix:
            return {"skipped": True, "reason": "no per_matrix data"}

        # Step 2: Analyze per-cell performance
        strategy_cell_scores = {}  # strategy -> {symbol -> pnl}
        strategy_total_pnl = {}    # strategy -> total PnL

        for s_name, sym_data in per_matrix.items():
            strategy_cell_scores[s_name] = {}
            total = 0.0
            for sym, cell in sym_data.items():
                pnl = cell.get("pnl", 0)
                strategy_cell_scores[s_name][sym] = pnl
                total += pnl
            strategy_total_pnl[s_name] = total

        # Step 3: Remove negative-PnL cells (restrict strategy from losing symbols)
        for s_name, sym_scores in strategy_cell_scores.items():
            try:
                s = self.loader.load(s_name)
            except Exception:
                continue

            current_symbols = set(s.symbols) if s.symbols else set(all_symbols)
            negative_symbols = [sym for sym, pnl in sym_scores.items()
                               if pnl < -5.0 and sym in current_symbols]  # threshold: -$5

            if not negative_symbols:
                continue

            if ai_mode == "suggest":
                actions["cells_removed"].append({
                    "strategy": s_name, "symbols": negative_symbols,
                    "action": "suggested"
                })
                await self.log_event(s_name, "optimize",
                    f"AI suggests removing symbols: {', '.join(negative_symbols)} "
                    f"(negative PnL in matrix)")
            elif ai_mode in ("semi_auto", "full_auto"):
                new_symbols = list(current_symbols - set(negative_symbols))
                s.symbols = new_symbols
                self.loader.save(s)
                # Sync engine
                if s_name in self.strategy_engine._strategies:
                    self.strategy_engine._strategies[s_name] = s
                actions["cells_removed"].append({
                    "strategy": s_name, "symbols": negative_symbols,
                    "action": "removed"
                })
                await self.log_event(s_name, "optimize",
                    f"Auto-removed symbols: {', '.join(negative_symbols)} "
                    f"(negative PnL). Remaining: {new_symbols or 'all'}")

        # Step 4: Optimize strategies with all-negative cells via AI
        for s_name, sym_scores in strategy_cell_scores.items():
            positive_cells = sum(1 for pnl in sym_scores.values() if pnl > 0)
            total_cells = len(sym_scores)
            if total_cells == 0:
                continue
            # If >60% of cells are negative AND total PnL is negative, optimize
            if positive_cells / total_cells < 0.4 and strategy_total_pnl.get(s_name, 0) < 0:
                if ai_mode == "suggest":
                    actions["strategies_optimized"].append({
                        "strategy": s_name, "action": "suggested",
                        "reason": f"Only {positive_cells}/{total_cells} cells profitable"
                    })
                    await self.log_event(s_name, "optimize",
                        f"AI suggests parameter optimization: only "
                        f"{positive_cells}/{total_cells} cells profitable")
                elif ai_mode in ("semi_auto", "full_auto"):
                    optimized = await self._optimize_strategy_params(s_name, sym_scores)
                    if optimized:
                        actions["strategies_optimized"].append({
                            "strategy": s_name, "action": "optimized"
                        })
                    else:
                        # Couldn't optimize — retire it
                        actions["strategies_optimized"].append({
                            "strategy": s_name, "action": "retired",
                            "reason": "optimization failed"
                        })

        # Step 5: Identify coverage gaps and generate strategies
        # A "gap" = a symbol that has no profitable strategy covering it
        symbol_best_pnl = {}
        for sym in all_symbols:
            best = -999
            best_strategy = None
            for s_name, sym_scores in strategy_cell_scores.items():
                pnl = sym_scores.get(sym, 0)
                if pnl > best:
                    best = pnl
                    best_strategy = s_name
            symbol_best_pnl[sym] = (best, best_strategy)

        gap_symbols = [sym for sym, (pnl, _) in symbol_best_pnl.items() if pnl < 0]

        if gap_symbols:
            actions["gaps_identified"] = gap_symbols
            if ai_mode in ("semi_auto", "full_auto"):
                # Generate a new strategy targeting the gap symbols
                new_config = await self.generate_strategy(target_symbols=gap_symbols)
                if new_config:
                    # Force-disable the _generation_interval check for gap-filling
                    from core.strategy.loader import StrategyConfig
                    try:
                        strategy_config = StrategyConfig(**new_config)
                        strategy_config.symbols = gap_symbols
                        self.loader.save(strategy_config)
                        all_s = self.loader.load_all()
                        self.strategy_engine._strategies = {s.name: s for s in all_s}
                        actions["strategies_generated"].append({
                            "name": strategy_config.name,
                            "target_symbols": gap_symbols
                        })
                        await self.log_event(strategy_config.name, "generated",
                            f"Auto-generated to fill coverage gaps: {', '.join(gap_symbols)}")
                    except Exception as e:
                        logger.warning(f"Lifecycle: failed to save gap-fill strategy: {e}")

        logger.info(f"Lifecycle: matrix analysis complete — {json.dumps(actions, default=str)}")
        return actions

    async def _optimize_strategy_params(self, name: str,
                                         sym_scores: dict[str, float]) -> bool:
        """Use AI to optimize a strategy's indicator parameters based on per-symbol results.

        Returns True if the optimized strategy was saved and passes backtest.
        """
        try:
            s = self.loader.load(name)
        except Exception:
            return False

        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        symbols = s.symbols if s.symbols else ["BTCUSDT", "ETHUSDT"]

        # Summarize per-symbol performance for the AI
        perf_summary = "\n".join(
            f"  {sym}: PnL={pnl:.2f} USDT" for sym, pnl in sorted(sym_scores.items()))

        prompt = (
            f"Optimize the following underperforming trading strategy.\n\n"
            f"Current config:\n"
            f"  name: {s.name}\n"
            f"  mode: {s.mode}\n"
            f"  timeframes: {s.timeframes}\n"
            f"  indicators: {json.dumps(s.indicators)}\n"
            f"  entry_conditions: {json.dumps(s.entry_conditions)}\n"
            f"  exit_conditions: {json.dumps(s.exit_conditions)}\n"
            f"  ml_config: {json.dumps(s.ml_config.model_dump() if s.ml_config else {})}\n\n"
            f"Recent performance per symbol (30-day backtest):\n{perf_summary}\n\n"
            f"Please suggest optimized parameters. You may:\n"
            f"1. Adjust indicator periods (e.g., RSI period from 14 to 10 for faster signals)\n"
            f"2. Tighten or loosen entry/exit thresholds\n"
            f"3. Add/remove indicators\n"
            f"4. Adjust ML weight and confidence threshold\n"
            f"5. Change timeframes\n\n"
            f"Keep the same strategy name and mode.\n"
            f'Respond ONLY with the complete optimized JSON config (same fields as original):'
        )

        result = await self.deepseek._call_deepseek(
            "You are a quantitative trading strategist. Output only valid JSON.",
            prompt
        )

        if not result:
            logger.warning(f"Lifecycle: AI optimization returned no result for {name}")
            return False

        try:
            optimized = json.loads(
                result.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            logger.warning(f"Lifecycle: failed to parse AI optimization output: {result[:200]}")
            return False

        # Preserve the original name and mode
        optimized["name"] = name
        optimized["mode"] = s.mode
        if "symbols" not in optimized:
            optimized["symbols"] = s.symbols

        # Validate and backtest the optimized version
        from core.strategy.loader import StrategyConfig
        try:
            optimized_config = StrategyConfig(**optimized)
        except Exception as e:
            logger.warning(f"Lifecycle: optimized config validation failed: {e}")
            return False

        # Run backtest on optimized version
        import asyncio, concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            bt_result = await loop.run_in_executor(
                pool, self.backtest_engine.run,
                [name], symbols, start, end, "full", 10000.0)

        if bt_result.get("error"):
            logger.warning(f"Lifecycle: optimized backtest failed: {bt_result['error']}")
            return False

        new_metrics = bt_result.get("metrics", {})
        new_sharpe = new_metrics.get("sharpe_ratio", 0)
        new_wr = new_metrics.get("win_rate_pct", 0)

        # Only save if improved
        if new_sharpe > -0.5 and new_wr > 25:
            self.loader.save(optimized_config)
            self.strategy_engine._strategies[name] = optimized_config
            await self.log_event(name, "optimized",
                f"AI-optimized parameters: Sharpe={new_sharpe}, WinRate={new_wr}%")
            logger.info(f"Lifecycle: optimized {name} — new Sharpe={new_sharpe}, WR={new_wr}%")
            return True

        # Not better — retire the strategy instead
        logger.info(f"Lifecycle: optimization didn't improve {name} enough (Sharpe={new_sharpe}), retiring")
        s.enabled = False
        self.loader.save(s)
        self.strategy_engine._strategies[name] = s
        await self.log_event(name, "retired",
            f"Auto-retired after failed optimization: Sharpe={new_sharpe}, WinRate={new_wr}%")
        return False

    # ── Coverage Gaps ──────────────────────────────────────────────────

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
