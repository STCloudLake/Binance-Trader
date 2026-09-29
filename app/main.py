#!/usr/bin/env python3
"""Binance Trader — Automated trading system with ML prediction, news analysis, and AI decision-making."""

import asyncio
import argparse
import os
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
import uvicorn

from app.config import Config, ConfigError
from app.event_bus import EventBus, Event, EventType
from db.database import init_database, load_sim_balance, save_sim_balance, atomic_adjust_balance, DEFAULT_BALANCE
from core.market_data.provider import (
    DEFAULT_INTERVALS, DEFAULT_ML_INTERVAL, MarketDataProvider, poll_seconds)
from core.market_data.universe import DEFAULT_WATCHLIST, Universe
from core.strategy.engine import StrategyEngine
from core.strategy.loader import StrategyLoader
from core.ml.predictor import MLPredictor
from core.news.analyzer import NewsAnalyzer
from core.risk.manager import RiskManager
from core.risk.position_guard import PositionGuard
from core.executor.executor import OrderExecutor
from core.executor.pending_orders import start_matcher_task, stop_matcher
from core.ai.deepseek_ctl import DeepSeekController
from alerts.manager import AlertManager
from web.server import create_app


def load_config_or_exit(mode: str = "sim") -> Config:
    """Startup boundary for configuration loading.

    A malformed ``config/config.yaml`` / ``config/secrets.yaml`` used to escape as
    a raw ``yaml.parser.ParserError`` traceback (exit 2).  ``Config._load_yaml``
    now raises :class:`ConfigError` with ``<file> is not valid YAML (line N):
    <reason>``; here that becomes one logged line and a clean exit 1.
    """
    try:
        return Config.load(mode)
    except ConfigError as e:
        logger.error(str(e))
        raise SystemExit(1) from None


def warn_if_alert_rules_missing(alert_manager) -> bool:
    """WARN when ``config/alert_rules.json`` is absent (default rules apply).

    ``AlertManager`` silently substitutes ``alerts.rules.DEFAULT_RULES``, so the
    operator had no way to tell "my rule file was read" from "it was never
    found".  Returns False (and logs) when the file the manager will read is
    missing; True otherwise.  The path is taken from the manager itself so the
    warning can never name a different file than the one actually loaded.
    """
    rules_path = Path(getattr(alert_manager, "_rules_path", "") or "")
    if rules_path and not rules_path.exists():
        logger.warning(f"{rules_path} not found — using built-in default alert rules")
        return False
    return True


def port_in_use(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    """True when something already accepts connections on ``host:port``.

    Deliberately a *connect* probe, not a bind probe: binding is refused while
    the previous instance's just-closed connections sit in TIME_WAIT (up to 4
    minutes on Windows), so a bind probe would abort a perfectly valid restart.
    A refused connection proves nothing is listening; a timeout is inconclusive
    (e.g. a firewalled loopback) and is treated as free so the check can never
    block a legitimate startup.
    """
    import socket
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


async def apply_engine_exit(order_executor, risk_manager, db_path: str, data: dict,
                            reduce_pct: float, default_reason: str) -> dict:
    """Close/reduce a position for an engine event and credit the cash back.

    Module-level (not a closure in ``main()``) so the money path is directly
    testable — and so the event's real ``reason`` is threaded into the exit row:
    it used to be dropped, which recorded every engine exit as
    ``exit_reason='manual'`` and made "why did this position close?" unanswerable
    from the DB.
    """
    result = await order_executor.close_position(
        data["symbol"], reduce_pct, data.get("price", 0),
        reason=data.get("reason") or default_reason)
    if result.get("ok"):
        trade_pnl = result.get("pnl", 0)
        new_balance = await atomic_adjust_balance(
            result.get("invested_returned", 0) + trade_pnl, db_path)
        risk_manager.update_balance(new_balance)
        logger.info(f"Engine exit {data['symbol']} ({data.get('reason', default_reason)}): "
                    f"PnL={trade_pnl:.2f} | Balance={new_balance:.0f}")
    return result


async def persist_tightened_stop(order_executor, symbol: str, pos: dict,
                                 price: float) -> float:
    """Tighten ``pos``'s stop to just below/above ``price`` and PERSIST it.

    The circuit breaker's ``tighten_stops`` action used to mutate
    ``pos["stop_loss"]`` in memory only, so the protective stop silently reverted
    to the pre-breaker value (the executor's 2% default after a restart) exactly
    when the market had gone against the book.
    """
    side = pos.get("side", "long")
    new_sl = round(price * 0.98 if side == "long" else price * 1.02, 2)
    pos["stop_loss"] = new_sl
    persist = getattr(order_executor, "update_stop_loss", None)
    if persist is not None:
        await persist(symbol, new_sl)
    return new_sl


async def starting_balance_for_unpersisted_wallet(db_path: str,
                                                  open_notional: float) -> float:
    """The balance implied by the ledger when no ``sim_balance`` row exists.

    ``10000 − Σ(open qty×entry_price) + Σ(close pnl)`` — the identity itself.  The
    old startup repair used ``10000 − invested``, silently dropping the realised
    PnL term and leaving the identity broken by exactly that sum for ever.
    """
    realised = 0.0
    try:
        import aiosqlite
        db = await aiosqlite.connect(db_path)
        try:
            cursor = await db.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status='closed'")
            realised = float((await cursor.fetchone())[0] or 0.0)
        finally:
            await db.close()
    except Exception as e:  # pragma: no cover - defensive (a pre-trades DB)
        logger.warning(f"Could not read realised PnL for the balance repair: {e}")
    return DEFAULT_BALANCE - open_notional + realised


async def main():
    parser = argparse.ArgumentParser(description="Binance Trader")
    parser.add_argument("--mode", choices=["sim", "live", "backtest"], default="sim",
                        help="Running mode (default: sim)")
    parser.add_argument("--port", type=int, default=None, help="Web UI port")
    # Redirect every persistent artefact. Used by smoke tests / staging runs so a
    # throwaway instance never touches the live data/binance_trader.db, the WAL
    # sidecars or the shipped config/*.yaml.
    parser.add_argument("--db", default=None,
                        help="SQLite database path (default: data/binance_trader.db)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory for runtime data (default: <project>/data)")
    parser.add_argument("--config-dir", default=None,
                        help="Directory persisted settings/secrets are written to")
    args = parser.parse_args()

    logger.info(f"Starting Binance Trader in {args.mode} mode")

    # 1. Load config (a broken YAML logs one actionable line and exits 1)
    config = load_config_or_exit(args.mode)
    if args.port:
        config.web_port = args.port
    if args.data_dir:
        config.data_dir = str(Path(args.data_dir).resolve())
    if args.db:
        config.db_path = str(Path(args.db).resolve())
    elif args.data_dir:
        config.db_path = str(Path(config.data_dir) / "binance_trader.db")
    if args.config_dir:
        config.config_dir = str(Path(args.config_dir).resolve())
        Path(config.config_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Config loaded. DB: {config.db_path}")

    # Pre-flight: fail fast (before the ~70-90 s warm-up) when the port is taken.
    # A connect probe — see ``port_in_use`` — so a TIME_WAIT socket from the
    # previous instance cannot produce a false "port busy".
    if port_in_use(config.web_port):
        logger.error(f"Port {config.web_port} is already in use on 127.0.0.1 — "
                     f"stop the other instance or start with --port <free port>")
        raise SystemExit(1)

    # 2. Init database
    await init_database(config.db_path)
    logger.info("Database initialized")

    # Initialize auth and create default admin if no users exist
    from core.auth.auth import AuthManager
    auth_cfg = config._get("auth", {}) if isinstance(config._get("auth", {}), dict) else {}
    jwt_secret = auth_cfg.get("jwt_secret", "")
    if not jwt_secret:
        # Try environment variable first for persistence across restarts
        jwt_secret = os.environ.get("JWT_SECRET", "")
    if not jwt_secret:
        # Generate and persist to secrets.yaml so tokens survive restarts
        import secrets as _secrets
        jwt_secret = _secrets.token_hex(32)
        secrets_path = Path(getattr(config, "config_dir", PROJECT_ROOT / "config")) / "secrets.yaml"
        try:
            existing = {}
            if secrets_path.exists():
                existing = yaml.safe_load(secrets_path.read_text(encoding="utf-8")) or {}
            if "auth" not in existing:
                existing["auth"] = {}
            existing["auth"]["jwt_secret"] = jwt_secret
            secrets_path.write_text(yaml.dump(existing, default_flow_style=False), encoding="utf-8")
            # Tighten permissions on POSIX (no-op on Windows, where mode bits do
            # not express "readable by others").
            try:
                if os.name == "posix":
                    os.chmod(secrets_path, 0o600)
            except OSError:
                pass
            logger.info("JWT secret generated and persisted to secrets.yaml "
                        "(set JWT_SECRET in the environment for production instead)")
        except Exception as e:
            logger.warning(f"Could not persist JWT secret to secrets.yaml: {e}")
        import hashlib
        logger.info(f"JWT secret fingerprint: {hashlib.sha256(jwt_secret.encode()).hexdigest()[:16]}")
    auth_manager = AuthManager(config.db_path, jwt_secret, auth_cfg.get("session_hours", 24))
    if await auth_manager.count_users() == 0:
        admin_pass = AuthManager.generate_random_password()
        await auth_manager.create_user("admin", admin_pass, "admin", "Administrator")
        logger.warning(f"=== DEFAULT ADMIN CREATED: username=admin (password not logged) ===")
        # Print to stderr only so it's visible in terminal but not in log files
        import sys as _sys
        _sys.stderr.write(f"\n{'='*60}\nDEFAULT ADMIN: admin / {admin_pass}\n{'='*60}\n\n")

    # 3. Create event bus
    event_bus = EventBus()
    await event_bus.start()
    logger.info("EventBus started")

    # 4. Initialize components
    market_data = MarketDataProvider(config, event_bus)
    strategy_engine = StrategyEngine(config, event_bus, market_data)
    ml_predictor = MLPredictor(config, event_bus, market_data)
    news_analyzer = NewsAnalyzer(config, event_bus, market_data)
    risk_manager = RiskManager(config, event_bus)
    order_executor = OrderExecutor(config, event_bus)
    position_guard = PositionGuard(config, event_bus)
    deepseek_ctl = DeepSeekController(config, event_bus)
    alert_manager = AlertManager(config, event_bus)
    warn_if_alert_rules_missing(alert_manager)

    deepseek_ctl.wire(market_data, order_executor, risk_manager, strategy_engine)
    strategy_engine.wire_executor(order_executor)
    order_executor.wire_risk_manager(risk_manager)
    risk_manager.wire_executor(order_executor)  # accurate position lookup in update_balance
    position_guard.wire(order_executor, market_data, risk_manager)

    # 4.5 Init backtest engine and strategy lifecycle manager
    from core.backtest.engine import BacktestEngine
    from core.ai.strategy_lifecycle import StrategyLifecycleManager

    backtest_engine = BacktestEngine(config, strategy_engine, risk_manager, order_executor)

    lifecycle_manager = StrategyLifecycleManager(
        config=config,
        deepseek_ctl=deepseek_ctl,
        backtest_engine=backtest_engine,
        strategy_loader=strategy_engine.loader,
        strategy_engine=strategy_engine,
        alert_manager=alert_manager,
        db_path=config.db_path,
    )

    deepseek_ctl.wire_lifecycle(lifecycle_manager)

    # 5. Set DeepSeek key for news analyzer
    if config.deepseek_api_key:
        await news_analyzer.set_deepseek(config.deepseek_api_key, config.ai_base_url)

    # 5.5 Wire auto-close and auto-reduce handlers BEFORE starting components
    # that generate kline events, so no exit/reduce events are ever lost.
    async def _on_position_exit(event: Event):
        await apply_engine_exit(order_executor, risk_manager, config.db_path,
                                event.data, 100, "engine_exit")

    async def _on_position_reduce(event: Event):
        await apply_engine_exit(order_executor, risk_manager, config.db_path,
                                event.data, event.data.get("reduce_pct", 50),
                                "engine_reduce")

    event_bus.subscribe(EventType.POSITION_EXIT, _on_position_exit)
    event_bus.subscribe(EventType.POSITION_REDUCE, _on_position_reduce)

    async def _execute_breaker_action(action: str, reason: str):
        """Execute the configured circuit breaker response action."""
        if action == "block_only":
            logger.info(f"Breaker action: block_only — {reason}")
            return

        open_positions = order_executor.get_open_positions()
        if not open_positions:
            logger.info(f"Breaker tripped but no open positions — {reason}")
            return

        if action == "close_all":
            logger.warning(f"Breaker action: close_all — closing {len(open_positions)} positions")
            for sym in list(open_positions.keys()):
                price = market_data.get_current_price(sym) or open_positions[sym].get("current_price", open_positions[sym]["entry_price"])
                result = await order_executor.close_position(sym, 100, price)
                if result.get("ok"):
                    invested_returned = result.get("invested_returned", 0)
                    trade_pnl = result.get("pnl", 0)
                    new_balance = await atomic_adjust_balance(invested_returned + trade_pnl, config.db_path)
                    risk_manager.update_balance(new_balance)
                    logger.info(f"Breaker close_all: {sym} PnL={trade_pnl:.2f} Balance={new_balance:.0f}")

        elif action == "close_worst":
            worst_sym = None
            worst_pnl = float("inf")
            for sym, pos in open_positions.items():
                entry = pos["entry_price"]
                qty = pos["quantity"]
                side = pos["side"]
                cur_price = market_data.get_current_price(sym) or pos.get("current_price", entry)
                unrealized = (cur_price - entry) * qty if side == "long" else (entry - cur_price) * qty
                if unrealized < worst_pnl:
                    worst_pnl = unrealized
                    worst_sym = sym

            if worst_sym:
                logger.warning(f"Breaker action: close_worst — closing {worst_sym} (uPnL={worst_pnl:.2f})")
                price = market_data.get_current_price(worst_sym) or open_positions[worst_sym]["entry_price"]
                result = await order_executor.close_position(worst_sym, 100, price)
                if result.get("ok"):
                    invested_returned = result.get("invested_returned", 0)
                    trade_pnl = result.get("pnl", 0)
                    new_balance = await atomic_adjust_balance(invested_returned + trade_pnl, config.db_path)
                    risk_manager.update_balance(new_balance)
                    logger.info(f"Breaker close_worst: {worst_sym} PnL={trade_pnl:.2f} Balance={new_balance:.0f}")

        elif action == "tighten_stops":
            logger.warning(f"Breaker action: tighten_stops — adjusting stops on {len(open_positions)} positions")
            for sym, pos in open_positions.items():
                price = market_data.get_current_price(sym) or pos.get("current_price", pos["entry_price"])
                new_sl = await persist_tightened_stop(order_executor, sym, pos, price)
                logger.info(f"Breaker tighten_stops: {sym} {pos['side']} SL→{new_sl:.2f} (persisted)")

    async def _on_risk_breach(event: Event):
        if event.data.get("event_type") != "circuit_breaker_trip":
            return
        data = event.data
        reason = data.get("detail", "Unknown")
        logger.error(f"Circuit breaker TRIPPED: {reason}")

        if config.ai_mode == "full_auto" and deepseek_ctl.client:
            action = await deepseek_ctl.decide_breaker_action(data)
        else:
            action = config.hard_limits.circuit_breaker_action

        await _execute_breaker_action(action, reason)

        if config.ai_mode == "full_auto" and deepseek_ctl.client:
            asyncio.create_task(deepseek_ctl._breaker_recovery_loop())

    event_bus.subscribe(EventType.RISK_BREACH, _on_risk_breach)

    # Schedule daily/weekly circuit breaker resets
    async def _circuit_breaker_reset_loop():
        import time as _time
        while True:
            now = _time.localtime()
            # Sleep until next hour boundary + 2 minutes
            seconds_to_next_hour = (60 - now.tm_min) * 60 - now.tm_sec + 120
            await asyncio.sleep(max(60, seconds_to_next_hour))
            now = _time.localtime()
            # Daily reset near 00:xx
            if now.tm_hour == 0 and now.tm_min < 10:
                risk_manager.breaker.reset_daily()
                logger.info("Circuit breaker: daily reset")
            # Weekly reset on Monday near 00:xx
            if now.tm_wday == 0 and now.tm_hour == 0 and now.tm_min < 10:
                risk_manager.breaker.reset_weekly()
                logger.info("Circuit breaker: weekly reset")

    # Long-running background loops are tracked so shutdown can cancel them
    # (previously fire-and-forget tasks outlived the shutdown sequence).
    background_tasks: list[asyncio.Task] = []
    background_tasks.append(asyncio.create_task(_circuit_breaker_reset_loop(),
                                                name="circuit_breaker_reset_loop"))

    # 6. Start components (handlers are already subscribed above)
    # The watched symbol set comes from the persisted watchlist
    # (`system_config.watchlist_symbols`, written by POST /api/market/watchlist),
    # with DEFAULT_WATCHLIST (core/market_data/universe.py) as the fallback.
    default_symbols = list(DEFAULT_WATCHLIST)
    default_intervals = list(DEFAULT_INTERVALS)

    # Share ONE coin universe (tick/step sizes, min notional, base assets) with
    # the executor and the web layer: it is preloaded from disk here, so the
    # trading path can round to real LOT_SIZE rules without a network call.
    universe = Universe(config)
    universe.preload_from_disk()
    order_executor.wire_universe(universe)
    deepseek_ctl.wire_universe(universe)

    # `symbols=None` → MarketDataProvider reads the persisted watchlist itself.
    await market_data.start(None, default_intervals)
    watchlist = market_data.watched_symbols or default_symbols
    logger.info(f"Watchlist in effect: {', '.join(watchlist)}")
    await strategy_engine.start()
    await ml_predictor.start()
    # Core vs satellite is a *position-sizing* classification
    # (`core_position.max_symbols`), not "the first three list entries".
    core_max = max(0, int(getattr(config, "core_max_symbols", 0) or 0))
    await news_analyzer.start(watchlist[:core_max], watchlist[core_max:])
    await risk_manager.start()
    await order_executor.start()
    await position_guard.start()

    # Sync balance BEFORE AI starts — AI reads balance for risk adjustment
    bal = await load_sim_balance(config.db_path)
    risk_manager.update_balance(bal)
    logger.info(f"Risk manager balance set: {bal:.0f}")

    await deepseek_ctl.start()
    await alert_manager.start()

    logger.info(f"All components started. {len(watchlist)} symbols monitored")

    # Trigger initial strategy evaluation (seed cache, no trades)
    await strategy_engine.evaluate_all_now()
    logger.info("Initial strategy evaluation complete")

    # REST polling fallback: publishes MARKET_KLINE events when WebSocket hasn't
    # delivered a recent kline for a symbol/interval, ensuring continuous evaluation.
    async def _rest_polling_loop():
        import time as _time
        await asyncio.sleep(30)
        while True:
            try:
                now = _time.time()
                last_ws = getattr(market_data, '_last_kline_time', {})
                polled: list[str] = []
                never_seen: list[str] = []
                for symbol in watchlist:
                    for interval in default_intervals:
                        key = f"{symbol}_{interval}"
                        last_ts = last_ws.get(key, 0)
                        # Poll if WebSocket hasn't delivered in 2x the expected interval
                        interval_secs = poll_seconds(interval)
                        if last_ts and (now - last_ts) < interval_secs:
                            continue
                        polled.append(key)
                        if not last_ts:
                            # `now - 0` used to be logged as "1790681971s ago"
                            never_seen.append(key)
                        df = await market_data.get_historical(symbol, interval, limit=52)
                        if df is not None and len(df) >= 51:
                            # Use second-to-last candle — guaranteed to be closed.
                            # The last candle may still be forming (incomplete).
                            candle = {
                                "close_time": int(df.index[-2].timestamp() * 1000),
                                "open": float(df["open"].iloc[-2]),
                                "high": float(df["high"].iloc[-2]),
                                "low": float(df["low"].iloc[-2]),
                                "close": float(df["close"].iloc[-2]),
                                "volume": float(df["volume"].iloc[-2]),
                            }
                            await event_bus.publish(Event(EventType.MARKET_KLINE, {
                                "symbol": symbol, "interval": interval, "candle": candle,
                            }))
                # One summary line per cycle instead of 25 (the old per-key logging
                # produced ~3000 lines/hour and printed nonsense for "never seen").
                if polled:
                    logger.info(
                        f"REST poll: {len(polled)} symbol/interval pairs"
                        + (f" (WebSocket has not delivered yet for {len(never_seen)})" if never_seen else "")
                    )
                await asyncio.sleep(30)
            except Exception as e:
                logger.warning(f"REST polling loop error: {e}")
                await asyncio.sleep(60)

    background_tasks.append(asyncio.create_task(_rest_polling_loop(), name="rest_polling_loop"))

    # Train ML models for each symbol on the registry's ML interval (1h)
    for symbol in watchlist:
        try:
            result = await ml_predictor.train_model(symbol, "default", DEFAULT_ML_INTERVAL)
            if "error" in result:
                logger.warning(f"ML training skipped for {symbol}: {result['error']}")
            else:
                logger.info(f"ML model trained: {symbol} — accuracy={result.get('accuracy', 'N/A')}, f1={result.get('f1', 'N/A')}")
        except Exception as e:
            logger.warning(f"ML training failed for {symbol}: {e}")
    logger.info("ML training round complete")

    # Publish ML predictions and directly seed strategy engine cache
    from core.strategy.indicators import compute_all
    from core.ml.features import REQUIRED_INDICATORS
    for symbol in watchlist:
        try:
            df = await market_data.get_historical(symbol, DEFAULT_ML_INTERVAL, limit=200)
            if df is not None and len(df) >= 50:
                # Use REQUIRED_INDICATORS to match training feature set (prevents 29≠30 mismatch)
                df = compute_all(df, REQUIRED_INDICATORS)
                # Use the predictor's full feature list (matches training)
                confidence = await ml_predictor.predict(symbol, df)
                # Directly seed engine cache (bypasses async event queue)
                strategy_engine._ml_confidence[symbol] = confidence
                await event_bus.publish(Event(EventType.ML_PREDICTION, {
                    "symbol": symbol, "interval": DEFAULT_ML_INTERVAL,
                    "confidence": confidence,
                }))
        except Exception as e:
            # Previously silent: a failure here means the strategy engine keeps
            # running with a neutral (0.5) ML confidence for the whole session.
            logger.warning(f"Initial ML prediction failed for {symbol}: {e}")
    logger.info("Initial ML predictions published and seeded")

    # Re-evaluate strategies with real ML confidence values
    await strategy_engine.evaluate_all_now()
    logger.info("Post-ML strategy evaluation complete")

    # 7. Start Web UI
    web_app = create_app(config, event_bus, auth_manager)
    web_app.state.strategy_loader = strategy_engine.loader
    web_app.state.strategy_engine = strategy_engine
    web_app.state.config = config
    web_app.state.executor = order_executor
    web_app.state.ai_controller = deepseek_ctl
    web_app.state.risk_manager = risk_manager
    web_app.state.auth_manager = auth_manager
    web_app.state.backtest_engine = backtest_engine
    web_app.state.lifecycle_manager = lifecycle_manager
    web_app.state.alert_manager = alert_manager
    # The same warm universe the executor uses — `web/routes/market.py` reuses
    # `app.state.universe` instead of building a second cache.
    web_app.state.universe = universe
    web_app.state.get_price = market_data.get_current_price
    web_app.state.balance = await load_sim_balance(config.db_path)

    # Adjust balance for legacy positions that were opened before balance
    # persistence.  A database with no `sim_balance` row has no recorded cash, so
    # the only honest starting point is the ledger identity (see the helper): the
    # old `10000 − invested` repair ignored realised PnL and left the identity
    # broken by that sum.
    open_positions = order_executor.get_open_positions()
    total_invested = sum(
        p.get("amount_usdt", p.get("quantity", 0) * p.get("entry_price", 0))
        for p in open_positions.values())
    if web_app.state.balance == DEFAULT_BALANCE and total_invested > 0 and total_invested < web_app.state.balance:
        web_app.state.balance = await starting_balance_for_unpersisted_wallet(
            config.db_path, total_invested)
        await save_sim_balance(web_app.state.balance, config.db_path)
        logger.info(f"Adjusted balance for {len(open_positions)} legacy positions: "
                    f"→ {web_app.state.balance:.2f} (ledger identity)")

    # Sync balance and positions to risk manager so position sizing works
    risk_manager.update_balance(web_app.state.balance)
    risk_manager.sync_positions(open_positions)
    logger.info(f"Risk manager synced: balance={web_app.state.balance:.0f}, positions={len(open_positions)}")

    # Start the pending limit-order matcher (restores open orders from the
    # `pending_orders` table and fills them every ~5s).  It only starts after the
    # web app state exists, because it resolves prices through app.state.
    start_matcher_task(web_app, config, event_bus)
    logger.info("Limit-order matcher started (5s interval)")

    config_uvicorn = uvicorn.Config(
        web_app, host="127.0.0.1", port=config.web_port, log_level="info"
    )
    server = uvicorn.Server(config_uvicorn)

    logger.info(f"Web UI starting at http://127.0.0.1:{config.web_port}")

    try:
        await server.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down...")
    finally:
        # Cancel background loops first so they cannot touch components that are
        # being stopped below.
        for _task in background_tasks:
            _task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        await stop_matcher(web_app)
        await alert_manager.stop()
        await position_guard.stop()
        await deepseek_ctl.stop()
        await order_executor.stop()
        await risk_manager.stop()
        await news_analyzer.stop()
        await ml_predictor.stop()
        await strategy_engine.stop()
        await market_data.stop()
        await event_bus.shutdown()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
