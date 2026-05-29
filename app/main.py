#!/usr/bin/env python3
"""Binance Trader — Automated trading system with ML prediction, news analysis, and AI decision-making."""

import asyncio
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
import uvicorn

from app.config import Config
from app.event_bus import EventBus, Event, EventType
from db.database import init_database, load_sim_balance, save_sim_balance, atomic_adjust_balance, DEFAULT_BALANCE
from core.market_data.provider import MarketDataProvider
from core.strategy.engine import StrategyEngine
from core.strategy.loader import StrategyLoader
from core.ml.predictor import MLPredictor
from core.news.analyzer import NewsAnalyzer
from core.risk.manager import RiskManager
from core.risk.position_guard import PositionGuard
from core.executor.executor import OrderExecutor
from core.ai.deepseek_ctl import DeepSeekController
from alerts.manager import AlertManager
from web.server import create_app


async def main():
    parser = argparse.ArgumentParser(description="Binance Trader")
    parser.add_argument("--mode", choices=["sim", "live", "backtest"], default="sim",
                        help="Running mode (default: sim)")
    parser.add_argument("--port", type=int, default=None, help="Web UI port")
    args = parser.parse_args()

    logger.info(f"Starting Binance Trader in {args.mode} mode")

    # 1. Load config
    config = Config.load(args.mode)
    if args.port:
        config.web_port = args.port

    logger.info(f"Config loaded. DB: {config.db_path}")

    # 2. Init database
    await init_database(config.db_path)
    logger.info("Database initialized")

    # Initialize auth and create default admin if no users exist
    from core.auth.auth import AuthManager
    auth_cfg = config._get("auth", {}) if isinstance(config._get("auth", {}), dict) else {}
    jwt_secret = auth_cfg.get("jwt_secret", "")
    if not jwt_secret:
        import secrets
        jwt_secret = secrets.token_hex(32)
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

    deepseek_ctl.wire(market_data, order_executor, risk_manager, strategy_engine)
    strategy_engine.wire_executor(order_executor)
    order_executor.wire_risk_manager(risk_manager)
    risk_manager.wire_executor(order_executor)  # accurate position lookup in update_balance
    position_guard.wire(order_executor, market_data, risk_manager)

    # 5. Set DeepSeek key for news analyzer
    if config.deepseek_api_key:
        await news_analyzer.set_deepseek(config.deepseek_api_key, config.ai_base_url)

    # 5.5 Wire auto-close and auto-reduce handlers BEFORE starting components
    # that generate kline events, so no exit/reduce events are ever lost.
    async def _on_position_exit(event: Event):
        data = event.data
        result = await order_executor.close_position(data["symbol"], 100, data.get("price", 0))
        if result.get("ok"):
            invested_returned = result.get("invested_returned", 0)
            trade_pnl = result.get("pnl", 0)
            new_balance = await atomic_adjust_balance(invested_returned + trade_pnl, config.db_path)
            risk_manager.update_balance(new_balance)
            logger.info(f"Auto-close {data['symbol']}: PnL={trade_pnl:.2f} | Balance={new_balance:.0f} | {data.get('reason','')}")

    async def _on_position_reduce(event: Event):
        data = event.data
        result = await order_executor.close_position(
            data["symbol"], data.get("reduce_pct", 50), data.get("price", 0))
        if result.get("ok"):
            invested_returned = result.get("invested_returned", 0)
            trade_pnl = result.get("pnl", 0)
            new_balance = await atomic_adjust_balance(invested_returned + trade_pnl, config.db_path)
            risk_manager.update_balance(new_balance)
            logger.info(f"Auto-reduce {data['symbol']} {data.get('reduce_pct',50)}%: PnL={trade_pnl:.2f} | Balance={new_balance:.0f} | {data.get('reason','')}")

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
                side = pos["side"]
                new_sl = price * 0.98 if side == "long" else price * 1.02
                pos["stop_loss"] = round(new_sl, 2)
                logger.info(f"Breaker tighten_stops: {sym} {side} SL→{new_sl:.2f}")

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

    asyncio.create_task(_circuit_breaker_reset_loop())

    # 6. Start components (handlers are already subscribed above)
    default_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    default_intervals = ["1m", "5m", "15m", "1h", "4h"]

    await market_data.start(default_symbols, default_intervals)
    await strategy_engine.start()
    await ml_predictor.start()
    await news_analyzer.start(default_symbols[:3], default_symbols[3:])
    await risk_manager.start()
    await order_executor.start()
    await position_guard.start()

    # Sync balance BEFORE AI starts — AI reads balance for risk adjustment
    bal = await load_sim_balance(config.db_path)
    risk_manager.update_balance(bal)
    logger.info(f"Risk manager balance set: {bal:.0f}")

    await deepseek_ctl.start()
    await alert_manager.start()

    logger.info(f"All components started. {len(default_symbols)} symbols monitored")

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
                for symbol in default_symbols:
                    for interval in default_intervals:
                        key = f"{symbol}_{interval}"
                        last_ts = last_ws.get(key, 0)
                        # Poll if WebSocket hasn't delivered in 2x the expected interval
                        interval_secs = {"1m": 120, "5m": 300, "15m": 900, "1h": 3600, "4h": 7200}.get(interval, 300)
                        if now - last_ts < interval_secs:
                            continue
                        logger.info(f"REST poll: {symbol} {interval} (WS last: {now - last_ts:.0f}s ago)")
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
                await asyncio.sleep(30)
            except Exception:
                await asyncio.sleep(60)

    asyncio.create_task(_rest_polling_loop())

    # Train ML models for each symbol on 4h data (most reliable for ML)
    for symbol in default_symbols:
        try:
            result = await ml_predictor.train_model(symbol, "default", "1h")
            if "error" in result:
                logger.warning(f"ML training skipped for {symbol}: {result['error']}")
            else:
                logger.info(f"ML model trained: {symbol} — accuracy={result.get('accuracy', 'N/A')}, f1={result.get('f1', 'N/A')}")
        except Exception as e:
            logger.warning(f"ML training failed for {symbol}: {e}")
    logger.info("ML training round complete")

    # Publish ML predictions and directly seed strategy engine cache
    from core.strategy.indicators import compute_all
    for symbol in default_symbols:
        try:
            df = await market_data.get_historical(symbol, "1h", limit=200)
            if df is not None and len(df) >= 50:
                df = compute_all(df, {"rsi": {"period": 14}, "macd": {"fast": 12, "slow": 26, "signal": 9}, "bollinger": {"period": 20, "stddev": 2}})
                features = ["rsi", "macd_histogram", "bollinger_width", "volume_ratio", "price_momentum_24h"]
                available = [f for f in features if f in df.columns]
                confidence = await ml_predictor.predict(symbol, df, available)
                # Directly seed engine cache (bypasses async event queue)
                strategy_engine._ml_confidence[symbol] = confidence
                await event_bus.publish(Event(EventType.ML_PREDICTION, {
                    "symbol": symbol, "interval": "1h", "confidence": confidence,
                }))
        except Exception:
            pass
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
    web_app.state.alert_manager = alert_manager
    web_app.state.get_price = market_data.get_current_price
    web_app.state.balance = await load_sim_balance(config.db_path)

    # Adjust balance for legacy positions that were opened before balance persistence
    open_positions = order_executor.get_open_positions()
    total_invested = sum(
        p.get("amount_usdt", p.get("quantity", 0) * p.get("entry_price", 0))
        for p in open_positions.values())
    if web_app.state.balance == DEFAULT_BALANCE and total_invested > 0 and total_invested < web_app.state.balance:
        web_app.state.balance -= total_invested
        await save_sim_balance(web_app.state.balance, config.db_path)
        logger.info(f"Adjusted balance for {len(open_positions)} legacy positions: -{total_invested:.2f}")

    # Sync balance and positions to risk manager so position sizing works
    risk_manager.update_balance(web_app.state.balance)
    for sym, pos in open_positions.items():
        risk_manager._open_positions[sym] = pos
    logger.info(f"Risk manager synced: balance={web_app.state.balance:.0f}, positions={len(open_positions)}")

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
