"""Settings write endpoints (DeepSeek, AI/news, risk, Binance) plus
circuit-breaker reset and server restart."""
from pathlib import Path

import yaml
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse

from db.database import get_db

from web.deps import _require_trader, _require_admin, _save_balance
from web.rendering import DEFAULT_BALANCE, _T


def register(app: FastAPI, ctx) -> None:
    config = ctx.config

    @app.post("/api/ai-mode")
    async def set_ai_mode(request: Request, mode: str = Form(...)):
        if err := _require_trader(request): return err
        config.ai_mode = mode
        # Persist to config.yaml
        config_path = Path(getattr(config, "config_dir", Path(__file__).parent.parent.parent / "config")) / "config.yaml"
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("ai", {})["mode"] = mode
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        except Exception:
            pass
        return HTMLResponse(f'<span class="text-green-400">{_T("AI 模式已更新")}</span>')

    @app.post("/api/signal-weights")
    async def update_signal_weights(request: Request, indicator: float = Form(0.5), ml: float = Form(0.3), news: float = Form(0.2)):
        if err := _require_trader(request): return err
        config.update_signal_weights(indicator=indicator, ml=ml, news=news)
        return HTMLResponse('<span class="text-green-400">Weights updated</span>')

    @app.post("/api/settings/deepseek")
    async def save_deepseek_settings(request: Request, api_key: str = Form(""), base_url: str = Form(""), model: str = Form("")):
        if err := _require_admin(request): return err
        secrets_path = Path(getattr(config, "config_dir", Path(__file__).parent.parent.parent / "config")) / "secrets.yaml"
        data = {}
        if secrets_path.exists():
            with open(secrets_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        # Preserve existing binance keys only if they were already in the YAML file
        if "binance" not in data:
            existing_binance = data.get("binance", {})
            if config.binance_api_key and not existing_binance.get("api_key"):
                existing_binance["api_key"] = config.binance_api_key
            if config.binance_api_secret and not existing_binance.get("api_secret"):
                existing_binance["api_secret"] = config.binance_api_secret
            if existing_binance:
                data["binance"] = existing_binance
        # An EMPTY field means "leave unchanged", never "erase". A default/partial
        # form post used to blank the deepseek key AND the base_url / model.
        existing_ds = data.get("deepseek") or {}
        new_ds_key = api_key.strip() or (existing_ds.get("api_key") or config.deepseek_api_key or "")
        data["deepseek"] = {"api_key": new_ds_key}
        with open(secrets_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
        config.deepseek_api_key = new_ds_key
        base_url = base_url.strip() or getattr(config, "ai_base_url", "") or "https://api.deepseek.com"
        model = model.strip() or getattr(config, "ai_model", "") or "deepseek-chat"
        config.ai_base_url = base_url
        config.ai_model = model
        # Persist model + base_url to config YAML
        config_path = Path(getattr(config, "config_dir", Path(__file__).parent.parent.parent / "config")) / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("ai", {})["model"] = model
            cfg.setdefault("ai", {})["base_url"] = base_url
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False)
        return HTMLResponse('<span class="text-green-400 text-sm">✓ DeepSeek 设置已保存</span>')

    @app.post("/api/settings/ai-news")
    async def save_ai_news_settings(request: Request,
        language: str = Form("en"),
        news_fetch_interval: int = Form(30), max_articles: int = Form(10),
        anomaly_threshold: float = Form(3.0),
        task_market_assessment: int = Form(60), task_coin_selection: int = Form(240),
        task_strategy_optimization: int = Form(1440), task_risk_adjustment: int = Form(1440)):
        if err := _require_trader(request): return err
        config.language = language
        config.news_fetch_interval = news_fetch_interval
        config.news_max_articles = max_articles
        config.anomaly_threshold_pct = anomaly_threshold
        config.ai_task_intervals = {
            "market_assessment": task_market_assessment * 60,
            "coin_selection": task_coin_selection * 60,
            "strategy_optimization": task_strategy_optimization * 60,
            "risk_adjustment": task_risk_adjustment * 60,
        }
        config_path = Path(getattr(config, "config_dir", Path(__file__).parent.parent.parent / "config")) / "config.yaml"
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("ai", {})["tasks"] = {
                "market_assessment_minutes": task_market_assessment,
                "coin_selection_minutes": task_coin_selection,
                "strategy_optimization_minutes": task_strategy_optimization,
                "risk_adjustment_minutes": task_risk_adjustment,
            }
            cfg.setdefault("news", {})["fetch_interval_minutes"] = news_fetch_interval
            cfg.setdefault("news", {})["max_articles_per_symbol"] = max_articles
            cfg.setdefault("news", {})["anomaly_threshold_pct"] = anomaly_threshold
            cfg["language"] = language
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        except Exception:
            pass
        return HTMLResponse('<span class="text-green-400 text-sm">✓ AI & 新闻设置已保存</span>')

    @app.post("/api/settings/risk")
    async def save_risk_settings(request: Request, max_daily_drawdown: float = Form(5.0), max_daily_loss: float = Form(500.0),
                                  max_open_trades: int = Form(8), max_position_size_pct: float = Form(10.0),
                                  max_leverage: int = Form(3), max_consecutive_losses: int = Form(5),
                                  circuit_breaker_action: str = Form("block_only"),
                                  trailing_stop_enabled: str = Form("0"),
                                  trailing_stop_distance_pct: float = Form(2.0),
                                  emergency_stop_enabled: str = Form("0"),
                                  emergency_stop_threshold_pct: float = Form(-5.0)):
        # Risk limits are admin-only: loosening drawdown/leverage limits is a
        # capital-safety decision, not a routine trading action.
        if err := _require_admin(request): return err
        config.hard_limits.max_daily_drawdown_pct = max_daily_drawdown
        config.hard_limits.max_daily_loss_usdt = max_daily_loss
        config.hard_limits.max_open_trades = max_open_trades
        config.hard_limits.max_position_size_pct = max_position_size_pct
        config.hard_limits.max_leverage = max_leverage
        config.hard_limits.max_consecutive_losses = max_consecutive_losses
        config.hard_limits.circuit_breaker_action = circuit_breaker_action
        config.hard_limits.trailing_stop_enabled = trailing_stop_enabled == "1"
        config.hard_limits.trailing_stop_distance_pct = trailing_stop_distance_pct
        config.hard_limits.emergency_stop_enabled = emergency_stop_enabled == "1"
        config.hard_limits.emergency_stop_threshold_pct = emergency_stop_threshold_pct
        # Sync to running circuit breaker
        rm = getattr(app.state, "risk_manager", None)
        if rm:
            rm.breaker.max_consecutive_losses = max_consecutive_losses
            rm.breaker.max_daily_drawdown_pct = max_daily_drawdown
            rm.breaker.max_daily_loss_usdt = max_daily_loss
        # Persist to risk_params.yaml (loaded after config.yaml, overrides it)
        risk_path = Path(getattr(config, "config_dir", Path(__file__).parent.parent.parent / "config")) / "risk_params.yaml"
        try:
            with open(risk_path, encoding="utf-8") as f:
                rp = yaml.safe_load(f) or {}
            hl = rp.setdefault("hard_limits", {})
            hl["max_daily_drawdown_pct"] = max_daily_drawdown
            hl["max_daily_loss_usdt"] = max_daily_loss
            hl["max_open_trades"] = max_open_trades
            hl["max_position_size_pct"] = max_position_size_pct
            hl["max_leverage"] = max_leverage
            hl["max_consecutive_losses"] = max_consecutive_losses
            hl["circuit_breaker_action"] = circuit_breaker_action
            hl["trailing_stop_enabled"] = trailing_stop_enabled == "1"
            hl["trailing_stop_distance_pct"] = trailing_stop_distance_pct
            hl["emergency_stop_enabled"] = emergency_stop_enabled == "1"
            hl["emergency_stop_threshold_pct"] = emergency_stop_threshold_pct
            with open(risk_path, "w", encoding="utf-8") as f:
                yaml.dump(rp, f, default_flow_style=False, allow_unicode=True)
        except Exception:
            pass
        return HTMLResponse('<span class="text-green-400 text-sm">✓ Risk settings saved</span>')

    @app.post("/api/settings/binance")
    async def save_binance_settings(request: Request, api_key: str = Form(""), api_secret: str = Form(""), testnet: str = Form(None)):
        # Exchange credentials are admin-only: a trader must not be able to
        # redirect orders to different API keys.
        if err := _require_admin(request): return err
        # `testnet` is only changed when explicitly supplied. Previously it
        # defaulted to "0", so any call that omitted the field (a probe, a script,
        # a partial form post) silently switched the system to LIVE trading.
        if testnet is None:
            testnet_flag = bool(getattr(config, "binance_testnet", True))
        else:
            testnet_flag = (testnet == "1")
        secrets_path = Path(getattr(config, "config_dir", Path(__file__).parent.parent.parent / "config")) / "secrets.yaml"
        data = {}
        if secrets_path.exists():
            with open(secrets_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        # Preserve existing deepseek key if present
        if "deepseek" not in data:
            data["deepseek"] = {"api_key": config.deepseek_api_key}
        # An EMPTY field means "leave unchanged", never "erase the credential".
        # A route sweep / partial form post with default (empty) values used to
        # overwrite the stored keys with 1-character placeholders, silently
        # disconnecting the exchange and the AI provider.
        existing_binance = data.get("binance") or {}
        new_key = api_key.strip() or (existing_binance.get("api_key") or config.binance_api_key or "")
        new_secret = api_secret.strip() or (existing_binance.get("api_secret") or config.binance_api_secret or "")
        data["binance"] = {"api_key": new_key, "api_secret": new_secret}
        with open(secrets_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
        config.binance_api_key = new_key
        config.binance_api_secret = new_secret
        config.binance_testnet = testnet_flag
        # Persist testnet to config.yaml
        config_path = Path(getattr(config, "config_dir", Path(__file__).parent.parent.parent / "config")) / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("binance", {})["testnet"] = testnet_flag
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False)
        return HTMLResponse('<span class="text-green-400 text-sm">✓ Binance settings saved</span>')

    @app.post("/api/settings/reset-sim")
    async def reset_sim_trading(request: Request):
        if err := _require_admin(request): return err
        """Clear all sim trading records and reset balance to 10000."""
        cancelled = 0
        try:
            db = await get_db()
            try:
                # One transaction: the reset is a single state change.  Open limit
                # orders must go with it — they used to survive, and the running
                # matcher then filled a stale order against the FRESH 10000,
                # opening a position whose ledger row had just been erased (and
                # whose cash the reset had already restored).  Cancelled, never
                # deleted: the record of what was withdrawn is kept, and
                # `frozen` (the sum over status='open') is released at once.
                await db.execute("BEGIN IMMEDIATE")
                await db.execute("DELETE FROM trades")
                await db.execute("DELETE FROM positions")
                cursor = await db.execute(
                    "UPDATE pending_orders SET status='cancelled',"
                    " reason=COALESCE(NULLIF(reason, ''), 'sim reset')"
                    " WHERE status='open'")
                cancelled = cursor.rowcount
                await db.commit()
            finally:
                await db.close()
            if cancelled:
                ctx.logger.info(f"reset-sim: cancelled {cancelled} open limit order(s)")
        except Exception as e:
            ctx.logger.warning(f"reset-sim could not clear the sim state: {e}")

        executor = getattr(app.state, "executor", None)
        if executor:
            executor._positions.clear()
            # Keep the per-symbol revisions monotonic: a reduce that read a
            # position before the reset must not be applied to the fresh state.
            for _sym in list(getattr(executor, "_position_rev", {})):
                executor._bump_revision(_sym)

        app.state.balance = DEFAULT_BALANCE
        await _save_balance(ctx)

        resp = HTMLResponse(f'<span class="text-green-400 text-sm">✓ 模拟交易已重置 — 所有记录已清除，余额恢复至 {DEFAULT_BALANCE:.2f} USDT</span>')
        resp.headers["HX-Trigger"] = "tradeUpdated"
        return resp

    @app.post("/api/circuit-breaker/reset")
    async def reset_circuit_breaker(request: Request):
        if err := _require_trader(request): return err
        """Reset the circuit breaker trip state (for manual override)."""
        rm = getattr(app.state, "risk_manager", None)
        if not rm:
            return HTMLResponse('<span class="text-red-400">风控未就绪</span>')
        rm.breaker.reset_trip()
        rm.breaker.reset_daily()
        # Force immediate re-evaluation of all strategies
        engine = getattr(app.state, "strategy_engine", None)
        if engine:
            import asyncio
            asyncio.create_task(engine.evaluate_all_now(publish=True))
        return HTMLResponse('<span class="text-green-400 text-sm">✓ 熔断器已重置 — 交易恢复</span>')

    @app.post("/api/settings/restart")
    async def restart_server(request: Request):
        if err := _require_admin(request): return err
        """Schedule a server restart by spawning a new process and exiting."""
        import subprocess, sys, os, asyncio

        async def _do_restart():
            await asyncio.sleep(0.3)
            cwd = str(Path(__file__).parent.parent.parent)
            subprocess.Popen([sys.executable, "-m", "app.main"], cwd=cwd,
                           creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
            os._exit(0)

        asyncio.ensure_future(_do_restart())
        return HTMLResponse('<span class="text-green-400 text-sm">✓ 服务器正在重启，请等待 5 秒后刷新页面...</span>')
