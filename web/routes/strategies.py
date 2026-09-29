"""Strategy CRUD, symbol mapping, reload and AI recommendation."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from web.deps import _require_trader


def register(app: FastAPI, ctx) -> None:
    config = ctx.config

    # ---- Strategy CRUD ----
    @app.get("/api/strategy/{name}")
    async def get_strategy(name: str):
        loader = getattr(app.state, "strategy_loader", None)
        if not loader:
            return JSONResponse({"error": "No loader"}, status_code=500)
        try:
            s = loader.load(name)
            return s.model_dump()
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    @app.post("/api/strategy")
    async def create_strategy(request: Request):
        if err := _require_trader(request): return err
        loader = getattr(app.state, "strategy_loader", None)
        if not loader:
            return JSONResponse({"error": "No loader"}, status_code=500)
        try:
            from core.strategy.loader import StrategyConfig, MLConfig
            body = await request.json()
            ml = body.get("ml_config")
            if ml:
                body["ml_config"] = MLConfig(**ml)
            config = StrategyConfig(**body)
            loader.save(config)
            # Sync engine so new strategy is immediately available
            engine = getattr(app.state, "strategy_engine", None)
            if engine:
                engine._strategies[config.name] = config
                engine._purge_stale_cache()
            return JSONResponse({"ok": True, "name": config.name})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.put("/api/strategy/{name}")
    async def update_strategy(name: str, request: Request):
        if err := _require_trader(request): return err
        loader = getattr(app.state, "strategy_loader", None)
        if not loader:
            return JSONResponse({"error": "No loader"}, status_code=500)
        try:
            from core.strategy.loader import StrategyConfig, MLConfig
            body = await request.json()
            new_name = body.get("name", name)
            ml = body.get("ml_config")
            if ml:
                body["ml_config"] = MLConfig(**ml)
            config = StrategyConfig(**body)
            # Save new config first; only delete old file if normalized names differ
            loader.save(config)
            # Handle rename: delete old file if normalized name changed
            old_normalized = loader._normalize(name)
            if loader._normalize(new_name) != old_normalized:
                loader.delete(name)
            # Sync engine
            engine = getattr(app.state, "strategy_engine", None)
            if engine:
                if old_normalized != loader._normalize(new_name):
                    engine._strategies.pop(name, None)  # remove old name
                engine._strategies[config.name] = config
                engine._purge_stale_cache()
            return JSONResponse({"ok": True, "name": config.name})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/strategy/{name}/toggle")
    async def toggle_strategy(name: str, request: Request):
        if err := _require_trader(request): return err
        loader = getattr(app.state, "strategy_loader", None)
        if not loader:
            return JSONResponse({"error": "No loader"}, status_code=500)
        try:
            s = loader.load(name)
            s.enabled = not s.enabled
            loader.save(s)
            # Sync engine's in-memory strategies so monitor panel reflects change
            engine = getattr(app.state, "strategy_engine", None)
            if engine and name in engine._strategies:
                engine._strategies[name] = s
                engine._purge_stale_cache()
            return JSONResponse({"ok": True, "enabled": s.enabled})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/strategy-symbols/{name}")
    async def update_strategy_symbols(name: str, request: Request):
        """Update which symbols a strategy applies to."""
        if err := _require_trader(request): return err
        loader = getattr(app.state, "strategy_loader", None)
        if not loader:
            return JSONResponse({"error": "No loader"}, status_code=500)
        try:
            body = await request.json()
            symbols = body.get("symbols", [])
            s = loader.load(name)
            s.symbols = list(symbols)
            loader.save(s)
            # Sync engine
            engine = getattr(app.state, "strategy_engine", None)
            if engine and name in engine._strategies:
                engine._strategies[name] = s
            return JSONResponse({"ok": True, "symbols": s.symbols})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.delete("/api/strategy/{name}")
    async def delete_strategy(name: str, request: Request):
        if err := _require_trader(request): return err
        loader = getattr(app.state, "strategy_loader", None)
        if not loader:
            return JSONResponse({"error": "No loader"}, status_code=500)
        try:
            loader.delete(name)
            # Remove from engine
            engine = getattr(app.state, "strategy_engine", None)
            if engine:
                engine._strategies.pop(name, None)
                engine._purge_stale_cache()
            return JSONResponse({"ok": True})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/strategy/reload")
    async def reload_strategies(request: Request):
        if err := _require_trader(request): return err
        """Reload all strategy YAML files into the engine without restarting."""
        engine = getattr(app.state, "strategy_engine", None)
        if not engine:
            return JSONResponse({"error": "No engine"}, status_code=500)
        try:
            all_s = engine.loader.load_all()
            for s in all_s:
                if not s.timeframes:
                    from loguru import logger
                    logger.warning(f"Strategy '{s.name}' has no timeframes — will never evaluate!")
            engine._strategies = {s.name: s for s in all_s}
            engine._purge_stale_cache()
            # Re-evaluate to populate signal cache immediately
            await engine.evaluate_all_now()
            count = len(engine._strategies)
            names = list(engine._strategies.keys())
            return JSONResponse({"ok": True, "count": count, "strategies": names})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/strategy-recommend")
    async def recommend_strategy(request: Request):
        if err := _require_trader(request): return err
        api_key = config.deepseek_api_key
        if not api_key:
            return JSONResponse({"error": "No API key"}, status_code=500)
        loader = getattr(app.state, "strategy_loader", None)
        existing = []
        if loader:
            for s in loader.load_all():
                existing.append({
                    "name": s.name, "mode": s.mode, "timeframes": s.timeframes,
                    "indicators": list(s.indicators.keys()),
                    "entry_long": s.entry_conditions.get("long", []),
                    "exit_long": s.exit_conditions.get("long", []),
                })
        existing_str = "\n".join([str(e) for e in existing]) if existing else "No existing strategies"
        prompt = f"""Based on these existing trading strategies:
{existing_str}

Recommend a NEW trading strategy for cryptocurrency. Choose indicators, timeframes, entry/exit/reduce conditions that complement (not duplicate) the existing ones.

CRITICAL RULES — your strategy MUST pass these conflict checks:
1. LONG vs SHORT entry: Use mutually exclusive conditions (e.g. RSI<35 long vs RSI>70 short, or close>middle long vs close<middle short). Never allow both long AND short entry to trigger simultaneously.
2. Entry vs Exit: Entry and exit thresholds must have a gap to prevent whipsaw. Example: enter at RSI<35, exit at RSI>65 (30-point gap). Never use adjacent thresholds.
3. Exit vs Reduce: Exit conditions should be stricter than reduce conditions (e.g. reduce 50% at RSI>55, full exit at RSI>65). Reduce fires first, lock in partial profit, then exit later if trend reverses.
4. Include BOTH long AND short conditions for all sections (entry/exit/reduce). No one-sided strategies.
5. All numeric parameters must be explicit (never use empty strings or null). Default: period=14, stddev=2, fast=12, slow=26, signal=9.

Return ONLY valid JSON in this exact format:
{{
  "name": "Strategy Name",
  "mode": "trend",
  "timeframes": ["1h", "4h"],
  "indicators": {{"rsi": {{"period":14, "source":"close"}}, "macd": {{"fast":12, "slow":26, "signal":9}}}},
  "entry_conditions": {{"long": ["rsi < 35 and macd_histogram > 0"], "short": ["rsi > 70 and macd_histogram < 0"]}},
  "exit_conditions": {{"long": ["rsi > 65"], "short": ["rsi < 35"]}},
  "reduce_conditions": {{"long": [{{"condition": "rsi > 55 and close > bollinger_upper", "reduce_pct": 50}}], "short": [{{"condition": "rsi < 40 and close < bollinger_lower", "reduce_pct": 50}}]}},
  "ml_config": {{"enabled": true, "confidence_threshold": 0.6, "features": ["rsi","macd_histogram","volume_ratio","price_momentum_24h"], "weight": 0.3}},
  "rationale": "Why this strategy complements existing ones AND how it passes all conflict checks"
}}"""
        raw = ""
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, base_url=config.ai_base_url)
            resp = await client.chat.completions.create(
                model=config.ai_model,
                messages=[{"role": "system", "content": "You are a quantitative crypto trading strategist. Design strategies with no logical conflicts: long/short entries must be mutually exclusive, entry/exit thresholds must have gaps, reduce fires before exit, both long and short sides required, all parameters numeric. Always respond in valid JSON only, no markdown."},
                          {"role": "user", "content": prompt}],
                max_tokens=4000, temperature=0.6,
            )
            raw = (resp.choices[0].message.content or "").strip()
            import json as j
            # Extract JSON: find the first { and last }
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                text = raw[start:end + 1]
            else:
                text = raw
            # Strip markdown code fences
            for prefix in ["```json", "```"]:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            for suffix in ["```", "```json"]:
                if text.endswith(suffix):
                    text = text[:-len(suffix)].strip()
            if not text:
                return JSONResponse({"error": "AI returned empty response"}, status_code=500)
            return j.loads(text)
        except Exception as e:
            err = str(e)
            detail = err[:200]
            if "JSONDecodeError" in type(e).__name__ or "Expecting" in err:
                detail = f"JSON解析失败 [{err[:100]}] 原始响应前200字: {raw[:200]}"
            return JSONResponse({"error": detail}, status_code=500)
