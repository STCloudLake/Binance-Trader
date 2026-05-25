import pytest
import json
from app.config import Config


def test_deepseek_controller_init():
    from core.ai.deepseek_ctl import DeepSeekController
    from app.event_bus import EventBus

    Config._instance = None
    config = Config.load("sim")
    bus = EventBus()

    ctl = DeepSeekController(config, bus)
    assert ctl is not None
    assert ctl.config == config


def test_market_assessment_no_api_key():
    from core.ai.deepseek_ctl import DeepSeekController
    from app.event_bus import EventBus

    Config._instance = None
    config = Config.load("sim")
    bus = EventBus()

    ctl = DeepSeekController(config, bus)
    # With empty API key, should handle gracefully
    import asyncio
    result = asyncio.run(ctl.assess_market())
    assert result is None  # No API key means no result


def test_prompts_format():
    from core.ai import prompts
    assert "{market_summary}" in prompts.COIN_SELECTION_PROMPT
    assert "{recent_trades}" in prompts.STRATEGY_OPTIMIZATION_PROMPT
    assert "{balance}" in prompts.RISK_ADJUSTMENT_PROMPT
    assert "{price_data}" in prompts.MARKET_ASSESSMENT_PROMPT
    assert "{title}" in prompts.NEWS_ANALYSIS_PROMPT
