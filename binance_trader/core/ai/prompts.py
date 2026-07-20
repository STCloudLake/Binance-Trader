"""Prompt templates for DeepSeek AI controller's 5 responsibilities."""

COIN_SELECTION_PROMPT = """You are a professional cryptocurrency portfolio analyst. Based on the following context, recommend the best coins for trading.

Current state:
{context}

Your task:
1. Select 3-5 "core" coins for swing/trend trading (4h-1d timeframe)
2. Select 5-10 "satellite" coins for day trading/scalping opportunities
3. For each coin, provide a score (0-100) and brief rationale
4. Consider: trend strength, volatility, volume, recent news sentiment

Return ONLY valid JSON format:
{{
  "core": [
    {{"symbol": "BTCUSDT", "score": 85, "rationale": "Strong uptrend, high volume"}}
  ],
  "satellite": [
    {{"symbol": "SOLUSDT", "score": 72, "rationale": "High volatility, good momentum"}}
  ],
  "market_overview": "Brief market assessment"
}}"""

STRATEGY_OPTIMIZATION_PROMPT = """You are a quantitative trading strategist. Analyze the current state and suggest parameter optimizations.

Current state:
{context}

Suggest optimizations for strategy parameters. Consider:
1. Entry/exit threshold adjustments
2. Indicator period changes
3. Signal weight rebalancing
4. Timeframe adjustments

Return ONLY valid JSON:
{{
  "suggestions": [
    {{
      "parameter": "rsi_entry_threshold",
      "current_value": 30,
      "suggested_value": 28,
      "rationale": "Lower threshold reduces false signals in current market"
    }}
  ],
  "overall_assessment": "Summary of strategy health"
}}"""

RISK_ADJUSTMENT_PROMPT = """You are a risk management expert. Based on the current portfolio state and market conditions, adjust risk parameters.

Current state:
{context}

Determine:
1. Risk appetite (conservative/balanced/aggressive)
2. Position size adjustment
3. Stop-loss width adjustment
4. Leverage adjustment

Return ONLY valid JSON:
{{
  "risk_appetite": "balanced",
  "position_size_pct": 5.0,
  "stop_loss_pct": 2.0,
  "leverage": 2,
  "rationale": "Explanation of adjustments"
}}"""

MARKET_ASSESSMENT_PROMPT = """You are a crypto market analyst. Assess the current market state.

Current state:
{context}

Determine:
1. Market regime (bullish/bearish/ranging/high_volatility)
2. Recommended strategy types
3. Signal weight adjustments (indicator/ml/news)
4. Key risk factors to watch

Return ONLY valid JSON:
{{
  "regime": "bullish",
  "confidence": 0.75,
  "recommended_strategies": ["trend", "momentum"],
  "signal_weights": {{
    "indicator": 0.5,
    "ml": 0.35,
    "news": 0.15
  }},
  "risk_factors": ["Fed announcement", "BTC halving proximity"],
  "summary": "Brief market analysis"
}}"""

NEWS_ANALYSIS_PROMPT = """Analyze this cryptocurrency news for trading sentiment.

Headline: {title}
Summary: {summary}
Related symbol: {symbol}

Provide a structured analysis:
1. Sentiment direction (positive/negative/neutral)
2. Impact level (1-5)
3. Duration of impact (short/medium/long)
4. Confidence in assessment
5. Actionable trading implication

Return ONLY valid JSON:
{{
  "sentiment": "positive",
  "sentiment_score": 0.7,
  "impact_level": 3,
  "duration": "medium",
  "confidence": 0.8,
  "trading_implication": "Potential upward movement in next 48h"
}}"""

BREAKER_ACTION_PROMPT = """You are a risk management expert. A circuit breaker has just tripped. Decide the immediate action.

Breaker state:
{context}

Available actions:
- block_only: Only block new entries, leave existing positions alone
- tighten_stops: Tighten stop-loss on all positions to 2% from current price
- close_all: Market-close ALL open positions immediately
- close_worst: Only close the position with the largest unrealized loss

Choose the action that best protects the portfolio given the current drawdown and positions.

Return ONLY valid JSON:
{{
  "action": "close_all",
  "rationale": "Brief explanation of why this action was chosen"
}}"""

BREAKER_RECOVERY_PROMPT = """You are a risk management expert. The circuit breaker is currently tripped. Evaluate whether it is safe to resume trading.

{context}

Consider:
1. Has enough time passed since the trip?
2. Is the market regime favorable for re-entry?
3. Have positions been resolved?

Return ONLY valid JSON:
{{
  "reset": true,
  "reason": "Brief explanation of decision"
}}"""
