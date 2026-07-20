"""
Shared Evaluation Kernel — single source of truth for strategy signal evaluation.

Extracted from live StrategyEngine and BacktestEngine to eliminate the
backtest/live code fork (Audit Root Cause #2). Both engines now call the
same functions, guaranteeing identical behaviour.

Components:
  evaluate_entry_conditions  — OR-logic entry signal evaluation
  evaluate_exit_conditions   — per-side exit signal evaluation
  fuse_signals               — weighted fusion of indicator + ML + news
  check_higher_tf_trend      — confidence multiplier from EMA(50) alignment
  resolve_entry_side         — convert fused score to a trading side
"""

from __future__ import annotations

import pandas as pd

from core.strategy.indicators import evaluate_condition


# ── Entry Condition Evaluation ──────────────────────────────────────────

def evaluate_entry_conditions(df: pd.DataFrame,
                              entry_conditions: dict[str, list[str]]
                              ) -> tuple[bool, bool]:
    """Evaluate entry conditions with OR logic (ANY condition met = side active).

    Args:
        df: Indicator DataFrame (already passed through compute_all).
        entry_conditions: ``strategy.entry_conditions`` dict, e.g.
            ``{"long": ["close > sma_20"], "short": ["rsi < 30"]}``.

    Returns:
        (long_active, short_active) — both can be True simultaneously
        (caller should treat that as ambiguous / no-trade).
    """
    long_active = False
    short_active = False

    for side in ("long", "short"):
        conditions = entry_conditions.get(side, [])
        for cond in conditions:
            mask = evaluate_condition(df, cond)
            met = bool(hasattr(mask, 'iloc') and mask.iloc[-1])
            if met:
                if side == "long":
                    long_active = True
                else:
                    short_active = True
                break  # OR logic — first met condition activates the side

    return long_active, short_active


# ── Exit Condition Evaluation ───────────────────────────────────────────

def evaluate_exit_conditions(df: pd.DataFrame,
                             exit_conditions: dict[str, list[str]],
                             pos_side: str) -> bool:
    """Check whether any exit condition for *pos_side* is met (OR logic).

    Args:
        df: Indicator DataFrame.
        exit_conditions: ``strategy.exit_conditions`` dict.
        pos_side: ``"long"`` or ``"short"`` — only this side's exits are checked.

    Returns:
        True if the position should be closed.
    """
    conditions = exit_conditions.get(pos_side, [])
    for cond in conditions:
        mask = evaluate_condition(df, cond)
        if hasattr(mask, 'iloc') and mask.iloc[-1]:
            return True
    return False


# ── Signal Fusion ───────────────────────────────────────────────────────

def fuse_signals(*,
                 indicator_signal: float,
                 ml_confidence: float = 0.5,
                 news_sentiment: float | None = None,
                 w_indicator: float = 0.6,
                 w_ml: float = 0.3,
                 w_news: float = 0.1,
                 ml_enabled: bool = True,
                 strategy_ml_weight: float | None = None,
                 ) -> float:
    """Weighted fusion of indicator, ML, and news signals.

    Formula::

        ml_directional = (ml_confidence - 0.5) * 2   # 0→1  maps to  -1→+1
        total_weight   = w_indicator + effective_ml_weight + w_news
        final_score    = (indicator*w_ind + ml_dir*effective_ml + news*w_news)
                         / total_weight

    Args:
        indicator_signal: -1 (short), 0 (neutral), or +1 (long) from indicators.
        ml_confidence:   ML model confidence in [0, 1].  0.5 = neutral.
        news_sentiment:  News sentiment in [-1, +1].  Pass **None** to exclude
                         from numerator (backtest mode — no historical news).
        w_indicator:     Weight for indicator signal (default 0.6).
        w_ml:            Base weight for ML signal (default 0.3).
        w_news:          Weight for news signal (default 0.1).
        ml_enabled:      Whether ML is enabled for this strategy.
        strategy_ml_weight: Per-strategy ML weight override.  If *None*,
                         falls back to *w_ml*.

    Returns:
        Fused final score in [-1, +1].  |score| ≥ 0.5 is the standard
        entry threshold.
    """
    # ML directional transform: confidence ∈ [0,1] → signal ∈ [-1, +1]
    # When ML is disabled, ignore any stale confidence values (defense in depth)
    ml_directional = 0.0 if not ml_enabled else (ml_confidence - 0.5) * 2.0

    # Effective ML weight:
    # - Per-strategy override takes priority (when ml_enabled and configured)
    # - Otherwise falls back to base w_ml (always in divisor, even when disabled —
    #   this is intentional: when ML produces no signal, its weight still dilutes
    #   the indicator to prevent non-ML strategies from being too aggressive)
    if ml_enabled and strategy_ml_weight is not None:
        effective_ml_weight = strategy_ml_weight
    else:
        effective_ml_weight = w_ml

    total_weight = w_indicator + effective_ml_weight + w_news

    if total_weight <= 0:
        return float(indicator_signal)

    # News is included in the DIVISOR to dampen the score when news weight
    # is configured but no actual news signal is available.  In live mode
    # (news_sentiment is not None) it also contributes to the numerator.
    news_term = (news_sentiment or 0.0) * w_news if news_sentiment is not None else 0.0

    final_score = (
        indicator_signal * w_indicator
        + ml_directional * effective_ml_weight
        + news_term
    ) / total_weight

    return float(final_score)


# ── Higher-Timeframe Trend Alignment ────────────────────────────────────

def check_higher_tf_trend(df: pd.DataFrame, entry_side: str) -> float:
    """Return a confidence multiplier (0.0–1.0) based on EMA(50) alignment.

    Instead of a hard block, this penalises counter-trend entries:
    - 1.0 = strongly aligned (price on correct side of EMA)
    - 0.6 = weakly counter-trend (within 2% of EMA)
    - 0.0 = extreme counter-trend (should not enter)

    Args:
        df: Higher-timeframe DataFrame with at least a ``"close"`` column.
        entry_side: ``"long"`` or ``"short"``.

    Returns:
        Confidence multiplier in [0.0, 1.0].
    """
    if len(df) < 50:
        return 1.0  # not enough data — no penalty

    close = df["close"].values
    ema50 = float(pd.Series(close).ewm(span=50, adjust=False).mean().iloc[-1])
    last_close = float(close[-1])

    if ema50 <= 0:
        return 1.0

    # deviation = how far price is from EMA, as a fraction
    deviation = (last_close - ema50) / ema50  # positive = above EMA, negative = below

    if entry_side == "long":
        if deviation >= 0:
            return 1.0  # price above EMA — aligned
        elif deviation > -0.02:  # within 2% below EMA
            return 0.6  # mild penalty
        else:
            return 0.0  # extreme counter-trend
    else:  # short
        if deviation <= 0:
            return 1.0  # price below EMA — aligned
        elif deviation < 0.02:  # within 2% above EMA
            return 0.6  # mild penalty
        else:
            return 0.0  # extreme counter-trend


# ── Side Resolution ─────────────────────────────────────────────────────

def resolve_entry_side(final_score: float, threshold: float = 0.5) -> str | None:
    """Convert a fused final_score to a trading side.

    Args:
        final_score: Signal from :func:`fuse_signals`.
        threshold:   Minimum |score| to produce a signal (default 0.5).

    Returns:
        ``"long"``, ``"short"``, or **None** (below threshold).
    """
    if abs(final_score) < threshold:
        return None
    return "long" if final_score > 0 else "short"
