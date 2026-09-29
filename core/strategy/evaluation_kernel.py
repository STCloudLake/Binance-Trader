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
  detect_market_regime       — EMA(20)/EMA(50) bull/bear/range classification

Thresholds (:data:`ENTRY_THRESHOLD` / :data:`COUNTER_TREND_THRESHOLD`) are
applied by the vectorized side-resolution helpers below; the per-tick engines
apply the same values through :func:`fuse_signals`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy.indicators import evaluate_condition

# ── Entry thresholds (single source of truth for live + backtest) ───────
ENTRY_THRESHOLD = 0.5
#: Counter-trend entries require a stronger signal (regime-aware gate).
COUNTER_TREND_THRESHOLD = 0.65



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

def detect_market_regime(df: pd.DataFrame | None) -> str:
    """Detect the prevailing regime from EMA(20)/EMA(50) alignment.

    Returns ``"bull"``, ``"bear"`` or ``"range"``. Requires >= 100 bars,
    otherwise ``"range"`` (no regime penalty).
    """
    if df is None or len(df) < 100:
        return "range"
    close = df["close"].values
    ema20 = float(pd.Series(close).ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(pd.Series(close).ewm(span=50, adjust=False).mean().iloc[-1])
    last_close = float(close[-1])
    if last_close > ema20 > ema50:
        return "bull"
    if last_close < ema20 < ema50:
        return "bear"
    return "range"


# ── Vectorized counterparts (used by the hybrid/vectorized backtest engine) ──
#
# The two-phase hybrid engine evaluates conditions in bulk with pandas instead of
# per-tick Python. These helpers guarantee it applies the *same* semantics as
# `evaluate_entry_conditions` + `fuse_signals` + `check_higher_tf_trend`, which
# is what the live engine uses. Before this existed the vectorized path used AND
# logic and skipped fusion/threshold entirely, so GA optimised strategies against
# different semantics than the ones used to trade them.

def check_higher_tf_trend_series(df_htf: pd.DataFrame,
                                 index: pd.DatetimeIndex,
                                 entry_side: str) -> pd.Series:
    """Vectorized :func:`check_higher_tf_trend` over a whole timestamp index."""
    baseline = pd.Series(1.0, index=index, dtype="float64")
    if df_htf is None or len(df_htf) < 50:
        return baseline

    close = df_htf["close"]
    ema50 = close.ewm(span=50, adjust=False).mean()
    deviation = (close - ema50) / ema50.where(ema50 != 0)

    # Bars available as of each timestamp (mirrors `df[df.index <= ts]` + len<50 guard)
    counts = pd.Series(1, index=df_htf.index).cumsum()
    dev_r = deviation.reindex(index, method="ffill")
    cnt_r = counts.reindex(index, method="ffill").fillna(0)

    mult = pd.Series(1.0, index=index, dtype="float64")
    if entry_side == "long":
        mult[(dev_r < 0) & (dev_r > -0.02)] = 0.6
        mult[dev_r <= -0.02] = 0.0
    else:
        mult[(dev_r > 0) & (dev_r < 0.02)] = 0.6
        mult[dev_r >= 0.02] = 0.0
    mult = mult.where(dev_r.notna(), 1.0)
    mult[cnt_r < 50] = 1.0
    return mult


def fuse_signals_series(indicator: pd.Series, *,
                        w_indicator: float = 0.6,
                        w_ml: float = 0.3,
                        w_news: float = 0.1,
                        ml_enabled: bool = True,
                        strategy_ml_weight: float | None = None,
                        ml_confidence: float = 0.5,
                        news_sentiment: float | None = None) -> pd.Series:
    """Vectorized :func:`fuse_signals` (same formula, same divisor rules)."""
    ml_directional = 0.0 if not ml_enabled else (ml_confidence - 0.5) * 2.0
    if ml_enabled and strategy_ml_weight is not None:
        effective_ml_weight = strategy_ml_weight
    else:
        effective_ml_weight = w_ml

    total_weight = w_indicator + effective_ml_weight + w_news
    indicator = indicator.astype("float64")
    if total_weight <= 0:
        return indicator

    news_term = (news_sentiment or 0.0) * w_news if news_sentiment is not None else 0.0
    return (indicator * w_indicator + ml_directional * effective_ml_weight + news_term) / total_weight


def build_entry_signals(long_active: pd.Series | None,
                        short_active: pd.Series | None, *,
                        index: pd.DatetimeIndex,
                        w_indicator: float = 0.6,
                        w_ml: float = 0.3,
                        w_news: float = 0.1,
                        ml_enabled: bool = False,
                        strategy_ml_weight: float | None = None,
                        htf_frames: list[pd.DataFrame] | None = None,
                        regime: str | None = "range") -> pd.Series:
    """Vectorized equivalent of the live entry pipeline.

    Reproduces, in order:
      1. OR-combined entry conditions per side (``evaluate_entry_conditions``)
      2. ambiguity guard — both sides active → no signal
      3. weighted fusion (``fuse_signals``, news absent in backtest)
      4. higher-timeframe EMA(50) alignment multiplier (worst case over TFs)
      5. regime-aware threshold (:data:`ENTRY_THRESHOLD` base, raised to
         :data:`COUNTER_TREND_THRESHOLD` for the counter-trend side only)

    Returns an int8 Series aligned to *index*: 1 long, -1 short, 0 no signal.
    """
    zeros = pd.Series(False, index=index, dtype=bool)
    long_s = long_active.reindex(index, fill_value=False).astype(bool) if long_active is not None else zeros
    short_s = short_active.reindex(index, fill_value=False).astype(bool) if short_active is not None else zeros

    indicator = pd.Series(0.0, index=index, dtype="float64")
    indicator[long_s & ~short_s] = 1.0
    indicator[short_s & ~long_s] = -1.0

    score = fuse_signals_series(
        indicator,
        w_indicator=w_indicator, w_ml=w_ml, w_news=w_news,
        ml_enabled=ml_enabled, strategy_ml_weight=strategy_ml_weight,
    )

    if htf_frames:
        mult_long = pd.Series(1.0, index=index, dtype="float64")
        mult_short = pd.Series(1.0, index=index, dtype="float64")
        for df_htf in htf_frames:
            if df_htf is None or len(df_htf) < 50:
                continue
            mult_long = pd.concat(
                [mult_long, check_higher_tf_trend_series(df_htf, index, "long")], axis=1
            ).min(axis=1)
            mult_short = pd.concat(
                [mult_short, check_higher_tf_trend_series(df_htf, index, "short")], axis=1
            ).min(axis=1)
        multiplier = pd.Series(1.0, index=index, dtype="float64")
        multiplier[indicator > 0] = mult_long[indicator > 0]
        multiplier[indicator < 0] = mult_short[indicator < 0]
        score = score * multiplier

    # Base threshold applies to BOTH sides; only the counter-trend side is raised.
    # (An earlier helper returned the counter-trend value for *both* sides whenever
    # regime == "bear", so with-trend shorts were held to the counter-trend
    # threshold as well; it had no remaining caller and was removed.)
    threshold = pd.Series(ENTRY_THRESHOLD, index=index, dtype="float64")
    if regime in ("bull", "bear"):
        counter_mask = ((indicator > 0) & (regime == "bear")) | ((indicator < 0) & (regime == "bull"))
        threshold[counter_mask] = COUNTER_TREND_THRESHOLD

    passed = score.abs() >= threshold
    values = np.sign(score).astype("int8")
    values[~passed] = 0
    return pd.Series(values, index=index, dtype="int8")

