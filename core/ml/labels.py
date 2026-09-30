"""Cost-aware label construction for the ML pipeline (Phase P2).

Why this module exists
----------------------
The legacy live target (``create_binary_label(df, forward_periods=4,
threshold=0.005)`` in :mod:`core.ml.features`) drops every bar whose 4-bar
forward return stays inside ±0.5 %:

* BTCUSDT 1h keeps only 40.5 % of bars, BTCUSDT 1m only 15.3 %,
* the model is nevertheless asked to decide on **every** bar,
* and the round-trip cost (≈0.09 % = 2 × 0.04 % taker + 0.01 % half-spread)
  is never mentioned, so "up 0.5 %" is treated as a win even though the
  realised edge is ≈0.41 %.

Two labels are provided here:

``create_three_class_label``
    up / flat / down — the "no-move" regime is kept as its own class instead
    of being silently deleted, so the decision path can *abstain* on it
    (see :func:`decision_from_probs`) and coverage is reportable.

``create_triple_barrier_label_vol``
    volatility-scaled (ATR) barriers with the time barrier tied to the
    strategy's real maximum holding period.

Both are pure functions of OHLCV — no I/O, no config object — so they are
safe to use from the trainer, the predictor and a research script alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Column layout of the three-class label (also the softmax class order).
CLASS_DOWN = 0
CLASS_UP = 1
CLASS_FLAT = 2
CLASS_NAMES = ("down", "up", "flat")


# ── cost helpers ─────────────────────────────────────────────────────────

def round_trip_cost_pct(
    *,
    taker_fee_pct: float = 0.04,
    half_spread_pct: float = 0.01,
    slippage_bps: float = 2.0,
    use_bnb_discount: bool = False,
) -> float:
    """Round-trip cost in **percent of notional**, same components as the fills.

    Mirrors ``app.config.sim_cost_quote`` (the single source of truth for real
    fills): one taker fee on each side, plus ``half_spread + slippage`` on each
    side (the aggressor pays half the quoted spread and the slippage).

    ``0.04 %`` taker × 2 + ``0.01 %`` half-spread × 2 + ``2 bp`` slippage × 2
    = ``0.04*2 + 0.01*2 + 0.02*2`` / ... — expressed directly in percent here.
    """
    fee = float(taker_fee_pct) * (0.75 if use_bnb_discount else 1.0)
    per_side = fee + float(half_spread_pct) + float(slippage_bps) / 100.0
    return float(2.0 * per_side)


def net_return(fwd_return: np.ndarray | pd.Series, cost_pct: float) -> np.ndarray:
    """Forward return (fraction) minus the round-trip cost (percent) → fraction."""
    arr = np.asarray(fwd_return, dtype=float)
    return arr - float(cost_pct) / 100.0


# ── labels ───────────────────────────────────────────────────────────────

def create_three_class_label(
    df: pd.DataFrame,
    forward_periods: int = 4,
    threshold: float = 0.005,
    *,
    cost_pct: float | None = None,
    cost_multiple: float = 0.0,
) -> pd.Series:
    """3-class target: 1 = up, 0 = down, 2 = flat (the dropped regime, kept).

    Parameters
    ----------
    df : DataFrame with a ``close`` column.
    forward_periods : int
        Label horizon in bars (this is also the label *span* used by the
        purged/embargoed splitter).
    threshold : float
        Fractional move that separates a direction from "no move" (0.005 = 0.5 %).
    cost_pct : float | None
        Round-trip cost in percent.  When given together with
        ``cost_multiple > 0`` the effective threshold becomes
        ``max(threshold, cost_multiple * cost_pct / 100)`` so the target is
        never smaller than a multiple of the cost it has to beat.
    cost_multiple : float
        Default 0 (pure threshold, backward compatible with the old numbers).

    Returns
    -------
    pd.Series of ``Int64`` with values in {0, 1, 2}; the last
    ``forward_periods`` rows are ``NA`` (no complete forward window).
    """
    close = df["close"].astype(float)
    fwd = (close.shift(-int(forward_periods)) - close) / close

    thr = abs(float(threshold))
    if cost_pct is not None and cost_multiple and cost_multiple > 0:
        thr = max(thr, float(cost_multiple) * abs(float(cost_pct)) / 100.0)

    labels = pd.Series(pd.NA, index=df.index, dtype="Int64")
    labels[fwd >= thr] = CLASS_UP
    labels[fwd <= -thr] = CLASS_DOWN
    labels[fwd.notna() & (fwd > -thr) & (fwd < thr)] = CLASS_FLAT
    return labels


def decision_from_probs(
    proba: np.ndarray,
    *,
    threshold_up: float = 0.5,
    threshold_down: float | None = None,
) -> np.ndarray:
    """Signed decision from 3-class probabilities with an explicit abstention.

    ``flat`` is an abstention, not a hidden 0.5 vote: a row is only traded when
    its ``p_up`` (resp. ``p_down``) clears its threshold.  Returns 1 / -1 / 0.
    """
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("decision_from_probs expects an (n, 3) probability array")
    t_down = threshold_up if threshold_down is None else threshold_down
    out = np.zeros(len(p), dtype="int8")
    out[p[:, CLASS_UP] >= threshold_up] = 1
    out[p[:, CLASS_DOWN] >= t_down] = -1
    return out


def flat_share(labels: pd.Series) -> float:
    """Fraction of labelled rows in the flat/abstain class (0.0 when empty)."""
    valid = labels.dropna()
    if len(valid) == 0:
        return 0.0
    return float((valid == CLASS_FLAT).sum()) / float(len(valid))


# ── volatility-scaled triple barrier ─────────────────────────────────────

def rolling_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR, computed without TA-Lib so this module has no heavy imports."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / max(int(period), 1), adjust=False, min_periods=period).mean()


def barrier_widths(
    df: pd.DataFrame,
    *,
    atr_period: int = 14,
    atr_multiple: float = 1.5,
    min_pct: float = 0.004,
    max_pct: float = 0.06,
) -> tuple[pd.Series, pd.Series]:
    """Per-bar ``(upper_pct, lower_pct)`` barriers, scaled by ATR/close.

    A fixed 2 % barrier is ~7× the median 1h BTC ATR%, so almost nothing is
    touched and the timeout class swallows the sample (44.9 % measured).  The
    width here is ``atr_multiple × ATR / close`` clamped to
    ``[min_pct, max_pct]`` — high-volatility regimes widen (fewer premature
    stops), low-volatility regimes narrow (the model still gets a signal).
    """
    atr = rolling_atr(df, atr_period)
    close = df["close"].astype(float)
    width = (atr_multiple * atr / close).clip(lower=min_pct, upper=max_pct)
    width = width.bfill().fillna(min_pct)
    return width, width


def create_triple_barrier_label_vol(
    df: pd.DataFrame,
    *,
    forward_periods: int = 24,
    atr_period: int = 14,
    atr_multiple: float = 1.5,
    min_pct: float = 0.004,
    max_pct: float = 0.06,
    timeout_label: float | None = 2.0,
    max_rows: int | None = None,
) -> pd.Series:
    """Path-aware, volatility-scaled triple-barrier label.

    Class order matches the model's softmax: **0 = lower barrier first
    (bearish), 1 = upper barrier first (bullish), 2 = timeout** (or ``NA`` when
    ``timeout_label is None``).

    ``forward_periods`` is the time barrier and should equal the strategy's real
    maximum holding period (bar count), not an arbitrary 24: the old fixed
    ``24 × 2 %`` combination made 44.9 % of labels timeouts.

    Only rows with a **full** forward window are labelled: the last
    ``forward_periods`` rows are always ``NA`` (audit P2 #6).  With
    ``timeout_label`` set, ``NA`` *inside* the sample is filled with the timeout
    class — the tail is not, because those rows have no complete window at all
    (forcing them to timeout moved 24 bars and destroyed 17 genuine hits).

    ``max_rows`` keeps the O(n × horizon) scan bounded for research use (the
    most recent ``max_rows`` bars are labelled, earlier rows become ``NA``).
    """
    n = len(df)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    up_w, lo_w = barrier_widths(
        df, atr_period=atr_period, atr_multiple=atr_multiple,
        min_pct=min_pct, max_pct=max_pct)
    up_w = up_w.values.astype(np.float64)
    lo_w = lo_w.values.astype(np.float64)

    labels = np.full(n, np.nan)
    horizon = max(int(forward_periods), 1)
    start = 0
    if max_rows is not None:
        start = max(0, n - int(max_rows))
    # Only rows with a FULL forward window are labelled (audit P2 #6).  Scanning
    # to ``n - 1`` also touched the last ``horizon - 1`` bars, whose window is
    # truncated at the end of the data, and then forced them into the timeout
    # class — that alone moved 24 bars out of the directional classes and
    # destroyed 17 genuine barrier hits.
    stop = max(start, n - horizon)
    for i in range(start, stop):
        entry = close[i]
        upper = entry * (1.0 + up_w[i])
        lower = entry * (1.0 - lo_w[i])
        end = min(i + horizon, n - 1)
        hit = np.nan
        for j in range(i + 1, end + 1):
            if high[j] >= upper:
                hit = 1.0
                break
            if low[j] <= lower:
                hit = 0.0
                break
        labels[i] = hit

    result = pd.Series(labels, index=df.index)
    # The last ``horizon`` rows have no complete forward window and stay ``NA``
    # in **both** modes: with ``timeout_label`` they are *not* filled (only
    # genuine timeouts are), and with ``timeout_label = None`` the whole
    # timeout class is ``NA`` — the pre-P2 docstring claimed the opposite.
    return result


def class_distribution(labels: pd.Series) -> dict:
    """``{class_value: share}`` over the non-NA rows plus ``timeout_share``."""
    valid = pd.Series(labels).dropna()
    total = len(valid)
    if total == 0:
        return {"n": 0, "timeout_share": 0.0}
    shares = {int(k): float(v) / total for k, v in valid.value_counts().items()}
    return {
        "n": int(total),
        "shares": shares,
        "timeout_share": float(shares.get(2, 0.0)),
    }
