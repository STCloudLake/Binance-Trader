"""Meta-labelling (López de Prado, *Advances in Financial ML* ch. 3) for the ML gate.

Principle
---------
The measured problem with the current ML pipeline is that the model is asked to
**choose a direction** on every bar while its OOS AUC is 0.396–0.447 against a
0.543–0.667 majority baseline — i.e. a measured negative contribution.  Prado's
answer is to stop asking ML for a direction:

* **Primary model** (an existing rule strategy, a GA champion signature, or any
  ``{-1, 0, +1}`` series) decides *when* to trade and *which side*;
* **Secondary model** predicts one thing only: *does this primary trade reach
  its profit barrier before its loss barrier?*
* the secondary output is used as a **filter** (drop trades whose probability is
  below the cost-aware threshold) and a **size multiplier** (scale the ones that
  pass).  It can never flip a side.

That is the highest-ROI use of ML here: the primary rule supplies whatever
directional edge it has, and the meta-model only has to rank trades — a strictly
easier problem than forecasting the sign of a return.

Label
-----
``y_meta = 1`` when the primary trade's **profit barrier is touched first**,
``0`` when the loss barrier is touched first, ``NA`` on a time-out (the barrier
labels come from :func:`core.ml.labels.create_triple_barrier_label_vol`
semantics: class 1 = upper barrier first, 0 = lower barrier first, 2 = timeout;
a long primary is profitable on class 1, a short on class 0).  Time-outs are
**dropped**, never forced into a class: a trade that hit neither barrier has no
meta-label to learn from.

Metrics and the gate (the honest protocol)
-------------------------------------------
:func:`evaluate_meta_oos` runs the *same* nested protocol as
:func:`core.ml.credibility.evaluate_model_oos`:

1. ``purged_kfold_splits`` (embargo = label horizon) with
   ``sample_uniqueness_weights`` — both from :mod:`core.ml.evaluation`;
2. per fold, the tail 20 % of the training block is the **calibration stream**;
   the fold's model is fitted on the rest;
3. the decision threshold is selected **inside** the fold on that stream
   (:func:`meta_cost_aware_threshold`, one-sided: ``p ≥ t`` means *take the
   primary trade* — the secondary model is never allowed to short the spread);
4. the folded threshold is applied to the fold's test rows, and the gate consumes
   the resulting **outer** net expectancy, trade count, t-stat and PSR through
   :func:`core.ml.credibility.gate_from_evaluation` (AUC > 0.55, net
   expectancy > 0, ≥ 100 trades, t > 2 **and** PSR ≥ 0.95, where the PSR is
   Prado's skew/kurtosis-corrected PSR of the outer net-trade returns — the
   higher-moment correction landed with re-audit finding 4, so the PSR floor is a
   genuine second condition and not just ``t >= 1.645``).

Costs come from ``core.ml.credibility.cost_pct_for`` (the sim cost model), so the
gate can never gate on a cheaper cost than the fills pay.

Size multiplier::

    take   = p ≥ threshold
    size   = META_SIZE_FLOOR + (META_MAX_SIZE_MULTIPLIER − META_SIZE_FLOOR)
             · clip((p − threshold) / (1 − threshold), 0, 1)

Limitations
-----------
* The primary signal must be **causal**: it is the caller's job to pass a series
  whose value at ``t`` uses only bars ≤ ``t`` (both helpers here do).
* The meta-label is a *barrier hit* classification, so it does not price the
  path (a trade that touches the profit barrier after a 5× drawdown scores 1).
* The primary sample is much smaller than the bar count: the power of the gate
  is limited by the number of primary trades, and 100 outer trades is the floor
  — a primary rule that fires 40 times cannot produce a passing meta-model, by
  construction.
* Nothing here is wired into the live path: :data:`META_LABELING_ENABLED` is
  ``False`` and ``StrategyEngine`` only consults a meta model that a caller
  explicitly registered.

Note on ``core/ml/volatility.py`` (phase P3): it did not exist when this module
was written, so :func:`_barrier_widths` takes the ATR barrier construction from
``core.ml.labels`` (still the shared source) and falls back to a local Wilder ATR
if that helper disappears.  ``core.ml.volatility``'s forecasts are the natural
next refinement of the barrier width (a conditional volatility instead of a
trailing ATR) — that is a documented seam, not an implemented one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.ml.calibration import ProbabilityCalibrator
from core.ml.credibility import (
    GATE_MIN_TRADES, MIN_CAL_ROWS, cost_pct_for, gate_from_evaluation,
    net_expectancy, net_trade_stats,
)
from core.ml.evaluation import (
    binary_metrics, purged_kfold_splits, sample_uniqueness_weights,
)

# ── switches / tunables (all OFF by default) ─────────────────────────────

#: Master switch.  ``False`` means no live code path consults a meta model.
META_LABELING_ENABLED = False
#: Trade floor for the *outer* meta-gate (the plan's hard gate value).
META_MIN_TRADES = GATE_MIN_TRADES
#: Absolute floor on the filter probability, independent of the search: below
#: this the secondary model is saying "no better than a coin flip".
META_MIN_PROBABILITY = 0.5
#: Size of a trade that just clears the threshold, and of one at p = 1.
META_SIZE_FLOOR = 0.25
META_MAX_SIZE_MULTIPLIER = 1.0
#: Default barrier geometry (ATR-scaled, same defaults as `core.ml.labels`).
META_ATR_PERIOD = 14
META_ATR_MULTIPLE = 1.5
META_BARRIER_MIN_PCT = 0.004
META_BARRIER_MAX_PCT = 0.06
#: Threshold grid (upper tail only — this is a filter, not a direction bet).
META_THRESHOLD_GRID = tuple(np.round(np.arange(0.30, 0.951, 0.01), 4))

#: Meta-label classes (mirrors `core.ml.labels`).
META_NO_HIT = 0
META_PROFIT_HIT = 1
META_LOSS_HIT = 0
META_TIMEOUT = 2


# ── the primary signal ───────────────────────────────────────────────────

def primary_signal_from_rules(
    df: pd.DataFrame,
    long_conditions: list[str],
    short_conditions: list[str] | None = None,
) -> pd.Series:
    """``{-1, 0, +1}`` primary signal from rule strings (OR within each side).

    Uses the **shared** condition kernel
    (:func:`core.strategy.indicators.evaluate_condition`), so a meta-label is
    built on exactly the rule the live engine evaluates.  Both sides active at
    the same bar ⇒ 0 (the engine's ambiguity rule).  The signal at bar ``t``
    uses only that bar's (already closed) indicator values — the caller must not
    pass a frame whose indicators peek forward.

    A GA champion signature plugs in the same way: use
    ``StrategyConfig.entry_sides`` per bar, or simply pass its ``indicator_signal``
    series to :func:`build_meta_dataset`.
    """
    from core.strategy.indicators import evaluate_condition

    long_active = pd.Series(False, index=df.index)
    for cond in long_conditions or []:
        mask = evaluate_condition(df, cond)
        long_active |= mask.fillna(False).astype(bool)
    short_active = pd.Series(False, index=df.index)
    for cond in short_conditions or []:
        mask = evaluate_condition(df, cond)
        short_active |= mask.fillna(False).astype(bool)
    out = pd.Series(0.0, index=df.index)
    out[long_active & ~short_active] = 1.0
    out[short_active & ~long_active] = -1.0
    return out


def primary_forward_returns(
    close,
    primary_side,
    forward_periods: int = 24,
) -> pd.Series:
    """Fractional return of the primary trade, **signed** by its side.

    ``side_t · (close_{t+h} / close_t − 1)`` — the return the primary trade would
    have earned before costs, for the rows where the primary is non-zero.  Rows
    without a full forward window are ``NaN``.
    """
    c = pd.Series(close).astype(float)
    h = max(int(forward_periods), 1)
    fwd = c.shift(-h) / c - 1.0
    side = pd.Series(primary_side).astype(float).reindex(c.index).fillna(0.0)
    return (side * fwd).where(side != 0.0)


# ── meta labels ──────────────────────────────────────────────────────────

def _barrier_widths(df: pd.DataFrame, *, atr_period: int, atr_multiple: float,
                    min_pct: float, max_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """ATR-scaled barrier widths, from :mod:`core.ml.labels` when importable.

    The import is defensive on purpose: ``core/ml/labels.py`` belongs to phase
    P3 in this overhaul, and a meta-label must not turn into an ImportError if
    that module's helpers are reorganised.  The local fallback is the identical
    Wilder-ATR construction.
    """
    try:
        from core.ml.labels import barrier_widths
        up, lo = barrier_widths(df, atr_period=atr_period,
                                atr_multiple=atr_multiple,
                                min_pct=min_pct, max_pct=max_pct)
        return up.to_numpy(dtype=float), lo.to_numpy(dtype=float)
    except Exception:  # pragma: no cover - fallback path
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev = close.shift(1)
        tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                       axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / max(int(atr_period), 1), adjust=False,
                     min_periods=atr_period).mean()
        width = (float(atr_multiple) * atr / close).clip(lower=min_pct,
                                                         upper=max_pct)
        width = width.bfill().fillna(min_pct).to_numpy(dtype=float)
        return width, width


def profit_barrier_labels(
    df: pd.DataFrame,
    primary_side,
    *,
    forward_periods: int = 24,
    atr_period: int = META_ATR_PERIOD,
    atr_multiple: float = META_ATR_MULTIPLE,
    min_pct: float = META_BARRIER_MIN_PCT,
    max_pct: float = META_BARRIER_MAX_PCT,
) -> pd.Series:
    """1 if the primary trade touches its **profit** barrier first, 0 if the loss
    barrier is touched first, ``NA`` on timeout / no primary signal.

    Barriers are ATR-scaled (``atr_multiple × ATR/close``, clipped) and
    side-aware: for a long primary the profit barrier is ``entry·(1+w)`` and the
    loss barrier ``entry·(1−w)``; for a short primary they swap.  Only rows with
    a **full** forward window are labelled, so the last ``forward_periods`` rows
    are always ``NA`` (no truncated windows).
    """
    side = pd.Series(primary_side).astype(float).reindex(df.index).fillna(0.0)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    up_w, lo_w = _barrier_widths(df, atr_period=atr_period,
                                 atr_multiple=atr_multiple,
                                 min_pct=min_pct, max_pct=max_pct)
    n = len(df)
    horizon = max(int(forward_periods), 1)
    labels = np.full(n, np.nan)
    sides = side.to_numpy(dtype=float)
    for i in range(0, max(n - horizon, 0)):
        s = sides[i]
        if s == 0.0 or not np.isfinite(close[i]) or close[i] <= 0:
            continue
        if s > 0:
            profit = close[i] * (1.0 + up_w[i])
            loss = close[i] * (1.0 - lo_w[i])
        else:
            profit = close[i] * (1.0 - lo_w[i])
            loss = close[i] * (1.0 + up_w[i])
        label = np.nan
        for j in range(i + 1, min(i + horizon, n - 1) + 1):
            if s > 0:
                if high[j] >= profit:
                    label = float(META_PROFIT_HIT)
                    break
                if low[j] <= loss:
                    label = float(META_LOSS_HIT)
                    break
            else:
                if low[j] <= profit:
                    label = float(META_PROFIT_HIT)
                    break
                if high[j] >= loss:
                    label = float(META_LOSS_HIT)
                    break
        labels[i] = label
    return pd.Series(labels, index=df.index)


def meta_label_from_barrier(triple_labels, side) -> pd.Series:
    """Binary meta-label from a 3-class triple-barrier label series.

    ``create_triple_barrier_label_vol`` semantics: ``1`` = upper barrier first,
    ``0`` = lower barrier first, ``2`` = timeout.  A **long** primary profits on
    class 1, a **short** primary on class 0; timeouts (and rows without a primary
    signal) become ``NA`` — dropped, never coerced into a class.
    """
    lab = pd.Series(triple_labels).astype(float)
    s = pd.Series(side).astype(float).reindex(lab.index).fillna(0.0)
    out = pd.Series(np.nan, index=lab.index, dtype=float)
    long_ok = (s > 0) & lab.isin([0.0, 1.0])
    short_ok = (s < 0) & lab.isin([0.0, 1.0])
    out[long_ok] = (lab[long_ok] == 1.0).astype(float)
    out[short_ok] = (lab[short_ok] == 0.0).astype(float)
    return out


def build_meta_dataset(
    df: pd.DataFrame,
    primary_side,
    *,
    forward_periods: int = 24,
    feature_matrix: pd.DataFrame | None = None,
    label_series=None,
) -> dict:
    """Assemble ``(X, y_meta, trade_returns)`` for the secondary model.

    ``feature_matrix`` defaults to the canonical 39-column contract
    (:func:`core.ml.features.compute_features`, which requires the indicator
    columns — call ``compute_all(df, REQUIRED_INDICATORS)`` first).
    ``label_series`` lets a caller pass
    ``core.ml.labels.create_triple_barrier_label_vol``'s output instead of the
    built-in ATR barrier scan.
    """
    side = pd.Series(primary_side).astype(float).reindex(df.index).fillna(0.0)
    trade_returns = primary_forward_returns(df["close"], side, forward_periods)
    if label_series is not None:
        y = meta_label_from_barrier(label_series, side)
    else:
        y = profit_barrier_labels(df, side, forward_periods=forward_periods)
    if feature_matrix is None:
        from core.ml.features import compute_features
        feature_matrix = compute_features(df)
    X = feature_matrix.reindex(df.index)
    ok = (side != 0.0) & y.notna() & trade_returns.notna()
    return {
        "X": X.loc[ok], "y_meta": y.loc[ok].astype(float),
        "trade_returns": trade_returns.loc[ok].astype(float),
        "side": side.loc[ok], "n_primary": int((side != 0.0).sum()),
        "n_labelled": int(ok.sum()),
        "timeout_share": float(
            (side != 0.0).sum() - ok.sum()) / float((side != 0.0).sum())
        if int((side != 0.0).sum()) else 0.0,
    }


# ── cost-aware one-sided threshold search ────────────────────────────────

def meta_cost_aware_threshold(
    y_meta,
    p_meta,
    trade_returns,
    *,
    cost_pct: float,
    weights=None,
    grid=META_THRESHOLD_GRID,
    min_trades: int = 20,
) -> dict:
    """Highest-net-expectancy probability at which to **take** the primary trade.

    ``take = p_meta ≥ t`` and ``net = trade_return − cost`` (``trade_returns`` is
    already signed by the primary side).  One-sided by design: the meta-model
    filters, it never shorts the primary's signal.  A candidate must have
    ``≥ min_trades`` and a **positive** net expectancy to be selected; otherwise
    ``side`` is ``None`` and the caller must read that as "no tradeable
    threshold", never as an implied edge.
    """
    y = np.asarray(y_meta, dtype=float)
    p = np.asarray(p_meta, dtype=float)
    r = np.asarray(trade_returns, dtype=float)
    ok = np.isfinite(p) & np.isfinite(r) & np.isfinite(y)
    w = None if weights is None else np.asarray(weights, dtype=float)[ok]
    p, r, y = p[ok], r[ok], y[ok]
    n = len(p)
    base = float(y.mean()) if n else 0.5
    curve = []
    best = {"threshold": None, "expectancy": 0.0, "n_taken": 0, "coverage": 0.0}
    for t in grid:
        take = p >= float(t)
        cnt = int(take.sum())
        e = net_expectancy(r, take, +1, cost_pct, w)
        curve.append({"threshold": float(t), "expectancy": e, "n_taken": cnt})
        if cnt < int(min_trades) or e <= 0.0:
            continue
        if e > best["expectancy"]:
            best = {"threshold": float(t), "expectancy": e, "n_taken": cnt,
                    "coverage": cnt / n if n else 0.0}
    threshold = best["threshold"]
    if threshold is None:
        take = np.zeros(n, dtype=bool)
    else:
        take = p >= float(threshold)
    stats = net_trade_stats(r, take, +1, cost_pct, w)
    out = dict(best)
    out.update({
        "base_rate": base, "n": n, "curve": curve, "min_trades": int(min_trades),
        "stats": stats, "t_stat": stats["t_stat"], "psr": stats["psr"],
        "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
        "n_taken": int(take.sum()),
        "coverage": float(take.sum()) / n if n else 0.0,
        "selection": "one_sided_take_primary",
    })
    return out


def fold_min_trades(min_trades: int, n_rows: int) -> int:
    """Trade floor for a fold's calibration stream (scaled to its size)."""
    return int(min(int(min_trades), max(20, int(n_rows) // 10)))


# ── the nested OOS evaluation ────────────────────────────────────────────

def _proba_positive(model, X: pd.DataFrame) -> np.ndarray:
    if X is None or len(X) == 0:
        return np.array([])
    p = np.asarray(model.predict_proba(X), dtype=float)
    if p.ndim == 1:
        return p
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return p[:, classes.index(1)]
    return p[:, -1]


def resolve_model_factory(model_factory=None):
    """Return a ``(X, y, sample_weight) -> model`` callable.

    Accepts either that callable or the **builder** form
    (``core.ml.trainer.default_binary_factory``, which has to be *called* first
    to produce the factory).  The distinction is not cosmetic:
    ``core.ml.credibility.evaluate_model_oos`` binds
    ``model_factory = default_binary_factory`` on its default path and then
    calls ``model_factory(X, y, w)``, which raises
    ``TypeError: default_binary_factory() takes from 0 to 2 positional arguments
    but 3 were given`` — measured here on the first real meta evaluation (all
    five folds failed).  ``credibility.py`` is outside this phase's write scope,
    so the fix is reported to the Lead instead; this helper makes the meta path
    immune to the same mistake in either direction.
    """
    if model_factory is None:
        from core.ml.trainer import default_binary_factory as builder
        return builder()
    try:
        import inspect
        sig = inspect.signature(model_factory)
    except (TypeError, ValueError):  # builtins / C callables: assume usable
        return model_factory
    params = [p for p in sig.parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                            p.KEYWORD_ONLY)]
    if len(params) <= 2 and {"feature_names", "n_estimators"} & {p.name for p in params}:
        return model_factory()
    return model_factory


def evaluate_meta_oos(
    X: pd.DataFrame,
    y_meta: pd.Series,
    trade_returns: pd.Series,
    *,
    cost_pct: float,
    n_splits: int = 5,
    label_span: int = 24,
    embargo: int | None = None,
    model_factory=None,
    calibrate: str = "isotonic",
    min_train: int = 100,
    min_trades: int = META_MIN_TRADES,
    threshold_floor: float = META_MIN_PROBABILITY,
) -> dict:
    """Purged/embargoed K-fold evaluation of the **secondary** model.

    Same nested protocol as
    :func:`core.ml.credibility.evaluate_model_oos` (fold model → in-fold
    calibration stream → threshold selected inside the fold → applied to that
    fold's test rows), with the one-sided take-the-primary decision instead of
    the two-sided direction bet.  Returns a dict shaped so that
    :func:`core.ml.credibility.gate_from_evaluation` consumes it directly.
    """
    if model_factory is None:
        model_factory = resolve_model_factory()
    else:
        model_factory = resolve_model_factory(model_factory)

    y = pd.Series(y_meta).astype(float)
    r = pd.Series(trade_returns).astype(float)
    ok = y.notna() & r.notna()
    Xv, yv, rv = X.loc[ok], y.loc[ok], r.loc[ok]
    n = len(Xv)
    splits = purged_kfold_splits(n, n_splits, label_span=label_span,
                                 embargo=embargo)
    if not splits or n < min_train:
        return {"error": f"insufficient data for purged K-fold (n={n})",
                "n": n, "n_splits": len(splits)}

    p_parts, p_raw_parts, y_parts, r_parts, folds, used = [], [], [], [], [], []
    for f, (train_idx, test_idx, n_purged) in enumerate(splits):
        if len(train_idx) < min_train or len(test_idx) == 0:
            continue
        Xtr, ytr = Xv.iloc[train_idx], yv.iloc[train_idx]
        Xte, yte = Xv.iloc[test_idx], yv.iloc[test_idx]
        rte = rv.iloc[test_idx]
        cal_frac = 0.2 if calibrate != "none" else 0.0
        cal_cut = int(len(train_idx) * (1.0 - cal_frac))
        if calibrate == "none" or cal_cut < 60 or len(train_idx) - cal_cut < 20:
            cal_cut = len(train_idx)
        Xf, yf = Xtr.iloc[:cal_cut], ytr.iloc[:cal_cut]
        try:
            model = model_factory(Xf, yf, sample_uniqueness_weights(cal_cut, label_span))
        except Exception as e:  # a fold that cannot be fitted is skipped loudly
            folds.append({"fold": f, "error": f"{type(e).__name__}: {e}"})
            continue
        if model is None:
            continue
        Xcal, ycal = Xtr.iloc[cal_cut:], ytr.iloc[cal_cut:]
        rcal = rv.iloc[train_idx].iloc[cal_cut:]
        p_cal_raw = _proba_positive(model, Xcal)
        p_te_raw = _proba_positive(model, Xte)

        calibrator = ProbabilityCalibrator(calibrate)
        if calibrate != "none" and len(p_cal_raw) >= MIN_CAL_ROWS \
                and len(np.unique(ycal.values)) > 1:
            calibrator.fit(p_cal_raw, ycal.values)
        cal_fallback = not calibrator.fitted
        p_cal = calibrator.transform(p_cal_raw)
        p_te = calibrator.transform(p_te_raw)

        cal_thr = meta_cost_aware_threshold(
            ycal.values, p_cal, rcal.values, cost_pct=cost_pct,
            min_trades=fold_min_trades(min_trades, len(p_cal)))
        thr = cal_thr.get("threshold")
        if thr is not None:
            thr = max(float(thr), float(threshold_floor))
        take = (p_te >= float(thr)) if thr is not None else np.zeros(len(p_te), bool)
        fold_stats = net_trade_stats(rte.values, take, +1, cost_pct)

        p_parts.append(p_te)
        p_raw_parts.append(p_te_raw)
        y_parts.append(yte.values)
        r_parts.append(rte.values)
        used.append(splits[f])
        folds.append({
            "fold": f, "train": int(len(train_idx)), "test": int(len(test_idx)),
            "purged": int(n_purged), "n_fit": int(cal_cut),
            "n_cal": int(len(train_idx) - cal_cut),
            "n_cal_trades": int(cal_thr.get("n_taken", 0)),
            "cal_threshold": thr, "cal_expectancy": float(cal_thr.get("expectancy", 0.0)),
            "cal_min_trades": fold_min_trades(min_trades, len(p_cal)),
            "calibrated": bool(calibrator.fitted), "cal_fallback": bool(cal_fallback),
            "oos_n_trades": int(fold_stats["n"]),
            "oos_expectancy": float(fold_stats["mean"]),
            "oos_t_stat": float(fold_stats["t_stat"]),
            "auc": float(binary_metrics(yte.values, p_te_raw)["auc"]),
        })

    if not p_parts:
        return {"error": "no fold produced a model", "n": n, "folds": folds}

    p_oos = np.concatenate(p_parts)
    p_raw_oos = np.concatenate(p_raw_parts)
    y_oos = np.concatenate(y_parts)
    r_oos = np.concatenate(r_parts)
    metrics = binary_metrics(y_oos, p_oos)
    metrics_raw = binary_metrics(y_oos, p_raw_oos)
    # The gate reads the calibration-independent AUC (a calibrator cannot
    # inflate it) — same rule as `credibility.evaluate_model_oos`.
    metrics["auc"] = float(metrics_raw["auc"])
    metrics["brier_uncalibrated"] = float(metrics_raw["brier"])
    for fold in folds:
        if "auc" in fold:
            fold["auc_uncalibrated"] = fold["auc"]

    take_oos = np.zeros(len(p_oos), dtype=bool)
    offset = 0
    for fold, p_te in zip([f for f in folds if "cal_threshold" in f], p_parts):
        m = len(p_te)
        if fold["cal_threshold"] is not None:
            take_oos[offset:offset + m] = p_te >= float(fold["cal_threshold"])
        offset += m
    stats_oos = net_trade_stats(r_oos, take_oos, +1, cost_pct)
    pooled = _pooled_threshold(p_oos, y_oos, r_oos, cost_pct, min_trades)
    nested = _fold_threshold_summary(folds)
    return {
        "n": n, "n_oos": int(len(y_oos)), "n_splits": len(folds), "folds": folds,
        "metrics": metrics, "metrics_uncalibrated": metrics_raw,
        # `thresholds_oos` is the nested (deployable) summary; `thresholds` is
        # the pooled search on the reported rows and is diagnostic only — the
        # same split as `credibility.evaluate_model_oos`.
        "thresholds_oos": nested, "thresholds": pooled,
        "threshold": nested.get("threshold"),
        "pooled_threshold": pooled.get("threshold"),
        "net_expectancy_pooled": float(pooled.get("expectancy", 0.0)),
        "base_rate": float(yv.mean()), "cost_pct": float(cost_pct),
        "net_expectancy_oos": float(stats_oos["mean"]),
        "n_trades_oos": int(stats_oos["n"]),
        "t_stat_oos": float(stats_oos["t_stat"]),
        "psr_oos": float(stats_oos["psr"]),
        "ci_low_oos": float(stats_oos["ci_low"]),
        "ci_high_oos": float(stats_oos["ci_high"]),
        "p_oos": p_oos, "p_raw_oos": p_raw_oos, "y_oos": y_oos,
        "fwd_oos": r_oos, "take_oos": take_oos,
        "index_oos": (Xv.index.to_numpy()[np.concatenate([s[1] for s in used])]
                      if used else None),
    }


def _fold_threshold_summary(folds: list[dict]) -> dict:
    """The single deployable threshold, from the nested per-fold selections.

    Mirrors ``core.ml.credibility._fold_threshold_summary``: the median of the
    thresholds each fold's calibration stream selected, or ``None`` when no fold
    found a candidate with positive net expectancy.  ``None`` means "take
    nothing on this evidence", never "use the pooled optimum".
    """
    values = [float(f["cal_threshold"]) for f in folds
              if f.get("cal_threshold") is not None]
    out = {
        "selection": "one_sided_take_primary",
        "n_folds": len(folds), "n_folds_with_candidates": len(values),
        "threshold": None, "n_trades_oos": int(sum(f.get("oos_n_trades", 0)
                                                   for f in folds)),
    }
    if not values:
        out["reason"] = ("no fold's calibration stream produced a threshold with "
                         "positive net expectancy")
        return out
    out["threshold"] = float(np.median(values))
    out["min_threshold"] = float(np.min(values))
    out["max_threshold"] = float(np.max(values))
    out["thresholds"] = values
    return out


def _pooled_threshold(p_oos, y_oos, r_oos, cost_pct, min_trades) -> dict:
    """Diagnostic only: the search on the rows it reports (optimistic by design).

    Reported next to ``net_expectancy_oos`` for the same reason
    ``credibility.evaluate_model_oos`` reports its pooled numbers — so the two
    can be contrasted, never so the pooled one can be gated on.
    """
    out = meta_cost_aware_threshold(y_oos, p_oos, r_oos, cost_pct=cost_pct,
                                    min_trades=min_trades)
    out["selection"] = "pooled_optimistic"
    return out


def meta_gate(eval_result: dict, **kwargs) -> dict:
    """:func:`core.ml.credibility.gate_from_evaluation` on a meta evaluation."""
    return gate_from_evaluation(eval_result, **kwargs)


# ── the filter + size multiplier ─────────────────────────────────────────

@dataclass(frozen=True)
class MetaDecision:
    """The only two things a meta-model is allowed to produce."""

    take: bool
    size_multiplier: float
    probability: float
    threshold: float
    side: float
    reason: str

    def apply(self, primary_side: float) -> float:
        """Gated position: ``primary_side × size_multiplier`` or ``0``.

        The **sign is the primary's** — the meta-model can only scale it down to
        zero, never invert it.
        """
        if not self.take:
            return 0.0
        return float(np.sign(self.side)) * float(self.size_multiplier)


class MetaLabeler:
    """Secondary model: filter + size multiplier over a primary signal.

    Lifecycle::

        lab = MetaLabeler()
        res = lab.evaluate(X, y_meta, trade_returns, cost_pct=cost)   # gate
        if lab.enabled:            # `res` passed `credibility.credibility_gate`
            lab.fit(X, y_meta, trade_returns, cost_pct=cost)
            dec = lab.decide(p_meta=0.72, side=+1)

    :attr:`enabled` is ``False`` until an evaluation has *passed* the gate, so a
    meta-model that does not beat its own gate is inert — it returns
    ``take=False`` for every probability rather than a small position.
    """

    def __init__(
        self,
        *,
        model_factory=None,
        calibrate: str = "isotonic",
        size_floor: float = META_SIZE_FLOOR,
        max_size_multiplier: float = META_MAX_SIZE_MULTIPLIER,
    ):
        self.model_factory = model_factory
        self.calibrate = calibrate
        self.size_floor = float(size_floor)
        self.max_size_multiplier = float(max_size_multiplier)
        self.evaluation: dict | None = None
        self.gate: dict | None = None
        self.model = None
        self.calibrator: ProbabilityCalibrator | None = None
        self.threshold: float | None = None
        self.cost_pct: float | None = None

    # -- gate -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.gate and self.gate.get("allowed"))

    def evaluate(self, X, y_meta, trade_returns, *, cost_pct: float,
                 **kwargs) -> dict:
        """Run the nested OOS protocol and the hard gate; store both."""
        res = evaluate_meta_oos(X, y_meta, trade_returns, cost_pct=cost_pct,
                                model_factory=self.model_factory,
                                calibrate=self.calibrate, **kwargs)
        self.evaluation = res
        self.gate = meta_gate(res)
        self.cost_pct = float(cost_pct)
        if not self.enabled:
            # Never leave a stale threshold/model behind a refused gate.
            self.threshold = None
            self.model = None
            self.calibrator = None
        else:
            self.threshold = float(res.get("threshold") or META_MIN_PROBABILITY)
        return res

    # -- fit ------------------------------------------------------------
    def fit(self, X, y_meta, trade_returns, *, cost_pct: float) -> "MetaLabeler":
        """Fit the deployable model on the whole sample (call ``evaluate`` first).

        The threshold is the one the *outer* protocol selected, not a fresh
        in-sample optimum, so the deployed filter carries the OOS number the
        gate approved.
        """
        if self.gate is None:
            raise RuntimeError("MetaLabeler.fit requires evaluate() first — "
                               "fitting an ungated meta-model is not allowed")
        if not self.enabled:
            return self
        from core.ml.trainer import default_binary_factory
        factory = resolve_model_factory(self.model_factory or default_binary_factory)
        y = pd.Series(y_meta).astype(float)
        r = pd.Series(trade_returns).astype(float)
        ok = y.notna() & r.notna()
        Xv, yv = X.loc[ok], y.loc[ok]
        weights = sample_uniqueness_weights(len(Xv), 24)
        self.model = factory(Xv, yv, weights)
        self.calibrator = ProbabilityCalibrator(self.calibrate)
        p = _proba_positive(self.model, Xv)
        if len(p) >= MIN_CAL_ROWS and len(np.unique(yv.values)) > 1:
            self.calibrator.fit(p, yv.values)
        return self

    # -- inference -------------------------------------------------------
    def probability(self, X) -> np.ndarray:
        if self.model is None:
            return np.array([])
        p = _proba_positive(self.model, X)
        if self.calibrator is not None and self.calibrator.fitted:
            p = self.calibrator.transform(p)
        return p

    def decide(self, p_meta: float, side: float = 1.0,
               threshold: float | None = None) -> MetaDecision:
        """Filter + size for one probability. Inert while the gate refuses."""
        thr = float(threshold if threshold is not None
                    else (self.threshold if self.threshold is not None
                          else META_MIN_PROBABILITY))
        p = float(p_meta)
        if not self.enabled:
            return MetaDecision(False, 0.0, p, thr, float(side),
                                "meta-model not enabled (gate refused or not run)")
        if not np.isfinite(p) or p < thr:
            return MetaDecision(False, 0.0, p, thr, float(side),
                                f"p={p:.4f} < threshold {thr:.4f}")
        span = max(1.0 - thr, 1e-9)
        frac = float(min(max((p - thr) / span, 0.0), 1.0))
        size = self.size_floor + (self.max_size_multiplier - self.size_floor) * frac
        return MetaDecision(True, float(min(size, self.max_size_multiplier)), p,
                            thr, float(side), "take")

    def filter_series(self, primary_side, X, *, threshold: float | None = None) -> pd.Series:
        """Gated position series for a whole primary signal (never flips a side)."""
        side = pd.Series(primary_side).astype(float)
        p = self.probability(X)
        if len(p) == 0:
            return pd.Series(0.0, index=side.index)
        out = np.zeros(len(side), dtype=float)
        for i, (s, pi) in enumerate(zip(side.to_numpy(dtype=float), p)):
            out[i] = self.decide(pi, s, threshold=threshold).apply(s)
        return pd.Series(out, index=side.index)


def default_meta_cost_pct(config=None, *, symbol: str = "BTCUSDT") -> float:
    """Round-trip cost (%) for the primary trade, from the sim cost model.

    The point of this helper is that the meta path cannot charge a cheaper cost
    than the fills pay, so ``config=None`` **resolves the active configuration**
    (``app.config.Config.load()``, the same object the executor uses) instead of
    falling through to :func:`core.ml.credibility.cost_pct_for`'s hard-coded
    fallbacks.  Measured before this fix: ``default_meta_cost_pct(None,
    "ETHUSDT")`` returned **0.14 %** (the fallback defaults: 0.04 % fee + 0.01 %
    half spread + 2 bp slippage) while the sim cost a fill actually pays is
    **0.26 %**; ``docs/core-algorithms/12``'s "same source as fills" claim was
    therefore only true when a config object was passed.  Now it is true by
    construction.  If even the active config cannot be loaded the cost falls back
    to the documented defaults (never to a fabricated cheaper number).
    """
    if config is None:
        try:
            from app.config import Config
            config = Config.load()
        except Exception:
            config = None
    return float(cost_pct_for(config, symbol=symbol, order_type="market"))


def breakeven_hit_rate(cost_pct: float, reward_risk: float) -> float:
    """Hit rate a barrier trade needs just to pay the cost.

    With reward ``R`` and risk ``1`` the expectancy is
    ``p·R − (1−p) − c`` ⇒ ``p* = (1 + c) / (1 + R)``.  Reported next to the
    measured meta base rate so a base rate above ``p*`` is not mistaken for an
    edge.
    """
    c = float(cost_pct) / 100.0
    rr = max(float(reward_risk), 1e-9)
    return float(min(max((1.0 + c) / (1.0 + rr), 0.0), 1.0))


def meta_summary(res: dict) -> dict:
    """Flat research record: base rate, majority, AUC, net expectancy, gate."""
    m = res.get("metrics", {})
    gate = meta_gate(res)
    return {
        "n": int(res.get("n", 0)), "n_oos": int(res.get("n_oos", 0)),
        "base_rate": m.get("base_rate"), "majority_accuracy": m.get("majority_accuracy"),
        "accuracy": m.get("accuracy"), "auc": m.get("auc"),
        "brier": m.get("brier"), "net_expectancy_oos": res.get("net_expectancy_oos"),
        "n_trades_oos": res.get("n_trades_oos"), "t_stat_oos": res.get("t_stat_oos"),
        "psr_oos": res.get("psr_oos"), "cost_pct": res.get("cost_pct"),
        "threshold": res.get("threshold"), "gate_allowed": gate.get("allowed"),
        "gate_reason": gate.get("reason"),
    }


__all__ = [
    "META_LABELING_ENABLED", "META_MIN_TRADES", "META_MIN_PROBABILITY",
    "META_SIZE_FLOOR", "META_MAX_SIZE_MULTIPLIER", "META_ATR_PERIOD",
    "META_ATR_MULTIPLE", "META_THRESHOLD_GRID",
    "primary_signal_from_rules", "primary_forward_returns", "meta_label_from_barrier",
    "profit_barrier_labels", "build_meta_dataset", "meta_cost_aware_threshold",
    "fold_min_trades", "evaluate_meta_oos", "meta_gate", "MetaDecision",
    "MetaLabeler", "default_meta_cost_pct", "breakeven_hit_rate", "meta_summary",
]
