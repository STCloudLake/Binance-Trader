"""The ML credibility gate (Phase P2 items 2, 3 and 7).

Nothing in the pipeline checked out-of-sample skill before a model was used.
Measured on real cached data the live model scores **OOS AUC 0.396–0.447** and
0.409–0.472 accuracy against a 0.543–0.667 majority-class baseline, i.e. it is a
measured negative contribution — yet ``ml.enabled`` was free to be true.

This module is the single decision point:

* :func:`cost_pct_for` resolves the round-trip cost from the **sim cost model**
  (``app.config.sim_cost_quote``), so the gate can never gate on a cheaper cost
  than the one the fills actually pay.
* :func:`cost_aware_threshold` picks the probability threshold that maximises
  **net-of-cost expectancy** (never accuracy) and reports the breakeven
  threshold.
* :func:`evaluate_model_oos` runs purged K-fold (``embargo = label horizon``)
  with sample-uniqueness weights, fits **per-fold calibrators** on each fold's
  own calibration stream and selects the decision threshold **inside the fold**
  so the reported/gated net expectancy is an outer number.
* :func:`credibility_gate` returns a status dict — ``allowed = OOS AUC > 0.55
  AND net expectancy > 0 AND trades >= min_trades AND t > 2 AND PSR >= 0.95``
  (audit F3: the significance floors are a conjunction and missing t/PSR is a
  refusal, not a skipped check).  :func:`probabilistic_sharpe` is Prado's PSR
  **with** the skew/kurtosis correction (re-audit finding 4 — it used to be
  ``Φ(mean/se)``, which made the PSR floor an alias of ``t >= 1.645`` while the
  docstrings claimed it read higher moments).
* :func:`ml_accuracy_neutral_abstention` is the corrected diagnostic for
  ``engine.py`` (see the TODO handed to the Lead).

Selection-optimism fix (audit P2 #1)
------------------------------------
The first P2 revision fitted one pooled calibrator **and** selected the
threshold on the same pooled OOS rows it reported and gated: the per-fold
"calibration stream" was dead code (``calibrator.n_fit == n_oos == 600`` while
``sum(n_cal) == 477``), in-fit ECE was 0.0000 against 0.0784 on a genuine
holdout, and a half-split experiment flipped the OOS net expectancy to
**−0.075 % (BTC) / −0.342 % (ETH)**.  Every probability the gate now consumes is
produced by a calibrator fitted only on that fold's held-out stream, and every
threshold is selected on that same in-fold stream (nested), then applied to the
fold's test rows.  :func:`gate_from_evaluation` therefore gates on
``net_expectancy_oos`` (the outer number).  The old pooled-search numbers are
still reported under ``thresholds`` as ``*_optimistic`` for comparison, and
:func:`evaluate_model_oos` still runs that search so the two can be contrasted.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.ml.calibration import ProbabilityCalibrator
from core.ml.evaluation import (
    binary_metrics, purged_kfold_splits, sample_uniqueness_weights,
)

#: Plan §二 P2.1 / §三.2 — the hard gate.
GATE_AUC_MIN = 0.55
GATE_MIN_NET_EXPECTANCY = 0.0
#: Audit P2 #3 — 20 trades cannot support a verdict: BTC's "best" candidate had
#: 27 trades / t = 1.28 / 95 % CI [−0.19 %, +0.91 %].
GATE_MIN_TRADES = 100
#: Audit P2 #3 — a t-stat (or equivalently a probabilistic Sharpe) floor.
GATE_MIN_T_STAT = 2.0
#: Probability that the true Sharpe is > 0 required by the PSR alternative.
#: Audit F3: this is an **additional** floor (``t > GATE_MIN_T_STAT`` **and**
#: ``PSR >= GATE_MIN_PSR``), not a substitute — as an ``or`` it is equivalent to
#: ``t >= 1.645`` under normality and silently weakens the 2.0 t floor.
GATE_MIN_PSR = 0.95

#: ml.ml settings fallbacks (config `ml:` block wins when present).
DEFAULT_TAKER_FEE_PCT = 0.04
DEFAULT_HALF_SPREAD_PCT = 0.01
DEFAULT_SLIPPAGE_BPS = 2.0

#: Smallest calibration stream that may fit a per-fold calibrator.
MIN_CAL_ROWS = 60


# ── cost: single source of truth (audit P2 #2) ───────────────────────────

def _sim_fee_inputs(config, symbol: str) -> tuple[float, float, float, bool]:
    """``(market_fee_pct, limit_fee_pct, half_spread_pct, bnb)`` from the sim model.

    Uses ``app.config``'s own helpers so the numbers are the ones the fills pay.
    """
    defaults = (DEFAULT_TAKER_FEE_PCT, DEFAULT_TAKER_FEE_PCT,
                DEFAULT_HALF_SPREAD_PCT, DEFAULT_SLIPPAGE_BPS, False)
    if config is None:
        return defaults
    try:
        from app.config import (_sim_settings_from_config, sim_fee_pct,
                                sim_spread_pct)
    except Exception:  # pragma: no cover - import cycle guard
        return defaults
    try:
        settings = _sim_settings_from_config(config)
        market = float(sim_fee_pct(settings, "market"))
        limit = float(sim_fee_pct(settings, "limit"))
        # `sim.spread_pct` is the per-side spread the aggressor pays
        # (`sim_cost_quote`: `edge_pct = spread_pct / 2 + slippage_bps / 100`,
        # where `spread_pct/2` is the half spread).  Following the code rather
        # than its comment, this is exactly `labels.round_trip_cost_pct`'s
        # `half_spread_pct`, so `cost_pct_for` and the fill model agree.
        half = float(sim_spread_pct(settings, symbol)) / 2.0
        slip = float(settings.get("slippage_bps", DEFAULT_SLIPPAGE_BPS))
        bnb = bool(settings.get("use_bnb_discount", False))
        # `ml.use_bnb_discount: null` follows the sim setting; an explicit
        # true/false is a research override (audit P2 #8 keeps the key live).
        override = getattr(config, "ml_use_bnb_discount", None)
        if override is not None:
            bnb = bool(override)
    except Exception:  # pragma: no cover - defensive
        return defaults
    return market, limit, half, slip, bnb


def round_trip_cost_pct_from_quote(
    config,
    *,
    symbol: str = "BTCUSDT",
    order_type: str = "market",
    taker_fee_pct: float | None = None,
    half_spread_pct: float | None = None,
    slippage_bps: float | None = None,
) -> float:
    """Round-trip cost (%) computed with ``sim_cost_quote`` semantics.

    Components (percent of notional, one round trip):

    ``market``
        ``2 × (fee + half_spread + slippage_bps / 100)`` — the aggressor pays the
        fee, half the quoted spread and the slippage on each side.
    ``limit``
        ``2 × maker_fee`` — a limit order fills at its own price and pays no
        spread/slippage.

    The fee tier is resolved through ``sim_fee_pct`` (so a persisted tier
    override and the BNB discount are honoured) unless the caller pins one.
    """
    symbol = str(symbol or "").upper()
    is_limit = str(order_type or "market").strip().lower() == "limit"
    market_fee, limit_fee, half, sim_slip, bnb = _sim_fee_inputs(config, symbol)

    if taker_fee_pct is not None:
        fee = float(taker_fee_pct)
    else:
        fee = limit_fee if is_limit else market_fee
        # `ml.taker_fee_pct` may pin the fee explicitly (research override).
        pinned = getattr(config, "ml_taker_fee_pct", None) if config is not None else None
        if pinned is not None:
            fee = float(pinned)
    if half_spread_pct is not None:
        half = float(half_spread_pct)
    else:
        pinned_half = getattr(config, "ml_half_spread_pct", None) if config is not None else None
        if pinned_half is not None:
            half = float(pinned_half)
        elif is_limit:
            half = 0.0
    if slippage_bps is not None:
        slip = float(slippage_bps)
    else:
        pinned_slip = getattr(config, "ml_slippage_bps", None) if config is not None else None
        slip = float(pinned_slip) if pinned_slip is not None else float(sim_slip)

    if bnb:
        fee *= 0.75
    if is_limit:
        # A resting limit order is not crossed: no spread, no slippage.
        per_side = float(fee)
    else:
        per_side = float(fee) + float(half) + float(slip) / 100.0
    return float(2.0 * per_side)


def cost_pct_for(
    config=None,
    *,
    symbol: str = "BTCUSDT",
    order_type: str = "market",
    taker_fee_pct: float | None = None,
    half_spread_pct: float | None = None,
    slippage_bps: float | None = None,
) -> float:
    """Round-trip cost (%) resolved from config with the **sim** cost semantics.

    Audit P2 #2: this used to read ``backtest_taker_fee_pct`` (0.04 %), which made
    :func:`cost_pct_for` return 0.13 %/0.14 % while ``sim_cost_quote`` charged
    0.28 %/0.30 % round trip (VIP0 taker 0.10 %), i.e. the gate was fed a cost
    2.0× cheaper than the fills.  The reported ETH "+0.058 %" net expectancy is
    **−0.062 %** at the true cost.  The sim model is now the source of truth;
    the ``ml.*`` keys only act as explicit research overrides.
    """
    return round_trip_cost_pct_from_quote(
        config, symbol=symbol, order_type=order_type,
        taker_fee_pct=taker_fee_pct, half_spread_pct=half_spread_pct,
        slippage_bps=slippage_bps)


# ── significance (audit P2 #3) ───────────────────────────────────────────

def _norm_cdf(z: float) -> float:
    """Standard normal CDF via ``erf`` (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def net_trade_stats(
    fwd_returns,
    take_mask,
    side,
    cost_pct: float,
    weights=None,
) -> dict:
    """Mean/SD/SE/t/CI of the net return of every *taken* trade.

    ``net = side × forward_return − cost``; ``t = mean / (sd / sqrt(n))`` and the
    95 % CI is ``mean ± 1.96 × SE``.  A mean alone is not evidence — 27 trades at
    +0.36 % with SE 0.28 % is t = 1.28, i.e. indistinguishable from zero.
    Returns zeros with ``n = 0`` when nothing is taken.

    ``psr`` is the **skew/kurtosis-corrected** Prado PSR of those same net-trade
    returns (re-audit finding 4): the net series is handed to
    :func:`probabilistic_sharpe` so its higher moments are actually read, which is
    what makes the gate's ``PSR >= 0.95`` floor a real second condition rather than
    a restatement of ``t > 2``.  The ``t_stat``/``se`` fields stay the plain
    normal-theory numbers — only the PSR is tail-aware.
    """
    r = np.asarray(fwd_returns, dtype=float)
    mask = np.asarray(take_mask, dtype=bool) & np.isfinite(r)
    n = int(mask.sum())
    empty = {"n": 0, "mean": 0.0, "sd": 0.0, "se": 0.0, "t_stat": 0.0,
             "ci_low": 0.0, "ci_high": 0.0, "psr": 0.0}
    if n == 0:
        return empty
    net = r[mask] * float(side) - float(cost_pct) / 100.0
    if weights is not None:
        w = np.asarray(weights, dtype=float)[mask]
        if w.sum() > 0:
            mean = float(np.average(net, weights=w))
        else:
            mean = float(net.mean())
    else:
        mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else 0.0
    t = mean / se if se > 0 else 0.0
    return {
        "n": n, "mean": mean, "sd": sd, "se": se, "t_stat": float(t),
        "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
        "psr": probabilistic_sharpe(n, mean, sd, returns=net),
    }


def probabilistic_sharpe(n: int, mean: float, sd: float,
                         benchmark: float = 0.0,
                         returns=None) -> float:
    """``P(true mean > benchmark)`` for per-trade returns (PSR).

    Prado's Probabilistic Sharpe Ratio *with* the higher-moment correction
    (Bailey & López de Prado 2012, eq. 5–6; the same formula
    ``core.ml.calibration`` cites): the sampling standard error of the mean
    return is wider than ``sd/√n`` whenever the returns are skewed or fat-tailed,

        ``SE_adj = sd · √( (1 − γ₃·SR + (γ₄ − 1)/4 · SR²) / (n − 1) )``

    with ``SR = (mean − benchmark)/sd`` and ``γ₃``/``γ₄`` the sample skewness and
    (non-excess) kurtosis of the **net-trade returns** supplied as ``returns``.
    ``PSR = Φ((mean − benchmark)/SE_adj)``; for a normal sample (``γ₃ = 0``,
    ``γ₄ = 3``) the bracket is exactly 1 and the result *is* ``Φ(√n·SR)`` — the
    old normal approximation, unchanged.

    Re-audit finding 4 fixed the justification, not just the docstring: the
    docstrings used to *claim* this function read skew and kurtosis while the code
    was ``Φ(mean/se)``, which made the gate's ``AND`` exactly ``t > 2`` and the
    ``PSR >= 0.95`` floor dead weight.  It genuinely reads them now, so the two
    floors are a real conjunction: measured on a fat-left-tailed series with
    ``t = 2.0`` the corrected PSR is **0.937 < 0.95** (refused), while the normal
    approximation would report 0.977 (allowed).  See
    ``tests/test_reaudit_fixes.py::test_psr_floor_is_stricter_than_the_t_floor_for_fat_tails``.

    ``returns`` is optional for backwards compatibility and for callers that hold
    only the summary statistics (``core.strategy.pairs``); with ``returns=None``,
    fewer than 4 usable values, or a non-finite moment, the correction is skipped
    (equivalent to assuming normality) rather than guessing a tail shape.  ``0.0``
    for ``n < 2`` or ``sd <= 0`` — the unchanged degenerate cases.
    """
    n = int(n)
    sd = float(sd)
    if n < 2 or sd <= 0:
        return 0.0
    diff = float(mean) - float(benchmark)
    se = sd / math.sqrt(n)
    if returns is not None:
        try:
            r = np.asarray(returns, dtype=float)
            r = r[np.isfinite(r)]
            if r.size > 3:
                sd_r = float(r.std(ddof=1))
                if sd_r > 0.0:
                    sr = diff / sd_r
                    g3 = float(((r - r.mean()) ** 3).mean()) / sd_r ** 3
                    g4 = float(((r - r.mean()) ** 4).mean()) / sd_r ** 4
                    bracket = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
                    if math.isfinite(bracket) and bracket > 0.0:
                        se = sd_r * math.sqrt(bracket / (r.size - 1))
        except Exception:
            se = sd / math.sqrt(n)  # never let a diagnostic break the gate
    if se <= 0.0:
        return 0.0
    return float(_norm_cdf(diff / se))


def _signed_net_stats(fwd_returns, take_mask, sides, cost_pct: float) -> dict:
    """Net-trade stats for a mask that may mix long (+1) and short (−1) trades."""
    r = np.asarray(fwd_returns, dtype=float)
    take = np.asarray(take_mask, dtype=bool) & np.isfinite(r)
    s = np.asarray(sides, dtype=float)
    n = int(take.sum())
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "se": 0.0, "t_stat": 0.0,
                "ci_low": 0.0, "ci_high": 0.0, "psr": 0.0}
    net = np.sign(s[take]) * r[take] - float(cost_pct) / 100.0
    mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else 0.0
    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "t_stat": float(mean / se) if se > 0 else 0.0,
        "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
        "psr": probabilistic_sharpe(n, mean, sd, returns=net),
    }


# ── cost-aware threshold selection ───────────────────────────────────────

def net_expectancy(
    fwd_returns,
    take_mask,
    side,
    cost_pct: float,
    weights=None,
) -> float:
    """Average net return (fraction) per taken trade, net of round-trip cost.

    A trade is *taken* when ``take_mask`` is true; ``side`` is ``+1`` (long) or
    ``-1`` (short).  Unweighted mean unless ``weights`` (sample uniqueness) is
    supplied.  Returns ``0.0`` when nothing is taken — no data is not an edge.
    """
    r = np.asarray(fwd_returns, dtype=float)
    mask = np.asarray(take_mask, dtype=bool) & np.isfinite(r)
    if not mask.any():
        return 0.0
    net = r[mask] * float(side) - float(cost_pct) / 100.0
    if weights is None:
        return float(net.mean())
    w = np.asarray(weights, dtype=float)[mask]
    return float(np.average(net, weights=w)) if w.sum() > 0 else float(net.mean())


def cost_aware_threshold(
    y_up,
    p_up,
    fwd_returns,
    *,
    cost_pct: float,
    weights=None,
    grid: np.ndarray | None = None,
    min_trades: int = 20,
) -> dict:
    """Threshold maximising net expectancy on **this** sample; reports breakeven.

    Positive class = "price up" (label 1).  Both thresholds live in **``p_up``
    space**: a long is taken when ``p_up >= threshold_up`` and a short when
    ``p_up <= threshold_down`` (``threshold_down = 1 − t`` for the mirrored
    upper-tail grid point ``t``).  The two are independent decision bands — the
    winning threshold is written to *its own* side only, so a winning short band
    is never stored in the long field (audit P2 #4).

    Every threshold candidate also has to clear ``GATE_MIN_NET_EXPECTANCY`` on
    the calibration stream: a threshold does not become "selected" merely because
    it is the least-bad negative one.

    Returns ``{"threshold_up", "threshold_down", "threshold", "side",
    "breakeven_up", "breakeven_down", "expectancy_long", "expectancy_short",
    "expectancy", "coverage", "n_taken", "curve"}``.  ``threshold`` is the upper
    grid point ``t`` of the selected side — the long band when the long side
    wins, or ``1 − threshold_down`` when it is the short side;
    ``expectancy`` is the best of the two sides.  ``side`` is ``None`` when no
    candidate reached ``min_trades`` with positive net expectancy — the caller
    must treat that as "no tradeable threshold on this sample", never as an
    implied edge.

    NOTE (audit P2 #1): whoever calls this owns the selection-optimism question.
    :func:`evaluate_model_oos` calls it **inside** each fold on the fold's
    calibration stream, never on the rows it reports.
    """
    y = np.asarray(y_up, dtype=float)
    p = np.asarray(p_up, dtype=float)
    r = np.asarray(fwd_returns, dtype=float)
    ok = np.isfinite(p) & np.isfinite(r) & np.isfinite(y)
    w = None if weights is None else np.asarray(weights, dtype=float)[ok]
    if w is not None and len(w) != int(ok.sum()):
        w = None
    p, r, y = p[ok], r[ok], y[ok]
    n = len(p)
    # Grid of upper-tail thresholds; the lower tail is its mirror (1 − t).
    if grid is None:
        grid = np.round(np.arange(0.30, 0.951, 0.01), 4)

    base = float(y.mean()) if n else 0.5
    curve = []
    best = {"expectancy": 0.0, "side": None, "threshold": None, "coverage": 0.0,
            "n_taken": 0, "expectancy_long": 0.0, "expectancy_short": 0.0,
            "threshold_up": base, "threshold_down": 1.0 - base}
    for t in grid:
        take_up = p >= t
        take_dn = p <= (1.0 - t)
        e_up = net_expectancy(r, take_up, +1, cost_pct, w)
        e_dn = net_expectancy(r, take_dn, -1, cost_pct, w)
        n_up, n_dn = int(take_up.sum()), int(take_dn.sum())
        curve.append({"threshold": float(t), "expectancy_long": e_up,
                      "expectancy_short": e_dn, "n_long": n_up, "n_short": n_dn})
        for e, side, cnt in ((e_up, "long", n_up), (e_dn, "short", n_dn)):
            if cnt < int(min_trades):
                continue
            # A candidate must both have enough trades and a *positive* net
            # expectancy: "least bad" is not an edge (audit P2 #1/#3).
            if e <= 0.0:
                continue
            if e > best["expectancy"]:
                # The two sides are independent decision bands (audit P2 #4):
                # a long trades when `p >= threshold_up`, a short when
                # `p <= threshold_down`.  The pre-fix code overwrote both fields
                # with the *long* threshold when the short side won, so the
                # persisted short band was `1 - t` (0.70 for a winning t = 0.30)
                # instead of `t` — the exact bug that made the live predictor
                # rebuild a meaningless band.
                best = {
                    "expectancy": e, "side": side, "threshold": float(t),
                    "coverage": cnt / n if n else 0.0, "n_taken": cnt,
                    "expectancy_long": e_up, "expectancy_short": e_dn,
                    "threshold_up": float(t) if side == "long" else best["threshold_up"],
                    "threshold_down": float(t) if side == "short" else best["threshold_down"],
                }
    # Breakeven: the threshold at which expectancy crosses zero (most selective
    # threshold that is still non-negative), for the chosen side.
    side = best.get("side")
    breakeven_up = breakeven_down = None
    if side is not None:
        key = "expectancy_long" if side == "long" else "expectancy_short"
        for row in curve:
            if row[key] > 0 and row["n_long" if side == "long" else "n_short"] >= int(min_trades):
                if side == "long":
                    breakeven_up = row["threshold"]
                else:
                    breakeven_down = 1.0 - row["threshold"]
        if side == "long" and breakeven_up is None:
            breakeven_up = best.get("threshold")
        if side == "short" and breakeven_down is None:
            breakeven_down = best.get("threshold")

    best["breakeven_up"] = breakeven_up
    best["breakeven_down"] = breakeven_down
    best["base_rate"] = base
    best["n"] = n
    best["curve"] = curve
    best["min_trades"] = int(min_trades)
    # Per-side trade statistics on the selected side (t-stat / CI / PSR).
    if side == "long":
        take = p >= float(best["threshold_up"])
        signed_side = +1
        best["threshold_down"] = 1.0 - float(best["threshold_up"])
    elif side == "short":
        take = p <= float(best["threshold_down"])
        signed_side = -1
        best["threshold_up"] = 1.0 - float(best["threshold_down"])
    else:
        take = np.zeros(n, dtype=bool)
        signed_side = +1
    best["n_taken"] = int(take.sum())
    best["coverage"] = float(take.sum()) / n if n else 0.0
    stat = net_trade_stats(r, take, signed_side, cost_pct, w)
    best["stats"] = stat
    best["t_stat"] = stat["t_stat"]
    best["psr"] = stat["psr"]
    best["ci_low"] = stat["ci_low"]
    best["ci_high"] = stat["ci_high"]
    return best


# ── OOS evaluation + gate ────────────────────────────────────────────────

def evaluate_model_oos(
    X: pd.DataFrame,
    y_up: pd.Series,
    fwd_returns: pd.Series,
    *,
    n_splits: int = 5,
    label_span: int = 4,
    embargo: int | None = None,
    cost_pct: float,
    model_factory=None,
    calibrate: str = "isotonic",
    min_train: int = 100,
    min_trades: int = GATE_MIN_TRADES,
    calibrate_method: str | None = None,
) -> dict:
    """Purged/embargoed K-fold evaluation with **no selection optimism**.

    Protocol (audit P2 #1 — each step is out-of-sample for the thing it feeds):

    1. **Fold models** — per fold, fit on the purged + embargoed training rows
       with sample-**uniqueness** weights.  The last 20 % of that block (the
       *calibration stream*) is held out of the fit.
    2. **Per-fold calibrator** — fitted on the fold's own calibration stream
       (``n_cal`` rows the fold model never saw) and used to transform that
       fold's test probabilities.  A stream smaller than
       :data:`MIN_CAL_ROWS`/single-class degrades to the identity and is flagged
       (``cal_fallback``), never silently calibrated on test rows.
    3. **Threshold selected inside the fold** — ``cost_aware_threshold`` runs on
       the calibration stream, and the winning threshold is applied to the fold's
       test rows.  ``net_expectancy_oos`` is the mean net return of those pooled
       outer trades: the number :func:`credibility_gate` consumes.
    4. **Reported comparison** — the old *pooled* search (calibrator + threshold
       fitted and measured on the same rows) is still computed and reported under
       ``thresholds`` with ``selection`` = ``"pooled_optimistic"``; it is what the
       audit reproduced, and ``net_expectancy_oos`` is the honest counterpart.

    Returns the pooled OOS metrics, the pooled calibrated arrays, the nested OOS
    decision statistics, the optimistic threshold table and per-fold diagnostics.
    """
    if model_factory is None:
        from core.ml.trainer import default_binary_factory
        # `default_binary_factory(...)` is a *builder*: it returns the
        # `fit(X, y, sample_weight)` factory. Assigning the builder itself made the
        # default path call `default_binary_factory(X, y, w)` with weights as
        # `n_estimators` and then fail with a TypeError in every fold
        # (found by the phase-P4 meta-labelling work).
        model_factory = default_binary_factory()
    if calibrate_method:
        calibrate = calibrate_method

    y = pd.Series(y_up).astype(float)
    r = pd.Series(fwd_returns).astype(float)
    idx = X.index
    ok = y.notna() & r.notna()
    Xv = X.loc[ok]
    yv = y.loc[ok]
    rv = r.loc[ok]
    n = len(Xv)
    splits = purged_kfold_splits(n, n_splits, label_span=label_span, embargo=embargo)
    if not splits or n < min_train:
        return {"error": f"insufficient data for purged K-fold (n={n})",
                "n": n, "n_splits": len(splits)}

    p_parts, p_raw_parts, y_parts, r_parts, folds, used_splits = [], [], [], [], [], []
    for f, (train_idx, test_idx, n_purged) in enumerate(splits):
        if len(train_idx) < min_train or len(test_idx) == 0:
            continue
        Xtr, ytr = Xv.iloc[train_idx], yv.iloc[train_idx]
        Xte, yte, rte = Xv.iloc[test_idx], yv.iloc[test_idx], rv.iloc[test_idx]
        # Hold the tail of the training block out of the fit: those rows supply
        # the fold's calibration stream (out-of-sample for this fold's model).
        cal_frac = 0.2 if calibrate != "none" else 0.0
        cal_cut = int(len(train_idx) * (1.0 - cal_frac))
        if calibrate == "none" or cal_cut < 60 or len(train_idx) - cal_cut < 20:
            cal_cut = len(train_idx)
        Xf, yf = Xtr.iloc[:cal_cut], ytr.iloc[:cal_cut]
        w_fit = sample_uniqueness_weights(cal_cut, label_span)
        model = model_factory(Xf, yf, w_fit)
        if model is None:
            continue
        Xcal, ycal, rcal = Xtr.iloc[cal_cut:], ytr.iloc[cal_cut:], rv.iloc[train_idx].iloc[cal_cut:]
        p_cal_raw = _proba_up(model, Xcal)
        p_te_raw = _proba_up(model, Xte)

        # (2) per-fold calibrator, fitted on the fold's own calibration stream.
        calibrator = ProbabilityCalibrator(calibrate)
        cal_fallback = False
        if calibrate != "none" and len(p_cal_raw) >= MIN_CAL_ROWS \
                and len(np.unique(ycal.values)) > 1:
            calibrator.fit(p_cal_raw, ycal.values)
        if not calibrator.fitted:
            cal_fallback = True
        p_cal = calibrator.transform(p_cal_raw)
        p_te = calibrator.transform(p_te_raw)

        # (3) threshold selected INSIDE the fold (on the calibration stream),
        # then applied to the fold's test rows.
        cal_thr = cost_aware_threshold(
            ycal.values, p_cal, rcal.values, cost_pct=cost_pct,
            min_trades=fold_min_trades(min_trades, len(p_cal)))
        fold_side = cal_thr.get("side")
        if fold_side == "long":
            take = p_te >= float(cal_thr["threshold_up"])
            side = +1
        elif fold_side == "short":
            take = p_te <= float(cal_thr["threshold_down"])
            side = -1
        else:
            take = np.zeros(len(p_te), dtype=bool)
            side = +1
        fold_stats = net_trade_stats(rte.values, take, side, cost_pct)

        p_parts.append(p_te)
        p_raw_parts.append(p_te_raw)
        y_parts.append(yte.values)
        r_parts.append(rte.values)
        used_splits.append(splits[f])
        folds.append({
            "fold": f, "train": int(len(train_idx)), "test": int(len(test_idx)),
            "purged": int(n_purged), "n_fit": int(cal_cut),
            "n_cal": int(len(train_idx) - cal_cut),
            "auc_uncalibrated": float(binary_metrics(yte.values, p_te_raw)["auc"]),
            "calibrated": bool(calibrator.fitted),
            "cal_fallback": bool(cal_fallback),
            "calibrator_n_fit": int(calibrator.n_fit),
            "cal_threshold_side": fold_side,
            "cal_threshold": cal_thr.get("threshold"),
            "cal_threshold_up": cal_thr.get("threshold_up"),
            "cal_threshold_down": cal_thr.get("threshold_down"),
            "cal_expectancy": float(cal_thr.get("expectancy", 0.0)),
            "cal_n_taken": int(cal_thr.get("n_taken", 0)),
            "cal_no_positive_candidate": bool(fold_side is None),
            "cal_min_trades": int(fold_min_trades(min_trades, len(p_cal))),
            "oos_n_trades": int(fold_stats["n"]),
            "oos_expectancy": float(fold_stats["mean"]),
            "oos_t_stat": float(fold_stats["t_stat"]),
        })

    if not p_parts:
        return {"error": "no fold produced a model", "n": n}

    p_oos = np.concatenate(p_parts)
    p_raw_oos = np.concatenate(p_raw_parts)
    y_oos = np.concatenate(y_parts)
    r_oos = np.concatenate(r_parts)

    metrics = binary_metrics(y_oos, p_oos)
    metrics_raw = binary_metrics(y_oos, p_raw_oos)
    # The gate reads the calibration-independent AUC: it cannot be inflated by
    # any calibrator, so it is the one metric the honest protocol cannot move.
    metrics["auc"] = float(metrics_raw["auc"])
    metrics["brier_uncalibrated"] = float(metrics_raw["brier"])
    metrics["log_loss_uncalibrated"] = float(metrics_raw["log_loss"])
    for fold in folds:
        fold["auc"] = fold["auc_uncalibrated"]

    # ── the honest outer number: each fold's own threshold on its test rows ──
    take_oos = np.zeros(len(p_oos), dtype=bool)
    side_oos = np.ones(len(p_oos), dtype=float)
    offset = 0
    for fold, p_te in zip(folds, p_parts):
        m = len(p_te)
        if fold["cal_threshold_side"] == "long":
            take_oos[offset:offset + m] = p_te >= float(fold["cal_threshold_up"])
        elif fold["cal_threshold_side"] == "short":
            take_oos[offset:offset + m] = p_te <= float(fold["cal_threshold_down"])
            side_oos[offset:offset + m] = -1.0
        offset += m
    stats_oos = _signed_net_stats(r_oos, take_oos, side_oos, cost_pct)

    # (4) the old pooled search, reported for comparison only.
    pooled_calibrator = ProbabilityCalibrator(calibrate)
    if calibrate != "none" and len(p_raw_oos) >= MIN_CAL_ROWS:
        pooled_calibrator.fit(p_raw_oos, y_oos)
    p_pooled = pooled_calibrator.transform(p_raw_oos)
    thresholds = cost_aware_threshold(
        y_oos, p_pooled, r_oos, cost_pct=cost_pct, min_trades=min_trades)
    thresholds["selection"] = "pooled_optimistic"
    thresholds["n_fit"] = int(pooled_calibrator.n_fit)
    thresholds["note"] = ("calibrator and threshold fitted on the reported rows "
                          "(audit P2 #1); use net_expectancy_oos for the gate")

    base_rate = float(yv.mean())
    oos_thresholds = _fold_threshold_summary(folds)
    return {
        "n": n, "n_oos": int(len(y_oos)), "n_splits": len(folds),
        "folds": folds, "metrics": metrics, "metrics_uncalibrated": metrics_raw,
        "thresholds": thresholds, "thresholds_oos": oos_thresholds,
        "calibrator": pooled_calibrator.to_dict(),
        "base_rate": base_rate, "cost_pct": float(cost_pct),
        "net_expectancy_oos": float(stats_oos["mean"]),
        "n_trades_oos": int(stats_oos["n"]),
        "t_stat_oos": float(stats_oos["t_stat"]),
        "psr_oos": float(stats_oos["psr"]),
        "ci_low_oos": float(stats_oos["ci_low"]),
        "ci_high_oos": float(stats_oos["ci_high"]),
        "net_expectancy_pooled": float(thresholds.get("expectancy", 0.0)),
        "min_trades": int(min_trades),
        "p_oos": p_oos, "p_oos_calibrated": p_oos, "p_raw_oos": p_raw_oos,
        "y_oos": y_oos, "fwd_oos": r_oos,
        "take_oos": take_oos, "decisions_oos": take_oos,
        "side_oos": side_oos,
        "index_oos": np.concatenate([idx.to_numpy()[s[1]] for s in used_splits])
        if hasattr(idx, "to_numpy") else None,
    }


def fold_min_trades(min_trades: int, n_cal: int) -> int:
    """Trade floor for a *calibration stream* (scaled to its size).

    The OOS floor (:data:`GATE_MIN_TRADES`) applies to the pooled outer sample;
    a single fold's stream is ~1/5 of it, so requiring 100 calibration trades
    would make every fold abstain and hide a real edge.  The stream floor is
    ``min(min_trades, max(50, n_cal // 10))`` — never below 50, because a
    threshold selected on 25 trades is exactly the "27 trades, t = 1.28"
    mistake the gate now refuses at the OOS level.
    """
    return int(min(int(min_trades), max(50, int(n_cal) // 10)))


def _fold_threshold_summary(folds: list[dict]) -> dict:
    """The single threshold a live consumer can use, from the nested folds."""
    sides = [f["cal_threshold_side"] for f in folds if f.get("cal_threshold_side")]
    out = {
        "selection": "nested_per_fold",
        "n_folds": len(folds),
        "n_folds_with_candidates": len(sides),
        "side_votes": {s: sides.count(s) for s in ("long", "short")},
        "threshold_up": None, "threshold_down": None, "side": None,
        "n_trades_oos": int(sum(f["oos_n_trades"] for f in folds)),
    }
    if not sides:
        out["reason"] = ("no fold's calibration stream produced a candidate "
                         "threshold with positive net expectancy")
        return out
    side = "long" if sides.count("long") >= sides.count("short") else "short"
    key = "cal_threshold_up" if side == "long" else "cal_threshold_down"
    values = [float(f[key]) for f in folds
              if f.get("cal_threshold_side") == side and f.get(key) is not None]
    if not values:
        return out
    out["side"] = side
    median = float(np.median(values))
    if side == "long":
        out["threshold_up"] = median
        out["threshold_down"] = 1.0 - median
    else:
        out["threshold_down"] = median
        out["threshold_up"] = 1.0 - median
    return out


def _proba_up(model, X: pd.DataFrame) -> np.ndarray:
    if X is None or len(X) == 0:
        return np.array([])
    p = np.asarray(model.predict_proba(X), dtype=float)
    if p.ndim == 1:
        return p
    if p.shape[1] == 1:
        return p[:, 0]
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return p[:, classes.index(1)]
    return p[:, -1]


def credibility_gate(
    metrics: dict,
    net_expectancy_value: float,
    *,
    auc_min: float = GATE_AUC_MIN,
    min_net_expectancy: float = GATE_MIN_NET_EXPECTANCY,
    n_oos: int | None = None,
    min_oos: int = 100,
    min_trades: int = GATE_MIN_TRADES,
    min_t_stat: float = GATE_MIN_T_STAT,
    t_stat: float | None = None,
    psr: float | None = None,
    n_trades: int | None = None,
    min_psr: float = GATE_MIN_PSR,
) -> dict:
    """The hard gate: ``enabled`` may only be true for a passing model.

    ``allowed`` requires **all** of:

      1. ``OOS AUC > auc_min`` (0.55 by default),
      2. ``net expectancy > min_net_expectancy`` (0.0 = costs are at least paid),
      3. enough OOS rows to mean anything (``n_oos >= min_oos``),
      4. at least ``min_trades`` (100) **outer** trades — audit P2 #3: the
         previous gate accepted 27 trades with t = 1.28 and a 95 % CI spanning
         [−0.19 %, +0.91 %],
      5. significance, **both** floors: ``t > min_t_stat`` (2.0) **and**
         ``PSR >= min_psr`` (0.95), with **no** evidence being a refusal — a
         missing ``t_stat`` or ``psr`` is reported as ``no significance
         evidence`` rather than skipped (audit F3).  Both numbers are reported in
         the reason either way.

    **Why AND** (audit F3, re-audit finding 4): the module previously *documented*
    the two floors as a conjunction but *implemented* ``or``, and there the 2.0 t
    floor is dead — ``PSR >= 0.95`` is the one-sided normal probability, so under
    normality it is exactly ``t >= 1.645`` (measured: ``t=1.65, PSR=0.9505`` was
    allowed, while the audited floor is 2.0).

    The two floors are genuinely non-redundant **because
    :func:`probabilistic_sharpe` now really does read skew and kurtosis** —
    Prado's ``Φ((mean − benchmark)/SE_adj)`` with
    ``SE_adj ∝ √(1 − γ₃·SR + (γ₄−1)/4·SR²)``.  A fat left tail or a fat right tail
    widens that standard error, so the PSR at ``t = 2`` can sit below 0.95
    (measured: **0.937** on a fat-left-tailed sample at ``t = 2.0``, where the
    normal approximation reports 0.977).  This is the *fixed* version of a claim
    that used to be false: the function was ``Φ(mean/se)`` with no moment terms,
    which made the ``PSR`` floor nothing but ``t >= 1.645`` and the conjunction
    exactly ``t > 2``.  AND keeps both the audited t floor and the now-real
    tail-aware floor; it can only refuse more than the old ``or``, never less.

    Today's candidates therefore fail on every count, which is the documented
    outcome: no symbol measured on real cached data has reached 100 outer trades
    with a significant positive net expectancy, so ``ml.enabled`` stays ``false``.

    Returns a status dict the caller can log verbatim: ``{"allowed", "reason",
    "enabled", "auc", "net_expectancy", "thresholds", ...}``.  ``enabled`` is
    the value to persist — it is ``False`` whenever ``allowed`` is ``False``.
    """
    auc = float(metrics.get("auc", 0.5))
    n = int(metrics.get("n", 0) if n_oos is None else n_oos)
    exp = float(net_expectancy_value)
    if t_stat is None:
        t_stat = metrics.get("t_stat")
    if psr is None:
        psr = metrics.get("psr")
    if n_trades is None:
        n_trades = metrics.get("n_trades")
    t_val = float(t_stat) if t_stat is not None else 0.0
    psr_val = float(psr) if psr is not None else 0.0
    n_tr = int(n_trades) if n_trades is not None else 0
    reasons = []
    if n < int(min_oos):
        reasons.append(f"insufficient OOS rows ({n} < {min_oos})")
    if not (auc > float(auc_min)):
        reasons.append(f"OOS AUC {auc:.4f} <= {float(auc_min):.2f}")
    if not (exp > float(min_net_expectancy)):
        reasons.append(
            f"net expectancy {exp * 100:.4f}% <= {float(min_net_expectancy) * 100:.4f}% "
            f"(after {float(metrics.get('cost_pct', 0.0)):.4f}% round-trip cost)"
            if "cost_pct" in metrics else
            f"net expectancy {exp * 100:.4f}% <= {float(min_net_expectancy) * 100:.4f}%")
    if n_trades is not None and n_tr < int(min_trades):
        reasons.append(f"too few trades ({n_tr} < {min_trades})")
    # ── significance (audit F3, two defects in one check) ──────────────────
    # 1. *Absence of evidence is not evidence*: `t_stat is None` used to skip the
    #    check entirely, so a payload without a t-stat (the legacy
    #    `gate_from_evaluation` branch, and any hand-built metrics dict) passed the
    #    gate with `allowed=True, reason="pass"` — contradicting this docstring.
    #    Missing t **or** PSR is now a refusal that names what is missing.
    # 2. The documented floors (`t > min_t_stat` **and** `PSR >= min_psr`) are now
    #    an **AND**.  As implemented before, the OR collapsed them: PSR is the
    #    one-sided normal probability, so `PSR >= 0.95` is exactly `t >= 1.645`
    #    under normality, i.e. the 2.0 t floor was silently 18 % weaker than the
    #    audited value (measured: t=1.65, PSR=0.9505 → allowed).  The floors are
    #    NOT redundant *now*: `probabilistic_sharpe` applies Prado's
    #    skew/kurtosis correction (re-audit finding 4), so a fat tail at t=2 puts
    #    the PSR below 0.95 (measured 0.937) where the old `Φ(mean/se)` reported
    #    0.977 — the second floor is a real condition, which is what this
    #    conjunction always claimed.  AND is strictly stricter than OR: it can
    #    only refuse more.
    if t_stat is None or psr is None:
        missing = [name for name, value in (("t_stat", t_stat), ("psr", psr))
                   if value is None]
        reasons.append(
            f"no significance evidence ({', '.join(missing)} missing; the gate "
            f"requires t > {float(min_t_stat):.2f} AND PSR >= {float(min_psr):.2f})")
    elif not (t_val > float(min_t_stat) and psr_val >= float(min_psr)):
        reasons.append(
            f"not significant (requires t > {float(min_t_stat):.2f} AND "
            f"PSR >= {float(min_psr):.2f}; got t={t_val:.2f}, PSR={psr_val:.3f})")
    allowed = not reasons
    return {
        "allowed": allowed,
        "enabled": bool(allowed),
        "reason": "pass" if allowed else "; ".join(reasons),
        "auc": auc,
        "auc_min": float(auc_min),
        "net_expectancy": exp,
        "net_expectancy_min": float(min_net_expectancy),
        "n_oos": n,
        "n_trades": n_tr if n_trades is not None else None,
        "min_trades": int(min_trades),
        "t_stat": t_val if t_stat is not None else None,
        "min_t_stat": float(min_t_stat),
        "psr": psr_val if psr is not None else None,
        "majority_accuracy": metrics.get("majority_accuracy"),
        "accuracy": metrics.get("accuracy"),
        "brier": metrics.get("brier"),
        "log_loss": metrics.get("log_loss"),
        "base_rate": metrics.get("base_rate"),
        "cost_pct": metrics.get("cost_pct"),
        "thresholds": metrics.get("thresholds"),
    }


def gate_from_evaluation(eval_result: dict, *, auc_min: float = GATE_AUC_MIN,
                         min_trades: int = GATE_MIN_TRADES,
                         min_t_stat: float = GATE_MIN_T_STAT,
                         min_psr: float = GATE_MIN_PSR,
                         min_net_expectancy: float = GATE_MIN_NET_EXPECTANCY,
                         min_oos: int = 100) -> dict:
    """:func:`credibility_gate` fed straight from :func:`evaluate_model_oos`.

    Gated on the **outer** numbers: ``net_expectancy_oos`` / ``n_trades_oos`` /
    ``t_stat_oos`` (per-fold thresholds applied to each fold's test rows), not on
    the pooled search that produced the old optimistic figure.

    The legacy branch (no ``net_expectancy_oos``) reads its numbers out of
    ``thresholds``; it can therefore hand :func:`credibility_gate` a ``None``
    t-stat / PSR, which since audit F3 is a **refusal** ("no significance
    evidence") and never a pass — a model cannot be enabled without outer stats.
    """
    if "metrics" not in eval_result:
        return {
            "allowed": False, "enabled": False,
            "reason": eval_result.get("error", "no evaluation result"),
            "auc": 0.5, "net_expectancy": 0.0, "n_oos": int(eval_result.get("n", 0)),
        }
    metrics = dict(eval_result["metrics"])
    metrics["cost_pct"] = eval_result.get("cost_pct")
    metrics["thresholds"] = eval_result.get("thresholds_oos") or eval_result.get("thresholds")
    if eval_result.get("net_expectancy_oos") is not None:
        exp = float(eval_result.get("net_expectancy_oos") or 0.0)
        n_trades = eval_result.get("n_trades_oos")
        t = eval_result.get("t_stat_oos")
        psr = eval_result.get("psr_oos")
    else:  # legacy payload: no nested numbers available
        exp = float((eval_result.get("thresholds") or {}).get("expectancy", 0.0))
        n_trades = (eval_result.get("thresholds") or {}).get("n_taken")
        t = (eval_result.get("thresholds") or {}).get("t_stat")
        psr = (eval_result.get("thresholds") or {}).get("psr")
    return credibility_gate(
        metrics, exp, auc_min=auc_min, n_oos=int(eval_result.get("n_oos", 0)),
        min_trades=min_trades, min_t_stat=min_t_stat, n_trades=n_trades,
        t_stat=t, psr=psr, min_psr=min_psr,
        min_net_expectancy=min_net_expectancy, min_oos=min_oos)


# ── corrected diagnostic (engine.py is out of scope) ─────────────────────

def ml_accuracy_neutral_abstention(
    predictions,
    realised_returns,
    *,
    threshold: float = 0.005,
) -> dict:
    """Corrected ``ml_accuracy_pct``: neutral = abstention, not a 0.5 vote.

    ``core.backtest.engine`` (out of this phase's write scope) computes
    ``ml_accuracy_pct`` at ``engine.py:1074-1082`` (and the band at
    ``engine.py:1233``) by scoring ``conf >= 0.5`` as bullish and counting the
    0.38–0.62 neutral band *as if it were a directional call*, which is
    bullish-biased because 23.8–34.3 % of predictions land in that band.

    Correct behaviour:
      * neutral predictions are abstentions — excluded from accuracy and
        reported separately as ``coverage``,
      * the label's own threshold is applied to the realised move.

    TODO(phase-P2, handed to the Lead): ``core/backtest/engine.py`` should call
    this helper instead of its inline loop —
    ``core.ml.credibility.ml_accuracy_neutral_abstention(conf_list, ret_list,
    threshold=th)`` — and expose ``coverage_pct`` next to ``ml_accuracy_pct``.
    ``engine.py`` is owned by another agent, so this was deliberately *not*
    edited.

    ``predictions`` may be ``(confidence, value)`` pairs or plain confidences;
    values at exactly 0.5 count as neutral.
    """
    confs, rets = [], []
    for item in predictions:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            confs.append(float(item[0]))
            rets.append(float(item[1]))
        else:
            confs.append(float(item))
    if not rets:
        rets = [float(r) for r in realised_returns]
    conf = np.asarray(confs, dtype=float)
    ret = np.asarray(rets, dtype=float)
    n_total = len(conf)
    neutral = np.isclose(conf, 0.5, atol=1e-9)
    moved = np.abs(ret) >= float(threshold)
    scored = (~neutral) & moved & np.isfinite(ret) & np.isfinite(conf)
    correct = ((ret[scored] >= threshold) & (conf[scored] > 0.5)) | \
              ((ret[scored] <= -threshold) & (conf[scored] < 0.5))
    n_scored = int(scored.sum())
    return {
        "n_predictions": int(n_total),
        "n_neutral": int(neutral.sum()),
        "n_scored": n_scored,
        "accuracy_pct": round(float(correct.sum()) / n_scored * 100.0, 1)
        if n_scored else 0.0,
        "coverage_pct": round(n_scored / n_total * 100.0, 1) if n_total else 0.0,
        "neutral_pct": round(float(neutral.sum()) / n_total * 100.0, 1) if n_total else 0.0,
    }


__all__ = [
    "GATE_AUC_MIN", "GATE_MIN_NET_EXPECTANCY", "GATE_MIN_TRADES",
    "GATE_MIN_T_STAT", "GATE_MIN_PSR", "MIN_CAL_ROWS",
    "cost_pct_for", "round_trip_cost_pct_from_quote",
    "net_expectancy", "net_trade_stats", "probabilistic_sharpe",
    "cost_aware_threshold", "evaluate_model_oos", "credibility_gate",
    "gate_from_evaluation", "ml_accuracy_neutral_abstention",
    "fold_min_trades",
]
# NOTE (re-audit finding 7): ``signed_score`` used to be listed here but is not
# defined in this module — it lives in :mod:`core.ml.calibration` — so
# ``from core.ml.credibility import *`` raised ``AttributeError``.  Every entry
# above is a name that really resolves here; the star-import is pinned by
# ``tests/test_reaudit_fixes.py::test_star_import_of_credibility_resolves``.
