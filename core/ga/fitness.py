"""Fitness evaluation — runs backtests to score strategy chromosomes.

Scoring contract (P1, GA credibility)
-------------------------------------
The fitness of a genome is a *risk-adjusted, trade-count-aware, benchmark-relative*
score.  The pieces that used to be missing are the reason the GA could not
optimise: profit factor was unbounded (a 5-trade all-winner genome scored 490
while a 200-trade PF-2.0 genome scored 7.3), Sharpe and max drawdown were
hardcoded to 0 in every batch path, and the buy & hold return of the same window
was never subtracted (pure market drift scored as alpha).

    fitness = base (win-rate / PF / ROC / long-short balance)
              + weight_alpha * (DSR-deflated Sharpe * min(1, trades/30) - max_dd)
              - trade-count, loss, overtrading and complexity penalties

with

    pf  = gross_win / (gross_loss + mean_win)      # shrunk: bounded even at 0 losses
    pf_term = min(pf, PF_TERM_CAP)                 # capped
    pf_term *= min(1, trades / PF_TRADE_FLOOR)     # scaled by evidence

and the alpha term subtracting the equal-weighted buy & hold return of the same
symbols and window, so beta is not scored as alpha.
"""

import time
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from core.ga.genome import chromosome_to_strategy
from core.strategy.loader import StrategyLoader

# ── Scoring constants (single source of truth) ──────────────────────────
#: Synthetic loss used to shrink the profit factor: ``pf = gross_win/(gross_loss+mean_win)``.
#: A genome with a single lucky winner therefore caps near 2.0 instead of 100.
PF_SHRINK = True
#: Hard ceiling on the profit-factor term (was effectively 500 with weight 5).
PF_TERM_CAP = 10.0
#: Trade count at which the profit-factor term reaches full weight.
PF_TRADE_FLOOR = 50
#: Trade count at which the Sharpe term reaches full weight.
SHARPE_TRADE_FLOOR = 30
#: Below this many trades a genome is flagged (and cannot be published).
MIN_TRADES_GATE = 30
#: Weight of ``DSR_sharpe * min(1, trades/30) - max_dd_pct`` in the fitness.
ALPHA_WEIGHT = 1.0
#: Number of return observations below which Sharpe/DSR are not estimated at all.
MIN_OBSERVATIONS = 20
#: Annualisation factor used to convert a per-period Sharpe into a per-year one.
DSR_PERIODS_PER_YEAR = 365
#: Legacy default weights (kept for backward-compatible callers).
DEFAULT_WEIGHTS = {"wr": 0.15, "pf": 5.0, "roc": 50, "bal": 10.0}


def isolated_eval_kwargs() -> dict:
    """Engine flags for GA evaluation: per-genome positions AND per-genome cash.

    ``per_strategy_isolation`` namespaces position keys; ``per_genome_ledger``
    adds one position-slot budget and one cash/equity sub-ledger per genome,
    which is what makes a 20-genome chunk evaluate 20 independent strategies.
    The second flag is probed so an older engine still works (and so the
    hybrid/legacy parity contract, which passes neither, is untouched).
    """
    import inspect
    from core.backtest.engine import BacktestEngine

    kwargs = {"per_strategy_isolation": True}
    try:
        params = inspect.signature(BacktestEngine.run_with_exit_evaluation).parameters
        if "per_genome_ledger" in params:
            kwargs["per_genome_ledger"] = True
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pass
    return kwargs


def _finite(value, default: float = 0.0, ndigits: int | None = None) -> float:
    """``float(value)`` or *default* — never NaN/inf (fitness must stay ordered)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return round(number, ndigits) if ndigits is not None else number


def profit_factor_shrunk(gross_win: float, gross_loss: float,
                         mean_win: float) -> float:
    """Profit factor with a synthetic average loss in the denominator.

    ``gross_loss == 0`` used to be scored as PF 100 (×5 weight ⇒ up to 500
    points, more than every legitimate term combined).  Adding one average win
    as a synthetic loss bounds the term: N winners with no loser score ≈ N,
    and the ``min(1, trades/50)`` scaling in :func:`score_stats` keeps a
    5-trade genome from outranking a 200-trade one.
    """
    gw = max(_finite(gross_win), 0.0)
    gl = max(_finite(gross_loss), 0.0)
    mw = max(_finite(mean_win), 0.0)
    if gw <= 0:
        return 0.1
    denom = gl + (mw if mw > 0 else gw)
    if denom <= 0:
        return 0.1
    return gw / denom



def evaluate_chromosome(
    chromosome: dict,
    symbols: list[str],
    date_start: str,
    date_end: str,
    engine,
    loader: StrategyLoader,
    ga_loader: StrategyLoader | None = None,
    initial_balance: float = 10000.0,
    n_trials: int = 1,
    use_live_spread: bool = False,
) -> dict:
    """Evaluate a single chromosome via backtest.

    Uses *ga_loader* (isolated temp dir) to save/load strategy files
    so GA never touches the main strategies/ directory.

    Scoring goes through the same :func:`score_stats` used by the batch paths,
    so the train fitness of a champion and the fitness printed during evolution
    are one formula (they used to be two different ones).

    Returns a dict with fitness components.
    """
    save_loader = ga_loader or loader
    try:
        config = chromosome_to_strategy(chromosome)
        if config.ml_config:
            config.ml_config.enabled = False

        # Pass StrategyConfig directly — no file I/O needed
        result = engine.run_with_exit_evaluation(
            strategies=[config],  # StrategyConfig object, not file name
            symbols=symbols,
            date_start=date_start,
            date_end=date_end,
            initial_balance=initial_balance,
            mode="full",
            simulate_ai_weights=False,
            ml_engine="lightgbm",
            use_live_spread=use_live_spread,
            **isolated_eval_kwargs(),
        )

        if "error" in result:
            return {"fitness": -999, "error": result["error"]}

        metrics = result.get("metrics", {})
        per_eq = (result.get("per_strategy_equity") or {}).get(config.name, {})
        equity_curve = per_eq.get("equity_curve") or result.get("equity_curve") or []
        trades = [t for t in result.get("trades", [])
                  if t.get("strategy") == config.name]
        if not trades:
            trades = result.get("trades", [])

        stats = stats_from_trades(trades, equity_curve, initial_balance)
        stats["buy_hold_pct"] = metrics.get("buy_hold_pct")
        stats["max_dd"] = stats["max_dd_pct"]
        if not stats["max_dd"]:
            stats["max_dd"] = abs(_finite(metrics.get("max_drawdown_pct", 0)))
        score = score_stats(stats, chromosome, n_trials=n_trials)
        score["sharpe"] = round(_finite(score.get("sharpe")), 4)
        score["win_rate"] = round(_finite(stats["win_rate"]), 2)
        score["profit_factor"] = round(_finite(stats["profit_factor"]), 4)
        score["max_dd"] = round(_finite(score["max_dd"]), 2)
        score["total_return"] = round(_finite(stats["total_return_pct"]), 2)
        score["trade_count"] = int(stats["trades"])
        score["strategy_name"] = config.name
        score["dsr"] = round(_finite(score["deflated_sharpe"]), 4)
        score["raw_profit_factor"] = round(_finite(stats["raw_profit_factor"]), 4)
        score["buy_hold_pct"] = (round(_finite(stats["buy_hold_pct"]), 4)
                                 if stats["buy_hold_pct"] is not None else None)
        score["alpha_pct"] = round(_finite(stats["alpha_pct"]), 4)
        score["observations"] = int(stats["observations"])
        score["spread_sources"] = metrics.get("spread_sources", {})
        return score

    except Exception as e:
        logger.debug(f"Fitness eval failed: {e}")
        return {"fitness": -999, "error": str(e)}


def complexity_penalty(chromosome: dict) -> float:
    """Penalize overparameterized strategies.

    More conditions + more indicators + more params = higher risk of overfitting.
    Returns a penalty value to subtract from fitness.
    """
    structural = chromosome.get("structural", [])
    continuous = chromosome.get("continuous", [])

    # Count conditions
    n_conditions = sum(len(g.conditions) for g in structural)

    # Count enabled indicators — prefer BooleanGene, fallback to parsing continuous genes
    indicator_genes = chromosome.get("indicator_genes", [])
    if indicator_genes:
        n_indicators = sum(1 for g in indicator_genes if g.value)
    else:
        # Backward compat: parse from continuous gene names
        indicators_used = set()
        for g in continuous:
            name = g.name.split("_")[0]  # "rsi_period" -> "rsi"
            if name not in ("ml",):
                indicators_used.add(name)
        n_indicators = len(indicators_used)

    penalty = 0.0
    penalty += n_conditions * 0.8        # each condition adds overfit risk
    penalty += n_indicators * 1.2        # each indicator type
    penalty += len(continuous) * 0.3     # each tunable parameter
    return penalty


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    observation_periods: int = 365,
    variance_sharpe: float = 1.0,
    sharpe_is_annualized: bool = True,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """Deflated Sharpe Ratio (probability the Sharpe survives multiple testing).

    Based on Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".

    **Units.** The previous implementation compared an *annualised* Sharpe with a
    *per-period* ``E[max]`` and hardcoded ``observation_periods=365``, which made a
    per-period significance threshold of ~0.107 look like a legitimate annual
    hurdle.  Both sides are now the same units:

    * ``sharpe_is_annualized=True`` (default, legacy callers): the annualised
      Sharpe is first divided by ``sqrt(DSR_PERIODS_PER_YEAR)`` to obtain the
      per-period Sharpe ``SR``.
    * the hurdle is the expected maximum of ``n_trials`` independent per-period
      Sharpes, ``E[max] = sqrt(1/T) * sqrt(2 ln N)`` with ``T`` the REAL number of
      observations (``observation_periods``).

    ``DSR`` (returned as ``dsr``) is the deflated per-period Sharpe
    ``SR - E[max]`` — negative means "not distinguishable from data mining" and
    must block publication.  ``p_value`` additionally applies the PSR
    non-normality correction ``sqrt(1 - skew*SR + (kurtosis-1)/4 * SR**2)``.

    Parameters
    ----------
    observed_sharpe : float
        Sharpe of the champion (annualised by default).
    n_trials : int
        Number of strategies evaluated — population × generations **plus every
        earlier walk-forward window's trials** (see
        :func:`core.ga.trial_counter.total_trials`).
    observation_periods : int
        Real number of return observations ``T`` (not a hardcoded 365).
    variance_sharpe : float
        Variance of the per-period Sharpe under the null (≈1 for i.i.d. returns).
    sharpe_is_annualized : bool
        Convert ``observed_sharpe`` from annualised to per-period first.
    skew, kurtosis : float
        Sample skewness / (non-excess) kurtosis of the return series.
    """
    import math
    from scipy import stats as _stats

    n_trials = int(max(n_trials, 1))
    t_periods = int(max(observation_periods, 1))
    var_sr = max(_finite(variance_sharpe, 1.0), 1e-12)

    sr = _finite(observed_sharpe, ndigits=6)
    if sharpe_is_annualized:
        sr = sr / math.sqrt(DSR_PERIODS_PER_YEAR)

    if sr <= 0 or n_trials <= 1 or t_periods < 2:
        return {"dsr": 0.0, "p_value": 1.0, "significant": False,
                "n_trials": n_trials, "observation_periods": t_periods,
                "sharpe_per_period": round(sr, 6), "expected_max_random": 0.0}

    # Expected maximum Sharpe from n_trials random trials (per-period units).
    expected_max = math.sqrt(var_sr / t_periods) * math.sqrt(2.0 * math.log(n_trials))
    dsr = sr - expected_max

    # Probabilistic Sharpe Ratio with the Bailey–López de Prado variance
    # correction for skewness / kurtosis of the return series.
    var_term = (1.0 - _finite(skew) * sr
                + ((_finite(kurtosis, 3.0) - 1.0) / 4.0) * sr * sr)
    var_term = max(var_term, 1e-9)
    z_score = dsr * math.sqrt(t_periods - 1) / math.sqrt(var_term)
    p_value = 1.0 - _stats.norm.cdf(z_score)

    return {
        "dsr": round(dsr, 6),
        "expected_max_random": round(expected_max, 6),
        "p_value": round(max(min(_finite(p_value, 1.0), 1.0), 0.0), 4),
        "significant": dsr > 0 and p_value < 0.05,
        "n_trials": n_trials,
        "observation_periods": t_periods,
        "sharpe_per_period": round(sr, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Equity-series statistics + the single scoring function shared by every path
# (single chromosome, threaded batch, multiprocess chunk, walk-forward validation)
# ═══════════════════════════════════════════════════════════════════════════════

def daily_returns(equity_curve: list[dict]) -> np.ndarray:
    """Daily returns resampled from an equity curve (same basis as metrics.py)."""
    if not equity_curve or len(equity_curve) < 2:
        return np.array([])
    try:
        eq = pd.Series(
            [float(p["equity"]) for p in equity_curve],
            index=pd.DatetimeIndex([pd.Timestamp(p["time"]) for p in equity_curve]),
        )
        daily = eq.resample("1D").last().dropna()
        if len(daily) < 2:
            return np.array([])
        return daily.pct_change().dropna().values
    except Exception:
        return np.array([])


def per_period_sharpe(returns) -> float:
    """Non-annualised Sharpe of a return series (per observation)."""
    arr = np.asarray(returns, dtype=float) if returns is not None else np.array([])
    if arr.size < 2:
        return 0.0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if std <= 0:
        return 0.0
    return float(arr.mean() / std)


def max_drawdown_pct(equity_curve: list[dict]) -> float:
    """Peak-to-trough drawdown (%) of an equity curve."""
    if not equity_curve:
        return 0.0
    equities = np.array([_finite(p.get("equity")) for p in equity_curve], dtype=float)
    peak = equities[0] if equities.size else 0.0
    max_dd = 0.0
    for value in equities:
        if value > peak:
            peak = value
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak * 100.0)
    return float(max_dd)


def stats_from_trades(trades: list[dict], equity_curve: list[dict],
                      initial_balance: float = 10000.0) -> dict:
    """Per-genome statistics derived from ITS OWN trades and equity series."""
    pnls = np.array([_finite(t.get("pnl")) for t in (trades or [])], dtype=float)
    n = int(pnls.size)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(abs(losses.sum())) if losses.size else 0.0
    mean_win = float(wins.mean()) if wins.size else 0.0
    mean_loss = float(abs(losses.mean())) if losses.size else 0.0
    raw_pf = (gross_win / gross_loss) if gross_loss > 0 else (
        100.0 if gross_win > 0 else 0.1)

    long_trades = sum(1 for t in trades or [] if t.get("side") == "long")
    short_trades = n - long_trades
    pnl = float(pnls.sum()) if n else 0.0

    equity_curve = equity_curve or []
    initial = max(_finite(initial_balance, 10000.0), 1e-9)
    final = _finite(equity_curve[-1].get("equity"), initial) if equity_curve else initial
    rets = daily_returns(equity_curve)
    periods_per_year = max(len(rets), 1) * (365.0 / max(_days_span(equity_curve), 1.0))
    return {
        "trades": n,
        "wins": int(wins.size),
        "losses": int(losses.size),
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "mean_win": mean_win,
        "mean_loss": mean_loss,
        "win_rate": (wins.size / n * 100.0) if n else 0.0,
        "raw_profit_factor": raw_pf,
        "profit_factor": profit_factor_shrunk(gross_win, gross_loss, mean_win),
        "pnl": pnl,
        "total_return_pct": (final - initial) / initial * 100.0,
        # Return on capital as a FRACTION (the unit ``w["roc"]`` was calibrated
        # against).  ``total_return_pct`` above is the readable percentage.
        "return_on_capital": pnl / initial,
        "pct_per_trade_return": (pnl / initial) * 100.0,
        "sharpe": per_period_sharpe(rets) * (periods_per_year ** 0.5),
        "sharpe_per_period": per_period_sharpe(rets),
        "max_dd_pct": max_drawdown_pct(equity_curve),
        "observations": int(rets.size),
        "skew": float(pd.Series(rets).skew()) if rets.size > 2 else 0.0,
        "kurtosis": float(pd.Series(rets).kurtosis() + 3.0) if rets.size > 3 else 3.0,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "buy_hold_pct": None,
        "alpha_pct": 0.0,
    }


def _days_span(equity_curve: list[dict]) -> float:
    """Calendar days covered by an equity curve (≥1)."""
    if not equity_curve or len(equity_curve) < 2:
        return 1.0
    try:
        span = (pd.Timestamp(equity_curve[-1]["time"])
                - pd.Timestamp(equity_curve[0]["time"])).days
        return float(max(span, 1))
    except Exception:
        return 1.0


def score_stats(stats: dict, chromosome: dict | None = None,
                weights: dict | None = None, n_trials: int = 1,
                prior_trials: int = 0) -> dict:
    """The ONE fitness formula every GA path uses.

    See the module docstring for the formula and the rationale of each term.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: v for k, v in weights.items() if k in w})

    trades = int(_finite(stats.get("trades")))
    win_rate = _finite(stats.get("win_rate"))
    pf_term = min(max(_finite(stats.get("profit_factor"), 0.1), 0.1), PF_TERM_CAP)
    pf_term *= min(1.0, trades / float(PF_TRADE_FLOOR))
    roc = _finite(stats.get("return_on_capital"))
    if not roc:
        # Backward-compatible fallback for callers that only set the percentage.
        roc = _finite(stats.get("pct_per_trade_return")) / 100.0
    pnl = _finite(stats.get("pnl"))

    if trades > 0:
        imbalance = abs(_finite(stats.get("long_trades")) / trades - 0.5) * 2.0
    else:
        imbalance = 1.0

    fitness = (
        win_rate * w["wr"]
        + pf_term * w["pf"]
        + roc * w["roc"]
        - imbalance * w["bal"]
    )

    # Evidence penalties (unchanged thresholds — the labels are the contract).
    if trades < 5:
        fitness -= 20
    elif trades < 15:
        fitness -= 5
    elif trades > 500:
        fitness -= (trades - 500) * 0.02

    if pnl < -50:
        fitness -= abs(pnl) * 0.3

    # ── Risk-adjusted selection metric (DSR-deflated Sharpe, benchmark-relative) ──
    stats_out = dict(stats)
    stats_out["profit_factor"] = pf_term
    stats_out["sharpe"] = _finite(stats.get("sharpe"))
    stats_out["max_dd"] = _finite(stats.get("max_dd_pct"))
    _obs = int(_finite(stats.get("observations")))
    if _obs >= MIN_OBSERVATIONS:
        _dsr = deflated_sharpe_ratio(
            stats_out["sharpe"], int(n_trials) + int(prior_trials),
            observation_periods=_obs,
            sharpe_is_annualized=True,
            skew=_finite(stats.get("skew")),
            kurtosis=_finite(stats.get("kurtosis"), 3.0),
        )
    else:
        # Too few observations to estimate a Sharpe at all — no alpha credit and
        # no DSR (the genome is flagged via ``insufficient_data`` below).
        _dsr = {"dsr": 0.0, "p_value": 1.0, "significant": False,
                "n_trials": int(n_trials) + int(prior_trials),
                "observation_periods": _obs, "expected_max_random": 0.0,
                "sharpe_per_period": 0.0}
    stats_out["dsr_detail"] = _dsr
    dsr_sharpe = _dsr["dsr"] * (365.0 ** 0.5)          # annualise the deflated SR
    evidence = min(1.0, trades / float(SHARPE_TRADE_FLOOR))
    alpha = dsr_sharpe * evidence - stats_out["max_dd"]
    stats_out["alpha_pct"] = alpha
    stats_out["deflated_sharpe"] = _dsr["dsr"]
    fitness += alpha * ALPHA_WEIGHT

    if chromosome is not None:
        fitness -= complexity_penalty(chromosome)

    stats_out["fitness"] = round(fitness, 4)
    stats_out["fitness_base"] = round(fitness - alpha * ALPHA_WEIGHT, 4)
    stats_out["fitness_alpha"] = round(alpha * ALPHA_WEIGHT, 4)
    # Alpha versus the equal-weighted buy & hold of the SAME window/symbols.
    baseline = stats.get("buy_hold_pct")
    if baseline is None:
        stats_out["alpha_vs_buy_hold_pct"] = 0.0
    else:
        stats_out["alpha_vs_buy_hold_pct"] = (
            _finite(stats.get("total_return_pct")) - _finite(baseline))
    stats_out["insufficient_data"] = trades < MIN_TRADES_GATE
    if trades == 0:
        stats_out["flag"] = "no_trades"
    elif trades < MIN_TRADES_GATE:
        stats_out["flag"] = "insufficient_trades"
    else:
        stats_out["flag"] = ""
    return stats_out


def stats_from_engine_result(result: dict, strategy_name: str,
                             initial_balance: float = 10000.0) -> dict:
    """Per-genome stats from an isolated engine result (falls back gracefully)."""
    per = (result.get("per_strategy_equity") or {}).get(strategy_name)
    if per is None:
        all_per = result.get("per_strategy_equity") or {}
        # The engine names the ledger after the config it was handed; a caller
        # that renamed the config afterwards (GA chunk naming) still gets its own
        # data, but ONLY when the run held a single strategy (otherwise the
        # per-genome split would silently merge siblings).
        if len(all_per) == 1:
            per = next(iter(all_per.values()))
    per = per or {}
    trades = per.get("trades")
    if trades is None:
        trades = [t for t in result.get("trades", [])
                  if t.get("strategy") == strategy_name]
        if not trades and len(result.get("strategies", []) or []) == 1:
            trades = list(result.get("trades", []) or [])
    equity_curve = per.get("equity_curve") or result.get("equity_curve") or []
    stats = stats_from_trades(trades, equity_curve, initial_balance)
    metrics = result.get("metrics", {}) or {}
    stats["buy_hold_pct"] = metrics.get("buy_hold_pct")
    if not stats["max_dd_pct"]:
        stats["max_dd_pct"] = abs(_finite(metrics.get("max_drawdown_pct", 0)))
    if not stats["trades"]:
        # A genome may still show matrix cells; keep its trades=0 flag honest.
        stats["flag"] = "no_trades"
    baseline = stats.get("buy_hold_pct")
    if baseline is not None:
        # Alpha vs buy & hold of the same window/symbols, in return points.
        stats["alpha_vs_buy_hold_pct"] = stats["total_return_pct"] - _finite(baseline)
    else:
        stats["alpha_vs_buy_hold_pct"] = 0.0
    return stats


def evaluate_population_batch(
    population: list[dict],
    symbols: list[str],
    date_start: str,
    date_end: str,
    engine,
    loader: StrategyLoader,
    ga_loader: StrategyLoader | None = None,
    batch_size: int = 10,
    initial_balance: float = 10000.0,
    progress_callback=None,
    max_workers: int = 4,
    weights: dict | None = None,
    use_live_spread: bool = False,
    batch_trials: int = 1,
    prior_trials: int = 0,
) -> list[dict]:
    """Evaluate chromosomes in parallel batched backtests.

    Args:
        weights: Optional dict with keys 'wr', 'pf', 'roc', 'bal' to override
                 the default fitness formula weights. Loaded from calibration
                 data when available.
        use_live_spread: False (default) → never price historical fills from
                 today's order book; unmapped symbols take the configured
                 default spread and the source is reported in every result.
        batch_trials:  Number of genomes evaluated in THIS generation (feeding
                 the DSR multiple-testing correction).
        prior_trials:  Trials accumulated by earlier GA/WF runs.

    Strategies are split into *max_workers* groups and processed in parallel
    threads. Since each strategy is isolated (per_strategy_isolation=True),
    there is zero cross-contamination between parallel workers.

    Args:
        max_workers: Number of parallel threads for batch evaluation.
    """
    total = len(population)
    results: list = [None] * total

    # ── Split population into worker groups ──
    workers = min(max_workers, total)
    # Distribute remainder evenly: 7 strategies ÷ 4 workers → [2,2,2,1]
    base = total // workers
    rem = total % workers
    chunks = []
    cursor = 0
    for w in range(workers):
        size = base + (1 if w < rem else 0)
        if size > 0:
            chunks.append((cursor, cursor + size))
            cursor += size

    logger.info(f"GA batch eval: {total} strategies in {len(chunks)} parallel groups "
                f"on {date_start}~{date_end} with {len(symbols)} symbols")

    completed_lock = __import__('threading').Lock()
    completed = [0]

    def _eval_chunk(chunk_start: int, chunk_end: int):
        """Evaluate one chunk of strategies (called in a thread)."""
        chunk_pop = population[chunk_start:chunk_end]

        # Build StrategyConfig objects
        chunk_configs = []
        for i, chrom in enumerate(chunk_pop):
            config = chromosome_to_strategy(chrom)
            if config.ml_config:
                config.ml_config.enabled = False
            config.name = f"ga_chunk_{chunk_start}_{chunk_start + i}_{random.randint(1000,9999)}"
            chunk_configs.append(config)

        # Single backtest for this chunk
        result = engine.run_with_exit_evaluation(
            strategies=chunk_configs,
            symbols=symbols,
            date_start=date_start,
            date_end=date_end,
            initial_balance=initial_balance,
            mode="full",
            simulate_ai_weights=False,
            ml_engine="lightgbm",
            per_strategy_isolation=True,
            per_genome_ledger=True,
            use_live_spread=use_live_spread,
        )

        if "error" in result:
            for i in range(len(chunk_configs)):
                results[chunk_start + i] = {
                    "fitness": -999, "error": result["error"],
                    "flag": "engine_error", "trade_count": 0}
            return

        # Extract per-strategy stats and score with the shared formula.
        for i, config in enumerate(chunk_configs):
            idx = chunk_start + i
            chrom = population[idx]
            stats = stats_from_engine_result(result, config.name, initial_balance)
            stats = score_stats(stats, chrom, weights=weights,
                                n_trials=batch_trials,
                                prior_trials=prior_trials)
            results[idx] = {
                "fitness": stats["fitness"],
                "fitness_base": stats.get("fitness_base"),
                "fitness_alpha": stats.get("fitness_alpha"),
                "sharpe": round(_finite(stats["sharpe"]), 4),
                "win_rate": round(_finite(stats["win_rate"]), 2),
                "profit_factor": round(_finite(stats["profit_factor"]), 4),
                "raw_profit_factor": round(_finite(stats["raw_profit_factor"]), 4),
                "max_dd": round(_finite(stats["max_dd"]), 2),
                "total_return": round(_finite(stats["total_return_pct"]), 2),
                "buy_hold_pct": stats["buy_hold_pct"],
                "alpha_vs_buy_hold_pct": round(
                    _finite(stats["alpha_vs_buy_hold_pct"]), 4),
                "dsr": round(_finite(stats["deflated_sharpe"]), 4),
                "dsr_detail": stats["dsr_detail"],
                "observations": int(stats["observations"]),
                "trade_count": int(stats["trades"]),
                "long_trades": int(stats["long_trades"]),
                "short_trades": int(stats["short_trades"]),
                "flag": stats.get("flag", ""),
                "strategy_name": config.name,
            }

            with completed_lock:
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0], total)

    # ── Run chunks in parallel threads ──
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(_eval_chunk, s, e) for s, e in chunks]
            for f in futures:
                f.result(timeout=3600)  # 1h timeout per chunk
    else:
        _eval_chunk(chunks[0][0], chunks[0][1])

    # Apply results to population
    for i, r in enumerate(results):
        if r is not None:
            population[i]["fitness_result"] = r
        else:
            population[i]["fitness_result"] = {"fitness": -999, "error": "batch eval failed"}

    return population


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Process Batch Evaluation — uses ProcessPoolExecutor to avoid TA-Lib
# C-extension crashes under multi-threading.
# ═══════════════════════════════════════════════════════════════════════════════

def _mp_worker(worker_args: dict) -> list:
    """Module-level picklable worker for ProcessPoolExecutor.

    Each worker process creates its OWN engine stack from scratch.
    No shared state with the parent or sibling processes.
    TA-Lib is imported fresh in each process — no thread-safety issues.

    Args:
        worker_args: dict with keys:
            population_chunk, symbols, date_start, date_end,
            initial_balance, cost_enabled, taker_fee_pct, spread_pct,
            weights, chunk_start_idx, engine_mode

    Returns:
        list of (index, result_dict) tuples
    """
    import random as _random
    import numpy as _np
    from app.config import Config
    from core.strategy.loader import StrategyLoader
    from core.backtest.engine import BacktestEngine
    from core.risk.manager import RiskManager
    from core.executor.executor import OrderExecutor
    from app.event_bus import EventBus
    from core.ga.genome import chromosome_to_strategy as _c2s

    # ── Determinism: each worker seeds itself from the job's seed ──
    _seed = int(worker_args.get("seed") or 0)
    if _seed:
        _random.seed(_seed)
        _np.random.seed(_seed % (2 ** 32))

    # ── Per-process engine stack ──
    config = Config.load("sim")
    config.backtest_cost_enabled = worker_args["cost_enabled"]
    config.backtest_taker_fee_pct = worker_args["taker_fee_pct"]
    config.backtest_spread_pct = worker_args["spread_pct"]
    config.backtest_engine_mode = worker_args.get("engine_mode", "legacy")

    event_bus = EventBus()
    risk_manager = RiskManager(config, event_bus)
    order_executor = OrderExecutor(config, event_bus)

    loader = StrategyLoader(str(Path(config.data_dir).parent / "strategies"))
    engine = BacktestEngine(config, None, risk_manager, order_executor)

    # ── Evaluate chunk ──
    chunk_start = worker_args["chunk_start_idx"]
    population_chunk = worker_args["population_chunk"]
    symbols = worker_args["symbols"]
    date_start = worker_args["date_start"]
    date_end = worker_args["date_end"]
    initial_balance = worker_args["initial_balance"]
    weights = worker_args.get("weights")

    # ── Test hook: lets a suite inject a deterministic/failing evaluator without
    # replacing this module-level worker (which must stay picklable). ──
    _hook = worker_args.get("evaluate_hook")
    if _hook is not None:
        return _hook(worker_args)

    chunk_configs = []
    for i, chrom in enumerate(population_chunk):
        config_obj = _c2s(chrom)
        if config_obj.ml_config:
            config_obj.ml_config.enabled = False
        config_obj.name = f"ga_mp_{chunk_start}_{chunk_start + i}_{_random.randint(1000, 9999)}"
        chunk_configs.append(config_obj)

    result = engine.run_with_exit_evaluation(
        strategies=chunk_configs,
        symbols=symbols,
        date_start=date_start,
        date_end=date_end,
        initial_balance=initial_balance,
        mode="full",
        simulate_ai_weights=False,
        ml_engine="lightgbm",
        per_strategy_isolation=True,
        per_genome_ledger=True,
        use_live_spread=worker_args.get("use_live_spread", False),
    )

    if "error" in result:
        return [(chunk_start + i, {"fitness": -999, "error": result["error"],
                                   "flag": "engine_error", "trade_count": 0,
                                   "strategy_name": c.name})
                for i, c in enumerate(chunk_configs)]

    # ── Extract per-strategy stats — the SAME shared scorer as the threaded path.
    # (Before this, the two paths were separate copy-pasted formulas and the
    # multiprocess one hardcoded sharpe=0/max_dd=0.)
    results = []
    for i, config_obj in enumerate(chunk_configs):
        idx = chunk_start + i
        chrom = population_chunk[i]
        stats = stats_from_engine_result(result, config_obj.name, initial_balance)
        stats = score_stats(stats, chrom, weights=weights,
                            n_trials=worker_args.get("batch_trials", 1),
                            prior_trials=worker_args.get("prior_trials", 0))
        results.append((idx, {
            "fitness": stats["fitness"],
            "fitness_base": stats.get("fitness_base"),
            "fitness_alpha": stats.get("fitness_alpha"),
            "sharpe": round(_finite(stats["sharpe"]), 4),
            "win_rate": round(_finite(stats["win_rate"]), 2),
            "profit_factor": round(_finite(stats["profit_factor"]), 4),
            "raw_profit_factor": round(_finite(stats["raw_profit_factor"]), 4),
            "max_dd": round(_finite(stats["max_dd"]), 2),
            "total_return": round(_finite(stats["total_return_pct"]), 2),
            "buy_hold_pct": stats["buy_hold_pct"],
            "alpha_vs_buy_hold_pct": round(
                _finite(stats["alpha_vs_buy_hold_pct"]), 4),
            "dsr": round(_finite(stats["deflated_sharpe"]), 4),
            "dsr_detail": stats["dsr_detail"],
            "observations": int(stats["observations"]),
            "trade_count": int(stats["trades"]),
            "long_trades": int(stats["long_trades"]),
            "short_trades": int(stats["short_trades"]),
            "flag": stats.get("flag", ""),
            "strategy_name": config_obj.name,
        }))

    return results


def _single_worker_retry(args_for_chunk: dict, error: Exception) -> tuple[list, str]:
    """Retry one crashed chunk **in this process** (``max_workers=1``).

    Production showed every generation losing its whole population because one
    crashed chunk scored all of its genomes −999 with no fallback.  Returns
    ``(results, path)`` where ``path`` is ``"retry_single_worker"`` on success or
    ``"failed"`` (results filled with −999) when even the serial retry dies.
    """
    chunk_start = args_for_chunk["chunk_start_idx"]
    chunk_size = len(args_for_chunk.get("population_chunk") or [])
    try:
        retried = _mp_worker(dict(args_for_chunk))
    except Exception as e2:
        logger.error(f"Single-worker retry failed too: {e2}")
        retried = None
    if retried:
        return list(retried), "retry_single_worker"
    return ([(chunk_start + i, {"fitness": -999, "error": str(error),
                                "eval_path": "failed"})
             for i in range(chunk_size)], "failed")


def evaluate_population_multiprocess(
    population: list[dict],
    symbols: list[str],
    date_start: str,
    date_end: str,
    initial_balance: float = 10000.0,
    max_workers: int = 4,
    progress_callback=None,
    weights: dict | None = None,
    cost_enabled: bool = True,
    taker_fee_pct: float = 0.04,
    spread_pct: dict | None = None,
    engine_mode: str = "legacy",
    use_live_spread: bool = False,
    batch_trials: int = 1,
    prior_trials: int = 0,
    seed: int = 0,
) -> list[dict]:
    """Evaluate chromosomes in parallel PROCESSES (not threads).

    Uses ProcessPoolExecutor to avoid TA-Lib C-extension thread-safety crashes.
    Each process independently creates its own engine, loader, etc.

    Args:
        max_workers: Number of parallel processes. Each gets ~ceil(N/max_workers)
                     strategies and runs one batched backtest.
        progress_callback: Called as callback(completed_count, total_count).
        seed: Job seed; each worker seeds ``random``/``numpy`` from it.
        batch_trials / prior_trials: DSR multiple-testing counts.
        use_live_spread: False → no order-book lookups for historical fills.

    A chunk that dies is retried **once in-process at max_workers=1** before it
    is marked -999: production showed every generation losing its whole
    population because one crashed chunk poisoned all of them.
    """
    total = len(population)
    results: list = [None] * total
    paths_used: dict[int, str] = {}

    workers = min(max_workers, total)
    base = total // workers
    rem = total % workers
    chunks = []
    cursor = 0
    for w in range(workers):
        size = base + (1 if w < rem else 0)
        if size > 0:
            chunks.append((cursor, size))
            cursor += size

    spread = spread_pct or {}

    logger.info(
        f"GA multiprocess eval: {total} strategies in {len(chunks)} processes "
        f"on {date_start}~{date_end} with {len(symbols)} symbols"
    )

    # Prepare worker args for each chunk
    futures_args = []
    for _ci, (chunk_start, chunk_size) in enumerate(chunks):
        args = {
            "population_chunk": population[chunk_start:chunk_start + chunk_size],
            "symbols": symbols,
            "date_start": date_start,
            "date_end": date_end,
            "initial_balance": initial_balance,
            "cost_enabled": cost_enabled,
            "taker_fee_pct": taker_fee_pct,
            "spread_pct": spread,
            "weights": weights,
            "chunk_start_idx": chunk_start,
            "engine_mode": engine_mode,
            "use_live_spread": use_live_spread,
            "batch_trials": batch_trials,
            "prior_trials": prior_trials,
            # Deterministic per-chunk seed (identical for a given job seed).
            "seed": (int(seed) + _ci) if seed else 0,
        }
        futures_args.append(args)

    completed_count = 0

    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {executor.submit(_mp_worker, a): a["chunk_start_idx"]
                   for a in futures_args}

        for future in as_completed(futures):
            chunk_start = futures[future]
            chunk_size = next(
                (s for cs, s in chunks if cs == chunk_start), total - chunk_start
            )
            args_for_chunk = next(a for a in futures_args
                                  if a["chunk_start_idx"] == chunk_start)
            try:
                chunk_results = future.result(timeout=3600)
            except Exception as e:
                logger.error(f"Multiprocess chunk failed ({e}); retrying "
                             f"chunk@{chunk_start} at max_workers=1")
                chunk_results, path = _single_worker_retry(args_for_chunk, e)
                if path == "failed":
                    logger.error("Single-worker retry failed too")
            else:
                path = "process"
            for idx, r in chunk_results:
                results[idx] = r
            completed_count += len(chunk_results)
            paths_used[chunk_start] = path

            if progress_callback:
                progress_callback(completed_count, total)

    # Apply results to population
    for i, r in enumerate(results):
        if r is not None:
            population[i]["fitness_result"] = r
        else:
            population[i]["fitness_result"] = {"fitness": -999, "error": "mp eval failed"}

    logger.info(f"GA multiprocess eval paths: {paths_used}")
    return population
