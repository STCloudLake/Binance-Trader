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
    volume_context: "VolumeContext | None" = None,
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
        # ── P6-D executability (impact term + volume-aware sizing) ──
        if executability_applies(chromosome, getattr(engine, "config", None)):
            modelled = apply_executability_model(
                list(trades), list(equity_curve), chromosome,
                getattr(engine, "config", None), volume_context=volume_context,
                initial_balance=initial_balance)
            if modelled["applied"]:
                stats = stats_from_trades(modelled["trades"],
                                          modelled["equity_curve"],
                                          initial_balance)
                stats["executability"] = modelled["summary"]
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


# ═══════════════════════════════════════════════════════════════════════════════
# P6-D — executability in fitness: the P6-A impact term and volume-aware sizing
# ═══════════════════════════════════════════════════════════════════════════════
#
# The GA used to score a genome on fills that ignored what its own size costs the
# market.  This block lets the fitness **cost model** carry P6-A's impact term
# (``risk.liquidity.impact_k``) and a volume-aware size model, both OFF by default:
#
# * ``impact_k <= 0`` (the shipped value) and ``volume_scale_k == 0`` (the gene's
#   neutral value) short-circuit at the top of :func:`apply_executability_model`,
#   which then returns its inputs **unchanged** — the OFF path is the identity,
#   not an approximation.
# * The impact charge is *replaced*, never added: the engine may already have
#   priced it (P6-B wired ``recent_quote_volume`` into
#   ``BacktestEngine._close_position``), so the model measures the impact already
#   inside ``trade["cost"]`` and recomputes the cost from the modelled notional.
# * Size is shrink-only (``≤ 1.0``) and is then put through P6-A's participation
#   ceiling (``cap_notional``), which only ever shrinks further — the model can
#   never assume more capacity than ``risk.liquidity.max_participation_pct``
#   allows.
#
# Units: ``recent_quote_volume`` is quote (USDT) notional over a bar window,
# ``max_participation_pct`` is a PERCENT (P6-A's config unit), ``impact_k`` is the
# dimensionless square-root-law coefficient.

#: Bars behind the RVOL the size gene reads (``volume_ratio`` in the templates is
#: ``volume / sma(volume, 20)`` — the same 20, so the gene and the condition
#: family measure one quantity).
RVOL_LOOKBACK_BARS = 20
#: Bounds of the modelled size factor.  ``1.0`` is the engine's size, so the gene
#: is shrink-only: it can never claim more size than the backtest actually traded.
VOLUME_SCALE_MIN = 0.25
VOLUME_SCALE_CAP = 1.0
#: Tolerance for "the engine already charged an impact term on this trade".
_IMPACT_EPSILON = 1e-9


def _gene_value(chromosome: dict | None, name: str, default: float = 0.0) -> float:
    """Value of the continuous gene *name* (0.0 when absent — pre-P6 genomes)."""
    for gene in (chromosome or {}).get("continuous", []) or []:
        if getattr(gene, "name", None) == name:
            try:
                return float(getattr(gene, "value", default))
            except (TypeError, ValueError):
                return default
    return default


def chromosome_volume_scale_k(chromosome: dict | None) -> float:
    """``volume_scale_k`` gene clamped to its documented bounds (0.0 = off)."""
    value = _gene_value(chromosome, "volume_scale_k", 0.0)
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(2.0, value))


def chromosome_volume_filter_rvol(chromosome: dict | None) -> float:
    """``volume_filter_rvol`` gene (0.0 = no filter) — for reporting/tests."""
    value = _gene_value(chromosome, "volume_filter_rvol", 0.0)
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(5.0, value))


def volume_size_factor(rvol, k: float,
                       floor: float = VOLUME_SCALE_MIN,
                       cap: float = VOLUME_SCALE_CAP) -> float:
    """Size factor for a bar whose relative volume is *rvol*, gene value *k*.

    ``factor = clip(1 + k·(rvol − 1), floor, cap)``: above-average volume keeps
    (or, at ``cap = 1``, cannot exceed) the base size, below-average volume
    shrinks it, and the result is always inside ``[floor, cap]``.  ``k <= 0``,
    an unknown ``rvol`` or a non-positive one → ``1.0`` (no change).
    """
    try:
        k = float(k)
        rvol = float(rvol)
    except (TypeError, ValueError):
        return 1.0
    if k <= 0.0 or rvol != rvol or rvol <= 0.0:
        return 1.0
    factor = 1.0 + k * (rvol - 1.0)
    return float(max(floor, min(cap, factor)))


def executability_params(config) -> dict:
    """P6-A knobs for the fitness cost model (never raises; off when absent)."""
    params = {"impact_k": 0.0, "impact_exponent": 0.5,
              "max_participation_pct": 0.0, "lookback_bars": RVOL_LOOKBACK_BARS,
              "participation_enabled": False}
    try:
        from core.risk.liquidity import resolve_liquidity_config

        block = resolve_liquidity_config(config)
    except Exception:  # pragma: no cover - duck-typed configs
        block = None
    if block is None:
        return params
    try:
        params["impact_k"] = float(getattr(block, "impact_k", 0.0) or 0.0)
        params["impact_exponent"] = float(
            getattr(block, "impact_exponent", 0.5) or 0.5)
        params["max_participation_pct"] = float(
            getattr(block, "max_participation_pct", 0.0) or 0.0)
        params["lookback_bars"] = int(
            getattr(block, "lookback_bars", RVOL_LOOKBACK_BARS)
            or RVOL_LOOKBACK_BARS)
        params["participation_enabled"] = bool(
            getattr(block, "enabled", False))
    except (TypeError, ValueError):  # pragma: no cover - malformed block
        return params
    if params["impact_k"] != params["impact_k"]:
        params["impact_k"] = 0.0
    return params


def executability_applies(chromosome: dict | None, config) -> bool:
    """True when the model would change anything (the OFF gate)."""
    if chromosome_volume_scale_k(chromosome) > 0.0:
        return True
    return executability_params(config)["impact_k"] > 0.0


class VolumeContext:
    """Per-(symbol, interval) volume statistics for one GA run.

    Built once per run (not per genome/generation) and picklable, so the
    multiprocess evaluation path can share one build.  ``lookup`` returns
    ``(rvol, recent_quote_volume)`` for a bar timestamp or ``None``.

    ``window_quote_volume`` is a **cumulative** sum of the per-bar quote notional
    with non-finite entries counted as 0, so a window is
    ``cum[cut] − cum[cut − lookback]`` — the exact same number
    :func:`core.risk.liquidity.recent_quote_volume` returns for the same slice
    (it sums the finite values of the last ``lookback`` rows).  ``rvol`` is
    ``volume / mean(volume, 20)``, i.e. the ``volume_ratio`` column the new
    condition templates read.
    """

    __slots__ = ("frames", "lookback_bars")

    def __init__(self, frames: dict | None = None,
                 lookback_bars: int = RVOL_LOOKBACK_BARS):
        self.frames = frames or {}
        self.lookback_bars = max(int(lookback_bars or RVOL_LOOKBACK_BARS), 1)

    def lookup(self, symbol: str, interval: str, ts):
        entry = self.frames.get((str(symbol), str(interval or "1h")))
        if entry is None or ts is None:
            return None
        try:
            cut = int(np.searchsorted(entry["index"], np.datetime64(
                pd.Timestamp(ts).to_datetime64()), side="right"))
        except Exception:
            return None
        if cut <= 0:
            return None
        rvol = entry["rvol"][cut - 1]
        start = max(0, cut - self.lookback_bars)
        volume = float(entry["cum_qv"][cut] - entry["cum_qv"][start])
        rvol = None if rvol != rvol else float(rvol)
        return rvol, volume


def build_volume_context(config, symbols, intervals, date_start: str,
                         date_end: str, lookback_bars: int = RVOL_LOOKBACK_BARS
                         ) -> VolumeContext | None:
    """Read the parquet cache once and pre-compute the per-bar volume statistics.

    Returns ``None`` (never raises) when there is no data dir / no cache — the
    executability model then falls back to what the engine itself priced.
    """
    try:
        from core.backtest.data_feeder import DataFeeder
        from pathlib import Path

        cache_dir = str(Path(getattr(config, "data_dir", "data")) / "market")
        frames: dict = {}
        for symbol in symbols or []:
            for interval in intervals or []:
                feeder = DataFeeder(cache_dir, [symbol], [interval],
                                    date_start, date_end)
                feeder.load()
                raw = feeder.get_all_data_for_symbol(symbol, interval)
                if raw is None or len(raw) == 0:
                    continue
                volume = pd.to_numeric(raw["volume"], errors="coerce").astype(float)
                mean = volume.rolling(RVOL_LOOKBACK_BARS).mean()
                rvol = np.where(mean > 0, volume / mean, np.nan)
                quote = None
                if "quote_volume" in raw.columns:
                    candidate = pd.to_numeric(raw["quote_volume"],
                                              errors="coerce").to_numpy(dtype=float)
                    if np.isfinite(candidate).any():
                        quote = candidate
                if quote is None:
                    close = pd.to_numeric(raw["close"], errors="coerce").to_numpy(dtype=float)
                    quote = volume.to_numpy(dtype=float) * close
                quote = np.where(np.isfinite(quote), quote, 0.0)
                frames[(str(symbol), str(interval))] = {
                    "index": raw.index.to_numpy(),
                    "rvol": np.asarray(rvol, dtype=float),
                    "cum_qv": np.concatenate([[0.0], np.cumsum(quote)]),
                }
        if not frames:
            return None
        return VolumeContext(frames, lookback_bars)
    except Exception as exc:  # pragma: no cover - a missing cache is not fatal
        logger.debug(f"volume context unavailable: {exc}")
        return None


def _legacy_cost(trade: dict, config) -> float:
    """Fees + half-spread for this trade's ORIGINAL notional (no impact term).

    ``recent_quote_volume=None`` makes :func:`apply_trading_costs` short-circuit to
    the pre-P6 arithmetic even when ``impact_k > 0``, which is exactly what is
    needed to measure how much impact the engine already charged.
    """
    from core.backtest.cost_model import apply_trading_costs

    entry = _finite(trade.get("entry_price"))
    exit_price = _finite(trade.get("exit_price"))
    qty = _finite(trade.get("quantity"))
    if entry <= 0 or qty <= 0:
        return 0.0
    return _finite(apply_trading_costs(
        entry, exit_price, qty, str(trade.get("symbol") or ""), config,
        recent_quote_volume=None))


def _engine_impact_in_cost(trade: dict, config) -> float:
    """Impact already inside ``trade["cost"]`` (0.0 when the engine priced none)."""
    charged = _finite(trade.get("cost"))
    legacy = _legacy_cost(trade, config)
    return max(0.0, charged - legacy)


def apply_executability_model(trades: list[dict], equity_curve: list[dict],
                              chromosome: dict | None, config,
                              volume_context: VolumeContext | None = None,
                              initial_balance: float = 10000.0) -> dict:
    """Modelled trades/equity under the P6-A impact term and the size gene.

    Returns ``{"applied", "trades", "equity_curve", "summary"}``.  When the model
    is off — ``impact_k <= 0`` and ``volume_scale_k == 0`` — it returns the
    **same objects** it was given, so the OFF path cannot perturb a single bit of
    the fitness.
    """
    summary = {"applied": False, "impact_k": 0.0, "scale_k": 0.0,
               "trades": 0, "trades_scaled": 0, "trades_capped": 0,
               "unmeasured_volume": 0, "impact_usdt": 0.0,
               "engine_impact_usdt": 0.0, "legacy_cost_usdt": 0.0,
               "notional_before": 0.0, "notional_after": 0.0,
               "min_factor": 1.0, "mean_factor": 1.0}
    params = executability_params(config)
    scale_k = chromosome_volume_scale_k(chromosome)
    summary["impact_k"] = round(params["impact_k"], 6)
    summary["scale_k"] = round(scale_k, 6)
    if not trades or (params["impact_k"] <= 0.0 and scale_k <= 0.0):
        return {"applied": False, "trades": trades,
                "equity_curve": equity_curve, "summary": summary}

    from core.risk.liquidity import cap_notional, total_impact_usdt

    adjusted: list[dict] = []
    pnl_delta: list[tuple] = []          # (closed_at, d_pnl, opened_at, d_notional)
    factors: list[float] = []
    for trade in trades:
        record = dict(trade)
        entry = _finite(trade.get("entry_price"))
        exit_price = _finite(trade.get("exit_price"))
        qty = _finite(trade.get("quantity"))
        cost_old = _finite(trade.get("cost"))
        pnl_old = _finite(trade.get("pnl"))
        entry_notional = qty * entry
        exit_notional = qty * exit_price
        if entry_notional <= 0 or qty <= 0:
            adjusted.append(record)
            factors.append(1.0)
            continue

        symbol = str(trade.get("symbol") or "")
        interval = str(trade.get("timeframe") or "1h")
        entry_bar = volume_context.lookup(symbol, interval,
                                          trade.get("opened_at")) \
            if volume_context is not None else None
        exit_bar = volume_context.lookup(symbol, interval,
                                         trade.get("closed_at")) \
            if volume_context is not None else None
        rvol = entry_bar[0] if entry_bar else None
        window = exit_bar[1] if exit_bar else None
        if window is None and volume_context is not None:
            summary["unmeasured_volume"] += 1

        factor = volume_size_factor(rvol, scale_k) if scale_k > 0 else 1.0
        target = entry_notional * factor
        capped, _reason = target, ""
        if window is not None and window > 0 and params["max_participation_pct"] > 0:
            capped, _reason = cap_notional(target, window,
                                           params["max_participation_pct"])
        f = capped / entry_notional if entry_notional > 0 else 1.0
        f = max(0.0, f)
        if f < factor - 1e-12:
            summary["trades_capped"] += 1
        factors.append(f)

        engine_impact = _engine_impact_in_cost(trade, config)
        impact_new = 0.0
        measured = window is not None and window > 0
        if params["impact_k"] > 0.0 and measured:
            impact_new = _finite(total_impact_usdt(
                capped, exit_notional * f, window, params["impact_k"],
                params["impact_exponent"]))
        # The engine may already have priced impact into ``trade["cost"]`` (P6-B
        # wired the P6-A seam into the engine).  Replace it ONLY when this model
        # can re-measure the same window; with an unmeasured window the charge is
        # left inside the cost (never stripped, never added twice).
        base_cost = cost_old - (engine_impact if measured else 0.0)
        legacy_new = base_cost * f
        gross = pnl_old + cost_old
        pnl_new = gross * f - legacy_new - impact_new

        record["pnl"] = round(pnl_new, 2)
        record["pnl_pct"] = (round(pnl_new / entry_notional * 100.0, 2)
                             if entry_notional > 0 else 0)
        record["amount_usdt"] = round(capped, 2)
        record["cost"] = round(legacy_new + impact_new, 4)
        record["exec_scale"] = round(f, 6)
        record["exec_impact_usdt"] = round(impact_new, 4)
        record["exec_engine_impact_usdt"] = round(engine_impact, 4)
        record["exec_recent_quote_volume"] = (round(float(window), 4)
                                             if window is not None else None)
        adjusted.append(record)
        pnl_delta.append((trade.get("closed_at"), pnl_new - pnl_old,
                          trade.get("opened_at"), capped - entry_notional))
        summary["impact_usdt"] += impact_new
        summary["engine_impact_usdt"] += engine_impact
        summary["legacy_cost_usdt"] += legacy_new
        summary["notional_before"] += entry_notional
        summary["notional_after"] += capped

    summary["applied"] = True
    summary["trades"] = len(trades)
    summary["trades_scaled"] = sum(1 for f in factors if abs(f - 1.0) > 1e-12)
    if factors:
        summary["min_factor"] = round(min(factors), 6)
        summary["mean_factor"] = round(sum(factors) / len(factors), 6)
    for key in ("impact_usdt", "engine_impact_usdt", "legacy_cost_usdt",
                "notional_before", "notional_after"):
        summary[key] = round(summary[key], 4)

    return {"applied": True, "trades": adjusted,
            "equity_curve": _adjust_equity_curve(equity_curve, pnl_delta),
            "summary": summary}


def _adjust_equity_curve(equity_curve: list[dict], deltas: list[tuple]) -> list[dict]:
    """Apply the modelled trade deltas **on top of** the engine's own curve.

    For each original point ``t``::

        equity(t) += Σ (pnl_new − pnl_old)   over trades closed at/before t
                   + Σ (notional_new − notional_old)  over trades open at t

    which mirrors the engine's per-genome ledger (``balance + invested``).  With
    no deltas the points are returned unchanged, so the identity holds exactly.
    """
    if not equity_curve or not deltas:
        return equity_curve
    out: list[dict] = []
    for point in equity_curve:
        try:
            ts = pd.Timestamp(point.get("time"))
        except Exception:  # pragma: no cover - malformed point
            out.append(dict(point))
            continue
        realised = 0.0
        invested = 0.0
        for closed_at, d_pnl, opened_at, d_notional in deltas:
            if closed_at is not None and pd.Timestamp(closed_at) <= ts:
                realised += d_pnl
            elif (opened_at is not None and pd.Timestamp(opened_at) <= ts
                  and closed_at is not None and pd.Timestamp(closed_at) > ts):
                invested += d_notional
        if realised == 0.0 and invested == 0.0:
            out.append(point)
            continue
        updated = dict(point)
        updated["equity"] = round(_finite(point.get("equity")) + realised + invested, 2)
        if "balance" in point:
            updated["balance"] = round(_finite(point.get("balance")) + realised, 2)
        if "invested" in point:
            updated["invested"] = round(_finite(point.get("invested")) + invested, 2)
        out.append(updated)
    return out


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
                             initial_balance: float = 10000.0,
                             chromosome: dict | None = None,
                             config=None,
                             volume_context: "VolumeContext | None" = None) -> dict:
    """Per-genome stats from an isolated engine result (falls back gracefully).

    With *chromosome*/*config* given, the P6-D executability model is applied to
    this genome's own trades and equity points before the statistics are derived
    (impact term + volume-aware sizing).  It is the identity when the model is
    off, and ``stats["executability"]`` carries the model's summary — so a fitness
    that was priced with the P6-A impact term says so.
    """
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
    modelled = None
    if chromosome is not None and config is not None \
            and executability_applies(chromosome, config):
        modelled = apply_executability_model(
            list(trades), list(equity_curve), chromosome, config,
            volume_context=volume_context, initial_balance=initial_balance)
    if modelled is not None and modelled["applied"]:
        trades = modelled["trades"]
        equity_curve = modelled["equity_curve"]
    stats = stats_from_trades(trades, equity_curve, initial_balance)
    if modelled is not None:
        stats["executability"] = modelled["summary"]
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
    volume_context: "VolumeContext | None" = None,
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
            stats = stats_from_engine_result(
                result, config.name, initial_balance,
                chromosome=chrom, config=getattr(engine, "config", None),
                volume_context=volume_context)
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
    volume_context = worker_args.get("volume_context")

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
        stats = stats_from_engine_result(
            result, config_obj.name, initial_balance,
            chromosome=chrom, config=getattr(engine, "config", None),
            volume_context=volume_context)
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
    volume_context: "VolumeContext | None" = None,
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
            # P6-D: one shared, picklable volume context for the executability
            # model (None when the model is off — nothing is measured then).
            "volume_context": volume_context,
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
