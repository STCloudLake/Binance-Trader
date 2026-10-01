"""Selectable performance benchmark for the GA publication gate.

Why this module exists (measured, this session)
-----------------------------------------------
The gate rejected a champion whose ``alpha_vs_buy_hold_pct`` was
``−12.1206`` while the same champion's out-of-sample window showed Sharpe
``5.69`` and max drawdown ``0.03 %`` (``strategies/ga_champion_1790844776.yaml``).
The benchmark it lost to was a **100 %-invested** buy & hold of the whole window
(``core/backtest/engine.py`` → ``metrics["buy_hold_pct"]``), compared against a
strategy that was only *in the market a fraction of the time*.  Raw total return
against a fully-invested benchmark is not an exposure- or risk-matched
comparison, so the gate punished the strategy for holding cash.

``ga.benchmark_mode`` therefore selects the benchmark the gate consumes:

``buy_hold``
    The historical, fully-invested equal-weighted buy & hold of the window
    (the **code default**; byte-identical to the pre-existing behaviour).
``exposure_matched``
    The same basket, held **only while the strategy held a position** — the
    recommended mode, shipped in ``config/config.yaml``.
``risk_matched``
    The fully-invested benchmark scaled to the strategy's realised per-period
    volatility (equal risk, not equal exposure).
``none``
    No benchmark criterion at all (DSR/PSR and net-expectancy still gate).

The gate itself lives in ``core.ga.evolver._publication_decision``; this module
only *computes* the benchmarks.  Nothing here changes DSR/PSR, the fitness sum,
selection, chunking, the progress stream or the timeframe pool.

Definitions (all formulas are per-run, per-strategy)
----------------------------------------------------
Let the run window be ``W = [t0, t1]`` and the strategy's own trade list be
``T`` (each trade carries ``symbol``, ``opened_at``, ``closed_at`` and
``amount_usdt``).  For symbol *s*:

* in-market interval — ``[opened_at, closed_at]`` clipped to ``W``; a trade with
  no ``closed_at`` (still open at the window end) is clipped to ``t1``; the raw
  intervals are merged into their **union** so overlapping positions are never
  double-counted;
* ``R_s`` — the buy & hold return **over that union**: the interval returns
  ``last_close / first_close − 1`` are compounded
  (``Π (1 + r_i) − 1``), i.e. hold *s* during those intervals and stay in cash
  (0 %) in between.  An interval with fewer than two bars inside it contributes
  nothing; a symbol with no usable interval is dropped from the basket;
* ``w_s`` — the deployed-capital share
  ``mean(amount_usdt over s's trades) / initial_balance`` clipped to ``[0, 1]``
  (``"capital"`` weighting).  This is *the same relative exposure the strategy
  used*, measured — not assumed — from its own fills, and it makes the idle
  cash in the comparison earn exactly what it earned for the strategy.  When no
  trade carries a usable ``amount_usdt`` the weights fall back to
  ``1 / len(symbols)`` for the traded symbols (``"equal_share_of_basket"``),
  stated in the report.

``exposure_matched`` benchmark = ``Σ_s w_s · R_s · 100`` (percent).  With **no
trades at all** the benchmark is ``0.0`` and ``alpha = strategy return``: a
strategy that never held a position has no exposure to match, and the gate
still rejects it on ``no_trades`` / DSR / net expectancy.  When no symbol yields
a usable interval (missing bars) ``benchmark_pct`` is ``None`` — the criterion
is then **skipped** (exactly like the legacy ``buy_hold_pct is None`` case) and
``benchmark_available=False`` says so in the provenance.

``risk_matched`` benchmark = ``buy_hold_pct · (σ_strategy / σ_benchmark)`` where
both ``σ`` are the standard deviations of the **daily** return series the
fidelity code already builds (``core.ga.fitness.daily_returns`` /
``per_period_sharpe``; ``core/backtest/metrics.py`` resamples the same way).
The reverse direction is reported too
(``strategy_risk_matched_pct = strategy_return · σ_benchmark / σ_strategy``).
When ``σ_benchmark == 0`` the scale is undefined and the raw benchmark is used
(``risk_scale=1.0``, ``risk_scale_fallback=True``), because a benchmark with no
risk is already the equal-risk comparison.

Everything the *report* adds (benchmark Sharpe / max drawdown / time-in-market,
information ratio, Jensen alpha, per-trade net edge) is **reported, never
gated**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# ── Mode registry ──────────────────────────────────────────────────────

#: The historical, fully-invested equal-weighted buy & hold of the window.
BUY_HOLD = "buy_hold"
#: The same basket, held only while the strategy held a position (recommended).
EXPOSURE_MATCHED = "exposure_matched"
#: The fully-invested benchmark scaled to the strategy's realised volatility.
RISK_MATCHED = "risk_matched"
#: No benchmark criterion (DSR/PSR and net expectancy still gate).
NONE = "none"

#: Every accepted ``ga.benchmark_mode`` value, in documentation order.
BENCHMARK_MODES = (BUY_HOLD, EXPOSURE_MATCHED, RISK_MATCHED, NONE)
#: The **code** default — used when the key is absent or blank.
CODE_DEFAULT_BENCHMARK_MODE = BUY_HOLD
#: Modes that require the matched-benchmark computation (all the others are
#: reported from the legacy value without touching the price cache).
COMPUTED_BENCHMARK_MODES = (EXPOSURE_MATCHED, RISK_MATCHED)


class BenchmarkModeError(ValueError):
    """``ga.benchmark_mode`` is unusable (wrong type / blank after stripping)."""


class UnknownBenchmarkModeError(BenchmarkModeError):
    """``ga.benchmark_mode`` names a mode this build does not implement."""


def parse_benchmark_mode(raw) -> str:
    """``ga.benchmark_mode`` value → validated mode string.

    ``None``/blank ⇒ the code default (:data:`CODE_DEFAULT_BENCHMARK_MODE`,
    ``buy_hold``), so an absent key keeps the historical behaviour exactly.  Any
    other value must be one of :data:`BENCHMARK_MODES`; an unknown mode raises
    the named :class:`UnknownBenchmarkModeError` (config load and GA job load
    both call this, so a typo fails **at load** instead of silently selecting a
    different gate).
    """
    if raw is None:
        return CODE_DEFAULT_BENCHMARK_MODE
    if not isinstance(raw, str):
        raise BenchmarkModeError(
            "ga.benchmark_mode must be a string, got "
            f"{type(raw).__name__}: {raw!r}")
    mode = raw.strip().lower()
    if not mode:
        return CODE_DEFAULT_BENCHMARK_MODE
    if mode not in BENCHMARK_MODES:
        raise UnknownBenchmarkModeError(
            f"unknown ga.benchmark_mode '{raw}'; accepted: "
            f"{', '.join(BENCHMARK_MODES)}")
    return mode


def coerce_benchmark_mode(raw) -> str:
    """Best-effort mode for **untrusted** input (a result dict / old caller).

    Config load and job load use :func:`parse_benchmark_mode` and *raise*; this
    variant is for the gate reading a ``train_result`` written by an older
    build: an unknown value falls back to the code default with one warning
    instead of taking the run down.
    """
    try:
        return parse_benchmark_mode(raw)
    except BenchmarkModeError as exc:
        logger.warning(f"{exc} — falling back to '{CODE_DEFAULT_BENCHMARK_MODE}'")
        return CODE_DEFAULT_BENCHMARK_MODE


def alpha_label(mode: str) -> str:
    """Gate reason label for *mode*: ``alpha_vs_buy_hold`` / ``..._exposure_matched``.

    ``buy_hold`` keeps the exact pre-existing string, so a legacy rejection
    reads identically (``alpha_vs_buy_hold=-12.12% <= 0 (no edge over buy &
    hold)``).
    """
    return f"alpha_vs_{mode}"


#: Human name of each benchmark, used in the gate's rejection reason.  The
#: ``buy_hold`` entry is the pre-existing wording **verbatim**, so a legacy
#: rejection reason is byte-identical.
_MODE_DESCRIPTIONS = {
    BUY_HOLD: "buy & hold",
    EXPOSURE_MATCHED: "the exposure-matched benchmark",
    RISK_MATCHED: "the risk-matched benchmark",
    NONE: "no benchmark",
}


def mode_description(mode: str) -> str:
    """Readable name of *mode* for a log line / rejection reason."""
    return _MODE_DESCRIPTIONS.get(mode, str(mode))


# ── Small numeric helpers ──────────────────────────────────────────────

def _finite(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _optional_float(value):
    """``float`` when *value* is a finite number, else ``None`` (JSON/YAML safe)."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _parse_ts(value):
    """``pd.Timestamp`` or ``None`` — never raises on a malformed trade stamp."""
    if value is None or value == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    return ts


def annualized_sharpe(returns) -> float:
    """``mean / std(ddof=1) * sqrt(365)`` of a daily series (0.0 when undefined).

    The same construction as ``core.backtest.metrics._annualized_sharpe`` and
    ``core.ga.fitness.per_period_sharpe(..) * sqrt(365)`` — the benchmark's
    Sharpe must be comparable with the strategy's reported Sharpe.
    """
    arr = np.asarray(returns, dtype=float) if returns is not None else np.array([])
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    std = float(arr.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(arr.mean() / std * (365.0 ** 0.5))


def max_drawdown_of_returns(returns) -> float:
    """Peak-to-trough drawdown (%) of the curve compounded from *returns*.

    ``core.ga.fitness.max_drawdown_pct`` runs on equity points; a benchmark is
    only ever a return series, so this compounds it to a unit curve first and
    applies the identical peak/trough rule.
    """
    arr = np.asarray(returns, dtype=float) if returns is not None else np.array([])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    curve = np.cumprod(1.0 + arr)
    peak = 1.0
    worst = 0.0
    for value in curve:
        if value > peak:
            peak = value
        if peak > 0:
            worst = max(worst, (peak - value) / peak * 100.0)
    return float(worst)


# ── In-market intervals and their buy & hold return ────────────────────

def merge_intervals(intervals) -> list:
    """Union of ``(start, end)`` pairs — sorted, overlapping/touching merged."""
    ordered = sorted((a, b) for a, b in intervals if a is not None and b is not None and b >= a)
    merged: list = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def in_market_intervals(trades, symbol: str, window_start, window_end) -> list:
    """Merged in-market intervals of *symbol*, reconstructed from *trades*.

    Each trade contributes ``[opened_at, closed_at]`` clipped to the window; a
    missing/invalid ``closed_at`` (a position still open at the window end) is
    clipped to ``window_end``.  A malformed ``opened_at`` is skipped.
    """
    start_w, end_w = _parse_ts(window_start), _parse_ts(window_end)
    raw = []
    for trade in trades or []:
        if str(trade.get("symbol")) != str(symbol):
            continue
        opened = _parse_ts(trade.get("opened_at"))
        if opened is None:
            continue
        closed = _parse_ts(trade.get("closed_at"))
        if closed is None:
            closed = end_w
        if start_w is not None:
            opened = max(opened, start_w)
        if end_w is not None:
            closed = min(closed, end_w)
        if closed < opened:
            continue
        raw.append((opened, closed))
    return merge_intervals(raw)


def intervals_return(closes, intervals):
    """Compounded buy & hold return over the union of *intervals*, or ``None``.

    *closes* is the symbol's close series (already restricted to the window).
    Each interval takes the first/last close **inside** it and compounds the
    interval returns; an interval with fewer than two bars inside contributes
    nothing (a data gap is never traded through).  ``None`` when no interval
    yields a usable return.
    """
    if closes is None or len(closes) == 0 or not intervals:
        return None
    total = 1.0
    used = 0
    for start, end in intervals:
        segment = closes[(closes.index >= start) & (closes.index <= end)]
        if len(segment) < 2:
            continue
        first = _finite(segment.iloc[0])
        last = _finite(segment.iloc[-1])
        if first <= 0:
            continue
        total *= (last / first)
        used += 1
    if not used:
        return None
    return float(total - 1.0)


def covered_bars(index, intervals) -> int:
    """How many bars of *index* fall inside any of *intervals*."""
    if index is None or len(index) == 0 or not intervals:
        return 0
    covered = np.zeros(len(index), dtype=bool)
    positions = index
    for start, end in intervals:
        covered |= (positions >= start) & (positions <= end)
    return int(covered.sum())


# ── Daily benchmark series (volatility / Sharpe / IR / Jensen basis) ───

def equity_daily_returns(equity_curve) -> pd.Series:
    """Daily returns of an equity curve — ``core.ga.fitness.daily_returns``, dated.

    The identical resample (``1D`` last → ``pct_change`` → dropna); the index is
    kept so a benchmark series can be aligned with it for the information ratio
    and the Jensen alpha.
    """
    if not equity_curve or len(equity_curve) < 2:
        return pd.Series(dtype=float)
    try:
        eq = pd.Series(
            [float(p["equity"]) for p in equity_curve],
            index=pd.DatetimeIndex([pd.Timestamp(p["time"]) for p in equity_curve]),
        )
    except Exception:
        return pd.Series(dtype=float)
    daily = eq.resample("1D").last().dropna()
    if len(daily) < 2:
        return pd.Series(dtype=float)
    return daily.pct_change().dropna()


def _daily_closes(frame) -> pd.Series:
    try:
        closes = frame["close"].astype(float)
    except Exception:
        return pd.Series(dtype=float)
    return closes.resample("1D").last().dropna()


def _inside_days(daily_index, intervals) -> np.ndarray:
    mask = np.zeros(len(daily_index), dtype=bool)
    for start, end in intervals or []:
        mask |= (daily_index >= start) & (daily_index <= end)
    return mask


def basket_daily_returns(frames: dict, weights: dict, window_start, window_end,
                         in_market: dict | None = None) -> pd.Series:
    """Exposure-weighted daily return series of the basket.

    ``weights`` are the **relative** weights (they need not sum to 1: the
    unweighted remainder is cash and returns 0, which is exactly the exposure
    matching).  With *in_market* given (``{symbol: merged intervals}``) each
    symbol's daily return is masked to the days it was actually held.
    """
    pieces = []
    for symbol, weight in (weights or {}).items():
        if not weight or weight <= 0:
            continue
        frame = (frames or {}).get(symbol)
        if frame is None or len(frame) == 0:
            continue
        window = frame[(frame.index >= window_start) & (frame.index <= window_end)]
        closes = _daily_closes(window)
        if len(closes) < 2:
            continue
        returns = closes.pct_change()
        if in_market is not None:
            intervals = (in_market or {}).get(symbol) or []
            if not intervals:
                continue
            returns = returns.where(_inside_days(returns.index, intervals), 0.0)
        pieces.append((returns * float(weight)).rename(symbol))
    if not pieces:
        return pd.Series(dtype=float)
    combined = pd.concat(pieces, axis=1)
    # The first daily return of every symbol is NaN (pct_change); a row where
    # NO symbol has a return yet is not a 0 % benchmark day, it is the series
    # start.  Dropping it keeps the benchmark's volatility / Sharpe on the same
    # footing as the strategy's own daily series.
    combined = combined.dropna(how="all")
    return combined.fillna(0.0).sum(axis=1)


def _information_ratio(strategy_rets: pd.Series, benchmark_rets: pd.Series):
    """``(mean(excess) / std(excess)) * sqrt(365)`` on the aligned daily grid."""
    if strategy_rets is None or benchmark_rets is None:
        return None
    if len(strategy_rets) < 2 or len(benchmark_rets) < 2:
        return None
    joined = pd.concat([strategy_rets.rename("s"), benchmark_rets.rename("b")],
                       axis=1, join="inner").dropna()
    if len(joined) < 2:
        return None
    excess = joined["s"] - joined["b"]
    std = float(excess.std(ddof=1))
    if std <= 0:
        return None
    return float(excess.mean() / std * (365.0 ** 0.5))


def _jensen_alpha(strategy_rets: pd.Series, benchmark_rets: pd.Series):
    """``(beta, alpha_annual_pct)`` from ``r_s = a + b * r_b`` on daily returns.

    Ordinary least squares through the aligned daily grid; the intercept is
    annualised (× 365) and reported in percent.  ``(None, None)`` when the
    benchmark has no variance or fewer than two overlapping observations.
    """
    if strategy_rets is None or benchmark_rets is None:
        return None, None
    joined = pd.concat([strategy_rets.rename("s"), benchmark_rets.rename("b")],
                       axis=1, join="inner").dropna()
    if len(joined) < 2:
        return None, None
    var_b = float(joined["b"].var(ddof=1))
    if var_b <= 0:
        return None, None
    cov = float(joined["s"].cov(joined["b"]))
    beta = cov / var_b
    alpha_daily = float(joined["s"].mean()) - beta * float(joined["b"].mean())
    return float(beta), float(alpha_daily * 365.0 * 100.0)


def _per_trade_net_edge(trades) -> tuple:
    """``(mean pnl, mean pnl / notional %)`` — net of the recorded costs.

    ``trade["pnl"]`` is already net of ``trade["cost"]``
    (``core.backtest.trade_book.close_position``), so this is the per-trade net
    edge the gate's net-expectancy criterion is about.
    """
    pnls = [_finite(t.get("pnl")) for t in trades or []]
    if not pnls:
        return None, None
    mean_pnl = float(np.mean(pnls))
    pct = [_finite(t.get("pnl")) / _finite(t.get("amount_usdt")) * 100.0
           for t in trades or [] if _finite(t.get("amount_usdt")) > 0]
    mean_pct = float(np.mean(pct)) if pct else None
    return mean_pnl, mean_pct


# ── The report ─────────────────────────────────────────────────────────

def _base_report(mode: str, buy_hold_pct) -> dict:
    """The report shape every mode shares (keys are always present)."""
    return {
        "mode": mode,
        "benchmark_pct": None,
        "buy_hold_pct": _optional_float(buy_hold_pct),
        "benchmark_available": False,
        "benchmark_sharpe": None,
        "benchmark_max_dd_pct": None,
        "benchmark_time_in_market_pct": None,
        "strategy_time_in_market_pct": None,
        "information_ratio": None,
        "jensen_alpha_annual_pct": None,
        "benchmark_beta": None,
        "risk_scale": None,
        "risk_scale_fallback": False,
        "strategy_risk_matched_pct": None,
        "net_edge_per_trade": None,
        "net_edge_per_trade_pct": None,
        "symbol_weights": {},
        "weighting": None,
        "notes": "",
    }


def equity_total_return_pct(equity_curve) -> float | None:
    """``(last / first − 1) * 100`` of an equity curve, or ``None``.

    Used only for the reported reverse of the equal-risk comparison, so the
    benchmark report is self-contained (the gated number is always the
    strategy's own ``total_return_pct`` from the fitness stats).
    """
    if not equity_curve or len(equity_curve) < 2:
        return None
    try:
        first = _finite(equity_curve[0].get("equity"))
        last = _finite(equity_curve[-1].get("equity"))
    except Exception:  # pragma: no cover - defensive
        return None
    if first <= 0:
        return None
    return (last / first - 1.0) * 100.0


def legacy_report(mode: str, buy_hold_pct, trades=None) -> dict:
    """Report for ``buy_hold`` / ``none`` — **no price read, no computation**.

    Under ``buy_hold`` the benchmark is exactly the legacy
    ``metrics["buy_hold_pct"]`` (a fully-invested basket is 100 % in the market
    by definition); under ``none`` the report names the mode and carries no
    benchmark at all, so the gate's benchmark criterion is skipped.  The
    per-trade net edge is filled in either case: it needs only the strategy's
    own trades, so the default path stays free of price I/O.
    """
    report = _base_report(mode, buy_hold_pct)
    mean_pnl, mean_pct = _per_trade_net_edge(trades)
    report["net_edge_per_trade"] = _optional_float(mean_pnl)
    report["net_edge_per_trade_pct"] = _optional_float(mean_pct)
    if mode == BUY_HOLD:
        report["benchmark_pct"] = _optional_float(buy_hold_pct)
        report["benchmark_available"] = buy_hold_pct is not None
        report["benchmark_time_in_market_pct"] = 100.0
        report["notes"] = (
            "fully-invested equal-weighted buy & hold of the run's window "
            "(legacy benchmark; exposure/risk statistics not computed on the "
            "default path)")
    else:
        report["notes"] = (
            "benchmark_mode=none — the benchmark criterion is disabled; the "
            "DSR/PSR and net-expectancy criteria still gate")
    return report


def build_benchmark(mode: str, *,
                    trades, symbols, frames, window_start, window_end,
                    initial_balance: float = 10000.0,
                    strategy_equity=None,
                    buy_hold_pct=None) -> dict:
    """The selected mode's benchmark report for ONE strategy.

    *frames* is ``{symbol: 1h DataFrame}`` (the same feeder data the legacy
    buy & hold reads).  The returned dict is JSON/YAML-safe (plain floats) and
    is what the engine stores under ``per_strategy_equity[name]["benchmark"]``.
    """
    mode = coerce_benchmark_mode(mode)
    if mode not in COMPUTED_BENCHMARK_MODES:
        return legacy_report(mode, buy_hold_pct)

    report = _base_report(mode, buy_hold_pct)
    start = _parse_ts(window_start)
    end = _parse_ts(window_end)
    strategy_returns = equity_daily_returns(strategy_equity)

    # ── Reported-only, mode-independent: per-trade net edge after costs ──
    mean_pnl, mean_pct = _per_trade_net_edge(trades)
    report["net_edge_per_trade"] = _optional_float(mean_pnl)
    report["net_edge_per_trade_pct"] = _optional_float(mean_pct)

    traded_symbols = {str(t.get("symbol")) for t in (trades or []) if t.get("symbol")}
    if start is None or end is None or end <= start:
        report["notes"] = "window unavailable — benchmark not computed"
        return report
    if not traded_symbols:
        # No trades ⇒ no exposure to match: benchmark 0, alpha = strategy return.
        # The gate still rejects such a genome on trade count / DSR / expectancy.
        report["benchmark_pct"] = 0.0
        report["benchmark_available"] = True
        report["strategy_time_in_market_pct"] = 0.0
        report["benchmark_time_in_market_pct"] = 0.0
        report["notes"] = ("no trades — the strategy had no exposure, so the "
                           "matched benchmark is 0 (the gate still rejects it on "
                           "the trade-count/DSR/net-expectancy criteria)")
        return report

    # ── Per-symbol in-market intervals + their buy & hold returns ──
    intervals: dict = {}
    symbol_returns: dict = {}
    symbol_windows: dict = {}
    per_symbol_share: dict = {}
    for symbol in symbols or []:
        frame = (frames or {}).get(symbol)
        if frame is None or len(frame) == 0:
            continue
        windows = frame[(frame.index >= start) & (frame.index <= end)]
        if len(windows) < 2:
            continue
        symbol_windows[symbol] = windows
        merged = in_market_intervals(trades, symbol, start, end)
        if not merged:
            continue
        ret = intervals_return(windows["close"].astype(float), merged)
        if ret is None:
            continue
        intervals[symbol] = merged
        symbol_returns[symbol] = float(ret)
        per_symbol_share[symbol] = (covered_bars(windows.index, merged) /
                                    float(len(windows)))
    if not symbol_windows:
        report["notes"] = ("no usable bars for the run's symbols in the window "
                           "(missing data) — benchmark unavailable")
        return report

    # ── Exposure-matched branch ──
    if mode == EXPOSURE_MATCHED:
        if not symbol_returns:
            report["notes"] = ("no usable in-market bars for the traded symbols "
                               "(missing data) — benchmark unavailable")
            report["strategy_time_in_market_pct"] = 0.0
            return report

        # Weights: the strategy's own deployed-capital shares (its relative
        # exposure, measured from its fills) — equal share of the basket when no
        # fill carries a usable notional.
        notionals: dict = {}
        for trade in trades or []:
            symbol = str(trade.get("symbol") or "")
            amount = _finite(trade.get("amount_usdt"))
            if symbol and amount > 0:
                notionals.setdefault(symbol, []).append(amount)
        balance = _finite(initial_balance, 10000.0)
        weights = {s: min(float(np.mean(v)) / balance, 1.0)
                   for s, v in notionals.items() if s in symbol_returns and balance > 0}
        weighting = "capital" if weights else "equal_share_of_basket"
        if not weights:
            weights = {s: 1.0 / max(len(symbols or []), 1) for s in symbol_returns}
        total_w = float(sum(weights.values()))
        report["symbol_weights"] = {s: float(w) for s, w in sorted(weights.items())}
        report["weighting"] = weighting

        # Strategy time-in-market on the union bar grid; benchmark share is the
        # exposure-weighted mean of the per-symbol covered shares.
        grid = None
        for windows in symbol_windows.values():
            grid = windows.index if grid is None else grid.union(windows.index)
        all_intervals = [iv for merged in intervals.values() for iv in merged]
        if grid is not None and len(grid):
            report["strategy_time_in_market_pct"] = float(
                covered_bars(grid, all_intervals) / float(len(grid)) * 100.0)
        report["benchmark_time_in_market_pct"] = float(
            sum(weights.get(s, 0.0) * per_symbol_share.get(s, 0.0)
                for s in symbol_returns) / total_w * 100.0) if total_w > 0 else None

        benchmark_series = basket_daily_returns(frames, weights, start, end,
                                               in_market=intervals)
        report["benchmark_pct"] = float(
            sum(weights[s] * symbol_returns[s] for s in symbol_returns) * 100.0)
        report["benchmark_available"] = True
        report["benchmark_sharpe"] = float(annualized_sharpe(benchmark_series))
        report["benchmark_max_dd_pct"] = float(
            max_drawdown_of_returns(benchmark_series))
        report["information_ratio"] = _optional_float(
            _information_ratio(strategy_returns, benchmark_series))
        beta, alpha = _jensen_alpha(strategy_returns, benchmark_series)
        report["benchmark_beta"] = _optional_float(beta)
        report["jensen_alpha_annual_pct"] = _optional_float(alpha)
        report["notes"] = (
            "exposure-matched basket held only during the strategy's own "
            f"in-market intervals; weights={weighting}, Σw={total_w:.4f}, "
            f"symbols={len(symbol_returns)}")
        return report

    # ── risk_matched branch (the fully-invested benchmark, scaled) ──
    basket_weights = {s: 1.0 / len(symbol_windows) for s in symbol_windows}
    basket = basket_daily_returns(frames, basket_weights, start, end)
    sigma_strategy = (float(strategy_returns.std(ddof=1))
                      if len(strategy_returns) > 1 else 0.0)
    sigma_benchmark = float(basket.std(ddof=1)) if len(basket) > 1 else 0.0
    if sigma_benchmark > 0 and sigma_strategy > 0:
        scale = sigma_strategy / sigma_benchmark
    else:
        scale = 1.0
        report["risk_scale_fallback"] = True
    report["risk_scale"] = float(scale)
    report["symbol_weights"] = {s: float(w) for s, w in sorted(basket_weights.items())}
    report["weighting"] = "equal_basket"
    scaled = basket * scale
    report["benchmark_pct"] = _optional_float(
        None if buy_hold_pct is None else _finite(buy_hold_pct) * scale)
    report["benchmark_available"] = report["benchmark_pct"] is not None
    report["benchmark_sharpe"] = float(annualized_sharpe(scaled))
    report["benchmark_max_dd_pct"] = float(max_drawdown_of_returns(scaled))
    report["benchmark_time_in_market_pct"] = 100.0
    report["information_ratio"] = _optional_float(
        _information_ratio(strategy_returns, scaled))
    beta, alpha = _jensen_alpha(strategy_returns, basket)
    report["benchmark_beta"] = _optional_float(beta)
    report["jensen_alpha_annual_pct"] = _optional_float(alpha)
    strategy_return = equity_total_return_pct(strategy_equity)
    if strategy_return is not None and scale > 0:
        report["strategy_risk_matched_pct"] = _optional_float(strategy_return / scale)
    if symbol_returns:
        grid = None
        for windows in symbol_windows.values():
            grid = windows.index if grid is None else grid.union(windows.index)
        all_intervals = [iv for merged in intervals.values() for iv in merged]
        if grid is not None and len(grid):
            report["strategy_time_in_market_pct"] = float(
                covered_bars(grid, all_intervals) / float(len(grid)) * 100.0)
    else:
        report["strategy_time_in_market_pct"] = 0.0
    report["notes"] = (
        "fully-invested benchmark scaled to the strategy's realised daily "
        f"volatility (σ_strategy={sigma_strategy:.6g}, "
        f"σ_benchmark={sigma_benchmark:.6g}, scale={scale:.4f})"
        + ("; σ=0 ⇒ raw benchmark used" if report["risk_scale_fallback"] else ""))
    return report


def reverse_risk_matched_return(strategy_return_pct, report: dict):
    """``strategy_return · σ_benchmark / σ_strategy`` — the other direction.

    Reported next to ``benchmark_pct`` so the equal-risk comparison is visible
    in both directions (scaling the benchmark to the strategy's risk and the
    strategy to the benchmark's risk give the same *ordering*, but the second
    number is the one an operator reads as "what would this strategy have
    returned at buy & hold's risk").
    """
    scale = (report or {}).get("risk_scale")
    if not scale or scale <= 0:
        return None
    return _optional_float(_finite(strategy_return_pct) / float(scale))


def strategy_vs_benchmark(report: dict, strategy_sharpe) -> dict:
    """The reported-only comparison block (never gated).

    ``sharpe`` (strategy, from the fidelity stats) next to ``benchmark_sharpe``,
    the information ratio, the Jensen alpha/beta and the per-trade net edge.
    """
    report = report or {}
    return {
        "strategy_sharpe": _optional_float(strategy_sharpe),
        "benchmark_sharpe": _optional_float(report.get("benchmark_sharpe")),
        "information_ratio": _optional_float(report.get("information_ratio")),
        "jensen_alpha_annual_pct": _optional_float(report.get("jensen_alpha_annual_pct")),
        "benchmark_beta": _optional_float(report.get("benchmark_beta")),
        "net_edge_per_trade": _optional_float(report.get("net_edge_per_trade")),
        "net_edge_per_trade_pct": _optional_float(report.get("net_edge_per_trade_pct")),
        "strategy_time_in_market_pct": _optional_float(
            report.get("strategy_time_in_market_pct")),
        "benchmark_time_in_market_pct": _optional_float(
            report.get("benchmark_time_in_market_pct")),
        "benchmark_max_dd_pct": _optional_float(report.get("benchmark_max_dd_pct")),
    }
