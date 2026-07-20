"""Performance metrics calculator for backtesting results.

Calculates standard trading metrics plus risk-adjusted measures:
- Core: total return, annualized return, max drawdown, Sharpe, win rate, profit factor
- Risk: Sortino, Calmar, VaR, CVaR, Omega, tail ratio
- Trade quality: consecutive losses, recovery factor, avg hold time

All metrics are derived from trades + equity curve.  The daily return series
is resampled from the equity curve and reused across multiple calculations.
"""

import numpy as np
import pandas as pd


def calculate_metrics(trades: list[dict], equity_curve: list[dict],
                      initial_balance: float, final_balance: float) -> dict:
    """Compute all performance metrics from backtest output.

    Args:
        trades: List of trade dicts with keys: pnl, exit_price, opened_at, closed_at
        equity_curve: List of dicts with keys: time, equity
        initial_balance: Starting account balance
        final_balance: Ending account balance

    Returns:
        dict with ~19 keys covering return, risk, and trade-quality dimensions.
        Backward compatible — all previous callers continue to work.
    """
    n_trades = len([t for t in trades if t.get("exit_price") is not None])
    if n_trades == 0:
        return {"total_return_pct": 0, "total_trades": 0,
                "error": "No completed trades"}

    # ── Daily return series (reused by Sharpe, Sortino, VaR, CVaR, Omega) ──
    days = 1
    daily_returns: np.ndarray = np.array([])
    if equity_curve and len(equity_curve) >= 2:
        eq_series = pd.Series(
            [p["equity"] for p in equity_curve],
            index=pd.DatetimeIndex([p["time"] for p in equity_curve]),
        )
        daily_eq = eq_series.resample("1D").last().dropna()
        if len(daily_eq) >= 2:
            days = (pd.Timestamp(equity_curve[-1]["time"]) -
                    pd.Timestamp(equity_curve[0]["time"])).days
            days = max(days, 1)
            daily_returns = daily_eq.pct_change().dropna().values

    total_return_pct = (final_balance - initial_balance) / initial_balance * 100

    # ── Annualized return ──
    annualized_return = ((1 + total_return_pct / 100) ** (365 / days) - 1) * 100

    # ── Max drawdown from peak equity ──
    equities = np.array([p["equity"] for p in equity_curve])
    max_dd = 0.0
    if len(equities) > 0:
        peak = equities[0]
        for e in equities:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

    # ── Win rate & profit factor ──
    pnl_values = np.array([t.get("pnl", 0) for t in trades])
    winning = pnl_values[pnl_values > 0]
    losing = pnl_values[pnl_values < 0]
    win_rate = len(winning) / n_trades * 100 if n_trades > 0 else 0
    total_gains = float(winning.sum()) if len(winning) > 0 else 0.0
    total_losses = float(abs(losing.sum())) if len(losing) > 0 else 0.0
    profit_factor = (total_gains / total_losses if total_losses > 0
                     else (999.0 if total_gains > 0 else 0.0))

    # ── Sharpe ratio (annualized, daily-resampled) ──
    sharpe = _annualized_sharpe(daily_returns)

    # ── Sortino ratio (annualized, downside deviation only) ──
    sortino = _annualized_sortino(daily_returns)

    # ── Calmar ratio (annualized return / max drawdown) ──
    calmar = annualized_return / max_dd if max_dd > 0 else 0.0

    # ── VaR & CVaR (historical simulation, daily returns) ──
    var_95, cvar_95 = _var_cvar(daily_returns, confidence=0.95)
    var_99, cvar_99 = _var_cvar(daily_returns, confidence=0.99)

    # ── Omega ratio (E[gain] / E[loss], threshold = 0) ──
    omega = _omega_ratio(daily_returns)

    # ── Tail ratio (95th pct positive return / abs(5th pct negative)) ──
    tail = _tail_ratio(daily_returns)

    # ── Consecutive losses ──
    max_consec_losses = _max_consecutive_losses(pnl_values)

    # ── Recovery factor (absolute return / max drawdown in USDT) ──
    max_dd_usdt = initial_balance * max_dd / 100
    recovery_factor = abs(final_balance - initial_balance) / max_dd_usdt if max_dd_usdt > 0 else 0.0

    # ── Avg PnL and hold time ──
    avg_pnl = float(pnl_values.sum()) / n_trades if n_trades > 0 else 0.0
    avg_hold_minutes = 0.0
    closed_with_times = [t for t in trades if t.get("opened_at") and t.get("closed_at")]
    if closed_with_times:
        durations = [
            (pd.Timestamp(t["closed_at"]) - pd.Timestamp(t["opened_at"])).total_seconds() / 60
            for t in closed_with_times
        ]
        avg_hold_minutes = sum(durations) / len(durations)

    return {
        # ── Core (existing keys — backward compatible) ──
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(annualized_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "total_trades": n_trades,
        "avg_pnl": round(avg_pnl, 2),
        "avg_hold_minutes": round(avg_hold_minutes, 0),
        "days": days,

        # ── Risk-adjusted (new) ──
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "var_95_daily_pct": round(var_95, 4),
        "cvar_95_daily_pct": round(cvar_95, 4),
        "var_99_daily_pct": round(var_99, 4),
        "cvar_99_daily_pct": round(cvar_99, 4),
        "max_consecutive_losses": max_consec_losses,
        "recovery_factor": round(recovery_factor, 2),
        "omega_ratio": round(omega, 2),
        "tail_ratio": round(tail, 2),
    }


# ── Helper functions ─────────────────────────────────────────────────────

def _annualized_sharpe(daily_returns: np.ndarray) -> float:
    """Annualized Sharpe from daily returns."""
    if len(daily_returns) < 2 or daily_returns.std() <= 0:
        return 0.0
    return float((daily_returns.mean() / daily_returns.std()) * np.sqrt(365))


def _annualized_sortino(daily_returns: np.ndarray) -> float:
    """Annualized Sortino — uses downside deviation only."""
    if len(daily_returns) < 2:
        return 0.0
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2 or downside.std() <= 0:
        return 0.0
    return float((daily_returns.mean() / downside.std()) * np.sqrt(365))


def _var_cvar(daily_returns: np.ndarray,
              confidence: float = 0.95) -> tuple[float, float]:
    """Historical VaR and CVaR (Expected Shortfall).

    Args:
        daily_returns: Array of daily return fractions (e.g. 0.01 = +1%).
        confidence: Confidence level (0.95 → 95% VaR).

    Returns:
        (var_pct, cvar_pct) expressed as percentages of portfolio value.
        var_pct is negative (loss).  cvar_pct is *also* negative (mean
        loss beyond VaR).
    """
    if len(daily_returns) < 20:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    var = float(np.percentile(daily_returns, alpha * 100))  # e.g. 5th percentile
    # CVaR = mean of returns below VaR threshold
    tail_returns = daily_returns[daily_returns <= var]
    cvar = float(tail_returns.mean()) if len(tail_returns) > 0 else var
    return round(var * 100, 4), round(cvar * 100, 4)


def _omega_ratio(daily_returns: np.ndarray, threshold: float = 0.0) -> float:
    """Omega ratio: E[gain above threshold] / E[loss below threshold]."""
    if len(daily_returns) < 2:
        return 0.0
    gains = daily_returns[daily_returns > threshold]
    losses = daily_returns[daily_returns < threshold]
    avg_gain = gains.mean() if len(gains) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
    return float(avg_gain / avg_loss) if avg_loss > 0 else 999.0


def _tail_ratio(daily_returns: np.ndarray) -> float:
    """Tail ratio: 95th pct positive return / |5th pct negative return|."""
    if len(daily_returns) < 50:
        return 0.0
    p95 = float(np.percentile(daily_returns, 95))
    p05 = float(np.percentile(daily_returns, 5))
    denom = abs(p05) if p05 != 0 else 1e-9
    return p95 / denom


def _max_consecutive_losses(pnl_values: np.ndarray) -> int:
    """Longest streak of consecutive losing trades."""
    max_streak = 0
    current = 0
    for pnl in pnl_values:
        if pnl < 0:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak


# ── Benchmark comparison ─────────────────────────────────────────────────

def calculate_benchmark_metrics(
    equity_curve: list[dict],
    benchmark_equity: list[dict],
    risk_free_annual: float = 0.02,
) -> dict:
    """Compare strategy returns against a benchmark (e.g. BTC buy-and-hold).

    Args:
        equity_curve: Strategy equity curve [{time, equity}, ...].
        benchmark_equity: Benchmark equity curve [{time, equity}, ...].
        risk_free_annual: Annual risk-free rate (default 2%).

    Returns:
        dict with alpha, beta, information_ratio, tracking_error.
        Returns zeros if data is insufficient.
    """
    if len(equity_curve) < 2 or len(benchmark_equity) < 2:
        return {"alpha": 0.0, "beta": 0.0,
                "information_ratio": 0.0, "tracking_error": 0.0}

    # Resample both to daily
    eq_s = pd.Series(
        [p["equity"] for p in equity_curve],
        index=pd.DatetimeIndex([p["time"] for p in equity_curve]),
    ).resample("1D").last().dropna()

    bm_s = pd.Series(
        [p["equity"] for p in benchmark_equity],
        index=pd.DatetimeIndex([p["time"] for p in benchmark_equity]),
    ).resample("1D").last().dropna()

    # Align on common dates
    common_idx = eq_s.index.intersection(bm_s.index)
    if len(common_idx) < 30:
        return {"alpha": 0.0, "beta": 0.0,
                "information_ratio": 0.0, "tracking_error": 0.0}

    strategy_ret = eq_s.loc[common_idx].pct_change().dropna()
    benchmark_ret = bm_s.loc[common_idx].pct_change().dropna()

    common = strategy_ret.index.intersection(benchmark_ret.index)
    if len(common) < 20:
        return {"alpha": 0.0, "beta": 0.0,
                "information_ratio": 0.0, "tracking_error": 0.0}

    s = strategy_ret.loc[common].values
    b = benchmark_ret.loc[common].values

    # Beta = Cov(s, b) / Var(b)
    cov = np.cov(s, b)[0, 1]
    var_b = np.var(b)
    beta = cov / var_b if var_b > 0 else 0.0

    # Alpha = annualized excess return not explained by beta
    ann_s = float(s.mean() * 365)
    ann_b = float(b.mean() * 365)
    alpha = ann_s - (risk_free_annual + beta * (ann_b - risk_free_annual))

    # Tracking error & information ratio
    excess = s - b
    tracking_error = float(excess.std() * np.sqrt(365))
    ir = (excess.mean() * 365) / tracking_error if tracking_error > 0 else 0.0

    return {
        "alpha": round(alpha, 4),
        "beta": round(beta, 2),
        "information_ratio": round(ir, 2),
        "tracking_error": round(tracking_error, 4),
    }
