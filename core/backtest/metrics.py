"""Performance metrics calculator for backtesting results."""
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
        dict with keys: total_return_pct, annualized_return_pct, max_drawdown_pct,
        sharpe_ratio, win_rate_pct, profit_factor, total_trades, avg_pnl, avg_hold_minutes, days
    """
    n_trades = len([t for t in trades if t.get("exit_price") is not None])
    if n_trades == 0:
        return {"total_return_pct": 0, "total_trades": 0, "error": "No completed trades"}

    total_return_pct = (final_balance - initial_balance) / initial_balance * 100

    # Annualized return
    days = 1
    if equity_curve and len(equity_curve) >= 2:
        days = (pd.Timestamp(equity_curve[-1]["time"]) -
                pd.Timestamp(equity_curve[0]["time"])).days
        days = max(days, 1)
    annualized_return = ((1 + total_return_pct / 100) ** (365 / days) - 1) * 100

    # Max drawdown from peak equity
    equities = [p["equity"] for p in equity_curve]
    max_dd = 0.0
    if equities:
        peak = equities[0]
        for e in equities:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

    # Win rate & profit factor
    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) < 0]
    win_rate = len(winning) / n_trades * 100 if n_trades > 0 else 0
    total_gains = sum(t.get("pnl", 0) for t in winning)
    total_losses = abs(sum(t.get("pnl", 0) for t in losing))
    profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")

    # Sharpe ratio (annualized)
    # Resample to daily equity first to get correct annualization regardless
    # of the backtest's candle interval (1m, 5m, 1h, etc.)
    sharpe = 0.0
    if equity_curve and len(equity_curve) >= 2:
        eq_series = pd.Series(
            [p["equity"] for p in equity_curve],
            index=pd.DatetimeIndex([p["time"] for p in equity_curve]),
        )
        # Resample to daily: take last equity value of each day
        daily_eq = eq_series.resample("1D").last().dropna()
        if len(daily_eq) >= 2:
            daily_returns = daily_eq.pct_change().dropna()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)

    # Avg PnL and hold time
    avg_pnl = sum(t.get("pnl", 0) for t in trades) / n_trades if n_trades > 0 else 0
    avg_hold_minutes = 0.0
    closed_with_times = [t for t in trades if t.get("opened_at") and t.get("closed_at")]
    if closed_with_times:
        durations = [
            (pd.Timestamp(t["closed_at"]) - pd.Timestamp(t["opened_at"])).total_seconds() / 60
            for t in closed_with_times
        ]
        avg_hold_minutes = sum(durations) / len(durations)

    return {
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
    }
