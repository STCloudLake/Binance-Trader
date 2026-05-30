"""Backtest report generator — produces structured JSON for Web UI display."""
import json


def generate_report(backtest_result: dict) -> dict:
    """Generate structured report data for Web UI display and JSON export.

    Args:
        backtest_result: Output dict from BacktestEngine.run_with_exit_evaluation()

    Returns:
        dict with keys: summary, chart_data, monthly_returns, pnl_distribution, config
    """
    metrics = backtest_result.get("metrics", {})
    trades = backtest_result.get("trades", [])
    equity_curve = backtest_result.get("equity_curve", [])

    summary = {
        "total_return_pct": metrics.get("total_return_pct", 0),
        "annualized_return_pct": metrics.get("annualized_return_pct", 0),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
        "win_rate_pct": metrics.get("win_rate_pct", 0),
        "profit_factor": metrics.get("profit_factor", 0),
        "total_trades": metrics.get("total_trades", 0),
        "avg_pnl": metrics.get("avg_pnl", 0),
        "avg_hold_minutes": metrics.get("avg_hold_minutes", 0),
        "ml_accuracy_pct": metrics.get("ml_accuracy_pct", 0),
    }

    chart_data = {
        "equity_curve": [{"time": p["time"], "value": p["equity"]} for p in equity_curve],
        "drawdown_curve": _compute_drawdown_curve(equity_curve),
    }

    monthly = _compute_monthly_returns(equity_curve)
    pnl_values = [t.get("pnl", 0) for t in trades if t.get("pnl") is not None]

    return {
        "summary": summary,
        "chart_data": chart_data,
        "monthly_returns": monthly,
        "pnl_distribution": pnl_values,
        "config": {
            "strategies": backtest_result.get("strategies", []),
            "symbols": backtest_result.get("symbols", []),
            "date_start": backtest_result.get("date_start", ""),
            "date_end": backtest_result.get("date_end", ""),
            "initial_balance": backtest_result.get("initial_balance", 0),
            "final_balance": backtest_result.get("final_balance", 0),
        },
    }


def _compute_drawdown_curve(equity_curve: list[dict]) -> list[dict]:
    """Compute drawdown percentage series from equity curve."""
    if not equity_curve:
        return []
    dd_curve = []
    peak = equity_curve[0]["equity"]
    for p in equity_curve:
        e = p["equity"]
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        dd_curve.append({"time": p["time"], "value": round(dd, 2)})
    return dd_curve


def _compute_monthly_returns(equity_curve: list[dict]) -> list[dict]:
    """Compute monthly return percentages from equity curve."""
    if len(equity_curve) < 2:
        return []
    monthly = {}
    for i in range(1, len(equity_curve)):
        t = equity_curve[i]["time"][:7]  # "YYYY-MM"
        prev_e = equity_curve[i - 1]["equity"]
        curr_e = equity_curve[i]["equity"]
        ret = (curr_e - prev_e) / prev_e * 100 if prev_e > 0 else 0
        monthly[t] = monthly.get(t, 0) + ret
    return [{"month": k, "return_pct": round(v, 2)}
            for k, v in sorted(monthly.items())]


def report_to_json(report: dict, indent: int = 2) -> str:
    """Serialize report to JSON string."""
    return json.dumps(report, ensure_ascii=False, indent=indent, default=str)
