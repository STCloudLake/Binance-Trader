"""Tests for backtest metrics calculator and report generator."""
import pytest
from core.backtest.metrics import calculate_metrics
from core.backtest.report import generate_report, _compute_drawdown_curve, _compute_monthly_returns


class TestMetricsCalculator:
    def test_basic_metrics(self):
        trades = [
            {"symbol": "BTCUSDT", "pnl": 100, "exit_price": 50000,
             "opened_at": "2025-01-01", "closed_at": "2025-01-02"},
            {"symbol": "ETHUSDT", "pnl": -50, "exit_price": 3000,
             "opened_at": "2025-01-01", "closed_at": "2025-01-02"},
            {"symbol": "BNBUSDT", "pnl": 200, "exit_price": 600,
             "opened_at": "2025-01-01", "closed_at": "2025-01-02"},
        ]
        equity_curve = [
            {"time": "2025-01-01", "equity": 10000},
            {"time": "2025-01-02", "equity": 10250},
        ]
        metrics = calculate_metrics(trades, equity_curve, 10000, 10250)

        assert metrics["total_return_pct"] == 2.5
        assert metrics["total_trades"] == 3
        assert metrics["profit_factor"] > 1.0  # gains > losses
        assert metrics["days"] >= 1

    def test_no_trades(self):
        metrics = calculate_metrics([], [], 10000, 10000)
        assert metrics["total_trades"] == 0
        assert "error" in metrics

    def test_zero_equity_curve(self):
        trades = [{"symbol": "BTC", "pnl": 10, "exit_price": 50000,
                   "opened_at": "2025-01-01", "closed_at": "2025-01-02"}]
        equity_curve = [{"time": "2025-01-01", "equity": 10000}]
        metrics = calculate_metrics(trades, equity_curve, 10000, 10010)
        assert metrics["total_return_pct"] == 0.1

    def test_all_losing_trades(self):
        trades = [
            {"symbol": "A", "pnl": -100, "exit_price": 100,
             "opened_at": "2025-01-01", "closed_at": "2025-01-02"},
            {"symbol": "B", "pnl": -50, "exit_price": 200,
             "opened_at": "2025-01-03", "closed_at": "2025-01-04"},
        ]
        equity_curve = [
            {"time": "2025-01-01", "equity": 10000},
            {"time": "2025-01-04", "equity": 9850},
        ]
        metrics = calculate_metrics(trades, equity_curve, 10000, 9850)
        assert metrics["win_rate_pct"] == 0.0
        assert metrics["profit_factor"] == 0.0  # no gains, total_gains = 0


class TestReportGenerator:
    def test_generate_report(self):
        result = {
            "trades": [
                {"symbol": "BTCUSDT", "pnl": 50, "exit_price": 50000,
                 "entry_price": 49900, "side": "long", "strategy": "test"},
            ],
            "equity_curve": [
                {"time": "2025-01-01", "equity": 10000},
                {"time": "2025-01-02", "equity": 10050},
            ],
            "metrics": {
                "total_return_pct": 0.5, "sharpe_ratio": 1.2, "max_drawdown_pct": 2.0,
                "win_rate_pct": 100, "profit_factor": 2.0, "total_trades": 1,
                "avg_pnl": 50, "ml_accuracy_pct": 60,
            },
            "strategies": ["test"], "symbols": ["BTCUSDT"],
            "date_start": "2025-01-01", "date_end": "2025-01-02",
            "initial_balance": 10000, "final_balance": 10050,
        }
        report = generate_report(result)
        assert "summary" in report
        assert "chart_data" in report
        assert report["summary"]["total_return_pct"] == 0.5
        assert report["config"]["strategies"] == ["test"]

    def test_drawdown_computation(self):
        equity = [
            {"time": "t1", "equity": 10000},
            {"time": "t2", "equity": 9500},
            {"time": "t3", "equity": 9800},
            {"time": "t4", "equity": 9000},
        ]
        dd = _compute_drawdown_curve(equity)
        assert dd[0]["value"] == 0
        assert dd[1]["value"] == 5.0   # (10000-9500)/10000
        assert dd[2]["value"] == 2.0   # (10000-9800)/10000
        assert dd[3]["value"] == 10.0  # (10000-9000)/10000

    def test_peak_update(self):
        equity = [
            {"time": "t1", "equity": 10000},
            {"time": "t2", "equity": 11000},
            {"time": "t3", "equity": 10500},
        ]
        dd = _compute_drawdown_curve(equity)
        assert dd[0]["value"] == 0
        assert dd[1]["value"] == 0   # new peak, no drawdown
        assert dd[2]["value"] == pytest.approx(4.55, rel=0.1)  # (11000-10500)/11000

    def test_monthly_returns(self):
        equity = [
            {"time": "2025-01-01", "equity": 10000},
            {"time": "2025-01-15", "equity": 10100},
            {"time": "2025-02-01", "equity": 10200},
            {"time": "2025-02-15", "equity": 10098},
        ]
        monthly = _compute_monthly_returns(equity)
        assert len(monthly) == 2
        assert monthly[0]["month"] == "2025-01"
        assert monthly[1]["month"] == "2025-02"
