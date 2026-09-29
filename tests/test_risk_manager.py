import pytest


def test_circuit_breaker_trips_on_drawdown():
    from core.risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(max_daily_drawdown_pct=5.0)
    cb.set_equity(10000)
    cb.set_equity(9400)
    tripped, reason = cb.check()
    assert tripped
    assert "drawdown" in reason.lower()


def test_circuit_breaker_trips_on_daily_loss():
    from core.risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(max_daily_loss_usdt=500)
    cb.add_trade_result(-300)
    cb.add_trade_result(-250)
    tripped, reason = cb.check()
    assert tripped
    assert "loss" in reason.lower()


def test_circuit_breaker_trips_on_consecutive_losses():
    from core.risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(max_consecutive_losses=5)
    for _ in range(5):
        cb.add_trade_result(-10)
    tripped, reason = cb.check()
    assert tripped
    assert "consecutive" in reason.lower()


def test_circuit_breaker_ok_with_profits():
    from core.risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker()
    cb.set_equity(10000)
    cb.add_trade_result(100)
    cb.add_trade_result(-50)
    cb.add_trade_result(200)
    tripped, reason = cb.check()
    assert not tripped
    assert cb.consecutive_losses == 0


def test_position_sizer():
    from core.risk.position_sizer import PositionSizer
    from app.config import HardRiskLimits, SoftRiskParams
    hard = HardRiskLimits()
    soft = SoftRiskParams()
    sizer = PositionSizer(hard, soft)
    qty, risk = sizer.calculate_position_size(10000, 50000, "core")
    assert qty > 0
    assert risk > 0


def test_stop_loss_calculation():
    from core.risk.position_sizer import PositionSizer
    from app.config import HardRiskLimits, SoftRiskParams
    hard = HardRiskLimits()
    soft = SoftRiskParams(stop_loss_pct=2.0)
    sizer = PositionSizer(hard, soft)
    sl = sizer.calculate_stop_loss(50000, "long")
    assert sl == 49000.0


def test_take_profits():
    from core.risk.position_sizer import PositionSizer
    from app.config import HardRiskLimits, SoftRiskParams
    hard = HardRiskLimits()
    soft = SoftRiskParams()
    sizer = PositionSizer(hard, soft)
    tps = sizer.calculate_take_profits(50000, "long")
    assert len(tps) == 3
    assert tps[0][0] == 51500.0


# ── Weekly drawdown guard (regression: the config advertised a weekly limit
#    that was never enforced anywhere in check()) ──────────────────────────

def test_circuit_breaker_trips_on_weekly_drawdown():
    from core.risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(max_daily_drawdown_pct=10.0, max_weekly_drawdown_pct=8.0)
    cb.set_equity(10000)          # weekly peak
    cb.set_equity(9000)           # -10% from weekly peak, within daily limit? no: daily peak also 10000
    tripped, reason = cb.check()
    assert tripped
    assert "weekly drawdown" in reason.lower()


def test_weekly_drawdown_survives_daily_reset():
    """A fresh daily peak must not hide a cumulative weekly slide."""
    from core.risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(max_daily_drawdown_pct=5.0, max_weekly_drawdown_pct=6.0)
    cb.set_equity(10000)
    cb.set_equity(9600)          # -4% week, -4% day: no trip
    assert cb.check()[0] is False
    cb.reset_daily()             # new day: daily peak moves down to 9600
    cb.set_equity(9300)          # -7% from the weekly peak (10000) but only -3.1% daily
    tripped, reason = cb.check()
    assert tripped, "weekly drawdown was masked by the daily peak reset"
    assert "weekly" in reason.lower()


def test_weekly_reset_clears_weekly_peak():
    from core.risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(max_daily_drawdown_pct=20.0, max_weekly_drawdown_pct=5.0)
    cb.set_equity(10000)
    cb.set_equity(9000)
    assert cb.check()[0] is True
    cb.reset_trip()
    cb.reset_weekly()            # new week starts from the current equity
    assert cb.check()[0] is False


def test_circuit_breaker_ok_within_both_limits():
    from core.risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(max_daily_drawdown_pct=5.0, max_weekly_drawdown_pct=10.0)
    cb.set_equity(10000)
    cb.set_equity(9700)          # -3% both
    cb.set_equity(9800)
    assert cb.check()[0] is False
