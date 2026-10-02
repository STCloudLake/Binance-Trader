"""The single-chromosome scorer must report the REAL benchmark alpha.

``evaluate_chromosome`` used to call ``benchmark_result_fields(stats, score)``
with the **raw** ``stats`` dict while ``score_stats`` writes
``alpha_vs_benchmark_pct`` (and its siblings) into **its own copy** and returns
that copy.  The raw dict never carries the key, ``benchmark_result_fields`` read
``None`` and the ``score.update(...)`` put ``alpha_vs_benchmark_pct = 0.0`` back
over the real value.  The batch paths rebind ``stats = score_stats(...)`` and were
never affected — so the champion's ``provenance.validation`` block (the one field
that comes from the single-chromosome path,
``core/ga/evolver.py:869``) recorded 0.0 for every published champion.

The synthetic engine below makes the arithmetic exact (equity 10000 → 11250 is a
+12.5 % window, benchmark +2.0 %) so the assertion is not a rounding artefact:
``alpha_vs_benchmark_pct`` must be 10.5, and the batch path must agree with the
single path on the same candidate.
"""
from __future__ import annotations

import pytest

from core.ga import fitness as F
from core.ga.benchmark import EXPOSURE_MATCHED
from core.ga.genome import strategy_to_chromosome
from core.strategy.loader import MLConfig, StrategyConfig

INITIAL_BALANCE = 10000.0
FINAL_EQUITY = 11250.0                  # total_return_pct == 12.5, exactly
BENCHMARK_PCT = 2.0                     # alpha_vs_benchmark_pct must be 10.5
BUY_HOLD_PCT = 1.0                      # alpha_vs_buy_hold_pct must be 11.5

_BENCHMARK_REPORT = {
    "mode": EXPOSURE_MATCHED, "benchmark_pct": BENCHMARK_PCT,
    "buy_hold_pct": BUY_HOLD_PCT, "benchmark_available": True,
    "benchmark_sharpe": 1.2, "benchmark_max_dd_pct": 4.0,
    "benchmark_time_in_market_pct": 22.0, "strategy_time_in_market_pct": 30.0,
    "information_ratio": 0.4, "jensen_alpha_annual_pct": 1.1,
    "benchmark_beta": 0.3, "risk_scale": None, "risk_scale_fallback": False,
    "strategy_risk_matched_pct": None, "net_edge_per_trade": 2.5,
    "net_edge_per_trade_pct": 0.1, "symbol_weights": {"BTCUSDT": 0.2},
    "weighting": "capital", "notes": "stub",
}


class _SyntheticEngine:
    """One engine result per call — no backtest, no cached data, no file I/O."""

    config = None                       # executability model stays off

    def run_with_exit_evaluation(self, strategies, **kwargs):
        name = strategies[0].name
        trades = [
            {"strategy": name, "symbol": "BTCUSDT", "side": "long",
             "opened_at": f"2026-01-{i + 1:02d} 00:00",
             "closed_at": f"2026-01-{i + 1:02d} 12:00",
             "entry_price": 100.0, "exit_price": 101.0, "quantity": 1.0,
             "amount_usdt": 100.0, "cost": 1.0,
             "pnl": 30.0 if i % 3 else -10.0}
            for i in range(45)
        ]
        curve = [{"time": "2026-01-01 00:00", "equity": INITIAL_BALANCE},
                 {"time": "2026-01-02 00:00", "equity": 10600.0},
                 {"time": "2026-01-03 00:00", "equity": FINAL_EQUITY}]
        return {
            "metrics": {"buy_hold_pct": BUY_HOLD_PCT, "spread_sources": {}},
            "trades": trades,
            "equity_curve": curve,
            "per_strategy_equity": {
                name: {"trades": trades, "equity_curve": curve,
                       "benchmark": dict(_BENCHMARK_REPORT)}},
        }


def _chromosome(name: str = "alpha_probe") -> dict:
    config = StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=["1h"],
        indicators={"rsi": {"period": 14, "source": "close"}},
        entry_conditions={"long": ["close > sma"], "short": ["rsi > 70"]},
        exit_conditions={"long": ["close < sma"], "short": ["rsi < 30"]},
        ml_config=MLConfig(enabled=False),
    )
    return strategy_to_chromosome(config)


def _single(chrom, engine):
    return F.evaluate_chromosome(
        chrom, ["BTCUSDT"], "2026-01-01", "2026-01-31", engine, loader=None,
        initial_balance=INITIAL_BALANCE, n_trials=10,
        benchmark_mode=EXPOSURE_MATCHED)


def _batch(chrom, engine):
    population = [dict(chrom)]
    F.evaluate_population_batch(
        population, ["BTCUSDT"], "2026-01-01", "2026-01-31", engine, loader=None,
        batch_size=1, max_workers=1, initial_balance=INITIAL_BALANCE,
        batch_trials=10, benchmark_mode=EXPOSURE_MATCHED)
    return population[0]["fitness_result"]


def test_single_chromosome_reports_the_real_benchmark_alpha():
    """``evaluate_chromosome`` must not reset the scorer's alpha to 0.0."""
    score = _single(_chromosome(), _SyntheticEngine())

    assert score["total_return"] == pytest.approx(12.5)
    assert score["benchmark_pct"] == pytest.approx(BENCHMARK_PCT)
    # The regression: this was 0.0 for every single-chromosome (validation) call.
    assert score["alpha_vs_benchmark_pct"] == pytest.approx(
        score["total_return"] - score["benchmark_pct"], abs=1e-9)
    assert score["alpha_vs_benchmark_pct"] == pytest.approx(10.5)
    # ... and the sibling alpha the scorer already reported is untouched.
    assert score["alpha_vs_buy_hold_pct"] == pytest.approx(
        score["total_return"] - BUY_HOLD_PCT, abs=1e-9)


def test_batch_path_reports_the_same_benchmark_alpha_as_the_single_path():
    """The two scoring paths must agree on the benchmark block (they did not)."""
    engine = _SyntheticEngine()
    single = _single(_chromosome(), engine)
    batch = _batch(_chromosome(), engine)

    assert batch["total_return"] == pytest.approx(single["total_return"])
    assert batch["benchmark_pct"] == pytest.approx(single["benchmark_pct"])
    assert batch["alpha_vs_benchmark_pct"] == pytest.approx(
        batch["total_return"] - batch["benchmark_pct"], abs=1e-9)
    assert batch["alpha_vs_benchmark_pct"] == pytest.approx(
        single["alpha_vs_benchmark_pct"], abs=1e-9)
