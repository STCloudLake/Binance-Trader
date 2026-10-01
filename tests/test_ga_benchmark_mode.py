"""Selectable GA benchmark (``ga.benchmark_mode``) + evaluation/execution parity.

The gate used to reject a champion on ``alpha_vs_buy_hold_pct`` alone, where the
benchmark is a **100 %-invested** buy & hold of the window
(``core/backtest/engine.py``): a strategy that is in the market a fraction of
the time was compared against a fully-invested basket by raw total return, with
no exposure or risk matching.  These tests pin the four properties the feature
must have:

* ``buy_hold`` — the code default — is **byte-identical** to HEAD: the scored
  values, the gate verdict and the legacy provenance keys are compared against a
  ``git worktree`` at the pre-change revision;
* each mode's benchmark is computable by hand on a synthetic trade/equity series
  (a known up-move, disjoint in-market intervals, an open position at the window
  end, zero trades, missing bars);
* an unknown mode fails **at config load and at job load** with a named error;
* a champion's recorded basket reaches the execution path, so an enabled
  champion can only trade what it was evaluated on.

Nothing here runs a real GA: the batch scorer is stubbed where a run is needed
(same technique as ``tests/test_ga_timeframe_pool.py``).
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import Config
from core.ga.benchmark import (
    BENCHMARK_MODES,
    BUY_HOLD,
    EXPOSURE_MATCHED,
    NONE,
    RISK_MATCHED,
    BenchmarkModeError,
    UnknownBenchmarkModeError,
    build_benchmark,
    equity_daily_returns,
    in_market_intervals,
    intervals_return,
    merge_intervals,
    parse_benchmark_mode,
)
from core.ga.evolver import GARunConfig, GAStrategyEvolver
from core.strategy.loader import StrategyLoader

ROOT = Path(__file__).resolve().parents[1]
BASELINE_REVISION = "ba8c212"          # the pre-``benchmark_mode`` revision


# ══════════════════════════════════════════════════════════════════════
# synthetic series helpers (hand-computable by construction)
# ══════════════════════════════════════════════════════════════════════

def _frame(closes, start="2026-01-01", freq="1D"):
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame({"close": [float(c) for c in closes]}, index=idx)


def _equity(values, start="2026-01-01", step="1D"):
    idx = pd.date_range(start, periods=len(values), freq=step)
    return [{"time": str(ts), "equity": float(v)} for ts, v in zip(idx, values)]


def _trade(symbol, opened, closed, amount=5000.0, pnl=10.0):
    return {"symbol": symbol, "opened_at": opened, "closed_at": closed,
            "amount_usdt": amount, "pnl": pnl, "side": "long", "cost": 1.0}


W0, W1 = "2026-01-01 00:00", "2026-01-05 23:59"


def _exposure(frames, trades, symbols, buy_hold=10.0, balance=10000.0):
    return build_benchmark(
        EXPOSURE_MATCHED, trades=trades, symbols=symbols, frames=frames,
        window_start=W0, window_end=W1, initial_balance=balance,
        strategy_equity=_equity([balance, balance]), buy_hold_pct=buy_hold)


# ══════════════════════════════════════════════════════════════════════
# (a) mode semantics on hand-computable series
# ══════════════════════════════════════════════════════════════════════

def test_exposure_matched_equals_the_known_up_move_of_the_held_interval():
    """In market only during a known +21 % move ⇒ benchmark = w · 21 %.

    UP closes 100 → 110 → 121 → 133.1 on four consecutive days; the strategy
    holds [day1, day3] with 5 000 of a 10 000 account, so the benchmark holds
    the same symbol for the same interval at the same 0.5 exposure:
    0.5 · 21 % = 10.5 %.
    """
    frames = {"UP": _frame([100, 110, 121, 133.1])}
    trades = [_trade("UP", "2026-01-01 00:00", "2026-01-03 00:00", amount=5000.0)]
    report = _exposure(frames, trades, ["UP"])
    assert report["benchmark_pct"] == pytest.approx(10.5)
    assert report["weighting"] == "capital"
    assert report["symbol_weights"] == {"UP": pytest.approx(0.5)}
    # 3 of the 4 window bars are covered by the interval.
    assert report["strategy_time_in_market_pct"] == pytest.approx(75.0)
    assert report["benchmark_time_in_market_pct"] == pytest.approx(75.0)
    assert report["benchmark_available"] is True


def test_exposure_matched_ignores_symbols_the_strategy_never_held():
    """Relative exposure: an untraded symbol cannot dilute the benchmark."""
    frames = {"UP": _frame([100, 110, 121, 133.1]),
              "FLAT": _frame([100, 100, 100, 100])}
    trades = [_trade("UP", "2026-01-01 00:00", "2026-01-03 00:00")]
    both = _exposure(frames, trades, ["UP", "FLAT"])
    only = _exposure({"UP": frames["UP"]}, trades, ["UP"])
    assert both["benchmark_pct"] == pytest.approx(10.5)
    assert both["benchmark_pct"] == pytest.approx(only["benchmark_pct"])
    assert "FLAT" not in both["symbol_weights"]


def test_exposure_matched_compounds_disjoint_intervals_and_merges_overlaps():
    """A gap between two trades is NOT held; touching trades are one interval."""
    # closes 100 110 100 121 133.1 — the dip at day 3 is only held if a trade
    # covers it.
    frames = {"S": _frame([100, 110, 100, 121, 133.1])}
    disjoint = [_trade("S", "2026-01-01 00:00", "2026-01-02 00:00", amount=10000.0),
                _trade("S", "2026-01-04 00:00", "2026-01-05 00:00", amount=10000.0)]
    report = _exposure(frames, disjoint, ["S"], balance=10000.0)
    # day1→day2 = +10 %, day4→day5 = +10 % ⇒ 1.1 · 1.1 − 1 = 21 %, and the
    # fully-invested window return would be 33.1 % (the dip is not held).
    full = _frame([100, 110, 100, 121, 133.1])
    assert intervals_return(full["close"], [(pd.Timestamp("2026-01-01"),
                                             pd.Timestamp("2026-01-05"))]) \
        == pytest.approx(0.331, abs=1e-9)
    assert report["benchmark_pct"] == pytest.approx(21.0)

    overlapping = [_trade("S", "2026-01-01 00:00", "2026-01-03 00:00", amount=10000.0),
                   _trade("S", "2026-01-02 00:00", "2026-01-05 00:00", amount=10000.0)]
    assert merge_intervals(in_market_intervals(
        overlapping, "S", pd.Timestamp(W0), pd.Timestamp(W1))) == \
        [(pd.Timestamp("2026-01-01 00:00"), pd.Timestamp("2026-01-05 00:00"))]
    # One merged interval ⇒ the whole 100 → 133.1 move, not two overlapping legs.
    merged = _exposure(frames, overlapping, ["S"])
    assert merged["benchmark_pct"] == pytest.approx(33.1)


def test_an_open_position_is_clipped_to_the_window_end():
    frames = {"S": _frame([100, 110, 121, 133.1])}
    trades = [{"symbol": "S", "opened_at": "2026-01-03 00:00", "closed_at": None,
               "amount_usdt": 10000.0, "pnl": 0.0}]
    merged = in_market_intervals(trades, "S", pd.Timestamp(W0), pd.Timestamp(W1))
    assert merged == [(pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-05 23:59"))]
    report = _exposure(frames, trades, ["S"])
    # day3 (121) → day4 (133.1) = +10 %.
    assert report["benchmark_pct"] == pytest.approx(10.0, abs=1e-6)


def test_zero_trades_gives_a_zero_benchmark_and_alpha_is_the_strategy_return():
    report = _exposure({"S": _frame([100, 110, 121, 133.1])}, [], ["S"])
    assert report["benchmark_pct"] == 0.0
    assert report["benchmark_available"] is True
    assert report["strategy_time_in_market_pct"] == 0.0
    assert "no trades" in report["notes"]


def test_missing_bars_make_the_benchmark_unavailable():
    trades = [_trade("S", "2026-01-01 00:00", "2026-01-03 00:00")]
    report = _exposure({}, trades, ["S"])
    assert report["benchmark_pct"] is None
    assert report["benchmark_available"] is False
    assert "unavailable" in report["notes"]
    # A one-bar "window" cannot express a return either.
    assert _exposure({"S": _frame([100])}, trades, ["S"])["benchmark_pct"] is None


def test_risk_matched_scales_the_benchmark_to_the_strategy_volatility():
    """scale = σ_strategy / σ_benchmark, both from the daily series."""
    closes = [100, 102, 100, 102, 100, 102]
    frames = {"S": _frame(closes)}
    # A strategy whose own equity moves half as much per day.
    equity = _equity([10000, 10100, 10000, 10100, 10000, 10100])
    trades = [_trade("S", "2026-01-01 00:00", "2026-01-05 00:00")]
    report = build_benchmark(RISK_MATCHED, trades=trades, symbols=["S"],
                             frames=frames, window_start=W0, window_end=W1,
                             initial_balance=10000.0, strategy_equity=equity,
                             buy_hold_pct=12.5)
    bench_rets = _frame(closes)["close"]
    bench_rets = bench_rets[(bench_rets.index >= pd.Timestamp(W0))
                            & (bench_rets.index <= pd.Timestamp(W1))]
    bench_rets = bench_rets.pct_change().dropna()
    strat_rets = equity_daily_returns(equity)
    expected = float(strat_rets.std(ddof=1) / bench_rets.std(ddof=1))
    assert report["risk_scale"] == pytest.approx(expected, rel=1e-9)
    assert report["benchmark_pct"] == pytest.approx(12.5 * expected, rel=1e-9)
    # ... and the reverse direction is reported too.
    assert report["strategy_risk_matched_pct"] is not None
    assert report["benchmark_time_in_market_pct"] == 100.0


def test_risk_matched_falls_back_to_the_raw_benchmark_without_benchmark_risk():
    frames = {"S": _frame([100, 100, 100, 100])}
    report = build_benchmark(RISK_MATCHED, trades=[_trade("S", "2026-01-01 00:00",
                                                          "2026-01-03 00:00")],
                             symbols=["S"], frames=frames, window_start=W0,
                             window_end=W1, initial_balance=10000.0,
                             strategy_equity=_equity([10000, 10000]),
                             buy_hold_pct=7.0)
    assert report["risk_scale"] == 1.0
    assert report["risk_scale_fallback"] is True
    assert report["benchmark_pct"] == pytest.approx(7.0)


def test_buy_hold_and_none_are_reported_without_touching_the_prices():
    """The default path is a pure copy of the legacy value (no frame needed)."""
    buy_hold = build_benchmark(BUY_HOLD, trades=[], symbols=[], frames={},
                               window_start=W0, window_end=W1,
                               buy_hold_pct=17.739)
    assert buy_hold["benchmark_pct"] == 17.739
    assert buy_hold["benchmark_available"] is True
    assert buy_hold["benchmark_time_in_market_pct"] == 100.0
    none = build_benchmark(NONE, trades=[], symbols=[], frames={},
                           window_start=W0, window_end=W1, buy_hold_pct=17.739)
    assert none["benchmark_pct"] is None
    assert none["benchmark_available"] is False


def test_mode_parsing_is_the_code_default_and_rejects_unknown_values():
    assert BENCHMARK_MODES == (BUY_HOLD, EXPOSURE_MATCHED, RISK_MATCHED, NONE)
    assert parse_benchmark_mode(None) == BUY_HOLD          # key absent
    assert parse_benchmark_mode("") == BUY_HOLD            # blank
    assert parse_benchmark_mode(" Risk_Matched ") == RISK_MATCHED
    with pytest.raises(UnknownBenchmarkModeError) as excinfo:
        parse_benchmark_mode("buy-and-hold")
    assert "buy-and-hold" in str(excinfo.value)
    assert "accepted:" in str(excinfo.value)
    with pytest.raises(BenchmarkModeError):
        parse_benchmark_mode(3)


def test_daily_series_matches_the_fidelity_helper():
    """The report's series is the same one ``core.ga.fitness`` builds."""
    from core.ga.fitness import daily_returns, per_period_sharpe

    curve = _equity([10000, 10050, 10020, 10100, 10090, 10150])
    assert np.allclose(equity_daily_returns(curve).values, daily_returns(curve))
    rets = equity_daily_returns(curve).values
    assert per_period_sharpe(rets) == pytest.approx(
        float(rets.mean() / rets.std(ddof=1)))


# ══════════════════════════════════════════════════════════════════════
# (b) the gate consumes the selected mode's alpha
# ══════════════════════════════════════════════════════════════════════

def _gate_evolver(tmp_path):
    class _Cfg:
        ga_min_champion_trades = 30

    class _Engine:
        config = _Cfg()

    loader = StrategyLoader(str(tmp_path / "gate_strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    return GAStrategyEvolver(_Engine(), loader, GARunConfig(population_size=2))


def _train_result(**overrides):
    result = {"trade_count": 40, "total_return": 20.0, "profit_factor": 2.0,
              "dsr": 0.5, "buy_hold_pct": 10.0, "alpha_vs_buy_hold_pct": 10.0}
    result.update(overrides)
    return result


def test_the_gate_keeps_the_legacy_rejection_reason_verbatim(tmp_path):
    """A result dict without the new field is the pre-change ``buy_hold`` case."""
    evolver = _gate_evolver(tmp_path)
    published, reasons = evolver._publication_decision(
        _train_result(alpha_vs_buy_hold_pct=-1.0), None)
    assert published is False
    assert "alpha_vs_buy_hold=-1.00% <= 0 (no edge over buy & hold)" in reasons


def test_the_gate_consumes_the_exposure_matched_alpha(tmp_path):
    """A positive matched alpha publishes even when raw buy & hold was beaten..."""
    evolver = _gate_evolver(tmp_path)
    published, reasons = evolver._publication_decision(
        _train_result(benchmark_mode=EXPOSURE_MATCHED, benchmark_pct=5.0,
                      alpha_vs_benchmark_pct=15.0, alpha_vs_buy_hold_pct=-5.0), None)
    assert published is True, reasons
    assert reasons == []

    published, reasons = evolver._publication_decision(
        _train_result(benchmark_mode=EXPOSURE_MATCHED, benchmark_pct=25.0,
                      alpha_vs_benchmark_pct=-2.5, alpha_vs_buy_hold_pct=8.0), None)
    assert published is False
    assert ("alpha_vs_exposure_matched=-2.50% <= 0 "
            "(no edge over the exposure-matched benchmark)") in reasons
    # ... and the raw buy & hold alpha is NOT what rejected it.
    assert not any("alpha_vs_buy_hold" in r for r in reasons)


def test_risk_matched_gates_on_its_own_alpha(tmp_path):
    evolver = _gate_evolver(tmp_path)
    published, reasons = evolver._publication_decision(
        _train_result(benchmark_mode=RISK_MATCHED, benchmark_pct=1.0,
                      alpha_vs_benchmark_pct=-0.5, alpha_vs_buy_hold_pct=20.0), None)
    assert published is False
    assert any(r.startswith("alpha_vs_risk_matched=-0.50%") for r in reasons)


def test_none_never_emits_a_benchmark_rejection_reason(tmp_path):
    evolver = _gate_evolver(tmp_path)
    published, reasons = evolver._publication_decision(
        _train_result(benchmark_mode=NONE, benchmark_pct=None,
                      alpha_vs_benchmark_pct=0.0, alpha_vs_buy_hold_pct=-99.0), None)
    assert published is True, reasons
    assert reasons == []

    # ... but the other criteria still gate: `none` disables ONLY the benchmark.
    published, reasons = evolver._publication_decision(
        _train_result(benchmark_mode=NONE, benchmark_pct=None, dsr=-0.3855,
                      alpha_vs_benchmark_pct=0.0, alpha_vs_buy_hold_pct=-99.0), None)
    assert published is False
    assert "dsr=-0.3855 <= 0 (indistinguishable from data mining)" in reasons
    assert not any("no edge over" in r for r in reasons)


def test_score_stats_reports_both_alphas_and_is_bit_identical_for_buy_hold():
    """``buy_hold`` ⇒ ``alpha_vs_benchmark_pct`` is the same float, same way."""
    from core.ga.fitness import score_stats, stats_from_trades

    stats = stats_from_trades([{"pnl": 0.0, "side": "long"}],
                              _equity([0.0]), 10000.0)
    stats["buy_hold_pct"] = 12.3456789
    stats["total_return_pct"] = 20.0
    stats["benchmark_mode"] = BUY_HOLD
    stats["benchmark_pct"] = 12.3456789
    scored = score_stats(dict(stats), None, n_trials=100)
    assert scored["alpha_vs_benchmark_pct"] == scored["alpha_vs_buy_hold_pct"]
    assert scored["benchmark_pct"] == 12.3456789

    # No benchmark fields at all (a hand-built stats dict) ⇒ legacy behaviour.
    legacy = score_stats({"trades": 5, "win_rate": 50.0, "profit_factor": 1.5,
                          "pnl": 10.0, "total_return_pct": 20.0,
                          "buy_hold_pct": 12.3456789, "max_dd_pct": 2.0,
                          "sharpe": 1.0}, None, n_trials=100)
    assert legacy["alpha_vs_benchmark_pct"] == legacy["alpha_vs_buy_hold_pct"]
    assert legacy["benchmark_mode"] == BUY_HOLD

    # `none` reports no benchmark and no benchmark alpha.
    stats["benchmark_mode"] = NONE
    stats["benchmark_pct"] = None
    none_scored = score_stats(dict(stats), None, n_trials=100)
    assert none_scored["benchmark_pct"] is None
    assert none_scored["alpha_vs_benchmark_pct"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# (c) unknown mode → named error at CONFIG load and at JOB load
# ══════════════════════════════════════════════════════════════════════

def test_config_reader_exposes_the_shipped_mode(tmp_path, monkeypatch):
    """The key has a reader, and the shipped value is a valid mode."""
    import yaml

    real = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    shipped = (real.get("ga") or {}).get("benchmark_mode")
    Config._instance = None
    try:
        config = Config.load("sim")
        assert config.ga_benchmark_mode == parse_benchmark_mode(shipped)
        assert config.ga_benchmark_mode in BENCHMARK_MODES
        # The recommended mode is the one shipped (the deliverable's choice).
        assert config.ga_benchmark_mode == EXPOSURE_MATCHED
    finally:
        Config._instance = None


def test_unknown_mode_is_rejected_at_config_load(tmp_path, monkeypatch):
    import yaml
    import app.config as config_mod

    data = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    data.setdefault("ga", {})["benchmark_mode"] = "buy-and-hold"
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)

    Config._instance = None
    try:
        with pytest.raises(UnknownBenchmarkModeError) as excinfo:
            Config.load("sim")
        assert "buy-and-hold" in str(excinfo.value)
    finally:
        Config._instance = None


def _worker_module():
    spec = importlib.util.spec_from_file_location(
        "ga_worker_benchmark_mode_test", ROOT / "scripts" / "ga_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_job(tmp_path, name, job):
    path = tmp_path / name
    path.write_text(json.dumps(job), encoding="utf-8")
    return path


def test_worker_rejects_an_unknown_mode_at_job_load(tmp_path, monkeypatch):
    worker = _worker_module()
    job_file = _write_job(tmp_path, "ga_bad_mode.json", {
        "symbols": ["BTCUSDT"], "population_size": 4, "generations": 1,
        "benchmark_mode": "sharpe_matched"})

    monkeypatch.setattr(sys, "argv", [
        "ga_worker.py", "--job-type", "ga", "--job-file", str(job_file)])
    worker.main()

    result = json.loads(Path(str(job_file) + ".result").read_text())
    assert result["error_type"] == "UnknownBenchmarkModeError"
    assert "benchmark_mode" in result["error"]
    assert "sharpe_matched" in result["error"]
    assert "UnknownBenchmarkModeError" in result["traceback"]
    # Failed at LOAD: no progress file was ever started, no GA work happened.
    assert not Path(str(job_file) + ".progress").exists()


def test_worker_accepts_and_forwards_a_valid_mode():
    worker = _worker_module()
    assert worker.job_benchmark_mode({"benchmark_mode": "risk_matched"}) == RISK_MATCHED
    assert worker.job_benchmark_mode({}) is None          # follow the config
    assert worker.job_benchmark_mode({"benchmark_mode": "   "}) is None
    with pytest.raises(UnknownBenchmarkModeError):
        worker.job_benchmark_mode({"benchmark_mode": "nope"})

    cfg = worker.ga_run_config({"benchmark_mode": " risk_matched "}, 20, 5, 4242)
    assert cfg.benchmark_mode == RISK_MATCHED
    # An absent field is NOT an override: the run follows `ga.benchmark_mode`.
    assert worker.ga_run_config({}, 20, 5, 1).benchmark_mode is None


# ══════════════════════════════════════════════════════════════════════
# (d) provenance + evaluation/execution consistency
# ══════════════════════════════════════════════════════════════════════

_STUB_BENCHMARK = {
    "mode": EXPOSURE_MATCHED, "benchmark_pct": 5.0, "buy_hold_pct": 3.0,
    "benchmark_available": True, "benchmark_sharpe": 1.2,
    "benchmark_max_dd_pct": 4.0, "benchmark_time_in_market_pct": 22.0,
    "strategy_time_in_market_pct": 30.0, "information_ratio": 0.4,
    "jensen_alpha_annual_pct": 1.1, "benchmark_beta": 0.3,
    "risk_scale": None, "risk_scale_fallback": False,
    "strategy_risk_matched_pct": None, "net_edge_per_trade": 2.5,
    "net_edge_per_trade_pct": 0.1, "symbol_weights": {"BTCUSDT": 0.2},
    "weighting": "capital", "notes": "stub",
}


def _stubbed_evolve(tmp_path, symbols, mode=None, population=6, generations=1):
    """``evolve()`` with the batch scorer stubbed — no backtest, no data reads."""
    import core.ga.fitness as fitness_mod

    recorded: list = []

    def _stub(population_arg, symbols_arg, date_start, date_end, engine, loader,
              **kwargs):
        recorded.append({"symbols": list(symbols_arg),
                         "benchmark_mode": kwargs.get("benchmark_mode")})
        out = []
        for i, chrom in enumerate(population_arg):
            chrom = dict(chrom)
            chrom["fitness_result"] = {
                "fitness": 10.0 - i, "trade_count": 40, "profit_factor": 1.6,
                "dsr": 0.5, "total_return": 12.0, "sharpe": 2.0, "win_rate": 55.0,
                "max_dd": 1.0, "buy_hold_pct": 3.0, "alpha_vs_buy_hold_pct": 9.0,
                "dsr_detail": {"dsr": 0.5, "n_trials": 0},
                "benchmark_mode": EXPOSURE_MATCHED, "benchmark_pct": 5.0,
                "alpha_vs_benchmark_pct": 7.0, "benchmark_sharpe": 1.2,
                "benchmark_max_dd": 4.0, "benchmark_time_in_market_pct": 22.0,
                "strategy_time_in_market_pct": 30.0, "information_ratio": 0.4,
                "jensen_alpha_annual_pct": 1.1, "benchmark_beta": 0.3,
                "net_edge_per_trade": 2.5, "net_edge_per_trade_pct": 0.1,
                "strategy_risk_matched_pct": None,
                "benchmark": dict(_STUB_BENCHMARK),
            }
            out.append(chrom)
        return out

    class _Cfg:
        data_dir = str(tmp_path)
        backtest_cost_enabled = True
        backtest_taker_fee_pct = 0.04
        backtest_spread_pct = {}
        backtest_engine_mode = "legacy"
        ga_alpha_weight = 1.0
        ga_min_champion_trades = 30
        ga_benchmark_mode = EXPOSURE_MATCHED

    class _Engine:
        config = _Cfg()

    loader = StrategyLoader(str(tmp_path / "stub_strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    original = fitness_mod.evaluate_population_batch
    fitness_mod.evaluate_population_batch = _stub
    try:
        evolver = GAStrategyEvolver(
            _Engine(), loader,
            GARunConfig(population_size=population, generations=generations,
                        elite_count=2, immigrant_count=2, max_workers=1,
                        seed=4242, benchmark_mode=mode))
        result = evolver.evolve(list(symbols), "2026-01-01", "2026-02-01")
    finally:
        fitness_mod.evaluate_population_batch = original
    return result, recorded, loader


def test_the_run_forwards_the_job_mode_to_every_evaluation(tmp_path):
    _, recorded, _ = _stubbed_evolve(tmp_path, ["BTCUSDT"], mode=RISK_MATCHED)
    assert recorded and all(r["benchmark_mode"] == RISK_MATCHED for r in recorded)
    _, recorded, _ = _stubbed_evolve(tmp_path / "b", ["BTCUSDT"], mode=None)
    # ``None`` = "follow the config", never a silent override.
    assert recorded and all(r["benchmark_mode"] is None for r in recorded)


def test_the_champion_records_the_evaluated_basket(tmp_path):
    basket = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    result, recorded, loader = _stubbed_evolve(tmp_path, basket)
    assert recorded[0]["symbols"] == basket
    assert result["champion_config"]["symbols"] == basket
    saved = loader.load(result["champion_name"])
    assert saved.symbols == basket, "the champion YAML does not name its basket"


def test_the_champion_provenance_records_the_benchmark_block(tmp_path):
    result, _, _ = _stubbed_evolve(tmp_path, ["BTCUSDT", "ETHUSDT"])
    provenance = result["provenance"]
    benchmark = provenance["benchmark"]
    assert benchmark["mode"] == EXPOSURE_MATCHED
    assert benchmark["buy_hold_pct"] == 3.0
    assert benchmark["benchmark_pct"] == 5.0
    assert benchmark["alpha_vs_benchmark_pct"] == 7.0
    # The benchmark's OWN risk numbers (only the return used to be recorded).
    assert benchmark["benchmark_sharpe"] == 1.2
    assert benchmark["benchmark_max_dd_pct"] == 4.0
    assert benchmark["benchmark_time_in_market_pct"] == 22.0
    # Reported, never gated.
    assert benchmark["information_ratio"] == 0.4
    assert benchmark["jensen_alpha_annual_pct"] == 1.1
    assert benchmark["net_edge_per_trade"] == 2.5
    assert benchmark["strategy_time_in_market_pct"] == 30.0
    assert benchmark["report"]["mode"] == EXPOSURE_MATCHED
    # Additive keys, and the legacy ones are untouched.
    assert provenance["fitness_components"]["benchmark_pct"] == 5.0
    assert provenance["fitness_components"]["alpha_vs_benchmark_pct"] == 7.0
    assert provenance["fitness_components"]["fitness"] == 10.0
    assert provenance["fitness_components"]["buy_hold_pct"] == 3.0
    assert provenance["fitness_components"]["alpha_vs_buy_hold_pct"] == 9.0
    assert result["benchmark_mode"] == EXPOSURE_MATCHED
    assert result["published"] is True


class _FakeMarketData:
    """Minimal stand-in for MarketDataProvider (no network, no frames needed)."""

    def __init__(self, watched):
        self.watched_symbols = list(watched)


@pytest.mark.asyncio
async def test_enabling_a_champion_can_only_trade_its_recorded_basket(tmp_path):
    """The evaluated basket reaches the LIVE evaluation path (not just the YAML).

    The watcher holds five symbols; the champion was evaluated on two.  A real
    ``StrategyEngine.evaluate_all_now()`` must only evaluate the champion on the
    two it was scored on — otherwise an enabled champion would trade pairs it
    was never evaluated on.
    """
    from app.event_bus import EventBus
    from core.strategy.engine import StrategyEngine

    basket = ["BTCUSDT", "ETHUSDT"]
    result, _, loader = _stubbed_evolve(tmp_path, basket)
    champion = loader.load(result["champion_name"])
    assert champion.enabled is True

    Config._instance = None
    try:
        config = Config.load("sim")
        engine = StrategyEngine(
            config, EventBus(),
            _FakeMarketData(["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]))
    finally:
        Config._instance = None
    engine._strategies = {champion.name: champion}

    evaluated: list = []

    async def _record(symbol, interval, strategy, publish=False):
        evaluated.append(symbol)

    engine._evaluate = _record  # type: ignore[assignment]
    await engine.evaluate_all_now()

    assert set(evaluated) == set(basket)
    assert not (set(evaluated) & {"BNBUSDT", "SOLUSDT", "XRPUSDT"}), (
        f"the live path evaluated symbols outside the champion's basket: {evaluated}")


# ══════════════════════════════════════════════════════════════════════
# (e) byte-identity with the pre-``benchmark_mode`` revision
# ══════════════════════════════════════════════════════════════════════

_IDENTITY_HARNESS = r'''"""Identity harness: scored values, gate verdict, legacy provenance."""
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

TREE = Path.cwd()
sys.path.insert(0, str(TREE))

MODE = sys.argv[1]          # "buy_hold" | "absent"
OUT = Path(sys.argv[2])


def write_market(root):
    """Deterministic OHLCV parquet tree (identical in both trees)."""
    rng = np.random.default_rng(20261001)
    base = pd.date_range("2026-01-01", periods=120 * 96, freq="15min")
    for symbol in ("BTCUSDT", "ETHUSDT"):
        close = 20000 + np.cumsum(rng.normal(0, 40, len(base)))
        m15 = pd.DataFrame({
            "open": close, "high": close + 30, "low": close - 30,
            "close": close, "volume": rng.random(len(base)) * 100 + 10,
        }, index=base)
        market = root / "market" / symbol
        market.mkdir(parents=True, exist_ok=True)
        for tf in ("15m", "1h", "4h"):
            frame = m15 if tf == "15m" else m15.resample(tf).agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"}).dropna()
            frame.to_parquet(market / f"{tf}.parquet")


tmp = Path(tempfile.mkdtemp())
write_market(tmp / "data")

from app.config import Config
from app.event_bus import EventBus
from core.backtest.engine import BacktestEngine
from core.executor.executor import OrderExecutor
from core.risk.manager import RiskManager
from core.strategy.loader import StrategyLoader
from core.ga.fitness import evaluate_population_batch
from core.ga.genome import random_chromosome
from core.ga.evolver import GAStrategyEvolver, GARunConfig

Config._instance = None
cfg = Config.load("sim")
cfg.data_dir = str(tmp / "data")
cfg.backtest_engine_mode = "legacy"
cfg.backtest_ml_enabled = False
cfg.backtest_live_spread_enabled = False
# The pre-change revision has no such attribute at all: "absent" reproduces
# exactly that, "buy_hold" selects the historical mode explicitly.
if hasattr(cfg, "ga_benchmark_mode"):
    if MODE == "absent":
        del cfg.ga_benchmark_mode
    else:
        cfg.ga_benchmark_mode = "buy_hold"

bus = EventBus()
engine = BacktestEngine(cfg, None, RiskManager(cfg, bus), OrderExecutor(cfg, bus))
loader = StrategyLoader(str(tmp / "strategies"))
loader.strategies_dir.mkdir(parents=True, exist_ok=True)

# ── 1. a fixed population, scored by the real batch path ──
random.seed(20260101)
population = [random_chromosome(f"ident_{i}") for i in range(2)]
evaluate_population_batch(population, ["BTCUSDT", "ETHUSDT"],
                          "2026-02-01", "2026-02-20", engine, loader,
                          max_workers=1, use_live_spread=False,
                          batch_trials=2, prior_trials=0)

SCORE_KEYS = ["fitness", "fitness_base", "fitness_alpha", "sharpe", "win_rate",
              "profit_factor", "raw_profit_factor", "max_dd", "total_return",
              "buy_hold_pct", "alpha_vs_buy_hold_pct", "dsr", "observations",
              "trade_count", "long_trades", "short_trades", "flag"]
scored = [{k: c["fitness_result"].get(k) for k in SCORE_KEYS} for c in population]

# ── 2. the gate verdict on the best genome ──
evolver = GAStrategyEvolver(engine, loader, GARunConfig(population_size=2))
best = max(scored, key=lambda r: r["fitness"] if r["fitness"] is not None else -1e9)
published, reasons = evolver._publication_decision(dict(best), None)

# ── 3. the champion provenance, from a stubbed run (no data reads) ──
import core.ga.fitness as fitness_mod


def _stub(population_arg, symbols, date_start, date_end, engine_, loader_, **kwargs):
    out = []
    for i, chrom in enumerate(population_arg):
        chrom = dict(chrom)
        chrom["fitness_result"] = {
            "fitness": 10.0 - i, "trade_count": 40, "profit_factor": 1.6,
            "dsr": 0.5, "total_return": 12.0, "sharpe": 2.0, "win_rate": 55.0,
            "max_dd": 1.0, "buy_hold_pct": 3.0, "alpha_vs_buy_hold_pct": 9.0,
            "dsr_detail": {"dsr": 0.5, "n_trials": 0}}
        out.append(chrom)
    return out


original = fitness_mod.evaluate_population_batch
fitness_mod.evaluate_population_batch = _stub
try:
    runner = GAStrategyEvolver(engine, loader, GARunConfig(
        population_size=4, generations=1, elite_count=2, immigrant_count=2,
        max_workers=1, seed=4242))
    run = runner.evolve(["BTCUSDT", "ETHUSDT"], "2026-01-01", "2026-02-01")
finally:
    fitness_mod.evaluate_population_batch = original

PROV_KEYS = ["seed", "window", "symbols", "timeframes", "timeframe_pool",
             "condition_logic", "generations", "population_size", "n_trials",
             "prior_trials", "validation", "published", "rejection_reasons", "eval"]
FIT_KEYS = ["fitness", "fitness_base", "fitness_alpha", "sharpe", "deflated_sharpe",
            "max_dd", "trade_count", "profit_factor", "raw_profit_factor",
            "buy_hold_pct", "alpha_vs_buy_hold_pct"]
provenance = run["provenance"]
payload = {
    "scored": scored,
    "gate": {"published": published, "reasons": reasons},
    "provenance": {k: provenance.get(k) for k in PROV_KEYS},
    "fitness_components": {k: provenance["fitness_components"].get(k) for k in FIT_KEYS},
    "run": {"published": run["published"], "rejection_reasons": run["rejection_reasons"],
            "fitness": run["fitness"], "sharpe": run["sharpe"]},
}
blob = json.dumps(payload, sort_keys=True, indent=1).encode("utf-8")
OUT.write_bytes(blob)
print(hashlib.sha256(blob).hexdigest())
'''


def _run_harness(tree: Path, tmp_path: Path, name: str, mode: str):
    """Run the identity harness with *tree* as the import root."""
    harness = tmp_path / "benchmark_identity_harness.py"
    harness.write_text(_IDENTITY_HARNESS, encoding="utf-8", newline="\n")
    out = tmp_path / f"{name}.json"
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    proc = subprocess.run([sys.executable, str(harness), mode, str(out)],
                          cwd=str(tree), capture_output=True, text=True,
                          timeout=1800, env=env)
    assert proc.returncode == 0, f"harness failed in {tree} [{mode}]:\n{proc.stderr}"
    return out.read_bytes(), proc.stdout.strip()


def test_buy_hold_is_byte_identical_to_the_head_worktree(tmp_path):
    """``buy_hold`` (and the key absent) ⇒ HEAD's scored values, gate, provenance.

    A ``git worktree`` is checked out at the pre-change revision, the same
    harness runs in both trees (the working tree with the mode selected and with
    the attribute removed), and the serialised scored values, gate verdict and
    legacy provenance keys are compared **as bytes**.
    """
    worktree = tmp_path / "head_tree"
    add = subprocess.run(["git", "worktree", "add", "--detach", str(worktree),
                          BASELINE_REVISION],
                         cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    if add.returncode != 0:
        pytest.skip(f"cannot create a HEAD worktree at {BASELINE_REVISION}: "
                    f"{add.stderr.strip()}")
    try:
        head_bytes, head_digest = _run_harness(worktree, tmp_path, "head", "buy_hold")
        for variant in ("buy_hold", "absent"):
            tree_bytes, tree_digest = _run_harness(
                ROOT, tmp_path, f"tree_{variant}", variant)
            assert tree_digest == head_digest, (
                f"benchmark_mode={variant} changed the scored result:\n"
                f"  HEAD {BASELINE_REVISION}: {head_digest}\n"
                f"  working tree: {tree_digest}")
            assert tree_bytes == head_bytes, (
                f"the harness output differs byte-for-byte for {variant}")

        payload = json.loads(head_bytes.decode("utf-8"))
        # The comparison is only meaningful if it really covered the seams.
        assert len(payload["scored"]) == 2
        assert max(s["trade_count"] for s in payload["scored"]) > 0
        assert payload["scored"][0]["buy_hold_pct"] is not None
        assert payload["provenance"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
        assert payload["fitness_components"]["buy_hold_pct"] == 3.0
        assert payload["fitness_components"]["fitness"] == 10.0
        assert hashlib.sha256(head_bytes).hexdigest() == head_digest
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=600)

