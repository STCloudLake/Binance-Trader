"""Audit D-5/D-18: the DSR deflation must use the trials actually performed.

``core/ga/evolver.py`` used two different ``N`` for the same champion: every
in-generation score was deflated by ``population + prior`` (``_batch_trials =
len(self._population)``) while the champion/provenance used
``population × generations + prior``.  The reported champion DSR and the number
of strategies the run claims to have tried therefore disagreed.

``dsr_trial_counts`` is now the ONE formula for both: the count for the
generation being scored is the ledger total (earlier runs plus every earlier
generation of this run, floored by the deterministic count) and the champion
reuses exactly the same number once the last generation has been performed.

This file pins the arithmetic, the ledger floor, and — end to end — the numbers
the evolver really passes to the scorer and writes into ``provenance``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_dsr_trial_counts_match_the_generations_performed():
    from core.ga.evolver import dsr_trial_counts

    population, prior = 6, 0
    seen = []
    trials_this_run = 0
    for gen in range(1, 5):
        prior_trials, cumulative = dsr_trial_counts(
            prior, 0, population, trials_this_run)
        assert prior_trials == population * (gen - 1)
        assert cumulative == population * gen
        seen.append((prior_trials, cumulative))
        trials_this_run += population
    assert seen == [(0, 6), (6, 12), (12, 18), (18, 24)]

    # The champion (after the last generation) reuses the SAME number the last
    # generation was scored against: 6 × 4 + prior.  By then the trials are all
    # performed, so it is `prior` (the ledger), not `prior + population`.
    champion_n, would_double_count = dsr_trial_counts(
        prior, 0, population, trials_this_run)
    assert champion_n == seen[-1][1] == 24
    assert would_double_count == 30      # why the champion must not use this


def test_the_ledger_raises_the_count_and_never_lowers_it():
    from core.ga.evolver import dsr_trial_counts

    # 24 earlier walk-forward jobs recorded in the ledger → the 1st generation of
    # this run is already deflated by all of them.
    prior_trials, cumulative = dsr_trial_counts(10, 1440, 6, 0)
    assert prior_trials == 1440 and cumulative == 1446
    # A failed ledger write (best-effort `record_trials`) can only understate, so
    # the deterministic count floors it.
    prior_trials, cumulative = dsr_trial_counts(10, 0, 6, 18)
    assert prior_trials == 28 and cumulative == 34
    # Resume: `prior_trials` already holds the resumed generations, so counting
    # them again in `trials_this_run` is not possible (they are not re-run) and
    # the ledger wins when it is larger.
    assert dsr_trial_counts(30, 42, 6, 6) == (42, 48)


def test_score_stats_receives_the_same_n_for_the_generation_and_the_champion():
    """`dsr_detail["n_trials"]` is `prior + population` — one number, one path."""
    from core.ga.evolver import dsr_trial_counts
    from core.ga.fitness import deflated_sharpe_ratio, score_stats

    stats = {"trades": 60, "win_rate": 55.0, "profit_factor": 1.5,
             "return_on_capital": 0.02, "long_trades": 30, "pnl": 200.0,
             "sharpe": 2.0, "max_dd_pct": 1.0, "observations": 100,
             "skew": 0.0, "kurtosis": 3.0}

    prior_for_gen, cumulative = dsr_trial_counts(0, 0, 6, 12)     # 3rd generation
    scored = score_stats(dict(stats), None, n_trials=6,
                         prior_trials=prior_for_gen)
    assert scored["dsr_detail"]["n_trials"] == cumulative == 18
    assert scored["dsr_detail"]["n_trials"] == deflated_sharpe_ratio(
        2.0, 18, observation_periods=100)["n_trials"]


def test_alpha_weight_is_wired_and_bit_identical_at_its_default():
    """`ga.alpha_weight` was loaded from config and read nowhere."""
    from core.ga.fitness import ALPHA_WEIGHT, score_stats

    stats = {"trades": 60, "win_rate": 55.0, "profit_factor": 1.5,
             "return_on_capital": 0.02, "long_trades": 30, "pnl": 200.0,
             "sharpe": 2.0, "max_dd_pct": 1.0, "observations": 100,
             "skew": 0.0, "kurtosis": 3.0}
    default = score_stats(dict(stats), None, n_trials=10)
    explicit = score_stats(dict(stats), None, n_trials=10,
                           alpha_weight=ALPHA_WEIGHT)
    assert default["fitness"] == explicit["fitness"]        # bit-identical
    assert default["fitness_alpha"] == explicit["fitness_alpha"]

    off = score_stats(dict(stats), None, n_trials=10, alpha_weight=0.0)
    assert off["fitness"] == off["fitness_base"]
    assert off["fitness_alpha"] == 0.0
    doubled = score_stats(dict(stats), None, n_trials=10, alpha_weight=2.0)
    # `fitness_alpha` is rounded to 4 dp, hence the 1e-3 tolerance.
    assert doubled["fitness_alpha"] == pytest.approx(
        2.0 * default["fitness_alpha"], abs=1e-3)
    # The term is real (so the wiring is observable, not a no-op).
    assert abs(default["fitness_alpha"]) > 0.0


def test_the_dead_ga_switches_are_gone_or_wired():
    """The 7 knobs the audit listed as read-nowhere (D-26/§1.10) are resolved.

    Deleted: ``PF_SHRINK`` (the shrink is unconditional), ``WEIGHT_GRID`` (the
    calibration search it belonged to was already removed),
    ``GARunConfig.overfit_penalty`` (no sensitivity penalty exists),
    ``ga.evaluation_leverage`` (the engine has no leverage input).
    Wired: ``ga.alpha_weight``, ``total_trials`` and ``_GARCH_MLE_X0``.
    """
    import dataclasses
    import inspect

    import core.ga.fitness as fitness_mod
    import core.ga.fitness_calibrate as calibrate_mod
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.ml import volatility as vol

    assert not hasattr(fitness_mod, "PF_SHRINK")
    assert not hasattr(calibrate_mod, "WEIGHT_GRID")
    assert "overfit_penalty" not in {
        f.name for f in dataclasses.fields(GARunConfig)}

    evolve_src = inspect.getsource(GAStrategyEvolver.evolve)
    assert "total_trials(" in evolve_src, "total_trials has no production caller"
    assert "dsr_trial_counts(" in evolve_src
    assert "_GARCH_MLE_X0[" in inspect.getsource(vol._garch11_grid_scan)
    # ... and the wired GARCH start is still the documented RiskMetrics pair.
    from core.ml.volatility import _GARCH_MLE_X0
    assert _GARCH_MLE_X0 == (0.06, 0.93)
    start = vol._garch11_grid_scan(
        __import__("numpy").array([1.0, 2.0, 3.0, 4.0]), 2.5)
    assert len(start) == 3


def _run_evolve_with_a_stub_scorer(tmp_path, *, population, generations,
                                   prior_ledger=0, ga_alpha_weight=1.0):
    """Run `evolve()` with the batch scorer stubbed — no backtest, no data reads."""
    import core.ga.fitness as fitness_mod
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    calls: list[dict] = []

    def _stub(population_arg, symbols, date_start, date_end, engine, loader,
              **kwargs):
        calls.append({"batch_trials": kwargs.get("batch_trials"),
                      "prior_trials": kwargs.get("prior_trials"),
                      "alpha_weight": kwargs.get("alpha_weight")})
        out = []
        for i, chrom in enumerate(population_arg):
            chrom = dict(chrom)
            chrom["fitness_result"] = {
                "fitness": 10.0 - i, "trade_count": 40, "profit_factor": 1.6,
                "dsr": 0.5, "total_return": 12.0, "sharpe": 2.0, "win_rate": 55.0,
                "max_dd": 1.0, "buy_hold_pct": 3.0, "alpha_vs_buy_hold_pct": 9.0,
                "dsr_detail": {"dsr": 0.5, "n_trials": 0},
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

    class _Engine:
        config = _Cfg()

    _Engine.config.ga_alpha_weight = ga_alpha_weight
    loader = StrategyLoader(str(tmp_path / "ga_strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    if prior_ledger:
        ledger = Path(tmp_path) / "data" / "ga_trials.json"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps({"trials": prior_ledger}), encoding="utf-8")

    original = fitness_mod.evaluate_population_batch
    fitness_mod.evaluate_population_batch = _stub
    try:
        evolver = GAStrategyEvolver(
            _Engine(), loader,
            GARunConfig(population_size=population, generations=generations,
                        elite_count=1, immigrant_count=1, max_workers=1,
                        seed=4242))
        result = evolver.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01")
    finally:
        fitness_mod.evaluate_population_batch = original
    return result, calls


def test_evolve_passes_the_performed_trial_count_to_every_generation(tmp_path):
    population, generations = 4, 3
    result, calls = _run_evolve_with_a_stub_scorer(
        tmp_path, population=population, generations=generations)

    assert len(calls) == generations
    assert [c["batch_trials"] for c in calls] == [population] * generations
    # Generation g is scored against everything already performed.
    assert [c["prior_trials"] for c in calls] == [
        population * (g - 1) for g in range(1, generations + 1)]

    # The champion's provenance uses the SAME formula: the last generation's
    # prior + its own population == population × generations == n_trials.
    last = calls[-1]
    assert result["provenance"]["n_trials"] == last["prior_trials"] + last["batch_trials"]
    assert result["provenance"]["n_trials"] == population * generations


def test_evolve_adds_the_ledger_and_the_configured_alpha_weight(tmp_path):
    population, generations = 4, 2
    result, calls = _run_evolve_with_a_stub_scorer(
        tmp_path, population=population, generations=generations,
        prior_ledger=1440, ga_alpha_weight=2.5)

    assert [c["prior_trials"] for c in calls] == [1440, 1440 + population]
    assert result["provenance"]["n_trials"] == 1440 + population * generations
    assert result["provenance"]["prior_trials"] == 1440
    # `ga.alpha_weight` really reaches the scorer (it used to reach nothing).
    assert [c["alpha_weight"] for c in calls] == [2.5] * generations
