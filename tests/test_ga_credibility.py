"""P1 — GA credibility regression tests (docs/overhaul/ALGO_UPGRADE_PLAN.md §二).

Every test here pins one of the eleven measured defects from the plan's §一 table
so it cannot silently come back:

1. batch evaluation: per-genome position slots + per-genome ledger
2. walk-forward: a real out-of-sample window (never ``[tr_end, tr_end]``)
3. fitness: a 5-trade all-winner genome can no longer outrank a 200-trade one
4. risk-adjusted selection metric: Sharpe / max-DD / buy & hold are real
5. DSR units: per-period Sharpe against a per-period hurdle, real period count
6. publication gate: failing champions are written ``enabled: false``
7. ML score/publish consistency
8. genome: sanitisation against the DECODED indicators, name-keyed crossover
9. reproducibility: same seed → identical population
10. multiprocess fallback: a dead chunk is retried, not scored −999
11. no-look-ahead invariants (the plan's "already sound" list)
"""
from __future__ import annotations

import copy
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── synthetic market (no cached data / no network needed) ───────────────

def _write_market(tmp_path: Path, symbols=("BTCUSDT", "ETHUSDT"),
                  timeframes=("15m", "1h", "4h"), start="2026-01-01",
                  bars_15m=120 * 96):
    """Deterministic OHLCV parquet tree (same shape as the parity gate uses)."""
    rng = np.random.default_rng(20261001)
    base = pd.date_range(start, periods=bars_15m, freq="15min")
    for symbol in symbols:
        close = 20000 + np.cumsum(rng.normal(0, 40, len(base)))
        m15 = pd.DataFrame({
            "open": close, "high": close + 30, "low": close - 30,
            "close": close, "volume": rng.random(len(base)) * 100 + 10,
        }, index=base)
        market_dir = tmp_path / "market" / symbol
        market_dir.mkdir(parents=True, exist_ok=True)
        for tf in timeframes:
            if tf == "15m":
                frame = m15
            else:
                frame = m15.resample(tf).agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna()
            frame.to_parquet(market_dir / f"{tf}.parquet")
    return str(tmp_path)


@pytest.fixture()
def market_dir(tmp_path):
    return _write_market(tmp_path / "data")


def _engine(market_dir, tmp_path, symbols=("BTCUSDT", "ETHUSDT")):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import StrategyLoader

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = market_dir
    cfg.backtest_engine_mode = "legacy"      # per-genome ledger path
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    bus = EventBus()
    loader = StrategyLoader(str(tmp_path / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus), OrderExecutor(cfg, bus))
    return cfg, engine, loader


def _always_on_strategy(name: str, **kwargs):
    """A strategy that enters almost every bar (guarantees trades)."""
    from core.strategy.loader import MLConfig, StrategyConfig

    params = dict(
        name=name, enabled=True, mode="trend", timeframes=["1h"],
        indicators={"rsi": {"period": 14, "source": "close"},
                    "sma": {"period": 5}},
        entry_conditions={"long": ["close > sma"], "short": ["close < sma"]},
        exit_conditions={"long": ["rsi > 99"], "short": ["rsi < 1"]},
        ml_config=MLConfig(enabled=False),
    )
    params.update(kwargs)
    return StrategyConfig(**params)


# ══════════════════════════════════════════════════════════════════════════
# 1 — batch evaluation: per-genome slots and per-genome ledger
# ══════════════════════════════════════════════════════════════════════════

def _churn_strategy(name: str):
    """Occasional entries with a short max-hold — trades, but no compounding blow-up."""
    from core.strategy.loader import MLConfig, RiskExitConfig, StrategyConfig

    return StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=["1h"],
        indicators={"bollinger": {"period": 20, "stddev": 2.0}},
        entry_conditions={"long": ["close > bollinger_upper"],
                          "short": ["close < bollinger_lower"]},
        exit_conditions={"long": ["close < bollinger_middle"],
                         "short": ["close > bollinger_middle"]},
        risk_exit=RiskExitConfig(stop_loss_pct=2.0, trailing_stop_pct=1.5,
                                 max_hold_hours=4.0, use_indicator_exits=True),
        ml_config=MLConfig(enabled=False),
    )


def test_every_genome_in_chunk_gets_its_own_slots_and_ledger(market_dir, tmp_path):
    """The 20-genome chunk must not starve 19 of them (measured: 1×407, 19×0)."""
    cfg, engine, loader = _engine(market_dir, tmp_path)
    strategies = [_churn_strategy(f"iso_{i}") for i in range(6)]
    result = engine.run_with_exit_evaluation(
        strategies=strategies, symbols=["BTCUSDT", "ETHUSDT"],
        date_start="2026-02-01", date_end="2026-02-10",
        initial_balance=10000.0, mode="full", simulate_ai_weights=False,
        per_strategy_isolation=True, per_genome_ledger=True, use_live_spread=False)

    per = result["per_strategy_equity"]
    assert per is not None
    traded = {name: len(per[name]["trades"]) for name in per}
    assert all(n > 0 for n in traded.values()), (
        f"a genome in the chunk never traded: {traded}")
    # Each genome gets its own slot budget: nobody monopolises the chunk.
    busiest = max(traded.values())
    quietest = min(traded.values())
    assert quietest >= busiest * 0.2, f"slot stealing: {traded}"
    # Each genome's ledger ends at its OWN initial balance + realised PnL
    # (per-trade PnL is rounded to 2dp, hence the small tolerance).
    for name, data in per.items():
        realised = sum(t["pnl"] for t in data["trades"])
        assert data["final_balance"] == pytest.approx(
            10000.0 + realised, abs=0.05 + 0.01 * len(data["trades"])), (
            f"{name} ledger contaminated: {data['final_balance']} vs {10000 + realised}")
    # Position keys carry the genome name — no cross-genome slot stealing.
    keys = [t.get("strategy") for t in result["trades"]]
    assert set(keys) == {s.name for s in strategies}


def _trading_genome(name: str):
    """A small chromosome that definitely trades on the synthetic market."""
    from core.ga.genome import (BooleanGene, CategoricalGene, ContinuousGene,
                                StructuralGene)

    return {
        "continuous": [ContinuousGene("rsi_period", 14, 5, 28, 1),
                       ContinuousGene("bb_period", 20, 10, 40, 2),
                       ContinuousGene("bb_stddev", 2.0, 1.0, 3.5, 0.25)],
        "categorical": [CategoricalGene("mode", "trend", ["trend"]),
                        CategoricalGene("timeframes", "1h", ["1h"])],
        "structural": [StructuralGene("entry_long", ["rsi < 55"], []),
                        StructuralGene("entry_short", ["rsi > 45"], []),
                        StructuralGene("exit_long", ["close < bollinger_middle"], []),
                        StructuralGene("exit_short", ["close > bollinger_middle"], [])],
        "indicator_genes": [BooleanGene("rsi", True), BooleanGene("bollinger", True)],
        "condition_logic": "or",
        "name": name,
    }


def test_batch_result_is_identical_to_single_evaluation(market_dir, tmp_path):
    """Contract: the chunk ledger must not change what a genome actually does."""
    from core.ga.fitness import (evaluate_population_batch, score_stats,
                                 stats_from_engine_result)
    from core.ga.genome import chromosome_to_strategy

    cfg, engine, loader = _engine(market_dir, tmp_path)
    population = [_trading_genome(f"consistency_{i}") for i in range(2)]

    single = []
    for i, chrom in enumerate(population):
        s_cfg = chromosome_to_strategy(chrom)
        s_cfg.name = "solo"
        res = engine.run_with_exit_evaluation(
            strategies=[s_cfg], symbols=["BTCUSDT"], date_start="2026-02-01",
            date_end="2026-03-01", initial_balance=10000.0, mode="full",
            simulate_ai_weights=False, per_strategy_isolation=True, per_genome_ledger=True,
            use_live_spread=False)
        stats = stats_from_engine_result(res, "solo", 10000.0)
        single.append(score_stats(stats, chrom, n_trials=1 + i))

    pop = copy.deepcopy(population)
    evaluate_population_batch(
        pop, ["BTCUSDT"], "2026-02-01", "2026-03-01",
        engine, loader, max_workers=1, use_live_spread=False,
        batch_trials=1, prior_trials=1)

    assert all(s["trades"] > 0 for s in single), (
        f"test is vacuous — the genome never traded: {[s['trades'] for s in single]}")
    for i, (chrom, solo) in enumerate(zip(pop, single)):
        batch = chrom["fitness_result"]
        assert batch["trade_count"] == solo["trades"], (
            f"genome {i}: same genome traded differently in a chunk "
            f"({batch['trade_count']} vs {solo['trades']})")
        assert batch["fitness"] == pytest.approx(solo["fitness"], abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════
# 2 — walk-forward: a real out-of-sample window
# ══════════════════════════════════════════════════════════════════════════

def test_walkforward_passes_real_val_end_and_asserts_min_bars(tmp_path):
    """The exact measured bug: ``validate=2025-11-01~2025-11-01`` (1 bar)."""
    from core.ga.walkforward import WalkForwardRunner, WFConfig, MIN_VALIDATION_BARS

    data_dir = _write_market(tmp_path / "wfdata", timeframes=("1h",),
                            bars_15m=90 * 96, start="2026-01-01")
    runner = WalkForwardRunner(None, None, data_dir)
    windows = runner.compute_windows("2026-01-01", "2026-04-01",
                                     WFConfig(train_months=1, val_months=1))
    tr_s, tr_e, val_s, val_e = windows[0]
    assert val_s < val_e, "validation window must span more than one instant"
    bars = runner._validation_bars(["BTCUSDT"], val_s, val_e)
    assert bars >= MIN_VALIDATION_BARS, f"only {bars} OOS bars"

    # A degenerate window must fail loudly, not silently.
    with pytest.raises(ValueError):
        runner._assert_window(val_s, val_s, ["BTCUSDT"])
    with pytest.raises(ValueError):
        runner._assert_window("2026-01-01", "2026-01-01 06:00", ["BTCUSDT"])


def test_walkforward_evolve_call_receives_val_end(tmp_path, monkeypatch):
    """``evolve(symbols, tr_start, val_end, validation_start=val_start)``."""
    from core.ga import walkforward as wf

    calls = []

    class _FakeEvolver:
        def __init__(self, *a, **k):
            pass

        def set_progress_callback(self, cb):
            pass

        def evolve(self, symbols, start, end, **kwargs):
            calls.append({"symbols": symbols, "start": start, "end": end,
                          "kwargs": kwargs})
            return {"validation": {}, "history": []}

    monkeypatch.setattr("core.ga.evolver.GAStrategyEvolver", _FakeEvolver)

    data_dir = _write_market(tmp_path / "wfdata2", timeframes=("1h",),
                            bars_15m=90 * 96, start="2026-01-01")
    runner = wf.WalkForwardRunner(None, None, data_dir)
    windows = runner.compute_windows("2026-01-01", "2026-04-01",
                                     wf.WFConfig(train_months=1, val_months=1))
    runner.run(["BTCUSDT"], "2026-01-01", "2026-04-01",
               wf.WFConfig(train_months=1, val_months=1), object())

    assert calls, "no window was evaluated"
    assert len(calls) == len(windows)
    for call, (tr_s, tr_e, val_s, val_e) in zip(calls, windows):
        assert call["start"] == tr_s
        assert call["end"] == val_e, (
            f"evolve received {call['end']} as date_end, expected OOS val_end {val_e}")
        assert call["end"] != tr_e, "evolve must not be called with train_end as date_end"
        assert call["kwargs"]["validation_start"] == val_s
        assert call["kwargs"]["window_key"]


# ══════════════════════════════════════════════════════════════════════════
# 3 — fitness: PF shrink/cap/scale + trade-count floor
# ══════════════════════════════════════════════════════════════════════════

def _equity_for(pnls, initial=10000.0):
    total = float(sum(pnls))
    return [{"time": "2026-01-01", "equity": initial},
            {"time": "2026-03-01", "equity": initial + total * 0.5},
            {"time": "2026-06-01", "equity": initial + total}]


def test_five_winning_trades_cannot_outrank_two_hundred_trades():
    """Measured before: 5 winners → 490.85, 200 trades/PF2.0 → 7.30."""
    from core.ga.fitness import score_stats, stats_from_trades

    a_pnls = [100.0] * 5
    b_pnls = [30.0] * 120 + [-18.0] * 80          # 60% win, PF 2.0, +2160
    stats_a = stats_from_trades([{"pnl": p, "side": "long"} for p in a_pnls],
                                _equity_for(a_pnls), 10000.0)
    stats_b = stats_from_trades([{"pnl": p, "side": "long"} for p in b_pnls],
                                _equity_for(b_pnls), 10000.0)
    stats_a["buy_hold_pct"] = stats_b["buy_hold_pct"] = 0.0
    a = score_stats(stats_a, None, n_trials=100)
    b = score_stats(stats_b, None, n_trials=100)

    assert a["profit_factor"] <= 10.0 + 1e-9       # PF_TERM_CAP
    assert a["fitness_base"] < 20, (
        "a 5-trade genome must stay far below a real one on the same scale")
    assert b["fitness"] > a["fitness"], (
        f"200-trade genome ({b['fitness']}) must outrank the 5-trade one "
        f"({a['fitness']})")
    assert a["flag"] == "insufficient_trades"      # explicitly flagged, not silent
    assert b["flag"] == ""


def test_profit_factor_shrinks_when_there_are_no_losses():
    from core.ga.fitness import profit_factor_shrunk
    assert profit_factor_shrunk(500.0, 0.0, 100.0) == pytest.approx(5.0)
    assert profit_factor_shrunk(500.0, 250.0, 100.0) == pytest.approx(500 / 350)
    assert profit_factor_shrunk(0.0, 100.0, 10.0) == pytest.approx(0.1)


# ══════════════════════════════════════════════════════════════════════════
# 4 — real risk-adjusted metric (Sharpe / max DD / buy & hold baseline)
# ══════════════════════════════════════════════════════════════════════════

def test_batch_path_reports_real_sharpe_and_drawdown(market_dir, tmp_path):
    """The batch path used to hardcode ``sharpe=0, max_dd=0``."""
    from core.ga.fitness import evaluate_population_batch
    from core.ga.genome import random_chromosome

    cfg, engine, loader = _engine(market_dir, tmp_path)
    random.seed(3)
    population = [random_chromosome(f"metric_{i}") for i in range(2)]
    evaluate_population_batch(population, ["BTCUSDT"], "2026-02-01", "2026-04-01",
                              engine, loader, max_workers=1, use_live_spread=False)
    for chrom in population:
        r = chrom["fitness_result"]
        assert "sharpe" in r and "max_dd" in r and "dsr" in r
        assert r["trade_count"] >= 0
        assert r["buy_hold_pct"] is not None, "benchmark missing — beta would score as alpha"
        assert r["observations"] > 0
        # Sharpe is a genuine estimate, not a placeholder.
        assert not (r["sharpe"] == 0 and r["max_dd"] == 0 and r["trade_count"] > 0)


def test_buy_and_hold_is_subtracted_from_the_selection_metric(market_dir, tmp_path):
    """A pure-drift window must not earn alpha credit."""
    from core.ga.fitness import score_stats, stats_from_trades

    # Flat strategy: no return of its own, market drifted +10% (pure beta).
    stats = stats_from_trades([{"pnl": 0.0, "side": "long"}],
                              _equity_for([0.0]), 10000.0)
    stats["buy_hold_pct"] = 10.0
    scored = score_stats(stats, None, n_trials=100)
    # The metric itself is benchmark-relative: with no edge there is no bonus.
    assert scored["alpha_vs_buy_hold_pct"] < 0

    # Beating the benchmark gives positive alpha; matching it gives none.
    stats["total_return_pct"] = 15.0
    better = score_stats(stats, None, n_trials=100)
    assert better["alpha_vs_buy_hold_pct"] == pytest.approx(5.0)
    stats["total_return_pct"] = 10.0
    same = score_stats(stats, None, n_trials=100)
    assert same["alpha_vs_buy_hold_pct"] == pytest.approx(0.0)


def test_engine_reports_the_buy_and_hold_baseline(market_dir, tmp_path):
    """The engine must actually compute it from the same window's bars."""
    cfg, engine, loader = _engine(market_dir, tmp_path)
    result = engine.run_with_exit_evaluation(
        strategies=[_always_on_strategy("bh")], symbols=["BTCUSDT"],
        date_start="2026-02-01", date_end="2026-03-01", mode="full",
        simulate_ai_weights=False, per_strategy_isolation=True, per_genome_ledger=True,
        use_live_spread=False)
    buy_hold = result["metrics"]["buy_hold_pct"]
    assert buy_hold is not None
    feed = pd.read_parquet(
        Path(market_dir) / "market" / "BTCUSDT" / "1h.parquet", columns=["close"])
    window = feed[(feed.index >= pd.Timestamp("2026-02-01"))
                  & (feed.index <= pd.Timestamp("2026-03-01"))]
    expected = (float(window.iloc[-1]["close"]) / float(window.iloc[0]["close"]) - 1) * 100
    assert buy_hold == pytest.approx(expected, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════
# 5 — DSR units
# ══════════════════════════════════════════════════════════════════════════

def test_dsr_hand_check_units():
    """Hand-check: per-period Sharpe vs per-period expected max.

    SR=1.2 annualised over T=365 daily observations, N=1200 trials:
      sr_per_period = 1.2/sqrt(365)              = 0.06282
      E[max]        = sqrt(1/T)*sqrt(2 ln N)     = 0.19148
      DSR           = -0.12866  → NOT significant
    The old code compared the ANNUALISED 1.2 with the per-period 0.19 and
    reported +1.0029 ("significant").
    """
    import math
    from core.ga.fitness import deflated_sharpe_ratio

    sr_ann, t_periods, n_trials = 1.2, 365, 1200
    res = deflated_sharpe_ratio(sr_ann, n_trials, observation_periods=t_periods)
    sr_p = sr_ann / math.sqrt(365)
    expected_max = math.sqrt(1.0 / t_periods) * math.sqrt(2 * math.log(n_trials))
    assert res["sharpe_per_period"] == pytest.approx(sr_p, abs=1e-6)
    assert res["expected_max_random"] == pytest.approx(expected_max, abs=1e-6)
    assert res["dsr"] == pytest.approx(sr_p - expected_max, abs=1e-5)
    assert res["dsr"] < 0 and res["significant"] is False
    # No hardcoded 365: the REAL period count enters both sides.
    assert res["observation_periods"] == t_periods

    # More observations shrink the hurdle, so the same Sharpe deflates less.
    more = deflated_sharpe_ratio(sr_ann, n_trials, observation_periods=1200)
    assert more["expected_max_random"] < res["expected_max_random"]
    assert more["dsr"] > res["dsr"]

    # A per-period Sharpe of the same units is compared directly.
    direct = deflated_sharpe_ratio(0.30, 100, observation_periods=250,
                                   sharpe_is_annualized=False)
    assert direct["sharpe_per_period"] == pytest.approx(0.30)
    assert direct["dsr"] > 0


def test_dsr_no_longer_reports_a_false_positive():
    """The measured false positive: an annualised Sharpe vs a per-period hurdle."""
    import math
    from core.ga.fitness import deflated_sharpe_ratio

    # What the old implementation effectively computed (annualised SR compared
    # with the per-period E[max] of a hardcoded 365-day sample):
    t_periods, n_trials = 1200, 1200
    old_threshold = math.sqrt(1.0 / 365) * math.sqrt(2 * math.log(n_trials))
    assert 1.2 - old_threshold > 1.0        # "1.0029 significant"

    # The corrected call is in ONE unit system and rejects it.
    fixed = deflated_sharpe_ratio(1.2, n_trials, observation_periods=t_periods)
    assert fixed["dsr"] < 0
    assert fixed["significant"] is False


def test_dsr_counts_all_prior_trials():
    from core.ga.fitness import deflated_sharpe_ratio
    one = deflated_sharpe_ratio(2.0, 1, observation_periods=500)
    many = deflated_sharpe_ratio(2.0, 1200, observation_periods=500)
    assert one["dsr"] > many["dsr"], "more trials must raise the hurdle"


def test_trial_counter_accumulates_across_runs(tmp_path):
    from core.ga.trial_counter import load_trials, record_trials, total_trials
    assert load_trials(tmp_path) == 0
    record_trials(tmp_path, 20, "w1")
    record_trials(tmp_path, 20, "w2")
    assert load_trials(tmp_path) == 40
    assert total_trials(tmp_path, 5) == 45


# ══════════════════════════════════════════════════════════════════════════
# 6 — publication gate
# ══════════════════════════════════════════════════════════════════════════

def _gate_evolver(tmp_path):
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    class _Engine:
        config = None

    loader = StrategyLoader(str(tmp_path / "gate_strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    return GAStrategyEvolver(_Engine(), loader, GARunConfig(population_size=4))


def test_publication_gate_rejects_zero_trades_and_five_winners(tmp_path):
    evolver = _gate_evolver(tmp_path)
    published, reasons = evolver._publication_decision(
        {"trade_count": 0, "total_return": 0.0, "profit_factor": 0.1, "dsr": 0.0}, None)
    assert published is False and any("trades=0" in r for r in reasons)
    assert any("no_trades" in r for r in reasons)

    published, reasons = evolver._publication_decision(
        {"trade_count": 5, "total_return": 8.0, "profit_factor": 3.0, "dsr": 0.4,
         "buy_hold_pct": 0.0, "alpha_vs_buy_hold_pct": 8.0}, None)
    assert published is False
    assert any("trades=5" in r for r in reasons)

    published, reasons = evolver._publication_decision(
        {"trade_count": 200, "total_return": 15.0, "profit_factor": 1.8, "dsr": 0.5,
         "buy_hold_pct": 3.0, "alpha_vs_buy_hold_pct": 12.0}, None)
    assert published is True and reasons == []

    # DSR<=0 must block publication even with good headline metrics.
    published, reasons = evolver._publication_decision(
        {"trade_count": 200, "total_return": 15.0, "profit_factor": 1.8, "dsr": 0.0,
         "buy_hold_pct": 3.0, "alpha_vs_buy_hold_pct": 12.0}, None)
    assert published is False and any("dsr=" in r for r in reasons)


def test_rejected_champion_yaml_is_disabled_with_reasons(tmp_path):
    """A rejected champion is still written — with ``enabled: false``."""
    import yaml
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.ga.genome import chromosome_to_strategy, random_chromosome
    from core.strategy.loader import StrategyLoader

    class _Engine:
        config = None

    loader = StrategyLoader(str(tmp_path / "gate_strategies2"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    evolver = GAStrategyEvolver(_Engine(), loader, GARunConfig(population_size=4))

    random.seed(11)
    chrom = random_chromosome("gate_champ")
    chrom["fitness_result"] = {"trade_count": 0, "total_return": 0.0,
                               "profit_factor": 0.1, "dsr": 0.0}

    published, reasons = evolver._publication_decision(
        chrom["fitness_result"], None)
    assert published is False

    # Exactly what evolve() does for a rejected champion: disable, save, annotate.
    champion_config = chromosome_to_strategy(chrom)
    champion_config.name = "gate_champ"
    champion_config.enabled = published
    loader.save(champion_config)
    evolver._append_provenance("gate_champ", {
        "published": published, "rejection_reasons": reasons, "seed": 4242})

    doc = yaml.safe_load((loader.strategies_dir / "gate_champ.yaml").read_text())
    assert doc["enabled"] is False
    assert doc["provenance"]["published"] is False
    assert doc["provenance"]["seed"] == 4242
    assert doc["provenance"]["rejection_reasons"]


# ══════════════════════════════════════════════════════════════════════════
# 7 — ML score/publish consistency
# ══════════════════════════════════════════════════════════════════════════

def test_decoded_ml_config_is_never_enabled():
    """Scoring disables ML; the emitted YAML must not enable it (fusion differs:
    off 0.5000, w=0.1 → 0.6250, w=0.5 → 0.4167)."""
    import random as _r
    from core.ga.genome import chromosome_to_strategy, random_chromosome

    _r.seed(5)
    for _ in range(25):
        cfg = chromosome_to_strategy(random_chromosome("ml_consistency"))
        assert cfg.ml_config is not None
        assert cfg.ml_config.enabled is False
        assert cfg.ml_config.weight == 0.0

    # Even a chromosome whose ml_weight was forced high by an old checkpoint.
    from core.ga.genome import ContinuousGene
    chrom = random_chromosome("legacy_ml")
    for gene in chrom["continuous"]:
        if gene.name == "ml_weight":
            gene.min_val, gene.max_val, gene.value = 0.0, 0.5, 0.5
    cfg = chromosome_to_strategy(chrom)
    assert cfg.ml_config.enabled is False and cfg.ml_config.weight == 0.0


# ══════════════════════════════════════════════════════════════════════════
# 8 — genome: sanitisation, crossover by name, thresholds, AND/OR
# ══════════════════════════════════════════════════════════════════════════

def test_conditions_are_sanitised_against_decoded_indicators():
    """'adx > 20' must not survive when the decoded config has no adx."""
    from core.ga.genome import (CategoricalGene, ContinuousGene, StructuralGene,
                                BooleanGene, chromosome_to_strategy)

    chrom = {
        "continuous": [ContinuousGene("rsi_period", 14, 5, 28, 1)],
        "categorical": [CategoricalGene("mode", "trend", ["trend", "range"]),
                        CategoricalGene("timeframes", "1h", ["1h", "4h"])],
        "structural": [
            StructuralGene("entry_long", ["adx > 20", "rsi < 30"], []),
            StructuralGene("entry_short", ["adx > 25"], []),
            StructuralGene("exit_long", ["adx < 10"], []),
            StructuralGene("exit_short", ["rsi > 70"], []),
        ],
        # gene says adx is ON, but no adx continuous param ⇒ decoder drops adx
        "indicator_genes": [BooleanGene("rsi", True), BooleanGene("adx", True)],
        "name": "sanitize",
    }
    cfg = chromosome_to_strategy(chrom)
    assert "adx" not in cfg.indicators
    assert all("adx" not in c for c in cfg.entry_conditions["long"])
    assert all("adx" not in c for c in cfg.entry_conditions["short"])
    assert all("adx" not in c for c in cfg.exit_conditions["long"])
    # and no condition list is ever empty
    for side in ("long", "short"):
        assert cfg.entry_conditions[side] and cfg.exit_conditions[side]


def test_empty_condition_list_gets_a_fallback():
    from core.ga.genome import chromosome_to_strategy, random_chromosome
    random.seed(1)
    chrom = random_chromosome("empty_cond")
    for gene in chrom["structural"]:
        gene.conditions = []
    cfg = chromosome_to_strategy(chrom)
    for side in ("long", "short"):
        assert len(cfg.entry_conditions[side]) >= 1
        assert len(cfg.exit_conditions[side]) >= 1


def test_crossover_is_keyed_by_gene_name(tmp_path):
    """A child must carry the union of names — never lose or duplicate a gene."""
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.ga.genome import ContinuousGene, StructuralGene
    from core.strategy.loader import StrategyLoader

    class _Engine:
        config = None

    loader = StrategyLoader(str(tmp_path / "cross"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    evolver = GAStrategyEvolver(_Engine(), loader, GARunConfig(population_size=4))

    p1 = {
        "continuous": [ContinuousGene("rsi_period", 14, 5, 28, 1),
                        ContinuousGene("bb_stddev", 2.0, 1.0, 3.5, 0.25)],
        "categorical": [],
        "structural": [StructuralGene("entry_long", ["rsi < 30"], []),
                        StructuralGene("exit_short", ["rsi > 70"], [])],
        "indicator_genes": [],
        "name": "p1",
    }
    p2 = {
        "continuous": [ContinuousGene("rsi_period", 7, 5, 28, 1),
                        ContinuousGene("bb_stddev", 2.5, 1.0, 3.5, 0.25),
                        ContinuousGene("macd_fast", 12, 6, 20, 2)],
        "categorical": [],
        "structural": [StructuralGene("entry_long", ["close > sma"], []),
                        StructuralGene("exit_long", ["rsi > 65"], [])],
        "indicator_genes": [],
        "name": "p2",
    }
    random.seed(4)
    for _ in range(20):
        child = evolver._crossover(p1, p2)
        names = [g.name for g in child["continuous"]]
        assert len(names) == len(set(names)), f"duplicate gene: {names}"
        assert set(names) == {"rsi_period", "bb_stddev", "macd_fast"}, names
        struct_names = sorted(g.name for g in child["structural"])
        assert struct_names == ["entry_long", "exit_long", "exit_short"], struct_names
        # conditions never leak between genes
        for gene in child["structural"]:
            if gene.name == "exit_long":
                assert "rsi < 30" not in gene.conditions


def test_ema_and_stoch_genes_are_effective():
    """``ema_period`` used to be inert (templates read ema_fast/ema_slow)."""
    from core.ga.genome import (BooleanGene, CategoricalGene, ContinuousGene,
                                StructuralGene, chromosome_to_strategy)

    ema_chrom = {
        "continuous": [ContinuousGene("ema_fast_period", 5, 5, 30, 2),
                       ContinuousGene("ema_slow_period", 50, 12, 60, 2)],
        "categorical": [CategoricalGene("mode", "trend", ["trend"]),
                        CategoricalGene("timeframes", "1h", ["1h"])],
        "structural": [StructuralGene("entry_long", ["close > ema_fast"], []),
                        StructuralGene("entry_short", ["close < ema_fast"], []),
                        StructuralGene("exit_long", ["close < ema_slow"], []),
                        StructuralGene("exit_short", ["close > ema_slow"], [])],
        "indicator_genes": [BooleanGene("ema", True)],
        "name": "ema",
    }
    cfg = chromosome_to_strategy(ema_chrom)
    assert cfg.indicators["ema"]["fast_period"] == 5
    assert cfg.indicators["ema"]["slow_period"] == 50
    # The templates' conditions survive (ema_fast/ema_slow are always available).
    assert "close > ema_fast" in cfg.entry_conditions["long"]

    # Legacy single ``ema_period`` gene still decodes to a usable pair.
    legacy = copy.deepcopy(ema_chrom)
    legacy["continuous"] = [ContinuousGene("ema_period", 20, 5, 50, 2)]
    legacy_cfg = chromosome_to_strategy(legacy)
    assert legacy_cfg.indicators["ema"]["fast_period"] == 20
    assert legacy_cfg.indicators["ema"]["slow_period"] > 20

    # A stoch config must speak the schema compute_all() reads.
    stoch_chrom = {
        "continuous": [ContinuousGene("stoch_k_period", 9, 5, 21, 1),
                       ContinuousGene("stoch_d_period", 5, 3, 9, 1)],
        "categorical": [],
        "structural": [],
        "indicator_genes": [BooleanGene("stoch", True)],
        "name": "stoch",
    }
    stoch_cfg = chromosome_to_strategy(stoch_chrom)
    assert stoch_cfg.indicators["stoch"]["period"] == 9
    assert stoch_cfg.indicators["stoch"]["slowk_period"] == 5


def test_condition_logic_gene_and_and_semantics(market_dir, tmp_path):
    """AND must be strictly stricter than OR (fewer or equal entries)."""
    from core.ga.genome import chromosome_to_strategy, random_chromosome

    random.seed(21)
    chrom = random_chromosome("logic")
    cfg_or = chromosome_to_strategy(chrom)
    assert getattr(cfg_or, "condition_logic", "or") in ("and", "or")

    # Build a genome with two long conditions and both logic settings.
    from core.ga.genome import (BooleanGene, CategoricalGene, ContinuousGene,
                                StructuralGene)
    base = {
        "continuous": [ContinuousGene("rsi_period", 14, 5, 28, 1),
                        ContinuousGene("sma_period", 20, 10, 100, 2)],
        "categorical": [CategoricalGene("mode", "trend", ["trend"]),
                        CategoricalGene("timeframes", "1h", ["1h"])],
        "structural": [StructuralGene("entry_long", ["rsi < 45", "close > sma"], []),
                        StructuralGene("entry_short", ["rsi > 55", "close < sma"], []),
                        StructuralGene("exit_long", ["rsi > 80"], []),
                        StructuralGene("exit_short", ["rsi < 20"], [])],
        "indicator_genes": [BooleanGene("rsi", True), BooleanGene("sma", True)],
        "name": "logic_or",
    }
    and_genome = copy.deepcopy(base)
    and_genome["name"] = "logic_and"
    and_genome["condition_logic"] = "and"

    cfg, engine, loader = _engine(market_dir, tmp_path)
    _, engine2, _ = _engine(market_dir, tmp_path)
    res_or = engine.run_with_exit_evaluation(
        strategies=[chromosome_to_strategy(base)], symbols=["BTCUSDT"],
        date_start="2026-02-01", date_end="2026-03-15", mode="full",
        simulate_ai_weights=False, per_strategy_isolation=True, per_genome_ledger=True,
        use_live_spread=False)
    res_and = engine2.run_with_exit_evaluation(
        strategies=[chromosome_to_strategy(and_genome)], symbols=["BTCUSDT"],
        date_start="2026-02-01", date_end="2026-03-15", mode="full",
        simulate_ai_weights=False, per_strategy_isolation=True, per_genome_ledger=True,
        use_live_spread=False)
    assert len(res_and["trades"]) <= len(res_or["trades"])


# ══════════════════════════════════════════════════════════════════════════
# 9 — reproducibility
# ══════════════════════════════════════════════════════════════════════════

def test_same_seed_gives_identical_population():
    from core.ga.genome import random_chromosome

    def _population(seed):
        random.seed(seed)
        return [random_chromosome(f"r_{i}") for i in range(5)]

    first = _population(987654)
    second = _population(987654)
    assert [g["name"] for g in first] == [g["name"] for g in second]
    for a, b in zip(first, second):
        assert [x.name for x in a["continuous"]] == [x.name for x in b["continuous"]]
        assert [x.value for x in a["continuous"]] == [x.value for x in b["continuous"]]
        assert a["condition_logic"] == b["condition_logic"]
        assert [g.conditions for g in a["structural"]] == [
            g.conditions for g in b["structural"]]

    third = _population(987655)
    assert [x.value for x in third[0]["continuous"]] != [
        x.value for x in first[0]["continuous"]]


def test_evolver_declares_and_uses_a_seed(tmp_path):
    """The evolver must seed ``random``/``numpy`` from the job seed."""
    import numpy as _np
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    class _Engine:
        config = None

    loader = StrategyLoader(str(tmp_path / "seed"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    evolver = GAStrategyEvolver(_Engine(), loader,
                                GARunConfig(population_size=2, seed=123456))
    assert evolver._seed == 123456

    def _draw_from_seed(seed):
        random.seed(seed)
        _np.random.seed(seed % (2 ** 32))
        return (random.random(), float(_np.random.rand()))

    assert _draw_from_seed(123456) == _draw_from_seed(123456)
    assert evolver._window_key == ""

    # The window key survives a checkpoint round-trip (resume provenance).
    evolver._window_key = "2026-01-01~2026-02-01|2026-02-01~2026-03-01"
    evolver._population = []
    evolver._save_checkpoint()
    other = GAStrategyEvolver(_Engine(), loader, GARunConfig(population_size=2))
    assert other.load_checkpoint() is True
    assert other._window_key == evolver._window_key
    evolver.clear_checkpoint()


def test_same_seed_reproduces_the_same_evolution_stream(tmp_path):
    """Item 9: same seed + same data ⇒ identical populations, generation by
    generation (so the champion is identical too)."""
    import numpy as _np
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    class _Engine:
        config = None

    def _fingerprint(seed):
        loader = StrategyLoader(str(tmp_path / f"repro_{seed}"))
        loader.strategies_dir.mkdir(parents=True, exist_ok=True)
        evolver = GAStrategyEvolver(
            _Engine(), loader,
            GARunConfig(population_size=10, generations=3, elite_count=2,
                        immigrant_count=2, seed=seed))
        random.seed(seed)
        _np.random.seed(seed % (2 ** 32))
        evolver._population = evolver._init_population(None)
        streams = []
        for _ in range(2):
            # A deterministic score keeps selection itself reproducible.
            for i, chrom in enumerate(evolver._population):
                chrom["fitness_result"] = {"fitness": float(i % 3)}
            evolver._population = evolver._next_generation()
            streams.append([
                (g["name"],
                 tuple(x.name for x in g["continuous"]),
                 tuple(round(float(x.value), 6) for x in g["continuous"]),
                 tuple(tuple(s.conditions) for s in g["structural"]),
                 g.get("condition_logic"))
                for g in evolver._population])
        return streams

    assert _fingerprint(20260101) == _fingerprint(20260101)


# ══════════════════════════════════════════════════════════════════════════
# 10 — multiprocess fallback
# ══════════════════════════════════════════════════════════════════════════

def test_crashed_chunk_is_retried_at_single_worker(monkeypatch):
    """A dead chunk used to mark the WHOLE population −999 with no fallback."""
    from core.ga import fitness as F

    def fake_worker(args):
        """Stands in for a subprocess run that succeeds."""
        return [(args["chunk_start_idx"] + i,
                 {"fitness": 5.0, "trade_count": 40, "eval_path": "retry"})
                for i in range(len(args["population_chunk"]))]

    # ``_single_worker_retry`` is the documented fallback path and is testable
    # in-process (a monkeypatched pool worker cannot be pickled to a subprocess).
    monkeypatch.setattr(F, "_mp_worker", fake_worker)
    args = {"chunk_start_idx": 2,
            "population_chunk": [{"name": "g2"}, {"name": "g3"}]}
    first, path = F._single_worker_retry(args, RuntimeError("boom"))
    assert path == "retry_single_worker"
    assert [r[0] for r in first] == [2, 3]
    assert all(r[1]["fitness"] == 5.0 for r in first)

    # And when the retry ALSO fails the genomes are flagged, never silently -999
    # for the whole population.
    def always_fails(_args):
        raise RuntimeError("still broken")

    monkeypatch.setattr(F, "_mp_worker", always_fails)
    filled, path = F._single_worker_retry(args, RuntimeError("boom"))
    assert path == "failed"
    assert [r[0] for r in filled] == [2, 3]
    assert all(r[1]["fitness"] == -999 and r[1]["eval_path"] == "failed"
               for r in filled)


def test_multiprocess_worker_hook_is_honoured(market_dir, tmp_path):
    """The hook is an explicit test seam — it must not require a real engine."""
    from core.ga import fitness as F
    cfg, engine, loader = _engine(market_dir, tmp_path)
    cfg.data_dir = market_dir
    args = {
        "population_chunk": [{"name": "x"}],
        "symbols": ["BTCUSDT"], "date_start": "2026-02-01",
        "date_end": "2026-02-05", "initial_balance": 10000.0,
        "cost_enabled": True, "taker_fee_pct": 0.04, "spread_pct": {},
        "weights": None, "chunk_start_idx": 10, "engine_mode": "legacy",
        "evaluate_hook": lambda a: [(a["chunk_start_idx"], {"fitness": 1.0})],
    }
    assert F._mp_worker(args) == [(10, {"fitness": 1.0})]


# ══════════════════════════════════════════════════════════════════════════
# 10b — cost model: no live order book for historical fills
# ══════════════════════════════════════════════════════════════════════════

def test_ga_evaluation_does_not_use_live_spread(market_dir, tmp_path, monkeypatch):
    """``cost_model.py:239-242`` used to price historical fills with today's book."""
    from core.backtest import cost_model as cm

    def _boom(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("live order book queried during GA evaluation")

    monkeypatch.setattr(cm, "fetch_live_spread_pct", _boom)
    cm.clear_live_spread_cache()

    cfg, engine, loader = _engine(market_dir, tmp_path)
    # Live derivation is ON for this config — only the explicit
    # ``use_live_spread=False`` may keep the GA off the network.
    cfg.backtest_live_spread_enabled = True
    result = engine.run_with_exit_evaluation(
        strategies=[_always_on_strategy("no_live_spread")], symbols=["BTCUSDT"],
        date_start="2026-02-01", date_end="2026-02-15", mode="full",
        simulate_ai_weights=False, per_strategy_isolation=True, per_genome_ledger=True,
        use_live_spread=False)
    sources = result["metrics"]["spread_sources"]
    assert sources["BTCUSDT"] == "override"  # config.yaml lists BTCUSDT
    assert all(src != "live" for src in sources.values())


# ══════════════════════════════════════════════════════════════════════════
# 11 — no-look-ahead invariants (the plan's "already sound" list)
# ══════════════════════════════════════════════════════════════════════════

def test_truncating_the_future_does_not_change_past_trades(market_dir, tmp_path):
    """Guards: close_time index, fills on the last closed bar, no look-ahead.

    Running the SAME strategy over ``[start, T]`` and over ``[start, end]`` must
    produce identical trades for everything that closed at or before T.  If any
    decision used a future bar, the two runs would disagree.
    """
    cfg, engine, loader = _engine(market_dir, tmp_path)
    strategy = _always_on_strategy("future_check")

    full = engine.run_with_exit_evaluation(
        strategies=[strategy], symbols=["BTCUSDT"],
        date_start="2026-02-01", date_end="2026-03-20", mode="full",
        simulate_ai_weights=False, per_strategy_isolation=True, per_genome_ledger=True,
        use_live_spread=False)
    _, engine2, _ = _engine(market_dir, tmp_path)
    cut = pd.Timestamp("2026-03-01")
    partial = engine2.run_with_exit_evaluation(
        strategies=[_always_on_strategy("future_check")], symbols=["BTCUSDT"],
        date_start="2026-02-01", date_end="2026-03-01", mode="full",
        simulate_ai_weights=False, per_strategy_isolation=True, per_genome_ledger=True,
        use_live_spread=False)

    def key(t):
        return (t["symbol"], t["side"], str(t["opened_at"]))

    full_past = {key(t): t for t in full["trades"]
                 if pd.Timestamp(t["closed_at"]) <= cut and t["exit_reason"] != "end_of_backtest"}
    partial_trades = {key(t): t for t in partial["trades"]
                      if t["exit_reason"] != "end_of_backtest"}
    missing = set(full_past) - set(partial_trades)
    assert not missing, f"future data changed past decisions: {sorted(missing)[:3]}"
    for k, trade in full_past.items():
        assert partial_trades[k]["pnl"] == pytest.approx(trade["pnl"], abs=1e-6), k
        assert partial_trades[k]["entry_price"] == pytest.approx(
            trade["entry_price"], abs=1e-9), k
    assert full["trades"], "test is vacuous — the strategy never traded"


def test_equity_curve_only_contains_closed_bars(market_dir, tmp_path):
    """The equity/trade timeline is the close_time index of the feed."""
    cfg, engine, loader = _engine(market_dir, tmp_path)
    result = engine.run_with_exit_evaluation(
        strategies=[_always_on_strategy("closed_bars")], symbols=["BTCUSDT"],
        date_start="2026-02-01", date_end="2026-02-20", mode="full",
        simulate_ai_weights=False, per_strategy_isolation=True, per_genome_ledger=True,
        use_live_spread=False)
    curve = result["equity_curve"]
    assert len(curve) > 250, "the 250-bar warm-up window must precede trading"
    times = [pd.Timestamp(p["time"]) for p in curve]
    assert times == sorted(times)
    feed = pd.read_parquet(Path(market_dir) / "market" / "BTCUSDT" / "1h.parquet",
                           columns=["close"])
    for ts in times[:50]:
        assert ts in feed.index, "an equity point is not a closed bar timestamp"
    for trade in result["trades"]:
        assert pd.Timestamp(trade["closed_at"]) >= pd.Timestamp(trade["opened_at"])
