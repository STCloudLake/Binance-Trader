"""Unit tests for Genetic Algorithm module — genome, fitness, evolver."""

import pytest
import tempfile
import os
import random
from pathlib import Path


# ── Genome Tests ────────────────────────────────────────────────────────

def test_strategy_to_chromosome_roundtrip():
    """Encode a StrategyConfig → chromosome → decode back; verify consistency."""
    from core.ga.genome import strategy_to_chromosome, chromosome_to_strategy
    from core.strategy.loader import StrategyConfig, MLConfig

    config = StrategyConfig(
        name="test_strategy",
        enabled=True,
        mode="trend",
        timeframes=["1h", "4h"],
        indicators={
            "rsi": {"period": 14, "source": "close"},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"period": 20, "stddev": 2.0},
            "adx": {"period": 14},
            "ema": {"period": 9, "source": "close"},
        },
        entry_conditions={
            "long": ["rsi < 30", "close > ema_fast"],
            "short": ["rsi > 70", "close < ema_fast"],
        },
        exit_conditions={
            "long": ["rsi > 65"],
            "short": ["rsi < 35"],
        },
        ml_config=MLConfig(enabled=True, weight=0.3, confidence_threshold=0.65),
    )

    chrom = strategy_to_chromosome(config)
    decoded = chromosome_to_strategy(chrom)

    assert decoded.mode == "trend"
    assert decoded.timeframes == ["1h", "4h"]
    assert "rsi" in decoded.indicators
    assert decoded.indicators["rsi"]["period"] == 14
    assert "macd" in decoded.indicators
    assert decoded.indicators["macd"]["fast"] == 12
    assert decoded.indicators["macd"]["slow"] == 26
    assert "bollinger" in decoded.indicators
    assert abs(decoded.indicators["bollinger"]["stddev"] - 2.0) < 0.01
    assert "adx" in decoded.indicators
    assert "ema" in decoded.indicators
    assert decoded.entry_conditions["long"] == ["rsi < 30", "close > ema_fast"]
    assert decoded.exit_conditions["long"] == ["rsi > 65"]
    assert decoded.ml_config.enabled is True
    assert decoded.ml_config.weight == 0.3


def test_strategy_to_chromosome_minimal():
    """Minimal strategy with no indicators should still encode/decode."""
    from core.ga.genome import strategy_to_chromosome, chromosome_to_strategy
    from core.strategy.loader import StrategyConfig, MLConfig

    config = StrategyConfig(
        name="minimal",
        enabled=True,
        mode="scalp",
        timeframes=["1m"],
        indicators={},
        entry_conditions={"long": [], "short": []},
        exit_conditions={"long": [], "short": []},
        ml_config=MLConfig(enabled=False, weight=0.0, confidence_threshold=0.6),
    )

    chrom = strategy_to_chromosome(config)
    decoded = chromosome_to_strategy(chrom)

    assert decoded.mode == "scalp"
    assert decoded.timeframes == ["1m"]
    assert decoded.indicators == {}


def test_random_chromosome():
    """Random chromosome should produce valid decodeable output."""
    from core.ga.genome import random_chromosome, chromosome_to_strategy

    random.seed(0)
    for i in range(10):
        chrom = random_chromosome("ga_test")
        config = chromosome_to_strategy(chrom)

        assert config.name == "ga_test"
        assert config.mode in ("trend", "range", "scalp", "momentum")
        assert len(config.timeframes) >= 1
        for tf in config.timeframes:
            assert tf in ("1m", "5m", "15m", "1h", "4h")
        # Should have at least one indicator (RSI at 90% probability)
        assert len(config.indicators) >= 1

        # Verify chromosome structure
        assert "continuous" in chrom
        assert "categorical" in chrom
        assert "structural" in chrom
        assert len(chrom["categorical"]) == 2  # mode + timeframes


def test_continuous_gene_mutation():
    """ContinuousGene mutation should stay within bounds."""
    from core.ga.genome import ContinuousGene

    random.seed(42)
    gene = ContinuousGene("rsi_period", 14, 5, 28, 1)

    # Mutate multiple times, check bounds
    for _ in range(100):
        gene.mutate(strength=1.0)
        assert 5 <= gene.value <= 28
        assert isinstance(gene.value, (int, float))


def test_categorical_gene_mutation():
    """CategoricalGene mutation should change value eventually."""
    from core.ga.genome import CategoricalGene

    random.seed(42)
    gene = CategoricalGene("mode", "trend", ["trend", "range", "scalp"])
    mutated = False
    for _ in range(20):
        old = gene.value
        gene.mutate()
        if old != gene.value:
            mutated = True
    assert mutated  # should have changed at some point


def test_structural_gene_mutation():
    """StructuralGene mutation should add or remove conditions."""
    from core.ga.genome import StructuralGene

    random.seed(42)
    gene = StructuralGene(
        "entry_long",
        conditions=["rsi < 30"],
        template_pool=["rsi < 30", "rsi < 35", "adx > 20", "volume_ratio > 1.5"],
    )

    for _ in range(20):
        gene.mutate()
        assert len(gene.conditions) >= 1
        # Mutations from pool should not exceed pool size
        assert len(gene.conditions) <= len(gene.template_pool)

    # After many mutations, should sometimes add from pool
    assert "adx > 20" in gene.conditions or "rsi < 35" in gene.conditions


def test_structural_gene_mutate_remove_single_condition():
    """Single condition should not be removed (min 1)."""
    from core.ga.genome import StructuralGene

    gene = StructuralGene(
        "entry_long",
        conditions=["rsi < 30"],
        template_pool=["rsi < 30", "rsi < 35"],
    )

    # mutate_remove should skip when len == 1
    gene.mutate_remove()
    assert len(gene.conditions) == 1  # unchanged


def test_chromosome_to_strategy_partial_indicators():
    """Decoding a chromosome with only some indicators should still work."""
    from core.ga.genome import chromosome_to_strategy, ContinuousGene, CategoricalGene, StructuralGene

    # Only RSI + EMA — missing MACD, BB, ADX
    chrom = {
        "continuous": [
            ContinuousGene("rsi_period", 10, 5, 28, 1),
            ContinuousGene("ema_period", 21, 5, 50, 2),
            ContinuousGene("ml_weight", 0.1, 0.0, 0.5, 0.05),
            ContinuousGene("ml_threshold", 0.6, 0.5, 0.85, 0.05),
        ],
        "categorical": [
            CategoricalGene("mode", "range", ["trend", "range"]),
            CategoricalGene("timeframes", "5m,15m", ["1m,5m", "5m,15m"]),
        ],
        "structural": [
            StructuralGene("entry_long", ["rsi < 30"], []),
            StructuralGene("entry_short", [], []),
            StructuralGene("exit_long", [], []),
            StructuralGene("exit_short", [], []),
        ],
        "name": "partial_test",
    }

    config = chromosome_to_strategy(chrom)
    assert "rsi" in config.indicators
    assert "ema" in config.indicators
    assert "macd" not in config.indicators
    assert "bollinger" not in config.indicators
    assert "adx" not in config.indicators
    assert config.mode == "range"
    assert config.timeframes == ["5m", "15m"]


# ── Fitness Tests ───────────────────────────────────────────────────────

def test_complexity_penalty():
    """Complexity penalty should increase with more conditions and indicators."""
    from core.ga.fitness import complexity_penalty
    from core.ga.genome import ContinuousGene, StructuralGene

    simple_chrom = {
        "continuous": [ContinuousGene("rsi_period", 14, 5, 28, 1)],
        "structural": [StructuralGene("entry_long", ["rsi < 30"], [])],
    }

    complex_chrom = {
        "continuous": [
            ContinuousGene("rsi_period", 14, 5, 28, 1),
            ContinuousGene("macd_fast", 12, 6, 20, 2),
            ContinuousGene("macd_slow", 26, 18, 40, 2),
            ContinuousGene("adx_period", 14, 7, 28, 1),
        ],
        "structural": [
            StructuralGene("entry_long", ["rsi < 30", "adx > 20", "volume_ratio > 1.5"], []),
            StructuralGene("entry_short", ["rsi > 70", "adx > 20"], []),
        ],
    }

    simple_p = complexity_penalty(simple_chrom)
    complex_p = complexity_penalty(complex_chrom)
    assert complex_p > simple_p


def test_complexity_penalty_empty():
    """Empty chromosome should have zero penalty."""
    from core.ga.fitness import complexity_penalty
    assert complexity_penalty({"continuous": [], "structural": []}) == 0.0


def test_deflated_sharpe_ratio():
    """DSR should be lower than raw Sharpe and compute significance."""
    from core.ga.fitness import deflated_sharpe_ratio

    # With 500 trials and short observation, DSR should be penalized
    result = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=500, observation_periods=100)
    assert result["dsr"] < 2.0
    assert 0.0 <= result["p_value"] <= 1.0
    assert "significant" in result

    # Zero Sharpe → DSR = 0, p_value = 1
    result_zero = deflated_sharpe_ratio(observed_sharpe=0.0, n_trials=100)
    assert result_zero["dsr"] == 0.0
    assert result_zero["p_value"] == 1.0
    assert result_zero["significant"] is False

    # Negative Sharpe should return zero DSR
    result_neg = deflated_sharpe_ratio(observed_sharpe=-1.0, n_trials=100)
    assert result_neg["dsr"] == 0.0


# ── Evolver Config Tests ────────────────────────────────────────────────

def test_ga_config_defaults():
    """GARunConfig should have sensible defaults."""
    from core.ga.evolver import GARunConfig
    cfg = GARunConfig()
    assert cfg.population_size >= 10
    assert cfg.generations >= 1
    assert cfg.elite_count < cfg.population_size
    assert cfg.immigrant_count < cfg.population_size
    assert 0.0 <= cfg.mutation_rate <= 1.0
    assert 0.0 <= cfg.crossover_rate <= 1.0
    assert cfg.tournament_size >= 1


def test_ga_config_custom():
    """Custom GARunConfig should accept all parameters."""
    from core.ga.evolver import GARunConfig
    cfg = GARunConfig(
        population_size=100, generations=50, elite_count=10,
        immigrant_count=5, mutation_rate=0.3, early_stop_generations=15,
    )
    assert cfg.population_size == 100
    assert cfg.generations == 50
    assert cfg.elite_count == 10
    assert cfg.immigrant_count == 5
    assert cfg.mutation_rate == 0.3
    assert cfg.early_stop_generations == 15


# ── Checkpoint Tests ────────────────────────────────────────────────────

def test_checkpoint_save_load_roundtrip():
    """Pickle-based checkpoint save/load should preserve state."""
    import pickle
    from core.ga.evolver import GARunConfig

    tmp_dir = tempfile.mkdtemp()
    ckpt_path = Path(tmp_dir) / "ga_checkpoint.pkl"

    cfg = GARunConfig(population_size=10, generations=5)
    test_state = {
        "population": [{"id": 1, "fitness_result": {"fitness": 10.0}}],
        "generation": 3,
        "best_fitness": 10.0,
        "best_chromosome": {"id": 1},
        "history": [{"generation": 1, "best_fitness": 5.0}],
        "config": cfg,
    }

    with open(ckpt_path, "wb") as f:
        pickle.dump(test_state, f)

    with open(ckpt_path, "rb") as f:
        loaded = pickle.load(f)

    assert loaded["generation"] == 3
    assert loaded["best_fitness"] == 10.0
    assert len(loaded["population"]) == 1
    assert loaded["population"][0]["id"] == 1


# ── Cross-module Integration Tests ──────────────────────────────────────

def test_genome_fitness_integration():
    """A random chromosome should produce a valid fitness evaluation result
    (without actually running a backtest — just test that the interface works)."""
    from core.ga.genome import random_chromosome, chromosome_to_strategy

    random.seed(123)
    chrom = random_chromosome("integration_test")
    config = chromosome_to_strategy(chrom)

    # Verify the config is loadable/saveable via StrategyLoader
    from core.strategy.loader import StrategyLoader
    tmp_dir = tempfile.mkdtemp()
    loader = StrategyLoader(tmp_dir)
    loader.save(config)

    loaded = loader.load(config.name)
    assert loaded.name == config.name
    assert loaded.mode == config.mode
    assert loaded.indicators == config.indicators
