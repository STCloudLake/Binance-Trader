"""Fitness evaluation — runs backtests to score strategy chromosomes."""

import time
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from core.ga.genome import chromosome_to_strategy
from core.strategy.loader import StrategyLoader


def evaluate_chromosome(
    chromosome: dict,
    symbols: list[str],
    date_start: str,
    date_end: str,
    engine,
    loader: StrategyLoader,
    initial_balance: float = 10000.0,
) -> dict:
    """Evaluate a single chromosome via backtest.

    Returns a dict with fitness components.
    """
    try:
        config = chromosome_to_strategy(chromosome)
        # Save temporarily so the backtest engine can load it
        loader.save(config)

        result = engine.run_with_exit_evaluation(
            strategies=[config.name],
            symbols=symbols,
            date_start=date_start,
            date_end=date_end,
            initial_balance=initial_balance,
            mode="full",
            simulate_ai_weights=False,
            ml_engine="lightgbm",
        )

        if "error" in result:
            return {"fitness": -999, "error": result["error"]}

        metrics = result.get("metrics", {})
        sharpe = metrics.get("sharpe_ratio", -10)
        win_rate = metrics.get("win_rate_pct", 0)
        profit_factor = metrics.get("profit_factor", 0)
        max_dd = abs(metrics.get("max_drawdown_pct", 20))
        total_return = metrics.get("total_return_pct", -100)
        trade_count = metrics.get("total_trades", 0)

        # ── Fitness score ──
        # Higher is better. Penalize extreme values and instability.
        fitness = (
            max(sharpe, -5) * 2.0          # risk-adjusted return (capped floor)
            + win_rate * 0.15               # consistency
            + max(profit_factor, 0.1) * 5   # reward good risk/reward
            - max_dd * 0.3                  # penalize drawdowns
        )

        # Trade count penalty: too few = unreliable, too many = overtrading
        if trade_count < 5:
            fitness -= 20  # not enough data
        elif trade_count < 15:
            fitness -= 5   # barely enough
        elif trade_count > 500:
            fitness -= (trade_count - 500) * 0.02  # overtrading penalty

        # Negative total return is heavily penalized
        if total_return < -5:
            fitness -= abs(total_return) * 0.5

        return {
            "fitness": round(fitness, 4),
            "sharpe": round(sharpe, 4),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 4),
            "max_dd": round(max_dd, 2),
            "total_return": round(total_return, 2),
            "trade_count": trade_count,
            "strategy_name": config.name,
        }

    except Exception as e:
        logger.debug(f"Fitness eval failed: {e}")
        return {"fitness": -999, "error": str(e)}


def evaluate_population(
    population: list[dict],
    symbols: list[str],
    date_start: str,
    date_end: str,
    engine,
    loader: StrategyLoader,
    max_workers: int = 4,
    initial_balance: float = 10000.0,
    progress_callback=None,
) -> list[dict]:
    """Evaluate all chromosomes in parallel.

    Returns the population list with 'fitness_result' key added to each.
    """
    results = []
    total = len(population)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, chrom in enumerate(population):
            # Give each a unique name for parallel evaluation
            chrom_copy = dict(chrom)
            chrom_copy["name"] = f"ga_gen_{i}_{random.randint(1000, 9999)}"
            future = executor.submit(
                evaluate_chromosome,
                chrom_copy, symbols, date_start, date_end,
                engine, loader, initial_balance,
            )
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result(timeout=120)
            except Exception as e:
                result = {"fitness": -999, "error": str(e)}

            population[idx]["fitness_result"] = result
            completed += 1

            if progress_callback:
                progress_callback(completed, total)

            if result.get("fitness", -999) > -900:
                logger.debug(
                    f"[{completed}/{total}] {result.get('strategy_name','?')}: "
                    f"fitness={result['fitness']:.2f} "
                    f"sharpe={result.get('sharpe',0):.2f} "
                    f"trades={result.get('trade_count',0)}")

    return population


def parameter_sensitivity_test(
    chromosome: dict,
    symbols: list[str],
    date_start: str,
    date_end: str,
    engine,
    loader: StrategyLoader,
    perturbations: int = 5,
) -> float:
    """Test fitness sensitivity to small parameter changes.

    A high sensitivity = likely overfit. Returns std of fitness across
    perturbed variants.
    """
    base_result = evaluate_chromosome(
        chromosome, symbols, date_start, date_end, engine, loader)
    base_fitness = base_result.get("fitness", -999)
    if base_fitness < -100:
        return 1.0  # invalid base, assume overfit

    perturbed_fitnesses = []
    for _ in range(perturbations):
        mutated = dict(chromosome)
        for gene in mutated.get("continuous", []):
            gene.mutate(strength=0.3)
        result = evaluate_chromosome(
            mutated, symbols, date_start, date_end, engine, loader)
        perturbed_fitnesses.append(result.get("fitness", -999))

    if not perturbed_fitnesses:
        return 0.0

    std = float(np.std(perturbed_fitnesses))
    # Normalize: std > 50% of base fitness = overfit
    sensitivity = std / max(abs(base_fitness), 1.0)
    return min(sensitivity, 1.0)
