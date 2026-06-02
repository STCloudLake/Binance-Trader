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
    ga_loader: StrategyLoader | None = None,
    initial_balance: float = 10000.0,
) -> dict:
    """Evaluate a single chromosome via backtest.

    Uses *ga_loader* (isolated temp dir) to save/load strategy files
    so GA never touches the main strategies/ directory.

    Returns a dict with fitness components.
    """
    save_loader = ga_loader or loader
    try:
        config = chromosome_to_strategy(chromosome)
        if config.ml_config:
            config.ml_config.enabled = False

        # Pass StrategyConfig directly — no file I/O needed
        result = engine.run_with_exit_evaluation(
            strategies=[config],  # StrategyConfig object, not file name
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

        # ── Complexity penalty ──
        # Penalize overparameterized strategies to reduce overfitting risk
        fitness -= complexity_penalty(chromosome)

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


def complexity_penalty(chromosome: dict) -> float:
    """Penalize overparameterized strategies.

    More conditions + more indicators + more params = higher risk of overfitting.
    Returns a penalty value to subtract from fitness.
    """
    structural = chromosome.get("structural", [])
    continuous = chromosome.get("continuous", [])
    indicators_used = set()

    # Count conditions
    n_conditions = sum(len(g.conditions) for g in structural)

    # Count unique indicator types from continuous genes
    for g in continuous:
        name = g.name.split("_")[0]  # "rsi_period" -> "rsi"
        indicators_used.add(name)

    penalty = 0.0
    penalty += n_conditions * 0.8       # each condition adds overfit risk
    penalty += len(indicators_used) * 1.2  # each indicator type
    penalty += len(continuous) * 0.3     # each tunable parameter
    return penalty


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    observation_periods: int = 365,
    variance_sharpe: float = 1.0,
) -> dict:
    """Compute Deflated Sharpe Ratio (DSR) — statistical significance test.

    Accounts for multiple testing: out of *n_trials* random strategies,
    what's the probability of seeing a Sharpe >= *observed_sharpe* purely
    by chance?

    Based on: Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"

    Parameters
    ----------
    observed_sharpe : float
        The Sharpe ratio of the champion strategy.
    n_trials : int
        Number of strategies evaluated (population × generations).
    observation_periods : int
        Number of return observations (trading days).
    variance_sharpe : float
        Variance of the Sharpe ratio under null (≈1 for daily returns).

    Returns
    -------
    dict with dsr (deflated SR), p_value, significant (bool at 95%).
    """
    import math
    from scipy import stats as _stats

    if observed_sharpe <= 0 or n_trials <= 1:
        return {"dsr": 0.0, "p_value": 1.0, "significant": False}

    # Expected maximum Sharpe from n_trials random trials
    # E[max(SR)] ≈ sqrt(2 * log(n_trials))
    expected_max = math.sqrt(variance_sharpe / observation_periods) * math.sqrt(2 * math.log(n_trials))

    # Deflated Sharpe = observed - expected_max
    dsr = observed_sharpe - expected_max

    # P-value: is DSR significantly > 0?
    # Test: H0: true Sharpe = expected_max (just lucky data mining)
    se = math.sqrt(variance_sharpe / observation_periods)
    z_score = max(dsr, 0) / max(se, 1e-9)
    p_value = 1.0 - _stats.norm.cdf(z_score)

    return {
        "dsr": round(dsr, 4),
        "expected_max_random": round(expected_max, 4),
        "p_value": round(max(p_value, 0.0), 4),
        "significant": dsr > 0 and p_value < 0.05,
        "n_trials": n_trials,
    }


def evaluate_population_batch(
    population: list[dict],
    symbols: list[str],
    date_start: str,
    date_end: str,
    engine,
    loader: StrategyLoader,
    ga_loader: StrategyLoader | None = None,
    batch_size: int = 10,
    initial_balance: float = 10000.0,
    progress_callback=None,
) -> list[dict]:
    """Evaluate chromosomes in batched backtests — 4-5x faster than individual.

    Uses *ga_loader* (isolated temp dir) for strategy files so GA never
    touches the main strategies/ directory.
    """
    save_loader = ga_loader or loader
    results = [None] * len(population)
    total = len(population)
    completed = 0

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = population[batch_start:batch_end]

        # Build StrategyConfig objects directly — no file I/O needed
        batch_configs = []
        for i, chrom in enumerate(batch):
            config = chromosome_to_strategy(chrom)
            if config.ml_config:
                config.ml_config.enabled = False
            config.name = f"ga_batch_{batch_start + i}_{random.randint(1000,9999)}"
            batch_configs.append(config)

        # Single backtest for entire batch — pass configs directly
        result = engine.run_with_exit_evaluation(
            strategies=batch_configs,  # list of StrategyConfig, not names
            symbols=symbols,
            date_start=date_start,
            date_end=date_end,
            initial_balance=initial_balance,
            mode="full",
            simulate_ai_weights=False,
            ml_engine="lightgbm",
            per_strategy_isolation=True,
        )

        # Extract per-strategy metrics from per_matrix
        per_matrix = result.get("per_matrix", {})
        for i, config in enumerate(batch_configs):
            idx = batch_start + i
            chrom = population[idx]
            cell_data = per_matrix.get(config.name, {})

            trades = sum(c.get("trades", 0) for c in cell_data.values())
            pnl = sum(c.get("pnl", 0) for c in cell_data.values())
            wins = sum(c.get("winning", 0) for c in cell_data.values())
            losses = sum(c.get("losing", 0) for c in cell_data.values())

            # Compute fitness from per-strategy data
            win_rate = (wins / max(trades, 1)) * 100
            # Approximate Sharpe from PnL sequence (simplified)
            # Use profit_factor as proxy for risk/reward
            avg_win = sum(c.get("pnl", 0) for c in cell_data.values() if c.get("pnl", 0) > 0) / max(wins, 1)
            avg_loss = abs(sum(c.get("pnl", 0) for c in cell_data.values() if c.get("pnl", 0) < 0)) / max(losses, 1)
            profit_factor = (wins * avg_win) / max(losses * avg_loss, 1e-9)

            fitness = (
                win_rate * 0.15
                + max(profit_factor, 0.1) * 5
                - abs(pnl / max(initial_balance, 1)) * 50  # drawdown proxy
            )

            # Trade count penalty
            if trades < 5:
                fitness -= 20
            elif trades < 15:
                fitness -= 5
            elif trades > 500:
                fitness -= (trades - 500) * 0.02

            if pnl < -50:
                fitness -= abs(pnl) * 0.3

            fitness -= complexity_penalty(chrom)

            results[idx] = {
                "fitness": round(fitness, 4),
                "sharpe": 0,  # not directly available in batch mode
                "win_rate": round(win_rate, 2),
                "profit_factor": round(profit_factor, 4),
                "max_dd": 0,
                "total_return": round(pnl / initial_balance * 100, 2),
                "trade_count": trades,
                "strategy_name": name,
            }

            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    # Apply results to population
    for i, r in enumerate(results):
        if r is not None:
            population[i]["fitness_result"] = r
        else:
            population[i]["fitness_result"] = {"fitness": -999, "error": "batch eval failed"}

    return population


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
    """Evaluate all chromosomes in parallel (individual backtests).

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
