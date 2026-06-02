"""Genetic Algorithm strategy evolver.

Orchestrates the full evolution cycle:
    1. Initialize random population
    2. Evaluate fitness (parallel backtests)
    3. Select parents via tournament
    4. Crossover to produce offspring
    5. Mutate offspring
    6. Elite preservation + diversity injection
    7. Repeat for N generations
    8. Final champion → save as YAML strategy
"""

import copy
import random
import time
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from loguru import logger

from core.ga.genome import (
    strategy_to_chromosome, chromosome_to_strategy,
    random_chromosome, ContinuousGene, CategoricalGene, StructuralGene,
)
from core.ga.fitness import evaluate_population, parameter_sensitivity_test
from core.strategy.loader import StrategyLoader


@dataclass
class GARunConfig:
    """Configuration for a GA evolution run."""
    population_size: int = 80
    generations: int = 30
    elite_count: int = 8       # top N preserved unchanged
    immigrant_count: int = 8   # new random individuals each generation
    tournament_size: int = 3
    mutation_rate: float = 0.25
    crossover_rate: float = 0.7
    overfit_penalty: float = 0.3  # weight of sensitivity penalty
    max_workers: int = 4       # parallel backtest workers
    early_stop_generations: int = 10  # stop if no improvement for N gens


class GAStrategyEvolver:
    """Genetic algorithm for evolving trading strategies."""

    def __init__(self, engine, loader: StrategyLoader,
                 config: GARunConfig | None = None):
        self.engine = engine
        self.loader = loader
        self.config = config or GARunConfig()
        self._population: list[dict] = []
        self._generation = 0
        self._best_fitness = -999
        self._best_chromosome: dict | None = None
        self._stagnation_count = 0
        self._history: list[dict] = []
        self._running = False
        self._stop_after_gen = False  # graceful stop flag
        self._progress_callback = None
        self._checkpoint_path = Path(loader.strategies_dir).parent / "data" / "ga_checkpoint.pkl"

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def best_fitness(self) -> float:
        return self._best_fitness

    @property
    def population(self) -> list[dict]:
        return self._population

    @property
    def history(self) -> list[dict]:
        return self._history

    def set_progress_callback(self, callback):
        self._progress_callback = callback

    # ── Main evolution loop ───────────────────────────────────────

    def evolve(self,
               symbols: list[str],
               date_start: str,
               date_end: str,
               seed_strategies: list[str] | None = None,
               validation_start: str | None = None,
               resume: bool = False) -> dict:
        """Run the full GA evolution.

        Parameters
        ----------
        seed_strategies : list[str] | None
            Names of existing strategies to include in the initial population.
        validation_start : str | None
            If set, the final champion is also tested on this→date_end
            (out-of-sample).
        resume : bool
            If True, try to load a checkpoint and continue from there.
        """
        self._running = True
        cfg = self.config
        t_start = time.time()

        # Training period: if validation_start is set, stop training there
        train_end = validation_start if validation_start else date_end
        has_validation = validation_start is not None

        logger.info(f"GA: population={cfg.population_size}, "
                    f"generations={cfg.generations}, "
                    f"train={date_start}~{train_end}"
                    + (f", validate={validation_start}~{date_end}" if has_validation else ""))

        # ── Initialize or resume ──
        if resume and self.load_checkpoint():
            # Resume from checkpoint — skip initialization
            self._stagnation_count = 0
            logger.info(f"GA resuming from generation {self._generation}")
        else:
            self._population = self._init_population(seed_strategies)
            self._generation = 0
            self._best_fitness = -999
            self._best_chromosome = None
            self._stagnation_count = 0
            self._history = []

        # ── Evolution loop ──
        for gen in range(cfg.generations):
            if not self._running:
                break

            self._generation = gen + 1
            gen_start = time.time()

            # 1. Evaluate fitness
            self._population = evaluate_population(
                self._population, symbols, date_start, train_end,
                self.engine, self.loader,
                max_workers=cfg.max_workers,
                progress_callback=lambda c, t: self._report_progress(self._generation or 1, c, t))

            # 2. Sort by fitness
            self._population.sort(
                key=lambda c: c.get("fitness_result", {}).get("fitness", -999),
                reverse=True)

            best = self._population[0]
            best_fit = best.get("fitness_result", {}).get("fitness", -999)
            avg_fit = self._compute_avg_fitness()

            # 3. Record history
            gen_info = {
                "generation": self._generation,
                "best_fitness": best_fit,
                "avg_fitness": avg_fit,
                "best_sharpe": best.get("fitness_result", {}).get("sharpe", 0),
                "best_win_rate": best.get("fitness_result", {}).get("win_rate", 0),
                "best_trades": best.get("fitness_result", {}).get("trade_count", 0),
                "population_diversity": self._compute_diversity(),
                "elapsed": time.time() - gen_start,
            }
            self._history.append(gen_info)

            logger.info(
                f"Gen {self._generation:3d}/{cfg.generations} | "
                f"best={best_fit:.2f} avg={avg_fit:.2f} "
                f"sharpe={gen_info['best_sharpe']:.2f} "
                f"trades={gen_info['best_trades']} "
                f"div={gen_info['population_diversity']:.3f} "
                f"time={gen_info['elapsed']:.0f}s")

            if self._progress_callback:
                self._progress_callback((self._generation, cfg.generations, gen_info))

            # 4. Save checkpoint (for resume after stop/crash)
            self._save_checkpoint()

            # 5. Check improvement
            if best_fit > self._best_fitness + 0.01:
                self._best_fitness = best_fit
                self._best_chromosome = copy.deepcopy(best)
                self._stagnation_count = 0
            else:
                self._stagnation_count += 1

            # 5. Early stop
            if self._stagnation_count >= cfg.early_stop_generations:
                logger.info(f"GA early stop: no improvement for "
                           f"{cfg.early_stop_generations} generations")
                break

            # 6. Graceful stop check
            if self._stop_after_gen:
                logger.info(f"GA stopped gracefully after generation {self._generation}")
                break

            # 7. Create next generation
            if gen < cfg.generations - 1:
                self._population = self._next_generation()

        # ── Final champion ──
        self._running = False
        elapsed = time.time() - t_start

        if self._best_chromosome and not self._stop_after_gen:
            self.clear_checkpoint()  # clean completion — no resume needed

        if self._best_chromosome:
            champion_config = chromosome_to_strategy(self._best_chromosome)
            champion_config.name = f"ga_champion_{int(time.time())}"
            self.loader.save(champion_config)

            train_result = self._best_chromosome.get("fitness_result", {})

            # ── Out-of-sample validation ──
            validation = None
            dsr = None
            if has_validation:
                logger.info(f"GA: validating champion on {validation_start}~{date_end}")
                from core.ga.fitness import evaluate_chromosome
                val_result = evaluate_chromosome(
                    self._best_chromosome, symbols,
                    validation_start, date_end,
                    self.engine, self.loader)
                validation = {
                    "sharpe": val_result.get("sharpe", 0),
                    "win_rate": val_result.get("win_rate", 0),
                    "trade_count": val_result.get("trade_count", 0),
                    "total_return": val_result.get("total_return", 0),
                }
                # ── Statistical significance ──
                n_trials = cfg.population_size * self._generation
                from core.ga.fitness import deflated_sharpe_ratio
                dsr = deflated_sharpe_ratio(
                    val_result.get("sharpe", 0), n_trials)
                logger.info(
                    f"GA validation: sharpe={validation['sharpe']:.2f} "
                    f"DSR={dsr['dsr']:.2f} sig={dsr['significant']} "
                    f"(train sharpe={train_result.get('sharpe',0):.2f})")

            logger.info(
                f"GA complete: {self._generation} gens in {elapsed:.0f}s | "
                f"champion={champion_config.name} "
                f"fitness={train_result.get('fitness',0):.2f} "
                f"sharpe={train_result.get('sharpe',0):.2f}")

            return {
                "champion_name": champion_config.name,
                "champion_config": champion_config.model_dump(),
                "fitness": train_result.get("fitness", 0),
                "sharpe": train_result.get("sharpe", 0),
                "win_rate": train_result.get("win_rate", 0),
                "trade_count": train_result.get("trade_count", 0),
                "generations": self._generation,
                "elapsed_seconds": elapsed,
                "history": self._history,
                "validation": validation,
                "dsr": dsr,
            }
        else:
            return {"error": "No valid champion found"}

    def stop(self):
        """Graceful stop — finish current generation, save checkpoint."""
        self._stop_after_gen = True

    # ── Internal methods ──────────────────────────────────────────

    def _init_population(self, seed_strategies: list[str] | None) -> list[dict]:
        """Create initial population mixing random + seeded."""
        cfg = self.config
        population = []

        # Seeded individuals from existing strategies
        if seed_strategies:
            for name in seed_strategies[:cfg.elite_count]:
                try:
                    s_config = self.loader.load(name)
                    chrom = strategy_to_chromosome(s_config)
                    chrom["name"] = f"seed_{name}"
                    population.append(chrom)
                except Exception as e:
                    logger.warning(f"Failed to seed '{name}': {e}")

        # Fill remainder with random
        needed = cfg.population_size - len(population)
        for i in range(needed):
            population.append(random_chromosome(f"ga_rand_{i}"))

        return population

    def _next_generation(self) -> list[dict]:
        """Selection → Crossover → Mutation → next population."""
        cfg = self.config
        current = self._population
        current_fitnesses = [
            c.get("fitness_result", {}).get("fitness", -999) for c in current]

        new_pop = []

        # ── Elite preservation ──
        for i in range(min(cfg.elite_count, len(current))):
            new_pop.append(copy.deepcopy(current[i]))

        # ── Crossover + Mutation ──
        while len(new_pop) < cfg.population_size - cfg.immigrant_count:
            if random.random() < cfg.crossover_rate:
                p1 = self._tournament_select(current, current_fitnesses)
                p2 = self._tournament_select(current, current_fitnesses)
                child = self._crossover(p1, p2)
            else:
                parent = self._tournament_select(current, current_fitnesses)
                child = copy.deepcopy(parent)

            if random.random() < cfg.mutation_rate:
                child = self._mutate(child)

            child["fitness_result"] = {}  # clear stale result
            new_pop.append(child)

        # ── Diversity injection ──
        for i in range(cfg.immigrant_count):
            new_pop.append(random_chromosome(f"ga_immigrant_{i}"))

        # Trim to exact population size
        return new_pop[:cfg.population_size]

    def _tournament_select(self, population, fitnesses) -> dict:
        """Tournament selection: pick k random, return best."""
        cfg = self.config
        k = min(cfg.tournament_size, len(population))
        candidates = random.sample(range(len(population)), k)
        best_idx = max(candidates, key=lambda i: fitnesses[i])
        return population[best_idx]

    def _crossover(self, p1: dict, p2: dict) -> dict:
        """Two-point crossover on continuous genes, random exchange on structural."""
        child_cont = []
        for g1, g2 in zip(p1["continuous"], p2["continuous"]):
            if random.random() < 0.5:
                child_cont.append(copy.deepcopy(g1))
            else:
                child_cont.append(copy.deepcopy(g2))

        child_cat = []
        for g1, g2 in zip(p1["categorical"], p2["categorical"]):
            if random.random() < 0.5:
                child_cat.append(copy.deepcopy(g1))
            else:
                child_cat.append(copy.deepcopy(g2))

        child_struct = []
        for g1, g2 in zip(p1["structural"], p2["structural"]):
            # Random exchange of individual conditions
            all_conds = list(set(g1.conditions + g2.conditions))
            n = random.randint(1, len(all_conds))
            child_struct.append(StructuralGene(
                g1.name,
                conditions=random.sample(all_conds, min(n, len(all_conds))),
                template_pool=g1.template_pool,
            ))

        return {
            "continuous": child_cont,
            "categorical": child_cat,
            "structural": child_struct,
            "name": f"ga_child_{random.randint(1000,9999)}",
        }

    def _mutate(self, chrom: dict) -> dict:
        """Apply mutation to all gene types."""
        for gene in chrom["continuous"]:
            if random.random() < 0.2:
                gene.mutate()
        for gene in chrom["categorical"]:
            if random.random() < 0.1:
                gene.mutate()
        for gene in chrom["structural"]:
            if random.random() < 0.15:
                gene.mutate()
        chrom["fitness_result"] = {}
        return chrom

    def _compute_avg_fitness(self) -> float:
        fits = [c.get("fitness_result", {}).get("fitness", -999)
                for c in self._population]
        valid = [f for f in fits if f > -900]
        return sum(valid) / max(len(valid), 1)

    def _compute_diversity(self) -> float:
        """Measure population diversity as pairwise fitness spread."""
        if len(self._population) < 2:
            return 0.0
        fits = [c.get("fitness_result", {}).get("fitness", -999)
                for c in self._population if c.get("fitness_result", {}).get("fitness", -999) > -900]
        if len(fits) < 2:
            return 0.0
        import numpy as np
        return float(np.std(fits) / (abs(np.mean(fits)) + 1e-9))

    # ── Checkpoint / Resume ────────────────────────────────────────

    def _save_checkpoint(self):
        """Save current GA state to disk for resume."""
        try:
            import pickle
            state = {
                "population": self._population,
                "generation": self._generation,
                "best_fitness": self._best_fitness,
                "best_chromosome": self._best_chromosome,
                "history": self._history,
                "config": self.config,
            }
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._checkpoint_path, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.warning(f"GA checkpoint save failed: {e}")

    def load_checkpoint(self) -> bool:
        """Load saved GA state. Returns True if checkpoint was loaded."""
        try:
            import pickle
            if not self._checkpoint_path.exists():
                return False
            with open(self._checkpoint_path, "rb") as f:
                state = pickle.load(f)
            self._population = state["population"]
            self._generation = state["generation"]
            self._best_fitness = state["best_fitness"]
            self._best_chromosome = state["best_chromosome"]
            self._history = state["history"]
            logger.info(f"GA checkpoint loaded: gen={self._generation}, "
                       f"best_fitness={self._best_fitness:.2f}")
            return True
        except Exception as e:
            logger.warning(f"GA checkpoint load failed: {e}")
            return False

    def clear_checkpoint(self):
        """Remove checkpoint file after successful completion."""
        try:
            if self._checkpoint_path.exists():
                self._checkpoint_path.unlink()
        except Exception:
            pass

    def _report_progress(self, gen: int, completed: int, total: int):
        if self._progress_callback:
            self._progress_callback({
                "generation": gen,
                "eval_completed": completed,
                "eval_total": total,
                "phase": "evolving",
            })
