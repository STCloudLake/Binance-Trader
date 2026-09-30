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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from loguru import logger

from core.ga.genome import (
    strategy_to_chromosome, chromosome_to_strategy,
    random_chromosome, ContinuousGene, CategoricalGene, StructuralGene,
)
from core.strategy.loader import StrategyLoader


def _inherit_genes_by_name(genes_a: list, genes_b: list) -> list:
    """One child gene per NAME present in either parent, randomly inherited.

    A positional ``zip`` loses genes as soon as the parents' gene lists differ
    in length or order (a disabled indicator removes its continuous genes), and
    duplicated a gene when one parent's list was shorter.  Name-keyed
    inheritance makes ``len`` and order deterministic: the child carries the
    union of the parents' gene names, in sorted order.
    """
    a = {g.name: g for g in genes_a}
    b = {g.name: g for g in genes_b}
    child = []
    for name in sorted(a.keys() | b.keys()):
        if name in a and name in b:
            chosen = a[name] if random.random() < 0.5 else b[name]
        else:
            chosen = a.get(name) or b.get(name)
        child.append(copy.deepcopy(chosen))
    return child


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
    seed: int = 0              # job seed (0 = derive from the clock, logged)
    window_key: str = ""       # walk-forward window identity for the checkpoint


class GAStrategyEvolver:
    """Genetic algorithm for evolving trading strategies."""

    def __init__(self, engine, loader: StrategyLoader,
                 config: GARunConfig | None = None):
        self.engine = engine
        self.loader = loader
        self.config = config or GARunConfig()

        # Isolated loader for GA temp strategies — never touches strategies/
        _ga_dir = Path(loader.strategies_dir).parent / "data" / "ga_strategies"
        _ga_dir.mkdir(parents=True, exist_ok=True)
        self.ga_loader = StrategyLoader(str(_ga_dir))

        self._population: list[dict] = []
        self._generation = 0
        self._best_fitness = -999
        self._best_chromosome: dict | None = None
        self._stagnation_count = 0
        self._history: list[dict] = []
        self._running = False
        self._stop_after_gen = False
        self._progress_callback = None
        self._seed = int(getattr(self.config, "seed", 0) or 0)
        self._window_key = str(getattr(self.config, "window_key", "") or "")
        self._prior_trials = 0
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
               resume: bool = False,
               seed: int | None = None,
               window_key: str = "") -> dict:
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
        seed : int | None
            Job seed.  When given (or when ``config.seed`` is set) ``random`` and
            ``numpy`` are seeded before the population is created, so the same
            job file yields the same champions.  ``None``/0 → a clock-derived
            seed which is still reported in the provenance block.
        window_key : str
            Walk-forward window identity, stored with the checkpoint so a
            resumed run can prove which window it belongs to.
        """
        self._running = True
        cfg = self.config
        t_start = time.time()

        # ── Determinism: seed from the job before anything random happens ──
        import numpy as np
        effective_seed = int(seed or getattr(cfg, "seed", 0) or 0)
        if not effective_seed:
            effective_seed = int(time.time()) % (2 ** 31)
        random.seed(effective_seed)
        np.random.seed(effective_seed % (2 ** 32))
        self._seed = effective_seed
        self._window_key = window_key or f"{date_start}~{date_end}"
        logger.info(f"GA seed={effective_seed} window={self._window_key}")

        # Training period: if validation_start is set, stop training there
        train_end = validation_start if validation_start else date_end
        has_validation = validation_start is not None

        _data_dir = (str(Path(self.loader.strategies_dir).parent)
                     if hasattr(self.loader, 'strategies_dir') else "data")
        from core.ga.trial_counter import load_trials, record_trials, total_trials
        self._prior_trials = load_trials(_data_dir)

        logger.info(f"GA: population={cfg.population_size}, "
                    f"generations={cfg.generations}, "
                    f"train={date_start}~{train_end}"
                    + (f", validate={validation_start}~{date_end}" if has_validation else "")
                    + f", prior_trials={self._prior_trials}")

        # ── Initialize or resume ──
        if resume and self.load_checkpoint():
            # Resume from checkpoint — skip initialization
            self._stagnation_count = 0
            logger.info(f"GA resuming from generation {self._generation} "
                        f"(window {getattr(self, '_window_key', '')})")
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

            # 1. Evaluate fitness — route to multi-process when max_workers > 1
            from core.ga.fitness_calibrate import FitnessCalibrator
            _calibrated_weights = FitnessCalibrator.load_weights_static(_data_dir)
            _bt_cfg = getattr(self.engine, 'config', None)
            _batch_trials = len(self._population)

            if getattr(cfg, 'max_workers', 1) > 1:
                # Multi-process: each worker creates its own engine (avoids TA-Lib thread crash)
                from core.ga.fitness import evaluate_population_multiprocess
                self._population = evaluate_population_multiprocess(
                    self._population, symbols, date_start, train_end,
                    initial_balance=10000.0,
                    max_workers=cfg.max_workers,
                    weights=_calibrated_weights,
                    cost_enabled=getattr(_bt_cfg, 'backtest_cost_enabled', True) if _bt_cfg else True,
                    taker_fee_pct=getattr(_bt_cfg, 'backtest_taker_fee_pct', 0.04) if _bt_cfg else 0.04,
                    spread_pct=getattr(_bt_cfg, 'backtest_spread_pct', {}) if _bt_cfg else {},
                    engine_mode=getattr(_bt_cfg, 'backtest_engine_mode', 'legacy') if _bt_cfg else 'legacy',
                    use_live_spread=False,
                    batch_trials=_batch_trials,
                    prior_trials=self._prior_trials,
                    seed=self._seed,
                    progress_callback=lambda c, t: self._report_progress(self._generation or 1, c, t))
            else:
                # Single-process: use existing threaded batch evaluation
                from core.ga.fitness import evaluate_population_batch
                self._population = evaluate_population_batch(
                    self._population, symbols, date_start, train_end,
                    self.engine, self.loader,
                    ga_loader=self.ga_loader,
                    batch_size=cfg.population_size,
                    max_workers=1,  # force serial: avoids TA-Lib thread-safety crashes
                    weights=_calibrated_weights,
                    use_live_spread=False,
                    batch_trials=_batch_trials,
                    prior_trials=self._prior_trials,
                    progress_callback=lambda c, t: self._report_progress(self._generation or 1, c, t))

            # Count this generation's trials for the DSR ledger.
            try:
                record_trials(_data_dir, _batch_trials, self._window_key)
            except Exception:  # pragma: no cover - never fail a run on bookkeeping
                pass

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
            train_result = dict(self._best_chromosome.get("fitness_result", {}) or {})

            # ── Out-of-sample validation (BEFORE publishing) ──
            validation = None
            # Multiple-testing count for the DSR / gate: this run's
            # population × generations PLUS every trial earlier runs recorded in
            # the ledger (`data/ga_trials.json`), so the 24th walk-forward
            # champion is not treated as if only 450 strategies had been tried.
            n_trials = (cfg.population_size * max(self._generation, 1)
                        + int(getattr(self, "_prior_trials", 0)))
            if has_validation:
                logger.info(f"GA: validating champion on {validation_start}~{date_end}")
                from core.ga.fitness import evaluate_chromosome
                val_result = evaluate_chromosome(
                    self._best_chromosome, symbols,
                    validation_start, date_end,
                    self.engine, self.loader,
                    ga_loader=self.ga_loader,
                    n_trials=n_trials)
                validation = {
                    "sharpe": val_result.get("sharpe", 0),
                    "win_rate": val_result.get("win_rate", 0),
                    "trade_count": val_result.get("trade_count", 0),
                    "total_return": val_result.get("total_return", 0),
                    "profit_factor": val_result.get("profit_factor", 0),
                    "max_dd": val_result.get("max_dd", 0),
                    "dsr": val_result.get("dsr", 0),
                    "buy_hold_pct": val_result.get("buy_hold_pct"),
                    "start": validation_start, "end": date_end,
                }
                logger.info(
                    f"GA validation: sharpe={validation['sharpe']:.2f} "
                    f"DSR={validation['dsr']:.2f} trades={validation['trade_count']} "
                    f"(train sharpe={train_result.get('sharpe', 0):.2f})")

            # ── Publication gate ──────────────────────────────────────────
            # A non-empty population used to be enough to write an ENABLED
            # champion (a published artefact had fitness -42.3 with 0 trades).
            # Metrics now decide: failing genomes are still written, but with
            # ``enabled: false`` and a machine-readable rejection list.
            published, rejection_reasons = self._publication_decision(
                train_result, validation)
            champion_config.enabled = published
            self.loader.save(champion_config)

            # ── Provenance + gate metadata on the YAML ──
            provenance = {
                "seed": int(getattr(self, "_seed", 0)),
                "window": {
                    "train_start": date_start, "train_end": train_end,
                    "validation_start": validation_start, "validation_end": date_end,
                    "key": getattr(self, "_window_key", ""),
                },
                "symbols": list(symbols),
                "timeframes": list(champion_config.timeframes),
                # The evolved entry structure, so a champion's AND/OR gene is
                # traceable without parsing the strategy body.
                "condition_logic": champion_config.condition_logic,
                "generations": self._generation,
                "population_size": cfg.population_size,
                "n_trials": (cfg.population_size * max(self._generation, 1)
                             + int(getattr(self, "_prior_trials", 0))),
                "prior_trials": int(getattr(self, "_prior_trials", 0)),
                "fitness_components": {
                    "fitness": train_result.get("fitness"),
                    "fitness_base": train_result.get("fitness_base"),
                    "fitness_alpha": train_result.get("fitness_alpha"),
                    "sharpe": train_result.get("sharpe"),
                    "deflated_sharpe": train_result.get("dsr"),
                    "max_dd": train_result.get("max_dd"),
                    "trade_count": train_result.get("trade_count"),
                    "profit_factor": train_result.get("profit_factor"),
                    "raw_profit_factor": train_result.get("raw_profit_factor"),
                    "buy_hold_pct": train_result.get("buy_hold_pct"),
                    "alpha_vs_buy_hold_pct": train_result.get("alpha_vs_buy_hold_pct"),
                },
                "validation": validation,
                "published": published,
                "rejection_reasons": rejection_reasons,
                "eval": {"engine_mode": "legacy", "use_live_spread": False},
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._append_provenance(champion_config.name, provenance)

            logger.info(
                f"GA complete: {self._generation} gens in {elapsed:.0f}s | "
                f"champion={champion_config.name} published={published} "
                f"fitness={train_result.get('fitness',0):.2f} "
                f"sharpe={train_result.get('sharpe',0):.2f}"
                + (f" rejected={rejection_reasons}" if rejection_reasons else ""))

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
                "dsr": train_result.get("dsr_detail") or (
                    {"dsr": train_result.get("dsr", 0),
                     "n_trials": provenance["n_trials"]}),
                "provenance": provenance,
                "published": published,
                "rejection_reasons": rejection_reasons,
                "enabled": published,
                "seed": provenance["seed"],
            }
        else:
            return {"error": "No valid champion found"}

    # ── Publication gate ──────────────────────────────────────────────

    #: Minimum completed trades a champion needs before it may be enabled
    #: (overridden by ``ga.min_champion_trades`` in config.yaml when present).
    MIN_CHAMPION_TRADES = 30

    def _publication_decision(self, train_result: dict,
                              validation: dict | None) -> tuple[bool, list]:
        """``(published, rejection_reasons)`` for the champion.

        Gate: ``trades >= 30 AND net_pnl > 0 AND pf > 1 AND dsr > 0 AND
        validation > 0``.  The validation terms are only required when an
        out-of-sample window was actually provided (a plain GA run has none and
        is gated on its train metrics + DSR).
        """
        _cfg = getattr(self.engine, "config", None)
        min_trades = int(getattr(_cfg, "ga_min_champion_trades",
                                 self.MIN_CHAMPION_TRADES) or self.MIN_CHAMPION_TRADES)
        reasons: list = []
        trades = int(train_result.get("trade_count") or 0)
        pnl = float(train_result.get("total_return") or 0.0)
        pf = float(train_result.get("profit_factor") or 0.0)
        dsr = float(train_result.get("dsr") or 0.0)
        alpha = float(train_result.get("alpha_vs_buy_hold_pct") or 0.0)

        if trades < min_trades:
            reasons.append(
                f"trades={trades} < {min_trades} (not enough evidence)")
        if trades == 0:
            reasons.append("no_trades")
        if pnl <= 0:
            reasons.append(f"net_pnl={pnl:.2f} <= 0")
        if pf <= 1.0:
            reasons.append(f"profit_factor={pf:.3f} <= 1 (after shrink cap)")
        if dsr <= 0:
            reasons.append(f"dsr={dsr:.4f} <= 0 (indistinguishable from data mining)")
        if train_result.get("buy_hold_pct") is not None and alpha <= 0:
            reasons.append(
                f"alpha_vs_buy_hold={alpha:.2f}% <= 0 (no edge over buy & hold)")

        if validation is not None:
            v_sharpe = float(validation.get("sharpe") or 0.0)
            v_trades = int(validation.get("trade_count") or 0)
            if v_trades <= 0:
                reasons.append("validation_trades=0")
            if v_sharpe <= 0:
                reasons.append(f"validation_sharpe={v_sharpe:.2f} <= 0")
            v_dsr = float(validation.get("dsr") or 0.0)
            if v_dsr <= 0:
                reasons.append(f"validation_dsr={v_dsr:.4f} <= 0")

        return (len(reasons) == 0), reasons

    def _append_provenance(self, strategy_name: str, provenance: dict) -> None:
        """Append a ``provenance:`` block to the saved champion YAML.

        ``StrategyConfig`` is the runtime schema and deliberately does not carry
        run metadata, so the block is added to the YAML document instead of the
        model (a strategy that predates this has no block and loads unchanged).
        """
        try:
            import yaml
            path = self.loader.strategies_dir / f"{self.loader._normalize(strategy_name)}.yaml"
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            data["provenance"] = provenance
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"GA provenance written to {path.name}")
        except Exception as e:
            logger.warning(f"GA provenance write failed: {e}")

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
        """Crossover keyed by GENE NAME (never by ``zip`` position).

        Positional ``zip`` assumed both parents held the same genes in the same
        order.  Two parents whose continuous lists differ (a disabled indicator
        drops its genes, mutation can insert others) silently produced a child
        that kept 6 of 8 genes with one duplicated — the union of names is the
        only safe basis.  Structural (condition) genes are matched by name too,
        so ``entry_long`` conditions are never mixed into ``exit_short``.
        """
        child_cont = _inherit_genes_by_name(
            p1.get("continuous", []), p2.get("continuous", []))
        child_cat = _inherit_genes_by_name(
            p1.get("categorical", []), p2.get("categorical", []))

        # Indicator boolean genes: random inheritance from either parent
        child_ind = []
        p1_ind = {g.name: g for g in p1.get("indicator_genes", [])}
        p2_ind = {g.name: g for g in p2.get("indicator_genes", [])}
        for name in sorted(p1_ind.keys() | p2_ind.keys()):
            if name in p1_ind and name in p2_ind:
                chosen = p1_ind[name] if random.random() < 0.5 else p2_ind[name]
            elif name in p1_ind:
                chosen = p1_ind[name]
            else:
                chosen = p2_ind[name]
            child_ind.append(copy.deepcopy(chosen))

        # Structural genes: one child gene per NAME, conditions drawn from the
        # union of that gene's conditions in both parents.
        p1_struct = {g.name: g for g in p1.get("structural", [])}
        p2_struct = {g.name: g for g in p2.get("structural", [])}
        child_struct = []
        for name in sorted(p1_struct.keys() | p2_struct.keys()):
            g1 = p1_struct.get(name)
            g2 = p2_struct.get(name)
            if g1 is None:
                child_struct.append(copy.deepcopy(g2))
                continue
            if g2 is None:
                child_struct.append(copy.deepcopy(g1))
                continue
            all_conds = list(dict.fromkeys(list(g1.conditions) + list(g2.conditions)))
            if not all_conds:
                # A gene with no conditions cannot be sampled — inherit whole.
                child_struct.append(copy.deepcopy(g1))
                continue
            n = random.randint(1, len(all_conds))
            child_struct.append(StructuralGene(
                name,
                conditions=random.sample(all_conds, min(n, len(all_conds))),
                template_pool=list(g1.template_pool or g2.template_pool or []),
            ))

        child = {
            "continuous": child_cont,
            "categorical": child_cat,
            "structural": child_struct,
            "indicator_genes": child_ind,
            "name": f"ga_child_{random.randint(1000,9999)}",
        }
        # The genome's condition-logic gene is inherited like any other.
        if "condition_logic" in p1 or "condition_logic" in p2:
            source = p1 if random.random() < 0.5 else p2
            child["condition_logic"] = source.get("condition_logic", "or")
        return child

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
        for gene in chrom.get("indicator_genes", []):
            gene.mutate()
        # Evolvable condition logic (OR = looser, AND = stricter)
        if random.random() < 0.1:
            chrom["condition_logic"] = random.choice(["or", "and"])
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
                "window_key": getattr(self, "_window_key", ""),
                "seed": getattr(self, "_seed", 0),
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
            self._window_key = state.get("window_key", getattr(self, "_window_key", ""))
            logger.info(f"GA checkpoint loaded: gen={self._generation}, "
                       f"best_fitness={self._best_fitness:.2f}, "
                       f"window={self._window_key}")
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
