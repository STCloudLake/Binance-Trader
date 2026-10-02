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
    random_chromosome, confine_timeframe_gene, confine_regime_gene,
    ContinuousGene, CategoricalGene, StructuralGene,
)
from core.ga.benchmark import (
    BUY_HOLD, NONE, alpha_label, coerce_benchmark_mode, mode_description,
)
from core.strategy.loader import StrategyLoader


class CheckpointWindowMismatchError(RuntimeError):
    """A ``resume=True`` whose window differs from the checkpoint's window.

    Named — not a bare ``RuntimeError`` — so the refusal is machine-readable:
    ``scripts/ga_worker.py`` records ``error_type`` in the job's result file, so
    "refused: wrong window" can be told apart from a crash without parsing text.
    A resume must never silently evolve a population on a window it was never
    scored on.
    """


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
    max_workers: int = 4       # parallel backtest workers
    early_stop_generations: int = 10  # stop if no improvement for N gens
    seed: int = 0              # job seed (0 = derive from the clock, logged)
    window_key: str = ""       # walk-forward window identity for the checkpoint
    #: Per-job timeframe whitelist (``web/routes/ga.py`` job field
    #: ``timeframe_pool``; ``None`` = **no restriction**, the historical search
    #: space).  Confined into the timeframe gene on init, mutation and decode.
    timeframe_pool: list[str] | None = None
    #: The benchmark the publication gate consumes (``core.ga.benchmark``).
    #: ``None`` = **follow ``config.ga_benchmark_mode``** (whose code default is
    #: the legacy ``buy_hold``) — an absent job field must not override the
    #: operator's config.  Set by ``scripts/ga_worker.py`` from a validated job
    #: field, so a job can pin the mode per run.
    benchmark_mode: str | None = None
    #: Keep the checkpoint after a **clean** completion?  ``True`` (this default,
    #: and the shipped ``ga.keep_checkpoint: true``) means a finished run leaves
    #: ``data/ga_checkpoint.pkl`` behind so evolution can be continued later
    #: (e.g. another 20 generations on top of a 12-generation run); ``False``
    #: restores the pre-field behaviour (checkpoint deleted on completion).
    #: Every layer defaults to **retention**: an absent job field falls through
    #: to ``config.ga.keep_checkpoint`` and finally to this ``True``.  A run that
    #: was stopped (or crashed) always keeps its checkpoint — that is the
    #: pre-existing crash-resume path and is not affected by this flag.
    keep_checkpoint: bool = True
    #: P7-S1: evolve the causal regime filter gene (``ga.regime_conditioning``)?
    #: ``None`` = **follow ``config.ga_regime_conditioning``** (code default
    #: ``False``), so an absent job field can never turn conditioning on.  While
    #: this resolves to ``False`` the genome carries no ``regime_filter`` gene,
    #: no evaluation builds a regime context and the whole run is bit-identical
    #: to the pre-P7 build.  ``True`` makes the gene part of the search space and
    #: makes the backtest entry path evaluate each genome only on bars whose
    #: causal regime label is in its declaration.
    regime_conditioning: bool | None = None


def dsr_trial_counts(prior_trials: int, ledger_total: int, population: int,
                     trials_this_run: int) -> tuple[int, int]:
    """``(prior_for_this_generation, cumulative)`` trial counts for the DSR.

    The deflation must use the number of trials **actually performed**.  Two
    numbers are needed and they come from this one formula (audit D-18):

    * ``prior`` — everything tried *before* the generation being scored: the
      persisted ledger (earlier runs plus every earlier generation of this run,
      which ``record_trials`` updated at the end of each generation), floored by
      the deterministic ``prior_trials + trials_this_run`` so a failed ledger
      write can only understate, never overstate;
    * ``cumulative`` — ``prior + population``, i.e. the count once the
      generation being scored has performed its own trials.  The champion's DSR
      uses the **first** number instead: by then every generation has already
      been performed, so the ledger (== ``prior``) *is* the total — adding
      another population there would double-count the last generation.

    Before the fix the champion used ``population × generations + prior`` while
    every in-generation score used ``population + prior``, so the reported
    champion DSR and the gates disagreed about how many strategies had been
    tried.  The two are now one number by construction
    (``prior(last_generation) == champion_n_trials ==
    cumulative(last_generation)``), which ``tests/test_ga_dsr_trial_counts.py``
    pins.
    """
    deterministic = (max(int(prior_trials or 0), 0)
                     + max(int(trials_this_run or 0), 0))
    prior = max(int(ledger_total or 0), deterministic)
    return prior, prior + max(int(population or 0), 0)


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
        #: Wall-clock start of the run, so every progress payload can carry an
        #: honest ``elapsed_s`` (set by :meth:`evolve`).
        self._run_t_start = 0.0
        self._seed = int(getattr(self.config, "seed", 0) or 0)
        self._window_key = str(getattr(self.config, "window_key", "") or "")
        self._prior_trials = 0
        #: Trials performed by THIS run's generations so far (the ledger floor).
        self._trials_this_run = 0
        #: ``ga.keep_checkpoint`` / the job field of the same name.  ``True`` =
        #: a cleanly completed run keeps its checkpoint for a later resume.
        self._keep_checkpoint = bool(getattr(self.config, "keep_checkpoint", True))
        #: Generation this run resumed from (``None`` = a fresh run).
        self._resumed_from_generation: int | None = None
        #: What the checkpoint said (window/hash/trials/symbols), so the resumed
        #: run's provenance can prove what it continued from.
        self._checkpoint_meta: dict = {}
        #: The basket this run evolves on (recorded with the checkpoint).
        self._run_symbols: list = []
        self._checkpoint_path = Path(loader.strategies_dir).parent / "data" / "ga_checkpoint.pkl"
        #: P6-D: one lazily-built volume context per run (None while the
        #: executability model is off for the whole population).
        self._volume_context = None
        #: P7-S1: the effective regime-conditioning switch.  ``evolve`` resolves
        #: it from the job field (``GARunConfig.regime_conditioning``) and then
        #: from ``config.ga_regime_conditioning`` (code default ``False``) before
        #: the population exists, and mirrors it into
        #: ``core.ga.genome.REGIME_CONDITIONING_ENABLED`` — the genome module owns
        #: the flag because gene creation, decoding and random init must agree on
        #: it.  Pre-resolution the value is the code default, which is what a
        #: direct operator call (a test crossing two chromosomes, say) should see.
        self._regime_conditioning = bool(
            getattr(self.config, "regime_conditioning", False))

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
        self._run_t_start = t_start
        # The evaluated basket — stored with the checkpoint and reported in the
        # provenance, so a resumed run proves which symbols it continued on.
        self._run_symbols = list(symbols)

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

        # ── Timeframe whitelist: the effective pool is logged up front ──
        # Auditable from the job log alone; ``None`` (the default) means the
        # pre-existing search space and is logged as such.
        timeframe_pool = list(getattr(cfg, "timeframe_pool", None) or [])
        logger.info(
            "GA timeframe_pool=" + (
                ",".join(timeframe_pool) + " (timeframe gene confined to it)"
                if timeframe_pool else "unrestricted (all intervals)"))

        # ── Benchmark mode: the gate's comparison, logged up front ──
        # ``None`` on the run config = follow ``config.ga_benchmark_mode``
        # (code default ``buy_hold`` = the legacy fully-invested benchmark).
        _job_mode = getattr(cfg, "benchmark_mode", None)
        _benchmark_mode = coerce_benchmark_mode(
            _job_mode or getattr(getattr(self.engine, "config", None),
                                 "ga_benchmark_mode", None) or BUY_HOLD)
        logger.info(f"GA benchmark_mode={_benchmark_mode} "
                    f"({'job field' if _job_mode else 'config'})"
                    f" — the publication gate compares against it")

        # ── P7-S1: causal regime conditioning — resolved ONCE per run ──
        # The genome module owns the switch (``REGIME_CONDITIONING_ENABLED``)
        # because gene creation, decoding and random init must all agree on it;
        # the run sets it here from the job field (``GARunConfig``) and finally
        # from ``config.ga_regime_conditioning`` (code default ``False``).  A
        # ``False`` run therefore creates no regime gene, draws no extra RNG and
        # decodes every genome with an empty filter — bit-identical to pre-P7.
        from core.ga import genome as _genome_mod

        _job_regime = getattr(cfg, "regime_conditioning", None)
        self._regime_conditioning = bool(
            _job_regime if _job_regime is not None
            else getattr(getattr(self.engine, "config", None),
                         "ga_regime_conditioning", False))
        _genome_mod.REGIME_CONDITIONING_ENABLED = self._regime_conditioning
        logger.info(
            f"GA regime_conditioning={self._regime_conditioning} "
            f"({'job field' if _job_regime is not None else 'config'})"
            + (" — the regime_filter gene is evolved and the entry path "
               "evaluates each genome only on its allowed causal regimes"
               if self._regime_conditioning else
               " — no regime gene, every decoded genome has an empty filter "
               "(bit-identical to the pre-P7 search space)"))

        # Training period: if validation_start is set, stop training there
        train_end = validation_start if validation_start else date_end
        has_validation = validation_start is not None

        _data_dir = (str(Path(self.loader.strategies_dir).parent)
                     if hasattr(self.loader, 'strategies_dir') else "data")
        from core.ga.trial_counter import load_trials, record_trials, total_trials
        self._prior_trials = load_trials(_data_dir)
        self._trials_this_run = 0
        # `ga.alpha_weight`: weight of the DSR alpha term in the fitness (wired
        # here — P1 loaded the key from config and no code ever read it).  When
        # the key is absent or the shipped `1.0`, the term is multiplied by
        # exactly the same constant as before, so the default is bit-identical.
        _bt_cfg = getattr(self.engine, 'config', None)
        _alpha_weight = getattr(_bt_cfg, "ga_alpha_weight", None)

        logger.info(f"GA: population={cfg.population_size}, "
                    f"generations={cfg.generations}, "
                    f"train={date_start}~{train_end}"
                    + (f", validate={validation_start}~{date_end}" if has_validation else "")
                    + f", prior_trials={self._prior_trials}"
                    + f", keep_checkpoint={self._keep_checkpoint}")

        # ── Initialize or resume ──
        if resume and self.load_checkpoint(expected_window_key=self._window_key):
            # Resume from checkpoint — skip initialization
            self._stagnation_count = 0
            self._resumed_from_generation = int(self._generation)
            # ── Trial-count continuity (DSR honesty across a resume) ──
            # The checkpoint carries the interrupted run's trial accounting
            # (``prior_trials + trials_this_run`` at its last saved generation).
            # The resumed segment's DSR must be deflated by those trials, so the
            # carry is a FLOOR on ``_prior_trials`` (``max``): it can only raise
            # the count, never lower it — including when the ledger file
            # (``data/ga_trials.json``) was pruned between the two runs.  The
            # resumed segment's own trials start at 0 exactly like a fresh run's.
            _ckpt_trials = int(self._checkpoint_meta.get("trials_total") or 0)
            self._prior_trials = max(int(self._prior_trials or 0), _ckpt_trials)
            _ckpt_pop = int(self._checkpoint_meta.get("population_size") or 0)
            if _ckpt_pop and _ckpt_pop != int(cfg.population_size):
                logger.warning(
                    f"GA resume: the checkpoint holds {_ckpt_pop} genomes but "
                    f"this job asks for population_size={cfg.population_size} — "
                    f"the continued generation is resized to the job's size")
            logger.info(f"GA resuming from generation {self._generation} "
                        f"(window {getattr(self, '_window_key', '')}, "
                        f"checkpoint trials={_ckpt_trials}, "
                        f"prior_trials={self._prior_trials}, "
                        f"population_hash={self._checkpoint_meta.get('population_hash')})")
        else:
            self._population = self._init_population(seed_strategies)
            self._generation = 0
            self._best_fitness = -999
            self._best_chromosome = None
            self._stagnation_count = 0
            self._history = []
            self._resumed_from_generation = None

        # ── Confine the whole population to the job's timeframe pool ──
        # Covers a resumed checkpoint, a seeded strategy and any genome created
        # by an older (pre-pool) code path, before a single backtest is scored.
        if timeframe_pool:
            for chrom in self._population:
                confine_timeframe_gene(chrom, timeframe_pool)

        # ── P7-S1: confine (or strip) every genome's regime gene ──
        # Unconditional, and for the same reason: a checkpoint written by a
        # conditioned run, a seeded YAML carrying ``regime_filter`` or a
        # hand-built chromosome must all agree with THIS run's switch before a
        # single backtest is scored.  While the switch is off this strips the
        # gene, so the decoded genome is the pre-P7 one.
        for chrom in self._population:
            confine_regime_gene(chrom, self._regime_conditioning)

        # ── Evolution loop ──
        # The loop's START is the checkpoint's generation (0 for a fresh run):
        # a resumed run continues at g+1 and the upper bound stays the JOB's
        # ``generations``.  ``generations: 32`` resumed from a checkpoint at 12
        # therefore runs 13..32 and reports 32; it no longer re-numbers the
        # loaded population from generation 1.
        _first_gen = int(self._generation or 0)
        if _first_gen >= int(cfg.generations):
            logger.warning(
                f"GA resume: checkpoint is already at generation {_first_gen} "
                f">= generations={cfg.generations} — nothing left to evolve; "
                f"the checkpointed champion is published as-is")
        for gen in range(_first_gen, cfg.generations):
            if not self._running:
                break

            self._generation = gen + 1
            gen_start = time.time()

            # 1. Evaluate fitness — route to multi-process when max_workers > 1
            from core.ga.fitness_calibrate import FitnessCalibrator
            _calibrated_weights = FitnessCalibrator.load_weights_static(_data_dir)
            _bt_cfg = getattr(self.engine, 'config', None)
            _batch_trials = len(self._population)
            # The DSR's N is the number of trials ACTUALLY performed: this
            # generation is scored against the ledger (earlier generations of
            # this run + every earlier run) and the champion reuses the same
            # formula, so the reported DSR and the gate cannot disagree (D-18).
            _dsr_prior, _ = dsr_trial_counts(
                self._prior_trials, total_trials(_data_dir, 0),
                _batch_trials, self._trials_this_run)

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
                    prior_trials=_dsr_prior,
                    alpha_weight=_alpha_weight,
                    seed=self._seed,
                    benchmark_mode=_job_mode,
                    # P6-D: None unless the executability model is on for some genome.
                    volume_context=self.volume_context_for(
                        symbols, date_start, train_end),
                    progress_callback=lambda c, t: self._report_progress(self._generation or 1, c, t),
                    # Sub-chunk liveness: this branch reports a chunk only when
                    # the WHOLE chunk has been evaluated (one engine pass over
                    # every genome of the chunk).  For the shipped 30-genome /
                    # 9-month job that is hours, so the progress file stayed at
                    # ``{"phase": "starting"}`` for the entire first generation.
                    # ``progress_detail_callback`` carries the engine's existing
                    # bar-step ticks (no engine change, no extra computation).
                    progress_detail_callback=lambda d: self._report_progress(
                        self._generation or 1, d.get("eval_completed", 0),
                        d.get("eval_total", 0), detail=d))
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
                    prior_trials=_dsr_prior,
                    alpha_weight=_alpha_weight,
                    benchmark_mode=_job_mode,
                    # P6-D: None unless the executability model is on for some genome.
                    volume_context=self.volume_context_for(
                        symbols, date_start, train_end),
                    progress_callback=lambda c, t: self._report_progress(self._generation or 1, c, t))

            # Count this generation's trials for the DSR ledger.
            try:
                record_trials(_data_dir, _batch_trials, self._window_key)
            except Exception:  # pragma: no cover - never fail a run on bookkeeping
                pass
            self._trials_this_run += _batch_trials

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

        # ── Checkpoint retention (``keep_checkpoint``) ──
        # A CLEAN completion (a champion exists and the run was not asked to
        # stop) used to unconditionally delete the checkpoint, so a finished job
        # left nothing to resume from.  Retention is now the default; only an
        # explicit ``keep_checkpoint: false`` reproduces the old behaviour.
        # A stopped run always keeps it — that is the crash-resume path.
        checkpoint_note = ""
        if self._best_chromosome and not self._stop_after_gen:
            if self._keep_checkpoint:
                checkpoint_note = (f"checkpoint kept at {self._checkpoint_path} "
                                   f"(generation {self._generation})")
            else:
                self.clear_checkpoint()
                checkpoint_note = (f"checkpoint cleared at {self._checkpoint_path} "
                                   f"(generation {self._generation})")
            logger.info(f"GA {checkpoint_note}")
        elif self._checkpoint_path.exists():
            # No champion (nothing scored above the floor) or a graceful stop:
            # both keep the checkpoint so the run remains resumable — the old
            # code left it in place here too.
            why = ("run stopped, resumable" if self._stop_after_gen
                   else "no champion scored, resumable")
            checkpoint_note = (f"checkpoint kept at {self._checkpoint_path} "
                               f"(generation {self._generation}) — {why}")
            logger.info(f"GA {checkpoint_note}")
        checkpoint_kept = self._checkpoint_path.exists()

        if self._best_chromosome:
            champion_config = chromosome_to_strategy(
                self._best_chromosome, timeframe_pool=cfg.timeframe_pool or None)
            champion_config.name = f"ga_champion_{int(time.time())}"
            train_result = dict(self._best_chromosome.get("fitness_result", {}) or {})

            # ── Out-of-sample validation (BEFORE publishing) ──
            validation = None
            # Multiple-testing count for the DSR / gate.  `prior` is the count of
            # trials the run has ACTUALLY performed (the ledger, floored by
            # `prior_trials + trials_this_run`) — the same formula and therefore
            # the same number the last generation was scored against, never a
            # smaller one (audit D-18).
            n_trials, _ = dsr_trial_counts(
                self._prior_trials, total_trials(_data_dir, 0),
                cfg.population_size, self._trials_this_run)
            if has_validation:
                logger.info(f"GA: validating champion on {validation_start}~{date_end}")
                from core.ga.fitness import evaluate_chromosome
                val_result = evaluate_chromosome(
                    self._best_chromosome, symbols,
                    validation_start, date_end,
                    self.engine, self.loader,
                    ga_loader=self.ga_loader,
                    n_trials=n_trials,
                    alpha_weight=_alpha_weight,
                    benchmark_mode=_job_mode)
                validation = {
                    "sharpe": val_result.get("sharpe", 0),
                    "win_rate": val_result.get("win_rate", 0),
                    "trade_count": val_result.get("trade_count", 0),
                    "total_return": val_result.get("total_return", 0),
                    "profit_factor": val_result.get("profit_factor", 0),
                    "max_dd": val_result.get("max_dd", 0),
                    "dsr": val_result.get("dsr", 0),
                    "buy_hold_pct": val_result.get("buy_hold_pct"),
                    # The selected benchmark's own OOS numbers (reported only).
                    "benchmark_mode": val_result.get("benchmark_mode"),
                    "benchmark_pct": val_result.get("benchmark_pct"),
                    "alpha_vs_benchmark_pct": val_result.get("alpha_vs_benchmark_pct"),
                    "benchmark_sharpe": val_result.get("benchmark_sharpe"),
                    "benchmark_max_dd": val_result.get("benchmark_max_dd"),
                    "benchmark_time_in_market_pct": val_result.get(
                        "benchmark_time_in_market_pct"),
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
            # ── Evaluation/execution consistency ──────────────────────────
            # The champion was SCORED on the job's basket, while the live
            # watcher only honours a strategy's own ``symbols`` list
            # (``core/strategy/engine.py`` skips a symbol outside it in
            # ``_on_kline``/``evaluate_all_now``).  The YAML used to ship
            # ``symbols: []`` = "every symbol the watchlist holds", so an
            # enabled champion would have traded pairs it was never evaluated
            # on.  The evaluated basket is written here, and the engine's
            # existing restriction + startup log line
            # (``Strategy '<name>' restricted to symbols: [...]``) is what makes
            # evaluation and execution agree.  An empty basket (an old job file
            # with no symbols) keeps the historical "all symbols" behaviour.
            champion_config.symbols = list(symbols)
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
                # The job's timeframe whitelist: ``None`` = unrestricted.  Makes
                # a champion's timeframe constraint auditable afterwards.
                "timeframe_pool": list(cfg.timeframe_pool) if cfg.timeframe_pool else None,
                # The evolved entry structure, so a champion's AND/OR gene is
                # traceable without parsing the strategy body.
                "condition_logic": champion_config.condition_logic,
                "generations": self._generation,
                "population_size": cfg.population_size,
                "n_trials": n_trials,
                "prior_trials": int(getattr(self, "_prior_trials", 0)),
                # ── Checkpoint retention + resume identity ──────────────────
                # Whether a cleanly completed run kept ``data/ga_checkpoint.pkl``
                # (the job's ``keep_checkpoint`` / ``ga.keep_checkpoint``) and
                # what a resumed run continued FROM, so the continuation is
                # auditable from the champion YAML alone.
                "keep_checkpoint": bool(self._keep_checkpoint),
                "resumed_from_generation": self._resumed_from_generation,
                "checkpoint": {
                    "path": str(self._checkpoint_path),
                    "kept": bool(checkpoint_kept),
                    "keep_checkpoint": bool(self._keep_checkpoint),
                    "generation": int(self._generation),
                    "window_key": getattr(self, "_window_key", ""),
                    "resumed": self._resumed_from_generation is not None,
                    "resumed_from_generation": self._resumed_from_generation,
                    "resumed_population_hash": self._checkpoint_meta.get(
                        "population_hash"),
                    "resumed_trials": self._checkpoint_meta.get("trials_total"),
                    "resumed_symbols": list(self._checkpoint_meta.get("symbols") or []),
                    "resumed_timeframe_pool": list(
                        self._checkpoint_meta.get("timeframe_pool") or []),
                    "resumed_population_size": self._checkpoint_meta.get(
                        "population_size"),
                    "population_hash": self.population_hash(),
                    "population_size": cfg.population_size,
                    "symbols": list(symbols),
                    "timeframe_pool": (list(cfg.timeframe_pool)
                                       if cfg.timeframe_pool else None),
                    "trials_this_run": int(getattr(self, "_trials_this_run", 0) or 0),
                },
                # ── Trial accounting across a resume (DSR honesty) ──────────
                # ``prior_trials`` = everything performed BEFORE this invocation
                # (the ledger, floored by the checkpoint's carry on a resume);
                # ``trials_this_run`` = the generations THIS invocation actually
                # performed (on a resume: the 13..32 segment, not 1..32);
                # ``n_trials`` above is ``prior_trials + trials_this_run`` = the
                # cumulative count the deployment's DSR was deflated by.
                "trials": {
                    "prior_trials": int(getattr(self, "_prior_trials", 0)),
                    "trials_this_run": int(getattr(self, "_trials_this_run", 0) or 0),
                    "resumed_trials": self._checkpoint_meta.get("trials_total"),
                    "n_trials": int(n_trials),
                },
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
                    # The selected benchmark (additive; the keys above are the
                    # pre-``benchmark_mode`` contract and do not move).
                    "benchmark_mode": train_result.get("benchmark_mode"),
                    "benchmark_pct": train_result.get("benchmark_pct"),
                    "alpha_vs_benchmark_pct": train_result.get(
                        "alpha_vs_benchmark_pct"),
                },
                # ── Benchmark provenance (``ga.benchmark_mode``) ──────────
                # Only the mode's alpha GATES; everything else is reported so a
                # published champion can be risk-compared afterwards.  Before
                # this, the YAML kept the raw buy & hold return and nothing
                # else, so no exposure/risk comparison was possible at all.
                "benchmark": {
                    "mode": (train_result.get("benchmark_mode")
                             or _benchmark_mode),
                    "buy_hold_pct": train_result.get("buy_hold_pct"),
                    "benchmark_pct": train_result.get("benchmark_pct"),
                    "alpha_vs_benchmark_pct": train_result.get(
                        "alpha_vs_benchmark_pct"),
                    "alpha_vs_buy_hold_pct": train_result.get(
                        "alpha_vs_buy_hold_pct"),
                    # The benchmark's OWN risk numbers.
                    "benchmark_sharpe": train_result.get("benchmark_sharpe"),
                    "benchmark_max_dd_pct": train_result.get("benchmark_max_dd"),
                    "benchmark_time_in_market_pct": train_result.get(
                        "benchmark_time_in_market_pct"),
                    # Strategy side of the same comparison (reported, not gated).
                    "strategy_time_in_market_pct": train_result.get(
                        "strategy_time_in_market_pct"),
                    "information_ratio": train_result.get("information_ratio"),
                    "jensen_alpha_annual_pct": train_result.get(
                        "jensen_alpha_annual_pct"),
                    "benchmark_beta": train_result.get("benchmark_beta"),
                    "net_edge_per_trade": train_result.get("net_edge_per_trade"),
                    "net_edge_per_trade_pct": train_result.get(
                        "net_edge_per_trade_pct"),
                    "strategy_risk_matched_pct": train_result.get(
                        "strategy_risk_matched_pct"),
                    # The full report (weights, per-mode notes, availability).
                    "report": train_result.get("benchmark") or {},
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
                "timeframe_pool": provenance["timeframe_pool"],
                # Checkpoint retention + what this run continued from, at the
                # top level of the result as well as inside ``provenance``
                # (``scripts/ga_worker.py`` logs it and the status CLI reads it).
                "keep_checkpoint": provenance["keep_checkpoint"],
                "resumed_from_generation": provenance["resumed_from_generation"],
                "checkpoint": provenance["checkpoint"],
                "checkpoint_note": checkpoint_note,
                "trials": provenance["trials"],
                # The benchmark this run's gate consumed + the full report.
                "benchmark_mode": provenance["benchmark"]["mode"],
                "benchmark": provenance["benchmark"],
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
        validation > 0`` plus the **benchmark** criterion, which consumes the
        alpha of the mode selected by ``ga.benchmark_mode``
        (``core.ga.benchmark``).  ``buy_hold`` — the code default — reads
        ``alpha_vs_buy_hold_pct`` exactly as before, so the verdict and the
        reason string are unchanged; ``exposure_matched`` / ``risk_matched``
        consume their own alpha, and ``none`` skips only this criterion (the
        DSR/PSR and net-expectancy terms still gate).  The validation terms are
        only required when an out-of-sample window was actually provided (a
        plain GA run has none and is gated on its train metrics + DSR).
        """
        _cfg = getattr(self.engine, "config", None)
        min_trades = int(getattr(_cfg, "ga_min_champion_trades",
                                 self.MIN_CHAMPION_TRADES) or self.MIN_CHAMPION_TRADES)
        reasons: list = []
        trades = int(train_result.get("trade_count") or 0)
        pnl = float(train_result.get("total_return") or 0.0)
        pf = float(train_result.get("profit_factor") or 0.0)
        dsr = float(train_result.get("dsr") or 0.0)

        # ── Benchmark criterion: the SELECTED mode's alpha ──
        # ``benchmark_mode`` travels with the scored result, so the gate consumes
        # exactly the benchmark the evaluation used.  A result dict without the
        # field (an older caller) is the legacy ``buy_hold`` case — the reason
        # string is then byte-identical to the pre-``benchmark_mode`` one.
        bench_mode = coerce_benchmark_mode(train_result.get("benchmark_mode"))
        alpha = train_result.get("alpha_vs_benchmark_pct")
        if alpha is None:
            alpha = train_result.get("alpha_vs_buy_hold_pct")
        alpha = float(alpha or 0.0)
        bench_available = train_result.get("benchmark_pct")
        if bench_available is None and bench_mode == BUY_HOLD:
            bench_available = train_result.get("buy_hold_pct")

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
        if bench_mode != NONE and bench_available is not None and alpha <= 0:
            reasons.append(
                f"{alpha_label(bench_mode)}={alpha:.2f}% <= 0 "
                f"(no edge over {mode_description(bench_mode)})")

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

    def volume_context_for(self, symbols: list[str], date_start: str,
                           date_end: str):
        """P6-D volume context for the executability model — built once, lazily.

        Returns ``None`` (and reads nothing) while the model is off for the whole
        population: ``risk.liquidity.impact_k <= 0`` **and** no genome carries a
        non-neutral ``volume_scale_k``.  That is the shipped configuration, so a
        default run does no extra I/O and the evaluations are bit-identical to
        pre-P6.  The context is picklable, so the multiprocess path shares one
        build instead of re-reading the parquet cache per chunk.
        """
        from core.ga.fitness import (build_volume_context,
                                     chromosome_volume_scale_k,
                                     executability_params)
        cfg = getattr(self.engine, "config", None)
        params = executability_params(cfg)
        needed = params["impact_k"] > 0.0 or any(
            chromosome_volume_scale_k(chrom) > 0.0 for chrom in self._population)
        if not needed:
            self._volume_context = None
            return None
        if self._volume_context is None:
            intervals = sorted({
                tf for chrom in self._population
                for gene in chrom.get("categorical", []) or []
                if getattr(gene, "name", "") == "timeframes"
                for tf in str(getattr(gene, "value", "")).split(",") if tf})
            self._volume_context = build_volume_context(
                cfg, symbols, intervals or ["1h"], date_start, date_end)
            logger.info(
                f"P6-D executability model active: impact_k={params['impact_k']} "
                f"volume context={'built' if self._volume_context else 'unavailable'}")
        return self._volume_context

    def _init_population(self, seed_strategies: list[str] | None) -> list[dict]:
        """Create initial population mixing random + seeded."""
        cfg = self.config
        population = []

        # Seeded individuals from existing strategies
        if seed_strategies:
            for name in seed_strategies[:cfg.elite_count]:
                try:
                    s_config = self.loader.load(name)
                    chrom = strategy_to_chromosome(
                        s_config, timeframe_pool=getattr(cfg, "timeframe_pool", None))
                    chrom["name"] = f"seed_{name}"
                    population.append(chrom)
                except Exception as e:
                    logger.warning(f"Failed to seed '{name}': {e}")

        # Fill remainder with random
        needed = cfg.population_size - len(population)
        for i in range(needed):
            population.append(random_chromosome(
                f"ga_rand_{i}",
                timeframe_pool=getattr(cfg, "timeframe_pool", None),
                # P7-S1: passed explicitly (not left to the module flag), so a
                # direct ``_init_population`` call — a test, a resumed run — gets
                # the gene this evolver's switch implies.
                regime_conditioning=getattr(self, "_regime_conditioning", None)))

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
            new_pop.append(random_chromosome(
                f"ga_immigrant_{i}",
                timeframe_pool=getattr(cfg, "timeframe_pool", None),
                regime_conditioning=getattr(self, "_regime_conditioning", None)))

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
        # The timeframe gene is re-confined on every mutation, so a gene
        # inherited from a pre-pool checkpoint can never leave the job's pool.
        confine_timeframe_gene(chrom, getattr(self.config, "timeframe_pool", None))
        # P7-S1: the regime gene is re-confined on every mutation too, and REMOVED
        # while the run's switch is off — so a checkpoint written by a conditioned
        # run cannot leak a filter into an unconditioned one.
        confine_regime_gene(chrom, getattr(self, "_regime_conditioning", None))
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

    def population_hash(self) -> str:
        """``sha256`` (first 16 hex) of the population's gene content.

        Order-sensitive (the population IS fitness-sorted after every
        generation), so a resumed run's provenance can prove which population it
        continued from.  Genes are reduced to ``(name, value)`` / structural
        genes to their conditions — a hash of the pickle itself would move with
        interpreter details for identical content.
        """
        import hashlib

        def _values(genes):
            return tuple(sorted(
                (str(getattr(g, "name", "")), repr(getattr(g, "value", None)))
                for g in genes or []))

        items = []
        for chrom in self._population:
            items.append((
                str(chrom.get("name", "")),
                str(chrom.get("condition_logic", "")),
                _values(chrom.get("continuous")),
                _values(chrom.get("categorical")),
                _values(chrom.get("indicator_genes")),
                tuple(sorted(
                    (str(getattr(g, "name", "")),
                     tuple(getattr(g, "conditions", None) or []))
                    for g in chrom.get("structural", []) or [])),
            ))
        return hashlib.sha256(repr(items).encode("utf-8")).hexdigest()[:16]

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
                # ── Trial accounting (DSR honesty across a resume) ──
                # Everything the resumed segment must be deflated by.  Before
                # this the checkpoint carried NO trial count, so a resume whose
                # ledger was pruned restarted the DSR's N at the job's
                # population instead of the trials actually performed.
                "prior_trials": int(getattr(self, "_prior_trials", 0) or 0),
                "trials_this_run": int(getattr(self, "_trials_this_run", 0) or 0),
                # ── Identity: what a resumed run continues from ──
                "population_hash": self.population_hash(),
                "symbols": list(getattr(self, "_run_symbols", []) or []),
                "timeframe_pool": (list(self.config.timeframe_pool)
                                   if getattr(self.config, "timeframe_pool", None)
                                   else []),
                "resumed_from_generation": self._resumed_from_generation,
                "keep_checkpoint": bool(self._keep_checkpoint),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._checkpoint_path, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.warning(f"GA checkpoint save failed: {e}")

    def load_checkpoint(self, expected_window_key: str | None = None) -> bool:
        """Load saved GA state. Returns True if checkpoint was loaded.

        Parameters
        ----------
        expected_window_key : str | None
            The window this run is about to evolve on.  When given and the
            checkpoint records a **different, non-empty** window the resume is
            REFUSED with :class:`CheckpointWindowMismatchError` — continuing
            would silently evolve a population on a window it was never scored
            on, publishing a champion whose provenance claims the old window.
            The check runs outside the loader's ``except Exception`` so a refusal
            can never be swallowed into a quiet "no checkpoint".
        """
        import pickle
        try:
            if not self._checkpoint_path.exists():
                return False
            with open(self._checkpoint_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            logger.warning(f"GA checkpoint load failed: {e}")
            return False

        ckpt_window = str(state.get("window_key", "") or "")
        requested = str(expected_window_key or "")
        if requested and ckpt_window and ckpt_window != requested:
            raise CheckpointWindowMismatchError(
                f"checkpoint {self._checkpoint_path} belongs to window "
                f"'{ckpt_window}' but this run requested '{requested}' — "
                f"refusing to resume on a different window (start a fresh run "
                f"without resume=True, or use the checkpoint's window)")

        try:
            self._population = state["population"]
            self._generation = state["generation"]
            self._best_fitness = state["best_fitness"]
            self._best_chromosome = state["best_chromosome"]
            self._history = state["history"]
            if ckpt_window:
                # Only a NON-EMPTY checkpoint window may overwrite the run's
                # window; an empty one (pre-``window_key`` checkpoint) leaves the
                # run's own key intact.
                self._window_key = ckpt_window
            self._checkpoint_meta = {
                "generation": int(state.get("generation", 0) or 0),
                "window_key": ckpt_window,
                "population_hash": state.get("population_hash"),
                "prior_trials": int(state.get("prior_trials", 0) or 0),
                "trials_this_run": int(state.get("trials_this_run", 0) or 0),
                # The number the resumed segment's DSR must be floored by.
                "trials_total": (int(state.get("prior_trials", 0) or 0)
                                 + int(state.get("trials_this_run", 0) or 0)),
                "symbols": list(state.get("symbols", []) or []),
                "timeframe_pool": list(state.get("timeframe_pool", []) or []),
                "population_size": len(state.get("population") or []),
                "saved_at": state.get("saved_at"),
            }
            logger.info(f"GA checkpoint loaded: gen={self._generation}, "
                       f"best_fitness={self._best_fitness:.2f}, "
                       f"window={self._window_key}, "
                       f"trials={self._checkpoint_meta['trials_total']}, "
                       f"population_hash={self._checkpoint_meta['population_hash']}")
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

    def _report_progress(self, gen: int, completed: int, total: int,
                         detail: dict | None = None):
        """Emit one progress payload (dict) to the registered callback.

        The four keys the UI has always read (``generation``, ``eval_completed``,
        ``eval_total``, ``phase``) are unchanged; everything else is additive so
        an operator can judge a long generation without the UI:

        ``total_generations`` / ``elapsed_s`` / ``best_fitness`` / ``best_trades``
        (best-so-far — ``None`` until the first generation has been scored) plus,
        when the multiprocess branch supplies them, ``eval_equivalent`` (work done
        in strategy-equivalents, a float) and the in-flight chunk's
        ``bar_step``/``bar_total``/``chunk_progress_pct``.
        """
        if not self._progress_callback:
            return
        generations = int(getattr(self.config, "generations", 0) or 0)
        payload = {
            "generation": int(gen or 0),
            "total_generations": generations,
            "eval_completed": int(completed or 0),
            "eval_total": int(total or 0),
            "phase": "evolving",
        }
        if self._run_t_start:
            payload["elapsed_s"] = round(time.time() - self._run_t_start, 1)
        if self._best_chromosome is not None:
            try:
                payload["best_fitness"] = round(
                    float(self._best_chromosome.get("fitness_result", {})
                          .get("fitness", 0) or 0), 6)
                payload["best_trades"] = int(
                    self._best_chromosome.get("fitness_result", {})
                    .get("trade_count", 0) or 0)
            except Exception:  # pragma: no cover - progress must never raise
                pass
        for key, value in (detail or {}).items():
            if key in ("eval_completed", "eval_total") or value is None:
                continue
            payload[key] = value
        self._progress_callback(payload)
