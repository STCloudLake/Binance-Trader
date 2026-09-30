"""Real-cached-data, bounded multi-generation GA fitness curve (P1 evidence).

The P1 behaviour ("best fitness now improves across generations because every
genome trades") was asserted on *synthetic* parquet because a real run with
1m-timeframe genomes exceeded 10 minutes under load.  This tool produces the
same evidence on the **real cached market data** inside a hard wall-clock cap.

Design
------
* The GA loop itself is the production ``GAStrategyEvolver.evolve`` — real
  engine (``BacktestEngine``), real genome operators, real DSR/publication gate.
* The timeframe gene is pinned to one timeframe (default ``1h``) by wrapping
  ``core.ga.evolver.random_chromosome``, ``GAStrategyEvolver._mutate`` and
  ``_init_population``.  A genome carrying ``1m`` needs ~44k bars and dominates
  the wall clock without adding anything to this measurement.  No source file
  is modified.
* The run happens in a child process.  The parent kills it at ``--max-seconds``
  (hard cap, default/max 480 s), so a stuck generation can never hang a
  CI-like run.  The child appends one JSON record per completed generation to
  ``--out``, so a killed run still leaves usable evidence.
* ``--resume`` reuses the work directory, so the evolver's own
  ``data/ga_checkpoint.pkl`` continues the previous run instead of restarting.

Everything is written under the system temp directory (never the live DB,
never ``strategies/``, never ``data/market``); the repo is only read.

    python tools/ga_real_data_curve.py
    python tools/ga_real_data_curve.py --generations 4 --population 8
    python tools/ga_real_data_curve.py --resume
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HARD_CAP_SECONDS = 480
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ga_real_data_curve.json"
DEFAULT_WORKDIR = Path(tempfile.gettempdir()) / "ga_real_data_curve_work"


# ── helpers ──────────────────────────────────────────────────────────────

def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _prepare_workdir(workdir: Path, resume: bool) -> Path:
    """A throwaway data root: real parquet by symlink, temp DB + strategies."""
    workdir = workdir.resolve()
    if not resume and workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    (workdir / "strategies").mkdir(parents=True, exist_ok=True)

    target = workdir / "data" / "market"
    if not target.exists():
        source = PROJECT_ROOT / "data" / "market"
        try:
            target.symlink_to(source, target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError):
            shutil.copytree(source, target)
    return workdir


def _build_engine(workdir: Path):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import StrategyLoader

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = str(workdir / "data")
    cfg.backtest_engine_mode = "legacy"
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    cfg.backtest_cost_enabled = True
    bus = EventBus()
    loader = StrategyLoader(str(workdir / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus),
                            OrderExecutor(cfg, bus))
    return cfg, engine, loader


def _pin_timeframe(evolver, timeframe: str) -> None:
    """Force every genome (initial, mutated or immigrant) onto one timeframe."""
    import core.ga.evolver as evolver_mod

    def _pin(chrom: dict) -> dict:
        for gene in chrom.get("categorical", []):
            if gene.name == "timeframes":
                gene.value = timeframe
        return chrom

    original_random = evolver_mod.random_chromosome

    def _random_chromosome(name: str = "ga_strategy") -> dict:
        return _pin(original_random(name))

    evolver_mod.random_chromosome = _random_chromosome

    original_init = evolver._init_population

    def _init(seed_strategies):
        return [_pin(c) for c in original_init(seed_strategies)]

    evolver._init_population = _init

    original_mutate = evolver._mutate

    def _mutate(chrom: dict) -> dict:
        return _pin(original_mutate(chrom))

    evolver._mutate = _mutate


def _structure(chrom: dict) -> dict:
    """Decoded genome shape — explains where a fitness delta came from."""
    try:
        from core.ga.fitness import complexity_penalty
        penalty = round(complexity_penalty(chrom), 4)
    except Exception:  # pragma: no cover - structure is best-effort metadata
        penalty = None
    structural = chrom.get("structural", []) or []
    return {
        "conditions": sum(len(g.conditions) for g in structural),
        "numeric_genes": len(chrom.get("continuous", []) or []),
        "indicators": sum(1 for g in (chrom.get("indicator_genes") or [])
                          if getattr(g, "value", False)),
        "condition_logic": chrom.get("condition_logic", "or"),
        "complexity_penalty": penalty,
    }


def _genome_ledger(population: list[dict]) -> list[dict]:
    rows = []
    for chrom in population:
        result = chrom.get("fitness_result", {}) or {}
        row = {
            "name": chrom.get("name", "?"),
            "fitness": result.get("fitness"),
            "trades": int(result.get("trade_count") or 0),
            "sharpe": result.get("sharpe"),
            "dsr": result.get("dsr"),
            "max_dd": result.get("max_dd"),
            "total_return": result.get("total_return"),
            "flag": result.get("flag", "") or result.get("error", ""),
        }
        row.update(_structure(chrom))
        rows.append(row)
    return rows


def _generation_record(evolver, elapsed: float) -> dict:
    population = list(getattr(evolver, "population", []) or [])
    ledger = _genome_ledger(population)
    scored = [g for g in ledger if isinstance(g["fitness"], (int, float))
              and g["fitness"] > -900]
    fits = [g["fitness"] for g in scored]
    best = max(scored, key=lambda g: g["fitness"], default=None)
    dsrs = [g["dsr"] for g in scored if isinstance(g["dsr"], (int, float))]
    best_result = {}
    if best is not None:
        best_result = next(
            (c.get("fitness_result", {}) or {} for c in population
             if c.get("name") == best["name"]), {})
    published, reasons = evolver._publication_decision(best_result, None)
    return {
        "generation": int(getattr(evolver, "_generation", 0)),
        "population": len(population),
        "genomes_scored": len(scored),
        "genomes_traded": sum(1 for g in ledger if g["trades"] > 0),
        "genomes_zero_trade": sum(1 for g in ledger if g["trades"] == 0),
        "total_trades": sum(g["trades"] for g in ledger),
        "best_fitness": round(best["fitness"], 4) if best else None,
        "mean_fitness": round(sum(fits) / len(fits), 4) if fits else None,
        "best_trades": best["trades"] if best else None,
        "best_sharpe": best["sharpe"] if best else None,
        "best_dsr": best["dsr"] if best else None,
        "max_dsr": round(max(dsrs), 4) if dsrs else None,
        "would_publish": published,
        "rejection_reasons": reasons,
        "elapsed_seconds": round(elapsed, 1),
        "wall_seconds": round(time.time() - evolver._t0, 1),
        "ledger": ledger,
    }


def _write_artifact(out: Path, payload: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ── child: the actual bounded GA run ─────────────────────────────────────

def run_inner(args) -> int:
    from core.ga.evolver import GAStrategyEvolver, GARunConfig

    symbols = _split(args.symbols)
    out = Path(args.out)
    workdir = _prepare_workdir(Path(args.workdir), args.resume)
    _, engine, loader = _build_engine(workdir)

    cfg = GARunConfig(population_size=args.population,
                      generations=args.generations,
                      elite_count=2, immigrant_count=2, max_workers=1,
                      seed=args.seed)
    evolver = GAStrategyEvolver(engine, loader, cfg)
    _pin_timeframe(evolver, args.timeframe)

    artifact = {
        "script": "tools/ga_real_data_curve.py",
        "mode": "real_cached_parquet",
        "data": {"root": str(PROJECT_ROOT / "data" / "market"),
                 "symbols": symbols, "timeframe": args.timeframe,
                 "window": [args.start, args.end]},
        "ga": {"population": args.population, "generations": args.generations,
               "elite": 2, "immigrants": 2, "max_workers": 1, "seed": args.seed,
               "batch_trials_per_generation": args.population},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generations": [],
        "completed": False,
        "truncated": False,
        "notes": [],
    }
    if args.resume:
        artifact["notes"].append("resumed from existing GA checkpoint")
    _write_artifact(out, artifact)

    deadline = time.time() + max(args.max_seconds - 20, 30)
    state = {"stopped_for_budget": False, "gen_started": time.time()}
    evolver._t0 = time.time()

    def _on_progress(payload):
        if isinstance(payload, dict):
            if time.time() > deadline and not state["stopped_for_budget"]:
                state["stopped_for_budget"] = True
                artifact["notes"].append(
                    "stopped accepting new generations at the wall-clock budget; "
                    "the running generation was allowed to finish")
                _write_artifact(out, artifact)
                evolver.stop()
            return
        # End-of-generation tuple (generation, generations, gen_info)
        record = _generation_record(evolver, time.time() - state["gen_started"])
        state["gen_started"] = time.time()
        artifact["generations"].append(record)
        _write_artifact(out, artifact)
        print(f"[gen {record['generation']}] best={record['best_fitness']} "
              f"mean={record['mean_fitness']} traded={record['genomes_traded']}/"
              f"{record['population']} trades={record['total_trades']} "
              f"dsr={record['best_dsr']} publish={record['would_publish']} "
              f"({record['elapsed_seconds']}s)", flush=True)

    evolver.set_progress_callback(_on_progress)

    result = evolver.evolve(symbols, args.start, args.end, seed=args.seed,
                            validation_start=args.validation_start or None,
                            resume=args.resume,
                            window_key=f"real-curve-{args.timeframe}")

    artifact["completed"] = bool(result) and "error" not in result
    artifact["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    artifact["stopped_for_budget"] = state["stopped_for_budget"]
    if "error" in result:
        artifact["error"] = result["error"]
    else:
        artifact["champion"] = {
            "name": result.get("champion_name"),
            "fitness": result.get("fitness"),
            "trades": result.get("trade_count"),
            "sharpe": result.get("sharpe"),
            "dsr": (result.get("dsr") or {}).get("dsr"),
            "n_trials": (result.get("dsr") or {}).get("n_trials"),
            "published": result.get("published"),
            "rejection_reasons": result.get("rejection_reasons"),
            "validation": result.get("validation"),
            "provenance_seed": (result.get("provenance") or {}).get("seed"),
            "generations": result.get("generations"),
            "timeframes": (result.get("provenance") or {}).get("timeframes"),
        }
        artifact["history"] = [
            {k: h.get(k) for k in ("generation", "best_fitness", "avg_fitness",
                                   "best_sharpe", "best_trades", "elapsed")}
            for h in result.get("history", [])]
        artifact["total_seconds"] = round(
            result.get("elapsed_seconds") or 0, 1)
    _write_artifact(out, artifact)
    print(json.dumps({k: v for k, v in artifact.items()
                      if k in ("completed", "truncated", "stopped_for_budget",
                               "total_seconds", "champion", "generations")},
                     indent=2, default=str), flush=True)
    return 0 if artifact["completed"] or artifact["generations"] else 1


# ── parent: hard wall-clock cap + reporting ──────────────────────────────

def run_parent(args) -> int:
    cap = min(int(args.max_seconds), HARD_CAP_SECONDS)
    out = Path(args.out)
    if not args.resume and out.exists():
        out.unlink()

    argv = [sys.executable, str(Path(__file__).resolve()), "--inner"]
    for flag, value in (("--population", args.population),
                        ("--generations", args.generations),
                        ("--symbols", args.symbols), ("--start", args.start),
                        ("--end", args.end), ("--seed", args.seed),
                        ("--timeframe", args.timeframe),
                        ("--max-seconds", cap), ("--out", str(out)),
                        ("--workdir", args.workdir)):
        argv += [flag, str(value)]
    if args.resume:
        argv.append("--resume")
    if args.validation_start:
        argv += ["--validation-start", args.validation_start]

    t0 = time.time()
    timeout_hit = False
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=cap)
        child_stdout, child_rc = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timeout_hit = True
        child_stdout = (exc.stdout or b"").decode("utf-8", "replace") \
            if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        child_rc = -1
    wall = round(time.time() - t0, 1)

    artifact = {}
    if out.exists():
        try:
            artifact = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            artifact = {}
    artifact["hard_cap_seconds"] = cap
    artifact["wall_seconds"] = wall
    artifact["killed_at_cap"] = timeout_hit
    artifact["child_exit_code"] = child_rc
    artifact["stdout_tail"] = child_stdout.strip().splitlines()[-12:]
    if timeout_hit:
        artifact["truncated"] = True
        artifact.setdefault("notes", []).append(
            f"child killed at the {cap}s wall-clock cap; "
            f"{len(artifact.get('generations', []))} generation(s) completed")
    artifact["_out_path"] = str(out)
    _write_artifact(out, artifact)

    _print_report(artifact)
    if not artifact.get("generations"):
        return 1
    return 0 if (artifact.get("completed") or artifact.get("truncated")) else 1


def _print_report(artifact: dict) -> None:
    gens = artifact.get("generations", [])
    print("\n=== GA real-cached-data curve ===")
    if not gens:
        print("no generation completed; see artifact for the failure")
        return
    header = (f"{'gen':>3} {'best_fit':>9} {'mean_fit':>9} {'traded':>7} "
              f"{'zero':>5} {'trades':>7} {'best_sharpe':>11} {'best_dsr':>9} "
              f"{'publish':>7}  rejection_reasons")
    print(header)
    print("-" * len(header))
    for g in gens:
        print(f"{g['generation']:>3} {g['best_fitness']:>9} {g['mean_fitness']:>9} "
              f"{g['genomes_traded']:>3}/{g['population']:<3} "
              f"{g['genomes_zero_trade']:>5} {g['total_trades']:>7} "
              f"{g['best_sharpe']:>11} {g['best_dsr']:>9} "
              f"{str(g['would_publish']):>7}  "
              f"{'; '.join(g['rejection_reasons']) or '-'}")
    champion = artifact.get("champion") or {}
    print(f"\nchampion: {champion.get('name')} fitness={champion.get('fitness')} "
          f"trades={champion.get('trades')} dsr={champion.get('dsr')} "
          f"published={champion.get('published')}")
    print(f"rejection_reasons: {champion.get('rejection_reasons')}")
    print(f"wall_seconds={artifact.get('wall_seconds')} "
          f"cap={artifact.get('hard_cap_seconds')}s "
          f"killed_at_cap={artifact.get('killed_at_cap')} "
          f"completed={artifact.get('completed')}")
    if artifact.get("notes"):
        print("notes: " + " | ".join(artifact["notes"]))
    print(f"artifact: {artifact.get('_out_path', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--symbols", default="ETHUSDT,SOLUSDT,BNBUSDT")
    parser.add_argument("--start", default="2025-07-01")
    parser.add_argument("--end", default="2025-10-01")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--validation-start", default="")
    parser.add_argument("--max-seconds", type=int, default=420,
                        help=f"hard wall-clock cap (max {HARD_CAP_SECONDS})")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--resume", action="store_true",
                        help="continue the previous run's GA checkpoint")
    parser.add_argument("--inner", action="store_true",
                        help="internal: run in-process (no wall-clock wrapper)")
    args = parser.parse_args()

    if args.inner:
        return run_inner(args)
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
