#!/usr/bin/env python3
"""Evidence harness for GA job progress reporting (no GA algorithm changes).

    python tools/ga_progress_harness.py

Parent mode creates a scratch job file, launches THIS script in ``--child`` mode
with stderr redirected into ``<job>.json.log`` — exactly how
``web/routes/ga.py`` spawns ``scripts/ga_worker.py`` — then polls
``<job>.json.progress`` every 0.2 s and records every distinct payload it sees.

Child mode runs a real ``GAStrategyEvolver`` (4 genomes, 2 generations) with a
**stubbed evaluator that goes through the real 2-process pool** and the real
``scripts/ga_worker.make_progress_reporter``.  Nothing here is a GA algorithm:
the fitness values come from the stub, so the run takes ~5 s and needs no market
data.

It asserts:
  * ``eval_completed`` is written at least once per evaluated strategy and never
    decreases, reaching ``eval_total``;
  * a final payload with ``phase == "gen_complete"`` and ``total_generations``;
  * the enriched keys are present and JSON-serialisable;
  * the job log contains one ``[ga_worker] gen N/M complete`` INFO line per
    generation (plus the evolver's own per-generation line).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Stub evaluator, written to a scratch dir so the pool's worker processes can
#: import it by reference (spawn passes ``sys.path`` to the children).
STUB_SRC = '''
"""Stub GA evaluator for the progress harness (picklable, runs in the pool)."""
import time


def chunk_stub(worker_args):
    queue = worker_args.get("progress_queue")
    start = int(worker_args.get("chunk_start_idx", 0))
    chunk = worker_args.get("population_chunk") or []
    out = []
    for i, _chrom in enumerate(chunk):
        time.sleep(0.5)                      # make the parent poll see the tick
        idx = start + i
        out.append((idx, {
            "fitness": float((idx % 4) + 1) * 1.5,
            "fitness_base": 1.0, "fitness_alpha": 0.5,
            "sharpe": 1.2, "win_rate": 55.0, "profit_factor": 1.4,
            "raw_profit_factor": 1.4, "max_dd": 5.0, "total_return": 3.0,
            "buy_hold_pct": 1.0, "alpha_vs_buy_hold_pct": 2.0, "dsr": 0.4,
            "dsr_detail": {"dsr": 0.4, "n_trials": 10}, "observations": 100,
            "trade_count": 40 + idx, "long_trades": 20, "short_trades": 20,
            "flag": "", "strategy_name": "harness_%d" % idx,
        }))
        if queue is not None:
            queue.put(("genome", start, i + 1))
            queue.put(("bar", start, (i + 1) * 10, max(1, len(chunk)) * 10))
    return out
'''


def _load_ga_worker():
    """Import ``scripts/ga_worker.py`` without executing its main()."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ga_worker_under_test", ROOT / "scripts" / "ga_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_child(job_file: Path) -> int:
    sys.path.insert(0, str(ROOT))
    stub_dir = Path(tempfile.mkdtemp(prefix="ga_progress_stub_"))
    (stub_dir / "ga_progress_stub.py").write_text(STUB_SRC)
    sys.path.insert(0, str(stub_dir))
    import ga_progress_stub

    from core.ga import fitness as F
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    worker = _load_ga_worker()

    # ── The only substitution: the chunk evaluator (real pool, stub work) ──
    real_mp = F.evaluate_population_multiprocess

    def stubbed_mp(population, symbols, date_start, date_end, **kwargs):
        kwargs.pop("max_workers", None)
        kwargs["max_workers"] = 2
        kwargs["evaluate_hook"] = ga_progress_stub.chunk_stub
        return real_mp(population, symbols, date_start, date_end, **kwargs)

    F.evaluate_population_multiprocess = stubbed_mp

    scratch = job_file.parent
    loader = StrategyLoader(str(scratch / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)

    class _Engine:                    # the evolver only reads ``engine.config``
        config = None

    cfg = GARunConfig(population_size=4, generations=2, elite_count=1,
                      immigrant_count=1, max_workers=2, seed=24680)
    evolver = GAStrategyEvolver(_Engine(), loader, cfg)
    # The reporter exposes its merged state, so the child can record EVERY write
    # exactly (the parent's polling below only shows what a UI/operator sees).
    reporter = worker.make_progress_reporter(str(job_file), "ga")
    writes_log = Path(str(job_file) + ".writes.jsonl")

    def tee(info):
        reporter(info)
        with open(writes_log, "a") as handle:
            handle.write(json.dumps(dict(reporter.state)) + "\n")

    evolver.set_progress_callback(tee)
    result = evolver.evolve(["BTCUSDT"], "2025-10-01", "2025-10-05",
                            seed=24680, window_key="harness")
    print(f"child done: generations={result.get('generations')} "
          f"fitness={result.get('fitness')} published={result.get('published')}")
    return 0


def read_progress(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def run_parent(tail: int) -> int:
    scratch = Path(tempfile.mkdtemp(prefix="ga_progress_demo_"))
    job_file = scratch / "ga_harness00.json"
    job_file.write_text(json.dumps({
        "population_size": 4, "generations": 2, "symbols": ["BTCUSDT"],
        "max_workers": 2, "seed": 24680, "_harness": True}, indent=2))
    progress_file = Path(str(job_file) + ".progress")
    log_file = Path(str(job_file) + ".log")

    print(f"scratch job    : {job_file}")
    # Same spawn shape as web/routes/ga.py:359 (worker stderr -> <job>.log).
    with open(log_file, "w") as log_handle:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child",
             "--job-file", str(job_file)],
            stdout=subprocess.DEVNULL, stderr=log_handle, cwd=str(ROOT))

        observed = []
        while proc.poll() is None:
            payload = read_progress(progress_file)
            if payload is not None and (not observed or payload != observed[-1]):
                observed.append(payload)
            time.sleep(0.2)
        payload = read_progress(progress_file)
        if payload is not None and (not observed or payload != observed[-1]):
            observed.append(payload)

    if proc.returncode != 0:
        print(f"child FAILED rc={proc.returncode}")
        print(log_file.read_text(errors="replace"))
        return 1

    print(f"\nprogress payloads seen by a 0.2s poller ({len(observed)} distinct):")
    for payload in observed:
        print("  " + json.dumps({k: payload.get(k) for k in (
            "phase", "generation", "total_generations", "eval_completed",
            "eval_total", "eval_equivalent", "chunk_progress_pct",
            "best_fitness", "best_trades", "elapsed_s", "updated_at")}))

    # Authoritative record: one line per write the worker actually made.
    writes = [json.loads(line) for line in
              Path(str(job_file) + ".writes.jsonl").read_text().splitlines() if line]
    print(f"\nevery write the worker made ({len(writes)}), as "
          "(gen, phase, eval_completed, eval_equivalent):")
    for payload in writes:
        print(f"  ({payload.get('generation')}, {payload.get('phase')}, "
              f"{payload.get('eval_completed')}, {payload.get('eval_equivalent')})")

    per_gen: dict = {}
    for payload in writes:
        if payload.get("phase") == "evolving":
            per_gen.setdefault(payload.get("generation"), []).append(
                payload.get("eval_completed", 0))
    monotone = {g: all(b >= a for a, b in zip(v, v[1:])) for g, v in per_gen.items()}
    gen_complete = [p for p in writes if p.get("phase") == "gen_complete"]
    checks = {
        "eval_completed non-decreasing inside each generation": all(monotone.values()),
        ">=1 write per evaluated strategy (counts 1..4, twice each: count+detail)":
            all(sorted(set(v)) == [1, 2, 3, 4] and len(v) >= 4
                for v in per_gen.values()) and bool(per_gen),
        "each generation reaches eval_total=4":
            all(v and max(v) == 4 for v in per_gen.values()),
        "one gen_complete payload per generation":
            len(gen_complete) == 2 and [p["generation"] for p in gen_complete] == [1, 2],
        "gen_complete has total_generations == 2":
            bool(gen_complete) and all(p.get("total_generations") == 2 for p in gen_complete),
        "gen_complete carries best_fitness/best_trades":
            bool(gen_complete) and all("best_fitness" in p and "best_trades" in p
                                       for p in gen_complete),
        "every write has started_at/updated_at/elapsed_s":
            all(all(k in p for k in ("started_at", "updated_at", "elapsed_s"))
                for p in writes),
        "every write is plain JSON (no numpy scalars)":
            all(json.dumps(p) for p in writes),
    }

    log_text = log_file.read_text(errors="replace")
    gen_lines = [line for line in log_text.splitlines()
                 if "[ga_worker] gen" in line]
    evolver_lines = [line for line in log_text.splitlines()
                     if "core.ga.evolver:evolve" in line and "Gen " in line]
    checks["one INFO line per generation in the job log"] = len(gen_lines) == 2
    checks["evolver per-generation line present"] = len(evolver_lines) == 2

    print(f"\njob log ({log_file}) — {len(log_text.splitlines())} lines:")
    for line in log_text.splitlines()[-tail:]:
        print("  | " + line)

    print("\nchecks:")
    failed = 0
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        failed += 0 if ok else 1
    print(f"\n{'ALL CHECKS PASSED' if not failed else str(failed) + ' CHECK(S) FAILED'}")
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--job-file", help=argparse.SUPPRESS)
    parser.add_argument("--tail", type=int, default=12, help="log lines to print")
    args = parser.parse_args()
    if args.child:
        return run_child(Path(args.job_file))
    os.chdir(ROOT)
    return run_parent(args.tail)


if __name__ == "__main__":
    sys.exit(main())
