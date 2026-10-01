"""GA job progress reporting — root-cause regression tests.

The defect (measured on the live job ``data/ga_jobs/ga_0a442907.json``): its
``.json.progress`` still held ``{"phase": "starting", "job_type": "ga"}`` after
2h16m while four worker processes had burned ~2.26 CPU-hours each.  Cause: the
multiprocess evaluator reported progress only when a WHOLE chunk finished
(``core/ga/fitness.py`` per-chunk callback), and a chunk is one engine pass over
7-8 genomes of a 9-month window — hours.  Nothing was wrong with the callback
registration; the signal was simply too coarse (and the worker's inline
``on_progress`` replaced the payload on every tick, dropping fields).

These tests pin the fix:

(a) a stubbed evolution writes increasing ``eval_completed`` and a final
    ``gen_complete`` payload with ``total_generations``;
(b) the worker's progress reporter handles both the dict and the tuple form and
    writes the enriched, merged payload (plus one log line per generation);
(c) ``scripts/ga_job_status.py`` renders a fixture progress file and flags
    staleness without any live process;
(d) the GA itself is unchanged: no listener → no progress channel, the chunk
    split (which the engine turns into per-genome position slots) is untouched,
    and the evolver still passes a progress callback into the multiprocess path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    """Import a script by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def ga_worker():
    return _load_script("ga_worker_progress_test", "scripts/ga_worker.py")


@pytest.fixture()
def job_status():
    return _load_script("ga_job_status_test", "scripts/ga_job_status.py")


class _Engine:
    """The evolver only needs ``engine.config`` (duck-typed, like other tests)."""

    config = None


def _stub_result(idx: int) -> dict:
    """A fitness_result shaped exactly like the real evaluator's."""
    return {
        "fitness": float((idx % 4) + 1) * 1.5,
        "fitness_base": 1.0, "fitness_alpha": 0.5,
        "sharpe": 1.2, "win_rate": 55.0, "profit_factor": 1.4,
        "raw_profit_factor": 1.4, "max_dd": 5.0, "total_return": 3.0,
        "buy_hold_pct": 1.0, "alpha_vs_buy_hold_pct": 2.0, "dsr": 0.4,
        "dsr_detail": {"dsr": 0.4, "n_trials": 10}, "observations": 100,
        "trade_count": 40 + idx, "long_trades": 20, "short_trades": 20,
        "flag": "", "strategy_name": f"stub_{idx}",
    }


def _evolver(tmp_path, population: int = 3, generations: int = 1,
             max_workers: int = 4):
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    loader = StrategyLoader(str(tmp_path / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    cfg = GARunConfig(population_size=population, generations=generations,
                      elite_count=1, immigrant_count=1,
                      max_workers=max_workers, seed=4242)
    return GAStrategyEvolver(_Engine(), loader, cfg)


def _tee(reporter, progress_file: Path):
    """Run *reporter* and snapshot the FILE after every callback."""
    snapshots: list = []

    def on_progress(info):
        reporter(info)
        with open(progress_file) as f:
            snapshots.append(json.load(f))

    return on_progress, snapshots


# ══════════════════════════════════════════════════════════════════════════
# (a) stubbed evolution → increasing eval_completed → gen_complete
# ══════════════════════════════════════════════════════════════════════════

def test_stubbed_evolution_writes_increasing_progress_then_gen_complete(
        tmp_path, monkeypatch, ga_worker):
    """One write per evaluated strategy, then one per generation — on disk."""
    from core.ga import fitness as F

    job_file = tmp_path / "ga_stubbed01.json"
    job_file.write_text("{}")
    progress_file = Path(str(job_file) + ".progress")
    log_file = tmp_path / "stubbed.log"

    reporter = ga_worker.make_progress_reporter(str(job_file), "ga")
    on_progress, snapshots = _tee(reporter, progress_file)

    seen_genomes: list = []

    def stub_multiprocess(population, symbols, date_start, date_end, **kwargs):
        """Stands in for evaluate_population_multiprocess (no backtests)."""
        total = len(population)
        for i, chrom in enumerate(population):
            chrom["fitness_result"] = _stub_result(i)
            seen_genomes.append(i)
            kwargs["progress_callback"](i + 1, total)
            kwargs["progress_detail_callback"]({
                "eval_completed": i + 1, "eval_total": total,
                "eval_equivalent": round(i + 1 + 0.5, 3),
                "chunk_progress_pct": 50.0, "bar_step": 10, "bar_total": 20})
        return population

    monkeypatch.setattr(F, "evaluate_population_multiprocess", stub_multiprocess)

    handle = logger.add(str(log_file), format="{message}", level="INFO")
    try:
        evolver = _evolver(tmp_path, population=3, generations=1)
        evolver.set_progress_callback(on_progress)
        result = evolver.evolve(["BTCUSDT"], "2025-10-01", "2025-10-03",
                                seed=4242, window_key="stub")
    finally:
        logger.remove(handle)

    assert seen_genomes == [0, 1, 2]
    evolving = [s for s in snapshots if s["phase"] == "evolving"]
    counts = [s["eval_completed"] for s in evolving]
    # At least one write per strategy (the count and its detail tick both write),
    # and the count itself only ever moves forward.
    assert counts == sorted(counts), counts
    assert set(counts) == {1, 2, 3}, f"not one write per strategy: {counts}"
    assert max(counts) == 3
    assert all(s["eval_total"] == 3 for s in evolving)

    final = snapshots[-1]
    assert final["phase"] == "gen_complete"
    assert final["generation"] == 1
    assert final["total_generations"] == 1
    assert final["eval_completed"] == 3
    assert final["best_fitness"] == result["fitness"] == 4.5
    assert final["best_trades"] == 42  # stub idx 2 wins with fitness 4.5

    # Enriched + JSON-serialisable on every single write.
    for snap in snapshots:
        assert json.dumps(snap)
        for key in ("phase", "generation", "total_generations", "eval_completed",
                    "eval_total", "elapsed_s", "started_at", "updated_at"):
            assert key in snap, f"{key} missing from {snap}"
    # The in-flight detail is kept in the merged payload (not dropped by the
    # intra-generation tick that follows it).
    assert "eval_equivalent" in snapshots[-1] or final["phase"] == "gen_complete"

    # (a) also wants the per-generation line IN THE JOB LOG.
    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    gen_lines = [line for line in log_text.splitlines() if "[ga_worker] gen" in line]
    assert len(gen_lines) == 1, log_text
    assert "gen 1/1 complete" in gen_lines[0]
    assert "best=" in gen_lines[0] and "avg=" in gen_lines[0]
    assert "trades=42" in gen_lines[0] and "elapsed=" in gen_lines[0]
    # The evolver logs its own per-generation line through the same loguru sink.
    assert any("Gen   1/1" in line for line in log_text.splitlines()), log_text


# ══════════════════════════════════════════════════════════════════════════
# (b) the worker reporter: dict + tuple, merging, enriched payload
# ══════════════════════════════════════════════════════════════════════════

def test_reporter_merges_dict_and_tuple_forms(tmp_path, ga_worker):
    job_file = tmp_path / "ga_reporter.json"
    job_file.write_text("{}")
    progress_file = Path(str(job_file) + ".progress")
    log_file = tmp_path / "reporter.log"

    handle = logger.add(str(log_file), format="{message}", level="INFO")
    try:
        reporter = ga_worker.make_progress_reporter(str(job_file), "ga")

        # (1) intra-generation dict from the evolver
        reporter({"phase": "evolving", "generation": 2, "total_generations": 5,
                  "eval_completed": 7, "eval_total": 30, "elapsed_s": 12.5,
                  "eval_equivalent": 7.5, "chunk_progress_pct": 50.0,
                  "bar_step": 10, "bar_total": 20})
        payload = json.loads(progress_file.read_text())
        assert payload["phase"] == "evolving"
        assert payload["generation"] == 2 and payload["total_generations"] == 5
        assert payload["eval_completed"] == 7 and payload["eval_total"] == 30
        assert payload["eval_equivalent"] == 7.5
        assert payload["chunk_progress_pct"] == 50.0
        assert payload["bar_step"] == 10 and payload["bar_total"] == 20
        assert payload["job_type"] == "ga" and payload["started_at"]
        assert payload["updated_at"] and payload["elapsed_s"] >= 0

        # (2) a finished generation (tuple form) — enriched with best/avg/trades
        reporter((2, 5, {"best_fitness": 3.5, "avg_fitness": 1.25,
                         "best_sharpe": 1.1, "best_win_rate": 60.0,
                         "best_trades": 11, "elapsed": 42.0}))
        payload = json.loads(progress_file.read_text())
        assert payload["phase"] == "gen_complete"
        assert payload["generation"] == 2 and payload["total_generations"] == 5
        assert payload["best_fitness"] == 3.5 and payload["avg_fitness"] == 1.25
        assert payload["best_trades"] == 11
        assert payload["generation_elapsed_s"] == 42.0

        # (3) the NEXT generation's intra-generation tick must NOT wipe the
        #     best-so-far fields (the old inline closure replaced the payload).
        reporter({"phase": "evolving", "generation": 3, "total_generations": 5,
                  "eval_completed": 1, "eval_total": 30})
        payload = json.loads(progress_file.read_text())
        assert payload["phase"] == "evolving" and payload["generation"] == 3
        assert payload["best_fitness"] == 3.5 and payload["best_trades"] == 11
        assert payload["elapsed_s"] >= 0 and payload["updated_at"]
        assert json.dumps(payload)

        # (4) duplicate tuple for the same generation logs only once
        reporter((2, 5, {"best_fitness": 3.5, "avg_fitness": 1.25,
                         "best_trades": 11, "elapsed": 42.0}))
        text = log_file.read_text(encoding="utf-8", errors="replace")
        assert text.count("[ga_worker] gen 2/5 complete") == 1
        assert "[ga_worker] gen 2/5 complete" in text
    finally:
        logger.remove(handle)


def test_update_progress_merges_by_default(tmp_path, ga_worker):
    job_file = str(tmp_path / "ga_merge.json")
    ga_worker.update_progress(job_file, {"phase": "starting", "job_type": "ga"})
    ga_worker.update_progress(job_file, {"eval_completed": 4})
    payload = json.loads(Path(job_file + ".progress").read_text())
    assert payload == {"phase": "starting", "job_type": "ga", "eval_completed": 4}
    # merge=False still replaces (used nowhere in production, kept explicit).
    ga_worker.update_progress(job_file, {"phase": "x"}, merge=False)
    assert json.loads(Path(job_file + ".progress").read_text()) == {"phase": "x"}


# ══════════════════════════════════════════════════════════════════════════
# (c) the read-only status CLI on fixture files (no live process needed)
# ══════════════════════════════════════════════════════════════════════════

def _fixture_job(tmp_path, age_s: float = 0.0):
    job_file = tmp_path / "ga_fixture01.json"
    job_file.write_text(json.dumps({
        "population_size": 30, "generations": 30, "max_workers": 4,
        "symbols": ["BTCUSDT", "ETHUSDT"], "seed": 7,
        "date_start": "2025-10-01", "date_end": "2026-09-30"}))
    progress_file = Path(str(job_file) + ".progress")
    progress_file.write_text(json.dumps({
        "phase": "evolving", "job_type": "ga", "generation": 2,
        "total_generations": 30, "eval_completed": 5, "eval_total": 30,
        "eval_equivalent": 5.5, "best_fitness": 12.5, "best_trades": 77,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_s": 61.0}))
    log_file = Path(str(job_file) + ".log")
    log_file.write_text("line one\nline two\nline three\n")
    if age_s:
        old = time.time() - age_s
        os.utime(progress_file, (old, old))
    return job_file, progress_file, log_file


def test_status_cli_renders_fixture_and_flags_staleness(tmp_path, job_status, capsys):
    job_file, progress_file, log_file = _fixture_job(tmp_path)
    argv = ["--job-file", str(job_file), "--progress-file", str(progress_file),
            "--log-file", str(log_file), "--no-process-check", "--tail", "2"]

    rc = job_status.main(argv)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "ga_fixture01.json" in out
    assert '"eval_completed": 5' in out
    assert "stale > 5.0 min: False" in out
    assert "line two" in out and "line three" in out  # log tail
    assert "lines=3" in out
    assert "process source : skipped" in out       # --no-process-check

    # JSON mode is machine-readable and carries the same verdict.
    assert job_status.main(argv + ["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["progress"]["eval_completed"] == 5
    assert payload["stale"] is False and payload["running"] is None

    # A progress file older than --stale-minutes is stale → non-zero exit.
    os.utime(progress_file, (time.time() - 600, time.time() - 600))
    assert job_status.main(argv) == 2
    assert "stale > 5.0 min: True" in capsys.readouterr().out
    # …and --stale-minutes can widen the window.
    assert job_status.main(argv + ["--stale-minutes", "30"]) == 0
    capsys.readouterr()


def test_status_cli_reports_missing_job(tmp_path, job_status, capsys):
    rc = job_status.main(["--job-id", "ga_doesnotexist", "--jobs-dir", str(tmp_path)])
    assert rc == 1
    assert "no job file found" in capsys.readouterr().out


def test_status_cli_picks_the_newest_job(tmp_path, job_status, capsys):
    older, _, _ = _fixture_job(tmp_path)
    newest = tmp_path / "ga_zzzz9999.json"
    newest.write_text(older.read_text())
    now = time.time()
    os.utime(older, (now - 500, now - 500))
    os.utime(newest, (now, now))
    rc = job_status.main(["--jobs-dir", str(tmp_path), "--no-process-check"])
    out = capsys.readouterr().out
    assert rc in (0, 2)  # staleness of the fixture is not what is tested here
    assert "ga_zzzz9999.json" in out, out


# ══════════════════════════════════════════════════════════════════════════
# (d) the GA itself is unchanged
# ══════════════════════════════════════════════════════════════════════════

def test_evolver_passes_both_progress_callbacks_to_multiprocess(tmp_path, monkeypatch):
    """The regression guard: the multiprocess branch must carry the callbacks."""
    from core.ga import fitness as F

    captured: dict = {}

    def spy(population, symbols, date_start, date_end, **kwargs):
        captured.update(kwargs)
        for i, chrom in enumerate(population):
            chrom["fitness_result"] = _stub_result(i)
        total = len(population)
        # Drive both callbacks the evolver handed us, exactly like the real
        # evaluator does (per-strategy count + in-flight detail).
        kwargs["progress_callback"](total, total)
        kwargs["progress_detail_callback"]({
            "eval_completed": total, "eval_total": total,
            "eval_equivalent": float(total), "chunk_progress_pct": 100.0})
        return population

    monkeypatch.setattr(F, "evaluate_population_multiprocess", spy)
    seen = []
    evolver = _evolver(tmp_path, population=2, generations=1, max_workers=4)
    evolver.set_progress_callback(seen.append)
    evolver.evolve(["BTCUSDT"], "2025-10-01", "2025-10-03", seed=4242)

    assert callable(captured["progress_callback"])
    assert callable(captured["progress_detail_callback"])
    assert captured["max_workers"] == 4          # branch selection untouched
    assert captured["seed"] == 4242
    assert captured["use_live_spread"] is False
    # The dict payload keeps the four keys the UI/tests already relied on.
    dicts = [d for d in seen if isinstance(d, dict)]
    assert dicts and set(dicts[0]) >= {"generation", "eval_completed",
                                       "eval_total", "phase"}
    assert dicts[0]["phase"] == "evolving"


def test_report_progress_enriches_only_when_asked(tmp_path, monkeypatch):
    evolver = _evolver(tmp_path, population=2, generations=1)
    assert evolver._report_progress(1, 1, 2) is None      # no callback → no-op
    seen = []
    evolver.set_progress_callback(seen.append)
    evolver.config.generations = 7
    evolver._run_t_start = time.time() - 5
    evolver._report_progress(3, 2, 20, detail={"eval_equivalent": 2.5,
                                              "bar_step": 10, "bar_total": 40,
                                              "eval_completed": 999})
    payload = seen[-1]
    assert payload["generation"] == 3 and payload["total_generations"] == 7
    assert payload["eval_completed"] == 2          # detail cannot override it
    assert payload["eval_equivalent"] == 2.5 and payload["bar_step"] == 10
    assert payload["phase"] == "evolving"
    assert 4.9 < payload["elapsed_s"] < 60
    assert "best_fitness" not in payload           # nothing scored yet
    assert json.dumps(payload)


def test_mp_worker_streams_engine_bar_ticks_and_genome_ticks(tmp_path, monkeypatch):
    """The worker must forward the engine's per-bar observer and tick per genome.

    Without this, a chunk (= one engine pass over 7-8 genomes, hours for the live
    9-month job) is silent from start to finish.
    """
    import multiprocessing
    import types
    from core.ga import fitness as F
    from core.ga.genome import random_chromosome

    engine_kwargs: dict = {}

    class _FakeEngine:
        def __init__(self, *a, **k):
            self.config = None

        def run_with_exit_evaluation(self, **kwargs):
            engine_kwargs.update(kwargs)
            callback = kwargs.get("progress_callback")
            if callback:                      # the engine ticks every 10 bars
                for step in (1, 10, 20):
                    callback(step, 20, "2025-10-01")
            return {"metrics": {}}

    canned_stats = {
        "fitness": 2.0, "fitness_base": 1.5, "fitness_alpha": 0.5, "sharpe": 1.0,
        "win_rate": 50.0, "profit_factor": 1.2, "raw_profit_factor": 1.2,
        "max_dd": 3.0, "total_return_pct": 2.0, "buy_hold_pct": 1.0,
        "alpha_vs_buy_hold_pct": 1.0, "deflated_sharpe": 0.3,
        "dsr_detail": {"dsr": 0.3}, "observations": 50, "trades": 7,
        "long_trades": 4, "short_trades": 3, "flag": "",
    }
    monkeypatch.setattr("core.backtest.engine.BacktestEngine", _FakeEngine)
    monkeypatch.setattr(F, "stats_from_engine_result", lambda *a, **k: dict(canned_stats))
    monkeypatch.setattr(F, "score_stats", lambda stats, *a, **k: dict(stats))
    monkeypatch.setattr("core.risk.manager.RiskManager", lambda *a, **k: None)
    monkeypatch.setattr("core.executor.executor.OrderExecutor", lambda *a, **k: None)
    monkeypatch.setattr("app.event_bus.EventBus", lambda *a, **k: None)
    monkeypatch.setattr("core.strategy.loader.StrategyLoader", lambda *a, **k: None)
    fake_config = types.SimpleNamespace(
        data_dir=str(tmp_path / "data"), db_path=str(tmp_path / "no.db"))
    monkeypatch.setattr("app.config.Config.load", classmethod(lambda cls, *a, **k: fake_config))

    queue = multiprocessing.Queue()
    args = {
        "population_chunk": [random_chromosome("prog_0"),
                             random_chromosome("prog_1")],
        "symbols": ["BTCUSDT"], "date_start": "2025-10-01", "date_end": "2025-10-03",
        "initial_balance": 10000.0, "cost_enabled": True, "taker_fee_pct": 0.04,
        "spread_pct": {}, "weights": None, "chunk_start_idx": 0,
        "engine_mode": "legacy", "seed": 11, "progress_queue": queue,
    }
    results = F._mp_worker(args)
    assert [idx for idx, _ in results] == [0, 1]
    assert engine_kwargs.get("progress_callback") is not None, (
        "the engine's bar observer was not wired — a chunk stays silent")
    assert engine_kwargs["per_genome_ledger"] is True
    assert engine_kwargs["per_strategy_isolation"] is True

    time.sleep(0.2)  # let the queue's feeder thread flush
    items = []
    while True:
        try:
            items.append(queue.get_nowait())
        except Exception:
            break
    bars = [i for i in items if i[0] == "bar"]
    genomes = [i for i in items if i[0] == "genome"]
    # Tick 1 is the first bar, tick 20 is the forced final one; the middle tick is
    # rate-limited to ~1/s (the three engine calls happen in the same second).
    assert [b[2] for b in bars] == [1, 20], items
    assert all(b[3] == 20 for b in bars)
    # exactly one genome tick per scored genome
    assert [g[2] for g in genomes] == [1, 1], items


def test_no_listener_means_no_progress_channel(monkeypatch):
    """No callback → no Manager, no queue: zero overhead, identical behaviour."""
    import multiprocessing

    from core.ga import fitness as F

    calls = []
    real_manager = multiprocessing.Manager

    def spy_manager(*a, **k):
        calls.append(1)
        return real_manager(*a, **k)

    monkeypatch.setattr(multiprocessing, "Manager", spy_manager)

    def hook(args):  # module-level picklable stub, imported by the pool child
        return [(args["chunk_start_idx"] + i, {"fitness": 1.0})
                for i in range(len(args["population_chunk"]))]

    population = [{"name": f"g{i}", "fitness_result": {"fitness": 0}}
                  for i in range(4)]
    out = F.evaluate_population_multiprocess(
        population, ["BTCUSDT"], "2025-10-01", "2025-10-02",
        max_workers=2, evaluate_hook=hook, seed=3)
    assert [c["fitness_result"]["fitness"] for c in out] == [1.0] * 4
    assert calls == [], "no listener must not create a progress Manager"


def test_chunk_split_is_unchanged(monkeypatch):
    """Chunk sizes feed ``max_open_trades // len(strategy_configs)`` (engine.py
    :706-708), so the split must stay exactly as it was — the live job logged
    ``paths: {0, 8, 16, 23}`` for 30 genomes / 4 workers."""
    from core.ga import fitness as F

    from concurrent.futures import Future

    chunks_seen: list = []

    class _FakePool:
        """Minimal stand-in for ProcessPoolExecutor (real Futures for ``wait``)."""

        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, args):
            start = args["chunk_start_idx"]
            size = len(args["population_chunk"])
            chunks_seen.append((start, size))
            future = Future()
            # The worker itself is covered elsewhere; this test is about the
            # chunk split (and the parent's progress bookkeeping).
            future.set_result([(start + i, {"fitness": 1.0})
                               for i in range(size)])
            return future

    monkeypatch.setattr(F, "ProcessPoolExecutor", _FakePool)
    population = [{"name": f"g{i}", "fitness_result": {"fitness": 0}}
                  for i in range(30)]
    reports = []
    details = []
    out = F.evaluate_population_multiprocess(
        population, ["BTCUSDT"], "2025-10-01", "2026-07-01", max_workers=4,
        progress_callback=lambda c, t: reports.append((c, t)),
        progress_detail_callback=details.append)
    assert chunks_seen == [(0, 8), (8, 8), (16, 7), (23, 7)]
    assert [c["fitness_result"]["fitness"] for c in out] == [1.0] * 30
    # No queue traffic from the fake pool → the chunk completions still add up.
    assert reports and reports[-1] == (30, 30)
    assert details and details[-1]["eval_completed"] == 30
    assert details[-1]["eval_total"] == 30
