"""GA checkpoint retention + resume — measurable behaviour, no real backtests.

Before this change ``core/ga/evolver.py`` called ``clear_checkpoint()`` on **clean
completion**, so a finished GA job left nothing to resume from: the checkpoint is
``<data_dir>/data/ga_checkpoint.pkl`` and it did not exist after the last shipped
run.  ``resume=True`` could therefore only recover from a crash.

These tests pin the new contract with a stubbed scorer (no data, no backtest,
sub-second runtime):

(a) a cleanly completed ``evolve()`` KEEPS the checkpoint by default and deletes
    it with ``keep_checkpoint=False`` (a stopped run keeps it either way);
(b) ``evolve(resume=True)`` with a larger ``generations`` continues at g+1 and
    stops at the job's total (the loop's start is the checkpoint's generation);
(c) the DSR trial count carries forward across the resume — the exact numbers the
    scorer receives and the ones written into the provenance;
(d) a resume whose window differs from the checkpoint's ``window_key`` refuses
    with the named ``CheckpointWindowMismatchError`` (and leaves the checkpoint
    intact);
(e) ``scripts/ga_job_status.py`` renders the checkpoint line (exists, generation,
    mtime) for a fixture job, and says so when it is missing.

Every artefact lives under ``tmp_path`` (``<tmp>/data/...`` mirrors the real
``<data_dir>/data/...`` layout), so no test writes the operator's ``data/``.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    """Import a script by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def job_status():
    return _load_script("ga_job_status_ckpt_test", "scripts/ga_job_status.py")


class _Cfg:
    """The engine config the evolver reads (duck-typed, like the other GA tests)."""

    data_dir = ""
    backtest_cost_enabled = True
    backtest_taker_fee_pct = 0.04
    backtest_spread_pct = {}
    backtest_engine_mode = "legacy"
    ga_alpha_weight = 1.0
    ga_min_champion_trades = 30


class _Engine:
    config = _Cfg()


def _checkpoint_path(tmp_path: Path) -> Path:
    return Path(tmp_path) / "data" / "ga_checkpoint.pkl"


def _ledger_path(tmp_path: Path) -> Path:
    return Path(tmp_path) / "data" / "ga_trials.json"


def _make_evolver(tmp_path: Path, *, population: int, generations: int,
                  keep_checkpoint: bool = True, max_workers: int = 1):
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    loader = StrategyLoader(str(tmp_path / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    cfg = GARunConfig(population_size=population, generations=generations,
                      elite_count=1, immigrant_count=1,
                      max_workers=max_workers, seed=4242,
                      keep_checkpoint=keep_checkpoint)
    return GAStrategyEvolver(_Engine(), loader, cfg)


def _stub_scorer(calls: list):
    """Stand-in for ``evaluate_population_batch`` — records the DSR inputs."""

    def _stub(population, symbols, date_start, date_end, engine, loader, **kwargs):
        calls.append({"prior_trials": kwargs.get("prior_trials"),
                      "batch_trials": kwargs.get("batch_trials"),
                      "symbols": list(symbols)})
        out = []
        for i, chrom in enumerate(population):
            chrom = dict(chrom)
            chrom["fitness_result"] = {
                "fitness": 10.0 - i, "trade_count": 40, "profit_factor": 1.6,
                "dsr": 0.5, "total_return": 12.0, "sharpe": 2.0, "win_rate": 55.0,
                "max_dd": 1.0, "buy_hold_pct": 3.0, "alpha_vs_buy_hold_pct": 9.0,
                "dsr_detail": {"dsr": 0.5, "n_trials": 0},
            }
            out.append(chrom)
        return out

    return _stub


def _run(tmp_path: Path, *, population: int, generations: int,
         keep_checkpoint: bool = True, resume: bool = False,
         window_key: str | None = None, date_start: str = "2026-01-01",
         date_end: str = "2026-02-01", stop: bool = False):
    """One stubbed ``evolve()`` → ``(result, scorer_calls, progress_tuples)``."""
    import core.ga.fitness as fitness_mod

    calls: list = []
    evolver = _make_evolver(tmp_path, population=population,
                            generations=generations,
                            keep_checkpoint=keep_checkpoint)
    original = fitness_mod.evaluate_population_batch
    fitness_mod.evaluate_population_batch = _stub_scorer(calls)
    seen: list = []
    evolver.set_progress_callback(seen.append)
    if stop:
        evolver.stop()
    try:
        result = evolver.evolve(["BTCUSDT"], date_start, date_end, seed=4242,
                                resume=resume, window_key=window_key)
    finally:
        fitness_mod.evaluate_population_batch = original
    return result, calls, seen


# ══════════════════════════════════════════════════════════════════════════
# (a) retention on a clean completion — the default — and its opt-out
# ══════════════════════════════════════════════════════════════════════════

def test_clean_completion_keeps_the_checkpoint_by_default(tmp_path, monkeypatch):
    """A finished run leaves ``ga_checkpoint.pkl`` behind (retention default)."""
    import core.ga.fitness as fitness_mod

    evolver = _make_evolver(tmp_path, population=2, generations=1)
    calls: list = []
    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer(calls))
    result = evolver.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242)

    ckpt = _checkpoint_path(tmp_path)
    assert ckpt.exists(), "a cleanly completed run must keep its checkpoint"
    assert result["keep_checkpoint"] is True
    assert result["checkpoint"]["kept"] is True
    assert result["checkpoint"]["generation"] == 1
    assert "checkpoint kept at" in result["checkpoint_note"]
    assert result["provenance"]["keep_checkpoint"] is True
    assert result["provenance"]["checkpoint"]["kept"] is True

    # The checkpoint itself carries the identity a resume must prove.
    state = pickle.loads(ckpt.read_bytes())
    assert state["generation"] == 1
    assert state["window_key"] == "2026-01-01~2026-02-01"
    assert state["symbols"] == ["BTCUSDT"]
    assert state["population_hash"]
    assert len(state["population"]) == 2


def test_keep_checkpoint_false_reproduces_the_old_delete_on_completion(tmp_path,
                                                                      monkeypatch):
    """``keep_checkpoint: false`` = the pre-field behaviour (deleted)."""
    import core.ga.fitness as fitness_mod

    evolver = _make_evolver(tmp_path, population=2, generations=1,
                            keep_checkpoint=False)
    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer([]))
    result = evolver.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242)

    ckpt = _checkpoint_path(tmp_path)
    assert not ckpt.exists(), "keep_checkpoint=false must delete on completion"
    assert result["keep_checkpoint"] is False
    assert result["checkpoint"]["kept"] is False
    assert "checkpoint cleared at" in result["checkpoint_note"]


def test_a_stopped_run_keeps_its_checkpoint_even_without_retention(tmp_path,
                                                                   monkeypatch):
    """The crash/stop path is unchanged: it is what ``resume`` recovered from."""
    import core.ga.fitness as fitness_mod

    evolver = _make_evolver(tmp_path, population=2, generations=3,
                            keep_checkpoint=False)
    evolver.stop()
    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer([]))
    result = evolver.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242)

    assert _checkpoint_path(tmp_path).exists()
    assert result["checkpoint"]["kept"] is True
    assert "run stopped" in result["checkpoint_note"]


# ══════════════════════════════════════════════════════════════════════════
# (b) resume CONTINUES at g+1 and stops at the job's generations
# ══════════════════════════════════════════════════════════════════════════

def test_resume_continues_from_the_checkpoint_generation(tmp_path, monkeypatch):
    import core.ga.fitness as fitness_mod

    first_calls: list = []
    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer(first_calls))
    first = _make_evolver(tmp_path, population=2, generations=2)
    seen_first: list = []
    first.set_progress_callback(seen_first.append)
    res1 = first.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242,
                        window_key="2026-01-01~2026-02-01")

    assert len(first_calls) == 2
    assert [t[0] for t in seen_first if isinstance(t, tuple)] == [1, 2]
    assert res1["generations"] == 2
    assert res1["resumed_from_generation"] is None
    assert _checkpoint_path(tmp_path).exists()

    # ── The resume: a LARGER target, the same window ──
    resumed_calls: list = []
    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer(resumed_calls))
    second = _make_evolver(tmp_path, population=2, generations=4)
    seen_second: list = []
    second.set_progress_callback(seen_second.append)
    res2 = second.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242,
                         resume=True, window_key="2026-01-01~2026-02-01")

    # ONLY generations 3 and 4 were scored (not 1..4 again).
    assert len(resumed_calls) == 2
    assert [t[0] for t in seen_second if isinstance(t, tuple)] == [3, 4]
    assert all(t[1] == 4 for t in seen_second if isinstance(t, tuple))
    # The result reports the new total and what it continued from.
    assert res2["generations"] == 4
    assert res2["resumed_from_generation"] == 2
    assert res2["provenance"]["resumed_from_generation"] == 2
    assert res2["checkpoint"]["resumed"] is True
    assert res2["checkpoint"]["resumed_from_generation"] == 2
    assert res2["checkpoint"]["resumed_population_hash"] == res1["checkpoint"][
        "population_hash"]
    # History is the union: the checkpoint's 1..2 plus the resumed 3..4.
    assert [h["generation"] for h in res2["history"]] == [1, 2, 3, 4]


def test_resume_without_a_checkpoint_starts_fresh(tmp_path, monkeypatch):
    """No checkpoint on disk → a normal run (the pre-existing fallback)."""
    import core.ga.fitness as fitness_mod

    calls: list = []
    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer(calls))
    evolver = _make_evolver(tmp_path, population=2, generations=2)
    seen: list = []
    evolver.set_progress_callback(seen.append)
    result = evolver.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242,
                            resume=True)
    assert [t[0] for t in seen if isinstance(t, tuple)] == [1, 2]
    assert result["resumed_from_generation"] is None
    assert result["generations"] == 2


# ══════════════════════════════════════════════════════════════════════════
# (c) the DSR trial count continues across the resume
# ══════════════════════════════════════════════════════════════════════════

def test_trial_count_carries_forward_across_a_resume(tmp_path, monkeypatch):
    """Exact numbers, with the ledger deleted to isolate the checkpoint carry."""
    import core.ga.fitness as fitness_mod

    calls1: list = []
    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer(calls1))
    first = _make_evolver(tmp_path, population=2, generations=2)
    res1 = first.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242,
                        window_key="W")
    assert [c["prior_trials"] for c in calls1] == [0, 2]
    assert [c["batch_trials"] for c in calls1] == [2, 2]
    assert res1["provenance"]["n_trials"] == 4
    assert res1["provenance"]["trials"]["prior_trials"] == 0
    assert res1["provenance"]["trials"]["trials_this_run"] == 4
    # The ledger (data/ga_trials.json) recorded the same 4 trials…
    assert json.loads(_ledger_path(tmp_path).read_text())["trials"] == 4
    # …and the checkpoint now records them too (it did NOT before this change).
    state = pickle.loads(_checkpoint_path(tmp_path).read_bytes())
    assert state["prior_trials"] == 0 and state["trials_this_run"] == 4

    # ── Prune the ledger: only the checkpoint can carry the count forward ──
    _ledger_path(tmp_path).unlink()

    calls2: list = []
    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer(calls2))
    second = _make_evolver(tmp_path, population=2, generations=4)
    res2 = second.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242,
                         resume=True, window_key="W")

    # Generations 3 and 4 are deflated by the 4 trials already performed.
    assert [c["prior_trials"] for c in calls2] == [4, 6]
    assert [c["batch_trials"] for c in calls2] == [2, 2]
    # The champion's N is the cumulative 8, not the resumed segment's 4.
    assert res2["provenance"]["n_trials"] == 8
    assert res2["provenance"]["trials"] == {
        "prior_trials": 4, "trials_this_run": 4, "resumed_trials": 4,
        "n_trials": 8}
    # The ledger is repaired by the resumed run's own `record_trials`.
    assert json.loads(_ledger_path(tmp_path).read_text())["trials"] == 4


def test_trial_count_with_an_intact_ledger_is_not_double_counted(tmp_path,
                                                                 monkeypatch):
    """The ledger already holds the resumed generations → the same 4, not 8."""
    import core.ga.fitness as fitness_mod

    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer([]))
    first = _make_evolver(tmp_path, population=2, generations=2)
    first.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242, window_key="W")

    calls2: list = []
    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer(calls2))
    second = _make_evolver(tmp_path, population=2, generations=4)
    res2 = second.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242,
                         resume=True, window_key="W")
    assert [c["prior_trials"] for c in calls2] == [4, 6]
    assert res2["provenance"]["n_trials"] == 8


# ══════════════════════════════════════════════════════════════════════════
# (d) the window/identity guard
# ══════════════════════════════════════════════════════════════════════════

def test_resume_on_a_different_window_refuses_with_a_named_error(tmp_path,
                                                                 monkeypatch):
    import core.ga.fitness as fitness_mod
    from core.ga.evolver import CheckpointWindowMismatchError

    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer([]))
    first = _make_evolver(tmp_path, population=2, generations=1)
    first.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242)

    # Same evolver instance would keep its own window; a NEW run with another
    # window must be refused, not silently continued.
    second = _make_evolver(tmp_path, population=2, generations=4)
    with pytest.raises(CheckpointWindowMismatchError) as exc:
        second.evolve(["BTCUSDT"], "2026-03-01", "2026-04-01", seed=4242,
                      resume=True)

    message = str(exc.value)
    assert "2026-01-01~2026-02-01" in message      # the checkpoint's window
    assert "2026-03-01~2026-04-01" in message      # the requested one
    # Named for the worker's result file (`error_type`), so a refusal can be
    # told apart from a crash without parsing the message.
    assert type(exc.value).__name__ == "CheckpointWindowMismatchError"
    # The refusal did not consume or damage the checkpoint.
    state = pickle.loads(_checkpoint_path(tmp_path).read_bytes())
    assert state["generation"] == 1
    assert state["window_key"] == "2026-01-01~2026-02-01"

    # A matching window still resumes (the guard is not a blanket refusal).
    calls: list = []
    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer(calls))
    third = _make_evolver(tmp_path, population=2, generations=2)
    result = third.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242,
                          resume=True)
    assert result["resumed_from_generation"] == 1
    assert result["generations"] == 2


def test_an_empty_checkpoint_window_does_not_block_a_resume(tmp_path,
                                                            monkeypatch):
    """A pre-``window_key`` checkpoint (empty key) resumes instead of refusing."""
    import core.ga.fitness as fitness_mod

    monkeypatch.setattr(fitness_mod, "evaluate_population_batch",
                        _stub_scorer([]))
    first = _make_evolver(tmp_path, population=2, generations=1)
    first.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01", seed=4242)

    ckpt = _checkpoint_path(tmp_path)
    state = pickle.loads(ckpt.read_bytes())
    state["window_key"] = ""            # as an old checkpoint would be
    ckpt.write_bytes(pickle.dumps(state))

    second = _make_evolver(tmp_path, population=2, generations=2)
    result = second.evolve(["BTCUSDT"], "2026-03-01", "2026-04-01", seed=4242,
                           resume=True)
    assert result["resumed_from_generation"] == 1
    # The run's own window is kept (an empty checkpoint key cannot overwrite it).
    assert result["provenance"]["window"]["key"] == "2026-03-01~2026-04-01"


def test_walkforward_degrades_a_cross_window_checkpoint_refusal(tmp_path,
                                                              monkeypatch):
    """A retained checkpoint from an earlier window must not kill a WF resume.

    With retention ON the single fixed checkpoint survives a window's clean
    completion, so ``WalkForwardRunner`` (whose own resume state is
    ``ga_wf_state.json``) must treat the evolver's named refusal as "start this
    window's GA fresh", not as a job failure.
    """
    import core.ga.evolver as evolver_mod
    from core.ga.evolver import CheckpointWindowMismatchError
    from core.ga.walkforward import WalkForwardRunner, WFConfig

    seen_resume: list = []

    class _FakeEvolver:
        def __init__(self, engine, loader, config):
            self.config = config

        def set_progress_callback(self, callback):
            self._callback = callback

        def evolve(self, symbols, date_start, date_end, **kwargs):
            seen_resume.append(bool(kwargs.get("resume")))
            if kwargs.get("resume"):
                raise CheckpointWindowMismatchError(
                    "checkpoint belongs to window 'w1' but this run requested 'w2'")
            return {"validation": {}, "sharpe": 1.0, "champion_name": "stub"}

    monkeypatch.setattr(evolver_mod, "GAStrategyEvolver", _FakeEvolver)

    runner = WalkForwardRunner(None, None, str(tmp_path))
    monkeypatch.setattr(runner, "compute_windows",
                        lambda *a, **k: [("t1", "t2", "v1", "v2")])
    monkeypatch.setattr(runner, "_assert_window", lambda *a, **k: 100)
    monkeypatch.setattr(runner, "_save_state", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_clear_state", lambda: None)
    monkeypatch.setattr(runner, "_load_state", lambda: None)

    report = runner.run(["BTCUSDT"], "2025-01-01", "2026-01-01", WFConfig(),
                        object(), resume=True)

    assert seen_resume == [True, False], "the refusal must be retried fresh"
    assert [w.champion_name for w in report.windows] == ["stub"]


def test_walkforward_does_not_resume_the_ga_in_later_windows(tmp_path,
                                                             monkeypatch):
    """Only the first window of a WF run may consume the GA checkpoint."""
    import core.ga.evolver as evolver_mod
    from core.ga.walkforward import WalkForwardRunner, WFConfig

    seen_resume: list = []

    class _FakeEvolver:
        def __init__(self, engine, loader, config):
            self.config = config

        def set_progress_callback(self, callback):
            self._callback = callback

        def evolve(self, symbols, date_start, date_end, **kwargs):
            seen_resume.append(bool(kwargs.get("resume")))
            # Varying train/validation Sharpes keep the report's correlation
            # math well-posed (a constant series divides by a zero stddev).
            n = len(seen_resume)
            return {"validation": {"sharpe": 0.5 + 0.1 * n},
                    "sharpe": 1.0 + 0.1 * n, "champion_name": "stub"}

    monkeypatch.setattr(evolver_mod, "GAStrategyEvolver", _FakeEvolver)

    runner = WalkForwardRunner(None, None, str(tmp_path))
    monkeypatch.setattr(runner, "compute_windows",
                        lambda *a, **k: [("t1", "t2", "v1", "v2"),
                                         ("t2", "t3", "v2", "v3"),
                                         ("t3", "t4", "v3", "v4")])
    monkeypatch.setattr(runner, "_assert_window", lambda *a, **k: 100)
    monkeypatch.setattr(runner, "_save_state", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_clear_state", lambda: None)
    monkeypatch.setattr(runner, "_load_state", lambda: None)

    runner.run(["BTCUSDT"], "2025-01-01", "2026-01-01", WFConfig(), object(),
               resume=True)
    assert seen_resume == [True, False, False]

# ══════════════════════════════════════════════════════════════════════════
# (e) the status CLI renders the checkpoint line
# ══════════════════════════════════════════════════════════════════════════

def _fixture_job(tmp_path: Path):
    job_file = tmp_path / "ga_ckpt0001.json"
    job_file.write_text(json.dumps({
        "population_size": 30, "generations": 30, "max_workers": 4,
        "symbols": ["BTCUSDT", "ETHUSDT"], "seed": 7,
        "keep_checkpoint": True, "resume": True,
        "date_start": "2025-10-01", "date_end": "2026-09-30"}))
    progress_file = Path(str(job_file) + ".progress")
    progress_file.write_text(json.dumps({
        "phase": "gen_complete", "job_type": "ga", "generation": 12,
        "total_generations": 30, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_s": 61.0}))
    log_file = Path(str(job_file) + ".log")
    log_file.write_text("line one\nline two\n")
    checkpoint = tmp_path / "ga_checkpoint.pkl"
    checkpoint.write_bytes(pickle.dumps({
        "population": [{"name": "g0"}, {"name": "g1"}],
        "generation": 12, "window_key": "2025-10-01~2026-09-30",
        "population_hash": "0123456789abcdef",
        "prior_trials": 400, "trials_this_run": 60,
        "keep_checkpoint": True, "saved_at": "2026-09-30T10:00:00"}))
    return job_file, progress_file, log_file, checkpoint


def test_status_cli_renders_the_checkpoint_line(tmp_path, job_status, capsys):
    job_file, progress_file, log_file, checkpoint = _fixture_job(tmp_path)
    argv = ["--job-file", str(job_file), "--progress-file", str(progress_file),
            "--log-file", str(log_file), "--checkpoint-file", str(checkpoint),
            "--no-process-check"]

    assert job_status.main(argv) == 0
    out = capsys.readouterr().out
    assert f"checkpoint     : {checkpoint} exists=True" in out, out
    assert "generation=12" in out
    assert "mtime=" in out
    assert "window=2025-10-01~2026-09-30" in out
    assert "hash=0123456789abcdef" in out
    assert "trials=460" in out
    assert "keep_ckpt=True" in out            # straight from the job file
    assert "continues at generation 13" in out

    # JSON mode carries the same facts.
    assert job_status.main(argv + ["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint"]["generation"] == 12
    assert payload["checkpoint"]["window_key"] == "2025-10-01~2026-09-30"
    assert payload["checkpoint"]["trials_this_run"] == 60

    # A missing checkpoint is reported as such (never silently absent).
    missing = tmp_path / "no_checkpoint.pkl"
    args_missing = list(argv)
    args_missing[args_missing.index("--checkpoint-file") + 1] = str(missing)
    assert job_status.main(args_missing) == 0
    out = capsys.readouterr().out
    assert "exists=False" in out and "nothing to resume from" in out


def test_status_cli_prints_the_last_result_checkpoint_decision(tmp_path,
                                                               job_status,
                                                               capsys):
    job_file, progress_file, log_file, checkpoint = _fixture_job(tmp_path)
    Path(str(job_file) + ".result").write_text(json.dumps({
        "keep_checkpoint": True, "resumed_from_generation": 2,
        "checkpoint": {"path": str(checkpoint), "kept": True, "generation": 12}}))
    rc = job_status.main(["--job-file", str(job_file),
                          "--progress-file", str(progress_file),
                          "--log-file", str(log_file),
                          "--checkpoint-file", str(checkpoint),
                          "--no-process-check"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "last result    : keep_checkpoint=True checkpoint_kept=True " \
           "resumed_from_generation=2" in out, out
