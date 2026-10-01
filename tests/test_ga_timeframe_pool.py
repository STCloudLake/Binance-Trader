"""Job-level timeframe whitelist (``timeframe_pool``).

The timeframe is part of the genome, so an unconstrained GA run spends most of
its wall clock on ``1m`` genomes: a 3-month × 3-symbol 1m backtest is ~390 000
bars per symbol — 15× a 15m run and 60× an 1h run (measured: a 20-genome job with
12 workers finished 11 genomes in 25 minutes).  These tests pin the four
properties the whitelist must have:

* a pool restricts the timeframe gene's **choices** on init *and* mutation, and
  every member stays reachable (many samples, seeded);
* **no pool ⇒ the previous full choice set** — ``1m`` still initialises and still
  decodes, i.e. the shipped search space is untouched;
* an unknown interval fails the **job at load** with a named error, before any
  progress/GA work happens;
* a genome from a pooled run decodes with ``timeframes ⊆ pool`` (non-empty), and
  the run's provenance/result records the effective pool.

Nothing here runs a real GA: the fitness batch scorer is stubbed (same technique
as ``tests/test_ga_dsr_trial_counts.py``) and the worker is exercised through
``ga_run_config`` / ``main`` only.
"""
import asyncio
import copy
import importlib.util
import itertools
import json
import random
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from db.database import init_database
from core.ga.genome import (
    TIMEFRAME_OPTIONS, UnknownTimeframeError, TimeframePoolError,
    chromosome_to_strategy, confine_timeframe_gene, confine_timeframes,
    known_timeframes, parse_timeframe_pool, random_chromosome,
    timeframe_gene_options,
)
from web.routes import ga as routes_ga

ROOT = Path(__file__).resolve().parents[1]
POOL = ["15m", "1h", "4h"]


def _gene(chrom: dict):
    return next(g for g in chrom["categorical"] if g.name == "timeframes")


def _members(chrom: dict) -> set:
    return {tf for tf in str(_gene(chrom).value).split(",") if tf}


def _worker_module():
    spec = importlib.util.spec_from_file_location(
        "ga_worker_timeframe_pool_test", ROOT / "scripts" / "ga_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ══════════════════════════════════════════════════════════════════════
# (a) a pool restricts init and mutation to its members
# ══════════════════════════════════════════════════════════════════════

def test_pool_restricts_random_init_to_the_pool():
    random.seed(20260101)
    seen = set()
    for i in range(200):
        chrom = random_chromosome(f"p{i}", timeframe_pool=POOL)
        gene = _gene(chrom)
        assert gene.options == ["15m,1h", "15m,4h", "1h,4h"], gene.options
        members = _members(chrom)
        assert members and members <= set(POOL), members
        assert chromosome_to_strategy(chrom).timeframes, "decoded timeframes empty"
        seen |= members
    # Every member is reachable: the pool narrows the gene, it does not collapse
    # it onto one cycle.
    assert seen == set(POOL)


def _evolver(tmp_path, pool):
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    class _Engine:
        config = None

    loader = StrategyLoader(str(tmp_path / "pool_strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    return GAStrategyEvolver(
        _Engine(), loader,
        GARunConfig(population_size=4, elite_count=1, immigrant_count=1,
                    seed=4242, timeframe_pool=list(pool)))


def test_pool_restricts_mutation_to_the_pool(tmp_path):
    evolver = _evolver(tmp_path, POOL)
    random.seed(11)
    chrom = random_chromosome("m", timeframe_pool=POOL)
    mutated = set()
    for _ in range(300):
        child = evolver._mutate(copy.deepcopy(chrom))
        members = _members(child)
        assert members and members <= set(POOL), members
        assert set(_gene(child).options) <= {"15m,1h", "15m,4h", "1h,4h"}
        mutated |= members
    assert mutated == set(POOL)


def test_confine_rewrites_a_foreign_gene_value_and_options():
    """A checkpoint written before the pool existed cannot leak ``1m`` in."""
    random.seed(3)
    chrom = random_chromosome("legacy")           # no pool: may carry 1m
    _gene(chrom).value = "1m,5m"
    confine_timeframe_gene(chrom, POOL)
    assert _members(chrom) == {"15m"}             # 1m/5m are gone, none survive
    assert _gene(chrom).options == ["15m,1h", "15m,4h", "1h,4h"]
    # Idempotent, and a no-op without a pool.
    confine_timeframe_gene(chrom, POOL)
    assert _members(chrom) == {"15m"}
    untouched = random_chromosome("keep")
    before = _gene(untouched).value
    confine_timeframe_gene(untouched, None)
    assert _gene(untouched).value == before


def test_single_interval_pool_keeps_the_gene_mutable():
    """``combinations(["1h"], 2)`` is empty — the gene must stay usable."""
    random.seed(5)
    chrom = random_chromosome("s", timeframe_pool=["1h"])
    gene = _gene(chrom)
    assert gene.options == ["1h"]
    for _ in range(10):
        gene.mutate()
    assert gene.value == "1h"
    assert chromosome_to_strategy(chrom, timeframe_pool=["1h"]).timeframes == ["1h"]


# ══════════════════════════════════════════════════════════════════════
# (b) no pool ⇒ the previous full choice set (regression)
# ══════════════════════════════════════════════════════════════════════

def test_without_a_pool_the_full_choice_set_is_unchanged():
    expected = [",".join(c) for c in itertools.combinations(TIMEFRAME_OPTIONS, 2)]
    assert timeframe_gene_options(None) == expected
    assert timeframe_gene_options([]) == expected

    random.seed(4242)
    seen = set()
    for i in range(300):
        chrom = random_chromosome(f"n{i}")            # no pool at all
        assert _gene(chrom).options == expected
        seen |= _members(chrom)
    assert seen == set(TIMEFRAME_OPTIONS)             # 1m/5m/… all reachable again
    assert seen - set(POOL), "the unrestricted run must reach cycles off the pool"


def test_without_a_pool_a_1m_genome_still_decodes_as_1m():
    random.seed(1)
    decoded = None
    for _ in range(300):
        chrom = random_chromosome("x")
        if "1m" in _members(chrom):
            decoded = chromosome_to_strategy(chrom)   # no pool: no clamping
            break
    assert decoded is not None, "1m never initialised without a pool"
    assert "1m" in decoded.timeframes


def test_explicit_none_is_the_same_population_as_omitting_the_argument():
    random.seed(9)
    omitted = random_chromosome("a")
    random.seed(9)
    explicit = random_chromosome("b", timeframe_pool=None)
    assert [_gene(omitted).value, _gene(omitted).options] == \
        [_gene(explicit).value, _gene(explicit).options]


def test_pool_parsing_and_the_canonical_interval_registry():
    from core.market_data.provider import INTERVAL_SPEC

    assert known_timeframes() == list(INTERVAL_SPEC)
    assert parse_timeframe_pool(None) is None
    assert parse_timeframe_pool([]) is None
    assert parse_timeframe_pool("") is None
    assert parse_timeframe_pool(" 1h , 15m ") == ["15m", "1h"]     # shortest first
    assert parse_timeframe_pool(["1h", "1h", "15m"]) == ["15m", "1h"]
    # ``--list-intervals`` names the wider downloadable set; an interval with no
    # registry entry (no bar length) is refused.
    assert "1s" not in known_timeframes()
    with pytest.raises(UnknownTimeframeError) as excinfo:
        parse_timeframe_pool(["1s"])
    assert "1s" in str(excinfo.value)
    with pytest.raises(TimeframePoolError):
        parse_timeframe_pool(123)


# ══════════════════════════════════════════════════════════════════════
# (c) an unknown interval is rejected at job load
# ══════════════════════════════════════════════════════════════════════

def _write_job(tmp_path, name, job):
    path = tmp_path / name
    path.write_text(json.dumps(job), encoding="utf-8")
    return path


@pytest.mark.parametrize("pool,expected_type", [
    (["15m", "7m"], "UnknownTimeframeError"),
    ("1m,1y", "UnknownTimeframeError"),
    (123, "TimeframePoolError"),
])
def test_worker_rejects_a_bad_pool_at_job_load(tmp_path, monkeypatch, pool,
                                               expected_type):
    worker = _worker_module()
    job_file = _write_job(tmp_path, "ga_bad.json", {
        "symbols": ["BTCUSDT"], "population_size": 4, "generations": 1,
        "timeframe_pool": pool})

    monkeypatch.setattr(sys, "argv", [
        "ga_worker.py", "--job-type", "ga", "--job-file", str(job_file)])
    worker.main()

    result = json.loads(Path(str(job_file) + ".result").read_text())
    assert result["error_type"] == expected_type
    assert "timeframe_pool" in result["error"]
    assert expected_type in result["traceback"]
    if expected_type == "UnknownTimeframeError":
        assert "accepted:" in result["error"]
    else:
        assert "list of intervals" in result["error"]
    # Failed at LOAD: no progress file was ever started, no GA work happened.
    assert not Path(str(job_file) + ".progress").exists()


def test_worker_accepts_a_good_pool_at_job_load(tmp_path, monkeypatch):
    worker = _worker_module()
    assert worker.job_timeframe_pool({"timeframe_pool": ["15m", "1h", "4h"]}) == POOL
    assert worker.job_timeframe_pool({}) is None
    assert worker.job_timeframe_pool({"timeframe_pool": []}) is None


# ══════════════════════════════════════════════════════════════════════
# (d) a decoded pooled genome has timeframes ⊆ pool
# ══════════════════════════════════════════════════════════════════════

def test_decoded_pooled_genome_is_non_empty_and_inside_the_pool():
    random.seed(31337)
    for i in range(60):
        chrom = random_chromosome(f"d{i}", timeframe_pool=POOL)
        config = chromosome_to_strategy(chrom, timeframe_pool=POOL)
        assert config.timeframes, "a decoded genome must name a timeframe"
        assert set(config.timeframes) <= set(POOL), config.timeframes

    # A pre-pool chromosome forced to a foreign cycle is clamped on decode.
    chrom = random_chromosome("legacy_decode")
    _gene(chrom).value = "1m,4h"
    assert chromosome_to_strategy(chrom).timeframes == ["1m", "4h"]  # no pool: as-is
    assert chromosome_to_strategy(chrom, timeframe_pool=POOL).timeframes == ["4h"]
    _gene(chrom).value = "1m"
    assert chromosome_to_strategy(chrom, timeframe_pool=POOL).timeframes == ["15m"]
    assert confine_timeframes([], POOL) == ["15m"]
    assert confine_timeframes(["5m"], POOL) == ["15m"]


# ══════════════════════════════════════════════════════════════════════
# (e) the worker passes the pool through, and a pooled run records it
# ══════════════════════════════════════════════════════════════════════

def test_worker_passes_the_pool_into_the_run_config():
    worker = _worker_module()
    cfg = worker.ga_run_config(
        {"timeframe_pool": ["15m", "1h", "4h"], "max_workers": 3}, 20, 5, 4242)
    assert cfg.timeframe_pool == POOL
    assert (cfg.population_size, cfg.generations, cfg.max_workers, cfg.seed) == \
        (20, 5, 3, 4242)
    # Absent field ⇒ unrestricted, exactly as before the feature.
    assert worker.ga_run_config({}, 20, 5, 1).timeframe_pool is None
    with pytest.raises(UnknownTimeframeError):
        worker.ga_run_config({"timeframe_pool": ["7m"]}, 20, 5, 1)


def _run_pooled_evolve(tmp_path, pool, population=8, generations=2):
    """``evolve()`` with the batch scorer stubbed — no backtest, no data reads."""
    import core.ga.fitness as fitness_mod
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    recorded: list = []

    def _stub(population_arg, symbols, date_start, date_end, engine, loader,
              **kwargs):
        recorded.append(copy.deepcopy(list(population_arg)))
        out = []
        for i, chrom in enumerate(population_arg):
            chrom = dict(chrom)
            chrom["fitness_result"] = {
                "fitness": 10.0 - i, "trade_count": 40, "profit_factor": 1.6,
                "dsr": 0.5, "total_return": 12.0, "sharpe": 2.0, "win_rate": 55.0,
                "max_dd": 1.0, "buy_hold_pct": 3.0, "alpha_vs_buy_hold_pct": 9.0,
                "dsr_detail": {"dsr": 0.5, "n_trials": 0},
            }
            out.append(chrom)
        return out

    class _Cfg:
        data_dir = str(tmp_path)
        backtest_cost_enabled = True
        backtest_taker_fee_pct = 0.04
        backtest_spread_pct = {}
        backtest_engine_mode = "legacy"
        ga_alpha_weight = 1.0
        ga_min_champion_trades = 30

    class _Engine:
        config = _Cfg()

    loader = StrategyLoader(str(tmp_path / "pooled_run_strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)

    original = fitness_mod.evaluate_population_batch
    fitness_mod.evaluate_population_batch = _stub
    try:
        evolver = GAStrategyEvolver(
            _Engine(), loader,
            GARunConfig(population_size=population, generations=generations,
                        elite_count=2, immigrant_count=2, max_workers=1,
                        seed=4242, timeframe_pool=list(pool)))
        result = evolver.evolve(["BTCUSDT"], "2026-01-01", "2026-02-01")
    finally:
        fitness_mod.evaluate_population_batch = original
    return result, recorded


def test_a_pooled_run_only_scores_genomes_inside_the_pool(tmp_path):
    result, recorded = _run_pooled_evolve(tmp_path, POOL)

    assert len(recorded) == 2, "both generations were scored"
    for scored_population in recorded:
        assert scored_population
        for chrom in scored_population:
            # The GENE is confined — decode without a pool to prove it is not the
            # decoder's clamp doing the work.
            members = _members(chrom)
            assert members and members <= set(POOL), members
            assert set(chromosome_to_strategy(chrom).timeframes) <= set(POOL)

    # The effective pool is auditable in the run's result and its provenance.
    assert result["timeframe_pool"] == POOL
    assert result["provenance"]["timeframe_pool"] == POOL
    assert set(result["champion_config"]["timeframes"]) <= set(POOL)
    assert result["champion_config"]["timeframes"]


def test_an_unrestricted_run_records_a_null_pool(tmp_path):
    result, recorded = _run_pooled_evolve(tmp_path, [], population=6, generations=1)
    assert result["provenance"]["timeframe_pool"] is None
    assert result["timeframe_pool"] is None
    seen = set()
    for chrom in recorded[0]:
        seen |= _members(chrom)
    assert seen - set(POOL), "an unrestricted run still reaches 1m/5m"


# ══════════════════════════════════════════════════════════════════════
# console wiring: route → job file → worker, and the panel field
# ══════════════════════════════════════════════════════════════════════

TRADER = ("ga_tf_pool_trader", "T1aderPass!")


class FakeSymbolInfo:
    def __init__(self, symbol, status="TRADING", quote_asset="USDT"):
        self.symbol = symbol
        self.status = status
        self.quote_asset = quote_asset
        self.base_asset = symbol[:-len(quote_asset)] if quote_asset else symbol


class FakeUniverse:
    def __init__(self, symbols=("BTCUSDT", "ETHUSDT", "ADAUSDT")):
        self._table = {s: FakeSymbolInfo(s) for s in symbols}

    def get(self, symbol):
        return self._table.get(str(symbol).upper())

    async def get_symbols(self, force: bool = False):
        return list(self._table.values())

    def get_symbols_cached_count(self):
        return len(self._table)


class FakeProc:
    calls: list = []

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.returncode = 0
        FakeProc.calls.append({"argv": argv, "kwargs": kwargs})

    def poll(self):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


@pytest.fixture(scope="module")
def web_app():
    tmpdir = Path(tempfile.mkdtemp(prefix="ga_tf_pool_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "ga.db")
    config.data_dir = str(tmpdir / "data")
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")

    async def _setup():
        await init_database(config.db_path)
        manager = AuthManager(config.db_path, "test-secret-at-least-32-bytes-long!!", 24)
        await manager.create_user(TRADER[0], TRADER[1], "trader", TRADER[0])
        return manager

    auth = asyncio.run(_setup())
    from web.server import create_app
    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    app.state.strategy_loader = SimpleNamespace(
        strategies_dir=str(tmpdir / "strategies"))
    app.state.backtest_engine = SimpleNamespace(name="fake-engine")
    app.state.universe = FakeUniverse()
    yield app


@pytest.fixture()
def trader_client(web_app, monkeypatch):
    monkeypatch.setattr(routes_ga.subprocess, "Popen", FakeProc)
    routes_ga._ga_state.update({"running": False, "error": None, "job_file": None,
                                "params": None, "timeframe_pool": None})
    routes_ga._wf_state.update({"running": False, "error": None, "job_file": None,
                                "params": None, "report": None, "phase": "idle"})
    FakeProc.calls = []
    client = TestClient(web_app)
    response = client.post("/api/auth/login",
                           json={"username": TRADER[0], "password": TRADER[1]})
    assert response.status_code == 200, response.text
    return client


def _job(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_the_console_writes_the_pool_into_the_job_file(trader_client):
    response = trader_client.post("/api/ga/evolve", json={
        "symbols": ["BTCUSDT"], "population_size": 4, "generations": 2,
        "timeframe_pool": ["15m", "1h", "4h"]})
    assert response.status_code == 200, response.text
    job = _job(routes_ga._ga_state["job_file"])
    assert job["timeframe_pool"] == POOL
    assert routes_ga._ga_state["params"]["timeframe_pool"] == POOL
    assert routes_ga._ga_state["timeframe_pool"] == POOL
    # The worker reads exactly that job file.
    assert FakeProc.calls[0]["argv"][-1] == routes_ga._ga_state["job_file"]


def test_the_console_omits_the_pool_when_every_box_is_unticked(trader_client):
    response = trader_client.post("/api/ga/evolve", json={
        "symbols": ["BTCUSDT"], "population_size": 4, "generations": 2,
        "timeframe_pool": []})
    assert response.status_code == 200, response.text
    job = _job(routes_ga._ga_state["job_file"])
    assert "timeframe_pool" not in job          # byte-identical default payload


def test_the_console_rejects_an_unknown_interval(trader_client):
    response = trader_client.post("/api/ga/evolve", json={
        "symbols": ["BTCUSDT"], "timeframe_pool": ["15m", "7m"]})
    assert response.status_code == 400
    error = response.json()["error"]
    assert "7m" in error and "accepted:" in error
    assert routes_ga._ga_state["job_file"] is None
    assert FakeProc.calls == []


def test_the_walkforward_endpoint_accepts_the_pool(trader_client):
    response = trader_client.post("/api/ga/walkforward", json={
        "symbols": ["BTCUSDT"], "population_size": 4, "generations": 2,
        "timeframe_pool": "15m,1h"})
    assert response.status_code == 200, response.text
    assert _job(routes_ga._wf_state["job_file"])["timeframe_pool"] == ["15m", "1h"]
    assert routes_ga._wf_state["params"]["timeframe_pool"] == ["15m", "1h"]


def test_the_panel_exposes_the_pool_picker(trader_client):
    html = trader_client.get("/partials/ga-panel").text
    assert 'id="ga-timeframe-pool"' in html
    assert "gaTimeframePool()" in html
    # Every accepted interval is offered, and the tradeable cycles are pre-checked.
    for tf in known_timeframes():
        assert re.search(r'class="ga-tf-pool[^"]*" value="%s"' % tf, html), tf
    for tf in routes_ga.DEFAULT_GA_TIMEFRAME_POOL:
        assert re.search(r'value="%s"\s+checked' % tf, html), tf
    assert not re.search(r'value="1m"\s+checked', html), "1m must not be pre-checked"
