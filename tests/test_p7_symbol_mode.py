"""P7-S2: job-level per-symbol evolution with honest trial accounting.

What this file pins
-------------------
1. **Mode semantics** — ``symbol_mode`` is a *job field* (``GARunConfig``), its
   default is ``"pooled"`` (an absent field), a typo raises the named
   ``UnknownSymbolModeError`` at load, and there is deliberately **no**
   ``ga.symbol_mode`` config key to inherit.
2. **One population per symbol** — ``per_symbol`` scores each candidate on
   exactly one symbol, one independent population per symbol, in ``symbols``
   order; ``pooled`` still scores every candidate on the whole basket.
3. **One symbol per champion** — every per-symbol champion records exactly the
   symbol it was evaluated on, in ``StrategyConfig.symbols`` (the field the live
   engine already enforces) and in its provenance; the live evaluation path only
   ever evaluates it on that symbol.
4. **Trial accounting** — a per-symbol run performs
   ``len(symbols) × population × generations`` evaluations and **every**
   champion's ``n_trials`` is that total (plus the prior ledger), not the
   per-symbol population: the DSR is deflated by what was really tried.
5. **Default-off identity** — with the field absent (or ``"pooled"``) the run is
   the pre-S2 run: byte-identical population, history, champion, trial counts and
   result payload, both with a stubbed scorer and on the **real** batch scorer,
   and byte-identical to a ``git worktree`` at the pre-change revision.
6. **The mechanism itself** — on a synthetic landscape where the signal exists on
   one symbol only, per-symbol evolution finds it while pooled selection averages
   it away.  This validates the mechanism even if real data has no edge.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
#: The revision this stage builds on (HEAD when S2 started).
BASELINE_REVISION = "d9849a2"

#: Sentinel: "do not pass ``symbol_mode`` at all" (the HEAD-compatible shape).
ABSENT = object()


# ══════════════════════════════════════════════════════════════════════
# (a) the mode flag: parsing, default, and no config key
# ══════════════════════════════════════════════════════════════════════

def test_parse_symbol_mode_defaults_to_pooled():
    from core.ga.evolver import (PER_SYMBOL, POOLED, SYMBOL_MODES,
                                 parse_symbol_mode)

    assert SYMBOL_MODES == ("pooled", "per_symbol")
    assert parse_symbol_mode(None) == POOLED
    assert parse_symbol_mode("") == POOLED
    assert parse_symbol_mode("   ") == POOLED
    assert parse_symbol_mode("pooled") == POOLED
    assert parse_symbol_mode(" POOLED ") == POOLED
    assert parse_symbol_mode("per_symbol") == PER_SYMBOL
    assert parse_symbol_mode("Per_Symbol") == PER_SYMBOL


def test_an_unknown_mode_is_a_named_error():
    from core.ga.evolver import UnknownSymbolModeError, parse_symbol_mode

    with pytest.raises(UnknownSymbolModeError) as excinfo:
        parse_symbol_mode("per-symbol")
    assert "per-symbol" in str(excinfo.value)
    assert "pooled" in str(excinfo.value) and "per_symbol" in str(excinfo.value)
    with pytest.raises(UnknownSymbolModeError):
        parse_symbol_mode("perSymbol")


def test_the_run_config_defaults_to_pooled():
    from core.ga.evolver import POOLED, GARunConfig

    assert GARunConfig().symbol_mode == POOLED
    assert GARunConfig(symbol_mode="per_symbol").symbol_mode == "per_symbol"


def test_there_is_no_config_key_for_the_mode():
    """A **job field**, not a config key: no operator inherits a GA shape.

    ``config/config.yaml`` has no ``ga.symbol_mode`` and ``Config`` exposes no
    ``ga_symbol_mode`` attribute, so specialisation can only be requested per run
    (the same discipline as ``timeframe_pool``/``benchmark_mode``).
    """
    from app.config import Config
    import yaml

    shipped = yaml.safe_load(
        (ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert "symbol_mode" not in (shipped.get("ga") or {})
    Config._instance = None
    try:
        config = Config.load("sim")
        assert not hasattr(config, "ga_symbol_mode")
    finally:
        Config._instance = None


def _worker_module():
    path = ROOT / "scripts" / "ga_worker.py"
    spec = importlib.util.spec_from_file_location(
        "ga_worker_symbol_mode_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_reads_the_job_field_and_raises_by_name():
    from core.ga.evolver import UnknownSymbolModeError

    worker = _worker_module()
    assert worker.job_symbol_mode({}) == "pooled"          # absent = pooled
    assert worker.job_symbol_mode({"symbol_mode": "  "}) == "pooled"
    assert worker.job_symbol_mode({"symbol_mode": "per_symbol"}) == "per_symbol"
    with pytest.raises(UnknownSymbolModeError):
        worker.job_symbol_mode({"symbol_mode": "perSymbol"})

    assert worker.ga_run_config({}, 20, 5, 1).symbol_mode == "pooled"
    assert worker.ga_run_config({"symbol_mode": "per_symbol"},
                                20, 5, 1).symbol_mode == "per_symbol"


def test_a_symbol_mode_typo_fails_the_job_at_load(tmp_path, monkeypatch):
    """The typo lands in the result file before any GA work happens."""
    worker = _worker_module()
    job_file = tmp_path / "ga_bad_symbol_mode.json"
    job_file.write_text(json.dumps({
        "symbols": ["BTCUSDT"], "population_size": 4, "generations": 1,
        "symbol_mode": "per-symbol"}), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "ga_worker.py", "--job-type", "ga", "--job-file", str(job_file)])
    worker.main()

    result = json.loads(Path(str(job_file) + ".result").read_text())
    assert result["error_type"] == "UnknownSymbolModeError"
    assert "per-symbol" in result["error"]
    assert not Path(str(job_file) + ".progress").exists()


# ══════════════════════════════════════════════════════════════════════
# (b) the stub-scorer seam: what the run actually evaluates
# ══════════════════════════════════════════════════════════════════════

_STUB_RESULT = {
    "fitness": 10.0, "trade_count": 40, "profit_factor": 1.6, "dsr": 0.5,
    "total_return": 12.0, "sharpe": 2.0, "win_rate": 55.0, "max_dd": 1.0,
    "buy_hold_pct": 3.0, "alpha_vs_buy_hold_pct": 9.0,
    "dsr_detail": {"dsr": 0.5, "n_trials": 0},
}


class _Cfg:
    """Duck-typed engine config: only what the evolver/scorer read."""

    ga_alpha_weight = 1.0
    ga_min_champion_trades = 30
    ga_benchmark_mode = "buy_hold"
    ga_regime_conditioning = False
    backtest_cost_enabled = True
    backtest_taker_fee_pct = 0.04
    backtest_spread_pct = {}
    backtest_engine_mode = "legacy"


class _Engine:
    config = _Cfg()


def _stub_run(root: Path, symbols, *, symbol_mode=ABSENT, population=4,
              generations=2, prior_ledger=0, seed=4242, resume=False,
              keep_checkpoint=False, fitness_of=None, extra_cfg=None):
    """Run ``evolve`` with the batch scorer stubbed — no backtest, no data reads.

    Returns ``(result, calls, loader, evolver)`` where ``calls`` records one entry
    per engine pass: the basket it was handed, the DSR batch/prior trial counts,
    and a repr of the population it scored (so "independent population" is
    checkable).
    """
    import core.ga.fitness as fitness_mod
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    if prior_ledger:
        (root / "data" / "ga_trials.json").write_text(
            json.dumps({"trials": int(prior_ledger)}), encoding="utf-8")

    loader = StrategyLoader(str(root / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)

    calls: list[dict] = []

    def _stub(population_arg, symbols_arg, date_start, date_end, engine, loader_,
              **kwargs):
        calls.append({
            "symbols": list(symbols_arg),
            "batch_trials": kwargs.get("batch_trials"),
            "prior_trials": kwargs.get("prior_trials"),
            "population": [[g.name for g in c.get("continuous", [])]
                           for c in population_arg],
        })
        out = []
        for i, chrom in enumerate(population_arg):
            chrom = dict(chrom)
            score = dict(_STUB_RESULT)
            score["fitness"] = (fitness_of(chrom, list(symbols_arg), i)
                                if fitness_of else 10.0 - i)
            chrom["fitness_result"] = score
            out.append(chrom)
        return out

    kwargs = dict(population_size=population, generations=generations,
                  elite_count=2, immigrant_count=1, max_workers=1, seed=seed,
                  keep_checkpoint=keep_checkpoint)
    if symbol_mode is not ABSENT:
        kwargs["symbol_mode"] = symbol_mode

    engine = _Engine()
    if extra_cfg:
        for key, value in extra_cfg.items():
            setattr(engine.config, key, value)

    original = fitness_mod.evaluate_population_batch
    fitness_mod.evaluate_population_batch = _stub
    try:
        evolver = GAStrategyEvolver(engine, loader, GARunConfig(**kwargs))
        result = evolver.evolve(list(symbols), "2026-01-01", "2026-02-01",
                                resume=resume, seed=seed,
                                window_key="2026-01-01~2026-02-01")
    finally:
        fitness_mod.evaluate_population_batch = original
    return result, calls, loader, evolver


def test_pooled_mode_scores_every_candidate_on_the_whole_basket(tmp_path):
    basket = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    result, calls, _, _ = _stub_run(tmp_path / "absent", basket,
                                    population=3, generations=2)
    assert [c["symbols"] for c in calls] == [basket] * 2
    # The pooled result has no per-symbol envelope at all.
    assert "symbol_mode" not in result and "champions" not in result
    assert result["champion_config"]["symbols"] == basket


def test_per_symbol_mode_evolves_one_population_per_symbol(tmp_path):
    basket = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    result, calls, _, _ = _stub_run(tmp_path / "per_symbol", basket,
                                    symbol_mode="per_symbol",
                                    population=3, generations=2)

    # One engine pass per (symbol, generation), each on exactly that symbol.
    assert [c["symbols"] for c in calls] == [
        [s] for s in basket for _ in range(2)]
    # Arms are independent: the second arm's first population is not the first
    # arm's (each arm draws its own genomes from the continuing RNG stream).
    assert calls[0]["population"] != calls[2]["population"]
    # The envelope: one entry per symbol, in `symbols` order.
    assert result["symbol_mode"] == "per_symbol"
    assert result["n_symbols"] == 3
    assert [c["symbol"] for c in result["champions"]] == basket


def test_each_per_symbol_champion_carries_exactly_one_symbol(tmp_path):
    basket = ["BTCUSDT", "ETHUSDT"]
    result, _, loader, _ = _stub_run(tmp_path / "champions", basket,
                                     symbol_mode="per_symbol",
                                     population=3, generations=2)
    assert len(result["champions"]) == 2
    for symbol, champion in zip(basket, result["champions"]):
        assert champion["champion_symbol"] == symbol
        assert champion["symbols_evaluated"] == [symbol]
        # The YAML the execution path loads names exactly that symbol.
        assert champion["champion_config"]["symbols"] == [symbol]
        saved = loader.load(champion["champion_name"])
        assert saved.symbols == [symbol], (
            f"{champion['champion_name']} does not restrict itself to {symbol}")
        assert symbol in champion["champion_name"]
        provenance = champion["provenance"]
        assert provenance["symbols"] == [symbol]
        assert provenance["symbol_mode"] == "per_symbol"
        assert provenance["champion_symbol"] == symbol
        assert provenance["n_symbols"] == 2
        # The trial-count basis is self-described on the artefact.
        search = provenance["trials"]["search"]
        assert search["symbol_mode"] == "per_symbol"
        assert search["arm_symbol"] == symbol
        assert search["arm_population"] == 3
        assert search["evaluations_this_arm"] == 3 * 2
        assert search["evaluations_whole_run"] == 3 * 2 * 2


class _FakeMarketData:
    """Minimal stand-in for MarketDataProvider (no network, no frames needed)."""

    def __init__(self, watched):
        self.watched_symbols = list(watched)


@pytest.mark.asyncio
async def test_a_per_symbol_champion_can_only_trade_its_own_symbol(tmp_path):
    """The single recorded symbol reaches the LIVE evaluation path.

    The watcher holds five symbols; the per-symbol champion was evaluated on one.
    A real ``StrategyEngine.evaluate_all_now()`` must only evaluate it on that one
    — the mechanism the per-symbol mode relies on to restrict trading.
    """
    from app.config import Config
    from app.event_bus import EventBus
    from core.strategy.engine import StrategyEngine

    basket = ["BTCUSDT", "ETHUSDT"]
    result, _, loader, _ = _stub_run(tmp_path / "live_path", basket,
                                     symbol_mode="per_symbol",
                                     population=3, generations=2)
    champion = loader.load(result["champions"][1]["champion_name"])
    assert champion.enabled is True
    assert champion.symbols == ["ETHUSDT"]

    Config._instance = None
    try:
        config = Config.load("sim")
        engine = StrategyEngine(
            config, EventBus(),
            _FakeMarketData(["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
                             "XRPUSDT"]))
    finally:
        Config._instance = None
    engine._strategies = {champion.name: champion}

    evaluated: list = []

    async def _record(symbol, interval, strategy, publish=False):
        evaluated.append(symbol)

    engine._evaluate = _record  # type: ignore[assignment]
    await engine.evaluate_all_now()

    assert set(evaluated) == {"ETHUSDT"}, (
        f"the per-symbol champion traded symbols outside its own: {evaluated}")
    assert not (set(evaluated) & {"BTCUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"})


# ══════════════════════════════════════════════════════════════════════
# (c) honest trial accounting
# ══════════════════════════════════════════════════════════════════════

def test_the_trial_counter_counts_every_evaluation_not_the_population(tmp_path):
    """``N_symbols × population × generations`` — for every champion."""
    population, generations = 4, 3
    basket = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    result, calls, _, _ = _stub_run(
        tmp_path / "counter", basket, symbol_mode="per_symbol",
        population=population, generations=generations)

    evaluations = population * generations * len(basket)
    assert len(calls) == len(basket) * generations == 9
    # Each pass is scored against EVERYTHING already performed this run.
    assert [c["batch_trials"] for c in calls] == [population] * 9
    assert [c["prior_trials"] for c in calls] == [
        population * k for k in range(9)]

    for champion in result["champions"]:
        n_trials = champion["provenance"]["n_trials"]
        assert n_trials == evaluations == 36
        # ... and it is NOT the per-symbol population, nor one arm's generations
        # (the two numbers an under-counting implementation would report).
        assert n_trials != population
        assert n_trials != population * generations
        trials = champion["provenance"]["trials"]
        assert trials["n_trials"] == evaluations
        assert trials["prior_trials"] == 0
        assert trials["trials_this_run"] == evaluations
    # The top-level mirror of the best arm carries the same number.
    assert result["provenance"]["n_trials"] == evaluations


def test_a_prior_ledger_raises_the_per_symbol_count(tmp_path):
    population, generations, prior = 4, 2, 1440
    result, calls, _, _ = _stub_run(
        tmp_path / "ledger", ["BTCUSDT", "ETHUSDT"], symbol_mode="per_symbol",
        population=population, generations=generations, prior_ledger=prior)

    assert [c["prior_trials"] for c in calls] == [prior, prior + 4,
                                                  prior + 8, prior + 12]
    for champion in result["champions"]:
        assert champion["provenance"]["n_trials"] == prior + 16


def test_pooled_mode_still_counts_one_population_per_generation(tmp_path):
    """The pooled arithmetic is untouched (the pre-S2 number)."""
    result, calls, _, _ = _stub_run(tmp_path / "pooled_counter",
                                    ["BTCUSDT", "ETHUSDT"],
                                    population=4, generations=3)
    assert [c["prior_trials"] for c in calls] == [0, 4, 8]
    assert result["provenance"]["n_trials"] == 12


# ══════════════════════════════════════════════════════════════════════
# (d) the mechanism: a signal on ONE symbol survives per-symbol, not pooled
# ══════════════════════════════════════════════════════════════════════

#: A synthetic landscape built on the ALWAYS-PRESENT ``ml_threshold`` gene
#: (range 0.5..0.85, decoded to ``ml_config.confidence_threshold``): symbol A
#: rewards one value, symbol B another — and A's term is much steeper, so the
#: AVERAGE (what one pooled equity curve over the basket measures) is dominated by
#: A and B's reward is invisible to the pooled search.
_SIGNAL_TARGET = {"BTCUSDT": 0.58, "ETHUSDT": 0.80}
_SIGNAL_STEEP = {"BTCUSDT": 6.0, "ETHUSDT": 0.6}


def _signal_gene_value(chrom) -> float:
    for gene in chrom.get("continuous", []):
        if gene.name == "ml_threshold":
            return float(gene.value)
    return 0.6


def _signal_score(value: float, symbol: str) -> float:
    return -_SIGNAL_STEEP[symbol] * abs(value - _SIGNAL_TARGET[symbol])


def _pooled_signal_fitness(chrom, symbols) -> float:
    """The pooled evaluation = the mean of the per-symbol results (one curve)."""
    value = _signal_gene_value(chrom)
    return sum(_signal_score(value, s) for s in symbols) / len(symbols)


def _published_signal_value(result, index=None) -> float:
    """The champion's gene value, read back from its published artefact."""
    champion = result if index is None else result["champions"][index]
    return float(champion["champion_config"]["ml_config"]["confidence_threshold"])


def test_per_symbol_finds_a_one_symbol_signal_that_pooled_averages_away(tmp_path):
    """Synthetic validation of the mechanism (real data may have no edge at all).

    The stub scorer gives symbol A a steep reward around 0.58 and symbol B a
    shallow one around 0.80, so a **pooled** population (scored on the mean over
    both symbols) converges on A's value and never sees B, while **per-symbol**
    populations find each symbol's own value.  Both arms then get the same final
    exam: the champion's B-only score.
    """
    basket = ["BTCUSDT", "ETHUSDT"]

    def _fitness(chrom, symbols, index):
        return _pooled_signal_fitness(chrom, symbols)

    root = tmp_path / "synthetic"
    pooled, calls, _, _ = _stub_run(
        root / "pooled", basket, population=40, generations=40,
        fitness_of=_fitness, seed=11)
    per_symbol, _, _, _ = _stub_run(
        root / "per_symbol", basket, symbol_mode="per_symbol",
        population=40, generations=40, fitness_of=_fitness, seed=11)

    assert [c["symbol"] for c in per_symbol["champions"]] == basket
    pooled_value = _published_signal_value(pooled)
    btc_value = _published_signal_value(per_symbol, 0)
    eth_value = _published_signal_value(per_symbol, 1)

    # The pooled search chased the steep symbol and never captured the shallow one.
    assert abs(pooled_value - _SIGNAL_TARGET["BTCUSDT"]) < 0.05, (
        f"the pooled population did not converge on the steep symbol: "
        f"{pooled_value}")
    # The per-symbol arms each found their OWN symbol's signal.
    assert abs(btc_value - _SIGNAL_TARGET["BTCUSDT"]) < 0.05, btc_value
    assert abs(eth_value - _SIGNAL_TARGET["ETHUSDT"]) < 0.06, (
        f"the per-symbol ETH population did not find its own signal: {eth_value}")

    # Same final exam: score the pooled champion (and the per-symbol ETH champion)
    # on ETHUSDT alone.  Pooling lost that signal; specialisation kept it.
    pooled_on_eth = _signal_score(pooled_value, "ETHUSDT")
    per_symbol_on_eth = _signal_score(eth_value, "ETHUSDT")
    assert abs(pooled_value - _SIGNAL_TARGET["ETHUSDT"]) > 0.15, (
        "the pooled champion accidentally landed on the ETH signal — the "
        "synthetic landscape is not discriminating")
    assert per_symbol_on_eth > pooled_on_eth, (
        f"per-symbol specialisation did not beat pooling on the ETH-only signal: "
        f"{per_symbol_on_eth} vs {pooled_on_eth}")
    assert per_symbol_on_eth - pooled_on_eth > 0.05
    # The pooled arm really did evaluate the whole basket every time.
    assert all(c["symbols"] == basket for c in calls)


# ══════════════════════════════════════════════════════════════════════
# (e) checkpoint semantics across the two shapes
# ══════════════════════════════════════════════════════════════════════

def test_a_checkpoint_cannot_be_resumed_as_the_other_symbol_mode(tmp_path):
    from core.ga.evolver import CheckpointSymbolModeMismatchError

    root = tmp_path / "mode_switch"
    basket = ["BTCUSDT", "ETHUSDT"]
    _stub_run(root, basket, population=3, generations=2, keep_checkpoint=True)
    with pytest.raises(CheckpointSymbolModeMismatchError) as excinfo:
        _stub_run(root, basket, symbol_mode="per_symbol", population=3,
                  generations=2, resume=True, keep_checkpoint=True)
    assert "pooled" in str(excinfo.value)
    assert "per_symbol" in str(excinfo.value)


def test_a_per_symbol_checkpoint_only_resumes_its_own_arm(tmp_path):
    basket = ["BTCUSDT", "ETHUSDT"]
    root = tmp_path / "arm_resume"
    first, _, _, _ = _stub_run(root, basket, symbol_mode="per_symbol",
                               population=3, generations=2, keep_checkpoint=True)
    # The checkpoint left behind belongs to the LAST arm (ETHUSDT).
    import pickle

    with open(root / "data" / "ga_checkpoint.pkl", "rb") as handle:
        state = pickle.load(handle)
    assert state["symbol_mode"] == "per_symbol"
    assert state["arm_symbol"] == "ETHUSDT"

    second, _, _, _ = _stub_run(root, basket, symbol_mode="per_symbol",
                                population=3, generations=2, resume=True,
                                keep_checkpoint=True)
    assert second["champions"][0]["resumed_from_generation"] is None
    assert second["champions"][1]["resumed_from_generation"] == 2
    assert first["champions"][1]["provenance"]["checkpoint"][
        "resumed_symbol_mode"] is None


# ══════════════════════════════════════════════════════════════════════
# (f) default-off identity
# ══════════════════════════════════════════════════════════════════════

_VOLATILE_KEYS = {"written_at", "elapsed", "elapsed_seconds", "saved_at",
                  "champion_name", "path", "name",
                  # a log line that embeds the (per-run) checkpoint path:
                  "checkpoint_note",
                  # the documented ADDITIVE audit key (P7-S2, fitness.py):
                  "symbols_evaluated"}


def _scrub(value):
    """Drop timing/name noise and the additive audit keys, recursively."""
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()
                if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _run_payload(result, calls, population_hash, history=None) -> bytes:
    payload = {
        "population_hash": population_hash,
        "calls": _scrub(calls),
        "history": _scrub(history if history is not None else result["history"]),
        "run": _scrub(result),
    }
    return json.dumps(payload, sort_keys=True, indent=1).encode("utf-8")


def test_absent_and_pooled_are_byte_identical_with_a_stub_scorer(tmp_path):
    basket = ["BTCUSDT", "ETHUSDT"]
    absent, calls_a, _, ev_a = _stub_run(tmp_path / "id_absent", basket,
                                         population=4, generations=3,
                                         prior_ledger=1440)
    pooled, calls_p, _, ev_p = _stub_run(tmp_path / "id_pooled", basket,
                                         symbol_mode="pooled", population=4,
                                         generations=3, prior_ledger=1440)
    blob_a = _run_payload(absent, calls_a, ev_a.population_hash())
    blob_p = _run_payload(pooled, calls_p, ev_p.population_hash())
    assert blob_a == blob_p, "an explicit 'pooled' changed the run"
    digest = hashlib.sha256(blob_a).hexdigest()
    assert digest == hashlib.sha256(blob_p).hexdigest()
    # The comparison is meaningful: it covered the trial ledger and the champion.
    assert [c["prior_trials"] for c in calls_a] == [1440, 1444, 1448]
    assert absent["provenance"]["n_trials"] == 1452
    assert absent["fitness"] == 10.0


# ── the same identity, on the REAL batch scorer ────────────────────────

def _write_synthetic_market(root: Path) -> None:
    """Deterministic OHLCV parquet tree (identical for both arms)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20261001)
    base = pd.date_range("2026-01-01", periods=60 * 96, freq="15min")
    for symbol in ("BTCUSDT", "ETHUSDT"):
        close = 20000 + np.cumsum(rng.normal(0, 40, len(base)))
        m15 = pd.DataFrame({
            "open": close, "high": close + 30, "low": close - 30,
            "close": close, "volume": rng.random(len(base)) * 100 + 10,
        }, index=base)
        market = root / "market" / symbol
        market.mkdir(parents=True, exist_ok=True)
        for tf in ("15m", "1h", "4h"):
            frame = m15 if tf == "15m" else m15.resample(tf).agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"}).dropna()
            frame.to_parquet(market / f"{tf}.parquet")


def _real_scorer_run(root: Path, symbol_mode=ABSENT, seed=20260101,
                     population=3, generations=2):
    """One real-batch GA run over synthetic data, with the scorer observed."""
    import core.ga.fitness as fitness_mod
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.risk.manager import RiskManager
    from core.strategy.loader import StrategyLoader

    root.mkdir(parents=True, exist_ok=True)
    _write_synthetic_market(root / "data")

    Config._instance = None
    try:
        config = Config.load("sim")
        config.data_dir = str(root / "data")
        config.backtest_engine_mode = "legacy"
        config.backtest_ml_enabled = False
        config.backtest_live_spread_enabled = False
        # Pin every knob the run consumes, so a config.yaml edit by anyone else
        # cannot be mistaken for a P7-S2 regression.
        config.ga_alpha_weight = 1.0
        config.ga_benchmark_mode = "buy_hold"
        config.ga_min_champion_trades = 30
        config.ga_regime_conditioning = False
        bus = EventBus()
        engine = BacktestEngine(config, None, RiskManager(config, bus),
                                OrderExecutor(config, bus))
    finally:
        Config._instance = None

    loader = StrategyLoader(str(root / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)

    calls: list[dict] = []
    real = fitness_mod.evaluate_population_batch

    def _observed(population_arg, symbols_arg, date_start, date_end, *args,
                  **kwargs):
        calls.append({"symbols": list(symbols_arg),
                      "batch_trials": kwargs.get("batch_trials"),
                      "prior_trials": kwargs.get("prior_trials")})
        return real(population_arg, symbols_arg, date_start, date_end, *args,
                    **kwargs)

    kwargs = dict(population_size=population, generations=generations,
                  elite_count=2, immigrant_count=1, max_workers=1, seed=seed,
                  keep_checkpoint=False,
                  # Both synthetic trees hold 15m/1h/4h only: confine the
                  # timeframe gene so every genome is actually evaluable.
                  timeframe_pool=["1h"])
    if symbol_mode is not ABSENT:
        kwargs["symbol_mode"] = symbol_mode

    fitness_mod.evaluate_population_batch = _observed
    try:
        evolver = GAStrategyEvolver(engine, loader, GARunConfig(**kwargs))
        result = evolver.evolve(["BTCUSDT", "ETHUSDT"], "2026-02-01",
                                "2026-02-20", seed=seed,
                                window_key="2026-02-01~2026-02-20")
    finally:
        fitness_mod.evaluate_population_batch = real
    return result, calls, evolver


def test_absent_and_pooled_are_byte_identical_on_the_real_scorer(tmp_path):
    """The real backtest path: the field's presence changes nothing at all."""
    absent, calls_a, ev_a = _real_scorer_run(tmp_path / "real_absent")
    pooled, calls_p, ev_p = _real_scorer_run(tmp_path / "real_pooled",
                                             symbol_mode="pooled")
    assert absent.get("error") is None, absent
    assert absent["trade_count"] and absent["trade_count"] > 0, (
        "the identity run scored no trades — it proves nothing")
    assert [c["symbols"] for c in calls_a] == [["BTCUSDT", "ETHUSDT"]] * 2
    assert calls_a == calls_p
    blob_a = _run_payload(absent, calls_a, ev_a.population_hash())
    blob_p = _run_payload(pooled, calls_p, ev_p.population_hash())
    assert blob_a == blob_p, "an explicit 'pooled' changed the real run"
    assert hashlib.sha256(blob_a).hexdigest() == \
        hashlib.sha256(blob_p).hexdigest()


_IDENTITY_HARNESS = r'''"""Identity harness: a stubbed GA run, serialised canonically."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

TREE = Path.cwd()
sys.path.insert(0, str(TREE))

MODE = sys.argv[1]                 # "absent" | "pooled" | "per_symbol"
OUT = Path(sys.argv[2])
TMP = Path(sys.argv[3])

from core.ga.evolver import GAStrategyEvolver, GARunConfig
from core.strategy.loader import StrategyLoader
import core.ga.fitness as fitness_mod

POP, GENS = 5, 3
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SEED = 20261009
LEDGER = 1440
VOLATILE = {"written_at", "elapsed", "elapsed_seconds", "saved_at",
            "champion_name", "path", "name", "checkpoint_note",
            "symbols_evaluated",
            # P9: the champion provenance `eval` block gained `fill_convention`.
            # It is not part of the S2 contract, the baseline tree cannot have it,
            # and it is constant ("close") in every tree here — so it is scrubbed
            # from both sides rather than compared.
            "fill_convention"}


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


class _Cfg:
    data_dir = str(TMP)
    ga_alpha_weight = 1.0
    ga_min_champion_trades = 30
    ga_benchmark_mode = "buy_hold"
    ga_regime_conditioning = False


class _Engine:
    config = _Cfg()


TMP.mkdir(parents=True, exist_ok=True)
(TMP / "data").mkdir(parents=True, exist_ok=True)
(TMP / "data" / "ga_trials.json").write_text(json.dumps({"trials": LEDGER}),
                                             encoding="utf-8")
loader = StrategyLoader(str(TMP / "strategies"))
loader.strategies_dir.mkdir(parents=True, exist_ok=True)

calls = []


def _stub(population, symbols, date_start, date_end, engine, loader_, **kwargs):
    calls.append({"symbols": list(symbols),
                  "batch_trials": kwargs.get("batch_trials"),
                  "prior_trials": kwargs.get("prior_trials")})
    out = []
    for i, chrom in enumerate(population):
        chrom = dict(chrom)
        chrom["fitness_result"] = {
            "fitness": 10.0 - i, "trade_count": 40, "profit_factor": 1.6,
            "dsr": 0.5, "total_return": 12.0, "sharpe": 2.0, "win_rate": 55.0,
            "max_dd": 1.0, "buy_hold_pct": 3.0, "alpha_vs_buy_hold_pct": 9.0,
            "dsr_detail": {"dsr": 0.5, "n_trials": 0}}
        out.append(chrom)
    return out


kwargs = dict(population_size=POP, generations=GENS, elite_count=1,
              immigrant_count=1, max_workers=1, seed=SEED,
              keep_checkpoint=False)
if MODE != "absent":
    kwargs["symbol_mode"] = MODE

fitness_mod.evaluate_population_batch = _stub
evolver = GAStrategyEvolver(_Engine(), loader, GARunConfig(**kwargs))
run = evolver.evolve(list(SYMBOLS), "2026-01-01", "2026-02-01", seed=SEED,
                     window_key="2026-01-01~2026-02-01")

payload = {
    "population_hash": evolver.population_hash(),
    "calls": scrub(calls),
    "run": scrub(run),
}
blob = json.dumps(payload, sort_keys=True, indent=1).encode("utf-8")
OUT.write_bytes(blob)
print(hashlib.sha256(blob).hexdigest())
'''


def _run_identity_harness(tree: Path, tmp_path: Path, name: str,
                          mode: str) -> tuple[bytes, str]:
    """Run the identity harness with *tree* as the import root."""
    import os

    harness = tmp_path / "symbol_mode_identity_harness.py"
    harness.write_text(_IDENTITY_HARNESS, encoding="utf-8", newline="\n")
    out = tmp_path / f"{name}.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    proc = subprocess.run(
        [sys.executable, str(harness), mode, str(out),
         str(tmp_path / f"data_{name}")],
        cwd=str(tree), capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, (
        f"identity harness failed in {tree} [{mode}]:\n{proc.stderr}")
    return out.read_bytes(), proc.stdout.strip()


#: The harness's own constants (they live inside the harness string; repeated
#: here so the test can assert the comparison really covered these seams).
_HARNESS_POP, _HARNESS_GENS, _HARNESS_LEDGER = 5, 3, 1440


def test_pooled_is_byte_identical_to_the_head_worktree(tmp_path):
    """A ``git worktree`` at the pre-S2 revision runs the same harness.

    ``absent`` (no field at all) and an explicit ``"pooled"`` must produce the
    same serialised population hash, trial ledger and result payload as the
    baseline tree — byte for byte.
    """
    worktree = tmp_path / "head_tree"
    add = subprocess.run(["git", "worktree", "add", "--detach", str(worktree),
                          BASELINE_REVISION],
                         cwd=str(ROOT), capture_output=True, text=True,
                         timeout=600)
    if add.returncode != 0:
        pytest.skip(f"cannot create a HEAD worktree at {BASELINE_REVISION}: "
                    f"{add.stderr.strip()}")
    try:
        head_bytes, head_digest = _run_identity_harness(
            worktree, tmp_path, "head", "absent")
        for variant in ("absent", "pooled"):
            tree_bytes, tree_digest = _run_identity_harness(
                ROOT, tmp_path, f"tree_{variant}", variant)
            assert tree_digest == head_digest, (
                f"symbol_mode={variant} changed the run:\n"
                f"  HEAD {BASELINE_REVISION}: {head_digest}\n"
                f"  working tree: {tree_digest}")
            assert tree_bytes == head_bytes, (
                f"the harness output differs byte-for-byte for {variant}")

        payload = json.loads(head_bytes.decode("utf-8"))
        # The comparison only means something if it really covered the seams.
        assert payload["calls"] == [
            {"symbols": ["BTCUSDT", "ETHUSDT"],
             "batch_trials": _HARNESS_POP,
             "prior_trials": _HARNESS_LEDGER + _HARNESS_POP * i}
            for i in range(_HARNESS_GENS)]
        assert payload["run"]["provenance"]["n_trials"] == \
            _HARNESS_LEDGER + _HARNESS_POP * _HARNESS_GENS
        assert payload["run"]["fitness"] == 10.0
        assert hashlib.sha256(head_bytes).hexdigest() == head_digest
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                       cwd=str(ROOT), capture_output=True, text=True,
                       timeout=600)
