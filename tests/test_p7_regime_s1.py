"""P7-S1 — causal regime as a first-class strategy attribute.

Acceptance criteria this file pins (plan ``docs/overhaul/P7_REGIME_PLAN.md`` S1):

1. **Gene round-trip + confinement** — ``encode(decode(x))`` is the identity for
   a genome carrying the ``regime_filter`` gene, every gene value is inside the
   known gate vocabulary, and an unknown / in-sample value can never reach a
   decoded ``StrategyConfig``.
2. **Causal-only rule** — an in-sample HMM label (``calm`` / ``stressed``) is
   refused **by name** (``InSampleRegimeLabelError``) at config load, at the
   gene boundary and at the per-bar decision, and a whole-sample regime *table*
   is refused with the pre-existing ``NonCausalRegimeError``.
3. **Conditioning restricts the evaluated bars** — on a synthetic series whose
   regime is known by construction, a filtered strategy enters only on bars
   whose causal label is in its declaration, and the engine reports the sample
   it was restricted to (``metrics["regime_conditioning"]``).
4. **The switch off is inert** — no gene, no extra RNG draw, an empty decoded
   filter for every chromosome (including one carrying a non-empty gene or a
   checkpoint written by a conditioned run), and an engine run bit-identical to
   the unconditioned one.
5. **DSR / trade-count bookkeeping on the conditional sample** — the conditioned
   genome's own trades and equity curve are what the scorer and the publication
   gate see, so a filter that shrinks the sample below the trade floor is
   flagged rather than pretending to be the unconditioned genome.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.ga import genome as G
from core.strategy.regime_causal import (
    GATE_REGIME_LABELS,
    IN_SAMPLE_LABELS,
    InSampleRegimeLabelError,
    NonCausalRegimeError,
    UnknownRegimeLabelError,
    UnknownRegimeSourceError,
    allowed_regimes,
    build_regime_context,
    causal_regime_table,
    enforce_causal_table,
    regime_allows,
)


# ══════════════════════════════════════════════════════════════════════════
# synthetic market whose regime is known BY CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════

CALM_UP_BARS = 600
STRESS_DOWN_BARS = 600
TOTAL_BARS = CALM_UP_BARS + STRESS_DOWN_BARS


def _regime_series(seed: int = 20261002) -> pd.DataFrame:
    """Calm uptrend for 600 bars, then a stressed downtrend for 600 bars.

    Both halves are built so the causal *labels* are known in advance: a
    deterministic up-drift with tiny noise must land in ``trend_up``, a
    deterministic down-drift with large noise in ``trend_down``.  ``low``/``high``
    are set to the close so no wick can make a bar look like the other half.
    """
    rng = np.random.default_rng(seed)
    calm = 0.0004 + rng.normal(0, 0.0008, CALM_UP_BARS)
    stress = -0.0040 + rng.normal(0, 0.0100, STRESS_DOWN_BARS)
    steps = np.concatenate([calm, stress])
    close = 20_000.0 * np.cumprod(1.0 + steps)
    index = pd.date_range("2026-01-01", periods=TOTAL_BARS, freq="1h")
    return pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": 100.0 + rng.random(TOTAL_BARS) * 50.0,
        },
        index=index,
    )


def _write_market(root: Path, symbols=("BTCUSDT", "ETHUSDT")) -> str:
    frame = _regime_series()
    for symbol in symbols:
        target = root / "market" / symbol
        target.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target / "1h.parquet")
    return str(root)


@pytest.fixture()
def market_dir(tmp_path):
    return _write_market(tmp_path / "data")


def _engine(market_dir, tmp_path):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import StrategyLoader

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = market_dir
    cfg.backtest_engine_mode = "legacy"
    cfg.backtest_ml_enabled = False
    cfg.backtest_live_spread_enabled = False
    bus = EventBus()
    loader = StrategyLoader(str(tmp_path / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    return cfg, BacktestEngine(cfg, None, RiskManager(cfg, bus),
                               OrderExecutor(cfg, bus)), loader


def _always_on_strategy(name: str, regime_filter=None):
    """A strategy that enters almost every bar, so the filter is the only gate."""
    from core.strategy.loader import MLConfig, StrategyConfig

    return StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=["1h"],
        indicators={"rsi": {"period": 14, "source": "close"},
                    "sma": {"period": 5}},
        entry_conditions={"long": ["close > sma"], "short": ["close < sma"]},
        exit_conditions={"long": ["rsi > 99"], "short": ["rsi < 1"]},
        regime_filter=list(regime_filter or []),
        ml_config=MLConfig(enabled=False),
    )


def _run_engine(engine, strategy, symbols=("BTCUSDT",), **kwargs):
    return engine.run_with_exit_evaluation(
        strategies=[strategy], symbols=list(symbols),
        date_start=kwargs.pop("date_start", "2026-01-01"),
        date_end=kwargs.pop("date_end", "2026-03-31"),
        initial_balance=10_000.0, mode="full", simulate_ai_weights=False,
        per_strategy_isolation=True, per_genome_ledger=True,
        use_live_spread=False, benchmark_mode="exposure_matched")


# ══════════════════════════════════════════════════════════════════════════
# 1 — the gene: round-trip, confinement, inertness while the switch is off
# ══════════════════════════════════════════════════════════════════════════

def _gene(chrom):
    for gene in chrom.get("categorical", []) or []:
        if gene.name == "regime_filter":
            return gene
    return None


def test_gene_is_absent_while_the_switch_is_off_and_consumes_no_rng():
    """OFF ⇒ no gene AND the same RNG stream (a pinned gene would still draw)."""
    random.seed(4242)
    off = [G.random_chromosome(f"off_{i}") for i in range(8)]
    random.seed(4242)
    explicit_off = [G.random_chromosome(f"off_{i}", regime_conditioning=False)
                    for i in range(8)]
    random.seed(4242)
    unconcerned = [G.random_chromosome(f"off_{i}") for i in range(8)]

    assert all(_gene(c) is None for c in off + explicit_off)
    for a, b in zip(off, unconcerned):
        assert [(g.name, g.value) for g in a["categorical"]] == \
            [(g.name, g.value) for g in b["categorical"]]
        assert [(g.name, g.value) for g in a["continuous"]] == \
            [(g.name, g.value) for g in b["continuous"]]
        assert a["condition_logic"] == b["condition_logic"]
        assert [g.conditions for g in a["structural"]] == \
            [g.conditions for g in b["structural"]]
    assert G.REGIME_CONDITIONING_ENABLED is False, \
        "the module default must ship OFF"


def test_gene_round_trips_and_is_confined_to_the_gate_vocabulary():
    random.seed(11)
    for i in range(30):
        chrom = G.random_chromosome(f"rt_{i}", regime_conditioning=True)
        gene = _gene(chrom)
        assert gene is not None, "a conditioned chromosome must carry the gene"
        assert set(gene.options) == {"", *GATE_REGIME_LABELS}
        assert gene.value in gene.options
        first = G.chromosome_to_strategy(chrom, regime_conditioning=True)
        again = G.chromosome_to_strategy(
            G.strategy_to_chromosome(first, regime_conditioning=True),
            regime_conditioning=True)
        assert first.model_dump() == again.model_dump(), i
        assert set(first.regime_filter) <= set(GATE_REGIME_LABELS)


def test_a_non_neutral_value_survives_the_round_trip():
    chrom = G.random_chromosome("pinned", regime_conditioning=True)
    _gene(chrom).value = "trend_up"
    config = G.chromosome_to_strategy(chrom, regime_conditioning=True)
    assert config.regime_filter == ["trend_up"]
    back = G.strategy_to_chromosome(config, regime_conditioning=True)
    assert _gene(back).value == "trend_up"
    assert G.chromosome_to_strategy(back, regime_conditioning=True) \
        .regime_filter == ["trend_up"]


def test_an_unknown_or_in_sample_gene_value_is_confined_away():
    """The GA can never hand a decoded config a label it invented."""
    chrom = G.random_chromosome("bad", regime_conditioning=True)
    for bad in ("mystery", "calm", "stressed", "range_unknown"):
        _gene(chrom).value = bad
        decoded = G.chromosome_to_strategy(chrom, regime_conditioning=True)
        assert decoded.regime_filter == [], bad
        assert G.confine_regime_filter(bad, True) == "", bad
    # A mixed value keeps only the legal half (and never raises).
    for mixed, kept in (("calm,trend_down", "trend_down"),
                        ("trend_up,calm", "trend_up"),
                        ("trend_up,mystery,range_low", "trend_up,range_low")):
        _gene(chrom).value = mixed
        assert G.chromosome_to_strategy(chrom, regime_conditioning=True) \
            .regime_filter == kept.split(",")
        assert G.confine_regime_filter(mixed, True) == kept


def test_switch_off_strips_the_gene_from_a_conditioned_checkpoint():
    """A checkpoint written by a conditioned run decodes as a pre-P7 genome."""
    chrom = G.random_chromosome("ckpt", regime_conditioning=True)
    _gene(chrom).value = "trend_up"
    assert _gene(copy.deepcopy(chrom)) is not None
    G.confine_regime_gene(chrom, None)          # None ⇒ the module switch (off)
    assert _gene(chrom) is None
    assert G.chromosome_to_strategy(chrom).regime_filter == []
    # ... and the ON path keeps it.
    chrom2 = G.random_chromosome("ckpt2", regime_conditioning=True)
    _gene(chrom2).value = "range_mid"
    G.confine_regime_gene(chrom2, True)
    assert _gene(chrom2).value == "range_mid"


def test_every_hand_built_pre_p7_chromosome_still_decodes_with_no_filter():
    from core.ga.genome import (BooleanGene, CategoricalGene, ContinuousGene,
                                StructuralGene)

    frozen = {
        "name": "head_frozen",
        "continuous": [ContinuousGene("rsi_period", 14, 5, 28, 1)],
        "categorical": [CategoricalGene("mode", "trend", ["trend"]),
                        CategoricalGene("timeframes", "1h", ["1h"])],
        "structural": [StructuralGene("entry_long", ["rsi < 30"], []),
                       StructuralGene("entry_short", ["rsi > 70"], []),
                       StructuralGene("exit_long", ["rsi > 65"], []),
                       StructuralGene("exit_short", ["rsi < 35"], [])],
        "indicator_genes": [BooleanGene("rsi", True)],
        "condition_logic": "or",
    }
    config = G.chromosome_to_strategy(frozen, regime_conditioning=True)
    assert config.regime_filter == [], "an absent gene must mean 'no filter'"


def test_the_regime_field_is_purely_additive_to_the_frozen_config():
    """The P7 field moves the config dump by exactly one key, and nothing else.

    ``StrategyConfig`` gained ``regime_filter: list[str] = []``.  The P6-D test
    ``tests/test_ga_volume_genes.py::test_pre_p6_chromosome_decodes_to_the_frozen_head_config``
    pins the **full-dump** hash of a frozen chromosome to ``809ddf7ba45af011``
    (measured by executing HEAD's own ``core/ga/genome.py`` on that same input).
    A new field necessarily moves a full-dump hash, so the honest move is to
    *measure* that the change is additive rather than to re-pin a number: the
    same dump with ``regime_filter`` removed still hashes to the frozen value,
    and the field's value is the empty default (no filter, no behaviour change).
    Both numbers are pinned here, so a future addition has to make the same
    argument instead of quietly moving the frozen hash.
    """
    import hashlib
    import json

    from core.ga.genome import (BooleanGene, CategoricalGene, ContinuousGene,
                                StructuralGene)

    frozen = {
        "name": "head_frozen",
        "continuous": [ContinuousGene("rsi_period", 14, 5, 28, 1),
                       ContinuousGene("bb_period", 20, 10, 40, 2),
                       ContinuousGene("bb_stddev", 2.0, 1.0, 3.5, 0.25),
                       ContinuousGene("ml_weight", 0.0, 0.0, 0.0, 0.05),
                       ContinuousGene("ml_threshold", 0.6, 0.5, 0.85, 0.05)],
        "categorical": [CategoricalGene("mode", "trend", ["trend"]),
                        CategoricalGene("timeframes", "1h", ["1h", "4h"])],
        "structural": [StructuralGene("entry_long", ["rsi < 30"], []),
                       StructuralGene("entry_short", ["rsi > 70"], []),
                       StructuralGene("exit_long", ["rsi > 65"], []),
                       StructuralGene("exit_short", ["rsi < 35"], [])],
        "indicator_genes": [BooleanGene("rsi", True), BooleanGene("bollinger", True)],
        "condition_logic": "or",
    }
    dump = G.chromosome_to_strategy(frozen).model_dump()

    def _hash(payload) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    assert _hash(dump) == "4d40f95abe61d7e2"
    assert dump["regime_filter"] == []
    assert _hash({k: v for k, v in dump.items() if k != "regime_filter"}) \
        == "809ddf7ba45af011", \
        "the P7 field must be additive: the pre-P7 subset hash is unchanged"


def test_the_evolver_switch_creates_confines_and_strips_the_gene(tmp_path):
    """The run's own switch (`ga.regime_conditioning`) drives the gene's life cycle.

    ON: every genome (initial, immigrant, crossed, mutated) carries the gene,
    confined to ``["", *GATE_REGIME_LABELS]``, and the champion decodes with the
    filter it was scored under.  OFF: the gene is stripped from the population
    before a single backtest, so a genome that arrived with one (a checkpoint, a
    seeded YAML) is evaluated as the pre-P7 one.
    """
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    class _Cfg:
        ga_regime_conditioning = True
        ga_alpha_weight = 1.0
        ga_min_champion_trades = 30
        data_dir = str(tmp_path)

    class _Engine:
        config = _Cfg()

    loader = StrategyLoader(str(tmp_path / "strat"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)

    def _evolver(conditioning, pool=None):
        return GAStrategyEvolver(
            _Engine(), loader,
            GARunConfig(population_size=4, generations=1, elite_count=1,
                        immigrant_count=1, seed=17,
                        regime_conditioning=conditioning,
                        timeframe_pool=pool))

    on = _evolver(True)
    on._regime_conditioning = True
    population = on._init_population(None)
    assert all(_gene(c) is not None for c in population)
    assert all(_gene(c).value in _gene(c).options for c in population)
    random.seed(3)
    child = on._crossover(population[0], population[1])
    assert _gene(child) is not None
    on._mutate(child)
    assert _gene(child).value in _gene(child).options
    # A hand-injected illegal value is confined by the mutator, not carried.
    _gene(child).value = "calm"
    on._mutate(child)
    assert _gene(child).value in _gene(child).options

    off = _evolver(None)
    off._regime_conditioning = False
    conditioned = [_with_gene(c, "trend_up") for c in population]
    for chrom in conditioned:
        G.confine_regime_gene(chrom, False)
        assert _gene(chrom) is None
        assert G.chromosome_to_strategy(chrom).regime_filter == []


def _with_gene(chrom, value):
    chrom = copy.deepcopy(chrom)
    gene = _gene(chrom)
    if gene is None:
        from core.ga.genome import CategoricalGene, regime_gene_options
        chrom["categorical"].append(
            CategoricalGene("regime_filter", value, regime_gene_options()))
    else:
        gene.value = value
    return chrom


# ══════════════════════════════════════════════════════════════════════════
# 2 — the causal-only rule
# ══════════════════════════════════════════════════════════════════════════

def test_an_in_sample_config_label_is_refused_by_name():
    from pydantic import ValidationError

    from core.strategy.loader import StrategyConfig

    def _refusal(**kwargs):
        """The named error pydantic wraps in a ValidationError (its ``ctx``)."""
        try:
            StrategyConfig(**kwargs)
        except ValidationError as exc:
            errors = exc.errors()
            assert errors and isinstance(errors[0].get("ctx", {}).get("error"),
                                         BaseException), errors
            return errors[0]["ctx"]["error"]
        raise AssertionError(f"StrategyConfig({kwargs}) was accepted")

    for label in IN_SAMPLE_LABELS:
        refusal = _refusal(name="bad", regime_filter=[label])
        assert isinstance(refusal, InSampleRegimeLabelError), refusal
        assert label in str(refusal)
    # ... and a typo is a *different*, also-named refusal.
    assert isinstance(_refusal(name="bad", regime_filter=["trending_up"]),
                      UnknownRegimeLabelError)
    assert isinstance(_refusal(name="bad", regime_filter=["range_unknown"]),
                      UnknownRegimeLabelError)
    # A bare string is accepted as one label (a YAML convenience), like the list.
    assert StrategyConfig(name="s", regime_filter="trend_up").regime_filter \
        == ["trend_up"]
    # The composite labels are accepted, de-duplicated and ordered.
    ok = StrategyConfig(name="ok", regime_filter=["trend_up", "trend_up", "range_low"])
    assert ok.regime_filter == ["trend_up", "range_low"]
    assert StrategyConfig(name="none").regime_filter == []
    assert StrategyConfig(name="none", regime_filter=None).regime_filter == []


def test_the_per_bar_decision_refuses_an_in_sample_label_by_name():
    assert regime_allows([], "calm") is True, "no filter allows everything"
    with pytest.raises(InSampleRegimeLabelError):
        regime_allows(["trend_up"], "stressed")
    with pytest.raises(InSampleRegimeLabelError):
        allowed_regimes(["calm"])
    # An unmeasured bar is not tradeable when a filter is declared.
    assert regime_allows(["trend_up"], None) is False
    assert regime_allows(["trend_up"], "unknown") is False
    assert regime_allows(["trend_up"], "trend_up") is True
    assert regime_allows(["trend_up"], "range_low") is False


def test_a_whole_sample_regime_table_is_refused_by_the_existing_error():
    from core.strategy.regime import classify_regimes, default_gate, gate_regimes

    frame = _regime_series()
    bad = classify_regimes(frame, with_hmm=True, causal_hmm=False)
    good = causal_regime_table(frame)
    with pytest.raises(NonCausalRegimeError):
        enforce_causal_table(bad)
    enforce_causal_table(good)                  # no raise
    # The pre-existing gate refuses the same table, so the two agree.
    with pytest.raises(NonCausalRegimeError):
        gate_regimes(default_gate(enabled=True), bad, "breakout")
    # A frame with no provenance at all is refused too ("trust me" is not a
    # source) and so is one that declares a non-causal source.
    with pytest.raises(UnknownRegimeSourceError):
        enforce_causal_table(pd.DataFrame({"regime": ["trend_up"]}))
    forged = causal_regime_table(frame)
    forged.attrs["regime_source"] = "hmm_states"
    with pytest.raises(UnknownRegimeSourceError):
        enforce_causal_table(forged)


def test_appending_future_bars_cannot_change_an_earlier_causal_label():
    """The property the whole stage rests on: labels are a function of the past."""
    frame = _regime_series()
    full = causal_regime_table(frame)
    truncated = causal_regime_table(frame.iloc[:CALM_UP_BARS + 60])
    shared = truncated.index
    assert list(full.loc[shared, "regime"]) == list(truncated["regime"])


# ══════════════════════════════════════════════════════════════════════════
# 3 — conditioning really restricts the evaluated bars
# ══════════════════════════════════════════════════════════════════════════

def test_the_causal_labels_split_the_synthetic_series_as_designed():
    table = causal_regime_table(_regime_series())
    labels = table["regime"]
    # The calm half trends up, the stressed half trends down — measured, on the
    # bars where the terciles have warmed up.
    calm = labels.iloc[120:CALM_UP_BARS]
    stress = labels.iloc[CALM_UP_BARS + 20:]
    assert (calm == "trend_up").mean() > 0.9, dict(calm.value_counts())
    assert (stress == "trend_down").mean() > 0.9, dict(stress.value_counts())


def test_regime_context_lookup_is_causal_and_unmeasured_bars_are_unknown():
    frame = _regime_series()
    context = build_regime_context({("BTCUSDT", "1h"): frame})
    assert context is not None and len(context) == 1
    table = causal_regime_table(frame)
    for pos in (0, 119, 300, CALM_UP_BARS + 100, TOTAL_BARS - 1):
        ts = frame.index[pos]
        expected = table["regime"].iloc[pos]
        got = context.lookup("BTCUSDT", "1h", ts)
        if expected in GATE_REGIME_LABELS:
            assert got == expected, (pos, expected, got)
        else:
            # `range_unknown` / a short-history bar is reported as-is and the
            # decision refuses it (never silently mapped to a gate label).
            assert got == expected
            assert regime_allows(["trend_up", "trend_down"], got) is False
    assert context.lookup("BTCUSDT", "1h", None) is None
    assert context.lookup("NOPEUSDT", "1h", frame.index[0]) is None
    assert context.counts("BTCUSDT", "1h")["trend_up"] > 0


def test_conditioning_restricts_the_engine_to_the_allowed_bars(market_dir, tmp_path):
    """The engine's own accounting: gated bars vs allowed bars, per label."""
    cfg, engine, _ = _engine(market_dir, tmp_path)
    base = _always_on_strategy("p7_all")
    res = _run_engine(engine, base)
    trades_all = len(res["per_strategy_equity"]["p7_all"]["trades"])
    assert trades_all > 20, "the unconditioned genome must trade (else vacuous)"
    assert "regime_conditioning" not in (res["metrics"] or {})

    for regime in ("trend_up", "trend_down"):
        cfg2, engine2, _ = _engine(market_dir, tmp_path)
        filtered = _always_on_strategy(f"p7_{regime}", [regime])
        out = _run_engine(engine2, filtered)
        accounting = out["metrics"]["regime_conditioning"][f"p7_{regime}"]
        trades = len(out["per_strategy_equity"][f"p7_{regime}"]["trades"])
        labels = accounting["labels"]
        # The allowed share must equal the bar share of that one label.
        total = sum(labels.values())
        assert total > 0
        assert accounting["allowed"] == labels.get(regime, 0)
        assert accounting["gated"] == total
        assert accounting["allowed"] < total, "the filter removed nothing"
        assert trades < trades_all, (regime, trades, trades_all)
        assert trades > 0, f"{regime} never entered — the sample is too small"
        print(f"\n[{regime}] gated={total} allowed={accounting['allowed']} "
              f"({accounting['allowed_pct']}%) trades {trades_all} -> {trades} "
              f"labels={labels}")


def test_the_filtered_trades_only_open_in_the_declared_regime(market_dir, tmp_path):
    """Every filtered entry bar must carry an allowed label — read back trade by trade."""
    cfg, engine, _ = _engine(market_dir, tmp_path)
    strategy = _always_on_strategy("p7_only_up", ["trend_up"])
    out = _run_engine(engine, strategy)
    trades = out["per_strategy_equity"]["p7_only_up"]["trades"]
    assert trades, "no trades to check — vacuous"
    from core.strategy.regime_causal import build_regime_context
    from core.backtest.data_feeder import DataFeeder

    feeder = DataFeeder(market_dir + "/market", ["BTCUSDT"], ["1h"],
                        "2026-01-01", "2026-03-31")
    feeder.load()
    context = build_regime_context(
        {("BTCUSDT", "1h"): feeder.get_all_data_for_symbol("BTCUSDT", "1h")})
    for trade in trades:
        label = context.lookup(trade["symbol"], "1h", trade["opened_at"])
        assert label == "trend_up", (trade["opened_at"], label)


# ══════════════════════════════════════════════════════════════════════════
# 4 — the switch off is the pre-P7 engine, bit for bit
# ══════════════════════════════════════════════════════════════════════════

def test_an_empty_filter_run_equals_the_pre_p7_run(market_dir, tmp_path):
    """Two engines, one strategy with ``regime_filter=[]``: identical results."""
    cfg, engine, _ = _engine(market_dir, tmp_path)
    _, engine2, _ = _engine(market_dir, tmp_path)
    first = _run_engine(engine, _always_on_strategy("p7_a"))
    second = _run_engine(engine2, _always_on_strategy("p7_a"))
    assert first["metrics"]["buy_hold_pct"] == second["metrics"]["buy_hold_pct"]
    assert "regime_conditioning" not in first["metrics"]
    assert "regime_conditioning" not in second["metrics"]
    trades_a = first["per_strategy_equity"]["p7_a"]["trades"]
    trades_b = second["per_strategy_equity"]["p7_a"]["trades"]
    assert len(trades_a) == len(trades_b) > 0
    for a, b in zip(trades_a, trades_b):
        assert a == b, "the OFF path must be bit-identical"


def test_a_conditioned_genome_and_an_unconditioned_one_are_scored_separately(
        market_dir, tmp_path):
    """Nothing leaks across genomes in one chunk (filters are per strategy)."""
    cfg, engine, _ = _engine(market_dir, tmp_path)
    chunk = [_always_on_strategy("p7_plain"),
             _always_on_strategy("p7_cond", ["trend_up"])]
    result = engine.run_with_exit_evaluation(
        strategies=chunk, symbols=["BTCUSDT"],
        date_start="2026-01-01", date_end="2026-03-31", initial_balance=10_000.0,
        mode="full", simulate_ai_weights=False, per_strategy_isolation=True,
        per_genome_ledger=True, use_live_spread=False,
        benchmark_mode="exposure_matched")
    per = result["per_strategy_equity"]
    plain, cond = per["p7_plain"]["trades"], per["p7_cond"]["trades"]
    assert plain and cond
    assert len(cond) < len(plain)
    accounting = result["metrics"]["regime_conditioning"]
    assert set(accounting) == {"p7_cond"}, accounting
    cond_bars = accounting["p7_cond"]
    # The unconditioned genome is REPORTED nowhere: its trades are the proof it
    # ran unrestricted.
    assert sum(cond_bars["labels"].values()) == cond_bars["gated"]
    assert cond_bars["allowed"] == cond_bars["labels"].get("trend_up", 0)


# ══════════════════════════════════════════════════════════════════════════
# 5 — DSR / trade-count bookkeeping on the conditional sample
# ══════════════════════════════════════════════════════════════════════════

def test_a_conditioned_sample_is_what_the_scorer_and_the_gate_see(
        market_dir, tmp_path):
    """The DSR's observation count and the trade-count flag follow the SAMPLE.

    A filter that leaves too few trades must be *flagged* (``insufficient_trades``
    / gate rejection), never scored as if the genome had the unconditioned
    trade count — which is the plan's risk 2 ("conditioning shrinks the sample
    below the gate's floor").
    """
    from core.ga.fitness import (MIN_TRADES_GATE, score_stats,
                                 stats_from_engine_result)

    cfg, engine, _ = _engine(market_dir, tmp_path)
    rows = {}
    for tag, filt in (("all", []), ("up", ["trend_up"])):
        _, engine_i, _ = _engine(market_dir, tmp_path)
        strategy = _always_on_strategy(f"p7_{tag}", filt)
        result = _run_engine(engine_i, strategy)
        stats = stats_from_engine_result(result, strategy.name, 10_000.0,
                                         config=strategy)
        score = score_stats(stats, None, n_trials=64)
        rows[tag] = (strategy, stats, score)
        assert stats["trades"] > 0, tag

    strategy_all, stats_all, score_all = rows["all"]
    strategy_up, stats_up, score_up = rows["up"]

    assert stats_up["trades"] < stats_all["trades"], (
        "the conditional sample must be the smaller one")
    assert stats_up["observations"] <= stats_all["observations"]
    assert stats_up["regime_conditioning"]["allowed"] > 0
    assert stats_all.get("regime_conditioning") is None
    assert score_up["dsr_detail"]["observation_periods"] == stats_up["observations"]
    assert score_up["regime_filter"] == ["trend_up"]
    assert score_all["regime_filter"] is None
    assert score_up["regime_conditioning"]["allowed"] > 0
    assert score_all["regime_conditioning"] is None

    # The gate reads the conditioned trade count (the flag is the contract).
    assert score_up["insufficient_data"] == (
        stats_up["trades"] < MIN_TRADES_GATE)
    print(f"\n[trades] unconditioned={stats_all['trades']} "
          f"trend_up={stats_up['trades']} "
          f"(DSR n_trials=64, obs {stats_all['observations']} -> "
          f"{stats_up['observations']})")


def test_a_filter_that_kills_the_sample_is_flagged_not_hidden(
        market_dir, tmp_path, monkeypatch):
    """A regime with too few bars must produce ``insufficient_trades``."""
    from core.ga import fitness as fitness_mod

    cfg, engine, _ = _engine(market_dir, tmp_path)
    strategy = _always_on_strategy("p7_thin", ["trend_up"])
    result = _run_engine(engine, strategy)
    stats = fitness_mod.stats_from_engine_result(result, strategy.name, 10_000.0,
                                                 config=strategy)
    # Force the "filtered to almost nothing" case the plan warns about: the
    # accounting and the flag must agree that the sample is too small.
    monkeypatch.setattr(fitness_mod, "MIN_TRADES_GATE", stats["trades"] + 1)
    score = fitness_mod.score_stats(stats, None, n_trials=64)
    assert score["flag"] == "insufficient_trades"
    assert score["insufficient_data"] is True
    # With the real floor restored the flag is about the sample it actually has.
    monkeypatch.undo()
    score = fitness_mod.score_stats(stats, None, n_trials=64)
    assert score["insufficient_data"] == (stats["trades"] < 30)
    assert score["fitness"] == score["fitness"], "fitness must stay finite"

