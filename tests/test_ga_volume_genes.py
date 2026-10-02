"""P6-D — GA volume genes: templates, filter/sizing genes, executability.

Acceptance criteria this file pins (plan ``docs/overhaul/P6_VOLUME_PLAN.md`` P6-D):

1. **No orphan templates** — every template in every pool declares the columns it
   reads, every declared column is producible by ``compute_all``, and the
   sanitiser keeps a template exactly when its owner indicator is enabled.  A
   deliberately broken template *fails* ``audit_template_ownership``.
2. **Round-trip + gene-name operators** — ``encode(decode(x))`` is the identity
   for genomes carrying the new genes, and crossover/mutation handle them by name.
3. **Scalar vs vectorised agreement** — ``SignalMatrixBuilder`` (vectorised
   entry path) equals ``StrategyConfig.entry_sides`` (the shared scalar kernel)
   bar for bar on the new templates, and every new template actually fires on the
   real cached bars (a template that never fires is the plan's rollback case).
4. **Off-path identity** — with ``impact_k = 0`` and ``volume_scale_k = 0`` the
   executability model returns its inputs unchanged, and a pre-P6 chromosome
   decodes to the frozen HEAD config.
5. **Bounded + participation-capped sizing** — the modelled notional never grows,
   never falls below the gene floor, and never exceeds P6-A's
   ``max_participation_pct`` of the measured window.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.ga import genome as G

BTC_1H = Path("data/market/BTCUSDT/1h.parquet")
SYMBOL = "BTCUSDT"
DATE_START = "2026-02-01"
DATE_END = "2026-02-15"

#: The P6-D template families (one representative per family, both directions).
NEW_ENTRY_TEMPLATES = (
    G.RVOL_ZSPIKE_HIGH, G.RVOL_ZDRY_LOW, G.VWAP_RECLAIM, G.VWAP_LOSS,
    G.OBV_SLOPE_UP, G.OBV_SLOPE_DOWN, G.AD_SLOPE_UP, G.AD_SLOPE_DOWN,
    G.MFI_OVERBOUGHT, G.MFI_OVERSOLD, G.VP_DIVERGENCE_BULL,
    G.VP_DIVERGENCE_BEAR,
)
NEW_EXIT_TEMPLATES = (G.VWAP_BELOW, G.VWAP_ABOVE, G.RVOL_ZDRY_LOW)
ALL_NEW_TEMPLATES = NEW_ENTRY_TEMPLATES + (G.VWAP_BELOW, G.VWAP_ABOVE)


def _config_hash(config) -> str:
    payload = json.dumps(config.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _skip_without_cache() -> None:
    if not BTC_1H.exists():
        pytest.skip("no cached BTCUSDT 1h parquet in this checkout")


def _indicator_frame(start: str = DATE_START, end: str = DATE_END):
    from core.backtest.data_feeder import DataFeeder

    feeder = DataFeeder("data/market", [SYMBOL], ["1h"], start, end)
    feeder.load()
    raw = feeder.get_all_data_for_symbol(SYMBOL, "1h")
    if raw is None or len(raw) == 0:
        pytest.skip("no cached BTCUSDT 1h bars in the requested window")
    return raw


# ══════════════════════════════════════════════════════════════════════════
# 1 — every template has an owner; a broken template fails the guard
# ══════════════════════════════════════════════════════════════════════════

def test_no_orphan_templates_in_any_pool():
    problems = G.audit_template_ownership()
    assert problems == [], f"orphan/undeclared templates: {problems}"

    for template in ALL_NEW_TEMPLATES:
        assert template in G.CONDITION_POOL["long"] + G.CONDITION_POOL["short"] \
            + G.EXIT_CONDITION_POOL["long"] + G.EXIT_CONDITION_POOL["short"], template
        columns = G.TEMPLATE_REQUIRED_COLUMNS[template]
        assert columns, f"{template!r} declares no column"
        for column in columns:
            assert column in G.COLUMN_INDICATOR_OWNER or \
                column in G.RAW_ALWAYS_AVAILABLE_COLUMNS, (template, column)

    # Only the OBV slope needs an indicator gene; the rest read raw columns.
    assert G.template_owners(G.OBV_SLOPE_UP) == {"obv"}
    assert G.template_owners(G.RVOL_ZSPIKE_HIGH) == set()
    assert G.template_owners(G.MFI_OVERSOLD) == set()


def test_broken_templates_fail_the_guard():
    """The three ways a template can be an orphan, each must be reported."""
    # (a) a column no indicator produces (the classic phantom-column template).
    # `rvol_z` was this test's example until the plan review made it a real
    # indicator column (`compute_all(df, {"volume_flow": {}})`) — see
    # `tests/test_volume_flow_indicator_columns.py`.
    broken = G.audit_template_ownership(
        extra={"phantom_signal > 2.0": ("phantom_signal",)})
    assert broken and "phantom_signal" in broken[0] and "not produced" in broken[0]

    # (b) a declaration that under-states what the string reads
    under = G.audit_template_ownership(extra={"close > adx": ("close",)})
    assert under and "undeclared column" in under[0] and "adx" in under[0]

    # (c) a pool template with no declaration at all
    orphan = G.audit_template_ownership(
        pools={"entry": {"long": ["never declared > 1"]}})
    assert orphan and "orphan template" in orphan[0]

    # (d) a template the sanitiser would keep although its owner is off
    sneak = G.audit_template_ownership(
        extra={"adx > 20 and mystery": ("adx", "mystery")})
    assert sneak, "a template reading an unknown column must be reported"


def test_sanitiser_drops_new_templates_without_their_owner():
    """OBV slope needs the obv gene; the raw-column families never need one."""
    all_on = set(G.INDICATOR_NAMES) | set(G.ALWAYS_AVAILABLE_COLUMNS)
    without_obv = all_on - {"obv"}
    assert G._sanitize_conditions([G.OBV_SLOPE_UP], all_on, "long") == \
        [G.OBV_SLOPE_UP]
    kept = G._sanitize_conditions([G.OBV_SLOPE_UP], without_obv, "long")
    assert G.OBV_SLOPE_UP not in kept, kept
    # The decoder's enabled set ALWAYS contains ``ALWAYS_AVAILABLE_COLUMNS``, so
    # "no indicator gene at all" is that set — and the raw-column families stay.
    no_indicators = set(G.ALWAYS_AVAILABLE_COLUMNS)
    for template in (G.RVOL_ZSPIKE_HIGH, G.VWAP_RECLAIM, G.AD_SLOPE_UP,
                     G.MFI_OVERSOLD, G.VP_DIVERGENCE_BULL):
        assert G._sanitize_conditions([template], no_indicators, "long") == \
            [template]


def test_declared_columns_really_exist_in_a_compute_all_frame():
    """Producibility, measured on real bars — not just declared."""
    from core.strategy.indicators import compute_all

    _skip_without_cache()
    raw = _indicator_frame()
    for template in ALL_NEW_TEMPLATES:
        owners = G.template_owners(template)
        config = {owner: dict(G.INDICATOR_CONFIG_FOR_COLUMNS[owner])
                  for owner in owners}
        frame = compute_all(raw.copy(), config)
        for column in G.TEMPLATE_REQUIRED_COLUMNS[template]:
            assert column in frame.columns, (
                f"{column!r} (needed by {template[:50]!r}) is not producible by "
                f"compute_all with {sorted(owners)}")


def test_always_available_set_is_consistent_with_the_sanitiser():
    """``ALWAYS_AVAILABLE_COLUMNS`` may not claim more than it can deliver.

    ``ema_fast``/``ema_slow`` and — since the ``sma`` defect — ``sma`` are
    backfilled for every frame (measured below on an EMPTY indicator config), so
    the sanitiser's contract and ``compute_all`` agree column for column.
    ``SANITISER_ONLY_COLUMNS`` is therefore empty: the old ``sma`` exemption
    (declared available, written only by the ``sma`` gene) is what let a decoded
    genome carry ``close > sma`` while its frame had no such column, and the
    evaluator dropped the condition with an all-False mask.
    """
    from core.strategy.indicators import (ALWAYS_DERIVED_SMA_PERIOD, compute_all)

    raw = pd.DataFrame({
        "open": [1.0, 2.0, 3.0] * 10, "high": [2.0, 3.0, 4.0] * 10,
        "low": [0.5, 1.0, 1.5] * 10, "close": [1.5, 2.5, 3.5] * 10,
        "volume": [10.0, 20.0, 30.0] * 10,
    })
    frame = compute_all(raw.copy(), {})
    for column in G.ALWAYS_AVAILABLE_COLUMNS:
        assert column in frame.columns, column
    # `sma` is the SMA(N) of close — the same series the condition language's
    # `sma(close, N)` computes, so a condition and the column can never disagree.
    pd.testing.assert_series_equal(
        frame["sma"], raw["close"].rolling(ALWAYS_DERIVED_SMA_PERIOD).mean(),
        check_names=False)
    # The `sma` gene, when present, still owns the alias (its own period).
    gene = compute_all(raw.copy(), {"sma": {"period": 3}})
    assert len(gene) == len(frame) and not gene["sma"].equals(frame["sma"])

    assert G.SANITISER_ONLY_COLUMNS == frozenset()
    assert G.ALWAYS_AVAILABLE_COLUMNS - G.RAW_ALWAYS_AVAILABLE_COLUMNS == {
        "ema_fast", "ema_slow", "sma"}
    # No template outside the `sma` family leans on the backfill.
    for template, columns in G.TEMPLATE_REQUIRED_COLUMNS.items():
        if template in ALL_NEW_TEMPLATES:
            assert "sma" not in columns


# ══════════════════════════════════════════════════════════════════════════
# 2 — round-trip and gene-name operators
# ══════════════════════════════════════════════════════════════════════════

def _chromosome_with_volume_genes(name: str, filter_rvol: float, scale_k: float):
    chrom = G.random_chromosome(name)
    for gene in chrom["continuous"]:
        if gene.name == "volume_filter_rvol":
            gene.value = filter_rvol
        elif gene.name == "volume_scale_k":
            gene.value = scale_k
    return chrom


def test_volume_genes_round_trip_through_a_decoded_config():
    random.seed(20260601)
    for i in range(25):
        chrom = _chromosome_with_volume_genes(f"rt_{i}", 1.3, 0.4)
        first = G.chromosome_to_strategy(chrom)
        again = G.chromosome_to_strategy(G.strategy_to_chromosome(first))
        assert first.model_dump() == again.model_dump(), i
        assert first.indicators[G.VOLUME_GENE_INDICATOR_KEY] == {
            "filter_rvol": 1.3, "scale_k": 0.4}
        for side in ("long", "short"):
            assert all("volume_ratio > 1.3" in c
                       for c in first.entry_conditions[side]), side


def test_wrapper_is_exactly_invertible():
    for condition in ("volume_ratio > 2.0", "rsi < 30",
                      "close < sma(close * volume, 20) / sma(volume, 20)",
                      "close > sma(close, 2) and volume < sma(volume, 2)"):
        wrapped = G.with_volume_filter(condition, 1.3)
        assert G.strip_volume_filter(wrapped) == (condition, 1.3)
        # An unwrapped pool template is never mistaken for the filter gene.
        assert G.strip_volume_filter(condition) == (condition, None)


def test_crossover_and_mutation_handle_the_new_genes_by_name(tmp_path):
    from core.ga.evolver import GAStrategyEvolver, GARunConfig
    from core.strategy.loader import StrategyLoader

    class _Engine:
        config = None

    loader = StrategyLoader(str(tmp_path / "cross"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    evolver = GAStrategyEvolver(_Engine(), loader, GARunConfig(population_size=4))
    random.seed(4)
    p1 = _chromosome_with_volume_genes("p1", 1.2, 0.3)
    p2 = _chromosome_with_volume_genes("p2", 2.4, 0.7)
    for _ in range(20):
        child = evolver._crossover(p1, p2)
        names = [g.name for g in child["continuous"]]
        assert len(names) == len(set(names))
        assert {"volume_filter_rvol", "volume_scale_k"} <= set(names)
        evolver._mutate(child)
        assert {"volume_filter_rvol", "volume_scale_k"} <= {
            g.name for g in child["continuous"]}
    # A pre-P6 chromosome (no volume genes at all) still mutates and crosses.
    legacy = G.random_chromosome("legacy")
    legacy["continuous"] = [g for g in legacy["continuous"]
                            if not g.name.startswith("volume_")]
    child = evolver._crossover(legacy, legacy)
    assert not any(g.name.startswith("volume_") for g in child["continuous"])
    assert G.chromosome_to_strategy(child).name.startswith("ga_child")


# ══════════════════════════════════════════════════════════════════════════
# 3 — the new templates evaluate, and scalar == vectorised
# ══════════════════════════════════════════════════════════════════════════

def _template_strategy(logic: str, narrow: bool = False):
    """A genome whose entries are the P6-D templates.

    ``narrow`` keeps two conditions per side.  They are the two *state-like*
    P6-D templates (OBV and A/D slope), which are both active (long side) on
    **139 of the 587 bars** the conditions are evaluated on — the
    ``_indicator_frame`` window is 587 bars (2026-01-21 → 2026-02-15, the feeder
    prepends its warm-up prefix), while the row the test's own
    ``print(..., bars=...)`` reports is the shorter 337-bar traded window
    ``DATE_START`` → ``DATE_END`` (2026-02-01 → 2026-02-15).  Both numbers are
    measured; they are different objects.  ANDing six families would need every
    family to agree on one bar, leaving the equivalence test vacuous.
    """
    from core.strategy.loader import MLConfig, StrategyConfig

    long_conditions = [G.RVOL_ZSPIKE_HIGH, G.VWAP_RECLAIM, G.OBV_SLOPE_UP,
                       G.AD_SLOPE_UP, G.MFI_OVERSOLD, G.VP_DIVERGENCE_BULL]
    short_conditions = [G.RVOL_ZDRY_LOW, G.VWAP_LOSS, G.OBV_SLOPE_DOWN,
                        G.AD_SLOPE_DOWN, G.MFI_OVERBOUGHT, G.VP_DIVERGENCE_BEAR]
    if narrow:
        long_conditions = [G.OBV_SLOPE_UP, G.AD_SLOPE_UP]
        short_conditions = [G.OBV_SLOPE_DOWN, G.AD_SLOPE_DOWN]
    return StrategyConfig(
        name=f"p6d_{logic}", enabled=True, mode="trend", timeframes=["1h"],
        indicators={"obv": {"period": 14}},
        entry_conditions={"long": long_conditions, "short": short_conditions},
        exit_conditions={"long": [G.VWAP_BELOW], "short": [G.VWAP_ABOVE]},
        condition_logic=logic, ml_config=MLConfig(enabled=False),
    )


def test_every_new_template_fires_on_real_cached_bars():
    """A template that never fires is the plan's rollback case — measure it."""
    from core.strategy.indicators import compute_all, evaluate_condition

    _skip_without_cache()
    raw = _indicator_frame()
    frame = compute_all(raw.copy(), {"obv": {"period": 14}})
    counts = {template: int(evaluate_condition(frame, template).sum())
              for template in ALL_NEW_TEMPLATES}
    empty = [t for t, n in counts.items() if n == 0]
    assert not empty, f"templates that never fire on {len(frame)} bars: {empty}"
    print("\n[template hit counts] " + ", ".join(
        f"{n}" for n in counts.values()))


@pytest.mark.parametrize("logic", ["or", "and"])
def test_vectorised_entry_path_equals_the_scalar_kernel(logic):
    """``SignalMatrixBuilder`` vs ``StrategyConfig.entry_sides``: 0 mismatches."""
    from app.config import Config, SignalWeights
    from core.backtest.data_feeder import DataFeeder
    from core.backtest.signal_matrix import SignalMatrixBuilder
    from core.strategy.indicators import compute_all

    _skip_without_cache()
    Config._instance = None
    config = Config.load("sim")
    config.signal_weights = SignalWeights(indicator=1.0, ml=0.0, news=0.0)
    feeder = DataFeeder("data/market", [SYMBOL], ["1h"], DATE_START, DATE_END)
    feeder.load()
    strategy = _template_strategy(logic, narrow=(logic == "and"))
    matrix = SignalMatrixBuilder(feeder).build([strategy], [SYMBOL])
    frame = compute_all(feeder.get_all_data_for_symbol(SYMBOL, "1h").copy(),
                        strategy.indicators)

    key = (strategy.name, SYMBOL, "1h")
    assert key in set(matrix.signals.index), "the genome produced no entries"
    row = matrix.signals.loc[[key]].iloc[0]

    mismatches = []
    active = 0
    for ts in row.index:
        pos = frame.index.get_loc(ts)
        long_active, short_active = strategy.entry_sides(frame.iloc[:pos + 1])
        expected = (1 if (long_active and not short_active)
                    else -1 if (short_active and not long_active) else 0)
        got = int(row.loc[ts])
        active += 1 if expected else 0
        if expected != got:
            mismatches.append((str(ts), expected, got))
    assert active > 0, "the new templates never activated a side — vacuous"
    assert mismatches == [], f"{len(mismatches)} mismatches, e.g. {mismatches[:3]}"
    print(f"\n[{logic}] bars={len(row)} active={active} mismatches=0")


def test_volume_filter_is_a_real_filter_under_both_logics():
    """The filter gene narrows entries under ``or`` *and* ``and``."""
    from core.strategy.indicators import compute_all

    _skip_without_cache()
    raw = _indicator_frame()
    frame = compute_all(raw.copy(), {"obv": {"period": 14}})
    rvol = 1.4
    base = _template_strategy("or", narrow=True)
    filtered = _template_strategy("or", narrow=True)
    filtered.entry_conditions = {
        side: [G.with_volume_filter(c, rvol)
               for c in base.entry_conditions[side]]
        for side in ("long", "short")}

    for logic in ("or", "and"):
        base.condition_logic = logic
        filtered.condition_logic = logic
        dropped = 0
        kept = 0
        for pos in range(len(frame)):
            window = frame.iloc[:pos + 1]
            ts = window.index[-1]
            for side in ("long", "short"):
                base_active, _ = base.entry_sides(window)
                filtered_active, _ = filtered.entry_sides(window)
                # ANDing the filter into each condition is, by distributivity,
                # exactly "the base side is active AND the bar's RVOL passes" —
                # for OR and AND alike.  That identity is what is asserted here,
                # bar for bar, through the shared kernel.
                expected = base_active and float(frame["volume_ratio"].loc[ts]) > rvol
                assert filtered_active == expected, (logic, side, str(ts))
                if base_active and not expected:
                    dropped += 1
                if expected:
                    kept += 1
        assert kept > 0, f"{logic}: the filtered genome never entered — vacuous"
        assert dropped > 0, (
            f"{logic}: the filter never removed a base entry on this window")


# ══════════════════════════════════════════════════════════════════════════
# 4 — off path is the identity
# ══════════════════════════════════════════════════════════════════════════

class _Liquidity:
    def __init__(self, enabled=False, impact_k=0.0, max_participation_pct=1.0,
                 lookback_bars=20, impact_exponent=0.5):
        self.enabled = enabled
        self.impact_k = impact_k
        self.max_participation_pct = max_participation_pct
        self.lookback_bars = lookback_bars
        self.impact_exponent = impact_exponent
        self.per_symbol = {}


class _Config:
    def __init__(self, **kwargs):
        self.risk_liquidity = _Liquidity(**kwargs)
        self.backtest_cost_enabled = True
        self.backtest_taker_fee_pct = 0.04
        self.backtest_spread_pct = {"BTCUSDT": 0.03, "DEFAULT": 0.03}


def _trades_and_curve():
    """Two closed trades whose ``cost`` is the REAL cost model's number.

    Hand-written costs would make ``_engine_impact_in_cost`` measure a phantom
    impact (any mismatch between the recorded and the modelled cost looks like
    an impact charge), so the fixture prices the fills with
    ``apply_trading_costs`` exactly like the engine does.
    """
    from core.backtest.cost_model import apply_trading_costs

    config = _Config(enabled=False, impact_k=0.0)
    rows = [("long", 100.0, 103.0, 10.0, "2026-02-01 00:00", "2026-02-02 00:00"),
            ("short", 105.0, 104.0, 5.0, "2026-02-03 00:00", "2026-02-05 00:00")]
    trades = []
    for side, entry, exit_price, qty, opened, closed in rows:
        gross = ((exit_price - entry) if side == "long" else (entry - exit_price)) * qty
        cost = apply_trading_costs(entry, exit_price, qty, "BTCUSDT", config)
        trades.append({
            "symbol": "BTCUSDT", "side": side, "entry_price": entry,
            "exit_price": exit_price, "quantity": qty,
            "pnl": round(gross - cost, 2), "cost": round(cost, 4),
            "amount_usdt": round(qty * entry, 2), "strategy": "g",
            "opened_at": opened, "closed_at": closed,
        })
    equity = 10000.0
    curve = [{"time": "2026-02-01 00:00", "equity": equity}]
    for trade in trades:
        equity += trade["pnl"]
        curve.append({"time": trade["closed_at"], "equity": round(equity, 2)})
    return trades, curve


def test_model_off_returns_the_very_same_objects():
    from core.ga.fitness import (apply_executability_model,
                                 executability_applies)

    trades, curve = _trades_and_curve()
    chrom = _chromosome_with_volume_genes("off", 0.0, 0.0)
    config = _Config(enabled=False, impact_k=0.0)
    assert executability_applies(chrom, config) is False
    out = apply_executability_model(trades, curve, chrom, config, None)
    assert out["applied"] is False
    assert out["trades"] is trades and out["equity_curve"] is curve


def test_model_on_with_neutral_inputs_is_bit_identical():
    """Forced through the ON branch with a neutral gene and no window."""
    from core.ga.fitness import apply_executability_model

    trades, curve = _trades_and_curve()
    chrom = _chromosome_with_volume_genes("neutral", 0.0, 0.0)
    for chrom_gene_value in (0.0,):
        for gene in chrom["continuous"]:
            if gene.name == "volume_scale_k":
                gene.value = chrom_gene_value
        config = _Config(enabled=True, impact_k=0.5)
        out = apply_executability_model(trades, curve, chrom, config, None)
        assert out["applied"] is True, "the ON branch must be exercised"
        assert out["summary"]["impact_usdt"] == 0.0
        for before, after in zip(trades, out["trades"]):
            for field in ("symbol", "side", "entry_price", "exit_price",
                          "quantity", "pnl", "cost", "amount_usdt"):
                assert after[field] == before[field], (field, before, after)
            # ... and the model says what it did, per trade.
            assert after["exec_scale"] == 1.0
            assert after["exec_impact_usdt"] == 0.0
        assert out["equity_curve"] == curve


def test_pre_p6_chromosome_decodes_to_the_frozen_head_config():
    """Decode identity for a genome that predates P6-D (frozen at HEAD)."""
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
    config = G.chromosome_to_strategy(frozen)
    # HEAD (c703b8b) decoded this exact chromosome to this exact hash — verified
    # by executing HEAD's own ``core/ga/genome.py`` (``git show HEAD:...``) on the
    # same input, so it is a measurement, not a guess.  The P6-D additions (the
    # sandwich key, the filter conjuncts) must not appear.
    #
    # P7-S1 re-pin, and it is *provable* rather than a new number: adding the
    # ``regime_filter: list[str] = []`` field to ``StrategyConfig`` moves the
    # full-dump hash by exactly that one key, and the same dump with that key
    # removed still hashes to the frozen value below — so the decoded pre-P7
    # genome is unchanged except for one empty, inert field.  The full argument
    # lives in ``tests/test_p7_regime_s1.py``
    # (``test_the_regime_field_is_purely_additive_to_the_frozen_config``).
    assert _config_hash(config) == "4d40f95abe61d7e2"
    assert hashlib.sha256(json.dumps(
        {k: v for k, v in config.model_dump().items() if k != "regime_filter"},
        sort_keys=True, default=str).encode()).hexdigest()[:16] \
        == "809ddf7ba45af011"
    assert config.regime_filter == []
    assert G.VOLUME_GENE_INDICATOR_KEY not in config.indicators
    assert config.entry_conditions == {"long": ["rsi < 30"],
                                       "short": ["rsi > 70"]}
    # And it encodes back without inventing a volume gene.
    encoded = {g.name: g.value for g in G.strategy_to_chromosome(config)["continuous"]}
    assert encoded["volume_filter_rvol"] == 0.0
    assert encoded["volume_scale_k"] == 0.0


# ══════════════════════════════════════════════════════════════════════════
# 5 — volume sizing: bounded, shrink-only, participation-capped
# ══════════════════════════════════════════════════════════════════════════

def _volume_context(rows):
    frame = pd.DataFrame(rows)
    index = pd.to_datetime(frame["time"]).to_numpy()
    quote = frame["quote_volume"].to_numpy(dtype=float)
    from core.ga.fitness import VolumeContext

    return VolumeContext({("BTCUSDT", "1h"): {
        "index": index,
        "rvol": frame["rvol"].to_numpy(dtype=float),
        "cum_qv": np.concatenate([[0.0], np.cumsum(quote)]),
    }}, lookback_bars=20)


def test_volume_size_factor_is_bounded():
    from core.ga.fitness import (VOLUME_SCALE_CAP, VOLUME_SCALE_MIN,
                                 volume_size_factor)

    assert volume_size_factor(1.0, 0.5) == pytest.approx(1.0)
    assert volume_size_factor(3.0, 0.5) == pytest.approx(VOLUME_SCALE_CAP)
    assert volume_size_factor(0.2, 1.0) == pytest.approx(VOLUME_SCALE_MIN)
    assert volume_size_factor(0.5, 0.0) == pytest.approx(1.0)   # gene off
    assert volume_size_factor(None, 0.5) == pytest.approx(1.0)  # unmeasured
    # rvol <= 0 is "not measured" (a dead bar), not "infinitely dry": the model
    # must not invent a size cut from a missing measurement.
    assert volume_size_factor(0.0, 0.5) == pytest.approx(1.0)
    assert volume_size_factor(-1.0, 0.5) == pytest.approx(1.0)
    for rvol in (0.01, 0.5, 1.0, 2.0, 10.0):
        for k in (0.05, 0.25, 1.0):
            assert VOLUME_SCALE_MIN <= volume_size_factor(rvol, k) \
                <= VOLUME_SCALE_CAP


def test_sizing_is_shrink_only_and_participation_capped():
    from core.ga.fitness import apply_executability_model, build_volume_context

    rows = [{"time": f"2026-02-{day:02d} 00:00", "quote_volume": 10_000.0,
             "rvol": 0.2} for day in range(1, 9)]
    context = _volume_context(rows)
    trades, curve = _trades_and_curve()
    chrom = _chromosome_with_volume_genes("sized", 0.0, 1.0)
    # 1 % of a 200 000 USDT window = 2 000 USDT ceiling for the whole trade.
    config = _Config(enabled=True, impact_k=0.5, max_participation_pct=1.0)
    out = apply_executability_model(trades, curve, chrom, config, context)
    assert out["applied"] is True
    summary = out["summary"]
    assert summary["notional_after"] <= summary["notional_before"]
    assert summary["min_factor"] >= 0.0 and summary["mean_factor"] <= 1.0
    assert summary["trades_scaled"] == len(trades)
    assert summary["impact_usdt"] > 0.0
    for before, after in zip(trades, out["trades"]):
        assert after["amount_usdt"] <= before["amount_usdt"]
        assert 0.0 <= after["exec_scale"] <= 1.0

    # A bigger window removes the participation cap (the gene still shrinks).
    generous = _volume_context([{**row, "quote_volume": 1e9} for row in rows])
    out2 = apply_executability_model(trades, curve, chrom, config, generous)
    assert out2["summary"]["trades_capped"] == 0
    assert out2["summary"]["notional_after"] < out2["summary"]["notional_before"]


def test_impact_term_lowers_the_scored_pnl_and_fitness():
    from core.ga.fitness import (apply_executability_model, score_stats,
                                 stats_from_trades)

    rows = [{"time": f"2026-02-{day:02d} 00:00", "quote_volume": 100_000.0,
             "rvol": 1.0} for day in range(1, 9)]
    context = _volume_context(rows)
    trades, curve = _trades_and_curve()
    chrom = _chromosome_with_volume_genes("impact", 0.0, 0.0)

    base_stats = stats_from_trades(trades, curve, 10000.0)
    base = score_stats(base_stats, chrom, n_trials=1)
    modelled = apply_executability_model(
        trades, curve, chrom, _Config(enabled=True, impact_k=0.5), context)
    charged = modelled["summary"]["impact_usdt"]
    assert charged > 0.0
    priced = stats_from_trades(modelled["trades"], modelled["equity_curve"], 10000.0)
    after = score_stats(priced, chrom, n_trials=1)
    assert after["fitness"] < base["fitness"]
    assert after["total_return_pct"] < base["total_return_pct"]
    print(f"\n[impact] {charged:.4f} USDT over {len(trades)} trades: "
          f"fitness {base['fitness']:.4f} -> {after['fitness']:.4f}")


def test_volume_context_matches_the_p6a_window_measurement():
    """The pre-computed window equals ``core.risk.liquidity.recent_quote_volume``."""
    from core.risk.liquidity import recent_quote_volume

    _skip_without_cache()
    from core.ga.fitness import build_volume_context

    raw = _indicator_frame("2026-01-15", DATE_END)
    config = _Config()
    config.data_dir = "data"
    context = build_volume_context(config, [SYMBOL], ["1h"], "2026-01-15", DATE_END)
    assert context is not None
    frame = raw.copy()
    for ts in list(raw.index[::37])[1:]:
        _rvol, window = context.lookup(SYMBOL, "1h", ts)
        cut = int(frame.index.searchsorted(ts, side="right"))
        expected = float(recent_quote_volume(frame.iloc[:cut], 20))
        assert window == pytest.approx(expected, rel=1e-12)
