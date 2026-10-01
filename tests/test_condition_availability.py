"""`sma` is really always available — and an unevaluable condition fails loudly.

The defect this file pins (measured in the live GA job ``ga_0a442907``)::

    WARNING core.strategy.indicators:evaluate_condition:390 - Condition rejected
      or unevaluable: 'close < sma' — unknown column/identifier: sma

``core.ga.genome.ALWAYS_AVAILABLE_COLUMNS`` declared ``sma`` available, the
condition pools (and the sanitiser's own fallback) generate ``close > sma`` /
``close < sma``, but ``compute_all`` wrote the column only for the ``sma``
indicator gene — so with the gene off the evaluator answered with an all-False
mask and the genome was scored as if it declared fewer conditions than it does
(the last champion's entry was ``close > sma`` and its exit ``close < sma`` with
the ``sma`` gene off).

Covered here:

1. ``sma`` conditions evaluate on the frame a decoded genome produces;
2. a genuinely unknown column makes the genome fail loudly (decoder +
   ``evaluate_chromosome`` + the batch path), counted and not selectable;
3. a guard that ties ``ALWAYS_AVAILABLE_COLUMNS`` to what ``compute_all``
   actually produces, so the declaration cannot drift again.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.ga.genome as G
from core.ga import fitness as F
from core.ga.genome import UnevaluableConditionError
from core.strategy.loader import MLConfig, StrategyConfig


def _bars(bars: int = 400) -> pd.DataFrame:
    """Deterministic OHLCV frame — no cached data, no I/O."""
    rng = np.random.default_rng(20261001)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, bars))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": rng.uniform(10.0, 60.0, bars),
        },
        index=pd.date_range("2026-01-01", periods=bars, freq="1h"),
    )


def _genome_with(entry_long: list[str], name: str,
                 exit_long: tuple[str, ...] = ("close < sma",)) -> dict:
    """Chromosome whose decoded config carries *entry_long* and only ``rsi`` on.

    With ``macd`` off the decoder's sanitiser drops a MACD entry condition and
    injects its own fallback — ``close > sma`` for this indicator set
    (``_sanitize_conditions``: no ema, no bollinger, ``sma`` «always
    available»).  ``close < sma`` is the ``EXIT_CONDITION_POOL["long"]`` template.
    Together they are exactly the champion shape the defect produced, so the
    fixture reaches the defect through the real code path, not by hand.
    """
    config = StrategyConfig(
        name=name, enabled=True, mode="trend", timeframes=["1h"],
        indicators={"rsi": {"period": 14, "source": "close"}},
        entry_conditions={"long": list(entry_long),
                          "short": ["rsi > 70"]},
        exit_conditions={"long": list(exit_long),
                         "short": ["rsi < 30"]},
        ml_config=MLConfig(enabled=False),
    )
    return G.strategy_to_chromosome(config)


# ══════════════════════════════════════════════════════════════════════════
# (a) `sma` conditions evaluate — the column exists with the expected values
# ══════════════════════════════════════════════════════════════════════════

def test_sma_conditions_evaluate_on_a_decoded_genome_frame():
    from core.strategy.indicators import (ALWAYS_DERIVED_SMA_PERIOD, compute_all,
                                          condition_failure_log,
                                          evaluate_condition)

    decoded = G.chromosome_to_strategy(_genome_with(["macd_histogram > 0"],
                                                   "sma_genome"))
    assert "sma" not in decoded.indicators, \
        "the fixture must reach the defect with the sma gene OFF"
    assert decoded.entry_conditions["long"] == ["close > sma"]
    assert decoded.exit_conditions["long"] == ["close < sma"]

    frame = compute_all(_bars(), decoded.indicators)
    assert "sma" in frame.columns
    # The column is the same series the condition language's `sma(close, N)`
    # computes, so a condition and the column can never disagree.
    pd.testing.assert_series_equal(
        frame["sma"],
        frame["close"].rolling(ALWAYS_DERIVED_SMA_PERIOD).mean(),
        check_names=False)

    failures_before = condition_failure_log()
    entries = evaluate_condition(frame, decoded.entry_conditions["long"][0])
    exits = evaluate_condition(frame, decoded.exit_conditions["long"][0])
    assert condition_failure_log() == failures_before, \
        "an sma condition must not be rejected any more"

    assert entries.equals(frame["close"] > frame["sma"])
    assert exits.equals(frame["close"] < frame["sma"])
    # Not vacuously all-False: the condition fires on this window.
    assert entries.any() and exits.any() and not entries.equals(exits)


def test_sanitiser_fallback_condition_is_evaluable():
    """The decoder's own fallback must be a condition its frame can evaluate."""
    enabled = set(G.ALWAYS_AVAILABLE_COLUMNS) | {"rsi"}
    fallback = G._sanitize_conditions(["macd_histogram > 0"], enabled, "long")
    assert fallback == ["close > sma"], fallback
    columns = G.evaluable_columns({"rsi": {"period": 14, "source": "close"}})
    assert G.template_identifier_columns(fallback[0]) <= columns


# ══════════════════════════════════════════════════════════════════════════
# (b) an unknown column fails loudly — counted, logged, excluded
# ══════════════════════════════════════════════════════════════════════════

def test_unknown_column_is_refused_by_the_decoder():
    chrom = _genome_with(["phantom_signal > 2"], "phantom_genome")
    with pytest.raises(UnevaluableConditionError) as exc:
        G.chromosome_to_strategy(chrom)
    message = str(exc.value)
    assert "phantom_signal" in message and "entry.long" in message, message


def test_unknown_column_scores_minus_999_and_is_counted():
    class _NeverCalled:
        def run_with_exit_evaluation(self, **kwargs):  # pragma: no cover
            raise AssertionError("an unevaluable genome must not reach the engine")

    chrom = _genome_with(["phantom_signal > 2"], "phantom_fitness")
    before = F.condition_rejection_counts()
    result = F.evaluate_chromosome(chrom, ["BTCUSDT"], "2026-01-01", "2026-01-05",
                                   _NeverCalled(), loader=None)
    after = F.condition_rejection_counts()

    assert result["fitness"] == -999
    assert result["flag"] == "unevaluable_condition"
    assert "phantom_signal" in result["error"]
    assert sum(after.values()) == sum(before.values()) + 1, \
        "the rejection must be counted"
    assert any("phantom_signal" in reason for reason in after), after


def test_batch_eval_isolates_a_rejected_genome_without_shifting_indices():
    """One bad genome is scored −999; the chunk still runs with the same shape.

    The rejected genome's slot is held by a never-trading placeholder, so the
    engine's per-genome slot divisor (``max_open_trades // len(strategies)``) is
    the one the surviving genomes had before the rejection.
    """
    class _ErrorEngine:
        def __init__(self):
            self.config = None
            self.strategies = None

        def run_with_exit_evaluation(self, **kwargs):
            self.strategies = list(kwargs["strategies"])
            return {"error": "engine down"}

    population = [
        _genome_with(["macd_histogram > 0"], "good_0"),
        _genome_with(["phantom_signal > 2"], "bad_1"),
        _genome_with(["macd_histogram > 0"], "good_2"),
    ]
    engine = _ErrorEngine()
    F.evaluate_population_batch(
        population, ["BTCUSDT"], "2026-01-01", "2026-01-05", engine, loader=None,
        batch_size=3, max_workers=1, progress_callback=None)

    assert population[1]["fitness_result"]["fitness"] == -999
    assert population[1]["fitness_result"]["flag"] == "unevaluable_condition"
    assert "phantom_signal" in population[1]["fitness_result"]["error"]
    # Chunk shape preserved: three strategies in population order, the rejected
    # slot held by a never-trading placeholder.
    assert engine.strategies is not None and len(engine.strategies) == 3
    assert [c.name.split("_")[3] for c in engine.strategies] == ["0", "1", "2"]
    assert engine.strategies[1].name.endswith("_rejected")

    # The placeholder cannot trade: no bar satisfies its conditions.
    from core.strategy.indicators import compute_all, evaluate_condition

    placeholder = engine.strategies[1]
    frame = compute_all(_bars(64), placeholder.indicators)
    for side in ("long", "short"):
        for condition in (placeholder.entry_conditions[side]
                          + placeholder.exit_conditions[side]):
            assert not evaluate_condition(frame, condition).any(), condition

    for index in (0, 2):
        assert population[index]["fitness_result"]["flag"] == "engine_error"
        assert population[index]["fitness_result"]["fitness"] == -999


# ══════════════════════════════════════════════════════════════════════════
# (c) guard — the always-available set equals what the evaluator can produce
# ══════════════════════════════════════════════════════════════════════════

def test_always_available_columns_are_all_produced():
    """``ALWAYS_AVAILABLE_COLUMNS`` may not claim a column ``compute_all`` omits.

    Measured on an EMPTY indicator config — the frame a genome with every gene
    off is evaluated on — so the declaration cannot drift from the producer the
    way ``sma`` did.
    """
    from core.strategy.indicators import compute_all

    produced = set(compute_all(_bars(64), {}).columns)
    assert G.ALWAYS_AVAILABLE_COLUMNS - produced == set()
    # The decoder's own probe agrees with the evaluator's producer.
    assert G.evaluable_columns({}) >= G.ALWAYS_AVAILABLE_COLUMNS
    # The old exemption is closed and stays closed.
    assert G.SANITISER_ONLY_COLUMNS == frozenset()


def test_every_pool_template_is_evaluable_on_its_own_indicator_frame():
    for pool in (G.CONDITION_POOL, G.EXIT_CONDITION_POOL):
        for side, templates in pool.items():
            for template in templates:
                owners = G.template_owners(template)
                config = {owner: dict(G.INDICATOR_CONFIG_FOR_COLUMNS[owner])
                          for owner in owners}
                columns = G.evaluable_columns(config)
                missing = G.template_identifier_columns(template) - columns
                assert not missing, (template, sorted(missing))


def test_unknown_column_still_cannot_be_declared_available():
    """The sanitiser must not keep a condition whose column nothing produces."""
    with pytest.raises(UnevaluableConditionError):
        G.chromosome_to_strategy(_genome_with(["mystery > 1"], "mystery"))
