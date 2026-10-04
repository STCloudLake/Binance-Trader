"""Regression: the vectorized/hybrid path must honour ``condition_logic``.

Why this file exists
--------------------
``SignalMatrixBuilder`` (the entry path of the hybrid engine, and therefore the
engine GA scores champions with) used to combine a side's entry conditions with
its own **OR-only** loop and never read ``StrategyConfig.condition_logic``.  The
legacy engine (``engine.py``) and the live path (``StrategyEngine._evaluate``)
both call ``StrategyConfig.entry_sides``, which honours the field.  A champion
evolved with ``condition_logic: and`` was therefore *scored* under OR and
*traded* under AND — the "scoring ≠ publishing" divergence the GA work removed
everywhere else.

The three properties pinned here, on the real cached BTCUSDT 1h series:

  a. the AND entry set is a **strict subset** of the OR entry set (the field
     genuinely changes which bars are entries, so the bug was reachable);
  b. per bar, the matrix entry equals ``entry_sides`` (0 mismatches) — the
     vectorized fold is the same rule as the shared evaluator;
  c. the legacy engine and the hybrid engine produce the **same trades and the
     same metrics** when both run with the same field (the regression that
     matters, and the one that failed before the fix).

The window is chosen inside a range that is byte-identical in the cached file
(2026-01-01 … 2026-03-31), so the assertions do not depend on the volatility /
gap repair of the cache.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from app.config import Config, SignalWeights
from core.strategy.loader import MLConfig, StrategyConfig

BTC_1H = Path("data/market/BTCUSDT/1h.parquet")
SYMBOL = "BTCUSDT"
#: Matrix/parity window: one month inside the cached range.
DATE_START = "2026-02-01"
DATE_END = "2026-03-01"
#: Engine window: long enough for the AND genome to take a non-trivial number
#: of trades (AND=16 vs OR=83; legacy and hybrid agree for each logic, so this
#: is selectivity, not an engine divergence), short enough to stay a fast test.
ENGINE_START = "2026-01-01"
ENGINE_END = "2026-03-31"
INITIAL_BALANCE = 10000.0

INDICATORS = {"rsi": {"period": 14, "source": "close"},
              "sma": {"period": 20}}
#: Mirrored, non-overlapping conditions: OR follows the price/SMA trend on most
#: bars, AND requires RSI to agree as well (8 of 673 bars in the pinned window).
ENTRY_CONDITIONS = {"long": ["rsi < 45", "close > sma"],
                    "short": ["rsi > 55", "close < sma"]}
EXIT_CONDITIONS = {"long": ["rsi > 60"], "short": ["rsi < 40"]}


def _strategy(logic: str) -> StrategyConfig:
    return StrategyConfig(
        name=f"cl_{logic}", enabled=True, mode="trend", timeframes=["1h"],
        indicators=dict(INDICATORS), entry_conditions=dict(ENTRY_CONDITIONS),
        exit_conditions=dict(EXIT_CONDITIONS), condition_logic=logic,
        ml_config=MLConfig(enabled=False),
    )


def _skip_without_cache() -> None:
    if not BTC_1H.exists():
        pytest.skip("no cached BTCUSDT 1h parquet in this checkout")


def _config(mode: str):
    Config._instance = None
    config = Config.load("sim")
    config.backtest_engine_mode = mode
    config.backtest_ml_enabled = False
    # Neutralise fusion (100 % indicator) so the matrix threshold is the
    # indicator sign — the same neutralisation the other parity gates use.
    config.signal_weights = SignalWeights(indicator=1.0, ml=0.0, news=0.0)
    # P9 re-pin: this gate compares legacy against the hybrid engine, which has no
    # fill seam and prices the decision bar's close.  The shipped default is now
    # `next_open`, so the one convention both engines share is requested
    # explicitly here (values unchanged, the comparison stays apples-to-apples).
    config.backtest_fill_convention = "close"
    return config


def _matrix_and_frame(logic: str):
    """Build the signal matrix and the indicator frame the builder used."""
    from core.backtest.data_feeder import DataFeeder
    from core.backtest.signal_matrix import SignalMatrixBuilder
    from core.strategy.indicators import compute_all

    Config._instance = None
    config = Config.load("sim")
    feeder = DataFeeder(str(config.data_dir) + "/market", [SYMBOL], ["1h"],
                        DATE_START, DATE_END)
    feeder.load()
    strategy = _strategy(logic)
    matrix = SignalMatrixBuilder(feeder).build([strategy], [SYMBOL])
    frame = compute_all(
        feeder.get_all_data_for_symbol(SYMBOL, "1h").copy(), strategy.indicators)
    return strategy, matrix, frame


def _matrix_row(matrix, strategy):
    if matrix.signals.empty:
        return None
    key = (strategy.name, SYMBOL, "1h")
    if key not in set(matrix.signals.index):
        return None
    return matrix.signals.loc[[key]].iloc[0]


def _entry_sides_series(strategy, frame, timestamps) -> pd.Series:
    """``entry_sides`` for every timestamp, as the matrix's -1/0/+1 signal."""
    values = []
    for ts in timestamps:
        pos = frame.index.get_loc(ts)
        long_active, short_active = strategy.entry_sides(frame.iloc[:pos + 1])
        values.append(1 if (long_active and not short_active)
                      else (-1 if (short_active and not long_active) else 0))
    return pd.Series(values, index=timestamps, dtype="int8")


def test_and_entry_set_is_a_strict_subset_of_the_or_entry_set():
    """(a) The gene materially changes the entry set — on the real cache."""
    _skip_without_cache()
    s_and, m_and, frame = _matrix_and_frame("and")
    s_or, m_or, _ = _matrix_and_frame("or")

    row_and = _matrix_row(m_and, s_and)
    row_or = _matrix_row(m_or, s_or)
    assert row_and is not None, "AND genome produced no entries at all"
    assert row_or is not None, "OR genome produced no entries at all"

    support_and = set(row_and.index[row_and.to_numpy() != 0])
    support_or = set(row_or.index[row_or.to_numpy() != 0])
    assert support_and, "AND matrix support is empty — the test would be vacuous"
    assert support_and < support_or, (
        f"AND support must be a STRICT subset of OR: |and|={len(support_and)} "
        f"|or|={len(support_or)} and_only={len(support_and - support_or)}")

    # The same nesting holds for the per-(bar, side) entry sets, which is what
    # "entry set" means before the ambiguity guard turns sides into signals.
    def side_pairs(strategy, matrix):
        row = _matrix_row(matrix, strategy)
        timestamps = list(row.index) if row is not None else list(
            frame.index[frame.index >= pd.Timestamp(DATE_START)])
        pairs = set()
        for ts in timestamps:
            pos = frame.index.get_loc(ts)
            long_active, short_active = strategy.entry_sides(frame.iloc[:pos + 1])
            if long_active:
                pairs.add((ts, "long"))
            if short_active:
                pairs.add((ts, "short"))
        return pairs

    pairs_and = side_pairs(s_and, m_and)
    pairs_or = side_pairs(s_or, m_or)
    assert pairs_and < pairs_or, (
        f"side-active pairs: and={len(pairs_and)} or={len(pairs_or)}")

    print(f"\n[item 1a] bars={len(row_and)} matrix support AND={len(support_and)} "
          f"OR={len(support_or)} (strict subset); side-active pairs "
          f"AND={len(pairs_and)} OR={len(pairs_or)}")


@pytest.mark.parametrize("logic", ["and", "or"])
def test_matrix_entries_equal_entry_sides_bar_for_bar(logic):
    """(b) The vectorized fold == ``StrategyConfig.entry_sides`` on every bar."""
    _skip_without_cache()
    strategy, matrix, frame = _matrix_and_frame(logic)
    row = _matrix_row(matrix, strategy)
    assert row is not None, f"{logic} genome produced no entries"

    expected = _entry_sides_series(strategy, frame, list(row.index))
    got = row.astype("int8")
    mismatches = int((expected.to_numpy() != got.to_numpy()).sum())
    assert mismatches == 0, (
        f"{logic}: {mismatches} bar(s) where the matrix entry differs from "
        f"entry_sides, e.g. "
        f"{[ (str(t), int(expected.loc[t]), int(got.loc[t])) for t in row.index[expected.to_numpy() != got.to_numpy()][:3] ]}")

    print(f"\n[item 1b] logic={logic}: bars={len(row)} mismatches={mismatches} "
          f"long_bars={int((got == 1).sum())} short_bars={int((got == -1).sum())}")


def _run_both_engines(logic: str):
    """Legacy and hybrid engine on identical inputs for one ``condition_logic``."""
    from core.backtest.engine import BacktestEngine
    from core.backtest.engine_hybrid import run_hybrid
    from core.strategy.loader import StrategyLoader

    tmp = Path(tempfile.mkdtemp(prefix="bt_cond_logic_")) / "strategies"
    tmp.mkdir(parents=True, exist_ok=True)
    loader = StrategyLoader(str(tmp))
    strategy = _strategy(logic)
    loader.save(strategy)

    engine = BacktestEngine.__new__(BacktestEngine)
    engine.config = _config("legacy")
    engine.strategy_engine = type("obj", (object,), {"loader": loader})()
    legacy = engine.run_with_exit_evaluation(
        strategies=[strategy.name], symbols=[SYMBOL], date_start=ENGINE_START,
        date_end=ENGINE_END, initial_balance=INITIAL_BALANCE, mode="full",
        simulate_ai_weights=False)
    hybrid = run_hybrid(
        strategies=[strategy], symbols=[SYMBOL], date_start=ENGINE_START,
        date_end=ENGINE_END, config=_config("hybrid"), loader=loader,
        initial_balance=INITIAL_BALANCE)
    return legacy, hybrid


def _assert_engine_parity(legacy, hybrid, label: str) -> None:
    def key(trade):
        return (trade.get("strategy", ""), trade.get("symbol", ""),
                trade.get("side", ""), str(trade.get("opened_at", "")))

    legacy_trades = legacy.get("trades", [])
    hybrid_trades = hybrid.get("trades", [])
    legacy_map = {key(t): t for t in legacy_trades}
    hybrid_map = {key(t): t for t in hybrid_trades}
    assert set(legacy_map) == set(hybrid_map), (
        f"[{label}] trade sets differ: legacy-only="
        f"{sorted(set(legacy_map) - set(hybrid_map))[:3]} hybrid-only="
        f"{sorted(set(hybrid_map) - set(legacy_map))[:3]} "
        f"(legacy={len(legacy_trades)} hybrid={len(hybrid_trades)})")
    for k, ltrade in legacy_map.items():
        htrade = hybrid_map[k]
        assert ltrade["quantity"] == pytest.approx(htrade["quantity"], abs=1e-6), \
            f"[{label}] size mismatch on {k}"
        assert ltrade["pnl"] == pytest.approx(htrade["pnl"], abs=0.05), \
            f"[{label}] pnl mismatch on {k}: {ltrade['pnl']} vs {htrade['pnl']}"
        assert ltrade["exit_price"] == pytest.approx(htrade["exit_price"], abs=0.05), \
            f"[{label}] exit price mismatch on {k}"
    for metric in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                   "win_rate_pct", "profit_factor"):
        lv = legacy["metrics"].get(metric)
        hv = hybrid["metrics"].get(metric)
        assert lv == pytest.approx(hv, abs=1e-6, rel=1e-6), \
            f"[{label}] metric '{metric}' differs: legacy={lv} hybrid={hv}"


def test_legacy_and_hybrid_agree_with_the_same_condition_logic():
    """(c) The regression that matters: same field ⇒ same trades in both engines.

    Before the fix the hybrid run for the ``and`` genome matched the ``or``
    result (it never read the field), so trade counts differed from the legacy
    engine's. The ``and``-vs-``or`` split asserted at the end pins that the two
    runs really are different strategies, i.e. the parity above is not vacuous.
    """
    _skip_without_cache()
    results = {}
    for logic in ("and", "or"):
        legacy, hybrid = _run_both_engines(logic)
        _assert_engine_parity(legacy, hybrid, label=f"condition_logic={logic}")
        results[logic] = (len(legacy.get("trades", [])),
                          len(hybrid.get("trades", [])),
                          legacy["metrics"].get("total_return_pct"))
        assert results[logic][0] > 0, (
            f"{logic} genome took no trades — the parity assertion is vacuous")

    assert results["and"][0] < results["or"][0], (
        f"the AND genome should be strictly more selective: {results}")
    print(f"\n[item 1c] legacy==hybrid trade-for-trade and metric-for-metric; "
          f"trades AND={results['and'][0]} OR={results['or'][0]} "
          f"(>0 difference proves the field is read)")
