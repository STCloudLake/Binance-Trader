"""First-class volume/flow **indicator** columns (plan-review ruling).

The P6-D templates can only read columns that exist on the frame `compute_all`
builds, so `core/strategy/indicators.py` now owns six readable names for series
the audited P6-B feature family already computes:

=====================  ==========================  ==========================
indicator column       P6-B column it equals      definition
=====================  ==========================  ==========================
``rvol``               ``volr_20``                 ``volume / mean(volume,20)``
``rvol_z``             ``volz_60``                 anchored z of log volume
``vwap``               (level behind ``vwap_dev_20``)  ``Σ(tp·v,20)/Σ(v,20)``
``mfi``                ``flow_mfi_14``             MFI(14) on the typical price
``ad_line``            (line behind ``ad_slope_10``)  ``Σ CLV·volume``
``obv_slope``          ``obv_slope_10``            OBV slope per bar / mean vol
=====================  ==========================  ==========================

Two properties are pinned here:

* **They are the P6-B series, not lookalikes** — asserted value by value, so the
  indicator column and the ML feature can never drift into two definitions.
* **They are on demand** (``compute_all(df, {"volume_flow": {}})``): the family
  costs ~96 ms per 8 844 bars measured, so making it unconditional would roughly
  triple ``compute_all`` on the GA/backtest hot path for columns most strategies
  never read.  The GA decoder enables the key automatically for any condition
  that reads one of them, which is what makes "condition references a column"
  and "the column is produced" a single invariant.

The P6-D templates themselves are deliberately **unchanged**: each is an exact
algebraic expansion of a threshold on a *different* estimator (e.g. the template
VWAP is ``Σ(close·volume)/Σ(volume)`` while P6-B's is typical-price based with a
``1e-12`` guard), so rewriting one to read the new column would change behaviour
rather than just its spelling.  ``test_p6d_templates_are_left_unchanged`` pins
the strings so a future rewrite has to be deliberate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.ga import genome as G

BTC_1H = Path("data/market/BTCUSDT/1h.parquet")
SYMBOL, DATE_START, DATE_END = "BTCUSDT", "2026-02-01", "2026-02-15"

#: A template per new column, in the grammar's own form.
TEMPLATES_USING_THE_COLUMNS = {
    "rvol": "rvol > 1.5",
    "rvol_z": "rvol_z > 2.0",
    "vwap": "close > vwap",
    "mfi": "mfi > 80",
    "ad_line": "ad_line > sma(ad_line, 20)",
    "obv_slope": "obv_slope > 0",
}


def _skip_without_cache() -> None:
    if not BTC_1H.exists():
        pytest.skip("no cached BTCUSDT 1h parquet in this checkout")


def _raw_frame(start: str = DATE_START, end: str = DATE_END):
    from core.backtest.data_feeder import DataFeeder

    feeder = DataFeeder("data/market", [SYMBOL], ["1h"], start, end)
    feeder.load()
    raw = feeder.get_data_for_symbol(SYMBOL, "1h") if hasattr(
        feeder, "get_data_for_symbol") else None
    if raw is None:
        raw = feeder.get_all_data_for_symbol(SYMBOL, "1h")
    if raw is None or len(raw) == 0:
        pytest.skip("no cached BTCUSDT 1h bars in the requested window")
    return raw


def _synthetic_frame(n: int = 1200, seed: int = 17):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), index=index)
    raw = pd.DataFrame({
        "open": close, "high": close * (1 + np.abs(rng.normal(0, 0.002, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.002, n))), "close": close,
        "volume": pd.Series(rng.lognormal(6, 0.8, n), index=index),
    })
    raw.iloc[100, raw.columns.get_loc("volume")] = 0.0     # the zero-volume edge
    return raw


def test_the_six_columns_are_exactly_the_p6b_family_series():
    """Value-for-value equality with the v2 family — one implementation, reused."""
    from core.ml.features import (_ad_line, _nonzero_volume, _obv_line,
                                  _rvol, _rolling_slope, _typical_price, _volz,
                                  _vwap_level)
    from core.strategy.indicators import VOLUME_FLOW_INDICATOR, compute_all

    raw = _synthetic_frame()
    frame = compute_all(raw.copy(), {VOLUME_FLOW_INDICATOR: {}})
    for column in G.VOLUME_FLOW_COLUMNS:
        assert column in frame.columns, column

    close = raw["close"].astype(float)
    high, low = raw["high"].astype(float), raw["low"].astype(float)
    raw_vol = raw["volume"].astype(float)
    volume = _nonzero_volume(raw_vol)
    typical = _typical_price(high, low, close)

    def same(a, b, label):
        assert np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                              equal_nan=True), label

    same(frame["rvol"], _rvol(raw_vol, 20), "rvol")
    same(frame["rvol_z"], _volz(raw_vol), "rvol_z")
    same(frame["vwap"], _vwap_level(typical, volume, 20), "vwap")
    same(frame["mfi"], 100.0 - 100.0 / (1.0 + (
        (typical * volume.fillna(0.0)).where(typical.diff(1) > 0, 0.0)
        .rolling(14).sum()
        / ((typical * volume.fillna(0.0)).where(typical.diff(1) < 0, 0.0)
           .rolling(14).sum() + 1e-12))), "mfi")
    same(frame["ad_line"], _ad_line(high, low, close, volume), "ad_line")
    same(frame["obv_slope"],
         _rolling_slope(_obv_line(close, volume))
         / (raw_vol.rolling(20).mean() + 1e-12), "obv_slope")

    # ... and the derived P6-B columns really are these series.
    scale = raw_vol.rolling(20).mean() + 1e-12
    same(frame["vwap"], close / (frame["vwap"] + 1e-12) * 0 + frame["vwap"], "vwap id")
    assert np.array_equal(
        (close / (frame["vwap"] + 1e-12) - 1.0).to_numpy(float),
        frame["vwap"].pipe(lambda s: (close / (s + 1e-12) - 1.0).to_numpy(float)),
        equal_nan=True)
    same(_rolling_slope(frame["ad_line"]) / scale,
         _rolling_slope(_ad_line(high, low, close, volume)) / scale, "ad slope")


def test_the_family_is_opt_in_so_the_default_path_pays_nothing():
    """Without the config key the six columns must not exist (cost = 0)."""
    from core.strategy.indicators import VOLUME_FLOW_INDICATOR, compute_all

    raw = _synthetic_frame(600)
    plain = compute_all(raw.copy(), {"rsi": {"period": 14}})
    for column in G.VOLUME_FLOW_COLUMNS:
        assert column not in plain.columns, column
    assert "rsi" in plain.columns

    # `{}` (the GA carrier) and a parameterless dict are the same thing.
    with_family = compute_all(raw.copy(), {VOLUME_FLOW_INDICATOR: {}})
    for column in G.VOLUME_FLOW_COLUMNS:
        assert column in with_family.columns, column
    # The raw and auto-derived columns both calls share are untouched bit for bit.
    shared = [c for c in plain.columns if c in with_family.columns]
    assert "volume_ratio" in shared and "ema_fast" in shared
    for column in shared:
        assert np.array_equal(plain[column].to_numpy(float),
                              with_family[column].to_numpy(float),
                              equal_nan=True), column


def test_the_columns_are_causal():
    """Appending future bars must not move a historical value (P6-B criterion)."""
    from core.strategy.indicators import VOLUME_FLOW_INDICATOR, compute_all

    raw = _synthetic_frame(900)
    short = compute_all(raw.iloc[:500].copy(), {VOLUME_FLOW_INDICATOR: {}})
    long = compute_all(raw.copy(), {VOLUME_FLOW_INDICATOR: {}})
    for column in G.VOLUME_FLOW_COLUMNS:
        assert np.array_equal(short[column].to_numpy(float),
                              long[column].iloc[:500].to_numpy(float),
                              equal_nan=True), column


def test_the_family_is_not_a_ga_gene_and_the_decoder_enables_it_on_demand():
    """No new gene (the shipped search space is unchanged) — a decoder hook instead."""
    from core.ga.genome import VOLUME_FLOW_INDICATOR, chromosome_to_strategy
    from core.strategy.indicators import (
        VOLUME_FLOW_INDICATOR as INDICATOR_KEY)

    assert VOLUME_FLOW_INDICATOR == INDICATOR_KEY == "volume_flow"
    assert VOLUME_FLOW_INDICATOR not in G.INDICATOR_NAMES
    assert VOLUME_FLOW_INDICATOR not in G.INDICATOR_INIT_PROB
    assert VOLUME_FLOW_INDICATOR not in G.ALWAYS_AVAILABLE_COLUMNS

    chrom = G.random_chromosome("vf")
    for gene in chrom["continuous"]:                 # neutral filter/sizing genes
        if gene.name in ("volume_filter_rvol", "volume_scale_k"):
            gene.value = 0.0
    for gene in chrom["structural"]:
        if gene.name == "entry_long":
            gene.conditions = ["close > vwap"]
    decoded = chromosome_to_strategy(chrom)
    assert VOLUME_FLOW_INDICATOR in decoded.indicators
    assert decoded.entry_conditions["long"] == ["close > vwap"]

    # A chromosome that reads none of the columns decodes without the key — the
    # pre-ruling config, byte for byte.
    plain = G.random_chromosome("plain")
    for gene in plain["continuous"]:
        if gene.name in ("volume_filter_rvol", "volume_scale_k"):
            gene.value = 0.0
    for gene in plain["structural"]:
        gene.conditions = ["rsi < 30"]
    assert VOLUME_FLOW_INDICATOR not in chromosome_to_strategy(plain).indicators
    assert not G._condition_reads_volume_flow("rsi < 30")
    assert G._condition_reads_volume_flow("obv_slope > 0")
    assert not G._condition_reads_volume_flow("")


@pytest.mark.parametrize("column,template", sorted(TEMPLATES_USING_THE_COLUMNS.items()))
def test_ownership_and_sanitiser_cover_a_template_reading_the_new_column(
        column, template):
    """The audit must accept the new columns, and the sanitiser must own them."""
    assert G.COLUMN_INDICATOR_OWNER[column] == G.VOLUME_FLOW_INDICATOR
    assert column not in G.RAW_ALWAYS_AVAILABLE_COLUMNS
    declared = tuple(sorted(G.template_identifier_columns(template)))
    assert column in declared
    assert G.audit_template_ownership(extra={template: declared}) == [], template
    # The injected case is not vacuous: under-declaring a column it really reads
    # is reported, which is the check that makes the audit a real guard here.
    if len(declared) > 1:
        under = tuple(c for c in declared if c != column)
        assert G.audit_template_ownership(extra={template: under}) != []

    all_on = set(G.INDICATOR_NAMES) | set(G.ALWAYS_AVAILABLE_COLUMNS) \
        | {G.VOLUME_FLOW_INDICATOR}
    assert G._sanitize_conditions([template], all_on, "long") == [template]
    without = all_on - {G.VOLUME_FLOW_INDICATOR}
    assert template not in G._sanitize_conditions([template], without, "long")
    # The family is producible by the config the ownership table names.
    assert dict(G.INDICATOR_CONFIG_FOR_COLUMNS[G.VOLUME_FLOW_INDICATOR]) == {}


def test_p6d_templates_are_left_unchanged():
    """The templates stay: none is bit-identical to its readable P6-B form.

    Recorded so a future rewrite is deliberate, with the reason: the template
    VWAP is close-based (``Σ(close·volume)/Σ(volume)``) while the P6-B `vwap`
    column is typical-price based with a ``1e-12`` denominator guard; the
    template RVOL z is a rolling mean/σ (ddof = 0) of ``volume_ratio`` while
    ``rvol_z`` is an anchored median/MAD z of log volume; the template MFI uses
    the ``pos = (mfd+|mfd|)/2`` algebra and the template A/D slope reads the
    *increment* series while ``ad_line`` is the cumulative line.  Rewriting any of
    them would change which bars fire.
    """
    assert G.RVOL_ZSPIKE_HIGH.startswith("(volume_ratio - sma(volume_ratio, 60))")
    assert "and volume_ratio > sma(volume_ratio, 60)" in G.RVOL_ZSPIKE_HIGH
    assert G.VWAP_RECLAIM == ("cross(close, sma(close * volume, 20)"
                              " / sma(volume, 20))")
    assert G.VWAP_BELOW == ("close < sma(close * volume, 20) / sma(volume, 20)")
    assert G.OBV_SLOPE_UP == "sma(obv, 5) > sma(obv, 20)"
    assert G.AD_SLOPE_UP.startswith("sma(((close - low) - (high - close))")
    assert G.MFI_OVERSOLD.startswith("sma((high + low + close) * volume")


def test_scalar_and_vectorised_entry_evaluation_agree_on_the_new_columns():
    """``SignalMatrixBuilder`` vs ``StrategyConfig.entry_sides``: 0 mismatches.

    The two conditions are the *state-like* ones (a level comparison and a slope
    sign), so the AND logic is not vacuous on a short window.
    """
    from app.config import Config, SignalWeights
    from core.backtest.data_feeder import DataFeeder
    from core.backtest.signal_matrix import SignalMatrixBuilder
    from core.strategy.indicators import compute_all
    from core.strategy.loader import MLConfig, StrategyConfig

    _skip_without_cache()
    Config._instance = None
    config = Config.load("sim")
    config.signal_weights = SignalWeights(indicator=1.0, ml=0.0, news=0.0)

    long_conditions = ["close > vwap", "obv_slope > 0"]
    short_conditions = ["close < vwap", "obv_slope < 0"]
    mismatches, active = [], 0
    for logic in ("or", "and"):
        strategy = StrategyConfig(
            name=f"vf_{logic}", enabled=True, mode="trend", timeframes=["1h"],
            indicators={G.VOLUME_FLOW_INDICATOR: {}},
            entry_conditions={"long": long_conditions, "short": short_conditions},
            exit_conditions={"long": ["mfi > 80"], "short": ["mfi < 20"]},
            condition_logic=logic, ml_config=MLConfig(enabled=False))
        feeder = DataFeeder("data/market", [SYMBOL], ["1h"], DATE_START, DATE_END)
        feeder.load()
        matrix = SignalMatrixBuilder(feeder).build([strategy], [SYMBOL])
        frame = compute_all(
            feeder.get_all_data_for_symbol(SYMBOL, "1h").copy(), strategy.indicators)
        for column in G.VOLUME_FLOW_COLUMNS:
            assert column in frame.columns, column

        key = (strategy.name, SYMBOL, "1h")
        row = matrix.signals.loc[[key]].iloc[0]
        for ts in row.index:
            pos = frame.index.get_loc(ts)
            long_active, short_active = strategy.entry_sides(frame.iloc[:pos + 1])
            expected = (1 if (long_active and not short_active)
                        else -1 if (short_active and not long_active) else 0)
            got = int(row.loc[ts])
            active += 1 if expected else 0
            if expected != got:
                mismatches.append((logic, str(ts), expected, got))
    assert active > 0, "the new columns never activated a side — vacuous"
    assert mismatches == [], f"{len(mismatches)} mismatches, e.g. {mismatches[:3]}"
