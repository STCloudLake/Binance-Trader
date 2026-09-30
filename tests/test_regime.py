"""Regression tests for phase P4 — regime detection and gating.

The detector claims are tested on a **synthetic series with a known regime
structure** (three regimes: calm/up, stressed/down, calm/up) so accuracy and, as
importantly, **latency** are measured rather than asserted by construction:

* the 2-state Gaussian HMM (numpy EM + Viterbi) recovers the known volatilities
  and separates the regimes with ≥ 95 % accuracy and a latency of a few bars;
* the tercile classifier is causal (its thresholds come from the *past* only)
  and the trend filter flags the known direction;
* both are deterministic, bar for bar, on repeated runs;
* the gate is **off by default** and allows everything when off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.strategy.regime import (
    DEFAULT_REGIME_MAP, HMM_TOL, REGIME_GATING_ENABLED, RegimeGate,
    classify_last, classify_regimes, default_gate, detection_metrics,
    hmm_two_state, regime_persistence, rolling_volatility, trend_regimes,
    volatility_terciles,
)


def _synth(n: int = 3000, *, seed: int = 5, change_points=(1000, 2000)):
    """Three regimes: (σ=0.002, drift +) → (σ=0.01, drift −) → (σ=0.002, +)."""
    rng = np.random.default_rng(seed)
    spec = [(0.002, +0.0004), (0.010, -0.0008), (0.002, +0.0004)]
    bounds = (0, change_points[0], change_points[1], n)
    rets, truth, trend = [], [], []
    for k, (sigma, drift) in enumerate(spec):
        m = bounds[k + 1] - bounds[k]
        rets.append(rng.standard_normal(m) * sigma + drift)
        truth += ["high" if k == 1 else "low"] * m
        trend += ["up" if k != 1 else "down"] * m
    r = np.concatenate(rets)
    close = 100.0 * np.exp(np.cumsum(r))
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": 1.0}, index=idx)
    return df, np.asarray(truth), np.asarray(trend), tuple(change_points)


# ── volatility / trend classifiers ──────────────────────────────────────

def test_rolling_volatility_is_the_past_window_only():
    r = pd.Series(np.arange(10, dtype=float))
    vol = rolling_volatility(r, window=3, min_periods=3)
    assert vol.iloc[:2].isna().all()
    assert vol.iloc[3] == pytest.approx(np.std([1, 2, 3], ddof=1))


def test_volatility_terciles_use_expanding_thresholds_from_the_past_only():
    df, truth, _, _ = _synth(seed=5)
    vol = rolling_volatility(np.log(df["close"]).diff())
    labels = volatility_terciles(np.log(df["close"]).diff())
    # The label at t must be reproducible from vol[:t] alone: recomputing on a
    # truncated series must give the same label at that bar.
    for t in (400, 1200, 2400):
        truncated = volatility_terciles(np.log(df["close"]).diff().iloc[:t + 1])
        assert truncated.iloc[t] == labels.iloc[t]
    # A 100x volatility jump must show up as "high" rather than "mid", and the
    # labels are *relative to the past* by construction.  Inside a homogeneous
    # block the three labels split roughly a third each — the thresholds ARE
    # that sample's own quantiles — which is exactly the documented limitation
    # ("high" means high for this sample, not high in absolute terms).
    rng = np.random.default_rng(3)
    scales = np.r_[np.full(500, 1e-5), np.full(500, 1e-4), np.full(500, 1e-2)]
    jumped = volatility_terciles(pd.Series(rng.standard_normal(1500) * scales))
    assert (jumped.iloc[-100:] == "high").mean() > 0.9
    assert (jumped.iloc[-100:] == "low").mean() == 0.0
    first_block = jumped.iloc[100:450]
    assert 0.2 < (first_block == "low").mean() < 0.9      # 1/3-ish, not decisive
    assert (first_block == "high").mean() < (jumped.iloc[600:900] == "high").mean()
    assert set(labels.unique()) <= {"low", "mid", "high", "unknown"}
    assert labels.iloc[0] == "unknown"
    assert len(vol) == len(labels)


def test_trend_regimes_flag_a_known_uptrend_and_downtrend():
    df, _, trend, _ = _synth(seed=5)
    labels = trend_regimes(df["close"], fast=50, slow=200)
    # The EMAs need ~200 bars to line up, so the first ~300 bars of each regime
    # are legitimately labelled "range"; the rest must follow the drift.
    up_share = (labels.iloc[300:950] == "trend_up").mean()
    down_share = (labels.iloc[1300:1950] == "trend_down").mean()
    assert up_share > 0.7
    assert down_share > 0.7
    assert set(labels.unique()) <= {"trend_up", "trend_down", "range"}


# ── HMM ─────────────────────────────────────────────────────────────────

def test_hmm_recovers_the_known_sigmas_and_is_deterministic():
    df, truth, _, _ = _synth(seed=5)
    logret = np.log(df["close"]).diff()
    a = hmm_two_state(logret)
    b = hmm_two_state(logret)
    assert np.array_equal(a["states"], b["states"])
    assert a["loglik"] == b["loglik"]
    # State 0 is always the calm one (relabelled by sigma).
    assert a["sigma"][0] < a["sigma"][1]
    assert a["sigma"][0] == pytest.approx(0.002, rel=0.35)
    assert a["sigma"][1] == pytest.approx(0.010, rel=0.25)
    assert a["mu"][1] < 0 < a["mu"][0]
    assert a["transition"][0, 0] > 0.95          # regimes are persistent
    assert a["n_iter"] <= 50
    assert a["loglik"] > 0
    assert a["posterior_filtered"].shape == (len(logret.dropna()), 2)
    assert np.allclose(a["posterior_filtered"].sum(axis=1), 1.0, atol=1e-9)


def test_hmm_detection_accuracy_and_latency_on_a_known_structure():
    df, truth, _, cp = _synth(seed=5)
    logret = np.log(df["close"]).diff()
    fit = hmm_two_state(logret)
    predicted = np.where(fit["states"] == 1, "high", "low")
    metrics = detection_metrics(truth[1:], predicted, positive="high",
                                change_points=(cp[0] - 1, cp[1] - 1))
    assert metrics["accuracy"] > 0.95
    assert all(l is not None and l <= 10 for l in metrics["latency_bars"])
    # The smoothed (whole-sample) posterior is a different object and must not
    # be presented as the tradeable signal.
    import core.strategy.regime as regime
    assert "posterior_smoothed" in fit
    assert fit["posterior_smoothed"].shape[0] == fit["posterior_filtered"].shape[0]
    assert regime.__doc__.find("smoothed") > 0


def test_hmm_handles_a_short_sample_without_crashing():
    short = pd.Series(np.random.default_rng(0).standard_normal(20) * 0.01)
    fit = hmm_two_state(short)
    assert "too few rows" in fit["note"]
    assert (fit["state"] == "unknown").all()
    assert fit["states"].shape == (20,)


def test_hmm_bounds_the_sample_it_fits():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.standard_normal(5000) * 0.01)
    fit = hmm_two_state(r, max_rows=1000)
    assert len(fit["state"]) == 1000
    assert fit["state"].index[0] == r.index[-1000]


# ── the composite table + persistence ───────────────────────────────────

def test_classify_regimes_aligns_the_hmm_series_and_labels_the_composite():
    df, _, _, _ = _synth(seed=6)
    table = classify_regimes(df, with_hmm=True)
    assert len(table) == len(df)
    assert table.index.equals(df.index)
    assert table["hmm_prob_stressed"].iloc[0] != table["hmm_prob_stressed"].iloc[0] \
        or np.isnan(table["hmm_prob_stressed"].iloc[0])   # leading NaN, no shift
    assert set(table["regime"].unique()) <= {
        "trend_up", "trend_down", "range_low", "range_mid", "range_high",
        "range_unknown"}
    last = classify_last(df, with_hmm=True)
    assert last["index"] == str(df.index[-1])   # JSON-safe, not a Timestamp
    assert isinstance(last["regime"], str)
    assert last["hmm_state"] in {"calm", "stressed", "unknown"}


def test_regime_persistence_measures_run_lengths():
    series = ["calm"] * 10 + ["stressed"] * 5 + ["calm"] * 3
    res = regime_persistence(series)
    assert res["n_runs"] == 3
    assert res["max_run"] == 10
    assert res["by_regime"]["calm"] == pytest.approx((10 + 3) / 2)
    assert regime_persistence([])["n_runs"] == 0


def test_detection_metrics_reports_latency_not_just_accuracy():
    truth = ["low"] * 100 + ["high"] * 100
    # Perfect after a 10-bar lag: accuracy is still high, latency is 10.
    predicted = ["low"] * 110 + ["high"] * 90
    res = detection_metrics(truth, predicted, positive="high",
                            change_points=(100,))
    assert res["accuracy"] == pytest.approx(0.95)
    assert res["latency_bars"] == [10]
    never = detection_metrics(truth, ["low"] * 200, positive="high",
                              change_points=(100,))
    assert never["latency_bars"] == [None]


# ── the gate ────────────────────────────────────────────────────────────

def test_gate_is_disabled_by_default_and_allows_everything():
    assert REGIME_GATING_ENABLED is False
    gate = RegimeGate(allowed={"breakout": {"trend_up"}})
    assert gate.enabled is False
    assert gate.allows("breakout", "range_high") is True
    assert gate.allows("anything", "anything") is True
    assert gate.size_multiplier("breakout", "range_high") == 1.0
    assert gate.blocked_kinds("range_high") == []


def test_gate_is_deterministic_and_documented_when_enabled():
    gate = default_gate(enabled=True)
    assert gate.allows("breakout", "trend_up") is True
    assert gate.allows("breakout", "range_high") is False
    assert gate.allows("pairs", "range_high") is True
    assert gate.allows("trend", "range_mid") is False
    # A kind with no entry in the map keeps the historical behaviour.
    assert gate.allows("unmapped_kind", "range_high") is True
    assert "breakout" in gate.blocked_kinds("range_high")
    assert gate.size_multiplier("breakout", "range_high", base=1.0,
                                blocked=0.0) == 0.0
    for kind, regimes in DEFAULT_REGIME_MAP.items():
        for regime in regimes:
            assert gate.allows(kind, regime) is True


def test_gate_defaults_and_unknown_regimes_fall_back_safely():
    gate = RegimeGate(allowed={}, enabled=True, default_allow=False)
    assert gate.allows("anything", "range_low") is False
    assert gate.allows("anything", "range_unknown") is False
    loose = RegimeGate(allowed={}, enabled=True)
    assert loose.allows("anything", "range_unknown") is True


def test_tolerance_constant_is_the_documented_one():
    assert HMM_TOL == 1e-6
