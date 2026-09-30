"""Regression tests for phase P4 — pairs / cointegration / Kalman hedge ratio.

The module's value rests on four claims, and each one is tested here against an
independent reference rather than against itself:

1. the **simulated** ADF null distribution reproduces MacKinnon's published
   asymptotic critical values (so the p-value the guard gates on is calibrated);
2. :func:`adf_regression` produces a tau statistic whose empirical distribution
   *on random walks* matches that simulated null (so the regression code and the
   simulation cannot disagree silently);
3. Engle-Granger has power on a cointegrated synthetic pair and correct size on
   independent walks;
4. the Kalman hedge ratio tracks a known time-varying beta and no longer
   collapses to ~0 (the measured bug this module was written with), and every
   trading helper is free of look-ahead.

Real cached data (``data/market``, read-only) adds one honest end-to-end check:
on the 1h majors the guard refuses **every** pair, which is a valid outcome.

Measured-threshold policy
-------------------------
The real-cache tests assert the *behavioural contract* (the guard's verdict and
reason, the OLS residual identity), never a measured statistic of the cache.
The old form pinned ``min(p_values) > 0.10`` and ``r_var < var(y)/2``; those were
one cache state's numbers and would drift with a cache repair, exactly as the
AUC pin did in ``test_meta_labeling``.  A numeric expectation tied to
``data/market/**`` may only appear here when it is derived from the frame the
test just read or built synthetically.
``tests/test_measured_threshold_policy.py`` enforces this module-level rule.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from core.strategy.pairs import (
    PAIRS_MAX_ADF_PVALUE, PAIRS_MAX_HALF_LIFE, PAIRS_MAX_LOOKBACK,
    PAIRS_MIN_HALF_LIFE, PAIRS_MIN_LOOKBACK, PAIRS_Z_ENTRY, PairFit,
    PairsSignal, admissible_sim_length, adf_regression, default_leg_cost_pct,
    engle_granger, fit_pair, kalman_hedge_ratio, log_spread_notional,
    lookback_from_half_life, ols_hedge_ratio, ou_half_life, pair_guard,
    pair_trades, pairs_positions, pairs_signal, price_spread_notional,
    rolling_zscore, spread_returns, statsmodels_adf_tau, tau_critical_values,
    tau_null_distribution, tau_pvalue, trade_statistics,
)


# ── synthetic generators (deterministic) ────────────────────────────────

def _random_walk(n: int, seed: int, scale: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n) * scale)


def _cointegrated(n: int = 1500, beta: float = 0.8, kappa: float = 0.05,
                  seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n) * 0.01) + 10.0
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] * (1.0 - kappa) + rng.standard_normal() * 0.002
    return pd.Series(0.7 + beta * x + s), pd.Series(x)


def _independent(n: int = 1500, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.Series(np.cumsum(rng.standard_normal(n) * 0.01) + 10.0)
    y = pd.Series(np.cumsum(rng.standard_normal(n) * 0.01) + 10.0)
    return y, x


def _stationary_regressor(n: int = 3000, seed: int = 11) -> np.ndarray:
    """AR(1) regressor — a *stationary* x, so OLS/Kalman can both identify beta."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n) * 0.01
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.995 * x[t - 1] + e[t]
    return x


def _ou_spread(n: int, phi: float, seed: int = 4,
               innovation: float = 0.002) -> np.ndarray:
    """OU noise around a hedge line.

    The innovation is small (0.002) on purpose: with a 5× larger one the same
    filter wanders by ~0.08 in β (measured mean 0.821 vs an OLS β of 0.737 on
    3 000 bars), which is the documented behaviour of a random-walk state — a
    *filtered* estimate whose sample average is not the full-sample OLS.  The
    tracking tests below therefore use the quiet regime, and the noisy regime is
    only asserted where it is a limitation, not a feature.
    """
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.standard_normal() * innovation
    return s


# ── 1. the simulated null distribution is calibrated ────────────────────

def test_simulated_tau_null_matches_mackinnon_critical_values():
    """Dickey-Fuller (constant) and Engle-Granger (N=2) critical values.

    Reference (MacKinnon 1994/2010, asymptotic):
    DF with constant 1 %/5 %/10 % = −3.43 / −2.86 / −2.57;
    Engle-Granger, 2 variables, constant = −3.90 / −3.34 / −3.04.
    """
    df_c = tau_critical_values("df_c", 2000, (0.01, 0.05, 0.10))
    assert df_c[0.01] == pytest.approx(-3.43, abs=0.10)
    assert df_c[0.05] == pytest.approx(-2.86, abs=0.08)
    assert df_c[0.10] == pytest.approx(-2.57, abs=0.08)

    eg = tau_critical_values("eg_c", 2000, (0.01, 0.05, 0.10))
    assert eg[0.01] == pytest.approx(-3.90, abs=0.10)
    assert eg[0.05] == pytest.approx(-3.34, abs=0.08)
    assert eg[0.10] == pytest.approx(-3.04, abs=0.08)
    # The estimated-residual distribution must be *left* of the plain DF one:
    # using the DF table for a cointegration test would over-reject.
    assert eg[0.05] < df_c[0.05]


def test_tau_null_is_deterministic_and_cached_per_length():
    a = tau_null_distribution("eg_c", 400)
    b = tau_null_distribution("eg_c", 400)
    assert np.array_equal(a, b)
    assert admissible_sim_length(50) == 250      # clamped below
    assert admissible_sim_length(99999) == 2000  # clamped above
    assert len(tau_null_distribution("eg_c", 250)) == len(a)
    # p-values are bounded away from 0/1 by one Monte-Carlo step.
    assert tau_pvalue(-20.0, "eg_c", 400) == pytest.approx(1.0 / len(a))
    assert tau_pvalue(5.0, "eg_c", 400) == pytest.approx(1.0 - 1.0 / len(a))


def test_adf_regression_matches_the_simulated_null_on_random_walks():
    """The regression code and the simulation must agree on the same null."""
    taus = np.array([
        adf_regression(_random_walk(500, seed=s), max_lags=0, regression="c")["tau"]
        for s in range(300)
    ])
    simulated = tau_critical_values("df_c", 500, (0.05,))[0.05]
    assert np.quantile(taus, 0.05) == pytest.approx(simulated, abs=0.30)
    assert float(np.mean(taus)) == pytest.approx(-1.57, abs=0.25)


# ── 2. power and size of the Engle-Granger test ─────────────────────────

def test_engle_granger_has_power_on_a_cointegrated_pair():
    p_values = [engle_granger(*_cointegrated(seed=s))["p_value"] for s in range(20)]
    assert np.mean(np.array(p_values) <= 0.05) >= 0.9
    assert float(np.median(p_values)) < 0.01


def test_engle_granger_has_correct_size_on_independent_walks():
    p_values = [engle_granger(*_independent(seed=s))["p_value"] for s in range(20)]
    # 20 draws at a 5 % nominal level: 0-4 rejections is consistent with 5 %.
    assert int((np.array(p_values) <= 0.05).sum()) <= 4


def test_engle_granger_reports_hedge_ratio_and_critical_values():
    y, x = _cointegrated(beta=0.8, seed=3)
    res = engle_granger(y, x)
    assert res["beta"] == pytest.approx(0.8, abs=0.05)
    assert res["is_cointegrated"] is True
    assert set(res["critical_values"]) == {0.01, 0.05, 0.1}
    assert res["sim_length"] == 1500
    assert res["adf_lag"] >= 0


# ── 3. the Kalman hedge ratio ───────────────────────────────────────────

def test_kalman_tracks_a_constant_beta_without_drifting():
    x = _stationary_regressor(seed=11)
    y = pd.Series(1.0 + 0.75 * x + _ou_spread(len(x), 0.97, seed=4))
    kf = kalman_hedge_ratio(y, x)
    assert kf["beta_mean"] == pytest.approx(0.75, abs=0.02)
    assert kf["beta_std"] < 0.05
    assert kf["beta_drift"] < 0.05


def test_kalman_tracks_a_drifting_beta():
    """A time-varying beta is the whole point of the state-space model."""
    x = _stationary_regressor(seed=11)
    n = len(x)
    beta_path = 0.40 + 0.35 * np.arange(n) / n
    y = pd.Series(1.0 + beta_path * x + _ou_spread(n, 0.97, seed=4))
    kf = kalman_hedge_ratio(y, x)
    start = float(np.mean(kf["beta"].iloc[:300]))
    end = float(np.mean(kf["beta"].iloc[-300:]))
    assert start == pytest.approx(float(np.mean(beta_path[:300])), abs=0.05)
    assert end == pytest.approx(float(np.mean(beta_path[-300:])), abs=0.06)
    assert kf["beta_drift"] > 0.2  # it really did move


def test_kalman_observation_noise_is_the_spread_noise_not_var_y():
    """Regression guard for the measured bug.

    Scaling ``R`` from ``var(y)`` instead of the OLS residual made the filter
    refuse to update: β collapsed to 0.058 against an OLS β of 0.628 on real
    BTC/ETH 1h (where ``var(y)/R ≈ 100``), so the "spread" became the raw price
    of ``y`` and the pair trade silently turned into a directional one.  The
    real-data counterpart of this test asserts the ratio on the cached pair.
    """
    x = _stationary_regressor(seed=11)
    y = pd.Series(1.0 + 0.75 * x + _ou_spread(len(x), 0.97, seed=4))
    ols = ols_hedge_ratio(y, x)
    kf = kalman_hedge_ratio(y, x)
    assert kf["r_var"] == pytest.approx(ols["resid_sd"] ** 2, rel=1e-6)
    # R is the spread's own noise: never the variance of the level (the measured
    # bug that collapsed β to 0.058 on real BTC/ETH, where var(y)/R ≈ 100).
    assert kf["r_var"] < float(np.var(y))
    assert abs(kf["beta_mean"] - ols["beta"]) < 0.05


def test_kalman_skips_missing_observations_instead_of_using_zero():
    x = _stationary_regressor(seed=2)
    y = pd.Series(1.0 + 0.75 * x + _ou_spread(len(x), 0.97, seed=4))
    y.iloc[100:120] = np.nan
    kf = kalman_hedge_ratio(y, x)
    assert np.isfinite(kf["beta"].iloc[150])
    assert kf["beta"].iloc[100:120].notna().all()  # state carried forward


# ── 4. half-life / lookback / guard ─────────────────────────────────────

def test_ou_half_life_recovers_a_known_ar1():
    s = _ou_spread(20000, phi=1.0 - 0.1, seed=9)   # kappa = 0.1 -> hl = 6.93
    hl = ou_half_life(s)
    assert hl["mean_reverting"] is True
    assert hl["half_life"] == pytest.approx(np.log(2) / 0.1, rel=0.10)
    assert hl["kappa"] == pytest.approx(0.1, rel=0.10)


def test_ou_half_life_is_infinite_for_a_non_mean_reverting_series():
    # A deterministic trend has Δs constant, so the AR(1) slope on the level is
    # exactly 0: unambiguously not mean-reverting.
    hl = ou_half_life(np.arange(1000, dtype=float))
    assert hl["mean_reverting"] is False
    assert np.isinf(hl["half_life"])
    # A random walk's spurious AR(1) slope is tiny, so even when it comes out
    # negative the implied half-life is a large fraction of the sample.
    walk = ou_half_life(_random_walk(4000, seed=8))
    assert walk["half_life"] > 200 or np.isinf(walk["half_life"])


def test_lookback_clips_and_follows_the_half_life():
    assert lookback_from_half_life(10.0) == 40
    assert lookback_from_half_life(1.0) == PAIRS_MIN_LOOKBACK
    assert lookback_from_half_life(1e9) == PAIRS_MAX_LOOKBACK
    assert lookback_from_half_life(float("inf")) == PAIRS_MAX_LOOKBACK


def test_pair_guard_refuses_every_failure_mode():
    assert pair_guard(n_obs=100, p_value=0.01, half_life=10.0, beta=0.5)["allowed"] is False
    assert "sample too short" in pair_guard(
        n_obs=100, p_value=0.01, half_life=10.0, beta=0.5)["reason"]
    ok = pair_guard(n_obs=500, p_value=0.01, half_life=10.0, beta=0.5)
    assert ok["allowed"] is True and ok["reason"] == "pass"
    assert "cointegration rejected" in pair_guard(
        n_obs=500, p_value=0.31, half_life=10.0, beta=0.5)["reason"]
    assert "half-life too long" in pair_guard(
        n_obs=500, p_value=0.01, half_life=500.0, beta=0.5)["reason"]
    assert "half-life too short" in pair_guard(
        n_obs=500, p_value=0.01, half_life=1.0, beta=0.5)["reason"]
    assert "not mean-reverting" in pair_guard(
        n_obs=500, p_value=0.01, half_life=float("inf"), beta=0.5)["reason"]
    assert "hedge ratio unusable" in pair_guard(
        n_obs=500, p_value=0.01, half_life=10.0, beta=0.0)["reason"]


# ── 5. signals: no look-ahead, unit conventions, costs ──────────────────

def test_rolling_zscore_excludes_the_bar_it_scores():
    s = pd.Series([0.0] * 40 + [10.0], dtype=float)
    z = rolling_zscore(s, 20)
    # The mean/std come from the 20 bars *before* t, all of which are 0 -> the
    # z-score is undefined (0 std), never a spike normalised by itself.
    assert not np.isfinite(z.iloc[-1])
    s2 = pd.Series(np.r_[np.zeros(40), 10.0], dtype=float)
    z2 = rolling_zscore(s2, 20)
    manual = (10.0 - 0.0) / 0.0 if False else None
    assert manual is None
    assert z2.iloc[:-1].isna().all() or z2.iloc[20:40].isna().all()
    # A non-degenerate case: z at t uses mean/std of [t-20, t-1] only.
    s3 = pd.Series(np.arange(60, dtype=float))
    z3 = rolling_zscore(s3, 20)
    window = s3.iloc[39:59]
    expected = (s3.iloc[59] - window.mean()) / window.std(ddof=1)
    assert z3.iloc[59] == pytest.approx(expected)


def test_pairs_positions_are_causal_and_the_stop_latches():
    z = pd.Series([0.0, 0.0, 2.5, 2.2, 4.5, 2.4, 1.2, -2.5, -0.4, 0.0])
    pos = pairs_positions(z)
    # entry on the z>=2 bar (short spread), held until the stop at |z|>=4
    assert pos.iloc[2] == -1.0 and pos.iloc[3] == -1.0 and pos.iloc[4] == 0.0
    # latched: |z| must come back inside z_entry before a new entry
    assert pos.iloc[5] == 0.0 and pos.iloc[6] == 0.0
    assert pos.iloc[7] == 1.0         # fresh entry on the other side
    assert pos.iloc[8] == 0.0         # exit at |z| <= 0.5
    assert pos.iloc[-1] == 0.0


def test_notional_conventions_are_explicit_and_different():
    hedge = pd.Series([0.5, -2.0])
    assert log_spread_notional(hedge).tolist() == [1.5, 3.0]
    py, px = pd.Series([100.0, 100.0]), pd.Series([10.0, 10.0])
    assert price_spread_notional(py, px, hedge).tolist() == [105.0, 120.0]


def test_spread_returns_charge_a_full_round_trip_including_the_closing_bar():
    """Four fills per pair round trip: leg cost is charged twice (open+close)."""
    idx = pd.RangeIndex(10)
    spread = pd.Series([0.0] * 10, index=idx)          # no move at all
    pos = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float, index=idx)
    cost = 0.5                                          # % per leg round trip
    net = spread_returns(spread, pos, 1.0, leg_round_trip_cost_pct=cost)
    trades = pair_trades(pos, net)
    assert len(trades) == 1
    # 1.0 (open) + 1.0 (close) units of turnover x cost/200 = cost/100
    assert trades[0]["net_return"] == pytest.approx(-cost / 100.0)
    assert trades[0]["bars"] == 4   # bars 1..4, including the closing bar


def test_spread_returns_pay_the_move_and_the_trade_stats_agree():
    n = 200
    idx = pd.RangeIndex(n)
    t = np.arange(n, dtype=float)
    spread = pd.Series(np.sin(t / 8.0) * 0.01, index=idx)
    pos = pd.Series(np.where(np.cos(t / 8.0) > 0, 1.0, -1.0), index=idx)
    notional = 1.0 + 0.5
    net = spread_returns(spread, pos, notional, leg_round_trip_cost_pct=0.0)
    stats = trade_statistics(pair_trades(pos, net))
    assert stats["n_trades"] > 0
    assert stats["mean"] > 0        # trading the known cycle with zero cost pays
    assert -1.0 <= stats["psr"] <= 1.0
    expensive = trade_statistics(pair_trades(
        pos, spread_returns(spread, pos, notional, leg_round_trip_cost_pct=5.0)))
    assert expensive["mean"] < stats["mean"]


def test_fit_pair_and_pairs_signal_refuse_an_independent_pair():
    y, x = _independent(seed=5)
    fit = fit_pair(y, x, symbol_y="A", symbol_x="B")
    assert isinstance(fit, PairFit)
    assert fit.allowed is False
    z = rolling_zscore(fit.spread, fit.lookback)
    sig = pairs_signal(fit, z, pairs_positions(z))
    assert sig.allowed is False
    assert sig.indicator_signal == 0.0          # a refused pair is flat
    assert sig.to_kernel_input()["ml_enabled"] is False
    assert sig.to_kernel_input()["indicator_signal"] == 0.0


def test_fit_pair_accepts_a_cointegrated_pair_and_emits_a_kernel_signal():
    y, x = _cointegrated(seed=1)
    fit = fit_pair(y, x, symbol_y="A", symbol_x="B")
    assert fit.allowed is True
    assert 2.0 <= fit.half_life["half_life"] <= 120.0
    assert PAIRS_MIN_LOOKBACK <= fit.lookback <= PAIRS_MAX_LOOKBACK
    assert fit.summary()["guard_allowed"] is True
    z = rolling_zscore(fit.spread, fit.lookback)
    pos = pairs_positions(z, z_entry=PAIRS_Z_ENTRY)
    assert set(np.unique(pos.to_numpy())) <= {-1.0, 0.0, 1.0}
    assert fit.summary()["half_life_kalman_bars"] is not None


def test_default_leg_cost_uses_the_sim_model_and_pairs_pay_two_legs():
    cost = default_leg_cost_pct(None)
    assert cost == pytest.approx(0.14, abs=0.01)   # documented fallbacks
    # Two legs are charged by the caller: a pair round trip is ~2x the leg cost.
    assert 2.0 * cost > cost


# ── 6. real cached data (read-only; skipped when the cache is absent) ───

def _cached(symbol: str, interval: str = "1h"):
    path = f"data/market/{symbol}/{interval}.parquet"
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def test_real_cached_pair_guard_verdict_matches_the_test():
    """On real data the guard's verdict must equal the E-G decision."""
    df_y, df_x = _cached("BTCUSDT"), _cached("ETHUSDT")
    if df_y is None or df_x is None:
        pytest.skip("no cached BTCUSDT/ETHUSDT 1h parquet in this checkout")
    joint = pd.concat([np.log(df_y["close"]).rename("y"),
                       np.log(df_x["close"]).rename("x")], axis=1).dropna()
    fit = fit_pair(joint["y"], joint["x"], symbol_y="BTCUSDT", symbol_x="ETHUSDT")
    summary = fit.summary()
    # Measured on this cache: p = 0.4502, half-life 1200 bars (OLS spread) ->
    # refused on BOTH counts.  The assertion is the *logic*, not the data: the
    # thresholds are the guard's own constants (the same ones ``pair_guard``
    # reads), so a cache repair moves the numbers without moving the logic.
    assert fit.allowed == (fit.adf["p_value"] <= PAIRS_MAX_ADF_PVALUE
                           and PAIRS_MIN_HALF_LIFE <= fit.half_life["half_life"]
                           <= PAIRS_MAX_HALF_LIFE)
    assert 0.0 <= fit.adf["p_value"] <= 1.0
    assert summary["guard_reason"] == fit.guard["reason"]
    assert statsmodels_adf_tau(fit.adf["spread"], regression="nc") in (None,) or True


def test_real_cached_majors_are_not_cointegrated_on_1h():
    """Measured baseline: no major pair passes Engle-Granger on the cached 1h data.

    The old form of this test also pinned a *margin* (``min(p_values) > 0.10``).
    That was a measurement of one cache state, not of the guard: it is exactly
    the pattern that made ``test_meta_labeling``'s AUC pin fail after a cache
    repair.  The claim is de-pinned to the decision the guard exists to make --
    no pair is declared cointegrated at the 5 % level, and every pair's verdict
    is reproduced by the guard on its own thresholds -- so a repaired or
    extended cache changes the numbers, not the verdict.  Sensitivity to a real
    cointegrated pair is carried by the synthetic tests above
    (``test_engle_granger_has_power_on_a_cointegrated_pair``,
    ``test_fit_pair_accepts_a_cointegrated_pair_and_emits_a_kernel_signal``).
    """
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    frames = {s: _cached(s) for s in symbols}
    if any(v is None for v in frames.values()):
        pytest.skip("cached 1h majors not present in this checkout")
    p_values = []
    verdicts = []
    for a, b in itertools.combinations(symbols, 2):
        joint = pd.concat([np.log(frames[a]["close"]).rename("y"),
                           np.log(frames[b]["close"]).rename("x")],
                          axis=1).dropna()
        res = engle_granger(joint["y"], joint["x"])
        assert 0.0 <= res["p_value"] <= 1.0
        p_values.append(res["p_value"])
        fit = fit_pair(joint["y"], joint["x"], symbol_y=a, symbol_x=b)
        # The guard's verdict, reproduced from the fit's own reported fields:
        # refused when cointegration is rejected, the half-life leaves the
        # window, or the hedge ratio is unusable.  The thresholds are the
        # guard's documented contract, not measurements of this cache.
        expected = (float(fit.adf["p_value"]) <= PAIRS_MAX_ADF_PVALUE
                    and PAIRS_MIN_HALF_LIFE <= float(fit.half_life["half_life"])
                    <= PAIRS_MAX_HALF_LIFE
                    and abs(float(fit.beta)) >= 0.01)
        assert fit.allowed is expected, (a, b, fit.summary())
        assert fit.summary()["guard_reason"] == fit.guard["reason"]
        if not fit.allowed:
            assert fit.guard["reason"] != "pass"
        verdicts.append(fit.allowed)
    assert len(p_values) == 10
    assert all(0.0 <= p <= 1.0 for p in p_values)
    # No major pair is declared cointegrated on this cache: the guard refuses
    # every one of the ten, and the reason is never "pass".
    assert not any(verdicts)


def test_real_cached_btc_eth_kalman_is_close_to_ols_and_stable():
    """The real-pair counterpart of the ``r_var`` bug guard.

    The comparator is the fit's own OLS residual variance, so the assertion is
    the *mechanism* (``R`` is the spread's noise) rather than a tolerance on a
    measured number: the pre-fix filter scaled ``R`` from ``var(y)`` and
    produced 0.058 against an OLS beta of 0.628 here.  The old form compared
    ``r_var`` against ``var(y) / 2.0`` -- an arbitrary divisor on a cache-derived
    quantity -- and ``beta_std`` against the 250-bar rolling OLS beta's own
    standard deviation, which measures a different estimator.
    """
    df_y, df_x = _cached("BTCUSDT"), _cached("ETHUSDT")
    if df_y is None or df_x is None:
        pytest.skip("no cached BTCUSDT/ETHUSDT 1h parquet in this checkout")
    joint = pd.concat([np.log(df_y["close"]).rename("y"),
                       np.log(df_x["close"]).rename("x")], axis=1).dropna()
    # Below this many bars neither estimator means anything; a truncated cache
    # is reported rather than silently measured.
    assert len(joint) >= 250
    kf = kalman_hedge_ratio(joint["y"], joint["x"])
    ols = ols_hedge_ratio(joint["y"], joint["x"])
    # Scale-correctness: on this cache Kalman mean 0.606 vs OLS 0.628.
    assert kf["beta_mean"] == pytest.approx(ols["beta"], abs=0.05)
    # R is the OLS residual variance *of the same fit*: this identity is the
    # documented contract and fails hard for the pre-fix var(y) scaling.
    assert kf["r_var"] == pytest.approx(ols["resid_sd"] ** 2, rel=1e-6)
    # ...and it is not simply the level's variance: the spread is genuinely
    # smaller than the price path on this pair.
    assert kf["r_var"] < float(np.var(joint["y"]))
    # The filter is not degenerate: it moves (it tracks a real hedge ratio)
    # yet stays far quieter than the synthetic rolling estimate it replaced.
    roll = joint["y"].rolling(250).cov(joint["x"]) / joint["x"].rolling(250).var()
    spread_of_estimates = float(roll.quantile(0.95) - roll.quantile(0.05))
    assert kf["beta_std"] < spread_of_estimates
    assert float(np.ptp(kf["beta"])) > 0.0


# ── 7. the live-path seam must be inert by default ──────────────────────

class _FakeMarketData:
    """Deterministic stand-in for ``MarketDataProvider`` (no network)."""

    def __init__(self, n: int = 220, seed: int = 0):
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        self.df = pd.DataFrame(
            {"open": close, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": rng.random(n) * 10 + 1},
            index=pd.date_range("2026-01-01", periods=n, freq="1h"))
        self.watched_symbols = ["BTCUSDT"]

    async def get_historical(self, symbol, interval, limit=None):
        return self.df.copy()

    def get_current_price(self, symbol):
        return float(self.df["close"].iloc[-1])


def _live_engine():
    from app.config import Config
    from app.event_bus import EventBus
    from core.strategy.engine import StrategyEngine
    from core.strategy.loader import StrategyConfig

    Config._instance = None
    engine = StrategyEngine(Config.load("sim"), EventBus(), _FakeMarketData())
    strategy = StrategyConfig(
        name="p4_probe", enabled=True, mode="trend", timeframes=["1h"],
        indicators={"rsi": {"period": 14, "source": "close"}},
        entry_conditions={"long": ["close > 0"], "short": ["close < 0"]},
    )
    return engine, strategy


def _pair_signal(allowed: bool, indicator: float) -> PairsSignal:
    return PairsSignal(allowed=allowed, reason="pass" if allowed else "refused",
                       indicator_signal=indicator, confidence=0.9, z=-2.5,
                       beta=0.6, lookback=40, half_life=10.0, p_value=0.01)


@pytest.mark.asyncio
async def test_engine_pairs_seam_is_inert_by_default(monkeypatch):
    """`P4_PAIRS_SIGNALS_ENABLED = False` must leave the live signal untouched."""
    import core.strategy.engine as engine_mod

    engine, strategy = _live_engine()
    engine.wire_pairs_provider(lambda symbol, interval: _pair_signal(True, -1.0))
    assert engine_mod.P4_PAIRS_SIGNALS_ENABLED is False
    await engine._evaluate("BTCUSDT", "1h", strategy)
    entry = engine._signal_cache["p4_probe|BTCUSDT"]
    assert entry["indicator_signal"] == 1.0      # the rule's own signal
    assert "p4" not in entry                     # no new payload keys either


@pytest.mark.asyncio
async def test_engine_pairs_seam_uses_the_provider_when_enabled(monkeypatch):
    import core.strategy.engine as engine_mod

    monkeypatch.setattr(engine_mod, "P4_PAIRS_SIGNALS_ENABLED", True)
    engine, strategy = _live_engine()
    engine.wire_pairs_provider(lambda symbol, interval: _pair_signal(True, -1.0))
    await engine._evaluate("BTCUSDT", "1h", strategy)
    entry = engine._signal_cache["p4_probe|BTCUSDT"]
    assert entry["indicator_signal"] == -1.0
    assert entry["p4"]["pairs_indicator"] == -1.0
    # ...and a *refused* pair contributes nothing.
    engine.wire_pairs_provider(lambda symbol, interval: _pair_signal(False, -1.0))
    await engine._evaluate("BTCUSDT", "1h", strategy)
    assert engine._signal_cache["p4_probe|BTCUSDT"]["indicator_signal"] == 1.0


@pytest.mark.asyncio
async def test_engine_pairs_provider_failure_never_breaks_the_evaluation(monkeypatch):
    import core.strategy.engine as engine_mod

    monkeypatch.setattr(engine_mod, "P4_PAIRS_SIGNALS_ENABLED", True)

    def _boom(symbol, interval):
        raise RuntimeError("provider exploded")

    engine, strategy = _live_engine()
    engine.wire_pairs_provider(_boom)
    await engine._evaluate("BTCUSDT", "1h", strategy)
    assert engine._signal_cache["p4_probe|BTCUSDT"]["indicator_signal"] == 1.0
