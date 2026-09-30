"""Phase P2 (ML credibility) regression tests.

Covers the audit findings that motivated the rewrite, and the 12 defects the
independent P2 audit then found in that rewrite (selection optimism, gate cost,
significance floor, per-side thresholds, live signed score, barrier tail, live
feature cost, dead config, training with ML off, wrong docs/tests):

1. labels keep the "no-move" regime instead of silently dropping it,
2. purged/embargoed out-of-sample evaluation + sample-uniqueness weights,
3. the hard gate (OOS AUC > 0.55 AND net expectancy > 0 AND >=100 trades AND
   t > 2) that forces ``enabled: false``,
4. the signed score ``clip((p/base_rate − 1)/scale)`` (no sign inversion, real
   formula and asymmetry) and probability calibration on held-out rows,
5. one feature contract, no near-constant columns, bounded feature cost,
6. volatility-scaled triple barriers with an honest tail,
7. the corrected ``ml_accuracy_pct`` diagnostic (neutral = abstention),
8. the live fusion path carrying base rate + signed score + abstention.

Everything is seeded — same seed, same numbers.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────

def _ohlcv(n: int = 1500, seed: int = SEED) -> pd.DataFrame:
    """Deterministic pseudo-market OHLCV with volatility clustering."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vol = 0.004 * (1.0 + 0.5 * np.sin(np.arange(n) / 40.0))
    ret = rng.normal(scale=vol)
    close = 30_000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(scale=0.001, size=n)))
    low = close * (1 - np.abs(rng.normal(scale=0.001, size=n)))
    volume = rng.lognormal(mean=6.0, sigma=0.4, size=n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx)


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    from core.ml.features import REQUIRED_INDICATORS
    from core.strategy.indicators import compute_all
    return compute_all(df.copy(), REQUIRED_INDICATORS)


# ── 1. labels ────────────────────────────────────────────────────────────

def test_three_class_label_keeps_the_no_move_regime():
    from core.ml.labels import (CLASS_DOWN, CLASS_FLAT, CLASS_UP, CLASS_NAMES,
                                create_three_class_label, flat_share)
    df = _ohlcv(1200)
    y = create_three_class_label(df, forward_periods=4, threshold=0.005)
    assert set(y.dropna().unique()) <= {CLASS_DOWN, CLASS_UP, CLASS_FLAT}
    assert CLASS_NAMES == ("down", "up", "flat")
    # A ±0.5 % threshold on 1h bars must classify a large minority as flat —
    # the old binary label deleted exactly these rows.
    share = flat_share(y)
    assert 0.2 < share < 0.95, share
    # The last `forward_periods` rows have no complete forward window.
    assert y.iloc[-4:].isna().all()


def test_three_class_label_cost_aware_threshold():
    from core.ml.labels import create_three_class_label, flat_share
    df = _ohlcv(1200)
    base = create_three_class_label(df, 4, 0.005)
    wide = create_three_class_label(df, 4, 0.005, cost_pct=0.13, cost_multiple=4.0)
    # 4 × 0.13 % = 0.52 % > 0.5 %, so more bars fall into the flat class.
    assert flat_share(wide) > flat_share(base)


def test_decision_abstains_on_flat_class():
    from core.ml.labels import decision_from_probs
    proba = np.array([
        [0.1, 0.8, 0.1],   # up
        [0.8, 0.1, 0.1],   # down
        [0.2, 0.3, 0.5],   # flat → abstain
    ])
    out = decision_from_probs(proba, threshold_up=0.5)
    assert list(out) == [1, -1, 0]


# ── 2. purged K-fold / embargo / uniqueness ──────────────────────────────

def test_purged_kfold_removes_overlapping_labels_and_embargoes():
    from core.ml.evaluation import purged_kfold_splits
    n, span = 1000, 4
    splits = purged_kfold_splits(n, 5, label_span=span)
    assert len(splits) == 5
    for train_idx, test_idx, purged in splits:
        assert purged > 0
        assert not (set(train_idx) & set(test_idx))
        lo, hi = int(test_idx.min()), int(test_idx.max()) + 1
        # (a) no training label may reach into the test fold …
        ends = np.minimum(train_idx + span, n - 1)
        assert not ((ends >= lo) & (train_idx < lo)).any()
        # (b) … and the embargo after the fold is empty.
        assert not ((train_idx >= hi) & (train_idx < hi + span)).any()
    # Fold 0's test block starts at 20 % of the data; the rows purged around the
    # boundary are exactly the overlapping labels plus the embargo.
    first_train, first_test, first_purged = splits[0]
    assert first_purged >= span


def test_uniqueness_weights_downweight_overlapping_labels():
    from core.ml.evaluation import average_label_overlap, sample_uniqueness_weights
    n, span = 500, 4
    w = sample_uniqueness_weights(n, span)
    assert len(w) == n
    # Normalised so the weights sum to the number of labels (no free lunch).
    assert w.sum() == pytest.approx(n, rel=1e-9)
    assert w.min() > 0
    # Overlapping labels are worth less than 1 on average.
    assert w.mean() == pytest.approx(1.0, rel=1e-9)
    assert average_label_overlap(n, span) == pytest.approx(0.75)


def test_purged_evaluation_never_scores_training_rows():
    from core.ml.credibility import evaluate_model_oos
    from core.ml.trainer import default_binary_factory
    df = _ohlcv(1500)
    ind = _indicators(df)
    from core.ml.features import compute_features
    X = compute_features(ind)
    y = pd.Series((df["close"].shift(-4) > df["close"]).astype(float), index=df.index)
    fwd = (df["close"].shift(-4) - df["close"]) / df["close"]
    keep = X.index.intersection(y.dropna().index)[:-4]
    res = evaluate_model_oos(X.loc[keep], y.loc[keep], fwd.loc[keep],
                             n_splits=5, label_span=4, cost_pct=0.13,
                             model_factory=default_binary_factory())
    assert "error" not in res
    assert res["n_oos"] > 0
    for fold in res["folds"]:
        assert fold["purged"] >= 4
    assert len(res["p_oos"]) == res["n_oos"] == len(res["y_oos"])


# ── 3. the gate ──────────────────────────────────────────────────────────

def test_gate_refuses_below_auc_threshold():
    from core.ml.credibility import credibility_gate
    status = credibility_gate(
        {"auc": 0.512, "accuracy": 0.51, "majority_accuracy": 0.5,
         "brier": 0.25, "log_loss": 0.69, "n": 2000, "base_rate": 0.5,
         "cost_pct": 0.13},
        0.004, n_oos=2000)
    assert status["allowed"] is False
    assert status["enabled"] is False            # forced off
    assert "AUC" in status["reason"]             # and the reason is explicit


def test_gate_refuses_negative_net_expectancy_even_with_high_auc():
    from core.ml.credibility import credibility_gate
    status = credibility_gate(
        {"auc": 0.62, "accuracy": 0.6, "majority_accuracy": 0.5,
         "brier": 0.22, "log_loss": 0.65, "n": 2000, "base_rate": 0.5,
         "cost_pct": 0.13},
        -0.0005, n_oos=2000)
    assert status["allowed"] is False
    assert "expectancy" in status["reason"]


def test_gate_refuses_too_few_oos_rows():
    from core.ml.credibility import credibility_gate
    status = credibility_gate({"auc": 0.9, "n": 10}, 0.01, n_oos=10)
    assert status["allowed"] is False
    assert "insufficient OOS rows" in status["reason"]


def test_gate_passes_a_model_with_real_signal_and_positive_expectancy():
    from core.ml.credibility import credibility_gate
    # Audit F3: the significance evidence is now mandatory (`or` → `and`, and a
    # missing t/PSR is a refusal), so a passing payload must carry both numbers.
    status = credibility_gate(
        {"auc": 0.70, "accuracy": 0.62, "majority_accuracy": 0.5,
         "brier": 0.21, "log_loss": 0.64, "n": 3000, "base_rate": 0.5,
         "cost_pct": 0.13},
        0.0032, n_oos=3000, n_trades=300, t_stat=2.6, psr=0.99)
    assert status["allowed"] is True
    assert status["enabled"] is True
    assert status["reason"] == "pass"


def test_synthetic_signal_passes_and_noise_fails_the_gate():
    """End-to-end: a model trained on real synthetic signal passes the gate."""
    from core.ml.credibility import evaluate_model_oos, gate_from_evaluation
    from core.ml.trainer import default_binary_factory

    rng = np.random.default_rng(SEED)

    def _run(strength: float) -> dict:
        n = 2500
        x = rng.normal(size=(n, 3))
        p = 1.0 / (1.0 + np.exp(-strength * x[:, 0] * 2.5))
        y = (rng.random(n) < p).astype(float)
        fwd = np.where(y > 0.5, 0.006, -0.006) + rng.normal(scale=0.001, size=n)
        res = evaluate_model_oos(
            pd.DataFrame(x, columns=["a", "b", "c"]), pd.Series(y), pd.Series(fwd),
            n_splits=5, label_span=2, cost_pct=0.13,
            model_factory=default_binary_factory(), min_train=50)
        assert "error" not in res
        return gate_from_evaluation(res)

    good = _run(0.55)
    bad = _run(0.0)
    assert good["auc"] > 0.55 and good["allowed"] is True
    assert bad["auc"] <= 0.55 and bad["allowed"] is False
    assert bad["enabled"] is False


# ── 4. signed score + calibration ────────────────────────────────────────

def test_signed_score_is_centred_on_base_rate_and_signed_correctly():
    from core.ml.calibration import signed_score
    # Bullish probability above the base rate → positive score.
    assert signed_score(0.60, 0.45) > 0
    # Bearish probability below the base rate → negative score.
    assert signed_score(0.40, 0.45) < 0
    # Exactly at the base rate → no vote.
    assert signed_score(0.45, 0.45) == pytest.approx(0.0)
    # 0.35 with a 0.30 base rate is still *bullish* information (this is the
    # call the old (p − 0.5)·2 formula scored as bearish).
    assert signed_score(0.35, 0.30) > 0
    # Symmetric and bounded.
    assert signed_score(1.0, 0.5) == pytest.approx(1.0)
    assert signed_score(0.0, 0.5) == pytest.approx(-1.0)
    assert signed_score(0.5, 0.5) == pytest.approx(0.0)
    # base_rate 0.5 reproduces the legacy transform exactly.
    for p in (0.1, 0.35, 0.5, 0.72, 0.95):
        assert signed_score(p, 0.5) == pytest.approx((p - 0.5) * 2.0)


def test_fusion_carries_a_bullish_probability_to_a_positive_score():
    from core.strategy.evaluation_kernel import fuse_signals
    # Indicator neutral, ML bullish above its base rate → positive fused score.
    up = fuse_signals(indicator_signal=0.0, ml_confidence=0.62, ml_base_rate=0.5,
                      news_sentiment=None, ml_enabled=True)
    down = fuse_signals(indicator_signal=0.0, ml_confidence=0.38, ml_base_rate=0.5,
                        news_sentiment=None, ml_enabled=True)
    assert up > 0 > down
    # A "confidence" of 0.35 with a 0.30 base rate is bullish, not bearish.
    assert fuse_signals(indicator_signal=0.0, ml_confidence=0.35, ml_base_rate=0.30,
                        news_sentiment=None, ml_enabled=True) > 0
    # Default base rate keeps the historical behaviour bit-for-bit.
    assert fuse_signals(indicator_signal=0.0, ml_confidence=0.35,
                        news_sentiment=None) < 0
    # An explicit signed score wins over the scalar probability.
    assert fuse_signals(indicator_signal=0.0, ml_confidence=0.35, ml_score=0.5,
                        news_sentiment=None) > 0


def test_vectorized_fusion_matches_scalar_fusion():
    from core.strategy.evaluation_kernel import fuse_signals, fuse_signals_series
    idx = pd.date_range("2024-01-01", periods=5, freq="1h")
    indicator = pd.Series([1.0, -1.0, 0.0, 1.0, -1.0], index=idx)
    series = fuse_signals_series(indicator, ml_enabled=True, ml_confidence=0.62,
                                 ml_base_rate=0.45)
    scalar = fuse_signals(indicator_signal=1.0, ml_confidence=0.62, ml_base_rate=0.45,
                          ml_enabled=True, news_sentiment=None)
    assert series.iloc[0] == pytest.approx(scalar)


def test_miscalibrated_probabilities_become_monotone_on_held_out_data():
    """The plan's acceptance test: corrected reliability curve is monotone."""
    from core.ml.calibration import ProbabilityCalibrator, reliability_curve
    rng = np.random.default_rng(SEED)
    n = 6000
    z = rng.normal(size=n)
    p_good = 1.0 / (1.0 + np.exp(-1.6 * z))
    y = (rng.random(n) < p_good).astype(float)
    # Rank preserved, but wildly overconfident (the shape isotonic must fix).
    p_bad = 0.5 + 0.6 * (p_good - 0.5)

    cut = n // 2
    before = reliability_curve(y[cut:], p_bad[cut:])
    cal = ProbabilityCalibrator("isotonic").fit(p_bad[:cut], y[:cut])
    assert cal.fitted
    after = reliability_curve(y[cut:], cal.transform(p_bad[cut:]))

    assert after["ece"] < before["ece"]
    assert after["monotone"] is True
    assert after["spearman"] >= 0.8
    # Calibration fixes the probabilities, not the ranking (AUC is unchanged).
    from core.ml.evaluation import binary_metrics
    assert binary_metrics(y[cut:], cal.transform(p_bad[cut:]))["auc"] == pytest.approx(
        binary_metrics(y[cut:], p_bad[cut:])["auc"], abs=0.02)


def test_real_data_shape_calibration_removes_overconfidence_on_held_out_rows():
    """Same protocol as the evidence script, on a real-data-shaped model.

    Pools purged out-of-sample probabilities, fits the calibrator on the first
    half and measures reliability on the second half, so the reported curve is
    genuine generalisation (not the in-fit curve, which is monotone by
    construction).  On an uninformative feature matrix the AUC is ≈0.5 and no
    calibration can produce a monotone curve — that part is proven separately
    on a signal-bearing synthetic model.
    """
    from core.ml.calibration import ProbabilityCalibrator, reliability_curve
    from core.ml.credibility import evaluate_model_oos
    from core.ml.trainer import default_binary_factory

    df = _ohlcv(2200)
    ind = _indicators(df)
    from core.ml.features import compute_features
    X = compute_features(ind)
    fwd = (df["close"].shift(-4) - df["close"]) / df["close"]
    y = pd.Series(np.nan, index=df.index)
    y[fwd >= 0.005] = 1.0
    y[fwd <= -0.005] = 0.0
    keep = X.index.intersection(y.dropna().index)
    res = evaluate_model_oos(X.loc[keep], y.loc[keep], fwd.loc[keep],
                             n_splits=5, label_span=4, cost_pct=0.13,
                             model_factory=default_binary_factory(),
                             calibrate="isotonic")
    assert "error" not in res
    p, yv = res["p_raw_oos"], res["y_oos"]
    cut = len(p) // 2
    assert cut >= 60
    cal = ProbabilityCalibrator("isotonic").fit(p[:cut], yv[:cut])
    before = reliability_curve(yv[cut:], p[cut:])
    after = reliability_curve(yv[cut:], cal.transform(p[cut:]))
    # On a no-signal matrix (AUC ≈ 0.5) calibration cannot manufacture a
    # monotone curve — the honest, testable guarantee is that it removes
    # overconfidence without making the ranking worse.
    assert after["ece"] < before["ece"]
    assert after["brier"] <= before["brier"] + 1e-6
    from core.ml.evaluation import binary_metrics
    assert binary_metrics(yv[cut:], cal.transform(p[cut:]))["auc"] == pytest.approx(
        binary_metrics(yv[cut:], p[cut:])["auc"], abs=0.05)


def test_inverted_probabilities_are_detected_by_the_reliability_curve():
    from core.ml.calibration import reliability_curve
    rng = np.random.default_rng(SEED + 1)
    n = 4000
    z = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-1.6 * z))
    y = (rng.random(n) < p).astype(float)
    curve = reliability_curve(y, 1.0 - p)      # the audit's inversion
    assert curve["slope"] < 0
    assert curve["monotone"] is False
    assert curve["spearman"] < 0


def test_calibrator_is_deterministic_and_persistable():
    from core.ml.calibration import ProbabilityCalibrator
    rng = np.random.default_rng(SEED)
    p = rng.uniform(size=600)
    y = (rng.random(600) < p).astype(float)
    a = ProbabilityCalibrator("isotonic").fit(p, y)
    b = ProbabilityCalibrator("isotonic").fit(p, y)
    assert np.allclose(a.transform(p), b.transform(p))
    restored = ProbabilityCalibrator.from_dict(a.to_dict())
    assert np.allclose(restored.transform(p), a.transform(p))


# ── 5. feature contract ──────────────────────────────────────────────────

def test_no_near_constant_column_and_one_feature_list():
    from core.ml.features import (FEATURE_NAMES, compute_features,
                                  near_constant_columns)
    df = _ohlcv(1500)
    ind = _indicators(df)
    X = compute_features(ind)                      # None → the contract
    assert list(X.columns) == list(FEATURE_NAMES)
    # P6-B: the contract grew from 39 to 54 columns (the 15-column volume/flow
    # family).  The count is DERIVED from the contract, not restated as a literal
    # (`tests/test_measured_threshold_policy.py`), so a future column addition
    # updates one place.
    assert X.shape[1] == len(FEATURE_NAMES) == 54
    assert len(FEATURE_NAMES) == 39 + 15, "P6-B added the 15-column volume family"
    assert near_constant_columns(X) == []
    # A nested restatement of the contract must produce the same columns.
    X2 = compute_features(ind, list(FEATURE_NAMES))
    assert list(X2.columns) == list(X.columns)
    assert np.allclose(X.values, X2.values)


def test_compute_features_refuses_raw_ohlcv():
    from core.ml.features import FeatureContractError, compute_features
    with pytest.raises(FeatureContractError):
        compute_features(_ohlcv(300))


def test_compute_features_and_the_live_predictor_agree_on_the_row_floor():
    """Audit P2 #10: the feature code needed 200 rows, the predictor guarded 100.

    Each side now reads the same ``MIN_FEATURE_ROWS`` constant.
    """
    from core.ml.features import (MIN_FEATURE_ROWS, FeatureContractError,
                                  compute_features)

    assert MIN_FEATURE_ROWS == 200
    below = _ohlcv(MIN_FEATURE_ROWS - 1)
    with pytest.raises(FeatureContractError, match="at least 200 rows"):
        compute_features(_indicators(below))
    at_floor = compute_features(_indicators(_ohlcv(MIN_FEATURE_ROWS)))
    assert len(at_floor) == MIN_FEATURE_ROWS
    # The live guard uses the same constant, so 100–199 rows cannot be scored.
    import inspect
    from core.ml import predictor
    source = inspect.getsource(predictor.MLPredictor._on_kline)
    assert "MIN_FEATURE_ROWS" in source


def test_feature_pipeline_cost_is_bounded():
    """Audit P2 #7: the indicator+feature pipeline must not regress to 16 s.

    Measured on 8 845 real BTC 1h bars: `compute_all(REQUIRED_INDICATORS)` +
    `compute_features` cost **16.62 s** before the fix (a 10-lag R/S Hurst per
    bar) and **1.08 s** after (bounded 60-bar/4-lag Hurst, stride 4).  The bound
    is deliberately generous (3.0 s, ~2.8×) so it catches a regression without
    being flaky on a loaded CI machine.  Skipped when the cached parquet is
    absent (the repo does not ship data/).
    """
    import time
    from pathlib import Path

    from core.ml.features import REQUIRED_INDICATORS, compute_features
    from core.strategy.indicators import compute_all

    path = Path("data/market/BTCUSDT/1h.parquet")
    if not path.exists():
        pytest.skip("no cached BTCUSDT 1h parquet in this checkout")
    df = pd.read_parquet(path)
    t0 = time.perf_counter()
    ind = compute_all(df.copy(), REQUIRED_INDICATORS)
    X = compute_features(ind)
    elapsed = time.perf_counter() - t0
    assert len(X) == len(df)
    assert elapsed < 3.0, (
        f"feature pipeline took {elapsed:.2f}s on {len(df)} bars "
        f"(pre-P2 cost was 16.6s; the audit's item-7 regression)")
    # The bounded Hurst is still a real, varying feature — not a constant 0.5.
    assert X["hurst_signal"].std() > 0.01
    assert X["roll_hurst_20"].std() > 0.01


def test_feature_contract_fails_loudly_on_a_short_matrix():
    from core.ml.features import (FEATURE_NAMES, FeatureContractError,
                                  feature_schema_hash, validate_feature_matrix)
    df = pd.DataFrame({c: np.arange(50, dtype=float) for c in FEATURE_NAMES[:10]})
    with pytest.raises(FeatureContractError):
        validate_feature_matrix(df, list(FEATURE_NAMES))
    # A constant (or absent) column is a contract violation too.
    full = pd.DataFrame({c: np.linspace(0, 1, 50) for c in FEATURE_NAMES})
    full["adx"] = 20.0
    with pytest.raises(FeatureContractError):
        validate_feature_matrix(full, near_constant=True)
    assert feature_schema_hash() == feature_schema_hash(list(FEATURE_NAMES))


def test_bounded_hurst_still_uses_the_rs_estimator():
    """The cheap path must be the same estimator, not a stub.

    Checked on the property the estimator is actually good at (and that the
    legacy 10-lag version agrees on): a strongly **persistent** series scores a
    higher Hurst than a **mean-reverting** one of the same scale.
    """
    from core.ml.features import _rolling_hurst_bounded, _rs_hurst_lags
    idx = pd.date_range("2024-01-01", periods=600, freq="1h")
    rng = np.random.default_rng(SEED)
    noise = rng.normal(scale=0.01, size=600)
    persistent = np.zeros(600)
    reverting = np.zeros(600)
    for i in range(1, 600):
        persistent[i] = 0.7 * persistent[i - 1] + noise[i]
        reverting[i] = -0.7 * reverting[i - 1] + noise[i]
    p_series = pd.Series(100 * np.exp(np.cumsum(persistent)), index=idx)
    r_series = pd.Series(100 * np.exp(np.cumsum(reverting)), index=idx)
    assert _rolling_hurst_bounded(p_series).iloc[-1] > \
        _rolling_hurst_bounded(r_series).iloc[-1]
    # The warm-up is a documented NaN window, never a fabricated 0.5.
    assert _rolling_hurst_bounded(p_series).iloc[:60].isna().all()
    # Too few returns → the documented neutral 0.5.
    assert _rs_hurst_lags(np.diff(np.log(np.arange(1.0, 11.0)))) == 0.5



# ── 6. triple barrier ────────────────────────────────────────────────────

def test_volatility_scaled_barriers_reduce_timeout_dominance():
    from core.ml.features import create_triple_barrier_label
    from core.ml.labels import class_distribution, rolling_atr
    df = _ohlcv(2000)
    atr_pct = float((rolling_atr(df, 14) / df["close"]).median())
    assert atr_pct > 0
    legacy = create_triple_barrier_label(
        df, forward_periods=24, upper_pct=0.02, lower_pct=0.02,
        timeout_label=2.0, vol_scaled=False)
    scaled = create_triple_barrier_label(df, forward_periods=24, timeout_label=2.0)
    legacy_timeout = class_distribution(legacy)["timeout_share"]
    scaled_timeout = class_distribution(scaled)["timeout_share"]
    # Fixed 2 % barriers are far wider than the ATR, so almost nothing is hit.
    assert legacy_timeout > scaled_timeout
    assert scaled_timeout < 0.6
    # Both keep the three-class contract the model expects.
    assert set(pd.Series(scaled).dropna().unique()) <= {0.0, 1.0, 2.0}


def test_barrier_widths_scale_with_volatility():
    from core.ml.labels import barrier_widths
    df = _ohlcv(800)
    upper, lower = barrier_widths(df, atr_multiple=1.5, min_pct=0.001, max_pct=0.05)
    assert (upper == lower).all()
    assert upper.min() >= 0.001 and upper.max() <= 0.05
    assert upper.std() > 0          # genuinely volatility-scaled, not constant


def test_triple_barrier_tail_is_na_and_genuine_hits_survive():
    """Audit P2 #6: the last ``horizon`` rows must not be forced to timeout.

    The pre-fix scan ran to ``n - 1`` and then ``fillna(timeout_label)``-ed every
    remaining NA, so a bar whose forward window is truncated at the end of the
    data was labelled "timeout" — 24 bars moved on a 2 000-bar sample, and on the
    real 8 845-bar BTC 1h cache it destroyed **16 genuine barrier hits**.
    """
    from core.ml.labels import (create_triple_barrier_label_vol, barrier_widths)
    horizon = 24
    df = _ohlcv(2000)
    lab = create_triple_barrier_label_vol(df, forward_periods=horizon,
                                          timeout_label=2.0)
    assert lab.iloc[-horizon:].isna().all()
    # Only the tail is unlabelled (the ATR warm-up is filled, not dropped).
    body = lab.iloc[:-horizon]
    assert body.notna().sum() > len(body) - 5
    assert set(lab.dropna().unique()) <= {0.0, 1.0, 2.0}

    # A genuine upper-barrier hit whose forward window is truncated by the end
    # of the data must stay NA — under the old code it became a timeout, and on
    # the real 8 845-bar BTC 1h cache that destroyed 16 genuine hits.
    close = df["close"].to_numpy(float).copy()
    high = df["high"].to_numpy(float).copy()
    up_w, _ = barrier_widths(df, atr_multiple=1.5, min_pct=0.004, max_pct=0.06)
    up_w = up_w.to_numpy(float)
    i = len(df) - horizon // 2                     # window truncated by the tail
    assert i + horizon > len(df) - 1
    high[i + 1] = close[i] * (1.0 + up_w[i]) * 1.5  # unambiguous upper hit
    df2 = df.copy()
    df2["high"] = high
    lab2 = create_triple_barrier_label_vol(df2, forward_periods=horizon,
                                           timeout_label=2.0)
    assert np.isnan(lab2.iloc[i]), "a truncated forward window must stay NA"
    # Every row in the tail is NA, so no truncated row can be a directional hit.
    assert lab2.iloc[-horizon:].isna().all()

    # `timeout_label=None` puts the whole timeout class in NA — the old
    # docstring claimed the opposite.
    none_lab = create_triple_barrier_label_vol(df, forward_periods=horizon,
                                               timeout_label=None)
    assert none_lab.isna().any()
    assert set(none_lab.dropna().unique()) <= {0.0, 1.0}
    assert none_lab.iloc[-horizon:].isna().all()

    # The legacy fixed-width wrapper keeps the same tail contract.
    from core.ml.features import create_triple_barrier_label
    legacy = create_triple_barrier_label(
        df, forward_periods=horizon, upper_pct=0.02, lower_pct=0.02,
        timeout_label=2.0, vol_scaled=False)
    assert legacy.iloc[-horizon:].isna().all()


def test_uniqueness_weights_range_is_not_documented_as_one():
    """Audit P2 #10: weights are mean-normalised to 1, max ≈ 2.08, not "(0,1]"."""
    from core.ml.evaluation import sample_uniqueness_weights
    w = sample_uniqueness_weights(5000, 4)
    assert w.mean() == pytest.approx(1.0, rel=1e-9)
    assert w.max() > 1.0                      # edges must scale UP
    assert w.max() == pytest.approx(2.0827, abs=1e-3)
    assert w.min() > 0.0


# ── 7. cost-aware threshold + diagnostics ────────────────────────────────

def test_cost_aware_threshold_reports_breakeven_and_beats_accuracy_selection():
    from core.ml.credibility import cost_aware_threshold
    rng = np.random.default_rng(SEED)
    n = 4000
    p = rng.uniform(0.2, 0.8, size=n)
    # Edge only exists at the top of the probability range.
    r = np.where(p > 0.6, 0.006, -0.001) + rng.normal(scale=0.0005, size=n)
    y = (r > 0).astype(float)
    res = cost_aware_threshold(y, p, r, cost_pct=0.13, min_trades=20)
    assert res["expectancy"] > 0
    assert res["side"] == "long"
    assert res["threshold_up"] > 0.5
    assert res["breakeven_up"] is not None
    assert res["coverage"] <= 1.0
    # A threshold chosen for accuracy alone would sit at 0.5, where expectancy
    # is negative because most of the range carries no edge.
    at_half = [row for row in res["curve"] if abs(row["threshold"] - 0.5) < 1e-9]
    assert at_half and at_half[0]["expectancy_long"] <= res["expectancy"]


def test_neutral_predictions_are_abstentions_in_the_corrected_metric():
    from core.ml.credibility import ml_accuracy_neutral_abstention
    # 3 correct directional calls, 2 wrong, 2 neutral (must not be scored).
    pairs = [(0.9, 0.01), (0.1, -0.01), (0.7, 0.02),
             (0.9, -0.02), (0.1, 0.02), (0.5, 0.03), (0.5, -0.03)]
    out = ml_accuracy_neutral_abstention(pairs, [], threshold=0.005)
    assert out["n_predictions"] == 7
    assert out["n_neutral"] == 2
    assert out["n_scored"] == 5
    assert out["accuracy_pct"] == pytest.approx(60.0)
    assert out["neutral_pct"] == pytest.approx(round(200.0 / 7, 1), abs=0.05)
    assert out["coverage_pct"] == pytest.approx(round(500.0 / 7, 1), abs=0.05)


def test_cost_pct_for_matches_the_sim_cost_model():
    """Audit P2 #2: the gate must gate on the cost the fills actually pay.

    The pre-fix `cost_pct_for` read `backtest_taker_fee_pct` (0.04 %) and
    returned 0.13 %/0.14 % while `sim_cost_quote` charged 0.28 %/0.30 % round
    trip — a 2× understatement that turned ETH's "+0.058 %" into −0.062 %.
    """
    from app.config import Config, _sim_settings_from_config, sim_cost_quote
    from core.ml.credibility import cost_pct_for

    config = Config.load("sim")
    settings = _sim_settings_from_config(config)

    def _leg_cost_pct(quote):
        """One leg's cost as a fraction of the price it was quoted at, in %."""
        return 100.0 * quote["per_unit_cost"] / quote["quoted_price"]

    def _sim_round_trip_pct(symbol, order_type):
        """Two one-unit legs through the real fill model → round-trip cost %.

        Measured against the **quoted** price (the convention
        ``labels.round_trip_cost_pct`` documents: the fill's cost as a fraction
        of the price it was quoted at).  Measuring each leg against its own fill
        price instead understates the trip by half the spread — this is the check
        that `cost_pct_for` uses the same convention as the fills.
        """
        unit = 1.0
        leg_in = sim_cost_quote(symbol, "long", order_type, 100.0, unit, settings)
        leg_out = sim_cost_quote(symbol, "short", order_type, 100.0, unit, settings)
        return (100.0 * leg_in["per_unit_cost"] / leg_in["quoted_price"]
                + 100.0 * leg_out["per_unit_cost"] / leg_out["quoted_price"])

    for symbol in ("BTCUSDT", "ETHUSDT", "ZZZUSDT"):
        assert cost_pct_for(config, symbol=symbol) == pytest.approx(
            _sim_round_trip_pct(symbol, "market"), abs=1e-6), symbol
        assert cost_pct_for(config, symbol=symbol, order_type="limit") == \
            pytest.approx(_sim_round_trip_pct(symbol, "limit"), abs=1e-6), symbol

    # VIP0 taker 0.10 %/side + half-spread 0.005 %/0.01 % + 2 bp slippage ⇒ BTC
    # 0.25 %, ETH 0.26 % round trip.  The old 0.04 % fee fallback produced 0.13 %/
    # 0.14 % — roughly half.  (Measured from the code, not from the audit's
    # quoted 0.2500 %/0.2600 %: `sim_cost_quote` charges `spread_pct / 2`, so the
    # table's 0.01/0.02 contributes 0.005/0.01 per side.)
    btc = cost_pct_for(config, symbol="BTCUSDT")
    assert btc == pytest.approx(0.25, abs=1e-6), btc
    assert cost_pct_for(config, symbol="ETHUSDT") == pytest.approx(0.26, abs=1e-6)
    assert cost_pct_for(config, symbol="ZZZUSDT") > btc  # default spread is wider


def test_gate_cost_actually_flows_into_the_threshold_search():
    """The cost is what turns a candidate negative: ETH at the true cost."""
    from app.config import Config
    from core.ml.credibility import cost_aware_threshold, cost_pct_for
    from core.ml.labels import round_trip_cost_pct

    config = Config.load("sim")
    true_cost = cost_pct_for(config, symbol="ETHUSDT")
    assert true_cost > 0.25
    rng = np.random.default_rng(SEED)
    n = 2000
    p = rng.uniform(0.3, 0.7, size=n)
    # A raw edge of exactly +0.20 % on the taken rows …
    r = np.where(p > 0.6, 0.0020, -0.0020)
    y = (r > 0).astype(float)
    # The floor forces the search to take at least half the sample, so the cost
    # difference is what decides: +0.20 % clears 0.14 % and loses to the true cost.
    floor = n // 2
    cheap = cost_aware_threshold(y, p, r, cost_pct=0.14, min_trades=floor)
    expensive = cost_aware_threshold(y, p, r, cost_pct=true_cost, min_trades=floor)
    assert cheap["expectancy"] > 0
    assert expensive["side"] is None or expensive["expectancy"] < 0
    # The gate's cost now equals `labels.round_trip_cost_pct` for the same
    # components (the two helpers used to disagree: 0.13 %/0.14 % vs 0.26 %).
    # ETH's `sim_spread_pct` is 0.02 and `sim_cost_quote` charges half of it.
    assert round_trip_cost_pct(taker_fee_pct=0.10, half_spread_pct=0.01,
                               slippage_bps=2.0) == pytest.approx(true_cost)


def test_gate_requires_a_significance_floor():
    """Audit P2 #3: 27 trades / t = 1.28 must never gate a model on."""
    from core.ml.credibility import (GATE_MIN_TRADES, credibility_gate,
                                     net_trade_stats)

    stats = net_trade_stats(
        np.full(27, 0.005) + np.random.default_rng(SEED).normal(scale=0.014, size=27),
        np.ones(27, dtype=bool), +1, cost_pct=0.26)
    # The audited BTC candidate: +0.358 %, SE 0.280 %, t ≈ 1.28.
    status = credibility_gate(
        {"auc": 0.62, "n": 3000, "base_rate": 0.5, "cost_pct": 0.26},
        0.00358, n_oos=3000, min_trades=GATE_MIN_TRADES,
        n_trades=27, t_stat=1.28, psr=0.90)
    assert status["allowed"] is False
    assert "too few trades (27 < 100)" in status["reason"]
    assert "not significant" in status["reason"]
    assert status["n_trades"] == 27 and status["t_stat"] == pytest.approx(1.28)
    assert stats["n"] == 27

    # A candidate that clears every floor may pass — and only such a candidate.
    ok = credibility_gate(
        {"auc": 0.62, "n": 3000, "base_rate": 0.5, "cost_pct": 0.26},
        0.003, n_oos=3000, n_trades=250, t_stat=2.6, psr=0.995)
    assert ok["allowed"] is True and ok["enabled"] is True
    # Audit F3 changed this case from OR to AND: t = 2.0 exactly is not "> 2" and
    # PSR is no longer a substitute for the t floor, so this now refuses.
    edge = credibility_gate(
        {"auc": 0.62, "n": 3000, "cost_pct": 0.26},
        0.003, n_oos=3000, n_trades=250, t_stat=2.0, psr=0.975)
    assert edge["allowed"] is False
    edge2 = credibility_gate(
        {"auc": 0.62, "n": 3000, "cost_pct": 0.26},
        0.003, n_oos=3000, n_trades=250, t_stat=2.0, psr=0.30)
    assert edge2["allowed"] is False


def test_probabilistic_sharpe_matches_the_normal_approximation():
    from core.ml.credibility import probabilistic_sharpe
    assert probabilistic_sharpe(0, 0.0, 0.0) == 0.0
    assert probabilistic_sharpe(10, 0.01, 0.0) == 0.0
    assert probabilistic_sharpe(100, 0.001, 0.005) > 0.9
    assert probabilistic_sharpe(100, -0.001, 0.005) < 0.1
    # t = 2 ⇔ PSR ≈ 0.977 (the two floors are consistent).
    assert probabilistic_sharpe(100, 2 * 0.005 / 10, 0.005) == pytest.approx(
        0.9772, abs=1e-3)
    assert probabilistic_sharpe(100, 0.0, 0.005) == pytest.approx(0.5)


def test_selection_is_nested_not_pooled():
    """Audit P2 #1: the gate number must come from the folds, not the report.

    The reported quantity is ``net_expectancy_oos`` — the mean of each fold's
    calibration-stream threshold applied to that fold's **test** rows.  The old
    pooled search (calibrator + threshold fitted on the reported rows) is kept
    only for comparison and is labelled as optimistic.
    """
    from core.ml.credibility import evaluate_model_oos, gate_from_evaluation
    from core.ml.trainer import default_binary_factory

    rng = np.random.default_rng(SEED)
    n = 2400
    x = rng.normal(size=(n, 3))
    p_true = 1.0 / (1.0 + np.exp(-1.5 * x[:, 0]))
    y = (rng.random(n) < p_true).astype(float)
    fwd = np.where(p_true > 0.6, 0.004, -0.002) + rng.normal(scale=0.001, size=n)
    res = evaluate_model_oos(
        pd.DataFrame(x, columns=["a", "b", "c"]), pd.Series(y), pd.Series(fwd),
        n_splits=5, label_span=2, cost_pct=0.26,
        model_factory=default_binary_factory(), min_train=50, min_trades=50)
    assert "error" not in res
    # The dead calibration stream is gone: every fold's calibrator was fitted on
    # exactly its own stream (sum of n_fit == sum of n_cal).
    assert sum(f["calibrator_n_fit"] for f in res["folds"]) == \
        sum(f["n_cal"] for f in res["folds"]) > 0
    assert all(f["calibrator_n_fit"] <= f["n_cal"] for f in res["folds"])
    # The reported/gated expectancy is the outer one, not the pooled search.
    assert res["thresholds"]["selection"] == "pooled_optimistic"
    assert res["thresholds_oos"]["selection"] == "nested_per_fold"
    # The pooled outer mean is the trade-weighted combination of the folds (the
    # folds have different test sizes, so an unweighted mean would differ).
    took = [f for f in res["folds"] if f["oos_n_trades"]]
    weighted = sum(f["oos_n_trades"] * f["oos_expectancy"] for f in took) \
        / sum(f["oos_n_trades"] for f in took)
    assert res["net_expectancy_oos"] == pytest.approx(weighted, abs=1e-9)
    assert res["n_trades_oos"] == sum(f["oos_n_trades"] for f in res["folds"])
    gate = gate_from_evaluation(res)
    assert gate["net_expectancy"] == pytest.approx(res["net_expectancy_oos"])
    assert gate["n_trades"] == res["n_trades_oos"]
    assert gate["t_stat"] == pytest.approx(res["t_stat_oos"])
    # A fold with no positive candidate contributes no trades at all.
    for fold in res["folds"]:
        if fold["cal_threshold_side"] is None:
            assert fold["oos_n_trades"] == 0
            assert fold["cal_no_positive_candidate"] is True


def test_threshold_search_never_selects_a_negative_expectancy():
    """A "best of the bad" threshold is not an edge (audit P2 #1/#3).

    The returns are symmetric noise around zero, so both the long and the short
    side are negative once the 0.26 % round trip is charged — no threshold may be
    reported.
    """
    from core.ml.credibility import cost_aware_threshold
    rng = np.random.default_rng(SEED)
    n = 1500
    p = rng.uniform(0.3, 0.7, size=n)
    # Deterministic ±0.10 % so both sides have exactly zero gross edge; the
    # 0.26 % round trip makes every candidate negative on both sides.
    r = np.where(np.arange(n) % 2 == 0, 0.001, -0.001)
    y = (r > 0).astype(float)
    res = cost_aware_threshold(y, p, r, cost_pct=0.26, min_trades=20)
    assert res["side"] is None
    assert res["expectancy"] == 0.0
    assert res["n_taken"] == 0
    assert res["t_stat"] == 0.0 and res["psr"] == 0.0
    # Every candidate with enough trades is negative on both sides — that is why
    # nothing is chosen (the empty-tail candidates carry no trades at all).
    for row in res["curve"]:
        if row["n_long"] >= 20:
            assert row["expectancy_long"] < 0, row
        if row["n_short"] >= 20:
            assert row["expectancy_short"] < 0, row


def test_cost_aware_threshold_prefers_the_short_side_when_it_is_better():
    from core.ml.credibility import cost_aware_threshold
    rng = np.random.default_rng(SEED)
    n = 3000
    p = rng.uniform(0.2, 0.8, size=n)
    # Only the LOWER tail of p_up is profitable → the short side must win.
    r = np.where(p < 0.4, -0.006, -0.008) + rng.normal(scale=0.0005, size=n)
    y = (r > 0).astype(float)
    res = cost_aware_threshold(y, p, r, cost_pct=0.26, min_trades=100)
    assert res["side"] == "short"
    assert res["threshold_down"] < 0.5
    assert res["threshold_up"] == pytest.approx(1.0 - res["threshold_down"])
    assert res["expectancy"] > 0



# ── 8. train/serve scaler ────────────────────────────────────────────────

def test_train_time_scaler_is_persistable_and_deterministic():
    from core.ml.scalers import TrainTimeScaler
    rng = np.random.default_rng(SEED)
    data = rng.normal(loc=5.0, scale=2.0, size=(400, 3))
    scaler = TrainTimeScaler.fit(data)
    out = scaler.transform(data)
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-6)
    restored = TrainTimeScaler.from_dict(scaler.to_dict())
    assert np.allclose(restored.transform(data), out)
    # A degenerate (constant) feature must not divide by zero.
    flat = np.full((10, 2), 3.0)
    assert np.isfinite(TrainTimeScaler.fit(flat).transform(flat)).all()


def test_predictor_defaults_to_disabled_and_refuses_unverified_models(tmp_path):
    """ML ships disabled, and a model without a gate sidecar cannot load."""
    from loguru import logger
    from core.ml.predictor import MLPredictor

    class _Cfg:
        data_dir = str(tmp_path)
        ml_enabled = False
        ml_model_type = "lightgbm"
        ml_feature_list = None

    class _Bus:
        def subscribe(self, *a, **k):
            pass

        def unsubscribe(self, *a, **k):
            pass

    class _Md:
        watched_symbols: list = []

    predictor = MLPredictor(_Cfg(), _Bus(), _Md())
    from core.ml.features import FEATURE_NAMES
    assert predictor.feature_count == len(FEATURE_NAMES) == 54  # P6-B: was 39
    assert predictor.gate_status["allowed"] is False
    # A model file with no `_meta.json` sidecar must be refused (logged).
    model_path = tmp_path / "BTCUSDT_default_binary.pkl"
    model_path.write_bytes(b"not-a-model")
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        assert predictor.load_model("BTCUSDT", str(model_path)) is None
    finally:
        logger.remove(sink)
    assert any("REFUSED" in m for m in messages)


def test_predictor_rejects_a_model_with_a_different_feature_contract(tmp_path):
    from core.ml.features import FeatureContractError
    from core.ml.predictor import MLPredictor
    import json

    class _Cfg:
        data_dir = str(tmp_path)
        ml_enabled = False
        ml_model_type = "lightgbm"
        ml_feature_list = None

    class _Bus:
        def subscribe(self, *a, **k):
            pass

        def unsubscribe(self, *a, **k):
            pass

    class _Md:
        watched_symbols: list = []

    predictor = MLPredictor(_Cfg(), _Bus(), _Md())
    (tmp_path / "BTCUSDT_default_binary.pkl").write_bytes(b"x")
    (tmp_path / "BTCUSDT_default_binary_meta.json").write_text(json.dumps({
        "feature_names": ["rsi", "macd_histogram"],
        "gate": {"allowed": True, "reason": "pass", "auc": 0.6,
                 "net_expectancy": 0.001},
        "train_base_rate": 0.5,
    }), encoding="utf-8")
    with pytest.raises(FeatureContractError):
        predictor.load_model("BTCUSDT", str(tmp_path / "BTCUSDT_default_binary.pkl"))


def test_trainer_persists_metadata_and_reports_f1_under_both_names(tmp_path):
    from core.ml.trainer import MLTrainer
    rng = np.random.default_rng(SEED)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    trainer = MLTrainer(str(tmp_path))
    res = trainer.train_binary("BTCUSDT", "unit", X, y, base_rate=float(y.mean()),
                               threshold=0.55,
                               gate={"allowed": False, "enabled": False,
                                     "reason": "OOS AUC 0.51 <= 0.55"})
    assert "error" not in res
    assert res["f1"] == res["f1_score"]
    assert res["enabled"] is False
    meta = trainer.load_meta("BTCUSDT", "unit")
    assert meta is not None
    assert meta["feature_names"] == list("abcd")
    assert meta["n_features"] == 4
    assert meta["threshold"] == pytest.approx(0.55)
    assert meta["gate"]["enabled"] is False
    assert meta["seed"] == 42


def test_trainer_is_deterministic_for_the_same_seed(tmp_path):
    from core.ml.trainer import MLTrainer
    rng = np.random.default_rng(SEED)
    n = 500
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = pd.Series((X["a"] > 0).astype(int))
    a = MLTrainer(str(tmp_path / "a")).train_binary("S", "u", X, y)
    b = MLTrainer(str(tmp_path / "b")).train_binary("S", "u", X, y)
    assert a["auc"] == pytest.approx(b["auc"])
    assert a["accuracy"] == pytest.approx(b["accuracy"])


# ── 9. per-side thresholds + the zero-band abstention bug ────────────────

class _StubModel:
    """``predict_proba`` returning a fixed P(up)."""

    def __init__(self, p_up: float):
        self.p_up = float(p_up)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p_up), np.full(n, self.p_up)])


class _PredictorHarness:
    """Build an MLPredictor whose model/gate metadata we control directly."""

    def __init__(self, tmp_path, p_up: float):
        from core.ml.predictor import MLPredictor

        class _Cfg:
            data_dir = str(tmp_path)
            ml_enabled = False
            ml_model_type = "lightgbm"
            ml_feature_list = None
            ml_confidence_threshold = None
            ml_default_confidence_threshold = 0.55

        class _Bus:
            def subscribe(self, *a, **k):
                pass

            def unsubscribe(self, *a, **k):
                pass

        class _Md:
            watched_symbols: list = []

        self.predictor = MLPredictor(_Cfg(), _Bus(), _Md())
        self.predictor._models["BTCUSDT_binary"] = _StubModel(p_up)


def test_zero_width_band_abstains_instead_of_always_trading(tmp_path):
    """Audit P2 #4: ``base_rate 0.4 / threshold_up 0.4`` must abstain.

    The pre-fix code computed ``band = |threshold − base_rate|`` with a
    ``band > 0`` guard, so a zero band fell through and the model published a
    directional call on **every** bar.
    """
    import asyncio
    h = _PredictorHarness(tmp_path, p_up=0.40)
    X = _ohlcv(300)
    h.predictor._meta["BTCUSDT_binary"] = {
        "base_rate": 0.4, "threshold": 0.4,
        "thresholds": {"threshold_up": 0.4, "threshold_down": 0.6, "side": "long"},
    }
    out = asyncio.run(h.predictor._predict_lgb("BTCUSDT", X.iloc[-1:]))
    assert out["abstained"] is True
    assert out["score"] == 0.0

    # A usable band still trades.
    h2 = _PredictorHarness(tmp_path, p_up=0.60)
    h2.predictor._meta["BTCUSDT_binary"] = {
        "base_rate": 0.40, "threshold": 0.55,
        "thresholds": {"threshold_up": 0.55, "threshold_down": 0.45, "side": "long"},
    }
    ok = asyncio.run(h2.predictor._predict_lgb("BTCUSDT", X.iloc[-1:]))
    assert ok["abstained"] is False
    assert ok["p_up"] == pytest.approx(0.60)
    assert "base_rate" in ok and ok["base_rate"] == pytest.approx(0.40)


def test_short_side_threshold_is_honoured_not_threshold_up(tmp_path):
    """Audit P2 #4: when the short side won, only ``threshold_up`` was persisted."""
    import asyncio
    h = _PredictorHarness(tmp_path, p_up=0.30)
    X = _ohlcv(300)
    # Short side selected with threshold_down 0.45 < base_rate 0.50 → trade when
    # p_up <= 0.45.  The pre-fix code read `threshold_up` (0.55) instead, so the
    # short band was evaluated at the wrong number entirely.
    h.predictor._meta["BTCUSDT_binary"] = {
        "base_rate": 0.50, "threshold": 0.45,
        "thresholds": {"threshold_up": 0.55, "threshold_down": 0.45, "side": "short"},
    }
    out = asyncio.run(h.predictor._predict_lgb("BTCUSDT", X.iloc[-1:]))
    assert out["abstained"] is False and out["side"] == "short"
    assert out["score"] < 0          # a short call is a negative signed score

    # p_up above the short threshold → abstain.
    h2 = _PredictorHarness(tmp_path, p_up=0.60)
    h2.predictor._meta["BTCUSDT_binary"] = {
        "base_rate": 0.50, "threshold": 0.45,
        "thresholds": {"threshold_up": 0.55, "threshold_down": 0.45, "side": "short"},
    }
    out2 = asyncio.run(h2.predictor._predict_lgb("BTCUSDT", X.iloc[-1:]))
    assert out2["abstained"] is True and out2["score"] == 0.0


def test_trainer_persists_both_side_thresholds_and_the_selected_side(tmp_path):
    from core.ml.trainer import MLTrainer
    rng = np.random.default_rng(SEED)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    trainer = MLTrainer(str(tmp_path))
    trainer.train_binary("BTCUSDT", "unit", X, y, base_rate=float(y.mean()),
                         threshold=0.45, threshold_up=0.55, threshold_down=0.45,
                         threshold_side="short",
                         gate={"allowed": False, "enabled": False, "reason": "no"})
    meta = trainer.load_meta("BTCUSDT", "unit")
    assert meta["thresholds"]["threshold_up"] == pytest.approx(0.55)
    assert meta["thresholds"]["threshold_down"] == pytest.approx(0.45)
    assert meta["thresholds"]["side"] == "short"
    assert meta["threshold"] == pytest.approx(0.45)


def test_predictor_verifies_the_stored_schema_hash(tmp_path):
    """Audit P2 #10: a sidecar whose hash disagrees with its columns is refused."""
    import json
    from core.ml.features import FEATURE_NAMES, FeatureContractError
    from core.ml.predictor import MLPredictor

    class _Cfg:
        data_dir = str(tmp_path)
        ml_enabled = False
        ml_model_type = "lightgbm"
        ml_feature_list = None

    class _Bus:
        def subscribe(self, *a, **k):
            pass

        def unsubscribe(self, *a, **k):
            pass

    class _Md:
        watched_symbols: list = []

    predictor = MLPredictor(_Cfg(), _Bus(), _Md())
    (tmp_path / "BTCUSDT_default_binary.pkl").write_bytes(b"x")
    (tmp_path / "BTCUSDT_default_binary_meta.json").write_text(json.dumps({
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_hash": "deadbeefcafe",
        "gate": {"allowed": True, "reason": "pass", "auc": 0.6,
                 "net_expectancy": 0.001},
        "train_base_rate": 0.5,
    }), encoding="utf-8")
    with pytest.raises(FeatureContractError, match="feature schema"):
        predictor.load_model("BTCUSDT", str(tmp_path / "BTCUSDT_default_binary.pkl"))


# ── 10. the live fusion contribution ─────────────────────────────────────

def test_live_fusion_inputs_carry_base_rate_score_and_abstention():
    """Audit P2 #5: an abstention must contribute 0, a bullish call > 0.

    The live engine kept only ``confidence`` and called ``fuse_signals`` without
    ``ml_base_rate``/``ml_score``, so an abstention published as
    ``confidence = base_rate = 0.40`` fused ``(0.40 − 0.5) × 2 = −0.20`` — a
    small **bearish** vote from a model that said nothing.
    """
    from core.strategy.engine import StrategyEngine
    from core.strategy.evaluation_kernel import fuse_signals

    class _Cfg:
        signal_weights = type("W", (), {"indicator": 0.5, "ml": 0.3, "news": 0.2})()
        strategies_dir = "strategies"

    class _Bus:
        pass

    class _Md:
        watched_symbols: list = []

    engine = StrategyEngine(_Cfg(), _Bus(), _Md())

    # ── an abstaining model at base rate 0.40 contributes exactly 0 ──
    engine._ml_confidence["BTCUSDT"] = 0.40
    engine._ml_prediction["BTCUSDT"] = {"confidence": 0.40, "base_rate": 0.40,
                                        "score": 0.0, "abstained": True}
    inputs = engine._ml_fusion_inputs("BTCUSDT")
    assert inputs["ml_base_rate"] == pytest.approx(0.40)
    assert inputs["ml_score"] == 0.0
    neutral = fuse_signals(indicator_signal=0.0, news_sentiment=None,
                           ml_enabled=True, **inputs)
    assert neutral == pytest.approx(0.0)
    # … while the pre-fix scalar-only call was bearish.
    legacy = fuse_signals(indicator_signal=0.0, ml_confidence=0.40,
                          news_sentiment=None, ml_enabled=True)
    assert legacy < 0

    # ── a bullish p_up above the base rate contributes > 0 ──
    engine._ml_confidence["BTCUSDT"] = 0.55
    engine._ml_prediction["BTCUSDT"] = {"confidence": 0.55, "base_rate": 0.40,
                                        "score": 0.375, "abstained": False}
    bullish = engine._ml_fusion_inputs("BTCUSDT")
    assert bullish["ml_score"] > 0
    assert fuse_signals(indicator_signal=0.0, news_sentiment=None,
                        ml_enabled=True, **bullish) > 0
    # An abstention with a stale non-zero score still contributes 0.
    engine._ml_prediction["BTCUSDT"] = {"confidence": 0.55, "base_rate": 0.40,
                                        "score": 0.9, "abstained": True}
    assert engine._ml_fusion_inputs("BTCUSDT")["ml_score"] == 0.0

    # A plain-float legacy cache entry keeps the historical behaviour.
    engine._ml_prediction.pop("BTCUSDT")
    legacy_inputs = engine._ml_fusion_inputs("BTCUSDT")
    assert legacy_inputs == {"ml_confidence": 0.55}
    assert "ml_score" not in legacy_inputs


@pytest.mark.asyncio
async def test_evaluate_publishes_the_full_ml_verdict_into_the_cache():
    """The real `_evaluate()` path must store the verdict, not just confidence."""
    from app.config import Config
    from app.event_bus import EventBus
    from core.strategy.engine import StrategyEngine
    from core.strategy.loader import MLConfig, StrategyConfig

    class _Md:
        def __init__(self):
            n = 200
            rng = np.random.default_rng(SEED)
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

    Config._instance = None
    config = Config.load("sim")
    engine = StrategyEngine(config, EventBus(), _Md())
    engine.loader = type("L", (), {"load_all": staticmethod(lambda: [])})()
    engine._ml_confidence["BTCUSDT"] = 0.40
    engine._ml_prediction["BTCUSDT"] = {"confidence": 0.40, "base_rate": 0.40,
                                        "score": 0.0, "abstained": True}
    strategy = StrategyConfig(
        name="live_ml_probe", enabled=True, mode="trend", timeframes=["1h"],
        indicators={"rsi": {"period": 14, "source": "close"}},
        entry_conditions={"long": ["close > 0"], "short": ["close < 0"]},
        exit_conditions={"long": ["rsi > 100"], "short": ["rsi < 0"]},
        ml_config=MLConfig(enabled=True))
    await engine._evaluate("BTCUSDT", "1h", strategy, publish=False)
    entry = engine._signal_cache["live_ml_probe|BTCUSDT"]
    assert entry["ml_base_rate"] == pytest.approx(0.40)
    assert entry["ml_score"] == 0.0
    # The pre-fix path (scalar confidence only) would have fused −0.20; the fixed
    # path fuses the model's signed score 0.0, so the score must move up by the
    # exact weight the ML term carries.
    from core.strategy.evaluation_kernel import fuse_signals
    weights = engine.config.signal_weights
    w_ml = strategy.ml_config.weight if strategy.ml_config.weight is not None else weights.ml
    legacy = fuse_signals(indicator_signal=1.0, ml_confidence=0.40,
                          news_sentiment=0.0, w_indicator=weights.indicator,
                          w_ml=weights.ml, w_news=weights.news,
                          ml_enabled=True, strategy_ml_weight=w_ml)
    assert legacy < entry["final_score"]
    total_weight = weights.indicator + w_ml + weights.news
    assert entry["final_score"] - legacy == pytest.approx(
        (0.0 - (0.40 - 0.5) * 2.0) * w_ml / total_weight, abs=1e-9)


# ── 11. signed-score documentation and edge cases (audit P2 #10) ─────────

def test_signed_score_formula_is_the_implemented_one_and_is_asymmetric():
    from core.ml.calibration import signed_score
    # The exact implemented formula, evaluated by hand.
    for p, base in ((0.7, 0.4), (0.3, 0.4), (1.0, 0.2), (0.0, 0.2), (0.9, 0.6)):
        scale = max(1.0 / base - 1.0, 1.0 / (1.0 - base) - 1.0)
        expected = max(-1.0, min(1.0, (p / base - 1.0) / scale))
        assert signed_score(p, base) == pytest.approx(expected, abs=1e-12)
    # The documented-but-wrong `2p/base − 1` differs by up to 1.0 and is not
    # what the code does.
    assert signed_score(1.0, 0.2) == pytest.approx(1.0)
    assert abs(signed_score(1.0, 0.2) - (2 * 1.0 / 0.2 - 1.0)) > 1.0
    # Asymmetric: the bearish tail is shorter than the bullish one at b = 0.2.
    assert signed_score(1.0, 0.2) == pytest.approx(1.0)
    assert signed_score(0.0, 0.2) == pytest.approx(-0.25)
    assert abs(signed_score(0.0, 0.2)) < abs(signed_score(1.0, 0.2))
    # Symmetric at the default base rate (the legacy transform).
    assert signed_score(0.0, 0.5) == pytest.approx(-1.0)


def test_signed_score_guards_subnormal_base_rates():
    """``base_rate = 5e-324`` used to divide to ``inf`` and return NaN."""
    from core.ml.calibration import signed_score
    for bad in (5e-324, 1e-12, 0.0, -0.1, 1.0, 1.5, float("nan")):
        out = signed_score(0.3, bad)
        assert np.isfinite(out), bad
        assert out == pytest.approx(signed_score(0.3, 0.5)), bad


def test_fusion_kernel_docs_state_the_real_formula():
    """The kernel docstring must not advertise the wrong transform.

    It used to state ``clip(2 * (ml_confidence / ml_base_rate) - 1, -1, +1)``,
    which differs from the implemented ``clip(edge / scale)`` by up to 1.0 at base
    rate 0.2/0.8.  The docstring now states the real formula (and calls out the
    wrong one explicitly as wrong), so the assertion is on the *formula text*
    rather than the absence of the words.
    """
    import inspect
    from core.strategy import evaluation_kernel
    src = inspect.getsource(evaluation_kernel.fuse_signals)
    assert "ml_directional = clip(edge / scale, -1, +1)" in src
    assert "not** ``2·ml_confidence/base_rate" in src or "not** ``2" in src
    assert "asymmetric" in src.lower()
    # The module that owns the formula documents the asymmetry too.
    from core.ml import calibration
    assert "asymmetric" in calibration.signed_score.__doc__.lower()


def test_evaluation_module_documents_the_real_weight_range():
    """``(0, 1]`` was wrong: the weights are mean-normalised (max ≈ 2.08)."""
    import inspect
    from core.ml import evaluation
    doc = inspect.getdoc(evaluation.sample_uniqueness_weights)
    assert "not** ``(0, 1]``" in doc
    assert "2.0827" in doc or "2.08" in doc


def test_calibration_test_would_fail_with_a_no_op_calibrator():
    """Audit P2 #10: the old calibration test passed with no-op calibration.

    It asserted ``after["ece"] < before["ece"]`` on rows the calibrator was
    fitted on, so the identity "calibrator" also won (in-fit ECE is 0 by
    construction).  The honest statement is: the *fitted* map improves a
    genuinely held-out slice, and this assertion **fails** for a no-op.
    """
    from core.ml.calibration import ProbabilityCalibrator, reliability_curve

    rng = np.random.default_rng(SEED)
    n = 6000
    z = rng.normal(size=n)
    p_good = 1.0 / (1.0 + np.exp(-1.6 * z))
    y = (rng.random(n) < p_good).astype(float)
    p_bad = 0.5 + 0.6 * (p_good - 0.5)        # rank kept, spread compressed
    cut = n // 2

    before = reliability_curve(y[cut:], p_bad[cut:])
    identity = ProbabilityCalibrator("none")
    noop_after = reliability_curve(y[cut:], identity.transform(p_bad[cut:]))
    fitted = ProbabilityCalibrator("isotonic").fit(p_bad[:cut], y[:cut])
    fitted_after = reliability_curve(y[cut:], fitted.transform(p_bad[cut:]))

    # The identity is exactly the "before" curve — the old assertion is FALSE.
    assert noop_after["ece"] == pytest.approx(before["ece"])
    assert not (noop_after["ece"] < before["ece"])
    # The fitted map wins on the same held-out rows, and its improvement is not
    # an artefact of fitting where it is measured.
    assert fitted_after["ece"] < before["ece"]
    assert fitted.fitted is True


