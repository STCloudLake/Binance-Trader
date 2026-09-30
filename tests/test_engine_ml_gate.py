"""P1.5 — engine ML-path fixes handed back by phase P2 (plan §二 P1/P2).

Phase P2 built the ML credibility layer (``core/ml/labels|calibration|
evaluation|credibility|scalers``, one feature contract, a hard gate) but could
not edit ``core/backtest/engine.py``.  These four tests pin the hand-back:

1. the neutrality-honest accuracy diagnostic
   (``core.ml.credibility.ml_accuracy_neutral_abstention``) is what the engine
   reports, with ``coverage_pct`` / ``neutral_pct`` next to ``ml_accuracy_pct``;
2. models **refit** on the round-robin schedule (the stale
   ``key not in ml_models`` conjunct used to freeze each model after one fit);
3. a backtest with ML enabled runs end-to-end on the single 39-column feature
   contract (its own ``_ML_INDICATORS`` used to make ``compute_features`` raise
   ``FeatureContractError``);
4. the **signed** score (centred on the model's own base rate) is what reaches
   ``fuse_signals``: a bullish P(up) contributes positively, an abstention is
   neutral.

Everything runs on a synthetic parquet tree in ``tmp_path`` — no live DB, no
``strategies/`` writes, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── synthetic market (no cached data / no network needed) ───────────────

def _write_market(tmp_path, symbols=("BTCUSDT",), timeframes=("1h", "4h"),
                  start="2022-01-02", bars_1h=24 * 365 * 4):
    """Deterministic OHLCV parquet tree (same shape as the parity gate uses).

    Built from hourly bars (the finest resolution this suite needs) over four
    years, so every test window below is covered with plenty of warm-up.
    """
    rng = np.random.default_rng(20260101)
    base = pd.date_range(start, periods=bars_1h, freq="1h")
    for symbol in symbols:
        close = 20000 + np.cumsum(rng.normal(0, 40, len(base)))
        # Intrabar high/low are jittered per bar: a bar whose close sits exactly
        # at (high + low) / 2 makes the contract's `close_position` column
        # constant, which the feature validator rejects.
        up = rng.random(len(base)) * 60 + 5
        down = rng.random(len(base)) * 60 + 5
        m1h = pd.DataFrame({
            "open": close, "high": close + up, "low": close - down,
            "close": close, "volume": rng.random(len(base)) * 100 + 10,
        }, index=base)
        market_dir = tmp_path / "market" / symbol
        market_dir.mkdir(parents=True, exist_ok=True)
        for tf in timeframes:
            if tf == "1h":
                frame = m1h
            else:
                frame = m1h.resample(tf).agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna()
            frame.to_parquet(market_dir / f"{tf}.parquet")
    return str(tmp_path)


@pytest.fixture(scope="module")
def market_dir(tmp_path_factory):
    """One synthetic four-year market for the whole module (building it is free,
    the ML feature pass over it is not)."""
    return _write_market(tmp_path_factory.mktemp("ml_gate") / "data")


@pytest.fixture()
def run_tmp(tmp_path):
    """Per-test scratch dir — never the project's ``data/`` or ``strategies/``."""
    return tmp_path


def _engine(market_dir, tmp_path, ml_enabled=True, symbols=("BTCUSDT",)):
    from app.config import Config
    from app.event_bus import EventBus
    from core.backtest.engine import BacktestEngine
    from core.executor.executor import OrderExecutor
    from core.risk.manager import RiskManager
    from core.strategy.loader import StrategyLoader

    Config._instance = None
    cfg = Config.load("sim")
    cfg.data_dir = market_dir
    cfg.backtest_engine_mode = "legacy"      # the ML path lives in the legacy engine
    cfg.backtest_ml_enabled = bool(ml_enabled)
    cfg.backtest_live_spread_enabled = False
    bus = EventBus()
    loader = StrategyLoader(str(tmp_path / "strategies"))
    loader.strategies_dir.mkdir(parents=True, exist_ok=True)
    engine = BacktestEngine(cfg, None, RiskManager(cfg, bus), OrderExecutor(cfg, bus))
    return cfg, engine, loader


def _ml_strategy(name: str = "ml_probe", **kwargs):
    """Simple 4h strategy with the ML path switched on.

    4h (not 1h) keeps the synthetic windows short: the engine's ML path
    recomputes indicators per bar for the whole history slice, so the number of
    *bars* — not the wall clock — is what makes this suite fast.
    """
    from core.strategy.loader import MLConfig, StrategyConfig

    params = dict(
        name=name, enabled=True, mode="trend", timeframes=["4h"],
        indicators={"rsi": {"period": 14, "source": "close"},
                    "sma": {"period": 5}},
        entry_conditions={"long": ["close > sma"], "short": ["close < sma"]},
        exit_conditions={"long": ["rsi > 99"], "short": ["rsi < 1"]},
        ml_config=MLConfig(enabled=True),
    )
    params.update(kwargs)
    return StrategyConfig(**params)


# ══════════════════════════════════════════════════════════════════════════
# 1 — the neutral band is an abstention, not a bearish call
# ══════════════════════════════════════════════════════════════════════════

def test_neutral_band_is_abstention_not_a_bearish_call():
    """The helper excludes neutrals; the old inline loop kept them in-sample.

    Six predictions, two on the legacy 0.5 neutral band.  Both neutral bars
    *rose*, so the old inline loop scored each as a bearish-ish long (the
    ``conf >= 0.5`` long test) and reported 5/6 = 83.3 %.  Abstention scores only
    the 4 genuinely called bars (3/4 = 75.0 %) and reports the rest as coverage
    — the in-sample number moves with however the band's bars happened to move,
    which is the point: a neutral is not a directional call either way.
    """
    from core.ml.credibility import ml_accuracy_neutral_abstention

    preds = [0.10, 0.60, 0.50, 0.50, 0.55, 0.45]
    rets = [-0.01, -0.01, 0.01, 0.008, 0.006, -0.006]

    out = ml_accuracy_neutral_abstention(
        list(zip(preds, rets)), rets, threshold=0.005)

    assert out["n_predictions"] == 6
    assert out["n_neutral"] == 2
    assert out["n_scored"] == 4
    assert out["accuracy_pct"] == 75.0
    assert out["coverage_pct"] == pytest.approx(round(4 / 6 * 100, 1))
    assert out["neutral_pct"] == pytest.approx(round(2 / 6 * 100, 1))

    # The old inline scoring, for the record: neutrals kept in-sample and scored
    # on the wrong side of an up move (5 of 6 correct = 83.3 % — 8.3 pp above the
    # abstention figure here, and moved by however the band's bars went; the
    # measured band share in production is 25.7 %/24.1 %).
    old_total, old_correct = 0, 0
    for conf, ret in zip(preds, rets):
        if abs(ret) >= 0.005:
            old_total += 1
            if (ret >= 0.005 and conf >= 0.5) or (ret <= -0.005 and conf < 0.5):
                old_correct += 1
    assert (old_total, old_correct) == (6, 5)
    assert old_correct / old_total * 100 == pytest.approx(83.3, abs=0.05)
    assert out["accuracy_pct"] != old_correct / old_total * 100


def test_neutral_predictions_are_excluded_from_the_engine_diagnostic(market_dir, run_tmp):
    """Integration: the engine reports the abstention-corrected diagnostic.

    ``_predict_ml`` is stubbed to return the legacy neutral 0.5 on every bar, so
    the run is one big neutral band.  The run previously reported a (biased)
    non-zero accuracy with no coverage number at all; now every prediction is
    counted as neutral, none is scored, and the accuracy is 0.0 %.
    """
    cfg, engine, _loader = _engine(market_dir, run_tmp)
    engine._predict_ml = lambda *a, **k: {
        "p_up": 0.5, "base_rate": None, "score": None, "abstained": False}

    result = engine.run_with_exit_evaluation(
        strategies=[_ml_strategy()], symbols=["BTCUSDT"],
        date_start="2025-01-02", date_end="2025-02-15",
        initial_balance=10000.0, mode="full", simulate_ai_weights=False,
        use_live_spread=False)

    assert "error" not in result
    m = result["metrics"]
    assert m["ml_predictions"] > 0, "the ML path never produced a prediction"
    assert m["ml_neutral_pct"] == 100.0
    assert m["ml_coverage_pct"] == 0.0
    assert m["ml_accuracy_pct"] == 0.0
    assert m["ml_scored"] == 0


def test_report_surfaces_coverage_and_neutral_shares():
    """``report.py`` exposes the three honesty numbers, defaulting to 0."""
    from core.backtest.report import generate_report

    report = generate_report({"metrics": {
        "ml_accuracy_pct": 61.5, "ml_coverage_pct": 74.3,
        "ml_neutral_pct": 25.7, "ml_predictions": 1000, "ml_scored": 743,
    }})
    summary = report["summary"]
    assert summary["ml_accuracy_pct"] == 61.5
    assert summary["ml_coverage_pct"] == 74.3
    assert summary["ml_neutral_pct"] == 25.7
    assert summary["ml_predictions"] == 1000
    assert summary["ml_scored"] == 743

    empty = generate_report({"metrics": {}})["summary"]
    assert empty["ml_coverage_pct"] == 0
    assert empty["ml_neutral_pct"] == 0


# ══════════════════════════════════════════════════════════════════════════
# 2 — models refit on the round-robin schedule
# ══════════════════════════════════════════════════════════════════════════

class _CountingStubModel:
    """Cheap stand-in with the ``predict_proba`` shape the engine consumes."""

    def __init__(self):
        self.built_from = None

    def predict_proba(self, X):
        p = 0.5 + 0.4 * float(np.nanmean(np.asarray(X, dtype=float)[:, 0]) > 0)
        return np.array([[1.0 - p, p]])


def test_multi_window_backtest_retrains_the_model(market_dir, run_tmp):
    """A 2-year walk-forward must fit the model more than once.

    Before the fix ``should_retrain`` also required ``key not in ml_models``, so
    each model was fitted exactly once (count = 1) and then frozen for the whole
    run while the comments claimed round-robin retraining.  Training is stubbed
    (the call is counted, not the wall time) so the test measures the engine's
    schedule and not the trainer.
    """
    from core.backtest.engine import BacktestEngine

    cfg, engine, _loader = _engine(market_dir, run_tmp)
    calls: list[int] = []
    original = BacktestEngine._train_ml_model

    def counting_train(self, *args, **kwargs):
        fresh = original(self, *args, **kwargs)
        if fresh is not None:
            calls.append(len(args[0]) if args else len(kwargs.get("df", [])))
        return _CountingStubModel()

    engine._train_ml_model = counting_train.__get__(engine, BacktestEngine)

    result = engine.run_with_exit_evaluation(
        strategies=[_ml_strategy()], symbols=["BTCUSDT"],
        date_start="2024-01-02", date_end="2025-03-15",
        initial_balance=10000.0, mode="full", simulate_ai_weights=False,
        use_live_spread=False)

    assert "error" not in result
    assert len(calls) > 1, (
        f"the model was trained {len(calls)} time(s) — it is frozen, not "
        "refit on the rotation schedule")
    # Every refit sees a strictly larger window (walk-forward, no look-ahead).
    assert calls == sorted(calls)
    assert calls[-1] > calls[0]


# ══════════════════════════════════════════════════════════════════════════
# 3 — the backtest ML path uses the single feature contract
# ══════════════════════════════════════════════════════════════════════════

def test_ml_enabled_backtest_runs_on_the_39_column_contract(market_dir, run_tmp):
    """``ml_enabled: true`` runs end-to-end with a 39-column feature matrix.

    The engine's private ``_ML_INDICATORS`` (rsi/macd/bollinger/adx) plus
    ``feature_list=None`` raised ``FeatureContractError`` from the P2
    ``compute_features``.  It now feeds ``REQUIRED_INDICATORS`` and scores
    ``FEATURE_NAMES``.
    """
    from core.ml.features import FEATURE_NAMES

    assert len(FEATURE_NAMES) == 39

    cfg, engine, _loader = _engine(market_dir, run_tmp)
    trained_on: list[tuple[int, ...]] = []
    original = engine._ml_features_up_to

    def recording(df, ts=None, **kwargs):
        matrix = original(df, ts, **kwargs)
        trained_on.append(tuple(matrix.columns))
        return matrix

    engine._ml_features_up_to = recording

    result = engine.run_with_exit_evaluation(
        strategies=[_ml_strategy()], symbols=["BTCUSDT"],
        date_start="2025-05-01", date_end="2025-07-15",
        initial_balance=10000.0, mode="full", simulate_ai_weights=False,
        use_live_spread=False)

    assert "error" not in result
    assert trained_on, "the ML feature path was never used"
    assert all(cols == tuple(FEATURE_NAMES) for cols in trained_on), (
        "the engine used a different column set than the contract: "
        f"{[len(c) for c in trained_on]} columns")
    # An ML-enabled run really did predict (the model trained, not just tried).
    assert result["metrics"]["ml_predictions"] > 0


# ══════════════════════════════════════════════════════════════════════════
# 4 — the signed score / abstention reaches the fusion
# ══════════════════════════════════════════════════════════════════════════

def test_bullish_prediction_is_positive_and_abstention_is_neutral():
    """Item 4: fusion contribution ordered bullish > neutral = legacy 0.5.

    ``_normalise_ml_prediction`` + ``_ml_fusion_inputs`` are the engine's
    contract adapters; ``fuse_signals`` is the kernel that consumes them.
    """
    from core.backtest.engine import _ml_fusion_inputs, _normalise_ml_prediction
    from core.strategy.evaluation_kernel import fuse_signals

    def contribution(**ml_kwargs):
        return fuse_signals(indicator_signal=1.0, news_sentiment=None,
                            w_indicator=0.6, w_ml=0.3, w_news=0.1,
                            ml_enabled=True, **ml_kwargs)

    neutral = contribution(**_ml_fusion_inputs(_normalise_ml_prediction(0.5)))
    legacy_half = contribution(ml_confidence=0.5)

    # A model whose base rate is 0.4 (the traded subset is not balanced):
    # P(up) = 0.6 is *bullish* — before P2 that same 0.6 was fused as a
    # below-0.5... no: as a weak 0.2 vote, and anything in the 0.38–0.62 band
    # was flattened to a neutral 0.5 first.
    bullish_pred = _normalise_ml_prediction(
        {"p_up": 0.60, "base_rate": 0.40, "score": None, "abstained": False})
    assert bullish_pred["base_rate"] == 0.40
    bullish = contribution(**_ml_fusion_inputs(bullish_pred))

    # An abstention (P2 publishes the base rate as the probability + score 0).
    abstain_pred = _normalise_ml_prediction(
        {"p_up": 0.40, "base_rate": 0.40, "score": 0.0, "abstained": True})
    abstain = contribution(**_ml_fusion_inputs(abstain_pred))

    # A bearish call below the base rate is negative.
    bearish_pred = _normalise_ml_prediction(
        {"p_up": 0.20, "base_rate": 0.40, "score": None, "abstained": False})
    bearish = contribution(**_ml_fusion_inputs(bearish_pred))

    assert bullish > neutral > bearish
    assert abstain == pytest.approx(neutral)
    assert neutral == pytest.approx(legacy_half)
    # The legacy bare-float contract is untouched, band included.
    assert _ml_fusion_inputs(_normalise_ml_prediction(0.60)) == {"ml_confidence": 0.5}
    assert _ml_fusion_inputs(_normalise_ml_prediction(0.70)) == {"ml_confidence": 0.70}


def test_engine_fuses_the_signed_score_of_a_bullish_and_an_abstaining_model(
        market_dir, run_tmp):
    """End-to-end item 4 evidence: an abstaining model removes the ML vote.

    The same strategy/bar/prediction path is run twice with a bullish model
    (P(up) 0.9, base rate 0.4) and an abstaining one (base rate published as the
    probability, score 0).  The bullish run's fused entry scores must be higher,
    and the abstaining run must equal the ML-neutral baseline.
    """
    def run(model_output, tmp):
        cfg, engine, _loader = _engine(market_dir, tmp)
        # Patch the *model output* only: normalisation, the fusion inputs and the
        # kernel all stay real.
        engine._predict_ml = lambda *a, **k: dict(model_output)
        result = engine.run_with_exit_evaluation(
            strategies=[_ml_strategy()], symbols=["BTCUSDT"],
            date_start="2025-05-01", date_end="2025-06-15",
            initial_balance=10000.0, mode="full", simulate_ai_weights=False,
            use_live_spread=False)
        scores = [e["signal_score"] for e in result["events"]
                  if e.get("type") == "entry"]
        return scores

    bullish = run({"p_up": 0.9, "base_rate": 0.4, "score": None, "abstained": False},
                  run_tmp / "bullish")
    abstain = run({"p_up": 0.4, "base_rate": 0.4, "score": 0.0, "abstained": True},
                  run_tmp / "abstain")
    legacy_flat = run({"p_up": 0.5, "base_rate": None, "score": None, "abstained": False},
                      run_tmp / "legacy")

    assert bullish and abstain, "the probe strategy produced no entries"
    min_len = min(len(bullish), len(abstain), len(legacy_flat))
    assert min_len > 0
    # A bullish model is a strictly positive ML vote (0.75 × w_ml); it must lift
    # entries above the ML-neutral 0.5 that the legacy band produced.
    assert max(bullish) > 0.5
    # An abstention must be exactly the legacy neutral score — no ML vote at all.
    assert max(abstain) == pytest.approx(0.5)
    assert abstain == pytest.approx(legacy_flat)
