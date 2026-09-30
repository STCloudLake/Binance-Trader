"""Regression tests for phase P4 — meta-labelling (Prado, AFML ch. 3).

The claims under test:

* the secondary model produces a **filter + size multiplier** and can never flip
  a side;
* the label is side-aware and drops time-outs rather than coercing them;
* the threshold search is **one-sided** (a meta-model is not allowed to short the
  primary's signal) and is selected inside each purged/embargoed fold;
* the gate is the same hard gate as P2 (``credibility.credibility_gate``) and it
  genuinely refuses a noise meta-model — the labeler is then **inert**
  (``take=False``), not "small";
* real cached BTC/ETH 1h data refuses every tested primary rule, which is a
  valid outcome.

Measured-threshold policy
-------------------------
The real-cache test below asserts **gate semantics against the response's own
fields** (``allowed``/``reason`` vs the gate's reported numbers), never a
measured statistic.  Earlier it pinned ``auc <= 0.55``; the cache was repaired
and the same, still-correct behaviour measured AUC 0.5866 while the gate kept
refusing, which failed the test.  The behavioural claim is unchanged: the
candidate is refused, and the reason names the criterion that refused it.  A
numeric expectation tied to ``data/market/**`` may only appear here when it is
derived from the frame the test just read or built synthetically.
``tests/test_measured_threshold_policy.py`` enforces this module-level rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.ml.calibration import ProbabilityCalibrator
from core.ml.credibility import (
    GATE_AUC_MIN, GATE_MIN_NET_EXPECTANCY, GATE_MIN_TRADES,
)
from core.ml.meta import (
    META_MAX_SIZE_MULTIPLIER, META_MIN_TRADES, META_SIZE_FLOOR, MetaDecision,
    MetaLabeler, breakeven_hit_rate, build_meta_dataset, evaluate_meta_oos,
    fold_min_trades, meta_cost_aware_threshold, meta_gate, meta_label_from_barrier,
    meta_summary, primary_forward_returns, primary_signal_from_rules,
    profit_barrier_labels, resolve_model_factory,
)
from core.ml.trainer import default_binary_factory


# ── labels ──────────────────────────────────────────────────────────────

def test_meta_label_from_barrier_is_side_aware_and_drops_timeouts():
    # classes: 1 = upper barrier first, 0 = lower barrier first, 2 = timeout
    labels = pd.Series([1.0, 0.0, 2.0, 1.0, 0.0])
    side = pd.Series([1.0, 1.0, 1.0, -1.0, -1.0])
    out = meta_label_from_barrier(labels, side)
    assert out.iloc[0] == 1.0      # long + upper first = profit
    assert out.iloc[1] == 0.0      # long + lower first = loss
    assert np.isnan(out.iloc[2])   # timeout is dropped, never a class
    assert out.iloc[3] == 0.0      # short + upper first = loss
    assert out.iloc[4] == 1.0      # short + lower first = profit


def test_primary_signal_from_rules_uses_the_shared_kernel_and_handles_ambiguity():
    df = pd.DataFrame({
        "close": [10, 11, 12, 13, 14],
        "rsi": [30, 50, 70, 30, 70],
    })
    sig = primary_signal_from_rules(df, ["rsi < 35"], ["rsi > 65"])
    assert sig.tolist() == [1.0, 0.0, -1.0, 1.0, -1.0]
    both = primary_signal_from_rules(df, ["rsi > 25"], ["rsi < 75"])
    assert (both == 0.0).all()      # both sides active -> ambiguous -> flat


def test_primary_forward_returns_are_signed_by_the_primary_side():
    close = pd.Series([100.0, 110.0, 121.0, 100.0, 110.0])
    side = pd.Series([1.0, -1.0, 0.0, 1.0, 0.0])
    fwd = primary_forward_returns(close, side, forward_periods=2)
    assert fwd.iloc[0] == pytest.approx(0.21)              # long, 100 -> 121
    assert fwd.iloc[1] == pytest.approx(1.0 - 100.0 / 110.0)  # short, 110 -> 100
    assert np.isnan(fwd.iloc[2])                  # no primary signal
    # Rows without a full forward window are never fabricated.
    assert np.isnan(fwd.iloc[3]) and np.isnan(fwd.iloc[4])


def test_profit_barrier_labels_only_label_full_forward_windows():
    n = 60
    close = np.linspace(100.0, 120.0, n)
    df = pd.DataFrame({"open": close, "high": close * 1.001,
                       "low": close * 0.999, "close": close,
                       "volume": 1.0})
    side = pd.Series(1.0, index=df.index)
    labels = profit_barrier_labels(df, side, forward_periods=10, atr_multiple=0.5)
    assert labels.iloc[:-10].notna().all()
    assert labels.iloc[-10:].isna().all()   # truncated windows are never labelled
    assert set(labels.dropna().unique()) <= {0.0, 1.0}


def test_build_meta_dataset_aligns_and_reports_the_timeout_share():
    rng = np.random.default_rng(1)
    n = 300
    close = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01)))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 1.0})
    side = pd.Series(np.where(np.arange(n) % 2 == 0, 1.0, -1.0), index=df.index)
    X = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)},
                     index=df.index)
    ds = build_meta_dataset(df, side, forward_periods=8, feature_matrix=X)
    assert ds["n_primary"] == n
    assert ds["n_labelled"] <= n
    assert list(ds["X"].columns) == ["f1", "f2"]
    assert len(ds["X"]) == len(ds["y_meta"]) == len(ds["trade_returns"])
    assert ds["X"].index.equals(ds["y_meta"].index)
    assert 0.0 <= ds["timeout_share"] <= 1.0


# ── the one-sided threshold search ──────────────────────────────────────

def test_meta_threshold_search_is_one_sided_and_positive_only():
    n = 500
    rng = np.random.default_rng(3)
    p = np.clip(rng.normal(0.5, 0.2, n), 0.01, 0.99)
    # Profitable only when p is HIGH: a two-sided search would pick the low tail.
    r = np.where(p >= 0.7, 0.02, -0.01)
    y = (p >= 0.7).astype(float)
    res = meta_cost_aware_threshold(y, p, r, cost_pct=0.05, min_trades=20)
    assert res["threshold"] is not None and res["threshold"] >= 0.6
    assert res["expectancy"] > 0
    take = p >= res["threshold"]
    assert (r[take] > 0).all()
    assert res["selection"] == "one_sided_take_primary"
    # The low-probability tail is never traded, whatever it returns.
    assert not (p <= 1.0 - res["threshold"]).any() or res["threshold"] > 0.5


def test_meta_threshold_search_refuses_a_negative_expectancy_grid():
    n = 200
    p = np.linspace(0.05, 0.95, n)
    r = np.full(n, -0.01)             # every trade loses
    res = meta_cost_aware_threshold(np.zeros(n), p, r, cost_pct=0.05, min_trades=10)
    assert res["threshold"] is None
    assert res["expectancy"] == 0.0
    assert res["n_taken"] == 0


def test_fold_min_trades_scales_with_the_stream():
    assert fold_min_trades(100, 400) == 40      # 400//10
    assert fold_min_trades(100, 2000) == 100    # capped by the gate floor
    assert fold_min_trades(100, 50) == 20       # floor


# ── the nested evaluation + hard gate ───────────────────────────────────

def _synthetic_meta(n: int = 1200, *, signal: bool, seed: int = 7):
    """A meta dataset with a genuine (or absent) secondary signal."""
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n)
    X = pd.DataFrame({
        "f_signal": rng.normal(0, 1, n),
        "f_noise": rng.normal(0, 1, n),
    }, index=idx)
    if signal:
        y = (X["f_signal"] + rng.normal(0, 0.6, n) > 0).astype(float)
    else:
        y = (rng.normal(0, 1, n) > 0).astype(float)
    returns = pd.Series(np.where(y > 0, 0.012, -0.008), index=idx)
    return X, y, returns


def test_meta_evaluation_uses_purged_folds_and_reports_the_metric_table():
    X, y, r = _synthetic_meta(signal=True)
    res = evaluate_meta_oos(X, y, r, cost_pct=0.05, n_splits=5, label_span=24,
                            min_trades=50)
    assert "error" not in res
    assert res["folds"] and all(f["purged"] > 0 for f in res["folds"])
    m = res["metrics"]
    for key in ("base_rate", "majority_accuracy", "accuracy", "auc", "brier"):
        assert key in m
    assert res["n_oos"] > 0
    nested = res["thresholds_oos"]
    assert nested["selection"] == "one_sided_take_primary"
    assert nested["n_folds_with_candidates"] >= 1
    # The deployed threshold is the median of the per-fold selections, never the
    # pooled optimum on the rows being reported.
    assert nested["min_threshold"] <= res["threshold"] <= nested["max_threshold"]
    assert res["thresholds"]["selection"] == "pooled_optimistic"
    # Every OOS row is a test row of exactly one fold: no row is scored by a
    # model that was fitted on it.
    seen = np.concatenate([res["index_oos"]])
    assert len(seen) == len(np.unique(seen)) == res["n_oos"]
    summary = meta_summary(res)
    assert summary["gate_allowed"] is True
    # The AUC floor is the gate's own configured constant, not a measured value:
    # this dataset is synthetic, and the expectation is read from the response's
    # gate rather than written into the test.
    assert summary["auc"] > meta_gate(res)["auc_min"]
    assert summary["net_expectancy_oos"] > 0
    assert summary["n_trades_oos"] >= META_MIN_TRADES


def test_meta_gate_refuses_a_noise_model():
    X, y, r = _synthetic_meta(signal=False, seed=11)
    res = evaluate_meta_oos(X, y, r, cost_pct=0.05, n_splits=5, label_span=24,
                            min_trades=20)
    gate = meta_gate(res)
    assert gate["allowed"] is False
    assert gate["enabled"] is False
    assert "AUC" in gate["reason"] or "expectancy" in gate["reason"]
    # The noise model fails the gate's own AUC criterion.  Synthetic data, and
    # the threshold is read from the gate rather than written into the test.
    assert gate["auc"] <= gate["auc_min"]


def test_meta_gate_refuses_positive_looking_but_insignificant_results():
    """A high AUC with a net expectancy inside the noise must not pass."""
    X, y, r = _synthetic_meta(signal=True, seed=5)
    res = evaluate_meta_oos(X, y, r, cost_pct=0.05, n_splits=5, label_span=24,
                            min_trades=20)
    gate = meta_gate(res, min_trades=10_000)
    assert gate["allowed"] is False
    assert "too few trades" in gate["reason"]


# ── the labeler (filter + size, never a direction) ──────────────────────

def _enabled_labeler() -> MetaLabeler:
    X, y, r = _synthetic_meta(signal=True)
    lab = MetaLabeler()
    lab.evaluate(X, y, r, cost_pct=0.05, n_splits=5, label_span=24, min_trades=50)
    assert lab.enabled is True
    return lab


def test_labeler_is_inert_until_the_gate_passes():
    X, y, r = _synthetic_meta(signal=False, seed=11)
    lab = MetaLabeler()
    lab.evaluate(X, y, r, cost_pct=0.05, n_splits=5, label_span=24, min_trades=20)
    assert lab.enabled is False
    assert lab.threshold is None
    decision = lab.decide(0.99, side=+1)
    assert decision.take is False
    assert decision.size_multiplier == 0.0
    assert decision.apply(+1) == 0.0
    assert "not enabled" in decision.reason
    # fit() refuses to build a deployment artefact behind a refused gate
    lab.fit(X, y, r, cost_pct=0.05)
    assert lab.model is None
    assert lab.filter_series(pd.Series([1.0, -1.0], index=X.index[:2]),
                             X.iloc[:2]).tolist() == [0.0, 0.0]


def test_fit_without_evaluate_is_refused():
    X, y, r = _synthetic_meta(signal=True)
    lab = MetaLabeler()
    with pytest.raises(RuntimeError, match="evaluate"):
        lab.fit(X, y, r, cost_pct=0.05)


def test_size_multiplier_is_bounded_and_never_flips_the_side():
    lab = _enabled_labeler()
    thr = float(lab.threshold)
    below = lab.decide(thr - 0.01, side=+1)
    assert below.take is False and below.apply(+1) == 0.0
    at = lab.decide(thr, side=+1)
    assert at.take is True
    assert at.size_multiplier == pytest.approx(META_SIZE_FLOOR, abs=1e-9)
    top = lab.decide(1.0, side=+1)
    assert top.size_multiplier == pytest.approx(META_MAX_SIZE_MULTIPLIER)
    # The sign always comes from the primary: a short stays short, scaled.
    short = lab.decide(1.0, side=-1)
    assert short.apply(-1) == pytest.approx(-META_MAX_SIZE_MULTIPLIER)
    assert lab.decide(1.0, side=-1).side == -1.0
    # A non-finite probability is a refusal, not a NaN position.
    assert lab.decide(float("nan"), side=1).take is False


def test_filter_series_only_scales_the_primary():
    lab = _enabled_labeler()
    X, y, r = _synthetic_meta(signal=True)
    side = pd.Series(np.where(np.arange(len(X)) % 2 == 0, 1.0, -1.0), index=X.index)
    gated = lab.filter_series(side, X)
    assert set(np.unique(gated.to_numpy())) <= {1.0, -1.0, 0.0}
    scaled = gated[gated != 0.0]
    assert (np.sign(scaled.to_numpy()) == np.sign(side.loc[scaled.index].to_numpy())).all()
    # The gate can only remove or shrink trades, never add one.
    assert (gated.abs() <= side.abs() + 1e-9).all()


def test_meta_decision_dataclass_contract():
    d = MetaDecision(True, 0.5, 0.8, 0.6, -1.0, "take")
    assert d.apply(-1.0) == pytest.approx(-0.5)
    assert MetaDecision(False, 0.9, 0.5, 0.6, 1.0, "no").apply(1.0) == 0.0


# ── cost plumbing + the P2 factory defect ───────────────────────────────

def test_breakeven_hit_rate_prices_the_cost():
    # reward:risk 1:1 with 0.25 % cost -> 50.12 % of barrier trades must win
    assert breakeven_hit_rate(0.25, 1.0) == pytest.approx(0.50125, abs=1e-6)
    assert breakeven_hit_rate(0.25, 2.0) == pytest.approx(1.0025 / 3.0, abs=1e-6)
    assert breakeven_hit_rate(0.0, 1.0) == pytest.approx(0.5)


def test_resolve_model_factory_accepts_both_forms():
    """``credibility.evaluate_model_oos``'s default path passes the *builder*.

    Binding ``model_factory = default_binary_factory`` and calling
    ``model_factory(X, y, w)`` raises
    ``TypeError: default_binary_factory() takes from 0 to 2 positional arguments
    but 3 were given`` — measured on the first real meta evaluation, where all
    five folds failed.  ``meta.resolve_model_factory`` accepts either form.
    """
    X = pd.DataFrame({"a": np.r_[np.zeros(60), np.ones(60)],
                      "b": np.arange(120, dtype=float)})
    y = pd.Series(np.r_[np.zeros(60), np.ones(60)])
    w = np.ones(len(X))
    for factory in (default_binary_factory, default_binary_factory(), None):
        resolved = resolve_model_factory(factory)
        model = resolved(X, y, w)
        assert hasattr(model, "predict_proba")
    with pytest.raises(TypeError):
        default_binary_factory(X, y, w)   # the defect itself, pinned


def test_calibrator_is_still_the_one_the_gate_uses():
    cal = ProbabilityCalibrator("isotonic")
    assert cal.method == "isotonic"
    assert cal.fitted is False


# ── real cached data (read-only; skipped when the cache is absent) ──────

def _gate_refusal_criteria(gate: dict) -> set[str]:
    """The criteria the gate's *own* reported numbers say must have refused.

    The inequality set of the gate that produced ``gate`` -- an AUC at or below
    ``auc_min``, a net expectancy not above ``net_expectancy_min``, fewer than
    ``min_trades``, or significance short of **both** floors (``min_t_stat`` and
    ``min_psr``) -- reconstructed from its response, never from the fixture.
    Every threshold is read from the response, so the set is independent of which
    cache state produced it.
    """
    min_psr = float(gate.get("min_psr") or 0.95)
    criteria: set[str] = set()
    if not (float(gate["auc"]) > float(gate["auc_min"])):
        criteria.add("auc")
    if not (float(gate["net_expectancy"]) > float(gate["net_expectancy_min"])):
        criteria.add("expectancy")
    if gate.get("n_trades") is not None and int(gate["n_trades"]) < int(gate["min_trades"]):
        criteria.add("trades")
    if gate.get("t_stat") is None or gate.get("psr") is None:
        criteria.add("significance")
    elif not (float(gate["t_stat"]) > float(gate["min_t_stat"])
              and float(gate["psr"]) >= min_psr):
        criteria.add("significance")
    return criteria


def test_real_primary_rules_are_refused_by_the_meta_gate():
    """Every tested primary rule is refused **by the gate**, on this cache.

    The evaluation runs on the last 3 000 cached bars so the suite stays fast.
    Nothing here pins a measured value: the refusal is asserted through the
    response's own ``allowed``/``reason``/gate fields, so a repaired or extended
    cache changes only which criterion refuses the candidate, not the verdict.
    ``rsi_meanrev`` on the 11 677-bar cache measures AUC 0.5866 (above
    ``GATE_AUC_MIN``) and is refused on expectancy, trades and significance;
    on the repaired cache it measures AUC 0.5067 and is refused on AUC too.
    """
    try:
        raw = pd.read_parquet("data/market/BTCUSDT/1h.parquet")
    except Exception:
        pytest.skip("no cached BTCUSDT 1h parquet in this checkout")
    from core.strategy.indicators import compute_all
    from core.ml.features import REQUIRED_INDICATORS, compute_features

    df = compute_all(raw.tail(3000), dict(REQUIRED_INDICATORS))
    X = compute_features(df)
    evaluated = []
    for name, (long_c, short_c) in {
        "rsi_meanrev": (["rsi < 35"], ["rsi > 65"]),
        "bb_breakout": (["close > bollinger_upper"],
                        ["close < bollinger_lower"]),
    }.items():
        side = primary_signal_from_rules(df, long_c, short_c)
        ds = build_meta_dataset(df, side, forward_periods=24, feature_matrix=X)
        if ds["n_labelled"] < 250:
            continue
        res = evaluate_meta_oos(ds["X"], ds["y_meta"], ds["trade_returns"],
                                cost_pct=0.25, label_span=24, n_splits=5,
                                min_trades=50)
        assert "error" not in res, res.get("folds")
        gate = meta_gate(res)
        # The gate reports the same statistic the evaluation measured, and it
        # reports the thresholds it was configured with.
        assert gate["auc"] == pytest.approx(res["metrics"]["auc"])
        assert gate["auc_min"] == pytest.approx(GATE_AUC_MIN)
        assert gate["net_expectancy"] == pytest.approx(res["net_expectancy_oos"])
        assert gate["min_trades"] == GATE_MIN_TRADES
        assert gate["net_expectancy_min"] == pytest.approx(GATE_MIN_NET_EXPECTANCY)
        assert gate["n_oos"] == res["n_oos"]
        # The behavioural claim: refused, inert, and the refusal is not vacuous.
        assert gate["allowed"] is False
        assert gate["enabled"] is False
        assert gate["reason"] and gate["reason"] != "pass"
        # ...and the reason names the criterion its own numbers failed.
        failing = _gate_refusal_criteria(gate)
        assert failing, (f"{name}: the gate refused but none of its reported "
                         f"numbers fails: {gate}")
        reason = gate["reason"]
        named = set()
        if "AUC" in reason:
            named.add("auc")
        if "expectancy" in reason:
            named.add("expectancy")
        if "trades" in reason:
            named.add("trades")
        if "significant" in reason or "PSR" in reason:
            named.add("significance")
        assert named == failing, (
            f"{name}: the reason names {sorted(named) or 'nothing'} but the "
            f"gate's numbers fail {sorted(failing)}: {reason!r}")
        # ...and the reason quotes the gate's own number for each failing
        # criterion, so it cannot be quoting a hard-coded statistic.
        if "auc" in failing:
            assert f"{float(gate['auc']):.4f}" in reason
        if "expectancy" in failing:
            assert f"{float(gate['net_expectancy']) * 100:.4f}" in reason
        if "trades" in failing:
            assert f"{int(gate['n_trades'])}" in reason
        if "significance" in failing and gate.get("t_stat") is not None:
            assert f"{float(gate['t_stat']):.2f}" in reason
        evaluated.append((name, res, gate))

    assert evaluated, "no primary rule produced enough labelled trades"
    for name, res, gate in evaluated:
        # The decision is what a caller receives: an evaluation that failed its
        # gate leaves the labeler inert rather than "small".
        lab = MetaLabeler()
        lab.gate = gate
        lab.threshold = res.get("threshold")
        assert lab.enabled is False
        assert lab.decide(1.0, side=+1).take is False

    # Positive control: nothing here refuses unconditionally.  With the gate's
    # own thresholds relaxed below the same response's numbers, the *same*
    # candidate passes and the labeler becomes active -- so the refusal above is
    # the gate's decision, not a hard-coded verdict.
    name, res, _ = evaluated[0]
    permissive = meta_gate(res, auc_min=0.0, min_net_expectancy=-float("inf"),
                           min_trades=0, min_t_stat=-float("inf"),
                           min_psr=-float("inf"))
    assert permissive["allowed"] is True, (name, permissive)
    assert permissive["reason"] == "pass"
    lab = MetaLabeler()
    lab.gate = permissive
    lab.threshold = res.get("threshold")
    assert lab.enabled is True
    assert lab.decide(1.0, side=+1).take is True


def test_gate_auc_boundary_is_refused_at_the_constant_not_at_a_measured_value():
    """The AUC criterion flips exactly at the configured ``auc_min``.

    Built from synthetic numbers, so it holds on any cache state: the same
    candidate is refused at the constant the gate was given and accepted one AUC
    step above it.  This is what the real-cache test used to assert with a
    hard-coded ``0.55``, expressed without a measured statistic and with the
    other floors satisfied so the AUC criterion is the one under test.
    """
    from core.ml.credibility import credibility_gate

    passing_floors = dict(net_expectancy_value=0.01, n_oos=500, n_trades=500,
                          t_stat=5.0, psr=0.99)

    below = credibility_gate({"auc": GATE_AUC_MIN - 0.01, "n": 500},
                             **passing_floors)
    assert below["allowed"] is False
    assert "AUC" in below["reason"]

    at_threshold = credibility_gate({"auc": GATE_AUC_MIN, "n": 500},
                                    **passing_floors)
    assert at_threshold["allowed"] is False
    assert "AUC" in at_threshold["reason"]

    above = credibility_gate({"auc": GATE_AUC_MIN + 0.01, "n": 500},
                             **passing_floors)
    assert above["allowed"] is True
    assert above["reason"] == "pass"


# ── the live-path seam must be inert by default ─────────────────────────

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
        name="meta_probe", enabled=True, mode="trend", timeframes=["1h"],
        indicators={"rsi": {"period": 14, "source": "close"}},
        entry_conditions={"long": ["close > 0"], "short": ["close < 0"]},
    )
    return engine, strategy


def _passing_labeler(size_floor: float = 0.5) -> MetaLabeler:
    lab = MetaLabeler(size_floor=size_floor)
    lab.gate = {"allowed": True, "enabled": True, "reason": "pass"}
    lab.threshold = 0.5
    return lab


@pytest.mark.asyncio
async def test_engine_meta_seam_is_inert_by_default():
    """`P4_META_FILTER_ENABLED = False` must leave the live score untouched."""
    import core.strategy.engine as engine_mod

    assert engine_mod.P4_META_FILTER_ENABLED is False
    engine, strategy = _live_engine()
    engine.wire_meta_filter(_passing_labeler(), lambda df, symbol: 0.5)
    await engine._evaluate("BTCUSDT", "1h", strategy)
    entry = engine._signal_cache["meta_probe|BTCUSDT"]
    assert entry["final_score"] >= 0.5            # unfiltered rule score
    assert "p4" not in entry


@pytest.mark.asyncio
async def test_engine_meta_seam_scales_and_suppresses_when_enabled(monkeypatch):
    import core.strategy.engine as engine_mod

    engine, strategy = _live_engine()
    await engine._evaluate("BTCUSDT", "1h", strategy)
    base = engine._signal_cache["meta_probe|BTCUSDT"]["final_score"]

    monkeypatch.setattr(engine_mod, "P4_META_FILTER_ENABLED", True)
    engine.wire_meta_filter(_passing_labeler(0.5), lambda df, symbol: 0.5)
    await engine._evaluate("BTCUSDT", "1h", strategy)
    scaled = engine._signal_cache["meta_probe|BTCUSDT"]
    assert scaled["final_score"] == pytest.approx(base * 0.5)
    assert scaled["p4"]["meta"]["take"] is True
    assert scaled["p4"]["meta"]["size_multiplier"] == pytest.approx(0.5)
    # ...and the side is never flipped by the meta model.
    assert scaled["indicator_signal"] == 1.0

    # A refusal suppresses the entry entirely (score 0, threshold not met).
    engine.wire_meta_filter(_passing_labeler(0.5), lambda df, symbol: 0.1)
    await engine._evaluate("BTCUSDT", "1h", strategy)
    refused = engine._signal_cache["meta_probe|BTCUSDT"]
    assert refused["final_score"] == 0.0
    assert refused["threshold_met"] is False
    assert refused["p4"]["meta"]["take"] is False


@pytest.mark.asyncio
async def test_engine_meta_probability_failure_keeps_the_entry(monkeypatch):
    import core.strategy.engine as engine_mod

    monkeypatch.setattr(engine_mod, "P4_META_FILTER_ENABLED", True)

    def _boom(df, symbol):
        raise RuntimeError("no features")

    engine, strategy = _live_engine()
    engine.wire_meta_filter(_passing_labeler(), _boom)
    await engine._evaluate("BTCUSDT", "1h", strategy)
    entry = engine._signal_cache["meta_probe|BTCUSDT"]
    assert entry["final_score"] >= 0.5            # unfiltered, loudly logged
