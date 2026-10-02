"""P7-S4 — the composite (orchestrator + sub-strategies) out-of-sample contract.

Covers the frozen plan's S4 acceptance criteria that can be tested without the
real cache (``docs/overhaul/P7_REGIME_PLAN.md`` §3 S4):

* **contract** — the composite curve is recomputable from the sub-strategy curves
  and the fixed weights (relative error < 1e-9), idle bars return 0 %, and the
  fixed weights come from the training window only;
* **one-shot holdout** — the counter records the first evaluation and *refuses* a
  second one of the same window (and records the reuse when forced);
* **same-window controls** — the random arm is seeded, reproducible, and matched
  to the orchestrator's realized exposure, so the comparison isolates selection;
* **matched benchmarks** — ``exposure_matched`` reuses :mod:`core.ga.benchmark`
  and both it and ``buy_hold`` are computed over the composite's own intervals;
* **trial counting** — every candidate, window, arm and orchestrator rule set
  deflates the composite DSR;
* **usability bar** — ≥ 100 out-of-sample composite trades *and* DSR > 0, with the
  failing criterion named.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.ai import composite as cp
from core.ai import holdout as ho
from core.ai.orchestrator import (
    RegimeOrchestrator,
    TimelineEvent,
    rules_fingerprint,
)
from core.strategy.regime_causal import GATE_REGIME_LABELS

START = "2026-02-01"
END = "2026-03-01"


# ── helpers ────────────────────────────────────────────────────────────

def _curve(values, stamps):
    return [{"time": stamp, "equity": float(value)}
            for stamp, value in zip(stamps, values)]


def _stamps(n: int, freq: str = "1h") -> list:
    return list(pd.date_range(START, periods=n, freq=freq))


def _trade(symbol, opened, closed, pnl, amount=1000.0):
    return {"symbol": symbol, "opened_at": opened, "closed_at": closed,
            "pnl": pnl, "amount_usdt": amount}


def _linear(n: int) -> list:
    return [10_000.0 * (1.0 + 0.0004 * i) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════
# 1 — the weighting rule (training window only)
# ══════════════════════════════════════════════════════════════════════

def test_weights_are_the_measured_capital_shares_and_sum_to_one():
    train = {"a": [_trade("BTCUSDT", "2026-01-01", "2026-01-02", 5, 5000.0)],
             "b": [_trade("ETHUSDT", "2026-01-01", "2026-01-02", -5, 2500.0)]}
    spec = cp.build_composite_spec(train, initial_balance=10_000.0,
                                   symbols=["BTCUSDT", "ETHUSDT"],
                                   train_window=("2026-01-01", "2026-01-05"))
    assert spec.deployed == {"a": 0.5, "b": 0.25}
    assert spec.weights["a"] == pytest.approx(2.0 / 3.0)
    assert spec.weights["b"] == pytest.approx(1.0 / 3.0)
    assert sum(spec.weights.values()) == pytest.approx(1.0)
    assert spec.sources == {"a": "capital", "b": "capital"}
    assert spec.as_dict()["weighting_rule"] == "capital"
    assert spec.symbols == ("BTCUSDT", "ETHUSDT")


def test_a_child_without_notional_falls_back_to_an_equal_share_and_is_labelled():
    train = {"a": [_trade("BTCUSDT", "2026-01-01", "2026-01-02", 5, 0.0)],
             "b": [_trade("ETHUSDT", "2026-01-01", "2026-01-02", 5, 1000.0)]}
    spec = cp.build_composite_spec(train, symbols=["BTCUSDT", "ETHUSDT"])
    assert spec.sources["a"] == "equal_share"
    assert spec.deployed["a"] == pytest.approx(1.0)
    assert spec.sources["b"] == "capital"
    assert spec.deployed["b"] == pytest.approx(0.1)


def test_deployed_shares_are_clipped_to_at_most_full_balance():
    train = {"a": [_trade("BTCUSDT", "2026-01-01", "2026-01-02", 1, 250_000.0)]}
    shares = cp.deployed_shares(train, initial_balance=10_000.0)
    assert shares["a"][0] == 1.0


def test_no_train_trades_is_a_refusal_not_a_silent_equal_weight():
    with pytest.raises(cp.CompositeContractError):
        cp.build_composite_spec({}, symbols=["BTCUSDT"])


def test_the_spec_is_a_pure_function_of_the_train_trades():
    train = {"a": [_trade("BTCUSDT", "2026-01-01", "2026-01-02", 5, 4000.0)],
             "b": [_trade("ETHUSDT", "2026-01-01", "2026-01-02", -1, 1000.0)]}
    first = cp.build_composite_spec(train, symbols=["BTCUSDT", "ETHUSDT"])
    second = cp.build_composite_spec(train, symbols=["BTCUSDT", "ETHUSDT"])
    assert first.as_dict() == second.as_dict()
    # The evaluation window cannot move it: only train trades are an input.
    assert "out_of_sample" not in json.dumps(first.as_dict())


# ══════════════════════════════════════════════════════════════════════
# 2 — the composite curve, recomputable to < 1e-9
# ══════════════════════════════════════════════════════════════════════

def _two_child_fixture():
    stamps = _stamps(30)
    spec = cp.CompositeSpec(weights={"a": 0.6, "b": 0.4},
                            deployed={"a": 0.6, "b": 0.4},
                            sources={"a": "capital", "b": "capital"},
                            symbols=("BTCUSDT", "ETHUSDT"))
    trades = {
        "a": [_trade("BTCUSDT", stamps[2], stamps[10], 8.0, 6000.0),
              _trade("BTCUSDT", stamps[20], stamps[26], -3.0, 6000.0)],
        "b": [_trade("ETHUSDT", stamps[5], stamps[15], 4.0, 4000.0)],
    }
    equities = {
        "a": _curve(_linear(30), stamps),
        "b": _curve([10_000.0 * (1.0 - 0.0003 * i) for i in range(30)], stamps),
    }
    return stamps, spec, trades, equities


def test_the_curve_is_deterministic_for_identical_inputs():
    stamps, spec, trades, equities = _two_child_fixture()
    first = cp.composite_fund(spec, trades, equities, stamps=stamps)
    second = cp.composite_fund(spec, trades, equities, stamps=stamps)
    assert first.equity_curve == second.equity_curve
    assert first.exposure == second.exposure


def test_the_curve_recomputes_from_the_children_to_1e_9_relative_error():
    """``C`` rebuilt from ``composite_return`` must equal the fund's own points."""
    stamps, spec, trades, equities = _two_child_fixture()
    fund = cp.composite_fund(spec, trades, equities, stamps=stamps,
                             initial_balance=10_000.0)
    returns = cp.composite_return(spec, equities, fund.deployed)
    assert len(returns) == len(fund.equity_curve) - 1
    rebuilt = 10_000.0
    assert fund.equity_curve[0]["equity"] == pytest.approx(rebuilt)
    for index, value in enumerate(returns):
        rebuilt *= (1.0 + value)
        point = fund.equity_curve[index + 1]["equity"]
        assert abs(point - rebuilt) / max(abs(rebuilt), 1e-12) < 1e-9


def test_the_curve_is_not_renormalised_when_every_child_goes_flat():
    """A gain made while deployed is kept — the composite never resets to start."""
    stamps = _stamps(12)
    spec = cp.CompositeSpec(weights={"a": 1.0}, symbols=("BTCUSDT",))
    trades = {"a": [_trade("BTCUSDT", stamps[2], stamps[6], 1.0)]}
    equities = {"a": _curve(_linear(12), stamps)}
    fund = cp.composite_fund(spec, trades, equities, stamps=stamps)
    curve = [point["equity"] for point in fund.equity_curve]
    assert curve[-1] > 10_000.0                    # the gain survived the flat bars
    assert curve[-1] == pytest.approx(curve[-2])   # ... and then stayed put
    # The composite earns the child's return on its deployment bars only (the child
    # is not held before bar 2), compounded.
    expected = 1.0
    for index in (2, 3, 4, 5, 6, 7):
        expected *= equities["a"][index]["equity"] / equities["a"][index - 1]["equity"]
    assert curve[-1] / 10_000.0 == pytest.approx(expected, rel=1e-12)


def test_the_curve_matches_the_weighted_returns_of_its_children():
    """``C``'s per-bar return is the weighted return of the deployed children.

    The weights are **fixed on the training window** but the children's equities
    drift, so the effective share of each child moves with its own result — the
    identity is therefore the deployment-weighted mean of the children's *own bar
    returns*, and it compounds into the curve.
    """
    stamps, spec, trades, equities = _two_child_fixture()
    fund = cp.composite_fund(spec, trades, equities, stamps=stamps)
    curve = [point["equity"] for point in fund.equity_curve]
    returns = cp.composite_return(spec, equities, fund.deployed)
    compounded = 1.0
    for index in range(1, len(curve)):
        held = [name for name in spec.names if fund.deployed[name][index]]
        if not held:
            expected_return = 0.0
            assert curve[index] == pytest.approx(curve[index - 1])
        else:
            weighted = sum(
                spec.weight_of(name)
                * (equities[name][index]["equity"]
                   / equities[name][index - 1]["equity"] - 1.0)
                for name in held)
            expected_return = weighted / sum(spec.weight_of(name) for name in held)
        actual = curve[index] / curve[index - 1] - 1.0
        assert actual == pytest.approx(expected_return, rel=1e-12, abs=1e-12)
        assert returns[index - 1] == pytest.approx(expected_return, rel=1e-12,
                                                   abs=1e-12)
        compounded *= (1.0 + expected_return)
    assert curve[-1] / 10_000.0 == pytest.approx(compounded, rel=1e-12)


def test_idle_bars_return_zero_and_the_curve_is_flat_then():
    stamps = _stamps(20)
    spec = cp.CompositeSpec(weights={"a": 1.0}, symbols=("BTCUSDT",))
    trades = {"a": [_trade("BTCUSDT", stamps[5], stamps[8], 1.0)]}
    equities = {"a": _curve(_linear(20), stamps)}
    fund = cp.composite_fund(spec, trades, equities, stamps=stamps)
    curve = [point["equity"] for point in fund.equity_curve]
    assert curve[0] == pytest.approx(10_000.0)
    assert curve[3] == pytest.approx(curve[2])          # flat before the position
    assert curve[19] == pytest.approx(curve[9])         # flat after it
    returns = cp.composite_return(spec, equities, fund.deployed)
    assert len(returns) == 19
    assert all(abs(value) < 1e-12 for value in returns[:4])


def test_a_child_with_no_trades_never_moves_the_curve():
    stamps = _stamps(12)
    spec = cp.CompositeSpec(weights={"a": 0.5, "b": 0.5},
                            symbols=("BTCUSDT", "ETHUSDT"))
    trades = {"a": [_trade("BTCUSDT", stamps[2], stamps[6], 1.0)], "b": []}
    equities = {"a": _curve(_linear(12), stamps),
                "b": _curve([10_000.0 + 500.0 * i for i in range(12)], stamps)}
    fund = cp.composite_fund(spec, trades, equities, stamps=stamps)
    # Child `a` holds stamps 2..6 (5 position bars) plus the cash-settle bar 7, at
    # weight 0.5, so the average exposure is 0.5 · 6/12 = 25 %.
    assert cp.deployed_time_in_market_pct(fund) == pytest.approx(0.5 * 6 / 12 * 100.0)
    assert cp.union_time_in_market_pct(fund) == pytest.approx(5 / 12 * 100.0)
    # Only child `a` is ever deployed, so the composite earns exactly the child's
    # returns on its deployment bars (2..7, including the cash-settle bar that
    # carries the realized PnL in) and keeps them over the flat bars afterwards.
    expected = 1.0
    for index in range(2, 8):
        expected *= equities["a"][index]["equity"] / equities["a"][index - 1]["equity"]
    assert fund.final_equity / 10_000.0 == pytest.approx(expected, rel=1e-12)
    assert fund.total_return_pct == pytest.approx((expected - 1.0) * 100.0,
                                                  rel=1e-12)


def test_position_intervals_merge_overlaps():
    stamps = _stamps(20)
    trades = [_trade("BTCUSDT", stamps[0], stamps[5], 1.0),
              _trade("BTCUSDT", stamps[3], stamps[9], 1.0),
              _trade("BTCUSDT", stamps[15], stamps[17], 1.0)]
    assert cp.position_intervals(trades) == [
        (stamps[0], stamps[9]), (stamps[15], stamps[17])]


# ══════════════════════════════════════════════════════════════════════
# 3 — the matched benchmarks (core.ga.benchmark reused, composite intervals)
# ══════════════════════════════════════════════════════════════════════

def _frames(symbols, n=200):
    stamps = pd.date_range(START, periods=n, freq="1h")
    out = {}
    for index, symbol in enumerate(sorted(symbols)):
        close = 100.0 * np.cumprod(1.0 + 0.0005 * (1.0 + 0.1 * index)
                                   + 0.0 * np.arange(n))
        out[symbol] = pd.DataFrame({"open": close, "high": close * 1.001,
                                    "low": close * 0.999, "close": close,
                                    "volume": 10.0}, index=stamps)
    return out


def test_the_matched_benchmarks_use_the_composites_own_intervals():
    stamps = _stamps(200)
    frames = _frames(["BTCUSDT", "ETHUSDT"])
    rows = [_trade("BTCUSDT", stamps[10], stamps[50], 1.0, 5000.0),
            _trade("ETHUSDT", stamps[100], stamps[140], 1.0, 5000.0)]
    spec = cp.CompositeSpec(weights={"a": 0.5, "b": 0.5},
                            deployed={"a": 0.5, "b": 0.5},
                            symbols=("BTCUSDT", "ETHUSDT"))
    bench = cp.matched_benchmarks(rows, spec, frames, window_start=START,
                                  window_end=str(stamps[-1]),
                                  buy_hold_pct=-3.5)
    matched = bench["exposure_matched"]
    assert matched["mode"] == "exposure_matched"
    assert matched["benchmark_available"] is True
    assert matched["symbol_weights"] == {"BTCUSDT": 0.5, "ETHUSDT": 0.5}
    intervals = bench["in_market_intervals"]
    assert set(intervals) == {"BTCUSDT", "ETHUSDT"}
    hold = bench["buy_hold_matched"]
    assert hold["benchmark_available"] is True
    # The exposure-matched benchmark is the basket weighted by the *measured*
    # deployed shares, so at Σw = 1 (0.5 + 0.5 from the fills) it equals the
    # fully-invested matched buy & hold inside the same intervals.  The
    # deployment scaling is the only difference between the two definitions, and
    # ``buy_hold_matched`` is invariant to it (weights are normalised).
    assert matched["benchmark_pct"] == pytest.approx(hold["benchmark_pct"], rel=1e-9)
    assert sum(spec.deployed.values()) == pytest.approx(1.0)
    assert bench["window_buy_hold_pct"] == -3.5
    # The composite's intervals are the union over its children's trades.
    assert len(intervals["BTCUSDT"]) == 1 and len(intervals["ETHUSDT"]) == 1


def test_a_composite_with_no_trades_has_an_unavailable_benchmark_not_a_zero():
    frames = _frames(["BTCUSDT"])
    spec = cp.CompositeSpec(weights={"a": 1.0}, symbols=("BTCUSDT",))
    bench = cp.matched_benchmarks([], spec, frames, window_start=START,
                                  window_end=END)
    assert bench["exposure_matched"]["benchmark_pct"] == 0.0   # no exposure to match
    assert bench["exposure_matched"]["notes"].startswith("no trades")
    assert bench["buy_hold_matched"]["benchmark_available"] is False


def test_time_in_market_counts_the_union_of_the_childrens_positions():
    stamps = _stamps(100)
    trades = {"a": [_trade("BTCUSDT", stamps[0], stamps[19], 1.0)],
              "b": [_trade("BTCUSDT", stamps[10], stamps[29], 1.0)]}
    share = cp.composite_time_in_market_pct(trades, ["BTCUSDT"], START,
                                            str(stamps[-1]), stamps=stamps)
    assert share == pytest.approx(30.0)          # 0..29, not 40 (no double count)


# ══════════════════════════════════════════════════════════════════════
# 4 — the controls: matched exposure, fixed seed, same window
# ══════════════════════════════════════════════════════════════════════

def test_the_random_gate_is_seeded_and_reproducible():
    stamps = _stamps(500)
    fractions = {"a": 0.3, "b": 0.6}
    first = cp.random_enabled_stamps(stamps, ["a", "b"], fractions, seed=7)
    again = cp.random_enabled_stamps(stamps, ["a", "b"], fractions, seed=7)
    other = cp.random_enabled_stamps(stamps, ["a", "b"], fractions, seed=8)
    assert first == again
    assert first != other
    realized = cp.enable_fractions({name: [stamp in first[name] for stamp in stamps]
                                    for name in ("a", "b")})
    assert realized["a"] == pytest.approx(0.3, abs=0.06)
    assert realized["b"] == pytest.approx(0.6, abs=0.06)


def test_a_disabled_strategy_contributes_no_trades_to_any_variant():
    stamps = _stamps(10)
    trades = {"a": [_trade("BTCUSDT", stamps[1], stamps[2], 1.0),
                    _trade("BTCUSDT", stamps[5], stamps[6], 1.0)]}
    enabled = {stamp: {"a": stamp == stamps[5], "b": True} for stamp in stamps}
    kept = cp.filter_trades_by_timeline({"a": trades["a"], "b": []}, enabled)
    assert [t["opened_at"] for t in kept["a"]] == [stamps[5]]


def test_enable_fractions_report_the_exposure_the_gate_allowed():
    assert cp.enable_fractions({"a": [True, True, False, False]}) == {"a": 0.5}
    assert cp.enable_fractions({"a": []}) == {"a": 0.0}


# ══════════════════════════════════════════════════════════════════════
# 5 — trials, DSR and the usability bar
# ══════════════════════════════════════════════════════════════════════

def test_trial_counting_adds_every_variant_window_and_rule_set():
    trials = cp.CompositeTrials(sub_strategy_candidates=8, windows=2,
                               arm_variants=3, orchestrator_configs=5)
    assert trials.total == 8 * 2 + 3 + 5
    block = trials.as_dict()
    assert block["total"] == trials.total
    assert block["windows"] == 2
    # Never below one trial: a DSR deflated by zero is not a DSR.
    assert cp.CompositeTrials().total == 1


def test_the_dsr_is_deflated_by_the_declared_trial_count():
    stamps, spec, trades, equities = _two_child_fixture()
    fund = cp.composite_fund(spec, trades, equities, stamps=stamps)
    flat = cp.flatten_trades(trades)
    few = cp.composite_metrics(fund, flat, trials=cp.CompositeTrials(1, 1, 2, 1),
                               window_start=START, window_end=END)
    many = cp.composite_metrics(
        fund, flat, trials=cp.CompositeTrials(200, 2, 10, 5),
        window_start=START, window_end=END)
    assert few["dsr_detail"]["n_trials"] == 4
    assert many["dsr_detail"]["n_trials"] == 415
    assert many["deflated_sharpe"] <= few["deflated_sharpe"]


def test_the_metrics_report_every_number_the_plan_asks_for():
    stamps = _stamps(60)
    frames = _frames(["BTCUSDT"])
    spec = cp.CompositeSpec(weights={"a": 1.0}, deployed={"a": 0.5},
                            symbols=("BTCUSDT",))
    trades = {"a": [_trade("BTCUSDT", stamps[5], stamps[25], 12.0, 5000.0),
                    _trade("BTCUSDT", stamps[30], stamps[50], -4.0, 5000.0)]}
    equities = {"a": _curve(_linear(60), stamps)}
    fund = cp.composite_fund(spec, trades, equities, stamps=stamps)
    flat = cp.flatten_trades(trades)
    bench = cp.matched_benchmarks(flat, spec, frames, window_start=START,
                                  window_end=str(stamps[-1]))
    metrics = cp.composite_metrics(fund, flat, trials=cp.CompositeTrials(4, 2, 3, 5),
                                   window_start=START, window_end=str(stamps[-1]),
                                   benchmarks=bench)
    for key in ("trades", "total_return_pct", "max_drawdown_pct", "sharpe",
                "time_in_market_pct", "deflated_sharpe", "dsr_detail",
                "alpha_vs_exposure_matched_pct", "alpha_vs_buy_hold_pct",
                "exposure_matched_pct", "buy_hold_matched_pct",
                "window_buy_hold_pct", "usability"):
        assert key in metrics, key
    assert metrics["trades"] == 2
    assert metrics["alpha_vs_exposure_matched_pct"] == pytest.approx(
        metrics["total_return_pct"] - metrics["exposure_matched_pct"], abs=1e-6)


def test_a_zero_sharpe_variant_reports_dsr_not_estimated():
    stamps = _stamps(40)
    spec = cp.CompositeSpec(weights={"a": 1.0}, symbols=("BTCUSDT",))
    trades = {"a": []}
    equities = {"a": _curve([10_000.0] * 40, stamps)}
    fund = cp.composite_fund(spec, trades, equities, stamps=stamps)
    metrics = cp.composite_metrics(fund, [], trials=cp.CompositeTrials(2, 2, 1, 1),
                                   window_start=START, window_end=END)
    assert metrics["deflated_sharpe"] == 0.0
    assert metrics["dsr_estimated"] is False
    assert "not estimated" in metrics["dsr_note"]


def test_the_usability_bar_is_100_trades_and_a_positive_dsr():
    assert cp.DEFAULT_USABILITY_MIN_TRADES == 100
    passing = cp.usability_verdict(150, 0.25)
    assert passing["usable"] is True and passing["reason"] == ""
    assert "100" in passing["statement"] and "DSR" in passing["statement"]
    few = cp.usability_verdict(99, 0.25)
    assert few["usable"] is False and few["reason"] == "insufficient_trades"
    flat = cp.usability_verdict(150, 0.0)
    assert flat["usable"] is False and flat["reason"] == "dsr_not_positive"
    both = cp.usability_verdict(10, -0.5)
    assert both["usable"] is False and both["reason"] == "both"


def test_a_high_sharpe_low_trade_composite_is_not_usable():
    stamps = _stamps(60)
    spec = cp.CompositeSpec(weights={"a": 1.0}, symbols=("BTCUSDT",))
    trades = {"a": [_trade("BTCUSDT", stamps[5], stamps[25], 50.0)]}
    equities = {"a": _curve(_linear(60), stamps)}
    fund = cp.composite_fund(spec, trades, equities, stamps=stamps)
    metrics = cp.composite_metrics(fund, cp.flatten_trades(trades),
                                   trials=cp.CompositeTrials(2, 2, 3, 1),
                                   window_start=START, window_end=END)
    assert metrics["trades"] < cp.DEFAULT_USABILITY_MIN_TRADES
    assert metrics["usability"]["usable"] is False
    assert metrics["usability"]["reason"] in ("insufficient_trades", "both")


# ══════════════════════════════════════════════════════════════════════
# 6 — the one-shot holdout counter
# ══════════════════════════════════════════════════════════════════════

def test_the_first_evaluation_is_claimed_and_recorded(tmp_path):
    store = tmp_path / "p7_holdout.json"
    assert ho.evaluations("s4", "2026-02-01", "2026-06-01", "1h",
                          path=store) == 0
    claim = ho.claim_holdout("s4", "2026-02-01", "2026-06-01", timeframe="1h",
                             path=store, label="unit",
                             rules_fingerprint="abc123",
                             detail={"variants": ["always_on"]})
    assert claim["reuse"] is False and claim["reuse_index"] == 1
    assert claim["label"] == "unit" and claim["rules_fingerprint"] == "abc123"
    assert claim["detail"] == {"variants": ["always_on"]}
    payload = json.loads(store.read_text(encoding="utf-8"))
    key = ho.holdout_key("s4", "2026-02-01", "2026-06-01", "1h")
    assert payload["claims"][key]["evaluations"][0]["rules_fingerprint"] == "abc123"
    status = ho.holdout_status("s4", "2026-02-01", "2026-06-01", "1h", path=store)
    assert status["evaluated"] is True and status["evaluations"] == 1
    assert status["reused"] is False


def test_a_second_evaluation_of_the_same_window_is_refused(tmp_path):
    store = tmp_path / "p7_holdout.json"
    ho.claim_holdout("s4", "2026-02-01", "2026-06-01", timeframe="1h",
                     path=store, rules_fingerprint="first")
    with pytest.raises(ho.HoldoutRefusal) as excinfo:
        ho.claim_holdout("s4", "2026-02-01", "2026-06-01", timeframe="1h",
                         path=store, rules_fingerprint="second")
    message = str(excinfo.value)
    assert "already evaluated" in message
    assert "first" in message and "refusing" in message
    assert excinfo.value.evaluations == 1
    assert "s4|2026-02-01|2026-06-01|1h" == excinfo.value.key
    # The refusal did NOT append a claim.
    assert ho.evaluations("s4", "2026-02-01", "2026-06-01", "1h",
                          path=store) == 1


def test_reuse_is_possible_but_is_recorded_as_a_second_look(tmp_path):
    store = tmp_path / "p7_holdout.json"
    ho.claim_holdout("s4", "2026-02-01", "2026-06-01", timeframe="1h", path=store)
    second = ho.claim_holdout("s4", "2026-02-01", "2026-06-01", timeframe="1h",
                              path=store, allow_reuse=True)
    assert second["reuse"] is True and second["reuse_index"] == 2
    status = ho.holdout_status("s4", "2026-02-01", "2026-06-01", "1h", path=store)
    assert status["reused"] is True and status["evaluations"] == 2


def test_a_different_window_or_timeframe_is_a_different_holdout(tmp_path):
    store = tmp_path / "p7_holdout.json"
    ho.claim_holdout("s4", "2026-02-01", "2026-06-01", timeframe="1h", path=store)
    ho.claim_holdout("s4", "2026-02-01", "2026-06-01", timeframe="4h", path=store)
    ho.claim_holdout("s4", "2026-03-01", "2026-06-01", timeframe="1h", path=store)
    ho.claim_holdout("other", "2026-02-01", "2026-06-01", timeframe="1h", path=store)
    assert len(ho.describe_claims(path=store)) == 4
    assert ho.evaluations("s4", "2026-02-01", "2026-06-01", "1h", path=store) == 1


def test_the_registry_survives_reload_and_a_corrupt_file_reads_as_empty(tmp_path):
    store = tmp_path / "p7_holdout.json"
    ho.claim_holdout("s4", "2026-02-01", "2026-06-01", timeframe="1h", path=store)
    assert ho.load_registry(store) != {}
    store.write_text("{not json", encoding="utf-8")
    assert ho.load_registry(store) == {}
    # ... and the corrupt file can be re-claimed rather than wedging the tool.
    claim = ho.claim_holdout("s4", "2026-02-01", "2026-06-01", timeframe="1h",
                             path=store)
    assert claim["reuse_index"] == 1


def test_the_default_holdout_store_lives_under_the_gitignored_data_dir():
    path = ho.DEFAULT_HOLDOUT_PATH
    assert path.parent.name == "data" and path.name == "p7_holdout.json"
    gitignore = (Path(ho.ROOT) / ".gitignore").read_text(encoding="utf-8")
    assert "data/" in gitignore.split()


def test_claiming_reports_the_rules_fingerprint_and_revision(tmp_path):
    store = tmp_path / "p7_holdout.json"
    fingerprint = rules_fingerprint({"enabled": True,
                                     "regime": {"allowed": {"a": ["trend_up"]}}})
    claim = ho.claim_holdout("s4", "2026-02-01", "2026-06-01", timeframe="1h",
                             path=store, rules_fingerprint=fingerprint)
    assert claim["rules_fingerprint"] == fingerprint
    assert claim["revision"] == ho.current_revision()
    assert ho.holdout_status("s4", "2026-02-01", "2026-06-01", "1h",
                             path=store)["rules_fingerprints"] == [fingerprint]


# ══════════════════════════════════════════════════════════════════════
# 7 — the orchestrator timeline the composite consumes
# ══════════════════════════════════════════════════════════════════════

def test_the_replayed_timeline_is_deterministic_and_regime_gated():
    stamps = _stamps(12)
    events = [TimelineEvent("decide", at=stamp,
                            label=(GATE_REGIME_LABELS[0] if index % 2 == 0
                                   else GATE_REGIME_LABELS[1]))
              for index, stamp in enumerate(stamps)]
    rules = {"enabled": True,
             "regime": {"allowed": {"a": [GATE_REGIME_LABELS[0]]},
                        "default_action": "deny",
                        "unknown_label_action": "deny"}}
    first = RegimeOrchestrator.replay(rules, events, names=["a"])
    second = RegimeOrchestrator.replay(rules, events, names=["a"])
    assert first == second
    enabled = [row["rows"][0]["enabled"] for row in first]
    assert enabled == [index % 2 == 0 for index in range(12)]
    assert rules_fingerprint(rules) == rules_fingerprint(dict(rules))


def test_a_denied_regime_produces_no_trades_for_that_child():
    stamps = _stamps(10)
    events = [TimelineEvent("decide", at=stamp,
                            label=(GATE_REGIME_LABELS[0] if index < 5
                                   else GATE_REGIME_LABELS[1]))
              for index, stamp in enumerate(stamps)]
    rules = {"enabled": True,
             "regime": {"allowed": {"a": [GATE_REGIME_LABELS[0]]},
                        "default_action": "deny", "unknown_label_action": "deny"}}
    timeline = RegimeOrchestrator.replay(rules, events, names=["a"])
    enabled = {pd.Timestamp(row["at"]): {"a": row["rows"][0]["enabled"]}
               for row in timeline}
    trades = {"a": [_trade("BTCUSDT", stamps[1], stamps[2], 1.0),
                    _trade("BTCUSDT", stamps[7], stamps[8], 1.0)]}
    kept = cp.filter_trades_by_timeline(trades, enabled)
    assert [t["opened_at"] for t in kept["a"]] == [stamps[1]]


# ══════════════════════════════════════════════════════════════════════
# 8 — the per-regime contribution decomposition
# ══════════════════════════════════════════════════════════════════════

def test_the_regime_contribution_attributes_trades_by_entry_label():
    stamps = _stamps(10)
    rows = [_trade("BTCUSDT", stamps[0], stamps[1], 10.0),
            _trade("BTCUSDT", stamps[2], stamps[3], -4.0),
            _trade("BTCUSDT", stamps[4], stamps[5], 6.0)]
    labels = {stamps[0]: "trend_up", stamps[2]: "trend_up", stamps[4]: "range_low"}
    block = cp.regime_contribution(rows, labels, 10_000.0)
    assert block["trend_up"]["trades"] == 2
    assert block["trend_up"]["pnl"] == pytest.approx(6.0)
    assert block["range_low"]["trades"] == 1
    assert block["range_low"]["pnl_pct_of_initial"] == pytest.approx(0.06)


def test_an_unknown_label_is_reported_not_dropped():
    stamps = _stamps(4)
    rows = [_trade("BTCUSDT", stamps[0], stamps[1], 1.0)]
    block = cp.regime_contribution(rows, {}, 10_000.0)
    assert list(block) == ["unknown"]
    assert block["unknown"]["trades"] == 1
