"""P8 — the core arithmetic of ``tools/p8_beta_harvest_measure.py``.

Every test here is a **pure-maths** check on a synthetic series: no parquet, no
engine, no config file, so the suite stays fast and the assertions are exact
rather than "in the right direction".  What is pinned:

* the vol scale is the production ``PositionSizer.vol_scale`` (``clip(target /
  forecast, min_scale, max_scale)``), including both bounds and the
  volatility-unavailable fallback;
* the tracked book reproduces the intended gross exposure, pays the repo's own
  one-side cost, and its equity is the mark-to-market of cash + units;
* ``metrics`` agrees with ``core.ga.fitness`` on the Sharpe / max-drawdown basis
  and gets Calmar, Sortino and the cost share right on hand-computable curves;
* the matched-risk scaling is ``k = sigma_ref / sigma_arm`` compounded;
* the basket rule excludes partial-coverage and USD-stable symbols;
* the pre-stated verdict logic rejects when any of T1/T2/T3 fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.p8_beta_harvest_measure import (      # noqa: E402
    GRID_ESTIMATORS,
    GRID_TARGETS,
    GRID_WINDOWS,
    Overlay,
    Costs,
    THRESHOLD,
    basket_levels,
    basket_symbols,
    dsr_for,
    matched_risk,
    metrics,
    rescaled_daily_returns,
    simulate_hold,
    simulate_vol_target,
    verdict,
)


# ── fixtures / doubles ──────────────────────────────────────────────────

class _StubConfig:
    """A duck-typed config: the only keys the cost model reads."""

    backtest_cost_enabled = True
    backtest_taker_fee_pct = 0.04
    backtest_default_spread_pct = 0.03
    backtest_spread_pct = {"BTCUSDT": 0.01, "ETHUSDT": 0.02}
    backtest_live_spread_enabled = False


def _frame(close: np.ndarray, start="2026-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(close), freq="1h")
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close}, index=index)


def _panel(closes: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    return {sym: _frame(np.asarray(values, dtype=float))
            for sym, values in closes.items()}


def _windowed(closes: dict[str, np.ndarray], warmup: int = 48):
    """``(panel, prices)`` where *panel* carries *warmup* flat pre-window bars.

    The live path always has ``RiskManager._VOL_HISTORY_BARS`` bars available
    before the first decision, so a test of the *first* decision must supply the
    same: ``prices`` is the window, ``panel`` is window + warm-up.
    """
    frames = {}
    for sym, values in closes.items():
        values = np.asarray(values, dtype=float)
        frames[sym] = _frame(np.concatenate([np.full(warmup, values[0]), values]))
    prices = pd.DataFrame({sym: frame["close"] for sym, frame in frames.items()})
    prices = prices.iloc[warmup:]
    return frames, prices


def _prices(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({sym: frame["close"] for sym, frame in panel.items()})


class _FixedOverlay(Overlay):
    """``Overlay`` whose forecast is a constant — makes exposure hand-computable."""

    def __init__(self, forecast: float, **kwargs):
        super().__init__(**kwargs)
        self._forecast = float(forecast)

    def forecast_pct(self, frames):                      # noqa: D102
        return self._forecast

    def per_symbol_scales(self, frames):                 # noqa: D102
        return {sym: self.scale(self._forecast) for sym in frames}


# ── 1. the vol scale is the production arithmetic ───────────────────────

def test_vol_scale_is_clip_of_target_over_forecast():
    overlay = Overlay(target_vol_pct=0.45, min_scale=0.25, max_scale=2.0)
    assert overlay.scale(0.45) == pytest.approx(1.0)
    assert overlay.scale(0.225) == pytest.approx(2.0)          # 2.0, exactly max
    assert overlay.scale(0.10) == pytest.approx(2.0)           # clipped up
    assert overlay.scale(3.60) == pytest.approx(0.25)          # 0.125 -> clipped
    # The scale *is* the sizer's, not a copy of its formula.
    assert overlay.scale(0.30) == pytest.approx(overlay.sizer.vol_scale(0.30))


def test_vol_scale_falls_back_to_one_when_the_forecast_is_unusable():
    overlay = Overlay(target_vol_pct=0.45)
    for bad in (0.0, -1.0, float("nan"), None):
        assert overlay.scale(bad) == pytest.approx(1.0)


# ── 2. the tracked book ─────────────────────────────────────────────────

def test_hold_is_equal_weight_and_unrebalanced():
    prices = _prices(_panel({"A": [100.0, 200.0], "B": [100.0, 100.0]}))
    costs = Costs(_StubConfig(), ["A", "B"])
    result = simulate_hold(prices, costs, initial=10_000.0)
    # 5 000 into each leg, one-side cost = 0.04 % fee + 0.03/2 % half-spread
    # = 0.055 % of the traded notional.
    per_side = 5_000.0 * (0.04 + 0.03 / 2) / 100.0
    assert per_side == pytest.approx(2.75)
    assert costs.one_side("A", 5_000.0) == pytest.approx(per_side)
    entry = 2 * per_side                                    # 5 000 per leg
    liquidation = 3 * per_side                              # A is 10 000, B is 5 000
    assert result["costs"] == pytest.approx(entry + liquidation)
    # Leg A doubles, leg B is flat; the weights are never rebalanced, so the
    # final mark is cash + 10 000 (A) + 5 000 (B) minus the liquidation charge.
    assert float(result["equity"].iloc[0]) == pytest.approx(10_000.0 - 2 * per_side)
    assert float(result["equity"].iloc[-1]) == pytest.approx(
        10_000.0 + 5_000.0 - entry - liquidation)


def test_vol_target_hold_reaches_the_target_gross_exposure():
    closes = {"A": 100.0 * (1.0 + 0.001) ** np.arange(240),
              "B": 100.0 * (1.0 - 0.0005) ** np.arange(240)}
    panel, prices = _windowed(closes)
    overlay = _FixedOverlay(forecast=0.30, target_vol_pct=0.45)   # scale 1.5
    result = simulate_vol_target(prices, panel, overlay, Costs(_StubConfig(),
                                                              ["A", "B"]))
    # Bar 0 is the first decision of the first UTC day; the exposure then drifts
    # with the marks until the next daily rebalance, so gross/equity is 1.5 plus
    # the day's return and minus the cost drag.
    assert float(result["exposure"].iloc[0]) == pytest.approx(1.5, rel=1e-3)
    assert float(result["exposure"].iloc[1]) == pytest.approx(1.5, rel=3e-3)
    assert result["rebalances"] == 11        # 10 daily decisions + the liquidation


def test_vol_target_no_leverage_variant_never_exceeds_full_investment():
    panel, prices = _windowed({"A": 100.0 * (1.0 + 0.001) ** np.arange(240)})
    overlay = _FixedOverlay(forecast=0.10, target_vol_pct=0.45, max_scale=1.0)
    result = simulate_vol_target(prices, panel, overlay,
                                 Costs(_StubConfig(), ["A"]))
    # The scale is pinned at 1.0, so gross/equity can only exceed 1 by the cost
    # drag charged out of cash (0.055 % per rebalance).
    assert float(result["exposure"].max()) <= 1.0 + 1e-3
    assert float(result["exposure"].iloc[0]) == pytest.approx(1.0, rel=1e-3)


def test_per_leg_scaling_uses_each_legs_own_forecast():
    panel, prices = _windowed({"A": 100.0 * (1.0 + 0.002) ** np.arange(120),
                               "B": 100.0 * (1.0 - 0.001) ** np.arange(120)})
    overlay = _FixedOverlay(forecast=0.10, target_vol_pct=0.45)   # clipped to 2.0
    result = simulate_vol_target(prices, panel, overlay,
                                 Costs(_StubConfig(), ["A", "B"]), per_leg=True)
    # Each leg is scaled by 2.0 and gets half the equity: gross = 2 x equity.
    assert float(result["exposure"].iloc[0]) == pytest.approx(2.0, rel=5e-3)


def test_proportional_scaling_preserves_relative_weights():
    """The overlay must not silently rebalance — that is what arm Hreb is for."""
    panel, prices = _windowed({"A": 100.0 * (1.0 + 0.01) ** np.arange(120),
                               "B": 100.0 * (1.0 - 0.001) ** np.arange(120)})
    overlay = _FixedOverlay(forecast=0.45, target_vol_pct=0.45)   # scale 1.0
    costs = Costs(_StubConfig(), ["A", "B"])
    proportional = simulate_vol_target(prices, panel, overlay, costs)
    rebalanced = simulate_vol_target(prices, panel, overlay, costs,
                                     rebalance_to_equal=True)
    # A compounds 1 %/bar, B is flat: the drifting book lets A's weight grow, so
    # it ends materially above the daily-rebalanced one.
    assert float(proportional["equity"].iloc[-1]) > float(rebalanced["equity"].iloc[-1])
    assert float(proportional["exposure"].iloc[0]) == pytest.approx(1.0, rel=1e-3)
    # Inside a single UTC day there is no rebalance to differ on, so the two
    # rules must agree bar for bar over the first 23 bars.
    assert list(proportional["equity"].iloc[:23].round(6)) == \
        list(rebalanced["equity"].iloc[:23].round(6))


# ── 3. metrics ──────────────────────────────────────────────────────────

def test_metrics_on_a_flat_curve_has_no_risk_and_no_drawdown():
    index = pd.date_range("2026-01-01", periods=240, freq="1h")
    equity = pd.Series(10_000.0, index=index)
    out = metrics(equity, pd.Series(1.0, index=index), pd.Series(1.0, index=index),
                  0.0, 0.0, 0)
    assert out["total_return_pct"] == pytest.approx(0.0)
    assert out["ann_vol_pct"] == pytest.approx(0.0)
    assert out["sharpe"] == pytest.approx(0.0)
    assert out["max_dd_pct"] == pytest.approx(0.0)
    assert out["calmar"] is None
    assert out["time_in_market_pct"] == pytest.approx(100.0)


def test_metrics_max_drawdown_and_calmar_on_a_known_curve():
    index = pd.date_range("2026-01-01", periods=366, freq="1D")   # exactly 365 days
    equities = np.concatenate([
        np.linspace(10_000.0, 12_000.0, 100),        # up
        np.linspace(12_000.0, 9_000.0, 100),         # -25 % drawdown
        np.linspace(9_000.0, 13_140.0, 166),         # recover
    ])
    equity = pd.Series(equities, index=index)
    out = metrics(equity, pd.Series(1.0, index=index), pd.Series(1.0, index=index),
                  100.0, 0.0, 0)
    assert out["max_dd_pct"] == pytest.approx(25.0, abs=1e-6)
    assert out["total_return_pct"] == pytest.approx(31.4, abs=1e-6)
    assert out["cagr_pct"] == pytest.approx(31.4, abs=1e-6)
    assert out["calmar"] == pytest.approx(31.4 / 25.0, rel=1e-6)
    # a 100 USDT cost on a 3 040 USDT net gain is 100/3 240 of the gross 3 140
    assert out["cost_share_of_gross_pct"] == pytest.approx(
        100.0 / 3_240.0 * 100.0, abs=1e-4)
    assert out["turnover_x_per_year"] == pytest.approx(0.0)


def test_metrics_sharpe_matches_the_repo_scorer_basis():
    from core.ga.fitness import daily_returns, per_period_sharpe

    rng = np.random.default_rng(20261002)
    index = pd.date_range("2026-01-01", periods=2400, freq="1h")
    returns = rng.normal(0.0004, 0.01, size=index.size)
    equity = pd.Series(10_000.0 * np.cumprod(1.0 + returns), index=index)
    out = metrics(equity, pd.Series(1.0, index=index), pd.Series(1.0, index=index),
                  0.0, 0.0, 0)
    curve = [{"time": t, "equity": float(v)} for t, v in equity.items()]
    daily = daily_returns(curve)
    span = (index[-1] - index[0]).total_seconds() / 86400.0
    expected = per_period_sharpe(daily) * np.sqrt(len(daily) * 365.0 / span)
    assert out["sharpe"] == pytest.approx(expected, abs=1e-4)
    assert out["daily_observations"] == len(daily)


def test_sortino_only_penalises_downside():
    index = pd.date_range("2026-01-01", periods=400, freq="1D")
    up = pd.Series(10_000.0 * (1.0 + 0.002) ** np.arange(400), index=index)
    out = metrics(up, pd.Series(1.0, index=index), pd.Series(1.0, index=index),
                  0.0, 0.0, 0)
    assert out["sortino"] == pytest.approx(0.0)      # no downside -> undefined -> 0.0
    assert out["max_dd_pct"] == pytest.approx(0.0)


# ── 4. matched risk ────────────────────────────────────────────────────

def test_matched_risk_scales_to_the_reference_volatility():
    rets_a = np.array([0.01, -0.01, 0.02, -0.02, 0.01, -0.01])
    rets_b = 2.0 * rets_a
    arm = {"total_return_pct": 10.0, "_matched_returns": rets_b,
           "sharpe": 1.0, "daily_observations": rets_b.size,
           "skew": 0.0, "kurtosis": 3.0}
    ref = {"total_return_pct": 5.0, "_matched_returns": rets_a}
    out = matched_risk(arm, ref)
    assert out["k"] == pytest.approx(0.5)            # sigma_ref / sigma_arm
    assert out["compounded_pct"] == pytest.approx(
        (np.prod(1.0 + 0.5 * rets_b) - 1.0) * 100.0, abs=1e-4)
    assert out["linear_pct"] == pytest.approx(5.0)
    assert out["fallback"] is False
    assert out["k_extrapolation"] is False


def test_matched_risk_is_the_identity_for_the_reference_arm():
    rets = np.array([0.01, -0.005, 0.002, -0.001])
    total = (np.prod(1.0 + rets) - 1.0) * 100.0
    arm = {"total_return_pct": total, "_matched_returns": rets}
    out = matched_risk(arm, {"total_return_pct": total, "_matched_returns": rets})
    assert out["k"] == pytest.approx(1.0)
    assert out["compounded_pct"] == pytest.approx(total, abs=1e-4)


def test_matched_risk_flags_an_extrapolated_scale():
    rets = np.array([0.001, -0.001, 0.002])
    arm = {"total_return_pct": 0.2, "_matched_returns": rets}
    ref = {"_matched_returns": 10.0 * rets}
    out = matched_risk(arm, ref)
    assert out["k"] == pytest.approx(10.0)
    assert out["k_extrapolation"] is True


def test_matched_risk_falls_back_when_a_volatility_is_zero():
    out = matched_risk({"total_return_pct": 1.0, "_matched_returns": np.zeros(5)},
                       {"_matched_returns": np.array([0.01, -0.01])})
    assert out["fallback"] is True and out["k"] == 1.0


# ── 5. basket rule and the overlay's forecast plumbing ─────────────────

def test_basket_rule_excludes_stables_and_partial_coverage(tmp_path):
    full = pd.date_range("2025-04-01", "2026-10-01", freq="1h")
    for name in ("BTCUSDT", "ETHUSDT", "USDCUSDT"):
        frame = pd.DataFrame({"close": np.linspace(100.0, 200.0, len(full))},
                             index=full)
        (tmp_path / name).mkdir()
        frame.to_parquet(tmp_path / name / "1h.parquet")
    late = pd.date_range("2026-09-23", "2026-10-01", freq="1h")
    (tmp_path / "ENAUSDT").mkdir()
    pd.DataFrame({"close": np.linspace(1.0, 2.0, len(late))},
                 index=late).to_parquet(tmp_path / "ENAUSDT" / "1h.parquet")
    (tmp_path / "MOVRUSDT").mkdir()                 # no 1h file at all
    symbols, meta = basket_symbols(tmp_path)
    assert symbols == ["BTCUSDT", "ETHUSDT"]
    assert "USD-stable" in meta["excluded"]["USDCUSDT"]
    assert "partial" in meta["excluded"]["ENAUSDT"]
    assert "no 1h.parquet" in meta["excluded"]["MOVRUSDT"]


def test_basket_levels_is_the_equal_weight_rebalanced_portfolio():
    index = pd.date_range("2026-01-01", periods=4, freq="1h")
    closes = pd.DataFrame({"A": [100.0, 110.0, 110.0, 121.0],
                           "B": [100.0, 100.0, 100.0, 100.0]}, index=index)
    level = basket_levels(closes)
    # per-bar basket returns: +5 %, 0 %, +5 %
    assert list(level.round(8)) == [1.0, 1.05, 1.05, 1.1025]


def test_overlay_forecast_reads_the_repo_estimator_on_real_frames():
    from core.ml.volatility import forecast_vol, to_pct

    frame = _frame(100.0 * (1.0 + 0.001) ** np.arange(600))
    overlay = Overlay()
    mine = overlay.forecast_pct({"A": frame})
    theirs = to_pct(forecast_vol(frame, method="ewma", window=500, lam=0.94,
                                interval="1h"))
    assert mine == pytest.approx(theirs)


# ── 6. DSR + the pre-stated verdict ────────────────────────────────────

def test_dsr_falls_with_trials_and_is_definitionally_zero_at_one_trial():
    arm = {"sharpe": 1.6, "daily_observations": 92, "skew": 0.0, "kurtosis": 3.0}
    assert dsr_for(arm, 1)["dsr"] == 0.0
    twenty_five = dsr_for(arm, 25)["dsr"]
    assert twenty_five < dsr_for(arm, 5)["dsr"]
    assert twenty_five == pytest.approx(1.6 / np.sqrt(365)
                                        - np.sqrt(1.0 / 92) * np.sqrt(2 * np.log(25)),
                                        rel=1e-6)


def _row(arm, window, total, calmar, retention, dsr):
    return {"arm": arm, "window": window, "total_return_pct": total,
            "calmar": calmar, "matched_to_H": {"compounded_pct": retention},
            "dsr": {"dsr": dsr}}


def test_verdict_requires_every_pre_stated_condition():
    windows = ["W0", "W1", "W2", "W3"]
    hold = [_row("H", w, 10.0, 1.0, 10.0, 0.0) for w in windows]
    good = [_row("V", w, 12.0, 2.0, 9.0, 0.01) for w in windows]
    out = verdict(hold + good)
    assert out["passes"] == {"T1": True, "T2": True, "T3": True}
    assert out["recommend_enable"] is True
    # T1 broken in ONE window is enough to refuse.
    weak = [_row("V", w, 12.0, 2.0 if w != "W2" else 0.5, 9.0, 0.01) for w in windows]
    assert verdict(hold + weak)["passes"]["T1"] is False
    # T2: keep 80 % of H's return, not less.
    thin = [_row("V", w, 8.0, 2.0,
                 10.0 * THRESHOLD["T2_matched_risk_return_retention"] - 0.01, 0.01)
            for w in windows]
    assert verdict(hold + thin)["passes"]["T2"] is False
    # T3: 2 of 4 positive is enough, 1 of 4 is not.
    one = [_row("V", w, 12.0, 2.0, 9.0, 0.01 if w == "W0" else -0.01) for w in windows]
    assert verdict(hold + one)["passes"]["T3"] is False
    assert "leave risk.vol_targeting.enabled" in verdict(hold + weak)["recommendation"]


def test_verdict_t2_reads_losses_in_the_right_direction():
    """When H loses money the test is "V must not lose more", not a ratio."""
    windows = ["W0", "W1"]
    hold = [_row("H", w, -10.0, -1.0, -10.0, 0.0) for w in windows]
    # V loses less than H at matched risk -> T2 holds even though the "ratio" is
    # only 50 %.
    better = [_row("V", w, -5.0, 5.0, -5.0, 0.01) for w in windows]
    assert verdict(hold + better)["passes"]["T2"] is True
    # V loses more than H -> T2 fails.
    worse = [_row("V", w, -20.0, 5.0, -20.0, 0.01) for w in windows]
    assert verdict(hold + worse)["passes"]["T2"] is False


def test_rescaled_daily_returns_compound_to_the_total_return():
    index = pd.date_range("2026-01-01", periods=2400, freq="1h")
    equity = pd.Series(10_000.0 * (1.0 + 0.0003) ** np.arange(2400), index=index)
    rets = rescaled_daily_returns(equity, 10_000.0)
    total = float(equity.iloc[-1]) / 10_000.0 - 1.0
    assert float(np.prod(1.0 + rets) - 1.0) == pytest.approx(total, rel=1e-9)
    # ... and the repo's own basis does NOT, which is why this helper exists.
    from core.ga.fitness import daily_returns
    repo = daily_returns([{"time": t, "equity": float(v)} for t, v in equity.items()])
    assert float(np.prod(1.0 + repo) - 1.0) < total


def test_grid_size_is_the_declared_trial_count():
    assert len(GRID_ESTIMATORS) * len(GRID_TARGETS) * len(GRID_WINDOWS) == 24
    assert "garch11" not in GRID_ESTIMATORS       # too slow for the bounded budget
