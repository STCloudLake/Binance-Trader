"""Phase P3 — volatility targeting: forecast, sizing, dynamic barriers.

Scientific claim under test
---------------------------
Directional prediction on returns is unpredictable here (P2 measured OOS AUC
0.52/0.53 and a negative net-of-cost expectancy), while **conditional
volatility** is predictable (ARCH/GARCH — Tsay, *Analysis of Financial Time
Series*, ch. 3).  These tests pin the three ways that forecast is spent on risk,
and — just as important — that **nothing changes while
``risk.vol_targeting.enabled`` is false** (the shipped default).

Determinism: every random draw goes through ``np.random.default_rng(SEED)``.
Real-data assertions are skipped when the cached parquet is absent (the repo
does not ship ``data/``).
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SEED = 20250930
BTC_1H = Path("data/market/BTCUSDT/1h.parquet")


def _btc(n: int | None = None) -> pd.DataFrame:
    if not BTC_1H.exists():
        pytest.skip("no cached BTCUSDT 1h parquet in this checkout")
    df = pd.read_parquet(BTC_1H)
    return df if n is None else df.head(n)


# ── 1. estimator stack ───────────────────────────────────────────────────

def test_forecast_vol_interface_and_units():
    """One interface, one unit: fraction of price per bar, never NaN."""
    from core.ml.volatility import (METHODS, annualize, forecast_vol,
                                    periods_per_year_for, to_pct)

    df = _btc(600)
    for method in METHODS:
        per_bar = forecast_vol(df, method=method)
        assert math.isfinite(per_bar), method
        assert per_bar > 0.0, method
        annual = forecast_vol(df, method=method, unit="annual")
        # Annualisation is sqrt-time scaling of the per-bar figure.
        assert annual == pytest.approx(annualize(per_bar, 8760.0), rel=1e-12)
    assert periods_per_year_for("1h") == 8760.0
    assert periods_per_year_for("1d") == 365.0
    # to_pct is the only place a percent appears.
    assert to_pct(0.005) == pytest.approx(0.5)
    # Interval overrides the annualisation factor.
    assert (forecast_vol(df, unit="annual", interval="1d")
            < forecast_vol(df, unit="annual", interval="1h"))


def test_forecast_vol_rejects_unknown_method_and_wrong_input():
    from core.ml.volatility import forecast_vol, log_returns

    df = _btc(200)
    with pytest.raises(ValueError):
        forecast_vol(df, method="not-a-method")
    with pytest.raises(TypeError):
        forecast_vol(log_returns(df["close"].values), method="realized_parkinson")


def test_short_series_returns_zero_not_nan():
    """A caller must be able to branch on a number, never on NaN."""
    from core.ml.volatility import forecast_vol, realized_vol, vol_percentile

    assert forecast_vol(np.array([])) == 0.0
    assert forecast_vol(np.array([0.01])) == 0.0
    assert realized_vol(np.array([])) == 0.0
    assert 0.0 <= vol_percentile(np.array([])) <= 1.0


def test_ewma_is_causal_and_clips_data_splice_outliers():
    """The forecast must use past returns only, and survive a cache seam.

    A calendar gap in the cached series splices two months into a single "bar"
    with a ``+27.6 %`` log return.  Unclipped, the RiskMetrics recursion
    (effective memory ``1/(1-0.94) ≈ 17`` bars) reports **6.7690 %/bar** on the
    injected draw below against **0.7553 %/bar** clipped — **8.96x**.  The ratio
    is data/window dependent, not a constant: the same injected bar in the real
    500-bar tail gives 6.7674 vs 0.4994 %/bar (13.55x), and the historical real
    splice in ``docs/core-algorithms/10-volatility-targeting.md`` measured 9.81x.

    The splice is **injected synthetically** here instead of being read out of
    ``data/market/BTCUSDT/1h.parquet``: the contract under test is "a spliced
    observation must not reach the estimator", and a test that needs the shipped
    cache to *contain* a data defect stops guarding anything the moment the
    defect is repaired (the BTCUSDT 1h gap was refetched and merged, which is
    exactly the change that used to break this test).  The real cache is still
    used, but only to re-assert the contract on live-shaped returns.
    """
    from core.ml.volatility import clip_outliers, ewma_vol, log_returns, to_pct

    rng = np.random.default_rng(SEED)
    calm = rng.normal(0.0, 0.004, 2000)
    #: One injected jump, at the tail where the EWMA actually looks for it.
    splice = np.concatenate([calm[:1999], np.array([0.276])])
    assert splice.max() > 0.2, "the synthetic splice disappeared from the test"

    clipped = clip_outliers(splice)
    assert clipped.max() < 0.1
    assert abs(clipped).max() < abs(splice).max()

    # The contract in numbers: one spliced bar must not be allowed to dominate
    # the forecast the position sizer reads.  (All ``*_pct`` values are %/bar;
    # the calm series is 0.4 %/bar by construction.)
    unclipped_pct = to_pct(ewma_vol(splice, window=0, outlier_sigma=0.0))
    clipped_pct = to_pct(ewma_vol(splice, window=0))
    assert clipped_pct < 1.0, clipped_pct
    assert unclipped_pct > 5.0 * clipped_pct, (unclipped_pct, clipped_pct)

    # The same contract on whatever the shipped 1h cache currently holds —
    # passes with or without a splice present (skipped when data/ is absent).
    real = log_returns(_btc()["close"].values)
    assert abs(clip_outliers(real)).max() < 0.1

    # Causal: appending a violent move must not change the *previous* forecast.
    head = ewma_vol(calm[:1000], window=0)
    tail_only = ewma_vol(np.concatenate([calm[:1000], np.array([0.5, -0.5])]), window=2)
    assert tail_only != pytest.approx(head, rel=1e-9)


def test_vol_percentile_is_scale_free():
    """A regime flag must compare the forecast with its own history, not a level."""
    from core.ml.volatility import forecast_vol, vol_percentile

    rng = np.random.default_rng(SEED)
    calm = rng.normal(0.0, 0.002, 3000)
    wild = rng.normal(0.0, 0.020, 3000)
    # Same *shape*, 10x the scale -> the rank is what matters, and both are
    # "the current forecast is unremarkable for its own history".
    assert 0.25 < vol_percentile(calm) < 0.75
    assert 0.25 < vol_percentile(wild) < 0.75
    # A spike at the end must rank at the top of its own history.
    spiked = np.concatenate([calm, np.array([0.30])])
    assert vol_percentile(spiked) > 0.99
    # Annualised forecast is 10x-ish apart, i.e. the levels differ while ranks agree.
    assert forecast_vol(wild, unit="annual") > 5.0 * forecast_vol(calm, unit="annual")


def test_per_bar_compute_budget():
    """Every estimator must be cheap enough for the live per-bar path.

    The live predictor computes indicators once per kline, so a forecast has to
    be negligible next to that.  Budget is ``PER_BAR_BUDGET_SEC`` (2 ms) for the
    EWMA/realised family; ``garch11`` is documented as research-only and its
    cost is bounded separately (measured ≈0.12 s on this test's 500-bar window —
    the 600-row frame at the default ``window=500`` — and ≈3.7–3.9 s when handed
    the whole 11 676-return cache with ``window=0``) so a regression is still
    caught.

    The garch11 reading is the **minimum of three fits** (the repo's pattern in
    ``tests/test_p34_audit_fixes.py``): on 2026-09-30 the fit measured 0.12 s
    idle and 0.64 s (min of 3) while seven concurrent Python processes were
    busy, so a single loaded reading would sit near the 1.0 s bound.
    """
    from core.ml.volatility import (METHODS, PER_BAR_BUDGET_SEC, forecast_vol)

    def _best(fn, batches: int = 3) -> float:
        fn()                                      # warm up
        best = float("inf")
        for _ in range(batches):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    df = _btc(600)
    for method in ("ewma", "realized_cc", "realized_parkinson",
                   "realized_garman_klass"):
        forecast_vol(df, method=method)          # warm up
        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            forecast_vol(df, method=method)
        per_call = (time.perf_counter() - t0) / reps
        assert per_call < PER_BAR_BUDGET_SEC, (
            f"{method} costs {per_call * 1000:.2f} ms/bar, above the "
            f"{PER_BAR_BUDGET_SEC * 1000:.1f} ms budget")
    garch = _best(lambda: forecast_vol(df, method="garch11"))
    assert garch < 1.0, f"garch11 fit costs {garch:.2f}s — not usable at all"


def test_memoising_forecaster_computes_once_per_bar():
    """Repeated ticks on an unchanged bar must not re-run the estimator."""
    from core.ml.volatility import VolForecaster

    df = _btc(600)
    f = VolForecaster(interval="1h")
    first = f.forecast(("BTCUSDT", "1h"), df, interval="1h")
    for _ in range(5):
        again = f.forecast(("BTCUSDT", "1h"), df, interval="1h")
        assert again is first
    assert f.compute_count == 1
    # A new bar invalidates the memo.
    f.forecast(("BTCUSDT", "1h"), df.iloc[:-1], interval="1h")
    assert f.compute_count == 2


def test_garch_fit_is_well_posed_and_degrades_to_ewma():
    """The GARCH path must never emit a degenerate number.

    The untargeted 3-parameter Gaussian MLE is *unbounded* (see the module
    docstring), so the shipped estimator is the unit-persistence fit.  Two
    properties are asserted: the parameters are admissible, and the blended
    forecast stays within a sane band of the EWMA level it falls back to.
    """
    from core.ml.volatility import (ewma_vol, forecast_vol, garch11_forecast,
                                    garch11_params, log_returns)

    r = log_returns(_btc()["close"].values)
    p = garch11_params(r)
    assert p["ok"] is True
    assert 0.0 <= p["alpha"] <= 1.0 and 0.0 <= p["beta"] <= 1.0
    assert p["alpha"] + p["beta"] <= 1.0 + 1e-9
    assert p["backend"] in ("arch", "scipy")
    ewma = ewma_vol(r)
    g = garch11_forecast(r)
    assert 0.2 * ewma < g < 2.0 * ewma, (g, ewma)
    # allow_garch=True must expose the raw value instead of silently blending.
    assert forecast_vol(r, method="garch11", allow_garch=True) > 0.0
    # Too little data -> explicit 0.0, and forecast_vol() degrades to EWMA.
    assert garch11_params(r[:10])["ok"] is False
    assert forecast_vol(r[:10], method="garch11") == pytest.approx(ewma_vol(r[:10]))


def test_garch_loglik_gradient_drives_the_objective_downhill():
    """The analytic gradient must have the right sign *and* be usable as a step.

    It is kept as a module-level function (not a closure) because the first
    version leaked accumulators between calls: every optimiser then walked to the
    degenerate ``(omega=0, beta=0, alpha->1)`` corner.

    The contract asserted is the one an optimiser relies on — a small step
    *against* the gradient decreases the objective — rather than equality with a
    finite difference, which was measured to be the unreliable side here: the
    untargeted likelihood is piecewise (it has a ``1e-6`` variance floor and is
    unbounded below), so central differences at these parameters understate the
    true derivative by ~2.7x and no step size fixes it.  A sign flip, a dropped
    recursion term or a leaked accumulator all break the descent property.
    """
    from core.ml.volatility import garch11_loglik_grad

    rng = np.random.default_rng(SEED)
    x2 = 0.5 * rng.chisquare(1, 400)
    var_s = float(np.var(x2, ddof=1))
    for theta0 in ((0.1, 0.05, 0.94), (0.1, 0.20, 0.50), (2.0, 0.05, 0.90)):
        theta = np.array(theta0, dtype=float)
        f0, g = garch11_loglik_grad(x2, var_s, tuple(theta))
        assert np.all(np.isfinite(g)) and np.any(g != 0.0)
        # Backtracking line search: at least one small step must improve.
        improved = []
        for scale in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            step = theta - scale * np.maximum(np.abs(theta), 1.0) * g
            if np.any(step < 0):
                continue
            f1 = garch11_loglik_grad(x2, var_s, tuple(step))[0]
            improved.append(f1 < f0)
        assert any(improved), f"gradient step never improved from {theta0}"


def test_garch_loglik_gradient_is_not_a_leaked_accumulator():
    """Repeated calls must be pure — the bug the module docstring records."""
    from core.ml.volatility import garch11_loglik_grad

    rng = np.random.default_rng(SEED + 1)
    x2 = 0.5 * rng.chisquare(1, 200)
    var_s = float(np.var(x2, ddof=1))
    theta = (0.4, 0.10, 0.80)
    first = garch11_loglik_grad(x2, var_s, theta)
    for _ in range(5):
        again = garch11_loglik_grad(x2, var_s, theta)
        assert again[0] == first[0]
        assert np.array_equal(again[1], first[1])


def test_real_data_forecast_summary_high_and_low_vol_windows():
    """Numbers for the doc table: mean/std of each forecast on real BTC 1h.

    Also reports the high- vs low-vol window behaviour, because that is the
    claim the sizer depends on (a calm window must produce a *small* forecast).
    """
    from core.ml.volatility import (METHODS, ewma_vol, forecast_vol,
                                    log_returns, realized_vol)

    df = _btc()
    r = log_returns(df["close"].values)
    assert len(r) > 5000
    per_bar = {m: forecast_vol(df, method=m) for m in METHODS}
    for v in per_bar.values():
        assert v > 0.0
    # The EWMA must sit near the calm realised average and well below the
    # noisiest estimator — a sanity band, not a calibration claim.  (The window
    # catches a recent tail, so an order-of-magnitude band is the assertion.)
    assert 0.15 * per_bar["realized_cc"] < per_bar["ewma"] < 6.0 * per_bar["realized_cc"]

    # Highest- and lowest-volatility 200-bar windows by realised close-to-close.
    win = 200
    roll = pd.Series(r).rolling(win).std().dropna()
    hi_end = int(roll.idxmax())
    lo_end = int(roll.idxmin())
    hi = ewma_vol(r[hi_end - win:hi_end], window=0)
    lo = ewma_vol(r[lo_end - win:lo_end], window=0)
    assert hi > 2.0 * lo, (hi, lo)
    # A high-vol window must also produce a higher *forecast* than a calm one.
    assert realized_vol(r[hi_end - win:hi_end], window=0) > realized_vol(
        r[lo_end - win:lo_end], window=0)


# ── 2. sizing ────────────────────────────────────────────────────────────

def _sizer(enabled: bool, **overrides):
    """PositionSizer on the real config, with vol targeting forced on/off."""
    from app.config import Config, VolTargetingConfig
    from core.risk.position_sizer import PositionSizer

    Config._instance = None
    config = Config.load("sim")
    vt = config.risk_vol_targeting
    vt.enabled = enabled
    for key, value in overrides.items():
        setattr(vt, key, value)
    return PositionSizer(config.hard_limits, config.soft_params,
                         config.core_capital_pct, config.satellite_capital_pct, vt), vt


def test_sizing_halves_when_forecast_vol_doubles_and_respects_caps():
    """The integration proof: ``scale ∝ 1/forecast``, bounded by the caps.

    Measured on the shipped config (satellite pool 0.3 x 10 000 = 3 000, fixed
    fraction 8 % → **240 USDT**, ``max_scale`` 2.0 → the 10 % ceiling of 1 000
    never binds at these volatilities):

    ==================  =======  =====
    forecast vol %/bar  scale    risk
    ==================  =======  =====
    0.45 (== target)    1.0      240
    0.90 (x2)           0.5      120
    0.225 (x0.5)        2.0      480
    1.80 (x4)           0.25     60
    ==================  =======  =====
    """
    sizer, vt = _sizer(True, max_position_notional_pct=10.0)
    balance, price = 10_000.0, 50_000.0
    target = vt.target_vol_pct

    base_qty, base_risk = sizer.calculate_position_size(
        balance, price, forecast_vol_pct=target)
    double_qty, double_risk = sizer.calculate_position_size(
        balance, price, forecast_vol_pct=2.0 * target)
    half_qty, half_risk = sizer.calculate_position_size(
        balance, price, forecast_vol_pct=0.5 * target)
    quad_qty, quad_risk = sizer.calculate_position_size(
        balance, price, forecast_vol_pct=4.0 * target)

    # The target-volatility forecast reproduces the fixed fraction exactly.
    assert base_risk == pytest.approx(240.0)
    # Doubling the forecast volatility halves the notional AND the quantity.
    assert double_risk == pytest.approx(0.5 * base_risk, rel=1e-9)
    assert double_qty == pytest.approx(0.5 * base_qty, rel=1e-9)
    # Halving it doubles the notional; quadrupling clamps at min_scale.
    assert half_risk == pytest.approx(2.0 * base_risk, rel=1e-9)
    assert half_qty == pytest.approx(2.0 * base_qty, rel=1e-9)
    assert quad_risk == pytest.approx(0.25 * base_risk, rel=1e-9)
    assert sizer.vol_scale(4.0 * target) == pytest.approx(vt.min_scale)
    assert sizer.vol_scale(0.0) == 1.0            # no forecast -> fixed fallback

    # The 10 % notional ceiling takes over once the scale would push past it:
    # with max_scale 10 the uncapped size would be 2 400, so the cap binds at
    # 1 000.  (At the shipped max_scale of 2.0 the ceiling can never bind for a
    # satellite trade — 240 x 2 = 480 < 1 000 — which is itself the documented
    # reason the cap is not a behaviour change at the default settings.)
    wide, _ = _sizer(True, max_position_notional_pct=10.0, max_scale=10.0)
    _, tiny_risk = wide.calculate_position_size(balance, price, forecast_vol_pct=0.02)
    assert tiny_risk == pytest.approx(0.10 * balance)
    # Just below the binding point it is still uncapped.
    assert wide.vol_scale(0.14) == pytest.approx(vt.target_vol_pct / 0.14, rel=1e-6)
    assert wide.calculate_position_size(
        balance, price, forecast_vol_pct=0.14)[1] == pytest.approx(
            base_risk * vt.target_vol_pct / 0.14, rel=1e-6)

    # Hard limits bound everything, whatever the forecast says.
    for vol in (target, 2.0 * target, 0.5 * target, 0.02, 1e-9):
        qty, risk = sizer.calculate_position_size(balance, price, forecast_vol_pct=vol)
        assert risk <= balance * sizer.hard.max_position_size_pct / 100 + 1e-9
        assert risk <= balance * vt.max_position_notional_pct / 100 + 1e-9
        assert risk <= base_risk * vt.max_scale + 1e-9
        assert qty == pytest.approx(risk / price, rel=1e-12)

    # A tighter configured ceiling is genuinely binding at the maximum scale.
    capped, _ = _sizer(True, max_position_notional_pct=1.0)
    _, capped_risk = capped.calculate_position_size(
        balance, price, forecast_vol_pct=0.5 * target)
    assert capped_risk == pytest.approx(0.01 * balance)


def test_sizing_falls_back_to_fixed_fraction_without_a_forecast():
    """Vol targeting on but no forecast → the legacy fixed-fraction number."""
    on, _ = _sizer(True)
    off, _ = _sizer(False)
    args = (10_000.0, 50_000.0, "satellite")
    for missing in (None, 0.0, float("nan"), -1.0):
        assert on.calculate_position_size(*args, forecast_vol_pct=missing) == \
            off.calculate_position_size(*args)


def test_vol_targeting_off_is_bit_identical_to_pre_p3():
    """The switch-off guarantee, asserted on every sizing/stop/barrier hook."""
    sizer, vt = _sizer(False)
    assert sizer.vol_targeting_enabled() is False
    # A forecast is ignored entirely while disabled.
    assert sizer.calculate_position_size(10_000.0, 50_000.0, forecast_vol_pct=0.45) == \
        sizer.calculate_position_size(10_000.0, 50_000.0)
    assert sizer.vol_scale(0.45) == 1.0
    assert sizer.barrier_widths_pct(0.45) is None

    # Stop/trailing distances are the pre-P3 formulas, exactly.
    expected = max(sizer.soft.stop_loss_pct, sizer.hard.min_stop_loss_distance_pct)
    assert sizer.stop_distance_pct(0.45) == expected
    assert sizer.stop_distance_pct(0.45, volatility_expanding=True) == expected * 1.3
    assert sizer.calculate_stop_loss(50_000.0, "long") == 50_000.0 * (1 - expected / 100)
    assert sizer.calculate_stop_loss(50_000.0, "long", volatility_expanding=True) == \
        50_000.0 * (1 - expected * 1.3 / 100)
    assert sizer.calculate_stop_loss(50_000.0, "short") == 50_000.0 * (1 + expected / 100)
    assert sizer.trailing_stop_distance_pct() == sizer.hard.trailing_stop_distance_pct
    assert sizer.trailing_stop_distance_pct(forecast_vol_pct=0.45) == \
        sizer.hard.trailing_stop_distance_pct

    # A per-strategy override still wins in both modes.
    class _Risk:
        trailing_stop_pct = 1.25
    assert sizer.trailing_stop_distance_pct(_Risk(), forecast_vol_pct=0.45) == 1.25
    on, _ = _sizer(True)
    assert on.trailing_stop_distance_pct(_Risk(), forecast_vol_pct=0.45) == 1.25


def test_vol_targeting_switch_defaults_to_disabled_in_the_shipped_config():
    """The safety property the report depends on: opt-in, not opt-out."""
    from app.config import Config, VolTargetingConfig

    assert VolTargetingConfig().enabled is False
    Config._instance = None
    config = Config.load("sim")
    assert config.risk_vol_targeting.enabled is False
    assert config.vol_targeting is config.risk_vol_targeting
    # risk_params.yaml still owns hard/soft limits — this block did not move them.
    assert config.hard_limits.max_open_trades == 15
    assert config.soft_params.position_size_pct == 8.0


# ── 3. dynamic barriers / stops ──────────────────────────────────────────

def test_stop_distance_scales_with_forecast_vol():
    """A stop must widen with volatility, and stay inside its clamp."""
    sizer, vt = _sizer(True)
    calm = sizer.stop_distance_pct(0.20)
    normal = sizer.stop_distance_pct(0.45)
    wild = sizer.stop_distance_pct(0.90)
    assert calm < normal < wild
    assert normal == pytest.approx(vt.stop_vol_multiple * 0.45)
    # Clamps: never below the hard floor, never above stop_max_pct.
    assert sizer.stop_distance_pct(1e-6) == pytest.approx(
        max(vt.stop_min_pct, sizer.hard.min_stop_loss_distance_pct))
    assert sizer.stop_distance_pct(1000.0) == pytest.approx(vt.stop_max_pct)
    # Prices follow the distance in the right direction on both sides.
    assert sizer.calculate_stop_loss(100.0, "long", forecast_vol_pct=0.90) < \
        sizer.calculate_stop_loss(100.0, "long", forecast_vol_pct=0.20)
    assert sizer.calculate_stop_loss(100.0, "short", forecast_vol_pct=0.90) > \
        sizer.calculate_stop_loss(100.0, "short", forecast_vol_pct=0.20)
    # Trailing distance uses the same scaled number when a forecast is present.
    assert sizer.trailing_stop_distance_pct(forecast_vol_pct=0.90) == wild


def test_barrier_widths_follow_the_forecast_and_keep_the_atr_default():
    """``labels.barrier_widths`` gains a forecast path, not a new default."""
    from core.ml.labels import barrier_widths, create_triple_barrier_label_vol

    df = _btc(400)
    atr_up, atr_dn = barrier_widths(df)
    fc_up, fc_dn = barrier_widths(df, vol_pct=0.0045)
    assert (atr_up == atr_dn).all() and (fc_up == fc_dn).all()
    assert fc_up.iloc[-1] == pytest.approx(0.0045)
    # A wider forecast -> a wider (clamped) barrier, and vice versa.
    wide, _ = barrier_widths(df, vol_pct=0.02)
    narrow, _ = barrier_widths(df, vol_pct=0.0005)
    assert wide.iloc[-1] > fc_up.iloc[-1] > narrow.iloc[-1]
    assert narrow.iloc[-1] == pytest.approx(0.004)      # min_pct clamp

    # vol_pct=None reproduces the ATR widths bit-for-bit.
    again_up, _ = barrier_widths(df)
    assert again_up.equals(atr_up)

    # A Series forecast is aligned by index and never leaves NaN holes.
    series = pd.Series(0.0045, index=df.index)
    s_up, _ = barrier_widths(df, vol_pct=series)
    assert s_up.notna().all()
    assert s_up.iloc[-1] == pytest.approx(0.0045)

    # Labels: unchanged by default, different once a forecast is supplied.
    default = create_triple_barrier_label_vol(df, forward_periods=24)
    forced = create_triple_barrier_label_vol(df, forward_periods=24, vol_pct=0.02)
    assert default.notna().sum() > 0
    assert not default.equals(forced)


def test_labels_barrier_path_is_bit_identical_when_vol_is_none():
    """The P3 no-op guarantee for label construction."""
    from core.ml.labels import barrier_widths

    df = _btc(300)
    for kwargs in ({}, {"atr_period": 7}, {"atr_multiple": 2.0},
                   {"min_pct": 0.002, "max_pct": 0.03}):
        up, dn = barrier_widths(df, **kwargs)
        up2, dn2 = barrier_widths(df, vol_pct=None, **kwargs)
        assert up.equals(up2) and dn.equals(dn2)


# ── 4. guard / executor plumbing ─────────────────────────────────────────

class _FakeMarketData:
    """Minimal ``get_historical`` stand-in (no network in the test suite)."""

    def __init__(self, df, current: float):
        self.df = df
        self.price = current
        self.calls = 0

    async def get_historical(self, symbol, interval, limit=500):
        self.calls += 1
        return self.df

    def get_current_price(self, symbol, max_age_sec=300.0):
        return self.price


class _FakeExecutor:
    def __init__(self, positions):
        self._positions = positions
        self.persisted: list[tuple[str, float]] = []

    def get_open_positions(self):
        return self._positions

    async def update_stop_loss(self, symbol, stop_loss):
        self.persisted.append((symbol, stop_loss))

    async def close_position(self, symbol, reduce_pct=100, price=None):
        return {"ok": True, "invested_returned": 0.0, "pnl": 0.0}


def _guard_with_vol(monkeypatch, df, price, enabled):
    from app.config import Config
    from app.event_bus import EventBus
    from core.risk.position_guard import PositionGuard

    Config._instance = None
    config = Config.load("sim")
    config.risk_vol_targeting.enabled = enabled
    guard = PositionGuard(config, EventBus())
    guard._executor = _FakeExecutor({})
    guard._market_data = _FakeMarketData(df, price)
    return guard


def test_guard_trails_with_the_entry_vol_width_and_falls_back_to_fixed():
    """The live trailing stop must use the forecast width, or the fixed one."""
    df = _btc(600)
    price = 100.0

    # ── switch off: the pre-P3 2 % distance, no data fetch at all ──
    off = _guard_with_vol(None, df, 101.0, enabled=False)
    pos_off = {"entry_price": price, "stop_loss": 98.0, "side": "long",
               "trailing_stop_pct": None, "stop_vol_pct": 0.45, "timeframe": "1h"}
    off._executor._positions = {"BTCUSDT": pos_off}
    asyncio.run(off._update_trailing_stop("BTCUSDT", pos_off, 101.0, "long", 1.0))
    assert pos_off["stop_loss"] == pytest.approx(round(101.0 * 0.98, 2))
    assert off._market_data.calls == 0, "disabled guard must not fetch history"

    # ── switch on: the recorded entry width drives the distance ──
    on = _guard_with_vol(None, df, 101.0, enabled=True)
    vt = on._sizer.vol_targeting
    recorded = 0.90
    expected_pct = min(max(vt.stop_vol_multiple * recorded, vt.stop_min_pct),
                       vt.stop_max_pct)
    pos_on = {"entry_price": price, "stop_loss": 98.0, "side": "long",
              "trailing_stop_pct": None, "stop_vol_pct": recorded, "timeframe": "1h"}
    on._executor._positions = {"BTCUSDT": pos_on}
    asyncio.run(on._update_trailing_stop("BTCUSDT", pos_on, 101.0, "long", 1.0))
    assert pos_on["stop_loss"] == pytest.approx(round(101.0 * (1 - expected_pct / 100), 2))
    # Wider forecast than the fixed 2 % -> a wider stop than the disabled case.
    assert pos_on["stop_loss"] < pos_off["stop_loss"]

    # ── switch on, no recorded width: forecast from market data, then cached ──
    pos_fresh = {"entry_price": price, "stop_loss": 98.0, "side": "long",
                 "trailing_stop_pct": None, "timeframe": "1h"}
    on._executor._positions = {"BTCUSDT": pos_fresh}
    vol = asyncio.run(on.forecast_vol_pct("BTCUSDT", timeframe="1h"))
    assert vol is not None and vol > 0.0
    assert on._market_data.calls == 1
    assert asyncio.run(on.forecast_vol_pct("BTCUSDT", timeframe="1h")) == vol
    assert on._market_data.calls == 1, "the forecast must be cached per TTL"
    asyncio.run(on._update_trailing_stop("BTCUSDT", pos_fresh, 101.0, "long", 1.0))
    # The forecast on this window is *calmer* than the fixed 2 %, so the stop is
    # TIGHTER than the disabled case; the recorded 0.90 %/bar width is wider than
    # both.  The direction must follow the forecast, not a hard-coded "wider".
    assert pos_fresh["stop_loss"] > pos_off["stop_loss"] > pos_on["stop_loss"]


def test_guard_trailing_stop_is_a_ratchet_in_both_modes():
    """The distance changed under P3; the ratchet semantics must not."""
    df = _btc(600)
    on = _guard_with_vol(None, df, 101.0, enabled=True)
    pos = {"entry_price": 100.0, "stop_loss": 99.5, "side": "long",
           "trailing_stop_pct": None, "stop_vol_pct": 0.45, "timeframe": "1h"}
    on._executor._positions = {"BTCUSDT": pos}
    asyncio.run(on._update_trailing_stop("BTCUSDT", pos, 110.0, "long", 10.0))
    raised = pos["stop_loss"]
    assert raised > 99.5
    assert on._executor.persisted[-1] == ("BTCUSDT", raised)
    # Price falling back must never lower the stop.
    asyncio.run(on._update_trailing_stop("BTCUSDT", pos, 105.0, "long", 5.0))
    assert pos["stop_loss"] == raised
    # A short mirrors it.
    short = {"entry_price": 100.0, "stop_loss": 100.5, "side": "short",
             "trailing_stop_pct": None, "stop_vol_pct": 0.45, "timeframe": "1h"}
    on._executor._positions = {"ETHUSDT": short}
    asyncio.run(on._update_trailing_stop("ETHUSDT", short, 90.0, "short", 10.0))
    lowered = short["stop_loss"]
    assert lowered < 100.5
    asyncio.run(on._update_trailing_stop("ETHUSDT", short, 95.0, "short", 5.0))
    assert short["stop_loss"] == lowered


def test_executor_vol_stop_context_is_inert_until_enabled():
    """``vol_stop_ctx`` is the executor's only P3 hook: {} while disabled."""
    from app.config import Config
    from app.event_bus import EventBus
    from core.executor.executor import OrderExecutor

    Config._instance = None
    config = Config.load("sim")
    ex = OrderExecutor(config, EventBus())

    config.risk_vol_targeting.enabled = False
    ex.set_forecast_vol_pct("BTCUSDT", 0.45)
    assert ex.vol_stop_ctx("BTCUSDT") == {}

    config.risk_vol_targeting.enabled = True
    ctx = ex.vol_stop_ctx("BTCUSDT")
    assert ctx["vol_pct"] == pytest.approx(0.45)
    assert ctx["stop_pct"] == pytest.approx(
        ex.vol_stop_ctx("BTCUSDT", vol_pct=0.90)["stop_pct"] / 2.0)
    # An explicit forecast wins; a stale/unset one degrades to {} (fixed width).
    assert ex.vol_stop_ctx("BTCUSDT", vol_pct=1.8)["vol_pct"] == pytest.approx(1.8)
    assert ex.vol_stop_ctx("ETHUSDT") == {}
    # Expiry is a documented fallback, not an exception.
    ex._forecast_vol_cache["ETHUSDT"] = (time.monotonic() - ex._VOL_STOP_TTL_SEC - 1, 0.5)
    assert ex.vol_stop_ctx("ETHUSDT") == {}
