"""Regression tests for the nine defects an independent audit found in P3/P4.

Each test names the defect it pins, quotes the number it is pinning, and asserts
the *property* rather than the constant where a property is what broke.  All
random draws go through ``np.random.default_rng(SEED)``; real-data assertions
skip when the cached parquet is absent.

The defects, and the file each one lives in:

1. look-ahead in the regime HMM (``core/strategy/regime.py``),
2. the microstructure cache ignoring ``as_of_ms`` (``core/market_data/…``),
3. the falsified GARCH rationale (``core/ml/volatility.py``),
4. the Engle-Granger null not matching its own regression (``core/strategy/pairs.py``),
5. inert ``barrier_*`` config keys (reported, not fixed here — see the module note),
6. the dead ``_price_slice_cache`` (``core/backtest/engine.py``),
7. retroactive volatility clipping (``core/ml/volatility.py``),
8. unbounded forecast caches (``core/ml/volatility.py``),
9. documentation numbers (``docs/core-algorithms/10,11``).
"""
from __future__ import annotations

import asyncio
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


def _synth(n: int = 3000, *, seed: int = 5, change_points=(1000, 2000)):
    """Three regimes: (σ=0.002, drift +) → (σ=0.01, drift −) → (σ=0.002, +)."""
    rng = np.random.default_rng(seed)
    spec = [(0.002, +0.0004), (0.010, -0.0008), (0.002, +0.0004)]
    bounds = (0, change_points[0], change_points[1], n)
    rets, truth = [], []
    for k, (sigma, drift) in enumerate(spec):
        m = bounds[k + 1] - bounds[k]
        rets.append(rng.standard_normal(m) * sigma + drift)
        truth += ["high" if k == 1 else "low"] * m
    r = np.concatenate(rets)
    return r, np.asarray(truth), tuple(change_points)


# ── 1. the regime HMM must be causal when it gates ───────────────────────

def test_whole_sample_hmm_labels_move_when_future_bars_are_appended():
    """The defect, pinned: the shipped fit is *not* causal.

    ``hmm_two_state``'s default fits μ/σ/A on the whole sample, so truncating the
    series moves σ (0.00202/0.00968 → 0.00199/0.00968 at t=2000, seed 5) and can
    flip a label *before* the truncation point.  This test asserts the defect
    still has a witness (so the reason the causal path exists cannot quietly
    disappear) without asserting a particular flip count, which is data-dependent.
    """
    from core.strategy.regime import hmm_two_state

    r, _, _ = _synth(seed=5)
    full = hmm_two_state(r)
    trunc = hmm_two_state(r[:2000])
    assert full["causal"] is False
    # σ is fit on the whole sample, so it must differ from the truncated fit.
    assert not np.allclose(full["sigma"], trunc["sigma"], rtol=0, atol=1e-12)
    a = full["states"][:2000]
    b = trunc["states"][:2000]
    assert a.shape == b.shape
    # At most a handful of pre-truncation labels move; the point is that the
    # number is not zero for the whole-sample fit on every seed.
    moved = int((a != b).sum())
    assert moved >= 0
    if moved == 0:  # pragma: no cover - seed 5 is the documented flipper
        for seed in (6, 7):
            r2, _, _ = _synth(seed=seed)
            f2, t2 = hmm_two_state(r2), hmm_two_state(r2[:2000])
            moved += int((f2["states"][:2000] != t2["states"][:2000]).sum())
    assert moved > 0, "the whole-sample fit is causal on every seed — re-check the claim"


def test_causal_hmm_labels_do_not_move_when_future_bars_are_appended():
    """The fix: appending bars cannot change any earlier label."""
    from core.strategy.regime import hmm_two_state_causal

    r, _, _ = _synth(seed=5)
    full = hmm_two_state_causal(r)
    t = 2000
    trunc = hmm_two_state_causal(r[:t])
    assert full["causal"] is True and trunc["causal"] is True

    def labels(fit):
        return fit["state"].astype(str)

    all_labels = labels(full)
    # The truncated run may not have emitted a label for the very last bar (its
    # segment boundary lands there), so compare every bar it *did* label and
    # require its coverage to be all but that boundary.
    got = labels(trunc)
    common = got.index.intersection(all_labels.index)
    assert (got.loc[common].to_numpy() == all_labels.loc[common].to_numpy()).all()
    assert len(got) >= t - 1
    covered = (got != "unknown").sum()
    assert covered >= t - full["warmup"] - 1


def test_causal_hmm_reports_its_warmup_and_is_deterministic():
    """A warm-up is a documented knob, not a hidden one."""
    from core.strategy.regime import (HMM_CAUSAL_WARMUP, hmm_two_state_causal)

    r, _, _ = _synth(seed=6)
    a = hmm_two_state_causal(r)
    b = hmm_two_state_causal(r)
    assert a["warmup"] == HMM_CAUSAL_WARMUP
    assert a["first_label_index"] == HMM_CAUSAL_WARMUP
    assert (a["state"].astype(str).iloc[:HMM_CAUSAL_WARMUP] == "unknown").all()
    assert np.array_equal(a["states"], b["states"])
    assert (a["state"].astype(str).to_numpy() == b["state"].astype(str).to_numpy()).all()
    assert a["refit_every"] > 0 and a["n_refits"] > 1


def test_causal_hmm_out_of_sample_accuracy_is_reported_not_assumed():
    """Accuracy on the synthetic regimes is **out of sample**, and modest.

    The shipped whole-sample fit reports 0.9987 accuracy on the same series, but
    that is *in-sample* (its parameters saw the regime it is being scored on).
    The causal path scores honestly lower and cannot be claimed as 99.9 %; this
    test pins the gap so a future "improvement" cannot re-introduce the
    in-sample number as if it were tradeable.
    """
    from core.strategy.regime import (detection_metrics, hmm_two_state,
                                      hmm_two_state_causal)

    r, truth, cp = _synth(seed=5)
    truth = truth[1:]                       # the HMM consumes log returns
    fit = hmm_two_state_causal(r)
    pred = np.where(fit["states"] == 1, "high", "low")
    oos = detection_metrics(truth, pred, positive="high",
                            change_points=(cp[0] - 1, cp[1] - 1))
    assert 0.3 < oos["accuracy"] < 0.95
    assert len(oos["latency_bars"]) == 2
    ins = detection_metrics(
        truth, np.where(hmm_two_state(r)["states"] == 1, "high", "low"),
        positive="high", change_points=(cp[0] - 1, cp[1] - 1))
    assert ins["accuracy"] > 0.95
    assert oos["accuracy"] < ins["accuracy"] - 0.2


def test_regime_gate_refuses_non_causal_hmm_labels():
    """A gate cannot silently consume a look-ahead label."""
    from core.strategy.regime import (NonCausalRegimeError, classify_regimes,
                                      default_gate, gate_regimes)

    r, _, _ = _synth(seed=5)
    close = 100.0 * np.exp(np.cumsum(r))
    idx = pd.date_range("2025-01-01", periods=len(r), freq="h")
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": 1.0}, index=idx)

    bad = classify_regimes(df, with_hmm=True, causal_hmm=False)
    good = classify_regimes(df, with_hmm=True, causal_hmm=True)
    assert bad.attrs["causal_hmm"] is False and bad.attrs["hmm_present"] is True
    assert good.attrs["causal_hmm"] is True

    gate = default_gate(enabled=True)
    with pytest.raises(NonCausalRegimeError):
        gate_regimes(gate, bad, "breakout")
    res = gate_regimes(gate, good, "breakout")
    assert res["allowed"] in (True, False) and res["regime"]

    # With gating off the gate cannot change behaviour, so it does not refuse.
    assert gate_regimes(default_gate(enabled=False), bad, "breakout")["allowed"] is True
    # An explicit opt-out stays available for a research caller.
    from core.strategy.regime import RegimeGate
    loose = RegimeGate(allowed={}, enabled=True, require_causal=False)
    assert gate_regimes(loose, bad, "breakout")["allowed"] is True


def test_hmm_default_mode_follows_the_gating_switch():
    """Gating off keeps the historical fit bit-for-bit; gating on makes it causal."""
    import core.strategy.regime as regime

    r, _, _ = _synth(seed=5)
    assert regime.REGIME_GATING_ENABLED is False
    assert regime.hmm_two_state(r)["causal"] is False
    saved = regime.REGIME_GATING_ENABLED
    regime.REGIME_GATING_ENABLED = True
    try:
        fit = regime.hmm_two_state(r)
        assert fit["causal"] is True
    finally:
        regime.REGIME_GATING_ENABLED = saved


# ── 2. the microstructure cache must key on the decision time ────────────

class _StubClient:
    """Counts fetches so a cache hit is observable."""

    def __init__(self):
        self.book_calls = 0
        self.trade_calls = 0

    async def order_book(self, symbol, limit=20):
        self.book_calls += 1
        return {"bids": [["100.0", "1.0"]], "asks": [["100.1", "1.0"]]}

    async def recent_trades(self, symbol, limit=100):
        self.trade_calls += 1
        return [{"price": "100.05", "qty": "1.0", "quoteQty": "100.05",
                 "time": 1_700_000_000_000, "isBuyerMaker": False}]


def test_microstructure_cache_distinguishes_point_in_time_requests():
    """``as_of_ms`` is part of the identity: a stale payload is a miss."""
    from core.market_data.microstructure import MicrostructureCache, fetch_features

    async def run():
        client = _StubClient()
        cache = MicrostructureCache(ttl_secs=60.0)
        t = 1_700_000_000_000.0
        first = await fetch_features(client, "BTCUSDT", cache=cache, as_of_ms=t)
        assert first is not None and first["as_of_ms"] == t
        calls = client.book_calls
        # Same decision time -> hit, no new fetch.
        again = await fetch_features(client, "BTCUSDT", cache=cache, as_of_ms=t)
        assert client.book_calls == calls
        assert again["as_of_ms"] == t
        # Earlier decision time (t - 1h) -> must NOT be answered by the t payload.
        stale = await fetch_features(client, "BTCUSDT", cache=cache,
                                     as_of_ms=t - 3_600_000.0)
        assert client.book_calls == calls + 1, "the t payload was served for t-1h"
        assert stale is not None
        assert stale["as_of_ms"] == t - 3_600_000.0
        assert stale["as_of_ms"] != first["as_of_ms"]
        # Both entries coexist (the key is not just the symbol).
        assert len(cache) == 2
        # The live (as_of_ms=None) key is separate from both stamped keys.
        live = await fetch_features(client, "BTCUSDT", cache=cache)
        assert live is not None
        assert client.book_calls == calls + 2
        assert len(cache) == 3
        return True

    assert asyncio.run(run()) is True


def test_microstructure_cache_bypass_is_explicit():
    """``use_cache=False`` guarantees a fresh snapshot and writes nothing."""
    from core.market_data.microstructure import MicrostructureCache, fetch_features

    async def run():
        client = _StubClient()
        cache = MicrostructureCache(ttl_secs=60.0)
        a = await fetch_features(client, "BTCUSDT", cache=cache, as_of_ms=1000.0)
        b = await fetch_features(client, "BTCUSDT", cache=cache, as_of_ms=1000.0,
                                 use_cache=False)
        assert a is not None and b is not None
        assert a["as_of_ms"] == b["as_of_ms"] == 1000.0
        # The bypassed call did fetch (2 fetches: a + b) and cached nothing new.
        assert client.book_calls == 2
        assert len(cache) == 1
        return True

    assert asyncio.run(run()) is True


# ── 3. the GARCH rationale / the fit ─────────────────────────────────────

def _synthetic_garch(n: int = 3000, *, seed: int = SEED,
                     params=(0.10, 0.12, 0.80)):
    """A stationary GARCH(1,1) series (omega in percent², returned as fractions)."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = params
    x = np.zeros(n)
    v = np.zeros(n)
    v[0] = omega / (1.0 - alpha - beta)
    for t in range(1, n):
        v[t] = omega + alpha * x[t - 1] ** 2 + beta * v[t - 1]
        x[t] = np.sqrt(v[t]) * rng.standard_normal()
    return x / 100.0


def test_free_omega_mle_converges_and_beats_the_igarch_corner():
    """The audit's refutation, re-measured on a controlled series.

    The shipped claim was that the free-ω MLE is unbounded and that
    "Nelder-Mead, L-BFGS-B and SLSQP … all converged to that corner and rejected
    the generating parameters of a synthetic GARCH(1,1)".  On a synthetic
    GARCH(1,1) with a splice-scale outlier injected (the data-quality problem the
    shipped doc is about) none of that reproduces: all three optimisers converge
    to the *generating* region and agree with each other, while the shipped
    IGARCH grid lands on the ``α=1, β=0`` corner whose mean objective is a large
    positive number — the corner, not the MLE, is the ill-posed fit.
    """
    from scipy.optimize import minimize

    from core.ml.volatility import (_garch11_avg_ll, _garch11_best_beta,
                                    clip_outliers, garch11_loglik_grad,
                                    garch11_params)

    base = _synthetic_garch(1500)
    splice = 0.25
    r = np.concatenate([base, np.array([splice]), base])   # inject the splice
    assert r[1500] == splice, "the injected bar is not where the test says"
    assert np.abs(r).max() > 0.20
    # ``r[-500:]`` is the *tail* of the second copy of ``base``: the seam sits at
    # index 1500, so the tail window physically cannot contain it (max |r| there
    # is the ~0.039 of the synthetic series).  The old test asserted the tail —
    # which is why it passed for the wrong reason and could never check the
    # splice.  Fit the 500-bar window that really brackets the injected bar.
    assert np.abs(r[-500:]).max() < 0.20, "the tail must not carry the splice"
    rr = r[1251:1751]
    assert np.abs(rr).max() > 0.20                       # the splice is in window
    p = garch11_params(rr, window=0)
    assert p["ok"] is True and p["fitted"] is True
    assert p["omega"] > 0.0
    assert 0.02 < p["alpha"] < 0.4 and 0.5 < p["beta"] < 0.95
    assert 0.5 < p["alpha"] + p["beta"] < 1.0

    # All three optimisers on the free-ω likelihood agree with each other.
    x = clip_outliers(rr, sigma=8.0) * 100.0
    x2 = x * x
    var_s = float(np.var(x, ddof=1))

    def nll(t):
        return garch11_loglik_grad(x2, var_s, t)[0]

    fits = {}
    for method in ("Nelder-Mead", "L-BFGS-B", "SLSQP"):
        res = minimize(nll, np.array([var_s * 0.05, 0.06, 0.93]), method=method,
                       bounds=[(1e-12, 10 * var_s), (0.0, 0.999), (0.0, 0.999)])
        fits[method] = (float(res.fun), float(res.x[1]), float(res.x[2]))
    best = min(v[0] for v in fits.values())
    for method, (ll, a, b) in fits.items():
        assert abs(ll - best) < 1.0, (method, ll, best)
        assert 0.02 < a < 0.4 and 0.5 < b < 0.95, (method, a, b)

    # The shipped grid lands on the corner, with a far worse objective.
    best_beta, mean_ll = _garch11_best_beta(x2, var_s)
    assert best_beta == pytest.approx(0.0, abs=1e-9)
    assert mean_ll > 1.0
    assert _garch11_avg_ll(x2, var_s, p["alpha"], p["beta"]) < mean_ll - 1.0


def test_garch_fit_is_scale_invariant():
    """Fitting ``x`` and ``100·x`` must give the same model (the box is relative).

    The intercept box is expressed in the data's units, so an un-normalised fit
    silently changed basin between unit systems — the percent² fit picked a
    1e4-larger intercept with a much worse objective.
    """
    from core.ml.volatility import garch11_params

    rr = _synthetic_garch(800)[-400:]
    a = garch11_params(rr, window=0)
    b = garch11_params(rr * 100.0, window=0)
    assert a["fitted"] and b["fitted"]
    assert b["alpha"] == pytest.approx(a["alpha"], abs=3e-3)
    assert b["beta"] == pytest.approx(a["beta"], abs=3e-3)
    assert b["omega"] == pytest.approx(a["omega"] * 1e4, rel=0.10)


def test_garch_recovers_synthetic_parameters():
    """A properly fitted GARCH(1,1) recovers its own generating parameters."""
    from core.ml.volatility import garch11_params

    omega, alpha, beta = 0.10, 0.12, 0.80
    p = garch11_params(_synthetic_garch(3000), window=0)
    assert p["fitted"] is True
    assert p["omega"] * 1e4 == pytest.approx(omega, rel=0.35)
    assert p["alpha"] == pytest.approx(alpha, abs=0.06)
    assert p["beta"] == pytest.approx(beta, abs=0.06)


def test_garch_forecast_is_not_degenerate_and_uses_the_fit_s_own_units():
    """The forecast must be near the EWMA level, not a corner artefact.

    Two unit/data bugs are pinned here: a fraction² intercept added to a percent²
    recursion inflated the forecast by ~1e4, and filtering the *unclipped* series
    with parameters fitted on the *clipped* one inflated it ~9.6× (the splice bar
    is ~1e4 in percent² against a conditional level near 0.2).  The fixed
    forecast is within a factor of ~2 of EWMA on every window.
    """
    from core.ml.volatility import (ewma_vol, garch11_forecast, log_returns,
                                    _last)

    r = log_returns(_btc()["close"].values)
    rr = _last(r, 500)
    ewma = ewma_vol(rr, window=0)
    g = garch11_forecast(rr, window=0)
    assert g > 0.0
    assert 0.5 < g / ewma < 2.0, (g, ewma)
    # Windows around the splice (and away from it) must all be sane.
    for win in (300, 1000, 2000):
        e = ewma_vol(_last(r, win), window=0)
        gg = garch11_forecast(_last(r, win), window=0)
        assert 0.1 < gg / e < 5.0, (win, gg, e)


def test_garch_params_documents_the_igarch_zero_over_zero():
    """``omega/(1-alpha-beta)`` is 0/0 for the unit-persistence fallback."""
    from core.ml.volatility import garch11_params, log_returns

    p = garch11_params(log_returns(_btc()["close"].values)[:10])
    assert p["ok"] is False and p["fitted"] is False
    # The docstring must not promise a long-run variance for that branch.
    doc = garch11_params.__doc__ or ""
    assert "0/0" in doc or "not defined" in doc


# ── 4. the Engle-Granger null must be selectable to match its regression ──

def test_matched_tau_null_uses_the_tests_own_lag_rule():
    """A lag-augmented null is available and differs from the baseline."""
    from core.strategy.pairs import tau_critical_values, tau_null_distribution

    base = tau_critical_values("eg_c", 500, max_lags=0)
    matched = tau_critical_values("eg_c", 500, max_lags=1)
    assert set(base) == set(matched) == {0.01, 0.05, 0.10}
    for lv in base:
        assert matched[lv] < base[lv]          # augmentation pushes the tail left
    assert matched[0.05] == pytest.approx(base[0.05], abs=0.05)
    # The cache key includes the lag rule (two different distributions).
    a = tau_null_distribution("eg_c", 500, max_lags=0, reps=100)
    b = tau_null_distribution("eg_c", 500, max_lags=1, reps=100)
    assert not np.array_equal(a, b)
    assert a.shape == b.shape


def test_engle_granger_reports_which_null_it_used():
    """The approximation is stated in the return value, not implied."""
    from core.strategy.pairs import engle_granger

    rng = np.random.default_rng(SEED)
    y = pd.Series(np.cumsum(rng.standard_normal(600)))
    x = pd.Series(np.cumsum(rng.standard_normal(600)))
    plain = engle_granger(y, x)
    assert plain["null_matched"] is False and plain["null_lags"] == 0
    matched = engle_granger(y, x, matched_null=True)
    assert matched["null_matched"] is True
    assert matched["null_lags"] == int(matched["adf_lag"])
    assert matched["adf_lag"] >= 0
    # The statistic itself is identical; only its reference distribution moves.
    assert matched["adf_tau"] == pytest.approx(plain["adf_tau"], rel=1e-12)
    doc = engle_granger.__doc__ or ""
    assert "approximation" in doc and "matched_null" in doc


# ── 5. inert config keys (reported, not fixed here) ──────────────────────

def test_barrier_widths_hook_has_no_production_caller():
    """Documents defect 5 as measured: the three ``barrier_*`` keys are inert.

    ``PositionSizer.barrier_widths_pct`` is the only reader of
    ``risk.vol_targeting.barrier_*``; the live label path builds its widths from
    ``ml.barrier_min/max_pct`` (``MLPredictor._barrier_params``) and never calls
    it.  This test is a tripwire: if the wiring lands, it fails and the config
    comment must stop saying "inert".
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    callers = []
    for path in (root / "core").rglob("*.py"):
        if path.name == "position_sizer.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"barrier_widths_pct\s*\(", text):
            callers.append(path.relative_to(root).as_posix())
    assert callers == [], f"barrier_widths_pct gained a caller: {callers}"


# ── 6. the price-slice cache must survive more than one bar ──────────────

def test_price_slice_cache_is_hoisted_in_the_engine_source():
    """The cache is allocated outside the per-timestamp loop and cleared per bar.

    A behavioural test would need a full GA backtest; the defect was a failing
    *claim* about where the allocation lives, so the source is inspected directly
    plus the keying rule is asserted (it must include the timestamp, or a hit
    would return another bar's slice).
    """
    import inspect

    from core.backtest.engine import BacktestEngine

    src = inspect.getsource(BacktestEngine)
    loop = src.index("for slice_data in feeder:")
    alloc = src.index("_price_slice_cache: dict[tuple, pd.DataFrame] = {}")
    assert alloc < loop, "the cache is still allocated inside the bar loop"
    assert "_slice_key = (ts, sym, pos_tf)" in src
    # The frame is cut positionally (bit-identical to the boolean mask).
    assert "searchsorted(ts, side=\"right\")" in src
    # And released once the bar is done, so it cannot grow per bar.
    assert "_price_slice_cache.clear()" in src


def test_positional_slice_is_identical_to_the_boolean_mask():
    """`searchsorted`+`iloc` must equal `df[df.index <= ts]` on the real feed."""
    if not Path("data/market/BTCUSDT/1h.parquet").exists():
        pytest.skip("no cached parquet")
    from core.backtest.data_feeder import DataFeeder

    feeder = DataFeeder("data/market", ["BTCUSDT"], ["1h"], "2026-01-01",
                        "2026-06-01")
    feeder.load()
    if not len(feeder):
        pytest.skip("no rows in the requested window")
    df = feeder.get_all_data_for_symbol("BTCUSDT", "1h")
    for ts in list(feeder._timestamps)[::53]:
        mask = df[df.index <= ts]
        cut = int(df.index.searchsorted(ts, side="right"))
        assert mask.equals(df.iloc[:cut])
        if len(mask):
            assert float(mask.iloc[-1]["close"]) == float(df.iloc[:cut].iloc[-1]["close"])


# ── 7. the clip decision must be stable per observation ──────────────────

def test_clip_decision_is_stable_when_the_window_slides():
    """An anchored clip never re-values an observation as the window rolls.

    The old per-window MAD recomputed the Winsor limit from every window it was
    handed, so the same bar could be clipped in one window, relaxed in the next
    and re-clipped later — an observation-dependent quantity that is a function
    of *when* it was asked.  With one :func:`build_anchor` per series the clipped
    value of a bar is unchanged by any window that contains it.
    """
    from core.ml.volatility import (build_anchor, clip_outliers)

    rng = np.random.default_rng(SEED)
    r = rng.normal(0.0, 0.002, 600)
    r[120] = 0.15                       # splice-scale outlier
    r[300] = -0.12
    anchor = build_anchor(r)

    # Track every observation while later windows are appended: with the anchor,
    # once seen it can never change value.
    seen = {}
    flips = 0
    for end in range(40, len(r) + 1):
        for window in (60, 100, 250):
            w = r[max(0, end - window):end]
            c = clip_outliers(w, sigma=6.0, anchor=anchor)
            for pos in range(len(w)):
                # Positional index into the window is not a stable identity, so
                # track by the value's own absolute index.
                idx = max(0, end - window) + pos
                val = float(c[pos])
                prev = seen.get(idx)
                if prev is not None and abs(val - prev) > 1e-15:
                    flips += 1
                seen[idx] = val
    assert flips == 0, f"{flips} observations changed value across windows"

    # The *un-anchored* path is the defect: it does move.
    moved = 0
    base = None
    for end in range(160, len(r) + 1):
        w = r[max(0, end - 100):end]
        c = clip_outliers(w, sigma=6.0, anchored=False)
        if base is not None:
            for pos in range(len(w) - 1):
                if abs(float(c[pos]) - base[pos]) > 1e-15:
                    moved += 1
        base = [float(v) for v in c[1:]]
    assert moved > 0, "the per-window MAD is stable on this sample — re-check"

    # And the anchored clip still removes the outliers.
    clipped = clip_outliers(r, sigma=6.0, anchor=anchor)
    assert np.abs(clipped).max() < np.abs(r).max()
    assert np.abs(clipped).max() < 0.02


def test_anchored_clip_matches_the_documented_recipe():
    """Anchoring does not change *how tight* the clip is, only *when* it moves.

    Over the full series the anchor is the same ``median`` / ``1.4826·MAD``
    recipe, so the clipped array is identical to the un-anchored one — the fix
    removes the window dependence without re-tuning the threshold.
    """
    from core.ml.volatility import build_anchor, clip_outliers, log_returns

    r = log_returns(_btc()["close"].values)
    anchor = build_anchor(r)
    a = clip_outliers(r, sigma=6.0, anchor=anchor)
    b = clip_outliers(r, sigma=6.0, anchored=False)
    assert np.array_equal(a, b)
    assert np.abs(a).max() < np.abs(r).max()
    # Disabling the clip returns the input untouched (unchanged contract).
    assert np.array_equal(clip_outliers(r, sigma=0.0), r)
    # Too-short input is returned as-is, with a zero anchor.
    short = np.array([0.01, -0.01])
    assert clip_outliers(short, sigma=6.0, anchor=build_anchor(short)).size == 2


def test_estimators_accept_an_anchor_and_agree_across_windows():
    """With one anchor, ``window=500`` and the same 500 bars as a tail agree."""
    from core.ml.volatility import build_anchor, ewma_vol, log_returns

    r = log_returns(_btc()["close"].values)
    anchor = build_anchor(r)
    whole = ewma_vol(r, window=500, anchor=anchor)
    tail = ewma_vol(r[-500:], window=0, anchor=anchor)
    assert whole == pytest.approx(tail, rel=1e-12)
    assert whole > 0.0
    # Without the anchor the two calls need not agree (that is the defect).
    assert ewma_vol(r, window=500) > 0.0


# ── 8. the forecast caches must be bounded ───────────────────────────────

def test_vol_forecaster_cache_is_bounded():
    """The (symbol, interval) memo cannot grow without limit."""
    from core.ml.volatility import MAX_FORECAST_CACHE_ENTRIES, VolForecaster

    df = _btc(120)
    f = VolForecaster(interval="1h")
    for i in range(MAX_FORECAST_CACHE_ENTRIES * 4):
        f.forecast((f"SYM{i:04d}", "1h"), df, interval="1h")
    assert len(f._cache) == MAX_FORECAST_CACHE_ENTRIES
    # A small explicit bound is honoured too, and a re-forecast still hits.
    g = VolForecaster(interval="1h", max_cache_entries=4)
    for i in range(20):
        g.forecast((f"S{i}", "1h"), df, interval="1h")
    assert len(g._cache) == 4
    before = g.compute_count
    g.forecast(("S19", "1h"), df, interval="1h")
    assert g.compute_count == before       # still a hit after evictions


def test_executor_forecast_cache_size_is_reported_not_fixed():
    """Executor cache (out of this change's write scope) — record the boundary.

    ``OrderExecutor._forecast_vol_cache`` is bounded only by its 300 s TTL, and a
    symbol that stops being pushed is never re-read, so nothing expires it.  The
    keys are symbol names from the configured universe (bounded in practice), so
    this is a documented follow-up rather than a live leak.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "core/executor/executor.py").read_text(encoding="utf-8",
                                                          errors="ignore")
    assert "_forecast_vol_cache" in text
    assert not re.search(r"_forecast_vol_cache[^\n]*MAX_", text), (
        "the executor cache gained a bound — update this test and defect 8")


# ── 9. documentation numbers ─────────────────────────────────────────────

def test_doc_ten_numbers_are_reproducible_or_labelled():
    """Doc 10's headline numbers must be reproducible *or* labelled by window.

    The cache in ``data/market`` grows hourly (and gained a repaired splice
    during this audit), so a hard-coded decimal in the doc cannot be asserted
    against live data.  What *can* be asserted is the contract the audit asked
    for: the refuted 10.5× appears **only inside the correction that refutes it**,
    the splice ratio is quoted as ~9.8×, every headline number carries the
    window/sample it was measured on, and the measured ratio on the current cache
    is consistent with the quoted one whenever the splice is still present.
    """
    from core.ml.volatility import DEFAULT_WINDOW, ewma_vol, log_returns, to_pct

    doc = (Path(__file__).resolve().parents[1]
           / "docs/core-algorithms/10-volatility-targeting.md").read_text(
               encoding="utf-8")
    assert DEFAULT_WINDOW == 500
    r = log_returns(_btc()["close"].values)
    # The refuted ratio must be labelled as refuted, with the corrected one given.
    assert "9.8" in doc or "9.814" in doc
    assert "10.5" not in doc or "早期文档" in doc
    assert "9.8" in doc
    # The window dependence is stated, with the windows named.
    assert "window = 500" in doc and "window = 400" in doc
    # The sample the numbers came from is stated (row count or date range).
    assert "8 846" in doc or "8846" in doc
    # On the current cache: if the splice is still in the measured window the
    # quoted ratio must hold there too; otherwise the doc must say so.
    rr = r[-DEFAULT_WINDOW:]
    clip_on = ewma_vol(rr, window=0)
    clip_off = ewma_vol(rr, window=0, outlier_sigma=0.0)
    assert clip_on > 0.0
    if np.abs(rr).max() > 0.1:                       # splice still in the window
        assert clip_off / clip_on == pytest.approx(9.8, abs=1.0)
    else:
        assert "接缝" in doc or "splice" in doc.lower()
        # The doc's "clip is inert" row is labelled with *its own* snapshot
        # (11 675 rows @ 12:35, "1.000x, 相对差 2.7e-6"), and the live file gains
        # a bar every hour: one appended bar can exceed this window's ±6·MAD
        # limit without any splice (measured 0.5188 vs 0.5377 %/bar, 1.0365x), so
        # pinning the snapshot equality is flaky by construction.  Recompute the
        # *contract* from the frame just read instead — the clip may only replace
        # bars above its own anchored limit, which keeps a splice-free window an
        # order of magnitude below the splice regime (the doc's two labelled
        # states are 1.000x and 9.49x) — and check the estimator really used the
        # window's own anchor.
        from core.ml.volatility import (DEFAULT_OUTLIER_SIGMA, clip_outliers,
                                        series_anchor)

        assert clip_off / clip_on < 5.0
        anchor = series_anchor(rr)
        assert clip_on == pytest.approx(
            ewma_vol(clip_outliers(rr, sigma=DEFAULT_OUTLIER_SIGMA, anchor=anchor),
                     window=0, outlier_sigma=0.0), rel=1e-12)
    assert to_pct(clip_on) > 0.0


def test_doc_ten_garch_section_states_the_refitted_path():
    """Doc 10 §3.3 must describe the fitted GARCH, and quote it as a *refutation*.

    The section may still contain the words "unbounded" — but only inside the
    correction that says the claim was wrong.  The concrete check is that the
    old conclusion markers are gone and the corrected facts are present.
    """
    doc = (Path(__file__).resolve().parents[1]
           / "docs/core-algorithms/10-volatility-targeting.md").read_text(
               encoding="utf-8")
    assert "自由 ω 的 MLE" in doc                    # the primary path is named
    # The wrong figures may appear only inside the correction.
    assert "1.8e4" not in doc or "求和" in doc
    assert "0/0" in doc
    assert "IGARCH" in doc                          # the fallback is documented
    assert "fitted" in doc or "回退" in doc
    # The refuted claim must appear as refuted, not as a standing reason.
    assert "审计" in doc and ("纠正" in doc or "推翻" in doc or "错" in doc)


def test_doc_eleven_states_the_unmatched_null_and_the_measured_size():
    """Doc 11 must state the un-augmented approximation and the measured size."""
    doc = (Path(__file__).resolve().parents[1]
           / "docs/core-algorithms/11-pairs-cointegration.md").read_text(
               encoding="utf-8")
    assert "matched_null" in doc
    assert "2 / 20" in doc or "2/20" in doc
    assert "6/200" in doc or "6 / 200" in doc
    # The regime table's accuracy must be labelled in-sample, and the causal
    # out-of-sample number must be given next to it.
    assert "in-sample" in doc.lower() or "样本内" in doc
    assert "样本外" in doc or "out-of-sample" in doc.lower()
    # The meta-labelling table caveat lives in this page's appendix by design.
    assert "adx_ema" in doc or "元标签" in doc


def test_adf_size_experiment_reproduces():
    """The corrected size experiment: 200 walks → 0.035 at 5 %, 0.005 at 1 %."""
    from core.strategy.pairs import engle_granger

    rng = np.random.default_rng(12345)
    n = 300
    reject5 = reject1 = 0
    walks = 200
    for _ in range(walks):
        y = np.cumsum(rng.standard_normal(n))
        x = np.cumsum(rng.standard_normal(n))
        res = engle_granger(pd.Series(y), pd.Series(x))
        cv = res["critical_values"]
        if res["adf_tau"] <= cv[0.05]:
            reject5 += 1
        if res["adf_tau"] <= cv[0.01]:
            reject1 += 1
    assert reject5 / walks <= 0.08          # nominal 5 %, finite-sample slack
    assert reject1 / walks <= 0.03
    assert reject5 >= reject1


# ── 10. the causal decoder's buffer/lag bookkeeping (fixer: D1) ──────────

def test_causal_hmm_posterior_is_the_decoded_bar_not_a_lagged_one():
    """The emitted posterior must belong to the bar it is labelled for.

    The buffer starts at ``start = max(0, t - limit)`` while the segment starts at
    ``t``, so reading ``fwd[k]`` (buffer row ``k``) for the k-th row of the
    *segment* labelled bar ``tt`` with the posterior of bar
    ``tt - (t - start)``.  The offset was 0 only while the buffer was still
    filling (bar < 250, then < 500, …) and reached 2500 bars afterwards, which is
    why the decode measured 0.156 instead of 0.758.
    """
    from core.strategy.regime import (HMM_CAUSAL_WARMUP, detection_metrics,
                                      hmm_two_state_causal)

    r, truth, cp = _synth(seed=11)
    fit = hmm_two_state_causal(r)
    # The lag is reported, not implied: 0 means "the label is this bar's".
    assert fit["lag_bars"] == 0
    filtered = fit["posterior_filtered"]
    # The row kept in ``posterior_filtered[tt]`` must be the argmax of the state
    # that (after the same volatility ordering) produced ``states[tt]``.
    implied = np.where(filtered[:, 1] > filtered[:, 0], 1, 0)
    assert np.array_equal(implied, fit["states"])
    assert np.allclose(filtered.sum(axis=1), 1.0)
    # Same construction via the public metric: the labels track the true regime.
    pred = np.where(fit["states"] == 1, "high", "low")
    acc = detection_metrics(truth[1:], pred, positive="high",
                            change_points=(cp[0] - 1, cp[1] - 1))["accuracy"]
    assert acc > 0.65, f"causal decode collapsed to {acc:.3f} — likely re-lagged"
    assert fit["state"].astype(str).iloc[:HMM_CAUSAL_WARMUP].eq("unknown").all()
    # Short input still reports the field (it is part of the contract).
    small = hmm_two_state_causal(r[:10])
    assert small["lag_bars"] == 0 and small["first_label_index"] is None


def test_vol_forecaster_cache_keeps_recently_used_entries():
    """A bounded memo must evict the *least recently used* key, not the oldest.

    The insertion-order bound already kept the size at the cap (the defect's
    first half), but recency was not tracked at all, so a key read on every bar
    was still the first to be dropped once the cap was reached.  Here ``K0`` is
    read on *every* iteration while 12 one-off keys churn through a 4-entry
    cache: under a pure insertion-order eviction it is gone after the third
    insert, and only a recency-aware policy keeps it resident to the end.  (A
    touch on the way in, as in the previous version of this test, is not enough:
    a key read once and then never again is genuinely the least recently used.)
    """
    from core.ml.volatility import MAX_FORECAST_CACHE_ENTRIES, VolForecaster

    df = _btc(120)
    f = VolForecaster(interval="1h", max_cache_entries=4)
    f.forecast(("K0", "1h"), df, interval="1h")
    for i in range(12):
        f.forecast(("K0", "1h"), df, interval="1h")     # the live symbol
        f.forecast((f"S{i}", "1h"), df, interval="1h")  # one-off symbols
    assert len(f._cache) == 4
    assert ("K0", "1h") in f._cache, "the actively used key was evicted"
    # ...and the eviction really did happen to the one-off keys.
    assert ("S11", "1h") in f._cache and ("S0", "1h") not in f._cache
    # The default cap is still enforced under heavy churn.
    g = VolForecaster(interval="1h")
    for i in range(MAX_FORECAST_CACHE_ENTRIES * 8):
        g.forecast((f"S{i:04d}", "1h"), df, interval="1h")
    assert len(g._cache) == MAX_FORECAST_CACHE_ENTRIES
    # A hit on a resident key still returns the memoised object.
    key = next(iter(g._cache))
    assert g.forecast(key, df, interval="1h") is g._cache[key][2]


# ── 11. the per-bar budget must hold for the default path (fixer: D3) ────

def test_per_bar_budget_holds_for_the_default_path_and_garch_is_opt_in():
    """The documented per-bar budget is a *default-path* promise.

    Re-measured on the frozen revision, the budget is a promise about the shape
    the live path actually hands in — this test's ``_btc(600)`` frame — where the
    default ``ewma`` path costs **≈0.15–0.16 ms/call**, ~13× inside
    ``PER_BAR_BUDGET_SEC`` = 2 ms (``ewma_variance`` is a per-bar Python
    recursion, so the earlier ~0.2 ms figure belongs to an older, vectorised
    implementation).  The shape that *does* approach the budget is the
    whole-history one: all 11 676 returns of the cached frame at the default
    ``window=500`` cost **≈1.5 ms/call**, ~92 % of it the O(history) clip anchor
    ``series_anchor`` rather than ``ewma_variance`` — and the live risk path
    passes ≤600 bars, so that shape is not the budgeted one.  The realised family
    is 0.02–0.05 ms.  The free-ω GARCH MLE costs ≈0.12–0.14 s/call on a 500-bar
    window (≈3.7–3.9 s on the full 11 676-return history, ``window=0``) — two to
    three orders of magnitude over budget, so it cannot be the default and the
    docs must not quote the grid fallback's ~2.2 ms for it.  This test pins both
    halves: the default path meets the budget, and the expensive path is opt-in
    rather than silently reachable from the per-bar method list.  Timings use the
    minimum of a few batches, which is robust to a loaded machine (a single
    20-call batch of the whole-history shape was measured at 2.06 ms under load;
    the 600-bar frame stayed ≤0.62 ms even then).
    """
    import time

    from core.ml.volatility import (METHODS, PER_BAR_BUDGET_SEC, VolForecaster,
                                    forecast_vol)

    def per_call_seconds(fn, *, batches: int = 3, reps: int = 10) -> float:
        fn()                                       # warm up
        best = float("inf")
        for _ in range(batches):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            best = min(best, (time.perf_counter() - t0) / reps)
        return best

    df = _btc(600)
    budget_ms = PER_BAR_BUDGET_SEC * 1e3
    # Default method only — this is what the live per-bar path runs.
    per_call = per_call_seconds(lambda: forecast_vol(df), reps=20)
    assert per_call < PER_BAR_BUDGET_SEC, (
        f"default path costs {per_call * 1e3:.2f} ms, above the {budget_ms:.1f} ms budget")
    # garch11 is a declared method but must not be the default, and the live
    # memoising wrapper must not select it unless the caller opted in.
    assert METHODS[0] == "ewma" and "garch11" in METHODS
    f = VolForecaster(interval="1h")
    assert f.method == "ewma" and f.allow_garch is False
    # The default path must also meet the budget for the realised family, so the
    # "budget" is a property of everything the live path can select.
    for method in ("ewma", "realized_cc", "realized_parkinson",
                   "realized_garman_klass"):
        cost = per_call_seconds(lambda m=method: forecast_vol(df, method=m))
        assert cost < PER_BAR_BUDGET_SEC, (method, cost)
    # The expensive path is opt-in.  Its cost is data-dependent (measured
    # 0.12–0.14 s on 500-bar windows: the Nelder-Mead polish runs ≈320
    # likelihood passes on the shipped window — the older "~575" is ~1.8× high —
    # and 264–343 passes on synthetic 500-bar windows over seeds
    # 5/7/11/20250930), so the assertion is a cost *ordering* against the default
    # path, not a constant.
    if BTC_1H.exists():
        from core.ml.volatility import garch11_forecast, log_returns
        rr = log_returns(df["close"].values)[-500:]
        t0 = time.perf_counter()
        g = garch11_forecast(rr, window=0)
        elapsed = time.perf_counter() - t0
        assert g > 0.0
        assert elapsed < 5.0, f"garch11 fit took {elapsed:.2f}s — check the backend"
        assert elapsed > 10 * per_call, (
            "garch11 is no longer materially more expensive than the default "
            "path — re-measure the documented cost before relaxing the opt-in")
