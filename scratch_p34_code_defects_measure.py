"""Item 7 (clip stability) + item 5 measurement / before-after fingerprint.

Scratch tooling — not part of the test suite.  ``fingerprint`` prints a stable
list of the public estimator outputs so the pre/post-edit comparison is exact.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import core.ml.volatility as V

W = 500
SIGMA = V.DEFAULT_OUTLIER_SIGMA


def _anchor(r):
    return getattr(V, "series_anchor", V.build_anchor)(r)


def btc_returns():
    df = pd.read_parquet("data/market/BTCUSDT/1h.parquet")
    return V.log_returns(df["close"].values)


def synthetic(n=8844, seed=20250930, splice_at=None):
    """A calm series of the *audited* length with the audited +0.276 splice.

    The shipped cache no longer contains the calendar splice (max |log return|
    0.0494), so the audit's before-number can only be reproduced on a series of
    the same length carrying the same defect: ``n = 8 844`` returns → 8 345
    rolling 500-bar windows, exactly the audit's denominator.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.004, n)
    if splice_at is None:
        splice_at = int(n * 0.55)
    r[splice_at] = 0.276
    return r


def fingerprint():
    r = btc_returns()
    rng = np.random.default_rng(20250930)
    calm = rng.normal(0.0, 0.004, 2000)
    s = np.concatenate([calm[:1999], np.array([0.276])])
    df = pd.read_parquet("data/market/BTCUSDT/1h.parquet")
    out = {}
    out["doc_injected_clipped_pct"] = V.to_pct(V.ewma_vol(s, window=0))
    out["doc_injected_unclipped_pct"] = V.to_pct(
        V.ewma_vol(s, window=0, outlier_sigma=0.0))
    out["ewma(r,window=0)"] = V.ewma_vol(r, window=0)
    out["ewma(r,window=500)"] = V.ewma_vol(r, window=500)
    out["ewma(r,window=400)"] = V.ewma_vol(r, window=400)
    out["ewma(r[-500:],window=0)"] = V.ewma_vol(r[-500:], window=0)
    out["ewma_series(r,window=0)[-1]"] = float(
        V.ewma_vol_series(r, window=0).iloc[-1])
    out["ewma(calm1000,window=0)"] = V.ewma_vol(calm[:1000], window=0)
    out["forecast_vol(df,window=500)"] = V.forecast_vol(df, window=500)
    out["forecast_vol(df,window=400)"] = V.forecast_vol(df, window=400)
    out["forecast_vol(r,realized_cc)"] = V.forecast_vol(r, method="realized_cc")
    out["clip(r[-500:]).max"] = float(np.abs(V.clip_outliers(r[-500:])).max())
    out["clip(s).max"] = float(np.abs(V.clip_outliers(s)).max())
    out["vol_percentile(r)"] = V.vol_percentile(r)
    p = V.garch11_params(r[-500:], window=0)
    out["garch11_params"] = (round(p["omega"], 12), round(p["alpha"], 10),
                             round(p["beta"], 10), p["ok"], p["fitted"])
    out["garch11_forecast(r[-500:])"] = V.garch11_forecast(r[-500:], window=0)
    return out


def rolling_experiment(r, *, mode, window=W):
    """Slide ``window`` forward one bar at a time over the fixed series ``r``.

    ``mode`` selects the clip rule applied inside every window:

    * ``old``  — the pre-anchor per-window MAD (``anchored=False``), i.e. exactly
      the algorithm the audit measured;
    * ``window_ew`` — the shipped-P3 default before this fix (``_anchored_mad``
      recomputed on each window);
    * ``series_anchor`` — ONE anchor per series (``series_anchor(r)`` or
      ``build_anchor(r)``), which is what the production estimators now build.

    Two quantities are tracked, both "previously emitted":

    * the Winsor **limit** of the window (the audit's 8 344 / 8 345 count and its
      ``max |Δ| = 0.048`` are a *limit* movement);
    * the **clipped value of every observation**, keyed by its absolute index.
    """
    anchor = _anchor(r) if mode == "series_anchor" else None
    seen: dict[int, float] = {}
    changed_steps = changed_obs = 0
    limit_changed = 0
    max_delta = 0.0
    max_limit_delta = 0.0
    prev_limit = None
    for end in range(window, r.size + 1):
        start = end - window
        w = r[start:end]
        if mode == "old":
            med = float(np.median(w))
            scale = float(np.median(np.abs(w - med))) * 1.4826
            c = V.clip_outliers(w, sigma=SIGMA, anchored=False)
        elif mode == "window_ew":
            med, scale = V._anchored_mad(w, V.DEFAULT_MAD_HALF_LIFE)
            c = V.clip_outliers(w, sigma=SIGMA)
        else:
            med, scale = float(anchor.centre), float(anchor.scale)
            c = V.clip_outliers(w, sigma=SIGMA, anchor=anchor)
        limit = SIGMA * scale
        if prev_limit is not None:
            if limit != prev_limit:
                limit_changed += 1
                max_limit_delta = max(max_limit_delta, abs(limit - prev_limit))
        prev_limit = limit
        step = False
        for pos in range(c.size):
            idx = start + pos
            val = float(c[pos])
            prev = seen.get(idx)
            if prev is not None and val != prev:
                changed_obs += 1
                step = True
                max_delta = max(max_delta, abs(val - prev))
            seen[idx] = val
        changed_steps += int(step)
    steps = r.size - window + 1
    return dict(steps=steps, changed_steps=changed_steps,
                changed_obs=changed_obs, max_delta=max_delta,
                limit_changed=limit_changed, max_limit_delta=max_limit_delta)


def growing_experiment(r, *, window=W):
    """The live path: the series grows, the estimator window stays W.

    At each bar end the estimator is handed ``r[:end]``; the clipped value of
    every bar already emitted must not move.  This is the invariant the fix
    installs (prefix/append stability).
    """
    out = {}
    for mode in ("old", "window_ew", "default", "series_anchor"):
        seen: dict[int, float] = {}
        changed_steps = changed_obs = 0
        max_delta = 0.0
        for end in range(8, r.size + 1):
            prefix = r[:end]
            if mode == "old":
                c = V.clip_outliers(V._last(prefix, window), sigma=SIGMA,
                                    anchored=False)
                start = max(0, end - window)
            elif mode == "window_ew":
                c = V.clip_outliers(V._last(prefix, window), sigma=SIGMA)
                start = max(0, end - window)
            elif mode == "series_anchor":
                c = V.clip_outliers(prefix, sigma=SIGMA,
                                    anchor=_anchor(prefix))
                start = 0
            else:
                c = V.clip_outliers(prefix, sigma=SIGMA)
                start = 0
            step = False
            for pos in range(c.size):
                idx = start + pos
                val = float(c[pos])
                prev = seen.get(idx)
                if prev is not None and val != prev:
                    changed_obs += 1
                    step = True
                    max_delta = max(max_delta, abs(val - prev))
                seen[idx] = val
            changed_steps += int(step)
        out[mode] = (r.size - 8 + 1, changed_steps, changed_obs, max_delta)
    return out


def synthetic_regime(n=8844, seed=20250930):
    """The audited length with a regime-switching level (calm ↔ wild ↔ splice).

    The real BTC cache the audit measured carries volatility regimes, so its
    per-window MAD moved by ~5e-2 between windows; a constant-vol synthetic
    series cannot show that magnitude.  This one alternates 0.4 %/bar and
    2.0 %/bar blocks and carries the +0.276 splice, which is the smallest
    series that reproduces both the audit's count *and* its max |Δ|.
    """
    rng = np.random.default_rng(seed)
    sd = np.where((np.arange(n) // 700) % 2 == 0, 0.004, 0.020)
    r = rng.normal(0.0, sd)
    r[int(n * 0.55)] = 0.276
    return r


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    src = Path("core/ml/volatility.py").read_bytes()
    print(f"volatility.py sha256={hashlib.sha256(src).hexdigest()}")
    if which in ("all", "fingerprint"):
        for k, v in fingerprint().items():
            print(f"FP {k} = {v!r}")
    if which in ("all", "exp"):
        r = btc_returns()
        print(f"# shipped series n={r.size} max|r|={np.abs(r).max():.6f}")
        for mode in ("old", "window_ew", "series_anchor"):
            d = rolling_experiment(r, mode=mode)
            print(f"rolling[{mode:13s}] {d}")
        for label, series in (("synth", synthetic()),
                              ("regime", synthetic_regime())):
            print(f"# {label} spliced series n={series.size} "
                  f"max|r|={np.abs(series).max():.6f}")
            for mode in ("old", "window_ew", "series_anchor"):
                d = rolling_experiment(series, mode=mode)
                print(f"rolling-{label}[{mode:13s}] {d}")
