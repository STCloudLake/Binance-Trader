"""Read-only measurement harness for the P3/P4 audit defects (temporary file)."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from core.ml import volatility as V  # noqa: E402
from core.strategy import regime as R  # noqa: E402

BTC = "data/market/BTCUSDT/1h.parquet"
ETH = "data/market/ETHUSDT/1h.parquet"


def load_returns(path=BTC):
    df = pd.read_parquet(path)
    return df, V.log_returns(df["close"].values)


def section(name):
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)


# ── D3: GARCH re-fit ────────────────────────────────────────────────────
def free_omega_mle(x2, var_s, method="Nelder-Mead", x0=None):
    from scipy.optimize import minimize
    def obj(theta):
        ll, _ = V.garch11_loglik_grad(x2, var_s, theta)
        return ll
    if x0 is None:
        x0 = np.array([var_s * 0.1, 0.08, 0.9])
    res = minimize(obj, x0, method=method,
                   bounds=[(1e-12, 10 * var_s), (0.0, 0.999), (0.0, 0.999)]
                   if method != "Nelder-Mead" else None,
                   options={"maxiter": 2000})
    return res


def garch_measure():
    df, r = load_returns()
    d3 = {}
    for win in (500, 0):
        rr = V._last(r, win)
        x = V.clip_outliers(rr, sigma=8.0) * 100.0
        x2 = x * x
        var_s = float(np.var(x, ddof=1))
        row = {"n": int(x.size), "var_s": var_s, "sample_sd_pct": float(np.sqrt(var_s))}
        for meth in ("Nelder-Mead", "L-BFGS-B", "SLSQP"):
            try:
                res = free_omega_mle(x2, var_s, meth)
                w, a, b = [float(v) for v in res.x]
                ll, _ = V.garch11_loglik_grad(x2, var_s, res.x)
                row[meth] = {"omega": w, "alpha": a, "beta": b, "sum_ll_0p5": float(ll),
                             "mean_ll": float(ll) / x.size, "ok": bool(res.success)}
            except Exception as e:  # pragma: no cover
                row[meth] = {"error": f"{type(e).__name__}: {e}"}
        # shipped IGARCH grid
        p = V.garch11_params(rr, window=0)
        row["igarch_grid"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                              for k, v in p.items()}
        bb, bll = V._garch11_best_beta(x2, var_s)
        row["igarch_best_beta"] = {"beta": bb, "mean_ll": bll}
        # what a proper fit implies as a forecast
        row["params_now"] = V.garch11_params(rr, window=0)
        row["forecast_garch11"] = V.garch11_forecast(rr, window=0)
        row["forecast_ewma"] = V.ewma_vol(rr, window=0)
        row["forecast_realized_cc"] = V.realized_vol(rr, window=0)
        row["ewma_current_pct"] = V.to_pct(V.ewma_vol(rr, window=0))
        row["ewma_400_pct"] = V.to_pct(V.ewma_vol(rr, window=400))
        d3[f"win{win}"] = row
    section("D3 GARCH")
    print(json.dumps(d3, indent=2, default=str))
    return d3


# ── D1: HMM causality ───────────────────────────────────────────────────
def synth_regimes(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.002, n // 3)
    b = rng.normal(0.0, 0.010, n // 3)
    c = rng.normal(0.0, 0.002, n - 2 * (n // 3))
    return np.concatenate([a, b, c])


def hmm_causality():
    out = {}
    for seed in (5, 6, 7):
        x = synth_regimes(3000, seed)
        full = R.hmm_two_state(x)
        t = 2000
        trunc = R.hmm_two_state(x[:t])
        n = len(trunc["states"])
        same = int((full["states"][:n] == trunc["states"]).sum())
        out[f"seed{seed}"] = {
            "sigma_full": [float(v) for v in full["sigma"]],
            "sigma_trunc2000": [float(v) for v in trunc["sigma"]],
            "labels_equal_before_t": same, "n": n,
            "share_equal": same / n,
            "sigma_shift": [float(a - b) for a, b in zip(full["sigma"], trunc["sigma"])],
        }
    section("D1 HMM causality (labels before t unchanged?)")
    print(json.dumps(out, indent=2, default=str))
    return out


# ── D6: price slice cache ───────────────────────────────────────────────
def slice_cache_measure():
    section("D6 price slice cache")
    df = pd.read_parquet(BTC)
    n = len(df)
    for npos, nbar in ((5, 300), (20, 300)):
        t0 = time.perf_counter()
        ops = 0
        for i in range(nbar):
            inner_cache: dict = {}
            ts = df.index[min(i + 100, n - 1)]
            for _ in range(npos):
                key = ("BTCUSDT", "1h")
                sl = inner_cache.get(key)
                if sl is None:
                    sl = df[df.index <= ts]
                    inner_cache[key] = sl
                    ops += 1
        t1 = time.perf_counter()
        print(f"inside-loop cache: positions={npos} bars={nbar} "
              f"slice_ops={ops} wall={t1 - t0:.3f}s")
    for npos, nbar in ((5, 300), (20, 300)):
        t0 = time.perf_counter()
        ops = 0
        outer: dict = {}
        for i in range(nbar):
            ts = df.index[min(i + 100, n - 1)]
            for _ in range(npos):
                key = ("BTCUSDT", "1h", ts)
                sl = outer.get(key)
                if sl is None:
                    sl = df[df.index <= ts]
                    outer[key] = sl
                    ops += 1
        t1 = time.perf_counter()
        print(f"hoisted cache:      positions={npos} bars={nbar} "
              f"slice_ops={ops} wall={t1 - t0:.3f}s")


# ── D7: clipping monotonicity ───────────────────────────────────────────
def clip_measure():
    section("D7 retroactive clipping")
    rng = np.random.default_rng(11)
    r = rng.normal(0.0, 0.002, 300)
    r[120] = 0.15  # a splice-scale outlier
    # rolling window clip, as clip_outliers does per call
    jump = []
    for end in range(140, 300):
        w = r[max(0, end - 100):end]
        c = V.clip_outliers(w, sigma=6.0)
        jump.append(float(c[-1]))
    diffs = np.abs(np.diff(np.asarray(jump)))
    print("rolling-clip |Δ| max", float(diffs.max()), "count>1e-6",
          int((diffs > 1e-6).sum()))
    # does the outlier un-clip as the window moves?
    v = np.asarray(jump)
    print("value at end=140", v[0], "end=299", v[-1])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "d3"):
        garch_measure()
    if which in ("all", "d1"):
        hmm_causality()
    if which in ("all", "d6"):
        slice_cache_measure()
    if which in ("all", "d7"):
        clip_measure()
