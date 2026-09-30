"""Statistical-arbitrage pairs trading: cointegration, Kalman hedge ratio, z-score signals.

Principle (Tsay, *Analysis of Financial Time Series*)
----------------------------------------------------
Single-asset returns are near-unpredictable, but **multivariate** structure is
one of the few robustly documented sources of predictability (Tsay ch. 8,
cointegration / error correction) and **state-space** models with time-varying
parameters are the other (Tsay ch. 7, Kalman filter).  If two log-prices are
cointegrated, the spread ``y − β·x`` is stationary: its deviations from its own
mean are transient and can be traded without forecasting either leg.  Because
the position is long one leg and short the other, the trade is largely immune to
the market factor — the P&L comes from the spread closing, not from direction.

Formulas
--------
Engle-Granger (1987) two-step, implemented here without statsmodels:

1. OLS hedge ratio on **levels** (logs, so β is an elasticity)::

       y_t = a + b·x_t + e_t

2. ADF on the residual ``e_t``::

       Δe_t = φ·e_{t-1} + Σ_{i=1..p} ψ_i·Δe_{t-i} + u_t     (no constant:
                                                              E[e] ≈ 0 by OLS)
       tau = φ̂ / se(φ̂)

   ``tau`` is compared with the null distribution of the *estimated-residual*
   ADF statistic (Engle-Granger / MacKinnon N=2), never with the plain DF
   table: the residual is estimated, which pushes the distribution left.  The
   null distribution is **simulated** here (see :func:`tau_null_distribution`)
   and validated against MacKinnon's published asymptotic values in the tests.

Kalman time-varying hedge ratio (Tsay ch. 7)::

    state:      θ_t = [β_t, α_t]ᵀ,      θ_t = θ_{t-1} + w_t,  w ~ N(0, Q)
    observation: y_t = [x_t, 1]·θ_t + v_t,                    v ~ N(0, R)
    Q = δ/(1 − δ)·I₂          (random-walk state, δ = KALMAN_DELTA)
    R = OLS residual variance  (the spread's own noise around the hedge line —
                                NOT var(y), which is 4.3× larger on BTC/ETH 1h
                                and makes the filter refuse to move: measured
                                β → 0.058 against an OLS β of 0.628)

    predict:  θ_{t|t−1} = θ_{t−1},  P_{t|t−1} = P_{t−1} + Q
              (θ₀ = OLS [β, α] warm start, P₀ = KALMAN_P0·I)
    update:   v_t = y_t − z_tᵀθ_{t|t−1}          (one-step prediction error)
              F_t = z_tᵀP_{t|t−1}z_t + R
              K_t = P_{t|t−1}z_t / F_t
              θ_t = θ_{t|t−1} + K_t·v_t,  P_t = (I − K_t z_tᵀ)P_{t|t−1}

   ``v_t`` is the **out-of-sample** spread innovation: β_t is estimated from
   information up to ``t−1`` only, so a signal on ``v_t`` cannot use the return
   it is about to trade.

Half-life (Ornstein-Uhlenbeck / AR(1) discretisation)::

    Δs_t = a + b·s_{t-1} + ε_t,   κ = −b  (mean-reversion speed, per bar)
    half_life = ln 2 / κ  bars,   equilibrium = −a/b

z-score rule (documented windows, all parameters module constants)::

    z_t = (s_t − mean(s_{t−L..t−1})) / std(s_{t−L..t−1})       L = lookback
    enter short spread when z ≥  Z_ENTRY   (short y, long β·x)
    enter long  spread when z ≤ −Z_ENTRY   (long y, short β·x)
    exit                 when |z| ≤ Z_EXIT
    stop                 when |z| ≥ Z_STOP (the relationship is breaking, not
                          stretched) and stand down until |z| < Z_ENTRY again

Lookback: ``L = clip(round(PAIRS_LOOKBACK_MULT × half_life), MIN_LOOKBACK,
MAX_LOOKBACK)`` — the plan's "half-life estimation to set the lookback".

Limitations (stated plainly)
----------------------------
* Engle-Granger finds **one** relationship; with k>2 assets use Johansen (not
  implemented) — the guard therefore only supports two-leg pairs.
* The ADF p-value is a **simulated** finite-sample value (``SIM_REPS`` paths,
  fixed seed): resolution ``1/reps`` and an accuracy of a few 1e-3, validated
  against MacKinnon's asymptotic table but not exact.
* Cointegration is a statement about the *past*: the OOS half of the sample is
  the only honest test, and it is reported separately by the research script.
* Costs dominate: a pairs round trip crosses **four** fills, so
  ``2 × round_trip_cost`` is charged (``leg_round_trip_cost_pct``).  A pair whose
  spread does not move ≥ ~4× cost per round trip cannot pay for itself.
* Nothing here touches the live path: :data:`PAIRS_ENABLED` is ``False`` and the
  engine only consumes a pairs signal if a caller explicitly wires one.

Note on scope: another agent owns ``core/ml/volatility.py`` (phase P3), which did
not exist when this module was written, so the (small) volatility helpers this
module needs are local and deliberately simple.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.ml.credibility import cost_pct_for, probabilistic_sharpe

# ── switches (all OFF: no live behaviour change) ─────────────────────────

#: Master switch for registering pairs strategies in the live path.  ``False``
#: is the only value any shipped code path sets; the engine seam checks it.
PAIRS_ENABLED = False
#: Minimum usable observations for a cointegration test.  Below this the ADF
#: regression has no power at all and the guard refuses, full stop.
PAIRS_MIN_OBS = 250
#: Engle-Granger p-value above which the pair is refused (5 % convention).
PAIRS_MAX_ADF_PVALUE = 0.05
#: Half-life bounds in bars; outside them the lookback cannot be set sensibly
#: (too fast = noise, too slow = the position is a directional bet).
PAIRS_MIN_HALF_LIFE = 2.0
PAIRS_MAX_HALF_LIFE = 120.0
#: Lookback = this multiple of the measured half-life, clipped.
PAIRS_LOOKBACK_MULT = 4.0
PAIRS_MIN_LOOKBACK = 30
PAIRS_MAX_LOOKBACK = 250
#: z-score thresholds.
PAIRS_Z_ENTRY = 2.0
PAIRS_Z_EXIT = 0.5
PAIRS_Z_STOP = 4.0
#: Kalman random-walk state variance factor / observation-noise scale.
KALMAN_DELTA = 1e-4
KALMAN_P0 = 1.0

# ── simulated null distribution of the ADF tau statistic ────────────────

#: Number of simulated paths (Monte-Carlo resolution of the p-value is 1/reps).
SIM_REPS = 4000
SIM_BLOCK = 250
SIM_SEED = 20240617
#: The simulated series length is clamped to this range: below 250 the
#: finite-sample distribution is very wide, above 2000 it equals the asymptote.
SIM_MIN_T = 250
SIM_MAX_T = 2000

_NULL_CACHE: dict[tuple[str, int], np.ndarray] = {}
_NULL_CACHE_MAX = 8


def _tau_block(kind: str, length: int, paths: int, seed: int) -> np.ndarray:
    """Simulated tau statistics of one Monte-Carlo block under the null."""
    rng = np.random.default_rng(seed)
    if kind == "df_c":
        # ADF on a single unit-root series with a constant.
        walk = np.cumsum(rng.standard_normal((paths, length)), axis=1)
        dy = np.diff(walk, axis=1)
        y1 = walk[:, :-1]
        xc = y1 - y1.mean(axis=1, keepdims=True)
        yc = dy - dy.mean(axis=1, keepdims=True)
        ss = (xc * xc).sum(axis=1)
        b = (xc * yc).sum(axis=1) / ss
        resid = yc - b[:, None] * xc
        s2 = (resid * resid).sum(axis=1) / (length - 3)
        return b / np.sqrt(s2 / ss)
    # "eg_c": Engle-Granger — two *independent* random walks (the null of no
    # cointegration), OLS on levels, then an ADF **without constant** on the
    # estimated residual (the residual has zero mean by construction; this is
    # what ``statsmodels.coint`` does and what MacKinnon's N=2 table indexes).
    y = np.cumsum(rng.standard_normal((paths, length)), axis=1)
    x = np.cumsum(rng.standard_normal((paths, length)), axis=1)
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    b = (xc * yc).sum(axis=1) / (xc * xc).sum(axis=1)
    resid = yc - b[:, None] * xc
    de = np.diff(resid, axis=1)
    r1 = resid[:, :-1]
    ss = (r1 * r1).sum(axis=1)
    b2 = (r1 * de).sum(axis=1) / ss
    r2 = de - b2[:, None] * r1
    s2 = (r2 * r2).sum(axis=1) / (length - 2)
    return b2 / np.sqrt(s2 / ss)


def tau_null_distribution(
    kind: str = "eg_c",
    n_obs: int = 500,
    *,
    reps: int = SIM_REPS,
    seed: int = SIM_SEED,
) -> np.ndarray:
    """Sorted Monte-Carlo draws of the ADF tau statistic under the null.

    ``kind = "eg_c"``
        Engle-Granger with two independent random walks (no cointegration) —
        the correct reference for a **tested** spread.
    ``kind = "df_c"``
        Plain ADF with a constant on a single unit-root series — used by the
        module's self-test against the Dickey-Fuller/MacKinnon table.

    Deterministic: a fixed seed and chunked simulation (``SIM_BLOCK`` paths at
    a time) give the same array on every machine.  Cached per
    ``(kind, simulated length)``, bounded to :data:`_NULL_CACHE_MAX` entries.
    """
    if kind not in ("eg_c", "df_c"):
        raise ValueError(f"unknown tau null kind: {kind!r}")
    length = int(min(max(int(n_obs), SIM_MIN_T), SIM_MAX_T))
    key = (kind, length)
    cached = _NULL_CACHE.get(key)
    if cached is not None:
        return cached
    parts = []
    for start in range(0, int(reps), SIM_BLOCK):
        m = min(SIM_BLOCK, int(reps) - start)
        parts.append(_tau_block(kind, length, m, int(seed) + start))
    taus = np.sort(np.concatenate(parts))
    if len(_NULL_CACHE) >= _NULL_CACHE_MAX:
        _NULL_CACHE.clear()
    _NULL_CACHE[key] = taus
    return taus


def admissible_sim_length(n_obs: int) -> int:
    """The simulated series length actually used for ``n_obs`` (clamped)."""
    return int(min(max(int(n_obs), SIM_MIN_T), SIM_MAX_T))


def tau_pvalue(tau: float, kind: str = "eg_c", n_obs: int = 500) -> float:
    """Left-tail Monte-Carlo p-value of ``tau`` under the simulated null.

    ``p = share of simulated taus ≤ tau``, floored/capped at one Monte-Carlo
    resolution step so the value is never exactly 0 or 1 (which would claim
    more precision than the simulation has).
    """
    taus = tau_null_distribution(kind, n_obs)
    n = len(taus)
    if not np.isfinite(tau):
        return 1.0
    p = float(np.searchsorted(taus, float(tau), side="left")) / n
    return float(min(max(p, 1.0 / n), 1.0 - 1.0 / n))


def tau_critical_values(
    kind: str = "eg_c",
    n_obs: int = 500,
    levels: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict[float, float]:
    """Simulated critical values of ``tau`` at the requested significance levels."""
    taus = tau_null_distribution(kind, n_obs)
    return {float(lv): float(np.quantile(taus, float(lv))) for lv in levels}


# ── ADF / Engle-Granger ─────────────────────────────────────────────────

def _schwert_max_lags(n: int) -> int:
    """Schwert (1989) rule of thumb for the ADF augmentation order, capped."""
    return int(min(8, max(0, math.floor(12.0 * (max(n, 1) / 100.0) ** 0.25))))


def adf_regression(
    series,
    *,
    max_lags: int | None = None,
    regression: str = "nc",
) -> dict:
    """ADF ``tau`` with AIC-selected augmentation lags, in plain numpy.

    ``regression = "nc"`` (no deterministic terms — the default, for an
    estimated cointegration residual) or ``"c"`` (with a constant).
    Lags are chosen by AIC over ``0..max_lags`` (Schwert rule, capped at 8 so
    the search stays bounded); ties go to the smaller lag.
    """
    s = np.asarray(pd.Series(series).astype(float).to_numpy(), dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 20:
        return {"tau": float("nan"), "lag": 0, "n_obs": n, "aic": float("nan"),
                "regression": regression, "sigma": float("nan")}
    if max_lags is None:
        max_lags = _schwert_max_lags(n)
    max_lags = int(min(max(max_lags, 0), max(0, n // 5 - 2)))

    best: dict | None = None
    for lag in range(0, max_lags + 1):
        dy = np.diff(s)[lag:]
        y1 = s[lag:-1]
        if len(dy) < 10:
            continue
        cols = [y1]
        if regression == "c":
            cols.append(np.ones_like(y1))
        for i in range(1, lag + 1):
            cols.append(np.diff(s)[lag - i:-i])
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
        resid = dy - X @ beta
        dof = len(dy) - X.shape[1]
        if dof <= 0:
            continue
        sigma2 = float(resid @ resid) / dof
        aic = len(dy) * math.log(max(sigma2, 1e-300)) + 2.0 * X.shape[1]
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = math.sqrt(max(sigma2 * float(xtx_inv[0, 0]), 0.0))
        tau = float(beta[0] / se) if se > 0 else float("nan")
        candidate = {
            "tau": tau, "lag": int(lag), "n_obs": int(len(dy)),
            "aic": float(aic), "regression": regression,
            "sigma": float(math.sqrt(max(sigma2, 0.0))),
            "phi": float(beta[0]),
        }
        if best is None or candidate["aic"] < best["aic"]:
            best = candidate
    if best is None:
        return {"tau": float("nan"), "lag": 0, "n_obs": n, "aic": float("nan"),
                "regression": regression, "sigma": float("nan")}
    return best


#: Probe for the optional statsmodels cross-check (plan: "use statsmodels if
#: installed — probe first").  It is NOT required: every statistic here is
#: computed with numpy, and statsmodels is only used to *validate* the numbers.
try:  # pragma: no cover - environment dependent
    import statsmodels  # noqa: F401
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False


def statsmodels_adf_tau(series, *, max_lags: int | None = None,
                        regression: str = "nc") -> float | None:
    """``statsmodels.adfuller``'s tau for the same regression, or ``None``.

    Only a **validation** helper (used by the tests when statsmodels happens to
    be installed); the production path never imports it.
    """
    if not HAS_STATSMODELS:
        return None
    try:  # pragma: no cover - optional dependency
        from statsmodels.tsa.stattools import adfuller
        res = adfuller(np.asarray(pd.Series(series).astype(float), dtype=float),
                       maxlag=max_lags, regression=regression, autolag=None)
        return float(res[0])
    except Exception:  # pragma: no cover
        return None


def ols_hedge_ratio(y, x) -> dict:
    """Static OLS hedge ratio ``y = alpha + beta·x`` with standard errors."""
    yv = np.asarray(pd.Series(y).astype(float), dtype=float)
    xv = np.asarray(pd.Series(x).astype(float), dtype=float)
    ok = np.isfinite(yv) & np.isfinite(xv)
    yv, xv = yv[ok], xv[ok]
    n = len(yv)
    if n < 10:
        return {"beta": float("nan"), "alpha": float("nan"), "n": n,
                "se_beta": float("nan"), "r2": float("nan")}
    X = np.column_stack([xv, np.ones(n)])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = n - 2
    sigma2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    cov = sigma2 * np.linalg.pinv(X.T @ X) if dof > 0 else np.full((2, 2), np.nan)
    tss = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else float("nan")
    return {"beta": float(beta[0]), "alpha": float(beta[1]), "n": int(n),
            "se_beta": float(math.sqrt(max(cov[0, 0], 0.0))),
            "resid_sd": float(math.sqrt(max(sigma2, 0.0))),
            "r2": float(r2)}


def engle_granger(y, x, *, max_lags: int | None = None) -> dict:
    """Engle-Granger test of ``y`` on ``x`` (levels).

    Returns the OLS hedge ratio, the residual, its ADF tau and the simulated
    Engle-Granger p-value plus critical values.  ``is_cointegrated`` is the
    plan's threshold decision (``p ≤ PAIRS_MAX_ADF_PVALUE``); callers must use
    :func:`pair_guard` — the test alone does not check sample length or the
    half-life.
    """
    ols = ols_hedge_ratio(y, x)
    yv = pd.Series(y).astype(float)
    xv = pd.Series(x).astype(float)
    spread = yv - (ols["alpha"] + ols["beta"] * xv)
    s = spread.to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    adf = adf_regression(s, max_lags=max_lags, regression="nc")
    p = tau_pvalue(adf["tau"], "eg_c", len(s))
    return {
        "beta": ols["beta"], "alpha": ols["alpha"], "se_beta": ols["se_beta"],
        "r2": ols["r2"], "n_obs": int(len(s)), "adf_tau": adf["tau"],
        "adf_lag": adf["lag"], "p_value": p,
        "critical_values": tau_critical_values("eg_c", len(s)),
        "sim_length": admissible_sim_length(len(s)),
        "is_cointegrated": bool(p <= PAIRS_MAX_ADF_PVALUE),
        "spread": spread,
    }


def rolling_cointegration(
    y,
    x,
    *,
    window: int = 500,
    step: int | None = None,
    max_windows: int = 60,
    max_lags: int = 1,
) -> dict:
    """Share of rolling windows in which the pair passes the E-G test.

    A single full-sample p-value hides the question that matters for a live
    pairs strategy: *the pair is only tradeable while its window is
    cointegrated*.  This walks non-overlapping (or ``step``-strided) windows,
    runs :func:`engle_granger` on each and reports the pass share, the median
    p-value and the distribution of the fitted hedge ratios.  Bounded by
    ``max_windows`` so a 35 k-bar 15 m series stays cheap.

    A pass share near ``PAIRS_MAX_ADF_PVALUE`` (0.05) is the **data-mining
    baseline**: it is what pure noise produces, so a pair is only interesting
    when its share is materially higher.
    """
    yv = pd.Series(y).astype(float)
    xv = pd.Series(x).astype(float)
    if not yv.index.equals(xv.index):
        joined = pd.concat([yv.rename("y"), xv.rename("x")], axis=1).dropna()
        yv, xv = joined["y"], joined["x"]
    n = len(yv)
    w = int(max(window, 60))
    stride = int(step) if step else w
    p_values: list[float] = []
    betas: list[float] = []
    starts: list[int] = []
    i = 0
    while i + w <= n and len(p_values) < int(max_windows):
        res = engle_granger(yv.iloc[i:i + w], xv.iloc[i:i + w], max_lags=max_lags)
        p_values.append(float(res["p_value"]))
        betas.append(float(res["beta"]))
        starts.append(int(i))
        i += max(stride, 1)
    if not p_values:
        return {"n_windows": 0, "pass_share": 0.0, "median_p": float("nan"),
                "beta_std": float("nan"), "window": w, "baseline": PAIRS_MAX_ADF_PVALUE}
    arr = np.asarray(p_values, dtype=float)
    return {
        "n_windows": int(len(arr)), "window": w,
        "pass_share": float((arr <= PAIRS_MAX_ADF_PVALUE).mean()),
        "median_p": float(np.median(arr)), "min_p": float(arr.min()),
        "beta_mean": float(np.mean(betas)), "beta_std": float(np.std(betas)),
        "pass_indices": [starts[k] for k in range(len(arr))
                         if arr[k] <= PAIRS_MAX_ADF_PVALUE],
        "baseline": float(PAIRS_MAX_ADF_PVALUE),
    }


# ── Kalman hedge ratio (Tsay ch. 7) ─────────────────────────────────────

def kalman_hedge_ratio(
    y,
    x,
    *,
    delta: float = KALMAN_DELTA,
    p0: float = KALMAN_P0,
    r_var: float | None = None,
    warm_start: bool | None = None,
) -> dict:
    """Time-varying ``(beta_t, alpha_t)`` by a 2-state Kalman filter.

    See the module docstring for the state/observation equations.

    Two settings decide whether the filter is usable at all, and both are the
    reason this function takes explicit arguments:

    ``r_var`` (observation noise)
        defaults to the **OLS residual variance** of ``y`` on ``x`` — the noise
        of ``y`` around the hedge line.  Scaling it from ``var(y)`` instead is a
        measured bug: on BTC/ETH 1h ``var(y) = 0.0400`` against an OLS residual
        variance of ``0.0094`` (4.3×), and with that ``R`` the filter barely
        updates its state and ``β`` collapses to 0.058 against an OLS β of
        0.628 — i.e. the "spread" becomes the raw price of ``y`` and the pair
        trade silently turns into a directional one.
    ``warm_start``
        ``True`` (default) initialises the state at the OLS ``[β, α]`` and
        ``P₀ = p0·I``; ``False`` reproduces a cold start (``β₀ = 0``,
        ``α₀ = mean(y)``), which needs hundreds of bars to converge.

    Non-finite observations are **skipped** (the filter carries its state
    forward) — a missing bar must never become a zero return.

    Returns ``beta``, ``alpha`` (arrays aligned to the input index),
    ``innovation`` (the one-step prediction error ``v_t``, the honest
    out-of-sample spread) and ``spread`` (``y − α_t − β_t·x``), plus stability
    diagnostics ``beta_std``, ``beta_drift`` (|last − first| / mean|β|) and
    ``innovation_sd``.
    """
    yv = pd.Series(y).astype(float)
    xv = pd.Series(x).astype(float)
    n = len(yv)
    yn = yv.to_numpy(dtype=float)
    xn = xv.to_numpy(dtype=float)
    if n == 0:
        empty = pd.Series(dtype=float, index=yv.index)
        return {"beta": empty, "alpha": empty, "innovation": empty,
                "spread": empty, "beta_std": float("nan"),
                "beta_drift": float("nan"), "innovation_sd": float("nan"),
                "r_var": float("nan"), "delta": float(delta)}

    d = float(delta)
    if not 0.0 < d < 1.0:
        raise ValueError("delta must be in (0, 1)")
    q = d / (1.0 - d) * np.eye(2)

    ols = ols_hedge_ratio(yn, xn)
    if r_var is None:
        resid_sd = ols.get("resid_sd")
        # Measurement noise = the spread's own noise around the hedge line.
        r = float(resid_sd) ** 2 if resid_sd and np.isfinite(resid_sd) else 1e-6
        r = max(r, 1e-12)
    else:
        r = float(r_var)
    if warm_start is None:
        warm_start = True
    if warm_start and np.isfinite(ols["beta"]):
        theta = np.array([float(ols["beta"]), float(ols["alpha"])])
    else:
        theta = np.array([0.0,
                          float(np.nanmean(yn)) if np.isfinite(yn).any() else 0.0])
    P = np.eye(2) * float(p0)

    betas = np.full(n, np.nan)
    alphas = np.full(n, np.nan)
    innov = np.full(n, np.nan)
    for t in range(n):
        z = np.array([xn[t], 1.0])
        if not (np.isfinite(z[0]) and np.isfinite(yn[t])):
            betas[t], alphas[t] = theta[0], theta[1]
            continue
        P = P + q                      # predict
        zP = z @ P
        F = float(zP @ z) + r
        v = float(yn[t] - z @ theta)   # one-step-ahead innovation
        K = zP / F                     # gain
        theta = theta + K * v
        P = P - np.outer(K, zP)
        betas[t], alphas[t] = theta[0], theta[1]
        innov[t] = v

    spread = pd.Series(yn - alphas - betas * xn, index=yv.index)
    finite_beta = betas[np.isfinite(betas)]
    mean_abs = float(np.mean(np.abs(finite_beta))) if len(finite_beta) else float("nan")
    # Drift compares the mean of the first and last **10 %** of the path, not the
    # two endpoint values: a single filtered β can be noise, and an endpoint
    # comparison reported 0.19 for a synthetic β that truly moved 0.40 → 0.75.
    win = max(1, len(finite_beta) // 10) if len(finite_beta) else 0
    drift = float("nan")
    if win and mean_abs > 0:
        drift = abs(float(finite_beta[-win:].mean() - finite_beta[:win].mean())) / mean_abs
    return {
        "beta": pd.Series(betas, index=yv.index),
        "alpha": pd.Series(alphas, index=yv.index),
        "innovation": pd.Series(innov, index=yv.index),
        "spread": spread,
        "beta_std": float(np.std(finite_beta)) if len(finite_beta) else float("nan"),
        "beta_drift": float(drift),
        "innovation_sd": float(np.nanstd(innov)) if np.isfinite(innov).any() else float("nan"),
        "r_var": float(r), "delta": d,
        "beta_mean": float(np.mean(finite_beta)) if len(finite_beta) else float("nan"),
        "warm_start": bool(warm_start),
    }


# ── half-life / lookback ────────────────────────────────────────────────

def ou_half_life(spread) -> dict:
    """AR(1)/OU half-life of mean reversion, in bars.

    ``Δs_t = a + b·s_{t−1} + ε`` ⇒ ``κ = −b``, ``half_life = ln2/κ``,
    ``equilibrium = −a/b``.  ``b ≥ 0`` means the series is not mean-reverting
    and the half-life is reported as ``inf`` with ``mean_reverting = False``.
    """
    s = pd.Series(spread).astype(float).dropna().to_numpy(dtype=float)
    if len(s) < 20:
        return {"half_life": float("inf"), "kappa": 0.0, "equilibrium": float("nan"),
                "b": float("nan"), "n": int(len(s)), "mean_reverting": False}
    ds = np.diff(s)
    lag = s[:-1]
    X = np.column_stack([lag, np.ones_like(lag)])
    beta, *_ = np.linalg.lstsq(X, ds, rcond=None)
    b, a = float(beta[0]), float(beta[1])
    if b >= 0:
        return {"half_life": float("inf"), "kappa": 0.0,
                "equilibrium": float("nan"), "b": b, "n": int(len(s)),
                "mean_reverting": False}
    kappa = -b
    return {
        "half_life": float(math.log(2.0) / kappa), "kappa": float(kappa),
        "equilibrium": float(-a / b), "b": b, "n": int(len(s)),
        "mean_reverting": True,
    }


def lookback_from_half_life(
    half_life: float,
    *,
    multiple: float = PAIRS_LOOKBACK_MULT,
    min_lookback: int = PAIRS_MIN_LOOKBACK,
    max_lookback: int = PAIRS_MAX_LOOKBACK,
) -> int:
    """``L = clip(round(multiple × half_life), min, max)`` (bars)."""
    if not np.isfinite(half_life) or half_life <= 0:
        return int(max_lookback)
    return int(min(max(int(round(float(multiple) * float(half_life))),
                       int(min_lookback)), int(max_lookback)))


# ── the hard guard ──────────────────────────────────────────────────────

def pair_guard(
    *,
    n_obs: int,
    p_value: float,
    half_life: float,
    beta: float,
    min_obs: int = PAIRS_MIN_OBS,
    max_p_value: float = PAIRS_MAX_ADF_PVALUE,
    min_half_life: float = PAIRS_MIN_HALF_LIFE,
    max_half_life: float = PAIRS_MAX_HALF_LIFE,
) -> dict:
    """Refuse to trade a pair unless every precondition holds.

    Refuses on: too few observations (``< min_obs``), a failed cointegration
    test (``p > max_p_value`` or a non-finite tau), a non-mean-reverting or
    out-of-range half-life, and a non-finite/non-positive hedge ratio.  Returns
    ``{"allowed", "reason", ...}`` with every input echoed so a journal record
    can be reproduced — the same contract as
    :func:`core.ml.credibility.credibility_gate`.
    """
    reasons: list[str] = []
    if int(n_obs) < int(min_obs):
        reasons.append(f"sample too short ({int(n_obs)} < {int(min_obs)} bars)")
    if not np.isfinite(p_value):
        reasons.append("cointegration p-value is not finite (ADF failed)")
    elif float(p_value) > float(max_p_value):
        reasons.append(
            f"cointegration rejected (ADF p={float(p_value):.4f} > {float(max_p_value):.2f})")
    if not np.isfinite(half_life):
        reasons.append("half-life not finite (spread is not mean-reverting)")
    elif float(half_life) < float(min_half_life):
        reasons.append(
            f"half-life too short ({float(half_life):.2f} < {float(min_half_life):.2f} bars) — noise")
    elif float(half_life) > float(max_half_life):
        reasons.append(
            f"half-life too long ({float(half_life):.2f} > {float(max_half_life):.2f} bars) — directional bet")
    if not np.isfinite(beta) or abs(float(beta)) <= 0:
        reasons.append(f"hedge ratio unusable (beta={beta})")
    return {
        "allowed": not reasons,
        "reason": "pass" if not reasons else "; ".join(reasons),
        "n_obs": int(n_obs), "p_value": float(p_value),
        "half_life": float(half_life), "beta": float(beta),
        "min_obs": int(min_obs), "max_p_value": float(max_p_value),
    }


# ── fitting + signals ───────────────────────────────────────────────────

@dataclass
class PairFit:
    """Everything the guard needs plus the series a signal is built from."""

    symbol_y: str
    symbol_x: str
    interval: str
    method: str
    beta: float
    alpha: float
    hedge: pd.Series
    spread: pd.Series
    adf: dict
    half_life: dict
    lookback: int
    guard: dict
    index: pd.Index
    #: Half-life of the *Kalman* spread — a diagnostic, never a test: the filter
    #: absorbs level shifts, so this number is small even for a non-stationary
    #: relationship (measured: 2.1 bars for BTC/ETH whose OLS spread is *not*
    #: mean-reverting at all).
    half_life_kalman: dict | None = None

    @property
    def allowed(self) -> bool:
        return bool(self.guard.get("allowed"))

    def summary(self) -> dict:
        """Flat, JSON-able research record (no Series)."""
        return {
            "pair": f"{self.symbol_y}/{self.symbol_x}", "interval": self.interval,
            "method": self.method, "n_obs": int(self.adf.get("n_obs", 0)),
            "beta_ols": self.adf.get("beta"), "alpha_ols": self.adf.get("alpha"),
            "adf_tau": self.adf.get("adf_tau"), "adf_lag": self.adf.get("adf_lag"),
            "adf_p_value": self.adf.get("p_value"),
            "critical_values": self.adf.get("critical_values"),
            "half_life_bars": self.half_life.get("half_life"),
            "half_life_kalman_bars": (self.half_life_kalman or {}).get("half_life"),
            "lookback_bars": int(self.lookback),
            "guard_allowed": self.allowed, "guard_reason": self.guard.get("reason"),
        }


def fit_pair(
    y: pd.Series,
    x: pd.Series,
    *,
    symbol_y: str = "Y",
    symbol_x: str = "X",
    interval: str = "1h",
    method: str = "kalman",
    max_lags: int | None = None,
    kalman_delta: float = KALMAN_DELTA,
) -> PairFit:
    """Fit a pair: OLS test → (optionally) Kalman hedge → half-life → lookback.

    ``method = "kalman"`` uses the Kalman ``beta_t`` for the traded spread (the
    ADF test is still run on the OLS residual, which is what Engle-Granger
    specifies); ``method = "ols"`` uses the static OLS spread.
    """
    yv = pd.Series(y).astype(float)
    xv = pd.Series(x).astype(float)
    if not yv.index.equals(xv.index):
        joined = pd.concat([yv.rename("y"), xv.rename("x")], axis=1).dropna()
        yv, xv = joined["y"], joined["x"]
    eg = engle_granger(yv, xv, max_lags=max_lags)
    # The half-life / equilibrium are properties of the **tested** (OLS) spread:
    # measuring them on the Kalman spread is not evidence, because the filter
    # absorbs level shifts by construction and makes any spread look
    # mean-reverting.  The Kalman one is reported as a diagnostic only.
    hl_ols = ou_half_life(eg["spread"])
    if method == "kalman":
        kf = kalman_hedge_ratio(yv, xv, delta=kalman_delta)
        hedge = kf["beta"]
        spread = kf["spread"]
        hl_kalman = ou_half_life(spread)
    else:
        hedge = pd.Series(float(eg["beta"]), index=yv.index)
        spread = eg["spread"]
        hl_kalman = dict(hl_ols)
    lookback = lookback_from_half_life(hl_ols["half_life"])
    guard = pair_guard(n_obs=eg["n_obs"], p_value=eg["p_value"],
                       half_life=hl_ols["half_life"], beta=float(eg["beta"]))
    return PairFit(
        symbol_y=symbol_y, symbol_x=symbol_x, interval=interval, method=method,
        beta=float(eg["beta"]), alpha=float(eg["alpha"]), hedge=hedge,
        spread=spread.dropna(), adf=eg, half_life=hl_ols, lookback=lookback,
        guard=guard, index=yv.index, half_life_kalman=hl_kalman,
    )


def rolling_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """``z_t`` from the **previous** ``lookback`` bars (excludes bar ``t``).

    Using ``mean/std`` of ``[t−L, t−1]`` (i.e. ``shift(1)``) is what makes the
    z-score tradeable: the bar being decided on is not part of its own
    normalisation, so a large move cannot shrink its own z-score.
    """
    s = pd.Series(spread).astype(float)
    window = max(int(lookback), 2)
    mean = s.shift(1).rolling(window).mean()
    sd = s.shift(1).rolling(window).std(ddof=1)
    z = (s - mean) / sd.replace(0.0, np.nan)
    return z.astype(float)


def pairs_positions(
    z: pd.Series,
    *,
    z_entry: float = PAIRS_Z_ENTRY,
    z_exit: float = PAIRS_Z_EXIT,
    z_stop: float = PAIRS_Z_STOP,
) -> pd.Series:
    """Deterministic z-score state machine → target position in {−1, 0, +1}.

    ``+1`` = long the spread (long y, short ``β``·x); ``−1`` = short the spread.
    A stop (``|z| ≥ z_stop``) flattens **and** latches: no new entry until
    ``|z| < z_entry`` again, because a spread that blew through the stop is
    evidence the relationship broke rather than stretched.  No look-ahead: the
    position at bar ``t`` is decided from ``z_t`` only and earns the return from
    ``t`` to ``t+1``.
    """
    pos = np.zeros(len(z), dtype=float)
    state = 0.0
    latched = False
    zv = z.to_numpy(dtype=float)
    for i in range(len(zv)):
        zi = zv[i]
        if not np.isfinite(zi):
            pos[i] = state
            continue
        if state != 0.0:
            if abs(zi) <= float(z_exit) or abs(zi) >= float(z_stop):
                if abs(zi) >= float(z_stop):
                    latched = True
                state = 0.0
        else:
            if latched and abs(zi) < float(z_entry):
                latched = False
            if not latched:
                if zi >= float(z_entry):
                    state = -1.0
                elif zi <= -float(z_entry):
                    state = 1.0
        pos[i] = state
    return pd.Series(pos, index=z.index)


def log_spread_notional(hedge: pd.Series) -> pd.Series:
    """Gross notional of one log-spread unit, in units of the y leg.

    With ``s = ln y − β·ln x`` the position is one unit of notional in ``y`` and
    ``|β|`` units in ``x``, so the gross notional is ``1 + |β|`` and a spread
    move ``Δs`` is the **fraction** ``Δs / (1 + |β|)`` of the capital deployed.
    """
    return 1.0 + pd.Series(hedge).astype(float).abs()


def price_spread_notional(close_y: pd.Series, close_x: pd.Series,
                          hedge: pd.Series) -> pd.Series:
    """Gross notional of one **price-unit** spread, in y price units.

    For a spread in price units (``s = y − α − β·x``) the gross notional is
    ``|y| + |β|·|x|``.  Mixing the two notional conventions (a log spread
    divided by a price notional) understates every return by four orders of
    magnitude — that is why the two helpers are separate and explicit.
    """
    return (pd.Series(close_y).astype(float).abs()
            + pd.Series(hedge).astype(float).abs() * pd.Series(close_x).astype(float).abs())


def spread_returns(
    spread: pd.Series,
    positions: pd.Series,
    gross_notional: pd.Series | float,
    *,
    leg_round_trip_cost_pct: float,
) -> pd.Series:
    """Per-bar **net** spread return for a held position (fraction of notional).

    ``gross_t = pos_t · (s_{t+1} − s_t) / N_t`` with ``N_t`` the gross notional
    from :func:`log_spread_notional` (log spread) or
    :func:`price_spread_notional` (price spread) — the return from bar ``t`` to
    ``t+1`` on the position decided at ``t`` (no look-ahead).

    Cost: a full round trip (open + close, both legs = four fills) costs
    ``leg_round_trip_cost_pct`` of the gross notional, so each unit of position
    change (open or close) is charged half of it::

        cost_t = leg_round_trip_cost_pct / 200 · |pos_t − pos_{t−1}|

    (the ``/100`` converts percent to a fraction).
    """
    s = pd.Series(spread).astype(float)
    pos = pd.Series(positions).astype(float).reindex(s.index).fillna(0.0)
    if np.ndim(gross_notional) == 0:
        denom: float | pd.Series = float(gross_notional)
    else:
        denom = pd.Series(gross_notional).astype(float).reindex(s.index)
        denom = denom.replace(0.0, np.nan)
    delta = s.shift(-1) - s
    gross = pos * delta / denom
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * float(leg_round_trip_cost_pct) / 200.0
    return (gross - cost).rename("net_return")


def pair_trades(
    positions: pd.Series,
    net_returns: pd.Series,
) -> list[dict]:
    """Contiguous position segments with their accumulated net return.

    One "trade" = a maximal run of the same non-zero position.  Its net return
    is the sum of the per-bar net returns **including the bar on which the
    position closes** — that bar carries the closing half of the round-trip
    cost, and dropping it (as the first revision did) charged every trade only
    once, i.e. half of the true cost.  Costs are already inside
    :func:`spread_returns`.
    """
    pos = pd.Series(positions).astype(float)
    r = pd.Series(net_returns).astype(float).reindex(pos.index).fillna(0.0)
    idx = list(pos.index)
    values = pos.to_numpy(dtype=float)
    trades: list[dict] = []
    i = 0
    n = len(values)
    while i < n:
        if values[i] == 0.0:
            i += 1
            continue
        side = "long" if values[i] > 0 else "short"
        start = i
        # The run ends at the first bar with a *different* position; that bar
        # (the closing fill, or the reversal fill) is part of this trade.
        j = i + 1
        while j < n and values[j] == values[i]:
            j += 1
        end = min(j, n - 1)
        seg = r.iloc[start:end + 1]
        trades.append({
            "side": side, "entry": idx[start], "exit": idx[end],
            "bars": int(end - start + 1), "net_return": float(seg.sum()),
        })
        i = j
    return trades


def trade_statistics(trades: list[dict]) -> dict:
    """Mean / SD / t / PSR / win-rate of a list of pair trades."""
    rets = np.asarray([t["net_return"] for t in trades], dtype=float)
    n = len(rets)
    if n == 0:
        return {"n_trades": 0, "mean": 0.0, "sd": 0.0, "t_stat": 0.0,
                "psr": 0.0, "win_rate": 0.0, "total": 0.0, "bars": 0}
    sd = float(rets.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else 0.0
    return {
        "n_trades": int(n), "mean": float(rets.mean()), "sd": sd,
        "t_stat": float(rets.mean() / se) if se > 0 else 0.0,
        "psr": probabilistic_sharpe(n, float(rets.mean()), sd),
        "win_rate": float((rets > 0).mean()), "total": float(rets.sum()),
        "bars": int(sum(t["bars"] for t in trades)),
    }


@dataclass
class PairsSignal:
    """The pairs signal in the shape the shared kernel consumes.

    :attr:`indicator_signal` is the ``{-1, 0, +1}`` input of
    ``core.strategy.evaluation_kernel.fuse_signals`` (with ``ml_enabled=False``
    and no news in backtest) and :attr:`confidence` the ``|z|``-based conviction
    in ``(0, 1]``.  Nothing in the live path consumes this unless a caller wires
    a pairs strategy explicitly (:data:`PAIRS_ENABLED` is ``False``).
    """

    allowed: bool
    reason: str
    indicator_signal: float
    confidence: float
    z: float
    beta: float
    lookback: int
    half_life: float
    p_value: float

    def to_kernel_input(self) -> dict:
        """``fuse_signals`` keyword arguments for this signal."""
        return {"indicator_signal": self.indicator_signal, "ml_enabled": False,
                "ml_confidence": 0.5}


def pairs_signal(fit: PairFit, z: pd.Series, positions: pd.Series) -> PairsSignal:
    """Last-bar signal for a fitted pair, gated by :attr:`PairFit.guard`.

    When the guard refuses, ``indicator_signal`` is 0 and ``allowed`` is
    ``False`` — a refused pair is flat, never a small position.
    """
    if len(z) == 0 or len(positions) == 0:
        return PairsSignal(False, "no data", 0.0, 0.0, float("nan"),
                           fit.beta, fit.lookback, float("nan"), float("nan"))
    z_last = float(pd.Series(z).iloc[-1])
    pos_last = float(pd.Series(positions).iloc[-1])
    if not fit.allowed:
        return PairsSignal(False, str(fit.guard.get("reason")), 0.0, 0.0, z_last,
                           fit.beta, fit.lookback,
                           float(fit.half_life.get("half_life", float("nan"))),
                           float(fit.adf.get("p_value", float("nan"))))
    conf = float(min(1.0, abs(z_last) / max(PAIRS_Z_STOP, 1e-9)))
    return PairsSignal(
        True, "pass", float(np.sign(pos_last)), conf, z_last, fit.beta,
        fit.lookback, float(fit.half_life.get("half_life", float("nan"))),
        float(fit.adf.get("p_value", float("nan"))))


def default_leg_cost_pct(config=None, *, symbol: str = "BTCUSDT",
                         order_type: str = "market") -> float:
    """Round-trip cost **per leg** (%), from the sim cost model.

    ``core.ml.credibility.cost_pct_for`` is the single source of truth for the
    cost the fills actually pay; a pairs trade crosses two legs, so the caller
    charges it twice (see :func:`spread_returns`).  Without a config object the
    documented defaults (0.04 % taker / 0.01 % half-spread / 2 bp slippage →
    0.14 %) are used.
    """
    return float(cost_pct_for(config, symbol=symbol, order_type=order_type))


__all__ = [
    "PAIRS_ENABLED", "PAIRS_MIN_OBS", "PAIRS_MAX_ADF_PVALUE",
    "PAIRS_MIN_HALF_LIFE", "PAIRS_MAX_HALF_LIFE", "PAIRS_LOOKBACK_MULT",
    "PAIRS_MIN_LOOKBACK", "PAIRS_MAX_LOOKBACK", "PAIRS_Z_ENTRY",
    "PAIRS_Z_EXIT", "PAIRS_Z_STOP", "KALMAN_DELTA", "SIM_REPS", "SIM_SEED",
    "HAS_STATSMODELS",
    "tau_null_distribution", "admissible_sim_length", "tau_pvalue",
    "tau_critical_values", "adf_regression", "statsmodels_adf_tau",
    "ols_hedge_ratio", "engle_granger", "rolling_cointegration",
    "kalman_hedge_ratio", "ou_half_life",
    "lookback_from_half_life", "pair_guard", "PairFit", "fit_pair",
    "rolling_zscore", "pairs_positions", "log_spread_notional",
    "price_spread_notional", "spread_returns",
    "pair_trades", "trade_statistics", "PairsSignal", "pairs_signal",
    "default_leg_cost_pct",
]
