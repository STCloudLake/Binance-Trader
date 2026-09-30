"""Regime detection and gating (Tsay ch. 4: nonlinear / regime-switching models).

Principle
---------
A single parameter set cannot be right in every market state: a breakout rule
that earns in a trending, low-volatility tape gives it back in a choppy,
high-volatility one (and vice versa).  Tsay ch. 4 covers the two families that
model this switch-like behaviour — threshold/SETAR models and Markov-switching
models — and this module implements both sides of that choice at the cheapest
useful level:

* **volatility terciles** (+ a trend filter): fully transparent, no estimation,
  no dependency beyond numpy/pandas;
* **a 2-state Gaussian HMM** fitted with numpy EM (Baum-Welch) and decoded with
  Viterbi: the Markov-switching view, with a hard, deterministic fit.

Formulas
--------
Volatility terciles (causal — the thresholds come from the **past** only)::

    σ_t      = std(r_{t−w+1..t})                       (w = VOL_WINDOW bars)
    low/mid/high: σ_t compared with the expanding 1/3 and 2/3 quantiles of
                  σ_{1..t−1}   (expanding().quantile, shifted by one bar)

Trend filter::

    up    if close_t > EMA_slow and EMA_fast > EMA_slow
    down  if close_t < EMA_slow and EMA_fast < EMA_slow
    range otherwise

2-state Gaussian HMM (returns, ``x_t ∈ {0, 1}``)::

    P(x_t = j | x_{t−1} = i) = A_ij,      r_t | x_t = j ~ N(μ_j, σ_j²)
    E-step: scaled forward α, backward β, γ, ξ
    M-step: A_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i);  μ_j, σ_j² = weighted moments
    Viterbi: argmax path (deterministic, no sampling)

Initialisation is deterministic (μ from the return quartiles, σ from the
within-half dispersions, A = [[0.95, 0.05], [0.05, 0.95]]) — no random seed, so
two runs on the same series give the same states.  States are relabelled by
volatility afterwards, so state ``0`` is always the calm one.

No look-ahead
-------------
Both classifiers are **causal**: the tercile thresholds use expanding quantiles
of the past only, the trend filter is an EMA of the past, and the HMM's
``forward``/Viterbi decode of bar ``t`` uses bars ``≤ t`` (the *smoothed*
posterior over the whole sample is reported separately as
``posterior_smoothed`` and is explicitly **not** tradeable).  Tests assert that
the smoothed path differs from the filtered one in the expected direction.

Limitations
-----------
* Two states and a Gaussian emission is a caricature of a market; the HMM will
  happily split a single true state in two if its variance drifts.
* Terciles are relative: a "high" volatility regime is high *for this sample*,
  so the labels drift with the sample's own history (documented, and the reason
  the gate consumes the label rather than a fitted threshold).
* Regime labels are lagging by construction (any detector that is not lagging is
  using future data); :func:`detection_metrics` measures the latency instead of
  hiding it.
* Nothing is enabled by default: :data:`REGIME_GATING_ENABLED` and
  :data:`REGIME_DIAGNOSTICS_ENABLED` are ``False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── switches / tunables (all OFF by default) ────────────────────────────

#: Master switch: ``False`` ⇒ no strategy/parameter set is gated by a regime.
REGIME_GATING_ENABLED = False
#: Switch for merely *reporting* the regime in the live path (no gating).
REGIME_DIAGNOSTICS_ENABLED = False
#: Rolling window for realised volatility, and the trend EMAs.
VOL_WINDOW = 50
TREND_FAST = 50
TREND_SLOW = 200
#: EM controls.
HMM_ITER = 50
HMM_TOL = 1e-6
#: Minimum bars before any classifier emits a non-default label.
MIN_REGIME_ROWS = 100

VOL_REGIMES: tuple[str, ...] = ("low", "mid", "high")
TREND_REGIMES: tuple[str, ...] = ("trend_up", "trend_down", "range")
#: Strategy "kinds" the gate understands; a caller can pass any string.
STRATEGY_KINDS: tuple[str, ...] = (
    "trend", "mean_reversion", "breakout", "pairs", "scalp",
)


# ── tercile classifier ──────────────────────────────────────────────────

def rolling_volatility(returns, *, window: int = VOL_WINDOW,
                       min_periods: int = 20) -> pd.Series:
    """Rolling standard deviation of **past** returns (bar ``t`` included)."""
    r = pd.Series(returns, dtype=float)
    return r.rolling(int(window), min_periods=int(min_periods)).std()


def volatility_terciles(
    returns,
    *,
    window: int = VOL_WINDOW,
    min_periods: int = 20,
    quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
) -> pd.Series:
    """``low`` / ``mid`` / ``high`` volatility labels, causally determined.

    The thresholds at bar ``t`` are the **expanding** quantiles of
    ``σ_{1..t−1}`` — i.e. what was known before ``t`` — so a bar can never be
    labelled against a threshold that only its own volatility moved.  Bars with
    fewer than ``min_periods`` observations are ``"unknown"``.
    """
    vol = rolling_volatility(returns, window=window, min_periods=min_periods)
    lo = vol.shift(1).expanding(min_periods=min_periods).quantile(quantiles[0])
    hi = vol.shift(1).expanding(min_periods=min_periods).quantile(quantiles[1])
    out = pd.Series("unknown", index=vol.index, dtype=object)
    valid = vol.notna() & lo.notna() & hi.notna()
    out[valid & (vol <= lo)] = "low"
    out[valid & (vol > lo) & (vol <= hi)] = "mid"
    out[valid & (vol > hi)] = "high"
    return out


def trend_regimes(
    close,
    *,
    fast: int = TREND_FAST,
    slow: int = TREND_SLOW,
) -> pd.Series:
    """``trend_up`` / ``trend_down`` / ``range`` from a two-EMA alignment."""
    c = pd.Series(close, dtype=float)
    ema_f = c.ewm(span=int(fast), adjust=False).mean()
    ema_s = c.ewm(span=int(slow), adjust=False).mean()
    out = pd.Series("range", index=c.index, dtype=object)
    up = (c > ema_s) & (ema_f > ema_s)
    down = (c < ema_s) & (ema_f < ema_s)
    out[up] = "trend_up"
    out[down] = "trend_down"
    out.iloc[:min(MIN_REGIME_ROWS, len(out))] = "range"
    return out


# ── 2-state Gaussian HMM (numpy EM) ─────────────────────────────────────

def _emission_logpdf(r: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    var = np.maximum(sigma ** 2, 1e-12)
    return (-0.5 * (np.log(2.0 * math.pi * var))[None, :]
            - 0.5 * ((r[:, None] - mu[None, :]) ** 2) / var[None, :])


def _forward_backward(log_b: np.ndarray, A: np.ndarray,
                      pi: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Scaled forward/backward recursions → ``alpha``, ``beta``, log-likelihood."""
    n, k = log_b.shape
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    alpha = np.zeros((n, k))
    c = np.zeros(n)
    alpha[0] = pi * b[0]
    c[0] = alpha[0].sum() or 1e-300
    alpha[0] /= c[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ A) * b[t]
        c[t] = alpha[t].sum() or 1e-300
        alpha[t] /= c[t]
    beta = np.zeros((n, k))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (A @ (b[t + 1] * beta[t + 1])) / c[t + 1]
    loglik = float(np.log(c).sum() + log_b.max(axis=1).sum())
    return alpha, beta, loglik


def _viterbi(log_b: np.ndarray, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    n, k = log_b.shape
    delta = np.zeros((n, k))
    psi = np.zeros((n, k), dtype=int)
    with np.errstate(divide="ignore"):
        logA = np.log(A)
        logpi = np.log(pi)
    delta[0] = logpi + log_b[0]
    for t in range(1, n):
        trans = delta[t - 1][:, None] + logA
        psi[t] = np.argmax(trans, axis=0)
        delta[t] = trans[psi[t], np.arange(k)] + log_b[t]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def hmm_two_state(
    returns,
    *,
    n_iter: int = HMM_ITER,
    tol: float = HMM_TOL,
    max_rows: int = 20000,
) -> dict:
    """Fit a 2-state Gaussian HMM by EM; deterministic, numpy only.

    Returns ``states`` (Viterbi path, relabelled so ``0`` = calm / low σ),
    ``posterior_filtered`` (P(state | data ≤ t) — the tradeable one),
    ``posterior_smoothed`` (P(state | all data) — diagnostic only),
    ``transition``, ``mu``, ``sigma``, ``loglik``, ``n_iter`` and
    ``state`` (a ``"calm"``/``"stressed"`` Series for the last ``max_rows`` bars).

    ``max_rows`` bounds the O(n·k²) recursions: with more rows the **most
    recent** ``max_rows`` bars are fitted, which is what a live consumer needs.
    """
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) > int(max_rows):
        r = r.iloc[-int(max_rows):]
    x = r.to_numpy(dtype=float)
    n = len(x)
    if n < 50:
        idx = r.index
        return {"states": np.zeros(n, dtype=int), "transition": np.eye(2),
                "mu": np.zeros(2), "sigma": np.zeros(2), "loglik": float("nan"),
                "n_iter": 0, "posterior_filtered": np.full((n, 2), 0.5),
                "posterior_smoothed": np.full((n, 2), 0.5),
                "state": pd.Series("unknown", index=idx, dtype=object),
                "note": f"too few rows ({n} < 50)"}

    q1, q3 = np.quantile(x, [0.25, 0.75])
    lower = x[x <= q1]
    upper = x[x >= q3]
    mu = np.array([float(lower.mean()), float(upper.mean())])
    sigma = np.array([max(float(lower.std(ddof=0)), 1e-6),
                      max(float(upper.std(ddof=0)), 1e-6)])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    loglik = float("-inf")
    used = 0
    for it in range(int(n_iter)):
        log_b = _emission_logpdf(x, mu, sigma)
        alpha, beta, ll = _forward_backward(log_b, A, pi)
        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        # xi (expected transitions) for the M-step, using the scaled recursions.
        b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
        xi_sum = np.zeros((2, 2))
        for t in range(n - 1):
            num = (alpha[t][:, None] * A) * (b[t + 1] * beta[t + 1])[None, :]
            s = num.sum()
            xi_sum += num / s if s > 0 else num
        A_new = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
        pi_new = gamma[0] / max(gamma[0].sum(), 1e-300)
        w = gamma.sum(axis=0)
        mu_new = (gamma * x[:, None]).sum(axis=0) / np.maximum(w, 1e-300)
        var_new = (gamma * (x[:, None] - mu_new[None, :]) ** 2).sum(axis=0) \
            / np.maximum(w, 1e-300)
        A, pi, mu = A_new, pi_new, mu_new
        sigma = np.sqrt(np.maximum(var_new, 1e-12))
        used = it + 1
        if abs(ll - loglik) < float(tol):
            loglik = ll
            break
        loglik = ll

    log_b = _emission_logpdf(x, mu, sigma)
    alpha, beta, ll = _forward_backward(log_b, A, pi)
    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    states = _viterbi(log_b, A, pi)
    # Relabel by volatility: state 0 = the calm one, so labels are comparable
    # across fits (an HMM's state indices are arbitrary).
    order = np.argsort(sigma)
    remap = np.zeros(2, dtype=int)
    remap[order[0]], remap[order[1]] = 0, 1
    states = remap[states]
    gamma = gamma[:, order]
    sigma = sigma[order]
    mu = mu[order]
    return {
        "states": states, "transition": A, "mu": mu, "sigma": sigma,
        "loglik": float(ll), "n_iter": int(used),
        "posterior_filtered": alpha[:, order],
        "posterior_smoothed": gamma,
        "state": pd.Series(np.where(states == 0, "calm", "stressed"), index=r.index),
        "note": "",
    }


# ── feature bundle + gating ─────────────────────────────────────────────

def classify_regimes(
    df: pd.DataFrame,
    *,
    vol_window: int = VOL_WINDOW,
    trend_fast: int = TREND_FAST,
    trend_slow: int = TREND_SLOW,
    with_hmm: bool = False,
    hmm_iter: int = HMM_ITER,
) -> pd.DataFrame:
    """Per-bar regime table: ``vol_regime``, ``trend_regime``, ``regime``, HMM.

    ``regime`` is the composite label the gate consumes — the trend label when
    the tape is trending, ``"range_<vol>"`` otherwise, so a gate can say "only
    trade the breakout strategy in ``range_low``".  Time-outs and short samples
    fall back to ``"unknown"`` / ``"range"`` rather than to a guess.
    """
    close = pd.Series(df["close"], dtype=float)
    logret = np.log(close).diff()
    out = pd.DataFrame(index=close.index)
    out["vol_regime"] = volatility_terciles(logret, window=vol_window)
    out["trend_regime"] = trend_regimes(close, fast=trend_fast, slow=trend_slow)
    out["regime"] = np.where(out["trend_regime"] == "range",
                             "range_" + out["vol_regime"].astype(str),
                             out["trend_regime"])
    if with_hmm:
        fit = hmm_two_state(logret)
        idx = fit["state"].index
        # The HMM drops the leading NaN log-return, so its series is one bar
        # shorter than the frame: reindex rather than assume equal length (a
        # length mismatch here is a hard ValueError, not a silent shift).
        out["hmm_state"] = fit["state"].reindex(close.index).fillna("unknown")
        pf = pd.Series(fit["posterior_filtered"][:, 1], index=idx)
        out["hmm_prob_stressed"] = pf.reindex(close.index)
        smooth = pd.Series(fit["posterior_smoothed"][:, 1], index=idx)
        smooth = smooth.reindex(close.index)
        smoothed = pd.Series("unknown", index=close.index, dtype=object)
        smoothed[smooth > 0.5] = "stressed"
        smoothed[smooth <= 0.5] = "calm"
        smoothed[smooth.isna()] = "unknown"
        out["hmm_state_smoothed"] = smoothed
        out["hmm_sigma"] = fit["sigma"][1]
    return out


def classify_last(df: pd.DataFrame, **kwargs) -> dict:
    """The last bar's regime, as a flat dict (live-path seam)."""
    table = classify_regimes(df, **kwargs)
    if len(table) == 0:
        return {"vol_regime": "unknown", "trend_regime": "range",
                "regime": "unknown", "index": None}
    row = table.iloc[-1]
    out = {k: (v if isinstance(v, str) else
               (None if v is None or (isinstance(v, float) and not np.isfinite(v))
                else float(v)))
           for k, v in row.items()}
    # ISO string, not a Timestamp: this dict is serialised into the engine's
    # monitor payload (`StrategyEngine._sanitize` handles numpy, not pandas).
    out["index"] = str(table.index[-1])
    return out


def regime_persistence(regimes) -> dict:
    """Run-length statistics of a regime series (persistence in bars)."""
    s = pd.Series(regimes)
    runs: list[tuple[str, int]] = []
    if len(s) == 0:
        return {"n_runs": 0, "mean_run": 0.0, "max_run": 0, "by_regime": {}}
    current = s.iloc[0]
    length = 1
    for value in s.iloc[1:]:
        if value == current:
            length += 1
        else:
            runs.append((str(current), length))
            current, length = value, 1
    runs.append((str(current), length))
    lengths = np.asarray([l for _, l in runs], dtype=float)
    by: dict[str, float] = {}
    for name, length in runs:
        by.setdefault(name, []).append(length)  # type: ignore[arg-type]
    return {
        "n_runs": len(runs), "mean_run": float(lengths.mean()),
        "max_run": int(lengths.max()),
        "by_regime": {k: float(np.mean(v)) for k, v in by.items()},
    }


def detection_metrics(
    true_labels,
    predicted_labels,
    *,
    positive: str = "high",
    change_points: tuple[int, ...] = (),
    settle_run: int = 5,
) -> dict:
    """Accuracy plus the **latency** of a regime detector (diagnostic helper).

    ``accuracy`` is the share of bars whose predicted label matches the truth
    under the mapping ``predicted == positive`` ⟺ ``true == positive`` (other
    labels are treated as "not positive", which is how a gate consumes them).

    ``latency_bars`` (needs ``change_points``) is, for each change point, the
    number of bars until the detector has produced ``settle_run`` consecutive
    correct-and-final labels — ``None`` when it never settles inside the sample.
    Reporting latency is the point: a detector that is right 99 % of the time
    but 200 bars late cannot gate a trade.
    """
    true = pd.Series(true_labels).astype(str).to_numpy()
    pred = pd.Series(predicted_labels).astype(str).to_numpy()
    n = min(len(true), len(pred))
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "latency_bars": []}
    t_pos = true[:n] == positive
    p_pos = pred[:n] == positive
    acc = float((t_pos == p_pos).mean())
    latencies: list[int | None] = []
    for cp in change_points:
        cp = int(cp)
        if cp >= n:
            latencies.append(None)
            continue
        run = 0
        found: int | None = None
        for i in range(cp, n):
            if p_pos[i] == t_pos[i]:
                run += 1
                if run >= int(settle_run):
                    found = i - cp - int(settle_run) + 1
                    break
            else:
                run = 0
        latencies.append(found)
    return {"n": int(n), "accuracy": acc, "latency_bars": latencies,
            "positive": positive}


@dataclass
class RegimeGate:
    """Deterministic strategy/parameter gate by regime.

    ``allowed`` maps a strategy *kind* (see :data:`STRATEGY_KINDS`) to the set of
    regimes in which it may trade; a kind with no entry is allowed everywhere
    (the historical behaviour).  :attr:`enabled` is
    :data:`REGIME_GATING_ENABLED` unless a caller overrides it — with gating off,
    :meth:`allows` returns ``True`` for everything, which is the "no live
    behaviour change by default" contract.
    """

    allowed: dict[str, set[str]] = field(default_factory=dict)
    enabled: bool = REGIME_GATING_ENABLED
    default_allow: bool = True

    def allows(self, strategy_kind: str, regime: str) -> bool:
        if not self.enabled:
            return True
        regimes = self.allowed.get(str(strategy_kind))
        if regimes is None:
            return bool(self.default_allow)
        return str(regime) in {str(r) for r in regimes}

    def blocked_kinds(self, regime: str,
                      kinds: tuple[str, ...] = STRATEGY_KINDS) -> list[str]:
        """Which known strategy kinds this regime refuses (diagnostics)."""
        return [k for k in kinds if not self.allows(k, regime)]

    def size_multiplier(self, strategy_kind: str, regime: str,
                        base: float = 1.0, blocked: float = 0.0) -> float:
        """``base`` when allowed, ``blocked`` otherwise — for position sizing."""
        return float(base) if self.allows(strategy_kind, regime) else float(blocked)


#: The one documented default map, used only when a caller opts in.
DEFAULT_REGIME_MAP: dict[str, set[str]] = {
    "breakout": {"trend_up", "trend_down", "range_low"},
    "trend": {"trend_up", "trend_down"},
    "mean_reversion": {"range_low", "range_mid"},
    "scalp": {"range_low", "range_mid"},
    "pairs": {"range_low", "range_mid", "range_high"},
}


def default_gate(enabled: bool = REGIME_GATING_ENABLED) -> RegimeGate:
    """A gate with :data:`DEFAULT_REGIME_MAP` (disabled unless asked for)."""
    return RegimeGate(allowed={k: set(v) for k, v in DEFAULT_REGIME_MAP.items()},
                      enabled=enabled)


__all__ = [
    "REGIME_GATING_ENABLED", "REGIME_DIAGNOSTICS_ENABLED", "VOL_WINDOW",
    "TREND_FAST", "TREND_SLOW", "HMM_ITER", "HMM_TOL", "MIN_REGIME_ROWS",
    "VOL_REGIMES", "TREND_REGIMES", "STRATEGY_KINDS", "DEFAULT_REGIME_MAP",
    "rolling_volatility", "volatility_terciles", "trend_regimes",
    "hmm_two_state", "classify_regimes", "classify_last", "regime_persistence",
    "detection_metrics", "RegimeGate", "default_gate",
]
