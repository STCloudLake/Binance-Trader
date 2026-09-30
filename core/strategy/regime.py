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
The tercile classifier and the trend filter are **causal**: the tercile
thresholds use expanding quantiles of the past only, the trend filter is an EMA
of the past.

The HMM is causal **only in the mode that says so**.  ``hmm_two_state``'s
default (whole-sample EM) fits ``μ, σ, A`` on every bar including the future, so
its "filtered" posterior is causal *given the parameters* but the labels are
not: appending future bars moves σ and can move a Viterbi label before the
appended region (measured: 2 of 3 synthetic seeds).  Its reported accuracy is
**in-sample** and is labelled that way in
``tests/test_p34_audit_fixes.py``/doc 11.

**Two accuracies, and the 99.9 % one is the in-sample one.**  On the structure
this module's docs advertise — 3000 bars, true σ 0.002/0.010, change points at
1000 and 2000 — the whole-sample Viterbi decodes at **0.9987** (the figure the
docs quote as "≈99.9 %"), but its parameters saw the very regime it is being
scored on, so that number is a fit diagnostic and not tradeable.  The causal
path is the honest number: **0.758–0.815** (out of sample; seeds 5/7/11 of the
generator in ``tests/test_p34_audit_fixes.py``, which also pins it), at a
detection latency of 5–145 bars at the first change point and 24–27 at the
second.  The two are different quantities and only the second is a live claim.

``hmm_two_state_causal`` (selected automatically the moment
:data:`REGIME_GATING_ENABLED` is turned on, or by ``causal=True``) refits the
parameters on a documented schedule using only the past and decodes each bar
with a forward-only pass, so appending bars cannot change an earlier label.  The
*smoothed* posterior over the whole sample is reported separately as
``posterior_smoothed`` and is explicitly **not** tradeable.
:func:`gate_regimes` refuses to gate on a table whose HMM labels are not causal
rather than trusting the caller.

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
#: Causal-HMM controls: refit the parameters every N bars on the past only, and
#: emit no label for the first ``HMM_CAUSAL_WARMUP`` bars (the filtered posterior
#: has not forgotten its ``pi`` seed before that).  Both are documented warm-up
#: knobs, not hidden magic: they are reported in the causal fit's return value.
HMM_CAUSAL_REFIT_EVERY = 250
HMM_CAUSAL_WARMUP = 250
#: A refit whose two fitted σ are within this ratio has not found two states —
#: it has split one distribution in half, and EM on a window with only *calm*
#: data does exactly that (measured on the synthetic series: σ̂ = 0.00141 /
#: 0.00178 at t = 250, against a true 0.002 / 0.010).  The filtered posterior is
#: then two near-identical densities, so a 0.5 % bar reads as "stressed" forever
#: and the decode collapses to ~0.50 accuracy.  A refit below the ratio is
#: replaced by a deterministic quantile seed with the ratio imposed (the benign
#: case — a genuinely homoscedastic window — is unaffected in practice because
#: such a window has no state to detect).
HMM_MIN_SIGMA_RATIO = 1.5

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
    causal: bool | None = None,
) -> dict:
    """Fit a 2-state Gaussian HMM by EM; deterministic, numpy only.

    Returns ``states`` (Viterbi path, relabelled so ``0`` = calm / low σ),
    ``posterior_filtered`` (P(state | data ≤ t) — the tradeable one),
    ``posterior_smoothed`` (P(state | all data) — diagnostic only),
    ``transition``, ``mu``, ``sigma``, ``loglik``, ``n_iter`` and
    ``state`` (a ``"calm"``/``"stressed"`` Series for the last ``max_rows`` bars).

    ``max_rows`` bounds the O(n·k²) recursions: with more rows the **most
    recent** ``max_rows`` bars are fitted, which is what a live consumer needs.

    ``causal`` — read this before using the labels for anything that trades
        ``False`` (the value this function has always used) fits EM on the
        **whole** sample, so ``sigma``/``mu``/``A`` see the future and the
        "filtered" posterior is only causal *given* those parameters: truncating
        the series to 2000 bars moves σ from 0.00202/0.00968 to 0.00199/0.00968
        and flips an earlier Viterbi label on 2 of 3 synthetic seeds (measured;
        see ``tests/test_p34_audit_fixes.py``; the truncated pair used to be
        quoted as 0.00200/0.00973, which re-measures to 0.00199/0.00968 on this
        checkout).  The accuracy reported for this
        mode is **in-sample** — the 0.9987/"≈99.9 %" figure in the module
        docstring — and must be labelled as such; the causal path's out-of-sample
        accuracy on the same synthetic structure is 0.758–0.815.
        ``True`` delegates to :func:`hmm_two_state_causal`, which refits on a
        schedule and decodes each bar with a forward-only pass, so a label at bar
        ``t`` depends only on bars ≤ ``t``.
        ``None`` (default) resolves to ``REGIME_GATING_ENABLED``: full-sample
        while gating is off (the shipped default, bit-identical to the previous
        behaviour), causal the moment a caller turns gating on — a gate can then
        never consume a label that saw the future without asking for it.
        :func:`gate_regimes` enforces that contract independently.
    """
    use_causal = bool(REGIME_GATING_ENABLED) if causal is None else bool(causal)
    if use_causal:
        return hmm_two_state_causal(returns, n_iter=n_iter, tol=tol,
                                    max_rows=max_rows)
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
                "causal": False,
                "note": f"too few rows ({n} < 50)"}

    mu, sigma, A, pi, loglik, used = _hmm_em(x, n_iter=n_iter, tol=tol)

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
        "causal": False,
        "note": "",
    }


def _hmm_em(x: np.ndarray, *, n_iter: int, tol: float,
            sigma0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray,
                                                       float, int]:
    """Baum-Welch EM on one fixed sample → ``(mu, sigma, A, pi, loglik, iters)``.

    Extracted from :func:`hmm_two_state` verbatim: the whole-sample fit calls it
    once (bit-identical to the previous inline loop) and the causal refit calls
    it once per refit point on a strictly-past window.  The initialisation is
    deterministic — μ from the return quartiles, σ from the within-half
    dispersions, ``A = [[0.95, 0.05], [0.05, 0.95]]`` — so no random seed and no
    two runs can disagree.  ``sigma0`` overrides only the σ seed, which the
    causal decoder's minimum-separation fallback uses.
    """
    n = x.size
    q1, q3 = np.quantile(x, [0.25, 0.75])
    lower = x[x <= q1]
    upper = x[x >= q3]
    mu = np.array([float(lower.mean()), float(upper.mean())])
    if sigma0 is None:
        sigma = np.array([max(float(lower.std(ddof=0)), 1e-6),
                          max(float(upper.std(ddof=0)), 1e-6)])
    else:
        sigma = np.array([max(float(v), 1e-6) for v in np.asarray(sigma0, dtype=float)])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    loglik = float("-inf")
    used = 0
    for it in range(int(n_iter)):
        log_b = _emission_logpdf(x, mu, sigma)
        alpha, beta, ll = _forward_backward(log_b, A, pi)
        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        # xi (expected transitions) for the M-step.  The sum over t is contracted
        # in one einsum instead of a per-bar Python loop: same value, and it is
        # what keeps the causal decoder's per-refit cost affordable.
        b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
        num = alpha[:-1, :, None] * A[None, :, :] * (b[1:] * beta[1:])[:, None, :]
        den = num.sum(axis=(1, 2))
        safe = np.where(den > 0, den, 1.0)
        xi_sum = (num / safe[:, None, None]).sum(axis=0)
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
    return mu, sigma, A, pi, float(loglik), int(used)


def _hmm_forward_only(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                      A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Forward-only filtered posterior ``P(x_t | r_0..r_t)``, one row per bar.

    The backward pass — and therefore the whole-sample ``gamma`` and the Viterbi
    path — is dropped on purpose: it is what makes a label at bar ``t`` a
    function of bars ``> t``.  The recursion is the scaled forward pass of
    :func:`_forward_backward`, so ``out[t]`` is exactly that function's
    ``alpha[t]`` and needs ``O(t)`` work per bar.
    """
    log_b = _emission_logpdf(x, mu, sigma)
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    n, k = b.shape
    out = np.zeros((n, k))
    cur = pi * b[0]
    c = cur.sum() or 1e-300
    out[0] = cur / c
    for t in range(1, n):
        cur = (out[t - 1] @ A) * b[t]
        c = cur.sum() or 1e-300
        out[t] = cur / c
    return out


def _hmm_forward_last(x: np.ndarray, params: tuple) -> np.ndarray:
    """Filtered posterior at the **last** bar, ``P(x_n | r_0..r_n)``.

    The single-bar projection of :func:`_hmm_forward_only`: the recursion starts
    at the buffer start (the documented burn-in) and the intermediate rows are
    never materialised, which is all the causal decoder needs and halves the
    allocation.
    """
    mu, sigma, A, pi = params[0], params[1], params[2], params[3]
    log_b = _emission_logpdf(x, mu, sigma)
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    cur = pi * b[0]
    c = cur.sum() or 1e-300
    cur = cur / c
    for t in range(1, b.shape[0]):
        cur = (cur @ A) * b[t]
        c = cur.sum() or 1e-300
        cur = cur / c
    return cur


def _hmm_fit_separated(window: np.ndarray, *, n_iter: int, tol: float,
                       min_ratio: float = HMM_MIN_SIGMA_RATIO,
                       ) -> tuple[tuple, bool]:
    """EM fit that is guaranteed to return two **distinguishable** states.

    Returns ``(params, separated)``.  ``separated`` is ``True`` when the EM fit
    itself produced ``σ_max/σ_min ≥ min_ratio``; otherwise the fit is redone once
    from a deterministic quantile seed with the ratio imposed (the seed is
    :func:`hmm_two_state`'s own initialisation, so this is not a new estimator —
    only a floor on how far the two states may collapse into each other).
    """
    p = _hmm_em(window, n_iter=n_iter, tol=tol)
    s = np.sort(np.asarray(p[1], dtype=float))
    if s[0] > 0 and s[1] / s[0] >= float(min_ratio):
        return p, True
    q1, q3 = np.quantile(window, [0.25, 0.75])
    lo = window[window <= q1]
    hi = window[window >= q3]
    s_lo = max(float(lo.std(ddof=0)) if lo.size > 1 else 0.0, 1e-9)
    s_hi = max(s_lo * float(min_ratio), float(hi.std(ddof=0)) if hi.size > 1 else 0.0)
    p2 = _hmm_em(window, n_iter=n_iter, tol=tol, sigma0=np.array([s_lo, s_hi]))
    s2 = np.sort(np.asarray(p2[1], dtype=float))
    if s2[0] > 0 and s2[1] / s2[0] >= float(min_ratio):
        return p2, False
    return p, False


def hmm_two_state_causal(
    returns,
    *,
    refit_every: int = HMM_CAUSAL_REFIT_EVERY,
    warmup: int = HMM_CAUSAL_WARMUP,
    n_iter: int = HMM_ITER,
    tol: float = HMM_TOL,
    max_rows: int = 20000,
    params: tuple | None = None,
) -> dict:
    """Causal 2-state HMM labels: parameters only ever see the past.

    Construction (this is the whole causal argument):

    1. **Parameter schedule.** Parameters are refit at bar indices
       ``warmup, warmup + refit_every, …`` using only ``returns[:t]``; between
       refits the frozen fit is used.  The fit at index ``t`` therefore depends
       on bars ``< t`` only.
    2. **Forward-only decode.** Each bar's filtered posterior comes from the
       scaled forward recursion run from the **buffer start** (the documented
       burn-in, see below) up to that bar — no backward pass, no Viterbi, so no
       future bar can enter any label.  The recursion is advanced **once per
       refit segment** and the posterior rows for the segment are read off it,
       instead of restarting the sweep at every bar: within a segment the
       parameters (and therefore the emission matrix) are frozen, and the
       recursion is Markov in its own last row, so the two are the same numbers.
    3. **Burn-in.** The first ``warmup`` bars are ``"unknown"``: the filtered
       posterior has not forgotten its ``pi`` seed there.  The warm-up is
       reported in ``warmup`` and ``first_label_index`` rather than hidden.

    The buffer is the model window actually used (the expanding prefix capped at
    ``max_rows``).  A label is reported only when the buffer has at least 50 bars
    (:func:`hmm_two_state`'s own minimum), so very short inputs return
    ``"unknown"``.

    **Index mapping.**  Buffer row ``j`` is bar ``start + j`` and the segment's
    bars ``t … end-1`` are buffer rows ``t - start … end-1 - start``, so the
    label for bar ``tt`` is ``fwd[tt - start]``; ``lag_bars`` in the return value
    is that fixed offset (``0``).  The pre-fix code read ``fwd[k]`` for the k-th
    row of the segment, which labelled bar ``tt`` with the posterior of
    ``tt - (t - start)`` — lag 0, then 250, 500, … 2500 as the buffer refilled.
    Measured on the advertised synthetic structure, that turned a 0.758 decode
    into **0.52–0.56** on seeds 5/7/11 of the generator in
    ``tests/test_p34_audit_fixes.py`` — no better than the verifier's draw — and
    on some draws worse, because the label described a bar up to 2500 bars old.
    (An earlier revision of this docstring quoted 0.156 for the same defect; that
    figure does not reproduce on any of those three seeds.)

    Cost: one EM fit per refit point plus ``refits`` forward sweeps of the buffer
    (not one per bar), so a 3 000-bar series with the defaults (250-bar refit,
    250-bar warm-up) is 12 fits + 12 sweeps, measured **≈2.3–2.6 s** on this
    checkout (seeds 5/7/11 of the generator in ``tests/test_p34_audit_fixes.py``;
    the EM fit dominates and the figure is machine-dependent — the older
    "2.0–4.5 s" band brackets it, and the "≈18–21 s" quoted in
    ``docs/core-algorithms/11-pairs-cointegration.md`` is not in this file and
    does not reproduce).  That is a
    research/diagnostic cost, not a per-bar one; the live path is
    :data:`REGIME_DIAGNOSTICS_ENABLED`-gated and off.

    Determinism: same inputs → same labels, bar for bar (no random seed
    anywhere), and **appending future bars cannot change any earlier label** —
    that is the property ``tests/test_p34_audit_fixes.py`` asserts, together with
    the index mapping and the accuracy recovery on the synthetic structure.
    """
    r = pd.Series(returns, dtype=float).dropna()
    x_full = r.to_numpy(dtype=float)
    n = x_full.size
    limit = int(max(int(max_rows), 50))
    step = max(int(refit_every), 1)
    warm = max(int(warmup), 0)
    if n < 50:
        return {"states": np.zeros(n, dtype=int), "transition": np.eye(2),
                "mu": np.zeros(2), "sigma": np.zeros(2), "loglik": float("nan"),
                "n_iter": 0, "posterior_filtered": np.full((n, 2), 0.5),
                "posterior_smoothed": np.full((n, 2), 0.5),
                "state": pd.Series("unknown", index=r.index, dtype=object),
                "causal": True, "warmup": warm, "first_label_index": None,
                "refit_every": step, "n_refits": 0, "lag_bars": 0,
                "note": f"too few rows ({n} < 50)"}

    states = np.zeros(n, dtype=int)
    filtered = np.full((n, 2), 0.5)
    labels = np.array(["unknown"] * n, dtype=object)
    n_refits = 0
    n_degenerate = 0
    refit_points: list[int] = []
    fixed_params = params is not None
    t = 0
    while t < n:
        need_fit = params is None or (t >= warm and (t - warm) % step == 0)
        if need_fit and t >= 50 and not fixed_params:
            window = x_full[max(0, t - limit):t]
            if window.size >= 50:
                params, separated = _hmm_fit_separated(window, n_iter=n_iter, tol=tol)
                n_refits += 1
                refit_points.append(int(t))
                if not separated:
                    n_degenerate += 1
        if params is None:
            t += 1
            continue
        # The next bar whose (t - warm) hits the refit grid: the frozen
        # parameters are valid for [t, end).
        if t < warm:
            end = warm
        else:
            end = t + step if warm == 0 else warm + ((t - warm) // step + 1) * step
        end = int(min(max(end, t + 1), n))
        start = max(0, t - limit)
        buf = x_full[start:end]
        mu, sigma, A, pi = params[0], params[1], params[2], params[3]
        order = np.argsort(sigma)
        fwd = _hmm_forward_only(buf, mu, sigma, A, pi)
        # Index mapping (the defect this line fixes).  ``fwd[j]`` is the filtered
        # posterior of buffer row ``j``, and buffer row ``j`` is bar
        # ``start + j``; the segment's rows are bars ``t … end-1``, i.e. buffer
        # rows ``t - start … end-1 - start``.  Reading ``fwd[k]`` for the k-th row
        # of the *segment* therefore emitted the posterior of bar
        # ``start + k = tt - (t - start)`` — the label for bar ``tt`` was the
        # posterior of bar ``tt - lag`` with ``lag = t - start`` (0 on the warm-up
        # segment, then 250, 500, … as the buffer filled to ``max_rows``).  On the
        # advertised synthetic structure (3000 bars, σ 0.002/0.010, change points
        # 1000/2000) that lag turned a 0.76-accuracy decode into 0.16–0.52, i.e.
        # worse than a coin flip.  ``tt - start`` is the mapping that labels the
        # bar being decoded.
        for k, tt in enumerate(range(t, end)):
            post = fwd[tt - start]
            # Relabel by the fit's own volatility order, so state 0 is the calm
            # one even across refits where the EM indices could swap.  The
            # argmax must be taken on the *relabelled* row: taking it before the
            # permutation and then remapping the index silently inverts the
            # label (measured as a 0.50-accuracy decode).
            ordered = post[order]
            filtered[tt] = ordered
            state_t = int(np.argmax(ordered))
            states[tt] = state_t
            if tt >= warm:
                labels[tt] = "calm" if state_t == 0 else "stressed"
        t = end

    first = int(np.argmax(labels != "unknown")) if (labels != "unknown").any() else None
    return {
        "states": states, "transition": (params[2] if params else np.eye(2)),
        "mu": (params[0] if params else np.zeros(2)),
        "sigma": (params[1] if params else np.zeros(2)),
        "loglik": float("nan"), "n_iter": int(len(refit_points)),
        "posterior_filtered": filtered,
        "posterior_smoothed": np.full((n, 2), np.nan),
        "state": pd.Series(labels, index=r.index),
        "causal": True, "warmup": warm, "first_label_index": first,
        "refit_every": step, "n_refits": int(n_refits),
        "n_degenerate_fits": int(n_degenerate),
        #: Fixed label-vs-bar offset: ``0`` because the emitted posterior for bar
        #: ``tt`` is the filtered posterior of bar ``tt`` (buffer row
        #: ``tt - start``), not of an earlier row.  Reported so the buffer/lag
        #: bookkeeping is checkable from the return value instead of inferred
        #: from the code — the pre-fix value was ``t - start``, i.e. 0…2500.
        "lag_bars": 0,
        "note": "causal: forward-only decode, parameters refit on the past only",
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
    causal_hmm: bool | None = None,
) -> pd.DataFrame:
    """Per-bar regime table: ``vol_regime``, ``trend_regime``, ``regime``, HMM.

    ``regime`` is the composite label the gate consumes — the trend label when
    the tape is trending, ``"range_<vol>"`` otherwise, so a gate can say "only
    trade the breakout strategy in ``range_low``".  Time-outs and short samples
    fall back to ``"unknown"`` / ``"range"`` rather than to a guess.

    ``causal_hmm`` is forwarded to :func:`hmm_two_state` (``None`` →
    :data:`REGIME_GATING_ENABLED`, so the shipped all-off configuration keeps the
    historical whole-sample fit bit-for-bit while turning gating on switches the
    HMM to the causal path).  Either way the table records which mode produced
    its HMM columns in ``out.attrs["causal_hmm"]``, which
    :func:`gate_regimes` checks before letting a gate consume them.
    """
    close = pd.Series(df["close"], dtype=float)
    logret = np.log(close).diff()
    out = pd.DataFrame(index=close.index)
    out["vol_regime"] = volatility_terciles(logret, window=vol_window)
    out["trend_regime"] = trend_regimes(close, fast=trend_fast, slow=trend_slow)
    out["regime"] = np.where(out["trend_regime"] == "range",
                             "range_" + out["vol_regime"].astype(str),
                             out["trend_regime"])
    #: The tercile/trend columns are causal by construction (see the module
    #: docstring).  The flag is False until an HMM column set is added *and*
    #: that HMM was fitted causally, so a gate reading a table that never had an
    #: HMM is unaffected while a gate reading future-fitted HMM labels is refused.
    out.attrs["causal_hmm"] = True
    out.attrs["hmm_present"] = False
    if with_hmm:
        fit = hmm_two_state(logret, n_iter=hmm_iter, causal=causal_hmm)
        out.attrs["causal_hmm"] = bool(fit.get("causal"))
        out.attrs["hmm_present"] = True
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


class NonCausalRegimeError(RuntimeError):
    """Raised when a gate would consume regime labels that saw the future.

    A refusal, not a warning: a look-ahead label silently gating live trades is
    the exact failure mode ``tests/test_p34_audit_fixes.py`` exists to prevent,
    and a log line would not stop it.
    """


@dataclass
class RegimeGate:
    """Deterministic strategy/parameter gate by regime.

    ``allowed`` maps a strategy *kind* (see :data:`STRATEGY_KINDS`) to the set of
    regimes in which it may trade; a kind with no entry is allowed everywhere
    (the historical behaviour).  :attr:`enabled` is
    :data:`REGIME_GATING_ENABLED` unless a caller overrides it — with gating off,
    :meth:`allows` returns ``True`` for everything, which is the "no live
    behaviour change by default" contract.

    **Causality contract.**  ``allows`` only ever sees a *label string*, so the
    gate cannot tell where that label came from; :func:`gate_regimes` is the
    checked entry point.  It inspects ``table.attrs["causal_hmm"]`` (set by
    :func:`classify_regimes`) and raises :class:`NonCausalRegimeError` when the
    table carries HMM labels that were fitted on the whole sample.  The
    ``hmm_state`` / ``hmm_state_smoothed`` columns of a non-causal table are
    diagnostics only — the *composite* ``regime`` column is causal either way,
    so a caller that gates on ``regime`` alone is unaffected.
    """

    allowed: dict[str, set[str]] = field(default_factory=dict)
    enabled: bool = REGIME_GATING_ENABLED
    default_allow: bool = True
    require_causal: bool = True

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

    def check_table(self, table) -> None:
        """Raise unless ``table``'s HMM labels are causal (see the class doc).

        No-op when ``enabled`` is ``False`` (the shipped default, where the gate
        cannot change behaviour at all) or when ``require_causal`` is ``False``
        (an explicit opt-out for a research caller that wants the diagnostic
        labels anyway).
        """
        if not self.enabled or not self.require_causal:
            return
        attrs = getattr(table, "attrs", {}) or {}
        if bool(attrs.get("hmm_present", False)) and not bool(
                attrs.get("causal_hmm", False)):
            raise NonCausalRegimeError(
                "refusing to gate on whole-sample (look-ahead) HMM labels: "
                "rebuild the table with classify_regimes(..., causal_hmm=True) "
                "or gate on the composite 'regime' column only")
        return None

    def gate_row(self, table, strategy_kind: str, *, row: int = -1) -> dict:
        """Checked ``allows`` decision for one row of a regime table."""
        self.check_table(table)
        if len(table) == 0:
            return {"allowed": True, "regime": "unknown", "checked": True}
        regime = str(table["regime"].iloc[int(row)])
        return {"allowed": self.allows(strategy_kind, regime),
                "regime": regime, "checked": True}


def gate_regimes(gate: RegimeGate, table, strategy_kind: str) -> dict:
    """Refuse-or-allow a gate decision on a regime table (checked seam).

    The one call site a live path should use: it applies the causality contract
    of :meth:`RegimeGate.check_table` and then evaluates the last row, so an
    operator cannot accidentally gate on labels that saw the future.
    """
    return gate.gate_row(table, strategy_kind)


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
    "HMM_CAUSAL_REFIT_EVERY", "HMM_CAUSAL_WARMUP",
    "VOL_REGIMES", "TREND_REGIMES", "STRATEGY_KINDS", "DEFAULT_REGIME_MAP",
    "rolling_volatility", "volatility_terciles", "trend_regimes",
    "hmm_two_state", "hmm_two_state_causal", "classify_regimes", "classify_last",
    "regime_persistence", "detection_metrics", "RegimeGate", "default_gate",
    "gate_regimes", "NonCausalRegimeError",
]
