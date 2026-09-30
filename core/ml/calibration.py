"""Probability calibration + the signed ML score (Phase P2, items 1 & 4).

The deployed model is not merely uninformative, it is **anti-calibrated**:
the audit measured ``P(up) = 0.91`` buckets realising ``0.22``, with OOS AUC
0.396–0.447 — i.e. the probability the engines fuse into ``(p − 0.5) × 2``
points the *wrong way*.  This module provides the two missing pieces:

* :class:`ProbabilityCalibrator` — Platt (logistic) or isotonic calibration,
  fitted on a hold-out slice, with a persistence bundle.
* :func:`signed_score` — the definition the fusion kernel now uses:
  ``score = clip((p / base_rate − 1) / scale, −1, +1)`` with
  ``scale = max(1/base_rate − 1, 1/(1−base_rate) − 1)``.  Centring on the model's
  **own base rate** (not a hard-coded 0.5) is what makes the sign meaningful
  when 59.5 % of bars are "no-move" and the traded subset is not balanced.  The
  score is **asymmetric** in the base rate (a maximal bearish call scores less
  in magnitude than a maximal bullish one when the base rate is below 0.5); see
  :func:`signed_score` for the exact numbers.

Consumers: :mod:`core.ml.credibility` (the gate), :mod:`core.ml.trainer`
(persisted with every model) and ``core.strategy.evaluation_kernel`` (the
fusion path).
"""

from __future__ import annotations

import numpy as np

#: Score is clipped into this range so one confident-but-wrong call cannot
#: dominate a fused score.
SCORE_CLIP = 1.0


# ── reliability curve ────────────────────────────────────────────────────

def _weighted_spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Spearman rank correlation (no scipy dependency)."""
    if len(x) < 3:
        return 0.0

    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(len(v), dtype=float)
        ranks[order] = np.arange(len(v), dtype=float)
        # Average tied ranks so identical bin frequencies do not break the rank.
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sums = np.zeros(len(counts))
            np.add.at(sums, inv, ranks)
            ranks = (sums / counts)[inv]
        return ranks

    rx, ry = _rank(x), _rank(y)
    wsum = float(w.sum())
    if wsum <= 0:
        return 0.0
    mx = float((w * rx).sum() / wsum)
    my = float((w * ry).sum() / wsum)
    cov = float((w * (rx - mx) * (ry - my)).sum())
    vx = float((w * (rx - mx) ** 2).sum())
    vy = float((w * (ry - my) ** 2).sum())
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return float(cov / np.sqrt(vx * vy))


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Binned reliability curve of a **binary** probability.

    Returns ``{"bin_centres", "observed", "counts", "monotone", "slope",
    "spearman", "brier", "ece"}``.

    ``monotone`` is the property the audit says is violated (predicted 0.91 →
    realised 0.22): a calibrated model has a weakly increasing observed
    frequency across predicted-probability bins, **and** a strongly positive
    rank correlation between bin confidence and bin frequency (``spearman``;
    note 1.0 is not required — bin frequencies are noisy at the extremes, while
    an inverted curve scores ``-1.0``).  A small monotonicity tolerance is
    allowed because bin noise is real; the caller can inspect ``observed``.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return {"bin_centres": [], "observed": [], "counts": [],
                "monotone": True, "slope": 0.0, "spearman": 0.0,
                "brier": 0.0, "ece": 0.0}

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, int(n_bins) - 1)

    centres, observed, counts = [], [], []
    for b in range(int(n_bins)):
        sel = idx == b
        if not sel.any():
            continue
        centres.append(float(p[sel].mean()))
        observed.append(float(y[sel].mean()))
        counts.append(int(sel.sum()))

    obs = np.asarray(observed, dtype=float)
    cnt = np.asarray(counts, dtype=float)
    spearman = 0.0
    # Weighted least-squares slope vs predicted probability.
    if len(centres) >= 3 and cnt.sum() > 0:
        x = np.asarray(centres, dtype=float)
        w = cnt / cnt.sum()
        xm = float((w * x).sum())
        ym = float((w * obs).sum())
        denom = float((w * (x - xm) ** 2).sum())
        slope = float((w * (x - xm) * (obs - ym)).sum() / denom) if denom > 1e-12 else 0.0
        spearman = _weighted_spearman(x, obs, cnt)
        # A calibrated curve must be non-decreasing across bins AND rank-aligned
        # with confidence.  0.7 (not 1.0) because bin frequencies are noisy at
        # the extremes; an inverted curve scores -1.0 and still fails.
        monotone = bool(np.all(np.diff(obs) >= -0.05) and spearman >= 0.7)
    else:
        slope, monotone = 0.0, True

    brier = float(np.mean((p - y) ** 2))
    ece = float(np.sum(cnt * np.abs(obs - np.asarray(centres, dtype=float))) / cnt.sum()) \
        if cnt.sum() else 0.0
    return {
        "bin_centres": centres, "observed": observed, "counts": counts,
        "monotone": monotone, "slope": slope, "spearman": spearman,
        "brier": brier, "ece": ece,
    }


# ── calibrator ───────────────────────────────────────────────────────────

class ProbabilityCalibrator:
    """Isotonic or Platt calibration for a binary ``P(positive)``.

    Usage::

        cal = ProbabilityCalibrator("isotonic").fit(p_val, y_val)
        p_cal = cal.transform(p_test)

    A calibrator fitted on fewer than 30 rows (or a single class) degrades to
    the identity, never to a silent 0.5 — callers must be able to tell the
    difference, hence :attr:`fitted`.
    """

    def __init__(self, method: str = "isotonic"):
        method = str(method or "isotonic").strip().lower()
        if method not in ("isotonic", "platt", "sigmoid", "none"):
            raise ValueError(f"unknown calibration method: {method!r}")
        self.method = "platt" if method == "sigmoid" else method
        self._iso = None
        self._coef: tuple[float, float] | None = None
        self.fitted = False
        self.n_fit = 0
        self.base_rate = 0.5

    # -- fitting ---------------------------------------------------------
    def fit(self, y_prob, y_true) -> "ProbabilityCalibrator":
        p = np.asarray(y_prob, dtype=float).ravel()
        y = np.asarray(y_true, dtype=float).ravel()
        mask = ~(np.isnan(p) | np.isnan(y))
        p, y = np.clip(p[mask], 1e-6, 1 - 1e-6), y[mask]
        self.n_fit = int(len(y))
        if self.n_fit:
            self.base_rate = float(y.mean())
        if self.method == "none" or self.n_fit < 30 or len(np.unique(y)) < 2:
            self.fitted = False
            return self

        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(p, y)
            self._iso = iso
        else:  # platt / sigmoid
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1), y)
            self._coef = (float(lr.coef_[0][0]), float(lr.intercept_[0]))
        self.fitted = True
        return self

    # -- inference -------------------------------------------------------
    def transform(self, y_prob) -> np.ndarray:
        p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
        if not self.fitted:
            return p
        if self._iso is not None:
            return np.clip(self._iso.predict(p), 0.0, 1.0)
        a, b = self._coef  # type: ignore[misc]
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return 1.0 / (1.0 + np.exp(-(a * logit + b)))

    def __call__(self, y_prob) -> np.ndarray:
        return self.transform(y_prob)

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "method": self.method, "fitted": self.fitted, "n_fit": self.n_fit,
            "base_rate": self.base_rate, "coef": self._coef,
            "iso_x": list(self._iso.X_thresholds_) if self._iso is not None else None,
            "iso_y": list(self._iso.y_thresholds_) if self._iso is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ProbabilityCalibrator":
        obj = cls(payload.get("method", "isotonic"))
        obj.fitted = bool(payload.get("fitted"))
        obj.n_fit = int(payload.get("n_fit", 0))
        obj.base_rate = float(payload.get("base_rate", 0.5) or 0.5)
        coef = payload.get("coef")
        obj._coef = (float(coef[0]), float(coef[1])) if coef else None
        xs, ys = payload.get("iso_x"), payload.get("iso_y")
        if xs and ys:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
            obj._iso = iso
            obj.fitted = True
        return obj


# ── the signed score (Phase P2 item 3) ───────────────────────────────────

def signed_score(
    p_up: float | np.ndarray,
    base_rate: float = 0.5,
    *,
    clip: float = SCORE_CLIP,
) -> float | np.ndarray:
    """Signed directional score centred on the model's own base rate.

    Definition (the exact formula the fusion kernel and the P2 docs state)::

        edge     = p_up / base_rate - 1
        scale    = max(1/base_rate - 1, 1/(1-base_rate) - 1)
        score    = clip(edge / scale, -clip, +clip)

    This is **not** the ``2·p_up/base_rate − 1`` that the first P2 revision
    documented (audit P2 #10): the implemented score is divided by the same
    ``scale`` on both sides, which (a) keeps it inside ``[-1, +1]`` and (b) is
    *asymmetric* — for ``base_rate = 0.2`` a maximal bullish ``p = 1.0`` scores
    ``+1.0`` while ``p = 0.0`` scores only ``−0.25``, because the bearish tail
    from 0.2 to 0 is a quarter of the bullish tail from 0.2 to 1.

    The scale factor keeps the score in [-1, +1]; a probability above the base
    rate always scores positive and one below always negative.

    Properties the audit demanded:

    * ``p_up == base_rate``  → ``score == 0`` (no information, no vote),
    * ``p_up >  base_rate``  → ``score > 0`` (bullish),
    * ``p_up <  base_rate``  → ``score < 0`` (bearish),
    * ``base_rate == 0.5``   → ``score == (p_up - 0.5) * 2``, i.e. exactly the
      historical ``(conf − 0.5) × 2``.  That is what makes the change backward
      compatible for callers that cannot be edited, while the *measured* base
      rate is what stops a bullish 0.35 being scored as bearish when only 30 %
      of the sample rises.

    A non-finite or out-of-range base rate falls back to 0.5.  Subnormal base
    rates are rejected too (audit P2 #10): ``base_rate = 5e-324`` divided into
    ``p_up`` overflows to ``inf`` and ``inf/inf`` is ``NaN``, which the fusion
    kernel would then push into the score — below ``1e-9`` the sample cannot
    support a base-rate-centred score, so 0.5 is used.
    """
    b = float(base_rate)
    if not np.isfinite(b) or b < 1e-9 or b > 1.0 - 1e-9:
        b = 0.5
    scale = max(1.0 / b - 1.0, 1.0 / (1.0 - b) - 1.0)
    arr = (np.asarray(p_up, dtype=float) / b - 1.0) / scale
    arr = np.clip(arr, -abs(float(clip)), abs(float(clip)))
    return float(arr) if np.isscalar(p_up) or arr.ndim == 0 else arr
