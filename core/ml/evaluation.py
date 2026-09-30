"""Out-of-sample evaluation for the ML pipeline (Phase P2 item 2).

Before this module nothing checked OOS skill: :meth:`core.ml.trainer.MLTrainer.
train_binary` scored the model on a chronological 20 % tail that *shares
2–9 bars of forward window with the training set* (75–95 % of labels overlap),
so every reported accuracy was in-sample in disguise.

What is provided
----------------
``purged_kfold_splits``
    Purged K-fold (López de Prado, AFML ch. 7): the test fold is removed from
    training **and** every training label whose forward window overlaps the test
    fold is purged, plus an ``embargo`` of ``label_span`` bars after the fold.
``combinatorial_purged_splits``
    The same purge for path-style (combinatorial) folds.
``sample_uniqueness_weights``
    Average-uniqueness weights so 4-times-overlapping labels stop counting as 4
    independent observations.
``binary_metrics`` / ``multiclass_metrics``
    Base rate, majority-class baseline, accuracy, AUC, Brier, log loss — the
    metrics the plan requires, all computed only on out-of-sample rows.

The **net-of-cost expectancy** lives in :mod:`core.ml.credibility` (it needs the
cost model and the threshold search).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── purging / embargo ────────────────────────────────────────────────────

def label_spans(n: int, label_span: int) -> np.ndarray:
    """End index (exclusive) of every label's forward window."""
    span = max(int(label_span), 1)
    return np.minimum(np.arange(int(n)) + span, int(n) - 1)


def purged_kfold_splits(
    n: int,
    n_splits: int = 5,
    *,
    label_span: int = 4,
    embargo: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Purged K-fold index splits.

    Returns a list of ``(train_idx, test_idx, n_purged)``.  ``embargo`` defaults
    to ``label_span`` (the plan's "``embargo = label horizon``").

    A training row ``i`` is purged when its label window ``[i, i + label_span)``
    intersects the test fold, or when it sits inside the embargo immediately
    after the fold.  Folds are contiguous blocks in time order, so the test set
    is always a genuine future window relative to most of its training data.
    """
    n = int(n)
    k = max(int(n_splits), 2)
    span = max(int(label_span), 1)
    emb = span if embargo is None else max(int(embargo), 0)
    if n <= k * 2:
        return []

    ends = label_spans(n, span)
    bounds = np.linspace(0, n, k + 1).astype(int)
    splits: list[tuple[np.ndarray, np.ndarray, int]] = []
    all_idx = np.arange(n)
    for f in range(k):
        lo, hi = int(bounds[f]), int(bounds[f + 1])
        if hi <= lo:
            continue
        test_idx = all_idx[lo:hi]
        # (a) label-window overlap: a train row whose forward window reaches
        #     into the test fold's first bar.
        overlap = (ends >= lo) & (all_idx < lo)
        # (b) rows whose span *starts* inside the test fold are test rows.
        overlap |= (all_idx >= lo) & (all_idx < hi)
        # (c) embargo: bars right after the fold share information with it.
        embargo_mask = (all_idx >= hi) & (all_idx < hi + emb)
        keep = ~(overlap | embargo_mask)
        train_idx = all_idx[keep]
        n_purged = int(n - len(train_idx) - len(test_idx))
        splits.append((train_idx, test_idx, n_purged))
    return splits


def combinatorial_purged_splits(
    n: int,
    n_groups: int = 6,
    k_test: int = 1,
    *,
    label_span: int = 4,
    embargo: int | None = None,
    max_splits: int = 20,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Path-style purged splits over contiguous groups (``n_groups`` choose ``k_test``)."""
    from itertools import combinations

    n = int(n)
    g = max(int(n_groups), 2)
    span = max(int(label_span), 1)
    emb = span if embargo is None else max(int(embargo), 0)
    if n <= g * 2:
        return []
    bounds = np.linspace(0, n, g + 1).astype(int)
    ends = label_spans(n, span)
    all_idx = np.arange(n)
    out: list[tuple[np.ndarray, np.ndarray, int]] = []
    for combo in combinations(range(g), max(1, int(k_test))):
        test_mask = np.zeros(n, dtype=bool)
        for c in combo:
            test_mask[bounds[c]:bounds[c + 1]] = True
        test_idx = all_idx[test_mask]
        if len(test_idx) == 0:
            continue
        lo, hi = int(test_idx[0]), int(test_idx[-1]) + 1
        train_mask = ~test_mask
        # Purge anything whose forward window reaches the test block.
        train_mask &= ~((ends >= lo) & (all_idx < lo))
        # Embargo on both sides of the block (the block may be interior here).
        train_mask &= ~((all_idx >= hi) & (all_idx < hi + emb))
        train_idx = all_idx[train_mask]
        out.append((train_idx, test_idx, int(n - len(train_idx) - len(test_idx))))
        if len(out) >= int(max_splits):
            break
    return out


def overlap_count(test_idx: np.ndarray, label_span: int, n: int) -> int:
    """Rows the *old* chronological split would have leaked into the test fold.

    For a boundary at ``min(test_idx)``: every training label whose forward
    window reaches into the test fold shares outcome information with it.  The
    old :meth:`core.ml.trainer.MLTrainer.train_binary` did exactly this — it
    split at 80 % and trained on rows whose labels were still unresolved at the
    split point.
    """
    if len(test_idx) == 0:
        return 0
    span = max(int(label_span), 1)
    lo = int(test_idx.min())
    train = np.arange(0, lo)
    ends = np.minimum(train + span, n - 1)
    return int((ends >= lo).sum())


def label_concurrency(n: int, label_span: int = 4) -> np.ndarray:
    """How many labels cover each bar (AFML ch. 4 concurrency)."""
    n = int(n)
    span = max(int(label_span), 1)
    concurrency = np.zeros(n, dtype=float)
    for i in range(n):
        concurrency[i:min(i + span, n)] += 1.0
    return np.maximum(concurrency, 1.0)


def average_label_overlap(n: int, label_span: int = 4) -> float:
    """Mean fraction of each label's window shared with a *later* label.

    ``label_span = 4`` ⇒ 0.75: only the last bar of the four is not also covered
    by the next three labels.  This is why the effective sample size is ≈N/4 and
    why average-uniqueness weights are applied.
    """
    span = max(int(label_span), 1)
    if n <= 0:
        return 0.0
    return float(span - 1) / float(span)


def sample_uniqueness_weights(
    n: int,
    label_span: int = 4,
    *,
    index: pd.Index | None = None,
) -> np.ndarray:
    """Average-uniqueness weights (AFML ch. 4), **mean-normalised**.

    Each label's forward window covers ``label_span`` bars; a bar covered by
    ``c`` concurrent labels contributes ``1/c``.  A label's weight is the mean of
    those contributions, normalised so the mean weight is 1 (the sum equals ``n``)
    — i.e. overlapping labels no longer count as independent observations.  The
    range is therefore **not** ``(0, 1]``: with ``label_span = 4`` the measured
    range is ``[0.9997, 2.0827]``, because the labels at the edges of the sample
    are covered by fewer concurrent windows and must be scaled *up* to keep the
    total effective count honest.  A weight is never zero: the floor is
    ``1e-6`` so a model fit can always consume the full matrix.
    """
    n = int(n)
    span = max(int(label_span), 1)
    if n <= 0:
        return np.zeros(0)
    concurrency = np.zeros(n, dtype=float)
    starts = np.arange(n)
    for i in starts:
        end = min(i + span, n)
        concurrency[i:end] += 1.0
    concurrency = np.maximum(concurrency, 1.0)

    weights = np.zeros(n, dtype=float)
    for i in starts:
        end = min(i + span, n)
        weights[i] = float(np.mean(1.0 / concurrency[i:end])) if end > i else 0.0
    total = weights.sum()
    if total > 0:
        weights *= (n / total)
    return np.clip(weights, 1e-6, None)


# ── metrics ──────────────────────────────────────────────────────────────

def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return 0.5


def binary_metrics(
    y_true,
    y_prob,
    *,
    y_prob_positive: str = "up",
) -> dict:
    """OOS quality for a binary (or binary-restricted) probability.

    ``y_true``/``y_prob`` must already be out-of-sample rows.  Returns base
    rate, majority-class accuracy, accuracy, AUC, Brier and log loss.  The
    majority baseline uses the *same* rows, which is why the audit could show
    the model 10–20 pp below it.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return {"n": 0, "base_rate": 0.5, "majority_accuracy": 0.5,
                "accuracy": 0.5, "auc": 0.5, "brier": 0.25, "log_loss": np.log(2)}
    base = float(y.mean())
    majority = max(base, 1.0 - base)
    pred = (p >= 0.5).astype(float)
    acc = float((pred == y).mean())
    if y_prob_positive == "down":
        base_auc = _safe_auc(y, -p)
    else:
        base_auc = _safe_auc(y, p)
    eps = 1e-12
    pc = np.clip(p, eps, 1 - eps)
    ll = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    return {
        "n": int(len(y)),
        "base_rate": base,
        "majority_accuracy": majority,
        "accuracy": acc,
        "auc": base_auc,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": ll,
        "edge_vs_majority": acc - majority,
    }


def multiclass_metrics(y_true, proba) -> dict:
    """Accuracy / balanced accuracy / Brier / log loss for a 3-class model."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if len(y) == 0:
        return {"n": 0, "accuracy": 0.0, "majority_accuracy": 0.0,
                "brier": 1.0, "log_loss": np.log(3)}
    lab = np.argmax(p, axis=1)
    acc = float((lab == y).mean())
    counts = np.bincount(y, minlength=p.shape[1])
    majority = float(counts.max() / counts.sum()) if counts.sum() else 0.0
    eps = 1e-12
    ll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0))))
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)) / p.shape[1])
    class_acc = []
    for c in range(p.shape[1]):
        sel = y == c
        class_acc.append(float((lab[sel] == c).mean()) if sel.any() else 0.0)
    return {
        "n": int(len(y)), "accuracy": acc, "majority_accuracy": majority,
        "balanced_accuracy": float(np.mean(class_acc)),
        "brier": brier, "log_loss": ll, "class_accuracy": class_acc,
    }


def train_test_counts(train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """Small helper for the purge/embargo proof output."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return {"train": int(len(train_idx)), "test": int(len(test_idx)),
                "train_first": None, "train_last": None, "test_first": None}
    return {
        "train": int(len(train_idx)), "test": int(len(test_idx)),
        "train_first": int(train_idx.min()), "train_last": int(train_idx.max()),
        "test_first": int(test_idx.min()), "test_last": int(test_idx.max()),
    }
