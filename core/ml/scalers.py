"""Train-time feature scaler (Phase P2 item 8 — train/serve skew).

The sequence trainers used to normalize training windows with *expanding-window*
statistics (per window, using all data up to that window) and then normalize
inference with the *global* mean/std of the supplied history.  Those two
transforms disagree, so the deployed network saw inputs from a different
distribution than it was trained on.

:class:`TrainTimeScaler` is fitted once on the training block and reused
verbatim at inference; it is persisted next to the checkpoint so a freshly
loaded model cannot silently fall back to a different normalization.
"""

from __future__ import annotations

import numpy as np


class TrainTimeScaler:
    """Per-feature ``(x − mean) / std`` with a persisted, fixed statistic."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    # -- fitting ---------------------------------------------------------
    @classmethod
    def fit(cls, data: np.ndarray) -> "TrainTimeScaler":
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        return cls(mean, std)

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    # -- inference -------------------------------------------------------
    def transform(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        if not self.fitted:
            return arr.astype(np.float32)
        out = (arr - self.mean) / self.std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mean": None if self.mean is None else [float(v) for v in self.mean],
            "std": None if self.std is None else [float(v) for v in self.std],
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "TrainTimeScaler":
        if not payload:
            return cls()
        return cls(payload.get("mean"), payload.get("std"))
