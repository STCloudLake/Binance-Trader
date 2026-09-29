"""Fitness weight loading for the GA.

The GA fitness formula reads a per-run weight set (``wr`` / ``pf`` / ``roc`` /
``bal``) so an operator can retune it without editing code.  This module owns
that *loading* path and the documented defaults it falls back to:

* :data:`DEFAULT_WEIGHTS` / :data:`WEIGHT_GRID` — the hand-tuned defaults and the
  search grid they were picked from.
* :meth:`FitnessCalibrator.load_weights` — instance-level load from
  ``<data_dir>/data/ga_fitness_weights.json``.
* :meth:`FitnessCalibrator.load_weights_static` — the same load as a static
  helper, which is what :mod:`core.ga.evolver` calls before a run.
* :meth:`FitnessCalibrator._save_weights_temp` — write a weight set for a
  running GA process to pick up.

The calibration *search* itself (Spearman rank correlation over a grid +
walk-forward validation) was never wired to any route or caller and has been
removed; only the load/save contract above is live.
"""

import json
from pathlib import Path

from loguru import logger

# Default weights (hand-tuned, used when no calibration exists)
DEFAULT_WEIGHTS = {"wr": 0.15, "pf": 5.0, "roc": 50, "bal": 10.0}

# Search grid for weight calibration
WEIGHT_GRID = {
    "wr":  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "pf":  [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0],
    "roc": [10, 20, 30, 40, 50, 60, 80, 100],
    "bal": [2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0],
}


class FitnessCalibrator:
    """Loads (and temporarily persists) the GA fitness weight set."""

    def __init__(self, engine, loader, data_dir: str):
        self.engine = engine
        self.loader = loader
        self.data_dir = Path(data_dir)
        self._save_path = self.data_dir / "data" / "ga_fitness_weights.json"

    def load_weights(self) -> dict:
        """Load calibrated weights, falling back to defaults."""
        try:
            if self._save_path.exists():
                with open(self._save_path) as f:
                    data = json.load(f)
                return data.get("weights", DEFAULT_WEIGHTS)
        except Exception:
            pass
        return dict(DEFAULT_WEIGHTS)

    def _save_weights_temp(self, weights: dict):
        """Temporarily save weights for use by running GA processes."""
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._save_path, "w") as f:
            json.dump({"weights": weights, "temp": True}, f)

    @staticmethod
    def load_weights_static(data_dir: str) -> dict:
        """Static helper: load weights from disk, return defaults if not found."""
        path = Path(data_dir) / "data" / "ga_fitness_weights.json"
        try:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                return data.get("weights", DEFAULT_WEIGHTS)
        except Exception:
            pass
        return dict(DEFAULT_WEIGHTS)
