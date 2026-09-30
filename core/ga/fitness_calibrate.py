"""Fitness weight loading for the GA.

The GA fitness formula reads a per-run weight set (``wr`` / ``pf`` / ``roc`` /
``bal``) so an operator can retune it without editing code.  This module owns
that *loading* path and the documented defaults it falls back to:

* :data:`DEFAULT_WEIGHTS` — the hand-tuned defaults and the only weight set the
  formula falls back to.
* :meth:`FitnessCalibrator.load_weights` — instance-level load from
  ``<data_dir>/data/ga_fitness_weights.json``.
* :meth:`FitnessCalibrator.load_weights_static` — the same load as a static
  helper, which is what :mod:`core.ga.evolver` calls before a run.
* :meth:`FitnessCalibrator._save_weights_temp` — write a weight set for a
  running GA process to pick up.

The calibration *search* itself (Spearman rank correlation over a grid +
walk-forward validation) was never wired to any route or caller and has been
removed; only the load/save contract above is live.  Its ``WEIGHT_GRID`` search
grid was left behind as dead data until the audit removed it — do not revive it
without a caller.
"""

import json
from pathlib import Path

from loguru import logger

# Default weights (hand-tuned, used when no calibration exists)
DEFAULT_WEIGHTS = {"wr": 0.15, "pf": 5.0, "roc": 50, "bal": 10.0}


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
