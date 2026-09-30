"""Cumulative trial counter for the Deflated Sharpe Ratio.

The DSR hurdle depends on ``n_trials`` — how many strategy configurations were
tried before the champion was selected.  Counting only the current run
(``population × generations``) understates the multiple-testing burden: the
production log holds ~24 walk-forward jobs, each running its own GA over several
windows, so thousands of configurations were examined in total.

This module owns that count:

* :func:`load_trials` / :func:`record_trials` read/write
  ``<data_dir>/data/ga_trials.json`` (best-effort — a missing file means 0).
* :func:`total_trials` returns ``prior + current`` for the DSR call.

The file is deliberately tiny and never raises: a GA run must not fail because
its trial ledger is unreadable.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from loguru import logger

_LOCK = threading.Lock()
_FILE_NAME = "ga_trials.json"


def _path(data_dir) -> Path:
    return Path(data_dir) / "data" / _FILE_NAME


def load_trials(data_dir) -> int:
    """Prior trial count stored on disk (0 when absent or unreadable)."""
    try:
        path = _path(data_dir)
        if not path.exists():
            return 0
        with open(path) as f:
            payload = json.load(f)
        return max(int(payload.get("trials", 0) or 0), 0)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"GA trial counter unreadable: {e}")
        return 0


def record_trials(data_dir, additional: int, last_window: str = "") -> int:
    """Add *additional* trials to the ledger; return the new total."""
    additional = max(int(additional or 0), 0)
    try:
        with _LOCK:
            path = _path(data_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {}
            if path.exists():
                try:
                    with open(path) as f:
                        payload = json.load(f) or {}
                except Exception:
                    payload = {}
            total = max(int(payload.get("trials", 0) or 0), 0) + additional
            payload["trials"] = total
            if last_window:
                payload["last_window"] = str(last_window)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            return total
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"GA trial counter not writable: {e}")
        return additional


def total_trials(data_dir, current: int = 0) -> int:
    """``prior + current`` — the value the DSR multiple-testing term needs."""
    return load_trials(data_dir) + max(int(current or 0), 0)
