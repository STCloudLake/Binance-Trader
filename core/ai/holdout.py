"""P7-S4 — the **one-shot holdout counter**: a test window may be opened twice only
if a human says so, out loud.

WHY THIS MODULE EXISTS
----------------------
``P7_REGIME_PLAN.md`` §3 S4 requires the orchestrator to be **fixed on the training
window** and the out-of-sample window to be evaluated **once**: "任何'看过样本外再调'
的迭代都要在证据里如实记为过拟合" — every iteration that looked at the test window
before freezing the rules is overfitting and must be recorded as such.  Nothing in
the repository implemented that counter, so a measurement tool could silently
re-evaluate the same holdout after tuning on it and no artifact would show it.

THE CONTRACT
------------
* A **holdout key** is ``(holdout_id, window_start, window_end, timeframe)`` — the
  window identity that must not be reused.  ``holdout_id`` is a free label
  (``"p7-s4-oos-2026-02..06"`` by convention) so two different studies can hold
  separate windows without colliding.
* :func:`claim_holdout` appends an evaluation record to the registry and returns
  it.  The **second** claim of the same key raises :class:`HoldoutRefusal` unless
  ``allow_reuse=True`` is passed explicitly — a silent second look is impossible.
* With ``allow_reuse=True`` the claim is recorded as
  ``reuse=True`` with ``reuse_index`` ≥ 2, and :func:`holdout_status` reports
  ``reused=True`` so the evidence file can say "this window was opened twice".
* The registry is a plain JSON file (default ``data/p7_holdout.json``, already
  covered by ``.gitignore``'s ``data/`` rule) written atomically per claim, and it
  is **never** consulted by the live or backtest path: this module is a research
  bookkeeping object, not a gate.

A claim records the rules fingerprint and the git revision, so "the orchestrator
was fixed before the window was opened" is checkable after the fact rather than
asserted.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

__all__ = ["HoldoutRefusal", "DEFAULT_HOLDOUT_PATH", "holdout_key",
           "load_registry", "save_registry", "claim_holdout", "evaluations",
           "holdout_status", "current_revision", "describe_claims"]

ROOT = Path(__file__).resolve().parents[2]
#: ``data/`` is gitignored as a whole (``.gitignore:91``), so this file can never
#: be committed by accident — the counter is a local research ledger.
DEFAULT_HOLDOUT_PATH = ROOT / "data" / "p7_holdout.json"


class HoldoutRefusal(RuntimeError):
    """The same holdout window was already evaluated and reuse was not allowed."""

    def __init__(self, message: str, *, key: str, evaluations: int):
        super().__init__(message)
        self.key = str(key)
        self.evaluations = int(evaluations)


def holdout_key(holdout_id, window_start, window_end, timeframe=None) -> str:
    """The identity of a holdout window: ``id|start|end|timeframe`` (canonical).

    Deliberately includes the *window bounds*, not just the id: renaming a study
    cannot make a used window look fresh, and the same id with a different window
    is a different holdout.
    """
    return "|".join([str(holdout_id or "default").strip(),
                     str(window_start).strip(), str(window_end).strip(),
                     str(timeframe or "").strip()])


def load_registry(path=None) -> dict:
    """The registry dict (``{key: {...}}``); a missing/corrupt file reads as empty."""
    target = Path(path or DEFAULT_HOLDOUT_PATH)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    claims = payload.get("claims")
    return claims if isinstance(claims, dict) else {}


def save_registry(claims: dict, path=None) -> Path:
    """Atomically write the registry (tmp file in the same dir + ``os.replace``)."""
    target = Path(path or DEFAULT_HOLDOUT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1,
               "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "claims": claims or {}}
    handle, temp = tempfile.mkstemp(prefix=".p7_holdout_", suffix=".json",
                                    dir=str(target.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        os.replace(temp, target)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    return target


def evaluations(holdout_id, window_start, window_end, timeframe=None,
                path=None) -> int:
    """How many times this window has been evaluated (0 for a fresh window)."""
    key = holdout_key(holdout_id, window_start, window_end, timeframe)
    return len(((load_registry(path).get(key) or {}).get("evaluations") or []))


def holdout_status(holdout_id, window_start, window_end, timeframe=None,
                   path=None) -> dict:
    """A report-shaped view of the window's claims (never raises)."""
    key = holdout_key(holdout_id, window_start, window_end, timeframe)
    entry = load_registry(path).get(key) or {}
    rows = list(entry.get("evaluations") or [])
    return {"key": key, "evaluated": bool(rows), "evaluations": len(rows),
            "reused": len(rows) > 1,
            "first_at": rows[0].get("at") if rows else None,
            "last_at": rows[-1].get("at") if rows else None,
            "revisions": [row.get("revision") for row in rows],
            "rules_fingerprints": [row.get("rules_fingerprint") for row in rows]}


def current_revision() -> str | None:
    """Short HEAD sha of this checkout, or ``None`` outside a git work tree."""
    try:
        proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=str(ROOT),
                              timeout=15)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def claim_holdout(holdout_id, window_start, window_end, *,
                  timeframe=None, path=None, label: str = "",
                  rules_fingerprint: str | None = None,
                  allow_reuse: bool = False,
                  detail: dict | None = None) -> dict:
    """Claim the window for one evaluation; refuse a silent second one.

    Returns the appended record (``at``, ``revision``, ``rules_fingerprint``,
    ``label``, ``reuse``, ``reuse_index``, ``detail``).  Raises
    :class:`HoldoutRefusal` when the window already has a claim and
    ``allow_reuse`` is false — the message names the previous evaluations, the
    rules fingerprints and the revisions, so the operator can see *what* was
    already spent on this window.
    """
    key = holdout_key(holdout_id, window_start, window_end, timeframe)
    claims = load_registry(path)
    entry = claims.get(key) or {}
    rows = list(entry.get("evaluations") or [])
    if rows and not allow_reuse:
        previous = rows[-1]
        raise HoldoutRefusal(
            f"holdout window '{key}' was already evaluated "
            f"{len(rows)} time(s) — refusing a second look. That window is now "
            f"in-sample; tune on the training window instead, or re-run with "
            f"allow_reuse=True (the reuse is recorded as overfitting in the "
            f"artifact). Previous: at={previous.get('at')} "
            f"rules_fingerprint={previous.get('rules_fingerprint')} "
            f"revision={previous.get('revision')} label={previous.get('label')}",
            key=key, evaluations=len(rows))
    record = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "revision": current_revision(),
        "rules_fingerprint": (None if rules_fingerprint is None
                              else str(rules_fingerprint)),
        "label": str(label or ""),
        "reuse": bool(rows),
        "reuse_index": len(rows) + 1,
        "window": {"holdout_id": str(holdout_id or "default"),
                   "start": str(window_start), "end": str(window_end),
                   "timeframe": (None if timeframe is None else str(timeframe))},
        "detail": dict(detail or {}),
    }
    rows.append(record)
    claims[key] = {"evaluations": rows}
    save_registry(claims, path)
    return record


def describe_claims(path=None) -> list:
    """``[{key, evaluations, reused, revisions, rules_fingerprints}]`` — the ledger."""
    claims = load_registry(path)
    out = []
    for key, entry in sorted(claims.items()):
        rows = list((entry or {}).get("evaluations") or [])
        out.append({"key": key, "evaluations": len(rows), "reused": len(rows) > 1,
                    "first_at": rows[0].get("at") if rows else None,
                    "last_at": rows[-1].get("at") if rows else None,
                    "rules_fingerprints": [row.get("rules_fingerprint")
                                           for row in rows],
                    "revisions": [row.get("revision") for row in rows]})
    return out
