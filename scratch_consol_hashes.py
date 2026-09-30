"""Consolidation fixer scratch: print sha256_16 + mtime for the files relied on."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = [
    "core/ml/volatility.py",
    "core/strategy/regime.py",
    "core/strategy/pairs.py",
    "core/backtest/signal_matrix.py",
    "core/backtest/engine.py",
    "core/backtest/engine_hybrid.py",
    "core/ml/labels.py",
    "tests/test_p34_audit_fixes.py",
    "tests/test_volatility_targeting.py",
    "tests/test_consolidation_fixes.py",
    "tests/test_hybrid_condition_logic.py",
    "docs/core-algorithms/10-volatility-targeting.md",
    "docs/core-algorithms/11-pairs-cointegration.md",
]
for rel in FILES:
    path = ROOT / rel
    if not path.exists():
        print(f"{rel}: MISSING")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    mtime = path.stat().st_mtime
    print(f"{digest}  {path.stat().st_size:>7}  {rel}")
