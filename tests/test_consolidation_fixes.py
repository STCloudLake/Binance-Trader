"""Consolidation guards: the docs must stay reproducible and the suite must not
depend on a defect in the shipped data cache.

Why this file exists (two defects the consolidation pass had to fix):

1. **Doc 10/11 numbers that no longer reproduce.**  The BTCUSDT 1h cache was
   repaired by the data vendor during the audit (the 1 484 h calendar gap is
   gone, ``max|log return| = 0.0494``), so the old "9.8x unclipped/clipped EWMA"
   figure stopped describing the shipped file at all, and the GARCH cost was
   quoted as the *grid fallback's* 2.2 ms while the shipped path is the free-omega
   MLE.  Docs 10 and 11 now quote measured figures and mark the historical /
   synthetic / window-dependent ones as such.  These tests recompute the headline
   figures from the recipe the docs print, so a doc that drifts from the code
   fails here instead of silently misleading a reader.

2. **An order-dependent failure that was really a data-content failure.**
   ``test_ewma_is_causal_and_clips_data_splice_outliers`` used to assert
   ``r.max() > 0.2`` on ``data/market/BTCUSDT/1h.parquet``: it *required* the
   shipped cache to contain a data defect.  It passed in isolation and failed in
   a full run only because the cache was rewritten (gap-repaired) between the two.
   The fix injects the splice synthetically; the tripwire below keeps the
   shipped-data version from coming back, and the write-target scan keeps any
   test from rewriting the live cache (tests that do write use ``tmp_path`` /
   ``mkdtemp``).

All random draws go through ``np.random.default_rng(SEED)`` and no test sleeps.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SEED = 20250930
ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
BTC_1H = ROOT / "data/market/BTCUSDT/1h.parquet"
DOC_TEN = ROOT / "docs/core-algorithms/10-volatility-targeting.md"
DOC_ELEVEN = ROOT / "docs/core-algorithms/11-pairs-cointegration.md"


def _doc(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _btc_returns():
    if not BTC_1H.exists():
        pytest.skip("no cached BTCUSDT 1h parquet in this checkout")
    from core.ml.volatility import log_returns

    return log_returns(pd.read_parquet(BTC_1H)["close"].values)


# ── 1. doc 10: the injected-splice figure and the repaired cache ─────────

def test_doc_ten_injected_splice_figure_is_the_measured_one():
    """Doc 10's clip-effectiveness evidence must be its own recipe's output.

    The splice is injected (``+0.276`` on a 0.4 %/bar synthetic series), so the
    ratio is a property of that injection, not of the shipped cache — and the
    doc must quote the numbers the recipe below returns.
    """
    from core.ml.volatility import ewma_vol, to_pct

    doc = _doc(DOC_TEN)
    rng = np.random.default_rng(SEED)
    calm = rng.normal(0.0, 0.004, 2000)
    splice = np.concatenate([calm[:1999], np.array([0.276])])
    unclipped = to_pct(ewma_vol(splice, window=0, outlier_sigma=0.0))
    clipped = to_pct(ewma_vol(splice, window=0))
    ratio = unclipped / clipped
    assert 5.0 < ratio < 20.0                     # the injection really dominates
    assert f"{clipped:.4f}" in doc, f"doc 10 must quote {clipped:.4f} %/bar"
    assert f"{ratio:.2f}" in doc, f"doc 10 must quote the ratio {ratio:.2f}x"
    assert "合成" in doc or "注入" in doc, "the figure must be labelled injected"
    # ...and it must say so in the same breath as the historical real-cache 9.8x.
    assert "9.8" in doc and ("历史" in doc or "不可" in doc)


def test_doc_ten_says_the_shipped_cache_splice_is_repaired():
    """The shipped cache no longer carries the defect, so the clip is a no-op."""
    from core.ml.volatility import DEFAULT_WINDOW, ewma_vol

    doc = _doc(DOC_TEN)
    r = _btc_returns()
    tail = r[-DEFAULT_WINDOW:]
    clipped = ewma_vol(tail, window=0)
    unclipped = ewma_vol(tail, window=0, outlier_sigma=0.0)
    if np.abs(tail).max() > 0.1:                  # a splice is back in the window
        assert unclipped / clipped > 5.0
        assert "接缝" in doc
        return
    # Current state: the repair holds, so unclipped == clipped and the doc says so.
    assert unclipped == pytest.approx(clipped, rel=1e-4)
    assert "修复" in doc
    assert "0.0494" in doc                        # the measured max |log return|


def test_doc_ten_garch_cost_is_not_the_grid_fallback_number():
    """Doc 10 must quote the MLE cost, and must not present 2.2 ms as its own."""
    doc = _doc(DOC_TEN)
    assert "自由 ω 的 MLE" in doc
    assert "IGARCH" in doc and "0/0" in doc
    # The 2.2 ms figure survives only as the optimiser-free grid fallback.
    assert "2.2 ms" in doc and "网格" in doc
    # The measured MLE costs are named (default window and full history).
    assert "0.24–0.49" in doc or "0.24-0.49" in doc
    assert "6.2–6.4" in doc or "6.2-6.4" in doc
    assert "opt-in" in doc
    # And the refuted corner means are the re-measured ones, not 36 / 7.2e5.
    assert "91.19" in doc and "2 004.18" in doc
    assert "7.2e5" not in doc


# ── 2. doc 11: the causal HMM's honest numbers ───────────────────────────

def _synth(n: int = 3000, *, seed: int = 5, change_points=(1000, 2000)):
    """The synthetic three-regime structure the docs quote (same as test_p34)."""
    rng = np.random.default_rng(seed)
    spec = [(0.002, +0.0004), (0.010, -0.0008), (0.002, +0.0004)]
    bounds = (0, change_points[0], change_points[1], n)
    rets, truth = [], []
    for k, (sigma, drift) in enumerate(spec):
        m = bounds[k + 1] - bounds[k]
        rets.append(rng.standard_normal(m) * sigma + drift)
        truth += ["high" if k == 1 else "low"] * m
    return np.concatenate(rets), np.asarray(truth), tuple(change_points)


def test_doc_eleven_causal_accuracy_is_reproducible_and_labelled():
    """In-sample ~0.999 and the causal 0.758 / 0.815 must both be in doc 11."""
    from core.strategy.regime import (detection_metrics, hmm_two_state,
                                      hmm_two_state_causal)

    doc = _doc(DOC_ELEVEN)
    assert "in-sample" in doc.lower() or "样本内" in doc
    assert "样本外" in doc or "out-of-sample" in doc.lower()
    assert "0.156" in doc                     # the pre-index-fix decode
    assert "修正前" in doc or "索引" in doc

    for seed, low, high in ((5, 0.70, 0.80), (7, 0.78, 0.85)):
        r, truth, cp = _synth(seed=seed)
        fit = hmm_two_state_causal(r)
        pred = np.where(fit["states"] == 1, "high", "low")
        acc = detection_metrics(truth[1:], pred, positive="high",
                                change_points=(cp[0] - 1, cp[1] - 1))["accuracy"]
        assert low < acc < high, (seed, acc)
        assert f"{acc:.3f}" in doc, f"doc 11 must quote the causal {acc:.3f}"

    r, truth, cp = _synth(seed=5)
    ins = detection_metrics(
        truth[1:], np.where(hmm_two_state(r)["states"] == 1, "high", "low"),
        positive="high", change_points=(cp[0] - 1, cp[1] - 1))["accuracy"]
    assert ins > 0.95
    assert f"{ins:.4f}" in doc, f"doc 11 must label the in-sample {ins:.4f}"


# ── 3. the data-dependence tripwires ─────────────────────────────────────

def test_no_test_asserts_a_defect_in_the_shipped_cache():
    """The splice must be injected synthetically, never read out of data/market.

    The failure this pins: a test that needs the shipped cache to contain a
    calendar-gap splice stops guarding anything (and fails) the moment the cache
    is repaired or rewritten — the "passes alone, fails in a full run" symptom.
    """
    src = (TESTS / "test_volatility_targeting.py").read_text(encoding="utf-8")
    assert "splice.max() > 0.2" in src           # the injected series
    assert "r.max() > 0.2" not in src            # the removed shipped-data assert
    assert "splice = np.concatenate" in src
    # The p34 file's doc test must keep the shipped-cache branch conditional.
    src34 = (TESTS / "test_p34_audit_fixes.py").read_text(encoding="utf-8")
    assert "if np.abs(rr).max() > 0.1:" in src34
    # And both may *read* the live cache, but only through the skip-guarded helper.
    for path in (TESTS / "test_volatility_targeting.py",
                 TESTS / "test_p34_audit_fixes.py"):
        text = path.read_text(encoding="utf-8")
        assert "pytest.skip(" in text and "BTCUSDT/1h.parquet" in text


def test_no_test_writes_the_live_market_cache():
    """Every parquet write in the suite must target a temporary directory.

    A test that writes ``data/market/...`` would rewrite the same file the
    volatility tests read, which is exactly the order/time dependence this pass
    removed.  (The suite's real writers build their paths from ``tmp_path`` /
    ``mkdtemp``, so no line pairs the live path with a write call.)
    """
    write = re.compile(r"to_parquet\(|write_bytes\(|\.unlink\(|mkdir\(")
    offenders: list[str] = []
    for path in sorted(TESTS.glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "data/market" in line and write.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert offenders == [], f"tests write the live market cache: {offenders}"
