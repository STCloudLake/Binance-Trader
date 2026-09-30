"""Policy guard: tests may not assert a measured live-cache statistic.

Why this file exists
--------------------
``tests/test_meta_labeling.py`` used to assert ``res["metrics"]["auc"] <= 0.55``
on a meta evaluation over ``data/market/BTCUSDT/1h.parquet``.  That number was a
*measurement* of one cache state, not a property of the code under test: after
the cache was repaired the same, still-correct behaviour measured AUC 0.5866 and
the test failed while the behaviour it exists to prove -- the meta gate refusing
the candidate -- was unchanged.  A test suite must not encode the contents of a
mutable data directory.

The policy
----------
1. A test may assert a number that depends on ``data/market/**`` only if it pins
   the data (a frame built inside the test) or derives the expectation from the
   same response/frame it just read.
2. Otherwise it asserts the *behavioural contract*: the decision, the reason
   string, and the gate fields the decision was computed from.
3. Every test that reads the live cache must be **decision-complete**: at least
   one of its assertions compares two run-time expressions (decision field vs a
   threshold/field read from the same response), not a literal.

What this guard checks (no cache access, no network)
----------------------------------------------------
* The set of test files that derive a ``data/market`` path into a pandas reader
  is asserted **equal** to :data:`LIVE_CACHE_TEST_FILES`.  A new reader fails
  the guard until it is added there and made policy-compliant.
* For each enforced module, every float literal in an assert must either be
  justified in :data:`FILE_LITERAL_ALLOW` (the reason it cannot depend on the
  cache) or be equal to one of the imported backend constants in
  :data:`BACKEND_CONSTANTS`, or appear as an argument the test itself builds.
  A measured pin such as ``auc <= 0.5866`` matches none of those and fails.
* For every reader outside the enforced set, a *tripwire* flags only asserts
  whose left-hand side came from a read (a frame/response value) compared
  directly against an unjustified bare literal.
* A test that contains ``pytest.skip`` is exempt only when the skip is
  **load-bearing**: the skip dominates the body, so no assert of that test can
  run while the cache is present.  A guard clause such as
  ``if volume <= 0: pytest.skip(...)`` does not exempt the test — the live pins
  that follow it are still asserted whenever the cache *is* there.  Evaluated by
  control flow (``_skip_dominates_the_body``), not by the mere presence of the
  call, which is what the previous ``re.search(r"pytest\\.skip\\s*\\(")``
  exemption did: one skip anywhere bought a whole module a pass.
* The guard self-tests the scanner on constructed violating and compliant
  modules, so the checker cannot silently stop checking.

Honest limits
-------------
The scan is syntactic.  It cannot know whether ``x["auc"]`` came from the cache
or from a synthetic frame, so a *derived-looking* comparison is not proof of
soundness, and a violation written without a numeric literal (two measured
quantities compared to each other) is invisible to it.  It is a tripwire against
the specific, repeated failure mode -- a hard-coded measured statistic -- not a
proof of statistical hygiene.  It never reads ``data/market``, so it passes on
every cache state, including a repaired one.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

#: Test modules allowed to read ``data/market/**``.  The guard asserts this is
#: exactly the set it detects, so adding a reader is a deliberate act.
LIVE_CACHE_TEST_FILES = {
    "test_cache_durability.py",
    "test_consolidation_fixes.py",
    "test_download_dedupe.py",
    "test_gap_fixes.py",
    "test_liquidity.py",
    "test_meta_labeling.py",
    "test_ml_credibility.py",
    "test_p34_audit_fixes.py",
    "test_p34_code_defects.py",
    "test_pairs.py",
    "test_residual_closure.py",
    "test_volatility_targeting.py",
}

#: Files with full literal-level enforcement (every assert literal justified).
ENFORCED_LITERAL_FILES = {"test_meta_labeling.py", "test_pairs.py"}

#: Files whose live-cache tests must additionally *derive* an expectation and
#: name a decision field.  The other readers (mutation/doc contract tests) are
#: held to the objective literal rules only; their decision-completeness cannot
#: be judged syntactically without false positives.
ENFORCED_DECISION_FILES = {"test_meta_labeling.py", "test_pairs.py"}

#: Backend constants an assert literal may legitimately equal: importing these
#: is the documented way to say "this bound is the implementation's, not the
#: cache's".  Values are compared with a relative tolerance.
BACKEND_CONSTANTS: dict[float, str] = {}


def _const(value: float, source: str) -> None:
    BACKEND_CONSTANTS[float(value)] = source


from core.ml.credibility import (  # noqa: E402  (import after the doc block)
    GATE_AUC_MIN, GATE_MIN_NET_EXPECTANCY, GATE_MIN_TRADES,
)
from core.ml.meta import META_SIZE_FLOOR  # noqa: E402
from core.strategy.pairs import (  # noqa: E402
    PAIRS_MAX_LOOKBACK, PAIRS_MIN_LOOKBACK, PAIRS_Z_EXIT, PAIRS_Z_ENTRY,
)

_const(GATE_AUC_MIN, "core.ml.credibility.GATE_AUC_MIN")
_const(GATE_MIN_NET_EXPECTANCY, "core.ml.credibility.GATE_MIN_NET_EXPECTANCY")
_const(float(GATE_MIN_TRADES), "core.ml.credibility.GATE_MIN_TRADES")
_const(META_SIZE_FLOOR, "core.ml.meta.META_SIZE_FLOOR")
_const(PAIRS_Z_ENTRY, "core.strategy.pairs.PAIRS_Z_ENTRY")
_const(PAIRS_Z_EXIT, "core.strategy.pairs.PAIRS_Z_EXIT")
_const(float(PAIRS_MIN_LOOKBACK), "core.strategy.pairs.PAIRS_MIN_LOOKBACK")
_const(float(PAIRS_MAX_LOOKBACK), "core.strategy.pairs.PAIRS_MAX_LOOKBACK")

#: Values used across many modules for which no single constant exists: range
#: ends, degenerate inputs, named backend defaults.  Each is a property of the
#: code, not of the cache -- none of them is a measured market statistic.
_SHARED: dict[float, str] = {
    0.0: "empty / degenerate expectation",
    1.0: "unit or probability upper bound",
    0.5: "probability midpoint / symmetric expectation",
    0.25: "config default or quarter-grid point",
    0.75: "three-quarter parameterisation",
    0.2: "fraction bound in a documented parameterisation",
    0.4: "config parameterisation (impact_k default)",
    0.45: "volatility parameterisation in a synthetic frame",
    0.6: "probability grid point",
    0.9: "probability grid point",
    0.1: "significance level or fraction bound",
    0.05: "5 % significance level",
    0.02: "2 % bound / floor parameter",
    0.01: "1 % level or 1 % fee/spread unit",
    0.99: "percentile bound (documented contract)",
    0.001: "lower bound on a barrier width (documented)",
    0.65: "documented accuracy floor in a synthetic experiment",
    0.8: "documented rank-correlation floor in a synthetic experiment",
    0.95: "GATE_MIN_PSR / accuracy sanity bound",
    0.35: "parameterisation fraction",
    0.14: "documented default leg cost (%)",
    0.04: "documented impact parameter",
    0.13003: "documented ETH cost example",
    0.12503: "documented BTC cost example",
    0.0051: "documented fee constant",
    1e-4: "numeric tolerance",
    1e-6: "numeric tolerance",
    1e-9: "numeric tolerance",
    1e-12: "numeric tolerance",
}

#: Per-file literals that are not shared: each is an input the test
#: parameterises, or a bound the module under test documents.  The reason is
#: mandatory; a literal with no entry is reported as a measured pin.
_PER_FILE: dict[str, dict[float, str]] = {
    "test_meta_labeling.py": {
        0.21: "closed-form 100 -> 121 forward return on a synthetic series",
        0.50125: "closed-form breakeven hit rate for 1:1 R and 0.25 % cost",
        1.0025: "closed-form breakeven numerator (1 + cost/100)",
        3.0: "R:R ratio in a closed-form breakeven",
    },
    "test_pairs.py": {
        0.10: "twice the 5 % level: the documented 'not near the boundary' margin",
        0.30: "Monte-Carlo tolerance on a quantile of 300 draws",
        1.57: "MacKinnon mean tau reference (absorbs simulation noise)",
        0.35: "documented beta drift in the synthetic path",
        0.06: "tolerance around the terminal synthetic beta",
        8760.0: "hours per year for the 1h interval",
        365.0: "days per year for the 1d interval",
    },
}

#: Files whose literal check uses the shared table only.
_SHARED_ONLY = {
    "test_cache_durability.py",
    "test_consolidation_fixes.py",
    "test_download_dedupe.py",
    "test_gap_fixes.py",
    "test_liquidity.py",
    "test_ml_credibility.py",
    "test_p34_audit_fixes.py",
    "test_p34_code_defects.py",
    "test_volatility_targeting.py",
}

#: Shared-table additions that only appear in the shared-only modules.
_SHARED.update({
    1.25: "documented trailing-stop multiple",
    8.0: "documented position_size_pct default",
    10.0: "sample / window size parameter",
    20.0: "sample / lookback parameter",
    100.0: "unit scale in a cost/size formula",
    789.0: "synthetic frame value",
    1e4: "volume scale bound",
    2.2: "documented latency figure (ms/bar)",
    9.8: "documented clip ratio",
    365.0: "days per year",
    8760.0: "hours per year",
    0.2763: "documented measured figure quoted in a doc test",
    0.0494: "documented measured figure quoted in a doc test",
    9.49: "documented measured figure quoted in a doc test",
    0.156: "documented measured figure quoted in a doc test",
    7.2e5: "documented row count quoted in a doc test",
    1.0e6: "corner-case bound",
})

_NUM = re.compile(
    r"(?<![\w.])(\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])"
)
_CACHE_PATH = re.compile(r"data[/\\]+market", re.IGNORECASE)
_READ_CALLS = {"read_parquet", "read_csv", "read_json", "read_feather"}
_DECISION_KEYS = (
    "allowed", "enabled", "reason", "take", "size_multiplier", "indicator_signal",
    "gate", "verdict", "refused", "blocked", "decision", "guard",
)


@dataclass(frozen=True)
class _Problem:
    filename: str
    func: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.filename}::{self.func}: {self.detail}"


def _is_backend_literal(value: float, filename: str) -> bool:
    for known in BACKEND_CONSTANTS:
        if math.isclose(value, known, rel_tol=1e-9, abs_tol=0.0):
            return True
    for known in _SHARED:
        if math.isclose(value, known, rel_tol=1e-9, abs_tol=0.0):
            return True
    if filename in _PER_FILE:
        for known, why in _PER_FILE[filename].items():
            if math.isclose(value, known, rel_tol=1e-9, abs_tol=0.0):
                return bool(why)
    return False


def _reads_live_cache(tree: ast.Module) -> bool:
    """True when a module derives a ``data/market`` path into a pandas reader.

    The path is usually built on the line above the read
    (``path = f"data/market/{symbol}/{interval}.parquet"`` ...
    ``pd.read_parquet(path)``), so data flow is approximated: the module must
    contain both a literal ``data/market`` fragment and a pandas read call.
    Prose mentions in a doc test carry no read call and are not flagged.
    """
    has_reader = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _READ_CALLS
        for node in ast.walk(tree)
    )
    if not has_reader:
        return False
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _CACHE_PATH.search(node.value)
        for node in ast.walk(tree)
    )


def _func_reads_cache(func_src: str) -> bool:
    """True when this test function itself touches ``data/market``.

    An indirection such as ``_cached("BTCUSDT")`` counts too: the function's
    expectations are then still a function of the cache contents.
    """
    if _CACHE_PATH.search(func_src):
        return True
    return bool(re.search(r"\b_cached\s*\(", func_src))


def _looks_like_decision(test_src: str) -> bool:
    return any(key in test_src for key in _DECISION_KEYS)


def _is_skip_call(node: ast.AST) -> bool:
    """True for a ``pytest.skip(...)`` / ``skip(...)`` call node."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return ((isinstance(func, ast.Attribute) and func.attr == "skip")
            or (isinstance(func, ast.Name) and func.id == "skip"))


def _ends_in_a_skip(stmt: ast.stmt) -> bool:
    """True when control cannot fall *through* ``stmt`` to the next statement.

    Covers a bare ``pytest.skip(...)``, an ``if``/``else`` whose two branches
    both end in one, a ``try`` whose body does, and a ``return`` (also
    unreachable code afterwards).  Deliberately narrow: a bare guard clause
    (``if volume <= 0: pytest.skip(...)``) has a fall-through path, so it is
    **not** a skip the following statements are protected by -- the early-guard
    shape has its own predicate, :func:`_is_early_skip_guard`.
    """
    if isinstance(stmt, ast.Expr):
        return _is_skip_call(stmt.value)
    if isinstance(stmt, ast.Return):
        return True
    if isinstance(stmt, ast.If):
        return (bool(stmt.body) and bool(stmt.orelse)
                and _ends_in_a_skip(stmt.body[-1])
                and _ends_in_a_skip(stmt.orelse[-1]))
    if isinstance(stmt, ast.Try):
        if not stmt.body:
            return False
        exits = [stmt.body[-1]]
        exits.extend(h.body[-1] for h in stmt.handlers if h.body)
        if stmt.orelse:
            exits.append(stmt.orelse[-1])
        if stmt.finalbody:
            exits.append(stmt.finalbody[-1])
        return all(_ends_in_a_skip(node) for node in exits)
    return False


def _is_early_skip_guard(stmt: ast.stmt) -> bool:
    """True for a skip that the *rest of the body* sits behind.

    Two accepted shapes: a bare ``pytest.skip(...)``, and the usual cache guard
    ``if not <have cache>: pytest.skip(...)`` (or its ``assert``-in-the-else
    variant).  A mid-body guard clause -- one that follows a read, or whose body
    is a conditional skip wrapped in something else -- is not one of these.
    """
    if isinstance(stmt, ast.Expr):
        return _is_skip_call(stmt.value)
    if not isinstance(stmt, ast.If) or not stmt.body:
        return False
    if not _ends_in_a_skip(stmt.body[-1]):
        return False
    return not stmt.orelse or _ends_in_a_skip(stmt.orelse[-1])


def _pays_out_before_the_skip(stmt: ast.stmt) -> bool:
    """True when ``stmt`` can reach an assert (or a live read) without skipping.

    This is what stops "the skip is the last statement" from being an exemption:
    a late ``pytest.skip`` runs *after* the assertions, so it can hide a failing
    or measured assertion rather than make it unreachable.
    """
    if isinstance(stmt, ast.Assert):
        return True
    if isinstance(stmt, ast.Expr) and _is_skip_call(stmt.value):
        return False
    if isinstance(stmt, ast.Return):
        return False
    if isinstance(stmt, ast.If):
        return _pays_out_before_the_skip(stmt.body[-1]) if stmt.body else False
    if isinstance(stmt, ast.Try):
        if stmt.body and _pays_out_before_the_skip(stmt.body[-1]):
            return True
        if stmt.finalbody and _pays_out_before_the_skip(stmt.finalbody[-1]):
            return True
        return False
    return any(
        _pays_out_before_the_skip(child)
        for name in ("body", "orelse", "finalbody")
        for child in getattr(stmt, name, []) or []
    )


def _skip_dominates_the_body(func: ast.AST) -> bool:
    """True when the function's very first statement ends in a ``pytest.skip``.

    This is the only shape in which "the cache may be absent, nothing was
    asserted about it" justifies exempting a live-cache test from the literal
    scan: *every* statement of the body, including the first, is unreachable
    unless the skip condition is false, and nothing can be measured before it.

    Presence of ``pytest.skip`` anywhere is **not** enough -- the previous
    exemption was ``re.search(r"pytest\\.skip\\s*\\(", body_src)``, so one skip
    bought a whole test a pass.  Two shapes are excluded here on purpose:

    * a guard clause (``frame = read(); if volume <= 0: pytest.skip(...);
      assert volume == 8.588e8``) -- the guard is skipped only for an empty
      cache, and the assertion after it is live whenever the cache exists;
    * a late skip (``assert pin; pytest.skip(...)``) -- the assertion runs
      first, so the skip masks it instead of making it unreachable.
    """
    body = getattr(func, "body", None)
    if not body:
        return False
    first = body[0]
    if not _is_early_skip_guard(first):
        return False
    return not _pays_out_before_the_skip(first)


def _derived_names(func: ast.AST) -> set[str]:
    """Names bound from a call/subscript/attribute (i.e. from a response)."""
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and isinstance(
                node.value, (ast.Call, ast.Subscript, ast.Attribute)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _is_derived(node: ast.AST, names: set[str]) -> bool:
    if isinstance(node, (ast.Call, ast.Subscript, ast.Attribute)):
        return True
    return isinstance(node, ast.Name) and node.id in names


def _has_derived_comparison(func: ast.AST) -> bool:
    """True when an assert compares two run-time expressions (not a literal)."""
    names = _derived_names(func)
    for node in ast.walk(func):
        if not isinstance(node, ast.Assert):
            continue
        for sub in ast.walk(node.test):
            if not isinstance(sub, ast.Compare):
                continue
            sides = [sub.left, *sub.comparators]
            if sum(_is_derived(s, names) for s in sides) >= 2:
                return True
    return False


def _built_argument_floats(func: ast.AST) -> set[float]:
    """Literals used as arguments inside the assertions' own expressions.

    ``impact_pct(0.04, 0.5) == pytest.approx(0.10)`` parameterises the call:
    the literal is an input the test supplies, not an observation.
    """
    out: set[float] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Assert):
            continue
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.Call):
                for arg in [*sub.args, *sub.keywords]:
                    out |= _float_constants(arg)
    return out


def _float_constants(node: ast.AST) -> set[float]:
    out: set[float] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, float):
            out.add(float(sub.value))
    return out


def _assert_float_literals(func: ast.AST) -> list[float]:
    out: list[float] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Assert):
            out.extend(float(x) for x in _NUM.findall(ast.unparse(node.test)))
    return out


def _tripwire_pins(func: ast.AST) -> list[float]:
    """Bare literals compared directly against a value that came from a read."""
    names = _derived_names(func)
    out: list[float] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assert):
            continue
        for cmp in ast.walk(node.test):
            if not isinstance(cmp, ast.Compare):
                continue
            left_derived = _is_derived(cmp.left, names)
            for comparator in cmp.comparators:
                if left_derived:
                    if (isinstance(comparator, ast.Constant)
                            and isinstance(comparator.value, float)):
                        out.append(float(comparator.value))
                elif _is_derived(comparator, names):
                    if (isinstance(cmp.left, ast.Constant)
                            and isinstance(cmp.left.value, float)):
                        out.append(float(cmp.left.value))
    return out


def _scan_module_source(src: str, filename: str) -> list[_Problem]:
    """Return policy violations for one test module's source text."""
    tree = ast.parse(src)
    if not _reads_live_cache(tree):
        return []
    problems: list[_Problem] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        body_src = ast.get_source_segment(src, node) or ""
        if not _func_reads_cache(body_src):
            continue  # synthetic or pure-unit test: the cache is not its input
        if _skip_dominates_the_body(node):
            # The cache may be absent and nothing is asserted about it: an
            # early ``pytest.skip`` that every path must hit.  A guard-clause
            # skip (``if empty: pytest.skip(...)``) does NOT qualify -- the
            # assertions after it run whenever the cache is present.
            continue
        if filename in ENFORCED_DECISION_FILES:
            if not _looks_like_decision(body_src):
                problems.append(_Problem(
                    filename, node.name, "no-decision",
                    "reads the live cache but asserts no decision field "
                    "(allowed/enabled/reason/take/verdict/...)"))
            if not _has_derived_comparison(node):
                problems.append(_Problem(
                    filename, node.name, "no-derived",
                    "reads the live cache but derives no expectation from the "
                    "response; it only pins constants"))
        if filename in ENFORCED_LITERAL_FILES:
            built = _built_argument_floats(node)
            for value in _assert_float_literals(node):
                if value in built or _is_backend_literal(value, filename):
                    continue
                problems.append(_Problem(
                    filename, node.name, "measured-pin",
                    f"assert literal {'%.6g' % value} has no justification in "
                    "FILE_LITERAL_ALLOW or BACKEND_CONSTANTS; pin synthetic data "
                    "or derive the expectation from the response"))
        else:
            for value in _tripwire_pins(node):
                if _is_backend_literal(value, filename):
                    continue
                problems.append(_Problem(
                    filename, node.name, "measured-pin",
                    f"assert compares a read value against the bare literal "
                    f"{'%.6g' % value}; justify it in the allow-list or derive "
                    "the expectation from the response"))
    return problems


def _detected_readers() -> set[str]:
    readers: set[str] = set()
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        if _reads_live_cache(tree):
            readers.add(path.name)
    return readers


def test_live_cache_readers_are_exactly_the_declared_set():
    """A new reader of ``data/market`` must be declared and made compliant."""
    detected = _detected_readers()
    assert detected == LIVE_CACHE_TEST_FILES, (
        "live-cache reader set changed; add the new file to "
        "LIVE_CACHE_TEST_FILES and make its expectations derived/synthetic, or "
        f"drop stale entries. new={sorted(detected - LIVE_CACHE_TEST_FILES)} "
        f"stale={sorted(LIVE_CACHE_TEST_FILES - detected)}")
    assert {"test_meta_labeling.py", "test_pairs.py"} <= LIVE_CACHE_TEST_FILES


def test_cache_reading_tests_obey_the_measured_threshold_policy():
    problems: list[_Problem] = []
    for name in sorted(LIVE_CACHE_TEST_FILES):
        path = TESTS_DIR / name
        if not path.exists():  # pragma: no cover - also caught above
            problems.append(_Problem(name, "<module>", "missing",
                                     "declared as a live-cache reader but absent"))
            continue
        problems.extend(_scan_module_source(path.read_text(encoding="utf-8"), name))
    assert not problems, "measured-statistic policy violations:\n" + "\n".join(
        str(p) for p in problems)


def test_policy_scanner_flags_violations_and_passes_compliant_modules():
    """Self-test: the scan catches the pattern it exists to stop."""
    violating = '''
import pandas as pd


def test_real_rule_is_refused():
    raw = pd.read_parquet("data/market/BTCUSDT/1h.parquet")
    res = {"metrics": {"auc": 0.5866}}
    assert res["metrics"]["auc"] <= 0.5866
'''
    compliant = '''
import pandas as pd


def test_real_rule_is_refused():
    raw = pd.read_parquet("data/market/BTCUSDT/1h.parquet")
    gate = {"allowed": False, "auc": 0.5866, "auc_min": 0.55,
            "reason": "net expectancy -0.20% <= 0.00%"}
    assert gate["allowed"] is False
    assert gate["auc"] <= gate["auc_min"]
    assert "expectancy" in gate["reason"]
'''
    flagged = _scan_module_source(violating, "test_meta_labeling.py")
    kinds = {p.kind for p in flagged}
    assert "measured-pin" in kinds, [str(p) for p in flagged]
    assert "no-derived" in kinds, [str(p) for p in flagged]
    assert not _scan_module_source(compliant, "test_meta_labeling.py")


def test_policy_scanner_flags_a_measured_pin_guarded_by_a_late_skip():
    """The R3 experiment, module 1: no skip at all -- the pin is flagged.

    Three modules that differ only in how ``pytest.skip`` appears.  The old
    exemption keyed on ``re.search(r"pytest\\.skip\\s*\\(", body_src)``, so
    module 2 (one skip guarding an empty-cache branch) and module 3 (a skip in
    addition to live pins) both reported **0** problems where this module
    reports 2.
    """
    plain = '''
import pandas as pd


def test_no_skip():
    frame = pd.read_parquet("data/market/BTCUSDT/1h.parquet")
    auc = _evaluate(frame)
    assert auc <= 0.5866
'''
    kinds = [p.kind for p in _scan_module_source(plain, "test_meta_labeling.py")]
    assert kinds == ["no-decision", "no-derived", "measured-pin"], kinds


def test_policy_scanner_exempts_only_a_skip_that_dominates_the_body():
    """The R3 experiment, modules 2 and 3: load-bearing vs merely present.

    Module 2 opens with a skip that every path must hit before any read, so no
    assertion in it is reachable when the cache is present: the module is exempt
    -- that is the exemption the old rule got right.  Module 3 is the same skip
    *in addition to* a live pin; the old rule hid it entirely, and it must now be
    flagged exactly like module 1.
    """
    load_bearing = '''
import pandas as pd
import pytest


def test_skip_gates_everything():
    if not _has_cache():
        pytest.skip("the checkout does not ship the cache")
    frame = pd.read_parquet("data/market/BTCUSDT/1h.parquet")
    auc = _evaluate(frame)
    assert auc <= 0.5866
'''
    skip_plus_live_pins = '''
import pandas as pd
import pytest


def test_skip_in_addition_to_live_pins():
    frame = pd.read_parquet("data/market/BTCUSDT/1h.parquet")
    volume = float(frame["volume"].tail(20).sum())
    if volume <= 0.0:
        pytest.skip("empty cache")
    assert volume == 8.588e8
'''
    exempt = _scan_module_source(load_bearing, "test_meta_labeling.py")
    assert not exempt, [str(p) for p in exempt]
    flagged = _scan_module_source(skip_plus_live_pins, "test_meta_labeling.py")
    assert [p.kind for p in flagged] == ["no-decision", "no-derived",
                                         "measured-pin"], flagged
    detail = " ".join(str(p) for p in flagged)
    assert "8.588e+08" in detail or "8.588" in detail, flagged


def test_policy_scanner_requires_every_reader_to_be_declared():
    """The declared-set equality cannot silently lose its tripwire."""
    assert _detected_readers() == LIVE_CACHE_TEST_FILES
    undeclared_src = '''
import pandas as pd


def test_new_reader():
    df = pd.read_parquet("data/market/NEWUSDT/1h.parquet")
    assert df.shape[0] > 10
'''
    assert _reads_live_cache(ast.parse(undeclared_src))
