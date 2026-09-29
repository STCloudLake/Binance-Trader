"""Security regression tests for strategy condition evaluation.

Strategy conditions are attacker-influenced: any `trader` account can save a
strategy via `POST /api/strategy`, the AI lifecycle generates them, and GA
chromosomes carry them. They are evaluated on live ticks and in backtests.

The previous implementation used `pd.eval(..., engine="python")` guarded by a
regex *denylist*, which was bypassable — a condition such as

    close.to_csv(r'<path>', header=['im'+'port o'+'s;o'+'s.syste'+'m(chr(99))']) is not None

passed validation and executed the method call (arbitrary file write). These
tests pin the allowlist that replaced it.
"""
import ast
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.strategy.indicators import (
    UnsafeConditionError,
    _check_condition_ast,
    evaluate_condition,
)


@pytest.fixture()
def df():
    n = 60
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": rng.random(n) * 100,
        "rsi": rng.random(n) * 100,
        "macd_histogram": rng.normal(0, 1, n),
        "sma": close,
    }, index=pd.date_range("2026-01-01", periods=n, freq="1h"))


MALICIOUS = [
    # The exact bypass that worked against the old denylist
    "close.to_csv('/tmp/bt_pwned.csv') is not None",
    "close.to_csv(__import__('os').environ) is not None",
    "__import__('os').system('echo pwned')",
    "open('/tmp/bt_pwned.txt', 'w') is not None",
    "close.__class__",
    "(1).__class__.__mro__",
    "__builtins__",
    "eval('1+1')",
    "exec('x=1')",
    "close.values.tolist()",
    "getattr(close, 'to_csv')('/tmp/x.csv')",
    "close[0]",
    "lambda: 1",
    "[x for x in range(3)]",
    "'__cla' + 'ss__'",
    "f'{close}'",
    "close if close.any() else 0",
    "close @ close",
    "close ** 999999",
]


@pytest.mark.parametrize("condition", MALICIOUS)
def test_malicious_conditions_are_rejected(df, condition):
    """Rejected conditions must be all-False and must not execute anything."""
    result = evaluate_condition(df, condition)
    assert isinstance(result, pd.Series)
    assert result.dtype == bool
    assert not result.any(), f"{condition!r} should evaluate to all-False"
    # and the structural check must reject it outright
    with pytest.raises(UnsafeConditionError):
        _check_condition_ast(ast.parse(condition, mode="eval"))


def test_known_exploit_cannot_write_a_file(df):
    """The demonstrated file-write primitive must be impossible."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "pwned.csv"
        payload = (
            "close.to_csv(r'" + str(target) + "', "
            "header=['im' + 'port o' + 's;o' + 's.syste' + 'm(chr(99))'], index=False) is not None"
        )
        result = evaluate_condition(df, payload)
        assert not result.any()
        assert not target.exists(), "arbitrary file write via a strategy condition!"


def test_valid_conditions_still_work(df):
    assert evaluate_condition(df, "rsi < 30").dtype == bool
    combined = evaluate_condition(df, "rsi > 70 and close > 0")
    direct = (df["rsi"] > 70) & (df["close"] > 0)
    pd.testing.assert_series_equal(combined, direct, check_names=False)
    # OR
    either = evaluate_condition(df, "rsi < 10 or rsi > 90")
    pd.testing.assert_series_equal(either, (df["rsi"] < 10) | (df["rsi"] > 90), check_names=False)
    # chained comparison
    chained = evaluate_condition(df, "0 < rsi < 50")
    pd.testing.assert_series_equal(chained, (df["rsi"] > 0) & (df["rsi"] < 50), check_names=False)
    # arithmetic and whitelisted functions
    assert evaluate_condition(df, "abs(macd_histogram) < 1").dtype == bool
    assert evaluate_condition(df, "close / open > 0.9").dtype == bool
    assert evaluate_condition(df, "sma(rsi, 5) > 50").dtype == bool
    assert evaluate_condition(df, "cross(close, sma)").dtype == bool


def test_numeric_truthiness_is_coerced(df):
    """A bare numeric expression becomes an all-True/False boolean Series."""
    assert evaluate_condition(df, "1 > 0").all()
    assert not evaluate_condition(df, "1 < 0").any()


def test_unknown_column_is_all_false_and_does_not_raise(df):
    result = evaluate_condition(df, "nonexistent_column > 50")
    assert not result.any()


def test_empty_and_non_string_conditions_are_safe(df):
    for bad in ["", "   ", None, 123]:
        result = evaluate_condition(df, bad)
        assert not result.any()


def test_string_literals_are_forbidden(df):
    """Strings have no numeric use and enable identifier-splitting tricks."""
    for condition in ["close == 'x'", "close > 'a'", "'abc'"]:
        with pytest.raises(UnsafeConditionError):
            _check_condition_ast(ast.parse(condition, mode="eval"))
