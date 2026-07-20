"""Unit tests for i18n translation module."""

import pytest
from web.i18n import get_translator


def test_get_translator_chinese():
    """Chinese translator should return translations."""
    _ = get_translator("zh")
    result = _("仪表盘")
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_translator_english():
    """English translator should return translations."""
    _ = get_translator("en")
    result = _("Dashboard")
    assert isinstance(result, str)


def test_get_translator_missing_key():
    """Missing translation should return the key itself."""
    _ = get_translator("zh")
    result = _("this_key_does_not_exist_12345")
    assert result == "this_key_does_not_exist_12345"


def test_get_translator_unknown_language():
    """Unknown language should fall back to Chinese."""
    _ = get_translator("fr")
    result = _("仪表盘")
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_translator_functional():
    """Translator should be callable."""
    _ = get_translator("zh")
    assert callable(_)
    assert callable(get_translator("en"))
