"""Jinja2 environment and rendering helpers for the web layer.

Extracted verbatim from ``web/server.py`` (file-organization refactor only).
The module-level Jinja environment, the ``fmt_time`` filter, the ``CST``
display timezone and the ``_render`` / ``_T`` / ``_get_lang`` helpers are
unchanged in behaviour; they are simply no longer defined inside
``create_app()``.
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

# China Standard Time (UTC+8)
CST = timezone(timedelta(hours=8))
DEFAULT_BALANCE = 10000.0

#: Templates live next to this module.
_templates_dir = Path(__file__).parent / "templates"

# Jinja2 environment — created once and reused.
# SECURITY: autoescape is ON for HTML. Alert messages, AI-generated text, news
# headlines, strategy names and reflected query params all reach these templates;
# without escaping a stored/reflected XSS in an admin session was possible.
_jinja_env = Environment(
    loader=FileSystemLoader(str(_templates_dir)),
    autoescape=select_autoescape(enabled_extensions=("html", "htm", "xml"),
                                 default_for_string=True, default=True),
)


def _fmt_time(utc_str):
    """Convert a UTC timestamp string from SQLite to CST (UTC+8) for display."""
    if not utc_str:
        return '-'
    try:
        dt = datetime.strptime(str(utc_str)[:19], '%Y-%m-%d %H:%M:%S')
        dt = dt.replace(tzinfo=timezone.utc).astimezone(CST)
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return str(utc_str)[:16]


_jinja_env.filters["fmt_time"] = _fmt_time


from web.i18n import get_translator


def _T(key: str) -> str:
    """Translate a key for inline HTML usage. Uses current config language."""
    return get_translator(_get_lang())(key)


def _get_lang() -> str:
    try:
        from app.config import Config
        c = Config._instance
        if c and c._loaded:
            return getattr(c, "language", "zh")
    except Exception:
        pass
    return "zh"


def _render(template_name: str, context: dict, lang: str = None) -> HTMLResponse:
    template = _jinja_env.get_template(template_name)
    lang = lang or _get_lang()
    context["_"] = get_translator(lang)
    context["lang"] = lang
    if context.get("request") and hasattr(context["request"], "state"):
        context["current_user"] = getattr(context["request"].state, "user", None)
    return HTMLResponse(template.render(**context))
