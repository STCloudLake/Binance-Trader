"""In-app documentation manual: discovery, safe resolution and markdown rendering.

The repository already carries its documentation as markdown
(``docs/**/*.md``, ``README.md``, ``README_EN.md``).  This module turns that
tree into a browsable manual **without copying anything**: every request
re-reads the file from disk, so an edit to a document shows up on the next
refresh.

Doc-discovery rule (the source of truth)
---------------------------------------
A file belongs to the manual **iff** it is a *regular* ``*.md`` file that is
either

1. anywhere under ``<repo>/docs/`` (recursive), or
2. exactly ``<repo>/README.md`` or ``<repo>/README_EN.md``.

and, in both cases, no path segment of its repository-relative path starts
with ``.`` and the file (and the directory chain that reached it) is not a
symlink.  Everything else is *not* documentation for this manual: non-markdown
files (``docs/overhaul/route-baseline.json``, ``config/*.yaml``), hidden
entries, symlinks, and markdown outside ``docs/`` such as
``experimental/ml/README.md``.

Path safety
-----------
The requested URL suffix is repository-relative (``docs/overhaul/PLAN.md``).
It is *never* handed to the filesystem before it has been normalised and
checked, and it must additionally be a member of the discovered set — so a
document that the discovery rule excludes (hidden, symlinked, outside the
allowed roots) is a 404 even if the file exists.  ``..``, absolute paths,
drive letters, NUL bytes, empty segments, percent-encoded variants
(``%2e%2e%2f``, double-encoded ``%252e``) and symlinked files are all
rejected.  ``resolve_document()`` is the single entry point for that check.

Rendering
---------
markdown-it-py (already installed; declared in ``requirements.txt``) with the
CommonMark preset plus the ``table`` rule and ``html=False`` — raw HTML inside
a document is therefore escaped, never executed.  Three post/pre-processors
sit on top:

* math (``$$...$$`` / ``$...$``) is extracted *before* markdown parsing and
  re-inserted verbatim into ``span.math-inline`` / ``div.math-block`` so the
  LaTeX survives markdown escaping; KaTeX (CDN) renders it in the browser and
  degrades to the raw source when the CDN is unreachable;
* headings get GitHub-compatible ``id`` slugs plus an in-page table of
  contents;
* every intra-doc link is rewritten to its ``/manual/...`` route (or to the
  directory group anchor), and a link whose target is not part of the manual
  becomes a marked, non-navigating ``span.manual-dead-link`` instead of a 404.
"""
from __future__ import annotations

import os
import posixpath
import re
import urllib.parse
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml

#: Repository root (``<repo>/web/manual.py`` -> ``<repo>``).
REPO_ROOT = Path(__file__).resolve().parents[1]
#: Root of the recursive documentation tree.
DOCS_DIR = REPO_ROOT / "docs"
#: Repository-root documents that also belong to the manual.
ROOT_DOCUMENTS = ("README.md", "README_EN.md")

#: URL prefix of the rendered manual.
MANUAL_PREFIX = "/manual"
#: JSON tree endpoint.
TREE_ENDPOINT = "/api/manual/tree"
#: Markdown suffix the manual accepts.
DOC_SUFFIX = ".md"

#: Sidebar label per directory (falls back to the directory name).
DIR_LABELS = {
    "": "根目录 / Root",
    "docs": "交接 / Handover",
    "audit": "审计 / Audit",
    "core-algorithms": "核心算法 / Core algorithms",
    "overhaul": "改造 / Overhaul",
    "research": "研究 / Research",
    "superpowers": "过程记录 / Superpowers",
    "plans": "计划 / Plans",
    "specs": "规格 / Specs",
}

#: Landing-page "main documents" (only the ones that exist are shown).
MAIN_DOCUMENTS = (
    "README.md",
    "docs/HANDOVER.md",
    "docs/overhaul/PLAN.md",
    "docs/overhaul/CHANGELOG.md",
    "docs/overhaul/ALGO_UPGRADE_PLAN.md",
    "docs/overhaul/ALGO_UPGRADE_EVIDENCE.md",
    "docs/research/CORE_ALGORITHMS.md",
    "docs/core-algorithms/00-ERRATA.md",
    "docs/audit/README.md",
    "docs/superpowers/plans/2026-07-16-code-audit-plan.md",
    "README_EN.md",
)

# --------------------------------------------------------------------------
# caches — keyed by (path, mtime_ns, size) so an edit invalidates instantly
# --------------------------------------------------------------------------
_CACHE_LIMIT = 128
_render_cache: dict[tuple, dict] = {}
_title_cache: dict[tuple, str] = {}
_anchors_cache: dict[tuple, dict] = {}


def clear_caches() -> None:
    """Drop the mtime-keyed caches (used by tests)."""
    _render_cache.clear()
    _title_cache.clear()
    _anchors_cache.clear()


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def discover_documents(root: Path | None = None,
                       root_files: tuple[str, ...] = ROOT_DOCUMENTS,
                       repo_root: Path | None = None) -> list[str]:
    """Repository-relative posix paths of every document in the manual.

    See the module docstring for the exact rule.  The walk never follows
    symlinked directories (``os.walk(followlinks=False)``) and skips hidden
    directories/files, so the returned set *is* the allow-list used by
    :func:`resolve_document`.
    """
    base = Path(root) if root is not None else DOCS_DIR
    repo = Path(repo_root) if repo_root is not None else REPO_ROOT
    found: set[str] = set()

    if base.is_dir():
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames
                                 if not _is_hidden(d)
                                 and not (Path(dirpath) / d).is_symlink())
            for name in filenames:
                if _is_hidden(name) or not name.endswith(DOC_SUFFIX):
                    continue
                path = Path(dirpath) / name
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    rel = path.relative_to(repo).as_posix()
                except ValueError:  # pragma: no cover - root outside the repo
                    continue
                if any(_is_hidden(part) for part in rel.split("/")):
                    continue
                found.add(rel)

    for name in root_files:
        path = repo / name
        if _is_hidden(name) or not name.endswith(DOC_SUFFIX):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        found.add(name)

    return sorted(found)


def _clean_relpath(rel: str) -> str | None:
    """Normalise one URL suffix, or return ``None`` when it is unsafe.

    Rejects: non-strings, NUL bytes, percent-encoded traversal (decoded up to
    two levels), backslashes/UNC, absolute paths, ``~``, drive letters,
    ``.``/``..``/empty segments and non-markdown suffixes.
    """
    if not isinstance(rel, str) or not rel:
        return None
    if "\x00" in rel:
        return None
    text = rel
    for _ in range(2):  # ASGI decodes once; a second pass catches %252e style
        try:
            decoded = urllib.parse.unquote(text)
        except Exception:  # pragma: no cover - unquote is total for str
            return None
        if decoded == text:
            break
        text = decoded
    text = text.replace("\\", "/")
    if text.startswith("/") or text.startswith("~"):
        return None
    if len(text) > 1 and text[1] == ":":
        return None
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    if not text.endswith(DOC_SUFFIX):
        return None
    if any(ord(ch) < 32 for ch in text):
        return None
    return "/".join(parts)


def resolve_document(rel: str, known: set[str] | None = None) -> Path | None:
    """Absolute path of ``rel`` when it is a document of the manual, else ``None``.

    ``rel`` is the repository-relative posix path (e.g. ``docs/HANDOVER.md``).
    The path must pass :func:`_clean_relpath`, be a member of the discovered
    allow-list, resolve to a regular non-symlink ``.md`` file inside an allowed
    root (``docs/`` or the two root READMEs) *after* symlink resolution.
    """
    cleaned = _clean_relpath(rel)
    if cleaned is None:
        return None
    allowed = known if known is not None else set(discover_documents())
    if cleaned not in allowed:
        return None
    candidate = REPO_ROOT / cleaned
    if candidate.is_symlink():
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not resolved.is_file() or resolved.suffix.lower() != DOC_SUFFIX:
        return None
    repo_root = REPO_ROOT.resolve()
    if not resolved.is_relative_to(repo_root):
        return None
    if cleaned in ROOT_DOCUMENTS:
        return resolved
    docs_root = DOCS_DIR.resolve()
    if not resolved.is_relative_to(docs_root):
        return None
    # Belt and braces: a symlinked *directory* inside docs/ that points outside
    # the repository is already excluded by containment above; a symlink that
    # stays inside is excluded here so only real files are served.
    for parent in resolved.parents:
        if parent == docs_root:
            break
        if parent.is_symlink():
            return None
    return resolved


# --------------------------------------------------------------------------
# titles
# --------------------------------------------------------------------------
_H1_RE = re.compile(r"^#[ \t]+(.+?)[ \t]*$", re.M)
_ANY_HEADING_RE = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.M)


def _plain_title(text: str) -> str:
    """Strip inline markdown from a heading line."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("*_` ")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


def document_title(path: Path) -> str:
    """First ``#`` heading of ``path`` (markdown stripped), else its filename."""
    try:
        stat = path.stat()
    except OSError:
        return path.name
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    hit = _title_cache.get(key)
    if hit is not None:
        return hit
    try:
        text = _read_text(path)
    except OSError:  # pragma: no cover - race with an external delete
        return path.name
    match = _H1_RE.search(text) or _ANY_HEADING_RE.search(text)
    title = _plain_title(match.group(1)) if match else ""
    if not title:
        title = path.name
    if len(_title_cache) >= _CACHE_LIMIT:
        _title_cache.clear()
    _title_cache[key] = title
    return title


def document_url(rel: str) -> str:
    return f"{MANUAL_PREFIX}/{rel}"


def dir_anchor(dir_path: str) -> str:
    """Stable DOM id for a directory group in the sidebar."""
    slug = re.sub(r"[^a-z0-9]+", "-", dir_path.lower()).strip("-")
    return f"group-{slug or 'manual'}"


def _dir_label(dir_path: str) -> str:
    name = dir_path.rsplit("/", 1)[-1] if dir_path else ""
    return DIR_LABELS.get(name, name)


# --------------------------------------------------------------------------
# tree
# --------------------------------------------------------------------------
def build_tree() -> dict:
    """Nested directory tree + flat document list (the ``/api/manual/tree`` payload)."""
    docs = discover_documents()
    nodes: dict[str, dict] = {}
    roots: list[dict] = []

    def dir_node(path: str) -> dict:
        node = nodes.get(path)
        if node is not None:
            return node
        node = {
            "type": "dir",
            "name": path.rsplit("/", 1)[-1] if path else "",
            "path": path,
            "label": _dir_label(path),
            "anchor": dir_anchor(path),
            "count": 0,
            "children": [],
        }
        nodes[path] = node
        # Top level = the ``docs/`` tree and the repository-root README group;
        # everything deeper hangs off its parent directory.
        if path in ("", DOCS_DIR.name):
            roots.append(node)
        else:
            parent = path.rsplit("/", 1)[0]
            dir_node(parent)["children"].append(node)
        return node

    documents: list[dict] = []
    for rel in docs:
        dir_path = rel.rsplit("/", 1)[0] if "/" in rel else ""
        parent = dir_node(dir_path)
        path = REPO_ROOT / rel
        entry = {
            "type": "doc",
            "name": rel.rsplit("/", 1)[-1],
            "path": rel,
            "title": document_title(path),
            "url": document_url(rel),
            "group": dir_path,
            "group_label": _dir_label(dir_path),
            "anchor": dir_anchor(dir_path),
        }
        parent["children"].append(entry)
        documents.append(entry)

    def sort_and_count(node: dict) -> int:
        node["children"].sort(key=lambda c: (0 if c["type"] == "dir" else 1,
                                             c["path"].lower()))
        total = 0
        for child in node["children"]:
            total += sort_and_count(child) if child["type"] == "dir" else 1
        node["count"] = total
        return total

    for root in roots:
        sort_and_count(root)

    return {
        "roots": roots,
        "documents": documents,
        "dirs": [
            {"path": path, "label": _dir_label(path), "anchor": dir_anchor(path),
             "count": nodes[path]["count"]}
            for path in sorted(nodes)
        ],
        "doc_count": len(documents),
        "roots_allowed": [str(DOCS_DIR.relative_to(REPO_ROOT)), *ROOT_DOCUMENTS],
        "rule": ("docs/**/*.md (recursive, no hidden/symlinked entries) "
                 "+ README.md + README_EN.md"),
    }


# --------------------------------------------------------------------------
# markdown pipeline
# --------------------------------------------------------------------------
_MATH_BLOCK_TOKEN = "BTMATHBLOCK{}BT"
_MATH_INLINE_TOKEN = "BTMATHINLINE{}BT"
_CODE_SPAN_TOKEN = "BTCODESPAN{}BT"

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1")
_BLOCK_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
_INLINE_MATH_RE = re.compile(r"(?<![\\$\w])\$(?![\s$])([^$\n]{1,160}?)(?<![\s\\])\$(?![\w$])")
_MATH_SAFE_RE = re.compile(r"^[A-Za-z0-9_^={}\(\)\[\]+\-*/.,<>:;|!\\ \t]*$")


def _is_inline_math(body: str) -> bool:
    """Keep obvious money/code out of the math slots (only math survives KaTeX)."""
    if not _MATH_SAFE_RE.match(body):
        return False
    if "\\" in body:
        return True
    return len(body) <= 40 and any(ch.isalnum() for ch in body)


def _prose_chunks(text: str):
    """Split markdown into (is_fenced_code, chunk) pieces, line by line."""
    lines = text.split("\n")
    chunk: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in lines:
        match = _FENCE_RE.match(line)
        if not in_fence and match:
            if chunk:
                yield False, "\n".join(chunk)
                chunk = []
            in_fence = True
            fence_marker = match.group(1)
            chunk.append(line)
            continue
        if in_fence:
            chunk.append(line)
            if match and match.group(1)[0] == fence_marker[0] \
                    and len(match.group(1)) >= len(fence_marker) \
                    and not match.group(2).strip():
                yield True, "\n".join(chunk)
                chunk = []
                in_fence = False
                fence_marker = ""
            continue
        chunk.append(line)
    if chunk:
        yield in_fence, "\n".join(chunk)


def _protect_math(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """Replace math (and inline code) with plain-text placeholder tokens.

    Returns ``(prepared_markdown, slots)`` where ``slots[token] = (kind, raw)``
    and ``kind`` is ``"block"``, ``"inline"`` or ``"code"``.  Code spans are put
    back as markdown *before* parsing; only math stays slotted and is restored
    into the rendered HTML (escaped), so ``$..$`` never gets mangled by
    markdown escaping and KaTeX can pick the delimiters up in the browser.
    """
    slots: dict[str, tuple[str, str]] = {}
    counter = 0

    def next_token(template: str) -> str:
        nonlocal counter
        token = template.format(counter)
        counter += 1
        return token

    out: list[str] = []
    for is_code, chunk in _prose_chunks(text):
        if is_code:
            out.append(chunk)
            continue

        code_spans: list[tuple[str, str]] = []

        def _code(match: re.Match) -> str:
            token = next_token(_CODE_SPAN_TOKEN)
            code_spans.append((token, match.group(0)))
            return token

        body = _CODE_SPAN_RE.sub(_code, chunk)

        def _block(match: re.Match) -> str:
            token = next_token(_MATH_BLOCK_TOKEN)
            slots[token] = ("block", match.group(1))
            return token

        body = _BLOCK_MATH_RE.sub(_block, body)

        def _inline(match: re.Match) -> str:
            inner = match.group(1)
            if not _is_inline_math(inner):
                return match.group(0)
            token = next_token(_MATH_INLINE_TOKEN)
            slots[token] = ("inline", inner)
            return token

        body = _INLINE_MATH_RE.sub(_inline, body)

        for token, original in code_spans:  # code goes back, math stays slotted
            body = body.replace(token, original)
        out.append(body)
    # Chunks are consecutive line groups: re-joining them with "\n" reproduces
    # the source byte for byte (joining with "" would glue prose to a fence).
    return "\n".join(out), slots


def github_slug(text: str) -> str:
    """GitHub-compatible heading slug (keeps CJK, drops punctuation)."""
    out: list[str] = []
    for ch in text.strip().lower():
        if ch.isspace():
            out.append("-")
        elif ch in "-_" or ch.isalnum():
            out.append(ch)
    return "".join(out)


def slug_key(slug: str) -> str:
    """Hyphen-insensitive key so hand-written anchors still resolve.

    Several documents link to headings with a slug that differs from GitHub's
    only in the number of ``-`` runs around removed punctuation; matching on
    the alphanumeric core keeps those links alive.
    """
    return "".join(ch for ch in slug.lower() if ch.isalnum())


def _render_link_open(self, tokens, idx, options, env):
    token = tokens[idx]
    if token.meta and token.meta.get("dead"):
        return '<span class="manual-dead-link" title="%s">' % escapeHtml(
            token.meta.get("reason", ""))
    return self.renderToken(tokens, idx, options, env)


def _render_link_close(self, tokens, idx, options, env):
    token = tokens[idx]
    if token.meta and token.meta.get("dead"):
        return "</span>"
    return self.renderToken(tokens, idx, options, env)


def _render_image(self, tokens, idx, options, env):
    token = tokens[idx]
    if token.meta and token.meta.get("dead"):
        title = token.meta.get("reason", "")
        return '<span class="manual-dead-link" title="%s">[%s]</span>' % (
            escapeHtml(title), escapeHtml(token.content or ""))
    return self.renderToken(tokens, idx, options, env)


def build_markdown() -> MarkdownIt:
    """CommonMark + tables + the manual's link/image renderers."""
    md = MarkdownIt("commonmark", {"html": False, "linkify": False,
                                   "typographer": False, "breaks": False})
    md.enable("table")
    md.add_render_rule("link_open", _render_link_open)
    md.add_render_rule("link_close", _render_link_close)
    md.add_render_rule("image", _render_image)
    return md


_MD = build_markdown()


def _iter_tokens(tokens):
    for token in tokens:
        yield token
        if token.children:
            yield from _iter_tokens(token.children)


def _restore_token_text(text: str, slots: dict) -> str:
    for token, (kind, raw) in slots.items():
        if token in text:
            text = text.replace(token, raw if kind != "block" else f"$${raw}$$")
    return text


def _heading_anchors(tokens, slots) -> list[dict]:
    headings: list[dict] = []
    used: set[str] = set()
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        inline = tokens[index + 1] if index + 1 < len(tokens) else None
        raw = _restore_token_text(getattr(inline, "content", "") or "", slots)
        text = _plain_title(raw)
        slug = github_slug(text) or "section"
        unique = slug
        bump = 1
        while unique in used:
            unique = f"{slug}-{bump}"
            bump += 1
        used.add(unique)
        token.attrSet("id", unique)
        headings.append({"level": int(token.tag[1]), "text": text or unique,
                         "id": unique})
    return headings


def _target_dirs(docs) -> set[str]:
    dirs: set[str] = set()
    for rel in docs:
        parts = rel.split("/")[:-1]
        for i in range(len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    return dirs


def _file_anchor_map(path: Path) -> dict[str, str]:
    """``slug_key -> canonical heading id`` for another document (cached).

    Used to repair a cross-document ``#fragment``: a stale anchor would scroll
    nowhere, so it is dropped (the link still lands on the right document).
    """
    try:
        stat = path.stat()
    except OSError:  # pragma: no cover - race with an external delete
        return {}
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    hit = _anchors_cache.get(key)
    if hit is not None:
        return hit
    prepared, slots = _protect_math(_read_text(path))
    headings = _heading_anchors(_MD.parse(prepared), slots)
    mapping = {slug_key(head["id"]): head["id"] for head in headings}
    if len(_anchors_cache) >= _CACHE_LIMIT:
        _anchors_cache.clear()
    _anchors_cache[key] = mapping
    return mapping


def _rewrite_links(tokens, rel: str, docs: set[str], dirs: set[str],
                   anchors: dict[str, str]) -> None:
    """Point every intra-doc link at its manual route; mark the dead ones."""
    stack: list = []
    for token in _iter_tokens(tokens):
        if token.type == "image":
            href = (token.attrGet("src") or "").strip()
            if href and not href.startswith(("http://", "https://", "data:")):
                token.meta = {"dead": True,
                              "reason": f"图片不在手册范围内：{href}"}
            continue
        if token.type == "link_open":
            stack.append(token)
            _rewrite_one(token, rel, docs, dirs, anchors)
        elif token.type == "link_close" and stack:
            source = stack.pop()
            if source.meta.get("dead"):
                token.meta = {"dead": True}


def _mark_dead(token, reason: str) -> None:
    token.meta = {"dead": True, "reason": reason}
    if isinstance(token.attrs, dict):
        token.attrs.pop("href", None)
    else:  # pragma: no cover - other markdown-it attribute containers
        token.attrs = [(key, value) for key, value in (token.attrs or [])
                       if key != "href"]


def _rewrite_one(token, rel: str, docs: set[str], dirs: set[str],
                 anchors: dict[str, str]) -> None:
    href = (token.attrGet("href") or "").strip()
    if not href:
        return
    if href.startswith(("http://", "https://", "mailto:", "//")):
        return
    if href.startswith("#"):
        fragment = urllib.parse.unquote(href[1:])
        resolved = anchors.get(slug_key(fragment))
        if resolved:
            token.attrSet("href", f"#{resolved}")
        else:
            _mark_dead(token, f"本文档内不存在锚点：#{fragment}")
        return

    target, _, fragment = href.partition("#")
    target = urllib.parse.unquote(target)
    joined = posixpath.normpath(
        posixpath.join(posixpath.dirname(rel), target)).rstrip("/")
    if joined in dirs and joined not in docs:
        # A directory link (``docs/core-algorithms/``) becomes a jump to that
        # group of the manual tree — never a route that could 404.
        token.attrSet("href", f"{MANUAL_PREFIX}#{dir_anchor(joined)}")
        return
    if joined in docs:
        href_out = document_url(joined)
        if fragment:
            canonical = _file_anchor_map(REPO_ROOT / joined).get(
                slug_key(urllib.parse.unquote(fragment)))
            if canonical:
                href_out += f"#{canonical}"
        token.meta = {"kind": "doc"}
        token.attrSet("href", href_out)
        return
    _mark_dead(token, f"目标不在手册范围内或不存在：{href}")


def _restore_math(html: str, slots: dict) -> str:
    for token, (kind, raw) in slots.items():
        if token not in html:
            continue
        escaped = escapeHtml(raw)
        if kind == "block":
            block = f'<div class="math-block">$${escaped}$$</div>'
            # A lambda replacement: the LaTeX body must not be read as regex
            # escapes by ``re.sub`` (``\\Delta`` etc.).
            pattern = r"<p>\s*" + re.escape(token) + r"\s*</p>"
            if re.search(pattern, html):
                html = re.sub(pattern, lambda _match: block, html)
            else:
                html = html.replace(token, block)
        else:
            html = html.replace(token, f'<span class="math-inline">${escaped}</span>')
    return html


def render_markdown(text: str, rel: str, docs: set[str] | None = None,
                    dirs: set[str] | None = None) -> dict:
    """Render one document body: ``{html, headings, anchors}``."""
    known = docs if docs is not None else set(discover_documents())
    directories = dirs if dirs is not None else _target_dirs(known)
    prepared, slots = _protect_math(text)
    tokens = _MD.parse(prepared)
    headings = _heading_anchors(tokens, slots)
    anchors = {slug_key(h["id"]): h["id"] for h in headings}
    _rewrite_links(tokens, rel, known, directories, anchors)
    html = _MD.renderer.render(tokens, _MD.options, {})
    html = _restore_math(html, slots)
    return {"html": html, "headings": headings, "anchors": anchors}


def render_document(rel: str) -> dict | None:
    """Fully rendered document, or ``None`` when ``rel`` is not in the manual."""
    path = resolve_document(rel)
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:  # pragma: no cover - race with an external delete
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    hit = _render_cache.get(key)
    if hit is not None:
        return hit

    text = _read_text(path)
    rendered = render_markdown(text, rel)
    headings = rendered["headings"]
    title = next((h["text"] for h in headings if h["level"] == 1), None)
    if not title:
        title = headings[0]["text"] if headings else path.name
    doc = {
        "path": rel,
        "name": path.name,
        "title": title,
        "html": rendered["html"],
        "headings": headings,
        "toc": [h for h in headings if h["level"] >= 2],
        "anchors": rendered["anchors"],
        "mtime_ns": stat.st_mtime_ns,
        "bytes": stat.st_size,
    }
    if len(_render_cache) >= _CACHE_LIMIT:
        _render_cache.clear()
    _render_cache[key] = doc
    return doc


# --------------------------------------------------------------------------
# page context helpers
# --------------------------------------------------------------------------
def breadcrumbs(rel: str) -> list[dict]:
    """``[{'label','url'}, ...]`` from the manual root down to the document."""
    trail: list[dict] = [{"label": "手册 / Manual", "url": MANUAL_PREFIX}]
    parts = rel.split("/")
    for i, part in enumerate(parts[:-1]):
        path = "/".join(parts[:i + 1])
        trail.append({"label": _dir_label(path),
                      "url": f"{MANUAL_PREFIX}#{dir_anchor(path)}"})
    trail.append({"label": parts[-1], "url": None})
    return trail


def navigation(tree: dict, rel: str) -> tuple[dict | None, dict | None]:
    """Previous / next document in the flat (sorted) document order."""
    paths = [doc["path"] for doc in tree["documents"]]
    if rel not in paths:
        return None, None
    index = paths.index(rel)
    prev_doc = tree["documents"][index - 1] if index > 0 else None
    next_doc = tree["documents"][index + 1] if index + 1 < len(paths) else None
    return prev_doc, next_doc


def main_documents(tree: dict) -> list[dict]:
    """The curated landing-page list, filtered to documents that exist."""
    by_path = {doc["path"]: doc for doc in tree["documents"]}
    return [by_path[path] for path in MAIN_DOCUMENTS if path in by_path]
