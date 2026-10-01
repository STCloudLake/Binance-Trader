"""Tests for the in-app documentation manual (``/manual``).

Covered here:

* discovery completeness — every ``docs/**/*.md`` plus the two root READMEs
  appears in the tree, and nothing else does;
* rendering — 200 + the document's own ``#`` title, tables/code blocks, the
  KaTeX math slots, heading ids and the in-page table of contents;
* navigation — relative ``.md`` links (same dir, parent dir, cross dir) are
  rewritten to ``/manual/...``, directory links jump to a tree group, and a
  link whose target is not in the manual becomes a marked dead link instead of
  a 404;
* path safety — ``..``, absolute paths, drive letters, ``docs/../.env`` and
  percent-encoded variants are rejected both at the HTTP layer and by
  ``web.manual.resolve_document``;
* authorization — ``/manual*`` is behind the same ``AuthMiddleware`` as the
  other operator pages (302 for pages, 401 for ``/api/``).

Everything runs against a temporary SQLite database and an injected temp
``config/`` directory; the real ``data/binance_trader.db``, ``config/*.yaml``
and the live server on 127.0.0.1:8899 are never touched.  The manual itself is
read-only over ``docs/**``.
"""
from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web.manual as manual
from app.config import Config
from app.event_bus import EventBus
from core.auth.auth import AuthManager
from db.database import init_database

REPO_ROOT = Path(__file__).resolve().parents[1]
ADMIN = ("manualadmin", "Adm1nPass!")
VIEWER = ("manualviewer", "V1ewerPass!")

#: A document with a relative link, a table and stable headings.
KNOWN_DOC = "docs/core-algorithms/08-ml-triple-barrier.md"
KNOWN_LINK_TARGET = "/manual/docs/core-algorithms/10-volatility-targeting.md"


# ── fixtures ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def web_app():
    tmpdir = Path(tempfile.mkdtemp(prefix="bt_manual_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "manual.db")
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")

    async def _setup():
        await init_database(config.db_path)
        am = AuthManager(config.db_path, "manual-test-secret-32-bytes-min!!", 24)
        for username, password, role in ((ADMIN[0], ADMIN[1], "admin"),
                                         (VIEWER[0], VIEWER[1], "viewer")):
            await am.create_user(username, password, role, username)
        return am

    auth = asyncio.run(_setup())
    from web.server import create_app

    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    return app


def _login(web_app, creds):
    client = TestClient(web_app)
    response = client.post("/api/auth/login",
                           json={"username": creds[0], "password": creds[1]})
    assert response.status_code == 200, response.text
    return client


@pytest.fixture(scope="module")
def admin_client(web_app):
    return _login(web_app, ADMIN)


@pytest.fixture(scope="module")
def viewer_client(web_app):
    return _login(web_app, VIEWER)


def _expected_documents() -> set[str]:
    """Independent restatement of the documented discovery rule."""
    expected = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "docs").rglob("*.md")
        if not any(part.startswith(".")
                   for part in path.relative_to(REPO_ROOT).parts)
    }
    expected |= {"README.md", "README_EN.md"}
    return expected


# ── tree completeness ────────────────────────────────────────────────────
def test_tree_contains_every_docs_markdown_file(admin_client):
    expected = _expected_documents()
    payload = admin_client.get("/api/manual/tree").json()
    listed = {doc["path"] for doc in payload["documents"]}
    missing = sorted(expected - listed)
    extra = sorted(listed - expected)
    assert not missing, f"{len(missing)} document(s) missing from the manual: {missing[:5]}"
    assert not extra, f"the manual lists non-documents: {extra[:5]}"
    assert payload["doc_count"] == len(expected) == len(listed)
    # every document carries a title taken from its own first heading
    for doc in payload["documents"]:
        assert doc["title"].strip(), doc
        assert not doc["title"].startswith("#"), doc
        assert doc["url"] == f"/manual/{doc['path']}"


def test_tree_is_grouped_by_directory(admin_client):
    payload = admin_client.get("/api/manual/tree").json()
    labels = {root["label"] for root in payload["roots"]}
    assert any("Root" in label for label in labels)
    assert any("Handover" in label for label in labels)
    groups = {entry["path"]: entry for entry in payload["dirs"]}
    for expected in ("docs", "docs/audit", "docs/core-algorithms",
                     "docs/overhaul", "docs/research", "docs/superpowers",
                     "docs/superpowers/plans", "docs/superpowers/specs"):
        assert expected in groups, f"missing tree group {expected}"
        assert groups[expected]["label"]
        assert groups[expected]["count"] > 0
    # the labels name the directories of the requested grouping
    for path, needle in (("docs/overhaul", "Overhaul"),
                         ("docs/core-algorithms", "Core algorithms"),
                         ("docs/research", "Research"),
                         ("docs/audit", "Audit")):
        assert needle in groups[path]["label"], groups[path]


def test_discovery_rule_excludes_hidden_symlinked_and_non_markdown(tmp_path):
    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    (docs / ".hidden-dir").mkdir()
    (docs / "a.md").write_text("# A\n", encoding="utf-8")
    (docs / "sub" / "b.md").write_text("# B\n", encoding="utf-8")
    (docs / ".hidden.md").write_text("# Hidden\n", encoding="utf-8")
    (docs / ".hidden-dir" / "c.md").write_text("# C\n", encoding="utf-8")
    (docs / "notes.txt").write_text("not markdown\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\n", encoding="utf-8")
    link = docs / "linked.md"
    symlinked = False
    try:
        link.symlink_to(outside)
        symlinked = True
    except (OSError, NotImplementedError):  # Windows without symlink privilege
        symlinked = False
    (tmp_path / "README.md").write_text("# Repo\n", encoding="utf-8")

    found = manual.discover_documents(root=docs, root_files=("README.md",),
                                      repo_root=tmp_path)
    assert found == ["README.md", "docs/a.md", "docs/sub/b.md"]
    if symlinked:
        assert "docs/linked.md" not in found, "symlinked markdown is not a manual document"


def test_resolver_only_accepts_discovered_documents():
    assert manual.resolve_document("docs/HANDOVER.md") == (REPO_ROOT / "docs/HANDOVER.md")
    assert manual.resolve_document("README.md") == (REPO_ROOT / "README.md")
    for rejected in ("docs/overhaul/route-baseline.json",   # not markdown
                     "experimental/ml/README.md",            # markdown outside docs/
                     "docs",                                 # a directory
                     "docs/does-not-exist.md"):
        assert manual.resolve_document(rejected) is None, rejected


# ── rendering ────────────────────────────────────────────────────────────
def test_known_document_renders_with_its_title(admin_client):
    title = manual.document_title(REPO_ROOT / "docs/HANDOVER.md")
    response = admin_client.get("/manual/docs/HANDOVER.md")
    assert response.status_code == 200, response.text
    assert title in response.text
    assert 'id="manual-content"' in response.text
    # the root READMEs are documents too
    assert admin_client.get("/manual/README.md").status_code == 200
    assert admin_client.get("/manual/README_EN.md").status_code == 200


def test_markdown_tables_and_code_blocks_render(admin_client):
    html = admin_client.get("/manual/README.md").text
    assert "<table>" in html and "<th>" in html
    assert "<pre><code" in html


def test_markdown_link_rewrites_a_relative_doc_link(admin_client):
    html = admin_client.get(f"/manual/{KNOWN_DOC}").text
    assert f'href="{KNOWN_LINK_TARGET}"' in html
    # ... and the raw relative target is gone (it would 404 from /manual/...)
    assert 'href="10-volatility-targeting.md"' not in html


def test_markdown_link_rewrites_parent_and_cross_directory_links(admin_client):
    audit = admin_client.get("/manual/docs/audit/README.md").text
    assert 'href="/manual/docs/development-roadmap.md"' in audit
    # a directory link becomes a jump to that group of the manual tree
    assert 'href="/manual#group-docs-core-algorithms"' in audit

    web_split = admin_client.get("/manual/docs/overhaul/WEB_SPLIT.md").text
    assert 'href="/manual/README.md' in web_split


def test_unknown_link_target_becomes_a_marked_dead_link(admin_client):
    """README links to route-baseline.json / experimental/ml/README.md — neither
    is part of the manual, so the link must not navigate anywhere."""
    html = admin_client.get("/manual/README.md").text
    assert "manual-dead-link" in html
    assert 'href="/manual/docs/overhaul/route-baseline.json"' not in html
    assert 'href="docs/overhaul/route-baseline.json"' not in html


def test_every_rewritten_link_points_at_a_real_document():
    """No rendered document may contain a /manual/... href that would 404."""
    documents = set(manual.discover_documents())
    checked = 0
    for rel in sorted(documents):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        html = manual.render_markdown(text, rel)["html"]
        for href in re.findall(r'href="(/manual/[^"#]*)"', html):
            target = href[len("/manual/"):]
            assert target in documents, f"{rel} links to a missing document: {href}"
            checked += 1
        for href in re.findall(r'href="([^"]*\.md[^"]*)"', html):
            assert href.startswith(("/manual/", "http://", "https://")), \
                f"{rel} kept an unresolvable markdown link: {href}"
    assert checked > 0


def test_math_blocks_survive_for_katex(admin_client):
    response = admin_client.get("/manual/docs/research/CORE_ALGORITHMS.md")
    assert response.status_code == 200
    html = response.text
    assert 'class="math-block"' in html
    # the LaTeX body is intact (not markdown-escaped into something else) ...
    assert r"\text{train}" in html or r"\text{" in html
    assert r"\Delta" in html
    # ... and KaTeX is wired from the CDN, with a graceful fallback note
    assert "katex" in html
    assert "katex-unavailable" in html


def test_headings_get_ids_and_a_table_of_contents(admin_client):
    html = admin_client.get("/manual/README.md").text
    assert "本页目录" in html                      # in-page TOC block
    assert 'id="1-环境约束必读"' in html           # GitHub-style heading slug
    assert 'href="#1-环境约束必读"' in html


def test_render_cache_is_keyed_on_file_identity():
    """Only the rendered HTML is cached, keyed by (path, mtime_ns, size)."""
    manual.clear_caches()
    first = manual.render_document("docs/HANDOVER.md")
    second = manual.render_document("docs/HANDOVER.md")
    assert first is second
    stat = (REPO_ROOT / "docs/HANDOVER.md").stat()
    assert (str(REPO_ROOT / "docs/HANDOVER.md"), stat.st_mtime_ns, stat.st_size) \
        in manual._render_cache
    manual.clear_caches()
    assert manual.render_document("docs/HANDOVER.md") is not first


# ── 404 handling ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("doc_path", [
    "docs/does-not-exist.md",
    "docs",
    "docs/overhaul/route-baseline.json",
    "experimental/ml/README.md",
    "docs/HANDOVER.txt",
    "docs/HANDOVER.md/",
    "README.md.bak",
])
def test_missing_documents_are_404(admin_client, doc_path):
    response = admin_client.get(f"/manual/{doc_path}")
    assert response.status_code == 404, f"{doc_path} -> {response.status_code}"
    assert "Not found" in response.text


# ── path safety ──────────────────────────────────────────────────────────
TRAVERSAL_STRINGS = [
    "../config/secrets.yaml",
    "../../config/secrets.yaml",
    "docs/../../config/secrets.yaml",
    "docs/../.env",
    "..%2f..%2fconfig%2fsecrets.yaml",
    "%2e%2e%2fconfig%2fsecrets.yaml",
    "%2e%2e/%2e%2e/config/secrets.yaml",
    "%252e%252e%252fconfig%252fsecrets.yaml",      # double-encoded
    "..\\..\\config\\secrets.yaml",                # windows separator
    "/config/secrets.yaml",                        # absolute
    "C:/Windows/win.ini",                          # drive letter
    "//config/secrets.yaml",                       # empty segment
    "docs//HANDOVER.md",                           # empty segment
    "docs/./HANDOVER.md",                          # dot segment
    "~/.ssh/id_rsa",
    "docs/HANDOVER.md\x00.txt",                    # NUL byte
]


@pytest.mark.parametrize("hostile", TRAVERSAL_STRINGS)
def test_path_traversal_is_rejected_by_the_resolver(hostile):
    assert manual.resolve_document(hostile) is None, hostile


TRAVERSAL_URLS = [
    "/manual/../config/secrets.yaml",
    "/manual/../../config/secrets.yaml",
    "/manual/docs/../.env",
    "/manual/docs/../../config/secrets.yaml",
    "/manual/%2e%2e%2fconfig%2fsecrets.yaml",
    "/manual/%2e%2e%2f%2e%2e%2fconfig%2fsecrets.yaml",
    "/manual/..%2f..%2fconfig%2fsecrets.yaml",
    "/manual/%252e%252e%252fconfig%252fsecrets.yaml",
    "/manual//config/secrets.yaml",
    "/manual/C:/Windows/win.ini",
    "/manual/README.md/../../.env",
]


@pytest.mark.parametrize("url", TRAVERSAL_URLS)
def test_path_traversal_urls_never_serve_a_file(admin_client, url):
    response = admin_client.get(url, follow_redirects=False)
    assert response.status_code == 404, f"{url} -> {response.status_code}"
    body = response.text.lower()
    for needle in ("api_key", "api_secret", "jwt_secret", "deepseek"):
        assert needle not in body, f"{url} leaked '{needle}'"


def test_traversal_does_not_escape_the_repository_root():
    """A path that stays *inside* the repo but outside the allowed roots is out."""
    for rejected in ("config/config.yaml", "core/ga/genome.py", "web/server.py"):
        assert manual.resolve_document(rejected) is None, rejected


# ── authorization ────────────────────────────────────────────────────────
def test_manual_pages_require_authentication(web_app):
    anonymous = TestClient(web_app)
    for url in ("/manual", "/manual/README.md", "/manual/docs/HANDOVER.md"):
        response = anonymous.get(url, follow_redirects=False)
        assert response.status_code == 302, f"{url} -> {response.status_code}"
        assert response.headers["location"] == "/login"


def test_manual_tree_api_requires_authentication(web_app):
    response = TestClient(web_app).get("/api/manual/tree")
    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}


@pytest.mark.parametrize("creds", [ADMIN, VIEWER])
def test_authenticated_roles_can_read_the_manual(viewer_client, admin_client, creds):
    client = admin_client if creds is ADMIN else viewer_client
    assert client.get("/manual").status_code == 200
    assert client.get("/manual/README.md").status_code == 200
    assert client.get("/api/manual/tree").status_code == 200


# ── JSON shape ───────────────────────────────────────────────────────────
def test_json_tree_endpoint_shape(admin_client):
    payload = admin_client.get("/api/manual/tree").json()
    assert set(payload) == {"roots", "documents", "dirs", "doc_count",
                            "roots_allowed", "rule"}
    assert isinstance(payload["doc_count"], int) and payload["doc_count"] > 0
    assert payload["roots"] and all(node["type"] == "dir" for node in payload["roots"])
    assert "docs" in payload["roots_allowed"]
    assert "README.md" in payload["roots_allowed"]
    assert "docs/**/*.md" in payload["rule"]

    paths = [doc["path"] for doc in payload["documents"]]
    assert paths == sorted(paths), "documents must be in a stable (sorted) order"
    for doc in payload["documents"]:
        assert set(doc) == {"type", "name", "path", "title", "url", "group",
                            "group_label", "anchor"}
        assert doc["type"] == "doc"
        assert doc["path"].endswith(".md")
        assert doc["url"] == f"/manual/{doc['path']}"
        assert doc["title"] and not doc["title"].startswith("#")

    def walk(node):
        assert node["type"] in ("dir", "doc")
        if node["type"] == "dir":
            assert node["count"] >= len(node["children"])
            assert node["anchor"].startswith("group-")
            for child in node["children"]:
                walk(child)
        else:
            assert node["url"].startswith("/manual/")

    for node in payload["roots"]:
        walk(node)
    assert sum(node["count"] for node in payload["roots"]) == payload["doc_count"]


def test_code_fences_are_preserved_and_math_inside_them_is_not_extracted():
    """Regression: prose/fence chunks must keep their separating newlines.

    Joining them without a separator glued "intro:" onto the opening fence, so
    the block was no longer recognised as code (and `$$..$$` inside a fence was
    protected as math).
    """
    text = "intro:\n\n```python\nx = 1  # $$not math$$\n```\n\noutro $r_t$ text\n"
    html = manual.render_markdown(text, "docs/HANDOVER.md")["html"]
    assert "<p>intro:</p>" in html
    assert "<pre><code" in html
    assert "$$not math$$" in html                       # fence content untouched
    assert html.count('<span class="math-inline">') == 1  # only the real math
    assert "<p>outro" in html


def test_cross_document_fragment_is_validated():
    """A stale cross-document ``#fragment`` is dropped (the link survives); a
    valid one is rewritten to the target's canonical heading id."""
    text = ("[ok](../../README.md#7-文档索引) "
            "[stale](../../README.md#7-heading-that-does-not-exist) "
            "[doc](../core-algorithms/08-ml-triple-barrier.md)\n")
    html = manual.render_markdown(text, "docs/overhaul/PLAN.md")["html"]
    assert 'href="/manual/README.md#7-文档索引"' in html
    assert 'href="/manual/README.md"' in html
    assert 'href="/manual/docs/core-algorithms/08-ml-triple-barrier.md"' in html


def test_raw_html_inside_a_document_is_escaped_not_executed(admin_client):
    """`docs/overhaul/CHANGELOG.md` documents an XSS probe (`<img onerror>`):
    the manual must show it as text (markdown-it runs with html=False and the
    document-embedded HTML is escaped before it reaches the page)."""
    html = admin_client.get("/manual/docs/overhaul/CHANGELOG.md").text
    assert "&lt;img onerror&gt;" in html
    assert "<img onerror" not in html


def test_manual_is_reachable_from_the_main_navigation(admin_client):
    """The base template carries the 手册 nav entry on every page."""
    for url in ("/manual", "/alerts", "/market"):
        assert 'href="/manual"' in admin_client.get(url).text, url


def test_landing_page_lists_the_main_documents(admin_client):
    html = admin_client.get("/manual").text
    tree_docs = {doc["path"] for doc in admin_client.get("/api/manual/tree").json()["documents"]}
    for doc in manual.main_documents(manual.build_tree()):
        assert doc["path"] in tree_docs
        assert f'href="/manual/{doc["path"]}"' in html
    assert "主要文档" in html
    assert f'href="{manual.TREE_ENDPOINT}"' in html
