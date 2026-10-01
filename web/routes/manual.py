"""Manual routes: the in-app documentation manual.

    GET /manual                  landing page (intro + doc tree + main documents)
    GET /manual/{doc_path:path}  one rendered document
    GET /api/manual/tree         JSON tree (sidebar/search data, tooling)

Authorization is the app-wide ``AuthMiddleware`` (``core/auth/auth.py``) — the
same middleware that protects ``/trade``, ``/strategies`` or ``/users``.  It
treats every path that is not in its public allow-list as private, so

* ``GET /manual`` (page) without a session -> ``302 /login``,
* ``GET /api/manual/tree`` without a session -> ``401 {"error": "Unauthorized"}``
  (everything under ``/api/`` is a JSON 401),

and nothing here widens that allow-list.  ``web/manual.py`` owns discovery,
path resolution and rendering; this module only wires HTTP to it.
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from web import manual
from web.rendering import _render


def register(app: FastAPI, ctx) -> None:
    config = ctx.config

    def _sidebar(request: Request, current_doc: str = "") -> dict:
        """Context shared by the landing page and a document page."""
        tree = manual.build_tree()
        return {
            "request": request,
            "current_page": "manual",
            "mode": config.mode,
            "tree": tree,
            "current_doc": current_doc,
            "dir_anchor": manual.dir_anchor,
            "tree_endpoint": manual.TREE_ENDPOINT,
            "manual_prefix": manual.MANUAL_PREFIX,
        }

    # ---- JSON tree -------------------------------------------------------
    @app.get("/api/manual/tree")
    async def manual_tree(request: Request):
        """Machine-readable doc tree (nested ``roots`` + flat ``documents``).

        Authenticated like every other ``/api/`` route (401 when anonymous).
        """
        return JSONResponse(manual.build_tree())

    # ---- landing page ----------------------------------------------------
    @app.get("/manual", response_class=HTMLResponse)
    async def manual_home(request: Request):
        """Manual landing page: intro, the full doc tree and the main documents."""
        tree = manual.build_tree()
        context = _sidebar(request)
        context.update({
            "main_documents": manual.main_documents(tree),
            "doc_count": tree["doc_count"],
            "dir_count": len([d for d in tree["dirs"] if d["path"]]),
        })
        return _render("manual.html", context)

    # ---- one document ----------------------------------------------------
    @app.get("/manual/{doc_path:path}", response_class=HTMLResponse)
    async def manual_document(request: Request, doc_path: str):
        """Render one document; 404 (rendered page) when it is not in the manual."""
        doc = manual.render_document(doc_path)
        if doc is None:
            context = _sidebar(request)
            context.update({
                "doc": None,
                "requested": doc_path,
                "main_documents": manual.main_documents(context["tree"]),
            })
            response = _render("manual_doc.html", context)
            response.status_code = 404
            return response

        tree = manual.build_tree()
        prev_doc, next_doc = manual.navigation(tree, doc["path"])
        context = _sidebar(request, current_doc=doc["path"])
        context.update({
            "doc": doc,
            "breadcrumbs": manual.breadcrumbs(doc["path"]),
            "prev_doc": prev_doc,
            "next_doc": next_doc,
        })
        return _render("manual_doc.html", context)
