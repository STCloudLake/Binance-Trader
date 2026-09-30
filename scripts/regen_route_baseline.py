"""Regenerate ``docs/overhaul/route-baseline.json`` from the live route table.

The original baseline was produced by a helper that lived in ``%TEMP%`` and is
gone.  This script is the checked-in replacement.

What it does
------------
1. Builds the FastAPI app exactly as production does (``web.server.create_app``)
   against a **throwaway** SQLite database and an empty temp ``config/``
   directory.  The live ``data/binance_trader.db``, ``config/*.yaml`` and
   ``strategies/`` are never opened for writing.
2. Enumerates every ``APIRoute`` (one entry per method+path) and every
   ``WebSocketRoute`` ("WEBSOCKET" + path).  Mounts (e.g. ``/static``) are not
   routes and are skipped.
3. Sorts the ``[METHOD, PATH]`` pairs and writes them in the exact format of the
   existing baseline (``json.dumps(..., indent=1)``, CRLF, no trailing newline).
4. Prints the old vs new count plus the ADDED / REMOVED lists, and exits 1 when
   a previously documented route disappeared (a removal is a release blocker).

    python scripts/regen_route_baseline.py            # write + report
    python scripts/regen_route_baseline.py --check    # report only, never write
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASELINE = PROJECT_ROOT / "docs" / "overhaul" / "route-baseline.json"


def build_app():
    """A fully-wired app on temp storage — no live DB, config or strategies."""
    from app.config import Config
    from app.event_bus import EventBus
    from core.auth.auth import AuthManager
    from db.database import init_database

    tmpdir = Path(tempfile.mkdtemp(prefix="route_baseline_"))
    Config._instance = None
    config = Config.load("sim")
    config.db_path = str(tmpdir / "route_baseline.db")
    config.config_dir = str(tmpdir / "config")
    (tmpdir / "config").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "risk_params.yaml", "secrets.yaml"):
        (tmpdir / "config" / name).write_text("{}\n", encoding="utf-8")

    async def _setup():
        await init_database(config.db_path)
        return AuthManager(config.db_path,
                           "route-baseline-secret-at-least-32-bytes", 24)

    auth = asyncio.run(_setup())
    from web.server import create_app

    app = create_app(config, EventBus(), auth)
    app.state.config = config
    app.state.auth_manager = auth
    return app


def enumerate_routes(app) -> list[list[str]]:
    from fastapi.routing import APIRoute
    from starlette.routing import WebSocketRoute

    entries: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path:
            continue
        if isinstance(route, APIRoute):
            for method in (route.methods or set()):
                entries.add((str(method).upper(), path))
        elif isinstance(route, WebSocketRoute):
            entries.add(("WEBSOCKET", path))
    return [list(pair) for pair in sorted(entries)]


def _load(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [list(r) for r in json.loads(path.read_text(encoding="utf-8"))]


def _write(path: Path, rows: list[list[str]]) -> None:
    # Byte-identical to the committed baseline: indent=1, CRLF, no trailing \n.
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rows, indent=1, ensure_ascii=False)
    with open(path, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report the diff without writing the file")
    parser.add_argument("--path", default=str(BASELINE))
    args = parser.parse_args()

    target = Path(args.path)
    old = _load(target)
    new = enumerate_routes(build_app())

    old_set = {tuple(r) for r in old}
    new_set = {tuple(r) for r in new}
    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)

    print(f"route baseline: {target}")
    print(f"old routes: {len(old_set)}")
    print(f"new routes: {len(new_set)}")
    print(f"added:   {len(added)}")
    for method, path in added:
        print(f"  + {method:<10} {path}")
    print(f"removed: {len(removed)}")
    for method, path in removed:
        print(f"  - {method:<10} {path}")

    changed = old != new
    if not args.check:
        _write(target, new)
        print("written" if changed else "written (no change)")
    else:
        print("check mode: not written")

    return 1 if removed else 0


if __name__ == "__main__":
    sys.exit(main())
