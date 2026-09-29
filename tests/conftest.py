"""Suite-wide test isolation.

``db.database.DB_PATH`` is a *process-global*: ``init_database(path)`` overwrites
it and ``get_db()`` / ``db_connection()`` — read by most of ``web/routes/*`` —
use that global instead of ``config.db_path``.  Several fixtures (and the
``matcher_db`` fixture in ``tests/test_market_api.py``) point it at their own
temporary file, and those temp paths are deleted again (``tmp_path`` teardown,
``os.unlink``).  Without this guard the database a route touches depends on which
test ran *last* — the classic "fails in a full run, passes on its own" order
dependence.

Restoring the previous value after every test means each test starts from the
state its own fixture established, and no test can leave the suite pointing at
another test's (or the real ``data/binance_trader.db``) database.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _restore_global_db_path():
    import db.database as database

    previous = database.DB_PATH
    yield
    database.DB_PATH = previous
