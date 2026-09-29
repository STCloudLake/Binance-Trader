"""Application context shared by every ``web.routes.*`` module.

``create_app()`` builds one :class:`AppContext` and passes it to each route
module's ``register(app, ctx)`` function.  The context carries exactly the
values that used to be closure variables inside ``create_app()`` (``config``,
``event_bus``, ``app``, the balance lock, the backtest-run registry, ...), so
moving a handler into its own module does not change what it reads.

The GA / walk-forward / calibration subprocess helpers also live here because
they are shared by the GA endpoints and the backtest page.
"""
import asyncio
import logging
import subprocess
import time
from typing import Any

from fastapi import FastAPI

from app.event_bus import EventBus
from app.config import Config


class AppContext:
    """Closure-variable replacement for the former ``create_app()`` scope."""

    def __init__(self, config: Config, event_bus: EventBus, app: FastAPI):
        self.config = config
        self.event_bus = event_bus
        self.app = app
        self.logger = logging.getLogger(__name__)

        # Track running backtests for progress polling
        self._bt_runs: dict[str, dict] = {}

        self._balance_lock = asyncio.Lock()
        self._app_started_at = time.time()

    # ------------------------------------------------------------------
    # Live app lookups
    # ------------------------------------------------------------------
    # ``app.state.*`` is populated by ``app/main.py`` (or by tests) *after*
    # ``create_app()`` returns, so these must always be read from the live app
    # object rather than cached on the context.
    def state(self, name: str, default: Any = None) -> Any:
        return getattr(self.app.state, name, default)
