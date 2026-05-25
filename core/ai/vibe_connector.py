"""Vibe-Trading integration via MCP protocol.

Three collaboration scenarios:
1. Strategy research → live deployment
2. Live trade data → Shadow Account behavior diagnosis
3. AI collaborative decision-making via Multi-Agent Swarm
"""

import asyncio
import json
import subprocess
import csv
from pathlib import Path
from typing import Optional
import aiosqlite

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from db.database import get_db


class VibeTradingConnector:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self._running = False
        self._available = False
        self._vibe_command = "vibe-trading"

    async def start(self):
        self._available = self._check_installed()
        self._running = True
        self.event_bus.subscribe(EventType.AI_SUGGESTION, self._on_research_request)

    def _check_installed(self) -> bool:
        try:
            result = subprocess.run([self._vibe_command, "--version"],
                                    capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    async def _on_research_request(self, event: Event):
        pass  # Handle external research triggers

    def is_available(self) -> bool:
        return self._available

    # Scenario 1: Strategy Research → Live Deployment
    async def research_strategy(self, prompt: str) -> Optional[str]:
        """Send a natural-language trading idea to Vibe-Trading for backtesting."""
        if not self._available:
            return None

        try:
            proc = await asyncio.create_subprocess_exec(
                self._vibe_command, "run", "-p", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            return stdout.decode() if proc.returncode == 0 else None
        except (asyncio.TimeoutError, Exception):
            return None

    async def import_strategy_from_vibe(self, report_path: str) -> Optional[dict]:
        """Parse Vibe-Trading report and convert to Binance Trader strategy YAML."""
        path = Path(report_path)
        if not path.exists():
            return None
        return {"imported": True, "path": str(path), "status": "needs_review"}

    # Scenario 2: Live Trade Data → Shadow Account Analysis
    async def export_trades_for_shadow(self, output_path: str = None) -> Optional[str]:
        """Export trading history as CSV for Vibe-Trading Shadow Account analysis."""
        if output_path is None:
            output_path = str(Path(self.config.data_dir) / "trade_export.csv")

        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT * FROM trades WHERE status='closed' ORDER BY opened_at"
            )
            rows = await cursor.fetchall()
            if not rows:
                return None

            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows([dict(r) for r in rows])

            return output_path
        finally:
            await db.close()

    async def run_shadow_analysis(self, csv_path: str) -> Optional[str]:
        """Run Vibe-Trading Shadow Account analysis on exported trade data."""
        if not self._available:
            return None

        try:
            proc = await asyncio.create_subprocess_exec(
                self._vibe_command, "--upload", csv_path, "run", "-p",
                "Analyze my trading behavior, extract my shadow strategy, and compare it with my actual trades",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            return stdout.decode() if proc.returncode == 0 else stderr.decode()
        except (asyncio.TimeoutError, Exception) as e:
            return str(e)

    # Scenario 3: AI Collaborative Decision-Making
    async def consult_swarm(self, context: str) -> Optional[str]:
        """Consult Vibe-Trading's Multi-Agent Swarm for complex trading decisions."""
        if not self._available:
            return None

        try:
            proc = await asyncio.create_subprocess_exec(
                self._vibe_command, "run", "-p",
                f"Act as a multi-agent trading swarm. Analyze this context and provide collaborative advice: {context}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            return stdout.decode() if proc.returncode == 0 else stderr.decode()
        except (asyncio.TimeoutError, Exception) as e:
            return str(e)

    async def bench_alpha_zoo(self, symbol: str = "BTCUSDT", period: str = "2024-01-01_2025-01-01") -> Optional[str]:
        """Run Vibe-Trading's Alpha Zoo benchmark for strategy validation."""
        if not self._available:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                self._vibe_command, "alpha", "bench", "--zoo", "gtja191",
                "--universe", symbol, "--period", period, "--top", "10",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            return stdout.decode()
        except (asyncio.TimeoutError, Exception):
            return None

    async def stop(self):
        self._running = False
