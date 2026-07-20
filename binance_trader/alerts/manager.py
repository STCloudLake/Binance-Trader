import asyncio
import aiosqlite
import json
import time
from pathlib import Path
from loguru import logger

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from alerts.rules import AlertRule, load_rules, save_rules


class AlertManager:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self._running = False
        self._rules: list[AlertRule] = []
        self._ws_clients: dict[str, asyncio.Queue] = {}
        self._rules_path = str(Path(config.data_dir).parent / "config" / "alert_rules.json")

    async def start(self):
        self._running = True
        self._rules = load_rules(self._rules_path)
        self.event_bus.subscribe_all(self._on_any_event)
        logger.info(f"AlertManager started with {len(self._rules)} rules")

    async def _on_any_event(self, event: Event):
        """Evaluate incoming event against all rules."""
        event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
        data = event.data or {}

        for rule in self._rules:
            try:
                if rule.evaluate(event_type, data):
                    await self._fire_alert(rule, event_type, data)
            except Exception:
                pass

    async def _fire_alert(self, rule: AlertRule, event_type: str, data: dict):
        """Persist alert and broadcast to WebSocket clients."""
        message = data.get("message", "") or data.get("detail", "") or rule.name
        if isinstance(message, str):
            message = message[:200]
        else:
            message = str(message)[:200]

        symbol = data.get("symbol", "")

        # Persist to DB
        try:
            db = await aiosqlite.connect(self.config.db_path)
            await db.execute(
                "INSERT INTO alerts (level, type, message, symbol) VALUES (?,?,?,?)",
                (rule.level, event_type, message, symbol),
            )
            await db.commit()
            await db.close()
        except Exception as e:
            logger.warning(f"Failed to persist alert: {e}")

        # Translate rule name for display
        rule_name = rule.name
        try:
            from web.i18n import get_translator as _gt
            from app.config import Config as _Cfg
            c = _Cfg._instance
            lang = getattr(c, "language", "zh") if c and c._loaded else "zh"
            rule_name = _gt(lang)(rule.name)
        except Exception:
            pass

        # Broadcast to WebSocket clients
        alert_data = {
            "level": rule.level,
            "type": event_type,
            "message": message,
            "symbol": symbol,
            "rule": rule_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        await self._broadcast(alert_data)

    async def _broadcast(self, data: dict):
        """Push alert to all connected WebSocket clients."""
        dead = []
        payload = json.dumps(data, ensure_ascii=False)
        for cid, queue in self._ws_clients.items():
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(cid)
        for cid in dead:
            self.unregister_ws(cid)

    def register_ws(self, client_id: str) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=200)
        self._ws_clients[client_id] = queue
        return queue

    def unregister_ws(self, client_id: str):
        self._ws_clients.pop(client_id, None)

    async def get_alerts(self, limit: int = 50, level: str = None, alert_type: str = None,
                         search: str = None) -> list[dict]:
        db = await aiosqlite.connect(self.config.db_path)
        db.row_factory = aiosqlite.Row
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        if level:
            query += " AND level = ?"
            params.append(level)
        if alert_type:
            # Support comma-separated LIKE patterns: "signal,risk" → type LIKE %signal% OR type LIKE %risk%
            patterns = [p.strip() for p in alert_type.split(",") if p.strip()]
            if patterns:
                clauses = ["type LIKE ?" for _ in patterns]
                query += " AND (" + " OR ".join(clauses) + ")"
                params.extend([f"%{p}%" for p in patterns])
        if search:
            query += " AND (message LIKE ? OR symbol LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        rows = [dict(r) for r in await cursor.fetchall()]
        await db.close()
        return rows

    async def get_counts(self) -> dict:
        db = await aiosqlite.connect(self.config.db_path)
        counts = {}
        for level in ("critical", "warning", "info"):
            cursor = await db.execute(
                "SELECT COUNT(*) FROM alerts WHERE level = ?", (level,))
            row = await cursor.fetchone()
            counts[level] = row[0] if row else 0
        await db.close()
        return counts

    async def acknowledge_alert(self, alert_id: int):
        db = await aiosqlite.connect(self.config.db_path)
        await db.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
        await db.commit()
        await db.close()

    def get_rules(self) -> list[dict]:
        return [r.to_dict() for r in self._rules]

    async def update_rule(self, index: int, updates: dict):
        if 0 <= index < len(self._rules):
            rule = self._rules[index]
            for k, v in updates.items():
                if hasattr(rule, k):
                    setattr(rule, k, v)
            save_rules(self._rules, self._rules_path)

    async def add_rule(self, rule_dict: dict):
        rule = AlertRule.from_dict(rule_dict)
        self._rules.append(rule)
        save_rules(self._rules, self._rules_path)
        return len(self._rules) - 1

    async def remove_rule(self, index: int):
        if 0 <= index < len(self._rules):
            self._rules.pop(index)
            save_rules(self._rules, self._rules_path)

    async def stop(self):
        self._running = False
