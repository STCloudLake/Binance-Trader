# Alerts Center Redesign (A+B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the passive alert log into an active dashboard with a rule engine for filtering noise, plus real-time WebSocket push.

**Architecture:** AlertManager becomes a rule engine that evaluates incoming events against configured rules. Matching alerts are persisted and broadcast via WebSocket. The frontend shows a categorized dashboard with live updates, rule management, and filtering.

**Tech Stack:** Python 3.11+, asyncio, FastAPI WebSocket, HTMX, SQLite

---

## File Map

| File | Role |
|---|---|
| `alerts/manager.py` | Rewrite: rule engine + WebSocket broadcast |
| `alerts/rules.py` | **New**: AlertRule dataclass, default rules, rule store |
| `config/alert_rules.json` | **New**: Persisted rule configurations |
| `web/templates/alerts.html` | Rewrite: dashboard layout with tabs, rules panel |
| `web/server.py` | Add WebSocket endpoint, rule API endpoints, filtered query |
| `app/event_bus.py` | Add `subscribe_all` helper |

---

### Task 1: AlertRule Dataclass + Default Rules

**Files:**
- Create: `binance_trader/alerts/rules.py`
- Create: `binance_trader/config/alert_rules.json`

- [ ] **Step 1: Create `alerts/rules.py`**

```python
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlertRule:
    name: str
    event_type: str              # EventType value, or "*" for all
    condition: dict              # {"field": "level", "op": "eq", "value": "critical"}
    level: str = "warning"       # alert level when triggered
    cooldown_seconds: int = 60   # min seconds between repeat triggers
    enabled: bool = True
    description: str = ""

    # Runtime state (not persisted)
    _last_fired: float = field(default=0.0, repr=False)
    _fire_count: int = field(default=0, repr=False)

    def evaluate(self, event_type: str, data: dict) -> bool:
        """Check if this rule matches the incoming event. Returns True if alert should fire."""
        if not self.enabled:
            return False

        # Event type filter
        if self.event_type != "*" and event_type != self.event_type:
            return False

        # Condition check
        for field, expected in self.condition.items():
            actual = data.get(field)
            if actual != expected:
                return False

        # Cooldown
        now = time.time()
        if now - self._last_fired < self.cooldown_seconds:
            return False

        self._last_fired = now
        self._fire_count += 1
        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "event_type": self.event_type,
            "condition": self.condition,
            "level": self.level,
            "cooldown_seconds": self.cooldown_seconds,
            "enabled": self.enabled,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AlertRule":
        return cls(
            name=d["name"],
            event_type=d["event_type"],
            condition=d.get("condition", {}),
            level=d.get("level", "warning"),
            cooldown_seconds=d.get("cooldown_seconds", 60),
            enabled=d.get("enabled", True),
            description=d.get("description", ""),
        )


DEFAULT_RULES = [
    AlertRule(
        name="熔断触发",
        event_type="risk.breach",
        condition={"event_type": "circuit_breaker_trip"},
        level="critical",
        cooldown_seconds=300,
        description="风控熔断器触发时立即通知",
    ),
    AlertRule(
        name="紧急止损",
        event_type="alert.trigger",
        condition={"type": "emergency_stop"},
        level="critical",
        cooldown_seconds=60,
        description="PositionGuard 强制执行紧急平仓",
    ),
    AlertRule(
        name="大额亏损",
        event_type="position.update",
        condition={},
        level="warning",
        cooldown_seconds=120,
        description="单笔平仓亏损超过 50 USDT（条件由代码内判断）",
    ),
    AlertRule(
        name="AI 建议待审",
        event_type="ai.suggestion",
        condition={"status": "pending"},
        level="info",
        cooldown_seconds=300,
        description="AI 生成新建议待审核",
    ),
    AlertRule(
        name="新闻紧急抓取",
        event_type="news.alert",
        condition={},
        level="warning",
        cooldown_seconds=120,
        description="检测到异常波动触发紧急新闻抓取",
    ),
    AlertRule(
        name="信号拒绝",
        event_type="alert.trigger",
        condition={"type": "signal_rejected"},
        level="warning",
        cooldown_seconds=300,
        description="交易信号被风控拒绝",
    ),
]


def load_rules(path: str) -> list[AlertRule]:
    import json, os
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [AlertRule.from_dict(r) for r in data]
    return list(DEFAULT_RULES)


def save_rules(rules: list[AlertRule], path: str):
    import json
    data = [r.to_dict() for r in rules]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 2: Create `config/alert_rules.json` with defaults**

```json
[
  {
    "name": "熔断触发",
    "event_type": "risk.breach",
    "condition": {"event_type": "circuit_breaker_trip"},
    "level": "critical",
    "cooldown_seconds": 300,
    "enabled": true,
    "description": "风控熔断器触发时立即通知"
  },
  {
    "name": "紧急止损",
    "event_type": "alert.trigger",
    "condition": {"type": "emergency_stop"},
    "level": "critical",
    "cooldown_seconds": 60,
    "enabled": true,
    "description": "PositionGuard 强制执行紧急平仓"
  },
  {
    "name": "AI 建议待审",
    "event_type": "ai.suggestion",
    "condition": {"status": "pending"},
    "level": "info",
    "cooldown_seconds": 300,
    "enabled": true,
    "description": "AI 生成新建议待审核"
  },
  {
    "name": "新闻紧急抓取",
    "event_type": "news.alert",
    "condition": {},
    "level": "warning",
    "cooldown_seconds": 120,
    "enabled": true,
    "description": "检测到异常波动触发紧急新闻抓取"
  },
  {
    "name": "信号拒绝",
    "event_type": "alert.trigger",
    "condition": {"type": "signal_rejected"},
    "level": "warning",
    "cooldown_seconds": 300,
    "enabled": true,
    "description": "交易信号被风控拒绝"
  }
]
```

- [ ] **Step 3: Verify rules load**

Run:
```
cd binance_trader && python -c "from alerts.rules import load_rules; r = load_rules('config/alert_rules.json'); print(f'Loaded {len(r)} rules: {[x.name for x in r]}')"
```
Expected: `Loaded 5 rules: ['熔断触发', '紧急止损', 'AI 建议待审', '新闻紧急抓取', '信号拒绝']`

- [ ] **Step 4: Commit**

---

### Task 2: AlertManager Rewrite — Rule Engine + WebSocket

**Files:**
- Modify: `binance_trader/alerts/manager.py` (full rewrite)

- [ ] **Step 1: Rewrite AlertManager**

```python
import asyncio
import aiosqlite
import json
import time
from pathlib import Path
from loguru import logger

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from alerts.rules import AlertRule, load_rules, save_rules, DEFAULT_RULES


class AlertManager:
    def __init__(self, config: Config, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self._running = False
        self._rules: list[AlertRule] = []
        self._ws_clients: dict[str, asyncio.Queue] = {}  # client_id -> message queue
        self._rules_path = str(Path(config.data_dir) / ".." / "config" / "alert_rules.json")

    async def start(self):
        self._running = True
        self._rules = load_rules(self._rules_path)
        # Subscribe to all event types for rule evaluation
        self.event_bus.subscribe_all(self._on_any_event)
        logger.info(f"AlertManager started with {len(self._rules)} rules, rules_path={self._rules_path}")

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
        message = rule.name
        if "reason" in data:
            message = f"{rule.name}: {data['reason']}"
        elif "message" in data:
            message = str(data["message"])[:200]
        elif "detail" in data:
            message = f"{rule.name}: {str(data['detail'])[:200]}"

        symbol = data.get("symbol", "")

        # Persist to DB
        db_path = self.config.db_path
        try:
            db = await aiosqlite.connect(db_path)
            await db.execute(
                "INSERT INTO alerts (level, type, message, symbol) VALUES (?,?,?,?)",
                (rule.level, event_type, message, symbol),
            )
            await db.commit()
            await db.close()
        except Exception as e:
            logger.warning(f"Failed to persist alert: {e}")

        # Broadcast to WebSocket clients
        alert_data = {
            "level": rule.level,
            "type": event_type,
            "message": message,
            "symbol": symbol,
            "rule": rule.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        await self._broadcast(alert_data)

    async def _broadcast(self, data: dict):
        """Push alert to all connected WebSocket clients."""
        dead = []
        for cid, queue in self._ws_clients.items():
            try:
                queue.put_nowait(json.dumps(data, ensure_ascii=False))
            except asyncio.QueueFull:
                dead.append(cid)
        for cid in dead:
            self.unregister_ws(cid)

    def register_ws(self, client_id: str) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=200)
        self._ws_clients[client_id] = queue
        logger.debug(f"WS client connected: {client_id}")
        return queue

    def unregister_ws(self, client_id: str):
        self._ws_clients.pop(client_id, None)
        logger.debug(f"WS client disconnected: {client_id}")

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
            query += " AND type = ?"
            params.append(alert_type)
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
```

- [ ] **Step 2: Verify AlertManager loads**

Run:
```
cd binance_trader && python -c "from alerts.manager import AlertManager; print('Import OK')"
```

- [ ] **Step 3: Commit**

---

### Task 3: EventBus — Add subscribe_all

**Files:**
- Modify: `binance_trader/app/event_bus.py`

- [ ] **Step 1: Add `subscribe_all` method to EventBus**

```python
    def subscribe_all(self, callback: Callable[[Event], Awaitable[None]]):
        """Subscribe to ALL event types. Used by AlertManager rule engine."""
        for event_type in EventType:
            self._subscribers[event_type].append(callback)
```

Insert after the `subscribe` method (line 45).

- [ ] **Step 2: Verify**

Run:
```
cd binance_trader && python -c "from app.event_bus import EventBus; eb = EventBus(); eb.subscribe_all(lambda e: None); print('subscribe_all OK')"
```

- [ ] **Step 3: Commit**

---

### Task 4: WebSocket Endpoint + Rule API

**Files:**
- Modify: `binance_trader/web/server.py`

- [ ] **Step 1: Add WebSocket endpoint after existing routes**

```python
    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/ws/alerts")
    async def ws_alerts(websocket: WebSocket):
        await websocket.accept()
        alert_mgr = getattr(app.state, "alert_manager", None)
        if not alert_mgr:
            await websocket.close()
            return

        client_id = str(id(websocket))
        queue = alert_mgr.register_ws(client_id)

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                    await websocket.send_text(msg)
                except asyncio.TimeoutError:
                    await websocket.send_text('{"ping": true}')
        except WebSocketDisconnect:
            pass
        finally:
            alert_mgr.unregister_ws(client_id)
```

- [ ] **Step 2: Add rule management API endpoints**

```python
    @app.get("/api/alert-rules")
    async def get_alert_rules():
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr:
            return {"rules": []}
        return {"rules": mgr.get_rules()}

    @app.post("/api/alert-rules/{index}/toggle")
    async def toggle_alert_rule(index: int):
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr or index >= len(mgr.get_rules()):
            return {"ok": False}
        rules = mgr.get_rules()
        new_enabled = not rules[index].get("enabled", True)
        await mgr.update_rule(index, {"enabled": new_enabled})
        return {"ok": True, "enabled": new_enabled}

    @app.post("/api/alert-rules/{index}/remove")
    async def remove_alert_rule(index: int):
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr:
            return {"ok": False}
        await mgr.remove_rule(index)
        return {"ok": True}
```

- [ ] **Step 3: Add filtered alert query endpoint**

```python
    @app.get("/api/alerts/filtered")
    async def get_filtered_alerts(limit: int = 50, level: str = None,
                                   type: str = None, search: str = None):
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr:
            return []
        return await mgr.get_alerts(limit=limit, level=level, alert_type=type, search=search)
```

- [ ] **Step 4: Wire alert_manager to app.state in create_app**

Find `create_app()` function, add after other state assignments:
```python
    app.state.alert_manager = alert_manager
```
(If not already present — check current state wiring.)

- [ ] **Step 5: Verify server starts**

Run syntax check and import check.

- [ ] **Step 6: Commit**

---

### Task 5: New Alerts UI Page

**Files:**
- Rewrite: `binance_trader/web/templates/alerts.html`

- [ ] **Step 1: Rewrite alerts.html with dashboard layout**

```html
{% extends "base.html" %}
{% block content %}
<h2 class="text-xl mb-4">预警中心</h2>

<!-- Count cards -->
<div class="grid grid-cols-4 gap-4 mb-4" id="alert-counts" hx-get="/api/alerts/counts" hx-trigger="every 15s">
    {% include "partials/alert_counts.html" %}
</div>

<!-- Category tabs + search -->
<div class="flex gap-2 mb-4 items-center" id="alert-tabs">
    <button class="px-3 py-1 rounded text-sm bg-slate-700 text-white" onclick="filterAlerts('')">全部</button>
    <button class="px-3 py-1 rounded text-sm bg-slate-800 text-slate-400" onclick="filterAlerts('risk.breach')">风控</button>
    <button class="px-3 py-1 rounded text-sm bg-slate-800 text-slate-400" onclick="filterAlerts('alert.trigger')">信号</button>
    <button class="px-3 py-1 rounded text-sm bg-slate-800 text-slate-400" onclick="filterAlerts('system')">系统</button>
    <button class="px-3 py-1 rounded text-sm bg-slate-800 text-slate-400" onclick="filterAlerts('news.alert')">新闻</button>
    <div class="flex-1"></div>
    <input type="text" placeholder="搜索..." class="bg-slate-800 border border-slate-600 rounded px-3 py-1 text-white text-sm w-48"
           oninput="filterAlerts(null, this.value)">
</div>

<!-- Alert list -->
<div id="alert-list" class="card max-h-[50vh] overflow-y-auto">
    {% include "partials/alert_list.html" %}
</div>

<!-- Rules panel -->
<div class="card mt-4">
    <div class="flex justify-between items-center mb-3">
        <h3 class="text-lg">预警规则</h3>
    </div>
    <div id="rules-list" hx-get="/partials/alert-rules" hx-trigger="every 30s">
        {% include "partials/alert_rules.html" %}
    </div>
</div>

<script>
// WebSocket connection for live alerts
var ws = null;
function connectWS() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws/alerts');
    ws.onmessage = function(e) {
        var data = JSON.parse(e.data);
        if (data.ping) return;
        // Prepend new alert to the list
        prependAlert(data);
    };
    ws.onclose = function() { setTimeout(connectWS, 3000); };
}
connectWS();

function prependAlert(data) {
    var list = document.getElementById('alert-list');
    var div = document.createElement('div');
    div.className = 'py-2 border-b border-slate-800 flex justify-between items-center';
    var levelColors = {critical: 'text-red-400', warning: 'text-yellow-400', info: 'text-blue-400'};
    var levelNames = {critical: '紧急', warning: '警告', info: '信息'};
    div.innerHTML = '<div><span class="text-xs ' + (levelColors[data.level] || '') + '">' +
        (levelNames[data.level] || data.level) + '</span>' +
        '<span class="text-xs text-slate-500 ml-2">' + (data.rule || data.type) + '</span>' +
        '<span class="text-sm ml-2">' + (data.message || '').substring(0, 120) + '</span></div>' +
        '<div class="text-xs text-slate-500">' + (data.timestamp || '') + '</div>';
    list.insertBefore(div, list.firstChild);
}

var currentType = '';
var currentSearch = '';
function filterAlerts(type, search) {
    if (type !== null && type !== undefined) currentType = type;
    if (search !== null && search !== undefined) currentSearch = search;
    // Update tab styles
    document.querySelectorAll('#alert-tabs button').forEach(function(btn, i) {
        var types = ['', 'risk.breach', 'alert.trigger', 'system', 'news.alert'];
        btn.className = 'px-3 py-1 rounded text-sm ' + (types[i] === currentType ? 'bg-slate-700 text-white' : 'bg-slate-800 text-slate-400');
    });
    // Reload list via HTMX
    var url = '/partials/alerts-filtered?limit=50';
    if (currentType) url += '&type=' + encodeURIComponent(currentType);
    if (currentSearch) url += '&search=' + encodeURIComponent(currentSearch);
    htmx.ajax('GET', url, '#alert-list');
}
</script>

<style>
.max-h-\[50vh\] { max-height: 50vh; }
</style>
{% endblock %}
```

- [ ] **Step 2: Create partial templates**

Create `binance_trader/web/templates/partials/alert_counts.html`:
```html
<div class="card border-red-700"><h3 class="text-lg text-red-400">紧急</h3><div class="text-3xl font-bold text-red-400" id="count-critical">{{ counts.critical }}</div></div>
<div class="card border-yellow-700"><h3 class="text-lg text-yellow-400">警告</h3><div class="text-3xl font-bold text-yellow-400" id="count-warning">{{ counts.warning }}</div></div>
<div class="card border-blue-700"><h3 class="text-lg text-blue-400">信息</h3><div class="text-3xl font-bold text-blue-400" id="count-info">{{ counts.info }}</div></div>
<div class="card border-slate-700"><h3 class="text-lg text-slate-400">活跃规则</h3><div class="text-3xl font-bold text-slate-400" id="active-rules-count">{{ rules_enabled }}</div></div>
```

Create `binance_trader/web/templates/partials/alert_list.html`:
```html
{% for alert in alerts %}
<div class="py-2 border-b border-slate-800 flex justify-between items-center">
    <div>
        <span class="text-xs {{ 'text-red-400' if alert.level == 'critical' else 'text-yellow-400' if alert.level == 'warning' else 'text-blue-400' }}">
            {{ {'critical':'紧急','warning':'警告','info':'信息'}.get(alert.level, alert.level) }}</span>
        <span class="text-xs text-slate-500 ml-2">{{ alert.type }}</span>
        <span class="text-sm ml-2">{{ alert.message[:150] }}</span>
        {% if alert.symbol %}<span class="text-xs text-sky-400 ml-2">{{ alert.symbol }}</span>{% endif %}
    </div>
    <div class="text-xs text-slate-500">{{ alert.created_at }}</div>
</div>
{% endfor %}
{% if not alerts %}
<div class="py-4 text-center text-slate-500">暂无告警</div>
{% endif %}
```

Create `binance_trader/web/templates/partials/alert_rules.html`:
```html
{% for rule in rules %}
<div class="flex items-center justify-between py-2 border-b border-slate-800">
    <div class="flex items-center gap-2">
        <span class="text-xs {{ 'text-green-400' if rule.enabled else 'text-slate-600' }}">{{ '✅' if rule.enabled else '⬜' }}</span>
        <span class="text-sm">{{ rule.name }}</span>
        <span class="text-xs text-slate-500">{{ rule.description }}</span>
    </div>
    <div class="flex items-center gap-2">
        <span class="badge {{ 'badge-red' if rule.level == 'critical' else 'badge-yellow' if rule.level == 'warning' else 'badge-blue' }} text-xs">{{ {'critical':'紧急','warning':'警告','info':'信息'}.get(rule.level, rule.level) }}</span>
        <button class="text-xs text-sky-400 hover:underline"
                hx-post="/api/alert-rules/{{ loop.index0 }}/toggle" hx-target="#rules-list">
            {{ '停用' if rule.enabled else '启用' }}</button>
    </div>
</div>
{% endfor %}
```

- [ ] **Step 3: Verify templates render**

Start app, navigate to `/alerts`, visually confirm layout.

- [ ] **Step 4: Commit**

---

### Task 6: Web Server — Wire Partial Routes + Counts

**Files:**
- Modify: `binance_trader/web/server.py`

- [ ] **Step 1: Add partials routes and counts endpoint**

```python
    @app.get("/partials/alerts-filtered")
    async def partials_alerts_filtered(request: Request, limit: int = 50,
                                        type: str = None, search: str = None):
        mgr = getattr(app.state, "alert_manager", None)
        alerts = await mgr.get_alerts(limit=limit, alert_type=type, search=search) if mgr else []
        return _render("partials/alert_list.html", {"request": request, "alerts": alerts})

    @app.get("/partials/alert-rules")
    async def partials_alert_rules(request: Request):
        mgr = getattr(app.state, "alert_manager", None)
        rules = mgr.get_rules() if mgr else []
        return _render("partials/alert_rules.html", {"request": request, "rules": rules, "enumerate": enumerate})

    @app.get("/api/alerts/counts")
    async def get_alert_counts():
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr:
            return {"counts": {"critical": 0, "warning": 0, "info": 0}, "rules_enabled": 0}
        counts = await mgr.get_counts()
        rules = mgr.get_rules()
        return {
            "counts": counts,
            "rules_enabled": sum(1 for r in rules if r.get("enabled")),
        }
```

- [ ] **Step 2: Wire alert_manager to app.state**

In `create_app`, ensure:
```python
    app.state.alert_manager = alert_manager
```

Check current wiring and add if missing.

- [ ] **Step 3: Update settings_page context for rules**

In the alerts page handler, pass rules to template:
```python
    @app.get("/alerts", response_class=HTMLResponse)
    async def alerts_page(request: Request):
        mgr = getattr(app.state, "alert_manager", None)
        alerts = await mgr.get_alerts(limit=50) if mgr else []
        rules = mgr.get_rules() if mgr else []
        counts = await mgr.get_counts() if mgr else {"critical": 0, "warning": 0, "info": 0}
        return _render("alerts.html", {
            "request": request, "current_page": "alerts",
            "alerts": alerts, "rules": rules,
            "counts": counts,
            "rules_enabled": sum(1 for r in rules if r.get("enabled")),
        })
```

- [ ] **Step 4: Commit**

---

### Task 7: Integration Verification

**Files:** None (verification only)

- [ ] **Step 1: Start app and verify**

```bash
cd binance_trader && python -m app.main --mode sim --port 8899
```

- [ ] **Step 2: Open http://127.0.0.1:8899/alerts**

Check:
- Count cards show correct numbers
- Alert list loads with historical data
- Rules panel shows default 5 rules
- Toggle a rule → immediately reflects
- WebSocket connects (check browser console)
- New alerts appear in real-time without page refresh

- [ ] **Step 3: Trigger a test alert and verify live push**

```python
# In a separate terminal:
import asyncio, aiosqlite
async def test():
    from app.event_bus import EventBus, Event, EventType
    from app.config import Config
    from alerts.manager import AlertManager
    config = Config.load('sim')
    eb = EventBus()
    await eb.start()
    mgr = AlertManager(config, eb)
    await mgr.start()
    await eb.publish(Event(EventType.ALERT_TRIGGER, {"level": "warning", "type": "test", "message": "测试告警"}))
    await asyncio.sleep(1)
    counts = await mgr.get_counts()
    print(f"Counts: {counts}")
asyncio.run(test())
```

Expected: counts include the test alert. WebSocket clients receive it.

- [ ] **Step 4: Commit**
