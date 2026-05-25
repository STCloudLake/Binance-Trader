import time
from dataclasses import dataclass, field


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
        for field_name, expected in self.condition.items():
            actual = data.get(field_name)
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
    import json
    import os
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
