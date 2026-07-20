# 审计报告 R9 — Database + Config + Alerts

**日期**: 2026-07-16
**审查文件**: `db/database.py`, `app/config.py`, `alerts/manager.py`, `alerts/rules.py`, `config/secrets.yaml`
**审查代码行数**: ~600 行
**审计方法**: SQL 注入检测 + 配置安全审查 + 告警系统可靠性分析

---

## 摘要

| 等级 | 数量 |
|------|------|
| Critical | 1 |
| High | 2 |
| Medium | 3 |
| Low | 2 |
| **总计** | **8** |

---

## 漏洞清单

| ID | 等级 | 文件:行 | 描述 |
|----|------|---------|------|
| R9-001 | **Critical** | database.py:263-270 | 余额非事务性操作 — DB 连接异常时可丢失余额状态 |
| R9-002 | High | database.py:20-31 | Schema 迁移使用 bare except + 无版本化 — 静默失败风险 |
| R9-003 | High | config.py:65-71 | secrets.yaml 无文件权限检查 — 可能被其他用户读取 |
| R9-004 | Medium | alerts/manager.py:28-38 | 每个事件评估所有规则 — 无事件类型预过滤 |
| R9-005 | Medium | alerts/manager.py:89-94 | WebSocket 广播 `put_nowait` 丢弃 — 客户端无感知 |
| R9-006 | Medium | config.py:46-57 | Config 单例模式在运行时修改无锁保护 |
| R9-007 | Low | database.py:108-109 | AlertManager 使用字符串拼接构建 LIKE 查询 |
| R9-008 | Low | alerts/rules.py | 告警规则 cooling 使用内存状态 — 重启丢失 |

---

## 详细分析

### R9-001 🔴 Critical — 余额操作非事务性（与 R4-001 联动）

**文件**: `db/database.py:263-270`

此问题与 R4-001 相同，但在数据库层面。`atomic_adjust_balance()` 使用 `asyncio.Lock` + 两次独立 DB 连接实现"原子性"，但不是真正的数据库事务。

详见 R4-001 的完整分析。此处补充：如果在 `load_sim_balance` 和 `save_sim_balance` 之间系统崩溃，余额状态丢失。

---

### R9-002 🟠 High — Schema 迁移静默失败

**文件**: `db/database.py:20-31`

```python
for col, default in [
    ("trader", "'manual'"), ("strategy_name", "''"), ...
]:
    try:
        await db.execute(f"ALTER TABLE trades ADD COLUMN {col} TEXT DEFAULT {default}")
        await db.commit()
    except Exception:
        pass  # ← 静默吞所有异常
```

**描述**: 数据库迁移使用 `bare except pass`：
1. **如果列已存在**: SQLite 抛出异常 → 被吞没 → 看起来正常（这是预期行为）
2. **如果 SQL 语法错误**: 异常被吞没 → 列未创建 → 后续 INSERT 使用该列时失败
3. **如果 DB 被锁定**: 异常被吞没 → 迁移跳过 → 下次启动可能再次失败

此外，没有 schema 版本号。无法知道数据库处于哪个版本，也无法执行有序迁移（如先创建列 A、再修改列 B）。

**修复建议**: 使用 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`（SQLite 3.35+）或检查 `PRAGMA table_info` 后再执行。添加 `schema_version` 表追踪迁移状态。

---

### R9-003 🟠 High — Secrets 文件权限无检查

**文件**: `app/config.py:65-71`

```python
secrets_path = PROJECT_ROOT / "config" / "secrets.yaml"
if secrets_path.exists():
    self._load_yaml("config/secrets.yaml")
```

**描述**: `secrets.yaml` 包含 Binance API Key/Secret 和 DeepSeek API Key。代码直接读取但**不检查文件权限**。在 Linux/macOS 上，如果文件权限为 0644（其他用户可读），任何系统用户可以读取密钥。

当前开发环境为 Windows，但部署到 Linux 服务器后此问题变得严重。

**修复建议**: 
```python
import stat
if secrets_path.exists():
    mode = secrets_path.stat().st_mode
    if mode & stat.S_IROTH or mode & stat.S_IWOTH:
        logger.error("secrets.yaml is readable/writable by others! Fix permissions (chmod 600)")
```

---

### R9-004 🟡 Medium — 告警规则无事件类型预过滤

**文件**: `alerts/manager.py:28-38`

```python
async def _on_any_event(self, event: Event):
    for rule in self._rules:
        try:
            if rule.evaluate(event_type, data):
                await self._fire_alert(rule, event_type, data)
```

**描述**: 每个事件（包括每秒多次的 MARKET_KLINE）都会触发所有规则的评估。虽然有 cooldown 机制限制告警频率，但规则评估本身的开销（字符串匹配、条件检查）是每次事件都要执行的。

如果用户添加了大量自定义告警规则，频繁的事件处理会显著增加 CPU 开销。

**修复建议**: 在规则中维护一个 `event_types` 集合，先按事件类型过滤再评估条件：
```python
if event_type not in rule.event_types:
    continue
```

---

### R9-005 🟡 Medium — WebSocket 广播静默丢弃

**文件**: `alerts/manager.py:89-94`

```python
for cid, queue in self._ws_clients.items():
    try:
        queue.put_nowait(payload)
    except asyncio.QueueFull:
        dead.append(cid)
```

**描述**: 如果客户端 WebSocket 队列满（200条），告警被静默丢弃。客户端不会收到任何通知告知它错过了告警。对于 Critical 级别的告警，这是不可接受的。

**修复建议**: 对 Critical 告警使用 `queue.put()` 阻塞等待（带超时），确保重要告警不丢失。

---

### R9-006 🟡 Medium — Config 单例无锁保护

**文件**: `app/config.py:46-57`

```python
class Config:
    _instance = None
    def __new__(cls, mode: str = "sim"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
```

**描述**: Config 是类级单例，但在 `update_signal_weights()` 和 `update_soft_params()` 中没有锁保护。如果 AI controller（后台 asyncio task）和 Web UI（HTTP handler）同时修改配置，可能存在竞态条件。

Python 的 GIL 使得简单的 attribute 赋值是原子的，但 `update_soft_params(**kwargs)` 中的循环（config.py:178-181）涉及多次赋值，不是原子操作。

---

### R9-007 🟢 Low — LIKE 查询使用字符串拼接

**文件**: `alerts/manager.py:119`

```python
query += " AND (" + " OR ".join(clauses) + ")"
```

虽然使用了 `?` 占位符（params 列表），但 SQL 结构是用字符串拼接构建的。拼接的元素来自 `alert_type` 参数的分解（`p.strip()`），再包装在 `%{p}%` 中通过参数传递。这实际上是安全的（参数通过 `?` 传递），但字符串拼接构建 SQL 的模式容易在未来引入注入漏洞。

---

### R9-008 🟢 Low — 告警 cooldown 使用内存状态

**文件**: `alerts/rules.py`

告警规则的 cooldown 计时使用 `rule.last_triggered` (内存中的时间戳)。系统重启后：
1. 所有 cooldown 重置 → 可能立即触发大量告警（"告警风暴"）
2. 在重启前刚刚触发的告警会被再次触发

**修复建议**: 将 `last_triggered` 持久化到 DB，重启后恢复 cooldown 状态。

---

## 累计统计

| 等级 | 累计 (R1-R8) | R9 | 累计 |
|------|-------------|-----|------|
| Critical | 8 | 1 | 9 |
| High | 21 | 2 | 23 |
| Medium | 29 | 3 | 32 |
| Low | 15 | 2 | 17 |
| **总计** | **73** | **8** | **81** |
