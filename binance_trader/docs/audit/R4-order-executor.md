# 审计报告 R4 — Order Executor + Balance Management

**日期**: 2026-07-16
**审查文件**: `core/executor/executor.py`, `db/database.py` (余额函数), `app/main.py` (exit/reduce handlers)
**审查代码行数**: ~410 行
**审计方法**: 余额操作原子性分析 + PNL 公式验证 + 调用链追踪 + 竞态条件分析

---

## 摘要

| 等级 | 数量 |
|------|------|
| Critical | 1 |
| High | 3 |
| Medium | 4 |
| Low | 1 |
| **总计** | **9** |

---

## 漏洞清单

| ID | 等级 | 文件:行 | 描述 |
|----|------|---------|------|
| R4-001 | **Critical** | database.py:263-270 | `atomic_adjust_balance` 非真正原子 — 读写在分离的 DB 连接中完成 |
| R4-002 | High | executor.py:79 | `order_id` 使用毫秒时间戳 — 同毫秒并发碰撞 |
| R4-003 | High | executor.py:101 | `_execute_sim` 直接覆盖同 symbol 已有仓位 — 静默丢失 |
| R4-004 | High | executor.py:244-245 | `close_position` 的 PNL 计算不含手续费 — 模拟与实盘结果不一致 |
| R4-005 | Medium | main.py:117-123 | exit/reduce handler 中 PNL + invested 使用了快照数据 |
| R4-006 | Medium | executor.py:38-68 | `restore_positions` 恢复的仓位无止损/止盈参数 |
| R4-007 | Medium | executor.py:174-175 | `_execute_live` 中 `order["price"]` 可能为 0 导致 fallback 不准确 |
| R4-008 | Medium | executor.py:158-213 | 实盘下单失败后无余额回滚逻辑 |
| R4-009 | Low | main.py:350-353 | 遗留仓位处理仅对初始余额判断 — restart 后不适用 |

---

## 详细分析

### R4-001 🔴 Critical — 余额调整非原子性

**文件**: `db/database.py:263-270`

```python
async def atomic_adjust_balance(delta: float, db_path: str = None) -> float:
    async with _balance_lock:
        current = await load_sim_balance(db_path)    # 打开连接1 → 读取 → 关闭
        new_balance = current + delta
        await save_sim_balance(new_balance, db_path)  # 打开连接2 → 写入 → 关闭
        return new_balance
```

**描述**: 虽然使用了 `asyncio.Lock` 防止并发读写，但这**不是数据库事务**。`load_sim_balance()` 和 `save_sim_balance()` 使用两个独立的数据库连接：

1. `load_sim_balance()` 打开连接 → `SELECT` → 关闭连接
2. `save_sim_balance()` 打开连接 → `INSERT OR REPLACE` → 关闭连接

如果在步骤1和步骤2之间发生异常（如 asyncio task 被取消），余额已经被内存修改但未持久化：
- DB 中的余额是旧值
- 下次 `load_sim_balance()` 返回旧值
- 但系统状态（position）可能已经是新状态 → **状态不一致**

此外，如果未来引入多进程架构，`asyncio.Lock` 无法跨进程保护。

**修复建议**: 使用单个数据库连接 + 事务：
```python
async with _balance_lock:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute("SELECT value FROM system_config WHERE key='sim_balance'")
        row = await cursor.fetchone()
        current = float(row["value"]) if row else DEFAULT_BALANCE
        new_balance = current + delta
        await db.execute("INSERT OR REPLACE INTO system_config ...")
        await db.commit()
        return new_balance
```

---

### R4-002 🟠 High — 订单 ID 碰撞

**文件**: `core/executor/executor.py:79`

```python
order_id = f"sim_{int(time.time() * 1000)}"
```

**描述**: `time.time() * 1000` 提供毫秒级时间戳。在同一毫秒内如果两个信号同时触发（来自不同策略对不同 symbol 的评估），会产生相同的 `order_id`。`self._orders[order_id] = order` 会导致前一个订单被静默覆盖。

虽然从 Python 单线程 asyncio 的角度看在 `await` 点之后时间已过，但以下流程可能在同一毫秒内：
- 同一 K 线触发多个策略 → 多个 `_evaluate()` 调用 → 多个 `STRATEGY_SIGNAL` 事件 → 多个 `_on_signal` 回调 → 多个 `ORDER_REQUEST` 事件
- 如果 RiskManager 快速连续批准，它们可能在同一毫秒到达 executor

**复现路径**: 同一 K 线触发 BTCUSDT 的长信号和 ETHUSDT 的长信号 → 两个 `_execute_sim` 几乎同时执行 → 同毫秒 order_id → 前一个订单被覆盖。

**修复建议**: 使用 `uuid.uuid4().hex[:12]` 或 `f"sim_{int(time.time()*1000)}_{symbol}_{random_suffix}"`。

---

### R4-003 🟠 High — 同 Symbol 仓位静默覆盖

**文件**: `core/executor/executor.py:101`

```python
self._positions[symbol] = {...}
```

**描述**: 如果同 symbol 已有仓位（可能因 restart 恢复、POSITION_UPDATE 事件延迟等），直接赋值覆盖会导致旧仓位信息完全丢失，包括：
- 旧的 trade_group（用于 DB 关联）
- 旧的 PNL 跟踪数据

虽然 RiskManager 在第138行用 `_pending_signals` 做了防护，但如果 `_pending_signals` 失效（见 R2-003），仍可能产生同 symbol 的重复入场。

**复现路径**: `_pending_signals` 未正确清除 → 同 symbol 信号再次通过 RiskManager → `_positions[symbol]` 被覆盖 → 旧仓位孤立（executor 记忆丢失但 DB 中有记录）。

**修复建议**: 在覆盖前检查并记录告警：
```python
if symbol in self._positions:
    logger.warning(f"Overwriting existing position for {symbol}")
    # Optionally force-close old position first
```

---

### R4-004 🟠 High — PNL 计算不含手续费

**文件**: `core/executor/executor.py:241`

```python
pnl = (current_price - entry) * close_qty if side == "long" else (entry - current_price) * close_qty
```

**描述**: 模拟盘中的 PNL 计算不包含交易手续费。这导致：
1. 模拟盘余额增长比实盘快（每笔交易多 ~0.08% 手续费差异）
2. 回测结果乐观偏差
3. 用户从模拟切换到实盘后体验差异

对比：回测引擎的 `EventDrivenExecutor._close_position` 包含完整的成本计算（event_executor.py:62-72）。

**复现路径**: 模拟盘中频繁交易 → 余额增长至 12000 → 实盘以相同策略运行 → 实际余额 11500（手续费消耗）。

**修复建议**: 复用回测引擎的 `apply_trading_costs()` 或 `cost_model.py` 中的成本计算逻辑。

---

### R4-005 🟡 Medium — Exit/reduce handler 快照数据不一致

**文件**: `app/main.py:117-123`

```python
async def _on_position_exit(event: Event):
    data = event.data
    result = await order_executor.close_position(data["symbol"], 100, data.get("price", 0))
    if result.get("ok"):
        invested_returned = result.get("invested_returned", 0)
        trade_pnl = result.get("pnl", 0)
        new_balance = await atomic_adjust_balance(invested_returned + trade_pnl, config.db_path)
```

**描述**: `event.data` 中的 `price` 是信号产生时的价格（可能几秒前），而 `close_position()` 实际执行时的价格可能已经变化。在模拟模式中，`close_position` 直接使用传入的 `current_price` 计算 PNL，所以使用的是**旧价格**。

对比手动平仓（通过 Web UI），用户总是以最新价格平仓。

**复现路径**: 信号产生时 BTC=100,000 → 几秒延迟后实际价格=100,200 → 但仍以 100,000 计算 PNL → 用户损失 200 点的收益。

**修复建议**: 在 handler 中重新获取当前价格：
```python
price = market_data.get_current_price(symbol) or data.get("price", 0)
```

---

### R4-006 🟡 Medium — 恢复仓位无风控参数

**文件**: `core/executor/executor.py:53-67`

```python
self._positions[symbol] = {
    "symbol": symbol, "side": r["side"], "quantity": qty,
    "entry_price": entry, "current_price": entry,
    "unrealized_pnl": 0, "stop_loss": None,  # ← 无止损
    ...
}
```

**描述**: 系统重启后从 DB 恢复的仓位：
1. `stop_loss = None` — 无止损保护
2. `current_price = entry` — unrealized_pnl 始终为 0，即使行情已大幅变动
3. 无 `take_profits` — 无止盈目标

这些仓位在重启后处于"裸奔"状态，仅依赖 emergency_stop (-5%) 作为最后保护。

**修复建议**: 恢复仓位后立即从 market_data 获取当前价格，重新计算 unrealized_pnl，并基于当前价格重新设置止损。

---

### R4-007 🟡 Medium — 实盘订单价格 fallback 不准确

**文件**: `core/executor/executor.py:174-175`

```python
price = float(order.get("price", data.get("price", 0)))
if price == 0:
    price = data.get("price", 0)
```

**描述**: 市价单的 `order["price"]` 在 Binance API 响应中通常为 0（因为是市价成交，不是限价）。代码 fallback 到 `data["price"]`（即下单前获取的价格）。这个价格可能与实际成交价格有显著偏差（滑点）。

对于模拟盘这是合理的近似，但对于实盘，应使用 `/api/v3/myTrades` 查询实际成交价格。

---

### R4-008 🟡 Medium — 实盘下单失败无余额回滚

**文件**: `core/executor/executor.py:206-213`

```python
except Exception as e:
    if attempt < 2:
        await asyncio.sleep(2 ** attempt)
    else:
        await self.event_bus.publish(Event(EventType.ALERT_TRIGGER, {...}))
```

**描述**: 实盘下单3次重试全部失败后，仅发布告警。但在此之前，如果余额已在某处被扣除（或 position 已在内存中创建），不会回滚。

当前代码中，position 在 `_execute_live` 的成功路径中创建（line 178）。如果前两次重试期间 position 被外部创建，第三次失败后不会清理。

**复现路径**: 网络暂时不可用 → 3次重试全部失败 → 如果 balance 已被预扣 → 余额不一致。

---

### R4-009 🟢 Low — 遗留仓位仅首次适用

**文件**: `app/main.py:350-355`

```python
if web_app.state.balance == DEFAULT_BALANCE and total_invested > 0 and total_invested < web_app.state.balance:
    web_app.state.balance -= total_invested
```

**描述**: 遗留仓位处理仅在余额恰好等于 `DEFAULT_BALANCE (10000)` 时生效。如果用户非首次运行（余额已被修改为非默认值），这段逻辑永远不会触发。这是正确的设计（只在首次运行时调整），但条件依赖于一个可修改的变量，增加了脆弱性。

---

## 累计统计

| 等级 | R1 | R2 | R3 | R4 | 累计 |
|------|-----|-----|-----|-----|------|
| Critical | 1 | 1 | 2 | 1 | 5 |
| High | 4 | 3 | 1 | 3 | 11 |
| Medium | 5 | 3 | 2 | 4 | 14 |
| Low | 3 | 2 | 1 | 1 | 7 |
| **总计** | **13** | **9** | **6** | **9** | **37** |
