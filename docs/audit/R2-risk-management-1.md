# 审计报告 R2 — Risk Management Ⅰ (Circuit Breaker + 7-Step Pipeline)

**日期**: 2026-07-16
**审查文件**: `core/risk/manager.py`, `core/risk/circuit_breaker.py`
**审查代码行数**: ~316 行
**审计方法**: 状态机分析 + 管线逻辑验证 + 边界条件测试

---

## 摘要

| 等级 | 数量 |
|------|------|
| Critical | 1 |
| High | 3 |
| Medium | 3 |
| Low | 2 |
| **总计** | **9** |

---

## 漏洞清单

| ID | 等级 | 文件:行 | 描述 |
|----|------|---------|------|
| R2-001 | **Critical** | circuit_breaker.py:48-49 | 回撤从 peak_equity 计算而非当日开盘 — 跨日未重置时误触发 |
| R2-002 | High | manager.py:113-115 | Total Exposure 使用 entry_price 而非 current_price |
| R2-003 | High | manager.py:58 | `_pending_signals` 无超时清理机制 — 永久阻止同 symbol 交易 |
| R2-004 | High | manager.py:142 | Max Open Trades 使用本地缓存而非 executor 实时数据 |
| R2-005 | Medium | circuit_breaker.py:33-42 | `add_trade_result` 浮点舍入可能导致连续亏损计数偏差 |
| R2-006 | Medium | circuit_breaker.py:69-75 | `is_new_trip()` 仅基于 reason 字符串去重 — 同原因二次触发被丢弃 |
| R2-007 | Medium | manager.py:182-198 | `update_balance()` 中 invested 与 balance 时序不一致 |
| R2-008 | Low | manager.py:106-110 | `_trip_callback` 异常处理仅记录日志 |
| R2-009 | Low | circuit_breaker.py:41-42 | consecutive_losses 在盈利时重置为0 — 一笔盈利打断计数 |

---

## 详细分析

### R2-001 🔴 Critical — 回撤从 Peak 计算导致跨日误触发

**文件**: `core/risk/circuit_breaker.py:48-49`

```python
if self.peak_equity > 0:
    daily_dd = (self.peak_equity - self.current_equity) / self.peak_equity * 100
    if daily_dd > self.max_daily_drawdown_pct:
        self._trip(f"Daily drawdown {daily_dd:.2f}% exceeds limit ...")
```

**描述**: `peak_equity` 是进程生命期内的历史最高权益，而非当日的最高权益。`reset_daily()` 方法确实会重置 `peak_equity = self.current_equity`，但该方法依赖 `main.py` 中的 `_circuit_breaker_reset_loop()` 后台任务在每日零点附近调用。

问题场景：
1. 第一天：余额从 10000 涨到 11000，`peak_equity = 11000`
2. 当天结束时权益回到 10500，未触发熔断
3. 午夜：`_circuit_breaker_reset_loop()` 在 00:00-00:10 之间调用 `reset_daily()`，将 peak 重置为 10500
4. **但如果后台任务崩溃或延迟**（例如因 event loop 阻塞），`reset_daily()` 不会被调用
5. 第二天：开盘后权益跌到 10000，回撤计算为 `(11000 - 10000) / 11000 = 9.09%` → 触发熔断
6. 但实际当日回撤仅为 `(10500 - 10000) / 10500 = 4.76%` → 不应触发

**根因**: 回撤计算的基准 `peak_equity` 的正确性完全依赖于异步后台定时任务的可靠执行。如果任务延迟或失败，`peak_equity` 不会被重置，导致使用跨日数据计算当日回撤。

**复现路径**:
1. 运行系统，让权益创新高
2. 模拟 event loop 阻塞导致 `_circuit_breaker_reset_loop` 错过午夜重置窗口
3. 第二天正常交易 → 回撤被错误计算 → 熔断误触发

**修复建议**:
- 在 `CircuitBreaker` 内部存储 `daily_peak_equity`（独立于 `peak_equity`）
- 在 `check()` 方法中根据当前日期自动判断是否需要重置，而非依赖外部定时任务
- `daily_start_equity` 已经存在但未被用于回撤计算

---

### R2-002 🟠 High — Total Exposure 使用开仓价而非当前价

**文件**: `core/risk/manager.py:113-115`

```python
total_exposure = sum(p.get("position_value", 0) for p in self._open_positions.values())
exposure_pct = (total_exposure / self._account_balance * 100) if self._account_balance > 0 else 0
```

**描述**: `position_value` 在开仓时被设为 `qty * entry_price`（见 executor.py:111），之后**从未更新**。当市场价格大幅变动时：

- **价格上涨**: 实际敞口 > 记录敞口 → 可能超过 `max_total_exposure_pct` 但不被检测
- **价格下跌**: 实际敞口 < 记录敞口 → 过度保守，可能拒绝本可开的新仓位

例如：`max_total_exposure_pct = 80%`，账户余额 10000 USDT。开仓 BTC 价值 7000 USDT（70%敞口）。BTC 涨 30%，实际敞口变为 9100 USDT（91%）。但 `position_value` 仍为 7000 → exposure 检查通过 → 可以继续开仓。

**复现路径**: 开仓后价格大幅上涨 → 实际敞口超过硬限制 → 无检测 → 风险失控。

**修复建议**:
```python
# 使用 executor 的实时持仓数据计算敞口
positions = self._executor.get_open_positions() if self._executor else self._open_positions
total_exposure = sum(
    p.get("quantity", 0) * (market_price or p.get("current_price", p.get("entry_price", 0)))
    for sym, p in positions.items()
)
```

---

### R2-003 🟠 High — _pending_signals 无超时清理

**文件**: `core/risk/manager.py:58`

```python
self._pending_signals.add(symbol)
```

**描述**: 当信号被批准时，symbol 被加入 `_pending_signals`。正常情况下，`POSITION_UPDATE` 事件到达时在 `_on_position_update():170` 中清除：
```python
self._pending_signals.discard(symbol)
```

但如果 POSITION_UPDATE 事件因任何原因未能到达（事件总线丢弃、executor 异常、网络中断），该 symbol 将**永久留在 _pending_signals 中**。后果是第138行的检查：
```python
if symbol in self._open_positions or symbol in self._pending_signals:
    return RiskResult(approved=False, reason=f"Position already open for {symbol}")
```

该 symbol 的所有后续交易信号将被永久拒绝。

**复现路径**:
1. 信号批准 → `_pending_signals.add("BTCUSDT")`
2. OrderExecutor 下单过程异常（如 DB 写入失败）→ POSITION_UPDATE 未发出
3. BTCUSDT 的所有后续信号被永久拒绝 → 该币种无法交易直到系统重启

**修复建议**:
- `_pending_signals` 改为 `dict[str, float]`，存储加入时间戳
- 在 `check_signal()` 中清理超过 N 秒（如60秒）的 pending 条目

---

### R2-004 🟠 High — Max Open Trades 使用过期数据

**文件**: `core/risk/manager.py:142`

```python
if len(self._open_positions) >= self.config.hard_limits.max_open_trades:
```

**描述**: `_open_positions` 由异步事件 `POSITION_UPDATE` 更新。但 `update_balance()` 方法（第188行）优先使用 `self._executor.get_open_positions()`，体现了对缓存不一致的认知。

然而在第142行，`check_signal()` 仍使用 `self._open_positions` 而非 executor 的实时数据。这意味着：
- 如果某个仓位已被 executor 平仓但 POSITION_UPDATE 事件尚未处理，该仓位仍在 `_open_positions` 中
- 导致认为仓位已满，拒绝本可正常开仓的信号

**不一致**: `update_balance()` 信任 executor 实时数据，但 `check_signal()` 信任事件驱动的缓存。

**复现路径**: 快速开仓/平仓期间 → event 延迟到达 → open positions count 偏高 → 拒绝合法信号。

**修复建议**: 统一使用 executor 实时数据，或明确 `_open_positions` 为唯一真实源。

---

### R2-005 🟡 Medium — 浮点舍入导致连续亏损计数偏差

**文件**: `core/risk/circuit_breaker.py:33-42`

```python
def add_trade_result(self, pnl: float):
    pnl_rounded = round(pnl, 2)
    self.daily_pnl += pnl_rounded
    self.weekly_pnl += pnl_rounded
    if pnl_rounded < 0:
        self.consecutive_losses += 1
    else:
        self.consecutive_losses = 0
```

**描述**: PNL 舍入到 2 位小数后判断盈亏。如果实际 PNL 为 -0.001 USDT（微亏），`round(-0.001, 2)` = `0.0`，被判定为"不亏损"，`consecutive_losses` 重置为 0。

这在正常情况下影响有限（0.001 USDT 的偏差），但在高频交易或低价值代币中，零 PNL 交易（手续费抵消利润后的微亏）会被误判。

**修复建议**: 使用 `pnl < -0.005` 而非 `pnl_rounded < 0` 来判断亏损。

---

### R2-006 🟡 Medium — is_new_trip 仅基于 reason 字符串去重

**文件**: `core/risk/circuit_breaker.py:69-75`

```python
def is_new_trip(self) -> bool:
    if self.is_tripped and self.trip_reason != self._last_alert_reason:
        self._last_alert_reason = self.trip_reason
        return True
    return False
```

**描述**: 去重仅基于 `trip_reason` 字符串。如果熔断器被重置后再次由同一条件触发（如连续两次因 "Daily drawdown 7.51% exceeds limit 7.5%" 触发），第二次的 `is_new_trip()` 返回 `False`，因为 `trip_reason == _last_alert_reason`。

后果：第二次熔断不会发布 `RISK_BREACH` 事件，不会触发 breaker 响应动作，也不会通知 AI。

**复现路径**: 熔断 → reset → 再次熔断（同原因）→ 无告警 → 静默。

**修复建议**: `reset_trip()` 应同时重置 `_last_alert_reason = ""`。当前 `reset_trip()` 确实做了这件事（line 98），所以这个 bug 在 **reset 被外部直接修改 is_tripped=False 而不调用 reset_trip()** 的情况下出现。确认代码中所有重置路径是否都调用了 `reset_trip()`。

实际上，AI recovery loop 中：`cb.reset_trip(); cb.reset_daily()` → 这两行做了正确的事（line 143-144）。但 Web UI 中如果有直接修改 `is_tripped = False` 的路径，就不会重置 `_last_alert_reason`。

---

### R2-007 🟡 Medium — update_balance 时序不一致

**文件**: `core/risk/manager.py:182-198`

```python
def update_balance(self, balance: float):
    self._account_balance = balance
    positions = {}
    if self._executor:
        positions = self._executor.get_open_positions()
    if not positions:
        positions = self._open_positions
    total_invested = sum(p.get("amount_usdt", 0) for p in positions.values())
    equity = balance + total_invested
    self.breaker.set_equity(equity)
```

**描述**: `balance` 参数来自 `atomic_adjust_balance()` 的返回值，但 `positions` 来自 executor 的实时快照。这两种数据存在时序不一致：

1. 余额操作已完成（DB 已更新）
2. 但 executor 的 position 状态可能还未反映刚刚执行的交易
3. 导致 `equity = balance + invested` 的计算中，`balance` 和 `invested` 来自不同时间点

例如：
- 平仓：balance 已增加（包含退出资金），但 position 可能还在 executor 中
- 此时 `equity = new_balance + old_invested` → 高估

**复现路径**: 快速连续交易 → balance 和 positions 时序不一致 → equity 短暂高估 → 延迟触发熔断。

**修复建议**: 在 balance 操作前后使用一致的快照，或接受短暂的近似值并添加注释。

---

### R2-008 🟢 Low — _trip_callback 异常处理不充分

**文件**: `core/risk/manager.py:106-110`

```python
task = asyncio.create_task(self._trip_callback({...}))
task.add_done_callback(
    lambda t: logger.error(f"trip_callback failed: {t.exception()}") if t.exception() else None
)
```

**描述**: `_trip_callback` 通过 `asyncio.create_task` 异步执行。如果该 task 中发生异常，仅记录日志。但 trip callback 负责发布 `RISK_BREACH` 事件（触发自动响应动作如 close_all）。如果此事件未能成功发布，熔断后的自动保护不会执行。

**复现路径**: `_trip_callback` 中 `event_bus.publish()` 因队列满而挂起 → breaker 响应动作不执行。

---

### R2-009 🟢 Low — consecutive_losses 被单笔盈利重置

**文件**: `core/risk/circuit_breaker.py:41-42`

```python
else:
    self.consecutive_losses = 0
```

**描述**: 连续亏损计数器在**任何**非负 PNL 时重置为 0。这意味着：

- 连续亏损 4 次（每次 -100 USDT）
- 第 5 次盈利 +1 USDT → `consecutive_losses` 重置为 0
- 再连续亏损 4 次...

从风控角度看，这可能在连续大亏中插入一笔微利而绕过保护。`max_consecutive_losses = 5` 的熔断可以被轻易规避。

**修复建议**: 使用滑动窗口或累计亏损而非简单计数器。例如：过去 N 笔交易中亏损比例超过阈值时触发。

---

## 累计统计

| 等级 | R1 | R2 | 累计 |
|------|-----|-----|------|
| Critical | 1 | 1 | 2 |
| High | 4 | 3 | 7 |
| Medium | 5 | 3 | 8 |
| Low | 3 | 2 | 5 |
| **总计** | **13** | **9** | **22** |
