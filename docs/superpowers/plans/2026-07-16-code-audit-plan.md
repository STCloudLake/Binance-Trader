# Binance Trader 多轮次代码审计 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 Binance Trader 自动化交易平台执行 10 轮模块化代码审计 + 1 轮深度收尾审计，每轮产出独立漏洞报告，最后汇总并提供开发方向建议与核心算法详解文档。

**Architecture:** 审计以模块为单位，每轮聚焦1-2个模块，逐文件逐行审查。采用静态代码审查方法，结合数据流追踪和逻辑验证。每轮产出独立 Markdown 报告存入 `docs/audit/`。收尾阶段产出算法文档和开发路线图。

**Tech Stack:** Python 3.11+, SQLite, asyncio, FastAPI, TA-Lib, pandas, numpy

## Global Constraints

- 金融交易逻辑优先：策略引擎→风控→执行器→回测→基础设施
- 每轮产出独立 Markdown 报告，含漏洞清单 + 严重等级 + 复现路径 + 修复建议
- 严重等级：Critical（资金损失）、High（潜在资金损失）、Medium（稳定性）、Low（代码质量）
- 所有报告存入 `binance_trader/docs/audit/`
- 仅列出漏洞，不做修改

---

### Task 1: 审计环境准备与基线建立

**Files:**
- Create: `binance_trader/docs/audit/README.md`
- Create: `binance_trader/docs/audit/cumulative-findings.csv`
- Create: `binance_trader/docs/audit/severity-guidelines.md`

**Interfaces:**
- Produces: 审计目录结构、漏洞追踪 CSV schema、严重等级判定指南

- [ ] **Step 1: 创建审计目录 README 和基础结构**

```bash
mkdir -p binance_trader/docs/audit
```

创建 `binance_trader/docs/audit/README.md`:

```markdown
# Binance Trader 代码审计

**开始日期**: 2026-07-16
**审计范围**: 全部核心模块，共 10+1 轮
**审计方法**: 静态代码审查 + 数据流追踪 + 逻辑验证

## 目录结构
- R1-strategy-engine.md ~ R10-ml-news-integration.md: 各轮审计报告
- R11-deep-audit-summary.md: 深度收尾审计
- cumulative-findings.csv: 机器可读漏洞清单（汇总）
- fix-tracker.md: 修复进度跟踪

## 严重等级定义
| 等级 | 定义 |
|------|------|
| Critical | 直接导致资金损失、API密钥泄露、余额计算错误 |
| High | 可能导致资金损失或系统不可用，需要特定条件 |
| Medium | 影响系统稳定性、数据完整性，不直接导致资金损失 |
| Low | 代码质量问题、潜在风险、最佳实践偏离 |
```

- [ ] **Step 2: 创建漏洞追踪 CSV**

创建 `binance_trader/docs/audit/cumulative-findings.csv`:

```csv
id,round,severity,module,file,line_range,title,description,reproduction,fix_suggestion,status
```

- [ ] **Step 3: 创建严重等级判定指南**

创建 `binance_trader/docs/audit/severity-guidelines.md`:

```markdown
# 严重等级判定指南

## Critical
- PNL 计算公式错误（符号、乘除错误）
- 余额操作非原子性导致丢失或重复记账
- API Key/Secret 明文暴露（日志、响应、文件权限）
- 未授权交易执行（权限绕过直接下单）
- 熔断器逻辑错误导致无限亏损

## High
- 竞态条件导致重复下单或重复扣款
- 止损/止盈计算错误导致错误价格平仓
- 模拟/实盘代码路径混淆
- 认证绕过或权限提升
- WebSocket 数据丢失无补偿机制
- AI 全自动模式无安全边界

## Medium
- 事件队列溢出丢失信号
- 数据库连接泄漏
- 缓存不一致导致错误信号
- 异常吞没（bare except）
- 配置验证不完整

## Low
- 日志泄露敏感信息
- 缺少输入验证
- 硬编码魔法数字
- 废弃代码未清理
- 文档与实际行为不一致
```

- [ ] **Step 4: 验证目录结构**

```bash
ls -la binance_trader/docs/audit/
```

Expected: `README.md`, `cumulative-findings.csv`, `severity-guidelines.md` 均存在。

---

### Task 2: R1 — Strategy Engine + Indicators 审计

**Files 审计范围:**
- `binance_trader/core/strategy/engine.py` (全部 489 行)
- `binance_trader/core/strategy/indicators.py` (全部)
- `binance_trader/core/strategy/loader.py` (全部)
- `binance_trader/app/event_bus.py` (全部 84 行，事件流正确性)

**Interfaces:**
- Consumes: 审计目录结构（Task 1）
- Produces: `binance_trader/docs/audit/R1-strategy-engine.md`

- [ ] **Step 1: 审查 `strategy/indicators.py` — `compute_all()` 和 `evaluate_condition()` 函数**

审查要点：
- TA-Lib 指标计算参数是否正确传递
- `evaluate_condition()` 的 `eval()` 安全性 → 是否可能注入恶意表达式
- 指标计算完成后，DataFrame 列名是否与条件表达式中引用的一致
- 边界条件：空 DataFrame、NaN 值、仅1行数据

关键代码路径：
```python
# indicators.py 中的 evaluate_condition 实现
# 检查是否使用了 eval/exec 等危险函数
# 检查公式解析是否正确处理了所有 TA-Lib 指标名
```

- [ ] **Step 2: 审查 `strategy/engine.py` — 信号融合公式**

审查要点：
- `final_score` 计算公式的数值稳定性（除零保护）
- `ml_directional` 变换 `(ml_conf - 0.5) * 2` 是否正确映射 [0,1]→[-1,1]
- `total_weight == 0` 时的回退逻辑
- ML weight 取值逻辑：`strategy.ml_config.weight if enabled else w.ml` 是否正确
- 信号缓存 `_signal_cache` 的 key 冲突可能性

关键代码（engine.py:150-160）:
```python
ml_weight = strategy.ml_config.weight if (strategy.ml_config and strategy.ml_config.enabled) else w.ml
total_weight = w.indicator + ml_weight + w.news
if total_weight > 0:
    final_score = (indicator_signal * w.indicator + ml_directional * ml_weight + news_sent * w.news) / total_weight
else:
    final_score = float(indicator_signal)
```

- [ ] **Step 3: 审查多时间框架趋势对齐逻辑**

审查要点（engine.py:166-198）：
- `_TF_MIN` 映射是否覆盖所有可能的 interval 值
- `tf_multiplier` 的累积逻辑是否正确（`min(tf_multiplier, mult)` 取最严格）
- 如果所有 higher TF 数据获取失败，默认 multiplier=1.0 是否安全
- EMA(50) 计算在数据不足50行时的回退

- [ ] **Step 4: 审查入场/出场信号冲突处理**

审查要点（engine.py:200-214）：
- `exit_blocks_entry` 的判断逻辑是否完整
- 多空同时活跃时的 `indicator_signal = 0.0` 是否正确（静默丢弃）
- strategy→position 归属检查：位置是否被正确策略管理

- [ ] **Step 5: 审查减仓（reduce）逻辑**

审查要点（engine.py:276-313）：
- `reduce_count` 的 key 格式 `reduce_count_{symbol}_{side}` 防止跨策略干扰
- reduce_pct 的最小/最大值保护
- 4次减仓上限后是否还有路径触发完全退出

- [ ] **Step 6: 审查 `strategy/loader.py` — YAML 解析安全性**

审查要点：
- `yaml.safe_load()` 是否已被使用（非 `yaml.load()`）
- Pydantic 模型验证是否完整（字段类型、范围约束）
- 策略文件路径遍历风险（`name` 参数是否可导致读取任意文件）

- [ ] **Step 7: 审查 `event_bus.py` — 事件系统可靠性**

审查要点：
- `_queue` maxsize=10000，队列满时的行为（阻塞 vs 丢弃）
- `_process()` 中 `asyncio.gather(*tasks, return_exceptions=True)` 是否正确处理异常
- `subscribe_all()` 的调用者（AlertManager）是否会因一个慢回调阻塞其他订阅者
- EventBus 单实例 vs 多实例的线程安全

- [ ] **Step 8: 撰写并保存 R1 审计报告**

创建 `binance_trader/docs/audit/R1-strategy-engine.md`，按格式撰写：

```markdown
# 审计报告 R1 — Strategy Engine + Indicators + Event Bus

**日期**: 2026-07-16
**审查文件**: engine.py, indicators.py, loader.py, event_bus.py
**审查代码行数**: ~1200 行

## 摘要
| 等级 | 数量 |
|------|------|
| Critical | N |
| High | N |
| Medium | N |
| Low | N |

## 漏洞清单
| ID | 等级 | 文件:行 | 描述 | 复现路径 | 修复建议 |
|----|------|---------|------|----------|----------|
| R1-001 | ... | ... | ... | ... | ... |

## 详细分析
### R1-001: [标题]
**文件**: `core/strategy/engine.py:xxx`
**描述**: ...
**复现路径**: ...
**修复建议**: ...

## 累计统计
(所有已完成的轮次汇总)
```

- [ ] **Step 9: 更新 cumulative-findings.csv**

将 R1 发现的所有问题追加到 CSV 文件中。

---

### Task 3: R2 — Risk Management Ⅰ (Circuit Breaker + 7-Step Pipeline) 审计

**Files 审计范围:**
- `binance_trader/core/risk/manager.py` (全部 217 行)
- `binance_trader/core/risk/circuit_breaker.py` (全部 99 行)

**Interfaces:**
- Consumes: R1 报告（Task 2）
- Produces: `binance_trader/docs/audit/R2-risk-management-1.md`

- [ ] **Step 1: 审查 `circuit_breaker.py` — 状态机正确性**

审查要点：
- `check()` 中 `daily_drawdown` 计算公式：`(peak_equity - current_equity) / peak_equity * 100`
  - `peak_equity` 为 0 时的除零保护 ✓（已有 `if self.peak_equity > 0`）
  - 回撤是从 peak 计算，非当日开盘 → 跨日未重置 peak 可能导致误触发
- `add_trade_result()` 中 `pnl_rounded = round(pnl, 2)` — 浮点舍入精度
- `is_new_trip()` 去重仅基于 `trip_reason`，如果同一原因在不同时间触发两次，第二次被去重
- `reset_daily()` 重置 `peak_equity = current_equity` — 但不清除 `is_tripped`
- `reset_weekly()` 仅重置 `weekly_pnl` + `week_start_equity`，但不检查 `is_tripped`
- 日/周切换的时间判断在 `main.py` 的 `_circuit_breaker_reset_loop()` 中，若该 task 崩溃则永远不会重置

- [ ] **Step 2: 审查 `risk/manager.py` — 7步检查管线**

审查要点（逐步骤）：

**Step 1 — Circuit Breaker Check:**
- `breaker.check()` 返回 `(tripped, reason)`，但 `_trip_callback` 是异步创建的 task
- 如果 trip callback 抛出异常，`task.add_done_callback` 仅记录日志，不影响后续逻辑

**Step 2 — Total Exposure Check:**
```python
total_exposure = sum(p.get("position_value", 0) for p in self._open_positions.values())
exposure_pct = (total_exposure / self._account_balance * 100) if self._account_balance > 0 else 0
```
- `position_value` 使用的是 `entry_price * quantity`（开仓价值），不是 `current_price * quantity`（当前市值）
- 这导致 exposure 计算在价格大幅变动时不准确

**Step 3 — Position Size Check:**
- `calculate_position_size()` 返回 `(qty, risk_amount)`，其中 qty 可能为浮点微小值
- 无最小 notional value 检查（Binance 有 min notional 限制）

**Step 4 — Leverage Check:**
- 仅做上限截断，无下限检查
- `signal` 中的 leverage 来源未被验证

**Step 5 — Stop Loss Check:**
- `calculate_stop_loss()` 逻辑在 `position_sizer.py` 中

**Step 6 — Same Symbol Check:**
```python
if symbol in self._open_positions or symbol in self._pending_signals:
```
- `_pending_signals` 在什么情况下可能永久残留？（POSITION_UPDATE 丢失时）
- 无超时清理机制

**Step 7 — Max Open Trades Check:**
- `len(self._open_positions)` 使用本地缓存，非 executor 实时数据
- 与 `update_balance()` 中优先使用 `_executor.get_open_positions()` 不一致

- [ ] **Step 3: 审查 `update_balance()` 方法**

审查要点（manager.py:182-198）：
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
    if total_invested == 0 and self.breaker.peak_equity > equity * 1.05:
        self.breaker.clamp_peak_to_current()
```
- `amount_usdt` 可能在某些位置记录中缺失（如在 executor 中 position 的 amount_usdt 来自 `qty * entry_price` 或 `data.get("amount_usdt", qty * price)`）
- `invested_returned` 在 executor.close_position() 中返回，但在 bal 更新时 position 已被删除 → 时序问题

- [ ] **Step 4: 审查 `_trip_callback` 和 RISK_BREACH 事件流**

审查要点：
- 事件数据中 `daily_drawdown_pct` 的计算在 `check_signal()` 中即时计算，非 breaker 内部值
- `open_positions` 以 `.copy()` 传递，但 dict 内的子 dict 仍是引用（浅拷贝）

- [ ] **Step 5: 撰写并保存 R2 审计报告**

创建 `binance_trader/docs/audit/R2-risk-management-1.md`，格式同 R1。

- [ ] **Step 6: 更新 cumulative-findings.csv**

---

### Task 4: R3 — Risk Management Ⅱ (Position Sizer + Position Guard) 审计

**Files 审计范围:**
- `binance_trader/core/risk/position_sizer.py` (全部 49 行)
- `binance_trader/core/risk/position_guard.py` (全部 157 行)

**Interfaces:**
- Consumes: R1, R2 报告
- Produces: `binance_trader/docs/audit/R3-risk-management-2.md`

- [ ] **Step 1: 审查 `position_sizer.py` — 仓位计算**

审查要点：
- `capital_pool` 分段：core = balance * 0.7, satellite = balance * 0.3
  - 但 total capital_pool = balance，未考虑已有持仓占用的资金
- `effective_pct = max(self.soft.position_size_pct, 1.0)` — floor 1%
  - 当 AI 在 full_auto 模式下将 position_size_pct 降为0时被覆盖到1%
  - 但当用户手动设为极低值（如0.1%）时也被覆盖，可能不符合用户意图
- `max_risk = account_balance * (self.hard.max_position_size_pct / 100)` — 硬上限
  - 这个硬上限在 core 仓位中也适用，可能导致 core+c satellite 总仓位受限于单个 cap
- `quantity = risk_per_trade / current_price` — 无取整，Binance 有 step size 限制

- [ ] **Step 2: 审查 `position_sizer.py` — 止损止盈计算**

```python
def calculate_stop_loss(self, entry_price: float, side: str) -> float:
    sl_pct = max(self.soft.stop_loss_pct / 100, self.hard.min_stop_loss_distance_pct / 100)
```
- `soft.stop_loss_pct` 是百分比（如2.0表示2%），除以100转为小数 — 正确
- `hard.min_stop_loss_distance_pct` 也是百分比 — 正确
- 但 `max()` 取的是两者中较大的 → 确保止损距离不低于硬限制 — 逻辑正确

- [ ] **Step 3: 审查 `position_guard.py` — 紧急止损**

审查要点：
```python
if getattr(limits, "emergency_stop_enabled", False):
    threshold = getattr(limits, "emergency_stop_threshold_pct", -5.0)
    if pnl_pct <= threshold:
```
- `pnl_pct` 是从 `(price - entry) / entry * 100` 计算的未实现盈亏
- `threshold` 默认为 -5.0（即亏损超过5%触发）
- 此值是否可由用户在 Web UI 中修改？修改 range 是否受限？
- 如果用户错误地将 threshold 设置为正值或0，行为异常

- [ ] **Step 4: 审查 `position_guard.py` — 移动止损**

审查要点：
```python
# Long position
new_sl = price * (1 - distance_pct / 100)
entry_sl = entry * (1 - distance_pct / 100)
floor_sl = max(entry_sl, entry * 0.99)  # at worst 1% below entry
if current_sl:
    new_sl = max(new_sl, current_sl, floor_sl)  # only move up
```
- `floor_sl` 保护逻辑：对于 long，最差情况止损设在 entry * 0.99
- 但对于 short：`ceiling_sl = min(entry_sl, entry * 1.01)` — 最差止损设在 entry * 1.01
- 问题：当价格快速下跌（long时），`new_sl` 可能比 `current_sl` 还低 → `max()` 保持不变 → 止损不更新，但价格可能已跌破止损 → 依赖下一个15s检查周期
- 移动止损 **仅调整 stop_loss 值**，并不检查当前价格是否已触发止损 → 触发逻辑在哪？
  - 在 executor._execute_sim 中也没有基于 stop_loss 的自动平仓逻辑
  - 止损依赖策略引擎的 exit conditions → 但 condition 中可能没有引用 stop_loss 变量
  - **这是重大遗漏：止损仅在 memory 中更新，从未被 enforce**

- [ ] **Step 5: 撰写并保存 R3 审计报告**

创建 `binance_trader/docs/audit/R3-risk-management-2.md`。

- [ ] **Step 6: 更新 cumulative-findings.csv**

---

### Task 5: R4 — Order Executor + Balance Management 审计

**Files 审计范围:**
- `binance_trader/core/executor/executor.py` (全部 310 行)
- `binance_trader/db/database.py` (余额相关函数)
- `binance_trader/app/main.py` (POSITION_EXIT/POSITION_REDUCE 处理)

**Interfaces:**
- Consumes: R1-R3 报告
- Produces: `binance_trader/docs/audit/R4-order-executor.md`

- [ ] **Step 1: 审查 `_execute_sim()` — 模拟下单**

审查要点：
```python
order_id = f"sim_{int(time.time() * 1000)}"
```
- ID 生成：`time.time() * 1000` 毫秒精度 — 同一毫秒内多次调用会碰撞
- `amount_usdt = data.get("amount_usdt", qty * price)` — 默认值计算使用未验证的 qty 和 price
- `self._positions[symbol] = {...}` 直接覆盖 — 如果已有同 symbol 持仓，静默覆盖不报警

- [ ] **Step 2: 审查 `close_position()` — PNL 计算与余额恢复**

审查要点：
```python
pnl = (current_price - entry) * close_qty if side == "long" else (entry - current_price) * close_qty
invested_close = original_amount * reduce_pct / 100
```
- PNL 公式：`(exit - entry) * qty` — 标准计算，但未扣除手续费
- `invested_returned` 使用 `original_amount * reduce_pct / 100` — 使用原始投入金额而非当前市值
  - 意味着返还的 USDT 是 `投入本金 + PNL`，而非 `当前市值`
  - 这对 sim 模式是合理的（简化处理），但对实盘需验证

- [ ] **Step 3: 审查 `atomic_adjust_balance()` — 余额原子性**

审查要点：
```python
async def atomic_adjust_balance(delta: float, db_path: str = None) -> float:
    async with _balance_lock:
        current = await load_sim_balance(db_path)
        new_balance = current + delta
        await save_sim_balance(new_balance, db_path)
        return new_balance
```
- `_balance_lock` 是模块级全局锁 — 进程内安全，但多进程不安全
- `load_sim_balance` 每次都打开新连接 → 非事务性
- `save_sim_balance` 使用 `INSERT OR REPLACE` → 如果 key 被并发删除，会创建新行
- 中途异常（如 save 失败）→ balance 已内存修改但未持久化 → **不一致**

- [ ] **Step 4: 审查余额操作的调用链**

审查 `main.py` 中的余额调整路径：
1. POSITION_EXIT handler (`_on_position_exit`): `atomic_adjust_balance(invested_returned + trade_pnl)`
2. POSITION_REDUCE handler (`_on_position_reduce`): `atomic_adjust_balance(invested_returned + trade_pnl)`
3. Circuit breaker close_all/close_worst: 直接调用 `atomic_adjust_balance`
4. Position guard emergency_close: 直接调用 `atomic_adjust_balance`
5. Executor._execute_sim: 开仓时 `atomic_adjust_balance(-amount_usdt)`

每种路径的 delta 计算是否正确？是否所有路径都正确更新了 `risk_manager.update_balance()`？

- [ ] **Step 5: 审查 `restore_positions()` 逻辑**

审查要点：
- 从 DB 恢复持仓时，`amount_usdt` 使用 `qty * entry` — 固定值
- 重启后 `current_price` 设为 `entry_price` — unrealized_pnl 为 0，但实际行情可能已大幅变动
- `amount_usdt` 在 DB 中不存储 → 恢复时重新计算，可能与实际不一致
- 恢复的 position 没有 `stop_loss`, `take_profits` — 风控缺失

- [ ] **Step 6: 审查 `_execute_live()` 实盘路径**

审查要点：
- 重试逻辑（3次指数退避）：2s, 4s — 合理
- 订单失败后无余额回滚（实盘中交易所不扣余额，但代码无验证）
- `order["price"]` 在实盘中可能为 0 → fallback 到 `data["price"]` → 可能不准确
- 无 order status 同步：下单后不轮询订单状态确认成交

- [ ] **Step 7: 撰写并保存 R4 审计报告**

创建 `binance_trader/docs/audit/R4-order-executor.md`。

- [ ] **Step 8: 更新 cumulative-findings.csv**

---

### Task 6: R5 — Market Data + Event Bus 审计

**Files 审计范围:**
- `binance_trader/core/market_data/provider.py` (全部 272 行)
- `binance_trader/core/market_data/ohlcv_cache.py` (全部)
- `binance_trader/app/event_bus.py` (深度审查)

**Interfaces:**
- Consumes: R1-R4 报告
- Produces: `binance_trader/docs/audit/R5-market-data.md`

- [ ] **Step 1: 审查 WebSocket 连接管理**

审查要点：
- 重连逻辑：`await asyncio.sleep(5)` 延迟固定
- 无指数退避 — 可能在网络不稳定时频繁重连
- `_handle_ws_message` 中 `kline["x"]` (isFinal) 检查 → 仅处理已关闭的K线，正确
- `_price_cache` 更新在 kline 未关闭时也会执行 → partial update 被缓存
  - `self._price_cache[symbol] = float(kline["c"])` 在 isFinal 检查之前执行
  - 意味着价格缓存可能包含未关闭K线的价格

- [ ] **Step 2: 审查 REST 降级轮询逻辑**

审查要点（main.py:262-296）：
```python
interval_secs = {"1m": 120, "5m": 300, "15m": 900, "1h": 3600, "4h": 7200}
```
- 检查间隔 = K线周期的 2 倍 — 合理，避免 WS 短暂中断时立即 REST
- 使用倒数第二根 K线（`df.index[-2]`）避免 incomplete candle — 正确
- 但如果 REST 调用本身返回的是正在形成的 K线，`index[-2]` 可能是旧数据 → 重复发布

- [ ] **Step 3: 审查 OHLCV Cache 一致性**

审查要点：
- `cache.append_candle()` 追加单根 K线
- `cache.update()` 批量合并（使用 `combine_first` 还是直接覆盖？）
- 磁盘 flush 周期 5min → 崩溃最多丢失 5min WS 数据
- Cache 为 HashMap `{symbol_interval: DataFrame}` — 无 LRU 淘汰 → 长时间运行内存增长
- `_prefetch_history()` 的 MIN_CANDLES 检查 → 跳过已足够数据的 interval — 但若用户手动修改了 parquet 文件，缓存不一致

- [ ] **Step 4: 审查 EventBus 深度问题**

- `_queue.put()` 在队列满时（maxsize=10000）→ `put()` 是 async 的，会阻塞生产者
- 如果某个 subscriber 回调耗时过长，其他事件的处理被延迟
- `subscribe_all()` 注册的回调（AlertManager）在每次事件时都被调用 → 高频 MARKET_KLINE 事件触发告警规则评估

- [ ] **Step 5: 撰写并保存 R5 审计报告**

创建 `binance_trader/docs/audit/R5-market-data.md`。

- [ ] **Step 6: 更新 cumulative-findings.csv**

---

### Task 7: R6 — Backtest Engine 审计

**Files 审计范围:**
- `binance_trader/core/backtest/engine.py` (全部)
- `binance_trader/core/backtest/engine_hybrid.py` (全部)
- `binance_trader/core/backtest/event_executor.py` (全部)
- `binance_trader/core/backtest/signal_matrix.py` (全部)
- `binance_trader/core/backtest/cost_model.py` (全部)
- `binance_trader/core/backtest/metrics.py` (全部)

**Interfaces:**
- Consumes: R1-R5 报告
- Produces: `binance_trader/docs/audit/R6-backtest-engine.md`

- [ ] **Step 1: 审查 SignalMatrix 构建正确性**

审查要点：
- `IndicatorGrouper._config_hash()` 使用 JSON 序列化 + SHA256 → 确保相同配置的策略共享计算结果
  - 但 indicator dict 中如果有 float 值，JSON 序列化精度问题可能导致 hash 不一致
- `SignalMatrixBuilder.build()` 中 `finest_tf` 的选择逻辑
- 信号矩阵的 MultiIndex 构建 → 确保 `(strategy, symbol, tf)` 组合唯一

- [ ] **Step 2: 审查 EventDrivenExecutor 交易模拟**

审查要点：
- Entry 信号处理：`matrix.get_entry()` → 查找信号
- Exit 信号处理：`matrix.get_exit()` → 查找退出信号
- Stop Loss / Take Profit 触发：
  - 止损逻辑在每 tick 检查 `pos["stop_loss"]` vs `current_price`
  - 但止损只在 entry 时设定一次，无移动止损
  - 与策略引擎的 exit conditions 不同 → 回测与实盘行为不一致
- Position 管理：`per_strategy_isolation` 模式下的仓位隔离

- [ ] **Step 3: 审查回测指标计算**

审查要点（metrics.py）:
```python
sharpe_ratio = mean(daily_returns) / std(daily_returns) × sqrt(365)
```
- 使用365天年化 → 加密货币市场365天交易，合理
- 但 `daily_returns` 的计算方式（简单差值 vs 对数收益率）
- `max_drawdown` 的计算：是否使用滚动 peak

- [ ] **Step 4: 审查 Legacy vs Hybrid 引擎等价性**

审查要点：
- 两种引擎在相同输入下是否产生相同结果
- 测试文件 `test_hybrid_equivalence.py` 的覆盖范围
- Engine 选择逻辑（engine.py:_select_engine）的边界条件：
  - ML enabled → 强制 legacy
  - < 3 strategies → 强制 legacy
  - 子进程 worker → 强制 legacy

- [ ] **Step 5: 审查成本模型**

审查要点（cost_model.py, engine_hybrid.py:60-71）：
```python
costs = (entry_notional + exit_notional) * fee + (entry_notional + exit_notional) * (spread_pct / 2.0)
```
- Fee 应在 entry 时收一次 taker fee，exit 时再收一次 — 当前公式是 `(entry + exit) * fee` → 正确
- Spread 使用 `spread_pct / 2.0` → 假设双向 spread 各承担一半 → 合理

- [ ] **Step 6: 撰写并保存 R6 审计报告**

创建 `binance_trader/docs/audit/R6-backtest-engine.md`。

- [ ] **Step 7: 更新 cumulative-findings.csv**

---

### Task 8: R7 — AI Controller + Strategy Lifecycle 审计

**Files 审计范围:**
- `binance_trader/core/ai/deepseek_ctl.py` (全部 411 行)
- `binance_trader/core/ai/prompts.py` (全部)
- `binance_trader/core/ai/vibe_connector.py` (全部)
- `binance_trader/core/ai/strategy_lifecycle.py` (全部)

**Interfaces:**
- Consumes: R1-R6 报告
- Produces: `binance_trader/docs/audit/R7-ai-controller.md`

- [ ] **Step 1: 审查 `_call_deepseek()` — API 调用安全性**

审查要点：
```python
response = await self.client.chat.completions.create(
    model=self.config.ai_model,
    messages=[...],
    max_tokens=2000,
    temperature=0.3,
)
return response.choices[0].message.content
```
- 无 rate limit 处理 — API 限频时静默失败（bare `except Exception` 返回 None）
- 无 response validation — `choices` 可能为空
- `temperature=0.3` — 较低随机性，适合决策任务，合理

- [ ] **Step 2: 审查 JSON 解析安全性**

审查要点（多处）：
```python
data = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
```
- 如果 AI 返回的 JSON 包含额外字段，`json.loads()` 不会拒绝（宽松解析）
- 如果 AI 返回非 JSON 或格式错误 → `json.JSONDecodeError` 被捕获
- 但如果 AI 返回了合法的但内容错误的 JSON（如负数 position_size），后续代码是否验证？
  - `adjust_risk()` 中有 `max(result.get("position_size_pct", 5.0), 1.0)` — floor 保护 ✓
  - 但其他方法（assess_market, select_coins）直接使用返回值无验证

- [ ] **Step 3: 审查全自动模式风险边界**

审查要点：
- `full_auto` 模式下 AI 可直接修改 `signal_weights`（market_assessment_loop:228）
- `full_auto` 模式下 AI 可直接修改 `soft_params`（risk_adjustment_loop:271-276）
- `full_auto` 模式下 breaker action 由 AI 决定（decide_breaker_action:94-119）
- 硬风控的 `HardRiskLimits` 不可被 AI 修改 → 确认代码中无修改路径
- AI 是否可能将 `position_size_pct` 设置为极大值？
  - `max(result.get("position_size_pct", 5.0), 1.0)` 只有 floor 没有 ceiling！

- [ ] **Step 4: 审查 Breaker 决策超时处理**

审查要点（deepseek_ctl.py:94-119）：
```python
result = await asyncio.wait_for(self._call_deepseek(...), timeout=15.0)
```
- 超时 fallback 到 `close_all` — 安全，但过于激进
- 如果 AI 返回了无效 action → fallback 到 `close_all`

- [ ] **Step 5: 审查 Prompts 注入风险**

审查要点（prompts.py）：
- 用户输入（策略名、参数值）是否被拼接到 prompt 中？
- `_build_market_context()` 从持仓数据构建字符串 → 如果持仓数据包含恶意内容
- `_build_breaker_context()` 同样拼接数据到 context

- [ ] **Step 6: 审查 Strategy Lifecycle Manager**

审查要点：
- `generate_strategy()` — AI 生成的策略是否经过安全验证？
- `check_and_retire()` — 停用策略的阈值是否合理？
- `analyze_and_optimize()` — 优化建议是否经过回测验证？

- [ ] **Step 7: 撰写并保存 R7 审计报告**

创建 `binance_trader/docs/audit/R7-ai-controller.md`。

- [ ] **Step 8: 更新 cumulative-findings.csv**

---

### Task 9: R8 — Web Server + Auth + API 审计

**Files 审计范围:**
- `binance_trader/web/server.py` (全部 ~60+ routes)
- `binance_trader/core/auth/auth.py` (全部 213 行)
- `binance_trader/web/templates/` (Jinja2 模板)

**Interfaces:**
- Consumes: R1-R7 报告
- Produces: `binance_trader/docs/audit/R8-web-auth-api.md`

- [ ] **Step 1: 审查认证中间件**

审查要点：
```python
if path in ("/login", "/api/auth/login", "/api/auth/logout") or \
   path.startswith("/static") or path.startswith("/ws/"):
    return await call_next(request)
```
- 白名单路径 — 确认无敏感端点被遗漏
- `/ws/` 路径不验证身份 → WebSocket 可被未认证用户连接
- Session token 验证：`auth.verify_session(session_token)` → 查找内存 dict → 重启后全部失效
- JWT fallback：`auth.verify_jwt(auth_header[7:])` → 验证 HS256 签名

- [ ] **Step 2: 审查 JWT 安全性**

审查要点（auth.py）：
- `jwt_secret` 在启动时随机生成 → 每次重启所有 token 失效
- 但 fingerprint 仅记录到日志 → 无法用于恢复
- `session_hours` 默认为 24 → JWT exp 设为 `time.time() + session_hours * 3600`
- 无 token revocation 机制

- [ ] **Step 3: 审查密码安全**

审查要点：
- bcrypt hash → 安全 ✓
- 初始 admin 密码：`secrets.token_urlsafe(12)` → ~72 bits entropy → 足够
- 但密码被打印到 stderr → 可能被日志系统捕获
- 无密码强度要求 → 用户可设置弱密码

- [ ] **Step 4: 审查 API 端点权限控制**

审查 `_require_trader()` 和 `_require_admin()` 的使用：
- 逐端点检查是否都正确使用了权限检查
- Trading 端点：`/api/trade`, `/api/trade/close/{symbol}` — trader+
- Settings 端点：`/api/settings/*` — trader+
- User management：`/api/users/*` — admin
- DB manager：`/api/db/*` — admin

- [ ] **Step 5: 审查输入验证**

审查要点：
- POST/PUT 请求体的验证（Pydantic models vs raw dict access）
- 路径参数的 SQL 注入防护（参数化查询 vs 字符串拼接）
- WebSocket 消息的验证

- [ ] **Step 6: 审查 Jinja2 模板注入风险**

审查要点：
- 所有 `{{ }}` 表达式是否可能被注入
- `|safe` 过滤器使用是否安全
- 用户输入（如策略名、交易对名）是否正确转义

- [ ] **Step 7: 撰写并保存 R8 审计报告**

创建 `binance_trader/docs/audit/R8-web-auth-api.md`。

- [ ] **Step 8: 更新 cumulative-findings.csv**

---

### Task 10: R9 — Database + Config + Alerts 审计

**Files 审计范围:**
- `binance_trader/db/database.py` (全部 271 行)
- `binance_trader/app/config.py` (全部 187 行)
- `binance_trader/alerts/manager.py` (全部 171 行)
- `binance_trader/alerts/rules.py` (全部)
- `binance_trader/config/secrets.yaml`

**Interfaces:**
- Consumes: R1-R8 报告
- Produces: `binance_trader/docs/audit/R9-database-config-alerts.md`

- [ ] **Step 1: 审查 SQL 注入风险**

审查要点 — 检查所有 SQL 查询：
- `database.py` 中所有 SQL 语句是否使用参数化查询
- `alert_manager.py:108-124` 的动态 SQL 构建：
  ```python
  query += " AND level = ?"
  ```
  - `level` 来自函数参数，外部可控 → 但使用了 `?` 占位符 ✓
  - LIKE 模式：`query += " AND (" + " OR ".join(clauses) + ")"` → 使用占位符 ✓
- `web/server.py` 中的查询是否安全

- [ ] **Step 2: 审查配置安全管理**

审查要点：
- `secrets.yaml` 是否在 `.gitignore` 中
- `config.py` 中 API key 的获取：
  ```python
  self.binance_api_key = os.getenv("BINANCE_API_KEY", self._get_nested("binance", "api_key") or "")
  ```
  - 环境变量优先于文件 → 好的实践 ✓
  - 但如果 secrets.yaml 被误提交，keys 泄露
- `config.py` 的 `Config._instance` 是进程级单例 → 多线程安全但无锁

- [ ] **Step 3: 审查告警风暴防护**

审查要点（alert_manager.py）：
- `_on_any_event()` 对每个事件评估所有规则 → 高频事件（MARKET_KLINE）触发大量评估
- 冷却机制：`AlertRule.evaluate()` 中的 `cooldown_seconds` 检查
  - 确认 cooldown 实现是否正确（使用 `time.time()` 比较）
- `_broadcast()` 中 `put_nowait()` 在队列满时丢弃 → 无错误反馈

- [ ] **Step 4: 审查数据库迁移逻辑**

审查要点（database.py:20-31）：
```python
for col, default in [("trader", "'manual'"), ...]:
    try:
        await db.execute(f"ALTER TABLE trades ADD COLUMN {col} TEXT DEFAULT {default}")
    except Exception:
        pass
```
- `bare except` 静默吞所有异常 → 如果列已存在则跳过，但也吞其他错误
- `{col}` 直接拼接到 SQL 中 → 列名硬编码，非用户输入 → 安全但脆弱
- 无版本化迁移 → 无法追踪数据库 schema 版本

- [ ] **Step 5: 撰写并保存 R9 审计报告**

创建 `binance_trader/docs/audit/R9-database-config-alerts.md`。

- [ ] **Step 6: 更新 cumulative-findings.csv**

---

### Task 11: R10 — ML + News + 跨模块集成 审计

**Files 审计范围:**
- `binance_trader/core/ml/predictor.py` (全部)
- `binance_trader/core/ml/trainer.py` (全部)
- `binance_trader/core/ml/features.py` (全部)
- `binance_trader/core/ml/tft_model.py` (全部)
- `binance_trader/core/ml/patchtst_model.py` (全部)
- `binance_trader/core/news/analyzer.py` (全部)
- `binance_trader/core/news/fetcher.py` (全部)
- `binance_trader/core/news/source_manager.py` (全部)
- 跨模块数据流审查

**Interfaces:**
- Consumes: R1-R9 报告
- Produces: `binance_trader/docs/audit/R10-ml-news-integration.md`

- [ ] **Step 1: 审查 ML 预测器数据流**

审查要点：
- `predictor.py` 的 `predict()` 方法：
  - 特征顺序是否与训练时一致
  - 缺失特征时的 fallback 行为
  - ML confidence 作为信号融合输入 → 如果 ML 模型过时，如何降级
- `trainer.py` 的训练触发条件：
  - 是否有足够数据
  - 训练失败时的模型状态

- [ ] **Step 2: 审查 News 分析器安全性**

审查要点：
- `fetcher.py` 的 HTTP 请求：
  - URL 验证 — 防止 SSRF
  - 超时设置
- `analyzer.py` 的异常检测逻辑：
  - `anomaly_threshold_pct` 和 `volume_spike_multiplier` 的使用
  - 异常触发时紧急抓取的 race condition

- [ ] **Step 3: 审查跨模块事件流完整性**

端到端追踪信号流：
```
WS Kline → MarketData._handle_ws_message → EventBus.publish(MARKET_KLINE)
→ StrategyEngine._on_kline → _evaluate → EventBus.publish(STRATEGY_SIGNAL)
→ RiskManager._on_signal → check_signal → EventBus.publish(ORDER_REQUEST)
→ OrderExecutor._on_order_request → _execute_sim → EventBus.publish(POSITION_UPDATE)
→ RiskManager._on_position_update → update_balance → breaker.set_equity
```

每个环节的中断点分析：
- 如果任一 EventBus.publish 失败，后续链路是否继续？
- 如果 RiskManager 拒绝信号，是否通知 StrategyEngine？
- 如果 OrderExecutor 执行失败，余额是否回滚？

- [ ] **Step 4: 审查并发竞态条件**

全局竞态检查：
- `main.py` 中的 `atomic_adjust_balance` + `risk_manager.update_balance` 调用
- PositionGuard 的 15秒检查循环与策略引擎信号处理重叠
- 同一 symbol 的并发入场信号（_pending_signals 防护是否足够）
- `_pending_signals` 的超时清理缺失

- [ ] **Step 5: 审查资源泄漏**

- `aio.connect()` 是否每次调用 `await db.close()`
- WebSocket 连接管理：`_run_websocket()` 循环中的 socket 泄漏
- asyncio Task 管理：`asyncio.create_task()` 创建的 task 是否可被正确取消

- [ ] **Step 6: 撰写并保存 R10 审计报告**

创建 `binance_trader/docs/audit/R10-ml-news-integration.md`。

- [ ] **Step 7: 更新 cumulative-findings.csv**

---

### Task 12: R11 — 深度收尾审计

**Files 审计范围:**
- 前10轮中出现 Critical/High 级别问题的所有文件
- 逐个问题进行深度追踪

**Interfaces:**
- Consumes: R1-R10 所有报告
- Produces: `binance_trader/docs/audit/R11-deep-audit-summary.md`

- [ ] **Step 1: 汇总前10轮所有 Critical/High 问题**

从 `cumulative-findings.csv` 提取所有 Critical 和 High 级别的问题，按模块分组。

- [ ] **Step 2: 逐问题深度追踪**

对每个 Critical/High 问题：
- 追踪完整代码路径（从触发到影响）
- 检查同一文件中是否有类似模式的其他问题
- 评估修复后是否引入新风险

- [ ] **Step 3: 跨模块交互分析**

对涉及多个模块的问题（如余额操作链），分析跨模块交互中的隐性假设：
- 模块 A 假设模块 B 已经验证了 X
- 模块 B 假设模块 A 不会在 Y 情况下调用
- 这些假设是否成立？如果不成立，后果是什么？

- [ ] **Step 4: 撰写 R11 深度审计报告**

创建 `binance_trader/docs/audit/R11-deep-audit-summary.md`，包含：
- 所有 Critical/High 问题的深度分析
- 修复优先级排序
- 系统性改进建议

- [ ] **Step 5: 生成最终漏洞汇总表**

更新 `cumulative-findings.csv` 并创建 `fix-tracker.md`：
```markdown
# 修复进度跟踪

| ID | 等级 | 状态 | 修复人 | 修复日期 | 验证结果 |
|----|------|------|--------|----------|----------|
| R1-001 | High | pending | - | - | - |
```

---

### Task 13: 开发方向建议文档

**Files:**
- Create: `binance_trader/docs/development-roadmap.md`

**Interfaces:**
- Consumes: 所有审计报告
- Produces: 开发路线图文档

- [ ] **Step 1: 撰写当前架构优势与瓶颈分析**

- 事件驱动架构的优势
- 当前单进程架构的扩展性瓶颈
- 数据存储方案的局限性

- [ ] **Step 2: 撰写短期改进计划（1-2周）**

基于审计发现的优先级：
- 修复所有 Critical 问题
- 修复所有 High 问题
- 补全止损执行逻辑
- 改进余额操作的原子性

- [ ] **Step 3: 撰写中期演进计划（1-3月）**

- 策略系统增强（multi-position per symbol）
- ML 管线升级（online learning）
- 风险模型完善（VaR, CVaR, 压力测试）
- 回测系统完善（walk-forward 自动化）

- [ ] **Step 4: 撰写长期愿景（3-12月）**

- 多交易所支持（OKX, Bybit）
- 分布式部署（策略执行与 Web UI 分离）
- 策略市场（社区共享策略模板）
- 移动端适配

- [ ] **Step 5: 保存文档**

```bash
git add binance_trader/docs/development-roadmap.md
```

---

### Task 14: 核心算法详解文档

**Files:**
- Create: `binance_trader/docs/core-algorithms/01-strategy-signal-fusion.md`
- Create: `binance_trader/docs/core-algorithms/02-risk-check-pipeline.md`
- Create: `binance_trader/docs/core-algorithms/03-circuit-breaker.md`
- Create: `binance_trader/docs/core-algorithms/04-position-sizing-kelly.md`
- Create: `binance_trader/docs/core-algorithms/05-hybrid-backtest-engine.md`
- Create: `binance_trader/docs/core-algorithms/06-ga-evolution.md`
- Create: `binance_trader/docs/core-algorithms/07-deflated-sharpe-ratio.md`
- Create: `binance_trader/docs/core-algorithms/08-ml-triple-barrier.md`
- Create: `binance_trader/docs/core-algorithms/09-trailing-stop-algorithm.md`

**Interfaces:**
- Consumes: 源代码分析 + 学术文献
- Produces: 9篇算法详解文档

- [ ] **Step 1: 撰写信号融合算法文档 (01)**

内容大纲：
1. **算法原理**：加权线性融合的数学基础、为什么用线性组合而非非线性
2. **本项目实现**：`final_score = (indicator*w_ind + ml*w_ml + news*w_news) / total_weight`
3. **相关研究**：Ensemble methods, stacked generalization, Bayesian model averaging
4. **效用分析**：各权重组合在回测中的夏普比率对比
5. **改进策略**：动态权重（Kalman filter）、非线性融合（XGBoost meta-model）

- [ ] **Step 2: 撰写7步风控管线文档 (02)**

内容大纲：
1. **算法原理**：Sequential decision pipeline, each step is a binary gate
2. **本项目实现**：熔断→敞口→仓位→杠杆→止损→同币种→最大交易数
3. **相关研究**：Pre-trade risk controls in electronic trading (SEC Rule 15c3-5)
4. **效用分析**：各步骤拒绝率统计
5. **改进策略**：并行化非依赖步骤、可配置的步骤顺序

- [ ] **Step 3: 撰写熔断器文档 (03)**

内容大纲：
1. **算法原理**：状态机模型、三个触发条件（回撤%、亏损$、连续亏损次数）
2. **本项目实现**：CircuitBreaker dataclass、check() 方法、daily/weekly reset
3. **相关研究**：Trading circuit breakers in financial markets (NYSE, crypto exchanges)
4. **效用分析**：熔断器触发频率与最大回撤关系
5. **改进策略**：动态阈值（基于近期波动率自适应）、分级响应（partial pause vs full halt）

- [ ] **Step 4: 撰写仓位计算文档 (04)**

内容大纲：
1. **算法原理**：Kelly Criterion 及其变体
   - Full Kelly: `f* = (p*b - q) / b`
   - Half Kelly: `f* / 2`
2. **本项目实现**：Fixed-fraction position sizing（非严格 Kelly）
   - `capital_pool = balance * cap_pct; risk = capital_pool * pos_size_pct / 100`
3. **相关研究**：Kelly (1956), Thorp (1997), 实际交易中的 Kelly 修正
4. **效用分析**：不同 position_size_pct 下的回测表现对比
5. **改进策略**：实现动态 Kelly sizing、考虑持仓相关性

- [ ] **Step 5: 撰写混合回测引擎文档 (05)**

内容大纲：
1. **算法原理**：Vectorized signal computation + Event-driven trade execution
2. **本项目实现**：SignalMatrixBuilder → EventDrivenExecutor 两阶段流水线
3. **相关研究**：Vectorized backtesting (Bt, Zipline, VectorBT 的设计对比)
4. **效用分析**：Hybrid vs Legacy 性能对比数据
5. **改进策略**：GPU 加速信号计算、multi-asset portfolio 优化

- [ ] **Step 6: 撰写遗传算法文档 (06)**

内容大纲：
1. **算法原理**：GA 的 5 个核心算子
2. **本项目实现**：4种基因类型、10种可进化指标、适应度公式
3. **相关研究**：GA in trading (Prado 2020), NSGA-II for multi-objective optimization
4. **效用分析**：GA 优化前后策略表现对比
5. **改进策略**：CMA-ES、Bayesian Optimization、多目标优化（Sharpe + Stability）

- [ ] **Step 7: 撰写 DSR 文档 (07)**

内容大纲：
1. **算法原理**：Bailey & López de Prado (2014) 的 Deflated Sharpe Ratio
   - `E[max(SR)] = sqrt(1/T) * sqrt(2*log(N))`
2. **本项目实现**：`core/ga/fitness.py` 中的 DSR 计算
3. **相关研究**：多重测试校正 (Bonferroni, Holm, BHY)
4. **效用分析**：DSR 过滤前后策略的样本外表现
5. **改进策略**：Probabilistic Sharpe Ratio (PSR), 考虑偏度和峰度

- [ ] **Step 8: 撰写 Triple Barrier 标签法文档 (08)**

内容大纲：
1. **算法原理**：Prado (2018) 的 Triple Barrier Method
   - Upper barrier (take profit), Lower barrier (stop loss), Time barrier (horizon)
2. **本项目实现**：`core/ml/features.py` 中的标签构建
3. **相关研究**：Fixed-time horizon vs path-aware labeling
4. **效用分析**：Triple Barrier vs Binary label 的 ML 准确率对比
5. **改进策略**：动态 barrier 宽度（基于波动率）、meta-labeling

- [ ] **Step 9: 撰写移动止损文档 (09)**

内容大纲：
1. **算法原理**：Trailing stop as a ratchet mechanism
2. **本项目实现**：PositionGuard 的 `_update_trailing_stop()`
   - Long: `new_sl = max(price*(1-d), current_sl, floor_sl)`
   - Short: `new_sl = min(price*(1+d), current_sl, ceiling_sl)`
3. **相关研究**：Chandelier Exit, Parabolic SAR, SuperTrend
4. **效用分析**：固定止损 vs 移动止损的回测表现
5. **改进策略**：ATR-based trailing stop, multi-level trailing

- [ ] **Step 10: 保存所有算法文档**

```bash
git add binance_trader/docs/core-algorithms/
git commit -m "docs: add core algorithm documentation (9 chapters)"
```

---

### Task 15: 最终整合与提交

**Files:**
- Modify: `binance_trader/docs/audit/README.md` (更新最终状态)
- Modify: `binance_trader/docs/audit/cumulative-findings.csv` (最终汇总)
- Create: `binance_trader/docs/audit/fix-tracker.md` (修复跟踪)

- [ ] **Step 1: 生成最终统计**

从 cumulative-findings.csv 计算：
- 各严重等级的总数
- 各模块的问题分布
- 按优先级排序的修复列表

- [ ] **Step 2: 更新审计 README**

添加最终统计汇总和结论。

- [ ] **Step 3: 创建修复跟踪文件**

初始化 fix-tracker.md 并标记所有问题的初始状态。

- [ ] **Step 4: 最终 Git 提交**

```bash
git add binance_trader/docs/audit/ binance_trader/docs/core-algorithms/ binance_trader/docs/development-roadmap.md
git commit -m "audit: complete 11-round code audit + algorithm docs + roadmap"
```

---

## 文件结构总览

```
binance_trader/docs/
├── audit/
│   ├── README.md
│   ├── severity-guidelines.md
│   ├── cumulative-findings.csv
│   ├── fix-tracker.md
│   ├── R1-strategy-engine.md
│   ├── R2-risk-management-1.md
│   ├── R3-risk-management-2.md
│   ├── R4-order-executor.md
│   ├── R5-market-data.md
│   ├── R6-backtest-engine.md
│   ├── R7-ai-controller.md
│   ├── R8-web-auth-api.md
│   ├── R9-database-config-alerts.md
│   ├── R10-ml-news-integration.md
│   └── R11-deep-audit-summary.md
├── core-algorithms/
│   ├── 01-strategy-signal-fusion.md
│   ├── 02-risk-check-pipeline.md
│   ├── 03-circuit-breaker.md
│   ├── 04-position-sizing-kelly.md
│   ├── 05-hybrid-backtest-engine.md
│   ├── 06-ga-evolution.md
│   ├── 07-deflated-sharpe-ratio.md
│   ├── 08-ml-triple-barrier.md
│   └── 09-trailing-stop-algorithm.md
├── development-roadmap.md
└── superpowers/
    ├── specs/2026-07-16-code-audit-design.md
    └── plans/2026-07-16-code-audit-plan.md
```
