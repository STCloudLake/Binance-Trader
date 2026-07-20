# 审计报告 R11 — 深度收尾审计

**日期**: 2026-07-16
**审计范围**: 前 10 轮中所有 Critical 和 High 级别问题的深度追踪
**审计方法**: 跨模块交互分析 + 根因归类 + 修复优先级排序

---

## 1. Critical 问题汇总 (10项)

| ID | 轮次 | 模块 | 简述 | 根因类别 |
|----|------|------|------|----------|
| R1-001 | R1 | Strategy | `pd.eval()` 代码注入 | 输入验证 |
| R2-001 | R2 | Risk | 回撤从 peak_equity 跨日计算 | 状态管理 |
| R3-001 | R3 | Risk | 止损设置但从未 enforce | **逻辑缺失** |
| R3-002 | R3 | Risk | AI 降仓位被 floor 覆盖 | 设计冲突 |
| R4-001 | R4 | Executor | 余额操作非原子 | 数据一致性 |
| R6-001 | R6 | Backtest | 回测 AND vs 实盘 OR 逻辑 | 引擎不一致 |
| R7-001 | R7 | AI | AI 可设无上限仓位 | 输入验证 |
| R8-001 | R8 | Auth | JWT secret 每次重启随机 | 配置管理 |
| R9-001 | R9 | DB | 余额非事务（与 R4-001 联动） | 数据一致性 |
| R10-001 | R10 | Integration | 交易信号链无补偿事务 | 数据一致性 |

## 2. High 问题汇总 (26项)

| 类别 | 数量 | 代表性问题 |
|------|------|-----------|
| 数据一致性 | 6 | R2-002 Exposure 用旧价格、R4-004 PNL 不含手续费、R5-001 价格缓存 |
| 状态管理 | 5 | R2-003 pending_signals 超时、R2-004 仓位数据过期、R3-003 min notional |
| 引擎不一致 | 3 | R6-002 ML 前瞻偏差、R6-003 Hybrid vs Legacy 仓位计算 |
| 输入验证 | 3 | R7-002 API 异常吞没、R7-003 权重无边界、R8-004 登录无限速 |
| 安全性 | 4 | R8-002 WS 无认证、R8-003 Session 脆弱、R9-002 Schema 迁移、R9-003 Secrets 权限 |
| 错误处理 | 3 | R4-002 order_id 碰撞、R10-002 特征顺序、R10-003 SSRF |
| 并发/竞态 | 2 | R4-003 仓位覆盖、R5-002 重连策略 |

---

## 3. 系统性根因分析

### 根因 1: 交易信号链无事务性 (最严重)

**影响范围**: R10-001, R4-001, R9-001, R2-003, R4-005

5个问题指向同一个根本原因：**从信号到执行的完整链路缺少 ACID 保证**。

```
受影响路径:
  StrategyEngine → RiskManager → OrderExecutor → Database
  
  每个环节独立运行，任一失败无补偿
```

**修复策略**: 实现轻量级 Saga 模式：
1. 每步记录操作日志（event sourcing）
2. 失败时执行补偿操作（undo）
3. 定期对账（DB vs executor vs exchange）

### 根因 2: 回测与实盘逻辑分叉 (最隐蔽)

**影响范围**: R6-001, R6-002, R6-003, R6-009

策略条件评估（AND vs OR）、仓位计算（固定 vs 策略特定）、ML 使用方式（有/无）在回测和实盘中使用不同代码路径。

```
回测路径: BacktestEngine._evaluate → AND logic → strategy-specific sizing
实盘路径: StrategyEngine._evaluate → OR logic → global soft_params sizing
```

**修复策略**: 提取共享的评估核心（Shared Evaluation Kernel），回测和实盘调用同一代码。

### 根因 3: 风控参数分层不一致 (最危险)

**影响范围**: R3-001, R3-002, R7-001

止损设置了三层（PositionSizer 计算 → PositionGuard 更新 → 策略引擎的 exit conditions），但**没有一层真正执行止损检查**。

```
PositionSizer → 计算止损价格 ✓
PositionGuard → 更新 trailing stop ✓  
Executor._execute_sim → 无止损检查 ✗ (缺失环节)
StrategyEngine._evaluate → 无止损条件 ✗
```

**修复策略**: 在 Executor 或 PositionGuard 中添加止损触发检查（优先级最高）。

### 根因 4: 状态管理依赖异步事件

**影响范围**: R2-001, R2-003, R2-004, R2-007

CircuitBreaker 的日重置依赖后台定时任务、仓位状态依赖 POSITION_UPDATE 事件到达、余额同步依赖异步回调。任一异步环节失败导致状态不一致。

**修复策略**: 状态变更改为主动查询 + 被动通知双模式。所有定时重置增加基于当前时间的自动判断。

---

## 4. 修复优先级矩阵

### P0 — 立即修复（涉及资金安全）

| Priority | ID | 问题 | 预计工时 |
|----------|----|------|----------|
| P0-1 | R3-001 | 止损不执行 — 在 PositionGuard 添加止损触发 | 2h |
| P0-2 | R6-001 | 回测与实盘条件逻辑统一 | 3h |
| P0-3 | R4-001 | 余额操作事务化 | 4h |
| P0-4 | R1-001 | pd.eval 代码注入修复 | 2h |
| P0-5 | R7-001 | AI 参数上限保护 | 1h |
| P0-6 | R3-002 | AI floor 覆盖问题 | 1h |

**P0 总计: 13 小时**

### P1 — 24小时内修复（高风险）

| ID | 问题 | 预计工时 |
|----|------|----------|
| R2-001 | 回撤跨日计算 | 2h |
| R2-003 | pending_signals 超时 | 1h |
| R4-003 | 仓位覆盖检测 | 1h |
| R8-001 | JWT secret 持久化 | 2h |
| R8-002 | WS 认证 | 3h |
| R10-003 | SSRF 防护 | 1h |
| R6-002 | ML 前瞻偏差 | 2h |
| R7-002 | API 异常分类处理 | 2h |

**P1 总计: 14 小时**

### P2 — 1周内修复（稳定性）

| ID | 问题 | 预计工时 |
|----|------|----------|
| R2-002 | Exposure 实时计算 | 2h |
| R2-004 | 仓位数据统一 | 2h |
| R4-002 | order_id UUID | 0.5h |
| R5-001 | 价格缓存分层 | 2h |
| R6-003 | Hybrid sizing 统一 | 2h |
| R8-004 | 登录限速 | 2h |
| R9-002 | Schema 迁移改进 | 3h |
| R9-003 | Secrets 权限检查 | 0.5h |
| R10-002 | ML 特征顺序 | 1h |

**P2 总计: 15 小时**

### P3 — 下个迭代（代码质量，31项 Medium/Low）

预计 20-30 小时，分批修复。

---

## 5. 跨模块交互分析

### 脆弱链路 1: 余额同步

```
调用点:
  main.py:_on_position_exit     → atomic_adjust_balance → risk_manager.update_balance
  main.py:_on_position_reduce   → atomic_adjust_balance → risk_manager.update_balance
  main.py:_execute_breaker_action → atomic_adjust_balance → risk_manager.update_balance
  position_guard:_emergency_close → atomic_adjust_balance → risk_manager.update_balance
  executor:_execute_sim         → atomic_adjust_balance → risk_manager.update_balance
```

**问题**: 5 个独立路径各自管理余额 → 任一异常导致不一致。

### 脆弱链路 2: Position 生命周期

```
创建: executor._execute_sim → self._positions[sym] = {...}
更新: position_guard._update_trailing_stop → pos["stop_loss"] = ...
      risk_manager._on_position_update → self._open_positions[sym] = merged
删除: executor.close_position → del self._positions[symbol]
```

**问题**: 三个模块（Executor, RiskManager, PositionGuard）各自维护 position 状态 → 三份独立副本。

---

## 6. 架构改进建议（概要）

1. **引入 Event Sourcing**: 所有状态变更记录为不可变事件，状态=事件重放
2. **统一 Position Store**: 单一声源，Executor 为唯一写入者，其他模块只读
3. **Shared Evaluation Kernel**: 提取回测和实盘共享的信号评估核心
4. **配置版本化**: 所有运行时配置修改记录版本号和操作者
5. **健康检查端点**: `/health` 端点报告各组件连接状态

（详细架构建议见 `development-roadmap.md`）

---

## 7. 最终统计

| 等级 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | **总计** |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|----------|
| Critical | 1 | 1 | 2 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | **10** |
| High | 4 | 3 | 1 | 3 | 2 | 2 | 3 | 3 | 2 | 3 | **26** |
| Medium | 5 | 3 | 2 | 4 | 4 | 4 | 3 | 4 | 3 | 3 | **35** |
| Low | 3 | 2 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | **19** |
| **总计** | **13** | **9** | **6** | **9** | **8** | **9** | **9** | **10** | **8** | **9** | **90** |

**审计完成。** 10 轮模块审计 + 1 轮深度分析 = **90 个漏洞**。4 个系统性根因已识别，修复优先级已排序。
