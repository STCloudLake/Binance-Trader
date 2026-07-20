# 审计报告 R3 — Risk Management Ⅱ (Position Sizer + Position Guard)

**日期**: 2026-07-16
**审查文件**: `core/risk/position_sizer.py`, `core/risk/position_guard.py`
**审查代码行数**: ~206 行
**审计方法**: 公式推导验证 + 止损触发链追踪 + 边界条件测试

---

## 摘要

| 等级 | 数量 |
|------|------|
| Critical | 2 |
| High | 1 |
| Medium | 2 |
| Low | 1 |
| **总计** | **6** |

---

## 漏洞清单

| ID | 等级 | 文件:行 | 描述 |
|----|------|---------|------|
| R3-001 | **Critical** | position_guard.py:108-151 | **止损从未被 enforce** — stop_loss 值在内存中更新但无任何代码检查价格是否触发止损并执行平仓 |
| R3-002 | **Critical** | position_sizer.py:20-21 | soft position_size_pct floor 1% 在 AI full_auto 模式下可能覆盖失效意图 |
| R3-003 | High | position_sizer.py:25 | quantity 无最小 notional 检查和取整 — Binance 有 min notional 限制 |
| R3-004 | Medium | position_guard.py:15 | 15秒检查间隔 — 价格可能在检查间隙跌破止损而不被检测 |
| R3-005 | Medium | position_sizer.py:12-17 | capital_pool 使用总余额而非可用余额 — 未扣除已占用资金 |
| R3-006 | Low | position_guard.py:140-141 | trailing stop 的 0.01 更新阈值是硬编码魔法数字 |

---

## 详细分析

### R3-001 🔴 Critical — 止损设置但从未执行

**文件**: `core/risk/position_guard.py:108-151`

**描述**: 这是本次审计中发现的最严重的交易逻辑缺陷。

PositionGuard 的 `_update_trailing_stop()` 方法正确地计算并更新了 `pos["stop_loss"]` 的值（在 executor 的 `_positions` dict 中）。**但是，没有任何代码检查当前价格是否已触发 stop_loss 并执行平仓。**

追踪止损触发链路：
1. `_guard_loop()` → `_check_all_positions()` (interval=15s)
2. `_check_all_positions()` → 检查 emergency_stop → 调用 `_update_trailing_stop()`
3. `_update_trailing_stop()` → 更新 `pos["stop_loss"]` 值 → **不检查是否触发**

策略引擎中的 exit conditions 也**不引用 stop_loss 值**。策略条件使用指标值（如 `rsi > 70`），但 `stop_loss` 是存储在 executor.position 中的自定义字段，不在 DataFrame 列中，无法被 `evaluate_condition()` 引用。

**止损的三层失效**：
- `position_guard.py`: 计算止损值但不检查触发
- `strategy/engine.py`: exit conditions 不包含 stop_loss 检查
- `executor/executor.py`: `_execute_sim` 中无止损监控

**唯一在回测中可用的止损**: 回测引擎 `EventDrivenExecutor` 有自己的止损检查逻辑（event_executor.py 的 `_check_exits` 方法），但那是独立的回测代码路径，实盘/模拟盘不使用。

**复现路径**: 
1. 模拟盘中以 100 USDT 开仓 BTC long，止损设在 98 USDT
2. 价格跌至 95 USDT
3. 止损从未触发 → 仓位继续亏损 → 直到 emergency_stop (-5%) 或手动平仓

**修复建议**: 在 `_check_all_positions()` 的 emergency_stop 检查**之前**添加止损触发检查：
```python
# Check stop-loss BEFORE emergency stop
if pos.get("stop_loss"):
    if (side == "long" and price <= pos["stop_loss"]) or \
       (side == "short" and price >= pos["stop_loss"]):
        await self._stop_loss_close(symbol, pos, price)
        continue
```

---

### R3-002 🔴 Critical — AI 降低仓位意图被 floor 覆盖

**文件**: `core/risk/position_sizer.py:20-21`

```python
effective_pct = max(self.soft.position_size_pct, 1.0)
```

**描述**: 当 AI 在 full_auto 模式下判定市场极度危险，将 `position_size_pct` 设为 0 或 0.5%（意图暂停交易或极小仓位），`max(0.5, 1.0)` 会覆盖为 1.0%。虽然 hard limit 更关心上限，但这个行为在以下场景中危险：

1. AI 检测到市场崩盘 → 设置 `position_size_pct = 0.1` 意图极小仓位
2. floor 强制为 1.0% → 仓位比 AI 预期大 10 倍
3. 在 10000 USDT 账户中，AI 期望每次 7 USDT（satellite 30% × 0.1%），实际为 30 USDT

**设计意图冲突**: 硬风控的 floor 本意是防止误配置导致零交易，但它覆盖了 AI 的合理风控决策。

**复现路径**: AI 在风险调整中将 position_size_pct 设为极低值 → 被 floor 覆盖 → 实际仓位远超 AI 预期 → 高波动期过度交易。

**修复建议**: 区分"用户手动设置"和"AI 动态调整"两种场景，仅在用户手动设置时应用 floor：
```python
effective_pct = self.soft.position_size_pct
if self.soft.position_size_pct > 0 and effective_pct < 1.0:
    if ai_adjusted:
        effective_pct = max(effective_pct, 0.1)  # AI can go very low
    else:
        effective_pct = max(effective_pct, 1.0)  # User gets conservative floor
```

---

### R3-003 🟠 High — 无最小交易量检查

**文件**: `core/risk/position_sizer.py:25`

```python
quantity = risk_per_trade / current_price if current_price > 0 else 0
```

**描述**: 计算出的 `quantity` 未经验证：
1. **Binance 最小名义价值**: 每笔订单至少 10 USDT（或等价）。如果 `risk_per_trade` 太小，quantity 可能不满足交易所要求
2. **Step size**: 大多数交易对有最小数量步长（如 BTCUSDT 的 step size = 0.001）。未取整的 quantity 会被交易所拒绝
3. **Min notional**: 某些交易对有最小名义价值限制（如 5 USDT）

这些在实盘模式 (`_execute_live`) 中会导致订单被 Binance API 拒绝，但在模拟模式中被静默接受。

**复现路径**: 使用小额账户 + 高价币（如 BTC 100,000 USDT）+ 1% 仓位 → risk = 1000 USDT × 1% = 10 USDT → quantity = 0.0001 BTC。如果 step size 是 0.001，则被拒绝。

**修复建议**: 添加 quantity 验证：
```python
min_notional = 10.0  # or per-symbol config
if quantity * current_price < min_notional:
    return 0, 0  # insufficient for minimum trade
```

---

### R3-004 🟡 Medium — 15秒检查间隔可能错过止损

**文件**: `core/risk/position_guard.py:15,44`

```python
self._check_interval_sec = 15
await asyncio.sleep(self._check_interval_sec)
```

**描述**: 在加密货币市场中，15秒足以发生显著的价格变动。极端情况下（闪崩），价格可能在两次检查之间下跌远超止损位。

示例：BTC 从 100,000 跌至 95,000 用了 5 秒 → 止损设在 98,000 → 15秒后才检查 → 此时价格可能是 95,000（已远超止损）。

**复现路径**: 高波动行情中 → 价格快速穿越止损位 → 下一次 guard 检查时价格已远远超过止损 → 实际平仓价格远差于止损价格。

**修复建议**: 在 executor 层面实现基于 WebSocket 实时价格的止损监控，而非依赖 15 秒轮询。

---

### R3-005 🟡 Medium — 资金池计算忽略已占用资金

**文件**: `core/risk/position_sizer.py:12-17`

```python
if position_type == "core":
    capital_pool = account_balance * self.core_capital_pct
else:
    capital_pool = account_balance * self.satellite_capital_pct
```

**描述**: `capital_pool` 基于总余额的百分比计算，但未扣除已开仓占用的资金。如果已经开了 5 个仓位，可用资金实际上少于 `account_balance * cap_pct`。

这可能导致超额配置：在总敞口检查（R2-002）触发之前，每笔独立交易的仓位都在允许范围内，但合计可能超出预期。

**复现路径**: 已有多个仓位占用了 50% 资金 → 新信号到达 → capital_pool 仍按 100% balance 计算 → 新仓位可能大于剩余可用资金。

**修复建议**:
```python
available_balance = account_balance - existing_invested
capital_pool = available_balance * cap_pct
```

---

### R3-006 🟢 Low — 更新阈值硬编码

**文件**: `core/risk/position_guard.py:140-141`

```python
elif side == "long" and new_sl > current_sl + 0.01:
    should_update = True
```

`0.01` 的阈值是一个未文档化的魔法数字。对于价格 0.01 USDT 的代币，0.01 = 100%；对于价格 100,000 USDT 的 BTC，0.01 = 0.00001%。应基于价格百分比而非绝对值。

**修复建议**: 使用 `new_sl > current_sl * (1 + min_update_pct / 100)` 替代绝对差值。

---

## 累计统计

| 等级 | R1 | R2 | R3 | 累计 |
|------|-----|-----|-----|------|
| Critical | 1 | 1 | 2 | 4 |
| High | 4 | 3 | 1 | 8 |
| Medium | 5 | 3 | 2 | 10 |
| Low | 3 | 2 | 1 | 6 |
| **总计** | **13** | **9** | **6** | **28** |
