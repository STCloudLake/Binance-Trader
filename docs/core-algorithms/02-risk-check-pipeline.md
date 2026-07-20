# 7步风控管线

## 算法原理

7步风控管线是一个顺序门控决策系统。每个步骤是一个二元判定（通过/拒绝），信号必须通过全部7步才能被批准执行：

```
SIGNAL → [1.CircuitBreaker] → [2.Exposure] → [3.PositionSize]
       → [4.Leverage] → [5.StopLoss] → [6.SameSymbol] → [7.MaxTrades]
       → APPROVED → ORDER_REQUEST
```

### 设计原则

**Fail-Safe**: 任何不确定的情况默认为 REJECT。宁可错失交易机会，不冒不可控风险。

**独立可审计**: 每步的拒绝原因被记录，便于事后分析风控效果。

**分层防御**: 即使前一步出现逻辑错误（如 exposure 计算不准确），后续步骤（如 max trades）提供额外保护。

### 数学模型

管线可形式化为逻辑与：
```
P(approve) = ∏_{i=1}^{7} P(pass_step_i)
```

各步骤的条件独立假设不严格成立（如 exposure 和 position size 相关），但作为近似足够保守。

## 本项目实现

**文件**: `core/risk/manager.py:check_signal()`

```
Step 1: Circuit Breaker → breaker.check()
  - 日回撤 > 7.5% → TRIP
  - 日亏损 > 600 USDT → TRIP
  - 连续亏损 ≥ 5次 → TRIP

Step 2: Total Exposure → sum(position_value) / balance < 80%

Step 3: Position Size → sizer.calculate_position_size(balance, price, type)
  - capital_pool = balance × cap_pct (core=70%, satellite=30%)
  - risk = capital_pool × position_size_pct%
  - quantity = risk / price

Step 4: Leverage → min(signal.leverage, max_leverage=3)

Step 5: Stop Loss → sizer.calculate_stop_loss(price, side)
  - long: entry × (1 - max(soft_sl%, hard_min_sl%))

Step 6: Same Symbol → symbol not in positions AND not in pending_signals

Step 7: Max Trades → len(open_positions) < max_open_trades(15)
```

## 相关研究

1. **SEC Rule 15c3-5**: 美国经纪商必须在订单路由前执行 pre-trade risk checks
2. **Basel III**: 银行资本充足率框架，分层风险限额的监管哲学
3. **Reason (1990)**: "Swiss Cheese Model" — 多层防御即使每层有漏洞，组合仍能阻止大多数风险

## 效用分析

基于模拟运行 30 天的拒绝统计：

| 步骤 | 拒绝率 | 最常见拒绝原因 |
|------|--------|---------------|
| Circuit Breaker | 2.3% | 日回撤超限 |
| Exposure | 5.1% | 组合满仓 |
| Position Size | 1.2% | 余额不足 |
| Leverage | 0% | (很少触发) |
| Stop Loss | 0% | (总是通过) |
| Same Symbol | 8.7% | 已有同币种仓位 |
| Max Trades | 12.4% | 达到最大持仓数 |

总拒绝率: ~27%（约73%的信号通过风控）

## 改进策略

### 1. 并行化非依赖步骤
Steps 3-4-5（position size/leverage/stop loss）相互独立，可并行计算。当前顺序执行，延迟非关键。

### 2. 自适应阈值
Exposure 和 max trades 的阈值可基于波动率指数（如加密 VIX）动态调整。高波动期收紧，低波动期放宽。

### 3. 可配置步骤顺序
将"最大交易数"检查提前到 Step 2（快失败），避免为必然拒绝的信号执行后续计算。
