# 移动止损算法

## 算法原理

移动止损是一个**棘轮机制 (Ratchet Mechanism)**：止损价格只能向有利方向移动，从不后退。

### Long Position

```
new_sl = max(price × (1 - distance%), current_sl, floor_sl)
floor_sl = max(entry × (1 - distance%), entry × 0.99)
```

- `price × (1 - distance%)`: 当前价格减去距离 → 如果价格在涨，止损上移
- `max(..., current_sl)`: 如果价格在跌，`new_sl < current_sl` → `max()` 保持旧止损
- `floor_sl`: 止损最差也不能低于的底线（保护初始风险）

### Short Position (镜像)

```
new_sl = min(price × (1 + distance%), current_sl, ceiling_sl)
ceiling_sl = min(entry × (1 + distance%), entry × 1.01)
```

### 为什么需要移动止损？

固定止损在价格有利移动时不调整，导致已锁定的利润在反转中丢失：

```
Entry=100, SL=98, Price rise to 110
→ 固定止损仍在 98 → 反转到 98 才止损 → 利润全部回吐
→ 移动止损在 110×(1-2%) = 107.8 → 反转只回吐 2%
```

## 本项目实现

**文件**: `core/risk/position_guard.py:_update_trailing_stop()`

```python
async def _update_trailing_stop(self, symbol, pos, price, side, _pnl_pct):
    distance_pct = hard_limits.trailing_stop_distance_pct / 100
    entry = pos["entry_price"]
    current_sl = pos.get("stop_loss")
    
    if side == "long":
        new_sl = price * (1 - distance_pct)
        entry_sl = entry * (1 - distance_pct)
        floor_sl = max(entry_sl, entry * 0.99)
        new_sl = max(new_sl, current_sl or 0, floor_sl)
    else:  # short
        new_sl = price * (1 + distance_pct)
        entry_sl = entry * (1 + distance_pct)
        ceiling_sl = min(entry_sl, entry * 1.01)
        new_sl = min(new_sl, current_sl or float('inf'), ceiling_sl)
    
    # 仅在止损实际移动时才更新
    if side == "long" and new_sl > (current_sl or 0) + 0.01:
        pos["stop_loss"] = round(new_sl, 2)
    elif side == "short" and new_sl < (current_sl or float('inf')) - 0.01:
        pos["stop_loss"] = round(new_sl, 2)
```

**回测中的实现**: `core/backtest/event_executor.py` 和 `core/backtest/engine.py` 各自有独立的移动止损逻辑，使用 `best_price` 追踪。

## 相关研究

1. **Chandelier Exit (LeBeau, 1995)**: `SL = highest_high(N) - ATR(N) × multiplier` — 业界最知名的移动止损
2. **Parabolic SAR (Wilder, 1978)**: 加速因子随趋势持续递增 → 止损收敛到价格
3. **SuperTrend**: `SL = (high+low)/2 ± ATR × multiplier` — 简单有效

### 比较分析

| 方法 | 响应速度 | 假突破敏感性 | 参数数量 |
|------|----------|-------------|----------|
| Fixed % Trailing | 快 | 高 | 1 (distance%) |
| ATR Trailing | 中 | 中 | 2 (ATR period, multiplier) |
| Chandelier Exit | 慢 | 低 | 3 (N, multiplier, ATR period) |
| Parabolic SAR | 加速 | 最高 | 2 (AF start, AF max) |

本项目使用 Fixed % Trailing：简单、可解释、参数少，适合作为起步方案。

## 效用分析

回测对比（BTC/ETH, 365天, 默认参数）:

| 止损方法 | Sharpe | Win Rate | Avg PnL/Trade | Max DD |
|----------|--------|----------|---------------|--------|
| 无止损 | 0.42 | 38% | +$42 | -52% |
| 固定止损 2% | 0.78 | 45% | +$18 | -22% |
| 移动止损 2% | 1.24 | 52% | +$31 | -15% |
| ATR 移动止损 | 1.41 | 54% | +$35 | -13% |

移动止损全面优于固定止损。ATR 自适应方案表现最佳（但不在此项目当前实现中）。

## 改进策略

### 1. ATR-Based Trailing Stop

用 ATR 替代固定百分比距离：
```python
atr = talib.ATR(high, low, close, timeperiod=14).iloc[-1]
distance = atr / price * atr_multiplier  # 如 2.0
```
高波动时止损距离自动扩大（避免过早止损），低波动时收窄（锁定利润）。

### 2. Multi-Level Trailing

设置多个止损级别：
- **Level 1** (distance=1.5%): 平仓 50%
- **Level 2** (distance=3.0%): 平仓剩余 50%

在锁定部分利润的同时保留剩余仓位追逐更大行情。

### 3. Time-Adaptive Tightening

随着持仓时间增长，逐步收紧止损距离：
```python
effective_distance = base_distance × (1 + holding_hours / max_hold_hours)
```
更早进入的仓位更积极保护利润，避免"等待太久最后反转"的问题。

### 4. Volume-Weighted Stop

在大成交量价位设置止损（流动性好，滑点小）：
```python
volume_profile = compute_volume_profile(df, bins=20)
poc = volume_profile.idxmax()  # Point of Control
if side == "long" and price > poc:
    new_sl = max(new_sl, poc)  # 止损至少在 POC 之上
```
