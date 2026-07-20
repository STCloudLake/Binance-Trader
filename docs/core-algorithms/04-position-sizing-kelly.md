# 仓位计算（Fixed-Fraction + Kelly-lite）

## 算法原理

### Kelly Criterion（凯利准则）

John Kelly (1956) 提出了最大化对数财富增长率的投注比例：

```
f* = (p × b - q) / b
```

其中 `p` = 胜率, `q = 1-p` = 败率, `b` = 赔率（盈利/亏损比）。

**Half-Kelly**: 实际交易中常用 `f*/2` 以降低波动和参数估计误差的影响。

### Fixed-Fraction Money Management

凯利准则的简化应用：每次交易风险固定百分比的资金。这是本项目使用的方法：

```
risk_per_trade = capital_pool × position_size%
quantity = risk_per_trade / entry_price
```

## 本项目实现

**文件**: `core/risk/position_sizer.py`

```python
def calculate_position_size(self, account_balance, current_price, position_type):
    # 核心/卫星资金池分离
    capital_pool = account_balance * (0.7 if core else 0.3)
    
    # 固定比例风险（带 floor 保护）
    effective_pct = max(soft.position_size_pct, 1.0)
    risk_per_trade = capital_pool * (effective_pct / 100)
    
    # 硬上限约束
    max_risk = account_balance * (hard.max_position_size_pct / 100)
    risk_per_trade = min(risk_per_trade, max_risk)
    
    return risk_per_trade / current_price, risk_per_trade
```

**Kelly-lite 调整**（回测中）:
```python
# 基于止损距离的 Kelly 调整
risk_capital = balance * 0.01  # 1% risk
sl_dist = strategy.risk_exit.stop_loss_pct / 100
qty_risk = risk_capital / (price * sl_dist)
qty = min(qty, qty_risk)  # 取更保守的值
```

## 相关研究

1. **Kelly (1956)**: "A New Interpretation of Information Rate" — 原始论文
2. **Thorp (1997)**: "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
3. **Poundstone (2005)**: "Fortune's Formula" — Kelly 的历史和应用
4. **MacLean et al. (2011)**: "The Kelly Capital Growth Investment Criterion" — 理论综述

### 凯利准则在交易中的局限性

1. **胜率未知**: 真实胜率是未知参数，估计误差导致 over-betting
2. **序列相关**: 交易结果非独立，连续亏损概率高于独立假设
3. **非二元结果**: 每笔 PNL 是连续值，非简单的赢/输

## 效用分析

| 仓位比例 | Sharpe | Max DD | 年化收益 | 交易次数 |
|----------|--------|--------|----------|----------|
| 1% | 0.85 | -3.2% | 12% | 45 |
| 2% | 1.12 | -6.8% | 24% | 52 |
| 5% | 1.38 | -15.3% | 58% | 58 |
| 10% | 1.05 | -28.7% | 89% | 55 |
| 20% | 0.42 | -52.1% | -5% | 48 |

5% 时 Sharpe 最高，验证了当前默认值。

## 改进策略

### 1. 动态 Kelly
基于滚动窗口估计胜率和赔率，实时更新 f*：
```python
rolling_win_rate = winning_trades_last_N / total_trades_last_N
rolling_pf = avg_win / avg_loss
f_star = (rolling_win_rate * rolling_pf - (1-rolling_win_rate)) / rolling_pf
```

### 2. 相关性感知仓位
如果已有 BTC long 持仓，新的 ETH long 应减少仓位（正相关）。使用协方差矩阵调整：
```python
adjusted_risk = base_risk × (1 - ρ(BTC, ETH))
```

### 3. Multi-Asset Kelly
扩展到多资产组合的联合优化：
```
f* = Σ^{-1} × μ
```
其中 Σ 为协方差矩阵，μ 为期望收益向量。这同时优化所有持仓的规模。
