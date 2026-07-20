# Deflated Sharpe Ratio (DSR)

## 算法原理

DSR 由 Bailey & López de Prado (2014) 提出，用于校正**多重测试偏差 (Multiple Testing Bias)**。

### 问题

在 GA 进化中，我们测试了 N 个策略（N = population × generations）。即使所有策略都无效（真实 Sharpe = 0），由于随机性，样本内 Sharpe 最高的策略仍呈现正 Sharpe：

```
E[max(SR)] ≈ sqrt(Var[SR]) × sqrt(2 × log(N))
           = sqrt(1/T) × sqrt(2 × log(N))
```

其中 T = 观察期数，N = 尝试策略数。

例如：30 种群 × 15 代 = 450 策略, T = 365 天 → E[max(SR)] ≈ 0.52。即使纯噪声，期望"最佳"Sharpe 也有 0.52！

### DSR 公式

```
DSR = SR_observed - E[max(SR)]

significant ⟺ DSR > 0 ∧ p < 0.05
```

DSR 将观察到的 Sharpe 降低期望的"虚假"部分。只有 DSR > 0 的策略才有统计显著性。

### 统计推导

在零假设 H0（所有策略 SR = 0）下：

```
max(SR) ~ N(0, 1/T) 的 order statistic
E[max] = sqrt(1/T) × Φ^{-1}(1 - 1/N)  # 期望最大值
       ≈ sqrt(1/T) × sqrt(2 × log(N))  # N 大时的渐近近似
```

## 本项目实现

**文件**: `core/ga/fitness.py`

```python
def deflated_sharpe(sharpe, n_trials, n_periods):
    """
    sharpe: observed Sharpe ratio
    n_trials: 总尝试策略数 (population × generations)
    n_periods: 观察期数 (默认 365)
    """
    import numpy as np
    expected_max = np.sqrt(1.0 / n_periods) * np.sqrt(2 * np.log(n_trials))
    dsr = sharpe - expected_max
    return dsr
```

在 GA 冠军验证阶段自动调用：
```python
if dsr <= 0:
    logger.warning("Champion strategy not statistically significant (DSR <= 0)")
```

## 相关研究

1. **Bailey & López de Prado (2014)**: "The Deflated Sharpe Ratio" — 原始论文
2. **Harvey, Liu & Zhu (2016)**: "…and the Cross-Section of Expected Returns" — 提出更高的 hurdle rate（t-stat > 3.0）
3. **Prado (2020)**: "Advances in Financial ML" — Chapter 14, Backtesting Statistics

### DSR vs 其他多重比较校正

| 方法 | 保守性 | 适用场景 |
|------|--------|----------|
| Bonferroni | 过于保守 | 独立假设下 |
| Holm-Bonferroni | 略弱 | 层级检验 |
| BHY (2007) | 适中 | False Discovery Rate 控制 |
| DSR (2014) | 适中 | 交易策略选择 |

DSR 的优势在于直接校正预期最大值，而非调整 p-value 阈值，更直观。

## 效用分析

在 10 次 GA 运行中（每次 450 策略, 365 天数据）:

| 指标 | 数值 |
|------|------|
| E[max(SR)] | 0.52 |
| 冠军平均 SR | 1.47 |
| 冠军平均 DSR | 0.95 |
| DSR > 0 的比例 | 100% |
| 样本外 Sharpe 平均衰减 | 24% |

DSR > 0 确认了冠军策略确实包含信号（非纯噪声），但 24% 的样本外衰减显示仍有过拟合。

## 改进策略

### 1. Probabilistic Sharpe Ratio (PSR)
PSR 将 DSR 扩展为概率陈述：
```
PSR = P(SR_true > 0 | SR_observed, T, N, skewness, kurtosis)
```
考虑了收益分布的非正态性（偏度、峰度），比 DSR 更精确。

### 2. Combinatorial Purged Cross-Validation (CPCV)
不依赖单一 DSR 检验，而是在多个训练/验证分组上交叉验证，统计策略在未见数据上的平均表现。

### 3. Deflated Information Coefficient
将 DSR 框架扩展到信息系数 (IC = corr(prediction, outcome))，适用于 ML 预测模型的评估。
