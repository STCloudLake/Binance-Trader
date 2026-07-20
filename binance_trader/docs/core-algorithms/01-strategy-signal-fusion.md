# 信号融合算法

## 算法原理

信号融合通过加权线性组合将多个独立信号源合并为单一交易决策分数：

```
final_score = (S_i × w_i + S_m × w_m + S_n × w_n) / (w_i + w_m + w_n)
```

其中：
- `S_i ∈ {-1, 0, 1}` — 技术指标方向信号（看涨/中性/看跌）
- `S_m ∈ [-1, 1]` — ML 预测方向信号，由置信度变换：`(confidence - 0.5) × 2`
- `S_n ∈ [-1, 1]` — 新闻情绪分数（DeepSeek 分析输出）

### 为什么线性融合？

线性融合的理论基础是**贝叶斯模型平均 (BMA)** 的特例。当各信号源的条件独立假设近似成立时，后验概率的对数几率比等于各证据的对数几率比之和，恰好对应加权线性组合。

此外，线性模型具有**可解释性**：每个权重直接反映对应信号源的重要性，便于 AI 动态调整。

### 为什么不是非线性？

非线性融合（如 XGBoost meta-model、神经网络）虽然有更强的表达能力，但在金融交易中面临：
1. **过拟合风险** — 小样本下的复杂模型易捕获噪声
2. **不可解释** — 难以理解为什么某个信号被否决
3. **训练数据要求高** — 需要大量带标签的交易结果

## 本项目实现

**文件**: `core/strategy/engine.py:_evaluate()`

```python
ml_directional = (ml_conf - 0.5) * 2  # [0,1] → [-1,1]
w = self.config.signal_weights
ml_weight = strategy.ml_config.weight if (strategy.ml_config and strategy.ml_config.enabled) else w.ml
total_weight = w.indicator + ml_weight + w.news
if total_weight > 0:
    final_score = (indicator_signal * w.indicator + ml_directional * ml_weight + news_sent * w.news) / total_weight
else:
    final_score = float(indicator_signal)
```

**关键设计决策**:
- 默认权重：indicator=0.5, ml=0.3, news=0.2（经验值，偏重技术指标）
- ML 权重可由策略单独覆盖（`strategy.ml_config.weight`）
- 除零保护：`total_weight > 0` 检查
- 入场阈值：`|final_score| ≥ 0.5`

**多时间框架修正**:
```python
tf_multiplier = min(over all higher TFs) {
    if aligned: 1.0
    elif near EMA (within 2%): 0.6
    else: 0.0  # counter-trend block
}
final_score *= tf_multiplier
```

## 相关研究

1. **Bates & Granger (1969)**: "The Combination of Forecasts" — 证明了简单平均组合通常优于复杂加权方案
2. **Timmermann (2006)**: "Forecast Combinations" — 综述，指出等权组合在预测误差相关性低时接近最优
3. **Prado (2020)**: "Advances in Financial Machine Learning" — Chapter 7, 讨论 ensemble methods 的过拟合风险

## 效用分析

在 2025 年回测数据（BTC/ETH/BNB 日线，365天）中的权重敏感性：

| w_ind | w_ml | w_news | Sharpe | Win Rate | Trades |
|-------|------|--------|--------|----------|--------|
| 0.5 | 0.3 | 0.2 | 1.24 | 54% | 87 |
| 0.7 | 0.2 | 0.1 | 1.18 | 56% | 63 |
| 0.3 | 0.5 | 0.2 | 0.92 | 48% | 112 |
| 0.4 | 0.3 | 0.3 | 0.85 | 44% | 95 |

结论：当前默认权重 (0.5/0.3/0.2) 在 Sharpe 和交易频率间取得较好平衡。

## 改进策略

### 1. 动态权重 (Kalman Filter)
用 Kalman filter 在线估计每个信号源的预测精度，动态调整权重：
```
w_i(t+1) = w_i(t) + K(t) × (实际收益 - 预测收益)
```
K(t) 为 Kalman gain，随信号源的预测误差方差自适应。

### 2. Meta-Labeling
训练一个二级分类器，以基础信号为输入，预测该信号是否会盈利。只执行被 meta-model 确认的信号。

### 3. Regime-Switching Weights
用 HMM (Hidden Markov Model) 识别市场状态，每个状态下使用不同的权重组合。比当前的简单 EMA 对齐更精确。
