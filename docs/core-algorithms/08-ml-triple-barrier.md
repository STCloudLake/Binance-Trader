# Triple Barrier 标签法

> **状态（Phase P2 更新）**：本文档中的实现说明已与 `core/ml/features.py` +
> `core/ml/labels.py` 对齐；原「效用分析」表**无仓库来源**，已按审计要求标记为
> 示意/非实测（见 §效用分析）。ML 生产默认 `ml.enabled: false`，只有通过
> OOS 门槛的模型才允许上线。
>
> **状态（Phase P3 更新）**：barrier 宽度**默认值未变**（仍是
> `atr_multiple × ATR / close`，clamp 到 `[0.004, 0.06]`，见 §1）。P3 只是额外
> 增加了一条**可选**路径：`barrier_widths(df, vol_pct=..., vol_multiple=...)` /
> `create_triple_barrier_label_vol(..., vol_pct=...)` 可以用
> `core/ml/volatility.forecast_vol` 的条件波动率预报替代 ATR 代理。
> `vol_pct=None`（默认，也是所有现有调用）走原路径，逐位相同，有回归测试
> （`tests/test_volatility_targeting.py`）。配置项是 `risk.vol_targeting`
> 下的 `barrier_vol_multiple` / `barrier_min_pct` / `barrier_max_pct`，整块默认
> `enabled: false`。详见 [10-volatility-targeting.md](10-volatility-targeting.md)。
> 目标波动率实测（BTCUSDT 1h，8 846 根）：目标 0.45 %/bar（全样本 200 根块
> realized vol 中位数 0.4064），当前 EWMA 预报 0.5242 %/bar（≈ 年化 49.06 %）。

## 算法原理

Triple Barrier Method 由 Prado (2018) 提出，克服了传统固定时间标签的缺陷。

### 传统标签的问题

二元分类（T+1 涨/跌）有三个缺陷：
1. **无视路径**: 先涨 5% 再跌 3% → 标记为"涨 2%"，但多空都被止损
2. **无视波动率**: 低波动期的 1% 涨幅与高波动期的 1% 涨幅不可比
3. **时间框架任意**: 为什么是 T+1 而不是 T+2？
4. **丢弃"无变动"**: 旧实现 `create_binary_label(forward_periods=4,
   threshold=0.005)` 把 ±0.5% 以内的 bar 直接设为 NaN 丢弃
   （BTCUSDT 1h 仅保留 40.5%，1m 仅 15.3%），但决策在**每一根** bar 上进行。

### Triple Barrier 设计

三个屏障定义在每笔交易上：

```
Upper Barrier (止盈):  entry × (1 + upper_pct)     → 标签 1 (long profitable)
Lower Barrier (止损):  entry × (1 - lower_pct)     → 标签 0 (stop loss)
Time Barrier  (时限):  entry_time + forward_periods → 标签 2 (timeout/neutral)
```

**标签规则**: 三个屏障中**哪个先被触及**决定标签：
- 上轨先触及 → 标签 1（看涨）
- 下轨先触及 → 标签 0（看跌）
- 时间到，两者均未触及 → 标签 2（中性/超时）

类顺序与模型 softmax 一致：`[P(down), P(up), P(timeout)]` = `[0, 1, 2]`。

### 路径感知

Triple Barrier 使用**整个价格路径**（从入场到第一个屏障触及），而非仅使用两端点。这捕捉了日内波动和止损事件。

## 本项目实现

**文件**: `core/ml/labels.py`（标签实现）与 `core/ml/features.py`
（`create_triple_barrier_label` 兼容包装，默认 `vol_scaled=True`）。

### 1. 波动率缩放屏障（Phase P2）

固定 `24 × 2%` 的屏障宽度约为 BTC 1h 中位 ATR% 的 7 倍，导致**44.9% 的标签是
超时类**——一个类别而不是信号。现在宽度按 ATR 缩放：

```
width_i = clip(atr_multiple × ATR_14(i) / close_i, min_pct, max_pct)
upper_i = entry_i × (1 + width_i)
lower_i = entry_i × (1 - width_i)
```

时间屏障 `forward_periods` 必须等于策略的**真实最大持仓 bar 数**
（`ml.max_hold_bars`），而不是任意的 24。

```python
from core.ml.features import create_triple_barrier_label
y = create_triple_barrier_label(
    df, forward_periods=24,          # = 策略最大持仓
    timeout_label=2.0,               # 2 = 超时类
    vol_scaled=True,                 # 默认；ATR 缩放
    atr_period=14, atr_multiple=1.5, min_pct=0.004, max_pct=0.06)
```

`vol_scaled=False` 保留旧的固定宽度行为（仅供复现审计数字）。

### 2. 三分类目标：保留"无变动"（Phase P2）

`core.ml.labels.create_three_class_label(df, forward_periods=4, threshold=0.005)`
返回 `1 = up / 0 = down / 2 = flat`。**`flat` 是独立类别**，决策路径对它
**弃权**（`decision_from_probs` 返回 0），因此"只学 40% 样本却每根 bar 决策"
的问题被消除，覆盖率可被报告。

可选成本感知阈值：`cost_pct` + `cost_multiple=k` 时有效阈值变为
`max(threshold, k × 往返成本%)`（`config.yaml: ml.label_cost_multiple`）。

### 3. 校准与门槛（Phase P2）

- `core/ml/calibration.py`：isotonic / Platt 概率校准 + 可靠性曲线
  （`monotone` / `spearman` / `ECE`），随模型持久化。
- `core/ml/evaluation.py`：purged K-fold（`embargo = 标签时长`）+ 样本唯一性
  权重，只报样本外指标（base rate / 多数类 / ACC / AUC / Brier / log loss）。
- 成本口径与 `app.config.sim_cost_quote` **同源**（VIP0 双边 taker 0.10% + 半价差 +
  滑点，往返 BTC **0.2500%** / ETH **0.2600%**，实测 `cost_pct_for` 与
  `sim_cost_quote` 逐边相等）。审计（P2 #2）发现修复前门槛用的是
  `backtest_taker_fee_pct` 0.04%，即比真实成交便宜约 2 倍的 0.13%/0.14%，因此 ETH 的
  “+0.058%” 在真实成本下是 **−0.062%**。
- **显著性下限**（审计 P2 #3）：外围净期望必须来自 ≥100 笔交易且 `t > 2`（或
  `PSR ≥ 0.95`）；被审计的“最佳”候选只有 27 笔 / t=1.28（BTC）与 77 笔 / t=0.30
  （ETH），因此**当前所有候选都被拒绝**，`ml.enabled` 必须保持 `false`。
- 融合分数：`clip((p_up / base_rate − 1) / scale, −1, +1)`，其中
  `scale = max(1/base_rate − 1, 1/(1−base_rate) − 1)`（以模型自身基础率为中心），
  修复符号反转。注意它**不是** `2·p_up/base_rate − 1`：两者在基础率 0.2/0.8 时相差
  最多 1.0，且真实公式**不对称**（`base_rate = 0.2` 时 `p = 1.0` 得 `+1.0`，而
  `p = 0.0` 只得 `−0.25`）。

## 相关研究

1. **Prado (2018)**: "Advances in Financial ML" — Chapter 3, Triple Barrier Labeling
2. **Dixon et al. (2020)**: "Machine Learning in Finance" — Section 4.2, Meta-Labeling
3. **Sezer et al. (2020)**: 综述了 2010-2020 年 87 篇金融 ML 论文，Triple Barrier 是最推荐的标签方法

## 效用分析

> ⚠️ **原表为示意，非本项目实测（已按审计要求标注）**
>
> 下表**在仓库中没有任何数据来源**，与实测结果矛盾，仅作为文献/示意保留：
> 它声称 Triple Barrier 达到 58.7% 准确率与 Sharpe 1.12，但本项目实测
> （`scripts/ml_credibility_measure.py`，真实缓存数据 BTCUSDT/ETHUSDT 1h，
> 8 845 根）：固定 2%/24h 屏障的超时类占 **44.9%**，三分类新目标在
> purged K-fold 下的 **OOS AUC 仅 0.51–0.53**、外围净期望为负，低于 `0.55` 门槛，
> 因此 **模型保持禁用**。请勿引用下表的数字作为证据。

| 标签方法 | ML Accuracy | Sharpe (策略) | 训练样本数 | 来源 |
|----------|-------------|---------------|------------|------|
| Binary (T+1) | 52.3% | 0.68 | 365 | 示意/非实测 |
| Binary (T+4) | 54.1% | 0.84 | 361 | 示意/非实测 |
| Triple Barrier | 58.7% | 1.12 | 342 | 示意/非实测 |

### 实测对照（Phase P2 证据，`scripts/ml_credibility_measure.py`）

真实缓存数据（`data/market/*/1h.parquet`，8 845 根/符号），purged K-fold（5 折，
`embargo = 4`）+ 样本唯一性权重，往返成本 BTC **0.2500%** / ETH **0.2600%**
（`sim.cost_model` VIP0 同源）：

| 符号 | 旧 ACC / 多数类 | 旧 AUC | 旧净期望@0.5 | 新 AUC（未校准，门槛用） | 外围净期望（嵌套，门槛用） | 笔数 / t / 95% CI | 门槛 |
|---|---|---|---|---|---|---|---|
| BTCUSDT 1h | 0.6702 / **0.8128** | 0.5560 | −0.1316% | **0.5207** | **−0.2494%** | 1365 / −4.46 / [−0.359%, −0.140%] | **FAIL**（AUC、期望、显著性全不达标） |
| ETHUSDT 1h | 0.6408 / **0.7760** | 0.5412 | −0.1758% | **0.5342** | **−0.1822%** | 585 / −2.28 / [−0.339%, −0.026%] | **FAIL**（AUC、期望、显著性全不达标） |

- **选择偏差修复（审计 P2 #1）**：修复前把校准器与阈值都拟合在**被报告的那批 OOS
  行**上（所谓 per-fold 校准流是死代码：`calibrator.n_fit == n_oos == 600`，而
  `sum(n_cal) = 477`；in-fit ECE 0.0000 vs 真留出 0.0784），半样本外实验的 OOS 净
  期望是 **−0.075%（BTC）/ −0.342%（ETH）**，且 ETH 后半段**没有任何** ≥20 笔的
  正阈值。现在：校准器只用该折自己的 calibration stream，阈值在该折内部选出后再套
  到该折的测试行上，门槛读的是这个**外围**数字。修复后 `calibrator_n_fit` 之和
  2862 == `n_cal` 之和 2862（BTC）/ 3867 == 3867（ETH），死代码已消除。
- 每个折的**流内**期望为正（例如 BTC fold 0 +0.3113%），套到测试行上却是 −0.2984%
  ——这正是选择偏差的量级，也是门槛必须用外围数字的原因。
- 旧流水线准确率**低于多数类 6.7–14.3pp**，且在 0.5 阈值下的净期望为**负**。
- 旧概率可靠性曲线（实测，非示意）：BTC ECE 0.2048 / Spearman +0.54，
  ETH ECE 0.1824 / +0.72，`monotone=False`。**部署 pkl 在 `p ≥ 0.9` 分箱的实际上涨率
  仅 0.327 / 0.473**（`n_features_in_=40`；中性带 0.38–0.62 占 25.7% / 24.1%，
  而 `engine.py` 把该带压成 0.5 并按"看跌"计分），与审计的"反向校准"一致。
- 新流水线校准（拟合/评估半样本外拆分）：ECE 0.1377→0.0784（BTC）、
  0.1179→0.0752（ETH），Spearman +0.54→+0.80、+0.72→+0.96。真实数据 AUC≈0.52
  时校准**无法**制造单调曲线（无信号可校准）；单调性由
  `test_miscalibrated_probabilities_become_monotone_on_held_out_data`
  在带信号模型（AUC 0.815）上证明（ECE 0.0918→0.0159，Spearman 1.00，单调）。
- 结论：两个符号都未通过 `OOS AUC > 0.55` 且外围净期望为负、显著性为负，
  `ml.enabled` 保持 `false`。

### 标签尾部与特征成本（审计 P2 #6 / #7）

- **尾部不再被强制成超时类**：`create_triple_barrier_label_vol` 只给具有**完整**前向
  窗口的行打标签，最后 `forward_periods` 行在两种模式下都是 `NA`。修复前扫描到
  `n-1` 并用 `fillna(timeout_label)` 把截断窗口的行全部塞进超时类——在 2 000 根合成
  数据上这一项就移动了 24 根 bar；在真实 8 845 根 BTC 1h 数据上它摧毁了 16 个真实
  屏障触及（修复前标签数 8 840 / 修复后 8 844 中最后 4 行为 `NA`）。
  `timeout_label=None` 时整类超时为 `NA`（旧文档写反了）。
- **滚动 Hurst 成本**：`compute_features` 现在用**有界** R/S Hurst
  （`HURST_LOOKBACK=60`、4 个 lag、每 4 根刷新并 ffill），不再依赖
  `REQUIRED_INDICATORS["hurst"]`（10 lag × 100 窗口 × 每根 bar，实测 8 845 根
  16.4 s）。实测（本项目机器，8 845 根 BTC 1h）：
  `compute_all(REQUIRED_INDICATORS)` **16.402 s → 0.122 s**，
  `compute_features` **0.219 s → 0.960 s**，合计 **16.62 s → 1.08 s（15.4×）**；
  其中 Hurst 一项 12.89 s → 0.96 s。`predictor._on_kline` 另加一层按
  `(symbol, interval, 最后一根 bar)` 的缓存，同一根 bar 的重复 tick 不再重算。
  回归断言 `test_feature_pipeline_cost_is_bounded` 绑定 3.0 s/8 845 根（当前
  1.08 s，约 2.8× 余量）。

## 改进策略

### 1. 动态 Barrier 宽度 ✅ 已实现

见上文 §1：`atr_multiple × ATR / close`，clamp 到 `[min_pct, max_pct]`。
高波动期放宽上/下轨，低波动期收窄。

### 2. Meta-Labeling（P4）

Triple Barrier 标签作为**二级模型**的训练目标。一级模型（策略引擎）给出入场信号，
二级模型（meta-labeler）预测该信号是否会触及上轨。只执行 meta-model 确认的信号。
（本阶段只完成评估/校准/门槛基础设施，meta-labeling 属 P4。）

### 3. Multi-Horizon Barriers（未实现）

同时使用多个时间屏障（如 6h, 12h, 24h），model 学习不同时间尺度的模式。
多任务学习架构（shared backbone + 3 task-specific heads）。

## 诊断口径（Phase P2）

旧 `engine.py` 的 `ml_accuracy_pct` 把中性带（0.38–0.62，被压成 0.5）
当作**看跌预测**参与计分，因此有看涨偏差。正确口径实现在
`core.ml.credibility.ml_accuracy_neutral_abstention`：中性 = **弃权**，
不计入准确率，并单独报告覆盖率。`engine.py` 属于另一个 agent 的写范围，
需由 Lead 安排替换（文件中已留 TODO 说明）。
