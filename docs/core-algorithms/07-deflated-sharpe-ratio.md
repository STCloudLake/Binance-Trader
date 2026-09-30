# Deflated Sharpe Ratio (DSR)

> **P1 更新（GA 可信性）**：本文档原先记录的是**旧实现**：用年化 Sharpe 直接减一个
> **每期**门槛、`observation_periods` 写死 365，并声称代码里有 `if dsr <= 0: logger.warning(...)`
> 的门控（当时并不存在）。现已按 `core/ga/fitness.py::deflated_sharpe_ratio` 重写。
> 文末「效用分析」的表格为**示意值（非实测）**。

## 算法原理

DSR 由 Bailey & López de Prado (2014) 提出，用于校正**多重测试偏差**（multiple testing bias）。

### 问题

GA 会评估 N 个策略（N = population × generations **+ 历史试验数**）。即使所有策略都无效
（真实 Sharpe = 0），样本内 Sharpe 最高的那个仍会呈现正 Sharpe：

```
E[max(SR)] ≈ sqrt(Var[SR]) × sqrt(2 ln N)
Var[SR]    ≈ 1 / T                     # 每期 Sharpe 的方差（i.i.d. 收益）
⇒ E[max(SR)] ≈ sqrt(1/T) × sqrt(2 ln N)
```

其中 **T = 观察期数（收益样本数）**，**N = 试验次数**。二者必须与 Sharpe **同单位**。

## 单位：本项目最容易错的地方

`observed_sharpe` 默认是**年化** Sharpe（`metrics["sharpe_ratio"]` / 净值序列年化），
而 `E[max]` 是**每期**量。因此实现先把年化 Sharpe 折成每期：

```
SR_per_period = SR_annualized / sqrt(365)
E[max]        = sqrt(1/T) × sqrt(2 ln N)      # T = 真实期数
DSR           = SR_per_period - E[max]        # 每期口径的"去偏 Sharpe"
```

**旧实现的错误**：直接拿年化 Sharpe 去减每期门槛，且 `T` 写死 365。这样门槛是一个与数据长度
无关的常数（N=1200 时 ≈0.191），任何年化 Sharpe > 0.2 都会被判"显著"。实测差异：

| 算例 | SR | N | T | 旧实现 | 现在的实现 |
|------|----|---|---|--------|-----------|
| 审计算例 | 1.2（年化） | 1200 | 1200 | **+1.0029「显著」** | **−0.0459（不显著）** |
| 同 Sharpe，T=365 | 1.2（年化） | 1200 | 365 | +1.0029 | −0.1343（不显著） |
| 每期 Sharpe | 0.30（每期） | 100 | 250 | — | +0.1081（显著） |

> 审计报告给出的"正确值 −0.6135"采用了另一种 Sharpe 口径假设；无论哪种口径，
> 修正后的结论都是**不显著**，而旧实现给出的是**假阳性**。手算校验见
> `tests/test_ga_credibility.py::test_dsr_hand_check_units`。

### p 值与 PSR 修正

```
var_term = 1 - skew·SR + ((kurtosis - 1)/4)·SR²      # Bailey–López de Prado 非正态修正
z        = DSR × sqrt(T - 1) / sqrt(var_term)
p_value  = 1 - Φ(z)
significant ⟺ DSR > 0 ∧ p < 0.05
```

`skew` / `kurtosis` 由该基因**自己的**日收益序列给出（不足 3/4 个样本时取 0 / 3）。

### 期数不足

`T < 20`（`MIN_OBSERVATIONS`）时不估计 Sharpe/DSR：DSR 记 0、alpha 项记 0，
基因通过 `flag = insufficient_trades`（<30 笔）显式标注。

## 本项目实现

**文件**：`core/ga/fitness.py`

```python
deflated_sharpe_ratio(
    observed_sharpe,            # 年化 Sharpe（默认）
    n_trials,                   # population×generations + 历史试验数
    observation_periods=T,      # 真实期数（不是 365）
    variance_sharpe=1.0,
    sharpe_is_annualized=True,
    skew=..., kurtosis=...,
) -> {"dsr", "expected_max_random", "p_value", "significant",
      "n_trials", "observation_periods", "sharpe_per_period"}
```

**试验次数**：`core/ga/trial_counter.py` 维护 `data/ga_trials.json`。
每代评估后按 `population_size` 累加，`total_trials = 历史 + 当前`，
所以第 24 个 walk-forward 任务的冠军不会被当成"只试过 450 个策略"。

**门控**：`core/ga/evolver.py::_publication_decision` —— `DSR <= 0` 时冠军写成
`enabled: false`，原因进 `provenance.rejection_reasons`；面板会显示 DSR / p / T / N。

## 相关研究

1. **Bailey & López de Prado (2014)**: "The Deflated Sharpe Ratio" — 原始论文
2. **Harvey, Liu & Zhu (2016)**: "…and the Cross-Section of Expected Returns" — t-stat > 3.0
3. **Prado (2020)**: "Advances in Financial ML" — Chapter 14, Backtesting Statistics

### DSR vs 其他多重比较校正

| 方法 | 保守性 | 适用场景 |
|------|--------|----------|
| Bonferroni | 过于保守 | 独立假设下 |
| Holm-Bonferroni | 略弱 | 层级检验 |
| BHY (2007) | 适中 | False Discovery Rate 控制 |
| DSR (2014) | 适中 | 交易策略选择 |

## 文档一致性

- 本文档与 `core/ga/fitness.py`、`core/ga/trial_counter.py`、`core/ga/evolver.py` 对齐。
- 下面的「效用分析」表格是**示意值（非实测）**：原表格声称 10 次 GA 运行、450 策略、
  E[max]=0.52、冠军平均 SR 1.47、DSR>0 比例 100%，没有任何可复现的产物支撑，
  且与上面的单位修正结论矛盾，不应作为参照。

## 效用分析（示意 / 非实测）

| 指标 | 数值 |
|------|------|
| E[max(SR)] | 0.52（示意） |
| 冠军平均 SR | 1.47（示意） |
| 冠军平均 DSR | 0.95（示意） |
| DSR > 0 的比例 | 100%（示意） |

## 改进策略

### 1. Probabilistic Sharpe Ratio (PSR)
本实现已包含 PSR 的偏度/峰度修正项；进一步可改用 bootstrap 的 Sharpe 分布。

### 2. Combinatorial Purged Cross-Validation (CPCV)
不依赖单一 DSR 检验，而是在多个训练/验证分组上交叉验证。

### 3. Deflated Information Coefficient
将 DSR 框架扩展到信息系数 (IC = corr(prediction, outcome))。
