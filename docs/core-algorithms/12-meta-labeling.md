# Meta-labeling（二级模型：过滤 + 仓位缩放）

> **P4 新增（phase P4）**：`core/ml/meta.py`。全部数字均为**实测**（缓存 BTC/ETH 1h，
> 8 845 bar，成本 0.25%/往返，取自 sim 成本模型）。
> **本模块默认关闭**（`META_LABELING_ENABLED = False`），且**未通过自己的门控**——
> 因此 `MetaLabeler.enabled = False`，对任何概率都返回 `take=False`。

## 算法原理

P2 审计已经证明：让 ML **选方向**是有害的（OOS AUC 0.396–0.447，低于多数类基线
10–20pp）。López de Prado（AFML 第 3 章）的 meta-labeling 把问题换掉：

* **一级模型**（现有规则信号 / GA 冠军签名）决定 *何时* 交易、*哪个方向*；
* **二级模型**只预测一件事：*这笔一级交易会先碰到止盈屏障吗？*
* 二级输出只用于**过滤**（低于成本感知阈值就不做）与**缩放**（通过的按概率缩放仓位），
  **永远不能反向**。

## 公式与标签

```
标签：y_meta = 1  先碰止盈屏障
              = 0  先碰止损屏障
              = NA 超时（既没碰止盈也没碰止损）→ 丢弃，不塞进任何类
屏障宽度：w = clip(1.5 × ATR(14) / close, 0.4%, 6%)        # 与 core.ml.labels 同源
多头止盈 = entry·(1+w)、止损 = entry·(1−w)；空头互换
一级交易收益：r_t = side_t × (close_{t+h}/close_t − 1),  h = 24 bar
净收益：net = r − cost        （cost = credibility.cost_pct_for，与成交同源）
决策：take = p ≥ threshold；否则 0
仓位：size = 0.25 + 0.75 × clip((p − threshold)/(1 − threshold), 0, 1)
一级方向：position = sign(一级方向) × size     ← 符号永远来自一级
```

成本感知阈值搜索是**单边**的（`p ≥ t` 才做），因为二级模型不允许做空一级信号
（`meta_cost_aware_threshold`，与 `credibility.cost_aware_threshold` 的双边搜索不同）。

## 评估协议（与 P2 同一套，无选择乐观）

1. `core.ml.evaluation.purged_kfold_splits`（`embargo = label_span = 24`）+
   `sample_uniqueness_weights`；
2. 每折：训练块尾部 20% 作为**该折自己的校准流**，其余用于拟合；
3. 阈值在**折内**的校准流上选出（`meta_cost_aware_threshold`），再作用于该折的测试行；
4. 门控（`core.ml.credibility.gate_from_evaluation`）消费的是**外层**数字
   （`net_expectancy_oos` / `n_trades_oos` / `t_stat_oos` / `psr_oos`）：
   AUC > 0.55、净期望 > 0、交易数 ≥ 100、t > 2 **且** PSR ≥ 0.95（`core/ml/credibility.py`
   的门是 **AND**，不是 OR：`elif not (t_val > min_t_stat and psr_val >= min_psr)`，见
   `core/ml/credibility.py:865`；doc 12 此前的 "或" 是错的）。
   池化搜索（在同一批被汇报的行上选阈值）只作为对照报出（`selection = pooled_optimistic`）。
5. `thresholds_oos`（可部署阈值）= 各折选出阈值的**中位数**；若无折选出候选 →
   `threshold = None`（"无阈值"而不是"用池化最优"）。

## 实现位置

| 内容 | 位置 |
|---|---|
| 一级信号（共享条件核） | `core/ml/meta.py::primary_signal_from_rules` |
| 屏障标签 | `profit_barrier_labels`、`meta_label_from_barrier`（对接 `core.ml.labels`） |
| 数据集装配 | `build_meta_dataset`（特征用 **39 列契约** `core.ml.features.compute_features`） |
| 单边成本感知阈值 | `meta_cost_aware_threshold` |
| 嵌套 OOS 评估 | `evaluate_meta_oos`（`thresholds_oos` 为可部署值） |
| 硬门 | `meta_gate` → `core.ml.credibility.gate_from_evaluation` |
| 过滤 + 缩放 | `MetaLabeler.decide` / `filter_series` / `MetaDecision.apply` |
| 引擎集成缝 | `core/strategy/engine.py::StrategyEngine.wire_meta_filter`（`P4_META_FILTER_ENABLED = False`） |
| 测试 | `tests/test_meta_labeling.py`（**24** 条；`python -m pytest tests/test_meta_labeling.py --collect-only -q` 末行实测 24） |

## 实测数字（BTC/ETH 1h，5 条一级规则，共 10 次评估）

成本 0.25%/往返；1:1 屏障下的**盈亏平衡命中率 = 0.5012**（`breakeven_hit_rate`）。
标签基准率全部落在 0.469–0.511，即一级交易的止盈/止损命中**近似抛硬币**。

| 品种 | 一级规则 | 触发 | 有效标签 | 超时率 | 基准率 | 多数类 | 准确率 | **AUC** | Brier | **净期望(OOS)** | 笔数 | t | PSR | 阈值 | 门控 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTC | rsi 均值回归(35/65) | 1 967 | 1 880 | 4.4% | 0.488 | 0.512 | 0.505 | 0.470 | 0.263 | 0.000% | 0 | 0.00 | 0.000 | None | **拒绝** |
| BTC | rsi 顺势(55/45) | 6 002 | 5 707 | 4.9% | 0.509 | 0.509 | 0.501 | 0.489 | 0.261 | 0.000% | 0 | 0.00 | 0.000 | None | **拒绝** |
| BTC | 布林突破 | 1 127 | 1 094 | 2.9% | 0.505 | 0.505 | 0.488 | 0.481 | 0.274 | −0.679% | 186 | −2.10 | 0.018 | 0.51 | **拒绝** |
| BTC | MACD 交叉 | 8 813 | 8 289 | 5.9% | 0.492 | 0.508 | 0.493 | 0.477 | 0.254 | −0.297% | 1 658 | −4.00 | 0.000 | 0.50 | **拒绝** |
| BTC | ADX+EMA | 2 510 | 2 367 | 5.7% | 0.499 | 0.502 | 0.502 | 0.504 | 0.269 | −0.084% | 1 144 | −1.41 | 0.080 | 0.51 | **拒绝** |
| ETH | rsi 均值回归(35/65) | 2 176 | 2 094 | 3.8% | 0.469 | 0.531 | **0.446** | 0.501 | 0.272 | −1.246% | 362 | −5.91 | 0.000 | 0.60 | **拒绝** |
| ETH | rsi 顺势(55/45) | 5 958 | 5 664 | 4.9% | 0.511 | 0.511 | 0.497 | 0.517 | 0.254 | −0.777% | 1 046 | −5.68 | 0.000 | 0.50 | **拒绝** |
| ETH | 布林突破 | 1 147 | 1 104 | 3.7% | 0.504 | 0.504 | 0.522 | 0.512 | 0.254 | −1.172% | 200 | −4.37 | 0.000 | 0.51 | **拒绝** |
| ETH | MACD 交叉 | 8 813 | 8 347 | 5.3% | 0.497 | 0.504 | 0.487 | 0.496 | 0.254 | −0.726% | 1 650 | −7.38 | 0.000 | 0.51 | **拒绝** |
| ETH | ADX+EMA | 2 568 | 2 431 | 5.3% | 0.508 | 0.508 | 0.490 | 0.516 | 0.258 | −1.236% | 75 | −2.91 | 0.002 | 0.595 | **拒绝** |

**门控结论：上表 10/10 全部拒绝**（最高 AUC 0.5167 < 0.55；净期望 −1.246% … 0.000% ≤ 0；
t 最高 −1.41）。ETH 的 rsi 均值回归准确率 0.446 对多数类 0.531，**低于基线 8.5pp**——
与 P2 审计测到的失效模式完全一致。**注意：这是 2025-06 → 2026-09 单一样本上的历史快照**；
2026-10-01 换成 1.5 年无缺口缓存后复测，第一次出现通过硬门的候选，见下一节。

**结构发现**：对"几乎每根 bar 都触发"的一级规则（MACD 交叉 8 813/8 845），多折的校准流
选不出任何正期望阈值（`n_cal_trades = 0`，该折 OOS 笔数 0）——这是"无阈值"这条诚实
路径在起作用，而不是被优化掩盖。同理 `thresholds_oos.n_folds_with_candidates` 在
BTC rsi 两条规则上为 **0**，`threshold` 因此是 `None`。

## 2026-10-01 复测（1.5 年无缺口缓存，尾部 3 000 bar 窗口）

本节是**当前缓存的实测记录，不是断言**：`tests/test_meta_labeling.py::test_real_primary_rules_are_refused_by_the_meta_gate`
现在断言门控的**语义**（`allowed` ⇔ 每一项判据都满足，由响应自身的数值与阈值推出，
`tests/test_measured_threshold_policy.py` 强制该口径），所以下面这个"通过"是被**记录**
下来的结果，而不是被钉死的期望——缓存再变，测试只改记录的口径，不改结论的写法。

| 品种 | 一级规则 | 窗口 | 有效标签 | AUC | 净期望(OOS) | 笔数 | t | PSR | 阈值 | 门控 |
|---|---|---|---|---|---|---|---|---|---|---|
| BTC | 布林突破 | 2026-05-29 → 2026-10-01 | 377 | **0.5729** | **+0.3175%** | **178** | **2.061** | **0.990** | 0.50 | **通过** |
| BTC | rsi 均值回归(35/65) | 同上 | 605 | 0.5519 | −0.0685% | 92 | −0.58 | 0.281 | 0.505 | 拒绝（净期望/笔数/显著性） |
| ETH | 布林突破 | 同上 | 342 | 0.5100 | +0.1420% | 155 | 0.52 | 0.704 | 0.62 | 拒绝（AUC/显著性） |
| ETH | rsi 均值回归(35/65) | 同上 | 586 | 0.4745 | −1.9540% | 97 | −8.32 | 0.000 | 0.50 | 拒绝（全部） |

成本 0.25%/往返（= `core.ml.meta.default_meta_cost_pct()` 的实测值，与仿真成交成本同源）。
BTC 布林突破是本项目**第一个**通过硬门的候选：AUC 0.5729 > 0.55、净期望 +0.3175% > 0、
178 笔 ≥ 100、t 2.061 > 2.0、PSR 0.990 ≥ 0.95；外层 95% CI **[+0.0156%, +0.6194%]**（不含 0）；
5 折校准流**全部**选出阈值（中位数 0.50，范围 0.50–0.50）；两个独立进程实测逐位相同
（`random_state=SEED`，非随机）。

**鲁棒性（同一复算口径）**：把两个品种的全部 **22 个重叠 3 000-bar 窗口**按测试同样的方式
（窗口内重算指标）各评一次，88 个候选中 **只有 2 个通过**，都是布林突破，t 分别为
2.061 与 2.141（ETH 2025-08-04 → 2025-12-06：AUC 0.5583、净期望 +0.517%、150 笔、
PSR 0.983），都只比 2.0 的 t 门槛高一点点。另两种同样合理的口径都**拒绝**同一个窗口：
(a) 用整段 13 160 bar 缓存热身指标后再切片，AUC 0.5513、净期望 +0.318%（与通过时相同）、
151 笔、**t 1.862 ≤ 2.0**；(b) 直接用整段缓存评估，AUC 0.5161、净期望 −0.2775%、
498 笔、t −2.66。即这是**边界个案**（t 2.061 对门槛 2.0），不是稳健边缘；88 个候选里
2 个边际通过，与"在 t=2.0 门槛上做多重比较"应产生的噪声一致。

**结论与动作**：P4 接缝的**验收状态本身没有改变**——`META_LABELING_ENABLED = False`、
`P4_META_FILTER_ENABLED = False` 全部保持默认关闭，没有启用任何东西；本节的用途是
记录"能力现在能通过自己的门"这一事实及其边界性。要谈部署，需要的是样本外/多窗口/多品种
的稳健通过，而不是单窗口的 t=2.06。

## 一个顺带修掉的 P2 缺陷（`credibility.py` 不在本阶段写权限内）

`core/ml/credibility.py::evaluate_model_oos` 的 `model_factory=None` 默认路径把
**工厂构造器**当作工厂本身绑定：

```python
model_factory = default_binary_factory      # 这是 (feature_names, n_estimators) -> 工厂
model = model_factory(Xf, yf, w_fit)        # TypeError: takes from 0 to 2 positional arguments
```

实测：第一次真实 meta 评估 5 折**全部**因此失败（`no fold produced a model`）。
本阶段的 `core/ml/meta.py::resolve_model_factory` 两种形态都接受，并用测试把该缺陷钉住
（`tests/test_meta_labeling.py::test_resolve_model_factory_accepts_both_forms`）。
**建议 Lead 在 P5 修 `credibility.py`：`model_factory = default_binary_factory()`。**

## 局限（明写）

1. 标签是"先碰哪个屏障"的分类，**不含路径**：止盈前先浮亏 5 倍风险也算 1。
2. 一级样本远小于 bar 数：二级模型的统计功效受一级触发次数限制；门控要求 100 笔外层
   交易，触发 40 次的规则**在结构上不可能**通过——这是设计，不是 bug。
3. 屏障宽度只按 ATR 缩放，未按一级信号的方向强度调整。
4. 阈值网格 0.30–0.95（步长 0.01）单边搜索；更细的网格不改变结论（净期望全为负）。
5. 本页数字来自单一样本（2025-06 → 2026-09）；换样本必须重测，`evaluate_meta_oos`
   每次都会重跑，不做缓存。缓存换成 1.5 年无缺口数据后的复测见 §"2026-10-01 复测"：
   88 个窗口中 2 个边际通过，属边界个案而非稳健边缘。
6. `META_LABELING_ENABLED = False`：实盘路径逐字节未变。

## 开关（全部默认 off）

| 常量 | 默认 | 作用 |
|---|---|---|
| `core.ml.meta.META_LABELING_ENABLED` | `False` | 主开关（当前无任何实盘代码读它） |
| `core.strategy.engine.P4_META_FILTER_ENABLED` | `False` | 是否让已注册的 `MetaLabeler` 过滤/缩放实盘入场 |
| `META_MIN_TRADES` | 100 | 外层交易数下限（与 P2 硬门一致） |
| `META_MIN_PROBABILITY` | 0.50 | 过滤概率绝对下限 |
| `META_SIZE_FLOOR` / `META_MAX_SIZE_MULTIPLIER` | 0.25 / 1.0 | 仓位缩放区间 |
| `META_ATR_PERIOD` / `META_ATR_MULTIPLE` | 14 / 1.5 | 屏障几何 |
| `META_BARRIER_MIN_PCT` / `META_BARRIER_MAX_PCT` | 0.004 / 0.06 | 屏障宽度截断 |
| `META_THRESHOLD_GRID` | 0.30–0.95 步长 0.01 | 单边阈值网格 |
