# 波动率目标化与动态 Barrier（Phase P3）

> **状态**：已实现 · **前置**：[P1 可信性](../../overhaul/ALGO_UPGRADE_PLAN.md#p1--ga-可信性结构改动)
> 与 P2（ML 可信性）已落地 · **默认关闭**：`risk.vol_targeting.enabled: false`，
> 开关打开前仓位/止损/barrier 与改造前**逐位相同**（有回归测试固定）。

## 1. 算法原理：为什么预测波动率而不是方向

P2 的实测结论是**收益方向不可预测**：真实缓存 BTC/ETH 1h 上，全部 ML 候选
在 purged K-fold 下 OOS AUC 仅 **0.5207 / 0.5342**，扣成本后的外围净期望为
**负**（−0.2494 % / −0.1822 %），因此 `ml.enabled` 必须保持 `false`
（见 [08-ml-triple-barrier.md](08-ml-triple-barrier.md) §实测对照）。

但**条件方差是可预测的**，这是时间序列里最稳固的经验规律之一：

* 收益率平方存在自相关（波动率聚集，volatility clustering）；
* 大波动后倾向于继续大波动，小波动后倾向于继续小波动；
* 该现象由 ARCH（Engle 1982）/ GARCH（Bollerslev 1986）族刻画，
  见 Tsay, *Analysis of Financial Time Series*, 第 3 章。

因此预测的最高价值用途是**风险**而非方向：

1. **波动率目标化仓位**（`core/risk/position_sizer.py`）——
   让每笔仓位的预期波动贡献相等；
2. **动态止损 / 移动止损宽度**（`core/risk/position_guard.py`、
   `core/executor/executor.py`）——高波动期放宽（少被噪声扫损），
   低波动期收紧（少回吐利润）；
3. **动态 Triple Barrier 宽度**（`core/ml/labels.py`）；
4. **风险阈值 / 体制判断**（`vol_percentile`）。

## 2. 公式与单位（读数字前必看）

所有估计量统一返回**每根 bar 的价格比例**，不是百分比：

| 量 | 公式 | 单位 |
|---|---|---|
| 对数收益 | `r_t = ln(P_t / P_{t-1})` | 无量纲 |
| 收盘价实现波动（close-to-close） | `sqrt( Σ(r-mean)² / (n-1) )` | 比例/bar |
| Parkinson (1980) | `sqrt( mean(ln(H/L)²) / (4 ln2) )` | 比例/bar |
| Garman-Klass (1980) | `sqrt( mean(0.5 ln(H/L)² − (2ln2−1) ln(C/O)²) )` | 比例/bar |
| EWMA (RiskMetrics) | `σ²_{t+1} = λσ²_t + (1−λ) r_t²`，`λ = 0.94` | 比例²/bar |
| GARCH(1,1) 一步 | `σ²_{t+1} = ω + α r_t² + β σ²_t` | 比例²/bar |
| 年化 | `σ_annual = σ_per_bar × sqrt(periods_per_year)` | 比例/年 |

* `annualize` / `deannualize` 是唯一的单位换算入口；**年化假设收益 i.i.d.**，
  是汇报口径，不要用它给单根 bar 定价。
* `to_pct` 是唯一出现百分号的地方（配置项是百分比类型）。
* `periods_per_year`：`1h → 8760`、`1d → 365`（`PERIODS_PER_YEAR`）。
* 仓位缩放：`scale = clip(target_vol_pct / forecast_vol_pct, min_scale, max_scale)`，
  **无量纲**，`1.0` 即原有的固定比例仓位。
* 止损宽度：`stop_pct = clip(stop_vol_multiple × forecast_vol_pct, stop_min_pct, stop_max_pct)`；
  barrier 宽度是**分数**（`0.004` = 0.4 %），由 `barrier_vol_multiple × vol%/100` 得到。

### 数据质量：缓存的接缝（剪裁是**防御性**措施）

> **数据快照**：`data/market/BTCUSDT/1h.parquet` 由运行中的服务持续追加。
> 2026-09-30 复核：**11 675 根**（2025-06-03 → 2026-09-30），接缝**已被数据
> 供应商修复**——`|最大对数收益| = 0.0494`，且在 `DEFAULT_WINDOW = 500` 的尾窗上
> 未剪裁 / 剪裁 == **1.000×**（命令见下）。因此 §3.2 的表是"接缝仍存在时"的历史
> 记录，剪裁代码保留为**防御性**措施，而不是当前数据必需的修复。

历史缺陷（**已修复；在当前缓存上不可复现**）：8 846 根切片曾有 11 处 >1.5 h 的空洞，
最大 **1 484 小时**（2026-07-29 → 2026-09-29 被当作"一根 bar"，产生一个
`+27.63 %` 的对数收益；正常 bar 中位绝对值 0.19 %）。EWMA 的**有效记忆只有
`1/(1−λ) ≈ 16.7` 根**，不剪裁时该异常值直接成为未来 17 根的预测：当时实测
**5.3056 %/bar** 对剪裁后 **0.5242 %/bar**（**9.81×**；早期文档写的 10.5× 是拿
剪裁后的 0.52 去除未剪裁的 5.47，属单位/口径混用）。这一行**不可在当前缓存上
复现**（接缝已被供应商修复），只作为历史对照保留。

**剪裁有效性的证据 = 注入式对照（合成，且窗口相关——不是当前缓存的属性）**：
往平静序列尾部注入**一条** `+0.276` 的对数收益：

| 注入对象 | 未剪裁 %/bar | 剪裁后 %/bar | 比值 |
|---|---|---|---|
| 合成 2 000 根（σ = 0.4 %/bar，seed 20250930） | **6.7690** | **0.7553** | **8.96×** |
| 真实 500 根尾窗（覆盖 `r[-1]`） | 6.7674 | 0.4994 | 13.55× |

```python
# 复现（当前修订实测；*_pct 单位是 %/bar，即已乘 100）
import numpy as np, pandas as pd
from core.ml.volatility import ewma_vol, log_returns, to_pct
rng = np.random.default_rng(20250930)
calm = rng.normal(0.0, 0.004, 2000)
s = np.concatenate([calm[:1999], np.array([0.276])])      # 注入的接缝
to_pct(ewma_vol(s, window=0, outlier_sigma=0.0))          # 6.7690
to_pct(ewma_vol(s, window=0))                             # 0.7553 -> 8.96x
r = log_returns(pd.read_parquet("data/market/BTCUSDT/1h.parquet")["close"].values)
rr = r[-500:].copy(); rr[-1] = 0.276                      # 真实尾窗 + 注入
to_pct(ewma_vol(rr, window=0, outlier_sigma=0.0))         # 6.7674
to_pct(ewma_vol(rr, window=0))                            # 0.4994 -> 13.55x
# 当前缓存上的同一测量（无注入）：剪裁是 no-op
tail = r[-500:]
to_pct(ewma_vol(tail, window=0))                          # 0.303674
to_pct(ewma_vol(tail, window=0, outlier_sigma=0.0))       # 0.303674（相对差 2.7e-6）
```

因此所有估计量先用 `clip_outliers`（`±6 × 1.4826 × MAD`，可关）做 Winsorize；
用 MAD 而不是标准差，因为要被防的正是会污染后者的量。

**剪裁必须对同一根 bar 稳定**：早期实现每次调用都用"当前窗口"重算 MAD，窗口滑动时
限值跟着动，实测在 8 344 个连续 500 根窗口里有 **8 343 个**的 Winsor 限值发生
变化（被剪裁值的最大 |Δ| ≈ 4.8e-2），即一根陈旧异常值可能被**重新放回**估计量，
α_t 因此非单调。现在用 `build_anchor(returns)` **对整个序列只算一次**中心与尺度
（同一 `median` / `1.4826·MAD` 配方，因此在全序列上输出与旧实现逐位相同），
然后把 `AnchorMAD` 传给 `clip_outliers` / `ewma_variance` / `ewma_vol`：同一根
bar 在任何包含它的窗口里被剪裁到同一个值，且**永远不会被解除剪裁**。

## 3. 本项目实现

**文件**：`core/ml/volatility.py`（估计量栈，单一接口 `forecast_vol`）、
`core/risk/position_sizer.py`（目标化仓位与止损宽度）、
`core/risk/position_guard.py`（动态移动止损）、
`core/executor/executor.py`（开仓时的止损宽度管道 + 预测缓存）、
`core/ml/labels.py`（barrier 宽度）。

### 3.1 单一接口

```python
from core.ml.volatility import forecast_vol, to_pct
vol = forecast_vol(df, method="ewma", window=500)      # 比例/bar
vol_pct = to_pct(vol)                                   # 0.52 (%/bar)
annual = forecast_vol(df, method="ewma", unit="annual") # 49.06 %
```

`method` ∈ `METHODS = (ewma, realized_cc, realized_parkinson,
realized_garman_klass, garch11)`；输入可以是收益序列**或** OHLCV DataFrame
（用 `close`）。`window` 对两种输入**都**生效（此前 DataFrame 分支会忽略它，
实测导致同一条数据在 frame / series 两种写法下 GARCH 结果不一致且慢 15×，
已修）。

`VolForecaster` 是给实时逐 bar 路径用的记忆化包装：按 `(symbol, interval)`
缓存，只有**最新 bar 变化**时才重算（`_on_kline` 每 tick 都会重建特征矩阵）。

### 3.2 实测：预报汇总（BTCUSDT 1h，**8 846 根切片**，window = 500）

> "当前值"是**窗口相关的**：EWMA 的 clip 锚随传入窗口变化，因此同一份数据在
> window = 500 与 window = 400 下给出不同末值（实测 **0.524216 %/bar** 对
> **0.524062 %/bar**）。下表全部是 **window = 500（`DEFAULT_WINDOW`）** 的值。
> 传入 `build_anchor(全部收益)` 后两种写法一致（锚固定，见 §2）。

| 方法 | 当前值 %/bar | 年化 % | 滚动均值 %/bar | 滚动 std | ms/次 |
|---|---|---|---|---|---|
| **ewma（默认）** | 0.5242 | 49.06 | 0.3969 | 0.1666 | **0.20** |
| realized_cc | 1.3224 | 123.77 | 0.4316 | 0.1147 | 0.053 |
| realized_parkinson | 1.2551 | 117.47 | 0.4534 | 0.1473 | 0.082 |
| realized_garman_klass | 1.4601 | 136.66 | 0.4606 | 0.1618 | 0.087 |
| garch11 | **0.4455** | 41.71 | 0.3707 | 0.2299 | **≈240–490** |

（滚动统计 = 从第 500 根起每 25 根重算一次。`ms/次` 一列**不是**旧的 IGARCH 网格版
`2.2 ms`：那是**无优化器网格回退**的成本（≈3 ms，见下），当前 `garch11` 走
自由 ω 的 MLE，实测见下。garch11 现在是**自由 ω 的
GARCH(1,1) MLE**：ω ≈ 2.96e-7（分数²）、α ≈ 0.0667、β ≈ 0.9199，持久性 ≈ 0.9866，
`0.5·ΣLL ≈ −225.9`。
**成本（当前修订实测；复现命令见 §3.3 末）**：默认 `ewma` 路径
`forecast_vol(df)` = **0.20 ms/bar**（11 675 根 frame，20 次均值；模块 docstring 在
500 根窗口上记 0.27 ms）——在 2 ms 预算内；`garch11` 在默认 `window = 500` 上
**0.24–0.49 s/次**（本机本次 0.24/0.25/0.23 s，模块 docstring 记 ≈0.49 s），
`window = 0`（整段历史 11 674 根）**6.2–6.4 s/次**——比 2 ms 预算高 2–3 个数量级，
因此 `garch11` 是 **opt-in**，默认实时路径是 EWMA。
2026-09-30 在增长后的 **11 674 根**缓存上复核：α ≈ 0.252、β ≈ 0.144、
预报 0.2995 %/bar 对 EWMA 0.3037 %/bar，比值 0.986——两种数据快照下预报都与
EWMA 同量级，没有退化；两种快照的参数差异本身就说明**GARCH 参数是窗口/样本
相关的**，引用时必须带上样本。）

高/低波动窗口（按 200 根一块的 realized vol 排序，取两端）：

| 窗口 | bar 区间 | realized %/bar | EWMA(末值) %/bar | 年化 realized |
|---|---|---|---|---|
| **低波动** | 2400–2600 | 0.2218 | 0.1861 | 20.76 % |
| **高波动** | 5800–6000 | 0.9258 | 0.8071 | 86.65 % |

* 高/低 realized 比 **4.17×**，EWMA 预报比 **4.34×** —— 预报确实随体制移动，
  且两者量级一致（EWMA 略低于同期 realized，因为 λ=0.94 的记忆被最近几根
  平静 bar 拉低，这是条件预测的正常行为）。
* 全样本 200 根块 realized vol 分位：**p05 0.2841 / p50 0.4064 / p95 0.6675 %/bar**
  （这解释了 `risk.vol_targeting.target_vol_pct` 默认取 **0.45**，≈ 中位数）。
* `vol_percentile`（当前预测在自己历史中的分位）：**0.8397**（全历史）/
  **0.8720**（近 500 根）—— 当前处于偏高波动体制；该量**无量纲**，可直接做
  风险阈值/熔断的门限。

### 3.3 GARCH(1,1)：从"单位持续性 + 混合"改为**自由 ω 的 MLE**（审计纠正）

> **本节曾被审计推翻（第 3 项缺陷），以下是重测后的正确版本。** 旧版声称自由 ω
> 的三参数 MLE **无界**、且"Nelder-Mead / L-BFGS-B / SLSQP 全部奔向 `ω=0, β=0`
> 角落并拒绝合成数据的真实参数"。在本文的 500 根窗口上重测：三个优化器**全部
> 收敛到同一片参数区**（ω ≈ 0.0027 %²、α ≈ 0.067、β ≈ 0.920，
> `0.5·ΣLL ≈ −225.9`，相互差 < 1e-4），并在 4 000 根合成 GARCH(1,1)
> （真值 ω=0.10 %²、α=0.12、β=0.80）上恢复到 **ω ≈ 0.110、α ≈ 0.105、β ≈ 0.801**。
> 也就是说：**MLE 不是病态的，旧文档的理由是错的**。

被推翻的还有两处数字口径：旧文档引用的两个"角落"数值（`1.8e4` / `3.6e5`）是某个量
的**求和**，而代码把它定义为**逐观测均值**（`_garch11_avg_ll` 返回
`mean(0.5(log v + x²/v))`）。按同一实现在当前 11 674 根缓存上重算（未剪裁、%²）：

```python
import numpy as np, pandas as pd
from core.ml.volatility import _garch11_avg_ll, log_returns
x = log_returns(pd.read_parquet("data/market/BTCUSDT/1h.parquet")["close"].values) * 100.0
x2, var_s = x * x, float(np.var(x, ddof=1))
_garch11_avg_ll(x2, var_s, 1.0, 0.0)      # 91.1936   <- 网格角落 α=1, β=0
_garch11_avg_ll(x2, var_s, 0.001, 0.0)    # 2004.18
```

这两个数（**91.19 / 2 004.18**，随样本变化）描述的是**网格版 IGARCH**
（`α = 1, β = 0`，`x²/v` 无下界）的角落，而不是 MLE 的输出——良定拟合不会落到那里。

现在 `garch11_params` 的路径：

1. **主路径**：自由 ω 的 GARCH(1,1) 高斯 MLE，参数箱
   `ω ∈ (0, 10V]`、`α, β ≥ 0`、`α + β ≤ 0.999`，先做一次**方差目标化**
   （`ω = V(1−α−β)`）粗网格扫描，再从该点做 Nelder-Mead 精修；拟合在
   `z = x/sd(x)` 上做，避免参数箱随单位变化（未归一化时同一模型在 %² 与
   分数² 两种单位下会落到不同盆地）。
2. **回退路径**：优化器不可用/失败时退回 **IGARCH 网格**（`α = 1−β`、`ω = 0`，
   0.01 步长、无优化器、≈2 ms）。这条回退是**诚实且便宜**的，也是
   `GARCH_FIT_WEIGHT = 0.5` 混合存在的原因：网格在本文缓存上会落到
   `α=1, β=0` 的**常数方差角落**，纯一步预报等于"最后一根收益平方"
   （= `to_pct(abs(r[-1]))`；8 846 根切片上 0.00076 %/bar，**随切片变化**——当前
   11 675 根切片上同一命令给 0.0722 %/bar），混合把输出限制在拟合一步方差与 EWMA 水平之间，
   使**任何**退化参数组合都不会产生退化**预报**。

**局限（诚实说明）**：单位持续性版本无法表达波动率均值回复；自由 ω 版本可以，
但代价是 **0.24–0.49 s/次**（默认 `window = 500`；整段历史 `window = 0` 为
**6.2–6.4 s/次**，含粗网格 + Nelder-Mead），因此 `garch11` 仍只用于研究/汇报，
默认实时路径是 EWMA。`omega/(1-alpha-beta)` 对回退分支是 **0/0**
（`ω = 0` 且 `α+β = 1`），该分支下"长期方差"就是当前水平，`garch11_params` 用
`fitted=False` 标注它。

```python
# 成本复现（当前修订，本机本次；perf_counter）
import time, pandas as pd
from core.ml.volatility import (forecast_vol, garch11_forecast, garch11_params,
                                log_returns)
df = pd.read_parquet("data/market/BTCUSDT/1h.parquet")
r = log_returns(df["close"].values)                  # 11 674 根
def per_call(fn, n=5):
    fn(); t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n
per_call(lambda: forecast_vol(df), n=20)             # 0.20 ms/bar（默认路径，预算 2 ms）
per_call(lambda: garch11_params(r), n=3)             # 0.24 s（默认 window=500）
per_call(lambda: garch11_forecast(r), n=3)           # 0.25 s
per_call(lambda: forecast_vol(df, method="garch11"), n=3)   # 0.23 s
per_call(lambda: garch11_params(r, window=0), n=3)   # 6.4 s（整段历史 11 674 根）
```

**另修一个单位错误**：`_garch11_variance` 曾把"分数² 的 ω"加进"百分² 的递归"，
且用**未剪裁**的收益去跑**已剪裁数据**拟合出来的参数，实测把预报推到
**7.0× / 9.6× EWMA**；修正后 500 根窗口上预报 0.4455 %/bar 对 EWMA 0.5242 %/bar
（比值 0.85）。

`arch` **未安装**，且**故意没有**加进 `requirements.txt`：为一个可选估计量引入
重型编译依赖，而它跑在逐 bar 路径上。`garch_backend()` 运行时探测
（本机实测返回 **`"scipy"`**），`garch11_params` 的返回值用 `backend` 字段记录
实际跑了哪条路径。

### 3.4 接线位置

| 位置 | 做了什么 | 开关关闭时 |
|---|---|---|
| `core/risk/position_sizer.py:vol_scale` | `clip(target/forecast, min, max)` | 恒返回 `1.0` |
| `position_sizer.calculate_position_size` | 固定比例风险额 × scale，随后**重新施加**硬上限与 `max_position_notional_pct` | 与原算式逐位相同 |
| `position_sizer.stop_distance_pct` | `clip(multiple × forecast, min, max)` | 返回 `max(soft.stop_loss_pct, hard.min_stop_loss_distance_pct)`（`volatility_expanding` 的 ×1.3 保留） |
| `position_sizer.trailing_stop_distance_pct` | 同上；`strategy_risk_exit` 覆盖仍然优先 | 返回 `hard.trailing_stop_distance_pct` |
| `position_sizer.barrier_widths_pct` | 返回分数宽度 | 返回 `None` |
| `core/risk/position_guard.py:forecast_vol_pct` | 从 `market_data.get_historical` 取历史算预报，TTL 300 s 缓存 | 直接返回 `None`（**不发起任何请求**） |
| `position_guard._update_trailing_stop` | 用 `pos["stop_vol_pct"]`（开仓时记录）或新预报决定距离 | 用固定 2 %，棘轮语义不变 |
| `core/executor/executor.py:vol_stop_ctx` | 开仓时给出 `{vol_pct, stop_pct}`；预测由 `set_forecast_vol_pct` 推入 | 返回 `{}` |
| `core/ml/labels.py:barrier_widths(vol_pct=...)` | 用预报替代 ATR 代理 | `None` → ATR 路径不变 |

> **缓存边界（审计第 8 项）**：`VolForecaster._cache` 现在**有上界**
> `MAX_FORECAST_CACHE_ENTRIES = 64`（LRU 淘汰：命中会把键移到队尾，因此每根 bar
> 都读的活跃 symbol 不会被一次性 symbol 挤掉）；实测 500 个不同键后缓存长度恒为 64。
> `OrderExecutor._forecast_vol_cache`（`core/executor/executor.py`，**不在本次写入
> 范围**）仍只靠 300 s TTL 与"已推送 symbol 数"约束：键来自配置的币种集合，
> 不再推送的 symbol 不会被任何读取触发过期。这是**已登记待办**，不是活跃泄漏
> （`tests/test_p34_audit_fixes.py::test_executor_forecast_cache_size_is_reported_not_fixed`
> 记录该边界）。

`labels.py` 的 `vol_pct` 是**分数**（与 `min_pct`/`max_pct` 同单位），
percent↔fraction 的换算只在 `PositionSizer.barrier_widths_pct` 里做一次。

> **⚠️ 三个 `barrier_*` 配置项当前是 inert（审计发现的第 5 项缺陷，尚未接线）**：
> `risk.vol_targeting.barrier_vol_multiple` / `barrier_min_pct` / `barrier_max_pct`
> 的唯一读取者是 `PositionSizer.barrier_widths_pct`，而**没有任何生产代码调用它**
> （`core/` 全树搜索只有定义处与注释；`tests/test_p34_audit_fixes.py`
> 用一条 tripwire 测试固定这一点，`position_sizer.barrier_widths_pct` 的 docstring
> 也标注了）。实盘 label 路径的宽度来自
> `ml.barrier_atr_period` / `ml.barrier_atr_multiple` / `ml.barrier_min_pct` /
> `ml.barrier_max_pct`（`MLPredictor._barrier_params`）。因此**运维改这三个键现在
> 不会产生任何效果**。接线需要改 `core/ml/predictor.py`（不在本次写入范围）：
> 在 `_barrier_params` 里用 `executor`/`risk` 的同一条预报覆盖 ATR 宽度，
> 或让 `manager.py` 的 `resolve_forecast_vol_pct` 结果流到 label 构建处。
> 已接线的实盘部分只有**止损/移动止损宽度**与**仓位 scale**（`manager.py` 的
> `resolve_forecast_vol_pct` 现在会读 `executor.vol_stop_ctx` 的 `vol_pct`，
> 因此 `executor.set_forecast_vol_pct` 不再是死代码）。

## 4. 实测效果

### 4.1 仓位证明（余额 10 000、satellite 池 0.3、固定比例 8 % → 240 USDT）

| 预报 %/bar | scale | 名义 USDT | 止损距离 % | 移动止损 % |
|---|---|---|---|---|
| 0.225 | 2.000 | 480.00 | 0.675 | 0.675 |
| **0.45（= 目标）** | 1.000 | **240.00** | 1.350 | 1.350 |
| **0.90（波动率翻倍）** | **0.500** | **120.00** | **2.700** | **2.700** |
| 1.80（×4） | 0.250 | 60.00 | 5.400 | 5.400 |
| 0.02 | 2.000 | 480.00 | 0.500 | 0.500 |

* 波动率翻倍 → 名义**恰好减半**（`scale ∝ 1/forecast`）；×4 → 四分之一
  （`min_scale` 处截断）；预报极小时 `max_scale` 截断，再由
  `max_position_notional_pct` 兜底。
* 保留的硬上限：`hard.max_position_size_pct` 50 %（5 000）、
  `max_position_notional_pct` 10 %（本配置下 1 000；注意在默认
  `max_scale=2.0` 下卫星仓 240×2 = 480 < 1 000，**该上限在默认设置下不可能触发**，
  这也是"默认不改变行为"的一部分）。

### 4.2 开关关闭时 = 原值（实测）

| 量 | 开关关闭（无论是否传入预报） | 开关打开 |
|---|---|---|
| 名义 USDT | **240.00** | 240.00 @ 目标 / 120.00 @ 翻倍 |
| `stop_distance_pct` | **2.000** | 1.350 @ 0.45 %/bar |
| `trailing_stop_distance_pct` | **2.000** | 1.350 |
| `barrier_widths_pct` | **None**（ATR 路径） | 分数宽度 |
| 移动止损距离（`PositionGuard`） | **2 %**，且**不请求历史数据** | 预报宽度，TTL 缓存 |

## 5. 局限与后续

1. **默认关闭，且尚未接入实盘建仓路径**。`risk.vol_targeting.enabled` 默认
   `false`。更重要的是：实盘信号→仓位这一步在 `core/risk/manager.py:check_signal`
   里调用 `PositionSizer.calculate_position_size(...)`，而该文件**不在 P3 的写入
   范围内**，因此它没有传 `forecast_vol_pct` —— 实盘**仓位**目前仍走固定比例
   （即"波动率不可用"的文档回退路径）。已接线的实盘部分是**止损/移动止损宽度**
   （`PositionGuard` 直接读配置与行情）。要让仓位目标化在实盘生效，需在
   `manager.py`（或预测器推入 `executor.set_forecast_vol_pct` 的同一条路径）补一行
   传参，属于下一步工作。
2. **未做 A/B 回测对照**。P3 只交付了机制与回归测试；"目标化后 Sharpe / 最大回撤
   是否改善"需要在开关两侧跑同一段回测才能回答，本阶段没有做，因此本文档
   **不给出任何收益类数字**。（`09-trailing-stop-algorithm.md` 里那张
   "移动止损 Sharpe 1.24" 的表已按审计要求标注为示意/非实测，请勿引用。）
3. **预估量的选择未做样本外校准**。Parkinson/Garman-Klass 在本文数据上明显高于
   close-to-close（117 %/136 % vs 124 % 年化），与文献"更高效"一致，但这只是
   点估计；真实取舍应看预测-实现回归的 `R²`，未做。
4. **1 484 小时的数据接缝**是数据供应商问题，剪裁只是防御；缓存重新抓取后
   应复查 `DEFAULT_OUTLIER_SIGMA` 是否需要收紧。
5. 自由 ω 的 GARCH 能表达均值回复（`ω/(1−α−β)` 即长期方差），但**回退的 IGARCH
   网格不能**（见 §3.3），且两者都未做 Student-t / EGARCH；若要用它做多日风险预算，
   需要先换成带 `arch` 的路径。

## 相关研究

1. **Engle (1982)**: "Autoregressive Conditional Heteroscedasticity…", *Econometrica* 50(4)
2. **Bollerslev (1986)**: "Generalized Autoregressive Conditional Heteroskedasticity", *Journal of Econometrics* 31(3)
3. **Tsay (2010)**: *Analysis of Financial Time Series*, 3rd ed., ch. 3（条件异方差）
4. **RiskMetrics (1996)**: Technical Document, 4th ed., ch. 5.3（λ = 0.94）
5. **Parkinson (1980)** / **Garman & Klass (1980)**: 极值型波动率估计量
6. **LeBeau (1995)**: Chandelier Exit —— `k × ATR` 型止损，与本项目的
   `stop_vol_multiple × forecast_vol` 同构（把 ATR 换成方差型预报）
