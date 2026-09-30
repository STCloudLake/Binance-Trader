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

### 数据质量：缓存的接缝（实测，必须剪裁）

`data/market/BTCUSDT/1h.parquet` 有 8 846 根、11 处 >1.5 h 的空洞，
**最大空洞 1 484 小时**：2026-07-29 → 2026-09-29 被当作"一根 bar"，
产生一个 `+27.63 %` 的对数收益（正常 bar 中位绝对值 0.19 %）。
EWMA 的**有效记忆只有 `1/(1−λ) ≈ 16.7` 根**，不剪裁时该异常值直接成为未来
17 根的预测：实测 **5.47 %/bar**，而剪裁后为 **0.52 %/bar**，相差 **10.5×**。
因此所有估计量先用 `clip_outliers`（`±6 × 1.4826 × MAD`，可关）做 Winsorize；
用 MAD 而不是标准差，因为要被防的正是会污染后者的量。

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

### 3.2 实测：预报汇总（BTCUSDT 1h，8 846 根，window = 500）

| 方法 | 当前值 %/bar | 年化 % | 滚动均值 %/bar | 滚动 std | ms/次 |
|---|---|---|---|---|---|
| **ewma（默认）** | 0.5242 | 49.06 | 0.3969 | 0.1666 | **0.131** |
| realized_cc | 1.3224 | 123.77 | 0.4316 | 0.1147 | 0.053 |
| realized_parkinson | 1.2551 | 117.47 | 0.4534 | 0.1473 | 0.082 |
| realized_garman_klass | 1.4601 | 136.66 | 0.4606 | 0.1618 | 0.087 |
| garch11 | 0.3745 | 35.06 | 0.3707 | 0.2299 | 2.212 |

（滚动统计 = 从第 500 根起每 25 根重算一次。）

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

### 3.3 GARCH(1,1)：为什么是"单位持续性 + 混合"这一版

`arch` **未安装**，且**故意没有**加进 `requirements.txt`：为一个可选估计量引入
重型编译依赖，而它跑在逐 bar 路径上。因此 `garch_backend()` 运行时探测
（返回 `"arch"`/`"scipy"`，本机实测 **`scipy`**），两条路径都在 `garch11_params`
里实现并记录在返回值 `backend` 字段。

实现过程中三种"更直觉"的写法都被实测否决，理由保留在代码 docstring 与测试里：

1. **自由 ω 的三参数 MLE**（有解析梯度）：Gaussian 似然**无界** ——
   `ω=0, β=0` 时 `v_t = α x²_{t-1}`，一根接近 0 的收益就把 `log v_t → −inf`。
   Nelder-Mead / L-BFGS-B / SLSQP（解析与数值梯度都试过）**全部**奔向该角落，
   并拒绝合成数据的真实参数。（解析梯度本身是对的，已用"沿负梯度走一步必须
   降低目标函数"这条契约固定下来；有限差分在这里反而不可靠，因为目标是分段的。）
2. **只做方差目标化（`ω = V(1−α−β)`）的二维网格**：没有消除病态，因为
   `x_{t-1}` 极小时滤波仍会塌到方差下限 —— 合成数据上似然偏好
   `(α, β) = (0.001, 0)`，均值 **1.8e4**（下限本身成了最优解）。
3. **绝对方差下限**（`1e-9`）：同样让下限成为最优解（均值 **3.6e5**）。
   最终用**相对下限** `var_s × 1e-4`。

最终版：**单位持续性** IGARCH，`ω = 0, α + β = 1`，只在 `β ∈ [0, 0.99]`
（步长 0.01）上按高斯似然选最优，**无优化器**（确定性、不会发散、向量化后
**2.2 ms/次**）。预报再与 EWMA 水平按 `GARCH_FIT_WEIGHT = 0.5` 混合
（`arch` 包的 `forecast(horizon=1)` 默认也是 `0.5σ²_{t+1} + 0.5σ²_t`）——
这是必需的：在本文这条 500 根窗口上网格拟合落在 `α=1, β=0` 的**常数方差角落**，
纯一步预报会等于"最后一根收益的平方"（实测量级偏低 5.5×），混合后
`0.3745 %/bar` 与 EWMA `0.5242 %/bar` 同量级，退化参数不会产生退化预报。

**局限（诚实说明）**：单位持续性意味着该估计量**无法表达波动率均值回复**，
长期预测等于当前水平；它也不用做方向；`garch11` 的 2.2 ms/次适合研究/汇报，
默认实时路径用 EWMA。

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

`labels.py` 的 `vol_pct` 是**分数**（与 `min_pct`/`max_pct` 同单位），
percent↔fraction 的换算只在 `PositionSizer.barrier_widths_pct` 里做一次。

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
5. GARCH 未做均值回复（见 §3.3），且未做 Student-t / EGARCH；若要用它做
   多日风险预算，需要先换成带 `arch` 的路径。

## 相关研究

1. **Engle (1982)**: "Autoregressive Conditional Heteroscedasticity…", *Econometrica* 50(4)
2. **Bollerslev (1986)**: "Generalized Autoregressive Conditional Heteroskedasticity", *Journal of Econometrics* 31(3)
3. **Tsay (2010)**: *Analysis of Financial Time Series*, 3rd ed., ch. 3（条件异方差）
4. **RiskMetrics (1996)**: Technical Document, 4th ed., ch. 5.3（λ = 0.94）
5. **Parkinson (1980)** / **Garman & Klass (1980)**: 极值型波动率估计量
6. **LeBeau (1995)**: Chandelier Exit —— `k × ATR` 型止损，与本项目的
   `stop_vol_multiple × forecast_vol` 同构（把 ATR 换成方差型预报）
