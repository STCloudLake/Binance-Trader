# 配对交易 / 协整（Pairs & Cointegration）

> **P4 新增（phase P4）**：`core/strategy/pairs.py`。全部数字均为**实测**（缓存 1h/4h/15m
> 真实数据 2025-06-03 → 2026-09-30，成本取自 `app.config` 的 sim 成本模型）。
> **本模块默认关闭**（`PAIRS_ENABLED = False`），未接入任何自动下单路径。

## 算法原理

Tsay《Analysis of Financial Time Series》：单资产收益近乎不可预测，但**多变量结构**
（第 8 章：协整 / 误差修正）与**状态空间时变参数**（第 7 章：Kalman 滤波）是少数被
稳健记录的收益来源。若两条对数价格协整，价差 `y − βx` 平稳，其偏离均值的部分可以
在不预测任何一条腿方向的前提下交易——因为是**一多一空**，暴露主要是市场中性。

## 公式

### 1. Engle-Granger 两步法（本模块用 numpy 自行实现，不依赖 statsmodels）

```
第一步（水平值 OLS）：  y_t = α + β x_t + e_t
第二步（残差 ADF）  ：  Δe_t = φ e_{t-1} + Σ ψ_i Δe_{t-i} + u_t      (无常数项)
                        τ = φ̂ / se(φ̂)
```

τ 必须与**估计残差**的零分布比较（Engle-Granger / MacKinnon N=2），不能用普通 DF
临界值——残差是估计出来的，分布左移，用错表会**过度拒绝**。

**零分布必须与被检验的回归一致**（审计发现的第 4 项缺陷）。这是模块 docstring
自己的契约，而早期实现违反过它：模拟零分布用的是**无滞后增广、无常数**的 ADF，
而真实检验用 AIC 选滞后（15m 序列上选中 **lag 1**）——正是 docstring 禁止的错配。
现在两种零分布都可选：

* `tau_null_distribution(..., max_lags=0)`（**默认**）：无增广基线，便宜，是一个
  **近似**。在 15m 长度（模拟长度 2000）上实测 5% 分位 **−3.3015**（近似）对
  **−3.3098**（匹配），差 **0.0083**；BTC/XRP 15m 的 p 值 0.1718 对 0.1713，
  **结论不变**。近似不再是隐含的：`engle_granger` 的返回值带
  `null_matched` / `null_lags` 两个字段说明用了哪一种，docstring 也写明。
* `tau_null_distribution(..., max_lags=k)` / `engle_granger(..., matched_null=True)`：
  用**与检验相同的 AIC 滞后规则**模拟（每个候选滞后解一次有界 OLS，约 3 s/次，
  比无增广基线慢 ~10×，因此是一个"决定性单次检验"的选项，而不是 30 个币对扫描的
  默认）。

**statsmodels 探测**：本机**未安装** statsmodels（`HAS_STATSMODELS = False`），因此
零分布由 Monte-Carlo **模拟**产生并缓存（`tau_null_distribution`，4000 条路径、固定
种子、分块模拟，结果逐位可复现；缓存键包含 `max_lags`/`const`，上界 8 条）。
`statsmodels_adf_tau()` 只是在装了 statsmodels 的环境下做交叉验证用，生产路径不
依赖它。

### 2. Kalman 时变对冲比（Tsay 第 7 章）

```
状态：  θ_t = [β_t, α_t]ᵀ,      θ_t = θ_{t-1} + w_t,   w ~ N(0, Q),  Q = δ/(1−δ)·I₂
观测：  y_t = [x_t, 1]·θ_t + v_t,                       v ~ N(0, R)
预测：  θ_{t|t−1} = θ_{t−1},  P_{t|t−1} = P_{t−1} + Q      （θ₀ = OLS [β, α] 热启动）
更新：  v_t = y_t − zᵀθ_{t|t−1},  F_t = zᵀP z + R,  K_t = P z / F_t
        θ_t = θ_{t|t−1} + K_t v_t,  P_t = (I − K_t zᵀ) P_{t|t−1}
```

`v_t` 是**样本外**价差新息：`β_t` 只用到 `t−1` 的信息。

**`R` 必须取 OLS 残差方差，不能取 `var(y)`**——这是开发中实测到的 bug：BTC/ETH 1h 上
`var(y) = 0.0400` 而 OLS 残差方差只有 `0.0094`（4.3 倍），用 `var(y)` 时滤波器几乎不
更新，`β` 塌缩到 **0.058**（OLS 为 **0.628**），"价差"退化成 `y` 的原始价格，配对交易
悄悄变成了单边方向赌注。回归测试：
`tests/test_pairs.py::test_kalman_observation_noise_is_the_spread_noise_not_var_y` 与
`::test_real_cached_btc_eth_kalman_is_close_to_ols_and_stable`。

### 3. 半衰期 / 回看窗口（OU / AR(1)）

```
Δs_t = a + b s_{t-1} + ε_t,   κ = −b,   half_life = ln2/κ (bars),   均衡 = −a/b
L = clip(round(4 × half_life), 30, 250)          # 回看窗口
```

半衰期在 **OLS 残差**上估计（被检验的那个价差）；Kalman 价差的半衰期只作诊断——
它天然很小（见"局限"）。

### 4. z-score 规则（阈值即模块常量）

```
z_t = (s_t − mean(s_{t−L..t−1})) / std(s_{t−L..t−1})      # 不含 t 自身
z ≥ +2.0 → 做空价差（空 y、多 βx）      z ≤ −2.0 → 做多价差
|z| ≤ 0.5 → 平仓                        |z| ≥ 4.0 → 止损并锁定，
                                        直到 |z| < 2.0 才允许重新入场
```

### 5. 硬门（`pair_guard`）——任一条件不满足即**拒绝交易**

| 条件 | 常量 | 默认 |
|---|---|---|
| 样本长度 | `PAIRS_MIN_OBS` | ≥ 250 bar |
| 协整检验 | `PAIRS_MAX_ADF_PVALUE` | p ≤ 0.05（或 τ 非有限 → 拒绝） |
| 半衰期 | `PAIRS_MIN/MAX_HALF_LIFE` | 2 – 120 bar |
| 对冲比 | 有限且非零 | β ≠ 0 |

### 6. 成本

一次配对往返 = **四条腿的成交**，因此按两条腿各收一次往返成本：`leg_round_trip_cost_pct`
取自 `core.ml.credibility.cost_pct_for`（sim 成本模型），实测 **0.25%/腿**，即一次配对
往返约 **0.50%** 的毛名义价值。

## 实现位置

| 内容 | 位置 |
|---|---|
| ADF / E-G / 模拟零分布 | `core/strategy/pairs.py:250-520`（`adf_regression`, `engle_granger`, `tau_null_distribution`） |
| Kalman 对冲比 | `core/strategy/pairs.py::kalman_hedge_ratio` |
| 半衰期 / 回看 / 硬门 | `ou_half_life`, `lookback_from_half_life`, `pair_guard` |
| z-score / 状态机 / PnL | `rolling_zscore`, `pairs_positions`, `spread_returns`, `pair_trades` |
| 与信号核对接 | `PairsSignal.to_kernel_input()` → `evaluation_kernel.fuse_signals(indicator_signal=…, ml_enabled=False)` |
| 引擎集成缝 | `core/strategy/engine.py::StrategyEngine.wire_pairs_provider`（`P4_PAIRS_SIGNALS_ENABLED = False`） |
| 测试 | `tests/test_pairs.py`（25 条） |

## 实测数字

### 模拟零分布的标定（必须做对，否则门控无意义）

| 分布 | 实测临界值 1% / 5% / 10% | 文献值（MacKinnon） |
|---|---|---|
| DF，含常数（T→∞） | −3.40 / −2.85 / −2.54 | −3.43 / −2.86 / −2.57 |
| E-G，N=2，含常数 | −3.88 / −3.32 / −3.03 | −3.90 / −3.34 / −3.04 |

独立校验：在 300 条 T=500 的随机游走上跑 `adf_regression(regression="c")`，τ 的 5%
分位与模拟表相差 < 0.30，均值 −1.57。

**功效与尺寸**（合成数据，20 次重复）：协整对 **20/20** 在 5% 水平被拒绝（p 中位数 < 0.01）；
两条独立随机游走 **2/20**（5% 名义水平下 0–4 属正常）——审计复核时旧文档写的是
**1/20**，重跑 200 条 n=300 的随机游走得到的尺寸是 **6/200 = 0.030**（5% 水平）
与 **1/200 = 0.005**（1% 水平），即「20 次里 2 次」与「200 次里 6 次」是同一量级的
不同抽样；此处统一按 **2/20** 记录并给出 200 次的分辨率更高的测量。
复现：`tests/test_p34_audit_fixes.py::test_adf_size_experiment_reproduces`（seed
12345、200 条 n=300）本机本次实测 **6/200 = 0.030** 与 **1/200 = 0.005**（另一组
抽样给 7/200 = 0.035）；该测试断言的是 **≤ 0.08 / ≤ 0.03 的保守上界**（名义 5% 的
三倍余量），**不是点估计**——尺寸实验只用于证明门控没有被过度拒绝，不应该被当作
精确的 5% 校准。

### 真实数据：30 次 E-G 检验（10 个币对 × 1h/4h/15m）

**0 / 30 通过 5% 水平**。最好的三个：BTC/XRP 15m p = 0.168（n = 35 385）、
BTC/XRP 4h p = 0.171、ETH/SOL 4h p = 0.176。加密主流币在本样本内**不协整**。

### 5 个币对（1h, n ≈ 8 846）明细

| 币对 | β(OLS) | τ | p | 半衰期(OLS) | 半衰期(Kalman) | 门控结论 |
|---|---|---|---|---|---|---|
| BTC/ETH | 0.628 | −2.12 | 0.451 | 1 202 bar | 1.2 | 拒绝：p 过高 + 半衰期过长 |
| BTC/XRP | 0.624 | −2.70 | 0.190 | 422 bar | 3.2 | 拒绝：同上 |
| ETH/SOL | 0.751 | −2.69 | 0.196 | 637 bar | 1.6 | 拒绝：同上 |
| SOL/BNB | 1.320 | −1.80 | 0.623 | 1 439 bar | 2.6 | 拒绝：同上 |
| ETH/BNB | 1.156 | −1.51 | 0.750 | 1 347 bar | 1.9 | 拒绝：同上 |

**对冲比稳定性（Kalman vs OLS）**：Kalman 均值与 OLS 接近（0.606/0.628、0.510/0.624、
0.758/0.751、1.292/1.320、1.125/1.156），且远**平滑**于 250-bar 滚动 OLS
（β 标准差 0.019 / 0.146 / 0.025 / 0.053 / 0.043 对 0.245 / 0.301 / 0.348 / 0.561 / 0.754）。
Kalman β 的漂移（首尾各 10% 均值之差 / 平均 |β|）为 0.061–0.37；把 δ 从 1e-4 调到 1e-5
基本不改变平滑度（0.028 vs 0.019 BTC/ETH），说明**漂移来自数据不是滤波器设定**。

**若绕过硬门强行交易**（静态 OLS 价差、回看 250 bar、成本 0.50%/往返）：

| 币对 | 笔数 | 毛利均值 | **净利均值** | t | PSR | 胜率 | 累计 |
|---|---|---|---|---|---|---|---|
| BTC/ETH | 47 | −0.005% | **−0.255%** | −1.53 | 0.063 | 55% | −11.98% |
| BTC/XRP | 63 | +0.164% | **−0.084%** | −0.41 | 0.341 | 67% | −5.27% |
| ETH/SOL | 49 | +0.205% | **−0.046%** | −0.16 | 0.435 | 67% | −2.23% |
| SOL/BNB | 69 | +0.366% | **+0.118%** | 0.57 | 0.717 | 77% | +8.13% |
| ETH/BNB | 46 | −0.145% | **−0.393%** | −1.16 | 0.124 | 63% | −18.06% |

即：毛收益偶为正，扣掉成本后**没有一对**达到 |t| > 2；门控拒绝的正是这些成本主导的噪声。

**样本内 vs 样本外**（前半段拟合 β 与回看窗口并冻结，只交易后半段）：21–29 笔，
净利均值 −0.59% … +0.37%，t = −0.86 … +0.89，PSR 0.20–0.81 → 与 0 无法区分。
后半段的 OLS 价差半衰期 298–400 bar（仍远超 120 上限），再次触发拒绝。

**滚动窗口**（500 bar，17 个非重叠窗口/币对）：通过比例 0.00–0.176（0–3 个窗口），
而**数据挖掘基线就是 0.05**——即"某些窗口协整"完全是噪声；窗口间 β 标准差 0.18–0.76。

## 局限（明写）

1. Engle-Granger 只能找**一条**关系；k > 2 资产需要 Johansen（未实现），门控因此只支持两腿。
2. p 值是**模拟**的有限样本值（4000 条路径），分辨率为 1/4000，精度约 1e-3；已对
   MacKinnon 渐近表标定，但不是精确 p 值。
3. **Kalman 价差的半衰期不能当平稳性证据**：滤波器靠自适应 β 吸收水平漂移，任何价差
   在它眼里都"均值回复"（实测 1.2–3.2 bar）。协整检验只能建立在 OLS 残差上。
4. 协整是**关于过去**的陈述：样本外那一半才是诚实的检验，本页已分开报告。
5. 半衰期 400–1 400 bar 意味着"均值回复"的时间尺度比样本还长——门控因此拒绝；
   若强行交易，它就是一个带杠杆的方向性持仓。
6. 数据：`data/market/*.parquet` 由运行中的服务持续追加（每小时 +1 根），因此 p 值会
   在第 3 位小数上移动；本页数字对应 n ≈ 8 846 的切片。
7. **仅作研究能力**：等权的隐含假设（两腿同时成交、无融资费、无借币成本）未建模。
8. `PAIRS_ENABLED = False`：不注册任何配对策略，实盘路径逐字节未变。

## 开关（全部默认 off）

| 常量 | 默认 | 作用 |
|---|---|---|
| `core.strategy.pairs.PAIRS_ENABLED` | `False` | 是否允许在实盘注册配对策略（当前无任何代码读它） |
| `core.strategy.engine.P4_PAIRS_SIGNALS_ENABLED` | `False` | 是否让已注册的 provider 覆盖 `indicator_signal` |
| `PAIRS_MIN_OBS` / `PAIRS_MAX_ADF_PVALUE` | 250 / 0.05 | 硬门 |
| `PAIRS_MIN/MAX_HALF_LIFE` | 2 / 120 bar | 硬门 |
| `PAIRS_LOOKBACK_MULT` / `MIN` / `MAX_LOOKBACK` | 4.0 / 30 / 250 | 回看窗口 |
| `PAIRS_Z_ENTRY` / `PAIRS_Z_EXIT` / `PAIRS_Z_STOP` | 2.0 / 0.5 / 4.0 | 信号阈值 |
| `KALMAN_DELTA` / `KALMAN_P0` | 1e-4 / 1.0 | 状态随机游走 / 初始协方差 |
| `SIM_REPS` / `SIM_SEED` | 4000 / 20240617 | 零分布模拟规模 / 种子 |

---

## 附：P4 其余两个模块（microstructure / regime）的位置、开关与实测

两个模块的**原理、公式、局限**写在各自模块 docstring 顶部（`core/market_data/microstructure.py`、
`core/strategy/regime.py`），此处只登记位置、开关与验收数字。

### 盘口/微观结构（Tsay 第 5 章）

`core/market_data/microstructure.py`，测试 `tests/test_microstructure.py`（19 条）。
特征：`ofi_depth`（盘口买卖量不平衡）、`ofi_depth_weighted`（按距离衰减，半衰点 5 bp）、
`ofi_trades`（主动方成交量不平衡）、`microprice` / `microprice_dev_bps`、`spread_bps`、
`book_slope_ratio`、成交笔/量/大单占比、`rv_trade`（按成交价、每 10 笔采样）、
`arrival_rate_hz` / `activity_ratio`。

**实测真实快照**（`data-api.binance.vision`，`/api/v3/depth?limit=20` + `/api/v3/trades?limit=100`，
`as_of` = 最后一笔成交时间 + 1 ms，`book_age_ms = 120`）：

| | BTCUSDT | ETHUSDT | SOLUSDT |
|---|---|---|---|
| mid | 83 384.005 | 2 674.565 | 119.735 |
| microprice | 83 384.0005 | 2 674.5601 | 119.7317 |
| microprice 偏离 (bp) | −0.0005 | −0.018 | −0.279 |
| 价差 (bp) | 0.0012 | 0.037 | 0.835 |
| ofi_depth / 加权 | −0.849 / −0.849 | −0.729 / −0.752 | −0.139 / −0.209 |
| ofi_trades | +0.474 | −0.664 | −0.085 |
| book_slope_ratio | 12.20 | 6.29 | 1.32 |
| 笔数 / 中位成交量 | 100 / 0.0003 | 100 / 0.00195 | 100 / 0.046 |
| 大单量占比 | 0.665 | 0.865 | 0.871 |
| rv_trade | 0.000000 | 0.000021 | 0.000167 |
| 到达率 (笔/秒) | 7.06 | 18.94 | 7.74 |
| activity_ratio | 0.490 | 0.037 | 1.043 |

**前视论证与测试**：`compute_features(order_book, trades, as_of_ms=…)` 在**任何统计之前**
丢弃 `time > as_of_ms` 的成交（计在 `dropped_future_trades`）；深度簿无时间戳，因此报告
`book_age_ms`，由调用方拒绝过期快照。测试注入一笔 `as_of+5000 ms`、数量 1000 的主动买单：
`dropped_future_trades = 1` 且**所有特征逐位不变**（`lookahead_identical = true`）；
反向测试证明若不过滤，`ofi_trades` 会从 +1.0 翻到 < −0.9（该测试非平凡）。
`fetch_features` 实测：首次 0.171 s，命中 TTL 缓存 0.0000 s。

开关：`MICROSTRUCTURE_ENABLED = False`、`MICROSTRUCTURE_CACHE_TTL_SECS = 5.0`、
`MAX_DEPTH_LEVELS = 20`、`MAX_TRADES = 1000`、`MAX_CACHE_ENTRIES = 64`、
`RV_TRADE_STRIDE = 10`、`DEPTH_DECAY_BP = 5.0`。`core/market_data/data_client.py`
**未改动**（其 `order_book` / `recent_trades` 已够用）。

### Regime 门控（Tsay 第 4 章）

`core/strategy/regime.py`，测试 `tests/test_regime.py`（14 条）。两套分类器：
**波动率三分位**（阈值取 `expanding` 分位并 `shift(1)`，只用过去）+
**趋势过滤**（EMA50/200 对齐）；以及**2 状态高斯 HMM**（numpy EM/Baum-Welch + Viterbi，
按 σ 重标号使状态 0 恒为低波动，确定性初始化，无随机种子）。

合成序列（3 000 bar：低波+上行 → 高波+下行 → 低波+上行，切换点 1 000/2 000），3 个种子：

| 种子 | HMM σ 恢复（真值 0.002/0.010） | HMM 准确率 / 延迟(bar) | 三分位 准确率 / 延迟 | 趋势 准确率 / 延迟 |
|---|---|---|---|---|
| 5 | 0.00202 / 0.00968 | **0.9987** / [3, 0] | 0.826 / [3, 19] | 0.817 / [110, 92] |
| 6 | 0.00200 / 0.01015 | **0.9993** / [1, 0] | 0.860 / [0, 1] | 0.785 / [95, 110] |
| 7 | 0.00195 / 0.01026 | **0.9993** / [2, 0] | 0.783 / [2, 10] | 0.892 / [22, 117] |

> **⚠️ 这张 HMM 准确率是 in-sample（审计发现的第 1 项缺陷）。** 这些数字来自
> `hmm_two_state` 的**全样本 EM 拟合**：μ/σ/A 见过被评分的整段序列，所以"准确率
> 0.999"衡量的是拟合优度，**不是可交易性**。截断到 2 000 根会让 σ 从
> 0.00202/0.00968 变成 0.00199/0.00968，并让 t 之前的 Viterbi 标签发生移动——
> 即标签本身依赖未来数据。因此**门控现在拒绝非因果标签**：
> `classify_regimes(..., causal_hmm=True)` / `hmm_two_state_causal` 用"只在过去
> 重新拟合 + 纯前向滤波"产生标签，`REGIME_GATING_ENABLED = True` 时
> `hmm_two_state` 自动切到该路径，`gate_regimes` 对
> `attrs["causal_hmm"] is False` 的表抛 `NonCausalRegimeError`。
>
> **因果路径的样本外实测**（同一合成结构，warm-up 250、每 250 根重拟合，当前修订
> 重测）：准确率 **0.758 / 0.759 / 0.815**（seed 5 / 6 / 7）对 in-sample 的
> **0.9987 / 0.9993 / 0.9987**。引用时**必须**分开写：`0.816` 是好的那一档
> （seed 7 实测 0.8149），`0.758` 是漂移的那一档（seed 5/6 实测 0.7579/0.7589），
> 而 `0.999` 是 in-sample 的拟合优度。索引口径修正前（`fwd[k]` 错位，标签最多滞后
> 2 500 根）同一测量只有 **0.156**——约等于抛硬币。切换点延迟实测
> **5/24、29/22、145/27 bar**（seed 5/6/7，即首/次切换点），**不是**早期写的
> "0–28 bar"：延迟随种子波动很大，seed 7 的首个切换点要 145 根才认出来。
>
> 复现：`python -m pytest tests/test_p34_audit_fixes.py -q -k "causal_hmm"`（断言
> `0.3 < acc < 0.95`、`in-sample > 0.95`、`oos < in-sample − 0.2`）；逐 seed 数字用
> `_synth(seed=s)`（该测试文件的构造）+ `detection_metrics` 复算。
> 因果路径耗时实测 **≈3.4–3.8 s / 3 000 bar**（12 次 EM 拟合 + 12 次前向扫描）；
> `core/strategy/regime.py` 的 docstring 写"≈18–21 s"，那是旧实现/旧机器的数字，
> 本页以重测值为准（该 docstring 待改）。

同种子重跑逐位一致（`regime_deterministic = true`）；HMM 估计的持续性
`P(stay) = 0.999`，分段数 3、平均段长 999.7 bar。**平滑后验只作诊断**，
可交易的是 `posterior_filtered`（因果模式下为纯前向滤波结果），模块文档明确标注
`posterior_smoothed` *不可交易*。

开关：`REGIME_GATING_ENABLED = False`（禁用时 `RegimeGate.allows()` 恒为 `True`）、
`REGIME_DIAGNOSTICS_ENABLED = False`、`VOL_WINDOW = 50`、`TREND_FAST/SLOW = 50/200`、
`HMM_ITER/HMM_TOL = 50/1e-6`、`MIN_REGIME_ROWS = 100`、`DEFAULT_REGIME_MAP`
（仅当调用方显式启用时才被 `default_gate(enabled=True)` 使用）。
引擎侧：`P4_REGIME_DIAGNOSTICS_ENABLED = False`。

### P4 开关总览（**全部默认关闭，实盘行为逐字节不变**）

> **元标签（doc 12）表格的可复现性（审计第 9 项，登记在此）**：doc 12 的
> 10 行"一级策略标签基准率"表里 **9/10 行可复现**；第 10 行 **`ADX+EMA`** 在按规则
> 重建时**报错**（`model_factory(Xf, yf, w_fit)` 的签名不匹配），因此该行只能用原始
> 运行的输出复现，不能用重建脚本。doc 12 不在本次写入范围内，故在此登记：
> 引用该表时请只引用 9 行，或先修 `model_factory` 的调用签名。

| 文件 | 常量 | 默认 |
|---|---|---|
| `core/strategy/pairs.py` | `PAIRS_ENABLED` | `False` |
| `core/ml/meta.py` | `META_LABELING_ENABLED` | `False` |
| `core/market_data/microstructure.py` | `MICROSTRUCTURE_ENABLED` | `False` |
| `core/strategy/regime.py` | `REGIME_GATING_ENABLED` / `REGIME_DIAGNOSTICS_ENABLED` | `False` / `False` |
| `core/strategy/engine.py` | `P4_PAIRS_SIGNALS_ENABLED` / `P4_META_FILTER_ENABLED` / `P4_REGIME_DIAGNOSTICS_ENABLED` | `False` / `False` / `False` |

`core/strategy/loader.py`**未改动**：新增能力都不需要新的 YAML 字段——配对信号来自
注册的 provider，meta 过滤来自注册的 `MetaLabeler`，regime 标签在模块内计算。
因此**没有**为了本阶段而给策略 schema 增加字段。
