# P6 规划：成交量 / 资金流（Volume & Flow）

> 本文件冻结 P6 的目标、阶段划分、**量化验收标准**、可复现证据要求、回归测试与审计门。
> 与既有 `ALGO_UPGRADE_PLAN.md`（P1–P5）同一标准：每个阶段都必须有实测数字、逐阶段独立只读审计、提交推送，
> 并以"**是否被自己的门接受**"为准，而不是以"是否看起来有效"为准。
>
> 规划依据的本轮实测状态：`3e90013`（2026-09-30）。当前仓库 **1073 passed / 0 failed**。

---

## §0 为什么做 P6（问题陈述，含已实测的证据）

1. **系统对成交量的利用很薄。** ML 特征契约共 **39 列**，与量能相关的只有 5 列：
   `volume_ratio`、`vol_chg_5`、`vol_chg_20`、`vol_trend`、`frac_vol_10`
   （`core/ml/features.py`）；GA 可进化的量能条件只有 `volume_ratio > 1.5/2.0` 与 `obv > obv_sma`
   （`core/ga/genome.py:98,99,107`）。
2. **最强的短周期信息被实现却关着。** P4 的 `core/market_data/microstructure.py` 已有 **20** 个盘口/逐笔特征
   （`microprice`、`ofi_depth`、`ofi_trades`、`trade_large_share`、`rv_trade`、`arrival_rate_hz`、
   `activity_ratio` …）与逐位无前视证明，但**默认关闭且未并入 39 列契约**，因此对预测路径零贡献。
3. **量能的方向性价值在本项目的周期上已被证伪两次。** 1h 上的方向预测两次未过自家门：
   BTC OOS AUC 0.5235、ETH 0.5342，净期望为负（`0.5342 / −0.1822 % / t=−2.28 / PSR 0.011`），
   `ml.enabled` 默认 `false`。**因此 P6 不以"提升方向预测"为卖点**。
4. **成交量的最高价值用法是容量与成本**，其次是状态调节，最后才是短周期资金流特征。
   P6 按此排序，并把"是否真的更好"交给可复现实验与门槛判定。

## §1 非目标（明确不做 / 不宣称）

- 不宣称成交量能带来方向性 alpha；任何"看起来变好"的结果都必须过既有门（AUC / 净成本期望 / 交易数 / t / PSR）。
- 不引入盘口深度冲击模型（无 L2 历史数据）；`impact_k` 在没有标定数据前**保持示意值**。
- 不默认开启任何能力：`ml.enabled`、`risk.vol_targeting.enabled`、`risk.liquidity.enabled`、P4 八开关
  全部保持 `false` 直到对应阶段的验收标准被实测满足。
- 不修改线上数据库/策略文件；不在验证中写入 `data/binance_trader.db` 或 `strategies/`。

## §2 现状清单（规划所依据的代码事实）

| 组件 | 现状 | 位置 |
|---|---|---|
| ML 特征契约 | 39 列，量能占 5 列；有 `feature_schema_hash` 版本机制 | `core/ml/features.py` |
| GA 量能条件 | `volume_ratio > 1.5/2.0`、`obv >/ < obv_sma`；`INDICATOR_NAMES` 含 `obv` | `core/ga/genome.py:98-130,167` |
| 盘口/逐笔特征 | **20** 项 `FEATURE_KEYS`（本轮复核 `len(FEATURE_KEYS) == 20`），来自实时 book/trades，默认关 | `core/market_data/microstructure.py:125-131` |
| 缓存字段 | **只有 `open/high/low/close/volume`**，无 quote volume / trade count | `data/market/<SYM>/<itv>.parquet` |
| 成交额近似 | `Σ volume×close`（P6-A 已实现，作为缺失 quote volume 的代理） | `core/risk/liquidity.py:283` |
| 参与率与冲击 | `participation_pct` / `cap_notional` / `impact_pct` / `trade_impact_pct` | `core/risk/liquidity.py:370-561` |
| 成本模型 | 手续费 + 半价差 + 滑点；可选冲击项（默认 `impact_k=0`） | `core/backtest/cost_model.py` |
| 配置 | `risk.liquidity.{enabled,max_participation_pct,lookback_bars,impact_k,impact_exponent,per_symbol}`（默认关） | `config/config.yaml` |
| 相关文档 | `docs/core-algorithms/13-volume-liquidity-costs.md`（P6-A） | 本目录 |

> ⚠️ 关键缺口：**缓存里没有 quote volume**，跨币种可比量能目前只能靠 `volume×close` 代理。
> P6-B 的第一项任务就是补齐该字段（数据源 `data-api.binance.vision` 的 kline 返回 quote asset volume）。

---

## §3 阶段划分

### P6-A 执行真实性：参与率上限 + 冲击成本 —— ✅ 已完成（提交 `b49883b`）

- 交付：`core/risk/liquidity.py`、仓位参与率上限（最后应用、只缩不放）、回测成本可选冲击项、
  `risk.liquidity` 配置块（默认关）、**26** 项测试（`python -m pytest tests/test_liquidity.py --collect-only -q` 末行实测 26）、文档 13。
- 已实测验收：
  - XRP 1m、20 根窗口 265.8 万 USDT、50 000 USDT 单 → 参与率 **1.881 %** → **缩到 26 575.74 USDT**；
  - 1.0 BTC 往返（**按文档 13 所述具名窗口**，2026-09-30 重测）：0.01 BTC `+0.0017 USDT`（+0.0002 %）、1.0 BTC `+1.7455 USDT`（+0.0021 %）、10 BTC `+55.1963`、100 BTC `+1,745.4596`；`k=0` 与旧实现**逐位一致**（`75.48621426`）。⚠️ 更早版本曾引用 `+2.36 % / +11.82 %`，那是**另一个（小得多、未声明）窗口**下的数值，已在 `docs/core-algorithms/13-volume-liquidity-costs.md` 撤回；**不要再用**。
  - 关闭路径**逐位一致**（`enabled:false` 对任意入参元组相同；`k=0` 时 100 笔真实成交成本 `==`，含 `repr`）；
  - 每调用成本 `cap_notional` 0.92–0.94 µs、完整路径 1.89–1.94 µs（断言 <200 µs）。
- 剩余缝隙（**P6-B/E 收口**）：实盘 `RiskManager` 未传 `recent_quote_volume`；回测引擎未把逐 bar 成交量
  喂给 `apply_trading_costs`（各差一个关键字参数）。

### P6-B 量能特征契约 v2 + 缓存字段扩展

**目标**：把量能从"粗粒度相对量"升级为**一等特征族**，并对契约做**版本化**，使旧模型自动被拒。

任务：
1. **数据层**：kline 缓存增加 `quote_volume`（USDT 成交额）与 `trade_count` 两列（来源 kline 字段），
   写一个可复现的回填脚本；`download_history.py --merge` 与 `OHLVCache` 的 bar 键去重规则必须继续成立。
2. **特征层**（新增到契约，命名前缀 `vol_*` / `flow_*`，避免与既有 `vol_5/10/20`＝**收益波动率**混淆）：
   - RVOL 多窗口（5/20/60）与**成交量 z-score**（对 60 窗口中位数/1.4826·MAD，复用 P3 的 `AnchorMAD` 锚定思路以免回溯改值）；
   - VWAP 偏离（滚动 VWAP 与 session VWAP）及其分位；
   - OBV / A/D 线**斜率**（非水平值）、Chaikin 资金流、MFI(14)；
   - Amihud 非流动性（`|ret| / quote_volume`，滚动均值）；
   - 量价相关（滚动 20 的 `corr(|ret|, Δvolume)`）与"量价配合"（涨且放量、跌且放量）；
   - 成交量重心 / 收盘位置加权的量能比。
3. **契约版本化**：`feature_schema_hash` 升到 v2；**硬规则**：哈希不匹配的模型在
   `MLPredictor.load_model` 与回测预热门（P4-F4 已实现 sidecar 校验）一律拒绝，并加测试。
4. **不做**：不把 P4 的盘口特征并入 39 列契约（它们是流式、无历史回填），仅在 P6-C 里作为独立实验线。

**量化验收标准**：
| 指标 | 阈值 / 判据 | 证据命令 |
|---|---|---|
| 无前视 | 追加未来 500 根 bar，历史特征值 **0 变化**（逐列断言） | 新增 `tests/test_volume_features_v2.py` |
| 非退化 | `near_constant: []`（原 10 个常量特征修复的经验） | `scripts/ml_credibility_measure.py` 证据 JSON |
| 确定性 | 同输入两次运行 **逐位一致** | 同上 |
| 契约版本 | v1 模型被拒（明确原因），v2 通过 | 新增拒绝测试 + sidecar 校验测试 |
| 成本预算 | 特征管线整表耗时 ≤ 现基线（当前 0.88 s / 全量） | 现有 `test_feature_pipeline_cost_is_bounded` |
| 缓存扩展 | `quote_volume` 与 kline 源一致（抽样比对），缺口/孪生仍为 0 | `scripts/check_data_integrity.py` |
| **诚实的门判定** | 用 v2 契约重跑 ML 可信度测量并**如实报告**（预期仍可能被拒；被拒即结论） | `scripts/ml_credibility_measure.py --symbols BTCUSDT ETHUSDT --intervals 1h`（**空格分隔**：两个参数都是 `nargs="*"`，逗号写法会被当成**一个** symbol → `data/market/BTCUSDT,ETHUSDT/1h.parquet` → `FileNotFoundError`） |

**回归风险**：缓存列扩展会改变 `data/market/**` 的字节内容 → 凡是断言"文件哈希/行数"的文档与测试必须改为
"形状 + 测量时间戳"（P6-A 之后已建立该惯例）。

### P6-C 采样与状态：美元棒 / 量钟 + 量能广度

**目标**：检验"用成交量采样替代时间采样"是否改善统计性质与门槛表现；把"市场级量能广度"接入状态门控。

任务：
1. **美元棒（dollar bars）离线实验**：用 1m 缓存重建 2–3 个币种的美元棒与量钟序列，测：
   收益的偏度/峰度/Jarque-Bera、一阶自相关、波动聚集、以及**同一套 39+v2 特征在同一门槛下的表现**。
2. **量能广度**：用 **676 个可用 USDT 交易对**的 24h quote volume（ticker 端点）构造广度指标（总成交额、上涨家数占比、
   成交额集中度 HHI），作为 regime 门控的输入之一。
   > **实测（2026-09-30 13:25，`data-api.binance.vision` 的 `/api/v3/ticker/24hr`，
   > `fetch_tickers` + `parse_tickers` + `aggregate_breadth`）**：端点报告 **705** 个
   > plain USDT 交易对（`pair_count`），其中 **676** 个有非零 24h 成交额
   > （`usable_count`；`coverage = 676/496 = 136.3 %`——本节原先引用的 **496 已过期**，
   > `docs/core-algorithms/15-volume-bars-breadth.md` 的 "676/676" 是同一个可用数）。
   > 可用率 676/705 = **95.9 %**，仍满足下表的 ≥ 95 % 判据。
3. 两者都必须**因果构造**（只用 `t` 之前的数据），并接受与 P4 相同的 off-by-default 纪律。

**量化验收标准**：
| 指标 | 阈值 / 判据 |
|---|---|
| 因果性 | 追加未来数据后历史标签/特征 **0 变化** |
| 分布改善（若宣称） | 美元棒 vs 时间棒：Jarque-Bera 统计量与超额峰度**同时**下降，且给出置信区间/置换检验；否则**如实结论为"未改善"** |
| 广度可用性 | **676 可用 / 705 报告**（2026-09-30 13:25 实测 = 95.9 % ≥ 95 %；计划里旧的 496 偏小 36 %，见上）中有效样本 ≥ 95 %；广度序列与自身滞后值相关 < 0.99（非退化；n = 6 实测 0.9713 / 0.4858 / 0.6608，见 doc 15） |
| 门槛 | 以广度作为门控输入重跑可信度测量，**由门决定**是否启用 |

### P6-D GA 集成：量能基因 + 可执行性进入适应度

**目标**：让 GA 能"进化出"量能条件与成交量感知的仓位/过滤，并把**流动性成本**纳入适应度，避免选出不可执行的策略。

任务：
1. 新增条件模板（进入 `genome.py` 的入场/出场模板与 `INDICATOR_SANITISATION` 映射）：
   RVOL z-score、VWAP 收复/失守、OBV 斜率、MFI 超买超卖、量价背离。
2. 新增**量能过滤基因**（例如"仅当 RVOL > x 才允许入场"）与**成交量感知仓位缩放基因**（受 P6-A 参与率上限约束）。
3. 适应度成本项：把 P6-A 的 `impact_pct` 接入适应度所用的成本模型（默认 `k=0` 时**逐位不变**）。

**量化验收标准**：
| 指标 | 阈值 / 判据 |
|---|---|
| 基因可达 | 固定种子的 GA 运行中，新模板/新基因在种群中被选中 ≥1 次，且冠军可复现（同种子逐位一致） |
| 清洗完整 | 每个新模板都有 `INDICATOR_SANITISATION` 归属，无"孤儿模板"（有测试） |
| 关闭即不变 | `risk.liquidity.enabled=false` / `impact_k=0` 时，全流程逐位一致（复用 P6-A 的对照方法） |
| 可执行性 | 加入冲击项后，冠军的**净收益下降幅度**被如实报告；若冠军因此不再过发布门槛，如实记录 |

**一等指标列（计划评审裁定，2026-09-30 落地）**：`core/strategy/indicators.py` 现在把
P6-B 特征族已有的六条序列以可读名暴露给条件语法——`rvol`/`rvol_z`/`vwap`/`mfi`/
`ad_line`/`obv_slope`（分别等于 `volr_20`/`volz_60`/滚动 VWAP 水平/`flow_mfi_14`/
A/D 线/`obv_slope_10`，**同一实现**，不是重写）。它们是**按需**列：
`compute_all(df, {"volume_flow": {}})` 才计算（归档实测整族 **95.7 ms / 8 844 根**；
2026-09-30 本机复测空闲 96.7 ms，`compute_all` 57.1 → 153.7 ms）；无条件
计算会把 `compute_all` 从 56 ms 抬到 152 ms，即 GA/回测热路径 **≈2.6×（空闲实测
wall 2.61×/2.70×、cpu 3.33×）～ 3×（7 个并发 Python 进程 / CPU 饱和下实测
wall 2.39–2.88×）**——**该比值随负载变化，不是常数**；GA 解码器对任何
**读取这些列的条件**自动打开该键（`core/ga/genome.py::_condition_reads_volume_flow`），
因此"条件引用某列"与"该列被产出"是同一个不变量。P6-D 已有的模板**故意不改**：
它们每一个都是对**另一个估计量**的精确代数展开（模板 VWAP 用 `Σ(close·volume)/Σ(volume)`，
P6-B 的 `vwap` 用典型价 + `1e-12`；模板 RVOL z 是 `volume_ratio` 的滚动 mean/σ，
`rvol_z` 是对数成交量的锚定 median/MAD z；MFI/A-D 亦不同），改写成新列会**改变行为**
而不只是拼写。证据与不变量：`tests/test_volume_flow_indicator_columns.py`（逐值等于
P6-B 列、因果性、按需性、清洗归属、标量 vs 向量化 0 不匹配）。

**交付范围的诚实说明（P6-D 实际交付了什么，2026-09-30 核对代码）**：

- **交付的是"按可执行性重算 P&L 的评分模型"，不是引擎级的定仓基因。**
  `volume_scale_k` 的效果**只发生在适应度评分内部**：`core/ga/fitness.py` 的
  `apply_executability_model`（`:481`）用 `volume_size_factor`（`:307`）算出
  shrink-only 的规模因子，再按该规模重算每笔成交的 notional/cost/pnl 与净值点，
  之后才交给 `stats_from_trades`/`score_stats`。回测引擎**看不到**这个基因：
  `volume_scale_k` 在 `core/ga/**` 之外命中 **0** 次，`apply_executability_model` /
  `volume_size_factor` 在 `core/backtest/**` 命中 **0** 次（命令见下）。
- **原因**：`StrategyConfig` 是线上/回测共用的运行时 schema，只有
  `name/enabled/mode/timeframes/symbols/indicators/entry_conditions/condition_logic/
  exit_conditions/reduce_conditions/ml_config/risk_exit` **12 个字段，没有任何定仓字段**；
  两个量能基因因此存在 `indicators` 的自由字典键 `_ga_volume_genes` 里
  （`core/ga/genome.py:611`，唯一读者是同一个文件的 `_volume_genes_from_conditions`），
  该键在 `compute_all` 的 elif 链里是**惰性**的。要让它进引擎，必须新增
  `StrategyConfig` 字段 + 引擎/`PositionSizer` 消费端——那是 P6-D 写范围之外的改动。
- 因此本阶段的"量能感知仓位缩放"是**评估期**能力（冠军 provenance 里带
  `executability` 摘要），**不是**实盘/回测下单价量。若后续要落地为引擎级定仓，
  必须同时给出默认关闭的逐位一致证明。

证据命令（本仓库实测）：

```powershell
# 基因在 core/ga 之外零命中（交付范围不是引擎级）
Get-ChildItem -Recurse -Include *.py -Path core,app,web,scripts,tools |
  Where-Object { $_.FullName -notmatch '\\core\\ga\\' } |
  Select-String -Pattern 'volume_scale_k' -SimpleMatch      # → 0 行
# 定仓模型只在适应度里出现
(Get-ChildItem -Recurse -Include *.py -Path core/backtest |
  Select-String -Pattern 'apply_executability_model|volume_size_factor' -SimpleMatch |
  Measure-Object).Count                                      # → 0
python -c "from core.strategy.loader import StrategyConfig; print(list(StrategyConfig.model_fields))"
# → 12 个字段，无 sizing 字段
```

### P6-E 审计与发布（与 P1–P5 同一标准）

- 每阶段一次**独立只读审计**，逐项自测并给出 PASS/FAIL + 残余清单；判定"零发现"前不合入下一个阶段。
- 必查：账目恒等式（`scripts/audit_db.py`）、路由基线（`regen_route_baseline.py`，0 增 0 删）、
  全量测试连续两次全绿（含应用改写缓存之后的一次）、默认配置逐位一致、文档数字对照**钉死的数据修订**。
- 结束时：README/证据索引更新、`docs/overhaul/P6_VOLUME_EVIDENCE.md` 建立（每阶段：提交、命令、实测数字、测试钉）。

---

## §4 依赖与顺序

```
P6-A ✅ ──► P6-B（缓存字段 → 特征 v2 → 重跑判门） ──► P6-D（GA 集成，依赖 B 的特征）
                    └──► P6-C（离线采样/广度实验，可与 D 并行）
P6-E 贯穿每个阶段的末尾
```

- P6-B 的**数据层必须先行**：没有 `quote_volume`，B2 的 Amihud 与跨币种 RVOL 只能靠代理，结论会弱一档。
- P6-C 是**实验性**的：结论允许为"未改善"，此时不进入生产契约（但仍产出可复现证据）。
- P6-D 依赖 P6-B 的特征列；若 B 的门判定为"拒绝"，D 仍可只做"量能条件模板 + 可执行性适应度"（不依赖 ML）。

## §5 风险与可证伪声明

| 风险 | 处理 |
|---|---|
| `impact_k` 未标定 | 文档强制标注"示意值"；P6-D 报告净收益下降幅度，不宣称绝对成本准确 |
| 量能在 1h 无方向性价值 | 已被两次实测支持；P6 的价值主张限于**可执行性 / 状态 / 短周期**，并把"是否更好"交给门 |
| 缓存列扩展破坏既有断言 | 统一改为"形状 + 时间戳"；新增列后重跑完整性检查（缺口/孪生） |
| 流式盘口特征无历史 | P4 特征不并入契约；P6-C 只做前瞻采集可行性评估，不做回填声明 |
| 追加基因导致 GA 搜索空间爆炸 | 记录种群规模/代数与命中率；若命中率为 0，如实报告并回退模板集 |

## §6 完成定义（DoD）

1. P6-B/C/D 每阶段的**量化验收标准**逐条有实测数字与可复现命令；
2. 每阶段结束前完成一次**独立只读审计**且残余清空（或明确写为"文档化残余"）；
3. 全量测试连续两次全绿 + `compileall` + 账目恒等式 + 路由无删除；
4. **默认配置行为逐位不变**（关闭路径对照）；
5. 全部提交推送至 `origin/main`，`docs/overhaul/P6_VOLUME_EVIDENCE.md` 汇总每阶段证据；
6. README 与相关 `docs/core-algorithms/` 文档数字对照钉死的数据修订重测。
