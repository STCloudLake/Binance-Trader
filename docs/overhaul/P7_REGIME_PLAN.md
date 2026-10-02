# P7 规划：市场状态敏感度 / 状态条件化（Regime Conditioning）

> 本文件冻结 P7 的目标、阶段划分（S1–S4）、**量化验收标准**、可复现证据要求、回归测试与审计门。
> 与 `ALGO_UPGRADE_PLAN.md`（P1–P5）及 `P6_VOLUME_PLAN.md` 同一标准：每个阶段都必须有实测数字、
> 可复现命令、逐阶段独立只读审计，并以"**是否被自己的门接受**"为准，而不是以"是否看起来有效"为准。
>
> 规划依据的本轮实测状态：`9f86e63`。当前仓库 **1379 passed / 0 failed**（本次会话基线实测，
> `python -m pytest tests/ -q -p no:cacheprovider`，243.91 s）。

---

## §0 为什么做 P7（问题陈述，含已实测的证据）

1. **一次 32 代、5 币种、4 个月训练 + 3 个月样本外的 GA 运行，640 次评估里没有任何一个基因组的适应度为
   正**（最好的 −0.591 出现在第 26 代，随后平台期）；冠军样本外 −0.53 %，而买入持有 +51.36 %，
   是一套 **85 笔交易、逆着强趋势** 的系统。
2. **策略对市场状态零敏感度。** `StrategyConfig` 的 13 个字段里没有任何一个能表达"这个策略只适合
   某种市况"：`mode` 是自由字符串（`core/ga/genome.py` 的 `MODE_OPTIONS = ["trend","range","scalp",
   "momentum"]`），GA 可以改它，但**回测与实盘都不读它**——`mode` 命中 `core/backtest/**` 0 次
   （命令见 §3 S1 证据）。也就是说：一个在趋势里亏钱的均值回归规则，在震荡里继续被同样的规则交易。
3. **检测器已经存在且已被证明是因果的**，只是没有接到"策略能不能在这里交易"这条链路上：
   `core/strategy/regime.py` 的双状态高斯 HMM 有 forward-only 因果解码路径
   （`hmm_two_state_causal`，`NonCausalRegimeError`），合成结构上的**样本外**因果准确率
   **0.758–0.815**（同结构整样本 Viterbi 是 0.9987，但那是在它自己拟合过的制度上打分，
   是拟合诊断而不是可交易数字）；因果解码成本 ≈2.3–2.6 s / 3 000 根。**P7 不以 HMM 为卖点**：
   S1 消费的是更便宜的复合标签（扩张分位波动率三分位 + EMA 趋势），实测 **4.2–8.3 ms / 6 058 根
   / 币种**（本轮实测，见 §3 S1）。
4. **算力账已经能对上**：`core/ga/benchmark.py` 的 `exposure_matched` 已经是本运营商配置的
   **在册门基准**（`config/config.yaml` 的 `ga.benchmark_mode: exposure_matched`），它把篮子
   **只在策略自己的持仓区间内**持有——这正好是"条件化后的样本"需要的同口径比较基准，
   因此 S1 **复用**它，而不新造一个基准（§3 S1 证据）。
5. **操作者假设（P7 要检验的命题）**：策略缺的是**状态敏感度**，所以它们在自己不适合的市况里交易；
   一个上层引擎应当决定"某个状态专用策略现在可不可以跑"。**S1 只负责把"状态条件化"变成一等属性并
   如实测量它值不值**；把答案交给实测，允许答案是"没用"。

## §1 非目标（明确不做 / 不宣称）

- **不宣称条件化一定能改善表现。** 这是 S1 要测的假设，不是前提。S1 的验收标准是"被如实测量"，
  不是"变好"；若测量结论是"没有帮助"，那就是结论，并据此决定 S2–S4 是否继续。
- **不默认开启任何能力。** `ga.regime_conditioning` 默认 `false`，S1 交付的基因在开关关闭时
  **不产生、不读取、不写入**（连一次 RNG 抽取都不多花）；`experimental.regime_gating` 与
  `REGIME_GATING_ENABLED` 保持 `false`。
- **S1 不动实盘链路。** `regime_filter` 的执法点在**回测/GA 入场路径**；实盘 `StrategyEngine`
  的接线与"实盘是否允许它"是 **S3** 的决定，需要自己的门与逐位一致证明。
- **不改 `mode` 的语义。** `mode` 保持"策略类型标签"（现状就是自由字符串），
  状态声明走新字段 `regime_filter`；不把 `mode` 重新定义成状态门。
- **不引入 HMM 作为默认门。** 因果 HMM 的 2.3–2.6 s/3 000 根是研究成本；S1/S2 不使用它。
  任何把 HMM 标签作为门的尝试必须走 `NonCausalRegimeError` 那条拒绝路径，并单独做门。
- 不修改线上数据库/策略文件；不在验证中写入 `data/binance_trader.db` 或 `strategies/`。

## §2 现状清单（规划所依据的代码事实）

| 组件 | 现状 | 位置 |
|---|---|---|
| 复合状态标签 | `trend_up`/`trend_down`/`range_low`/`range_mid`/`range_high`；因果（扩张分位 + EMA） | `core/strategy/regime.py::classify_regimes` |
| 因果 HMM | forward-only 解码 + 定期重拟合；样本外准确率 0.758–0.815；≈2.3–2.6 s/3 000 根 | `core/strategy/regime.py::hmm_two_state_causal` |
| 既有门 | `RegimeGate.allowed` 按**策略种类**映射；`gate_regimes` 在 HMM 非因果时抛 `NonCausalRegimeError` | `core/strategy/regime.py:785-865` |
| 策略 schema | 13 字段（P7 前 12）；`mode: str = "trend"`；**无状态字段** | `core/strategy/loader.py::StrategyConfig` |
| `mode` 的可达性 | `mode` 在 `core/backtest/**` 命中 **0** 次 → 现状不可执法 | 证据命令见 §3 S1 |
| GA 基因 | continuous / categorical(mode,timeframes) / structural / indicator_genes / condition_logic | `core/ga/genome.py` |
| GA 状态开关 | `ga.regime_conditioning`（**本轮新增**，默认 `false`） | `config/config.yaml`、`app/config.py` |
| 门基准 | `exposure_matched`（在册）；`COMPUTED_BENCHMARK_MODES` 需要价格缓存 | `core/ga/benchmark.py` |
| DSR 试验计数 | 账本 + `dsr_trial_counts`（`prior`/`cumulative`），冠军用 `prior` | `core/ga/evolver.py:102-131` |
| 试验账本 | `data/ga_trials.json`（`load_trials`/`record_trials`/`total_trials`） | `core/ga/trial_counter.py` |

> ⚠️ 关键缺口：**没有任何一层把"状态"和"这个基因组可不可以跑"连起来**。
> `mode` 可被 GA 进化但不被引擎读取；`RegimeGate` 需要调用方自己构造且按种类映射；
> GA 的适应度里没有状态项。S1 补的就是这条缝。

---

## §3 阶段划分

### P7-S1 因果状态作为一等策略属性 —— ✅ 已完成（本轮，HEAD `9f86e63` 之上）

**目标**：让策略能声明"我只在哪些因果状态下允许交易"，让 GA 能进化这个声明，
让评估**只在被允许的 bar 上度量**，并**默认关闭、可证逐位一致**；然后**如实测量**它值不值。

任务（逐条已落地）：

1. **新模块 `core/strategy/regime_causal.py`**：状态判定的唯一受检缝。
   - `GATE_REGIME_LABELS`（5 个可门的因果复合标签）、`IN_SAMPLE_LABELS`（`("calm","stressed")`
     ——只有整样本 HMM 会产出的两个标签）。
   - `parse_regime_filter` / `allowed_regimes` / `regime_allows`：`None`/空 ⇒ `[]` ⇒
     **允许一切**（历史行为）；未知标签抛 `UnknownRegimeLabelError`；
     **样本内标签按名字拒绝**（`InSampleRegimeLabelError`，不返回 `False`——拒绝必须可见）。
   - `causal_regime_table(df)`：`classify_regimes(df, with_hmm=False)` 的单用途包装，
     并把 `regime_source="causal_composite"` 写进 `attrs`。
   - `enforce_causal_table(table)`：整样本 HMM 表 ⇒ **复用** `NonCausalRegimeError`；
     来源不明的表 ⇒ `UnknownRegimeSourceError`（"trust me" 不是来源）。
   - `RegimeContext` / `build_regime_context`：`(symbol, interval)` → 因果标签数组 + `searchsorted`
     查询（`t` 时刻只看 ≤ `t`），并保留每键的标签直方图作为"被限制到哪个样本"的证据。
2. **`StrategyConfig.regime_filter: list[str] = []`**（第 13 个字段）：`mode` 语义不变；
   校验在**加载时**发生（pydantic `field_validator`，惰性 import 以保持 `loader` 轻量）。
3. **GA 基因 `regime_filter`**（`CategoricalGene`，值 = 逗号连接，`""` = 无过滤）：
   `regime_gene_options()`、`confine_regime_filter()`、`confine_regime_gene()`；
   在编码/解码/随机初始化/变异/整种群收敛五处一致地生效。
4. **引擎执法**（`core/backtest/engine.py`）：仅当某个基因组的 `regime_filter` 非空时构造
   `RegimeContext`（每个 `(symbol, 主 timeframe)` 一次；实测 **4.2–8.3 ms / 6 058 根**），
   入场前查该 bar 的因果标签并 `regime_allows`；**未测量的 bar 不允许入场**（保守方向）；
   结果写进 `metrics["regime_conditioning"]`（`gated`/`allowed`/`allowed_pct`/`labels`），
   经 `benchmark_result_fields` 进入基因组结果字典（`regime_filter` + `regime_conditioning`）。
5. **开关**：`ga.regime_conditioning`（`config/config.yaml` + `app/config.py`）与
   `GARunConfig.regime_conditioning`（`None` = 跟随 config，**永不被缺省 job 字段打开**）。
   开关关闭时基因组**没有**该基因 ⇒ 连 RNG 流都不变。
6. **基准复用**：不新增基准。`exposure_matched` 只在策略自己的持仓区间内持有篮子，
   条件化把持仓区间限制到被允许的状态后，基准自动同口径（实测：无过滤 `bench=–0.797 %`，
   `trend_up` `bench=–0.706 %`，两次运行的 `buy_hold_pct` 都是 `–21.4446 %`）。
7. **测量工具** `tools/p7_regime_conditioning_measure.py`：固定种子队列 × 6 个条件化档
   （无过滤 + 5 个标签）× 训练窗 + 样本外窗，成对比较；可选 `--ga` 跑两臂 GA。
   结果表见 §3.1。

**量化验收标准**：

| 指标 | 阈值 / 判据 | 证据命令 |
|---|---|---|
| 词汇表封闭 | 可门标签恰为 5 个复合标签；未知标签 ⇒ `UnknownRegimeLabelError`；`range_unknown` 等不可门 | `tests/test_p7_regime_s1.py` |
| **因果唯一** | 样本内标签（`calm`/`stressed`）在 **3 处**（配置加载 / 基因边界 / 每 bar 决策）都被**按名拒绝**；整样本 HMM 表 ⇒ `NonCausalRegimeError`（复用既有错误类） | 同上 + `tests/test_p34_audit_fixes.py` |
| 因果性质 | 追加未来 bar 后，历史标签 **0 变化**（合成三段结构逐 bar 断言） | 同上 |
| 基因往返 | 30 个随机基因组 `encode(decode(x)) == x`；基因值恒在选项集合内 | 同上 |
| 关闭即逐位一致 | 开关关闭 ⇒ 基因组无 `regime_filter` 基因、**无额外 RNG 抽取**、解码过滤恒为 `[]`、两次运行逐笔 `==`；`StrategyConfig` 去掉 `regime_filter` 键后的哈希 = P6 冻结值 `809ddf7ba45af011` | `tests/test_p7_regime_s1.py`、`tests/test_ga_volume_genes.py` |
| **条件化真的限制样本** | 合成序列（构造已知状态）上：被允许 bar 数 == 该标签的 bar 数；交易数严格下降且 > 0；逐笔成交的开仓 bar 标签 == 声明标签 | 同上（引擎实测） |
| DSR / 交易数记账 | 条件化样本的交易数与 `observations` **严格小于**无过滤样本；`dsr_detail.observation_periods` == 条件样本的 `observations`；低于门限 ⇒ `insufficient_data`/`insufficient_trades`；试验数按**实际尝试的变体**计 | 同上 |
| 回归 | 全量 `1379 + 25 = 1404 passed / 0 failed`（连续两次）；`compileall` 退出 0 | 见 §6 |
| **诚实的门判定** | 用真实缓存数据、成对队列与两臂 GA 如实报告假设结论（预期可能是"没帮助"；没帮助即结论） | `python tools/p7_regime_conditioning_measure.py --population 8 [--ga]`；实跑见 §3.1 |

**实跑规模（本轮）**：`python -m pytest tests/ -q -p no:cacheprovider` 基线 **1379 passed / 0 failed**
（243.91 s）→ 实现后 **1404 passed / 0 failed**（两次全绿，见 §6）。

**回归测试**：`tests/test_p7_regime_s1.py`（19 项：基因往返/收敛、因果拒绝、条件化限制、
逐位一致、DSR 记账）；重新钉住 `tests/test_ga_volume_genes.py::test_pre_p6_chromosome_decodes_to_the_frozen_head_config`
（**可证明是加法**：整 dump 哈希 `4d40f95abe61d7e2`；去掉 `regime_filter` 后的子集哈希仍是冻结的
`809ddf7ba45af011`）。

**审计门**：独立只读审计逐条核对上表；必查"关闭即逐位一致"（含 `population_hash` 与 RNG 流）、
"样本内标签在三处都被拒绝"、以及 §3.1 的数字来自实际运行。

**回滚**：`ga.regime_conditioning: false` 是回滚（默认值），此时 S1 的全部新增代码都在**未触发**
分支上；`StrategyConfig.regime_filter` 是一个有默认值的加法字段，无法被任何既有 YAML 触发。
若要把字段本身也撤掉，撤销提交即可（无数据迁移、无外部产物）。

#### §3.1 S1 实测：假设检验（成对队列，真实缓存数据）

命令（本轮实际执行；工具、窗口、种子与队列全部记录在 JSON 产物里）：

```powershell
python tools/p7_regime_conditioning_measure.py --population 8 `
    --symbols BTCUSDT ETHUSDT --timeframe 1h `
    --train-start 2025-11-01 --train-end 2026-02-01 `
    --oos-start 2026-02-01 --oos-end 2026-06-01 `
    --out $env:TEMP\p7_regime_conditioning.json
```

**结论（实测，不是预期）：条件化没有带来风险调整后的 alpha；它带来的是一致、可观的样本缩减，
以及一个**随窗口翻转符号**的 alpha 均值变化。假设的第一部分（策略确实对状态不敏感）部分成立，
第二部分（条件化能救它们）在本测量里不成立。**

| 指标（训练 2025-11-01~2026-02-01；样本外 2026-02-01~2026-06-01；BTC+ETH 1h） | 无过滤（8 个队列） | 条件化（40 个 = 8 队列 × 5 状态） |
|---|---|---|
| 样本外交易数（中位） | **141.5** | **19.5** |
| 样本外 alpha vs `exposure_matched`（中位 / 均值，百分点） | −0.1233 / −0.2384 | **0.0000 / −0.0242** |
| 样本外 alpha > 0 的个数 | 2 / 8 | 13 / 40 |
| 样本外 **DSR > 0 的个数** | **0 / 8** | **0 / 40** |
| 在场时间占比（中位 %） | 24.03 | **2.10** |
| 训练窗交易数 ≥ 30（`ga.min_champion_trades`） | 7 / 8 | 21 / 40 |

成对变化（同一基因组，条件化 − 无过滤，样本外）：

| 状态 | 对数 | Δalpha 中位 | Δalpha 均值 | 变好 / 变差 | Δ交易数 中位 | Δ在场时间 中位 |
|---|---|---|---|---|---|---|
| `trend_up` | 8 | +0.0906 | +0.2292 | 5 / 1 | −105.5 | −17.45 |
| `trend_down` | 8 | 0.0000 | +0.0731 | 3 / 3 | −58.0 | −8.40 |
| `range_low` | 8 | +0.2039 | +0.2875 | 5 / 1 | −120.5 | −19.27 |
| `range_mid` | 8 | +0.1233 | +0.3125 | 4 / 2 | −132.5 | −22.01 |
| `range_high` | 8 | +0.0739 | +0.1686 | 4 / 2 | −139.5 | −23.70 |
| **合计** | **40** | **+0.0060** | **+0.2142** | **21 / 9** | **−118.5** | **−13.13** |

诚实读法（四条，全部来自上表）：

1. **没有风险调整后的优势。** 两臂的样本外 **DSR 全部 ≤ 0**（40 个条件化单元 × 96 次试验的
   去膨胀口径），也就是说：没有任何条件化候选能被"与数据挖掘区分开"。
2. **改善主要是"少交易"的曝光效应，而不是选到了更好的时机。** 中位 Δalpha 只有 **+0.006 个百分点**
   （均值 +0.2142 被少数大幅改善的单元拉动），而交易数中位掉了 **118.5 笔**、在场时间中位掉了
   **13.1 个百分点**；无过滤臂的样本外 alpha 本身就是负的（中位 −0.1233），
   把暴露削掉自然会把 alpha 拉向 0。
3. **代价是把样本压到门槛以下。** 条件化把训练窗交易数中位从 **150.5 压到 33.0**，
   40 个条件化单元里只有 **21 个** 达到 30 笔门槛（无过滤臂 7/8）；48 个单元里
   **13 个样本外零交易**（alpha 恒为 0，被计数而不是被丢弃）。
4. **状态之间没有稳定排序。** `range_low`/`range_mid` 的中位改善最大，`trend_down` 的中位改善为
   **0.0000** 且 3 好 3 坏——与"某些状态适合某些策略"的强版本不一致；
   数据支持的弱版本是"**在所有状态下都少交易一点**"。

这些数字全部来自上面那条命令；原始 JSON 在
`%TEMP%\p7_regime_conditioning.json`，工具是 `tools/p7_regime_conditioning_measure.py`
（固定队列种子 `20261007`，`n_trials_for_dsr = 96`）。

**第二个窗口（两臂 GA，`--ga`）翻转了符号**——这是本阶段最重要的发现：

```powershell
python tools/p7_regime_conditioning_measure.py --population 1 --ga `
    --ga-population 4 --ga-generations 2 --ga-seed 4242 --symbols BTCUSDT `
    --train-start 2025-08-01 --train-end 2025-11-01 `
    --oos-start 2025-11-01 --oos-end 2026-01-01 --out $env:TEMP\p7_ga_arms.json
```

- 关闭臂冠军：训练 fitness −13.9441、56 笔、训练 DSR −0.1451、样本外 **0 笔**、未发布；
- 打开臂冠军：训练 fitness −13.8991、83 笔、训练 DSR 0.0000、基因进化为 `["trend_down"]`、
  样本外 **0 笔**、未发布（`net_pnl<=0`、`profit_factor<=1`、`dsr<=0`、
  `alpha_vs_exposure_matched=−0.32 % <= 0`）；
- 该窗口五个状态 **5/5 变差**：成对 Δalpha 中位 **−0.5217**、均值 −0.4699、
  Δ交易数中位 −54、Δ在场时间中位 **−31.2 个百分点**。

因此 S1 的最终读法是：**"状态条件化能改善样本外表现"没有得到支持**（三个窗口的成对 Δalpha 均值
= **+0.2142 / −0.4699 / +0.2717**，符号随窗口翻转）；**能三处稳定复现的只有两条：
样本外 `dsr > 0` 的单元数为 0（无过滤 11 个单元、条件化 55 个单元里各 0 个），
以及条件化一致大幅削减交易数与在场时间**。**S1 到此为止：没有任何默认开关被打开，
也没有基于这些数字宣称"条件化有效"。**

**这个结论对 S2–S4 的含义**：S1 的交付物（属性、基因、因果执法、开关、测量工具）本身
是**中性的基础设施**——它让"状态敏感度"从口号变成可测的东西，并给出了第一个否证。
S2（搜索空间 + DSR 试验数）仍然必须做，因为**任何后续想宣称条件化有效的人都要付试验数的账**；
S4（组合体样本外）是唯一能把"编排器能不能挑对状态"这个更弱的问题问清楚的地方。
若 S2/S4 也给不出稳定为正的组合体 alpha，P7 的正确处置是**回退到默认关闭并记录否证**，
而不是继续加层。

### P7-S2 状态基因的搜索空间与 DSR 诚实性

**目标**：把"状态门"从"一个标签"扩展到"标签的析取/合取"，并把**每个尝试过的变体**计入 DSR 的
试验数——否则条件化本身就是在做多重检验而不付费。

任务：
1. 基因从 6 个选项（中性 + 5 标签）扩到**真子集**（析取）与可选的**波动率×趋势分离**
   （`trend_regime` × `vol_regime` 两个基因），并把选项集合与 `confine_*` 一起升级为
   "值 = 已排序子集字符串"的规范形式（往返仍逐位一致）。
2. **试验计数**：条件化把"每个基因组"变成"每个基因组 × 每个状态级"的族。
   `core/ga/trial_counter.py` 与 `dsr_trial_counts` 必须把**每一个被评估过的变体**计入；
   若做不到（例如一个 chunk 内不同基因组的状态级不同），就按**尝试过的状态级上限**计，
   并在冠军 provenance 里写明用了哪个口径。
3. 记录"状态基因命中率"（多少基因组带非中性基因、被选中的标签分布），
   命中率 0 时如实报告并回退选项集合（P6-D 的同类纪律）。

**量化验收标准**：

| 指标 | 阈值 / 判据 | 证据命令 |
|---|---|---|
| 往返与规范 | 子集基因值规范有序、往返逐位一致；空集 = 中性 | `tests/test_p7_regime_gene_subsets.py`（新增） |
| DSR 试验数 | 记录的 `n_trials` ≥ 实际评估的变体数（逐 chunk 对账）；冠军 provenance 口径自述 | 新增对账测试 + 实跑 JSON |
| 命中率 | 固定种子 GA 中非中性基因被选中 ≥1 次，且冠军可复现（同种子逐位一致） | `tools/p7_regime_conditioning_measure.py --ga` |
| 搜索空间账 | 选项数与理论空间大小被打印并断言；爆炸时如实报告并缩回单标签 | 同上 |

### P7-S3 上线接线：实盘/活路径的状态门（必须默认关闭）

**目标**：让实盘 `StrategyEngine` 也能消费 `regime_filter`，并把"上层引擎决定哪个状态专用策略可跑"
落成**一个显式的编排对象**——不是散落的 if。

任务：
1. 抽出 `RegimeOrchestrator`（S1 的 `RegimeContext` + 每策略声明 + 一个**固定的**（不许临场学习的）
   决策表），实盘入场前问它；只有 **因果** 标签可用（`enforce_causal_table` 在每次构造时执行）。
2. `experimental.regime_conditioning_live`（**默认 false**）；打开前必须有"关闭路径逐位一致"证明
   （信号缓存 + 下单量 + 路由基线都对照）。
3. 拒绝语义：实盘遇到样本内标签必须**拒绝并告警**，不得退化为"允许"。

**量化验收标准**：

| 指标 | 阈值 / 判据 | 证据命令 |
|---|---|---|
| 关闭即逐位一致 | 开关关闭 ⇒ 信号缓存/下单量与 HEAD 逐位一致（既有 `tests/test_experimental_switches.py` 方法） | 新增开关测试 |
| 因果拒绝 | 非因果标签在实盘缝上抛 `NonCausalRegimeError` | 同上 |
| 编排对象单一 | 决策只出在 `RegimeOrchestrator`（`grep` 证明没有第二个决策点） | 审计命令 |

### P7-S4 组合体（编排器 + 子策略）的样本外评估

**目标**：把"上层引擎 + 状态专用子策略"作为一个**整体**做样本外评估——不是把子策略各自的
样本外数字拼起来讲故事。这是 P7 的定义性阶段：目前 GA 只评单个基因组，条件化后必须能评一个
**组合体**。

任务：
1. 定义组合体的评估契约：
   - **组合体的资金曲线** = 子策略在各自允许状态内的成交按**固定的**资金分配合成
     （固定 = 在训练窗上定好、样本外不许再调）；空闲期现金收益 0（与 `exposure_matched` 同口径）。
   - **对照** = 同一窗口的 `exposure_matched` 篮子与 `buy_hold`，都按**组合体的持仓区间**匹配。
   - 报出的指标：组合体样本外总收益、对 `exposure_matched` 的 alpha、DSR（试验数含所有被试过的
     编排器与子策略变体）、交易数、**在场时间占比**，以及每个状态层的贡献分解。
2. **编排器必须在样本外固定**：`--holdout` 语义——选择编排器只用训练窗（含独立的
   `validation_start`），样本外只做一次评估；任何"看过样本外再调"的迭代都要在证据里
   如实记为**过拟合**。
3. 组合体评估必须能回答**操作者的原问题**：条件化之后的系统，在 3 个月样本外是否比
   买入持有/同口径篮子更接近"可用"（例如 alpha > 0、DSR > 0、交易数过门）。

**量化验收标准**：

| 指标 | 阈值 / 判据 | 证据命令 |
|---|---|---|
| 组合体契约 | 组合体资金曲线可复算：给定子策略成交与固定权重，逐点相等（相对误差 < 1e-9） | `tests/test_p7_composite.py`（新增） |
| 样本外一次性 | 编排器/holdout 划分被记录；样本外被访问次数 == 1（计数器断言） | 同上 + 证据 JSON |
| 同口径对照 | 组合体 alpha 对 `exposure_matched`（按组合体持仓区间匹配）与 `buy_hold` 同时给出 | `tools/p7_composite_oos.py`（新增） |
| 交易数门 | **组合体**交易数 ≥ 100（`.ml.gate_min_trades` 的口径）才允许宣称可用；低于则如实标注"证据不足" | 同上 |
| 结论纪律 | 允许结论为"条件化 + 编排器仍未过门"；此时不启用任何默认开关 | 证据文件 |

---

## §4 依赖与顺序

```
S1（属性 + 基因 + 执法 + 假设测量）✅
   ├──► S2（搜索空间 + DSR 试验数诚实性）
   ├──► S3（实盘接线 + 编排对象，默认关闭）
   └──► S4（组合体样本外评估，S2/S3 的可选组合）
```

- S2 依赖 S1 的基因与 `RegimeContext`；没有 S2 的试验计数，S4 的 DSR 不可信。
- S3 与 S2 可并行，但 S3 **不得先于** S2 的 DSR 账目进入任何"宣称有效"的结论。
- S4 是定义性阶段：S1–S3 的数字都只是**单策略**证据，不能替代组合体样本外。

## §5 风险与可证伪声明

| 风险 | 处理 |
|---|---|
| **状态误判会让事情更糟** | 因果解码的样本外准确率只有 0.758–0.815（相对合成结构），滞后 5–145 bar；S1 的门是"如实测量"，`RegimeContext` 报告每个标签的 bar 直方图，S4 报组合体 alpha；若条件化反而降低样本外 alpha，如实记录并停在这里 |
| **条件化把样本压到门槛以下** | 实测：`trend_up` 只允许 **66/1585 bar（4.16 %）**，`range_low` **40/1635（2.45 %）**（BTC+ETH，2025-11-01~2026-01-01）；合成序列上交易数从 200+ 掉到 20–70。GA 门的交易数下限是 `ga.min_champion_trades = 30`，而 ML 门是 `ml.gate_min_trades = 100`——**两个门的上限不一致**，S1 按前者报"过门数"，S4 才按 100 报组合体；任何低于门限的条件化候选都必须是 `insufficient_data`，不得被当成"有效" |
| **每币种 × 每状态 = 搜索空间爆炸，DSR 试验计数器必须把每个试过的变体算进去** | S1 的选项集合刻意只有 6 个（中性 + 5 标签）；S2 扩到子集时必须同步扩试验计数（`core/ga/trial_counter.py` + `dsr_trial_counts`），否则 DSR 的 N 会系统性偏小、把数据挖掘当 alpha |
| **编排器必须在样本外固定，且必须作为整体被评估** | S4 的 holdout 契约 + 一次性访问计数；训练窗内选编排器，样本外只评一次；组合体（编排器 + 子策略）而非子策略拼贴 |
| 状态标签会漂移（三分位是相对的） | 已在 `regime.py` 文档声明；S1 消费"当前样本内相对"的标签，S4 报标签分布随时间的变化 |
| HMM 成本高（2.3–2.6 s/3 000 根） | S1 不使用 HMM；任何 HMM 门都必须单独做成本/收益门，并且只能走因果路径 |
| 关闭开关仍有行为变化 | S1 的强形式：关闭时**连基因都不创建**（RNG 流不变），且用"去掉新字段后的配置哈希"证明加法性 |

## §6 完成定义（DoD）

1. S1–S4 每阶段的**量化验收标准**逐条有实测数字与可复现命令；每个数字都来自**实际运行过的命令**。
2. S1 已实测：本次会话 **1379 passed / 0 failed** 基线 → 实现后 **1404 passed / 0 failed**（连续两次），
   `python -m compileall -q app core web db scripts tools` 退出 0；每文件 `sha256_16` 前后对照
   （见 `docs/overhaul/P7_REGIME_EVIDENCE.md` §S1）。
3. 每阶段结束前完成一次**独立只读审计**且残余清空（或明确写为"文档化残余"）。
4. **默认配置行为逐位不变**：`ga.regime_conditioning: false` 下，种群、RNG 流、解码配置
   （去掉加法字段后的哈希）与 HEAD 一致。
5. 证据汇总在 `docs/overhaul/P7_REGIME_EVIDENCE.md`：每阶段提交、命令、实测数字、测试钉，
   以及**假设检验的结论**（含"没有帮助"这种结论）。
6. 不写 `data/binance_trader.db`，不写 `strategies/`；不为跑分而改数据或改门。

---

## 状态与偏差（as-built）

> 本节是 P7 四个阶段全部合并后的**追加对账**（写入时 HEAD `f7cd14b`）。§0–§6 的阶段定义**保持冻结、
> 一字未改**；本节只记录每阶段**实际交付**了什么、与规划差在哪里，以及实测出来的每一个负号。
> 所有数字都取自已提交的实测记录（本仓 `P7_REGIME_EVIDENCE.md`、`docs/core-algorithms/06-ga-evolution.md`、
> `docs/research/CORE_ALGORITHMS.md`）；**本节不改门、不改任何数字**。

### 一、逐阶段交付与偏差

| 阶段 | 提交 | 规划（§3） | 实际交付 | 偏差 |
|---|---|---|---|---|
| S1 | `1b33f81` | 因果状态作为一等属性 + 基因 + 引擎执法 + 默认关闭 + 成对测量 | **按规划交付**：`StrategyConfig.regime_filter`（第 13 字段）、`regime_filter` 基因、`core/strategy/regime_causal.py`、`ga.regime_conditioning`（默认 `false`）、`tools/p7_regime_conditioning_measure.py` | 无：默认关闭，且"关闭即逐位一致"可证（见下） |
| S2 | `9c44749` | 状态基因的**子集/析取搜索空间** + 逐变体 DSR 试验计数 | **改为逐币种进化**：job 字段 `symbol_mode: pooled\|per_symbol`（默认 `pooled`，与改前逐字节一致）；per_symbol = 每币一套独立种群、只在自币评分、每币一个冠军 | 交付物**不是** plan 原文的 "regime-gene subsets"；试验计数要求**改落在 per-symbol 的乘数上**：`n_trials = len(symbols) × population × generations`，每个冠军的 DSR 都用**整轮**计数并在 `provenance.trials.search` 自述；`pooled` 路径的算术与 HEAD 逐位一致 |
| S3 | `9c44749` | `experimental.regime_conditioning_live` + **每策略声明表** | `core/ai/orchestrator.py::RegimeOrchestrator`：regime→策略集映射 + 连亏熔断 + 因果波动率门 + 广度门；可重放、带规则指纹 | 开关**改名**为 `experimental.regime_orchestrator_live`；规则不是"每策略声明表"，而是一个**配置驱动的 `regime.allowed` 映射**（外加 `default_action`/`missing_regime_action`/`unknown_label_action`），全部在 `ai.orchestrator` 下 |
| S4 | `f7cd14b` | 组合体样本外评估 + `tools/p7_composite_oos.py` | `core/ai/composite.py`（组合体契约）、`core/ai/holdout.py`（**一次性 holdout 计数器——规划时并不存在**）、`tools/p7_composite_measure.py` | 工具**改名**为 `p7_composite_measure.py`；"一次性访问计数"从规划要求变成**已交付的产物**（`HoldoutRefusal` + `allow_reuse` 显式开关） |

### 二、S1（按规划交付）：假设否证

- 因果状态成为一等属性，默认关闭（`ga.regime_conditioning: false`）；关闭时基因组连该基因都不创建（RNG 流不变）。可加性证明：整 dump 哈希 `4d40f95abe61d7e2`，**去掉 `regime_filter` 后的同一 dump = P6 冻结值 `809ddf7ba45af011`**。
- **假设为负**：条件化把样本外交易数中位 **141.5 → 19.5**、在场时间 **24.0 % → 2.1 %**；无过滤 8 个单元与条件化 40 个单元里，样本外 `dsr > 0` 的都是 **0 个**（0/8、0/40）——**没有任何一臂达到 DSR > 0**。
- 成对 Δalpha 均值在三个窗口上是 **+0.2142 / −0.4699 / +0.2717** 个百分点——**符号随窗口翻转**。三处稳定复现的只有两条：`dsr > 0` 的单元数为 0，以及交易数与在场时间一致大幅下降。

### 三、S2（改为逐币种进化）：假设不支持

- pooled 2 个冠军：alpha 中位 **−0.526 pp** / **147** 笔；per_symbol 4 个冠军：**−0.3054 pp** / **81.5** 笔；样本外 `dsr > 0` 为 **0/2** 与 **0/4**。
- per_symbol 的 DSR 门槛是 pooled 的 **2 倍**（实测试验数 64 vs 32），因此"没有 DSR > 0"这个否定对 per_symbol 是**保守**的。
- 4 对同币成对单元里 **2 对退化**：per-symbol 的第一个臂与 pooled 臂同种子 ⇒ 共享初始种群，两个冠军 YAML 除名字外逐行相同，不提供信息；有信息的 2 对（ETH）**一好一坏**（变差 0.3293 / 变好 0.0276 个百分点）。
- 合成检验证明机制本身有效（信号只在单一币上时 per_symbol 找得到、pooled 找不到）——所以结论是"**真实数据上没有这个 alpha**"，不是机制不工作。

### 四、S3（改名 + 配置化规则）：降回撤也降收益

实测：样本外 2026-02-01~2026-06-01，BTC+ETH 1h，30 个变体，DSR 试验数 30：

| 臂 | 成交 | 收益 % | 最大回撤 % | 在场时间 % | DSR |
|---|---|---|---|---|---|
| always-on | 1407 | −1.76 | 2.5145 | 70.69 | 0.0 |
| orchestrated | 1229 | −2.00 | 2.2615 | 66.74 | 0.0 |

- 被拒成交 178 笔、合计 **+24.30 USDT 净盈利**（熔断 98 笔 / +1.93；状态映射 80 笔 / +22.37；波动率门 0 笔，门槛在这段窗口上从未触发）。
- 编排器**把回撤与在场时间压低了，也把收益压低了**：它拒绝的是净盈利的敞口。与 S1 同一教训——**削曝光是稳的，削对曝光不是**。

### 五、S4（工具改名 + holdout 计数器落地）：不可用

5 个变体（always-on、随机 seed 7/8/9、orchestrated）在样本外 2026-02-01~2026-06-01 上，`N = 133` 次试验：

| 变体 | 成交 | 收益 % | 最大回撤 % | DSR |
|---|---|---|---|---|
| always-on | 1398 | +0.5206 | 0.5583 | −0.232771 |
| 随机 seed 9（最好的对照） | 684 | +0.5033 | 0.4997 | −0.233703 |
| **orchestrated** | 717 | +0.4425 | **0.4238** | **−0.237314** |

- 编排器**没有打败任一对照**：对 always-on **−0.0781 pp**、对最优随机臂（seed 9）**−0.0608 pp**；而三个随机种子自身的跨度（0.3111 ~ 0.5033）比这个差额更大。**0/5 达到 `DSR > 0`**，可用性门 **`usable = false`**。
- 组合体 717 笔成交的**已实现 PnL 只有 −3.10 USDT**（按入场 bar 因果标签分解），所以 +0.4425 % 是**曝光路径**的结果，不是每笔优势。

### 六、两条随交付一并披露的局限

1. **广度数据在本机不存在。** 广度是只能向前记录、无法回填的序列；本机缓存在这些窗口上是空的，因此 S3 的广度门**只跑了缺失分支**（`missing_action: allow`，出货值），广度规则从未被真正触发——见 `tools/p7_orchestrator_measure.py:89-96` 的原注释。任何"广度门有效"的说法目前都没有证据。
2. **S4 这个 holdout 窗口的"新鲜度"已经花掉。** 窗口被打开 3 次：第 1 次是**无效测量**（`--rules_from_mapping` 的 enable 份额按 bar 计数而不是按标签集合计算，导致任何"≥2 个标签"的规则被算成 100 %，三个随机对照与 always-on **逐位相同**——`0.5206 / 0.5583 / DSR −0.232771`），第 2、3 次是在看过样本外数字之后为修 bug 重跑的（第 2、3 次七个头条指标逐位一致）。方向性结论不受影响，但 `p7-s4-composite-oos|2026-02-01|2026-06-01|1h` 应记为**已用**；将来要一次真正干净的一次性评估，必须换一个从未看过的窗口。详见 `P7_REGIME_EVIDENCE.md` §S4.6。
