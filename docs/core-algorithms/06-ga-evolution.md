# 遗传算法策略进化

> **P1 更新（GA 可信性）**：本文档已按 `docs/overhaul/ALGO_UPGRADE_PLAN.md` §二 P1 的落地代码重写。
> 旧版本描述的适应度公式是**验证路径**的公式（文档与批量评分路径不一致），并声称存在一个当时并不存在的 DSR 门控
> ——两者都已纠正。文末「效用分析」中的数字为**示意值（非实测）**，见 §文档一致性。

## 算法原理

遗传算法将策略优化建模为自然选择过程：

```
INIT → EVALUATE(每基因独立仓位槽位 + 独立分账) → SELECT → CROSSOVER(按基因名)
     → MUTATE → ELITE → NEXT GEN → PUBLISH GATE
  ↑____________________________________________________|
```

### 染色体编码

| 基因类型 | 编码内容 | 示例 |
|----------|----------|------|
| **ContinuousGene** | 连续参数（指标周期/阈值） | `rsi_period ∈ [5, 28]`、`ema_fast_period ∈ [5, 30]` |
| **CategoricalGene** | 离散选择 | `mode ∈ {trend, range, scalp, momentum}`、`timeframes`（受 job 的 `timeframe_pool` 限制，见下） |
| **StructuralGene** | 该方向的入场/出场条件列表 | `entry_long = ["rsi < 30", "close > ema_fast"]` |
| **BooleanGene** | 指标启用/禁用标志 | 13 种指标各 on/off |
| **condition_logic**（染色体级） | 入场条件组合方式 | `"or"`（默认，宽松）/ `"and"`（严格） |

**变异算子**
- 连续变异：高斯噪声 `N(0, step × strength)`，保持边界约束
- 分类变异：随机替换为不同的类别值
- 结构变异：50% 概率删除条件 / 50% 概率添加模板条件（条件数 ≥1）
- 布尔变异：15% 概率翻转；`condition_logic` 10% 概率在 or/and 间翻转

**交叉算子**：**按基因名**继承（`_inherit_genes_by_name`）。子代携带双亲基因名的**并集**，
每个名字随机取自 A 或 B。旧的 `zip` 按位置交叉在双亲基因列表长度/顺序不同时会**丢基因或重复基因**
（实测：子代只保留 8 个基因中的 6 个且 `bb_stddev` 重复）。

**选择算子**：锦标赛选择 (k=3)：随机抽取 3 个个体，选适应度最高者。

### 周期白名单（job 字段 `timeframe_pool`）

周期是基因组的一部分，所以不设限的搜索空间会把大部分墙钟时间花在 `1m` 基因上：3 个月 × 3 币的
1m 回测每币 ~390 000 根 bar，是 15m 的 15×、1h 的 60×。任务载荷因此可以带
`timeframe_pool`（如 `["15m","1h","4h"]`），把 `timeframes` 基因限制在操作者真正会交易的周期上：

| 关注点 | 约定 |
|---|---|
| 缺省 / 空 | `None`/字段缺失 = **不限制**，`timeframe_gene_options(None)` 仍是原来的 `combinations(TIMEFRAME_OPTIONS, 2)`，种群与冠军逐位不变 |
| 校验 | 取值必须来自唯一 interval registry `INTERVAL_SPEC`（`core.ga.genome.known_timeframes`）；非法值抛具名 `UnknownTimeframeError`，job **加载即失败**（接口侧同一校验返回 HTTP 400） |
| 约束点 | 初始化 `random_chromosome(..., timeframe_pool=…)`、编码 `strategy_to_chromosome`（连种子策略的取值一起收敛）、变异（基因 `options` 只含池内组合 + `confine_timeframe_gene`）、`resume`/精英（`evolve()` 开头整群收敛）、解码 `chromosome_to_strategy(..., timeframe_pool=…)`（非空且 ⊆ 池） |
| 审计 | 启动日志 `GA timeframe_pool=…`、progress/result 的 `timeframe_pool`、冠军 `provenance.timeframe_pool`（不限制时为 `null`） |

单周期池（如 `["1h"]`）下基因只有一个可选值：`CategoricalGene.mutate` 在"去掉当前值后为空"时
回退到 `options` 本身，因此不会出现 `random.choice([])`。

### 入场结构（`condition_logic`）：评分 = 发布

`condition_logic` 不是"只在评估时有效"的临时开关，而是**策略 schema 的一等字段**
（`core/strategy/loader.py` 的 `StrategyConfig.condition_logic`）：

| 项 | 约定 |
|----|------|
| 取值 | `"or"`（任一条件满足即激活该方向）/ `"and"`（所有条件都必须满足）；大小写不敏感 |
| 默认 | `"or"` —— 历史行为，P1 之前的 YAML、手写 YAML、AI 生成的 YAML 都不带这个键 |
| 非法值 | 记一条 `WARNING` 并**回退到 `"or"`**，绝不因为一个陌生值就加载失败 |
| YAML | 冠军/策略文件里显式写出，例如 `condition_logic: and`；`provenance.condition_logic` 同步记录 |
| 求值 | `StrategyConfig.entry_sides(df)` 是**唯一**入场结构求值器：`"or"` 直接委托共享内核 `evaluate_entry_conditions`，`"and"` 要求每个条件在**该 bar** 上成立。GA 打分与线上交易都走它（`core/strategy/engine.py:246` 线上、`core/backtest/engine.py:1209` 回测/GA 打分），回测侧原有的内联 AND 循环已在 P3/P4 收尾中删除，因此不存在第二份实现可供漂移。空条件列表在两种模式下都不激活，避免"无条件入场"。**残留（不在本阶段写权限内）**：向量化的混合引擎 `core/backtest/signal_matrix.py:285` 仍只做 OR，未读 `condition_logic`；因此 `condition_logic: and` 的冠军在 hybrid 模式下会被按 OR 评估 |

为什么必须是字段：P1 落地时 `StrategyConfig` 还没有这个字段，解码器只能用
`object.__setattr__(config, "condition_logic", ...)` 把基因挂在实例上。于是 GA 打分时 AND
真实生效（`core/backtest/engine.py` 读该属性），但**发布出去的 YAML 里没有这个基因**，
重新加载后只能按默认 OR 交易 —— 一个"评分用 AND、上线用 OR"的分叉（正是 P1 item 6
要消除的"评分 ≠ 发布"）。现在基因走普通字段赋值进入 YAML，线上 `StrategyEngine._evaluate`
与 GA 评估读同一个字段、应用同一条 AND/OR 判定，该分叉不再存在（`object.__setattr__`
已从 `core/ga/**` 全部移除）。

**单一求值器（P3/P4 收尾）**：`core/backtest/engine.py` 曾在此之外保留一份内联 AND 循环
（遍历条件、逐条 `evaluate_condition`），与共享内核构成"一条规则、两份实现"。该循环已删除，
回测入场路径直接调用 `strategy.entry_sides(df_primary)` —— 与线上 `StrategyEngine._evaluate`
**字面同一行调用**。等价性以真实缓存数据（BTCUSDT 1h，AND/OR 各两个基因组）逐条比对：
`tests/test_gap_fixes.py::test_legacy_inline_and_rule_equals_entry_sides_on_real_data`（旧内联
规则与 `entry_sides` 逐 bar 相等）与 `test_backtest_entry_path_calls_the_shared_evaluator`
（引擎确实调用该 helper，且每一笔成交都落在共享规则判定活跃的 bar 上）在改动前后给出
**完全相同的成交条目集合与逐 bar 信号集合**（40/0/0/40 笔；244/300/0/300 个信号 bar）。
注意 hybrid 的向量化路径（`core/backtest/signal_matrix.py`）仍只实现 OR，见上表"残留"一行。

回归测试 `tests/test_condition_logic.py`：

- `test_entry_signal_sets_are_identical_after_yaml_round_trip` —— 同一个 AND 基因组，
  序列化前后逐 bar 的入场信号集合（bar + 方向）必须完全一致；
- `test_and_genome_trades_identically_before_and_after_publishing` —— 走真实 GA/回测引擎，
  发布前后成交条目集合一致；
- `test_helper_entry_rule_matches_the_engine_entry_bars` —— 回测引擎实际成交的每一笔，
  都必须是 `entry_sides` 判定该方向活跃的 bar（helper 与 GA 入场路径无漂移）；
- `test_and_is_stricter_than_or_on_the_same_data` —— 同一份数据上 AND 的信号集合是 OR 的
  **真子集**（证明该基因不是惰性基因）；
- `test_invalid_condition_logic_warns_and_falls_back` —— 非法值告警 + 回退 OR。

## 批量评估：每基因独立仓位与独立分账

`BacktestEngine.run_with_exit_evaluation(..., per_strategy_isolation=True, per_genome_ledger=True)`
是 GA 专用语义（`per_genome_ledger` 默认 false，因此 hybrid/legacy 的既有等价性契约不受影响）：

1. **独立仓位槽位**：`max_positions = max(1, max_open_trades // n_strategies)`，
   并且**只统计本基因自己的持仓**（`_own_positions`），槽位用满时 `break` 当前基因的
   符号循环，而不是 `break` 到 chunk 之外。
   旧实现按**整个 chunk** 计持仓并 `break`，实测 20 基因中 **1 个交易 407 次、19 个 0 次**，
   那个基因单独评估是 1158 笔；适应度被 `trades < 5 → -20` 拉平，三代 best 恒为 −1.30。
2. **独立分账**：每个基因一份 `balance` + `equity_curve`（结果里的 `per_strategy_equity`）。
   旧实现共用一个 `balance`，一个基因的盈亏会污染兄弟基因，且"0 交易"的基因也显示整块的盈亏。
   关闭仓位的现金一律**记到该持仓自己的子账**上（`_close_and_credit`）；否则收盘循环会让
   第 2 个基因把第 1 个基因的整个账户当成自己的起始余额（复现：某基因已实现盈亏 −41.86 USDT，
   余额却是 90 390）。
3. 结果里额外返回 `per_strategy_equity[name]["trades"]`，适应度就用这份**每基因**的
   成交与净值序列算 Sharpe / 最大回撤 / DSR——共享序列是整块之和，无法评价单个基因。

## 适应度函数（唯一公式，所有路径共用 `score_stats`）

```python
# 利润因子：分母加入一个合成平均亏损，并封顶、按证据量缩放
gross_win  = Σ 盈利笔 pnl ; gross_loss = Σ |亏损笔 pnl| ; mean_win = 平均盈利
PF         = gross_win / (gross_loss + mean_win)        # gross_loss==0 时≈盈利笔数
pf_term    = min(PF, 10.0) × min(1, trades / 50)

fitness = win_rate × 0.15            # w["wr"]
        + pf_term × 5.0              # w["pf"]
        + ROC × 50                   # w["roc"]，ROC = pnl / initial_balance（小数）
        - imbalance × 10.0           # w["bal"]，多空失衡
        + alpha × ga.alpha_weight     # 默认 1.0（ALPHA_WEIGHT）
        - trade_penalty - loss_penalty - complexity_penalty

alpha = DSR_deflated_sharpe × √365 × min(1, trades / 30) - max_drawdown_pct
```

> ⚠️ 两处单位修正（审计 D-14/D-16，2026 重测）：
> * `alpha` 里的 DSR 是**每期**口径，代码先乘 `√365` 年化再乘证据量
>   （`core/ga/fitness.py` 的 `dsr_sharpe = dsr × √365`）。上面若漏掉 `√365`，
>   结果会比代码小约 19.1 倍：归档冠军（DSR 0.2119、75 笔、max_dd 0.12 %）按代码
>   = 0.2119 × 19.105 − 0.12 = **3.928**，按漏掉的公式 = **0.092**。
> * `ga.alpha_weight` 是**活配置**：`config.yaml` → `Config.ga_alpha_weight` →
>   `evolver` → `score_stats(alpha_weight=...)`，默认 1.0 与旧字面量逐位相同。

- **交易次数惩罚**：`<5 → −20`；`<15 → −5`；`>500 → −(trades−500)×0.02`
- **亏损惩罚**：`pnl < −50 → −|pnl|×0.3`
- **复杂度惩罚**：`n_conditions×0.8 + n_indicators×1.2 + n_params×0.3`
- **`flag`**：`no_trades` / `insufficient_trades`（<30 笔）/ 空 ——「0 交易」被**显式标注**而不是静默给低分

权重来自 `data/ga_fitness_weights.json`（`core/ga/fitness_calibrate.py`），缺失时用
`DEFAULT_WEIGHTS = {wr:0.15, pf:5.0, roc:50, bal:10.0}`。

**为什么改 PF**：旧代码 `gross_loss == 0 → PF = 100`，×5 权重最高 500 分，
而其余正当项合计 <60。实测：**5 笔全胜 → 490.85；200 笔 / 60% / PF 2.0 / 净赚 1500 → 7.30**，
于是"不交易"或者"5 笔全胜"成为最优解。现在同样两组数字为 **5.00 vs 22.04**（见
`tests/test_ga_credibility.py::test_five_winning_trades_cannot_outrank_two_hundred_trades`）。

**为什么加 alpha**：旧的两个批量路径把 `sharpe` / `max_dd` 写死为 0，
纯漂移的合成数据 Sharpe 13.7 也被当成 alpha。现在用**每基因净值序列**算
DSR 去偏 Sharpe × 证据量 − 最大回撤。

> ⚠️ **买持基准不进 fitness**（审计 D-16，已按代码修正本文三处说法）。
> `alpha_vs_buy_hold_pct = total_return_pct − metrics["buy_hold_pct"]`
> （`core/ga/fitness.py` 的 `score_stats`）只被**赋值并上报**，随后只被
> **发布门**消费（`core/ga/evolver.py` 的 `_publication_decision`：`alpha_vs_buy_hold > 0`）
> 与 provenance 记录。fitness 的和只有 `base + alpha × w − complexity`。
> 实测（同一份 stats，`buy_hold_pct` 由 `None` 改为 `25.0`）：
> `fitness` 两次都是 **−29.99**，只有 `alpha_vs_buy_hold_pct` 从 `0.0` 变成 **`5.0`**。
> 这组数字是
> `tests/test_ga_credibility.py::test_buy_and_hold_is_reported_and_gates_but_does_not_rescore`
> 里那份 stats（`stats_from_trades([{"pnl": 0.0, "side": "long"}], _equity_for([0.0]), 10000.0)`
> 且 `total_return_pct = 30.0`、`buy_hold_pct ∈ {None, 0.0, 10.0, −50.0, 25.0}`、
> `n_trials = 100`）实测得到的：五次调用 `fitness` 全为 **−29.99**（`fitness_base` 同值），
> 只有 `alpha_vs_buy_hold_pct` 随基准变化。**审计发现 7(b)**：此前本文、
> `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md`（只读，属 Lead）与
> `docs/research/CORE_ALGORITHMS.md` 就同一条不变性给了三个互不相同的实测值
> （16.0846 / 22.9878 / 29.1535）——那是三份**不同的 stats**，不是同一个实验的三次测量。
> 本文与 `CORE_ALGORITHMS.md` 现统一引用上面这份可复现 stats 的实测值；证据文档的那一行
> 留给 Lead 处理。
> 因此"beta 不再被计为 alpha"这句话对 **fitness** 不成立，对**发布门**成立——
> 冠军可以因为跑输等权买持而被拒（归档冠军的唯一拒绝原因就是
> `alpha_vs_buy_hold=-52.20% <= 0`）。
> **`ga.benchmark_mode`（下下节）可以选择门消费哪个基准**：`buy_hold` 保持上面这条公式与
> 数值**逐位不变**；`exposure_matched` / `risk_matched` 改消费 `alpha_vs_benchmark_pct`，
> `none` 跳过这条判据。这条不变性只对 **fitness** 成立，与基准选择无关。

## 统计显著性（DSR）与发布门槛

DSR 的完整单位约定见 `07-deflated-sharpe-ratio.md`。要点：

- 用**每期** Sharpe 与**每期**门槛比较（`E[max] = sqrt(1/T)·sqrt(2 ln N)`），
  `T` 是**真实**期数（不是写死的 365）。
- `N = population × generations + 历史试验数`。历史试验数来自
  `data/ga_trials.json`（`core/ga/trial_counter.py`），每代评估后累加——只数当前一次运行
  会严重低估多重检验负担（生产日志里有 ~24 个 walk-forward 任务）。
  > ⚠️ 修复前**两个 N 并存**（审计 D-18）：代内每个基因的 DSR 用
  > `population + prior`，而冠军/provenance 用 `population × generations + prior`，
  > 于是被上报的冠军 DSR 与它自称试过的策略数不是同一个数。现在两条路径都走
  > `core/ga/evolver.py::dsr_trial_counts`：第 g 代被评分的 N = 之前**已实际执行**的
  > 试验数（`total_trials(ledger)`，下限为 `prior + population×(g−1)`）+ 本代
  > `population`；冠军用同一个函数的 `prior`（此时所有代都已执行完，再加一个
  > population 会重复计最后一**代**）。`tests/test_ga_dsr_trial_counts.py` 逐代钉住
  > 传入的 `prior_trials` 与 provenance 的 `n_trials`。
- **`DSR <= 0` 禁止发布**（`evolver._publication_decision`）。

发布门槛（`core/ga/evolver.py`）：`trades ≥ 30`（`ga.min_champion_trades`）**且**
净盈亏 > 0 **且** PF > 1 **且** DSR > 0 **且**（有验证窗口时）验证 Sharpe > 0 **且**
基准 alpha > 0（见下节；`none` 时跳过这一条）。不满足时**仍然写文件**，但 `enabled: false`，
并在 `provenance.rejection_reasons` 里列出原因。
旧实现只要种群非空就发布**启用**的冠军——已发布过 fitness −42.3、0 交易的策略。

## 发布门基准：`ga.benchmark_mode`（`core/ga/benchmark.py`）

**为什么要有这个开关（实测）**：满仓买入持有基准（`metrics["buy_hold_pct"]`）与
**只在部分时间持仓**的策略按**总收益**比，没有做敞口或风险匹配。归档冠军
`strategies/ga_champion_1790844776.yaml`（窗口 2026-06-01~2026-09-01）实测
`buy_hold_pct = 17.739`、`alpha_vs_buy_hold_pct = −12.1206`，而同一冠军 OOS 窗口
Sharpe **5.6903**、最大回撤 **0.03 %**——它输给的是一个 100 % 在场的基准。
（诚实说明：该冠军同时还因 `validation_dsr = −0.3855 ≤ 0` 被拒，换基准**不能**让它变成可发布。）

四种取值：

| 取值 | 基准定义 | 门 |
|---|---|---|
| `buy_hold`（**代码缺省**） | 同窗口同币种等权满仓买入持有（`engine.metrics["buy_hold_pct"]`） | 消费 `alpha_vs_buy_hold_pct`（与旧版逐位一致） |
| `exposure_matched`（`config/config.yaml` 已启用） | 同一篮子**只在策略持仓期间**持有 | 消费 `alpha_vs_benchmark_pct` |
| `risk_matched` | 满仓基准按策略已实现日波动率缩放 | 同上 |
| `none` | 无 | **只**跳过基准判据 |

**`exposure_matched` 的精确定义**（也是唯一推荐默认）：

1. 对篮子里每个币，从**策略自己的成交**重建持仓区间 `[opened_at, closed_at]`，裁到窗口
   `[t0, t1]`；窗口结束时仍未平仓的按 `t1` 裁剪；重叠区间取**并集**（不重复计敞口）。
2. 该币的基准收益 `Rₛ` = 在**这些区间的并集**上买入持有的收益：每个区间取区间内首/末收盘价，
   区间收益**复合** `Π(1+rᵢ) − 1`（区间之间视为现金，0 %）。区间内 bar 少于 2 根 ⇒ 该区间不计；
   该币所有区间都不可用 ⇒ 从篮子剔除。
3. 权重 `wₛ = mean(amount_usdt) / initial_balance`（裁剪到 `[0,1]`）——**策略实际投入的保证金占比**，
   即"策略自己的相对敞口"，由成交直接测得；因此空仓部分在两个账号里都赚 0。
   成交不带可用名义金额时退回 `1/len(symbols)` 并记 `weighting: equal_share_of_basket`。
4. `benchmark_pct = Σₛ wₛ·Rₛ × 100`。**零成交** ⇒ 基准 `0.0`、alpha = 策略收益
   （门仍以 `no_trades`、DSR、净期望拒绝）；**无可用 bar** ⇒ `benchmark_pct = None`，
   门**跳过**该判据并在 provenance 记 `benchmark_available: false`。

**`risk_matched`**：`buy_hold_pct × (σ_strategy / σ_benchmark)`，两条 σ 都是既有
`daily_returns` / `per_period_sharpe`（`core.ga.fitness` / `core.backtest.metrics`）口径的
**日**收益标准差；`σ_benchmark = 0` 时缩放无定义 ⇒ 用原基准并记 `risk_scale_fallback: true`。
反向结果 `strategy_return × σ_benchmark/σ_strategy` 一并上报。

**校验与缺省**：`app/config.py` 在**配置加载**时用 `parse_benchmark_mode` 校验，未知取值抛
`UnknownBenchmarkModeError`；job 字段 `benchmark_mode` 在 `scripts/ga_worker.py` 的
**job 加载**阶段同样校验（与 `timeframe_pool` 同型）。字段不存在 ⇒ 跟 `config.ga_benchmark_mode`
（不是 job 级覆盖），键不存在 ⇒ 代码缺省 `buy_hold`。`GARunConfig.benchmark_mode = None`
表示"跟配置"，一路透传到 `engine.run_with_exit_evaluation(benchmark_mode=...)`。

**只上报、不门控**：策略 vs 基准 Sharpe、信息比率（日超额 `mean/std × √365`）、
Jensen 式 alpha/beta（OLS，年度化）、扣费后每笔净边际（`trade["pnl"]` 已扣 `cost`）、
在场时间占比。`buy_hold` / `none` 两种模式**不读行情**（直接复用旧值），所以缺省路径零额外 I/O。

**逐位一致性**：`buy_hold`（以及键不存在）时 `alpha_vs_benchmark_pct` 与
`alpha_vs_buy_hold_pct` 是**同一个浮点数**（同样的运算），门结论与拒绝原因字符串逐字相同
（`alpha_vs_buy_hold=... <= 0 (no edge over buy & hold)`）。
`tests/test_ga_benchmark_mode.py::test_buy_hold_is_byte_identical_to_the_head_worktree`
在 `git worktree` 里检出改动前版本、同一 harness 跑两棵树，把**评分值、门结论、旧 provenance 键**
按字节比对。

## 评估/执行一致性：冠军只交易它被评估过的币

冠军 YAML 过去写 `symbols: []`（= 交易自选列表 `system_config.watchlist_symbols` 里的**全部**币），
而 GA 只评估了 job 的篮子（实测：3 币 vs 自选 5 币）——启用冠军会交易**从未被评估过的币**。
现在 `core/ga/evolver.py::evolve` 在写冠军前把**被评估的篮子**写进 `champion_config.symbols`，
而执行路径本来就尊重它（`core/strategy/engine.py`：`_on_kline` / `evaluate_all_now` 在
`strategy.symbols` 非空时跳过表外币，启动时打
`Strategy '<name>' restricted to symbols: [...]`）。选"限制交易"而不是"拒绝启用不匹配的冠军"：
限制是更安全的一侧，且旧 YAML 的 `symbols: []` 继续表示"不限"，向后兼容。
`tests/test_ga_benchmark_mode.py::test_enabling_a_champion_can_only_trade_its_recorded_basket`
用真实 `StrategyEngine.evaluate_all_now()` 钉住这一点。

## 冠军 YAML 的 `provenance` 溯源块

```yaml
provenance:
  seed: 123456
  window: {train_start: ..., train_end: ..., validation_start: ..., validation_end: ..., key: ...}
  symbols: [BTCUSDT, ETHUSDT]
  timeframes: [1h]
  timeframe_pool: [15m, 1h, 4h]   # job 的周期白名单（null = 不限制）
  condition_logic: and     # 入场结构基因：or（默认）/ and
  generations: 8
  n_trials: 1234           # population×generations + 历史试验
  prior_trials: 1114
  fitness_components: {fitness: ..., fitness_base: ..., fitness_alpha: ..., sharpe: ...,
                       deflated_sharpe: ..., max_dd: ..., trade_count: ..., profit_factor: ...,
                       raw_profit_factor: ..., buy_hold_pct: ..., alpha_vs_buy_hold_pct: ...,
                       benchmark_mode: exposure_matched,      # 门消费的模式（新增）
                       benchmark_pct: ..., alpha_vs_benchmark_pct: ...}   # 新增
  benchmark:               # 新增：基准溯源（只有 mode 的 alpha 门控，其余只上报）
    mode: exposure_matched
    buy_hold_pct: 17.739           # 满仓买入持有（历史字段）
    benchmark_pct: ...             # 匹配后的基准收益
    alpha_vs_benchmark_pct: ...    # 门消费的 alpha
    alpha_vs_buy_hold_pct: ...
    benchmark_sharpe: ...          # 基准自身的风险指标（此前完全不记录）
    benchmark_max_dd_pct: ...
    benchmark_time_in_market_pct: ...      # 基准在场时间占比
    strategy_time_in_market_pct: ...       # 策略在场时间占比
    information_ratio: ...         # 只上报
    jensen_alpha_annual_pct: ...   # 只上报
    benchmark_beta: ...
    net_edge_per_trade: ...        # 扣费后每笔净边际（USDT / %）
    net_edge_per_trade_pct: ...
    strategy_risk_matched_pct: ... # risk_matched 的反方向
    report: {...}                  # core.ga.benchmark 的完整报告（权重、notes、可用性）
  validation: {...}
  published: false
  rejection_reasons: ["trades=12 < 30 (not enough evidence)", ...]
  eval: {engine_mode: legacy, use_live_spread: false}
  written_at: 2026-...
```

冠军的 `symbols:` 现在写**被评估的篮子**（不是 `[]`）——见"评估/执行一致性"一节。

`StrategyConfig`（线上/回测共用的 schema）不带运行元数据，所以该块写在 YAML 文档里，
旧策略没有这个块也能正常加载。

## 可复现性

- 任务载荷带 `seed`（`web/routes/ga.py` 未提供时生成随机种子），
  `scripts/ga_worker.py` 用它 `random.seed` / `np.random.seed`；
  多进程路径再按 chunk 派生 `seed + chunk_index`。
- 同 seed + 同数据 → 相同种群、相同冠军（`tests/test_ga_credibility.py`）。
- **不用今天的盘口给历史成交定价**：GA/回测评估传 `use_live_spread=False`，
  未在 `backtest.cost_model.spread_pct` 里映射的币种直接用 `default_spread_pct`，
  并在 `metrics["spread_sources"]` / `provenance.eval` 里记录来源。
  （旧代码会在评估时对未映射币种查询实时深度，等于让 2025 年的回测依赖今天的行情与网络。）
- 检查点里保存 `window_key`，`resume` 标志真正转发到 `evolve(resume=True)`。

## 检查点保留与续跑（`keep_checkpoint` / `resume`）

检查点路径固定为 `<data_dir>/data/ga_checkpoint.pkl`（`GAStrategyEvolver._checkpoint_path`）。
每一代结束时写一次（`_save_checkpoint`）。**旧行为**：干净完成时 `clear_checkpoint()`
无条件删除它，于是"完成的任务"没有任何东西可以续跑，`resume=True` 只对崩溃/手动停止有效。

现在：

1. **保留规则**。`GARunConfig.keep_checkpoint`（缺省 `True`）决定**干净完成**后是否保留。
   优先级：job 字段 `keep_checkpoint` → `config.ga.keep_checkpoint`（`config/config.yaml` 出货
   `true`，带注释）→ 代码缺省 `True`；即**字段缺失 = 保留**，只有显式 `false` 才回到
   "完成即删除"。`stop()`/崩溃的运行两条路径都保留（原有的崩溃续跑语义不变）。worker 与
   evolver 都打日志：`checkpoint kept at <path> (generation g)` / `checkpoint cleared at <path>`，
   并把决定写进 result 与冠军 `provenance.keep_checkpoint` / `provenance.checkpoint.kept`。
2. **续跑是"接着跑"，不是"重跑"**。旧代码
   `for gen in range(cfg.generations): self._generation = gen + 1` **忽略**检查点里的代数：
   载入第 12 代的种群后从"第 1 代"重新编号并再评估 `generations` 代。现在循环起点是
   `_first_gen = int(self._generation or 0)`（新任务为 0），
   `for gen in range(_first_gen, cfg.generations)`：`generations: 32` 从第 12 代续跑只评估
   13..32 代，`history` 为 1..32，`result["generations"] == 32`，
   `result["resumed_from_generation"] == 12`。检查点代数已 ≥ job 的 `generations` 时循环为空，
   只发警告并把检查点里的冠军按原样走发布门。
3. **DSR 试错计数跨续跑连续**。`_save_checkpoint` 保存 `prior_trials`（本次运行开始时
   `ga_trials.json` 的账）与 `trials_this_run`（本次运行已评估的代数×种群）；`load_checkpoint`
   把二者之和放进 `_checkpoint_meta["trials_total"]`，`evolve()` 用它给 `_prior_trials` 取
   **下限**（`max`，只升不降）——所以即使台账被清掉，续跑段的第 1 代仍按"已经试过 N 次"去膨胀
   DSR，而不是从本次 population 重新计数；台账完好时 `max` 也不会重复计数。冠军的 `n_trials`
   仍是 `prior_trials + trials_this_run`（D-18 的同一个公式），并写进
   `provenance.trials{prior_trials,trials_this_run,resumed_trials,n_trials}`。
4. **窗口/身份守卫**。检查点记录 `window_key`、`population_hash`、`symbols`、`timeframe_pool`。
   续跑请求的窗口与检查点不一致时抛**具名** `CheckpointWindowMismatchError`（不吞进
   "没有检查点"的分支；worker 的 result 里 `error_type` 同名），拒绝在另一个窗口上静默续跑。
   空 `window_key` 的旧检查点不阻拦续跑，且不会覆盖本次运行的窗口。续跑后的冠军 provenance
   记录 `resumed_from_generation` / `resumed_population_hash` / `resumed_trials` /
   `resumed_symbols` / `resumed_timeframe_pool`，即"从什么继续"。
   `WalkForwardRunner` 只对本次 WF 运行的第一窗口转发 `resume`，并且把跨窗口的检查点拒绝
   降级为"该窗口的 GA 从零开始"（WF 自己的续跑状态是 `ga_wf_state.json`），不会因为保留下来
   的上一个窗口的检查点而让整个 walk-forward 失败。
5. **可见性**。`scripts/ga_job_status.py` 打印检查点行（路径、`exists`、**generation**、
   **mtime**、窗口、population hash、试错数、`resume` 会从第 g+1 代继续），`--checkpoint-file`
   可覆盖路径，`--json` 里同样有 `checkpoint` 块；GA 面板有 **Keep checkpoint** 勾选框
   （默认勾选，job 字段同名），完成后状态行显示检查点代数并显示 Resume 按钮
   （`_ga_state.resumable` 由 result 的 `checkpoint.kept` 驱动——此前该字段从未被赋值）。

## 多进程评估失败降级

`evaluate_population_multiprocess` 的某个 chunk 崩溃时，先用
`_single_worker_retry` 在本进程内**以 `max_workers=1` 重跑该 chunk**，并记录
`paths_used`（`process` / `retry_single_worker` / `failed`）。只有重试也失败才标 −999。
旧实现一个 chunk 崩掉就把**整个种群**标成 −999（生产日志：每代 8 个 chunk 全部 abrupt）。

## 杠杆口径（已选择）

GA 评估是**现金模型**：评估的就是将来会被检验的那份策略，1× 口径下的净值/回撤可直接与
实盘信号对比。实盘 2–4× 杠杆（`config/risk_params.yaml`）会等比放大实盘盈亏与回撤，这个
口径差**写在冠军 provenance 里**，而不是隐藏。

> ⚠️ **没有 `ga.evaluation_leverage` 这个键**（审计：死配置开关）。它曾在
> `config/config.yaml` 里承诺一个"现金模型 / 杠杆评估"的选择，但 `leverage` 在
> `core/ga/**` 与 `core/backtest/engine.py` 里命中 **0** 次——现金模型是引擎的**唯一**行为，
> 不是某个开关选出来的，因此该键已从 `config.yaml` 与 `app/config.py` 删除
> （`tests/test_final_audit_fixes.py::test_every_ga_config_key_has_a_production_reader` 守住）。
> 要把杠杆纳入评分，需要同时给引擎加杠杆输入并改仓位管理与风控口径，属于另一阶段。

## 文档一致性

- 本文档与 `core/ga/{genome,evolver,fitness,walkforward,trial_counter}.py`、
  `core/strategy/{loader,engine}.py`（`condition_logic` 字段与 `entry_sides` 求值）、
  `core/backtest/engine.py` 对齐。
- **`08-market-regime-ml.md` 中 58.7% 的表格属于 P2/P3（ML 预测）阶段，不在 P1 范围内**，
  其数字与本文档无关，需由该阶段负责标注"示意/非实测"或删除。
- 下面「效用分析」的表格是**示意值（非实测）**，保留仅为说明指标维度，不应作为验收依据。

## 相关研究

1. **Holland (1975)**: "Adaptation in Natural and Artificial Systems" — GA 奠基
2. **Goldberg (1989)**: "Genetic Algorithms in Search, Optimization and Machine Learning"
3. **Prado (2020)**: "Advances in Financial ML" — Chapter 17 (GA), Chapter 14 (回测统计)
4. **Deb et al. (2002)**: "NSGA-II" — 多目标优化（同时优化 Sharpe + Stability + Simplicity）

## 改进策略

### 1. CMA-ES (协方差矩阵自适应进化策略)
在连续参数优化上远优于标准 GA。收敛速度快 3-10×。可对 ContinuousGene 使用 CMA-ES，对其他基因类型保留标准 GA（混合算法）。

### 2. Multi-Objective Optimization (NSGA-II)
同时优化多个目标（高 Sharpe + 低回撤 + 低复杂度），输出 Pareto 前沿而非单一冠军。

### 3. Bayesian Optimization
用 Gaussian Process 代理模型替代随机变异，适用训练成本高（深度 ML 模型）的场景。
