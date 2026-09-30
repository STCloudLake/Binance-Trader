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
| **CategoricalGene** | 离散选择 | `mode ∈ {trend, range, scalp, momentum}`、`timeframes` |
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
> `fitness` 两次都是 **16.0846**，只有 `alpha_vs_buy_hold_pct` 从 `0.0` 变成 `−25.0`。
> 因此"beta 不再被计为 alpha"这句话对 **fitness** 不成立，对**发布门**成立——
> 冠军可以因为跑输等权买持而被拒（归档冠军的唯一拒绝原因就是
> `alpha_vs_buy_hold=-52.20% <= 0`）。

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
净盈亏 > 0 **且** PF > 1 **且** DSR > 0 **且**（有验证窗口时）验证 Sharpe > 0。
不满足时**仍然写文件**，但 `enabled: false`，并在 `provenance.rejection_reasons` 里列出原因。
旧实现只要种群非空就发布**启用**的冠军——已发布过 fitness −42.3、0 交易的策略。

## 冠军 YAML 的 `provenance` 溯源块

```yaml
provenance:
  seed: 123456
  window: {train_start: ..., train_end: ..., validation_start: ..., validation_end: ..., key: ...}
  symbols: [BTCUSDT, ETHUSDT]
  timeframes: [1h]
  condition_logic: and     # 入场结构基因：or（默认）/ and
  generations: 8
  n_trials: 1234           # population×generations + 历史试验
  prior_trials: 1114
  fitness_components: {fitness: ..., fitness_base: ..., fitness_alpha: ..., sharpe: ...,
                       deflated_sharpe: ..., max_dd: ..., trade_count: ..., profit_factor: ...,
                       raw_profit_factor: ..., buy_hold_pct: ..., alpha_vs_buy_hold_pct: ...}
  validation: {...}
  published: false
  rejection_reasons: ["trades=12 < 30 (not enough evidence)", ...]
  eval: {engine_mode: legacy, use_live_spread: false}
  written_at: 2026-...
```

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
