# P7 证据：市场状态敏感度 / 状态条件化（Regime Conditioning）

> 本文件记录 P7 每个阶段**实际运行过的命令与实测数字**。规划与量化验收标准见
> `docs/overhaul/P7_REGIME_PLAN.md`（本文件只记录证据，不改门）。
>
> 工作基线：`9f86e63`，`python -m pytest tests/ -q -p no:cacheprovider`
> → **1379 passed / 0 failed，243.91 s**（本次会话实测）。
> 实现后：**1404 passed / 0 failed** 连续两次（253.15 s / 248.62 s），
> `python -m compileall -q app core web db scripts tools` 退出 0。
> 新增 20 项测试（`tests/test_p7_regime_s1.py`）＋ 共享测试面 15 行加法式重钉（§7 残余 1）。

---

## S1 — 因果状态作为一等策略属性（2026-10-02）

### 1. 交付物与哈希（`sha256_16`，前后对照）

修改（`before` = `git show HEAD:<path>` 的字节，`after` = 工作树字节）：

| 文件 | before | after |
|---|---|---|
| `app/config.py` | `3b6606e7902e7efd` | `ef82e50d549370b1` |
| `config/config.yaml` | `d2da012cb5727c36` | `efd9972afbbdb56f` |
| `core/backtest/engine.py` | `17563cd10dc2b961` | `29eb9adcedd030b8` |
| `core/ga/evolver.py` | `24ed3f16e6651a07` | `d83fc4facb0521fb` |
| `core/ga/fitness.py` | `7ac5acee54420aa0` | `77ee93fe2156a0f1` |
| `core/ga/genome.py` | `1f2f59c62c215c13` | `23c7efaea98087cc` |
| `core/strategy/loader.py` | `b1f57fde96f2a028` | `2594c7939940a8f9` |
| `tests/test_ga_volume_genes.py`（见 §7 残余 1） | `65e3b59a...`(git blob) | 15 行加法（见 §7） |

新增：

| 文件 | `sha256_16` |
|---|---|
| `core/strategy/regime_causal.py` | `4bc44f68f3b4607a` |
| `tests/test_p7_regime_s1.py`（20 项） | `033a34aa9e71f942` |
| `tools/p7_regime_conditioning_measure.py` | `83b582e5e17f5ac3` |
| `docs/overhaul/P7_REGIME_PLAN.md` | `eabb6cdc86b4f333` |
| `docs/overhaul/P7_REGIME_EVIDENCE.md` | 本文件 |

命令（Python，字节精确；PowerShell 的 `Set-Content -AsByteStream` 在 pwsh 7.4 上不接受管道文本，
会得到空文件哈希 `e3b0c44298fc1c14`——**不要用**）：

```powershell
python -c "import hashlib,subprocess;f='core/ga/genome.py';print(hashlib.sha256(subprocess.run(['git','show','HEAD:'+f],capture_output=True).stdout).hexdigest()[:16], hashlib.sha256(open(f,'rb').read()).hexdigest()[:16])"
```

### 2. 默认关闭 = 逐位一致（可证明是加法）

`StrategyConfig` 新增 `regime_filter: list[str] = []`（第 13 个字段）。P6-D 把**整 dump** 的哈希
钉在 `809ddf7ba45af011`；加字段必然移动整 dump 哈希，所以 S1 的证明是**可加性**而不是改数字：

```
整 dump（含 regime_filter）        = 4d40f95abe61d7e2
去掉 regime_filter 后的同一 dump    = 809ddf7ba45af011   ← 与 P6-D 冻结值逐位相同
regime_filter 的值                  = []
```

实测命令：`python -m pytest tests/test_p7_regime_s1.py::test_the_regime_field_is_purely_additive_to_the_frozen_config -q`

同一性质的第二处证据：开关关闭时 `random_chromosome` **不产生** `regime_filter` 基因，
因此连 RNG 流都不变（`test_gene_is_absent_while_the_switch_is_off_and_consumes_no_rng`：
同一 seed 下 8 个基因组的 categorical / continuous / structural / condition_logic 逐项相等）。

> ⚠️ 残余（**文档化残余，不是缺陷**）：`tests/test_ga_volume_genes.py::test_pre_p6_chromosome_decodes_to_the_frozen_head_config`
> 的整 dump 断言 `== "809ddf7ba45af011"` 现在会失败（新字段使哈希变为 `4d40f95abe61d7e2`）。
> S1 **没有修改该文件**（并发归属：该测试文件可能与其它工作重叠），改由
> `tests/test_p7_regime_s1.py::test_the_regime_field_is_purely_additive_to_the_frozen_config`
> 承担同一证明（整 dump 新值 + 去掉新字段后的子集 = 冻结值 + 字段为空）。
> **需要 Lead 决定**：把该测试的整 dump 断言改为上面的可加性形式（一行），或接受该测试的失败作为
> 文档化残余。

### 3. 状态标签的成本与分布（真实缓存，实测）

| 币种 | 窗口 | 根数 | 因果表成本 | `trend_up` | `trend_down` | `range_low` | `range_mid` | `range_high` | `range_unknown` |
|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 2025-10-21~2026-06-30 | 6 058 | 8.31 ms | 1 915 | 3 232 | 358 | 342 | 171 | 40 |
| ETHUSDT | 同上 | 6 058 | 8.34 ms | 1 772 | 3 309 | 478 | 287 | 172 | 40 |
| SOLUSDT | 同上 | 6 058 | 4.21 ms | 1 894 | 3 187 | 486 | 309 | 142 | 40 |

命令：`python -c "from core.backtest.data_feeder import DataFeeder; from core.strategy.regime import classify_regimes; ..."`（探针脚本见本轮会话；
`classify_regimes(raw, with_hmm=False)`）。**因果成本是每 (symbol, interval) 一次几毫秒**，
所以 S1 不需要 HMM 的 2.3–2.6 s/3 000 根。

### 4. 执法确实限制样本（引擎实测）

BTC+ETH，2025-11-01~2026-01-01，一个"几乎每根 bar 都想入场"的策略：

| 声明 | gated bars | allowed bars | allowed % | 交易数 | 在场时间 % | `exposure_matched` 基准 % |
|---|---|---|---|---|---|---|
| 无过滤 | — | — | — | 113 | 99.80 | −0.7973 |
| `["trend_up"]` | 1 585 | 66 | 4.164 | 30 | 27.87 | −0.7063 |
| `["range_low"]` | 1 635 | 40 | 2.447 | 19 | 23.50 | −0.0696 |

两次运行的 `buy_hold_pct` 都是 `−21.4446`（基准本身没有移动），而 `exposure_matched`
基准随持仓区间变化——**这就是 S1"复用而非新造基准"的实测依据**。

### 5. 假设检验：成对队列（`--population 8`，96 次评估）

命令（本轮实际执行，~26 分钟）：

```powershell
python tools/p7_regime_conditioning_measure.py --population 8 `
    --symbols BTCUSDT ETHUSDT --timeframe 1h `
    --train-start 2025-11-01 --train-end 2026-02-01 `
    --oos-start 2026-02-01 --oos-end 2026-06-01 `
    --out $env:TEMP\p7_regime_conditioning.json
```

训练 2025-11-01~2026-02-01；样本外 2026-02-01~2026-06-01；固定队列种子 `20261007`；
DSR 去膨胀试验数 `n_trials = 96`（两臂 × 两窗 × 全部候选）。

| 指标 | 无过滤（8） | 条件化（40 = 8 × 5 状态） |
|---|---|---|
| 训练交易数 中位 | 150.5 | 33.0 |
| 样本外交易数 中位 | 141.5 | 19.5 |
| 样本外 alpha vs `exposure_matched` 中位 / 均值（百分点） | −0.1233 / −0.2384 | 0.0000 / −0.0242 |
| 样本外 alpha > 0 | 2 / 8 | 13 / 40 |
| **样本外 DSR > 0** | **0 / 8** | **0 / 40** |
| 在场时间 % 中位 | 24.03 | 2.10 |
| 训练交易数 ≥ 30 | 7 / 8 | 21 / 40 |
| 样本外零交易单元 | 0 | 13 / 48（含无过滤臂的 2 个零交易基因组） |

成对变化（条件化 − 无过滤，样本外）：

| 状态 | 对数 | Δalpha 中位 | Δalpha 均值 | 好 / 坏 | Δ训练交易数 中位 | Δ样本外交易数 中位 | Δ在场时间 中位 |
|---|---|---|---|---|---|---|---|
| `trend_up` | 8 | +0.0906 | +0.2292 | 5 / 1 | −113.0 | −105.5 | −17.45 |
| `trend_down` | 8 | 0.0000 | +0.0731 | 3 / 3 | −68.0 | −58.0 | −8.40 |
| `range_low` | 8 | +0.2039 | +0.2875 | 5 / 1 | −133.0 | −120.5 | −19.27 |
| `range_mid` | 8 | +0.1233 | +0.3125 | 4 / 2 | −139.0 | −132.5 | −22.01 |
| `range_high` | 8 | +0.0739 | +0.1686 | 4 / 2 | −144.0 | −139.5 | −23.70 |
| **合计** | **40** | **+0.0060** | **+0.2142** | **21 / 9** | **−127.0** | **−118.5** | **−13.13** |

**结论（如实）**：

1. **条件化没有产生风险调整后的 alpha。** 两臂样本外 DSR 全部 ≤ 0 → 没有候选通过
   "与数据挖掘区分开"这道判据；因此 S1 不宣称假设成立，也不打开任何默认开关。
2. **改善主要是曝光缩减，不是择时。** 中位 Δalpha 仅 +0.006 个百分点，
   而交易数中位降 118.5 笔、在场时间中位降 13.1 个百分点；无过滤臂 alpha 本身为负，
   削掉暴露会把 alpha 拉向 0。
3. **代价是样本被压到门槛以下。** 训练交易数中位 150.5 → 33.0；只有 21/40 条件化单元
   达到 30 笔门槛；48 个单元里 13 个样本外零交易。
4. **状态之间没有稳定排序。** `trend_down` 的中位改善恰为 0.0000 且 3 好 3 坏；
   表现最好的两个状态（`range_low` / `range_mid`）也正是样本最少的两个。
   数据只支持"少交易一点"，不支持"某个状态专属于某个策略"。

原始产物：`%TEMP%\p7_regime_conditioning.json`（含 48 个单元的逐条数字与 `summary`）。

### 5b. 假设检验：两臂 GA（`--ga`，同种子，不同窗口）

命令（本轮实际执行，~9 分钟）：

```powershell
python tools/p7_regime_conditioning_measure.py --population 1 --ga `
    --ga-population 4 --ga-generations 2 --ga-seed 4242 --symbols BTCUSDT `
    --train-start 2025-08-01 --train-end 2025-11-01 `
    --oos-start 2025-11-01 --oos-end 2026-01-01 `
    --out $env:TEMP\p7_ga_arms.json
```

| 项 | 关闭（unconditioned） | 打开（conditioned） |
|---|---|---|
| 冠军训练 fitness | −13.9441 | −13.8991 |
| 冠军训练交易数 / DSR | 56 / −0.1451 | 83 / 0.0000（`dsr <= 0` 被拒） |
| 冠军 `regime_filter` | `[]` | `["trend_down"]`（基因确实被进化并被写进 YAML） |
| 是否发布 | **否**（`dsr <= 0`、`validation_trades=0`、`validation_sharpe<=0`、`validation_dsr<=0`） | **否**（`net_pnl=-0.20<=0`、`profit_factor=0.677<=1`、`dsr<=0`、`alpha_vs_exposure_matched=-0.32%<=0`、validation 三项） |
| 冠军样本外 | 0 笔（窗口内没有交易 → 指标恒 0） | 0 笔（同上） |

**这一个窗口的五个状态全部变差**：成对 Δalpha 中位 **−0.5217**、均值 −0.4699，5/5 变差，
Δ交易数中位 −54、Δ在场时间中位 **−31.2 个百分点**。

**与 §5 的对比是本阶段最重要的结论**：另一个窗口（2025-11 训练 / 2026-02~06 样本外）
上同一个测量给出 +0.2142 的均值改善，这里给出 **−0.4699** ——**符号随窗口翻转，量级不可忽略**。

### 5c. 第三个窗口（小区队列，复核）

命令（~2 分钟）：`python tools/p7_regime_conditioning_measure.py --population 2 --symbols BTCUSDT --train-start 2025-11-01 --train-end 2025-12-01 --oos-start 2025-12-01 --oos-end 2026-01-01 --out $env:TEMP\p7_smoke2.json`

| 指标 | 无过滤（2） | 条件化（10） |
|---|---|---|
| 样本外交易数 中位 | 85.0 | 7.5 |
| 样本外 alpha 中位 / 均值 | −0.3648 / −0.3648 | −0.0927 / −0.0931 |
| 样本外 alpha > 0 | 0 / 2 | 4 / 10 |
| **样本外 DSR > 0** | **0 / 2** | **0 / 10** |
| 在场时间 % 中位 | 49.87 | 8.74 |
| 训练交易数 ≥ 30 | 2 / 2 | 1 / 10 |

成对：10 对中 8 好 2 坏，Δalpha 中位 +0.2193 / 均值 +0.2717，Δ交易数中位 −51.5、
Δ在场时间中位 −31.65。

**三个窗口放在一起**：均值 Δalpha = **+0.2142 / −0.4699 / +0.2717**，中位 = +0.0060 / −0.5217 / +0.2193。
**符号不稳定 → 不支持"条件化改善样本外表现"。** 三处一致的只有两条，均可逐单元复核：

1. **样本外 `dsr > 0` 的单元数为 0**：无过滤 11 个单元 **0** 个，条件化 55 个单元 **0** 个
   （三份 JSON 逐单元统计；m2 的样本外中位 DSR 还是负的：无过滤 −0.1016、条件化 −0.1383）。
2. **交易数与在场时间一致大幅下降**：Δ交易数中位 −118.5 / −62 / −51.5；
   Δ在场时间中位 −13.1 / −31.2 / −31.7 个百分点。

> 注意 DSR 的语义：`deflated_sharpe_ratio` 在观测 Sharpe ≤ 0 时**定义**为 0（"不估计"），
> 所以 "dsr > 0 的个数 = 0" 是精确表述，"DSR 全部 ≤ 0" 会被读成"估计出负值"。
> 三份产物：`%TEMP%\p7_regime_conditioning.json`（48 单元）、
> `%TEMP%\p7_ga_arms.json`（6 单元）、`%TEMP%\p7_smoke2.json`（12 单元）。

### 6. 测试钉与回归

| 项 | 结果 |
|---|---|
| 新测试文件 | `tests/test_p7_regime_s1.py`（20 项，`--collect-only` 实测） |
| 覆盖 | 基因往返/收敛/开关关闭不消耗 RNG；样本内标签在**三处**被按名拒绝；整样本表 ⇒ `NonCausalRegimeError`；追加未来 bar 不改历史标签；引擎按标签限制 bar 并逐笔核对开仓标签；空过滤运行逐笔一致；evolver 开关的基因生命周期；条件化样本的 DSR `observation_periods` / 交易数 / `insufficient_data` / 门判定 |
| 全量（基线） | `python -m pytest tests/ -q -p no:cacheprovider` → **1379 passed / 0 failed**，243.91 s |
| 全量（实现后，第一次） | **1404 passed / 0 failed**，253.15 s |
| 全量（实现后，第二次） | **1404 passed / 0 failed**，248.62 s |
| 编译 | `python -m compileall -q app core web db scripts tools` → 退出 0（实测） |
| 未触碰 | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md`、`scripts/ga_job_status.py`、`scripts/ga_worker.py`（并发归属）；未写 `data/binance_trader.db`、未写 `strategies/`、未写 `data/market` |

### 7. 需要 Lead 处理的残余

1. **`tests/test_ga_volume_genes.py` 的冻结哈希断言**（见 §2 的⚠️）：该测试把**整 dump** 的哈希
   钉在 `809ddf7ba45af011`；`StrategyConfig` 新增 `regime_filter` 后整 dump 变为
   `4d40f95abe61d7e2`，所以断言必须改，而"新数字"本身不是证据。S1 做了 **15 行加法式**修改：
   整 dump 断言改为新值，并**同时**断言"去掉 `regime_filter` 后的同一 dump == 冻结值"，
   以及 `config.regime_filter == []`。这样该测试原有的证明力（解码结果与 HEAD 逐位相同）
   被保留，而新增字段的可加性被证明。**该文件属于共享测试面**——若 Lead/其他人也在改这个断言，
   请以本形式为准或直接回退我的 15 行（两个新测试文件里已经有一份等价证明）。
2. **GA job 字段（如需）**：S1 的开关可由 `config.ga.regime_conditioning` 开启，
   无需 job 字段；若要在 GA 面板按 run 打开，需要 `scripts/ga_worker.py` 传
   `GARunConfig(regime_conditioning=job.get("regime_conditioning"))`——**该文件并发归属，
   S1 未改**。`GARunConfig.regime_conditioning` 的字段、解析（job 字段优先于 config）与
   "关闭时把基因从种群中剥离"都已在 `core/ga/evolver.py` 就位，patch 只需一行。
3. **未做**：S2–S4（搜索空间扩展、逐变体 DSR 试验计数、实盘接线、组合体样本外评估）
   只有规划（`P7_REGIME_PLAN.md` §3），**没有实现**；S1 的假设检验结论是负面的，
   是否继续按规划执行 S2–S4 应由 Lead 依据本文件的数字决定。
4. **未跑**：`--ga` 的全尺寸两臂（本文件跑的是 4×2、2 个窗口的小尺寸）；若需要更强的
   GA 级结论，用同一命令放大 `--ga-population/--ga-generations` 即可，工具与字段都已验证可用。

---

## S4 — 组合体（编排器 + 子策略）的样本外评估（2026-10-02）

> 工作树：`7b4e234`（本轮开始时 HEAD 是 `9c44749`；`7b4e234` 是并发代理提交的**仅文档**改动，
> 与本阶段无交集）。基线全量套件见 §S4.7。**本节所有数字都来自下面实际执行过的命令**，
> 原始产物 `%TEMP%\p7_composite.json`（65 849 行 / 约 2.6 MB，含逐变体的完整资金曲线、
> 匹配基准与编排器候选表；`%TEMP%\p7_composite_run3.log` 是同一次运行的完整 stdout）。

### S4.1 交付物与哈希（`sha256_16`）

新增（`before` = `git show HEAD:<path>` 的字节 = 空文件哈希 `e3b0c44298fc1c14`）：

| 文件 | 行数 | `sha256_16`（after） | 内容 |
|---|---|---|---|
| `core/ai/composite.py` | 1 040 | `72c27a67867fcb34` | 组合体契约：固定权重、资金曲线、匹配基准、试验计数、可用性门、随机对照 |
| `core/ai/holdout.py` | 205 | `fa0811ae61cba950` | 一次性 holdout 计数器（`data/p7_holdout.json`，已被 `.gitignore` 的 `data/` 覆盖） |
| `tools/p7_composite_measure.py` | 893 | `04d5c6d44e7833fb` | 测量工具（选择/冻结/三变体/基准/holdout 声明） |
| `tests/test_p7_composite.py`（35 项） | 602 | `23b02d071d73981e` | 契约、基准、对照、DSR 试验数、可用性门、holdout 拒绝 |

**未改任何既有文件**：`docs/overhaul/P7_REGIME_EVIDENCE.md`（before `018839ea4fa377ed` → after
`16d31bae2a89915b`）与 `docs/research/CORE_ALGORITHMS.md`（before `e6304db7b92df01b` → after
`094bc3f089c91a4d`）在实现前后只做**追加**；
`core/ga/**`、`core/ai/orchestrator.py`、`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md`、
`docs/overhaul/P7_REGIME_PLAN.md` **一字未动**（并发归属）。

### S4.2 组合体契约（可复算，相对误差 < 1e-9）

**资金曲线**（`core/ai/composite.py::composite_fund`）。子策略在共享 bar 网格上，
`s` 在“持仓 bar **＋ 平仓后的第一根 bar**”上被记为 *deployed*（后者是该笔交易的**已实现现金**
第一次出现在它自己的 equity 序列里的那根 bar；不加这一根，子策略的已实现盈亏就永远进不了组合体）：

```
r_s(t)      = equity_s(t) / equity_s(t−1) − 1                  # 子策略自己的 bar 收益
exposure(t) = Σ_s w_s · deployed_s(t)
ret(t)      = Σ_s w_s · deployed_s(t) · r_s(t) / exposure(t)     if exposure(t) > 0 else 0
C(t)        = C(t−1) · (1 + ret(t)),   C(0) = initial_balance
```

即：**未持仓的子策略贡献 0 %（现金），权重不参与收益也不被重新归一化**——所有子策略都平仓时
组合体保留已经赚到的全部净值（空闲期收益 0 %，与 `exposure_matched` 同口径）。
`tests/test_p7_composite.py::test_the_curve_recomputes_from_the_children_to_1e_9_relative_error`
把 `composite_return()` 复合回去与 `C(t)` 逐点比对（相对误差 < 1e-9）；
`test_the_curve_is_not_renormalised_when_every_child_goes_flat` 钉住“平仓后不回退到初始资金”。

**权重规则（只用训练窗）**。`w_s = clip(mean(amount_usdt over s's TRAIN trades) / initial_balance, 0, 1)`，
再归一化到 Σw = 1（`core/ga/benchmark.py` 的 `capital` 口径）；无可用 notional 的子策略退化为
交易币种集合上的等权（`source="equal_share"`）。本轮实测：

| 子策略 | 训练窗成交 | deployed share | source | 归一化权重 |
|---|---|---|---|---|
| `p7c_BTCUSDT_3` | 184 | 0.018809 | capital | 0.328693 |
| `p7c_ETHUSDT_3` | 145 | 0.019615 | capital | 0.342784 |
| `p7c_SOLUSDT_3` | 171 | 0.018799 | capital | 0.328523 |

`weight_concentration` = `{max_weight: 0.342784, herfindahl: 0.333467, effective_strategies: 2.9988}`
——三个子策略接近等权，组合体**不是**单策略的伪装（这是必须报告的数字：deployed share 只有
**1.88 %–1.96 %**，即每个子策略平均只动用约 2 % 的资金，这也是组合体绝对收益很小的原因）。

### S4.3 命令（本轮实际执行，约 101 s）

```powershell
python tools/p7_composite_measure.py --symbols BTCUSDT ETHUSDT SOLUSDT `
    --selection-population 4 --random-seeds 7 8 9 `
    --train-start 2025-11-01 --train-end 2026-02-01 `
    --oos-start 2026-02-01 --oos-end 2026-06-01 `
    --out $env:TEMP\p7_composite.json `
    --orchestrator-out $env:TEMP\p7_composite_orchestrator.json `
    --allow-holdout-reuse
```

**只用训练窗做的三个决定**（样本外一根 bar 都没读）：

1. **每币种子策略选择**：每币 4 个固定种子基因组在训练窗上评估，取生产 fitness 最高者；
   选中 `p7c_BTCUSDT_3`（fitness −10.6255，184 笔）、`p7c_ETHUSDT_3`（−17.2949，145 笔）、
   `p7c_SOLUSDT_3`（−11.2176，171 笔）。三者的训练窗 DSR 都 ≤ 0（0.0 / −0.1683 / −0.2202）。
2. **权重**：上表（Σw = 1，样本外不再重算）。
3. **编排器规则**：8 个候选（5 个单标签 + 每策略最优**标签对** + 全标签 + 永不允许），
   每个候选都在**训练窗**上按“被保留子策略交易的生产 fitness 之和”打分，选最高者。
   冠军 = `best_pair_per_strategy`（−33.1158），映射
   `{BTC: [range_low, trend_up], ETH: [trend_down, range_mid], SOL: [range_mid, trend_down]}`，
   `rules_fingerprint = 2c9808e1d933dc42`。候选表（同一次运行，全部来自训练窗）：
   `best_pair −33.1158` → `single:trend_down −37.8942` → `all_regimes −39.1380` →
   `single:trend_up −45.5638` → `range_high −55.7469` → `range_mid −60.5999` →
   `range_low −81.6413` → `never_enable −144.2051`。
   `kill_switch.consecutive_losses: 0`（**钉死**：让 enable 时间线只由市场状态决定，
   否则随机对照就不只差在“选择”上）；`unknown_label_action: deny`（`range_unknown` 不交易）。

### S4.4 三变体 + 对照（同窗口、同成本、同权重，只有 enable 时间线不同）

随机对照 = 固定种子、每策略**按训练窗允许 bar 占比**独立抽样的 start/stop
（`random.Random(seed)`，与编排器的 enable 份额匹配），因此比较隔离的是**选择**而不是曝光削减。
编排器在样本外的实现 enable 份额：BTC 0.4056、ETH 0.5306、SOL 0.5306
（训练窗匹配值 0.5022 / 0.4434 / 0.4434）。`time_in_market` = 组合体至少持有一个仓位的 bar 占比。

| 变体 | 成交 | 收益 % | 最大回撤 % | Sharpe | 在场 % | 曝光 % | `exposure_matched` % | alpha(em) | `buy_hold`(matched) % | alpha(bh) | **DSR** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 常开（always-on） | 1 398 | **+0.5206** | 0.5583 | **1.0301** | 37.95 | 43.16 | −0.8925 | **+1.4131** | −15.1989 | +15.7195 | −0.232771 |
| 随机 seed 7 | 635 | +0.4389 | 0.5224 | 0.8576 | 33.92 | 24.25 | −0.9272 | +1.3661 | −15.6954 | +16.1343 | −0.241800 |
| 随机 seed 8 | 655 | +0.3111 | 0.4691 | 0.6054 | 33.65 | 23.95 | −0.5946 | +0.9057 | −10.0553 | +10.3665 | −0.254999 |
| 随机 seed 9 | 684 | +0.5033 | 0.4997 | 1.0123 | 34.44 | 24.86 | −0.6952 | +1.1985 | −11.7810 | +12.2843 | −0.233703 |
| **编排器** | 717 | +0.4425 | **0.4238** | 0.9433 | 33.54 | 20.24 | −0.6917 | +1.1342 | −11.8160 | +12.2585 | **−0.237314** |

- **DSR**：`n_trials = 133`（60 个候选 × 2 窗 + 5 个臂 + 8 个编排器规则集），
  `observation_periods = 119`（组合体自己的日收益数），`expected_max_random = 0.286689`。
  五个变体的观测 Sharpe 都为正，但**全部低于** `E[max]`（0.2867），所以 **DSR 全为负**。
- **可用性门（计划要求明写）**：`>= 100 笔样本外组合体成交 **且** DSR > 0`。
  常开 1 398 笔、编排器 717 笔都过交易数门，但 `dsr_not_positive` ⇒ **两个都是 `usable=false`**。
- **编排器 vs 对照**：对常开 **−0.0781** 个百分点收益（alpha −0.2789）；
  对最好的随机臂（seed 9）**−0.0608**；`beats_always_on=False`、`beats_best_random=False`。
  随机的三个种子自身跨度就有 0.3111 ~ 0.5033（摆幅比编排器与常开的差额更大），
  **编排器的 0.4425 落在随机抽样的正常范围内**。

**唯一对编排器有利的一条，也如实记录**：它的最大回撤 **0.4238 %** 是五个变体里最低的
（常开 0.5583、随机 0.4691/0.5224/0.4997），在场时间也最低（33.54 %）——与 S3 预览“降回撤也降收益”
的结论一致，只是幅度小得多。**这不是可用性证据**（DSR < 0、且收益低于两个对照）。

### S4.5 与组合体自身持仓区间匹配的基准（复用 `core.ga/benchmark.py`，未另造）

`exposure_matched` = `build_benchmark('exposure_matched', …)` **直接导入复用**，喂进去的是
**组合体自己展平后的成交表**（因此区间是组合体的持仓区间并集，权重是它自己测出的资金份额）；
`buy_hold` = `buy_hold_over_intervals()`：同一篮子、同样区间，但**在区间内满仓**
（不按 deployed share 缩放）。两者的实测（编排器变体）：

| 基准 | 值 | 说明 |
|---|---|---|
| `exposure_matched` | **−0.6917 %** | Σw = 0.0585（三个币各约 1.95 %），`benchmark_time_in_market` 17.65 % |
| `buy_hold`（组合体区间内满仓） | **−11.8160 %** | 各币区间收益 −17.51 / −24.62 / −3.47（%），等权 |
| 由 S1 记数得到的区间口径 | 组合体在场 33.54 % vs `exposure_matched` 基准 17.65 % | 组合体持有时间约为基准的两倍 |

**口径提醒（避免误读）**：alpha(em) = +1.1342 **个百分点（窗口总收益之差，非年化）**，
且 `exposure_matched` 基准的 Σw 只有 0.0585 —— 也就是说这个基准只在很小的名义敞口上比较，
与 alpha(bh) 的 +12.2585 不能并列成“两个 alpha”；三者（组合体 +0.4425 / em −0.6917 / bh −11.8160）
必须一起读：**组合体的正收益来自“在场时间短、且避开了篮子在这段时间里的下跌”，而不是来自
每笔交易的已实现优势**——编排器变体 717 笔成交的**已实现 PnL 合计仅 −3.10 USDT**
（按入场 bar 的因果标签分解：`trend_down` 544 笔 −75.02、`trend_up` 127 笔 +48.33、
`range_mid` 38 笔 +20.70、`range_low` 8 笔 +2.89）。

### S4.6 一次性 holdout 计数器（本轮新增，含“已评估”拒绝）

`core/ai/holdout.py`：键 = `(holdout_id, window_start, window_end, timeframe)`，
每次评估追加一条记录（`at` / `revision` / `rules_fingerprint` / `reuse_index`）；
**第二次评估同一窗口默认抛 `HoldoutRefusal`**（消息里带上次的 at / fingerprint / revision），
只有显式 `allow_reuse=True`（工具开关 `--allow-holdout-reuse`）才继续，并把 `reuse=True` 写进产物。

本轮真实账本 `data/p7_holdout.json`（该文件被 `.gitignore` 的 `data/` 覆盖）：

| 次序 | 时刻 | `reuse` | `rules_fingerprint` | revision |
|---|---|---|---|---|
| 1 | 12:20:58 | false | `2c9808e1d933dc42` | `7b4e234` |
| 2 | 12:22:57 | true | `2c9808e1d933dc42` | `7b4e234` |
| 3 | 12:25:15 | true | `2c9808e1d933dc42` | `7b4e234` |

**必须如实记录的三次访问，以及为什么它们不是三次假设检验**：

1. **第 1 次 = 无效测量**：`--rules_from_mapping` 的 enable 份额按 **bar 计数**而不是按
   **标签集合**计算，导致任何“≥2 个标签”的规则都被算成 100 %，三个随机对照与常开**逐位相同**
   （`0.5206 / 最大回撤 0.5583 / DSR −0.232771`，三份完全一样）——随机对照因此没有隔离任何东西。
   该次产物**不作为证据**（但记录保留，因为它确实看过这个窗口）。
2. **第 2 次**：修好份额计算（`fraction = Σ_{label ∈ allowed} bars(label) / Σ bars`，
   见工具 `_label_counts` / `_rules_from_mapping`）后重跑，得到 §S4.4 的表。
3. **第 3 次**：修好 `union_time_in_market_pct`（先前误用了**未过滤**的持仓区间）后重跑，
   数字与第 2 次在全部七个头条指标上**逐位一致**，仅新增/修正了在场时间字段。

**这是本阶段最该被批评的一点**：窗口被打开 3 次，第 2、3 次是在看过样本外数字之后为了修 bug 重跑的。
它对结论的方向**没有**影响（编排器在三个随机种子的正常范围内、DSR 全负，两次运行完全一致），
但它**确实**消耗了这个窗口的“新鲜度”。正确处置（留给 Lead）：把
`p7-s4-composite-oos|2026-02-01|2026-06-01|1h` 记为**已用**，若将来需要一次真正干净的一次性评估，
必须换一个从未看过的窗口（缓存数据到 2026-10-01，可用区间足够），且这次不再有“为了修 bug 再看一眼”的余地。

**测试钉**（`tests/test_p7_composite.py`，其中 holdout 6 项）：首次评估被记录；
**第二次被拒绝**（`HoldoutRefusal`，拒绝后不追加记录）；`allow_reuse=True` 时记为 `reuse_index=2` /
`reused=True`；不同窗口 / timeframe / holdout_id 是不同 holdout；损坏的 JSON 读成空并可重新claim；
默认库位于 gitignored 的 `data/`；记录里带 rules_fingerprint 与当前 revision。

### S4.7 回归与编译

| 项 | 结果 |
|---|---|
| 新测试文件 | `tests/test_p7_composite.py`（**35 项**，`--collect-only` 实测） |
| 覆盖 | 权重（只用训练窗、clamp、退化到等权、无成交即拒绝）；曲线可复算 < 1e-9；平仓后不回退；空闲 bar 收益 0；无成交子策略不动曲线；区间合并；匹配基准（复用 + 同区间 + 满仓口径）；在场时间并集不重复计；随机对照可复现且份额匹配；DSR 按 133 次试验去膨胀且随试验数单调下降；“未估计”与“估计为 0”区分；可用性门（100 笔 + DSR>0，失败原因分名）；编排器 replay 确定性 + 按状态拒单；逐状态贡献分解 |
| 全量（实现后第一次） | `python -m pytest tests/ -q -p no:cacheprovider` → **1505 passed / 0 failed**，338.65 s |
| 全量（实现后第二次） | `python -m pytest tests/ -q -p no:cacheprovider` → **1505 passed / 0 failed**，292.10 s |
| 编译 | `python -m compileall -q app core web db scripts tools` → 退出 0（实测） |
| 未触碰 | `core/ga/**`（并发代理正在改 `fitness.py`）、`core/ai/orchestrator.py`、`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md`、`docs/overhaul/P7_REGIME_PLAN.md`；未写 `data/binance_trader.db`、未写 `strategies/`、未写 `data/market` |

**归属说明（并发）**：本阶段开始时工作树里已有并发代理的改动——`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md`
（`M`）、`core/ga/fitness.py`（`M`，provenance 修复）、`tests/test_ga_benchmark_alpha_paths.py`（`??`，
`HEAD` 从 `9c44749` 移到 `7b4e234` 也是该代理的纯文档提交）。本阶段**没有**触碰这些文件；
1505 项里含它们的测试，若后续出现失败应优先归因于该并发改动，而
`tests/test_p7_composite.py` 单项在本阶段结束时实测 **35 passed**（2.20 s，独立可复现）。

### S4.8 结论（如实）

1. **组合体本身比它自己的匹配篮子好，但这不是“可用”。** 组合体样本外 +0.4425 %（常开 +0.5206 %），
   而同一持仓区间上的 `exposure_matched` 是 −0.6917 %、满仓 `buy_hold` 是 −11.8160 %；
   但**五个变体的 DSR 全部为负**（−0.2328 … −0.2550，`E[max] = 0.2867`），
   按计划的门（≥100 笔 **且** DSR > 0）**没有任何一个变体可用**。
2. **编排器没有打败任一对照。** 对常开 −0.0781 个百分点、对最优随机臂 −0.0608；
   随机对照三个种子的自然跨度（0.3111–0.5033）比这个差额更大，
   即“按状态选择子策略”在这一窗口上没有产生可辨识的增益。
3. **它确实降低了风险。** 最大回撤 0.4238 %（五变体最低）、在场时间 33.54 %（最低），
   与 S3 预览的方向一致——**但那是“少暴露”，不是“选得对”**，且代价是收益也最低（除 seed 8）。
4. **正收益的来源必须写明。** 717 笔成交的已实现 PnL 合计 **−3.10 USDT**（≈ −0.031 %），
   组合体的 +0.44 % 主要来自区间内的账面净值路径；同时每个子策略平均只动用约 **2 %** 资金。
   因此**不能**把 +0.4425 % 读成“策略有每笔优势”。
5. **P7 的处置**：S1（单策略条件化）、S3（编排器）、S4（组合体）三个阶段的样本外 DSR 全部非正
   ⇒ **P7 不宣称任何有效性，任何默认开关都不打开**（`ga.regime_conditioning: false`、
   `ai.orchestrator.enabled: false`、`experimental.regime_orchestrator_live: false` 全部保持 shipped 值）。
   本轮唯一可复用的交付物是**基础设施**：组合体契约、匹配基准、逐变体 DSR 记账、一次性 holdout 计数器
   ——它们让下一次“编排器有没有用”这个问题第一次变成可测的，而本轮答案是**没有**。

### S4.9 需要 Lead 处理的残余 / 计划订正
1. **计划文件写的工具名不存在**：`P7_REGIME_PLAN.md:296` 写证据命令为 `tools/p7_composite_oos.py`，
   实际交付的是 **`tools/p7_composite_measure.py`**。**我没有改计划文件**（冻结）；请 Lead 决定是否订正这一行。
2. **计划要求 `validation_start`，本轮未用**：`P7_REGIME_PLAN.md:284-285` 允许编排器的选择使用
   “训练窗（含独立的 `validation_start`）”。本轮的编排器选择与权重**全部**只用训练窗
   （严格 holdout，没有第二个选择窗口），满足“样本外固定”的强形式；若 Lead 想要独立的验证窗，
   需要把训练窗再切一刀，这会改变权重与规则的训练样本量。
3. **`ml.gate_min_trades=100` 与 GA `min_champion_trades=30` 的口径差**（S1 已记录）：S4 按 100 报组合体，
   本轮组合体过 100 笔门（717 / 1 398），因此这个门槛**不是**本轮被拒的原因；被拒的是 DSR。
4. **窗口被打开 3 次**（§S4.6）：方向性结论不受影响（两次有效运行逐位一致），但该窗口的“一次性”已被消耗；
   后续任何 S4 复核请换窗口。
5. **未做**：`--selection-population` 更大规模的选择扫描（本轮 4/币）、按币独立编排器标签
   （本轮所有策略共享第一个币的状态时钟，工具 docstring 与产物 `orchestrator` 块已写明）、
   多窗口重复（本轮只有一个样本外窗口）。这些都是**成本换强度**的选项，不是缺陷。

