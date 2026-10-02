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
