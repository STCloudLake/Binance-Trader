# 核心算法参考 / Core Algorithms Reference

**Repository**：`E:\Codes\Binance Trader` · **Revision**：`c703b8b`（`git rev-parse HEAD` = `c703b8ba485a5410dba72ea5e0659aade0268276`，测量时刻工作树干净）
**性质**：研究级公式/算法参考，为个人精读《金融时间序列分析》（Tsay）与 López de Prado 系列而写。
**纪律**：本文每一个数字都来自仓库里的文件或我在本 revision 上跑过的命令，并给出 `file:line`。凡无法验证者一律写 **未验证**，不含任何盈利性声明。

> **写作时的工作树状态（重要）**
> 本文件写入期间，兄弟代理正在编辑 `core/ml/features.py`（特征契约 v2）。因此：
> * `core/ml/features.py` 的**盘上内容处于编辑中状态**（曾一度无法通过 `ast.parse`，随后可解析但期望 52 列、实际产出 39 列；测得的 `DEFAULT_FEATURES` 一度为 **54** 列）；
> * `scripts/ml_credibility_measure.py` **在本 revision 上无法复现**，报错原文见 [§2.9](#29-本-revision-上无法复现的部分已实测记录) 与 [§10](#10-什么没有实现--什么被证伪-what-is-not-implemented--what-is-falsified)；
> * `core/strategy/volume_bars.py`（任务书中提到的兄弟代理新增文件）在**我开始工作时不存在**（`Test-Path` 为假）；写作期间它作为未跟踪文件出现了，我**没有读它**（超出我的阅读/引用范围）。
>
> **凡涉及这些文件的结论，我都以 commit 里的内容（`git show HEAD:<path>`）或仓库已归档的测量记录为据，并显式标注。**
>
> **行号稳定性**：本文写作期间工作树上有 **15 个已跟踪文件被修改**（`core/backtest/{cost_model,engine}.py`、`core/ga/{evolver,fitness,genome}.py`、`core/market_data/{ohlcv_cache,provider}.py`、`core/ml/{features,predictor}.py`、`core/risk/{liquidity,manager,position_guard}.py`、`scripts/download_history.py`、`tests/test_{engine_ml_gate,gap_fixes,ml_credibility}.py`）加 **2 个新未跟踪文件**。因此文中的 `file:line` 对应的是**我读取时的盘上或 commit 内容**；其中 `features.py`、`ohlcv_cache.py`、`genome.py`、`manager.py`、`cost_model.py`、`liquidity.py`、`fitness.py`、`evolver.py` 这 8 个的行号**在并行改动落地后会漂移**，请以**符号名**为锚再定位。
>
> **证据分级**（全文使用）：**(A)** 我在本 revision 上亲自跑过的命令读数；**(B)** 仓库归档的实测记录（docstring/日志/job 产物，含历史 revision 的快照）；**(C)** 明文标注为示意/非实测的数字。凡不能归入三者的，写 **未验证**。

---

## 目录 / Table of Contents

0. [符号与记号表](#0-符号与记号表-symbols--notation)
1. [GA 可信性（P1）](#1-ga-可信性-p1)
   - 1.1 [逐基因组账本与槽位](#11-逐基因组账本与槽位-per-genome-ledger--slots)
   - 1.2 [Walk-forward 切分与真实样本外窗口](#12-walk-forward-切分与真实样本外窗口)
   - 1.3 [适应度函数](#13-适应度函数-the-fitness-function)
   - 1.4 [Profit factor 的收缩 / 截断 / 按证据缩放](#14-profit-factor-的收缩--截断--按证据缩放)
   - 1.5 [Deflated Sharpe Ratio（DSR）](#15-deflated-sharpe-ratio-dsr)
   - 1.6 [试验计数](#16-试验计数-trial-counting)
   - 1.7 [发布门与精英保留、简约压力](#17-发布门精英保留与简约压力)
   - 1.8 [按基因名的交叉与变异](#18-按基因名的交叉与变异)
   - 1.9 [已实测数字](#19-已实测数字-measured-evidence-ga)
   - 1.10 [已知局限（GA）](#110-已知局限ga)
2. [机器学习（P2）](#2-机器学习-ml-p2)
   - 2.1 [三重屏障标签与波动率缩放屏障](#21-三重屏障标签与波动率缩放屏障)
   - 2.2 [样本唯一性权重](#22-样本唯一性权重-sample-uniqueness-weights)
   - 2.3 [Purged K-Fold 与 embargo](#23-purged-k-fold-与-embargo)
   - 2.4 [Isotonic 概率校准](#24-isotonic-概率校准)
   - 2.5 [成本感知阈值选择（嵌套）](#25-成本感知阈值选择嵌套-nested)
   - 2.6 [外层门](#26-外层门-the-outer-gate)
   - 2.7 [PSR：Prado 偏度/峰度修正标准误](#27-psr-prado-偏度峰度修正标准误)
   - 2.8 [版本化特征契约与 schema hash](#28-版本化特征契约与-schema-hash)
   - 2.9 [本 revision 上无法复现的部分](#29-本-revision-上无法复现的部分已实测记录)
   - 2.10 [诚实的实测判决](#210-诚实的实测判决-verdicts)
3. [波动率（P3）](#3-波动率-p3)
   - 3.1 [RiskMetrics / EWMA 递归与半衰期](#31-riskmetrics--ewma-递归与半衰期)
   - 3.2 [GARCH(1,1) 自由 ω MLE](#32-garch11-自由-ω-mle)
   - 3.3 [IGARCH 网格回退](#33-igarch-网格回退)
   - 3.4 [Parkinson / Garman-Klass / close-close](#34-parkinson--garman-klass--close-close)
   - 3.5 [锚定 MAD 与"为什么滚动 MAD 会改写历史"](#35-锚定-mad-与为什么滚动-mad-会改写历史)
   - 3.6 [波动率目标化：反比定仓与 clamp](#36-波动率目标化反比定仓与-clamp)
   - 3.7 [动态 barrier 宽度](#37-动态-barrier-宽度)
   - 3.8 [拼接闸门](#38-拼接闸门-gap-guard)
   - 3.9 [已实测数字（P3）](#39-已实测数字-p3)
4. [配对 / 协整（P4）](#4-配对--协整-p4)
5. [Meta-labelling（P4）](#5-meta-labellingp4)
6. [微观结构（P4）](#6-微观结构-p4)
7. [Regime 门控（P4）](#7-regime-门控-p4)
8. [执行真实性（P6-A）](#8-执行真实性p6-a)
9. [数据完整性（横切）](#9-数据完整性横切-data-integrity)
10. [什么没有实现 / 什么被证伪](#10-什么没有实现--什么被证伪-what-is-not-implemented--what-is-falsified)
11. [文档 ↔ 代码矛盾清单](#11-文档--代码矛盾清单-discrepancies)（D-1 … D-31）

---

## 0. 符号与记号表 / Symbols & Notation

| 符号 | 含义 | 单位 | 本项目取值的来源 |
|---|---|---|---|
| $r_t$ | 对数收益 $\ln(P_t/P_{t-1})$ | 无量纲 | `core/ml/volatility.py:201` `log_returns` |
| $\lambda$ | RiskMetrics 衰减因子 | 无量纲 | `DEFAULT_LAMBDA = 0.94`，`core/ml/volatility.py:104` |
| $\sigma_t^2$ | 条件方差（下一根 bar） | 比例²/bar | `ewma_variance`，`core/ml/volatility.py:508` |
| $T$ | 收益观测数（真实期数，**不是** 365） | 根 | `observation_periods` 参数，`core/ga/fitness.py:229` |
| $N$ | 试验次数（population×generations + 历史） | 次 | `core/ga/trial_counter.py` |
| $\mathrm{SR}$ | Sharpe（默认年化输入，内部折成每期） | 无量纲 | `deflated_sharpe_ratio`，`core/ga/fitness.py:226` |
| $\gamma_3,\gamma_4$ | 样本偏度、**非超额**峰度 | 无量纲 | `core/ml/credibility.py:303-304` |
| $w_i$ | 标签 $i$ 的平均唯一性权重（均值归一化） | 无量纲 | `sample_uniqueness_weights`，`core/ml/evaluation.py:171` |
| $h$ | 标签前向窗口长度（label span / embargo） | bar | `label_span`，`evaluate_model_oos`，`core/ml/credibility.py:501` |
| $p$ | 参与率 = 名义 / 窗口成交额 | **分数**（0.01 = 1 %） | `participation_pct`，`core/risk/liquidity.py:370` |
| $k$ | 冲击律系数（`impact_k`） | 无量纲 | `DEFAULT_IMPACT_K = 0.0`，`core/risk/liquidity.py:89` |
| $e$ | 冲击律指数（`impact_exponent`） | 无量纲 | `DEFAULT_IMPACT_EXPONENT = 0.5`，`core/risk/liquidity.py:92` |
| $\tau$ | ADF 统计量 $\hat\varphi/\mathrm{se}(\hat\varphi)$ | 无量纲 | `adf_regression`，`core/strategy/pairs.py:354` |
| $\theta$（本文亦记 $\kappa$） | OU 均值回复速度 $=-b$ | 1/bar | `ou_half_life`，`core/strategy/pairs.py:690` |
| $z_t$ | 价差 z-score（不含 $t$ 自身） | 无量纲 | `rolling_zscore`，`core/strategy/pairs.py:874` |
| $\sigma_{\max}/\sigma_{\min}$ | HMM 两状态的波动率比 | 无量纲 | `HMM_MIN_SIGMA_RATIO = 1.5`，`core/strategy/regime.py:128` |

记号约定：$t$ 是当前 bar；所有估计量默认**因果**（只用 $\le t$ 的信息），例外会在正文点名。

---

## 1. GA 可信性（P1）

> 证据主文件：`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md` §1（`:23-58`）、`docs/core-algorithms/06-ga-evolution.md`、`docs/core-algorithms/07-deflated-sharpe-ratio.md`。

### 1.1 逐基因组账本与槽位（per-genome ledger & slots）

**算法.** 一个 GA 代里 $\text{population}$ 个基因组被打包成若干 *chunk*（每 chunk 一次回测调用）。为了让 chunk 内的每个基因组是**独立策略**而不是互相抢仓位/抢现金，评估时打开两个引擎开关：

* `per_strategy_isolation=True` —— 给持仓键加命名空间；
* `per_genome_ledger=True` —— 再给每个基因组一份**独立的仓位槽位预算 + 现金/权益子账本**。

`isolated_eval_kwargs()`（`core/ga/fitness.py:58-77`）用 `inspect.signature` 探测第二个开关是否存在，所以旧引擎仍可用；`evaluate_population_batch` 的 chunk 路径显式传两个开关（`core/ga/fitness.py:640-641`）。

**为什么需要.** 修复前（P1 之前）一个 chunk 里 20 个基因组共享账本，实测"20 个基因中 1 个交易 407 次、其余 19 个 0 次"（该 before 数字的来源见 `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:48`，来自 P1 之前的只读审计，**本次未重跑旧代码**）。修复后每一个基因有自己的槽位与账本，`tests/test_ga_credibility.py::test_every_genome_in_chunk_gets_its_own_slots_and_ledger` 断言"最闲基因的成交数 ≥ 最忙基因 × 0.2"。

**实现位置.** `core/ga/fitness.py:58-77`（开关探测）、`:631-643`（chunk 调用）、`core/backtest/engine.py`（`run_with_exit_evaluation` 的 `per_genome_ledger` 形参）。

**已实测证据.** 第一代逐基因账本：`ga_rand_2=75`、`ga_rand_0=60`、`ga_rand_3=103`、`ga_rand_1=45`、`ga_rand_5=32`、`ga_rand_4=21` 笔（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:38`）。24 次评估中 20 次有交易（`:44`）。

**已知局限.**
1. 账本隔离**不保证**每个基因都交易：第 4 代 6 个基因中仍有 1 个 immigrant `flag=no_trades`、fitness −46.71（`:44`）。
2. "每基因独立账本"改变了每笔成交的可用资金语义，因此 GA 的 fitness 与**单策略回测**的 fitness 不是同一个数字（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:41-42` 用"第 1 代最优与第 4 代领先者的交易结果完全相同（75 笔、Sharpe 9.4388、DSR 0.2119）"说明差异只来自复杂度罚项）。
3. 冠军的**最终** train 分数走 `score_stats`（`core/ga/fitness.py:172`），与代内 batch 路径同一个公式；但 `docs/core-algorithms/06-ga-evolution.md` 里旧描述称两条路径曾是两个公式（已在代码里合并，见 `core/ga/fitness.py:130-132` 的说明）。

### 1.2 Walk-forward 切分与真实样本外窗口

**算法.** `WalkForwardRunner.compute_windows`（`core/ga/walkforward.py:149-175`）以 `DateOffset(months=…)` 滚动：

$$\text{train}=[c,\;c+\Delta_{\text{train}}],\qquad \text{val}=[c+\Delta_{\text{train}},\;c+\Delta_{\text{train}}+\Delta_{\text{val}}],\qquad c \leftarrow c+\Delta_{\text{step}}$$

`WFConfig` 默认 `train_months=6, val_months=1, step_months=1`（`core/ga/walkforward.py:23-29`），循环条件是 `cursor + train_delta + val_delta <= end`（`:163`）。

**关键修复.** 旧调用把 `train_end` 当成 `date_end` 传给 `evolve`，而 `val_start == train_end`，于是每个窗口"验证"在**单根 bar** 上——生产日志里 24 个 job 全是 `validate=2025-11-01~2025-11-01`（`core/ga/walkforward.py:245-252` 的注释）。现在调用为 `evolve(symbols, tr_start, val_end, validation_start=val_start)`（`:256-263`），并且在起跑前用 `_assert_window`（`:304-323`）**硬校验**：

* `val_start < val_end`，否则 `ValueError`；
* 样本外 bar 数 $\ge$ `MIN_VALIDATION_BARS = 30`（`core/ga/walkforward.py:20`），否则 `ValueError`（"refusing to validate on a degenerate window"）。

bar 数从缓存 parquet 里真实数出来（`_validation_bars`，`:325-349`，读 `close` 列并按时间戳切片）。

**为什么需要.** 一个"验证"在 1 根 bar 上的窗口产生的所有 WF 统计量（`wf_efficiency`、`train_val_correlation`、`positive_window_pct`）都没有意义，却会被写进报告并影响冠军选择。

**聚合量**（`WFReport.from_results`，`:64-102`）：

$$\text{wf\_efficiency}=\frac{\overline{\mathrm{SR}}_{\text{val}}}{\mathrm{sd}(\mathrm{SR}_{\text{val}})}\quad(\text{sd 用 ddof}=1),\qquad \text{positive\_window\_pct}=\frac{\#\{\mathrm{SR}_{\text{val}}>0\}}{n}\times100$$

$n\ge3$ 才有 `train_val_correlation`（否则 0，`:79-84`）。

**已知局限.**
1. `MIN_VALIDATION_BARS=30` 是一个**下限**，不是充分性证明；30 根 1h bar 上的 Sharpe 方差极大。
2. `_validation_bars` 取所有 symbol 的**最大** bar 数（`best = max(...)`，`:346`），因此一个 symbol 有数据就通过——多品种窗口的数据可用性没有被逐一校验。
3. 报告里没有任何多重检验校正；`wf_efficiency` 是均值/标准差之比，不是信息比率，也没有置信区间。
4. 未验证：本 revision 上我没有跑过完整的 `WalkForwardRunner.run`（需要真实回测引擎与长时间运行）。

### 1.3 适应度函数（the fitness function）

**算法（单公式，所有路径共用）.** `score_stats`（`core/ga/fitness.py:425-518`）是唯一的适应度求值器；单基因路径（`evaluate_chromosome`）与批量 chunk 路径（`evaluate_population_batch`）都调它。

$$\begin{aligned}
\text{fitness} =\;& \underbrace{w_{wr}\cdot \text{win\_rate}}_{\text{胜率}} + \underbrace{w_{pf}\cdot \widetilde{PF}}_{\text{收缩+截断+缩放的 PF}} + \underbrace{w_{roc}\cdot \text{ROC}}_{\text{资金回报率}} - \underbrace{w_{bal}\cdot \text{imbalance}}_{\text{多空失衡}}\\
& + \underbrace{\text{penalties}}_{\text{证据罚项}} + \underbrace{\alpha_{\text{term}}}_{\text{DSR 折减 Sharpe} - \text{回撤}} - \underbrace{\text{complexity\_penalty}}_{\text{简约压力}}
\end{aligned}$$

各项的精确实现：

| 项 | 公式 | 代码 |
|---|---|---|
| 权重 | $w_{wr}=0.15,\;w_{pf}=5.0,\;w_{roc}=50,\;w_{bal}=10.0$ | `DEFAULT_WEIGHTS`，`core/ga/fitness.py:55` |
| $\text{win\_rate}$ | $\dfrac{\#\{pnl>0\}}{n}\times100$（百分数） | `core/ga/fitness.py:391` |
| $\text{imbalance}$ | $\left|\dfrac{\#\text{long}}{n}-0.5\right|\times 2$；$n=0$ 时取 1 | `:446-449` |
| $\text{ROC}$ | $pnl/\text{initial\_balance}$（**分数**，不是百分数） | `:398` |
| 证据罚项 | $n<5:\;-20$；$5\le n<15:\;-5$；$n>500:\;-(n-500)\times0.02$；$pnl<-50:\;-|pnl|\times0.3$ | `:459-467` |
| $\alpha_{\text{term}}$ | $\mathrm{DSR}\cdot\sqrt{365}\cdot\min\!\left(1,\frac{n}{30}\right)-\text{max\_dd}\%$ | `:491-496` |
| 回归项 | 减去 `complexity_penalty` | `:498-499` |

> ⚠️ **$\sqrt{365}$ 是必须写出来的一步**：`dsr` 是**每期**口径（§1.5），代码显式年化后才乘证据量（`dsr_sharpe = _dsr["dsr"] * (365.0 ** 0.5)`，`core/ga/fitness.py:491`）。若照抄 `docs/core-algorithms/06-ga-evolution.md:115` 的 `alpha = DSR × min(1, trades/30) − max_dd`，结果会比代码**小 $\sqrt{365}\approx19.1$ 倍**；用归档冠军（DSR 0.2119、75 笔、max_dd 0.12 %）验算：代码 → $0.2119\cdot19.105-0.12=3.928$，文档公式 → $0.2119-0.12=0.092$。**代码是对的**（第 3 代 fitness 增量恰为复杂度罚项差 4.00 只有在代码口径下成立，见 §1.9）。见 §11 D-14。

复合项 $\alpha_{\text{term}}$ 的设计意图是**用证据量打折风险调整收益，再直接扣回撤**：`SHARPE_TRADE_FLOOR = 30`（`:45`）、`ALPHA_WEIGHT = 1.0`（`:49`）。当观测数 $<$ `MIN_OBSERVATIONS = 20`（`:51`）时**不做任何 Sharpe/DSR 估计**，$\mathrm{DSR}\equiv0$（`:483-489`），于是

$$\alpha_{\text{term}}\big|_{T<20} = 0\cdot\sqrt{365}\cdot\min(1,n/30) - \text{max\_dd} = -\,\text{max\_dd}$$

即 **alpha 项不是 0，而是一个纯回撤罚项**——`docs/core-algorithms/07-deflated-sharpe-ratio.md:62-63` 写"DSR 记 0、alpha 项记 0"是错的，见 §11 D-15。注意两项**单位不同**：年化 Sharpe 量级（可到 9.4）与百分点回撤直接相加，这就是 §1.10 局限 3 的来源。

**买入持有基准.** $\alpha_{\text{vs buy\&hold}} = \text{total\_return\_pct} - \text{buy\_hold\_pct}$（`:504-510`），其中 `buy_hold_pct` 来自引擎 metrics（同窗口、等权）：

$$\text{BH}\% = 100\cdot\frac{1}{|S|}\sum_{s\in S}\left(\frac{\text{close}_{s,\text{last}}}{\text{close}_{s,\text{first}}}-1\right)$$

（`core/backtest/engine.py:1504-1540`，按 `(data_dir, symbols, first_ts, last_ts)` 缓存）。

> ⚠️ **它不进入 `fitness`**。`docs/core-algorithms/06-ga-evolution.md:131-134` 说 fitness"**减去**同窗口同币种的等权买入持有收益"，`core/ga/fitness.py:22-23` 的模块 docstring 说同样的，`config/config.yaml:54-56` 也这么说——**三处都不成立**。我按代码路径核对：fitness 的和只有 `base + alpha*ALPHA_WEIGHT − complexity`（`:451-456`, `:496-499`），`alpha_vs_buy_hold_pct` 只被**赋值并上报**（`:505-510`），随后只被**发布门**消费（`core/ga/evolver.py:461-463`）。子代理用同一份 stats 改 `buy_hold_pct` 从 `None` 到 `25.0` 实测：`fitness` **两次都是 29.1535**，只有 `alpha_vs_buy_hold_pct` 从 0.0 变成 −14.739。见 §11 D-16。

**发布门额外条件.** 除上表五项，`_publication_decision` 在**提供了验证窗口**时还要求 $v_{\text{trades}}>0 \wedge v_{\text{Sharpe}}>0 \wedge v_{\text{DSR}}>0$（`:465-474`）——共 **8** 项条件。`docs/core-algorithms/06-ga-evolution.md:147-149` 的门清单**少了 `alpha_vs_buy_hold>0` 与三条 validation 条件**，而实测冠军的**唯一**拒绝理由恰好就是它漏掉的那一条（§1.9）。另外"净盈亏 > 0"里的 `pnl` 实际是 `train_result["total_return"]`，即**百分比**而不是 USDT（`core/ga/evolver.py:445`），但拒绝串仍打印 `net_pnl={pnl:.2f}`（见 §1.7）。

**为什么这样改.** 模块 docstring（`:1-24`）记录了修复前的三个缺陷：PF 无界（5 笔全胜得 490，200 笔 PF=2.0 得 7.3）、批量路径把 Sharpe 与最大回撤硬编码为 0、从未减去同窗口买入持有（纯 beta 被当成 alpha）。前两个已修；**第三个没有被修**（§1.3 的警告）。

**权重可被覆盖但事实上没有.** `w = dict(DEFAULT_WEIGHTS)` 只接受 `weights` 里已知的键（`:432-434`）；`FitnessCalibrator.load_weights_static` 从 `<data_dir>/data/ga_fitness_weights.json` 读（`core/ga/fitness_calibrate.py:64-75`，由 `core/ga/evolver.py:200` 调用）。**该文件在本 revision 不存在**（仓库里只有 `data/ga_wf_state.json`），所以每条路径用的都是硬编码默认值。`WEIGHT_GRID`（`fitness_calibrate.py:30-35`）是**死数据**——无任何读取者，模块 docstring 自述那个网格搜索"never wired to any route or caller and has been removed"。

**已知局限.**
1. `fitness` 是**量纲混合**的加权和，权重 $0.15/5/50/10$ 是遗留标定值（`DEFAULT_WEIGHTS` 的注释自称 "Legacy default weights"），没有重新标定的证据；`docs/ga/`… 未验证其来源。
2. `win_rate` 与 PF 正相关，二者同时进入线性组合有重复计分。
3. `max_dd` 以**百分点**直接扣（`stats_out["max_dd"]`，`core/ga/fitness.py:473`），而 DSR 项是年化 Sharpe 量级（可达 9.4，见 §1.9），因此回撤项的边际影响远小于 Sharpe 项——这是实测数字解读（`:42`）而不是文档声明的设计。
4. `_finite` 把所有非法值（NaN/inf）吞成 0（`:80-88`），因此一个损坏的评估会得到"中性"而非"最差"的分数（`-999` 只用于引擎返回 error 的路径，`:157`）。

### 1.4 Profit factor 的收缩 / 截断 / 按证据缩放

**算法.** 三步，缺一不可：

$$\begin{aligned}
\widetilde{PF} &= \frac{G_{\text{win}}}{G_{\text{loss}} + \overline{W}} &&\text{(收缩：}\overline{W}=\text{平均盈利}\text{)}\\
\widetilde{PF} &\leftarrow \min\!\big(\max(\widetilde{PF},0.1),\;10\big) &&\text{(截断：} \texttt{PF\_TERM\_CAP}=10\text{)}\\
\widetilde{PF} &\leftarrow \widetilde{PF}\cdot\min\!\left(1,\frac{n}{50}\right) &&\text{(按证据缩放：}\texttt{PF\_TRADE\_FLOOR}=50\text{)}
\end{aligned}$$

实现：收缩在 `profit_factor_shrunk`（`core/ga/fitness.py:91-109`；`gw<=0` 或 `denom<=0` 返回 0.1；`mean_win<=0` 时用 `gross_win` 兜底），截断+缩放在 `score_stats`（`:438-439`）。原始 PF 仍被报出（`raw_profit_factor`，`:392`，$G_{\text{loss}}=0$ 且 $G_{\text{win}}>0$ 时记 100.0）。

**为什么 raw PF 不可用.**
1. **无界**：分母为 0 时 PF → ∞。旧实现给 `gross_loss==0` 打 PF 100，乘权重 5 得 500 分，超过所有其他项之和（`core/ga/fitness.py:8-10, :95-99`）。
2. **小样本可刷分**：5 笔全胜的 PF 极高。旧公式实测 `490.85` 对 200 笔的 `7.30`（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:58`，旧公式留档在 `tools/p1_measure.py::_legacy_ranking`）。

**为什么"加一个平均盈利"就能定界.** 若 $N$ 笔全为盈利且无亏损，则 $\widetilde{PF}=G_{\text{win}}/\overline{W}=N$——上界恰好是**样本量**，再被 $\min(1,n/50)$ 与 `PF_TERM_CAP` 双重压制，于是"运气好"与"证据多"在数值上被分开。

**已知局限.**
1. 收缩常数是**平均盈利**，是一个临时选择而非推导出的先验；它使 PF 依赖盈利分布的形状。
2. `PF_TERM_CAP=10` 之后，$N>10$ 的全胜基因组彼此不再可区分（被压平）。
3. `raw_profit_factor` 的 100.0 是一个哨兵值，不是 PF 的估计（`:371-372`）。

### 1.5 Deflated Sharpe Ratio（DSR）

**算法.** `deflated_sharpe_ratio`（`core/ga/fitness.py:226-309`）。先统一单位，再减期望最大值，再做非正态修正：

$$\begin{aligned}
\mathrm{SR}_{\text{per}} &= \frac{\mathrm{SR}_{\text{ann}}}{\sqrt{365}} &&(\texttt{DSR\_PERIODS\_PER\_YEAR}=365,\;:53)\\[2pt]
E[\max \mathrm{SR}] &\approx \sqrt{\frac{\mathrm{Var}[\mathrm{SR}]}{T}}\sqrt{2\ln N} &&(\mathrm{Var}[\mathrm{SR}]\approx1 \Rightarrow \sqrt{1/T}\sqrt{2\ln N})\\[2pt]
\mathrm{DSR} &= \mathrm{SR}_{\text{per}} - E[\max \mathrm{SR}]\\[2pt]
\text{var\_term} &= 1 - \gamma_3\,\mathrm{SR}_{\text{per}} + \frac{\gamma_4-1}{4}\,\mathrm{SR}_{\text{per}}^2\\[2pt]
z &= \frac{\mathrm{DSR}\sqrt{T-1}}{\sqrt{\text{var\_term}}},\qquad p = 1-\Phi(z)\\[2pt]
\text{significant} &\iff \mathrm{DSR}>0 \;\wedge\; p<0.05
\end{aligned}$$

实现对照：`:281-282`（年化→每期）、`:290`（$E[\max]$，`math.sqrt(var_sr / t_periods) * math.sqrt(2*log(n_trials))`）、`:291`（DSR）、`:295-297`（var_term，并 `max(var_term, 1e-9)`）、`:298`（z）、`:305`（significant）。边界：`sr<=0`、$N\le1$ 或 $T<2$ 时返回 `dsr=0, p=1, significant=False`（`:284-287`）。

**单位是本项目最容易错的地方.** 旧实现拿**年化** Sharpe 直接减一个**每期**门槛，并把 `T` 写死 365，于是门槛是常数（$N=1200$ 时 ≈0.191），任何年化 Sharpe > 0.2 都被判"显著"。对照算例（`docs/core-algorithms/07-deflated-sharpe-ratio.md:39-43`）：

| 算例 | SR | N | T | 旧实现 | 现在 |
|---|---|---|---|---|---|
| 审计算例 | 1.2（年化） | 1200 | 1200 | **+1.0029「显著」** | **−0.0459（不显著）** |
| 同 Sharpe，T=365 | 1.2（年化） | 1200 | 365 | +1.0029 | −0.1343（不显著） |
| 每期 Sharpe | 0.30（每期） | 100 | 250 | — | +0.1081（显著） |

手算校验：`tests/test_ga_credibility.py::test_dsr_hand_check_units`（`docs/core-algorithms/07-deflated-sharpe-ratio.md:46-47`）。子代理在 HEAD 上重算过这三个算例：$T=1200$ → `dsr=-0.045894`、`expected_max_random=0.108705`；$T=365$ → `dsr=-0.134292`、`expected_max_random=0.197103`；每期 0.30 / $N=100$ / $T=250$ → `dsr=0.108059`、`p=0.0477`、`significant=True`，全部与文档一致。

> ⚠️ **同一份文档里的自相矛盾**：`docs/core-algorithms/07-deflated-sharpe-ratio.md:36-37` 写"门槛是一个与数据长度无关的常数（**N=1200 时 ≈0.191**）“，而同一页 `:41` 报旧实现输出 `+1.0029`。旧口径的门槛是 $\sqrt{1/365}\sqrt{2\ln1200}=0.197103$，与 $1.2-0.197103=1.002897$（即 1.0029）**自洽**；0.191 与它不自洽。同样的过期算术还留在 `tests/test_ga_credibility.py:386-391` 的 docstring 里（写 `E[max]=0.19148 / DSR=-0.12866`），而该测试**自己的断言**（`:399-402`）用的是 `T=365, N=1200`，实际求值为 `0.197103 / −0.134292`。**断言通过，docstring 数字过期**。见 §11 D-17。

**边界行为的两个细节.**
1. `sr <= 0` 时返回 `dsr = 0.0`（不是负数，`:284-287`）——门把两者同等对待，但这个裁剪让"负 Sharpe 有多负"在 DSR 里不可见。
2. DSR 内部的 $T<2$ 守卫（`:284`）与 `score_stats` 的 $T<20$ 门槛（`:475`）是**两个不同的阈值**，作用在不同层。

**$\gamma_3,\gamma_4$ 的来源.** 基因组**自己的日收益序列**（`stats_from_trades` 里 `pd.Series(rets).skew()`、`kurtosis()+3`，`core/ga/fitness.py:404-405`；样本不足 3/4 时取 0 / 3）。注意 `pandas.kurtosis()` 默认是**超额**峰度，代码 `+3.0` 换回非超额，与 $\gamma_4$ 的定义一致。

**$\sqrt{T-1}$ 与 $\sqrt{1/T}$ 的口径.** 门槛用 $\sqrt{1/T}$（`:290`），z 用 $\sqrt{T-1}$（`:298`）——这两个是不同来源的近似（前者是 $E[\max]$ 的极值近似，后者是 PSR 的抽样标准误），本项目**没有**把它们统一，也没有文档讨论。

**已知局限.**
1. $\mathrm{Var}[\mathrm{SR}]\approx1$ 假设 i.i.d. 收益（`:267` 的 docstring 自述）；金融收益的序列相关会让真实方差偏离 1，代码把它当参数（`variance_sharpe`，默认 1.0）而不是估计它。
2. 极值近似 $E[\max]\approx\sqrt{1/T}\sqrt{2\ln N}$ 假设 $N$ 个**独立**试验；GA 的基因组高度相关（实测：第 4 代 6 个基因中 4 个交易结果逐位相同，`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:45`），因此真实的有效试验数 $\ll N$，$E[\max]$ 会被**高估**（即门槛偏严）。
3. `p_value` 在返回时被 `round(...,4)`（`:304`），而 `significant` 用的是未取整的 `p_value`（`:305`）——两者可能不一致（例如真值 0.05001 与 0.050049 都显示 0.05，但一个 significant 一个不是）。
4. DSR 只用**每基因自己的**日收益，日频重采样（`:317-331`）在 1h 回测里把一天的多根 bar 压成最后一个权益值，丢弃了日内路径。

### 1.6 试验计数（trial counting）

**算法.** `core/ga/trial_counter.py` 维护 `data/ga_trials.json`；每代评估后按 `population_size` 累加，`total_trials = 历史 + 当前`。`score_stats` 接受 `n_trials`（本代）与 `prior_trials`（跨代/跨窗口累计），并把二者之和传给 DSR：`int(n_trials) + int(prior_trials)`（`core/ga/fitness.py:477`）。

**关键区分：两个不同的 $N$ 同时存在（文档没有说）.**

| 用途 | $N$ | 代码 |
|---|---|---|
| **代内每个基因的 fitness**（即 DSR 进入 fitness 的那一次） | $\text{population} + N_{\text{prior}}$ | `core/ga/evolver.py:202`（`_batch_trials = len(self._population)`）→ `core/ga/fitness.py:476-477` |
| **冠军 / 验证** 的那一次 | $\text{population}\times\text{generations} + N_{\text{prior}}$ | `core/ga/evolver.py:319-320` |

而 `evolver.py:413-415` 返回给调用方的 `dsr` 取的是 `train_result["dsr_detail"]`，即**代内**那一个；相邻的 `provenance["n_trials"]`（`:371-372`）却是**更大的**那一个。所以对归档的 4 代 / population 6 那次运行，被上报的冠军 DSR **0.2119** 是用 $N=6+N_{\text{prior}}$ 算的，而 provenance 会声称 $N=24+N_{\text{prior}}$——**同一个策略的两个 $N$**。见 §11 D-18。`docs/core-algorithms/06-ga-evolution.md:142-144` 只写了后者。

**为什么需要.** 否则第 24 个 walk-forward 任务的冠军会被当成"只试过 450 个策略"（`docs/core-algorithms/07-deflated-sharpe-ratio.md:81-83`），$E[\max]$ 被系统性低估，DSR 变成假阳性制造机。**"450"是 30×15 的举例**，不是账本读数（子代理核对：`data/ga_jobs/wf_1053fef2.json` 恰好是 population 30 / generations 15，但没有任何文档行做出这个绑定）。

**已实测证据（B 级，含仓库内的真实 job 产物）.**

* **账本负担的物理证据**：`data/ga_jobs/` 下有 **25** 个 `wf_*.json` job，其中 `wf_bad245f6.json` 是 `{"test": true}` → **24 个真实 walk-forward job**（population×generations 组合有 100×30、50×15、30×15、30×5），另有 5 个 `ga_*.json`。这正是"~24 个任务"（`docs/core-algorithms/06-ga-evolution.md:144`、`trial_counter.py:6`）的仓库级佐证。
* 证据索引 §1.1 的 DSR 列在 4 代内保持不变（0.2119），与"DSR 由交易结果 + 试验数决定、而交易结果未变"一致（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:31-34`）。
* `tests/test_ga_credibility.py:443-449`：`record_trials(tmp,20,"w1")` + `record_trials(tmp,20,"w2")` → `load_trials == 40`、`total_trials(tmp,5) == 45`；新目录上 `load_trials == 0`。
* **本 revision 上 `data/ga_trials.json` 与 `data/ga_fitness_weights.json` 都不存在**，所以新运行一律从 $N_{\text{prior}}=0$ 与默认权重开始。

**已知局限.**
1. 计数是**把每一代的 population_size 全量累加**，而不是按"被选择/被评估出 fitness 的个体"计数——一个早停或崩溃的代会照样计数。
2. 归档的那个 `best=-999.00` 崩溃代（§1.9）也走的是同一条 `record_trials` 路径（`core/ga/evolver.py:237-240` 的 try/except 只保护异常，不保护"全种群 −999"这种**有效返回**），因此崩溃的代会把 50 次试验加进账本。
3. `data/ga_trials.json` 是**全局跨币种/跨窗口**的，因此给 BTC 做 GA 会抬高 ETH 的 DSR 门槛。这在统计上可辩护（同一研究者多次试验），但意味着**结果不可独立复现**：换了历史文件，同一个种子的 DSR 会变。未验证：我没有读取或改动该文件（它当前不存在）。
4. `total_trials()`（`trial_counter.py:75-77`）**没有生产调用者**——调用方自己用 `load_trials` + 两个参数。
5. §1.9 上报的 `dsr=0.2119` 用的是**代内** $N$（见上表），因此它不是一个"跨 4 代的去偏 Sharpe"。

### 1.7 发布门、精英保留与简约压力

**发布门.** `_publication_decision`（`core/ga/evolver.py:431-476`）。冠军只有全部条件成立才写 `enabled: true`：

$$\text{publish} \iff \underbrace{n\ge n_{\min}}_{30} \wedge \underbrace{pnl>0}_{\text{net}} \wedge \underbrace{PF>1}_{\text{收缩后}} \wedge \underbrace{\mathrm{DSR}>0}_{\text{非数据挖掘}} \wedge \underbrace{\alpha_{\text{vs B\&H}}>0}_{\text{有基准超额}}$$

并且在**提供了样本外窗口时**额外要求 $\text{val\_trades}>0 \wedge \text{val\_sharpe}>0 \wedge \text{val\_DSR}>0$（`:465-474`）。拒绝原因以字符串列表写入 `provenance.rejection_reasons`（`:478-495`）。$n_{\min}$ 由 `ga.min_champion_trades` 覆盖，默认 `MIN_CHAMPION_TRADES = 30`（`:427-429`）。注意 $PF$ 用的是**收缩后**的 PF（`train_result["profit_factor"]`，`:446`），所以 §1.4 的收缩直接决定这一步。

**精英保留（elitism）.** `GARunConfig.elite_count = 8`（`core/ga/evolver.py:56`），`_next_generation` 先 `copy.deepcopy` 种群前 8 个（`:535-537`），再补交叉/变异个体，最后追加 `immigrant_count = 8` 个随机移民（`:555-557`），并裁到 `population_size`。默认 `population_size=80, generations=30, tournament_size=3, mutation_rate=0.25, crossover_rate=0.7, early_stop_generations=10`（`:51-65`）。

**简约压力（parsimony pressure）.** `complexity_penalty`（`core/ga/fitness.py:194-223`）：

$$\text{penalty} = 0.8\,n_{\text{conditions}} + 1.2\,n_{\text{indicators}} + 0.3\,n_{\text{continuous params}}$$

其中 $n_{\text{conditions}}=\sum_g |g.\text{conditions}|$（structural 基因），$n_{\text{indicators}}$ 优先取 `indicator_genes` 中被打开的数量，否则从 continuous 基因名（`rsi_period → rsi`）去重计数，且排除 `ml`（`:206-217`）。

**已实测证据（简约压力真的起作用）.** 真实缓存 4 代：best fitness `5.9411 → 5.9411 → 9.9411 → 9.9411`（第 3 代 +4.00）。同时第 1 代最优与第 4 代领先者的**交易结果完全相同**（75 笔、Sharpe 9.4388、DSR 0.2119、最大回撤 0.12 %、收益 1.56 %），差别只在条件数 9→4、`complexity_penalty` 12.9→8.9，差值 4.00 **恰好等于** fitness 增量（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:31-42`）。默认参数配置独立复现同样的 +4.00（`:52`）。

**已知局限.**
1. 因此"fitness 上升"**不等于**"更赚钱"。证据索引自己写了这一点（`:42`："即'更简约'，不是'更赚钱'"）。这是本项目最重要的一条诚实结论。
2. 精英保留使 `best` **结构性单调不降**（`:43`）。
3. 默认配置下 mean fitness 反而**恶化**（−11.19 → −13.86 → −16.33，`:52`），所以"best 上升"不能被读成"种群整体变好"。
4. 罚项系数 $0.8/1.2/0.3$ 是手工值，没有标定证据。

### 1.8 按基因名的交叉与变异

**算法.** `_crossover`（`core/ga/evolver.py:570-635`）**按基因名**（不是 `zip` 位置）继承：

* continuous / categorical：`_inherit_genes_by_name`（`:30-48`）按名字并集，两父都存在时各 50 %，否则取存在者；
* indicator 布尔基因：按名字并集，两父都有时随机取一（`:589-596`）；
* structural（条件）基因：按名字并集，子代条件从**两个父代该基因条件的并集**中随机取 $n\sim U\{1,\dots,|\text{union}|\}$ 个（`:598-622`）——因此 `entry_long` 的条件**不会**混进 `exit_short`；
* `condition_logic` 基因按 50 % 从任一父继承（`:632-634`）。

**为什么需要.** 位置式 `zip` 假定两父基因顺序相同；当某个指标被关闭（丢掉它的基因）或变异插入新基因时，`zip` 会静默产出"8 个基因里保留 6 个、其中 1 个重复"的子代（`:571-578` 的 docstring）。条件按名字匹配还防止了"入场条件被放进出场基因"这一类语义错误。`random_chromosome` 的 `condition_logic` 以 2:1 偏向 `"or"`（`core/ga/genome.py:533`）。

**变异.** `_mutate`（`:637-…`）对 continuous 每基因 0.2 概率、categorical 0.1、structural 等分别变；具体分布见 `core/ga/genome.py` 的基因类。**未验证**：我没有逐类读完 `genome.py` 的 `mutate()`（该文件正被兄弟代理编辑）。

**已知局限.**
1. `random.sample` / `random.random()` 用的是**全局** `random` 模块；确定性依赖 GA 运行时对全局种子的一次性设置（`seed` 参数，`GARunConfig.seed`，`core/ga/evolver.py:64`）。证据索引称同 seed 两次运行逐位相同（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:50`），但这只在"没有其他代码消费同一全局 RNG"时成立。
2. 子代命名 `ga_child_{1000..9999}` **有碰撞可能**（`:629`），而 chunk 路径随后用 `random.randint(1000,9999)` 再重命名一次（`core/ga/fitness.py:627`）——这也是全局 RNG。

### 1.9 已实测数字（measured evidence, GA）

| 量 | 值 | 出处 |
|---|---|---|
| 真实缓存 4 代 best fitness | `5.9411 → 5.9411 → 9.9411 → 9.9411`（+4.00，第 3 代） | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:31-34` |
| 4 代 mean fitness | `−7.5083 / −6.4140 / −9.5826 / −3.9661` | 同上 |
| 参与交易基因 | `6/6, 5/6, 4/6, 5/6`（旧实现 1/20） | `:31-34`, `:38` |
| 总交易数 | `336 / 428 / 300 / 543` | `:31-34` |
| 冠军 | `fitness=9.9411`、75 笔、`dsr=0.2119`、`published=False` | `:36` |
| 唯一拒绝原因 | `alpha_vs_buy_hold=-52.20% <= 0` | `:31-36` |
| 墙钟 | 54.3 s（cap 470 s，未被杀）；另两次 94.9 s / 118.3 s | `:36` |
| 默认参数独立复现 | best `−4.0926 → −4.0926 → −0.0926`（第 3 代 +4.00）、交易 811/947/656、DSR 0.0762、82.2 s | `:52` |
| 第 1 代逐基因账本 | `75/60/103/45/32/21` 笔 | `:38` |
| 旧实现 before 数字 | "20 基因中 1 个交易 407 次、19 个 0 次"、"三代 best 恒为 −1.30"、"5 笔全胜 490.85 vs 200 笔 7.30" | `:48`（**未重跑旧代码**，来自 P1 之前的只读审计） |
| 复杂度罚项 before→after | 条件数 9→4、penalty 12.9→8.9、Δfitness 4.00 | `:42` |
| 确定性 | 同 seed 两次逐位相同（fitness/交易数/Sharpe/DSR/账本），仅墙钟不同 | `:50` |
| 零交易基因 | 第 4 代 1 个 immigrant，`flag=no_trades`，fitness −46.71；24 次评估 20 次交易 | `:44` |
| 多样性 | 第 4 代 6 个基因中 4 个交易结果逐位相同 | `:45` |
| 测试钉 | `tests/test_ga_credibility.py` **29** 项（我实测计数） | `grep -c '^\s*(async )?def test'` |

> ⚠️ 上表大部分是 **B 级**（归档的历史测量，审计 revision `3e90013` / `1a452ce` / `5f50771`，见 `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:3`）或 **B 级（仓库内 job 产物）**。本 revision `c703b8b` 上我**没有**重跑 `tools/ga_real_data_curve.py`（它需要 >400 s 墙钟与真实缓存，且会启动子进程），因此这些数字是"仓库已归档的实测"，不是"我这次跑出来的"。原始 JSON（`ga_curve_real*.json`）**不在仓库里**（证据索引 `:50` 自述在系统临时目录；`glob **/*ga_curve*` 为空）。

**仓库内的 job 产物（B 级，可直接打开核对）**

| 产物 | 内容 | 意义 |
|---|---|---|
| `data/ga_jobs/ga_4b84b23f.json.result:55-58` | `"fitness": -42.3`、`"sharpe": 0`、`"trade_count": 0`；`:5` `"enabled": true`；`:65` `avg_fitness: -46.575`；`:60` `elapsed 2.344 s` | **修复前**"发布了一个 fitness −42.3、0 交易的策略"的仓库内铁证 |
| `data/ga_jobs/ga_4b84b23f.json.log:3` | `best=-42.30 avg=-46.57 sharpe=0.00 trades=0 div=0.054 time=2s` | 同上，日志侧 |
| `data/ga_jobs/wf_86f8f5da.json.log:2` | `GA: population=50, generations=15, train=2025-05-01~2025-11-01, validate=2025-11-01~2025-11-01` | **单 bar 样本外窗口**的原文 |
| 同上 `:4-11`（第 1 代）与 `:16-23`（第 2 代） | 各 **8** 次 `A process in the process pool was terminated abruptly` | "每代 8 个 chunk 全部 abrupt"的仓库内铁证（payload `max_workers: 8`） |
| 同上 `:12`, `:24` | `Gen 1/15 \| best=-999.00 avg=0.00 sharpe=0.00 trades=0 div=0.000` | 崩溃后全种群 −999 |
| 同上 `:14`, `:26`, `:31`, `:32`, `:34`, `:35` | `Condition evaluation failed: 'cci < -100' — name 'cci' is not defined`（及 `'adx > 20'`、`'stoch_k < stoch_d'`、`'macd_histogram < 0'`、`'close < sma' → '<' not supported between instances of 'float' and 'function'`） | 孤儿条件缺陷的仓库内铁证 |
| `data/ga_wf_state.json:1-9` | `current_window: 0`、`total_windows: 6`、`ga_gen: 2`、`ga_total_gen: 15`、`completed: []` | WF 作业形状，与日志 `1/6` 一致 |
| `data/ga_jobs/` 清单 | 25 个 `wf_*.json`（其中 `wf_bad245f6.json` 是 `{"test": true}`）→ **24 个真实 WF job**；5 个 `ga_*.json` | "~24 个 walk-forward 任务"的仓库级佐证 |

**+4.00 的算术自洽性（子代理重算，A 级）.** $12.9 = 0.8\cdot9+1.2\cdot4+0.3\cdot3$、$8.9 = 0.8\cdot4+1.2\cdot4+0.3\cdot3$，且 $5.9411+12.9 = 9.9411+8.9 = 18.8411$ ✓；差值 $5\times0.8=4.0$ 恰为条件数 9→4 的罚项差。**这证明本 revision 的 `complexity_penalty` 与归档数字一致**，也间接证明 §1.3 的 $\sqrt{365}$ 读数是对的（若 alpha 项按某文档口径缩了 19 倍，fitness 增量就不会恰好是 4.00）。

**`5.00 vs 22.04`（doc 06:128）可复现（A 级）.** 用 `tests/test_ga_credibility.py:289-297` 的同一组输入（A = 5 笔 +100、B = 120 笔 +30 / 80 笔 −18）重算：A `fitness = 5.0`（`pf_term` 0.5、`raw_pf` 100.0、5 笔触发 `trades<15` 的 −5、观测数 2 < 20 故 alpha 项 = −max_dd = 0）、B `fitness = 22.0449`（`pf_term` 2.449、`raw_pf` 2.5、多空失衡 1.0 ⇒ −10）。**但"旧公式的 before 数字 490.85 无法复现**：留档的 `tools/p1_measure.py::_legacy_ranking()` 在同组输入上给 **A = 500.2 / B = 24.2**；要凑出 490.85 需要额外的复杂度罚项 $0.8c+1.2i+0.3p=9.35$，即 $16c+24i+6p=187$——左边恒为偶数、右边为奇数，**无整数解**。所以 490.85/7.30 只能归为 B 级（旧代码未重跑，`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:48` 自述）。


### 1.10 已知局限（GA）

1. **GA 与回测的杠杆口径不一致**（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:115`）：`config/risk_params.yaml` 是 `leverage: 2` / `max_leverage: 4`，而 `core/ga/**` 与 `core/backtest/engine.py` 中 `leverage` 命中 **0** 次（我按子代理的全仓 grep 核对过）——GA/回测按现金模型计价，实盘盈亏与回撤约为其 2×。未修。
2. **曲线上升主要来自简约压力**，不是 alpha（见 §1.7）。
3. **买入持有基准没有进入 fitness**（§1.3 的警告），尽管三处文档/docstring/配置注释都说它进了。
4. **PF 收缩、罚项系数、DSR 的 $\mathrm{Var}[\mathrm{SR}]=1$** 都是未标定的选择。
5. **DSR 独立性假设与 GA 高度相关的基因组相冲突**（见 §1.5 局限 2）。
6. **两个 $N$ 并存**：上报的冠军 DSR 用代内 $N$，provenance 用 population×generations（§1.6）。
7. **`fitness` 是量纲混合的加权和**，权重 $0.15/5/50/10$ 是遗留标定值（`DEFAULT_WEIGHTS` 的注释自称 "Legacy default weights"），`data/ga_fitness_weights.json` 不存在，所以没有任何标定证据被落地。
8. **`win_rate` 与 PF 正相关**，二者同时进入线性组合有重复计分。
9. **`max_dd` 以百分点直接扣**，与年化 Sharpe 量级的 DSR 项单位不同（§1.3）。
10. **`_finite` 把所有非法值吞成 0**（`:80-88`），因此一个损坏的评估会得到"中性"而非"最差"的分数（`-999` 只用于引擎返回 error 的路径，`:157`）。
11. **死开关（文档/配置声称生效但其实没有读取者）**：

| 开关 | 定义处 | 状况 |
|---|---|---|
| `PF_SHRINK = True` | `core/ga/fitness.py:39` | 无读取者（行为由 `profit_factor_shrunk` 硬编码实现） |
| `ga.alpha_weight` | `config/config.yaml:57`、`app/config.py:493-494` | 无读取者；生效值是字面量 `ALPHA_WEIGHT = 1.0`（`fitness.py:49`） |
| `ga.evaluation_leverage` | `config/config.yaml:61`、`app/config.py:495-496` | 无读取者；"现金模型"只因 `leverage` 在 GA/回测里 0 次命中而**偶然**成立 |
| `GARunConfig.overfit_penalty = 0.3` | `core/ga/evolver.py:61` | 无读取者 |
| `WEIGHT_GRID` | `core/ga/fitness_calibrate.py:30-35` | 无读取者（模块自述网格搜索已移除） |
| `total_trials()` | `core/ga/trial_counter.py:75-77` | 无生产调用者 |
| `_GARCH_MLE_X0` | `core/ml/volatility.py:690` | 无读取者（数值以字面量留在 `_garch11_grid_scan:717`） |

12. **provenance 的 `eval` 是硬编码的**：`core/ga/evolver.py:390` 写死 `{"engine_mode": "legacy", "use_live_spread": False}`，而多进程路径可以用别的 `engine_mode` 启动（`core/ga/fitness.py:872`、`core/ga/evolver.py:215`）——provenance 可能记录与实跑不符的引擎模式。
13. **未验证**：`core/ga/genome.py` 的 `mutate()` 逐类细节（文件正被兄弟代理编辑；子代理读过并给出 13 个指标名、`ml_weight` 被钉在 0.0–0.0、`condition_logic` 默认 `"or"` 且 2:1 偏向等），以及 `core/backtest/engine.py` 的 `per_genome_ledger` 全部实现（我只读了 `fitness.py` 侧的调用面，子代理给出了 `engine.py:696-764`、`:1219-1227`、`:1411-1428` 的槽位/账本/权益点三处）。

---

## 2. 机器学习（P2）

> 证据主文件：`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md` §2（`:60-69`）、`docs/core-algorithms/08-ml-triple-barrier.md`。

### 2.1 三重屏障标签与波动率缩放屏障

**算法.** `create_triple_barrier_label_vol`（`core/ml/labels.py:213-292`）对每一根有**完整前向窗口**的 bar 打标：

$$\begin{aligned}
w_i &= \mathrm{clip}\!\left(\frac{m_{ATR}\cdot \mathrm{ATR}_{14}(i)}{\text{close}_i},\; \text{min\_pct},\; \text{max\_pct}\right) &&(m_{ATR}=1.5,\;\text{min}=0.004,\;\text{max}=0.06)\\
\text{upper}_i &= \text{close}_i(1+w_i),\qquad \text{lower}_i=\text{close}_i(1-w_i)\\
\text{label}_i &= \begin{cases}
1 & \exists\, j\in(i,\;i+h]:\; \text{high}_j\ge \text{upper}_i \text{ 且先于下轨}\\
0 & \exists\, j\in(i,\;i+h]:\; \text{low}_j\le \text{lower}_i \text{ 且先于上轨}\\
2\ (\text{timeout}) & \text{两者都没触及},\; h=\texttt{forward\_periods}
\end{cases}
\end{aligned}$$

扫描区间是 `range(i+1, min(i+horizon, n-1)+1)`，且 `stop = max(start, n - horizon)`（`:271`），因此**最后 $h$ 行永远是 `NA`**——这是审计 P2 #6 的修复：修复前扫到 `n-1` 并把截断窗口强行填成 timeout，在 2 000 根合成数据上移动了 24 根 bar，在真实 8 845 根 BTC 1h 上摧毁了 16 个真实屏障触及（`docs/core-algorithms/08-ml-triple-barrier.md:181-186`；另见 `:183` 的"24 根 bar"/"16 个"）。

ATR 用 Wilder 平滑（`rolling_atr`，`core/ml/labels.py:149-160`）：$\mathrm{TR}_t=\max(H_t-L_t,\,|H_t-C_{t-1}|,\,|L_t-C_{t-1}|)$，$\mathrm{ATR}_t=\text{EWM}(\alpha=1/14,\;\text{adjust=False})$。

**为什么用波动率缩放.** 固定 `24 × 2 %` 的屏障约为 BTC 1h 中位 ATR% 的 7 倍，实测 **44.9 %** 的标签落进 timeout 类——那不是信号，是一个类别（`docs/core-algorithms/08-ml-triple-barrier.md:62-63`，`:136`）。

**三分类目标保留"无变动".** `create_three_class_label`（`core/ml/labels.py:73-115`）：

$$\text{fwd}=\frac{C_{t+h}-C_t}{C_t},\qquad \text{label}=\begin{cases}1 & \text{fwd}\ge \theta\\ 0 & \text{fwd}\le-\theta\\ 2 & |\text{fwd}|<\theta\end{cases},\qquad \theta=\max(\text{threshold},\;k\cdot c/100)\ \text{若给出 } c,k$$

旧实现把 $|\text{fwd}|<0.5\%$ 的行直接丢成 NaN（BTC 1h 只剩 40.5 %、1m 只剩 15.3 %），却在**每一根** bar 上做决策（`core/ml/labels.py:5-13`）。`decision_from_probs`（`:118-136`）把 `flat` 变成**弃权**（返回 0），并保留 $p_{up}\ge t_{up}\Rightarrow+1$、$p_{down}\ge t_{down}\Rightarrow-1$。

**可选 P3 路径.** `barrier_widths(..., vol_pct=…)`（`:163-210`）可用 `forecast_vol` 的条件波动率**替代** ATR 代理：`width = clip(vol_multiple × vol, min_pct, max_pct)`。`vol_pct=None`（默认，且所有现有调用都传 None）逐位走 ATR 路径（`:186-189`）。

**已知局限.**
1. 标签是"先碰哪个屏障"的分类，**不含路径**：止盈前先浮亏 5 倍风险仍记 1。
2. 屏障宽度只按 ATR（或一个 $\sigma$ 预报）缩放，不随信号强度调整。
3. O(n·h) 的 Python 内层扫描（`:272-285`），`max_rows` 参数用于研究时限界（`:222`）。
4. ⚠️ **`timeout_label` 是死参数，vol-scaled 路径根本不产生类 2**（我逐行核对过，子代理独立复核）。`create_triple_barrier_label_vol(..., timeout_label=2.0)` 的形参（`core/ml/labels.py:221`）在函数体里**从未被读取**：`:261` 初始化全 `NaN`、`:278-285` 只写 `1.0/0.0`、`:287-292` 直接 `result = pd.Series(labels, ...); return result`——没有 `fillna(timeout_label)`。后果：
   * vol-scaled 路径的 timeout 是 `NaN` 而不是类 2，因此 `class_distribution(...)["timeout_share"]`（`:295-306`，`shares.get(2, 0.0)`）在**该路径上恒为 0.0**；
   * `core/ml/predictor.py:462` 传 `timeout_label=2.0` 并把 `class_distribution(barrier)` 持久化进模型元数据（`:514-518`），所以落盘的 `barrier.distribution.timeout_share` 会读成 **0.0**；
   * 与之对照，**遗留**的定宽包装 `core/ml/features.py:954-959` **确实**填：`result = result.fillna(timeout_label)` / `result.loc[tail.index] = np.nan`；
   * `labels.py` 的 docstring `:236-240` 写"with `timeout_label` set, `NA` *inside* the sample **is filled** with the timeout class"，与同文件 `:289-291` 的注释"with `timeout_label` they are **not** filled"**直接冲突**——即**同一个函数里 docstring 与实现互相矛盾**；
   * 现有测试抓不到：`tests/test_ml_credibility.py:529-533` 只断言末 $h$ 行为 NA 且 `body.notna().sum() > len(body)-5`（在合成的 ATR 缩放序列上，几乎每根 bar 都会在 24 bar 内触到屏障，所以该不等式成立），`:555-559` 只断言 `timeout_label=None` ⇒ NA。

   于是 `docs/core-algorithms/08-ml-triple-barrier.md:186` 同时写了两种口径："`timeout_label=None` 时整类超时为 `NA`"（✔）与"`timeout_label` 设定时样本内 NA 被填为超时类"（✘）。**已提交内容**（`git hash-object core/ml/labels.py` == `git rev-parse HEAD:core/ml/labels.py` == `8d08e27b07984ae3006d1ceb69c1efd9bbfffe1f`，最后改动 `a9e549e`）。见 §11 D-19。
5. `timeout_label=2.0` 与**右边缘** NA 的区分：`:287-292` 的注释称"最后 $h$ 行在两种模式下都保持 NA"，这一点是对的；错的是"样本内 timeout 会被填"。

### 2.2 样本唯一性权重（sample uniqueness weights）

**算法.** `sample_uniqueness_weights(n, label_span)`（`core/ml/evaluation.py:171-207`）：

$$\begin{aligned}
c_j &= \max\!\Big(1,\ \#\{i:\ j\in[i,\min(i+s,n))\}\,\Big) &&\text{bar } j \text{ 被多少个标签覆盖}\\
u_i &= \frac{1}{s}\sum_{j=i}^{\min(i+s,n)-1}\frac{1}{c_j} &&\text{标签 } i \text{ 的平均唯一性}\\
w_i &= \max\!\Big(10^{-6},\; u_i\cdot\frac{n}{\sum_k u_k}\Big) &&\text{均值归一化：}\textstyle\sum_i w_i=n
\end{aligned}$$

`s = label_span`。辅助量：`average_label_overlap(n,s) = (s-1)/s`（`:158-168`），$s=4 \Rightarrow 0.75$——即"有效样本量 ≈ N/4"。

**为什么需要.** 4 倍重叠的标签不能算 4 个独立观测。权重被传给模型拟合（`model_factory(Xf, yf, w_fit)`，`core/ml/credibility.py:571-572`）以及净期望加权平均（`net_expectancy(..., weights=w)`，`:338-359`）。

**实测范围.** 文档字符串记录 $s=4$ 时权重范围是 $[0.9997,\ 2.0827]$——**不是** $(0,1]$：样本边缘的标签被更少的并发窗口覆盖，必须被**上调**才能保持总有效计数诚实（`core/ml/evaluation.py:183-188`）。

**已知局限.**
1. 权重上界大于 1，因此模型拟合会**放大**边缘样本；这不是 AFML 原书的标准形式（原书用 $1/c_i$ 的平均，通常 ≤ 1）。
2. `concurrency` 计算是 O(n·s) 的 Python 循环（`:196-203`），与 `label_concurrency`（`:148-155`）重复实现。
3. 权重只反映"标签窗口重叠"，不反映**序列相关**或异方差。

### 2.3 Purged K-Fold 与 embargo

**算法.** `purged_kfold_splits(n, k, label_span, embargo)`（`core/ml/evaluation.py:41-85`）：

1. 时间顺序切成 $k$ 个连续块，第 $f$ 块为 test（`bounds = np.linspace(0, n, k+1)`）；
2. **purge**：训练行 $i$ 被剔除，若其标签窗口 $[i,\ i+s)$ 触到 test 块（`ends >= lo & all_idx < lo`）；
3. **embargo**：test 块之后 $[hi,\ hi+\text{emb})$ 的行也被剔除（$\text{emb}=s$ 默认，`:61`）。

`label_spans(n,s) = min(arange(n)+s, n-1)`（`:35-38`）。`combinatorial_purged_splits`（`:88-127`）是同一 purge 在分组组合上的版本（`n_groups` 选 `k_test`）。

**为什么需要（量化过的泄漏）.** 旧的时序 80/20 切分让训练集与测试集**共享 2–9 根前向窗口 bar**，75–95 % 的标签重叠（`core/ml/evaluation.py:3-6`）。`overlap_count`（`:130-145`）就是"旧切分会泄漏多少行"的度量。

**已知局限.**
1. `purged_kfold_splits` 在 $n\le 2k$ 时返回 `[]`（`:62-63`），调用方必须处理空列表（`evaluate_model_oos` 检查 `if not splits`，`core/ml/credibility.py:554`）。
2. embargo 只加在 test 块**之后**（`:80`），不加之前；对"标签窗口向前看"的泄漏这是正确的，但对"特征本身有前视"的泄漏无效。
3. 折内训练/测试的时间顺序并非严格"训练全在过去"：中间折的 test 两侧都有训练数据（文档字符串承认："the test set is always a genuine future window relative to **most** of its training data"，`:55-56`）。

### 2.4 Isotonic 概率校准

**算法.** `ProbabilityCalibrator`（`core/ml/calibration.py:136-224`）。isotonic 分支用 `sklearn.isotonic.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")`（`:173-177`）；Platt 分支用 `LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)`，自变量是 logit（`:178-182`）：

$$\text{logit}(p)=\ln\frac{p}{1-p},\qquad p_{\text{cal}}=\sigma\!\big(a\cdot\text{logit}(p)+b\big)$$

**降级规则.** 拟合样本 $<30$ 或单一类别时 `fitted=False`，`transform` 返回**恒等**（`:169-171`, `:187-190`）——绝不静默返回 0.5。

**为什么需要.** 部署的旧模型是**反校准**的：审计实测 $\hat p=0.91$ 的分箱实际上涨率只有 0.22，OOS AUC 0.396–0.447；而融合核把 $(p-0.5)\times2$ 直接当分数，方向是**反的**（`core/ml/calibration.py:3-8`）。融合分数现已改为以模型自身基准率为中心（`signed_score`，`:229-276`）：

$$\text{edge}=\frac{p}{b}-1,\quad \text{scale}=\max\!\left(\frac{1}{b}-1,\ \frac{1}{1-b}-1\right),\quad \text{score}=\mathrm{clip}\!\left(\frac{\text{edge}}{\text{scale}},-1,+1\right)$$

$b<10^{-9}$ 或非有限时退回 0.5（`:270-272`）。$b=0.5$ 时它恰好等于 $(p-0.5)\times2$——这是向后兼容点；$b=0.2$ 时 $p=1.0$ 得 $+1.0$ 而 $p=0.0$ 只得 $-0.25$（**不对称**，`:243-248`）。

**已知局限.**
1. 真实数据 AUC ≈ 0.52 时校准**无法**制造单调曲线——单调性只在带信号的合成模型（AUC 0.815）上被证明（`docs/core-algorithms/08-ml-triple-barrier.md:171-175`）。
2. `to_dict` 持久化 isotonic 的 `X_thresholds_/y_thresholds_` 后，`from_dict` 用这些阈值**重新 fit** 一个 isotonic（`:217-223`）——这不是无损往返（isotonic 的 fit 是单调回归，用阈值点再 fit 通常等价但不保证逐位相同）。**未验证**：我没有做逐位往返比较。
3. `reliability_curve` 的 `monotone` 判据含经验常量 `np.diff(obs) >= -0.05` 与 `spearman >= 0.7`（`:120`），是放宽的判据。

### 2.5 成本感知阈值选择（嵌套 / nested）

**算法（双边，方向模型）.** `cost_aware_threshold`（`core/ml/credibility.py:362-490`）：

$$\begin{aligned}
\text{grid} &= \{0.30,0.31,\dots,0.95\}\\
\text{take}^{\text{long}}(t) &: p\ge t,\qquad \text{take}^{\text{short}}(t): p\le 1-t\\
\text{net} &= \text{side}\cdot r - \frac{c}{100}\\
E_{\text{long}}(t) &= \overline{\text{net}}\big[\text{take}^{\text{long}}(t)\big],\qquad E_{\text{short}}(t)=\overline{\text{net}}\big[\text{take}^{\text{short}}(t)\big]\\
t^\star &= \arg\max_{t,\,E(t)>0,\,n(t)\ge n_{\min}} E(t)
\end{aligned}$$

阈值网格见 `:410`（`np.round(np.arange(0.30, 0.951, 0.01), 4)`）；两个候选都必须满足 $n(t)\ge$ `min_trades` **且** $E(t)>0$（`:426-431`，"least bad 不是 edge"）；两侧是**独立决策带**，获胜阈值只写进自己那一侧（`:432-446`，修掉了"short 胜出却把 long 字段写成 $1-t$"的 bug）。同时返回 breakeven 阈值（`:447-462`）与所选侧的 `net_trade_stats`（`:484`）。

**嵌套协议（消除选择乐观）.** `evaluate_model_oos`（`:495-699`）每折：

1. 该折训练块**尾部 20 %** 作为**该折自己的校准流**（`cal_cut = int(len(train_idx)*0.8)`，`:566-570`）；
2. 折模型只在 `Xtr[:cal_cut]` 上拟合，样本权重用 `sample_uniqueness_weights(cal_cut, label_span)`（`:570-572`）；
3. 校准器只用 `p_cal_raw`（该折模型**没见过**的行）拟合（`:582-588`）；
4. 阈值在**折内校准流**上选（`:592-594`），再作用到该折 test 行（`:596-604`）；
5. 门读的是这些外层交易的汇总（`net_expectancy_oos` / `n_trades_oos` / `t_stat_oos` / `psr_oos`，`:663`, `:685-690`）。

池化搜索（在同一批被汇报的行上选校准器+阈值）**仍然计算**，但只作为对照报出，标记 `selection = "pooled_optimistic"`（`:670-675`）。折内交易下限按流大小缩放：$\min(n_{\min}, \max(50, \lfloor n_{cal}/10\rfloor))$（`fold_min_trades`，`:702-712`），理由是不缩放下每折都会弃权、真 edge 被藏起来。

**为什么需要（量化过的选择偏差）.** 修复前：`calibrator.n_fit == n_oos == 600` 而 `sum(n_cal) == 477`（校准流是死代码）；in-fit ECE 0.0000 对真留出 0.0784；半样本外实验把 OOS 净期望翻到 **−0.075 %（BTC）/ −0.342 %（ETH）**（`core/ml/credibility.py:30-43`）。修复后逐折流内期望为正（BTC fold 0 **+0.3113 %**）而套到测试行上是 **−0.2984 %**——这个差就是选择偏差的量级（`docs/core-algorithms/08-ml-triple-barrier.md:164-165`）。修复后 `calibrator_n_fit` 之和 = `n_cal` 之和：BTC **2862 == 2862**、ETH **3867 == 3867**（同上 `:162-163`）。

**已知局限.**
1. 折内校准流是**训练块的尾部**，时间上紧邻 test 块，可能与时序结构耦合（不是随机留出）。
2. 单个折选出 `side=None`（无正期望候选）时该折完全弃权（`:602-604`），因此外层样本量会随折数下降——外层交易数低于门限时无法区分"没有 edge"与"没有足够数据"。
3. 阈值网格固定在 $[0.30,0.95]$，步长 0.01；`cost_aware_threshold` 的 docstring 明确把选择乐观的责任交给调用方（`:395-397`）。
4. **可部署阈值的来源是"投票 + 中位数"**：`_fold_threshold_summary`（`:715-744`）按多数票选边，再取该边阈值的**中位数**。这本身是一个未评估的启发式。

### 2.6 外层门（the outer gate）

**算法.** `credibility_gate`（`core/ml/credibility.py:761-892`）。全部条件为**合取**：

$$\text{allowed} \iff \underbrace{\mathrm{AUC}>\text{0.55}}_{:833} \wedge \underbrace{E_{\text{net}}>0}_{:835} \wedge \underbrace{n_{\text{oos}}\ge100}_{:831} \wedge \underbrace{n_{\text{trades}}\ge100}_{:841} \wedge \underbrace{t>2.0}_{:866} \wedge \underbrace{\mathrm{PSR}\ge0.95}_{:866}$$

常量：`GATE_AUC_MIN=0.55`、`GATE_MIN_NET_EXPECTANCY=0.0`、`GATE_MIN_TRADES=100`、`GATE_MIN_T_STAT=2.0`、`GATE_MIN_PSR=0.95`（`:59-70`）。**缺失 t 或 PSR 一律拒绝**，理由串点名缺失项（`:860-865`，审计 F3）："no significance evidence (t_stat missing; the gate requires t > 2.00 AND PSR >= 0.95)"。`enabled = allowed`（`:873`）。

**为什么是 AND 而不是 OR（审计 F3 + 复核 R4）.** 修复前实现用 `or`，而在正态下 $\mathrm{PSR}\ge0.95$ 恰好等价于 $t\ge1.645$（实测 $t=1.65,\ \mathrm{PSR}=0.9505$ **被放行**），于是 2.0 的 t 下限被悄悄放宽 18 %（`:792-796`，`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:132`）。AND 后两个下限**真正不冗余**——因为 PSR 现在真的读偏度/峰度（见 §2.7）：厚左尾在 $t=2$ 时把 PSR 压到 **0.9479777894541446 < 0.95 → 拒绝**，而正态近似报 0.9772498680518209（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:168`）。边界由 `test_significance_floors_are_a_conjunction_at_the_boundary` 钉住（$t=1.60/1.65/1.70/2.00$ 拒绝，$t=2.01$ 放行，$t=3.0\wedge\mathrm{PSR}=0.94$ 拒绝）。

**AUC 用未校准概率.** 门的 AUC 是 `metrics_raw["auc"]`（校准**无关**，`:643-647`），理由写在注释里：校准器无法抬高它，所以它是诚实协议唯一搬不动的指标。

**门槛的成本来源.** `cost_pct_for` → `round_trip_cost_pct_from_quote`（`:119-199`）从 **sim 成本模型**解析（`app.config.sim_*`），行情单边收 `fee + half_spread + slippage_bps/100`，往返 ×2：

$$c_{\text{market}} = 2\left(f + \frac{s}{2} + \frac{\text{slip}_{bp}}{100}\right),\qquad c_{\text{limit}}=2f_{\text{maker}}$$

（`:130-175`；BNB 折扣把 fee × 0.75，`:168-169`）。审计 P2 #2 的修复动机：修复前读 `backtest_taker_fee_pct`（0.04 %）给出 0.13 %/0.14 %，而 sim 实际收 **0.28 %/0.30 %** 往返，即门槛被喂了便宜 2 倍的成本；ETH 的 "+0.058 %" 在真成本下是 **−0.062 %**（`:189-194`）。

**已知局限.**
1. 门的每一个下限（0.55 / 100 / 2.0 / 0.95）都是**约定值**，没有从数据推导或做功效分析。`GATE_MIN_TRADES=100` 的理由是"20 笔不能支撑判决"（审计 P2 #3：BTC 最佳候选 27 笔 / t=1.28 / 95 % CI [−0.19 %, +0.91 %]，`:61-63`）。
2. `n_oos >= min_oos` 用的是**汇总** OOS 行数（`:819`），而 `n_trades` 是外层交易数；两者是不同的统计总体，门把它们并列。
3. `net_trade_stats` 的 `t` 与 `psr` 都用**未加权**的净收益序列（权重只影响 `mean`，`:238-253`），即唯一性权重不进入显著性检验。这是**刻意**的（"t_stat/se 保持普通正态理论数字"，`:226-228`），但不一致。
4. **试验次数没有进入 ML 门**：DSR 的多重检验校正只存在于 GA（`core/ga/fitness.py`），ML 门只做单模型判决。跨币种/跨周期的模型扫描没有被计数。
5. 门的 PSR 用**每笔交易**净收益（不是每期/年化 Sharpe），因此"$N$"是交易数而不是期数；这与 GA 的 DSR 口径**不同**，两处不可直接比较。

### 2.7 PSR：Prado 偏度/峰度修正标准误

**算法.** `probabilistic_sharpe(n, mean, sd, benchmark=0.0, returns=None)`（`core/ml/credibility.py:256-312`）：

$$\begin{aligned}
\mathrm{SR} &= \frac{\mu - b}{\mathrm{sd}_r},\qquad
\gamma_3 = \frac{\frac1n\sum (r_i-\bar r)^3}{\mathrm{sd}_r^3},\qquad
\gamma_4 = \frac{\frac1n\sum (r_i-\bar r)^4}{\mathrm{sd}_r^4}\\
\text{bracket} &= 1 - \gamma_3\,\mathrm{SR} + \frac{\gamma_4-1}{4}\mathrm{SR}^2\\
\mathrm{SE}_{\text{adj}} &= \mathrm{sd}_r\sqrt{\frac{\text{bracket}}{n-1}}\\
\mathrm{PSR} &= \Phi\!\left(\frac{\mu-b}{\mathrm{SE}_{\text{adj}}}\right)
\end{aligned}$$

`sd_r` 用 `ddof=1`（`:300`），$\gamma_3,\gamma_4$ 用 `ddof=0`（`:303-304`，`mean` 型矩）。**退化路径**：`returns=None`、有效样本 $\le3$、`sd_r<=0`、bracket 非有限或 $\le0$ → 保持正态近似 $\mathrm{SE}=\mathrm{sd}/\sqrt n$（`:295-309`）；$n<2$ 或 $\mathrm{sd}\le0$ → 返回 0.0（`:291-292`）。正态样本（$\gamma_3=0,\gamma_4=3$）时 bracket 恰为 1，结果与旧式 $\Phi(\sqrt n\,\mathrm{SR})$ **逐位相同**。

**接入点.** `net_trade_stats` 与 `_signed_net_stats` 把**净收益序列**传进去（`:252`, `:332`），因此 PSR 读的是真实尾部而不是汇总统计量。

**为什么需要（诚实性事件）.** 复核发现 R4：修复前 docstring **声称**读偏度/峰度，实现却是 $\Phi(\text{mean}/\mathrm{se})$，于是 AND 门恰好等价于 $t>2$，`PSR>=0.95` 是死条件。现在文档与实现一致（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:134`, `:168`）。实测值（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:168`，`:185`）：

| 样本 | $t$ | corrected PSR | 正态近似 | 门 |
|---|---|---|---|---|
| 正态 | 2.0000 | 0.9772498680518209 | 0.9772498680518209（退化为同一值） | 放行 |
| 厚左尾（10 个 −40σ） | 2.000000 | **0.9479777894541446** | 0.9772498680518209 | **拒绝** |
| 同形状，目标 $t=3$ | 2.9999999999999996 | 0.9869562688374416 | 0.9986501019683699 | 放行 |

**已知局限.**
1. `benchmark=0` 默认，PSR 的备择假设是"真实均值 > 0"，而不是"优于基准"。
2. 修正公式是**渐近**的（Bailey–López de Prado 2012）；$n$ 小时 $\gamma_4$ 的抽样误差很大（本项目的 ML 外层交易数常在 100–1700 之间）。
3. `core/strategy/pairs.py` 调用 `probabilistic_sharpe` 时**不传 returns**（`:284-286` 的注释明确列出这一类调用方），因此配对路径上的 PSR 是正态近似。
4. 三个入口（`core/ml/credibility.py:746-…`、`:803`、`:856` 附近的 docstring）曾经写错数值（`0.937`），已在复核 L3 改为实测值（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:185`）。

### 2.8 版本化特征契约与 schema hash

**算法（HEAD `c703b8b` 的提交内容，非工作树）.**

```python
def feature_schema_hash(feature_names=None) -> str:            # core/ml/features.py:276-279 (HEAD)
    names = list(FEATURE_NAMES if feature_names is None else feature_names)
    return hashlib.sha1(json.dumps(names).encode("utf-8")).hexdigest()[:12]
```

即 $H=\text{sha1}(\text{JSON}([\text{names}]))[:12]$。我在 HEAD 的 `features.py` 上独立复算过（见 `docs/core-algorithms/08-…` 与证据索引引用的 `335e63360104`）：

* 提交的 `DEFAULT_FEATURES` 含 **40** 个字面量：其中第 31 项是裸 `"hurst"`；39 项 = 去掉 `"hurst"` 后的列表（代码注释 `features.py:59-61` 声称"bare `hurst` was dropped"，但字面量仍在列表里）；
* $\text{sha1}(\text{json}(40\ \text{项}))[:12] = $ **`70899cff156d`**；
* $\text{sha1}(\text{json}(39\ \text{项，无 hurst}))[:12] = $ **`335e63360104`** ← **正好等于**三处独立记载的值：证据索引（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:62`）、`tests/test_final_audit_fixes.py:39`（`FEATURE_HASH = "335e63360104"  # core.ml.features.feature_schema_hash(FEATURE_NAMES)`）、以及工作树 v2 保留的冻结字面量 `FEATURE_SCHEMA_V1_HASH`。

**这是一处真实的代码/文档不一致**，见 §11 D-4。**P6-B v2 契约**（工作树，编辑中）为 v1 + 15 列量能/资金流族（`core/ml/features.py:81-96`，工作树），`FEATURE_SCHEMA_VERSION = 2`、`FEATURE_SCHEMA_V1_HASH = "335e63360104"` 作为**冻结字面量**保留（工作树 `:98-110`），理由是"`DEFAULT_FEATURES` 一旦增长，v1 哈希就**无法**再被重算"。子代理在我之后读到工作树的 `DEFAULT_FEATURES` 为 **54** 列（39 + `volr_5/10/20/60`、`volz_60`、`vwap_dev_20`、`vwap_dev_session`、`flow_close_position_weighted`、`obv_slope_10`、`ad_slope_10`、`flow_cmf_20`、`flow_mfi_14`、`flow_amihud_20`、`flow_vol_price_corr_20`、`flow_vol_centroid_20`），而我 mid-edit 撞到的 `FeatureContractError` 声称 `expected 52`——**52 与 54 都是"编辑中的中间态"，不是契约主张**。


**为什么需要.** 修复前 "live trains on 40, backtest uses 47" 的分叉：部署的 pkl 带 `n_features_in_=40` 而回测矩阵有 47 列（HEAD `core/ml/features.py:4-8`，工作树 docstring 保留同一段）。契约唯一化 + `compute_features` 拒绝返回短矩阵（`validate_feature_matrix`）使 v1 模型在重排矩阵上被**按名字拒绝**而不是静默打分。侧车校验链见 §9 / 证据索引 F4（`:128`）：sidecar 存在 → `gate` 存在 → `gate.allowed` → `feature_names` 等于契约 → `feature_schema_hash` 相等，缺 hash 也拒绝（复核 R5，`:169`）。

**测试钉.** `scripts/ml_credibility_measure.py` 与 `core/ml/predictor.py` 都消费该 hash；本仓库 `data/models` 实测 **15 个 `.pkl`、0 个 `_meta.json`**（我本次实测，与证据索引 `:96`、`:128` 一致）。

**已知局限.**
1. hash 是 sha1 截断到 12 位十六进制（48 bit）——用于**误配检测**足够，不是安全用途。
2. hash 只覆盖**列名列表**：列的顺序/类型/单位都不在哈希里（名字相同但语义改变不会被发现）。
3. 契约演进要求把旧 hash 当**字面量**保存（工作树注释 `:106-110` 自述），因此 v1→v2→v3 会线性堆积常量，且不能自动验证"这个字面量真的是当年的列表"——我这次是手工复算才确认 `335e63360104` 对应 39 列（不含 `hurst`）。
4. **未验证**：工作树的 v2 契约在写作时**无法导入**（见 §2.9），因此 v2 的 52/54 列数、hash、以及"v1 模型被拒"的行为我都没能实测。

### 2.9 本 revision 上无法复现的部分（已实测记录）

**我实际运行的命令与结果**（本 revision `c703b8b`，盘上被兄弟代理编辑中）：

```
$ python scripts/ml_credibility_measure.py --symbols ETHUSDT --intervals 1h --tail 9000 --out %TEMP%\mlmeas
Traceback (most recent call last):
  File "...\scripts\ml_credibility_measure.py", line 416, in main
    df, ind, X = prepare(symbol, interval, args.tail)
  File "...\scripts\ml_credibility_measure.py", line 65, in prepare
    X = compute_features(ind)
  File "...\core\ml\features.py", line 753, in compute_features
  File "...\core\ml\features.py", line 306, in validate_feature_matrix
core.ml.features.FeatureContractError: feature matrix is missing 13 required column(s):
['volr_5', 'volr_20', 'volr_60', 'volz_60', 'vwap_dev_20', 'vwap_dev_session',
 'obv_slope_10', 'ad_slope_10']... (have 39, expected 52)
```

另外，`core/ml/features.py` 的盘上内容在写作期间曾**无法通过 `ast.parse`**（`SyntaxError: unterminated string literal (detected at line 214)`），`git status --porcelain` 显示 `M core/ml/features.py`（唯一被改动的文件）。

**结论.** §2.10 的 ML 数字是**仓库归档的测量记录**（在冻结 revision 上产生），**不是**我在 `c703b8b` 上重跑的。任何"当前 ML 门仍然拒绝"的陈述都依赖于：(a) `ml.enabled: false` 的配置事实（`config/config.yaml:80`，我本次读到），以及 (b) `data/models` 15 个 pkl / 0 个 sidecar → 全部被拒绝的实测（证据索引 F4，`:128`）。**P6-B v2 契约下门是否仍拒绝：未验证。**

### 2.10 诚实的实测判决（verdicts）

**ETHUSDT 1h 被拒（证据索引 §2，`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:66-69`）**

| 口径 | 基率 | 多数类准确率 | 模型准确率 | AUC | Brier | log loss | 净成本期望 |
|---|---|---|---|---|---|---|---|
| legacy（时序 80/20、±0.5 % 二分类） | 0.2227 | 0.7773 | **0.6597** | 0.5621 | 0.2160 | 0.6224 | −0.041 %（阈值 0.5） |
| P2（purged K-fold + embargo + 唯一性权重 + isotonic） | 0.5068 | 0.5068 | 0.5159 | **0.5342**（5 折均值 0.5379） | 0.2536 | 0.8371 | **−0.1822 %** |

门（原文引用）：`OOS AUC 0.5342 <= 0.55; net expectancy -0.1822% <= 0.0000%（0.2600% 往返成本后）; not significant (t=-2.28 <= 2.00, PSR=0.011 < 0.95)`；OOS 样本 **4 840**、OOS 交易 **585**。

**BTCUSDT 1h 同样被拒**（`:69`）：legacy 准确率 **0.6834** vs 多数类 **0.8140**，AUC 0.5672；P2 门 **AUC 0.5207**、净期望 **−0.2494 %**、**t=−4.46**。（`docs/core-algorithms/08-ml-triple-barrier.md:154-155` 给出配套的笔数与 CI：BTC 1365 笔 / [−0.359 %, −0.140 %]；ETH 585 笔 / [−0.339 %, −0.026 %]。）

> **注意（两处文档口径差）**：`docs/core-algorithms/08-ml-triple-barrier.md:154-155` 把 legacy 准确率写作 **0.6702（BTC）/ 0.6408（ETH）**，而 `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:69` 写作 **0.6834 / 0.6597**。四个数字互为同族但不同值，见 §11 D-3。

**方向不可预测的独立复现**（`:73`）：OOS AUC **0.5207 / 0.5342**，与 P3 测试文档所述 0.52/0.53 一致。结论：**ML 保持关闭是实测结论，不是默认值**（`:69` 原文）。

**特征/管道成本与标签侧的数（P2 #6/#7）**（`docs/core-algorithms/08-ml-triple-barrier.md:187-196`）：滚动 Hurst 有界化后 `compute_all(REQUIRED_INDICATORS)` **16.402 s → 0.122 s**、`compute_features` **0.219 s → 0.960 s**、合计 **16.62 s → 1.08 s（15.4×）**，Hurst 一项 12.89 s → 0.96 s；回归断言 `test_feature_pipeline_cost_is_bounded` 绑 3.0 s/8 845 根（当前 1.08 s，约 2.8× 余量）。固定 2 %/24 h 屏障的超时类占 **44.9 %**（`:136`）。

**测试钉.** `tests/test_ml_credibility.py` **53** 项、`tests/test_engine_ml_gate.py` **7** 项（证据索引 `:12`）；我实测 `tests/test_ml_credibility.py` 的 `def test` 计数为 **53**，与文档一致。

---

## 3. 波动率（P3）

> 证据主文件：`docs/core-algorithms/10-volatility-targeting.md`、`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md` §3（`:71-73`）、`tests/test_gap_fixes.py`。

### 3.1 RiskMetrics / EWMA 递归与半衰期

**算法.** `ewma_variance`（`core/ml/volatility.py:508-551`）：

$$\sigma^2_{t+1}=(1-\lambda)\sum_{i\ge0}\lambda^i r_{t-i}^2 \quad\Longleftrightarrow\quad v \leftarrow \lambda v + (1-\lambda)r_t^2,\qquad v_0=\mathrm{Var}(r_0,r_1)_{\text{ddof}=0}$$

**种子只用前两个观测**（`:545-548`），因为用全样本方差会在递归最早一步泄漏未来信息。$\lambda$ clamp 到 $[0,0.9999]$（`:544`）。

**半衰期.** 让权重衰减到一半所需的 bar 数：

$$k_{1/2} = \frac{\ln 2}{-\ln\lambda}\quad\Big|_{\lambda=0.94} = \frac{0.693147}{0.061875} \approx 11.2\ \text{bars}$$

代码侧没有把半衰期写成常量，而是把**有效记忆**写成 $1/(1-\lambda)$：

$$\frac{1}{1-\lambda}\Big|_{\lambda=0.94} = \frac{1}{0.06} \approx 16.7\ \text{bars}$$

（`core/ml/volatility.py:522-523` 的注释："the **effective** memory: $1/(1-\lambda)\approx16.7$ bars … so `window` is only the code-side cap on the recursion, not the horizon that matters"）。两个数都合法但不是同一个量：$k_{1/2}\approx11.2$ 是**权重**半衰期，$1/(1-\lambda)\approx16.7$ 是 e-folding 尺度。

$$\text{种子残余权重} = \lambda^{500} = 0.94^{500} \approx 3\times10^{-14}\ (\text{见 } :519)$$

**为什么用它.** 它是 RiskMetrics 的文档化默认（λ=0.94，"稳健、无需调参"，`:101-104`），且在 1h 数据上要 3 周（500 bar）才忘记起点（`:106-109`）。

**已知局限.**
1. `window` **不是**估计器的记忆，只是代码侧的截断；同一个 $\lambda$ 下 `window=500` 与 `window=400` 应该给出近似相同的结果——实测在修复锚定后两者相对差 **4.2e-12**（`core/ml/volatility.py:531-538`），而在锚定修复前它们是 `0.524216` 对 `0.524062` %/bar（差别来自**clip 的限值**而不是递归）。
2. λ 固定 0.94，没有拟合也没有随周期调整（`DEFAULT_LAMBDA` 是常量）。
3. `ewma_variance` 是**逐 bar Python 循环**（`:549-550`），这是它成本的主要来源（见 §3.9）。
4. `ewma_vol_series`（`:564-597`）是滚动版本，但**实盘路径不走它**：`RiskManager` / `PositionGuard` 直接调标量 `forecast_vol` 并各自做 TTL 缓存（`:576-583` 的自述；`_VOL_CACHE_TTL_SEC = 300.0`，`core/risk/manager.py:27`）。

### 3.2 GARCH(1,1) 自由 ω MLE

**算法.** `garch11_params` → `_garch11_scipy_mle`（`core/ml/volatility.py:815-879`）→ `_garch11_mle`（`:733-812`）。模型与似然：

$$\begin{aligned}
v_t &= \omega + \alpha\,x^2_{t-1} + \beta\,v_{t-1},\qquad x=r\cdot100\ (\text{percent})\\
-\ell(\omega,\alpha,\beta) &= \tfrac12\sum_t\left(\log v_t + \frac{x_t^2}{v_t}\right) &&\text{(}= \texttt{garch11\_loglik\_grad}[0]\text{)}\\
\text{persistence} &= \alpha+\beta \le \texttt{GARCH\_MAX\_PERSISTENCE}=0.999,\quad \omega>0,\ \alpha,\beta\ge0
\end{aligned}$$

似然与梯度的精确递推（`:626-632` 的 docstring，实现 `:649-679`）：

$$\begin{aligned}
v_t &= \omega+\alpha\,x^2_{t-1}+\beta v_{t-1}\\
\frac{\partial v_t}{\partial\omega} &= 1+\beta\frac{\partial v_{t-1}}{\partial\omega},\quad
\frac{\partial v_t}{\partial\alpha}=x^2_{t-1}+\beta\frac{\partial v_{t-1}}{\partial\alpha},\quad
\frac{\partial v_t}{\partial\beta}=v_t+\beta\frac{\partial v_{t-1}}{\partial\beta}\\
\frac{\partial(\tfrac12\ell)}{\partial p} &= \tfrac12\left(1-\frac{x_t^2}{v_t}\right)\frac{\partial v_t}{\partial p}
\end{aligned}$$

**两个"技巧"（都不是可选装饰）.**

1. **方差目标化（variance targeting）种子.** 粗网格扫描时**强制** $\omega = V(1-\alpha-\beta)$，$V=$ 样本方差（`:707-730`）：`np.arange` 把 `_GARCH_GRID_ALPHA = (0.01, 0.20, 0.05)` 与 `_GARCH_GRID_BETA = (0.60, 0.99, 0.05)` 展开成 $\alpha\in\{0.01,0.06,0.11,0.16\}$（**4** 个）× $\beta\in\{0.60,\dots,0.95\}$（**8** 个）⇒ **32** 个候选，再跳过 $\alpha+\beta\ge0.999$ 与 $\omega\le0$（`:721-724`）。理由写在 `:691-697`：500 bar 窗口上的 GARCH 似然面**平坦且多峰**——实测 SLSQP 从三个相近起点落到三个不同点，$\alpha$ 跨 0.050…0.061，而 $0.5\Sigma LL$ 互差 < 0.4。确定性扫描 + 一次局部精修给出"可证明的最好点"，且时间是固定的。
   ⚠️ `:713` 的 docstring 写 "O(20 × 8) likelihood passes"，而实际网格是 **4 × 8**——注释高估 5×（见 §11 D-27）。
2. **参数箱 + 尺度归一化.** 优化在 $z=x/\mathrm{sd}(x)$ 上做（`:775-778`），箱是 $\omega\in(10^{-12},\,10\cdot\mathrm{Var}(z)]$（`:782`）、$\alpha,\beta\ge0$、$\alpha+\beta\le0.999$（`:786-790`）。理由（`:744-754`）：似然唯一的尺度承载项是截距，而箱是用数据单位表达的；因此拟合 $x$ 与 $100x$ **不是**同一个优化——实测百分² 拟合收敛到 $\omega=0.32$ 且目标函数为正的大值，而同一模型在分数² 下收敛到正确最优点。回代是精确的：$\omega_x=\omega_z\cdot\mathrm{Var}(x)$（`:810-812`）。
3. **方差下限.** `garch11_filter` 把 $v$ 下限设为 $\mathrm{Var}(x)\times10^{-4}$（**相对**下限，`:896`）。绝对下限被试过且是错的：$v\to0$ 时 $x^2/v$ 爆炸，退化参数对的平均似然实测到 $3.6\times10^5$，于是**下限本身成了最优点**，所有优化器都往那里走（`:890-894`）。
4. **优化器选择.** Nelder-Mead，`maxiter=4000, maxfev=4000, xatol=1e-10, fatol=1e-12`（`:795-797`）；SLSQP 在同一平坦面上实测 4–7 s/次且落点取决于种子，Nelder-Mead 约 0.2 s（`:766-769`）。

**一步预报是"混合"的.** `garch11_forecast`（`:1034-1068`）与 `_garch11_variance`（`:1071-1138`）：

$$\hat v = w\,v_{t+1} + (1-w)\,\underbrace{\max(v_t,\ v_{\text{EWMA}})}_{\text{base}},\qquad w=\texttt{GARCH\_FIT\_WEIGHT}=0.5$$

$$v_{t+1}=\omega+\alpha r_t^2+\beta v_t,\qquad v_t=\omega+\alpha r_{t-1}^2+\beta v_{t-1}$$

种子取模型自己的长期水平 $\omega/(1-\alpha-\beta)$（`:1101-1127`），而不是 500 bar 样本方差——理由是持续性高时样本方差种子会在整个窗口里被"报告"而不是被忘记（实测：$0.985^{500}\approx5\times10^{-4}$）。单位转换必须**同时**转截距与收益（$1$ 分数² $=10^4$ 百分²，`:1118-1120`）；只转一个的 bug 实测把预报推到 7.0×/9.6× EWMA（`:1093-1099`，`docs/core-algorithms/10-volatility-targeting.md:274-277`）。

**"自由 ω 无界"这一说法被实测推翻.** 旧 docstring 声称自由 ω 的 MLE 无界、三个优化器都奔向角落并拒绝合成真值。在 `fa028be` 的 shipped BTC 1h 500 bar 窗口上重测：三个优化器返回**同一个**拟合 $\omega\approx0.0813$（%²）、$\alpha\approx0.1965$、$\beta\approx0.1546$，6 位小数一致，且不拒绝任何东西（`core/ml/volatility.py:826-842`）。旧数字（$\omega\approx0.002265$、$\alpha\approx0.0697$、$\beta\approx0.9254$、$0.5\Sigma LL=-226.771$）是在**修复前的 8 846 根缓存**上测的，在当前 11 676 根上**不可复现**（`:834-838`）。IGARCH 网格在同一窗口上落到退化的 $\alpha=1,\beta=0$ 角落，其目标 $\overline{0.5(\log v+x^2/v)}=52.209$ 对 MLE 的 $-0.5726$——**病态的是网格，不是 MLE**（`:838-842`）。

**旧文档还有两处口径错误（逐位引用）**：`1.8e4` / `3.6e5` 是某个量的**求和**却被当成均值；按本模块定义（逐观测均值）它们应是 **36** 与 **720**（第二个曾被写成 `7.2e5`，是 1000 倍滑移），且都不是良定拟合会产生似然（`:844-848`）。

**已知局限.**
1. **成本**：完整 MLE 实测 **≈0.13 s**（shipped 500 bar 窗口，316 次似然评估）与 **≈0.21 s**（合成窗口，509 次），对 `PER_BAR_BUDGET_SEC = 2 ms` 是 65–105×（`:187`, `:985-989`）。因此 `garch11` 是**显式 opt-in**，绝不能放到逐 bar 路径（`:1050-1054`）。
2. 未做 Student-t / EGARCH；回退的 IGARCH 分支**不能**表达均值回复（$0/0$，`fitted=False` 标注，`:976-983`）。
3. 参数对样本敏感：`docs/core-algorithms/10-volatility-targeting.md:168-181` 记录了两个快照的参数差异（$\alpha\approx0.0667,\beta\approx0.9199$ 与 $\alpha\approx0.252,\beta\approx0.144$），作者自述"两种快照的参数差异本身说明 GARCH 参数是窗口/样本的函数"。
4. `garch_backend()` 运行时探测 `arch`（`:602-613`）；本机实测返回 `"scipy"`（`docs/core-algorithms/10-volatility-targeting.md:279-282`）。**未验证**：我这次没有调用 `garch_backend()`。

### 3.3 IGARCH 网格回退

**算法.** 当 scipy 不可用或 MLE 越界时（`_garch11_scipy_mle:869-879`）：令 $\alpha=1-\beta,\ \omega=0$（单位持续性），在 $\beta\in\{0.00,0.01,\dots,0.99\}$（`_GARCH_BETA_GRID`，`:947-950`）上最大化

$$\overline{\tfrac12\left(\log v + \frac{x^2}{v}\right)},\qquad v_t=(1-\beta)x^2_{t-1}+\beta v_{t-1}$$

用 `garch11_filter_grid`（`:914-944`）把整个 $\beta$ 网格**向量化**推进（每次一根 bar 推进整个网格），再由 `_garch11_best_beta` 取 argmax（`:959-970`）。网格**故意从 $\beta=0$ 起**，让拟合可以说"这里没有持续性"，而不是被逼成 EWMA 形状（`:947-949`）。

**为什么保留.** 它便宜（≈1.5 ms，无优化器、确定性），且它的角落在本缓存上**可观测**——这正是"混合"存在的原因：混合把输出限制在拟合一步方差与 EWMA 水平之间，使**任何**退化参数组合都不会产生退化**预报**（`:849-852`, `docs/core-algorithms/10-volatility-targeting.md:237-243`）。

**已知局限.**
1. $\alpha+\beta=1$ 时长期方差 $\omega/(1-\alpha-\beta)=0/0$ **未定义**；该分支下"长期方差就是当前水平"，`garch11_params` 用 `fitted=False` 标注（`:976-983`）。
2. 单位持续性版本**无法**表达波动率均值回复。
3. 角落 $\alpha=1,\beta=0$ 使 $x^2/v$ 无下界；文档记录了两个状态下的角落似然均值（**未修复 48.46 / 2 859.98**、**已修复 91.19 / 2 004.18**，随样本变化，`docs/core-algorithms/10-volatility-targeting.md:210-228`）。
4. 网格是**一维**的（只有 $\beta$），因此它甚至不是 IGARCH 的 MLE，只是一个受限扫描。

### 3.4 Parkinson / Garman-Klass / close-close

**算法**（`core/ml/volatility.py:431-503`；公式在 docstring 与实现里一致）：

$$\begin{aligned}
\text{close-close:}\quad & \hat\sigma=\mathrm{sd}(r,\ \text{ddof}=1) &&\texttt{realized\_vol},\ :431\\
\text{Parkinson:}\quad & \hat\sigma^2=\frac{\overline{\ln(H/L)^2}}{4\ln2} &&\texttt{parkinson\_vol},\ :458\\
\text{Garman-Klass:}\quad & \hat\sigma^2=\overline{\tfrac12\ln(H/L)^2-(2\ln2-1)\ln(C/O)^2},\quad \text{clamp}\ge0 &&\texttt{garman\_klass\_vol},\ :481
\end{aligned}$$

三者都对最近 `window`（默认 500）取均值；`unit="annual"` 时乘 $\sqrt{P}$（$P$ = 每年 bar 数，`annualize`，`:1150-1155`）。

**为什么.** Parkinson 用日内极差，在无漂移扩散下比 close-close 效率高 ≈5×；Garman-Klass 用全部四个价格，≈7×，代价是对坏 print 与跳空更敏感（`:462-465`, `:485-488`）。

**已知局限.**
1. 这些是**点估计**，没有置信区间；P3 文档明确"真实取舍应看预测-实现回归的 $R^2$，未做"（`docs/core-algorithms/10-volatility-targeting.md:369-371`）。
2. 两者都**忽略跳空**（Parkinson 只用 $H/L$），而本项目的数据里跳空正是最大风险源。
3. GK 的方差在样本少时第二项可主导，代码只做 `max(var,0)` 截断（`:502`），这会把一个负方差静默变成 0 而不是标记为失败。
4. 三者都**不做**离群裁剪（只有 EWMA / GARCH 路径走 `clip_outliers`）。

### 3.5 锚定 MAD 与"为什么滚动 MAD 会改写历史"

**算法.** `_anchored_mad`（`:220-258`）：先取样本中位数 $\tilde m$ 与 $\widehat{\mathrm{MAD}}\cdot1.4826$ 作为**种子**，再对每一根 bar 做指数加权递推（权重 $2^{-1/H}$，$H=$ `half_life`，默认 `DEFAULT_MAD_HALF_LIFE = 2000`）：

$$\text{decay}=2^{-1/H};\qquad m \leftarrow m+(1-\text{decay})(x-m);\qquad s\leftarrow s+(1-\text{decay})\big(1.4826\,|x-m|-s\big)$$

裁剪本身：`clip_outliers`（`:323-366`）把 $r$ Winsorize 到 $[\tilde m-k s,\ \tilde m+k s]$，$k=$ `DEFAULT_OUTLIER_SIGMA = 6.0`（`:143`），$1.4826$ 是 $\widehat{\mathrm{MAD}}$ 到 $\sigma$ 的一致性常数（`:145-147`）。

**为什么滚动 MAD 是错的（这是本节的要点）.** 一个朴素的滚动 MAD **从这个调用方交给它的那个窗口重算**，于是：

1. **限值随窗口滑动**，所以某一根 bar 在 $t$ 被裁掉，可能在 $t+1$ **被放开**——作者称之为 "an observation that was clipped at bar $t$ can be **un-clipped** later"。实测（shipped BTC 1h，`fa028be`，11 176 个收益）：相邻 500-bar 窗口的 Winsor 限值在 **11 176 / 11 176** 步上都变了（纯 per-window 中位数/MAD 版本是 7 678 / 11 176；修复前缓存在 8 844 根合成序列上是 8 343 / 8 344），被裁出的值在窗口间移动最多 ≈$1.6\times10^{-2}$（`:220-241`）。
2. 因此一个逐 bar 消费被裁序列的估计量（$\alpha_t$ 跳变）**取决于它什么时候被问**，而不是取决于数据本身（同上）。
3. 用**一个** `series_anchor`（`:369-397`）后，"限值"是数据 + 一对固定数的纯函数：同一根 bar 在含它的每个窗口里被裁成同一个值，且**永不**被窗口前移放开（实测：per-window 限值在 8 344 / 8 345 个相邻窗口上变化 → 单一 anchor 后 **0**）。
4. 顺序也被修了：`_clipped`（`:400-426`）先对**整条**序列建 anchor、再裁、**最后**才切 `window`，所以 `window` 只限制估计器自己的记忆，不触碰裁剪限值（P3/P4 审计第 7 项）。`clip_outliers(r, anchor=series_anchor(r))` 与 `clip_outliers(r)` 是**逐位相同**的（`:374-377`）——改变的是"这一对数**何时**被算出"，不是"裁得多紧"，所以 shipped 数字不动。

**被防护的机制（实测）.** 原始目击者是一次**数据拼接**：BTC 1h 帧在 2026-07-29 → 2026-09-29 之间在一个"bar"里跳了 **+27.63 %** 对数收益。往同一窗口注入一个 $+0.2763$ 的 bar，未裁剪的 RiskMetrics 递归（有效窗口 16.7 bar）报 **4.98 %/bar** 对裁剪后 **0.40 %/bar**——**12.3×** 高估，会让每个仓位按同一因子缩小（`:128-142`）。该目击者已被供应商修复：重测显示 "no gap wider than 61 min"、最大 $|\log r|$ = 4.94 %，clip 在 shipped 序列上**不生效**（最后 500 bar：0.289976 %/bar 裁剪对 0.289977 %/bar 未裁剪）。

**刻意的取舍（作者自述）.** 存在一个 leak-free 的因果变体（每根 bar 只用过去重算 anchor），**故意不做默认**：标量估计器需要的不变式是"$\text{clip}(x_i)$ 是 $x_i$ 与一对固定数的函数"，逐 bar 重锚会重新引入它修掉的不稳定性。中心量是稳健位置量，所以泄漏无关紧要（修复前缓存上一个 $+27.63\%$ 的拼接 bar 只把它移动 ~$1\times10^{-7}$；修复后缓存没有超过 4.94 % 的收益）（`:303-309`）。

**已知局限.**
1. 默认 anchor 用**整条序列**（含未来）计算，因此严格来说**不是因果的**——这是明写的取舍（`:303-309`），不是疏忽。
2. 该取舍使"同一根 bar 在不同长度的输入序列里"得到不同裁剪：`ewma_vol(r, window=500)` 与 `ewma_vol(r[-500:], window=0)` **仍然不同**（实测 0.289976 %/bar 对 0.002899771 一族的差异，`:531-538`）——调用方切了序列就必须自己传 `anchor=series_anchor(r)`。
3. 半衰期 2000 bar 是选择而非估计；在 1h 数据上约 83 天。
4. GJR/EGARCH 类不对称性完全没有处理：裁剪是对称的。

### 3.6 波动率目标化：反比定仓与 clamp

**算法.** `PositionSizer.vol_scale`（`core/risk/position_sizer.py:107-125`）：

$$\text{scale}=\mathrm{clip}\!\left(\frac{\text{target\_vol\_pct}}{\text{forecast\_vol\_pct}},\ \text{min\_scale},\ \text{max\_scale}\right)$$

`forecast` 缺失/非有限/非正 → **返回 1.0**（回到固定比例），而不是 0——"拒绝下单"是比"按旧法下单"更大的行为改变（`:110-114`）。配置默认（`config/config.yaml:154-181`）：`target_vol_pct: 0.45`（%/bar）、`min_scale: 0.25`、`max_scale: 2.0`、`max_position_notional_pct: 10.0`、`stop_vol_multiple: 3.0`、`stop_min_pct: 0.5`、`stop_max_pct: 6.0`，整块 `enabled: false`。

**消费顺序**（`calculate_position_size`，`:314-345`）：

$$\begin{aligned}
\text{capital\_pool} &= B\cdot c_{\text{core|sat}}\\
\text{risk\_per\_trade} &= \min\!\Big(\text{capital\_pool}\cdot\frac{p}{100},\ B\cdot\frac{\text{max\_position\_size\_pct}}{100}\Big)\\
\text{risk\_per\_trade} &\leftarrow \text{risk\_per\_trade}\cdot 0.7\ \text{若 } \texttt{volatility\_expanding}\\
\text{risk\_per\_trade} &\leftarrow \text{risk\_per\_trade}\cdot \text{scale}\quad(\text{>1 时重新施加硬上限与 } \texttt{max\_position\_notional\_pct})\\
\text{risk\_per\_trade} &\leftarrow \text{participation\_cap}(\cdot)\quad(\text{P6-A，**最后**，只会缩小})\\
q &= \text{risk\_per\_trade}/\text{price}
\end{aligned}$$

止损距离（`stop_distance_pct`，`:127-148`）：

$$\text{stop}=\mathrm{clip}\big(m_{\text{stop}}\cdot\text{forecast},\ \max(\text{stop\_min\_pct},\ \text{hard.min\_stop\_loss\_distance\_pct}),\ \text{stop\_max\_pct}\big)$$

**为什么这么做.** 方向不可预测（实测 AUC 0.52/0.53、扣成本期望为负）而**条件波动率可预测**（ARCH/GARCH：Engle 1982、Bollerslev 1986；Tsay 第 3 章），所以把预报花在风险上：scale $\propto1/$forecast 是波动率目标化的定义性质，$m_{stop}\cdot\text{forecast}$ 与 Chandelier exit $k\times$ATR（LeBeau 1995）同构（`:5-32`）。

**已实测效果（`docs/core-algorithms/10-volatility-targeting.md:327-343`；余额 10 000、satellite 池 0.3、固定 8 % → 240 USDT）**

| 预报 %/bar | scale | 名义 USDT | 止损距离 % |
|---|---|---|---|
| 0.225 | 2.000 | 480.00 | 0.675 |
| **0.45（=目标）** | 1.000 | **240.00** | 1.350 |
| **0.90（波动率翻倍）** | **0.500** | **120.00** | **2.700** |
| 1.80（×4） | 0.250 | 60.00 | 5.400 |
| 0.02 | 2.000 | 480.00 | 0.500 |

关闭开关时（`:345-353`）：名义 **240.00**、`stop_distance_pct` **2.000**、`trailing_stop_distance_pct` **2.000**、`barrier_widths_pct` **None**。注意默认 `max_scale=2.0` 下卫星仓 $240\times2=480<1000$，所以 `max_position_notional_pct` 在默认设置下**不可能触发**（`:340-343`）。

**已知局限.**
1. **实盘仓位路径当时未接线**：`manager.py:check_signal` 调 `calculate_position_size` 时**不传** `forecast_vol_pct`，所以实盘仓位仍走固定比例（`docs/core-algorithms/10-volatility-targeting.md:357-364`）。已接线的实盘部分是**止损/移动止损宽度**。⚠️ 这一条是**文档在特定 revision 上的快照**；证据索引 §5 记录 gap fix ① 已把 `RiskManager.check_signal` → `PositionSizer` 接线（开关关闭时余额 10 000 两侧均 240.0 USDT，行为不变，`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:86`）——两处文档口径不一致，见 §11 D-5。
2. $\text{scale}\le2$ 且 `max_position_notional_pct` 不可触发，意味着"目标化"在默认配置下**只有缩小方向**的效力。
3. 未做 A/B 回测对照；P3 文档明确**不给出任何收益类数字**（`docs/core-algorithms/10-volatility-targeting.md:365-368`）。
4. `volatility_expanding` 的 ×0.7 / ×1.3 是**遗留**启发式（`:325-326`, `:375-376`），与 vol targeting 同时启用时会叠加，二者从未被联合评估。

### 3.7 动态 barrier 宽度

**算法（消费侧）.** `PositionSizer.barrier_widths_pct`（`core/risk/position_sizer.py:150-197`）：

$$w=\mathrm{clip}\!\left(\frac{m_{\text{barrier}}\cdot\text{forecast\_pct}}{100},\ \text{barrier\_min\_pct},\ \max(\text{barrier\_max\_pct},\text{barrier\_min\_pct})\right),\qquad \text{返回 }(w,w)$$

单位换算是显式的：预报是**百分比/bar**（0.45），而 `labels.py` 的 `min_pct/max_pct` 是**分数**（0.004 = 0.4 %），所以除以 100（`:157-162`）。

**生产路径（P2 侧）.** `core/ml/labels.py::barrier_widths(..., vol_pct=…)` 用 `vol_multiple × vol` 取代 ATR，并 `clip(lower=min_pct, upper=max_pct)`（`core/ml/labels.py:198-206`）；`vol_pct` 可以是标量（广播）或 Series（按索引对齐 + ffill/bfill）。理由：ATR 是**向后**的真实波幅平均，只在体制变化**持续之后**才反应；预报是条件于最新平方收益的（RiskMetrics/GARCH），两者都被 $[\text{min},\text{max}]$ clamp，所以都不会产生不可交易的屏障（`:190-195`）。

**两个 ATR 与 vol 的路径互为默认.** `vol_pct=None`（默认，也是所有现有调用）逐位走 ATR 路径（`:186-189`）。

**已知局限（明写的"inert"）.**
1. **`barrier_*` 三个配置键当前无效**：`PositionSizer.barrier_widths_pct` **没有任何生产调用者**；实盘 label 宽度来自 `ml.barrier_atr_period` / `ml.barrier_atr_multiple` / `ml.barrier_min_pct` / `ml.barrier_max_pct`（`MLPredictor._barrier_params`）（`core/risk/position_sizer.py:164-183`，`config/config.yaml:182-195`）。`app.config.inert_barrier_key_warnings` 会在启动时对非默认值打 WARNING（`:179-183`）；`tests/test_p34_audit_fixes.py::test_barrier_widths_hook_has_no_production_caller` 是 tripwire。
2. 因此"动态 barrier 宽度"在**当前生产路径上不生效**；它只在 P2 的 `labels.py` API 层可选。
3. `core/ml/meta.py` 的屏障**也**只按 ATR 缩放，且 meta.py 的 docstring 明确把"用 vol 预报替代 ATR"写成**文档化的缝，不是已实现的**（`core/ml/meta.py:75-80`）。

### 3.8 拼接闸门（gap guard）

**算法.** `_series_has_gap(index, interval, max_bars=1.5)`（`core/risk/manager.py:87-131`）：

1. 查 bar 长度表 `_INTERVAL_HOURS`（`:49-53`）；未知区间 → 返回 `False`（惰性，不错误拒绝预报）；
2. 索引必须是 datetime-like：`raw.dtype.kind in "iufb"`（整数/浮点/bool，即 `RangeIndex`、`Float64Index`）→ **返回 True（拒绝）**；任何异常 → **返回 True**；
3. 少于 3 根 → `False`；
4. **先把时间戳折叠到声明的 bar 键**（`core.market_data.ohlcv_cache.bar_keys`），再差分；
5. 若存在相邻步长 $>\texttt{max\_bars}\times$ bar 长度（`_VOL_MAX_GAP_BARS = 1.5`，`:46`）→ `True`。

**为什么"折叠"是关键（D1 缺陷）.** 缓存能存两种时间戳口径：bar-**open** 与 Binance `close_time`（$=\text{open}+\text{length}-1\,\text{ms}$）。实盘 `BTCUSDT/1h.parquet` 的尾巴 `…06:00:00 → …07:59:59.999` 原始差是 **1.99997 h**，超过 1.5 的界，于是**连续的**序列被判为缺口：`_series_has_gap=True`，`RiskManager`/`PositionGuard` 预测 **None**，`check_data_integrity` 报 `BTCUSDT/1h` **GAP / missing 1**，全仓 **25/29** 带缺口（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:181`）。按 bar 键折叠后 max step 1.0 h、0 缺口。同一折叠也被加进 `scripts/check_data_integrity.py::gap_report`（`scripts/check_data_integrity.py:140-150`），两边必须一致。

**实测（D1 修复后，归档）.** 同形帧（300 open + 1 close）：`_series_has_gap` **False**；`RiskManager` = `PositionGuard` = **0.07684716805517097 %/bar**（逐位相同）；`gap_report` missing **0** / gaps **0** / flagged **False**；实盘文件 `--symbols BTCUSDT --intervals 1h` 由 GAP/missing 1 变 **ok / missing 0**；全量 **25/29 → 24/29**。真缺口仍拒绝：少 1 根 bar → `_series_has_gap=True`、预测 None；100-bar 拼接 → 最大缺口 100.0 h、missing 99（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:181`）。非时间索引（L6）：`pd.RangeIndex(300)` / `Float64Index` / object 字符串 / tz 混合 → 全部 **True**；`"7h"` 未知区间 → 仍 **False**；空/少于 3 根的合法日期索引 → **False**（`:188`）。

**已验证的常量与缓存边界（我本次研读源码）.** `_VOL_HISTORY_BARS = 600`（`:23`）、`_VOL_CACHE_TTL_SEC = 300.0`（`:27`）、`_DEFAULT_VOL_INTERVAL = "1h"`（`:29`）、`_VOL_MAX_GAP_BARS = 1.5`（`:46`）、`PositionGuard._VOL_HISTORY_LIMIT = 600`（`core/risk/position_guard.py:34`）。`PositionGuard.forecast_vol_pct` 调用**同一个** `_series_has_gap` 与同一 bar 长度表（`core/risk/position_guard.py:131-136`）——这是复核 R1 的修复：修复前 guard 自己建 `DatetimeIndex` 但不做缺口检查，同一条拼接序列 manager 给 `None` 而 guard 给 **0.062009291811329616 %/bar**，且该值喂给实盘移动止损距离（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:165`）。

**已知局限（**最重要的一条**）.** 实盘预测只看得见**最后 600 根 bar**，所以折叠后的闸门只能拒绝落在该窗口内的洞：位于 −100 或 −300 的洞会被拒绝，位于文件中间（例如 −900）的洞**对它不可见**。整文件口径的检查是 `scripts/check_data_integrity.py`。该限制写进 `manager.py` 的常量注释（`core/risk/manager.py:16-22`，`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:194`）。另外：拼接与 close 口径尾行在**实盘文件里当前都已不存在**（测量时刻 2026-09-30 17:28：原始最大相邻步 1.0 h，`clipped == unclipped = 0.3003 %/bar`，1.00×；`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:195`），所以缺陷机制现在只能由**重建帧/合成帧**证明。

### 3.9 已实测数字（P3）

**成本（归档，`core/ml/volatility.py:171-186` 的注释）**

| 路径 | 归档数字 | 我本次实测（`c703b8b`，见下） |
|---|---|---|
| `ewma` 500 bar | ≈0.14 ms/call | **0.2248 ms**（500 bar 数组） |
| `ewma` 600 bar（实盘形状） | ≈0.15–0.16 ms | **0.3093 ms**（600 bar 数组） |
| realised 家族 | ≈0.01 ms | 未测 |
| `forecast_vol` 默认 | — | **0.2806 ms**（600 bar）/ **0.2808 ms**（500 bar） |
| `garch11` 完整 MLE | ≈0.13 s（shipped 500 bar，316 次似然） / ≈0.21 s（合成，509 次） | 未测 |
| `garch11_params(window=0)` 全历史 | ≈4.1 s/次（11 674 根） | 未测 |
| 预算 | `PER_BAR_BUDGET_SEC = 2.0e-3`（2 ms） | — |

> **口径警告**：`docs/core-algorithms/10-volatility-targeting.md:157-177` 报 `ewma ≈1.5 ms/次`，而 `core/ml/volatility.py:176-181` 报 ≈0.14 ms 并自称"older '0.14–0.27 ms / ≈0.1 ms' pair is 2–10× high"——**文档与代码注释互相矛盾**，我实测 0.19–0.31 ms（见 §11 D-1）。我的测量命令（只读）：
>
> ```powershell
> python -c "
> import time, numpy as np, pandas as pd
> from core.ml.volatility import ewma_vol, log_returns, forecast_vol
> r = log_returns(pd.read_parquet('data/market/BTCUSDT/1h.parquet', columns=['close'])['close'].astype(float).values)
> def t(f, n=500):
>     f(); s=time.perf_counter()
>     for _ in range(n): f()
>     return (time.perf_counter()-s)/n*1e3
> print('ewma 600-bar %.4f ms' % t(lambda: ewma_vol(r[-600:], window=0)))
> print('ewma 500-bar %.4f ms' % t(lambda: ewma_vol(r[-500:], window=0)))
> print('forecast_vol 600-bar %.4f ms' % t(lambda: forecast_vol(r[-600:], window=500)))
> "
> ```
> 输出（本次）：`0.3093 / 0.2248 / 0.2806` ms。注意 `data/market/BTCUSDT/1h.parquet` 是**活文件**（运行中的进程在写），所以每次读到的行数不同；本次读到 11 627 行。

**裁剪与拼接（归档）**

| 量 | 值 | 出处 |
|---|---|---|
| BTC 1h 未裁剪 EWMA 对裁剪 | **5.31 %/bar 对 0.52 %/bar（10.12×）** | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:73` |
| 同源括号（doc 10 的版本） | **5.3056 对 0.5242（9.81×）**，并指出早期 10.5× 是"拿 0.52 去除 5.47"的口径混用 | `docs/core-algorithms/10-volatility-targeting.md:80-81` |
| `manager.py` 的版本 | 5.31 对 0.52 = **10.1×** | `core/risk/manager.py:33-45` |
| 注入实验 | $+0.2763$ bar → 未裁剪 **4.98 %/bar** 对裁剪 **0.40 %/bar**（**12.3×**） | `core/ml/volatility.py:137-141` |
| 修复后实盘（测量时刻 2026-09-30 17:28） | clipped == unclipped = **0.3003 %/bar（1.00×）** | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:195` |
| 本次实测（`c703b8b`） | clipped = **0.372748 %/bar**，clipped $\ne$ unclipped（活文件，随写入变化） | 我运行的命令（§3.9 上框） |

> ⚠️ **5.31 / 0.52 / 10.12× / 10.1× / 9.81× / 12.3× 是同一现象在四份文档里的四个不同数字对**。它们的差异来自样本 revision 与口径（用未裁剪除裁剪 vs 反过来、500 vs 600 bar、拼接文件 vs 修复文件）。见 §11 D-2。

**GARCH 参数（四份记录，互不相同）**

| 来源 | $\omega$ | $\alpha$ | $\beta$ | 备注 |
|---|---|---|---|---|
| `core/ml/volatility.py:832-833`（当前缓存 500 bar 重测） | 0.0813 %² | 0.1965 | 0.1546 | 三优化器 6 位一致 |
| `docs/core-algorithms/10-volatility-targeting.md:204-207` | ≈0.0027 %² | ≈0.067 | ≈0.920 | $0.5\Sigma LL\approx-225.9$，差 < 1e-4 |
| 同上 `:168`（另一快照） | 2.96e-7 分数² | 0.0667 | 0.9199 | 持久性 ≈0.9866 |
| 同上 `:179`（11 674 根复核） | — | 0.252 | 0.144 | — |
| 旧 docstring（**已作废**） | 0.002265 | 0.0697 | 0.9254 | $0.5\Sigma LL=-226.771$，8 846 根缓存，**不可复现** |

**修正后的 PSR 值**（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:185`）：厚左尾 $t=2.0000000000000004$ → **0.9479777894541446**（正态近似 **0.9772498680518209**）；同形状 $t=2.9999999999999996$ → **0.9869562688374416**（正态 **0.9986501019683699**）。同一份文档 `:168` 还记录了"本行曾误写 0.9864"的自我更正。

**测试钉.** `tests/test_volatility_targeting.py` **21** 项、`tests/test_gap_fixes.py` **12** 项（证据索引 `:13`）；我实测 `def test` 计数同为 **21 / 12**，一致。

---

## 4. 配对 / 协整（P4）

> 证据主文件：`docs/core-algorithms/11-pairs-cointegration.md`、`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md` §4（`:80`）。

### 4.1 ADF 与 Engle-Granger 两步法

**算法.** Engle-Granger（`core/strategy/pairs.py:465-514`）：

$$\begin{aligned}
\text{第一步（水平值 OLS）}:\quad & y_t=\alpha+\beta x_t+e_t &&(\texttt{ols\_hedge\_ratio},\ :441)\\
\text{第二步（残差 ADF，无常数）}:\quad & \Delta e_t=\varphi e_{t-1}+\sum_{i=1}^{p}\psi_i\Delta e_{t-i}+u_t\\
\tau &= \hat\varphi/\mathrm{se}(\hat\varphi) &&(\texttt{adf\_regression},\ :354)
\end{aligned}$$

滞后阶 $p$ 由 **AIC** 在 $0..p_{\max}$ 上选出，$p_{\max}$ 由 Schwert 规则给出并截断到 8：

$$p_{\max}=\min\!\Big(8,\ \Big\lfloor 12\big(\tfrac{n}{100}\big)^{1/4}\Big\rfloor\Big)\quad(\texttt{\_schwert\_max\_lags},\ :349),\qquad p\le\max(0,\lfloor n/5\rfloor-2)\ (:375)$$

AIC 用 $n_{\text{eff}}\log\hat\sigma^2+2k$（`:395`），其中 $k$ 是回归量个数（含 $e_{t-1}$ 与可选常数）；$\hat\sigma^2$ 是残差平方和 / 自由度。平局取更小的滞后（`candidate["aic"] < best["aic"]`，`:405`）。

**为什么不能用普通 DF 临界值.** $\tau$ 必须与**估计残差**的零分布比较（Engle-Granger / MacKinnon N=2），因为残差是估计出来的，分布左移；用错表会**过度拒绝**（`docs/core-algorithms/11-pairs-cointegration.md:24-25`）。

**MacKinnon 临界值：本项目用 Monte-Carlo 模拟而不是查表.** `tau_null_distribution`（`core/strategy/pairs.py:253-309`）用 4 000 条路径、固定种子 20240617（`SIM_REPS=4000`, `SIM_SEED=20240617`，`:137-139`）、分块模拟（`SIM_BLOCK=250`，`:138`）、长度取 `admissible_sim_length`（`:312-314`，界 $[250,2000]$）；缓存键是 `(kind, length, lags, use_const)`（`:296`），上界 8 条（`_NULL_CACHE_MAX=8`，`:148`），且**满了是整体清空而不是 LRU**（`:306-308`）。`tau_pvalue`（`:317-332`）与 `tau_critical_values`（`:334-344`）消费该分布：

$$p=\frac{\#\{\tau^{*}\le\tau\}}{n},\qquad p\in\left[\tfrac1n,\;1-\tfrac1n\right],\qquad \tau_{\alpha}=\text{quantile}(\tau^*,\alpha)$$

（$p$ 由 `np.searchsorted(taus, tau, side="left")/n` 得，`:330-331`；非有限 $\tau$ → 返回 1.0，`:328-329`。）

**生产代码里没有任何 MacKinnon 常数.** 仓库里唯一的 MacKinnon 数值在**测试**里（`tests/test_pairs.py:108-120`）：

```
Reference (MacKinnon 1994/2010, asymptotic):
DF with constant 1 %/5 %/10 % = −3.43 / −2.86 / −2.57;
Engle-Granger, 2 variables, constant = −3.90 / −3.34 / −3.04.
```

断言容差 `abs=0.10`（1 %）、`0.08`（5 %）、`0.08`（10 %），模拟长度 `n_obs=2000`（被 clamp 到 `SIM_MAX_T`）。所以被校验的是 MacKinnon 的 **asymptotic DF-with-constant** 表与 **E-G N=2-with-constant** 表；标定测试（`:138-146`）比较的是随机游走上的**无增广、含常数**回归。statsmodels **未安装**（子代理用文件系统核对 `site-packages` 与 `requirements.txt:3-8` 确认），仅作可选交叉验证（`HAS_STATSMODELS`，`:416-420`；`statsmodels_adf_tau`，`:423-438`）。


**不匹配的零分布只是近似（这是明写的一条）.** 默认 `matched_null=False` 用**无滞后增广、无常数**的零分布，而真实统计量带 AIC 选的增广滞后（15m 序列上选中 **lag 1**）——正是模块 docstring 自己禁止的错配。实测差异：15m 长度（模拟长度 2000）上 5 % 分位 **−3.3015（近似）对 −3.3098（匹配）**，差 **0.0083**；BTC/XRP 15m 的 $p$ 值 **0.1718 对 0.1713**，**结论不变**（`docs/core-algorithms/11-pairs-cointegration.md:32-36`；`core/strategy/pairs.py:475-489` 给出 5 % 点 −3.370 → −3.363 的另一个量法，差 0.007）。`matched_null=True` 用与检验**相同**的滞后规则模拟，成本约 10×（每个候选滞后每条路径解一次有界 OLS），因此是"决定性单次检验"的选项，而不是 30 币对扫描的默认。返回值带 `null_matched` / `null_lags` 字段说明用了哪一种（`:502-511`）。

**为什么 `const` 不暴露.** ADF 作用在 OLS 残差上，残差均值按构造为 0，所以 `regression="nc"` 是正确的设定，零分布必须与它匹配（`:491-493`）。

**已知局限.**
1. 默认零分布与统计量**不一致**（近似），差异虽小但方向未知。
2. 模拟长度被限制在 $[250,2000]$（`:142-143`），因此 35 385 根的 15m 序列使用的是**最长 2000** 的零分布——临界值随 $n$ 的收敛没有被处理。
3. `rolling_cointegration`（`:517-569`）的 `pass_share ≈ 0.05` 是**数据挖掘基线**（纯噪声的通过率），文档明确这一点（`:535-537`）；但报告没有做这个基线的显著性检验。
4. 单位根检验的**多重检验**（30 个币对）没有校正。

### 4.2 OU 半衰期

**算法.** `ou_half_life`（`core/strategy/pairs.py:690-715`）：对价差 $s$ 做 AR(1) 回归

$$\Delta s_t = a + b\,s_{t-1}+\varepsilon_t\ \Longrightarrow\ \kappa=-b,\qquad t_{1/2}=\frac{\ln 2}{\kappa},\qquad \text{equilibrium}=-\frac{a}{b}$$

实现用 `np.linalg.lstsq`（`:704`）；$b\ge0$（不均值回复）时返回 `half_life=inf`、`mean_reverting=False`（`:706-709`）；样本 $<20$ 同样返回 `inf`（`:698-700`）。

**回看窗口.** `lookback_from_half_life`（`:718-729`）：$L=\mathrm{clip}(\mathrm{round}(4\cdot t_{1/2}),30,250)$，常量 `PAIRS_LOOKBACK_MULT=4.0`、`PAIRS_MIN_LOOKBACK=30`、`PAIRS_MAX_LOOKBACK=250`（`:123-125`）。

**半衰期在哪个价差上估计.** 在 **OLS 残差**上（被检验的那个价差）；Kalman 价差的半衰期只作**诊断**——滤波器吸收水平漂移，所以这个数即使对非平稳关系也很小。代码记录的是 **2.1 bar** 用于 BTC/ETH，其 OLS 价差**根本不**均值回复（`core/strategy/pairs.py:800-803`）；而 `docs/core-algorithms/11-pairs-cointegration.md:145` 的表格写 **1.2**——同一个测量两个数，见 §11 D-25。

**Kalman 时变对冲比（Tsay 第 7 章，`:574-685`）**：状态 $\theta_t=[\beta_t,\alpha_t]^\top$，$\theta_t=\theta_{t-1}+w_t$，$w\sim N(0,Q)$，$Q=\delta/(1-\delta)I_2$；观测 $y_t=[x_t,1]\theta_t+v_t$，$v\sim N(0,R)$；$\theta_0=$ OLS 热启动。`KALMAN_DELTA = 1e-4`（`:131`）、`KALMAN_P0 = 1.0`（`:132`）。递推（`:627`, `:640-662`）：

$$P_{t|t-1}=P_{t-1}+Q,\quad F_t=z_t^\top P_{t|t-1}z_t+R,\quad K_t=\frac{P_{t|t-1}z_t}{F_t},\quad \nu_t=y_t-z_t^\top\theta_{t|t-1},\quad \theta_t=\theta_{t|t-1}+K_t\nu_t,\quad P_t=(I-K_tz_t^\top)P_{t|t-1}$$

非有限观测被**跳过**并把状态前推（`:651-653`），理由："a missing bar must never become a zero return"（`:603-604`）。

> ⚠️ **未声明的全样本依赖.** $\nu_t$ 只用 $\theta_{t|t-1}$（`:657` 在 `:659` 的更新之前），这一点是因果的；但**热启动与 $R$ 都是全样本量**：`ols = ols_hedge_ratio(yn, xn)`（`:629`）在全样本上拟合，同时喂给 $\theta_0$（`:640`）与 $R$（`:631-634`）。因此 $\nu$ 序列只在滤波器忘掉全样本 OLS 种子之后才是因果的。`core/strategy/pairs.py:57-59` 的措辞（"β_t is estimated from information up to t−1 only"）读起来像整条路径都因果——**它没有限定这一点**。相对地，`fit_pair`/`engle_granger` 按构造就是全样本的，`docs/core-algorithms/11-pairs-cointegration.md:183` 明说了。


**已知局限.**
1. AR(1) 的 $b$ 是**有偏**估计（小样本下偏向均值回复），因此 $t_{1/2}$ 系统性偏小；本项目没有做偏差校正。
2. `equilibrium = -a/b` 只在 $b<0$ 时有意义；$b\ge0$ 时返回 NaN（`:707-708`）。
3. $t_{1/2}$ 的置信区间没有计算，而 `pair_guard` 直接用它做**硬门**（$[2,120]$ bar，`:120-121`）。
4. 半衰期是**全样本**估计，不是因果滚动估计。

### 4.3 z-score 入场/出场与硬门

**算法.** `rolling_zscore`（`:874-…`）：

$$z_t=\frac{s_t-\overline{s}_{[t-L,\ t-1]}}{\mathrm{sd}(s_{[t-L,\ t-1]})}\qquad(\text{**不含 } t \text{ 自身**，见 } \texttt{docs/core-algorithms/11-…:79-80})$$

阈值（`core/strategy/pairs.py:127-129`）：`PAIRS_Z_ENTRY=2.0`、`PAIRS_Z_EXIT=0.5`、`PAIRS_Z_STOP=4.0`。`pairs_positions`（`:889-…`）按这三个阈值产生 $\{-1,0,+1\}$。

**硬门 `pair_guard`（`:734-778`）** 拒绝条件：$n_{\text{obs}}<\texttt{PAIRS\_MIN\_OBS}=250$；$p>\texttt{PAIRS\_MAX\_ADF\_PVALUE}=0.05$ 或 $\tau$ 非有限；半衰期非有限 / $<2$ / $>120$；$\beta$ 非有限或 $\le0$。理由串逐项拼接，与 `credibility_gate` 同一契约（`:744-752`）。

**Kalman 时变对冲比（Tsay 第 7 章，`:574-685`）**：状态 $\theta_t=[\beta_t,\alpha_t]^\top$，$\theta_t=\theta_{t-1}+w_t$，$w\sim N(0,Q)$，$Q=\delta/(1-\delta)I_2$；观测 $y_t=[x_t,1]\theta_t+v_t$，$v\sim N(0,R)$；$\theta_0=$ OLS 热启动。`KALMAN_DELTA = 1e-4`（`:131`）。

**一个实测到的 bug（文档记录）.** $R$ 必须取 **OLS 残差方差**，不能取 $\mathrm{Var}(y)$：BTC/ETH 1h 上 $\mathrm{Var}(y)=0.0400$ 而残差方差只有 $0.0094$（**4.3×**）；用 $\mathrm{Var}(y)$ 时滤波器几乎不更新，$\beta$ 塌缩到 **0.058**（OLS 为 **0.628**），"价差"退化成 $y$ 的原始价格，配对交易变成单边方向赌注（`docs/core-algorithms/11-pairs-cointegration.md:60-65`）。

**已知局限.**
1. `PAIRS_ENABLED = False`（`:112`）：模块默认关闭，未接入任何自动下单路径（`docs/core-algorithms/11-pairs-cointegration.md:5`）。
2. 阈值 $2.0/0.5/4.0$ 是常量，没有按半衰期或波动率标定。
3. `z` 的分母用滚动标准差，但没有做**波动率缩放**（与 P3 的机制不共享）。

### 4.4 已实测判决：真实数据 0/30

**已实测判决：0 / 30 的口径必须写清.**

**归档主张（B 级）**（`docs/core-algorithms/11-pairs-cointegration.md:136-138`）：30 次 E-G 检验 = **10 个币对 × 1h/4h/15m** 三个周期，样本 2025-06-03 → 2026-09-30。

> **0 / 30 通过 5 % 水平。**

最好的三个（doc-only，未测试断言）：**BTC/XRP 15m $p = 0.168$（$n = 35\,385$）**、BTC/XRP 4h $p = 0.171$、ETH/SOL 4h $p = 0.176$。P4 的提交信息也写 "Measured verdict: 0 of 30 real tests pass"（`git show a9e549e`）。

**测试真正钉住的只是它的子集（A 级）**：`tests/test_pairs.py:423` 的 symbols 是 5 个（BTC/ETH/SOL/BNB/XRP），`:429` 用 `itertools.combinations(symbols, 2)` = 10 对，`:450` 断言 `len(p_values) == 10`，`:454` 断言 `not any(verdicts)`——即 **0 / 10，且只在 1h 周期上**，并且拒绝理由从不是 `"pass"`（`:448`）。**仓库里没有任何脚本跑那 30 次扫描**（在 `core/`、`tests/` 之外 grep `engle_granger`/`pair_guard` 无命中）；5 个币种的 15m/4h parquet 都在，所以 10×3=30 在算术上可得，但**没有代码去算它**。另有两个内部不一致：`docs/core-algorithms/11-pairs-cointegration.md:34` 写 BTC/XRP 15m $p$ = **0.1718/0.1713**，而 `:138` 写 **0.168**（该文档自己的 §局限 6 承认 $p$ 随缓存增长在第 3 位小数漂移）。

**其余 P4 验收（归档）.** 模拟 ADF 零分布复现 MacKinnon 渐近临界值；`adf_regression` 在随机游走上与零分布一致；E-G 对协整对有效、对独立游走规模正确；Kalman 动态对冲比跟踪时变 beta（不再塌缩到 ~0）；交易辅助函数无前视；真实 1h 主流币拒绝全部配对（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:80`）。子代理核对了这批断言的**测试侧强度**：

| 主张 | 测试强度 |
|---|---|
| 300 条随机游走、$T=500$、`regression="c", max_lags=0`：5 % 分位与模拟 `df_c` 表相差 < 0.30、均值 **−1.57**（容差 0.25） | **测试断言**（`tests/test_pairs.py:138-146`；阈值登记在 `tests/test_measured_threshold_policy.py:169`） |
| 功效：20 个协整对 → 测试断言 `mean(p<=0.05) >= 0.9` 且 `median(p) < 0.01` | 测试断言**弱于** doc 11:125 的"20/20"（测试允许 18/20） |
| 规模：20 条独立游走 → 测试断言 `count <= 4`；doc 记录 **2/20**，更高分辨率的 200 条重跑给 **6/200 = 0.030**（5 %）与 **1/200 = 0.005**（1 %）；`tests/test_p34_audit_fixes.py:742-761` 只断言 `<= 0.08`/`<= 0.03` | 3 个不同的点估计（0.030 / 0.035 / 2/20）并存 |
| BTC/ETH 1h：$p = 0.4502$、OLS 半衰期 1200 bar（测试注释）vs doc 表 $p = 0.451$、半衰期 1202 | 互为印证；测试本身已"去钉"为逻辑断言（`:401-405`） |
| Kalman 均值 0.606 vs OLS 0.628（BTC/ETH） | **测试断言**（`tests/test_pairs.py:478-479`） |
| 强制交易表（doc 11:157-165）、IS-vs-OOS（`:169-171`）、滚动 500-bar 通过率 0.00–0.176（`:173-174`） | **doc-only，无测试**；`rolling_cointegration`（`core/strategy/pairs.py:517-569`）在 `tests/test_pairs.py`/`test_regime.py`/`test_microstructure.py` 里**零覆盖** |
| doc 11:119-120 的"实测临界值"列（DF c −3.40/−2.85/−2.54；E-G N=2 c −3.88/−3.32/−3.03） | **doc-only**：测试只钉"与文献值接近"（−3.43/−2.86/−2.57；−3.90/−3.34/−3.04） |


**已知局限（结论层面）.** 0/30 的样本是**10 个主流币对**，不是全市场扫描；`docs/overhaul/P6_VOLUME_PLAN.md:107` 提到 496 个 USDT 交易对，配对能力**没有**在那个广度上被测试。**未验证**：我没有独立重跑这 30 次检验。

---

## 5. Meta-labelling（P4）

> 证据主文件：`docs/core-algorithms/12-meta-labeling.md`、`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md` §4（`:79`）。

### 5.1 一级模型 → 二级（meta）分类器

**算法.** 不再让 ML **选方向**，而是让 ML 只回答"这笔一级交易会不会先碰到止盈屏障"（`core/ml/meta.py:1-20`）：

* **一级**：一条规则信号 / GA 冠军签名 / 任意 $\{-1,0,+1\}$ 序列，决定**何时**与**哪个方向**；
* **二级**：预测"先碰止盈还是先碰止损"；
* 二级输出只用于**过滤**（低于成本感知阈值就不做）与**缩放**（按概率放大），**永远不能反向**。

信号构造：`primary_signal_from_rules`（`:128-159`）用**共享的条件核** `core.strategy.indicators.evaluate_condition`（所以 meta 标签建立在实盘引擎会评估的同一条规则上）；两侧同时活跃 → 0（引擎的歧义规则）。一级交易的前向收益：

$$r_t = \text{side}_t\cdot\Big(\frac{C_{t+h}}{C_t}-1\Big)\qquad(\texttt{primary\_forward\_returns},\ :162-177)$$

### 5.2 标签构造

**算法.** `profit_barrier_labels`（`:212-269`）：

$$y^{\text{meta}}_t=\begin{cases}1 & \text{止盈屏障先被触及}\\0 & \text{止损屏障先被触及}\\ \mathrm{NA} & \text{超时或一级无信号}\end{cases}$$

屏障**方向感知**：多头 $\text{profit}=C_t(1+u_t),\ \text{loss}=C_t(1-l_t)$；空头**互换**（`:246-251`）。宽度用与 `core.ml.labels` **同源**的 ATR 构造 `_barrier_widths`（`:182-209`，优先 import `core.ml.labels.barrier_widths`，失败时用**完全相同**的 Wilder-ATR 本地回退），默认 `META_ATR_PERIOD=14`、`META_ATR_MULTIPLE=1.5`、`META_BARRIER_MIN_PCT=0.004`、`META_BARRIER_MAX_PCT=0.06`（`:111-115`）。**只有具有完整前向窗口的行才被打标**，所以最后 `forward_periods` 行永远是 `NA`（`:226-229`）。

**从三分类屏障标签转换.** `meta_label_from_barrier`（`:272-287`）：`create_triple_barrier_label_vol` 的语义是 $1$=上轨先、$0$=下轨先、$2$=超时；**多头**在类 1 上盈利，**空头**在类 0 上盈利；超时与无信号 → `NA`（**丢弃**，绝不塞进任何类）。

**数据集装配.** `build_meta_dataset`（`:290-326`）返回 `{X, y_meta, trade_returns, side, n_primary, n_labelled, timeout_share}`；`X` 默认用 39 列契约（`core.ml.features.compute_features`）。

**仓位缩放公式**（`MetaLabeler.decide`，`:752-769`）：

$$\text{take}=[p\ge\theta],\qquad \text{size}=\text{floor}+(\text{max}-\text{floor})\cdot\mathrm{clip}\!\left(\frac{p-\theta}{1-\theta},0,1\right)$$

`META_SIZE_FLOOR=0.25`、`META_MAX_SIZE_MULTIPLIER=1.0`（`:109-110`）。最终仓位 `position = sign(一级方向) × size`——**符号永远来自一级**（`MetaDecision.apply`，`:647-655`）。阈值搜索是**单边**的（`meta_cost_aware_threshold`，`:331-385`：`take = p ≥ t`），因为二级模型不允许做空一级信号。`META_MIN_PROBABILITY=0.5` 是独立于搜索的绝对下限（`:105-107`）。

**盈亏平衡命中率.** `breakeven_hit_rate(cost_pct, reward_risk)`（`:807-817`）：

$$p^\star=\frac{1+c}{1+R}\qquad(c=\text{cost}/100,\ R=\text{reward:risk})$$

$c=0.25\%$、$R=1$ 时 $p^\star=0.5012$（`docs/core-algorithms/12-meta-labeling.md:65`）。

### 5.3 拒绝它的门

**算法.** `evaluate_meta_oos`（`core/ml/meta.py:438-586`）用**与 P2 同一套嵌套协议**：`purged_kfold_splits(..., label_span=24)` + `sample_uniqueness_weights`；每折训练块尾部 20 % 作为**该折自己的校准流**；阈值在**折内**校准流上选（`meta_cost_aware_threshold`，`min_trades=fold_min_trades`）；再套到该折 test 行（`:510-517`）；门通过 `meta_gate` → `credibility.gate_from_evaluation` 消费**外层**数字（`:576-581`, `:629-631`）。可部署阈值 = 各折阈值的**中位数**（`_fold_threshold_summary`，`:589-613`）；无折选出候选 → **`threshold=None`**（"无阈值"，绝不是"用池化最优"）。池化搜索仅作对照（`selection="pooled_optimistic"`，`:616-626`）。

`MetaLabeler.enabled` 只有在 `gate["allowed"]` 为真时才为 `True`（`:694-696`）；`evaluate` 在门拒绝时**清空** threshold/model/calibrator（`:707-711`），`decide` 在未启用时对**任何**概率返回 `take=False`（`:759-761`）。`fit` 在没有先 `evaluate` 时抛 `RuntimeError`（`:724-726`）。

### 5.4 已实测判决：10/10 全部拒绝

**实测（`docs/core-algorithms/12-meta-labeling.md:63-88`）：** 缓存 BTC/ETH 1h、8 845 bar、成本 0.25 %/往返，**5 条一级规则 × 2 品种 = 10 次评估**。

> **门控结论：10/10 全部拒绝**（最高 AUC **0.5167 < 0.55**；净期望 **−1.246 % … 0.000 %** ≤ 0；$t$ 最高 **−1.41**）。

标签基准率全部落在 **0.469–0.511**，即一级交易的止盈/止损命中**近似抛硬币**（`:66`）。ETH 的 rsi 均值回归准确率 **0.446** 对多数类 **0.531**，**低于基线 8.5pp**——与 P2 审计测到的失效模式完全一致（`:82-83`）。

**结构发现（诚实路径在工作）**：对"几乎每根 bar 都触发"的一级规则（MACD 交叉 8 813/8 845），多折校准流选不出任何正期望阈值（`n_cal_trades=0`，该折 OOS 笔数 0）；`thresholds_oos.n_folds_with_candidates` 在 BTC rsi 两条规则上为 **0**，因此阈值是 `None`——这是"无阈值"这条诚实路径在起作用，而不是被优化掩盖（`:85-88`）。

**顺带修掉的 P2 缺陷（文档记录）**：`credibility.evaluate_model_oos` 的 `model_factory=None` 默认路径把**工厂构造器**当工厂绑定，第一次真实 meta 评估 5 折**全部**失败（`no fold produced a model`）；`meta.py::resolve_model_factory` 两种形态都接受（`docs/core-algorithms/12-meta-labeling.md:90-103`；`core/ml/meta.py:407-435` 的 docstring 记录了该 TypeErrors）。

**测试钉.** `tests/test_meta_labeling.py` 我实测 **24** 项（21 个 `def test` + 3 个 `async def test`，位于 `:535`/`:549`/`:576`）；`docs/core-algorithms/12-meta-labeling.md:61` 写 **20** 条，`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:14`/`:79` 写 **24**（并注明"正被并行修改"）——见 §11 D-6。

**一个与 §2.1 缺陷的耦合（重要）**：meta 路径**不使用** `create_triple_barrier_label_vol` 的 `timeout_label`（那是死参数，见 §2.1 局限 4）。它用的是自己的 `profit_barrier_labels`（`:212-269`）或 `meta_label_from_barrier`（`:272-287`）——两者都**显式**把超时置 `NA` 并**丢弃**，语义正确。所以 §2.1 的类 2 缺陷**不影响** meta-label 的标签，但它**影响** `core/ml/predictor.py:460-463` 持久化的 `barrier.distribution`（P2 侧）。


### 5.5 已知局限（meta）

1. 标签是"先碰哪个屏障"的分类，**不含路径**：止盈前先浮亏 5 倍风险也算 1（`core/ml/meta.py:65-66`；`docs/core-algorithms/12-meta-labeling.md:107`）。
2. 一级样本远小于 bar 数：功效受一级触发次数限制；门要求 100 笔外层交易，**触发 40 次的规则在结构上不可能通过**——这是设计，不是 bug（`core/ml/meta.py:67-70`）。
3. `META_LABELING_ENABLED = False`（`core/ml/meta.py:102`）且 `StrategyEngine` 只咨询调用方**显式注册**的 meta 模型（`:71-73`）。
4. 一级信号必须**因果**是调用方的责任（`:62-64`）。
5. 阈值网格 0.30–0.95 步长 0.01；文档称更细的网格不改变结论（净期望全为负），**未验证**。
6. 一级规则只有 5 条、样本单一（2025-06 → 2026-09）；换样本必须重测。
7. `MetaLabeler.fit` 里 `sample_uniqueness_weights(len(Xv), 24)` 把 label span **硬编码为 24**（`core/ml/meta.py:735`），与 `build_meta_dataset` 的 `forward_periods` 参数不联动——如果调用方用别的 $h$，权重会错。

---

## 6. 微观结构（P4）

> 代码：`core/market_data/microstructure.py`（611 行）。所有公式同时写在模块 docstring（`:14-46`）与实现里，我逐条核对过，**一致**。

### 6.1 公式

| 量 | 公式 | 代码 |
|---|---|---|
| 深度失衡 | $\text{OFI}_{\text{depth}}=\dfrac{\sum_i b_i-\sum_i a_i}{\sum_i b_i+\sum_i a_i}\in[-1,+1]$ | `order_flow_imbalance`，`:162-177` |
| 距离加权深度失衡 | $w_i=\dfrac{1}{1+d_i/\text{DEPTH\_DECAY\_BP}},\quad \text{DwOFI}=\dfrac{\sum w_i b_i-\sum w_i a_i}{\sum w_i b_i+\sum w_i a_i}$ | `depth_weighted_imbalance`，`:180-207` |
| 簿斜率比 | $\bar d^{\text{side}}=\frac1n\sum_i\frac{\lvert p_i-p^{\text{best}}\rvert}{p^{\text{best}}}10^4$；$\text{slope}^{\text{side}}=\frac{\bar q^{\text{side}}}{\max(\bar d^{\text{side}},\,\varepsilon)}$；$\text{ratio}=\text{slope}^{\text{ask}}/\text{slope}^{\text{bid}}$，$\varepsilon=$ `MIN_SLOPE_DIST_BP`$=0.5$ | `book_slope_ratio`，`:210-231` |
| Microprice | $\text{micro}=\dfrac{b_{px}a_{qty}+a_{px}b_{qty}}{b_{qty}+a_{qty}}$ | `microprice`，`:234-245` |
| Microprice 偏离 | $\text{dev}_{bp}=\dfrac{\text{micro}-\text{mid}}{\text{mid}}\times10^4$，$\text{mid}=\frac{b_{px}+a_{px}}2$ | `microprice_deviation_bps`，`:248-254` |
| 相对价差 | $\text{spread}_{bp}=\dfrac{a_{px}-b_{px}}{\text{mid}}\times10^4$ | `spread_bps`，`:257-265` |
| 成交失衡 | $\text{OFI}_{\text{trades}}=\dfrac{\sum\text{buy\_qty}-\sum\text{sell\_qty}}{\sum\text{buy\_qty}+\sum\text{sell\_qty}}$ | `trade_flow_imbalance`，`:322-340` |
| 到达率与活跃比 | $\lambda=\dfrac{n-1}{\text{span}_s}$；$\text{activity\_ratio}=\lambda_{\text{newer}}/\lambda_{\text{older}}$ | `trade_arrival_intensity`，`:404-433` |
| 逐笔已实现波动 | $\text{RV}=\sqrt{\sum_j(\ln p_j-\ln p_{j-1})^2}$（每 `RV_TRADE_STRIDE`$=10$ 笔取样，**每样本、非年化**） | `realized_volatility`，`:379-401` |
| 大单占比 | $\text{large\_share}=\dfrac{\sum_{q\ge Q_{0.9}}q}{\sum q}$（按**成交量**而非笔数） | `trade_size_stats`，`:343-376` |

**微观价格的加权方向（容易写错的地方）.** `microprice` **用对手方数量加权本方价格**：$b_{px}$ 乘 $a_{qty}$、$a_{px}$ 乘 $b_{qty}$（`:245`）。这与 Stoikov 的"size-weighted touch"一致：买压越大，microprice 越靠近卖价。

**衰减权重为什么不是 $1/(1+d_i)$.** 模块 docstring 明写：$1/(1+d_i)$（$d_i$ 以 bp 计）会变成一个近乎二值的"只看最优档"滤波器——10 bp 远的档位只值 0.09——因此被否决（`:29-32`）。选定的 $1/(1+d_i/5)$ 让 5 bp 远的档位恰好携带最优档一半的权重。

**aggressor 方向的 Binance 语义.** `isBuyerMaker = true` 表示**挂单方是买方**，即**主动方卖出**，所以 `aggressor_buy = not isBuyerMaker`（`:21-23`；`aggressor_is_buy`，`:314-319`）。

### 6.2 点对点缓存（keyed on `as_of_ms`）与前视过滤器

**前视过滤器.** `filter_trades(trades, as_of_ms=…, limit=…)`（`:270-294`）是**唯一**执行无前视规则的地方，且在任何统计量之前执行：

$$\text{kept}=\{\text{trade}:\ \text{time}\le \text{as\_of\_ms}\}\ \text{取最新}\ \texttt{MAX\_TRADES}=1000\ \text{条};\qquad \texttt{dropped\_future\_trades}=\#\{\text{time}>\text{as\_of\_ms}\}$$

`time` 不可解析 → 丢弃并计数（`:289-290`）；`as_of_ms=None` → 不过滤。

**点对点缓存.** `MicrostructureCache`（`:489-547`）：

$$\text{key}=(\text{symbol},\ \text{as\_of\_ms}),\qquad \text{TTL}=\texttt{MICROSTRUCTURE\_CACHE\_TTL\_SECS}=5.0\ \text{s},\qquad \text{capacity}=\texttt{MAX\_CACHE\_ENTRIES}=64$$

`put` 满了会淘汰**最旧**条目（`:519-521`）；`get` 过期即 pop 并返回 `None`（`:532-534`）；只缓存**计算后的特征字典，从不缓存原始 payload**（`:490`）。

**为什么键必须含 `as_of_ms`.** 只按 symbol 键的缓存从"调用方传 `as_of_ms`"的那一刻起就是前视 bug：`fetch_features(sym, as_of_ms=t−1h)` 会被 `t` 时刻的 payload 回答（实测：陈旧 payload 的 `as_of_ms` 被原样返回）。含 `as_of_ms` 后，点对点请求只能命中**同一决策时间戳**的 payload；陈旧条目只是 miss（`:494-501`）。`as_of_ms=None` 映射到 `None`（唯一的"现在"情形）。

**簿龄.** `book_age_ms = as_of_ms − book_time_ms`（当两个都给时），`n_depth_levels = len(bids)+len(asks)`（`:479-484`）。深度的年龄被**报告**而不是被强制——调用方必须拒绝陈旧的簿（`:56-58`）。

**有界性.** 每个列表都被截断：`MAX_DEPTH_LEVELS=20`、`MAX_TRADES=1000`（`:113-116`），所以每次决策的算力是 O(1)。`fetch_features`（`:550-…`）接受任何暴露 `order_book` / `recent_trades` 协程的对象，因此测试可以打桩、**没有任何测试触网**（`:71-72`）。

### 6.3 已实测证据

| 断言 | 内容 | 出处 |
|---|---|---|
| 纯函数性 | 每个特征是**单快照**的纯函数，并与手算订单簿一致 | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:82` |
| 无前视 | 注入巨量未来成交后特征**逐位不变** | 同上 |
| 硬上限 | 深度/成交数有硬上限 | 同上 |
| 缓存有界 | TTL 缓存有界 | 同上 |
| 传输失败 | 降级为 `None` | 同上 |
| 测试钉 | `tests/test_microstructure.py` **20** 项 | 同上；我实测 `def test` 计数 = 20，一致 |

### 6.4 已知局限

1. **`MICROSTRUCTURE_ENABLED = False`**（`:108`）：没有任何实盘组件计算这些特征；引擎缝只有调用方显式启用时才计算（`:84-86`）。子代理核对：`MICROSTRUCTURE_ENABLED` **没有任何读取者**（只有定义处 `:108`、docstring `:84`、`__all__` `:603`）。
2. 这 **20** 个 `FEATURE_KEYS`（`:125-131`，我逐项数过 = 20；`docs/overhaul/P6_VOLUME_PLAN.md:18`/`:40` 写 **19**）**不并入** 39 列 ML 契约——它们是流式的、无历史回填，因此对预测路径零贡献（`docs/overhaul/P6_VOLUME_PLAN.md:19`）。见 §11 D-22。
3. 单快照无法检测 spoofing、隐藏流动性或决定挂单是否成交的排队位置（`:78-79`）。
4. `/api/v3/trades` 上限 1 000 条（BTCUSDT 1h 成交量下只有几秒），所以 $\lambda$ 是**突发度量**，不是 session 强度（`:80-81`）。
5. 距离加权与 $k$-笔波动率取样是**文档化的选择，不是估计参数**——没有任何标定证据（`:82-83`）。
6. **`tests/test_microstructure.py` 的 20 项全是纯 stub**（在测试内构造 Binance 形状快照），**没有任何用例端到端跑真实 provider**；测试文件自己写明："nothing here exercises the real provider end to end"（`tests/test_microstructure.py:349-357`），并记录了删掉实盘断言的原因："3 of 40 standalone live fetches returned `arrival_rate_hz == 0.0`"（`:342-347`）。代价是真实 provider 的接线不再有测试覆盖（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:196`，限制 3）。
7. `book_slope_ratio` 用**算术**平均距离与平均数量，而不是按距离积分的深度；$>1$ 的解释（"卖墙"）依赖这个近似。
8. `realized_volatility` 与 P3 的 `realized_vol`/`ewma_vol` **不是重复**：前者测单快照内逐笔路径（$\sum\Delta\ln p^2$，每 10 笔取样），后者测滚动 bar 级波动（`:88-95`）。把两者合成一个年化数需要调用方自己决定口径——**未验证**是否有调用方这样做。
9. **`book_age_ms` 在"现在"这一形状下会是负数（未文档化的不对称）**：`fetch_features` 在两次 `await` **之前**取 `now_ms`（`:576`），却在两次 `await` **之后**取 `book_time_ms = time.time()*1000`（`:588`）。当 `as_of_ms=None`（实盘的"现在"）时，`as_of_ms − book_time_ms` 因往返延迟而为**负**——消费方若用 `book_age_ms > threshold` 判陈旧就永远不触发。这正是 `tests/test_microstructure.py:342-347` 记录的抖动机制；模块里没有任何注释说明它。
10. **成交笔数上限有两个来源**：`compute_features` 用 `tlimit`（受 `trades_limit` 与 `MAX_TRADES` 夹逼，`:454-456`），但 `trade_arrival_intensity` 在内部**不带 `as_of_ms`** 重新用 `limit=MAX_TRADES` 过滤（`:411`），因此到达率的样本可能与其余特征不同（对 `fetch_features` 的默认 `trades_limit=100` 尤其如此——它传的是 100，而到达率看到的是最多 1 000）。**未验证**：我没有构造帧验证这个差是否实际发生。
11. `DEPTH_DECAY_BP` 与 `MIN_SLOPE_DIST_BP` **不在模块 `__all__`**（`:602-611`），而 `docs/core-algorithms/11-pairs-cointegration.md:244-246` 的开关清单也漏了 `MIN_SLOPE_DIST_BP`。

---

## 7. Regime 门控（P4）

> 代码：`core/strategy/regime.py`（893 行）。证据：`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md` §4（`:81`）。

### 7.1 两状态高斯 HMM

**模型.**

$$x_t \mid s_t=k\ \sim\ N(\mu_k,\ \sigma_k^2),\qquad P(s_t=j\mid s_{t-1}=i)=A_{ij},\qquad \pi=(\tfrac12,\tfrac12)$$

**发射对数密度**（`_emission_logpdf`，`:193-196`）：

$$\log b_t(k)=-\tfrac12\log\!\big(2\pi\sigma_k^2\big)-\frac{(r_t-\mu_k)^2}{2\sigma_k^2},\qquad \sigma_k^2\leftarrow\max(\sigma_k^2,10^{-12})$$

**EM（Baum-Welch）更新**（`_hmm_em`，`:326-380`）。用缩放的前向/后向（`_forward_backward`，`:199-218`）得 $\alpha_t,\beta_t$，$\gamma_t=\alpha_t\odot\beta_t$ 归一到 1；再用 einsum 收缩期望转移数：

$$\begin{aligned}
\xi_{ij} &= \frac{\sum_t \alpha_{t,i}A_{ij}b_{t+1,j}\beta_{t+1,j}}{\sum_t(\text{同上对 }i,j\text{ 求和})} & \text{(实现 } :362-367)\\
A^{\text{new}}_{ij} &= \frac{\xi_{ij}}{\sum_j \xi_{ij}},\qquad \pi^{\text{new}}_k=\gamma_{0,k}\Big/\textstyle\sum_k\gamma_{0,k}\\
\mu^{\text{new}}_k &= \frac{\sum_t\gamma_{t,k}x_t}{\sum_t\gamma_{t,k}},\qquad
(\sigma^2)^{\text{new}}_k=\frac{\sum_t\gamma_{t,k}(x_t-\mu^{\text{new}}_k)^2}{\sum_t\gamma_{t,k}},\quad
\sigma^{\text{new}}_k=\sqrt{\max((\sigma^2)^{\text{new}}_k,10^{-12})}
\end{aligned}$$

**确定性初始化**（无随机种子）：$\mu$ 取收益的**四分位**（$q_1,q_3$ 各自半边的均值），$\sigma$ 取各自半边的离散度（`ddof=0`，下限 $10^{-6}$），$A=[[0.95,0.05],[0.05,0.95]]$，$\pi=(0.5,0.5)$（`:341-351`）。收敛判据 $|\ell_t-\ell_{t-1}|<\texttt{HMM\_TOL}=10^{-6}$，最多 `HMM_ITER=50` 次（`:109-110`, `:376-379`）。

> **精度细节（子代理指出）**：M-step（`:373-374`）在收敛判断**之前**执行，因此返回的参数比触发 break 的那个 $\ell$ **多走了一步 M-step**。`used = it + 1`（`:375`）。


**状态重标号.** EM 的状态编号是任意的，所以按 $\sigma$ **升序重排**，使 $0=$ calm、$1=$ stressed，标签跨拟合可比（`:306-314`）。

### 7.2 仅前向（因果）解码，以及为什么 in-sample Viterbi 不可交易

**算法.** `_hmm_forward_only`（`:383-404`）只跑缩放前向递推：

$$\alpha_0=\frac{\pi\odot b_0}{\sum},\qquad \alpha_t=\frac{(\alpha_{t-1}A)\odot b_t}{\sum},\qquad \text{out}[t]=\alpha_t=P(s_t\mid r_{0..t})$$

**后向递推与 Viterbi 被刻意丢弃**（`:387-391`）：它们使 $t$ 时刻的标签成为 $>t$ 的 bar 的函数。`hmm_two_state` 会返回三样东西，语义严格区分（`:250-254`）：

| 返回项 | 语义 | 可否交易 |
|---|---|---|
| `posterior_filtered` | $P(s_t\mid r_{0..t})$ —— **由全体参数决定的因果后验** | 只有参数也因果时才可以 |
| `posterior_smoothed` | $P(s_t\mid \text{全部数据})$ | **仅诊断** |
| `states` | Viterbi 路径（`_viterbi`，`:221-237`） | **仅诊断** |

**因果构造（`hmm_two_state_causal`，`:455-…`）的三条论证**（`:465-483`）：

1. **参数计划**：在 bar 索引 `warmup, warmup+refit_every, …` 用**只含过去**（`returns[:t]`）的窗口重拟合；两次重拟合之间用冻结参数。所以索引 $t$ 处的拟合只依赖 $<t$ 的 bar（`:543-552`）。默认 `HMM_CAUSAL_REFIT_EVERY = 250`、`HMM_CAUSAL_WARMUP = 250`（`:117-118`）。
2. **仅前向解码**：每根 bar 的滤波后验来自**从缓冲区起点**跑到该 bar 的缩放前向递推——没有后向、没有 Viterbi。
3. **Burn-in**：前 `warmup` 根标为 `"unknown"`（滤波后验尚未忘记 $\pi$ 种子），并显式报告在 `warmup` / `first_label_index`（`:481-483`）。

**为什么 in-sample Viterbi 不可交易（量化对比）.**

| 模式 | 合成结构（3000 bar，$\sigma$ 0.002/0.010，变点在 1000/2000）上的准确率 | 出处 |
|---|---|---|
| 全样本 Viterbi（**in-sample**） | **0.9987**（模块 docstring 里写作 "≈99.9 %"） | `core/strategy/regime.py:59`, `:268` |
| 仅前向（**out-of-sample**） | **0.758–0.815** | `core/strategy/regime.py:62`, `:270`, `:591` |

**测试侧的强度（子代理核对，A 级）.** 只有**不等式**被断言：`tests/test_regime.py:116-124`（种子 5）`assert metrics["accuracy"] > 0.95` 且 `all(l is not None and l <= 10 for l in latency_bars)`；`tests/test_p34_audit_fixes.py:147-153` `0.3 < oos["accuracy"] < 0.95`、`ins["accuracy"] > 0.95`、`oos["accuracy"] < ins["accuracy"] - 0.2`。**逐种子的精确值没有被钉住**，而且 `tests/test_regime.py:7-8` 的 docstring 宣传 "≥ 95 % accuracy" 时**没有加 in-sample 限定**，`tests/test_p34_audit_fixes.py:130-136` 则明确标注 in-sample——所以"≥95 %"这个数字在测试文件之间含义不同。

**种子集与数字的三处冲突（见 §11 D-25）.**

* 代码 docstring 用种子 **5/7/11**（`:62`, `:505-506`），doc 11:275-278 的表用 **5/6/7**（0.758/0.759/0.815）。两套都在 `tests/test_p34_audit_fixes.py` 里真实存在（`:779` 用种子 11；`:82`/`:118` 用 6/7）。
* 修复前索引错位（`fwd[k]`）导致的解码准确率在仓库里有**三个互不相容的值**：`core/strategy/regime.py:496-501` 写 **0.52–0.56**（并明确否证了更早的 "0.156"——"does not reproduce on any of those three seeds"）；同一个文件 `:577` 写 **0.16–0.52**；`docs/core-algorithms/11-pairs-cointegration.md:279` 与 `tests/test_p34_audit_fixes.py:774` 仍写 **0.156**。
* 变点延迟：doc 11:281 给 `5/24、29/22、145/27 bar`（种子 5/6/7），其中种子 6 的第二变点是 **22**，而代码 `:63-65` 的范围写 **24–27**——22 落在范围外。
* 另一个"标签反转"的实测（在置换前取 argmax）：**0.17–0.22**（种子 5/7/11），种子 5 上 **0.2234**——"比抛硬币还差"（`:585-588`）。

**参数不稳定性的实测（为什么必须因果）.** 把序列截断到 2000 bar 会把 $\sigma$ 从 0.00202/0.00968 移到 0.00199/0.00968，并在 3 个合成种子的 2 个上**翻转**早期的 Viterbi 标签（`:259-270`）。也就是说：全样本拟合下，"同一根 bar 的标签"取决于你给了多少未来数据。该测试只断言 `moved > 0`，不钉住 2/3（`tests/test_p34_audit_fixes.py:57-86`）。


**一个实测到的索引映射缺陷（已修）.** 修复前代码读 `fwd[k]`（段的第 $k$ 行）而不是 `fwd[tt - start]`，于是 bar `tt` 被贴上 bar `tt - (t-start)` 的后验——滞后 0, 250, 500, …, 2500。实测把 **0.758** 的解码变成 **0.52–0.56**（甚至更差）；当前代码用 `tt - start`（`:580-581`），并有 `lag_bars` 字段报告该偏移（`:493-495`, `:577-579`）。

**另一个实测缺陷（重标号顺序）.** 在**置换前**取 argmax 会反转标签（实测 0.17–0.22，种子 5/7/11；种子 5 上 0.2234，比抛硬币还差）；先按 $\sigma$ 排序再取 argmax 是代数等价的正确做法（`argmax(post[order]) == argsort(order)[argmax(post)]`，逐 bar 验证），解码 0.758–0.815（`:582-591`）。

### 7.3 最小分离守卫

**算法.** `_hmm_fit_separated`（`:428-452`）：先做一次普通 EM，若 $\sigma_{\max}/\sigma_{\min}\ge\texttt{HMM\_MIN\_SIGMA\_RATIO}=1.5$（`:128`）则接受（`separated=True`）；否则用**确定性分位数种子**重做一次并**强加**该比例：

$$s_{\text{lo}}=\max\!\big(\mathrm{sd}(\text{lower half}),\ 10^{-9}\big),\qquad s_{\text{hi}}=\max\!\big(1.5\,s_{\text{lo}},\ \mathrm{sd}(\text{upper half})\big)$$

返回 `(params, separated)`；两次都没达到比例时返回**第一次**的拟合并标 `separated=False`（`:450-452`），计数器 `n_degenerate` 会把它报出来（`:551-552`）。

**为什么需要.** 两状态的 EM 会把两个 $\sigma$ 塌缩到一起（"两个状态"退化成一个），此时解码塌缩到 ~0.50 准确率（`:124` 的注释）。守卫是**参数空间的 floor**，不是新估计量。实测（代码记录）：在 $t=250$ 的窗口上不加守卫时 EM 把同方差样本劈成 $\hat\sigma = 0.00141/0.00178$，而真值是 **0.002/0.010**；"一根 0.5 % 的 bar 会永远被读成 stressed，解码塌缩到 ~0.50"（`:119-127`）。退化次数由 `n_degenerate_fits` 上报（`:539`, `:551-552`, `:611`）。`HMM_MIN_SIGMA_RATIO` **不在模块 `__all__` 里**（`:884-892`），尽管两个时间常量在。

### 7.4 `NonCausalRegimeError`

**算法.** `NonCausalRegimeError(RuntimeError)`（`:776-783`）——"拒绝，不是警告：一个静默给实盘交易打门的前视标签，正是这个文件存在的意义，而一行日志不会阻止它"。`RegimeGate.check_table`（`:829-846`）检查 `table.attrs`：

$$(\texttt{hmm\_present} \wedge \neg\,\texttt{causal\_hmm}) \Rightarrow \text{raise NonCausalRegimeError}$$

例外：`enabled=False`（shipped 默认）或 `require_causal=False`（研究调用方的显式 opt-out）（`:837-839`）。`RegimeGate.allows`（`:811-817`）在 `enabled=False` 时对**一切**返回 `True`；`gate_row`（`:848-855`）与 `gate_regimes`（`:858-…`）是有检查的入口。`classify_regimes` 写入 `attrs`（`:656-659` 的注释说明 `causal_hmm` 标志只有在**加了 HMM 列且该 HMM 是因果拟合**时才为真）。

**为什么这是安全缝.** `allows` 只看**标签字符串**，无法知道标签来自哪里，所以 `check_table` 必须独立强制契约（`:796-803`）。

### 7.5 开关与已实测证据

| 常量 | 默认 | 位置 |
|---|---|---|
| `REGIME_GATING_ENABLED` | **False** | `core/strategy/regime.py:101` |
| `REGIME_DIAGNOSTICS_ENABLED` | **False** | `:103` |
| `VOL_WINDOW` / `TREND_FAST` / `TREND_SLOW` | 50 / 50 / 200 | `:105-107` |
| `MIN_REGIME_ROWS` | 100 | `:112` |
| `HMM_MIN_SIGMA_RATIO` | 1.5 | `:128` |
| `hmm_two_state(causal=None)` | 解析为 `REGIME_GATING_ENABLED` | `:274-280` |

即：门关闭时全样本拟合（与旧行为逐位相同），门一打开就自动切到因果路径——"门永远不会在没要求的情况下消费看过后来的标签"（`:274-278`）。

**归档实测：** 已知三 regime 合成序列上 HMM 恢复已知波动率、**≥95 % 准确率**、延迟数根 bar；tercile 分类器因果；两次运行逐位一致；门默认关闭且关闭时全放行（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:81`）。`tests/test_regime.py` 我实测 **14** 项 `def test`，与文档一致。

**成本（代码记录）.** 因果解码 3 000 bar、默认参数（250 bar 重拟合 / 250 bar warm-up）= 12 次拟合 + 12 次扫掠，实测 **≈2.3–2.6 s**（`:503-512`）。作者**明确否证**了 `docs/core-algorithms/11-pairs-cointegration.md` 里写的 "≈18–21 s"——"not in this file and does not reproduce"（`:508-510`）。这本身是一处**文档 cite 错误**（把 regime 的数字写进了 pairs 文档），见 §11 D-8。

### 7.6 已知局限

1. 只有 2 个状态、只有波动率（不是收益均值）作为区分维度；`mu` 也拟合了，但状态说明被解释为"calm/stressed"。⚠️ `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:14`/`:81` 写"HMM **三 regime** 合成序列 ≥95 % 准确率"——合成数据有**三个**变点区间，但**模型是两状态**（`hmm_two_state`，`mu`/`sigma` 都是长度 2），且那个"≥95 %"是 **in-sample** 数字（因果路径是 0.758–0.815）。见 §11 D-11 与 D-25。
2. 门默认关闭；关闭时既无收益也无风险（与 P3 同样的 off-by-default 纪律）。`REGIME_GATING_ENABLED` 是本文少数**真的有读取者**的总开关（`:280`, `:807`, `:878`），而 `PAIRS_ENABLED` 与 `MICROSTRUCTURE_ENABLED` 都没有（§11 D-26）。
3. 因果路径有 **250 bar 的 warm-up**（标 `"unknown"`），在 1h 上约 10 天无标签。
4. 因果解码的成本（**≈2.3–2.6 s / 3 000 bar**，12 次 EM 拟合 + 12 次前向扫掠，`:505-507`）不是逐 bar 成本，但也不是免费；`REGIME_DIAGNOSTICS_ENABLED=False` 意味着实盘根本不跑它。doc 11:286-287 给的是同一量级（≈2.30–2.52 s）。
5. `detection_metrics`（`:727-773`）报告 `latency_bars`（在变点后需要多少根 bar 才连续 `settle_run=5` 次正确），这是**正确**的口径（"99 % 正确但晚 200 根 bar 的检测器不能给交易打门"，`:744-745`），但它需要调用方提供 `change_points`。
6. `classify_regimes` 写入 `attrs`（`:660-664`）：默认 `causal_hmm=True, hmm_present=False`；加了 HMM 列时 `causal_hmm = bool(fit.get("causal"))`。`NonCausalRegimeError` 由 `tests/test_p34_audit_fixes.py:167-183` 钉住。
7. **死代码**：`_hmm_forward_last`（`:407-425`）无任何调用者（§11 D-26）。
8. **未验证**：我没有跑 `hmm_two_state_causal`（3 000 bar 需 2–3 s；本次没跑）；上述准确率是**代码 docstring 记录的实测**（B 级），不是我这次的读数。合成表的 tercile/trend 准确率列（doc 11:258-262）是 doc-only。

---

## 8. 执行真实性（P6-A）

> 证据主文件：`docs/core-algorithms/13-volume-liquidity-costs.md`、`docs/overhaul/P6_VOLUME_PLAN.md` §3（`:55-65`）。

### 8.1 参与率

**算法.** `recent_quote_volume`（`core/risk/liquidity.py:283-364`）：

$$\text{quote\_volume}=\sum_{i\in\text{最近 }N\text{ bars}}\begin{cases}\text{quote\_volume}_i & \text{若帧有该列}\\ \text{volume}_i\cdot\text{close}_i & \text{否则（代理）}\end{cases}$$

$$p=\frac{|\text{notional}|}{\text{quote\_volume}}\quad(\text{分数}),\qquad p\%=100\,p$$

$N=$ `lookback_bars`（默认 `DEFAULT_LOOKBACK_BARS = 20`，`:83`）。非有限 bar 被**丢弃**而不是毒化整个窗口（`:340-342`）；取**尾部** `values[-window:]`（`:337-339`）；价格序列被裁到**同一**尾部窗口（`:357-359`）——这是"20 bar 回看不该拿全历史价格配对"的修复。来源标记 `volume*close` / `quote_volume` / `volume`（`:259-269`）。窗口未知/为空 → 返回 **0.0**，并由 `participation_pct` 映射为 **`None`**（`:380-386`，`core/risk/liquidity.py:49-52` 解释：`None` 是唯一能表达"未知"的值，`0.0` 会被读成"没成交"，恰好相反）。

**参与率上限.** `cap_notional`（`:389-433`）返回 `(allowed, reason)`：

$$\text{allowed}=\min\Big(\text{notional},\ \frac{\texttt{max\_participation\_pct}}{100}\cdot\text{quote\_volume}\Big)$$

四种 reason（`:393-405`）：`"ok"`（原值**逐位**返回）、`"capped"`（被缩小）、`"no_volume"`（**拒绝**，返回 0.0——缺少分母时静默放行的上限不是上限）、`"disabled"`（上限 $\le0$，直通）。`notional<=0` 直通（`:416-418`）。

**为什么 fail closed.** 在回测或低流动性对上，"没有可测量的簿"时**拒绝**是保守选择（`:400-403`）。

**已知局限.**
1. 参与率针对**已成交**量度量，不是**挂单**流动性；看不到薄簿、假档，或穿价时变宽的价差（`docs/core-algorithms/13-volume-liquidity-costs.md:253-255`）。
2. 窗口是 **bar 聚合**，不是逐笔 tape：1h 的 20 根平滑掉突发——20 bar 均值不是下一分钟的样子（`:256-258`）。
3. 缓存**没有** `quote_volume` 列，所以跨币种可比量能只能靠 `volume×close` 代理（`docs/overhaul/P6_VOLUME_PLAN.md:41`, `:48`）。P6-B 的第一项任务就是补该列。
4. **实盘定仓路径当时未接线**：`RiskManager` 不传 `recent_quote_volume`，所以参与率上限只被 sizer API 与测试触发（`docs/core-algorithms/13-volume-liquidity-costs.md:260-265`）。
5. `per_symbol` 覆盖是**部分** dict 合并（`liquidity_for_symbol`，`:185-220`），一个写错键名的覆盖会被静默忽略（不在 `LiquidityConfig.__slots__` 的键被跳过，`:217-219`）。

### 8.2 平方根冲击律

**算法.** `impact_pct`（`:439-497`）：

$$\text{impact\_pct}=\mathrm{clip}\big(k\cdot p^{\,e},\ \text{floor},\ \text{cap}\big)\qquad[\text{% of that side's notional}]$$

$k=$ `impact_k`、$e=$ `impact_exponent`（默认 0.5）。`_impact_one`（`:480-497`）的实现顺序：$k\le0$ 或 $p$ 未知/非正 → 返回 `floor`（有 cap 时取 `min(floor,cap)`）；$k\cdot p^{e}$ 溢出/非有限 → `inf`（若 $k>0$）；最后 `max(value, floor)` 再 `min(value, cap)`（cap 非 None 时）。

**往返冲击（USDT）.** `total_impact_usdt`（`:561-586`）：

$$\text{impact\_usdt}=\sum_{\text{side}\in\{\text{entry},\text{exit}\}}\frac{\text{impact\_pct}(p_{\text{side}})}{100}\cdot\text{notional}_{\text{side}},\qquad p_{\text{side}}=\frac{\text{notional}_{\text{side}}}{\text{quote\_volume}}$$

**两边各自计价**（退出名义因 PnL 而不同），所以 50 000 → 51 000 的往返略高于入场侧数字的两倍（`docs/core-algorithms/13-volume-liquidity-costs.md:55-59`）。`trade_impact_pct`（`:531-558`）是同一件事以**入场名义的百分比**表达。

**为什么是平方根.** $e=0.5$ 是经验形状（Almgren-Chriss 2000；Grinold & Kahn 第 16 章）：冲击随规模增长但**次线性**，所以四倍大小的订单大约两倍每单位冲击（`core/risk/liquidity.py:91-92`, `:19-20`）。它是**经验形状，不是自然律**：在非常高参与率下冲击更接近线性，`impact_exponent` 因此可配置（`docs/core-algorithms/13-volume-liquidity-costs.md:267-268`）。

**成本分解.** `cost_model.apply_trading_costs`（`core/backtest/cost_model.py:310-404`）：

$$\text{legacy}=\underbrace{f\cdot \text{entry}+\underbrace{f\cdot\text{exit}}_{\text{手续费, per side}}+\underbrace{\frac{s}{2}\text{entry}+\frac{s}{2}\text{exit}}_{\text{半价差, per side}}\ ;\qquad \text{total}=\text{legacy}+\text{impact\_usdt}$$

其中 $s$ 是**全额**报价价差（表内单位），单边收 $s/2$（`:382-385`；语义来自 `sim_cost_quote`，`core/ml/credibility.py:101-106` 有同一口径的注释）。`k<=0` 或 `recent_quote_volume is None` 时**短路**到 legacy（`:394-396`）。`total_costs_with_impact`（`:446-484`）把三项**分开**报告：`{fees_usdt, spread_usdt, impact_usdt, total_usdt, impact_pct, recent_quote_volume, impact_k}`，其中 `impact_pct = impact/entry_notional*100`（`:482`）。

**实测（税前/费后语义）.** BTCUSDT 成交 50 012.5000、往返 **0.25005 %**；ETHUSDT 50 015.0000、**0.26006 %**（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:86`）。

### 8.3 实测容量例子

**订单规模阶梯（`docs/core-algorithms/13-volume-liquidity-costs.md:188-208`；窗口 **751 885 467.06 USDT**，ends 2026-09-30 06:00:00，price 83 043.14）**

| 规模 | 参与率 | $k=0$（pre-P6） | $k=0.1$ Δ | $k=0.5$ Δ |
|---|---|---|---|---|
| 0.01 BTC（830.43 USDT） | 0.000110 % | 0（base = legacy） | **+0.0017**（+0.0002 %） | **+0.0087**（+0.0011 %） |
| 1.0 BTC（83 043.14 USDT） | 0.011045 % | 0 | **+1.7455**（+0.0021 %） | **+8.7273**（+0.0105 %） |
| 10 BTC（830 431.40 USDT） | 0.110447 % | 0 | **+55.1963**（+0.0066 %） | **+275.9814**（+0.0332 %） |
| 100 BTC（8 304 314.00 USDT） | 1.104465 % | 0 | **+1 745.4596**（+0.0210 %） | **+8 727.2978**（+0.1051 %） |

**$k=0$ 与 legacy 逐位相同**（用 `==` 而非 `approx`）：实测 `75.48621426 == 75.48621426`（1.0 BTC 往返，83 043.14 → 84 704.00；`:210-214`）。

**单笔分解（`total_costs_with_impact`，0.6 BTC 往返 @ 83 384 → 85 000，同一窗口，$k=0.1$）**：fees **40.4122**、spread **5.0515**、impact **0.8281**、total **46.2918**、legacy 45.4637（`:226-235`, `:198-200`）。

**早期表格（$k=0.1$，另一窗口）**：BTCUSDT 1h 20-bar 窗口 750 661 812.73 USDT；500 USDT 单 → 参与率 0.000067 %、impact 0.000082 %/side、往返 0.0008 USDT；50 000 USDT → 0.006661 %、0.000816 %/side、0.8161 USDT。XRPUSDT 1m 窗口 2 657 574.07 USDT；500 USDT → 0.018814 %、0.001372 %/side、0.0137 USDT；**50 000 USDT → 参与率 1.881415 % → 被缩到 26 575.74 USDT**（恰为窗口的 1 %）（`:124-140`）。

> **⚠️ 一处已归档的重大数字更正（必须读）.** `docs/core-algorithms/13-volume-liquidity-costs.md:145-162` 记录了：此前该节写 1.0 BTC 往返 `+178.8965`（+2.36 %）/ `+894.4824`（+11.82 %），"钉在同一 20 根窗口"，但**同页引用的窗口给出的是 +1.7455（+0.0021 %）/ +8.7273（+0.0105 %）**。反解：`+178.8965` 隐含窗口 **71 576.03 USDT**（$7.16\times10^4$，比原主张小一个数量级），`+894.4824` 隐含 **2 863.04 USDT**；退役的两个数在 1:10 规模步长下呈 1:5 比例，即**对规模线性**，而 shipped 平方根律按 $\text{size}^{1.5}$ 缩放（×31.62）。退役的 `0.8288` 行**确实**属于该表命名的窗口（在该窗口复现为 **0.828812**），而 729.2 M 窗口给 **0.840920**（`:238-242`）。**但 `docs/overhaul/P6_VOLUME_PLAN.md:61` 仍然写着旧数字** `k=0.1 +2.36 % / k=0.5 +11.82 %`——见 §11 D-9。

### 8.4 开关与已实测不变量

| 开关 | 默认 | 关闭时效果 |
|---|---|---|
| `risk.liquidity.enabled` | **false** | `PositionSizer` 从不调用参与率 hook；定仓逐位相同 |
| `risk.liquidity.max_participation_pct` | **1.0**（percent） | inert |
| `risk.liquidity.lookback_bars` | **20** | inert |
| `risk.liquidity.impact_k` | **0.0** | `apply_trading_costs` 短路到 legacy |
| `risk.liquidity.impact_exponent` | **0.5** | 在 $k=0$ 时 inert |
| `risk.liquidity.per_symbol` | **{}** | 无覆盖 |

（我本次核对过 `config/config.yaml:207-227`，与上表一致。）

**归档不变量实测：** 关闭路径逐位一致（`test_participation_disabled_is_bit_identical`：`enabled:false` 时 `recent_quote_volume ∈ {None, 0.0, 123.0, 7.5e8, [1,2], lambda}` 返回同一 tuple；`test_impact_k_zero_is_bit_identical`：`k=0` 与无 volume 都复现 legacy，含 `repr`）；`cap_notional` 实测 **0.917–0.940 µs/call**，完整 20-bar-frame 路径（frame → sum → participation → impact）**1.887–1.936 µs/call**，断言 `< 200 µs`（`CALL_COST_BOUND_US`）（`docs/core-algorithms/13-volume-liquidity-costs.md:295-304`）。

**测试钉.** `tests/test_liquidity.py` 我实测 **26** 项 `def test`；`docs/core-algorithms/13-volume-liquidity-costs.md:291` 写 "24 passed"，`docs/overhaul/P6_VOLUME_PLAN.md:58` 也写 "24 项"，而证据索引 `:18`/`:209` 记录 25 → 本轮 **26** 项。见 §11 D-10。

### 8.5 已知局限

1. **系数未标定**：$k=0.1$ 是"为了让算术可读"的文档化例子；shipped 默认是 $0.0$。真实标定需要 trade-and-quote 数据（实现滑点 vs 参与率），本次部署**没有**（`docs/core-algorithms/13-volume-liquidity-costs.md:245-250`）。
2. **没有订单簿深度模型**：参与率对着**成交量**度量，不是挂单量（同上 `:251-255`）。
3. **回测本身还没有把逐 bar 成交量喂给 `apply_trading_costs`**——缝只差一个关键字参数，但没接（`:260-265`）。
4. 实盘 `RiskManager` 未传 `recent_quote_volume`（同一条）。
5. `impact_pct` 的 `floor`/`cap` 参数在 `apply_trading_costs` 路径上**没有被传入**（`cost_model.py:401-403` 只传 volume 与 $k$），因此 `risk.liquidity` 里没有 floor/cap 的配置项——这是一个未暴露的能力。

---

## 9. 数据完整性（横切 / data integrity）

> 代码：`core/market_data/ohlcv_cache.py`（525 行，**正被兄弟代理编辑**）、`scripts/check_data_integrity.py`（344 行）、`core/risk/manager.py::_series_has_gap`。

### 9.1 账本恒等式（ledger identity）

**陈述.** OHLVCache 的盘上文件是**磁盘帧与内存帧的并集**（`OHLVCache.save` → `merge_history`），因此"只知道较短窗口的写入者永远不能截断更长历史"（`core/market_data/ohlcv_cache.py:7-9`, `:417-418`）。第二条规则是并集必须**按 bar 键**（不是精确时间戳）：同一根 bar 用两种时间戳口径（bar-**open** 与 Binance `close_time` $=\text{open}+\text{length}-1\,\text{ms}$）存储时是**一行**，不是两行（`:10-12`）。

**为什么（实测缺陷）.** 实盘 `data/market/BTCUSDT/1h.parquet` 在 revision `0542e02` 实测 **11 677 行 = 8 767 个 bar-open 戳 + 2 910 个 `HH:59:59.999` close 戳 = 55 棵 bar 被存两次**。精确时间戳的并集看不到这种对——两个戳相差 3 599.999 s——所以运行中服务的周期性 flush 在修复移除它们之后又把重复行放回去了（`:14-18`）。

**为什么不能用"邻近"规则.** "戳间距小于区间的一个比例就算同一根 bar"不但修不好，还会毁数据：同一文件里 **54 对 1 毫秒相邻**的戳是小时 $H-1$ 的收盘与小时 $H$ 的开盘——**两根不同的 bar**。1 秒容差会把那 54 个真实小时全部合并，同时把 55 个真正的重复**全部留下**（`:20-26`）。所以唯一可靠的依据是**声明的 bar 长度**。

### 9.2 bar 键折叠（`bar_keys`）

**算法.** `bar_keys(index, interval)`（`:109-141`）：

1. `"1M"` → 取日历月起点（无固定长度，`:129-130`）；
2. `step_ns(interval)`（`:69-74`）查 `BAR_LENGTH_NS`（`:46-62`，与 `scripts.download_history.INTERVAL_LENGTH_NS` 逐字镜像、由 `tests/test_cache_durability.py` 钉住两份相等，`:40-45`）；未知或 $\le$ 1 ms → 返回 **`None`**（调用方退回精确时间戳并集）；
3. 锚点 `_grid_anchor_ns`（`:92-106`）：取**第一个落在规范残差**上的戳（残差 $0$ = bar open，或 $\text{step}-1\text{ms}$ = bar close），否则取 `ns[0]`；
4. 折叠 `_fold_to_bar`（`:77-89`）：

$$\text{off}=\text{step}-\text{tol},\quad \text{res}=(\text{stamps}-\text{base})\bmod\text{step},\quad
\text{shift}=\begin{cases}+\text{off} & \text{res}\ge\text{step}-\text{tol}\\ -\text{off} & \text{tol}\le\text{res}<2\,\text{tol}\\ 0 & \text{否则}\end{cases}$$

（`tol = CONVENTION_GAP_NS = 1_000_000`，即 1 ms，`:66`；它是**残差容差，不是邻近容差**，`:64-65`。）

5. 若锚点本身在 close 残差上，把键整体移到 bar-open 网格（`:137-140`），使键在任何地方含义一致。

键**只用于分组**，从不写回；存活行的时间戳由 `dominant_convention` / `_canonical_ns` 单独决定（`:117-124`）。

### 9.3 写时合并（merge-on-write）与去重

**算法.** `_canonical_write`（`:225-250`）+ `_frame_hash`（`:192-…`）+ `merge_history`（`:252-…`）：

1. `combined = merge_history(existing, incoming, interval)`（`:242`）；
2. `changed` 由**合并后**的帧（不是合并的输入）判定（`:237-241`）——理由：一个已经过时、但已是磁盘子集的内存窗口，合并结果与磁盘**逐位相同**，不该写盘；
3. `merge_history` 按 `bar_keys` 折叠后再去重：`merged[~merged.index.duplicated(keep="last")]`（`:309`），冲突 bar 保留**较新**的那一行（`:275`, `:324-338` 的"canonicalising 永远不会把两根**不同**的 bar 合并"不变式守卫）；
4. 无可用网格（`bar_keys` 返回 None）→ 退回精确时间戳去重（`:304-311`）。

`flush_all`（`:454-…`）两趟：先写 dirty 键，再对**所有已加载键**做一次去重（`dedupe`，`:499-…`），且只在内容真的改变时落盘。这条第二趟是**复核 R3 的修复**：根因是"`flush_all` 只重写 **dirty** 键，永不追加的文件永远不会被去重"（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:167`）。

**归档实测（R3）.** 实盘 `BTCUSDT/1h.parquet`：**11 678 → 11 623 行**，`twin_bars` **55 → 0**，`twin_rows` **110 → 0**，bar 键集合**完全相同**（11 623），55 棵冲突 bar 全部保留**较新**那一行；`check_data_integrity` 该行转 ok、`twin` 列 0；**第二次 flush 不写盘**（`dedupe → False`），文件**字节完全相同**（sha256 `0543b03d1ed8c3ff`）。后续复测：闭环审计 **11 624 行 / 11 624 bar 键 / 0 twin**；末轮（测量时刻 2026-09-30 17:28）**11 625 行 / 11 625 bar 键 / 0 twin**，原始与 bar 键相邻差全为 **3 600.0 s**（`:167`）。

### 9.4 缺口与孪生检查

**算法.** `scripts/check_data_integrity.py`：

* `duplicate_bar_rows(df, interval) = len(df) - bar_keys(df.index, interval).nunique()`（`:86-97`）——**按声明 bar 长度**，绝不用戳邻近；
* `gap_report`（`:100-171`）：先把时间戳折到 bar 键（`:140-150`），排序后
  $$\text{expected}=\mathrm{round}\Big(\frac{\text{span\_h}}{\text{bar}}\Big)+1,\qquad \text{missing}=\max(\text{expected}-\text{bars},0),\qquad \text{gaps}=\{\text{step}>\texttt{threshold\_bars}\times\text{bar}\}$$
  默认 `DEFAULT_GAP_THRESHOLD_BARS = 1.5`（`:78`）；`flagged = gap_count > 0`；
* `vol_report`（`:174-199`）：同一帧的 **clipped**（$\sigma=6$）与 **unclipped**（$\sigma=0$）RiskMetrics EWMA，**先把 close 换成对数收益**——第一版这个函数喂的是价格水平，报的是"价格的波动率"而不是收益的波动率（`:182-185`）。

**Guard 行为.** `--check-vol` 对**有缺口**的文件**拒绝**打印未裁剪波动率（打印 `REFUSED`），`--strict` 把它变成非零退出码。理由：无法重新抓取的缺口（`download_history.py --merge`）绝不能被静默定价进定仓/止损宽度（`:39-46`, `:312-317`）。`twin` **只报告不强制**：`--strict` 不为孪生行退出非零，因此一个混合口径的缓存不会打破既有 CI 门（`:32-35`, `:335-339`）。

**我本次实测（`c703b8b`，2026 年运行，活缓存）.**

```
$ python scripts/check_data_integrity.py
Cache root      : E:\Codes\Binance Trader\data\market
Gap threshold   : > 1.5 x bar length
Files inspected : 29
RESULT: 25/29 file(s) carry a gap beyond 1.5 x bar length.
（无 NOTE 行 → twin 列全为 0，twin 文件数 0/29）
```

**干净文件（我这次读到的 `ok` 行）**：`USDCUSDT/5m`（500 bar / span 41.6 h）、`ZECUSDT/5m`（500 bar / 41.6 h）。其余 27 个文件里至少 25 个带缺口；两个我无法从尾部输出确认（输出被截断到尾部 45 行）。缺口量级示例：`XRPUSDT/1h`（8 854 bar、missing 2 774、12 gaps、最大 1 484.0 h）、`XRPUSDT/1m`（531 216 bar、missing 166 462、80 gaps、最大 1 482.9 h）。

**归档计数（同一脚本，不同时刻）**：基线 `2751bbb` **25/29**；最终独立审计 `fe11ccf`+工作树 **24/29**；复核 `b49883b`+工作树 **24/29**；闭环审计 `09125bd` 原样 **25/29**（新增的一处是 close 口径尾行误报，D1 修复后回到 **24/29**）；干净文件清单为 `ADAUSDT/1h`、`BTCUSDT/1d`、`BTCUSDT/1h`、`USDCUSDT/5m`、`ZECUSDT/5m`；`twin` 列 **0/29**（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:16`, `:18`, `:114`）。

> **注意**：这些计数**只对测量时刻成立**——`data/market/**` 由运行中的进程持续重写（`:114` 原文明写）。我这次的 25/29 与归档的 24/29 差异**可能**来自：(a) `ADAUSDT/1h`/`BTCUSDT/1d` 在我这次读到的时刻也带了缺口，(b) 活文件新增了 bar。我没有逐文件比对两个时刻，因此**不主张**是哪一种。

### 9.5 已实测证据汇总

| 项 | 值 | 出处 |
|---|---|---|
| 实盘 1h 文件形状（末轮归档） | 11 625 行 / 11 625 bar 键 / 0 twin / `missing`=0 / 相邻差（原始与 bar 键）均 3 600.0 s | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:95`, `:167` |
| twin 修复 before→after | 11 678 行 / 11 623 bar / 55 重复 / 110 twin 行 → 11 623 行 / 0 twin | `:167` |
| 二次 flush | 不写盘、文件字节完全相同 | `:167` |
| D1（close 口径尾行）before→after | `_series_has_gap=True`、预测 None、全仓 25/29 → 重建帧 `False`、预测 `0.07684716805517097 %/bar`、24/29 | `:181` |
| 非时间索引（L6） | `RangeIndex`/`Float64Index`/object 字符串/tz 混合 → **True**（拒绝）；`"7h"` → **False**；空或 <3 根合法日期索引 → **False** | `:188` |
| 缓存文件总数 | **29**（我本次 `Get-ChildItem -Recurse -Filter *.parquet` 实测） | 我运行的命令 |
| 模型产物 | **15 个 `.pkl` / 0 个 `_meta.json`** | 我本次实测，与 `:96`/`:128` 一致 |

### 9.6 已知局限

1. **600-bar 可见性**：实盘预测只看最后 600 根 bar，所以折叠后的闸门只能拒绝落在该窗口内的洞（§3.8 已详述）。
2. **24/29（归档）或 25/29（本次）缓存文件仍带历史缺口**，其中最大的是 2026-07-16/07-26/09-29 附近的 1 484 h 级拼接。修复口径是 `python scripts/download_history.py --symbols <SYM> --intervals <tf> --start <first> --end <last> --merge`（`:114`, `:330-331`）。
3. 计数**不稳定**（活文件）。
4. `--strict` **不**为 twin 行失败，因此"混合口径缓存"可以存在于 CI 绿灯下。
5. `gap_report` 对**无法读取网格**的帧"按原样报告"（`:149-150` 吞掉异常），所以一个坏索引进到脚本里会被静默地按原始戳处理——而 `_series_has_gap` 在同样情况下会**拒绝**（fail closed，`core/risk/manager.py:118-119`）。两个消费者的失败方向**不同**，这是一个真实的语义缝隙。
6. `expected = round(span/bar) + 1` 用了 `round` 而不是 floor/ceil，跨 DST 或非整小时边界时可能偏 1 bar。
7. **未验证**：`core/market_data/ohlcv_cache.py` 正被兄弟代理编辑，我只读了当前盘上内容（`:1-145`, `:192-350` 的关键段落），没有逐行核对全部 525 行。

---

## 10. 什么没有实现 / 什么被证伪（What is NOT implemented / what is falsified）

本节只收录**仓库自己声明的**残余与证伪；逐条给出出处。

### 10.1 明写的"未实现 / 未接线"

| # | 残余 | 出处 |
|---|---|---|
| 1 | **实盘预测只看得见最后 600 根 bar**（`manager._VOL_HISTORY_BARS`），因此折叠缺口闸门对文件中间的洞不可见 | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:194`；`core/risk/manager.py:16-22` |
| 2 | **杠杆与评估口径不一致**：`config/risk_params.yaml` 是 `leverage: 2` / `max_leverage: 4`，而 `core/ga/*.py` 与 `core/backtest/engine.py` 中 `leverage` 命中 **0** 次 → GA/回测按现金模型计价，实盘盈亏与回撤约为其 2×。**未修** | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:115` |
| 3 | **`impact_k` 未标定**，shipped 默认 `0.0`；没有 L2 历史数据用于标定 | `docs/core-algorithms/13-volume-liquidity-costs.md:245-255`；`docs/overhaul/P6_VOLUME_PLAN.md:29` |
| 4 | **没有订单簿深度历史**；参与率对"已成交量"而非"挂单量"度量 | 同上 `:251-255`；`docs/overhaul/P6_VOLUME_PLAN.md:29` |
| 5 | **四个 P4 能力 + P3 + ML 全部默认关闭**（`MICROSTRUCTURE_ENABLED=False`、`META_LABELING_ENABLED=False`、`PAIRS_ENABLED=False`、`REGIME_GATING_ENABLED=False`/`REGIME_DIAGNOSTICS_ENABLED=False`、`risk.vol_targeting.enabled: false`、`risk.liquidity.enabled: false`、`impact_k: 0.0`、`ml.enabled: false`） | `core/market_data/microstructure.py:108`；`core/ml/meta.py:102`；`core/strategy/pairs.py:112`；`core/strategy/regime.py:101-103`；`config/config.yaml:80`, `:156`, `:209`, `:223`（我本次逐条读到） |
| 6 | **缓存只有 `open/high/low/close/volume`，没有 `quote_volume` / `trade_count`** | `docs/overhaul/P6_VOLUME_PLAN.md:41`, `:48` |
| 7 | **`risk.vol_targeting.barrier_*` 三个键当前 inert**（`barrier_widths_pct` 无生产调用者） | `core/risk/position_sizer.py:164-183`；`config/config.yaml:182-195`；`docs/core-algorithms/10-volatility-targeting.md:310-323` |
| 8 | **回测引擎未把逐 bar 成交量喂给 `apply_trading_costs`**；实盘 `RiskManager` 未传 `recent_quote_volume` | `docs/core-algorithms/13-volume-liquidity-costs.md:260-265` |
| 9 | **P6-B/C/D 未完成**：量能特征契约 v2、美元棒/量钟采样、GA 量能基因与可执行性进入适应度 | `docs/overhaul/P6_VOLUME_PLAN.md:67-134`（B/C/D 三节无 ✅） |
| 10 | **路由基线无自动化回归门**：`tests/` 中无任何用例引用 `route-baseline.json` 或 `regen_route_baseline.py` | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:116` |
| 11 | **`OrderExecutor._forecast_vol_cache` 无 LRU 上界**（只靠 300 s TTL 与已推送 symbol 数） | `docs/core-algorithms/10-volatility-targeting.md:298-305` |
| 12 | **未做 A/B 回测对照**：P3 只交付机制与回归测试，"目标化后 Sharpe/最大回撤是否改善"未回答 | `docs/core-algorithms/10-volatility-targeting.md:365-368` |
| 13 | **`tests/test_microstructure.py` 的 20 项全是纯 stub**，没有用例端到端跑真实 provider | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:196` |
| 14 | **无 Student-t / EGARCH**；IGARCH 回退分支不能表达均值回复 | `core/ml/volatility.py:976-983`；`docs/core-algorithms/10-volatility-targeting.md:374-376` |
| 15 | **`engine.py` 的 `ml_accuracy_pct` 把中性带当看跌计分**（有看涨偏差）；正确口径已在 `core.ml.credibility.ml_accuracy_neutral_abstention` 实现但**未接线**（`engine.py` 属另一 agent 写范围） | `docs/core-algorithms/08-ml-triple-barrier.md:216-222`；`core/ml/credibility.py:938-995` |
| 16 | **P5 独立只读复核未在本证据范围内执行** | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:117` |
| 17 | `core/strategy/volume_bars.py`（任务书提到的兄弟代理新增文件）在我开始工作时**不存在**（`Test-Path` 为假）；写作期间它作为未跟踪文件出现，我**未读** | 我本次实测 |
| 18 | **P6-B 工作树改动带来的一致性风险**：`core/ml/features.py` 正在从 39 列扩到 52–54 列，而 `tests/test_ml_credibility.py:377` 仍断言 `X.shape[1] == len(FEATURE_NAMES) == 39`；`core/ml/labels.py:238-240` 的 `timeout_label` docstring 与实现冲突（§11 D-19） | 我本次实测 + 子代理核对 |

### 10.2 明写的"被证伪"（falsified）

| # | 被证伪的主张 | 实测反证 | 出处 |
|---|---|---|---|
| F1 | "方向可预测" | BTC OOS AUC **0.5207**、ETH **0.5342**（门槛 0.55）；净期望 **−0.2494 % / −0.1822 %**；$t$ **−4.46 / −2.28**。**ML 保持关闭是实测结论，不是默认值** | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:69`, `:73` |
| F2 | "legacy ML 流水线有效" | legacy 准确率 **0.6597（ETH）/ 0.6834（BTC）** 对多数类 **0.7773 / 0.8140**，即低于基线 **11.76pp / 13.06pp** | `:66`, `:69` |
| F3 | "meta-labeling 能在本数据上救回方向" | **10/10 全部拒绝**；最高 AUC **0.5167 < 0.55**；净期望 **−1.246 % … 0.000 %**；$t$ 最高 **−1.41** | `docs/core-algorithms/12-meta-labeling.md:81-83` |
| F4 | "1h 主流币对协整可交易" | **0 / 30** 通过 5 % 水平（10 币对 × 1h/4h/15m）；最好 BTC/XRP 15m $p=0.168$ | `docs/core-algorithms/11-pairs-cointegration.md:136-138` |
| F5 | "GA 评分让 5 笔全胜压过 200 笔是合理的" | 旧公式 `490.85` 对 `7.30`；收缩后 5 笔全胜不再压过 200 笔（`tests/test_ga_credibility.py` 断言） | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:58` |
| F6 | "GA 曲线上升 = 策略变好" | 第 1 代最优与第 4 代领先者**交易结果完全相同**（75 笔 / Sharpe 9.4388 / DSR 0.2119 / 回撤 0.12 % / 收益 1.56 %），fitness 差 4.00 **恰好**等于复杂度罚项差（12.9→8.9）。即"更简约"，**不是**"更赚钱" | `:41-42` |
| F7 | "自由 ω 的 GARCH MLE 无界、优化器都奔向角落" | 三个优化器返回**同一**拟合（$\omega\approx0.0813$ %²、$\alpha\approx0.1965$、$\beta\approx0.1546$，6 位一致），且不拒绝任何东西；**病态的是 IGARCH 网格**（角落目标 52.209 对 MLE 的 −0.5726） | `core/ml/volatility.py:826-842` |
| F8 | "PSR 读偏度/峰度"（旧声明） | 旧实现是 $\Phi(\text{mean}/\mathrm{se})$、**不读**任何高阶矩，因此 AND 门恰好等价于 $t>2$，`PSR>=0.95` 是死条件。**现在真的读了**（厚左尾 $t=2$ → PSR **0.9479777894541446** 被拒，正态近似 0.9772498680518209） | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:134`, `:168` |
| F9 | "GA 每代 best 恒为 −1.30" | 真实缓存 4 代 best `5.9411→5.9411→9.9411→9.9411` | `:31-34` |
| F10 | "旧时序 80/20 切分是样本外" | 训练与测试**共享 2–9 根前向窗口 bar**，75–95 % 标签重叠；`overlap_count` 度量之 | `core/ml/evaluation.py:3-6`, `:130-145` |
| F11 | "per-fold 校准流在工作"（第一版 P2） | 该校准流是**死代码**：`calibrator.n_fit == n_oos == 600` 而 `sum(n_cal) == 477`；in-fit ECE 0.0000 对真留出 0.0784 | `core/ml/credibility.py:30-43` |
| F12 | "1.0 BTC 往返冲击 +2.36 % / +11.82 %" | 同页窗口给出 **+1.7455（+0.0021 %）/ +8.7273（+0.0105 %）**；旧数字隐含**三个互不一致**的窗口 | `docs/core-algorithms/13-volume-liquidity-costs.md:145-162` |
| F13 | "BTCUSDT/1h 的 55 个 twin 已合并" | **当时是假的**：`check_data_integrity` 报 1/29 带 twin，实测 11 678 行 / 11 623 bar / 55 重复 / 110 twin 行。根因是 `flush_all` 只重写 dirty 键 | `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:95`, `:167` |
| F14 | "`_series_has_gap` 的缺口闸门在实盘生效"（F1 缺陷） | 缓冲裸 float + RangeIndex 帧让 `1-0` 被当成 1 秒 → 永远 `False`；实测 300 根含 100-bar 空洞 → `gap_guard=False`、预测 **0.0698 %/bar** | `:126` |
| F15 | "`t_stat is None` 时显著性检查会拒绝"（F3 缺陷） | 修复前**整段跳过**：`credibility_gate({auc .60, n 5000, n_trades 300}, 0.004, t_stat=None, psr=None)` → `allowed=True, "pass"` | `:127` |
| F16 | "`skip_ml_training` 加载的模型是可信的"（F4 缺陷） | 修复前临时目录 4 个 pickle **4/4 全部加载**；本仓库 15 个 pickle 修复后 **15 拒绝**（本仓库有 15 `.pkl` / **0** `_meta.json`，我本次实测一致） | `:128` |
| F17 | "对未裁剪的**价格水平**算波动率"（`vol_report` 第一版） | 喂 close 给 `ewma_vol` 报的是价格的波动率，不是收益的；现已先取对数收益 | `scripts/check_data_integrity.py:182-185` |
| F18 | "HMM 因果解码的准确率是 ≥95 %"（归档表述） | 当前代码记录：in-sample Viterbi **0.9987**，**causal 0.758–0.815**。见 §11 D-7 | `core/strategy/regime.py:59`, `:62` |
| F19 | "`docs/core-algorithms/11` 里 HMM 因果解码 ≈18–21 s" | 代码明确否证："not in this file and does not reproduce"（真值 ≈2.3–2.6 s） | `core/strategy/regime.py:508-510` |
| F20 | "R 可以取 $\mathrm{Var}(y)$"（Kalman） | 用 $\mathrm{Var}(y)$ 时 $\beta$ 塌缩到 **0.058**（OLS **0.628**），"价差"退化成 $y$ 的原始价格，配对交易变成单边方向赌注 | `docs/core-algorithms/11-pairs-cointegration.md:60-65` |

### 10.3 默认配置的"逐位不变"契约（已实测的）

* 波动率目标化关闭时两侧（`RiskManager`/`PositionGuard`）**逐位相同**；关闭时移动距离与 pre-P3 逐位相同（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:165`, `:173`）。
* `risk.liquidity.enabled=false` / `impact_k=0` 时定仓与成本**逐位**等于 pre-P6（`:166` 的 R2 行、`:295-304` 的 doc 13）。
* ML / P4 全部开关关闭时逐位不变（`:173`）。
* `risk.vol_targeting.enabled=false` 时 `forecast_vol_pct` 的消费路径整体短路——因此 D2 记录："**运行中的进程不是修复的探针**"，因为开关没开，D1 的折叠闸门在实盘进程里**根本不会被走到**（`:182`, `:197`）。

---

## 11. 文档 ↔ 代码矛盾清单（Discrepancies）

> 每条给出**两侧的原文与 `file:line`**。我没有修改任何既有文档（写权限限制）。

> **状态跟踪（2026-09-30 更新，本表是这份清单的活口）**。标记含义：
> **fixed here** = 本轮由文档清扫代理在其写范围内修好；**fixed by the code agent** = Lead
> 分派给代码侧兄弟代理（D-4/D-16/D-19/死 GA 键/DSR 试验计数）；**fixed earlier** =
> 前一轮已修，本轮复核仍成立；**report-only (lead)** = 不在本轮任何写范围（`config/`、
> `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md`、doc 06/07），只报告不修改；**open** = 仍然存在、
> 本轮无人认领（含本轮清单未列的 doc 侧条目）。
>
> | # | 主题 | 状态 | 本轮记录 |
> |---|---|---|---|
> | D-1 | `ewma` 成本三值 | **fixed here** | `docs/core-algorithms/10-volatility-targeting.md` §3.2 按**形状**标注：实时 ≤600 根 `forecast_vol` 0.20 ms / 数组 0.16 ms；全历史 1.72 ms / 1.64 ms；`volatility.py` 的 ≈0.14 ms = 500 根数组；表的 ≈1.5 ms/bar = 全历史形状（本文 §3.9 的 0.2248/0.3093/0.2806 是同一批形状在负载下的读数） |
> | D-2 | 裁剪倍数四值 | **fixed here** | doc 10 §2 新增"D-2 口径对照"表：10.12×/10.1×/9.81× = 历史真实接缝的三种读数；12.3× = 合成注入；当前缓存 1.000000×（全历史）/1.0365×（尾窗 500） |
> | D-3 | legacy ML 准确率 | open | `ALGO_UPGRADE_EVIDENCE.md:66,:69` vs `docs/core-algorithms/08-ml-triple-barrier.md:154-155`，两侧都不在本轮写范围 |
> | D-4 | 特征契约 hash | **fixed by the code agent** | `core/ml/features.py` + `tests/test_feature_schema_v1.py` |
> | D-5 | 实盘波动率定仓接线 | open | doc 10:357-364 已过期（`core/risk/manager.py:577` 现调用 `resolve_forecast_vol_pct`）；本轮清单未列 |
> | D-6 | doc 12 测试数 20 | **fixed here** | doc 12:61 → **24**（`--collect-only` 实测） |
> | D-7 | 归档 "≥95 %" | report-only (lead) | `ALGO_UPGRADE_EVIDENCE.md:14,:81`；doc 11 本轮已把同一张表标注 in-sample（见 D-11） |
> | D-8 | pairs 文档 HMM 成本 cite 错误 | open | `core/strategy/regime.py:509` 引用了一份不含该数字的文档 |
> | D-9 | P6 计划冲击数字 | **fixed earlier** | P6 计划:61 已改为"已撤回、不要再用"（本轮复核） |
> | D-10 | 流动性测试数 24/25/26 | **fixed here** | doc 13:291 → **26**；P6 计划:58 → **26**（`--collect-only` 实测 26） |
> | D-11 | "三 regime"/≥95 % | **fixed here** | doc 11 表头 + ⚠️：0.9987/0.9993/0.9987 是 **in-sample**，可交易口径 **0.758/0.759/0.815**；归档两处表述待 lead |
> | D-12 | doc 12 门写 OR | **fixed here** | doc 12:44 → **AND**（`core/ml/credibility.py:865`） |
> | D-13 | config GARCH 成本注释 | report-only (lead) | `config/config.yaml:164`（HEAD 为 **:159**）"~12 ms/bar" vs 实测 ≈0.13 s/call / doc 10 ≈0.18–0.19 s |
> | D-14 | alpha 漏 √365 | **fixed by the code agent** | doc 06 + `core/ga/fitness.py` |
> | D-15 | T<20 时 alpha 记 0 | open | `docs/core-algorithms/07-deflated-sharpe-ratio.md:62-63` |
> | D-16 | fitness 减买入持有 | **fixed by the code agent** | doc 06 + `core/ga/fitness.py` |
> | D-17 | doc 07 门槛常数不自洽 | open | doc 07:36-37 与同页 :41 不自洽；docstring 在 `tests/test_ga_credibility.py` |
> | D-18 | DSR 的两个 $N$ | **fixed by the code agent** | DSR 试验计数 |
> | D-19 | `timeout_label` 自相矛盾 | **fixed by the code agent** | `core/ml/labels.py` |
> | D-20 | doc 06 hybrid"残留"过期 | report-only (lead) | doc 06:50,:68（该文件在兄弟代理写范围，但不在其清单） |
> | D-21 | doc 06 交叉引用失效 | report-only (lead) | doc 06:5,:210 引用不存在的"效用分析"节 |
> | D-22 | `FEATURE_KEYS` 19 vs 20 | **fixed here** | P6 计划:17,:40 → **20**（`core/market_data/microstructure.py:125-131`） |
> | D-23 | 配对成本两路径 | open | doc 11:95-99 只写 0.25 %/腿；`config=None` 的 0.14 %/腿 未写 |
> | D-24 | 测试计数汇总 | **fixed here + fixed earlier** | doc 12/13、P6 计划本轮修；doc 11:111 的 **28**（25 def + 3 async）在前轮已修并本轮复核；`tests/test_condition_logic.py` 的 9 vs 10 仍 open |
> | D-25 | 因果解码数字一处四值 | **fixed here（HMM 部分）** | doc 11 已标注 in-sample/causal（0.758–0.815）；Kalman 半衰期 `doc 11:145` 1.2 vs 代码 2.1、匹配零分布 `doc 11:33-35` −3.3015/−3.3098 vs 代码 −3.363/−3.370、`tests/test_pairs.py:203,:213` 的 `var(y)/R ≈ 100`（应为 4.3）仍 open |
> | D-26 | 死代码与导出面 | **fixed by the code agent（部分）** | 死 GA 配置键由兄弟代理处理；`_hmm_forward_last`、`HMM_MIN_SIGMA_RATIO`、`DEPTH_DECAY_BP`、`_GARCH_MLE_X0` 仍未处理 |
> | D-27 | garch 网格复杂度注释 | open | `core/ml/volatility.py:713` 写 O(20×8)，实际 4×8=32 |
> | D-28 | OHLC 估计量根本不裁剪 | open | doc 10:113 的"所有估计量"对 parkinson/garman（`volatility.py:472,:494`）与 GARCH 拟合（σ=8，`:815`）不成立 |
> | D-29 | `stop_distance_pct` 硬下限被省略 | open | doc 10:51；`core/risk/position_sizer.py:144` 是 `max(stop_min_pct, hard.min_stop_loss_distance_pct)` |
> | D-30 | 裁剪锚点函数名/窗口计数 | open | doc 10:119-122 应写 `series_anchor`；"8 343/8 344" vs 代码 ":226-231" 的 11 176/11 176 |
> | D-31 | `window` 语义 | open | doc 10:141-145 与 `_as_returns`（`core/ml/volatility.py:1214-1216`）显式忽略 window 冲突 |

### D-1 `ewma` 的单次成本：文档 vs 代码注释 vs 我的实测（三值）

* `docs/core-algorithms/10-volatility-targeting.md:157-159`：`| **ewma（默认）** | 0.5242 | 49.06 | … | **≈1.5** |`（ms/次），并在 `:171-177` 写 "`forecast_vol(df)` = **≈1.5 ms/bar** … 在 2 ms 预算内，但余量只有 ~25%"。
* `core/ml/volatility.py:176-181`：`Re-measured on this checkout: **≈0.14 ms** for ``ewma`` on a 500-bar window (≈0.15–0.16 ms on the 600-row live frame) … the older "0.14–0.27 ms / ≈0.1 ms" pair is 2–10× high.`
* **我本次实测**（命令见 §3.9）：`ewma` 500-bar 数组 **0.2248 ms**、600-bar 数组 **0.3093 ms**、`forecast_vol` 600-bar **0.2806 ms**。

三者相差约 **5–7×**（doc 10 的 1.5 ms vs 代码注释的 0.14 ms）。代码注释里"older pair is 2–10× high"的自我更正与 doc 10 并未同步；我的读数落在两者之间且更接近代码注释。**未验证**：doc 10 的 1.5 ms 是在哪个 revision/何种窗口上量的。

### D-2 拼接/裁剪的波动率倍数：四份文档四个数字对

* `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:73`：**5.31 %/bar vs 0.52 %/bar（10.12×）**。
* `docs/core-algorithms/10-volatility-targeting.md:80-81`：**5.3056 vs 0.5242（9.81×）**，并指出早期文档的 10.5× 是"拿剪裁后的 0.52 去除未剪裁的 5.47，属单位/口径混用"。
* `core/risk/manager.py:33-45`：`an un-clipped RiskMetrics recursion reported 5.31 %/bar instead of 0.52 %/bar, a **10.1x** overstatement`。
* `core/ml/volatility.py:137-141`：注入实验 **4.98 %/bar vs 0.40 %/bar（12.3×）**。

四者都在描述同一现象，但除数/被除数、窗口长度、样本 revision 与"注入 vs 真实拼接"各不相同。可复现的只有"未裁剪会高估约一个数量级"这一**定性**结论。

### D-3 legacy ML 准确率：两份文档四个数字

* `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:66`, `:69`：ETH legacy 准确率 **0.6597**（多数类 0.7773）、BTC legacy **0.6834**（多数类 0.8140）。
* `docs/core-algorithms/08-ml-triple-barrier.md:154-155`：BTC 旧 ACC **0.6702**（多数类 **0.8128**）、ETH 旧 ACC **0.6408**（多数类 **0.7760**）。

两处都自称"实测"且都在同一仓库；AUC 与净期望两侧一致（0.5621/0.5672、0.5342/0.5207、−0.1822 %/−0.2494 %），只有准确率与多数类不同——说明是**两次不同的运行/切片**（`docs/core-algorithms/08` 说 8 845 根，证据索引的 OOS 样本是 ETH 4 840 / BTC 1365 笔）。**结论不受影响**（都是"远低于多数类"）。

### D-4 特征契约：提交的列表 40 项 vs 声明的 hash 对应 39 项

* HEAD `core/ml/features.py:38`（注释）：`# ── Canonical feature list (39 features) ──`；`:59-61`（注释）：`# NOTE: bare "hurst" was dropped — it is perfectly collinear with roll_hurst_20 …`。
* HEAD `core/ml/features.py` 的 `DEFAULT_FEATURES` 字面量里**第 31 项仍是 `"hurst"`**（我按 AST/regex 从 `git show HEAD:core/ml/features.py` 提取，共 **40** 个字符串）。
* 证据索引与 P6 计划引用 `feature_schema_hash=335e63360104`、契约 **39 列**（`docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:62`；`docs/overhaul/P6_VOLUME_PLAN.md:13`, `:38`）。
* **我的独立复算**：$\text{sha1}(\text{json}(\text{39 项，去掉 `hurst`}))[:12]=$ `335e63360104` ✅；$\text{sha1}(\text{json}(\text{40 项，含 `hurst`}))[:12]=$ `70899cff156d` ❌（不等于声明值）。

**判定**：文档（39 列 / `335e63360104`）与**归档的已部署模型**一致；HEAD 代码里的字面量多了 `"hurst"`，因此代码当前算出的 hash 是 `70899cff156d`。即"注释说删了、列表没删"——一个真实的代码/文档不一致。（工作树里兄弟代理的 v2 编辑正在把该列表扩到 52–54 列，写作时无法导入，见 §2.9。）

### D-5 实盘波动率定仓是否已接线：P3 文档 vs 证据索引

* `docs/core-algorithms/10-volatility-targeting.md:357-364`：`manager.py:check_signal` 调 `calculate_position_size(...)` 而该文件不在 P3 写范围内，因此**它没有传 `forecast_vol_pct`** —— 实盘仓位仍走固定比例。
* `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:86`（gap fix ①）：`**①实盘波动率定仓接线**（`RiskManager.check_signal`→`PositionSizer`；开关关闭时 balance 10 000 两侧均 240.0 USDT，行为不变）`。

**判定**：doc 10 是 P3 当时的状态快照，证据索引记录了后续接线。两者不是同一 revision 的陈述，但 P3 文档**没有**加"已过期"标注（对比：doc 10 的其他小节有 `> ⚠️` 标注过期）。读者会误以为实盘仓位仍未接线。

### D-6 `tests/test_meta_labeling.py` 的用例数：20 vs 24

* `docs/core-algorithms/12-meta-labeling.md:61`：`| 测试 | `tests/test_meta_labeling.py`（20 条） |`。
* `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:14`, `:79`：**24**（并注明"正被并行代理改写"）。
* **我实测**：`Select-String -Pattern '^\s*(async )?def test'` → **24**。

判定：`docs/core-algorithms/12` 过期（应为 24）。同理 `docs/core-algorithms/13-volume-liquidity-costs.md:291` 写 "24 passed"、`docs/overhaul/P6_VOLUME_PLAN.md:58` 写 "24 项"，而我实测 `tests/test_liquidity.py` 是 **26** 项（证据索引 `:18` 记 25、`:209` 记本轮 26）——见 D-10。

### D-7 Regime 准确率：归档 "≥95 %" vs 代码 "in-sample 0.9987 / causal 0.758–0.815"

* `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:14`：`HMM 三 regime 合成序列 ≥95% 准确率`；`:81`：`已知三 regime 合成序列：HMM 恢复已知波动率、≥95% 准确率、延迟数根 bar`。
* `core/strategy/regime.py:59`：whole-sample Viterbi 解码 **0.9987**；`:62`, `:270`：causal 路径 **0.758–0.815**。

判定：归档的 "≥95 %" 与代码记录的 **causal 0.758–0.815** 不一致；0.9987 是 **in-sample**，代码明确要求"must be labelled as such"。归档表述没有区分两者，容易被读成"因果解码也有 95%+"。

### D-8 pairs 文档里的 HMM 成本数字（cite 错误）

* `docs/core-algorithms/11-pairs-cointegration.md:508-510` 被 `core/strategy/regime.py:508-510` 引用为 "the '≈18–21 s' quoted in `docs/core-algorithms/11-pairs-cointegration.md` is not in this file and does not reproduce"。
* **我核对**：`docs/core-algorithms/11-pairs-cointegration.md` 是配对/协整文档，其内容里**没有** HMM 成本那一节；代码的引文指向的是**错误的文档**（该数字应在 doc 10 或 regime 相关文档）。同时代码记录的实测值是 **≈2.3–2.6 s**。

判定：`core/strategy/regime.py:509` 的文档引用不成立（引用了一个不含该数字的文件）。

### D-9 P6 计划仍保留已被 doc 13 更正的冲击数字

* `docs/core-algorithms/13-volume-liquidity-costs.md:145-162`（**已更正**）：旧的 `+178.8965`（+2.36 %）/ `+894.4824`（+11.82 %）**不可复现**；同窗口给出 `+1.7455`（+0.0021 %）/ `+8.7273`（+0.0105 %）。
* `docs/overhaul/P6_VOLUME_PLAN.md:61`（**仍是旧数字**）：`- 1.0 BTC 往返：`k=0.1` **+2.36 %**、`k=0.5` **+11.82 %**；`。

判定：P6 计划的 P6-A 验收条目引用了已被撤回的数字。它自称是"`3e90013`（2026-09-30）"的状态快照，但同一仓库的 doc 13 在 `b49883b` 之后已更正；读者若只看计划会得到错误量级。（P6 计划的 XRP 行 `265.8 万 USDT → 1.881 % → 26 575.74` 与 doc 13 的表**一致**。）

### D-10 `tests/test_liquidity.py` 项数：24 / 25 / 26

* `docs/core-algorithms/13-volume-liquidity-costs.md:291`：`python -m pytest tests/test_liquidity.py -q          # 24 passed`。
* `docs/overhaul/P6_VOLUME_PLAN.md:58`：`… `risk.liquidity` 配置块（默认关）、24 项测试、文档 13。`
* `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:18`：`tests/test_liquidity.py` 当轮 **25** 项（本轮 26 项）；`:209`：`tests/test_liquidity.py`（R4，+1 项合成分支）。
* **我实测**：`def test` 计数 = **26**。

判定：doc 13 与 P6 计划过期（应为 26）。

### D-11 两处 regime 文档说"≥95 %"，与 `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:14` 的"三 regime"一致，但代码是**两状态** HMM

* `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md:14`, `:81`：`HMM 三 regime 合成序列`。
* `core/strategy/regime.py:240`：`def hmm_two_state(...)`；`REGIME_GATING_ENABLED` 等常量集中在**两状态** HMM 一节（`:191`）。

判定：合成测试序列可能被构造成三段（三个变点区间），但**模型**是两状态。归档表述"三 regime 合成序列"没有说明这一点，容易被读成"三状态 HMM"。代码里 `mu`/`sigma` 都是长度 2 的数组（`:291-294`, `:316`）。

### D-12 `docs/core-algorithms/12` 的门写 "t > 2 或 PSR ≥ 0.95"（OR），代码是 AND

* `docs/core-algorithms/12-meta-labeling.md:42-45`：`门控（…gate_from_evaluation`）消费的是**外层**数字 …：AUC > 0.55、净期望 > 0、交易数 ≥ 100、**t > 2 或 PSR ≥ 0.95**。`
* `core/ml/credibility.py:20-26`（模块 docstring）：`allowed = OOS AUC > 0.55 AND net expectancy > 0 AND trades >= min_trades AND t > 2 AND PSR >= 0.95`；`core/ml/credibility.py:866`：`elif not (t_val > float(min_t_stat) and psr_val >= float(min_psr))`。
* `core/ml/meta.py:46-50` 也写 AND。

判定：`docs/core-algorithms/12-meta-labeling.md:44` 的 **OR** 是**过期且方向错误**的（审计 F3 已把语义定为 AND；证据索引 `:132` 记录了这个决定）。doc 08 已经改对了（`docs/core-algorithms/08-ml-triple-barrier.md:106-114` 写 **AND**），doc 12 漏改。

### D-13 `config.yaml` 的 GARCH 成本注释

* `config/config.yaml:158-160`：`# Cheap methods are ~1 ms/bar; 'garch11' measures ~12 ms/bar and is for research/reporting, not the hot path.`
* `core/ml/volatility.py:182-186`：`garch11` **excluded on purpose**: a full MLE call measures **≈0.13 s** … (**not the ≈30 ms an earlier revision of this comment claimed** — that was the optimiser-free grid fallback's cost, ~100× cheaper than the shipped fit)。
* `docs/core-algorithms/10-volatility-targeting.md:246-247`：自由 ω 版本代价 **≈0.18–0.19 s/次**（默认 `window=500`），整段历史 **≈4.1 s/次**。

判定：`12 ms/bar` 与代码的 `≈0.13 s/call`（≈130 ms）相差一个数量级，且代码注释明确说 ≈30 ms 是**优化器无关的网格回退**的成本、比 shipped fit 便宜约 100×。config 注释里的 12 ms **不匹配任何一条已记录的口径**。

### D-14 `alpha` 项漏掉 $\sqrt{365}$ 年化（doc 06 vs 代码）

* `docs/core-algorithms/06-ga-evolution.md:115`：`alpha = DSR_deflated_sharpe × min(1, trades / 30) - max_drawdown_pct`。
* `core/ga/fitness.py:491-493`：`dsr_sharpe = _dsr["dsr"] * (365.0 ** 0.5)` / `evidence = min(1.0, trades / float(SHARPE_TRADE_FLOOR))` / `alpha = dsr_sharpe * evidence - stats_out["max_dd"]`。

判定：doc 06 的公式比代码**小 $\sqrt{365}\approx19.1$ 倍**。doc 07:33 自己定义 `DSR = SR_per_period - E[max]`（每期口径），所以代码的年化是对的；用归档冠军（DSR 0.2119 / 75 笔 / max_dd 0.12 %）验算：代码 3.928、文档公式 0.092，而**实测**的第 3 代 fitness 增量恰好等于复杂度罚项差 4.00（只有代码口径自洽）。

### D-15 `T < 20` 时"alpha 项记 0"是错的（doc 07 vs 代码）

* `docs/core-algorithms/07-deflated-sharpe-ratio.md:62-63`：`T < 20`（`MIN_OBSERVATIONS`）时不估计 Sharpe/DSR：DSR 记 0、**alpha 项记 0**。
* `core/ga/fitness.py:486-493`：`_dsr = {..., "dsr": 0.0, ...}` → `dsr_sharpe = 0.0 * √365 = 0.0` → `alpha = 0.0 * evidence - stats_out["max_dd"]` = **−max_dd**，并在 `:496` 加进 fitness。

判定：alpha 项变成**纯回撤罚项**，不是 0。

### D-16 "fitness 减去买入持有"在三处声明、零处实现

* `docs/core-algorithms/06-ga-evolution.md:131-134`：`… 并**减去同窗口同币种的等权买入持有收益**（metrics["buy_hold_pct"]），beta 不再被计为 alpha。`
* `core/ga/fitness.py:22-23`（模块 docstring）：`and the alpha term subtracting the equal-weighted buy & hold return of the same symbols and window, so beta is not scored as alpha.`
* `config/config.yaml:54-56`：`… the run's return is scored against the equal-weighted buy & hold of the same window/symbols.`
* **代码**：fitness 求和只有 `core/ga/fitness.py:451-456` + `:496-499`（base + alpha + 复杂度）；`alpha_vs_buy_hold_pct` 在 `:505-510` 只被**赋值**。
* **实测反证（A 级）**：同一份 stats，`buy_hold_pct=None` 与 `25.0` 两次调用 `score_stats` 都返回 `fitness=29.1535`；只有 `alpha_vs_buy_hold_pct` 从 0.0 变为 −14.73907131946944。`tests/test_ga_credibility.py:340`（`test_buy_and_hold_is_subtracted_from_the_selection_metric`）**只断言上报字段**（`:350`, `:355`, `:358`），从不断言 `fitness`——测试名描述的机制与它验证的东西不同。

判定：买入持有只进**发布门**（`core/ga/evolver.py:461-463`）与 provenance（`:384-385`），不进 fitness。文档三处均错（`ALGO_UPGRADE_PLAN.md` 的 P1.3 是这些说法的源头）。

### D-17 doc 07 的门槛常数与自己引用的旧输出不自洽

* `docs/core-algorithms/07-deflated-sharpe-ratio.md:36-37`：`这样门槛是一个与数据长度无关的常数（N=1200 时 ≈0.191）`。
* 同一页 `:41`：旧实现输出 **+1.0029**。旧门槛 $=\sqrt{1/365}\sqrt{2\ln1200}=0.197103$，$1.2-0.197103=1.002897$ ✓。
* 同样过期的算术在 `tests/test_ga_credibility.py:386-391` 的 docstring（`E[max] = 0.19148 / DSR = -0.12866`），而测试**自己的断言**（`:399-402`）用 `T=365` 求值为 `0.197103 / -0.134292`（子代理在 HEAD 上实算）。

判定：`≈0.191` 与 `0.19148` 都是过期中间值；断言正确、文档错误。

### D-18 DSR 的 $N$：doc 06 只写了两个 $N$ 中的一个

* `docs/core-algorithms/06-ga-evolution.md:142-144`：`N = population × generations + 历史试验数`。
* 代码：代内 fitness 用 `population + prior`（`core/ga/evolver.py:202` + `core/ga/fitness.py:476-477`）；冠军/验证用 `population × generations + prior`（`core/ga/evolver.py:319-320`）。返回给调用方的 `dsr` 取的是**代内**那个（`:413-415`），provenance 的 `n_trials` 取的是**大**那个（`:371-372`）。

判定：归档的冠军 `dsr=0.2119` 与 provenance 声称的 $N$ 不是同一个 $N$。

### D-19 `timeout_label` 在 `core/ml/labels.py` 内部自相矛盾

* `core/ml/labels.py:236-240`（docstring）：`with ``timeout_label`` set, ``NA`` *inside* the sample is filled with the timeout class — the tail is not …`
* `core/ml/labels.py:289-291`（同一函数的注释）：`in **both** modes: with ``timeout_label`` they are *not* filled (only genuine timeouts are)`
* 实现 `:261-292`：没有任何 `fillna(timeout_label)`。
* 对照 `core/ml/features.py:954-959`（遗留定宽路径）：**确实** `result = result.fillna(timeout_label)`。
* `docs/core-algorithms/08-ml-triple-barrier.md:186` 同时含两种口径。
* 消费侧 `core/ml/predictor.py:462` 传 `timeout_label=2.0` 并持久化 `class_distribution`（`:514-518`）⇒ 落盘 `timeout_share = 0.0`。

判定：docstring 与实现冲突（注释与 docstring 也互相冲突）；vol-scaled 路径**不产生类 2**。这是一个**有运行后果**的缺陷（持久化的标签分布是错的），不是纯文档问题；我只在本文档里报告，未修改 `core/ml/labels.py`（不在写权限内）。

### D-20 doc 06 的 hybrid 引擎"残留"已过期

* `docs/core-algorithms/06-ga-evolution.md:50`：`**残留（不在本阶段写权限内）**：向量化的混合引擎 core/backtest/signal_matrix.py:285 仍只做 OR，未读 condition_logic …`；`:68` 重复同一说法。
* **代码**：`core/backtest/signal_matrix.py:15-16` 导入 `CONDITION_LOGIC_AND`，`:303-304` `use_and = (str(getattr(s, "condition_logic", CONDITION_LOGIC_OR)).lower() == CONDITION_LOGIC_AND)`；`tests/test_hybrid_condition_logic.py` 存在（3 个函数 / 4 项收集，与证据索引 `:113` 的"收集 4 项"一致）。

判定：doc 06 的残留行未随 `beda096` 更新（证据索引 `:113` 已声明该项"已关闭"）。

### D-21 doc 06 引用了一个它自己没有的小节

* `docs/core-algorithms/06-ga-evolution.md:5` 与 `:210` 都写 `下面/文末「效用分析」的表格是**示意值（非实测）**`。
* 该文件（228 行）**没有**"效用分析"小节（grep 只找到这两处引用）；真正的示意表在 `docs/core-algorithms/07-deflated-sharpe-ratio.md:110-117`。

判定：doc 06 的交叉引用失效。

### D-22 `FEATURE_KEYS` 的项数：19 vs 20

* `docs/overhaul/P6_VOLUME_PLAN.md:18`：`已有 19 个盘口/逐笔特征（microprice、ofi_depth、ofi_trades、trade_large_share、rv_trade、arrival_rate_hz、activity_ratio …）`；`:40`：`19 项 FEATURE_KEYS，来自实时 book/trades，默认关`。
* **代码**：`core/market_data/microstructure.py:125-131` 的 `FEATURE_KEYS` 我逐项数过是 **20** 项：`mid`, `microprice`, `microprice_dev_bps`, `spread_bps`, `ofi_depth`, `ofi_depth_weighted`, `ofi_trades`, `book_slope_ratio`, `trade_count`, `trade_mean_qty`, `trade_median_qty`, `trade_large_share`, `trade_notional`, `rv_trade`, `arrival_rate_hz`, `activity_ratio`, `as_of_ms`, `book_age_ms`, `dropped_future_trades`, `n_depth_levels`。
* `docs/core-algorithms/11-pairs-cointegration.md` 与 `docs/core-algorithms/13` **没有**声称 19 项，因此冲突在 P6 计划一侧。

判定：P6 计划的"19 项"落后代码一列（`book_slope_ratio` 或某个元数据键）。另外 `DEPTH_DECAY_BP` 与 `MIN_SLOPE_DIST_BP` 都不在模块 `__all__` 里（`:602-611`），而 `HMM_MIN_SIGMA_RATIO` 也不在 `regime.py` 的 `__all__` 里（`:884-892`）——**导出面**与文档覆盖面不一致。

### D-23 配对成本的两个真实数字（不是矛盾，但文档只写了一个）

* `docs/core-algorithms/11-pairs-cointegration.md:95-99`：一次配对往返 = 四条腿的成交，`leg_round_trip_cost_pct` 取自 `cost_pct_for`（sim 成本模型），实测 **0.25 %/腿**，即一次配对往返约 **0.50 %**。
* **代码默认路径**：`core/strategy/pairs.py:1106-1107` docstring 写"Without a config object the documented defaults (0.04 % taker / 0.01 % half-spread / 2 bp slippage → **0.14 %**) are used"，且 `tests/test_pairs.py:373` 钉住 `default_leg_cost_pct(None) == 0.14 ± 0.01`。

判定：**两个都是真的，走不同代码路径**——`config=None` 的 research 默认是 0.14 %/腿（`core/ml/credibility.py:73-75` 的硬编码回退），shipped sim 配置是 0.25 %/腿（`config/config.yaml:246` VIP0 taker 0.10 % + `:251` slippage 2 bp + `:259` BTCUSDT spread 0.01 ⇒ 每边 0.125 ⇒ 往返 0.25）。文档只讲了后者；写公式时**必须点明用哪个**，否则回测成本差 1.8 倍。

### D-25 Regime / pairs 的因果解码数字：一处四值、一处三值

* **修复前索引错位（`fwd[k]`）的解码准确率**：`core/strategy/regime.py:496-501` 写 **0.52–0.56**（并自我否证 "0.156 does not reproduce on any of those three seeds"）；同文件 `:577` 写 **0.16–0.52**；`docs/core-algorithms/11-pairs-cointegration.md:279` 与 `tests/test_p34_audit_fixes.py:774` 写 **0.156**。→ **四个值**，其中 `regime.py` **与自身矛盾**。
* **Kalman 半衰期（BTC/ETH 1h）**：`core/strategy/pairs.py:800-803` 写 **2.1 bar**；`docs/core-algorithms/11-pairs-cointegration.md:145` 的表格写 **1.2**。→ 同一测量两个值，两者都未被测试断言。
* **`var(y)/R` 比值（BTC/ETH 1h）**：`core/strategy/pairs.py:45-48` 与 `:593-596` 写 **4.3×**（`var(y)=0.0400` vs OLS 残差方差 `0.0094`）；`docs/core-algorithms/11-pairs-cointegration.md:60-61` 与代码一致；但 `tests/test_pairs.py:203-204` 与 `:213` **两次**写 "`var(y)/R ≈ 100`"。→ $0.0400/0.0094=4.26$，**测试里的 ≈100 是错的**（doc + 源码 vs 测试）。
* **因果 HMM 成本**：`core/strategy/regime.py:505-507` 写 **≈2.3–2.6 s / 3 000 bar**；`docs/core-algorithms/11-pairs-cointegration.md:286-287` 写 **≈2.30–2.52 s**（种子 5/7/11 = 2.52/2.28/2.37）。子代理复核了重拟合次数（$t=50$ 初始 + 250,500,…,2750 ⇒ **12 次拟合 + 12 次扫掠**），与代码一致。代码还明确否证了一个"≈18–21 s"的旧值（`:508-510`），并指出该数字被归给了 `docs/core-algorithms/11-pairs-cointegration.md`——而那份文档现在**只在该否证句里**提到它，因此代码对文档的转述已过期（`:288-289`）。
* **未增广 vs 匹配零分布的 5 % 分位**：代码写 **−3.363（近似）/ −3.370（匹配）**，差 **0.007**（`core/strategy/pairs.py:276-277`, `:481-484`）；doc 11:33-35 写 **−3.3015 / −3.3098**，差 **0.0083**。→ 同一实验两套数，方向一致（匹配更靠左），**两者都没有被测试断言**。
* **配对检验的规模实验点估计**：doc 11:127/:131 的头条是 **6/200 = 0.030**；`tests/test_p34_audit_fixes.py:743` 的 docstring 写 **0.035**。doc 11:132-134 诚实声明测试只断言上界 `≤0.08`/`≤0.03`。
* **`≈3 s/次`**（lag-augmented 零分布成本，doc 11:38-40）：代码里确认了 "~10×"，但**绝对成本数字在仓库里不存在** → **未验证**。

### D-26 死代码与导出面

| 名称 | 位置 | 状况 |
|---|---|---|
| `_hmm_forward_last` | `core/strategy/regime.py:407-425` | **定义了但全仓库无调用者**（子代理 grep 确认） |
| `HMM_MIN_SIGMA_RATIO` | `core/strategy/regime.py:128` | 不在 `__all__`（`:884-892`），尽管 `HMM_CAUSAL_REFIT_EVERY` / `HMM_CAUSAL_WARMUP` 在 |
| `DEPTH_DECAY_BP` / `MIN_SLOPE_DIST_BP` | `core/market_data/microstructure.py:120`, `:123` | 不在 `__all__`（`:602-611`） |
| `_GARCH_MLE_X0` | `core/ml/volatility.py:690` | 无读取者（数值以字面量留在 `_garch11_grid_scan:717`） |
| `PAIRS_ENABLED` | `core/strategy/pairs.py:112` | 无任何代码读取（只有定义、docstring `:90`/`:1057`、`__all__` `:1113`）——`docs/core-algorithms/11-pairs-cointegration.md:195` 的"当前无任何代码读它"**正确** |
| `MICROSTRUCTURE_ENABLED` | `core/market_data/microstructure.py:108` | 同上，无读取者 |
| `REGIME_GATING_ENABLED` | `core/strategy/regime.py:101` | **有**读取者（`:280`, `:807`, `:878`） |

### D-27 `_garch11_grid_scan` 的复杂度注释与代码不符

* `core/ml/volatility.py:713`（docstring）：`Deterministic and O(20 × 8) likelihood passes`。
* 代码 `:698-699` 的 `_GARCH_GRID_ALPHA = (0.01, 0.20, 0.05)` / `_GARCH_GRID_BETA = (0.60, 0.99, 0.05)` 经 `np.arange` 展开是 **4 个 $\alpha$ × 8 个 $\beta$ = 32 个候选**（再经持续性过滤）。→ 注释高估了 5×。

### D-28 OHLC 估计量根本不裁剪，与 doc 10 的"所有估计量"冲突

* `docs/core-algorithms/10-volatility-targeting.md:113`：`所有估计量先用 clip_outliers（±6 × 1.4826 × MAD，可关）`。
* **代码**：`parkinson_vol`（`core/ml/volatility.py:472`）用 `high[-int(window):]/low[-int(window):]` 的**原始尾部切片**，`garman_klass_vol`（`:494`）用 `sl = slice(-int(window), None)`——**两个函数都没有调用 `clip_outliers`**。另外 GARCH **拟合**用的是 **$\sigma=8$**（`_garch11_scipy_mle` 的 `outlier_sigma: float = 8.0`，`:815`，理由 `:854-855`），而 `_garch11_variance` 过滤时回到 6（`:1073`, `:1115`）；doc 10 只写了 6。

判定：doc 10 的"所有估计量"对 5 个方法中的 3 个不成立（2 个 OHLC + GARCH 拟合阈值）。

### D-29 `stop_distance_pct` 的硬下限被文档省略

* `docs/core-algorithms/10-volatility-targeting.md:51`：`stop_pct = clip(stop_vol_multiple × forecast_vol_pct, stop_min_pct, stop_max_pct)`。
* **代码** `core/risk/position_sizer.py:144`：`lo = max(_f(getattr(vt, "stop_min_pct", 0.0)), _f(self.hard.min_stop_loss_distance_pct))`——下限是 `max(stop_min_pct, hard.min_stop_loss_distance_pct)`，即**硬下限也参与**。

判定：文档写漏了硬风控下限，会让人以为 `stop_min_pct` 可以单独把止损放得比硬限制更近。

### D-30 裁剪锚点的函数名：doc 10 指错了函数

* `docs/core-algorithms/10-volatility-targeting.md:119-122`：`现在用 build_anchor(returns) 对整个序列只算一次中心与尺度（同一 median / 1.4826·MAD 配方，因此在全序列上输出与旧实现逐位相同），然后把 AnchorMAD 传给 clip_outliers / ewma_variance / ewma_vol`。
* **代码**：估计量真正构建的是 **`series_anchor`**（= `_anchored_mad` 在 half-life 2000 上，`:369-397`, `:421-423`）；`build_anchor` 的 `half_life=0` 默认是**纯** `median`/`1.4826·MAD` 配方（`:280`, `:318-319`），**不是**与旧默认逐位相同的那一个（那个同一性是对 `series_anchor` 声明的，`:373-377`）。

判定：函数名与配方都指错了。另：窗口不稳定性计数也冲突——doc 10:117-118 写 "8 344 个连续窗口里 **8 343** 个变（最大 |Δ| ≈ **4.8e-2**）"，代码 `:226-231` 写 "**11 176 of 11 176**（纯 per-window 版本 7 678/11 176；修复前缓存 8 343/8 344 … 最大移动 ≈**1.6e-2**）"，而 `:382` 又写 "8 344 of 8 345"——**代码内部也不一致**。

### D-31 `window` 的语义：doc 10 说"对两种输入都生效"，`_as_returns` 明确忽略它

* `docs/core-algorithms/10-volatility-targeting.md:141-145`：`window 对两种输入都生效（此前 DataFrame 分支会忽略它 … 已修）`。
* **代码** `core/ml/volatility.py:1214-1216`（`_as_returns`）：`The array is **not** windowed here (the window argument is accepted and ignored, kept so existing callers keep working)`。窗口是在**估计量层**、裁剪**之后**才施加的（`:400-426`）。

判定：doc 的"都生效"只在估计量层成立；`_as_returns` 这一层是显式忽略。历史差异（frame 33.6 ms / 0.003860 vs returns 2.2 ms / 0.003745，≈15.3×）记录在 `:1225-1227`，与 doc 的"慢 15×"一致。

### D-24 测试计数（汇总）

| 文件 | 文档记载 | 我在 `c703b8b` 实测 `def test` 计数 | 判定 |
|---|---|---|---|
| `tests/test_ga_credibility.py` | 29（证据索引 `:58`） | **29** | ✔ |
| `tests/test_ml_credibility.py` | 53（证据索引 `:12`） | **53** | ✔ |
| `tests/test_volatility_targeting.py` | 21（证据索引 `:13`） | **21** | ✔ |
| `tests/test_gap_fixes.py` | 12（证据索引 `:13`） | **12** | ✔ |
| `tests/test_pairs.py` | 28（doc 11:111） | **28** | ✔ |
| `tests/test_regime.py` | 14（doc 11:251） | **14** | ✔ |
| `tests/test_microstructure.py` | 20（doc 11:213） | **20** | ✔ |
| `tests/test_meta_labeling.py` | **20**（doc 12:61） | **24**（证据索引 `:14`/`:79` 也写 24） | ✘ doc 12 过期 |
| `tests/test_liquidity.py` | **24**（doc 13:291、P6 计划 `:58`） | **26**（证据索引 `:18` 记 25、`:209` 记本轮 26） | ✘ 两处过期 |
| `tests/test_condition_logic.py` | 9（证据索引 `:58`） | **10** 个函数（其中 1 个参数化 ×8 ⇒ 17 项收集） | ✘ 计数口径或文件已增长 |

**测试文件行数（子代理核对，与任务书的数字不符）**：`core/strategy/pairs.py` **1127** 行（任务书写 990）、`core/strategy/regime.py` **893**（写 793）、`core/market_data/microstructure.py` **611**（写 520）、`core/ml/volatility.py` **1434**（写 1241）。


---

## 附录 A：本次会话实际运行的命令（可复现）

```powershell
# 1) revision / 工作树
git log --oneline -3 ; git rev-parse HEAD ; git status --porcelain
#   → c703b8b / c703b8ba485a5410dba72ea5e0659aade0268276 / "M core/ml/features.py"

# 2) P6 兄弟们提到的新文件是否存在
Test-Path core\strategy\volume_bars.py          # → False（不存在）

# 3) 数据完整性（只读）
python scripts/check_data_integrity.py
#   → Files inspected: 29 ; RESULT: 25/29 file(s) carry a gap ; 无 NOTE 行（twin 0/29）

# 4) 模型产物计数（只读）
(Get-ChildItem data\models -Filter *.pkl | Measure-Object).Count          # → 15
(Get-ChildItem data\models -Filter *_meta.json | Measure-Object).Count    # → 0
(Get-ChildItem data\market -Recurse -Filter *.parquet | Measure-Object).Count  # → 29

# 5) 特征契约 hash 独立复算（从提交内容读，不看工作树）
git show HEAD:core/ml/features.py > $env:TEMP\feat_head.py
python $env:TEMP\dsh_hashcheck.py $env:TEMP\feat_head.py
#   → n DEFAULT_FEATURES: 40 ; json(list) -> 70899cff156d
#     去掉 "hurst" 后 39 项 -> 335e63360104（== 仓库声明的 v1 hash）

# 6) 波动率成本实测（见 §3.9 的完整脚本）
#   → ewma 600-bar 0.3093 ms / 500-bar 0.2248 ms / forecast_vol 600-bar 0.2806 ms

# 7) ML 可信度测量（本 revision 上失败，报错见 §2.9）
python scripts/ml_credibility_measure.py --symbols ETHUSDT --intervals 1h --tail 9000
#   → core.ml.features.FeatureContractError: feature matrix is missing 13 required
#     column(s) … (have 39, expected 52)

# 8) 测试用例数（只读）
Select-String -Path tests\test_*.py -Pattern '^\s*(async )?def test' | Measure-Object
#   → test_ga_credibility 29 / test_ml_credibility 53 / test_volatility_targeting 21 /
#     test_gap_fixes 12 / test_pairs 28 / test_regime 14 / test_microstructure 20 /
#     test_meta_labeling 24 / test_liquidity 26 / test_reaudit_fixes 15 /
#     test_residual_closure 10
```

## 附录 B：本文**未验证**的清单

1. 任何 GA 实测数字（`tools/ga_real_data_curve.py` 未重跑：>400 s 墙钟 + 子进程）。**例外**：`data/ga_jobs/**` 里的 job 日志/结果我在仓库内直接读到（§1.9 的 B 级表）。
2. 任何 ML 实测数字（`scripts/ml_credibility_measure.py` 在本 revision 上因兄弟代理编辑 `features.py` 而失败，见 §2.9）。
3. 30 次配对 E-G 检验的独立复现（`docs/core-algorithms/11` 的 0/30 是归档测量；测试只钉 0/10 on 1h）。
4. `hmm_two_state_causal` 的准确率与 2.3–2.6 s 成本（我读的是 docstring 记录，未跑）。
5. `garch11_params` 的实际成本、`garch_backend()` 的返回值、`arch` 是否可导入。
6. `core/ml/calibration.py` 的 `to_dict` → `from_dict` 逐位往返。
7. `core/ga/genome.py` 的 `mutate()` 逐类细节（文件正被兄弟代理编辑；子代理读过头，见 §1.10 局限 13）。
8. 工作树 v2 特征契约的列数、hash 与"v1 模型被拒"的行为（52 与 54 都是编辑中间态）。
9. `core/market_data/ohlcv_cache.py` 全部 525 行的逐行核对（只读了 `:1-145` 与 `:192-350` 的关键段；`merge_history`/`dedupe` 的完整实现未逐行读）。
10. `data/ga_trials.json` 的内容（当前不存在；§1.6 的跨窗口累计只有文档依据）。
11. `docs/core-algorithms/00-ERRATA.md` 的内容（我未读该文件）。
12. doc 10 的 `≈1.5 ms/bar` 是在哪个 revision/何种窗口上量的（§11 D-1）。
13. `fetch_features` 的 `trades_limit=100` 与 `trade_arrival_intensity` 内部 `MAX_TRADES=1000` 是否实际造成特征间样本不一致（§6.4 局限 10）。
14. `core/backtest/signal_matrix.py` 的 hybrid 路径我只读了子代理引用的 `:15-16`/`:303-304`，没有通读全文件。
