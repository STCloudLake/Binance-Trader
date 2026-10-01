# 算法升级证据索引（P1 GA / P2 ML / P3 波动率 / P4 新能力 / gap fixes / 复核修复）

**状态**: 证据已归档（本次实测，非引用） · **生成**: 2026-10-01 · **审计 revision**: `dcf3fdf`（平台内手册站；本文档自身的编辑**只动文档**，不是被审计代码的一部分） · **上一轮审计 revision**: `9d6f423`（GA 逐基因组进度流）；更早为 `b59af1d`（GA `sma` 缺陷修复）、`5c010e4`（实验性开关层）、`ea9922f`（P6 审计修复）、`cc63efd`（P6-B/C/D）、`1a452ce`（§0 表、§6、§9–§10 的数字来源） · **提交链**: `git log --oneline f1f6a4c^..HEAD`（见 §8 提交清单，含被审计代码的 **32** 行；其后如有纯文档提交，按 §8 说明不并入被审计 revision）
**工作树**: 本轮闭环修复（已提交为 `1a452ce`）由单一代理写入；改动文件为 `core/risk/manager.py`（D1/LOW-6：bar-key 折叠 + 非时间索引拒绝 + 覆盖范围说明）、`scripts/check_data_integrity.py`（同一折叠）、`core/market_data/ohlcv_cache.py`（LOW-5 注释）、`core/ml/credibility.py`（LOW-3 实测值）、`core/risk/position_guard.py`（R1 实测值）、`docs/core-algorithms/13-volume-liquidity-costs.md`（LOW-2）、本文件（LOW-4）、`README.md`（LOW-1）、`tests/test_residual_closure.py`（新）+ `tests/test_final_audit_fixes.py` / `tests/test_gap_fixes.py` / `tests/test_reaudit_fixes.py` / `tests/test_cache_durability.py` / `tests/test_measured_threshold_policy.py`（随行为/注册表同步）。上一轮（`b49883b` 工作树，后续提交为 `828e375`）的复核修复见 §9，改动文件为 `core/risk/position_guard.py`、`core/ml/credibility.py`、`core/ml/meta.py`、`core/backtest/engine.py`（仅 preload 校验）、`core/ml/predictor.py`（仅同侧车校验）、`core/market_data/ohlcv_cache.py` + `core/market_data/provider.py`（仅惰性去重/flush）、`docs/core-algorithms/13-*.md`、本文件、新增 `tests/test_reaudit_fixes.py`。`data/models/` 仍为 15 个 `.pkl` / **0 个 `_meta.json`**。
**对应计划**: [`ALGO_UPGRADE_PLAN.md`](ALGO_UPGRADE_PLAN.md)（P1–P4 验收标准） · **审计快照**: [`REFACTOR_AUDIT.md`](REFACTOR_AUDIT.md)

## 0. 一页速览

| 阶段 | 提交 | 核心断言 | 本次实测/钉住数字 | 证据 | 复现命令 |
|---|---|---|---|---|---|
| P1 GA | `4791369` | 种群内每个基因独立仓位/分账，跨代适应度不再恒定 | 真实缓存 4 代 best `5.9411→5.9411→9.9411→9.9411`（+4.00）；每代 6/6、5/6、4/6、5/6 基因交易（旧实现 1/20）；默认参数 3 代同样 +4.00 且 82.2 s 跑完 | `tools/ga_real_data_curve.py`（新）· `tests/test_ga_credibility.py`（29 项） | `python tools/ga_real_data_curve.py` |
| P2 ML | `4791369` + `2751bbb` | 未过门（OOS AUC>0.55 且净成本期望>0）的模型强制 `enabled:false` | ETH 1h：legacy 准确率 0.6597 vs 多数类 0.7773（−11.76pp）、AUC 0.5621；P2 门 AUC 0.5342、净期望 −0.1822%、t=−2.28 → 拒绝 | `scripts/ml_credibility_measure.py` · `tests/test_ml_credibility.py`（53）· `tests/test_engine_ml_gate.py`（7） | `python scripts/ml_credibility_measure.py --symbols ETHUSDT --intervals 1h` |
| P3 波动率 | `a9e549e` + `2751bbb` | 方向不可预测、条件波动率可预测；开关默认关闭 | 方向 AUC 0.52/0.53（本次独立复现 0.5207/0.5342）；BTC 1h 未裁剪 EWMA 5.31%/bar vs 裁剪 0.52%/bar（10.12×） | `tests/test_volatility_targeting.py`（21 项）· `tests/test_gap_fixes.py`（12） | `python -m pytest tests/test_volatility_targeting.py tests/test_gap_fixes.py -q` |
| P4 能力 | `a9e549e` | 四个新能力有独立参照、无前视、默认惰性 | HMM 三 regime 合成序列 ≥95% 准确率；ADF 模拟零分布复现 MacKinnon 临界值；真实 1h 主流币：配对与 meta-label 全部拒绝（有效结论） | `tests/test_meta_labeling.py`（24，正被并行修改）·`tests/test_pairs.py`（28，正被并行修改）·`tests/test_regime.py`（14）·`tests/test_microstructure.py`（20） | `python -m pytest tests/test_meta_labeling.py tests/test_pairs.py tests/test_regime.py tests/test_microstructure.py -q` |
| gap fixes | `2751bbb` | 实盘波动率定仓接线、成本语义统一、单一入场求值器 | 开关关闭时两侧均 240.0 USDT（balance 10 000）；BTC 成交 50 012.5000 / 往返 0.25005%；四个 AND/OR 基因入场数一致 40/0/0/40；sklearn 警告 1 602 → 0 | `tests/test_gap_fixes.py`（12） | `python -m pytest tests/test_gap_fixes.py -q` |
| 发布审计 | `2751bbb` | 路由/测试/编译/数据完整性 | 路由 118→118，ADDED 0 / REMOVED 0；全量测试复跑 919 passed / 1 skipped / 0 failed；`compileall` 退出 0；缓存 `25/29` 文件带 gap | `scripts/regen_route_baseline.py`（新）· `scripts/check_data_integrity.py` | 见 §6 |
| **最终独立审计**（历史快照） | **`fe11ccf`→`b49883b`** | 审计的每条 H/M 缺陷已修或已量化上报 | 该次实测：全量测试 **1040 passed / 0 failed**（227.3 s，退出码 0）；`compileall` 退出 0；缓存 **24/29** 文件带 gap、**twin 0/29**（`BTCUSDT/1h` 已实测去重，见 §6 与 §9 R3）；`data/models` 15 `.pkl` / 0 `_meta.json`；F1/F3/F4/F5 的 before→after 见 §8，本轮复核 1–7 的 before→after 见 §9 | **新** `tests/test_final_audit_fixes.py`（13 项）+ **新** `tests/test_reaudit_fixes.py` | `python -m pytest tests/test_final_audit_fixes.py tests/test_reaudit_fixes.py -q` |
| **闭环审计 + 本轮修复** | **`828e375`→`1a452ce`** | 复核 7 项残留 + 闭环审计 D1/LOW 1–6 全部关闭（1 项 MEDIUM 行为修复、1 项 MEDIUM 运行态上报、6 项 LOW 过期数字；D1/LOW 修复与"过期数字复核"同在 `1a452ce` 内） | 全量测试 **1070 passed / 0 failed**（当轮历史快照，见 §6；该轮两次运行实测）；`compileall` 退出 0；缓存 **24/29** 文件带 gap、`twin` **0/29**（`BTCUSDT/1h` 的形状见 §6 —— 行数/sha/mtime 随实盘写入变化，故不再钉住）；D1/LOW 的 before→after 见 §10 | **新** `tests/test_residual_closure.py`（13 项） | `python scripts/check_data_integrity.py` · `python scripts/check_data_integrity.py --symbols BTCUSDT --intervals 1h` |
| **残留收口**（当前） | **`1a452ce`→`5f50771`** | 独立裁决轮的 5 项残留（R1 文档钉错 revision、R2 守卫不可能失败、R3 measured-pin 的 skip 豁免过宽、R4 一项未交付的实盘深度前提、R5 doc 11 计数）全部关闭；**运行态代码未改**，改动仅 `tests/` + 本文件 | 全量测试 **1073 passed / 0 failed**（`216.61 s`，退出码 0；`--collect-only` **1073**）；`compileall` 退出 0；R2 守卫已实测"故意回滚文档即失败"；R3 三模块实验 2 问题/0 问题/3 问题；R4 上界由当次窗口推导（实测 `share=0.0111 < bound=0.0117`，绑定的不再是 "BTC 1h 很深" 这一实盘前提）。逐条见 §11 | `tests/test_reaudit_fixes.py`（R2 守卫）· `tests/test_measured_threshold_policy.py`（R3）· `tests/test_liquidity.py`（R4） | `python -m pytest tests/test_reaudit_fixes.py tests/test_measured_threshold_policy.py tests/test_liquidity.py -q` |

---

## 1. P1 — GA 可信性

### 1.1 真实缓存数据多代曲线（本次新增，`tools/ga_real_data_curve.py`）

此前 P1 结论只在**合成** parquet 上验证过（1m 基因组真实运行 >10 分钟）。本次用真实缓存给出一条**有硬墙钟上限**的曲线：种群 6、4 代、ETHUSDT+SOLUSDT、2025-07-01~2025-08-15、1h、seed 20260101、`max-seconds=470`。运行在子进程中执行，父进程到点即杀；每代完成即落盘 JSON，被杀也留有证据。

| 代 | best fitness | mean fitness | 参与交易基因 | 零交易 | 总交易数 | best Sharpe | best DSR | 会发布? | 拒绝原因 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.9411 | −7.5083 | 6/6 | 0 | 336 | 9.4388 | 0.2119 | 否 | `alpha_vs_buy_hold=-52.20% <= 0` |
| 2 | 5.9411 | −6.4140 | 5/6 | 1 | 428 | 9.4388 | 0.2119 | 否 | 同上 |
| 3 | 9.9411 | −9.5826 | 4/6 | 2 | 300 | 9.4388 | 0.2119 | 否 | 同上 |
| 4 | 9.9411 | −3.9661 | 5/6 | 1 | 543 | 9.4388 | 0.2119 | 否 | 同上 |

冠军：`fitness=9.9411`、75 笔、`dsr=0.2119`、`published=False`，唯一拒绝原因 `alpha_vs_buy_hold=-52.20% <= 0`。墙钟 54.3 s（cap 470 s，未被杀；另两次同参数运行 94.9 s / 118.3 s，负载不同）。

**逐基因账本（第 1 代，全部独立成交）**：`ga_rand_2=75`、`ga_rand_0=60`、`ga_rand_3=103`、`ga_rand_1=45`、`ga_rand_5=32`、`ga_rand_4=21` 笔 —— 旧实现的「1/20 交易、其余 0」不再出现。

**结论（诚实版）**：
1. **曲线不再是平的**：best 从 5.9411 升到 9.9411（第 3 代，+4.00），旧代码三代恒为 −1.30。
2. **但增益不是 alpha**：第 1 代最优与第 4 代领先者的 **交易结果完全相同**（75 笔、Sharpe 9.4388、DSR 0.2119、最大回撤 0.12%、收益 1.56%），差别只在复杂度罚项：条件数 9→4，`complexity_penalty` 12.9→8.9，差值 4.00 恰好等于 fitness 增量。即"更简约"，不是"更赚钱"。
3. **精英保留使 best 单调不降是结构性的**；能证明"搜索有进展"的是变异确实产出了更简且不回退的个体（第 3 代），以及 mean fitness 由 −7.51 改善到 −3.97。
4. **每代仍有零交易基因**（第 4 代 6 个中的 1 个 immigrant，`flag=no_trades`，fitness −46.71）；24 次评估中 20 次交易。不是"每个基因都交易"，但已从 1/20 变为 20/24。
5. **多样性收敛**：第 4 代 6 个基因中 4 个的交易结果逐位相同。
6. **发布门有效**：4 代全部拒绝，且理由是真实的（买入持有基准 −52.20%）。

> **before 数字的来源**：本文引用的旧实现数字（"20 基因中 1 个交易 407 次、19 个 0 次"、"三代 best 恒为 −1.30"、"5 笔全胜 490.85 vs 200 笔 7.30"）来自 P1 之前的只读审计，留档于 `ALGO_UPGRADE_PLAN.md` §一 与 `docs/core-algorithms/06-ga-evolution.md:90/:127`，**本次未重跑旧代码**（旧代码需从 `4791369^` 建 worktree）。本文标注"本次实测"的数字均可由 §0 的命令复现。

**确定性**：同 seed、同参数、各自干净工作目录跑两次，逐代记录（fitness/交易数/Sharpe/DSR/账本）**逐位相同**，仅墙钟秒数不同；冠军 fitness/DSR/交易数一致。证据 JSON：`ga_curve_real.json`、`ga_curve_real_repeat.json`、`ga_curve_real_v2.json`（上表口径）与 `ga_real_data_curve.json`（默认口径），均在系统临时目录。

**默认参数也能跑完（第二种配置的独立复现）**：`python tools/ga_real_data_curve.py`（种群 6、3 代、ETH+SOL+BNB、2025-07-01~2025-10-01、cap 420 s）实测 **82.2 s 完成、未被杀**：best `−4.0926 → −4.0926 → −0.0926`（第 3 代同样是 **+4.00**，同一复杂度罚项机制），参与交易 `6/6 → 5/6 → 4/6`，总交易 811/947/656，best DSR 0.0762，三代均 `published=False`（`alpha_vs_buy_hold=-49.63% <= 0`）。**但该配置的 mean fitness 反而恶化**（−11.19 → −13.86 → −16.33），说明"best 上升"不能被读成"种群整体变好"。

**脚本性质**：不写 `strategies/`、不写 `data/binance_trader.db`、不写仓库（`--out`/`--workdir` 默认在系统临时目录）；`--resume` 复用 GA 自身 checkpoint；`--inner` 为子进程内部模式；时间框架基因被包装器钉在 1h（未改任何源文件）。

### 1.2 其余 P1 验收项的钉板（合成数据/单元）

`tests/test_ga_credibility.py`（29 项）逐条钉住：每基因独立槽位与分账（`test_every_genome_in_chunk_gets_its_own_slots_and_ledger`，断言最闲 ≥ 最忙 × 0.2）、批量=单独评估、WF 传真实 `val_end` 且 OOS ≥30 bar、5 笔全胜不再压过 200 笔（`490.85` vs `7.30` 的旧公式在 `tools/p1_measure.py::_legacy_ranking` 中留档）、PF 收缩、真实 Sharpe/回撤/买入持有基准、DSR 手算与试验次数累计、发布门与 `enabled:false`+原因、条件清洗、按基因名交叉、无效基因修复、同 seed 复现、chunk 崩溃降级为单 worker、未来截断不改过去成交。补丁：`tests/test_condition_logic.py`（9 项，对应 4 条断言）关闭「打分用 AND、发布后重载成 OR」缺口。

## 2. P2 — ML 可信性（本次实测）

`python scripts/ml_credibility_measure.py --symbols ETHUSDT --intervals 1h --out <dir>`（另有 `--synthetic` 分支，同样读真实缓存）。特征契约 39 列，`feature_schema_hash=335e63360104`。

| 口径 | 基率 | 多数类准确率 | 模型准确率 | AUC | Brier | log loss | 净成本期望 |
|---|---|---|---|---|---|---|---|
| ETHUSDT 1h legacy（时序 80/20、±0.5% 二分类） | 0.2227 | 0.7773 | 0.6597 | 0.5621 | 0.2160 | 0.6224 | −0.041%（阈值 0.5） |
| ETHUSDT 1h P2（purged K-fold + embargo + 唯一性权重 + isotonic） | 0.5068 | 0.5068 | 0.5159 | **0.5342**（5 折均值 0.5379） | 0.2536 | 0.8371 | **−0.1822%** |

P2 硬门（`enabled:false`）：`OOS AUC 0.5342 <= 0.55; net expectancy -0.1822% <= 0.0000%（0.2600% 往返成本后）; not significant (t=-2.28 <= 2.00, PSR=0.011 < 0.95)`；OOS 样本 4 840、OOS 交易 585。BTCUSDT 1h 同样被拒（legacy 准确率 0.6834 vs 多数类 0.8140，AUC 0.5672；P2 门 AUC 0.5207、净期望 −0.2494%、t=−4.46）。**ML 保持关闭是实测结论，不是默认值。**

## 3. P3 — 波动率目标化 / 动态 barrier

方向不可预测（本次独立复现 OOS AUC 0.5207 / 0.5342，与 P3 测试文档所述 0.52/0.53 一致），条件波动率可预测。`tests/test_volatility_targeting.py`（21 项）钉住：forecast → 仓位 → 动态 barrier 三处消费路径、`risk.vol_targeting.enabled=false` 时**逐位不变**、随机数全部走 `np.random.default_rng(SEED)`。`tests/test_gap_fixes.py` 记录同阶段实测：BTCUSDT/1h 11 处 >1.5h 日历缺口、最大 1 484 h；未裁剪 EWMA 5.31%/bar vs 裁剪 0.52%/bar（10.12×），未裁剪值对拼接序列被拒绝。

## 4. P4 — 新能力

| 能力 | 独立参照 / 实测断言 | 测试 |
|---|---|---|
| Meta-labeling | 只做过滤+仓位缩放、不能翻向；标签侧别感知、丢弃超时；阈值搜索单边且在 purged/embargo 折内选择；噪声 meta 模型被硬门拒绝→惰性（`take=False`）；真实 1h 数据拒绝全部被测一级规则（有效基线） | `tests/test_meta_labeling.py`（24，正被并行代理改写） |
| 配对/协整 | 模拟 ADF 零分布复现 MacKinnon 渐近临界值；`adf_regression` 在随机游走上与零分布一致；Engle-Granger 对协整对有效、对独立游走规模正确；Kalman 动态对冲比跟踪时变 beta（不再塌缩到 ~0）；交易辅助函数无前视；真实 1h 主流币拒绝全部配对 | `tests/test_pairs.py`（28，正被并行代理改写） |
| Regime 门控 | 已知三 regime 合成序列：HMM 恢复已知波动率、≥95% 准确率、延迟数根 bar；tercile 分类器因果；两次运行逐位一致；门默认关闭且关闭时全放行 | `tests/test_regime.py`（14） |
| 微观结构特征 | 每个特征是单快照的纯函数并与手算订单簿一致；决策时间戳之后的成交在任何统计前被丢弃（注入巨量未来成交后特征逐位不变）；深度/成交数有硬上限；TTL 缓存有界；传输失败降级为 `None` | `tests/test_microstructure.py`（20） |

## 5. gap fixes（`2751bbb`）

`tests/test_gap_fixes.py`（12 项）逐条带 before/after：**①实盘波动率定仓接线**（`RiskManager.check_signal`→`PositionSizer`；开关关闭时 balance 10 000 两侧均 240.0 USDT，行为不变）；**②数据完整性**（见 §6，未裁剪波动率被拒绝）；**③`sim_cost_quote` 价差语义**（表内为全额价差、单边收 `spread_pct/2`；BTCUSDT 成交 50 012.5000、往返 0.25005%；ETHUSDT 50 015.0000、0.26006%）；**④单一入场求值器**（四个 AND/OR 基因入场 40/0/0/40、信号 bar 244/300/0/300，与旧内联 AND 循环逐 bar 相同）；**⑤sklearn 特征名警告 1 602 → 0**。`condition_logic` 的持久化/打分-发布一致性缺口见 §1.2。

## 6. 发布审计（本次实测）

| 项 | 命令 | 结果 |
|---|---|---|
| 路由基线 | `python scripts/regen_route_baseline.py` | 旧 118 → 新 118；**ADDED 0 / REMOVED 0**；`docs/overhaul/route-baseline.json` 与提交版本**逐字节相同**（`git diff` 为空）。P1–P4 未新增也未删除任何 `APIRoute`/`WebSocketRoute` |
| 全量测试 | `python -m pytest tests/ -q -p no:cacheprovider` | **残留收口轮（`5f50771`）：`1073 passed / 0 failed`**（`216.61 s`，退出码 0；该次运行里唯一失败是 §11 R2 的新守卫按设计抓出的本文件链长过期，修文档后复跑全绿，见 §11）；`--collect-only` 末行 **1073 tests collected**（本轮新增 3 项：`test_measured_threshold_policy.py` 2 项 + `test_liquidity.py` 1 项）。**闭环审计末轮（`1a452ce`，历史快照）：`1070 passed / 0 failed`**，连续 **3** 次全绿（`221.15 s` / `220.94 s` / `218.41 s`，退出码均 0）；第 3 次是在运行中的实盘进程**重写了 `data/market/XRPUSDT/1m.parquet` 之后**跑的（该文件 18:16:08 被改写，其 20 根 1m 窗口从 2 395 029.65 变为 **227.03** USDT），仍全绿 —— 即"套件在缓存被改写时保持绿色"是当轮实测而非推断；`tests/test_residual_closure.py` **13** 项、`tests/test_liquidity.py` 当轮 **25** 项（本轮 26 项）、`tests/test_microstructure.py` **20** 项。此前各次为历史快照：闭环审计首轮（`09125bd`+工作树）**1067 passed / 0 failed**（218.91 s / 215.91 s，退出码 0），连续两次一致。最终独立审计（`fe11ccf`+工作树）两次全量运行 **1040 passed / 0 failed**（470 s，退出码 0）与 **1039 passed / 1 failed**（314 s）。唯一的失败 `tests/test_ml_credibility.py::test_feature_pipeline_cost_is_bounded` 是**负载/计算预算敏感**断言（`elapsed < 3.0 s`；单独运行为 `1 passed in 0.90s`），与修复文件无关。同一工作树的**首次**运行为 `1022 passed, 3 failed`（242 s），3 个失败**均不在修复文件内**——`tests/test_measured_threshold_policy.py`（2 项）与 `tests/test_meta_labeling.py::test_real_primary_rules_are_refused_by_the_meta_gate`——三者随后修好，复跑即 `41 passed`。基线（`2751bbb`）：`919 passed, 1 skipped, 0 failed`（375 s） |
| 编译 | `python -m compileall -q app core web db scripts tools` | 退出码 **0** |
| 数据完整性 | `python scripts/check_data_integrity.py` | **闭环审计实测：`09125bd` 原样 = 25/29**（close 口径尾行 `BTCUSDT 1h …06:00:00 → …07:59:59.999` 被按原始时间戳差值误判为 1.99997 h 缺口 / missing 1，见 §10 D1）→ **D1 修复后 = 24/29**，该 5 个干净文件为 `ADAUSDT/1h`、`BTCUSDT/1d`、`BTCUSDT/1h`、`USDCUSDT/5m`、`ZECUSDT/5m`；`twin` 列 **0/29**（无 NOTE 行）。`BTCUSDT/1h` 的**形状**（不是内容指纹 —— 正在运行的实盘进程会不断重写该文件）：**11 625 行 / 11 625 bar 键 / 0 twin 行 / `missing`=0**，相邻差在**原始**与**bar 键**两种口径下**都**是 3 600.0 s（即 D1 的 close 口径尾行现在不存在；测量时刻 **2026-09-30 17:28**，sha256_16 `9864a654df75c32c`、mtime 17:28:14 —— 这两个值是**当次**读数，下一次实盘写入即失效，故本文件不再把它们当作主张）。⚠️ **本行曾写"`BTCUSDT/1h` 的 55 个 `twin` 已合并"，那是错的**（复核发现 3）：当时 `check_data_integrity` 报 **1/29** 文件带 twin，实测该文件为 **11 678 行 / 11 623 棵 bar / 55 棵重复 / 110 行 twin**。根因是 `flush_all` 只重写 **dirty** 键，永不追加的文件永远不会被去重。已在 §9 R3 修复并实测（当时 **11 623 行 / 0 twin**，文件 sha256 前缀 `0543b03d1ed8c3ff` —— 该值属于当时的内容，此后实盘又写入过 close 口径尾行、又把它换掉），`twin` 至今保持 **0** |
| 模型产物 | `Get-ChildItem data/models` | **15 个 `.pkl`、0 个 `_meta.json`** —— 修复前 `skip_ml_training` 会把 15 个未过门的模型全部加载；修复后全部被拒绝（见 §8 F4） |

`scripts/regen_route_baseline.py` 用**临时 DB + 临时 config 目录**构建应用（绝不打开生产库），枚举每个 `APIRoute` 的 method+path 与 `WebSocketRoute`，排序后按既有格式（indent=1、CRLF、无尾换行）写回；有 REMOVED 时退出 1。`--check` 只报告不落盘。

## 6.1 后续工作：P6（成交量 / 资金流）规划

P6 的目标、阶段划分与量化验收标准已冻结于
[`P6_VOLUME_PLAN.md`](P6_VOLUME_PLAN.md)：**P6-A**（参与率上限 + 冲击成本，已完成，提交 `b49883b`）、
**P6-B**（量能特征契约 v2 + 缓存 `quote_volume`/`trade_count` 扩展 + 重跑 ML 门）、
**P6-C**（美元棒/量钟采样与量能广度，离线实验）、**P6-D**（GA 量能条件模板/过滤与仓位基因 + 可执行性进入适应度）、
**P6-E**（逐阶段独立审计与发布）。§7 中"成交量利用不足"由 P6-B 接手；
`docs/core-algorithms/13-volume-liquidity-costs.md` 是 P6-A 的详细文档。
规划所依据的实测现状：39 列契约中量能占 5 列、GA 量能模板 3 条、P4 盘口 19 项特征默认关、
缓存只有 `open/high/low/close/volume`（无 quote volume）。

## 7. 尚未闭环（not yet closed）

1. **hybrid 引擎的 `condition_logic`（已关闭）**：在基线提交 `2751bbb` 上 `git show 2751bbb:core/backtest/signal_matrix.py | grep -c condition_logic` = **0**，即向量化混合引擎当时仍只做 OR，`condition_logic: and` 的冠军在该模式下会被按 OR 评估（标量内核 `StrategyConfig.entry_sides` 已一致）。该项在测量期间由并行改动关闭，并**已随 `beda096` 提交**（`git log --oneline -1 -- core/backtest/signal_matrix.py` = `beda096`，工作树干净）：`signal_matrix.py` 读取 `condition_logic` 并区分 AND/OR，`tests/test_hybrid_condition_logic.py` 收集 **4** 项。原文写的"尚未提交，且不在本次写权限内"是当时的运行态，现已过期。
2. **缓存缺口文件**：基线（`2751bbb`）实测 **25/29**；最终独立审计（`fe11ccf`+工作树）实测 **24/29**；复核（`b49883b`+工作树）复测仍为 **24/29**；闭环审计（`09125bd` 原样）为 **25/29** —— 新增的一处不是真缺口，而是 close 口径尾行（§10 D1），D1 修复后回到 **24/29**，干净文件见 §6（`BTCUSDT/1h` 已转干净）。**twin（同一根 bar 两种时间戳口径各存一行）已从 1/29 修复为 0/29**（§9 R3）。修复口径：`python scripts/download_history.py --symbols <SYM> --intervals <tf> --start <first> --end <last> --merge`。注意这些计数**只对测量时刻**成立：`data/market/**` 由运行中的进程持续重写，缺口/twin 计数会随之变化。
3. **杠杆与评估口径不一致**：`config/risk_params.yaml` 为 `leverage: 2` / `max_leverage: 4`，而 `core/ga/*.py` 与 `core/backtest/engine.py` 中 `leverage` 命中 **0** 次 —— GA/回测仍按现金模型计价，实盘盈亏与回撤约为其 2×。未修。
4. **路由基线无自动化回归门**：`tests/` 中无任何用例引用 `route-baseline.json` 或 `regen_route_baseline.py`；脚本只在人工运行时以退出码 1 报告删除。建议加一条只读断言（本文件之外的工作）。
5. **P5 独立只读复核**未在本证据范围内执行；本文所有数字均为本次实测，来源脚本与命令已逐条给出。
6. 并行改动的影响：`core/backtest/signal_matrix.py`、`tests/test_volatility_targeting.py`、`data/market/BTCUSDT/1h.parquet` 在基线测量期间被其他代理修改；最终审计期间被并行修改的还有 `core/market_data/ohlcv_cache.py` + `web/routes/backtest.py`、`core/risk/position_sizer.py` + `core/risk/liquidity.py`、`tests/test_meta_labeling.py`、`tests/test_pairs.py`。涉及这些文件的数字会随其落地而变化。

## 8. 最终独立审计（`fe11ccf`）——缺陷、修复与 before/after

新证据文件：`tests/test_final_audit_fixes.py`（13 项）。全部数字均为本次实测。

| 编号 | 缺陷（修复前实测） | 修复 | 修复后实测 |
|---|---|---|---|
| **F1** (HIGH) | 实盘波动率缺口闸门**永不触发**：`RiskManager` 缓冲裸 float，回退帧 `pd.DataFrame({"close": closes})` 是 **RangeIndex**，`_series_has_gap` 把 `1-0` 当 1 秒 → 永远 `False`；且 `wire_market_data` 无生产调用者。实测 300 根含 100-bar 空洞 → index `int64`、`gap_guard=False`、预测 **0.0698 %/bar** | `core/risk/manager.py`：kline 处理器缓冲 `(close_time, close)`（毫秒 epoch → UTC `Timestamp`）；`_buffered_frame` 用 **DatetimeIndex**；**无 `close_time` 的 bar 被拒绝**（不缓冲、日志说明），缓冲序列任何一根无时间戳即整体拒绝（不静默跳过闸门） | 同一序列：index `datetime64[ns, UTC]`、`gap_guard=True`、预测 **None**（拒绝）；无缺口缓冲仍产出预测 **0.0907 %/bar**；无时间戳 → 缓冲空、预测 None；开关关闭时数量 `0.0048`、止损 `49000.0` **逐位相同** |
| **F3** (MEDIUM) | `t_stat is None` 时显著性检查**整段跳过**：`credibility_gate({auc .60, n 5000, n_trades 300}, 0.004, t_stat=None, psr=None)` → `allowed=True, "pass"`，与自身 docstring 矛盾；旧 `gate_from_evaluation` 分支可传 None。另：文档写 `t>2 或 PSR≥0.95`，正态下 `PSR≥0.95` 恰为 `t≥1.645`（实测 `t=1.65, PSR=0.9505` → 放行） | `core/ml/credibility.py`：缺失 t **或** PSR → 拒绝，理由 `no significance evidence (...missing; the gate requires t > 2.00 AND PSR >= 0.95)`；语义定为 **AND**（见下）；`gate_from_evaluation` 新增 `min_psr` 透传 | `t=None/psr=None` → `allowed=False`、理由点名缺失项；好数字（t=2.6/PSR=0.99）仍 `pass`；边界 `t=1.60`、`1.65`、`1.70`、`2.00` 全部拒绝，`t=2.01` 放行；legacy 分支无外层统计 → 拒绝，带 `t/PSR` → 放行 |
| **F4** (MEDIUM) | `skip_ml_training` 直接 `MLTrainer.load_model()` 读盘，**无门控、无 `_meta.json`、无 schema 校验**；本仓库 `data/models` 有 15 `.pkl` / 0 `_meta.json`、2 个策略 `ml_config.enabled: true` | `core/backtest/engine.py`：preload 全部经 `_verify_ml_model_sidecar`（sidecar 存在 → `gate` 存在 → `gate.allowed` → `feature_names` 等于契约 → `feature_schema_hash` 相等），失败即拒绝并 `logger.warning`；LightGBM/TFT/PatchTST 三条路径都走 | 临时 models 目录 4 个 pickle：修复前 **4/4 全部加载**；修复后仅 gated+schema 匹配的 1 个加载，其余分别为 `no metadata sidecar`、`gate refused the model`、`feature schema hash mismatch`；本仓库 15 个 pickle → **15 拒绝** |
| **F5.2** (LOW) | `ml.gate_min_psr` 在 `app/config.py` 加载但**无处读取**（门控硬编码 `GATE_MIN_PSR`） | 值经 `MLPredictor._gate_config()` → `min_psr` kwarg → `gate_from_evaluation` → `credibility_gate` 全链路打通；`scripts/ml_credibility_measure.py` 同步补上 | `ml:` 块 26 个键的"生产读取者"表：**未读键 = `[]`**（`tests/test_final_audit_fixes.py::test_every_ml_config_key_has_a_production_reader` 常驻断言） |
| **F5.3** (LOW) | `default_meta_cost_pct(None, "ETHUSDT")` 返回 **0.14 %**（credibility 的硬编码回退），sim 成本是 **0.26 %**；`docs/core-algorithms/12` 的"与成交同源"只在显式传 config 时成立 | `core/ml/meta.py`：`config=None` 时**解析当前配置**（`Config.load()`）再取 `cost_pct_for`；只有配置本身不可加载才退回文档化默认值 | `default_meta_cost_pct(None, "ETHUSDT")` = **0.26**（= 传 config 的值 = `cost_pct_for`），BTC **0.25**；`docs/core-algorithms/12` 的说法现由构造成立 |

**F3 的 AND/OR 决定（及理由）**：采用 **AND**（`t > gate_min_t_stat` **且** `PSR >= gate_min_psr`，值为文档化的 2.0 / 0.95）。理由：①模块 docstring 本就把两个下限写成合取，只有实现用了 `or`；②`PSR` 就是正态单侧概率，`or` 使 2.0 的 t 下限完全失效（等价 1.645，实测放行了 `t=1.65`），而 2.0 是审计（P2 #3）钉住的值；③两个下限**现在**确实不冗余——`probabilistic_sharpe` 已改用 Prado 的偏度/峰度修正（复核发现 4，见 §9 R4），厚尾在 `t=2` 时把 PSR 压到 **0.9480 < 0.95**（正态近似报 0.9772）；④AND 严格强于 OR，只会拒绝更多、不会放行更多。边界由 `test_significance_floors_are_a_conjunction_at_the_boundary` 钉住（`t=1.60/1.65/1.70/2.00` 拒绝，`t=2.01` 放行，`t=3.0 & PSR=0.94` 拒绝），厚尾边界由 `tests/test_reaudit_fixes.py::test_psr_floor_is_stricter_than_the_t_floor_for_fat_tails` 钉住。

> 更正（复核发现 4）：本段原先写"PSR 读偏度/峰度"作为 ③ 的理由，但当时的 `probabilistic_sharpe` 是 `Φ(mean/se)`、**不读**任何高阶矩，因此那个论证是**假的**——AND 在当时恰好等价于 `t > 2`。现已实现真实公式，③ 才成立；见 §9 R4。

**连带影响的既有测试（均为"旧行为被钉住"，随修复同步更正，非放宽）**：`tests/test_ml_credibility.py::test_gate_passes_a_model_with_real_signal_and_positive_expectancy`（补 `n_trades/t_stat/psr`）、`::test_gate_requires_a_significance_floor`（`t=2.0 & PSR=0.975` 由放行改为拒绝）、`tests/test_gap_fixes.py::test_live_kline_stream_is_the_last_resort_forecast_source`（candle 补 `close_time`，与两个生产发布者一致）。

### 提交清单（截至被审计代码 revision `363b4c0`）

`git log --oneline` 自计划冻结起的提交列表（**被审计代码 revision = `3e90013`**，即最新一个改动代码/测试的提交；其后若只有纯文档提交——例如本文件的 `P6_VOLUME_PLAN.md` 交叉链接——则按下方规则**不**并入被审计 revision，因为守卫只要求链覆盖到最新改动代码的提交）

| 提交 | 说明 |
|---|---|
| `f1f6a4c` | 冻结 GA + ML 算法升级计划（P1–P5） |
| `4791369` | P1+P2：可信 GA + 诚实 ML 门控 |
| `a9e549e` | P3+P4：波动率目标化、动态 barrier、四项新能力 |
| `2751bbb` | P1–P4 + 审计修复整合 |
| `beda096` | 关闭 P3/P4 审计发现（因果 regime、PIT 缓存、真实 GARCH MLE、稳定裁剪） |
| `fa028be` | 缓存修复：实盘缓存不再覆盖长历史 + 合并 |
| `fe11ccf` | 最终独立审计基线：更正全部被证伪的数字；按 bar-open 键去重合并 bar |
| `0542e02` | 更正最后四处过期数字 |
| `b49883b` | 关闭最终审计缺陷、修复数据路径缺口、加入成交量感知的执行现实性（`tests/test_final_audit_fixes.py` 13 项） |
| `828e375` | 关闭复核 7 项残留（guard 绕过、文档数字、twin bars、PSR），新增 `tests/test_reaudit_fixes.py` |
| `09125bd` | README 测试数字更正（693 → 1055 passed） |
| `1a452ce` | 闭环审计基线：bar-key 折叠先于缺口检测（D1）、非时间索引拒绝（LOW-6）、确定性实盘用例、末轮文档/数字更正；新增 `tests/test_residual_closure.py`，见 §10 |
| `5f50771` | 上一轮：§11 的 5 项残留收口（evidence 守卫改钉运行期 HEAD/链长、measured-pin 的 skip 豁免改为"跳过必须支配函数体"、`test_liquidity` 容量上界改为按当次窗口推导、doc 11 计数 25→28）。改动全部在 `tests/` 与本文件 |
| `3e90013` | 上一轮：让证据索引守卫真正会失败、measured-pin 豁免收窄为"skip 必须支配函数体"（并因此暴露并修掉 `test_pairs.py` 中钉死 `120.0` 的隐藏断言）、`test_liquidity` 的深度前提改为按当次窗口推导 |
| `508e54a` | 冻结 P6 规划（`docs/overhaul/P6_VOLUME_PLAN.md`：P6-A…P6-E、量化验收标准、依赖、风险、DoD）并交叉链接；**纯文档** |
| `c703b8b` | 证据索引守卫改为可维护：被审计 revision 定义为"最新改动经验证代码的提交"（`core/app/web/db/scripts/tools/tests/config`），守卫校验"连续最旧优先前缀 + 覆盖到该 revision + 命名该 revision"，纯文档提交不再强制刷新（此前任何新提交都会永久变红）。改动在 `tests/` |
| `0b9266d` | 新增研究级公式文档 `docs/research/CORE_ALGORITHMS.md`（11 章 + 符号表 + 118 公式 + A/B/C 证据分级 + D-1…D-31 不一致清单）；撤回规划里过期的 P6-A 冲击百分比；**纯文档** |
| `cc63efd` | 上一轮：P6-B/C/D 实现 —— 缓存新增 `quote_volume`/`trade_count`（含回填与向后兼容）、ML 契约 v1 39 列 → v2 54 列（hash `335e63360104` → `1f30fded996d`）与版本化拒绝、P6-A 成交量缝隙接线（默认全关逐位一致）、`volume_bars.py`/`breadth.py`/实验工具、GA 量能模板与基因 + 冲击进入适应度。v2 契约门判定：**两者均被拒**（BTCUSDT AUC 0.5228 / 0 笔；ETHUSDT AUC 0.5324 / 净 −0.4132% / 889 笔 / t=−3.67） |
| `9694c42` | 证据索引补链并记录被审计 revision；**纯文档** |
| `4a7aaed` | P6-C 结案文档：美元棒/量钟**未改善**（因果成立但门判定 FAIL/FAIL，AUC 差 −0.0527）；量能广度已建成未接线（676 可用/705 报告的 USDT 交易对）；**纯文档** |
| `117e5ea` | 研究文档 D 项修复：D-16 改为**改正三处声明**（buy_hold 只报告与门控、不重算适应度；`tests/test_ga_credibility.py::test_buy_and_hold_is_reported_and_gates_but_does_not_rescore` 实测适应度 **−29.99** 对 None/0.0/10.0/−50.0/25.0 全部相同，alpha 由 None→0.0、25.0→**5.0**）、**D-19 真实标签订签缺陷修复**（timeout 类现按 `[start, stop)` 填充，predictor 统一计算持久化分布）、D-4 v1 哈希改为**可重算**（`_schema_hash(FEATURE_V1_NAMES)`）、7 个死配置键接线或删除（`ga:` 现仅 3 键、0 未读取）、DSR 试次计数统一、新增 `volume_flow` 指标族（按需、默认零成本；模板保留原写法并给出"改写会改变触发 bar"的证据）、成交量缩放基因改为**诚实文档化**（评分层可执行性模型）；外加两条活缓存脆断言去钉（注入 +2.5%/+6%/−6%/+3 根/拼接式 +30% 全部通过）与一批文档数字扫正 |
| `363b4c0` | `config/config.yaml` 的 garch11 成本注释由 "≈12 ms/bar" 改为实测值（默认 500 窗口 ≈0.13–0.16 s/次，window=0 ≈3.7 s；廉价方法 ≤1 ms/bar，实时形状 ewma 0.14–0.20 ms） |
| `b81094b` | 证据索引补链；**纯文档** |
| `8dec7c2` | **README 重写**（1007 → 340 行；删除不可维护的导航/重复配置表/历史叙述/过期计数；每个数字重新测量；17 个相对链接可解析）与**研究文档续写 §12**（P6 量能：契约 v2、缓存列、缝隙与成本分解、量钟实验、广度、GA 模板与可执行性模型、"P6 未达成什么"；新增 D-32…D-34）；**纯文档** |
| `ea9922f` | 上一轮：关闭 P6 审计发现 —— ①未测量的 `quote_volume` 不再被伪造成"完美流动性"（全 NaN→`volume×close` 代理、部分 NaN→未测行保持 NaN，`out.attrs` 掩码穿透暖机填充）②"旧契约按名拒绝"对真实 v1 产物可达（可识别但不同的哈希在列数检查**之前**拒绝，两条路径都有真实 39 名 + v1 哈希的测试）③doc 13 过期声明改正 + **回测入场定仓接入参与率窗口**（惰性 callable，关闭时从不求值；48 笔整轮哈希两侧相同）④`test_p34` 残留缓存钉死改为 `ratio > 5` ⑤`breadth` 的 `max_stale_ms` 首次真正比较（暴露 `missing=True`）⑥未消费的 position_guard 缝隙在 docstring 说明 ⑦`cum_qv` 的 NaN 缺陷经实测**并不存在**（201/201 累计点与丢弃该 bar 一致），仅改注释 ⑧成本上限测试改为**三批取最小 + 机器无关比值守卫**（并证明 10× 变慢仍失败）⑨D-32/33/34 与规划 2.6×/3×、doc 15"美元棒更差而非略好"的结论纠正 |

| `3a140cf` | 新增 [`P6_VOLUME_EVIDENCE.md`](P6_VOLUME_EVIDENCE.md)（P6 逐阶段实测证据、v2 契约下的门判定、量钟实验的负面结论、广度覆盖与 TTL、审计残余处置、未达成清单）并把被审计 revision 钉到 `ea9922f`；**纯文档** |
| `5c010e4` | **当前被审计代码 revision**：①**实验性开关层**——`config/config.yaml` 的 `experimental:` 块（8 个开关、默认全 false）驱动原先只能改 Python 常量的能力（引擎的 regime 诊断/meta 过滤/pairs 信号、regime 门控与诊断、pairs 能力、meta-labelling、microstructure），`app/config.py` 增类型化配置 + 目标表 + 启动提示，`app/main.py` 启动时应用；模块默认值仍为 `False`，配置块缺失时逐位一致（有 unmodified HEAD 工作树对照哈希）；17 项新测试（键必有读取者、默认关、打开只翻自身、未知键报告、全关时信号/定仓逐位一致）+ README 增补"实验性开关"节 ②**回填导致的测试修正**——`data/market` 29/29 文件迁移到 v2 列后，3 条钉死 `Σ volume×close` 代理的老测试改为**从所读帧推导期望**（BTC 1h 真实 `quote_volume` 1 105 175 220 vs 代理 1 106 675 201，差 0.14%），代理规则另用合成 v1/v2 帧钉住，并有临时副本证明两状态都通过；另一条近临界断言改为推导带 |

| `4a5f0fd` | 证据索引补链（`3a140cf`、`5c010e4`，被审计 revision → `5c010e4`）；**纯文档** |
| `9d6f423` | **当前被审计代码 revision**：GA 真实进度流 —— 根因是多进程评估**只在"整块（7–8 个基因组）算完"时上报一次**（该任务一块需数小时），进程内路径才逐基因组 tick；现转发引擎已有的逐 bar 观察器 + 每完成一个基因组发一次 tick（队列仅在存在监听者时创建，否则零开销），载荷扩为 `phase/generation/total_generations/eval_completed/eval_total/eval_equivalent/chunk_progress_pct/bar_step/bar_total/elapsed_s/best_fitness/best_trades` + `started_at/updated_at`，每代写一行 INFO 日志，状态路由与面板显示新字段；新增只读 `scripts/ga_job_status.py`（对修前任务也可用，进度过期即非零退出）；**刻意不把块拆成逐基因组**（`engine.py:706-708` 的 `max_positions = max_open_trades // 策略数` 会让单基因组拿到 15 槽位而非 1，改变 GA 结果）。装置实测写入序列 `1,1,2,2,3,3,4,4 → gen_complete`；套件 1252 passed / 0 failed ×2 |

| `39d1684` | 证据索引补链（`4a5f0fd`、`9d6f423`）；**纯文档** |
| `b59af1d` | **GA `sma` 缺陷修复**：`compute_all` 的"总是派生"块补齐了 volume_sma/volume_ratio/ema_fast/ema_slow 却**漏了 `sma`**，而 `genome.py` 声明其总是可用、清洗器兜底正会生成 `close > sma`/`close < sma` → 求值器报 `unknown column` 并返回全 False 掩码，条件被**静默丢弃**（线上日志恰 6 次、全为 sma）。修法：① 补 `SMA(close,20)`（实测 9000 根上 1.323 → 1.404 ms/调用，+0.081 ms）② 含不可求值条件的基因组改为**显式失败**（`UnevaluableConditionError`、fitness −999 + flag + 计数 + 每基因组一条警告，结构性排除出选择；被拒槽位由永不交易占位者持有，保证 `max_open_trades // 策略数` 对幸存者不变）。装置：修前 11 处问题（`close > sma` 触发 0/706、fitness −35.9/0 笔、未知列被接受）→ 修后 0 拒绝（入场 342/706、出场 345/706、fitness −9.2155、62 笔）；36 个归属列全部核对、`ALWAYS_AVAILABLE_COLUMNS ⊆ compute_all(frame,{})` 成立、300/300 随机基因组可解码 |
| `dcf3fdf` | **当前被审计代码 revision**：**平台内手册站** —— `GET /manual`、`GET /manual/{doc_path}`、`GET /api/manual/tree`（沿用现有鉴权，匿名 302 / API 401）；发现规则为 `docs/**/*.md` + 根目录 `README.md`/`README_EN.md`（共 **64 篇**）；服务端渲染的可折叠目录树（按目录分组、计数、当前篇高亮、客户端过滤、hash 跳转自动展开祖先）+ 面包屑 + 页内 TOC + 前后篇 + 文档互链改写（失效目标渲染为不可点跨度）；markdown-it-py（已装，现声明）CommonMark+表格、`html=False`；数学公式**先抽槽再回填**，研究与公式文档的 **87 个 `$$` 块 + 538 个 `$…$`** 经 KaTeX CDN 渲染（不可达时回退原文）；路径安全：16 个恶意串 + 11 个恶意 URL（含编码变体、盘符、NUL）一律 404 且不泄漏，原始 HTML 被转义，`javascript:`/`data:` 等不作链接；路由基线 118 → **121**（added 3 / removed 0，GET 69→72）；58 项测试 |

## 9. 复核修复（`b49883b` 工作树）——7 项发现的 before/after

新证据文件：`tests/test_reaudit_fixes.py`。全部数字均为该轮实测；该轮改动已随 `828e375` 落地（本节即其归档）。

| 编号 | 缺陷（修复前实测） | 修复 | 修复后实测 |
|---|---|---|---|
| **R1** (MED) | `PositionGuard.forecast_vol_pct` 自行用 provider 历史建 `DatetimeIndex` 帧但**不做** `_series_has_gap`，于是同一条拼接序列 `RiskManager → None` 而 `PositionGuard → 0.062009291811329616 %/bar`，且该值喂给**实盘移动止损距离** | `core/risk/position_guard.py`：预测前调用与 manager **同一个** `core.risk.manager._series_has_gap`（同一 `_VOL_MAX_GAP_BARS`=1.5 与同一 bar 长度表），命中即 `return None` 并 `logger.warning`；`_resolve_vol_pct` 随之回落到固定距离 | 同一拼接序列：guard 预测 **None**（拒绝，日志可见），manager 同样 **None**；无缺口序列两侧同为 **0.061993341684305286 %/bar**（逐位相同）。这两个数字由闭环审计重测（`tests/test_reaudit_fixes.py::_vol_frames()`：clean 0.061993341684305286 / unguarded spliced 0.062009291811329616）；本节原写的 `0.41803165815 %/bar` 用同一数字描述了**两条不同序列**且无法由任何现有 fixture 复现，已更正。开关关闭时移动距离与 pre-P3 逐位相同 |
| **R2** (MED) | `docs/core-algorithms/13` §3 的影响成本百分比**不可复现**：声称 1.0 BTC 往返 `+178.8965`（+2.36 %）/`+894.4824`（+11.82 %）"钉在同一 20 根窗口"，但同页引用的窗口是 750,661,812.73 USDT，代码在该窗口给出 **+1.7455**（+0.0021 %）/ **+8.7273**（+0.0105 %）；`0.8288` 的 impact 行也属于另一个窗口 | 文档：给出**可复现命令**，并把整节重测到**一个显式命名的窗口**上，附订单规模递增表说明"零售规模影响≈0、只有成为窗口的可见比例才生效"；同时记录旧数字各自隐含的窗口（≈756.78 M / ≈541.7 M / ≈729.2 M），说明它们为何不可比对 | 同一命令：窗口 **751,885,467.06 USDT**（ends 2026-09-30 06:00:00，price 83 043.14）。0.01 BTC **+0.0017**（+0.0002 %）；1.0 BTC **+1.7455**（+0.0021 %）；10 BTC **+55.1963**（+0.0066 %）；100 BTC **+1 745.4596**（+0.0210 %）（`k=0.5` 分别为 +0.0087/+8.7273/+275.9814/+8 727.2978）；`k=0` 与 legacy **逐位相同**（实测 `75.48621426 == 75.48621426`）；0.6 BTC 分解 impact **0.8281**、total 46.2918 |
| **R3** (MED) | §6 原写 `BTCUSDT/1h` 的 55 个 twin "已合并"是**假**：`check_data_integrity` 报 **1/29** 文件带 twin，实盘文件 **11 678 行 / 11 623 棵 bar / 55 棵重复 / 110 行 twin**。根因：`flush_all` 只重写 **dirty** 键 | `core/market_data/ohlcv_cache.py`：新增 `_frame_hash` + `_canonical_write`（内容哈希判"写是否会改变文件"）与 `OHLVCache.dedupe`；`flush_all` 在 dirty 键之后对**所有已加载键**做一次去重，仅在内容真的改变时落盘 | 实盘 `data/market/BTCUSDT/1h.parquet`：**11 678 → 11 623 行**，`twin_bars` **55 → 0**，`twin_rows` **110 → 0**，bar 键集合**完全相同**（11623），55 棵冲突 bar 全部保留**较新**的那一行；`check_data_integrity` 该行转 **ok**、`twin` 列 **0**、"1/29 files store a bar twice"提示消失。第二次 flush **不写盘**（`dedupe → False`），文件 **字节完全相同**（sha256 `0543b03d1ed8c3ff`）。**闭环审计复测（当时文件）**：**11 624 行 / 11 624 bar 键 / 0 twin 行**（该 sha `0543b03d1ed8c3ff` 与行数属于当时的内容，此后实盘又写入过 §10 D1 的 close 口径尾行、又把它换掉）。**末轮复测（`1a452ce`，测量时刻 2026-09-30 17:28）**：**11 625 行 / 11 625 bar 键 / 0 twin 行**，原始与 bar 键相邻差全为 3 600.0 s —— 这些是**当次**读数（sha256_16 `9864a654df75c32c`、mtime 17:28:14），实盘继续交易即变，不作为主张 |
| **R4** (LOW) | `core/ml/credibility.py:246-259` 是 `Φ(mean/se)`、**不读**偏度/峰度，而 `:743-746` 与 `core/ml/meta.py:47` 及文档 §8 声称它读——AND 因此**恰好等价于 `t > 2`** | 选择**实现** Prado 的修正公式：`SE_adj = sd·√((1 − γ₃·SR + (γ₄−1)/4·SR²)/(n−1))`，`returns` 作为可选参数（缺省/样本不足/矩非有限时退化为正态近似）；`net_trade_stats` 与 `_signed_net_stats` 把净收益序列传入；docstring/doc 改为**真**陈述 | 正态样本 `t=2.0000` → PSR **0.9772498680518209**（本行曾写 `0.9773`；旧式即同一数值 `0.9772`，逐位不变）；厚左尾 10 个 −40σ 异常值、`t=2.000000` → **0.9479777894541446 < 0.95 → 门拒绝**（同序列旧式报 0.9772）；同形状 `t=3` → **0.9869562688374416（本行曾误写 `0.9864`）→ 放行**。`tests/test_ml_credibility.py::test_probabilistic_sharpe_matches_the_normal_approximation` 的 6 条断言全部仍然通过（`returns=None` 走同一公式） |
| **R5** (LOW) | `core/backtest/engine.py:1778` 的 `if stored and …` 与 `core/ml/predictor.py:672` 接受**缺 `feature_schema_hash`** 的侧车（审计 e2e 因此加载了 7 个产物中的 2 个） | 两处均改为**要求存在**：engine 返回 `False, "feature schema hash missing: …"`；predictor 抛 `FeatureContractError("… carries no feature schema hash …")`（与既有的 mismatch 异常同类） | 缺 hash 的侧车：engine **拒绝**并给出含 `feature schema hash missing` 的理由，predictor **抛异常**；完整侧车（`gate.allowed` + 契约名 + 正确 hash）**照常加载**；本仓库 `data/models` 15 个无侧车 `.pkl` 仍全部拒绝 |
| **R6** (LOW) | 本文件头仍写基线 `fe11ccf`、提交列表缺 `b49883b`、结尾写"最终审计（未提交）" | 头部改钉 `b49883b` 并列出工作树为本轮修复；提交清单补齐到 `b49883b`（含 `b49883b` 一行）；`data/models` 现状、全量测试数字、twin 行全部按本轮实测重写 | 当时的本文件（`git rev-parse HEAD` = `b49883b98a33354e83b0362fbd1c6eaca97a2ed8`）。**闭环审计再钉**：`git rev-parse HEAD` = `1a452cefdce041a74df5441028f363ef1e870913`，提交清单已补齐 `828e375` + `09125bd` + `1a452ce`。**残留收口轮再钉（§11）**：提交清单已补齐到 `5f50771`（`git log --oneline f1f6a4c^..HEAD` 共 **13** 行，由运行期断言 `len(chain) == git rev-list --count` 常驻校验） |
| **R7** (LOW) | `core/ml/credibility.py:939` 的 `__all__` 导出未定义的 `signed_score`，`from core.ml.credibility import *` 抛 `AttributeError` | 移除该条目（`signed_score` 属于 `core.ml.calibration`，无生产调用者从 credibility 导入它——已 grep 全仓库确认），并加注释说明 | 星号导入**成功**；`[n for n in credibility.__all__ if not hasattr(credibility, n)] == []`；`tests/test_reaudit_fixes.py::test_star_import_of_credibility_resolves` 常驻钉住 |

**默认配置不变**：波动率目标化关闭、`risk.liquidity` 关闭、ML 关闭、P4 全部开关关闭时为逐位不变；vol targeting 打开但序列有缺口时，止损回落到文档化的固定规则。

## 10. 闭环审计（`1a452ce`）——D1 与 LOW 1–6

新证据文件：`tests/test_residual_closure.py`（13 项）。全部数字均为本轮实测；命令与观测值逐条给出。

| 编号 | 缺陷（修复前实测） | 修复 | 修复后实测 |
|---|---|---|---|
| **D1** (MED) | 拼接口径闸门在实盘缓存上**误报**：`data/market/BTCUSDT/1h.parquet` 为 11 623 个 bar-open 时间戳 + **1 个** `07:59:59.999` close 口径尾行（共 11 624 行），原始时间戳相邻差 `06:00:00 → 07:59:59.999` = 1.99997 h > `_VOL_MAX_GAP_BARS`=1.5，于是**连续的**序列被判为缺口：`_series_has_gap=True`、`RiskManager`/`PositionGuard` 预测 **None**、`check_data_integrity` 报 `BTCUSDT/1h` **GAP / missing 1**、全仓 **25/29** 带缺口。按 bar 键折叠后 max step 1.0 h、0 缺口 | `core/risk/manager.py::_series_has_gap` 与 `scripts/check_data_integrity.py::gap_report` 在**差分之前**把时间戳折叠到声明的 bar 键（复用 `core.market_data.ohlcv_cache.bar_keys`，即缓存写路径与 `download_history.py` 同一条区间规则） | 同形帧（300 open + 1 close）：`_series_has_gap` **False**；`RiskManager` = `PositionGuard` = **0.07684716805517097 %/bar**（逐位相同，本轮复测一致）；`gap_report` missing **0** / gaps **0** / flagged **False**。实盘文件：`--symbols BTCUSDT --intervals 1h` 由 **GAP / missing 1** 变为 **ok / missing 0**；全量 **25/29 → 24/29**。真缺口仍拒绝：少 1 根 bar → `_series_has_gap=True`、预测 None、`gap_report` missing 1 / gaps 1；100-bar 拼接 → 仍 **True**（最大缺口 100.0 h、missing 99）。⚠️ **实盘 1h 文件现在不再呈现 D1 的形状**（测量时刻 2026-09-30 17:28：11 625 行，原始相邻差**也**是 3 600.0 s，close 口径尾行已不存在），所以"实盘文件由 GAP 变 ok"这一类**头条检查**只在**重建帧**（`tests/test_residual_closure.py::_reconstructed_live_shape`）上才有意义；实盘文件上仍然成立的不变量是"报告与 bar 键视图对同一份内容给出同一判决"，由 `::test_live_file_bar_key_shape_matches_the_reporter` 无条件断言（不再用 `pytest.skip` 放行"文件真有缺口"的情况） |
| **D2** (MED, 运行态) | 运行中的 PID **30628** 启动于 **15:14:35**，而 `828e375`（提交于 **16:06:45**）修改的 6 个 `core/*` 文件写盘于 **15:36:53–15:50:38** —— 正在跑的不是被审计的代码 | 不重启进程（由 Lead 负责） | 已重启：当前 PID **22352**（`python -m app.main --mode sim`），操作系统报告的创建时间 **2026-09-30 17:22:53**（`Get-CimInstance Win32_Process -Filter "ProcessId=22352"`，**不是** 17:23:55），即它加载的是含 D1 修复的当前代码。⚠️ **但这不构成"对修复的实盘探针"**：`config/config.yaml` 的 `risk.vol_targeting.enabled: false`（P3 默认关闭），该开关关闭时 `forecast_vol_pct` 的消费路径整体短路，D1 的折叠闸门在运行进程里**根本不会被走到**；同理"guard 行为被实盘探针验证"在开关打开之前是不可能的。本节所有实测均来自源码/脚本/测试帧，**不**依赖运行中的进程 |
| **L1** (LOW) | `README.md` 的 790/800/803/812/889 行仍用 `693` / `8 failed, 658 passed, 1 skipped` | 改为本轮实测值；干净克隆两个口径分别实测并标注 | §10.1 前提说明整段重写（见 `README.md` §10.1） |
| **L2** (LOW) | `docs/core-algorithms/13-volume-liquidity-costs.md` 的回溯窗口归属错误：`+178.8965` 被写成"≈ 7.1×10⁵ USDT 窗口"，`0.8288` 被写成"属于 ≈ 729.2 M 窗口，距 750.66 M 有 3 %" | 用代码反解并逐项重测，删除猜测性归属 | `total_impact_usdt` 反解：`+178.8965` → **71 576.03 USDT**（7.16×10⁴，比原主张小一个数量级）；`+894.4824` → **2 863.04 USDT**；`0.8288` 恰在表内 750,661,812.73 USDT 窗口复现为 **0.828812**，而 729.2 M 窗口给 **0.840920** |
| **L3** (LOW) | `core/ml/credibility.py:279` docstring 写 corrected PSR = `0.937`（`:803`、`:856` 同） | 三处改为实测值 | `net_trade_stats(_fat_left_returns())`（`t_stat=2.0000000000000004`）：corrected PSR **0.9479777894541446**（≈ `0.94798`，本文件曾写 ≈ `0.9480`，四舍五入无误）、正态近似 **0.9772498680518209**（≈ `0.97725`，曾写 ≈ `0.9772`）。同形状 `target_t=3.0`（`t_stat=2.9999999999999996`）：corrected PSR **0.9869562688374416**（曾误写 `0.9864`），正态近似 **0.9986501019683699** |
| **L4** (LOW) | 本文件 `:17`/`:91`/`:93`/`:148`/`:150`/`:153` 与提交清单过期（缺 `828e375`、`09125bd`；测试数 1040；缺口数 24/29；twin 行/sha；`0.41803165815`） | 全部按当前 revision 重钉，历史数字标注为历史快照 | 见本文件 §0 表、§6、§7、§9 R1/R3/R6 与上方提交清单。**末轮再钉（`1a452ce`）**：`git log --oneline f1f6a4c^..HEAD` = **12** 行；全量 **1070 passed / 0 failed**。**残留收口轮再钉（`5f50771`，§11）**：该守卫本身已改为运行期断言（链长 = `git rev-list --count`、HEAD 必须在文内），本节"12 行 / 1070"随之成为历史快照 |
| **L5** (LOW) | `core/market_data/ohlcv_cache.py:277` 注释称实盘 1h 缓存有"54 个 1 毫秒邻居" | 该数字属 revision `0542e02` 的历史快照，按当前实测改写 | 当时实测（**历史**）：`pairs=11623`、**min delta = 3600.0 s**、`Δ ≤ 1 ms` 的相邻对数 = **0**、唯一非 1 h 差值为 **7199.999 s**（即 D1 的 2 h − 1 ms 口径差）。本轮复测（`1a452ce`，测量时刻 2026-09-30 17:28）：`pairs=11625`、**min delta = max delta = 3600.0 s**、**不存在任何非 1 h 差值**——那个 7199.999 s 的口径差随 close 口径尾行一起从文件里消失了 |
| **L6** (LOW) | `_series_has_gap` 对非 datetime 索引吞掉异常并返回 **False**（"无缺口"），一旦调用方传入非时间索引即等于关掉闸门（审计 F1 的失效模式） | 不可读的索引一律**拒绝**（返回 True）；未知 interval 仍保持惰性（False） | `pd.RangeIndex(300)` / `Float64Index` / object 字符串 / tz 混合 → **True**；`"7h"` 未知 interval → 仍 **False**；空/少于 3 根的合法日期索引 → **False**；`PositionGuard` 收到 RangeIndex 帧 → 预测 **None** |

**复现命令（本轮，全部只读）**：`python scripts/check_data_integrity.py`；`python scripts/check_data_integrity.py --symbols BTCUSDT --intervals 1h`；`python -m pytest tests/ -q -p no:cacheprovider`；`python -m compileall -q app core web db scripts tools`；bar 键/相邻差与 `total_impact_usdt` 反解的 one-liner 见 §3 的命令块与本节各行的数字。

### 10.1 明确说明、**不**在代码里修复的限制

1. **实盘预测只看得见最后 600 根 bar**（`core/risk/manager.py::_VOL_HISTORY_BARS`）。折叠后的缺口闸门因此只能拒绝落在该窗口内的洞：位于 −100 或 −300 的洞会被拒绝，而位于文件中间（例如 −900）的洞**对它不可见**。整文件口径的检查是 `scripts/check_data_integrity.py`。该限制已写进 `manager.py` 的常量注释。
2. **+27.63 % 拼接与 close 口径尾行在实盘文件里都已不存在**（`core/risk/manager.py` 的 `_VOL_MAX_GAP_BARS` 注释、`scripts/check_data_integrity.py::gap_report` docstring 均已标注为历史）。测量时刻 2026-09-30 17:28：原始最大相邻步 **1.0 h**，`clipped == unclipped = 0.3003 %/bar`（**1.00×**，`python scripts/check_data_integrity.py --check-vol --symbols BTCUSDT --intervals 1h`）。缺陷机制本身仍真实，由重建帧/合成帧钉住（`tests/test_gap_fixes.py`、`tests/test_residual_closure.py`）。
3. **`tests/test_microstructure.py` 的改写用例是纯 stub**：所有 20 项都在测试内构造 Binance 形状的快照，**没有任何用例端到端地跑真实 provider**。这消除了一类实时时序抖动（见该用例 docstring），代价是真实 provider 的接线不再有测试覆盖；该限制已写进该用例的 docstring。
4. **运行中的进程不是修复的探针**：PID 22352 启动于 17:22:53、加载的是当前代码，但 `risk.vol_targeting.enabled: false`，D1 的闸门在实盘进程里不会被走到（见 §10 D2）。

## 11. 残留收口（`5f50771`）——独立裁决的 5 项

新证据：`tests/test_reaudit_fixes.py`（R1/R2 守卫重写）、`tests/test_measured_threshold_policy.py`（R3，+2 项自测）、`tests/test_liquidity.py`（R4，+1 项合成分支）。**本轮没有改动任何运行态代码**（`app/`、`core/`、`web/`、`db/`、`scripts/`、`tools/` 全部未动），所以被审计的**代码** revision 仍是 `1a452ce`；本文件自身的编辑是 **doc-only**，`tests/` 的改动是 test-only。

| 编号 | 残留 | 修复 | 修复后实测 |
|---|---|---|---|
| **R1** (LOW) | 本索引把"本文件所在的 HEAD"写成 `1a452ce`（实际 HEAD 已是 `5f50771`）、提交清单 **12** 行而 `git rev-list --count f1f6a4c^..HEAD` = **13**、§9 仍把已随 `828e375` 落地的工作写成"未提交" | 头部改为钉**审计 revision** `5f50771` 并显式区分"上一轮审计 revision `1a452ce`"与"本文档编辑是 doc-only"；提交清单补 `5f50771` 一行（13 行）；§9 的"未提交"改为"已随 `828e375` 落地" | `git log --oneline f1f6a4c^..HEAD` = **13** 行、链长由 R2 守卫运行期断言；头/§8/§9/§10 L4 全部改钉 `5f50771` |
| **R2** (MED) | `tests/test_reaudit_fixes.py:599-617` 的守卫把 `git rev-parse HEAD` 存进变量后只断言其**真值**，再接一个以 `b49883b` 结尾的**固定列表**——文档落后 5 个提交时它照样 `1 passed`，这正是过期反复出现三次的机制 | 守卫改为运行期断言：①链长 == `git rev-list --count f1f6a4c^..HEAD`；②当前 HEAD 短 hash 必须出现在文档中；③链首为 `f1f6a4c` 且无缺行；④链内已落地提交不得仍被写成"未提交"。失败信息给出缺失提交集合与表内容 | 见 §11.1：故意把文档副本回滚到 `1a452ce` 的链（12 行、不含 `5f50771`）→ 守卫 **失败**；当前 revision → **通过**。`.git` 缺失但 `git` 可用 → **失败**（不是 skip）；仅当 `git` 完全不可用时才 skip，且 docstring 写明 |
| **R3** (LOW) | `tests/test_measured_threshold_policy.py:382-383` 只要函数体出现 `pytest.skip(` 就整体豁免：同一模块带 measured pin（`assert auc <= 0.5866`）**无 skip = 有非法字面量**、**加一个 guard-clause skip = 0 个问题**（裁决轮在只做字面量检查的文件上量到 2 个问题） | 豁免条件改为"skip 必须**支配**函数体"：函数第一条语句必须是 `pytest.skip(...)` 或"守卫式 `if ...: pytest.skip(...)`"（`_skip_dominates_the_body` / `_is_early_skip_guard` / `_ends_in_a_skip`），且该语句之前不得有可达 `assert`；写在读取之后的 guard clause、或断言之后的 late skip 都不再豁免 | 见 §11.2 的三模块实验（构造按 `test_meta_labeling.py` 的**决策 + 字面量**双重强制，故每条命中 3 个问题）：无 skip = **3** 问题（如旧）；真·skip 支配 = **0** 问题（豁免仍有效）；"skip + live pin" 旧规则 **0** 问题 → 新规则 **3** 问题（`measured-pin` 命中 `8.588e8`）。收紧后扫描真实 `LIVE_CACHE_TEST_FILES` 又抓出 1 处被旧豁免掩盖的 pin（`test_pairs.py::test_real_cached_pair_guard_verdict_matches_the_test` 的 `120.0`），已改为引用守卫自身的常量 `PAIRS_MIN/MAX_HALF_LIFE`、`PAIRS_MAX_ADF_PVALUE`，行为不变 |
| **R4** (LOW) | `tests/test_liquidity.py:558` 的 `(impacted - legacy)/legacy < 0.05  # BTC 1h is deep` 是一个**实盘前提**：它要求当次 BTC 1h 20 根窗口 ≥ ~4.0e7 USDT（4.294e7 过、3.435e7 挂） | 上界改为**由本次真正读到的窗口推导**：`bound = 2·impact_pct(participation(max_notional, volume)/100, 0.5)·max_notional/legacy_total`；"任何窗口都成立"的不变量（`impacted > legacy`、off-path 逐位相同、参与度恒等式）保留；并新增**确定性合成分支**测试 | 见 §11.3：当次窗口 `volume=850 315 320.33`、`max_notional=871.6695`、`share=0.0111 < bound=0.0117`；合成分支用 1e5/4e5 两个窗口证明 `impact_usdt == 2·side_pct·notional` 且窗口深 4 倍时 share 恰好减半 |
| **R5** (LOW) | `docs/core-algorithms/11-pairs-cointegration.md:111` 写 `tests/test_pairs.py`（25 条），实际 `--collect-only` 收 **28** 条（25 个同步 + 3 个 `async def`） | 改为 **28**（同文件另两处 per-file 计数 `tests/test_microstructure.py`（20）、`tests/test_regime.py`（14）已按 `--collect-only` 复核，未变） | `pytest tests/test_pairs.py --collect-only -q` 末行 **28 tests collected**；同文档 `test_microstructure.py` 20、`test_regime.py` 14 一致 |

### 11.1 R2 的"故意过期即失败"证明

把**文档副本**（仓库文件不动）的链表最后一行 `5f50771` 删掉、并把头部改回 `1a452ce`，指向该副本运行守卫：

```
E   pytest.fail: the evidence index's commit chain has 12 row(s) but
    `git rev-list --count f1f6a4c^..HEAD` = 13:
    missing=['5f50771'], table=[...12 rows...]
```

同理，只把链表补到 13 行而**不**写入 `5f50771`（首条断言之外的情形）会命中第二条断言（"never names the current HEAD `5f50771`"）。对未改动的仓库文档运行守卫为 **1 passed**（见 §6 全量）。

### 11.2 R3 的三模块实验（同一份 scanner）

| 合成模块 | 形态 | 修复前旧豁免 | 修复后 |
|---|---|---|---|
| 模块 1 | 无 `pytest.skip`，`assert auc <= 0.5866` | 3 个问题 | **3 个问题**（`no-decision` + `no-derived` + `measured-pin`） |
| 模块 2 | 首条语句即 `if not have_cache: pytest.skip(...)`，其后断言 | 0 个问题 | **0 个问题**（真·skip 支配，仍豁免） |
| 模块 3 | 先读缓存、`if volume <= 0: pytest.skip(...)`，再 `assert volume == 8.588e8` | **0 个问题**（旧豁免的漏洞） | **3 个问题**（`no-decision` + `no-derived` + `measured-pin`，点名 `8.588e8`） |

（模块按 `test_meta_labeling.py` 强制，即决策 + 字面量两条规则都跑，所以每条命中 3 个问题；裁决轮在只做字面量检查的模块上量到的对应数字是 2。）旧规则的复现口径就是被替换掉的那一行 `re.search(r"pytest\.skip\s*\(", body_src)`：模块 3 命中即 `continue`，3 个问题全部消失。

### 11.3 R4 的推导上界与合成分支

当次实测（`data/market/BTCUSDT/1h.parquet`，实盘在写，故为当次读数）：窗口 **850 315 320.33 USDT**、采样最大名义 **871.6695 USDT**、`legacy_total=75.6031`、`impacted_total=76.4423`、`share=0.011100`、推导上界 **0.011673** —— 旧断言 `share < 0.05` 之所以"看起来安全"，只是因为窗口恰好很深；改为推导上界后，窗口变浅时上界随之上升，测试不再依赖"BTC 1h 很深"这一实盘前提。合成分支（1e5 与 4e5）：`share` 0.7896 / 0.3948，恰为 2× 关系（平方根律），且 `impact_usdt == 2·side_pct/100·notional`。

**复现命令**：`python -m pytest tests/test_reaudit_fixes.py tests/test_measured_threshold_policy.py tests/test_liquidity.py tests/test_pairs.py -q -p no:cacheprovider`；`python -m pytest tests/ --collect-only -q -p no:cacheprovider`；`git log --oneline f1f6a4c^..HEAD`。


