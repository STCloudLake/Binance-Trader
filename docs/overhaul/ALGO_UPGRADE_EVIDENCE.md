# 算法升级证据索引（P1 GA / P2 ML / P3 波动率 / P4 新能力 / gap fixes / 复核修复）

**状态**: 证据已归档（本次实测，非引用） · **生成**: 2026-09-30 · **基线提交**: `09125bd`（本文件所在的 HEAD；`09125bd` 之后的工作树改动 = 闭环审计 D1 + LOW 1–6 修复，见 §10）
**工作树**: 本轮闭环修复仅由单一代理写入；改动文件为 `core/risk/manager.py`（D1/LOW-6：bar-key 折叠 + 非时间索引拒绝）、`scripts/check_data_integrity.py`（同一折叠）、`core/market_data/ohlcv_cache.py`（LOW-5 注释）、`core/ml/credibility.py`（LOW-3 实测值）、`core/risk/position_guard.py`（R1 实测值）、`docs/core-algorithms/13-volume-liquidity-costs.md`（LOW-2）、本文件（LOW-4）、`README.md`（LOW-1）、`tests/test_residual_closure.py`（新）+ `tests/test_final_audit_fixes.py` / `tests/test_gap_fixes.py` / `tests/test_reaudit_fixes.py` / `tests/test_cache_durability.py` / `tests/test_measured_threshold_policy.py`（随行为/注册表同步）。上一轮（`b49883b` 工作树）的复核修复见 §9，改动文件为 `core/risk/position_guard.py`、`core/ml/credibility.py`、`core/ml/meta.py`、`core/backtest/engine.py`（仅 preload 校验）、`core/ml/predictor.py`（仅同侧车校验）、`core/market_data/ohlcv_cache.py` + `core/market_data/provider.py`（仅惰性去重/flush）、`docs/core-algorithms/13-*.md`、本文件、新增 `tests/test_reaudit_fixes.py`。`data/models/` 仍为 15 个 `.pkl` / **0 个 `_meta.json`**。
**对应计划**: [`ALGO_UPGRADE_PLAN.md`](ALGO_UPGRADE_PLAN.md)（P1–P4 验收标准） · **审计快照**: [`REFACTOR_AUDIT.md`](REFACTOR_AUDIT.md)

## 0. 一页速览

| 阶段 | 提交 | 核心断言 | 本次实测/钉住数字 | 证据 | 复现命令 |
|---|---|---|---|---|---|
| P1 GA | `4791369` | 种群内每个基因独立仓位/分账，跨代适应度不再恒定 | 真实缓存 4 代 best `5.9411→5.9411→9.9411→9.9411`（+4.00）；每代 6/6、5/6、4/6、5/6 基因交易（旧实现 1/20）；默认参数 3 代同样 +4.00 且 82.2 s 跑完 | `tools/ga_real_data_curve.py`（新）· `tests/test_ga_credibility.py`（29 项） | `python tools/ga_real_data_curve.py` |
| P2 ML | `4791369` + `2751bbb` | 未过门（OOS AUC>0.55 且净成本期望>0）的模型强制 `enabled:false` | ETH 1h：legacy 准确率 0.6597 vs 多数类 0.7773（−11.76pp）、AUC 0.5621；P2 门 AUC 0.5342、净期望 −0.1822%、t=−2.28 → 拒绝 | `scripts/ml_credibility_measure.py` · `tests/test_ml_credibility.py`（53）· `tests/test_engine_ml_gate.py`（7） | `python scripts/ml_credibility_measure.py --symbols ETHUSDT --intervals 1h` |
| P3 波动率 | `a9e549e` + `2751bbb` | 方向不可预测、条件波动率可预测；开关默认关闭 | 方向 AUC 0.52/0.53（本次独立复现 0.5207/0.5342）；BTC 1h 未裁剪 EWMA 5.31%/bar vs 裁剪 0.52%/bar（10.12×） | `tests/test_volatility_targeting.py`（21 项）· `tests/test_gap_fixes.py`（12） | `python -m pytest tests/test_volatility_targeting.py tests/test_gap_fixes.py -q` |
| P4 能力 | `a9e549e` | 四个新能力有独立参照、无前视、默认惰性 | HMM 三 regime 合成序列 ≥95% 准确率；ADF 模拟零分布复现 MacKinnon 临界值；真实 1h 主流币：配对与 meta-label 全部拒绝（有效结论） | `tests/test_meta_labeling.py`（24，正被并行修改）·`tests/test_pairs.py`（28，正被并行修改）·`tests/test_regime.py`（14）·`tests/test_microstructure.py`（19） | `python -m pytest tests/test_meta_labeling.py tests/test_pairs.py tests/test_regime.py tests/test_microstructure.py -q` |
| gap fixes | `2751bbb` | 实盘波动率定仓接线、成本语义统一、单一入场求值器 | 开关关闭时两侧均 240.0 USDT（balance 10 000）；BTC 成交 50 012.5000 / 往返 0.25005%；四个 AND/OR 基因入场数一致 40/0/0/40；sklearn 警告 1 602 → 0 | `tests/test_gap_fixes.py`（12） | `python -m pytest tests/test_gap_fixes.py -q` |
| 发布审计 | `2751bbb` | 路由/测试/编译/数据完整性 | 路由 118→118，ADDED 0 / REMOVED 0；全量测试复跑 919 passed / 1 skipped / 0 failed；`compileall` 退出 0；缓存 `25/29` 文件带 gap | `scripts/regen_route_baseline.py`（新）· `scripts/check_data_integrity.py` | 见 §6 |
| **最终独立审计**（历史快照） | **`fe11ccf`→`b49883b`** | 审计的每条 H/M 缺陷已修或已量化上报 | 该次实测：全量测试 **1040 passed / 0 failed**（227.3 s，退出码 0）；`compileall` 退出 0；缓存 **24/29** 文件带 gap、**twin 0/29**（`BTCUSDT/1h` 已实测去重，见 §6 与 §9 R3）；`data/models` 15 `.pkl` / 0 `_meta.json`；F1/F3/F4/F5 的 before→after 见 §8，本轮复核 1–7 的 before→after 见 §9 | **新** `tests/test_final_audit_fixes.py`（13 项）+ **新** `tests/test_reaudit_fixes.py` | `python -m pytest tests/test_final_audit_fixes.py tests/test_reaudit_fixes.py -q` |
| **闭环审计 + 本轮修复** | **`828e375`→`09125bd`+工作树** | 复核 7 项残留 + 闭环审计 D1/LOW 1–6 全部关闭（1 项 MEDIUM 行为修复、1 项 MEDIUM 运行态上报、6 项 LOW 过期数字） | 全量测试 **1067 passed / 0 failed**（218.91 s / 215.91 s，退出码 0，两次）；`compileall` 退出 0；缓存 **25/29** 文件带 gap（`09125bd` 原样：close 口径尾行把 `BTCUSDT/1h` 误报 GAP/missing 1）→ D1 修复后 **24/29**、`twin` **0/29**（`BTCUSDT/1h`：11 624 行 / 11 624 bar 键 / 0 twin 行，sha256_16 `51911e3d994ed273`，mtime 2026-09-30 16:01:03）；D1/LOW 的 before→after 见 §10 | **新** `tests/test_residual_closure.py`（12 项） | `python scripts/check_data_integrity.py` · `python scripts/check_data_integrity.py --symbols BTCUSDT --intervals 1h` |

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
| 微观结构特征 | 每个特征是单快照的纯函数并与手算订单簿一致；决策时间戳之后的成交在任何统计前被丢弃（注入巨量未来成交后特征逐位不变）；深度/成交数有硬上限；TTL 缓存有界；传输失败降级为 `None` | `tests/test_microstructure.py`（19） |

## 5. gap fixes（`2751bbb`）

`tests/test_gap_fixes.py`（12 项）逐条带 before/after：**①实盘波动率定仓接线**（`RiskManager.check_signal`→`PositionSizer`；开关关闭时 balance 10 000 两侧均 240.0 USDT，行为不变）；**②数据完整性**（见 §6，未裁剪波动率被拒绝）；**③`sim_cost_quote` 价差语义**（表内为全额价差、单边收 `spread_pct/2`；BTCUSDT 成交 50 012.5000、往返 0.25005%；ETHUSDT 50 015.0000、0.26006%）；**④单一入场求值器**（四个 AND/OR 基因入场 40/0/0/40、信号 bar 244/300/0/300，与旧内联 AND 循环逐 bar 相同）；**⑤sklearn 特征名警告 1 602 → 0**。`condition_logic` 的持久化/打分-发布一致性缺口见 §1.2。

## 6. 发布审计（本次实测）

| 项 | 命令 | 结果 |
|---|---|---|
| 路由基线 | `python scripts/regen_route_baseline.py` | 旧 118 → 新 118；**ADDED 0 / REMOVED 0**；`docs/overhaul/route-baseline.json` 与提交版本**逐字节相同**（`git diff` 为空）。P1–P4 未新增也未删除任何 `APIRoute`/`WebSocketRoute` |
| 全量测试 | `python -m pytest tests/ -q -p no:cacheprovider` | **闭环审计**（`09125bd`+工作树，当前）：**1067 passed / 0 failed**（218.91 s / 215.91 s，退出码 0），连续两次一致；`tests/test_residual_closure.py` 新增 12 项。此前各次为历史快照：最终独立审计（`fe11ccf`+工作树）两次全量运行 **1040 passed / 0 failed**（470 s，退出码 0）与 **1039 passed / 1 failed**（314 s）。唯一的失败 `tests/test_ml_credibility.py::test_feature_pipeline_cost_is_bounded` 是**负载/计算预算敏感**断言（`elapsed < 3.0 s`；单独运行为 `1 passed in 0.90s`），与修复文件无关。同一工作树的**首次**运行为 `1022 passed, 3 failed`（242 s），3 个失败**均不在修复文件内**——`tests/test_measured_threshold_policy.py`（2 项）与 `tests/test_meta_labeling.py::test_real_primary_rules_are_refused_by_the_meta_gate`——三者随后修好，复跑即 `41 passed`。基线（`2751bbb`）：`919 passed, 1 skipped, 0 failed`（375 s） |
| 编译 | `python -m compileall -q app core web db scripts tools` | 退出码 **0** |
| 数据完整性 | `python scripts/check_data_integrity.py` | **闭环审计实测：`09125bd` 原样 = 25/29**（close 口径尾行 `BTCUSDT 1h …06:00:00 → …07:59:59.999` 被按原始时间戳差值误判为 1.99997 h 缺口 / missing 1，见 §10 D1）→ **D1 修复后 = 24/29**，干净 5 个：`ADAUSDT/1h`、`BTCUSDT/1d`、`BTCUSDT/1h`、`USDCUSDT/5m`、`ZECUSDT/5m`；`twin` 列 **0/29**（无 NOTE 行）。`BTCUSDT/1h` 现为 **11 624 行 / 11 624 bar 键 / 0 twin 行**，文件 sha256_16 `51911e3d994ed273`、mtime `2026-09-30 16:01:03`。⚠️ **本行原写"`BTCUSDT/1h` 的 55 个 `twin` 已合并"是错的**（复核发现 3）：`check_data_integrity` 当时报 **1/29** 文件带 twin，实测该文件 **11 678 行 / 11 623 棵 bar / 55 棵重复 / 110 行 twin**。根因是 `flush_all` 只重写 **dirty** 键，永不追加的文件永远不会被去重。已在 §9 R3 修复并实测（当时 **11 623 行 / 0 twin**，sha256 `0543b03d1ed8c3ff` —— 该 sha 属于当时的文件内容，此后实盘又追加了 close 口径尾行） |
| 模型产物 | `Get-ChildItem data/models` | **15 个 `.pkl`、0 个 `_meta.json`** —— 修复前 `skip_ml_training` 会把 15 个未过门的模型全部加载；修复后全部被拒绝（见 §8 F4） |

`scripts/regen_route_baseline.py` 用**临时 DB + 临时 config 目录**构建应用（绝不打开生产库），枚举每个 `APIRoute` 的 method+path 与 `WebSocketRoute`，排序后按既有格式（indent=1、CRLF、无尾换行）写回；有 REMOVED 时退出 1。`--check` 只报告不落盘。

## 7. 尚未闭环（not yet closed）

1. **hybrid 引擎的 `condition_logic`**：在基线提交 `2751bbb` 上 `git show 2751bbb:core/backtest/signal_matrix.py | grep -c condition_logic` = **0**，即向量化混合引擎当时仍只做 OR，`condition_logic: and` 的冠军在该模式下会被按 OR 评估（标量内核 `StrategyConfig.entry_sides` 已一致）。**该项在测量期间由并行改动关闭**：当前工作树 `core/backtest/signal_matrix.py`（mtime 2026-09-30 12:16）已在 `:303` 读取 `condition_logic` 并区分 AND/OR，新增 `tests/test_hybrid_condition_logic.py`（4 项，本次运行 `4 passed in 6.17s`）。该修复**尚未提交，且不在本次写权限内**，故列为"已由他人关闭、待提交复核"。
2. **缓存缺口文件**：基线（`2751bbb`）实测 **25/29**；最终独立审计（`fe11ccf`+工作树）实测 **24/29**；复核（`b49883b`+工作树）复测仍为 **24/29**；闭环审计（`09125bd` 原样）为 **25/29** —— 新增的一处不是真缺口，而是 close 口径尾行（§10 D1），D1 修复后回到 **24/29**，干净文件见 §6（`BTCUSDT/1h` 已转干净）。**twin（同一根 bar 两种时间戳口径各存一行）已从 1/29 修复为 0/29**（§9 R3）。修复口径：`python scripts/download_history.py --symbols <SYM> --intervals <tf> --start <first> --end <last> --merge`。
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

### 提交清单（截至 `09125bd`）

`git log --oneline` 自计划冻结起的**完整**提交列表（本文件所在 HEAD = `09125bd`；命令 `git log --oneline f1f6a4c^..HEAD`，11 行）：

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
| `09125bd` | **当前基线**：README 测试数字更正（693 → 1055 passed） |
| *工作树（未提交）* | **闭环审计 D1 + LOW 1–6 修复 + `tests/test_residual_closure.py`**，见 §10 |

## 9. 复核修复（`b49883b` 工作树）——7 项发现的 before/after

新证据文件：`tests/test_reaudit_fixes.py`。全部数字均为本轮实测；未提交（本节即其归档）。

| 编号 | 缺陷（修复前实测） | 修复 | 修复后实测 |
|---|---|---|---|
| **R1** (MED) | `PositionGuard.forecast_vol_pct` 自行用 provider 历史建 `DatetimeIndex` 帧但**不做** `_series_has_gap`，于是同一条拼接序列 `RiskManager → None` 而 `PositionGuard → 0.062009291811329616 %/bar`，且该值喂给**实盘移动止损距离** | `core/risk/position_guard.py`：预测前调用与 manager **同一个** `core.risk.manager._series_has_gap`（同一 `_VOL_MAX_GAP_BARS`=1.5 与同一 bar 长度表），命中即 `return None` 并 `logger.warning`；`_resolve_vol_pct` 随之回落到固定距离 | 同一拼接序列：guard 预测 **None**（拒绝，日志可见），manager 同样 **None**；无缺口序列两侧同为 **0.061993341684305286 %/bar**（逐位相同）。这两个数字由闭环审计重测（`tests/test_reaudit_fixes.py::_vol_frames()`：clean 0.061993341684305286 / unguarded spliced 0.062009291811329616）；本节原写的 `0.41803165815 %/bar` 用同一数字描述了**两条不同序列**且无法由任何现有 fixture 复现，已更正。开关关闭时移动距离与 pre-P3 逐位相同 |
| **R2** (MED) | `docs/core-algorithms/13` §3 的影响成本百分比**不可复现**：声称 1.0 BTC 往返 `+178.8965`（+2.36 %）/`+894.4824`（+11.82 %）"钉在同一 20 根窗口"，但同页引用的窗口是 750,661,812.73 USDT，代码在该窗口给出 **+1.7455**（+0.0021 %）/ **+8.7273**（+0.0105 %）；`0.8288` 的 impact 行也属于另一个窗口 | 文档：给出**可复现命令**，并把整节重测到**一个显式命名的窗口**上，附订单规模递增表说明"零售规模影响≈0、只有成为窗口的可见比例才生效"；同时记录旧数字各自隐含的窗口（≈756.78 M / ≈541.7 M / ≈729.2 M），说明它们为何不可比对 | 同一命令：窗口 **751,885,467.06 USDT**（ends 2026-09-30 06:00:00，price 83 043.14）。0.01 BTC **+0.0017**（+0.0002 %）；1.0 BTC **+1.7455**（+0.0021 %）；10 BTC **+55.1963**（+0.0066 %）；100 BTC **+1 745.4596**（+0.0210 %）（`k=0.5` 分别为 +0.0087/+8.7273/+275.9814/+8 727.2978）；`k=0` 与 legacy **逐位相同**（实测 `75.48621426 == 75.48621426`）；0.6 BTC 分解 impact **0.8281**、total 46.2918 |
| **R3** (MED) | §6 原写 `BTCUSDT/1h` 的 55 个 twin "已合并"是**假**：`check_data_integrity` 报 **1/29** 文件带 twin，实盘文件 **11 678 行 / 11 623 棵 bar / 55 棵重复 / 110 行 twin**。根因：`flush_all` 只重写 **dirty** 键 | `core/market_data/ohlcv_cache.py`：新增 `_frame_hash` + `_canonical_write`（内容哈希判"写是否会改变文件"）与 `OHLVCache.dedupe`；`flush_all` 在 dirty 键之后对**所有已加载键**做一次去重，仅在内容真的改变时落盘 | 实盘 `data/market/BTCUSDT/1h.parquet`：**11 678 → 11 623 行**，`twin_bars` **55 → 0**，`twin_rows` **110 → 0**，bar 键集合**完全相同**（11623），55 棵冲突 bar 全部保留**较新**的那一行；`check_data_integrity` 该行转 **ok**、`twin` 列 **0**、"1/29 files store a bar twice"提示消失。第二次 flush **不写盘**（`dedupe → False`），文件 **字节完全相同**（sha256 `0543b03d1ed8c3ff`）。**闭环审计复测（当前文件）**：**11 624 行 / 11 624 bar 键 / 0 twin 行**（该 sha `0543b03d1ed8c3ff` 与行数属于当时的内容，此后实盘又追加了 §10 D1 的 close 口径尾行），sha256_16 `51911e3d994ed273`、mtime `2026-09-30 16:01:03` |
| **R4** (LOW) | `core/ml/credibility.py:246-259` 是 `Φ(mean/se)`、**不读**偏度/峰度，而 `:743-746` 与 `core/ml/meta.py:47` 及文档 §8 声称它读——AND 因此**恰好等价于 `t > 2`** | 选择**实现** Prado 的修正公式：`SE_adj = sd·√((1 − γ₃·SR + (γ₄−1)/4·SR²)/(n−1))`，`returns` 作为可选参数（缺省/样本不足/矩非有限时退化为正态近似）；`net_trade_stats` 与 `_signed_net_stats` 把净收益序列传入；docstring/doc 改为**真**陈述 | 正态样本 `t=2.0000` → PSR **0.9773**（与旧式 0.9772 一致，逐位不变）；厚左尾 10 个 −40σ 异常值、`t=2.000000` → **0.9480 < 0.95 → 门拒绝**（同序列旧式报 0.9772）；同形状 `t=3` → **0.9864 → 放行**。`tests/test_ml_credibility.py::test_probabilistic_sharpe_matches_the_normal_approximation` 的 6 条断言全部仍然通过（`returns=None` 走同一公式） |
| **R5** (LOW) | `core/backtest/engine.py:1778` 的 `if stored and …` 与 `core/ml/predictor.py:672` 接受**缺 `feature_schema_hash`** 的侧车（审计 e2e 因此加载了 7 个产物中的 2 个） | 两处均改为**要求存在**：engine 返回 `False, "feature schema hash missing: …"`；predictor 抛 `FeatureContractError("… carries no feature schema hash …")`（与既有的 mismatch 异常同类） | 缺 hash 的侧车：engine **拒绝**并给出含 `feature schema hash missing` 的理由，predictor **抛异常**；完整侧车（`gate.allowed` + 契约名 + 正确 hash）**照常加载**；本仓库 `data/models` 15 个无侧车 `.pkl` 仍全部拒绝 |
| **R6** (LOW) | 本文件头仍写基线 `fe11ccf`、提交列表缺 `b49883b`、结尾写"最终审计（未提交）" | 头部改钉 `b49883b` 并列出工作树为本轮修复；提交清单补齐到 `b49883b`（含 `b49883b` 一行）；`data/models` 现状、全量测试数字、twin 行全部按本轮实测重写 | 当时的本文件（`git rev-parse HEAD` = `b49883b98a33354e83b0362fbd1c6eaca97a2ed8`）。**闭环审计再钉**：`git rev-parse HEAD` = `09125bde7ebf1df339b07a32f4b8ff4ac33383b3`，提交清单已补齐 `828e375`+`09125bd`（见上方"提交清单"表，`git log --oneline f1f6a4c^..HEAD` 共 11 行） |
| **R7** (LOW) | `core/ml/credibility.py:939` 的 `__all__` 导出未定义的 `signed_score`，`from core.ml.credibility import *` 抛 `AttributeError` | 移除该条目（`signed_score` 属于 `core.ml.calibration`，无生产调用者从 credibility 导入它——已 grep 全仓库确认），并加注释说明 | 星号导入**成功**；`[n for n in credibility.__all__ if not hasattr(credibility, n)] == []`；`tests/test_reaudit_fixes.py::test_star_import_of_credibility_resolves` 常驻钉住 |

**默认配置不变**：波动率目标化关闭、`risk.liquidity` 关闭、ML 关闭、P4 全部开关关闭时为逐位不变；vol targeting 打开但序列有缺口时，止损回落到文档化的固定规则。

## 10. 闭环审计（`09125bd` + 工作树）——D1 与 LOW 1–6

新证据文件：`tests/test_residual_closure.py`（12 项）。全部数字均为本轮实测；命令与观测值逐条给出。

| 编号 | 缺陷（修复前实测） | 修复 | 修复后实测 |
|---|---|---|---|
| **D1** (MED) | 拼接口径闸门在实盘缓存上**误报**：`data/market/BTCUSDT/1h.parquet` 为 11 623 个 bar-open 时间戳 + **1 个** `07:59:59.999` close 口径尾行（共 11 624 行），原始时间戳相邻差 `06:00:00 → 07:59:59.999` = 1.99997 h > `_VOL_MAX_GAP_BARS`=1.5，于是**连续的**序列被判为缺口：`_series_has_gap=True`、`RiskManager`/`PositionGuard` 预测 **None**、`check_data_integrity` 报 `BTCUSDT/1h` **GAP / missing 1**、全仓 **25/29** 带缺口。按 bar 键折叠后 max step 1.0 h、0 缺口 | `core/risk/manager.py::_series_has_gap` 与 `scripts/check_data_integrity.py::gap_report` 在**差分之前**把时间戳折叠到声明的 bar 键（复用 `core.market_data.ohlcv_cache.bar_keys`，即缓存写路径与 `download_history.py` 同一条区间规则） | 同形帧（300 open + 1 close）：`_series_has_gap` **False**；`RiskManager` = `PositionGuard` = **0.07684716805517097 %/bar**（逐位相同）；`gap_report` missing **0** / gaps **0** / flagged **False**。实盘文件：`--symbols BTCUSDT --intervals 1h` 由 **GAP / missing 1** 变为 **ok / missing 0**；全量 **25/29 → 24/29**。真缺口仍拒绝：少 1 根 bar → `_series_has_gap=True`、预测 None、`gap_report` missing 1 / gaps 1；100-bar 拼接 → 仍 **True**（最大缺口 100.0 h、missing 99） |
| **D2** (MED, 运行态) | 运行中的 PID **30628** 启动于 **15:14:35**，而 `828e375`（提交于 **16:06:45**）修改的 6 个 `core/*` 文件写盘于 **15:36:53–15:50:38** —— 正在跑的不是被审计的代码 | 不重启进程（由 Lead 负责） | **需在本次编辑后重启应用**，否则 D1 等修复不会生效；本节所有实测均来自源码/脚本，不依赖运行中的进程 |
| **L1** (LOW) | `README.md` 的 790/800/803/812/889 行仍用 `693` / `8 failed, 658 passed, 1 skipped` | 改为本轮实测值；干净克隆两个口径分别实测并标注 | §10.1 前提说明整段重写（见 `README.md` §10.1） |
| **L2** (LOW) | `docs/core-algorithms/13-volume-liquidity-costs.md` 的回溯窗口归属错误：`+178.8965` 被写成"≈ 7.1×10⁵ USDT 窗口"，`0.8288` 被写成"属于 ≈ 729.2 M 窗口，距 750.66 M 有 3 %" | 用代码反解并逐项重测，删除猜测性归属 | `total_impact_usdt` 反解：`+178.8965` → **71 576.03 USDT**（7.16×10⁴，比原主张小一个数量级）；`+894.4824` → **2 863.04 USDT**；`0.8288` 恰在表内 750,661,812.73 USDT 窗口复现为 **0.828812**，而 729.2 M 窗口给 **0.840920** |
| **L3** (LOW) | `core/ml/credibility.py:279` docstring 写 corrected PSR = `0.937`（`:803`、`:856` 同） | 三处改为实测值 | `net_trade_stats(_fat_left_returns())`：`t_stat=2.0000000000000004`、corrected PSR **0.9479777894541446**（≈ `0.9480`）、正态近似 **0.9772498680518209**（≈ `0.9772`） |
| **L4** (LOW) | 本文件 `:17`/`:91`/`:93`/`:148`/`:150`/`:153` 与提交清单过期（缺 `828e375`、`09125bd`；测试数 1040；缺口数 24/29；twin 行/sha；`0.41803165815`） | 全部按当前 revision 重钉，历史数字标注为历史快照 | 见本文件 §0 表、§6、§7、§9 R1/R3/R6 与上方提交清单（`git log --oneline f1f6a4c^..HEAD` = **11** 行；全量 **1067 passed / 0 failed**，218.91 s / 215.91 s，两次） |
| **L5** (LOW) | `core/market_data/ohlcv_cache.py:277` 注释称实盘 1h 缓存有"54 个 1 毫秒邻居" | 该数字属 revision `0542e02` 的历史快照，按当前实测改写 | 当前文件：`pairs=11623`、**min delta = 3600.0 s**、`Δ ≤ 1 ms` 的相邻对数 = **0**、唯一非 1 h 差值为 **7199.999 s**（即 D1 的 2 h − 1 ms 口径差） |
| **L6** (LOW) | `_series_has_gap` 对非 datetime 索引吞掉异常并返回 **False**（"无缺口"），一旦调用方传入非时间索引即等于关掉闸门（审计 F1 的失效模式） | 不可读的索引一律**拒绝**（返回 True）；未知 interval 仍保持惰性（False） | `pd.RangeIndex(300)` / `Float64Index` / object 字符串 / tz 混合 → **True**；`"7h"` 未知 interval → 仍 **False**；空/少于 3 根的合法日期索引 → **False**；`PositionGuard` 收到 RangeIndex 帧 → 预测 **None** |

**复现命令（本轮，全部只读）**：`python scripts/check_data_integrity.py`；`python scripts/check_data_integrity.py --symbols BTCUSDT --intervals 1h`；`python -m pytest tests/ -q -p no:cacheprovider`；`python -m compileall -q app core web db scripts tools`；bar 键/相邻差与 `total_impact_usdt` 反解的 one-liner 见 §3 的命令块与本节各行的数字。


