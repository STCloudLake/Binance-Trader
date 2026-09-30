# 算法升级计划（GA 策略生成 + ML 预测）

**状态**: 执行中 · **依据**: 两份只读审计（GA 生成算法、ML 预测流水线）实测证据
**原则**: 先让"优化与评估"变得可信，再谈新算法。任何改动必须①有量化验收②有回归测试③有前后对照数字④可复现（同种子同结果）⑤由独立审计复核。

---

## 一、实测问题清单（必须逐条闭环）

### GA（策略生成）—— 目前基本无法优化
| 级别 | 位置 | 问题 | 实测 |
|---|---|---|---|
| 致命 | `core/backtest/engine.py:476-480` + `core/ga/fitness.py:263,431` | 批量评估每 chunk `max_positions=max(1,15//N)=1`，却按**全 chunk 所有基因**计仓位并 `break` | 20 基因：**1 个交易 407 次，19 个 0 次**；同基因单独评估 1158 次；三代 best fitness 恒为 −1.30（零进展）；共用 `balance` 污染盈亏 |
| 致命 | `core/ga/walkforward.py:240-244` | 未传 `val_end`，验证窗口 = `[tr_end, tr_end]` **单根 bar** | 生产日志 `validate=2025-11-01~2025-11-01`（报告称到 12-01），全部 24 个 wf 任务 |
| 高 | `core/ga/fitness.py:285-290,452-458` | `gross_loss==0` 时 PF 记 100 且 ×5 → 最高 500 分，正当项合计 <60 | 5 笔全胜 → **490.85**；200 笔/PF 2.0/净赚 1500 → **7.30** |
| 高 | `fitness.py:326-329,505-508` | 选择适应度无 Sharpe/回撤/暴露（写死 0），无基准 | 纯漂移合成数据 Sharpe **13.7**（beta 当 alpha） |
| 高 | `fitness.py:170-179` vs `evolver.py:269-270` | DSR 用年化 Sharpe 比每期 E[max]，`observation_periods=365` 写死；且从不门控/展示 | SR 1.2/N 1200 → 代码 **1.0029「显著」**，正确 **−0.6135** |
| 高 | `evolver.py:238-245` | 任何非空种群即发布**启用**冠军到 `strategies/` | 已发布 fitness **−42.3、0 交易**的冠军；用户看到"策略变多" |
| 高 | `fitness.py:32-33` vs `genome.py:398-402` | 评分时禁用 ML、发布时启用 | 融合分数：关 0.5000 / w=0.1 → 0.6250 / w=0.5 → 0.4167（线上入场集合不同） |
| 中 | `genome.py:407-411`；`evolver.py:374-405` | 按布尔位清洗 → 孤儿条件（生产日志 `'adx > 20' not defined`）；按位置交叉 → 基因丢失/重复；空条件列表抛未捕获异常 | 已复现 |
| 中 | `genome.py:246`；`indicators.py:147-151` | `ema_period`、`ml_weight/ml_threshold` **无效基因**；无可进化阈值；入场仅 OR（只能变松） | 5 vs 50 → 交易数同为 672 |
| 中 | `ga.py:290-300`、`cost_model.py:239-242`、`evolver.py:68` | 无 `random.seed`；未映射币种用**今日实时盘口**算成本；冠军无溯源；`resume` 未转发 | — |
| 中 | `fitness.py:584-602` | 多进程评估生产环境每代 8 chunk 全部异常 → 全种群 −999，无降级 | 生产日志 |
| 低 | `risk_params.yaml:17` vs `engine.py:982-987` | 实盘 2–4× 杠杆，GA 现金模型 → 实盘盈亏/回撤≈2× | — |

### ML（预测）—— 目前有害
| 级别 | 位置 | 问题 | 实测 |
|---|---|---|---|
| 致命 | `core/ml/predictor.py:228`、`features.py:261-273` | 标签=4 根收益符号±0.5%（丢弃"无变动"，BTC 1h 仅留 40.5%），但每根 bar 决策 | 准确率 0.4717/0.4339/0.4444/0.4589/0.4093 vs 多数类 0.6667/0.5767/0.5432/0.5797/0.6114 → **低于基线 10–20pp** |
| 致命 | `predictor.py:228` | 概率**反向校准**（预测 0.91 → 实际 0.22） | AUC 0.396–0.447（<0.5） |
| 致命 | `engine.py:553-558` | `should_retrain` 要求 `key not in ml_models` → 回测中模型**只训练一次**（注释声称轮换重训） | repo 内无 `del ml_models` |
| 致命 | `predictor.py:42,228` vs `engine.py:267-282` | 线上 40 特征/固定 fwd=4/th=0.005，回测 47 特征/按策略×周期 → 回测 ML 不能验证线上 | 部署 pkl `n_features_in_=40`，回测 X=47 |
| 高 | `engine.py:1371-1376`、`evaluation_kernel.py:130` | 符号反转：`direction>0` 当 `confidence`，`(conf−0.5)*2` 把看涨算看跌 | PatchTST conf 可 0.34 而 direction=+1 |
| 高 | `trainer.py:46-48`、`features.py:268` | 无 purge/embargo，标签重叠 75–95%，有效样本≈N/4，无唯一性权重 | grep 无 `purge/embargo/sample_weight` |
| 高 | `core/ml/**` | 目标函数**完全不含成本**（往返≈0.09%，标签阈值 0.5%） | grep 无 fee/cost/pnl |
| 中 | `features.py:212-244` | 40 个特征中 **10 个恒为常数**（`REQUIRED_INDICATORS` 只算 rsi/macd/bollinger/adx） | 常量列清单已列 |
| 中 | `features.py:285-349` | Triple Barrier 仅 PatchTST 用，固定 2%/2%/24h，超时类占 44.9% | — |
| 中 | `engine.py:1233,1077` | `ml_accuracy_pct` 有看涨偏差（中性带 23.8–34.3% 记为 0.5） | — |
| 中 | `tft_trainer.py:216-219`、`trainer.py:105,128` | 用被汇报的指标做选择；LGB/XGB 无 `eval_set` 早停；无概率校准；同一 holdout 反复使用 | — |
| 低 | `tft_trainer.py:109-119` vs `:280-284` | 训练用扩张窗口归一化、推理用全历史均值/方差（train/serve 偏移）；`confidence_threshold` 等死参数 | — |

**已做对、不要破坏**：250 根 warm-up + 收盘后交易时间轴；close_time 索引；用上一根已收盘 bar 成交；高周期 ffill 对齐 → 无同 bar 前视；真实成本入 PnL；回测/实盘共用信号核；复杂度惩罚/精英/移民/断点续跑/−999 保护。

---

## 二、阶段与验收标准

### P1 — GA 可信性（结构改动）
1. 批量评估：**每基因独立仓位槽位**（`mine = 同基因持仓数`，用 `continue` 而非 `break`）+ **每基因独立分账**（balance/equity 序列）。
2. Walk-forward：`evolve(symbols, tr_start, val_end, validation_start=val_start)`，并断言 `val_start < val_end` 且样本 ≥30 bar。
3. 适应度重构：PF 收缩（分母加一个合成平均亏损）、PF 上限、按 `min(1, trades/50)` 缩放；加入**每基因 Sharpe（DSR 去偏）×min(1,trades/30) − 最大回撤**，并**扣除买入持有基准收益**。
4. DSR：用「每期 Sharpe + 真实期数 T」计算，纳入历史试验次数；`DSR<=0` 禁止发布；前端展示（当前为死块）。
5. 发布门槛：`trades>=30 且 净盈亏>0 且 PF>1 且 DSR>0 且 验证>0`，否则写 `enabled: false`。
6. 评分/发布一致：发布 YAML 的 `ml_config.enabled` 必须与评分时一致（或统一置 0）。
7. 基因层：按**解码后的 indicators** 清洗条件；按**基因名**交叉；空条件/异常捕获；删除或修复无效基因（`ema_period`→`ema.fast/slow`）；加入**可进化阈值基因**；结构支持 AND/OR 与出场条件进化。
8. 可复现：任务载荷带 `seed` 并 `random.seed`/`np.random.seed`；未映射币种禁用实时盘口取价（改用配置默认并在结果中标注）；冠军 YAML 写 `provenance`（窗口/币种/周期/seed/fitness/DSR）。
9. 多进程失败降级：chunk 崩溃时以 `max_workers=1` 重试并记录，禁止全种群 −999。

**验收**：种群内**每个**基因都参与交易（不足 30 笔者明确标注而非 0 笔）；跨代 best fitness 连续 ≥3 代有改善（合成数据上可复现）；WF 日志显示 OOS ≥30 bar；DSR 数值与手算一致（同一算例）；发布门槛在"5 笔全胜"和"0 交易"两个反例上都拒绝；同 seed 两次运行结果一致。

### P2 — ML 可信性
1. **评估门**：purged K-fold + `embargo = 标签时长`、样本**唯一性权重**；仅报样本外指标；**OOS AUC>0.55 且净成本后期望>0** 才允许 `enabled: true`，否则强制关闭并在 UI/日志说明原因。
2. **成本感知**：目标/评估改为净成本期望（往返费+价差+滑点），与 `sim_cost_quote` 同源。
3. **符号与校准**：融合分数改为 `2·p_up − 1`（以基础率为中心）；概率做 isotonic 校准；PatchTST/TFT 的"confidence"改为真概率或改名为 score。
4. **特征奇偶**：线上/回测统一特征清单，加载时断言列一致；**删除或补算 10 个常量特征**。
5. **标签/决策一致**：决策只用有标签的样本（或用三分类保留"无变动"），消除"只学 40% 样本却每根 bar 决策"。
6. **重训门**：修掉 `key not in ml_models`，实现真正的轮换重训（含 `eval_set` 早停）。
7. **默认安全**：在上述门槛达标前，配置默认 `ml.enabled: false`（当前反校准模型对融合是负贡献）。

**验收**：跑一次完整训练+回测，产出 base rate / 多数类基线 / AUC / Brier / log-loss / 净成本期望的对照表；若 AUC≤0.55 则 ML 自动保持关闭（有日志与 UI 提示）；特征列数在线上与回测相等；同 seed 结果一致。

### P3 — 预测目标升级（先做，收益高、风险低）
1. **波动率目标**：GARCH(1,1)/EGARCH 或直接预测 log 已实现波动率；用于①**波动率目标化仓位**②**动态 barrier 宽度**③风险/熔断阈值④作为特征。
2. **Triple Barrier 与真实 SL/TP 对齐**：上下轨由 ATR/条件波动率缩放，时间屏障与 `max_hold` 一致；保留/合并"无变动"类。
3. 把波动率预测接入仓位与 barrier 的实际路径（`position_sizer`、`position_guard`、`executor`），并加回归测试。

### P4 — 新能力（结构性新增）
1. **Meta-labeling**：一级 = 现有规则/GA 信号决定"何时交易"，二级 = ML 预测"该信号能否盈利"，用于**过滤与仓位缩放**（不做方向）。
2. **协整/配对交易**（Tsay 第 8 章）：Engle-Granger/Johansen 检验 + **Kalman 动态对冲比**（第 7 章）+ z-score 入场/出场，作为新策略族接入同一信号核与风控。
3. **盘口/微观结构特征**（第 5 章）：order-flow imbalance、microprice、实现波动率、成交持续期（数据源已有 depth/trades）。
4. **Regime 门控**（第 4 章）：Markov 切换或 SETAR，按 regime 切模型/阈值。

### P5 — 审计
每个阶段结束后由**独立只读审计**复核：重建审计者的复现场景、给出前后数字对照、检查是否引入新缺陷；直到该阶段零发现。最终再跑一次发布就绪审计（测试/编译/路由/账目恒等式/文档一致性）后提交推送。

---

## 三、量化总验收（发布门槛）
1. GA：种群内无"0 交易"基因；跨代适应度有实质改善；WF OOS 有效；DSR 单位正确且门控生效；冠军发布有门槛且默认 `enabled:false`。
2. ML：有且仅有"OOS AUC>0.55 且净成本期望>0"的模型可上线；无校准反演；特征奇偶一致；报告含基线与成本。
3. 全量测试通过（当前 693 起的基线只增不减），`compileall` 退出 0，账目恒等式精确成立，路由基线可解释。
4. 文档（`docs/core-algorithms/06/07/08`）与代码一致；`08` 中无来源的 58.7% 表格必须标注"示意/非实测"或删除。
