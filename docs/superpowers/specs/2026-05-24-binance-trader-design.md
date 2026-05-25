# Binance Trader — 自动化交易系统设计文档

**日期**: 2026-05-24
**状态**: 已批准

---

## 1. 项目概述

基于 python-binance 库构建的自动化交易工具，支持现货和合约双模式交易，集成技术指标分析、ML 预测、新闻情绪分析和 DeepSeek AI 决策，通过 Web UI 进行可视化管理和监控。

### 1.1 核心需求

- 指标交易 + ML 预测融合（信号融合公式：指标 × 0.5 + ML × 0.3 + 新闻 × 0.2，权重由 AI 动态调整）
- 自动周期性获取币种相关新闻，异常波动时立即触发紧急抓取
- 策略 YAML 配置化，Web UI 可视化编辑（选择指标、调整权重、自定义公式）
- DeepSeek v4 Pro API 驱动策略学习与优化
- Vibe-Trading 集成：研究端（策略生成/回测/行为诊断）↔ 执行端（实盘交易）
- AI 决策三级安全控制（建议 / 半自动 / 全自动），硬风控不可绕过
- 核心+卫星仓位模型：AI 动态选择币种和分配资金

### 1.2 参考项目

| 项目 | 参考点 |
|------|--------|
| Freqtrade | Strategy 模式、pandas DataFrame 信号管线、Hyperopt 超参优化 |
| Jesse | 声明式策略语法、ML Pipeline、Monte Carlo 检验、信号无偏差保证 |
| Hummingbot | 模块化架构、做市/套利策略框架、事件驱动设计 |
| Vibe-Trading | 自然语言策略生成、Multi-Agent Swarm 辩论、Shadow Account 行为诊断 |

---

## 2. 整体架构

事件驱动模块化架构，9 个核心组件通过 asyncio EventBus 协作，单进程运行。

```
MarketDataProvider → StrategyEngine → MLPredictor → RiskManager → OrderExecutor
                        ↑                                  ↑
                   NewsAnalyzer                     DeepSeekController
                        ↑                                  ↑
                   [新闻源]                     [DeepSeek API v4 Pro]
                        
              AlertManager ←→ EventBus ←→ WebUI (FastAPI)
                                            ←→ VibeTradingConnector (MCP)
```

### 2.1 两条核心数据流

**高频交易信号流**（端到端 < 500ms）：
MarketData WebSocket 推送 → StrategyEngine 消费计算指标 → MLPredictor 调整置信度 → RiskManager 8步检查 → OrderExecutor 下单

**低频 AI 决策流**（定时触发）：
DeepSeekController 聚合绩效+行情+新闻+持仓 → DeepSeek API 分析 → 返回币种建议+策略参数+风控调整 → 人工/自动确认

### 2.2 技术选型

| 层 | 选择 | 原因 |
|----|------|------|
| 异步框架 | asyncio + python-binance AsyncClient | 原生异步，高性能 WebSocket |
| Web 框架 | FastAPI + Jinja2 + ECharts | 异步 · WebSocket 原生 · 高性能 |
| 前端 | HTMX + Alpine.js + Tailwind CSS | 轻量 · 无重型框架 · 暗色主题 |
| 数据存储 | SQLite + Parquet 文件 | 零运维 · 单用户 · 易备份 |
| ML 框架 | scikit-learn + XGBoost | 成熟稳定 · 表格数据优势 |
| 技术指标 | TA-Lib | 150+ 指标 · C底层 · 行业标准 |
| AI 引擎 | DeepSeek API v4 Pro | 策略优化 + 市场分析 + 新闻解释 |
| 集成 | vibe-trading-ai (MCP) | 研究端协作 |

---

## 3. 核心组件设计

### 3.1 StrategyEngine — 策略引擎

策略通过 YAML 声明式定义，支持四种类型：趋势跟踪、突破交易、网格交易、动量策略。

```yaml
# strategies/rsi_macd_trend.yaml
name: "RSI+MACD 突破策略"
mode: trend
timeframes: ["1h", "4h"]

indicators:
  rsi: { period: 14, source: close }
  macd: { fast: 12, slow: 26, signal: 9 }

entry_conditions:
  long:
    - "rsi < 35 AND macd.histogram > 0"
    - "close > bollinger.middle AND volume > sma(volume, 20) * 1.5"

exit_conditions:
  long:
    - "rsi > 65 OR close < bollinger.lower"

ml_config:
  enabled: true
  confidence_threshold: 0.6
  features: [rsi, macd_histogram, bollinger_width, volume_ratio, price_momentum_24h]
  weight: 0.4
```

**信号融合公式**：
```
final_score = indicator_signal * w_indicator (0.5)
            + ml_prediction   * w_ml        (0.3)
            + news_sentiment  * w_news      (0.2)
```
权重由 DeepSeek 根据市场状态动态调整。高波动期降低 w_indicator、提高 w_ml；重大新闻期提高 w_news。

**Web UI 策略编辑器**：可视化勾选指标、拖拽权重滑块、自定义指标公式、入场/出场条件编辑器、ML 配置面板。

### 3.2 MLPredictor — ML 预测器

三阶段 Pipeline（参考 Jesse 设计）：

**Phase 1 — 数据采集**：每个交易信号产生时记录特征快照 + 标签（盈利/亏损/收益率），保存到 `data/ml_training/{symbol}_{strategy}.parquet`

**Phase 2 — 模型训练**：定时（每日/每周）或手动触发训练。任务类型：二分类（涨/跌）、多分类（大涨/小涨/震荡/小跌/大跌）、回归（预期收益率）。自动生成训练报告（特征重要性、准确率、F1）。

**Phase 3 — 模型部署**：新模型自动替换旧模型，预测延迟 < 10ms。策略中通过 `ml_confidence` 引用预测结果。

### 3.3 NewsAnalyzer — 新闻分析器

**双触发机制**：

- **周期采集**（默认每30分钟）：获取核心+卫星币种列表 → 多源搜索新闻 → 去重排序 → DeepSeek 情绪分析 [-1.0, +1.0] → 发布 NewsUpdate 事件
- **异常触发**：价格5分钟波动 > X%、成交量突增 > 均量Y倍、技术指标突变时立即抓取 → DeepSeek 紧急分析 → 高优先级告警

**新闻源**：维护在 `news_sources` 表中，支持 API/ RSS/ Web 多种类型，Web UI 管理增删改，所有新闻持久化到 `news_articles` 表供后续 ML 训练。

### 3.4 DeepSeekController — AI 控制器

五大职责：

1. **币种筛选与轮动**：每4小时分析全市场，输出核心币种池推荐（含评分和理由）
2. **策略参数优化**：每日分析交易绩效，提出参数调整建议
3. **动态风控调参**：根据累计收益/市场波动/当前回撤调整风险偏好
4. **市场状态评估**：每小时判断牛市/熊市/震荡/高波动，决定策略类型和信号权重
5. **新闻深度解读**：对新闻做结构化分析（方向、影响程度、持续性、置信度）

**三级安全控制**：

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| 建议 | AI 提建议，需人工确认 | 初期/重大调整 |
| 半自动 | 风控参数自动生效，策略变更需确认 | 稳定运行期（推荐）|
| 全自动 | 所有决策自动执行，受硬风控约束 | 充分验证后 |

硬风控（回撤熔断、日亏损上限）在任何模式下不可绕过。

### 3.5 RiskManager — 风险管理器

**双层风控**：
- **硬风控**（AI 不可修改）：日内/周回撤熔断、日亏损上限、单笔最大仓位、最大杠杆、最小止损距离
- **软风控**（DeepSeek 可调整）：风险偏好等级、仓位百分比、止损宽度、止盈阶梯、杠杆选择、信号权重

**8 步风险检查链**：账户熔断 → 日亏损 → 敞口 → 仓位 → 杠杆 → 止损 → 同币种 → 通过/拒绝。每步 REJECT 记录原因并告警。

**核心+卫星资金分配**：核心仓位 60~80%（波段/趋势，3~5币种），卫星仓位 20~40%（日内/突破/网格，5~10币种）。

### 3.6 OrderExecutor — 订单执行器

- 订单类型自动路由（市价/限价/止损/追踪止损，现货/合约差异处理）
- 仓位定期对账（本地记录 vs 交易所实际）
- 指数退避重试（3次），API 限频自动等待
- 完整订单生命周期追踪（CREATED → SUBMITTED → PARTIALLY_FILLED → FILLED / CANCELLED）

### 3.7 AlertManager — 预警系统

三级预警：

| 级别 | 触发条件 | 示例 |
|------|----------|------|
| 紧急 | 价格闪崩、风控熔断、API 断开、订单异常 | BTC 5分钟跌超5% |
| 警告 | 接近风控上限、连续亏损、币种下架 | 日亏损达80%上限 |
| 信息 | AI 建议、订单成交、定时报告 | 日/周绩效摘要 |

用户可自定义预警规则（价格/波动率/指标/新闻关键词），通知方式：Web UI 弹窗 + 浏览器 Notification API + 后续扩展 Telegram/Webhook。

### 3.8 VibeTradingConnector — Vibe-Trading 集成

通过 Vibe-Trading MCP Server 实现双向通信：

1. **策略研发→实盘部署**：自然语言想法 → Vibe-Trading 生成+回测 → 导入为 YAML 策略 → 模拟盘→实盘
2. **实盘数据→行为诊断**：交易记录 CSV → Vibe-Trading Shadow Account 分析 → 行为偏差报告
3. **AI 协同决策**：不确定场景 → Multi-Agent Swarm 多方辩论 → 综合评审辅助决策

---

## 4. 数据模型

### 4.1 存储分层

| 层 | 技术 | 内容 |
|----|------|------|
| 结构化业务数据 | SQLite | 交易记录、订单、持仓、预警、AI 建议、风控事件、新闻源、新闻文章、ML 模型注册、系统配置 |
| 时序行情数据 | Parquet 文件 | K线 (OHLCV)、指标缓存、ML 训练特征、新闻历史存档 |
| 配置与策略 | YAML / JSON | 策略定义、系统配置、用户偏好 |

### 4.2 核心数据表

| 表 | 用途 | 关键字段 |
|----|------|----------|
| trades | 交易记录 | symbol, side, entry/exit_price, qty, pnl, strategy, timeframe, status |
| orders | 订单历史 | trade_id, order_type, price, qty, filled_qty, status, binance_order_id |
| positions | 持仓快照 | symbol, side, qty, entry/current_price, unrealized_pnl, stop_loss, take_profit, position_type |
| alerts | 预警日志 | level, type, message, symbol, acknowledged |
| ai_suggestions | AI 建议 | category, content, rationale, confidence, status |
| risk_events | 风控事件 | event_type, level, detail, triggered_by |
| news_sources | 新闻源配置 | name, type, endpoint, enabled, rate_limit, priority |
| news_articles | 新闻文章 | source_id, symbol, title, url, content_summary, sentiment_score, impact_level |
| ml_models | ML 模型注册 | symbol, strategy, model_type, file_path, accuracy, f1_score, deployed |
| system_config | 系统配置 | key, value, category |

### 4.3 配置合并优先级（低→高）

1. YAML 文件（默认值）
2. 数据库 system_config 表（Web UI 覆盖）
3. 环境变量（API Keys 等密钥）

---

## 5. 项目结构

```
binance_trader/
├── app/                    # 主应用包
│   ├── main.py             # 入口 · 组件启动
│   ├── config.py           # 配置合并管理
│   └── event_bus.py        # asyncio 事件总线
├── core/                   # 核心业务组件
│   ├── market_data/        # WebSocket + REST 数据
│   ├── strategy/           # 策略引擎 · YAML 加载
│   ├── ml/                 # ML 训练/预测
│   ├── news/               # 新闻抓取 · 情绪分析
│   ├── risk/               # 风控 · 熔断 · 仓位计算
│   ├── executor/           # 订单执行 · 对账
│   └── ai/                 # DeepSeek · Vibe-Trading
├── db/                     # 数据库层
├── web/                    # FastAPI · 模板 · 静态资源
├── alerts/                 # 预警规则引擎 · 通知
├── strategies/             # 策略 YAML 文件
├── config/                 # 全局配置 · 密钥
├── data/                   # SQLite DB · Parquet · 模型
└── tests/                  # 测试
```

### 5.1 运行模式

- `python -m app.main --mode sim` — 模拟盘
- `python -m app.main --mode live` — 实盘
- `python -m app.main --mode backtest --strategy <name>` — 回测

---

## 6. 依赖清单

| 类别 | 包 | 用途 |
|------|-----|------|
| 核心 | python-binance, asyncio | Binance API + WebSocket |
| 数据 | pandas, numpy, pyarrow | 数据处理 + Parquet |
| 指标 | TA-Lib | 150+ 技术指标 |
| ML | scikit-learn, xgboost | 训练 & 预测 |
| 数据库 | aiosqlite, SQLAlchemy | 异步 SQLite |
| Web | FastAPI, uvicorn, jinja2 | Web 服务器 |
| 前端 | htmx, alpinejs, echarts, tailwind | Web UI |
| AI | openai (DeepSeek 兼容) | API 调用 |
| 集成 | vibe-trading-ai, mcp | Vibe-Trading |
| 工具 | pydantic, pyyaml, loguru | 校验 · 配置 · 日志 |

---

## 7. 用户确认的需求细节

- 策略编辑器：可视化选择指标、拖拽权重、自定义指标公式
- 新闻数据源和条目数量可配置，持久化到数据库供 ML 训练
- 风控偏好由 AI 动态调整（累计盈利后渐进激进，触达止盈后重置保守）
- Vibe-Trading 协作：策略研发→实盘部署、行为诊断、协同决策
- 核心+卫星仓位模型：AI 持续评估并动态更新核心币种池
