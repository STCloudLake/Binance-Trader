# Binance Trader

基于 Python 的自动化加密货币交易系统，集成多策略引擎、ML 预测、AI 决策、三层风控和实时 Web 管理面板。

## 架构

```
binance_trader/
├── app/                          # 应用入口、配置、事件总线
│   ├── main.py                   # 入口：组件初始化、依赖注入、生命周期管理
│   ├── config.py                 # 单例配置加载器 (YAML → Pydantic 模型)
│   └── event_bus.py              # 异步事件总线 (17 种事件类型, 10000 容量)
├── core/
│   ├── strategy/                 # 策略引擎
│   │   ├── engine.py             # 策略求值、信号融合 (指标 + ML + 新闻)
│   │   ├── loader.py             # YAML 策略加载/保存/删除 (Pydantic 模型)
│   │   └── indicators.py         # TA-Lib 技术指标 (RSI, MACD, BB, EMA, ADX 等)
│   ├── market_data/              # 行情数据
│   │   ├── provider.py           # Binance WebSocket 多流 + REST 轮询混合
│   │   └── ohlcv_cache.py        # Parquet 文件 + 内存双层 OHLCV 缓存
│   ├── risk/                     # 风控系统 (三层)
│   │   ├── manager.py            # 7 步信号审核管线 (熔断→敞口→仓位→杠杆→止损→去重→上限)
│   │   ├── circuit_breaker.py    # 熔断器 (回撤/亏损/连续亏损, 去重告警)
│   │   ├── position_guard.py     # PositionGuard: 15s 循环 (移动止损 + 紧急止损)
│   │   └── position_sizer.py     # 仓位计算、止损/止盈价计算
│   ├── executor/
│   │   └── executor.py           # 订单执行 (模拟内存 + 实盘 Binance API, 3 次重试)
│   ├── ml/                       # 机器学习
│   │   ├── predictor.py          # ML 预测器 (XGBoost 二元分类, 实时预测发布)
│   │   ├── trainer.py            # 模型训练器 (XGBoost, 自动保存/加载)
│   │   └── features.py           # 特征工程 + 标签构造
│   ├── ai/                       # AI 决策
│   │   ├── deepseek_ctl.py       # DeepSeek 控制器 (4 个后台循环任务)
│   │   ├── prompts.py            # 5 组 AI 提示词模板
│   │   └── vibe_connector.py     # Vibe-Trading MCP 集成 (研究/影子分析/群体智能)
│   ├── auth/                     # 认证授权
│   │   └── auth.py               # JWT + bcrypt + 会话管理 + RBAC (3 角色)
│   └── news/                     # 新闻分析
│       ├── analyzer.py           # 新闻抓取、情感分析、异常监测 (价格/成交量)
│       ├── fetcher.py            # httpx 异步 HTTP 客户端 (API + RSS)
│       └── source_manager.py     # 新闻源配置管理 (DB CRUD)
├── web/
│   ├── server.py                 # FastAPI 应用工厂 (60+ 路由, WebSocket, HTMX partials)
│   ├── i18n.py                   # 中英文翻译字典 (200+ 条目)
│   └── templates/                # Jinja2 模板
│       ├── base.html             # 基础布局 + 导航栏
│       ├── dashboard.html        # 仪表盘 (K线图、持仓、交易、告警)
│       ├── strategies.html       # 策略编辑与管理
│       ├── ai_panel.html         # AI 控制面板
│       ├── alerts.html           # 预警中心
│       ├── settings.html         # 系统设置
│       ├── login.html            # 登录页
│       ├── users.html            # 用户管理 (管理员)
│       ├── db_manager.html       # 数据库管理 (管理员)
│       └── partials/             # HTMX 局部刷新组件 (9 个)
├── alerts/
│   ├── manager.py                # 预警管理器 (规则引擎 + DB 持久化 + WebSocket 广播)
│   └── rules.py                  # 预警规则定义 (dataclass, 5 条默认规则, JSON 持久化)
├── db/
│   └── database.py               # SQLite 表结构 (13 张表), 模拟余额原子操作
├── strategies/                   # 策略 YAML 定义
│   ├── ema_volume_breakout.yaml  # EMA + 成交量突破
│   ├── rsi_macd_trend.yaml       # RSI + MACD 趋势
│   ├── mean_reversion_rsi_bb.yaml# 均值回归 + RSI + 布林带
│   ├── bollinger_squeeze_vm.yaml # 布林带挤压 + 成交量动量
│   └── 超短线动量_scalp_momentum.yaml
├── config/
│   ├── config.yaml               # 主配置 (AI 模式、信号权重、新闻、资金分配)
│   ├── risk_params.yaml          # 风控参数 (回撤上限、持仓上限、止损参数)
│   ├── alert_rules.json          # 预警规则定义 (JSON 格式)
│   └── secrets.yaml              # 敏感凭证 (API 密钥, 已 gitignore)
├── data/                         # 运行时数据 (DB, 模型, 缓存, 均 gitignored)
└── tests/                        # 13 个测试文件
```

## 核心能力

### 事件驱动架构

系统通过 17 种事件类型实现组件间松耦合通信：

```
[Binance WebSocket] → MARKET_KLINE → [StrategyEngine]
[REST 轮询]        → MARKET_KLINE ↗     ↓ STRATEGY_SIGNAL
[MLPredictor]      → ML_PREDICTION →   [RiskManager]
[NewsAnalyzer]     → NEWS_UPDATE   →       ↓ ORDER_REQUEST
                                        [OrderExecutor]
```

**完整事件流**:
1. `MARKET_KLINE` 到达 → `StrategyEngine` 计算指标 → 融合 ML/新闻信号 → 发布 `STRATEGY_SIGNAL`
2. `RiskManager` 审核信号 (7 步管线) → 发布 `ORDER_REQUEST`
3. `OrderExecutor` 执行订单 → 发布 `POSITION_UPDATE` → `RiskManager` 更新持仓
4. 出场信号 → `POSITION_EXIT`/`POSITION_REDUCE` → 平仓/减仓
5. 熔断触发 → `RISK_BREACH` → AI 决策/预设动作

### 5 个内置策略

| 策略 | 指标 | 时间框架 | 描述 |
|------|------|----------|------|
| EMA+Volume Breakout | EMA9/21/50, VOL SMA20, OBV | 15m, 1h | 趋势突破 + 成交量确认 |
| RSI+MACD Trend | RSI14, MACD, ADX14 | 1h, 4h | 趋势跟踪 + 动量确认 |
| Mean Reversion RSI+BB | RSI14, BB20, ATR14 | 5m, 15m | 超买超卖回归 |
| Bollinger Squeeze+Volume | BB20, KDJ, VOL SMA20 | 15m, 1h | 布林带收缩爆发 |
| Scalp Momentum | EMA5/13, RSI7, MACD, Stochastic | 1m, 5m | 超短线动量交易 |

策略通过 YAML 文件声明式定义，可在 Web UI 中创建/编辑/删除/启用。

### 信号融合

```
final_score = (indicator_signal × W_ind + ml_confidence × W_ml + news_sentiment × W_news) / total_weight
```

权重可通过 Web UI 实时调整，`full_auto` 模式下 AI 自动优化。

### ML 预测

- **模型**: XGBoost 二元分类器 (`up`/`down` 方向预测)
- **特征**: RSI, MACD_histogram, BB_width, volume_ratio, price_momentum_24h
- **训练**: 每个币种独立模型，4h K线数据，自动增量训练
- **预测**: 每次入口信号评估时发布 `ML_PREDICTION` 事件

### AI 决策 (DeepSeek)

4 个后台循环任务，仅在 `full_auto` 模式下运行：

| 任务 | 默认间隔 | 功能 |
|------|----------|------|
| 市场评估 | 60 min | 分析当前市场状况、比特币主导地位 |
| 币种选择 | 240 min | 推荐交易币种、核心/卫星仓位分配 |
| 策略优化 | 1440 min | 分析策略表现、推荐参数调整 |
| 风控调整 | 1440 min | 评估风险敞口、调整风控参数 |

熔断恢复：AI 每 5 分钟评估是否安全恢复交易。

### 三层风控

| 层级 | 组件 | 检查频率 | 功能 |
|------|------|----------|------|
| 第 1 层 | CircuitBreaker | 每次信号 | 日回撤 % / 日亏损 $ / 连续亏损次数 |
| 第 2 层 | RiskManager | 每次信号 | 7 步审核管线，信号去重 |
| 第 3 层 | PositionGuard | 每 15 秒 | 移动止损更新 + 紧急强制平仓 |

### 预警规则引擎

5 条默认规则，支持条件匹配 + 冷却时间：

| 规则 | 事件 | 级别 | 冷却 |
|------|------|------|------|
| 熔断触发 | risk.breach | critical | 5 min |
| 紧急止损 | alert.trigger | critical | 1 min |
| AI 建议待审 | ai.suggestion | info | 5 min |
| 新闻紧急抓取 | news.alert | warning | 10 min |
| 信号拒绝 | alert.trigger | warning | 5 min |

WebSocket 实时推送至前端，无需刷新。

### 多用户认证

- **admin**: 完整访问权限 (交易、设置、用户管理、数据库)
- **trader**: 交易 + 设置 (不可管理用户和数据库)
- **viewer**: 只读访问 (仪表盘和页面查看，不可交易或修改设置)

安全特性:
- bcrypt 密码哈希
- JWT (HS256) + 会话 Cookie
- HttpOnly + SameSite Cookie
- API 端点角色校验
- JWT 密钥启动时随机生成 (仅日志记录指纹 hash)

## 快速开始

### 1. 安装

```bash
cd binance_trader
pip install -r requirements.txt
```

> **注意**: TA-Lib 需要系统级安装:
> - Windows: 下载预编译 wheel 从 [ta-lib-python](https://github.com/TA-Lib/ta-lib-python)
> - macOS: `brew install ta-lib`
> - Linux: `apt-get install ta-lib`

### 2. 配置

```bash
cp config/secrets.yaml.example config/secrets.yaml
```

编辑 `config/secrets.yaml`:

```yaml
binance:
  api_key: "your_binance_api_key"
  api_secret: "your_binance_api_secret"
deepseek:
  api_key: "your_deepseek_api_key"
```

也可通过环境变量:

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export DEEPSEEK_API_KEY="your_key"
```

### 3. 启动

```bash
# 模拟交易 (默认)
python -m app.main --mode sim --port 8899

# 实盘交易
python -m app.main --mode live --port 8899
```

访问 `http://127.0.0.1:8899`。
首次启动自动创建管理员账户，密码会打印在终端。

## 运行模式

| `--mode` | 说明 |
|----------|------|
| `sim` (默认) | 模拟交易，SQLite 本地记录，初始余额 10000 USDT |
| `live` | 实盘交易，连接 Binance 合约/现货 |
| `backtest` | 回测模式 (开发中) |

## AI 控制模式

在 `config/config.yaml` → `ai.mode` 或 Web UI 中设置：

| 模式 | 信号权重 | 风控参数 | 订单执行 | 熔断恢复 |
|------|----------|----------|----------|----------|
| `suggest` | 手动 | 手动 | 手动确认 | 预设动作 |
| `semi_auto` | AI 调整 | 手动 | 手动确认 | 预设动作 |
| `full_auto` | AI 自动 | AI 自动 | AI 自动 | AI 决定 |

## 风控参数

在 `config/risk_params.yaml` 或 Web UI 设置页中配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_daily_drawdown_pct` | 7.5% | 触发熔断的日最大回撤 |
| `max_daily_loss_usdt` | 600 | 触发熔断的日最大亏损 |
| `max_consecutive_losses` | 5 | 连续亏损触发熔断的次数 |
| `max_open_trades` | 15 | 最大同时持仓数 |
| `max_position_size_pct` | 10% | 单笔最大仓位占总本金百分比 |
| `max_leverage` | 3 | 最大杠杆倍数 |
| `trailing_stop_distance_pct` | 2.0% | 移动止损距当前价格百分比 |
| `emergency_stop_threshold_pct` | -5.0% | 紧急止损触发阈值 (浮动亏损) |
| `circuit_breaker_action` | block_only | 半自动模式熔断动作 (block_only / tighten_stops / close_all / close_worst) |

## API 端点 (60+)

### 交易
`POST /api/trade` `POST /api/trade/close/{symbol}` `GET /api/price/{symbol}` `GET /api/kline/{symbol}` `GET /api/trades`

### 策略管理 (trader+)
`GET/POST/PUT/DELETE /api/strategy/{name}` `POST /api/strategy/{name}/toggle` `POST /api/strategy/reload`

### AI 控制 (trader+)
`GET /api/ai-suggestions` `POST /api/ai-suggestions/{id}/approve|reject` `POST /api/ai-mode` `POST /api/consult`

### 预警
`GET /api/alerts` `POST /api/alerts/{id}/ack` `POST /api/alerts/clear` `WS /ws/alerts`

### 设置 (trader+)
`POST /api/settings/deepseek|binance|risk|ai-news`

### 用户管理 (admin)
`GET/POST /api/users` `POST /api/users/{id}/toggle`

### 数据库管理 (admin)
`GET /api/db/table/{table}` `GET /api/db/backup|export` `POST /api/db/restore|optimize|cleanup`

### 仪表盘
`GET /partials/stats|positions|trades|alerts|ai-suggestions`

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| 异步运行时 | asyncio |
| 行情数据 | python-binance (WebSocket 多流 + REST) |
| Web 框架 | FastAPI + Jinja2 + HTMX + ECharts |
| 数据库 | SQLite (aiosqlite) + Parquet (pyarrow) |
| ML | scikit-learn + XGBoost |
| AI | OpenAI SDK → DeepSeek API |
| 技术指标 | TA-Lib |
| 认证 | bcrypt + PyJWT (HS256) |
| 数据序列化 | Pydantic + YAML + JSON |

## 测试

```bash
cd tests
python -m pytest test_event_bus.py test_config.py test_database.py -v
```

13 个测试文件覆盖核心组件：事件总线、配置加载、数据库操作、行情数据、策略引擎、风控管理、订单执行、新闻分析、ML 预测、AI 决策、预警系统、集成测试。

## 许可

MIT License
