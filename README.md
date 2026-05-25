# Binance Trader

基于 Python 的自动化加密货币交易系统，集成多策略引擎、ML 预测、AI 决策、实时风控和 Web 管理面板。

## 架构概览

```
binance_trader/
├── app/                    # 应用入口、配置、事件总线
├── core/
│   ├── strategy/           # 策略引擎、指标计算、YAML 策略加载
│   ├── market_data/        # WebSocket + REST 行情数据 (Binance)
│   ├── ml/                 # XGBoost 预测器、特征工程、模型训练
│   ├── risk/               # 熔断器、风控管理器、移动止损、PositionGuard
│   ├── executor/           # 订单执行 (模拟 + 实盘)
│   ├── ai/                 # DeepSeek AI 控制器 (市场评估、策略优化、风控调整)
│   └── news/               # 新闻抓取、情感分析
├── web/                    # FastAPI Web UI + Jinja2/HTMX 模板
│   ├── templates/          # 页面模板 (中英文双语)
│   └── i18n.py             # 国际化翻译 (385 条)
├── alerts/                 # 预警规则引擎、WebSocket 实时推送
├── db/                     # SQLite 数据库、原子余额操作
├── strategies/             # 5 个 YAML 策略配置
├── config/                 # 系统配置、风控参数、告警规则
└── tests/                  # 13 个测试文件
```

## 核心能力

- **事件驱动架构** — 18 种事件类型，异步发布/订阅
- **5 个内置策略** — EMA 突破、RSI+MACD、均值回归、布林带挤压、超短线动量
- **ML 预测** — XGBoost 二元分类，自动训练/预测
- **AI 决策** — DeepSeek 全自动模式：市场评估、币种选择、策略优化、风控调整
- **三层风控** — 熔断器 + PositionGuard (移动止损 + 紧急止损) + 信号审核
- **实时预警** — 规则引擎 + WebSocket 推送，可配置 5 条默认规则
- **Web UI** — HTMX 无刷新交互，中英文双语，全页面自适应

## 快速开始

### 1. 安装依赖

```bash
cd binance_trader
pip install -r requirements.txt
```

> **注意**: TA-Lib 需要系统级安装。Windows 用户参见 [ta-lib-python](https://github.com/TA-Lib/ta-lib-python)。

### 2. 配置密钥

```bash
cp config/secrets.yaml.example config/secrets.yaml
```

编辑 `config/secrets.yaml`，填入你的 API 密钥：

```yaml
binance:
  api_key: "your_binance_api_key"
  api_secret: "your_binance_api_secret"
deepseek:
  api_key: "your_deepseek_api_key"
```

或者通过环境变量：

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export DEEPSEEK_API_KEY="your_key"
```

### 3. 启动

```bash
# 模拟交易 (默认)
python -m app.main --mode sim

# 实盘交易
python -m app.main --mode live

# 指定端口
python -m app.main --mode sim --port 8888
```

访问 `http://127.0.0.1:8899` 进入管理面板。

## 运行模式

| 模式 | 说明 |
|---|---|
| `sim` | 模拟交易，本地 SQLite 记录 |
| `live` | 实盘交易，连接 Binance API |
| `backtest` | 回测模式 (开发中) |

## AI 控制模式

| 模式 | 说明 |
|---|---|
| `suggest` | AI 仅建议，人工确认后执行 |
| `semi_auto` | 信号权重自动调整，操作需确认 |
| `full_auto` | 全自动：AI 决定信号权重、风控参数、熔断恢复 |

## 风控参数

参见 `config/risk_params.yaml`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| max_daily_drawdown_pct | 7.5% | 触发熔断的日最大回撤 |
| max_daily_loss_usdt | 600 | 触发熔断的日最大亏损 |
| trailing_stop_distance_pct | 2.0% | 移动止损距离 |
| emergency_stop_threshold_pct | -5.0% | 紧急止损触发阈值 |

## 技术栈

- **异步运行时**: asyncio
- **行情数据**: python-binance WebSocket + REST
- **ML**: scikit-learn + XGBoost
- **AI**: OpenAI SDK → DeepSeek
- **Web**: FastAPI + Jinja2 + HTMX + ECharts
- **数据**: SQLite + Parquet + aiosqlite
- **策略**: YAML 声明式配置，TA-Lib 指标

## 许可

MIT License
