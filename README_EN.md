# Binance Trader

An automated cryptocurrency trading system built in Python, integrating multi-strategy engine, ML prediction, AI decision-making, three-tier risk control, and a real-time web management panel.

## Architecture

```
binance_trader/
├── app/                          # Entry point, config, event bus
│   ├── main.py                   # Component initialization, dependency injection
│   ├── config.py                 # Singleton config loader (YAML → Pydantic models)
│   └── event_bus.py              # Async event bus (17 event types, 10000 capacity)
├── core/
│   ├── strategy/                 # Strategy engine
│   │   ├── engine.py             # Strategy evaluation, signal fusion (indicator + ML + news)
│   │   ├── loader.py             # YAML strategy CRUD (Pydantic models)
│   │   └── indicators.py         # TA-Lib technical indicators (RSI, MACD, BB, EMA, ADX, etc.)
│   ├── market_data/              # Market data
│   │   ├── provider.py           # Binance WebSocket multi-stream + REST polling hybrid
│   │   └── ohlcv_cache.py        # Parquet + in-memory dual-layer OHLCV cache
│   ├── risk/                     # Risk control (3 tiers)
│   │   ├── manager.py            # 7-step signal approval pipeline
│   │   ├── circuit_breaker.py    # Drawdown / daily loss / consecutive loss trip
│   │   ├── position_guard.py     # 15s loop: trailing stop + emergency stop
│   │   └── position_sizer.py     # Position sizing, SL/TP calculation
│   ├── executor/
│   │   └── executor.py           # Order execution (sim memory + live Binance API, 3 retries)
│   ├── ml/                       # Machine learning
│   │   ├── predictor.py          # XGBoost binary classifier, real-time prediction
│   │   ├── trainer.py            # Model trainer (auto save/load)
│   │   └── features.py           # Feature engineering + label construction
│   ├── ai/                       # AI decision-making
│   │   ├── deepseek_ctl.py       # DeepSeek controller (4 background loops)
│   │   ├── prompts.py            # 5 AI prompt template groups
│   │   └── vibe_connector.py     # Vibe-Trading MCP integration
│   ├── auth/                     # Authentication & authorization
│   │   └── auth.py               # JWT + bcrypt + session + RBAC (3 roles)
│   └── news/                     # News analysis
│       ├── analyzer.py           # News fetching, sentiment, anomaly detection
│       ├── fetcher.py            # httpx async HTTP client (API + RSS)
│       └── source_manager.py     # News source config CRUD
├── web/
│   ├── server.py                 # FastAPI app factory (60+ routes, WebSocket, HTMX)
│   ├── i18n.py                   # Chinese/English translation (200+ entries)
│   └── templates/                # Jinja2 templates (9 pages + 9 partials)
├── alerts/
│   ├── manager.py                # Alert rule engine + DB persistence + WebSocket broadcast
│   └── rules.py                  # AlertRule dataclass, 5 default rules, JSON persistence
├── db/
│   └── database.py               # SQLite schema (13 tables), atomic balance operations
├── strategies/                   # Strategy YAML definitions (5 built-in)
├── config/
│   ├── config.yaml               # Master config (AI mode, signal weights, news, allocation)
│   ├── risk_params.yaml          # Risk parameters (drawdown, position, stop limits)
│   ├── alert_rules.json          # Alert rule definitions (JSON format)
│   └── secrets.yaml              # Sensitive credentials (gitignored)
├── data/                         # Runtime data (DB, models, cache — all gitignored)
└── tests/                        # 13 test files
```

## Core Capabilities

### Event-Driven Architecture

17 event types enable loosely-coupled inter-component communication:

```
[Binance WebSocket] → MARKET_KLINE → [StrategyEngine]
[REST Polling]      → MARKET_KLINE ↗     ↓ STRATEGY_SIGNAL
[MLPredictor]       → ML_PREDICTION →   [RiskManager]
[NewsAnalyzer]      → NEWS_UPDATE   →       ↓ ORDER_REQUEST
                                          [OrderExecutor]
```

**Full event flow**:
1. `MARKET_KLINE` arrives → `StrategyEngine` computes indicators → fuses ML/news signals → publishes `STRATEGY_SIGNAL`
2. `RiskManager` approves signal (7-step pipeline) → publishes `ORDER_REQUEST`
3. `OrderExecutor` executes order → publishes `POSITION_UPDATE` → `RiskManager` updates positions
4. Exit signal → `POSITION_EXIT`/`POSITION_REDUCE` → close/reduce position
5. Circuit breaker trip → `RISK_BREACH` → AI decision / preset action

### 5 Built-in Strategies

| Strategy | Indicators | Timeframes | Description |
|----------|------------|------------|-------------|
| EMA+Volume Breakout | EMA9/21/50, VOL SMA20, OBV | 15m, 1h | Trend breakout + volume confirmation |
| RSI+MACD Trend | RSI14, MACD, ADX14 | 1h, 4h | Trend following + momentum confirmation |
| Mean Reversion RSI+BB | RSI14, BB20, ATR14 | 5m, 15m | Oversold/overbought reversion |
| Bollinger Squeeze+Volume | BB20, KDJ, VOL SMA20 | 15m, 1h | Bollinger squeeze breakout |
| Scalp Momentum | EMA5/13, RSI7, MACD, Stochastic | 1m, 5m | Ultra-short-term momentum |

Strategies are defined declaratively via YAML files. Create/edit/delete/toggle in the Web UI.

### Signal Fusion

```
final_score = (indicator_signal × W_ind + ml_confidence × W_ml + news_sentiment × W_news) / total_weight
```

Weights are adjustable in real-time via Web UI. In `full_auto` mode, AI optimizes weights automatically.

### ML Prediction

- **Model**: XGBoost binary classifier (predicts `up`/`down` direction)
- **Features**: RSI, MACD_histogram, BB_width, volume_ratio, price_momentum_24h
- **Training**: Per-symbol models on 4h kline data, incremental training
- **Prediction**: Publishes `ML_PREDICTION` events on each entry signal evaluation

### AI Decision-Making (DeepSeek)

4 background loops, active only in `full_auto` mode:

| Task | Default Interval | Function |
|------|-----------------|----------|
| Market Assessment | 60 min | Analyze market conditions, BTC dominance |
| Coin Selection | 240 min | Recommend trading symbols, core/satellite allocation |
| Strategy Optimization | 1440 min | Analyze strategy performance, recommend parameter changes |
| Risk Adjustment | 1440 min | Evaluate risk exposure, adjust risk parameters |

Breaker recovery: AI evaluates every 5 minutes whether it's safe to resume trading.

### Three-Tier Risk Control

| Tier | Component | Check Frequency | Function |
|------|-----------|-----------------|----------|
| Tier 1 | CircuitBreaker | Every signal | Daily drawdown % / daily loss $ / consecutive losses |
| Tier 2 | RiskManager | Every signal | 7-step approval pipeline, signal dedup |
| Tier 3 | PositionGuard | Every 15 sec | Trailing stop updates + emergency force close |

### Alert Rule Engine

5 default rules with condition matching + cooldown:

| Rule | Event | Level | Cooldown |
|------|-------|-------|----------|
| Circuit Breaker | risk.breach | critical | 5 min |
| Emergency Stop | alert.trigger | critical | 1 min |
| AI Suggestion Pending | ai.suggestion | info | 5 min |
| News Emergency Fetch | news.alert | warning | 10 min |
| Signal Rejected | alert.trigger | warning | 5 min |

Real-time WebSocket push to frontend — no refresh required.

### Multi-User Authentication

- **admin**: Full access (trading, settings, user management, database)
- **trader**: Trading + settings (cannot manage users or DB)
- **viewer**: Read-only access (view dashboard and pages, cannot trade or modify settings)

Security features:
- bcrypt password hashing
- JWT (HS256) + session cookies
- HttpOnly + SameSite cookies
- API endpoint role verification
- JWT secret randomly generated at startup (only fingerprint hash logged)

## Quick Start

### 1. Install

```bash
cd binance_trader
pip install -r requirements.txt
```

> **Note**: TA-Lib requires system-level installation:
> - Windows: Download precompiled wheel from [ta-lib-python](https://github.com/TA-Lib/ta-lib-python)
> - macOS: `brew install ta-lib`
> - Linux: `apt-get install ta-lib`

### 2. Configure

```bash
cp config/secrets.yaml.example config/secrets.yaml
```

Edit `config/secrets.yaml`:

```yaml
binance:
  api_key: "your_binance_api_key"
  api_secret: "your_binance_api_secret"
deepseek:
  api_key: "your_deepseek_api_key"
```

Or via environment variables:

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export DEEPSEEK_API_KEY="your_key"
```

### 3. Start

```bash
# Simulated trading (default)
python -m app.main --mode sim --port 8899

# Live trading
python -m app.main --mode live --port 8899
```

Visit `http://127.0.0.1:8899`.
On first startup, an admin account is auto-created — the password is printed in the terminal.

## Run Modes

| `--mode` | Description |
|----------|-------------|
| `sim` (default) | Simulated trading, local SQLite records, initial balance 10000 USDT |
| `live` | Live trading, connects to Binance Futures/Spot |
| `backtest` | Backtesting mode (in development) |

## AI Control Modes

Set in `config/config.yaml` → `ai.mode` or via Web UI:

| Mode | Signal Weights | Risk Parameters | Order Execution | Breaker Recovery |
|------|---------------|-----------------|-----------------|------------------|
| `suggest` | Manual | Manual | Manual confirm | Preset action |
| `semi_auto` | AI-adjusted | Manual | Manual confirm | Preset action |
| `full_auto` | AI auto | AI auto | AI auto | AI-decided |

## Risk Parameters

Configure in `config/risk_params.yaml` or Web UI settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_daily_drawdown_pct` | 7.5% | Max daily drawdown before breaker trip |
| `max_daily_loss_usdt` | 600 | Max daily loss before breaker trip |
| `max_consecutive_losses` | 5 | Consecutive losses triggering breaker |
| `max_open_trades` | 15 | Max simultaneous open positions |
| `max_position_size_pct` | 10% | Max single position % of balance |
| `max_leverage` | 3 | Max leverage multiplier |
| `trailing_stop_distance_pct` | 2.0% | Trailing stop distance from current price |
| `emergency_stop_threshold_pct` | -5.0% | Emergency stop threshold (unrealized PnL%) |
| `circuit_breaker_action` | block_only | Semi-auto breaker action (block_only / tighten_stops / close_all / close_worst) |

## API Endpoints (60+)

### Trading
`POST /api/trade` `POST /api/trade/close/{symbol}` `GET /api/price/{symbol}` `GET /api/kline/{symbol}` `GET /api/trades`

### Strategy Management (trader+)
`GET/POST/PUT/DELETE /api/strategy/{name}` `POST /api/strategy/{name}/toggle` `POST /api/strategy/reload`

### AI Control (trader+)
`GET /api/ai-suggestions` `POST /api/ai-suggestions/{id}/approve|reject` `POST /api/ai-mode` `POST /api/consult`

### Alerts
`GET /api/alerts` `POST /api/alerts/{id}/ack` `POST /api/alerts/clear` `WS /ws/alerts`

### Settings (trader+)
`POST /api/settings/deepseek|binance|risk|ai-news`

### User Management (admin)
`GET/POST /api/users` `POST /api/users/{id}/toggle`

### DB Manager (admin)
`GET /api/db/table/{table}` `GET /api/db/backup|export` `POST /api/db/restore|optimize|cleanup`

### Dashboard
`GET /partials/stats|positions|trades|alerts|ai-suggestions`

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11+ |
| Async Runtime | asyncio |
| Market Data | python-binance (WebSocket multi-stream + REST) |
| Web Framework | FastAPI + Jinja2 + HTMX + ECharts |
| Database | SQLite (aiosqlite) + Parquet (pyarrow) |
| ML | scikit-learn + XGBoost |
| AI | OpenAI SDK → DeepSeek API |
| Technical Indicators | TA-Lib |
| Auth | bcrypt + PyJWT (HS256) |
| Serialization | Pydantic + YAML + JSON |

## Testing

```bash
cd tests
python -m pytest test_event_bus.py test_config.py test_database.py -v
```

13 test files covering core components: event bus, config loading, database operations, market data, strategy engine, risk management, order execution, news analysis, ML prediction, AI decisions, alert system, and integration tests.

## License

MIT License
