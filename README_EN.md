# Binance Trader

An automated cryptocurrency trading system built with Python, featuring multi-strategy engine, ML prediction, AI decision-making, real-time risk control, and a web management dashboard.

## Architecture

```
binance_trader/
├── app/                    # Entry point, config, event bus
├── core/
│   ├── strategy/           # Strategy engine, indicators, YAML loader
│   ├── market_data/        # WebSocket + REST market data (Binance)
│   ├── ml/                 # XGBoost predictor, feature engineering, trainer
│   ├── risk/               # Circuit breaker, risk manager, trailing stop, PositionGuard
│   ├── executor/           # Order executor (simulation + live)
│   ├── ai/                 # DeepSeek AI controller (market, strategy, risk optimization)
│   └── news/               # News fetcher, sentiment analyzer
├── web/                    # FastAPI Web UI + Jinja2/HTMX templates
│   ├── templates/          # Page templates (Chinese/English bilingual)
│   └── i18n.py             # Internationalization (385 entries)
├── alerts/                 # Alert rule engine, WebSocket real-time push
├── db/                     # SQLite database, atomic balance operations
├── strategies/             # 5 YAML strategy configurations
├── config/                 # System config, risk params, alert rules
└── tests/                  # 13 test files
```

## Key Features

- **Event-driven architecture** — 18 event types, async pub/sub
- **5 built-in strategies** — EMA Breakout, RSI+MACD, Mean Reversion, Bollinger Squeeze, Scalp Momentum
- **ML prediction** — XGBoost binary classification, auto training/prediction
- **AI decision-making** — DeepSeek full-auto: market assessment, coin selection, strategy optimization, risk adjustment
- **3-layer risk control** — Circuit breaker + PositionGuard (trailing stop + emergency stop) + signal audit
- **Real-time alerts** — Rule engine + WebSocket push, 5 default configurable rules
- **Web UI** — HTMX reactive interface, Chinese/English bilingual, fully responsive

## Quick Start

### 1. Install Dependencies

```bash
cd binance_trader
pip install -r requirements.txt
```

> **Note**: TA-Lib requires system-level installation. Windows users: see [ta-lib-python](https://github.com/TA-Lib/ta-lib-python).

### 2. Configure API Keys

```bash
cp config/secrets.yaml.example config/secrets.yaml
```

Edit `config/secrets.yaml` with your keys:

```yaml
binance:
  api_key: "your_binance_api_key"
  api_secret: "your_binance_api_secret"
deepseek:
  api_key: "your_deepseek_api_key"
```

Or use environment variables:

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export DEEPSEEK_API_KEY="your_key"
```

### 3. Launch

```bash
# Simulation trading (default)
python -m app.main --mode sim

# Live trading
python -m app.main --mode live

# Custom port
python -m app.main --mode sim --port 8888
```

Open `http://127.0.0.1:8899` to access the dashboard.

## Trading Modes

| Mode | Description |
|---|---|
| `sim` | Paper trading with local SQLite records |
| `live` | Live trading via Binance API |
| `backtest` | Backtesting (in development) |

## AI Control Modes

| Mode | Description |
|---|---|
| `suggest` | AI suggests only, manual confirmation required |
| `semi_auto` | Signal weights auto-adjusted, actions need confirmation |
| `full_auto` | Fully autonomous: AI controls weights, risk params, breaker recovery |

## Risk Parameters

See `config/risk_params.yaml`:

| Parameter | Default | Description |
|---|---|---|
| max_daily_drawdown_pct | 7.5% | Max daily drawdown triggering circuit breaker |
| max_daily_loss_usdt | 600 | Max daily loss triggering circuit breaker |
| trailing_stop_distance_pct | 2.0% | Trailing stop distance from current price |
| emergency_stop_threshold_pct | -5.0% | Emergency stop unrealized PnL threshold |

## Tech Stack

- **Async runtime**: asyncio
- **Market data**: python-binance WebSocket + REST
- **ML**: scikit-learn + XGBoost
- **AI**: OpenAI SDK → DeepSeek
- **Web**: FastAPI + Jinja2 + HTMX + ECharts
- **Database**: SQLite + Parquet + aiosqlite
- **Strategies**: YAML declarative config, TA-Lib indicators

## License

MIT License
