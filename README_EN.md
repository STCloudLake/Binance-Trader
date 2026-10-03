# Binance Trader

**English** · [中文](README.md)

A research and measurement platform for crypto trading strategies on Binance spot, written in Python 3.12. The data pipeline caches public market data into local parquet files and checks them for gaps, the backtest engines model fees, spread and slippage, a genetic algorithm evolves strategies, ML and GA each pass their own credibility gate, and a FastAPI + ECharts web console puts market data, order entry, backtests and GA progress on one set of pages. `/manual` reads the repository's own markdown as an in-app handbook.

`VERSION` 2.0.1 · Python 3.12 (measured on 3.12.10) · Windows / Linux · listens on `127.0.0.1:8899` only. Every trading capability ships disabled, and the default `--mode sim` fills orders in a local simulation.

---

## Research warning

This is a research and measurement platform, and none of its strategies has been validated with real money. Six independent search directions failed to find a robust out-of-sample edge on 1.5 years of spot cache (2025-04-01 to 2026-10-01): ML gating, pairs cointegration, meta-labeling, the genetic algorithm, P7 market regime plus upstream orchestration, and P8 volatility targeting. None of them passed the gate it declared in advance.

The most direct set of numbers: the seven shipped champions return −4.07 / −2.46 / −1.59 / −4.91% raw across four non-overlapping windows at 1.2–2.8% annualised volatility, and they sit in cash almost the whole time. In every tested window, holding the basket beat the evolved strategies once exposure was matched.

Do not run this project with real money. What it has demonstrably earned is measurement, risk control and execution discipline — including the ability to refuse a candidate that looks good, such as fitness 41.82 / Sharpe 9.01 / 382 trades / 0.11% max drawdown.

The full measured report, with the source of every number: [`docs/research/FINDINGS.md`](docs/research/FINDINGS.md) — the itemised result of each search direction, how each one was rejected by its own gate, the limits of the evidence, and what would change the conclusion.

---

## 1. Environment constraints (read first)

Only Binance's public mirrors are reachable from this machine: `https://data-api.binance.vision` (REST market data) and `wss://data-stream.binance.vision` (live kline stream) work, `https://testnet.binance.vision` can be used for orders and account state, and `https://api.binance.com` hangs until it times out. Real-money trading cannot be completed here: `--mode live` actually reaches testnet, and the fee tier can only be set by hand.

All market data goes through `config.binance.market_data_host`, managed by `core/market_data/data_client.py::MarketDataClient` (15 s hard timeout, connection-pool reuse, raises `MarketDataError` on failure). Do not use a bare `AsyncClient.create()`: python-binance defaults to `api.binance.com`, and a trading client must be passed `testnet=config.binance_testnet` explicitly. The live stream carries one deliberate override: python-binance's `BinanceSocketManager` hard-codes an unreachable socket host, so `provider.py` overwrites `bsm._get_stream_url` after creating `bsm`; do not bypass that step when you add socket usage.

The web app binds to `127.0.0.1` only, and `app/main.py` has no `--host` option; to reach it from another machine, use SSH port forwarding or a reverse proxy with HTTPS rather than exposing `0.0.0.0` to the internet. The measured host-reachability table and the commands to reproduce it are in [`docs/operations.md`](docs/operations.md#1-环境约束必读).

---

## 2. What works today

### Data pipeline

`core/market_data/provider.py` pulls market data from `data-api.binance.vision` (REST) and `data-stream.binance.vision` (WS), writes it to `data/market/<SYM>/<tf>.parquet`, and then publishes `MARKET_KLINE`. `scripts/download_history.py` downloads by symbol / interval / date range, `--merge` backfills incrementally, and `--backfill` adds the `quote_volume` / `trade_count` columns to existing files. `scripts/check_data_integrity.py` reports the bar count, span and calendar gaps of every file and marks a `GAP` once the gap exceeds 1.5 × the bar length; a series with gaps is refused by the volatility path's splice guard, and `--strict` makes the script exit 1 when gaps are present.

### Backtest engines

Two engines share one evaluation kernel. legacy (`core/backtest/engine.py`) walks the timeline tick by tick and supports LightGBM / TFT / PatchTST plus partial-reduction conditions; hybrid (`core/backtest/engine_hybrid.py`) pre-builds a signal matrix with `SignalMatrixBuilder` and replays it, and supports neither ML nor `reduce_conditions`. `backtest.engine_mode: auto` picks hybrid when there are at least 3 strategies and no ML and no reduction conditions; a runtime exception logs a warning and falls back to legacy. The legacy, hybrid and signal-matrix paths report the same canonical message when data is missing. The cost model applies fee tiers, half-spread and slippage to the fill or exit price, and freezes the spread for every symbol a run uses when the run starts, so the trade loop performs no I/O.

![Backtest page](docs/images/backtest.png)

*Backtest page: engine selection and cost-model parameters, progress, results and run history.*

### Credibility machinery

The GA and ML release gates both consume the true trial count recorded by `core/ga/trial_counter.py`, which covers every evaluation from resumed runs, walk-forward and per-symbol arms; deflated Sharpe therefore uses the number of variants a round actually tried — pooled 32 against per_symbol 64, or 133 for P7-S4. The release-gate benchmark comes from `core/ga/benchmark.py`, and the shipped configuration is `ga.benchmark_mode: exposure_matched`: the same basket is held only while the strategy is in position and is then weighted by the margin fraction the strategy actually committed, which answers whether that risk exposure bought anything better than fully invested buy-and-hold. `core/ai/holdout.py` is a one-shot holdout counter keyed by `(holdout_id, window_start, window_end, timeframe)`; a second evaluation of the same window raises `HoldoutRefusal` by default.

### Genetic search

The GA in `core/ga/evolver.py` supports checkpoint resume: a finished run keeps `<data-dir>/data/ga_checkpoint.pkl` by default (`ga.keep_checkpoint: true`), and `resume: true` continues from generation g up to this job's `generations`. On resume, the larger of the checkpoint's `prior_trials` and this run's `trials_this_run` is folded into the search account, so deleting `data/ga_trials.json` does not flatter the DSR, and resuming against a different window raises `CheckpointWindowMismatchError`. The `timeframe_pool` field restricts the timeframe gene to a whitelist, by default `1m/5m/15m/1h/4h`, with the panel preselecting `15m/1h/4h` because 1m has 60× the bars of 1h. `symbol_mode: per_symbol` gives each coin its own population and its own champion, and the champion YAML lists under `symbols` only the coins it was evaluated on.

![GA panel](docs/images/ga-panel.png)

*GA panel: per-generation progress, evaluation count, checkpoint state and champion gate verdicts.*

### Causal regime seam

`core/strategy/regime_causal.py` accepts only causal HMM labels; in-sample labels are rejected by name with `InSampleRegimeLabelError`. `core/ai/orchestrator.py` answers, bar by bar along that seam, whether a strategy is currently allowed to be enabled; its inputs are the causal regime, causal EWMA volatility, market breadth and the strategy's own losing streak. Rules can only block, never allow, and `replay` guarantees that the same rules plus the same event stream produce a bit-identical decision chain. Thresholds ship disabled and must not be tuned on the evaluation window.

### Web console and manual

`web/` is FastAPI + Jinja2 + HTMX + Tailwind + ECharts. The route baseline is 121 entries (`python scripts/regen_route_baseline.py --check` reports 121 / 121 / added 0 / removed 0): 120 HTTP routes (GET 72 / POST 43 / DELETE 4 / PUT 1) plus one WebSocket `/ws/alerts`, and the authoritative list is [`docs/overhaul/route-baseline.json`](docs/overhaul/route-baseline.json). `/trade` is the spot trading page; `/market`, `/coin/{symbol}`, `/data` and `/audit` are market and screening pages; `/strategies`, `/backtest`, `/ai`, `/alerts`, `/settings`, `/db-manager` and `/users` are operator pages. There are three roles, `admin` / `trader` / `viewer`, and per-page permissions are in [`docs/operations.md`](docs/operations.md#4-路由清单121).

![Trade page](docs/images/dashboard.png)

*Trade page: klines, order book, order panel, account cards and positions on one screen.*

![Strategies page](docs/images/strategies.png)

*Strategies page: strategy list and CRUD, signal fusion weights, risk parameters and lifecycle events.*

`/manual` renders `docs/**/*.md` and the root README as an in-app document browser: directory tree and filter, in-page table of contents, breadcrumbs, previous and next, and relative links inside a document rewritten to manual routes — a target outside the manual renders as a dead link rather than a 404. Rendering uses markdown-it-py (`html=False`), math is handed to the KaTeX CDN, and when the CDN is unreachable the raw LaTeX source is shown instead.

![Manual](docs/images/manual.png)

*Manual: directory tree, breadcrumbs and in-page headings.*

![Manual math rendering](docs/images/manual-math.png)

*Math in the manual is rendered by KaTeX, falling back to the raw LaTeX source when the CDN is unreachable.*

---

## 3. Quick start

Python 3.12 is required (measured on 3.12.10). TA-Lib needs a system-level prebuilt library; install it first by following that library's documentation.

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

# Windows
copy config\secrets.yaml.example config\secrets.yaml
# Linux / macOS
cp config/secrets.yaml.example config/secrets.yaml
```

Configuration is deep-merged in the order `config/config.yaml` → `config/risk_params.yaml` → `config/secrets.yaml`, with environment variables taking the highest precedence. Every configuration key, the secret-handling conventions and the GA job fields are in [`docs/operations.md`](docs/operations.md).

Download data. The whole `data/` tree is gitignored, so a clean clone has no cached history:

```bash
python scripts/download_history.py --symbols BTCUSDT,ETHUSDT --intervals 1h,4h --start 2025-04-01
```

Strategy YAML is yours to supply as well: the repository tracks no strategy file, so on disk there is only the gitignored `strategies/ga_champion_*.yaml`. The schema is the pydantic fields of `core/strategy/loader.py::StrategyConfig`.

Start it and open <http://127.0.0.1:8899>:

```bash
python -m app.main --mode sim
```

On first start, if the `users` table is empty, an `admin` account is created and its random password is printed to stderr only; change it in `/settings` immediately. `--mode live` places real orders through the testnet client. `--mode backtest` is not a backtest; its orders are silently dropped. Real backtests run from the `/backtest` page or `POST /api/backtest/run`.

Run the tests:

```bash
python -m pytest tests/ -q -p no:cacheprovider     # 1528 passed / 0 failed
```

---

## 4. Project structure

```
app/       Process entry point, event bus, config loading (uvicorn and the start/stop order of each component)
core/      Trading kernel: market_data / strategy / risk / executor / backtest / ga / ml / ai / news
web/       FastAPI application: routes, Jinja templates, static assets, /manual rendering
db/        SQLite schema, migrations and the ledger's single write point atomic_adjust_balance()
scripts/   Data download, gap checks, ledger reconciliation, route baseline, GA worker and status queries
tools/     One-off experiments and measurement scripts (read-only, off the trading path)
tests/     pytest cases; pytest.ini sets testpaths=tests, asyncio_mode=strict, markers=slow
config/    config.yaml, risk_params.yaml, secrets.yaml.example
data/      Runtime data (gitignored): SQLite, kline cache, model artifacts, backtest results, GA jobs
docs/      Research conclusions, algorithm breakdowns, per-phase evidence, historical audits
```

---

## 5. Status and limitations

- **ML gating is a measured negative contribution.** The online models run at accuracy 0.41–0.47 (majority class 0.54–0.67) with OOS AUC 0.396–0.447, and they are miscalibrated in the wrong direction (predicting 0.91 where the outcome is 0.22). P7-S1 conditioning produced no risk-adjusted alpha (0/8 and 0/40 cells with out-of-sample `dsr > 0`); the P7-S3 orchestrator lowers drawdown, time in market and return at once, with a DSR of 0 in both arms; and P8 volatility targeting failed its gate as well. The related switches are off by default; the list is in [`docs/operations.md`](docs/operations.md#6-默认关闭的能力与实验开关).
- **Cost and leverage conventions are limited.** Sim slippage is a fixed `slippage_bps` plus a fixed half-spread per symbol, and does not vary with order size or book depth; the sim and backtest spread fallback constants are deliberately different (0.02 vs 0.03). Backtest and GA use a cash model, whereas `max_leverage: 4` in `risk_params.yaml` applies to live trading only, so their return and risk figures are not directly comparable. The liquidity-impact coefficient `impact_k` defaults to 0 and is labelled ILLUSTRATIVE, NOT CALIBRATED in the configuration.
- **The data cache is incomplete.** 20 of 52 measured parquet files contain gaps (15m, 1h, 1m and 5m for BTC / BNB / ETH / SOL / XRP), and three further symbols cover only some intervals: ENAUSDT has just 200 1h bars, MOVRUSDT just 500 5m bars, and VTHOUSDT has 200 / 500 bars of 1h and 5m. The live volatility path sees only the most recent 600 bars, and older holes need `scripts/check_data_integrity.py`.
- **The reachability of funding-rate data has never been verified**, so any cost or return inference that needs funding is still unsupported.
- **Token screening is heuristic.** It reads only public exchange market data, not contracts, holder distribution, mint authority or transfer tax, so it cannot detect honeypots or rug pulls; a clean score only means this pair's exchange market looks normal. Position sizing is fixed-fractional, not true Kelly, and `docs/core-algorithms/04-position-sizing-kelly.md` is a research description.
- **The security surface is not hardened.** There is no CSRF token (only a `samesite=lax` cookie), no HTTPS or reverse-proxy configuration and no operation audit log, and authorization checks are inlined in the handlers (`TODO(authz)`).
- `experimental/` is imported by no production module, and deleting it does not change trading behaviour. `docs/HANDOVER.md` and `docs/overhaul/REFACTOR_AUDIT.md` describe the state before the big refactor — read them as history and trust the code.

---

## 6. License and disclaimer

This repository is released under the **MIT** license (full text in [`LICENSE`](LICENSE), `Copyright (c) 2026 STCloudLake`): you may use, modify and redistribute it freely, including commercially, provided you keep the copyright notice and the full license text. The software comes with no warranty.

Dependencies and data sources: python-binance, FastAPI + uvicorn, aiosqlite, ECharts + HTMX + Tailwind CSS, TA-Lib / LightGBM / XGBoost / PyTorch, loguru, DeepSeek. Market and trading data come from Binance's public API (`data-api.binance.vision` / `testnet.binance.vision`). Third-party components keep their own licenses, independent of this repository's MIT license; the itemised list is in the dependency tables below.

Crypto trading carries the risk of losing your entire capital. The system runs simulated by default, and any decision to switch to real-money trading, along with its parameter settings and consequences, is the user's own responsibility. In the current environment `--mode live` only reaches Binance testnet, which is no guarantee of safety after any future configuration change.

### Third-party dependencies

The tables below list **direct dependencies** and measured versions only: version constraints for the Python packages are in [`requirements.txt`](requirements.txt), and page-asset versions are in `web/templates/`. Licenses come from the installed distributions' own metadata (`importlib.metadata`'s `License-Expression` / `License` / `Classifier`, falling back to the LICENSE text in that distribution's `dist-info` when the metadata is empty); front-end assets come from that version's own `package.json` on the CDN. Transitive dependencies are not listed here and carry their own licenses.

**Runtime dependencies**

| Dependency | Measured version | License |
|---|---|---|
| python-binance | 1.0.36 | MIT |
| pandas | 2.3.3 | BSD-3-Clause |
| numpy | 2.3.3 | BSD-3-Clause |
| pyarrow | 24.0.0 | Apache-2.0 |
| TA-Lib | 0.6.8 | BSD-2-Clause |
| scikit-learn | 1.7.2 | BSD-3-Clause |
| scipy | 1.16.2 | BSD-3-Clause |
| xgboost | 3.2.0 | Apache-2.0 |
| lightgbm | 4.6.0 | MIT |
| torch | 2.13.0.dev20260531+cu130 | BSD-3-Clause |
| fastapi | 0.136.3 | MIT |
| uvicorn[standard] | 0.47.0 | BSD-3-Clause |
| jinja2 | 3.1.6 | BSD-3-Clause |
| markdown-it-py | 4.0.0 | MIT |
| pydantic | 2.13.1 | MIT |
| pyyaml | 6.0.3 | MIT |
| aiohttp | 3.13.2 | Apache-2.0 |
| python-multipart | 0.0.29 | Apache-2.0 |
| aiosqlite | 0.22.1 | MIT |
| bcrypt | 5.0.0 | Apache-2.0 |
| PyJWT | 2.13.0 | MIT |
| loguru | 0.7.3 | MIT |
| openai | 2.32.0 | Apache-2.0 |
| httpx | 0.28.1 | BSD-3-Clause |

**Front-end assets (CDN)**

| Asset | Version in the template | License |
|---|---|---|
| Tailwind CSS (`cdn.tailwindcss.com` Play CDN) | 3.4.17 (not pinned by the template; taken from the CDN's current build) | MIT |
| HTMX (`base.html`) | 1.9.10 | BSD-2-Clause |
| Apache ECharts (`base.html`) | 5.5.0 | Apache-2.0 |
| KaTeX (`manual_doc.html`) | 0.16.11 | MIT |

**Development and testing**

| Dependency | Measured version | License |
|---|---|---|
| pytest | 9.0.3 | MIT |
| pytest-asyncio | 1.3.0 | Apache-2.0 |

Three further imports are optional probes behind `try/except`; they are not in `requirements.txt`, are not installed on this machine, and their licenses are `unconfirmed`: `arch` in `core/ml/volatility.py` (falls back to scipy's GARCH implementation when missing), `statsmodels` in `core/strategy/pairs.py` (used for cross-validation only; the production path uses numpy), and `psutil` in `scripts/ga_job_status.py` (falls back to CIM / tasklist when missing).

---

## 7. Documentation index

| Document | Contents |
|---|---|
| [`docs/research/FINDINGS.md`](docs/research/FINDINGS.md) | Research report: measured numbers for the six search directions, how each was rejected by its own gate, the limits of the evidence, and what would change the conclusion |
| [`docs/operations.md`](docs/operations.md) | Operations reference: CLI, configuration keys, route inventory, GA job fields, experiment switches, data and database operations, troubleshooting, testing notes, algorithm-layer details |
| [`docs/research/CORE_ALGORITHMS.md`](docs/research/CORE_ALGORITHMS.md) · [`docs/core-algorithms/`](docs/core-algorithms/) | Algorithm overview; the 16 subsystem breakdowns, starting with the doc-versus-code discrepancies collected in [`00-ERRATA.md`](docs/core-algorithms/00-ERRATA.md) |
| [`docs/overhaul/P6_VOLUME_EVIDENCE.md`](docs/overhaul/P6_VOLUME_EVIDENCE.md) · [`P7_REGIME_EVIDENCE.md`](docs/overhaul/P7_REGIME_EVIDENCE.md) · [`P8_BETA_HARVEST_EVIDENCE.md`](docs/overhaul/P8_BETA_HARVEST_EVIDENCE.md) | Per-phase measured evidence and conclusions for P6 volume, P7 regime conditioning and P8 volatility targeting |
| [`docs/overhaul/ALGO_UPGRADE_PLAN.md`](docs/overhaul/ALGO_UPGRADE_PLAN.md) · [`ALGO_UPGRADE_EVIDENCE.md`](docs/overhaul/ALGO_UPGRADE_EVIDENCE.md) | Acceptance criteria and per-phase measured evidence for the P1–P4 algorithm upgrades, plus the items left open |
| [`docs/overhaul/PLAN.md`](docs/overhaul/PLAN.md) · [`CHANGELOG.md`](docs/overhaul/CHANGELOG.md) | The S0–S7 long-game repair plan and the findings and dispositions of three audit rounds; detailed change history |
| [`docs/audit/`](docs/audit/) · [`docs/superpowers/`](docs/superpowers/) | Historical audit sub-reports with severity grading; design specifications and implementation plans |
| `/manual` (in-app) | The in-app rendering of the table above, covering `docs/**` and the root READMEs only |
