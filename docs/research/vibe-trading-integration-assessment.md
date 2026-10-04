# Vibe-Trading × Binance Trader — integration assessment

Read-only on both trees; nothing modified or run. Citations `path:line`; **[INFER]** = inferred, not observed.

## 1. What Vibe-Trading actually is

Entry points `pyproject.toml:93-95`, `package-dir={""="agent"}` (`:98`). `agent/cli/main.py:1458` → interactive REPL, else `cli/_legacy.py:1517` (parser `:5362`). **No `backtest` subcommand.** `serve` defaults 8000 (`agent/api_server.py:340-341`); `dev` **8899** (`agent/cli/main.py:1613`).

Loop is **hand-rolled, not LangGraph**: `AgentLoop` (`agent/src/agent/loop.py:908`), ReAct (`:909`), `while iteration < max_iterations` (`:1230`; default 50 `:924`), stream `:1416`, final turn text-only `:1395-1401`, termination `:2096-2152`, read-only tools batched over 8 threads `:2665-2720`. `langgraph` declared (`pyproject.toml:26`) but never imported **[INFER: vestigial]**. State in `~/.vibe-trading/{sessions,runs}` (`agent/src/config/paths.py:34-54`).

Tools auto-discovered by subclass walk (`agent/src/tools/__init__.py:34-73`), ~119 names, `is_readonly=True` default (`agent/src/agent/tools.py:32`), write roots whitelisted (`agent/src/tools/path_utils.py:172-204`). Skills are progressive-disclosure **documentation**; `load_skill` never executes, so the 15 `example_signal_engine.py` files are templates — the only executed strategy code is a run dir's `code/signal_engine.py` (`agent/backtest/runner.py:1249-1269`), AST-scrubbed against import-time effects (`:786-817`) and runtime exec/network/fs-write (`:324-379`). MCP is **both** client (`agent/src/tools/mcp.py:256-738`) and server (`agent/mcp_server.py:3127-3184`; port 8900; ~72 tools; no order tool). Swarm is a thread-based DAG (`agent/src/swarm/runtime.py:251-354, :866`; 4 workers `:266`), each worker its own ReAct loop (`agent/src/swarm/worker.py:3, :655`), per-agent YAML tool/skill whitelists (`presets/equity_research_team.yaml:5-29`).

Backtest is a subprocess over `config.json` + `code/signal_engine.py` (`agent/backtest/runner.py:1191`; `agent/src/tools/backtest_tool.py`), dispatch `runner.py:1480`, loop `agent/backtest/engines/base.py:879`. **Fills next-bar open**: signals shifted one bar per calendar (`base.py:328-331`), filled at the open (`:832-834, :1575-1584`). Costs per market: A-share 2.5 bp + ¥5 + 5 bp sell stamp + 10 bp slippage (`engines/china_a.py:34-38`); US equity **zero commission**/5 bp (`engines/global_equity.py:63-135`); crypto 2 bp maker/5 bp taker/5 bp slippage/1 bp per 8 h funding (`engines/crypto.py:57-64`). Capital, leverage, margin exist (`base.py:850-858, :1299-1322`); **liquidity, impact and borrow do not** — their module has zero production call sites (`agent/backtest/factor_costs.py:24-30`). Data from a 28-provider registry (`agent/backtest/loaders/registry.py:36-66`), incl. keyless native Binance spot (`loaders/binance_loader.py:23-29`), but bare `BTC-USDT` auto-routes to OKX first (`agent/src/market_data.py:49`; `registry.py:231`).

Statistics largely **decorative at product level**: `src/quantlib/crossvalidation.py` has zero call sites; BH-FDR (`src/quantlib/multipletesting.py:375`) and CSCV/PBO (`:445`) zero call sites; the only live deflation is `deflated_sharpe_ratio` on an **IC information ratio** with `n_trials=len(rows)`, current batch only, never persisted (`agent/src/factors/bench_runner.py:352-388`); `backtest/validation.py:216` "walk_forward" slices one fitted equity curve without refitting; no pre-registered threshold, no exposure-matched benchmark. It does have **output grounding**: ~6.5 k lines (`agent/src/agent/grounding/`) forcing numeric/identity claims to match untruncated tool results (`grounding/__init__.py:1-31`) — truthfulness, not alpha. Frontend React 19 on 5899 (`frontend/vite.config.ts:42`) proxying to `VITE_API_URL || http://127.0.0.1:8899` (`:25`); desktop Electron bundling pinned CPython 3.12.10 (`desktop/electron/scripts/build-backend.ps1:15-19`). The `alpha-191-in-2026` post is generated from measured JSON (`agent/scripts/w4a_patch_blog.py:1-16`), self-critical and deflated (`wiki/research-lab/posts/alpha-191-in-2026.html:262, :548-552, :588-594`), but its top-5 table is selected on the window it scores **[INFER]**.

## 2. Capability comparison

| Capability | Stronger | Proof |
|---|---|---|
| Data acquisition | **Binance Trader** | Gap-checked 8.14 M-bar cache (`core/market_data/ohlcv_cache.py:3, :63-70`; `scripts/check_data_integrity.py:78-100`) vs 28 generic providers (`registry.py:36-66`) |
| Cost realism | **Binance Trader** | 4 bp fee + half-spread/side + optional √-impact + live-quote spread (`core/backtest/cost_model.py:50, :369-422`); none in-engine theirs (`factor_costs.py:24-30`) |
| Fill realism | **Vibe-Trading** | 12 market engines, next-bar-open, funding/ticks (`engines/base.py:328-331`; `engines/__init__.py:1-30`); ours fills at current close (`core/backtest/engine.py:1541`) |
| Search / optimisation | **Binance Trader** | Resumable GA checkpoint (`core/ga/evolver.py:109-176, :1487-1615`) |
| Evaluation discipline | **Binance Trader, decisively** | Persistent trial ledger (`core/ga/trial_counter.py:31-77`; live `{"trials": 1570}`), 4-gate ML credibility (`core/ml/credibility.py:761-796`), one-shot holdout (`core/ai/holdout.py:145-173`), worktree bit-identity (`tests/test_experimental_switches.py:485`) |
| LLM / agent integration | **Vibe-Trading** | `agent/src/agent/loop.py:908-1424`; 119 tools; grounding gates |
| Reporting / artefacts | **Vibe-Trading** | `agent/backtest/run_card.py`; `agent/src/shadow_account/reporter.py` vs our 106-line `core/backtest/report.py` |
| UI | **Vibe-Trading** | React 19 SPA + Electron vs our single FastAPI page (`web/server.py:77-118`) |
| Extensibility | **Tie** | `@register` loaders + auto-discovered tools (`registry.py:69-75`; `tools/__init__.py:34-73`) vs our YAML strategies + importable API (`core/strategy/loader.py:56`; `core/backtest/engine.py:229`) |

## 3. Integration options

| # | Shape | Effort | Buys | Risks | Verdict |
|---|---|---|---|---|---|
| a | External tool, files only: `python -m backtest.runner <run_dir>` is LLM- and key-free (`runner.py:1191-1219`); the agent path is not (`agent/src/preflight.py:32-61`) | Hours | Second opinion, strong reports, zero coupling | Adds trials nobody counts | **Do**, suggestion generator only |
| b | Feed it our cache. **Pluggable**: `@register name="local"` reads CSV/Parquet/DuckDB from `~/.vibe-trading/data-bridge/config.yaml` (`loaders/local_loader.py:241-247, :48-49, :209-238`), failing closed rather than fetching (`registry.py:642-659`) | ~½ day | Their engines run on our exact bars | Config entry per symbol; writes under `~/.vibe-trading`; schema *should* load (index reset, unknown columns dropped — `local_loader.py:214-221, :158-206`) but untested **[INFER]** | **Do** — enables (a) and (d) |
| c | Adopt modules. Both MIT; both Python 3.12.10 (`pyproject.toml:5`; `setup.py:7`); overlap pandas/numpy/scipy/sklearn/fastapi/pydantic/httpx/jinja2 | Days | Reusable tested pieces | Licence hygiene; our stack adds langchain/langgraph/fastmcp | **Cherry-pick** `multipletesting.py`, `grounding/`, loader abstraction — not `factor_costs.py` (unwired) |
| d | Our gates in front of its output | ~1 day | The only configuration where its ideas are judged honestly | Its proposals must enter our trial ledger | **Highest value** |
| e | (extra) Mine `skills/` + alpha zoo as **hypothesis corpus**, not code: ~90 `SKILL.md` + ~349 factors | ~1 day | Large citable backlog for *our* harness | `qlib158/**` carries Apache-2.0 §4 duties (`NOTICE:6-8`); `wvma*` declare another commit and MIT (`qlib158/wvma5.py:1-2`) | **Worth naming** |

## 4. The honest read

No. An LLM research agent will **not** produce tradeable alpha here. Your evidence is decisive: six independent search lines over 8.14 M bars yielded zero units with out-of-sample `dsr > 0` (`docs/research/FINDINGS.md:19-35, :139-144`), and holding the basket won every tested window. An LLM adds no new edge source; it manufactures **the same hypotheses you already falsified, faster, and outside your trial ledger** — the exact failure your discipline prevents. Their scan agrees: 4 % of GTJA-191 alphas survive, framed as signal quality and explicitly not a profitability claim (`alpha-191-in-2026.html:266-271, :548-552`).

Realistic value, descending credibility: **(1) reporting/artefacts** — run cards, PDF/HTML shadow reports, `report_audit_tool` beat ours and are orthogonal to alpha; **(2) hypothesis generation with attribution** — a sourced corpus fed to our harness, where our gates judge; **(3) narrative grounding** — news/filings context, thin on our side (`core/news/analyzer.py`, 200 lines); **(4) the grounding package** as a truthfulness gate on LLM prose. Portfolio construction and execution plumbing are **not** value-adds: ours are wired end-to-end, theirs ship disabled (`README.md:13-17`).

## 5. Recommended first experiment (< 1 day)

**Goal:** test (b)+(d) mechanically on the LLM-free path. Cache root `data/market` (`app/config.py:997`; `core/market_data/universe.py:43`): 52 parquet, 274.6 MB, 2025-04-01 → 2026-10-01.

1. **Copy** (never move) one file to scratch outside both repos: `%TEMP%\vt-exp\BTCUSDT_1h.parquet`.
2. Write `~/.vibe-trading/data-bridge/config.yaml`, one source: `symbol: BTC-USDT`, `type: parquet`, `path:` that file (schema `loaders/local_loader.py:7-30`).
3. Create a run dir **inside** an allowed root (`agent/src/tools/path_utils.py:141-158`) with `config.json` (`source:"local"`, `codes:["BTC-USDT"]`, `interval:"1H"`, `engine:crypto`) and a trivial `code/signal_engine.py`.
4. From `E:\Codes\Vibe-Trading\agent`: `python -m backtest.runner <run_dir>`.
5. Feed the same file to our `DataFeeder` (`core/backtest/data_feeder.py:36`), run `BacktestEngine.run` (`core/backtest/engine.py:229`) on one fixed YAML strategy.

**Measure:** does `local` resolve or fail closed; row count/span equality across loaders; return gap between their next-bar-open and our current-close fill at 1 h and 1 d; their deflated output vs our exposure-matched benchmark on one window.

**Continue if** the loader ingests our parquet unchanged and the fill gap is small or explainable — (b) and (d) then become cheap. **Stop if** the loader rejects the schema or the gap makes comparison meaningless; only (e) survives, needing no integration.

**Must be installed:** nothing on our side. Vibe **cannot** run from its checked-out `.venv`: Python is right (3.12.10, `pyproject.toml:5`) but it holds `langchain 0.3.30`, `langgraph 0.2.76`, `pandas 3.0.3` against declared `>=1.3.9`, `>=1.2.5`, `<3.0.0` (`pyproject.toml:23-33`); `requirements-lock.txt:1593, 1618, 2429` pins 1.3.18 / 1.2.11 / 2.3.3 → needs a **fresh venv** **[INFER: stale venv predates 251b0943]**.

**Without API keys/network:** yes for step 4 — the runner loads config plus signal engine, resolves `local`, never calls an LLM; the LLM is only *critical* on the agent path (`agent/src/preflight.py:32-61`), and `local` sits in `_NO_NETWORK_FALLBACK_SOURCES`, so a missing file raises rather than fetches (`registry.py:642-659`). **Verified by reading control flow, not executing.**

## 6. Practical integration risks

- **Port 8899 collides.** Vibe's `dev` backend defaults there (`agent/cli/main.py:1613`), its Dockerfile exposes it (`Dockerfile:113, :120`), its frontend proxy targets it (`frontend/vite.config.ts:25`); ours defaults there too (`app/config.py:38`; `app/main.py:608-609`). I verified **nothing listens on 8899** now (only 127.0.0.1:3080). Always pass `--port` and set `VITE_API_URL`; a production `dist/` build hardcodes same-origin (`frontend/src/lib/api.ts:9`), so retargeting needs a rebuild. `serve`=8000, MCP=8900.
- **Database ownership: none.** Their state is files under `~/.vibe-trading/` (`agent/src/config/paths.py:34-54`); ours is `data/binance_trader.db` (SQLite+WAL, `app/config.py:996`). Keep it that way.
- **Licensing: MIT both, attribution still owed.** Copying `agent/src/factors/zoo/qlib158/**` carries **Apache-2.0 §4** duties (`NOTICE:6-8`; `qlib158/LICENSE.md:24-29`) — take NOTICE + LICENSE.md + per-file headers together. Fonts OFL-1.1. Copying `grounding/` or `multipletesting.py` requires a NOTICE line crediting HKUDS Vibe-Trading at `251b0943`.
- **Secrets.** `agent/.env` exists and is gitignored (`agent/.gitignore:9`); it holds a live-looking `DEEPSEEK_API_KEY` and `TUSHARE_TOKEN` (names read only). Never copy `agent/.env` or any `~/.vibe-trading` path into our repo. Their redaction is fail-closed (`agent/src/tools/redaction.py:330-548`) but does not cover the dotenv, and their code concedes unprefixed secrets survive (`redaction.py:117-121`).
- **Environments cannot be merged cheaply.** Both 3.12.10; ours has **no in-tree venv** (global `D:\Program Files\Python\Python312`, with `torch 2.13.0.dev+cu130`, TA-Lib 0.6.8). Their `.venv` lacks `torch`, TA-Lib, `xgboost`, `lightgbm`, `python-binance`, `loguru`, `aiosqlite`, `bcrypt`, `reportlab` — **no shared environment today**, and the union fights their `langchain<2`/`langgraph<1.3` caps. Keep two venvs, cross with **files**, as (a) and (b) do. Never let a foreign process unpickle `data/ga_checkpoint.pkl` (`core/ga/evolver.py:1487-1530`).

## Could not determine

Whether our parquet loads in their `local` loader (read as likely; not executed); whether Vibe-Trading runs at all at `251b0943` from a correct environment (nothing installed or run); its real LLM behaviour (cost, latency, tool-calling and grounding false-positive rates); `agent/runs/**` writers were not exhaustively audited for secret leakage, nor `.github/`/`.devcontainer/`; how many of the ~349 factors are algebraically distinct; which of the ~90 skills are worth anything — ranking them would itself consume trials.
