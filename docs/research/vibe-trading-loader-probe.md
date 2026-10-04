# Vibe-Trading `local` loader × Binance Trader cache — executed probe

**Question.** Can HKUDS/Vibe-Trading (`E:\Codes\Vibe-Trading`, HEAD `251b0943`, v0.1.16, MIT) ingest our
`data/market/<SYM>/<interval>.parquet` cache through its pluggable `local` data provider, and does it stay offline
while doing so? The predecessor assessment *inferred* it would load and **never executed it**
([vibe-trading-integration-assessment.md](vibe-trading-integration-assessment.md)). This document replaces that
inference with execution.

**Method.** Read-only on both trees. Vibe-Trading's own modules were imported and executed; no file in
`E:\Codes\Vibe-Trading` was edited, no package was installed into its `.venv`, its updater/server/agent loop were never
run, and `agent/.env` was never opened. Our parquet files were **copied** to `%TEMP%` before being handed to the loader.
The only Binance Trader change is this file.

**Environment used for execution.** Host interpreter `D:\Program Files\Python\Python312\python.exe` (CPython 3.12.10)
with `pandas 2.3.3`, `pyarrow 24.0.0`, `PyYAML 6.0.3`, `pydantic 2.13.1` — the exact pandas pin in
`requirements-lock.txt`. `duckdb` is *not* installed and is not needed (see §7). A throwaway venv was attempted first;
`pip install` stalled with zero packages written after ~30 minutes and was killed, so the host interpreter was used
instead. **No installation into Vibe-Trading's `.venv` and no install of any kind completed.** `PYTHONDONTWRITEBYTECODE=1`
plus `python -B` were set so importing its modules could not write `__pycache__`; `git status --porcelain` in
`E:\Codes\Vibe-Trading` is empty afterwards and no `.pyc` there is newer than the session.

---

## 1. Expected schema (read from the code, then confirmed by execution)

`agent/backtest/loaders/local_loader.py` — entry point `DataLoader.fetch` (`:271-322`), real work in `_fetch_one`
(`:324-382`) and `_normalize_columns` (`:158-206`).

| Requirement | Detail | Evidence |
|---|---|---|
| Config path | `Path.home() / ".vibe-trading" / "data-bridge" / "config.yaml"`. **No env var or parameter overrides it** — `_CONFIG_DIR`/`_CONFIG_PATH` are module constants (`:48-49`) and `_load_config` (`:151-155`) reads only that path. It is therefore redirected by overriding `USERPROFILE` (=`Path.home()` on Windows), not by configuration. | verified §2 |
| Config keys | top-level `sources:` list; per entry `symbol` (required, non-empty), `type` ∈ {`csv`,`parquet`,`duckdb`} (default `csv`, `:331-335`), `path` (`~` expanded with `expanduser`, **no env-var expansion**, `:359-364`), optional `columns: {date,open,high,low,close,volume}` name overrides (`:337-342`), optional `date_format` (`:344-347`). `duckdb` uses `db_path` + `query` instead of `path` (`:349-357`). | read |
| Default column names | `_DEFAULT_COLUMNS` (`:51-58`): `date, open, high, low, close, volume`. | read |
| Columns required | `open/high/low/close` must exist **after** renaming; a date column named by `col_map["date"]` must exist; `volume` is optional and back-filled with `0.0` (`:171-206`). | §4 |
| Index | `pd.read_parquet` result is `reset_index()`-ed **only if** it is a `DatetimeIndex` (`:216-221`); the date column is then taken from `col_map["date"]`, defaulting to the first column after the reset — which is the reset index itself, whatever it is named. Any other index type (e.g. `RangeIndex`) gets no such fallback. | §4 |
| Date parsing | `pd.to_datetime(..., utc=True)` then `.dt.tz_convert(None)` → **UTC-naive** `DatetimeIndex` named `trade_date`, `sort_index()`-ed, unparseable rows dropped (`:179-189`). | observed |
| Output | exactly `[open, high, low, close, volume]` in that order; all `float64`; `validate_ohlc` drops non-positive / non-bracketing bars (`:191-206`, `base.py:187-256`). | observed |
| Window | `start <= index` and `index < end_date + 1 day` (end day inclusive, `:369-376`). | observed |
| Interval coverage | decided **after** the window filter by median index spacing vs `pd.Timedelta(_RESAMPLE_RULES[interval])` (`:98-124`): `target < source` → warn + return source bars unchanged; `target == source` → unchanged; `target > source` → `df.resample(rule).agg(OHLCV)` with `volume=sum`. Unknown interval token → warn + return source bars (`:101-107`). Accepted tokens: `1m,5m,15m,30m,1H,1h,4H,4h,1D,1d` (`:63-74`). | §5 |
| Symbol → file | **No templating.** `symbol` is an opaque, exact, case-sensitive dict key (`:261-269`, lookup `:298`); each (symbol, interval) pair needs its own literal `path`. `fetch` strips a single leading `local:` (`:297`). | §6 |
| Fail-closed | `local` ∈ `_NO_NETWORK_FALLBACK_SOURCES` (`registry.py:155-157`); an unavailable explicit `local` request raises `NoAvailableSourceError` instead of falling back to a network loader (`registry.py:642-659`). | §8 |

## 2. Config path redirection (no writes to the real home)

`Path.home()` honours `USERPROFILE` on Windows, so the whole probe ran against a temporary home; the real
`C:\Users\23302\.vibe-trading` was never touched and **no `data-bridge/` directory exists there** (nothing had to be
backed up or restored — `C:\Users\23302\.vibe-trading` still holds only `memory/` and `sessions.db*` from 2026-05).

```
env: USERPROFILE = C:\Users\23302\AppData\Local\Temp\vt-probe\home
Path.home()      = C:\Users\23302\AppData\Local\Temp\vt-probe\home
module file      = E:\Codes\Vibe-Trading\agent\backtest\loaders\local_loader.py
_CONFIG_PATH     = C:\Users\23302\AppData\Local\Temp\vt-probe\home\.vibe-trading\data-bridge\config.yaml
config exists    = True
```

Config written (one entry per file, `symbol` = Vibe-Trading's own spelling, `path` = the copy):

```yaml
sources:
  - symbol: "BTCUSDT"
    type: parquet
    path: "~/market/BTCUSDT/1h.parquet"
```

## 3. Did it resolve our parquet? Yes — as-is

Command (`%TEMP%\vt-probe`; `probe.py` installs a socket guard, then imports the real module):

```powershell
$env:USERPROFILE = "$env:TEMP\vt-probe\home"; $env:PYTHONDONTWRITEBYTECODE = "1"
& "D:\Program Files\Python\Python312\python.exe" -B probe.py
```

Source copy verified byte-identical first:
`sha256(BTCUSDT\1h.parquet) = a0d1ec9efd783e06038cace6e9c0491d472e1259998b2392bb12542d8b245a2c = sha256(copy)`.

Real output (abridged):

```
DataLoader.name / markets / requires_auth = local ['a_share', 'crypto', 'forex', 'fund', 'futures', 'hk_equity', 'macro', 'us_equity'] False
loader.is_available() = True
registry register pass: 'local' in LOADER_REGISTRY = True
====================================================================
fetch(['local:BTCUSDT'], '2025-05-01', '2025-12-31', interval='1H')
returned keys: ['BTCUSDT']
rows: 5880
index type: DatetimeIndex dtype: datetime64[ns] tz: None name: trade_date
index.is_monotonic_increasing: True is_unique: True
columns: ['open', 'high', 'low', 'close', 'volume']
column order == ['open','high','low','close','volume'] : True
dtypes: open/high/low/close/volume float64
span: 2025-05-01 00:59:59.999000 -> 2025-12-31 23:59:59.999000
head(2):
trade_date
2025-05-01 00:59:59.999  94172.00  94405.47  94130.43  94405.47  326.24800
2025-05-01 01:59:59.999  94405.46  94710.00  94398.51  94636.51  502.12633
```

**Refusal: none.** No exception, no warning, no dropped symbol.

## 4. Row / span comparison — delta 0, values bit-identical

Same window applied to the same file with plain `pandas`:

```
-- comparison, identical window --
source-file rows in window : 5880
loader rows                : 5880
row delta                  : 0
source span                : 2025-05-01 00:59:59.999000 -> 2025-12-31 23:59:59.999000
loader span                : 2025-05-01 00:59:59.999000 -> 2025-12-31 23:59:59.999000
span-start equal           : True
span-end equal             : True
index set identical        : True
close series identical     : True
open series identical      : True
volume series identical    : True
```

Then repeated for **all 52 files** in `data/market` (287,909,742 bytes; 11 symbols × {1m,5m,15m,1h,4h,1d}), each copied
to `%TEMP%\vt-probe\home4\market_full\<SYM>\<iv>.parquet`, each fetched at its own native interval
(`1h→"1H"`, `4h→"4H"`, `1d→"1D"`, minutes unchanged) over 2025-03-31 → 2026-10-03:

```
files probed: 52 | mismatches: 0
dropped columns per file (source cols minus returned cols), distinct sets:
   x52: dropped ['quote_volume', 'trade_count']
```

Representative rows of that sweep (`src rows` = direct pandas read of the copy under the identical window):

```
file                   iv    src rows  ldr rows   d span ok vals ok idx ok cols
BTCUSDT_1m             1m      789871    789871   0 True    True    True   ['open','high','low','close','volume']
BTCUSDT_1h             1H       13164     13164   0 True    True    True   ['open','high','low','close','volume']
BTCUSDT_4h             4H        3291      3291   0 True    True    True   ['open','high','low','close','volume']
BTCUSDT_1d             1D         549       549   0 True    True    True   ['open','high','low','close','volume']
ENAUSDT_1h             1H         200       200   0 True    True    True   ['open','high','low','close','volume']
MOVRUSDT_5m            5m         500       500   0 True    True    True   ['open','high','low','close','volume']
```

**Column handling:** no renaming, no coercion, no reordering. `quote_volume` and `trade_count` are **dropped**
(`_normalize_columns` selects only the five OHLCV columns, `:191-195`) for all 52 files; nothing else is lost. Dtypes
are `float64` in both source and output (source already `double`), so no precision change. The dropped columns matter
only if a consumer wants notional volume — `base.py:279` (`_SUMMED_COLUMNS = {"volume","amount"}`) would have used an
`amount` column, which our files never had.

**Index verdict: yes, it is what the engines expect.** A UTC-naive `DatetimeIndex` named `trade_date`, sorted and
unique. `_normalize_columns` sets exactly that contract (`:186-189`); the engines consume `frame.index` directly
(`engines/base.py:266-277` merges per-symbol indices via `asi8` with an explicit `datetime64[us]`→`ns` guard for
"a duckdb-backed local source", `:736`, `:1515`, `:1568` test `ts in frame.index`). Our `datetime64[ns]` index is the
plain case. Note the *stamp convention* difference: our files stamp a bar at its close
(`…01:59:59.999`), seen above and preserved verbatim.

## 5. Interval handling (measured, not inferred)

```
interval='1H': rows=5880 median_spacing=0 days 01:00:00   (source bars returned unchanged)
interval='1h': rows=5880 median_spacing=0 days 01:00:00
interval='4H': rows=1470 median_spacing=0 days 04:00:00
interval='1D': rows=245  median_spacing=1 days 00:00:00
interval='15m': rows=5880 median_spacing=0 days 01:00:00
  LOG WARNING local loader: cannot upsample 0 days 01:00:00 source bars to 15m for BTCUSDT; returning source bars
interval='1W': rows=5880 median_spacing=0 days 01:00:00
  LOG WARNING local loader: unsupported interval '1W' for BTCUSDT; returning source bars
```

Two things follow. (a) A coarser request is served by resampling and the bar count divides exactly (5880/4=1470,
5880/24=245). (b) A request the file cannot satisfy **does not fail** — it logs and returns the source bars, so the
caller's declaration silently disagrees with what arrived; the runner compensates downstream with a median-spacing
gate (`backtest/runner.py:1337-1347`, `_SPACING_MISMATCH_RATIO = 1.5`, `_MIN_BARS_FOR_SPACING = 4`) that re-derives
annualisation from observed spacing. (c) Resampled bars are relabelled to the **period start** (the 4H frame above
starts at `2025-05-01 00:00:00`, the 1D frame at `00:00:00`), whereas our native files are stamped at bar **close**:
a resampled frame and a native file are not stamp-compatible, so 4H/1D comparisons must be done like-for-like.

## 6. Symbol mapping — the real adapter surface

* The loader's `symbol` is opaque and matched **exactly, case-sensitively**: `MiXeD` resolved, `mixed` returned `{}`
  with `LOG WARNING local loader: no config entry for symbol mixed`. `BTCUSDT` and `BTC-USDT` both work as keys
  (probed), and a bare code without the `local:` prefix works too.
* **Duplicate `symbol` keys silently collapse — last entry wins** (`_source_by_symbol` is a plain dict, `:266-269`).
  Probed with two `symbol: "DUP"` entries (1d file first, 1m file second):

  ```
  _source_by_symbol resolves to: {'DUP': '.../BTCUSDT_1m.parquet', 'MiXeD': '.../BTCUSDT_1d.parquet'}
  DUP -> 30 rows index=2025-06-01 00:00:00..2025-06-30 00:00:00   <-- the second entry, silently
  ```

  Row count alone would not reveal the substitution; only the stamp (`00:00:00` vs `23:59:59.999`) does. **A generator
  that emits one entry per file under a symbol-only key is therefore wrong** — the key must encode the interval
  (e.g. `BTCUSDT:1h`, which survives the single-`split(":", 1)` prefix strip at `:297`).

## 7. Timezone, dates and the loader cache

* **UTC-naive by contract** (`:186-189`): our naive stamps pass through unchanged; a tz-aware input would be converted
  to UTC and stripped. Our cache is already UTC-naive, so nothing is shifted. A `date` column of epoch **milliseconds**
  is accepted but parsed as **nanoseconds** (`pd.to_datetime` on ints) and lands in 1970 — measured:
  `_read_parquet(range_int) -> 50 rows, index[0]=1970-01-01 00:29:03.469199999`, after which the window filter drops
  everything. Epoch-ns columns parse correctly. Our files are unaffected (they carry a real `DatetimeIndex`), but the
  failure mode is silent, so any future epoch-ms parquet needs a `columns`/preprocessing fix rather than a config key.
* **Column-name requirement, measured on derived copies:**

  ```
  NAMED_DATE      (RangeIndex + column `date`)            -> 50 rows ACCEPTED
  NAMED_TIMESTAMP (RangeIndex + column `timestamp`)       -> None  REFUSED
  NAMED_TIMESTAMP_OVERRIDE (same file, columns.date=timestamp) -> 50 rows ACCEPTED
  RANGE_INT_NS    (RangeIndex + int64 ns column `date`)   -> 50 rows ACCEPTED
  RANGE_INT       (RangeIndex + int64 ms column `date`)   -> 1970 index, filtered away
  ```
  So the accepted date sources are: a `DatetimeIndex` (any name), or a column literally named `date` (or named by a
  `columns.date` override), holding datetime or epoch-ns. Our cache is the first case.
* **How it refuses — silently.** When `_normalize_columns` returns `None`, `_fetch_one` returns `None` and `fetch`
  simply omits the symbol from the result dict. There is **no exception and no warning** for a schema mismatch; the
  only visible signal is the *absence* of the key. (A missing/unreadable file does log, e.g.
  `local loader failed for DUP: [Errno 2] No such file or directory: '...BTCUSDT_1m.parquet'`, still returning `{}`.)
  Callers must check for missing keys.
* **Opt-in loader cache** (`VIBE_TRADING_DATA_CACHE`, default **off**; `base.py:457-505`). With it off, the read path
  is `pd.read_parquet` only. Forced on in the probe:

  ```
  loader_cache_enabled() = True
  cache path = ...\cache\local\bcb865e9...parquet
  fetch rows with cache enabled: 5880
  LOG WARNING loader cache write failed for ...: No module named 'duckdb'
  duckdb imported: False
  ```
  The cache is keyed by source/symbol/timeframe/range **plus** a hash of the reader settings (`_source_cache_identity`,
  `:131-148`), and a write failure is swallowed, so a missing `duckdb` costs the cache, not the data. Note it created
  `...\cache\local\` in passing — enabling the cache writes under the cache root (default
  `~/.vibe-trading/cache/loaders`), which is one more reason to set `VIBE_TRADING_DATA_CACHE_ROOT` to a scratch path.

## 8. Network access: it stayed offline

Static control flow first: `local_loader.py` imports only `hashlib, json, logging, pathlib, typing, pandas, yaml`,
`backtest.loaders.base` and `backtest.loaders.registry` (`:32-44`); `base.py` imports only stdlib + pandas; the HTTP
clients live in *other* loader modules. The parquet branch is `pd.read_parquet(path)` on a local file (`:214-221`) and
nothing else. `duckdb` is imported lazily and only for the `duckdb` source type or an enabled cache.

Observed: every probe run installed a guard that makes `socket.socket.connect`, `socket.connect_ex`,
`socket.create_connection`, `socket.getaddrinfo`, `urllib.request.urlopen` and `http.client.HTTPConnection.connect`
raise and record, then executed the loader. Results:

```
NET_ATTEMPTS recorded: []          # main probe, 5880-row fetch + 6 interval fetches + symbol probes
NET_ATTEMPTS: []                   # 52-file sweep
NET_ATTEMPTS: []                   # schema probes, cache-on probe, duplicate-symbol probe
NET_ATTEMPTS: []                   # fail-closed probe
```

The fail-closed path was exercised directly with **no config file present** (empty temp home):

```
config exists    = False
is_available()   = False
get_loader_cls_with_fallback('local') RAISED NoAvailableSourceError: Data source 'local' is unavailable and does
  not fall back to a network source. Check your Data Bridge config (~/.vibe-trading/data-bridge/config.yaml)
  — it must exist and list at least one source.
NET_ATTEMPTS: []
```

So the claim in `registry.py:642-659` holds under execution: an explicit `local` that cannot serve raises rather than
fetching, and zero sockets were opened across every run.

## 9. Reverse direction: their conventions ← our layout (paper only, nothing foreign was run against our engine)

* **Intervals.** Their canonical set is `_VALID_INTERVALS = {"1m","5m","15m","30m","1H","4H","1D","1W","1M"}`
  (`backtest/runner.py:58`); the OKX loader additionally accepts either case (`loaders/okx.py:36-47`), and the local
  loader's own rules accept both (`local_loader.py:63-74`). Our filenames are lowercase for hour/day. One mapping
  table is enough, exactly the one used in the sweep:
  `{"1m":"1m","5m":"5m","15m":"15m","1h":"1H","4h":"4H","1d":"1D"}` — the run config, not the file, must carry the
  uppercase form, because the runner validates the declared interval and `1W`/`1M` are resampled from `1D` in
  `base.py:273-342` rather than requested from a loader.
* **Symbols.** Their crypto spelling is `BTC-USDT`; CCXT parses it with `normalized.replace("-", "/")` and
  `BASE-USDT-PERP` → `BASE/USDT:USDT` (`loaders/ccxt_loader.py:60-70`), OKX re-normalises `/`→`-` (`okx.py:168`), and
  `src/market_data.py:49` routes `^[A-Z]+-USDT$` to OKX. Our `BTCUSDT` layout is nevertheless already understood by
  the market detector — `_market_hooks.py:85` matches concatenated `^[A-Z]{2,}(?:USDT|USDC|BUSD)$` as `crypto` — but
  the engine-side alignment helper does not (`engines/base.py:117` `_CRYPTO_RE = ^[A-Z]+-USDT$|^[A-Z]+/USDT$`), so
  `BTCUSDT` is classified as `equity` there and merely gets a different ffill limit (5 vs 10,
  `_detect_market_for_align`, `:132-138`).
* **What a one-line adapter looks like.** Because the `local` loader's `symbol` is an opaque alias for a literal file
  path, **no file renaming and no symbol rewriting on our side is needed** — the alias is the adapter:

  ```python
  sources = [{"symbol": re.sub(r"(USDT|USDC|BUSD)$", r"-\1", d.parent.name) + ":" + f.stem, "type": "parquet", "path": str(f)}
             for d in MARKET_ROOT.iterdir() for f in sorted(d.glob("*.parquet"))]
  ```

  This emits `BTC-USDT:1h` → `…/BTCUSDT/1h.parquet`, keeping their hyphen spelling (so the crypto hooks and the
  engine's `_CRYPTO_RE` agree) while the interval is carried in the alias key — mandatory, per the duplicate-symbol
  collapse in §6. Executed with exactly that config shape (`local_loader.py:297` strips only the first `:`):

  ```
  config symbol "BTC-USDT:1h" and "BTC-USDT:4h", both -> BTCUSDT/1h.parquet
  'local:BTC-USDT:1h'  -> ['BTC-USDT:1h']      (5880 rows, distinct entry)
  'local:BTC-USDT:4h'  -> ['BTC-USDT:4h']      (distinct entry, same file, served at 4H)
  'local:BTC-USDT'     -> []                   (no entry: the interval is part of the key)
  'BTC-USDT:1h'        -> ['BTC-USDT:1h']      (prefix optional)
  ```

  so the alias with a colon resolves and the two intervals stay distinct instead of collapsing. The returned dict key
  is the alias verbatim. The inverse (`BTC-USDT` → our directory) is `sym.replace("-", "").upper()`; the forward
  one-liner is the regex above. The only asymmetry left is the interval's case, and it lives in the run config, not
  in the filename.

## 10. Verdict

**Yes — our cache loads as-is, with no transformation and no adapter file.** All 52 parquet files
(11 symbols × {1m,5m,15m,1h,4h,1d}, 287.9 MB, 2025-04-01 → 2026-10-02) load through Vibe-Trading's own `local` loader
at their native intervals with row delta **0**, identical index objects and bit-identical OHLCV values. The only
column effect is the intentional drop of `quote_volume` and `trade_count`. The returned frame is exactly what their
engines consume (`trade_date`-named UTC-naive `datetime64[ns]` index, `[open,high,low,close,volume]` float64), and the
loader opened no sockets in any run.

What is required is not code but **config**: one `data-bridge/config.yaml` entry per file (no globbing, no `<SYM>`
`<interval>` templating), with the interval encoded in the `symbol` alias to avoid the silent duplicate-key collapse,
and the interval declared in their `1H/4H/1D` case. Config path redirection needs `USERPROFILE`/`Path.home()` — there
is no env var for it.

## 11. Cautions confirmed

* **Secrets — do not run their runner carelessly.** `backtest/runner.py:1209-1214` calls `load_dotenv()` at the start
  of `main()`. With no argument, python-dotenv walks up from the calling file to find `agent/.env`, so running
  `python -m backtest.runner <run_dir>` **would load `agent/.env`'s `DEEPSEEK_API_KEY` and `TUSHARE_TOKEN` into that
  process's environment**. I did not run it and never read the file (metadata only: 5650 bytes, mtime 2026-05-24,
  untouched). The import chain I did use never touches dotenv: `src/config/` contains no `load_dotenv`; the call sites
  are `cli/main.py:286-288`, `cli/_legacy.py:425-429`, `backtest/runner.py:1210-1212`, `scripts/w4a_run_benches.py:24-31`
  and `src/api/*` (settings only). Importing `local_loader`, `base`, `registry` and `src.config.accessor`
  (`base.py:483` / `:500`, pydantic `EnvConfig` over `os.environ`) loads no dotenv. Anyone continuing with the real
  runner should run it with a sanitised environment (e.g. `env -i`) and a run dir outside both repos.
* **Ports.** Confirmed free right now: the only listener among 3080/8000/8899/8900 is `127.0.0.1:3080` (this GUI).
  Vibe-Trading's `dev` default (8899), `serve` (8000) and MCP (8900) are unoccupied — but 8899 is *our* default too,
  so any future run must pass `--port`.
* **Environment.** Their in-tree `.venv` was not used and must not be (declared pins vs installed versions disagree).
  A fresh environment is *not* needed for this loader test: stock CPython 3.12.10 with `pandas 2.3.3` (`requirements-lock.txt`
  pins 2.3.3), `pyarrow`, `PyYAML` and `pydantic` suffices; only the opt-in cache and the `duckdb` source type need
  `duckdb`. Keep two environments; cross with files. Never let a foreign process unpickle our
  `data/ga_checkpoint.pkl`.
* **Writes.** With the cache off (default) this loader writes nothing anywhere. Enabling `VIBE_TRADING_DATA_CACHE`
  creates directories under its cache root, and the config lives under `Path.home()`; both were redirected to `%TEMP%`
  in this probe, and the real `~/.vibe-trading` was never modified.

## 12. What I could not determine

* Whether a full `backtest.runner` run (config + `code/signal_engine.py` + engine) succeeds end-to-end on our bars —
  deliberately not run, because `main()` sources `agent/.env` and installs/executes run-dir code. Only the **data
  layer** is proven here.
* Whether the loaders' `.pyc`-free import would also be clean for the full runner (the runner imports `ccxt`,
  `requests`, `dotenv`, `src.tools.path_utils`, … which the host interpreter may or may not have).
* Their behaviour on windows shorter than the file, on symbols that vanish mid-run, or on `1W`/`1M` declared against a
  minute file beyond the median-spacing gate — only the six native intervals were swept.
* Whether `resample`'s start-labelling (§5c) shifts their next-bar-open fill relative to a native 4H/1D file in a way
  that matters at the P&L level — that requires the full runner, which was out of scope.
* The `csv` and `duckdb` source types (only `parquet` was executed), and the loader cache's round-trip (needs
  `duckdb`, not installed).
* Whether upstream `local_loader` has changed since `251b0943`; v0.1.16 at that commit is what was probed.

## 13. Evidence kept in `%TEMP%` (not deleted; safe to remove)

`C:\Users\23302\AppData\Local\Temp\vt-probe\` — `probe.py`, `source_schema.py`, `schema_probe.py`, `schema2_probe.py`,
`sweep.py`, `cache_probe.py`, `failclosed.py`, `dup_probe.py`, `alias_probe.py`; `home/.vibe-trading/data-bridge/config.yaml` (single
1h entry) and copies at `home\market\BTCUSDT\1h.parquet`; `home2/`, `home3/`, `home5/` configs plus derived/refused
variants and copies of `BTCUSDT/1d.parquet` and `BTCUSDT/1m.parquet`; `home4/.vibe-trading/data-bridge/config.yaml`
(52 generated entries) and `home4\market_full\**` (all 52 files copied, 287.9 MB); `home-empty/` (no config);
`cache/` (empty, from the cache-on run). No source file in either repository was modified, and Vibe-Trading's
`git status` is clean.
