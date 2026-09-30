# 14 — Volume / flow features, the cache schema extension and the versioned contract (P6-B)

**Scope.** P6-A (page 13) made *execution* volume-aware. This page documents the
two things P6-B adds and one thing it leaves alone:

1. the **cache schema extension** — `quote_volume` (quote-asset / USDT traded
   notional) and `trade_count` beside the original five OHLCV columns, with a
   resumable backfill for files that predate them;
2. the **volume / flow feature family** — 15 new columns in the ML contract,
   taking it from 39 to 54, with an explicitly **versioned** schema hash so a
   model trained on the old contract is refused by name;
3. the **P6-A seam wiring** — the live sizer and the backtest cost model now
   actually receive a per-window quote volume. Both remain inert by default.

> **No predictive claim.** P6-B does not claim volume predicts direction. The
> 1h direction gate was refused twice before P6-B and is **refused again with the
> v2 contract** (§6) — that refusal is the result, not a failure to tune. Volume's
> documented value here stays where P6-A put it: capacity, cost, and (P6-C/D) state.

---

## 1. Cache schema: two new columns, one documented absent value

`data/market/<SYMBOL>/<interval>.parquet` now carries

```
index  : close_time (datetime64[ns], UTC — one row per closed candle)
columns: open, high, low, close, volume, quote_volume, trade_count   (float64)
```

`quote_volume` is the kline payload's field 7 (`quoteAssetVolume`) and
`trade_count` its field 8 (`numberOfTrades`); `core/market_data/ohlcv_cache.py`
owns the column list (`CACHE_COLUMNS`) and `scripts/download_history.py` the
kline→frame mapping (`KLINE_FIELDS`, `klines_to_frame`).

**Backward compatibility is a rule, not a hope.** A pre-P6-B file has neither
column, and every reader must tolerate that:

| situation | documented behaviour |
|---|---|
| column absent | readers must not crash; the feature layer falls back to the `volume × close` **proxy** (the same proxy page 13 documents) |
| column present, row `NaN` | `NaN` means "not measured". `core.risk.liquidity.recent_quote_volume` drops non-finite bars from the sum; `core.ml.features` uses the real column and does **not** substitute a proxy per row (a hole is reported as a hole, never filled with a lookalike) |
| a frame that has the column is fully `NaN` | `recent_quote_volume` falls back to the proxy rather than reporting `0.0` — `0.0` would read as "nothing traded" and refuse an order the proxy could size honestly |
| two writers disagree (one knows the columns, one does not) | the union (`merge_history`) fills the missing side with `NaN` and keeps the stored numbers; `incoming-wins` is per **row**, never per column |

`canonical_columns()` is the one place the column order/dtype is fixed. The write
paths call it with `reorder=False`, because `_frame_hash` hashes the column names
*in order* and reordering a legacy file would make every flush rewrite a file
nobody appended to — the "a no-op flush does not touch bytes" property pinned by
`tests/test_reaudit_fixes.py`.

### 1.1 Backfill (`--backfill`), and why a rewrite is necessary

`quote_volume` and `trade_count` come from the kline payload, so they exist only
for bars a P6-B writer downloaded. They **cannot be derived** from what is on
disk (`volume × close` is the documented *proxy*, not the source number), so a
pre-P6-B history must be re-read from the source. What the backfill does *not* do
is re-download history it does not need:

* `backfill_plan(frame)` returns the contiguous runs of bars whose
  `quote_volume` is absent (runs split at gaps wider than four bar lengths),
  oldest first — pure and network-free;
* each run becomes one request window `[bar open, bar close]`. The bar open is
  resolved from the stored stamp (`_bar_starts`), so **both** timestamp
  conventions produce the same window: a bar-open stamp *is* the open, while
  Binance's `close_time` stamp is the open plus one bar. (Treating a
  `close_time` stamp as an open stopped the first backfill one bar short of the
  end of the file — measured, and the reason `_bar_starts` exists.)
* only `quote_volume` / `trade_count` are taken from the fetch; the fetched bars
  are matched onto the stored timestamps by bar open and the **stored**
  prices/volumes are authoritative;
* **resumable**: a second run finds `missing == 0`, makes zero requests and
  reports `done`;
* **refuses a moved file**: if the row count or last stamp changed while the
  requests were in flight (the live service appends to the cache as it trades),
  `CacheMovedError` is raised and **nothing is written**.

```bash
python scripts/download_history.py --backfill --symbols BTCUSDT,ETHUSDT --intervals 1h
```

## 2. The feature family (15 columns, contract v2)

Contract v1 was 39 columns with five volume-ish columns
(`volume_ratio`, `vol_chg_5`, `vol_chg_20`, `vol_trend`, `frac_vol_10`). P6-B adds
15 (`core/ml/features.py`), **54 columns total**, hash `1f30fded996d`.

**Naming: `vol_5` is NOT volume.** `vol_5` / `vol_10` / `vol_20` / `vol_regime` are
the **rolling standard deviation of `ret_1`** (return volatility), shipped since
P2. The P6-B family therefore uses different prefixes, and the distinction is
pinned by a test: on a constant-volume series `vol_5` still varies (price moves)
while `volr_5 == 1` exactly.

| column | definition | why this form |
|---|---|---|
| `volr_5`, `volr_10`, `volr_20`, `volr_60` | `volume / rolling_mean(volume, w)` | multi-window RVOL; `1.0` = at the window average |
| `volz_60` | `(log volume − anchor_centre) / (1.4826·anchor_MAD)` | volume z-score against an **anchored** median/MAD, so a spike cannot inflate the scale and a rolling window cannot retroactively move a historical value |
| `vwap_dev_20` | `close / rolling_VWAP(20) − 1` | where price sits relative to what was actually paid in the window |
| `vwap_dev_session` | `close / expanding_VWAP − 1` | the same against the **causal expanding** VWAP from the first bar of the series ("session" = one pass over one series) |
| `flow_close_position_weighted` | `clip((c−l)/(h−l),0,1) · centroid_20` | closing position weighted by how much of the window's volume traded in the bar |
| `obv_slope_10` | least-squares slope of OBV over 10 bars ÷ mean volume(20) | the **slope**, not the level: a cumulative line is a random walk whose magnitude depends on where the series starts |
| `ad_slope_10` | the same for the accumulation/distribution line | idem |
| `flow_cmf_20` | `Σ(mfv,20) / Σ(volume,20)`, `mfv = ((c−l)−(h−c))/(h−l) · volume` | Chaikin money flow |
| `flow_mfi_14` | `100 − 100/(1 + Σup/Σdown)` on typical price × volume | Money Flow Index(14) |
| `flow_amihud_20` | `mean(|ret_1| / quote_volume, 20) × 1e6` | Amihud illiquidity (Amihud 2002): price move per unit of traded notional |
| `flow_vol_price_corr_20` | `corr(|ret_1|, Δ log volume)` over 20 bars | volume arriving *with* the move |
| `flow_vol_centroid_20` | `volume / Σ(volume,20)` | where in the window the volume sits |

### 2.1 "Anchored" means causal — and the difference is measured

Page 10's `AnchorMAD` builds one centre/scale **per estimator call** so a clip
decision cannot move with the window. Applying that recipe literally to a feature
(one anchor from the whole series) would be stable *within a call* but **not
causal**: appending bars moves every historical z-score. The acceptance criterion
for P6-B is per-column `0` changes after appending 500 future bars, so
`_expanding_mad` uses the causal block form instead:

```
anchor(t) = median/·MAD of series[: block_start(t)],   block_start(t) = floor(t/stride)·stride
```

* a bar can never influence its own anchor (only **closed** blocks are read);
* appending future bars cannot change any historical anchor — the property a
  whole-series anchor lacks and a rolling window was never claimed to have;
* the price is a bounded staleness: at most `stride − 1` bars. `stride = 1` gives
  the exact per-bar expanding median and is supported (the causality test uses it
  to prove the anchor is causal at *every* bar).

`tests/test_volume_features_v2.py` measures all three forms side by side.

## 3. The versioned contract

| version | columns | `feature_schema_hash` |
|---|---|---|
| v1 | 39 (P2) | `335e63360104` (frozen literal `FEATURE_SCHEMA_V1_HASH`) |
| v2 | 54 (P6-B) | `1f30fded996d` (`FEATURE_SCHEMA_VERSION = 2`) |

A model artefact carries `feature_schema_hash` in its `*_meta.json` sidecar. Both
load paths — the live `MLPredictor.load_model` and the backtest
`BacktestEngine._verify_ml_model_sidecar` — compare it against the contract they
will score with and **refuse on any difference**. P6-B routes both through one
function, `feature_schema_mismatch_reason`, so the refusal cannot be worded two
ways, and names the version on both sides:

```
feature schema hash mismatch: model was trained on v1 (39-column P2 contract)
[335e63360104] but the current contract is v2 (54-column P6-B contract)
[1f30fded996d] — a model trained on a different feature set must be retrained,
not scored
```

A sidecar with **no** hash stays a refusal too ("feature schema hash missing …"),
the re-audit finding that a `if stored and …` test let through.

## 4. The P6-A seams are now wired — and inert by default

P6-A shipped the participation cap and the impact term but left two wires
unconnected: the live `RiskManager` never passed `recent_quote_volume` to the
sizer, and the backtest engine never passed per-bar volume to
`apply_trading_costs`. P6-B connects both:

| wire | file | default behaviour |
|---|---|---|
| live | `RiskManager.resolve_recent_quote_volume` → `PositionSizer.calculate_position_size(recent_quote_volume=…, symbol=…)` | `risk.liquidity.enabled: false` → returns `None` **without touching market data**; the sizer calls the cap only for a non-`None` value while the switch is on |
| live (reader) | `PositionGuard.recent_quote_volume` / `quote_volume_enabled` | `None` without I/O while the switch is off |
| backtest | `BacktestEngine._recent_quote_volume_for(pos)` → `apply_trading_costs(recent_quote_volume=…)`, window = `risk.liquidity.lookback_bars` bars ending at the current bar | `impact_k: 0.0` → `_impact_bars()` is `0`, the lookup returns `0.0` immediately and the cost arithmetic is the pre-P6 sum (the k=0 identity page 13 pins) |

The number both sides read is the same function
(`core.risk.liquidity.recent_quote_volume`), so a live participation decision and
a backtest impact charge cannot be computed from two different definitions of
"recent volume".

## 5. Measured evidence (this revision)

| what | command | result |
|---|---|---|
| no look-ahead | `python -m pytest tests/test_volume_features_v2.py -q` | appending 500 bars: **0** changed values in all 54 columns |
| non-degenerate | `python scripts/ml_credibility_measure.py --symbols BTCUSDT ETHUSDT --intervals 1h --out data/p6b_evidence` | `near_constant: []` for both symbols under the v2 contract (54 columns) |
| deterministic | `tests/test_volume_features_v2.py::test_two_runs_of_the_same_inputs_are_bit_identical`; the evidence script re-run | 54 columns bit-identical across two `compute_features` calls; the re-run's JSON was **byte-identical** (`sha256_16 = 42a64e7c3f913bf4` both times) |
| cost | `tests/test_volume_features_v2.py::test_the_feature_pipeline_cost_stays_within_the_existing_bound` and `tests/test_ml_credibility.py::test_feature_pipeline_cost_is_bounded` | 11 627 synthetic bars: 0.710 s (61.1 µs/bar); 11 627 live bars: 0.688 s (indicators 0.081 s + features 0.608 s); bound 3.0 s |
| cache extension vs the source | 200 sampled 1h bars re-fetched from `data-api.binance.vision` | `quote_volume` max abs diff `0.000000`, max rel diff `0.0`, 200/200 exact; `trade_count` 200/200 equal |
| integrity | `python scripts/check_data_integrity.py` on a temp copy of the live file, before/after backfill | 11 627 bars, missing 1, gaps 1, **twin 0** — unchanged; the backfill filled 11 627/11 627 `quote_volume` cells |
| the live file | `sha256_16` of `data/market/BTCUSDT/1h.parquet` | `7f4e344062bddbdf` before and after; **5 columns**, rows 11 627 — the live cache was never written by P6-B |

### 5.1 The honest gate verdict (v2 contract)

`python scripts/ml_credibility_measure.py --symbols BTCUSDT ETHUSDT --intervals 1h`
(evidence: `data/p6b_evidence/ml_p2_measurements.json`; note the script's
`--symbols`/`--intervals` take **space-separated** values, not commas):

| symbol | features | OOS AUC | OOS net expectancy | trades | t | PSR | gate |
|---|---|---|---|---|---|---|---|
| BTCUSDT 1h | 54 | 0.5228 | 0.0000 % | 0 | 0.00 | 0.000 | **REFUSED** (`OOS AUC 0.5228 <= 0.55; … too few trades (0 < 100); not significant`) |
| ETHUSDT 1h | 54 | 0.5324 | −0.4132 % | 889 | −3.67 | 0.000 | **REFUSED** (`OOS AUC 0.5324 <= 0.55; net expectancy -0.4132% <= 0.0000%; not significant`) |

`ml.enabled` therefore stays **`false`**. The v2 family did not turn volume into a
directional edge, exactly as the phase plan anticipated (`§0.3`); nothing was
tuned to change that.

## 6. Limitations (stated, not hidden)

* **The impact coefficient is illustrative.** `impact_k: 0.0` ships; any non-zero
  value is an assumption, not a calibration (page 13 §5).
* **Quote volume is only as complete as the backfill.** A partially backfilled
  file has `NaN` rows; `recent_quote_volume` then sums *fewer* bars rather than
  substituting a proxy per row, so the window is conservative in the sense of
  "smaller denominator" only in expectation — a reader that needs an exact window
  should re-run `--backfill`.
* **`trade_count` is persisted but no feature consumes it yet.** It is the input
  P6-C/D need (dollar bars, amount concentration); keeping it in the cache now
  avoids a second backfill later. No config key, no feature, no reader claims
  otherwise.
* **The volume family is not sign-oriented by construction.** Amihud, correlation
  and the slopes are *state descriptors*; whether any of them carries 1h
  directional information is exactly what §5.1 measured and refused.
* **`vwap_dev_session` is an expanding VWAP, not a calendar session.** A
  calendar-day session VWAP is also causal but re-anchors at each UTC midnight,
  which makes it *not* invariant when bars are appended to a series — it cannot
  satisfy the "append 500 bars ⇒ 0 changes" criterion and is deliberately not
  used.
