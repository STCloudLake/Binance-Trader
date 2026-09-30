# 15 — Volume-clock sampling (dollar/volume bars) and market volume breadth (P6-C)

**Scope.** P6-C asked one experimental question and built one new input:

1. **Does sampling by traded activity instead of by the clock improve the
   statistical behaviour of returns — and, decisively, the ML credibility
   gate?**
2. **Can a market-level *breadth* series be built from the reachable public
   ticker without look-ahead and without fabricating a value?**

**Headline verdict (measured, not asserted).**

> **The gate does not improve, and the raw distribution comparison does not
> survive.** On the real cache the plan's literal criterion (JB **and** excess
> kurtosis both down, interval excluding 0) passes in **1 of 12** cells
> (SOLUSDT dollar bars at 500 target bars). The decisive protocol refuses
> **both** samplings — dollar AUC **0.4664** vs time AUC **0.5361** (re-measured
> 2026-09-30; the time row is stable, the dollar one moves with the cache), both
> net-negative after cost, both `allowed=false` — so the answer to the question
> P6-C actually asks is **"no improvement"**.
>
> The one thing that *does* replicate is conditional and comes with a caveat:
> the cached 1-minute series has **27 holes** (166 499 missing minutes = **23.9 %**
> of its 697 805-minute span, measured 2026-09-30, the largest 61.8 days). Once the returns that span
> those holes are removed from **both** samplings by one shared rule, activity
> sampling shows a lower excess kurtosis in all 12 cells and a
> block-bootstrap-supported reduction in 10 of them (all six at 4 000 bars, e.g.
> BTCUSDT −5.56, 95 % CI [−7.36, −3.72]). That is a property of *this cache's
> holes*, not a licence to switch clocks, and it changes no gate verdict.
>
> Breadth is built and measurable, with one criterion that passes on a small
> sample and one stale expectation: **(676 usable / 676 reported USDT pairs =
> 100 % of the endpoint's universe; 136.3 % of the plan's 496)**; lag-1
> autocorrelation 0.9713 / 0.4858 / 0.6608 for total quote volume / up-share /
> HHI, all inside the `< 0.99` bar on **n = 6**; variance > 0 everywhere;
> endpoint-unreachable returns `None` and caches nothing.

Nothing here is enabled by default and nothing is wired into a production gate:
P6-C ships a library, an experiment tool, tests and this page.

---

## 1. What was built

| file | role |
|---|---|
| `core/strategy/volume_bars.py` | `dollar_bars`, `volume_bars`, `time_bars`, `bars_as_of`, `VolumeClockBuilder`, causal thresholds, print-hole detection, distribution statistics, block bootstrap |
| `core/market_data/breadth.py` | 24h-ticker breadth: parse → aggregate → TTL cache → replay; standard library only |
| `tools/p6_volume_bars_experiment.py` | the reproducible runner (`bars` / `gate` / `breadth` / `all`) |
| `tests/test_volume_bars.py` (48 tests), `tests/test_breadth.py` (24 tests) | synthetic-fixture contracts; no network, no live-cache read |

Sampled frames carry **exactly** the cache's five columns
(`open/high/low/close/volume`, index named `close_time`). New cache columns that
P6-B is adding (`quote_volume`, `trade_count`) are dropped rather than passed
through, so a growing cache cannot change the sampled layout — pinned by
`test_extra_cache_columns_are_dropped`.

**The sampling rule.** A bar closes at the first print whose weight brings the
weight accumulated **since the previous bar closed** to the threshold; the
overshoot is discarded and the next bar starts at the following print. A bar
whose accumulated weight is still below the threshold at the end of the data is
**provisional** and is not emitted (`drop_partial=True`); `time_bars` applies
the same rule to its final resample bin (`drop_last=True`) and pins
`origin="start_day"` so appending data cannot move the bin grid.

**Comparability.** `notional_threshold(frame, target_bars, calibrate_on=0.2)`
uses only the **leading 20 %** of prints, so the grid is known before the first
bar exists. The time control's interval is then solved (`match_time_interval`)
so the control has the same number of non-empty bins as the dollar series
(4 000 target ⇒ 137 min/BTCUSDT, 179 min/ETHUSDT, 154 min/SOLUSDT); both the
interval and the realised counts are reported, because a short fatter-tailed bar
against a long thinner-tailed one would turn every metric into a restatement of
bar length.

---

## 2. Causality — the one hard requirement

Every constructor is a **prefix function**: bar *i* depends only on the prints up
to and including its own closing print. The measurement is
`core.strategy.volume_bars.as_of_consistency`, which rebuilds the series from 8
growing prefixes of the same frame and counts how many bars of the shorter build
the full build revises (`historical_bars_unchanged`, all five columns compared
bit for bit).

```
python tools/p6_volume_bars_experiment.py bars --symbols BTCUSDT ETHUSDT SOLUSDT \
    --target-bars 4000 --n-boot 400 --out %TEMP%/p6c
```

Cached revision measured: **531 248** 1-minute prints per symbol,
2025-06-03 00:0x → 2026-09-30 12:5x (BTCUSDT / ETHUSDT / SOLUSDT ±4 min). The
cache is rewritten by the running app (it grew by ~30 rows during this session),
so the row count and window are part of the record.

| symbol | dollar | volume | time (control) |
|---|---|---|---|
| BTCUSDT | 0 / 3 888 changed | 0 / 4 729 | 0 / 3 889 |
| ETHUSDT | 0 / 2 987 | 0 / 3 078 | 0 / 2 982 |
| SOLUSDT | 0 / 3 475 | 0 / 3 980 | 0 / 3 462 |

**Acceptance: appending future data changes 0 historical bars/values — MET**
(3 symbols × 3 samplings × 8 prefixes, worst case 0).

The streaming form agrees with the batch form on **boundaries and all four
prices exactly** (`boundaries_equal: true`, `ohlc_bars_changed: 0`). Its `volume`
column differs on the last bits for 2 279–2 915 of ~3 000 rows, maximum relative
difference **1.7e-15 / 2.5e-15 / 2.2e-15** — the builder adds print volumes
sequentially while the batch path uses pandas' own reduction. That is float
summation order, which is why `historical_bars_unchanged` reports it per column
instead of hiding it inside one count.

Keeping the partial bar is the one documented exception: with
`drop_partial=False`, exactly the trailing provisional bar may be revised and
nothing else (`test_keeping_the_partial_bar_is_the_only_thing_that_can_move`).

---

## 3. The data caveat that decides how the table must be read

`print_gap_times` finds **27 holes** in `BTCUSDT/1m` (spacing > 4 × the 1-minute
median): **166 499 missing minutes**, 23.9 % of the 697 805-minute span, the
largest **5 338 560 s ≈ 61.8 days** (a second hole of 41.6 days follows).
Missing minutes and span are measurements of a cache revision (2026-09-30,
531 307 prints); the *share* has stayed 23.9 %. They
are concentrated in the last four months of the cache; the first year is
contiguous. A time bar resampled across such a hole prices a jump that never
traded, and at coarse sampling a handful of those returns dominates a kurtosis:

| sampling (BTCUSDT, 4 000 target) | excess kurtosis, as-is | hole-spanning returns dropped | flagged |
|---|---|---|---|
| time | 355.06 | **6.87** | 19 |
| dollar | 335.88 | **1.31** | 5 |
| volume | 332.17 | **1.05** | 6 |

The same effect at 500 target bars: time 33.63 → 3.11 with 11 flagged, dollar
18.27 → 0.05 with 1 flagged, volume 46.74 → 0.47 with 2 flagged.

Every table below is therefore given **twice**: as-is (the literal comparison),
and with the hole-spanning returns removed from *both* series by the same
rule (`gap_spanning_returns`, anchored on the shared print frame — a
per-series rule would flag activity droughts in a dollar series and stay silent
about them in a time series, which is not a comparison). The asymmetry is
stated, not hidden: the time control loses 10–19 returns, the activity series
0–6, because that is exactly how many of each sampling's returns straddle the
cache's broken tail.

---

## 4. Distribution table (time vs dollar vs volume bars)

`n` = returns, `exkurt` = excess kurtosis, `JB` = Jarque-Bera statistic (p-value
is the exact `chi2(2)` survival `exp(-JB/2)`; every as-is cell rejects normality
at p ≈ 0), `acf1` = lag-1 autocorrelation of log returns, `absacf1` = lag-1 of the
`|returns|` ACF, `mean|acf|` = mean of its lag-1…5 terms (volatility clustering).

**As-is (all returns).** BTCUSDT — thresholds 1.569 7e8 USDT / 1 396.27 base:

| sampling | n | skew | exkurt | JB | acf1 | absacf1 | mean\|acf\| |
|---|---|---|---|---|---|---|---|
| time | 3 888 | 10.1193 | 355.06 | 2.049e7 | +0.0221 | 0.1017 | 0.0756 |
| dollar | 3 887 | 9.8350 | 335.88 | 1.833e7 | −0.0626 | 0.1695 | 0.0935 |
| volume | 4 728 | 9.2702 | 332.17 | 2.180e7 | +0.0006 | 0.1053 | 0.0718 |

ETHUSDT: time exkurt 173.68 / JB 3.767e6; dollar 186.22 / 4.337e6; volume
219.47 / 6.206e6. SOLUSDT: time 342.79 / 1.701e7; dollar 329.64 / 1.579e7;
volume 399.02 / 2.648e7.

**Hole-spanning returns removed (both series).**

| symbol | sampling | n | skew | exkurt | JB |
|---|---|---|---|---|---|
| BTCUSDT | time | 3 869 | −0.1809 | 6.87 | 7 641 |
| BTCUSDT | dollar | 3 882 | −0.0218 | **1.31** | **280** |
| BTCUSDT | volume | 4 722 | −0.0501 | **1.05** | **218** |
| ETHUSDT | time | 2 963 | −0.0533 | 7.08 | 6 188 |
| ETHUSDT | dollar | 2 983 | −0.0635 | **0.92** | **107** |
| ETHUSDT | volume | 3 073 | −0.0222 | **0.92** | **108** |
| SOLUSDT | time | 3 444 | −0.3368 | 7.18 | 7 470 |
| SOLUSDT | dollar | 3 470 | −0.1573 | **1.70** | **433** |
| SOLUSDT | volume | 3 973 | −0.1520 | **1.14** | **230** |

Both samplings are still non-normal after the exclusion (JB p ≈ 0), but the
activity bars sit an order of magnitude closer to normal in every cell.

### Verdict per metric

`claim` = the plan's rule on the as-is numbers (JB **and** exkurt down,
both bootstrap intervals excluding 0, 400 circular-block draws, seed 0);
`claim†` = the same rule on the hole-excluded numbers.

| symbol | candidate | skew | exkurt | JB | acf1 | vol-clustering | claim | claim† |
|---|---|---|---|---|---|---|---|---|
| BTCUSDT | dollar | IMPROVED | IMPROVED | IMPROVED | WORSE | WORSE | **No** | **Yes** |
| BTCUSDT | volume | IMPROVED | IMPROVED | WORSE | IMPROVED | IMPROVED | **No** | **Yes** |
| ETHUSDT | dollar | WORSE | WORSE | WORSE | IMPROVED | WORSE | **No** | **Yes** |
| ETHUSDT | volume | WORSE | WORSE | WORSE | WORSE | IMPROVED | **No** | **Yes** |
| SOLUSDT | dollar | IMPROVED | IMPROVED | IMPROVED | IMPROVED | WORSE | **No** | **Yes** |
| SOLUSDT | volume | WORSE | WORSE | WORSE | IMPROVED | IMPROVED | **No** | **Yes** |

Δexcess kurtosis (candidate − time), 95 % circular block-bootstrap interval:

| cell | as-is | hole-spanning returns removed |
|---|---|---|
| BTCUSDT dollar | −19.18 [−441.93, +420.49] | **−5.56 [−7.36, −3.72]** |
| BTCUSDT volume | −22.89 [−451.16, +417.91] | **−5.83 [−7.57, −3.75]** |
| ETHUSDT dollar | +12.54 [−240.95, +255.04] | **−6.16 [−9.61, −3.39]** |
| ETHUSDT volume | +45.79 [−238.01, +277.33] | **−6.16 [−9.96, −3.29]** |
| SOLUSDT dollar | −13.15 [−410.41, +402.58] | **−5.48 [−8.91, −2.71]** |
| SOLUSDT volume | +56.22 [−417.11, +449.51] | **−6.05 [−9.03, −3.05]** |

**Reading.** As-is, the point estimates flip sign across symbols and every
interval is an order of magnitude wider than the point estimate: the literal
criterion passes nowhere at 4 000 bars. After removing the hole-spanning
returns, the direction is negative in **12 of 12** cells (6 of 6 here, 6 of 6 at
500 target bars) and the interval excludes 0 in **10 of 12** (all six here, four
of six at 500 bars) — a real effect *on this cache*, produced by the two
multi-week holes rather than by anything about how the clock partitions a
continuous market. Anyone quoting "dollar bars give thinner tails" from this
repository must quote the hole-excluded table and this caveat with it.

### Resolution sensitivity (`--target-bars 500`)

At 500 target bars (1099/1431/1233-minute time grids, ~380–440 bars, matched
counts) the as-is table moves the other way — dollar bars look much better
(BTCUSDT exkurt 18.27 vs 33.63) — and **one cell passes the literal criterion**:
SOLUSDT dollar bars, Δexkurt −36.84 [−52.03, −0.63], ΔJB −25 901
[−51 596, −4] (`claim_supported: true`). It is the cell where the dollar series
happens to contain **0** hole-spanning returns while the time control contains
11 — i.e. the plan's criterion is met exactly where the hole exclusion is
implicitly applied to one side. With the exclusion applied to both samplings the
same cell still supports a reduction (−2.04 [−3.71, −0.23]) while ETHUSDT dollar
bars do not at this resolution (−0.75 [−2.20, +0.29]); across the two
resolutions the hole-excluded criterion is supported in **10 of 12** cells and
the direction is negative in **12 of 12**.

---

## 5. The decisive experiment: the same credibility protocol on both samplings

Same symbol (BTCUSDT), same window, same feature contract
(`feature_schema_hash 1f30fded996d`, 54 columns, **0 near-constant columns** on
either sampling), same cost (`0.2500 %` round trip), same label
(±0.5 %, 4-bar horizon), same purged/embargoed K-fold protocol, same nested
per-fold threshold selection:

```
python tools/p6_volume_bars_experiment.py gate --symbol BTCUSDT --target-bars 4000
```

> The contract was **landing concurrently** (P6-B): the plan froze a 39-column v1
> list, the run above measured 54 columns with hash `1f30fded996d`. What matters
> here is that *both* samplings went through the identical pipeline, hash and
> column set — the comparison is contract-agnostic, and the hash is recorded so
> a later re-run can say whether the contract moved.

| sampling | bars | OOS rows | AUC | accuracy (majority) | Brier | net OOS | trades | t | PSR | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| time | 3 890 | 2 236 | **0.5361** | 0.5004 (0.5058) | 0.2584 | **−0.3276 %** | 1 790 | −7.600 | 1.1e-14 | **FAIL** |
| dollar | 3 889 | 2 662 | 0.4664 | 0.4996 (0.5143) | 0.2677 | −0.3335 % | 939 | −4.461 | 2.2e-08 | **FAIL** |

**Verdict: dollar bars did NOT improve the gate verdict — they are worse.** Both
samplings are refused on the same counts (AUC ≤ 0.55, net expectancy ≤ 0,
significance with the wrong sign). Dollar bars are **worse on the
calibration-independent AUC (−0.0697)** *and* **worse net of cost (−0.0059 pp)**
on the 2026-09-30 22:17 run; a repeat nine minutes later (same time-bar series, a
dollar series that had completed more prints) gave AUC **−0.0613** and net
**−0.0978 pp**. The net delta is negative in both, so the earlier "marginally
better net of cost (+0.0103 pp)" reading was a stale revision's noise, not a
property of the sampling — this is ≈0.33 % lost per trade either way, i.e.
roughly the round-trip cost with no direction. Substituting the clock changed how
many observations the sample holds, not the information the features carry.
`ml.enabled` stays `false`.

**This measurement moves with the cache.** Repeating the whole gate protocol at
three cache revisions during one session gave dollar AUC 0.5079 → 0.5014 →
0.4823 while the time-bar number stayed 0.5350 (its bars are unchanged by
appended prints that fall in the dropped final bin). Two further back-to-back
runs on 2026-09-30 (22:17 and 22:26) gave dollar AUC **0.4664 → 0.4748** with the
time row identical (0.5361) and the net delta **−0.0059 → −0.0978 pp**: the
dollar series is rebuilt from the 1-minute prints, so every appended print can
move it, while the time series' extra prints fall in the dropped final bin. The
signs that matter are stable in every revision measured — dollar AUC lower
(−0.0527 to −0.0697) and both samplings net-negative — while the individual
figure is a measurement of a revision, which is why the row counts above are part
of the record.

---

## 6. Market volume breadth

Built from one endpoint, `GET https://data-api.binance.vision/api/v3/ticker/24hr`.
Three series per observation: **total quote volume** (`Σ quoteVolume`), **up-share**
(share of pairs with `priceChangePercent > 0`), **concentration HHI**
(`Σ (vᵢ/Σv)²`; `1/HHI` reported as `effective_pairs`).

**Counting rules** (so "coverage" cannot be read two ways): `symbol_count` = every
entry returned; `pair_count` = entries passing the USDT filter (leveraged
`UP/DOWN/BULL/BEAR` tokens excluded); `usable_count` = pairs with
`quoteVolume > 0`; `coverage = usable_count / expected_pair_count` when the
documented universe size is supplied, else `usable_count / pair_count`.

### Staleness / TTL policy (all of it in one place)

| constant | value | meaning |
|---|---|---|
| `TICKER24H_TTL_S` | 300 s | an observation is *fresh* for 5 minutes; the rolling 24 h aggregate moves on the scale of minutes and a full-universe call costs ~1.9 MB |
| `MAX_STALE_MS` | 1 800 000 (30 min) | beyond this a cached value is still returned, but only as `is_stale=True`; a caller needing fresh numbers must treat it as unavailable |
| `FETCH_TIMEOUT_S` | 45 s | per-request socket timeout (the measured wall clock is 55–98 s — so a caller must also bound its own loop) |
| `REQUEST_ATTEMPTS` | 2 | one retry for a transient stall; errors are never cached |
| `MAX_CACHE_LINES` | 20 000 | append-only JSONL in `data/breadth/breadth.jsonl` (deliberately **not** `data/market/**`), atomically compacted past this |

### Measured availability

```
python tools/p6_volume_bars_experiment.py breadth --samples 6 --interval 20 \
    --out %TEMP%/p6c     # then: --from-cache re-measures the same series
```

6 live observations, `1790771537878 → 1790771985323` (**447 s** of wall clock),
fetch latency **55.5 / 57.8 / 60.5 / 76.0 / 89.0 / 97.5 s**:

| field | n | variance | acf1 | acf1 of Δ | min | max | \|acf1\| < 0.99 | variance > 0 |
|---|---|---|---|---|---|---|---|---|
| `total_quote_volume` | 6 | 2.172e15 | 0.9713 | −0.7545 | 7.750e9 | 7.888e9 | **yes** | yes |
| `up_share` | 6 | 5.819e-4 | 0.4858 | −0.5437 | 0.4749 | 0.5429 | **yes** | yes |
| `hhi` | 6 | 1.417e-7 | 0.6608 | +0.2293 | 0.1242 | 0.1253 | **yes** | yes |

* **Coverage: 676 / 676 = 100 %** of the USDT universe the endpoint reports
  (3 723 symbols returned; up-share ranged 0.4749–0.5429 across the six samples,
  so the series is not a constant). Measured against the plan's documented 496
  pairs the same numbers give **136.3 %** — the expectation is stale, not the
  data short. The plan's "≥ 95 % of 496" row therefore passes for a different
  reason than it was written for, and is reported with that qualification.
* **Non-degeneracy: passes as measured, weakly.** All three lag-1
  autocorrelations are inside `< 0.99`, but n = 6, so each estimate carries a
  standard error around 0.4. The structural expectation runs the other way: a
  24 h rolling quantity overlaps itself by ~99.9 % between samples a minute
  apart, so a longer sample would be expected to sit nearer 1. Quote the
  criterion as passed-on-this-sample, not as a property of the series.
* **Causal labelling: verified, and that is all that can be verified.**
  `is_causal` confirms `as_of_ms` is non-decreasing and that no observation's
  `request_started_ms` post-dates its own label. The endpoint has no history, so
  a past breadth value **cannot be reconstructed** — the cache is a forward
  record and `replay(upto_ms)` only filters rows by their label.
* **Unreachable endpoint: no fabricated value.** `fetch_breadth` returns `None`
  and `BreadthCache.refresh` returns `("unavailable", None)` with an empty cache;
  a stale-but-present value is returned only with `is_stale=True`, `stale_ms` and
  `source="cache"` set. A torn JSONL line is skipped, never guessed.

Breadth is a **library plus evidence**: no production gate, no ML feature, no
config switch.

---

## 7. What did **not** improve (explicit)

1. **The gate.** Both samplings refused; dollar AUC 0.0613–0.0697 *lower* than the
   time model's (0.0527 at the archived revision), net expectancy negative on
   both, `ml.enabled` unchanged. This is the decisive result and it is negative.
2. **The plan's literal distribution criterion.** 1 of 12 cells passes (SOLUSDT
   dollar bars at 500 target bars); at 4 000 target bars, zero cells pass.
3. **Volatility clustering.** Worse for dollar bars on BTCUSDT
   (mean |ACF| 0.0935 vs 0.0756) and ETHUSDT (0.0741 vs 0.0596), and for volume
   bars on ETHUSDT/SOLUSDT in the as-is table.
4. **Volume bars as a general answer.** They win on Jarque-Bera nowhere except
   the hole-excluded table, and lose on it in 3 of 6 as-is cells.
5. **Breadth's non-degeneracy** is established on 6 samples only, and the
   "496 pairs" expectation in the frozen plan is out of date (676 measured).
6. **Bar `volume` is not bit-identical** between the streaming and batch forms
   (≤2.5e-15 relative, summation order); boundaries and prices are exact.

What *does* replicate — a hole-excluded excess-kurtosis reduction of ≈5.5–6.2
units (CI excluding 0 at 4 000 bars in all six cells) — is reported in §4 as a
property of this cache's holes. It changes no gate verdict and enables nothing.

## 8. Reproducibility

```
python tools/p6_volume_bars_experiment.py all --symbols BTCUSDT ETHUSDT SOLUSDT \
    --target-bars 4000 --n-boot 400 --samples 6 --interval 20 --out %TEMP%/p6c
```

* **Determinism, three ways.** (a) Every `bars` cell reports
  `determinism: {dollar: true, time: true, volume: true}` — an independent
  rebuild of all three series from the same frame is bit-identical. (b) Two
  back-to-back `bars` runs and two `gate` runs at the same cache revision, and
  two `breadth --from-cache` runs, produce JSON that is **byte-identical** once
  the volatile fields (`generated_by`, `argv`, `started_at`, `finished_at`,
  `build_seconds`) are removed, with matching per-sampling return-series
  SHA-256 (`sha256_16`): at revision 531 248, BTCUSDT dollar
  `ce5d749667d87938` / time `f0e8aaf7975d4aa3` / volume `bb279a7b90aeb2d1`,
  ETHUSDT `bc7c84141f510fe4` / `b53a660824474b24` / `7b0d7d57bebf8415`, SOLUSDT
  `fd8b5c292f31e99f` / `74609e412829c0a4` / `85b67435beb6d1f8`. (c) Across cache
  revisions the *numbers* move because the input moved: between revisions
  531 244 and 531 248 the time-bar SHA was unchanged (the appended prints fall
  in the dropped final bin) while the dollar and volume SHAs changed (they
  completed an extra bar). Every stochastic step takes `--seed` (default 0).
  `breadth` *live* necessarily polls new data; that is why `--from-cache`
  exists.
* **Bounded runtime.** `bars` ≈ 1 s/symbol to build; `gate` ≈ 1 min; `breadth` is
  bounded by `--samples × --interval` and `--max-seconds` (default 900 s; the
  6-sample run above took 447 s of wall clock).
* **Reads only cached parquet** — never `data/binance_trader.db`,
  `data/models/**` or `strategies/**`.
* **Tests.** `tests/test_volume_bars.py` (48) + `tests/test_breadth.py` (24) are
  synthetic-only: no network, no live-cache read, no pinned market statistic (the
  guard in `tests/test_measured_threshold_policy.py` passes with the live-cache
  reader set unchanged). The cache revision in §2 is quoted as a measurement
  timestamp, not asserted.

## 9. Residuals (documented, not fixed here)

* The time control's interval is a **specification** solved on the whole window's
  print density (`match_time_interval`); it is constant across the series and
  uses no prices, but unlike the notional threshold it is not a warm-up-only
  quantity. A warm-up-only solve would leave the two series with different bar
  counts, which is the artefact the solve exists to remove.
* The cache holes are a **data defect** (`data/market/**` is written by the
  running app): 23.9 % of BTCUSDT's 1-minute span is absent, in 27 holes up to
  61.8 days. P6-C works around them in the measurement; re-downloading the
  missing history is P6-B/`scripts/download_history.py` territory
  (`--merge`), not this phase's.
* The block bootstrap draws the two samplings **independently** (two samplings of
  one price process, not paired observations), so its interval is sampling
  uncertainty for a difference, not a paired test.
* Breadth's HHI is computed on the USDT subset only; a caller with
  `exchangeInfo` should pass its own `symbol_filter` to pin the universe exactly
  (the default excludes leveraged-token suffixes, nothing else).
* `dollar_bars`/`volume_bars` are not wired into the feature contract, the
  backtest feeder or any gate — by design (§1 of the frozen plan). Enabling them
  would be a P6-D/E decision taken against the numbers above, and those numbers
  do not support it.
