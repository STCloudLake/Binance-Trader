# P9 — the fill convention: what "one bar of execution latency" costs

**Status:** measured. The convention is now **configurable** (`backtest.fill_convention`),
the shipped default stays **`close`** (unchanged behaviour, bit-identical to the
pre-P9 engine), and the measured cost of the conventional alternative
(`next_open`) is on the table below. **The default is not flipped** — that is the
owner's call, and §7 states what flipping it would invalidate.

```powershell
# every number in this document
python -m pytest tests/ -q -p no:cacheprovider
python -m compileall -q app core web db scripts tools
python tools/p9_fill_convention_measure.py --hand-check          # the by-hand case
python tools/p9_fill_convention_measure.py --verify-mechanism    # the same claim on real parquet
python tools/p9_fill_convention_measure.py --out data/p9_evidence/fill_convention.json
python tools/p9_champion_regate.py --out data/p9_evidence/champion_regate.json
python data/p9_evidence/p8_fill_convention.py --fill-convention close     --out data/p9_evidence/p8_close.json
python data/p9_evidence/p8_fill_convention.py --fill-convention next_open --out data/p9_evidence/p8_next_open.json
python tools/p9_fill_convention_identity.py --tree <HEAD worktree> --data-dir data --out %TEMP%\p9_id_head.json
python tools/p9_fill_convention_identity.py --tree .               --data-dir data --out %TEMP%\p9_id_work.json
```

---

## 1. The finding being quantified

The engine decides and fills on the **same bar's close**:

* the entry signal is evaluated on the bar at `ts` — the indicators are sliced to
  `≤ ts` (`core/backtest/engine.py`, `df_primary = _get_cached_df(...)`) and the
  fill price is `float(df_primary["close"].iloc[-1])`, i.e. the close of the very
  bar the signal came from;
* every exit is priced the same way: the triggering bar's close
  (`indicator` / `max_hold` / `reduce`) or the barrier level (`stop_loss` /
  `tp_*`), again on the bar at `ts`.

That is **zero execution latency**: it assumes you can see a close and trade at
exactly that price. The conventional treatment — signal at the close of `ts`,
fill at the **open of `ts+1`** — is now selectable:

| value | meaning |
|---|---|
| `close` **(shipped default)** | the signal bar's own close (or the barrier level) — today's behaviour |
| `next_open` | signals unchanged; the fill price, **entry and exit**, is the `open` of the bar one row later on the same series |

### The mechanism, computed by hand

`python tools/p9_fill_convention_measure.py --hand-check` prints a three-bar
frame whose prices are chosen so both numbers are arithmetic:

| bar (1h, close-time index) | open | close |
|---|---|---|
| `2026-01-01 00:59:59.999` (**decision bar**) | 98.0000 | **100.0000** |
| `2026-01-01 01:59:59.999` | **101.5000** | 103.5000 |
| `2026-01-01 02:59:59.999` | 103.0000 | 104.0000 |

* `close` fill = `close[0]` = **100.0000**
* `next_open` fill = `open[1]` = **101.5000**
* difference = `(101.5 − 100.0) / 100.0 × 10 000` = **+150.00 bps**
* on the **last** bar the helper returns `None` (no following bar) — the engine
  then refuses the entry / falls back on an exit, never guesses a price.

The same claim on data nobody chose — `--verify-mechanism` (first trade of
`strategies/ga_champion_1780713878.yaml`, 1h, 2025-05-01→2025-06-01):

```
symbol              : ETHUSDT        side: short
decision_bar        : 2025-05-01 00:59:59.999
parquet_close       : 1797.75      close_arm_entry     : 1797.75
parquet_next_open   : 1797.76      next_open_arm_entry : 1797.76
difference_bps      : 0.0556
```

The two arms' entry prices **are** the parquet's `close` at the decision bar and
the parquet's `open` of the next row. The mechanism is confirmed twice (synthetic
frame + real cache) before any aggregate number is read.

---

## 2. Deliverable 1 — the measurement

**Command** (bounded: **362.0 s**, single process, no network):

```powershell
python tools/p9_fill_convention_measure.py --out data/p9_evidence/fill_convention.json
#   symbols=['BTCUSDT','ETHUSDT','SOLUSDT']  window=2025-05-01 -> 2025-07-01
#   timeframes 1h + 15m   seed=20261003   genomes=3
#   cost model: taker_fee_pct=0.04, spreads BNBUSDT 0.03 / BTCUSDT 0.01 /
#   ETHUSDT 0.02 / SOLUSDT 0.03 / XRPUSDT 0.04, default 0.03, live order book OFF
```

* **Arms** — `coverage is not a single strategy`: three shipped champions
  (`ga_champion_1780713878` native 1h, `ga_champion_1790844776` native 15m+4h,
  `ga_champion_1780642613` native 5m+1h) **plus** three random genomes drawn from
  the GA's own generator under one fixed seed (`random.seed(20261003)` →
  `random_chromosome` → `chromosome_to_strategy`). `strategies/**` is read-only.
* **Timeframes** — 1h and 15m. Each arm is pinned to a **single** timeframe so
  the per-timeframe rows isolate the bar size; a champion's other timeframes act
  as filters in its native run and never price a fill.
* **The GA's own costs** — the engine is driven exactly like a GA evaluation
  (`engine_mode='legacy'`, ML off, `use_live_spread=False`, `isolated_eval_kwargs()`
  = per-genome slots + per-genome cash ledger). Nothing is priced from today's
  order book.

### What "average difference in fill price (bps)" means here

Two families, because they answer different questions (`_fill_diffs`):

* **level** — `(P_next_open − P_close) / P_close × 10 000`, no direction. Its mean
  is ≈ 0 **by construction**: `open[k+1]` vs `close[k]` is one bar-boundary gap
  with no systematic sign. This is the evidence that the convention is *not* a
  price-level bias.
* **adverse** — the same difference signed by the trade's own side (`+1` long,
  `−1` short), so **positive = `next_open` filled worse for that trade** (a long
  paid more, or received less on the exit). The round-trip sum is the per-trade
  cost of one bar of latency, and this is the number that shows up in the P&L.

### 1h

| arm | trades | total return % | Sharpe | max DD % | level Δ entry bps | \|round trip\| bps | **adverse round trip bps** |
|---|---|---|---|---|---|---|---|
| `champion:ga_champion_1780642613` | 1649 → 1649 | −3.780 → −4.280 | −7.470 → −7.730 | 3.82 → 4.32 | +0.01 | 3.84 | **+1.42** |
| `champion:ga_champion_1780713878` | 1978 → 1977 | −2.910 → −3.490 | −7.990 → −8.650 | 2.95 → 3.52 | 0.00 | 4.61 | **+1.51** |
| `champion:ga_champion_1790844776` | 103 → 103 | +0.330 → −0.110 | +1.280 → −0.330 | 0.57 → 0.87 | −0.04 | 38.37 | **+20.93** |
| `genome_s20261003_0` | 250 → 250 | +0.670 → +0.430 | +1.550 → +0.700 | 0.56 → 0.98 | −0.00 | 43.77 | **+6.09** |
| `genome_s20261003_1` | 170 → 170 | −0.050 → −0.440 | −0.260 → −2.250 | 0.58 → 0.93 | −0.03 | 18.80 | **+12.70** |
| `genome_s20261003_2` | 984 → 984 | −1.550 → −1.540 | −4.540 → −3.880 | 1.95 → 2.25 | +0.02 | 8.03 | **−0.06** |

### 15m

| arm | trades | total return % | Sharpe | max DD % | level Δ entry bps | \|round trip\| bps | **adverse round trip bps** |
|---|---|---|---|---|---|---|---|
| `champion:ga_champion_1780642613` | 6446 → 6444 | −12.610 → −12.790 | −24.030 → −23.850 | 12.66 → 12.84 | +0.01 | 0.43 | **+0.16** |
| `champion:ga_champion_1780713878` | 6139 → 6137 | −11.890 → −11.960 | −32.630 → −32.040 | 11.91 → 11.99 | +0.01 | 0.90 | **+0.09** |
| `champion:ga_champion_1790844776` | 265 → 265 | −0.740 → −1.270 | −2.230 → −3.400 | 0.91 → 1.39 | +0.02 | 14.96 | **+10.32** |
| `genome_s20261003_0` | 471 → 471 | −0.280 → −0.520 | −0.870 → −1.210 | 1.18 → 1.62 | +0.01 | 17.81 | **+2.56** |
| `genome_s20261003_1` | 459 → 459 | −0.420 → −0.710 | −2.530 → −3.810 | 0.74 → 0.97 | +0.03 | 4.41 | **+3.12** |
| `genome_s20261003_2` | 2722 → 2722 | −4.760 → −4.570 | −11.140 → −9.680 | 4.87 → 4.72 | +0.01 | 1.60 | **−0.31** |

`a → b` is `close → next_open`.

### Aggregate verdict

| | 1h | 15m |
|---|---|---|
| mean level Δ **entry** | **−0.01 bps** | **+0.01 bps** |
| mean level Δ **exit** | +0.79 bps | −0.03 bps |
| mean **\|round trip\|** | 19.57 bps | 6.68 bps |
| mean **adverse** entry | +0.01 bps | +0.01 bps |
| mean **adverse** exit | +7.09 bps | +2.65 bps |
| mean **adverse round trip** (per trade) | **+7.10 bps** | **+2.66 bps** |
| mean Δ total return | **−0.357 pp** | **−0.187 pp** |
| mean Δ Sharpe | **−0.785** | **−0.093** |
| mean Δ max drawdown | **+0.407 pp** | **+0.210 pp** |
| arms with a worse return | **5 / 6** | **5 / 6** |
| arms with a worse Sharpe | 5 / 6 | 3 / 6 |
| trades that exist in only one arm (window end) | 1 | 6 |

(The one arm that improves at both timeframes is `genome_s20261003_2`, by
+0.010 pp at 1h and +0.190 pp at 15m — within the cost-model noise of a single
arm, and the reason the verdict is reported as 5/6 rather than "unanimous".)

**Reading it.**

1. **The level bias is zero; the timing cost is not.** The mean signed fill-price
   difference is ≈ 0 bps at both timeframes (−0.01 bps for a 1h entry, +0.01 for a
   15m entry) — the convention does not shift prices up or down. The **adverse**
   per-trade number is where the money is: **+7.10 bps at 1h and +2.66 bps at
   15m**, and the sign is *against the trade* because a signal that fires after a
   move is one bar late to a move that continues.
2. **`next_open` is worse in 5 of the 6 arms at each timeframe** (10/12 arm ×
   timeframe pairs lose return) and deepens drawdown in **11/12**; the single
   improving arm moves by +0.01 / +0.19 pp, i.e. inside the noise band.
3. **The per-trade penalty grows with the bar length** (7.10 bps at 1h against
   2.66 bps at 15m, a 2.7× ratio) — which is the mechanism: one 1h bar moves
   further than one 15m bar. The **aggregate** return hit in this window is
   *not* larger at 15m (15m trades 3.2× more often for ~1/3 of the per-trade
   cost, so the two roughly cancel and 1h happens to be worse here). The
   pre-stated expectation "the bias should grow as the timeframe shortens" is
   therefore **not confirmed in aggregate on this sample**; what is confirmed is
   "the per-trade cost of one bar of latency grows with the bar length", and that
   the direction of the effect is uniformly negative.
4. **Absolute size.** On a 2-month window with a 3-symbol basket, flipping the
   convention moves a champion's total return by 0.04–0.54 pp and its Sharpe by
   0.07–2.21 (the 4h-trading champions move most). It is not noise: it is
   one-directional, and it is bigger than the DSR differences the publication gate
   turns on (§4).

### How the exits were treated (the part that makes the comparison meaningful)

**The same one-bar shift is applied to every exit, and every exit trigger stays
exactly where it was.** The trigger is still observed on the bar at `ts` with the
same close/level comparison the `close` arm uses — *signals do not move* — and
only the fill price changes:

| exit reason | `close` fill | `next_open` fill |
|---|---|---|
| `indicator` | the triggering timeframe's bar close | that timeframe's **next open** |
| `stop_loss` / `tp_*` | the barrier **level** | the position timeframe's **next open** |
| `max_hold` | the position timeframe's bar close | its **next open** |
| `reduce` | the triggering timeframe's bar close | its **next open** |
| `end_of_backtest` | the window's last close | the last close (no next bar exists — counted, §3) |

Shifting only the entries would have measured half a round trip and left the exits
latency-free, which is exactly the asymmetry that makes a latency comparison
meaningless. The price of that choice is stated rather than hidden: under
`next_open` a stop **no longer fills at the barrier level**, so an SL/TP exit
differs by more than a pure price shift. The exit mix is therefore printed per arm
(`rows[].exit_reason_mix`); e.g. at 1h `ga_champion_1790844776` exits 51/103 on
`stop_loss` (the arm with the largest adverse cost, +20.93 bps), while
`ga_champion_1780713878` exits 1 826/1 978 on `indicator` (+1.51 bps). The exits
that move most are the ones where the level fill is replaced by the next open.

### How the window end was handled (explicitly, never silently)

Every frame the feeder hands out is trimmed to `date_end`, so the **last decision
bar has no following bar**. Two distinct, counted cases:

* an **entry** whose next bar is outside the window is **refused** — no position
  is opened — and counted in `unfilled_entries` (`ga_champion_1780713878` at 1h:
  1978 trades → 1977, `unfilled_entries=1`);
* an **exit** with no next bar is priced at the decision bar's close and counted
  in `window_end_fallback_fills`, broken down by reason
  (`{"indicator": 1, "end_of_backtest": 2}` in that same arm).

Both travel in `metrics["fill_convention_accounting"]` and are printed by the
tool. Total exposure of this rule: **1 trade at 1h and 6 at 15m** out of 5 134 /
16 502 — it is a boundary rule, not the effect.

---

## 3. Deliverable 2 — configurable, default unchanged

### The key

```yaml
backtest:
  # close = the signal bar's own close (SHIPPED DEFAULT, bit-identical to pre-P9)
  # next_open = signal unchanged, fill = the open of the bar one row later
  fill_convention: close
```

* validated **at config load** (`app/config.py`) through
  `core/backtest/fill_convention.py::parse_fill_convention`, which raises the
  NAMED `UnknownFillConventionError` naming the bad value and the valid ones for
  anything else (including a non-string). The key being **absent** means `close`,
  so a config written before P9 keeps the historical behaviour exactly.
* threaded through `BacktestEngine.run` / `run_with_exit_evaluation`
  (`fill_convention=` kwarg overrides the config for one run — the GA path passes
  no kwarg and therefore follows the config).
* **"one bar" with several timeframes loaded** is one row of the *fill
  timeframe's own* series, never one row of the feeder's union grid: the
  **primary (shortest configured) timeframe** prices an entry; the timeframe whose
  close supplied the exit price prices that exit (the position's own timeframe for
  SL/TP/max-hold, the triggering interval for an indicator exit). A 4h filter
  never prices a fill.
* the choice is recorded in **the run's result**: `result["fill_convention"]` and
  `metrics["fill_convention"]` / `metrics["fill_convention_accounting"]`
  (convention, refused entries, window-end fallbacks by reason). The GA champion
  provenance `eval` block gained `fill_convention` as well, so a champion YAML
  says which convention produced its fitness.

### The `close` arm is bit-identical — measured, not asserted

Same harness, two trees, byte comparison of **trades (timestamps, prices,
quantity, PnL, cost), every per-genome equity point, and a fixed metrics digest**
(`tools/p9_fill_convention_identity.py`):

```powershell
python tools/p9_fill_convention_identity.py --tree C:\Users\23302\AppData\Local\Temp\p9_head_worktree --data-dir data --out %TEMP%\p9_id_head.json
python tools/p9_fill_convention_identity.py --tree .                                            --data-dir data --out %TEMP%\p9_id_work.json
```

| run | tree | digest (sha256 of the payload) | trades | equity points |
|---|---|---|---|---|
| baseline | `git worktree` @ `cafdc1d` (HEAD, no `fill_convention` key at all) | `0d454ec490a45e030e32e901891de2e4825fba8a6d20d15c95f7abe1da072fdc` | 39 | 648 |
| working | this tree, `fill_convention: close` | `0d454ec490a45e030e32e901891de2e4825fba8a6d20d15c95f7abe1da072fdc` | 39 | 648 |

The payloads are **byte-identical** (real cached parquet, 2026-01-05→2026-02-01,
BTCUSDT 1h, `sma`/`rsi` rules with `risk_exit`, isolated per-genome ledger,
`benchmark_mode: none`). The same comparison runs as a standing test on a
synthetic market (304 trades, digest
`1dbeb98857238263c7da185c8a69a3f868eeaa4bc9df5d99b280656f47eaf8a8` in both trees),
so it cannot rot when the cache changes.

### The hybrid engine: refused by name, not mislabelled

`EventDrivenExecutor` prices every fill from the decision bar's close and has no
fill seam. It was **not** extended (that is a separate execution model, with the
hybrid/legacy parity contract pinned by tests), so:

* `backtest.engine_mode: hybrid` + `next_open` → **`FillConventionUnsupportedError`**
  naming the reason and the fix;
* `auto` + `next_open` (which would have selected hybrid for ≥3 strategies) →
  warning + the **legacy** engine, so the requested convention really is priced;
  the fallback is recorded in `fill_convention_accounting["engine_fallback"]`.

This is the same contract `reduce_conditions` already has with the hybrid engine.

---

## 4. Deliverable 3 — the documented headline numbers, re-evaluated

### 4.1 The champions behind the gate verdicts

**Command:** `python tools/p9_champion_regate.py --out data/p9_evidence/champion_regate.json`
(**292.2 s**). Same windows, symbols, costs and benchmark mode (`exposure_matched`)
as the artefacts, scored by the GA's own path (`stats_from_trades` →
`score_stats`, `n_trials` from each champion's provenance) and gated by the
production `GAStrategyEvolver._publication_decision`.

**Reproduction fidelity (read this before the deltas).** The champion YAML is a
*decoded strategy*, not the chromosome, so the genome-complexity penalty and the
P6-D executability model cannot be recomputed. The `close` arm reproduces the
recorded numbers **exactly** on every quantity except `fitness`, which is short by
exactly that missing penalty (recorded → reproduced): `41.8192 → 57.8192` (16.00),
`−5.9285 → 8.4715` (14.40), `−0.5909 → 7.3730` (7.96). Sharpe (`9.0101`, `3.2768`,
`3.9649`), trade count (`382`, `171`, `127`), train DSR (`0.0926`, `−0.1646`),
exposure-matched `alpha_vs_benchmark_pct` (`5.4481`, `1.8152`, `2.0372`) and
validation DSR (`−0.3855`, `0.0`) all match the recorded values (the only
exception: `1790867208`'s train DSR reproduces as `−0.1412` against a recorded
`−0.1393`). **The delta between the two conventions is therefore the readable
number**, since the missing terms are identical in both arms.

| champion (GA config) | arm | fitness | Sharpe | trades | train DSR | `alpha_vs_benchmark_pct` | validation Sharpe | validation DSR | gate |
|---|---|---|---|---|---|---|---|---|---|
| `1790844776` (A: seed 20261001, n_trials 690) | recorded | 41.8192 | 9.0101 | 382 | 0.0926 | +5.4481 † | 5.6903 | −0.3855 | **not published** |
| | `close` | 57.8192 | 9.0101 | 382 | 0.0926 | +5.4481 | 5.6903 | −0.3855 | not published |
| | `next_open` | 57.7183 | 8.9374 | 382 | 0.0888 | +5.4137 | 5.6847 | −0.3858 | not published |
| | **Δ** | **−0.1009** | **−0.0727** | 0 | **−0.0038** | **−0.0345 pp** | −0.0056 | −0.0003 | — |
| `1790855974` (B: seed 20261002, n_trials 930) | recorded | −5.9285 | 3.2768 | 171 | −0.1646 | +1.8152 | −1.8065 | 0.0 | **not published** |
| | `close` | 8.4715 | 3.2768 | 171 | −0.1646 | +1.8152 | −1.8065 | 0.0 | not published |
| | `next_open` | 3.5504 | 1.3732 | 171 | −0.2642 | +1.3389 | −2.9774 | 0.0 | not published |
| | **Δ** | **−4.9211** | **−1.9036** | 0 | **−0.0996** | **−0.4764 pp** | **−1.1709** | 0.0 | — |
| `1790867208` (B resumed: 32 gens, n_trials 1570) | recorded | −0.5909 | 3.9649 | 127 | −0.1393 | +2.0372 | −1.8065 | 0.0 | **not published** |
| | `close` | 7.3730 | 3.9649 | 127 | −0.1412 | +2.0372 | −1.8065 | 0.0 | not published |
| | `next_open` | 1.3312 | 1.7517 | 127 | −0.2571 | +1.4892 | −2.9774 | 0.0 | not published |
| | **Δ** | **−6.0418** | **−2.2132** | 0 | **−0.1159** | **−0.5479 pp** | **−1.1709** | 0.0 | — |

† `alpha_vs_benchmark_pct` is not stored in that champion's own `fitness_components`
(its `rejection_reasons` cite the `buy_hold` alpha `−12.1206`); the value shown is
the one recorded for it in `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md` under
`exposure_matched` (`+5.4481`), and it is reproduced to the digit by the `close`
arm — which is what makes the column trustworthy.

**Does any gate verdict flip? No.** All three champions were rejected before and
are rejected under `next_open`, and the **set of failing criteria is identical**
in every case (`validation_dsr ≤ 0` for A; `dsr ≤ 0` + `validation_sharpe ≤ 0` +
`validation_dsr ≤ 0` for both B champions). What changes is the *margin*: the DSR
and the validation Sharpe get worse by 0.004–0.116 and 0.006–1.171 respectively,
and the exposure-matched alpha loses 0.03–0.55 pp (a 0.6–27 % relative cut) — it
stays positive for both B champions, so the documented sentence "changing the
benchmark fixed the method but did not let the strategy through" survives the
convention flip unchanged.

### 4.2 The P8 three-arm comparison (hold / vol-targeted hold / champions)

Re-runnable, and re-run — but with one **honest caveat that changes what can be
compared**.

The recorded P8 run (`docs/overhaul/P8_BETA_HARVEST_EVIDENCE.md`, 454.3 s) used a
**nine-symbol basket**. `tools/p8_beta_harvest_measure.py` builds its basket from
whatever `data/market/**` currently holds, and the cache has moved since (the tool
now prints `excluded ZECUSDT: partial 1h coverage` and runs a **five-symbol**
basket: BNB, BTC, ETH, SOL, XRP). The recorded H / V rows therefore **cannot be
reproduced** from the current cache — not because of the fill convention, but
because the basket is a different basket. `data/p8_evidence/p8_beta_harvest.json`
still holds the recorded run.

So the convention was measured the only way that isolates it: **the same
unchanged P8 tool, run twice on the current cache, once per convention**
(`data/p9_evidence/p8_fill_convention.py` sets the convention **in memory**; the
tool's own sha256_16 `a9950a450220bf2b` is unmodified, and `config/config.yaml`
hashes before == after inside each run). Both runs are the full grid
(`--grid full`, default): **358.1 s** (`close`) and **363.2 s** (`next_open`).

Same cache, same basket (5 symbols), same costs — `close` vs `next_open`:

| arm | W0 | W1 | W2 | W3 | rows differing between the two conventions |
|---|---|---|---|---|---|
| H | +42.41 % | −28.35 % | −14.50 % | +10.62 % | **0** |
| Hreb | +41.06 % | −28.13 % | −14.32 % | +10.86 % | **0** |
| V | +32.98 % | −24.65 % | −5.57 % | +17.84 % | **0** |
| V<=1 | +29.34 % | −22.00 % | −4.47 % | +9.52 % | **0** |
| Vleg | +32.41 % | −24.81 % | −6.51 % | +14.33 % | **0** |
| **S (champions, production engine, 1h)** | **−3.39 % → −4.29 %** | **−2.29 % → −3.14 %** | **−1.31 % → −2.27 %** | **−4.06 % → −4.97 %** | **4 of 4** |

S-arm detail (the only arm the convention can reach):

| window | S return `close` → `next_open` | Δ pp | S Sharpe | Δ Sharpe | S max DD % | S trades |
|---|---|---|---|---|---|---|
| W0 | −3.3888 → −4.2892 | **−0.9004** | −11.6648 → −12.0320 | −0.367 | 3.4259 → 4.3293 | 12 957 → 12 943 |
| W1 | −2.2904 → −3.1442 | **−0.8538** | −8.7114 → −9.5470 | −0.836 | 2.4125 → 3.3184 | 10 994 → 10 987 |
| W2 | −1.3074 → −2.2740 | **−0.9666** | −2.7114 → −3.8444 | −1.133 | 2.1136 → 3.1017 | 10 550 → 10 547 |
| W3 | −4.0555 → −4.9711 | **−0.9156** | −15.3768 → −13.7764 | +1.600 | 4.3239 → 5.2189 | 14 178 → 14 172 |

**What this says.** The five beta arms are **bit-identical** between the two runs
(20 of 20 rows, every metric), which is the proof that the convention cannot reach
them — they never call the engine. The only arm it touches is the champions, and
there it costs **0.85–0.97 pp of return in every window** and deepens the
drawdown in all four — with the same sign as the §2 measurement, on a different
window set, a different basket and a different (1h, 7-champion composite) setup.
The one disagreement is W3's Sharpe, which *improves* while the return falls.

**P8 verdict:** unchanged — `T1 false / T2 false / T3 false` →
**"leave `risk.vol_targeting.enabled: false`"** in both conventions (and, for what
it is worth, also in the recorded nine-symbol run).

**Comparison with the recorded table** (`docs/overhaul/P8_BETA_HARVEST_EVIDENCE.md`
§3): the recorded S returns were −4.07 / −2.46 / −1.59 / −4.91 % on the nine-symbol
basket; the current cache gives −3.39 / −2.29 / −1.31 / −4.06 % under `close`. The
gap is the basket, not the convention — stated rather than papered over.

---

## 5. Tests

`tests/test_fill_convention.py` — **14 tests**, no network, no real-cache
dependency except the identity harness's short run:

| test | pins |
|---|---|
| `test_the_two_conventions_and_the_code_default` | `FILL_CONVENTIONS`, the code default `close`, key-absent ⇒ `close`, case/space tolerance |
| `test_an_unknown_convention_is_refused_by_name` | `UnknownFillConventionError` for `nextopen`/`open`/`next`/`""`/non-string |
| `test_the_shipped_key_is_close` | `config/config.yaml` ships `close` and `Config.load` agrees |
| `test_an_unknown_convention_is_rejected_at_config_load` | a temp `config.yaml` with `next-bar-open` raises **at load** |
| `test_next_bar_open_is_the_following_row_and_none_at_the_window_end` | the primitive on a hand-computable frame (+23.81 bps); last bar ⇒ `None`; mid-bar timestamp resolves to the bar at/before it |
| `test_next_open_shifts_entry_and_exit_by_exactly_one_bar` | engine-level: entry `close[k] → open[k+1]`, exit `close[k+1] → open[k+2]`, same decision bar, sizing follows the fill |
| `test_the_close_arm_reports_zero_accounting` | `close` never refuses/falls back (all counters 0) |
| `test_an_entry_on_the_last_window_bar_is_refused_and_counted` | window-end entry: 1 trade at `close`, 0 at `next_open`, `unfilled_entries == 1` |
| `test_an_exit_on_the_last_window_bar_falls_back_and_is_counted` | window-end exit: priced at the last close, `window_end_fallback_by_reason["max_hold"] == 1` |
| `test_the_result_and_the_metrics_name_the_convention` (×2) | provenance field on the result **and** in the metrics |
| `test_the_config_key_is_honoured_without_the_explicit_kwarg` | the GA path (no kwarg) follows the config, and really shifts |
| `test_the_hybrid_engine_never_prices_a_next_open_fill` | explicit `hybrid` raises; `auto` falls back to legacy, records it, and shifts |
| `test_close_is_bit_identical_to_the_pre_p9_worktree` | the `git worktree` byte comparison of §3 |

---

## 6. Suite counts and hashes

**Suite** (both runs on the final tree):

```powershell
python -m pytest tests/ -q -p no:cacheprovider
#   1542 passed, 4 warnings in 292.47 s
#   1542 passed, 4 warnings in 296.18 s
python -m compileall -q app core web db scripts tools     # exit 0
python scripts/regen_route_baseline.py --check            # 121 / 121, added 0, removed 0
```

Baseline at `cafdc1d` was **1528 passed / 0 failed**; this change adds
`tests/test_fill_convention.py` (**14 tests**) and touches no suite count
otherwise → 1542. Green on two consecutive runs.

Three existing `git worktree` byte-identity pins needed their **scope** narrowed by
exactly the new keys (never their values): `tests/test_p7_orchestrator.py`
(`metrics_keys` without the two P9 metric keys), `tests/test_p7_symbol_mode.py`
(`fill_convention` scrubbed from the champion `eval` provenance block) and
`tests/test_ga_benchmark_mode.py` (same, for the provenance payload). Each edit
carries a comment saying why; every pre-existing key is still compared, and all
three tests still pass — i.e. the pre-P9 outputs really are unchanged.

**`sha256_16` per file (before → after), recorded in this session:**

| file | before | after |
|---|---|---|
| `config/config.yaml` | `95abf9fd7186fe25` | `0d51ae38801770d2` |
| `app/config.py` | `02dcee69f37756fb` | `3be3631f625bda34` |
| `core/backtest/engine.py` | `41e632b35d1e4f43` | `6bcd09bfe27a91e9` |
| `core/backtest/fill_convention.py` | *(new)* | `61002b4518faba46` |
| `core/ga/evolver.py` | `e8be4825c64f0396` | `4bf3b12453bccda0` |
| `tools/p9_fill_convention_measure.py` | *(new)* | `c92d0e385ad51306` |
| `tools/p9_champion_regate.py` | *(new)* | `ea6df4d92cf3054b` |
| `tools/p9_fill_convention_identity.py` | *(new)* | `716afde3002ed612` |
| `tests/test_fill_convention.py` | *(new)* | `36befc0d1fad75dd` |
| `tests/test_p7_orchestrator.py` | *(unrecorded)* | `96a1ed8215f2fa7c` |
| `tests/test_p7_symbol_mode.py` | *(unrecorded)* | `3026796092dd0d7c` |
| `tests/test_ga_benchmark_mode.py` | *(unrecorded)* | `76819cd064bbeda5` |
| `docs/operations.md` | `74abeb3825388edf` | `b27e94ff33d7ea0a` |
| `docs/research/CORE_ALGORITHMS.md` | `094bc3f089c91a4d` | `ec2443964eafa391` |

`config/config.yaml` is the only shipped *behaviour* input touched, and the added
key is `close` — the value the engine already used. `data/binance_trader.db`,
`strategies/**`, `docs/overhaul/ALGO_UPGRADE_EVIDENCE.md` and
`docs/overhaul/P7_REGIME_PLAN.md` were not written. (This document is not in its
own table: every edit to it changes its own hash.)

---

## 7. Recommendation, and the README wording for the Lead

**Recommendation: leave `backtest.fill_convention: close` shipped.** The convention
is a *modelling assumption*, not a bug: `close` is the historical baseline every
recorded number was produced under. The measurement says the conventional
alternative is **worse for 10 of the 12 arm × timeframe pairs** on this cache (and
for 3/3 champions and 4/4 P8 windows), so flipping it would
(a) invalidate every previously recorded champion/fitness/DSR/P8 number and
(b) make the strategies look worse, without any compensating claim. The honest
statement is: **the engine's numbers carry a one-bar-free execution assumption, and
the alternative costs 2.7–7.1 bps per trade and 0.19–0.36 pp of two-month return
on the tested sample.** If the owner wants a latency-realistic headline, the
measurement is the input; the switch is theirs.

**README wording for the Lead** (not edited here — `README.md` is owned by the
Lead):

> **Backtest fills are latency-free by default.** The backtest engine prices every
> fill — entry and exit — at the close of the bar that produced the signal (zero
> execution latency). `backtest.fill_convention: next_open` moves the fill to the
> open of the following bar (signals unchanged); it costs **2.7–7.1 bps per trade**
> and **0.19–0.36 pp of two-month return** on the tested sample and invalidates
> every previously recorded backtest/champion number, so it ships **off**. See
> `docs/overhaul/P9_FILL_CONVENTION_EVIDENCE.md`.

---

## 8. What could not be done / limitations

1. **The hybrid engine was not extended** — it refuses `next_open` by name (and
   `auto` falls back to legacy with a warning). Extending `EventDrivenExecutor`
   with the same seam is the follow-up; it is a second execution model whose parity
   with legacy is pinned by `tests/test_engine_parity_variants.py`.
2. **The recorded P8 numbers are not reproducible on the current cache** (the
   basket moved from nine symbols to five), so §4.2 compares `close` against
   `next_open` on one cache state instead of against the document's table. The
   P8 *verdict* is unchanged either way.
3. **The champion fitness is reproduced without the genome-complexity penalty**
   (the YAML is not a chromosome). The `close` arm matches every other recorded
   quantity exactly; the deltas are unaffected because the missing terms are
   identical in both arms.
4. **SL/TP exits move by more than a pure price shift** under `next_open` (the
   level fill is replaced by the next open). That is a consequence of applying the
   shift uniformly, it is stated in §2, and the exit-reason mix is printed per arm
   so the reader can see which exits carry it.
5. **One window per timeframe.** The 1h/15m measurements use 2025-05-01→2025-07-01
   (2 months, 3 symbols). A regime where the result reverses cannot be ruled out;
   the direction held in **10 of the 12 arm × timeframe pairs** (and in 3/3
   champions over their own windows and 4/4 P8 windows), which is what makes it a
   finding rather than a coin toss.
