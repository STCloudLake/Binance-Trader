# P8 — Does volatility targeting improve a beta-harvesting basket?

**Status:** measured, verdict **no for the pre-stated threshold** — `risk.vol_targeting.enabled`
stays `false`. This document is the evidence; the switch is the operator's call.

**Tool:** [`tools/p8_beta_harvest_measure.py`](../../tools/p8_beta_harvest_measure.py)
(sha256_16 `a9950a450220bf2b`, printed by the run itself in §9) · **Tests:**
[`tests/test_p8_beta_harvest.py`](../../tests/test_p8_beta_harvest.py) · **Artifact:**
[`data/p8_evidence/p8_beta_harvest.json`](../../data/p8_evidence/p8_beta_harvest.json)
(the machine-readable form of every number below; `data/` is git-ignored, exactly like the
existing `data/p6b_evidence/`, so the file is on disk but not committed)

```powershell
python -m pytest tests/ -q -p no:cacheprovider
python -m compileall -q app core web db scripts tools
python tools/p8_beta_harvest_measure.py --out %TEMP%\p8_beta_harvest.json
```

One bounded, deterministic run: **454.3 s wall**, no network, no writes outside `--out`
(`config/config.yaml`, `strategies/**` and `data/binance_trader.db` are untouched and
hash-verified in-run). Two independent full runs produced **bit-identical metrics for all
24 arm × window rows**. Every cell of the two tables in §3 and §4 was re-checked
programmatically against the artifact before this document was written (240 metric cells
and 24 matched-risk rows, 0 mismatches).

---

## 1. The question, and why it is asked this way

Five searches for short-horizon **directional** alpha failed on this repository's real
cache (ML gates; pairs 0/30; meta-labelling; 52 GA generations; P7's four stages) — none
reached a positive out-of-sample deflated Sharpe. In the same windows buy-and-hold was
strongly positive. This run's arm ``S`` reproduces the operator's premise directly: the
seven shipped champions return **−4.07 % / −2.46 % / −1.59 % / −4.91 %** over four
quarters at **1.2–2.8 % annualised volatility**, i.e. they sat in cash and captured none
of the beta.

So the question is not "is there alpha?" but **"is a risk-managed beta harvest better than
a plain beta harvest?"** — deterministic, and it needs no forecasting skill, because
conditional *volatility* is predictable (ARCH/GARCH; Tsay ch. 3) even though the *sign* of
the next return is not. The machinery already exists and ships **off**:
`risk.vol_targeting` (`enabled: false`), `core/risk/position_sizer.py::PositionSizer.vol_scale`,
`core/risk/manager.py::RiskManager.forecast_vol_pct`, `core/ml/volatility.py::forecast_vol`.

## 2. What was measured

### Basket (state and justify the rule)

**ADAUSDT, BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, ZECUSDT** — *every cached symbol
whose `1h.parquet` covers the whole 2025-04-01 → 2026-10-01 cache span (within one day at
each end), minus USD-stable pairs.* The rule is data-driven, not hand-picked; the tool
prints the exclusions with reasons:

| excluded | reason |
|---|---|
| `USDCUSDT` | USD-stable pair — no beta exposure, ~0 volatility |
| `ENAUSDT`, `VTHOUSDT` | partial 1h coverage (starts 2026-09-23) |
| `MOVRUSDT` | no `1h.parquet` at all |

### Costs (the GA's own, not a re-derivation)

`backtest.cost_model`: **taker fee 0.04 %/side**, plus **half the per-symbol quoted
spread** per side — `BTCUSDT 0.01`, `ETHUSDT 0.02`, `BNBUSDT 0.03`, `SOLUSDT 0.03`,
`XRPUSDT 0.04`, default `0.03` for `ADAUSDT`/`ZECUSDT`. Every charge goes through
`core.backtest.cost_model.apply_trading_costs` (halved, since that function prices a round
trip): e.g. 1 000 USDT of BTC costs 1 000 × (0.04 + 0.01/2)/100 = **0.45 USDT per side**.

### Windows (and why these four)

| | window | bars | rebalanced basket | basket ann. vol | regime |
|---|---|---|---|---|---|
| **W0** | 2025-05-01 → 2025-08-01 | 2208 | **+31.01 %** | 53.6 % | broad bull, ETH +105.7 % |
| **W1** | 2025-10-01 → 2026-01-01 | 2208 | **−1.90 %** | 73.7 % | crash + one outlier (ZEC **+581.8 %**) |
| **W2** | 2026-02-01 → 2026-05-01 | 2136 | **−9.74 %** | 59.9 % | broad bear, all but ZEC down |
| **W3** | 2026-06-01 → 2026-09-01 | 2208 | **+12.99 %** | 57.1 % | mixed, high dispersion |

Disjoint ~92-day blocks (≈92 daily observations each — the basis
`core.ga.fitness.deflated_sharpe_ratio` consumes), all inside the cache, each with **700
hourly bars of estimator warm-up loaded before the window start** so the first decision
already sees a full estimator history. Their realised volatility spans 53.6–73.7 % and
their returns span −9.7 % to +31.0 %, so "different regimes" is measured here, not
asserted. `W0`–`W2` also precede the shipped champions' training windows (2026-03-01
onward); `W3` overlaps them — stated because it is one more reason arm `S` is a reference.

### Arms

| arm | rule |
|---|---|
| **H** | *buy-and-hold, the parameter-free baseline.* Split the account 1/7 at the window's first 1h close, buy at that close, **never rebalance** (weights drift), liquidate at the last close. |
| **Hreb** | *diagnostic.* The same basket **rebuilt to equal notional every UTC day**, overlay pinned at scale 1.0 — isolates the rebalancing rule. |
| **V** | **H's book, exposure scaled.** Once per UTC day total gross notional is set to `equity × clip(target_vol_pct / forecast_vol_pct, min_scale, max_scale)` by scaling **every leg by the same factor**, so the overlay cannot change relative weights. |
| **V<=1** | V with `max_scale = 1.0` (no leverage — a cash backtest cannot exceed 100 % gross). |
| **Vleg** | V applied **per leg** (`vol_scale` on each symbol's own forecast) — the literal production sizing rule, which *also* reweights to equal risk daily. Labelled as a different portfolio, not just a different exposure. |
| **S** | every `strategies/ga_champion_*.yaml` (read-only) through the production engine on 1h bars, composited at equal capital (1/7 each). **Reference only.** |

**Nothing is reimplemented.** The forecast is `core.ml.volatility.forecast_vol` (method
`ewma`, λ = 0.94, window 500, `interval="1h"`), fed the trailing
`RiskManager._VOL_HISTORY_BARS` = 600 bars **strictly before** the decision bar — exactly
what `RiskManager._history_frame` hands the live path. The scale is
`core.risk.position_sizer.PositionSizer.vol_scale`. The overlay is driven through an
**in-memory `VolTargetingConfig` copy** (`model_copy`/constructor, `enabled=True`);
`config/config.yaml` is never opened for writing, and its hash is compared before/after
(§9).

**Causality.** At bar *t* the decision uses the close of *t* and the estimator's view of
bars `< t`, so the exposure that earns the `t → t+1` return uses no future information.
Costs are charged on the traded notional at every rebalance and again at liquidation.

**Metric basis.** Identical to `core.ga.fitness.stats_from_trades`: daily returns from
`daily_returns(equity_curve)`, annualised by `√(len(daily) × 365 / span_days)`,
`max_drawdown_pct` reused verbatim. Sortino uses the same annualisation with a
downside-deviation denominator; Calmar = annualised (geometric) return ÷ max drawdown;
turnover is one-way traded notional per unit of mean equity per year; cost share is
`costs / (costs + net PnL)`.

### The confound that had to be removed first

`H` is a **drifting-weight** book. A daily-**rebalanced** equal-weight book is a
*different portfolio*, and on this cache the difference is enormous:

| window | H (drifting) | Hreb (rebalanced) | rebalanced index |
|---|---|---|---|
| W0 | +31.64 % | +30.94 % | +31.01 % |
| **W1** | **+54.41 %** | **−0.45 %** | **−1.90 %** |
| W2 | −10.71 % | −9.80 % | −9.74 % |
| W3 | +11.42 % | +12.80 % | +12.99 % |

`Hreb` tracks the rebalanced index to within a few tenths of a point — a good cross-check
on the simulator. In `W1` `ZECUSDT` returns **+581.8 %** while the other six lose 14.6–58.7 %;
a rebalanced book trims the winner every day and keeps none of it, a drifting book rides it.
**Half of any "H versus rebalanced-V" difference would have been the rebalancing rule, not
the overlay** — so `V` is built on `H`'s drifting book, and `Vleg` (which *does* reweight)
is reported separately and labelled. The same effect is why `Hreb`'s matched-risk figures
are uninformative in `W1`.

## 3. Full metrics table

Per arm, per window (net of the cost model above; `gross%` is mean gross notional ÷ equity,
so `>100` means levered):

| arm | window | total ret % | ann vol % | Sharpe | Sortino | max DD % | Calmar | time in mkt % | gross % | turnover ×/yr | cost share of gross % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **H** | W0 | **+31.64** | 50.77 | **2.263** | 3.716 | 22.62 | **8.744** | 100.00 | 100.05 | 7.87 | 0.39 |
| Hreb | W0 | +30.94 | 50.62 | 2.226 | 3.658 | 22.53 | 8.501 | 100.00 | 100.00 | 13.09 | 0.66 |
| **V** | W0 | +20.92 | 37.55 | 2.078 | 3.416 | 15.94 | 7.060 | 100.00 | 72.89 | 55.61 | 3.80 |
| V<=1 | W0 | +20.10 | 37.07 | 2.027 | 3.316 | 16.21 | 6.592 | 100.00 | 72.03 | 50.32 | 3.58 |
| Vleg | W0 | +24.84 | 39.37 | 2.323 | 3.880 | 16.08 | 8.779 | 100.00 | 85.87 | 75.58 | 4.34 |
| S | W0 | −4.07 | 1.35 | −12.031 | −10.879 | 4.12 | −3.690 | n/a | n/a | n/a | n/a |
| **H** | W1 | **+54.41** | 100.75 | **1.752** | 2.680 | 43.47 | **10.603** | 100.00 | 100.04 | 6.92 | 0.25 |
| Hreb | W1 | −0.45 | 64.23 | −0.428 | −0.587 | 29.48 | −0.061 | 100.00 | 100.00 | 16.63 | n/a |
| **V** | W1 | +19.88 | 54.24 | 1.041 | 1.510 | 27.60 | 3.817 | 100.00 | 55.43 | 45.38 | 3.66 |
| V<=1 | W1 | +20.25 | 54.18 | 1.064 | 1.545 | 27.60 | 3.910 | 100.00 | 55.20 | 43.73 | 3.47 |
| Vleg | W1 | −17.00 | 39.65 | −2.208 | −2.697 | 28.31 | −1.847 | 100.00 | 69.93 | 73.26 | n/a |
| S | W1 | −2.46 | 1.20 | −8.551 | −8.716 | 2.94 | −3.202 | n/a | n/a | n/a | n/a |
| **H** | W2 | −10.71 | 69.43 | −0.100 | −0.148 | 24.70 | −1.505 | 100.00 | 100.06 | 9.09 | n/a |
| Hreb | W2 | −9.80 | 69.72 | −0.037 | −0.055 | 24.75 | −1.394 | 100.00 | 100.00 | 13.76 | n/a |
| **V** | W2 | −1.77 | 43.71 | 0.195 | 0.297 | 14.16 | −0.498 | 100.00 | 73.18 | 65.43 | n/a |
| V<=1 | W2 | −0.89 | 43.00 | 0.275 | 0.425 | 13.71 | −0.264 | 100.00 | 70.64 | 51.77 | n/a |
| Vleg | W2 | −2.86 | 44.55 | 0.109 | 0.164 | 14.31 | −0.784 | 100.00 | 80.49 | 77.03 | n/a |
| S | W2 | −1.59 | 2.82 | −2.417 | −3.147 | 2.58 | −2.471 | n/a | n/a | n/a | n/a |
| **H** | W3 | +11.42 | 57.80 | 1.242 | 2.040 | 27.25 | 1.966 | 100.00 | 100.06 | 9.68 | 0.98 |
| Hreb | W3 | +12.80 | 58.05 | 1.323 | 2.207 | 26.62 | 2.303 | 100.00 | 100.00 | 14.90 | 1.36 |
| **V** | W3 | +14.16 | 47.22 | 1.598 | 3.096 | 17.29 | 4.000 | 100.00 | 86.50 | 65.57 | 5.51 |
| V<=1 | W3 | +8.61 | 39.91 | 1.322 | 2.307 | 17.29 | 2.245 | 100.00 | 77.75 | 41.51 | 5.68 |
| Vleg | W3 | +12.35 | 48.71 | 1.450 | 2.698 | 18.82 | 3.123 | 100.00 | 98.04 | 84.58 | 7.75 |
| S | W3 | −4.91 | 1.30 | −15.482 | −13.255 | 5.28 | −3.430 | n/a | n/a | n/a | n/a |

*Cost share is `n/a` whenever gross PnL ≤ 0 (the share of a loss is not meaningful);
arm `S` has no per-bar exposure series, so its time-in-market / turnover columns are `n/a`
rather than a fabricated 100 %.*

**Reading it.** `V` is a genuine risk reducer, in every window, mechanically: drawdown
falls (22.62 → 15.94, 43.47 → 27.60, 24.70 → 14.16, 27.25 → 17.29) and annualised
volatility falls (50.8 → 37.6, 100.8 → 54.2, 69.4 → 43.7, 57.8 → 47.2) — but it does that
mostly by **holding less** (mean gross exposure 55–87 %). Turnover rises 5–7× (7.9 → 55.6
per year in `W0`), so the cost share goes from 0.25–0.98 % to 3.5–5.5 % of gross PnL.

## 4. Return at matched risk

Each arm's **daily returns scaled to arm H's realised volatility**, `k = σ_H / σ_arm`,
compounded (`core.ga.fitness.daily_returns` drops the first day's intraday move, so the
tool seeds the series with the initial balance to make `k = 1` reproduce the arm's own
total return exactly). The repo's own linear `risk_matched` convention
(`core.ga.benchmark`) is reported beside it and agrees; both ignore that a scaled book
pays proportionally scaled costs, which is marginally optimistic at `k > 1`.

| arm | window | k | compounded % | linear % | H total % | retention % | DSR (trials) |
|---|---|---|---|---|---|---|---|
| H | W0 | 1.000 | +31.64 | +31.64 | +31.64 | 100.0 | 0.0000 (1) |
| Hreb | W0 | 1.003 | +31.03 | +31.03 | +31.64 | 98.1 | 0.0000 (1) |
| **V** | W0 | 1.353 | **+28.23** | +28.29 | +31.64 | **89.2** | **−0.1572 (25)** |
| V<=1 | W0 | 1.370 | +27.42 | +27.54 | +31.64 | 86.7 | −0.1599 (25) |
| Vleg | W0 | 1.291 | +32.20 | +32.05 | +31.64 | 101.8 | −0.1444 (25) |
| S | W0 | 37.608 | −79.96 | −152.96 | +31.64 | −252.7 | 0.0000 (1570) |
| H | W1 | 1.000 | +54.41 | +54.41 | +54.41 | 100.0 | 0.0000 (1) |
| Hreb | W1 | 1.503 | −5.02 | −0.68 | +54.41 | −9.2 | 0.0000 (1) |
| **V** | W1 | 1.836 | **+31.18** | +36.48 | +54.41 | **57.3** | **−0.2115 (25)** |
| V<=1 | W1 | 1.837 | +31.97 | +37.20 | +54.41 | 58.8 | −0.2103 (25) |
| Vleg | W1 | 2.505 | −42.42 | −42.58 | +54.41 | −78.0 | 0.0000 (25) |
| S | W1 | 84.996 | −89.79 | −209.34 | +54.41 | −165.0 | 0.0000 (1570) |
| H | W2 | 1.000 | −10.71 | −10.71 | −10.71 | 100.0 | 0.0000 (1) |
| Hreb | W2 | 0.996 | −9.74 | −9.76 | −10.71 | 90.9 | 0.0000 (1) |
| **V** | W2 | 1.593 | **−4.88** | −2.82 | −10.71 | *loses less* | **−0.2603 (25)** |
| V<=1 | W2 | 1.619 | −3.60 | −1.45 | −10.71 | *loses less* | −0.2561 (25) |
| Vleg | W2 | 1.563 | −6.41 | −4.47 | −10.71 | *loses less* | −0.2648 (25) |
| S | W2 | 24.719 | −36.48 | −39.30 | −10.71 | −340.6 | 0.0000 (1570) |
| H | W3 | 1.000 | +11.42 | +11.42 | +11.42 | 100.0 | 0.0000 (1) |
| Hreb | W3 | 0.996 | +12.76 | +12.75 | +11.42 | 111.8 | 0.0000 (1) |
| **V** | W3 | 1.221 | **+16.71** | +17.28 | +11.42 | **146.4** | **−0.1823 (25)** |
| V<=1 | W3 | 1.440 | +11.25 | +12.40 | +11.42 | 98.5 | −0.1968 (25) |
| Vleg | W3 | 1.183 | +14.07 | +14.61 | +11.42 | 123.3 | 0.0000 (25) |
| S | W3 | 44.442 | −90.02 | −218.04 | +11.42 | −788.4 | 0.0000 (1570) |

**The finding.** Once the exposure difference is removed, `V` is **not** a free
improvement. It keeps 89 % of `H` in `W0` and beats `H` by 46 % in `W3` — but in `W1`,
the one window with a 6.8× outlier, it keeps only **57.3 %**: the overlay de-risked the
basket precisely while the basket's return was concentrated in a single lottery ticket,
so the exposure cut removed most of the upside. Arm `S` at matched risk needs `k` of
25–85× to reach `H`'s volatility; those rows are flagged `k > 5×` in the output because
scaling an almost-all-cash ledger by 85× is far outside the cash model's range — the
honest statement about `S` is simply that its raw returns are −1.6 % to −4.9 % at
1.2–2.8 % volatility, i.e. no participation at all.

## 5. Multi-testing accounting and the deflated Sharpe

**Trials counted.** The primary arm `V` is the **one pre-registered setting** — the
documented default straight from `config/config.yaml`: `method="ewma"`, `lam=0.94`,
`window=500`, `target_vol_pct=0.45`, `min_scale=0.25`, `max_scale=2.0`. The tool *also*
sweeps a bounded sensitivity grid, so **every grid point is a trial**:

> **4 estimators** (`ewma`, `realized_cc`, `realized_parkinson`,
> `realized_garman_klass`) × **3 target vols** (0.30, 0.45, 0.60 %/bar) ×
> **2 lookbacks** (250, 500 bars) = **24 points**, × 4 windows = 96 simulations.

Hence **n_trials = 1 + 24 = 25** for every V-family DSR reported above, with
`observation_periods` = the arm's own number of daily observations (≈92 per window).
`garch11` is **excluded and the exclusion is measured**: 0.125 s per call × 600 bars ×
9 symbol-windows ≈ **11.3 min** of forecasts alone, outside a bounded-minutes budget —
and the repository already documents it as research-only.

**Arm `S`** is deflated by the champions' own published GA counts, `max(provenance
n_trials) = 1570` (the newest champion's cumulative chain; the three oldest shipped
artifacts carry no provenance at all, so this is a **lower bound**). The 6-arm menu and
the `H`/`Hreb`/`V`/`V<=1`/`Vleg` choice are themselves selections; deflating for those
too could only lower every DSR further.

**Result: no arm survives deflation.** `DSR(V)` is **negative in all four windows**
(−0.157, −0.212, −0.260, −0.182). At 92 daily observations and 25 trials the per-period
hurdle `E[max] = √(1/92)·√(2 ln 25) ≈ 0.265` exceeds the per-period Sharpe `1.6/√365 ≈
0.086`. This is the *honest* reading: even `V`'s best window is **not distinguishable
from data mining** on this sample size. (For reference, `deflated_sharpe_ratio` is
*defined* as 0.0 when `n_trials ≤ 1`, which is why the `H`/`Hreb` rows show 0.0000.)

**The grid has no stable winner** — the signature of a mined result:

| window | best by Calmar | total % | max DD % | default (ewma/0.45/500) Calmar |
|---|---|---|---|---|
| W0 | `realized_cc` / 0.60 / 500 | +34.62 | 20.74 | 7.060 |
| W1 | `realized_cc` / 0.60 / 500 | +45.06 | 27.52 | 3.817 |
| W2 | `ewma` / 0.30 / 250 | −0.16 | 9.06 | −0.498 |
| W3 | `realized_garman_klass` / 0.60 / 500 | +22.20 | 24.76 | 4.000 |

The setting that looks best changes every window, so *any* single pick from this grid is
data-mined even before the DSR is applied. Note also that the only windows where the
grid beats `H`'s Calmar are `W0`/`W1`, and there only by levering up to the `max_scale`
ceiling in a high-volatility regime — a bet, not a risk improvement.

## 6. Pre-stated threshold and its verdict

**Stated before any number was computed** (it is printed by the tool *before* the table,
and lives in `THRESHOLD` in the tool source): `V`, at the documented default, is worth
**enabling** only if **all three** hold —

* **T1** `Calmar(V) > Calmar(H)` in **every** window;
* **T2** at matched risk V keeps **≥ 80 %** of H's return in **every** window (when
  `H ≤ 0` the test is the equivalent "V must not lose more than H", because a ratio is
  direction-blind once the denominator is negative);
* **T3** `DSR(V) > 0` with the honest trial count in **≥ 2** of the 4 windows.

**Applied:**

| | W0 | W1 | W2 | W3 | overall |
|---|---|---|---|---|---|
| **T1** | ✗ (7.060 < 8.744) | ✗ (3.817 < 10.603) | ✓ (−0.498 > −1.505) | ✓ (4.000 > 1.966) | **FAIL** |
| **T2** | ✓ (89.2 %) | ✗ (57.3 %) | ✓ (V −4.88 % vs H −10.71 %) | ✓ (146.4 %) | **FAIL** |
| **T3** | ✗ | ✗ | ✗ | ✗ | **FAIL** |

> **Verdict: all three conditions fail.** `V` improves drawdown and volatility in every
> window but does **not** improve Calmar in every window, **loses 43 % of the return at
> matched risk** in the one window where the basket's return was concentrated in a single
> 6.8× outlier, and its Sharpe does not survive deflation at 25 trials. **Arm V does not
> beat arm H out-of-sample on risk-adjusted terms across the windows.** The threshold was
> fixed in advance and it says no.

## 7. Recommendation on the switch

**Leave `risk.vol_targeting.enabled: false`.** The tool never wrote the file, and the
in-run before/after hashes prove it (§9). This is stated as a recommendation only: the
operator decides, and the evidence does not support enabling.

If a future operator wants to re-examine the idea, the evidence here points at *where* it
could matter rather than at the setting: the overlay's only consistent effect is to cut
exposure, which helps in `W2` (the broad bear) and hurts in `W1` (concentrated upside). A
regime-conditional overlay — de-risk only when *breadth* is deteriorating, not when a
single symbol is volatile — is a different (and untested) hypothesis; so is targeting the
**portfolio's** realised volatility rather than a per-symbol average. Neither was tested
here, and neither should be enabled on this evidence.

## 8. Tests

`tests/test_p8_beta_harvest.py` — **23 tests**, all pure-maths on synthetic series (no
parquet, no engine, 1.85 s): the vol scale *is* `PositionSizer.vol_scale` including both
clamps and the volatility-unavailable fallback; the tracked book reproduces equal weight
and its own cost (the repo's `apply_trading_costs`, halved); the `V<=1` variant cannot
exceed 100 % gross; proportional scaling preserves relative weights while
`rebalance_to_equal` does not (the `Hreb` distinction); per-leg scaling uses each leg's own
forecast; `metrics` matches `core.ga.fitness`'s Sharpe basis and gets max drawdown, Calmar,
cost share and the zero-downside Sortino right on hand-computable curves;
`rescaled_daily_returns` compounds to the total return while the repo basis does not;
matched-risk `k` and compounding (including the `k > 5×` extrapolation flag and the
degenerate zero-volatility fallback); the basket rule excludes stables and partial
coverage on a synthetic cache; `dsr_for` falls with trials and is definitionally 0 at one
trial; and every branch of the pre-stated verdict, including the negative-`H` direction.

**Suite:** `python -m pytest tests/ -q -p no:cacheprovider` → **1528 passed / 0 failed**
(baseline 1505 + 23 new), green on two consecutive runs; `python -m compileall -q app core
web db scripts tools` → exit 0.

## 9. Hashes and reproduction

`sha256_16` recorded by the run itself (`config/config.yaml` before **and** after):

| file | sha256_16 |
|---|---|
| `config/config.yaml` | `95abf9fd7186fe25` → `95abf9fd7186fe25` (**unchanged**) |
| `tools/p8_beta_harvest_measure.py` | `a9950a450220bf2b` |
| `strategies/ga_champion_1780642613.yaml` | `3dc2baec8ba7a523` |
| `strategies/ga_champion_1780666294.yaml` | `84601c01c36e67f8` |
| `strategies/ga_champion_1780713878.yaml` | `b3d992de65d6d7fe` |
| `strategies/ga_champion_1790823879.yaml` | `bff2231874d06f89` |
| `strategies/ga_champion_1790844776.yaml` | `9639cbe42f06de21` |
| `strategies/ga_champion_1790855974.yaml` | `bf520a651a0faa84` |
| `strategies/ga_champion_1790867208.yaml` | `a0f561aff5296853` |

**No temporary config toggle was used**, so there is nothing to restore:
`config/config.yaml` is only ever *read* (for the fee/spread table and as the source of the
`VolTargetingConfig` the in-memory copy is built from). `strategies/**` is opened read-only,
`data/binance_trader.db` is never touched, and the only file the run creates is the `--out`
JSON (copied to `data/p8_evidence/p8_beta_harvest.json`). The whole run is deterministic —
no RNG, no network (`use_live_spread=False`, `backtest_live_spread_enabled=False`), no
clock-dependent branch — and completes in **454.3 s**.

## 10. What could not be done / limitations

* **`garch11` is not in the grid** (measured 0.125 s/call; ≈11.3 min of forecasts alone).
  Its exclusion is a budget decision, stated and measured, not a claim about its quality.
* **Arm `S` runs on 1h bars**, not the champions' declared `1m`/`5m`/`15m` frames: the
  engine evaluates a strategy's *shortest* declared timeframe and four shipped champions
  declare `1m`/`5m` (789 k / 158 k bars), so a native multi-month run is hours, not
  minutes. Every arm is therefore on the same 1h grid and the same cost model. This
  **degrades** the strategies relative to how they were evolved, which is one more reason
  `S` is a reference and never the yardstick (`--arm-s-timeframes native` restores them).
* **Arm `S`'s windows are not all out-of-sample for `S`**: `W0`–`W2` precede the champions'
  training windows but `W3` (2026-06 → 2026-09) overlaps them. This changes nothing for the
  verdict, which compares `V` to `H`.
* **Nine symbols, four windows, one asset class.** `W1`'s result is dominated by a single
  symbol (`ZECUSDT` +581.8 %); with one outlier driving one of four windows, the
  `W1` matched-risk gap is a real feature of this cache but not a law of markets.
* **The matched-risk rescale is an exposure rescale of the same trading rule, not a
  re-simulation**: it does not re-price the scaled book's costs, and above `k ≈ 5×` it
  leaves the cash model's range of validity. That is why `S`'s matched-risk rows are
  flagged and why the honest statement about `S` is its raw return.
* **`Vleg` mixes two changes** (exposure scaling *and* daily equal-risk reweighting) by
  construction; it is reported separately for that reason, and no threshold was applied
  to it.
* **T3 as pre-stated is a strict test.** With ≈92 daily observations the deflation hurdle
  is higher than any of these Sharpes, so T3 would have failed for the whole `V` family
  even had T1 and T2 passed. That is reported rather than relaxed; the verdict does not
  depend on it (`T1` and `T2` both fail independently).
