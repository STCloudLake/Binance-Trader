# 13 — Volume-aware liquidity: participation caps and market-impact costs (P6-A)

**Scope.** A review of how this system uses volume found one honest answer: only
as a *coarse relative* feature (volume vs its own rolling mean, i.e. a spike
detector), and never as a **capacity or cost** input. This page documents the two
P6-A additions that fix that. Neither one predicts direction; both make the
system's *execution* honest.

**The principle.** Volume's highest-value use here is not "will the price go up?"
— it is "can this order be filled at anything like the quoted price?" A fill that
is large relative to what the market recently traded pays **market impact**
(Kyle 1985; Almgren & Chriss 2000; Grinold & Kahn, *Active Portfolio
Management*, ch. 16), and an order that is a *large fraction* of recent traded
notional cannot be filled at the quoted price at all. Both are risk controls, and
both hold regardless of any forecast:

1. **Participation cap** — refuse or shrink an order whose notional exceeds
   `max_participation_pct` of the recent traded quote notional. A hard control,
   independent of prediction.
2. **Volume-dependent impact cost** — add a square-root impact term to the cost
   model the simulator and the backtest share, so a backtest of a large size is
   no longer silently optimistic about its own cost.

> **This improves realism and executability, not predictive edge.** No key here
> changes a signal, a probability, a label or a filter. With the shipped defaults
> (`enabled: false`, `impact_k: 0`) every number in the system is bit-identical
> to the pre-P6 code — pinned by `tests/test_liquidity.py`. See §6.

---

## 1. Formulas and units

One meaning per quantity, stated once (`core/risk/liquidity.py`):

| quantity | unit | meaning |
|---|---|---|
| `recent_quote_volume` | **quote currency (USDT)** | Σ `volume · price` over the last `lookback_bars` bars |
| participation | **fraction** | 0.01 = 1 % of that volume |
| `max_participation_pct` | **percent** | the config ceiling: 1.0 = at most 1 % |
| `impact_pct` | **percent of notional, per side** | 0.05 = 5 bp of that side's notional |
| `k` (`impact_k`) | dimensionless | scale of the impact law |
| `floor`, `cap` | percent | bounds on `impact_pct` |

```
quote_volume   = Σ_{i∈last N bars} volume_i · close_i           [USDT]
participation  = |notional| / quote_volume                      [fraction]
participation% = 100 · participation                            [percent]

allowed_notional = min(notional, max_participation_pct/100 · quote_volume)

impact_pct     = clip(k · participation ** exponent, floor, cap)   [% per side]
impact_usdt    = Σ_{sides ∈ {entry, exit}} notional_side · impact_pct_side / 100
```

`exponent = 0.5` is the square-root law: impact grows with size but *sub*-linearly,
so quadrupling an order roughly doubles the per-unit impact. The cost model
charges the entry and the exit side **separately**, each at its own participation
(the exit notional differs by the PnL), which is why a 50 000 → 51 000 round trip
costs slightly more than twice the entry-side figure.

**Why "percent" for `participation_pct` but "fraction" for `participation`?** The
config unit is what an operator reads (`max_participation_pct: 1.0` = 1 %), while
`impact_pct` is a *law* evaluated on the fraction. `participation_pct()` returns
`None` — never `0.0` — when the window is unknown: "we cannot measure
participation" is a different statement from "participation is zero", and `0.0`
would read as "nothing traded", the opposite of "unmeasured".

## 2. Where it is wired

| symbol | file | reader of |
|---|---|---|
| `recent_quote_volume`, `participation_pct`, `cap_notional`, `impact_pct` | `core/risk/liquidity.py` | the maths (pure, no I/O, no pandas at import) |
| `PositionSizer.apply_participation_cap` | `core/risk/position_sizer.py` | `enabled`, `max_participation_pct`, `lookback_bars`, `per_symbol` |
| `apply_trading_costs` (impact term) | `core/backtest/cost_model.py` | `impact_k`, `impact_exponent` |
| `total_costs_with_impact` (report split) | `core/backtest/cost_model.py` | reports fees / spread / **impact** separately |
| model + load + warnings | `app/config.py` (`LiquidityConfig`, `liquidity_key_warnings`, `LIQUIDITY_KEY_READERS`) | every key |

Two seams matter:

* **Sizing.** `calculate_position_size(..., recent_quote_volume=…, symbol=…)`
  applies the cap **last**, after the capital-pool split, the hard
  `max_position_size_pct` ceiling and the vol-targeting ceiling, so those stay
  authoritative; the cap can only ever *shrink*. With no `recent_quote_volume`
  (the live manager's current call) the hook is skipped entirely — there is no
  measured window to cap against, and the docs say so rather than inventing one.
* **Cost.** `apply_trading_costs(..., recent_quote_volume=…)` adds the impact
  term. `recent_quote_volume=None` (what every pre-P6 caller passes by omission)
  and `k <= 0` both short-circuit to the legacy `fees + spread/2` sum.

`PositionSizer` receives the block through the object it already gets: `Config._load`
attaches `risk_liquidity` to `config.risk_vol_targeting` (the argument
`core/risk/manager.py` passes today), and `core.risk.liquidity.resolve_liquidity_config`
resolves it (`explicit block → value.risk_liquidity → Config.load()`).

## 3. Measured examples on the cached data

Command (read-only; nothing is written, the DB is untouched):

```powershell
python -c "
import pandas as pd
from core.risk.liquidity import recent_quote_volume, participation_pct, cap_notional, impact_pct, total_impact_usdt
K=0.1
for sym, iv in (('BTCUSDT','1h'), ('XRPUSDT','1m')):
    df = pd.read_parquet(f'data/market/{sym}/{iv}.parquet')
    v = recent_quote_volume(df, 20)
    print(sym, iv, 'window ends', str(df.index[-1])[:19], f'quote_vol20={v:,.2f}')
    for n in (500.0, 50000.0):
        print('  ', f'{n:,.0f}', f'participation={participation_pct(n,v):.6f}%',
              f'impact={impact_pct(participation_pct(n,v)/100,K):.6f}%/side',
              f'roundtrip={total_impact_usdt(n,n,v,K):.4f} USDT',
              f'cap@1%={cap_notional(n,v,1.0)[0]:,.2f}')
"
```

Observed (`k = 0.1`; BTCUSDT 1h window ends 2026-09-30 05:59:59, XRPUSDT 1m window
ends 2026-09-30 06:27:59 — the running app keeps appending bars, so the volume
figures move while the *shape* of the table does not):

| symbol | tf | 20-bar quote vol (USDT) | order (USDT) | participation | impact (k=0.1) | round-trip impact | capped @1 % |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 1h | 750,661,812.73 | 500 | 0.000067 % | 0.000082 %/side | 0.0008 USDT | 500.00 (no cap) |
| BTCUSDT | 1h | 750,661,812.73 | 50 000 | 0.006661 % | 0.000816 %/side | 0.8161 USDT | 50 000.00 (no cap) |
| XRPUSDT | 1m | 2,657,574.07 | 500 | 0.018814 % | 0.001372 %/side | 0.0137 USDT | 500.00 (no cap) |
| XRPUSDT | 1m | 2,657,574.07 | 50 000 | 1.881415 % | 0.013716 %/side | 13.7165 USDT | **26 575.74 (shrunk)** |

Two facts this table states plainly:

* **A 500 USDT order is capacity-irrelevant everywhere** — 0.00007 % of BTC's 20-bar
  notional, 0.019 % of XRP's 1-minute notional. At `k = 0.1` it pays 0.0008 USDT of
  modelled impact on a round trip, i.e. nothing.
* **Size is what makes cost appear.** The same symbol at 100× the size pays 1000×
  the impact in USDT, and on a *thin* window (XRP 1m) the 50 000 USDT order is
  **shrunk to 26 575.74** — exactly 1 % of the window. On BTC 1h the identical
  order is untouched, because a 1 % ceiling there is 7.5 M USDT. The cap binds on
  the *ratio*, not on the order size, and the accepted numbers show it.

Pre/post comparison on the **same** run (100 consecutive BTC 1h pairs, cost from
the shipped config, fee 0.04 %, spread 0.01 % for BTCUSDT, `recent_quote_volume`
pinned to the same 20-bar window):

| size | `k = 0` (pre-P6) | `k = 0.1` | Δ | `k = 0.5` | Δ |
|---|---|---|---|---|---|
| 0.01 BTC | 75.6777 USDT | 75.8566 | +0.1789 (+0.24 %) | 76.5722 | +0.8945 (+1.18 %) |
| 1.0 BTC | 7 567.7708 USDT | 7 746.6673 | +178.8965 (+2.36 %) | 8 462.2533 | +894.4824 (+11.82 %) |

The `k = 0` column equals the legacy cost **bit for bit** (`==`, not `approx`), and
that is asserted twice: in the loop above (with and without the volume argument)
and in `tests/test_liquidity.py::test_impact_k_zero_is_bit_identical`.
Per-trade breakdown from `total_costs_with_impact` for a 0.6 BTC round trip at
83 384 → 85 000 (≈ 50 030 USDT) on the same window, `k = 0.1`:

| component | USDT |
|---|---|
| fees (0.04 % × both sides) | 40.4122 |
| spread (0.005 % × both sides) | 5.0515 |
| **impact** | **0.8288** |
| total | 46.2925 |

That split is the point: the impact line is *visible* next to fees and spread
instead of being folded invisibly into PnL.

## 4. Limits — read this before setting a non-zero `impact_k`

* **The coefficient is illustrative, not calibrated.** `k = 0.1` above is a
  documented example chosen to make the arithmetic legible; the shipped default is
  `0.0`. A real calibration needs trade-and-quote data (realised slippage vs
  participation) that this deployment does not have. Until then, treat any
  non-zero `k` as a *stress* assumption and publish the experiment that produced
  it.
* **No order-book depth model yet.** Participation is measured against *traded*
  volume, not resting liquidity. It cannot see a thin book, a spoofed level, or a
  spread that widens as you cross it; the impact term inherits the fixed
  `spread/2` from the existing model rather than deriving it from depth.
* **The window is a bar aggregation, not a trade tape.** `lookback_bars` of 1h
  bars smooths bursts: a 20-bar mean is not what the next minute will look like.
  Shorter intervals (1m) are more conservative, and the XRP row above shows the
  difference a window choice makes.
* **The live sizing path is not yet wired.** `RiskManager` does not pass a
  `recent_quote_volume`, so the participation cap is exercised by the sizer API
  and the tests, not by live signals; the impact term is exercised by the backtest
  cost model and by `total_costs_with_impact`. The backtest itself does not yet
  feed per-bar volume into `apply_trading_costs` (the seam is one keyword
  argument). These are stated as gaps, not implied as done.
* **Square-root is an empirical shape, not a law of nature.** It fits liquid
  futures/equities well at moderate participation; at very high participation
  impact is closer to linear. `impact_exponent` is configurable for that reason.

## 5. Switch names and shipped defaults

| switch | default | effect when off |
|---|---|---|
| `risk.liquidity.enabled` | **false** | `PositionSizer` never calls the participation hook; sizing is bit-identical |
| `risk.liquidity.max_participation_pct` | **1.0** (percent) | inert |
| `risk.liquidity.lookback_bars` | **20** | inert |
| `risk.liquidity.impact_k` | **0.0** | `apply_trading_costs` short-circuits to the legacy sum |
| `risk.liquidity.impact_exponent` | **0.5** | inert while `impact_k` is 0 |
| `risk.liquidity.per_symbol` | **{}** | no per-symbol override; `default` / `*` covers the rest |

Nothing is enabled by default. Every key is read by code — the audit's "unread
config key" defect cannot recur because `app.config.LIQUIDITY_KEY_READERS` names
the reader of each key and
`tests/test_liquidity.py::test_every_liquidity_config_key_has_a_reader` asserts
both that the table covers every model field **and** that the named file really
references the key (`grep`-based, so the table cannot rot into a comment).

## 6. Reproduce the identities

```powershell
python -m pytest tests/test_liquidity.py -q          # 24 passed
python -m pytest tests/test_liquidity.py -q -s -k "call_cost"   # measured per-call cost
```

* off-path sizing: `test_participation_disabled_is_bit_identical` — with
  `enabled: false`, `calculate_position_size(..., recent_quote_volume=X)` returns
  the same tuple for `X ∈ {None, 0.0, 123.0, 7.5e8, [1,2], lambda}` as the call
  without any volume argument.
* off-path costs: `test_impact_k_zero_is_bit_identical` — `k = 0` (and no volume)
  reproduces the legacy cost with `==` and identical `repr`.
* measured cost: `cap_notional` measured at **0.917–0.940 µs/call** across runs and
  the full 20-bar-frame path (frame → sum → participation → impact) at
  **1.887–1.936 µs/call**, asserted `< 200 µs` (`CALL_COST_BOUND_US`) so a
  complexity regression fails while a slow machine does not.
