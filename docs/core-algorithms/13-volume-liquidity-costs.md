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

Observed (`k = 0.1`; a snapshot taken when the BTCUSDT 1h window ended
2026-09-30 05:59:59 and the XRPUSDT 1m window ended 2026-09-30 06:27:59 — the
running app keeps appending bars, so the volume figures move while the *shape* of
the table does not. Re-running the command above after the twin-bar repair reported
a BTCUSDT window of 751 885 467.06 USDT, i.e. **+0.16 %** on the figure below; the
500 USDT row then reads participation 0.000066 % and round trip 0.0008 USDT, still
the same rounded values):

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

Pre/post comparison (cost from the shipped config: fee 0.04 %, spread 0.01 % for
BTCUSDT, `recent_quote_volume` **pinned** so the two runs price the same window).

**What changed in this section (re-audit finding 2).** The numbers that used to
sit here (a 1.0 BTC Δ of `+178.8965` / `+894.4824` = `+2.36 %` / `+11.82 %`, and a
`0.8288` impact line) were **not reproducible against the window quoted one table
above**: a 1.0 BTC order is 0.011 % of a 750 M USDT window, so the code gives
`+1.7455` (+0.0021 %), not `+178.8965`. Working backwards from
`total_impact_usdt(e, e, v, 0.1) = 2·e·0.1·√(e/v)/100` at the table's
83 043.14 USDT price, `+178.8965` implies a window of **71 576.03 USDT**
(7.16 × 10⁴ — an order of magnitude *smaller* than the ≈ 7.1 × 10⁵ previously
claimed here) and `+894.4824` implies a window of **2 863.04 USDT**: the retired
pair sits in a 1:5 ratio for a 1:10 size step, i.e. it is linear in size, where the
shipped square-root law scales as size^1.5 (×31.62). The retired `0.8288` line is
the one figure that *does* belong to the window the table names — at
750,661,812.73 USDT the code returns **0.828812** — so the old figures implied at
least three mutually inconsistent windows, none of them stated. The tables below
are re-measured in one pass
against **one explicitly named window**, and the window moves whenever the running
service appends a bar — so the *shape* is the claim, the exact window is a
timestamped measurement.

Reproduce (read-only; nothing is written, the DB is untouched):

```powershell
python -c "
import pandas as pd
from app.config import Config
from core.risk.liquidity import recent_quote_volume, total_impact_usdt, participation_pct
from core.backtest.cost_model import total_costs_with_impact, apply_trading_costs
Config._instance=None; cfg=Config.load('sim')
df=pd.read_parquet('data/market/BTCUSDT/1h.parquet')
v=recent_quote_volume(df,20); price=float(df['close'].iloc[-1])
print('window',f'{v:,.2f}','ends',df.index[-1],'price',price)
for qty in (0.01,1.0,10.0,100.0):
    e=qty*price
    print(f'{qty:>7} BTC notional={e:,.2f} part={participation_pct(e,v):.6f}%',
          f'k0.1={total_impact_usdt(e,e,v,0.1):.4f}',
          f'k0.5={total_impact_usdt(e,e,v,0.5):.4f}')
print('k=0 identical:', apply_trading_costs(price,price*1.02,1.0,'BTCUSDT',cfg)
      == apply_trading_costs(price,price*1.02,1.0,'BTCUSDT',cfg,recent_quote_volume=v))
cfg.risk_liquidity.impact_k=0.1
print(total_costs_with_impact(83384.0,85000.0,0.6,'BTCUSDT',cfg,recent_quote_volume=v))
"
```

Observed (window ends **2026-09-30 06:00:00**, after the twin-bar repair of §9 R3 of
`ALGO_UPGRADE_EVIDENCE.md`; a later run is a different window, see the note above):

```text
window 751,885,467.06  ends 2026-09-30 06:00:00  price 83043.14
   0.01 BTC notional=830.43   part=0.000110% k0.1=0.0017 k0.5=0.0087
    1.0 BTC notional=83,043.14 part=0.011045% k0.1=1.7455 k0.5=8.7273
   10.0 BTC notional=830,431.40 part=0.110447% k0.1=55.1963 k0.5=275.9814
  100.0 BTC notional=8,304,314.00 part=1.104465% k0.1=1745.4596 k0.5=8727.2978
k=0 identical: True 75.48621426 75.48621426
{'fees_usdt': 40.4122, 'spread_usdt': 5.0515, 'impact_usdt': 0.8281,
 'total_usdt': 46.2918, 'legacy_usdt': 45.4637, 'impact_pct': 0.0017,
 'recent_quote_volume': 751885467.0637, 'impact_k': 0.1}
```

| size | participation | `k = 0` (pre-P6) | `k = 0.1` Δ | `k = 0.5` Δ |
|---|---|---|---|---|
| 0.01 BTC (830.43 USDT) | 0.000110 % | 0 (base = legacy) | **+0.0017** (+0.0002 %) | **+0.0087** (+0.0011 %) |
| 1.0 BTC (83 043.14 USDT) | 0.011045 % | 0 (base = legacy) | **+1.7455** (+0.0021 %) | **+8.7273** (+0.0105 %) |
| 10 BTC (830 431.40 USDT) | 0.110447 % | 0 (base = legacy) | **+55.1963** (+0.0066 %) | **+275.9814** (+0.0332 %) |
| 100 BTC (8 304 314.00 USDT) | 1.104465 % | 0 (base = legacy) | **+1 745.4596** (+0.0210 %) | **+8 727.2978** (+0.1051 %) |

The `k = 0` column equals the legacy cost **bit for bit** (`==`, not `approx`):
measured `75.48621426 == 75.48621426` for a 1.0 BTC round trip priced at
83 043.14 → 84 704.00, and asserted in
`tests/test_liquidity.py::test_impact_k_zero_is_bit_identical` plus in the command
above.

**Why the impact column is small here, and when it is not.** A 20-bar 1h window on
BTCUSDT is ≈ 7.5 × 10⁸ USDT traded notional, so a retail order (≤ 0.01 BTC) is
0.0001 % of it and a 1.0 BTC order is 0.011 % — the square root of a very small
number is a small number. The order has to be a *visible fraction* of the window
before the term matters: at 100 BTC (1.1 % of the window) the model charges
0.021 %/0.105 % of notional, i.e. the same order of magnitude as the 0.05 % round
trip the fee-plus-spread model already charged. That scaling — not a large
absolute number at retail size — is the claim, and it is exactly why the shipped
default is `k = 0` (§5).

Per-trade breakdown from `total_costs_with_impact` for a 0.6 BTC round trip at
83 384 → 85 000 (≈ 50 030 USDT entry) on the **same** 751 885 467.06 USDT window,
`k = 0.1`:

| component | USDT |
|---|---|
| fees (0.04 % × both sides) | 40.4122 |
| spread (0.005 % × both sides) | 5.0515 |
| **impact** | **0.8281** |
| total | 46.2918 |

That split is the point: the impact line is *visible* next to fees and spread
instead of being folded invisibly into PnL. (The previous `0.8288` on this line
reproduces on the window the table names — 0.828812 at 750,661,812.73 USDT, not on
an unstated ≈ 729.2 M window, which would return 0.840920. It is the retired Δ
rows, not this line, that belonged to windows of their own. On the window named
above the code returns 0.8281.)

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
* **Both sizing/as-cost paths are wired, and both are still inert by default.**
  The live path is `RiskManager.resolve_recent_quote_volume` (called at
  `core/risk/manager.py:583-587`) → `PositionSizer.calculate_position_size(
  recent_quote_volume=…, symbol=…)`, so a live signal *is* capped once
  `risk.liquidity.enabled` is true. The backtest feeds both consumers of one
  measurement: `BacktestEngine._recent_quote_volume_for` reaches
  `apply_trading_costs` (the impact term) and
  `BacktestEngine._quote_volume_provider` reaches the entry sizing call in
  `run_with_exit_evaluation` (the participation cap). The provider is passed as a
  **callable**, which the sizer evaluates only *after* it has checked the switch —
  that is what keeps the default path from slicing a frame. It is also what keeps
  a capped entry honest: `cap_notional` returns `0.0` (`no_volume`) for an unknown
  window, so an unmeasured book refuses the order instead of being sized against
  an invented denominator. At the shipped defaults (`enabled: false`,
  `impact_k: 0.0`) neither wire is evaluated and a run is bit-identical to the
  pre-P6 engine (`tests/test_volume_seams.py` pins the disabled pass-through, the
  lazy provider and the whole-run k=0 identity). These are wired, not gaps.
* **The event-driven (hybrid) engine is a separate, unwired site — reported as a
  limitation.** `core/backtest/engine_hybrid.py::run_hybrid` (line 123) drives
  `core/backtest/event_executor.py`, and that module neither passes
  `recent_quote_volume` to entry sizing (`:266`) nor routes its close through
  `apply_trading_costs`: `EventDrivenExecutor._close_position` (`:72-78`)
  computes fees + fixed half-spread inline. So in `backtest_engine_mode: hybrid`
  neither the participation cap nor the impact term can price a trade, whatever
  `impact_k` says. Legacy/full runs are covered (the bullet above). Both sites are
  outside this audit's write scope, so this is reported rather than fixed; wiring
  the sizing argument is the same one-argument change, with the same
  `_quote_volume_provider` laziness requirement, and routing the close through the
  shared cost model is a larger, separate decision.
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
python -m pytest tests/test_liquidity.py -q          # 26 passed
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
