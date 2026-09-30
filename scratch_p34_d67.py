"""D6/D7 measurement (temporary): slice-op count on real frames + clip behaviour."""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from core.ml import volatility as V  # noqa: E402
from core.backtest.data_feeder import DataFeeder  # noqa: E402

CACHE = "data/market"
SYMS = ["BTCUSDT", "ETHUSDT"]
TVFS = ["1h", "4h"]


def slice_ops(hoisted: bool, positions=20, bars=400):
    feeder = DataFeeder(CACHE, SYMS, TVFS, "2026-01-01", "2026-06-01")
    feeder.load()
    ts_list = feeder._timestamps[:bars]
    n = 0
    outer: dict = {}
    for ts in ts_list:
        if not hoisted:
            inner: dict = {}
        for p in range(positions):
            sym = SYMS[p % len(SYMS)]
            tf = TVFS[(p // len(SYMS)) % len(TVFS)]
            if hoisted:
                key = (sym, tf, ts)
                sl = outer.get(key)
                if sl is None:
                    sl = feeder.get_all_data_for_symbol(sym, tf)
                    sl = sl[sl.index <= ts]
                    outer[key] = sl
                    n += 1
            else:
                key = (sym, tf)
                sl = inner.get(key)
                if sl is None:
                    sl = feeder.get_all_data_for_symbol(sym, tf)
                    sl = sl[sl.index <= ts]
                    inner[key] = sl
                    n += 1
    return n


def main():
    print("frames:", {s: {t: len(DataFeeder(CACHE, [s], [t], '2026-01-01', '2026-06-01')._data.get('x', {}) or []) for t in TVFS} for s in SYMS[:0]})
    for positions in (5, 20, 50):
        t0 = time.perf_counter()
        n_in = slice_ops(False, positions)
        t1 = time.perf_counter()
        n_out = slice_ops(True, positions)
        t2 = time.perf_counter()
        print(f"positions={positions:3d} bars=400  inner_cache ops={n_in:4d} "
              f"({t1-t0:.3f}s)  hoisted ops={n_out:4d} ({t2-t1:.3f}s)")

    # realistic steady state: 1 position per symbol
    for positions in (1, 2):
        n_in = slice_ops(False, positions)
        n_out = slice_ops(True, positions)
        print(f"steady positions={positions} bars=400 inner={n_in} hoisted={n_out}")

    print("\n--- frame sizes / slice cost ---")
    f = DataFeeder(CACHE, SYMS, TVFS, "2026-01-01", "2026-06-01")
    f.load()
    df = f.get_all_data_for_symbol("BTCUSDT", "1h")
    print("BTC 1h rows", len(df))
    ts = f._timestamps[200]
    t0 = time.perf_counter()
    for _ in range(200):
        _ = df[df.index <= ts]
    t1 = time.perf_counter()
    print(f"boolean-mask slice on {len(df)} rows: {(t1-t0)/200*1e6:.1f} us/op")

    print("\n--- D7 clip: real BTC 1h, rolling 500-bar window, how many clips reverse ---")
    r = V.log_returns(pd.read_parquet("data/market/BTCUSDT/1h.parquet")["close"].values)
    w = 500
    flips = 0
    checked = 0
    max_jump = 0.0
    prev = None
    for end in range(w + 1, len(r) + 1):
        win = r[end - w:end]
        c = V.clip_outliers(win, sigma=6.0)
        # the observation that leaves the window at the *next* step never matters;
        # look at the oldest still-present observation, which can be un-clipped
        i = 1  # r[end-w] leaves next step; r[end-w+1] is the oldest survivor
        if prev is not None:
            checked += 1
            jump = abs(float(c[i]) - prev)
            max_jump = max(max_jump, jump)
            if jump > 1e-12:
                flips += 1
        prev = float(c[i])
    print(f"oldest-surviving observation changed while the window rolled: "
          f"{flips}/{checked} steps, max |jump| = {max_jump:.3e}")

    # find a concrete un-clipping event on real data
    lim_prev = None
    events = []
    for end in range(w + 1, len(r) + 1):
        win = r[end - w:end]
        med = float(np.median(win))
        mad = float(np.median(np.abs(win - med))) * 1.4826
        lim = 6.0 * mad
        if lim_prev is not None and lim > lim_prev * 1.05:
            events.append((end, lim_prev, lim))
        lim_prev = lim
    print("Winsor limit expanded >5% between consecutive windows:",
          len(events), "events; first:", events[:3])


if __name__ == "__main__":
    main()
